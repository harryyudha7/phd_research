from mpi4py import MPI
import gmsh
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
from dolfinx import plot

# 1) Load MRST data (Pa) with header x,y,pressure
# data = np.loadtxt("pressure_field_lagrange_example1.csv", delimiter=",", skiprows=1)
# x_mrst, y_mrst, p_mrst = data[:, 0], data[:, 1], data[:, 2]

# --- before the refinement loop (rank 0 only collects) ---
h_list, L2_list, L2_fem_list, rL2_list, rL2s_fem_list, H1s_list, rH1s_list = [], [], [], [], [], [], []
L2_fem_list_pf, rL2s_fem_list_pf, H1s_list_pf, rH1s_list_pf = [], [], [], []
L2_fem_list_lmbd, rL2s_fem_list_lmbd, H1s_list_lmbd, rH1s_list_lmbd = [], [], [], []

plotter = pv.Plotter(shape=(2, 4), window_size=(2000, 1000))
N_ref = 10

Lx, Ly = 1.0, 1.0
y_start, y_end = 0.0, 1.0
x_start, x_end = 0.0, 1.0
# y_start, y_end = None, None
# x_start, x_end = None, None

from collections import Counter
import math
import sys

# MAIN loop (as in your original code)
order = 1
# MESH GENERATION
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# MAIN loop (minimal changes)
for ref in range(8,N_ref+1):
    # Initialize gmsh (use sys.argv so gmsh can see MPI args if needed)
    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("rect_with_partial_interface")

    # Geometry creation: only rank 0 performs the occ.addPoint / addLine / addPlaneSurface / fragment / physical group steps
    if rank == 0:
        # --------------------------
        # Parameters
        # --------------------------
        lc = 1.0 / (2**ref)          # requested mesh size
        h = lc

        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.SaveAll", 1)   # <-- add this before meshing
        order = 1          # element order

        # --------------------------
        # 1. Structured grid parameters
        # --------------------------
        nx = max(1, int(round(Lx / h)))
        ny = max(1, int(round(Ly / h)))
        hx = Lx / nx
        hy = Ly / ny

        # --------------------------
        # 2. Create grid points
        # --------------------------
        points = [[None for _ in range(ny + 1)] for _ in range(nx + 1)]
        for i in range(nx + 1):
            x = i * hx
            for j in range(ny + 1):
                y = j * hy
                points[i][j] = gmsh.model.occ.addPoint(x, y, 0)

        # --------------------------
        # 3. Create two triangles per square cell
        # --------------------------
        triangle_surfaces = []
        total_cells = nx * ny
        from tqdm import tqdm
        with tqdm(total=total_cells, desc=f"ref={ref} | building cells", ncols=100) as pbar:
            for i in range(nx):
                for j in range(ny):
                    p00 = points[i][j]
                    p10 = points[i+1][j]
                    p11 = points[i+1][j+1]
                    p01 = points[i][j+1]

                    # triangle 1
                    l1 = gmsh.model.occ.addLine(p00, p10)
                    l2 = gmsh.model.occ.addLine(p10, p11)
                    l3 = gmsh.model.occ.addLine(p11, p00)
                    loop1 = gmsh.model.occ.addCurveLoop([l1, l2, l3])
                    t1 = gmsh.model.occ.addPlaneSurface([loop1])
                    triangle_surfaces.append(t1)

                    # triangle 2
                    l4 = gmsh.model.occ.addLine(p00, p11)
                    l5 = gmsh.model.occ.addLine(p11, p01)
                    l6 = gmsh.model.occ.addLine(p01, p00)
                    loop2 = gmsh.model.occ.addCurveLoop([l4, l5, l6])
                    t2 = gmsh.model.occ.addPlaneSurface([loop2])
                    triangle_surfaces.append(t2)

                    pbar.update(1)

        # --------------------------
        # 4. Optional internal fracture line (same as before)
        # --------------------------
        if (x_start is not None) and (x_end is not None) and (y_start is not None) and (y_end is not None):
            p5 = gmsh.model.occ.addPoint(x_start, y_start, 0)
            p6 = gmsh.model.occ.addPoint(x_end, y_end, 0)
            l_fract = gmsh.model.occ.addLine(p5, p6)
            surf_tags = [(2, s) for s in triangle_surfaces]
            gmsh.model.occ.fragment(surf_tags, [(1, l_fract)])
        else:
            l_fract = None

        # push rank-0 geometry into the model
        gmsh.model.occ.synchronize()

        # --------------------------
        # 5. Define physical groups (rank 0 only)
        # --------------------------
        omega = [t for (d, t) in gmsh.model.getEntities(2)]
        if omega:
            gmsh.model.addPhysicalGroup(2, omega, 1)
            gmsh.model.setPhysicalName(2, 1, "Omega")

        all_edges = []
        for s in omega:
            b = gmsh.model.getBoundary([(2, s)], oriented=False, recursive=False)
            all_edges.extend([t for (d, t) in b])
        counts = Counter(all_edges)
        boundary_curves = [t for t, c in counts.items() if c == 1]
        if boundary_curves:
            pgB = gmsh.model.addPhysicalGroup(1, boundary_curves, 3)
            gmsh.model.setPhysicalName(1, pgB, "Boundary")

        all_1d = [t for (d, t) in gmsh.model.getEntities(1)]
        boundary_1d = set(gmsh.model.getEntitiesForPhysicalGroup(1, 3)) if boundary_curves else set()
        gamma_curves = [t for t in all_1d if t not in boundary_1d]
        if gamma_curves:
            gmsh.model.addPhysicalGroup(1, gamma_curves, 2)
            gmsh.model.setPhysicalName(1, 2, "Gamma")

        # ensure all geometry and physical groups are in the model on rank 0
        gmsh.model.occ.synchronize()

    # --------------------------
    # 6. Minimal sync/parallel meshing steps (called on all ranks)
    # --------------------------
    # make sure rank 0 finished creating geometry before other ranks proceed
    comm.Barrier()

    # synchronize model on every rank so meshing sees the geometry & physical groups
    # safe to call on all ranks even if only rank 0 created the geometry
    gmsh.model.occ.synchronize()

    # Generate the mesh on ALL ranks (this will be parallel only if your gmsh is MPI-enabled)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # write mesh only from rank 0 (simple & safe)
    if rank == 0:
        filename = f"mpi_diagonal_fracture_regular_{ref}.msh"
        gmsh.write(filename)

    # finalize and synchronize ranks before next ref
    gmsh.finalize()
    comm.Barrier()
# --- end loop ---
