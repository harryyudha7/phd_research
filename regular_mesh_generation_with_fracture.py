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
N_ref = 8

Lx, Ly = 1.0, 1.0
y_start, y_end = 0.25, 0.75
x_start, x_end = 0.25, 0.75
# y_start, y_end = None, None
# x_start, x_end = None, None

from collections import Counter
import math
import sys
from tqdm import tqdm

# MAIN loop (as in your original code)
order = 1
# MESH GENERATION
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# MAIN loop (minimal changes)
for ref in range(2, N_ref + 1):
    # Initialize gmsh
    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("rect_with_partial_interface")

    # Geometry creation: only rank 0
    if rank == 0:
        # --------------------------
        # Parameters
        # --------------------------
        lc = 1.0 / (2**ref)
        h = lc * np.sqrt(2)

        # gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        # gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        order = 1

        # --------------------------
        # 1. Structured grid parameters
        # --------------------------
        nx = max(1, int(round(Lx / lc)))
        ny = max(1, int(round(Ly / lc)))
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
        with tqdm(total=total_cells, desc=f"ref={ref} | building cells", ncols=100) as pbar:
            for i in range(nx):
                for j in range(ny):
                    p00 = points[i][j]
                    p10 = points[i + 1][j]
                    p11 = points[i + 1][j + 1]
                    p01 = points[i][j + 1]

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

        # push rank-0 geometry into the model
        gmsh.model.occ.synchronize()

        # --------------------------
        # 5. Define physical groups
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

        # -----------------------------
        # Fracture mesh section: tag Gamma as the diagonal x = y
        # -----------------------------
        all_1d = [t for (d, t) in gmsh.model.getEntities(1)]

        tol = min(hx, hy) * 0.25
        # diag_curves = []

        # for t in all_1d:
        #     bbox = gmsh.model.getBoundingBox(1, t)
        #     mx = 0.5 * (bbox[0] + bbox[3])
        #     my = 0.5 * (bbox[1] + bbox[4])

        #     if abs(mx - my) <= tol:
        #         if 0.0 - tol <= mx <= 1.0 + tol and 0.0 - tol <= my <= 1.0 + tol:
        #             diag_curves.append(t)

        # boundary_1d = set(gmsh.model.getEntitiesForPhysicalGroup(1, 3)) if boundary_curves else set()
        # diag_curves = [t for t in diag_curves if t not in boundary_1d]

        diag_curves = []
        tol = 1e-10

        for t in all_1d:
            bnd = gmsh.model.getBoundary([(1, t)], oriented=False)
            p_tags = [pt for dim, pt in bnd if dim == 0]

            if len(p_tags) != 2:
                continue

            x1, y1, z1 = gmsh.model.getValue(0, p_tags[0], [])
            x2, y2, z2 = gmsh.model.getValue(0, p_tags[1], [])

            # both endpoints must lie on x = y
            # if abs(x1 - y1) < tol and abs(x2 - y2) < tol:
            #     # exclude degenerate zero-length just in case
            #     if abs(x1 - x2) > tol or abs(y1 - y2) > tol:
            #         diag_curves.append(t)
            s_min = x_start
            s_max = x_end

            if (
                abs(x1 - y1) < tol and abs(x2 - y2) < tol
                and s_min - tol <= x1 <= s_max + tol
                and s_min - tol <= x2 <= s_max + tol
            ):
                diag_curves.append(t)
        
        if diag_curves:
            gmsh.model.addPhysicalGroup(1, diag_curves, 2)
            gmsh.model.setPhysicalName(1, 2, "Gamma")

        # ensure all geometry and physical groups are in the model on rank 0
        gmsh.model.occ.synchronize()

        # --------------------------
        # Force transfinite mesh
        # --------------------------
        for (dim, tag) in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(tag, 2)

        for s in triangle_surfaces:
            gmsh.model.mesh.setTransfiniteSurface(s)

    # --------------------------
    # 6. Minimal sync/parallel meshing steps
    # --------------------------
    comm.Barrier()
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    if rank == 0:
        filename = f"regular_mesh_with_fracture_interior_{ref}.msh"
        gmsh.write(filename)

    gmsh.finalize()
    comm.Barrier()