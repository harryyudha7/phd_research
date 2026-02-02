from mpi4py import MPI
import gmsh
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
from dolfinx import plot

# --- setup (same as before) ---
h_list, L2_list, L2_fem_list, rL2_list, rL2s_fem_list, H1s_list, rH1s_list = [], [], [], [], [], [], []
L2_fem_list_pf, rL2s_fem_list_pf, H1s_list_pf, rH1s_list_pf = [], [], [], []
L2_fem_list_lmbd, rL2s_fem_list_lmbd, H1s_list_lmbd, rH1s_list_lmbd = [], [], [], []

plotter = pv.Plotter(shape=(2, 4), window_size=(2000, 1000))
N_ref = 10

Lx, Ly = 1.0, 1.0
y_start, y_end = 0.0, 1.0
x_start, x_end = 0.0, 1.0

from collections import Counter
import math
import sys
from tqdm import tqdm

# MAIN loop (minimal changes)
order = 1
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

for ref in range(1, N_ref + 1):
    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("rect_with_partial_interface")

    if rank == 0:
        # Parameters
        lc = 1.0 / (2**ref)
        h = lc

        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        order = 1

        # Structured grid
        nx = max(1, int(round(Lx / h)))
        ny = max(1, int(round(Ly / h)))
        hx = Lx / nx
        hy = Ly / ny

        # Create grid points
        points = [[None for _ in range(ny + 1)] for _ in range(nx + 1)]
        for i in range(nx + 1):
            x = i * hx
            for j in range(ny + 1):
                y = j * hy
                points[i][j] = gmsh.model.occ.addPoint(x, y, 0)

        # Create two triangles per square cell
        triangle_surfaces = []
        total_cells = nx * ny
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
                    # l4 = gmsh.model.occ.addLine(p00, p11)
                    l5 = gmsh.model.occ.addLine(p00, p01)
                    l6 = gmsh.model.occ.addLine(p01, p11)
                    loop2 = gmsh.model.occ.addCurveLoop([l3, l5, l6])
                    t2 = gmsh.model.occ.addPlaneSurface([loop2])
                    triangle_surfaces.append(t2)

                    pbar.update(1)

        # push rank-0 geometry into the model
        gmsh.model.occ.synchronize()

        # Define physical groups (Omega + Boundary)
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
        # NEW: Tag Gamma as only the diagonal from (0,0) to (1,1)
        # -----------------------------
        # get all 1D entities (curves)
        all_1d = [t for (d, t) in gmsh.model.getEntities(1)]

        # tolerance based on cell size to detect "on the diagonal"
        tol = min(hx, hy) * 0.25  # somewhat generous tolerance

        diag_curves = []
        for t in all_1d:
            # bounding box: xmin, ymin, zmin, xmax, ymax, zmax
            bbox = gmsh.model.getBoundingBox(1, t)
            mx = 0.5 * (bbox[0] + bbox[3])
            my = 0.5 * (bbox[1] + bbox[4])
            # pick segments whose midpoint lies on x == y within tol
            if abs(mx - my) <= tol:
                # additionally ensure the midpoint lies within domain extents [0,1]
                if 0.0 - tol <= mx <= 1.0 + tol and 0.0 - tol <= my <= 1.0 + tol:
                    diag_curves.append(t)

        # Remove any curve that is part of the outer boundary (just in case)
        boundary_1d = set(gmsh.model.getEntitiesForPhysicalGroup(1, 3)) if boundary_curves else set()
        diag_curves = [t for t in diag_curves if t not in boundary_1d]

        if diag_curves:
            gmsh.model.addPhysicalGroup(1, diag_curves, 2)
            gmsh.model.setPhysicalName(1, 2, "Gamma")

        # finalize geometry sync
        gmsh.model.occ.synchronize()

    # --------------------------
    # Minimal sync/parallel meshing steps (called on all ranks)
    # --------------------------
    comm.Barrier()
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    if rank == 0:
        filename = f"diagonal_fracture_regular_{ref}.msh"
        gmsh.write(filename)

    gmsh.finalize()
    comm.Barrier()
