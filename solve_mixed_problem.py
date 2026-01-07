try:
    from petsc4py import PETSc
    import dolfinx
    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
except ModuleNotFoundError:
    print("This demo requires petsc4py.")
    exit(0)

from mpi4py import MPI
import numpy as np

from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import (Measure, SpatialCoordinate, TestFunctions, TrialFunctions,
                 div, inner, sin, pi, FacetNormal, ds)
import time

rank = MPI.COMM_WORLD.Get_rank()
# Start timer
t0 = time.time()

msh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.quadrilateral)

k = 1
Q_el = element("BDMCF", msh.basix_cell(), k, dtype=default_real_type)
P_el = element("DG", msh.basix_cell(), k - 1, dtype=default_real_type)
V_el = mixed_element([Q_el, P_el])
V = fem.functionspace(msh, V_el)

(sigma, u) = TrialFunctions(V)
(tau, v) = TestFunctions(V)

x = SpatialCoordinate(msh)
n = FacetNormal(msh)

# Manufactured solution and RHS: u_ex = sin(3πx) sin(3πy), f = -Δu_ex
u_ex = sin(3*pi*x[0]) * sin(3*pi*x[1])
f = 18.0 * pi * pi * sin(3*pi*x[0]) * sin(3*pi*x[1])

dx = Measure("dx", msh)

# Mixed variational form with Dirichlet data imposed weakly:
# (sigma, tau) - (u, div tau) + (div sigma, v) = (f, v) - <u_D, tau·n>_∂Ω
a = inner(sigma, tau) * dx - inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
L = inner(f, v) * dx - inner(u_ex, inner(tau, n)) * ds

# No essential (strong) BCs; Dirichlet handled via boundary term above
bcs = []

problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "superlu_dist",
    },
)

try:
    w_h = problem.solve()
except PETSc.Error as e:  # type: ignore
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e

sigma_h, u_h = w_h.split()

with io.XDMFFile(msh.comm, "out_mixed_poisson_py/u.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u_h)

# End timer
t1 = time.time()

# Print only from rank 0
# if rank == 0:
print(f"Total computation time: {t1 - t0:.4f} seconds")