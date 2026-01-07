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
                 div, inner, sin, cos, pi, FacetNormal, ds, grad, conditional, And, ge, le)
import time

t0 = time.time()
# Define domain corners
p0 = [0.0, 0.0]
p1 = [9.0, 9.0]

# Triangular mesh with 64x64 cells
msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [p0, p1],
    [128, 128],
    cell_type=mesh.CellType.triangle
)

k = 1
Q_el = element("RT", msh.basix_cell(), k, dtype=default_real_type)
P_el = element("DG", msh.basix_cell(), k - 1, dtype=default_real_type)
V_el = mixed_element([Q_el, P_el])
V = fem.functionspace(msh, V_el)

(sigma, u) = TrialFunctions(V)
(tau, v) = TestFunctions(V)

x = SpatialCoordinate(msh)
n = FacetNormal(msh)

kappa = 1


# --- Tag boundary facets: LEFT=1, RIGHT=2, BOTTOM=3, TOP=4
LEFT, RIGHT, BOTTOM, TOP = 1, 2, 3, 4

def on_left(x):   return np.isclose(x[0], 0.0)
def on_right(x):  return np.isclose(x[0], 1.0)
def on_bottom(x): return np.isclose(x[1], 0.0)
def on_top(x):    return np.isclose(x[1], 1.0)

tdim = msh.topology.dim
msh.topology.create_connectivity(tdim-1, tdim)

facets_left   = mesh.locate_entities_boundary(msh, tdim-1, on_left)
facets_right  = mesh.locate_entities_boundary(msh, tdim-1, on_right)
facets_bottom = mesh.locate_entities_boundary(msh, tdim-1, on_bottom)
facets_top    = mesh.locate_entities_boundary(msh, tdim-1, on_top)

facet_indices = np.concatenate([facets_left, facets_right, facets_bottom, facets_top])
facet_values  = np.concatenate([
    np.full_like(facets_left,   LEFT),
    np.full_like(facets_right,  RIGHT),
    np.full_like(facets_bottom, BOTTOM),
    np.full_like(facets_top,    TOP)
])

facet_tags = mesh.meshtags(msh, tdim-1, facet_indices, facet_values)

dx = Measure("dx", domain=msh)
ds = Measure("ds", domain=msh, subdomain_data=facet_tags)

# --- Exact/Dirichlet data, piecewise per side
# x = fem.SpatialCoordinate(msh)
# u_ex = cos(3*pi*x[0]) * cos(3*pi*x[1])   # for f construction

# Side-specific traces (as you proposed)
u_D1 = cos(3*pi*0) * cos(3*pi*x[1])*0 + 10      # LEFT   (x=0)
u_D2 = cos(3*pi*1) * cos(3*pi*x[1])*0 + 1  # RIGHT  (x=1)
u_D3 = cos(3*pi*x[0]) * cos(3*pi*0)*0 + 1     # BOTTOM (y=0)
u_D4 = cos(3*pi*x[0]) * cos(3*pi*1)*0 + 10    # TOP    (y=1)

# u_D1 = cos(3*pi*x[0]) * cos(3*pi*x[1])      # LEFT   (x=0)
# u_D2 = cos(3*pi*x[0]) * cos(3*pi*x[1])      # RIGHT  (x=1)
# u_D3 = cos(3*pi*x[0]) * cos(3*pi*x[1])      # BOTTOM (y=0)
# u_D4 = cos(3*pi*x[0]) * cos(3*pi*x[1])      # TOP    (y=1)

k_bg = 1
k_hi = 1000
w = 0.1 #fracture width
# tips of the diagonal fracture
xa, ya = (3,6)    # tuple or list (x_a, y_a)
xb, yb = (6,2)    # tuple or list (x_b, y_b)

# direction vector of fracture
ddx = xb - xa
ddy = yb - ya
len2 = ddx*ddx + ddy*ddy

# vector from tip1 to point x
vx = x[0] - xa
vy = x[1] - ya

# projection of (vx,vy) onto fracture direction, normalized to [0,1]
t = (vx*ddx + vy*ddy) / len2

# coordinates of closest point on the segment
px = xa + t*ddx
py = ya + t*ddy

# squared distance from x to segment
dist2 = (x[0] - px)**2 + (x[1] - py)**2

# inside if projection is within [0,1] and distance < (w/2)
in_seg = And(ge(t, 0.0), le(t, 1.0))
in_band = le(dist2, (0.5*w)**2)

# indicator: 1 inside the strip, 0 outside
# in_x = And(ge(x[0], x1_min), le(x[0], x1_max))
# in_y = le(abs(x[1] - y0), 0.5*w)    # use Python abs
# mask = conditional(And(in_x, in_y), 1.0, 0.0)

mask = conditional(And(in_seg, in_band), 1.0, 0.0)

kappa     = k_bg + (k_hi - k_bg) * mask
kappa_inv = 1.0 / kappa

# Permeability (unit here). If you have kappa, use: f = -div(kappa*grad(u_ex))
# f = -div(kappa*grad(u_ex))
f = 0

# Mixed variational form (κ = I here). Replace inner(sigma,tau) by inner(kappa_inv*sigma,tau) if needed.
a = inner(kappa_inv*sigma, tau) * dx - inner(u, div(tau)) * dx + inner(div(sigma), v) * dx

# RHS with piecewise boundary integrals:
L = inner(f, v) * dx \
    - u_D1 * inner(tau, n) * ds(LEFT) \
    - u_D2 * inner(tau, n) * ds(RIGHT) \
    - u_D3 * inner(tau, n) * ds(BOTTOM) \
    - u_D4 * inner(tau, n) * ds(TOP)

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