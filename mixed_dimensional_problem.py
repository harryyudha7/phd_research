from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
import numpy as np

# ---------------------------------------------------------------------
# 1. Mesh and interface tags (similar to Γ tagging in the workshop)
# ---------------------------------------------------------------------
# Load Gmsh mesh (2D bulk Ω + 1D interface Γ)
from dolfinx.io import gmsh as gmshio
# Read the same .msh
msh, cell_markers, facet_markers = gmshio.read_from_msh("horizontal.msh", MPI.COMM_WORLD, 0, gdim=2)[0:3]


# Extract the 1D interface mesh from the 2D mesh (like igridView)
# interface = mesh.create_submesh(domain, domain.topology.dim - 1, np.arange(domain.topology.index_map(domain.topology.dim - 1).size_local, dtype=np.int32))[0]

tdim = msh.topology.dim  # 2
fdim = tdim - 1             # 1

# Find all facets tagged as Gamma (tag=2)
omega = msh
gamma_entities = facet_markers.find(2)

# Create a submesh for Γ (same as gamma in the JSDokken example)
gamma, gamma_to_omega = mesh.create_submesh(omega, fdim, gamma_entities)[0:2]

# Define interface tag ID (as defined in Gmsh)
Gamma_tag = 2  # <-- replace with your actual tag ID for the interface Γ

# ---------------------------------------------------------------------
# 2. Function spaces
# ---------------------------------------------------------------------
order = 2
V_m = fem.functionspace(omega, ("Lagrange", order))  # bulk Ω
V_f = fem.functionspace(gamma, ("Lagrange", order))   # interface Γ
V_l = fem.functionspace(gamma, ("Lagrange", order))   # multiplier space on Γ
W = ufl.MixedFunctionSpace(V_m, V_f, V_l)

# ---------------------------------------------------------------------
# 3. Trial and Test functions
# ---------------------------------------------------------------------
phi, psi, mu = ufl.TestFunctions(W)
dp_m, dp_f, dl = ufl.TrialFunctions(W)
p_m = fem.Function(V_m, name="p_m")
p_f = fem.Function(V_f, name="p_f")
lmbd = fem.Function(V_f, name="lmbd")

# ---------------------------------------------------------------------
# 4. Spatial coordinates and given data
# ---------------------------------------------------------------------
x = ufl.SpatialCoordinate(msh)
f_m = fem.Constant(msh, 0.0)
f_f = fem.Constant(msh, 0.0)

dx = ufl.Measure("dx", domain=omega)
ds = ufl.Measure("ds", domain=omega, subdomain_data=facet_markers, subdomain_id=Gamma_tag)

# ---------------------------------------------------------------------
# 5. Weak formulations
# ---------------------------------------------------------------------

# --- Bulk domain Ω ---------------------------------------------------
a_m0 = ufl.inner(ufl.grad(p_m), ufl.grad(phi)) * dx
# Coupling term with interface (approximate; corresponds to avg/trace in DUNE)
# In the Dokken example, coupling to Γ is done via ds(Gamma_tag)
# Here we use the interface integral directly on ds(Gamma_tag)
# (You can later replace lmbd with an actual trace variable or interface function)
a_m1 = -lmbd * phi * ds
a_m = a_m0 + a_m1
L_m = f_m * phi * dx

# --- Interface Γ -----------------------------------------------------
a_f0 = ufl.inner(ufl.grad(p_f), ufl.grad(psi)) * ds(Gamma_tag)
a_f1 = lmbd * psi * ds(Gamma_tag)
a_f = a_f0 + a_f1
L_f = f_f * psi * ds(Gamma_tag)

# --- Lagrange multiplier (constraint tr(p_m) = p_f) ------------------
# Dokken’s example handles this through interface coupling on ds(Gamma_tag)
a_l0 = ufl.avg(p_m) * mu * ds(Gamma_tag)
a_l1 = -p_f * mu * ds(Gamma_tag)
a_l = a_l0 + a_l1
L_l = fem.Constant(msh, 0.0) * mu * ds(Gamma_tag)

# ---------------------------------------------------------------------
# 6. Combine interface weak form
# ---------------------------------------------------------------------
a_gamma = a_f + a_l
L_gamma = L_f + L_l

# ---------------------------------------------------------------------
# Print symbolic forms (optional sanity check)
# ---------------------------------------------------------------------
print("a_m =", a_m)
print("a_gamma =", a_gamma)

F = a_m - L_m + a_gamma - L_gamma
residual = ufl.extract_blocks(F)

jac = ufl.derivative(F, p_m, dp_m) + ufl.derivative(F, p_f, dp_f) + ufl.derivative(F, lmbd, dl)
J = ufl.extract_blocks(jac)

# Compute bounding box to locate sides
coords = omega.geometry.x
x = coords[:, 0]
y = coords[:, 1]

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

# Tolerance for side detection
tol = 1e-10 * max(xmax - xmin, ymax - ymin)

# Locate dofs on each side
left_dofs   = fem.locate_dofs_geometrical(V_m, lambda x: np.isclose(x[0], xmin, atol=tol))
right_dofs  = fem.locate_dofs_geometrical(V_m, lambda x: np.isclose(x[0], xmax, atol=tol))
bottom_dofs = fem.locate_dofs_geometrical(V_m, lambda x: np.isclose(x[1], ymin, atol=tol))
top_dofs    = fem.locate_dofs_geometrical(V_m, lambda x: np.isclose(x[1], ymax, atol=tol))

# Union of all boundary dofs for which we enforce a value
all_dofs = np.unique(np.concatenate([left_dofs, right_dofs, bottom_dofs, top_dofs]))

# Build a Function-valued BC so we can assign side-dependent values
p_m_bc = fem.Function(V_m)
p_m_bc.x.array[:] = 0.0  # default 0 (bottom + right)
p_m_bc.x.array[left_dofs] = 1.0
p_m_bc.x.array[top_dofs]  = 1.0

# One BC object over the union of dofs with piecewise values
bc_pm = fem.dirichletbc(p_m_bc, all_dofs)

bcs = [bc_pm]

print("DOFs in p_m:", p_m.function_space.dofmap.index_map.size_global)
print("DOFs in p_f:", p_f.function_space.dofmap.index_map.size_global)
print("DOFs in λ:", lmbd.function_space.dofmap.index_map.size_global)
total_dofs = (
    p_m.function_space.dofmap.index_map.size_global
    + p_f.function_space.dofmap.index_map.size_global
    + lmbd.function_space.dofmap.index_map.size_global
)
print("Total DOFs:", total_dofs)
from dolfinx.fem import petsc
a = a_m + a_gamma
entity_maps = [gamma_to_omega]
# --- Assemble J once (with entity_maps) ---
A = petsc.assemble_matrix(
    fem.form(a, entity_maps=entity_maps),  # <-- pass it here
    bcs=bcs,
)
A.assemble()

# (A) Assemble J once
# from dolfinx.fem import petsc
# entity_maps = [gamma_to_omega]
# # --- Assemble J once (with entity_maps) ---
# A = petsc.assemble_matrix(
#     fem.form(J, entity_maps=entity_maps),  # <-- pass it here
#     bcs=bcs,
# )
# A.assemble()

# nlp = petsc.NonlinearProblem(
#     residual,
#     u=[p_m, p_f, lmbd],
#     # J=J,
#     bcs=bcs,
#     entity_maps=entity_maps,
#     petsc_options={
#         "snes_monitor": None,
#         "ksp_type": "preonly",
#         "pc_type": "lu",
#         "pc_factor_mat_solver_type": "mumps",
#         "mat_mumps_icntl_14": 120,
#         "ksp_error_if_not_converged": True,
#         "snes_error_if_not_converged": True,
#     },
#     petsc_options_prefix="pmix_",
# )
# max_iterations = 25
# normed_diff = 0
# tol = 1e-5

# nlp.solve()
# iterations = nlp.solver.getIterationNumber()
# print(f"Converged in {iterations} Newton iterations")