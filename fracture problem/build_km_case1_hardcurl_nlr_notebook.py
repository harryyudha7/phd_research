"""Build the KM Case-1 hard-curl/NLR notebook from the validated source cells."""

from __future__ import annotations

import json
import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "LCG_nonconforming_PINN_flux_reconstruction_KoppelMartin_case1_2d_neumann_tips.ipynb"
NLR_SOURCE = HERE / "LCG_Deng_flux_reconstruction_KoppelMartin_case1_rect_nonconforming_no_fenicsx_ii.ipynb"
TARGET = HERE / "LCG_KoppelMartin_case1_neumann_tips_hardcurl_PINN_NLR.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


def extract_functions(source: str, names: list[str]) -> str:
    """Copy selected top-level functions verbatim from a source notebook cell."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found[node.name] = "".join(lines[node.lineno - 1:node.end_lineno])
    missing = [name for name in names if name not in found]
    if missing:
        raise RuntimeError(f"Could not port source transport functions: {missing}")
    return "\n\n".join(found[name] for name in names)


source_nb = json.loads(SOURCE.read_text())
nlr_nb = json.loads(NLR_SOURCE.read_text())

setup = "".join(source_nb["cells"][1]["source"])
setup = setup.replace(
    "REF = 6  # n_bulk = 128, matching the MRST/EDFM export grid",
    "REF = 5  # n_bulk = 64; transport uses the 65x65 vertex-centred dual grid",
)
setup = setup.replace("torch.manual_seed(1234)", "torch.manual_seed(1729)")
setup = setup.replace("np.random.seed(1234)", "np.random.seed(1729)")
setup += """

import json
import sys
if str(pathlib.Path.cwd()) not in sys.path:
    sys.path.insert(0, str(pathlib.Path.cwd()))
import fracture_hardcurl_common as hc

SEED = 1729
OUTDIR = pathlib.Path("result_km_case1_hardcurl_nlr")
OUTDIR.mkdir(exist_ok=True)
torch.set_default_dtype(torch.float64)
print("output directory:", OUTDIR.resolve())
"""

adapter_and_gates = r'''
# ============================================================
# Option-A common adapter and mandatory construction gates
# ============================================================

def q_cg_cell_local_numpy(points, cell_ids):
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    cids = np.asarray(cell_ids, dtype=np.int32).reshape(-1)
    out = np.zeros((len(pts), 2), dtype=float)
    for cid in np.unique(cids):
        idx = np.flatnonzero(cids == cid)
        bulk_cell = sol["bulk_data"]["cells"][int(cid)]
        uvals = p_m.x.array[bulk_cell["dofs"]]
        _, grads = q1_values_grads_on_cell(bulk_cell, pts[idx])
        out[idx] = -K_M_VALUE * np.einsum("qad,a->qd", grads, uvals)
    return out


def integrate_hosted_segments(p0, p1, normals, host_cell, order=32, evaluator=None):
    evaluator = q_cg_cell_local_numpy if evaluator is None else evaluator
    xg, wg = hc._gauss01(order)
    d = p1 - p0
    ell = np.linalg.norm(d, axis=1)
    points = (p0[:, None, :] + xg[None, :, None] * d[:, None, :]).reshape(-1, 2)
    cells = np.repeat(np.asarray(host_cell, dtype=np.int32), len(xg))
    values = evaluator(points, cells).reshape(len(p0), len(xg), 2)
    return ell * np.einsum("fqk,fk,q->f", values, normals, wg)


def q_p_f_custom_numpy(points):
    return np.zeros((len(np.asarray(points).reshape(-1, 2)), 2), dtype=float)


def q_cg_common(ctx, gdata, points):
    return q_cg_numpy(points)


def lambda_h_common(ctx, gdata, s):
    return lambda_h_on_s(s)


def f_m_exact_points(points):
    return np.zeros(len(np.asarray(points).reshape(-1, 2)), dtype=float)


def build_km_dual_problem(face_order=32, line_order=12):
    coords = np.asarray(rect["vertices_xy"], dtype=float)
    primal = []
    for cid, cell in enumerate(rect["cells"]):
        # rect local order is [BL, BR, TL, TR]; convert to CCW.
        primal.append((np.asarray(cell["vids"])[[0, 1, 3, 2]], int(cid)))
    coords_check, cv_polys, raw_faces = hc._build_raw_dual_from_subtriangles(coords, primal)
    if not np.allclose(coords_check, coords, rtol=0.0, atol=1.0e-14):
        raise RuntimeError("dual adapter changed the primal vertex ordering")

    pieces, crossings = hc._split_faces_at_fracture(
        raw_faces, FRAC_A, tau_np, normal_np, L_gamma
    )
    p0 = np.vstack([face[0] for face in pieces])
    p1 = np.vstack([face[1] for face in pieces])
    normals = np.vstack([face[2] for face in pieces])
    owner = np.asarray([face[3] for face in pieces], dtype=np.int64)
    neighbor = np.asarray([face[4] for face in pieces], dtype=np.int64)
    host_cell = np.asarray([face[5] for face in pieces], dtype=np.int64)
    side = np.asarray([face[6] for face in pieces], dtype=np.int8)

    lambda_s_nodes = hc._unique_sorted(np.r_[0.0, multiplier_s_nodes(), L_gamma])
    lambda_mid = 0.5 * (lambda_s_nodes[:-1] + lambda_s_nodes[1:])
    lambda_density = np.asarray(lambda_h_on_s(lambda_mid), dtype=float)
    lambda_nodal = np.asarray(lambda_h_on_s(lambda_s_nodes), dtype=float)
    lambda_seg_a = FRAC_A[None, :] + lambda_s_nodes[:-1, None] * tau_np[None, :]
    lambda_seg_b = FRAC_A[None, :] + lambda_s_nodes[1:, None] * tau_np[None, :]

    breaks = hc._unique_sorted(np.r_[lambda_s_nodes, crossings, 0.0, L_gamma])
    density_fn = lambda s: np.asarray(lambda_h_on_s(s), dtype=float)
    src_points, src_weights, _ = hc._common_line_quadrature(
        breaks, FRAC_A, tau_np, density_fn, order=max(4, line_order // 2)
    )
    qpl_face = hc._line_source_face_flux(p0, p1, normals, src_points, src_weights)
    cv_lambda, split_cut, split_touch = hc._distribute_line_quadrature_to_cvs(
        cv_polys, src_points, src_weights, normal_np
    )
    cv_source = np.zeros(len(coords), dtype=float)
    boundary = (
        np.isclose(coords[:, 0], 0.0, atol=2.0e-12)
        | np.isclose(coords[:, 0], 1.0, atol=2.0e-12)
        | np.isclose(coords[:, 1], 0.0, atol=2.0e-12)
        | np.isclose(coords[:, 1], 1.0, atol=2.0e-12)
    )
    cv_class = np.full(len(coords), "interior-away", dtype=object)
    cv_class[split_cut] = "fracture-cut"
    cv_class[split_touch & ~split_cut] = "fracture-adjacent"
    cv_class[boundary] = "boundary"

    cg_face = integrate_hosted_segments(p0, p1, normals, host_cell, order=face_order)
    gdata = {
        "FRAC_A": FRAC_A.copy(), "FRAC_B": FRAC_B.copy(),
        "tau_np": tau_np.copy(), "normal_np": normal_np.copy(),
        "L_gamma": float(L_gamma), "h_est": float(h_est),
        "omega_geometry": coords.copy(), "cell_polys": cell_polys,
        "local_cell_vertices": local_cell_vertices,
    }
    common_ns = {
        "K_M_VALUE": K_M_VALUE, "K_F_VALUE": K_F_VALUE,
        "q_cg_numpy": q_cg_common, "lambda_h_on_s": lambda_h_common,
        "q_p_f_custom_numpy": q_p_f_custom_numpy,
        "f_m_exact_points": f_m_exact_points,
        "exact_q": lambda points: q_cg_numpy(points),
        "has_exact_lambda": False,
        "order": ORDER, "lambda_order": 0,
    }
    return hc.DualProblem(
        variant="km_case1_neumann_tips", ctx={}, gdata=gdata,
        coords=coords, p0=p0, p1=p1, normals=normals,
        owner=owner, neighbor=neighbor, host_cell=host_cell, side=side,
        cv_polys=cv_polys, cv_source=cv_source,
        cv_lambda_exact=cv_lambda.copy(), cv_lambda_h=cv_lambda,
        cv_class=cv_class, lambda_s_nodes=lambda_s_nodes,
        lambda_h_density=lambda_density,
        lambda_h_nodal_density=lambda_nodal,
        lambda_seg_a=lambda_seg_a, lambda_seg_b=lambda_seg_b,
        qpf_face=np.zeros(len(p0)), qpl_exact_face=qpl_face.copy(),
        qpl_h_face=qpl_face, cg_face=cg_face, exact_face=cg_face.copy(),
        curl_sign=hc._curl_endpoint_sign(p0, p1, normals),
        source_quad_exact_points=src_points,
        source_quad_exact_weights=src_weights,
        source_quad_h_points=src_points, source_quad_h_weights=src_weights,
        polynomial_order=ORDER, multiplier_order=0, globals_ns=common_ns,
    )


def gate_mrst_dual_decomposition():
    """64x64 vertex duals must be exact unions of 128x128 MRST cells."""
    h_cg = 1.0 / 64.0
    h_mrst = 1.0 / 128.0
    assert abs(h_cg - 2.0 * h_mrst) <= 1.0e-15
    lines = (np.arange(64, dtype=float) + 0.5) * h_cg
    mrst_lines = (2 * np.arange(64) + 1) * h_mrst
    error = float(np.max(np.abs(lines - mrst_lines)))
    if error > 5.0e-15:
        raise RuntimeError(f"MRST/dual decomposition gate failed: {error:.3e}")
    counts = {"interior": [2, 2], "edge": [1, 2], "corner": [1, 1]}
    print(f"MRST-to-dual geometric gate: PASS, max line error={error:.3e}, unions={counts}")
    return {"passed": True, "max_line_error": error, "union_shapes": counts}


def gate_sealed_tip_exchange(tol=1.0e-10):
    mids = 0.5 * (multiplier_s_nodes()[:-1] + multiplier_s_nodes()[1:])
    widths = np.diff(multiplier_s_nodes())
    integral = float(np.dot(widths, lambda_h_on_s(mids)))
    scale = max(1.0, float(np.dot(widths, np.abs(lambda_h_on_s(mids)))))
    passed = abs(integral) <= tol * scale
    print(
        f"sealed-tip exchange gate: {'PASS' if passed else 'FAIL'}, "
        f"integral_Gamma(lambda_h)={integral:.6e}, tol={tol * scale:.3e}"
    )
    if not passed:
        raise RuntimeError("sealed-tip net exchange does not balance")
    return {"passed": passed, "integral": integral, "tolerance": tol * scale}


def _max_box_offset(point, direction):
    limits = []
    for coordinate, velocity in zip(point, direction):
        if velocity > 0.0:
            limits.append((1.0 - coordinate) / velocity)
        elif velocity < 0.0:
            limits.append((0.0 - coordinate) / velocity)
    return min(value for value in limits if value >= 0.0)


def gate_a0_km_boundary_tips(problem, tol=2.0e-6):
    panel_mid = 0.5 * (
        problem.lambda_s_nodes[:-1] + problem.lambda_s_nodes[1:]
    )
    samples = panel_mid[[0, len(panel_mid) // 2, -1]]
    rows = []
    for s in samples:
        x = FRAC_A + s * tau_np
        eps_max = min(_max_box_offset(x, normal_np), _max_box_offset(x, -normal_np))
        eps_values = np.minimum(
            np.asarray([1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8]),
            0.2 * eps_max,
        )
        eps_values = np.unique(eps_values[eps_values > 0.0])[::-1]
        if len(eps_values) == 0:
            raise RuntimeError("no interior mirrored pair available near a boundary tip")
        density = float(lambda_h_on_s(np.array([s]))[0])
        sequence = []
        for eps in eps_values:
            xp = x + eps * normal_np
            xm = x - eps * normal_np
            if np.any(xp < -1.0e-14) or np.any(xp > 1.0 + 1.0e-14):
                raise RuntimeError("A0 plus point left Omega")
            if np.any(xm < -1.0e-14) or np.any(xm > 1.0 + 1.0e-14):
                raise RuntimeError("A0 minus point left Omega")
            qp = hc.q_p_lambda_p0_numpy(
                xp[None, :], problem.lambda_seg_a, problem.lambda_seg_b,
                problem.lambda_h_density,
            )[0]
            qm = hc.q_p_lambda_p0_numpy(
                xm[None, :], problem.lambda_seg_a, problem.lambda_seg_b,
                problem.lambda_h_density,
            )[0]
            jump = float(np.dot(qp - qm, normal_np))
            rel = abs(jump - density) / max(1.0, abs(density))
            sequence.append({"eps": float(eps), "jump": jump, "relative_error": rel})
        rows.append({"s": float(s), "density": density, "sequence": sequence,
                     "final_relative_error": sequence[-1]["relative_error"]})
    worst = max(row["final_relative_error"] for row in rows)
    passed = worst <= tol
    print(f"Gate A0-KM boundary tips: {'PASS' if passed else 'FAIL'}, worst rel={worst:.3e}")
    for row in rows:
        print(" ", row)
    if not passed:
        raise RuntimeError("Gate A0-KM failed")
    return {"passed": passed, "tolerance": tol, "rows": rows}


mrst_dual_gate = gate_mrst_dual_decomposition()
sealed_tip_gate = gate_sealed_tip_exchange()
problem = build_km_dual_problem()
print("dual CV classes:", {
    name: int(np.count_nonzero(problem.cv_class == name))
    for name in hc.class_names() if name != "source"
})
a0_generic = hc.gate_a0()
a0_km = gate_a0_km_boundary_tips(problem)
a1 = hc.gate_a1(problem)
KM_STAGE1_GATES_COMPLETE = True
print("Stage 1 gates complete.")
'''

training = r'''
# ============================================================
# Canonical Option-A training with exact horizontal streamline traces
# ============================================================

accepted_lbfgs = []
STAGNATION_WINDOW = 200
STAGNATION_RTOL = 1.0e-6
CHECKPOINT_PATH = OUTDIR / "km_case1_option_a_streamline_bc_v2.pt"


def km_zero_constant_boundary_trace_numpy(x_coordinate, y_boundary):
    source = np.asarray(problem.source_quad_h_points, dtype=float)
    weight = np.asarray(problem.source_quad_h_weights, dtype=float)
    distance = float(y_boundary) - source[:, 1]
    angle = (
        np.arctan((np.asarray(x_coordinate)[:, None] - source[None, :, 0])
                  / distance[None, :])
        - np.arctan((-source[:, 0] / distance))[None, :]
    )
    return angle @ weight / (2.0 * np.pi)


# Horizontal streamline traces are determined up to one constant each. Their
# difference controls total horizontal throughput. Choose the top-minus-bottom
# constant to match the CG total right-boundary flux, without prescribing its
# pointwise distribution.
_g0_p0 = km_zero_constant_boundary_trace_numpy(problem.p0[:, 0], 0.0)
_g1_p0 = km_zero_constant_boundary_trace_numpy(problem.p0[:, 0], 1.0)
_g0_p1 = km_zero_constant_boundary_trace_numpy(problem.p1[:, 0], 0.0)
_g1_p1 = km_zero_constant_boundary_trace_numpy(problem.p1[:, 0], 1.0)
_psi_fixed_p0 = (1.0 - problem.p0[:, 1]) * _g0_p0 + problem.p0[:, 1] * _g1_p0
_psi_fixed_p1 = (1.0 - problem.p1[:, 1]) * _g0_p1 + problem.p1[:, 1] * _g1_p1
_right_face = (
    (problem.neighbor < 0)
    & np.isclose(problem.p0[:, 0], 1.0, atol=2.0e-14)
    & np.isclose(problem.p1[:, 0], 1.0, atol=2.0e-14)
)
_right_fixed_flux = np.sum(
    problem.qpf_face[_right_face] + problem.qpl_h_face[_right_face]
    + problem.curl_sign[_right_face]
    * (_psi_fixed_p1[_right_face] - _psi_fixed_p0[_right_face])
)
_right_constant_coefficient = np.sum(
    problem.curl_sign[_right_face]
    * (problem.p1[_right_face, 1] - problem.p0[_right_face, 1])
)
KM_TOP_TRACE_CONSTANT = float(
    (np.sum(problem.cg_face[_right_face]) - _right_fixed_flux)
    / _right_constant_coefficient
)
print(
    f"streamline trace constant: c_top-c_bottom={KM_TOP_TRACE_CONSTANT:.8e}; "
    f"right total fixed from {_right_fixed_flux:.8e} "
    f"to {np.sum(problem.cg_face[_right_face]):.8e}"
)


class KMStreamlinePsiNet(nn.Module):
    """Single-valued potential with exact q.n=0 traces on y=0 and y=1."""

    def __init__(self, frequencies, width, depth):
        super().__init__()
        self.interior = hc.FourierPsiNet(frequencies, width, depth)
        source = np.asarray(problem.source_quad_h_points, dtype=float)
        weight = np.asarray(problem.source_quad_h_weights, dtype=float)
        self.register_buffer("source_x", torch.as_tensor(source[:, 0], dtype=hc.TORCH_DTYPE))
        self.register_buffer("source_y", torch.as_tensor(source[:, 1], dtype=hc.TORCH_DTYPE))
        self.register_buffer("source_weight", torch.as_tensor(weight, dtype=hc.TORCH_DTYPE))
        self.register_buffer(
            "top_trace_constant",
            torch.as_tensor(KM_TOP_TRACE_CONSTANT, dtype=hc.TORCH_DTYPE),
        )
        self._training_trace_cache = {}

    def boundary_trace(self, x_coordinate, y_boundary):
        # Exact x-antiderivative of the point-source representation used by
        # problem.qpl_h_face. Gauss source nodes lie strictly inside their
        # panels, so y_boundary-source_y is nonzero even at the fracture tips.
        distance = float(y_boundary) - self.source_y
        ratio = (x_coordinate[:, None] - self.source_x[None, :]) / distance[None, :]
        ratio_zero = (-self.source_x / distance)[None, :]
        angle = torch.atan(ratio) - torch.atan(ratio_zero)
        trace = torch.sum(angle * self.source_weight[None, :], dim=1) / (2.0 * torch.pi)
        if float(y_boundary) == 1.0:
            trace = trace + self.top_trace_constant
        return trace

    def _fixed_value(self, x):
        y = x[:, 1]
        g_bottom = self.boundary_trace(x[:, 0], 0.0)
        g_top = self.boundary_trace(x[:, 0], 1.0)
        return (1.0 - y) * g_bottom + y * g_top

    def training_fixed_curl_numpy(self, points):
        """Curl of the fixed boundary blend, evaluated once for pointwise data."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        source_x = self.source_x.detach().cpu().numpy()
        source_y = self.source_y.detach().cpu().numpy()
        weight = self.source_weight.detach().cpu().numpy()

        def value_and_dx(y_boundary):
            distance = float(y_boundary) - source_y
            delta = points[:, 0, None] - source_x[None, :]
            angle = (
                np.arctan(delta / distance[None, :])
                - np.arctan((-source_x / distance))[None, :]
            )
            value = angle @ weight / (2.0 * np.pi)
            derivative = (
                (distance[None, :] / (delta * delta + distance[None, :] ** 2))
                @ weight / (2.0 * np.pi)
            )
            return value, derivative

        g0, dg0 = value_and_dx(0.0)
        g1, dg1 = value_and_dx(1.0)
        g1 = g1 + float(self.top_trace_constant.detach().cpu())
        y = points[:, 1]
        return np.column_stack((
            g1 - g0,
            -((1.0 - y) * dg0 + y * dg1),
        ))

    def forward(self, x):
        y = x[:, 1]
        if self.training:
            # The face endpoints and pointwise grid are fixed for every closure.
            # Cache the parameter-free trace values; its curl on the pointwise
            # grid is included once in pointwise_base by run_option_a.
            key = (int(x.data_ptr()), tuple(x.shape), str(x.device), str(x.dtype))
            fixed = self._training_trace_cache.get(key)
            if fixed is None:
                with torch.no_grad():
                    fixed = self._fixed_value(x).detach()
                self._training_trace_cache[key] = fixed
        else:
            # Evaluation needs the true coordinate derivative of the fixed trace.
            fixed = self._fixed_value(x)
        interior_window = y * (1.0 - y)
        return fixed + interior_window * self.interior(x)


def km_model_factory(frequencies, width, depth):
    return KMStreamlinePsiNet(frequencies, width, depth)


horizontal_boundary_face = (
    (problem.neighbor < 0)
    & (
        (np.isclose(problem.p0[:, 1], 0.0, atol=2.0e-14)
         & np.isclose(problem.p1[:, 1], 0.0, atol=2.0e-14))
        | (np.isclose(problem.p0[:, 1], 1.0, atol=2.0e-14)
           & np.isclose(problem.p1[:, 1], 1.0, atol=2.0e-14))
    )
)
training_face_mask = ~horizontal_boundary_face
print(
    "face-loss mask: active=", int(np.count_nonzero(training_face_mask)),
    "excluded horizontal-boundary pieces=", int(np.count_nonzero(horizontal_boundary_face)),
)


def gate_hard_streamline_boundary(tol=2.0e-12):
    if np.max(np.abs(problem.qpf_face[horizontal_boundary_face])) > 1.0e-15:
        raise RuntimeError("KM hard-boundary trace assumes the smooth particular field is zero")
    probe_model = km_model_factory((1, 2, 4, 8), 32, 3).to(dtype=hc.TORCH_DTYPE)
    probe_model.eval()

    # First check every native horizontal boundary face piece.
    ids = np.flatnonzero(horizontal_boundary_face)
    with torch.no_grad():
        psi0 = probe_model(torch.as_tensor(problem.p0[ids], dtype=hc.TORCH_DTYPE)).cpu().numpy()
        psi1 = probe_model(torch.as_tensor(problem.p1[ids], dtype=hc.TORCH_DTYPE)).cpu().numpy()
    native_flux = (
        problem.qpf_face[ids] + problem.qpl_h_face[ids]
        + problem.curl_sign[ids] * (psi1 - psi0)
    )

    # Then check a separate dense family of subsegments, including segments
    # adjacent to the boundary fracture tips. This is independent of the face
    # loss and verifies the cumulative-trace formula segment by segment.
    x_nodes = np.unique(np.r_[
        np.linspace(0.0, 1.0, 257), FRAC_A[0], FRAC_B[0],
        np.asarray([0.249999, 0.250001, 0.749999, 0.750001]),
    ])
    arbitrary = []
    for y_boundary, normal_y in ((0.0, -1.0), (1.0, 1.0)):
        a = np.column_stack((x_nodes[:-1], np.full(len(x_nodes) - 1, y_boundary)))
        b = np.column_stack((x_nodes[1:], np.full(len(x_nodes) - 1, y_boundary)))
        normals = np.column_stack((np.zeros(len(a)), np.full(len(a), normal_y)))
        base_flux = hc._line_source_face_flux(
            a, b, normals, problem.source_quad_h_points, problem.source_quad_h_weights
        )
        with torch.no_grad():
            va = probe_model(torch.as_tensor(a, dtype=hc.TORCH_DTYPE)).cpu().numpy()
            vb = probe_model(torch.as_tensor(b, dtype=hc.TORCH_DTYPE)).cpu().numpy()
        tangent = b - a
        n_right = np.column_stack((tangent[:, 1], -tangent[:, 0]))
        n_right /= np.linalg.norm(tangent, axis=1)[:, None]
        orientation = np.sign(np.einsum("ij,ij->i", normals, n_right))
        arbitrary.append(base_flux + orientation * (vb - va))
    arbitrary_flux = np.concatenate(arbitrary)
    native_max = float(np.max(np.abs(native_flux)))
    arbitrary_max = float(np.max(np.abs(arbitrary_flux)))
    passed = max(native_max, arbitrary_max) <= tol
    print(
        f"hard streamline-boundary gate: {'PASS' if passed else 'FAIL'}, "
        f"native max={native_max:.3e}, dense subsegment max={arbitrary_max:.3e}"
    )
    if not passed:
        raise RuntimeError("hard streamline-boundary trace gate failed")
    return {
        "passed": True, "tolerance": tol,
        "native_face_max_abs_flux": native_max,
        "dense_subsegment_max_abs_flux": arbitrary_max,
        "excluded_face_loss_pieces": int(np.count_nonzero(horizontal_boundary_face)),
        "left_right_boundaries_constrained": False,
    }


hard_streamline_gate = gate_hard_streamline_boundary()


def accepted_step_logger(iteration, model, loss_fn):
    loss = float(loss_fn().detach().cpu())
    accepted_lbfgs.append({"iteration": int(iteration), "loss": loss})
    if iteration == 0 or iteration % 50 == 0:
        print(f"accepted L-BFGS {iteration:4d}: loss={loss:.8e}")
    if iteration < STAGNATION_WINDOW:
        return False
    old = accepted_lbfgs[-1 - STAGNATION_WINDOW]["loss"]
    relative_decrease = (old - loss) / max(abs(old), 1.0e-30)
    if relative_decrease < STAGNATION_RTOL:
        print(
            f"relative stagnation at accepted step {iteration}: "
            f"{relative_decrease:.3e} < {STAGNATION_RTOL:.1e}"
        )
        return True
    return False


if CHECKPOINT_PATH.exists():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if (
        int(checkpoint["width"]) != 32 or int(checkpoint["depth"]) != 3
        or tuple(checkpoint["frequencies"]) != (1, 2, 4, 8)
        or int(checkpoint["seed"]) != SEED
        or not bool(checkpoint.get("hard_streamline_bc", False))
        or not np.isclose(
            float(checkpoint.get("top_trace_constant", np.nan)),
            KM_TOP_TRACE_CONSTANT, rtol=0.0, atol=1.0e-14,
        )
    ):
        raise RuntimeError("existing Option-A checkpoint does not match the frozen protocol")
    model = km_model_factory(
        checkpoint["frequencies"], checkpoint["width"], checkpoint["depth"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x0 = torch.as_tensor(problem.p0, dtype=hc.TORCH_DTYPE)
    x1 = torch.as_tensor(problem.p1, dtype=hc.TORCH_DTYPE)
    sign = torch.as_tensor(problem.curl_sign, dtype=hc.TORCH_DTYPE)
    with torch.no_grad():
        curl_face = (sign * (model(x1) - model(x0))).cpu().numpy()
    option_a = {
        "model": model,
        "face_flux": problem.qpf_face + problem.qpl_h_face + curl_face,
        "width": checkpoint["width"], "depth": checkpoint["depth"],
        "frequencies": tuple(checkpoint["frequencies"]),
        "seed": checkpoint["seed"], "history": checkpoint["history"],
        "optimization": checkpoint["optimization"],
        "wall_time": checkpoint["wall_time"],
        "particular_lambda_mode": "h",
        "hard_streamline_bc": True,
    }
    accepted_lbfgs = list(checkpoint["accepted_step_history"])
    print("loaded validated Option-A checkpoint:", CHECKPOINT_PATH)
else:
    option_a = hc.run_option_a(
        problem, adam_steps=2000, lbfgs_steps=3000,
        width=32, depth=3, frequencies=(1, 2, 4, 8), lr=2.0e-3,
        seed=SEED, face_weight=1.0, potential_weight=0.0,
        pointwise_weight=1.0, target_mode="cg", particular_lambda_mode="h",
        lbfgs_iteration_callback=accepted_step_logger,
        model_factory=km_model_factory, face_mask=training_face_mask,
    )
    option_a["hard_streamline_bc"] = True

    checkpoint = {
        "state_dict": option_a["model"].state_dict(),
        "width": option_a["width"], "depth": option_a["depth"],
        "frequencies": list(option_a["frequencies"]), "seed": option_a["seed"],
        "history": option_a["history"], "optimization": option_a["optimization"],
        "wall_time": option_a["wall_time"],
        "accepted_step_history": accepted_lbfgs,
        "face_weight": 1.0, "pointwise_weight": 1.0, "potential_weight": 0.0,
        "hard_streamline_bc": True,
        "top_trace_constant": KM_TOP_TRACE_CONSTANT,
        "excluded_horizontal_boundary_face_pieces": int(
            np.count_nonzero(horizontal_boundary_face)
        ),
        "FRAC_A": FRAC_A, "FRAC_B": FRAC_B, "normal": normal_np, "tau": tau_np,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print("saved:", CHECKPOINT_PATH)
'''

cg_figure = "".join(source_nb["cells"][6]["source"])
cg_figure = cg_figure.replace(
    "fig.tight_layout()\nplt.show()",
    "fig.tight_layout()\nfig.savefig(OUTDIR / 'fig_km_cg_solution.png', dpi=600, bbox_inches='tight')\nplt.show()\nprint('saved:', OUTDIR / 'fig_km_cg_solution.png')",
)

zeta_source = "".join(nlr_nb["cells"][11]["source"])
zeta_source = zeta_source.split("_abs_R_zeta_deng =", 1)[0]
zeta_source = zeta_source.split('print("Rotated rectangular transport-CV residual', 1)[0]
zeta_source = zeta_source.replace("residual_stats(", "summary_stats(")
zeta_setup = r'''
# Rotated 45-degree zeta geometry shared by all three FEM-based fluxes.
def summary_stats(values, ids=None):
    values = np.asarray(values, dtype=float).reshape(-1)
    if ids is not None:
        values = values[np.asarray(ids, dtype=np.int64)]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "mean_abs": None, "max": None}
    return {
        "n": int(len(values)), "mean_abs": float(np.mean(np.abs(values))),
        "max": float(np.max(np.abs(values))),
    }

''' + zeta_source

_nlr_source = "".join(nlr_nb["cells"][8]["source"])
_nlr_defs, _nlr_run_tail = _nlr_source.split("t0 = time.perf_counter()", 1)
_nlr_run = "t0 = time.perf_counter()" + _nlr_run_tail
nlr_solve = _nlr_defs + r'''
# Cache the deterministic local reconstruction coefficients. This keeps notebook
# restarts cheap during diagnostics without changing the reconstruction.
NLR_CACHE = OUTDIR / f"km_case1_nlr_local_data_ref{REF}_v2.npz"
if NLR_CACHE.exists():
    _cached = np.load(NLR_CACHE)
    _coeffs = np.asarray(_cached["coeffs"], dtype=float)
    if _coeffs.shape != (len(rect["cells"]), 4):
        raise RuntimeError(f"stale NLR cache shape {_coeffs.shape}")
    local_fluxes = {
        cid: {
            "coeffs": _coeffs[cid].copy(),
            "source_t": np.asarray(_cached["source_t"][cid], dtype=float),
            "source_phi": np.asarray(_cached["source_phi"][cid], dtype=float),
            "rhs": np.asarray(_cached["rhs"][cid], dtype=float),
        }
        for cid in range(len(rect["cells"]))
    }
    print("loaded deterministic NLR coefficient cache:", NLR_CACHE)
else:
''' + "".join("    " + line if line.strip() else line for line in _nlr_run.splitlines(keepends=True)) + r'''
    _coeffs = np.vstack([local_fluxes[cid]["coeffs"] for cid in range(len(rect["cells"]))])
    np.savez_compressed(
        NLR_CACHE, coeffs=_coeffs,
        source_t=np.vstack([local_fluxes[cid]["source_t"] for cid in range(len(rect["cells"]))]),
        source_phi=np.vstack([local_fluxes[cid]["source_phi"] for cid in range(len(rect["cells"]))]),
        rhs=np.vstack([local_fluxes[cid]["rhs"] for cid in range(len(rect["cells"]))]),
    )
    print("saved deterministic NLR coefficient cache:", NLR_CACHE)
'''

diagnostics = r'''
# ============================================================
# Corrected common audit: CG / NLR / PINN on dual and zeta CVs
# ============================================================

def q_nlr_cell_local_numpy(points, cell_ids):
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    cids = np.asarray(cell_ids, dtype=np.int32).reshape(-1)
    out = np.zeros((len(pts), 2), dtype=float)
    for cid in np.unique(cids):
        idx = np.flatnonzero(cids == cid)
        cell = rect["cells"][int(cid)]
        local = pts[idx]
        x0, y0 = cell["xy"][0]
        xi = np.column_stack((
            (local[:, 0] - x0) / cell["hx"],
            (local[:, 1] - y0) / cell["hy"],
        ))
        grads = q1_grad_phys(xi, cell["hx"], cell["hy"])
        out[idx] = -K_M_VALUE * np.einsum(
            "qad,a->qd", grads, local_fluxes[int(cid)]["coeffs"]
        )
    return out


nlr_face = integrate_hosted_segments(
    problem.p0, problem.p1, problem.normals, problem.host_cell,
    order=32, evaluator=q_nlr_cell_local_numpy,
)
audit_cg = hc.print_audit(problem, problem.cg_face, "CG corrected dual-CV audit")
audit_nlr = hc.print_audit(problem, nlr_face, "NLR corrected dual-CV audit")
audit_pinn = hc.print_audit(problem, option_a["face_flux"], "PINN corrected dual-CV audit")

nlr_tolerances = {
    "interior-away": 1.0e-12, "fracture-adjacent": 1.0e-12,
    "fracture-cut": 1.0e-11,
}
nlr_gates = {}
for cls in ("interior-away", "fracture-cut", "fracture-adjacent", "boundary"):
    stats = audit_nlr["stats"]["lambda_h"][cls]
    if stats["n"] == 0:
        nlr_gates[cls] = {"status": "N/A", "passed": True, "tolerance": None}
    elif cls == "boundary":
        nlr_gates[cls] = {
            "status": "reported separately; ungated", "passed": True,
            "tolerance": None, "max": stats["max"],
        }
    else:
        tolerance = nlr_tolerances[cls]
        passed = bool(stats["max"] <= tolerance)
        nlr_gates[cls] = {
            "status": "gated", "passed": passed,
            "tolerance": tolerance, "max": stats["max"],
        }
        if not passed:
            raise RuntimeError(
                f"NLR {cls} gate failed: max={stats['max']:.3e}, tol={tolerance:.1e}"
            )

for cls in ("interior-away", "fracture-cut", "fracture-adjacent", "boundary"):
    stats = audit_pinn["stats"]["lambda_h"][cls]
    if stats["n"] and stats["max"] > 1.0e-12:
        raise RuntimeError(f"PINN {cls} gate failed: max={stats['max']:.3e}")


# Rebuild the rotated-volume RHS with the same node-split line-source
# convention used by Gate A1 and the common dual audit. The source notebook's
# legacy zeta RHS used its older interval convention and is intentionally not
# reused for paper numbers.
zeta_cv_polys = [[order_polygon_vertices(poly)] for poly in zeta_polys]
zeta_lambda_h, zeta_cut_mask, zeta_touch_mask = hc._distribute_line_quadrature_to_cvs(
    zeta_cv_polys,
    problem.source_quad_h_points,
    problem.source_quad_h_weights,
    normal_np,
)
zeta_frac_ids = np.flatnonzero(zeta_cut_mask).astype(np.int32)


def corrected_cg_nlr_zeta_residuals():
    q_cg = q_rect_cell_local_numpy(
        zeta_data["edge_points"], zeta_data["edge_cell_ids"], use_rec=False
    )
    q_nlr = q_rect_cell_local_numpy(
        zeta_data["edge_points"], zeta_data["edge_cell_ids"], use_rec=True
    )
    flux_cg = np.zeros(len(zeta_polys), dtype=float)
    flux_nlr = np.zeros(len(zeta_polys), dtype=float)
    np.add.at(
        flux_cg, zeta_data["edge_eta_ids"],
        np.sum(q_cg * zeta_data["edge_nw"], axis=1),
    )
    np.add.at(
        flux_nlr, zeta_data["edge_eta_ids"],
        np.sum(q_nlr * zeta_data["edge_nw"], axis=1),
    )
    rhs = zeta_data["source"] + zeta_lambda_h
    return flux_cg - rhs, flux_nlr - rhs


R_zeta_cg, R_zeta_deng = corrected_cg_nlr_zeta_residuals()


def native_pinn_zeta_residual(polys):
    faces = []
    owners = []
    for zid, poly in enumerate(polys):
        poly = order_polygon_vertices(poly)
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            edge = b - a
            ell = float(np.linalg.norm(edge))
            if ell <= 1.0e-14:
                continue
            normal = np.array([edge[1], -edge[0]], dtype=float) / ell
            hit = hc._segment_fracture_intersection(
                a, b, FRAC_A, tau_np, normal_np, L_gamma
            )
            endpoints = [a]
            if hit is not None:
                endpoints.append(a + hit[0] * edge)
            endpoints.append(b)
            for p0, p1 in zip(endpoints[:-1], endpoints[1:]):
                if np.linalg.norm(p1 - p0) > 1.0e-14:
                    faces.append((p0.copy(), p1.copy(), normal.copy()))
                    owners.append(int(zid))
    p0 = np.vstack([face[0] for face in faces])
    p1 = np.vstack([face[1] for face in faces])
    normals = np.vstack([face[2] for face in faces])
    qpl = hc._line_source_face_flux(
        p0, p1, normals,
        problem.source_quad_h_points, problem.source_quad_h_weights,
    )
    x0 = torch.as_tensor(p0, dtype=hc.TORCH_DTYPE)
    x1 = torch.as_tensor(p1, dtype=hc.TORCH_DTYPE)
    sign = torch.as_tensor(
        hc._curl_endpoint_sign(p0, p1, normals), dtype=hc.TORCH_DTYPE
    )
    with torch.no_grad():
        curl = sign * (option_a["model"](x1) - option_a["model"](x0))
    face_flux = qpl + curl.cpu().numpy()
    total = np.zeros(len(polys), dtype=float)
    np.add.at(total, np.asarray(owners, dtype=np.int32), face_flux)
    return total - zeta_data["source"] - zeta_lambda_h


R_zeta_pinn = native_pinn_zeta_residual(zeta_polys)
zeta_stats = {
    "CG": {
        "all": summary_stats(R_zeta_cg),
        "fracture-cut": summary_stats(R_zeta_cg, zeta_frac_ids),
    },
    "NLR": {
        "all": summary_stats(R_zeta_deng),
        "fracture-cut": summary_stats(R_zeta_deng, zeta_frac_ids),
    },
    "PINN": {
        "all": summary_stats(R_zeta_pinn),
        "fracture-cut": summary_stats(R_zeta_pinn, zeta_frac_ids),
    },
}
for scope, stats in zeta_stats["PINN"].items():
    if stats["n"] and stats["max"] > 1.0e-12:
        raise RuntimeError(f"PINN zeta {scope} gate failed: max={stats['max']:.3e}")

r_tau_pinn = hc.native_r_tau(problem, option_a)


def audit_class_table(audit):
    out = {}
    for cls in ("interior-away", "fracture-cut", "fracture-adjacent", "boundary"):
        stats = audit["stats"]["lambda_h"][cls]
        out[cls] = {
            "n": int(stats["n"]),
            "mean_abs": None if stats["n"] == 0 else float(stats["mean_abs"]),
            "max": None if stats["n"] == 0 else float(stats["max"]),
        }
    return out


conservation_report = {
    "case": "Koppel-Martin Case 1, sealed fracture tips",
    "matrix_primal_grid": "64x64 Q1 rectangles",
    "transport_cv_grid": "65x65 vertex-centred median dual",
    "audit_formula": (
        "closed-boundary flux - integral_CV(f_m) "
        "- integral_(Gamma intersect CV)(lambda_h)"
    ),
    "gates": {
        "mrst_dual_geometry": mrst_dual_gate,
        "sealed_tip_exchange": sealed_tip_gate,
        "A0_generic": a0_generic, "A0_boundary_tip_geometry": a0_km,
        "A1": a1, "hard_streamline_boundary": hard_streamline_gate,
        "NLR": nlr_gates,
    },
    "R_xi": {
        "CG": audit_class_table(audit_cg),
        "NLR": audit_class_table(audit_nlr),
        "PINN": audit_class_table(audit_pinn),
    },
    "R_zeta": zeta_stats,
    "R_tau_PINN_native": {
        "lambda_h_RHS": r_tau_pinn["lambda_h_stats"],
        "formula": (
            "native closed element-boundary flux - integral_cell(f_m) "
            "- integral_(Gamma intersect cell)(lambda_h)"
        ),
    },
    "PINN_training": {
        "wall_time_s": option_a["wall_time"],
        "final_loss": option_a["optimization"]["lbfgs"]["final_loss"],
        "final_grad_l2": option_a["optimization"]["lbfgs"]["final_grad_l2"],
        "final_grad_inf": option_a["optimization"]["lbfgs"]["final_grad_inf"],
        "iterations": option_a["optimization"]["lbfgs"]["iterations"],
        "stop_reason": option_a["optimization"]["lbfgs"]["stop_reason"],
        "loss": "integrated dual-face CG data plus fixed-grid pointwise CG data",
        "excluded_horizontal_boundary_face_pieces": int(
            np.count_nonzero(horizontal_boundary_face)
        ),
        "hard_streamline_boundary": True,
    },
}


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


(OUTDIR / "km_case1_conservation.json").write_text(
    json.dumps(json_clean(conservation_report), indent=2, allow_nan=False) + "\n"
)
print(json.dumps(json_clean(conservation_report), indent=2, allow_nan=False))
print("saved:", OUTDIR / "km_case1_conservation.json")
'''

transport_source = "".join(source_nb["cells"][24]["source"])
transport_port = extract_functions(transport_source, [
    "fractional_flow_bl",
    "pressure_driven_fracture_direction",
    "orient_fracture_q_face",
    "fracture_direction_label",
    "lcg_fracture_pressure_nodes",
    "build_lcg_fracture_transport_grid",
    "_mrst_fracture_edges",
    "build_mrst_fracture_transport_grid",
    "_split_interval_at_knots",
    "integrate_lcg_exchange_interval_to_nodes",
    "fracture_outflow_sum",
    "fracture_advection_update",
])

mrst_aggregation = r'''
# ============================================================
# Stage 5: exact MRST-to-dual aggregation and validation
# ============================================================
from scipy.io import loadmat

MRST_CASE3_PATH = pathlib.Path("case3_ecmor/case3_mrst_export_noflow_v2.mat")
if not MRST_CASE3_PATH.exists():
    MRST_CASE3_PATH = pathlib.Path(
        "fenicsx/code/fracture problem/case3_ecmor/case3_mrst_export_noflow_v2.mat"
    )
if not MRST_CASE3_PATH.exists():
    raise FileNotFoundError("case3_mrst_export_noflow_v2.mat was not found")
mrst = loadmat(MRST_CASE3_PATH, squeeze_me=True)

_required_mrst = [
    "xc_matrix", "p_matrix", "xc_frac", "s_frac", "p_frac",
    "face_p1", "face_p2", "face_centroid", "face_normal", "face_len",
    "face_flux", "face_neighbors", "nnc_mat_cell", "nnc_frac_cell",
    "nnc_flux_m2f", "nnc_s", "snap_T_abs", "sw_matrix_snaps",
]
_missing_mrst = [name for name in _required_mrst if name not in mrst]
if _missing_mrst:
    raise RuntimeError(f"MRST export is missing required fields: {_missing_mrst}")

mrst_xc_matrix = np.asarray(mrst["xc_matrix"], dtype=float)
mrst_p_matrix = np.asarray(mrst["p_matrix"], dtype=float).reshape(-1)
mrst_xc_frac = np.asarray(mrst["xc_frac"], dtype=float)
mrst_s_frac = np.asarray(mrst["s_frac"], dtype=float).reshape(-1)
mrst_p_frac = np.asarray(mrst["p_frac"], dtype=float).reshape(-1)
mrst_face_p1 = np.asarray(mrst["face_p1"], dtype=float)
mrst_face_p2 = np.asarray(mrst["face_p2"], dtype=float)
mrst_face_centroid = np.asarray(mrst["face_centroid"], dtype=float)
mrst_face_normal = np.asarray(mrst["face_normal"], dtype=float)
mrst_face_len = np.asarray(mrst["face_len"], dtype=float).reshape(-1)
mrst_face_flux = np.asarray(mrst["face_flux"], dtype=float).reshape(-1)
mrst_face_neighbors = np.asarray(mrst["face_neighbors"], dtype=np.int64)

_mrst_dims = tuple(np.asarray(mrst["meta_celldim"], dtype=int).reshape(-1)[:2])
if _mrst_dims != (128, 128):
    raise RuntimeError(f"Expected a 128x128 MRST matrix grid, got {_mrst_dims}")
if len(mrst_xc_matrix) != 128 * 128:
    raise RuntimeError("MRST matrix-cell count is inconsistent with 128x128")


def polygon_area(poly):
    poly = np.asarray(poly, dtype=float)
    return 0.5 * abs(float(
        np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
        - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
    ))


def _coord_to_cv_lookup():
    lookup = {}
    for cid, xy in enumerate(problem.coords):
        key = tuple(np.rint(64.0 * np.asarray(xy)).astype(np.int64))
        if key in lookup:
            raise RuntimeError(f"duplicate dual coordinate key {key}")
        lookup[key] = int(cid)
    return lookup


_cv_lookup = _coord_to_cv_lookup()
_ix = np.floor(128.0 * mrst_xc_matrix[:, 0]).astype(np.int64)
_iy = np.floor(128.0 * mrst_xc_matrix[:, 1]).astype(np.int64)
_vi = (_ix + 1) // 2
_vj = (_iy + 1) // 2
mrst_cell_to_cv = np.asarray([
    _cv_lookup[(int(i), int(j))] for i, j in zip(_vi, _vj)
], dtype=np.int32)
_mrst_per_cv = np.bincount(mrst_cell_to_cv, minlength=len(problem.coords))
_expected_per_cv = np.asarray([
    (1 if np.isclose(x, 0.0) or np.isclose(x, 1.0) else 2)
    * (1 if np.isclose(y, 0.0) or np.isclose(y, 1.0) else 2)
    for x, y in problem.coords
], dtype=np.int32)
if not np.array_equal(_mrst_per_cv, _expected_per_cv):
    raise RuntimeError("MRST matrix cells do not form the expected 1x1/1x2/2x2 dual unions")
dual_areas = np.asarray([
    sum(polygon_area(subpoly) for subpoly in subpolys)
    for subpolys in problem.cv_polys
], dtype=float)
_area_from_mrst = _mrst_per_cv.astype(float) / (128.0 ** 2)
_area_error = float(np.max(np.abs(dual_areas - _area_from_mrst)))
if _area_error > 5.0e-15:
    raise RuntimeError(f"MRST-to-dual union area gate failed: {_area_error:.3e}")

_nnc_mat_cell = np.asarray(mrst["nnc_mat_cell"], dtype=np.int64).reshape(-1)
_nnc_q_m2f = np.asarray(mrst["nnc_flux_m2f"], dtype=float).reshape(-1)
_nnc_frac_cell = np.asarray(mrst["nnc_frac_cell"], dtype=np.int64).reshape(-1)
_nnc_s = np.asarray(mrst["nnc_s"], dtype=float).reshape(-1)
if not (
    len(_nnc_mat_cell) == len(_nnc_q_m2f)
    == len(_nnc_frac_cell) == len(_nnc_s)
):
    raise RuntimeError("MRST NNC arrays have inconsistent lengths")
if np.any(_nnc_mat_cell < 1) or np.any(_nnc_mat_cell > len(mrst_cell_to_cv)):
    raise RuntimeError("MRST NNC exchange is not resolved by valid host matrix-cell indices")
nnc_host_cv = mrst_cell_to_cv[_nnc_mat_cell - 1]

mrst_aggregation_gate = {
    "passed": True,
    "mrst_grid": [128, 128],
    "dual_grid": [65, 65],
    "matrix_cells_assigned_once": int(len(mrst_cell_to_cv)),
    "max_union_area_error": _area_error,
    "union_count_histogram": {
        str(k): int(np.count_nonzero(_mrst_per_cv == k)) for k in (1, 2, 4)
    },
    "nnc_connections": int(len(_nnc_mat_cell)),
    "nnc_resolution": "per matrix cell (direct nnc_mat_cell mapping)",
}
print("MRST exact aggregation gate: PASS")
print(json.dumps(mrst_aggregation_gate, indent=2))


def _face_group_key(fid):
    return (
        int(problem.owner[fid]), int(problem.neighbor[fid]),
        int(problem.host_cell[fid]),
        tuple(np.rint(1.0e12 * problem.normals[fid]).astype(np.int64)),
    )


def build_dual_face_groups():
    groups = {}
    for fid in range(len(problem.owner)):
        groups.setdefault(_face_group_key(fid), []).append(int(fid))
    records = []
    for key, ids in groups.items():
        ids = np.asarray(ids, dtype=np.int32)
        normal = np.asarray(problem.normals[ids[0]], dtype=float)
        tangent = np.array([-normal[1], normal[0]], dtype=float)
        endpoints = np.vstack((problem.p0[ids], problem.p1[ids]))
        projection = endpoints @ tangent
        p0 = endpoints[int(np.argmin(projection))]
        p1 = endpoints[int(np.argmax(projection))]
        span = float(np.linalg.norm(p1 - p0))
        pieces = float(np.sum(np.linalg.norm(problem.p1[ids] - problem.p0[ids], axis=1)))
        if abs(span - pieces) > 2.0e-13:
            raise RuntimeError("split dual-face pieces did not recombine into one segment")
        records.append({
            "piece_ids": ids, "owner": key[0], "neighbor": key[1],
            "host_cell": key[2], "normal": normal,
            "p0": p0.copy(), "p1": p1.copy(), "length": span,
            "centroid": 0.5 * (p0 + p1),
        })
    return records


dual_face_groups = build_dual_face_groups()


def _segment_key(a, b):
    ia = tuple(np.rint(128.0 * np.asarray(a)).astype(np.int64))
    ib = tuple(np.rint(128.0 * np.asarray(b)).astype(np.int64))
    return tuple(sorted((ia, ib)))


_mrst_face_by_segment = {}
for fid, (a, b) in enumerate(zip(mrst_face_p1, mrst_face_p2)):
    key = _segment_key(a, b)
    if key in _mrst_face_by_segment:
        raise RuntimeError(f"duplicate MRST matrix face segment {key}")
    _mrst_face_by_segment[key] = int(fid)

dual_group_mrst_fid = np.empty(len(dual_face_groups), dtype=np.int32)
dual_group_mrst_flux = np.empty(len(dual_face_groups), dtype=float)
for gid, group in enumerate(dual_face_groups):
    key = _segment_key(group["p0"], group["p1"])
    if key not in _mrst_face_by_segment:
        raise RuntimeError(f"dual face is not an exact exported MRST face: {key}")
    fid = _mrst_face_by_segment[key]
    if abs(group["length"] - mrst_face_len[fid]) > 2.0e-13:
        raise RuntimeError("dual/MRST face lengths differ")
    sign = float(np.sign(np.dot(group["normal"], mrst_face_normal[fid])))
    if sign == 0.0:
        raise RuntimeError("dual and MRST face normals are orthogonal")
    dual_group_mrst_fid[gid] = fid
    dual_group_mrst_flux[gid] = sign * mrst_face_flux[fid]

_face_use = np.bincount(dual_group_mrst_fid, minlength=len(mrst_face_flux))
_ip1 = np.rint(128.0 * mrst_face_p1).astype(np.int64)
_ip2 = np.rint(128.0 * mrst_face_p2).astype(np.int64)
_vertical = _ip1[:, 0] == _ip2[:, 0]
_gridline_index = np.where(_vertical, _ip1[:, 0], _ip1[:, 1])
_is_physical_boundary = np.any(mrst_face_neighbors == 0, axis=1)
_lies_on_dual_boundary = _is_physical_boundary | ((_gridline_index % 2) == 1)
_expected_face_use = _lies_on_dual_boundary.astype(np.int64)
if not np.array_equal(_face_use, _expected_face_use):
    raise RuntimeError(
        "MRST faces were not used exactly once on dual boundaries and zero "
        "times inside dual-CV unions"
    )


def aggregate_piece_flux(piece_flux):
    piece_flux = np.asarray(piece_flux, dtype=float)
    return np.asarray([
        float(np.sum(piece_flux[group["piece_ids"]])) for group in dual_face_groups
    ])


dual_group_flux = {
    "CG": aggregate_piece_flux(problem.cg_face),
    "NLR": aggregate_piece_flux(nlr_face),
    "PINN": aggregate_piece_flux(option_a["face_flux"]),
    "MRST": dual_group_mrst_flux.copy(),
}


def validation_stats(diff, ref):
    diff = np.asarray(diff, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return {
        "n": int(diff.size),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
        "relative_rmse": float(
            np.sqrt(np.mean(diff ** 2)) / max(np.sqrt(np.mean(ref ** 2)), 1.0e-30)
        ),
    }


p_matrix_lcg_on_mrst = eval_fem_function(p_m, mrst_xc_matrix, omega).reshape(-1)
p_matrix_diff_lcg_mrst = p_matrix_lcg_on_mrst - mrst_p_matrix
_unique_group = np.asarray([
    group["neighbor"] < 0 or group["owner"] < group["neighbor"]
    for group in dual_face_groups
], dtype=bool)
dual_flux_diff_pinn_mrst = (
    dual_group_flux["PINN"][_unique_group]
    - dual_group_flux["MRST"][_unique_group]
)
mrst_validation = {
    "pressure_LCG_minus_MRST": validation_stats(
        p_matrix_diff_lcg_mrst, mrst_p_matrix
    ),
    "dual_face_PINN_minus_aggregated_MRST": validation_stats(
        dual_flux_diff_pinn_mrst, dual_group_flux["MRST"][_unique_group]
    ),
}

from mpl_toolkits.axes_grid1 import make_axes_locatable
with plt.rc_context({"font.size": 13}):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    lim_p = max(float(np.percentile(np.abs(p_matrix_diff_lcg_mrst), 99.0)), 1.0e-14)
    sc0 = axes[0].scatter(
        mrst_xc_matrix[:, 0], mrst_xc_matrix[:, 1],
        c=p_matrix_diff_lcg_mrst, s=3, cmap="RdBu_r",
        vmin=-lim_p, vmax=lim_p, rasterized=True,
    )
    axes[0].set_title(r"i) $\Delta p=p_h-p_{\mathrm{FV}}$")
    lim_f = max(float(np.percentile(np.abs(dual_flux_diff_pinn_mrst), 99.5)), 1.0e-14)
    centroids = np.vstack([g["centroid"] for g in dual_face_groups])[_unique_group]
    sc1 = axes[1].scatter(
        centroids[:, 0], centroids[:, 1], c=dual_flux_diff_pinn_mrst,
        s=3, cmap="RdBu_r", vmin=-lim_f, vmax=lim_f, rasterized=True,
    )
    axes[1].set_title(
        r"ii) $\int_e(\mathbf{v}_\theta-\mathbf{v}_{\mathrm{FV}})\cdot\mathbf{n}$"
    )
    for ax, sc in zip(axes, (sc0, sc1)):
        ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_aspect("equal")
        ax.set_xlabel(r"$x$"); ax.grid(False)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(sc, cax=cax)
    axes[0].set_ylabel(r"$y$")
    axes[1].tick_params(labelleft=False)
    fig.savefig(OUTDIR / "fig_km_mrst_validation.png", dpi=600, bbox_inches="tight")
    plt.show()
print("saved:", OUTDIR / "fig_km_mrst_validation.png")
'''

transport = r'''
# ============================================================
# Stage 6: synchronized four-track transport on the 65x65 dual grid
# ============================================================
TRANSPORT_CFL = float(np.asarray(mrst.get("meta_CFL", 0.45)).reshape(-1)[0])
TRANSPORT_FPRIME_MAX = float(
    np.asarray(mrst.get("meta_FPRIME_MAX", 2.0)).reshape(-1)[0]
)
TRANSPORT_RIGHT_BC_S = 1.0
TRANSPORT_DEFAULT_INFLOW_S = 0.0
TRANSPORT_FRAC_POROSITY = 1.0
TRANSPORT_FRAC_TIP_S_A = 0.0
TRANSPORT_FRAC_TIP_S_B = 0.0
TRANSPORT_EXCHANGE_TOL = 1.0e-14
TRANSPORT_MAX_STEPS = 200000

''' + transport_port + r'''


def build_lcg_dual_exchange(fracture):
    conn_cv, conn_frac, conn_E, conn_s = [], [], [], []
    for cid, subpolys in enumerate(problem.cv_polys):
        for poly in subpolys:
            interval = hc._line_interval_in_polygon(
                np.asarray(poly), FRAC_A, tau_np, normal_np, L_gamma
            )
            if interval is None:
                continue
            s0, s1 = interval
            for frac_id, value in integrate_lcg_exchange_interval_to_nodes(
                s0, s1, fracture, sign=1.0
            ):
                conn_cv.append(int(cid)); conn_frac.append(int(frac_id))
                conn_E.append(float(value)); conn_s.append(float(fracture["s_centers"][frac_id]))
    conn_cv = np.asarray(conn_cv, dtype=np.int32)
    conn_frac = np.asarray(conn_frac, dtype=np.int32)
    conn_E = np.asarray(conn_E, dtype=float)
    e_by_cv = np.bincount(conn_cv, weights=conn_E, minlength=len(problem.coords))
    mismatch = float(np.max(np.abs(e_by_cv - problem.cv_lambda_h)))
    if mismatch > 1.0e-12:
        raise RuntimeError(f"LCG exchange-to-dual mapping failed: {mismatch:.3e}")
    return {
        "kind": "lcg_lambda_h", "conn_cv_ids": conn_cv,
        "conn_frac_ids": conn_frac, "conn_E": conn_E,
        "conn_s": np.asarray(conn_s, dtype=float), "E_by_cell": e_by_cv,
        "max_cv_integral_mismatch": mismatch,
    }


def build_mrst_dual_exchange(fracture):
    conn_cv = nnc_host_cv.astype(np.int32, copy=True)
    conn_E = -_nnc_q_m2f.astype(float, copy=True)
    edges = np.asarray(fracture["s_faces"], dtype=float)
    conn_frac = np.searchsorted(edges, _nnc_s, side="right") - 1
    conn_frac = np.clip(conn_frac, 0, len(edges) - 2).astype(np.int32)
    e_by_cv = np.bincount(conn_cv, weights=conn_E, minlength=len(problem.coords))
    return {
        "kind": "mrst_nnc_per_host_matrix_cell", "conn_cv_ids": conn_cv,
        "conn_frac_ids": conn_frac, "conn_E": conn_E,
        "conn_s": _nnc_s.copy(), "conn_mrst_frac_cell": _nnc_frac_cell.copy(),
        "E_by_cell": e_by_cv,
    }


def group_side(group):
    if group["neighbor"] >= 0:
        return "interior"
    mid = group["centroid"]
    normal = group["normal"]
    if np.isclose(mid[0], 0.0): return "left"
    if np.isclose(mid[0], 1.0): return "right"
    if np.isclose(mid[1], 0.0): return "bottom"
    if np.isclose(mid[1], 1.0): return "top"
    raise RuntimeError(f"unclassified boundary dual face at {mid}, normal={normal}")


def oriented_track(name):
    return {
        "name": name,
        "owner": np.asarray([g["owner"] for g in dual_face_groups], dtype=np.int32),
        "neighbor": np.asarray([g["neighbor"] for g in dual_face_groups], dtype=np.int32),
        "side": np.asarray([group_side(g) for g in dual_face_groups], dtype=object),
        "p0": np.vstack([g["p0"] for g in dual_face_groups]),
        "p1": np.vstack([g["p1"] for g in dual_face_groups]),
        "length": np.asarray([g["length"] for g in dual_face_groups]),
        "F_out": dual_group_flux[name].copy(),
    }


fracture_lcg = build_lcg_fracture_transport_grid()
fracture_mrst = build_mrst_fracture_transport_grid()
exchange_lcg = build_lcg_dual_exchange(fracture_lcg)
exchange_mrst = build_mrst_dual_exchange(fracture_mrst)

tracks = {}
for name in ("CG", "NLR", "PINN", "MRST"):
    fracture = fracture_mrst if name == "MRST" else fracture_lcg
    exchange = exchange_mrst if name == "MRST" else exchange_lcg
    tracks[name] = {
        "oriented": oriented_track(name), "fracture": fracture,
        "exchange": exchange,
        "S": np.zeros(len(problem.coords), dtype=float),
        "S_frac": np.zeros(len(fracture["length"]), dtype=float),
    }


def track_raw_cfl(track):
    oriented = track["oriented"]
    exchange = track["exchange"]
    fracture = track["fracture"]
    matrix_out = np.bincount(
        oriented["owner"], weights=np.maximum(oriented["F_out"], 0.0),
        minlength=len(dual_areas),
    )
    interior = oriented["neighbor"] >= 0
    matrix_out += np.bincount(
        oriented["neighbor"][interior],
        weights=np.maximum(-oriented["F_out"][interior], 0.0),
        minlength=len(dual_areas),
    )
    matrix_out += np.bincount(
        exchange["conn_cv_ids"], weights=np.abs(exchange["conn_E"]),
        minlength=len(dual_areas),
    )
    active_m = matrix_out > 1.0e-30
    dt_m = float(np.min(dual_areas[active_m] / (TRANSPORT_FPRIME_MAX * matrix_out[active_m])))
    frac_out = fracture_outflow_sum(fracture)
    frac_out += np.bincount(
        exchange["conn_frac_ids"], weights=np.abs(exchange["conn_E"]),
        minlength=len(frac_out),
    )
    active_f = frac_out > 1.0e-30
    dt_f = float(np.min(
        fracture["pore_volume"][active_f]
        / (TRANSPORT_FPRIME_MAX * frac_out[active_f])
    )) if np.any(active_f) else np.inf
    max_face_speed = float(np.max(
        np.abs(oriented["F_out"]) / np.maximum(oriented["length"], 1.0e-30)
    ))
    return {"dt_matrix_raw": dt_m, "dt_fracture_raw": dt_f,
            "max_dual_face_speed": max_face_speed, "dt_raw": min(dt_m, dt_f)}


cfl_by_track = {name: track_raw_cfl(track) for name, track in tracks.items()}
shared_dt_cfl = TRANSPORT_CFL * min(row["dt_raw"] for row in cfl_by_track.values())
# Use one fixed step for every track and align it with all requested output times.
# Since 0.10, 0.25, and 0.75 are integer multiples of 0.05, choosing an integer
# number of stable substeps per 0.05 interval gives exact snapshots without
# interpolation or a shortened per-snapshot step.
transport_snapshot_times = np.asarray([0.10, 0.25, 0.75], dtype=float)
transport_time_quantum = 0.05
steps_per_quantum = int(np.ceil(transport_time_quantum / shared_dt_cfl))
shared_dt = float(transport_time_quantum / steps_per_quantum)
snapshot_steps = np.rint(transport_snapshot_times / shared_dt).astype(np.int64)
if not np.allclose(snapshot_steps * shared_dt, transport_snapshot_times,
                   rtol=0.0, atol=5.0e-15):
    raise RuntimeError("requested transport snapshots do not lie on the shared time grid")
binding_cfl_track = min(cfl_by_track, key=lambda name: cfl_by_track[name]["dt_raw"])
global_max_face_speed = max(row["max_dual_face_speed"] for row in cfl_by_track.values())
print(
    f"shared transport dt={shared_dt:.8e}: one CFL choice over all four fields; "
    f"binding={binding_cfl_track}, max dual-face speed={global_max_face_speed:.8e}"
)
print("requested snapshots:", {
    float(T): int(step) for T, step in zip(transport_snapshot_times, snapshot_steps)
})
for name, row in cfl_by_track.items():
    print(" ", name, row)


def advance_track_one_step(track, dt):
    oriented = track["oriented"]
    owner, neighbor, F = oriented["owner"], oriented["neighbor"], oriented["F_out"]
    S, S_frac = track["S"], track["S_frac"]
    outflow = F >= 0.0
    interior_in = (~outflow) & (neighbor >= 0)
    boundary_in = (~outflow) & (neighbor < 0)
    up = np.empty_like(F)
    up[outflow] = S[owner[outflow]]
    up[interior_in] = S[neighbor[interior_in]]
    inlet = oriented["side"] == "right"
    up[boundary_in] = np.where(
        inlet[boundary_in], TRANSPORT_RIGHT_BC_S, TRANSPORT_DEFAULT_INFLOW_S
    )
    water_face_flux = fractional_flow_bl(up) * F
    matrix_update = np.bincount(owner, weights=water_face_flux, minlength=len(S))
    interior = neighbor >= 0
    matrix_update -= np.bincount(
        neighbor[interior], weights=water_face_flux[interior], minlength=len(S)
    )

    frac_update, _, _ = fracture_advection_update(S_frac, track["fracture"])
    exchange = track["exchange"]
    cv = exchange["conn_cv_ids"]
    fi = exchange["conn_frac_ids"]
    E = exchange["conn_E"]
    S_up = np.where(E > 0.0, S_frac[fi], S[cv])
    W = fractional_flow_bl(S_up) * E
    matrix_exchange = np.bincount(cv, weights=W, minlength=len(S))
    frac_exchange = np.bincount(fi, weights=-W, minlength=len(S_frac))

    S_new = S - dt * matrix_update / dual_areas + dt * matrix_exchange / dual_areas
    S_frac_new = (
        S_frac - dt * frac_update / track["fracture"]["pore_volume"]
        + dt * frac_exchange / track["fracture"]["pore_volume"]
    )
    track["S"] = np.clip(S_new, 0.0, 1.0)
    track["S_frac"] = np.clip(S_frac_new, 0.0, 1.0)


left_cv = np.isclose(problem.coords[:, 0], 0.0, atol=1.0e-14)
breakthrough_threshold = 1.0e-3
first_failed = None
snapshots = {}
step_to_snapshot = {
    int(step): float(T) for T, step in zip(transport_snapshot_times, snapshot_steps)
}
final_step = int(snapshot_steps[-1])
if final_step > TRANSPORT_MAX_STEPS:
    raise RuntimeError(
        f"final requested step {final_step} exceeds TRANSPORT_MAX_STEPS={TRANSPORT_MAX_STEPS}"
    )
for step in range(1, final_step + 1):
    for track in tracks.values():
        advance_track_one_step(track, shared_dt)
    left_max = {
        name: float(np.max(track["S"][left_cv])) for name, track in tracks.items()
    }
    if first_failed is None and any(
        value > breakthrough_threshold for value in left_max.values()
    ):
        first_failed = {"step": step, "time": step * shared_dt, "left_max": left_max}
    if step in step_to_snapshot:
        T_snapshot = step_to_snapshot[step]
        snapshots[T_snapshot] = {
            name: {"S": track["S"].copy(), "S_frac": track["S_frac"].copy()}
            for name, track in tracks.items()
        }
        print(f"captured shared transport snapshot T={T_snapshot:.2f} at step {step}")
if len(snapshots) != len(transport_snapshot_times):
    raise RuntimeError("one or more requested transport snapshots were not captured")

T_transport = float(transport_snapshot_times[-1])
binding_breakthrough_track = (
    max(first_failed["left_max"], key=first_failed["left_max"].get)
    if first_failed is not None else None
)
print(
    f"transport completed through T={T_transport:.2f} at shared step {final_step}; "
    f"first breakthrough-threshold crossing="
    f"{None if first_failed is None else first_failed['time']}; "
    f"binding track={binding_breakthrough_track}"
)

transport_metadata = {
    "cv_family": "65x65 vertex-centred median dual",
    "mrst_aggregation": (
        "exact summation of 128x128 matrix faces/cells; NNC exchange mapped "
        "directly by per-matrix-cell nnc_mat_cell"
    ),
    "time_grid": "every step of one shared fixed CFL dt",
    "shared_dt": shared_dt,
    "shared_dt_CFL_upper_bound": float(shared_dt_cfl),
    "time_quantum": transport_time_quantum,
    "steps_per_time_quantum": int(steps_per_quantum),
    "CFL": TRANSPORT_CFL,
    "FPRIME_MAX": TRANSPORT_FPRIME_MAX,
    "global_max_dual_face_speed": global_max_face_speed,
    "CFL_binding_track": binding_cfl_track,
    "CFL_by_track": cfl_by_track,
    "requested_snapshot_times": transport_snapshot_times.tolist(),
    "snapshot_steps": snapshot_steps.astype(int).tolist(),
    "snapshot_times_actual": (snapshot_steps * shared_dt).tolist(),
    "final_time": T_transport,
    "final_step": int(final_step),
    "breakthrough_rule": (
        "diagnostic only: first shared step at which any track has maximum "
        "left-boundary dual-CV saturation above 1e-3; transport continues "
        "through every requested snapshot"
    ),
    "threshold": breakthrough_threshold,
    "first_failed_step": first_failed,
    "binding_breakthrough_track": binding_breakthrough_track,
    "full_right_boundary_injection": True,
    "right_boundary_face_records": {
        name: int(np.count_nonzero(track["oriented"]["side"] == "right"))
        for name, track in tracks.items()
    },
}


def dual_field_grid(values):
    grid = np.full((65, 65), np.nan, dtype=float)
    for cid, (x, y) in enumerate(problem.coords):
        grid[int(round(64.0 * y)), int(round(64.0 * x))] = values[cid]
    return grid


names = ("CG", "NLR", "PINN", "MRST")
xv = np.linspace(0.0, 1.0, 65)
yv = np.linspace(0.0, 1.0, 65)
with plt.rc_context({"font.size": 12}):
    fig, axes = plt.subplots(
        len(transport_snapshot_times), 4, figsize=(15.2, 10.6),
        constrained_layout=True, squeeze=False,
    )
    pcm = None
    for row, T_snapshot in enumerate(transport_snapshot_times):
        for col, name in enumerate(names):
            ax = axes[row, col]
            pcm = ax.pcolormesh(
                xv, yv, dual_field_grid(snapshots[float(T_snapshot)][name]["S"]),
                shading="nearest", cmap="Blues", vmin=0.0, vmax=1.0,
                rasterized=True,
            )
            ax.plot([FRAC_A[0], FRAC_B[0]], [FRAC_A[1], FRAC_B[1]], "k-", lw=1.0)
            if row == 0: ax.set_title(name)
            ax.set_aspect("equal"); ax.set_xlabel(r"$x$")
            ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.grid(False)
            if col == 0:
                ax.set_ylabel(rf"$T={T_snapshot:.2f}$" + "\n" + r"$y$")
            else:
                ax.tick_params(labelleft=False)
    fig.colorbar(pcm, ax=axes, fraction=0.018, pad=0.02, label=r"$S_w$")
    fig.savefig(OUTDIR / "fig_km_saturation_4panel.png", dpi=300, bbox_inches="tight")
    plt.show()

diff_names = ("CG", "NLR", "PINN")
diffs = {
    float(T): {
        name: snapshots[float(T)][name]["S"] - snapshots[float(T)]["MRST"]["S"]
        for name in diff_names
    }
    for T in transport_snapshot_times
}
diff_limits = {
    float(T): max(float(np.percentile(np.concatenate([
        np.abs(diffs[float(T)][name]) for name in diff_names
    ]), 99.5)), 1.0e-12)
    for T in transport_snapshot_times
}
transport_metadata["saturation_difference_color_limit"] = {
    "by_snapshot": {f"{float(T):.2f}": diff_limits[float(T)]
                    for T in transport_snapshot_times},
    "policy": (
        "one symmetric 99.5th-percentile limit per time, shared by CG/NLR/PINN; "
        "larger values saturate"
    ),
}
with plt.rc_context({"font.size": 12}):
    fig, axes = plt.subplots(
        len(transport_snapshot_times), 3, figsize=(12.4, 10.6),
        constrained_layout=True, squeeze=False,
    )
    for row, T_snapshot in enumerate(transport_snapshot_times):
        diff_lim = diff_limits[float(T_snapshot)]
        pcm = None
        for col, name in enumerate(diff_names):
            ax = axes[row, col]
            pcm = ax.pcolormesh(
                xv, yv, dual_field_grid(diffs[float(T_snapshot)][name]),
                shading="nearest", cmap="RdBu_r", vmin=-diff_lim,
                vmax=diff_lim, rasterized=True,
            )
            ax.plot([FRAC_A[0], FRAC_B[0]], [FRAC_A[1], FRAC_B[1]], "k-", lw=1.0)
            if row == 0: ax.set_title(f"{name} $-$ MRST")
            ax.set_aspect("equal"); ax.set_xlabel(r"$x$")
            ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.grid(False)
            if col == 0:
                ax.set_ylabel(rf"$T={T_snapshot:.2f}$" + "\n" + r"$y$")
            else:
                ax.tick_params(labelleft=False)
        fig.colorbar(pcm, ax=axes[row, :], fraction=0.024, pad=0.02,
                     label=r"$\Delta S_w$")
    fig.savefig(OUTDIR / "fig_km_saturation_diff.png", dpi=300, bbox_inches="tight")
    plt.show()

styles = {
    "CG": ("C0", "-"), "NLR": ("C1", "--"),
    "PINN": ("C2", "-."), "MRST": ("C3", ":"),
}
with plt.rc_context({"font.size": 13}):
    fig, axes = plt.subplots(1, len(transport_snapshot_times), figsize=(14.4, 4.2),
                             constrained_layout=True, sharey=True)
    for ax, T_snapshot in zip(axes, transport_snapshot_times):
        for name in names:
            color, linestyle = styles[name]
            fracture = tracks[name]["fracture"]
            order = np.argsort(fracture["s_centers"])
            ax.plot(
                np.asarray(fracture["s_centers"])[order],
                snapshots[float(T_snapshot)][name]["S_frac"][order],
                color=color, ls=linestyle, lw=1.8, label=name,
            )
        ax.set_title(rf"$T={T_snapshot:.2f}$")
        ax.set_xlim(0.0, L_gamma); ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"fracture arc length $s$"); ax.grid(False)
    axes[0].set_ylabel(r"$S_{w,\Gamma}$")
    axes[-1].legend(frameon=False)
    fig.savefig(OUTDIR / "fig_km_sgamma_profiles.png", dpi=600, bbox_inches="tight")
    plt.show()

mrst_E_by_frac = np.bincount(
    exchange_mrst["conn_frac_ids"], weights=exchange_mrst["conn_E"],
    minlength=len(fracture_mrst["length"]),
)
mrst_E_density = mrst_E_by_frac / fracture_mrst["length"]
s_lambda = np.linspace(0.0, L_gamma, 1600)
with plt.rc_context({"font.size": 13}):
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.axhline(0.0, color="0.6", lw=0.7, ls=":")
    ax.plot(s_lambda, lambda_h_on_s(s_lambda), color="C0", lw=2.0,
            label=r"FEM ($\lambda_h$)")
    order = np.argsort(fracture_mrst["s_centers"])
    ax.plot(np.asarray(fracture_mrst["s_centers"])[order], mrst_E_density[order],
            color="C3", lw=1.0, marker=".", ms=3,
            label="MRST/EDFM (NNC)")
    ax.set_xlim(0.0, L_gamma); ax.set_xlabel(r"arc length $s$")
    ax.set_ylabel(r"exchange density $E(s)$"); ax.legend(frameon=False); ax.grid(False)
    fig.savefig(OUTDIR / "fig_km_exchange.png", dpi=600, bbox_inches="tight")
    plt.show()

np.savez_compressed(
    OUTDIR / "km_case1_transport.npz",
    names=np.asarray(names), T=transport_snapshot_times, dt=shared_dt,
    snapshot_steps=snapshot_steps,
    S_matrix=np.stack([
        np.vstack([snapshots[float(T)][name]["S"] for name in names])
        for T in transport_snapshot_times
    ]),
    S_matrix_final=np.vstack([tracks[name]["S"] for name in names]),
    S_frac_CG=np.stack([
        snapshots[float(T)]["CG"]["S_frac"] for T in transport_snapshot_times
    ]),
    S_frac_NLR=np.stack([
        snapshots[float(T)]["NLR"]["S_frac"] for T in transport_snapshot_times
    ]),
    S_frac_PINN=np.stack([
        snapshots[float(T)]["PINN"]["S_frac"] for T in transport_snapshot_times
    ]),
    S_frac_MRST=np.stack([
        snapshots[float(T)]["MRST"]["S_frac"] for T in transport_snapshot_times
    ]),
    s_frac_LCG=fracture_lcg["s_centers"], s_frac_MRST=fracture_mrst["s_centers"],
    dual_coords=problem.coords, dual_areas=dual_areas,
)
conservation_report["MRST_aggregation"] = mrst_aggregation_gate
conservation_report["MRST_validation"] = mrst_validation
conservation_report["transport"] = transport_metadata
(OUTDIR / "km_case1_conservation.json").write_text(
    json.dumps(json_clean(conservation_report), indent=2, allow_nan=False) + "\n"
)
for filename in (
    "fig_km_saturation_4panel.png", "fig_km_saturation_diff.png",
    "fig_km_sgamma_profiles.png", "fig_km_exchange.png",
    "km_case1_transport.npz", "km_case1_conservation.json",
):
    print("saved:", OUTDIR / filename)
'''

transport_validations = r'''
# ============================================================
# Stage 7a: dt-halving sanity check on the PINN track
# ============================================================
def fresh_track(template):
    return {
        "oriented": template["oriented"],
        "fracture": template["fracture"],
        "exchange": template["exchange"],
        "S": np.zeros_like(template["S"]),
        "S_frac": np.zeros_like(template["S_frac"]),
    }


def advance_track_to_times(template, requested_times, nominal_dt):
    track = fresh_track(template)
    states = {}
    current_time = 0.0
    for target_time in np.asarray(requested_times, dtype=float):
        if target_time < current_time - 1.0e-14:
            raise ValueError("transport validation times must be increasing")
        while current_time + nominal_dt < target_time - 5.0e-15:
            advance_track_one_step(track, nominal_dt)
            current_time += nominal_dt
        remainder = target_time - current_time
        if remainder > 5.0e-15:
            advance_track_one_step(track, remainder)
            current_time = float(target_time)
        states[float(target_time)] = {
            "S": track["S"].copy(), "S_frac": track["S_frac"].copy()
        }
    return states


first_snapshot = float(transport_snapshot_times[0])
pinn_half_states = advance_track_to_times(
    tracks["PINN"], [first_snapshot], 0.5 * shared_dt
)
pinn_dt = snapshots[first_snapshot]["PINN"]["S"]
pinn_dt2 = pinn_half_states[first_snapshot]["S"]
dt_half_difference = pinn_dt2 - pinn_dt
dt_half_rmse = float(np.sqrt(np.mean(dt_half_difference ** 2)))
dt_half_max = float(np.max(np.abs(dt_half_difference)))
between_track_rmse = {
    name: float(np.sqrt(np.mean(
        (pinn_dt - snapshots[first_snapshot][name]["S"]) ** 2
    )))
    for name in ("CG", "NLR", "MRST")
}
between_scale = min(value for value in between_track_rmse.values() if value > 1.0e-15)
dt_half_ratio = dt_half_rmse / between_scale
dt_halving_validation = {
    "passed": bool(dt_half_ratio <= 0.15),
    "track": "PINN",
    "snapshot_time": first_snapshot,
    "dt": shared_dt,
    "dt_half": 0.5 * shared_dt,
    "saturation_rmse": dt_half_rmse,
    "saturation_max_abs": dt_half_max,
    "between_track_rmse_at_same_time": between_track_rmse,
    "ratio_to_smallest_between_track_rmse": dt_half_ratio,
    "acceptance": "dt/2 RMSE <= 15% of the smallest PINN-to-other-track RMSE",
}
print("dt-halving validation:", json.dumps(json_clean(dt_halving_validation), indent=2))
if not dt_halving_validation["passed"]:
    raise RuntimeError("PINN dt-halving sanity check failed")


# ============================================================
# Stage 7b: end-to-end validation against native MRST snapshots
# ============================================================
mrst_snapshot_times = np.asarray(mrst["snap_T_abs"], dtype=float).reshape(-1)
mrst_native_snapshots = np.asarray(mrst["sw_matrix_snaps"], dtype=float)
if mrst_native_snapshots.shape == (len(mrst_snapshot_times), len(mrst_cell_to_cv)):
    mrst_native_snapshots = mrst_native_snapshots.T
if mrst_native_snapshots.shape != (len(mrst_cell_to_cv), len(mrst_snapshot_times)):
    raise RuntimeError(
        "sw_matrix_snaps shape is inconsistent with the MRST matrix grid and snap_T_abs"
    )
mrst_native_on_dual = np.column_stack([
    np.bincount(
        mrst_cell_to_cv, weights=mrst_native_snapshots[:, j],
        minlength=len(problem.coords),
    ) / _mrst_per_cv
    for j in range(len(mrst_snapshot_times))
])
mrst_aggregated_states = advance_track_to_times(
    tracks["MRST"], mrst_snapshot_times, shared_dt
)
mrst_aggregated_snapshots = np.column_stack([
    mrst_aggregated_states[float(T)]["S"] for T in mrst_snapshot_times
])
mrst_snapshot_rows = []
for j, T in enumerate(mrst_snapshot_times):
    reference = mrst_native_on_dual[:, j]
    difference = mrst_aggregated_snapshots[:, j] - reference
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    mrst_snapshot_rows.append({
        "time": float(T),
        "rmse": rmse,
        "mean_abs": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "relative_rmse": float(
            rmse / max(np.sqrt(np.mean(reference ** 2)), 1.0e-30)
        ),
        "mean_saturation_aggregated_track": float(np.average(
            mrst_aggregated_snapshots[:, j], weights=dual_areas
        )),
        "mean_saturation_native_aggregated": float(np.average(
            reference, weights=dual_areas
        )),
    })
mrst_snapshot_max_rmse = max(row["rmse"] for row in mrst_snapshot_rows)
mrst_snapshot_validation = {
    "passed": bool(mrst_snapshot_max_rmse <= 0.12),
    "source": MRST_CASE3_PATH.name,
    "native_grid": "128x128 MRST matrix cells",
    "comparison_grid": "65x65 vertex-centred dual; exact 1/2/4-cell volume averages",
    "aggregated_track_time_step": shared_dt,
    "snapshot_count": int(len(mrst_snapshot_times)),
    "max_snapshot_rmse": mrst_snapshot_max_rmse,
    "acceptance": "max saturation RMSE <= 0.12 (grid-family comparison gate)",
    "rows": mrst_snapshot_rows,
}
print(
    "native-MRST snapshot validation:",
    json.dumps(json_clean(mrst_snapshot_validation), indent=2),
)
if not mrst_snapshot_validation["passed"]:
    raise RuntimeError("aggregated-MRST end-to-end snapshot validation failed")


# Report the actual physical-boundary fluxes used by transport. The PINN row is
# expected at roundoff; left/right are intentionally not included in this gate.
horizontal_boundary_transport_flux = {}
for name in names:
    oriented = tracks[name]["oriented"]
    mask = (oriented["side"] == "bottom") | (oriented["side"] == "top")
    values = oriented["F_out"][mask]
    horizontal_boundary_transport_flux[name] = {
        "n": int(np.count_nonzero(mask)),
        "mean_abs": float(np.mean(np.abs(values))),
        "max_abs": float(np.max(np.abs(values))),
        "net": float(np.sum(values)),
    }
if horizontal_boundary_transport_flux["PINN"]["max_abs"] > 2.0e-12:
    raise RuntimeError("PINN hard no-flow boundary flux is above roundoff tolerance")

transport_metadata["dt_halving_validation"] = dt_halving_validation
transport_metadata["MRST_native_snapshot_validation"] = mrst_snapshot_validation
transport_metadata["horizontal_boundary_flux"] = horizontal_boundary_transport_flux
transport_metadata["method_boundary_conditions"] = {
    "PINN": "exact top/bottom no-flow by prescribed streamline trace",
    "CG": "weak natural no-flow boundary condition",
    "NLR": "inherits/reconstructs CG data without a hard boundary trace",
    "MRST": "native finite-volume no-flow boundary",
    "left_right": "unconstrained flux for every reconstruction; pressure Dirichlet",
}
conservation_report["transport"] = transport_metadata

# Rewrite the archive once, now including both validation data sets.
np.savez_compressed(
    OUTDIR / "km_case1_transport.npz",
    names=np.asarray(names), T=transport_snapshot_times, dt=shared_dt,
    snapshot_steps=snapshot_steps,
    S_matrix=np.stack([
        np.vstack([snapshots[float(T)][name]["S"] for name in names])
        for T in transport_snapshot_times
    ]),
    S_matrix_final=np.vstack([tracks[name]["S"] for name in names]),
    S_frac_CG=np.stack([
        snapshots[float(T)]["CG"]["S_frac"] for T in transport_snapshot_times
    ]),
    S_frac_NLR=np.stack([
        snapshots[float(T)]["NLR"]["S_frac"] for T in transport_snapshot_times
    ]),
    S_frac_PINN=np.stack([
        snapshots[float(T)]["PINN"]["S_frac"] for T in transport_snapshot_times
    ]),
    S_frac_MRST=np.stack([
        snapshots[float(T)]["MRST"]["S_frac"] for T in transport_snapshot_times
    ]),
    s_frac_LCG=fracture_lcg["s_centers"], s_frac_MRST=fracture_mrst["s_centers"],
    dual_coords=problem.coords, dual_areas=dual_areas,
    S_PINN_dt_half_first=pinn_dt2,
    mrst_validation_times=mrst_snapshot_times,
    S_MRST_aggregated_track_snaps=mrst_aggregated_snapshots,
    S_MRST_native_aggregated_snaps=mrst_native_on_dual,
)
(OUTDIR / "km_case1_conservation.json").write_text(
    json.dumps(json_clean(conservation_report), indent=2, allow_nan=False) + "\n"
)
print("horizontal-boundary transport flux:", json.dumps(
    json_clean(horizontal_boundary_transport_flux), indent=2
))
print("saved validated transport archive and JSON")
'''

cells = [
    markdown(r'''# Köppel–Martin Case 1: conservative stream-function reconstruction and NLR

This notebook uses the sealed-tip nonconforming CG–LMDFM solve, one Option-A
stream-function network, batched Deng NLR, the corrected fracture-aware audit,
and a common vertex-centred dual transport grid for CG, NLR, PINN, and MRST/EDFM.

The PINN loss contains only integrated dual-face and fixed-grid pointwise CG
data; horizontal physical-boundary faces are excluded because they conflict
with the prescribed streamline trace. Conservation, the multiplier jump, and
top/bottom no-flow for the PINN are imposed by the field architecture, with no
penalty terms. CG and NLR retain their weak/discrete boundary behavior, while
left/right fluxes remain unconstrained for every reconstruction.'''),
    code(setup),
    markdown("## Stage 1 — sealed-tip CG–LMDFM solve\n"),
    source_nb["cells"][2],
    source_nb["cells"][3],
    source_nb["cells"][4],
    source_nb["cells"][5],
    markdown("## CG–LMDFM solution\n"),
    code(cg_figure),
    markdown("## Shared rectangular geometry for the Deng dual grid\n"),
    nlr_nb["cells"][7],
    markdown("## Mandatory geometric, sealed-tip, A0, and A1 gates\n"),
    code(adapter_and_gates),
    markdown(r'''## Stage 2 — Option-A canonical training

The trainable potential is multiplied by $y(1-y)$ and added to fixed bottom
and top traces obtained from the analytic cumulative normal flux of the
particular line-source field. Hence each horizontal-boundary subsegment has
zero integrated normal flux by endpoint telescoping. No condition is imposed
on the left or right boundary.'''),
    code(training),
    markdown("## Stage 3 — Deng nonlinear local reconstruction\n"),
    code(nlr_solve),
    markdown("## Stage 4 — corrected dual and rotated-volume diagnostics\n"),
    code(zeta_setup),
    code(diagnostics),
    markdown(r'''## Stage 5 — MRST reference at twice the matrix resolution

The FEM pressure solve uses a 64×64 primal grid and a 65×65 vertex-centred
median dual. The MRST/EDFM reference remains on 128×128 matrix cells. Each
dual volume is an exact union of MRST cells (2×2 interior, 1×2 on an edge, and
1×1 at a corner), and each unsplit dual-face piece is exactly one MRST face.
The following cell checks these index identities before summing any flux.

The export provides signed matrix-face fluxes with orientations and NNC
exchange per host matrix cell (`nnc_mat_cell`). Thus both face flux and
matrix–fracture exchange are aggregated by exact addition, without
interpolation.'''),
    code(mrst_aggregation),
    markdown(r'''## Stage 6 — synchronized four-track transport

CG, NLR, PINN, and aggregated MRST are advanced on the same 65×65
vertex-centred dual grid. A single fixed time step is selected once from the
most restrictive CFL bound over all four matrix-face fields, exchange terms,
and fracture conduits. Every track is then advanced in the same loop on that
shared time grid. Water saturation equals one on the full right boundary;
both fracture tips have zero throughflow. The PINN additionally satisfies
top/bottom no-flow exactly through its streamline trace; CG and NLR retain the
weak/discrete natural-boundary traces inherited from the pressure solve.

Snapshots are recorded at exactly $T=0.10$, $0.25$, and $0.75$. The fixed CFL
step is aligned to their common time quantum, so no temporal interpolation or
track-specific final step is used. The first left-boundary saturation above
$10^{-3}$ is retained as a breakthrough diagnostic but does not stop the run.'''),
    code(transport),
    markdown(r'''## Stage 7 — temporal and MRST end-to-end validation

The first cell repeats the PINN transport to the first displayed snapshot with
$dt/2$ and compares that change with the simultaneous between-track differences.
The second comparison advances the aggregated-MRST flux/exchange track to every
native `snap_T_abs` time and compares it with exact volume averages of
`sw_matrix_snaps` on the 65×65 dual grid. These are independent of the static
geometry and face-flux aggregation gates.'''),
    code(transport_validations),
]

notebook = {
    "cells": cells,
    "metadata": source_nb.get("metadata", {}),
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print("wrote", TARGET)
