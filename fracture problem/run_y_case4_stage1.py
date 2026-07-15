"""Session-1 driver for the SPE10/Y-fracture study.

The driver is deliberately resumable: the expensive canonical PINN training is
stored separately from the deterministic geometry/CG/NLR rebuild.  Its stopping
point is the agreed handoff: all gates pass, the PINN is trained, and one fully
coupled transport step succeeds for all three flux tracks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import fracture_hardcurl_common as hc
import y_fracture_case4 as yc


HERE = Path(__file__).resolve().parent
OUT = HERE / "result_y_fracture_spe10_case4"
OUT.mkdir(exist_ok=True)
CHECKPOINT = OUT / "y_case4_pinn_initial.pt"
SUMMARY = OUT / "y_case4_stage1_summary.json"
ARRAYS = OUT / "y_case4_stage1_arrays.npz"

SEED = 20250308
FREQUENCIES = (1, 2, 4, 8)
WIDTH = 32
DEPTH = 3
ADAM_STEPS = 2000
LBFGS_STEPS = 3000
STAGNATION_RTOL = 1.0e-6
STAGNATION_WINDOW = 200


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def require(label, passed):
    print(f"{label}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise RuntimeError(f"{label} failed")


config = yc.load_case1_configuration()
geom = yc.build_y_geometry(config)
placement = yc.placement_check(config, geom)
require("Y placement", placement["passed"])

system = yc.YLCGSystem(config, geom)
solution = system.solve()
fracture_gate = system.fracture_gate(solution)
require("fracture two-point/Junction Proposition", fracture_gate["passed"])
exchange = yc.exchange_integrals(system, solution)
require("sealed-network exchange", exchange["passed"])
a0 = yc.gate_a0_interior_tip()
require("Gate A0 interior tip", a0["passed"])

built = yc.build_hardcurl_problem(system, solution)
a1 = yc.gate_a1_multibranch(built)
require("Gate A1 multi-branch hard-curl identity", a1["passed"])

nlr = yc.nlr_reconstruct(system, solution, built)
nlr_limits = {
    "interior": 1.0e-12, "single-cut": 1.0e-11,
    "multi-cut": 1.0e-11, "source": 1.0e-12,
}
nlr_passed = all(
    nlr["audit"]["stats"][name]["max_abs"] <= limit
    for name, limit in nlr_limits.items()
)
require("NLR class-scoped conservation", nlr_passed)

accepted_lbfgs = []


def accepted_step_logger(iteration, model, loss_fn):
    loss = float(loss_fn().detach().cpu())
    accepted_lbfgs.append({"iteration": int(iteration), "loss": loss})
    if iteration == 0 or iteration % 50 == 0:
        print(f"accepted L-BFGS {iteration:4d}: loss={loss:.8e}")
    if len(accepted_lbfgs) <= STAGNATION_WINDOW:
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


if CHECKPOINT.exists():
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    signature = (saved.get("width"), saved.get("depth"), tuple(saved.get("frequencies", ())),
                 saved.get("seed"), saved.get("face_weight"), saved.get("pointwise_weight"))
    expected = (WIDTH, DEPTH, FREQUENCIES, SEED, 1.0, 1.0)
    if signature != expected:
        raise RuntimeError(f"stale PINN checkpoint signature {signature}, expected {expected}")
    model = yc.LogKFourierPsiNet(config, FREQUENCIES, WIDTH, DEPTH).to(dtype=yc.TORCH_DTYPE)
    model.load_state_dict(saved["state_dict"])
    option_a = {
        "model": model, "face_flux": np.asarray(saved["face_flux"]),
        "history": saved["history"], "optimization": saved["optimization"],
        "wall_time": saved["wall_time"], "parameters": saved["parameters"],
    }
    accepted_lbfgs = saved.get("accepted_lbfgs", [])
    print("loaded canonical initial PINN checkpoint:", CHECKPOINT)
else:
    option_a = hc.run_option_a(
        built.dual_problem,
        adam_steps=ADAM_STEPS, lbfgs_steps=LBFGS_STEPS,
        width=WIDTH, depth=DEPTH, lr=2.0e-3,
        frequencies=FREQUENCIES, seed=SEED,
        face_weight=1.0, pointwise_weight=1.0, potential_weight=0.0,
        target_mode="cg", particular_lambda_mode="h",
        lbfgs_iteration_callback=accepted_step_logger,
        model_factory=yc.model_factory(config),
    )
    torch.save({
        "state_dict": option_a["model"].state_dict(),
        "face_flux": option_a["face_flux"],
        "history": option_a["history"],
        "optimization": option_a["optimization"],
        "wall_time": option_a["wall_time"],
        "parameters": option_a["parameters"],
        "accepted_lbfgs": accepted_lbfgs,
        "width": WIDTH, "depth": DEPTH, "frequencies": FREQUENCIES,
        "seed": SEED, "face_weight": 1.0, "pointwise_weight": 1.0,
        "potential_weight": 0.0,
    }, CHECKPOINT)
    print("saved canonical initial PINN checkpoint:", CHECKPOINT)

pinn_audit = yc.audit_flux(built, option_a["face_flux"], "PINN conservation audit")
pinn_passed = all(
    row["max_abs"] is None or row["max_abs"] <= 1.0e-12
    for row in pinn_audit["stats"].values()
)
require("PINN all-class conservation", pinn_passed)

# Build and certify the production PoU head, then execute exactly one coupled
# pressure/flux/transport refresh for each track.
pou = yc.PoUHead(built, option_a["model"], window_shape=(16, 16), r=16)
pou_fit = pou.fit(built.dual_problem.cg_face, built=built, ridge_rel=1.0e-8)
pou_audit = yc.audit_flux(built, pou_fit["prediction"], "PINN-PoU conservation audit")
require(
    "PINN-PoU all-class conservation",
    all(row["max_abs"] is None or row["max_abs"] <= 1.0e-12
        for row in pou_audit["stats"].values()),
)

connections = yc.build_exchange_connections(system, built)
require("matrix-fracture exchange mapping", connections["max_cv_mismatch"] <= 1.0e-13)

S0 = np.zeros(system.n_m)
Sf0 = np.zeros(system.n_f)
injector_nodes = np.unique(
    system.cell_nodes[np.flatnonzero(config.source_rate_cell > 0)].ravel()
)
S0[injector_nodes] = 1.0

track_flux = {
    "CG": built.dual_problem.cg_face,
    "NLR": nlr["face_flux"],
    "PINN": pou_fit["prediction"],
}
dt_candidates = {
    name: yc.initial_cfl_dt(system, built, flux, solution, connections)
    for name, flux in track_flux.items()
}
dt = min(dt_candidates.values())
one_step = {}
for name, flux in track_flux.items():
    S1, Sf1, report = yc.one_transport_step(
        system, built, flux, solution, S0.copy(), Sf0.copy(), dt,
        connections=connections,
    )
    finite = np.all(np.isfinite(S1)) and np.all(np.isfinite(Sf1))
    require(f"one coupled transport step ({name})", finite)
    one_step[name] = {
        "report": report, "S_min": float(S1.min()), "S_max": float(S1.max()),
        "S_gamma_min": float(Sf1.min()), "S_gamma_max": float(Sf1.max()),
    }

summary = {
    "stage": "session-1 handoff",
    "configuration_source_notebook": str(config.source_notebook),
    "mrst_export": str(config.data_file),
    "placement": placement,
    "geometry": {
        "junction": geom.junction, "tips": geom.tips,
        "branch_lengths": [b.length for b in geom.branches],
        "n_gamma_per_branch": [b.n_gamma for b in geom.branches],
        "n_lambda_per_branch": [b.n_lambda for b in geom.branches],
        "K_gamma": geom.k_gamma,
    },
    "linear_solve": {
        "solve_s": solution["solve_s"],
        "constraint_max": solution["constraint_max"],
        "residual_before_refinement": solution["linear_residual_before_refinement"],
        "residual_after_refinement": solution["linear_residual_after_refinement"],
    },
    "gates": {"fracture": fracture_gate, "sealed_exchange": exchange,
              "A0": a0, "A1": a1},
    "cv_class_counts": {name: int(np.count_nonzero(built.cv_class == name))
                        for name in yc.CLASS_NAMES},
    "conservation": {
        "NLR": nlr["audit"]["stats"],
        "PINN": pinn_audit["stats"],
        "PINN_PoU": pou_audit["stats"],
    },
    "training": {
        "wall_time": option_a["wall_time"], "parameters": option_a["parameters"],
        "optimization": option_a["optimization"],
        "accepted_lbfgs": accepted_lbfgs,
        "loss": {"face_weight": 1.0, "pointwise_weight": 1.0,
                 "potential_weight": 0.0},
    },
    "pou": {"window_shape": [16, 16], "rank": 16,
            "build_s": pou.build_s, **{k: v for k, v in pou_fit.items()
                                      if k not in ("prediction", "theta")}},
    "one_step": {"shared_dt": dt, "dt_candidates": dt_candidates,
                 "tracks": one_step},
    "machine": yc.machine_info(),
}
SUMMARY.write_text(json.dumps(jsonable(summary), indent=2) + "\n")

np.savez_compressed(
    ARRAYS,
    lambda_h=solution["lambda"], p_matrix=solution["p_m"],
    p_gamma=solution["p_f"], cg_face=track_flux["CG"],
    nlr_face=track_flux["NLR"], pinn_face=option_a["face_flux"],
    pinn_pou_face=track_flux["PINN"], pou_theta=pou.theta,
)
print("saved:", SUMMARY)
print("saved:", ARRAYS)
print("SESSION-1 HANDOFF READY: all gates pass, PINN trained, one coupled step runs")
