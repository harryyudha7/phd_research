"""Finalize the Y-fracture Case 4 production run from its resume checkpoint."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import y_fracture_case4 as yc


HERE = Path(__file__).resolve().parent
OUT = HERE / "result_y_fracture_spe10_case4"
PINN_CKPT = OUT / "y_case4_pinn_initial.pt"
RUN_CKPT = OUT / "y_case4_production_resume.pt"
TRANSPORT_NPZ = OUT / "y_case4_transport.npz"
TRANSPORT_JSON = OUT / "y_case4_transport.json"

LOG_EVERY = 10
BT_LEVEL = 1.0e-3
HALF_LEVEL = 0.5
CFL_SAFETY = 0.45


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


def nofracture_view(template: yc.BuiltProblem, solution: dict, cg_face: np.ndarray) -> yc.BuiltProblem:
    p0 = template.dual_problem
    problem = SimpleNamespace(**p0.__dict__)
    zcv = np.zeros_like(p0.cv_lambda_h)
    zface = np.zeros_like(p0.qpl_h_face)
    problem.ctx = solution
    problem.cv_lambda_h = zcv
    problem.cv_lambda_exact = zcv.copy()
    problem.qpl_h_face = zface
    problem.qpl_exact_face = zface.copy()
    problem.cg_face = cg_face
    problem.lambda_h_density = np.zeros_like(p0.lambda_h_density)
    problem.lambda_h_nodal_density = np.zeros_like(p0.lambda_h_nodal_density)
    return yc.BuiltProblem(
        problem,
        template.dual,
        template.cv_class,
        template.cv_cut_count,
        template.source_mask,
        template.boundary_mask,
        {},
        template.qpf,
    )


def stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "n": int(len(a)),
    }


def watercut_milestones(curves, track_names, plateau_fraction=0.2):
    steps = np.asarray(curves["step"], dtype=int)
    times = np.asarray(curves["time"], dtype=float)
    nwin = max(10, int(np.ceil(plateau_fraction * len(steps))))
    out = {
        "definition": "plateau = mean over final 20% of logged water-cut values; t90 = first logged time at 90% of that plateau",
        "plateau_window_fraction": float(plateau_fraction),
        "plateau_window_n_logged": int(nwin),
    }
    for name in track_names:
        wc = np.asarray(curves[f"wc_{name}"], dtype=float)
        plateau = float(np.mean(wc[-nwin:]))
        target = 0.9 * plateau
        idx = np.flatnonzero(wc >= target)
        out[name] = {
            "plateau_value": plateau,
            "t90_plateau_level": float(target),
            "t90_step": None if len(idx) == 0 else int(steps[idx[0]]),
            "t90_time": None if len(idx) == 0 else float(times[idx[0]]),
            "final_logged_watercut": float(wc[-1]),
        }
    return out


def main():
    if not RUN_CKPT.exists():
        raise FileNotFoundError(RUN_CKPT)

    config = yc.load_case1_configuration()
    geom = yc.build_y_geometry(config)
    system = yc.YLCGSystem(config, geom)
    solution0 = system.solve()
    built0 = yc.build_hardcurl_problem(system, solution0)
    dynamic = yc.YDynamicGeometry(system, built0)
    fast_nlr = yc.FastNLR(system, dynamic)

    saved = torch.load(PINN_CKPT, map_location="cpu", weights_only=False)
    model = yc.LogKFourierPsiNet(
        config,
        tuple(saved["frequencies"]),
        int(saved["width"]),
        int(saved["depth"]),
    ).to(dtype=yc.TORCH_DTYPE)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    pou = yc.PoUHead(built0, model, window_shape=(16, 16), r=16)
    pou.factorize(1.0e-8)

    state = torch.load(RUN_CKPT, map_location="cpu", weights_only=False)
    step = int(state["step"])
    tracks = state["tracks"]
    curves = state["curves"]
    events = state["events"]
    snapshots = state["snapshots"]
    timing = state["timing"]
    pou.theta = np.asarray(state["pou_theta"])

    # Recompute the shared initial dt exactly as in the production runner.
    initial_conn = dynamic.connections(solution0)
    initial_flux = {
        "CG": dynamic.cg_face(solution0),
        "NLR": fast_nlr.reconstruct(solution0)["face_flux"],
    }
    view0 = dynamic.step_view(solution0, initial_flux["CG"])
    theta_before_dt = pou.theta.copy()
    initial_flux["PINN"] = pou.fit(initial_flux["CG"], built=view0)["prediction"]
    pou.theta = theta_before_dt
    track_names = ("CG", "NLR", "PINN")
    dt_candidates = {
        n: yc.initial_cfl_dt(system, built0, initial_flux[n], solution0, initial_conn, cfl=CFL_SAFETY)
        for n in track_names
    }
    dt = min(dt_candidates.values())

    events["stop_reason"] = "manual_stop_unbalanced_well_watercut_ceiling"
    events["stop_note"] = (
        "Original all-tracks water-cut >= 0.5 criterion is unattainable for the inherited "
        "unbalanced wells: injection rate +1, producer rate -5, theoretical ceiling about q_inj/|q_prod| = 0.2."
    )
    q_inj = float(np.sum(config.source_rate_cell[config.source_rate_cell > 0]))
    q_prod = float(-np.sum(config.source_rate_cell[config.source_rate_cell < 0]))
    events["watercut_ceiling_estimate"] = None if q_prod == 0.0 else q_inj / q_prod

    snapshots["stop"] = {
        "step": step,
        "time": step * dt,
        **{f"S_{n}": tracks[n]["S"].copy() for n in track_names},
        **{f"Sf_{n}": tracks[n]["Sf"].copy() for n in track_names},
    }

    # No-fracture PINN twin at the same dt and the same event steps.
    inj = np.unique(system.cell_nodes[np.flatnonzero(config.source_rate_cell > 0)].ravel())
    nf = yc.NoFractureQ1System(system)
    Snf = np.zeros(system.n_m)
    Snf[inj] = 1.0
    nf0 = nf.solve(yc.mobility_factor(yc.project_dual_to_cells(system, Snf)))
    cg_nf0 = dynamic.cg_face(nf0)
    view_nf0 = nofracture_view(built0, nf0, cg_nf0)
    pou_nf = yc.PoUHead(view_nf0, model, window_shape=(16, 16), r=16)
    pou_nf.factorize(1.0e-8)
    event_steps = {}
    for label, row in snapshots.items():
        event_steps.setdefault(int(row["step"]), []).append(label)
    nf_snapshots = {}
    nf_curve_step, nf_curve_time, nf_curve_wc = [], [], []
    t0 = time.perf_counter()
    for k in range(1, step + 1):
        mob = yc.mobility_factor(yc.project_dual_to_cells(system, Snf))
        sol = nf.solve(mob)
        cg = dynamic.cg_face(sol)
        view = nofracture_view(built0, sol, cg)
        flux = pou_nf.fit(cg, built=view, anchor=pou_nf.theta, ridge_rel=1.0e-8)["prediction"]
        Snf, rep = yc.one_matrix_transport_step(system, built0, flux, Snf, dt)
        if k % LOG_EVERY == 0 or k == 1:
            nf_curve_step.append(k)
            nf_curve_time.append(k * dt)
            nf_curve_wc.append(rep["producer_watercut"])
        if k in event_steps:
            for label in event_steps[k]:
                nf_snapshots[label] = Snf.copy()
        if k == 1 or k % 1000 == 0:
            print("twin step", k, "of", step, "wall", time.perf_counter() - t0, flush=True)

    events["final_step"] = step
    events["final_time"] = step * dt
    events["breakthrough_time"] = {
        n: (None if v is None else v * dt) for n, v in events["breakthrough_step"].items()
    }
    events["half_rise_time"] = {
        n: (None if v is None else v * dt) for n, v in events["half_rise_step"].items()
    }
    events["tracks_below_half_at_stop"] = [n for n, v in events["half_rise_step"].items() if v is None]
    events["posthoc_watercut_milestones"] = watercut_milestones(curves, track_names)

    arrays = {
        "dt": np.array(dt),
        "log_step": np.asarray(curves["step"]),
        "log_time": np.asarray(curves["time"]),
        "twin_log_step": np.asarray(nf_curve_step),
        "twin_log_time": np.asarray(nf_curve_time),
        "twin_watercut": np.asarray(nf_curve_wc),
    }
    for key, val in curves.items():
        if key not in ("step", "time"):
            arrays[key] = np.asarray(val)
    for event, row in snapshots.items():
        arrays[f"snapshot_{event}_step"] = np.array(row["step"])
        arrays[f"snapshot_{event}_time"] = np.array(row["time"])
        for name in track_names:
            arrays[f"snapshot_{event}_S_{name}"] = row[f"S_{name}"]
            arrays[f"snapshot_{event}_Sf_{name}"] = row[f"Sf_{name}"]
        if event in nf_snapshots:
            arrays[f"snapshot_{event}_S_twin"] = nf_snapshots[event]
    np.savez_compressed(TRANSPORT_NPZ, **arrays)

    meta = {
        "events": events,
        "dt": dt,
        "dt_candidates": dt_candidates,
        "CFL_safety": CFL_SAFETY,
        "BT_level": BT_LEVEL,
        "half_rise_level": HALF_LEVEL,
        "snapshot_times": {k: v["time"] for k, v in snapshots.items()},
        "timing_development_run": {n: {k: stats(v) for k, v in timing[n].items()} for n in track_names},
        "one_time": {
            "PINN_training_s": float(saved["wall_time"]),
            "PoU_build_s": pou.build_s,
            "PoU_factor_s": pou._factor["factor_s"],
            "NLR_geometry_s": fast_nlr.build_s,
            "dynamic_geometry_s": dynamic.build_s,
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "single_threaded_timing_pending": True,
        },
        "outputs": {"npz": str(TRANSPORT_NPZ)},
    }
    TRANSPORT_JSON.write_text(json.dumps(jsonable(meta), indent=2) + "\n")
    print("saved", TRANSPORT_NPZ)
    print("saved", TRANSPORT_JSON)


if __name__ == "__main__":
    main()
