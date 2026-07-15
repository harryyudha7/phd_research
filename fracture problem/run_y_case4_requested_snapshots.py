"""Collect fixed-time saturation snapshots for SPE10 Y-fracture Case 4.

This reuses the trained initial PINN and PoU refresh machinery.  It does not
run any full-network training; it only reruns coupled transport to the requested
absolute times and stores the nearest time-step fields.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import y_fracture_case4 as yc


HERE = Path(__file__).resolve().parent
OUT = HERE / "result_y_fracture_spe10_case4"
PINN_CKPT = OUT / "y_case4_pinn_initial.pt"
SNAP_NPZ = OUT / "y_case4_requested_time_snapshots.npz"
SNAP_JSON = OUT / "y_case4_requested_time_snapshots.json"
RESUME = OUT / "y_case4_requested_time_snapshots_resume.pt"

TARGET_TIMES = (0.01, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 1.00, 1.50)
CFL_SAFETY = 0.45
CHECKPOINT_EVERY = 500


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


def save_resume(payload):
    tmp = RESUME.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(RESUME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--max-target-time",
        type=float,
        default=max(TARGET_TIMES),
        help="Only collect requested snapshots up to this time.",
    )
    args = parser.parse_args()

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

    track_names = ("CG", "NLR", "PINN")
    initial_conn = dynamic.connections(solution0)
    initial_flux = {
        "CG": dynamic.cg_face(solution0),
        "NLR": fast_nlr.reconstruct(solution0)["face_flux"],
    }
    view0 = dynamic.step_view(solution0, initial_flux["CG"])
    theta_before_dt = pou.theta.copy()
    initial_flux["PINN"] = pou.fit(initial_flux["CG"], built=view0)["prediction"]
    pou.theta = theta_before_dt
    dt_candidates = {
        n: yc.initial_cfl_dt(system, built0, initial_flux[n], solution0, initial_conn, cfl=CFL_SAFETY)
        for n in track_names
    }
    dt = min(dt_candidates.values())
    active_targets = tuple(t for t in TARGET_TIMES if t <= args.max_target_time + 1.0e-14)
    if not active_targets:
        raise ValueError("no target times selected")
    target_steps = {int(round(t / dt)): float(t) for t in active_targets}
    max_step = max(target_steps)

    inj = np.unique(system.cell_nodes[np.flatnonzero(config.source_rate_cell > 0)].ravel())
    if RESUME.exists() and not args.fresh:
        state = torch.load(RESUME, map_location="cpu", weights_only=False)
        step = int(state["step"])
        tracks = state["tracks"]
        snapshots = state["snapshots"]
        watercut = state["watercut"]
        pou.theta = np.asarray(state["pou_theta"])
        print("resumed requested snapshots at step", step, flush=True)
    else:
        step = 0
        tracks = {name: {"S": np.zeros(system.n_m), "Sf": np.zeros(system.n_f)} for name in track_names}
        for tr in tracks.values():
            tr["S"][inj] = 1.0
        snapshots = {}
        watercut = {name: {} for name in track_names}

    t0 = time.perf_counter()
    while step < max_step:
        step += 1
        post = {}
        for name in track_names:
            S = tracks[name]["S"]
            Sf = tracks[name]["Sf"]
            mob = yc.mobility_factor(yc.project_dual_to_cells(system, S))
            mobf = yc.mobility_factor(Sf)
            sol = system.solve(mob, mobf)
            conn = dynamic.connections(sol)
            if conn["max_cv_mismatch"] > 1.0e-12:
                raise RuntimeError(f"exchange mapping failed at step {step}: {conn['max_cv_mismatch']}")
            if name == "CG":
                flux = dynamic.cg_face(sol)
            elif name == "NLR":
                flux = fast_nlr.reconstruct(sol)["face_flux"]
            else:
                cg = dynamic.cg_face(sol)
                qpl = dynamic.qpl_face(sol)
                view = dynamic.step_view(sol, cg, qpl)
                flux = pou.fit(cg, built=view, anchor=pou.theta, ridge_rel=1.0e-8)["prediction"]
            S1, Sf1, report = yc.one_transport_step(system, built0, flux, sol, S, Sf, dt, connections=conn)
            post[name] = {"S": S1, "Sf": Sf1, "wc": float(report["producer_watercut"])}
        tracks = {n: {"S": post[n]["S"], "Sf": post[n]["Sf"]} for n in track_names}

        if step in target_steps:
            key = f"T{target_steps[step]:.2f}".replace(".", "p")
            snapshots[key] = {
                "requested_time": target_steps[step],
                "step": step,
                "time": step * dt,
                **{f"S_{n}": tracks[n]["S"].copy() for n in track_names},
                **{f"Sf_{n}": tracks[n]["Sf"].copy() for n in track_names},
            }
            for name in track_names:
                watercut[name][key] = post[name]["wc"]
            print("saved snapshot", key, "step", step, "time", step * dt, flush=True)

        if step == 1 or step % 1000 == 0:
            msg = " ".join(f"{n}:wc={post[n]['wc']:.3e}" for n in track_names)
            print(f"step={step}/{max_step} T={step*dt:.6g} {msg} wall={time.perf_counter()-t0:.1f}s", flush=True)

        if step % CHECKPOINT_EVERY == 0:
            save_resume(
                {
                    "step": step,
                    "tracks": tracks,
                    "snapshots": snapshots,
                    "watercut": watercut,
                    "pou_theta": pou.theta,
                }
            )

    arrays = {"dt": np.array(dt)}
    for key, row in snapshots.items():
        arrays[f"{key}_requested_time"] = np.array(row["requested_time"])
        arrays[f"{key}_step"] = np.array(row["step"])
        arrays[f"{key}_time"] = np.array(row["time"])
        for name in track_names:
            arrays[f"{key}_S_{name}"] = row[f"S_{name}"]
            arrays[f"{key}_Sf_{name}"] = row[f"Sf_{name}"]
    np.savez_compressed(SNAP_NPZ, **arrays)

    meta = {
        "target_times": TARGET_TIMES,
        "target_steps": {str(k): v for k, v in target_steps.items()},
        "dt": dt,
        "dt_candidates": dt_candidates,
        "snapshots": {
            k: {"requested_time": v["requested_time"], "step": v["step"], "time": v["time"]}
            for k, v in snapshots.items()
        },
        "watercut_at_snapshots": watercut,
        "outputs": {"npz": str(SNAP_NPZ)},
        "note": "No full-network training was rerun; this reuses y_case4_pinn_initial.pt and PoU refreshes during transport.",
    }
    SNAP_JSON.write_text(json.dumps(jsonable(meta), indent=2) + "\n")
    if RESUME.exists():
        RESUME.unlink()
    print("saved", SNAP_NPZ)
    print("saved", SNAP_JSON)


if __name__ == "__main__":
    main()
