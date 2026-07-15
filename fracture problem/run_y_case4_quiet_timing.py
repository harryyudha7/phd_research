"""Single-threaded flux-stage timing for SPE10 Y-fracture Case 4."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import y_fracture_case4 as yc


HERE = Path(__file__).resolve().parent
OUT = HERE / "result_y_fracture_spe10_case4"
PINN_CKPT = OUT / "y_case4_pinn_initial.pt"
TRANSPORT = OUT / "y_case4_transport.npz"
TIMING_JSON = OUT / "y_case4_timing.json"


def stats(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n": int(len(a)),
    }


def build_model(config):
    saved = torch.load(PINN_CKPT, map_location="cpu", weights_only=False)
    model = yc.LogKFourierPsiNet(
        config,
        tuple(saved["frequencies"]),
        int(saved["width"]),
        int(saved["depth"]),
    ).to(dtype=yc.TORCH_DTYPE)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def load_states(system):
    if not TRANSPORT.exists():
        return {"initial": (np.zeros(system.n_m), np.zeros(system.n_f))}
    z = np.load(TRANSPORT)
    labels = [key[len("snapshot_") : -len("_S_PINN")] for key in z.files if key.startswith("snapshot_") and key.endswith("_S_PINN")]
    labels = sorted(set(labels), key=lambda k: float(z[f"snapshot_{k}_time"]))
    chosen = labels[-2:] if len(labels) >= 2 else labels
    out = {}
    for label in chosen:
        out[f"{label}_NLR"] = (np.asarray(z[f"snapshot_{label}_S_NLR"]), np.asarray(z[f"snapshot_{label}_Sf_NLR"]))
        out[f"{label}_PINN"] = (np.asarray(z[f"snapshot_{label}_S_PINN"]), np.asarray(z[f"snapshot_{label}_Sf_PINN"]))
    return out or {"initial": (np.zeros(system.n_m), np.zeros(system.n_f))}


def main(repeats=20):
    config = yc.load_case1_configuration()
    geom = yc.build_y_geometry(config)
    system = yc.YLCGSystem(config, geom)
    solution0 = system.solve()
    built0 = yc.build_hardcurl_problem(system, solution0)
    dynamic = yc.YDynamicGeometry(system, built0)
    fast_nlr = yc.FastNLR(system, dynamic)
    model, saved = build_model(config)
    pou = yc.PoUHead(built0, model, window_shape=(16, 16), r=16)
    pou.factorize(1.0e-8)

    states = load_states(system)
    nlr_times = []
    pinn_times = []
    details = {}
    for label, (S, Sf) in states.items():
        mob = yc.mobility_factor(yc.project_dual_to_cells(system, S))
        mobf = yc.mobility_factor(Sf)
        sol = system.solve(mob, mobf)
        label_nlr = []
        label_pinn = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = fast_nlr.reconstruct(sol)
            label_nlr.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            cg = dynamic.cg_face(sol)
            qpl = dynamic.qpl_face(sol)
            view = dynamic.step_view(sol, cg, qpl)
            _ = pou.fit(cg, built=view, anchor=pou.theta, ridge_rel=1.0e-8)
            label_pinn.append(time.perf_counter() - t0)
        nlr_times.extend(label_nlr)
        pinn_times.extend(label_pinn)
        details[label] = {"NLR": stats(label_nlr), "PINN_PoU": stats(label_pinn)}

    out = {
        "single_threaded": True,
        "repeats_per_state": int(repeats),
        "states": details,
        "summary": {"NLR": stats(nlr_times), "PINN_PoU": stats(pinn_times)},
        "one_time": {
            "PINN_training_s": float(saved["wall_time"]),
            "PoU_build_s": float(pou.build_s),
            "PoU_factor_s": float(pou._factor["factor_s"]),
            "NLR_geometry_s": float(fast_nlr.build_s),
            "dynamic_geometry_s": float(dynamic.build_s),
        },
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
    }
    TIMING_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"], indent=2))
    print("saved", TIMING_JSON)


if __name__ == "__main__":
    main()
