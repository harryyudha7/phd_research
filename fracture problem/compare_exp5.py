#!/usr/bin/env python3
"""Compare exp5 IMPES tracks after the runs finish.

Outputs are written to ``impes_runs/exp5_comparison``:

* saturation_differences.csv
* timing_table.csv
* comparison_summary.md
* difference maps at t=0.05 and t=0.10
* face-RMSE / saturation-response figure
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "impes_runs"
OUT_DIR = RUN_ROOT / "exp5_comparison"
SNAP_TIMES = [i / 100.0 for i in range(1, 11)]
MAP_TIMES = [0.05, 0.10]
NX = 64
NY = 64


@dataclass(frozen=True)
class RunInfo:
    key: str
    label: str
    path: Path
    dt_outer: float
    n_time: int

    @property
    def step_for_time(self) -> dict[float, int]:
        return {t: int(round(t / self.dt_outer)) for t in SNAP_TIMES}


RUNS = {
    "cg": RunInfo("cg", "CG", RUN_ROOT / "exp5_CG_M1", 1.0e-5, 10000),
    "frozen": RunInfo("frozen", "Frozen", RUN_ROOT / "exp5_frozen_M1", 1.0e-5, 10000),
    "nlr": RunInfo("nlr", "NLR@1", RUN_ROOT / "exp5_NLR_M1", 1.0e-5, 10000),
    "linear": RunInfo("linear", "Linear", RUN_ROOT / "exp5_linear_M1", 1.0e-5, 10000),
    "pou": RunInfo("pou", "PoU@1", RUN_ROOT / "exp5_pou_M1", 1.0e-5, 10000),
    "full": RunInfo("full", "PINN full @1000", RUN_ROOT / "exp5_full_M1", 1.0e-2, 10),
    "nlr_coarse": RunInfo("nlr_coarse", "NLR@1000", RUN_ROOT / "exp5_NLR_coarse_M1", 1.0e-2, 10),
}


PAIR_GROUPS = [
    ("conservation", "CG vs NLR@1", "cg", "nlr"),
    ("staleness", "NLR@1000 vs NLR@1", "nlr_coarse", "nlr"),
    ("staleness", "Frozen vs NLR@1", "frozen", "nlr"),
    ("method matched cadence", "PINN full @1000 vs NLR@1000", "full", "nlr_coarse"),
    ("method vs NLR@1", "PINN full @1000 vs NLR@1", "full", "nlr"),
    ("update expressiveness", "Linear vs Frozen", "linear", "frozen"),
    ("update expressiveness", "Linear vs NLR@1", "linear", "nlr"),
    ("update expressiveness", "PoU@1 vs Frozen", "pou", "frozen"),
    ("method matched cadence", "PoU@1 vs NLR@1", "pou", "nlr"),
    ("CG anchor", "Frozen vs CG", "frozen", "cg"),
    ("CG anchor", "Linear vs CG", "linear", "cg"),
    ("CG anchor", "PoU@1 vs CG", "pou", "cg"),
    ("CG anchor", "PINN full @1000 vs CG", "full", "cg"),
    ("CG anchor", "NLR@1 vs CG", "nlr", "cg"),
]


def dual_pv(nx: int = NX, ny: int = NY) -> np.ndarray:
    hx = 1.0 / nx
    hy = 1.0 / ny
    pv = np.zeros((ny + 1, nx + 1), dtype=float)
    for j in range(ny + 1):
        ylo = max(0.0, (j - 0.5) * hy)
        yhi = min(1.0, (j + 0.5) * hy)
        for i in range(nx + 1):
            xlo = max(0.0, (i - 0.5) * hx)
            xhi = min(1.0, (i + 0.5) * hx)
            pv[j, i] = (xhi - xlo) * (yhi - ylo)
    return pv.reshape(-1)


PV = dual_pv()


def snapshot_path(run: RunInfo, time: float) -> Path:
    step = run.step_for_time[time]
    return run.path / f"S_step{step:04d}.npy"


def load_saturation(run: RunInfo, time: float) -> np.ndarray:
    path = snapshot_path(run, time)
    if not path.exists():
        raise FileNotFoundError(f"Missing snapshot for {run.label} at t={time:g}: {path}")
    arr = np.asarray(np.load(path), dtype=float).reshape(-1)
    if arr.size != (NX + 1) * (NY + 1):
        raise ValueError(f"{path} has {arr.size} entries; expected {(NX + 1) * (NY + 1)}")
    return arr


def diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    l2 = float(np.sqrt(np.sum(PV * diff * diff)))
    rmse = float(np.sqrt(np.sum(PV * diff * diff) / np.sum(PV)))
    max_abs = float(np.max(np.abs(diff)))
    absdiff = np.abs(diff)
    energy = PV * diff * diff
    total_energy = float(np.sum(energy))
    if total_energy > 0.0:
        top = max(1, int(math.ceil(0.05 * energy.size)))
        top_frac = float(np.sum(np.sort(energy)[-top:]) / total_energy)
    else:
        top_frac = 0.0
    area_halfmax = float(np.sum(PV[absdiff >= 0.5 * max_abs])) if max_abs > 0.0 else 0.0
    return {
        "l2": l2,
        "rmse": rmse,
        "max_abs": max_abs,
        "top5_energy_frac": top_frac,
        "area_halfmax": area_halfmax,
    }


def classify_pattern(metrics: dict[str, float]) -> str:
    if metrics["max_abs"] < 1.0e-12:
        return "zero"
    if metrics["top5_energy_frac"] >= 0.70 or metrics["area_halfmax"] <= 0.10:
        return "front-localized"
    if metrics["top5_energy_frac"] <= 0.45 and metrics["area_halfmax"] >= 0.20:
        return "diffuse"
    return "mixed/front-broadened"


def read_log(run: RunInfo) -> list[dict[str, str]]:
    path = run.path / "step_log.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing log: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compute_pair_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, float], np.ndarray] = {}

    def get(key: str, t: float) -> np.ndarray:
        cache_key = (key, t)
        if cache_key not in cache:
            cache[cache_key] = load_saturation(RUNS[key], t)
        return cache[cache_key]

    for t in SNAP_TIMES:
        for group, pair, key_a, key_b in PAIR_GROUPS:
            metrics = diff_metrics(get(key_a, t), get(key_b, t))
            rows.append(
                {
                    "time": f"{t:.2f}",
                    "group": group,
                    "pair": pair,
                    "method": RUNS[key_a].label,
                    "reference": RUNS[key_b].label,
                    "l2": metrics["l2"],
                    "rmse": metrics["rmse"],
                    "max_abs": metrics["max_abs"],
                    "top5_energy_frac": metrics["top5_energy_frac"],
                    "area_halfmax": metrics["area_halfmax"],
                    "pattern": classify_pattern(metrics),
                }
            )
    return rows


def plot_maps(time: float, out_dir: Path) -> None:
    ref = load_saturation(RUNS["nlr"], time)
    keys = ["cg", "frozen", "linear", "pou", "full", "nlr_coarse"]
    data = [np.abs(load_saturation(RUNS[key], time) - ref).reshape(NY + 1, NX + 1) for key in keys]
    vmax = max(float(np.max(d)) for d in data)
    fig, axes = plt.subplots(1, len(keys), figsize=(18.0, 3.4), constrained_layout=True)
    for ax, key, arr in zip(axes, keys, data):
        im = ax.pcolormesh(np.linspace(0.0, 1.0, NX + 1), np.linspace(0.0, 1.0, NY + 1), arr, shading="auto", cmap="Reds", vmin=0.0, vmax=vmax)
        ax.set_title(f"{RUNS[key].label} vs NLR@1")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("|Delta S|")
    fig.suptitle(f"Exp5 saturation differences at t={time:.2f}")
    fig.savefig(out_dir / f"diff_maps_vs_NLR_t{time:.2f}.png", dpi=220)
    plt.close(fig)


def plot_response(rows: list[dict[str, object]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True, constrained_layout=True)
    for key, style in [("frozen", "-"), ("linear", "-"), ("pou", "-"), ("full", "-o")]:
        log = read_log(RUNS[key])
        t = np.array([float(r["time"]) for r in log], dtype=float)
        y = np.array([float(r["pinn_rmse"]) for r in log], dtype=float)
        axes[0].plot(t, y, style, label=RUNS[key].label, lw=1.8, ms=4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("face RMSE vs CG")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)

    for pair_name in ["Frozen vs NLR@1", "Linear vs NLR@1", "PoU@1 vs NLR@1", "PINN full @1000 vs NLR@1", "NLR@1000 vs NLR@1"]:
        rr = [r for r in rows if r["pair"] == pair_name]
        axes[1].plot(
            [float(r["time"]) for r in rr],
            [float(r["rmse"]) for r in rr],
            "-o",
            label=pair_name.replace(" vs NLR@1", ""),
            lw=1.8,
            ms=4,
        )
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("RMSE(Delta S) vs NLR@1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.savefig(out_dir / "face_rmse_and_saturation_response.png", dpi=220)
    plt.close(fig)


def timing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in RUNS.values():
        log = read_log(run)
        flux = np.array([float(r["flux_s"]) for r in log], dtype=float)
        solve = np.array([float(r["solve_s"]) for r in log], dtype=float)
        transport = np.array([float(r["transport_s"]) for r in log], dtype=float)
        rxi_int = np.array([float(r.get("R_xi_int_rmse", "nan")) for r in log], dtype=float)
        rxi_bnd = np.array([float(r.get("R_xi_bnd_rmse", "nan")) for r in log], dtype=float)
        rows.append(
            {
                "method": run.label,
                "n_steps": len(log),
                "total_solve_s": float(np.sum(solve)),
                "total_flux_s": float(np.sum(flux)),
                "total_transport_s": float(np.sum(transport)),
                "mean_flux_s": float(np.mean(flux)),
                "median_flux_s": float(np.median(flux)),
                "max_flux_s": float(np.max(flux)),
                "mean_R_xi_int_rmse": float(np.nanmean(rxi_int)),
                "final_R_xi_int_rmse": float(rxi_int[-1]),
                "mean_R_xi_bnd_rmse": float(np.nanmean(rxi_bnd)),
                "final_R_xi_bnd_rmse": float(rxi_bnd[-1]),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, object]], time: float) -> str:
    selected = [r for r in rows if abs(float(r["time"]) - time) < 5.0e-12]
    lines = [f"### t = {time:.2f}", "", "| group | pair | L2 | RMSE | max | pattern |", "|---|---:|---:|---:|---:|---|"]
    for r in selected:
        lines.append(
            f"| {r['group']} | {r['pair']} | {float(r['l2']):.6e} | "
            f"{float(r['rmse']):.6e} | {float(r['max_abs']):.6e} | {r['pattern']} |"
        )
    return "\n".join(lines)


def write_summary(out_dir: Path, rows: list[dict[str, object]], timings: list[dict[str, object]]) -> None:
    lines = [
        "# Exp5 IMPES Comparison",
        "",
        "Interpretation guardrails:",
        "",
        "- If Frozen is close to NLR@1 at T=0.1, the scoped conclusion is that flux updates do not matter much at M=1 for this weak-coupling/factor-2 drift setting, not that updates never matter.",
        "- CG differences from NLR@1 isolate non-conservation rather than flux staleness. If CG is closest to NLR@1, report that plainly because it challenges the conservation thesis in this dynamic setting.",
        "- NLR is exact on interior/source dual CVs; boundary-CV residuals are reported separately in the run logs and timing table context.",
        "- PoU@1 is the per-step adaptive hard-curl head: compare it to NLR@1 for apples-to-apples refresh cadence and to Linear/Frozen for update expressiveness.",
        "",
        markdown_table(rows, 0.05),
        "",
        markdown_table(rows, 0.10),
        "",
        "## Timing",
        "",
        "| method | steps | total flux s | mean flux s | median flux s | final Rxi int | final Rxi bnd | total solve s | total transport s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in timings:
        lines.append(
            f"| {r['method']} | {int(r['n_steps'])} | {float(r['total_flux_s']):.3f} | "
            f"{float(r['mean_flux_s']):.6f} | {float(r['median_flux_s']):.6f} | "
            f"{float(r['final_R_xi_int_rmse']):.3e} | {float(r['final_R_xi_bnd_rmse']):.3e} | "
            f"{float(r['total_solve_s']):.3f} | {float(r['total_transport_s']):.3f} |"
        )
    (out_dir / "comparison_summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = compute_pair_rows()
    write_csv(
        out_dir / "saturation_differences.csv",
        rows,
        ["time", "group", "pair", "method", "reference", "l2", "rmse", "max_abs", "top5_energy_frac", "area_halfmax", "pattern"],
    )
    timings = timing_rows()
    write_csv(
        out_dir / "timing_table.csv",
        timings,
        [
            "method",
            "n_steps",
            "total_solve_s",
            "total_flux_s",
            "total_transport_s",
            "mean_flux_s",
            "median_flux_s",
            "max_flux_s",
            "mean_R_xi_int_rmse",
            "final_R_xi_int_rmse",
            "mean_R_xi_bnd_rmse",
            "final_R_xi_bnd_rmse",
        ],
    )
    for t in MAP_TIMES:
        plot_maps(t, out_dir)
    plot_response(rows, out_dir)
    write_summary(out_dir, rows, timings)
    print(f"Wrote exp5 comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
