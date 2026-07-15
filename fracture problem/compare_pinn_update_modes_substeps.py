#!/usr/bin/env python3
"""Compare PINN update modes for different transport substep counts.

The comparison runs/reads four PINN update modes:

    frozen, full, k_last_layer, last_layer

for two transport settings:

    n_substep = 1 and 1000, with DT_outer = n_substep * dt_transport.

Outputs:

    impes_runs/pinn_update_substep_comparison/
        comparison_data.csv
        comparison_step_bars.png
        comparison_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SIMULATOR = ROOT / "impes_spe10_simulator.py"
VENV_PYTHON = ROOT.parents[1] / "bin" / "python"
RUN_ROOT = ROOT / "impes_runs" / "pinn_update_substep_comparison"

MODES = [
    ("frozen", "Frozen"),
    ("full", "Full"),
    ("k_last_layer", "K-last"),
    ("last_layer", "Linear last"),
]
UPDATE_MODES = [item for item in MODES if item[0] != "frozen"]


def read_log(path: Path) -> list[dict[str, str]]:
    with (path / "step_log.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def complete_run(path: Path, n_time: int) -> bool:
    log = path / "step_log.csv"
    if not log.exists():
        return False
    try:
        rows = read_log(path)
    except Exception:
        return False
    return len(rows) >= int(n_time)


def run_dir_for(out_root: Path, n_substep: int, mode: str, k_layers: int) -> Path:
    suffix = {
        "frozen": "frozen",
        "full": "full",
        "k_last_layer": f"k_last{k_layers}",
        "last_layer": "linear_last",
    }[mode]
    return out_root / f"nsub_{int(n_substep):04d}" / f"run_PINN_M1_{suffix}"


def run_case(args: argparse.Namespace, n_substep: int, mode: str) -> Path:
    out_dir = run_dir_for(args.out_root, n_substep, mode, args.pinn_k_layers)
    if complete_run(out_dir, args.n_time) and not args.force:
        print(f"skip complete: n_substep={n_substep}, mode={mode}, out={out_dir}")
        return out_dir

    dt_outer = float(n_substep) * float(args.dt_transport)
    cmd = [
        str(args.python),
        str(SIMULATOR),
        "--method",
        "PINN",
        "--pinn_mode",
        mode,
        "--pinn_k_layers",
        str(args.pinn_k_layers),
        "--N_time",
        str(args.n_time),
        "--DT_outer",
        f"{dt_outer:.17g}",
        "--transport_dt",
        f"{args.dt_transport:.17g}",
        "--viz_every",
        str(args.viz_every),
        "--save_every",
        str(args.save_every),
        "--lbfgs_max_iter",
        str(args.lbfgs_max_iter),
        "--lbfgs_min_calls",
        str(args.lbfgs_min_calls),
        "--lbfgs_early_stop_patience",
        str(args.lbfgs_early_stop_patience),
        "--lbfgs_rel_min_delta",
        f"{args.lbfgs_rel_min_delta:.17g}",
        "--out_dir",
        str(out_dir),
    ]
    print(f"run: n_substep={n_substep}, mode={mode}, DT_outer={dt_outer:g}")
    subprocess.run(cmd, check=True)
    return out_dir


def collect(out_root: Path, n_substeps: list[int], n_time: int, k_layers: int) -> list[dict[str, float | str | int]]:
    records: list[dict[str, float | str | int]] = []
    for n_substep in n_substeps:
        for mode, _label in MODES:
            out_dir = run_dir_for(out_root, n_substep, mode, k_layers)
            if not complete_run(out_dir, n_time):
                raise FileNotFoundError(f"Missing or incomplete run: {out_dir}")
            rows = read_log(out_dir)
            for row in rows[:n_time]:
                rec = {
                    "n_substep": int(n_substep),
                    "mode": mode,
                    "step": int(row["step"]),
                    "time": float(row["time"]),
                    "pinn_rmse": float(row["pinn_rmse"]),
                    "pinn_pre_rmse": float(row.get("pinn_pre_rmse", "nan") or "nan"),
                    "flux_s": float(row["flux_s"]),
                    "pinn_wall_s": float(row.get("pinn_wall_s", 0.0) or 0.0),
                    "pinn_pre_eval_s": float(row.get("pinn_pre_eval_s", "nan") or "nan"),
                    "pinn_linear_s": float(row.get("pinn_linear_s", "nan") or "nan"),
                    "pinn_sync_s": float(row.get("pinn_sync_s", "nan") or "nan"),
                    "pinn_error_s": float(row.get("pinn_error_s", "nan") or "nan"),
                    "R_xi_rmse": float(row["R_xi_rmse"]),
                    "CFL": float(row["CFL"]),
                    "S_post_max": float(row["S_post_max"]),
                }
                records.append(rec)
    return records


def write_records_csv(path: Path, records: list[dict[str, float | str | int]]) -> None:
    fieldnames = [
        "n_substep", "mode", "step", "time", "pinn_rmse", "pinn_pre_rmse",
        "flux_s", "pinn_wall_s", "pinn_pre_eval_s", "pinn_linear_s",
        "pinn_sync_s", "pinn_error_s", "R_xi_rmse", "CFL", "S_post_max",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def improvement_percent(records: list[dict[str, float | str | int]], step: int, n_substep: int, mode: str) -> float:
    baseline = lookup(records, step, n_substep, "frozen", "pinn_rmse")
    value = lookup(records, step, n_substep, mode, "pinn_rmse")
    return 100.0 * (baseline - value) / baseline


def step_update_percent(records: list[dict[str, float | str | int]], step: int, n_substep: int, mode: str) -> float:
    pre = lookup(records, step, n_substep, mode, "pinn_pre_rmse")
    value = lookup(records, step, n_substep, mode, "pinn_rmse")
    if not np.isfinite(pre) or pre <= 0.0:
        return np.nan
    return 100.0 * (pre - value) / pre


def improvement_records(
    records: list[dict[str, float | str | int]],
    n_substeps: list[int],
    n_time: int,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for n_substep in n_substeps:
        for mode, label in UPDATE_MODES:
            for step in range(1, n_time + 1):
                baseline = lookup(records, step, n_substep, "frozen", "pinn_rmse")
                pre = lookup(records, step, n_substep, mode, "pinn_pre_rmse")
                value = lookup(records, step, n_substep, mode, "pinn_rmse")
                rows.append(
                    {
                        "n_substep": int(n_substep),
                        "mode": mode,
                        "label": label,
                        "step": int(step),
                        "frozen_rmse": baseline,
                        "pre_update_rmse": pre,
                        "mode_rmse": value,
                        "improvement_percent": 100.0 * (baseline - value) / baseline,
                        "step_update_percent": step_update_percent(records, step, n_substep, mode),
                    }
                )
    return rows


def write_improvement_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    fieldnames = ["n_substep", "mode", "label", "step", "frozen_rmse", "pre_update_rmse", "mode_rmse", "improvement_percent", "step_update_percent"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def lookup(records: list[dict[str, float | str | int]], step: int, n_substep: int, mode: str, key: str) -> float:
    for rec in records:
        if int(rec["step"]) == step and int(rec["n_substep"]) == n_substep and str(rec["mode"]) == mode:
            return float(rec[key])
    raise KeyError((step, n_substep, mode, key))


def plot_step_bars(
    path: Path,
    records: list[dict[str, float | str | int]],
    n_substeps: list[int],
    n_time: int,
    error_ymin: float,
) -> None:
    colors = {"frozen": "#8e8e8e", "full": "#4c78a8", "k_last_layer": "#f58518", "last_layer": "#54a24b"}
    hatches = {n_substeps[0]: "", n_substeps[1]: "//"} if len(n_substeps) >= 2 else {n_substeps[0]: ""}
    width = 0.34
    x = np.arange(len(MODES), dtype=float)

    fig, axes = plt.subplots(n_time, 2, figsize=(11.5, 2.2 * n_time), constrained_layout=True)
    if n_time == 1:
        axes = axes.reshape(1, 2)

    err_ymin = min(float(error_ymin), min(float(rec["pinn_rmse"]) for rec in records) * 0.98)
    time_vals = [max(float(rec["pinn_wall_s"]), 1.0e-8) for rec in records]
    time_min = min(time_vals)
    time_max = max(time_vals)

    for step in range(1, n_time + 1):
        ax_err = axes[step - 1, 0]
        ax_time = axes[step - 1, 1]
        step_err_max = max(lookup(records, step, n_substep, mode, "pinn_rmse") for n_substep in n_substeps for mode, _ in MODES)
        err_pad = max(0.02 * step_err_max, 0.015 * max(step_err_max - err_ymin, 1.0e-12))
        for si, n_substep in enumerate(n_substeps):
            offset = (si - 0.5 * (len(n_substeps) - 1)) * width
            err = [lookup(records, step, n_substep, mode, "pinn_rmse") for mode, _ in MODES]
            tim = [lookup(records, step, n_substep, mode, "pinn_wall_s") for mode, _ in MODES]
            labels = [label for _mode, label in MODES]
            bars_err = ax_err.bar(x + offset, err, width=width, label=rf"$n_s={n_substep}$", color=[colors[m] for m, _ in MODES])
            bars_time = ax_time.bar(x + offset, tim, width=width, label=rf"$n_s={n_substep}$", color=[colors[m] for m, _ in MODES])
            for bar in [*bars_err, *bars_time]:
                bar.set_hatch(hatches.get(n_substep, ""))
                bar.set_edgecolor("black")
                bar.set_linewidth(0.4)
            for bar, (mode, _label) in zip(bars_err, MODES):
                if mode == "frozen":
                    continue
                pct_frozen = improvement_percent(records, step, n_substep, mode)
                pct_prev = step_update_percent(records, step, n_substep, mode)
                ax_err.text(
                    bar.get_x() + 0.5 * bar.get_width(),
                    bar.get_height() + err_pad,
                    f"F {pct_frozen:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    clip_on=False,
                )
                if np.isfinite(pct_prev):
                    ax_err.text(
                        bar.get_x() + 0.5 * bar.get_width(),
                        bar.get_height() + 2.35 * err_pad,
                        f"P {pct_prev:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color="#7b3294",
                        clip_on=False,
                    )

        ax_err.set_title(f"step {step}: face RMSE")
        ax_err.set_ylabel(r"$\mathrm{RMSE}(F_\theta,F_{CG})$")
        ax_err.set_ylim(err_ymin, step_err_max + 7.2 * err_pad)
        ax_err.set_xticks(x, labels)
        ax_err.grid(False)

        ax_time.set_title(f"step {step}: PINN-update time")
        ax_time.set_ylabel("seconds")
        ax_time.set_yscale("log")
        ax_time.set_ylim(max(time_min * 0.5, 1.0e-8), time_max * 2.0)
        ax_time.set_xticks(x, labels)
        ax_time.grid(False)
        if step == 1:
            ax_time.legend(frameon=False, loc="upper right")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, records: list[dict[str, float | str | int]], n_substeps: list[int], n_time: int, args: argparse.Namespace) -> None:
    lines = [
        "# PINN Update-Mode Substep Comparison",
        "",
        f"`dt_transport = {args.dt_transport:g}` and `DT_outer = n_substep * dt_transport`.",
        f"`N_time = {n_time}`, `pinn_k_layers = {args.pinn_k_layers}`.",
        "",
        "| n_substep | mode | final RMSE | mean RMSE | total PINN update [s] | mean PINN update [s] | total flux stage [s] |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for n_substep in n_substeps:
        for mode, label in MODES:
            vals = [rec for rec in records if int(rec["n_substep"]) == n_substep and str(rec["mode"]) == mode]
            vals = sorted(vals, key=lambda r: int(r["step"]))
            rmse = np.array([float(v["pinn_rmse"]) for v in vals], dtype=float)
            flux = np.array([float(v["flux_s"]) for v in vals], dtype=float)
            update = np.array([float(v["pinn_wall_s"]) for v in vals], dtype=float)
            lines.append(
                f"| {n_substep} | {label} | {rmse[-1]:.6e} | {rmse.mean():.6e} | {update.sum():.6e} | {update.mean():.6e} | {flux.sum():.6e} |"
            )
    lines.extend(
        [
            "",
            "## RMSE Improvement Relative To Frozen",
            "",
            "Positive values mean the update reduced the face RMSE compared with the frozen checkpoint flux.",
            "",
            "| n_substep | mode | final improvement | mean improvement |",
            "|---:|---|---:|---:|",
        ]
    )
    for n_substep in n_substeps:
        for mode, label in UPDATE_MODES:
            improvements = np.array([improvement_percent(records, step, n_substep, mode) for step in range(1, n_time + 1)], dtype=float)
            lines.append(f"| {n_substep} | {label} | {improvements[-1]:.2f}% | {improvements.mean():.2f}% |")
    lines.extend(
        [
            "",
            "## RMSE Improvement Relative To Previous-Step Network",
            "",
            "This uses the network inherited at the start of the same outer step as the baseline.",
            "",
            "| n_substep | mode | final step-update improvement | mean step-update improvement |",
            "|---:|---|---:|---:|",
        ]
    )
    for n_substep in n_substeps:
        for mode, label in UPDATE_MODES:
            improvements = np.array([step_update_percent(records, step, n_substep, mode) for step in range(1, n_time + 1)], dtype=float)
            lines.append(f"| {n_substep} | {label} | {improvements[-1]:.2f}% | {np.nanmean(improvements):.2f}% |")
    lines.extend(
        [
            "",
            "Per-step improvement data: `comparison_improvement.csv`.",
            "Figure: `comparison_step_bars.png`.",
            "Raw data: `comparison_data.csv`.",
            "",
            "RMSE-bar labels: black `F` = improvement relative to frozen; purple `P` = improvement relative to the previous-step network.",
            "Time bars use `pinn_wall_s`, i.e. the PINN update itself. The broader flux-stage time remains in `flux_s`.",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", type=Path, default=RUN_ROOT)
    parser.add_argument("--python", type=Path, default=VENV_PYTHON if VENV_PYTHON.exists() else Path("python3"))
    parser.add_argument("--n_time", type=int, default=5)
    parser.add_argument("--n_substeps", type=int, nargs="+", default=[1, 1000])
    parser.add_argument("--dt_transport", type=float, default=1.0e-5)
    parser.add_argument("--pinn_k_layers", type=int, default=2)
    parser.add_argument("--lbfgs_max_iter", type=int, default=300)
    parser.add_argument("--lbfgs_min_calls", type=int, default=50)
    parser.add_argument("--lbfgs_early_stop_patience", type=int, default=75)
    parser.add_argument("--lbfgs_rel_min_delta", type=float, default=1.0e-4)
    parser.add_argument("--viz_every", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--error_ymin", type=float, default=2.0e-3, help="Lower y-limit for face-RMSE bar plots. Use 0 for a zero baseline.")
    parser.add_argument("--skip_run", action="store_true", help="Only read existing run directories and regenerate outputs.")
    parser.add_argument("--force", action="store_true", help="Rerun cases even if a complete step_log.csv exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    n_substeps = [int(v) for v in args.n_substeps]
    if not args.skip_run:
        for n_substep in n_substeps:
            for mode, _label in MODES:
                run_case(args, n_substep, mode)

    records = collect(args.out_root, n_substeps, args.n_time, args.pinn_k_layers)
    improvements = improvement_records(records, n_substeps, args.n_time)
    write_records_csv(args.out_root / "comparison_data.csv", records)
    write_improvement_csv(args.out_root / "comparison_improvement.csv", improvements)
    plot_step_bars(args.out_root / "comparison_step_bars.png", records, n_substeps, args.n_time, args.error_ymin)
    write_summary(args.out_root / "comparison_summary.md", records, n_substeps, args.n_time, args)
    print("RMSE improvement relative to frozen at final step:")
    for n_substep in n_substeps:
        bits = []
        for mode, label in UPDATE_MODES:
            bits.append(f"{label}={improvement_percent(records, args.n_time, n_substep, mode):.2f}%")
        print(f"  n_substep={n_substep}: " + ", ".join(bits))
    print("RMSE improvement relative to previous-step network at final step:")
    for n_substep in n_substeps:
        bits = []
        for mode, label in UPDATE_MODES:
            bits.append(f"{label}={step_update_percent(records, args.n_time, n_substep, mode):.2f}%")
        print(f"  n_substep={n_substep}: " + ", ".join(bits))
    print(json.dumps({"out_root": str(args.out_root), "n_records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
