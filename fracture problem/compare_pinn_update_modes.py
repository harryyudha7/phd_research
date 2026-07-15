#!/usr/bin/env python3
"""Compare full, last-layer, and frozen PINN IMPES update modes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "impes_runs"


def read_log(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "step_log.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def col_float(rows: list[dict[str, str]], name: str, default: float = np.nan) -> np.ndarray:
    vals = []
    for row in rows:
        raw = row.get(name, "")
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            vals.append(default)
    return np.asarray(vals, dtype=float)


def load_final_saturation(run_dir: Path) -> np.ndarray:
    steps = sorted(run_dir.glob("S_step*.npy"))
    if not steps:
        raise FileNotFoundError(f"No S_step*.npy files found in {run_dir}")
    return np.load(steps[-1])


def run_metrics(run_dir: Path, reference_s: np.ndarray | None = None) -> dict[str, float]:
    rows = read_log(run_dir)
    metrics: dict[str, float] = {}
    for name in ["solve_s", "flux_s", "pinn_wall_s", "transport_s", "viz_s"]:
        arr = col_float(rows, name, default=0.0)
        metrics[f"{name}_sum"] = float(np.nansum(arr))
        metrics[f"{name}_mean"] = float(np.nanmean(arr))
    pinn_rmse = col_float(rows, "pinn_rmse")
    metrics["pinn_rmse_mean"] = float(np.nanmean(pinn_rmse))
    metrics["pinn_rmse_final"] = float(pinn_rmse[-1])
    metrics["R_xi_rmse_mean"] = float(np.nanmean(col_float(rows, "R_xi_rmse")))
    metrics["CFL_max"] = float(np.nanmax(col_float(rows, "CFL")))
    metrics["Smax_final"] = float(col_float(rows, "S_post_max")[-1])
    metrics["Smin_final"] = float(col_float(rows, "S_post_min")[-1])
    if reference_s is not None:
        S = load_final_saturation(run_dir)
        diff = S - reference_s
        metrics["final_S_rmse_vs_full"] = float(np.sqrt(np.mean(diff * diff)))
        metrics["final_S_max_abs_vs_full"] = float(np.max(np.abs(diff)))
    return metrics


def write_metric_csv(path: Path, labels: list[str], metrics_by_label: dict[str, dict[str, float]]) -> None:
    metric_names = sorted({key for data in metrics_by_label.values() for key in data})
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *labels])
        for metric in metric_names:
            writer.writerow([metric, *[metrics_by_label[label].get(metric, "") for label in labels]])


def write_report(path: Path, run_dirs: dict[str, Path], metrics: dict[str, dict[str, float]]) -> None:
    full = metrics["full"]
    last = metrics["last_layer"]
    frozen = metrics["frozen"]
    lines = [
        "# PINN update-mode comparison, M=1",
        "",
        "Inputs:",
        "",
    ]
    for label, run_dir in run_dirs.items():
        lines.append(f"- {label}: `{run_dir}`")
    lines.extend(
        [
            "",
            "Timing:",
            "",
            "| metric | full | last layer | frozen/no update | last/full | frozen/full |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("flux_s_sum", "total flux stage"),
        ("flux_s_mean", "mean flux stage / step"),
        ("pinn_wall_s_sum", "total PINN update wall time"),
    ]:
        lines.append(
            f"| {label} | {full[key]:.6e} | {last[key]:.6e} | {frozen[key]:.6e} | "
            f"{last[key] / full[key]:.6e} | {frozen[key] / full[key]:.6e} |"
        )
    lines.extend(
        [
            "",
            "Accuracy / diagnostics:",
            "",
            "| metric | full | last layer | frozen/no update |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("pinn_rmse_mean", "mean PINN-CG face RMSE"),
        ("pinn_rmse_final", "final PINN-CG face RMSE"),
        ("R_xi_rmse_mean", "mean R_xi RMSE"),
        ("CFL_max", "max CFL"),
        ("Smax_final", "final max S"),
    ]:
        lines.append(f"| {label} | {full[key]:.6e} | {last[key]:.6e} | {frozen[key]:.6e} |")
    lines.extend(
        [
            "",
            "Final saturation difference relative to the full update:",
            "",
            "| metric | last layer | frozen/no update |",
            "|---|---:|---:|",
            f"| RMSE | {last['final_S_rmse_vs_full']:.6e} | {frozen['final_S_rmse_vs_full']:.6e} |",
            f"| max abs | {last['final_S_max_abs_vs_full']:.6e} | {frozen['final_S_max_abs_vs_full']:.6e} |",
            "",
            "Improvement of the final face RMSE over the frozen checkpoint:",
            "",
            f"- Full update: {(1.0 - full['pinn_rmse_final'] / frozen['pinn_rmse_final']) * 100.0:.2f}%",
            f"- Last-layer update: {(1.0 - last['pinn_rmse_final'] / frozen['pinn_rmse_final']) * 100.0:.2f}%",
            "",
            "Figure: `compare_PINN_update_modes_M1.png`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def plot_comparison(path: Path, labels: list[str], metrics: dict[str, dict[str, float]]) -> None:
    x = np.arange(len(labels))
    face = [metrics[label]["pinn_rmse_final"] for label in labels]
    flux_time = [metrics[label]["flux_s_sum"] for label in labels]
    s_rmse = [metrics[label].get("final_S_rmse_vs_full", 0.0) for label in labels]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), constrained_layout=True)
    axes[0].bar(x, face, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_title("final face RMSE")
    axes[0].set_ylabel(r"$\|F_\theta-F_{CG}\|_{\mathrm{RMSE}}$")
    axes[1].bar(x, flux_time, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_yscale("log")
    axes[1].set_title("flux update time")
    axes[1].set_ylabel("seconds")
    axes[2].bar(x, s_rmse, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[2].set_title("final S RMSE vs full")
    axes[2].set_ylabel("RMSE")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    run_dirs = {
        "full": RUN_ROOT / "run_PINN_M1",
        "last_layer": RUN_ROOT / "run_PINN_M1_last_layer",
        "frozen": RUN_ROOT / "run_PINN_M1_frozen",
    }
    for run_dir in run_dirs.values():
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run directory: {run_dir}")
    reference_s = load_final_saturation(run_dirs["full"])
    metrics = {
        "full": run_metrics(run_dirs["full"]),
        "last_layer": run_metrics(run_dirs["last_layer"], reference_s),
        "frozen": run_metrics(run_dirs["frozen"], reference_s),
    }
    # Define the full-vs-full saturation metrics explicitly for easier table code.
    metrics["full"]["final_S_rmse_vs_full"] = 0.0
    metrics["full"]["final_S_max_abs_vs_full"] = 0.0

    labels = ["full", "last_layer", "frozen"]
    write_metric_csv(RUN_ROOT / "compare_PINN_update_modes_M1.csv", labels, metrics)
    write_report(RUN_ROOT / "compare_PINN_update_modes_M1.md", run_dirs, metrics)
    plot_comparison(RUN_ROOT / "compare_PINN_update_modes_M1.png", labels, metrics)

    # Keep the old filename refreshed for convenience, now with the frozen baseline too.
    write_metric_csv(RUN_ROOT / "compare_PINN_full_vs_last_layer_M1.csv", labels, metrics)
    write_report(RUN_ROOT / "compare_PINN_full_vs_last_layer_M1.md", run_dirs, metrics)
    plot_comparison(RUN_ROOT / "compare_PINN_full_vs_last_layer_M1.png", labels, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
