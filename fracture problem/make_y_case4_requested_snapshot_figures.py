"""Publication figures for requested fixed-time Y-fracture saturation snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import y_fracture_case4 as yc
from make_y_case4_outputs import (
    draw_wells_hollow,
    draw_y,
    node_field,
    savefig,
    setup_style,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "result_y_fracture_spe10_case4"
SNAP_NPZ = OUT / "y_case4_requested_time_snapshots.npz"
SNAP_JSON = OUT / "y_case4_requested_time_snapshots.json"

METHODS = ("CG", "NLR", "PINN")
REQUEST_GROUPS = (
    (0.01, 0.05, 0.10),
    (0.15, 0.25, 0.35),
    (0.50, 1.00, 1.50),
)
OUT_NAMES = (
    "fig_y_saturation_times_001_005_010.png",
    "fig_y_saturation_times_015_025_035.png",
    "fig_y_saturation_times_050_100_150.png",
)


def snapshot_catalog(meta: dict) -> list[tuple[str, float, float, int]]:
    rows = []
    for key, row in meta["snapshots"].items():
        rows.append((key, float(row["requested_time"]), float(row["time"]), int(row["step"])))
    return sorted(rows, key=lambda r: r[1])


def keys_for_requested(meta: dict, requested: tuple[float, ...]) -> list[tuple[str, float, float, int]] | None:
    rows = snapshot_catalog(meta)
    selected = []
    for t in requested:
        matches = [r for r in rows if abs(r[1] - t) <= 1.0e-12]
        if not matches:
            return None
        selected.append(matches[0])
    return selected


def have_group(z, meta: dict, requested: tuple[float, ...]) -> bool:
    picks = keys_for_requested(meta, requested)
    if picks is None:
        return False
    return all(f"{key}_S_CG" in z.files for key, *_ in picks)


def plot_group(config: yc.Case1Config, geom: yc.YGeometry, system: yc.YLCGSystem, z, meta, requested, out_name):
    picks = keys_for_requested(meta, requested)
    if picks is None:
        raise RuntimeError(f"requested snapshots unavailable: {requested}")
    fig, axes = plt.subplots(
        len(METHODS),
        len(picks),
        figsize=(10, 8.0),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.05, "hspace": 0.08},
    )

    last_im = None
    for i, method in enumerate(METHODS):
        for j, (key, requested_time, actual_time, step) in enumerate(picks):
            ax = axes[i, j]
            S = node_field(system, z[f"{key}_S_{method}"])
            last_im = ax.imshow(
                S,
                extent=(0, 1, 0, 1),
                origin="lower",
                cmap="Blues",
                vmin=0.0,
                vmax=1.0,
                aspect="equal",
                interpolation="nearest",
            )
            draw_y(ax, geom, color="k", linewidth=1.4)
            draw_wells_hollow(ax, config, size=50)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(rf"$T = {actual_time:.3f}$")
            if j == 0:
                ax.set_ylabel(method, rotation=90, labelpad=20, fontsize=16)

    cbar = fig.colorbar(
        last_im,
        ax=axes,
        location="right",
        fraction=0.030,
        pad=0.025,
        shrink=1.0,
    )
    cbar.ax.set_title(r"$S$", pad=6)
    savefig(fig, out_name, dpi=600)


def main() -> None:
    setup_style()
    config = yc.load_case1_configuration()
    geom = yc.build_y_geometry(config)
    system = yc.YLCGSystem(config, geom)
    z = np.load(SNAP_NPZ)
    meta = json.loads(SNAP_JSON.read_text())

    for requested, out_name in zip(REQUEST_GROUPS, OUT_NAMES):
        if have_group(z, meta, requested):
            plot_group(config, geom, system, z, meta, requested, out_name)
        else:
            print("skipping", out_name, "(snapshots not available)")


if __name__ == "__main__":
    main()
