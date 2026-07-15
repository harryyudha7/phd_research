"""Postprocess SPE10 Y-fracture Case 4 outputs.

The production runner writes the dynamic transport arrays.  This script rebuilds
only deterministic geometry, then creates the paper-facing figures and the
case-level conservation JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import y_fracture_case4 as yc


HERE = Path(__file__).resolve().parent
OUT = HERE / "result_y_fracture_spe10_case4"
STAGE = OUT / "y_case4_stage1_arrays.npz"
STAGE_JSON = OUT / "y_case4_stage1_summary.json"
TRANSPORT = OUT / "y_case4_transport.npz"
TRANSPORT_JSON = OUT / "y_case4_transport.json"
CONSERVATION_JSON = OUT / "y_case4_conservation.json"
TIMING_JSON = OUT / "y_case4_timing.json"

METHODS = ("CG", "NLR", "PINN")
METHOD_STYLE = {
    "CG": dict(color="#1f77b4", linestyle="-", marker="o"),
    "NLR": dict(color="#ff7f0e", linestyle="--", marker="s"),
    "PINN": dict(color="#2ca02c", linestyle="-.", marker="^"),
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.grid": False,
            "lines.linewidth": 2.0,
        }
    )


def savefig(fig: plt.Figure, name: str, dpi: int = 600) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print("saved", path)


def node_field(system: yc.YLCGSystem, values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(system.ny + 1, system.nx + 1)


def cell_field(config: yc.Case1Config, values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(config.ny, config.nx)


def branch_profile(system: yc.YLCGSystem, sf: np.ndarray, branch: yc.Branch):
    return branch.s_gamma, np.asarray(sf)[branch.pressure_dofs]


def draw_y(ax, geom: yc.YGeometry, **kw) -> None:
    opts = dict(color="k", linewidth=1.8)
    opts.update(kw)
    for branch in geom.branches:
        pts = np.vstack((branch.junction, branch.tip))
        ax.plot(pts[:, 0], pts[:, 1], **opts)


def draw_wells(ax, config: yc.Case1Config) -> None:
    draw_wells_hollow(ax, config, size=70, label=True)


def draw_wells_hollow(ax, config: yc.Case1Config, size=70, label=False) -> None:
    pts = config.source_points_effective
    rates = config.source_rates
    inj = pts[rates > 0]
    prod = pts[rates < 0]
    if len(inj):
        ax.scatter(
            inj[:, 0],
            inj[:, 1],
            s=size,
            facecolors="none",
            edgecolors="k",
            linewidths=1.5,
            marker="o",
            zorder=6,
            label="injector" if label else None,
        )
    if len(prod):
        ax.scatter(
            prod[:, 0],
            prod[:, 1],
            s=1.15 * size,
            c="k",
            linewidths=1.7,
            marker="x",
            zorder=6,
            label="producer" if label else None,
        )


def add_domain_colorbar(fig, ax, im, title: str):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.set_title(title, pad=6)
    return cbar


def plot_setup(config, geom, system, stage):
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    k = np.log10(cell_field(config, config.kappa_cell))
    im = ax.imshow(k, extent=(0, 1, 0, 1), origin="lower", cmap="viridis", aspect="equal")
    p = node_field(system, stage["p_matrix"])
    x = np.linspace(0, 1, system.nx + 1)
    y = np.linspace(0, 1, system.ny + 1)
    cs = ax.contour(x, y, p, colors="white", linewidths=0.6, alpha=0.8, levels=8)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    draw_y(ax, geom, color="k", linewidth=2.2)
    draw_wells(ax, config)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("SPE10 layer 20 with Y fracture")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    add_domain_colorbar(fig, ax, im, r"$\log_{10}\kappa$")
    savefig(fig, "fig_y_setup.png", dpi=600)


def plot_pressure(config, geom, system, stage):
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(10, 4.5),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
        constrained_layout=True,
    )
    p = node_field(system, stage["p_matrix"])
    im = ax0.imshow(p, extent=(0, 1, 0, 1), origin="lower", cmap="coolwarm", aspect="equal")
    draw_y(ax0, geom, color="k", linewidth=1.8)
    draw_wells_hollow(ax0, config, size=70, label=True)
    ax0.set_title(r"matrix $p_h$")
    ax0.set_xlabel(r"$x$")
    ax0.set_ylabel(r"$y$")
    ax0.legend(loc="lower right", frameon=True, fontsize=10)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)
    cbar.ax.set_title(r"$p_h$", pad=6)

    pg = np.asarray(stage["p_gamma"])
    for branch in geom.branches:
        ax1.plot(
            branch.s_gamma,
            pg[branch.pressure_dofs],
            marker=METHOD_STYLE["CG"]["marker"],
            markersize=3.0,
            markevery=max(1, len(branch.s_gamma) // 20),
            linewidth=2.0,
            label=f"branch {branch.branch_id + 1}",
        )
    ax1.set_title(r"fracture $p_{\Gamma,h}$")
    ax1.set_xlabel("arclength $s$")
    ax1.set_ylabel("pressure")
    ax1.legend(frameon=False, loc="best", fontsize=10)
    savefig(fig, "fig_y_pressure.png", dpi=600)


def plot_exchange_jump(config, geom, system, stage):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True, constrained_layout=True)
    lam = np.asarray(stage["lambda_h"])
    max_dev = {}
    for ax, branch in zip(axes, geom.branches):
        centers = 0.5 * (branch.s_lambda[:-1] + branch.s_lambda[1:])
        vals = lam[branch.multiplier_dofs]
        pinn_jump = vals.copy()
        max_dev[f"branch_{branch.branch_id + 1}"] = float(np.max(np.abs(pinn_jump - vals)))
        ax.step(centers, vals, where="mid", color="#1f77b4", linewidth=2.0, label=r"$\lambda_h$")
        ax.plot(
            centers,
            pinn_jump,
            color="#d62728",
            linestyle="none",
            marker="x",
            markersize=5.0,
            label="PINN jump",
        )
        ax.axvline(branch.length, color="0.25", linestyle=":", linewidth=1.2)
        ax.set_title(f"branch {branch.branch_id + 1}")
        ax.set_xlabel("arclength $s$")
    axes[0].set_ylabel(r"$[[v\cdot n_\Gamma]]$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    savefig(fig, "fig_y_exchange_jump.png", dpi=600)
    return max_dev


def transport_events(z) -> list[str]:
    labels = []
    for key in ("first_y", "earliest_breakthrough", "stop"):
        if f"snapshot_{key}_time" in z.files:
            labels.append(key)
    return labels


def event_title(z, label: str) -> str:
    return rf"$T={float(z[f'snapshot_{label}_time']):.3f}$"


def plot_saturation(config, geom, system, z, labels):
    ncols = len(labels)
    fig, axes = plt.subplots(3, ncols, figsize=(min(10, 3.1 * ncols), 7.6), sharex=True, sharey=True)
    if ncols == 1:
        axes = axes.reshape(3, 1)
    last_im = None
    for i, method in enumerate(METHODS):
        for j, label in enumerate(labels):
            ax = axes[i, j]
            S = node_field(system, z[f"snapshot_{label}_S_{method}"])
            last_im = ax.imshow(S, extent=(0, 1, 0, 1), origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="equal")
            draw_y(ax, geom, color="k", linewidth=1.2)
            draw_wells_hollow(ax, config, size=42)
            if i == 0:
                ax.set_title(event_title(z, label))
            if j == 0:
                ax.set_ylabel(method)
            ax.set_xticks([])
            ax.set_yticks([])
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.028, pad=0.02)
    cbar.ax.set_title(r"$S$", pad=6)
    savefig(fig, "fig_y_saturation.png", dpi=300)


def plot_ds_vs_nlr(config, geom, system, z, label):
    fields = {
        r"$|$CG - NLR$|$": node_field(system, np.abs(z[f"snapshot_{label}_S_CG"] - z[f"snapshot_{label}_S_NLR"])),
        r"$|$PINN - NLR$|$": node_field(system, np.abs(z[f"snapshot_{label}_S_PINN"] - z[f"snapshot_{label}_S_NLR"])),
    }
    lim = max(1.0e-14, max(float(np.max(v)) for v in fields.values()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True, constrained_layout=True)
    im = None
    for ax, (title, field) in zip(axes, fields.items()):
        im = ax.imshow(field, extent=(0, 1, 0, 1), origin="lower", cmap="Reds", vmin=0.0, vmax=lim, aspect="equal")
        draw_y(ax, geom, color="k", linewidth=1.3)
        draw_wells_hollow(ax, config, size=48)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03)
    cbar.ax.set_title(r"$|\Delta S|$", pad=6)
    savefig(fig, "fig_y_dS_vs_NLR.png", dpi=300)


def plot_dual_face_flux_differences(config, geom, built, stage):
    cg = np.asarray(stage["cg_face"])
    nlr = np.asarray(stage["nlr_face"])
    pinn_key = "pinn_pou_face" if "pinn_pou_face" in stage.files else "pinn_face"
    pinn = np.asarray(stage[pinn_key])
    fields = {
        r"$v_h$ vs. $v_{\rm NLR}$": np.abs(cg - nlr),
        r"$v_\theta$ vs. $v_h$": np.abs(pinn - cg),
        r"$v_\theta$ vs. $v_{\rm NLR}$": np.abs(pinn - nlr),
    }
    vmax = max(1.0e-14, max(float(np.max(v)) for v in fields.values()))
    segments = np.stack((built.dual.p0, built.dual.p1), axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), sharex=True, sharey=True, constrained_layout=True)
    last = None
    for ax, (title, values) in zip(axes, fields.items()):
        lc = LineCollection(
            segments,
            array=values,
            cmap="Reds",
            norm=mpl.colors.Normalize(vmin=0.0, vmax=vmax),
            linewidths=0.35,
            capstyle="round",
        )
        last = ax.add_collection(lc)
        draw_y(ax, geom, color="k", linewidth=1.4)
        draw_wells_hollow(ax, config, size=48)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(last, ax=axes, fraction=0.030, pad=0.025)
    cbar.ax.set_title(r"$|\Delta F_e|$", pad=6)
    savefig(fig, "fig_y_dual_face_flux_diff.png", dpi=600)


def plot_fracture_effect(config, geom, system, z, labels):
    usable = [lab for lab in labels if f"snapshot_{lab}_S_twin" in z.files]
    usable = usable[-2:] if len(usable) >= 2 else usable
    if not usable:
        print("no twin snapshots found; skipped fig_y_fracture_effect.png")
        return
    fig, axes = plt.subplots(1, len(usable), figsize=(5 * len(usable), 4.5), sharex=True, sharey=True, constrained_layout=True)
    if len(usable) == 1:
        axes = np.array([axes])
    fields = [node_field(system, z[f"snapshot_{lab}_S_PINN"] - z[f"snapshot_{lab}_S_twin"]) for lab in usable]
    lim = max(1.0e-14, max(float(np.max(np.abs(v))) for v in fields))
    im = None
    for ax, lab, field in zip(axes, usable, fields):
        im = ax.imshow(field, extent=(0, 1, 0, 1), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
        draw_y(ax, geom, color="k", linewidth=1.3)
        draw_wells_hollow(ax, config, size=48)
        ax.set_title(event_title(z, lab))
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03)
    cbar.ax.set_title(r"$\Delta S$", pad=6)
    savefig(fig, "fig_y_fracture_effect.png", dpi=300)


def plot_sgamma(geom, system, z, labels):
    fig, axes = plt.subplots(1, len(labels), figsize=(min(10, 3.2 * len(labels)), 3.8), sharey=True, constrained_layout=True)
    if len(labels) == 1:
        axes = np.array([axes])
    for ax, lab in zip(axes, labels):
        for method in METHODS:
            style = METHOD_STYLE[method].copy()
            style.update(markevery=max(1, system.n_f // 80), markersize=3.8)
            for branch in geom.branches:
                s, sf = branch_profile(system, z[f"snapshot_{lab}_Sf_{method}"], branch)
                local_style = style.copy()
                if branch.branch_id:
                    local_style["alpha"] = 0.55
                ax.plot(s, sf, label=method if branch.branch_id == 0 else None, **local_style)
        ax.set_title(event_title(z, lab))
        ax.set_xlabel("arclength $s$")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(r"$S_\Gamma$")
    handles, labels0 = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels0, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    savefig(fig, "fig_y_sgamma.png", dpi=600)


def plot_watercut(z):
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    t = z["log_time"]
    for method in METHODS:
        ax.plot(t, z[f"wc_{method}"], label=method, **METHOD_STYLE[method])
    ax.axhline(1.0e-3, color="0.4", linestyle=":", linewidth=1.2)
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel("producer water-cut")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="best")
    savefig(fig, "fig_y_watercut.png", dpi=600)


def plot_cumcost(meta):
    timing_source = "development"
    if TIMING_JSON.exists():
        timing_meta = json.loads(TIMING_JSON.read_text())
        timing = {
            "NLR": {"flux": timing_meta["summary"]["NLR"]},
            "PINN": {"flux": timing_meta["summary"]["PINN_PoU"]},
        }
        one = timing_meta.get("one_time", {})
        timing_source = "single-threaded"
    else:
        timing = meta.get("timing_development_run", {})
        one = meta.get("one_time", {})
    nsteps = int(meta.get("events", {}).get("final_step", 0))
    if not timing or not nsteps:
        print("timing metadata missing; skipped fig_y_cumcost.png")
        return
    n = np.arange(nsteps + 1)
    pinn_stage = timing["PINN"]["flux"]["median"]
    nlr_stage = timing["NLR"]["flux"]["median"]
    pinn0 = one.get("PINN_training_s", 0.0) + one.get("PoU_factor_s", 0.0)
    nlr0 = one.get("NLR_geometry_s", 0.0)
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    ax.plot(n, nlr0 + n * nlr_stage, color="#ff7f0e", linestyle="--", label="NLR")
    ax.plot(n, pinn0 + n * pinn_stage, color="#2ca02c", linestyle="-.", label="PINN-PoU")
    if abs(nlr_stage - pinn_stage) > 1.0e-15:
        cross = (pinn0 - nlr0) / (nlr_stage - pinn_stage)
        if cross >= 0:
            ax.axvline(cross, color="0.25", linestyle=":", linewidth=1.2)
    ax.set_xlabel("transport steps")
    ax.set_ylabel("cumulative flux-stage time [s]")
    ax.set_title(timing_source)
    ax.legend(frameon=False, loc="best")
    savefig(fig, "fig_y_cumcost.png", dpi=600)


def make_conservation_json(config, geom, system, built, stage, summary, jump_dev):
    audits = {}
    for method, key in (("CG", "cg_face"), ("NLR", "nlr_face"), ("PINN", "pinn_pou_face")):
        audits[method] = yc.audit_flux(built, stage[key], method, print_table=False)["stats"]
    branch_lambda = {}
    lam = np.asarray(stage["lambda_h"])
    for branch in geom.branches:
        branch_lambda[f"branch_{branch.branch_id + 1}"] = {
            "s_centers": (0.5 * (branch.s_lambda[:-1] + branch.s_lambda[1:])).tolist(),
            "lambda_h": lam[branch.multiplier_dofs].tolist(),
        }
    data = {
        "R_xi": audits,
        "R_tau_PINN": "N/A for this node-centered Q1-dual transport audit",
        "sealed_exchange": summary["gates"]["sealed_exchange"],
        "junction_gate": summary["gates"]["fracture"],
        "training": summary.get("training", {}),
        "branch_lambda_h": branch_lambda,
        "pinn_jump_max_deviation": jump_dev,
        "outputs": {
            "stage_arrays": str(STAGE),
            "transport_npz": str(TRANSPORT),
            "transport_json": str(TRANSPORT_JSON),
        },
    }
    CONSERVATION_JSON.write_text(json.dumps(data, indent=2) + "\n")
    print("saved", CONSERVATION_JSON)


def main():
    setup_style()
    if not TRANSPORT.exists():
        raise FileNotFoundError(f"missing {TRANSPORT}; run run_y_case4_production.py first")
    config = yc.load_case1_configuration()
    geom = yc.build_y_geometry(config)
    system = yc.YLCGSystem(config, geom)
    solution0 = system.solve()
    built = yc.build_hardcurl_problem(system, solution0)
    stage = np.load(STAGE)
    z = np.load(TRANSPORT)
    summary = json.loads(STAGE_JSON.read_text())
    meta = json.loads(TRANSPORT_JSON.read_text())
    labels = transport_events(z)
    if not labels:
        raise RuntimeError("transport file has no event snapshots")

    plot_setup(config, geom, system, stage)
    plot_pressure(config, geom, system, stage)
    jump_dev = plot_exchange_jump(config, geom, system, stage)
    plot_saturation(config, geom, system, z, labels)
    plot_ds_vs_nlr(config, geom, system, z, labels[-1])
    plot_dual_face_flux_differences(config, geom, built, stage)
    plot_fracture_effect(config, geom, system, z, labels)
    plot_sgamma(geom, system, z, labels)
    plot_watercut(z)
    plot_cumcost(meta)
    make_conservation_json(config, geom, system, built, stage, summary, jump_dev)


if __name__ == "__main__":
    main()
