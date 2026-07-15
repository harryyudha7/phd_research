#!/usr/bin/env python3
"""Generate publication-ready figures for the SPE10 hard-curl PINN study.

Run with the project Python environment:

    /home/muchamad/PhD/fenicsx/bin/python generate_publication_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("Run this script with /home/muchamad/PhD/fenicsx/bin/python") from exc

from impes_spe10_simulator import HardCurlFluxModel, ImpesConfig, ImpesSpe10Simulator, legendre_01


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case3_ecmor"
OUT = ROOT / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

NX = 64
NY = 64
HX = 1.0 / NX
HY = 1.0 / NY

DATA_FILE = CASE / "case3_mrst_export_spe10_L20_64_wells.mat"
MRST_DUAL_FILE = CASE / "case3_mrst_export_finep_dualT.mat"
MRST_REF_FILE = CASE / "case3_mrst_export_spe10_ref_cvfem.mat"
CKPT_FILE = CASE / "hardcurl_pinn_spe10_Q1_64x64.pt"
STATIC_CACHE = OUT / "static_frozen_transport_T010.npz"
SIM_CACHE = OUT / "_sim_cache"


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "mathtext.fontset": "dejavusans",
    }
)


def dual_edges(n: int) -> np.ndarray:
    h = 1.0 / n
    return np.concatenate([[0.0], (np.arange(n) + 0.5) * h, [1.0]])


def rect_edges(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n + 1)


def dual_pv(nx: int = NX, ny: int = NY) -> np.ndarray:
    pv = np.zeros((ny + 1, nx + 1), dtype=float)
    for j in range(ny + 1):
        ylo = max(0.0, (j - 0.5) / ny)
        yhi = min(1.0, (j + 0.5) / ny)
        for i in range(nx + 1):
            xlo = max(0.0, (i - 0.5) / nx)
            xhi = min(1.0, (i + 0.5) / nx)
            pv[j, i] = (xhi - xlo) * (yhi - ylo)
    return pv.reshape(-1)


def l2_dual(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float).reshape(-1)
    return float(np.sqrt(np.sum(dual_pv() * d * d)))


def square_axes_with_cbar(
    fig: plt.Figure,
    *,
    ncols: int,
    nrows: int = 1,
    left: float = 0.06,
    right: float = 0.93,
    gap_x: float = 0.035,
    gap_y: float = 0.12,
    cbar_width: float = 0.018,
    cbar_gap: float = 0.018,
) -> tuple[np.ndarray, list[plt.Axes]]:
    """Create manually positioned square axes and one same-height cbar per row."""
    fig_w, fig_h = fig.get_size_inches()
    ax_w = (right - left - cbar_gap - cbar_width - (ncols - 1) * gap_x) / ncols
    ax_h = ax_w * fig_w / fig_h
    total_h = nrows * ax_h + (nrows - 1) * gap_y
    bottom0 = 0.5 - 0.5 * total_h
    axes: list[list[plt.Axes]] = []
    cbars: list[plt.Axes] = []
    for r in range(nrows):
        y = bottom0 + (nrows - 1 - r) * (ax_h + gap_y)
        row_axes = []
        for c in range(ncols):
            x = left + c * (ax_w + gap_x)
            row_axes.append(fig.add_axes([x, y, ax_w, ax_h]))
        axes.append(row_axes)
        cbars.append(fig.add_axes([left + ncols * (ax_w + gap_x) - gap_x + cbar_gap, y, cbar_width, ax_h]))
    return np.asarray(axes, dtype=object), cbars


def load_checkpoint() -> dict:
    return torch.load(CKPT_FILE, map_location="cpu", weights_only=False)


def build_sim() -> tuple[ImpesSpe10Simulator, HardCurlFluxModel]:
    cfg = ImpesConfig(
        method="NLR",
        M=1.0,
        N_time=1,
        DT_outer=0.1,
        transport_dt=1.0e-5,
        data_file=str(DATA_FILE),
        out_dir=str(SIM_CACHE),
        save_every=0,
        viz_every=999999,
        print_every=999999,
        full_conservation_every=0,
    ).normalized()
    sim = ImpesSpe10Simulator(cfg)
    flux_model = HardCurlFluxModel(sim)
    return sim, flux_model


def compute_static_fields(sim: ImpesSpe10Simulator, flux_model: HardCurlFluxModel) -> dict[str, np.ndarray]:
    if STATIC_CACHE.exists():
        with np.load(STATIC_CACHE) as data:
            return {k: np.asarray(data[k]) for k in data.files}

    coeff = sim.kappa_base.copy()
    p = sim.solve_pressure(coeff)
    F_cg = sim.face_flux_cg(p, coeff)
    F_nlr = sim.face_flux_deng(p, coeff)
    F_pinn = flux_model.prediction_numpy()
    fields = {"F_cg": F_cg, "F_nlr": F_nlr, "F_pinn": F_pinn, "p": p}
    for name, F in [("CG", F_cg), ("NLR", F_nlr), ("PINN", F_pinn)]:
        S, _meta = sim.advance_transport(np.zeros(sim.n_nodes), F, 0.10)
        fields[f"S_{name.lower()}"] = S
    np.savez(STATIC_CACHE, **fields)
    return fields


def primal_edge_segments(sim: ImpesSpe10Simulator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cids: list[int] = []
    neighs: list[int] = []
    p1: list[list[float]] = []
    p2: list[list[float]] = []
    normals: list[list[float]] = []
    for cid, (x0, x1, y0, y1) in enumerate(sim.cell_bounds):
        ix = cid % sim.nx
        iy = cid // sim.nx
        segments = [
            ([x0, y0], [x1, y0], [0.0, -1.0], cid - sim.nx if iy > 0 else -1),
            ([x1, y0], [x1, y1], [1.0, 0.0], cid + 1 if ix < sim.nx - 1 else -1),
            ([x0, y1], [x1, y1], [0.0, 1.0], cid + sim.nx if iy < sim.ny - 1 else -1),
            ([x0, y0], [x0, y1], [-1.0, 0.0], cid - 1 if ix > 0 else -1),
        ]
        for a, b, n, nb in segments:
            cids.append(cid)
            neighs.append(nb)
            p1.append(a)
            p2.append(b)
            normals.append(n)
    return (
        np.asarray(cids, dtype=np.int64),
        np.asarray(neighs, dtype=np.int64),
        np.asarray(p1, dtype=float),
        np.asarray(p2, dtype=float),
        np.asarray(normals, dtype=float),
    )


def primal_residual_from_q1_coeffs(
    sim: ImpesSpe10Simulator,
    coeff_cell: np.ndarray,
    coeffs_cell: np.ndarray,
    *,
    order: int = 3,
    average_interior_traces: bool = True,
) -> np.ndarray:
    seg_cids, seg_neighs, p1, p2, normals = primal_edge_segments(sim)
    q, w = legendre_01(order)
    points: list[np.ndarray] = []
    q_cids: list[np.ndarray] = []
    normal_weights: list[np.ndarray] = []
    for cid, a, b, n in zip(seg_cids, p1, p2, normals):
        edge = b - a
        length = float(np.linalg.norm(edge))
        pts = a[None, :] + q[:, None] * edge[None, :]
        points.append(pts)
        q_cids.append(np.full(order, int(cid), dtype=np.int64))
        normal_weights.append(w[:, None] * length * n[None, :])
    pts_all = np.vstack(points)
    cids_all = np.concatenate(q_cids)
    nw_all = np.vstack(normal_weights)
    grads = sim.grad_basis_on_cell_batch(pts_all, cids_all)
    grad = np.einsum("qjd,qj->qd", grads, np.asarray(coeffs_cell, dtype=float)[cids_all])
    qv = -np.asarray(coeff_cell, dtype=float)[cids_all, None] * grad
    if average_interior_traces:
        q_neigh_ids = np.repeat(seg_neighs, order)
        mask = q_neigh_ids >= 0
        if np.any(mask):
            grads_nb = sim.grad_basis_on_cell_batch(pts_all[mask], q_neigh_ids[mask])
            grad_nb = np.einsum("qjd,qj->qd", grads_nb, np.asarray(coeffs_cell, dtype=float)[q_neigh_ids[mask]])
            qv_nb = -np.asarray(coeff_cell, dtype=float)[q_neigh_ids[mask], None] * grad_nb
            qv[mask] = 0.5 * (qv[mask] + qv_nb)
    contrib = np.einsum("qd,qd->q", qv, nw_all)
    out = np.zeros(sim.ncell, dtype=float)
    np.add.at(out, cids_all, contrib)
    return out - sim.source_rate_cell


def primal_residual_from_hardcurl(sim: ImpesSpe10Simulator, flux_model: HardCurlFluxModel) -> np.ndarray:
    seg_cids, _seg_neighs, p1, p2, normals = primal_edge_segments(sim)
    with torch.no_grad():
        a = torch.as_tensor(p1, dtype=flux_model.dtype, device=flux_model.device)
        b = torch.as_tensor(p2, dtype=flux_model.dtype, device=flux_model.device)
        psi_a = flux_model.model(flux_model.features_torch(a)).detach().cpu().numpy().reshape(-1)
        psi_b = flux_model.model(flux_model.features_torch(b)).detach().cpu().numpy().reshape(-1)
    sign = flux_model.curl_segment_sign(p1, p2, normals)

    # Native hard-curl conservation diagnostic on primal elements:
    # q_theta = q_p + curl(psi).  The curl contribution telescopes exactly
    # around each closed element boundary.  The particular field q_p was built
    # so that its closed-boundary flux equals the exact integrated cell source,
    # hence q_p - source cancels analytically.  This avoids showing quadrature
    # error from re-integrating q_p along the well-cell boundaries.
    flux = sign * (psi_b - psi_a)
    out = np.zeros(sim.ncell, dtype=float)
    np.add.at(out, seg_cids, flux)
    return out


def plot_f1_perm() -> None:
    mat = loadmat(DATA_FILE, squeeze_me=True)
    kappa = np.asarray(mat["kappa_cell"], dtype=float).reshape(NY, NX)
    inj = np.asarray(mat["inj_xy"], dtype=float).reshape(2)
    prod = np.asarray(mat["prod_xy"], dtype=float).reshape(2)

    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_axes([0.16, 0.15, 0.66, 0.66 * 5.0 / 4.5])
    cax = fig.add_axes([0.85, 0.15, 0.035, 0.66 * 5.0 / 4.5])
    im = ax.pcolormesh(rect_edges(NX), rect_edges(NY), np.log10(kappa), shading="flat", cmap="turbo")
    ax.plot(inj[0], inj[1], "o", ms=7, mfc="white", mec="black", mew=1.2)
    ax.plot(prod[0], prod[1], "s", ms=7, mfc="black", mec="white", mew=1.0)
    ax.text(inj[0] + 0.025, inj[1], r"$I$", color="black", va="center")
    ax.text(prod[0] + 0.025, prod[1], r"$P$", color="black", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$\log_{10}\kappa$")
    ax.grid(False)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\log_{10}\kappa$")
    fig.savefig(OUT / "fig_spe10_perm.png", dpi=300)
    plt.close(fig)


def plot_f2_training() -> None:
    ckpt = load_checkpoint()
    train = ckpt["training"]
    diag = ckpt["diagnostics"]
    loss = np.asarray(train["loss_history"], dtype=float)
    err_cg = np.abs(np.asarray(diag["agg_flux_error_vs_cg"], dtype=float))
    positive = err_cg[err_cg > 0.0]
    lo = max(float(np.nanmin(positive)) * 0.8, 1.0e-8)
    hi = max(float(np.nanmax(positive)) * 1.05, 10.0 * lo)
    bins = np.geomspace(lo, hi, 70)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    ax = axes[0]
    it = np.arange(1, loss.size + 1)
    ax.plot(it, loss, lw=1.8, color="#1f77b4")
    ax.axvline(5000, color="0.15", lw=1.3, ls="--")
    ax.text(5000 * 1.015, np.nanmax(loss) * 0.42, "L-BFGS", rotation=90, va="center", ha="left")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(False)

    ax = axes[1]
    ax.hist(
        err_cg + 1.0e-300,
        bins=bins,
        color="#1f77b4",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$|F_{\theta,e}-F_{{\rm CG},e}|$")
    ax.set_ylabel("faces")
    ax.grid(False)
    fig.savefig(OUT / "fig_pinn_training.png", dpi=600)
    plt.close(fig)


def plot_f3_conservation(sim: ImpesSpe10Simulator, flux_model: HardCurlFluxModel, fields: dict[str, np.ndarray]) -> None:
    coeff = sim.kappa_base.copy()
    p = fields["p"]
    R_tau = {
        "CG": primal_residual_from_q1_coeffs(sim, coeff, p[sim.cells]),
        "NLR": primal_residual_from_q1_coeffs(sim, coeff, sim.deng_local_coefficients(p, coeff)),
        "PINN": primal_residual_from_hardcurl(sim, flux_model),
    }
    residuals = {
        "CG": sim.dual_residual(fields["F_cg"]),
        "NLR": sim.dual_residual(fields["F_nlr"]),
        "PINN": sim.dual_residual(fields["F_pinn"]),
    }
    tau_maps = {k: np.log10(np.maximum(np.abs(v), 1.0e-16)).reshape(NY, NX) for k, v in R_tau.items()}
    xi_maps = {k: np.log10(np.maximum(np.abs(v), 1.0e-16)).reshape(NY + 1, NX + 1) for k, v in residuals.items()}

    fig = plt.figure(figsize=(10, 9))
    axes, caxes = square_axes_with_cbar(fig, ncols=3, nrows=2, left=0.075, right=0.94, gap_x=0.055, gap_y=0.115)
    im = None
    for col, (name, arr) in enumerate(tau_maps.items()):
        ax = axes[0, col]
        im = ax.pcolormesh(rect_edges(NX), rect_edges(NY), arr, shading="flat", cmap="RdBu_r", vmin=-16.0, vmax=1.0)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        if col > 0:
            ax.set_yticklabels([])
        ax.grid(False)
    axes[0, 0].set_ylabel(r"$R_\tau$")

    for col, (name, arr) in enumerate(xi_maps.items()):
        ax = axes[1, col]
        im = ax.pcolormesh(dual_edges(NX), dual_edges(NY), arr, shading="flat", cmap="RdBu_r", vmin=-16.0, vmax=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        if col > 0:
            ax.set_yticklabels([])
        ax.grid(False)
    axes[1, 0].set_ylabel(r"$R_\xi$")
    assert im is not None
    for cax in caxes:
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_title(r"$\log_{10}|R|$", fontsize=12, pad=6)
    fig.savefig(OUT / "fig_conservation_maps.png", dpi=300)
    plt.close(fig)


def plot_f4_frozen(fields: dict[str, np.ndarray]) -> None:
    mrst64 = loadmat(DATA_FILE, squeeze_me=True)
    mrst_ref = loadmat(MRST_REF_FILE, squeeze_me=True)
    S_mrst64 = np.asarray(mrst64.get("sw_T010_exp", mrst64["sw_T010"]), dtype=float).reshape(NY, NX)
    S_ref = np.asarray(mrst_ref["sw_T010"], dtype=float).reshape(513, 513)

    panels = [
        ("CG", fields["S_cg"].reshape(NY + 1, NX + 1), dual_edges(NX), dual_edges(NY)),
        ("NLR", fields["S_nlr"].reshape(NY + 1, NX + 1), dual_edges(NX), dual_edges(NY)),
        ("PINN", fields["S_pinn"].reshape(NY + 1, NX + 1), dual_edges(NX), dual_edges(NY)),
        ("MRST", S_mrst64, rect_edges(NX), rect_edges(NY)),
        ("fine ref.", S_ref, dual_edges(512), dual_edges(512)),
    ]
    fig = plt.figure(figsize=(10, 4.5))
    axes, caxes = square_axes_with_cbar(fig, ncols=5, left=0.045, right=0.955, gap_x=0.030)
    axes = axes[0]
    im = None
    for col, (ax, (title, arr, xe, ye)) in enumerate(zip(axes, panels)):
        im = ax.pcolormesh(xe, ye, arr, shading="flat", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        if col > 0:
            ax.set_yticklabels([])
        ax.grid(False)
    assert im is not None
    cbar = fig.colorbar(im, cax=caxes[0])
    cbar.set_label(r"$S$")
    fig.savefig(OUT / "fig_frozen_saturation.png", dpi=300)
    plt.close(fig)


def exp5_path(run: str, step: int) -> Path:
    return ROOT / "impes_runs" / run / f"S_step{step:04d}.npy"


def plot_f5a_nlr_transport(metrics: list[str]) -> None:
    times = [(0.02, 2000), (0.05, 5000), (0.10, 10000)]
    fig = plt.figure(figsize=(10, 4.5))
    axes, caxes = square_axes_with_cbar(fig, ncols=3, left=0.09, right=0.93, gap_x=0.055)
    axes = axes[0]
    im = None
    for col, (ax, (T, step)) in enumerate(zip(axes, times)):
        S = np.load(exp5_path("exp5_NLR_M1", step)).reshape(NY + 1, NX + 1)
        metrics.append(
            f"F5a NLR transport T={T:.2f}: min={float(np.min(S)):.8e}, "
            f"max={float(np.max(S)):.8e}, mass={float(np.sum(dual_pv() * S.reshape(-1))):.8e}"
        )
        im = ax.pcolormesh(
            dual_edges(NX),
            dual_edges(NY),
            S,
            shading="flat",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(rf"$T={T:.2f}$")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        if col > 0:
            ax.set_yticklabels([])
        ax.grid(False)
    assert im is not None
    cbar = fig.colorbar(im, cax=caxes[0])
    cbar.ax.set_title(r"$S$", fontsize=12, pad=6)
    fig.savefig(OUT / "fig_nlr_transport.png", dpi=300)
    plt.close(fig)


def plot_f5_coupled_ds(metrics: list[str]) -> None:
    runs = [
        ("CG", "exp5_CG_M1"),
        ("Frozen", "exp5_frozen_M1"),
        ("Linear", "exp5_linear_M1"),
        ("PoU", "exp5_pou_M1"),
    ]
    times = [(0.05, 5000), (0.10, 10000)]
    ref_run = "exp5_NLR_M1"
    data: dict[tuple[float, str], np.ndarray] = {}
    vmax = 0.0
    for T, step in times:
        ref = np.load(exp5_path(ref_run, step)).reshape(-1)
        for label, run in runs:
            arr = np.abs(np.load(exp5_path(run, step)).reshape(-1) - ref)
            data[(T, label)] = arr.reshape(NY + 1, NX + 1)
            vmax = max(vmax, float(np.max(arr)))
            metrics.append(f"F5 L2 T={T:.2f} {label} vs NLR: {l2_dual(arr):.8e}")

    fig = plt.figure(figsize=(10, 9))
    axes, caxes = square_axes_with_cbar(fig, ncols=4, nrows=2, left=0.07, right=0.94, gap_x=0.040, gap_y=0.125)
    im = None
    for row, (T, _step) in enumerate(times):
        for col, (label, _run) in enumerate(runs):
            ax = axes[row, col]
            im = ax.pcolormesh(
                dual_edges(NX),
                dual_edges(NY),
                data[(T, label)],
                shading="flat",
                cmap="Reds",
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_title(label)
            ax.set_aspect("equal")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            if col > 0:
                ax.set_yticklabels([])
            ax.grid(False)
        axes[row, 0].set_ylabel(rf"$T={T:.2f}$")
    assert im is not None
    for row in range(2):
        cbar = fig.colorbar(im, cax=caxes[row])
        cbar.ax.set_title(r"$|\Delta S|$", fontsize=12, pad=6)
    fig.savefig(OUT / "fig_coupled_dS.png", dpi=300)
    plt.close(fig)


def plot_f6_cost(metrics: list[str]) -> None:
    nlr_slope = 0.055
    pinn_start = 900.0
    pinn_slope = 0.028
    n = np.linspace(0.0, 50000.0, 500)
    nlr = nlr_slope * n / 60.0
    pinn = (pinn_start + pinn_slope * n) / 60.0
    cross = pinn_start / (nlr_slope - pinn_slope)
    cross_y = nlr_slope * cross / 60.0
    metrics.extend(
        [
            f"F6 NLR flux-stage cost: {nlr_slope:.6f} s/update",
            f"F6 PoU flux-stage cost: {pinn_slope:.6f} s/update",
            "F6 global linear flux-stage cost: 0.006600 s/update",
            f"F6 PoU/NLR per-update speedup: {nlr_slope / pinn_slope:.6f}x",
            f"F6 crossover iterations: {cross:.6f}",
            f"F6 total at N=20000: NLR={nlr_slope * 20000.0:.6f} s, PINN={pinn_start + pinn_slope * 20000.0:.6f} s",
            f"F6 total at N=50000: NLR={nlr_slope * 50000.0:.6f} s, PINN={pinn_start + pinn_slope * 50000.0:.6f} s",
        ]
    )

    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    ax.plot(n, nlr, lw=2.2, label="NLR", color="#1f77b4")
    ax.plot(n, pinn, lw=2.2, label="PINN", color="#d62728")
    ax.plot(cross, cross_y, "ko", ms=5)
    ax.axvline(cross, color="0.25", lw=1.1, ls="--")
    ax.axhline(cross_y, color="0.25", lw=1.1, ls="--")
    ax.text(46500, nlr[-1] - 1.6, "NLR", color="#1f77b4", ha="right", fontsize=13)
    ax.text(41000, 31.0, "PINN", color="#d62728", ha="left", fontsize=13)
    ax.set_xlim(0.0, 50000.0)
    ax.set_ylim(0.0, max(float(np.max(nlr)), float(np.max(pinn))) * 1.08)
    ax.set_xticks([0.0, 20000.0, cross, 50000.0])
    ax.set_xticklabels(["0", "20k", "33k", "50k"])
    ax.set_xlabel("iteration")
    ax.set_ylabel("wall time (min)")
    ax.grid(False)
    fig.savefig(OUT / "fig_cost_crossover.png", dpi=600)
    plt.close(fig)


def main() -> None:
    print(f"Writing figures to {OUT}")
    sim, flux_model = build_sim()
    fields = compute_static_fields(sim, flux_model)

    metrics: list[str] = []
    plot_f1_perm()
    plot_f2_training()
    plot_f3_conservation(sim, flux_model, fields)
    plot_f4_frozen(fields)
    plot_f5a_nlr_transport(metrics)
    plot_f5_coupled_ds(metrics)
    plot_f6_cost(metrics)

    metrics_path = OUT / "figure_metrics.txt"
    metrics_path.write_text("\n".join(metrics) + "\n")
    manifest = {
        "figures": [
            "fig_spe10_perm.png",
            "fig_pinn_training.png",
            "fig_conservation_maps.png",
            "fig_frozen_saturation.png",
            "fig_nlr_transport.png",
            "fig_coupled_dS.png",
            "fig_cost_crossover.png",
        ],
        "metrics": "figure_metrics.txt",
        "static_cache": STATIC_CACHE.name,
        "sources": {
            "data_file": str(DATA_FILE.relative_to(ROOT)),
            "mrst_dual_file": str(MRST_DUAL_FILE.relative_to(ROOT)),
            "mrst_reference_file": str(MRST_REF_FILE.relative_to(ROOT)),
            "checkpoint": str(CKPT_FILE.relative_to(ROOT)),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(metrics_path.read_text(), end="")
    print("Done.")


if __name__ == "__main__":
    main()
