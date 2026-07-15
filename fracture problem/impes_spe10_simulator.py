#!/usr/bin/env python3
"""Sequential IMPES simulator for the SPE10/Deng Q1 dual-mesh test.

The script ports the frozen-flux notebook workflow into a reusable time-stepper:

    S^n on nodal dual CVs -> Q1 CG pressure -> selected face flux -> upwind BL transport.

The PINN mode uses the hard-curl stream-function checkpoint produced by
``LCG_DengGinting_example4_spe10_Q1_no_fracture_hardcurl_PINN.ipynb``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu, spsolve

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
    TORCH_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
CASE_DIR = ROOT / "case3_ecmor"


@dataclass
class ImpesConfig:
    method: str = "NLR"
    M: float = 1.0
    N_time: int = 1000
    DT_outer: float = 0.01
    transport_dt: float | None = 1.0e-5
    cfl_threshold: float = 1.0
    pinn_mode: str = "full"
    pinn_k_layers: int = 2
    lbfgs_max_iter: int = 1000
    lbfgs_history_size: int = 50
    lbfgs_print_every: int = 25
    lbfgs_early_stop_patience: int = 75
    lbfgs_min_delta: float = 1.0e-10
    lbfgs_rel_min_delta: float = 1.0e-4
    lbfgs_min_calls: int = 50
    t0_checkpoint_path: str = str(CASE_DIR / "hardcurl_pinn_spe10_Q1_64x64.pt")
    pou_checkpoint_path: str = str(CASE_DIR / "hardcurl_pinn_spe10_Q1_64x64_pou.pt")
    pou_anchor_mode: str = "previous"
    data_file: str = str(CASE_DIR / "case3_mrst_export_spe10_L20_64_wells.mat")
    out_dir: str = str(ROOT / "impes_runs" / "run_NLR_M1")
    viz_every: int | None = None
    full_conservation_every: int = 0
    save_every: int = 0
    print_every: int = 1
    nx: int = 64
    ny: int = 64
    phi: float = 1.0
    clip_saturation: bool = True
    boundary_inflow_s: float = 0.0
    source_saturation: float = 1.0
    dry_run: bool = False
    ridge_last_layer: float = 1.0e-10
    qp_quad_order: int = 128
    face_quad_order: int = 2
    streamline_grid: int = 32
    pressure_vmin: float = -1.0
    pressure_vmax: float = 0.5
    validate_deng: bool = False
    validate_only: bool = False
    deng_reference_flux: str = ""
    save_pinn_checkpoints: bool = False

    def normalized(self) -> "ImpesConfig":
        self.method = self.method.upper()
        if self.method not in {"CG", "NLR", "PROJ", "PINN"}:
            raise ValueError("method must be one of CG, NLR, PROJ, PINN")
        self.pinn_mode = self.pinn_mode.lower().replace("-", "_")
        if self.pinn_mode == "no_update":
            self.pinn_mode = "frozen"
        if self.pinn_mode in {"pou_head", "pouhead"}:
            self.pinn_mode = "pou"
        if self.pinn_mode not in {"full", "last_layer", "k_last_layer", "frozen", "pou"}:
            raise ValueError("pinn_mode must be 'full', 'last_layer', 'k_last_layer', 'frozen', or 'pou'")
        self.pou_anchor_mode = self.pou_anchor_mode.lower().replace("-", "_")
        if self.pou_anchor_mode not in {"previous", "theta_bar"}:
            raise ValueError("pou_anchor_mode must be 'previous' or 'theta_bar'")
        self.pinn_k_layers = int(self.pinn_k_layers)
        if self.pinn_k_layers < 1:
            raise ValueError("pinn_k_layers must be at least 1")
        if self.dry_run:
            self.N_time = 1
            self.method = "PINN"
            self.full_conservation_every = 1
        if self.DT_outer <= 0.0:
            raise ValueError("DT_outer must be positive")
        if self.transport_dt is None:
            self.transport_dt = float(self.DT_outer)
        if self.transport_dt <= 0.0:
            raise ValueError("transport_dt must be positive")
        if self.viz_every is None:
            self.viz_every = int(self.N_time)
        return self


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def stats_dict(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        f"{prefix}_mean_abs": float(np.mean(np.abs(arr))),
        f"{prefix}_median_abs": float(np.median(np.abs(arr))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(arr * arr))),
        f"{prefix}_max_abs": float(np.max(np.abs(arr))),
        f"{prefix}_sum": float(np.sum(arr)),
    }


def fractional_flow(S: np.ndarray | float, M: float) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    den = M * S * S + (1.0 - S) * (1.0 - S)
    return M * S * S / np.maximum(den, 1.0e-300)


def mobility_factor(S: np.ndarray, M: float) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    return M * S * S + (1.0 - S) * (1.0 - S)


def max_fractional_flow_derivative(M: float, n: int = 200001) -> float:
    S = np.linspace(0.0, 1.0, int(n))
    den = M * S * S + (1.0 - S) * (1.0 - S)
    # derivative of M S^2 / den
    dden = 2.0 * M * S - 2.0 * (1.0 - S)
    num = 2.0 * M * S * den - M * S * S * dden
    fp = num / np.maximum(den * den, 1.0e-300)
    return float(np.max(np.abs(fp)))


def legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    q, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (q + 1.0), 0.5 * w


class HardCurlPsiNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_dim: int, hidden_dim: int = 96, depth: int = 4):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PINN mode")
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def hidden_features(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x


class ImpesSpe10Simulator:
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0

    def __init__(self, cfg: ImpesConfig):
        self.cfg = cfg.normalized()
        self.out_dir = Path(self.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.out_dir / "figures"
        self.fig_dir.mkdir(exist_ok=True)
        self.csv_path = self.out_dir / "step_log.csv"
        self.nx = int(self.cfg.nx)
        self.ny = int(self.cfg.ny)
        self.hx = (self.xmax - self.xmin) / self.nx
        self.hy = (self.ymax - self.ymin) / self.ny
        self.ncell = self.nx * self.ny
        self.n_nodes = (self.nx + 1) * (self.ny + 1)
        self.x_edges = np.linspace(self.xmin, self.xmax, self.nx + 1)
        self.y_edges = np.linspace(self.ymin, self.ymax, self.ny + 1)
        self.x_nodes = self.x_edges.copy()
        self.y_nodes = self.y_edges.copy()
        self.cell_area = self.hx * self.hy
        self.load_problem_data()
        self.build_q1_topology()
        self.build_sources()
        self.build_dual_mesh()
        self.build_dual_diagnostic_masks()
        self.precompute_pressure_assembly()
        self.deng_geometry_ready = False
        if self.cfg.method == "NLR" or self.cfg.validate_deng:
            self.precompute_deng_geometry()
        self.precompute_face_quadrature()
        self.build_dual_incidence()
        self.fprime_max = max_fractional_flow_derivative(self.cfg.M)
        self.flux_model: HardCurlFluxModel | None = None
        if self.cfg.method == "PINN":
            self.flux_model = HardCurlFluxModel(self)
        with (self.out_dir / "config.json").open("w") as f:
            json.dump(asdict(self.cfg), f, indent=2, default=json_safe)
        self.write_implementation_notes()
        self.write_csv_header()

    def node_id(self, i: int | np.ndarray, j: int | np.ndarray) -> int | np.ndarray:
        return np.asarray(j) * (self.nx + 1) + np.asarray(i)

    def load_problem_data(self) -> None:
        path = Path(self.cfg.data_file)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find data_file: {path}")
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "kappa_cell" not in mat:
            raise KeyError(f"{path.name} must contain kappa_cell")
        kappa = np.asarray(mat["kappa_cell"], dtype=float).reshape(-1)
        if kappa.size != self.ncell:
            raise ValueError(f"kappa_cell has {kappa.size} entries, expected {self.ncell}")
        self.data_file = path
        self.kappa_base = kappa.copy()
        self.source_points_requested = np.array([[0.2, 0.4], [0.8, 0.5]], dtype=float)
        self.source_rates = np.array([1.0, -5.0], dtype=float)
        if "inj_xy" in mat and "prod_xy" in mat:
            self.source_points_requested = np.vstack([
                np.asarray(mat["inj_xy"], dtype=float).reshape(1, 2),
                np.asarray(mat["prod_xy"], dtype=float).reshape(1, 2),
            ])
            self.source_rates = np.array([
                float(np.asarray(mat.get("q_inj", 1.0)).reshape(())),
                float(np.asarray(mat.get("q_prod", -5.0)).reshape(())),
            ])

    def rect_cell_ids_from_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        ix = np.floor(np.clip((pts[:, 0] - self.xmin) / self.hx, 0.0, np.nextafter(float(self.nx), 0.0))).astype(int)
        iy = np.floor(np.clip((pts[:, 1] - self.ymin) / self.hy, 0.0, np.nextafter(float(self.ny), 0.0))).astype(int)
        return np.clip(iy, 0, self.ny - 1) * self.nx + np.clip(ix, 0, self.nx - 1)

    def build_q1_topology(self) -> None:
        cells = []
        bounds = []
        centers = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                cells.append([
                    int(self.node_id(ix, iy)),
                    int(self.node_id(ix + 1, iy)),
                    int(self.node_id(ix, iy + 1)),
                    int(self.node_id(ix + 1, iy + 1)),
                ])
                x0 = self.xmin + ix * self.hx
                y0 = self.ymin + iy * self.hy
                bounds.append((x0, x0 + self.hx, y0, y0 + self.hy))
                centers.append((x0 + 0.5 * self.hx, y0 + 0.5 * self.hy))
        self.cells = np.asarray(cells, dtype=np.int64)
        self.cell_bounds = np.asarray(bounds, dtype=float)
        self.cell_centers = np.asarray(centers, dtype=float)
        boundary = np.zeros(self.n_nodes, dtype=bool)
        for j in range(self.ny + 1):
            boundary[int(self.node_id(0, j))] = True
            boundary[int(self.node_id(self.nx, j))] = True
        for i in range(self.nx + 1):
            boundary[int(self.node_id(i, 0))] = True
            boundary[int(self.node_id(i, self.ny))] = True
        self.boundary_nodes = np.flatnonzero(boundary)
        self.free_nodes = np.flatnonzero(~boundary)

    def build_sources(self) -> None:
        self.source_cell_ids = self.rect_cell_ids_from_points(self.source_points_requested)
        self.source_rate_cell = np.zeros(self.ncell, dtype=float)
        for cid, rate in zip(self.source_cell_ids, self.source_rates):
            self.source_rate_cell[int(cid)] += float(rate)
        self.source_density_cell = self.source_rate_cell / self.cell_area
        self.load_vector = np.zeros(self.n_nodes, dtype=float)
        for cid, rate in enumerate(self.source_rate_cell):
            if rate != 0.0:
                self.load_vector[self.cells[cid]] += float(rate) / 4.0
        self.dual_source_rate = np.zeros(self.n_nodes, dtype=float)
        for cid, rate in enumerate(self.source_rate_cell):
            if rate != 0.0:
                self.dual_source_rate[self.cells[cid]] += float(rate) / 4.0

    def dual_span(self, i: int, h: float) -> tuple[float, float]:
        return max(0.0, (int(i) - 0.5) * h), min(1.0, (int(i) + 0.5) * h)

    def build_dual_mesh(self) -> None:
        owner: list[int] = []
        neigh: list[int] = []
        p1: list[list[float]] = []
        p2: list[list[float]] = []
        normal: list[list[float]] = []

        def add(o: int, n: int, a: list[float], b: list[float], nor: list[float]) -> None:
            owner.append(int(o))
            neigh.append(int(n))
            p1.append(a)
            p2.append(b)
            normal.append(nor)

        for k in range(self.ny + 1):
            ylo, yhi = self.dual_span(k, self.hy)
            for j in range(self.nx):
                x = self.xmin + (j + 0.5) * self.hx
                add(int(self.node_id(j, k)), int(self.node_id(j + 1, k)), [x, ylo], [x, yhi], [1.0, 0.0])
        for j in range(self.nx + 1):
            xlo, xhi = self.dual_span(j, self.hx)
            for k in range(self.ny):
                y = self.ymin + (k + 0.5) * self.hy
                add(int(self.node_id(j, k)), int(self.node_id(j, k + 1)), [xlo, y], [xhi, y], [0.0, 1.0])
        for k in range(self.ny + 1):
            ylo, yhi = self.dual_span(k, self.hy)
            add(int(self.node_id(0, k)), -1, [self.xmin, ylo], [self.xmin, yhi], [-1.0, 0.0])
            add(int(self.node_id(self.nx, k)), -1, [self.xmax, ylo], [self.xmax, yhi], [1.0, 0.0])
        for j in range(self.nx + 1):
            xlo, xhi = self.dual_span(j, self.hx)
            add(int(self.node_id(j, 0)), -1, [xlo, self.ymin], [xhi, self.ymin], [0.0, -1.0])
            add(int(self.node_id(j, self.ny)), -1, [xlo, self.ymax], [xhi, self.ymax], [0.0, 1.0])

        self.dual_owner = np.asarray(owner, dtype=np.int64)
        self.dual_neigh = np.asarray(neigh, dtype=np.int64)
        self.dual_has_neighbor = self.dual_neigh >= 0
        self.dual_p1 = np.asarray(p1, dtype=float)
        self.dual_p2 = np.asarray(p2, dtype=float)
        self.dual_normal = np.asarray(normal, dtype=float)
        self.dual_len = np.linalg.norm(self.dual_p2 - self.dual_p1, axis=1)
        self.dual_centroid = 0.5 * (self.dual_p1 + self.dual_p2)
        self.dual_pv = np.zeros(self.n_nodes, dtype=float)
        for k in range(self.ny + 1):
            ylo, yhi = self.dual_span(k, self.hy)
            for j in range(self.nx + 1):
                xlo, xhi = self.dual_span(j, self.hx)
                self.dual_pv[int(self.node_id(j, k))] = self.cfg.phi * (xhi - xlo) * (yhi - ylo)

    def build_dual_diagnostic_masks(self) -> None:
        ids = np.arange(self.n_nodes, dtype=np.int64)
        ix = ids % (self.nx + 1)
        iy = ids // (self.nx + 1)
        self.dual_boundary_mask = (ix == 0) | (ix == self.nx) | (iy == 0) | (iy == self.ny)
        self.dual_interior_mask = ~self.dual_boundary_mask
        self.dual_source_mask = np.abs(self.dual_source_rate) > 1.0e-300

    def precompute_pressure_assembly(self) -> None:
        q = np.array([0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0)])
        w = np.array([0.5, 0.5])
        Ke = np.zeros((4, 4), dtype=float)
        det = self.hx * self.hy
        for xi, wx in zip(q, w):
            for eta, wy in zip(q, w):
                grads = self.q1_grads_phys(np.array([[xi, eta]], dtype=float))[0]
                Ke += wx * wy * det * (grads @ grads.T)
        self.Ke_ref = Ke
        rows = np.repeat(self.cells, 4, axis=1).reshape(-1)
        cols = np.tile(self.cells, (1, 4)).reshape(-1)
        self.K_rows = rows
        self.K_cols = cols

    def q1_grads_phys(self, xi_eta: np.ndarray) -> np.ndarray:
        xi_eta = np.asarray(xi_eta, dtype=float).reshape(-1, 2)
        xi = xi_eta[:, 0]
        eta = xi_eta[:, 1]
        grads = np.empty((len(xi), 4, 2), dtype=float)
        grads[:, 0, 0] = -(1.0 - eta) / self.hx
        grads[:, 0, 1] = -(1.0 - xi) / self.hy
        grads[:, 1, 0] = (1.0 - eta) / self.hx
        grads[:, 1, 1] = -xi / self.hy
        grads[:, 2, 0] = -eta / self.hx
        grads[:, 2, 1] = (1.0 - xi) / self.hy
        grads[:, 3, 0] = eta / self.hx
        grads[:, 3, 1] = xi / self.hy
        return grads

    def precompute_face_quadrature(self) -> None:
        q, w = legendre_01(self.cfg.face_quad_order)
        points: list[np.ndarray] = []
        normal_weights: list[np.ndarray] = []
        face_ids: list[np.ndarray] = []
        cell_ids: list[np.ndarray] = []
        for fid, (a, b, n) in enumerate(zip(self.dual_p1, self.dual_p2, self.dual_normal)):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            edge = b - a
            breaks = self.segment_grid_breakpoints(a, b)
            for t0, t1 in zip(breaks[:-1], breaks[1:]):
                if t1 - t0 <= 1.0e-14:
                    continue
                aa = a + t0 * edge
                bb = a + t1 * edge
                seg = bb - aa
                length = float(np.linalg.norm(seg))
                pts = aa[None, :] + q[:, None] * seg[None, :]
                mid = 0.5 * (aa + bb)
                cid = int(self.rect_cell_ids_from_points(mid[None, :])[0])
                points.append(pts)
                normal_weights.append(w[:, None] * (length * n[None, :]))
                face_ids.append(np.full(len(q), fid, dtype=np.int64))
                cell_ids.append(np.full(len(q), cid, dtype=np.int64))
        self.face_q_points = np.vstack(points)
        self.face_q_normal_weights = np.vstack(normal_weights)
        self.face_q_face_ids = np.concatenate(face_ids)
        self.face_q_cell_ids = np.concatenate(cell_ids)

    def segment_grid_breakpoints(self, a: np.ndarray, b: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
        d = b - a
        ts = [0.0, 1.0]
        if abs(d[0]) > tol:
            lo, hi = sorted([a[0], b[0]])
            for xline in self.x_edges[1:-1]:
                if lo + tol < xline < hi - tol:
                    ts.append(float((xline - a[0]) / d[0]))
        if abs(d[1]) > tol:
            lo, hi = sorted([a[1], b[1]])
            for yline in self.y_edges[1:-1]:
                if lo + tol < yline < hi - tol:
                    ts.append(float((yline - a[1]) / d[1]))
        return np.asarray(sorted(set(round(t, 15) for t in ts)), dtype=float)

    def build_dual_incidence(self) -> None:
        col = np.arange(len(self.dual_owner), dtype=np.int64)
        rows = [self.dual_owner]
        cols = [col]
        vals = [np.ones(len(col), dtype=float)]
        m = self.dual_has_neighbor
        rows.append(self.dual_neigh[m])
        cols.append(col[m])
        vals.append(-np.ones(np.count_nonzero(m), dtype=float))
        self.B = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n_nodes, len(col))).tocsc()
        L = (self.B @ self.B.T).tocsc()
        self.graph_lu = splu(L + 1.0e-12 * csc_matrix(np.eye(self.n_nodes)))

    def project_dual_to_cells(self, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=float)
        return np.mean(S[self.cells], axis=1)

    def solve_pressure(self, coeff_cell: np.ndarray) -> np.ndarray:
        vals = (coeff_cell[:, None, None] * self.Ke_ref[None, :, :]).reshape(-1)
        K = coo_matrix((vals, (self.K_rows, self.K_cols)), shape=(self.n_nodes, self.n_nodes)).tocsc()
        Kff = K[self.free_nodes][:, self.free_nodes]
        rhs = self.load_vector[self.free_nodes]
        p = np.zeros(self.n_nodes, dtype=float)
        try:
            p[self.free_nodes] = splu(Kff).solve(rhs)
        except RuntimeError:
            p[self.free_nodes] = spsolve(Kff, rhs)
        return p

    def grad_p_on_cell(self, points: np.ndarray, cell_ids: np.ndarray, p: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        cids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
        b = self.cell_bounds[cids]
        xi = (pts[:, 0] - b[:, 0]) / self.hx
        eta = (pts[:, 1] - b[:, 2]) / self.hy
        grads = self.q1_grads_phys(np.column_stack([xi, eta]))
        p_loc = p[self.cells[cids]]
        return np.einsum("ni,nid->nd", p_loc, grads)

    def face_flux_cg(self, p: np.ndarray, coeff_cell: np.ndarray) -> np.ndarray:
        grad = self.grad_p_on_cell(self.face_q_points, self.face_q_cell_ids, p)
        qv = -coeff_cell[self.face_q_cell_ids, None] * grad
        contrib = np.einsum("ij,ij->i", qv, self.face_q_normal_weights)
        F = np.zeros(len(self.dual_owner), dtype=float)
        np.add.at(F, self.face_q_face_ids, contrib)
        return F

    def q1_values_phys(self, points: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        cids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
        b = self.cell_bounds[cids]
        xi = (pts[:, 0] - b[:, 0]) / self.hx
        eta = (pts[:, 1] - b[:, 2]) / self.hy
        vals = np.empty((len(pts), 4), dtype=float)
        vals[:, 0] = (1.0 - xi) * (1.0 - eta)
        vals[:, 1] = xi * (1.0 - eta)
        vals[:, 2] = (1.0 - xi) * eta
        vals[:, 3] = xi * eta
        return vals

    def precompute_deng_geometry(self) -> None:
        G = np.zeros((self.ncell, 4, 4), dtype=float)
        source_t = np.zeros((self.ncell, 4), dtype=float)
        source_phi = np.zeros((self.ncell, 4), dtype=float)
        mass_vec = np.zeros((self.ncell, 4), dtype=float)
        ephi_points: list[np.ndarray] = []
        ephi_wlen: list[np.ndarray] = []
        ephi_cell_ids: list[np.ndarray] = []
        ephi_neigh_ids: list[np.ndarray] = []
        ephi_normals: list[np.ndarray] = []
        ephi_vals: list[np.ndarray] = []
        ei_points: list[np.ndarray] = []
        ei_wlen: list[np.ndarray] = []
        ei_cell_ids: list[np.ndarray] = []
        ei_neigh_ids: list[np.ndarray] = []
        ei_gi_ids: list[np.ndarray] = []
        ei_normals: list[np.ndarray] = []
        for cid in range(self.ncell):
            source_t[cid], source_phi[cid], mass_vec[cid] = self.deng_source_terms(cid)
            for gi in range(4):
                for a, b, n_out in self.deng_internal_segments(cid, gi):
                    G[cid, gi, :] += self.deng_segment_flux_basis(cid, 1.0, a, b, n_out)

            for side, a, b, n_out in self.deng_physical_edges(cid):
                pts, wlen = self.segment_points_weights(a, b)
                neigh = self.deng_neighbor_cell(cid, side)
                ephi_points.append(pts)
                ephi_wlen.append(wlen)
                ephi_cell_ids.append(np.full(len(pts), cid, dtype=np.int64))
                ephi_neigh_ids.append(np.full(len(pts), -1 if neigh is None else neigh, dtype=np.int64))
                ephi_normals.append(np.tile(np.asarray(n_out, dtype=float), (len(pts), 1)))
                ephi_vals.append(self.q1_values_phys(pts, np.full(len(pts), cid, dtype=np.int64)))

            for gi in range(4):
                for side, a, b, n_out in self.deng_physical_edge_segments(cid, gi):
                    pts, wlen = self.segment_points_weights(a, b)
                    neigh = self.deng_neighbor_cell(cid, side)
                    ei_points.append(pts)
                    ei_wlen.append(wlen)
                    ei_cell_ids.append(np.full(len(pts), cid, dtype=np.int64))
                    ei_neigh_ids.append(np.full(len(pts), -1 if neigh is None else neigh, dtype=np.int64))
                    ei_gi_ids.append(np.full(len(pts), gi, dtype=np.int64))
                    ei_normals.append(np.tile(np.asarray(n_out, dtype=float), (len(pts), 1)))
        self.deng_G = G
        self.deng_G_pinv = np.linalg.pinv(G, rcond=1.0e-10)
        self.deng_source_delta = source_t - source_phi
        self.deng_source_t = source_t
        self.deng_source_phi = source_phi
        self.deng_mass_vec = mass_vec
        self.deng_mass_area = np.sum(mass_vec, axis=1)
        self.deng_ephi_points = np.vstack(ephi_points)
        self.deng_ephi_wlen = np.concatenate(ephi_wlen)
        self.deng_ephi_cell_ids = np.concatenate(ephi_cell_ids)
        self.deng_ephi_neigh_ids = np.concatenate(ephi_neigh_ids)
        self.deng_ephi_normals = np.vstack(ephi_normals)
        self.deng_ephi_vals = np.vstack(ephi_vals)
        self.deng_ei_points = np.vstack(ei_points)
        self.deng_ei_wlen = np.concatenate(ei_wlen)
        self.deng_ei_cell_ids = np.concatenate(ei_cell_ids)
        self.deng_ei_neigh_ids = np.concatenate(ei_neigh_ids)
        self.deng_ei_gi_ids = np.concatenate(ei_gi_ids)
        self.deng_ei_normals = np.vstack(ei_normals)
        self.deng_geometry_ready = True

    @staticmethod
    def solve_deng_singular_system(B: np.ndarray, rhs: np.ndarray, rcond: float = 1.0e-10) -> np.ndarray:
        U, s, Vt = np.linalg.svd(B, full_matrices=False)
        if s.size == 0 or s[0] == 0.0:
            return np.zeros(B.shape[1], dtype=float)
        keep = s > rcond * s[0]
        return Vt[keep, :].T @ ((U[:, keep].T @ rhs) / s[keep])

    def deng_cv_bounds(self, cid: int, gi: int) -> tuple[float, float, float, float]:
        x0, x1, y0, y1 = self.cell_bounds[int(cid)]
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        if gi == 0:
            return x0, xm, y0, ym
        if gi == 1:
            return xm, x1, y0, ym
        if gi == 2:
            return x0, xm, ym, y1
        if gi == 3:
            return xm, x1, ym, y1
        raise ValueError("Q1 Deng local index must be 0, 1, 2, or 3")

    def deng_source_over_subrect(self, cid: int, x_l: float, x_r: float, y_b: float, y_t: float) -> float:
        area = max(0.0, float(x_r) - float(x_l)) * max(0.0, float(y_t) - float(y_b))
        return float(self.source_density_cell[int(cid)] * area)

    def deng_source_terms(self, cid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Match the notebook's local Deng system: source_t is the source
        # integrated over each quarter nodal CV, while source_phi is the Q1
        # load projection over the element. For the current cell-integrated
        # well source these are both rate/4, but keeping the two paths separate
        # avoids silently changing the method for non-uniform source terms.
        source_t = np.zeros(4, dtype=float)
        for gi in range(4):
            source_t[gi] = self.deng_source_over_subrect(cid, *self.deng_cv_bounds(cid, gi))

        q, w = legendre_01(2)
        x0, x1, y0, y1 = self.cell_bounds[int(cid)]
        source_phi = np.zeros(4, dtype=float)
        mass_vec = np.zeros(4, dtype=float)
        for xi, wx in zip(q, w):
            for eta, wy in zip(q, w):
                pt = np.array([[x0 + self.hx * xi, y0 + self.hy * eta]], dtype=float)
                vals = self.q1_values_phys(pt, np.array([cid], dtype=np.int64))[0]
                weight = float(wx * wy * self.cell_area)
                source_phi += weight * self.source_density_cell[int(cid)] * vals
                mass_vec += weight * vals

        defect = float(np.sum(source_phi) - np.sum(source_t))
        area = float(np.sum(mass_vec))
        if area > 0.0:
            source_t = source_t + defect * mass_vec / area
        return source_t, source_phi, mass_vec

    def deng_internal_segments(self, cid: int, gi: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        x0, x1, y0, y1 = self.cell_bounds[int(cid)]
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        if gi == 0:
            return [
                (np.array([xm, y0]), np.array([xm, ym]), np.array([1.0, 0.0])),
                (np.array([x0, ym]), np.array([xm, ym]), np.array([0.0, 1.0])),
            ]
        if gi == 1:
            return [
                (np.array([xm, y0]), np.array([xm, ym]), np.array([-1.0, 0.0])),
                (np.array([xm, ym]), np.array([x1, ym]), np.array([0.0, 1.0])),
            ]
        if gi == 2:
            return [
                (np.array([xm, ym]), np.array([xm, y1]), np.array([1.0, 0.0])),
                (np.array([x0, ym]), np.array([xm, ym]), np.array([0.0, -1.0])),
            ]
        if gi == 3:
            return [
                (np.array([xm, ym]), np.array([xm, y1]), np.array([-1.0, 0.0])),
                (np.array([xm, ym]), np.array([x1, ym]), np.array([0.0, -1.0])),
            ]
        raise ValueError("Q1 Deng local index must be 0, 1, 2, or 3")

    def deng_physical_edges(self, cid: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        x0, x1, y0, y1 = self.cell_bounds[int(cid)]
        return [
            ("bottom", np.array([x0, y0]), np.array([x1, y0]), np.array([0.0, -1.0])),
            ("right", np.array([x1, y0]), np.array([x1, y1]), np.array([1.0, 0.0])),
            ("top", np.array([x0, y1]), np.array([x1, y1]), np.array([0.0, 1.0])),
            ("left", np.array([x0, y0]), np.array([x0, y1]), np.array([-1.0, 0.0])),
        ]

    def deng_physical_edge_segments(self, cid: int, gi: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        x0, x1, y0, y1 = self.cell_bounds[int(cid)]
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        if gi == 0:
            return [
                ("bottom", np.array([x0, y0]), np.array([xm, y0]), np.array([0.0, -1.0])),
                ("left", np.array([x0, y0]), np.array([x0, ym]), np.array([-1.0, 0.0])),
            ]
        if gi == 1:
            return [
                ("bottom", np.array([xm, y0]), np.array([x1, y0]), np.array([0.0, -1.0])),
                ("right", np.array([x1, y0]), np.array([x1, ym]), np.array([1.0, 0.0])),
            ]
        if gi == 2:
            return [
                ("top", np.array([x0, y1]), np.array([xm, y1]), np.array([0.0, 1.0])),
                ("left", np.array([x0, ym]), np.array([x0, y1]), np.array([-1.0, 0.0])),
            ]
        if gi == 3:
            return [
                ("top", np.array([xm, y1]), np.array([x1, y1]), np.array([0.0, 1.0])),
                ("right", np.array([x1, ym]), np.array([x1, y1]), np.array([1.0, 0.0])),
            ]
        raise ValueError("Q1 Deng local index must be 0, 1, 2, or 3")

    def deng_neighbor_cell(self, cid: int, side: str) -> int | None:
        ix = int(cid) % self.nx
        iy = int(cid) // self.nx
        if side == "left" and ix > 0:
            return int(cid) - 1
        if side == "right" and ix < self.nx - 1:
            return int(cid) + 1
        if side == "bottom" and iy > 0:
            return int(cid) - self.nx
        if side == "top" and iy < self.ny - 1:
            return int(cid) + self.nx
        return None

    @staticmethod
    def segment_points_weights(a: np.ndarray, b: np.ndarray, order: int = 3) -> tuple[np.ndarray, np.ndarray]:
        q, w = legendre_01(order)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        edge = b - a
        length = float(np.linalg.norm(edge))
        return a[None, :] + q[:, None] * edge[None, :], w * length

    def deng_segment_flux_basis(self, cid: int, coeff: float, a: np.ndarray, b: np.ndarray, n_out: np.ndarray) -> np.ndarray:
        pts, wlen = self.segment_points_weights(a, b)
        grads = self.grad_basis_on_cell(pts, cid)
        return -coeff * np.einsum("q,qjd,d->j", wlen, grads, n_out)

    def grad_basis_on_cell(self, points: np.ndarray, cid: int) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        b = self.cell_bounds[int(cid)]
        xi = (pts[:, 0] - b[0]) / self.hx
        eta = (pts[:, 1] - b[2]) / self.hy
        return self.q1_grads_phys(np.column_stack([xi, eta]))

    def deng_segment_flux_coeffs(self, cid: int, coeff_cell: np.ndarray, coeffs: np.ndarray, a: np.ndarray, b: np.ndarray, n_out: np.ndarray) -> float:
        pts, wlen = self.segment_points_weights(a, b)
        grads = self.grad_basis_on_cell(pts, cid)
        grad = np.einsum("qjd,j->qd", grads, coeffs)
        return float(-coeff_cell[int(cid)] * np.einsum("q,qd,d->", wlen, grad, n_out))

    def deng_cgrad_dot_n_avg(
        self,
        cid: int,
        side: str,
        points: np.ndarray,
        n_out: np.ndarray,
        p: np.ndarray,
        coeff_cell: np.ndarray,
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        cids = np.full(len(pts), int(cid), dtype=np.int64)
        val = coeff_cell[int(cid)] * self.grad_p_on_cell(pts, cids, p)
        nb = self.deng_neighbor_cell(cid, side)
        if nb is not None:
            nbcids = np.full(len(pts), int(nb), dtype=np.int64)
            val_nb = coeff_cell[int(nb)] * self.grad_p_on_cell(pts, nbcids, p)
            val = 0.5 * (val + val_nb)
        return val @ np.asarray(n_out, dtype=float)

    def deng_cgrad_dot_n_avg_batch(
        self,
        points: np.ndarray,
        cell_ids: np.ndarray,
        neigh_ids: np.ndarray,
        normals: np.ndarray,
        p: np.ndarray,
        coeff_cell: np.ndarray,
    ) -> np.ndarray:
        cids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
        nids = np.asarray(neigh_ids, dtype=np.int64).reshape(-1)
        val = coeff_cell[cids, None] * self.grad_p_on_cell(points, cids, p)
        mask = nids >= 0
        if np.any(mask):
            val_nb = coeff_cell[nids[mask], None] * self.grad_p_on_cell(points[mask], nids[mask], p)
            val[mask] = 0.5 * (val[mask] + val_nb)
        return np.einsum("qd,qd->q", val, normals)

    def deng_local_coefficients(self, p: np.ndarray, coeff_cell: np.ndarray) -> np.ndarray:
        coeff_cell = np.asarray(coeff_cell, dtype=float).reshape(-1)
        u_all = np.asarray(p, dtype=float)[self.cells]
        a_term_all = coeff_cell[:, None] * np.einsum("ij,cj->ci", self.Ke_ref, u_all)
        e_I_all = np.zeros((self.ncell, 4), dtype=float)
        e_phi_all = np.zeros((self.ncell, 4), dtype=float)
        ephi_avg = self.deng_cgrad_dot_n_avg_batch(
            self.deng_ephi_points,
            self.deng_ephi_cell_ids,
            self.deng_ephi_neigh_ids,
            self.deng_ephi_normals,
            p,
            coeff_cell,
        )
        ephi_weighted = self.deng_ephi_wlen * ephi_avg
        for j in range(4):
            np.add.at(e_phi_all[:, j], self.deng_ephi_cell_ids, ephi_weighted * self.deng_ephi_vals[:, j])

        ei_avg = self.deng_cgrad_dot_n_avg_batch(
            self.deng_ei_points,
            self.deng_ei_cell_ids,
            self.deng_ei_neigh_ids,
            self.deng_ei_normals,
            p,
            coeff_cell,
        )
        np.add.at(e_I_all, (self.deng_ei_cell_ids, self.deng_ei_gi_ids), self.deng_ei_wlen * ei_avg)

        rhs = self.deng_source_delta + a_term_all + e_I_all - e_phi_all
        coeffs_rec = np.einsum("cij,cj->ci", self.deng_G_pinv, rhs) / np.maximum(coeff_cell[:, None], 1.0e-300)
        mass_area = np.maximum(self.deng_mass_area, 1.0e-300)
        correction = (
            np.einsum("ci,ci->c", self.deng_mass_vec, u_all)
            - np.einsum("ci,ci->c", self.deng_mass_vec, coeffs_rec)
        ) / mass_area
        coeffs_rec += correction[:, None]
        return coeffs_rec

    def face_flux_deng(self, p: np.ndarray, coeff_cell: np.ndarray) -> np.ndarray:
        if not self.deng_geometry_ready:
            self.precompute_deng_geometry()
        coeffs_rec = self.deng_local_coefficients(p, coeff_cell)
        cids = self.face_q_cell_ids
        grads = self.grad_basis_on_cell_batch(self.face_q_points, cids)
        grad = np.einsum("qjd,qj->qd", grads, coeffs_rec[cids])
        qv = -coeff_cell[cids, None] * grad
        contrib = np.einsum("ij,ij->i", qv, self.face_q_normal_weights)
        F = np.zeros(len(self.dual_owner), dtype=float)
        np.add.at(F, self.face_q_face_ids, contrib)
        return F

    def grad_basis_on_cell_batch(self, points: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        cids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
        b = self.cell_bounds[cids]
        xi = (pts[:, 0] - b[:, 0]) / self.hx
        eta = (pts[:, 1] - b[:, 2]) / self.hy
        return self.q1_grads_phys(np.column_stack([xi, eta]))

    def face_flux_projection(self, F_cg: np.ndarray) -> np.ndarray:
        # Conservative graph projection on the Deng nodal-dual transport mesh.
        # This is not the Deng-Ginting local NLR reconstruction: it is a global
        # minimal-L2 correction that enforces conservation on the dual CVs.
        r = self.dual_residual(F_cg)
        lam = self.graph_lu.solve(r)
        return F_cg - self.B.T @ lam

    def dual_residual(self, F: np.ndarray) -> np.ndarray:
        return self.B @ np.asarray(F, dtype=float).reshape(-1) - self.dual_source_rate

    def source_water_rate(self, S: np.ndarray) -> np.ndarray:
        q = self.dual_source_rate
        return np.where(q >= 0.0, q * fractional_flow(self.cfg.source_saturation, self.cfg.M), q * fractional_flow(S, self.cfg.M))

    def compute_cfl(self, F: np.ndarray, dt: float) -> float:
        F = np.asarray(F, dtype=float)
        up = np.where(F >= 0.0, self.dual_owner, np.where(self.dual_has_neighbor, self.dual_neigh, self.dual_owner))
        ratio = np.max(np.abs(F) / np.maximum(self.dual_pv[up], 1.0e-300))
        return float(dt * ratio * self.fprime_max)

    def advance_transport(self, S: np.ndarray, F: np.ndarray, dt_total: float) -> tuple[np.ndarray, dict[str, float]]:
        n_sub = max(1, int(math.ceil(dt_total / self.cfg.transport_dt)))
        dt = dt_total / n_sub
        S = np.asarray(S, dtype=float).copy()
        min_pre = np.inf
        max_pre = -np.inf
        total_in = 0.0
        total_out = 0.0
        stored0 = float(np.dot(S, self.dual_pv))
        for _ in range(n_sub):
            face_water = np.zeros_like(F)
            internal = self.dual_has_neighbor
            pos = F >= 0.0
            up_internal = np.where(pos[internal], self.dual_owner[internal], self.dual_neigh[internal])
            face_water[internal] = fractional_flow(S[up_internal], self.cfg.M) * F[internal]
            bnd = ~internal
            out_bnd = bnd & (F >= 0.0)
            in_bnd = bnd & (F < 0.0)
            face_water[out_bnd] = fractional_flow(S[self.dual_owner[out_bnd]], self.cfg.M) * F[out_bnd]
            face_water[in_bnd] = fractional_flow(self.cfg.boundary_inflow_s, self.cfg.M) * F[in_bnd]

            divw = np.zeros_like(S)
            np.add.at(divw, self.dual_owner, face_water)
            np.add.at(divw, self.dual_neigh[internal], -face_water[internal])
            sw_src = self.source_water_rate(S)
            S_new = S - dt * divw / self.dual_pv + dt * sw_src / self.dual_pv
            min_pre = min(min_pre, float(np.min(S_new)))
            max_pre = max(max_pre, float(np.max(S_new)))
            if self.cfg.clip_saturation:
                S_new = np.clip(S_new, 0.0, 1.0)

            total_in += dt * (
                float(np.sum(-face_water[in_bnd]))
                + float(np.sum(np.maximum(sw_src, 0.0)))
            )
            total_out += dt * (
                float(np.sum(face_water[out_bnd]))
                + float(np.sum(np.maximum(-sw_src, 0.0)))
            )
            S = S_new
        stored1 = float(np.dot(S, self.dual_pv))
        return S, {
            "n_substeps": int(n_sub),
            "dt_substep": float(dt),
            "S_preclip_min": float(min_pre),
            "S_preclip_max": float(max_pre),
            "S_post_min": float(np.min(S)),
            "S_post_max": float(np.max(S)),
            "water_in": float(total_in),
            "water_out": float(total_out),
            "mass_balance": float(stored1 - stored0 - total_in + total_out),
            "stored_water": stored1,
        }

    def write_csv_header(self) -> None:
        cols = [
            "step", "time", "method", "M", "solve_s", "flux_s", "transport_s", "viz_s",
            "p_l2_drift", "flux_l2_drift", "pinn_rmse", "pinn_pre_rmse", "pinn_iterations", "pinn_wall_s",
            "pinn_pre_eval_s", "pinn_linear_s", "pinn_sync_s", "pinn_error_s",
            "pinn_trainable_params", "pinn_stop_reason",
            "R_xi_mean_abs", "R_xi_median_abs", "R_xi_rmse", "R_xi_max_abs", "R_xi_sum",
            "R_xi_int_mean_abs", "R_xi_int_median_abs", "R_xi_int_rmse", "R_xi_int_max_abs", "R_xi_int_sum",
            "R_xi_src_mean_abs", "R_xi_src_median_abs", "R_xi_src_rmse", "R_xi_src_max_abs", "R_xi_src_sum",
            "R_xi_bnd_mean_abs", "R_xi_bnd_median_abs", "R_xi_bnd_rmse", "R_xi_bnd_max_abs", "R_xi_bnd_sum",
            "CFL", "cfl_violated", "S_preclip_min", "S_preclip_max", "S_post_min", "S_post_max",
            "mass_balance", "stored_water", "n_substeps", "dt_substep",
        ]
        with self.csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
        self.csv_cols = cols

    def append_log(self, row: dict[str, Any]) -> None:
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_cols, extrasaction="ignore")
            writer.writerow({k: row.get(k, "") for k in self.csv_cols})

    def write_implementation_notes(self) -> None:
        notes = f"""# IMPES Simulator Implementation Notes

These choices are simulator implementation details in addition to `impes_simulator_spec.md`.

- Pressure figures use a fixed color scale `[pressure_vmin, pressure_vmax] = [{self.cfg.pressure_vmin:g}, {self.cfg.pressure_vmax:g}]`.
- Flux figures use magnitude-scaled arrows at sampled cell centers, matching the earlier visualization style.
- By default this script runs the Deng/NLR track with `transport_dt = 1e-5`, so each pressure solve is followed by stable explicit transport subcycling over `DT_outer`. Passing `--transport_dt <value>` overrides this.
- By default `N_time = 1000` and `viz_every = N_time`, so the simulator performs 1000 pressure/transport updates and writes one figure at the final step. Passing `--viz_every k` restores intermediate figures.
- By default `save_every = 0`, so pressure/saturation/flux `.npy` snapshots are disabled to avoid large disk usage. Pass `--save_every k` to save every `k` outer steps.
- PINN checkpoints are not written with snapshots unless `--save-pinn-checkpoints` is passed.
- `print_every` controls terminal progress only. CSV logging still records every outer step.
- Figure time labels use the outer IMPES time `t = step * DT_outer`.
- L-BFGS progress is printed on one overwritten terminal line, so the final wall-time and error messages remain readable.
- `pinn_iterations` counts L-BFGS closure calls, including line-search evaluations, so it can exceed `lbfgs_max_iter`.
- The final PINN-CG face-flux error history is saved as `pinn_cg_error_history.csv` and `pinn_cg_error_history.png` when PINN mode is used. The x-axis is the optimizer call/update index, not an Adam epoch, because the simulator updates the network by L-BFGS.
- `pinn_mode="frozen"` evaluates the loaded checkpoint without any update. The frozen face flux is cached after the first evaluation, so this is the no-update baseline for measuring how much the full/last-layer updates help.
- `pinn_mode="k_last_layer"` freezes the early network layers and updates only the final `pinn_k_layers` linear layers by L-BFGS. This is a middle option between full-weight L-BFGS and direct linear last-layer regression.
- `pinn_mode="pou"` loads the saved partition-of-unity linear head and solves its sparse ridge least-squares system each pressure step.  By default the ridge anchor is the previous PoU coefficient vector, so this is the sequential warm-start version.
- `method="NLR"` uses the Deng-Ginting local Q1 postprocessing on element-local nodal control volumes, then assembles the resulting fluxes on the nodal-dual transport mesh.
- `method="PROJ"` is an extra global conservative graph projection on the nodal-dual transport mesh. It is not the Deng-Ginting local NLR reconstruction.
- The Deng source term intentionally keeps the notebook's two local quantities separate: the source integrated over each quarter nodal CV and the Q1 load projection. They cancel for the current equal-split, cell-integrated well source, but the split is retained for auditing and future non-uniform sources.
- Deng reconstruction precomputes the time-independent geometric local matrices and their pseudo-inverses at startup. During each IMPES step the local solve is a batched pseudo-inverse multiply scaled by the current scalar cell coefficient; no per-cell SVD is performed in the time loop.
- `--validate-deng --validate-only` runs the t=0 Deng port gate (`S=0`, `coeff=K`) and reports the dual residual before any IMPES time stepping. `--deng-reference-flux` can compare against a saved frozen-notebook Deng face-flux vector when available.
"""
        (self.out_dir / "IMPLEMENTATION_NOTES.md").write_text(notes)

    def save_step(self, step: int, p: np.ndarray, S: np.ndarray, F: np.ndarray) -> None:
        if self.cfg.save_every <= 0:
            return
        if step % self.cfg.save_every != 0:
            return
        np.save(self.out_dir / f"p_step{step:04d}.npy", p)
        np.save(self.out_dir / f"S_step{step:04d}.npy", S)
        np.save(self.out_dir / f"fluxes_step{step:04d}.npy", F)
        if self.flux_model is not None and self.cfg.save_pinn_checkpoints:
            self.flux_model.save_state(self.out_dir / f"pinn_state_step{step:04d}.pt")

    def visualize(self, step: int, t: float, p: np.ndarray, S: np.ndarray, F: np.ndarray, coeff_cell: np.ndarray) -> None:
        if self.cfg.viz_every <= 0 or step % self.cfg.viz_every != 0:
            return
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
        P = p.reshape(self.ny + 1, self.nx + 1)
        im0 = axes[0].pcolormesh(
            self.x_nodes,
            self.y_nodes,
            P,
            shading="auto",
            cmap="viridis",
            vmin=self.cfg.pressure_vmin,
            vmax=self.cfg.pressure_vmax,
        )
        axes[0].set_title(f"p, t={t:.4g}")
        fig.colorbar(im0, ax=axes[0])
        Sg = S.reshape(self.ny + 1, self.nx + 1)
        im1 = axes[1].pcolormesh(self.x_nodes, self.y_nodes, Sg, shading="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"S, t={t:.4g}")
        fig.colorbar(im1, ax=axes[1])
        cc = self.cell_centers
        grad = self.grad_p_on_cell(cc, np.arange(self.ncell), p)
        q = -coeff_cell[:, None] * grad
        skip = max(1, self.nx // self.cfg.streamline_grid)
        axes[2].quiver(
            cc[::skip, 0],
            cc[::skip, 1],
            q[::skip, 0],
            q[::skip, 1],
            scale=None,
            width=0.002,
        )
        axes[2].set_title("cell-center flux")
        fig.suptitle(
            f"step {step}, t={t:.4g}; DT_outer={self.cfg.DT_outer:g}, transport_dt={self.cfg.transport_dt:g}",
            fontsize=11,
        )
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(False)
        fig.savefig(self.fig_dir / f"step{step:04d}.png", dpi=180)
        plt.close(fig)

    def load_deng_reference_flux(self) -> tuple[np.ndarray, str] | None:
        if not self.cfg.deng_reference_flux:
            return None
        path = Path(self.cfg.deng_reference_flux)
        if not path.exists():
            raise FileNotFoundError(f"Deng reference flux file not found: {path}")
        if path.suffix == ".npy":
            return np.asarray(np.load(path), dtype=float).reshape(-1), path.name
        if path.suffix == ".npz":
            data = np.load(path)
            keys = ["face_flux_deng", "F_deng", "face_flux_nlr", "F_nlr", "dual_flux_nlr", "dual_flux", "face_flux"]
            for key in keys:
                if key in data:
                    return np.asarray(data[key], dtype=float).reshape(-1), f"{path.name}:{key}"
            for key in data.files:
                arr = np.asarray(data[key])
                if arr.size == len(self.dual_owner):
                    return np.asarray(arr, dtype=float).reshape(-1), f"{path.name}:{key}"
        if path.suffix == ".mat":
            data = loadmat(path, squeeze_me=True, struct_as_record=False)
            keys = ["face_flux_deng", "F_deng", "face_flux_nlr", "F_nlr", "dual_flux_nlr", "dual_flux", "face_flux"]
            for key in keys:
                if key in data:
                    return np.asarray(data[key], dtype=float).reshape(-1), f"{path.name}:{key}"
            for key, arr in data.items():
                if key.startswith("__"):
                    continue
                arr_np = np.asarray(arr)
                if arr_np.size == len(self.dual_owner):
                    return np.asarray(arr_np, dtype=float).reshape(-1), f"{path.name}:{key}"
        raise KeyError(f"Could not find a face-flux vector of length {len(self.dual_owner)} in {path}")

    def validate_deng_port(self) -> None:
        coeff = self.kappa_base.copy()
        p = self.solve_pressure(coeff)
        F = self.face_flux_deng(p, coeff)
        rxi = self.dual_residual(F)
        ids = np.arange(self.n_nodes)
        ix = ids % (self.nx + 1)
        iy = ids // (self.nx + 1)
        boundary = (ix == 0) | (ix == self.nx) | (iy == 0) | (iy == self.ny)
        source = np.abs(self.dual_source_rate) > 1.0e-300
        stats_all = stats_dict("R_xi", rxi)
        stats_int = stats_dict("R_xi_int", rxi[~boundary])
        stats_src = stats_dict("R_xi_src", rxi[source])
        stats_bnd = stats_dict("R_xi_bnd", rxi[boundary])
        active_cells = np.flatnonzero(np.abs(self.source_rate_cell) > 1.0e-300)
        source_term_diffs = []
        source_compat_defects = []
        for cid in active_cells:
            source_t, source_phi, _ = self.deng_source_terms(int(cid))
            source_term_diffs.append(float(np.max(np.abs(source_t - source_phi))))
            source_compat_defects.append(float(np.sum(source_phi) - np.sum(source_t)))
        print("Deng t=0 validation gate (S=0, coeff=K):")
        if active_cells.size:
            print(
                "  source split audit: "
                f"max_i|source_t-source_phi|={max(source_term_diffs):.6e} "
                f"max|sum(source_phi)-sum(source_t)|={max(np.abs(source_compat_defects)):.6e} "
                f"over {active_cells.size} well cell(s)"
            )
        print(
            "  R_xi interior gate: "
            f"mean|R|={stats_int['R_xi_int_mean_abs']:.6e} "
            f"median|R|={stats_int['R_xi_int_median_abs']:.6e} "
            f"rmse={stats_int['R_xi_int_rmse']:.6e} "
            f"max|R|={stats_int['R_xi_int_max_abs']:.6e} "
            f"sum(R)={stats_int['R_xi_int_sum']:+.6e}"
        )
        print(
            "  R_xi source nodes:   "
            f"mean|R|={stats_src['R_xi_src_mean_abs']:.6e} "
            f"median|R|={stats_src['R_xi_src_median_abs']:.6e} "
            f"rmse={stats_src['R_xi_src_rmse']:.6e} "
            f"max|R|={stats_src['R_xi_src_max_abs']:.6e} "
            f"sum(R)={stats_src['R_xi_src_sum']:+.6e}"
        )
        print(
            "  R_xi all nodes:      "
            f"mean|R|={stats_all['R_xi_mean_abs']:.6e} "
            f"median|R|={stats_all['R_xi_median_abs']:.6e} "
            f"rmse={stats_all['R_xi_rmse']:.6e} "
            f"max|R|={stats_all['R_xi_max_abs']:.6e} "
            f"sum(R)={stats_all['R_xi_sum']:+.6e}"
        )
        print(
            "  R_xi boundary nodes: "
            f"mean|R|={stats_bnd['R_xi_bnd_mean_abs']:.6e} "
            f"median|R|={stats_bnd['R_xi_bnd_median_abs']:.6e} "
            f"rmse={stats_bnd['R_xi_bnd_rmse']:.6e} "
            f"max|R|={stats_bnd['R_xi_bnd_max_abs']:.6e} "
            f"sum(R)={stats_bnd['R_xi_bnd_sum']:+.6e}"
        )
        ref = self.load_deng_reference_flux()
        if ref is not None:
            F_ref, label = ref
            if F_ref.size != F.size:
                raise ValueError(f"Reference flux has {F_ref.size} entries; expected {F.size}")
            diff = F - F_ref
            dstats = stats_dict("dF", diff)
            print(
                f"  vs reference {label}: "
                f"mean|dF|={dstats['dF_mean_abs']:.6e} "
                f"median|dF|={dstats['dF_median_abs']:.6e} "
                f"rmse={dstats['dF_rmse']:.6e} "
                f"max|dF|={dstats['dF_max_abs']:.6e}"
            )

    def run(self) -> None:
        S = np.zeros(self.n_nodes, dtype=float)
        prev_p = None
        prev_F = None
        cfl_values: list[float] = []
        first_bad = None
        polluted_from = None
        for step in range(1, self.cfg.N_time + 1):
            t = step * self.cfg.DT_outer
            row: dict[str, Any] = {"step": step, "time": t, "method": self.cfg.method, "M": self.cfg.M}

            t0 = time.perf_counter()
            S_cell = self.project_dual_to_cells(S)
            coeff = self.kappa_base * mobility_factor(S_cell, self.cfg.M)
            p = self.solve_pressure(coeff)
            row["solve_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            F_cg = self.face_flux_cg(p, coeff)
            if self.cfg.method == "CG":
                F = F_cg
                pinn_report = {}
            elif self.cfg.method == "PROJ":
                F = self.face_flux_projection(F_cg)
                pinn_report = {}
            elif self.cfg.method == "NLR":
                F = self.face_flux_deng(p, coeff)
                pinn_report = {}
            else:
                assert self.flux_model is not None
                F = self.flux_model.update(F_cg)
                pinn_report = self.flux_model.report()
            row["flux_s"] = time.perf_counter() - t0

            row["p_l2_drift"] = float(np.linalg.norm(p - prev_p)) if prev_p is not None else 0.0
            row["flux_l2_drift"] = float(np.linalg.norm(F - prev_F)) if prev_F is not None else 0.0
            row.update(pinn_report)
            row.setdefault("pinn_rmse", np.nan)
            row.setdefault("pinn_pre_rmse", np.nan)
            row.setdefault("pinn_iterations", "")
            row.setdefault("pinn_wall_s", 0.0)
            row.setdefault("pinn_pre_eval_s", np.nan)
            row.setdefault("pinn_linear_s", np.nan)
            row.setdefault("pinn_sync_s", np.nan)
            row.setdefault("pinn_error_s", np.nan)
            row.setdefault("pinn_trainable_params", 0)
            row.setdefault("pinn_stop_reason", "")

            rxi = self.dual_residual(F)
            row.update(stats_dict("R_xi", rxi))
            row.update(stats_dict("R_xi_int", rxi[self.dual_interior_mask]))
            row.update(stats_dict("R_xi_src", rxi[self.dual_source_mask]))
            row.update(stats_dict("R_xi_bnd", rxi[self.dual_boundary_mask]))
            cfl = self.compute_cfl(F, self.cfg.transport_dt)
            cfl_values.append(cfl)
            violated = cfl > self.cfg.cfl_threshold
            row["CFL"] = cfl
            row["cfl_violated"] = int(violated)
            if violated and first_bad is None:
                first_bad = step
            if violated:
                print(f"WARNING: CFL violation at step {step}: CFL={cfl:.4e} > {self.cfg.cfl_threshold:.4e}")

            t0 = time.perf_counter()
            S, tr = self.advance_transport(S, F, self.cfg.DT_outer)
            row.update(tr)
            row["transport_s"] = time.perf_counter() - t0
            if (tr["S_preclip_min"] < -1.0e-10 or tr["S_preclip_max"] > 1.0 + 1.0e-10) and polluted_from is None:
                polluted_from = step

            t0 = time.perf_counter()
            self.visualize(step, t, p, S, F, coeff)
            row["viz_s"] = time.perf_counter() - t0
            self.save_step(step, p, S, F)
            self.append_log(row)
            pinn_extra = ""
            if self.cfg.method == "PINN":
                pinn_extra = (
                    f" pinn_iter={row['pinn_iterations']} "
                    f"pinn_wall={float(row['pinn_wall_s']):.2f}s "
                    f"pinn_rmse={float(row['pinn_rmse']):.3e} "
                    f"trainable={row['pinn_trainable_params']} "
                    f"stop={row['pinn_stop_reason']}"
                )
            if self.cfg.print_every > 0 and (step == 1 or step == self.cfg.N_time or step % self.cfg.print_every == 0):
                print(
                    f"step {step:04d} t={t:.5g} method={self.cfg.method} "
                    f"CFL={cfl:.3e} "
                    f"Rxi(int/src/bnd/all)="
                    f"{row['R_xi_int_rmse']:.3e}/{row['R_xi_src_rmse']:.3e}/"
                    f"{row['R_xi_bnd_rmse']:.3e}/{row['R_xi_rmse']:.3e} "
                    f"S=[{row['S_post_min']:.3e},{row['S_post_max']:.3e}] "
                    f"wall(solve/flux/transport/viz)="
                    f"{row['solve_s']:.2f}/{row['flux_s']:.2f}/{row['transport_s']:.2f}/{row['viz_s']:.2f}s"
                    f"{pinn_extra}"
                )
            prev_p = p
            prev_F = F

        summary = {
            "n_steps": self.cfg.N_time,
            "max_CFL": float(np.max(cfl_values)) if cfl_values else 0.0,
            "n_cfl_violations": int(np.count_nonzero(np.asarray(cfl_values) > self.cfg.cfl_threshold)),
            "first_cfl_violation_step": first_bad,
            "polluted_from_step": polluted_from,
        }
        with (self.out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, default=json_safe)
        if self.flux_model is not None:
            self.flux_model.save_error_history(self.out_dir)
        print("End-of-run summary:", json.dumps(summary, default=json_safe))


class HardCurlFluxModel:
    def __init__(self, sim: ImpesSpe10Simulator):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PINN mode requires PyTorch, but torch is not importable in this Python environment.")
        self.sim = sim
        self.cfg = sim.cfg
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.freqs = np.array([1, 2, 4, 8, 16, 32], dtype=float)
        self.hidden_dim = 96
        self.depth = 4
        self.use_logk = True
        self.build_logk_feature()
        self.p1 = sim.dual_p1
        self.p2 = sim.dual_p2
        self.normals = sim.dual_normal
        self.sign = torch.as_tensor(self.curl_segment_sign(self.p1, self.p2, self.normals), dtype=self.dtype, device=self.device)
        self.qp_flux_np = self.segment_flux_qp_batch(self.p1, self.p2, self.normals, n_quad=self.cfg.qp_quad_order)
        self.qp_flux = torch.as_tensor(self.qp_flux_np, dtype=self.dtype, device=self.device)
        feat_dim = self.features_np(np.array([[0.5, 0.5]], dtype=float)).shape[1]
        self.model = HardCurlPsiNet(feat_dim, self.hidden_dim, self.depth).to(device=self.device, dtype=self.dtype)
        self.load(Path(self.cfg.t0_checkpoint_path))
        self.feat_a = self.features_torch(torch.as_tensor(self.p1, dtype=self.dtype, device=self.device))
        self.feat_b = self.features_torch(torch.as_tensor(self.p2, dtype=self.dtype, device=self.device))
        self.last_report: dict[str, Any] = {"pinn_rmse": np.nan, "pinn_pre_rmse": np.nan, "pinn_iterations": 0, "pinn_wall_s": 0.0}
        self.last_layer_factor_ready = False
        self.error_history: list[float] = []
        self.frozen_prediction_cache: np.ndarray | None = None
        if self.cfg.pinn_mode == "last_layer":
            self.prepare_last_layer()
        elif self.cfg.pinn_mode == "k_last_layer":
            self.prepare_k_last_layer()
        elif self.cfg.pinn_mode == "pou":
            self.prepare_pou_head()

    def load(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"PINN checkpoint not found: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        arch = ckpt.get("architecture", {})
        self.hidden_dim = int(arch.get("hidden_dim", self.hidden_dim))
        self.depth = int(arch.get("depth", self.depth))
        self.model.load_state_dict({k: v.to(device=self.device, dtype=self.dtype) for k, v in ckpt["state_dict"].items()})
        self.model.eval()
        print(f"Loaded hard-curl PINN checkpoint: {checkpoint_path}")

    def build_logk_feature(self) -> None:
        logk_cell = np.log10(np.maximum(self.sim.kappa_base, 1.0e-300)).reshape(self.sim.ny, self.sim.nx)
        pad = np.pad(logk_cell, 1, mode="edge")
        self.logk_node = 0.25 * (pad[:-1, :-1] + pad[:-1, 1:] + pad[1:, :-1] + pad[1:, 1:])
        vals = self.logk_node.reshape(-1)
        self.logk_mean = float(np.mean(vals))
        self.logk_std = float(np.std(vals) + 1.0e-12)

    def logk_feature_torch(self, x: torch.Tensor) -> torch.Tensor:
        gx = (x[:, 0] - self.sim.xmin) / self.sim.hx
        gy = (x[:, 1] - self.sim.ymin) / self.sim.hy
        ix = torch.clamp(torch.floor(gx.detach()).long(), 0, self.sim.nx - 1)
        iy = torch.clamp(torch.floor(gy.detach()).long(), 0, self.sim.ny - 1)
        s = torch.clamp(gx - ix.to(dtype=x.dtype), 0.0, 1.0)
        t = torch.clamp(gy - iy.to(dtype=x.dtype), 0.0, 1.0)
        tab = torch.as_tensor(self.logk_node, dtype=x.dtype, device=x.device)
        val = (
            (1.0 - s) * (1.0 - t) * tab[iy, ix]
            + s * (1.0 - t) * tab[iy, ix + 1]
            + (1.0 - s) * t * tab[iy + 1, ix]
            + s * t * tab[iy + 1, ix + 1]
        )
        return ((val - self.logk_mean) / self.logk_std).unsqueeze(1)

    def features_torch(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x[:, 0:1], x[:, 1:2]]
        for freq in self.freqs:
            argx = x.new_tensor(2.0 * np.pi * float(freq)) * x[:, 0:1]
            argy = x.new_tensor(2.0 * np.pi * float(freq)) * x[:, 1:2]
            feats.extend([torch.sin(argx), torch.cos(argx), torch.sin(argy), torch.cos(argy)])
        if self.use_logk:
            feats.append(self.logk_feature_torch(x))
        return torch.cat(feats, dim=1)

    def features_np(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        x = pts[:, 0:1]
        y = pts[:, 1:2]
        feats = [x, y]
        for freq in self.freqs:
            feats.extend([
                np.sin(2.0 * np.pi * freq * x),
                np.cos(2.0 * np.pi * freq * x),
                np.sin(2.0 * np.pi * freq * y),
                np.cos(2.0 * np.pi * freq * y),
            ])
        gx = (pts[:, 0] - self.sim.xmin) / self.sim.hx
        gy = (pts[:, 1] - self.sim.ymin) / self.sim.hy
        ix = np.floor(np.clip(gx, 0.0, np.nextafter(float(self.sim.nx), 0.0))).astype(int)
        iy = np.floor(np.clip(gy, 0.0, np.nextafter(float(self.sim.ny), 0.0))).astype(int)
        s = np.clip(gx - ix, 0.0, 1.0)
        t = np.clip(gy - iy, 0.0, 1.0)
        logk = (
            (1.0 - s) * (1.0 - t) * self.logk_node[iy, ix]
            + s * (1.0 - t) * self.logk_node[iy, ix + 1]
            + (1.0 - s) * t * self.logk_node[iy + 1, ix]
            + s * t * self.logk_node[iy + 1, ix + 1]
        )
        feats.append(((logk - self.logk_mean) / self.logk_std).reshape(-1, 1))
        return np.hstack(feats)

    @staticmethod
    def curl_segment_sign(p1: np.ndarray, p2: np.ndarray, normals: np.ndarray) -> np.ndarray:
        edge = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
        length = np.linalg.norm(edge, axis=1)
        right_normal = np.column_stack([edge[:, 1], -edge[:, 0]]) / np.maximum(length[:, None], 1.0e-300)
        sign = np.einsum("ij,ij->i", np.asarray(normals, dtype=float), right_normal)
        return np.where(sign >= 0.0, 1.0, -1.0)

    @staticmethod
    def qp_antiderivative(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        r2 = u * u + v * v
        out = np.zeros_like(r2)
        mask = r2 > 0.0
        out[mask] = 0.5 * v[mask] * np.log(r2[mask]) + u[mask] * np.arctan2(v[mask], u[mask])
        return out

    def q_p_numpy(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        qx = np.zeros(len(pts), dtype=float)
        qy = np.zeros(len(pts), dtype=float)
        for cid in np.flatnonzero(np.abs(self.sim.source_rate_cell) > 1.0e-300):
            x0, x1, y0, y1 = self.sim.cell_bounds[int(cid)]
            rho = self.sim.source_rate_cell[int(cid)] / self.sim.cell_area
            xa = x - x0
            xb = x - x1
            yc = y - y0
            yd = y - y1
            qx += -rho / (2.0 * np.pi) * (
                self.qp_antiderivative(xa, yc) - self.qp_antiderivative(xa, yd)
                - self.qp_antiderivative(xb, yc) + self.qp_antiderivative(xb, yd)
            )
            qy += -rho / (2.0 * np.pi) * (
                self.qp_antiderivative(yc, xa) - self.qp_antiderivative(yc, xb)
                - self.qp_antiderivative(yd, xa) + self.qp_antiderivative(yd, xb)
            )
        return np.column_stack([qx, qy])

    def segment_flux_qp_batch(self, p1: np.ndarray, p2: np.ndarray, normals: np.ndarray, n_quad: int) -> np.ndarray:
        xi, wi = legendre_01(n_quad)
        edge = p2 - p1
        length = np.linalg.norm(edge, axis=1)
        flux = np.zeros(len(p1), dtype=float)
        for s, w in zip(xi, wi):
            pts = p1 + s * edge
            qp = self.q_p_numpy(pts)
            flux += w * length * np.einsum("ij,ij->i", qp, normals)
        return flux

    def prediction_torch(self) -> torch.Tensor:
        psi_a = self.model(self.feat_a)
        psi_b = self.model(self.feat_b)
        return self.qp_flux + self.sign * (psi_b - psi_a)

    def prediction_numpy(self) -> np.ndarray:
        with torch.no_grad():
            return self.prediction_torch().detach().cpu().numpy()

    def pre_update_rmse(self, target: np.ndarray) -> float:
        err = self.prediction_numpy() - np.asarray(target, dtype=float)
        return float(np.sqrt(np.mean(err * err)))

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def linear_layers(self) -> list[nn.Linear]:
        return [layer for layer in self.model.net.children() if isinstance(layer, nn.Linear)]

    @staticmethod
    def _checkpoint_array(obj: Any) -> np.ndarray:
        if TORCH_AVAILABLE and torch.is_tensor(obj):
            return obj.detach().cpu().numpy()
        return np.asarray(obj)

    @staticmethod
    def _cosine_bump_1d(x: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        centers = np.asarray(centers, dtype=np.float64).reshape(1, -1)
        z = np.abs(x - centers) / float(radius)
        out = np.zeros_like(z, dtype=np.float64)
        mask = z < 1.0
        out[mask] = 0.5 * (1.0 + np.cos(np.pi * z[mask]))
        return out

    def pou_window_weights(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        nxw, nyw = self.pou_window_shape
        cx = np.linspace(0.0, 1.0, nxw) if nxw > 1 else np.array([0.5])
        cy = np.linspace(0.0, 1.0, nyw) if nyw > 1 else np.array([0.5])
        rx = (1.0 / max(nxw - 1, 1)) / max(1.0 - self.pou_overlap, 1.0e-12) * 0.5
        ry = (1.0 / max(nyw - 1, 1)) / max(1.0 - self.pou_overlap, 1.0e-12) * 0.5
        wx = np.ones((len(pts), 1), dtype=np.float64) if nxw == 1 else self._cosine_bump_1d(pts[:, 0], cx, rx)
        wy = np.ones((len(pts), 1), dtype=np.float64) if nyw == 1 else self._cosine_bump_1d(pts[:, 1], cy, ry)
        W = (wy[:, :, None] * wx[:, None, :]).reshape(len(pts), nyw * nxw)
        denom = np.sum(W, axis=1, keepdims=True)
        if np.any(denom <= 0.0):
            bad = np.flatnonzero(denom[:, 0] <= 0.0)[:10]
            raise RuntimeError(f"PoU windows do not cover endpoint(s): {bad}")
        return W / denom

    def prepare_pou_head(self) -> None:
        path = Path(self.cfg.pou_checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"PoU checkpoint not found: {path}. Run "
                "LCG_DengGinting_example4_spe10_Q1_PoU_head.ipynb to save it first."
            )
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch
            ckpt = torch.load(path, map_location="cpu")
        if "frozen_state_dict" in ckpt:
            self.model.load_state_dict({k: v.to(device=self.device, dtype=self.dtype) for k, v in ckpt["frozen_state_dict"].items()})
            self.model.eval()
        if "pou" not in ckpt:
            raise KeyError(f"{path} does not contain a 'pou' checkpoint dictionary")
        pou = ckpt["pou"]
        self.pou_checkpoint_path = path
        self.pou_window_shape = tuple(int(v) for v in pou["window_shape"])
        self.pou_overlap = float(pou.get("overlap", 0.5))
        self.pou_P = np.asarray(self._checkpoint_array(pou["P"]), dtype=np.float64)
        self.pou_r = int(pou.get("r", self.pou_P.shape[1]))
        self.pou_theta = np.asarray(self._checkpoint_array(pou["theta"]), dtype=np.float64).reshape(-1, self.pou_r)
        self.pou_theta_bar = np.asarray(self._checkpoint_array(pou["theta_bar"]), dtype=np.float64).reshape(self.pou_theta.shape)
        self.pou_K = int(np.prod(self.pou_window_shape))
        if self.pou_theta.shape != (self.pou_K, self.pou_r):
            raise ValueError(f"PoU theta has shape {self.pou_theta.shape}, expected {(self.pou_K, self.pou_r)}")
        if self.pou_P.shape != (self.hidden_dim, self.pou_r):
            raise ValueError(f"PoU P has shape {self.pou_P.shape}, expected {(self.hidden_dim, self.pou_r)}")

        t0 = time.perf_counter()
        with torch.no_grad():
            ha = self.model.hidden_features(self.feat_a).detach().cpu().numpy().astype(np.float64)
            hb = self.model.hidden_features(self.feat_b).detach().cpu().numpy().astype(np.float64)
        Wa = self.pou_window_weights(self.p1)
        Wb = self.pou_window_weights(self.p2)
        phi_a = ha @ self.pou_P
        phi_b = hb @ self.pou_P
        sign_np = self.sign.detach().cpu().numpy().astype(np.float64)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        ar = np.arange(self.pou_r, dtype=np.int64)
        for f in range(len(self.p1)):
            for W, phi, sgn in ((Wb, phi_b, sign_np[f]), (Wa, phi_a, -sign_np[f])):
                active = np.flatnonzero(W[f] > 1.0e-14)
                if active.size == 0:
                    continue
                block_cols = active[:, None] * self.pou_r + ar[None, :]
                block_vals = sgn * W[f, active][:, None] * phi[f][None, :]
                rows.append(np.full(block_cols.size, f, dtype=np.int64))
                cols.append(block_cols.reshape(-1).astype(np.int64))
                vals.append(block_vals.reshape(-1).astype(np.float64))
        self.pou_Phi = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(self.p1), self.pou_K * self.pou_r),
            dtype=np.float64,
        ).tocsc()
        self.pou_build_s = time.perf_counter() - t0

        t_fac = time.perf_counter()
        normal = (self.pou_Phi.T @ self.pou_Phi).tocsc()
        self.pou_diag_max = float(np.max(np.abs(normal.diagonal())))
        self.pou_ridge_rel = float(pou.get("ridge_lambda_rel", 1.0e-8))
        self.pou_ridge_abs = self.pou_ridge_rel * max(self.pou_diag_max, 1.0)
        ndof = normal.shape[0]
        diag = coo_matrix(
            (np.full(ndof, self.pou_ridge_abs, dtype=np.float64), (np.arange(ndof), np.arange(ndof))),
            shape=(ndof, ndof),
            dtype=np.float64,
        ).tocsc()
        self.pou_lu = splu(normal + diag)
        self.pou_factor_s = time.perf_counter() - t_fac
        self.pou_dof = int(ndof)
        warm_err = self.pou_prediction(self.pou_theta_bar) - self.prediction_numpy()
        warm_max = float(np.max(np.abs(warm_err)))
        if warm_max > 1.0e-10:
            raise RuntimeError(f"PoU warm-start identity check failed: max face-flux error {warm_max:.3e}")
        print(
            f"Loaded PoU head: windows={self.pou_window_shape}, r={self.pou_r}, dof={self.pou_dof}, "
            f"ridge_rel={self.pou_ridge_rel:.1e}, build={self.pou_build_s:.3f}s, "
            f"factor={self.pou_factor_s:.3f}s, warm_max={warm_max:.3e}"
        )

    def pou_prediction(self, theta: np.ndarray | None = None) -> np.ndarray:
        if theta is None:
            theta = self.pou_theta
        theta_vec = np.asarray(theta, dtype=np.float64).reshape(-1)
        return self.qp_flux_np + np.asarray(self.pou_Phi @ theta_vec).reshape(-1)

    def update_pou(self, target: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        target_np = np.asarray(target, dtype=np.float64).reshape(-1)
        pred_old = self.pou_prediction(self.pou_theta)
        pre_rmse = float(np.sqrt(np.mean((pred_old - target_np) ** 2)))
        anchor = self.pou_theta if self.cfg.pou_anchor_mode == "previous" else self.pou_theta_bar
        t_linear = time.perf_counter()
        rhs = np.asarray(self.pou_Phi.T @ (target_np - self.qp_flux_np)).reshape(-1)
        rhs += self.pou_ridge_abs * np.asarray(anchor, dtype=np.float64).reshape(-1)
        theta_vec = self.pou_lu.solve(rhs)
        self.pou_theta = theta_vec.reshape(self.pou_K, self.pou_r)
        pred = self.qp_flux_np + np.asarray(self.pou_Phi @ theta_vec).reshape(-1)
        linear_s = time.perf_counter() - t_linear
        err = pred - target_np
        rmse = float(np.sqrt(np.mean(err * err)))
        self.error_history.append(rmse)
        self.last_report = {
            "pinn_rmse": rmse,
            "pinn_pre_rmse": pre_rmse,
            "pinn_iterations": "direct",
            "pinn_wall_s": float(time.perf_counter() - t0),
            "pinn_pre_eval_s": 0.0,
            "pinn_linear_s": linear_s,
            "pinn_sync_s": 0.0,
            "pinn_error_s": 0.0,
            "pinn_trainable_params": self.pou_dof,
            "pinn_stop_reason": f"direct_pou_{self.cfg.pou_anchor_mode}",
        }
        return pred

    def prepare_k_last_layer(self) -> None:
        linear_layers = self.linear_layers()
        k = max(1, min(int(self.cfg.pinn_k_layers), len(linear_layers)))
        for param in self.model.parameters():
            param.requires_grad_(False)
        for layer in linear_layers[-k:]:
            for param in layer.parameters():
                param.requires_grad_(True)
        self.k_last_layers_active = k
        print(
            f"K-last-layer PINN mode: optimizing final {k} linear layer(s), "
            f"trainable parameters={self.trainable_parameter_count()} / "
            f"{sum(p.numel() for p in self.model.parameters())}"
        )

    def prepare_last_layer(self) -> None:
        with torch.no_grad():
            ha = self.model.hidden_features(self.feat_a).detach().cpu().numpy()
            hb = self.model.hidden_features(self.feat_b).detach().cpu().numpy()
        self.Phi = self.sign.detach().cpu().numpy()[:, None] * (hb - ha)
        lhs = self.Phi.T @ self.Phi + self.cfg.ridge_last_layer * np.eye(self.Phi.shape[1])
        self.Phi_lu = splu(csc_matrix(lhs))
        self.last_layer_factor_ready = True

    def update_last_layer(self, target: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        target_np = np.asarray(target, dtype=float)
        t_pre = time.perf_counter()
        final = list(self.model.net.children())[-1]
        with torch.no_grad():
            w_old = final.weight.detach().cpu().numpy().reshape(-1)
        pred_old = self.qp_flux_np + self.Phi @ w_old
        pre_rmse = float(np.sqrt(np.mean((pred_old - target_np) ** 2)))
        pre_s = time.perf_counter() - t_pre
        t_linear = time.perf_counter()
        rhs = self.Phi.T @ (target_np - self.qp_flux_np)
        w = self.Phi_lu.solve(rhs)
        pred = self.qp_flux_np + self.Phi @ w
        linear_s = time.perf_counter() - t_linear
        # Keep the torch final layer synchronized for saved state/debugging.
        t_sync = time.perf_counter()
        with torch.no_grad():
            final.weight[:] = torch.as_tensor(w.reshape(1, -1), dtype=self.dtype)
            final.bias[:] = 0.0
        sync_s = time.perf_counter() - t_sync
        t_err = time.perf_counter()
        err = pred - target_np
        rmse = float(np.sqrt(np.mean(err * err)))
        error_s = time.perf_counter() - t_err
        self.error_history.append(rmse)
        self.last_report = {
            "pinn_rmse": rmse,
            "pinn_pre_rmse": pre_rmse,
            "pinn_iterations": "direct",
            "pinn_wall_s": float(time.perf_counter() - t0),
            "pinn_pre_eval_s": pre_s,
            "pinn_linear_s": linear_s,
            "pinn_sync_s": sync_s,
            "pinn_error_s": error_s,
            "pinn_trainable_params": int(self.Phi.shape[1]),
            "pinn_stop_reason": "direct_last_layer",
        }
        return pred

    def update_frozen(self, target: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if self.frozen_prediction_cache is None:
            self.frozen_prediction_cache = self.prediction_numpy()
        pred = self.frozen_prediction_cache
        err = pred - np.asarray(target, dtype=float)
        rmse = float(np.sqrt(np.mean(err * err)))
        self.error_history.append(rmse)
        self.last_report = {
            "pinn_rmse": rmse,
            "pinn_pre_rmse": rmse,
            "pinn_iterations": "frozen",
            "pinn_wall_s": float(time.perf_counter() - t0),
            "pinn_pre_eval_s": 0.0,
            "pinn_linear_s": 0.0,
            "pinn_sync_s": 0.0,
            "pinn_error_s": 0.0,
            "pinn_trainable_params": 0,
            "pinn_stop_reason": "no_update",
        }
        return pred

    def update_full(self, target: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        target_np = np.asarray(target, dtype=float)
        t_pre = time.perf_counter()
        pre_rmse = self.pre_update_rmse(target_np)
        pre_s = time.perf_counter() - t_pre
        target_t = torch.as_tensor(target_np, dtype=self.dtype, device=self.device)
        weights = torch.ones_like(target_t)
        best_score = [np.inf]
        best_state = [None]
        calls = [0]
        last_improve = [0]
        status_printed = [False]
        stop_reason = ["optimizer_done"]

        def loss_fn() -> tuple[torch.Tensor, torch.Tensor]:
            pred = self.prediction_torch()
            err = pred - target_t
            return torch.mean(weights * err * err), torch.mean(err * err)

        class EarlyStop(Exception):
            pass

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable PINN parameters are active for L-BFGS.")

        opt = torch.optim.LBFGS(
            trainable_params,
            max_iter=int(self.cfg.lbfgs_max_iter),
            history_size=int(self.cfg.lbfgs_history_size),
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            loss, mse = loss_fn()
            loss.backward()
            mse_val = float(mse.detach().cpu())
            rmse_val = math.sqrt(mse_val)
            self.error_history.append(rmse_val)
            calls[0] += 1
            if np.isinf(best_score[0]):
                improve_tol = 0.0
            else:
                improve_tol = max(
                    float(self.cfg.lbfgs_min_delta),
                    float(self.cfg.lbfgs_rel_min_delta) * max(float(best_score[0]), 1.0e-300),
                )
            if mse_val < best_score[0] - improve_tol:
                best_score[0] = mse_val
                best_state[0] = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                last_improve[0] = calls[0]
            if calls[0] == 1 or calls[0] % self.cfg.lbfgs_print_every == 0:
                print(
                    f"\r  L-BFGS call {calls[0]:5d}: "
                    f"face-rmse={rmse_val:.6e}, best={math.sqrt(best_score[0]):.6e}, "
                    f"elapsed={time.perf_counter() - t0:.1f}s",
                    end="",
                    flush=True,
                )
                status_printed[0] = True
            if calls[0] >= self.cfg.lbfgs_min_calls and calls[0] - last_improve[0] >= self.cfg.lbfgs_early_stop_patience:
                stop_reason[0] = "relative_patience"
                raise EarlyStop
            return loss

        early_stopped = False
        try:
            opt.step(closure)
        except EarlyStop:
            early_stopped = True
        if status_printed[0]:
            print()
        if early_stopped:
            print(f"  L-BFGS early stop after {calls[0]} calls ({stop_reason[0]})", flush=True)
        if best_state[0] is not None:
            self.model.load_state_dict(best_state[0])
        pred = self.prediction_numpy()
        err = pred - target_np
        self.last_report = {
            "pinn_rmse": float(np.sqrt(np.mean(err * err))),
            "pinn_pre_rmse": pre_rmse,
            "pinn_iterations": int(calls[0]),
            "pinn_wall_s": float(time.perf_counter() - t0),
            "pinn_pre_eval_s": pre_s,
            "pinn_linear_s": np.nan,
            "pinn_sync_s": np.nan,
            "pinn_error_s": np.nan,
            "pinn_trainable_params": self.trainable_parameter_count(),
            "pinn_stop_reason": stop_reason[0],
        }
        return pred

    def update(self, target_face_fluxes: np.ndarray) -> np.ndarray:
        if self.cfg.pinn_mode == "frozen":
            return self.update_frozen(target_face_fluxes)
        if self.cfg.pinn_mode == "last_layer":
            return self.update_last_layer(target_face_fluxes)
        if self.cfg.pinn_mode == "pou":
            return self.update_pou(target_face_fluxes)
        return self.update_full(target_face_fluxes)

    def report(self) -> dict[str, Any]:
        return dict(self.last_report)

    def save_state(self, path: Path) -> None:
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}, path)

    def save_error_history(self, out_dir: Path) -> None:
        if not self.error_history:
            return
        idx = np.arange(1, len(self.error_history) + 1, dtype=int)
        err = np.asarray(self.error_history, dtype=float)
        csv_path = out_dir / "pinn_cg_error_history.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["optimizer_call", "face_rmse_vs_cg"])
            writer.writerows(zip(idx, err))
        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        ax.semilogy(idx, err, lw=1.8)
        ax.set_xlabel("optimizer call/update")
        ax.set_ylabel("PINN-CG face RMSE")
        ax.set_title("PINN flux error history")
        ax.grid(False)
        fig.savefig(out_dir / "pinn_cg_error_history.png", dpi=180)
        plt.close(fig)


def normalize_method_name(method: str) -> str:
    return str(method).upper()


def normalize_pinn_mode_name(mode: str) -> str:
    mode_norm = str(mode).lower().replace("-", "_")
    return "frozen" if mode_norm == "no_update" else mode_norm


def parse_args() -> ImpesConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="NLR", choices=["CG", "NLR", "PROJ", "PINN"], help="NLR is Deng-Ginting local reconstruction; PROJ is the extra global projection baseline.")
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--N_time", type=int, default=1000)
    parser.add_argument("--DT_outer", type=float, default=0.01)
    parser.add_argument("--transport_dt", type=float, default=1.0e-5, help="Transport substep. Default: 1e-5 for stable NLR explicit transport.")
    parser.add_argument("--cfl_threshold", type=float, default=1.0)
    parser.add_argument("--pinn_mode", default="full", choices=["full", "last_layer", "k_last_layer", "k-last-layer", "frozen", "no_update", "no-update", "pou", "pou_head", "pou-head"])
    parser.add_argument("--pinn_k_layers", type=int, default=2, help="Number of final linear layers optimized when --pinn_mode k_last_layer.")
    parser.add_argument("--lbfgs_max_iter", type=int, default=1000)
    parser.add_argument("--lbfgs_history_size", type=int, default=50)
    parser.add_argument("--lbfgs_print_every", type=int, default=25)
    parser.add_argument("--lbfgs_early_stop_patience", type=int, default=75)
    parser.add_argument("--lbfgs_min_delta", type=float, default=1.0e-10)
    parser.add_argument("--lbfgs_rel_min_delta", type=float, default=1.0e-4)
    parser.add_argument("--lbfgs_min_calls", type=int, default=50)
    parser.add_argument("--t0_checkpoint_path", default=str(CASE_DIR / "hardcurl_pinn_spe10_Q1_64x64.pt"))
    parser.add_argument("--pou_checkpoint_path", default=str(CASE_DIR / "hardcurl_pinn_spe10_Q1_64x64_pou.pt"))
    parser.add_argument("--pou_anchor_mode", default="previous", choices=["previous", "theta_bar", "theta-bar"])
    parser.add_argument("--data_file", default=str(CASE_DIR / "case3_mrst_export_spe10_L20_64_wells.mat"))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--viz_every", type=int, default=None, help="Visualization interval. Default: N_time, i.e. one final plot.")
    parser.add_argument("--full_conservation_every", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=0, help="Save .npy snapshots every k outer steps. Default 0 disables snapshots.")
    parser.add_argument("--print_every", type=int, default=1, help="Print progress every k outer steps. Default 1 prints every step; 0 disables step prints.")
    parser.add_argument("--pressure_vmin", type=float, default=-1.0)
    parser.add_argument("--pressure_vmax", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--validate-deng", action="store_true", help="Run the t=0 Deng/NLR port validation gate.")
    parser.add_argument("--validate-only", action="store_true", help="Run requested validation gates and exit before the IMPES loop.")
    parser.add_argument("--deng-reference-flux", default="", help="Optional .npy/.npz/.mat frozen-notebook Deng face-flux vector for validation.")
    parser.add_argument("--save-pinn-checkpoints", action="store_true", help="Also save PINN state files at snapshot steps.")
    args = parser.parse_args()
    mode_for_dir = normalize_pinn_mode_name(args.pinn_mode)
    method_for_dir = normalize_method_name(args.method)
    run_name = f"run_{method_for_dir}_M{args.M:g}"
    if method_for_dir == "PINN" and mode_for_dir == "last_layer":
        run_name += "_last_layer"
    elif method_for_dir == "PINN" and mode_for_dir == "k_last_layer":
        run_name += f"_k_last{int(args.pinn_k_layers)}"
    elif method_for_dir == "PINN" and mode_for_dir == "frozen":
        run_name += "_frozen"
    elif method_for_dir == "PINN" and mode_for_dir in {"pou", "pou_head"}:
        run_name += "_pou"
    out_dir = args.out_dir or str(ROOT / "impes_runs" / run_name)
    return ImpesConfig(
        method=args.method,
        M=args.M,
        N_time=args.N_time,
        DT_outer=args.DT_outer,
        transport_dt=args.transport_dt,
        cfl_threshold=args.cfl_threshold,
        pinn_mode=args.pinn_mode,
        pinn_k_layers=args.pinn_k_layers,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_history_size=args.lbfgs_history_size,
        lbfgs_print_every=args.lbfgs_print_every,
        lbfgs_early_stop_patience=args.lbfgs_early_stop_patience,
        lbfgs_min_delta=args.lbfgs_min_delta,
        lbfgs_rel_min_delta=args.lbfgs_rel_min_delta,
        lbfgs_min_calls=args.lbfgs_min_calls,
        t0_checkpoint_path=args.t0_checkpoint_path,
        pou_checkpoint_path=args.pou_checkpoint_path,
        pou_anchor_mode=args.pou_anchor_mode,
        data_file=args.data_file,
        out_dir=out_dir,
        viz_every=args.viz_every,
        full_conservation_every=args.full_conservation_every,
        save_every=args.save_every,
        print_every=args.print_every,
        pressure_vmin=args.pressure_vmin,
        pressure_vmax=args.pressure_vmax,
        dry_run=args.dry_run,
        clip_saturation=not args.no_clip,
        validate_deng=args.validate_deng,
        validate_only=args.validate_only,
        deng_reference_flux=args.deng_reference_flux,
        save_pinn_checkpoints=args.save_pinn_checkpoints,
    )


def main() -> None:
    cfg = parse_args()
    sim = ImpesSpe10Simulator(cfg)
    if cfg.validate_deng:
        sim.validate_deng_port()
    if cfg.validate_only:
        return
    sim.run()


if __name__ == "__main__":
    main()
