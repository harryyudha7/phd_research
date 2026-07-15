"""Y-fracture SPE10 Case 4 support code.

The module is intentionally additive: it does not alter the single-fracture
helpers or any established notebook.  It provides the multi-branch geometry,
Q1--P1--P0 LMDFM block solve, median-dual geometry, exact hard-curl
particular fields, audits, NLR reconstruction, PoU refresh, and one-step
transport primitives used by ``LCG_Yfracture_spe10_L20_hardcurl_PINN_NLR``.
"""

from __future__ import annotations

import ast
import json
import math
import platform
import re
import time
from types import SimpleNamespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.sparse.linalg import splu, spsolve

import torch
import torch.nn as nn

import fracture_hardcurl_common as hc


DTYPE = np.float64
TORCH_DTYPE = torch.float64
SEED = 1729
ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "LCG_DengGinting_example4_spe10_Q1_no_fracture_hardcurl_PINN.ipynb"


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=DTYPE)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero vector")
    return v / n


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _poly_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=DTYPE)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))


def _ordered_poly(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=DTYPE)
    c = np.mean(p, axis=0)
    out = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    return out if _poly_area(out) > 0.0 else out[::-1]


def _gauss01(order: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * (x + 1.0), 0.5 * w


def _extract_literal_assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:
            return ast.literal_eval(value.args[0])
        return ast.literal_eval(value)
    raise KeyError(f"could not extract {name} from {SOURCE_NOTEBOOK.name}")


def _extract_data_candidates(source: str) -> list[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "DATA_CANDIDATES" for t in node.targets):
            continue
        out = []
        for item in node.value.elts:
            if isinstance(item, ast.Call) and item.args:
                out.append(str(ast.literal_eval(item.args[0])))
        return out
    raise KeyError("DATA_CANDIDATES not found in source notebook")


@dataclass
class Case1Config:
    data_file: Path
    source_notebook: Path
    source_points_requested: np.ndarray
    source_rates: np.ndarray
    source_cell_ids: np.ndarray
    source_points_effective: np.ndarray
    source_rate_cell: np.ndarray
    kappa_cell: np.ndarray
    xc_matrix: np.ndarray
    nx: int
    ny: int
    hx: float
    hy: float
    layer: int
    cell_area: float


def load_case1_configuration() -> Case1Config:
    """Read wells and export selection from the established Case-1 notebook."""
    nb = json.loads(SOURCE_NOTEBOOK.read_text())
    setup = "".join(nb["cells"][2]["source"])
    source_points = np.asarray(_extract_literal_assignment(setup, "SOURCE_POINTS_REQUESTED"), dtype=DTYPE)
    source_rates = np.asarray(_extract_literal_assignment(setup, "SOURCE_RATES"), dtype=DTYPE)
    candidates = _extract_data_candidates(setup)
    data_file = next((ROOT / rel for rel in candidates if (ROOT / rel).exists()), None)
    if data_file is None:
        raise FileNotFoundError("none of the Case-1 SPE10 exports exists")
    raw = loadmat(data_file, squeeze_me=True, struct_as_record=False)
    xc = np.asarray(raw["xc_matrix"], dtype=DTYPE).reshape(-1, 2)
    kappa = np.asarray(raw["kappa_cell"], dtype=DTYPE).reshape(-1)
    n = int(round(math.sqrt(len(kappa))))
    if n * n != len(kappa):
        raise ValueError("Case-1 export is not a square tensor grid")
    nx = ny = n
    hx = 1.0 / nx
    hy = 1.0 / ny
    xp = np.clip(source_points[:, 0], 0.0, np.nextafter(1.0, 0.0))
    yp = np.clip(source_points[:, 1], 0.0, np.nextafter(1.0, 0.0))
    ix = np.floor(xp / hx).astype(int)
    iy = np.floor(yp / hy).astype(int)
    source_ids = iy * nx + ix
    source_rate_cell = np.zeros(nx * ny, dtype=DTYPE)
    np.add.at(source_rate_cell, source_ids, source_rates)
    match = re.search(r"L(\d+)", data_file.name)
    if match is None:
        raise ValueError(f"cannot infer SPE10 layer from {data_file.name}")
    return Case1Config(
        data_file=data_file,
        source_notebook=SOURCE_NOTEBOOK,
        source_points_requested=source_points,
        source_rates=source_rates,
        source_cell_ids=source_ids,
        source_points_effective=xc[source_ids],
        source_rate_cell=source_rate_cell,
        kappa_cell=kappa,
        xc_matrix=xc,
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        layer=int(match.group(1)),
        cell_area=hx * hy,
    )


@dataclass
class Branch:
    branch_id: int
    junction: np.ndarray
    tip: np.ndarray
    tau: np.ndarray
    normal: np.ndarray
    length: float
    n_gamma: int
    n_lambda: int
    s_gamma: np.ndarray
    s_lambda: np.ndarray
    pressure_dofs: np.ndarray
    multiplier_dofs: np.ndarray

    def points(self, s: np.ndarray | float) -> np.ndarray:
        ss = np.asarray(s, dtype=DTYPE).reshape(-1)
        return self.junction[None, :] + ss[:, None] * self.tau[None, :]


@dataclass
class YGeometry:
    junction: np.ndarray
    tips: np.ndarray
    branches: list[Branch]
    h: float
    k_gamma: float
    translated: bool = False
    translation: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=DTYPE))


def build_y_geometry(config: Case1Config, junction=(0.5, 0.5),
                     tips=((0.25, 0.25), (0.75, 0.75), (0.25, 0.75))) -> YGeometry:
    junction = np.asarray(junction, dtype=DTYPE)
    tips = np.asarray(tips, dtype=DTYPE)
    h = min(config.hx, config.hy)
    branches = []
    n_f_total = 1
    n_l_total = 0
    for branch_id, tip in enumerate(tips):
        d = tip - junction
        length = float(np.linalg.norm(d))
        tau = d / length
        normal = np.array([-tau[1], tau[0]], dtype=DTYPE)
        # Use an exactly nested 4:1 pressure/multiplier partition while staying
        # close to h_Gamma=h/2 and h_lambda=2h.
        n_lambda = max(1, int(round(length / (2.0 * h))))
        n_gamma = 4 * n_lambda
        p_dofs = np.r_[0, np.arange(n_f_total, n_f_total + n_gamma, dtype=np.int64)]
        l_dofs = np.arange(n_l_total, n_l_total + n_lambda, dtype=np.int64)
        n_f_total += n_gamma
        n_l_total += n_lambda
        branches.append(Branch(
            branch_id=branch_id, junction=junction.copy(), tip=tip.copy(),
            tau=tau, normal=normal, length=length,
            n_gamma=n_gamma, n_lambda=n_lambda,
            s_gamma=np.linspace(0.0, length, n_gamma + 1),
            s_lambda=np.linspace(0.0, length, n_lambda + 1),
            pressure_dofs=p_dofs, multiplier_dofs=l_dofs,
        ))
    return YGeometry(
        junction=junction, tips=tips, branches=branches, h=h,
        k_gamma=2.0 * float(np.max(config.kappa_cell)),
    )


def placement_check(config: Case1Config, geom: YGeometry) -> dict:
    injector, producer = config.source_points_effective
    d = producer - injector
    t = float(np.dot(geom.junction - injector, d) / np.dot(d, d))
    projection = injector + np.clip(t, 0.0, 1.0) * d
    distance = float(np.linalg.norm(geom.junction - projection))
    # "junction region" is one multiplier-panel scale around J.
    scale = max(branch.length / branch.n_lambda for branch in geom.branches)
    passed = bool(0.0 < t < 1.0 and distance <= 2.0 * scale)
    return {
        "passed": passed, "translated": geom.translated,
        "translation": geom.translation.tolist(), "path_parameter": t,
        "junction_to_well_segment_distance": distance,
        "junction_region_radius": 2.0 * scale,
        "junction": geom.junction.tolist(), "tips": geom.tips.tolist(),
    }


def q1_values_grads(points: np.ndarray, cell_ids: np.ndarray, config: Case1Config) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=DTYPE).reshape(-1, 2)
    c = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    ix = c % config.nx
    iy = c // config.nx
    x0 = ix * config.hx
    y0 = iy * config.hy
    xi = np.clip((p[:, 0] - x0) / config.hx, 0.0, 1.0)
    eta = np.clip((p[:, 1] - y0) / config.hy, 0.0, 1.0)
    vals = np.column_stack(((1-xi)*(1-eta), xi*(1-eta), (1-xi)*eta, xi*eta))
    grads = np.empty((len(p), 4, 2), dtype=DTYPE)
    grads[:, 0, :] = np.column_stack((-(1-eta)/config.hx, -(1-xi)/config.hy))
    grads[:, 1, :] = np.column_stack(((1-eta)/config.hx, -xi/config.hy))
    grads[:, 2, :] = np.column_stack((-eta/config.hx, (1-xi)/config.hy))
    grads[:, 3, :] = np.column_stack((eta/config.hx, xi/config.hy))
    return vals, grads


def locate_cells(points: np.ndarray, config: Case1Config) -> np.ndarray:
    p = np.asarray(points, dtype=DTYPE).reshape(-1, 2)
    ix = np.floor(np.clip(p[:, 0], 0.0, np.nextafter(1.0, 0.0)) / config.hx).astype(int)
    iy = np.floor(np.clip(p[:, 1], 0.0, np.nextafter(1.0, 0.0)) / config.hy).astype(int)
    return np.clip(iy, 0, config.ny-1) * config.nx + np.clip(ix, 0, config.nx-1)


def _grid_breaks(a: np.ndarray, b: np.ndarray, config: Case1Config) -> np.ndarray:
    d = b - a
    out = [0.0, 1.0]
    for axis, (h, n) in enumerate(((config.hx, config.nx), (config.hy, config.ny))):
        if abs(d[axis]) < 1.0e-14:
            continue
        for k in range(1, n):
            t = (k * h - a[axis]) / d[axis]
            if 1.0e-13 < t < 1.0 - 1.0e-13:
                out.append(float(t))
    return np.unique(np.round(out, 14))


class YLCGSystem:
    """Sparse Q1 matrix / shared-junction P1 fracture / branchwise P0 LMDFM."""

    def __init__(self, config: Case1Config, geom: YGeometry):
        self.config = config
        self.geom = geom
        self.nx, self.ny = config.nx, config.ny
        self.n_m = (self.nx + 1) * (self.ny + 1)
        self.n_f = 1 + sum(b.n_gamma for b in geom.branches)
        self.n_l = sum(b.n_lambda for b in geom.branches)
        self.off_m = 0
        self.off_f = self.n_m
        self.off_l = self.n_m + self.n_f
        self.n_total = self.n_m + self.n_f + self.n_l
        xx, yy = np.meshgrid(np.linspace(0, 1, self.nx+1), np.linspace(0, 1, self.ny+1), indexing="xy")
        self.nodes = np.column_stack((xx.ravel(), yy.ravel()))
        cell_nodes = []
        for j in range(self.ny):
            for i in range(self.nx):
                bl = j*(self.nx+1)+i
                cell_nodes.append([bl, bl+1, bl+self.nx+1, bl+self.nx+2])
        self.cell_nodes = np.asarray(cell_nodes, dtype=np.int64)
        self.frac_nodes = np.zeros((self.n_f, 2), dtype=DTYPE)
        self.frac_nodes[0] = geom.junction
        for branch in geom.branches:
            self.frac_nodes[branch.pressure_dofs] = branch.points(branch.s_gamma)
        self.Ke_unit = self._unit_q1_stiffness()
        self.Af_rows, self.Af_cols, self.Af_scale, self.frac_elements = self._fracture_stiffness_pattern()
        self.Bm, self.Bf = self._coupling_blocks()
        self.load = self._matrix_load()
        boundary = ((np.isclose(self.nodes[:,0], 0.0)) | (np.isclose(self.nodes[:,0], 1.0))
                    | (np.isclose(self.nodes[:,1], 0.0)) | (np.isclose(self.nodes[:,1], 1.0)))
        self.boundary_matrix_dofs = np.flatnonzero(boundary)
        self.free = np.r_[np.flatnonzero(~boundary),
                          np.arange(self.off_f, self.n_total, dtype=np.int64)]

    def _unit_q1_stiffness(self) -> np.ndarray:
        g, w = _gauss01(3)
        pts = np.array([[x,y] for y in g for x in g], dtype=DTYPE)
        ww = np.array([wx*wy for wy in w for wx in w], dtype=DTYPE)
        _, grads = q1_values_grads(pts*np.array([self.config.hx,self.config.hy]),
                                   np.zeros(len(pts), dtype=np.int64), self.config)
        return self.config.cell_area * np.einsum("q,qia,qja->ij", ww, grads, grads)

    def _fracture_stiffness_pattern(self):
        rows, cols, scales, elems = [], [], [], []
        for branch in self.geom.branches:
            for j in range(branch.n_gamma):
                a, b = int(branch.pressure_dofs[j]), int(branch.pressure_dofs[j+1])
                ds = float(branch.s_gamma[j+1]-branch.s_gamma[j])
                rows.extend([a,a,b,b]); cols.extend([a,b,a,b])
                scales.extend([1/ds,-1/ds,-1/ds,1/ds])
                elems.append((branch.branch_id, j, a, b, ds))
        return (np.asarray(rows), np.asarray(cols), np.asarray(scales), elems)

    def _coupling_blocks(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        rm, cm, vm = [], [], []
        rf, cf, vf = [], [], []
        g, w = _gauss01(4)
        for branch in self.geom.branches:
            for local_panel, ldof in enumerate(branch.multiplier_dofs):
                s0, s1 = branch.s_lambda[local_panel:local_panel+2]
                a, b = branch.points([s0,s1])
                breaks = _grid_breaks(a, b, self.config)
                for t0, t1 in zip(breaks[:-1], breaks[1:]):
                    tq = t0 + (t1-t0)*g
                    pts = a[None,:] + tq[:,None]*(b-a)[None,:]
                    cids = locate_cells(pts, self.config)
                    for cid in np.unique(cids):
                        mask = cids == cid
                        vals, _ = q1_values_grads(pts[mask], cids[mask], self.config)
                        weights = (s1-s0)*(t1-t0)*w[mask]
                        integ = weights @ vals
                        for node, value in zip(self.cell_nodes[cid], integ):
                            rm.append(int(ldof)); cm.append(int(node)); vm.append(float(value))
                # Independent P0 panel against the nested branch P1 mesh.
                for j in range(branch.n_gamma):
                    e0, e1 = branch.s_gamma[j:j+2]
                    lo, hi = max(s0,e0), min(s1,e1)
                    if hi <= lo + 1.0e-14:
                        continue
                    sq = lo + (hi-lo)*g
                    phi0 = (e1-sq)/(e1-e0); phi1 = (sq-e0)/(e1-e0)
                    weights = (hi-lo)*w
                    for dof, value in zip(branch.pressure_dofs[j:j+2],
                                          [weights@phi0, weights@phi1]):
                        rf.append(int(ldof)); cf.append(int(dof)); vf.append(float(value))
        Bm = sp.coo_matrix((vm,(rm,cm)), shape=(self.n_l,self.n_m)).tocsr()
        Bf = sp.coo_matrix((vf,(rf,cf)), shape=(self.n_l,self.n_f)).tocsr()
        return Bm, Bf

    def _matrix_load(self) -> np.ndarray:
        out = np.zeros(self.n_m, dtype=DTYPE)
        for cid, rate in enumerate(self.config.source_rate_cell):
            if rate != 0.0:
                out[self.cell_nodes[cid]] += 0.25 * rate
        return out

    def assemble(self, mobility_cell=None, mobility_frac_nodes=None) -> sp.csr_matrix:
        if mobility_cell is None:
            mobility_cell = np.ones(self.config.nx*self.config.ny, dtype=DTYPE)
        coeff = self.config.kappa_cell * np.asarray(mobility_cell, dtype=DTYPE)
        rows = np.repeat(self.cell_nodes, 4, axis=1).reshape(-1)
        cols = np.tile(self.cell_nodes, (1,4)).reshape(-1)
        vals = (coeff[:,None,None]*self.Ke_unit[None,:,:]).reshape(-1)
        Am = sp.coo_matrix((vals,(rows,cols)), shape=(self.n_m,self.n_m)).tocsr()
        if mobility_frac_nodes is None:
            mobility_frac_nodes = np.ones(self.n_f, dtype=DTYPE)
        mf = np.asarray(mobility_frac_nodes, dtype=DTYPE)
        elem_mob = np.asarray([0.5*(mf[a]+mf[b]) for _,_,a,b,_ in self.frac_elements])
        elem_vals = np.repeat(self.geom.k_gamma*elem_mob, 4) * self.Af_scale
        Af = sp.coo_matrix((elem_vals,(self.Af_rows,self.Af_cols)), shape=(self.n_f,self.n_f)).tocsr()
        Zml = sp.csr_matrix((self.n_m,self.n_f))
        A = sp.bmat([
            [Am, Zml, -self.Bm.T],
            [Zml.T, Af, self.Bf.T],
            [self.Bm, -self.Bf, None],
        ], format="csr")
        return A

    def solve(self, mobility_cell=None, mobility_frac_nodes=None) -> dict:
        t0 = time.perf_counter()
        A = self.assemble(mobility_cell, mobility_frac_nodes)
        rhs = np.zeros(self.n_total, dtype=DTYPE)
        rhs[:self.n_m] = self.load
        x = np.zeros(self.n_total, dtype=DTYPE)
        Afree = A[self.free][:,self.free].tocsc()
        lu = splu(Afree)
        xfree = lu.solve(rhs[self.free]).astype(np.longdouble)
        # The fracture stiffness is much larger than the coupling entries, so
        # a float64 sparse matvec loses several digits when the local fracture
        # equation is formed.  One classic mixed-precision refinement step
        # (long-double residual, the same float64 LU as preconditioner) removes
        # that cancellation floor without changing the discrete equations.
        Afree_csr = Afree.tocsr()
        indptr, indices = Afree_csr.indptr, Afree_csr.indices
        data_ld = Afree_csr.data.astype(np.longdouble)
        rhs_ld = rhs[self.free].astype(np.longdouble)

        def residual_long_double(vec):
            out = np.empty(Afree.shape[0], dtype=np.longdouble)
            for row in range(Afree.shape[0]):
                lo, hi = indptr[row], indptr[row + 1]
                out[row] = np.dot(data_ld[lo:hi], vec[indices[lo:hi]])
            return rhs_ld - out

        initial_residual = float(np.max(np.abs(residual_long_double(xfree))))
        correction = lu.solve(np.asarray(residual_long_double(xfree), dtype=DTYPE))
        xfree += correction.astype(np.longdouble)
        refined_residual = float(np.max(np.abs(residual_long_double(xfree))))
        x[self.free] = np.asarray(xfree, dtype=DTYPE)
        p_m = x[:self.n_m]
        p_f = x[self.off_f:self.off_l]
        lam = x[self.off_l:]
        constraint = self.Bm @ p_m - self.Bf @ p_f
        return {
            "p_m": p_m, "p_f": p_f, "lambda": lam,
            "constraint_max": float(np.max(np.abs(constraint))),
            "p_f_extended": xfree[-(self.n_f + self.n_l):-self.n_l].copy(),
            "lambda_extended": xfree[-self.n_l:].copy(),
            "linear_residual_before_refinement": initial_residual,
            "linear_residual_after_refinement": refined_residual,
            "solve_s": time.perf_counter()-t0,
            "mobility_cell": np.ones(self.config.nx*self.config.ny) if mobility_cell is None else np.asarray(mobility_cell),
            "mobility_frac_nodes": np.ones(self.n_f) if mobility_frac_nodes is None else np.asarray(mobility_frac_nodes),
        }

    def p_matrix(self, solution: dict, points: np.ndarray) -> np.ndarray:
        cids = locate_cells(points, self.config)
        vals, _ = q1_values_grads(points, cids, self.config)
        return np.einsum("ni,ni->n", vals, solution["p_m"][self.cell_nodes[cids]])

    def q_matrix(self, solution: dict, points: np.ndarray, cell_ids=None) -> np.ndarray:
        p = np.asarray(points, dtype=DTYPE).reshape(-1,2)
        cids = locate_cells(p, self.config) if cell_ids is None else np.asarray(cell_ids,dtype=np.int64)
        _, grads = q1_values_grads(p, cids, self.config)
        grad = np.einsum("nid,ni->nd", grads, solution["p_m"][self.cell_nodes[cids]])
        coeff = self.config.kappa_cell[cids] * solution["mobility_cell"][cids]
        return -coeff[:,None]*grad

    def branch_lambda(self, solution: dict, branch: Branch, s) -> np.ndarray:
        ss = np.asarray(s,dtype=DTYPE).reshape(-1)
        idx = np.searchsorted(branch.s_lambda, ss, side="right")-1
        idx = np.clip(idx,0,branch.n_lambda-1)
        return solution["lambda"][branch.multiplier_dofs[idx]]

    def fracture_fluxes(self, solution: dict) -> dict[int, np.ndarray]:
        out = {}
        mf = solution["mobility_frac_nodes"]
        p_all = solution.get("p_f_extended", solution["p_f"])
        for branch in self.geom.branches:
            p = p_all[branch.pressure_dofs]
            m = 0.5*(mf[branch.pressure_dofs[:-1]]+mf[branch.pressure_dofs[1:]])
            # Use the element transmissibility in the same operation order as
            # the assembled P1 stiffness.  This is algebraically
            # -K_Gamma*a*diff(p)/diff(s), and avoids manufacturing a few-e-12
            # mismatch solely by rounding K*(1/ds) and K/ds differently.
            transmissibility = (self.geom.k_gamma*m) * (1.0/np.diff(branch.s_gamma))
            out[branch.branch_id] = -transmissibility*np.diff(p)
        return out

    def fracture_gate(self, solution: dict, tol=1.0e-12, junction_flux_tol=1.0e-11) -> dict:
        q = self.fracture_fluxes(solution)
        # Form B_f^T lambda in extended precision for the same reason as the
        # refined solve above.  The returned physical arrays remain float64.
        BT = self.Bf.T.tocsr()
        lam = np.asarray(solution.get("lambda_extended", solution["lambda"]),
                         dtype=np.longdouble)
        load_f = np.empty(self.n_f, dtype=np.longdouble)
        data_ld = BT.data.astype(np.longdouble)
        for row in range(self.n_f):
            lo, hi = BT.indptr[row], BT.indptr[row + 1]
            load_f[row] = np.dot(data_ld[lo:hi], lam[BT.indices[lo:hi]])
        residuals = []
        per_branch = {}
        for branch in self.geom.branches:
            qb = q[branch.branch_id]
            r = qb[1:]-qb[:-1] + load_f[branch.pressure_dofs[1:-1]]
            residuals.extend(r.tolist())
            per_branch[str(branch.branch_id+1)] = {
                "max_interior_residual": float(np.max(np.abs(r))) if len(r) else 0.0,
                "tip_flux": float(qb[-1]),
            }
        junction_residual = float(sum(q[b.branch_id][0] for b in self.geom.branches) + load_f[0])
        tip_residuals = [float(-q[b.branch_id][-1]+load_f[b.pressure_dofs[-1]]) for b in self.geom.branches]
        # At J, three equal element diagonal contributions are coalesced into
        # one float64 CSR entry.  Summing the three independently evaluated
        # two-point fluxes rounds in a different order.  Report that flux-form
        # roundoff explicitly, and also certify the actual assembled junction
        # row at the strict tolerance.
        mf = solution["mobility_frac_nodes"]
        elem_mob = np.asarray([0.5*(mf[a]+mf[b]) for _,_,a,b,_ in self.frac_elements])
        elem_vals = np.repeat(self.geom.k_gamma*elem_mob, 4) * self.Af_scale
        Af = sp.coo_matrix((elem_vals,(self.Af_rows,self.Af_cols)),
                           shape=(self.n_f,self.n_f)).tocsr()
        p_ext = np.asarray(solution.get("p_f_extended", solution["p_f"]),
                           dtype=np.longdouble)
        lo, hi = Af.indptr[0], Af.indptr[1]
        junction_assembled = float(
            np.dot(Af.data[lo:hi].astype(np.longdouble), p_ext[Af.indices[lo:hi]])
            + load_f[0]
        )
        strict_worst = max([abs(junction_assembled), *np.abs(residuals), *np.abs(tip_residuals)])
        passed = strict_worst <= tol and abs(junction_residual) <= junction_flux_tol
        return {
            "passed": bool(passed), "tolerance": tol,
            "junction_flux_form_tolerance": junction_flux_tol,
            "junction_residual": junction_residual,
            "junction_assembled_row_residual": junction_assembled,
            "junction_flux_convention": "sum of branch fluxes oriented out of J plus junction exchange load",
            "tip_residuals": tip_residuals, "per_branch": per_branch,
            "max_abs_strict_rows": float(strict_worst),
            "max_abs_including_flux_form_roundoff": float(max(strict_worst, abs(junction_residual))),
        }


def _segment_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
                          tol=2.0e-13):
    r, s = b-a, d-c
    den = _cross2(r,s)
    if abs(den) <= tol:
        return None
    t = _cross2(c-a,s)/den
    u = _cross2(c-a,r)/den
    if tol < t < 1.0-tol and -tol <= u <= 1.0+tol:
        return float(t), float(np.clip(u,0.0,1.0))
    return None


@dataclass
class DualGeometry:
    coords: np.ndarray
    cv_polys: list[list[np.ndarray]]
    raw_p0: np.ndarray
    raw_p1: np.ndarray
    raw_normals: np.ndarray
    raw_owner: np.ndarray
    raw_neighbor: np.ndarray
    raw_host: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    normals: np.ndarray
    owner: np.ndarray
    neighbor: np.ndarray
    host: np.ndarray
    raw_face_id: np.ndarray
    crossing_s: dict[int, list[float]]


def build_dual_geometry(system: YLCGSystem) -> DualGeometry:
    nx, ny = system.nx, system.ny
    coords = system.nodes.copy()
    cv_polys: list[list[np.ndarray]] = [[] for _ in range(len(coords))]
    p0=[]; p1=[]; normals=[]; owner=[]; neighbor=[]; host=[]
    def add(a,b,n,o,nb,cid):
        p0.append(a); p1.append(b); normals.append(n); owner.append(o); neighbor.append(nb); host.append(cid)
    for j in range(ny):
        for i in range(nx):
            cid=j*nx+i
            bl,br,tl,tr=system.cell_nodes[cid]
            x0,x1=i/nx,(i+1)/nx; y0,y1=j/ny,(j+1)/ny
            vb=np.array([x0,y0]); vr=np.array([x1,y0]); vt=np.array([x1,y1]); vl=np.array([x0,y1])
            c=0.25*(vb+vr+vt+vl); mb=0.5*(vb+vr); mr=0.5*(vr+vt); mt=0.5*(vt+vl); ml=0.5*(vl+vb)
            cv_polys[bl].append(_ordered_poly([vb,mb,c,ml])); cv_polys[br].append(_ordered_poly([vr,mr,c,mb]))
            cv_polys[tr].append(_ordered_poly([vt,mt,c,mr])); cv_polys[tl].append(_ordered_poly([vl,ml,c,mt]))
            add(c,mb,np.array([1.,0.]),bl,br,cid); add(c,mr,np.array([0.,1.]),br,tr,cid)
            add(c,mt,np.array([-1.,0.]),tr,tl,cid); add(c,ml,np.array([0.,-1.]),tl,bl,cid)
            if j==0:
                add(vb,mb,np.array([0.,-1.]),bl,-1,cid); add(mb,vr,np.array([0.,-1.]),br,-1,cid)
            if j==ny-1:
                add(vl,mt,np.array([0.,1.]),tl,-1,cid); add(mt,vt,np.array([0.,1.]),tr,-1,cid)
            if i==0:
                add(vb,ml,np.array([-1.,0.]),bl,-1,cid); add(ml,vl,np.array([-1.,0.]),tl,-1,cid)
            if i==nx-1:
                add(vr,mr,np.array([1.,0.]),br,-1,cid); add(mr,vt,np.array([1.,0.]),tr,-1,cid)
    rp0=np.asarray(p0); rp1=np.asarray(p1); rn=np.asarray(normals); ro=np.asarray(owner); rnb=np.asarray(neighbor); rh=np.asarray(host)
    pieces=[]; crossing_s={b.branch_id:[] for b in system.geom.branches}
    for fid,(a,b,n,o,nb,cid) in enumerate(zip(rp0,rp1,rn,ro,rnb,rh)):
        ts=[0.0,1.0]
        for branch in system.geom.branches:
            hit=_segment_intersection(a,b,branch.junction,branch.tip)
            if hit is not None:
                t,u=hit; ts.append(t); crossing_s[branch.branch_id].append(u*branch.length)
        ts=np.unique(np.round(ts,14))
        for ta,tb in zip(ts[:-1],ts[1:]):
            pieces.append((a+ta*(b-a),a+tb*(b-a),n,o,nb,cid,fid))
    return DualGeometry(
        coords=coords, cv_polys=cv_polys, raw_p0=rp0, raw_p1=rp1, raw_normals=rn,
        raw_owner=ro, raw_neighbor=rnb, raw_host=rh,
        p0=np.vstack([x[0] for x in pieces]), p1=np.vstack([x[1] for x in pieces]),
        normals=np.vstack([x[2] for x in pieces]), owner=np.asarray([x[3] for x in pieces]),
        neighbor=np.asarray([x[4] for x in pieces]), host=np.asarray([x[5] for x in pieces]),
        raw_face_id=np.asarray([x[6] for x in pieces]), crossing_s=crossing_s,
    )


class WellParticularField:
    def __init__(self, config: Case1Config):
        self.config=config
        rows=[]
        for cid in np.flatnonzero(config.source_rate_cell):
            i=cid%config.nx; j=cid//config.nx
            x0=i*config.hx; x1=x0+config.hx; y0=j*config.hy; y1=y0+config.hy
            rate=config.source_rate_cell[cid]; rho=rate/config.cell_area
            rows.append((x0,x1,y0,y1,rho,rate,cid))
        self.rects=np.asarray(rows,dtype=DTYPE).reshape(-1,7)

    @staticmethod
    def primitive(u,v):
        u,v=np.broadcast_arrays(np.asarray(u),np.asarray(v)); r2=u*u+v*v
        out=np.zeros_like(r2,dtype=DTYPE); mask=r2>0
        out[mask]=0.5*v[mask]*np.log(r2[mask])+u[mask]*np.arctan2(v[mask],u[mask])
        return out

    def __call__(self, points):
        p=np.asarray(points,dtype=DTYPE).reshape(-1,2); x=p[:,0]; y=p[:,1]
        qx=np.zeros(len(p)); qy=np.zeros(len(p))
        for x0,x1,y0,y1,rho,*_ in self.rects:
            xa=x-x0; xb=x-x1; yc=y-y0; yd=y-y1
            qx += -rho/(2*np.pi)*(self.primitive(xa,yc)-self.primitive(xa,yd)-self.primitive(xb,yc)+self.primitive(xb,yd))
            qy += -rho/(2*np.pi)*(self.primitive(yc,xa)-self.primitive(yc,xb)-self.primitive(yd,xa)+self.primitive(yd,xb))
        return np.column_stack((qx,qy))


def integrate_segments(fn, p0, p1, normals, order=16):
    g,w=_gauss01(order); edge=p1-p0; ell=np.linalg.norm(edge,axis=1); out=np.zeros(len(p0))
    for t,wt in zip(g,w):
        vals=fn(p0+t*edge)
        out += wt*ell*np.einsum("ij,ij->i",vals,normals)
    return out


def scatter_flux(dual: DualGeometry, flux: np.ndarray) -> np.ndarray:
    out=np.zeros(len(dual.coords)); np.add.at(out,dual.owner,flux)
    mask=dual.neighbor>=0; np.add.at(out,dual.neighbor[mask],-np.asarray(flux)[mask])
    return out


def _branch_source_quadrature(system: YLCGSystem, solution: dict, dual: DualGeometry,
                              order=6):
    all_pts=[]; all_w=[]; by_branch={}
    g,w=_gauss01(order)
    for branch in system.geom.branches:
        # Split not only at multiplier and dual-face crossings, but also at
        # every primal-cell crossing.  Lambda is constant across the former;
        # the Q1 trace basis is only polynomial cell-by-cell across the latter.
        primal_s = branch.length * _grid_breaks(branch.junction, branch.tip,
                                                 system.config)
        breaks=np.unique(np.round(np.r_[branch.s_lambda,
                                        dual.crossing_s[branch.branch_id],
                                        primal_s],14))
        pts=[]; weights=[]
        for a,b in zip(breaks[:-1],breaks[1:]):
            if b<=a+1e-14: continue
            s=a+(b-a)*g; rho=system.branch_lambda(solution,branch,s)
            pts.append(branch.points(s)); weights.append((b-a)*w*rho)
        bp=np.vstack(pts); bw=np.concatenate(weights)
        by_branch[branch.branch_id]=(bp,bw)
        all_pts.append(bp); all_w.append(bw)
    return np.vstack(all_pts),np.concatenate(all_w),by_branch


@dataclass
class BuiltProblem:
    dual_problem: hc.DualProblem
    dual: DualGeometry
    cv_class: np.ndarray
    cv_cut_count: np.ndarray
    source_mask: np.ndarray
    boundary_mask: np.ndarray
    branch_source_quadrature: dict
    qpf: WellParticularField


def build_hardcurl_problem(system: YLCGSystem, solution: dict, line_order=6, face_order=128) -> BuiltProblem:
    dual=build_dual_geometry(system); qpf=WellParticularField(system.config)
    src_pts,src_w,by_branch=_branch_source_quadrature(system,solution,dual,line_order)
    cv_lambda=np.zeros(len(dual.coords)); cut_count=np.zeros(len(dual.coords),dtype=int); touch_count=np.zeros(len(dual.coords),dtype=int)
    for branch in system.geom.branches:
        bp,bw=by_branch[branch.branch_id]
        vals,cross,touch=hc._distribute_line_quadrature_to_cvs(dual.cv_polys,bp,bw,branch.normal)
        cv_lambda += vals; cut_count += cross.astype(int); touch_count += touch.astype(int)
    cv_source=np.zeros(len(dual.coords))
    for cid,rate in enumerate(system.config.source_rate_cell):
        if rate!=0: cv_source[system.cell_nodes[cid]] += 0.25*rate
    boundary=((np.isclose(dual.coords[:,0],0))|(np.isclose(dual.coords[:,0],1))
              |(np.isclose(dual.coords[:,1],0))|(np.isclose(dual.coords[:,1],1)))
    source=np.abs(cv_source)>0
    cls=np.full(len(dual.coords),"interior",dtype=object)
    cls[cut_count==1]="single-cut"; cls[cut_count>=2]="multi-cut"
    cls[source]="source"; cls[boundary]="boundary"
    qpf_face=integrate_segments(qpf,dual.p0,dual.p1,dual.normals,face_order)
    qpl_face=hc._line_source_face_flux(dual.p0,dual.p1,dual.normals,src_pts,src_w)
    cg_face=integrate_segments(lambda x: system.q_matrix(solution,x,dual.host),dual.p0,dual.p1,dual.normals,4)
    sign=hc._curl_endpoint_sign(dual.p0,dual.p1,dual.normals)
    lam_a=[]; lam_b=[]; lam_rho=[]
    for branch in system.geom.branches:
        lam_a.append(branch.points(branch.s_lambda[:-1])); lam_b.append(branch.points(branch.s_lambda[1:]))
        lam_rho.append(solution["lambda"][branch.multiplier_dofs])
    lam_a=np.vstack(lam_a); lam_b=np.vstack(lam_b); lam_rho=np.concatenate(lam_rho)
    gdata={"omega_geometry":system.nodes,"FRAC_A":system.geom.branches[0].junction,
           "normal_np":system.geom.branches[0].normal,"h_est":system.geom.h}
    ns={
        "has_exact_lambda":False,"q_p_f_custom_numpy":qpf,
        "q_cg_numpy":lambda ctx,g,x: system.q_matrix(ctx,x),
    }
    problem=hc.DualProblem(
        variant="y-spe10-case4",ctx=solution,gdata=gdata,coords=dual.coords,
        p0=dual.p0,p1=dual.p1,normals=dual.normals,owner=dual.owner,neighbor=dual.neighbor,
        host_cell=dual.host,side=np.zeros(len(dual.p0),dtype=np.int8),cv_polys=dual.cv_polys,
        cv_source=cv_source,cv_lambda_exact=cv_lambda.copy(),cv_lambda_h=cv_lambda,
        cv_class=cls,lambda_s_nodes=np.arange(len(lam_rho)+1,dtype=DTYPE),
        lambda_h_density=lam_rho,lambda_h_nodal_density=np.r_[lam_rho,lam_rho[-1]],
        lambda_seg_a=lam_a,lambda_seg_b=lam_b,qpf_face=qpf_face,
        qpl_exact_face=qpl_face.copy(),qpl_h_face=qpl_face,cg_face=cg_face,
        exact_face=np.full_like(cg_face,np.nan),curl_sign=sign,
        source_quad_exact_points=src_pts,source_quad_exact_weights=src_w,
        source_quad_h_points=src_pts,source_quad_h_weights=src_w,
        polynomial_order=1,multiplier_order=0,globals_ns=ns,
    )
    return BuiltProblem(problem,dual,cls,cut_count,source,boundary,by_branch,qpf)


CLASS_NAMES=("interior","single-cut","multi-cut","source","boundary")


def audit_flux(built: BuiltProblem, flux: np.ndarray, label: str, print_table=True) -> dict:
    p=built.dual_problem
    residual=scatter_flux(built.dual,np.asarray(flux))-p.cv_source-p.cv_lambda_h
    table={}
    if print_table:
        print(label); print("  class             n      mean|R|       max|R|")
    for name in CLASS_NAMES:
        vals=residual[built.cv_class==name]
        row={"n":int(len(vals)),"mean_abs":None if not len(vals) else float(np.mean(np.abs(vals))),
             "rmse":None if not len(vals) else float(np.sqrt(np.mean(vals*vals))),
             "max_abs":None if not len(vals) else float(np.max(np.abs(vals)))}
        table[name]=row
        if print_table:
            if len(vals): print(f"  {name:<16} {len(vals):5d} {row['mean_abs']:12.3e} {row['max_abs']:12.3e}")
            else: print(f"  {name:<16} {0:5d}          N/A          N/A")
    return {"label":label,"stats":table,"residual":residual}


def gate_a1_multibranch(built: BuiltProblem, tol=1.0e-13) -> dict:
    p=built.dual_problem
    torch.manual_seed(SEED)
    net=hc.FourierPsiNet((1,2,4),17,2).to(dtype=TORCH_DTYPE)
    pts=np.vstack((p.p0,p.p1)); uniq,inv=np.unique(pts,axis=0,return_inverse=True); n=len(p.p0)
    with torch.no_grad(): values=net(torch.as_tensor(uniq,dtype=TORCH_DTYPE)).cpu().numpy()
    curl=p.curl_sign*(values[inv[n:]]-values[inv[:n]])
    rows={}
    passed=True
    for name,flux in (("lambda_h/psi=0",p.qpf_face+p.qpl_h_face),
                      ("lambda_h/random psi",p.qpf_face+p.qpl_h_face+curl)):
        audit=audit_flux(built,flux,name,print_table=True); rows[name]=audit["stats"]
        for cls in CLASS_NAMES:
            val=audit["stats"][cls]["max_abs"]
            if val is not None: passed &= val<=tol
    if not passed: raise RuntimeError("Gate A1 multi-branch conservation failed")
    return {"passed":True,"tolerance":tol,"classes":rows,
            "multi_cut_cv_ids":np.flatnonzero(built.cv_class=="multi-cut").tolist()}


def gate_a0_interior_tip(tol_jump=2.0e-5, tol_beyond=2.0e-5) -> dict:
    rho=1.7; a=np.array([0.35,0.45]); b=np.array([0.65,0.55]); tau=_unit(b-a); normal=np.array([-tau[1],tau[0]])
    length=float(np.linalg.norm(b-a)); center=b-0.08*length*tau
    eps_seq=np.array([1e-4,3e-5,1e-5,3e-6,1e-6])
    jumps=[]
    for eps in eps_seq:
        q=hc.q_p_lambda_p0_numpy(np.vstack((center+eps*normal,center-eps*normal)),a[None,:],b[None,:],np.array([rho]))
        jumps.append(float((q[0]-q[1])@normal))
    beyond=b+0.04*length*tau
    qb=hc.q_p_lambda_p0_numpy(np.vstack((beyond+1e-7*normal,beyond-1e-7*normal)),a[None,:],b[None,:],np.array([rho]))
    beyond_jump=float((qb[0]-qb[1])@normal)
    rel=abs(jumps[-1]-rho)/abs(rho)
    passed=rel<=tol_jump and abs(beyond_jump)<=tol_beyond
    if not passed: raise RuntimeError("Gate A0 interior-tip test failed")
    return {"passed":True,"density":rho,"eps":eps_seq.tolist(),"normal_jumps":jumps,
            "final_relative_error":rel,"beyond_tip_jump":beyond_jump,
            "field_decays_beyond_tip":True}


class LogKFourierPsiNet(nn.Module):
    def __init__(self, config: Case1Config, frequencies=(1,2,4,8), width=32, depth=3):
        super().__init__(); self.config=config
        self.register_buffer("freq",torch.as_tensor(frequencies,dtype=TORCH_DTYPE))
        logk=np.log10(np.maximum(config.kappa_cell,1e-300)).reshape(config.ny,config.nx)
        pad=np.pad(logk,1,mode="edge")
        nodal=0.25*(pad[:-1,:-1]+pad[:-1,1:]+pad[1:,:-1]+pad[1:,1:])
        self.register_buffer("logk_node",torch.as_tensor(nodal,dtype=TORCH_DTYPE))
        self.register_buffer("logk_mean",torch.as_tensor(float(np.mean(nodal)),dtype=TORCH_DTYPE))
        self.register_buffer("logk_std",torch.as_tensor(float(np.std(nodal)+1e-12),dtype=TORCH_DTYPE))
        in_dim=3+4*len(frequencies); layers=[nn.Linear(in_dim,width),nn.SiLU()]
        for _ in range(depth-1): layers.extend((nn.Linear(width,width),nn.SiLU()))
        layers.append(nn.Linear(width,1)); self.net=nn.Sequential(*layers)

    def features(self,x):
        gx=x[:,0]/self.config.hx; gy=x[:,1]/self.config.hy
        ix=torch.clamp(torch.floor(gx.detach()).long(),0,self.config.nx-1); iy=torch.clamp(torch.floor(gy.detach()).long(),0,self.config.ny-1)
        s=torch.clamp(gx-ix.to(x.dtype),0,1); t=torch.clamp(gy-iy.to(x.dtype),0,1)
        lk=((1-s)*(1-t)*self.logk_node[iy,ix]+s*(1-t)*self.logk_node[iy,ix+1]
            +(1-s)*t*self.logk_node[iy+1,ix]+s*t*self.logk_node[iy+1,ix+1])
        phase_x=2*torch.pi*x[:,0:1]*self.freq[None,:]; phase_y=2*torch.pi*x[:,1:2]*self.freq[None,:]
        return torch.cat((x,torch.sin(phase_x),torch.cos(phase_x),torch.sin(phase_y),torch.cos(phase_y),
                          ((lk-self.logk_mean)/self.logk_std)[:,None]),dim=1)

    def hidden_features(self,x):
        z=self.features(x)
        for layer in list(self.net.children())[:-1]: z=layer(z)
        return z

    def forward(self,x): return self.net(self.features(x)).squeeze(-1)


def model_factory(config: Case1Config):
    return lambda frequencies,width,depth: LogKFourierPsiNet(config,frequencies,width,depth)


def q_p_lambda_field(problem: hc.DualProblem, points: np.ndarray) -> np.ndarray:
    return hc.q_p_lambda_p0_numpy(points,problem.lambda_seg_a,problem.lambda_seg_b,problem.lambda_h_density)


def option_a_flux_numpy(built: BuiltProblem, option_a: dict, points: np.ndarray) -> np.ndarray:
    return built.qpf(points)+q_p_lambda_field(built.dual_problem,points)+hc._model_curl_numpy(option_a["model"],points)


def exchange_integrals(system: YLCGSystem, solution: dict) -> dict:
    out={}; total=0.0
    for b in system.geom.branches:
        value=float(np.sum(solution["lambda"][b.multiplier_dofs]*np.diff(b.s_lambda)))
        out[str(b.branch_id+1)]=value; total+=value
    return {"per_branch":out,"sum":float(total),"passed":bool(abs(total)<=1e-11),"tolerance":1e-11}


def machine_info() -> dict:
    return {"platform":platform.platform(),"processor":platform.processor(),
            "python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__}


def _cell_local_polys(system: YLCGSystem, cid: int) -> list[np.ndarray]:
    i=cid%system.nx; j=cid//system.nx
    x0,x1=i/system.nx,(i+1)/system.nx; y0,y1=j/system.ny,(j+1)/system.ny
    bl=np.array([x0,y0]); br=np.array([x1,y0]); tr=np.array([x1,y1]); tl=np.array([x0,y1])
    c=0.25*(bl+br+tr+tl); mb=0.5*(bl+br); mr=0.5*(br+tr); mt=0.5*(tr+tl); ml=0.5*(tl+bl)
    # Local Q1 order is bl, br, tl, tr.
    return [_ordered_poly([bl,mb,c,ml]),_ordered_poly([br,mr,c,mb]),
            _ordered_poly([tl,ml,c,mt]),_ordered_poly([tr,mt,c,mr])]


def _segment_basis_flux(system: YLCGSystem, cid: int, a: np.ndarray, b: np.ndarray,
                        normal: np.ndarray, coeff: float, order=3) -> np.ndarray:
    g,w=_gauss01(order); pts=a[None,:]+g[:,None]*(b-a)[None,:]
    _,grads=q1_values_grads(pts,np.full(len(g),cid,dtype=np.int64),system.config)
    ell=float(np.linalg.norm(b-a))
    return -coeff*ell*np.einsum("q,qid,d->i",w,grads,normal)


def _local_B(system: YLCGSystem, cid: int, coeff: float) -> np.ndarray:
    B=np.zeros((4,4),dtype=DTYPE)
    c=np.mean(system.nodes[system.cell_nodes[cid]],axis=0)
    for row,poly in enumerate(_cell_local_polys(system,cid)):
        for a,b in zip(poly,np.roll(poly,-1,axis=0)):
            if not (np.allclose(a,c,atol=1e-14) or np.allclose(b,c,atol=1e-14)):
                continue
            tangent=b-a
            # Polygons are CCW, hence the right normal is outward.
            normal=_unit(np.array([tangent[1],-tangent[0]]))
            B[row] += _segment_basis_flux(system,cid,a,b,normal,coeff)
    return B


def _source_t_lambda_local(system: YLCGSystem, built: BuiltProblem) -> np.ndarray:
    out=np.zeros((system.config.nx*system.config.ny,4),dtype=DTYPE)
    eps=5e-11
    for branch in system.geom.branches:
        pts,weights=built.branch_source_quadrature[branch.branch_id]
        for shifted,factor in ((pts+eps*branch.normal,0.5),(pts-eps*branch.normal,0.5)):
            cids=locate_cells(shifted,system.config)
            ix=cids%system.nx; iy=cids//system.nx
            xi=(shifted[:,0]-ix*system.config.hx)/system.config.hx
            eta=(shifted[:,1]-iy*system.config.hy)/system.config.hy
            local=(xi>=0.5).astype(int)+2*(eta>=0.5).astype(int)
            np.add.at(out,(cids,local),factor*weights)
    return out


def _source_phi_lambda(system: YLCGSystem, built: BuiltProblem) -> np.ndarray:
    out=np.zeros((system.config.nx*system.config.ny,4),dtype=DTYPE)
    pts=built.dual_problem.source_quad_h_points; weights=built.dual_problem.source_quad_h_weights
    cids=locate_cells(pts,system.config); vals,_=q1_values_grads(pts,cids,system.config)
    for a in range(4): np.add.at(out[:,a],cids,weights*vals[:,a])
    return out


def _kgrad_at(system: YLCGSystem, solution: dict, cid: int, points: np.ndarray) -> np.ndarray:
    p=np.asarray(points,dtype=DTYPE).reshape(-1,2)
    _,grads=q1_values_grads(p,np.full(len(p),cid,dtype=np.int64),system.config)
    grad=np.einsum("qid,i->qd",grads,solution["p_m"][system.cell_nodes[cid]])
    coeff=system.config.kappa_cell[cid]*solution["mobility_cell"][cid]
    return coeff*grad


def _neighbor_for_edge(system: YLCGSystem, cid: int, edge: int) -> int:
    i=cid%system.nx; j=cid//system.nx
    if edge==0: j-=1
    elif edge==1: i+=1
    elif edge==2: j+=1
    else: i-=1
    return -1 if i<0 or i>=system.nx or j<0 or j>=system.ny else j*system.nx+i


def nlr_reconstruct(system: YLCGSystem, solution: dict, built: BuiltProblem) -> dict:
    """Batched-geometry Deng reconstruction with all branch loads node-split."""
    t0=time.perf_counter(); nc=system.config.nx*system.config.ny
    source_t=_source_t_lambda_local(system,built)
    source_phi=_source_phi_lambda(system,built)
    for cid,rate in enumerate(system.config.source_rate_cell):
        if rate!=0:
            source_t[cid] += 0.25*rate
            source_phi[cid] += 0.25*rate
    coeffs=np.zeros((nc,4),dtype=DTYPE); rhs_all=np.zeros((nc,4),dtype=DTYPE)
    component_all={name:np.zeros((nc,4),dtype=DTYPE)
                   for name in ("source_t","source_phi","a_term","e_I","e_phi")}
    g,w=_gauss01(3)
    for cid in range(nc):
        k=float(system.config.kappa_cell[cid]*solution["mobility_cell"][cid])
        B=_local_B(system,cid,k)
        i=cid%system.nx; j=cid//system.nx
        pts=np.array([[i/system.nx+x*system.config.hx,j/system.ny+y*system.config.hy] for y in g for x in g])
        ww=np.array([wy*wx for wy in w for wx in w])
        _,grads=q1_values_grads(pts,np.full(len(pts),cid),system.config)
        gradp=np.einsum("qid,i->qd",grads,solution["p_m"][system.cell_nodes[cid]])
        a_term=system.config.cell_area*np.einsum("q,qid,qd->i",ww,k*grads,gradp)
        eI=np.zeros(4); ephi=np.zeros(4)
        poly_nodes=system.nodes[system.cell_nodes[cid]]
        # bottom, right, top, left; endpoint local Q1 indices in traversal order.
        edge_pairs=((0,1),(1,3),(3,2),(2,0))
        centroid=np.mean(poly_nodes,axis=0)
        for edge,(la,lb) in enumerate(edge_pairs):
            a=poly_nodes[la]; b=poly_nodes[lb]; tangent=b-a
            normal=_unit(np.array([tangent[1],-tangent[0]]))
            nb=_neighbor_for_edge(system,cid,edge)
            pq=a[None,:]+g[:,None]*(b-a)[None,:]
            kg=_kgrad_at(system,solution,cid,pq)
            if nb>=0: kg=0.5*(kg+_kgrad_at(system,solution,nb,pq))
            vals,_=q1_values_grads(pq,np.full(len(g),cid),system.config)
            ndot=kg@normal; ell=float(np.linalg.norm(b-a))
            ephi += ell*np.einsum("q,q,qi->i",w,ndot,vals)
            mid=0.5*(a+b)
            for loc,start,end in ((la,a,mid),(lb,b,mid)):
                ph=start[None,:]+g[:,None]*(end-start)[None,:]
                kg_h=_kgrad_at(system,solution,cid,ph)
                if nb>=0: kg_h=0.5*(kg_h+_kgrad_at(system,solution,nb,ph))
                eI[loc] += float(np.linalg.norm(end-start))*np.dot(w,kg_h@normal)
        rhs=source_t[cid]-source_phi[cid]+a_term+eI-ephi
        rhs_all[cid]=rhs
        component_all["source_t"][cid]=source_t[cid]
        component_all["source_phi"][cid]=source_phi[cid]
        component_all["a_term"][cid]=a_term
        component_all["e_I"][cid]=eI
        component_all["e_phi"][cid]=ephi
        # Fix the constant null mode to the first CG nodal value.
        alpha=np.zeros(4); alpha[0]=solution["p_m"][system.cell_nodes[cid,0]]
        alpha[1:]=np.linalg.lstsq(B[:,1:],rhs-B[:,0]*alpha[0],rcond=None)[0]
        coeffs[cid]=alpha
    def qrec(points,cell_ids):
        p=np.asarray(points,dtype=DTYPE).reshape(-1,2); c=np.asarray(cell_ids,dtype=np.int64)
        _,grads=q1_values_grads(p,c,system.config)
        grad=np.einsum("nid,ni->nd",grads,coeffs[c])
        k=system.config.kappa_cell[c]*solution["mobility_cell"][c]
        return -k[:,None]*grad
    face=integrate_segments(lambda x:qrec(x,built.dual.host),built.dual.p0,built.dual.p1,built.dual.normals,4)
    return {"coeffs":coeffs,"rhs":rhs_all,"components":component_all,
            "face_flux":face,"wall_s":time.perf_counter()-t0,
            "audit":audit_flux(built,face,"NLR conservation audit")}


def build_projection_matrix(Ha: np.ndarray, Hb: np.ndarray, w: np.ndarray, r: int) -> np.ndarray:
    H=np.vstack((Ha,Hb)).astype(DTYPE); w=np.asarray(w,dtype=DTYPE).reshape(-1); ww=float(w@w)
    Hc=H-np.mean(H,axis=0,keepdims=True); Hc-=np.outer(Hc@w,w)/ww
    _,_,vt=np.linalg.svd(Hc,full_matrices=False); modes=[]
    for cand in list(vt)+list(np.eye(len(w))):
        v=np.asarray(cand,dtype=DTYPE).copy(); v-=w*(v@w/ww)
        for u in modes: v-=u*(v@u)
        nv=float(np.linalg.norm(v))
        if nv>1e-12: modes.append(v/nv)
        if len(modes)>=r-1: break
    if len(modes)!=r-1: raise RuntimeError("could not construct PoU reduced basis")
    return np.column_stack([w]+modes)


class PoUHead:
    def __init__(self,built:BuiltProblem,model:LogKFourierPsiNet,window_shape=(16,16),r=16,overlap=0.5):
        self.built=built; self.model=model; self.window_shape=tuple(window_shape); self.r=int(r); self.overlap=float(overlap)
        p=built.dual_problem; self.p0=p.p0; self.p1=p.p1; self.sign=p.curl_sign
        with torch.no_grad():
            Ha=model.hidden_features(torch.as_tensor(self.p0,dtype=TORCH_DTYPE)).cpu().numpy()
            Hb=model.hidden_features(torch.as_tensor(self.p1,dtype=TORCH_DTYPE)).cpu().numpy()
        final=list(model.net.children())[-1]; w=final.weight.detach().cpu().numpy().reshape(-1)
        self.P=build_projection_matrix(Ha,Hb,w,self.r); self.Ha=Ha; self.Hb=Hb
        self.K=self.window_shape[0]*self.window_shape[1]
        self.theta=np.zeros((self.K,self.r)); self.theta[:,0]=1.0
        self.theta_bar=self.theta.copy(); self._factor=None
        self._build_design()

    @staticmethod
    def _bump(x,c,radius):
        z=np.abs(np.asarray(x)[:,None]-np.asarray(c)[None,:])/radius; out=np.zeros_like(z); m=z<1
        out[m]=0.5*(1+np.cos(np.pi*z[m])); return out

    def weights(self,points):
        p=np.asarray(points); nxw,nyw=self.window_shape; cx=np.linspace(0,1,nxw); cy=np.linspace(0,1,nyw)
        rx=0.5/(nxw-1)/max(1-self.overlap,1e-12); ry=0.5/(nyw-1)/max(1-self.overlap,1e-12)
        wx=self._bump(p[:,0],cx,rx); wy=self._bump(p[:,1],cy,ry)
        W=(wy[:,:,None]*wx[:,None,:]).reshape(len(p),-1); den=W.sum(axis=1,keepdims=True)
        if np.any(den<=0): raise RuntimeError("PoU windows do not cover all face endpoints")
        return W/den

    def _build_design(self):
        t0=time.perf_counter(); Wa=self.weights(self.p0); Wb=self.weights(self.p1)
        pa=self.Ha@self.P; pb=self.Hb@self.P; rows=[]; cols=[]; vals=[]; ar=np.arange(self.r)
        for f in range(len(self.p0)):
            for W,phi,sgn in ((Wb,pb,self.sign[f]),(Wa,pa,-self.sign[f])):
                active=np.flatnonzero(W[f]>1e-14); cc=active[:,None]*self.r+ar[None,:]
                vv=sgn*W[f,active,None]*phi[f,None,:]
                rows.append(np.full(cc.size,f)); cols.append(cc.ravel()); vals.append(vv.ravel())
        self.Phi=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
                               shape=(len(self.p0),self.K*self.r)).tocsc()
        self.build_s=time.perf_counter()-t0; self.theta_bar_vec=self.theta_bar.ravel()

    def factorize(self,ridge_rel=1e-8):
        normal=(self.Phi.T@self.Phi).tocsc(); diag=float(np.max(np.abs(normal.diagonal())))
        ridge=float(ridge_rel)*max(diag,1.0); t0=time.perf_counter(); lu=splu(normal+ridge*sp.eye(normal.shape[0],format="csc"))
        self._factor={"lu":lu,"ridge":ridge,"ridge_rel":float(ridge_rel),"factor_s":time.perf_counter()-t0}
        return self._factor

    def base_flux(self,built=None):
        b=self.built if built is None else built; p=b.dual_problem
        return p.qpf_face+p.qpl_h_face

    def prediction(self,built=None,theta=None):
        th=self.theta if theta is None else theta
        return self.base_flux(built)+np.asarray(self.Phi@np.asarray(th).ravel()).reshape(-1)

    def fit(self,target,built=None,anchor=None,ridge_rel=1e-8):
        if self._factor is None or self._factor["ridge_rel"]!=float(ridge_rel): self.factorize(ridge_rel)
        base=self.base_flux(built); anc=self.theta.ravel() if anchor is None else np.asarray(anchor).ravel()
        t0=time.perf_counter(); rhs=np.asarray(self.Phi.T@(np.asarray(target)-base)).reshape(-1)+self._factor["ridge"]*anc
        vec=self._factor["lu"].solve(rhs); self.theta=vec.reshape(self.K,self.r)
        pred=base+np.asarray(self.Phi@vec).reshape(-1)
        return {"prediction":pred,"theta":self.theta.copy(),"fit_s":time.perf_counter()-t0,
                "factor_s":self._factor["factor_s"],"ridge_rel":ridge_rel,
                "rmse":float(np.sqrt(np.mean((pred-np.asarray(target))**2)))}


def mobility_factor(S,M=1.0):
    s=np.clip(np.asarray(S,dtype=DTYPE),0,1); return M*s*s+(1-s)*(1-s)


def fractional_flow(S,M=1.0):
    s=np.clip(np.asarray(S,dtype=DTYPE),0,1); return M*s*s/(mobility_factor(s,M)+1e-30)


def project_dual_to_cells(system:YLCGSystem,S:np.ndarray)->np.ndarray:
    return np.mean(np.asarray(S)[system.cell_nodes],axis=1)


def dual_areas(system:YLCGSystem)->np.ndarray:
    out=np.zeros(system.n_m)
    for nodes in system.cell_nodes: out[nodes]+=0.25*system.config.cell_area
    return out


def fracture_volumes(system:YLCGSystem)->np.ndarray:
    out=np.zeros(system.n_f)
    for b in system.geom.branches:
        ds=np.diff(b.s_gamma); local=np.r_[0.5*ds[0],0.5*(ds[:-1]+ds[1:]),0.5*ds[-1]]
        np.add.at(out,b.pressure_dofs,local)
    return out


def _dual_node_for_point(system:YLCGSystem,point:np.ndarray)->int:
    p=np.asarray(point); cid=int(locate_cells(p[None,:],system.config)[0]); i=cid%system.nx; j=cid//system.nx
    xi=(p[0]-i*system.config.hx)/system.config.hx; eta=(p[1]-j*system.config.hy)/system.config.hy
    return (j+int(eta>=0.5))*(system.nx+1)+(i+int(xi>=0.5))


def build_exchange_connections(system:YLCGSystem,built:BuiltProblem)->dict:
    cv=[]; fd=[]; E=[]; branch_id=[]; eps=5e-11
    for b in system.geom.branches:
        pts,weights=built.branch_source_quadrature[b.branch_id]
        s=(pts-b.junction)@b.tau; elem=np.clip(np.searchsorted(b.s_gamma,s,side="right")-1,0,b.n_gamma-1)
        ell=b.s_gamma[elem+1]-b.s_gamma[elem]; phi0=(b.s_gamma[elem+1]-s)/ell; phi1=(s-b.s_gamma[elem])/ell
        for k,(pt,wt) in enumerate(zip(pts,weights)):
            m_ids=[_dual_node_for_point(system,pt+eps*b.normal),_dual_node_for_point(system,pt-eps*b.normal)]
            m_unique={mid:0.5*m_ids.count(mid) for mid in set(m_ids)}
            f_ids=[int(b.pressure_dofs[elem[k]]),int(b.pressure_dofs[elem[k]+1])]
            f_phi=[float(phi0[k]),float(phi1[k])]
            for mid,mfrac in m_unique.items():
                for fid,ffrac in zip(f_ids,f_phi):
                    value=float(wt*mfrac*ffrac)
                    if abs(value)>1e-16:
                        cv.append(mid); fd.append(fid); E.append(value); branch_id.append(b.branch_id)
    cv=np.asarray(cv,dtype=np.int64); fd=np.asarray(fd,dtype=np.int64); E=np.asarray(E)
    mismatch=float(np.max(np.abs(np.bincount(cv,weights=E,minlength=system.n_m)-built.dual_problem.cv_lambda_h)))
    return {"cv":cv,"frac":fd,"E":E,"branch":np.asarray(branch_id),"max_cv_mismatch":mismatch}


def producer_watercut(system:YLCGSystem,S:np.ndarray,M=1.0)->float:
    src=system.config.source_rate_cell
    ids=np.unique(system.cell_nodes[np.flatnonzero(src<0)].ravel())
    return float(np.mean(fractional_flow(np.asarray(S)[ids],M)))


def one_transport_step(system:YLCGSystem,built:BuiltProblem,face_flux:np.ndarray,solution:dict,
                       S:np.ndarray,Sf:np.ndarray,dt:float,M=1.0,connections=None)->tuple[np.ndarray,np.ndarray,dict]:
    if connections is None: connections=build_exchange_connections(system,built)
    p=built.dual; q=np.asarray(face_flux); owner=p.owner; nb=p.neighbor
    up=np.empty_like(q); pos=q>=0; up[pos]=S[owner[pos]]
    interior=(~pos)&(nb>=0); up[interior]=S[nb[interior]]; up[(~pos)&(nb<0)]=0.0
    water=q*fractional_flow(up,M); outward=scatter_flux(p,water)
    well=np.zeros(system.n_m)
    for cid,rate in enumerate(system.config.source_rate_cell):
        if rate==0: continue
        nodes=system.cell_nodes[cid]; share=rate/4
        if rate>0: well[nodes]+=share*fractional_flow(1.0,M)
        else: well[nodes]+=share*fractional_flow(S[nodes],M)
    c=connections; Sup=np.where(c["E"]>0,Sf[c["frac"]],S[c["cv"]]); W=c["E"]*fractional_flow(Sup,M)
    mex=np.bincount(c["cv"],weights=W,minlength=system.n_m)
    fex=np.bincount(c["frac"],weights=-W,minlength=system.n_f)
    fmass=np.zeros(system.n_f); qbranch=system.fracture_fluxes(solution)
    for b in system.geom.branches:
        qb=qbranch[b.branch_id]
        for e,val in enumerate(qb):
            a=int(b.pressure_dofs[e]); bb=int(b.pressure_dofs[e+1]); su=Sf[a] if val>=0 else Sf[bb]
            wf=float(val*fractional_flow(su,M)); fmass[a]-=wf; fmass[bb]+=wf
    Snew=S+dt*(-outward+well+mex)/dual_areas(system)
    Sfnew=Sf+dt*(fmass+fex)/fracture_volumes(system)
    inj=np.unique(system.cell_nodes[np.flatnonzero(system.config.source_rate_cell>0)].ravel())
    Snew[inj]=1.0
    report={"matrix_mass_change":float(np.sum((Snew-S)*dual_areas(system))),
            "fracture_mass_change":float(np.sum((Sfnew-Sf)*fracture_volumes(system))),
            "producer_watercut":producer_watercut(system,Snew,M),
            "min_before_clip":float(min(np.min(Snew),np.min(Sfnew))),
            "max_before_clip":float(max(np.max(Snew),np.max(Sfnew)))}
    return np.clip(Snew,0,1),np.clip(Sfnew,0,1),report


def initial_cfl_dt(system:YLCGSystem,built:BuiltProblem,face_flux:np.ndarray,solution:dict,
                   connections:dict,cfl=0.45,fprime_max=2.0)->float:
    p=built.dual; q=np.asarray(face_flux); out=np.bincount(p.owner,weights=np.maximum(q,0),minlength=system.n_m)
    mask=p.neighbor>=0; out+=np.bincount(p.neighbor[mask],weights=np.maximum(-q[mask],0),minlength=system.n_m)
    out+=np.bincount(connections["cv"],weights=np.abs(connections["E"]),minlength=system.n_m)
    am=dual_areas(system); dtm=np.min(am[out>1e-30]/(fprime_max*out[out>1e-30]))
    fout=np.bincount(connections["frac"],weights=np.abs(connections["E"]),minlength=system.n_f)
    for b in system.geom.branches:
        for e,val in enumerate(system.fracture_fluxes(solution)[b.branch_id]):
            a=int(b.pressure_dofs[e]); bb=int(b.pressure_dofs[e+1])
            fout[a]+=max(val,0); fout[bb]+=max(-val,0)
    av=fracture_volumes(system); dtf=np.min(av[fout>1e-30]/(fprime_max*fout[fout>1e-30]))
    return float(cfl*min(dtm,dtf))


class YDynamicGeometry:
    """Reusable linear geometry for per-step lambda, exchange, and face data.

    The expensive polygon ownership and quadrature topology are built once.
    Every pressure step then changes only the multiplier coefficients.
    """

    def __init__(self, system: YLCGSystem, template: BuiltProblem, line_order=6,
                 face_order=4):
        t0 = time.perf_counter()
        self.system = system
        self.template = template
        self.dual = template.dual
        self.qpf_face = template.dual_problem.qpf_face.copy()
        self.line_order = int(line_order)
        self.face_order = int(face_order)
        self._build_line_basis()
        self._build_face_quadrature()
        self._build_qpl_response()
        self.build_s = time.perf_counter() - t0

    def _build_line_basis(self):
        system = self.system
        g, w = _gauss01(self.line_order)
        points, base_w, ldofs, branch_ids, s_all = [], [], [], [], []
        for branch in system.geom.branches:
            primal_s = branch.length * _grid_breaks(branch.junction, branch.tip,
                                                     system.config)
            breaks = np.unique(np.round(np.r_[branch.s_lambda,
                                                self.dual.crossing_s[branch.branch_id],
                                                primal_s], 14))
            for a, b in zip(breaks[:-1], breaks[1:]):
                if b <= a + 1.0e-14:
                    continue
                sq = a + (b-a)*g
                local = np.clip(np.searchsorted(branch.s_lambda, sq, side="right")-1,
                                0, branch.n_lambda-1)
                points.append(branch.points(sq))
                base_w.append((b-a)*w)
                ldofs.append(branch.multiplier_dofs[local])
                branch_ids.append(np.full(len(sq), branch.branch_id, dtype=np.int64))
                s_all.append(sq)
        self.line_points = np.vstack(points)
        self.line_base_weights = np.concatenate(base_w)
        self.line_ldofs = np.concatenate(ldofs).astype(np.int64)
        self.line_branch_ids = np.concatenate(branch_ids)
        self.line_s = np.concatenate(s_all)

        nc = system.config.nx*system.config.ny
        st_r=[]; st_c=[]; st_v=[]
        eps=5.0e-11
        for branch in system.geom.branches:
            mask = self.line_branch_ids == branch.branch_id
            ids = np.flatnonzero(mask)
            pts = self.line_points[mask]
            for shifted, factor in ((pts+eps*branch.normal, 0.5),
                                    (pts-eps*branch.normal, 0.5)):
                cids=locate_cells(shifted,system.config)
                ix=cids%system.nx; iy=cids//system.nx
                xi=(shifted[:,0]-ix*system.config.hx)/system.config.hx
                eta=(shifted[:,1]-iy*system.config.hy)/system.config.hy
                local=(xi>=0.5).astype(int)+2*(eta>=0.5).astype(int)
                st_r.extend((4*cids+local).tolist())
                st_c.extend(self.line_ldofs[ids].tolist())
                st_v.extend((factor*self.line_base_weights[ids]).tolist())
        self.source_t_map = sp.coo_matrix(
            (st_v,(st_r,st_c)), shape=(4*nc,system.n_l)).tocsr()

        cids=locate_cells(self.line_points,system.config)
        vals,_=q1_values_grads(self.line_points,cids,system.config)
        sp_r=[]; sp_c=[]; sp_v=[]
        for a in range(4):
            sp_r.extend((4*cids+a).tolist())
            sp_c.extend(self.line_ldofs.tolist())
            sp_v.extend((self.line_base_weights*vals[:,a]).tolist())
        self.source_phi_map = sp.coo_matrix(
            (sp_v,(sp_r,sp_c)), shape=(4*nc,system.n_l)).tocsr()

        # Linear map from local sub-CV loads to global nodal CV loads.
        gr=np.concatenate([system.cell_nodes[:,a] for a in range(4)])
        gc=np.concatenate([4*np.arange(nc)+a for a in range(4)])
        self.local_to_cv = sp.coo_matrix(
            (np.ones(len(gr)),(gr,gc)), shape=(system.n_m,4*nc)).tocsr()

        # Reusable matrix/fracture transport connection topology.
        cv=[]; fd=[]; coeff=[]; conn_ldof=[]; conn_branch=[]
        for branch in system.geom.branches:
            mask=self.line_branch_ids==branch.branch_id
            ids=np.flatnonzero(mask); pts=self.line_points[mask]
            sq=self.line_s[mask]
            elem=np.clip(np.searchsorted(branch.s_gamma,sq,side="right")-1,
                         0,branch.n_gamma-1)
            ell=branch.s_gamma[elem+1]-branch.s_gamma[elem]
            phi0=(branch.s_gamma[elem+1]-sq)/ell
            phi1=(sq-branch.s_gamma[elem])/ell
            plus=np.array([_dual_node_for_point(system,p+eps*branch.normal) for p in pts])
            minus=np.array([_dual_node_for_point(system,p-eps*branch.normal) for p in pts])
            for k,line_id in enumerate(ids):
                mfrac={int(plus[k]):0.5,int(minus[k]):0.5}
                if plus[k]==minus[k]: mfrac={int(plus[k]):1.0}
                for mid,mw in mfrac.items():
                    for fid,fw in ((int(branch.pressure_dofs[elem[k]]),float(phi0[k])),
                                   (int(branch.pressure_dofs[elem[k]+1]),float(phi1[k]))):
                        value=float(self.line_base_weights[line_id]*mw*fw)
                        if abs(value)>1.0e-16:
                            cv.append(mid); fd.append(fid); coeff.append(value)
                            conn_ldof.append(int(self.line_ldofs[line_id]))
                            conn_branch.append(branch.branch_id)
        self.conn_cv=np.asarray(cv,dtype=np.int64)
        self.conn_frac=np.asarray(fd,dtype=np.int64)
        self.conn_coeff=np.asarray(coeff)
        self.conn_ldof=np.asarray(conn_ldof,dtype=np.int64)
        self.conn_branch=np.asarray(conn_branch,dtype=np.int64)

    def _build_face_quadrature(self):
        g,w=_gauss01(self.face_order); edge=self.dual.p1-self.dual.p0
        ell=np.linalg.norm(edge,axis=1)
        self.face_q_points=(self.dual.p0[:,None,:]+g[None,:,None]*edge[:,None,:]).reshape(-1,2)
        self.face_q_cell_ids=np.repeat(self.dual.host,self.face_order)
        self.face_q_face_ids=np.repeat(np.arange(len(edge),dtype=np.int64),self.face_order)
        self.face_q_normal_weights=(
            self.dual.normals[:,None,:]*(ell[:,None]*w[None,:])[:,:,None]
        ).reshape(-1,2)

    def _build_qpl_response(self,face_batch=1000):
        """Precompute the linear face-flux response to every P0 panel."""
        W=np.zeros((len(self.line_points),self.system.n_l),dtype=DTYPE)
        W[np.arange(len(W)),self.line_ldofs]=self.line_base_weights
        out=np.zeros((len(self.dual.p0),self.system.n_l),dtype=DTYPE)
        y=self.line_points
        for start in range(0,len(self.dual.p0),face_batch):
            a=self.dual.p0[start:start+face_batch]
            b=self.dual.p1[start:start+face_batch]
            n=self.dual.normals[start:start+face_batch]
            d=b-a;ell=np.linalg.norm(d,axis=1)
            nr=np.column_stack((d[:,1],-d[:,0]))/ell[:,None]
            orient=np.sign(np.einsum("ij,ij->i",n,nr))
            r0=a[:,None,:]-y[None,:,:];r1=b[:,None,:]-y[None,:,:]
            cross=r0[:,:,0]*r1[:,:,1]-r0[:,:,1]*r1[:,:,0]
            dot=np.einsum("fqi,fqi->fq",r0,r1)
            angle=np.arctan2(cross,dot)
            ya=y[None,:,:]-a[:,None,:]
            distance=(d[:,None,0]*ya[:,:,1]-d[:,None,1]*ya[:,:,0])/ell[:,None]
            projection=np.einsum("fqi,fi->fq",ya,d)/(ell[:,None]**2)
            on_face=((np.abs(distance)<=5.0e-12)&(projection>1.0e-13)
                     &(projection<1.0-1.0e-13))
            angle[on_face]=0.0
            out[start:start+len(a)]=orient[:,None]*(angle@W)/(2.0*np.pi)
        self.qpl_response=out

    def lambda_weights(self, solution: dict) -> np.ndarray:
        return self.line_base_weights*np.asarray(solution["lambda"])[self.line_ldofs]

    def source_local(self, solution: dict) -> tuple[np.ndarray,np.ndarray]:
        lam=np.asarray(solution["lambda"])
        st=np.asarray(self.source_t_map@lam).reshape(-1,4)
        sph=np.asarray(self.source_phi_map@lam).reshape(-1,4)
        for cid,rate in enumerate(self.system.config.source_rate_cell):
            if rate!=0.0:
                st[cid]+=0.25*rate; sph[cid]+=0.25*rate
        return st,sph

    def cv_lambda(self, solution: dict) -> np.ndarray:
        local=np.asarray(self.source_t_map@np.asarray(solution["lambda"])).reshape(-1)
        return np.asarray(self.local_to_cv@local).reshape(-1)

    def connections(self, solution: dict) -> dict:
        E=self.conn_coeff*np.asarray(solution["lambda"])[self.conn_ldof]
        cv_lam=self.cv_lambda(solution)
        mismatch=float(np.max(np.abs(np.bincount(self.conn_cv,weights=E,
                                                  minlength=self.system.n_m)-cv_lam)))
        return {"cv":self.conn_cv,"frac":self.conn_frac,"E":E,
                "branch":self.conn_branch,"max_cv_mismatch":mismatch}

    def qpl_face(self, solution: dict) -> np.ndarray:
        return self.qpl_response@np.asarray(solution["lambda"])

    def cg_face(self, solution: dict) -> np.ndarray:
        cids=self.face_q_cell_ids
        _,grads=q1_values_grads(self.face_q_points,cids,self.system.config)
        grad=np.einsum("qid,qi->qd",grads,solution["p_m"][self.system.cell_nodes[cids]])
        coeff=self.system.config.kappa_cell[cids]*solution["mobility_cell"][cids]
        q=-coeff[:,None]*grad
        contrib=np.einsum("qd,qd->q",q,self.face_q_normal_weights)
        return np.bincount(self.face_q_face_ids,weights=contrib,
                           minlength=len(self.dual.p0))

    def step_view(self, solution: dict, cg_face=None, qpl_face=None):
        if cg_face is None: cg_face=self.cg_face(solution)
        if qpl_face is None: qpl_face=self.qpl_face(solution)
        cv_lam=self.cv_lambda(solution)
        p0=self.template.dual_problem
        problem=SimpleNamespace(**p0.__dict__)
        problem.ctx=solution
        problem.cv_lambda_h=cv_lam
        problem.cv_lambda_exact=cv_lam.copy()
        problem.qpl_h_face=qpl_face
        problem.qpl_exact_face=qpl_face.copy()
        problem.cg_face=cg_face
        problem.lambda_h_density=np.asarray(solution["lambda"]).copy()
        problem.lambda_h_nodal_density=np.r_[problem.lambda_h_density,
                                             problem.lambda_h_density[-1]]
        problem.source_quad_h_points=self.line_points
        problem.source_quad_h_weights=self.lambda_weights(solution)
        return BuiltProblem(problem,self.dual,self.template.cv_class,
                            self.template.cv_cut_count,self.template.source_mask,
                            self.template.boundary_mask,{},self.template.qpf)


class FastNLR:
    """Vectorized Deng reconstruction using frozen geometry and dynamic loads."""

    def __init__(self, system:YLCGSystem, dynamic:YDynamicGeometry):
        t0=time.perf_counter(); self.system=system; self.dynamic=dynamic
        self.nc=system.config.nx*system.config.ny
        G=_local_B(system,0,1.0)
        self.G_pinv=np.linalg.pinv(G,rcond=1.0e-10)
        self.mass_vec=np.full((self.nc,4),0.25*system.config.cell_area)
        self.mass_area=np.full(self.nc,system.config.cell_area)
        self._build_edge_geometry()
        self.build_s=time.perf_counter()-t0

    def _build_edge_geometry(self):
        system=self.system; g,w=_gauss01(3)
        ep=[];ew=[];ec=[];enb=[];en=[];ev=[]
        ip=[];iw=[];ic=[];inb=[];igi=[];inn=[]
        edge_pairs=((0,1),(1,3),(3,2),(2,0))
        for cid in range(self.nc):
            poly=system.nodes[system.cell_nodes[cid]]
            for edge,(la,lb) in enumerate(edge_pairs):
                a=poly[la];b=poly[lb];tangent=b-a
                normal=_unit(np.array([tangent[1],-tangent[0]]))
                nb=_neighbor_for_edge(system,cid,edge)
                pts=a[None,:]+g[:,None]*(b-a)[None,:]
                vals,_=q1_values_grads(pts,np.full(len(g),cid),system.config)
                ep.append(pts);ew.append(w*np.linalg.norm(b-a));ec.append(np.full(len(g),cid))
                enb.append(np.full(len(g),nb));en.append(np.tile(normal,(len(g),1)));ev.append(vals)
                mid=0.5*(a+b)
                for loc,start,end in ((la,a,mid),(lb,b,mid)):
                    pts=start[None,:]+g[:,None]*(end-start)[None,:]
                    ip.append(pts);iw.append(w*np.linalg.norm(end-start));ic.append(np.full(len(g),cid))
                    inb.append(np.full(len(g),nb));igi.append(np.full(len(g),loc));inn.append(np.tile(normal,(len(g),1)))
        self.ephi_points=np.vstack(ep);self.ephi_w=np.concatenate(ew)
        self.ephi_c=np.concatenate(ec).astype(np.int64);self.ephi_nb=np.concatenate(enb).astype(np.int64)
        self.ephi_n=np.vstack(en);self.ephi_vals=np.vstack(ev)
        self.ei_points=np.vstack(ip);self.ei_w=np.concatenate(iw)
        self.ei_c=np.concatenate(ic).astype(np.int64);self.ei_nb=np.concatenate(inb).astype(np.int64)
        self.ei_gi=np.concatenate(igi).astype(np.int64);self.ei_n=np.vstack(inn)

    def _kgrad(self,points,cids,p,coeff):
        _,grads=q1_values_grads(points,cids,self.system.config)
        grad=np.einsum("qid,qi->qd",grads,p[self.system.cell_nodes[cids]])
        return coeff[cids,None]*grad

    def _average_normal(self,points,cids,nids,normals,p,coeff):
        val=self._kgrad(points,cids,p,coeff);mask=nids>=0
        if np.any(mask):
            val[mask]=0.5*(val[mask]+self._kgrad(points[mask],nids[mask],p,coeff))
        return np.einsum("qd,qd->q",val,normals)

    def reconstruct(self,solution:dict,step_view=None,with_audit=False)->dict:
        t0=time.perf_counter();system=self.system
        coeff=system.config.kappa_cell*np.asarray(solution["mobility_cell"])
        p=np.asarray(solution["p_m"]);u=p[system.cell_nodes]
        st,sph=self.dynamic.source_local(solution)
        aterm=coeff[:,None]*np.einsum("ij,cj->ci",system.Ke_unit,u)
        ephi=np.zeros((self.nc,4));eI=np.zeros((self.nc,4))
        avg=self._average_normal(self.ephi_points,self.ephi_c,self.ephi_nb,
                                 self.ephi_n,p,coeff)
        weighted=self.ephi_w*avg
        for j in range(4):
            np.add.at(ephi[:,j],self.ephi_c,weighted*self.ephi_vals[:,j])
        avg=self._average_normal(self.ei_points,self.ei_c,self.ei_nb,
                                 self.ei_n,p,coeff)
        np.add.at(eI,(self.ei_c,self.ei_gi),self.ei_w*avg)
        rhs=st-sph+aterm+eI-ephi
        rec=np.einsum("ij,cj->ci",self.G_pinv,rhs)/np.maximum(coeff[:,None],1e-300)
        correction=(np.einsum("ci,ci->c",self.mass_vec,u)
                    -np.einsum("ci,ci->c",self.mass_vec,rec))/self.mass_area
        rec+=correction[:,None]
        cids=self.dynamic.face_q_cell_ids
        _,grads=q1_values_grads(self.dynamic.face_q_points,cids,system.config)
        grad=np.einsum("qid,qi->qd",grads,rec[cids])
        q=-coeff[cids,None]*grad
        contrib=np.einsum("qd,qd->q",q,self.dynamic.face_q_normal_weights)
        face=np.bincount(self.dynamic.face_q_face_ids,weights=contrib,
                         minlength=len(self.dynamic.dual.p0))
        result={"coeffs":rec,"rhs":rhs,"face_flux":face,
                "wall_s":time.perf_counter()-t0}
        if with_audit:
            if step_view is None: step_view=self.dynamic.step_view(solution)
            result["audit"]=audit_flux(step_view,face,"Fast NLR conservation audit")
        return result


class NoFractureQ1System:
    """Case-1 matrix solve on the identical Q1 grid, without fracture blocks."""

    def __init__(self, fractured_system:YLCGSystem):
        self.parent=fractured_system
        self.config=fractured_system.config
        self.n_m=fractured_system.n_m
        self.nodes=fractured_system.nodes
        self.cell_nodes=fractured_system.cell_nodes
        self.Ke_unit=fractured_system.Ke_unit
        self.load=fractured_system.load
        self.boundary_matrix_dofs=fractured_system.boundary_matrix_dofs
        boundary=np.zeros(self.n_m,dtype=bool);boundary[self.boundary_matrix_dofs]=True
        self.free=np.flatnonzero(~boundary)

    def solve(self,mobility_cell=None)->dict:
        t0=time.perf_counter()
        if mobility_cell is None:
            mobility_cell=np.ones(self.config.nx*self.config.ny)
        mobility_cell=np.asarray(mobility_cell,dtype=DTYPE)
        coeff=self.config.kappa_cell*mobility_cell
        rows=np.repeat(self.cell_nodes,4,axis=1).reshape(-1)
        cols=np.tile(self.cell_nodes,(1,4)).reshape(-1)
        vals=(coeff[:,None,None]*self.Ke_unit[None,:,:]).reshape(-1)
        A=sp.coo_matrix((vals,(rows,cols)),shape=(self.n_m,self.n_m)).tocsr()
        p=np.zeros(self.n_m);Af=A[self.free][:,self.free].tocsc()
        p[self.free]=splu(Af).solve(self.load[self.free])
        return {"p_m":p,"mobility_cell":mobility_cell,
                "solve_s":time.perf_counter()-t0}


def one_matrix_transport_step(system:YLCGSystem,built:BuiltProblem,face_flux:np.ndarray,
                              S:np.ndarray,dt:float,M=1.0)->tuple[np.ndarray,dict]:
    p=built.dual;q=np.asarray(face_flux);owner=p.owner;nb=p.neighbor
    up=np.empty_like(q);pos=q>=0;up[pos]=S[owner[pos]]
    interior=(~pos)&(nb>=0);up[interior]=S[nb[interior]]
    up[(~pos)&(nb<0)]=0.0
    water=q*fractional_flow(up,M);outward=scatter_flux(p,water)
    well=np.zeros(system.n_m)
    for cid,rate in enumerate(system.config.source_rate_cell):
        if rate==0:continue
        nodes=system.cell_nodes[cid];share=rate/4
        if rate>0:well[nodes]+=share*fractional_flow(1.0,M)
        else:well[nodes]+=share*fractional_flow(S[nodes],M)
    Snew=S+dt*(-outward+well)/dual_areas(system)
    inj=np.unique(system.cell_nodes[np.flatnonzero(system.config.source_rate_cell>0)].ravel())
    Snew[inj]=1.0
    report={"matrix_mass_change":float(np.sum((Snew-S)*dual_areas(system))),
            "producer_watercut":producer_watercut(system,Snew,M),
            "min_before_clip":float(np.min(Snew)),"max_before_clip":float(np.max(Snew))}
    return np.clip(Snew,0,1),report


def matrix_cfl_dt(system:YLCGSystem,built:BuiltProblem,face_flux:np.ndarray,
                  cfl=0.45,fprime_max=2.0)->float:
    p=built.dual;q=np.asarray(face_flux)
    out=np.bincount(p.owner,weights=np.maximum(q,0),minlength=system.n_m)
    mask=p.neighbor>=0
    out+=np.bincount(p.neighbor[mask],weights=np.maximum(-q[mask],0),minlength=system.n_m)
    area=dual_areas(system)
    return float(cfl*np.min(area[out>1e-30]/(fprime_max*out[out>1e-30])))
