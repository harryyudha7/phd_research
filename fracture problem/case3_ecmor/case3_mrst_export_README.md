# `case3_mrst_export.mat` — field dictionary

MRST EDFM solution of the Köppel–Martin boundary-tip fracture benchmark (Case 3),
exported for cross-validation against the FEM/CG + PINN notebook.

- **Format:** MATLAB `-v7` → read with `scipy.io.loadmat` (no `h5py` needed).
- **Units:** fully **dimensionless**. Matrix `k_m = 1`, `μ = 1`, so MRST's integrated
  face flux `face_flux` **equals** `∫_face v·n ds` — directly comparable to the
  notebook's flux integrals (which also use `k=1`, `μ=1`). No rescaling needed.
- **Indexing:** every `*_cell`, `*_neighbors`, `tip_cells` value is a **MATLAB
  1-based** index. These are metadata only — all comparisons use the exported
  *geometry*, not indices. Subtract 1 if you ever index a Python array with them.
- **Geometry:** matrix grid is `128×128` on the unit square (identical to the FEM
  `n_bulk=128` mesh), so every matrix face here corresponds to a notebook edge.

## Loading in Python

```python
from scipy.io import loadmat
m = loadmat('case3_mrst_export.mat', squeeze_me=True)   # squeeze_me flattens (n,1)->(n,)
p_matrix   = m['p_matrix']       # (16384,)
xc_matrix  = m['xc_matrix']      # (16384, 2)
face_p1    = m['face_p1']        # (33024, 2)  ... etc
```

---

## (1) Pressure

| field | shape | meaning |
|---|---|---|
| `p_matrix`  | (Nm,)  | matrix cell pressures (cell-wise constant FV values) |
| `xc_matrix` | (Nm,2) | matrix cell centroids `(x,y)` in `[0,1]²` |
| `p_frac`      | (Nf,)  | fracture cell pressures |
| `xc_frac`     | (Nf,2) | fracture cell centroids |
| `s_frac`      | (Nf,)  | **fracture arc-length coordinate** along Γ from `A` (the notebook's fracture grid coordinate; monotonic, 0→1.1136) |
| `s_frac_arc`  | (Nf,)  | alias of `s_frac` (kept for back-compat / `lam_s`) |

`Nm = 16384` (matrix cells), `Nf = 125` (fracture cells).

> **Naming (important):** `s_frac` is the fracture **arc-length coordinate**, NOT a
> saturation. Water saturations are `sw_frac` / `sw_matrix` (section 4). All per-fracture
> arrays (`s_frac`, `sw_frac`, `lam_*`) share one cell ordering, monotonic in `s_frac`.

**Compare:** sample the FEM `p_m` at `xc_matrix` and difference. Expect close
agreement in the bulk; a band of disagreement near Γ (EDFM finite normal coupling
vs FEM exact pressure continuity — a *model* difference, not a bug). For the
fracture, compare FEM `p_Γ(s)` against `p_frac` vs `s_frac_arc`.

## (2) Matrix face-normal flux  → use for the PINN comparison

All faces of the matrix grid (interior **and** boundary). Fracture-internal faces
are excluded.

| field | shape | meaning |
|---|---|---|
| `face_p1`, `face_p2` | (F,2) | the two endpoints of each face (a segment) |
| `face_centroid`      | (F,2) | face midpoint |
| `face_normal`        | (F,2) | **unit** normal, oriented from `face_neighbors[:,0]` → `[:,1]` (outward on boundary) |
| `face_len`           | (F,)  | face length |
| `face_flux`          | (F,)  | MRST conservative flux `∫_face v·n ds`, measured along `face_normal` |
| `face_neighbors`     | (F,2) | the two matrix cells sharing the face (`0` = exterior) |
| `face_is_boundary`   | (F,)  | 1 if a domain-boundary face |
| `face_frac_cut`      | (F,)  | 1 if the face touches a fracture-cut matrix cell |

`F = 33024`, of which `448` are fracture-cut.

**Compare with the PINN flux `v_θ`:** for each face, integrate `v_θ·n` along the
segment `face_p1→face_p2` with the **same** `face_normal` (so signs match `face_flux`
automatically), choosing `q_plus_net`/`q_minus_net` per quadrature point by the sign
of `φ = (x − A)·n_Γ`:

```python
import numpy as np
gp = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)]);  gw = np.array([5/9, 8/9, 5/9])
def pinn_face_flux(p1, p2, n):
    mid, half = 0.5*(p1+p2), 0.5*(p2-p1)
    xq = mid[None,:] + gp[:,None]*half[None,:]      # 3 quadrature points
    vq = eval_v_theta(xq)                            # (3,2): pick q_plus/q_minus by sign((xq-A)@n_gamma)
    L  = np.linalg.norm(p2-p1)
    return (L/2) * np.sum(gw * (vq @ n))             # ∫ v_θ·n ds
```

Recommended plots: scatter `flux_pinn` vs `face_flux` (and `flux_cg` vs `face_flux`
for contrast) against `y=x`; histograms of the difference split by `face_frac_cut`;
spatial map of `|flux_pinn − face_flux|`. Story: bulk agreement everywhere; near Γ,
does the PINN correction track the independent conservative MRST flux?

## (3) Lambda — matrix↔fracture exchange (EDFM analog of `λ_h`)

Per fracture cell, the net flux exchanged with the matrix.

| field | shape | meaning |
|---|---|---|
| `lam_s`       | (Nf,) | arc length along Γ (= `s_frac_arc`) |
| `lam_xy`      | (Nf,2)| physical location (= `xc_frac`) |
| `lam_seglen`  | (Nf,) | fracture cell length `Δs` |
| `lam_flux`    | (Nf,) | net **matrix→fracture** flux for the cell. **Sign: + = matrix INTO fracture** |
| `lam_density` | (Nf,) | `lam_flux / lam_seglen` — the `λ(s)` density to compare with FEM `λ_h(s)` |

Raw NNC-level data (transparency; one row per matrix–fracture connection):
`nnc_mat_cell`, `nnc_frac_cell`, `nnc_flux_m2f` (matrix→fracture), `nnc_s`.

**Compare:** overlay `lam_density` vs `lam_s` against the FEM `λ_h(s)`. Mind the sign
convention (flip if the notebook defines `λ` as fracture→matrix). `sum(lam_flux)≈0`:
the through-fracture transport is carried by the tips, not by net matrix exchange.

## (3b) Fracture along-flux `Q_par` (fracture–fracture internal faces)

The exact tangential flux between adjacent fracture cells — use this directly as the
along-fracture flux, **instead of reconstructing it from a `p_frac` gradient**.

| field | shape | meaning |
|---|---|---|
| `frac_face_flux`      | (Ff,)  | `∫ v·n` across each fracture-internal face (signed along stored normal `neighbor[0]→[1]`) |
| `frac_face_neighbors` | (Ff,2) | the two fracture cells (global 1-based indices, both `> meta_nc`) |

`sign`: positive = flux from `frac_face_neighbors[:,0]` to `[:,1]`. Per-cell single-phase
balance closes to machine precision: `Σ(frac face flux out) − Σ(NNC into cell) − tip = 0`.

## (4) MRST-native transport saturation (validation target)

Buckley–Leverett `f(S)=S²/(S²+(1−S)²)`, fixed velocity, run to `PVI = meta_PVI`.
`S` = **water saturation**.

| field | shape | meaning |
|---|---|---|
| `sw_matrix` | (Nm,) | matrix Sw at PVI — **from MRST's own `explicitTransport`** (independent SINTEF solver) |
| `sw_frac`   | (Nf,) | fracture Sw at PVI — **from MRST's own `explicitTransport`** (ordered by `s_frac`) |
| `s_matrix`  | (Nm,) | = `sw_matrix` (kept as-is; the matrix name never collided) |
| `sw_matrix_matched`, `sw_frac_matched` | (Nm,)/(Nf,) | reference: hand-coded explicit upwind *matched* to the notebook's scheme |
| `s_matrix_matched` | (Nm,) | = `sw_matrix_matched` (kept) |

> ⚠️ Use **`sw_frac`** for fracture saturation. `s_frac` is the arc-length coordinate (§1).

`meta_transport_solver` names the solver. **`sw_matrix`/`sw_frac` are from MRST's
INDEPENDENT solver** — not the notebook's scheme — so comparing the notebook's rect-MRST
against them is a genuine cross-check, *not circular*. The independent solver and the
matched-scheme reference agree to RMSE **2.1e-4** (matrix) / **9.3e-6** (fracture), so the
field is corroborated two ways. Both: matrix Sw∈[0,0.9955], fracture Sw∈[0,0.2126].
Scheme: explicit 1st-order upwind, `f(S)=S²/(S²+(1−S)²)`, `CFL=0.45`, `FPRIME_MAX=2`,
matrix-PV PVI basis, `T_final=0.47938` (single-phase fixed velocity).

**Validate:** RMSE of the notebook's rect-MRST `S_Γ`/`S` against `sw_frac`/`sw_matrix`
(use `s_frac` only as the fracture coordinate). If large, suspect (a) along-flux source —
use `frac_face_flux` not a `p_frac` gradient; (b) storage — fracture is **aperture-free**
`φ·ℓ`; (c) tip-flux sign.

## Tip boundary condition

The fracture endpoint Dirichlet (`p_Γ(A)=1`, `p_Γ(B)=4`) is imposed with high-WI
bhp wells on the two tip fracture cells (penalty enforcement; pins p to ~3e-4).

| field | shape | meaning |
|---|---|---|
| `tip_cells` | (2,) | global cell index, **order `[B, A]`** (B near `(0.75,1)`, A near `(0.25,0)`) |
| `tip_flux`  | (2,) | well flux, **`[+28.09, −28.09]`** (`+`=injection). B injects **oil** (`S=0`); A produces |
| `tip_xy`    | (2,2)| tip cell locations, order `[B, A]` |

In transport: tip B is an oil inflow (`S=0`), tip A a producer. If you build a
control-volume residual containing a tip cell, include `tip_flux` as a source term
(those two cells are Dirichlet/source cells, not homogeneous).

## Meta

`meta_celldim` `[128 128]`, `meta_physdim` `[1 1]`, `meta_fracA` `[0.25 0]`,
`meta_fracB` `[0.75 1]`, `meta_tau` (unit tangent A→B), `meta_aperture` `0.01`,
`meta_kf` `1000`, `meta_km` `1`, `meta_Kgamma` `10`, `meta_nc` `16384`,
`meta_nfrac` `125`, `README` (inline summary string).

**Transport timing / storage** (so the notebook runs to the identical physical time):
`meta_PVI` `1.2`, `meta_T_final` `0.47938` (notebook's exact value), `meta_phi_matrix` `1.0`,
`meta_phi_fracture` `1.0` (aperture-free), `meta_Q_water` `2.5032` (water rate at x=1 inlet),
`meta_PV_matrix` `1.0`, `meta_PV_frac` `1.1180`, `meta_PV_total` `2.1180`,
`meta_CFL` `0.45`, `meta_FPRIME_MAX` `2.0`, `meta_dt` `3.58e-5`, `meta_nsteps` `13384`.
PVI basis is **matrix PV only** (`PVI = meta_Q_water · t / meta_PV_matrix`, `PV_matrix=1.0`).
Match `meta_T_final` to compare at the same physical time.

> **Caveat to keep in the writeup:** MRST EDFM and the FEM solve the same matrix
> PDE/BCs but *different* fracture-coupling models (EDFM finite CI vs FEM exact
> pressure continuity). Expect strong bulk agreement and trend-level (not
> machine-level) agreement near Γ.
