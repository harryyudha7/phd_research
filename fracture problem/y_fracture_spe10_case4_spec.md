# Instructions: Case 4 — Y-shaped fracture in SPE10 layer 20 (stream-function PINN + NLR)

## Goal

New notebook `LCG_Yfracture_spe10_L20_hardcurl_PINN_NLR.ipynb` in
`fenicsx/code/fracture problem/`. It is the CASE-1 CONFIGURATION PLUS A FRACTURE
NETWORK: clone the flow setup of
`LCG_DengGinting_example4_spe10_Q1_no_fracture_hardcurl_PINN.ipynb` exactly
(domain (0,1)^2, 64x64 Q1 grid, SPE10 layer-20 cell permeability from the same
MRST export, homogeneous Dirichlet p=0 on the whole boundary, the two Deng
source points x_I=(0.2,0.4) rate +1.0 and x_P=(0.8,*) rate -5.0 with the same
source machinery and particular field q_p_f), and add the Köppel–Martin Case-2
Y-shaped fracture. Compare CG / NLR / PINN in the COUPLED SEQUENTIAL transport
setting of case 1 (K a(S) updated and the flux reconstructed EVERY saturation
step; PINN updates via the PoU head), ending with the PINN-vs-NLR wall-time
comparison. No MRST arm (the boundary-tip case carries the cross-code
comparison; NLR is the conservative reference here). No water-balance
functionals.

Inherit the case-1 configuration PROGRAMMATICALLY (read the source points,
rates, layer, and kappa from the same export/constants the case-1 notebook
uses) — do not re-type numbers.

## Fracture geometry and properties

- Y-shape per Köppel–Martin (2018) Case 2: three straight branches meeting at a
  junction; default coordinates junction J=(0.5,0.5), tips
  T1=(0.25,0.25), T2=(0.75,0.75), T3=(0.25,0.75). All three tips are INTERIOR
  and SEALED (homogeneous Neumann in the fracture pressure solve, zero tip
  throughflow in transport).
- Placement check before anything runs: the branch T1–J–T2 must intercept the
  injector-to-producer flow path. Overlay the Y on the case-1 pressure/streamline
  field; if the junction region is not between the wells, translate the whole Y
  (do not change its shape) and record the final coordinates.
- Fracture conductance: K_Gamma = 2 x max(kappa_cell) over the layer, one value
  for all branches, stated as the effective (aperture-included) conductance.
- Junction conditions in the 1D fracture system: pressure continuity and flux
  balance (Kirchhoff) among the three branches at J. Fracture pressure P1 with
  h_Gamma ~ h/2 per branch; multiplier P0 with h_lambda ~ 2h per branch, on
  independent branch meshes (same conventions as the boundary-tip case).

## Fixed conventions

- One fixed normal orientation PER BRANCH, used consistently in jump, single
  layer, traces, and audit RHS. Branch arclengths measured FROM THE JUNCTION
  (so the three tips are the far ends; state this once).
- float64, fixed seeds, fracture-aware audit convention (shared functions from
  `fracture_hardcurl_common.py`); every gate prints PASS/FAIL and STOPS the
  notebook on failure.

## Stage 1 — CG-LMDFM solve and gates

1. Assemble the case-1 matrix problem + three-branch fracture coupling.
   Junction assembly is the only new solver element; the branch machinery is
   the boundary-tip notebook's.
2. Sealed-network gate: with sealed tips and no fracture source, the exchange
   summed over ALL THREE branches must vanish: sum_b int_{Gamma_b} lambda_h = 0
   to stated tolerance. (Per-branch integrals need NOT vanish — the junction
   transfers flux between branches; report the three per-branch integrals and
   their sum.)
3. Junction gate (fracture-flux Proposition at the junction node): the
   three-branch discrete balance r_J = sum of the three two-point fluxes into J
   + junction-CV exchange integral = 0 to ~1e-12, alongside the standard
   interior r_j gate on every branch (the accepted-step/no-searchsorted lesson:
   build q_elem as -K_F * diff(p)/diff(s) per branch).
4. Gate A0 at an INTERIOR tip: mirrored-pair jump test for a single-layer
   segment END inside the domain (density -> jump relation as the pair
   approaches an interior point of the segment; also verify the field decays
   smoothly beyond the tip rather than jumping).
5. Gate A1 (training-free conservation identity): q = q_p_f + sum of the three
   branch single-layer fields + curl(psi), psi in {0, random}; max |R_xi| <=
   1e-13 on ALL dual-CV classes: interior, single-branch-cut, MULTI-CUT
   (junction-adjacent CVs crossed by >= 2 branches — enumerate them
   explicitly), source (well) CVs, boundary. Faces crossing any branch are
   split at every crossing (a face may cross two branches near J — the split
   machinery must handle multiple crossing points per face).

## Stage 2 — PINN

- q_theta = q_p_f + sum_b q_p_lambda_b + curl(psi_theta). Note in a markdown
  cell: this is the first case combining the SINGULAR (point-source well)
  particular field with fracture single-layer fields (case 1: wells, no
  fracture; MMS: fracture + smooth distributed source; boundary-tip: fracture,
  f = 0).
- Features and network as case 1 (width 32, depth 3, SiLU, Fourier
  frequencies as in case 1, log-kappa feature WITH the kappa-tilde
  interpolation fix; the kappa feature uses matrix permeability only — the
  fracture enters through lambda_h, never the feature).
- Canonical loss: dual-face integrated CG target fluxes + pointwise CG samples,
  weights as in the MMS/KM canonical runs.
- Budget: Adam 2000 + L-BFGS up to 3000 with the 1e-6 / 200-accepted-step
  relative stagnation rule; persist loss history and termination metadata.
- The network receives NO junction information of any kind — that is the
  point; state it in the markdown.

## Stage 3 — NLR

- Batched Deng NLR with node-split machinery extended to MULTI-BRANCH splits:
  dual faces near J may cross two branches; split at every crossing.
- Gates: interior and adjacent <= 1e-12, single-cut <= 1e-11; MULTI-CUT
  reported with the same 1e-11 gate (if it fails, report — that is a finding
  about the multi-branch split, not something to hide); boundary/source per
  case-1 conventions.

## Stage 4 — Diagnostics (one JSON: `y_case4_conservation.json`)

For CG / NLR / PINN: R_xi mean/max per class {interior, single-cut, multi-cut,
source, boundary}; R_tau native summary for the PINN; the per-branch exchange
integrals and their sum; the junction gate residual; PINN training metadata.
Also per-branch lambda_h arrays and the PINN jump sampled along each branch
(for the figure and the jump==lambda_h verification: report max deviation per
branch INCLUDING the panels nearest the junction).

## Stage 5 — Coupled sequential transport (K(S) updated, flux refreshed EVERY step)

This is the case-1 coupled setting (the IMPES-style sequential loop of the
simulator / exp5 campaign), now with the fracture network. NOT frozen-velocity.

- Coefficient coupling exactly as case 1 / `impes_spe10_simulator.py`:
  a(S) = M S^2 + (1-S)^2, F(S) = M S^2 / a(S), M as in the case-1 campaign
  (M=1 unless the user says otherwise); the mobility multiplies BOTH the matrix
  permeability (K a(S)) and the fracture conductance (K_Gamma a(S_Gamma)).
- Per time step, for EACH track (CG / NLR / PINN):
  1. re-solve the fractured CG-LMDFM pressure with the current coefficients
     -> p_h, p_Gamma_h, lambda_h^n;
  2. rebuild the flux: CG = raw flux; NLR = batched local reconstruction with
     multi-branch node-split; PINN = PoU closed-form refresh (below);
  3. one explicit upwind transport step on the 65x65 dual grid + three-branch
     fracture transport (junction mixing rule: junction node mixes inflowing
     branch fluxes flux-weighted and supplies F(S_J) to outflowing branches;
     zero tip throughflow), S=1 held at the injector CV, producer sink at
     F(S_upwind), Dirichlet-boundary faces admit resident fluid on inflow.
- PINN per-step update = PoU HEAD (the paper's sequential-update method):
  train psi_theta ONCE at t=0 (Stage 2), convert to the PoU head with the
  case-1 production settings (16x16 C1 cosine windows, r=16, ridge anchored to
  the previous step's coefficients); each step recompute q_p_lambda from
  lambda_h^n (closed-form panels — part of the PINN flux stage, time it) and
  refresh theta by the factorize-once closed-form update against the new
  dual-face CG targets. Full L-BFGS refresh is NOT run in the loop (one
  optional spot-check at a single step to bound the PoU accuracy gap).
- Time stepping as the simulator: fixed transport dt chosen from the initial
  CFL with the usual safety factor; per-step CFL monitored and violations
  REPORTED, never aborted. Record everything needed to reproduce.
- Duration: run to producer water-cut breakthrough and beyond (report the
  breakthrough time, water-cut at producer CV > 1e-3); ~3 saturation snapshots
  at recorded absolute times (front reaching the Y / past the junction / after
  breakthrough).
- NO-FRACTURE TWIN: the identical coupled loop without the fracture (case-1
  configuration verbatim) for the PINN track at the same dt and snapshot
  times — the fracture-effect comparison.
- Accuracy endpoint as in exp5: Delta-S at snapshots vs the NLR track (the
  conservative reference), plus per-branch S_Gamma and the PRODUCER WATER-CUT
  CURVES vs time for all three tracks (the engineering-facing output).
- Outputs `y_case4_transport.npz`: all tracks incl. twin, S_Gamma per branch,
  water-cut curves, dt/CFL/breakthrough metadata, per-step face-RMSE of each
  reconstructed flux vs its own CG targets.

## Stage 6 — Wall-time comparison (PINN-PoU vs NLR)

- Granularity rule (established): time the FULL FLUX STAGE per step, defined as
  everything between "pressure solved" and "transport-ready face fluxes",
  including for the PINN: q_p_lambda panel rebuild + target/RHS assembly + PoU
  solve + face-flux evaluation; and for NLR: node-split geometry reuse + local
  solves + assembly. Exclude the shared CG pressure solve and lambda_h
  extraction (identical for both). Report mean/median/p95 per step over the
  whole run, on the same machine, single-threaded, plus the one-time costs
  (PINN training at t=0 + PoU factorization; NLR geometry construction).
- Deliverables: a small table (one-time cost, per-step flux stage, steps run)
  and a cumulative-cost-vs-steps line figure with the crossover step marked
  (the case-1 amortization figure, now fractured).
- Honesty guards: both single-threaded; note NLR's per-step cost is the current
  implementation, not a floor; record machine info in the JSON.

## Figures (dpi 600 lines / 300 fields; absolute time labels, never PVI; no
"hard-curl" anywhere)

- `fig_y_setup.png`: kappa map (log scale) + Y overlay + wells + pressure
  contours.
- `fig_y_exchange_jump.png`: three panels (one per branch, junction at s=0):
  lambda_h and the PINN jump overlaid (they must coincide to machine
  precision; state max deviation), tips marked.
- `fig_y_saturation.png`: 3 methods x 2-3 times, shared colorbar.
- `fig_y_dS_vs_NLR.png`: CG-NLR and PINN-NLR at the final snapshot.
- `fig_y_fracture_effect.png`: PINN(case 4) - PINN(no-fracture twin) at 2
  times — the effect of the Y on the flood.
- `fig_y_sgamma.png`: per-branch fracture saturation at the snapshots
  (junction at s=0 so the three profiles share the origin).
- `fig_y_watercut.png`: producer water-cut vs time, three tracks.
- `fig_y_cumcost.png`: cumulative flux-stage cost vs steps, PINN-PoU vs NLR,
  one-time costs as offsets, crossover step marked.

## Guardrails

- Do not modify the case-1 notebook, the KM notebook, or the shared module
  beyond ADDING multi-branch-capable functions (extend, never change existing
  signatures/behavior — the KM notebook must still run).
- A failed gate stops the notebook. No penalties in the loss under any
  circumstance.
- Everything the paper quotes must be in the JSON; figures carry no numbers
  that are not also in the JSON.

## Out of scope

MRST/EDFM, water balance, convergence-in-h study, rotated-zeta volumes,
full-network L-BFGS refresh inside the loop (spot-check only), the hydrocoin
geometry, any paper text.
