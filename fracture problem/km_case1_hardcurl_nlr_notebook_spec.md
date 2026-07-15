# Instructions: Köppel–Martin Case 1 (sealed tips) — stream-function PINN (Option A) + NLR notebook

## Goal

Recreate `LCG_nonconforming_PINN_flux_reconstruction_KoppelMartin_case1_2d_neumann_tips.ipynb`
as a NEW notebook

`LCG_KoppelMartin_case1_neumann_tips_hardcurl_PINN_NLR.ipynb`

in `fenicsx/code/fracture problem/`, with two changes of substance:

1. The two-network ghost-partition penalty PINN is REPLACED by the Option-A
   stream-function reconstruction (one network; jump carried by the analytic
   single-layer field of lambda_h; conservative by construction; NO conservation
   penalties, no ramp, no matrix_only/full staging).
2. A batched Deng NLR track is ADDED, so the notebook produces all four fluxes
   (CG, NLR, PINN, MRST/EDFM) on the same solve, audited under the same
   convention, and compared in the same transport driver.

Do NOT modify the source notebook or any MMS notebook. This notebook produces the
paper's boundary-tip study case (diagnostics table + transport figures); the
old notebook's water-balance content is dropped by decision.

## Sources (port, do not re-derive)

- Problem setup, sealed-tip CG-LMDFM solve, dual/CV machinery, rotated-zeta
  machinery, MRST validation, transport driver, fracture diagnostics, figures:
  the source notebook above. KEEP its defining variant: homogeneous Neumann at
  both fracture tips in the pressure solve AND zero fracture-tip throughflow in
  transport; MRST reference from `case3_ecmor/case3_mrst_export_noflow_v2.mat`
  (additive superset of v1: same field names plus a conventions struct, per-face
  geometry/neighbors, per-connection NNC triplets `nnc_mat_cell`/`nnc_frac_cell`/
  `nnc_flux_m2f` (positive = matrix→fracture), 10 transport snapshots
  `sw_matrix_snaps`/`sw_frac_snaps` at `snap_PVI`/`snap_T_abs`, conversion
  `meta_PVI_to_Tabs` = 0.28877, and recorded sealed-tip checks).
- Option-A machinery (q_p_f, single-layer q_p_lambda with closed-form P0 segment
  integrals, Fourier-feature network, endpoint-difference face fluxes, gates
  A0/A1, canonical loss, trainer with the accepted-step logger):
  `fracture_hardcurl_common.py` and the MMS AB notebooks
  (`LCG_fracture_MMS_hardcurl_PINN_*_AB.ipynb`). Reuse the shared module; do not
  fork it.
- Batched NLR with fracture node-split machinery: the KM NLR notebook (the one
  whose node-split integration achieved machine-eps on fracture-cut dual CVs)
  and/or the batched Deng port in `impes_spe10_simulator.py`.

## Fixed conventions

- ONE fracture normal orientation, consistent across jump definition,
  single-layer field, one-sided traces, and every audit RHS (as in the MMS
  notebooks). float64 everywhere; fixed recorded seeds.
- Fracture-aware conservation statement for every audit:
  R(omega) = closed-boundary flux integral − ∫_omega f − ∫_{Gamma∩omega} lambda_h.
  There is NO exact lambda in this benchmark: lambda_h RHS only.
- Faces crossing Gamma: split integrals at the crossing point (single-layer
  normal trace jumps there). R_tau in the NATIVE evaluation (psi endpoint
  differences + analytic/high-order q_p face integrals).
- The audit convention must match the MMS notebooks (the corrected one), NOT the
  source notebook's legacy residual cells — re-derive every reported residual
  with the shared audit functions; reuse no legacy numbers.

## Stage 1 — CG solve and gates

1. Run the sealed-tip CG-LMDFM solve on the UNIFORM 64×64 primal mesh (standard
   mesh, no staggering). The transport/comparison grid is its 65×65
   vertex-centered dual. The MRST reference stays at 128×128: because dual-CV
   faces lie on lines at odd multiples of 1/128, every dual CV is an EXACT union
   of MRST cells (2×2 interior, 1×2 edge, 1×1 corner) and every dual face an
   exact union of MRST faces. Geometric gate (before any aggregation): assert
   this decomposition by index arithmetic to round-off. P0 multiplier with
   h_lambda ≈ 2h as in the source.
2. Sanity gate (new, cheap): sealed tips imply the net exchange balances —
   ∫_Gamma lambda_h ds must equal the net fracture source (zero here). Print and
   assert to a stated tolerance.
3. Gate A0 (single-segment jump test) rerun for THIS geometry: the fracture
   endpoints A=(0.25,0), B=(0.75,1) lie ON the domain boundary — the mirrored
   point pairs near the tips must stay inside Omega; verify the jump relation
   still converges to the density.
4. Gate A1 (training-free conservation identity, psi ≡ 0 and random psi): max
   |R_xi| ≤ 1e-13 on ALL dual-CV classes (interior / fracture-cut /
   fracture-adjacent / boundary). Gate failure STOPS the notebook.

## Stage 2 — Option-A PINN

- q = q_p_f + q_p_lambda(lambda_h) + curl(psi_theta); single network, SMALL
  (width 32, depth 3, SiLU, Fourier frequencies [1,2,4,8]; no log-kappa feature,
  K = I).
- Canonical loss: dual-face integrated CG target fluxes (face term) + pointwise
  CG flux samples on a fixed collocation grid (as in the MMS canonical runs).
- Training budget: Adam 2000, then L-BFGS up to 3000 with the persisted loss
  history and termination metadata (logger already in
  `fracture_hardcurl_common.py`), stopping early on relative stagnation:
  (L_{k-200} − L_k)/L_{k-200} < 1e-6 over the trailing 200 accepted steps,
  active only after the first 200 accepted steps. Rationale (do not shrink the budget): the
  MMS continuation experiment showed L-BFGS 500 terminates well before
  stationarity and understates accuracy ~2×; this is a single fixed-mesh run, so
  there is no frozen-protocol constraint. Record wall time, final loss, final
  gradient norm, termination reason.

## Stage 3 — NLR track

- Batched Deng NLR on the same solve, with the fracture node-split machinery for
  dual faces crossing Gamma.
- Audit acceptance (established gates): interior and fracture-adjacent dual CVs
  ≤ 1e-12; fracture-cut ≤ 1e-11; boundary dual CVs UNGATED but reported (their
  violation is a finding, not a failure). PINN: ≤ 1e-12 in every class.
- The old KM NLR notebook used Dirichlet fracture tips: its numbers are NOT
  anchors for this sealed-tip notebook; port machinery only, cross-check nothing.

## Stage 4 — Diagnostics table (the paper's Block-3 numbers)

One JSON (`km_case1_conservation.json`) with, for CG / NLR / PINN:

- R_xi per dual-CV class (interior, cut, adjacent, boundary): mean and max;
- R_zeta on rotated volumes (reuse the source notebook's 45° rotated grid):
  mean/max over ALL zeta and separately over FRACTURE-CUT zeta (this is the new
  cell that exists nowhere else — rotated volumes crossing Gamma, lambda_h term
  in the RHS);
- R_tau (native) summary for the PINN.

Expected pattern to verify: PINN ≤ 1e-12 everywhere including boundary and cut
zeta; NLR machine-precision on interior+cut dual CVs, non-conservative on
boundary dual CVs and unguaranteed on zeta; CG non-conservative except the
mesh-aligned Q1 cancellation.

## Stage 5 — MRST comparison (regenerate)

Regenerate the Delta-p and face-integrated flux-difference panels of the source
notebook with the NEW PINN flux. Note the resolutions now differ (CG at 64×64,
MRST at 128×128): frame MRST as a reference at twice the matrix resolution; for
the flux panel, compare on the dual faces using the exact MRST face-flux
aggregation of Stage 6. This remains a consistency check, not an error
measurement.

## Stage 6 — Transport (four tracks, no water balance)

- Keep the source notebook's driver and settings: sealed tips, S=1 on the FULL
  right boundary (all 128 right-boundary faces — the corner-inlet variable in
  the source was a leftover experiment; use the full edge), explicit first-order
  upwind, each method's own matrix–fracture exchange coupling
  (q_{m→Gamma} = −lambda_h for the FEM-based fluxes; NNC exchange for MRST).
- Transport CV family: ONE grid for all four tracks — the 65×65 vertex-centered
  dual grid. CG, NLR, PINN provide dual-face fluxes natively. MRST's 128×128
  face fluxes AND its NNC fracture-exchange terms are aggregated onto the dual
  faces / dual CVs by EXACT SUMMATION over the Stage-1 decomposition — no
  interpolation; conservation is preserved exactly by additivity of the cell
  balances. (This also puts the CG track where its interior non-conservation is
  active — intended, it is the point of the comparison.) State the CV family
  and the aggregation in a markdown cell.
- Time stepping: ONE shared dt for all four tracks, from the CFL condition on
  the dual grid using the MAX face speed over all four flux fields, with the
  usual safety factor. Plus one dt-halving sanity cell: rerun one track (PINN)
  with dt/2 to the first snapshot and confirm the saturation change is
  negligible against the between-track differences (licenses the
  "time-discretization error is subdominant" statement).
- Aggregation validation cell (unconditional — the v2 export includes MRST's own
  transport): run the aggregated-MRST track to the exported snapshot times
  (`snap_T_abs`) and compare against `sw_matrix_snaps`; agreement up to the
  grid-family difference validates the face/NNC aggregation end-to-end. A sign
  or indexing error in the aggregation shows up here first.
- Final time: STOP BEFORE outflow breakthrough at x=0. Deterministic rule:
  T = the largest time on the shared snapshot grid at which ALL FOUR tracks
  satisfy max over left-boundary transport cells of S ≤ 1e-3. Record T, the
  criterion, and which track was binding. (For orientation: MRST breakthrough
  data suggest the usable window is well below 1 PVI; T_abs = PVI × 0.28877.)
- DELETE the water-balance cells (global residual epsilon_w, clipped/unclipped
  comparison, drainage metrics table). Conservation evidence = maps and
  profiles only.
- Deliverables:
  - 4-panel saturation map at final T (CG / NLR / PINN / MRST), shared colorbar;
  - Delta-S panel(s): each FEM-based track minus MRST at final T, shared
    symmetric colorbar (this is where CG's diffuse deficit and NLR's
    inlet/boundary artifact should be visible; the PINN panel should show only
    the exchange-discretization difference);
  - S_Gamma(s) fracture-saturation profiles for the four tracks;
  - the exchange figure E(s): lambda_h vs MRST NNC exchange (port as-is).

## Outputs

Directory `result_km_case1_hardcurl_nlr/`: checkpoint, `km_case1_conservation.json`,
transport arrays, figures as PNG (dpi 600 line plots / 300 field maps):
`fig_km_cg_solution.png`, `fig_km_mrst_validation.png`,
`fig_km_saturation_4panel.png`, `fig_km_saturation_diff.png`,
`fig_km_sgamma_profiles.png`, `fig_km_exchange.png`. Figure titles/labels must
not contain the phrase "hard-curl" (internal name only).

## Guardrails

- A failed gate stops the notebook; do not train on a failed construction.
- No edits to the source notebook, the MMS notebooks, or the MRST export.
- No penalty terms of any kind in the PINN loss; if the Option-A field fails
  Gate A1 on this geometry, STOP and report rather than compensating with a
  penalty.
- Report NLR boundary-CV violations and any transport artifact factually; they
  are findings for the paper, not bugs to suppress.
