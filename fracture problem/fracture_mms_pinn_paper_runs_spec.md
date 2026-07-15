# Instructions: single-fracture MMS — PINN runs, data, and figures for the paper

## Purpose

Produce every number and figure needed for the PINN part of the paper's
single-fracture benchmark section (four "beats": conservation table row; jump
verification figure; conforming convergence-with-capacity figure; nonconforming
half-order question). The NLR/CG side of the section is already written — this task
adds the PINN data on the SAME meshes, discretizations, and error metrics, so
everything is directly comparable.

Do not modify the validated notebooks
(`LCG_fracture_MMS_hardcurl_PINN_conforming_tri_AB.ipynb`, `..._nonconforming_tri_AB
.ipynb`, `..._nonconforming_rect_AB.ipynb`). Create a new sweep notebook (or script)
that imports/reuses their machinery: the CG-LMDFM solves and multiplier settings
come from the existing NLR convergence notebooks (which produced the paper's rate
tables); the PINN machinery (q_p_f, q_p_lambda, gates, training, audits) from the AB
notebooks / shared module.

## Fixed conventions for ALL runs

- Error metric identical to the NLR tables: reconstructed-flux L2(Omega) error vs
  the exact manufactured flux (with K = I this equals the reconstructed-pressure H1
  seminorm error, as the section states). Evaluate the PINN flux (q_p + curl psi) at
  the same element quadrature used for the CG/NLR errors. Report ALSO the dual-face
  integrated flux errors (vs exact and vs CG targets) as secondary columns.
- Multiplier settings per family, exactly as in the paper text: conforming
  h_lambda = 2h (h_Gamma = h); nonconforming h_Gamma = 0.5h, h_lambda = 3h.
- Training protocol FROZEN across every run of a sweep (this is what makes the
  convergence curves clean): fixed seed, Adam 2000 iterations + L-BFGS 500, float64,
  same loss normalization. Capacity policy (adaptive):
  - DEFAULT: SMALL only — width 32, depth 3, Fourier frequencies [1, 2, 4, 8]
    (no log-kappa feature; K = I). The exact solution is smooth; SMALL is expected
    to suffice.
  - CONTINGENT: LARGE — width 96, depth 3, frequencies [1, 2, 4, 8, 16] — is run
    ONLY for a family where a capacity plateau is detected in the SMALL sweep:
    trigger = fitted slope over the last two refinements < half the expected cap
    order for that family. Then rerun that family's 5 meshes at LARGE so the figure
    shows two plateau levels (the floor as an architecture knob). Expected: k=1
    families never trigger; k=2/P1 at the finest levels is the plausible case.
  If a single run fails to converge, rerun with seed+1 and record that this
  happened; do not tune anything per-h.
- Conservation audits at every run: R_xi per CV class with BOTH RHS variants
  (lambda_h and exact lambda); R_tau in the NATIVE evaluation (stream-function
  endpoint differences + analytic/high-order q_p face integrals — NOT moderate-order
  quadrature of the raw field, to avoid the known ~1e-8 artifact).
- Gates A0/A1 re-run once per mesh family before its sweep; a gate failure stops
  that family and is reported.

## Run matrix

| # | family | discretization | meshes h | capacities | purpose |
|---|--------|----------------|----------|------------|---------|
| R1 | conforming tri | k=1, P0 multiplier | 2^-3 .. 2^-7 | SMALL (+LARGE if triggered) | Beat 3, cap = 1 |
| R2 | conforming tri | k=2, P1 multiplier | 2^-3 .. 2^-7 | SMALL (+LARGE if triggered) | Beat 3, cap = 2 |
| R3 | nonconf. tri (opposite diagonal) | k=1, P0 | 2^-3 .. 2^-7 | SMALL (+LARGE if triggered) | Beat 4, half-order question |
| R4 (optional) | nonconf. rect | k=1, Q1/P0 | 2^-3 .. 2^-7 | SMALL only | Beat 4 cross-check (text says rect matches tri) |
| R5 | conforming tri | k=1, P0 | h = 1/64 only | SMALL | Beat 1 table row + Beat 2 figures |

Training count (SMALL-only default): R1 = R2 = R3 = 5 each; R4 (optional) = 5.
R5 requires NO new training — h = 1/64 = 2^-6 with k=1/P0 at SMALL capacity is
already one of R1's runs; R5 is additional audits and figures on that trained
model. Total: 15 mandatory trainings (20 with R4), plus 5 per family where the
LARGE trigger fires, plus at most a few seed-retries. Execution order: run R1 SMALL
end-to-end first (including CSV and figure generation) to shake out the pipeline on
~10 minutes before launching the rest; the remaining families may run as parallel
processes (fully independent). NLR and CG errors per (family, discretization,
h): reuse the archived convergence logs (`case1_convergence_logs.txt`) if parseable;
otherwise recompute (NLR is fast). The CG/NLR curves MUST be the same numbers as the
paper's tables — cross-check the finest-level values against
`tab:single_fracture_reconstruction` before plotting.

## Deliverables

### Data (CSV/JSON; one row per run)

`fracture_mms_pinn_sweep.csv` with columns: family, k, multiplier, h, capacity,
flux_L2_vs_exact, flux_L2_vs_CG_targets(face), lambda_h_L2_error, R_xi_max_by_class
(interior/cut/boundary, lambda_h RHS), R_xi_cut_max_exactlambda_RHS,
R_tau_native_max, train_wall_s, n_params, seed. Plus
`fracture_mms_pinn_fixedmesh.json` for R5: the Beat-1 table row values (R_tau native
mean/max, R_xi mean/max, both RHS variants) and the cut-CV exact-lambda residual.

### Figures (PNG; dpi=600 for line plots, dpi=300 for field plots; captions use T/h
notation consistent with the paper; never the words "hard-curl")

- `fig_frac_conv_conforming.png` (Beat 3): two panels (k=1/P0, k=2/P1), log-log
  error vs h; curves: CG, NLR, PINN-SMALL (plus PINN-LARGE only where triggered);
  dashed reference slopes at the cap orders 1 and 2; if LARGE was run, the two PINN
  plateaus must be visually distinguishable.
- `fig_frac_conv_nonconforming.png` (Beat 4): one panel, k=1; curves CG, NLR,
  PINN-SMALL (plus LARGE only if triggered); dashed slope 1/2 reference. If R4 is run, add rect as
  hollow markers of the same colors, no new curve style.
- `fig_frac_jump.png` (Beat 2): two panels (conforming h=1/64, nonconforming
  h=2^-6): exact lambda, lambda_h, raw CG jump (facet midpoints), PINN jump vs
  arclength s in [0, sqrt(2)]. Legend OUTSIDE the axes (the notebook version
  overlaps the y-label — fix). State in the caption data that the PINN jump
  coincides with lambda_h to machine precision.
- `fig_frac_conservation_flat.png` (optional, Beat 1 support): max R_xi (lambda_h
  RHS, worst class) vs h for the PINN across R1+R2+R3 — a flat line at ~1e-15.
  If not plotted, report the single number max-over-all-runs in the JSON.

### Numbers the text will quote (must appear in the JSON explicitly)

- Beat 1: PINN row (R_tau native mean/max, R_xi mean/max) at conforming h=1/64;
  cut-CV residual under exact-lambda RHS at the same mesh; max R_xi over the entire
  sweep (the "flat at machine precision for all h, k, multiplier choices" claim).
- Beat 3: per-curve observed slope over the pre-plateau segment (fit only the h
  range before the floor; report which h were used) and the plateau level per
  capacity. Do NOT compute an "order at finest refinement" for the PINN.
- Beat 4: the nonconforming PINN pre-plateau slope with its fit range — reported
  neutrally, no interpretation (the text will discuss whether it matches 1/2).

## Guardrails

- Everything float64; seeds fixed and recorded; per-run wall time recorded.
- The convergence sweeps must use the SAME exact-flux quadrature order for CG, NLR,
  and PINN errors (state the order once in the notebook).
- The lambda_h stability settings are per-family constants (above); if the
  multiplier fails to converge at some h (known risk in under-resolved settings),
  stop and report rather than silently changing h_lambda.
- No paper text; no edits to the AB notebooks or the NLR convergence notebooks;
  outputs written next to the sweep notebook.
