# Instructions: PoU-head notebook for the hard-curl PINN

## Goal

Replace the network's global linear last layer with a **partition-of-unity (PoU) linear
head** with a reduced per-window feature basis, validate accuracy / conservation / cost
against the existing baselines, and save a checkpoint usable by the IMPES simulator.
Create a new notebook `LCG_DengGinting_example4_spe10_Q1_PoU_head.ipynb` in
`fenicsx/code/fracture problem/`. Do not modify the source notebook or the simulator.

**Source material (port, do not re-derive):**

- Mesh / dual-face / q_p / feature machinery and all metric conventions from
  `LCG_DengGinting_example4_spe10_Q1_no_fracture_hardcurl_PINN.ipynb`.
- Trained checkpoint: `case3_ecmor/hardcurl_pinn_spe10_Q1_64x64.pt`.
- Batched Deng NLR and CG flux extraction: import from `impes_spe10_simulator.py`
  rather than re-porting.
- MRST oracle flux data: the same coarse dual oracle the source notebook uses
  (MRST1024 / CVFEM576).

All linear algebra in float64.

## The PoU head — definition

psi(x) = sum_k w_k(x) * phi_r(x)^T theta_k

- **Windows:** K overlapping patches covering [0,1]^2 (default 8x8 = 64, ~50% overlap).
  **Smooth (C1) window functions** (e.g., cosine bumps), normalized so
  sum_k w_k(x) = 1 exactly (divide by the sum). Do NOT use C0 bilinear hats —
  smoothness is required for pointwise curl evaluation (zeta-type checks).
- **Reduced per-window features phi_r(x) (r <= 96), built from the frozen hidden
  features h(x) of the checkpoint — no retraining:**
  - **feature 1 = h(x)^T w**, where w is the checkpoint's trained last-layer weight
    vector — i.e., the trained stream function itself is the first basis function;
  - **features 2..r = the top (r-1) SVD/PCA modes** of the hidden features evaluated
    at all dual-face endpoints, computed once, taken orthogonal to the w-direction.
  - Store the resulting projection matrix P (96 x r); phi_r(x) = P^T h(x).
- **Unknowns:** Theta = (theta_1 ... theta_K), K*r total (e.g., 64x16 = 1024).
- Face fluxes remain endpoint differences and are **linear in Theta**:
  F_f = F_qp,f + sign_f * (psi(b_f) - psi(a_f)).
  Sparse design matrix Phi (8580 x K*r): row f has entries
  [w_k(b_f) phi_r(b_f) - w_k(a_f) phi_r(a_f)] for each window k overlapping either
  endpoint of face f.
- **Fit:** ridge least-squares
  min_Theta ||Phi vec(Theta) - (F_target - F_qp)||^2 + lambda ||vec(Theta) - vec(Theta_bar)||^2,
  with the anchor Theta_bar = the warm start (theta_k = e_1 for all k), NOT zero —
  under-constrained windows must fall back to the trained solution.
  Factorize the normal matrix once (sparse); every subsequent fit is one sparse
  matvec + one back-substitution. lambda swept in {1e-8, 1e-6, 1e-4} relative to the
  largest normal-matrix diagonal.

## Mandatory validation gates (in order; stop and report on failure)

**Gate 0 — warm-start identity.** With theta_k = e_1 for all k (any r), the PoU psi
must reproduce the checkpoint's face fluxes to <= 1e-12. This is exact by construction
(sum_k w_k = 1 and feature 1 = trained psi); failure means window normalization,
projection, or feature wiring is wrong.

**Gate 1 — t=0 fit (no drift).** Fit to the CG dual-face fluxes at S=0 (coeff = K).
Report dual-face integrated flux L2 / RMSE / max in the source notebook's format for:
CG vs MRST-oracle, original-PINN vs CG, original-PINN vs oracle, **PoU-fit vs CG,
PoU-fit vs oracle**, NLR vs CG, NLR vs oracle.
**Overfitting guard:** if PoU-vs-CG improves but PoU-vs-oracle degrades relative to the
original network, the head is fitting CG's noise — increase lambda and report the trend.

**Gate 2 — conservation.** R_xi of the PoU-fit flux on all dual CVs
(interior / source / boundary split) must be <= 1e-13. Architectural (single-valued psi
=> endpoint telescoping); a violation is a dtype or indexing bug, not a modeling issue.

**Gate 3 — drift test (the decisive gate).** Load S from
`impes_runs/exp5_NLR_M1/S_step5000.npy` (t = 0.05). Build coeff = K * a(S) with
a(S) = S^2 + (1-S)^2 (M = 1), solve the CG pressure, extract drifted target fluxes
(reuse simulator functions). Starting from the Gate-1 fitted state, fit with each head
and report face-RMSE vs the drifted targets for:
(a) frozen (no update), (b) global linear-last, (c) **PoU LSQ**, (d) full L-BFGS
(reference ceiling; reuse the simulator's update_full capped at ~300 calls), (e) NLR.
Known anchors from exp5 at this drift level: frozen ~ 3.2e-3, linear ~ 3.1e-3,
full ~ 2.1e-3.
**Success criterion: PoU meaningfully below linear-last, ideally within ~20% of full.**
Repeat with `S_step10000.npy` (t = 0.10, stronger drift). Report R_xi after every
drift fit.

**Gate 4 — timing.** Report: Phi build + factorization (one-time), and per-fit wall
time (repeated work only: matvec + backsub). Targets: per-fit well under NLR's
~55 ms/update; at r ~ 16 expect the ~1–5 ms class.

## Ablation (only after all gates pass at the default config)

Two-axis sweep, one table: window grid {4x4, 8x8, 16x16} x features per window
r in {8, 16, 32, 96}, reporting Gate-3 drift RMSE (both snapshots), DOF count, per-fit
time, and factorization time. Note the diagonal trade explicitly: 16x16 windows x r=8
(2048 DOFs) vs 4x4 x r=96 (1536 DOFs) tests whether locality or per-window richness
matters more. Recommend a final configuration.

## Checkpoint output

Save `case3_ecmor/hardcurl_pinn_spe10_Q1_64x64_pou.pt` containing:

- the frozen hidden-layer state_dict (unchanged from the source checkpoint),
- a `pou` dict: {window grid shape, overlap, window-function type and parameters,
  projection matrix P (96 x r, float64), r, theta (K x r, float64), ridge lambda,
  anchor Theta_bar},
- provenance: source checkpoint path, chosen config from the ablation.

Document the format in a markdown cell — the simulator's FluxModel will need a matching
loader later (explicitly NOT this task). Save the Gate-1/Gate-3 metric tables as JSON
next to the checkpoint.

## Do not

- Retrain or fine-tune the hidden layers (frozen features are the point of the design).
- Change q_p, the dual mesh, metric conventions, or the M=1 relperm model.
- Integrate into the IMPES simulator (separate task, gated on this notebook's results).
- Use C0 (bilinear hat) windows.
- Anchor the ridge to zero.
