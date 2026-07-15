# Instructions: fracture MMS notebook — stream-function (hard-curl) PINN, Option A then Option B

## Goal

Create THREE notebooks in `fenicsx/code/fracture problem/`, one per mesh variant of
the same MMS single-fracture problem (fracture along the main diagonal), implemented
IN THIS ORDER:

1. `LCG_fracture_MMS_hardcurl_PINN_conforming_tri_AB.ipynb` — regular triangular
   mesh, squares split by the diagonal from (0,0) to (1,1): the fracture COINCIDES
   with element edges (conforming). Simplest lambda integration; do this first.
2. `LCG_fracture_MMS_hardcurl_PINN_nonconforming_tri_AB.ipynb` — triangular mesh with
   the OPPOSITE diagonal, (0,1) to (1,0): the fracture cuts through elements.
3. `LCG_fracture_MMS_hardcurl_PINN_nonconforming_rect_AB.ipynb` — rectangular mesh:
   the fracture crosses cells diagonally.

Mesh/CG/dual machinery is ported from the corresponding existing NLR notebooks:
`LCG_Deng_flux_reconstruction_MMS_conforming_P0.ipynb`,
`LCG_Deng_flux_reconstruction_MMS_nonconforming_opposite_diagonal_P0.ipynb`,
`LCG_Deng_flux_reconstruction_MMS_rect_nonconforming_P0.ipynb`.

STRUCTURAL NOTE the notebooks must exploit and state: the reconstruction itself —
the network, q_p_f, q_p_lambda, and both gates A0/A1 — is MESH-INDEPENDENT (it never
sees elements, only points and dual-face segments). Only the CG solve, the dual mesh,
the target fluxes, and the audits change between the three notebooks. Build the
Option A/B machinery once (identical cells, or a small shared .py the three notebooks
import) so the three variants differ ONLY in their mesh/target blocks; the trio then
demonstrates mesh-independence of the construction, which is a result in itself.

KNOWN ISSUE to carry over (conforming case): the conforming MMS multiplier is
unstable when lambda_h uses a fine P0 space — reuse the established fix (coarse P0
multiplier on an independent mesh with h_lambda = 2h, hand-assembled coupling) from
the conforming NLR notebook. Build q_p_lambda from THAT lambda_h.

Each notebook replaces the soft-penalty PINN of the source notebooks by the
conservative-by-construction stream-function reconstruction, in TWO variants
implemented sequentially in the same notebook:

- **Option A (implement FIRST, to completion):** the exchange flux lambda is absorbed
  into the particular field as a line source (single-layer potential along the
  fracture); ONE smooth network; no interface penalties.
- **Option B (implement AFTER A is finished):** per-subdomain stream functions with
  the jump enforced by penalty (the construction described in the paper draft), using
  SMALLER networks than Option A (each subdomain field is simpler).

Scope: the MMS single-fracture problem only (from the split-diagnostic notebook). The
Koeppel–Martin nonconforming case is a separate later task. Do not modify any source
notebook.

**Source material (port, do not re-derive):**
- Problem definition, exact solution, exact lambda, CG-LMDFM solve, fracture geometry,
  and the EXISTING verification cells (lambda_h vs lambda plots, CG jump traces,
  fracture-adjacent conservation audits):
  `LCG_ghost_partition_PINN_flux_reconstruction_MMS_split_diagnostic.ipynb`.
- Dual-mesh construction, fracture-cut CV identification, and the node-split
  integration of lambda on faces crossing the fracture: the nonconforming NLR
  notebooks (`LCG_nonconforming_PINN_flux_reconstruction_MMS_2d_no_fenicsx_ii.ipynb`
  and the KM NLR notebook where the node-split machinery achieved machine-eps on
  fracture-cut CVs).
- Stream-function pipeline (features, endpoint-difference face fluxes, Adam+L-BFGS
  loop, float64 conventions):
  `LCG_DengGinting_example4_spe10_Q1_no_fracture_hardcurl_PINN.ipynb` /
  `impes_spe10_simulator.py`.

## Conventions (one markdown cell at the top; enforced everywhere)

- ONE fixed unit normal n_gamma per fracture segment; the SAME orientation is used in:
  the jump definition [[q·n]] = q(+side)·n − q(−side)·n = lambda, the single-layer
  formula, the CG one-sided traces, and the conservation RHS. Document it once;
  never flip locally.
- float64 throughout; fixed seeds; every gate prints PASS/FAIL against a stated
  tolerance; a failed gate STOPS the notebook (raise), do not proceed to training on
  a failed construction.
- Fracture-aware conservation statement used by all audits:
  for any control volume omega, ∮_∂omega q·n ds − ∫_omega f dx − ∫_{gamma∩omega}
  lambda ds = 0. Two RHS variants are computed: with the EXACT lambda and with the
  discrete lambda_h (audits report both).

## Common infrastructure (built once, used by both options)

1. CG-LMDFM solve of the MMS problem → p_h, lambda_h, and the dual-face integrated CG
   target fluxes. Faces crossing the fracture use one-sided integration split at the
   crossing point (node-split machinery).
2. Dual-CV classification: {interior-away-from-fracture, fracture-CUT (gamma passes
   through the CV), fracture-ADJACENT (CV touches gamma but is not cut), boundary,
   source}. All conservation audits report statistics per class.
3. The exact-solution flux for error measurement (MMS).

## OPTION A — stages and gates

**A0 — particular line-source field q_p_lambda.** Single-layer potential
q_p_lambda(x) = (1/2pi) ∫_gamma lambda(s) (x − y(s))/|x − y(s)|^2 ds.
Implement with closed-form segment integrals for piecewise-constant (P0) density —
each multiplier element is a constant-density segment source with a textbook
closed-form field — and a subdivided high-order Gauss fallback for general density.
Two densities supported: exact lambda and lambda_h.
**Gate A0 (jump test, no mesh, no network):** for a single straight segment with
constant density, evaluate q_p_lambda·n at mirrored point pairs approaching the
segment: the difference must converge to lambda (relative error ≤ 1e-8 at distance
1e-6), and the TANGENTIAL component must be continuous across it. This gate kills the
sign/orientation bugs first.

**A1 — conservation identity WITHOUT training.** Set q = q_p_f + q_p_lambda + curl(psi)
with (i) psi ≡ 0 and (ii) a randomly initialized network. Compute R_xi on ALL dual
CVs (all classes, including fracture-cut) against the fracture-aware RHS with the
MATCHING density (q_p_lambda built from exact lambda → RHS with exact lambda; built
from lambda_h → RHS with lambda_h).
**Gate A1: max|R_xi| ≤ 1e-13 in every class.** Face integrals of q_p_lambda across
faces that CROSS gamma must be split at the crossing point (its normal trace jumps
there). Failure means the splitting or the quadrature near gamma is wrong — fix
before proceeding.

**A2 — training.** Single network, SMALL: width 32–64, depth 3, SiLU, Fourier
frequencies [1, 2, 4, 8]; drop the log-kappa feature if kappa is homogeneous in this
MMS. q_p_lambda built from lambda_h (the data the method would have in practice).
Fit the dual-face CG target fluxes by least squares; Adam (~2000 iters) + short
L-BFGS. Face fluxes as psi endpoint differences + (analytic/high-order) q_p integrals.

**A3 — verification battery (mirror the soft-penalty notebook's diagnostics):**
- Lambda comparison along the fracture arclength, one figure: exact lambda(s),
  lambda_h(s), the CG jump [[K grad p_h · n]](s) from one-sided traces, and the PINN
  jump [[q_theta·n]](s). NOTE the expected outcome, to be stated in a markdown cell:
  in Option A the PINN jump equals the jump of q_p_lambda, i.e. lambda_h EXACTLY (the
  network part is continuous) — the curve should overlay lambda_h to machine
  precision; verify numerically at fracture quadrature points.
- Flux errors: dual-face L2/RMSE/max vs the exact MMS flux and vs the CG flux.
- Conservation audit: R_xi per CV class (both RHS variants). Expectation to verify:
  machine precision with the lambda_h-consistent RHS in EVERY class including
  fracture-cut; with the exact-lambda RHS the residual on cut CVs equals the
  lambda_h−lambda consistency error (report it as such, it is a property of the CG
  solve, not of the reconstruction).
- Optional visual: pressure/flux field maps as in the source notebook.

## OPTION B — stages (start only after all A gates pass)

**B0 — partition.** Reuse the subdomain split of the ghost-partition notebook: the
fracture lies on the interface between subdomains; where the fracture ends inside the
domain, the interface continues as an ARTIFICIAL extension (no jump there).

**B1 — networks.** One stream function psi_k per subdomain, SMALLER than Option A's
single network (each one-sided field is simpler): width 16–32, depth 2–3, Fourier
[1, 2, 4]. Subdomain flux q_k = q_p_f + curl(psi_k). NO q_p_lambda in Option B — the
jump must be produced by the difference of the neighboring networks.

**B2 — how lambda enters (this answers the distribution question).** lambda is NOT
distributed, split, or apportioned to the subdomains. It is a constraint on the
DIFFERENCE of the two one-sided normal fluxes:
(q_1 − q_2)·n_gamma = lambda_h on gamma.
The individual one-sided values are determined by each subdomain's DATA term — each
side fits its own one-sided CG dual-face fluxes, which already encode how the
physical flux divides between the sides. Explicitly FORBIDDEN: any 50/50 (or other
fixed-ratio) splitting heuristic of lambda between the sides; it has no physical
basis and fights the data term. Loss structure per training step:
- data: one-sided CG dual-face flux misfit, per subdomain (cut faces contribute their
  one-sided segments to the owning subdomain);
- interface-jump penalty: [(q_1 − q_2)·n − lambda_h]^2 at fracture quadrature points;
- interface-continuity penalty: [(q_1 − q_2)·n]^2 on the artificial extension
  (zero-jump condition beyond the tips).
Sweep the penalty weights coarsely (e.g., {1, 10, 100} relative to the data term) and
report the sensitivity — do not tune further; the sensitivity itself is a result.

**B3 — verification battery: IDENTICAL functions as A3** (same lambda-comparison
figure, same flux errors, same per-class R_xi audit). Additional required plot:
fracture-cut-CV max|R_xi| versus penalty weight (the expected penalty floor), and the
PINN jump along arclength vs lambda_h (in B this holds only approximately — show the
residual).

## Final comparison (one cell + one markdown conclusion, per notebook)

Side-by-side table, Option A vs Option B: flux error vs exact; jump error vs
lambda_h along gamma; R_xi per CV class (worst class highlighted); number of
trainable parameters; training wall time. The markdown conclusion states plainly
which construction achieves machine-precision conservation on fracture-cut CVs and at
what cost, without editorializing beyond the table.

After all three notebooks are done, a small cross-mesh summary (may live in the
third notebook): one table with rows = {conforming tri, nonconforming tri,
nonconforming rect} and columns = {A flux error, B flux error, A worst-class R_xi,
B worst-class R_xi} — the demonstration that Option A's conservation is unaffected by
mesh alignment with the fracture, which is the mesh-independence claim in one table.

## Pitfalls (in the order they will bite)

1. Sign/orientation of lambda and the jump — killed by Gate A0; do not skip it.
2. Dual faces crossing gamma: split the integral at the crossing; the q_p_lambda
   normal trace is discontinuous exactly there.
3. Quadrature near gamma: closed-form segment antiderivatives where available;
   subdivided high-order Gauss otherwise (same lesson as the well cells in the
   non-fractured case).
4. Option B tips: without the continuity penalty on the artificial extension, the two
   networks are free to disagree beyond the tip and conservation of CVs straddling
   the extension silently degrades.
5. The 1D fracture flux needs NO network: antiderivative of (f_gamma + lambda) per
   branch, constants fixed at the tips — implement as the exact closed-form piece and
   include its conservation check.

## Out of scope

The Koeppel–Martin nonconforming benchmark (separate task after this notebook is
reviewed); any change to the SPE10 notebooks or the simulator; PoU heads (single
training is enough here); any paper text.
