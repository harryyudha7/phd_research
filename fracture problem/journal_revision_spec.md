# Instructions: journal draft revision — hard-curl conservative PINN + PoU

## Goal

Create `journal_harry_coupled_hardcurl.tex` in `fenicsx/code/fracture problem/`, based on
`journal_harry_coupled.tex` (copy first, then edit the copy; never modify the original).
The revision (1) replaces the penalty-based cPINN methodology with the
conservative-by-construction hard-curl PINN plus its partition-of-unity (PoU) sequential
update, presented at full generality in the fractured setting, and (2) adds a highly
heterogeneous non-fractured study case (SPE10) with the coupled IMPES campaign.

**Do not touch:** title, abstract, highlights, keywords (add one orange TODO comment above
the abstract noting it must be rewritten later — it currently promises the penalty cPINN).
Keep the document class, macros (`\vflux` etc.), bibliography style, and the existing
`\textcolor{orange}{...}` convention for provisional content.

**Compile check required:** the new file must build with pdflatex (or latexmk) before the
task is done. Add any new BibTeX entries to the same .bib the draft uses.

## Target structure

1. Introduction (rewritten positioning; see below)
2. Mathematical Model (kept — fractured setting; ADD the saturation-dependent coefficient)
3. CG Discretization and Local Conservation (kept — fractured setting; ADD arbitrary-CV
   and boundary-CV conservation functionals)
4. Locally Conservative Flux Reconstruction
   - 4.1 Numerical local reconstruction (NLR) with fracture exchange — KEEP as is
   - 4.2 PINN-based reconstruction of CG fluxes — KEEP the existing subsection title;
     REPLACE its content (current lines ~910–1445) with the conservative-by-construction
     formulation
     - 4.2.x Partition-of-unity head for sequential updates (subsubsection)
   - 4.3 Local conservation of the fracture flux — KEEP (currently orange)
   - 4.4 Coupled flow–transport algorithm — EXTEND with flux refresh / IMPES
5. Numerical Studies
   - 5.1 Highly heterogeneous non-fractured case (SPE10) — NEW
   - 5.2 Single-fracture manufactured benchmark — keep NLR results; penalty-PINN results
     removed, hard-curl placeholders (orange)
   - 5.3 Boundary-tip Köppel–Martin benchmark — same treatment as 5.2
   - 5.4 Two intersecting fractures — keep stub, orange
   - 5.5 Fracture network — keep stub, orange
   - 5.6 Three-dimensional extension (optional) — keep stub, orange
6. Conclusions and Future Work — rewritten (see below)

## Section-by-section requirements

### 1. Introduction

Rewrite the flux-reconstruction positioning around a three-way taxonomy:
(i) deterministic local post-processing (Deng–Ginting element-wise reconstruction — keep
existing citations); (ii) penalty-based physics-informed methods (keep the Jagtap cPINN
citation but as related work: conservation only approximate, penalty weights to tune);
(iii) **conservation by construction**: parameterize only exactly-conservative fields via
q = q_p + curl(psi_theta), so local conservation holds to machine precision on every
control volume with no penalty. Cite: Richter-Powell, Lipman & Chen, "Neural Conservation
Laws: A Divergence-Free Perspective" (NeurIPS 2022); Kelliher (2021) on stream functions
for divergence-free fields. For the PoU update cite: Chen, Chi, E & Yang (Random Feature
Method, J. Mach. Learn. 2022); Lee/Trask et al. POUnets (2021); Moseley, Markham &
Nissen-Meyer FBPINNs (Adv. Comput. Math. 2023); Melenk & Babuška (CMAME 1996).

Contributions list (rewrite the existing one):
(a) hard-curl conservative flux reconstruction for CG solutions, exact on arbitrary
control volumes including boundary CVs; (b) PoU linear head enabling closed-form
sequential flux updates at deterministic-post-processing cost; (c) a coupled IMPES study
isolating the effects of reconstruction method, refresh cadence, staleness, and
conservation on transport accuracy in highly heterogeneous media; (d) fracture-aware NLR
(existing contribution, kept); (e) the fracture extension of the hard-curl construction
is formulated and identified as ongoing work.

State explicitly (one sentence, not hidden): the neural construction is validated here in
the non-fractured heterogeneous setting; fracture validation is the subject of ongoing
work.

### 2–3. Model and discretization

Keep the fractured presentation. Two additions:
- In the model: the transport-coupled coefficient K·a(S) with a(S) = M·S^2 + (1−S)^2 and
  fractional flow F(S) = M·S^2/a(S) (quadratic Corey, end-point mobility ratio M), used by
  the sequential scheme in 4.4 and study 5.1.
- In Section 3: alongside the existing element (R_tau) and nodal-dual (R_xi) conservation
  functionals, define the conservation residual over an ARBITRARY control volume
  (the rotated-diamond zeta family used in 5.1) and note that boundary dual CVs are
  reported separately. These functionals are used to distinguish "conservative on the
  designed dual mesh" (NLR) from "conservative on any CV" (hard-curl).

### 4.2 PINN-based reconstruction of CG fluxes (the core new methodology)

Replace the entire penalty-PINN subsection content, keeping the subsection title
"PINN-based reconstruction of CG fluxes". Present in the fractured setting, with the
non-fractured construction as the base case.

**Terminology rule (applies everywhere in the tex):** the method is called
"PINN-based reconstruction" or "the neural reconstruction"; the construction is
described in prose as a divergence-free stream-function representation /
"conservative by construction". The label "hard-curl" is internal shorthand and must
NOT appear anywhere in the paper. Where the old text used penalty-PINN vocabulary
(loss weights, divergence penalty, adaptive activations), it is removed, and one
sentence may note that, unlike penalty-based physics-informed approaches, conservation
here is enforced through the representation itself rather than through the loss.

Required content, in order:

1. **Construction (non-fractured base case).** q_theta = q_p + curl(psi_theta), with
   curl psi = (dpsi/dy, -dpsi/dx), q_p a fixed particular field with div q_p = f (state
   its construction: assembled analytically from the well/source terms once). Then
   div q_theta = f identically, and for ANY control volume omega,
   \oint_{\partial omega} q_theta·n ds − \int_omega f dx = 0 in exact arithmetic —
   state this as a proposition with the two-line proof (divergence theorem + curl
   identity). Emphasize: face fluxes are evaluated as endpoint differences of psi
   (\int_e q·n = [psi] + \int_e q_p·n), so discrete conservation telescopes exactly,
   independent of quadrature.
2. **Architecture and training.** MLP (width 96, depth 4, SiLU), Fourier feature
   embedding (frequencies 1,2,4,8,16,32) motivated by spectral bias in heterogeneous
   media (cite Tancik et al. 2020; Wang, Wang & Perdikaris CMAME 2021), plus a
   log-permeability input feature. Training: fit integrated dual-face fluxes of the CG
   solution by least squares — the loss has NO conservation penalty because none is
   needed (contrast explicitly with the penalty cPINN loss it replaces). Offline cost:
   Adam (5000 iters) + L-BFGS (3000 iters), ~15 minutes for the 64x64 case.
3. **Fracture generalization (mark the whole passage orange — pending validation).**
   KEEP the current draft's domain-decomposition architecture: one network per
   subdomain, with the fracture gamma lying on subdomain boundaries, and interface
   conditions enforced BY PENALTY between adjacent networks — exactly as the existing
   penalty-cPINN subsection does. The only change is what each network represents:
   a per-subdomain stream function psi_k, so the subdomain flux is
   q_theta^k = q_p + curl(psi_k). Consequences to state explicitly:
   (a) the interior divergence penalty of the old cPINN loss DISAPPEARS — conservation
   inside each subdomain is exact by construction; the training loss retains only the
   data term (fit to CG dual fluxes) and the INTERFACE penalties: the normal-flux jump
   across gamma equal to the exchange flux (Lagrange multiplier), and normal-flux
   continuity across artificial (non-fracture) subdomain interfaces;
   (b) property split, stated honestly: control volumes contained in a single
   subdomain are conservative to machine precision; fracture-cut CVs (spanning two
   subdomains) are conservative up to the interface-penalty residual;
   (c) reuse the current draft's interface/tip treatment and notation — this passage
   should read as a modification of the existing subsection, not a new architecture.
   Do NOT add any remark about alternative fracture constructions to the tex — the
   draft presents only the formulation above.

   [INTERNAL NOTE — for this spec only, never written into the tex: the planned next
   design iteration ("Option A") absorbs the known exchange flux lambda into the
   particular field as a line source, q_p_lambda = single-layer potential of lambda
   along gamma, whose classical jump relation gives [[q_p_lambda . n]] = lambda
   exactly; then a single smooth stream-function network suffices, all penalties
   disappear, and fracture-cut CVs are conservative to machine precision. Limitation:
   tangential-flux jumps across gamma are smeared (accuracy-only). This is recorded
   here as roadmap context for the human author; the writing agent must not include
   it in the paper.]
   Keep everything at formulation level (no results claimed); end with one sentence
   that numerical validation is ongoing. Drafted for supervisor review — do not
   overclaim.
4. **4.2.x PoU head (subsubsection).** Definition psi(x) = sum_k w_k(x) phi_r(x)^T
   theta_k with C^1 normalized cosine windows; reduced basis phi_r = P^T h with
   P = [w | top-(r−1) PCA modes of the trained hidden features orthogonal to w] so that
   theta_k = e_1 reproduces the trained network EXACTLY (warm-start identity, measured
   8e-15); the ridge-anchored closed-form update
   (Phi^T Phi + lambda I) theta = Phi^T (F_target − F_qp) + lambda theta_anchor with the
   normal matrix factorized once; conservation is preserved because the blended psi
   remains single-valued. Frame the two-stage design as method, not workaround: the
   96-weight global head acts as a feature-learning bottleneck during offline training
   (linear-probe argument); sequential adaptation then operates in the local PoU space
   (precedent: RFM, POUnets). Implementation details (window formulas, design-matrix
   assembly, ablation) go to Appendix B, not here.

### 4.4 Coupled flow–transport algorithm

Extend the existing subsection: sequential lagged-coefficient (IMPES-type) scheme —
per outer step solve the linear elliptic problem with coefficient K·a(S^n), reconstruct
the flux (NLR / hard-curl+PoU), advance saturation by explicit upwind transport on the
dual mesh with fixed dt and CFL monitoring. Introduce the REFRESH CADENCE N (flux
recomputed every N-th transport step) as an explicit algorithmic parameter — it is the
central variable of study 5.1b. Note q_p is computed once (the source never changes);
only the head coefficients update per refresh.

### 5.1 Numerical studies — SPE10 non-fractured (NEW; the main new section)

Two parts. All numbers below are measured and final — use them as given; figure PNGs
exist at the stated paths (reference via \includegraphics placeholders with the path in
a comment if conversion to PDF is needed).

**(a) Frozen-flux reconstruction quality.** Setup: SPE10 layer permeability on 64x64 Q1
mesh, wells, 65x65 nodal-dual transport mesh. Report:
- Dual-face flux errors (L2/RMSE/max) vs CG and vs the MRST fine-grid oracle:
  CG vs oracle RMSE 2.83e-3; PINN vs CG 2.53e-3; PINN vs oracle 2.69e-3;
  NLR vs CG 1.97e-3; NLR vs oracle 2.05e-3.
- Conservation table (R_tau, R_xi split interior/source/boundary, R_zeta rotated CVs):
  hard-curl PINN at machine precision (~1e-15..1e-8 depending on functional) on ALL
  families including boundary and rotated CVs; CG violates (R_xi mean 1.4e-3, max 0.109,
  worst at wells; R_zeta rmse 3.2e-2); NLR exact on interior/source dual CVs but carries
  ~1.4e-3 residuals on BOUNDARY dual CVs (inherited weak no-flow error) — this
  interior-vs-anywhere distinction is a key result; present it prominently.
- Saturation transport with frozen fluxes, validated against two independent 512^2
  references (MRST and CVFEM, mutual L2 ~1e-3): L2 saturation errors on the dual mesh:
  CG 0.0452, CVFEM 0.0411, MRST 0.0417, NLR 0.0411, PINN 0.0416 — the conservative
  methods cluster; non-conservative CG is the outlier despite the most accurate flux.

**(b) Coupled IMPES campaign (M=1, T=0.1, dt=1e-5, 10k transport steps).** Tracks:
CG@1, frozen-PINN, NLR@1, NLR@1000, full-retrain PINN@1000, PoU-PINN@1
(global-linear-head track appears only as one motivating table row). Key numbers
(Delta S = PV-weighted saturation RMSE vs the NLR@1 reference at t=0.1):
- frozen (never refreshed, conservative): 0.034 — WORSE than non-conservative CG 0.026;
- NLR@1000 (10 refreshes): 0.0068 — cadence removes ~80% of staleness;
- full-PINN@1000 vs NLR@1000 (matched cadence): 0.0035 — methods interchangeable;
- global linear head: 0.038 (worse than frozen) — the one-row justification of locality;
- **PoU-PINN@1: 0.0020 at both t=0.05 and t=0.10 — the closest track to the reference**,
  face-RMSE flat 1.90–1.95e-3 over all 10k sequential closed-form updates,
  conservation 2.2e-15 interior AND 2.7e-15 boundary at every step.
- Drift-test table (fit to drifted CG targets at t=0.05/0.10 from a saturation snapshot):
  frozen 3.15e-3/7.63e-3; global linear 3.04e-3/5.25e-3; PoU(16x16,r=16) 1.97e-3/1.95e-3;
  full L-BFGS (300-call cap) 2.44e-3/2.45e-3; include the cap caveat in the caption.
- Cost: per-update flux stage PoU 28 ms vs NLR 55 ms; full retrain ~25–100 s. Cumulative
  cost figure: PINN line starts at 900 s (15-min offline training) + 28 ms/step; NLR from
  origin at 55 ms/step; crossover ~33,000 steps (~one simulation to T≈0.33); note
  amortization to zero across parameter studies reusing the checkpoint, and that the
  conservation advantage holds from step 1. Caveat sentence: NLR's 55 ms is the current
  implementation, not a floor.
- All Delta-S difference patterns are front-localized (report as shock displacement).
Figures available: `impes_runs/exp5_comparison/face_rmse_and_saturation_response.png`,
`diff_maps_vs_NLR_t0.05.png`, `diff_maps_vs_NLR_t0.10.png`; data CSVs in the same folder;
PoU run in `impes_runs/exp5_pou_M1/`.

**Interpretation paragraph (required, use this framing):** transport accuracy is
controlled by refresh cadence, not reconstruction method; staleness can be bought back
by refreshing while CG's conservation error is structural; the PoU update makes per-step
refreshing affordable, achieving parity with NLR in accuracy and cost with strictly
stronger conservation guarantees (all CVs incl. boundary).

### 5.2–5.6 Fracture studies

- 5.2 and 5.3: keep all NLR content and results unchanged. KEEP the existing PINN
  figures, tables, and result numbers IN PLACE as placeholders — do not delete or
  comment them out. Instead, add an orange note at the start of each PINN-results
  passage: "\textcolor{orange}{[Placeholder: results below were obtained with the
  penalty-based reconstruction and will be replaced by the hard-curl construction of
  Section 4.2.]}" Rewrite only the surrounding METHOD references in these subsections
  (loss names, penalty-weight discussion) so the text refers to Section 4.2's
  formulation, flagging orange wherever text and placeholder numbers temporarily
  disagree.
- 5.4, 5.5, 5.6: keep as stubs, orange, one short paragraph each stating the planned
  scope.

### 6. Conclusions and future work

Rewrite: summarize the three-way comparison outcome (parity in transport accuracy at
matched cadence; cost crossover; conservation-by-construction as the differentiator);
future work list: (1) fracture extension of the hard-curl construction via the
stream-function jump (the immediate next step), (2) strong-coupling regime M >> 1,
(3) compressible flow via space-time divergence-free construction (cite Richter-Powell),
(4) training without CG data via complementary-energy minimization.

### Appendices

- Appendix A: exact-solution (MMS) validation of the reconstruction machinery —
  kappa = 1, p = sin(3 pi x) sin(3 pi y): CG rate 2.0; exact-flux residuals ~1e-17 on all
  CV families; PINN drops the conservation residual to machine precision while CG shows
  3e-5 (exact-integral RHS); source notebook
  `LCG_DengGinting_MMS_sin_kpi_Q1_hardcurl_PINN.ipynb`. Keep to ~1 page.
- Appendix B: PoU implementation (window function formula, design-matrix assembly,
  factorization) + the window-grid x r ablation table from
  `LCG_DengGinting_example4_spe10_Q1_PoU_head.ipynb` (4x4/8x8/16x16 x r=8/16/32/96;
  chosen config 16x16, r=16, ridge 1e-8; locality beats per-window richness at equal
  DOFs).

## New BibTeX entries to add

Richter-Powell/Lipman/Chen NeurIPS 2022 (Neural Conservation Laws); Kelliher 2021
(Quart. Appl. Math., stream functions); Tancik et al. NeurIPS 2020 (Fourier features);
Wang, Wang & Perdikaris CMAME 2021 (multiscale Fourier PINNs); Chen, Chi, E & Yang 2022
(Random Feature Method, J. Machine Learning); Lee, Trask et al. 2021 (POUnets);
Moseley, Markham & Nissen-Meyer 2023 (FBPINNs, Adv. Comput. Math.); Melenk & Babuška
1996 (CMAME, PUFEM). Verify each entry's fields before adding; do not invent DOIs.

## Conventions and guardrails

- `\textcolor{orange}{...}` marks everything provisional: the fracture hard-curl
  formulation (4.2 item 3), all pending fracture-PINN results, stubs, and the
  abstract-rewrite TODO.
- Keep existing notation (`\vflux`, mesh symbols); introduce psi, q_p, theta, and PoU
  symbols consistently and add them to any notation table if one exists.
- Numbers in Section 5.1 are final measured values — transcribe exactly as given above;
  do not recompute, round to 2–3 significant figures in prose, full values in tables.
- Do not modify `journal_harry_coupled.tex`, the notebooks, the simulator, or any run
  outputs.
- Deliverable: `journal_harry_coupled_hardcurl.tex` compiling cleanly, plus a short
  `journal_revision_notes.md` listing every section changed, every orange block added,
  and any place where the agent had to make a judgment call the user should review.
