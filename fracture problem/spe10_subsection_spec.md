# Instructions: Section 5.1 "Highly heterogeneous, non-fractured benchmark"

Target: `\subsection{Highly heterogeneous, non-fractured benchmark}`
(`\label{subsec:heterogeneous-nonfractured}`) in `journal_harry_hardcurl.tex`.
This spec contains the agreed structure, ALL measured numbers (transcribe as given;
round to 2–3 significant figures in prose, full values in tables), and the list of
figures. Figures marked AUTHOR are produced by the author — insert
`\includegraphics` placeholders with the stated filename and a caption per the spec.
Never use the term "hard-curl" (internal shorthand). Follow the main
`journal_revision_spec.md` conventions (orange = provisional / to verify).

## Narrative structure (two blocks after a setup block)

Opening sentence states what the subsection demonstrates: (1) at fixed conductivity,
both reconstructions deliver reference-grade transport where the raw CG flux fails;
(2) in the coupled problem, the PoU update makes the PINN as accurate as NLR at lower
per-update cost, with a quantified training-amortization threshold.

### Block A — setup (consolidated; one figure, one paragraph cluster)

- Coupled pressure equation −∇·[(S² + (1−S)²) K(x) ∇p] = f with saturation-dependent
  conductivity; note that at iteration 0 (S ≡ 0) it reduces to −∇·[K∇p] = f, so the
  frozen-flux study of Block B is the coupled problem's first step. One sentence on
  assumptions: incompressible two-phase flow, unit viscosities, quadratic relative
  permeabilities, hence a(S) = S² + (1−S)² and fractional flow F(S) = S²/a(S);
  cross-reference the coupled-algorithm subsection (4.4) for the
  solve-pressure → reconstruct-flux → advance-S loop instead of re-deriving it.
- Domain and data (all verified against the MRST export metadata): unit square;
  permeability K(x) from SPE10 layer 20 (Tarbert), normalized and resampled
  60×220 → 64×64 (nearest), cell-wise constant on the 64×64 Q1 mesh, contrast
  ~1.0e5 (values 4.02e-3 to 409.2) (Figure F1, AUTHOR); one injector with integrated
  cell rate q = +1 at (0.20, 0.40) and one producer with q = −5 at (0.80, 0.51);
  homogeneous Dirichlet boundary condition p = 0 on ALL boundaries (NOT no-flow);
  transport: injector cell at S = 1, producer removes water at the local fractional
  flow, inflow across the boundary carries S = 0; transport on the DUAL MESH — one
  sentence explaining its construction: the dual control volumes are centered at the
  nodes of the primal mesh and are obtained by connecting the centers of the
  adjacent primal cells (cross-reference the discretization section if it defines
  this already; do not write "65×65", which confuses readers against the 64×64
  primal grid — say "dual mesh" throughout); explicit upwind, dt = 1e-5, CFL ≈ 0.22;
  saturation snapshots reported at times T = 0.02, 0.05, 0.10 (do NOT use the term
  "PVI" anywhere).
- References, introduced HERE (used by both blocks): (i) MRST (TPFA) on the same 64×64
  grid for cross-code comparison; (ii) a fine-mesh reference obtained with MRST on a
  512×512 grid, cross-validated against an independent fine-mesh CVFEM solution —
  mutual L2 differences 1.03e-3, 9.54e-4, 1.23e-3 at the three snapshot times, i.e.
  ~40× below the method-level differences reported below, so the reference can rank
  the coarse-grid methods. Fine-grid fluxes aggregated to the dual faces serve as the
  "oracle" flux in Table T1.

### Block B — frozen-flux validation (iteration 0)

1. CG solve and its flux: one or two sentences that the raw CG flux is highly
   irregular across the permeability contrasts and not locally conservative
   (forward-reference Table T2). Reconstruction by NLR (cite Deng — NLR on highly
   heterogeneous fields is established there; our contribution here is the neural
   track and the conservation scope) and by the PINN of Section 4.2.
2. PINN training setup, as a small table (T4) plus 2–3 sentences: why Adam first
   (robust on the initial rough landscape) then L-BFGS (fast local convergence to the
   plateau); total offline cost ~900 s. Reference Figure F2 (AUTHOR: training curve +
   flux-error visualization).
3. Flux accuracy — Table T1 (dual-face integrated flux errors; RMSE / L2 / max):

   | pair                      | L2      | RMSE     | max     |
   |---------------------------|---------|----------|---------|
   | CG vs oracle              | 0.2622  | 2.831e-3 | 6.32e-2 |
   | PINN vs CG                | 0.2340  | 2.526e-3 | 4.43e-2 |
   | PINN vs oracle            | 0.2490  | 2.689e-3 | 4.05e-2 |
   | NLR vs CG                 | 0.1828  | 1.973e-3 | 5.16e-2 |
   | NLR vs oracle             | 0.1900  | 2.051e-3 | 3.73e-2 |

   Interpretation sentence: all three coarse-grid fluxes differ from the oracle at the
   same 2–3e-3 level; accuracy is NOT what separates them.
4. Conservation — Table T2 (residual statistics; mean|R| and max|R| per family), the
   structural heart of the block:

   | functional (mean / max) | CG                    | NLR                     | PINN                  |
   |-------------------------|-----------------------|-------------------------|-----------------------|
   | R_tau (elements)        | 7.17e-3 / 3.70        | 7.17e-3 / 3.70 (unchanged; NLR corrects dual CVs, not elements) | 2.2e-11 / 1.8e-8 |
   | R_xi interior           | 1.49e-3 / 0.109       | 4.2e-16 / 4.0e-14        | 1.9e-15 / ~1e-14      |
   | R_xi source (well) CVs  | 3.43e-2 / 7.79e-2     | 1.3e-15 / 2.7e-15        | ~2e-15                |
   | R_xi boundary CVs       | 2.89e-4 / 1.30e-2     | 1.4e-3 / 1.4e-2          | 2.1e-15 / ~1e-14      |

   No R_zeta row: conservation on arbitrary control volumes is covered theoretically
   by the Proposition of Section 4.2 and is not measured separately here; one prose
   clause may state this.

   Required prose points: (i) CG's violations concentrate at the wells — the direct
   cause of its transport failure below; (ii) NLR is machine-exact exactly on the CV
   family it is built for (interior/source dual CVs) but carries a residual on
   BOUNDARY dual CVs (~1.4e-3 mean, inherited from the boundary treatment of the CG
   flux it corrects — note it is LARGER than CG's own boundary residual), and its
   guarantee is tied to that one CV family; (iii) the PINN is exact on every dual CV
   including the boundary, and on ANY control volume by the Proposition of
   Section 4.2 — conservation is a property of the representation, not of a target
   mesh; (iv) the PINN's R_tau max ~1.8e-8 occurs only in the well cells and stems
   from the numerical quadrature of the particular-field flux integral (the integrand
   is nearly singular at the wells), not from the construction. Reference Figure F3
   (AUTHOR, optional but recommended: log10|R_xi| maps for CG/NLR/PINN).
5. Frozen-flux transport — Figure F4 (AUTHOR: saturation maps at T = 0.10, the
   snapshot at which the front has reached the producer) and Table T3: saturation L2
   on the dual mesh vs the MRST 512×512 reference (restricted to the dual mesh), at
   all three snapshots plus the mean (verified from the notebook; full precision
   below, round in the tex):

   | method          | T = 0.02  | T = 0.05  | T = 0.10  | mean     |
   |-----------------|-----------|-----------|-----------|----------|
   | CG (raw flux)   | 2.830e-2  | 4.916e-2  | 5.810e-2  | 4.518e-2 |
   | NLR             | 2.609e-2  | 4.548e-2  | 5.172e-2  | 4.110e-2 |
   | PINN            | 2.628e-2  | 4.547e-2  | 5.298e-2  | 4.157e-2 |
   | MRST (64×64)    | 2.663e-2  | 4.671e-2  | 5.172e-2  | 4.168e-2 |
   | CVFEM (64×64)   | 2.609e-2  | 4.549e-2  | 5.178e-2  | 4.112e-2 |

   (Errors vs the independent CVFEM 512 reference agree to the third digit — one
   sentence may say so; no second table.)

   Required concluding sentence (the conservation thesis): the conservative methods
   cluster at the coarse-grid discretization level at every snapshot; raw CG is the
   consistent outlier despite its flux being pointwise as accurate as any — exact
   local conservation, not pointwise flux accuracy, governs transport quality.

### Block C — coupled pressure–saturation study

1. Transition: NLR's accuracy is now established against external references, so the
   coupled study compares only NLR and the PINN variants; external references and CG
   are dropped except as the no-reconstruction baseline. State the loop parameters:
   pressure re-solved and flux reconstructed at EVERY transport step; dt = 1e-5,
   T = 0.1, 10,000 steps; NLR updated every step is the reference track.
2. PINN update variants and their per-update costs. GRANULARITY RULE: all times are
   the FULL per-step flux stage (target-flux extraction + assembly/solve), the same
   footing as NLR's stage — solve-only numbers may appear once in a parenthetical,
   never in the tables. Final values: full-weight re-optimization (nonlinear, tens of
   seconds — quoted, not run in the comparison); global linear last layer, 0.0066 s
   (solve alone 0.0008 s); PoU layer, 0.028 s mean (solve alone 0.016 s); NLR
   0.055 s. Per-update speedup of PoU over NLR: about 2x.
   PoU configuration for reproducibility: 16×16 window grid (N_w = 256), r = 16
   features per window, 50% overlap, ridge 1e-8 anchored to the previous step's
   coefficients, design matrix and normal-equations factorization computed once
   (0.72 s) before the loop.
3. Results — Figure F5 (AUTHOR: saturation-difference maps vs the NLR track at two
   snapshots) with the PV-weighted saturation L2 differences vs NLR:

   | track                        | t = 0.05 | t = 0.10 |
   |------------------------------|----------|----------|
   | (a) raw CG flux (no reconstruction) | 0.01667 | 0.02589 |
   | (b) frozen PINN flux (never updated) | 0.03191 | 0.0343  |
   | (c) global linear last layer  | 0.02659 | 0.0385  |
   | (d) PoU layer                 | 0.00207 | 0.00205 |

   Required prose: (b) and (c) drift — a conservative but stale or globally-updated
   flux is WORSE than the non-conservative CG track, so updating accurately matters;
   (d) is nearly indistinguishable from NLR, an order of magnitude below every other
   effect; all difference patterns are front-localized (shock displacement).
   Conservation during the coupled run: the PINN tracks remain at machine precision on
   all dual CVs including the boundary at every step (R_xi ~2e-15), while NLR retains
   its ~1.4e-3 boundary residual.
4. Cost and the crossover — Figure F6 (AUTHOR) and the arithmetic (full-stage
   granularity): one-time training 900 s; totals at 20,000 iterations: NLR 1,100 s vs
   PINN 900 + 560 = 1,460 s; crossover at ~=33,000 iterations; at 50,000 iterations:
   PINN 900 + 1,400 = 2,300 s vs NLR 2,750 s. Required caveats:
   (i) both implementations are single-threaded; NLR's assembly is embarrassingly
   parallel, so 0.055 s is not a lower bound — present the timing table and the
   crossover as "for the present implementations", and rest the conclusion on cost
   PARITY plus the strictly stronger conservation guarantees, not on a speed win;
   (ii) the training cost is paid once per permeability field and amortizes to zero
   when the trained network is reused across runs, as done throughout this section.
5. Closing paragraph of the subsection: at matched per-step updating the two
   reconstructions are interchangeable in accuracy; their per-update costs are of the
   same order (about half for the PoU update in the present single-threaded
   implementations); the PINN's conservation guarantee is strictly stronger (all
   control volumes, including boundary and arbitrary families); the training
   threshold quantifies when the neural route becomes cheaper end-to-end for these
   implementations.

## Data sources for verification (filenames only; the author copies these files next to this spec — agent: cross-check numbers where the files are present, orange-flag mismatches, and trust the spec tables otherwise)

- Flux/conservation/training numbers: outputs of
  `LCG_DengGinting_example4_spe10_Q1_no_fracture_hardcurl_PINN.ipynb` and
  `LCG_DengGinting_example4_spe10_Q1_PoU_head.ipynb` (Gate-1/Gate-2 cells).
- Coupled-run numbers: `step_log.csv` (PoU run),
  `saturation_differences.csv`, timing table in `comparison_summary.md`.
- Reference-agreement numbers (1.03e-3 / 9.54e-4 / 1.23e-3 at T = 0.02/0.05/0.10) and
  the frozen saturation L2 table: VERIFIED in the frozen-flux notebook outputs
  ("Fine-reference saturation comparison" and "saturation errors on the native 65×65
  dual mesh" cells). All numbers in this spec are final; no orange flags remain in
  the tables.

## Figures — TO BE GENERATED BY THE AUTHOR (agent inserts placeholders)

Format: PNG throughout (the author's standard workflow), saved with
`savefig(..., dpi=300)` — use dpi=600 for the line plots (F2a, F6) so text and
thin lines stay crisp. Captions must use "dual mesh" (never "65×65") and times
T = 0.02/0.05/0.10 (never "PVI").

- **F1 `fig_spe10_perm.png`** — log10 permeability on the 64×64 grid, wells marked.
  Source: the frozen-flux notebook's data loading cell.
- **F2 `fig_pinn_training.png`** — two panels: (a) training loss vs iteration with the
  Adam→L-BFGS switch marked (vertical line at iteration 5,000); (b) per-face absolute
  flux difference of the trained network vs CG and vs the MRST oracle (map on dual
  faces or histogram — author's choice). Source: frozen-flux notebook training cells.
- **F3 `fig_conservation_maps.png`** (optional, recommended) — log10|R_xi| on the dual
  grid for CG / NLR / PINN, shared colorbar; boundary ring visibly nonzero for NLR.
  Source: frozen notebook conservation cells + simulator validation gate.
- **F4 `fig_frozen_saturation.png`** — saturation maps at T = 0.10:
  CG, NLR, PINN, MRST 64×64, fine reference (5 panels, shared colorbar).
  Source: frozen-flux notebook transport cells.
- **F5 `fig_coupled_dS.png`** — |ΔS| vs the NLR track at t = 0.05 and t = 0.10 for
  tracks (a)–(d), shared colorbar. NOTE: the existing
  `impes_runs/exp5_comparison/diff_maps_vs_NLR_*.png` predate the PoU run — the author
  must add the `exp5_pou_M1` track to `compare_exp5.py`'s RUNS dict and regenerate so
  panel (d) is PoU (replacing the full@1000 panel).
- **F6 `fig_cost_crossover.png`** — cumulative wall time vs iteration count: NLR line
  from the origin (slope 0.055 s), PINN line from 900 s (slope 0.028 s), crossing
  marked at ~=33,000; annotate "1 simulation (10k)" and "50k"; caption notes
  single-threaded implementations. Small standalone matplotlib script from the
  constants above.

## Out of scope for this subsection

The refresh-cadence sweep, NLR@1000/full@1000 tracks, and the drift-test table (they
live in the methodology/appendix per the main spec); any recomputation of numbers;
any use of the words "hard-curl", "exp5", or internal run names in the tex.
