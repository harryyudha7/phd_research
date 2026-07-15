# Implementation spec: sequential IMPES simulator with switchable flux reconstruction

## Goal

Convert the frozen-flux SPE10 pipeline into a time-stepping `.py` simulator where pressure,
flux, and saturation co-evolve via a saturation-dependent permeability coefficient.
Single file, e.g. `impes_spe10_simulator.py`.

**Source material (port, do not re-derive):**

- `fenicsx/code/fracture problem/LCG_DengGinting_example4_spe10_Q1_no_fracture_hardcurl_PINN.ipynb`
  — mesh construction, Q1 CG solve, dual-mesh face fluxes, `q_p`, hard-curl PINN
  (architecture, features, checkpoint I/O, L-BFGS stagnation logic), explicit upwind
  Buckley-Leverett transport, R_xi / R_tau / R_zeta conservation diagnostics.
- The NLR (Deng local reconstruction) from
  `LCG_DengGinting_example4_spe10_Q1_no_fracture.ipynb`.
- t=0 PINN checkpoint: the saved `hardcurl_pinn_*_Q1_64x64.pt` from the SPE10 hard-curl
  notebook.

## Mathematical model

Sequential (IMPES, single-pass lagged) scheme for the quasilinear elliptic-hyperbolic
system. With constant `M >= 1` from config, define:

- `a(S) = M*S^2 + (1-S)^2` (note `a(0) = 1`)
- `F(S) = M*S^2 / a(S)` (transport flux function; reduces to the current BL flux at M=1)

Per outer step n, given `S^n` on the dual mesh:

1. Project `S^n` to primal cells (mean of the 4 corner-node dual-CV values; precomputed
   index array — fix this convention and keep it).
2. Coefficient `c^n = K(x) * a(S^n)` per cell.
3. Solve `-div(c^n grad p^n) = f`, same BCs and same `f` as the frozen-flux case.
   **`f`, the load vector, and `q_p` never change — build once.**
4. Obtain face fluxes per the chosen `method`.
5. Advance `S^n -> S^{n+1}` by explicit upwind BL transport with the flux frozen within
   the step, fixed-dt substeps inside the outer `DT_outer`.

## Config (dataclass at top; dump to JSON in out_dir at start)

- `method` in {"CG", "NLR", "PINN"} — **single track per run**; the three methods are
  compared by running the simulator three times.
- `M` — coupling strength (run M=1 first as sanity, then M=5, M=10).
- `N_time` — number of outer steps (default 5 for debugging).
- `DT_outer` — outer step size; **identical across method runs** (comparison happens at
  t = n*DT_outer).
- `transport_dt` — fixed transport substep (see "Time step and CFL monitoring").
- `cfl_threshold` — default 1.0.
- `pinn_mode` in {"full", "last_layer"}.
- `lbfgs_max_iter = 1000` plus stagnation early-stop parameters (reuse notebook values).
- `t0_checkpoint_path`.
- `out_dir` — encode method and M in the name, e.g. `run_PINN_M5/`.
- `viz_every = 1`, `full_conservation_every` (default: final step only), `save_every = 1`.
- Torch dtype: **float64 for all PINN work** (load, training, evaluation).
- `--dry-run` CLI flag.

## Time step and CFL monitoring

- Transport dt is **fixed from config** (`transport_dt`; default: the known
  CFL-satisfying value from the frozen-flux notebook at t=0). Inner substepping divides
  `DT_outer` into equal substeps of size <= `transport_dt`. A fixed dt makes all three
  method runs share the identical substep grid, removing time discretization as a
  confound.
- Evaluate `max|F'(S)|` **once at startup** for the configured M by dense sampling of S in
  [0,1]. Do not reuse the M=1 constant 2.0.
- At **every outer step** (after the flux update), compute the implied CFL number with the
  same expression previously used to derive dt, now as a diagnostic:
  `CFL_n = dt * max_faces(|F_face| / PV_upwind) * max|F'(S)|`.
- Log `CFL_n` in the per-step CSV. If `CFL_n > cfl_threshold`: print a clearly visible
  warning with step number and value, set a `cfl_violated` flag column, and **continue the
  run to completion — never abort on CFL violation**.
- End-of-run summary: number of violating steps, first violated step, max CFL over the run.
- Instability tell-tale: log pre-clip min/max of S per step (already computed by the
  transport kernel). CFL > 1 together with pre-clip S escaping [0,1] means results after
  that step are polluted; the summary must state from which step onward.

Expectation: with M = 5-10 the flux behind the front grows roughly by factor M, so a dt
fixed from t=0 may cross CFL ~ 1 mid-run. That is the monitoring working, not a bug; the
user halves `transport_dt` and reruns. The M=1 run should stay below threshold throughout
(doubles as a check that the monitoring is wired correctly).

## Performance requirements (vectorize; no Python loops inside the time loop)

- **Assembly:** uniform rectangular mesh + per-cell constant coefficient means every
  element stiffness = `c_e` x (a single precomputed 4x4 reference matrix). Precompute COO
  row/col index arrays once; per-step values by broadcasting
  (`vals = c_cells[:,None,None] * Ke_ref`). Assemble the load vector once. Dirichlet/well
  row indices precomputed once. Sparse solve via `splu` each step (matrix changes with the
  coefficient, so no reusable factorization).
- **Dual-face CG fluxes:** precompute the geometric sparse operator so per-step integrated
  face fluxes = coefficient-weighted operator applied to nodal p (no per-face loops).
- **NLR:** batch the local solves as stacked dense systems (`np.linalg.solve` on 3-D
  arrays), not a Python loop over CVs.
- **Transport:** port the existing vectorized kernel; only `F(S)`, `max|F'|`, and the
  fixed dt change.

## FluxModel interface (PINN path only; the time loop calls nothing else)

- `FluxModel.load(checkpoint_path)` — t=0 initialization. Loading (not retraining) is
  exact, because `S = 0` initially implies `a = 1`, so the t=0 problem equals the
  frozen-flux problem the checkpoint was trained on. Convert to float64 on load.
- `FluxModel.update(target_face_fluxes) -> face_fluxes` — dispatch on `pinn_mode`:
  - `"full"`: warm-start from current weights; torch L-BFGS (closure,
    `line_search_fn="strong_wolfe"`, history ~ 50), cap `lbfgs_max_iter = 1000`,
    early-stop on stagnation (notebook logic).
  - `"last_layer"`: predicted flux = `Phi @ w + F_qp`, where `Phi` (n_faces x width) is
    built **once** from the frozen hidden features (psi-endpoint differences of each
    hidden feature, per face). Per step solve
    `min_w ||Phi w - (F_target - F_qp)||^2 + ridge*||w||^2` via normal equations
    factorized once; per-step cost = one back-substitution.
- `FluxModel.report() -> dict` — same keys in both modes: final face-RMSE, iterations
  (or `"direct"`), wall-clock. The CSV schema must not depend on the mode.

## Per-step logging and outputs

Append to a CSV every step (crash-safe):

- Stage wall-clocks: S-projection + assembly + solve, flux stage (NLR or PINN update),
  transport, visualization.
- Consecutive drift: `||p^n - p^{n-1}||_L2` and `||F^n - F^{n-1}||_2`. Log PINN
  iterations next to flux drift (correlation expected: steps where the front hits new
  channels need more iterations).
- PINN diagnostics from `FluxModel.report()`.
- Cheap conservation: R_xi stats computed from the stored face fluxes against the exact
  per-CV source integrals (vectorized; every step). Full R_tau / R_zeta quadrature only
  every `full_conservation_every` steps — never every step (it costs ~50 s).
- `CFL_n`, `cfl_violated`, pre-clip min/max S, post-clip min/max S, mass balance,
  n_substeps.

Save per step (`save_every`): `p_step{n}.npy`, `S_step{n}.npy`, `fluxes_step{n}.npy`,
PINN state dict. Dump the config JSON at run start.

Figures (matplotlib **Agg** backend, save PNG, never `plt.show()`), every `viz_every`
steps: pressure map, flux streamlines (cell-centered velocities; evaluate the PINN once on
a fixed cell-center grid), saturation map. Shared colorbars across steps where feasible.

Cross-method comparison lives in a separate `compare.py` that consumes 2-3 out_dirs and
produces error tables and difference maps at the shared outer times — no comparison logic
inside the simulator.

## Validation gates (implement in this order; stop and report if one fails)

1. **`--dry-run`:** N_time=1, method=PINN, load checkpoint, full diagnostics, exit.
   Gate: p, face fluxes, and R_xi at step 0 must reproduce the frozen-flux notebook's
   numbers to machine precision (same coefficient, same weights). This is the regression
   anchor.
2. **M=1 short run:** coefficient varies only within [1/2, 1]*K; results stay close to
   frozen-flux; warm-started L-BFGS should early-stop after few iterations at early steps
   (S ~ 0 means tiny flux drift). Hundreds of iterations at step 1 indicates a bug
   (dtype on load, target scaling).
3. **Conservation invariant:** R_xi for the PINN stays at float64 machine precision at
   every step in both modes — conservation is architectural and must not degrade with time
   stepping. Degradation means a dtype or q_p-consistency bug was introduced.
4. **Last-layer drift:** expect the last-layer-mode face-RMSE to drift upward across steps
   as the frozen features age. Log it; do **not** implement an automatic full-refresh
   trigger.

## Out of scope (do not implement)

Parallelism beyond process-level (running the three method instances concurrently by
hand is fine); compressible flow; Picard iteration (single-pass lagged scheme only);
residual saturations; automatic mode-switching; adaptive dt; any change to the network
architecture, `q_p` construction, or transport source conventions.
