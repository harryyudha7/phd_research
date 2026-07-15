# IMPES Simulator Implementation Notes

These choices are simulator implementation details in addition to `impes_simulator_spec.md`.

- Pressure figures use a fixed color scale `[pressure_vmin, pressure_vmax] = [-1, 0.5]`.
- Flux figures use magnitude-scaled arrows at sampled cell centers, matching the earlier visualization style.
- By default this script runs the Deng/NLR track with `transport_dt = 1e-5`, so each pressure solve is followed by stable explicit transport subcycling over `DT_outer`. Passing `--transport_dt <value>` overrides this.
- By default `N_time = 1000` and `viz_every = N_time`, so the simulator performs 1000 pressure/transport updates and writes one figure at the final step. Passing `--viz_every k` restores intermediate figures.
- Figure time labels use the outer IMPES time `t = step * DT_outer`.
- L-BFGS progress is printed on one overwritten terminal line, so the final wall-time and error messages remain readable.
- `pinn_iterations` counts L-BFGS closure calls, including line-search evaluations, so it can exceed `lbfgs_max_iter`.
- The final PINN-CG face-flux error history is saved as `pinn_cg_error_history.csv` and `pinn_cg_error_history.png` when PINN mode is used. The x-axis is the optimizer call/update index, not an Adam epoch, because the simulator updates the network by L-BFGS.
- `pinn_mode="frozen"` evaluates the loaded checkpoint without any update. The frozen face flux is cached after the first evaluation, so this is the no-update baseline for measuring how much the full/last-layer updates help.
- `pinn_mode="k_last_layer"` freezes the early network layers and updates only the final `pinn_k_layers` linear layers by L-BFGS. This is a middle option between full-weight L-BFGS and direct linear last-layer regression.
- `method="NLR"` uses the Deng-Ginting local Q1 postprocessing on element-local nodal control volumes, then assembles the resulting fluxes on the nodal-dual transport mesh.
- `method="PROJ"` is an extra global conservative graph projection on the nodal-dual transport mesh. It is not the Deng-Ginting local NLR reconstruction.
- The Deng source term intentionally keeps the notebook's two local quantities separate: the source integrated over each quarter nodal CV and the Q1 load projection. They cancel for the current equal-split, cell-integrated well source, but the split is retained for auditing and future non-uniform sources.
- Deng reconstruction precomputes the time-independent geometric local matrices and their pseudo-inverses at startup. During each IMPES step the local solve is a batched pseudo-inverse multiply scaled by the current scalar cell coefficient; no per-cell SVD is performed in the time loop.
- `--validate-deng --validate-only` runs the t=0 Deng port gate (`S=0`, `coeff=K`) and reports the dual residual before any IMPES time stepping. `--deng-reference-flux` can compare against a saved frozen-notebook Deng face-flux vector when available.
