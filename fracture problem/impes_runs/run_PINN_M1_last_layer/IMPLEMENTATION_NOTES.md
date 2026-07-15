# IMPES Simulator Implementation Notes

These choices are simulator implementation details in addition to `impes_simulator_spec.md`.

- Pressure figures use a fixed color scale `[pressure_vmin, pressure_vmax] = [-1, 0.5]`.
- Flux figures use magnitude-scaled arrows at sampled cell centers, matching the earlier visualization style.
- By default `transport_dt = DT_outer`, so each pressure solve is followed by exactly one transport update. Passing `--transport_dt < DT_outer` enables optional subcycling.
- By default `N_time = 1000` and `viz_every = N_time`, so the simulator performs 1000 pressure/transport updates and writes one figure at the final step. Passing `--viz_every k` restores intermediate figures.
- Figure time labels use the outer IMPES time `t = step * DT_outer`.
- L-BFGS progress is printed on one overwritten terminal line, so the final wall-time and error messages remain readable.
- The final PINN-CG face-flux error history is saved as `pinn_cg_error_history.csv` and `pinn_cg_error_history.png` when PINN mode is used. The x-axis is the optimizer call/update index, not an Adam epoch, because the simulator updates the network by L-BFGS.
