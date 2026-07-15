# IMPES Simulator Implementation Notes

These choices are simulator implementation details in addition to `impes_simulator_spec.md`.

- Pressure figures use a fixed color scale `[pressure_vmin, pressure_vmax] = [-1, 0.5]`.
- Flux figures use equal-length arrows at sampled cell centers; arrow color represents `|v|` with blue low and red high.
- Figure time labels use the outer IMPES time `t = step * DT_outer`. The transport equation is still subcycled with `transport_dt`.
- L-BFGS progress is printed on one overwritten terminal line, so the final wall-time and error messages remain readable.
- The final PINN-CG face-flux error history is saved as `pinn_cg_error_history.csv` and `pinn_cg_error_history.png` when PINN mode is used. The x-axis is the optimizer call/update index, not an Adam epoch, because the simulator updates the network by L-BFGS.
