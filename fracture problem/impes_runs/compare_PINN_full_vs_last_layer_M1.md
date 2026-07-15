# PINN update-mode comparison, M=1

Inputs:

- full: `/home/muchamad/PhD/fenicsx/code/fracture problem/impes_runs/run_PINN_M1`
- last_layer: `/home/muchamad/PhD/fenicsx/code/fracture problem/impes_runs/run_PINN_M1_last_layer`
- frozen: `/home/muchamad/PhD/fenicsx/code/fracture problem/impes_runs/run_PINN_M1_frozen`

Timing:

| metric | full | last layer | frozen/no update | last/full | frozen/full |
|---|---:|---:|---:|---:|---:|
| total flux stage | 6.088064e+02 | 2.727723e-02 | 1.084450e-01 | 4.480444e-05 | 1.781272e-04 |
| mean flux stage / step | 1.217613e+02 | 5.455446e-03 | 2.168900e-02 | 4.480444e-05 | 1.781272e-04 |
| total PINN update wall time | 6.087826e+02 | 0.000000e+00 | 8.145234e-02 | 0.000000e+00 | 1.337954e-04 |

Accuracy / diagnostics:

| metric | full | last layer | frozen/no update |
|---|---:|---:|---:|
| mean PINN-CG face RMSE | 2.148491e-03 | 2.904576e-03 | 3.008868e-03 |
| final PINN-CG face RMSE | 2.116834e-03 | 3.106766e-03 | 3.251748e-03 |
| mean R_xi RMSE | 1.930156e-15 | 1.931164e-15 | 1.930692e-15 |
| max CFL | 2.198575e-01 | 2.198756e-01 | 2.187455e-01 |
| final max S | 9.828799e-01 | 9.828799e-01 | 9.828799e-01 |

Final saturation difference relative to the full update:

| metric | last layer | frozen/no update |
|---|---:|---:|
| RMSE | 2.531600e-02 | 2.979398e-02 |
| max abs | 4.503856e-01 | 5.040066e-01 |

Improvement of the final face RMSE over the frozen checkpoint:

- Full update: 34.90%
- Last-layer update: 4.46%

Figure: `compare_PINN_update_modes_M1.png`.
