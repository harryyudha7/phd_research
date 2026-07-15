# PINN Update-Mode Substep Comparison

`dt_transport = 1e-05` and `DT_outer = n_substep * dt_transport`.
`N_time = 5`, `pinn_k_layers = 2`.

| n_substep | mode | final RMSE | mean RMSE | total PINN update [s] | mean PINN update [s] | total flux stage [s] |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Frozen | 2.526847e-03 | 2.526399e-03 | 5.351245e-02 | 1.070249e-02 | 7.782672e-02 |
| 1 | Full | 2.244414e-03 | 2.296099e-03 | 1.570257e+02 | 3.140515e+01 | 1.570547e+02 |
| 1 | K-last | 2.471594e-03 | 2.473675e-03 | 4.413010e+01 | 8.826021e+00 | 4.415675e+01 |
| 1 | Linear last | 2.524948e-03 | 2.524582e-03 | 1.346027e-02 | 2.692053e-03 | 6.628677e-02 |
| 1000 | Frozen | 3.251748e-03 | 3.008868e-03 | 6.800873e-02 | 1.360175e-02 | 9.265518e-02 |
| 1000 | Full | 2.317095e-03 | 2.341732e-03 | 1.516669e+02 | 3.033339e+01 | 1.516957e+02 |
| 1000 | K-last | 2.609880e-03 | 2.592299e-03 | 7.802892e+01 | 1.560578e+01 | 7.804834e+01 |
| 1000 | Linear last | 3.106766e-03 | 2.904576e-03 | 1.138755e-02 | 2.277510e-03 | 4.548753e-02 |

## RMSE Improvement Relative To Frozen

Positive values mean the update reduced the face RMSE compared with the frozen checkpoint flux.

| n_substep | mode | final improvement | mean improvement |
|---:|---|---:|---:|
| 1 | Full | 11.18% | 9.12% |
| 1 | K-last | 2.19% | 2.09% |
| 1 | Linear last | 0.08% | 0.07% |
| 1000 | Full | 28.74% | 21.49% |
| 1000 | K-last | 19.74% | 13.32% |
| 1000 | Linear last | 4.46% | 3.33% |

## RMSE Improvement Relative To Previous-Step Network

This uses the network inherited at the start of the same outer step as the baseline.

| n_substep | mode | final step-update improvement | mean step-update improvement |
|---:|---|---:|---:|
| 1 | Full | 0.72% | 2.32% |
| 1 | K-last | 0.00% | 0.44% |
| 1 | Linear last | 0.00% | 0.01% |
| 1000 | Full | 7.71% | 10.85% |
| 1000 | K-last | 4.63% | 6.19% |
| 1000 | Linear last | 0.65% | 1.24% |

Per-step improvement data: `comparison_improvement.csv`.
Figure: `comparison_step_bars.png`.
Raw data: `comparison_data.csv`.

RMSE-bar labels: black `F` = improvement relative to frozen; purple `P` = improvement relative to the previous-step network.
Time bars use `pinn_wall_s`, i.e. the PINN update itself. The broader flux-stage time remains in `flux_s`.