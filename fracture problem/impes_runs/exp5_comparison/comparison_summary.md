# Exp5 IMPES Comparison

Interpretation guardrails:

- If Frozen is close to NLR@1 at T=0.1, the scoped conclusion is that flux updates do not matter much at M=1 for this weak-coupling/factor-2 drift setting, not that updates never matter.
- CG differences from NLR@1 isolate non-conservation rather than flux staleness. If CG is closest to NLR@1, report that plainly because it challenges the conservation thesis in this dynamic setting.
- NLR is exact on interior/source dual CVs; boundary-CV residuals are reported separately in the run logs and timing table context.
- PoU@1 is the per-step adaptive hard-curl head: compare it to NLR@1 for apples-to-apples refresh cadence and to Linear/Frozen for update expressiveness.

### t = 0.05

| group | pair | L2 | RMSE | max | pattern |
|---|---:|---:|---:|---:|---|
| conservation | CG vs NLR@1 | 1.667424e-02 | 1.667424e-02 | 2.841580e-01 | front-localized |
| staleness | NLR@1000 vs NLR@1 | 4.588576e-03 | 4.588576e-03 | 8.061778e-02 | front-localized |
| staleness | Frozen vs NLR@1 | 3.190701e-02 | 3.190701e-02 | 5.216134e-01 | front-localized |
| method matched cadence | PINN full @1000 vs NLR@1000 | 2.226286e-03 | 2.226286e-03 | 4.076074e-02 | front-localized |
| method vs NLR@1 | PINN full @1000 vs NLR@1 | 5.190589e-03 | 5.190589e-03 | 9.147492e-02 | front-localized |
| update expressiveness | Linear vs Frozen | 1.338350e-02 | 1.338350e-02 | 2.348569e-01 | front-localized |
| update expressiveness | Linear vs NLR@1 | 2.658902e-02 | 2.658902e-02 | 4.424391e-01 | front-localized |
| update expressiveness | PoU@1 vs Frozen | 3.148758e-02 | 3.148758e-02 | 5.198161e-01 | front-localized |
| method matched cadence | PoU@1 vs NLR@1 | 2.072246e-03 | 2.072246e-03 | 4.500196e-02 | front-localized |
| CG anchor | Frozen vs CG | 3.575510e-02 | 3.575510e-02 | 5.274514e-01 | front-localized |
| CG anchor | Linear vs CG | 3.067832e-02 | 3.067832e-02 | 4.482771e-01 | front-localized |
| CG anchor | PoU@1 vs CG | 1.662340e-02 | 1.662340e-02 | 2.825023e-01 | front-localized |
| CG anchor | PINN full @1000 vs CG | 1.728485e-02 | 1.728485e-02 | 2.831862e-01 | front-localized |
| CG anchor | NLR@1 vs CG | 1.667424e-02 | 1.667424e-02 | 2.841580e-01 | front-localized |

### t = 0.10

| group | pair | L2 | RMSE | max | pattern |
|---|---:|---:|---:|---:|---|
| conservation | CG vs NLR@1 | 2.589555e-02 | 2.589555e-02 | 3.219534e-01 | front-localized |
| staleness | NLR@1000 vs NLR@1 | 6.827779e-03 | 6.827779e-03 | 1.753440e-01 | front-localized |
| staleness | Frozen vs NLR@1 | 3.431996e-02 | 3.431996e-02 | 5.872781e-01 | front-localized |
| method matched cadence | PINN full @1000 vs NLR@1000 | 3.500701e-03 | 3.500701e-03 | 9.415901e-02 | front-localized |
| method vs NLR@1 | PINN full @1000 vs NLR@1 | 7.621414e-03 | 7.621414e-03 | 1.710049e-01 | front-localized |
| update expressiveness | Linear vs Frozen | 2.812099e-02 | 2.812099e-02 | 5.291214e-01 | front-localized |
| update expressiveness | Linear vs NLR@1 | 3.846268e-02 | 3.846268e-02 | 5.902606e-01 | front-localized |
| update expressiveness | PoU@1 vs Frozen | 3.423464e-02 | 3.423464e-02 | 5.754979e-01 | front-localized |
| method matched cadence | PoU@1 vs NLR@1 | 2.051665e-03 | 2.051665e-03 | 3.782424e-02 | front-localized |
| CG anchor | Frozen vs CG | 4.194448e-02 | 4.194448e-02 | 5.301351e-01 | front-localized |
| CG anchor | Linear vs CG | 4.387140e-02 | 4.387140e-02 | 5.331175e-01 | front-localized |
| CG anchor | PoU@1 vs CG | 2.580903e-02 | 2.580903e-02 | 3.206460e-01 | front-localized |
| CG anchor | PINN full @1000 vs CG | 2.632052e-02 | 2.632052e-02 | 3.208861e-01 | front-localized |
| CG anchor | NLR@1 vs CG | 2.589555e-02 | 2.589555e-02 | 3.219534e-01 | front-localized |

## Timing

| method | steps | total flux s | mean flux s | median flux s | final Rxi int | final Rxi bnd | total solve s | total transport s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CG | 10000 | 50.235 | 0.005024 | 0.004663 | 5.918e-03 | 1.302e-03 | 173.149 | 4.427 |
| Frozen | 10000 | 48.922 | 0.004892 | 0.004481 | 1.923e-15 | 2.051e-15 | 166.511 | 4.420 |
| NLR@1 | 10000 | 547.651 | 0.054765 | 0.055344 | 1.684e-15 | 1.401e-03 | 162.914 | 4.246 |
| Linear | 10000 | 66.201 | 0.006620 | 0.006242 | 1.925e-15 | 2.051e-15 | 165.806 | 4.408 |
| PoU@1 | 10000 | 283.735 | 0.028374 | 0.021599 | 2.207e-15 | 2.726e-15 | 430.258 | 13.690 |
| PINN full @1000 | 10 | 929.814 | 92.981374 | 101.380428 | 1.923e-15 | 2.049e-15 | 0.185 | 4.037 |
| NLR@1000 | 10 | 0.607 | 0.060742 | 0.063667 | 1.700e-15 | 1.374e-03 | 0.179 | 2.796 |