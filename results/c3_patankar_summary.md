# C3 Brusselator: explicit positive substeps vs the Patankar integrator

Code: `hyperme/integrators.py` (`mprk`), `hyperme/c3_patankar.py`, test `hyperme/tests/test_patankar_c3.py`. CPU float64, machine 49, 16 `test_id` trajectories, sampling step 0.2, 200 steps (10T). "dist. to explicit" is the relative L2 distance at 10T to the rollout of the same model with the explicit substeps, which are converged in time. One seed.

The Patankar step weights every one-way channel by `x_m / σ_m` of the one species it consumes. Each stage is then one linear solve with an M-matrix. Concentrations stay positive and every moiety is conserved for any step size. MPRK22 (Kopecz and Meister 2018, α = 1) is second order, and Patankar–Euler is first order. Gradients pass through the dense solve, so the scheme can be trained through.

## 1. Same model (prototype checkpoint, trained with the explicit substeps)

| integrator | steps per 0.2 | wall s (200 steps, 16 traj) | 10T err all | homog / spatial | T err | dist. to explicit 10T | min c | min sigma |
|---|---|---|---|---|---|---|---|---|
| explicit | 28.6 | 20.0 | 0.0477 | 0.0579 / 0.0357 | 0.0229 | 0 | 0.299 | 434 |
| MPRK22 × 1 | 1 | 1.7 | 0.3375 | 0.2829 / 0.3638 | 0.1234 | 0.353 | 0.299 | 439 |
| MPRK22 × 2 | 2 | 3.0 | 0.3328 | 0.3881 / 0.2766 | 0.0636 | 0.266 | 0.299 | 435 |
| MPRK22 × 4 | 4 | 5.4 | 0.1419 | 0.1715 / 0.0206 | 0.0331 | 0.065 | 0.299 | 434 |
| MPRK22 × 8 | 8 | 12.7 | 0.0752 | 0.0955 / 0.0206 | 0.0240 | 0.021 | 0.299 | 434 |
| MPRK22 × 16 | 16 | 20.4 | 0.0511 | 0.0660 / 0.0227 | 0.0223 | 0.0058 | 0.299 | 434 |
| Patankar–Euler × 1 | 1 | 0.9 | 0.4088 | 0.4479 / 0.3632 | 0.2067 | 0.413 | 0.299 | 464 |
| Patankar–Euler × 4 | 4 | 2.7 | 0.3888 | 0.6204 / 0.2797 | 0.0930 | 0.370 | 0.299 | 440 |

Every integrator keeps c > 0 and σ > 0. MPRK22 converges to the explicit rollout at second order. The explicit step count is set by positivity (dt × consumption / c ≤ 0.5), and the error at large Patankar steps is phase drift of the limit cycle. At equal accuracy the two cost about the same on this model.

## 2. Retrained with MPRK22 (2000 iterations, seed 0)

Per training iteration at the end-of-training stiffness (K = 8, batch 8, same machine): explicit 2.19 s, MPRK22 × 4 0.39 s, MPRK22 × 8 0.76 s. The explicit prototype run took 2522 s on axiom.

| training integrator | train s (49) | 10T own integrator ID / OOD | 10T under explicit ID | dist. own to explicit |
|---|---|---|---|---|
| explicit (prototype) | 2522 (axiom) | 0.0477 / 0.560 | 0.0477 | 0 |
| MPRK22 × 1 | 178 | 0.222 / 0.385 | – | – |
| MPRK22 × 2 | 335 | 0.0387 / 0.364 | 0.371 | 0.40 |
| MPRK22 × 4 | 656 | 0.0257 / 0.513 | 0.0449 | 0.065 |
| MPRK22 × 8 | 1234 | 0.0330 / 0.484 | 0.0226 | 0.018 |

With 2 or fewer steps per sampling interval the model learns a map tied to the step size. It fits under its own integrator, but under the exact dynamics its error is 0.37, so its rates and thermodynamic readouts do not describe the continuous-time system. With 8 steps the model is accurate under both integrators (0.033 and 0.023) and trains 2.9 times faster per iteration than with explicit substeps. Four steps sit in between: 5.6 times faster, 0.026 under its own integrator and 0.045 under the exact one. OOD (b in [3.2, 3.7]) remains poor for every training integrator. One seed only.
