# Fisher-Hodge mechanism experiments - summary

Relative L2 error = trajectory-median over 16 test trajectories, then mean±std over 3 seeds (`*` = some seed non-finite, median shown; `inf` = majority of trajectories blew up). T = T_train = 20 steps (t=1.0); 10T = t=10.

## S1 - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | F_true monotone | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.00304±8.8e-05 | 0.00314±5.5e-05 | 0.00342±0.00012 | 0.00343±3.2e-05 | 3.02e-07 | 3.76e-04 | 0/48 | 0/48 | 1.00 | 1.00 |
| Euclid GF (B2) | 34691 | 0.0641±0.004 | 0.0755±0.0043 | 0.133±0.0038 | 0.254±0.0059 | 5.67e-01 | -2.11e+02 | 46/48 | 2/48 | 0.00 | 0.96 |
| GNN-ODE (B1) | 138369 | 0.146±0.022 | 0.16±0.018 | 0.251±0.02 | 0.285±0.039 | 1.96e+00 | -3.64e+01 | 11/48 | 0/48 | 0.02 | n/a |

## S1 - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | F_true monotone | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.0149±0.00061 | 0.0162±0.00055 | 0.0171±0.00033 | 0.0169±0.00055 | 4.02e-07 | 5.31e-05 | 0/48 | 0/48 | 0.94 | 0.90 |
| Euclid GF (B2) | 34691 | 0.221±0.0039 | 0.213±0.0047 | 0.196±0.0014 | 0.27±0.0063 | 6.67e-01 | -2.47e+01 | 46/48 | 5/48 | 0.00 | 0.88 |
| GNN-ODE (B1) | 138369 | 0.413±0.059 | 0.527±0.076 | 0.721±0.27 | 1.48±1.4 | 3.99e+00 | -6.53e+01 | 47/48 | 0/48 | 0.00 | n/a |

## S1 - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | F_true monotone | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.00303±0.0001 | 0.00266±0.00014 | 0.00234±0.00012 | 0.00256±7.2e-05 | 2.33e-07 | 5.56e-04 | 0/48 | 0/48 | 1.00 | 1.00 |
| Euclid GF (B2) | 34691 | 0.0672±0.0064 | 0.0786±0.0057 | 0.115±0.0028 | 0.147±0.0023 | 6.93e-01 | -2.98e-01 | 48/48 | 0/48 | 0.00 | 1.00 |
| GNN-ODE (B1) | 138369 | 0.197±0.016 | 0.19±0.0025 | 0.291±0.056 | 0.324±0.07 | 1.22e+00 | -1.90e+01 | 15/48 | 0/48 | 0.25 | n/a |

## S1 - cost

| model | params | train time (s) | rollout 16x200 steps (s) | substeps/DT mean (max) |
|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 4502 | 27.1 | 8.0 (8) |
| Euclid GF (B2) | 34691 | 1190 | 4.6 | 8.0 (8) |
| GNN-ODE (B1) | 138369 | 7169 | 21.9 | 8.0 (8) |

## S2 - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | curl(J) err @10T |
|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 52100 | 0.00371±6.6e-05 | 0.00355±4.8e-05 | 0.00318±0.00016 | 0.00325±0.00016 | 3.42e-07 | 4.68e-04 | 0/48 | 0/48 | 0.0125±0.00077 |
| Euclid GF (B2) | 52484 | 0.0231±0.00051 | 0.0254±0.00031 | 0.0275±0.0024 | 0.0296±0.002 | 4.61e-02 | -3.09e-02 | 32/48 | 0/48 | 0.621±0.14 |
| GNN-ODE (B1) | 138369 | 0.151±0.03 | 0.202±0.057 | 0.253±0.036 | 0.263±0.032 | 5.46e-01 | -4.11e-01 | 15/48 | 0/48 | n/a |

## S2 - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | curl(J) err @10T |
|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 52100 | 0.0147±0.00055 | 0.0154±0.0011 | 0.0139±0.001 | 0.014±0.00078 | 3.39e-07 | 1.82e-05 | 0/48 | 0/48 | 0.0331±0.0027 |
| Euclid GF (B2) | 52484 | 0.0511±0.0034 | 0.0598±0.00089 | 0.0605±0.0035 | 0.0689±0.0074 | 8.18e-02 | -7.17e-01 | 48/48 | 0/48 | 0.665±0.15 |
| GNN-ODE (B1) | 138369 | 0.312±0.045 | 0.472±0.12 | 0.846±0.35 | 0.957±0.5 | 3.04e+00 | -3.55e+02 | 48/48 | 0/48 | n/a |

## S2 - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | curl(J) err @10T |
|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 52100 | 0.00307±0.00016 | 0.00331±0.00013 | 0.00326±0.00012 | 0.00324±0.00011 | 3.74e-07 | 4.72e-04 | 0/48 | 0/48 | 0.0138±0.00073 |
| Euclid GF (B2) | 52484 | 0.0195±0.0015 | 0.0236±0.0019 | 0.0313±0.0025 | 0.0309±0.0012 | 9.39e-02 | -5.03e-02 | 33/48 | 0/48 | 0.708±0.17 |
| GNN-ODE (B1) | 138369 | 0.22±0.045 | 0.219±0.067 | 0.283±0.044 | 0.296±0.043 | 6.90e-01 | -4.00e-01 | 17/48 | 0/48 | n/a |

## S2 - cost

| model | params | train time (s) | rollout 16x200 steps (s) | substeps/DT mean (max) |
|---|---|---|---|---|
| Fisher-Hodge (M) | 52100 | 6154 | 40.3 | 8.0 (8) |
| Euclid GF (B2) | 52484 | 1689 | 5.5 | 8.0 (8) |
| GNN-ODE (B1) | 138369 | 7150 | 17.7 | 8.0 (8) |

## S3 - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | front-speed err | max overshoot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 51716 | 0.00242±0.00028 | 0.00421±0.00053 | 0.0556±0.018 | 0.162±0.1 | 4.39e-01 | 1.00e-04 | 0/48 | 0/48 | 0.0797±0.0082 | 5.18e-01 |
| Euclid GF (B2) | 51716 | 0.0118±0.0033 | 0.0177±0.0033 | 0.319±0.12 | 3.61±1.9 | 1.24e+01 | -9.60e-04 | 29/48 | 0/48 | 0.298±0.041 | 1.28e+01 |
| GNN-ODE (B1) | 138369 | 0.467±0.008 | 0.498±0.0074 | 0.855±0.0044 | 0.898±0.0027 | 2.38e+00 | 1.00e-04 | 0/48 | 0/48 | 1±0 | -5.31e-02 |

## S3 - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | front-speed err | max overshoot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 51716 | 0.00421±0.00031 | 0.00737±0.00029 | 0.0617±0.021 | 0.161±0.11 | 4.30e-01 | 1.00e-04 | 0/48 | 0/48 | 0.0786±0.0057 | 5.36e-01 |
| Euclid GF (B2) | 51716 | 0.0148±0.0022 | 0.0315±0.0044 | 0.362±0.12 | 3.81±1.9 | 1.06e+01 | -7.41e-04 | 48/48 | 0/48 | 0.201±0.064 | 1.11e+01 |
| GNN-ODE (B1) | 138369 | 0.467±0.017 | 1.02±0.69 | 1.68±0.95 | 2.43±1.6 | 3.31e+00 | -2.54e+01 | 32/48 | 0/48 | 1±0 | -3.39e-02 |

## S3 - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | front-speed err | max overshoot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 51716 | 0.00151±6.3e-05 | 0.00218±0.00048 | 0.0555±0.02 | 0.162±0.11 | 4.32e-01 | 1.00e-04 | 0/48 | 0/48 | 0.0564±0.0048 | 5.02e-01 |
| Euclid GF (B2) | 51716 | 0.00945±0.0021 | 0.00861±0.002 | 0.297±0.1 | 3.68±1.9 | 1.21e+01 | -4.77e-04 | 29/48 | 0/48 | 0.185±0.023 | 1.24e+01 |
| GNN-ODE (B1) | 138369 | 0.381±0.0054 | 0.332±0.01 | 0.859±0.0037 | 0.898±0.003 | 9.12e-01 | 1.00e-04 | 0/48 | 0/48 | 1±0 | -3.66e-01 |

## S3 - cost

| model | params | train time (s) | rollout 16x200 steps (s) | substeps/DT mean (max) |
|---|---|---|---|---|
| Fisher-Hodge (M) | 51716 | 5382 | 21.1 | 8.0 (8) |
| Euclid GF (B2) | 51716 | 1663 | 5.3 | 8.0 (8) |
| GNN-ODE (B1) | 138369 | 6615 | 14.5 | 8.0 (8) |

## Diagnostic: learned-Lyapunov violations of FH on S1 OOD-potential (check_lyap.py -> check_lyap_S1.json)

With the nominal 8 SSP-RK2 substeps per DT, 1-2 of 16 OOD-potential trajectories show a small increase of the learned
free energy F_theta (max +1.3e-3). Re-rolling the same trained models with 32 substeps gives 16/16 monotone in all 3 seeds
with unchanged accuracy (rel err @10T 0.0172 vs 0.0172), so the violation is an explicit time-step (energy-stability) effect
on the stiffer OOD potentials, not a failure of the continuous-time structure. Positivity-driven step halving was never
triggered beyond 8.1 substeps/DT on average.

## Notes
- Euclid GF and GNN-ODE runs had non-finite training losses in some iterations (Euclid S1: 23-46 skipped steps per run); see logs.
- GPU 0 was shared with other jobs throughout; train times above are wall-clock under contention, not clean benchmarks.
