# Fisher-Hodge stress tests (2026-10-19) - summary

Relative L2 error = trajectory-median over 16 test trajectories, then mean±std over seeds (`*` = some seed non-finite, median shown; `inf` = majority blew up). T = T_train = 20 steps (t=1); 10T = t=10. "peak err" = |max rho_model - max rho_true| / max rho_true at 10T. Rows marked (M)/(B2)/(B1, 4k it) on S1-S3 are the first-round runs (same data, same code path); everything else is new.

## 1. Strong baselines on S1 / S2 / S3

### S1 - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.00304±8.8e-05 | 0.00314±5.5e-05 | 0.00342±0.00012 | 0.00343±3.2e-05 | 3.02e-07 | 3.76e-04 | 0/48 | 0/48 | n/a | 1.00 |
| Euclid GF (B2) | 34691 | 0.0641±0.004 | 0.0755±0.0043 | 0.133±0.0038 | 0.254±0.0059 | 5.67e-01 | -2.11e+02 | 46/48 | 2/48 | n/a | 0.96 |
| GNN-ODE (B1, 4k it) | 138369 | 0.146±0.022 | 0.16±0.018 | 0.251±0.02 | 0.285±0.039 | 1.96e+00 | -3.64e+01 | 11/48 | 0/48 | n/a | n/a |
| JKO free energy, amortised (B5, RML-style) | 4418 | 0.0126±0.00023 | 0.0109±0.00024 | 0.00914±0.00035 | 0.00855±0.00032 | 4.79e-07 | 3.72e-04 | 0/48 | 0/48 | 0.0163±0.0007 | 1.00 |
| JKO free energy, per-trajectory fit (B5', RML transductive) | 0 | 0.0299 | 0.0253 | 0.0259 | 0.0274 | 5.07e-07 | 3.62e-04 | 0/16 | 0/16 | 0.02 | 1.00 |

### S1 - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.0149±0.00061 | 0.0162±0.00055 | 0.0171±0.00033 | 0.0169±0.00055 | 4.02e-07 | 5.31e-05 | 0/48 | 0/48 | n/a | 0.90 |
| Euclid GF (B2) | 34691 | 0.221±0.0039 | 0.213±0.0047 | 0.196±0.0014 | 0.27±0.0063 | 6.67e-01 | -2.47e+01 | 46/48 | 5/48 | n/a | 0.88 |
| GNN-ODE (B1, 4k it) | 138369 | 0.413±0.059 | 0.527±0.076 | 0.721±0.27 | 1.48±1.4 | 3.99e+00 | -6.53e+01 | 47/48 | 0/48 | n/a | n/a |
| JKO free energy, amortised (B5, RML-style) | 4418 | 0.021±0.00075 | 0.0212±0.00072 | 0.0175±0.0009 | 0.0167±0.00036 | 3.89e-07 | 1.76e-05 | 0/48 | 0/48 | 0.0269±0.00059 | 1.00 |
| JKO free energy, per-trajectory fit (B5', RML transductive) | 0 | 0.0241 | 0.0193 | 0.025 | 0.0308 | 6.05e-07 | 1.45e-05 | 0/16 | 0/16 | 0.0175 | 1.00 |

### S1 - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.00303±0.0001 | 0.00266±0.00014 | 0.00234±0.00012 | 0.00256±7.2e-05 | 2.33e-07 | 5.56e-04 | 0/48 | 0/48 | n/a | 1.00 |
| Euclid GF (B2) | 34691 | 0.0672±0.0064 | 0.0786±0.0057 | 0.115±0.0028 | 0.147±0.0023 | 6.93e-01 | -2.98e-01 | 48/48 | 0/48 | n/a | 1.00 |
| GNN-ODE (B1, 4k it) | 138369 | 0.197±0.016 | 0.19±0.0025 | 0.291±0.056 | 0.324±0.07 | 1.22e+00 | -1.90e+01 | 15/48 | 0/48 | n/a | n/a |
| JKO free energy, amortised (B5, RML-style) | 4418 | 0.0116±0.00017 | 0.0114±0.00017 | 0.0093±0.0002 | 0.00926±0.00029 | 6.27e-07 | 5.45e-04 | 0/48 | 0/48 | 0.0175±0.00049 | 1.00 |
| JKO free energy, per-trajectory fit (B5', RML transductive) | 0 | 0.0336 | 0.0279 | 0.031 | 0.031 | 3.41e-07 | 5.51e-04 | 0/16 | 0/16 | 0.0176 | 1.00 |

### S1 - cost

| model | params | iters | final train loss | train time (s) | rollout 16x200 steps (s) |
|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 4000 | 7.54e-06 | 4502 | 27.1 |
| Euclid GF (B2) | 34691 | 4000 | 1.31e-03 | 1190 | 4.6 |
| GNN-ODE (B1, 4k it) | 138369 | 4000 | 4.14e-03 | 7169 | 21.9 |
| JKO free energy, amortised (B5, RML-style) | 4418 | 4000 | 1.20e-02 | 308 | 31.4 |
| JKO free energy, per-trajectory fit (B5', RML transductive) | 0 | - | - | 0 | 34.2 |

### S2 - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | curl(J) err @10T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 52100 | 0.00371±6.6e-05 | 0.00355±4.8e-05 | 0.00318±0.00016 | 0.00325±0.00016 | 3.42e-07 | 4.68e-04 | 0/48 | 0/48 | n/a | 0.0125±0.00077 |
| Euclid GF (B2) | 52484 | 0.0231±0.00051 | 0.0254±0.00031 | 0.0275±0.0024 | 0.0296±0.002 | 4.61e-02 | -3.09e-02 | 32/48 | 0/48 | n/a | 0.621±0.14 |
| GNN-ODE (B1, 4k it) | 138369 | 0.151±0.03 | 0.202±0.057 | 0.253±0.036 | 0.263±0.032 | 5.46e-01 | -4.11e-01 | 15/48 | 0/48 | n/a | n/a |
| JKO free energy, amortised (B5, RML-style) | 4418 | 0.244±0.00026 | 0.218±0.00037 | 0.226±0.00038 | 0.249±0.00065 | 5.21e-07 | 3.95e-04 | 0/48 | 0/48 | 0.0743±0.0014 | 1±0 |
| JKO free energy, per-trajectory fit (B5', RML transductive) | 0 | 0.0732 | 0.193 | 0.377 | 0.402 | 4.93e-07 | 5.74e-04 | 0/16 | 0/16 | 0.274 | 1 |

### S2 - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | curl(J) err @10T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 52100 | 0.0147±0.00055 | 0.0154±0.0011 | 0.0139±0.001 | 0.014±0.00078 | 3.39e-07 | 1.82e-05 | 0/48 | 0/48 | n/a | 0.0331±0.0027 |
| Euclid GF (B2) | 52484 | 0.0511±0.0034 | 0.0598±0.00089 | 0.0605±0.0035 | 0.0689±0.0074 | 8.18e-02 | -7.17e-01 | 48/48 | 0/48 | n/a | 0.665±0.15 |
| GNN-ODE (B1, 4k it) | 138369 | 0.312±0.045 | 0.472±0.12 | 0.846±0.35 | 0.957±0.5 | 3.04e+00 | -3.55e+02 | 48/48 | 0/48 | n/a | n/a |
| JKO free energy, amortised (B5, RML-style) | 4418 | 0.39±0.0024 | 0.449±0.0017 | 0.452±0.0091 | 0.459±0.0021 | 7.51e-07 | 6.63e-06 | 0/48 | 0/48 | 0.232±0.019 | 1±3.5e-07 |
| JKO free energy, per-trajectory fit (B5', RML transductive) | 0 | 0.0957 | 0.261 | 0.464 | 0.533 | 3.17e-07 | 1.40e-05 | 0/16 | 0/16 | 0.373 | 1 |

### S2 - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | curl(J) err @10T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 52100 | 0.00307±0.00016 | 0.00331±0.00013 | 0.00326±0.00012 | 0.00324±0.00011 | 3.74e-07 | 4.72e-04 | 0/48 | 0/48 | n/a | 0.0138±0.00073 |
| Euclid GF (B2) | 52484 | 0.0195±0.0015 | 0.0236±0.0019 | 0.0313±0.0025 | 0.0309±0.0012 | 9.39e-02 | -5.03e-02 | 33/48 | 0/48 | n/a | 0.708±0.17 |
| GNN-ODE (B1, 4k it) | 138369 | 0.22±0.045 | 0.219±0.067 | 0.283±0.044 | 0.296±0.043 | 6.90e-01 | -4.00e-01 | 17/48 | 0/48 | n/a | n/a |
| JKO free energy, amortised (B5, RML-style) | 4418 | 0.209±0.0015 | 0.21±0.00046 | 0.233±0.00074 | 0.251±0.00045 | 4.44e-07 | 4.01e-04 | 0/48 | 0/48 | 0.168±0.0023 | 1±4.5e-07 |
| JKO free energy, per-trajectory fit (B5', RML transductive) | 0 | 0.0811 | 0.236 | 0.366 | 0.393 | 3.83e-07 | 4.52e-04 | 0/16 | 0/16 | 0.277 | 1 |

### S2 - cost

| model | params | iters | final train loss | train time (s) | rollout 16x200 steps (s) |
|---|---|---|---|---|---|
| Fisher-Hodge (M) | 52100 | 4000 | 5.03e-06 | 6154 | 40.3 |
| Euclid GF (B2) | 52484 | 4000 | 1.12e-04 | 1689 | 5.5 |
| GNN-ODE (B1, 4k it) | 138369 | 4000 | 1.80e-03 | 7150 | 17.7 |
| JKO free energy, amortised (B5, RML-style) | 4418 | 4000 | 2.62e-01 | 337 | 33.4 |
| JKO free energy, per-trajectory fit (B5', RML transductive) | 0 | - | - | 0 | 34.5 |

### S3 - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | front-speed err | max overshoot | Linf sat. err@10T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 51716 | 0.00242±0.00028 | 0.00421±0.00053 | 0.0556±0.018 | 0.162±0.1 | 4.39e-01 | 1.00e-04 | 0/48 | 0/48 | n/a | 0.0797±0.0082 | 5.18e-01 | n/a |
| Fisher-Hodge + reservoir-edge WFR reaction (M2') | 68357 | 0.0027±9.4e-05 | 0.00567±0.00011 | 0.0134±0.00078 | 0.0172±0.0014 | 3.53e-02 | 1.00e-04 | 0/48 | 0/48 | 0.0106±0.0019 | 0.0464±0.0017 | 2.30e-02 | 0.0361±0.0039 |
| Fisher-Hodge + FR reaction, free sigma-MLP (M2) | 68613 | 0.00402±0.00034 | 0.00572±0.00066 | 0.0927±0.0084 | 0.303±0.034 | 5.52e-01 | 1.00e-04 | 0/48 | 0/48 | 0.438±0.046 | 0.141±0.012 | 6.76e-01 | 0.437±0.046 |
| Fisher-Hodge (M), round-1 reaction, trained on S3b (super-saturated ICs) | 51716 | 0.0023±0.00012 | 0.00307±7e-05 | 0.00311±0.00012 | 0.00202±0.00012 | 1.03e-02 | 1.00e-04 | 0/32 | 0/32 | 0.00152±0.00039 | 0.0124±0.00056 | 3.03e-03 | 0.00485±0.00037 |
| M2 sigma-MLP reaction, trained on S3b | 68613 | 0.00285±0.00053 | 0.00393±0.00068 | 0.00297±0.00021 | 0.00289±0.00073 | 1.92e-02 | 1.00e-04 | 0/48 | 0/48 | 0.00385±0.0014 | 0.0145±0.0036 | 6.01e-03 | 0.0049±0.00087 |
| M2' reservoir-edge reaction, trained on S3b | 68357 | 0.00192±5e-05 | 0.00255±8.8e-05 | 0.00257±0.00022 | 0.00271±0.00015 | 1.04e-02 | 1.00e-04 | 0/48 | 0/48 | 0.00272±0.00044 | 0.0129±0.0017 | 3.51e-03 | 0.0035±0.00044 |
| Euclid GF (B2) | 51716 | 0.0118±0.0033 | 0.0177±0.0033 | 0.319±0.12 | 3.61±1.9 | 1.24e+01 | -9.60e-04 | 29/48 | 0/48 | n/a | 0.298±0.041 | 1.28e+01 | n/a |
| GNN-ODE (B1, 4k it) | 138369 | 0.467±0.008 | 0.498±0.0074 | 0.855±0.0044 | 0.898±0.0027 | 2.38e+00 | 1.00e-04 | 0/48 | 0/48 | n/a | 1±0 | -5.31e-02 | n/a |

### S3 - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | front-speed err | max overshoot | Linf sat. err@10T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 51716 | 0.00421±0.00031 | 0.00737±0.00029 | 0.0617±0.021 | 0.161±0.11 | 4.30e-01 | 1.00e-04 | 0/48 | 0/48 | n/a | 0.0786±0.0057 | 5.36e-01 | n/a |
| Fisher-Hodge + reservoir-edge WFR reaction (M2') | 68357 | 0.00481±0.00015 | 0.00898±0.0005 | 0.0137±0.00063 | 0.0158±0.0015 | 2.73e-02 | 1.00e-04 | 0/48 | 0/48 | 0.00838±0.0033 | 0.0503±0.0065 | 2.58e-02 | 0.0349±0.005 |
| Fisher-Hodge + FR reaction, free sigma-MLP (M2) | 68613 | 0.00759±0.00049 | 0.0132±0.00087 | 0.103±0.0082 | 0.284±0.031 | 5.22e-01 | 1.00e-04 | 0/48 | 0/48 | 0.457±0.045 | 0.119±0.0085 | 6.86e-01 | 0.456±0.045 |
| Fisher-Hodge (M), round-1 reaction, trained on S3b (super-saturated ICs) | 51716 | 0.00326±0.00012 | 0.00505±0.00013 | 0.00423±0.00036 | 0.00439±0.0003 | 1.30e-02 | 1.00e-04 | 0/32 | 0/32 | 0.000618±0.00017 | 0.0237±0.0016 | 2.07e-03 | 0.0117±0.00041 |
| M2 sigma-MLP reaction, trained on S3b | 68613 | 0.00554±0.00031 | 0.0106±0.00042 | 0.00779±0.00097 | 0.00263±0.00077 | 2.67e-02 | 1.00e-04 | 0/48 | 0/48 | 0.00383±0.00059 | 0.0357±0.0015 | 7.51e-03 | 0.00716±0.0022 |
| M2' reservoir-edge reaction, trained on S3b | 68357 | 0.00392±1.6e-05 | 0.00634±0.00012 | 0.00542±0.00045 | 0.00194±0.00014 | 1.80e-02 | 1.00e-04 | 0/48 | 0/48 | 0.00242±0.00029 | 0.0337±0.0021 | 2.86e-03 | 0.00329±0.00026 |
| Euclid GF (B2) | 51716 | 0.0148±0.0022 | 0.0315±0.0044 | 0.362±0.12 | 3.81±1.9 | 1.06e+01 | -7.41e-04 | 48/48 | 0/48 | n/a | 0.201±0.064 | 1.11e+01 | n/a |
| GNN-ODE (B1, 4k it) | 138369 | 0.467±0.017 | 1.02±0.69 | 1.68±0.95 | 2.43±1.6 | 3.31e+00 | -2.54e+01 | 32/48 | 0/48 | n/a | 1±0 | -3.39e-02 | n/a |

### S3 - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | front-speed err | max overshoot | Linf sat. err@10T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 51716 | 0.00151±6.3e-05 | 0.00218±0.00048 | 0.0555±0.02 | 0.162±0.11 | 4.32e-01 | 1.00e-04 | 0/48 | 0/48 | n/a | 0.0564±0.0048 | 5.02e-01 | n/a |
| Fisher-Hodge + reservoir-edge WFR reaction (M2') | 68357 | 0.00156±4.6e-05 | 0.00321±0.00033 | 0.0108±0.00067 | 0.0134±0.0011 | 3.06e-02 | 1.00e-04 | 0/48 | 0/48 | 0.00919±0.0014 | 0.0287±0.0031 | 2.09e-02 | 0.0287±0.0025 |
| Fisher-Hodge + FR reaction, free sigma-MLP (M2) | 68613 | 0.00321±0.00021 | 0.00368±0.00023 | 0.0999±0.0082 | 0.312±0.034 | 5.43e-01 | 1.00e-04 | 0/48 | 0/48 | 0.427±0.044 | 0.0835±0.011 | 6.57e-01 | 0.426±0.044 |
| Fisher-Hodge (M), round-1 reaction, trained on S3b (super-saturated ICs) | 51716 | 0.0017±8.2e-06 | 0.0021±0.00013 | 0.00229±1.2e-05 | 0.0018±0.00028 | 5.56e-03 | 1.00e-04 | 0/32 | 0/32 | 0.00149±0.00046 | 0.0129±0.00044 | 3.17e-03 | 0.00406±0.00041 |
| M2 sigma-MLP reaction, trained on S3b | 68613 | 0.00193±0.0005 | 0.00356±0.00068 | 0.00265±0.00032 | 0.003±0.00076 | 7.41e-03 | 1.00e-04 | 0/48 | 0/48 | 0.00385±0.0015 | 0.0136±0.0019 | 6.10e-03 | 0.00496±0.001 |
| M2' reservoir-edge reaction, trained on S3b | 68357 | 0.00132±1.9e-05 | 0.00174±5.9e-05 | 0.00179±0.00024 | 0.00273±0.00014 | 4.67e-03 | 1.00e-04 | 0/48 | 0/48 | 0.00314±0.00053 | 0.0114±0.0027 | 3.32e-03 | 0.00354±0.0004 |
| Euclid GF (B2) | 51716 | 0.00945±0.0021 | 0.00861±0.002 | 0.297±0.1 | 3.68±1.9 | 1.21e+01 | -4.77e-04 | 29/48 | 0/48 | n/a | 0.185±0.023 | 1.24e+01 | n/a |
| GNN-ODE (B1, 4k it) | 138369 | 0.381±0.0054 | 0.332±0.01 | 0.859±0.0037 | 0.898±0.003 | 9.12e-01 | 1.00e-04 | 0/48 | 0/48 | n/a | 1±0 | -3.66e-01 | n/a |

### S3 - cost

| model | params | iters | final train loss | train time (s) | rollout 16x200 steps (s) |
|---|---|---|---|---|---|
| Fisher-Hodge (M) | 51716 | 4000 | 1.90e-06 | 5382 | 21.1 |
| Fisher-Hodge + reservoir-edge WFR reaction (M2') | 68357 | 4000 | 2.07e-06 | 7773 | 35.2 |
| Fisher-Hodge + FR reaction, free sigma-MLP (M2) | 68613 | 4000 | 2.64e-06 | 6225 | 33.9 |
| Fisher-Hodge (M), round-1 reaction, trained on S3b (super-saturated ICs) | 51716 | 4000 | 1.62e-06 | 13558 | 39.6 |
| M2 sigma-MLP reaction, trained on S3b | 68613 | 4000 | 1.70e-06 | 14579 | 42.1 |
| M2' reservoir-edge reaction, trained on S3b | 68357 | 4000 | 1.54e-06 | 12054 | 36.7 |
| Euclid GF (B2) | 51716 | 4000 | 1.69e-05 | 1663 | 5.3 |
| GNN-ODE (B1, 4k it) | 138369 | 4000 | 2.89e-02 | 6615 | 14.5 |

## 2. Misspecified physics (reference dynamics outside the Fisher-Hodge hypothesis class)

### S4 - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.044 | 0.0642 | 0.0799 | 0.08 | 2.97e-07 | 4.60e-04 | 0/16 | 0/16 | 0.116 | 1.00 |
| Euclid GF (B2) | 34691 | 0.119±0.0013 | 0.174±0.0041 | 0.374±0.0089 | 0.678±0.029 | 1.09e+00 | -5.52e-01 | 47/48 | 0/48 | 0.463±0.032 | 1.00 |

### S4 - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.114 | 0.17 | 0.219 | 0.24 | 2.02e-06 | 8.02e-05 | 0/16 | 0/16 | 0.343 | 0.94 |
| Euclid GF (B2) | 34691 | 0.358±0.0051 | 0.478±0.0092 | 0.595±0.00057 | 0.809±0.01 | 1.15e+00 | -6.45e+01 | 48/48 | 15/48 | 0.608±0.0086 | 0.69 |

### S4 - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.0381 | 0.0596 | 0.0713 | 0.0734 | 1.80e-07 | 4.66e-04 | 0/16 | 0/16 | 0.0993 | 1.00 |
| Euclid GF (B2) | 34691 | 0.107±0.0012 | 0.164±0.0032 | 0.287±0.011 | 0.392±0.011 | 1.28e+00 | -2.83e-01 | 48/48 | 0/48 | 0.434±0.022 | 1.00 |

### S4 - cost

| model | params | iters | final train loss | train time (s) | rollout 16x200 steps (s) |
|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 4000 | 3.89e-04 | 11393 | 39.2 |
| Euclid GF (B2) | 34691 | 4000 | 5.95e-03 | 2527 | 9.2 |

### S5 - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.0154 | 0.0158 | 0.0183 | 0.0196 | 1.99e-07 | 6.50e-05 | 0/16 | 0/16 | 0.0236 | 1.00 |
| Euclid GF (B2) | 34691 | 0.0418±0.0004 | 0.055±0.0026 | 0.0736±0.0044 | 0.0959±0.0053 | 9.52e-01 | -2.73e-02 | 39/48 | 0/48 | 0.0797±0.0025 | 0.96 |

### S5 - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.0353 | 0.0285 | 0.0247 | 0.0218 | 1.70e-07 | 1.83e-05 | 0/16 | 0/16 | 0.0241 | 1.00 |
| Euclid GF (B2) | 34691 | 0.0817±0.0027 | 0.0867±0.0029 | 0.127±0.0031 | 0.195±0.0058 | 6.25e-01 | -3.06e+02 | 48/48 | 3/48 | 0.069±0.0046 | 0.73 |

### S5 - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 0.0153 | 0.0159 | 0.0149 | 0.0157 | 1.42e-07 | 2.53e-04 | 0/16 | 0/16 | 0.0204 | 1.00 |
| Euclid GF (B2) | 34691 | 0.0398±0.0011 | 0.0631±0.0031 | 0.095±0.0051 | 0.162±0.008 | 4.07e-01 | -5.79e-02 | 40/48 | 0/48 | 0.061±0.0045 | 1.00 |

### S5 - cost

| model | params | iters | final train loss | train time (s) | rollout 16x200 steps (s) |
|---|---|---|---|---|---|
| Fisher-Hodge (M) | 34563 | 4000 | 8.54e-05 | 4985 | 29.2 |
| Euclid GF (B2) | 34691 | 4000 | 5.06e-04 | 2228 | 8.0 |

### S6 - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Euclid GF (B2) | 34691 | 0.119±0.0018 | 0.149±0.0062 | 0.197±0.0052 | 0.248±0.011 | 5.07e-01 | -2.34e-01 | 32/32 | 0/32 | 0.207±0.0057 | 1.00 |

### S6 - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Euclid GF (B2) | 34691 | 0.38±0.0062 | 0.403±0.0018 | 0.472±0.01 | 0.575±0.0055 | 7.34e-01 | -1.25e+01 | 30/32 | 2/32 | 0.388±0.0026 | 0.81 |

### S6 - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Euclid GF (B2) | 34691 | 0.12±0.0012 | 0.123±0.00052 | 0.161±0.0046 | 0.205±0.0078 | 6.66e-01 | -4.93e-01 | 30/32 | 0/32 | 0.115±0.014 | 1.00 |

### S6 - cost

| model | params | iters | final train loss | train time (s) | rollout 16x200 steps (s) |
|---|---|---|---|---|---|
| Euclid GF (B2) | 34691 | 4000 | 5.15e-03 | 3159 | 8.7 |

## 3. Noisy training observations (multiplicative Gaussian noise on training snapshots only; test data clean)

### S1 noise 1% - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 34563 | 0.00304±8.8e-05 | 0.00314±5.5e-05 | 0.00342±0.00012 | 0.00343±3.2e-05 | 3.02e-07 | 3.76e-04 | 0/48 | 0/48 | n/a | 1.00 |
| Euclid GF (B2) [clean, round 1] | 34691 | 0.0641±0.004 | 0.0755±0.0043 | 0.133±0.0038 | 0.254±0.0059 | 5.67e-01 | -2.11e+02 | 46/48 | 2/48 | n/a | 0.96 |

### S1 noise 1% - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 34563 | 0.0149±0.00061 | 0.0162±0.00055 | 0.0171±0.00033 | 0.0169±0.00055 | 4.02e-07 | 5.31e-05 | 0/48 | 0/48 | n/a | 0.90 |
| Euclid GF (B2) [clean, round 1] | 34691 | 0.221±0.0039 | 0.213±0.0047 | 0.196±0.0014 | 0.27±0.0063 | 6.67e-01 | -2.47e+01 | 46/48 | 5/48 | n/a | 0.88 |

### S1 noise 1% - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 34563 | 0.00303±0.0001 | 0.00266±0.00014 | 0.00234±0.00012 | 0.00256±7.2e-05 | 2.33e-07 | 5.56e-04 | 0/48 | 0/48 | n/a | 1.00 |
| Euclid GF (B2) [clean, round 1] | 34691 | 0.0672±0.0064 | 0.0786±0.0057 | 0.115±0.0028 | 0.147±0.0023 | 6.93e-01 | -2.98e-01 | 48/48 | 0/48 | n/a | 1.00 |

### S1 noise 1% - cost

| model | params | iters | final train loss | train time (s) | rollout 16x200 steps (s) |
|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 34563 | 4000 | 7.54e-06 | 4502 | 27.1 |
| Euclid GF (B2) [clean, round 1] | 34691 | 4000 | 1.31e-03 | 1190 | 4.6 |

### S1 noise 5% - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 34563 | 0.00304±8.8e-05 | 0.00314±5.5e-05 | 0.00342±0.00012 | 0.00343±3.2e-05 | 3.02e-07 | 3.76e-04 | 0/48 | 0/48 | n/a | 1.00 |
| Euclid GF (B2) [clean, round 1] | 34691 | 0.0641±0.004 | 0.0755±0.0043 | 0.133±0.0038 | 0.254±0.0059 | 5.67e-01 | -2.11e+02 | 46/48 | 2/48 | n/a | 0.96 |

### S1 noise 5% - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 34563 | 0.0149±0.00061 | 0.0162±0.00055 | 0.0171±0.00033 | 0.0169±0.00055 | 4.02e-07 | 5.31e-05 | 0/48 | 0/48 | n/a | 0.90 |
| Euclid GF (B2) [clean, round 1] | 34691 | 0.221±0.0039 | 0.213±0.0047 | 0.196±0.0014 | 0.27±0.0063 | 6.67e-01 | -2.47e+01 | 46/48 | 5/48 | n/a | 0.88 |

### S1 noise 5% - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | learned-Lyap monotone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 34563 | 0.00303±0.0001 | 0.00266±0.00014 | 0.00234±0.00012 | 0.00256±7.2e-05 | 2.33e-07 | 5.56e-04 | 0/48 | 0/48 | n/a | 1.00 |
| Euclid GF (B2) [clean, round 1] | 34691 | 0.0672±0.0064 | 0.0786±0.0057 | 0.115±0.0028 | 0.147±0.0023 | 6.93e-01 | -2.98e-01 | 48/48 | 0/48 | n/a | 1.00 |

### S1 noise 5% - cost

| model | params | iters | final train loss | train time (s) | rollout 16x200 steps (s) |
|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 34563 | 4000 | 7.54e-06 | 4502 | 27.1 |
| Euclid GF (B2) [clean, round 1] | 34691 | 4000 | 1.31e-03 | 1190 | 4.6 |

### S2 noise 5% - ID (new init, same dist.)

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | curl(J) err @10T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 52100 | 0.00371±6.6e-05 | 0.00355±4.8e-05 | 0.00318±0.00016 | 0.00325±0.00016 | 3.42e-07 | 4.68e-04 | 0/48 | 0/48 | n/a | 0.0125±0.00077 |
| Euclid GF (B2) [clean, round 1] | 52484 | 0.0231±0.00051 | 0.0254±0.00031 | 0.0275±0.0024 | 0.0296±0.002 | 4.61e-02 | -3.09e-02 | 32/48 | 0/48 | n/a | 0.621±0.14 |

### S2 noise 5% - OOD potential / field

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | curl(J) err @10T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 52100 | 0.0147±0.00055 | 0.0154±0.0011 | 0.0139±0.001 | 0.014±0.00078 | 3.39e-07 | 1.82e-05 | 0/48 | 0/48 | n/a | 0.0331±0.0027 |
| Euclid GF (B2) [clean, round 1] | 52484 | 0.0511±0.0034 | 0.0598±0.00089 | 0.0605±0.0035 | 0.0689±0.0074 | 8.18e-02 | -7.17e-01 | 48/48 | 0/48 | n/a | 0.665±0.15 |

### S2 noise 5% - OOD initial condition

| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T | curl(J) err @10T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 52100 | 0.00307±0.00016 | 0.00331±0.00013 | 0.00326±0.00012 | 0.00324±0.00011 | 3.74e-07 | 4.72e-04 | 0/48 | 0/48 | n/a | 0.0138±0.00073 |
| Euclid GF (B2) [clean, round 1] | 52484 | 0.0195±0.0015 | 0.0236±0.0019 | 0.0313±0.0025 | 0.0309±0.0012 | 9.39e-02 | -5.03e-02 | 33/48 | 0/48 | n/a | 0.708±0.17 |

### S2 noise 5% - cost

| model | params | iters | final train loss | train time (s) | rollout 16x200 steps (s) |
|---|---|---|---|---|---|
| Fisher-Hodge (M) [clean, round 1] | 52100 | 4000 | 5.03e-06 | 6154 | 40.3 |
| Euclid GF (B2) [clean, round 1] | 52484 | 4000 | 1.12e-04 | 1689 | 5.5 |
