# TC-ME experiments on S2 (Fokker-Planck + divergence-free drift)

All numbers: median over 16 test trajectories, then mean±std over seeds (2 seeds). 2500 training iterations, train.py protocol (rollout loss K in [4,8], Adam 1e-3 cosine, batch 8). 10T = t = 10 (200 steps).

## E1: full training set (64 trajectories)

| model | params | 10T ID | 10T OOD-pot | 10T OOD-init | T ID | max mass err | min rho | traj w/ rho<0 | curl J err @10T (ID / OOD-pot) | J err at true steady rho (ID / OOD-pot) | final train loss | train time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TC-ME | 51595 | 0.00318±0.000122 | 0.0161±0.000338 | 0.00465±4.54e-05 | 0.00362±6.61e-05 | 6.0e-07 | 1.5e-05 | 0/96 | 0.0124±0.000609 / 0.0417±0.00391 | 0.0112±5.03e-05 / 0.0452±0.00392 | 6.38e-06±2.6e-07 | 845±10 |
| Fisher-Hodge | 52100 | 0.0032±0.000161 | 0.0146±0.000157 | 0.00348±0.000177 | 0.00368±1.62e-05 | 3.2e-07 | 2.0e-05 | 0/96 | 0.013±0.00127 / 0.032±0.00104 | 0.0134±0.00145 / 0.0457±0.00234 | 5.64e-06±6.31e-07 | 900±1 |
| Flux-GNN | 138369 | 0.00247±0.000245 | 0.0182±0.00169 | 0.00256±0.0002 | 0.00228±4.15e-05 | 4.2e-07 | 1.4e-05 | 0/96 | 0.0111±0.000329 / 0.0621±0.00546 | 0.00866±0.000379 / 0.0538±0.00141 | 3.74e-06±1.22e-07 | 1820±16 |

## E3: 8 training trajectories

| model | params | 10T ID | 10T OOD-pot | 10T OOD-init | T ID | max mass err | min rho | traj w/ rho<0 | curl J err @10T (ID / OOD-pot) | J err at true steady rho (ID / OOD-pot) | final train loss | train time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TC-ME | 51595 | 0.00529±0.000214 | 0.0272±0.00336 | 0.00451±6.28e-05 | 0.00491±9.09e-05 | 6.1e-07 | 1.5e-05 | 0/96 | 0.0212±0.00162 / 0.0989±0.0167 | 0.021±8.95e-05 / 0.0787±0.00958 | 3.53e-06±5.16e-08 | 850±1 |
| Fisher-Hodge | 52100 | 0.006±0.000774 | 0.0511±0.00156 | 0.00746±0.00243 | 0.0057±0.000681 | 5.5e-07 | 2.1e-05 | 0/96 | 0.015±0.000212 / 0.0652±0.00653 | 0.0171±0.000238 / 0.133±0.0131 | 2.93e-06±6.76e-08 | 906±1 |
| Flux-GNN | 138369 | 0.00426±2.7e-06 | 0.0529±0.00785 | 0.00532±0.00145 | 0.00367±0.000162 | 4.3e-07 | 1.9e-05 | 0/96 | 0.0138±0.000894 / 0.144±0.0161 | 0.0135±0.00105 / 0.122±0.0148 | 2.45e-06±8.97e-08 | 1846±3 |

## E1: thermodynamic readout of the learned TC-ME generator (E1 runs, 2 seeds)

pi = stationary state of the learned Q (fixed point of pi = stat(Q(pi))); reference steady state = 64x64 finite-volume stationary solve (exact linear solve, coarse-grained to 32x32). sigma_model = sum_e J_e f_e (probability units, = h^2 sum_e J f in density units); sigma_ref = sum |J|^2/(D rho) h^2 on the 64x64 grid ("fine") or with the coarse-grained J, rho on 32x32 ("coarse"). excess fraction = sum ell (d phi)^2 / sigma with the ell-weighted Hodge projection (hodge_lib.hodge); "Euclid" = same with the unweighted projection.

| quantity | ID | OOD-pot |
|---|---|---|
| J err at t=10, true rho_end fed in | 0.0112±5.58e-05 | 0.0452±0.00388 |
| J err at t=10, model rho_end | 0.00907±0.000191 | 0.027±0.00168 |
| curl J err at t=10, true rho_end | 0.0134±0.00131 | 0.0428±0.00568 |
| rho_ss (model pi) rel err | 0.00324±0.000123 | 0.0162±0.000214 |
| J_ss (model pi) rel err | 0.00909±0.000169 | 0.027±0.002 |
| curl J_ss rel err | 0.0124±0.000608 | 0.0417±0.00384 |
| sigma_model(pi) / sigma_ref fine | 1.01±0.00107 | 1±0.00313 |
| sigma_model(pi) / sigma_ref coarse | 1.02±0.00124 | 1.02±0.0039 |
| sigma_model(rho_end) / sigma_ref(J_end) coarse, t=10 | 1.02±0.000851 | 1.03±0.0018 |
| sigma_model(pi) absolute | 1.57±0.00043 | 6.41±0.0345 |
| sigma_ref fine absolute | 1.56±0 | 6.49±0 |
| excess frac at pi (ell-weighted) | 6.83e-16±2.49e-16 | 1.16e-16±1.45e-17 |
| excess frac at pi (Euclid proj.) | 0.0821±5.81e-05 | 0.175±0.000314 |
| excess frac at true rho_ss (ell) | 3.93e-05±5.09e-07 | 0.000561±3.69e-05 |
| excess frac at true rho_ss (Euclid) | 0.0817±0.000132 | 0.176±0.000271 |
| excess frac at true rho(t=10) (ell) | 3.86e-05±6.94e-07 | 0.00057±3.59e-05 |
| |b| / rms(A) (harmonic part of A) | 0.000418±9.46e-05 | 0.000188±4.28e-05 |
| A vs dpsi rel err (edge) | 0.0146±0.000695 | 0.0158±0.000215 |
| curl A vs curl dpsi rel err (face cycle affinities) | 0.0167±0.000281 | 0.0244±0.002 |
| regression slope A ~ dpsi | 1.01±0.000577 | 1.01±0.000857 |
| learned harmonic velocity v = b/h (units D/length; truth 0) | (-5.57e-04, -2.01e-03); (-1.40e-03, -3.19e-03) | (-5.57e-04, -2.01e-03); (-1.40e-03, -3.19e-03) |
| learned alpha / gam | 0.995 / 0.028; 0.995 / 0.025 | |

## E2: coarse-graining consistency (32x32 -> 16x16, same trained weights)

(a) = roll out on 32x32 then lump; (b) = lump inputs, roll out on 16x16. Errors vs the lumped reference. comm = ||rho_a - rho_b|| / ||rho_a||. SQRA oracle = untrained TC-ME with k = D/h^2, A = dpsi, phi = U/D (the exact-drift discretisation): its comm error is the floor set by discretisation alone.

### test_id

| model | err 32x32 @10T | err (a) @10T | err (b) @10T | comm @T | comm @2T | comm @5T | comm @10T | mass err (b) | min rho (b) |
|---|---|---|---|---|---|---|---|---|---|
| tcme | 0.00318±0.000122 | 0.00309±0.000124 | 0.0111±4.07e-06 | 0.0141±4.54e-05 | 0.0108±5.45e-05 | 0.00931±2.75e-05 | 0.00909±6.76e-06 | 1.4e-06 | 5.1e-04 |
| fisher_hodge | 0.0032±0.000161 | 0.00304±0.000177 | 0.0105±0.000299 | 0.0156±8.59e-05 | 0.0103±6.81e-05 | 0.0086±5.9e-05 | 0.00874±1.18e-05 | 1.1e-06 | 5.3e-04 |
| flux_gnn | 0.00247±0.000245 | 0.00239±0.000263 | 0.0163±0.000973 | 0.0157±0.00029 | 0.0115±0.000486 | 0.0135±0.00086 | 0.0144±0.000871 | 8.0e-07 | 5.3e-04 |
| sqra_oracle | 0.00412 | 0.00393 | 0.0125 | 0.0131 | 0.0104 | 0.00902 | 0.00935 | 1.3e-06 | 5.1e-04 |

### test_oodpot

| model | err 32x32 @10T | err (a) @10T | err (b) @10T | comm @T | comm @2T | comm @5T | comm @10T | mass err (b) | min rho (b) |
|---|---|---|---|---|---|---|---|---|---|
| tcme | 0.0161±0.000338 | 0.0154±6.48e-05 | 0.0772±0.000306 | 0.0478±5.27e-05 | 0.0537±0.000201 | 0.0592±0.000816 | 0.0579±0.000275 | 1.3e-06 | 2.6e-05 |
| fisher_hodge | 0.0146±0.000157 | 0.0134±0.000482 | 0.053±0.000495 | 0.0444±0.000201 | 0.0441±0.00053 | 0.0442±0.000467 | 0.0443±0.000521 | 9.1e-07 | 3.9e-05 |
| flux_gnn | 0.0182±0.00169 | 0.0167±0.0017 | 0.128±0.00627 | 0.105±0.0047 | 0.114±0.00633 | 0.118±0.00768 | 0.119±0.00785 | 1.1e-06 | 9.2e-05 |
| sqra_oracle | 0.0233 | 0.0202 | 0.0755 | 0.0576 | 0.0604 | 0.0596 | 0.0566 | 1.7e-06 | 2.5e-05 |

![E2](E2_commutation.png)
