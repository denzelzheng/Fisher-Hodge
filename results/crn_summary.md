# HyperME on benchmark C1 (chemostat-driven spatial phosphorylation cycle, 8x8 periodic, S=3)

Entries are `seed0 / seed1`. relL2 on the dynamic species (X, Xp), median over 16 trajectories; T = 20 steps (training window), 10T = 200 steps. ID: dG in [1,3], U std 0.7 (|k|<=2); OOD-dG: dG in [3,4]; OOD-U: U std 1.2 with |k|<=3. True local cycle affinity = dG + log(k2/k2') = dG + 3.912.

## Accuracy

| model | params | ID T | ID 10T | OOD-dG T | OOD-dG 10T | OOD-U T | OOD-U 10T | Gillespie: vs SSA / vs ODE (10T) | train s |
|---|---|---|---|---|---|---|---|---|---|
| HyperME | 14150 / 14150 | 0.012 / 0.0122 | 0.0232 / 0.0216 | 0.0241 / 0.0267 | 0.0419 / 0.026 | 0.0311 / 0.0343 | 0.057 / 0.0446 | 0.077 / 0.076 ; 0.023 / 0.020 | 191 / 196 |
| B3 HyperME (A=0) | 14150 / 14150 | 0.0158 / 0.0167 | 0.0237 / 0.0211 | 0.0306 / 0.0293 | 0.0546 / 0.0374 | 0.0449 / 0.0421 | 0.0706 / 0.0682 | 0.077 / 0.076 ; 0.023 / 0.020 | 236 / 216 |
| B1 CRNN | 4866 / 4866 | 0.0109 / 0.0115 | 0.0213 / 0.0247 | 0.0512 / 0.0691 | 0.0874 / 0.133 | 0.0256 / 0.029 | 0.0506 / 0.0602 | 0.079 / 0.083 ; 0.025 / 0.034 | 287 / 290 |
| B2 GNN-ODE | 138563 / 138563 | 0.0102 / 0.0109 | 0.107 / 0.113 | 0.0325 / 0.0267 | 0.182 / 0.151 | 0.103 / 0.127 | 0.735 / 0.218 | 0.128 / 0.127 ; 0.103 / 0.102 | 1594 / 1624 |

Gillespie (Omega=200) noise floor, ODE vs SSA at 10T: 0.075.

## Structure / physics

| model | max cons. err (kerS^T; E; X+Xp) over all test sets | traj with c<0 (ID/OOD-dG/OOD-U) | min c | sigma(t) rel err ID / OOD-dG | J(10T) corr all (rx; tr) ID | J(10T) rel err ID | affinity MAE ID / OOD-dG | A_cyc rel err ID |
|---|---|---|---|---|---|---|---|---|
| HyperME | 1.0e-15; 0.0e+00; 1.2e-15 / 1.1e-15; 0.0e+00; 8.2e-16 | 0,0,0 / 0,0,0 | 4.69e-03 / 4.75e-03 | 0.692 / 0.718 ; 0.604 / 0.715 | 0.978 / 0.984 (0.955 / 0.969; 0.740 / 0.743) | 0.331 / 0.273 | 3.6 / 3.9 ; 4.06 / 4.9 | 0.605 / 0.655 |
| B3 HyperME (A=0) | 1.0e-15; 0.0e+00; 8.2e-16 / 9.7e-16; 0.0e+00; 1.0e-15 | 0,0,0 / 0,0,0 | 3.36e-03 / 3.12e-03 | 0.975 / 0.976 ; 0.943 / 0.941 | 0.024 / 0.018 (0.033 / 0.056; 0.066 / 0.072) | 1 / 1 | 5.96 / 5.96 ; 7.48 / 7.48 | 1 / 1 |
| B1 CRNN | 9.7e-16; 0.0e+00; 8.2e-16 / 8.7e-16; 0.0e+00; 8.2e-16 | 0,0,0 / 0,0,0 | 2.11e-03 / 1.97e-03 | 0.425 / 0.44 ; 0.446 / 0.383 | 0.997 / 0.997 (0.994 / 0.993; 0.952 / 0.949) | 0.106 / 0.0993 | 3 / 3.01 ; 3.33 / 3.11 | 0.502 / 0.507 |
| B2 GNN-ODE | 4.1e-01; 2.7e-03; 7.6e-01 / 1.6e-01; 5.3e-03; 3.0e-01 | 1,0,16 / 0,0,16 | -2.33e-01 / -7.05e-01 | n/a / n/a ; n/a / n/a | n/a / n/a (n/a / n/a; n/a / n/a) | n/a / n/a | n/a / n/a ; n/a / n/a | n/a / n/a |

B3 has A = 0 by construction (affinity = 0, MAE = true affinity). B2 has no fluxes/EPR (n/a). J correlation is over all 384 hyperedges of each test trajectory at t = 10T (rx = R1/R2 edges, tr = transport edges).

## Affinity identifiability (learned site-mean <z_v,A> vs true dG + log(k2/k2'), pooled over the 3 test sets and both seeds)

C1: k2' = 0.02 (log(k2/k2') = 3.91, reverse R2 flux ~1% of forward); control C1b: k2' = 0.2 (log = 1.61), same everything else.

| data | model | slope | intercept | Pearson r | MAE | sigma rel err ID | J corr ID | relL2 ID 10T | relL2 OOD-dG 10T |
|---|---|---|---|---|---|---|---|---|---|
| C1 | HyperME | 0.561 | -1.20 | 0.842 | 4.044 | 0.692 / 0.718 | 0.978 / 0.984 | 0.0232 / 0.0216 | 0.0419 / 0.026 |
| C1 | B1 CRNN | 0.876 | -2.28 | 0.993 | 3.077 | 0.425 / 0.44 | 0.997 / 0.997 | 0.0213 / 0.0247 | 0.0874 / 0.133 |
| C1 | B3 HyperME (A=0) | 0.000 | 0.00 | n/a | 6.475 | 0.975 / 0.976 | 0.0244 / 0.0181 | 0.0237 / 0.0211 | 0.0546 / 0.0374 |
| C1b | HyperME | 0.622 | -0.37 | 0.877 | 1.942 | 0.572 / 0.603 | 0.975 / 0.98 | 0.0222 / 0.0207 | 0.0411 / 0.0252 |
| C1b | B1 CRNN | 0.862 | -0.45 | 0.991 | 1.027 | 0.174 / 0.192 | 0.998 / 0.998 | 0.0203 / 0.0248 | 0.0886 / 0.132 |
| C1b | B3 HyperME (A=0) | 0.000 | 0.00 | n/a | 4.173 | 0.945 / 0.948 | 0.0251 / 0.02 | 0.0218 / 0.0198 | 0.0586 / 0.0345 |

## Coarse-graining commutation (8x8 -> 4x4, same model on both graphs; X,Xp rel L2, median, ID test)

| model | block-sum comm. T | block-sum comm. 10T | block-mean comm. T | block-mean comm. 10T | block-mean coarse-rollout vs coarsened truth 10T |
|---|---|---|---|---|---|
| HyperME | 0.276 / 0.273 | 0.139 / 0.134 | 0.186 / 0.184 | 0.0465 / 0.0427 | 0.0461 / 0.0417 |
| B3 HyperME (A=0) | 0.272 / 0.31 | 0.126 / 0.185 | 0.185 / 0.187 | 0.0434 / 0.0403 | 0.0474 / 0.0477 |
| B1 CRNN | 0.327 / 0.328 | 0.209 / 0.208 | 0.193 / 0.195 | 0.0506 / 0.0523 | 0.0516 / 0.0542 |
| B2 GNN-ODE | 0.234 / 0.294 | 0.62 / 0.779 | 0.196 / 0.2 | 0.0825 / 0.106 | 0.0809 / 0.074 |
| true physics, naive coarse params | 0.283 | 0.168 | 0.191 | 0.0428 | – |

## Property self-test (hyperme.py --selftest, random-init models, 4 traj x 200 steps)

- A=0,g=0: conservation rel err kerS^T 1.1e-15, E 0.0e+00, X+Xp 6.1e-16; min c 5.10e-02; min sigma 3.37e-04; max dF/step -1.65e-05
- A!=0,g!=0 (random): conservation rel err kerS^T 1.2e-15, E 0.0e+00, X+Xp 5.2e-16; min c 5.10e-02; min sigma 2.21e-01; max dF/step -3.20e-05

## Assertions on trained HyperME rollouts (all 3 test sets x 200 steps)

- hyperme_s0: test_id: cons True, pos True, sigma>=0 True, test_oodG: cons True, pos True, sigma>=0 True, test_oodU: cons True, pos True, sigma>=0 True
- hyperme_s1: test_id: cons True, pos True, sigma>=0 True, test_oodG: cons True, pos True, sigma>=0 True, test_oodU: cons True, pos True, sigma>=0 True
- hyperme_rev_s0: test_id: cons True, pos True, sigma>=0 True, test_oodG: cons True, pos True, sigma>=0 True, test_oodU: cons True, pos True, sigma>=0 True
- hyperme_rev_s1: test_id: cons True, pos True, sigma>=0 True, test_oodG: cons True, pos True, sigma>=0 True, test_oodU: cons True, pos True, sigma>=0 True

## Notes (honest reading)

- Trajectories: HyperME, B3 and B1 are all ~2% at 10T ID (B1 and HyperME share the exact stoichiometry; positivity-free fixed-step B1 still never went negative here). HyperME is clearly better on OOD-dG (2.6-4.2% vs 8.7-13% for B1). On OOD-U it is about the same as B1 (4.5-5.7% vs 5.1-6.0%). B2 GNN-ODE fits the training window (1% at T) but drifts: 11% ID / 15-18% OOD-dG / 22-74% OOD-U at 10T, conservation error 10-76%, negative concentrations on all OOD-U trajectories.
- Conservation (every ker S^T invariant, E per site, X+Xp total) is at machine precision (<=1.5e-15) for HyperME/B3/B1; positivity and sigma>=0 hold on every HyperME rollout; F monotone for A=0, g=0 (self-test).
- Thermodynamics is NOT identified by trajectory fit: B3 (A=0, detailed balance) has the same trajectory error as HyperME but its sigma(t) relaxes to 0 (true NESS sigma ~300) and its 10T flux correlation is 0.02. Fluxes are determined by the dynamics only modulo ker S.
- Affinity: HyperME learns roughly the R1 part (dG) and misses log(k2/k2') almost completely (learned ~ dG, slope 0.37-0.81 by seed, spurious site-to-site spread ~0.2). B1 learns a slope ~1 with an offset of about -3. The R2 reverse flux is ~1% of the forward flux, so log(k2/k2') is hardly visible in the data. Control C1b (k2'=0.2) halves the MAE for both (HyperME 4.0->1.9, B1 3.1->1.0). B1 stays better than HyperME on affinity, sigma and flux correlation in both settings.
- Coarse commutation (8x8->4x4) is dominated by the coarse parameters being wrong: the true physics itself gives 17% (block-sum) / 4.3% (block-mean) at 10T with the same per-edge parameters. HyperME 13-14% / 4.3-4.6%, B1 21% / 5.1-5.2%, B2 62-78% / 8-11%.
- The NESS flux map (ness_flux_map.png) shows HyperME's local EPR well below the truth and a spurious vertical X-up / Xp-down transport circulation (a ker S mode that the dynamics cannot see).
- Engineering: on this tiny system (192 states, 384 hyperedges, float64) CPU was faster than the shared GPU, so the HyperME/B1/B3 runs used CPU (~3-5 min each). B2 ran on GPU 1 (~27 min each). Total GPU use was about 1 h.

## E5: cycle observables and identifiability

Training loss = trajectory loss + w * relative squared error of the observable (w = 0.02, so both terms are the same size at initialisation: obs ~1.4 x 0.02 vs trajectory ~2e-2). Observable truth from the true rates on the training frames. Entries `seed0 / seed1`. "R2 part" = mean(learned affinity - dG); true value log(k2/k2') = 3.91. Observable test error = rel. L2 of the model's r(t)=sum_v J_R1 along its own 10T rollout (test_id).

| model | observable | relL2 ID 10T | relL2 OOD-dG 10T | relL2 OOD-U 10T | sigma rel err ID | J corr ID | J rel err ID | affinity MAE | slope | R2 part | A_cyc rel err | r(t) test err | final obs loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HyperME | no observable | 0.0232 / 0.0216 | 0.0419 / 0.026 | 0.057 / 0.0446 | 0.692 / 0.718 | 0.978 / 0.984 | 0.331 / 0.273 | 3.8 / 4.29 | 0.74 / 0.38 | 0.11 / -0.37 | 0.605 / 0.655 | 0.239 / 0.218 | n/a / n/a |
| HyperME | (a) global r(t)=sum J_R1 | 0.0239 / 0.0225 | 0.0245 / 0.0478 | 0.0546 / 0.0502 | 0.479 / 0.524 | 0.983 / 0.982 | 0.165 / 0.169 | 2.96 / 3.36 | 0.76 / 0.53 | 0.96 / 0.56 | 0.479 / 0.525 | 0.0439 / 0.047 | 1.05e-03 / 1.13e-03 |
| HyperME | (b) J_R1 at 4 sites | 0.0195 / 0.0216 | 0.0268 / 0.0447 | 0.0666 / 0.0542 | 0.329 / 0.386 | 1.000 / 0.999 | 0.0337 / 0.0525 | 2.46 / 2.82 | 0.80 / 0.67 | 1.46 / 1.09 | 0.403 / 0.451 | 0.031 / 0.0341 | 1.81e-03 / 1.95e-03 |
| HyperME | (c) sum J_R2 | 0.02 / 0.0208 | 0.0241 / 0.0466 | 0.0516 / 0.06 | 0.444 / 0.47 | 0.990 / 0.990 | 0.135 / 0.136 | 2.55 / 2.86 | 0.85 / 0.65 | 1.36 / 1.06 | 0.425 / 0.458 | 0.0222 / 0.0221 | 9.63e-04 / 9.04e-04 |
| B3 HyperME (A=0) | no observable | 0.0237 / 0.0211 | 0.0546 / 0.0374 | 0.0706 / 0.0682 | 0.975 / 0.976 | 0.024 / 0.018 | 1 / 1 | 6.48 / 6.48 | 0.00 / 0.00 | -2.56 / -2.56 | 1 / 1 | 0.922 / 0.922 | n/a / n/a |
| B3 HyperME (A=0) | (a) global r(t)=sum J_R1 | 0.35 / 0.364 | 0.19 / 0.187 | 0.352 / 0.41 | 0.953 / 0.953 | 0.014 / 0.023 | 1 / 1 | 6.48 / 6.48 | 0.00 / 0.00 | -2.56 / -2.56 | 1 / 1 | 0.889 / 0.889 | 3.02e-01 / 2.98e-01 |
| B3 HyperME (A=0) | (b) J_R1 at 4 sites | 0.34 / 0.36 | 0.19 / 0.162 | 0.319 / 0.423 | 0.953 / 0.954 | 0.014 / 0.025 | 1 / 1 | 6.48 / 6.48 | 0.00 / 0.00 | -2.56 / -2.56 | 1 / 1 | 0.891 / 0.891 | 2.94e-01 / 2.88e-01 |
| B3 HyperME (A=0) | (c) sum J_R2 | 0.291 / – | 0.196 / – | 0.233 / – | 0.983 / – | -0.010 / – | 1 / – | 6.48 / – | 0.00 / – | -2.56 / – | 1 / – | 1 / – | 6.48e-01 / – |
| B1 CRNN | no observable | 0.0213 / 0.0247 | 0.0874 / 0.133 | 0.0506 / 0.0602 | 0.425 / 0.44 | 0.997 / 0.997 | 0.106 / 0.0993 | 3.11 / 3.05 | 0.82 / 0.94 | 0.80 / 0.87 | 0.502 / 0.507 | 0.0888 / 0.0826 | n/a / n/a |
| B1 CRNN | (a) global r(t)=sum J_R1 | 0.0243 / 0.0262 | 0.143 / 0.181 | 0.0474 / 0.0581 | 0.503 / 0.5 | 0.998 / 0.997 | 0.0602 / 0.0725 | 3.24 / 3.16 | 0.82 / 0.91 | 0.67 / 0.75 | 0.53 / 0.526 | 0.0419 / 0.0485 | 7.64e-04 / 9.59e-04 |
| B1 CRNN | (b) J_R1 at 4 sites | 0.0282 / 0.0291 | 0.152 / 0.182 | 0.055 / 0.0593 | 0.499 / 0.498 | 0.999 / 0.999 | 0.0531 / 0.0594 | 3.19 / 3.18 | 0.89 / 0.91 | 0.72 / 0.73 | 0.528 / 0.527 | 0.0227 / 0.0257 | 1.14e-03 / 1.34e-03 |
| B1 CRNN | (c) sum J_R2 | 0.0211 / 0.0253 | 0.142 / 0.171 | 0.0476 / 0.059 | 0.474 / 0.468 | 0.997 / 0.998 | 0.0767 / 0.0697 | 2.9 / 2.83 | 0.90 / 0.91 | 1.01 / 1.08 | 0.482 / 0.472 | 0.0214 / 0.0177 | 1.64e-03 / 1.26e-03 |

### Linear-algebra check (obs_hyper.py, true C1 hypergraph)

- H = 384 hyperedges (64 R1 + 64 R2 + 128 X-transport + 128 Xp-transport), 192 states, rank S = 127, dim ker S^T = 65 (E per site + X+Xp total), **dim ker S = beta_1 = 257 independent cycles** = 64 local R1-R2 loops + 65 X-transport + 65 Xp-transport + 63 mixed transport-reaction cycles.
- Concentration data fix only S J, so 257 flux directions are invisible; fixing J completely needs 257 independent cycle observables.
- Invisible flux dims ker[S; W]: none: 257, global: 256, globalR2: 256, global+globalR2: 256, local4: 253, all 64 local J_R1: 193. (a) and (c) are equivalent: w_R1 - w_R2 lies in im S^T (residual 2.5e-14), so together they still give +1.
- A-level (A = Z beta, linearised at the true NESS of test_id traj 0, k and mu known): identifiable beta directions none: 127, global: 128, globalR2: 128, global+globalR2: 128, local4: 131 of 257, i.e. **(a) adds exactly 1, (b) adds 4**.
- If per-hyperedge k is unknown as well, one snapshot identifies 0 beta directions. Over the whole trajectory (41 frames) it identifies none: 193, global: 209, globalR2: 209, global+globalR2: 209, local4: 197 (the 64 missing directions with no observable match the 64 local loops). Here (b) adds a robust +4. The gain from (a) depends on the rank tolerance: +11 / +16 / +21 at rtol 1e-6 / 1e-9 / 1e-12. The time series r(t) mixes 64 sites, so it is worth more than one number.


### E5 reading

- The linear algebra confirms the counting. beta_1 = dim ker S = 257, and concentrations leave 257 flux directions invisible. Observable (a) removes exactly 1 and (c) is the same direction modulo im S^T, so (a)+(c) also give 1. Observable (b) removes 4. At the A level (snapshot, k known) the identifiable directions go 127 -> 128 with (a).
- HyperME uses the observable. Trajectory error is unchanged (ID 10T 2.0-2.4%). sigma rel. err falls from 0.69-0.72 to 0.48-0.52 (a), 0.33-0.39 (b) and 0.44-0.47 (c). Flux rel. err at 10T falls from 0.27-0.33 to 0.17 (a) and 0.03-0.05 (b). Affinity MAE falls from 4.0 to 3.2 / 2.6 / 2.7.
- The log(k2/k2') part (true 3.91) is recovered only partly: the learned "R2 part" goes from ~0 to 0.6-1.5. Matching the cycle current fixes J ~ k2 Xp but not the tiny reverse rate k2' X. The force on a nearly irreversible edge (log of a ~1% rate) stays weakly identified even when the flux is known. Flux observables identify J; affinities additionally need a measurable reverse flux.
- B1 CRNN also matches r(t) (test err 2-5%) and its 10T flux rel. err improves (0.10 -> 0.05-0.07). Its affinity does not move (MAE 3.1 -> 2.8-3.2), and sigma error gets slightly worse (0.43 -> 0.47-0.50). This is the same irreversible-edge limitation, without the coupling through A in ker S.
- B3 (A = 0) cannot fit the trajectories and the observable at the same time. Its observable loss stalls at 0.29-0.65 (HyperME reaches ~1e-3), and its 10T trajectory error jumps from 2% to 29-36%. Under detailed balance r(t) must relax to 0, while the true r(t) stays at the NESS value (E5_sigma.png, right panel). B3 + (c) seed 1 was stopped at iteration 400: it went stiff (49 substeps per DT, ~1 h projected), and its logged losses at that point show the same conflict (obs 0.76).
