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
