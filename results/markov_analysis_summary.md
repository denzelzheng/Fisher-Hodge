# Onsager-Fisher Hodge decomposition of Markov-chain entropy production: numerical check

Code: ~/fisher_hodge_analysis/{hodge_lib,e1,e2,e3,report}.py (numpy/scipy, CPU). Data: e1_summary.json, e1_curves.json, e2.json, e3.json. Figures: e1_curves.png, e1_euclid_failure.png, e3_estimation.png.
Model: Q_ij = exp(-B_ij + (U_i-U_j)/2 + eps*A_ij/2), B~U(0,2), U~N(0,1), A = random unit-rms vector in ker d0^T; rho_t = rho0 expm(Qt). ell_e = logmean(a,b), a=rho_i Q_ij, b=rho_j Q_ji.

## Conclusions (honest)

**Claim 1 (Pythagorean split with L = Onsager-Fisher weight): holds.** sigma = ||g_L||^2_L + ||c_L||^2_L to 4e-16 (max over t, 10 seeds, n=20/30, ER/complete, eps>0; up to 6e-13 at eps=0 only because sigma->0 and the ratio is ill-conditioned). Caveat: this identity is just orthogonal projection in the L inner product, so it is an algebraic fact once ell=J/f is used; the non-trivial physical content is (i) the excess part vanishes at the NESS (observed: ex_L/sigma_ss <1e-27, equivalent to d0^T J_ss=0) and (ii) hk_L(t)->sigma_ss (relative error at t=10 <=2.3e-3, limited by spectral gap of the slowest seeds), ex_L(t) decays exponentially at rate ~ 2*(spectral gap): fitted rate / (2 gap) = 1.04-1.5 (median ~1.07). The geometric housekeeping hk_L(t) is NOT the Hatano-Sasa housekeeping during the transient (they differ by ~18% of max sigma on average); they coincide only at the NESS. HS split is also exact (4.7e-16) but it is a different decomposition.
**Euclidean (L=I) Hodge: the energy splitting fails, as claimed.** With energies measured in the L-norm, |sigma - ex_E - hk_E|/sigma has median over t of 6%-39% (max over t up to 130% at eps=0.5), and the sum overshoots sigma (ss: ex_E+hk_E = 1.03-1.21 sigma). At eps=0 (reversible) Euclidean is exact since f is exactly a gradient at all t. Spurious gradient energy at the NESS: ex_E/sigma_ss = 3-21% (ER n=20: 0.21, 0.16, 0.087, 0.047 for eps=0.5,1,2,4), always >0 vs 1e-28 for L-weighted. Two points in the claim only partly confirmed: (a) it does NOT grow with eps - the fraction DEcreases with eps (the absolute value grows, e.g. ER n=20: 0.04, 0.15, 0.41, 1.9 for eps=0.5..4, but sigma grows faster); (b) it grows with heterogeneity of ell in a controlled sweep (h = 0 to 2, CV(ell) 0.20 to 1.76: 0.9% -> 1.3% -> 2.7% -> 8.7% -> 43%), but the pooled correlation across all eps/graphs is not monotone (Pearson on logs -0.12, confounded by eps and density; complete graphs have a smaller effect than ER at the same CV). Note also the linear J-pairing split <J,g_E> + <J,c_E> = sigma does hold, and <J,g_E> = 0 exactly at the NESS (since d0^T J=0), so the Euclidean failure is in the quadratic (energy) bookkeeping: the Euclidean split gives a non-zero quadratic gradient norm at the NESS, not a non-zero pairing.
**Claim 2 (topology count): holds** (trivially in the linear-algebra sense; the map Q -> log(Q_ij/Q_ji) is onto all 1-cochains, so the rank mod exact is |E|-rank d0). Rank = beta1 in all 16 graphs (complete K4..K10: 3,6,10,15,21,28,36 = (n-1)(n-2)/2; ER, cycles, grid 4x4: 9; trees: 0). Two independent rank computations agree. Equivalence Kolmogorov (fundamental cycles, and all simple cycles up to length 6 on K4-K8 / ER8 - up to 2618 cycles) <=> harmonic component = 0 <=> J_ss = 0 held in all 960 random Q (30 reversible + 30 generic per graph); trees always reversible; reversible-constructed harmonic <6e-14, generic >=0.005, and on C5 the harmonic norm equals |sum of log-ratios|/sqrt(5) to 4 digits.
**Claim 3 (finite data): partly.** Start from stationarity; K6 (beta1=10) and ER10 (beta1=14), 20 runs, T=1e3,1e4,1e5 jumps. (i) Lynn direct count and the L-Hodge plug-in sigma_hat = ex_L + hk_L are the same estimator algebraically (rho_hat_i Q_hat_ij = n_ij/T); they differ only by zero-count handling (pseudo-count 0.5 vs dropping pairs). RMSE/sigma: K6 8.2%/3.0%/0.74%, ER10 14%/4.8%/1.4%, scaling ~ T^(-1/2); bias small (<=0.6%) except small-T/sparse (ER10, T=1e3: Lynn -5.8% from dropped zero-count pairs, plug-in +2.2%). (ii) The L-weighted split of the estimate is consistent: ex_L_hat ~ 0 (the counts flux is divergence-free up to O(1/T)), hk_L_hat = sigma_hat (same bias/variance). (iii) The Euclidean split is inconsistent: hk_E_hat and ex_E_hat converge to their own population values (K6: 3.23 and 0.30; ER10: 1.57 and 0.34) which differ from the physical targets (sigma, 0) by +10% sigma (K6) / +28% sigma (ER10) and do not vanish as T grows (ex_E bias vs 0: 0.10/0.28 sigma at T=1e5); their sum is 1.20/1.56 sigma. (iv) Cycle affinities: the line integral of the cyclic part along a closed cycle equals that of f for ANY Hodge weight because the exact part integrates to zero, so L- and Euclidean cyclic integrals agree to 2e-15 and are NOT distinguished; RMS error (fundamental cycles) K6 0.64/0.24/0.077, ER10 1.14/0.34/0.11 (eps*A has rms ~1.5), ~T^(-1/2), bias ~1e-2. Distinguishing L vs E would require the edgewise cyclic field (relative error ~ similar for both: K6 0.23/0.077/0.027 (L), but they estimate different targets), not the cycle integral. So Claim 3 as phrased - L-weighted better than Euclidean for cycle-affinity recovery - is NOT supported; for excess/housekeeping EPR it is supported (bias of Euclidean is O(1) in T, L-weighted is O(1/T)).
**Limitations**: single system per E3 class (K6 eps=1.5, ER10 eps=1.5, seeds fixed), E1 10 seeds per cell; the Dechant-Sasa/Kolchinsky identification of excess/hk with these geometric parts is assumed (not re-derived).

## E1: Pythagorean decomposition (L-weighted vs Euclidean)

10 random seeds per row; t in [0,10], 201 time points. resL = max_t |sigma-ex_L-hk_L|/sigma; resE = |sigma-ex_E-hk_E|/sigma, where ex_E, hk_E are the L-norm energies of the Euclidean-Hodge parts (cross term / sigma). ss_* = at the exact stationary state (energy / sigma_ss).

| graph | n | eps | sigma_ss | max resL | median_t resE (mean over seeds) | max_t resE | ss ex_L/sigma | ss ex_E/sigma | ss hk_E/sigma | ss <J,g_E>/sigma | CV(ell) | decay rate/(2*gap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| er | 20 | 0 | 5.25e-29 | 5.6e-13 | 3.13e-15 | 1.0e-12 | n/a | n/a | n/a | n/a | 0.89 | n/a |
| er | 20 | 0.5 | 0.216 | 1.1e-15 | 3.87e-01 | 1.3e+00 | 5.8e-28 | 0.206 | 1.206 | 2.5e-15 | 0.89 | 1.34 |
| er | 20 | 1 | 0.923 | 7.6e-16 | 3.12e-01 | 9.1e-01 | 2.2e-28 | 0.158 | 1.158 | 2.9e-16 | 0.86 | 1.08 |
| er | 20 | 2 | 4.71 | 6.3e-16 | 1.74e-01 | 3.7e-01 | 7.8e-29 | 0.087 | 1.087 | 2.0e-16 | 0.85 | 1.06 |
| er | 20 | 4 | 39.6 | 7.4e-16 | 9.46e-02 | 3.7e-01 | 9.0e-29 | 0.047 | 1.047 | 1.4e-15 | 1.04 | 1.47 |
| complete | 20 | 0 | 3.45e-29 | 5.3e-13 | 4.44e-15 | 1.6e-12 | n/a | n/a | n/a | n/a | 0.91 | n/a |
| complete | 20 | 0.5 | 0.787 | 4.8e-16 | 1.21e-01 | 2.1e-01 | 1.8e-28 | 0.061 | 1.061 | 1.2e-15 | 0.92 | 1.04 |
| complete | 20 | 1 | 3.4 | 4.4e-16 | 1.11e-01 | 1.9e-01 | 4.5e-29 | 0.055 | 1.055 | 4.2e-16 | 0.92 | 1.06 |
| complete | 20 | 2 | 18.2 | 5.8e-16 | 8.85e-02 | 2.9e-01 | 7.6e-30 | 0.044 | 1.044 | 1.0e-16 | 0.94 | 1.05 |
| complete | 20 | 4 | 183 | 4.6e-16 | 7.17e-02 | 3.0e-01 | 5.3e-29 | 0.036 | 1.036 | 6.6e-17 | 1.26 | 1.05 |
| er | 30 | 0 | 4.43e-29 | 6.9e-13 | 5.35e-15 | 1.1e-12 | n/a | n/a | n/a | n/a | 1.00 | n/a |
| er | 30 | 0.5 | 0.333 | 6.4e-16 | 3.03e-01 | 5.5e-01 | 2.8e-28 | 0.152 | 1.152 | 4.3e-16 | 0.98 | 1.09 |
| er | 30 | 1 | 1.44 | 7.8e-16 | 2.54e-01 | 4.5e-01 | 5.1e-29 | 0.127 | 1.127 | 3.4e-16 | 0.95 | 1.17 |
| er | 30 | 2 | 7.52 | 5.6e-16 | 1.59e-01 | 3.2e-01 | 5.0e-29 | 0.079 | 1.079 | 1.1e-16 | 0.91 | 1.10 |
| er | 30 | 4 | 64.8 | 6.4e-16 | 8.77e-02 | 3.5e-01 | 2.0e-29 | 0.044 | 1.044 | 2.0e-16 | 1.04 | 1.12 |
| complete | 30 | 0 | 1.17e-28 | 2.6e-13 | 4.01e-15 | 8.4e-13 | n/a | n/a | n/a | n/a | 0.99 | n/a |
| complete | 30 | 0.5 | 1.27 | 5.9e-16 | 7.88e-02 | 2.0e-01 | 1.1e-28 | 0.039 | 1.039 | 3.3e-16 | 0.98 | 1.06 |
| complete | 30 | 1 | 5.59 | 4.3e-16 | 7.25e-02 | 1.7e-01 | 3.4e-29 | 0.036 | 1.036 | 1.5e-16 | 0.98 | 1.07 |
| complete | 30 | 2 | 31.5 | 4.4e-16 | 6.27e-02 | 2.9e-01 | 7.9e-30 | 0.031 | 1.031 | 1.0e-16 | 1.00 | 1.09 |
| complete | 30 | 4 | 363 | 5.2e-16 | 5.68e-02 | 2.8e-01 | 4.6e-30 | 0.028 | 1.028 | 1.9e-16 | 1.49 | 1.04 |

Heterogeneity sweep (ER n=20, eps=2, B,U scaled by h; 10 seeds): 

| h | CV(ell_ss) | ss ex_E/sigma (L-energy) | Euclid-norm grad fraction |g_E|^2/|f|^2 |
|---|---|---|---|
| 0 | 0.20 | 0.009 | 0.011 |
| 0.25 | 0.29 | 0.013 | 0.016 |
| 0.5 | 0.46 | 0.027 | 0.031 |
| 1 | 0.85 | 0.087 | 0.087 |
| 2 | 1.76 | 0.426 | 0.235 |

Across all 160 samples with eps>0: Spearman-like corr(CV(ell), ex_E/sigma)=-0.12 (Pearson on logs); 

Hatano-Sasa check: max_t |hk_HS + ex_HS - sigma|/sigma_max over all: 4.7e-16; geometric hk_L differs from Hatano-Sasa hk at transient: mean over rows of max_t|hk_L-hk_HS|/sigma_max = 0.179

Convergence of hk_L(t=10) to sigma_ss: max rel err over all seeds with eps>0 = 2.3e-03


## E2: topology count

rank(mod exact) = rank of Euclidean-projected (out of im d0) log(Q_ij/Q_ji) cochains for 35 random Q; "aug" = rank[C|d0]-rank d0 (independent check). Equivalence test: for 30 detailed-balance-constructed and 30 generic Q per graph, Kolmogorov (fundamental cycles, and ALL simple cycles of length<=6 where enumerated) <=> exact cochain (harmonic part=0) <=> stationary current J_ss=0.

| graph | V | E | beta1=E-V+1 | (n-1)(n-2)/2 | rank mod exact | rank aug | #simple cycles checked | equivalence all OK | max harm (reversible) | min harm (irrev.) | max|J_ss| (rev.) | min|J_ss| (irrev.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complete | 4 | 6 | 3 | 3 | 3 | 3 | 7 | True | 1.8e-15 | 0.41 | 4.3e-15 | 0.046 |
| complete | 5 | 10 | 6 | 6 | 6 | 6 | 37 | True | 2.6e-15 | 1 | 1.6e-14 | 0.099 |
| complete | 6 | 15 | 10 | 10 | 10 | 10 | 197 | True | 4.8e-15 | 1.6 | 7.1e-15 | 0.14 |
| complete | 7 | 21 | 15 | 15 | 15 | 15 | 812 | True | 4.8e-15 | 2.7 | 1.0e-14 | 0.19 |
| complete | 8 | 28 | 21 | 21 | 21 | 21 | 2618 | True | 5.0e-15 | 2.7 | 4.0e-15 | 0.17 |
| complete | 9 | 36 | 28 | 28 | 28 | 28 | - | True | 8.6e-15 | 4.3 | 2.6e-15 | 0.25 |
| complete | 10 | 45 | 36 | 36 | 36 | 36 | - | True | 5.6e-15 | 4.6 | 2.6e-15 | 0.27 |
| ER p=0.4 | 8 | 15 | 8 | - | 8 | 8 | 67 | True | 7.6e-15 | 1.2 | 6.7e-15 | 0.052 |
| ER p=0.4 | 12 | 29 | 18 | - | 18 | 18 | - | True | 5.4e-14 | 3 | 4.3e-15 | 0.076 |
| ER p=0.4 | 16 | 45 | 30 | - | 30 | 30 | - | True | 1.6e-14 | 3.6 | 3.8e-15 | 0.091 |
| cycle | 5 | 5 | 1 | - | 1 | 1 | 1 | True | 2.0e-15 | 0.031 | 5.7e-15 | 0.0013 |
| cycle | 12 | 12 | 1 | - | 1 | 1 | - | True | 5.0e-14 | 0.005 | 8.8e-15 | 6.9e-05 |
| tree | 6 | 5 | 0 | - | 0 | 0 | - | True | 3.4e-15 | - | 2.5e-15 | - |
| tree | 12 | 11 | 0 | - | 0 | 0 | - | True | 7.2e-15 | - | 8.6e-15 | - |
| tree | 25 | 24 | 0 | - | 0 | 0 | - | True | 1.7e-14 | - | 3.8e-15 | - |
| grid 4x4 | 16 | 24 | 9 | - | 9 | 9 | - | True | 2.0e-14 | 1.2 | 6.6e-15 | 0.022 |

5-cycle demo (random rates; even trials built reversible): log(prod fwd / prod bwd), harmonic-component norm, predicted |A|/sqrt5, stationary current, EPR

| reversible-constructed | log(prod fwd/bwd) | harmonic norm | |A|/sqrt(5) | max|J_ss| | EPR |
|---|---|---|---|---|---|
| True | -2.22e-16 | 7.51e-16 | 9.93e-17 | 3.9e-16 | 1.64e-30 |
| False | -2.9 | 1.3 | 1.3 | 0.072 | 0.21 |
| True | 0 | 1.13e-15 | 0 | 1.7e-16 | 2.03e-31 |
| False | -0.259 | 0.116 | 0.116 | 0.0043 | 0.00112 |
| True | 2.22e-16 | 9.26e-16 | 9.93e-17 | 1.4e-15 | 7.87e-30 |
| False | 3.98 | 1.78 | 1.78 | 0.13 | 0.533 |
| True | 0 | 6.93e-16 | 0 | 1.9e-16 | 5.79e-31 |
| False | -1.21 | 0.543 | 0.543 | 0.036 | 0.0432 |

## E3: finite-trajectory estimation

### System K6: n=6, |E|=15, beta1=10, sigma_ss=2.9293, true ex_E=0.2984, true hk_E=3.2277 (E-sum 3.5261), CV(ell)=0.65

Mean (std over 20 runs) of each estimator. Truth for stationary start: sigma=2.9293; excess(L)=0; housekeeping(L)=sigma; Euclid targets: ex_E=0.2984, hk_E=3.2277.

| jumps T | Lynn direct | plug-in (rho_hat=occupancy, = ex_L+hk_L) | plug-in (rho=stat(Qhat)) | hk_L | ex_L (true 0) | hk_E | ex_E | ex_E+hk_E (not sigma) |
|---|---|---|---|---|---|---|---|---|
| 1000 | 2.9308 (0.2412) | 2.9406 (0.2377) | 2.9419 (0.2380) | 2.9406 (0.2377) | 0.0000 (0.0000) | 3.2663 (0.2791) | 0.3265 (0.1024) | 3.5928 (0.3467) |
| 10000 | 2.9384 (0.0881) | 2.9304 (0.0875) | 2.9304 (0.0874) | 2.9304 (0.0875) | 0.0000 (0.0000) | 3.2186 (0.1108) | 0.2883 (0.0297) | 3.5069 (0.1366) |
| 100000 | 2.9246 (0.0211) | 2.9238 (0.0210) | 2.9238 (0.0210) | 2.9238 (0.0210) | 0.0000 (0.0000) | 3.2250 (0.0257) | 0.3011 (0.0091) | 3.5261 (0.0324) |

Bias / RMSE relative to sigma_ss (bias = mean-truth):

| T | Lynn bias | Lynn RMSE | plug bias | plug RMSE | plug_stat bias | plug_stat RMSE | hk_L bias | hk_L RMSE | ex_L bias(truth 0) | hk_E bias vs L-target sigma | hk_E bias vs E-pop target | ex_E bias vs L-target 0 | ex_E bias vs E-pop target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0.0005 | 0.0823 | 0.0039 | 0.0812 | 0.0043 | 0.0814 | 0.0038 | 0.0812 | 0.0000 | 0.1151 | 0.0132 | 0.1114 | 0.0096 |
| 10000 | 0.0031 | 0.0302 | 0.0004 | 0.0299 | 0.0004 | 0.0298 | 0.0004 | 0.0299 | 0.0000 | 0.0988 | -0.0031 | 0.0984 | -0.0035 |
| 100000 | -0.0016 | 0.0074 | -0.0019 | 0.0074 | -0.0019 | 0.0074 | -0.0019 | 0.0074 | 0.0000 | 0.1009 | -0.0009 | 0.1028 | 0.0009 |

Cycle affinity (fundamental cycles, 10 of them): RMS error of Hodge-cyc line integral vs true sum of eps*A along cycle; per-edge cyclic-field error ||c_hat - c_true||_L2 (edgewise, relative to ||c_true||).

| T | RMS aff err (L-cyc integral) | RMS aff err (Eucl-cyc integral) | max |L-integral - Eucl-integral| | mean sqrt(sum 1/n) heuristic scale | rel. edgewise error ||c_L-c_L*|| /||c_L*|| | rel. edgewise error ||c_E-c_E*||/||c_E*|| | affinity bias (mean signed) |
|---|---|---|---|---|---|---|---|
| 1000 | 0.6402 | 0.6402 | 8.9e-16 | - | 0.2336 | 0.2215 | 0.0147 |
| 10000 | 0.2357 | 0.2357 | 1.8e-15 | - | 0.0771 | 0.0732 | -0.0166 |
| 100000 | 0.0770 | 0.0770 | 8.9e-16 | - | 0.0271 | 0.0252 | -0.0091 |
### System ER10: n=10, |E|=23, beta1=14, sigma_ss=1.2301, true ex_E=0.3426, true hk_E=1.5728 (E-sum 1.9154), CV(ell)=1.22

Mean (std over 20 runs) of each estimator. Truth for stationary start: sigma=1.2301; excess(L)=0; housekeeping(L)=sigma; Euclid targets: ex_E=0.3426, hk_E=1.5728.

| jumps T | Lynn direct | plug-in (rho_hat=occupancy, = ex_L+hk_L) | plug-in (rho=stat(Qhat)) | hk_L | ex_L (true 0) | hk_E | ex_E | ex_E+hk_E (not sigma) |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1.1589 (0.1665) | 1.2573 (0.1645) | 1.2577 (0.1658) | 1.2573 (0.1645) | 0.0000 (0.0000) | 1.6568 (0.1850) | 0.3998 (0.0900) | 2.0566 (0.2408) |
| 10000 | 1.2375 (0.0590) | 1.2287 (0.0586) | 1.2289 (0.0586) | 1.2287 (0.0586) | 0.0000 (0.0000) | 1.5738 (0.0840) | 0.3452 (0.0373) | 1.9190 (0.1160) |
| 100000 | 1.2325 (0.0169) | 1.2315 (0.0169) | 1.2315 (0.0169) | 1.2315 (0.0169) | 0.0000 (0.0000) | 1.5757 (0.0215) | 0.3442 (0.0105) | 1.9199 (0.0293) |

Bias / RMSE relative to sigma_ss (bias = mean-truth):

| T | Lynn bias | Lynn RMSE | plug bias | plug RMSE | plug_stat bias | plug_stat RMSE | hk_L bias | hk_L RMSE | ex_L bias(truth 0) | hk_E bias vs L-target sigma | hk_E bias vs E-pop target | ex_E bias vs L-target 0 | ex_E bias vs E-pop target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | -0.0579 | 0.1472 | 0.0221 | 0.1355 | 0.0224 | 0.1367 | 0.0220 | 0.1355 | 0.0000 | 0.3468 | 0.0683 | 0.3250 | 0.0465 |
| 10000 | 0.0060 | 0.0483 | -0.0012 | 0.0477 | -0.0010 | 0.0477 | -0.0012 | 0.0477 | 0.0000 | 0.2794 | 0.0008 | 0.2806 | 0.0021 |
| 100000 | 0.0019 | 0.0139 | 0.0011 | 0.0138 | 0.0011 | 0.0138 | 0.0011 | 0.0138 | 0.0000 | 0.2809 | 0.0024 | 0.2798 | 0.0012 |

Cycle affinity (fundamental cycles, 14 of them): RMS error of Hodge-cyc line integral vs true sum of eps*A along cycle; per-edge cyclic-field error ||c_hat - c_true||_L2 (edgewise, relative to ||c_true||).

| T | RMS aff err (L-cyc integral) | RMS aff err (Eucl-cyc integral) | max |L-integral - Eucl-integral| | mean sqrt(sum 1/n) heuristic scale | rel. edgewise error ||c_L-c_L*|| /||c_L*|| | rel. edgewise error ||c_E-c_E*||/||c_E*|| | affinity bias (mean signed) |
|---|---|---|---|---|---|---|---|
| 1000 | 1.1434 | 1.1434 | 1.8e-15 | - | 0.3687 | 0.3338 | 0.0421 |
| 10000 | 0.3432 | 0.3432 | 1.8e-15 | - | 0.1175 | 0.1075 | -0.0184 |
| 100000 | 0.1117 | 0.1117 | 1.8e-15 | - | 0.0394 | 0.0364 | 0.0156 |