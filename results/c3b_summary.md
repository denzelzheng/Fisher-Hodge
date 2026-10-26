# C3b: chemostat drive through the exact chemical-potential channel

HyperME-chem: mu_A = log a and mu_B = log b enter the rate exponents through the chemostat stoichiometry (a_h = k exp(<nu+, mu> + <nu+_ext, mu_ext> + A/2), b_h likewise with nu-_ext). k_h, A_raw and g_theta get **no** a, b features, so the only (a, b) dependence is the exact one. B1 CRNN-chem: kf, kb constant per hyperedge type, a and b multiply as mass-action concentrations. (Note: the original B1 of C2/C3 received a, b as normalised MLP features, not as concentrations.) B2 GNN-ODE keeps normalised feature input. Same data, protocol (2000 it, batch 8) and explicit positive SSP-RK2 integrator as before (no `mprk` integrator exists in hyperme/integrators.py). New models: 1 seed (s0); old models: s0 / s1.

## C3 Brusselator (OOD = b in [3.2, 3.7], training b in [2.3, 3.0])

| model | ID T | ID 10T (homog / spatial) | OOD T | OOD 2T | OOD 10T (homog / spatial) | OOD period rel err | OOD amp rel err | OOD traj c<0 | sigma rel err ID / OOD | learned cycle affinities ID (true 13.8, 14.8) |
|---|---|---|---|---|---|---|---|---|---|---|
| HyperME (a,b as MLP features) | 0.0229 / 0.0266 | 0.0477 (0.0579 / 0.0357) / 0.0411 (0.0643 / 0.0218) | 0.152 / 0.0381 | 0.349 / 0.366 | 0.56 (0.566 / 0.56) / 0.519 (0.765 / 0.395) | 0.047 / 0.073 | 0.138 / 0.086 | 0 / 0 | 0.493 ; 0.431 / 0.503 ; 0.428 | (5.5, 8.3) / (6.3, 8.3) |
| HyperME-chem (mu_A, mu_B in exponent, MLP G) | 0.0256 | 0.0351 (0.0576 / 0.0104) | 0.0369 | 0.0808 | 0.123 (0.127 / 0.0306) | 0.025 | 0.067 | 0 | 0.538 ; 0.507 | (4.4, 7.9) |
| HyperME-chem-cvx (+ convex separable G) | 0.12 | 0.232 (0.128 / 0.338) | 0.156 | 0.306 | 0.389 (0.428 / 0.358) | 0.090 | 0.197 | 0 | 0.682 ; 0.664 | (6.0, 10.1) |
| B1 CRNN (a,b as features) | 0.198 / 0.0317 | 0.692 (0.511 / 0.755) / 0.138 (0.138 / 0.133) | 0.294 / 0.0811 | 0.635 / 0.149 | 0.646 (0.412 / 0.682) / 0.444 (0.753 / 0.142) | 0.092 / 0.063 | 0.154 / 0.172 | 0 / 1 | 1.463 ; 1.981 / 0.397 ; 0.422 | (6.6, 9.6) / (7.5, 10.7) |
| B1 CRNN-chem (a,b as mass-action conc.) | 0.0133 | 0.0626 (0.0966 / 0.0086) | 0.0166 | 0.0172 | 0.0253 (0.0493 / 0.00896) | 0.012 | 0.044 | 0 | 0.400 ; 0.353 | (6.2, 10.7) |
| B2 GNN-ODE (features) | 0.116 / 0.135 | 0.357 (0.124 / 0.524) / 0.179 (0.0369 / 0.522) | 0.271 / 0.185 | 0.613 / 0.333 | 0.571 (0.371 / 0.678) / 0.355 (0.125 / 0.601) | 0.109 / 0.012 | 0.146 / 0.135 | 2 / 3 | n/a | n/a |
| B3 (A=0, features) | 0.349 / 0.469 | 0.575 (0.519 / 0.596) / 0.559 (0.601 / 0.535) | 0.522 / 0.62 | 0.586 / 0.593 | 0.599 (0.524 / 0.63) / 0.627 (0.701 / 0.614) | n/a / n/a | 0.998 / 0.995 | 0 / 0 | 0.999 ; 1.000 / 0.998 ; 0.999 | (0.0, 0.0) / (0.0, 0.0) |

## C2 Schlogl (OOD = a in [3.15, 3.35], b in [0.3, 1.0]; training a in [2.85, 3.10], b in [1.0, 3.2])

| model | final train loss | ID T | ID 10T | OOD T | OOD 10T | front 10T | OOD traj c<0 | max X / true max (OOD) | sigma rel err ID / OOD | affinity MAE ID / OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| HyperME (a,b as MLP features) | 1.27e-04 / 1.33e-04 | 0.0305 / 0.0358 | 0.41 / 0.512 | 1.44 / 3.1 | 5.26 / 20.7 | 0.29 / 0.403 | 0 / 0 | 6.25 / 6.55 | 0.519 ; 3.003 / 0.530 ; 8.577 | 0.44 ; 1.21 / 0.55 ; 1.51 |
| HyperME (features), 6000 it | 4.25e-05 | 0.019 | 0.0924 | 0.445 | 1.65 | 0.163 | 0 | 1.95 | 0.543 ; 1.723 | 0.31 ; 0.28 |
| HyperME-chem (mu_A, mu_B in exponent, MLP G) | 1.56e-03 | 0.179 | 0.335 | 0.741 | 0.677 | 0.278 | 0 | 1.00 | 0.997 ; 0.987 | 2.56 ; 2.56 |
| HyperME-chem-cvx (+ convex separable G) | 6.03e-04 / 4.86e-03 | 0.0643 / 0.217 | 0.501 / 2.3 | 0.108 / 0.368 | 0.833 / 2.94 | 0.557 / 0.377 | 0 / 0 | 1.00 / 1.00 | 0.305 ; 0.405 / 0.982 ; 0.994 | 0.06 ; 0.06 / 2.30 ; 2.30 |
| HyperME-chem-cvx, 6000 it | 2.73e-05 | 0.0111 | 0.087 | 0.0454 | 0.237 | 0.123 | 0 | 1.00 | 0.060 ; 0.073 | 0.04 ; 0.04 |
| B1 CRNN (a,b as features) | 1.41e-04 / 1.60e-04 | 0.032 / 0.0382 | 0.575 / 0.515 | 0.399 / 1.67 | 0.921 / 8.53 | 0.502 / 0.492 | 0 / 0 | 2.96 / 9.39 | 0.465 ; 0.295 / 0.457 ; 3.836 | 0.64 ; 1.05 / 0.72 ; 2.60 |
| B1 CRNN (features), 6000 it | 5.11e-05 | 0.0158 | 0.185 | 0.156 | 0.118 | 0.21 | 0 | 1.04 | 0.399 ; 0.233 | 0.15 ; 0.48 |
| B1 CRNN-chem (a,b as mass-action conc.) | 2.48e-06 | 0.00367 | 0.00788 | 0.0106 | 0.00689 | 0.00715 | 0 | 1.00 | 0.022 ; 0.031 | 0.00 ; 0.00 |
| B2 GNN-ODE (features) | 5.55e-06 / 5.03e-06 | 0.00769 / 0.0052 | 0.051 / 0.0744 | 0.304 / 0.186 | 0.912 / 0.846 | 0.0517 / 0.0075 | 0 / 8 | 1.00 / 1.00 | n/a | n/a |
| B3 (A=0, features) | 8.51e-04 / 8.74e-04 | 0.078 / 0.0934 | 0.348 / 0.308 | 0.441 / 0.52 | 0.862 / 0.888 | 0.409 / 0.419 | 0 / 0 | 1.00 / 1.00 | 0.994 ; 0.989 / 0.993 ; 0.990 | 2.71 ; 3.97 / 2.71 ; 3.97 |

## Notes (honest reading)

- **C3 OOD-b: the expected drop happens.** With mu_A, mu_B in the exponent and no a,b features, HyperME-chem (MLP G, s0) goes from 52-56% to **12.3%** OOD 10T error (homogeneous limit cycle 12.7%, Turing 3.1%). OOD period error drops from 5-7% to 2.5% and amplitude error from 9-14% to 6.7%. ID also improves (3.5% vs 4.1-4.8%). sigma rel err (~0.5) and the learned affinities (4.4, 7.9 vs true 13.8, 14.8) do not improve: the nearly irreversible edges (reverse rate 1e-3) remain weakly identified.
- **B1 CRNN-chem is better still:** 2.5% OOD 10T, 6.3% ID. Its mass-action monomials are the exact model class for this ideal system, and with no features it is essentially 5 x 2 constants to fit. The original B1 (a,b as normalised features) was at 44-65% OOD. Most of the OOD failure of every model in C2/C3 was the feature channel, not the architecture.
- **Convex-G variant on C3 is worse (s0):** 23% ID, 39% OOD. The separable monotone g_s(c_s) with 16 tanh units per species (no cross-species term) trains to a 10x higher loss in 2000 it (2.4e-3 vs 2e-4). Brusselator has no bistability, so the convexity constraint buys nothing here.
- **C2 OOD-a.** HyperME-chem with an MLP G (s0) fits worse than the feature version (train loss 1.6e-3). It ends with A ~ 0 (sigma rel err ~1, affinity MAE 2.56) and makes the bistability with a **non-convex learned free energy** instead of the drive. This is the equilibrium phase-separation mechanism of B3, and it is reachable once G has no drive inputs. Constraining G to be convex/separable (mu monotone in c: one stable phase, so multistability must come from A) fixes the thermodynamics immediately: affinity MAE 0.06, A_cyc err 0.02, sigma err 0.30 at 2000 it. At 6000 it it reaches ID 8.7%, **OOD 23.7%**, sigma err 0.06-0.07 and affinity MAE 0.04. The feature-input HyperME at 6000 it has 165% OOD and sigma err 0.54-1.7. At 2000 it the convex variant is still under-trained (ID 50% s0; s1 diverges in 10T to 230%), like every structured model on C2. B1 CRNN-chem (exact class) is at 0.7% everywhere.
- **Integrator.** `hyperme/integrators.py` has no `mprk` (Patankar). All runs use the same explicit positive SSP-RK2 as before. No negative concentrations for any HyperME/B1-chem rollout.


## C3b with the library wired mode and the chemostat prefactor (2026-10-26)

Code: `hyperme/presets.py` (`c3_wired`, `c3_true_wired`), `hyperme/c3b_wired.py` (metrics), tests `hyperme/tests/test_wired.py::test_c3_wired_*`. Runs and `c3b_wired_metrics.json` on machine 49 in `~/Fisher-Hodge/runs/hyperme/c3b/`. CPU, 1 seed, 2000 iterations, batch 8, MPRK22 (Patankar) with 8 steps per sampling interval. The `hyperme` library does have `mprk` now (commit b7f763b); the earlier note above predates it.

Model: μ = log c + μ⁰ (μ⁰ learned for X, Y), A = chemostat force of (a, b, d = 1, e = 1) plus learned chemostat μ⁰, log k a table over the 6 hyperedge types, 13 parameters in all. **New:** the chemostats also multiply the symmetric prefactor by exp(⟨ν⁺ + ν⁻, log c_Y⟩ / 2), i.e. sqrt(a) on R1 and sqrt(b) on R3 (`chem_prefactor=True`, the default). Without this factor the class cannot represent the generator: with the true constants R1 is off by sqrt(a) and R3 by sqrt(b) (test). The chemostat (a, b) enters nowhere else; no feature input.

| model | ID T | ID 10T (homog / spatial) | OOD T | OOD 2T | OOD 10T (homog / spatial) | OOD period rel err | OOD amp rel err | OOD traj c<0 | sigma rel err ID ; OOD | learned cycle affinities ID (true 13.8, 14.8) | final train loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| true constants, explicit substeps | 2.1e-05 | 6.4e-06 | 3.7e-05 | 8.7e-05 | **8.3e-06** | 0.000 | 0.000 | 0 | 0.000 ; 0.000 | (13.8, 14.8) | – |
| true constants, MPRK22 × 8 | 0.0056 | 0.0165 (0.025 / 4e-06) | 0.0068 | 0.0159 | **0.0067** (0.034 / 7e-11) | 0.007 | 0.018 | 0 | 0.050 ; 0.121 | (13.8, 14.8) | – |
| trained, neutral start, lr 1e-2 | 0.196 | 0.217 (0.115 / 0.377) | 0.245 | 0.451 | 0.400 (0.421 / 0.339) | 0.544 | 0.344 | 0 | 0.856 ; 0.861 | (0.1, 5.5) | 5.7e-03 |
| trained, neutral start, lr 2e-2 | 0.139 | 0.166 (0.159 / 0.352) | 0.153 | 0.390 | 0.323 (0.422 / 0.281) | 0.231 | 0.232 | 0 | 0.795 ; 0.808 | (0.2, 6.3) | 2.8e-03 |
| trained, neutral start, lr 5e-2 | 0.103 | 0.169 (0.169 / 0.147) | 0.106 | 0.346 | 0.368 (0.538 / 0.275) | 0.120 | 0.156 | 0 | 0.765 ; 0.794 | (0.3, 7.0) | 1.5e-03 |
| trained, neutral start, lr 1e-1 | 0.088 | 0.275 (0.608 / 0.139) | 0.088 | 0.299 | 0.307 (0.386 / 0.273) | 0.095 | 0.118 | 0 | 0.751 ; 0.767 | (0.3, 7.3) | 1.1e-03 |
| trained, neutral start, lr 1e-1, 6000 it | 0.053 | 0.283 (0.396 / 0.032) | 0.058 | 0.170 | 0.241 (0.254 / 0.137) | 0.052 | 0.083 | 0 | 0.675 ; 0.671 | (0.4, 8.3) | 5.0e-04 |
| trained, forward-biased start, lr 2e-2 | 0.033 | 0.169 (0.250 / 0.019) | 0.038 | 0.098 | **0.123** (0.159 / 0.021) | 0.035 | 0.051 | 0 | 0.514 ; 0.551 | (7.9, 9.1) | 1.5e-04 |
| trained, true constants + noise 0.3, lr 1e-2 (basin check) | 0.0061 | 0.0175 (0.026 / 0.003) | 0.0064 | 0.0067 | **0.0100** (0.014 / 0.002) | 0.002 | 0.026 | 0 | 0.063 ; 0.042 | (13.5, 14.8) | 1.1e-05 |
| for reference: HyperME-chem (above) | 0.0256 | 0.0351 | 0.0369 | 0.0808 | 0.123 | 0.025 | 0.067 | 0 | 0.538 ; 0.507 | (4.4, 7.9) | 2e-04 |
| for reference: B1 CRNN-chem (above) | 0.0133 | 0.0626 | 0.0166 | 0.0172 | 0.0253 | 0.012 | 0.044 | 0 | 0.400 ; 0.353 | (6.2, 10.7) | – |

Neutral start: μ⁰ = 0 and log k = 0, i.e. every reaction fully reversible with k = 1. Forward-biased start: every reaction runs in its written direction with forward rate 1 and forward/backward ratio e⁴ (log k = −2, ⟨ν⁺ − ν⁻, μ⁰⟩ = 4), transport k = 1; it uses no true constant. Basin check: the true 13 parameters plus Gaussian noise of 0.3.

Reading:

- **The prefactor makes the wired class exact for every b.** With the true constants the wired model reproduces the OOD-b data to 8e-6 (explicit substeps) and 0.67 percent with MPRK22 × 8, where the remainder is the phase drift of the limit cycle under the integrator. This is below the oracle CRNN-chem (2.5 percent). Without the prefactor no setting of the constants is exact once b leaves its training value, because R3 then scales as sqrt(b) instead of b.
- **It does not by itself explain the 12 percent of the trained model.** Trained from scratch for 2000 iterations, the wired model does not reach its exact point. From the neutral start it stalls at a training loss of 1e-3 to 6e-3 for every learning rate from 1e-2 to 1e-1 (OOD 10T 31 to 40 percent; 24 percent after 6000 iterations, still falling). The learned R1 and R4 stay nearly reversible (reaction affinities 0 instead of 6.9). From the forward-biased start it reaches 1.5e-4 and OOD 12.3 percent, the same as the earlier HyperME-chem, with period error 3.5 percent and amplitude error 5 percent. From the true constants plus noise it returns to 1e-5 and OOD 1.0 percent. So the basin around the truth is good and the obstacle is optimisation: the reverse rates must fall by a factor of about 1e3 in log space, where the gradient shrinks with the rate. The oracle CRNN-chem uses softplus (linear-scale) rate constants and reaches 2.5 percent in the same budget.
- **Thermodynamic readouts follow the fit.** sigma relative error is 0.05 to 0.12 at the true constants and in the basin check, 0.51 to 0.55 from the forward-biased start, and 0.67 to 0.86 from the neutral start. The learned cycle affinities are (13.5, 14.8) in the basin check, (7.9, 9.1) from the forward-biased start and (0.1 to 0.4, 5.5 to 8.3) from the neutral start; true (13.8, 14.8).
- Limits: one seed; the learning rate and start were not selected on test data, and the basin check uses the true constants, so it is a diagnostic, not a result.
