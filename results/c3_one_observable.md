# C3: one observable for the parallel-channel loop R1 reverse / R4 forward

Code: `hyperme/c3_observable.py`, the observable rows of `hyperme/linear_fit.assemble` and the observable term of `hyperme/train.train`, the wired reservoir affinity `presets.c3_wired(known_cycle=...)` (`WiredDrive(chem_mu0_known=...)`), tests `hyperme/tests/test_c3_observable.py`. Runs on the CPU of axiom (no GPU), float64, in `~/fh_anchor/runs/hyperme/observable/`, copied to machine 49 `~/Fisher-Hodge/runs/hyperme/observable/`.

## Prediction and setup

R1 reverse (X → A) and R4 forward (X → E) remove one X at a rate proportional to x under every chemostat setting. They form a loop of dynamically indistinguishable parallel channels, so by proposition 1′ densities cannot split them and one extra observable should. The split matters for σ: in the base runs R1 carries 0.015 of the true σ instead of 0.138 (`results/c3_equilibrium_anchor.md`).

- **V1, the turnover of one reservoir.** Each training trajectory also carries r_E(t) = Σ_sites J_R4(t), the net production rate of E. Its relative error enters the rollout loss with weight 1, the same term for every model with fluxes. It also enters the closed-form step as extra rows, weighted like the loss.
- **V2, known reservoir standard free energies.** μ⁰_A − μ⁰_E = log(k₁k₄/ε²) = 13.82 (true value) is wired into the wired model as a linear constraint on its chemostat potentials. No extra data. CRNN-chem and Flux-GNN have no standard potentials, so V2 does not apply to them (N/A).
- Everything else as before: the 48 driven training trajectories, no anchor, `fit()` with the closed-form step, 2000 Adam iterations, lr 1e-2, MPRK22 × 8, seeds 0 and 1. A Flux-GNN run whose final loss is above 0.05 is repeated once at lr 1e-3.

## Results

| variant | model | seed | ID 10T | OOD 10T | σ rel err ID ; OOD | log10(b/a) R1 / R2 / R3 / R4 | σ share R1 / R2 / R3 / R4, total / true (test_id) | final train loss |
|---|---|---|---|---|---|---|---|---|
| truth | | | | | | −3.21 / −3.63 / −2.78 / −2.79 | 0.138 / 0.332 / 0.382 / 0.135, 1 | |
| base | HyperME wired | 0 | 0.0026 | 0.0030 | 0.221 ; 0.283 | −0.51 / −6.63 / −2.04 / −5.48 | 0.015 / 0.688 / 0.302 / 0.126, 1.14 | 1.1e-5 |
| base | HyperME wired | 1 | 0.0027 | 0.0036 | 0.222 ; 0.279 | −0.51 / −6.63 / −2.04 / −5.48 | 0.015 / 0.688 / 0.302 / 0.125, 1.14 | 1.1e-5 |
| base | CRNN-chem | 0 | 0.0035 | 0.0033 | 0.233 ; 0.289 | −0.51 / −6.63 / −2.11 / −5.49 | 0.015 / 0.687 / 0.311 / 0.127, 1.15 | 1.0e-5 |
| base | CRNN-chem | 1 | 0.0037 | 0.0029 | 0.231 ; 0.287 | −0.51 / −6.63 / −2.10 / −5.49 | 0.015 / 0.688 / 0.309 / 0.127, 1.15 | 1.0e-5 |
| base | Flux-GNN | 0 | 0.219 | 0.504 | 2.29 ; 1.97 | −2.48 / −2.97 / −4.56 / −2.12 | 0.101 / 0.232 / 0.563 / 0.103, 3.67 | 4.4e-4 |
| base | Flux-GNN | 1 | 0.229 | 0.693 | 0.56 ; 0.52 | −2.36 / −3.09 / −4.58 / −2.10 | 0.098 / 0.277 / 0.599 / 0.102, 1.28 | 4.7e-4 |
| V1 | HyperME wired | 0 | 0.0043 | 0.0034 | 0.437 ; 0.433 | −6.21 / −6.63 / −2.20 / **−2.87** | 0.271 / 0.686 / 0.320 / **0.139**, 1.43 | 1.9e-5 |
| V1 | HyperME wired | 1 | 0.0047 | 0.0031 | 0.438 ; 0.433 | −6.21 / −6.63 / −2.20 / **−2.87** | 0.271 / 0.686 / 0.320 / **0.139**, 1.43 | 1.9e-5 |
| V1 | CRNN-chem | 0 | 0.0043 | 0.0040 | 0.437 ; 0.431 | −6.15 / −6.59 / −2.22 / **−2.97** | 0.268 / 0.680 / 0.323 / 0.144, 1.43 | 1.9e-5 |
| V1 | CRNN-chem | 1 | 0.0044 | 0.0043 | 0.432 ; 0.427 | −6.15 / −6.57 / −2.21 / **−2.96** | 0.268 / 0.678 / 0.321 / 0.143, 1.42 | 1.8e-5 |
| V1 | Flux-GNN | 0 | 0.223 | 0.557 | 5.53 ; 4.71 | −3.12 / −3.14 / −6.87 / −2.16 | 0.136 / 0.278 / 0.853 / 0.098, 7.67 | 6.4e-4 |
| V1 | Flux-GNN | 1 | 0.227 | 0.461 | 3.20 ; 2.57 | −3.16 / −3.20 / −7.64 / −2.03 | 0.137 / 0.276 / 0.927 / 0.102, 4.41 | 5.6e-4 |
| V2 | HyperME wired | 0 | 0.0026 | 0.0029 | 0.221 ; 0.283 | −0.51 / −6.63 / −2.04 / −5.49 | 0.015 / 0.688 / 0.302 / 0.126, 1.14 | 1.1e-5 |
| V2 | HyperME wired | 1 | 0.0027 | 0.0036 | 0.222 ; 0.280 | −0.51 / −6.63 / −2.04 / −5.49 | 0.015 / 0.688 / 0.302 / 0.126, 1.14 | 1.1e-5 |
| V2 | CRNN-chem, Flux-GNN | | N/A: no standard potentials to wire | | | | | |

log10(b/a) is the median over the test_id states. The base rows reproduce `results/c3_equilibrium_anchor.md`. No Flux-GNN run collapsed (all final losses below 0.05), so none was repeated at lr 1e-3. The Flux-GNN share totals exceed the sum of the four reactions because its transport channels are not symmetric and dissipate on their own (2.4 of the true σ in base seed 0).

**The closed-form step in V1** (wired, the class factors relative to the neutral start): R4 forward 1.00 and R4 reverse 8.3e-4 (true 1 and 1e-3), R1 reverse 0 (true 1e-3), R2 reverse 0 (true 1e-3). With the true constants as the reference, the derivative residual is 2.11e-4 at the truth and 2.02e-4 at the least-squares point. Moving R1 reverse from 0 to its true value raises the residual by 1.4e-7 (0.07 percent). In base the same move costs 0.27, because there R1 reverse also has to carry the X decay of R4.

**σ range.** The smallest σ over the class factors whose fit residual on the training data (densities, plus the observable in V1) is within 1 + tol of the best is found by a convex program. With a = xP and b = yM for the classes (x, y) of a hyperedge, σ sums exactly to Σ over class pairs of A x log(x/y) + B y log(y/x) + Cx − Dy, which is convex in about ten variables (`sigma_min_convex`, SLSQP from several starts, all converged). For V2 the class factors are a superset of the wired class, so this is still a valid lower bound. The upper end is infinite when a one-way class can be switched off within the tolerance.

| training data | tol | σ_min / σ_true ID ; OOD | true σ ≥ σ_min (fraction of test points) | one-way classes that can be switched off | upper end |
|---|---|---|---|---|---|
| densities (base, V2; wired and CRNN, both seeds) | 0.1 | 0.576 ; 0.618 | 1.000 ; 1.000 | 5 | ∞ |
| densities (base, V2) | 1.0 | 0.462 ; 0.497 | 1.000 ; 1.000 | 6 | ∞ |
| densities + E turnover (V1; wired and CRNN, both seeds) | 0.1 | **0.804 ; 0.811** | 1.000 ; 1.000 | 4 | ∞ |
| densities + E turnover (V1) | 1.0 | 0.66 ; 0.66 | 1.000 ; 1.000 | 4 | ∞ |

## Reading

- **The observable does separate the loop.** With the turnover of E, R4 is identified in both directions: the closed-form step gives R4 forward 1.00 and R4 reverse 8.3e-4 against true 1 and 1e-3. After training the R4 ratio is −2.87 (wired) and −2.96 (CRNN-chem) against −2.79, where it was −5.5 without the observable, and the R4 share of σ is 0.139 against 0.135. On a small synthetic Brusselator with a short sampling interval (Δt = 0.02, `tests/test_c3_observable.py`) the same observable also recovers R1 reverse to within 30 percent, against a factor of several hundred off without it.
- **On this benchmark R1 reverse is then below the resolution, and σ gets worse.** Once R4 forward is pinned, R1 reverse is what is left of the X decay, one part in a thousand. The Simpson error of the derivative fit at Δt = 0.2 is larger than that part (the truth itself fits worse than the least-squares point, 2.11e-4 against 2.02e-4), and moving R1 reverse to its true value changes the residual by 0.07 percent. The closed-form step puts it at zero, and the rollout training does not move it, because the gradient in log space shrinks with the rate. The model then treats R1 as almost irreversible (−6.2 against −3.2), R1 carries 0.27 of σ instead of 0.14, and the σ error rises from 0.22 / 0.28 to 0.44 / 0.43 in both seeds and both models. The density error stays small (ID 10T 0.43 to 0.47 percent against 0.26, OOD 0.31 to 0.43 percent).
- **V2 changes nothing.** Wiring μ⁰_A − μ⁰_E fixes the A → E cycle affinity, which the base fit already had right (13.80 against 13.81). The free split then moves from R1 reverse against R4 forward to R1 forward (proportional to a) against R4 reverse (constant, 1e-3 of the source at e = 1). With a in [0.9, 1.1], that split is again a one-in-a-thousand effect. Both seeds end where base ends, to the printed digit.
- **The lower bound tightens with the observable, and there is still no upper bound.** In V1 the smallest σ consistent with the data within 10 percent rises from 0.58 to 0.80 of the truth (from 0.46 to 0.66 at 100 percent). The true σ is above the bound at every test point in every run. Four one-way classes can still be switched off without leaving the tolerance (five without the observable), so σ has no finite upper end.
- **R2 is the largest remaining error, unchanged in every variant.** The R2 share of σ is 0.68 to 0.69 against a true 0.33 for the wired model and CRNN-chem in base, V1 and V2, with the R2 ratio at −6.6 against −3.6. Its reverse balances its forward rate only when y ≈ εx, a thousand times below x, which the relative rollout loss does not see. The prediction is that the next observable must look at Y near equilibrium, for example a local measurement of Y or a per-species weighted loss. This is not run here because it changes the loss protocol.
- **The black-box rates.** Flux-GNN with the observable gets the R1 ratio close to the truth (−3.1 against −3.2) but R3 far off (−6.9 to −7.6 against −2.8). It has a σ error of 2.6 to 5.5, density errors of 22 percent ID and 46 to 56 percent OOD, and spurious dissipation on its asymmetric transport channels. The near-correct R1 is not identification: with R3 three to four decades off, the same data are fitted by a different split.

## Limits

Two seeds, one observable design (the summed R4 flux at every sampling time, noise-free). The derivative residual is dominated by the Simpson error at Δt = 0.2, so the tolerances 0.1 and 1.0 are not statistical confidence levels. The switch-off test for V2 uses the free class factors, a superset of the wired class.
