# C3: does an undriven equilibrium anchor fix the entropy production?

Code: `hyperme/c3_anchor.py` (anchor data, Flux-GNN with Patankar stepping, rate ratios, σ lower bound), `hyperme/linear_fit.py` (`assemble`, `nnls_kappa`), test `hyperme/tests/test_c3_anchor.py`. Runs: machine 49 and the CPU of axiom (no GPU), float64, in `runs/hyperme/anchor/` on 49.

## Question and setup

On the driven Brusselator the trained wired model has a σ error of 0.22 to 0.28 (`results/convex_training.md`). Every reverse rate is about 1e-3 of its forward rate, so it barely moves the driven densities, while σ contains log(a/b). In the wired model the reverse rates follow from local detailed balance with the standard potentials μ⁰, which are material constants shared by every condition. An undriven trajectory at a detailed-balance point has every reverse rate as large as its forward rate. Can a few such trajectories pin down μ⁰ and σ?

**Anchor data.** Five trajectories on the same 8 × 8 grid and time grid (Δt = 0.2, 20 steps), solved with DOP853 at rtol 1e-10. Each picks an equilibrium x* in [0.7, 1.5] and d log-uniform in [300, 3000] and sets a = εx*, e = x*/ε, b = ε²d. Then both cycle affinities vanish and every reaction is balanced at x = x*, y = εx* (checked to 1e-12). The start is a random smooth field around x* for X and around a value in [0.3, 3] for Y. Y relaxes from order 1 to about 1e-3 within the window, and σ falls from about 2000 to 0.02 to 0.3.

**Training.** The 48 driven training trajectories plus the 5 anchors (9 percent), and otherwise the protocol of `results/convex_training.md`: `fit()` with the closed-form step where the model has a linear part, then 2000 Adam iterations, lr 1e-2, batch 8, K in [4, 8], MPRK22 × 8 for every model. Models: HyperME wired (`c3_wired`), CRNN-chem, and Flux-GNN (a hypergraph network that outputs a forward and a backward rate per hyperedge, times mass action, with the four chemostat log concentrations as input features, no closed-form step). Seeds 0 and 1 for the first two, seed 0 for Flux-GNN.

**Metrics.** Median 10T error on test_id and test_ood (driven, as before). σ relative error along the rollout. The reverse/forward ratio log10(b/a) per reaction, median over the test_id states, for the model and for the truth.

## Results

| model | anchor | seed | ID 10T | OOD 10T | σ rel err ID ; OOD | median log10(b/a), model (true): R1 / R2 / R3 / R4 | final train loss |
|---|---|---|---|---|---|---|---|
| HyperME wired | no | 0 | 0.0026 | 0.0030 | 0.221 ; 0.283 | −0.51 (−3.21) / −6.63 (−3.63) / −2.04 (−2.78) / −5.48 (−2.79) | 1.10e-5 |
| HyperME wired | no | 1 | 0.0027 | 0.0036 | 0.222 ; 0.279 | −0.51 / −6.63 / −2.04 / −5.48 | 1.05e-5 |
| HyperME wired | yes | 0 | 0.0232 | 0.0074 | 0.281 ; 0.358 | −0.51 / −6.63 / **−2.79** / **−2.48** | 1.07e-5 |
| HyperME wired | yes | 1 | 0.0233 | 0.0075 | 0.280 ; 0.359 | −0.51 / −6.64 / **−2.79** / **−2.48** | 1.07e-5 |
| CRNN-chem | no | 0 | 0.0035 | 0.0033 | 0.233 ; 0.289 | −0.51 / −6.63 / −2.11 / −5.49 | 1.02e-5 |
| CRNN-chem | no | 1 | 0.0037 | 0.0029 | 0.231 ; 0.287 | −0.51 / −6.63 / −2.10 / −5.49 | 1.01e-5 |
| CRNN-chem | yes | 0 | 0.0230 | 0.0130 | 0.269 ; 0.345 | −0.51 / −6.53 / **−2.79** / **−2.49** | 1.07e-5 |
| CRNN-chem | yes | 1 | 0.0236 | 0.0134 | 0.271 ; 0.348 | −0.51 / −6.54 / **−2.79** / **−2.49** | 1.07e-5 |
| Flux-GNN | no | 0 | 0.259 | 0.477 | 2.04 ; 1.74 | −2.23 / −2.77 / −4.46 / −1.89 | 5.0e-4 |
| Flux-GNN | yes | 0 | 0.728 | 0.706 | 1.00 ; 1.00 | −8.78 / −9.37 / −8.03 / −8.31 | 1.2e-1 (collapsed) |

The true ratios are the same in every row: −3.21 / −3.63 / −2.78 / −2.79. The Flux-GNN anchor run collapsed near iteration 100: the gradient norm fell to 1e-18 and stayed there, the loss stayed at 0.12, and all its rates went to near zero, so σ ≈ 0. Flux-GNN computes σ from its own forward and backward rates, so the readout exists, but it is off by a factor of 2 to 3 even without the anchor.

**Where the σ error comes from** (share of the true σ, summed over the test_id states, wired model, seed 0):

| | R1 | R2 | R3 | R4 | transport | total / true |
|---|---|---|---|---|---|---|
| truth | 0.138 | 0.332 | 0.382 | 0.135 | 0.012 | 1 |
| no anchor | 0.015 | 0.688 | 0.302 | 0.126 | 0.012 | 1.14 |
| anchor | 0.015 | 0.687 | 0.390 | 0.061 | 0.012 | 1.17 |

The OOD set gives the same picture (no anchor 1.19, anchor 1.23).

**Lower bound from the driven data alone.** The derivative-fit residual is a convex quadratic in the class factors κ, and σ = Σ (a − b) log(a/b) is jointly convex in (a, b), which are linear in κ. So the smallest σ among all models of the class whose fit is within a factor 1 + tol of the best fit is a convex program, and its optimum is global (`sigma_bound`, SLSQP from the least-squares point). The minimiser is fitted on the training states and evaluated on the true test states.

| trained on | tol | residual best → at the minimiser | σ_min / σ_true ID ; OOD | true σ ≥ σ_min at (fraction of test points) ID ; OOD |
|---|---|---|---|---|
| driven data, wired (s0, s1) | 0.1 | 5.06e-3 → 5.56e-3 | 0.576 ; 0.618 | 1.000 ; 1.000 |
| driven data, wired (s0, s1) | 1.0 | 5.06e-3 → 1.01e-2 | 0.462 ; 0.497 | 1.000 ; 1.000 |
| driven + anchor, wired (s0, s1) | 0.1 | 5.72e-3 → 6.29e-3 | 0.697 ; 0.743 | 1.000 ; 1.000 |
| driven + anchor, wired (s0, s1) | 1.0 | 5.72e-3 → 1.14e-2 | 0.597 ; 0.635 | 1.000 ; 1.000 |
| driven data, CRNN-chem (s0 / s1) | 0.1 | 5.06e-3 → 5.56e-3 | 0.576 / 0.607 ; 0.618 / 0.643 | 1.000 ; 1.000 |

There is no upper bound. The best fit puts the reverse rates of R2 and R4 at zero, which gives σ = ∞. With the 1e-6 floor of `fit_linear` the best fit has 1.9 to 2.0 times the true σ.

## Reading

- **The anchor fixes what it can see.** With 9 percent anchor data, the wired model and CRNN-chem both recover the reverse/forward ratio of R3 (−2.79 against −2.78) and move R4 from −5.5 to −2.5 (true −2.8), in both seeds. The R3 share of σ goes from 0.30 to 0.39 (true 0.38).
- **σ does not improve, because two errors the anchor cannot see dominate.** R1 reverse (X → A) and R4 forward (X → E) both remove one X at a rate proportional to x, under every chemostat setting, driven or not. No concentration data can split them, and both models put a large share of the X decay on R1 reverse (ratio −0.51 against −3.21), so R1 carries 0.015 of σ instead of 0.138. The reverse of R2 balances its forward rate only when y ≈ εx, a thousand times below x. In the relative rollout loss those states are invisible, so the R2 ratio stays at −6.6 (true −3.6) and R2 carries 0.69 of σ instead of 0.33. Net effect: σ error 0.22 / 0.28 without the anchor, 0.28 / 0.36 with it.
- **The anchor costs density accuracy under this protocol.** ID 10T goes from 0.26 to 2.3 percent in both seeds, almost all in the homogeneous limit cycle (0.2 to 0.4 → 3.6 percent), and OOD 10T from 0.30 to 0.74 percent. The training loss is the same (1.1e-5). The class contains the truth for both conditions, so this is the shared 2000-iteration budget and the phase sensitivity of the limit cycle, not a conflict in the model.
- **What the driven data do support is a lower bound.** Among the models that fit the driven data within 10 percent of the best residual, the smallest σ is 0.58 (ID) and 0.62 (OOD) of the truth, and 0.46 / 0.50 at a 100 percent tolerance. The true σ lies above the bound at every test point. The anchor tightens the bound to 0.70 / 0.74. This is the honest readout for a nearly irreversible network: σ ≥ σ_min, with no finite upper end.
- **The black-box rates do not give a usable σ.** Flux-GNN reaches 26 / 48 percent at 10T and a σ error of 2.0 / 1.7 without the anchor. With the anchor at lr 1e-2 its training collapsed. Its anchor features (log chemostat concentrations from about −8 to 7) are far outside the driven range, and this one seed does not separate a bad step from a systematic failure.

## What would identify σ

Two things are missing, and neither is more relaxation data of this kind. (1) For R1 against R4 the split needs a measurement that sees A or E, for example the turnover of one reservoir, or the standard potentials of the reservoirs, which fix the A → E cycle affinity. (2) For R2 the loss has to weight Y near equilibrium, for example a per-species relative error. Both change the protocol or the data type, so they are not run here.

## Limits

Seeds 0 and 1 for the wired model and CRNN-chem, seed 0 for Flux-GNN. One anchor design: five trajectories, x* in [0.7, 1.5]. The σ bound depends on the tolerance, and the derivative residual is dominated by the Simpson error at Δt = 0.2, so the tolerance is not a statistical confidence level. Wired and CRNN runs used 4 threads, Flux-GNN 8, which does not change the arithmetic.
