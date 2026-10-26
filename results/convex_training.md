# Training from scratch without a curriculum: the closed-form linear step and zero-gated residuals

Code: `hyperme/linear_fit.py` (the estimator), `hyperme/model.py` (`WiredKinetics`, `WiredFreeEnergy`, `HyperME.fit_linear`, `HyperME.fit`), `hyperme/baselines.py` (`CRNN.fit_linear`, Patankar stepping), `hyperme/presets.py` (`schlogl_wired`, `crnn_chem`, `set_true_mass_action`), experiment driver `hyperme/convex_training.py`, tests `hyperme/tests/test_linear_fit.py`. Runs on machine 49 in `~/Fisher-Hodge/runs/hyperme/convex/` (CPU, float64).

## What changed in the model

**1. The rate constants enter linearly, so their fit is convex.** In the wired mode every one-way rate factors as a_h = K⁺_h(θ) Φ⁺_h(c), b_h = K⁻_h(θ) Φ⁻_h(c). Here log K is affine in θ = (log-rate table, standard potentials μ⁰, chemostat μ⁰), and Φ holds everything that is wired: log c, U/T, the force line integrals, the chemostat factors and the current non-ideal correction. Hence dc/dt = S (K⁺Φ⁺ − K⁻Φ⁻) is linear in the one-way rate constants. The one-way channels whose log K is the same function of θ form a class: one reaction type in one direction, both directions of one transport type, or one hyperedge when the core is per edge (`kinetics_core='edge'`, the S2 case). Integrating over the sampled training trajectories with the Simpson rule gives

    c(t + 2Δt) − c(t) = Σ_r κ_r S ∫ Φ_r dt,   κ_r ≥ 0,

a non-negative least-squares problem with a global optimum. The windows are weighted like the relative rollout loss. The optimum is mapped back to θ by least squares on log κ, which is exact when the classes have independent directions in θ (true for C2, C3 and C5: projection error 1e-14). `HyperME.fit(data, iters, dt)` runs this step and then the ordinary rollout training. `python -m hyperme.train` now does the same by default (`--no_linear_init` skips it). The CRNN-chem baseline with a log table of rate constants is linear in the same way and gets the same step. GNN-ODE and Flux-GNN have no such structure.

**2. Zero-gated residuals.** The non-ideal correction is δμ = (physical family) + s_μ r(c), with r_s(c_s) a separable, non-decreasing sum of 16 tanh units (the gradient of a convex local excess free energy) and one gate s_μ per species. The log rate is κ_h = core + s_k r_θ(features_h) with an MLP residual. The gates start at zero and the residual weights at a generic non-zero value. At initialisation the model equals its core, and only the gates receive a gradient. The test checks both facts. No code path freezes, schedules or stages parameters: the warm-start option of `hyperme/mapk.py` (one-step derivative matching before the rollout) is removed, and the library never had another one.

**3. The core is one log rate per hyperedge type.** Features enter only through the gated residual. Old checkpoints load: an MLP for κ becomes the residual with s_k = 1, and a table over the type one-hot becomes the core.

## Protocol (every model)

Seed 0. The closed-form step when the model has a linear part, then 2000 Adam iterations of the rollout loss (K in [4, 8], batch 8, cosine schedule from lr 1e-2 to 1e-5, gradient clip 1). Time stepping is MPRK22 with 8 steps per sampling interval for every model. Metrics are the median relative L2 error at T and 10T on `test_id` and `test_ood`, and the relative error of σ(t) over the test window. "After the linear step" is the model before any Adam iteration. Models:

- HyperME wired, C3: `c3_wired` (log-rate table, learned μ⁰ and chemostat μ⁰, chemostat prefactor, SQRA form, 13 parameters).
- HyperME wired, C2 and C5: `schlogl_wired`. One model for both benchmarks: log-rate table, learned μ⁰ and chemostat μ⁰, the gated monotone δμ residual, activity-form rates a_h = k_h exp(⟨ν⁺, μ⟩ + A_h/2) (the rate law of the earlier C5 HyperME in `hyper/crn2.py`), 56 parameters.
- HyperME wired + gated δμ residual, C3: `c3_wired(nonideal_residual=True)`, to check that the residual does no harm when the truth is ideal.
- CRNN-chem: independent forward and backward log rate constants per hyperedge type, chemostats as exact mass-action factors. This is the oracle class for C2 and C3 and the wrong class for C5.
- "none (neutral start)": the same model without the closed-form step, i.e. μ⁰ = 0 and every rate constant 1.

## Results (one seed)

Errors are fractions (0.003 = 0.3 percent). "Train loss" is the mean relative rollout loss over the 100 iterations ending at the given iteration. No rollout of any model went negative or blew up.

### C3 Brusselator (OOD = b in [3.2, 3.7], training b in [2.3, 3.0])

| model | linear step | after the linear step: ID 10T / OOD 10T | ID T | ID 10T | OOD T | OOD 10T | σ rel err ID ; OOD | train loss at it 100 / 500 / 1000 / 2000 | OOD period / amp rel err |
|---|---|---|---|---|---|---|---|---|---|
| HyperME wired | NNLS | 0.0334 / 0.0071 | 0.0054 | **0.0026** | 0.0048 | **0.0030** | 0.221 ; 0.283 | 5.6e-4 / 4.2e-5 / 5.8e-5 / 1.1e-5 | 0.002 / 0.030 |
| HyperME wired + gated δμ residual | NNLS | 0.0334 / 0.0071 | 0.0054 | 0.0032 | 0.0044 | 0.0032 | 0.224 ; 0.280 | 6.7e-4 / 1.0e-3 / 1.5e-5 / 1.1e-5 | 0.002 / 0.031 |
| CRNN-chem | NNLS | 0.0334 / 0.0071 | 0.0054 | 0.0035 | 0.0040 | 0.0033 | 0.233 ; 0.289 | 1.8e-4 / 1.8e-5 / 1.7e-5 / 1.0e-5 | 0.001 / 0.034 |
| HyperME wired | none (neutral start) | 0.509 / 0.581 | 0.196 | 0.217 | 0.245 | 0.400 | 0.856 ; 0.861 | 1.4e-1 / 2.2e-2 / 8.8e-3 / 5.7e-3 | 0.544 / 0.344 |
| CRNN-chem | none (neutral start) | 0.509 / 0.581 | 0.125 | 0.254 | 0.155 | 0.356 | 0.786 ; 0.807 | 1.4e-1 / 9.3e-3 / 3.7e-3 / 2.4e-3 | 0.267 / 0.215 |
| true constants, MPRK22 × 8 | – | – | 0.0056 | 0.0165 | 0.0068 | 0.0067 | 0.050 ; 0.121 | – | 0.007 / 0.018 |

Earlier runs on the same data (`results/c3b_summary.md`): the wired model from the neutral start, lr 1e-2, 0.217 / 0.400 (the new "none" row reproduces it to every digit); neutral start, lr 1e-1, 6000 iterations, 0.283 / 0.241; forward-biased start, lr 2e-2, 0.169 / **0.123**, σ 0.51 ; 0.55; true constants plus noise 0.3, lr 1e-2, 0.0175 / 0.0100, σ 0.063 ; 0.042; the prototype CRNN-chem (softplus MLP rates, explicit substeps, lr 1e-3) 0.063 / 0.025.

### C2 Schlögl (OOD = a in [3.15, 3.35], b in [0.3, 1.0])

| model | linear step | after the linear step: ID 10T / OOD 10T | ID T | ID 10T | OOD T | OOD 10T | σ rel err ID ; OOD | train loss at it 100 / 500 / 1000 / 2000 |
|---|---|---|---|---|---|---|---|---|
| HyperME wired | NNLS | 0.00020 / 0.00016 | 0.00017 | **0.00079** | 0.00061 | **0.00076** | 0.002 ; 0.003 | 5.1e-5 / 1.1e-8 / 1.2e-8 / 8.2e-9 |
| CRNN-chem | NNLS | 0.00020 / 0.00016 | 0.00018 | 0.00048 | 0.00059 | 0.00042 | 0.002 ; 0.003 | 4.5e-5 / 2.9e-8 / 1.4e-8 / 8.5e-9 |
| HyperME wired | none (neutral start) | 2.87 / 6.22 | 0.165 | 0.188 | 0.631 | 0.553 | 0.895 ; 0.968 | 2.0e-2 / 3.6e-3 / 2.0e-3 / 1.7e-3 |
| CRNN-chem | none (neutral start) | 2.87 / 6.22 | 0.018 | 0.014 | 0.032 | 0.018 | 0.092 ; 0.107 | 3.5e-2 / 7.1e-4 / 1.4e-4 / 5.2e-5 |
| true constants, MPRK22 × 8 | – | – | 0.00076 | 7e-10 | 0.00091 | 2e-11 | 0.002 ; 0.002 | – |

Earlier runs (prototype `hyper/crn2.py`, explicit substeps, lr 1e-3, `hyper/runs/C3b_summary.md`): HyperME-chem-cvx 2000 iterations 0.50 / 0.83 (s0), σ 0.31 ; 0.41; 6000 iterations 0.087 / 0.237; B1 CRNN-chem 0.0079 / 0.0069.

### C5 non-ideal Schlögl (crowding, ln γ = φx/(1 − φx), φ = 0.25; OOD = a in [3.5, 4.0], b in [2.0, 3.5])

| model | linear step | after the linear step: ID 10T / OOD 10T | ID T | ID 10T | OOD T | OOD 10T | σ rel err ID ; OOD | train loss at it 100 / 500 / 1000 / 2000 | δμ vs ln γ: RMSE ID / slope |
|---|---|---|---|---|---|---|---|---|---|
| HyperME wired | NNLS | 0.144 / 0.499 | 0.0028 | **0.0025** | 0.036 | **0.036** | 0.012 ; 0.011 | 1.4e-3 / 9.2e-6 / 3.3e-6 / 1.6e-6 | 0.009 / 1.01 |
| CRNN-chem (wrong class) | NNLS | 0.144 / 0.499 | 0.071 | 0.167 | 0.519 | 0.519 | 0.359 ; 0.357 | 1.7e-3 / 1.5e-3 / 1.6e-3 / 1.6e-3 | – |
| HyperME wired | none (neutral start) | 3.24 / 0.778 | 0.135 | 0.227 | 0.397 | 0.295 | 0.973 ; 0.994 | 1.5e-1 / 1.9e-2 / 7.3e-3 / 4.4e-3 | 0.651 / −1.56 |
| CRNN-chem | none (neutral start) | 3.24 / 0.778 | 0.112 | 0.256 | 1.08 | 1.15 | 0.480 ; 0.402 | 1.0e-1 / 8.7e-3 / 4.2e-3 / 4.0e-3 | – |
| true constants of the ideal law | – | – | 0.152 | 0.231 | 1.72 | 1.83 | 0.392 ; 0.145 | – | – |

The C5 model is the C2 model; nothing about the crowding is wired. Earlier runs (prototype, explicit substeps, lr 1e-3, `hyper/runs/C5_summary.md`): HyperME-chem-cvx from scratch 2000 iterations 0.70 / 0.48, σ 0.99 ; 1.00 (reactions off); warm start from the C2 fit, 2000 iterations, 0.10 / 0.37; warm start, 6000 iterations, 0.0067 / 0.115, σ 0.008 ; 0.009, ln γ RMSE 0.015, slope 0.97; B1 CRNN-chem 0.155 / 0.98; B2 GNN-ODE 0.013 / 0.196.

## Reading

- **The closed-form step reaches the exact point of the class.** On C2 the derivative residual drops from 7.6 to 1.8e-7 and the model is at 0.02 percent before any gradient step. On C3 it drops from 403 to 5e-3 and the model is at 0.71 percent OOD, the error of the true constants under the same integrator (0.67 percent). On C5 it gives the best ideal fit (residual 0.08, 14 and 50 percent), which is the right starting point for the non-ideal correction. The step takes 0.2 to 1.7 s.
- **C3: 12.3 percent from scratch becomes 0.30 percent.** After the 2000 Adam iterations the wired model reaches 0.26 percent ID and 0.30 percent OOD, the period error 0.2 percent. That is below the true constants under MPRK22 × 8 (1.65 and 0.67 percent): the model is trained through the same integrator and absorbs part of its phase error. The same model without the closed-form step stays at 22 and 40 percent, exactly the earlier neutral-start run. The δμ residual leaves C3 unchanged (0.32 percent), so one architecture serves ideal and non-ideal data.
- **C5: the reactions-off minimum is gone.** With the closed-form step the C5 model reaches 0.25 percent ID and 3.6 percent OOD, and the learned correction matches ln γ (RMSE 0.009, slope 1.01, gate s_μ = 0.81). That beats the earlier warm start with 6000 iterations (0.67 and 11.5 percent) at a third of the iterations. Without the step the same model falls into the earlier minimum: the reaction fluxes are 0.2 instead of 6.7, the diffusion constant rises about 14 times, σ is 14 against a true 1035, and the gate turns negative (s_μ = −0.83, a non-convex excess free energy). The zero gate alone does not prevent the trap. The convex step does, because the reactions-off solution is not the least-squares optimum.
- **The baseline gets the same step and the same gain where its class is right.** CRNN-chem with the closed-form step reaches 0.35 / 0.33 percent on C3 and 0.05 / 0.04 percent on C2, level with the wired model. On C5 it stops at 17 / 52 percent with σ error 0.36, the bias of pure mass action. From the neutral start CRNN-chem is better than the wired model on C2 (1.4 against 19 percent), because its forward and backward constants are not coupled through μ⁰. With the closed-form step this difference disappears.
- **Adam at lr 1e-2 first moves away from the optimum.** The C2 loss rises from 6e-8 to 5e-5 in the first 100 iterations and returns to 1e-8 by iteration 300. The final C2 error, 0.08 percent, is above the 0.02 percent of the closed-form step alone. On C3 and C5 training improves on the step.
- **σ on C3 is limited by identifiability, not by the fit.** The reverse of R1 (X → A) and the forward of R4 (X → E) are both X → ∅ at a rate proportional to x, so concentration data cannot split them. The least-squares step splits them evenly (0.49 / 0.49 of the true total) and puts the reverse rates of R2 and R4 (true 1e-3) at the floor. After training the A → E cycle affinity is 13.80 (true 13.81) and the B → D affinity 20.0 (true 14.8), and the σ error is 0.22 / 0.28 against 0.05 / 0.12 for the true constants. This is the parallel-channel blind spot of `results/theory_notes.md`. The match of the first affinity is not evidence that it is identified.

## Limits

One seed. The learning rate 1e-2 was set once for all models, from the earlier C3 runs, and not tuned on test data. GNN-ODE and Flux-GNN were not rerun: they have no linear part, and their earlier numbers used explicit integrators. The C2 and C5 reference numbers from the prototype used explicit substeps, lr 1e-3, and a separately parameterised model. The S2 per-edge core is covered by a unit test (recovery of 32 random edge rates on a 4 × 4 torus), not by a benchmark run.
