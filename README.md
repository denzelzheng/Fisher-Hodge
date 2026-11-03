# Neural Chart

The Neural Chart learns the constitutive law of a driven physical system from trajectories.
It receives the state, the graph of reaction or transport steps, and the conditions of each trajectory: the temperature, an external potential and drive, and the chemostat levels.
It returns a master equation on that graph that satisfies the second law for every value of its parameters.

**Contents**

- [The idea](#the-idea)
- [What is learned in each benchmark](#what-is-learned-in-each-benchmark)
- [Results at a glance](#results-at-a-glance)
- [Learned constitutive laws under new conditions](#learned-constitutive-laws-under-new-conditions): porous media, porous media with crowding, non-ideal Schlögl model, phosphorylation cycle
- [Ideal reaction networks](#ideal-reaction-networks): Schlögl model, Brusselator, MAPK cascade
- [How each part contributes](#how-each-part-contributes)
- [Scope and limits](#scope-and-limits)
- [Reproduce](#reproduce)
- [Repository layout](#repository-layout)

## The idea

Fix the graph.
Each step $h$ of the graph (a reaction, or a transfer between two cells) runs forward at rate $a_h$ and backward at rate $b_h$.
Write the two rates in the coordinates

$$f_h=\log\frac{a_h}{b_h},\qquad g_h=\sqrt{a_h b_h}.$$

Local detailed balance says that $f_h$ is the thermodynamic force of the step, $f_h=-\langle S_h,\mu\rangle+A_h$.
It is set by the chemical potentials $\mu$ and the drive $A_h$, and so by the state and the conditions.
The rate laws that satisfy local detailed balance therefore form a manifold.
On it the force coordinate is a known function of the state and the conditions.
The other coordinate is free, $g_h=k_h\varepsilon_h$, where $\varepsilon_h$ is a known mass-action factor and $k_h>0$ is the mobility of the step.
The mobility is a property of the material.

A Neural Chart is a parametrisation of this manifold.
Physics supplies the force.
A network supplies the mobility from what the step sees,

$$\log k_h=\log\omega_{c(h)}+g_\theta(\xi_h)+s\,m(\bar c_h),$$

with one constant $\omega_c$ per class of steps, an MLP $g_\theta$ of local material descriptors $\xi_h$, and a monotone term $m$ of the local state $\bar c_h$.
On reaction networks the chemical potential $\mu=\log c+U/T+\mu^0_\theta/T$ has learned standard potentials $\mu^0_\theta$.
On non-ideal mixtures it also has a learned activity correction $\ln\gamma_\theta(c)$.

What the model learns is therefore a constitutive functional: a map from material descriptors and local state to mobilities and chemical potentials.
The dynamics follow from it through the master equation $\dot c=\sum_h S_h\,(a_h-b_h)$.
Three properties hold for every parameter value.

1. **Second law.** The flux of every step runs along the model's force, so the entropy production is non-negative. Local detailed balance, conservation of every moiety and positivity hold as well. No penalty or projection is needed.
2. **Conditions are coordinates.** The temperature, the drive and the chemostat levels enter only through $f$, and the mobility never reads them. A new condition moves the model along a coordinate it already has. No learned part has to extrapolate in the condition.
3. **Convex start.** With the force fixed, the flux $J_h=k_h\varphi_h$ is linear in the mobility. At initialisation the class constants $\omega_c$ and the standard potentials enter the log-rates linearly, and their fit to the data is a non-negative least-squares problem with a global optimum. Training on rollouts starts there.

[docs/theory.md](docs/theory.md) proves these statements (Theorem 0).
It also shows that gray-box models whose learned part reads the conditions, or learns the force freely, lose them.
[docs/method.md](docs/method.md) gives the fit and the evaluation protocol.

## What is learned in each benchmark

The same code runs every benchmark.
Each benchmark changes only the graph, the inputs that exist (descriptors, local state, potential, chemostats) and the training data.

| Benchmark | System | Learned part | Parameters |
|---|---|---|---|
| Porous media | drift and diffusion of a density through random porous media | mobility of each edge as a function of 12 porosity and connectivity descriptors | 5,381 |
| Porous media with crowding | the same, with a mobility that falls with the local density | mobility as a function of the descriptors and the local density | 5,381 |
| Non-ideal Schlögl model | bistable reaction and diffusion with crowded activities | activity correction $\ln\gamma(c)$, standard potentials, rate constants | 55 |
| Phosphorylation cycle | an ATP-driven cycle with diffusion in an external potential | standard potentials, rate constants | 9 |
| Schlögl model | ideal bistable reaction and diffusion | standard potentials, rate constants | 6 |
| Brusselator | ideal reaction and diffusion with oscillations and Turing patterns | standard potentials, rate constants | 12 |
| MAPK cascade | ideal enzyme cascade with 22 species and 20 reactions | standard potentials, rate constants | 48 |

In the first three benchmarks the learned part is a function of the material or of the state.
In the phosphorylation cycle the test is the drive and the potential, which enter as coordinates.
The last three are ideal mass action.
There the mobility reduces to a set of constants, and the question is whether the model recovers them as well as a mechanistic fit does.

## Results at a glance

Every model is trained once, with one seed, under the training conditions, and is then evaluated on every test set without retraining.
The error is the relative L2 error of the state at 10T (200 sampling steps), median over the test trajectories.
The best baseline is the most accurate baseline in that column.

| Benchmark | training conditions: Neural Chart | best baseline | hardest new condition | Neural Chart | best baseline |
|---|---|---|---|---|---|
| Porous media | **0.064** | 0.077 (UDE closure) | temperature ×4 | **0.090** | 0.472 (HelmFluid) |
| Porous media with crowding | 0.067 | **0.057** (Flux-GNN) | drive ×8 | **0.376** | 0.917 (UDE closure) |
| Non-ideal Schlögl model | **3.4e-04** | 1.8e-03 (Flux-GNN) | higher concentrations | **0.010** | 0.033 (FNO) |
| Phosphorylation cycle | **5.9e-06** | 2.1e-03 (Flux-GNN) | stronger, finer potential | **5.9e-06** | 0.011 (CRNN, condition features) |
| Schlögl model | **1.8e-06** | 4.1e-06 (CRNN, mass action) | new chemostat levels | **1.2e-06** | 3.7e-06 (CRNN, mass action) |
| Brusselator | 2.6e-03 | **2.4e-03** (CRNN, mass action) | both chemostats outside training | **3.7e-03** | 4.1e-03 (CRNN, mass action) |
| MAPK cascade | 6.6e-03 | **6.3e-03** (Flux-GNN) | ATP 0.15–0.3 | **0.011** | 0.020 (CRNN, mass action) |

On every benchmark and every test set no step of the Neural Chart carries flux against its own force, no sample has negative entropy production, and no concentration becomes negative.

The baselines are:
- the UDE closure, a free neural flux on the same graph
- Flux-GNN, a message-passing network with positive conservative fluxes
- CRNN with mass action, which is given the same graph and fits free forward and backward rate constants
- CRNN with the conditions as features
- BSMS-GNN, FNO, GENERIC-FNO, and HelmFluid on porous media only
- persistence, $u(t)=u(0)$

The physics tables count, for every model with its own edge flux, the steps whose flux runs against the true physical force, $J_hf_h<-10^{-6}(a_h+b_h)|f_h|$, and the samples with $\sigma<0$.
Models that return only a state update have N/A there.
The oracle is the Neural Chart class with the true rates, so its error is the discretisation and integration floor.
Full tables, including training times, are in `results/<benchmark>/`.
In the animations negative values are magenta, and the panel title counts the negative cells.

## Learned constitutive laws under new conditions

### Porous media

A density moves by diffusion and a divergence-free drive through random porous media.
The true mobility comes from an effective-medium rule on a 128×128 pore map that the models never see.
The models see 12 local porosity and connectivity descriptors per edge.
They are trained on 64 media at temperature 1 and drive amplitude 1 on a 32×32 grid.
The tests use 16 unseen media, other temperatures and drives, unseen low-porosity media (porosity 0.15–0.3, training 0.3–0.7), a 64×64 grid and a point cloud of 1024 Voronoi cells.

| Drive ×8 on an unseen medium | Point cloud (training conditions) |
|---|---|
| ![porous media, drive x8](assets/porous_media_drive_x8.gif) | ![porous media, point cloud](assets/porous_media_point_cloud.gif) |

| Method | training conditions | temperature ×0.25 | temperature ×4 | drive ×4 | drive ×8 | unseen low-porosity media | 64×64 grid | point cloud | parameters |
|---|---|---|---|---|---|---|---|---|---|
| **Neural Chart** | **0.064** | **0.084** | **0.090** | **0.195** | **0.407** | **0.081** | **0.071** | **0.090** | 5,381 |
| Oracle (true rates in the Neural Chart class) | 0.046 | 0.107 | 0.032 | 0.143 | 0.231 | 0.037 | 0.034 | 0.083 | 0 (true rates) |
| UDE closure | 0.077 | 0.700 | 1.023 | 0.588 | 0.811 | 0.104 | 0.093 | 0.255 | 5,441 |
| Flux-GNN | 0.078 | 0.797 | 2.138 | 1.046 | 1.091 | 0.116 | 0.515 | 0.458 | 151,234 |
| BSMS-GNN | 0.736 | 0.911 | 2.430 | 2.945 | 3.050 | 0.737 | 1.079 | 0.915 | 1,724,417 |
| FNO | 0.510 | 0.894 | 2.256 | 2.544 | 7.262 | 0.460 | 0.516 | N/A | 1,188,609 |
| GENERIC-FNO | 1.852 | 1.210 | 2.562 | 1.272 | 1.146 | 1.021 | 1.428 | N/A | 1,002,388 |
| HelmFluid | 0.846 | 1.154 | 0.472 | 0.930 | 0.976 | 0.946 | N/A | N/A | 9,930,381 |
| Persistence (u(t) = u(0)) | 1.603 | 1.118 | 2.256 | 1.238 | 1.115 | 0.696 | 1.607 | 1.616 | 0 |

| Method | edges with flux against the true force | samples with σ < 0 | negative values |
|---|---|---|---|
| Neural Chart | 0 | 0 | 0 |
| Oracle | 0 | 0 | 0 |
| UDE closure | 0.688 | 0.882 | 0.236 |
| Flux-GNN | 0.311 | 0 | 0 |
| BSMS-GNN | N/A | N/A | 0.518 |
| FNO | N/A | N/A | 0.512 |
| GENERIC-FNO | N/A | N/A | 0.335 |
| HelmFluid | N/A | N/A | 0.033 |

Under the training conditions the UDE closure and Flux-GNN are close to the Neural Chart (0.077 and 0.078 against 0.064).
Under changed temperature and drive their errors grow to 0.6–2.1.
The Neural Chart stays between 0.08 and 0.41, within a factor of 3 of the oracle.
The UDE closure produces negative densities on all 64 test trajectories.

**The learned law.** Mapped back to space, the learned mobility matches the true mobility of unseen media (`results/porous_media/law.md`).
The Pearson r of log k over edges is 0.994 on the training grid and on the 64×64 grid, and 0.992 on the low-porosity media.
The descriptors are local, so the same functional applies to the point cloud without retraining.

![Learned and true mobility](assets/medium_and_law.png)

**Flows.** The animations below follow one test trajectory in an unseen medium.
Tracer particles move with the probability velocity $v=J/\rho$ of each model's own edge flux, so only models with an edge flux appear.
A particle turns magenta in a cell where the local entropy production is negative or where an edge carries flux against the physical force.

![Tracers, drive x8](assets/porous_media_track_drive_x8.gif)

Under drive ×8 the tracers of the Neural Chart follow the reference, and no cell is flagged.
Flux-GNN flags about a third of the cells and the UDE closure about two thirds.

![Local entropy production, drive x8](assets/porous_media_entropy_drive_x8.gif)

The local entropy production of the same scene stays non-negative everywhere for the Neural Chart.
The UDE closure has negative entropy production in about half of the cells.

![Tracers, point cloud](assets/porous_media_track_point_cloud.gif)

On a random point cloud of 1024 cells, with the parameters trained on the 32×32 grid, the flow of the Neural Chart again follows the reference.
More animations are in the [gallery](results/porous_media/gallery/).

### Porous media with crowding

The same task with a crowded mobility $M(x)/(1+\bar\rho_e/2)$, where $\bar\rho_e$ is the mean density of the two cells of the edge.
The model is not given the crowding form.
It reads $\bar\rho_e$ through its monotone state term.

| Drive ×8 on an unseen medium | Point cloud (training conditions) |
|---|---|
| ![porous media with crowding, drive x8](assets/porous_media_crowding_drive_x8.gif) | ![porous media with crowding, point cloud](assets/porous_media_crowding_point_cloud.gif) |

| Method | training conditions | temperature ×0.25 | temperature ×4 | drive ×4 | drive ×8 | unseen low-porosity media | 64×64 grid | point cloud | parameters |
|---|---|---|---|---|---|---|---|---|---|
| **Neural Chart** | 0.067 | **0.114** | **0.101** | **0.168** | **0.376** | 0.062 | **0.077** | **0.091** | 5,381 |
| Oracle (true rates in the Neural Chart class) | 0.049 | 0.069 | 0.033 | 0.103 | 0.197 | 0.028 | 0.031 | 0.079 | 0 (true rates) |
| UDE closure | 0.066 | 0.679 | 1.916 | 0.601 | 0.917 | 0.075 | 0.756 | 0.485 | 5,441 |
| Flux-GNN | **0.057** | 0.808 | 2.053 | 0.856 | 1.049 | **0.057** | 0.573 | 0.473 | 151,234 |
| BSMS-GNN | 0.592 | 0.837 | 2.739 | 1.386 | 1.355 | 0.435 | 1.008 | 0.713 | 1,724,417 |
| FNO | 0.464 | 0.898 | 2.370 | 1.897 | 3.316 | 0.386 | 0.469 | N/A | 1,188,609 |
| GENERIC-FNO | 1.158 | 0.924 | 1.768 | 1.098 | 1.054 | 0.371 | 1.206 | N/A | 1,002,388 |
| HelmFluid | 14.957 | 7.914 | 284.284 | 199.367 | 149.037 | 14.371 | N/A | N/A | 9,930,381 |
| Persistence (u(t) = u(0)) | 1.363 | 0.989 | 2.161 | 1.156 | 1.083 | 0.447 | 1.360 | 1.392 | 0 |

| Method | edges with flux against the true force | samples with σ < 0 | negative values |
|---|---|---|---|
| Neural Chart | 0 | 0 | 0 |
| Oracle | 0 | 0 | 0 |
| UDE closure | 0.808 | 0.999 | 0.194 |
| Flux-GNN | 0.304 | 0 | 0 |
| BSMS-GNN | N/A | N/A | 0.704 |
| FNO | N/A | N/A | 0.560 |
| GENERIC-FNO | N/A | N/A | 0.279 |
| HelmFluid | N/A | N/A | 0.038 |

Under the training conditions and on the unseen low-porosity media Flux-GNN is as accurate as the Neural Chart or slightly better (0.057 against 0.067 and 0.062).
Under changed temperature, drive and discretisation the Neural Chart is 2.4 to 17.5 times more accurate than the best baseline of each column.
Without the state input the Neural Chart reaches 0.241 under the training conditions.
This equals the error of the oracle without the crowding factor, so the state term is what captures the crowding.
The learned mobility matches the true crowded mobility with a flux-weighted relative error of 0.11 on unseen media (`results/porous_media_crowding/law.md`).

### Non-ideal Schlögl model

The Schlögl model $A+2X\rightleftharpoons3X$, $X\rightleftharpoons B$ on an 8×8 periodic grid, with mass action in the activity $y=x\gamma(x)$, $\ln\gamma(x)=\Phi x/(1-\Phi x)$, $\Phi=0.25$.
The Neural Chart learns $\ln\gamma$ as a monotone correction of the chemical potential, together with the standard potentials and the rate constants.
In the closed-form start the rate constants are projected out, so $\ln\gamma$ is fitted against the least-squares optimum of the rest.

![Non-ideal Schlögl model, higher concentrations](assets/schlogl_nonideal_high_concentration.gif)

| Method | training conditions | a 3.5–4.0, b 2.0–3.5 (higher concentrations) | 16×16 grid | parameters |
|---|---|---|---|---|
| **Neural Chart** | **3.4e-04** | **0.010** | **3.1e-04** | 55 |
| Neural Chart with given standard potentials | 5.3e-04 | 0.012 | 4.0e-04 | 52 |
| Oracle (true rates in the Neural Chart class) | 1.4e-06 | 1.1e-07 | 1.9e-06 | 0 (true rates) |
| CRNN (mass action) | 0.060 | 0.726 | 0.062 | 6 |
| CRNN (condition features) | 0.026 | 0.340 | 0.035 | 4,674 |
| UDE closure | 0.046 | 0.645 | 0.046 | 4,737 |
| Flux-GNN | 1.8e-03 | 0.265 | 2.7e-03 | 150,850 |
| FNO | 2.6e-03 | 0.033 | 2.4e-03 | 172,545 |
| BSMS-GNN | 0.011 | 0.411 | 0.054 | 1,392,001 |
| GENERIC-FNO | 1.117 | 0.565 | 1.060 | 1,002,004 |
| Persistence (u(t) = u(0)) | 1.117 | 0.565 | 1.060 | 0 |

| Method | hyperedges with flux against the true force | samples with σ < 0 | negative values |
|---|---|---|---|
| Neural Chart | 2.2e-05 | 0 | 0 |
| Oracle | 0 | 0 | 0 |
| CRNN (mass action) | 0.253 | 0 | 0 |
| CRNN (condition features) | 0.395 | 0.604 | 0 |
| UDE closure | 0.195 | 0.292 | 0 |
| Flux-GNN | 0.517 | 0.890 | 0 |
| FNO | N/A | N/A | 9.5e-05 |
| BSMS-GNN | N/A | N/A | 0 |
| GENERIC-FNO | N/A | N/A | 0 |

CRNN with mass action has the right graph but cannot represent the non-ideal activity, and it reaches 0.73 at the higher concentrations.
FNO is the closest baseline (0.033 against 0.010 at the higher concentrations) and produces negative concentrations on two trajectories.
The learned $\ln\gamma$ has slope 0.998 against the truth over the training range, and its RMSE grows from 3.6e-3 there to 0.025 at the higher concentrations (`results/schlogl_nonideal/activity_coefficient.md`).
The entropy production read from the model has a relative error of 1.6e-3 to 0.022, and the cycle affinity matches the reference to three digits (`results/schlogl_nonideal/thermodynamic_readouts.md`).
The few hyperedges counted against the true force (2e-5) sit where that force is close to zero.

### Driven phosphorylation cycle

A substrate $X$ is phosphorylated by an immobile enzyme, $X+E+\mathrm{ATP}\rightleftharpoons X_p+E+\mathrm{ADP}$, and dephosphorylated, $X_p\rightleftharpoons X$, on an 8×8 periodic grid.
$X$ and $X_p$ diffuse in a smooth potential $U$.
The free energy of ATP hydrolysis $\Delta G$ drives a net current around the cycle.
The test sets change the drive, the potential and the grid.
All three enter the Neural Chart only through the force.

![Phosphorylation cycle, strong drive](assets/phosphorylation_cycle_strong_drive.gif)

The animation shows the phosphorylated fraction $X_p/(X+X_p)$.

| Method | training conditions (ΔG 1–3) | drive ΔG 3–4 | stronger, finer potential U | 16×16 grid | parameters |
|---|---|---|---|---|---|
| **Neural Chart** | **5.9e-06** | **6.5e-06** | **5.9e-06** | **8.2e-06** | 9 |
| Neural Chart with given standard potentials | 3.3e-06 | 2.3e-06 | 2.6e-06 | 3.4e-06 | 4 |
| Oracle (true rates in the Neural Chart class) | 2.0e-06 | 1.7e-06 | 1.4e-06 | 1.9e-06 | 0 (true rates) |
| CRNN (mass action) | 0.017 | 0.018 | 0.021 | 0.025 | 10 |
| CRNN (condition features) | 5.2e-03 | 0.045 | 0.011 | 7.6e-03 | 4,866 |
| UDE closure | 0.018 | 0.057 | 0.130 | 0.026 | 5,121 |
| Flux-GNN | 2.1e-03 | 0.038 | 0.113 | 4.1e-03 | 151,170 |
| FNO | 0.022 | 0.050 | 0.600 | 0.534 | 172,867 |
| BSMS-GNN | 0.180 | 0.189 | 0.449 | 0.350 | 1,392,515 |
| GENERIC-FNO | 1.049 | 0.967 | 1.034 | 0.904 | 1,005,172 |
| Persistence (u(t) = u(0)) | 0.836 | 0.909 | 0.917 | 0.795 | 0 |

| Method | hyperedges with flux against the true force | samples with σ < 0 | negative values |
|---|---|---|---|
| Neural Chart | 0 | 0 | 0 |
| Oracle | 0 | 0 | 0 |
| CRNN (mass action) | 0.068 | 0 | 0 |
| CRNN (condition features) | 0.222 | 0 | 0 |
| UDE closure | 0.405 | 0 | 0.032 |
| Flux-GNN | 0.140 | 0 | 0 |
| FNO | N/A | N/A | 0.037 |
| BSMS-GNN | N/A | N/A | 0.191 |
| GENERIC-FNO | N/A | N/A | 0.109 |

The Neural Chart stays within a factor of 5 of the oracle on every test set, including the stronger drive and the new potential.
The best baseline, Flux-GNN, is at 2e-3 under the training conditions and at 0.11 in the new potential.
CRNN with mass action has the right graph but one rate constant per class of steps, so it cannot represent the spatial potential, and it stays near 0.02.
The entropy production read from the model has a relative error of 1.3e-3 to 2.5e-3, and the cycle affinity is 5.945 against 5.960 under the training conditions and 7.462 against 7.477 under the stronger drive (`results/phosphorylation_cycle/thermodynamic_readouts.md`).

## Ideal reaction networks

In these three benchmarks the truth is ideal mass action.
The learned part of the Neural Chart reduces to standard potentials and rate constants.
CRNN with mass action is given the same graph and the same number of constants, so it is the fair comparison.

### Schlögl model: bistability and travelling fronts

The Schlögl model $A+2X\rightleftharpoons3X$, $X\rightleftharpoons B$ with chemostats $A$, $B$ on an 8×8 periodic grid, with diffusing $X$.
It is bistable, and a planar front between the two states travels with a speed whose sign changes at the Maxwell point.

![Schlögl model, travelling front](assets/schlogl_front.gif)

| Method | training conditions | a 3.15–3.35, b 0.3–1.0 | planar fronts, a = 3, b 1.7–2.3 | 16×16 grid | parameters |
|---|---|---|---|---|---|
| **Neural Chart** | **1.8e-06** | **1.2e-06** | 2.3e-06 | **7.9e-07** | 6 |
| Neural Chart with given standard potentials | 2.2e-06 | 1.7e-06 | **2.0e-06** | 2.1e-06 | 3 |
| Oracle (true rates in the Neural Chart class) | 2.0e-06 | 1.7e-06 | 2.0e-06 | 2.1e-06 | 0 (true rates) |
| CRNN (mass action) | 4.1e-06 | 3.7e-06 | 5.8e-06 | 3.5e-06 | 6 |
| CRNN (condition features) | 0.059 | 1.405 | 0.073 | 0.010 | 4,674 |
| UDE closure | 0.123 | 0.923 | 0.057 | 0.172 | 4,737 |
| Flux-GNN | 0.018 | 0.943 | 6.4e-03 | 1.4e-03 | 150,850 |
| FNO | 0.025 | 0.952 | 8.6e-03 | 0.014 | 172,545 |
| BSMS-GNN | 0.022 | 1.037 | 0.011 | 0.056 | 1,392,001 |
| GENERIC-FNO | 2.246 | 4.569 | 0.581 | 0.659 | 1,002,004 |
| Persistence (u(t) = u(0)) | 2.246 | 4.569 | 0.581 | 0.659 | 0 |

| Method | hyperedges with flux against the true force | samples with σ < 0 | negative values |
|---|---|---|---|
| Neural Chart | 0 | 0 | 0 |
| Oracle | 0 | 0 | 0 |
| CRNN (mass action) | 0 | 0 | 0 |
| CRNN (condition features) | 0.164 | 0 | 0 |
| UDE closure | 0.456 | 0.835 | 0.043 |
| Flux-GNN | 0.241 | 0.235 | 0 |
| FNO | N/A | N/A | 7.2e-03 |
| BSMS-GNN | N/A | N/A | 0.240 |
| GENERIC-FNO | N/A | N/A | 0 |

The Neural Chart and CRNN with mass action are both at the floor of the integrator, which the oracle shows at about 2e-6.
The learned standard potentials give the cycle affinity of the reference to four digits, and the entropy production has a relative error of 1e-5 (`results/schlogl/thermodynamic_readouts.md`).
The black boxes reach 1e-3 to 1e-2 in the training ranges and fail at the new chemostat levels, with errors near 1.
GENERIC-FNO equals persistence on this task: its instance normalisation removes the spatial mean of every update, and the update collapses to zero.

### Brusselator: oscillations and Turing patterns

The Brusselator $A\to X$, $2X+Y\to3X$, $B+X\to Y+D$, $X\to E$ with chemostats $A$, $B$, $D$, $E$ on a 32×32 periodic grid, with diffusing $X$ and $Y$.
Depending on the chemostat levels it shows a limit cycle or Turing patterns.

| Turing pattern formation (a = 1.90, b = 3.90) | Oscillation, both chemostats outside the training range (a = 1.39, b = 4.30) |
|---|---|
| ![Brusselator, Turing pattern](assets/brusselator_turing.gif) | ![Brusselator, oscillation](assets/brusselator_oscillation.gif) |

| Method | training conditions | b 3.2–3.7 | b 4.0–4.6 | a 1.3–1.5 (a and b outside training) | Turing regime (below the Hopf line) | 64×64 grid | parameters |
|---|---|---|---|---|---|---|---|
| **Neural Chart** | 2.6e-03 | **3.0e-03** | **3.4e-03** | **3.7e-03** | **2.1e-03** | **0.013** | 12 |
| Neural Chart with given standard potentials | 1.6e-04 | 4.7e-05 | 3.7e-05 | 1.3e-04 | 3.8e-05 | 1.7e-04 | 6 |
| Oracle (true rates in the Neural Chart class) | 1.4e-04 | 5.6e-05 | 2.6e-05 | 1.2e-04 | 6.4e-06 | 7.1e-05 | 0 (true rates) |
| CRNN (mass action) | **2.4e-03** | 3.0e-03 | 3.4e-03 | 4.1e-03 | 0.026 | 0.013 | 12 |
| CRNN (condition features) | 0.724 | 0.755 | 0.771 | 0.802 | 0.698 | 0.749 | 4,866 |
| UDE closure | 0.550 | 0.622 | 0.689 | 1.004 | 1.408 | 0.573 | 4,929 |
| Flux-GNN | 0.075 | 0.499 | 0.639 | 0.638 | 0.292 | 0.201 | 151,170 |
| BSMS-GNN | 0.729 | 0.805 | 0.687 | 0.858 | 1.849 | 0.761 | 1,723,522 |
| FNO | 0.297 | 0.370 | 0.678 | 0.639 | 0.321 | 0.697 | 1,188,514 |
| GENERIC-FNO | 0.780 | 0.835 | 1.016 | 0.762 | 0.580 | 0.825 | 1,003,588 |
| Persistence (u(t) = u(0)) | 0.780 | 0.835 | 1.016 | 0.762 | 0.580 | 0.825 | 0 |

| Method | hyperedges with flux against the true force | samples with σ < 0 | negative values |
|---|---|---|---|
| Neural Chart | 0.022 | 0 | 0 |
| Oracle | 0 | 0 | 0 |
| CRNN (mass action) | 0.067 | 0 | 0 |
| CRNN (condition features) | 0.311 | 0 | 0 |
| UDE closure | 0.627 | 1.000 | 0.498 |
| Flux-GNN | 0.139 | 0 | 0 |
| BSMS-GNN | N/A | N/A | 0.940 |
| FNO | N/A | N/A | 3.2e-03 |
| GENERIC-FNO | N/A | N/A | 0 |

The Neural Chart and CRNN with mass action are equally accurate here, and both are at least 15 times more accurate than every black box.
On every test set the Neural Chart oscillates on the same trajectories as the reference.
Outside the Turing regime its period error is at most 1.7e-3.
From spatial starts it forms patterns with the reference wavenumber on every trajectory (`results/brusselator/oscillation_patterns.md`).
Flux-GNN reaches 0.075 under the training conditions, but it loses the limit cycle and the Turing pattern when the conditions change (0.29 to 0.64).

The steps of the Brusselator are close to irreversible, and the affinity of such a step, and with it the standard potentials, is only weakly determined by data (Theorem 0(iii)(d) and Theorem 2 of the theory notes).
With learned potentials the model's force differs from the true force on up to 2.2 % of the hyperedges, and the entropy production is overestimated by 20 to 27 %, as for CRNN with mass action.
With the given standard potentials the error drops to the oracle level, and the entropy production error to 4e-5 (`results/brusselator/entropy_readout.md`).

In the Turing regime half of the test trajectories start from a uniform state, which is unstable to spatial perturbations.
In FP32 integration rounding noise can break the symmetry and grow into a pattern, so the error of these trajectories depends on rounding.
This is the source of the 0.026 of CRNN with mass action in that column (`results/brusselator/NOTES.md`).

![Brusselator phase portraits](assets/brusselator_phase_portraits.png)

### MAPK cascade

The Huang–Ferrell MAPK cascade with 22 species and 20 reactions (binding and catalysis of every enzymatic step) on a 4×4 grid.
ATP, ADP and $P_i$ are chemostatted, and the kinase MAPK diffuses.
The test sets change the ATP level and the initial state.

![MAPK cascade, low ATP](assets/mapk_cascade_low_atp.gif)

The animation shows all 22 species at the 16 sites.

| Method | training conditions (ATP 0.5–2) | ATP 3–5 | ATP 0.15–0.3 | phosphorylated initial states | parameters |
|---|---|---|---|---|---|
| **Neural Chart** | 6.6e-03 | **0.021** | **0.011** | 6.6e-03 | 48 |
| Neural Chart with given standard potentials | 6.0e-03 | 4.2e-03 | 9.9e-04 | 6.1e-03 | 23 |
| Oracle (true rates in the Neural Chart class) | 7.1e-06 | 5.8e-06 | 7.3e-06 | 6.0e-06 | 0 (true rates) |
| CRNN (mass action) | 8.5e-03 | 0.027 | 0.020 | 8.7e-03 | 46 |
| CRNN (condition features) | 8.1e-03 | 0.032 | 0.099 | 8.3e-03 | 5,890 |
| UDE closure | 0.090 | 0.745 | 0.390 | 0.084 | 8,641 |
| Flux-GNN | **6.3e-03** | 0.115 | 0.409 | **6.1e-03** | 153,410 |
| FNO | 0.469 | 0.819 | 0.675 | 0.472 | 61,206 |
| BSMS-GNN | 2.307 | 8.783 | 4.501 | 10.463 | 1,066,006 |
| GENERIC-FNO | 1.682 | 1.734 | 1.563 | 1.818 | 1,035,220 |
| Persistence (u(t) = u(0)) | 0.649 | 0.755 | 0.407 | 1.181 | 0 |

| Method | hyperedges with flux against the true force | samples with σ < 0 | negative values |
|---|---|---|---|
| Neural Chart | 0.021 | 0 | 0 |
| Oracle | 0 | 0 | 0 |
| CRNN (mass action) | 0.054 | 0 | 0 |
| CRNN (condition features) | 0.232 | 0 | 0 |
| UDE closure | 0.837 | 1.000 | 0.354 |
| Flux-GNN | 0.133 | 0 | 0 |
| FNO | N/A | N/A | 0.421 |
| BSMS-GNN | N/A | N/A | 0.495 |
| GENERIC-FNO | N/A | N/A | 0.362 |

Flux-GNN matches the Neural Chart under the training conditions (6.3e-3 against 6.6e-3).
At new ATP levels its error grows to 0.12 and 0.41, while the Neural Chart stays at 0.021 and 0.011 and CRNN with mass action at 0.027 and 0.020.
ATP enters the Neural Chart only through the force, so a new ATP level changes no learned parameter.
Here too the standard potentials are weakly determined: the cycle affinity of the first futile cycle is 20.7 against 17.0, and with the given potentials the error at low ATP drops from 0.011 to 1e-3 (`results/mapk_cascade/thermodynamic_readouts.md`).

## How each part contributes

Every variant is the Neural Chart with one change (`neuralchart/chart.py`, `CONFIGS`).
The tables are in `results/<benchmark>/ablations.md`.

| Part of the model | Variant without it | Benchmark | 10T error, Neural Chart → variant |
|---|---|---|---|
| material functional $g_\theta(\xi)$ | one constant mobility for all edges | porous media | 0.064 → 1.18 |
| state term $s\,m(\bar c)$ | no state input | porous media with crowding | 0.067 → 0.241, the level of the oracle without crowding |
| activity functional $\ln\gamma_\theta(c)$ | ideal activities | non-ideal Schlögl | 3.4e-4 → 0.072, and 0.010 → 0.585 at higher concentrations |
| variable projection in the closed-form start | ideal least-squares start, then rollouts | non-ideal Schlögl | 3.4e-4 → 0.016, and 0.010 → 0.119 at higher concentrations |
| convex start | $\omega=1$, $\mu^0=0$ at initialisation | Schlögl, phosphorylation cycle, non-ideal Schlögl, Brusselator, MAPK | 1.8e-6 → 2.6, 5.9e-6 → 0.014, 3.4e-4 → 0.47, 2.6e-3 → 0.50, 6.6e-3 → 0.69 |
| drive $A$ | $A=0$, a reversible model | the same five | errors of 0.08 to 1.5 under the training conditions: the driven steady states cannot be represented |
| force fixed by physics | a learned global scale on the drive | porous media | no change in accuracy (the scale converges to 0.996), but 0.1 % of the edges then carry flux against the true force |
| thermochemical data | standard potentials given instead of learned | reaction networks | similar in the training ranges, but on the Brusselator 2.6e-3 → 1.6e-4 and on the MAPK cascade at low ATP 0.011 → 9.9e-4 |

Without the convex start the same model, trained with the same budget, is 100 to 10^6 times less accurate on every reaction network.

## Scope and limits

- **The graph is given.** The model receives the reaction graph or the mesh. Learning which reactions exist from data is not part of this release.
- **Standard potentials are weakly determined by concentration data.** The model keeps the second law with respect to its own force. Where its learned potentials are off, a small fraction of steps (up to 2.2 % on the Brusselator and the MAPK cascade) carry flux against the true force, and readouts that depend on the potentials, such as the entropy production, inherit that error. Theorem 0(iii)(d) and Theorem 2 of [docs/theory.md](docs/theory.md) show why: the affinity of a step that is close to irreversible carries little information in the data.
- **One seed.** Every number is from one training run.

## Reproduce

**Environment.** Python 3.10 or later, PyTorch 2.4 or later, NumPy and SciPy.
The reported runs used PyTorch 2.14 with CUDA 13.0.

```bash
pip install -e ".[bench]"
pytest                      # unit tests, under a minute on a CPU
pytest -m slow              # regression tests on the benchmark data (needs data/, see below)
```

**Data.** Every benchmark generates its reference data.
The data go to `data/<benchmark>/`, or to `$NEURALCHART_DATA/<benchmark>` when that variable is set.

```bash
python -m benchmarks.porous_media.run generate --variant porous_media --device cuda           # finite volumes on 128×128 in FP64, about 3 h
python -m benchmarks.porous_media.run generate --variant porous_media_crowding --device cuda  # about 3 h
python -m benchmarks.brusselator.run generate                  # CPU processes, about a minute
python -m benchmarks.schlogl.run generate                      # CPU, seconds
python -m benchmarks.schlogl_nonideal.run generate
python -m benchmarks.phosphorylation_cycle.run generate
python -m benchmarks.mapk_cascade.run generate                 # CPU, about a minute
```

**Evaluate the shipped checkpoints.** The Neural Chart checkpoints of every benchmark are in `checkpoints/`.

```bash
python -m scripts.evaluate --benchmark porous_media --model chart --settings
python -m scripts.evaluate --benchmark schlogl_nonideal --model chart
```

**Train.** One command per model and benchmark.
Results go to `runs/<benchmark>/`.
The fast integrator (`--fast`, error control in FP32) is the setting of every reported number.

```bash
python -m scripts.train --benchmark porous_media --model chart --device cuda --fast
python -m scripts.train --benchmark porous_media_crowding --model chart --device cuda --fast
python -m scripts.train --benchmark brusselator --model chart --device cuda --fast
python -m scripts.train --benchmark schlogl --model chart --device cuda
python -m scripts.train --benchmark schlogl_nonideal --model chart --device cuda
python -m scripts.train --benchmark phosphorylation_cycle --model chart --device cuda
python -m scripts.train --benchmark mapk_cascade --model chart --device cuda
```

Model keys: `chart` and its variants (`chart-given-mu0`, `chart-no-state`, `chart-const-k`, `chart-learned-drive`, `chart-zero-init`, `chart-reversible`, `chart-ideal`, `chart-no-varpro`), `oracle`, and the baselines `ude`, `flux_gnn`, `crnn_mass_action`, `crnn_features`, `fno`, `bsms`, `generic_fno`, `helmfluid` (porous media only) and `persistence`.
HelmFluid needs its upstream code (see [THIRD_PARTY.md](THIRD_PARTY.md)).

**Tables and animations.**

```bash
python -m scripts.make_tables --benchmark all                        # results/<benchmark>/*.md and *.csv
python -m scripts.make_tables --benchmark schlogl --runs runs/schlogl # tables from your own runs
python -m scripts.make_gifs --benchmark porous_media --device cuda    # assets/*.gif (baseline weights from runs/)
```

**Training times.** Projected times for 4000 training steps on one NVIDIA L40 with the recommended flags.
The Neural Chart needs `--fast` only.
The UDE closure uses `--fast --control err_only --compile_rates`, and Flux-GNN `--fast --compile_rates --fuse_net`.
On the Brusselator both also use `--compact`.
≈ marks a time taken from a measurement on other hardware or without these flags.
The sources are in `results/<benchmark>/NOTES.md`.

| Benchmark | Neural Chart | UDE closure | Flux-GNN | FNO, BSMS-GNN, GENERIC-FNO, HelmFluid |
|---|---|---|---|---|
| porous media (each variant) | ≈ 15 to 25 min | 7 min | 1.1 h | 3 to 30 min |
| Brusselator | ≈ 20 min | 35 min | 6.6 h | ≈ 3 to 8 min |
| Schlögl, non-ideal Schlögl, phosphorylation cycle | ≈ 5 to 15 min | 4 min | ≈ 15 to 30 min | ≈ 1 to 3 min |
| MAPK cascade | ≈ 30 min | 6 min | ≈ 35 min | 1 to 5 min |

**Speed options** of the fast integrator, all off by default and with the same scheme:
`--compact` lets finished trajectories leave the batch,
`--compile_rates` compiles the rate evaluation with `torch.compile` (needs the CPython headers),
and `--fuse_net` merges the gathers of Flux-GNN.
They change results only at FP32 rounding level.

## Repository layout

```
neuralchart/            the library
  model.py              master equation: free energy, mobility, drive, rates
  chart.py              Neural Chart configurations (CONFIGS) and variants
  fit.py                closed-form NNLS, variable projection, rollout training
  integrators.py        positive SSP-RK2
  fast.py               fast integrator with error control (GPU and CPU)
  diagnostics.py        local detailed balance rule
  graphs.py             hypergraphs of reaction and transport networks
  baselines/            Flux-GNN, CRNN, BSMS-GNN, GENERIC-FNO, HelmFluid wrapper
benchmarks/<name>/      data generation, test sets, models, evaluation, tables, animations, entry point run.py
scripts/                train, evaluate, make_tables, make_gifs, physics_summary
results/<name>/         tables (markdown and csv)
checkpoints/<name>/     trained Neural Chart checkpoints
assets/                 animations of this page
docs/                   method and protocol, theory
tests/                  unit and regression tests
```

## Licence

MIT (see [LICENSE](LICENSE)). Third-party code and methods are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
