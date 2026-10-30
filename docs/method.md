# Method and protocol

This page describes the Neural Chart, its four-step fit, and the protocol that every benchmark and every model follows.
The theory behind the design is in [theory.md](theory.md).

## The model

The state $c$ collects species at sites (cells of a grid or of a point cloud).
A hyperedge $h$ is a reaction at one site or a transport step between two neighbouring cells.
It turns the reactants $\nu_h^+$ into the products $\nu_h^-$, and $S=\nu^--\nu^+$ is the stoichiometric matrix.
The forward and backward rates are

$$a_h=k_h\,e^{\langle\nu_h^+,\mu\rangle+A_h/2},\qquad b_h=k_h\,e^{\langle\nu_h^-,\mu\rangle-A_h/2},\qquad \dot c=S\,(a-b).$$

Equivalently $a_h,b_h=g_h\,e^{\pm f_h/2}$ with the force $f_h=\log(a_h/b_h)=-\langle S_h,\mu\rangle+A_h$.
Each rate carries the mass-action factor of the species it consumes.

**The force is fixed by physics.**
The chemical potential is $\mu=\log c+U/T+\mu^0/T$, plus an activity correction $\ln\gamma(c)$ on the non-ideal Schlögl model.
The drive $A_h$ is the line integral of the external force divided by $T$, plus the chemical potentials of the chemostatted species.
The temperature, the drive amplitude and the chemostat levels are the conditions of a test set.
They enter only through $f$.

**Only the mobility is learned.**

$$\log k_h=\log\omega_{c(h)}+g_\theta(\xi_h)+s\,m(\log s_h).$$

- $\omega_c>0$ is one constant per hyperedge class (one class for transport in porous media, and one per reaction type and per diffusing species in the reaction networks).
- $g_\theta$ is an MLP of local material descriptors $\xi_h$. Its last layer starts at zero. It is absent when a task has no descriptors.
- $s\,m(\cdot)$ is a state term. $m$ is non-decreasing and linear in both tails, and $s$ starts at zero. In porous media the state feature $s_h$ is the mean density of the two cells of the edge.
- Fixed geometric and physical prefactors stay outside $k$. In porous media $k_e=D\,T\,w_e\,\omega\,e^{g+s\,m}$, with $w_e$ the geometric weight of the finite-volume scheme on the grid or on the Voronoi cells.

**What holds for every parameter value.**
$\log(a_h/b_h)=f_h$ is local detailed balance.
$J_hf_h\ge0$ on every hyperedge, so the entropy production $\sigma=\sum_hJ_hf_h$ is non-negative.
Every moiety in $\ker S^\top$ is conserved, and the positive integrator keeps $c>0$.
None of this needs a penalty or a projection.
Because the conditions enter only through $f$ and $k$ does not read them, a change of temperature, drive or chemostat level maps the model by substitution.

## The four-step fit

1. **Physics.** Build $f$ from the conditions and the thermodynamic data ($U$, $\mu^0$). Nothing is learned here.
2. **Mobility.** Set up $\log k$ as above for the task (the configurations are in `neuralchart/chart.py`).
3. **Closed-form start, then rollout refinement.** At initialisation $g=0$ and $s=0$, so $k$ is linear in $\omega$. The derivative fit $c(t+2\Delta t)-c(t)=\sum_r\kappa_r\,S\int\Phi_r\,dt$ (Simpson rule) is a non-negative least-squares problem in the class factors $\kappa_r$. Its optimum is global, and $\omega$ starts there (`neuralchart.fit.fit_linear`). When $\mu$ carries a learned activity correction $\ln\gamma_\theta$, the step is variable projected. $\omega$ is the NNLS optimum at every value of $\ln\gamma_\theta$, and $\ln\gamma_\theta$ descends the projected residual (Adam, learning rate 1e-2, gradient clip 1, stop when the residual improves by less than 5 % over 100 iterations, at most 2000 iterations, `fit_varpro`). Then every parameter is refined on multi-step rollouts (`neuralchart.fit.train`).
4. **Integration.** A positivity-preserving adaptive SSP-RK2 (`neuralchart.fast`). Each substep is limited so that no species can be consumed below zero, and an embedded error estimate controls the step size per trajectory.

## Protocol

The same protocol applies to the Neural Chart, its ablations and every baseline.

| | porous media (both variants) | Brusselator | Schlögl, non-ideal Schlögl, phosphorylation cycle | MAPK cascade |
|---|---|---|---|---|
| sampling interval $\Delta t$ | 0.05 | 0.2 | 0.05 | 0.5 |
| training window | 100 steps (5 time units) | 50 steps (10 time units) | 100 steps (5 time units) | 20 steps (10 time units) |
| test horizon | 200 steps (10T) | 200 steps | 200 steps | 200 steps |
| rollout length $K$ in training | 4 to 20 | 4 to 20 | 4 to 20 | 4 to 20 |
| error-control tolerance (rtol = atol) | 0.3 | 5e-5 | 1e-4 | 1e-4 |
| UDE closure tolerance | 1e-3 | 5e-5 | 1e-4 | 1e-4 |
| substep cap per $\Delta t$, training / evaluation | 20000 / 20000 | 2000 / 4000 | 2000 / 4000 | 2000 / 4000 |

- **Training.** One seed (0). Batch 8, Adam with learning rate 1e-3 and a cosine schedule, gradient clip 1, 4000 steps or 12 hours of wall-clock time, whichever comes first. The time of the closed-form step counts in the budget. Baselines that train on one-step targets (FNO, BSMS-GNN, GENERIC-FNO, HelmFluid) keep their own loss and schedule with the same number of gradient steps.
- **Precision.** Training and evaluation integrate in FP32. The step control (time, step size) stays in FP64.
- **Tolerances.** The tolerance of each task was chosen against the oracle (the true rates in the Neural Chart class): it is the largest rtol whose integration error is negligible against the oracle's 10T error (`python -m benchmarks.<name>.run calibrate` on the reaction networks). In porous media the error control does not bind below rtol = 0.3, because the oracle's error there is the discretisation error of the 32×32 grid.
- **Metric.** The relative L2 error of the full state at 10T (200 sampling steps), median over the test trajectories. A rollout that is not finite, or whose error exceeds 1e3, is reported as diverged.
- **Local detailed balance.** A hyperedge violates local detailed balance when its flux runs against the physical force of the test condition beyond rounding: $J_hf_h<-10^{-6}\,(a_h+b_h)\,|f_h|$. The rounding error of $J=a-b$ scales with the gross rate $a+b$. For a model with a single edge flux and no forward and backward rates (the UDE closure) the scale is $|J_h|$. The same rule (`neuralchart.diagnostics.ldb_violation`) applies to every model and every task.
- **Entropy production.** $\sigma_{\rm phys}=\sum_hJ_hf_h$ with the physical force, sampled along the rollout. Models without their own edge flux (FNO, BSMS-GNN, GENERIC-FNO, HelmFluid, persistence) return a state update only, so these columns are N/A for them. A flux reconstructed from the state change would be curl-free and could not carry a driven circulation.
- **Negative values.** Counted for every model, as the fraction of negative state values and the number of trajectories that reach a negative value.

## The UDE closure: error control only

The UDE closure uses the same graph, inputs and conservative assembly $\dot c=SJ$ as the Neural Chart.
Its edge flux is a free MLP, so it has no forward and backward rates and no positivity structure.
The positivity rule of the integrator does not apply to such a flux.
On porous media it slowed the closure down 5 to 55 times without a gain in accuracy.
The closure therefore integrates with the same error control and no positivity rule (`control='err_only'`).
Its tolerance is set so that its integration error against a fine reference is at most 1e-3.
Negative values are allowed and reported as violations.

## Baselines

| baseline | what it learns | own edge flux |
|---|---|---|
| UDE closure | a free antisymmetric edge flux on the same graph | yes (single flux) |
| Flux-GNN | forward and backward rates from hypergraph message passing, with mass action | yes |
| CRNN (mass action) | mass action with free forward and backward rate constants per reaction type | yes |
| CRNN (condition features) | forward and backward rate constants from an MLP of the hyperedge and condition features | yes |
| FNO | a residual next-step map on the grid | no |
| BSMS-GNN | a multi-scale message-passing next-step map (bi-stride pooling) | no |
| GENERIC-FNO | an energy and an entropy functional with degenerate operators, one step per interval | no |
| HelmFluid | a Helmholtz-decomposed velocity that warps the field (porous media only) | no |
| persistence | $u(t)=u(0)$ | no |

The conditions enter the black boxes as input features, z-scored with training statistics.
FNO, GENERIC-FNO and HelmFluid run on grids only, so they are N/A on the point cloud.

## Hardware and reproducibility

The reported numbers come from one seed.
Re-evaluating the stored checkpoints with this code reproduces the reported 10T errors bitwise on the same GPU model.
A few baseline rows were trained and evaluated on a different GPU model. Re-evaluated here, they differ through the accumulation of FP32 rounding over the 200 steps of a rollout: at most 0.7 % (FNO on the Brusselator) and below 1e-5 elsewhere.
