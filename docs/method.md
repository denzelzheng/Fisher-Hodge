# Method and protocol

This page describes the Neural Chart, its fit, and the protocol that every benchmark and every model follows.
The theory is in [theory.md](theory.md), and composition is in [gluing.md](gluing.md).

## The model

The state $c$ collects species at sites, which are the cells of a grid or of a point cloud.
A hyperedge $h$ is a reaction at one site or a transport step between two neighbouring cells.
It turns the reactants $\nu_h^+$ into the products $\nu_h^-$, and $S=\nu^--\nu^+$ is the stoichiometric matrix.
Every task uses one rate form,

$$a_h,b_h=k_h\,\varepsilon_h\,e^{\pm f_h/2},\qquad \varepsilon_h=e^{\frac12\langle\nu_h^++\nu_h^-,y\rangle},\qquad f_h=\langle\nu_h^+-\nu_h^-,y+\phi\rangle+A_h,\qquad \dot c=S\,(a-b),$$

with $y=\log c+\nabla G_\theta(c/V)-\nabla G_\theta(0)$ and $\phi=(U+\mu^0_\theta)/T$.
Written out, $a_h=k_h\exp\big(\langle\nu_h^+,y\rangle+\tfrac12\langle\nu_h^+-\nu_h^-,\phi\rangle+\tfrac12A_h\big)$, so each rate carries the mass-action factor of the species it consumes.

**Inputs.**
The model receives the state, the hypergraph (the stoichiometry and the hyperedge classes), and the conditions of a trajectory: the temperature $T$, an external potential $U$, the line integrals of an external force, the chemostat levels, and the standard free energies of the chemostats, which are looked up in thermochemical tables.
Material descriptors $\xi_h$ are optional.

**The free energy.**
The free energy is $F(c)=\sum_sc_s(\log c_s-1)+\langle c,\phi\rangle+G_\theta(c)$, and $\mu=y+\phi=\partial_cF$.
The excess free energy $G_\theta$ is a scalar MLP per site over all local species, with weights shared across sites and a last layer that starts at zero.
Its tails are quadratic, so the excess potential $\nabla G_\theta$ is unbounded.
It has no monotone or separable constraint.
The standard potentials $\mu^0_\theta$ of the internal species are learned constants.
The drive $A_h$ is the line integral of the external force divided by $T$, plus the chemical potentials of the chemostats.
The conditions enter through $\phi$, $A$ and the chemostat mass-action factor, and $G_\theta$ does not read them.

**The mobility.**

$$\log k_h=\log\omega_{c(h)}+g_\theta(\text{class},\xi_h,\log\bar c_h).$$

- $\omega_c>0$ is one constant per hyperedge class.
- $g_\theta$ is one MLP for all hyperedges, with a last layer that starts at zero. It reads the one-hot class, the descriptors and $\bar c_h$, and it never reads the conditions.
- $\bar c_h$ is the mean, over the cells $h$ touches, of the local density vector of all species.
- Known prefactors stay outside the network: $T$, the geometric weight $w_e$ of a mesh edge, and the chemostat mass-action factor.

**What holds for every parameter value.**
$\log(a_h/b_h)=f_h$ is local detailed balance with respect to the model's own force.
$J_hf_h\ge0$ on every hyperedge, so the model's entropy production is non-negative.
Every moiety in $\ker S^\top$ is conserved, and the positive integrator keeps $c>0$.
The model's force equals the true force only as far as the learned $\mu^0_\theta$ and $G_\theta$ are right.
We therefore measure the entropy production against the truth and count the hyperedges whose flux runs against the true force.

**Ablations.** Each row of `VARIANTS` in `neuralchart/chart.py` is one override of the constructor.

| Key | Change |
|---|---|
| `chart` | the method |
| `flux_gnn` | no chart: a hypergraph network emits the one-way rates |
| `chart-const-k` | one constant mobility per hyperedge class, no $g_\theta$ |
| `chart-no-excess` | $G_\theta=0$ |
| `chart-zero-init` | no closed-form start, $\omega=1$ |
| `chart-reversible` | $A=0$ |
| `chart-learned-chem-mu0` | the chemostat standard free energies are fitted to the data |
| `chart-given-mu0` | the internal standard potentials are also given |

## The fit

1. **Closed-form start.** At initialisation $g_\theta=0$ and $G_\theta=0$. The rates are then linear in the one-way class factors $\kappa_r$, and the derivative fit $c(t+4\Delta t)-c(t)=\sum_r\kappa_rS\int\Phi_r\,dt$ (Boole's rule) is a non-negative least-squares problem. This problem is convex and its optimum is global.
2. **Back to the shared parameters.** The factors are mapped to $\omega$ and $\mu^0_\theta$ by least squares on $\log\kappa$. When classes are coupled, L-BFGS-B then minimises the residual over those parameters, which is not convex. The bound on each $\log\omega_c$ is $\log(10^{-3}\max_r\kappa_r)$ on every task, so that every class keeps a usable gradient.
3. **Variable projection.** $G_\theta$ and $g_\theta$ are trained jointly on the derivative residual for at most 500 steps (Adam, learning rate $10^{-3}$, all trajectories). At each step the class factors are solved again by non-negative least squares for the current networks (`neuralchart.fit.fit_varpro`). The residual also holds the error of the quadrature. Every 25 steps the integrator measures that error for the current model, as the same integral along the model's own path from the same start. The step stops when the rest of the residual is no larger than that error. This step is not convex.
4. **Rollouts.** Every parameter is trained on rollouts of 4 to 20 steps (`neuralchart.fit.train`). This step is not convex.

Networks and integration use FP32. The least-squares problems use float64.

The integrator is a positivity-preserving adaptive SSP-RK2 (`neuralchart.fast`), with the same settings on every task: rtol = atol = 1e-4, a substep bound of 1/8 of the sampling step, and a positivity fraction of 0.5.

## Composition

`compose(chart_a, chart_b, glue=...)` joins two fitted charts along shared species with no retraining, and `refine` fits the cross terms on data of the composite. [gluing.md](gluing.md) states the theory. `benchmarks/composition/__init__.py` fixes the design, which was committed before any result.

## Protocol

The same protocol applies to the Neural Chart, its ablations and every baseline.

- **Training.** One seed (0). Batch 8, Adam with learning rate 1e-3 and a cosine schedule, gradient clip 1, 4000 steps or 12 hours of wall-clock time, whichever comes first. The time of the closed-form step counts in the budget. Baselines that train on one-step targets (FNO, BSMS-GNN, GENERIC-FNO, HelmFluid) keep their own loss and schedule with the same number of gradient steps.
- **Precision.** Training and evaluation integrate in FP32. The step control (time, step size) stays in FP64.
- **Metric.** The relative L2 error of the full state at 10T (200 sampling steps), median over the test trajectories. A rollout that is not finite, or whose error exceeds 1e3, is reported as diverged.
- **Local detailed balance.** A hyperedge violates local detailed balance when its flux runs against the true physical force of the test condition beyond rounding, evaluated at the model's own state: $J_hf_h<-10^{-6}\,(a_h+b_h)\,|f_h|$. For a model with a single edge flux (the UDE closure) the scale is $|J_h|$. The same rule (`neuralchart.diagnostics.ldb_violation`) applies to every model and every task.
- **Entropy production.** $\sigma_{\rm phys}=\sum_hJ_hf_h$ with the physical force, sampled along the rollout. Its relative error is evaluated on $t>0$. Models without their own edge flux (FNO, BSMS-GNN, GENERIC-FNO, HelmFluid, persistence) return a state update only, so these columns are N/A for them.
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
