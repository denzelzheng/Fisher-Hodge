# Fisher–Hodge operators

Neural operators for density dynamics on graphs and meshes, written in the geometry the densities live in.

A density field on a graph evolves by moving mass along edges. We write the learned dynamics as a discrete Wasserstein gradient flow of a learned free energy, plus a learned circulation term and a learned reaction term:

```
dρ/dt = − dᵀ Θ(ρ) d μ_θ(ρ)   −   dᵀ ( Θ(ρ) c_θ )   −   ρ ⊙ r_θ(ρ)
          gradient flow            circulation           reaction
```

Here `d` is the fixed incidence matrix of the graph (node to edge), `Θ(ρ)` is the edge mobility given by the logarithmic mean of the two endpoint densities times a learned edge coefficient, and `μ_θ = ∂F_θ/∂ρ` is the chemical potential of a learned free energy `F_θ`. The first term is the Maas metric form of a Fokker–Planck equation. The second term lives in the cycle space of the graph and carries non-equilibrium circulation. The third term is a Fisher–Rao reaction written as an Otto flux to a reservoir node, which reduces to logistic growth when `F` is the entropy.

Three properties hold for every parameter value, so training never has to learn them:

- Total mass is conserved, because `1ᵀ dᵀ = 0`.
- Densities stay positive, because the logarithmic-mean mobility vanishes at zero density.
- The free energy decreases along the flow, `dF/dt = −‖d μ‖²_Θ ≤ 0`.

The operator `dᵀ Θ d` is the Hodge block of Riemannian Hodge Message Passing (RHMP) with the cochain metric set to the Otto metric of the current density. This repository contains the mechanism experiments, the Markov-chain analysis and the scripts that make the figures below.

## What the figures show

All rollouts train on one time unit (`T = 1`, 20 steps) and are tested out to `10 T`. The green frame marks the training window. Cyan and magenta cells are negative densities. The two numbers in each panel are the total mass and the minimum density.

### Non-equilibrium Fokker–Planck (potential plus divergence-free drift)

![S2 rollout](assets/s2_rollout.gif)

The Fisher–Hodge operator stays on top of the reference solution for the whole rollout, with total mass equal to one and no negative cells. The Euclidean gradient-flow GNN produces negative densities inside the training window. The unstructured GNN-ODE roughly doubles the total mass by `10 T`.

![S2 filmstrip](assets/s2_rollout_filmstrip.png)

### Steady-state circulation

![S2 flux streamlines](assets/s2_flux_streamlines.png)

At `10 T` the density no longer changes, but probability still flows around closed loops. The streamlines of the probability flux `J = ρ v` show these loops in the reference solution and in the Fisher–Hodge operator (peak flux 1.05 times the reference). A free-energy model that assumes reversible dynamics, trained in the style of Rancati, Maas and Locatello (2026), has a steady flux of one percent of the reference, so its panel is empty.

![S2 tracers](assets/s2_tracers.gif)

The same point, seen through tracer particles advected by each model's own steady velocity field `v = J/ρ`. Particles in the reference and in the Fisher–Hodge field orbit the same loops. Particles in the reversible model barely move.

### Reaction–diffusion (Fisher–KPP)

![S3 reaction](assets/s3_reaction.gif)

Green marks densities above the true carrying capacity of one. A free reaction network overshoots to 1.5, because the training trajectories never reach saturation and the carrying capacity is not identifiable from them. The Fisher–Rao reaction term saturates at 1.00 and recovers the carrying capacity and the growth rate from the same data. The GNN-ODE never grows and develops a negative pocket.

![S3 filmstrip](assets/s3_reaction_filmstrip.png)

### Mobility outside the model class

![S5 misspecified](assets/s5_misspecified.gif)

The reference solver here uses a crowding mobility `ρ(1 − ρ)`, which the operator cannot represent exactly. Its error grows from 0.003 to 0.02, and mass conservation and positivity still hold. The Euclidean gradient-flow model drifts in mass and goes negative as before.

### Hodge decomposition of a non-equilibrium Markov chain

![Markov decomposition](assets/markov_decomposition.png)

At a non-equilibrium steady state the edge force `f = log(π_i Q_ij / π_j Q_ji)` should contain no gradient part, because the density no longer moves downhill. The Euclidean Hodge decomposition still reports a potential and a set of gradient arrows (maximum 1.7). The decomposition weighted by the Onsager–Fisher coefficients `ℓ = J/f` reports zero (maximum 1e-14). The weighted decomposition also splits the entropy production exactly into an excess part and a housekeeping part. The Euclidean one misassigns 10 to 28 percent of the entropy production, and this bias does not shrink with more data.

![Markov relaxation](assets/markov_relaxation.gif)

## Numbers

Relative L2 error at `10 T`, median over 16 test trajectories, mean over three seeds. The three columns are the in-distribution test set, a test set with potentials from a different distribution, and a test set with initial conditions from a different distribution.

| System | Fisher–Hodge | Euclidean gradient-flow GNN | GNN-ODE | Reversible free energy (JKO) |
|---|---|---|---|---|
| S1 Fokker–Planck | 0.0034 / 0.017 / 0.0026 | 0.25 / 0.27 / 0.15 | 0.29 / 1.48 / 0.32 | 0.0086 / 0.017 / 0.0093 |
| S2 with circulation | 0.0033 / 0.014 / 0.0032 | 0.030 / 0.069 / 0.031 | 0.26 / 0.96 / 0.30 | 0.25 / 0.46 / 0.25 |
| S3 Fisher–KPP | 0.017 / 0.016 / 0.013 | 3.6 / 3.8 / 3.7 | 0.90 / 2.4 / 0.90 | – |
| S5 crowding mobility (one seed) | 0.020 / 0.022 / 0.016 | 0.096 / 0.195 / 0.162 | – | – |

Mass drift of the Fisher–Hodge operator is below 4e-7 in every run, and no test trajectory contains a negative cell. Full tables are in `results/`.

Two caveats. The GNN-ODE baseline was trained for 4000 steps and its loss was still falling. A longer run was interrupted at 7000 steps with a loss three times lower, so the GNN-ODE column overstates the gap. The reference solvers for S1 to S3 lie inside the operator's model class. The S4 and S5 systems are the first tests outside it, and a system with non-local interactions has not been run.

## Layout

```
fisher_hodge/   models, data generators, training and evaluation
  common.py     32×32 periodic grid, incidence operators d and dᵀ
  models.py     FisherHodge, EuclidGF, GNNODE
  models2.py    FluxGNN, JKOAmort, JKOInst, FisherHodgeWFR and other stress-test models
  gen_data.py   reference solutions for S1, S2, S3 (64×64 finite volume, averaged to 32×32)
  gen_data2.py  reference solutions for S4, S5, S6 and the S3b training set
  train.py      train and evaluate one (system, model, seed)
  train2.py     same for the stress-test models
analysis/       Onsager–Fisher weighted Hodge decomposition of Markov chains (numpy)
viz/            scripts that make the figures above
assets/         figures and animations used in this file
results/        summary tables of the first round, the stress tests and the Markov analysis
```

## Running

Set `FH_ROOT` to the repository root. Data and checkpoints go to `$FH_ROOT/runs/`.

```bash
export FH_ROOT=$(pwd)
pip install -r requirements.txt
cd fisher_hodge
python gen_data.py                                   # writes runs/data/S1.pt, S2.pt, S3.pt
python train.py --sys S2 --model fisher_hodge --seed 0
python train.py --sys S2 --model euclid_gf --seed 0
python analyze.py                                    # tables in runs/summary.md
```

One training run takes about one hour on a single GPU. The stress-test models use `train2.py` and `gen_data2.py` with the same flags.

To make the figures, first cache full rollouts of the trained checkpoints, then run the drawing scripts:

```bash
python viz/rollout_all.py S2
python viz/viz_grid.py S2 S2 test_oodpot 0 fisher_hodge,euclid_gf,gnn_ode "S2"
python viz/viz_s2_flow.py test_oodpot 0
python viz/viz_markov_hodge.py
```

The Markov-chain analysis runs on the CPU:

```bash
cd analysis
python e1.py && python e2.py && python e3.py && python report.py
```

## Background

- Maas, *Gradient flows of the entropy for finite Markov chains*, J. Funct. Anal. 2011. The discrete transport metric with logarithmic-mean mobility.
- Chow, Huang, Li, Zhou, *Fokker–Planck equations for a free energy functional or Markov process on a graph*, ARMA 2012.
- Dechant, Sasa, Ito, *Geometric decomposition of entropy production*, Phys. Rev. Research 2022. The excess and housekeeping split used in the Markov analysis.
- Rancati, Maas, Locatello, *Learning discrete diffusion of graphs via free-energy gradient flows*, arXiv:2604.11311. The reversible free-energy baseline.
- Zheng, Allen-Blanchette, *Learning discrete Riemannian metrics for physical fields with cochain-frame equivariance* (RHMP), NeurIPS 2026. The Hodge block `dᵀ H d` with a learned cochain metric.
- Zheng, Zhong, Allen-Blanchette, *Topology-preserving neural operator learning via Hodge decomposition* (HSD), ICML 2026.
