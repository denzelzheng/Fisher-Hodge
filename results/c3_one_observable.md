# C3: one observable for the parallel-channel loop R1 reverse / R4 forward

Code: `hyperme/c3_observable.py`, the observable rows of `hyperme/linear_fit.assemble` and the observable term of `hyperme/train.train`, the wired reservoir affinity `presets.c3_wired(known_cycle=...)` (`WiredDrive(chem_mu0_known=...)`), tests `hyperme/tests/test_c3_observable.py`. Runs on the CPU of axiom (no GPU), float64, in `~/fh_anchor/runs/hyperme/observable/`, copied to machine 49 `~/Fisher-Hodge/runs/hyperme/observable/`.

## Prediction and setup

R1 reverse (X → A) and R4 forward (X → E) remove one X at a rate proportional to x under every chemostat setting. They form a loop of dynamically indistinguishable parallel channels, so by proposition 1′ densities cannot split them and one extra observable should. The split matters for σ: in the base runs R1 carries 0.015 of the true σ instead of 0.138 (`results/c3_equilibrium_anchor.md`).

- **V1, the turnover of one reservoir.** Each training trajectory also carries r_E(t) = Σ_sites J_R4(t), the net production rate of E. Its relative error enters the rollout loss with weight 1, the same term for every model with fluxes. It also enters the closed-form step as extra rows, weighted like the loss.
- **V2, known reservoir standard free energies.** μ⁰_A − μ⁰_E = log(k₁k₄/ε²) = 13.82 (true value) is wired into the wired model as a linear constraint on its chemostat potentials. No extra data. CRNN-chem and Flux-GNN have no standard potentials, so V2 does not apply to them (N/A).
- Everything else as before: the 48 driven training trajectories, no anchor, `fit()` with the closed-form step, 2000 Adam iterations, lr 1e-2, MPRK22 × 8, seeds 0 and 1. A Flux-GNN run whose final loss is above 0.05 is repeated once at lr 1e-3.

PLACEHOLDER
