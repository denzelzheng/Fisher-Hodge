| Method | quantity | training conditions (ΔG 1–3) | drive ΔG 3–4 | stronger, finer potential U | X and Xp at the start × 1.5 | 16×16 grid |
|---|---|---|---|---|---|---|
| Neural Chart | σ, relative error | 0.826 | 0.594 | 0.787 | 0.838 | 0.807 |
| Neural Chart | cycle affinity, model; reference | 1.467; 5.984 | 2.996; 7.514 | 1.493; 6.010 | 1.417; 5.934 | 1.394; 5.911 |
| Neural Chart, given standard potentials (oracle-type information) | σ, relative error | 7.1e-03 | 5.6e-03 | 4.8e-03 | 2.7e-03 | 5.7e-03 |
| Neural Chart, given standard potentials (oracle-type information) | cycle affinity, model; reference | 5.984; 5.984 | 7.514; 7.514 | 6.010; 6.010 | 5.934; 5.934 | 5.911; 5.911 |
| No chart (Flux-GNN) | σ, relative error | 0.646 | 0.790 | 0.609 | 0.656 | 0.659 |
| No chart (Flux-GNN) | cycle affinity, model; reference | 2.183; 5.984 | 1.533; 7.514 | 2.178; 6.010 | 2.160; 5.934 | 2.177; 5.911 |
| CRNN (mass action) | σ, relative error | 0.869 | 0.701 | 0.789 | 0.861 | 0.853 |
| CRNN (mass action) | cycle affinity, model; reference | 1.411; 5.984 | 2.940; 7.514 | 1.437; 6.010 | 1.360; 5.934 | 1.338; 5.911 |
| CRNN (condition features) | σ, relative error | 0.865 | 0.809 | 0.800 | 0.857 | 0.862 |
| CRNN (condition features) | cycle affinity, model; reference | 1.129; 5.984 | 1.479; 7.514 | 1.135; 6.010 | 1.116; 5.934 | 1.115; 5.911 |
| Neural Chart, constant mobility per class | σ, relative error | 0.812 | 0.573 | 0.769 | 0.810 | 0.789 |
| Neural Chart, constant mobility per class | cycle affinity, model; reference | 1.479; 5.984 | 3.009; 7.514 | 1.505; 6.010 | 1.429; 5.934 | 1.406; 5.911 |
| Neural Chart, no excess free energy | σ, relative error | 0.803 | 0.630 | 0.805 | 0.841 | 0.785 |
| Neural Chart, no excess free energy | cycle affinity, model; reference | 1.643; 5.984 | 3.173; 7.514 | 1.669; 6.010 | 1.593; 5.934 | 1.570; 5.911 |
| Neural Chart, no convex initialisation | σ, relative error | 0.708 | 0.514 | 0.678 | 0.715 | 0.690 |
| Neural Chart, no convex initialisation | cycle affinity, model; reference | 2.018; 5.984 | 3.548; 7.514 | 2.044; 6.010 | 1.968; 5.934 | 1.945; 5.911 |
| Neural Chart, reversible (no drive) | σ, relative error | diverged | diverged | diverged | diverged | diverged |
| Neural Chart, reversible (no drive) | cycle affinity, model; reference | 2.8e-17; 5.984 | 0; 7.514 | -4.2e-17; 6.010 | -1.8e-16; 5.934 | 1.7e-16; 5.911 |
