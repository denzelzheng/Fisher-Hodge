| Method | what it removes | training conditions | temperature ×0.25 | temperature ×4 | drive ×4 | drive ×8 | unseen low-porosity media | 64×64 grid | point cloud |
|---|---|---|---|---|---|---|---|---|---|
| Neural Chart | full model | 0.064 | 0.077 | 0.091 | 0.191 | 0.401 | 0.077 | 0.073 | 0.090 |
| Neural Chart, constant mobility per class | g_theta: one constant omega per hyperedge class | 1.108 | 1.046 | 0.866 | 1.040 | 1.003 | 0.506 | 1.108 | 1.123 |
| Neural Chart, no excess free energy | the excess free energy G_theta | 0.068 | 0.083 | 0.092 | 0.201 | 0.413 | 0.080 | 0.075 | 0.093 |
| Neural Chart, no convex initialisation | the closed-form NNLS start: omega starts at 1 | 0.095 | 0.090 | 0.148 | 0.256 | 0.486 | 0.105 | 0.105 | 0.116 |
| Neural Chart, reversible (no drive) | the drive: A = 0, so the model is reversible | 0.702 | 0.828 | 0.344 | 0.950 | 0.999 | 0.338 | 0.709 | 0.698 |
| No chart (Flux-GNN) | the chart: a graph network gives the one-way rates, with no chemical potential, force or symmetric mobility | 0.078 | 0.797 | 2.138 | 1.046 | 1.091 | 0.116 | 0.515 | 0.458 |
