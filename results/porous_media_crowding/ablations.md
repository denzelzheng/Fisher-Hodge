| Method | what it removes | training conditions | temperature ×0.25 | temperature ×4 | drive ×4 | drive ×8 | unseen low-porosity media | 64×64 grid | point cloud |
|---|---|---|---|---|---|---|---|---|---|
| Neural Chart | full model | 0.056 | 0.066 | 0.121 | 0.155 | 0.373 | 0.057 | 0.069 | 0.083 |
| Neural Chart, constant mobility per class | g_theta: one constant omega per hyperedge class | 1.030 | 0.929 | 0.828 | 1.002 | 0.996 | 0.420 | 1.029 | 1.038 |
| Neural Chart, no excess free energy | the excess free energy G_theta | 0.058 | 0.070 | 0.123 | 0.163 | 0.389 | 0.059 | 0.071 | 0.084 |
| Neural Chart, no convex initialisation | the closed-form NNLS start: omega starts at 1 | 0.111 | 0.079 | 0.237 | 0.178 | 0.355 | 0.090 | 0.122 | 0.126 |
| Neural Chart, reversible (no drive) | the drive: A = 0, so the model is reversible | 0.608 | 0.651 | 0.319 | 0.912 | 0.954 | 0.195 | 0.614 | 0.601 |
| No chart (Flux-GNN) | the chart: a graph network gives the one-way rates, with no chemical potential, force or symmetric mobility | 0.057 | 0.808 | 2.053 | 0.856 | 1.049 | 0.057 | 0.573 | 0.473 |
