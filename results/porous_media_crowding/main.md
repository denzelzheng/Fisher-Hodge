| Method | training conditions | temperature ×0.25 | temperature ×4 | drive ×4 | drive ×8 | unseen low-porosity media | 64×64 grid | point cloud | trajectories with a negative density (of 64) | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| Neural Chart | 0.056 | 0.066 | 0.121 | 0.155 | 0.373 | 0.057 | 0.069 | 0.083 | 0 | 6,340 |
| Neural Chart, constant mobility per class | 1.030 | 0.929 | 0.828 | 1.002 | 0.996 | 0.420 | 1.029 | 1.038 | 0 | 1,155 |
| Neural Chart, no excess free energy | 0.058 | 0.070 | 0.123 | 0.163 | 0.389 | 0.059 | 0.071 | 0.084 | 0 | 5,187 |
| Neural Chart, no convex initialisation | 0.111 | 0.079 | 0.237 | 0.178 | 0.355 | 0.090 | 0.122 | 0.126 | 0 | 6,340 |
| Neural Chart, reversible (no drive) | 0.608 | 0.651 | 0.319 | 0.912 | 0.954 | 0.195 | 0.614 | 0.601 | 0 | 6,340 |
| No chart (Flux-GNN) | 0.057 | 0.808 | 2.053 | 0.856 | 1.049 | 0.057 | 0.573 | 0.473 | 0 | 151,234 |
| UDE closure | 0.066 | 0.679 | 1.916 | 0.601 | 0.917 | 0.075 | 0.756 | 0.485 | 64 | 5,441 |
| BSMS-GNN | 0.592 | 0.837 | 2.739 | 1.386 | 1.355 | 0.435 | 1.008 | 0.713 | 64 | 1,724,417 |
| FNO | 0.464 | 0.898 | 2.370 | 1.897 | 3.316 | 0.386 | 0.469 | N/A | 64 | 1,188,609 |
| GENERIC-FNO | 1.158 | 0.924 | 1.768 | 1.098 | 1.054 | 0.371 | 1.206 | N/A | 64 | 1,002,388 |
| HelmFluid | 14.957 | 7.914 | 284.284 | 199.367 | 149.037 | 14.371 | N/A | N/A | 64 | 9,930,381 |
| Persistence (u(t) = u(0)) | 1.363 | 0.989 | 2.161 | 1.156 | 1.083 | 0.447 | 1.360 | 1.392 | 0 | 0 |
