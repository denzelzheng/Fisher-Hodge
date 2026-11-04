| Method | training conditions | temperature ×0.25 | temperature ×4 | drive ×4 | drive ×8 | unseen low-porosity media | 64×64 grid | point cloud | trajectories with a negative density (of 64) | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| Neural Chart | 0.064 | 0.077 | 0.091 | 0.191 | 0.401 | 0.077 | 0.073 | 0.090 | 0 | 6,340 |
| Neural Chart, constant mobility per class | 1.108 | 1.046 | 0.866 | 1.040 | 1.003 | 0.506 | 1.108 | 1.123 | 0 | 1,155 |
| Neural Chart, no excess free energy | 0.068 | 0.083 | 0.092 | 0.201 | 0.413 | 0.080 | 0.075 | 0.093 | 0 | 5,187 |
| Neural Chart, no convex initialisation | 0.095 | 0.090 | 0.148 | 0.256 | 0.486 | 0.105 | 0.105 | 0.116 | 0 | 6,340 |
| Neural Chart, reversible (no drive) | 0.702 | 0.828 | 0.344 | 0.950 | 0.999 | 0.338 | 0.709 | 0.698 | 0 | 6,340 |
| No chart (Flux-GNN) | 0.078 | 0.797 | 2.138 | 1.046 | 1.091 | 0.116 | 0.515 | 0.458 | 0 | 151,234 |
| UDE closure | 0.077 | 0.700 | 1.023 | 0.588 | 0.811 | 0.104 | 0.093 | 0.255 | 64 | 5,441 |
| BSMS-GNN | 0.736 | 0.911 | 2.430 | 2.945 | 3.050 | 0.737 | 1.079 | 0.915 | 64 | 1,724,417 |
| FNO | 0.510 | 0.894 | 2.256 | 2.544 | 7.262 | 0.460 | 0.516 | N/A | 64 | 1,188,609 |
| GENERIC-FNO | 1.852 | 1.210 | 2.562 | 1.272 | 1.146 | 1.021 | 1.428 | N/A | 64 | 1,002,388 |
| HelmFluid | 0.846 | 1.154 | 0.472 | 0.930 | 0.976 | 0.946 | N/A | N/A | 64 | 9,930,381 |
| Persistence (u(t) = u(0)) | 1.603 | 1.118 | 2.256 | 1.238 | 1.115 | 0.696 | 1.607 | 1.616 | 0 | 0 |
