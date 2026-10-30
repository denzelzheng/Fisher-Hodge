| Method | training conditions | temperature ×0.25 | temperature ×4 | drive ×4 | drive ×8 | unseen low-porosity media | 64×64 grid | point cloud | trajectories with a negative density (of 64) | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| Neural Chart | 0.064 | 0.084 | 0.090 | 0.195 | 0.407 | 0.081 | 0.071 | 0.090 | 0 | 5,381 |
| Neural Chart, no state input | 0.070 | 0.084 | 0.099 | 0.199 | 0.404 | 0.085 | 0.077 | 0.092 | 0 | 5,058 |
| Neural Chart, constant mobility | 1.180 | 1.067 | 0.943 | 1.065 | 1.012 | 0.511 | 1.180 | 1.180 | 0 | 1 |
| Neural Chart, learned drive scale | 0.063 | 0.084 | 0.090 | 0.192 | 0.403 | 0.080 | 0.071 | 0.090 | 0 | 5,382 |
| Oracle (true rates in the Neural Chart class) | 0.046 | 0.107 | 0.032 | 0.143 | 0.231 | 0.037 | 0.034 | 0.083 | 0 | 0 (true rates) |
| UDE closure | 0.077 | 0.700 | 1.023 | 0.588 | 0.811 | 0.104 | 0.093 | 0.255 | 64 | 5,441 |
| Flux-GNN | 0.078 | 0.797 | 2.138 | 1.046 | 1.091 | 0.116 | 0.515 | 0.458 | 0 | 151,234 |
| BSMS-GNN | 0.736 | 0.911 | 2.430 | 2.945 | 3.050 | 0.737 | 1.079 | 0.915 | 64 | 1,724,417 |
| FNO | 0.510 | 0.894 | 2.256 | 2.544 | 7.262 | 0.460 | 0.516 | N/A | 64 | 1,188,609 |
| GENERIC-FNO | 1.852 | 1.210 | 2.562 | 1.272 | 1.146 | 1.021 | 1.428 | N/A | 64 | 1,002,388 |
| HelmFluid | 0.846 | 1.154 | 0.472 | 0.930 | 0.976 | 0.946 | N/A | N/A | 64 | 9,930,381 |
| Persistence (u(t) = u(0)) | 1.603 | 1.118 | 2.256 | 1.238 | 1.115 | 0.696 | 1.607 | 1.616 | 0 | 0 |
