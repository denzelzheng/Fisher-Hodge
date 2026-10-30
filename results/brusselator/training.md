| Method | gradient steps | training time (h) | final training loss | substeps per sampling step (mean) | largest substeps per sampling step |
|---|---|---|---|---|---|
| Neural Chart | 4000 | 0.36 | 4.0e-10 | 23.260 | 120 |
| Neural Chart, learned standard potentials | 4000 | 0.34 | 3.5e-07 | 23.247 | 120 |
| Neural Chart, no convex initialisation | 4000 | 1.68 | 0.187 | 215.960 | 1049 |
| Neural Chart, reversible (no drive) | 4000 | 0.07 | 0.191 | 8.000 | 9 |
| Oracle (true rates in the Neural Chart class) | 0 (not trained) | – | – | – | – |
| CRNN (mass action) | 4000 | 0.32 | 3.4e-07 | 23.247 | 120 |
| CRNN (condition features) | 4000 | 0.22 | 0.026 | 19.230 | 160 |
| UDE closure | 4000 | 0.48 | 0.031 | 10.073 | 47 |
| Flux-GNN | 4000 | 9.00 | 3.7e-05 | 18.501 | 168 |
| BSMS-GNN | 4000 | 0.09 | 0.064 | 1 (one-step map) | 1 |
| FNO | 4000 | 0.14 | 4.2e-04 | 1 (one-step map) | 1 |
| GENERIC-FNO | 4000 | 0.06 | 0.048 | 1 (one-step map) | 1 |
| Persistence (u(t) = u(0)) | 0 (not trained) | – | – | – | – |
