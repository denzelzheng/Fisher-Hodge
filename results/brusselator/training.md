| Method | gradient steps | training time (h) | final training loss | substeps per sampling step (mean) | largest substeps per sampling step |
|---|---|---|---|---|---|
| Neural Chart | 4000 | 4.06 | 1.2e-07 | 23.211 | 121 |
| Neural Chart, given standard potentials (oracle-type information) | 4000 | 4.05 | 1.4e-10 | 23.266 | 120 |
| Neural Chart, constant mobility per class | 4000 | 1.58 | 1.3e-07 | 23.198 | 120 |
| Neural Chart, no excess free energy | 4000 | 2.73 | 3.1e-07 | 23.259 | 121 |
| Neural Chart, reversible (no drive) | 4000 | 1.62 | 0.061 | 13.712 | 389 |
| No chart (Flux-GNN) | 4000 | 9.00 | 3.7e-05 | 18.501 | 168 |
| CRNN (mass action) | 4000 | 0.32 | 3.4e-07 | 23.247 | 120 |
| CRNN (condition features) | 4000 | 0.22 | 0.026 | 19.230 | 160 |
| UDE closure | 4000 | 0.48 | 0.031 | 10.073 | 47 |
| BSMS-GNN | 4000 | 0.09 | 0.064 | 1 (one-step map) | 1 |
| FNO | 4000 | 0.14 | 4.2e-04 | 1 (one-step map) | 1 |
| GENERIC-FNO | 4000 | 0.06 | 0.048 | 1 (one-step map) | 1 |
| Persistence (u(t) = u(0)) | 0 (not trained) | – | – | – | – |
