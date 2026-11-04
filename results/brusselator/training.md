| Method | gradient steps | training time (h) | final training loss | substeps per sampling step (mean) | largest substeps per sampling step |
|---|---|---|---|---|---|
| Neural Chart | 4000 | 1.08 | 3.5e-07 | 23.245 | 120 |
| Neural Chart, given standard potentials (oracle-type information) | 4000 | 1.07 | 3.7e-10 | 23.259 | 120 |
| Neural Chart, ideal activities | 4000 | 0.42 | 3.5e-07 | 23.247 | 120 |
| Neural Chart, reversible (no drive) | 4000 | 0.33 | 0.157 | 8.431 | 30 |
| No chart (Flux-GNN) | 4000 | 9.00 | 3.7e-05 | 18.501 | 168 |
| CRNN (mass action) | 4000 | 0.32 | 3.4e-07 | 23.247 | 120 |
| CRNN (condition features) | 4000 | 0.22 | 0.026 | 19.230 | 160 |
| UDE closure | 4000 | 0.48 | 0.031 | 10.073 | 47 |
| BSMS-GNN | 4000 | 0.09 | 0.064 | 1 (one-step map) | 1 |
| FNO | 4000 | 0.14 | 4.2e-04 | 1 (one-step map) | 1 |
| GENERIC-FNO | 4000 | 0.06 | 0.048 | 1 (one-step map) | 1 |
| Persistence (u(t) = u(0)) | 0 (not trained) | – | – | – | – |
