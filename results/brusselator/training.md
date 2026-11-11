| Method | gradient steps | training time (h) | final training loss | substeps per sampling step (mean) | largest substeps per sampling step |
|---|---|---|---|---|---|
| Neural Chart | 4000 | 0.05 | 4.0e-11 | 8.005 | 20 |
| CRNN (mass action) | 4000 | 0.04 | 1.9e-11 | 8.005 | 20 |
| CRNN (condition features) | 4000 | 0.04 | 4.9e-06 | 8.017 | 43 |
| UDE closure | 4000 | 0.15 | 1.1e-03 | 8.000 | 8 |
| BSMS-GNN | 4000 | 0.03 | 0.133 | 1 (one-step map) | 1 |
| FNO | 4000 | 0.09 | 3.6e-05 | 1 (one-step map) | 1 |
| GENERIC-FNO | 4000 | 0.05 | 5.5e-04 | 1 (one-step map) | 1 |
| Persistence (u(t) = u(0)) | 0 (not trained) | – | – | – | – |
