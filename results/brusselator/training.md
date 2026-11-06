| Method | gradient steps | training time (h) | final training loss | substeps per sampling step (mean) | largest substeps per sampling step |
|---|---|---|---|---|---|
| Neural Chart | 4000 | 0.32 | 2.8e-06 | 19.817 | 91 |
| CRNN (mass action) | 4000 | 0.31 | 3.4e-07 | 19.850 | 91 |
| CRNN (condition features) | 4000 | 0.22 | 0.026 | 15.996 | 114 |
| UDE closure | 4000 | 0.20 | 0.037 | 8.709 | 31 |
| BSMS-GNN | 4000 | 0.03 | 0.047 | 1 (one-step map) | 1 |
| FNO | 4000 | 0.09 | 4.5e-04 | 1 (one-step map) | 1 |
| GENERIC-FNO | 4000 | 0.06 | 0.048 | 1 (one-step map) | 1 |
| Persistence (u(t) = u(0)) | 0 (not trained) | – | – | – | – |
