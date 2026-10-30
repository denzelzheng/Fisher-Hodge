| Method | training conditions (ATP 0.5–2) | ATP 3–5 | ATP 0.15–0.3 | phosphorylated initial states | trajectories with a negative value | parameters |
|---|---|---|---|---|---|---|
| Neural Chart | 6.0e-03 | 4.2e-03 | 9.9e-04 | 6.1e-03 | 0 | 23 |
| Oracle (true rates in the Neural Chart class) | 7.1e-06 | 5.8e-06 | 7.3e-06 | 6.0e-06 | 0 | 0 (true rates) |
| CRNN (mass action) | 8.5e-03 | 0.027 | 0.020 | 8.7e-03 | 0 | 46 |
| CRNN (condition features) | 8.1e-03 | 0.032 | 0.099 | 8.3e-03 | 0 | 5,890 |
| UDE closure | 0.090 | 0.745 | 0.390 | 0.084 | 16 | 8,641 |
| Flux-GNN | 6.3e-03 | 0.115 | 0.409 | 6.1e-03 | 0 | 153,410 |
| FNO | 0.469 | 0.819 | 0.675 | 0.472 | 16 | 61,206 |
| BSMS-GNN | 2.307 | 8.783 | 4.501 | 10.463 | 16 | 1,066,006 |
| GENERIC-FNO | 1.682 | 1.734 | 1.563 | 1.818 | 16 | 1,035,220 |
| Persistence (u(t) = u(0)) | 0.649 | 0.755 | 0.407 | 1.181 | 0 | 0 |
