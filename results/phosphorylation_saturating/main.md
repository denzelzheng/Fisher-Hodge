| Method | training conditions (ΔG 1–3) | drive ΔG 3–4 | stronger, finer potential U | X and Xp at the start × 3 | 16×16 grid | trajectories with a negative value | parameters |
|---|---|---|---|---|---|---|---|
| Neural Chart | 1.3e-03 | 4.6e-03 | 9.8e-03 | 0.019 | 1.8e-03 | 0 | 5,963 |
| Neural Chart, given standard potentials (oracle-type information) | 1.0e-03 | 4.3e-03 | 8.3e-03 | 0.016 | 1.4e-03 | 0 | 5,958 |
| No chart (Flux-GNN) | 2.3e-03 | 0.031 | 0.117 | 0.080 | 3.3e-03 | 0 | 151,170 |
| CRNN (mass action) | 0.046 | 0.037 | 0.146 | 0.215 | 0.054 | 0 | 10 |
| CRNN (condition features) | 0.028 | 0.052 | 0.093 | 0.210 | 0.043 | 0 | 4,866 |
| UDE closure | 0.020 | 0.053 | 0.110 | 0.090 | 0.028 | 16 | 5,121 |
| FNO | 0.019 | 0.049 | 0.580 | 0.381 | 0.509 | 15 | 172,867 |
| BSMS-GNN | 0.484 | 0.333 | 0.455 | 0.648 | 0.311 | 16 | 1,392,515 |
| GENERIC-FNO | 0.712 | 0.808 | 0.876 | 0.590 | 0.799 | 16 | 1,005,172 |
| Persistence (u(t) = u(0)) | 0.767 | 0.877 | 0.857 | 0.760 | 0.763 | 0 | 0 |
