| Method | training conditions (ΔG 1–3) | drive ΔG 3–4 | stronger, finer potential U | 16×16 grid | trajectories with a negative value | parameters |
|---|---|---|---|---|---|---|
| Neural Chart | 3.3e-06 | 2.3e-06 | 2.6e-06 | 3.4e-06 | 0 | 4 |
| Oracle (true rates in the Neural Chart class) | 2.0e-06 | 1.7e-06 | 1.4e-06 | 1.9e-06 | 0 | 0 (true rates) |
| CRNN (mass action) | 0.017 | 0.018 | 0.021 | 0.025 | 0 | 10 |
| CRNN (condition features) | 5.2e-03 | 0.045 | 0.011 | 7.6e-03 | 0 | 4,866 |
| UDE closure | 0.018 | 0.057 | 0.130 | 0.026 | 16 | 5,121 |
| Flux-GNN | 2.1e-03 | 0.038 | 0.113 | 4.1e-03 | 0 | 151,170 |
| FNO | 0.022 | 0.050 | 0.600 | 0.534 | 16 | 172,867 |
| BSMS-GNN | 0.180 | 0.189 | 0.449 | 0.350 | 16 | 1,392,515 |
| GENERIC-FNO | 1.049 | 0.967 | 1.034 | 0.904 | 16 | 1,005,172 |
| Persistence (u(t) = u(0)) | 0.836 | 0.909 | 0.917 | 0.795 | 0 | 0 |
