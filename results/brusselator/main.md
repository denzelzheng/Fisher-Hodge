| Method | training conditions | b 3.2–3.7 | b 4.0–4.6 | a 1.3–1.5 (a and b outside training) | Turing regime (below the Hopf line) | 64×64 grid | trajectories with a negative value (max over sets) | parameters |
|---|---|---|---|---|---|---|---|---|
| Neural Chart | 1.6e-04 | 4.7e-05 | 3.7e-05 | 1.3e-04 | 3.8e-05 | 1.7e-04 | 0 | 6 |
| Oracle (true rates in the Neural Chart class) | 1.4e-04 | 5.6e-05 | 2.6e-05 | 1.2e-04 | 6.4e-06 | 7.1e-05 | 0 | 0 (true rates) |
| CRNN (mass action) | 2.4e-03 | 3.0e-03 | 3.4e-03 | 4.1e-03 | 0.026 | 0.013 | 0 | 12 |
| CRNN (condition features) | 0.724 | 0.755 | 0.771 | 0.802 | 0.698 | 0.749 | 0 | 4,866 |
| UDE closure | 0.550 | 0.622 | 0.689 | 1.004 | 1.408 | 0.573 | 32 | 4,929 |
| Flux-GNN | 0.075 | 0.499 | 0.639 | 0.638 | 0.292 | 0.201 | 0 | 151,170 |
| BSMS-GNN | 0.729 | 0.805 | 0.687 | 0.858 | 1.849 | 0.761 | 32 | 1,723,522 |
| FNO | 0.297 | 0.370 | 0.678 | 0.639 | 0.321 | 0.697 | 17 | 1,188,514 |
| GENERIC-FNO | 0.780 | 0.835 | 1.016 | 0.762 | 0.580 | 0.825 | 0 | 1,003,588 |
| Persistence (u(t) = u(0)) | 0.780 | 0.835 | 1.016 | 0.762 | 0.580 | 0.825 | 0 | 0 |
