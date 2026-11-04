| Method | quantity | training conditions (ATP 0.5–2) | ATP 3–5 | ATP 0.15–0.3 | phosphorylated initial states |
|---|---|---|---|---|---|
| Neural Chart | moieties, largest relative drift | 7.1e-06 | 5.3e-06 | 1.2e-05 | 5.3e-06 |
| Neural Chart | Kpp fraction at 10T, mean absolute error | 1.4e-03 | 7.0e-03 | 7.6e-04 | 1.3e-03 |
| Neural Chart, given standard potentials (oracle-type information) | moieties, largest relative drift | 9.3e-06 | 8.5e-06 | 1.4e-05 | 7.0e-06 |
| Neural Chart, given standard potentials (oracle-type information) | Kpp fraction at 10T, mean absolute error | 4.9e-03 | 2.5e-03 | 5.5e-05 | 5.0e-03 |
| No chart (Flux-GNN) | moieties, largest relative drift | 4.3e-06 | 2.4e-06 | 4.7e-06 | 3.9e-06 |
| No chart (Flux-GNN) | Kpp fraction at 10T, mean absolute error | 3.3e-03 | 0.103 | 0.127 | 3.5e-03 |
| CRNN (mass action) | moieties, largest relative drift | 6.8e-06 | 5.6e-06 | 9.6e-06 | 5.2e-06 |
| CRNN (mass action) | Kpp fraction at 10T, mean absolute error | 4.9e-03 | 0.012 | 7.9e-04 | 4.5e-03 |
| CRNN (condition features) | moieties, largest relative drift | 2.8e-06 | 5.5e-06 | 5.1e-06 | 2.5e-06 |
| CRNN (condition features) | Kpp fraction at 10T, mean absolute error | 5.1e-03 | 0.018 | 9.7e-03 | 5.0e-03 |
| UDE closure | moieties, largest relative drift | 6.0e-06 | 4.6e-06 | 2.3e-06 | 2.3e-06 |
| UDE closure | Kpp fraction at 10T, mean absolute error | 0.025 | 0.540 | 0.416 | 0.020 |
| FNO | moieties, largest relative drift | 0.241 | 0.554 | 0.226 | 0.345 |
| FNO | Kpp fraction at 10T, mean absolute error | 0.087 | 1.377 | 0.788 | 0.053 |
| BSMS-GNN | moieties, largest relative drift | 2.958 | 3.064 | 2.510 | 2.840 |
| BSMS-GNN | Kpp fraction at 10T, mean absolute error | 1.813 | 10.560 | 23.788 | 13.629 |
| GENERIC-FNO | moieties, largest relative drift | 1.572 | 1.912 | 1.615 | 1.306 |
| GENERIC-FNO | Kpp fraction at 10T, mean absolute error | 0.886 | 0.821 | 0.842 | 1.119 |
| Persistence (u(t) = u(0)) | moieties, largest relative drift | 0 | 0 | 0 | 0 |
| Persistence (u(t) = u(0)) | Kpp fraction at 10T, mean absolute error | 0.194 | 0.391 | 0.129 | 0.458 |
| Neural Chart, ideal activities | moieties, largest relative drift | 7.3e-06 | 5.6e-06 | 1.2e-05 | 5.1e-06 |
| Neural Chart, ideal activities | Kpp fraction at 10T, mean absolute error | 1.4e-03 | 7.0e-03 | 7.6e-04 | 1.3e-03 |
| Neural Chart, no convex initialisation | moieties, largest relative drift | 2.5e-05 | 2.0e-04 | 9.2e-06 | 2.3e-05 |
| Neural Chart, no convex initialisation | Kpp fraction at 10T, mean absolute error | 0.178 | 0.237 | 0.057 | 0.245 |
| Neural Chart, reversible (no drive) | moieties, largest relative drift | 4.3e-05 | 4.3e-05 | 3.9e-05 | 1.1e-05 |
| Neural Chart, reversible (no drive) | Kpp fraction at 10T, mean absolute error | 0.171 | 0.388 | 0.132 | 0.447 |
