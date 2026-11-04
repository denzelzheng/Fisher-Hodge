| Method | unseen media, training conditions (ΔG 1–3) | unseen media, drive ΔG 3–4 | unseen media, stronger, finer potential U | unseen media, 16×16 grid | trajectories with a negative value | parameters |
|---|---|---|---|---|---|---|
| Neural Chart | 0.029 | 0.036 | 0.047 | 0.033 | 0 | 6,155 |
| Neural Chart, given standard potentials (oracle-type information) | 0.030 | 0.036 | 0.046 | 0.034 | 0 | 6,150 |
| No chart (Flux-GNN) | 0.013 | 0.028 | 0.082 | 0.015 | 0 | 151,554 |
| CRNN (mass action) | 0.154 | 0.119 | 0.141 | 0.170 | 0 | 10 |
| CRNN (condition features) | 0.029 | 0.059 | 0.049 | 0.033 | 0 | 5,058 |
| UDE closure | 0.034 | 0.049 | 0.139 | 0.042 | 16 | 5,313 |
| FNO | 0.064 | 0.074 | 0.449 | 0.417 | 16 | 172,963 |
| BSMS-GNN | 0.983 | 0.753 | 0.963 | 5.009 | 16 | 1,392,899 |
| GENERIC-FNO | 0.942 | 1.002 | 0.967 | 0.864 | 16 | 1,005,316 |
| Persistence (u(t) = u(0)) | 0.840 | 0.888 | 0.930 | 0.858 | 0 | 0 |
