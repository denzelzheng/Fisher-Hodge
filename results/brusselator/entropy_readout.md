| Method | quantity | training conditions | b 3.2–3.7 | b 4.0–4.6 | a 1.3–1.5 (a and b outside training) | Turing regime (below the Hopf line) | 64×64 grid |
|---|---|---|---|---|---|---|---|
| Neural Chart | relative error of sigma(t) | 3.5e-05 | 3.7e-05 | 3.9e-05 | 3.7e-05 | 3.7e-05 | 3.4e-05 |
| Neural Chart | mean sigma, model / reference | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Neural Chart, learned standard potentials | relative error of sigma(t) | 0.227 | 0.297 | 0.309 | 0.271 | 0.244 | 0.209 |
| Neural Chart, learned standard potentials | mean sigma, model / reference | 1.201 | 1.244 | 1.273 | 1.257 | 1.241 | 1.204 |
| Neural Chart, no convex initialisation | relative error of sigma(t) | 27.425 | 26.759 | 26.290 | 26.777 | 27.211 | 27.295 |
| Neural Chart, no convex initialisation | mean sigma, model / reference | 28.693 | 28.060 | 27.611 | 27.975 | 28.217 | 28.300 |
| Neural Chart, reversible (no drive) | relative error of sigma(t) | 0.993 | 0.992 | 0.992 | 0.993 | 0.994 | 0.993 |
| Neural Chart, reversible (no drive) | mean sigma, model / reference | 6.7e-03 | 7.4e-03 | 7.9e-03 | 7.1e-03 | 6.2e-03 | 7.4e-03 |
| Oracle (true rates in the Neural Chart class) | relative error of sigma(t) | 7.4e-16 | 6.3e-16 | 5.7e-16 | 6.8e-16 | 6.1e-16 | 5.8e-16 |
| Oracle (true rates in the Neural Chart class) | mean sigma, model / reference | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| CRNN (mass action) | relative error of sigma(t) | 0.225 | 0.294 | 0.307 | 0.268 | 0.242 | 0.206 |
| CRNN (mass action) | mean sigma, model / reference | 1.199 | 1.242 | 1.270 | 1.254 | 1.238 | 1.202 |
| CRNN (condition features) | relative error of sigma(t) | 1.477 | 1.859 | 2.079 | 0.774 | 0.103 | 1.272 |
| CRNN (condition features) | mean sigma, model / reference | 2.469 | 2.928 | 3.569 | 1.676 | 0.958 | 2.269 |
| Flux-GNN | relative error of sigma(t) | 1.310 | 0.954 | 0.574 | 1.022 | 0.751 | 1.499 |
| Flux-GNN | mean sigma, model / reference | 2.325 | 1.997 | 1.257 | 2.018 | 0.253 | 2.506 |
