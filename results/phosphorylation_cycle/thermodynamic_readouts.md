| Method | quantity | training conditions (ΔG 1–3) | drive ΔG 3–4 | stronger, finer potential U | 16×16 grid |
|---|---|---|---|---|---|
| Neural Chart | σ, relative error | 2.1e-03 | 1.3e-03 | 2.0e-03 | 2.5e-03 |
| Neural Chart | cycle affinity, model; reference | 5.945; 5.960 | 7.462; 7.477 | 5.975; 5.990 | 5.308; 5.323 |
| Neural Chart, given standard potentials (oracle-type information) | σ, relative error | 2.2e-04 | 2.2e-04 | 2.1e-04 | 2.1e-04 |
| Neural Chart, given standard potentials (oracle-type information) | cycle affinity, model; reference | 5.960; 5.960 | 7.477; 7.477 | 5.990; 5.990 | 5.323; 5.323 |
| No chart (Flux-GNN) | σ, relative error | 0.362 | 0.594 | 0.332 | 0.396 |
| No chart (Flux-GNN) | cycle affinity, model; reference | 3.394; 5.960 | 3.224; 7.477 | 3.413; 5.990 | 2.858; 5.323 |
| CRNN (mass action) | σ, relative error | 0.662 | 0.492 | 0.618 | 0.730 |
| CRNN (mass action) | cycle affinity, model; reference | 2.603; 5.960 | 4.120; 7.477 | 2.633; 5.990 | 1.966; 5.323 |
| CRNN (condition features) | σ, relative error | 0.503 | 0.489 | 0.479 | 0.559 |
| CRNN (condition features) | cycle affinity, model; reference | 2.733; 5.960 | 3.764; 7.477 | 2.760; 5.990 | 2.206; 5.323 |
| Neural Chart, ideal activities | σ, relative error | 2.1e-03 | 1.3e-03 | 2.0e-03 | 2.5e-03 |
| Neural Chart, ideal activities | cycle affinity, model; reference | 5.945; 5.960 | 7.462; 7.477 | 5.975; 5.990 | 5.308; 5.323 |
| Neural Chart, no convex initialisation | σ, relative error | 0.710 | 0.467 | 0.658 | 0.796 |
| Neural Chart, no convex initialisation | cycle affinity, model; reference | 2.103; 5.960 | 3.620; 7.477 | 2.133; 5.990 | 1.466; 5.323 |
| Neural Chart, reversible (no drive) | σ, relative error | 0.979 | 0.976 | 0.932 | 0.973 |
| Neural Chart, reversible (no drive) | cycle affinity, model; reference | -2.5e-16; 5.960 | 3.6e-16; 7.477 | -4.7e-16; 5.990 | -1.1e-15; 5.323 |
