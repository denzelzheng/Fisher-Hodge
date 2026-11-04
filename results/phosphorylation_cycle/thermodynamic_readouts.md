| Method | quantity | training conditions (ΔG 1–3) | drive ΔG 3–4 | stronger, finer potential U | 16×16 grid |
|---|---|---|---|---|---|
| Neural Chart | σ, relative error | 2.3e-03 | 1.4e-03 | 2.1e-03 | 2.6e-03 |
| Neural Chart | cycle affinity, model; reference | 5.944; 5.960 | 7.461; 7.477 | 5.975; 5.990 | 5.307; 5.323 |
| Neural Chart, given standard potentials (oracle-type information) | σ, relative error | 2.4e-04 | 3.1e-04 | 1.9e-04 | 2.1e-04 |
| Neural Chart, given standard potentials (oracle-type information) | cycle affinity, model; reference | 5.960; 5.960 | 7.477; 7.477 | 5.990; 5.990 | 5.323; 5.323 |
| No chart (Flux-GNN) | σ, relative error | 0.362 | 0.594 | 0.332 | 0.396 |
| No chart (Flux-GNN) | cycle affinity, model; reference | 3.394; 5.960 | 3.224; 7.477 | 3.413; 5.990 | 2.858; 5.323 |
| CRNN (mass action) | σ, relative error | 0.662 | 0.492 | 0.618 | 0.730 |
| CRNN (mass action) | cycle affinity, model; reference | 2.603; 5.960 | 4.120; 7.477 | 2.633; 5.990 | 1.966; 5.323 |
| CRNN (condition features) | σ, relative error | 0.503 | 0.489 | 0.479 | 0.559 |
| CRNN (condition features) | cycle affinity, model; reference | 2.733; 5.960 | 3.764; 7.477 | 2.760; 5.990 | 2.206; 5.323 |
| Neural Chart, constant mobility per class | σ, relative error | 2.3e-03 | 1.5e-03 | 2.1e-03 | 2.7e-03 |
| Neural Chart, constant mobility per class | cycle affinity, model; reference | 5.944; 5.960 | 7.462; 7.477 | 5.975; 5.990 | 5.307; 5.323 |
| Neural Chart, no excess free energy | σ, relative error | 2.2e-03 | 1.4e-03 | 2.1e-03 | 2.6e-03 |
| Neural Chart, no excess free energy | cycle affinity, model; reference | 5.944; 5.960 | 7.461; 7.477 | 5.975; 5.990 | 5.307; 5.323 |
| Neural Chart, no convex initialisation | σ, relative error | 0.701 | 0.479 | 0.646 | 0.787 |
| Neural Chart, no convex initialisation | cycle affinity, model; reference | 1.997; 5.960 | 3.514; 7.477 | 2.028; 5.990 | 1.360; 5.323 |
| Neural Chart, reversible (no drive) | σ, relative error | diverged | diverged | diverged | diverged |
| Neural Chart, reversible (no drive) | cycle affinity, model; reference | -3.6e-16; 5.960 | 6.2e-16; 7.477 | -3.3e-16; 5.990 | 8.3e-16; 5.323 |
