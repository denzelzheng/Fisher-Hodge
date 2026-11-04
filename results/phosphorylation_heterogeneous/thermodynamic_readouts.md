| Method | quantity | unseen media, training conditions (ΔG 1–3) | unseen media, drive ΔG 3–4 | unseen media, stronger, finer potential U | unseen media, 16×16 grid |
|---|---|---|---|---|---|
| Neural Chart | σ, relative error | 1.312 | 0.654 | 0.615 | 1.070 |
| Neural Chart | cycle affinity, model; reference | 11.788; 5.984 | 13.208; 7.404 | 11.851; 6.047 | 11.977; 6.173 |
| Neural Chart, given standard potentials (oracle-type information) | σ, relative error | 0.340 | 0.485 | 0.315 | 0.424 |
| Neural Chart, given standard potentials (oracle-type information) | cycle affinity, model; reference | 5.984; 5.984 | 7.404; 7.404 | 6.047; 6.047 | 6.173; 6.173 |
| No chart (Flux-GNN) | σ, relative error | 0.478 | 0.551 | 0.457 | 0.475 |
| No chart (Flux-GNN) | cycle affinity, model; reference | 2.740; 5.984 | 2.810; 7.404 | 2.631; 6.047 | 3.185; 6.173 |
| CRNN (mass action) | σ, relative error | 0.838 | 0.823 | 0.823 | 0.834 |
| CRNN (mass action) | cycle affinity, model; reference | 2.329; 5.984 | 3.748; 7.404 | 2.392; 6.047 | 2.517; 6.173 |
| CRNN (condition features) | σ, relative error | 0.476 | 0.463 | 0.496 | 0.456 |
| CRNN (condition features) | cycle affinity, model; reference | 3.462; 5.984 | 5.332; 7.404 | 3.329; 6.047 | 4.293; 6.173 |
| Neural Chart, constant mobility per class | σ, relative error | 0.909 | 0.933 | 0.892 | 0.911 |
| Neural Chart, constant mobility per class | cycle affinity, model; reference | 11.777; 5.984 | 13.196; 7.404 | 11.839; 6.047 | 11.965; 6.173 |
| Neural Chart, no excess free energy | σ, relative error | 0.798 | 0.209 | 0.256 | 0.525 |
| Neural Chart, no excess free energy | cycle affinity, model; reference | 11.873; 5.984 | 13.293; 7.404 | 11.936; 6.047 | 12.062; 6.173 |
| Neural Chart, no convex initialisation | σ, relative error | 0.621 | 0.436 | 0.605 | 0.580 |
| Neural Chart, no convex initialisation | cycle affinity, model; reference | 2.319; 5.984 | 3.739; 7.404 | 2.382; 6.047 | 2.508; 6.173 |
| Neural Chart, reversible (no drive) | σ, relative error | diverged | diverged | diverged | diverged |
| Neural Chart, reversible (no drive) | cycle affinity, model; reference | 8.9e-16; 5.984 | 2.2e-16; 7.404 | 2.2e-15; 6.047 | -8.9e-16; 6.173 |
