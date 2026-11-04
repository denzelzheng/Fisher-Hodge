| Method | quantity | training conditions (ΔG 1–3) | drive ΔG 3–4 | stronger, finer potential U | X and Xp at the start × 3 | 16×16 grid |
|---|---|---|---|---|---|---|
| Neural Chart | σ, relative error | 0.578 | 0.446 | 0.552 | 0.591 | 0.554 |
| Neural Chart | cycle affinity, model; reference | 2.951; 5.879 | 4.531; 7.459 | 2.920; 5.848 | 2.937; 5.864 | 3.058; 5.986 |
| Neural Chart, given standard potentials (oracle-type information) | σ, relative error | 0.015 | 0.011 | 0.019 | 0.041 | 0.012 |
| Neural Chart, given standard potentials (oracle-type information) | cycle affinity, model; reference | 5.879; 5.879 | 7.459; 7.459 | 5.848; 5.848 | 5.864; 5.864 | 5.986; 5.986 |
| No chart (Flux-GNN) | σ, relative error | 0.511 | 0.691 | 0.474 | 0.361 | 0.521 |
| No chart (Flux-GNN) | cycle affinity, model; reference | 2.942; 5.879 | 2.975; 7.459 | 2.977; 5.848 | 3.284; 5.864 | 2.979; 5.986 |
| CRNN (mass action) | σ, relative error | 0.665 | 0.448 | 0.627 | 0.574 | 0.634 |
| CRNN (mass action) | cycle affinity, model; reference | 2.219; 5.879 | 3.799; 7.459 | 2.188; 5.848 | 2.205; 5.864 | 2.326; 5.986 |
| CRNN (condition features) | σ, relative error | 0.666 | 0.691 | 0.591 | 0.560 | 0.673 |
| CRNN (condition features) | cycle affinity, model; reference | 2.431; 5.879 | 3.048; 7.459 | 2.417; 5.848 | 2.424; 5.864 | 2.482; 5.986 |
| Neural Chart, constant mobility per class | σ, relative error | 0.591 | 0.430 | 0.560 | 0.561 | 0.567 |
| Neural Chart, constant mobility per class | cycle affinity, model; reference | 3.029; 5.879 | 4.609; 7.459 | 2.998; 5.848 | 3.015; 5.864 | 3.136; 5.986 |
| Neural Chart, no excess free energy | σ, relative error | 0.574 | 0.440 | 0.539 | 0.559 | 0.553 |
| Neural Chart, no excess free energy | cycle affinity, model; reference | 2.986; 5.879 | 4.565; 7.459 | 2.955; 5.848 | 2.971; 5.864 | 3.093; 5.986 |
| Neural Chart, no convex initialisation | σ, relative error | 0.753 | 0.529 | 0.716 | 0.725 | 0.718 |
| Neural Chart, no convex initialisation | cycle affinity, model; reference | 1.896; 5.879 | 3.476; 7.459 | 1.865; 5.848 | 1.882; 5.864 | 2.003; 5.986 |
| Neural Chart, reversible (no drive) | σ, relative error | 115.114 | 244.615 | 174.830 | 17.002 | 846.600 |
| Neural Chart, reversible (no drive) | cycle affinity, model; reference | -3.3e-16; 5.879 | 5.0e-16; 7.459 | 2.8e-16; 5.848 | 2.9e-16; 5.864 | 2.8e-16; 5.986 |
