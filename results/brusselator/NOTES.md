# Notes

The FNO and BSMS-GNN rows come from runs evaluated on a different GPU model (RTX A6000). Evaluated again with this code on an RTX PRO 6000 Blackwell, their 10T errors differ by FP32 rounding accumulated over the 200 one-step updates: at most 0.7 % for FNO and 4e-6 (relative) for BSMS-GNN.

## Uniform starts in the Turing regime

Half of the 32 test trajectories of the Turing regime (below the Hopf line) start from a spatially uniform state.
In exact arithmetic such a trajectory stays uniform.
The uniform state is unstable to spatial perturbations there (the Turing instability), so in FP32 integration rounding noise can break the symmetry and grow into a pattern within 10T.
The error of these trajectories therefore depends on rounding as well as on the model.
The table gives the 10T error, median over the 16 uniform starts and over the 16 spatial starts (from `oscillation_patterns.md`).

| Method | 16 uniform starts | 16 spatial starts |
|---|---|---|
| Neural Chart | 7.1e-06 | 1.7e-04 |
| Oracle (true rates in the Neural Chart class) | 2.0e-06 | 2.8e-05 |
| CRNN (mass action) | 0.531 | 8.5e-03 |
| CRNN (condition features) | 0.573 | 0.787 |
| UDE closure | 1.460 | 1.367 |
| Flux-GNN | 0.069 | 0.494 |
| BSMS-GNN | 1.906 | 1.801 |
| FNO | 0.138 | 0.565 |
| GENERIC-FNO | 0.499 | 0.607 |
| Persistence (u(t) = u(0)) | 0.499 | 0.607 |

The 0.026 of CRNN (mass action) in the Turing column of `main.md` is the median over all 32 trajectories, which mixes the two groups.

## Training times in the README

The README gives projected times for 4000 steps on one NVIDIA L40 with the recommended flags.

- Flux-GNN (`--compact --compile_rates --fuse_net`): 5.9 s per step measured on an L40, 6.6 h for 4000 steps.
- UDE closure (`--control err_only --compact --compile_rates`): 0.51 s per step measured on an L40, about 35 min.
- ≈ Neural Chart: measured 21.6 min on an RTX PRO 6000 Blackwell (no speed flags). The L40 time is not measured. The variant without convex initialisation needs more substeps and took 100 min.
- ≈ FNO, BSMS-GNN, GENERIC-FNO: measured 8.2, 5.5 and 3.4 min, FNO and BSMS-GNN on an RTX A6000, GENERIC-FNO on an unrecorded GPU.
