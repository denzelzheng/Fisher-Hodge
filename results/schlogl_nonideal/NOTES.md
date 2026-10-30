# Notes

## Training times in the README

The README gives projected times for 4000 steps on one NVIDIA L40 with the recommended flags.

- UDE closure (`--control err_only --compile_rates`): measured on an L40, 3.8 to 3.9 min on each of the Schlögl, non-ideal Schlögl and phosphorylation-cycle benchmarks.
- ≈ Neural Chart: measured on a CPU (8 threads), 4.5 min (Schlögl), 16.2 min (non-ideal Schlögl) and 5.5 min (phosphorylation cycle). The GPU time is not measured.
- ≈ Flux-GNN: measured 16.0, 15.3 and 30.1 min on an RTX PRO 6000 Blackwell without `--compile_rates --fuse_net`.
- ≈ FNO, BSMS-GNN, GENERIC-FNO: measured 0.6 to 3.4 min, on an RTX PRO 6000 Blackwell (FNO, BSMS-GNN) and an L40 (GENERIC-FNO, and BSMS-GNN on the phosphorylation cycle).
