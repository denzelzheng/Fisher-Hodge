# Notes

## Training times in the README

The README gives projected times for 4000 steps on one NVIDIA L40 with the recommended flags.

- UDE closure (`--control err_only --compile_rates`): 6.4 min measured on an L40.
- Flux-GNN: 36.0 min measured on an L40 (without `--fuse_net`).
- FNO, BSMS-GNN, GENERIC-FNO: 5.3, 1.0 and 3.2 min measured on an L40.
- ≈ Neural Chart: 28.6 min measured on a CPU (8 threads). The GPU time is not measured.
