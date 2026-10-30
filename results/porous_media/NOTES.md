# Notes

## Training times in the README

The README gives projected times for 4000 steps on one NVIDIA L40 with the recommended flags.

- Flux-GNN (`--compile_rates --fuse_net`): 1.0 s per step measured on an L40, 1.1 h for 4000 steps.
- UDE closure (`--control err_only --rtol 1e-3 --compile_rates`): 0.099 s per step measured on an L40, about 7 min.
- ≈ Neural Chart: measured 13.6 min (porous media) and 24.0 min (with crowding) on an RTX PRO 6000 Blackwell, without speed flags. The L40 time is not measured.
- FNO, BSMS-GNN, GENERIC-FNO and HelmFluid: measured on an L40 for the variant with crowding, 5.3, 2.0, 2.9 and 29.9 min. The models and the number of steps are the same for both variants.
