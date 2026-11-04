"""Driven phosphorylation cycle with a saturating rate constant.

The network and the conditions of the phosphorylation cycle, the rate constant of R1 multiplied by 1 / (1 + (c_X + c_Xp + c_E) / 2), symmetric
in the direction and independent of the conditions.  Test sets: training conditions, drive dG 3-4, a stronger and finer potential U,
X and Xp at the start x 3, and the 16 x 16 grid.  Code: benchmarks/reaction_networks."""
