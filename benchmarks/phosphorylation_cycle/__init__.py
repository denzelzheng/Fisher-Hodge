"""Driven phosphorylation cycle.

X + E + ATP <-> Xp + E + ADP and Xp <-> X on the 8 x 8 periodic grid; X and Xp diffuse in a smooth potential U; the drive
is dG = log(ATP / ADP).  Test sets: training conditions (dG 1–3, potential amplitude 0.7), drive dG 3–4, a stronger and
finer potential (amplitude 1.2), and the 16 x 16 grid.  Code: benchmarks/reaction_networks."""
