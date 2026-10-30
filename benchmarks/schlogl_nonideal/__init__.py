"""Schlögl model with non-ideal (crowded) activities.

The Schlögl network with mass action in the activity y = x gamma(x), ln gamma(x) = 0.25 x / (1 - 0.25 x).  The Neural
Chart learns ln gamma as a convex correction of the free energy.  Test sets: training conditions (a 2.85–3.10,
b 1.0–3.2), higher concentrations (a 3.5–4.0, b 2.0–3.5), and the 16 x 16 grid.  Code: benchmarks/reaction_networks."""
