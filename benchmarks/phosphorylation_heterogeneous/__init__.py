"""Driven phosphorylation cycle on heterogeneous reaction media.

The network and the conditions of the phosphorylation cycle.  Every medium is a hidden periodic map; the rate constant of
every hyperedge is a nonlinear function of the map, independent of the conditions.  The models see 3 local descriptors per
hyperedge, computed from the map, not k and not the map.  Test sets on unseen media: training conditions, drive dG 3-4, a
stronger and finer potential U, and the 16 x 16 grid with the same spacing.  Code: benchmarks/reaction_networks."""
