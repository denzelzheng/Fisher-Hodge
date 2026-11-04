"""Composition benchmarks: modules fitted on their own data, the composite tested on data neither module saw.

The design below was fixed before any result.

Main table.
  compose_reaction_transport   reaction module: the phosphorylation cycle on one site, well mixed (X, Xp, E, chemostats ATP, ADP).
                               transport module: X and Xp on the 8 x 8 periodic grid in a potential, no reaction.  Glued along X and Xp.
                               Tested on the sets of phosphorylation_cycle.  Gluing creates new cycles.  Every species of the transport
                               module is in the reaction module, so the cross terms of refine() vanish by construction.
  compose_cycle_cycle          two phosphorylation cycles on one site: X, Xp with the kinase E; Y, Yp with the kinase K glued to Xp; ATP and
                               ADP shared.  No new cycle.  Tested on a coupled cascade: training conditions, strong drive, and a start at
                               1.5 x the concentrations.

Appendix: two cases fixed in advance to fail without refine().  They have the modules of compose_cycle_cycle.
  compose_thermo_cross         the coupled cascade with G_true = CHI c_X c_Y, CHI = 0.5 (d_x d_z G != 0).
  compose_kinetic_cross        the coupled cascade with the rate constant of the first step divided by 1 + c_Yp / K_I, K_I = 0.5.

Protocol.  Each module is fitted only on its own 48 x 100 steps, with the one fit of the library.  Models in each table:
  composed            the composite of the two fitted charts, no retraining
  composed-refine-N   the same plus refine() on N coupled trajectories of 20 steps (N = 2, 8, 32; 500 steps, the cross terms only)
  chart               the same chart fitted from scratch on 48 x 100 coupled steps
  flux_gnn-transfer   one Flux-GNN on the composite hypergraph, trained on the data of both modules in the composite input format (the
                      species a module does not have at 1e-6) and tested with no retraining
  flux_gnn            the same network trained on 48 x 100 coupled steps
Reported: the 10T error, the relative error of the entropy production, the affinity around the new cycles against the truth, and the
diagnostics of the composition.  On compose_reaction_transport the rows chart and flux_gnn are those of phosphorylation_cycle.
"""
