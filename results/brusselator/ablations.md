| Variant | what it removes | training conditions | b 3.2–3.7 | b 4.0–4.6 | a 1.3–1.5 (a and b outside training) | Turing regime (below the Hopf line) | 64×64 grid | parameters |
|---|---|---|---|---|---|---|---|---|
| Neural Chart | nothing (the method) | 3.3e-03 | 2.1e-03 | 1.7e-03 | 5.2e-03 | 1.6e-03 | 0.013 | 5,998 |
| Neural Chart, given standard potentials (oracle-type information) | nothing: it adds the standard potentials from thermochemical data (oracle-type information) | 7.0e-05 | 1.9e-04 | 1.7e-04 | 1.5e-04 | 1.1e-04 | 1.7e-04 | 5,992 |
| Neural Chart, constant mobility per class | g_theta: one constant omega per hyperedge class | 3.2e-03 | 2.7e-03 | 2.4e-03 | 3.4e-03 | 1.6e-03 | 0.012 | 1,197 |
| Neural Chart, no excess free energy | the excess free energy G_theta | 2.2e-03 | 2.0e-03 | 3.2e-03 | 2.8e-03 | 2.1e-03 | 0.013 | 4,813 |
| Neural Chart, reversible (no drive) | the drive: A = 0, so the model is reversible | 0.570 | 0.670 | 0.721 | 0.706 | 0.680 | 0.573 | 5,998 |
| No chart (Flux-GNN) | the chart: a graph network gives the one-way rates, with no chemical potential, force or symmetric mobility | 0.075 | 0.499 | 0.639 | 0.638 | 0.292 | 0.201 | 151,170 |
