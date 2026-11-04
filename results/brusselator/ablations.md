| Variant | what it removes | training conditions | b 3.2–3.7 | b 4.0–4.6 | a 1.3–1.5 (a and b outside training) | Turing regime (below the Hopf line) | 64×64 grid | parameters |
|---|---|---|---|---|---|---|---|---|
| Neural Chart | nothing (the method) | 2.5e-03 | 3.0e-03 | 3.4e-03 | 3.6e-03 | 2.0e-03 | 0.013 | 206 |
| Neural Chart, given standard potentials (oracle-type information) | nothing: it adds the standard potentials from thermochemical data (oracle-type information) | 1.6e-04 | 4.3e-05 | 3.5e-05 | 1.2e-04 | 3.6e-05 | 1.7e-04 | 200 |
| Neural Chart, ideal activities | the activity correction: ln gamma = 0 | 2.6e-03 | 3.0e-03 | 3.4e-03 | 3.7e-03 | 2.1e-03 | 0.013 | 12 |
| Neural Chart, reversible (no drive) | the drive: A = 0, so the model is reversible | 0.702 | 0.776 | 0.850 | 0.852 | 0.882 | 0.695 | 206 |
| No chart (Flux-GNN) | the chart: a graph network gives the one-way rates, with no chemical potential, force or symmetric mobility | 0.075 | 0.499 | 0.639 | 0.638 | 0.292 | 0.201 | 151,170 |
