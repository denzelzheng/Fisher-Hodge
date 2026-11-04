| Method | what it removes | training conditions | temperature ×0.25 | temperature ×4 | drive ×4 | drive ×8 | unseen low-porosity media | 64×64 grid | point cloud |
|---|---|---|---|---|---|---|---|---|---|
| Neural Chart | full model | 0.067 | 0.114 | 0.101 | 0.168 | 0.376 | 0.062 | 0.077 | 0.091 |
| Neural Chart, ideal activities | the activity correction: ln gamma = 0 | 0.067 | 0.114 | 0.101 | 0.168 | 0.376 | 0.062 | 0.077 | 0.091 |
| Neural Chart, no state input | the state input s * m(log state) of the mobility | 0.241 | 0.318 | 0.144 | 0.412 | 0.601 | 0.127 | 0.254 | 0.250 |
| No chart (Flux-GNN) | the chart: a graph network gives the one-way rates, with no chemical potential, force or symmetric mobility | 0.057 | 0.808 | 2.053 | 0.856 | 1.049 | 0.057 | 0.573 | 0.473 |
