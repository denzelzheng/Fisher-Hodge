| Variant | what it removes | training conditions | b 3.2–3.7 | b 4.0–4.6 | a 1.3–1.5 (a and b outside training) | Turing regime (below the Hopf line) | 64×64 grid | parameters |
|---|---|---|---|---|---|---|---|---|
| Neural Chart | nothing (the method) | 1.6e-04 | 4.7e-05 | 3.7e-05 | 1.3e-04 | 3.8e-05 | 1.7e-04 | 6 |
| Neural Chart, learned standard potentials | the given standard potentials: mu0 is learned together with k | 2.6e-03 | 3.0e-03 | 3.4e-03 | 3.7e-03 | 2.1e-03 | 0.013 | 12 |
| Neural Chart, no convex initialisation | the closed-form NNLS start: omega starts at 1 | 0.490 | 0.711 | 0.842 | 0.575 | 0.283 | 0.504 | 6 |
| Neural Chart, reversible (no drive) | the drive: A = 0, so the model is reversible | 1.513 | 1.596 | 1.700 | 1.340 | 1.013 | 1.515 | 6 |
| Oracle (true rates in the Neural Chart class) | nothing learned: the true rates in the Neural Chart class | 1.4e-04 | 5.6e-05 | 2.6e-05 | 1.2e-04 | 6.4e-06 | 7.1e-05 | 0 (true rates) |
