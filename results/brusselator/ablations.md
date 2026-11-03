| Variant | what it removes | training conditions | b 3.2–3.7 | b 4.0–4.6 | a 1.3–1.5 (a and b outside training) | Turing regime (below the Hopf line) | 64×64 grid | parameters |
|---|---|---|---|---|---|---|---|---|
| Neural Chart | nothing (the method) | 2.6e-03 | 3.0e-03 | 3.4e-03 | 3.7e-03 | 2.1e-03 | 0.013 | 12 |
| Neural Chart, given standard potentials (oracle-type information) | nothing: it adds the standard potentials from thermochemical data (oracle-type information) | 1.6e-04 | 4.7e-05 | 3.7e-05 | 1.3e-04 | 3.8e-05 | 1.7e-04 | 6 |
| Neural Chart, no convex initialisation | the closed-form NNLS start: omega starts at 1 | 0.499 | 0.554 | 0.661 | 0.652 | 0.537 | 0.512 | 12 |
| Neural Chart, reversible (no drive) | the drive: A = 0, so the model is reversible | 0.702 | 0.776 | 0.850 | 0.852 | 0.882 | 0.695 | 12 |
| Oracle (true rates in the Neural Chart class) | nothing learned: the true rates in the Neural Chart class | 1.4e-04 | 5.6e-05 | 2.6e-05 | 1.2e-04 | 6.4e-06 | 7.1e-05 | 12 |
