| Variant | what it removes | training conditions | b 3.2–3.7 | b 4.0–4.6 | a 1.3–1.5 (a and b outside training) | Turing regime (below the Hopf line) | 64×64 grid | parameters |
|---|---|---|---|---|---|---|---|---|
| Neural Chart | nothing (the method) | 2.5e-03 | 3.0e-03 | 3.4e-03 | 3.6e-03 | 2.0e-03 | 0.013 | 206 |
| Neural Chart, given standard potentials (oracle-type information) | nothing: it adds the standard potentials from thermochemical data (oracle-type information) | 0.026 | 0.082 | 0.154 | 0.153 | 0.037 | 0.049 | 200 |
| Neural Chart, ideal activities | the activity correction: ln gamma = 0 | 2.6e-03 | 3.0e-03 | 3.4e-03 | 3.7e-03 | 2.1e-03 | 0.013 | 12 |
| Neural Chart, reversible (no drive) | the drive: A = 0, so the model is reversible | 0.636 | 0.683 | 0.815 | 0.743 | 0.813 | 0.694 | 206 |
| Oracle (true rates in the Neural Chart class) | nothing learned: the true rates in the Neural Chart class | 1.4e-04 | 5.6e-05 | 2.6e-05 | 1.2e-04 | 6.4e-06 | 7.1e-05 | 12 |
