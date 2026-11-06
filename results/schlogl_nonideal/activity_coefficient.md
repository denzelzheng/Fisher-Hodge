| Method | RMSE, training range | largest error, training range | RMSE, extension to the high-concentration range | slope, learned against true |
|---|---|---|---|---|
| Neural Chart | 8.2e-03 | 0.013 | 0.032 | 1.015 |
| Neural Chart, given standard potentials (oracle-type information) | 9.4e-03 | 0.016 | 0.039 | 1.022 |
| Neural Chart, learned chemostat standard free energies | 0.031 | 0.062 | 0.086 | 0.976 |
| Neural Chart, constant mobility per class | 0.499 | 1.016 | 1.116 | -8.3e-01 |
| Neural Chart, no excess free energy | 0.477 | 0.928 | 1.028 | 0 |
| Neural Chart, no variable projection | 0.655 | 1.324 | 1.465 | -5.0e-01 |
| Neural Chart, reversible (no drive) | 1.072 | 1.891 | 2.023 | -1.1e+00 |
