| Method | RMSE, training range | largest error, training range | RMSE, extension to the high-concentration range | slope, learned against true |
|---|---|---|---|---|
| Neural Chart | 1.9e-03 | 3.0e-03 | 0.012 | 1.000 |
| Neural Chart, given standard potentials (oracle-type information) | 1.7e-03 | 8.9e-03 | 0.022 | 0.999 |
| Neural Chart, ideal activities | 0.477 | 0.928 | 1.028 | 0 |
| Neural Chart, ideal-fit initialisation | 0.032 | 0.058 | 0.104 | 0.956 |
| Neural Chart, no convex initialisation | 0.649 | 1.212 | 1.327 | -3.0e-01 |
| Neural Chart, reversible (no drive) | 0.477 | 0.928 | 1.028 | 0 |
| Oracle (true rates in the Neural Chart class) | 0 | 0 | 0 | 1.000 |
