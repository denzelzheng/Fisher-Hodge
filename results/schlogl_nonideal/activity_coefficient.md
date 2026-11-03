| Method | RMSE, training range | largest error, training range | RMSE, extension to the high-concentration range | slope, learned against true |
|---|---|---|---|---|
| Neural Chart | 3.6e-03 | 5.8e-03 | 0.025 | 0.998 |
| Neural Chart, given standard potentials (oracle-type information) | 2.7e-03 | 0.013 | 0.034 | 0.997 |
| Neural Chart, ideal activities | 0.477 | 0.928 | 1.028 | 0 |
| Neural Chart, ideal-fit initialisation | 0.041 | 0.149 | 0.219 | 0.882 |
| Neural Chart, no convex initialisation | 0.537 | 1.018 | 1.121 | -8.6e-02 |
| Neural Chart, reversible (no drive) | 1.220 | 2.001 | 2.115 | -1.1e+00 |
| Oracle (true rates in the Neural Chart class) | 0 | 0 | 0 | 1.000 |
