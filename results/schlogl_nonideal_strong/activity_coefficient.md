| Method | RMSE, training range | largest error, training range | RMSE, extension to the high-concentration range | slope, learned against true |
|---|---|---|---|---|
| Neural Chart | 6.1e-03 | 9.9e-03 | 1.9e-03 | 0.993 |
| Neural Chart, given standard potentials (oracle-type information) | 3.2e-03 | 8.5e-03 | 0.014 | 1.002 |
| Neural Chart, ideal activities | 0.614 | 1.256 | 1.304 | 0 |
| Neural Chart, ideal-fit initialisation | 0.913 | 1.834 | 1.898 | -4.9e-01 |
| Neural Chart, no convex initialisation | 0.907 | 1.749 | 1.806 | -3.8e-01 |
| Neural Chart, reversible (no drive) | 0.614 | 1.256 | 1.304 | 0 |
| Oracle (true rates in the Neural Chart class) | 0 | 0 | 0 | 1.000 |
