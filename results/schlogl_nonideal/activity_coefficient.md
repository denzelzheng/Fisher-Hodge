| Method | RMSE, training range | largest error, training range | RMSE, extension to the high-concentration range | slope, learned against true |
|---|---|---|---|---|
| Neural Chart | 0.177 | 0.288 | 0.330 | 0.795 |
| Neural Chart, given standard potentials (oracle-type information) | 0.051 | 0.143 | 0.209 | 0.859 |
| Neural Chart, constant mobility per class | 0.120 | 0.156 | 0.197 | 1.018 |
| Neural Chart, no excess free energy | 0.477 | 0.928 | 1.028 | 0 |
| Neural Chart, no convex initialisation | 0.269 | 0.545 | 0.617 | 0.432 |
| Neural Chart, reversible (no drive) | 0.422 | 0.890 | 0.998 | -1.4e-02 |
