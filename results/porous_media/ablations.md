| Method | what it removes | training conditions | temperature ×0.25 | temperature ×4 | drive ×4 | drive ×8 | unseen low-porosity media | 64×64 grid | point cloud |
|---|---|---|---|---|---|---|---|---|---|
| Neural Chart | full model | 0.064 | 0.084 | 0.090 | 0.195 | 0.407 | 0.081 | 0.071 | 0.090 |
| Neural Chart, no state input | the state input s * m(log state) of the mobility | 0.070 | 0.084 | 0.099 | 0.199 | 0.404 | 0.085 | 0.077 | 0.092 |
| Neural Chart, constant mobility | the material network g_theta: one constant omega for all edges | 1.180 | 1.067 | 0.943 | 1.065 | 1.012 | 0.511 | 1.180 | 1.180 |
| Neural Chart, learned drive scale | the fixed drive scale: a global scale of the drive is learned | 0.063 | 0.084 | 0.090 | 0.192 | 0.403 | 0.080 | 0.071 | 0.090 |
| Oracle (true rates in the Neural Chart class) | nothing learned: the true rates in the Neural Chart class | 0.046 | 0.107 | 0.032 | 0.143 | 0.231 | 0.037 | 0.034 | 0.083 |
