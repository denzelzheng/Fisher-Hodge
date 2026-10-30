| Method | what it removes | training conditions | temperature ×0.25 | temperature ×4 | drive ×4 | drive ×8 | unseen low-porosity media | 64×64 grid | point cloud |
|---|---|---|---|---|---|---|---|---|---|
| Neural Chart | full model | 0.067 | 0.114 | 0.101 | 0.168 | 0.376 | 0.062 | 0.077 | 0.091 |
| Neural Chart, no state input | the state input s * m(log state) of the mobility | 0.241 | 0.318 | 0.144 | 0.412 | 0.601 | 0.127 | 0.254 | 0.250 |
| Oracle (true rates in the Neural Chart class) | nothing learned: the true rates in the Neural Chart class | 0.049 | 0.069 | 0.033 | 0.103 | 0.197 | 0.028 | 0.031 | 0.079 |
| Oracle without crowding | nothing learned: the true material mobility without the crowding factor | 0.241 | 0.704 | 0.094 | 0.692 | 0.920 | 0.224 | 0.262 | 0.254 |
