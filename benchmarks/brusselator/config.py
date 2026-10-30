"""The Brusselator benchmark: constants of the physics, the data and the protocol.

Physics (the reference, never shown to the models).  Four reactions at every site of a periodic n x n grid,
    R1 A -> X (k1 a),  R2 2X + Y -> 3X (k2 x^2 y),  R3 B + X -> Y + D (k3 b x),  R4 X -> E (k4 x),  k1 = ... = k4 = 1.
Every reverse rate is eps = 1e-3 (times the chemostatted product, d = e = 1), so all affinities are finite.  The
chemostats A, B, D, E are held fixed; a and b are the conditions.  X and Y diffuse with D_X = 0.4 and D_Y = 3.6
(transport rate D / h^2).  The Turing wavelength at a = 1 is lambda_c = 2 pi (D_X D_Y)^(1/4) = 6.88.  The training
grid is 32 x 32 with spacing h = 4 lambda_c / 32 = 0.860, and the 64 x 64 test grid keeps h.  The homogeneous state
loses stability at the Hopf line b = 1 + a^2, and stationary patterns appear above the Turing threshold
b_T = (1 + a sqrt(D_X / D_Y))^2.  The reference solution uses DOP853 (rtol 1e-10, atol 1e-12) in float64, sampled
every DT = 0.2.

Protocol (every trained model, seed 0).  Training windows of 50 sampling steps (10 time units).  Rollout loss over
K in [4, 20] steps from a random start inside the window, batch 8, Adam with learning rate 1e-3 and a cosine
schedule, 4000 steps or 12 h of wall clock, gradient clip 1.  The continuous-time models use the positive adaptive
SSP-RK2 of neuralchart.fast with error control (rtol 5e-5) in float32; the substep cap is 2000 per sampling interval
in training and 4000 in evaluation.  FNO, BSMS-GNN and GENERIC-FNO are one-step maps.
"""
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(os.environ.get('NEURALCHART_DATA', os.path.join(ROOT, 'data')), 'brusselator')
RESULTS = os.path.join(ROOT, 'results', 'brusselator')
CHECKPOINTS = os.path.join(ROOT, 'checkpoints', 'brusselator')      # the shipped Neural Chart checkpoints
RUNS = os.path.join(ROOT, 'runs', 'brusselator')                    # outputs of train and evaluate

DT, W_TRAIN, T_TEST, N_TRAIN, T_STEPS = 0.2, 50, 200, 32, 20      # T = 20 sampling steps, 10T = 200
KF, EPS, DIFF = (1.0, 1.0, 1.0, 1.0), 1e-3, (0.4, 3.6)
LAMBDA_C = 2 * math.pi * (DIFF[0] * DIFF[1]) ** 0.25                # Turing wavelength at a = 1
H_GRID = 4 * LAMBDA_C / N_TRAIN                                    # grid spacing (32 x 32 holds 4 wavelengths)
SPECIES = ['X', 'Y', 'A', 'B', 'D', 'E']
CHEM = ['A', 'B', 'D', 'E']
REACTIONS = [({'A': 1}, {'X': 1}), ({'X': 2, 'Y': 1}, {'X': 3}), ({'B': 1, 'X': 1}, {'Y': 1, 'D': 1}), ({'X': 1}, {'E': 1})]

STEP = dict(dt_max=DT / 8, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9)
MAXSUB_TRAIN = 2000
MAXSUB_EVAL = 4000
RTOL = 5e-5
ITERS, LR, BATCH, KMIN, KMAX = 4000, 1e-3, 8, 4, 20
BUDGET_S = 12 * 3600
