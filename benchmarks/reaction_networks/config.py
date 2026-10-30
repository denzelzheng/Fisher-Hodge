"""Settings of the four reaction-network benchmarks.  configure(name) selects one; the other modules of this package
read the module-level values (SYSTEM, DT, N_TRAIN, W_TRAIN, DATA, ...).

    schlogl                 Schloegl model, bistable, on the 8 x 8 periodic grid (DT 0.05)
    schlogl_nonideal        Schloegl model with crowded (non-ideal) activities, 8 x 8 grid (DT 0.05)
    phosphorylation_cycle   chemostat-driven phosphorylation cycle in a potential, 8 x 8 grid (DT 0.05)
    mapk_cascade            MAPK cascade with three tiers, 4 x 4 grid (DT 0.5)

Protocol (every model, seed 0): training window 100 sampling steps (MAPK: 20 steps, also 10 time units), rollouts of
K in [4, 20] steps, batch 8, Adam lr 1e-3 with a cosine schedule on p = max(step / 4000, elapsed / 12 h), gradient
clip 1.  Test rollouts run for 10T = 200 sampling steps.  Integration: neuralchart.fast, sample_err, float32 in training
and evaluation, rtol 1e-4 (chosen with the `calibrate` command: the oracle's 10T error is at the integrator floor);
substep cap 2000 per sampling interval in training and 4000 in evaluation.
"""
import os

from ..common import ROOT

BENCHMARKS = ('schlogl', 'schlogl_nonideal', 'phosphorylation_cycle', 'mapk_cascade')
SYSTEM = 'schlogl_nonideal'
DT, N_TRAIN, W_TRAIN = 0.05, 8, 100
T_TEST, T_STEPS = 200, 20                  # 10T, sampling steps per time unit T
GRID = dict(schlogl=(0.05, 8, 100), schlogl_nonideal=(0.05, 8, 100), phosphorylation_cycle=(0.05, 8, 100),
            mapk_cascade=(0.5, 4, 20))      # DT, training grid, training window
STEP = dict(dt_max=DT / 8, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9)
MAXSUB_TRAIN, MAXSUB_EVAL = 2000, 4000
ITERS, LR, BATCH, KMIN, KMAX = 4000, 1e-3, 8, 4, 20
BUDGET_S = 12 * 3600
RTOL = 1e-4
FAST = None                                # the integrator settings of a run (neuralchart.fast.accelerate kwargs)
DATA = None
OUT = None

# test sets: key -> (label, (count, steps, grid n, random seed)).  Ranges are those of data.py.
TEST_SETS = {
    'schlogl': {
        'train_conditions': 'training conditions',
        'high_a_low_b': 'a 3.15–3.35, b 0.3–1.0',
        'planar_fronts': 'planar fronts, a = 3, b 1.7–2.3',
        'grid16': '16×16 grid',
    },
    'schlogl_nonideal': {
        'train_conditions': 'training conditions',
        'high_concentration': 'a 3.5–4.0, b 2.0–3.5',
        'grid16': '16×16 grid',
    },
    'phosphorylation_cycle': {
        'train_conditions': 'training conditions (ΔG 1–3)',
        'strong_drive': 'drive ΔG 3–4',
        'strong_potential': 'stronger, finer potential U',
        'grid16': '16×16 grid',
    },
    'mapk_cascade': {
        'train_conditions': 'training conditions (ATP 0.5–2)',
        'high_atp': 'ATP 3–5',
        'low_atp': 'ATP 0.15–0.3',
        'phosphorylated_start': 'phosphorylated initial states',
    },
}
TITLES = dict(schlogl='Schlögl model: bistability and travelling fronts',
              schlogl_nonideal='Schlögl model with non-ideal (crowded) activities',
              phosphorylation_cycle='Driven phosphorylation cycle', mapk_cascade='MAPK cascade')


def configure(system, data=None, out=None):
    """select a benchmark.  data: directory of its data sets (default $NEURALCHART_DATA/<system> if NEURALCHART_DATA is
    set, else <repo>/data/<system>); out: directory of training and evaluation outputs (default <repo>/runs/<system>)"""
    global SYSTEM, DT, N_TRAIN, W_TRAIN, DATA, OUT
    assert system in BENCHMARKS, system
    SYSTEM = system
    DT, N_TRAIN, W_TRAIN = GRID[system]
    STEP['dt_max'] = DT / 8
    DATA = data or os.path.join(os.environ.get('NEURALCHART_DATA') or os.path.join(ROOT, 'data'), system)
    OUT = out or os.path.join(ROOT, 'runs', system)
    return system


def sets():
    return list(TEST_SETS[SYSTEM])


configure(SYSTEM)
