"""The Neural Chart: one model class, one configuration per benchmark, one fit.

    a_h, b_h = k_h exp(+- f_h / 2)                    (each rate also carries the mass-action factor of its reactants)
    f   fixed by physics: the conditions (temperature, drive, chemostat potentials) and thermodynamic data (U, mu0)
    log k_h = log omega_c(h) + g_theta(xi_h) + s * m(log s_h)     (times fixed geometric and physical prefactors)
        omega_c   one positive constant per hyperedge class
        g_theta   an MLP of the local material descriptors xi; its last layer is zero at initialisation (absent
                  without descriptors)
        s * m     m non-decreasing with linear tails, s = 0 at initialisation (absent without state features,
                  e.g. the log of the mean edge density in porous media)
    Fit: at initialisation k is linear in omega, so omega gets its closed-form NNLS estimate (variable projected when
    mu carries a learned activity correction); then every parameter is refined on rollouts (neuralchart.fit).

    from neuralchart import NeuralChart, chart_config, fit_chart
    cfg = chart_config('brusselator', 'chart')
    model = NeuralChart.build(cfg, n=32, norm=norm)     # task inputs: see the benchmark's build_chart
    model.accelerate()                                  # the integrator of the configuration (neuralchart.fast)
    info = fit_chart(model, train_data, dt, iters=4000, seed=0)

Every ablation is 'chart' with exactly one override (CONFIGS).
"""
import dataclasses
import importlib
from typing import Optional

from .model import MasterEquation
from .fit import fit_chart  # noqa: F401  (re-exported)

TASKS = ('porous_media', 'brusselator', 'schlogl', 'schlogl_nonideal', 'phosphorylation_cycle', 'mapk_cascade')
REACTION_TASKS = ('schlogl', 'schlogl_nonideal', 'phosphorylation_cycle', 'mapk_cascade')
# the module that assembles the Neural Chart of each task (build_chart(cfg, **task inputs))
BUILDERS = dict(porous_media='benchmarks.porous_media.models', brusselator='benchmarks.brusselator.models',
                **{t: 'benchmarks.reaction_networks.models' for t in REACTION_TASKS})


@dataclasses.dataclass(frozen=True)
class ChartConfig:
    """one Neural Chart.
    mu0          'given' (thermochemical data) or 'learned' (ablation)
    drive        'fixed' (the physical drive), 'off' (A = 0: the reversible ablation) or 'learned_scale' (a learned
                 global drive scale)
    descriptors  g_theta(xi) in log k;  state: s * m(log state) in log k (needs descriptors)
    kappa        'learned', 'true' (the true k: oracle) or 'true_material' (porous media with crowding: the true
                 material mobility without the crowding factor)
    init         'nnls' (closed-form omega) or 'one' (omega = 1: the ablation without convex initialisation)
    activity     reaction networks: 'ideal' (mu = log c + mu0), 'learned' (mu = log c + mu0 + ln gamma_theta(c)) or
                 'true' (the generator's ln gamma, oracle only)
    varpro       the closed-form step is variable projected (no effect without a learned state function)
    fast, control, rtol, precision: the integrator (neuralchart.fast)"""
    task: str
    mu0: str = 'given'
    drive: str = 'fixed'
    descriptors: bool = False
    state: bool = False
    kappa: str = 'learned'
    init: str = 'nnls'
    train: bool = True
    fast: bool = True
    control: str = 'sample_err'
    rtol: float = 5e-5
    precision: Optional[str] = 'fp32'
    activity: str = 'ideal'
    varpro: bool = True

    def __post_init__(s):
        assert s.task in TASKS and s.mu0 in ('given', 'learned') and s.drive in ('fixed', 'off', 'learned_scale')
        assert s.activity in ('ideal', 'learned', 'true') and (s.activity == 'ideal' or s.task in REACTION_TASKS)
        assert s.kappa in ('learned', 'true', 'true_material') and s.init in ('nnls', 'one')
        assert s.descriptors or not s.state, 'the state term needs the descriptor model'
        assert s.task == 'porous_media' or not (s.descriptors or s.state), 'only porous media has descriptors'
        assert s.task == 'porous_media' or s.kappa != 'true_material'

    def replace(s, **kw):
        return dataclasses.replace(s, **kw)


_PM = ChartConfig(task='porous_media', descriptors=True, state=True, rtol=0.3)
_BR = ChartConfig(task='brusselator', rtol=5e-5)
_C5 = ChartConfig(task='schlogl_nonideal', activity='learned', rtol=1e-4)


def _reaction(task):
    base = ChartConfig(task=task, rtol=1e-4)
    return {
        'chart': base,
        'chart-learned-mu0': base.replace(mu0='learned'),
        'chart-zero-init': base.replace(init='one'),
        'chart-reversible': base.replace(drive='off'),
        'oracle': base.replace(kappa='true', train=False),
    }


# the method and its ablations, per task (porous_media covers both variants, with and without crowding)
CONFIGS = {
    'porous_media': {
        'chart': _PM,
        'chart-no-state': _PM.replace(state=False),
        'chart-const-k': _PM.replace(descriptors=False, state=False),
        'chart-learned-drive': _PM.replace(drive='learned_scale'),
        'oracle': _PM.replace(descriptors=False, state=False, kappa='true', train=False),
        'oracle-no-crowding': _PM.replace(descriptors=False, state=False, kappa='true_material', train=False),
    },
    'brusselator': {
        'chart': _BR,
        'chart-learned-mu0': _BR.replace(mu0='learned'),
        'chart-zero-init': _BR.replace(init='one'),
        'chart-reversible': _BR.replace(drive='off'),
        'oracle': _BR.replace(kappa='true', train=False),
    },
    'schlogl_nonideal': {
        'chart': _C5,
        'chart-ideal': _C5.replace(activity='ideal'),
        'chart-learned-mu0': _C5.replace(mu0='learned'),
        'chart-zero-init': _C5.replace(init='one'),
        'chart-reversible': _C5.replace(drive='off'),
        'chart-no-varpro': _C5.replace(varpro=False),
        'oracle': _C5.replace(activity='true', kappa='true', train=False),
    },
    'schlogl': _reaction('schlogl'),
    'phosphorylation_cycle': _reaction('phosphorylation_cycle'),
    'mapk_cascade': _reaction('mapk_cascade'),
}

# what each ablation removes (README and tables)
ABLATIONS = {
    'chart-no-state': 'the state input s * m(log state) of the mobility',
    'chart-const-k': 'the material network g_theta: one constant omega for all edges',
    'chart-learned-mu0': 'the given standard potentials: mu0 is learned together with k',
    'chart-zero-init': 'the closed-form NNLS start: omega starts at 1',
    'chart-reversible': 'the drive: A = 0, so the model is reversible',
    'chart-learned-drive': 'the fixed drive scale: a global scale of the drive is learned',
    'chart-ideal': 'the activity correction: ln gamma = 0',
    'chart-no-varpro': 'the variable projection: ideal NNLS start, then rollout refinement',
    'oracle': 'nothing learned: the true rates in the Neural Chart class',
    'oracle-no-crowding': 'nothing learned: the true material mobility without the crowding factor',
}


def is_chart(task, name):
    return name in CONFIGS[task]


def chart_config(task, name):
    return CONFIGS[task][name]


class NeuralChart(MasterEquation):
    """MasterEquation with a ChartConfig.  Build it with NeuralChart.build(cfg, **task inputs)."""
    config = None

    @classmethod
    def build(cls, cfg, **task_inputs):
        """cfg: a ChartConfig, or (task, name) with a name of CONFIGS[task]"""
        if not isinstance(cfg, ChartConfig):
            cfg = chart_config(*cfg)
        return importlib.import_module(BUILDERS[cfg.task]).build_chart(cfg, **task_inputs)

    def accelerate(s, **override):
        """the integration of the configuration (neuralchart.fast); a no-op when cfg.fast is False"""
        from . import fast
        cfg = s.config
        if cfg is None or not cfg.fast:
            return s
        kw = dict(control=cfg.control, rtol=cfg.rtol, precision=cfg.precision)
        kw.update(override)
        return fast.accelerate(s, **kw)
