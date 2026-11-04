"""The Neural Chart: one model class and one fit on every benchmark.  Only the hypergraph, the inputs, the data and the
numerics (rtol, integrator theta, trainer budget) differ per task.

    a_h, b_h = k_h eps_h exp(+- f_h / 2)              (eps_h = exp(<nu+ + nu-, y> / 2), y = log c + dG(c))
    f   the force, <nu+ - nu-, y + (U + mu0_theta) / T> + A; the conditions (temperature, drive, chemostat levels) enter only
        through U / T, the drive and the chemostat log-activities
    dG(c) = grad G_theta(c / V) - grad G_theta(0)    the excess potential; G_theta, the excess free energy, is one scalar MLP per
                                                     site over all local species
    log k_h = log omega_c(h) + g_theta(class one-hot, xi_h, log cbar_h)    (times T and the mesh weight)
        omega_c   one positive constant per hyperedge class
        g_theta   one MLP with a zero last layer; xi_h the local descriptors where they exist, cbar_h the mean of the local
                  density vector over the cells h touches
    Fit: at initialisation g = 0 and G = 0, so the rates are linear in omega: omega gets its closed-form NNLS estimate; then
    every parameter is refined on rollouts (neuralchart.fit).

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

TASKS = ('porous_media', 'brusselator', 'schlogl', 'schlogl_nonideal', 'schlogl_nonideal_strong', 'phosphorylation_cycle',
         'phosphorylation_heterogeneous', 'mapk_cascade')
REACTION_TASKS = ('schlogl', 'schlogl_nonideal', 'schlogl_nonideal_strong', 'phosphorylation_cycle',
                  'phosphorylation_heterogeneous', 'mapk_cascade')
# the module that assembles the Neural Chart of each task (build_chart(cfg, **task inputs))
BUILDERS = dict(porous_media='benchmarks.porous_media.models', brusselator='benchmarks.brusselator.models',
                **{t: 'benchmarks.reaction_networks.models' for t in REACTION_TASKS})


@dataclasses.dataclass(frozen=True)
class ChartConfig:
    """one Neural Chart.
    mu0          'learned' or 'given' (thermochemical data, an oracle-type input)
    drive        'fixed' (the physical drive) or 'off' (A = 0: the reversible ablation)
    kappa        'learned' (omega and g_theta), 'constant' (omega alone), 'true' (the true k: oracle) or 'true_material' (porous
                 media with crowding: the true material mobility without the crowding factor)
    init         'nnls' (closed-form omega) or 'one' (omega = 1: the ablation without convex initialisation)
    excess       'learned' (G_theta), 'none' (no excess free energy) or 'true' (the generator's, oracle only)
    fast, control, rtol, precision: the integrator (neuralchart.fast)"""
    task: str
    mu0: str = 'learned'
    drive: str = 'fixed'
    kappa: str = 'learned'
    init: str = 'nnls'
    train: bool = True
    fast: bool = True
    control: str = 'sample_err'
    rtol: float = 1e-4
    precision: Optional[str] = 'fp32'
    excess: str = 'learned'
    varpro: bool = False

    def __post_init__(s):
        assert s.task in TASKS and s.mu0 in ('given', 'learned') and s.drive in ('fixed', 'off')
        assert s.excess in ('none', 'learned', 'true') and s.kappa in ('learned', 'constant', 'true', 'true_material')
        assert s.init in ('nnls', 'one')

    def replace(s, **kw):
        return dataclasses.replace(s, **kw)


def _ablations(base, oracle_excess='none', given=True):
    out = {
        'chart': base,
        'chart-const-k': base.replace(kappa='constant'),
        'chart-no-excess': base.replace(excess='none'),
        'chart-zero-init': base.replace(init='one'),
        'chart-reversible': base.replace(drive='off'),
        'oracle': base.replace(mu0='given', excess=oracle_excess, kappa='true', train=False),
    }
    if given:
        out['chart-given-mu0'] = base.replace(mu0='given')
    return out


# the method and its ablations, per task (porous_media covers both variants, with and without crowding).  Porous media has one
# species and transport only, so mu0 has no effect there: no given-mu0 ablation.  The rtol is calibrated per benchmark.
CONFIGS = {
    'porous_media': {**_ablations(ChartConfig(task='porous_media', rtol=0.3), given=False),
                     'oracle-no-crowding': ChartConfig(task='porous_media', rtol=0.3, mu0='given', excess='none', kappa='true_material', train=False)},
    'brusselator': _ablations(ChartConfig(task='brusselator', rtol=5e-5)),
    'schlogl': _ablations(ChartConfig(task='schlogl')),
    'schlogl_nonideal': _ablations(ChartConfig(task='schlogl_nonideal'), oracle_excess='true'),
    'schlogl_nonideal_strong': _ablations(ChartConfig(task='schlogl_nonideal_strong'), oracle_excess='true'),
    'phosphorylation_cycle': _ablations(ChartConfig(task='phosphorylation_cycle')),
    'phosphorylation_heterogeneous': _ablations(ChartConfig(task='phosphorylation_heterogeneous')),
    'mapk_cascade': _ablations(ChartConfig(task='mapk_cascade')),
}

# what each ablation removes (tables)
ABLATIONS = {
    'chart-const-k': 'g_theta: one constant omega per hyperedge class',
    'chart-given-mu0': 'nothing: it adds the standard potentials from thermochemical data (oracle-type information)',
    'chart-zero-init': 'the closed-form NNLS start: omega starts at 1',
    'chart-reversible': 'the drive: A = 0, so the model is reversible',
    'chart-no-excess': 'the excess free energy G_theta',
    'flux_gnn': 'the chart: a graph network gives the one-way rates, with no chemical potential, force or symmetric mobility',
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
