"""The Neural Chart: one model and one fit for every task.

    from neuralchart import Hypergraph, Conditions, Trajectories, NeuralChart
    hg = Hypergraph.from_strings(['A + X <-> 2 X', 'X <-> B'], chemostats=['A', 'B'], n_sites=1)
    chart = NeuralChart(hg, chemostats=[0.0, -1.0])
    chart.fit(Trajectories(c, Conditions(chem_logc=logc), times=t))
    chart.rollout(c0, Conditions(chem_logc=logc_new), nsteps=200)

    a_h, b_h = k_h eps_h exp(+- f_h / 2)       eps_h = exp(<nu+ + nu-, y> / 2), y = log c + dG(c)
    f   the force, <nu+ - nu-, y + (U + mu0_theta) / T> + A; the conditions enter only through U / T, the drive A and the chemostat
        log-activities
    dG(c) = grad G_theta(c / V) - grad G_theta(0)    the excess potential of the excess free energy G_theta
    log k_h = log omega_c(h) + g_theta(class one-hot, xi_h, log cbar_h)    (times T and the mesh weight)

Fit: at initialisation g = 0 and G = 0, so the rates are linear in omega and mu0: closed-form NNLS, then rollouts on every parameter.
One rule for the numerics on every task, nothing per task: the sampling step comes from the data, the substep bound is 1 / 8 of it,
rtol RTOL, the positivity fraction THETA, the rollouts of K in [KMIN, KMAX] steps from anywhere in the data.

An ablation is one override of the constructor (VARIANTS).
"""
import copy
import os
import warnings

import numpy as np
import torch

from .data import Conditions, Trajectories
from .fit import chart_summary, fit_linear, rollout, train
from .graphs import Hypergraph
from .model import Context, Drive, FreeEnergy, MasterEquation, Mobility, featurizer

RTOL, THETA = 1e-4, 0.5
ITERS, LR, BATCH, KMIN, KMAX = 4000, 1e-3, 8, 4, 20
MAX_SUB_TRAIN, MAX_SUB_EVAL = 2000, 4000
BUDGET_H = 12.0

# every ablation is one override of NeuralChart(...); 'mu0' is filled with the known values of the task
VARIANTS = {
    'chart': {},
    'chart-const-k': dict(material=False),
    'chart-no-excess': dict(excess=False),
    'chart-zero-init': dict(closed_form=False),
    'chart-reversible': dict(reversible=True),
    'chart-learned-chem-mu0': dict(chemostats=None),
    'chart-given-mu0': dict(mu0='known'),
}
ABLATIONS = {
    'chart-const-k': 'g_theta: one constant omega per hyperedge class',
    'chart-given-mu0': 'nothing: it adds the standard potentials of the internal species (oracle-type information)',
    'chart-zero-init': 'the closed-form NNLS start: omega starts at 1',
    'chart-reversible': 'the drive: A = 0, so the model is reversible',
    'chart-no-excess': 'the excess free energy G_theta',
    'chart-learned-chem-mu0': 'the known standard free energies of the chemostats: they are learned',
    'flux_gnn': 'the chart: a graph network gives the one-way rates, with no chemical potential, force or symmetric mobility',
}


def _cond(c):
    return c if isinstance(c, Context) else Conditions(c)


class NeuralChart(MasterEquation):
    """chemostats   the known standard free energies of the chemostats (one per chemostat), or None to learn them
    descriptors  the number of local descriptors per hyperedge in Conditions['edge']
    excess, material, closed_form, reversible, mu0   the ablations: no G_theta, no g_theta, no NNLS start, A = 0, known internal mu0"""

    def __init__(s, hypergraph, chemostats=None, descriptors=0, excess=True, material=True, closed_form=True, reversible=False, mu0=None):
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            hg = hypergraph
            feat = featurizer(hg, n_desc=descriptors)
            nC = len(hg.meta.get('chemostats', []))
            known = None if chemostats is None or not nC else (np.eye(nC), np.asarray(chemostats, np.float64))
            kin = Mobility(feat.n_edge_sym, hg, material=material)
            fe = FreeEnergy(hg, True, excess=excess)
            dr = Drive(hg, False, nC > 0, known)
            super().__init__(hg, fe, kin, dr, feat, dict(dt_max=None, theta0=THETA, theta1=THETA, max_sub=MAX_SUB_EVAL, t_tol=1e-9))
            s.register_buffer('dt', torch.tensor(float('nan')))
        finally:
            torch.set_default_dtype(old)
        s.drive.s_A.requires_grad_(False)                # the external force is known
        s.args = dict(chemostats=chemostats, descriptors=descriptors, excess=excess, material=material, closed_form=closed_form,
                      reversible=reversible, mu0=mu0)
        s.closed_form = closed_form
        s.diagnostics = {}
        if reversible:
            s.knobs['drive_scale'] = 0.0
        if mu0 is not None:
            with torch.no_grad():
                fe.mu0.copy_(torch.as_tensor(np.asarray(mu0), dtype=fe.mu0.dtype))
            fe.mu0.requires_grad_(False)

    # ---------------------------------------------------------------- numerics: one rule
    def _integrator(s, cap):
        from . import fast
        s.step = dict(s.step, dt_max=float(s.dt) / 8, max_sub=cap)
        fast.restore(s)
        return fast.accelerate(s, control='sample_err', rtol=RTOL, precision='fp32')

    def accelerate(s):
        """the integrator for rollouts at the sampling step of the fit"""
        assert np.isfinite(float(s.dt)), 'fit the chart or load a fitted one first'
        return s._integrator(MAX_SUB_EVAL)

    # ---------------------------------------------------------------- fit
    def fit(s, trajectories, conditions=None, times=None, iters=ITERS, seed=0, log=print, checkpoint=None, resume=False, lr=LR):
        """fit on Trajectories (or on states c (B, T + 1, M), Conditions and the sampling times).  Returns a dict of diagnostics."""
        tr = trajectories if isinstance(trajectories, Trajectories) else Trajectories(trajectories, conditions, times)
        data = tr.as_dict()
        torch.manual_seed(seed)
        s.dt.fill_(tr.dt)
        s._warn_inputs(log)
        s.diagnostics['density_max'] = s.free_energy.density(data['c'].reshape(-1, s.hg.M)).reshape(-1, s.hg.n_species).amax(0).tolist()
        s._integrator(MAX_SUB_TRAIN)
        resume = bool(resume and checkpoint and os.path.exists(checkpoint))
        info = {}
        if s.closed_form and not resume:
            info['linear_fit'] = fit_linear(s, dict(train=data), tr.dt, log=log)
            s.diagnostics.update(s._design_report(info['linear_fit'], log))
        info.update(train(s, data, iters, seed, tr.dt, batch=BATCH, lr=lr, kmin=KMIN, kmax=KMAX, log=log, budget_s=BUDGET_H * 3600,
                          ckpt=checkpoint, resume=resume))
        s.eval()
        info['summary'] = chart_summary(s, data)
        s.accelerate()
        return info

    def refine(s, trajectories, conditions=None, times=None, iters=500, seed=0, log=print, lr=1e-4):
        """a short fine-tune of every parameter on rollouts, from the current values (no closed-form step)"""
        tr = trajectories if isinstance(trajectories, Trajectories) else Trajectories(trajectories, conditions, times)
        s.dt.fill_(tr.dt)
        s._integrator(MAX_SUB_TRAIN)
        info = train(s, tr.as_dict(), iters, seed, tr.dt, batch=BATCH, lr=lr, kmin=KMIN, kmax=KMAX, log=log, eta_min=lr / 10)
        s.eval()
        s.accelerate()
        return info

    def _warn_inputs(s, log):
        if s.drive.chem_mu0 is not None:
            warnings.warn('the standard free energies of the chemostats are not given: they are fitted, and density data do not fix them, '
                          'so cycle affinities and entropy production are unreliable; pass chemostats=[...]')

    def _design_report(s, lf, log):
        rep = dict(design_rank=lf['rank'], design_parameters=lf['n_params'], design_classes=lf['n_classes'], factors_at_zero=lf['n_floored'], rates_at_floor=lf['n_at_floor'])
        nl = sum(p.numel() for p in s.linear_parameters())
        log(f"fit design: rank {lf['rank']} of {nl} linear parameters over {lf['n_classes']} classes; {lf['n_floored']} factors at zero, {lf['n_at_floor']} log rates at the floor")
        if lf['n_at_floor']:
            warnings.warn(f"{lf['n_at_floor']} hyperedge classes start at the floor of the closed-form step: the zero-initialised model does not explain them")
        return rep

    # ---------------------------------------------------------------- rollouts and readouts
    @torch.no_grad()
    def rollout(s, c0, conditions, nsteps):
        """the states (B, nsteps + 1, M) from c0 (B, M), sampled at the sampling step of the fit"""
        if not hasattr(s, '_fast'):
            s.accelerate()
        return rollout(s, _cond(conditions), c0, nsteps, float(s.dt))[0]

    @torch.no_grad()
    def excess_potential(s, c):
        """grad G(c / V) - grad G(0) at the states c (B, M)"""
        return s.free_energy.correction(c, None)

    @torch.no_grad()
    def mobility(s, c, conditions):
        """the symmetric rate constant k (B, H) at the states c"""
        return s.kinetics(s.context(_cond(conditions))['feats'], c)

    @torch.no_grad()
    def one_way_rates(s, c, conditions):
        return s.rates(c, _cond(conditions))

    @torch.no_grad()
    def flux(s, c, conditions):
        a, b = s.rates(c, _cond(conditions))
        return a - b

    @torch.no_grad()
    def force(s, c, conditions):
        a, b = s.rates(c, _cond(conditions))
        return torch.log(a) - torch.log(b)

    @torch.no_grad()
    def entropy_production(s, c, conditions):
        """sigma = sum_h J_h f_h (B,); per hyperedge with s.flux * s.force"""
        a, b = s.rates(c, _cond(conditions))
        return ((a - b) * (torch.log(a) - torch.log(b))).sum(-1)

    @torch.no_grad()
    def cycle_affinity(s, conditions, c=None):
        """<z, f> for the cycle basis z of ker S (B, n_cycles), independent of the state"""
        cond = _cond(conditions)
        ctx = s.context(cond)
        if c is None:
            c = torch.ones(ctx['A'].shape[0], s.hg.M, dtype=ctx['A'].dtype, device=ctx['A'].device)
        f = s.force(c, cond)
        Z = torch.as_tensor(s.hg.cycle_basis(), dtype=f.dtype, device=f.device)
        return f @ Z

    # ---------------------------------------------------------------- other hypergraphs, files
    def on(s, hypergraph):
        """the same fitted chart on another hypergraph with the same classes, species and descriptors (no refit)"""
        new = NeuralChart(hypergraph, **s.args).to(s.dt.device)
        new.load_state_dict(s.state_dict())
        new.train(s.training)
        return new

    def save(s, path):
        hg = s.hg
        torch.save(dict(args=s.args, state=s.state_dict(), hypergraph=hg.to_dict(), diagnostics=s.diagnostics), path)

    @classmethod
    def load(cls, path, hypergraph=None, map_location='cpu'):
        d = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(hypergraph or Hypergraph.from_dict(d['hypergraph']), **d['args'])
        m.load_state_dict(d['state'])
        m.diagnostics = d.get('diagnostics', {})
        return m.eval()
