"""Porous media as a Task of neuralchart.run: data, reference model, baselines, readouts.  No model code for the chart."""
import json
import os

import numpy as np
import torch

from neuralchart import NeuralChart, fast
from neuralchart.chart import MAX_SUB_EVAL, THETA
from neuralchart.diagnostics import relerr_traj
from neuralchart.fit import rollout, train
from ..common import check_gpu
from .data import DT, Disc, RHO_S, T_TRAIN, add_norm, data_dir_of, generate, load
from .evaluate import set_metrics, true_edge_input
from .fitters import fit_bsms, fit_generic_fno, fit_helmfluid
from .models import NotOnThisMesh, accelerate, build, load_weights, oracle, rebind
from .test_sets import TEST_SETS


def _as(sd, dtype):
    return dict(sd, c=sd['c'].to(dtype), drive={k: v.to(dtype) for k, v in sd['drive'].items()})


class Task:
    not_trained = ('persistence',)

    def __init__(s, name, data=None):
        s.name, s.data = name, data
        s.dt, s.t_steps, s.n_descriptors = DT, T_TRAIN, 12
        s._meta, s._discs = None, {}

    def _load(s, name, device):
        d = load(s.name, device, sets=[name], data_dir=s.data, meta=s.meta, discs=s._discs.setdefault(str(device), {}))
        if name in d:
            d[name].update(name=name, mesh=d['discs'][d[name]['disc']])
        return d

    @property
    def meta(s):
        if s._meta is None:
            s._meta = torch.load(os.path.join(data_dir_of(s.name, s.data), 'meta.pt'))
        return s._meta

    def _norm(s, device):
        """the z-scoring of the black-box inputs, from the training set"""
        if 'norm' not in s.meta:
            add_norm(s.meta, s._load('train', device))
        return s.meta

    def sets(s):
        return [k for k in TEST_SETS if os.path.exists(os.path.join(data_dir_of(s.name, s.data), f'{k}.pt'))]

    def load(s, name, device='cpu'):
        """a set in float64 (the Neural Chart integrates in float32 from a context computed in float64, the baselines take float32)"""
        sd = _as(s._load(name, device)[name], torch.float64)
        sd['drive']['force'] = sd['drive']['amp'][:, None] * sd['drive']['line']
        return sd

    def generate(s, sets=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        check_gpu(device)
        generate(s.name, device, s.data, sets)

    def hypergraph(s, sd=None):
        if sd is not None:
            return sd['mesh'].hg()
        return next((d['g16'] for d in s._discs.values() if 'g16' in d), None).hg() if any('g16' in d for d in s._discs.values()) else s._mesh('cpu').hg()

    def _mesh(s, device):
        """the training grid"""
        d = s._discs.setdefault(str(device), {})
        if 'g16' not in d:
            d['g16'] = Disc('g16', s.meta, device)
        return d['g16']

    def chemostats(s):
        return None

    def known_mu0(s):
        raise ValueError('porous media has no given standard potentials: there is no chart-given-mu0 run')

    def err_channels(s, sd):
        return slice(None)

    # ---------------------------------------------------------------- other models
    def baseline(s, name, device):
        if name.startswith('oracle'):
            raise ValueError('the oracle needs the true mobility and is for --cmd calibrate and the tests')
        return accelerate(build(name, s._norm(device), s._mesh(device)).to(device), name)

    def load_baseline(s, name, weights, device):
        model = s.baseline(name, device)
        return load_weights(model, name, weights, device) if weights else model

    def fit_baseline(s, name, model, tr, iters, seed, device, log, ckpt=None, resume=False, lr=1e-3):
        """the baselines train with the one trainer, BSMS-GNN, GENERIC-FNO and HelmFluid with the protocols of their papers"""
        if name in s.not_trained:
            return dict(train_time_s=0.0)
        tr = _as(tr, torch.float32)
        if name == 'bsms':
            return fit_bsms(model, tr, iters, seed, log)
        if name == 'generic_fno':
            return fit_generic_fno(model, tr, iters, seed, device, log)
        if name == 'helmfluid':
            return fit_helmfluid(model, tr, iters, seed, device, log)
        return train(model, tr, iters, seed, DT, batch=8, lr=lr, kmin=4, kmax=20, log=log, ckpt=ckpt, resume=resume)

    def rebind(s, name, model, sd, device):
        """the model on the discretisation of the set; the baselines take the set in float32"""
        m_ = rebind(name, model, s._norm(device), sd['mesh'], device)
        if m_ is None:
            return NotOnThisMesh()
        fast.copy_config(model, m_)
        sd.update(_as(sd, torch.float32))
        return m_

    # ---------------------------------------------------------------- readouts
    def metrics(s, name, model, sd, R, Yc, m):
        if isinstance(model, NotOnThisMesh):
            m['not_applicable'] = True
            return
        set_metrics(name, model, sd, R, Yc, m, sd['mesh'], bool(s.meta.get('crowd')))

    def extras(s, name, model):
        if not isinstance(model, NeuralChart):
            return {}
        return dict(drive_scale=float(model.drive.s_A.detach()), log_omega=float(model.kinetics.core.detach()))

    # ---------------------------------------------------------------- tables, animations, figures, calibration
    def tables(s, runs=None):
        from .tables import write_tables
        write_tables(s.name, runs_dir=runs)

    def gifs(s, device='cuda', weights=None, track=False):
        if track:
            from .gallery import make_gallery
            make_gallery(device, s.data, weights, variants=(s.name,))
        else:
            from .gifs import make_gifs
            make_gifs(device, s.data, weights, variants=(s.name,))

    def figures(s, device='cuda', weights=None):
        from .gallery import medium_and_law
        print(*medium_and_law(data_dir=s.data))

    @torch.no_grad()
    def calibrate(s, device='cuda', out='runs', rtols=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)):
        """the 10T error of the oracle (true mobility in the class) against the reference, per rtol and test set"""
        check_gpu(device)
        step = dict(dt_max=DT / 8, theta0=THETA, theta1=THETA, max_sub=MAX_SUB_EVAL, t_tol=1e-9)
        res = {}
        for r in rtols:
            res[r] = {}
            for name in s.sets():
                sd = s.load(name, device)
                disc = sd['mesh']
                sd['drive']['mtrue'] = true_edge_input(sd, disc)
                m = oracle(disc.hg(), RHO_S if s.meta.get('crowd') else None).to(device)
                fast.accelerate(m, control='sample_err', rtol=r, precision='fp32', step=step)
                R, ns, _ = rollout(m, sd['drive'], sd['c'][:, 0], sd['c'].shape[1] - 1, DT)
                w = disc.g['dens'].double().cpu()[None, None]
                e = relerr_traj(R.double().cpu() * w, sd['c'].double().cpu() * w).numpy()
                res[r][name] = dict(err_10T=float(np.median(e[:, -1])), err_max=float(e.max()), substeps=float(ns))
                print(f'rtol {r:g} {name}: 10T median {res[r][name]["err_10T"]:.3e} largest {res[r][name]["err_max"]:.3e} substeps/DT {ns:.1f}', flush=True)
        os.makedirs(os.path.join(out, s.name), exist_ok=True)
        json.dump({str(k): v for k, v in res.items()}, open(os.path.join(out, s.name, 'rtol_calibration.json'), 'w'), indent=1)
        return res
