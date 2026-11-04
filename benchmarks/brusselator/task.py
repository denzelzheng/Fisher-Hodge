"""The Brusselator as a Task of neuralchart.run: data, reference model, baselines, readouts.  No model code for the chart."""
import json
import os

import numpy as np
import torch

from neuralchart import NeuralChart, fast
from neuralchart.diagnostics import relerr_traj
from neuralchart.fit import fit_linear, rollout, train
from ..common import check_gpu
from . import config as C
from .data import generate, hg_of, load_set, train_norm
from .evaluate import set_metrics
from .fitters import train_bsms, train_generic_fno
from .models import FLOAT32_MODELS, accelerate, build, known_chem_mu0, known_mu0, oracle, rebind, set_max_sub, true_log_k
from .test_sets import GEN, TEST_SETS


def _float(sd):
    return dict(sd, c=sd['c'].float(), drive={k: v.float() for k, v in sd['drive'].items()})


class Task:
    not_trained = ('persistence', 'oracle')

    def __init__(s, name, data=None):
        s.name = name
        s.data = data or C.DATA
        s.dt, s.t_steps, s.n_descriptors = C.DT, C.T_STEPS, 0

    @property
    def norm(s):
        if not hasattr(s, '_norm'):
            s._norm = train_norm(data=s.data)
        return s._norm

    def sets(s):
        return list(TEST_SETS)

    def load(s, name, device='cpu'):
        return load_set(name, device, data=s.data)

    def generate(s, sets=None):
        generate(sets or list(GEN), data=s.data)

    def hypergraph(s, sd=None):
        return hg_of(sd['n'] if sd is not None else C.N_TRAIN)

    def chemostats(s):
        return known_chem_mu0()

    def known_mu0(s):
        return known_mu0()

    def err_channels(s, sd):
        return slice(None)

    # ---------------------------------------------------------------- other models
    def baseline(s, name, device):
        model = (oracle(C.N_TRAIN) if name == 'oracle' else build(name, C.N_TRAIN, s.norm)).to(device)
        if name in ('flux_gnn', 'ude'):
            model = model.float()
        set_max_sub(model, C.MAXSUB_TRAIN)
        return accelerate(model, name)

    def load_baseline(s, name, weights, device):
        model = s.baseline(name, device)
        if weights:
            model.load_state_dict(torch.load(weights, map_location=device))
        return model

    def fit_baseline(s, name, model, tr, iters, seed, device, log, ckpt=None, resume=False, lr=1e-3):
        """the baselines train with the one trainer, the one-step maps BSMS-GNN and GENERIC-FNO with the protocol of their papers"""
        if name in s.not_trained:
            return dict(train_time_s=0.0)
        if name in FLOAT32_MODELS:
            tr = _float(tr)
        if name == 'bsms':
            return train_bsms(model, tr, iters, seed, log)
        if name == 'generic_fno':
            return train_generic_fno(model, tr, iters, seed, device, log, s.norm)
        info = {}
        if hasattr(model, 'linear_parameters') and model.linear_parameters() and not (resume and os.path.exists(ckpt)):
            info['linear_fit'] = fit_linear(model, dict(train=tr), C.DT, log=log)
        info.update(train(model, tr, iters, seed, C.DT, batch=C.BATCH, lr=lr, kmin=C.KMIN, kmax=C.KMAX, log=log, ckpt=ckpt, resume=resume))
        set_max_sub(model, C.MAXSUB_EVAL)
        accelerate(model, name)
        return info

    def rebind(s, name, model, sd, device):
        """the model on the grid of the set; a float32 model takes the set in float32"""
        m_ = rebind(name, model, sd['n'], s.norm, device)
        fast.copy_config(model, m_)
        set_max_sub(m_, C.MAXSUB_EVAL)
        if name in FLOAT32_MODELS:
            sd.update(_float(sd))
        return m_

    # ---------------------------------------------------------------- readouts
    def metrics(s, name, model, sd, R, Yc, m):
        """limit cycle, patterns, local detailed balance against the true force, entropy production and cycle affinities"""
        set_metrics(model, sd, R, Yc, m)

    def extras(s, name, model):
        if not isinstance(model, NeuralChart):
            return {}
        return dict(log_k=model.kinetics.core.detach().cpu().tolist(), mu0=model.free_energy.mu0.detach().cpu().tolist(),
                    chem_mu0=model.drive.chem_mu0_value().detach().cpu().tolist(), log_k_true=true_log_k())

    # ---------------------------------------------------------------- tables, animations, figures, calibration
    def tables(s, runs=None):
        from .tables import make_tables
        make_tables(runs=runs)

    def gifs(s, device='cuda', weights=None, track=False):
        from .gifs import make_gifs, make_track_gif
        (make_track_gif if track else make_gifs)(device, data=s.data, weights=weights)

    def figures(s, device='cuda', weights=None):
        from .figures import make_figures
        make_figures(device, data=s.data, weights=weights)

    @torch.no_grad()
    def calibrate(s, device='cuda', out='runs', rtols=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)):
        """the 10T error of the oracle (true constants in the class) against the reference, per rtol and test set"""
        check_gpu(device)
        res = {}
        for r in rtols:
            res[r] = {}
            for name in s.sets():
                sd = s.load(name, device)
                m = oracle(sd['n']).to(device)
                fast.accelerate(m, control='sample_err', rtol=r, precision='fp32', step=C.STEP)
                set_max_sub(m, C.MAXSUB_EVAL)
                R, ns, _ = rollout(m, sd['drive'], sd['c'][:, 0], sd['c'].shape[1] - 1, C.DT)
                e = relerr_traj(R.double().cpu(), sd['c'].double().cpu()).numpy()
                res[r][name] = dict(err_10T=float(np.median(e[:, -1])), err_max=float(e.max()), substeps=float(ns))
                print(f'rtol {r:g} {name}: 10T median {res[r][name]["err_10T"]:.3e} largest {res[r][name]["err_max"]:.3e} substeps/DT {ns:.1f}', flush=True)
        os.makedirs(os.path.join(out, s.name), exist_ok=True)
        json.dump({str(k): v for k, v in res.items()}, open(os.path.join(out, s.name, 'rtol_calibration.json'), 'w'), indent=1)
        return res
