"""The reaction-network benchmarks as a Task of neuralchart.run: data, reference model, baselines, readouts.  No model code for the chart."""
import json
import os

import numpy as np
import torch

from neuralchart import NeuralChart, fast
from neuralchart.chart import RTOL
from neuralchart.diagnostics import LDB_RULE, ldb_violation, relerr_traj
from neuralchart.fit import rollout, train
from ..common import check_gpu
from . import config as C
from . import truth as TR
from .data import generate, load_set, train_norm
from .evaluate import err_channels, excess_law, k_law, lngamma_metrics, maxwell_equal_area, front_speed, zero_cross
from .models import FLOAT_MODELS, UDE, accelerate, build_baseline, known_chem_mu0, known_mu0, next_dtype, oracle, rebind, set_max_sub, true_log_k
from .fitters import run_bsms, run_generic_fno


class Task:
    not_trained = ('persistence', 'oracle')

    def __init__(s, name, data=None):
        s.name = name
        C.configure(name, data)
        s.dt, s.t_steps, s.n_train = C.DT, C.T_STEPS, C.N_TRAIN
        s.n_descriptors = 3 if C.hetero() else 0
        C.FAST = dict(control='sample_err', rtol=RTOL, precision='fp32', graphs=True)

    def sets(s):
        return C.sets()

    def load(s, name, device='cpu'):
        return load_set(name, device)

    def generate(s, sets=None):
        generate(sets)

    def hypergraph(s, sd=None):
        return TR.hg_of(sd['n'] if sd is not None else C.N_TRAIN)

    def chemostats(s):
        return known_chem_mu0()

    def known_mu0(s):
        return known_mu0()

    def err_channels(s, sd):
        return err_channels(sd['n'])

    # ---------------------------------------------------------------- other models
    def baseline(s, name, device):
        norm = train_norm()
        model = oracle(C.N_TRAIN) if name == 'oracle' else build_baseline(name, C.N_TRAIN, norm)
        model = model.to(device) if hasattr(model, 'to') else model
        if name in FLOAT_MODELS:
            model = model.float()
        set_max_sub(model, C.MAXSUB_TRAIN)
        accelerate(model, name)
        return model

    def load_baseline(s, name, weights, device):
        model = s.baseline(name, device)
        if weights:
            model.load_state_dict(torch.load(weights, map_location=device))
        return model

    def fit_baseline(s, name, model, tr, iters, seed, device, log, ckpt=None, resume=False, lr=1e-3):
        """the baselines train with the one trainer, the one-step maps with the protocol of their papers"""
        if name in ('persistence', 'oracle'):
            return dict(train_time_s=0.0)
        if name in FLOAT_MODELS:
            tr = dict(tr, c=tr['c'].float(), drive={k: v.float() for k, v in tr['drive'].items()})
        if name == 'bsms':
            return run_bsms(model, tr, iters, seed, log)
        if name == 'generic_fno':
            return run_generic_fno(model, tr, iters, seed, device, log, train_norm())
        info = {}
        if hasattr(model, 'linear_parameters') and model.linear_parameters() and getattr(model, 'kdu', None) is None:
            from neuralchart.fit import fit_linear
            if not (resume and os.path.exists(ckpt)):
                info['linear_fit'] = fit_linear(model, dict(train=tr), C.DT, log=log)
        info.update(train(model, tr, iters, seed, C.DT, batch=C.BATCH, lr=lr, kmin=C.KMIN, kmax=C.KMAX, log=log, ckpt=ckpt, resume=resume))
        set_max_sub(model, C.MAXSUB_EVAL)
        accelerate(model, name)
        return info

    def rebind(s, name, model, sd, device):
        n = sd['n']
        m_ = rebind(name, model, n, train_norm(), device) if n != C.N_TRAIN else model
        if m_ is model and name == 'ude':
            accelerate(m_, name)
        else:
            fast.copy_config(model, m_)
        set_max_sub(m_, C.MAXSUB_EVAL)
        if next_dtype(model) == torch.float32:
            sd['c'] = sd['c'].float(); sd['drive'] = {k: v.float() for k, v in sd['drive'].items()}
        return m_

    # ---------------------------------------------------------------- readouts
    @torch.no_grad()
    def metrics(s, name, model, sd, R, Yc, m):
        """the physics against the true force of the condition, the entropy production and cycle affinity from the model's own rates on the
        reference states, and the readouts of each task"""
        from .models import UDE
        B, n = R.shape[0], sd['n']
        hg = TR.hg_of(n)
        V = hg.V
        dev = sd['c'].device
        lab = {k: v.double().cpu() for k, v in sd['drive'].items()}
        if C.SYSTEM == 'mapk_cascade':
            L = hg.on('cpu', torch.float64)['L']
            q = torch.nan_to_num(R, nan=float('inf')) @ L
            m['moiety_drift'] = dict(all=float((torch.linalg.vector_norm(q - q[:, :1], dim=-1) / torch.linalg.vector_norm(q[:, :1], dim=-1)).max()))
            m['kpp_end_err'] = float(np.nanmedian((TR.kpp_fraction(R[:, -1]) - TR.kpp_fraction(Yc[:, -1])).abs().mean(-1).numpy()))
        if C.is_pc():
            E = R[..., 2 * V:]; Xt = R[..., :2 * V].sum(-1)
            m['moiety_drift'] = dict(E=float(((E - E[:, :1]).abs() / E[:, :1]).nan_to_num(float('inf')).max()),
                                     X_total=float(((Xt - Xt[:, :1]).abs() / Xt[:, :1]).nan_to_num(float('inf')).max()))
        if not hasattr(model, 'flux'):
            return
        nv = nval = nsn = ns_ = 0
        for t in range(0, R.shape[1], 10):
            c = R[:, t]
            ok_c = torch.isfinite(c).all(1) & (c > 0).all(1)
            a, b = model.rates(c.to(dev).to(next_dtype(model)).clamp_min(1e-30), sd['drive'])
            a, b = a.double().cpu(), b.double().cpu()
            J, Sab = a - b, a + b
            at, bt = TR.true_rates(hg, lab, c.clamp_min(1e-300))
            f = torch.log(at) - torch.log(bt)
            f = torch.where(ok_c[:, None], f, torch.full_like(f, float('nan')))
            ok = torch.isfinite(f) & torch.isfinite(J)
            val = ok & (J.abs() > 1e-9 * J.abs().nan_to_num(0).amax(-1, keepdim=True).clamp_min(1e-30))
            nv += int((ldb_violation(J, f, gross=None if isinstance(model, UDE) else Sab) & val).sum()); nval += int(val.sum())
            sig = torch.where(ok, J * f, torch.zeros_like(J)).sum(-1)
            good = ok_c & torch.isfinite(sig)
            nsn += int((sig[good] < 0).sum()); ns_ += int(good.sum())
        m['phys'] = dict(ldb_violation_frac=nv / max(nval, 1), sigma_neg_frac=nsn / max(ns_, 1), ldb_rule=LDB_RULE)
        if not isinstance(model, UDE):
            sgm, sgt = [], []
            for t in range(0, Yc.shape[1], 5):
                c = Yc[:, t]
                a, b = model.rates(c.to(dev).to(next_dtype(model)), sd['drive'])
                a, b = a.double().cpu(), b.double().cpu()
                sgm.append(((a - b) * (torch.log(a) - torch.log(b))).sum(-1))
                at, bt = TR.true_rates(hg, lab, c)
                sgt.append(((at - bt) * (torch.log(at) - torch.log(bt))).sum(-1))
            sgm, sgt = torch.stack(sgm, 1), torch.stack(sgt, 1)
            se = (torch.linalg.vector_norm(sgm - sgt, dim=1) / torch.linalg.vector_norm(sgt, dim=1)).numpy()
            m['sigma'] = dict(relerr=float(np.nanmedian(se)), ratio=float(np.median((sgm.mean(1) / sgt.mean(1)).numpy())))
            a, b = model.rates(Yc[:, 0].to(dev).to(next_dtype(model)), sd['drive'])
            ca = TR.model_cycle_affinity((torch.log(a) - torch.log(b)).double().cpu(), V).numpy()
            at, bt = TR.true_rates(hg, lab, Yc[:, 0])
            ct = TR.model_cycle_affinity((torch.log(at) - torch.log(bt)).double(), V).numpy()
            m['cycle_affinity'] = dict(model_mean=float(ca.mean()), true_mean=float(ct.mean()), mae=float(np.abs(ca - ct).mean()))
        if isinstance(model, NeuralChart):
            if C.is_pc() or C.SYSTEM == 'mapk_cascade':
                m['k_law'] = k_law(model, sd, hg)
            m['excess_law'] = excess_law(model, sd)
        if 'planar' in str(sd.get('name', '')) or (C.SYSTEM == 'schlogl' and R.shape[0] == 13):   # the front set
            bb = lab['logab'][:, 1].exp().numpy()
            vm, vt = front_speed(R.numpy(), bb, n), front_speed(Yc.numpy(), bb, n)
            ok = np.isfinite(vm) & (np.abs(vt) > 1e-3)
            m['front'] = dict(b=bb.tolist(), v_model=vm.tolist(), v_true=vt.tolist(), b_star_model=zero_cross(vm, bb), b_star_true=zero_cross(vt, bb),
                              speed_relerr_median=float(np.median(np.abs(vm[ok] - vt[ok]) / np.abs(vt[ok]))) if ok.any() else None)

    def extras(s, name, model):
        if not isinstance(model, NeuralChart):
            if C.SYSTEM == 'schlogl' and hasattr(model, 'flux'):
                return dict(maxwell=dict(equal_area_model=maxwell_equal_area(model), true=TR.maxwell_b(3.0), a=3.0))
            return {}
        norm = train_norm()
        x = dict(log_k=model.kinetics.core.detach().cpu().tolist(), mu0=model.free_energy.mu0.detach().cpu().tolist(),
                 chem_mu0=model.drive.chem_mu0_value().detach().cpu().tolist(), log_k_true=true_log_k())
        if C.nonideal():
            high = torch.load(os.path.join(C.DATA, f'{C.ood_set()}.pt'))
            x['lngamma'] = lngamma_metrics(model, norm, float(high['c'].max()))
        if C.SYSTEM == 'schlogl':
            x['maxwell'] = dict(equal_area_model=maxwell_equal_area(model), true=TR.maxwell_b(3.0), a=3.0)
        return x

    # ---------------------------------------------------------------- tables, animations, calibration
    def tables(s, runs=None):
        from .tables import make_tables
        make_tables(runs)

    def gifs(s, device='cuda', weights=None, track=False):
        from .gifs import make_gif, make_track_gif
        (make_track_gif if track else make_gif)(s.name, device, weights)

    @torch.no_grad()
    def calibrate(s, device='cuda', out='runs', rtols=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)):
        """the 10T error of the oracle (true constants in the class) against the reference, per rtol and test set"""
        check_gpu(device)
        res = {}
        for r in rtols:
            res[r] = {}
            for name in C.sets():
                sd = load_set(name, device)
                m = oracle(sd['n']).to(device)
                fast.accelerate(m, control='sample_err', rtol=r, precision='fp32', step=C.STEP)
                set_max_sub(m, C.MAXSUB_EVAL)
                R, ns, _ = rollout(m, sd['drive'], sd['c'][:, 0], sd['c'].shape[1] - 1, C.DT)
                ch = err_channels(sd['n'])
                e = relerr_traj(R.double().cpu()[..., ch], sd['c'].double().cpu()[..., ch]).numpy()
                res[r][name] = dict(err_10T=float(np.median(e[:, -1])), err_max=float(e.max()), substeps=float(ns))
                print(f'rtol {r:g} {name}: 10T median {res[r][name]["err_10T"]:.3e} largest {res[r][name]["err_max"]:.3e} substeps/DT {ns:.1f}', flush=True)
        os.makedirs(os.path.join(out, s.name), exist_ok=True)
        json.dump({str(k): v for k, v in res.items()}, open(os.path.join(out, s.name, 'rtol_calibration.json'), 'w'), indent=1)
        return res
