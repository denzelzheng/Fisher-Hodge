"""The composition benchmarks as a Task of neuralchart.run.  No model code for the chart."""
import os

import numpy as np
import torch

from neuralchart import NeuralChart, Trajectories, compose, fast
from neuralchart.chart import RTOL
from neuralchart.compose import ComposedChart, compose_graphs, new_cycles
from neuralchart.diagnostics import LDB_RULE, ldb_violation
from neuralchart.fit import train
from ..reaction_networks import config as RC
from ..reaction_networks.data import load_set
from ..reaction_networks.models import accelerate, edge_potential, next_dtype, set_max_sub
from . import truth as T
from .data import DT, TINY, generate

KIND = dict(compose_cycle_cycle='none', compose_thermo_cross='thermo', compose_kinetic_cross='kinetic')
REFINE_STEPS = 20


class Holder(torch.nn.Module):
    """the composite of a run; the modules are fitted in fit_baseline"""

    def __init__(s):
        super().__init__()
        s.chart = None


def featurizer(hg, norm, transport):
    """the inputs of the Flux-GNN: hyperedge type, z-scored drive, and on the grid the potential"""
    ntyp = int(hg.meta['typ'].max()) + 1

    def feat(F):
        z = ((F['dG'].reshape(-1, 1) - norm['dG_mean']) / norm['dG_std'])
        B, dev, dt = z.shape[0], z.device, z.dtype
        t = hg.on(dev, dt)
        cols = [torch.nn.functional.one_hot(t['typ'], ntyp).to(dt).expand(B, hg.H, ntyp), z[:, None, :].expand(B, hg.H, 1)]
        node = dict(z0=z.expand(B, hg.V))
        ori = torch.zeros(B, hg.H, dtype=dt, device=dev)
        if transport:
            dU, Um = edge_potential(hg, F)
            cols += [dU.abs()[..., None], Um[..., None]]
            node['U'] = F['U']
            ori = dU
        return dict(U=torch.zeros(B, hg.V, dtype=dt, device=dev), node=node, edge_sym=torch.cat(cols, -1), edge_ori=ori, force=None, raw=F, z=z)
    feat.n_edge_sym = ntyp + 1 + (2 if transport else 0)
    return feat


class Task:
    not_trained = ()
    n_descriptors = 0

    def __init__(s, name, data=None):
        s.name = name
        RC.configure('phosphorylation_cycle', data)         # the test sets of compose_reaction_transport, and the Flux-GNN inputs
        s.dt, s.t_steps = DT, 20
        s.transport = name == 'compose_reaction_transport'
        s.kind = KIND.get(name)
        s.n_train = 8 if s.transport else 1
        s.root = os.path.join(os.path.dirname(RC.DATA), 'composition')
        RC.FAST = dict(control='sample_err', rtol=RTOL, precision='fp32', graphs=True)

    # ---------------------------------------------------------------- data
    def generate(s, sets=None):
        generate(s.root)

    def sets(s):
        return RC.sets() if s.transport else ['train_conditions', 'strong_drive', 'high_concentration']

    def load(s, name, device='cpu'):
        if s.transport:
            return load_set(name, device)
        d = torch.load(os.path.join(s.root, f'cascade_{s.kind}_{name}.pt'))
        return dict(c=d['c'].to(device), drive={k: v.to(device) for k, v in d['drive'].items()}, n=1, name=name)

    def _module_set(s, name, device):
        d = torch.load(os.path.join(s.root, name + '.pt'))
        return d['c'].to(device), {k: v.to(device) for k, v in d['drive'].items()}

    def hypergraph(s, sd=None):
        if s.transport:
            return compose_graphs(T.reaction_hg(), T.transport_hg(sd['n'] if sd is not None else 8))
        return compose_graphs(T.reaction_hg(), T.second_hg(), {'K': 'Xp'})

    def chemostats(s):
        return list(T.CHEM_MU0)

    def known_mu0(s):
        a = T.CHEM_MU0[0]
        return [-a, a, 0.0] if s.transport else [-a, a, 0.0, -a, a]

    def err_channels(s, sd):
        return slice(None)

    def truth(s, sd):
        hg = s.hypergraph(sd)
        if s.transport:
            return T.Truth(hg, [T.K1, T.K2, T.D, T.D], [T.K1, T.K2R, T.D, T.D], moving=('X', 'Xp'))
        return T.Truth(hg, [T.K1, T.K2, T.K1B, T.K2B], [T.K1, T.K2R, T.K1B, T.K2RB], chi=T.CHI if s.kind == 'thermo' else 0.0, inhibit=s.kind == 'kinetic')

    # ---------------------------------------------------------------- the modules, one fit each
    def _module(s, name, hg, chem, setname, device, iters, seed, out, log):
        # the appendix tasks share the modules of compose_cycle_cycle
        task = s.name if s.transport else 'compose_cycle_cycle'
        p = os.path.join(out, task, name + '.pt')
        if os.path.exists(p):
            return NeuralChart.load(p, hg, map_location=device).to(device)
        c, drive = s._module_set(setname, device)
        m = NeuralChart(hg, chemostats=chem).to(device)
        m.fit(Trajectories(c, drive, dt=DT), iters=iters, seed=seed, log=log, checkpoint=p + '.ckpt')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        m.save(p)
        return m

    def _modules(s, device, iters, seed, out, log):
        a = s._module('module_reaction', T.reaction_hg(), s.chemostats(), 'module_reaction', device, iters, seed, out, log)
        if s.transport:
            return a, s._module('module_transport', T.transport_hg(8), None, 'module_transport', device, iters, seed, out, log), {}
        return a, s._module('module_second', T.second_hg(), s.chemostats(), 'module_second', device, iters, seed, out, log), {'K': 'Xp'}

    # ---------------------------------------------------------------- models
    def _norm(s):
        tr = torch.load(os.path.join(s.root, 'module_reaction.pt'))
        return dict(dG_mean=float(tr['drive']['dG'].mean()), dG_std=float(tr['drive']['dG'].std()), U_std=0.7, c_scale=float(tr['c'].mean()))

    def _flux(s, sd=None):
        from neuralchart.baselines import FluxGNN
        hg = s.hypergraph(sd)
        f = featurizer(hg, s._norm(), s.transport)
        V = hg.V
        fx = f(dict(dG=torch.zeros(1, dtype=torch.float64), U=torch.zeros(1, V, dtype=torch.float64)))
        return FluxGNN(hg, 1 + len(fx['node']) + hg.n_species, fx['edge_sym'].shape[-1] + 1, f, hidden=64, layers=3, rho_scale=s._norm()['c_scale'], step=dict(RC.STEP))

    def baseline(s, name, device):
        if name.startswith('composed'):
            return Holder().to(device)
        assert name in ('flux_gnn', 'flux_gnn-transfer'), name
        m = s._flux().to(device).float()
        set_max_sub(m, RC.MAXSUB_TRAIN)
        accelerate(m, 'flux_gnn')
        return m

    def load_baseline(s, name, weights, device):
        if name.startswith('composed'):
            h = Holder().to(device)
            h.chart = ComposedChart.load(weights, map_location=device).to(device)
            return h
        m = s.baseline(name, device)
        m.load_state_dict(torch.load(weights, map_location=device))
        return m

    def _transfer_data(s, device):
        """the data of both modules in the composite input format"""
        ca, da = s._module_set('module_reaction', device)
        if s.transport:
            cb, db = s._module_set('module_transport', device)
            V = cb.shape[-1] // 2
            ca = ca.repeat_interleave(V, -1)
            cb = torch.cat([cb, torch.full_like(cb[..., :V], TINY)], -1)
            ua, ub = torch.zeros(ca.shape[0], V, dtype=ca.dtype, device=device), db['U']
            dGb = torch.zeros(cb.shape[0], dtype=ca.dtype, device=device)
        else:
            cb, db = s._module_set('module_second', device)
            t = lambda x: torch.full_like(x, TINY)
            ca = torch.cat([ca, t(ca[..., :2])], -1)                                    # X Xp E, Y Yp at TINY
            cb = torch.stack([t(cb[..., 0]), cb[..., 2], t(cb[..., 0]), cb[..., 0], cb[..., 1]], -1)   # Y Yp K -> X Xp=K E Y Yp
            ua, ub, dGb = da['U'], db['U'], db['dG']
        dG = torch.cat([da['dG'], dGb])
        return dict(c=torch.cat([ca, cb]), drive=dict(U=torch.cat([ua, ub]), dG=dG, chem_logc=torch.stack([0.5 * dG, -0.5 * dG], 1)))

    def fit_baseline(s, name, model, tr, iters, seed, device, log, ckpt=None, resume=False, lr=1e-3):
        out = os.path.dirname(os.path.dirname(ckpt))
        if name.startswith('composed'):
            a, b, glue = s._modules(device, iters, seed, out, log)
            model.modules_ab = (a, b, glue)
            model.chart = compose(a, b, glue=glue)
            log('composition', {k: v for k, v in model.chart.composition.items() if k not in ('new_cycles', 'spurious_affinity')})
            info = {}
            if '-refine-' in name:
                n = int(name.split('-')[-1])
                d = Trajectories(tr['c'][:n, :REFINE_STEPS + 1], {k: v[:n] for k, v in tr['drive'].items()}, dt=DT)
                info['refine'] = model.chart.refine(d, log=log)
            model.chart.save(os.path.join(out, s.name, name + '.pt'))
            return info
        if name == 'flux_gnn-transfer':
            tr = s._transfer_data(device)
        tr = dict(tr, c=tr['c'].float(), drive={k: v.float() for k, v in tr['drive'].items()})
        info = train(model, tr, iters, seed, DT, batch=RC.BATCH, lr=lr, kmin=RC.KMIN, kmax=RC.KMAX, log=log, ckpt=ckpt, resume=resume)
        set_max_sub(model, RC.MAXSUB_EVAL)
        accelerate(model, 'flux_gnn')
        return info

    def rebind(s, name, model, sd, device):
        if name.startswith('composed'):
            n = sd['n']
            if not s.transport or n == s.n_train:
                model.chart.accelerate()
                return model.chart
            a, b, glue = model.modules_ab
            c = compose(a, b.on(T.transport_hg(n)), glue=glue)
            c.accelerate()
            return c
        if sd['n'] != s.n_train:
            new = s._flux(sd).to(device).float()
            gb = lambda k: k.split('.')[-1] in ('p_h', 'p_m', 'p_v', 'm_h', 'm_m', 'm_v')
            new.load_state_dict({k: v for k, v in model.state_dict().items() if not gb(k)}, strict=False)
            fast.copy_config(model, new)
            model = new.eval()
        set_max_sub(model, RC.MAXSUB_EVAL)
        sd['c'] = sd['c'].float(); sd['drive'] = {k: v.float() for k, v in sd['drive'].items()}
        return model

    # ---------------------------------------------------------------- readouts
    @torch.no_grad()
    def metrics(s, name, model, sd, R, Yc, m):
        tr = s.truth(sd)
        hg = tr.hg
        dev = sd['c'].device
        lab = {k: v.double().cpu() for k, v in sd['drive'].items()}
        dt = next_dtype(model)
        nv = nval = nsn = ns_ = 0
        for t in range(0, R.shape[1], 10):
            c = R[:, t]
            a, b = model.rates(c.to(dev).to(dt).clamp_min(1e-30), sd['drive'])
            at, bt = tr.rates(lab, c.clamp_min(1e-300))
            J, Sab, f = (a - b).double().cpu(), (a + b).double().cpu(), torch.log(at) - torch.log(bt)
            ok = torch.isfinite(f) & torch.isfinite(J)
            val = ok & (J.abs() > 1e-9 * J.abs().nan_to_num(0).amax(-1, keepdim=True).clamp_min(1e-30))
            nv += int((ldb_violation(J, f, gross=Sab) & val).sum()); nval += int(val.sum())
            sig = torch.where(ok, J * f, torch.zeros_like(J)).sum(-1)
            good = torch.isfinite(sig)
            nsn += int((sig[good] < 0).sum()); ns_ += int(good.sum())
        m['phys'] = dict(ldb_violation_frac=nv / max(nval, 1), sigma_neg_frac=nsn / max(ns_, 1), ldb_rule=LDB_RULE)
        sgm, sgt = [], []
        for t in range(0, Yc.shape[1], 5):
            c = Yc[:, t]
            a, b = model.rates(c.to(dev).to(dt), sd['drive'])
            a, b = a.double().cpu(), b.double().cpu()
            sgm.append(((a - b) * (torch.log(a) - torch.log(b))).sum(-1))
            at, bt = tr.rates(lab, c)
            sgt.append(((at - bt) * (torch.log(at) - torch.log(bt))).sum(-1))
        sgm, sgt = torch.stack(sgm, 1), torch.stack(sgt, 1)
        se = (torch.linalg.vector_norm(sgm - sgt, dim=1) / torch.linalg.vector_norm(sgt, dim=1)).numpy()
        m['sigma'] = dict(relerr=float(np.nanmedian(se)), ratio=float(np.median((sgm.mean(1) / sgt.mean(1)).numpy())))
        a, b = model.rates(Yc[:, 0].to(dev).to(dt), sd['drive'])
        f = (torch.log(a) - torch.log(b)).double().cpu()
        at, bt = tr.rates(lab, Yc[:, 0])
        ft = torch.log(at) - torch.log(bt)
        Z = torch.as_tensor(hg.cycle_basis())
        m['cycle_affinity'] = dict(model_mean=float((f @ Z).mean()), true_mean=float((ft @ Z).mean()), mae=float((f @ Z - ft @ Z).abs().mean()))
        mod = (T.reaction_hg(), T.transport_hg(sd['n'])) if s.transport else (T.reaction_hg(), T.second_hg())
        new = torch.as_tensor(new_cycles(mod[0], mod[1], hg))
        m['new_cycles'] = dict(n=int(new.shape[1]), mae=float((f @ new - ft @ new).abs().mean()) if new.shape[1] else None)

    def extras(s, name, model):
        if isinstance(model, NeuralChart):
            return dict(log_k=model.kinetics.core.detach().cpu().tolist(), mu0=model.free_energy.mu0.detach().cpu().tolist())
        if isinstance(model, Holder):
            c = model.chart.composition
            sp = np.abs(c['spurious_affinity'])
            return dict(shared_species=c['shared_species'], n_new_cycles=c['n_new_cycles'], spurious_affinity_max=float(sp.max()) if len(sp) else 0.0,
                        force_change=c['force_change'], self_energy_gap=c['self_energy_gap'])
        return {}
