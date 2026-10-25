"""Ready-made problems: featurizers, model presets and data loaders for the benchmarks of the prototype scripts.

s2_tcme   Fokker-Planck with circulation (S2) on the 32 x 32 torus, the TC-ME model of fisher_hodge/models_tcme.py
          written as a HyperME (split kinetics, direct local free energy, curl drive).  Data runs/data/S2.pt.
c1        chemostat-driven phosphorylation cycle (C1) on the 8 x 8 torus, the HyperME of hyper/hyperme.py.
          Data hyper/runs/data/crn.pt.  float64.
The presets create their parameters in the same order and with the same initialisation as the prototypes, so a
given torch seed gives the same initial weights, and load_legacy() maps a prototype checkpoint onto the preset.
"""
import os
import contextlib
import torch
from .complex import Hypergraph, GridOps as G
from .model import HyperME, FreeEnergy, Kinetics, ProjectedDrive, CurlDrive
import torch.nn.functional as Fn

ROOT = os.environ.get('FH_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = 0.05
T_TRAIN = 20


@contextlib.contextmanager
def default_dtype(dt):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


# ------------------------------------------------------------------------------------------ S2 / TC-ME
def s2_featurizer(hg):
    """drive = dict(U (B,n,n), dpsi (B,2,n,n), psin (B,n,n)) -> features of fisher_hodge/models.static_edge_feats."""
    def feat(F):
        U, P, pn = F['U'], F['dpsi'], F['psin']
        B = U.shape[0]
        dU = G.d0(U)
        ish = torch.zeros_like(dU); ish[:, 0] = 1
        sym = torch.stack([ish, 1 - ish, G.mean_e(U), dU.abs(), G.mean_e(pn), P.abs()], -1).reshape(B, -1, 6)
        Pf = G.flat_e(P)
        return dict(U=U.reshape(B, -1), node=dict(U=U.reshape(B, -1), psin=pn.reshape(B, -1)), edge_sym=sym,
                    edge_ori=Pf, force=Pf, raw=F)
    feat.n_edge_sym = 6
    return feat


def s2_tcme(n=32, hidden=128, rho_scale=10.0, D=0.02):
    hg = Hypergraph.grid_transport(n)
    kin = Kinetics(6, hidden, scale=D * n * n, split=True)
    fe = FreeEnergy(hg, ('U',), hidden, mode='direct', alpha_init=1.0, rho_scale=rho_scale)
    dr = CurlDrive(hidden)
    return HyperME(hg, fe, kin, dr, s2_featurizer(hg),
                   step=dict(dt_max=DT / 8, theta0=0.9, theta1=1.0, max_sub=4000, t_tol=1e-9))


def load_legacy_tcme(model, sd):
    """fisher_hodge/models_tcme.TCME state dict -> s2_tcme model"""
    mp = {'k.': 'kinetics.mlp.', 'g.': 'free_energy.mlp.', 'psi.': 'drive.psi.'}
    out = {}
    for k, v in sd.items():
        for a, b in mp.items():
            if k.startswith(a):
                out[b + k[len(a):]] = v
                break
        else:
            out[{'alpha': 'free_energy.alpha', 'gam': 'drive.gam', 'v': 'drive.v', 'Wb': 'drive.Wb'}[k]] = v.reshape(1) if k == 'alpha' else v
    model.load_state_dict(out)
    return model


def load_s2(root=ROOT, device='cpu', name='S2'):
    data = torch.load(os.path.join(root, 'runs', 'data', f'{name}.pt'))
    out = {}
    for k, d in data.items():
        B, T = d['rho'].shape[:2]
        out[k] = dict(c=d['rho'].reshape(B, T, -1).to(device),
                      drive={f: d[f].to(device) for f in ('U', 'dpsi', 'psin')})
        if 'J_end' in d:
            out[k]['J_end'] = d['J_end']
    return out


# ------------------------------------------------------------------------------------------ C1 / HyperME
def c1_featurizer(hg):
    """drive = dict(U (B,V), dG (B,)) -> features of hyper/hyperme.edge_feats."""
    V, H = hg.V, hg.H
    et, eh = hg.meta['et'][2 * V:], hg.meta['eh'][2 * V:]

    def feat(F):
        U, dG = F['U'], F['dG']
        B = U.shape[0]
        t = hg.on(U.device, U.dtype)
        oh = Fn.one_hot(t['typ'], 4).to(U.dtype).expand(B, H, 4)
        Ut, Uh = U[:, et], U[:, eh]
        Um = torch.cat([U, U, 0.5 * (Ut + Uh)], 1)
        dU = torch.cat([torch.zeros_like(U), torch.zeros_like(U), Uh - Ut], 1)
        dg = (dG[:, None] - 2.0).expand(B, H)
        sym = torch.cat([oh, Um[..., None], dU.abs()[..., None], dg[..., None]], -1)
        return dict(U=U, node=dict(U=U, dGm2=(dG[:, None] - 2.0).expand(B, V)), edge_sym=sym, edge_ori=dU, force=None, raw=F)
    feat.n_edge_sym = 7
    return feat


def c1(n=8, hidden=64, use_A=True):
    hg = Hypergraph.crn_phosphorylation_grid(n)
    with default_dtype(torch.float64):
        kin = Kinetics(7, hidden, last='default', last_bias=0.5, eps=1e-6)
        dr = ProjectedDrive(hg, 7, hidden, mlp=True, gain_init=None)
        fe = FreeEnergy(hg, ('U', 'dGm2'), hidden, mode='grad', alpha_init=1.0)
        if not use_A:
            from .model import NoDrive
            dr = NoDrive()
        m = HyperME(hg, fe, kin, dr, c1_featurizer(hg),
                    step=dict(dt_max=DT / 4, theta0=0.5, theta1=0.5, max_sub=2000, t_tol=1e-12))
    return m.double()


def load_legacy_c1(model, sd):
    mp = {'kmlp.': 'kinetics.mlp.', 'amlp.': 'drive.mlp.', 'G.': 'free_energy.mlp.'}
    out = {}
    for k, v in sd.items():
        for a, b in mp.items():
            if k.startswith(a):
                out[b + k[len(a):]] = v
                break
        else:
            out['free_energy.alpha'] = v
    model.load_state_dict(out)
    return model


def load_c1(root=ROOT, device='cpu', tag=''):
    data = torch.load(os.path.join(root, 'hyper', 'runs', 'data', f'crn{tag}.pt'))
    out = {}
    for k, d in data.items():
        B, T = d['c'].shape[:2]
        out[k] = dict(c=d['c'].reshape(B, T, -1).double().to(device),
                      drive=dict(U=d['U'].double().to(device), dG=d['dG'].double().to(device)))
        for extra in ('sigma', 'J', 'k', 'A'):
            if extra in d:
                out[k][extra] = d[extra]
    return out


PRESETS = dict(s2_tcme=(s2_tcme, load_s2), c1=(c1, load_c1))
