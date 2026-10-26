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


def s2_wired_featurizer(hg):
    """S2 grid drive -> wired features: U (already in units of the training temperature), the drift line integral
    dpsi as the force line integral, T (default 1 = training conditions), edge orientation as the only kinetic feature."""
    def feat(F):
        U, P = F['U'], F['dpsi']
        B = U.shape[0]
        T = torch.as_tensor(F.get('T', 1.0), dtype=U.dtype, device=U.device)
        T = T.expand(B) if T.dim() == 0 else T.reshape(B)
        ish = torch.zeros_like(P); ish[:, 0] = 1
        sym = torch.stack([ish, 1 - ish], -1).reshape(B, -1, 2)
        return dict(U=U.reshape(B, -1), T=T, line=G.flat_e(P), chem_logc=None, chem_mu0=None, edge_sym=sym,
                    edge_ori=torch.zeros_like(sym[..., 0]), node={}, raw=F)
    feat.n_edge_sym = 2
    return feat


def s2_wired(n=32, hidden=32, D=0.02, nonideal='none'):
    """the wired model on S2: mu = log rho + U / T, A = (s_A dpsi + harmonic) / T, k = (D / h^2) T exp(kappa(orientation)).
    Untrained (s_A = 1, b = 0, kappa = 0) it is the SQRA discretisation of the reference PDE (fisher_hodge/e7_wired.py).
    Temperature: drive['T'] = D_test / D_train, or model.edit(T=...)."""
    from .model import HyperME
    hg = Hypergraph.grid_transport(n)
    return HyperME.build_wired(hg, hidden=hidden, k_ref=D * n * n, T_exponent=1.0, nonideal=nonideal,
                               featurizer=s2_wired_featurizer(hg),
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


# ------------------------------------------------------------------------------------------ C3 / Brusselator
C3_RX = [({'A': 1}, {'X': 1}), ({'X': 2, 'Y': 1}, {'X': 3}), ({'B': 1, 'X': 1}, {'Y': 1, 'D': 1}), ({'X': 1}, {'E': 1})]


def c3_featurizer(hg, ntyp=None):
    """drive = dict(drv (B, 2) normalised (log a, log b)) -> features of hyper/crn2.edge_feats.  ntyp = 6 keeps the
    feature layout of the grid model on a hypergraph without transport (well-mixed)."""
    ntyp = ntyp or int(hg.meta['typ'].max()) + 1

    def feat(F):
        drv = F['drv']
        B = drv.shape[0]
        t = hg.on(drv.device, drv.dtype)
        oh = Fn.one_hot(t['typ'], ntyp).to(drv.dtype).expand(B, hg.H, ntyp)
        sym = torch.cat([oh, drv[:, None, :].expand(B, hg.H, drv.shape[1])], -1)
        node = dict(d0=drv[:, :1].expand(B, hg.V), d1=drv[:, 1:2].expand(B, hg.V))
        return dict(U=torch.zeros(B, hg.V, dtype=drv.dtype, device=drv.device), node=node, edge_sym=sym,
                    edge_ori=torch.zeros_like(sym[..., 0]), force=None, raw=F)
    feat.n_edge_sym = ntyp + 2
    return feat


def c3(n=8, hidden=64, use_A=True, method='ssp', nsub=1, well_mixed=False):
    """Brusselator C3 (hyper/gen_brusselator.py): state X, Y; A, B, D, E chemostatted (absorbed into k and A);
    the HyperME2 of hyper/crn2.py.  method='mprk22' integrates with the Patankar scheme (nsub steps per DT).
    well_mixed=True: one site and no transport, with the same parameters (the grid model restricted to homogeneous
    states: k, G and the projected A coincide)."""
    if well_mixed:
        hg = Hypergraph.reaction_transport(['X', 'Y', 'A', 'B', 'D', 'E'], C3_RX, n_sites=1,
                                           chemostats=['A', 'B', 'D', 'E'])
    else:
        hg = Hypergraph.reaction_transport(['X', 'Y', 'A', 'B', 'D', 'E'], C3_RX, grid_n=n, diffusing=['X', 'Y'],
                                           chemostats=['A', 'B', 'D', 'E'])
    feat = c3_featurizer(hg, ntyp=len(C3_RX) + 2)
    with default_dtype(torch.float64):
        kin = Kinetics(feat.n_edge_sym, hidden, last='default', last_bias=0.5, eps=1e-6)
        dr = ProjectedDrive(hg, feat.n_edge_sym, hidden, mlp=True, oriented=False, transport_zero=True)
        fe = FreeEnergy(hg, ('d0', 'd1'), hidden, mode='grad', alpha_init=1.0)
        if not use_A:
            from .model import NoDrive
            dr = NoDrive()
        step = dict(dt_max=C3_DT / 4, theta0=0.5, theta1=0.5, max_sub=2000, t_tol=1e-12)
        if method != 'ssp':
            step = dict(method=method, nsub=nsub)
        m = HyperME(hg, fe, kin, dr, feat, step=step)
    return m.double()


C3_DT = 0.2


def c3_wired_featurizer(hg):
    """drive = dict(logab (B, 2) raw (log a, log b)) -> wired features: chem_logc = (log a, log b, log d = 0, log e = 0)."""
    from .model import wired_featurizer
    assert hg.meta['chemostats'] == ['A', 'B', 'D', 'E']
    base = wired_featurizer(hg)

    def feat(F):
        lab = F['logab']
        return base(dict(chem_logc=torch.cat([lab, torch.zeros_like(lab)], -1)))
    feat.n_edge_sym = base.n_edge_sym
    return feat


def c3_wired(n=8, method='mprk22', nsub=8, chem_prefactor=True, well_mixed=False):
    """Brusselator C3 in the wired mode: mu = log c + mu0 (learned mu0 for X, Y), A = chemostat force of (a, b, d, e)
    plus learned chemostat mu0, log k a table over the hyperedge types, and (chem_prefactor) the chemostat mass-action
    factor in k.  With the true parameters (c3_true_wired) this is the generating model for every (a, b)."""
    kw = dict(n_sites=1) if well_mixed else dict(grid_n=n, diffusing=['X', 'Y'])
    hg = Hypergraph.reaction_transport(['X', 'Y', 'A', 'B', 'D', 'E'], C3_RX, chemostats=['A', 'B', 'D', 'E'], **kw)
    step = dict(dt_max=C3_DT / 4, theta0=0.5, theta1=0.5, max_sub=2000, t_tol=1e-12) if method == 'ssp' else dict(method=method, nsub=nsub)
    with default_dtype(torch.float64):
        m = HyperME.build_wired(hg, k_ref=1.0, T_exponent=0.0, harmonic=False, featurizer=c3_wired_featurizer(hg), step=step,
                                learn_mu0=True, learn_chem_mu0=True, kinetics_table=True, chem_prefactor=chem_prefactor)
    return m.double()


@torch.no_grad()
def c3_true_wired(model, k=(1.0, 1.0, 1.0, 1.0), eps=1e-3, DX=0.4, DY=3.6):
    """set a c3_wired model to the generator of hyper/gen_brusselator.py: reaction r has forward constant k_r (times its
    chemostat reactant) and reverse eps (times its chemostat product), so log k_sym = log sqrt(k_r eps) and the standard
    potentials satisfy <nu+ - nu-, mu0> = log(k_r / eps); transport has k = D_s."""
    x = torch.log(torch.as_tensor(k, dtype=torch.float64) / eps)
    # species stoichiometry per reaction (one site) for X, Y and the chemostats A, B, D, E
    nu = torch.tensor([[-1., 0., 1., 0., 0., 0.],       # rows: <nu+ - nu-> over (X, Y, A, B, D, E); R1 A -> X
                       [-1., 1., 0., 0., 0., 0.],       # R2 2X + Y -> 3X
                       [1., -1., 0., 1., -1., 0.],      # R3 B + X -> Y + D
                       [1., 0., 0., 0., 0., -1.]], dtype=torch.float64)   # R4 X -> E
    mu0 = torch.linalg.pinv(nu) @ x                    # minimum-norm solution (the 2 gauge directions set to 0)
    assert torch.allclose(nu @ mu0, x, atol=1e-12)
    model.free_energy.mu0.copy_(mu0[:2]); model.drive.chem_mu0.copy_(mu0[2:])
    lk = torch.cat([0.5 * torch.log(torch.as_tensor(k, dtype=torch.float64) * eps), torch.log(torch.tensor([DX, DY], dtype=torch.float64))])
    W = model.kinetics.mlp.net[0].weight
    W.copy_(lk[None, :W.shape[1]])
    return model


def load_legacy_c3(model, sd):
    mp = {'kmlp.': 'kinetics.mlp.', 'amlp.': 'drive.mlp.', 'G.': 'free_energy.mlp.'}
    out = {}
    for k, v in sd.items():
        a = next(a for a in mp if k.startswith(a))
        out[mp[a] + k[len(a):]] = v
    missing, unexpected = model.load_state_dict(out, strict=False)
    assert not unexpected and set(missing) <= {'free_energy.alpha'}, (missing, unexpected)
    return model


def load_c3(root=ROOT, device='cpu', name='brus'):
    data = torch.load(os.path.join(root, 'hyper', 'runs', 'data', f'{name}.pt'))
    meta = data['meta']
    dm, ds = meta['drv_mean'].double(), meta['drv_std'].double()
    out = dict(meta=meta)
    for k, d in data.items():
        if k == 'meta':
            continue
        B, T = d['c'].shape[:2]
        out[k] = dict(c=d['c'].reshape(B, T, -1).double().to(device),
                      drive=dict(drv=((d['drv'].double() - dm) / ds).to(device), logab=d['drv'].double().to(device)))
        for extra in ('sigma', 'homog', 'k', 'A'):
            if extra in d:
                out[k][extra] = d[extra]
    return out


PRESETS = dict(s2_tcme=(s2_tcme, load_s2), s2_wired=(s2_wired, load_s2), c1=(c1, load_c1), c3=(c3, load_c3), c3_wired=(c3_wired, load_c3))
