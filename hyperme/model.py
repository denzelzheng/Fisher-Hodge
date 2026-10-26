"""HyperME: a learned, thermodynamically consistent master equation on a hypergraph.

    a_h = k_h exp(<nu+_h, mu> + A_h/2),   b_h = k_h exp(<nu-_h, mu> - A_h/2),   J = a - b,   dc/dt = S J

and three slots: the free energy (mu = dF/dc), the symmetric kinetics k > 0 and the drive A.  Two modes:
  * mode='wired' (default): the thermodynamics is wired and only the kinetics is learned.  mu = log c + U/T + dmu(c),
    dmu from a physical family with few parameters (none, virial, Debye-Hueckel; zero = ideal);
    A = s_A int_e F.dl / T + chemostat potentials + harmonic coefficients / T, entered exactly (not projected);
    k = k_ref T^p exp(kappa_theta(local hyperedge features)), kappa starting at zero.  No knob (U, T, force,
    chemostat) is a network input, so model.edit(T=..., drive_scale=...) extrapolates along the knobs without
    retraining, and drive_scale=0 is the counterfactual A <- 0.
  * mode='flexible': learned free energy phi = alpha_s U + dG_theta/dc (+ non-local chi W*c), kinetics from hyperedge
    features and optionally c, and a learned drive map projected onto ker S (or a curl basis).
For every parameter value: every moiety in ker S^T is conserved, a_h and b_h carry their reactant concentrations so
c stays positive, log(a/b) = -S^T mu + A (local detailed balance), sigma = sum_h J_h log(a_h/b_h) >= 0, and with
A = 0 the free energy decreases, dF/dt = -sigma.

The kinetics has two equivalent forms.  'plain' uses k as written above.  'split' uses
k_h = kappa_h exp(-<nu+_h + nu-_h, phi>/2), so that a_h = kappa_h c^{nu+} exp((<nu+ - nu-, phi> + A_h)/2); on a grid
this is the TC-ME / SQRA form of fisher_hodge/models_tcme.py.

Typical use:
    model = HyperME(hg)                        # wired mode; HyperME(hg, mode='flexible') for the learned free energy
    dc = model(c, drive)                       # dc/dt
    out = model.readouts(c, drive)             # J, f, ell, sigma, excess, housekeeping, cycle affinities, moieties
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

SOFTPLUS_INV1 = math.log(math.e - 1)


class MLP(nn.Module):
    """Linear-SiLU-Linear-SiLU-Linear (the trunk of every prototype model).  last='zero' zeroes the last layer;
    last_bias sets the last bias to a constant after the default init."""
    def __init__(s, i, h, o, last='default', last_bias=None):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))
        if last == 'zero':
            nn.init.zeros_(s.net[-1].weight); nn.init.zeros_(s.net[-1].bias)
        if last_bias is not None:
            nn.init.constant_(s.net[-1].bias, last_bias)

    def forward(s, x):
        return s.net(x)


class Context(dict):
    """the drive-dependent, state-independent part of a model (features, static k, A); built by model.prepare."""

    def expand(s, B):
        """repeat a context prepared for one drive (batch 1) to batch B, e.g. for many stochastic trajectories."""
        def ex(v):
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
                return v.expand(B, *v.shape[1:])
            if isinstance(v, dict):
                return {k: ex(x) for k, x in v.items()}
            return v
        return Context({k: ex(v) for k, v in s.items()})


def logmean(a, b, f=None, eps=1e-6):
    """logarithmic mean (a - b) / log(a / b), = (a + b)/2 when the log ratio is below eps."""
    f = torch.log(a) - torch.log(b) if f is None else f
    sm = f.abs() < eps
    return torch.where(sm, 0.5 * (a + b), (a - b) / torch.where(sm, torch.ones_like(f), f))


def site_grad(fn, c):
    """d/dc of the scalar fn(c).sum() (autograd); keeps the graph when grad mode is on."""
    if torch.is_grad_enabled():
        x = c if c.requires_grad else c.detach().requires_grad_(True)
        return torch.autograd.grad(fn(x).sum(), x, create_graph=True)[0]
    with torch.enable_grad():
        x = c.detach().requires_grad_(True)
        return torch.autograd.grad(fn(x).sum(), x)[0].detach()


_GL = {}


def gauss_legendre(n, device, dtype):
    key = (n, str(device), dtype)
    if key not in _GL:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL[key] = (torch.tensor(0.5 * (x + 1), device=device, dtype=dtype), torch.tensor(0.5 * w, device=device, dtype=dtype))
    return _GL[key]


# ------------------------------------------------------------------------------------------ featurizer
def default_featurizer(hg):
    """drive dict -> feature dict.  Recognised drive keys (all optional, leading batch dimension B):
      'U'      (B, V) or (B, M)  external potential (applied as alpha_s U)
      'force'  (B, H)            drive along hyperedges: line integral of a non-conservative force for transport
                                 edges, chemical potential difference of chemostats for reactions
      'node'   dict name -> (B, V)  extra node features
      'edge'   (B, H, F)         extra orientation-invariant hyperedge features
    Hyperedge features: one-hot hyperedge type, |force|, |dU| on transport edges, extra edge features.
    The oriented scalar is the force (zero if absent)."""
    ntyp = int(hg.meta['typ'].max()) + 1

    def feat(drive):
        ref = next(v for v in drive.values() if torch.is_tensor(v))
        B, dev, dt = ref.shape[0], ref.device, ref.dtype
        t = hg.on(dev, dt)
        U = drive.get('U', torch.zeros(B, hg.V, device=dev, dtype=dt))
        Um = U if U.shape[1] == hg.M else U[:, t['site_of']]
        force = drive.get('force', torch.zeros(B, hg.H, device=dev, dtype=dt))
        dU = hg.dot(Um, '-') - hg.dot(Um, '+')
        cols = [Fn.one_hot(t['typ'], ntyp).to(dt).expand(B, -1, -1), force.abs()[..., None], (t['odd'] * dU).abs()[..., None]]
        if 'edge' in drive:
            cols.append(drive['edge'])
        node = dict(U=U if U.shape[1] == hg.V else U.view(B, hg.n_species, hg.V).mean(1))
        node.update(drive.get('node', {}))
        return dict(U=U, node=node, edge_sym=torch.cat(cols, -1), edge_ori=force, force=force, raw=drive)
    feat.n_edge_sym = ntyp + 2
    return feat


# ------------------------------------------------------------------------------------------ free energy
class FreeEnergy(nn.Module):
    """F(c) = sum_m c_m (log c_m - 1) + sum_m alpha_s U_m c_m + G_theta(c) + chi/2 sum_s c_s . W c_s.
    mode='grad': G_theta = sum_v MLP(c_{.,v}/rho_scale, node features_v [, (W c)_{.,v}]) and phi = dG/dc by autograd
                 (any number of species per site; non-local features allowed).
    mode='direct': phi_{s,v} = out_scale * MLP(c_{s,v}/rho_scale, node features_v) is the derivative of the local
                 G = sum_m int_0^{c_m} phi; one species per call, no non-local term (fisher_hodge/models_tcme.py)."""
    def __init__(s, hg, node_inputs=('U',), hidden=128, mode='grad', alpha_init=1.0, rho_scale=1.0, out_scale=1.0,
                 correction=True, nonlocal_kernel=None):
        super().__init__()
        assert mode in ('grad', 'direct')
        assert nonlocal_kernel is None or mode == 'grad'
        s.hg, s.mode, s.node_inputs, s.RS, s.out_scale = hg, mode, tuple(node_inputs), rho_scale, out_scale
        s.NS, s.V = hg.n_species, hg.V
        s.nonlocal_ = nonlocal_kernel is not None
        if correction:
            if mode == 'grad':
                nin = s.NS * (2 if s.nonlocal_ else 1) + len(s.node_inputs)
            else:
                nin = 1 + len(s.node_inputs) + (s.NS if s.NS > 1 else 0)
            s.mlp = MLP(nin, hidden, 1, last='zero')
        else:
            s.mlp = None
        s.alpha = nn.Parameter(torch.full((s.NS,), float(alpha_init)))
        if s.nonlocal_:
            W = torch.as_tensor(nonlocal_kernel, dtype=torch.get_default_dtype())
            assert W.shape == (s.V, s.V) and torch.allclose(W, W.T), 'non-local kernel must be a symmetric (V, V) matrix'
            s.register_buffer('W', W)
            s.chi = nn.Parameter(torch.zeros(()))
        s.force_zero_correction = False          # property tests: G = 0

    @property
    def c_dependent(s):
        return (s.mlp is not None and not s.force_zero_correction) or s.nonlocal_

    def potential(s, feats):
        """alpha_s U as a (B, M) field"""
        U = feats['U']; B = U.shape[0]
        if U.shape[1] == s.V * s.NS and s.NS > 1:
            return s.alpha[s.hg.on(U.device, U.dtype)['species_of']] * U
        return (s.alpha[None, :, None] * U[:, None, :]).reshape(B, -1)

    def _node_cols(s, feats):
        return [feats['node'][k] for k in s.node_inputs]

    def G_total(s, c, ctx):
        """grad mode: G_theta(c) + chi/2 c.Wc per sample, (B,)"""
        B = c.shape[0]
        cs = c.view(B, s.NS, s.V)
        out = 0.0
        nl = None
        if s.nonlocal_:
            nl = cs @ s.W.to(c.dtype)                                    # (B, NS, V)
            out = 0.5 * s.chi * (cs * nl).sum((1, 2))
        if s.mlp is not None and not s.force_zero_correction:
            cols = [cs.transpose(1, 2) / s.RS]
            if nl is not None:
                cols.append(nl.transpose(1, 2) / s.RS)
            cols += [x[..., None] for x in s._node_cols(ctx['feats'])]
            out = out + s.out_scale * s.mlp(torch.cat(cols, -1)).squeeze(-1).sum(-1)
        return out

    def _direct(s, c, ctx):
        B = c.shape[0]
        cols = [(c / s.RS).view(B, s.NS, s.V)]
        cols += [x[:, None, :].expand(B, s.NS, s.V) for x in s._node_cols(ctx['feats'])]
        x = torch.stack(cols, -1)
        if s.NS > 1:
            x = torch.cat([x, torch.eye(s.NS, dtype=c.dtype, device=c.device)[None, :, None, :].expand(B, s.NS, s.V, s.NS)], -1)
        return s.out_scale * s.mlp(x).squeeze(-1).reshape(B, -1)

    def correction(s, c, ctx):
        """dG/dc, (B, M)"""
        if not s.c_dependent:
            return torch.zeros_like(c)
        if s.mode == 'direct':
            return s._direct(c, ctx)
        return site_grad(lambda x: s.G_total(x, ctx), c)

    def phi(s, c, ctx):
        """excess chemical potential mu - log c, (B, M)"""
        return ctx['Ufull'] + s.correction(c, ctx)

    def value(s, c, ctx, nquad=24):
        """F(c) per sample, (B,)"""
        F = (c * (torch.log(c) - 1 + ctx['Ufull'])).sum(-1)
        if not s.c_dependent:
            return F
        if s.mode == 'grad':
            return F + s.G_total(c, ctx)
        x, w = gauss_legendre(nquad, c.device, c.dtype)                  # int_0^c phi_corr(s) ds
        vals = torch.stack([s._direct(c * xi, ctx) for xi in x], 0)      # (Q, B, M)
        return F + (c * (w[:, None, None] * vals).sum(0)).sum(-1)


# ------------------------------------------------------------------------------------------ kinetics
class Kinetics(nn.Module):
    """k_h = scale * softplus(MLP(orientation-invariant hyperedge features [, symmetric functions of c])) + eps.
    Symmetric by construction (one k per hyperedge for both directions).  split=True multiplies by
    exp(-<nu+ + nu-, phi>/2) (TC-ME form)."""
    def __init__(s, n_edge_feats, hidden=128, scale=1.0, split=False, rho_dependent=False, rho_scale=1.0,
                 last='zero', last_bias=SOFTPLUS_INV1, eps=0.0):
        super().__init__()
        s.scale, s.split, s.rho_dependent, s.RS, s.eps = scale, split, rho_dependent, rho_scale, eps
        s.mlp = MLP(n_edge_feats + (2 if rho_dependent else 0), hidden, 1, last=last, last_bias=last_bias)

    def forward(s, feats, hg=None, c=None):
        x = feats['edge_sym']
        if s.rho_dependent:
            p, m = hg.dot(c, '+') / s.RS, hg.dot(c, '-') / s.RS
            x = torch.cat([x, (p + m)[..., None], (p - m).abs()[..., None]], -1)
        return s.scale * Fn.softplus(s.mlp(x).squeeze(-1)) + s.eps


# ------------------------------------------------------------------------------------------ drive
class ProjectedDrive(nn.Module):
    """A = P_kerS A_raw,  A_raw = gain * force + MLP(sym, ori) [antisymmetrised on orientation-odd hyperedges].
    The projection removes the part of the drive that a change of mu could absorb."""
    def __init__(s, hg, n_edge_feats, hidden=64, mlp=True, gain_init=None, oriented=True, transport_zero=False):
        super().__init__()
        s.hg, s.oriented, s.transport_zero = hg, oriented, transport_zero
        s.mlp = MLP(n_edge_feats + (1 if oriented else 0), hidden, 1, last='zero') if mlp else None
        s.gain = nn.Parameter(torch.tensor(float(gain_init))) if gain_init is not None else None

    def forward(s, feats):
        sym, ori = feats['edge_sym'], feats['edge_ori']
        A = torch.zeros_like(sym[..., 0])
        if s.gain is not None and feats.get('force') is not None:
            A = A + s.gain * feats['force']
        if s.mlp is not None:
            odd = s.hg.on(sym.device, sym.dtype)['odd']
            if s.oriented:
                ap = s.mlp(torch.cat([sym, ori[..., None]], -1)).squeeze(-1)
                am = s.mlp(torch.cat([sym, -ori[..., None]], -1)).squeeze(-1)
                A = A + odd * (ap - am) + (1 - odd) * ap
            else:                                   # no oriented input: transport edges get no raw drive
                ap = s.mlp(sym).squeeze(-1)
                A = A + ((1 - odd) * ap if s.transport_zero else ap)
        return s.hg.project_cycles(A)


class NoDrive(nn.Module):
    """A = 0: detailed balance (reversible model)."""
    def forward(s, feats):
        return torch.zeros_like(feats['edge_sym'][..., 0])


class CurlDrive(nn.Module):
    """Single-species transport on the n x n torus: A = d1^T psi_theta + sum_k b_k h_k, which lies in ker S = ker d0^T
    without projection.  psi_theta lives on faces, psi = gam * P + MLP(P, U at the face), P = stream function of the
    drift (drive key 'dpsi' = line integral of the drift along every edge); the harmonic coefficients b = h v + W_b
    mean(dpsi).  This is the A of fisher_hodge/models_tcme.py."""
    def __init__(s, hidden=128):
        super().__init__()
        s.psi = MLP(2, hidden, 1, last='zero')
        s.gam = nn.Parameter(torch.tensor(0.0))
        s.v = nn.Parameter(torch.zeros(2))
        s.Wb = nn.Parameter(torch.zeros(2, 2))

    def forward(s, feats):
        from .complex import GridOps as G
        raw = feats['raw']
        U, dp = raw['U'], raw['dpsi']
        n = U.shape[-1]; h = 1.0 / n
        P = G.stream_from_dpsi(dp)
        psi = s.gam * P + s.psi(torch.stack([P, G.node_to_corner(U)], -1)).squeeze(-1)
        b = h * s.v.view(1, 2) + dp.mean((-1, -2)) @ s.Wb.T
        return G.flat_e(G.d1T(psi) + b[:, :, None, None])


# ------------------------------------------------------------------------------------------ wired thermodynamics
def wired_featurizer(hg, edge_keys=()):
    """drive dict for the wired mode -> features.  Drive keys (all optional, leading batch dimension B):
      'U'          (B, V) or (B, M)   potential energy
      'T'          (B,) or scalar     temperature (default 1)
      'line'       (B, H)             line integral of the external force along every hyperedge, int_e F . dl (energy)
      'chem_logc'  (B, n_chem)        log concentrations of the chemostatted species (or (B, n_chem * V))
      'chem_mu0'   (B, n_chem)        their standard chemical potentials (energy), optional
      'edge'       (B, H, F)          local geometric / material features of the hyperedges, for the kinetics only
    The kinetics sees only the hyperedge type and 'edge'; no knob (U, T, force, chemostat) is a network input."""
    ntyp = int(hg.meta['typ'].max()) + 1

    def feat(drive):
        ref = next(v for v in drive.values() if torch.is_tensor(v) and v.dim() >= 1)
        B, dev, dt = ref.shape[0], ref.device, ref.dtype
        t = hg.on(dev, dt)
        T = torch.as_tensor(drive.get('T', 1.0), dtype=dt, device=dev)
        T = T.expand(B) if T.dim() == 0 else T.reshape(B)
        cols = [Fn.one_hot(t['typ'], ntyp).to(dt).expand(B, -1, -1)]
        if 'edge' in drive:
            cols.append(drive['edge'])
        return dict(U=drive.get('U', torch.zeros(B, hg.V, dtype=dt, device=dev)), T=T,
                    line=drive.get('line'), chem_logc=drive.get('chem_logc'), chem_mu0=drive.get('chem_mu0'),
                    edge_sym=torch.cat(cols, -1), edge_ori=torch.zeros(B, hg.H, dtype=dt, device=dev), node={}, raw=drive)
    feat.n_edge_sym = ntyp + len(edge_keys)
    return feat


class WiredFreeEnergy(nn.Module):
    """F / T = sum_m c_m (log c_m - 1) + sum_m c_m U_m / T + G_nonideal(c):  mu = log c + U / T + dmu(c).
    The ideal part is fixed.  nonideal (per site, parameters few, zero = ideal):
      'none'
      'virial'         G = 1/2 sum_v c_v^T B c_v,  B = L^T L (positive semidefinite, so F stays convex);
                       L starts at 1e-3 I, i.e. close to ideal
      'debye_huckel'   dmu_s = -z_s^2 a sqrt(I) / (1 + sqrt(I)),  I = 1/2 sum_s z_s^2 c_s  (a starts at 0)
    """
    def __init__(s, hg, nonideal='none', charges=None):
        super().__init__()
        assert nonideal in ('none', 'virial', 'debye_huckel')
        s.hg, s.nonideal, s.NS, s.V = hg, nonideal, hg.n_species, hg.V
        if nonideal == 'virial':
            s.L = nn.Parameter(1e-3 * torch.eye(s.NS))
        elif nonideal == 'debye_huckel':
            z = torch.as_tensor(charges if charges is not None else [1.0] * s.NS, dtype=torch.get_default_dtype())
            s.register_buffer('z2', z ** 2)
            s.a = nn.Parameter(torch.zeros(()))

    @property
    def c_dependent(s):
        return s.nonideal != 'none'

    def potential(s, feats):
        U = feats['U']; B = U.shape[0]
        Um = U if U.shape[1] == s.hg.M else U.repeat(1, s.NS)
        return Um / feats['T'][:, None]

    def correction(s, c, ctx):
        if s.nonideal == 'none':
            return torch.zeros_like(c)
        cs = c.view(c.shape[0], s.NS, s.V)
        if s.nonideal == 'virial':
            Bm = s.L.T @ s.L
            return torch.einsum('st,btv->bsv', Bm.to(c.dtype), cs).reshape(c.shape)
        z2 = s.z2.to(c.dtype)
        sI = torch.sqrt(0.5 * (z2[None, :, None] * cs).sum(1, keepdim=True))
        return (-z2[None, :, None] * s.a * sI / (1 + sI)).expand_as(cs).reshape(c.shape)

    def phi(s, c, ctx):
        return ctx['Ufull'] + s.correction(c, ctx)

    def value(s, c, ctx):
        F = (c * (torch.log(c) - 1 + ctx['Ufull'])).sum(-1)
        if s.nonideal == 'virial':
            cs = c.view(c.shape[0], s.NS, s.V)
            F = F + 0.5 * torch.einsum('bsv,st,btv->b', cs, (s.L.T @ s.L).to(c.dtype), cs)
        elif s.nonideal == 'debye_huckel':
            cs = c.view(c.shape[0], s.NS, s.V)
            I = 0.5 * (s.z2.to(c.dtype)[None, :, None] * cs).sum(1)
            sI = torch.sqrt(I)
            F = F - 2 * s.a * (I - 2 * sI + 2 * torch.log1p(sI)).sum(-1)
        return F


class WiredKinetics(nn.Module):
    """k_h = k_ref * T^p * exp(kappa_theta(local features)), symmetric, split (SQRA) form.  kappa starts at zero, so the
    untrained model runs at the reference rates (for transport on a grid k_ref = D / h^2 gives the SQRA scheme).
    p = 1 is the Einstein relation (mobility at fixed friction), p = 0 keeps k independent of temperature."""
    split, rho_dependent = True, False

    def __init__(s, n_edge_feats, hidden=32, k_ref=1.0, T_exponent=1.0):
        super().__init__()
        s.mlp = MLP(n_edge_feats, hidden, 1, last='zero')
        s.register_buffer('k_ref', torch.as_tensor(k_ref, dtype=torch.get_default_dtype()))
        s.p = T_exponent

    def forward(s, feats, hg=None, c=None):
        k = s.k_ref.to(feats['edge_sym'].dtype) * torch.exp(s.mlp(feats['edge_sym']).squeeze(-1))
        return k * feats['T'][:, None] ** s.p if s.p else k


class WiredDrive(nn.Module):
    """A_h = s_A int_e F . dl / T + <nu+ - nu-, mu_chem> + sum_k b_k h_k / T, with mu_chem = log c_chem + mu0 / T.
    Not projected: the external force and the chemostats enter exactly (a gradient part of the force acts like a
    potential).  Learned: the global scale s_A (starts at 1) and one coefficient per harmonic cochain in
    hg.meta['harmonic'] (start at 0)."""
    def __init__(s, hg, harmonic=True):
        super().__init__()
        s.hg = hg
        s.s_A = nn.Parameter(torch.tensor(1.0))
        Hm = hg.meta.get('harmonic') if harmonic else None
        if Hm is not None and Hm.shape[1]:
            s.register_buffer('Hm', torch.as_tensor(Hm, dtype=torch.get_default_dtype()))
            s.b = nn.Parameter(torch.zeros(Hm.shape[1]))
        else:
            s.Hm, s.b = None, None

    def forward(s, feats):
        T = feats['T'][:, None]
        dt = feats['edge_sym'].dtype
        A = torch.zeros_like(feats['edge_sym'][..., 0])
        if feats.get('line') is not None:
            A = A + s.s_A * feats['line'] / T
        if s.b is not None:
            A = A + (s.Hm.to(dt) @ s.b.to(dt))[None] / T
        if feats.get('chem_logc') is not None:
            mu = feats['chem_logc']
            if feats.get('chem_mu0') is not None:
                mu = mu + feats['chem_mu0'] / T
            A = A + s.hg.chemostat_force(mu)
        return A


# ------------------------------------------------------------------------------------------ the model
class HyperME(nn.Module):
    """HyperME(hg) builds the default wired model; HyperME(hg, mode='flexible') the learned-free-energy model; or pass
    free_energy, kinetics, drive (and a featurizer) explicitly."""
    name = 'hyperme'

    def __init__(s, hg, free_energy=None, kinetics=None, drive=None, featurizer=None, step=None, mode='wired', **build_kw):
        super().__init__()
        if free_energy is None:
            proto = (HyperME.build_wired if mode == 'wired' else HyperME.build_flexible)(hg, featurizer=featurizer, step=step, **build_kw)
            free_energy, kinetics, drive, featurizer = proto.free_energy, proto.kinetics, proto.drive, proto.featurizer
        s.hg = hg
        s.kinetics, s.free_energy, s.drive = kinetics, free_energy, drive
        s.featurizer = featurizer or default_featurizer(hg)
        s.mode = 'wired' if isinstance(free_energy, WiredFreeEnergy) else 'flexible'
        s.step = dict(dt_max=None, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9)
        s.step.update(step or {})
        s.knobs = dict(T=None, drive_scale=1.0)

    @classmethod
    def build(cls, hg, mode='wired', **kw):
        """default components: mode='wired' (thermodynamics wired, kinetics learned) or 'flexible'."""
        return cls.build_wired(hg, **kw) if mode == 'wired' else cls.build_flexible(hg, **kw)

    @classmethod
    def build_wired(cls, hg, hidden=32, k_ref=1.0, T_exponent=1.0, nonideal='none', charges=None, harmonic=True,
                    featurizer=None, step=None):
        feat = featurizer or wired_featurizer(hg)
        kin = WiredKinetics(feat.n_edge_sym, hidden, k_ref=k_ref, T_exponent=T_exponent)
        fe = WiredFreeEnergy(hg, nonideal, charges)
        dr = WiredDrive(hg, harmonic)
        return cls(hg, fe, kin, dr, feat, step)

    @classmethod
    def build_flexible(cls, hg, hidden=64, free_energy='grad', node_inputs=('U',), nonlocal_kernel=None, alpha_init=1.0,
                       rho_scale=1.0, k_scale=1.0, split=False, rho_dependent_k=False, drive='project', force_gain_init=1.0,
                       featurizer=None, step=None):
        """learned free energy (MLP correction), learned kinetics and a learned drive map projected onto ker S, with
        the default featurizer (see default_featurizer)."""
        feat = featurizer or default_featurizer(hg)
        ne = feat.n_edge_sym
        kin = Kinetics(ne, hidden, scale=k_scale, split=split, rho_dependent=rho_dependent_k, rho_scale=rho_scale)
        fe = FreeEnergy(hg, node_inputs, hidden, mode=free_energy, alpha_init=alpha_init, rho_scale=rho_scale,
                        nonlocal_kernel=nonlocal_kernel)
        dr = NoDrive() if drive == 'none' else ProjectedDrive(hg, ne, hidden, gain_init=force_gain_init)
        return cls(hg, fe, kin, dr, feat, step)

    def edit(s, T=None, drive_scale=None):
        """a copy of the model (sharing its parameters) with the temperature fixed to T and / or the drive A
        multiplied by drive_scale (0 gives the counterfactual without drive, A <- 0).  No retraining: in the wired
        mode T enters as U / T, line integral / T and k * T^p."""
        m = copy.copy(s)
        m.knobs = dict(s.knobs)
        if T is not None:
            m.knobs['T'] = T
        if drive_scale is not None:
            m.knobs['drive_scale'] = drive_scale
        return m

    # ---------------------------------------------------------------- context
    def prepare(s, drive):
        feats = s.featurizer(drive)
        if s.knobs['T'] is not None:
            assert 'T' in feats, 'temperature edits need the wired featurizer'
            feats['T'] = torch.full_like(feats['T'], float(s.knobs['T']))
        ctx = Context(feats=feats)
        ctx['Ufull'] = s.free_energy.potential(feats)
        ctx['k'] = None if s.kinetics.rho_dependent else s.kinetics(feats)
        ctx['A'] = s.drive(feats) * s.knobs['drive_scale']
        return ctx

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    # ---------------------------------------------------------------- dynamics
    def phi(s, c, drive):
        return s.free_energy.phi(c, s.context(drive))

    def mu(s, c, drive):
        return torch.log(c) + s.phi(c, drive)

    def rates(s, c, drive, zero_safe=False):
        """one-way fluxes (a, b), each (B, H).  The default evaluates them in the log domain (needs c > 0, which the
        integrators guarantee); zero_safe=True multiplies the mass-action factors explicitly, so that a hyperedge with
        an absent reactant has an exactly zero rate (molecule numbers in the samplers)."""
        ctx = s.context(drive)
        phi = s.free_energy.phi(c, ctx)
        k = ctx['k'] if ctx['k'] is not None else s.kinetics(ctx['feats'], s.hg, c)
        A = ctx['A']
        dot = s.hg.dot
        if not zero_safe:
            lc = torch.log(c)
            if s.kinetics.split:
                u, v = lc + 0.5 * phi, 0.5 * phi
                return k * torch.exp(dot(u, '+') - dot(v, '-') + 0.5 * A), k * torch.exp(dot(u, '-') - dot(v, '+') - 0.5 * A)
            m = lc + phi
            return k * torch.exp(dot(m, '+') + 0.5 * A), k * torch.exp(dot(m, '-') - 0.5 * A)
        pp, pm = dot(phi, '+'), dot(phi, '-')
        cp, cm = s.hg.mass_action(c, '+'), s.hg.mass_action(c, '-')
        if s.kinetics.split:
            x = 0.5 * (pp - pm + A)
            return k * cp * torch.exp(x), k * cm * torch.exp(-x)
        return k * cp * torch.exp(pp + 0.5 * A), k * cm * torch.exp(pm - 0.5 * A)

    def flux(s, c, drive):
        a, b = s.rates(c, drive)
        return a - b

    def forward(s, c, drive):
        """dc/dt = S J"""
        return s.hg.apply_S(s.flux(c, drive))

    rhs = forward

    def escape_ratio(s, c, a, b):
        """max_m (net consumption of m) / c_m per sample, (B,); a forward-Euler stage of length dt keeps c > 0
        when dt * escape_ratio < 1."""
        D = s.hg.consumption(a, b)
        r = torch.where(c > 0, D / torch.where(c > 0, c, torch.ones_like(c)), torch.zeros_like(c))
        return r.amax(-1)

    def advance(s, c, drive, Dt, **kw):
        """one sampling interval with the model's integrator: step['method'] = 'ssp' (explicit positive substeps,
        default), 'mprk22' or 'patankar' (linearly implicit Patankar schemes, step['nsub'] steps per Dt)."""
        from .integrators import positive_ssp_rk2, mprk
        cfg = dict(s.step); cfg.update(kw)
        method = cfg.pop('method', 'ssp')
        if method == 'ssp':
            return positive_ssp_rk2(s, c, s.context(drive), Dt, **cfg)
        cfg.pop('vectorized', None)
        return mprk(s, c, s.context(drive), Dt, nsub=cfg.get('nsub', 1), order=2 if method == 'mprk22' else 1)

    def free_energy_value(s, c, drive):
        return s.free_energy.value(c, s.context(drive))

    # ---------------------------------------------------------------- linear case
    @property
    def is_linear(s):
        return s.hg.unimolecular and not s.free_energy.c_dependent and not s.kinetics.rho_dependent

    def transition_rates(s, c_ref, drive):
        """one-particle jump rates (q+, q-) of a unimolecular hypergraph, frozen at c_ref: a = q+ c_tail, b = q- c_head."""
        assert s.hg.unimolecular, 'transition rates need a unimolecular hypergraph'
        t = s.hg.on(c_ref.device, c_ref.dtype)
        a, b = s.rates(c_ref, drive)
        return a / c_ref[:, t['tail']], b / c_ref[:, t['head']]

    def generator(s, c_ref, drive):
        """dense generator L (B, M, M) with dc/dt = L c of the master equation frozen at c_ref (exact for a linear
        model).  The row-convention Markov generator is Q = L^T."""
        qp, qm = s.transition_rates(c_ref, drive)
        return generator_from_rates(s.hg, qp, qm)

    # ---------------------------------------------------------------- readouts
    @torch.no_grad()
    def readouts(s, c, drive, hodge=True, cycles=None):
        """J, f, ell, sigma (total and per hyperedge), excess / housekeeping split of sigma (ell-weighted projection
        of f onto im S^T), cycle affinities Z^T f (= Z^T A), moieties c @ L, mu, F, dF/dt."""
        ctx = s.context(drive)
        a, b = s.rates(c, ctx)
        f = torch.log(a) - torch.log(b)
        J = a - b
        ell = logmean(a, b, f)
        mu = s.mu(c, ctx)
        t = s.hg.on(c.device, c.dtype)
        out = dict(a=a, b=b, J=J, f=f, ell=ell, sigma_edge=J * f, sigma=(J * f).sum(-1), A=ctx['A'], mu=mu,
                   free_energy=s.free_energy.value(c, ctx), dFdt=(mu * s.hg.apply_S(J)).sum(-1),
                   conserved=c @ t['L'], cycle_affinity=f @ t['Z'])
        if cycles is not None:
            out['cycles'] = f @ torch.as_tensor(cycles, dtype=f.dtype, device=f.device).T
        if hodge:
            g = weighted_gradient_part(s.hg, f, ell)
            ex = (ell * g * g).sum(-1); hk = (ell * (f - g) ** 2).sum(-1)
            out.update(f_grad=g, f_cyc=f - g, excess=ex, housekeeping=hk)
        return out


def generator_from_rates(hg, qp, qm):
    """unimolecular one-way rates (B, H) -> dense generator L (B, M, M), dc/dt = L c."""
    t = hg.on(qp.device, qp.dtype)
    B, M = qp.shape[0], hg.M
    L = torch.zeros(B, M * M, dtype=qp.dtype, device=qp.device)
    ti, hi = t['tail'], t['head']
    L.index_add_(1, hi * M + ti, qp); L.index_add_(1, ti * M + ti, -qp)
    L.index_add_(1, ti * M + hi, qm); L.index_add_(1, hi * M + hi, -qm)
    return L.view(B, M, M)


def weighted_gradient_part(hg, f, w):
    """ell-weighted Hodge projection of a hyperedge field f onto im S^T: g = S^T psi with
    S diag(w) S^T psi = S diag(w) f (gauge fixed on ker S^T).  float64 dense solve, (B, H)."""
    dt = f.dtype
    f64, w64 = f.double(), w.double()
    S = torch.as_tensor(hg.S, dtype=torch.float64, device=f.device)               # (M, H)
    Lb = hg.on(f.device, torch.float64)['L']
    K = torch.einsum('mh,bh,nh->bmn', S, w64, S)
    K = K + (w64.mean(-1)[:, None, None]) * (Lb @ Lb.T)[None]
    rhs = torch.einsum('mh,bh->bm', S, w64 * f64)
    psi = torch.linalg.solve(K, rhs.unsqueeze(-1)).squeeze(-1)
    r = rhs - torch.einsum('bmn,bn->bm', K, psi)                                     # one refinement step
    psi = psi + torch.linalg.solve(K, r.unsqueeze(-1)).squeeze(-1)
    return (psi @ S).to(dt)
