"""The master equation behind the Neural Chart: one architecture on every benchmark.

For every hyperedge h (a reaction, or a transport step between two cells), with y = log c + dG(c) (dG the excess potential) and
phi = (U + mu0) / T the potential, the forward and backward rates are

    a_h, b_h = k_h eps_h exp(+- f_h / 2),   eps_h = exp(<nu+_h + nu-_h, y> / 2),   f_h = <nu+_h - nu-_h, y + phi> + A_h,

so a_h = k_h exp(<nu+_h, y> + (<nu+_h - nu-_h, phi> + A_h) / 2), and J = a - b,   dc/dt = S J.

The model has three parts.
  * FreeEnergy: mu = y + phi with dG(c) = grad G_theta(c / V) - grad G_theta(0), G_theta the excess free energy, one scalar MLP per
    site over all local species, weights shared across sites, last layer zero at initialisation.  No monotone or separable
    constraint; G_theta is quadratic in the tails, so the excess potential grows without bound.
  * Mobility: the symmetric prefactor k_h > 0, log k_h = log omega_c(h) + g_theta(class one-hot, xi_h, log cbar_h), times the known
    factors T, the mesh edge weight w_e (default 1) and the chemostat mass-action factor.  cbar_h is the mean over the cells h
    touches of the local density vector of all species, xi_h the local descriptors where a benchmark has them, g_theta one
    MLP whose last layer is zero at initialisation.
  * Drive: the non-conservative force A_h, fixed by the conditions (an external force field, chemostat potentials).
The force f_h = log(a_h / b_h) = -<S_h, mu> + A_h is therefore fixed by the free energy.  For every parameter value every
moiety in ker S^T is conserved, the rates carry the concentrations of the species they consume, so c stays positive
under the positive integrators, local detailed balance log(a / b) = f holds, and sigma = sum_h J_h f_h >= 0.
"""
import copy
import math
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as Fn

SOFTPLUS_INV1 = math.log(math.e - 1)


class MLP(nn.Module):
    """Linear-SiLU-Linear-SiLU-Linear.  last='zero' zeroes the last layer; last_bias sets the last bias to a constant."""
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
    """the condition-dependent, state-independent part of a model (features, k, A), built once by model.prepare."""

    def expand(s, B):
        """repeat a context prepared for one condition (batch 1) to batch B"""
        def ex(v):
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
                return v.expand(B, *v.shape[1:])
            if isinstance(v, dict):
                return {k: ex(x) for k, x in v.items()}
            return v
        return Context({k: ex(v) for k, v in s.items()})


def logmean(a, b, f=None, eps=1e-6):
    """logarithmic mean (a - b) / log(a / b), = (a + b) / 2 when the log ratio is below eps."""
    f = torch.log(a) - torch.log(b) if f is None else f
    sm = f.abs() < eps
    return torch.where(sm, 0.5 * (a + b), (a - b) / torch.where(sm, torch.ones_like(f), f))


# ------------------------------------------------------------------------------------------ features
def featurizer(hg, edge_keys=()):
    """condition dict -> features.  Keys (all optional, leading batch dimension B):
      'U'          (B, V) or (B, M)   potential energy
      'T'          (B,) or scalar     temperature (default 1)
      'line'       (B, H)             line integral of the external force along every hyperedge (energy)
      'chem_logc'  (B, n_chem)        log concentrations of the chemostatted species (or (B, n_chem * V))
      'chem_mu0'   (B, n_chem)        their standard chemical potentials (energy)
      'edge'       (B, H, F)          local geometric or material features of the hyperedges, for the mobility only
    The mobility sees only the hyperedge type and 'edge'.  No condition (U, T, force, chemostat) is a network input."""
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


# ------------------------------------------------------------------------------------------ free energy
def _site_volume(s, hg):
    """the buffer inv_v from the mesh input hg.meta['site_volume'] (None: c is the density)"""
    v = hg.meta.get('site_volume')
    s.register_buffer('inv_v', None if v is None else 1.0 / torch.as_tensor(np.asarray(v), dtype=torch.get_default_dtype()),
                      persistent=False)


def _linear(x, layer):
    """a linear layer in the dtype of x (the parameters may be float64 under an FP32 integrator)"""
    return Fn.linear(x, layer.weight.to(x.dtype), layer.bias.to(x.dtype))


def _mlp(net, x):
    n = net.net
    return _linear(Fn.silu(_linear(Fn.silu(_linear(x, n[0])), n[2])), n[4])


class FreeEnergy(nn.Module):
    """F / T = sum_m c_m (log c_m - 1) + sum_m c_m U_m / T + sum_sites V [G(c / V) - G(0) - <grad G(0), c / V>]:
    mu = log c + U / T + dG(c),  dG(c) = grad G(c / V) - grad G(0), the excess potential of the excess free energy G.
    excess=True: G is one scalar MLP of the local densities of all species, weights shared across sites: Linear-SiLU-Linear-q-
    Linear with q(z) = softplus(z)^2, which is quadratic in the tails, and a zero last layer (G = 0 at initialisation).  The
    gradient is written out, so it needs no second-order autograd.  learn_mu0=True adds standard potentials per species."""
    local_excess = True                                  # neuralchart.fast: the excess potential is site-local

    def __init__(s, hg, learn_mu0=False, excess=False, hidden=32):
        super().__init__()
        s.hg, s.NS, s.V = hg, hg.n_species, hg.V
        s.mu0 = nn.Parameter(torch.zeros(s.NS)) if learn_mu0 else None
        s.excess = bool(excess)
        _site_volume(s, hg)
        if excess:
            s.l1, s.l2, s.l3 = nn.Linear(s.NS, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 1)
            nn.init.zeros_(s.l3.weight); nn.init.zeros_(s.l3.bias)

    @property
    def c_dependent(s):
        return s.excess

    def excess_parameters(s):
        """the parameters of G_theta (empty without it)"""
        return [p for l in (s.l1, s.l2, s.l3) for p in l.parameters()] if s.excess else []

    def density(s, c):
        """the local densities (B, V, NS) of a state c (B, M)"""
        cs = c.view(c.shape[0], s.NS, s.V)
        rho = cs if s.inv_v is None else cs * s.inv_v.to(cs.dtype)[None, None, :]
        return rho.transpose(1, 2)

    def G(s, x):
        """G_theta at densities x (..., NS), (..., 1)"""
        return _linear(Fn.softplus(_linear(Fn.silu(_linear(x, s.l1)), s.l2)) ** 2, s.l3)

    def grad_G(s, x):
        """grad G_theta at densities x (..., NS), (..., NS)"""
        z1 = _linear(x, s.l1)
        p1 = torch.sigmoid(z1)
        z2 = _linear(Fn.silu(z1), s.l2)
        d2 = 2 * Fn.softplus(z2) * torch.sigmoid(z2)
        v = (d2 * s.l3.weight.to(x.dtype)) @ s.l2.weight.to(x.dtype)
        return (v * (p1 + z1 * p1 * (1 - p1))) @ s.l1.weight.to(x.dtype)

    def excess_potential(s, x):
        """the excess potential at densities x (..., NS)"""
        return s.grad_G(x) - s.grad_G(torch.zeros_like(x))

    def potential(s, feats):
        U = feats['U']
        Um = U if U.shape[1] == s.hg.M else U.repeat(1, s.NS)
        if s.mu0 is not None:
            Um = Um + s.mu0.to(Um.dtype).repeat_interleave(s.V)[None]
        return Um / feats['T'][:, None]

    def correction(s, c, ctx):
        """the excess potential at the state c, (B, M)"""
        if not s.excess:
            return torch.zeros_like(c)
        return s.excess_potential(s.density(c)).transpose(1, 2).reshape(c.shape)

    def phi(s, c, ctx):
        return ctx['Ufull'] + s.correction(c, ctx)

    def value(s, c, ctx):
        F = (c * (torch.log(c) - 1 + ctx['Ufull'])).sum(-1)
        if s.excess:
            x = s.density(c)
            x0 = torch.zeros_like(x)
            g = (s.G(x) - s.G(x0))[..., 0] - (s.grad_G(x0) * x).sum(-1)
            if s.inv_v is not None:
                g = g / s.inv_v.to(c.dtype)[None]
            F = F + g.sum(-1)
        return F


# ------------------------------------------------------------------------------------------ mobility
class Mobility(nn.Module):
    """k_h = T w_e chem_h exp(core_c(h) + g(class one-hot, xi_h, log cbar_h)), symmetric in the two directions.
      core     one log rate per hyperedge class, zero at initialisation (log omega);
      g        one MLP (64 hidden units) with a zero last layer; material=False drops it (constant k per class).  Its inputs:
               feats['edge_sym'] (the class one-hot and the local descriptors, where they exist) and log cbar_h, the mean over
               the cells h touches of the local density vector of all species.  cbar_h does not read a parameter.
      T        the temperature (k is proportional to T);  w_e  the mesh edge weight hg.meta['edge_weight'] (default 1);
      chem_h   the chemostat mass-action factor exp(<nu+ + nu-, log c_chem> / 2)."""

    def __init__(s, n_edge_feats, hg, material=True, hidden=64):
        super().__init__()
        s.hg, s.NS, s.V = hg, hg.n_species, hg.V
        s.core = nn.Parameter(torch.zeros(int(hg.meta['typ'].max()) + 1))
        s.net = MLP(n_edge_feats + s.NS, hidden, 1, last='zero') if material else None
        odd = np.asarray(hg.meta['odd'])
        site = np.asarray(hg.meta.get('site', np.full(hg.H, -1)))
        tail = np.where(odd, np.asarray(hg.meta['et']), site) if odd.any() else site
        head = np.where(odd, np.asarray(hg.meta['eh']), site) if odd.any() else site
        s.register_buffer('tail', torch.as_tensor(tail, dtype=torch.long), persistent=False)
        s.register_buffer('head', torch.as_tensor(head, dtype=torch.long), persistent=False)
        s.register_buffer('typ', torch.as_tensor(np.asarray(hg.meta['typ']), dtype=torch.long), persistent=False)
        _site_volume(s, hg)
        w = hg.meta.get('edge_weight')
        s.register_buffer('w', None if w is None else torch.as_tensor(np.asarray(w), dtype=torch.get_default_dtype()), persistent=False)

    @property
    def rho_dependent(s):
        return s.net is not None

    def log_cbar(s, c):
        """log cbar_h (B, H, NS) of a state c (B, M)"""
        cs = c.view(c.shape[0], s.NS, s.V)
        rho = cs if s.inv_v is None else cs * s.inv_v.to(cs.dtype)[None, None, :]
        return torch.log(0.5 * (rho[:, :, s.tail] + rho[:, :, s.head])).transpose(1, 2)

    def state_term(s, x, c):
        """g (B, H) at the features x (B, H, F) and the state c (B, M)"""
        u = s.log_cbar(c)
        return _mlp(s.net, torch.cat([x.to(u.dtype).expand(u.shape[0], -1, -1), u], -1)).squeeze(-1)

    def base(s, feats):
        """the part of k that does not depend on the state (B, H)"""
        x = feats['edge_sym']
        dt = x.dtype
        k = torch.exp(s.core.to(dt)[s.typ])[None] * feats['T'].to(dt)[:, None]
        if s.w is not None:
            k = k * s.w.to(dt)[None]
        hg = s.hg
        if feats.get('chem_logc') is not None and 'chem_Np' in hg.meta:
            # chemostats are mass-action species with known activity: besides the force they carry the symmetric
            # factor exp(<nu+ + nu-, log c_chem> / 2), so that a ~ c_ATP and b ~ c_ADP c_Pi exactly
            lc = feats['chem_logc']
            nC = len(hg.meta['chemostats'])
            lcv = lc.repeat_interleave(hg.V, 1) if lc.shape[1] == nC else lc
            Ysum = torch.as_tensor((hg.meta['chem_Np'] + hg.meta['chem_Nm']).toarray(), dtype=lc.dtype, device=lc.device)
            k = k * torch.exp(0.5 * lcv @ Ysum.T)
        return k

    def forward(s, feats, c):
        """k (B, H) at the state c"""
        k = s.base(feats)
        return k * torch.exp(s.state_term(feats['edge_sym'], c)) if s.rho_dependent else k


# ------------------------------------------------------------------------------------------ drive
class Drive(nn.Module):
    """A_h = s_A int_e F . dl / T + <nu+ - nu-, mu_chem> + sum_k b_k h_k / T, with mu_chem = log c_chem + mu0 / T.
    The external force and the chemostats enter exactly.  s_A (a global scale, starts at 1) is frozen unless the drive
    scale is learned.  One coefficient per harmonic cochain in hg.meta['harmonic'] when harmonic=True.
    learn_chem_mu0=True learns the chemostat standard potentials (an ablation); chem_mu0_known = (C, v) fixes the
    combinations C mu0_chem = v that are known physical data and learns only the rest."""
    def __init__(s, hg, harmonic=True, learn_chem_mu0=False, chem_mu0_known=None):
        super().__init__()
        s.hg = hg
        s.s_A = nn.Parameter(torch.tensor(1.0))
        nC = len(hg.meta.get('chemostats', []))
        s.chem_mu0 = nn.Parameter(torch.zeros(nC)) if (learn_chem_mu0 and nC and chem_mu0_known is None) else None
        s.chem_mu0_free = None
        if chem_mu0_known is not None:
            Cm = np.atleast_2d(np.asarray(chem_mu0_known[0], np.float64)); cv = np.atleast_1d(np.asarray(chem_mu0_known[1], np.float64))
            assert learn_chem_mu0 and Cm.shape[1] == nC
            s.register_buffer('mu0_p', torch.as_tensor(np.linalg.pinv(Cm) @ cv, dtype=torch.get_default_dtype()))
            N = scipy.linalg.null_space(Cm)
            s.register_buffer('mu0_N', torch.as_tensor(N, dtype=torch.get_default_dtype()))
            s.chem_mu0_free = nn.Parameter(torch.zeros(N.shape[1]))
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
            m0 = s.chem_mu0_value()
            if m0 is not None:
                mu = mu + m0.to(mu.dtype)[None] / T
            A = A + s.hg.chemostat_force(mu)
        return A

    def chem_mu0_value(s):
        """the learned chemostat standard potentials (None when they are not learned)"""
        if s.chem_mu0_free is not None:
            return s.mu0_p + s.mu0_N @ s.chem_mu0_free
        return s.chem_mu0


# ------------------------------------------------------------------------------------------ the model
class MasterEquation(nn.Module):
    """MasterEquation(hg, free_energy, kinetics, drive, featurizer, step); MasterEquation.from_parts(hg, ...) builds the
    default parts.  kinetics is the mobility k."""

    def __init__(s, hg, free_energy, kinetics, drive, featurizer_fn=None, step=None):
        super().__init__()
        s.hg = hg
        s.kinetics, s.free_energy, s.drive = kinetics, free_energy, drive
        s.featurizer = featurizer_fn or featurizer(hg)
        s.step = dict(dt_max=None, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9)
        s.step.update(step or {})
        s.knobs = dict(T=None, drive_scale=1.0)

    @classmethod
    def from_parts(cls, hg, harmonic=False, featurizer=None, step=None, learn_mu0=True, learn_chem_mu0=False, chem_mu0_known=None,
                   excess=True, material=True):
        """the default parts.  excess: G_theta;  material: g_theta"""
        feat = featurizer or globals()['featurizer'](hg)
        kin = Mobility(feat.n_edge_sym, hg, material=material)
        fe = FreeEnergy(hg, learn_mu0, excess=excess)
        dr = Drive(hg, harmonic, learn_chem_mu0, chem_mu0_known)
        return cls(hg, fe, kin, dr, feat, step)

    def linear_parameters(s):
        """the parameters that enter log a and log b additively and independently of c: the log-rate core and, when
        they are learned, the standard potentials"""
        out = [s.kinetics.core]
        out += [p for p in (s.free_energy.mu0, s.drive.chem_mu0, s.drive.chem_mu0_free) if p is not None]
        return out

    def edit(s, T=None, drive_scale=None):
        """a copy (sharing the parameters) with the temperature fixed to T and / or the drive A multiplied by
        drive_scale (0 gives the model without drive).  No retraining."""
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
            feats['T'] = torch.full_like(feats['T'], float(s.knobs['T']))
        ctx = Context(feats=feats)
        ctx['Ufull'] = s.free_energy.potential(feats)
        ctx['k'] = s.kinetics.base(feats)
        ctx['A'] = s.drive(feats) * s.knobs['drive_scale']
        return ctx

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    # ---------------------------------------------------------------- dynamics
    def phi(s, c, drive):
        return s.free_energy.phi(c, s.context(drive))

    def mu(s, c, drive):
        return torch.log(c) + s.phi(c, drive)

    def rates(s, c, drive):
        """one-way fluxes (a, b), each (B, H), evaluated in the log domain (c > 0, which the integrators guarantee)"""
        ctx = s.context(drive)
        y = torch.log(c) + s.free_energy.correction(c, ctx)
        phi = ctx['Ufull']
        k = ctx['k']
        if s.kinetics.rho_dependent:
            k = k * torch.exp(s.kinetics.state_term(ctx['feats']['edge_sym'], c))
        A = ctx['A']
        dot = s.hg.dot
        u, v = y + 0.5 * phi, 0.5 * phi
        return k * torch.exp(dot(u, '+') - dot(v, '-') + 0.5 * A), k * torch.exp(dot(u, '-') - dot(v, '+') - 0.5 * A)

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
        """one sampling interval with the positive SSP-RK2 of neuralchart.integrators"""
        from .integrators import positive_ssp_rk2
        cfg = dict(s.step); cfg.update(kw)
        cfg.pop('method', None)
        return positive_ssp_rk2(s, c, s.context(drive), Dt, **cfg)

    def free_energy_value(s, c, drive):
        return s.free_energy.value(c, s.context(drive))

    @torch.no_grad()
    def readouts(s, c, drive):
        """J, f, sigma (total and per hyperedge), mu, F and dF/dt"""
        ctx = s.context(drive)
        a, b = s.rates(c, ctx)
        f = torch.log(a) - torch.log(b)
        J = a - b
        mu = s.mu(c, ctx)
        return dict(a=a, b=b, J=J, f=f, sigma_edge=J * f, sigma=(J * f).sum(-1), A=ctx['A'], mu=mu,
                    free_energy=s.free_energy.value(c, ctx), dFdt=(mu * s.hg.apply_S(J)).sum(-1))
