"""The master equation behind the Neural Chart.

For every hyperedge h (a reaction, or a transport step between two cells) the forward and backward rates are

    a_h = k_h exp(<nu+_h, mu> + A_h / 2),   b_h = k_h exp(<nu-_h, mu> - A_h / 2),   J = a - b,   dc/dt = S J.

The model has three parts.
  * FreeEnergy: the chemical potential mu = log c + U / T (+ mu0 / T) (+ ln gamma(c)).  The ideal part is fixed by
    physics.  An optional activity correction ln gamma is a separable, non-decreasing function of c that starts at
    zero, so the free energy stays convex.
  * Mobility: the symmetric prefactor k_h > 0, one value per hyperedge for both directions.  This is the only part
    that carries learned functions of the local material.
  * Drive: the non-conservative force A_h, fixed by the conditions (an external force field, chemostat potentials).
The force f_h = log(a_h / b_h) = -<S_h, mu> + A_h is therefore fixed by physics.  For every parameter value every
moiety in ker S^T is conserved, the rates carry the concentrations of the species they consume, so c stays positive
under the positive integrators, local detailed balance log(a / b) = f holds, and sigma = sum_h J_h f_h >= 0.

The rates have two equivalent forms.  'split' writes a_h = k_h c^{nu+} exp((<nu+ - nu-, phi> + A_h) / 2) with
phi = mu - log c (the square-root approximation of transport on a grid).  The 'activity' form writes
a_h = k_h exp(<nu+, mu> + A_h / 2).  For an ideal free energy they differ only by a c-independent factor in k.
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
class FreeEnergy(nn.Module):
    """F / T = sum_m c_m (log c_m - 1) + sum_m c_m U_m / T + G(c):  mu = log c + U / T + dmu(c).
    The ideal part is fixed.  learn_mu0=True adds learned standard potentials mu0 per species (an ablation; by default
    they are given).  residual=True adds the activity correction dmu = s_mu * r(c), with one gate s_mu per species
    that starts at zero.  r_s(c_s) = mean_j softplus(alpha_sj) tanh(softplus(beta_sj) c_s / rho_scale + delta_sj) is
    separable and non-decreasing, i.e. the gradient of a convex local excess free energy.  residual_anchor=True uses
    r(c) - r(0), so that ln gamma(0) = 0 (the reference state of the standard potentials)."""
    def __init__(s, hg, learn_mu0=False, residual=False, residual_units=16, rho_scale=1.0, residual_anchor=False):
        super().__init__()
        s.hg, s.NS, s.V = hg, hg.n_species, hg.V
        s.mu0 = nn.Parameter(torch.zeros(s.NS)) if learn_mu0 else None
        s.residual, s.RS = residual, rho_scale
        s.residual_anchor = bool(residual_anchor)
        if residual:
            J = residual_units
            s.r_alpha = nn.Parameter(torch.zeros(s.NS, J))
            s.r_beta = nn.Parameter(torch.zeros(s.NS, J))
            s.r_delta = nn.Parameter(torch.linspace(-3.0, 1.0, J).repeat(s.NS, 1))
            s.s_mu = nn.Parameter(torch.zeros(s.NS))

    @property
    def c_dependent(s):
        return s.residual

    def _res_parts(s, cs):
        """per-unit pieces of the correction for cs (B, NS, V): amplitude (NS, J), slope (NS, J), argument (B, NS, V, J)"""
        al, be = Fn.softplus(s.r_alpha).to(cs.dtype), Fn.softplus(s.r_beta).to(cs.dtype) / s.RS
        x = be[None, :, None, :] * cs[..., None] + s.r_delta.to(cs.dtype)[None, :, None, :]
        return al / al.shape[-1], be, x

    def residual_mu(s, cs):
        """the ungated correction r(c) (B, NS, V)"""
        al, _, x = s._res_parts(cs)
        r = (al[None, :, None, :] * torch.tanh(x)).sum(-1)
        if s.residual_anchor:
            r = r - (al * torch.tanh(s.r_delta.to(cs.dtype))).sum(-1)[None, :, None]
        return r

    def _residual_G(s, cs):
        """int_0^c s_mu r per site and species, summed, (B,)"""
        al, be, x = s._res_parts(cs)
        x0 = s.r_delta.to(cs.dtype)[None, :, None, :]
        lc = lambda z: z.abs() + Fn.softplus(-2 * z.abs()) - math.log(2.0)          # log cosh
        G = ((al / be)[None, :, None, :] * (lc(x) - lc(x0))).sum(-1)               # (B, NS, V)
        if s.residual_anchor:
            G = G - (al * torch.tanh(s.r_delta.to(cs.dtype))).sum(-1)[None, :, None] * cs
        return (s.s_mu.to(cs.dtype)[None, :, None] * G).sum((1, 2))

    def potential(s, feats):
        U = feats['U']
        Um = U if U.shape[1] == s.hg.M else U.repeat(1, s.NS)
        if s.mu0 is not None:
            Um = Um + s.mu0.to(Um.dtype).repeat_interleave(s.V)[None]
        return Um / feats['T'][:, None]

    def correction(s, c, ctx):
        cs = c.view(c.shape[0], s.NS, s.V)
        out = torch.zeros_like(cs)
        if s.residual:
            out = out + s.s_mu.to(c.dtype)[None, :, None] * s.residual_mu(cs)
        return out.reshape(c.shape)

    def phi(s, c, ctx):
        return ctx['Ufull'] + s.correction(c, ctx)

    def value(s, c, ctx):
        F = (c * (torch.log(c) - 1 + ctx['Ufull'])).sum(-1)
        if s.residual:
            F = F + s._residual_G(c.view(c.shape[0], s.NS, s.V))
        return F


# ------------------------------------------------------------------------------------------ mobility
class Mobility(nn.Module):
    """k_h = k_ref * T^p * exp(kappa_h), symmetric in the two directions.  kappa_h = core + s_k * r_theta(features):
      core     one log rate per hyperedge type (core='type') or per hyperedge (core='edge'), zero at initialisation;
      r_theta  an MLP of the orientation-invariant hyperedge features, gated by s_k, which starts at zero.
               table=True drops it (log k is the core table alone).
    p = 1 is the Einstein relation, p = 0 keeps k independent of temperature.  chem_prefactor=True: chemostatted
    reactants and products also multiply k by their mass-action factor exp(<nu+ + nu-, log c_chem> / 2).
    split=True: a_h = k_h c^{nu+} exp((<nu+ - nu-, phi> + A_h) / 2); split=False: a_h = k_h exp(<nu+, mu> + A_h / 2)."""
    rho_dependent = False

    def __init__(s, n_edge_feats, hidden=32, k_ref=1.0, T_exponent=1.0, hg=None, table=False, chem_prefactor=True,
                 core='type', residual=None, split=True):
        super().__init__()
        assert core in ('type', 'edge') and hg is not None
        s.chem_prefactor, s.split, s.core_kind = chem_prefactor, split, core
        n_core = hg.H if core == 'edge' else int(hg.meta['typ'].max()) + 1
        s.core = nn.Parameter(torch.zeros(n_core))
        s.residual = MLP(n_edge_feats, hidden, 1) if (residual if residual is not None else not table) else None
        s.s_k = nn.Parameter(torch.zeros(())) if s.residual is not None else None
        s.register_buffer('k_ref', torch.as_tensor(k_ref, dtype=torch.get_default_dtype()))
        s.p = T_exponent
        s.hg = hg

    def log_k(s, feats):
        """kappa (B, H), without k_ref, T and the chemostat factor"""
        x = feats['edge_sym']
        core = s.core.to(x.dtype)
        kap = (core[s.hg.on(x.device, x.dtype)['typ']] if s.core_kind == 'type' else core)[None].expand(x.shape[0], -1)
        if s.residual is not None:
            kap = kap + s.s_k.to(x.dtype) * s.residual(x).squeeze(-1)
        return kap

    def forward(s, feats, hg=None, c=None):
        k = s.k_ref.to(feats['edge_sym'].dtype) * torch.exp(s.log_k(feats))
        if s.p:
            k = k * feats['T'][:, None] ** s.p
        hg = hg or s.hg
        if s.chem_prefactor and hg is not None and feats.get('chem_logc') is not None and 'chem_Np' in hg.meta:
            # chemostats are mass-action species with known activity: besides the force they carry the symmetric
            # factor exp(<nu+ + nu-, log c_chem> / 2), so that a ~ c_ATP and b ~ c_ADP c_Pi exactly
            lc = feats['chem_logc']
            nC = len(hg.meta['chemostats'])
            lcv = lc.repeat_interleave(hg.V, 1) if lc.shape[1] == nC else lc
            Ysum = torch.as_tensor((hg.meta['chem_Np'] + hg.meta['chem_Nm']).toarray(), dtype=lc.dtype, device=lc.device)
            k = k * torch.exp(0.5 * lcv @ Ysum.T)
        return k


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
    default parts.  kinetics is the mobility k (Mobility or a task-specific module with the same interface)."""

    def __init__(s, hg, free_energy, kinetics, drive, featurizer_fn=None, step=None):
        super().__init__()
        s.hg = hg
        s.kinetics, s.free_energy, s.drive = kinetics, free_energy, drive
        s.featurizer = featurizer_fn or featurizer(hg)
        s.step = dict(dt_max=None, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9)
        s.step.update(step or {})
        s.knobs = dict(T=None, drive_scale=1.0)

    @classmethod
    def from_parts(cls, hg, hidden=32, k_ref=1.0, T_exponent=1.0, harmonic=True, featurizer=None, step=None,
              learn_mu0=False, learn_chem_mu0=False, kinetics_table=False, chem_prefactor=True, kinetics_core='type',
              kinetics_split=True, activity_correction=False, rho_scale=1.0, chem_mu0_known=None, activity_anchor=False):
        """kinetics_table=True: log k is the core table alone; kinetics_core='edge': one log rate per hyperedge;
        activity_correction=True: the learned activity correction of FreeEnergy."""
        feat = featurizer or globals()['featurizer'](hg)
        kin = Mobility(feat.n_edge_sym, hidden, k_ref=k_ref, T_exponent=T_exponent, hg=hg, table=kinetics_table,
                       chem_prefactor=chem_prefactor, core=kinetics_core, split=kinetics_split)
        fe = FreeEnergy(hg, learn_mu0, residual=activity_correction, rho_scale=rho_scale, residual_anchor=activity_anchor)
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

    def rates(s, c, drive):
        """one-way fluxes (a, b), each (B, H), evaluated in the log domain (c > 0, which the integrators guarantee)"""
        ctx = s.context(drive)
        phi = s.free_energy.phi(c, ctx)
        k = ctx['k'] if ctx['k'] is not None else s.kinetics(ctx['feats'], s.hg, c)
        A = ctx['A']
        dot = s.hg.dot
        lc = torch.log(c)
        if s.kinetics.split:
            u, v = lc + 0.5 * phi, 0.5 * phi
            return k * torch.exp(dot(u, '+') - dot(v, '-') + 0.5 * A), k * torch.exp(dot(u, '-') - dot(v, '+') - 0.5 * A)
        m = lc + phi
        return k * torch.exp(dot(m, '+') + 0.5 * A), k * torch.exp(dot(m, '-') - 0.5 * A)

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
