"""HyperME: a learned, thermodynamically consistent master equation on a hypergraph.

    a_h = k_h exp(<nu+_h, mu> + A_h/2),   b_h = k_h exp(<nu-_h, mu> - A_h/2),   J = a - b,   dc/dt = S J

with three learned slots:
  * free energy F_theta[c], mu = dF/dc = log c + phi(c),  phi = alpha_s U + dG_theta/dc (+ non-local chi W*c),
  * kinetics k_theta > 0, symmetric (the same for both directions), from hyperedge features and optionally from c,
  * drive A in ker S, from the drive input (line integrals of a force along transport edges, chemical potential
    differences of chemostats) through a learned map followed by the projection onto ker S (or a curl basis).
For every parameter value: every moiety in ker S^T is conserved, a_h and b_h carry their reactant concentrations so
c stays positive, log(a/b) = -S^T mu + A (local detailed balance), sigma = sum_h J_h log(a_h/b_h) >= 0, and with
A = 0 the free energy decreases, dF/dt = -sigma.

The kinetics has two equivalent forms.  'plain' uses k as written above.  'split' uses
k_h = kappa_h exp(-<nu+_h + nu-_h, phi>/2), so that a_h = kappa_h c^{nu+} exp((<nu+ - nu-, phi> + A_h)/2); on a grid
this is the TC-ME / SQRA form of fisher_hodge/models_tcme.py.

Typical use:
    model = HyperME.build(hg)                  # default components
    dc = model(c, drive)                       # dc/dt
    out = model.readouts(c, drive)             # J, f, ell, sigma, excess, housekeeping, cycle affinities, moieties
"""
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
    def __init__(s, hg, n_edge_feats, hidden=64, mlp=True, gain_init=None):
        super().__init__()
        s.hg = hg
        s.mlp = MLP(n_edge_feats + 1, hidden, 1, last='zero') if mlp else None
        s.gain = nn.Parameter(torch.tensor(float(gain_init))) if gain_init is not None else None

    def forward(s, feats):
        sym, ori = feats['edge_sym'], feats['edge_ori']
        A = torch.zeros_like(sym[..., 0])
        if s.gain is not None and feats.get('force') is not None:
            A = A + s.gain * feats['force']
        if s.mlp is not None:
            odd = s.hg.on(sym.device, sym.dtype)['odd']
            ap = s.mlp(torch.cat([sym, ori[..., None]], -1)).squeeze(-1)
            am = s.mlp(torch.cat([sym, -ori[..., None]], -1)).squeeze(-1)
            A = A + odd * (ap - am) + (1 - odd) * ap
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


# ------------------------------------------------------------------------------------------ the model
class HyperME(nn.Module):
    name = 'hyperme'

    def __init__(s, hg, free_energy, kinetics, drive, featurizer=None, step=None):
        super().__init__()
        s.hg = hg
        s.kinetics, s.free_energy, s.drive = kinetics, free_energy, drive
        s.featurizer = featurizer or default_featurizer(hg)
        s.step = dict(dt_max=None, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9)
        s.step.update(step or {})

    @classmethod
    def build(cls, hg, hidden=64, free_energy='grad', node_inputs=('U',), nonlocal_kernel=None, alpha_init=1.0,
              rho_scale=1.0, k_scale=1.0, split=False, rho_dependent_k=False, drive='project', force_gain_init=1.0,
              featurizer=None, step=None):
        """default components for a hypergraph with the default featurizer (see default_featurizer)."""
        feat = featurizer or default_featurizer(hg)
        ne = feat.n_edge_sym
        kin = Kinetics(ne, hidden, scale=k_scale, split=split, rho_dependent=rho_dependent_k, rho_scale=rho_scale)
        fe = FreeEnergy(hg, node_inputs, hidden, mode=free_energy, alpha_init=alpha_init, rho_scale=rho_scale,
                        nonlocal_kernel=nonlocal_kernel)
        dr = NoDrive() if drive == 'none' else ProjectedDrive(hg, ne, hidden, gain_init=force_gain_init)
        return cls(hg, fe, kin, dr, feat, step)

    # ---------------------------------------------------------------- context
    def prepare(s, drive):
        feats = s.featurizer(drive)
        ctx = Context(feats=feats)
        ctx['Ufull'] = s.free_energy.potential(feats)
        ctx['k'] = None if s.kinetics.rho_dependent else s.kinetics(feats)
        ctx['A'] = s.drive(feats)
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
        from .integrators import positive_ssp_rk2
        cfg = dict(s.step); cfg.update(kw)
        return positive_ssp_rk2(s, c, s.context(drive), Dt, **cfg)

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
