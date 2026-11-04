"""The reaction networks and their reference dynamics.

Schloegl model (schlogl, schlogl_nonideal, schlogl_nonideal_strong): A + 2X <-> 3X, X <-> B at every site of the n x n periodic grid, X diffuses,
A and B are chemostats.  Every hyperedge is mass action in the activity y = x gamma(x),
    a_R1 = k1 a y^2,  b_R1 = k1' y^3,  a_R2 = k2 y,  b_R2 = k2' b,  transport D (y_i - y_j),
with k1 = 1, k1' = 0.5, k2 = 5, k2' = 1, D = 1.  ln gamma(x) = PHI x / (1 - PHI x) with PHI = 0.25 on the non-ideal
benchmarks and PHI = 0 (ideal, bistable) on schlogl.

Phosphorylation cycle (phosphorylation_cycle): species X, Xp, E on the n x n grid (E immobile and catalytic),
R1  X + E + ATP <-> Xp + E + ADP, driven by dG = log(ATP / ADP) with ATP ADP = 1, a = k1 X E e^{dG/2}, b = k1 Xp E e^{-dG/2};
R2  Xp <-> X (phosphatase), k2 Xp and k2' X;  X and Xp diffuse in a smooth potential U,
J = D (c_i e^{(U_i - U_j)/2} - c_j e^{(U_j - U_i)/2}).  k1 = 2, k2 = 1, k2' = 0.02, D = 1.

Heterogeneous phosphorylation cycle (phosphorylation_heterogeneous): the same network and conditions on a medium, a
hidden periodic map H(x) drawn per medium at 8 pixels per site.  Every hyperedge's symmetric rate constant is multiplied
by a nonlinear function of H, which does not read the conditions (the thermodynamic force is unchanged):
    R1 at site v   mean over the pixels of the cell of exp(1.2 tanh(1.5 H))
    R2 at site v   mean over the pixels of the cell of exp(1 - 2 exp(-H^2))        (non-monotone: inhibited near H = 0)
    transport of X  harmonic mean of exp(0.8 tanh(1.5 H)) over the 8 pixels between the two cell centres
    transport of Xp harmonic mean of exp(-0.6 tanh(1.5 H)) over the same pixels
The models never see H or k.  They see 3 descriptors per hyperedge, from the mean and the standard deviation of H in the
cells (medium_descriptors).

Interacting phosphorylation cycle (phosphorylation_interacting): the same network, every hyperedge mass action in the activity
y = log c + dG_true, with the regular-solution excess free energy per site G_true = CHI_XXP c_X c_Xp + CHI_XPE c_Xp c_E, one
attractive and one repulsive cross term: dG_X = CHI_XXP c_Xp, dG_Xp = CHI_XXP c_X + CHI_XPE c_E, dG_E = CHI_XPE c_Xp.

Saturating phosphorylation cycle (phosphorylation_saturating): the same network with the symmetric rate constant of R1 at site v
multiplied by 1 / (1 + (c_X + c_Xp + c_E) / K_SAT); it does not read the conditions.

MAPK cascade (mapk_cascade): three tiers with the Huang-Ferrell structure on the 4 x 4 grid.  Dual phosphorylation of
MAPKK and MAPK; every enzymatic step is binding E + S <-> ES followed by catalysis ES (+ ATP) <-> E + P (+ ADP or Pi);
mass action throughout; ATP, ADP and Pi are chemostats.  The rate constants are dimensionless example values (the
fastest and the slowest processes are within a factor of about 1e2).  The input kinase E1 sits at 4 corner sites; MAPK
in its three forms diffuses.  22 state species, 20 reactions.
"""
import math

import numpy as np
import torch
import torch.nn.functional as Fn

from neuralchart.graphs import Hypergraph
from . import config as C

# Schloegl
K1, K1R, K2, K2R, DIFF = 1.0, 0.5, 5.0, 1.0, 1.0
KF, KB, DD = (K1, K2), (K1R, K2R), (DIFF,)
PHI = 0.25
SCHLOGL_RX = [({'A': 1, 'X': 2}, {'X': 3}), ({'X': 1}, {'B': 1})]
# phosphorylation cycle
PC_SPECIES = ['X', 'Xp', 'E', 'ATP', 'ADP']
PC_RX = [({'X': 1, 'E': 1, 'ATP': 1}, {'Xp': 1, 'E': 1, 'ADP': 1}), ({'Xp': 1}, {'X': 1})]
PC_K1, PC_K2, PC_K2R, PC_D = 2.0, 1.0, 0.02, 1.0
# MAPK cascade
MAPK_N = 4
STATE = ['KKK', 'KKKp', 'E1', 'E2', 'KK', 'KKp', 'KKpp', 'P2', 'K', 'Kp', 'Kpp', 'P3',
         'ES1', 'ES2', 'ES3', 'ES4', 'ES5', 'ES6', 'ES7', 'ES8', 'ES9', 'ES10']
CHEM = ['ATP', 'ADP', 'Pi']
# (enzyme, substrate, complex, product, kinase?)
STEPS = [('E1', 'KKK', 'ES1', 'KKKp', True), ('E2', 'KKKp', 'ES2', 'KKK', False),
         ('KKKp', 'KK', 'ES3', 'KKp', True), ('KKKp', 'KKp', 'ES4', 'KKpp', True),
         ('P2', 'KKpp', 'ES5', 'KKp', False), ('P2', 'KKp', 'ES6', 'KK', False),
         ('KKpp', 'K', 'ES7', 'Kp', True), ('KKpp', 'Kp', 'ES8', 'Kpp', True),
         ('P3', 'Kpp', 'ES9', 'Kp', False), ('P3', 'Kp', 'ES10', 'K', False)]
DIFFUSING = ['K', 'Kp', 'Kpp']
A_BIND, D_UNBIND, K_KIN, K_PHOS, EPS, D_DIFF = 10.0, 2.0, 2.0, 1.0, 1e-3, 0.5
TOTALS = dict(KKK=1.0, KK=1.0, K=1.0, E2=0.2, P2=0.3, P3=0.3)
INPUT_SITES = [0, 1, 4, 5]                       # the 2 x 2 corner


def phi_of():
    return PHI if C.nonideal() else 0.0


def n_channels():
    """state species per site"""
    return 3 if C.is_pc() else 22 if C.SYSTEM == 'mapk_cascade' else 1


CHI_XXP, CHI_XPE, K_SAT = -0.15, 0.15, 2.0


def pc_excess(c):
    """interacting phosphorylation cycle: the true excess potential at the states c (..., 3 V)"""
    x, xp, e = c.chunk(3, -1)
    return torch.cat([CHI_XXP * xp, CHI_XXP * x + CHI_XPE * e, CHI_XPE * xp], -1)


def pc_excess_G(c):
    x, xp, e = c.chunk(3, -1)
    return (CHI_XXP * x * xp + CHI_XPE * xp * e).sum(-1)


def pc_state_logk(hg, c):
    """saturating phosphorylation cycle: the log factor of the true k per hyperedge at the states c (B, 3 V), (B, H)"""
    typ = torch.as_tensor(hg.meta['typ'], device=c.device)
    x, xp, e = c.chunk(3, -1)
    out = torch.zeros(c.shape[0], hg.H, dtype=c.dtype, device=c.device)
    out[:, typ == 0] = -torch.log1p((x + xp + e) / K_SAT)            # R1 hyperedges are in site order
    return out


# ------------------------------------------------------------------------------------------ Schloegl
def lng_np(x, phi=None):
    phi = phi_of() if phi is None else phi
    return phi * x / (1 - phi * x)


def lng_t(x, phi=None):
    phi = phi_of() if phi is None else phi
    return phi * x / (1 - phi * x)


def x_of_y(y, phi=None):
    """inverse of the activity y = x gamma(x) (bisection; the identity for phi = 0)"""
    phi = phi_of() if phi is None else phi
    y = np.asarray(y, float)
    if phi == 0:
        return y.copy()
    lo, hi = np.zeros_like(y), np.full_like(y, 1 / phi - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        big = mid * np.exp(lng_np(mid, phi)) > y
        hi = np.where(big, mid, hi); lo = np.where(big, lo, mid)
    return 0.5 * (lo + hi)


def roots(a, b):
    """positive steady activities of the well-mixed ideal model"""
    r = np.roots([-K1R, K1 * a, -K2, K2R * b])
    r = np.sort(r[np.abs(r.imag) < 1e-9].real)
    return r[r > 0]


def maxwell_b(a):
    """Maxwell point of the ideal model in the activity y: the middle root at the inflection y_i = 2a/3"""
    xi = K1 * a / (3 * K1R)
    return -(K1 * a * xi ** 2 - K1R * xi ** 3 - K2 * xi) / K2R


def true_cycle_affinity(a, b):
    """affinity of the local cycle R1 + R2 (A -> B)"""
    return math.log(K1 * K2 * a / (K1R * K2R * b))


# ------------------------------------------------------------------------------------------ MAPK network
def mapk_reactions():
    rx, kf, kb = [], [], []
    for e, s_, c, p, kin in STEPS:
        rx.append(({e: 1, s_: 1}, {c: 1})); kf.append(A_BIND); kb.append(D_UNBIND)
        if kin:
            rx.append(({c: 1, 'ATP': 1}, {e: 1, p: 1, 'ADP': 1})); kf.append(K_KIN); kb.append(EPS * K_KIN)
        else:
            rx.append(({c: 1}, {e: 1, p: 1, 'Pi': 1})); kf.append(K_PHOS); kb.append(EPS * K_PHOS)
    return rx, np.array(kf), np.array(kb)


def mapk_hg(n=MAPK_N):
    rx, _, _ = mapk_reactions()
    return Hypergraph.reaction_transport(STATE + CHEM, rx, grid_n=n, diffusing=DIFFUSING, chemostats=CHEM)


def mapk_true_constants(hg):
    """per-hyperedge forward / backward rate constants (chemostats excluded)"""
    _, kf, kb = mapk_reactions()
    R, V = len(kf), hg.V
    KF_ = np.concatenate([np.repeat(kf, V), np.full(hg.H - R * V, D_DIFF)])
    KB_ = np.concatenate([np.repeat(kb, V), np.full(hg.H - R * V, D_DIFF)])
    return KF_, KB_


def mapk_chem_factors(hg, logc):
    """forward / backward chemostat mass-action factors (B, H) from log chemostat concentrations (B, 3)"""
    Y = torch.as_tensor(hg.meta['chem_S'].toarray(), dtype=logc.dtype, device=logc.device)
    lcv = logc.repeat_interleave(hg.V, 1)
    return torch.exp(lcv @ (-Y).clamp_min(0).T), torch.exp(lcv @ Y.clamp_min(0).T)


def mapk_true_rates(hg, logc):
    KF_, KB_ = (torch.as_tensor(x, dtype=logc.dtype, device=logc.device) for x in mapk_true_constants(hg))
    ff, fb = mapk_chem_factors(hg, logc)
    pa, pb = KF_[None] * ff, KB_[None] * fb
    return lambda c: (pa * hg.mass_action(c, '+'), pb * hg.mass_action(c, '-'))


def kpp_fraction(c):
    """fraction of MAPK in the doubly phosphorylated form (free Kpp plus Kpp in the complex ES9), per site: (..., V)"""
    V = MAPK_N * MAPK_N
    sl = lambda nm: c[..., STATE.index(nm) * V:(STATE.index(nm) + 1) * V]
    tot = sl('K') + sl('Kp') + sl('Kpp') + sl('ES7') + sl('ES8') + sl('ES9') + sl('ES10')
    return (sl('Kpp') + sl('ES9')) / tot


def mapk_featurizer(hg, condition_features):
    """drive = dict(logc (B, 3) log concentrations of ATP, ADP, Pi).  condition_features=False (Neural Chart, CRNN with
    mass action): the mobility sees only the hyperedge type; True: also the chemostat force along each hyperedge and
    log ATP."""
    ntyp = len(STEPS) * 2 + len(DIFFUSING)

    def feat(drive):
        lc = drive['logc']
        B, dt, dev = lc.shape[0], lc.dtype, lc.device
        t = hg.on(dev, dt)
        force = hg.chemostat_force(lc)
        cols = [Fn.one_hot(t['typ'], ntyp).to(dt).expand(B, -1, -1)]
        if condition_features:
            cols.append(force.abs()[..., None])
        return dict(U=torch.zeros(B, hg.V, dtype=dt, device=dev), T=torch.ones(B, dtype=dt, device=dev), line=None,
                    chem_logc=lc, chem_mu0=None, edge_sym=torch.cat(cols, -1),
                    edge_ori=force if condition_features else torch.zeros_like(force),
                    force=force, node=dict(logATP=lc[:, :1].expand(B, hg.V)) if condition_features else {}, raw=drive)
    feat.n_edge_sym = ntyp + (1 if condition_features else 0)
    return feat


# ------------------------------------------------------------------------------------------ hypergraphs and rates
_HG = {}


def schlogl_hg(n=8):
    return Hypergraph.reaction_transport(['X', 'A', 'B'], SCHLOGL_RX, grid_n=n, diffusing=['X'], chemostats=['A', 'B'])


def hg_of(n):
    key = ('mapk_cascade' if C.SYSTEM == 'mapk_cascade' else 'phosphorylation_cycle' if C.is_pc() else 'schlogl', n)
    if key not in _HG:
        if C.SYSTEM == 'mapk_cascade':
            _HG[key] = mapk_hg(n)
        elif C.is_pc():
            _HG[key] = Hypergraph.reaction_transport(PC_SPECIES, PC_RX, grid_n=n, diffusing=['X', 'Xp'], chemostats=['ATP', 'ADP'])
        else:
            _HG[key] = schlogl_hg(n)
    return _HG[key]


def true_constants(hg, logab):
    """Schloegl: one-way rate constants (B, H) at chemostats logab (B, 2) = (log a, log b): reaction r forward kf_r
    (times its chemostat reactant), backward kb_r (times its chemostat product), transport D both ways"""
    typ = torch.as_tensor(hg.meta['typ'])
    kF = torch.tensor(list(KF) + list(DD), dtype=torch.float64)[typ]
    kB = torch.tensor(list(KB) + list(DD), dtype=torch.float64)[typ]
    lc = logab.double().repeat_interleave(hg.V, 1)
    Yp = torch.as_tensor(hg.meta['chem_Np'].toarray()); Ym = torch.as_tensor(hg.meta['chem_Nm'].toarray())
    return kF * torch.exp(lc @ Yp.T), kB * torch.exp(lc @ Ym.T)


def pc_log_constants(hg, drive):
    """phosphorylation cycle: log a = la0 + <nu+, log c>, log b = lb0 + <nu-, log c> (B, H) for drive U (B, V), dG (B,)"""
    U, dG = drive['U'].double().cpu(), drive['dG'].double().cpu().reshape(-1)
    Bn, V = U.shape
    typ = torch.as_tensor(hg.meta['typ'])
    lk = torch.tensor([math.log(PC_K1), 0.5 * math.log(PC_K2 * PC_K2R), math.log(PC_D), math.log(PC_D)], dtype=torch.float64)[typ]
    if drive.get('logm') is not None:                          # the medium: log factor of k per hyperedge
        lk = lk + drive['logm'].double().cpu()
    A = torch.zeros(Bn, hg.H, dtype=torch.float64)
    A[:, typ == 0] = dG[:, None]
    A[:, typ == 1] = math.log(PC_K2 / PC_K2R)
    Uf = torch.cat([U, U, torch.zeros_like(U)], 1)
    dU = hg.dot(Uf, '+') - hg.dot(Uf, '-')                     # <nu+ - nu-, U>: split form of the transport
    return lk + 0.5 * dU + 0.5 * A, lk - 0.5 * dU - 0.5 * A


def pc_true_log_k(hg, drive, c):
    """phosphorylation networks: the log of the true symmetric rate constant per hyperedge at the state c (B, H)"""
    typ = torch.as_tensor(hg.meta['typ'])
    lk = torch.tensor([math.log(PC_K1), 0.5 * math.log(PC_K2 * PC_K2R), math.log(PC_D), math.log(PC_D)], dtype=torch.float64)[typ]
    lk = lk[None].expand(c.shape[0], -1)
    if drive.get('logm') is not None:
        lk = lk + drive['logm'].double().cpu()
    if C.saturating():
        lk = lk + pc_state_logk(hg, c)
    return lk


def true_rates(hg, drive, c, ideal=False):
    """one-way rates of the reference model (mass action in the activity), float64 on the CPU; drive: the condition
    dict (or the (log a, log b) tensor of the Schloegl models)"""
    if C.SYSTEM == 'mapk_cascade':
        return mapk_true_rates(hg, drive['logc'].double().cpu())(c.double().cpu())
    if C.is_pc():
        la, lb = pc_log_constants(hg, drive)
        c = c.double().cpu()
        if not ideal and C.interacting():
            y = torch.log(c) + pc_excess(c)
            return torch.exp(la + hg.dot(y, '+')), torch.exp(lb + hg.dot(y, '-'))
        if not ideal and C.saturating():
            dk = pc_state_logk(hg, c)
            return torch.exp(la + dk) * hg.mass_action(c, '+'), torch.exp(lb + dk) * hg.mass_action(c, '-')
        return torch.exp(la) * hg.mass_action(c, '+'), torch.exp(lb) * hg.mass_action(c, '-')
    logab = drive['logab'] if isinstance(drive, dict) else drive
    kf, kb = true_constants(hg, logab.double().cpu())
    c = c.double().cpu()
    y = c * torch.exp(lng_t(c))
    return kf * hg.mass_action(y, '+'), kb * hg.mass_action(y, '-')


def mass_action_rates(hg, logab, c, kf, kb, D):
    """one-way rates of an ideal reaction-transport model: reaction r forward kf_r (times its chemostat reactants),
    backward kb_r (times its chemostat products), transport of the d-th diffusing species D_d both ways"""
    typ = torch.as_tensor(hg.meta['typ'])
    kF = torch.tensor(list(kf) + list(D), dtype=torch.float64)[typ]
    kB = torch.tensor(list(kb) + list(D), dtype=torch.float64)[typ]
    nC = len(hg.meta['chemostats'])
    lc = torch.cat([logab, torch.zeros(logab.shape[0], nC - logab.shape[1], dtype=logab.dtype)], -1).repeat_interleave(hg.V, 1)
    Yp = torch.as_tensor(hg.meta['chem_Np'].toarray()); Ym = torch.as_tensor(hg.meta['chem_Nm'].toarray())
    return (kF * torch.exp(lc @ Yp.T) * hg.mass_action(c, '+'), kB * torch.exp(lc @ Ym.T) * hg.mass_action(c, '-'))


def model_cycle_affinity(la, V):
    """a local cycle at site 0 from log(a / b) (B, H): R1 + R2 (Schloegl, phosphorylation cycle); MAPK: the first-tier
    futile cycle E1 + KKK <-> ES1 -> E1 + KKKp, E2 + KKKp <-> ES2 -> E2 + KKK (ATP -> ADP + Pi)"""
    if C.SYSTEM == 'mapk_cascade':
        return la[:, 0] + la[:, V] + la[:, 2 * V] + la[:, 3 * V]
    return la[:, 0] + la[:, V]


def truth_cycle_affinity(drive):
    if C.is_pc():
        return (drive['dG'].double().cpu().reshape(-1) + math.log(PC_K2 / PC_K2R)).numpy()
    return np.array([true_cycle_affinity(math.exp(x), math.exp(y)) for x, y in drive['logab'].double().cpu().tolist()])


# ------------------------------------------------------------------------------------------ reference solvers
class SchloeglRHS:
    def __init__(s, n, a, b):
        hg = hg_of(n)
        kf, kb = true_constants(hg, torch.tensor([[math.log(a), math.log(b)]]))
        s.kf, s.kb = kf[0].numpy(), kb[0].numpy()
        s.Np, s.Nm = hg.Np.tocsr(), hg.Nm.tocsr()
        s.S = (hg.Nm - hg.Np).T.tocsr()
        s.phi = phi_of()

    def rhs(s, t, x):
        x = np.clip(x, 1e-300, (1 / s.phi - 1e-9) if s.phi > 0 else np.inf)
        L = np.log(x) + lng_np(x, s.phi)
        return s.S @ (s.kf * np.exp(s.Np @ L) - s.kb * np.exp(s.Nm @ L))


def solve_schlogl(n, a, b, c0, nsteps):
    """DOP853 (rtol 1e-10, atol 1e-12) in float64, sampled every DT"""
    from scipy.integrate import solve_ivp
    tr = SchloeglRHS(n, a, b)
    ts = np.arange(nsteps + 1) * C.DT
    sol = solve_ivp(tr.rhs, (0, ts[-1]), c0, method='DOP853', t_eval=ts, rtol=1e-10, atol=1e-12)
    assert sol.success and sol.y.min() > 0, sol.message
    return sol.y.T


PIXELS = 8                                       # pixels of the hidden map per site, heterogeneous media


def medium_map(rng, n, smooth_field):
    """the hidden map H on (n * PIXELS)^2 pixels, standard deviation 1: wavenumbers up to 2 n / 8 (weight 0.7 of the
    variance) and up to 6 n / 8 (0.3), so the same correlation length in sites on every grid"""
    N, s = n * PIXELS, n // 8
    H = np.sqrt(0.7) * smooth_field(rng, N, 2 * s, 1.0) + np.sqrt(0.3) * smooth_field(rng, N, 6 * s, 1.0)
    return (H / H.std()).reshape(N, N)


def _cell_mean(f, n):
    return f.reshape(n, PIXELS, n, PIXELS).mean((1, 3)).ravel()


def _segments(hg, n):
    """pixels (rows, cols) (E, PIXELS) on the straight line between the cell centres of every transport hyperedge"""
    odd = np.asarray(hg.meta['odd'])
    et, eh = np.asarray(hg.meta['et'])[odd], np.asarray(hg.meta['eh'])[odd]
    ri, ci, rj, cj = et // n, et % n, eh // n, eh % n
    wrap = lambda d: (d + n // 2) % n - n // 2
    dr, dc = wrap(rj - ri), wrap(cj - ci)
    t = (np.arange(PIXELS) + 0.5) / PIXELS
    row = np.floor((ri[:, None] + 0.5 + t * dr[:, None]) * PIXELS).astype(int) % (n * PIXELS)
    col = np.floor((ci[:, None] + 0.5 + t * dc[:, None]) * PIXELS).astype(int) % (n * PIXELS)
    return row, col


def medium_log_factors(H, hg, n):
    """log factor of the true k per hyperedge (H,) for the map H (the law in the module docstring)"""
    typ, odd = np.asarray(hg.meta['typ']), np.asarray(hg.meta['odd'])
    V = n * n
    out = np.zeros(hg.H)
    site = np.arange(hg.H) % V
    r1 = _cell_mean(np.exp(1.2 * np.tanh(1.5 * H)), n)
    r2 = _cell_mean(np.exp(1.0 - 2.0 * np.exp(-H ** 2)), n)
    out[typ == 0] = np.log(r1[site[typ == 0]])
    out[typ == 1] = np.log(r2[site[typ == 1]])
    row, col = _segments(hg, n)
    tt = typ[odd]
    gx, gp = np.exp(0.8 * np.tanh(1.5 * H)), np.exp(-0.6 * np.tanh(1.5 * H))
    harm = lambda g: PIXELS / (1.0 / g[row, col]).sum(1)
    out[odd] = np.log(np.where(tt == 2, harm(gx), harm(gp)))
    return out


def medium_descriptors(H, hg, n):
    """what the models see of the medium.  Per site (V, 3): mean of H in the cell m, its standard deviation s in the
    cell, the mean of m over the 4 neighbours.  Per hyperedge (H, 3): the site values on a reaction; on a transport
    hyperedge the mean of m at the two ends, the mean of s, and |m_i - m_j| (orientation invariant)."""
    typ, odd = np.asarray(hg.meta['typ']), np.asarray(hg.meta['odd'])
    V = n * n
    mean = _cell_mean(H, n)
    std = np.sqrt(np.maximum(_cell_mean(H ** 2, n) - mean ** 2, 0.0))
    g = mean.reshape(n, n)
    nb = 0.25 * (np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)).ravel()
    site = np.stack([mean, std, nb], 1)
    edge = np.zeros((hg.H, 3))
    idx = np.arange(hg.H) % V
    edge[~odd] = site[idx[~odd]]
    et, eh = np.asarray(hg.meta['et'])[odd], np.asarray(hg.meta['eh'])[odd]
    edge[odd] = np.stack([0.5 * (mean[et] + mean[eh]), 0.5 * (std[et] + std[eh]), np.abs(mean[et] - mean[eh])], 1)
    return site, edge


def solve_pc(n, U, dG, c0, nsteps, logm=None):
    from scipy.integrate import solve_ivp
    hg = hg_of(n)
    la, lb = pc_log_constants(hg, dict(U=torch.as_tensor(U)[None], dG=torch.tensor([dG]),
                                       logm=None if logm is None else torch.as_tensor(logm)[None]))
    ka, kb = np.exp(la[0].numpy()), np.exp(lb[0].numpy())
    Np, Nm = hg.Np.tocsr(), hg.Nm.tocsr()
    S = (hg.Nm - hg.Np).T.tocsr()

    r1 = np.asarray(hg.meta['typ']) == 0

    def rhs(t, x):
        L = np.log(np.maximum(x, 1e-300))
        fac = 1.0
        if C.interacting():
            L = L + pc_excess(torch.as_tensor(x)).numpy()
        elif C.saturating():
            xv, xp, e = np.split(x, 3)
            fac = np.ones(hg.H)
            fac[r1] = 1.0 / (1.0 + (xv + xp + e) / K_SAT)
        return S @ (fac * (ka * np.exp(Np @ L) - kb * np.exp(Nm @ L)))
    ts = np.arange(nsteps + 1) * C.DT
    sol = solve_ivp(rhs, (0, ts[-1]), c0, method='DOP853', t_eval=ts, rtol=1e-10, atol=1e-12)
    assert sol.success and sol.y.min() > 0, sol.message
    return sol.y.T


def solve_mapk(hg, logc, x0, nsteps):
    """LSODA (rtol 1e-8, atol 1e-11) with the exact Jacobian; also returns sigma and the ATP turnover"""
    from scipy.integrate import solve_ivp
    rates = mapk_true_rates(hg, torch.tensor(logc[None], dtype=torch.float64))
    Sd = hg.S

    def f(t, x):
        a, b = rates(torch.tensor(np.maximum(x, 1e-300)[None]))
        return Sd @ (a - b)[0].numpy()

    St = torch.as_tensor(Sd)

    def rhs_t(y):
        a, b = rates(y)
        return St @ (a - b)[0]

    def jac(t, x):
        xt = torch.tensor(np.maximum(x, 1e-300)[None])
        return torch.autograd.functional.jacobian(rhs_t, xt)[:, 0].numpy()
    ts = np.arange(nsteps + 1) * C.DT
    sol = solve_ivp(f, (0, ts[-1]), x0, method='LSODA', t_eval=ts, rtol=1e-8, atol=1e-11, jac=jac)
    assert sol.success, sol.message
    Cs = sol.y.T
    a, b = rates(torch.tensor(Cs))
    a, b = a.numpy(), b.numpy()
    sig = ((a - b) * (np.log(a) - np.log(b))).sum(-1)
    kin = (hg.meta['chem_S'].toarray()[:, :hg.V] < 0).any(1)         # hyperedges that consume ATP
    turnover = ((a - b)[:, kin]).sum(-1)
    return Cs, sig, turnover
