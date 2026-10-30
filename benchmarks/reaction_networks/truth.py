"""The reaction networks and their reference dynamics.

Schloegl model (schlogl, schlogl_nonideal): A + 2X <-> 3X, X <-> B at every site of the n x n periodic grid, X diffuses,
A and B are chemostats.  Every hyperedge is mass action in the activity y = x gamma(x),
    a_R1 = k1 a y^2,  b_R1 = k1' y^3,  a_R2 = k2 y,  b_R2 = k2' b,  transport D (y_i - y_j),
with k1 = 1, k1' = 0.5, k2 = 5, k2' = 1, D = 1.  ln gamma(x) = PHI x / (1 - PHI x) with PHI = 0.25 on the non-ideal
benchmark and PHI = 0 (ideal, bistable) on schlogl.

Phosphorylation cycle (phosphorylation_cycle): species X, Xp, E on the n x n grid (E immobile and catalytic),
R1  X + E + ATP <-> Xp + E + ADP, driven by dG = log(ATP / ADP) with ATP ADP = 1, a = k1 X E e^{dG/2}, b = k1 Xp E e^{-dG/2};
R2  Xp <-> X (phosphatase), k2 Xp and k2' X;  X and Xp diffuse in a smooth potential U,
J = D (c_i e^{(U_i - U_j)/2} - c_j e^{(U_j - U_i)/2}).  k1 = 2, k2 = 1, k2' = 0.02, D = 1.

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
    return PHI if C.SYSTEM == 'schlogl_nonideal' else 0.0


def n_channels():
    """state species per site"""
    return dict(schlogl_nonideal=1, schlogl=1, phosphorylation_cycle=3, mapk_cascade=22)[C.SYSTEM]


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
    key = (C.SYSTEM if C.SYSTEM in ('phosphorylation_cycle', 'mapk_cascade') else 'schlogl', n)
    if key not in _HG:
        if C.SYSTEM == 'mapk_cascade':
            _HG[key] = mapk_hg(n)
        elif C.SYSTEM == 'phosphorylation_cycle':
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
    A = torch.zeros(Bn, hg.H, dtype=torch.float64)
    A[:, typ == 0] = dG[:, None]
    A[:, typ == 1] = math.log(PC_K2 / PC_K2R)
    Uf = torch.cat([U, U, torch.zeros_like(U)], 1)
    dU = hg.dot(Uf, '+') - hg.dot(Uf, '-')                     # <nu+ - nu-, U>: split form of the transport
    return lk + 0.5 * dU + 0.5 * A, lk - 0.5 * dU - 0.5 * A


def true_rates(hg, drive, c):
    """one-way rates of the reference model (mass action in the activity), float64 on the CPU; drive: the condition
    dict (or the (log a, log b) tensor of the Schloegl models)"""
    if C.SYSTEM == 'mapk_cascade':
        return mapk_true_rates(hg, drive['logc'].double().cpu())(c.double().cpu())
    if C.SYSTEM == 'phosphorylation_cycle':
        la, lb = pc_log_constants(hg, drive)
        c = c.double().cpu()
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
    if C.SYSTEM == 'phosphorylation_cycle':
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


def solve_pc(n, U, dG, c0, nsteps):
    from scipy.integrate import solve_ivp
    hg = hg_of(n)
    la, lb = pc_log_constants(hg, dict(U=torch.as_tensor(U)[None], dG=torch.tensor([dG])))
    ka, kb = np.exp(la[0].numpy()), np.exp(lb[0].numpy())
    Np, Nm = hg.Np.tocsr(), hg.Nm.tocsr()
    S = (hg.Nm - hg.Np).T.tocsr()

    def rhs(t, x):
        L = np.log(np.maximum(x, 1e-300))
        return S @ (ka * np.exp(Np @ L) - kb * np.exp(Nm @ L))
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
