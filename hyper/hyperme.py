"""HyperME: thermodynamically consistent master-equation operator on a hypergraph (CRN on a periodic grid).

State c in R^{S x V}_{>0}, S=3 species (X, Xp, E), V=n*n sites (periodic grid), flattened as idx = s*V + v.
Hyperedges h (forward direction nu+ -> nu-), ordering:
  [R1_v : X+E -> Xp+E]  (V)   [R2_v : Xp -> X] (V)   [TX_e : X@i -> X@j] (2V)   [TXp_e : Xp@i -> Xp@j] (2V)
Rates:  a_h = k_h exp(<nu+_h, mu> + A_h/2),  b_h = k_h exp(<nu-_h, mu> - A_h/2),  J = a - b,  dc/dt = S J,  S = nu- - nu+.
mu = log c + alpha_s U + dG_theta/dc (G_theta zero-init scalar per site => thermodynamic potential F_theta),
k_h = softplus(MLP(hyperedge feats)) symmetric (same for both directions),
A = P_kerS A_raw,   P_kerS = Z Z^T (Z = scipy null_space(S)) = I - S^T (S S^T)^+ S.
Integrator: SSP-RK2 with adaptive substeps s.t. dt * max_{s,v} (sum_h [nu+ - nu-]_+_{s,v} a_h + [nu- - nu+]_+_{s,v} b_h) / c_{s,v} <= 0.5
for both stages (net consumption; every consuming term carries a factor c_{s,v}, so each Euler stage keeps
c >= c/2 > 0; no clamping; catalyst E is excluded because its net stoichiometry is 0).
python hyperme.py --selftest   runs the property assertions."""
import math, os, sys
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as Fn

NS = 3                    # species X, Xp, E
SP_X, SP_XP, SP_E = 0, 1, 2
DT = 0.05                 # sampling interval
T_TRAIN = 20
T_TEST = 200
NSUB_NOM = 4              # nominal substeps per DT for adaptive HyperME (dt <= DT/4)
NSUB_FIX = 8              # fixed Heun substeps per DT for baselines
MAXSUB = 2000
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
ROOT = os.environ.get('FH_ROOT') or os.path.expanduser('~/Fisher-Hodge')
HROOT = os.path.join(ROOT, 'hyper')
RUNS = os.path.join(HROOT, 'runs')


# ------------------------------------------------------------------------------------------ hypergraph
_GCACHE = {}


def build_graph_np(n):
    """numpy hypergraph for an n x n periodic grid."""
    V = n * n
    idx = np.arange(V).reshape(n, n)
    tail = np.concatenate([idx.ravel(), idx.ravel()])
    head = np.concatenate([np.roll(idx, -1, 1).ravel(), np.roll(idx, -1, 0).ravel()])
    ish = np.concatenate([np.ones(V), np.zeros(V)])           # horizontal edge indicator
    E2 = 2 * V
    H = 2 * V + 2 * E2
    Np = np.zeros((H, NS * V)); Nm = np.zeros((H, NS * V))
    typ = np.zeros(H, dtype=int); site = np.full(H, -1); et = np.full(H, -1); eh = np.full(H, -1)
    v = np.arange(V)
    r1 = np.arange(V); r2 = V + np.arange(V); tx = 2 * V + np.arange(E2); txp = 2 * V + E2 + np.arange(E2)
    Np[r1, SP_X * V + v] = 1; Np[r1, SP_E * V + v] = 1; Nm[r1, SP_XP * V + v] = 1; Nm[r1, SP_E * V + v] = 1
    Np[r2, SP_XP * V + v] = 1; Nm[r2, SP_X * V + v] = 1
    Np[tx, SP_X * V + tail] = 1; Nm[tx, SP_X * V + head] = 1
    Np[txp, SP_XP * V + tail] = 1; Nm[txp, SP_XP * V + head] = 1
    typ[r1], typ[r2], typ[tx], typ[txp] = 0, 1, 2, 3
    site[r1] = v; site[r2] = v
    et[tx] = tail; eh[tx] = head; et[txp] = tail; eh[txp] = head
    S = (Nm - Np).T                                            # (NS*V, H)
    Z = scipy.linalg.null_space(S)                             # basis of ker S  (cycle space)
    L = scipy.linalg.null_space(S.T)                           # basis of ker S^T (conservation laws)
    P = Z @ Z.T
    # check against the pseudo-inverse form  I - S^T (S S^T)^+ S
    P2 = np.eye(H) - S.T @ np.linalg.pinv(S @ S.T, rcond=1e-10, hermitian=True) @ S
    assert np.abs(P - P2).max() < 1e-8, 'projector mismatch'
    zloc = np.zeros((V, H)); zloc[v, r1] = 1; zloc[v, r2] = 1   # local R1+R2 cycles (in ker S)
    assert np.abs(S @ zloc.T).max() == 0
    # reactant index pairs (for Gillespie / mass action), pad with index NS*V (=constant 1 slot)
    def pairs(N):
        out = np.full((H, 2), NS * V)
        for h in range(H):
            nz = np.nonzero(N[h])[0]
            out[h, :len(nz)] = nz
        return out
    return dict(n=n, V=V, H=H, E2=E2, Np=Np, Nm=Nm, S=S, Z=Z, L=L, P=P, typ=typ, site=site, et=et, eh=eh,
                ish=np.concatenate([np.zeros(2 * V), ish, ish]), zloc=zloc, rp=pairs(Np), rm=pairs(Nm),
                r1=r1, r2=r2, tx=tx, txp=txp)


def graph(n, dev):
    key = (n, str(dev))
    if key not in _GCACHE:
        g = build_graph_np(n)
        gt = dict(g)
        for k in ['Np', 'Nm', 'P', 'zloc', 'L', 'ish']:
            gt[k] = torch.tensor(g[k], device=dev)
        gt['Smat'] = torch.tensor(g['Nm'] - g['Np'], device=dev)    # (H, NS*V): dc = J @ Smat
        gt['Cp'] = torch.tensor(np.maximum(g['Np'] - g['Nm'], 0), device=dev)   # net reactants of the forward reaction
        gt['Cm'] = torch.tensor(np.maximum(g['Nm'] - g['Np'], 0), device=dev)   # net reactants of the backward reaction
        for k in ['typ', 'site', 'et', 'eh']:
            gt[k] = torch.tensor(g[k], device=dev)
        _GCACHE[key] = gt
    return _GCACHE[key]


def edge_feats(U, dG, g):
    """U (B,V), dG (B,) -> symmetric hyperedge feats (B,H,7) and the oriented scalar dU (B,H) (0 for reactions)."""
    B, V = U.shape
    H = g['H']
    oh = Fn.one_hot(g['typ'], 4).to(U.dtype).expand(B, H, 4)
    Ur = U  # reactions at site v
    Ut, Uh = U[:, g['et'][2 * V:]], U[:, g['eh'][2 * V:]]
    Um = torch.cat([Ur, Ur, 0.5 * (Ut + Uh)], 1)
    dU = torch.cat([torch.zeros_like(Ur), torch.zeros_like(Ur), Uh - Ut], 1)
    dg = (dG[:, None] - 2.0).expand(B, H)
    sym = torch.cat([oh, Um[..., None], dU.abs()[..., None], dg[..., None]], -1)
    return sym, dU


class MLP(nn.Module):
    def __init__(s, i, h, o, zero_last=False, bias_last=0.0):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))
        nn.init.constant_(s.net[-1].bias, bias_last)
        if zero_last:
            nn.init.zeros_(s.net[-1].weight); nn.init.zeros_(s.net[-1].bias)

    def forward(s, x):
        return s.net(x)


def heun_fixed(model, c, ctx, Dt, n=NSUB_FIX):
    dt = Dt / n
    for _ in range(n):
        k1 = model.rhs(c, ctx)
        k2 = model.rhs(c + dt * k1, ctx)
        c = c + 0.5 * dt * (k1 + k2)
    return c, n


# ------------------------------------------------------------------------------------------ HyperME
class HyperME(nn.Module):
    name = 'hyperme'

    def __init__(s, use_A=True, H=64):
        super().__init__()
        s.use_A = use_A
        s.kmlp = MLP(7, H, 1, bias_last=0.5)
        s.amlp = MLP(8, H, 1, zero_last=True)      # oriented feats: sym + dU, antisymmetrised for transport
        s.G = MLP(5, H, 1, zero_last=True)         # free-energy correction density G_theta(c_v, U_v, dG)
        s.alpha = nn.Parameter(torch.ones(NS))
        s.force_g0 = False                         # for the property test (g = 0)

    def prepare(s, U, dG, n):
        g = graph(n, U.device)
        sym, dU = edge_feats(U, dG, g)
        k = Fn.softplus(s.kmlp(sym).squeeze(-1)) + 1e-6
        if s.use_A:
            ap = s.amlp(torch.cat([sym, dU[..., None]], -1)).squeeze(-1)
            am = s.amlp(torch.cat([sym, -dU[..., None]], -1)).squeeze(-1)
            istr = (g['typ'] >= 2).to(U.dtype)
            Araw = istr * (ap - am) + (1 - istr) * ap     # transport: odd under orientation flip
            A = Araw @ g['P']                              # project onto ker S (cycle space)
        else:
            A = torch.zeros_like(k)
        return dict(g=g, k=k, A=A, U=U, dG=dG, n=n)

    def Gdens(s, c, ctx):
        """c (B,NS,V) -> per-site G (B,V)"""
        U, dG = ctx['U'], ctx['dG']
        x = torch.cat([c.transpose(1, 2), U[..., None], (dG[:, None, None] - 2.0).expand(-1, U.shape[1], 1)], -1)
        return s.G(x).squeeze(-1)

    def gcorr(s, c, ctx):
        if s.force_g0:
            return torch.zeros_like(c)
        if torch.is_grad_enabled():
            x = c if c.requires_grad else c.detach().requires_grad_(True)
            return torch.autograd.grad(s.Gdens(x, ctx).sum(), x, create_graph=True)[0]
        with torch.enable_grad():
            x = c.detach().requires_grad_(True)
            return torch.autograd.grad(s.Gdens(x, ctx).sum(), x)[0].detach()

    def mu(s, c, ctx):
        return torch.log(c) + s.alpha[None, :, None] * ctx['U'][:, None, :] + s.gcorr(c, ctx)

    def ab(s, c, ctx):
        g = ctx['g']
        m = s.mu(c, ctx).flatten(1)
        mp, mm = m @ g['Np'].T, m @ g['Nm'].T
        a = ctx['k'] * torch.exp(mp + 0.5 * ctx['A'])
        b = ctx['k'] * torch.exp(mm - 0.5 * ctx['A'])
        return a, b

    def flux(s, c, ctx):
        a, b = s.ab(c, ctx)
        return a - b

    def rhs_from_ab(s, a, b, ctx):
        return ((a - b) @ ctx['g']['Smat']).view(a.shape[0], NS, -1)

    def rhs(s, c, ctx):
        a, b = s.ab(c, ctx)
        return s.rhs_from_ab(a, b, ctx)

    def out_ratio(s, a, b, c, ctx):
        g = ctx['g']
        Dout = (a @ g['Cp'] + b @ g['Cm']).view_as(c)     # net consumption (catalysts excluded)
        return float((Dout / c).max())

    def epr(s, c, ctx):
        a, b = s.ab(c, ctx)
        return ((a - b) * (torch.log(a) - torch.log(b))).sum(-1)

    def lyapunov_F(s, c, ctx):
        """F = sum c (log c - 1 + alpha U) + G_theta  (equals sum c(log c -1 + U) at init)"""
        F = (c * (torch.log(c) - 1 + s.alpha[None, :, None] * ctx['U'][:, None, :])).sum((1, 2))
        if not s.force_g0:
            F = F + s.Gdens(c, ctx).sum(-1)
        return F

    def advance(s, c, ctx, Dt):
        t, nsub, dt_nom = 0.0, 0, Dt / NSUB_NOM
        while Dt - t > 1e-12:
            a0, b0 = s.ab(c, ctx)
            f0 = s.rhs_from_ab(a0, b0, ctx)
            with torch.no_grad():
                r0 = s.out_ratio(a0, b0, c, ctx)
            dt = min(dt_nom, Dt - t, 0.5 / r0 if r0 > 0 else float('inf'))
            while True:
                c1 = c + dt * f0
                a1, b1 = s.ab(c1, ctx)
                with torch.no_grad():
                    r1 = s.out_ratio(a1, b1, c1, ctx)
                if dt * r1 <= 0.5 or dt < 1e-10:
                    break
                dt *= 0.5
            c2 = c1 + dt * s.rhs_from_ab(a1, b1, ctx)
            c = 0.5 * (c + c2)                       # SSP-RK2: convex combination of positive Euler stages
            t += dt; nsub += 1
            if nsub >= MAXSUB:
                break
        return c, nsub

    def cycle_affinity_local(s, ctx):
        """<z_v, A> for the local R1+R2 cycle at every site: (B,V)"""
        return ctx['A'] @ ctx['g']['zloc'].T

    def A_cycle(s, ctx):
        return ctx['A']


# ------------------------------------------------------------------------------------------ property tests
def check_conservation(c_traj, g, tol=1e-6):
    """c_traj (B,T,NS,V). Returns max relative error of all ker S^T invariants + E_tot, (X+Xp)_tot."""
    L = g['L'] if isinstance(g['L'], np.ndarray) else g['L'].cpu().numpy()
    X = c_traj.reshape(*c_traj.shape[:2], -1)
    q = X @ L
    e_ker = np.linalg.norm(q - q[:, :1], axis=-1) / np.linalg.norm(q[:, :1], axis=-1)
    Et = c_traj[:, :, SP_E].sum(-1); Xt = c_traj[:, :, [SP_X, SP_XP]].sum((-1, -2))
    e_E = np.abs(Et - Et[:, :1]) / Et[:, :1]; e_X = np.abs(Xt - Xt[:, :1]) / Xt[:, :1]
    return dict(ker=float(np.nanmax(e_ker)), E=float(np.nanmax(e_E)), XXp=float(np.nanmax(e_X)))


def selftest(dev="cuda" if torch.cuda.is_available() else "cpu"):
    torch.manual_seed(0)
    n = 8; V = n * n
    g = graph(n, dev)
    print(f'graph n={n}: H={g["H"]}, dim ker S = {g["Z"].shape[1]}, dim ker S^T = {g["L"].shape[1]}')
    assert g['L'].shape[1] == V + 1, 'expected E per site + total X+Xp'
    B = 4
    U = torch.randn(B, V, device=dev) * 0.7
    dG = torch.rand(B, device=dev) * 2 + 1
    c0 = torch.rand(B, NS, V, device=dev) * 1.5 + 0.05
    res = {}
    for label, useA, g0, perturb in [('A=0,g=0', False, True, False), ('A!=0,g!=0 (random)', True, False, True)]:
        m = HyperME(use_A=useA).to(dev)
        m.force_g0 = g0
        if perturb:   # random non-trivial A and g to stress positivity/conservation
            with torch.no_grad():
                for p in list(m.amlp.net[-1].parameters()) + list(m.G.net[-1].parameters()):
                    p.normal_(0, 0.3)
        with torch.no_grad():
            ctx = m.prepare(U, dG, n)
            if useA:
                SA = (ctx['A'] @ g['Smat']).abs().max().item()
                print(f'  [{label}] |S A|_max = {SA:.2e} (A in ker S), |A|_max = {ctx["A"].abs().max().item():.2f}')
                assert SA < 1e-10
            c = c0.clone(); traj = [c.cpu().numpy()]; F = [m.lyapunov_F(c, ctx).cpu().numpy()]; sig = []
            nsub = []
            for _ in range(200):
                sig.append(m.epr(c, ctx).cpu().numpy())
                a, b = m.ab(c, ctx)
                Jf = (a - b) * (torch.log(a) - torch.log(b))
                assert Jf.min().item() >= 0, 'negative edge EPR'
                c, ns = m.advance(c, ctx, DT); nsub.append(ns)
                assert c.min().item() > 0, 'positivity violated'
                traj.append(c.cpu().numpy()); F.append(m.lyapunov_F(c, ctx).cpu().numpy())
        traj = np.stack(traj, 1); F = np.stack(F, 1); sig = np.stack(sig, 1)
        ce = check_conservation(traj, g)
        dF = np.diff(F, axis=1)
        print(f'  [{label}] conservation rel err: kerS^T {ce["ker"]:.2e}, E {ce["E"]:.2e}, X+Xp {ce["XXp"]:.2e}; '
              f'min c {traj.min():.3e}; sigma min {sig.min():.3e}; max dF {dF.max():.3e}; substeps/DT {np.mean(nsub):.1f}')
        assert ce['ker'] < 1e-6 and ce['E'] < 1e-6 and ce['XXp'] < 1e-6
        assert sig.min() >= 0
        if not useA:
            assert (dF <= 1e-10 * (1 + np.abs(F[:, :-1]))).all(), 'F not monotone for A=0,g=0'
        res[label] = dict(conservation=ce, min_c=float(traj.min()), sigma_min=float(sig.min()), max_dF=float(dF.max()))
    print('SELFTEST PASSED')
    return res


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        import json
        r = selftest()
        os.makedirs(RUNS, exist_ok=True)
        json.dump(r, open(os.path.join(RUNS, 'selftest.json'), 'w'), indent=1)
