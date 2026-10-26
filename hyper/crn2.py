"""Generic reaction-transport hypergraph + HyperME / baselines for benchmarks C2 (Schlogl) and C3 (Brusselator).

Internal (non-chemostatted) species s = 0..NS-1 live on an n x n periodic grid; state c (B, NS, V), flat idx = s*V + v.
Hyperedges: [reaction r at every site v, r = 0..nR-1] then [transport of mobile species j on the 2V grid edges].
Chemostatted species (A, B, D, E) are absorbed into the per-edge rate constants. The models see them only through
the per-trajectory drive vector drv = normalised (log a, log b).
HyperME: a_h = k_h exp(<nu+_h, mu> + A_h/2), b_h = k_h exp(<nu-_h, mu> - A_h/2), dc/dt = S (a - b),
         mu = log c + dG_theta/dc, k_h = softplus(MLP) (same for both directions), A = P_kerS A_raw.
Same adaptive positive SSP-RK2 integrator as hyperme.py (net-consumption ratio <= 0.5 per Euler stage).
B1 CRNN: independent kf, kb (mass action, fixed-step Heun). B2 GNN-ODE: encode-process-decode message passing.
B3: HyperME with A = 0."""
import os
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
ROOT = os.environ.get('FH_ROOT') or os.path.expanduser('~/Fisher-Hodge')
HROOT = os.path.join(ROOT, 'hyper')
RUNS = os.path.join(HROOT, 'runs')
NSUB_NOM = 4
NSUB_FIX = 8
MAXSUB = 2000

SPECS = {
    # Schlogl: R1  A + 2X <-> 3X ,  R2  X <-> B           (A, B chemostatted)
    'schlogl': dict(species=['X'], mobile=[0], rx=[([2], [3]), ([1], [0])], cycles=[[1, 1]], ext=[([1, 0], [0, 0]), ([0, 0], [0, 1])],
                    rx_names=['R1 A+2X<->3X', 'R2 X<->B'], cyc_names=['R1+R2 (A->B)']),
    # Brusselator: R1 A -> X, R2 2X+Y -> 3X, R3 B+X -> Y+D, R4 X -> E   (A, B, D, E chemostatted, tiny reverse rates)
    'brus': dict(species=['X', 'Y'], mobile=[0, 1],
                 rx=[([0, 0], [1, 0]), ([2, 1], [3, 0]), ([1, 0], [0, 1]), ([1, 0], [0, 0])],
                 cycles=[[1, 0, 0, 1], [0, 1, 1, 0]],
                 ext=[([1, 0], [0, 0]), ([0, 0], [0, 0]), ([0, 1], [0, 0]), ([0, 0], [0, 0])],
                 rx_names=['R1 A->X', 'R2 2X+Y->3X', 'R3 B+X->Y+D', 'R4 X->E'],
                 cyc_names=['R1+R4 (A->E)', 'R2+R3 (B->D)']),
}
SPECS["brus_eq"] = SPECS["brus"]
SPECS["schlogl_ni"] = SPECS["schlogl"]          # C5: non-ideal (crowding) Schlogl, same hypergraph


def build_graph_np(bench, n):
    sp = SPECS[bench]
    NS, nR, mob = len(sp['species']), len(sp['rx']), sp['mobile']
    V = n * n
    idx = np.arange(V).reshape(n, n)
    tail = np.concatenate([idx.ravel(), idx.ravel()])
    head = np.concatenate([np.roll(idx, -1, 1).ravel(), np.roll(idx, -1, 0).ravel()])
    E2 = 2 * V
    H = nR * V + len(mob) * E2
    Np = np.zeros((H, NS * V)); Nm = np.zeros((H, NS * V))
    Ep = np.zeros((H, 2)); Em = np.zeros((H, 2))            # chemostat (A, B) stoichiometry of each hyperedge
    typ = np.zeros(H, dtype=int); site = np.full(H, -1); et = np.full(H, -1); eh = np.full(H, -1)
    v = np.arange(V)
    for r, (nup, num) in enumerate(sp['rx']):
        hs = r * V + v
        for s in range(NS):
            Np[hs, s * V + v] = nup[s]; Nm[hs, s * V + v] = num[s]
        typ[hs] = r; site[hs] = v
        Ep[hs] = sp["ext"][r][0]; Em[hs] = sp["ext"][r][1]
    for j, s in enumerate(mob):
        hs = nR * V + j * E2 + np.arange(E2)
        Np[hs, s * V + tail] = 1; Nm[hs, s * V + head] = 1
        typ[hs] = nR + j; et[hs] = tail; eh[hs] = head
    S = (Nm - Np).T
    Z = scipy.linalg.null_space(S)
    L = scipy.linalg.null_space(S.T)
    P = Z @ Z.T
    P2 = np.eye(H) - S.T @ np.linalg.pinv(S @ S.T, rcond=1e-10, hermitian=True) @ S
    assert np.abs(P - P2).max() < 1e-8, 'projector mismatch'
    cyc = np.array(sp['cycles'], dtype=float)                       # (ncyc, nR) integer local cycles
    zloc = np.zeros((len(cyc), V, H))
    for ci, z in enumerate(cyc):
        for r in range(nR):
            zloc[ci, v, r * V + v] = z[r]
    assert np.abs(np.einsum('cvh,sh->cvs', zloc, S)).max() == 0, 'local cycles not in ker S'

    def plist(N):                                                   # reactant indices with multiplicity, pad = NS*V
        out = np.full((H, 3), NS * V)
        for h in range(H):
            l = []
            for i in np.nonzero(N[h])[0]:
                l += [i] * int(round(N[h, i]))
            out[h, :len(l)] = l
        return out
    return dict(n=n, V=V, H=H, E2=E2, NS=NS, nR=nR, ntyp=nR + len(mob), Np=Np, Nm=Nm, S=S, Z=Z, L=L, P=P, typ=typ,
                site=site, et=et, eh=eh, zloc=zloc, Ep=Ep, Em=Em, rp=plist(Np), rm=plist(Nm))


_GC = {}


def graph(bench, n, dev):
    key = (bench.replace("_eq", "").replace("_ni", ""), n, str(dev))
    if key not in _GC:
        g = build_graph_np(bench, n)
        gt = dict(g)
        for k in ["Np", "Nm", "P", "zloc", "Ep", "Em"]:
            gt[k] = torch.tensor(g[k], device=dev)
        gt['Smat'] = torch.tensor(g['Nm'] - g['Np'], device=dev)
        gt['Cp'] = torch.tensor(np.maximum(g['Np'] - g['Nm'], 0), device=dev)
        gt['Cm'] = torch.tensor(np.maximum(g['Nm'] - g['Np'], 0), device=dev)
        gt['isrx'] = torch.tensor((g['typ'] < g['nR']).astype(float), device=dev)
        for k in ['typ', 'site', 'et', 'eh', 'rp', 'rm']:
            gt[k] = torch.tensor(g[k], device=dev)
        _GC[key] = gt
    return _GC[key]


def edge_feats(drv, g):
    B, H = drv.shape[0], g['H']
    oh = Fn.one_hot(g['typ'], g['ntyp']).to(drv.dtype).expand(B, H, g['ntyp'])
    return torch.cat([oh, drv[:, None, :].expand(B, H, drv.shape[1])], -1)


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


class HyperME2(nn.Module):
    def __init__(s, bench, nd=2, use_A=True, Hd=64, chem=False, convex=False, J=16):
        """chem=True: the chemostat chemical potentials mu_A = log a, mu_B = log b enter the rate exponents through
        the chemostat stoichiometry (a_h = k exp(<nu+, mu> + <nu+_ext, mu_ext> + A/2)); k, A_raw and G get NO drive
        features (drv must then be the raw (log a, log b)).
        convex=True: separable convex free-energy correction, mu_s = log c_s + g_s(c_s) with g_s non-decreasing
        (g_s(c) = sum_j e^{alpha_sj} tanh(e^{beta_sj} c + delta_sj) + const_s), i.e. dmu/dc > 0: a single stable phase,
        so multistability can only come from the driving force A (not from a non-convex G)."""
        super().__init__()
        sp = SPECS[bench]
        s.bench, s.use_A, s.NS, s.chem, s.convex = bench, use_A, len(sp['species']), chem, convex
        nt = len(sp['rx']) + len(sp['mobile'])
        nf = 0 if chem else nd
        s.kmlp = MLP(nt + nf, Hd, 1, bias_last=0.5)
        s.amlp = MLP(nt + nf, Hd, 1, zero_last=True)
        s.G = MLP(s.NS + nf, Hd, 1, zero_last=True)
        if convex:
            s.g_a = nn.Parameter(torch.full((s.NS, J), -3.0))
            s.g_b = nn.Parameter(torch.zeros(s.NS, J))
            s.g_d = nn.Parameter(torch.linspace(-6.0, 1.0, J).repeat(s.NS, 1))
            s.g_c = nn.Parameter(torch.zeros(s.NS))

    def gmono(s, c):                            # c (B, NS, V) -> non-decreasing g_s(c_s)
        z = torch.tanh(torch.exp(s.g_b)[None, :, None, :] * c[..., None] + s.g_d[None, :, None, :])
        return (torch.exp(s.g_a)[None, :, None, :] * z).sum(-1) + s.g_c[None, :, None]

    def feat(s, drv):
        return drv[:, :0] if s.chem else drv

    def prepare(s, drv, n):
        g = graph(s.bench, n, drv.device)
        f = edge_feats(s.feat(drv), g)
        k = Fn.softplus(s.kmlp(f).squeeze(-1)) + 1e-6
        if s.use_A:
            Araw = s.amlp(f).squeeze(-1) * g['isrx']   # transport: no oriented feature (no potential) -> odd part = 0
            A = Araw @ g['P']
        else:
            A = torch.zeros_like(k)
        if s.chem:
            ep, em = drv @ g['Ep'].T, drv @ g['Em'].T
        else:
            ep = em = torch.zeros_like(k)
        return dict(g=g, k=k, A=A, drv=drv, n=n, ep=ep, em=em)

    def Gdens(s, c, ctx):
        V = c.shape[-1]
        d = s.feat(ctx['drv'])
        x = torch.cat([c.transpose(1, 2), d[:, None, :].expand(-1, V, -1)], -1)
        return s.G(x).squeeze(-1)

    def gcorr(s, c, ctx):
        if s.convex:
            return s.gmono(c)
        if torch.is_grad_enabled():
            x = c if c.requires_grad else c.detach().requires_grad_(True)
            return torch.autograd.grad(s.Gdens(x, ctx).sum(), x, create_graph=True)[0]
        with torch.enable_grad():
            x = c.detach().requires_grad_(True)
            return torch.autograd.grad(s.Gdens(x, ctx).sum(), x)[0].detach()

    def mu(s, c, ctx):
        return torch.log(c) + s.gcorr(c, ctx)

    def ab(s, c, ctx):
        g = ctx['g']
        m = s.mu(c, ctx).flatten(1)
        a = ctx['k'] * torch.exp(m @ g['Np'].T + ctx['ep'] + 0.5 * ctx['A'])
        b = ctx['k'] * torch.exp(m @ g['Nm'].T + ctx['em'] - 0.5 * ctx['A'])
        return a, b

    def flux(s, c, ctx):
        a, b = s.ab(c, ctx)
        return a - b

    def rhs_from_ab(s, a, b, ctx):
        return ((a - b) @ ctx['g']['Smat']).view(a.shape[0], s.NS, -1)

    def rhs(s, c, ctx):
        a, b = s.ab(c, ctx)
        return s.rhs_from_ab(a, b, ctx)

    def out_ratio(s, a, b, c, ctx):
        g = ctx['g']
        Dout = (a @ g['Cp'] + b @ g['Cm']).view_as(c)
        return float((Dout / c).max())

    def epr(s, c, ctx):
        a, b = s.ab(c, ctx)
        return ((a - b) * (torch.log(a) - torch.log(b))).sum(-1)

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
            c = 0.5 * (c + c2)
            t += dt; nsub += 1
            if nsub >= MAXSUB:
                break
        return c, nsub

    def A_full(s, ctx):                         # per-edge force incl. the chemostat part (gauge: A_raw ~ A_full mod im S^T)
        return ctx['A'] + ctx['ep'] - ctx['em']

    def A_cycle(s, ctx):
        return s.A_full(ctx) @ ctx['g']['P'] if s.chem else ctx['A']

    def cycle_affinity_local(s, ctx):           # (B, ncyc, V)
        return torch.einsum('bh,cvh->bcv', s.A_full(ctx), ctx['g']['zloc'])


class CRNN2(nn.Module):
    def __init__(s, bench, nd=2, Hd=64, chem=False):
        """chem=True: a, b enter as mass-action concentrations (kf a^{nu+_ext}, kb b^{nu-_ext}); kf, kb get no drive
        features (constant per hyperedge type)."""
        super().__init__()
        sp = SPECS[bench]
        s.bench, s.NS, s.chem = bench, len(sp['species']), chem
        nt = len(sp['rx']) + len(sp['mobile'])
        s.kmlp = MLP(nt + (0 if chem else nd), Hd, 2, bias_last=0.5)

    def prepare(s, drv, n):
        g = graph(s.bench, n, drv.device)
        kk = Fn.softplus(s.kmlp(edge_feats(drv[:, :0] if s.chem else drv, g))) + 1e-6
        kf, kb = kk[..., 0], kk[..., 1]
        if s.chem:
            kf = kf * torch.exp(drv @ g['Ep'].T); kb = kb * torch.exp(drv @ g['Em'].T)
        return dict(g=g, kf=kf, kb=kb, drv=drv, n=n)

    def ab(s, c, ctx):
        g = ctx['g']
        x = torch.cat([c.flatten(1), torch.ones_like(c[:, 0, :1])], 1)
        rp, rm = g['rp'], g['rm']
        a = ctx['kf'] * x[:, rp[:, 0]] * x[:, rp[:, 1]] * x[:, rp[:, 2]]
        b = ctx['kb'] * x[:, rm[:, 0]] * x[:, rm[:, 1]] * x[:, rm[:, 2]]
        return a, b

    def flux(s, c, ctx):
        a, b = s.ab(c, ctx)
        return a - b

    def rhs(s, c, ctx):
        return (s.flux(c, ctx) @ ctx['g']['Smat']).view(c.shape[0], s.NS, -1)

    def epr(s, c, ctx):
        a, b = s.ab(c, ctx)
        return ((a - b) * (torch.log(a) - torch.log(b))).sum(-1)

    def A_cycle(s, ctx):
        return torch.log(ctx['kf'] / ctx['kb']) @ ctx['g']['P']

    def cycle_affinity_local(s, ctx):
        return torch.einsum('bh,cvh->bcv', torch.log(ctx['kf'] / ctx['kb']), ctx['g']['zloc'])

    def advance(s, c, ctx, Dt):
        return heun_fixed(s, c, ctx, Dt)


def nbr_dirs_cl(h):
    return torch.stack([h.roll(-1, 2), h.roll(1, 2), h.roll(-1, 1), h.roll(1, 1)], 1)


class MLPLN(nn.Module):
    def __init__(s, i, h, o, ln=True, zero_last=False):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))
        s.ln = nn.LayerNorm(o) if ln else None
        if zero_last:
            nn.init.zeros_(s.net[-1].weight); nn.init.zeros_(s.net[-1].bias)

    def forward(s, x):
        y = s.net(x)
        return s.ln(y) if s.ln is not None else y


class GNN2(nn.Module):
    """B2 GNN-ODE: node inputs (c_1..c_NS, drv), edge inputs (dx, dy); 3 message-passing blocks; decodes dc/dt."""

    def __init__(s, bench, nd=2, Hd=64, L=3):
        super().__init__()
        s.NS = len(SPECS[bench]['species'])
        s.enc_n = MLPLN(s.NS + nd, Hd, Hd)
        s.enc_e = MLPLN(2, Hd, Hd)
        s.me = nn.ModuleList([MLPLN(3 * Hd, Hd, Hd) for _ in range(L)])
        s.mn = nn.ModuleList([MLPLN(2 * Hd, Hd, Hd) for _ in range(L)])
        s.dec = MLPLN(Hd, Hd, s.NS, ln=False, zero_last=True)
        s.register_buffer('dirs', torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]))

    def prepare(s, drv, n):
        B = drv.shape[0]
        e = s.dirs.view(1, 4, 1, 1, 2).expand(B, 4, n, n, 2)
        return dict(drv=drv, n=n, e0=s.enc_e(e))

    def rhs(s, c, ctx):
        B, n = c.shape[0], ctx['n']
        x = torch.cat([c.view(B, s.NS, n, n), ctx['drv'].view(B, -1, 1, 1).expand(B, -1, n, n)], 1)
        h = s.enc_n(x.permute(0, 2, 3, 1))
        e = ctx['e0']
        for me, mn in zip(s.me, s.mn):
            hn = nbr_dirs_cl(h)
            e = e + me(torch.cat([e, h.unsqueeze(1).expand_as(hn), hn], -1))
            h = h + mn(torch.cat([h, e.sum(1)], -1))
        return s.dec(h).permute(0, 3, 1, 2).reshape(B, s.NS, n * n)

    def advance(s, c, ctx, Dt):
        return heun_fixed(s, c, ctx, Dt)


def make_model(name, bench):
    return {'hyperme': lambda: HyperME2(bench, use_A=True), 'hyperme_rev': lambda: HyperME2(bench, use_A=False),
            'crnn': lambda: CRNN2(bench), 'gnn': lambda: GNN2(bench),
            'hyperme_chem': lambda: HyperME2(bench, use_A=True, chem=True),
            'crnn_chem': lambda: CRNN2(bench, chem=True),
            'hyperme_chemcvx': lambda: HyperME2(bench, use_A=True, chem=True, convex=True)}[name]()


# ----------------------------------------------------------------------------- true dynamics (numpy, HyperME form)
def true_ab_np(c, g, k, A, lng=None):
    """c (..., NS*V) > 0 -> a, b (..., H) with mu = log c (+ ln gamma(c) if lng is given)."""
    m = np.log(c) + (lng(c) if lng is not None else 0.0)
    return k * np.exp(m @ g['Np'].T + 0.5 * A), k * np.exp(m @ g['Nm'].T - 0.5 * A)


def solve_true(g, k, A, c0, nsteps, DT, rtol=1e-9, atol=1e-11, lng=None):
    from scipy.integrate import solve_ivp
    Sm = g['Nm'] - g['Np']

    def f(t, x):
        a, b = true_ab_np(np.maximum(x, 1e-300), g, k, A, lng)
        return (a - b) @ Sm
    ts = np.arange(nsteps + 1) * DT
    sol = solve_ivp(f, (0, ts[-1]), c0, method='DOP853', t_eval=ts, rtol=rtol, atol=atol)
    assert sol.success, sol.message
    C = sol.y.T
    assert C.min() > 0
    a, b = true_ab_np(C, g, k, A, lng)
    sig = ((a - b) * (np.log(a) - np.log(b))).sum(-1)
    return C, sig, a - b


def smooth_field(rng, n, K, std):
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    f = np.zeros((n, n))
    for kx in range(-K, K + 1):
        for ky in range(0, K + 1):
            if (kx, ky) == (0, 0) or kx * kx + ky * ky > K * K or (ky == 0 and kx < 0):
                continue
            ph = 2 * np.pi * (kx * b + ky * a) / n
            f += rng.normal() * np.cos(ph) + rng.normal() * np.sin(ph)
    f -= f.mean()
    return (std * f / (f.std() + 1e-12)).ravel()


def selftest(bench='brus', n=8):
    """random-init + randomly perturbed HyperME2: A in ker S, positivity, sigma >= 0 over 200 steps."""
    torch.manual_seed(0)
    g = graph(bench, n, 'cpu')
    print(f'[{bench}] H={g["H"]} dim kerS={g["Z"].shape[1]} dim kerS^T={g["L"].shape[1]}')
    m = HyperME2(bench)
    with torch.no_grad():
        for p in list(m.amlp.net[-1].parameters()) + list(m.G.net[-1].parameters()):
            p.normal_(0, 0.5)
        drv = torch.randn(4, 2)
        ctx = m.prepare(drv, n)
        SA = (ctx['A'] @ g['Smat']).abs().max().item()
        c = torch.rand(4, g['NS'], n * n) * 3 + 0.05
        smin, cmin = 1e9, 1e9
        for _ in range(200):
            smin = min(smin, m.epr(c, ctx).min().item())
            c, _ = m.advance(c, ctx, 0.1)
            cmin = min(cmin, c.min().item())
    print(f'  |S A|max {SA:.2e}  min c {cmin:.3e}  min sigma {smin:.3e}')
    assert SA < 1e-10 and cmin > 0 and smin >= 0
    return dict(SA=SA, min_c=cmin, min_sigma=smin)


if __name__ == '__main__':
    import json
    r = {b: selftest(b) for b in ['schlogl', 'brus']}
    os.makedirs(RUNS, exist_ok=True)
    json.dump(r, open(os.path.join(RUNS, 'C23_selftest.json'), 'w'), indent=1)
    print('SELFTEST PASSED')
