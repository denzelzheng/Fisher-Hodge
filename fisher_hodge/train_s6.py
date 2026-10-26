"""G2: nonlocal Keller-Segel-type interaction (S6 of gen_data2.py), TC-ME vs Flux-GNN, extrapolation in chi/D.
  python3 train_s6.py --gen                         # runs/data/S6.pt (if missing) + runs/data/S6_chi{c}.pt, c in CHIS
  python3 train_s6.py --model tcme_s6 --seed 0      # train (2500 it, train.py protocol) + evaluate on all chi sets
  python3 train_s6.py --model flux_gnn_s6 --seed 1
Both models receive the same nonlocal state-dependent input c = (chi/D) W*rho on the 32x32 grid (Gaussian kernel, sigma 0.1).
  tcme_s6      TC-ME rates with an explicit gradient-form potential (no double backward):
                 phi = alpha U + g(rho/RS, U) + beta c + (chi/D) W*[p(c, U)]
               = dE/drho for E = sum_i [alpha U rho + G(rho, U) + (beta/2) rho c + Phi(c, U)], G = int_0^rho g, Phi = int_0^c p
               (W symmetric; any scalar MLP of one variable is the derivative of its integral).  True S6: beta = -1, g = p = 0.
               F = sum rho (log rho - 1) + E is an exact Lyapunov function whenever the cycle part A = 0 (G, Phi by 8-pt Gauss-Legendre).
  flux_gnn_s6  Flux-GNN (black-box non-negative directed rates from an encode-process-decode GNN, d^T assembly) with c in the node encoder.
Outputs: runs/general/G2_{model}_s{seed}.{json,log,pt}."""
import argparse, json, time, os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint
from common import *
import gen_data2 as G2
import models2                                   # noqa: F401
from models2 import FluxGNN
from models import MLP, node_deriv, NSUB, static_edge_feats
from models_tcme import TCME
import models_tcme
from train_tcme import relerr_b, jclean

dev = 'cpu' if os.environ.get('FH_CPU') else 'cuda'
CHIS = [0.3, 0.5, 0.8]
CHI_TRAIN = 0.3
ODIR = os.path.join(RUNS, 'general')
_WK = {}


def Wk(n, device, cdtype):
    key = (n, str(device), cdtype)
    if key not in _WK:
        _WK[key] = G2.kernel_fft(n, G2.SIGW).to(device=device, dtype=cdtype)
    return _WK[key]


def convW(x):
    n = x.shape[-1]
    X = torch.fft.rfft2(x)
    return torch.fft.irfft2(X * Wk(n, x.device, X.dtype), s=(n, n)) / n ** 2


def cfield(rho, F):
    return F['chi'].view(-1, 1, 1) * convW(rho)


# ------------------------------------------------------------------ data
def chi_name(c):
    return os.path.join(RUNS, 'data', f'S6_chi{c:g}.pt')


def gen():
    os.makedirs(ODIR, exist_ok=True)
    p6 = os.path.join(RUNS, 'data', 'S6.pt')
    if not os.path.exists(p6):
        data, meta = {}, {}
        for setname in G2.SETS:
            off = {'train': 0, 'test_id': 1, 'test_oodpot': 2, 'test_oodinit': 3}[setname]
            out, info, params = G2.make_set2('S6', setname, 6000 + off, G2.CHI_OVER_D * D)
            data[setname] = out; meta[setname] = dict(info=info, params=params)
            print('S6', setname, json.dumps(info), flush=True)
        torch.save(data, p6)
        json.dump(dict(DT=DT, DT_REF=G2.DT_REF, NF=NF, N=N, D=D, chi_over_D=G2.CHI_OVER_D, sigw=G2.SIGW, sets=meta),
                  open(os.path.join(RUNS, 'data', 'S6_meta.json'), 'w'), default=float)
    info = {}
    for c in CHIS:
        G2.NF = NF; G2.coarsen = coarsen; G2.coarsen_flux = coarsen_flux
        out, inf, _ = G2.make_set2('S6', 'test_id', 6001, c * D)
        inf['nf'] = NF
        if inf['min_rho'] < 0 or c == max(CHIS):          # positivity failure -> refine; always check grid convergence at the largest chi
            G2.NF = 2 * NF
            G2.coarsen = lambda x: coarsen(coarsen(x)); G2.coarsen_flux = lambda J: coarsen_flux(coarsen_flux(J))
            out2, inf2, _ = G2.make_set2('S6', 'test_id', 6001, c * D)
            G2.NF = NF; G2.coarsen = coarsen; G2.coarsen_flux = coarsen_flux
            inf['grid_check_relerr'] = f'{float(relerr_b(out["rho"][:, -1].numpy(), out2["rho"][:, -1].numpy()).max()):.2e}'
            inf['min_rho_128'] = inf2['min_rho']; inf['max_rho_128'] = inf2['max_rho']
            if inf['min_rho'] < 0:
                out, inf = out2, dict(inf2, nf=2 * NF, grid_check_relerr=inf['grid_check_relerr'])
        out['chi'] = c
        torch.save(dict(test=out), chi_name(c))
        info[f'{c:g}'] = inf
        print('chi/D', c, json.dumps(inf), flush=True)
    json.dump(info, open(os.path.join(ODIR, 'G2_gen_info.json'), 'w'), indent=1)


def chi_name128(c):
    return os.path.join(RUNS, 'data', f'S6_chi{c:g}_nf128.pt')


def gen128():
    """128x128 fine-grid references (4x4 coarse-graining) for every chi: reference-resolution check."""
    ip = os.path.join(ODIR, 'G2_gen_info.json'); info = json.load(open(ip))
    G2.NF = 2 * NF
    G2.coarsen = lambda x: coarsen(coarsen(x)); G2.coarsen_flux = lambda J: coarsen_flux(coarsen_flux(J))
    for c in CHIS:
        out, inf, _ = G2.make_set2('S6', 'test_id', 6001, c * D)
        out['chi'] = c
        torch.save(dict(test=out), chi_name128(c))
        y64 = torch.load(chi_name(c))['test']['rho']
        inf['relerr_64_vs_128_10T_median'] = float(np.median(relerr_b(y64[:, -1].numpy(), out['rho'][:, -1].numpy())))
        inf['relerr_64_vs_128_10T_max'] = float(relerr_b(y64[:, -1].numpy(), out['rho'][:, -1].numpy()).max())
        info[f'nf128_{c:g}'] = inf
        print('nf128 chi/D', c, json.dumps(inf), flush=True)
    G2.NF = NF; G2.coarsen = coarsen; G2.coarsen_flux = coarsen_flux
    json.dump(info, open(ip, 'w'), indent=1)


def fields(setd, chi, idx=None):
    """chi = None: per-trajectory chi/D from setd['chi'] (multi-chi training set)."""
    F = {k: (setd[k] if idx is None else setd[k][idx]).to(dev) for k in ['U', 'dpsi', 'psin']}
    if chi is None:
        F['chi'] = (setd['chi'] if idx is None else setd['chi'][idx]).float().to(dev)
    else:
        F['chi'] = torch.full((F['U'].shape[0],), float(chi), device=dev)
    return F


# ------------------------------------------------------------------ models
class TCMES6(TCME):
    name = 'tcme'

    def __init__(s, H=128):
        super().__init__('S6', H)
        s.p = MLP(2, H, 1, zero_last=True)
        s.beta = nn.Parameter(torch.tensor(0.0))
        x, w = np.polynomial.legendre.leggauss(8)
        s.register_buffer('gl_t', torch.tensor((x + 1) / 2, dtype=torch.float32)); s.register_buffer('gl_w', torch.tensor(w / 2, dtype=torch.float32))

    def gloc(s, x, U):
        return s.g(torch.stack([x, U], -1)).squeeze(-1)

    def pfun(s, c, U):
        return s.p(torch.stack([c, U], -1)).squeeze(-1)

    def phi(s, rho, F):
        U = F['U']; c = cfield(rho, F)
        return (s.alpha * U + s.gloc(rho / s.RS, U) + s.beta * c
                + F['chi'].view(-1, 1, 1) * convW(s.pfun(c, U)))

    def free_energy(s, rho, F):
        U = F['U']; c = cfield(rho, F)
        G = sum(w * rho * s.gloc(t * rho / s.RS, U) for t, w in zip(s.gl_t, s.gl_w))
        Phi = sum(w * c * s.pfun(t * c, U) for t, w in zip(s.gl_t, s.gl_w))
        return (rho * (torch.log(rho) - 1) + s.alpha * U * rho + G + 0.5 * s.beta * rho * c + Phi).mean((-1, -2))


class FluxGNNS6(FluxGNN):
    def __init__(s):
        super().__init__('S6')

    def rates(s, rho, ctx):
        F = ctx['F']
        h = s.enc_n(torch.stack([rho / s.RS, F['U'], cfield(rho, F)], -1))
        e = ctx['e0']
        for me, mn in zip(s.me, s.mn):
            hn = nbr_dirs_cl(h)
            e = e + me(torch.cat([e, h.unsqueeze(1).expand_as(hn), hn], -1))
            h = h + mn(torch.cat([h, e.sum(1)], -1))
        return Fn.softplus(s.dec(e).squeeze(-1))


# ------------------------------------------------------------------ G2b: constrained free energies
class BoundedP:
    """(a) interaction channel with bounded slope: p(c) = -a c0 log(1 + c / c0), a = softplus >= 0, c0 = softplus > 0, |p'| <= a
    (purely attractive, sub-linear; Phi = int p grows at most like a c^2 / 2, so rho log rho + bounded local part dominates)."""
    def _init_p(s):
        del s.p
        s.pa_raw = nn.Parameter(torch.tensor(-3.0)); s.pc0_raw = nn.Parameter(torch.tensor(0.5413))   # a ~ 0.05, c0 = 1

    def pfun(s, c, U):
        a, c0 = Fn.softplus(s.pa_raw), Fn.softplus(s.pc0_raw)
        return -a * c0 * torch.log1p(c.clamp_min(0) / c0)


class ConvexLocal:
    """(b) convex local free energy: G(rho;U) = RS sum_k w_k(U) log cosh(s_k(U) x + b_k(U)) / s_k(U) + g0(U) x RS, x = rho / RS,
    a one-hidden-layer input-convex network in x (w, s >= 0 via softplus, log cosh convex; U enters through a hypernetwork),
    so g = dG/drho = sum_k w_k tanh(s_k x + b_k) + g0 is monotone non-decreasing and bounded in rho (closed form, no autograd)."""
    K = 32

    def _init_g(s, H=128):
        s.ghyper = MLP(1, H, 3 * s.K + 1, zero_last=True)
        nn.init.constant_(s.ghyper.net[-1].bias[s.K:2 * s.K], 0.5413)          # s_k = 1 at init
        nn.init.constant_(s.ghyper.net[-1].bias[:s.K], -6.0)                   # w_k ~ 0 at init
        del s.g

    def gloc(s, x, U):
        h = s.ghyper(U.unsqueeze(-1))
        K = s.K
        w, sl, b, g0 = Fn.softplus(h[..., :K]), Fn.softplus(h[..., K:2 * K]), h[..., 2 * K:3 * K], h[..., 3 * K]
        return (w * torch.tanh(sl * x.unsqueeze(-1) + b)).sum(-1) + g0


class TCMES6a(BoundedP, TCMES6):
    def __init__(s):
        TCMES6.__init__(s); s._init_p()


class TCMES6b(ConvexLocal, TCMES6):
    def __init__(s):
        TCMES6.__init__(s); s._init_g()


class TCMES6ab(BoundedP, ConvexLocal, TCMES6):
    def __init__(s):
        TCMES6.__init__(s); s._init_p(); s._init_g()


class TCMES6phys(TCME):
    """G2d: known physical form of the free energy, F = sum rho log rho + sum U rho - (s chi/D / 2) rho^T W*rho, i.e.
    phi = U - s c with c = (chi/D) W*rho;  learned: the scalar s (true 1, stored as beta = -s) and the edge rates k.
    No local g, no p, no cycle part (A = 0)."""
    name = 'tcme'

    def __init__(s, H=128):
        super().__init__('S6', H)
        del s.g, s.psi
        s.beta = nn.Parameter(torch.tensor(0.0))

    def prepare(s, F):
        U = F['U']; n = U.shape[-1]; S = D * n * n
        sym, _ = static_edge_feats(F)
        k = S * Fn.softplus(s.k(sym).squeeze(-1))
        return dict(F=F, k=k, A=torch.zeros_like(k), S=S, h=1.0 / n, n=n)

    def phi(s, rho, F):
        return F['U'] + s.beta * cfield(rho, F)

    def free_energy(s, rho, F):
        return (rho * (torch.log(rho) - 1) + F['U'] * rho + 0.5 * s.beta * rho * cfield(rho, F)).mean((-1, -2))


MODELS = dict(tcme_s6=TCMES6, flux_gnn_s6=FluxGNNS6, tcme_s6a=TCMES6a, tcme_s6b=TCMES6b, tcme_s6ab=TCMES6ab, tcme_s6phys=TCMES6phys)


def true_free_energy(rho, U, chi):
    """coarse-grid true free energy (units of D): mean[rho log rho + rho U/D - (chi/2D) rho W*rho]."""
    r = rho.clamp_min(1e-30)
    return (r * torch.log(r) + r * U - 0.5 * chi * r * convW(r)).mean((-1, -2))


def up_frac(Fs, tol=1e-6):
    """Fs (B, T+1): fraction of steps with F_{k+1} > F_k + tol * |F_k| (pooled over trajectories), and max relative increase."""
    d = Fs[:, 1:] - Fs[:, :-1]
    rel = d / Fs[:, :-1].abs().clamp_min(1e-12)
    ok = torch.isfinite(rel)
    return float((rel[ok] > tol).float().mean()), float(rel[ok].max()) if ok.any() else float('nan')


# ------------------------------------------------------------------ G2c: multi-chi training set + beta diagnostics
CHIS_TRAIN_MC = [0.15, 0.3, 0.45]


def gen_multichi():
    """S6 training set with chi/D in CHIS_TRAIN_MC, 16 trajectories each (20 steps), same generator / distributions as S6 train."""
    G2.NF = NF; G2.coarsen = coarsen; G2.coarsen_flux = coarsen_flux
    old = G2.SETS['train']; G2.SETS['train'] = (16, T_TRAIN)
    parts, info = [], {}
    for i, c in enumerate(CHIS_TRAIN_MC):
        out, inf, _ = G2.make_set2('S6', 'train', 6100 + i, c * D)
        out['chi'] = torch.full((out['rho'].shape[0],), c)
        parts.append(out); info[f'{c:g}'] = inf
        print('multichi train chi/D', c, json.dumps(inf), flush=True)
    G2.SETS['train'] = old
    tr = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
    torch.save(dict(train=tr), os.path.join(RUNS, 'data', 'S6_multichi.pt'))
    json.dump(info, open(os.path.join(ODIR, 'G2c_gen_info.json'), 'w'), indent=1)


@torch.no_grad()
def diag_beta(runs=('tcme_s6ab_s0', 'tcme_s6ab_mc_s0', 'tcme_s6_s0')):
    """Where does the learned model put the interaction?  On all training snapshots (single-chi S6 train and the multi-chi set):
    (1) collinearity of the true interaction feature c = (chi/D) W*rho with local quantities (R^2 of c on [1, rho, rho^2, log rho, U]);
    (2) for each model, regression of each phi component on the true interaction potential -c:
        beta c, chi W*p(c), g(rho, U) - g(0, U)  and their sum (the effective interaction the model has learned)."""
    sets = dict(single=(torch.load(os.path.join(RUNS, 'data', 'S6.pt'))['train'], None),
                multi=(torch.load(os.path.join(RUNS, 'data', 'S6_multichi.pt'))['train'], None))
    out = {}
    for sn, (tr, _) in sets.items():
        tr = dict(tr, rho=tr['rho'][:, ::4])                       # every 4th snapshot (t = 0, 0.2, ..., 1.0): memory
        B, T1 = tr['rho'].shape[:2]
        chi = tr['chi'] if 'chi' in tr else torch.full((B,), CHI_TRAIN)
        rho = tr['rho'].reshape(B * T1, N, N).to(dev); U = tr['U'].repeat_interleave(T1, 0).to(dev)
        chv = chi.repeat_interleave(T1).to(dev)
        c = chv.view(-1, 1, 1) * convW(rho)
        X = torch.stack([torch.ones_like(rho), rho, rho ** 2, torch.log(rho), U], -1).reshape(-1, 5).double()
        y = c.reshape(-1).double()
        coef = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution
        r2 = 1 - ((X @ coef).squeeze() - y).var() / y.var()
        Xr = torch.stack([torch.ones_like(rho), rho], -1).reshape(-1, 2).double()
        r2lin = 1 - ((Xr @ torch.linalg.lstsq(Xr, y.unsqueeze(-1)).solution).squeeze() - y).var() / y.var()
        S = dict(R2_c_on_local=float(r2), R2_c_on_rho_linear=float(r2lin), corr_c_rho=float(torch.corrcoef(torch.stack([y, rho.reshape(-1).double()]))[0, 1]))
        # within-snapshot variation after removing the per-chi mean: does chi itself vary the feature independently of rho?
        F = dict(U=U, dpsi=torch.zeros(B * T1, 2, N, N, device=dev), psin=torch.zeros(B * T1, N, N, device=dev), chi=chv)
        for rn in runs:
            pth = os.path.join(ODIR, f'G2_{rn}.pt')
            if not os.path.exists(pth):
                continue
            mname = rn.rsplit('_s', 1)[0].replace('_mc', '')
            m = MODELS[mname]().to(dev); m.load_state_dict(torch.load(pth, map_location=dev)); m.eval()
            comp = dict(beta_c=m.beta * c, chiWp=chv.view(-1, 1, 1) * convW(m.pfun(c, U)),
                        g_local=m.gloc(rho / m.RS, U) - m.gloc(torch.zeros_like(rho), U))
            comp['total'] = comp['beta_c'] + comp['chiWp'] + comp['g_local']
            t = (-c).reshape(-1).double()
            R = {}
            for k, v in comp.items():
                v = v.reshape(-1).double()
                R[k] = dict(slope_on_minus_c=float((v * t).sum() / (t * t).sum()),
                            R2=float(1 - ((v - t).pow(2).sum() / (t - t.mean()).pow(2).sum())) if k == 'total' else None)
            R['beta'] = float(m.beta)
            S[rn] = R
        out[sn] = S
        print(sn, json.dumps(S, indent=None), flush=True)
    json.dump(out, open(os.path.join(ODIR, 'G2c_diag.json'), 'w'), indent=1)


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gen', action='store_true')
    p.add_argument('--gen_multichi', action='store_true'); p.add_argument('--diag_beta', action='store_true')
    p.add_argument('--train_set', default='single', choices=['single', 'multichi'])
    p.add_argument('--model', default=None); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=2500); p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--kmin', type=int, default=4); p.add_argument('--kmax', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--maxsub_train', type=int, default=0, help='>0: cap TC-ME substeps per DT during training; an iteration that hits the cap (stiff rates) is skipped')
    p.add_argument('--ckpt', action='store_true', help='activation checkpointing per DT step for tcme_s6 too (memory cap)')
    p.add_argument('--gen128', action='store_true'); p.add_argument('--eval_only', action='store_true')
    args = p.parse_args()
    if args.gen:
        gen()
    if args.gen_multichi:
        gen_multichi()
    if args.diag_beta:
        diag_beta()
    if args.gen128:
        gen128()
    if args.model is None:
        sys.exit(0)
    os.makedirs(ODIR, exist_ok=True)
    if args.smoke:
        args.iters = 20
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(20_000 + args.seed)
    name = f'G2_{args.model}' + ('_mc' if args.train_set == 'multichi' else '') + f'_s{args.seed}' + ('_smoke' if args.smoke else '')
    logf = open(os.path.join(ODIR, name + '.log'), 'a' if args.eval_only else 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()

    data = torch.load(os.path.join(RUNS, 'data', 'S6_multichi.pt' if args.train_set == 'multichi' else 'S6.pt'))
    model = MODELS[args.model]().to(dev)
    if args.maxsub_train:
        models_tcme.MAXSUB = args.maxsub_train
    nparam = sum(q.numel() for q in model.parameters())
    log(f'{name}: params={nparam} args={vars(args)}')
    if args.eval_only:
        model.load_state_dict(torch.load(os.path.join(ODIR, name + '.pt'), map_location=dev)); args.iters = 0
        old = json.load(open(os.path.join(ODIR, name + '.json')))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=1e-5)
    tr = data['train']; RHO = tr['rho'].to(dev); ntr = RHO.shape[0]
    t0 = time.time(); hist = []; nskip = 0; subs = []
    model.train()
    for it in range(args.iters):
        idx = torch.tensor(rng.choice(ntr, args.batch, replace=False))
        K = int(rng.integers(args.kmin, args.kmax + 1))
        st = torch.tensor(rng.integers(0, T_TRAIN - K + 1, size=len(idx)), device=dev)
        F = fields(tr, None if 'chi' in tr else CHI_TRAIN, idx)
        ctx = model.prepare(F)
        r = RHO[idx.to(dev), st]
        loss = 0.0; ns_it = 0
        for k in range(1, K + 1):
            if args.model == 'flux_gnn_s6' or args.ckpt:
                nsl = []
                def stepf(rr):
                    out = model.advance(rr, ctx, DT); nsl.append(out[1]); return out[0]
                r = checkpoint(stepf, r, use_reentrant=False); ns = nsl[-1]
            else:
                r, ns = model.advance(r, ctx, DT)
            if args.maxsub_train and ns >= args.maxsub_train:
                loss = float('nan'); break
            ns_it += ns
            y = RHO[idx.to(dev), st + k]
            loss = loss + (((r - y) ** 2).sum((-1, -2)) / (y ** 2).sum((-1, -2))).mean()
        loss = loss / K
        if args.maxsub_train and not math.isfinite(float(loss)):
            nskip += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        subs.append(ns_it / K)
        if not torch.isfinite(loss):
            nskip += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(float(loss.detach()))
        if it % 100 == 0 or it == args.iters - 1:
            extra = f' beta {float(model.beta):.3f} alpha {float(model.alpha):.3f}' if hasattr(model, 'beta') else ''
            log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} gnorm {float(gn):.2e} K {K} substeps/DT {np.mean(subs[-100:]):.1f} skipped {nskip} t {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB' + extra)
    train_time = time.time() - t0
    if not args.eval_only:
        torch.save(model.state_dict(), os.path.join(ODIR, name + '.pt'))

    # ------------------------------------------------------------------ evaluation on chi sets
    model.eval()
    models_tcme.MAXSUB = 4000
    res = dict(name=name, model=args.model, seed=args.seed, params=nparam, train_time_s=train_time,
               final_train_loss=float(np.mean(hist[-100:])) if hist else float('nan'), train_nskip=nskip, chi={})
    if args.eval_only:
        res.update({k: old[k] for k in ['train_time_s', 'final_train_loss', 'train_nskip']})
    if hasattr(model, 'beta'):
        res['beta'] = round(float(model.beta), 4); res['alpha'] = round(float(model.alpha), 4)
    if hasattr(model, 'pa_raw'):
        res['p_a'] = round(float(Fn.softplus(model.pa_raw)), 4); res['p_c0'] = round(float(Fn.softplus(model.pc0_raw)), 4)
    for c in CHIS:
        te = time.time()
        sd = torch.load(chi_name(c))['test']
        F = fields(sd, c); Y = sd['rho'].double(); B, T1 = Y.shape[:2]
        with torch.no_grad():
            ctx = model.prepare(F)
            r = sd['rho'][:, 0].to(dev); out = [r.cpu()]; nsub = []
            for k in range(T1 - 1):
                r, ns = model.advance(r, ctx, DT); nsub.append(ns)
                bad = (~torch.isfinite(r)).any(-1).any(-1) | (r.abs().amax((-1, -2)) > 1e4)
                if bad.any():
                    r = torch.where(bad[:, None, None], torch.full_like(r, float('nan')), r)
                out.append(r.cpu())
            R = torch.stack(out, 1).double()
        err = (torch.linalg.vector_norm((R - Y).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T1, -1), dim=-1)).numpy()
        err = np.where(np.isfinite(err), err, np.inf)
        Rn = R.numpy()
        pk = np.abs(np.nanmax(Rn[:, -1], axis=(1, 2)) - Y[:, -1].amax((-1, -2)).numpy()) / Y[:, -1].amax((-1, -2)).numpy()
        pk = np.where(np.isfinite(pk), pk, np.inf)
        fin = torch.isfinite(R).all(-1).all(-1).all(-1)
        S = dict(err10T=float(np.median(err[:, -1])), errT=float(np.median(err[:, 20])), err5T=float(np.median(err[:, 100])),
                 err10T_per_traj=err[:, -1].tolist(), peak_err10T=float(np.median(pk)),
                 n_blown=int((~fin).sum()), mass_err=float(np.nanmax(np.abs(Rn.mean((-1, -2)) - 1))), min_rho=float(np.nanmin(Rn)),
                 max_rho=float(np.nanmax(Rn)), max_rho_ref=float(Y.max()), n_neg=int((Rn < 0).any((1, 2, 3)).sum()),
                 substeps=float(np.mean(nsub)), max_substeps=int(np.max(nsub)), infer_s=time.time() - te)
        Ug = F['U'].double().cpu()
        Ft = torch.stack([true_free_energy(R[:, k], Ug, c) for k in range(T1)], 1)
        Fr = torch.stack([true_free_energy(Y[:, k], Ug, c) for k in range(T1)], 1)
        S['Ftrue_up_frac'], S['Ftrue_max_rel_up'] = up_frac(Ft[fin])
        S['Ftrue_up_frac_ref'], S['Ftrue_max_rel_up_ref'] = up_frac(Fr)
        if hasattr(model, 'free_energy'):
            with torch.no_grad():
                Fl = torch.stack([model.free_energy(R[:, k].float().to(dev).clamp_min(1e-30), F).double().cpu() for k in range(T1)], 1)
            S['Flearn_up_frac'], S['Flearn_max_rel_up'] = up_frac(Fl[fin])
            S['A_rms'] = float(ctx['A'].pow(2).mean().sqrt())
        if os.path.exists(chi_name128(c)):
            Y2 = torch.load(chi_name128(c))['test']['rho'].double()
            e2 = (torch.linalg.vector_norm((R - Y2).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y2.reshape(B, T1, -1), dim=-1)).numpy()
            e2 = np.where(np.isfinite(e2), e2, np.inf)
            p2 = np.abs(np.nanmax(Rn[:, -1], axis=(1, 2)) - Y2[:, -1].amax((-1, -2)).numpy()) / Y2[:, -1].amax((-1, -2)).numpy()
            S['err10T_r128'] = float(np.median(e2[:, -1])); S['errT_r128'] = float(np.median(e2[:, 20]))
            S['peak_err10T_r128'] = float(np.median(np.where(np.isfinite(p2), p2, np.inf)))
            log(f'[chi/D={c:g}] vs 128-grid reference: 10T {S["err10T_r128"]:.3e} T {S["errT_r128"]:.3e} peak {S["peak_err10T_r128"]:.3e}')
        res['chi'][f'{c:g}'] = S
        log(f'[chi/D={c:g}] 10T {S["err10T"]:.3e} T {S["errT"]:.3e} peak {S["peak_err10T"]:.3e} blown {S["n_blown"]} mass {S["mass_err"]:.1e} '
            f'neg {S["n_neg"]} maxrho {S["max_rho"]:.2f} (ref {S["max_rho_ref"]:.2f}) Ftrue-up {S["Ftrue_up_frac"]:.3f} (ref {S["Ftrue_up_frac_ref"]:.3f}) '
            + (f'Flearn-up {S["Flearn_up_frac"]:.3f} maxrel {S["Flearn_max_rel_up"]:.1e} A_rms {S["A_rms"]:.1e} ' if 'Flearn_up_frac' in S else '')
            + f'sub {S["substeps"]:.1f}/{S["max_substeps"]} {S["infer_s"]:.0f}s')
    json.dump(jclean(res), open(os.path.join(ODIR, name + '.json'), 'w'))
    log('done', name)
