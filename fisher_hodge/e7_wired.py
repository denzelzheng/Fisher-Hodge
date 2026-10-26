"""E7: "wired thermodynamics, learn only the kinetics" (tcme_wired) vs Flux-GNN on S2, knob extrapolation x data size.

tcme_wired (TC-ME with everything thermodynamic fixed):
    mu = log rho + U                      (alpha = 1 fixed, no g)
    A_e = s_A * dpsi_e + b_dir            (dpsi_e = exact edge line integral of the drift / D, from the input; s_A init 1,
                                           2 harmonic coefficients b init 0; no free per-edge A)
    k_e = (D/h^2) exp(kappa_theta(local geometric edge features))   (small MLP, zero-initialised)
  untrained (N = 0) it is exactly the SQRA discretisation of the reference PDE.
  Temperature edit (G1 rule): k -> r k, phi -> phi / r, A -> A / r (r = D_test / D_train).
Flux-GNN: models2.FluxGNN (unchanged); temperature: input U -> U / r (best approximation in G1); counterfactual: drive inputs = 0.
  python3 e7_wired.py --train --model tcme_wired --ntrain 4 [--iters 1500 --max_minutes 40]
  python3 e7_wired.py --eval       # all checkpoints in runs/general/e7 (+ untrained wired, + Flux-GNN N=64 from runs/tcme) -> E7.json
  python3 e7_wired.py --report     # E7_summary.md, E7_knobs.png
"""
import argparse, json, os, time, math, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint
from common import *
import models2                                   # noqa: F401
from models2 import FluxGNN
from models import MLP, static_edge_feats, NSUB
from models_tcme import TCME

dev = 'cpu' if os.environ.get('FH_CPU') else 'cuda'
ODIR = os.path.join(RUNS, 'general')
EDIR = os.path.join(ODIR, 'e7')
D_TRAIN = 0.02
NS = [1, 4, 16, 64]


class TCMEWired(TCME):
    name = 'tcme'

    def __init__(s, H=32):
        nn.Module.__init__(s)
        s.sys, s.RS = 'S2', 10.0
        s.kap = MLP(6, H, 1, zero_last=True)
        s.sA = nn.Parameter(torch.tensor(1.0))
        s.bh = nn.Parameter(torch.zeros(2))
        s.tscale, s.Ascale, s.kscale = 1.0, 1.0, 1.0       # temperature edit knobs (not parameters)
        s.sub_log = []

    def prepare(s, F):
        U = F['U']; n = U.shape[-1]; S = D * n * n
        sym, _ = static_edge_feats(F)
        k = s.kscale * S * torch.exp(s.kap(sym).squeeze(-1))
        A = s.Ascale * (s.sA * F['dpsi'] + s.bh.view(1, 2, 1, 1))
        return dict(F=F, k=k, A=A, S=S, h=1.0 / n, n=n)

    def phi(s, rho, F):
        return s.tscale * F['U']


def make(model):
    return TCMEWired().to(dev) if model == 'tcme_wired' else FluxGNN('S2').to(dev)


def fields(sd, idx=None):
    return {k: (sd[k] if idx is None else sd[k][idx]).to(dev) for k in ['U', 'dpsi', 'psin']}


# ------------------------------------------------------------------ training (train_tcme.py protocol)
def train(a):
    os.makedirs(EDIR, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(10_000 + a.seed)
    name = f'E7_{a.model}_N{a.ntrain}_s{a.seed}'
    logf = open(os.path.join(EDIR, name + '.log'), 'w')

    def log(*x):
        s = ' '.join(str(v) for v in x); print(s, flush=True); logf.write(s + '\n'); logf.flush()

    tr = torch.load(os.path.join(RUNS, 'data', 'S2.pt'))['train']
    tr = {k: v[:a.ntrain] for k, v in tr.items()}
    RHO = tr['rho'].to(dev); ntr = RHO.shape[0]
    model = make(a.model)
    nparam = sum(q.numel() for q in model.parameters())
    log(f'{name}: params={nparam} args={vars(a)}')
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=1e-5)
    t0 = time.time(); hist = []; nskip = 0; it_done = 0
    for it in range(a.iters):
        if a.max_minutes and time.time() - t0 > 60 * a.max_minutes:
            log(f'time cap {a.max_minutes} min reached at iteration {it}'); break
        idx = torch.tensor(rng.choice(ntr, min(a.batch, ntr), replace=False))
        K = int(rng.integers(a.kmin, a.kmax + 1))
        st = torch.tensor(rng.integers(0, T_TRAIN - K + 1, size=len(idx)), device=dev)
        F = fields(tr, idx); ctx = model.prepare(F)
        r = RHO[idx.to(dev), st]; loss = 0.0
        for k in range(1, K + 1):
            if a.model == 'flux_gnn':
                r = checkpoint(lambda rr: model.advance(rr, ctx, DT)[0], r, use_reentrant=False)
            else:
                r, _ = model.advance(r, ctx, DT)
            y = RHO[idx.to(dev), st + k]
            loss = loss + (((r - y) ** 2).sum((-1, -2)) / (y ** 2).sum((-1, -2))).mean()
        loss = loss / K
        if not torch.isfinite(loss):
            nskip += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); hist.append(float(loss.detach())); it_done = it + 1
        if it % 100 == 0 or it == a.iters - 1:
            extra = f' sA {float(model.sA):.3f} b {model.bh.detach().cpu().numpy().round(4).tolist()}' if a.model == 'tcme_wired' else ''
            log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} gnorm {float(gn):.2e} K {K} t {time.time() - t0:.0f}s '
                f'mem {torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0:.2f}GB' + extra)
    torch.save(model.state_dict(), os.path.join(EDIR, name + '.pt'))
    json.dump(dict(name=name, model=a.model, ntrain=a.ntrain, iters_done=it_done, iters_planned=a.iters, train_time_s=time.time() - t0,
                   final_train_loss=float(np.mean(hist[-100:])) if hist else None, nskip=nskip, params=nparam),
              open(os.path.join(EDIR, name + '_train.json'), 'w'))
    log('done', name)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def rollout(model, ctx, rho0, nsteps=T_TEST):
    r = rho0.clone(); out = [r.cpu()]
    for k in range(nsteps):
        r, _ = model.advance(r, ctx, DT)
        bad = (~torch.isfinite(r)).any(-1).any(-1) | (r.abs().amax((-1, -2)) > 1e4)
        if bad.any():
            r = torch.where(bad[:, None, None], torch.full_like(r, float('nan')), r)
        out.append(r.cpu())
    return torch.stack(out, 1).double()


def err_stats(R, Y):
    B, T1 = Y.shape[:2]
    e = (torch.linalg.vector_norm((R - Y).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T1, -1), dim=-1)).numpy()
    e = np.where(np.isfinite(e), e, np.inf)
    Rn = R.numpy()
    return dict(err10T=float(np.median(e[:, -1])), errT=float(np.median(e[:, 20])), n_blown=int((~np.isfinite(Rn[:, -1])).any((1, 2)).sum()),
                mass_err=float(np.nanmax(np.abs(Rn.mean((-1, -2)) - 1))), n_neg=int((Rn < 0).any((1, 2, 3)).sum()))


def flux_phys(model, rho, ctx):
    n = rho.shape[-1]
    return model.flux(rho, ctx) / n if model.name == 'tcme' else -(D * n) * model.flux_q(rho, ctx)


def evaluate():
    sets = dict(ID=torch.load(os.path.join(RUNS, 'data', 'S2.pt'))['test_id'],
                amp2=torch.load(os.path.join(RUNS, 'data', 'S2_amp2.pt'))['test'], amp4=torch.load(os.path.join(RUNS, 'data', 'S2_amp4.pt'))['test'],
                D0p01=torch.load(os.path.join(RUNS, 'data', 'S2_D0.01.pt'))['test'], D0p04=torch.load(os.path.join(RUNS, 'data', 'S2_D0.04.pt'))['test'])
    cf = torch.load(os.path.join(RUNS, 'data', 'S2_amp0.pt'))['test']
    J1 = torch.load(os.path.join(RUNS, 'data', 'S2_amp1.pt'))['test']['J_end']
    cJ1 = np.linalg.norm(curl_f(J1).numpy().reshape(J1.shape[0], -1), axis=1)
    ck = [('tcme_wired', 0, None)]
    for f in sorted(glob.glob(os.path.join(EDIR, 'E7_*_s0.pt'))):
        b = os.path.basename(f)[3:-3]; m, Nn = b.rsplit('_N', 1); ck.append((m, int(Nn.split('_')[0]), f))
    if not any(m == 'flux_gnn' and Nn == 64 for m, Nn, _ in ck):
        ck.append(('flux_gnn', 64, os.path.join(RUNS, 'tcme', 'S2_flux_gnn_s0.pt')))
    res = {}
    for m, Nn, f in ck:
        model = make(m)
        if f is not None:
            model.load_state_dict(torch.load(f, map_location=dev))
        model.eval(); key = f'{m}_N{Nn}'; res[key] = dict(model=m, N=Nn, ckpt=f)
        if m == 'tcme_wired':
            res[key].update(sA=float(model.sA), b=model.bh.detach().cpu().tolist())
        for sn, sd in sets.items():
            t0 = time.time()
            F = fields(sd); r = 1.0
            if sn.startswith('D'):
                r = float(sn[1:].replace('p', '.')) / D_TRAIN
            if m == 'tcme_wired':
                model.tscale, model.Ascale, model.kscale = 1.0 / r, 1.0 / r, r
            elif r != 1.0:
                F['U'] = F['U'] / r
            ctx = model.prepare(F)
            S = err_stats(rollout(model, ctx, sd['rho'][:, 0].to(dev)), sd['rho'].double())
            res[key][sn] = S
            if m == 'tcme_wired':
                model.tscale, model.Ascale, model.kscale = 1.0, 1.0, 1.0
            print(f'{key:16s} {sn:6s} 10T {S["err10T"]:.3e} T {S["errT"]:.3e} blown {S["n_blown"]} neg {S["n_neg"]} mass {S["mass_err"]:.1e} {time.time() - t0:.0f}s', flush=True)
        # counterfactual: drive off
        with torch.no_grad():
            if m == 'tcme_wired':
                ctx = model.prepare(fields(sets['ID'])); ctx['A'] = torch.zeros_like(ctx['A'])
            else:
                ctx = model.prepare(fields(cf))
            R = rollout(model, ctx, cf['rho'][:, 0].to(dev))
            S = err_stats(R, cf['rho'].double())
            fin = torch.isfinite(R[:, -1]).all(-1).all(-1)
            rend = torch.where(fin[:, None, None], R[:, -1], torch.ones_like(R[:, -1])).float().to(dev)
            Jm = flux_phys(model, rend, ctx).cpu()
            S['curlJ_over_curlJ1'] = float(np.median(np.linalg.norm(curl_f(Jm).numpy().reshape(Jm.shape[0], -1), axis=1) / cJ1))
        res[key]['CF'] = S
        print(f'{key:16s} CF     10T {S["err10T"]:.3e} curlJ/curlJ1 {S["curlJ_over_curlJ1"]:.2e}', flush=True)
        tj = os.path.join(EDIR, f'E7_{m}_N{Nn}_s0_train.json')
        if os.path.exists(tj):
            res[key]['train'] = json.load(open(tj))
    json.dump(res, open(os.path.join(ODIR, 'E7.json'), 'w'), indent=1)
    print('reference amp0 curl J / curl J1:', float(np.median(np.linalg.norm(curl_f(cf['J_end']).numpy().reshape(16, -1), axis=1) / cJ1)))


# ------------------------------------------------------------------ report
def report():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = json.load(open(os.path.join(ODIR, 'E7.json')))
    cols = [('ID', 'ID'), ('amp2', 'drift x2'), ('amp4', 'drift x4'), ('D0p01', 'D = 0.01 (T/2)'), ('D0p04', 'D = 0.04 (2T)')]
    M = [('tcme_wired', 'wired TC-ME'), ('flux_gnn', 'Flux-GNN')]
    Ns = sorted({v['N'] for v in res.values()})
    g = lambda m, Nn: res.get(f'{m}_N{Nn}')
    L = ['# E7: wired thermodynamics (learn only the kinetics) vs Flux-GNN, knob extrapolation x training-set size (S2)', '',
         'wired TC-ME: mu = log rho + U (fixed), A = s_A dpsi + harmonic (s_A init 1), k = (D/h^2) exp(kappa_theta(edge features)) '
         '(zero-initialised MLP, 32 hidden); N = 0 = untrained = SQRA discretisation. Flux-GNN: unchanged (models2.FluxGNN). '
         'Training: first N S2 training trajectories (20 steps each), 1 seed, train_tcme.py protocol (rollout loss K in [4, 8], Adam 1e-3 cosine, '
         'batch min(8, N)), 1500 iterations or a 40-min wall-clock cap (iterations actually done are listed). Flux-GNN N = 64 = the existing '
         '2500-iteration S2 checkpoint (runs/tcme/S2_flux_gnn_s0.pt). Test: the 16 S2 test_id trajectories, 10T relative L2 (median). '
         'Knobs: drift amplitude x2 / x4 (E6 data), D = 0.01 / 0.04 (G1 data; wired: k -> r k, phi, A -> / r; Flux-GNN: U -> U / r), '
         'counterfactual drive off (psi = 0 reference): wired A <- 0, Flux-GNN drive inputs = 0; steady ||curl J|| / ||curl J_true(psi on)|| at t = 10.', '']
    L += ['## 10T relative L2', '', '| N | ' + ' | '.join(f'{lab}: wired / Flux' for _, lab in cols) + ' | drive off: wired / Flux |', '|---|' + '---|' * (len(cols) + 1)]
    fmt = lambda x: f'{x:.2e}' if x is not None and np.isfinite(x) else 'inf'
    for Nn in Ns:
        cells = []
        for c, _ in cols + [('CF', '')]:
            w, f = g('tcme_wired', Nn), g('flux_gnn', Nn)
            cells.append(f'{fmt(w[c]["err10T"]) if w else "-"} / {fmt(f[c]["err10T"]) if f else "-"}')
        L.append(f'| {Nn} | ' + ' | '.join(cells) + ' |')
    L += ['', '## drive off: steady ||curl J|| / ||curl J_true(psi on)|| (the psi = 0 reference itself has 1.1e-3 at t = 10: transient curl, the steady value is 0)', '', '| N | wired TC-ME | Flux-GNN |', '|---|---|---|']
    for Nn in Ns:
        w, f = g('tcme_wired', Nn), g('flux_gnn', Nn)
        L.append(f'| {Nn} | {fmt(w["CF"]["curlJ_over_curlJ1"]) if w else "-"} | {fmt(f["CF"]["curlJ_over_curlJ1"]) if f else "-"} |')
    L += ['', '## blow-ups / trajectories with rho<0 / max mass error (all knobs pooled)', '', '| N | wired TC-ME | Flux-GNN |', '|---|---|---|']
    for Nn in Ns:
        cells = []
        for m, _ in M:
            v = g(m, Nn)
            if not v:
                cells.append('-'); continue
            ks = [c for c, _ in cols] + ['CF']
            cells.append(f'{sum(v[c]["n_blown"] for c in ks)} / {sum(v[c]["n_neg"] for c in ks)} / {max(v[c]["mass_err"] for c in ks):.1e}')
        L.append(f'| {Nn} | ' + ' | '.join(cells) + ' |')
    L += ['', '## training', '', '| model | N | iterations done / planned | train time (s) | final train loss | wired s_A, b |', '|---|---|---|---|---|---|']
    for k, v in sorted(res.items(), key=lambda kv: (kv[1]['model'], kv[1]['N'])):
        t = v.get('train', {})
        L.append(f'| {v["model"]} | {v["N"]} | {t.get("iters_done", "-")} / {t.get("iters_planned", "-")} | {t.get("train_time_s", float("nan")):.0f} | '
                 f'{t.get("final_train_loss") or float("nan"):.2e} | ' + (f'{v["sA"]:.3f}, {np.round(v["b"], 4).tolist()}' if 'sA' in v else '-') + ' |')
    L += ['', '![E7](E7_knobs.png)', '']
    open(os.path.join(ODIR, 'E7_summary.md'), 'w').write('\n'.join(L)); print('\n'.join(L))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    sty = dict(tcme_wired=('#1b6ca8', 'o-'), flux_gnn=('#3b8b5a', 's--'))
    for m, lab in M:
        c, mk = sty[m]
        NN = [Nn for Nn in Ns if g(m, Nn) and Nn > 0]
        # panel 1: error vs data size, per knob
        for ck, alpha in [('ID', 1.0), ('amp2', 0.7), ('amp4', 0.5), ('D0p01', 0.35), ('D0p04', 0.25)]:
            y = [g(m, Nn)[ck]['err10T'] for Nn in NN]
            axs[0].plot(NN, y, mk, color=c, alpha=alpha, lw=1.8, label=f'{lab}: {dict(cols)[ck]}')
        # panel 2: drift amplitude extrapolation at N = 64 and N = 4
        for Nn, alpha in [(64, 1.0), (4, 0.45)]:
            v = g(m, Nn)
            if v:
                axs[1].plot([1, 2, 4], [v['ID']['err10T'], v['amp2']['err10T'], v['amp4']['err10T']], mk, color=c, alpha=alpha, lw=2, label=f'{lab}, N={Nn}')
                axs[2].plot([0.5, 1, 2], [v['D0p01']['err10T'], v['ID']['err10T'], v['D0p04']['err10T']], mk, color=c, alpha=alpha, lw=2, label=f'{lab}, N={Nn}')
    w0 = g('tcme_wired', 0)
    if w0:
        axs[1].plot([1, 2, 4], [w0['ID']['err10T'], w0['amp2']['err10T'], w0['amp4']['err10T']], ':', color='#1b6ca8', lw=1.5, label='wired, N=0 (SQRA)')
        axs[2].plot([0.5, 1, 2], [w0['D0p01']['err10T'], w0['ID']['err10T'], w0['D0p04']['err10T']], ':', color='#1b6ca8', lw=1.5, label='wired, N=0 (SQRA)')
    axs[0].set_xscale('log', base=2); axs[0].set_xlabel('training trajectories N'); axs[0].set_title('10T error vs data size (each knob)')
    axs[1].set_xscale('log', base=2); axs[1].set_xlabel('drift amplitude multiple (training: 1)'); axs[1].set_title('drift extrapolation')
    axs[2].set_xscale('log', base=2); axs[2].set_xlabel('temperature multiple D/D_train'); axs[2].set_title('temperature extrapolation')
    for ax in axs:
        ax.set_yscale('log'); ax.grid(alpha=0.3, which='both'); ax.set_ylabel('10T relative L2')
    axs[0].legend(fontsize=6, ncol=2); axs[1].legend(fontsize=7); axs[2].legend(fontsize=7)
    fig.suptitle('E7: wired thermodynamics vs Flux-GNN (S2)'); fig.tight_layout()
    fig.savefig(os.path.join(ODIR, 'E7_knobs.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', action='store_true'); p.add_argument('--eval', action='store_true'); p.add_argument('--report', action='store_true')
    p.add_argument('--model', default='tcme_wired'); p.add_argument('--ntrain', type=int, default=64); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=1500); p.add_argument('--max_minutes', type=float, default=40)
    p.add_argument('--batch', type=int, default=8); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--kmin', type=int, default=4); p.add_argument('--kmax', type=int, default=8)
    a = p.parse_args()
    if a.train:
        train(a)
    if a.eval:
        evaluate()
    if a.report:
        report()
