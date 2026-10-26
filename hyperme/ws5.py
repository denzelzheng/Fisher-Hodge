"""WS5: minimum-dissipation control of a density with a time-dependent potential, on S2 without drift.

Task: move the density from rho_0 (final state of test_id trajectory 0) to rho_T (final state of trajectory 1) in
time tau with a potential U(t) on the 32 x 32 nodes (K = 20 time nodes, linear in between, held constant over each
sampling interval DT = 0.05), minimising the time-integrated entropy production, with the terminal state imposed
by an augmented Lagrangian.  Gradients go through the rollout (explicit positive substeps).

Lower bound.  For the SQRA dynamics a = k rho_i e^{(U_i-U_j)/2}, b = k rho_j e^{(U_j-U_i)/2} the product
a b = k^2 rho_i rho_j does not depend on U, so a flux J costs sigma_e = 2 J asinh(J / 2g), g = k sqrt(rho_i rho_j).
For slow protocols this is J^2 / g, and the minimum over all protocols is the discrete Benamou-Brenier energy with
the geometric-mean mobility, W^2 / tau.  It is reached by U = -psi - log rho, psi the Kantorovich potential of the
geodesic.  The exact cost lies below J^2 / g by O((J/g)^2), reported as max J/g on the geodesic.

Models whose protocols are optimised: the wired model untrained (the SQRA discretisation, used as the true system),
the wired model trained on 16 trajectories, and the black-box Flux-GNN (models2.FluxGNN, trained on 64), which has
no entropy production; it minimises the proxy sum_e J_e^2 / (k rho_mean,e) instead.  Every protocol is then run in
the true system, which reports its true entropy production and terminal error.
    python -m hyperme.ws5 --stage bound | opt --model wired0|wired16|fluxgnn | eval | inverse | plot | report
"""
import argparse, json, os, sys, time, importlib.util
import numpy as np
import torch
import torch.utils.checkpoint
from . import presets
from .complex import GridOps as G

OUT = os.path.join(presets.ROOT, 'runs', 'hyperme', 'ws5')
TAU, DT, K_NODES, N = 2.0, 0.05, 20, 32
NSTEP = int(round(TAU / DT))
D = 0.02
KR = D * N * N


def dev_():
    return 'cuda' if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' else 'cpu'


def endpoints():
    te = presets.load_s2()['test_id']
    r0, rT = te['c'][0, -1].double(), te['c'][1, -1].double()
    rT = rT * r0.sum() / rT.sum()
    return r0, rT


def incidence(dtype=torch.float64, device='cpu'):
    I = torch.eye(N * N, dtype=dtype, device=device).reshape(N * N, N, N)
    return G.flat_e(G.d0(I)).T.contiguous()                                  # (E, V): (D u)_e = u_head - u_tail


# ------------------------------------------------------------------------------------------ Benamou-Brenier bound
def bb_cost(x, r0, rT, Dm, M, nsteps, return_all=False):
    """x (nsteps-1, V) logits of the interior densities (mass fixed by a softmax).  Per interval the cheapest flux
    with the required divergence is J = m D psi with D^T m D psi = r, and its cost is r . psi."""
    from .complex import grid_edge_list
    E = torch.as_tensor(grid_edge_list(N), device=x.device)
    ti, hi = E[:, 0], E[:, 1]
    V = N * N
    rho = torch.cat([r0[None], M * torch.softmax(x, -1), rT[None]], 0)
    dt = TAU / nsteps
    r = (rho[1:] - rho[:-1]) / dt                                             # = D^T J
    rb = 0.5 * (rho[1:] + rho[:-1])
    m = KR * torch.sqrt(rb[:, ti] * rb[:, hi])                               # geometric-mean mobility (SQRA)
    nk = m.shape[0]
    L = torch.zeros(nk, V * V, dtype=x.dtype, device=x.device)
    L = L.index_add(1, ti * V + ti, m).index_add(1, hi * V + hi, m).index_add(1, ti * V + hi, -m).index_add(1, hi * V + ti, -m)
    L = L.view(nk, V, V) + torch.ones(V, V, dtype=x.dtype, device=x.device) / V
    psi = torch.linalg.solve(L, r.unsqueeze(-1)).squeeze(-1)
    cost = dt * (r * psi).sum()
    if return_all:
        J = m * (psi @ Dm.T)
        return cost, dict(rho=rho, rb=rb, psi=psi, J=J, m=m)
    return cost


def stage_bound():
    dev = dev_()
    r0, rT = (x.to(dev) for x in endpoints())
    Dm = incidence(device=dev)
    # order of the edge endpoints in the rows of Dm: tail has -1, head +1 (one of each per row)
    M = float(r0.sum())
    s = (torch.arange(1, NSTEP, dtype=torch.float64, device=dev) / NSTEP)[:, None]
    prev = os.path.join(OUT, 'bound.pt')
    if os.path.exists(prev):                                  # warm start from a previous run
        x = torch.log(torch.load(prev)['rho_geo'][1:-1].to(dev)).clone().requires_grad_(True)
        hist0 = torch.load(prev).get('hist', [])
    else:
        x = torch.log((1 - s) * r0 + s * rT).clone().requires_grad_(True)
        hist0 = []
    opt = torch.optim.LBFGS([x], lr=1, max_iter=50, history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-12,
                            tolerance_change=1e-15)
    hist = []

    def closure():
        opt.zero_grad()
        c = bb_cost(x, r0, rT, Dm, M, NSTEP)
        c.backward()
        return c
    xl = torch.log((1 - s) * r0 + s * rT)
    c_lin = float(bb_cost(xl, r0, rT, Dm, M, NSTEP))
    t0 = time.time()
    for it in range(40):
        c = float(opt.step(closure).detach()); hist.append(c)
        print(f'bound outer {it} cost {c:.6f}', flush=True)
        if it > 3 and abs(hist[-2] - hist[-1]) < 1e-9 * abs(hist[-1]):
            break
    hist = hist0 + hist
    with torch.no_grad():
        c, al = bb_cost(x, r0, rT, Dm, M, NSTEP, return_all=True)
        g = al['m']
        jg = float((al['J'].abs() / g).max())
        U_geo = -al['psi'] - torch.log(al['rb'])                               # mu = -psi, U = mu - log rho
        U_geo = U_geo - U_geo.mean(-1, keepdim=True)
    res = dict(W2_over_tau=float(c), linear_interp_cost=c_lin, outer_iters=len(hist), time_s=time.time() - t0, max_J_over_g=jg,
               mass=M, last_rel_change=abs(hist[-1] - hist[-2]) / hist[-1] if len(hist) > 1 else None)
    torch.save(dict(res, U_geo=U_geo.cpu(), rho_geo=al['rho'].detach().cpu(), r0=r0.cpu(), rT=rT.cpu(), hist=hist),
               os.path.join(OUT, 'bound.pt'))
    print(json.dumps(res), flush=True)


# ------------------------------------------------------------------------------------------ models and rollouts
def load_models(dev):
    w0 = presets.s2_wired().to(dev).eval()
    w16 = presets.s2_wired().to(dev)
    p = os.path.join(OUT, 's2_wired_n16_s0.pt')
    if os.path.exists(p):
        w16.load_state_dict(torch.load(p, map_location=dev))
    w16.eval()
    return w0, w16


def load_fluxgnn(dev):
    fh = os.path.join(presets.ROOT, 'fisher_hodge')
    if fh not in sys.path:
        sys.path.insert(0, fh)
    import models2
    m = models2.FluxGNN('S2').to(dev)
    m.load_state_dict(torch.load(os.path.join(presets.ROOT, 'runs', 'tcme', 'S2_flux_gnn_s0.pt'), map_location=dev))
    return m.eval()


def U_path(nodes):
    """K time nodes (K, V) -> potential held over each sampling interval (NSTEP, V), linear in time between nodes."""
    tm = (torch.arange(NSTEP, dtype=nodes.dtype, device=nodes.device) + 0.5) * DT / TAU * (K_NODES - 1)
    i0 = tm.floor().long().clamp(max=K_NODES - 2); w = (tm - i0)[:, None]
    return (1 - w) * nodes[i0] + w * nodes[i0 + 1]


def wired_drive(U, dev, dtype):
    V = N * N
    return dict(U=U.reshape(1, N, N), dpsi=torch.zeros(1, 2, N, N, dtype=dtype, device=dev),
                psin=torch.zeros(1, N, N, dtype=dtype, device=dev))


def sigma_wired(model, rho, ctx):
    a, b = model.rates(rho, ctx)
    return ((a - b) * (torch.log(a) - torch.log(b))).sum(-1)


def rollout(kind, model, Upath, r0, record=False):
    """returns final density, integrated cost (true entropy production for the wired models, the proxy for the
    Flux-GNN) and, with record=True, the density and sigma at every sampling time."""
    dev, dtype = r0.device, r0.dtype
    rho = r0.reshape(1, -1).to(dtype)
    cost = 0.0; rhos = [rho.detach()]; sig = []
    for kstep in range(NSTEP):
        F = wired_drive(Upath[kstep], dev, dtype)
        if kind == 'fluxgnn':
            ctx = model.prepare(F)
            r = rho.view(1, N, N)
            s0 = proxy(model, r, ctx)
            r, _ = model.advance(r, ctx, DT)
            s1 = proxy(model, r, ctx)
            rho = r.reshape(1, -1)
        else:
            ctx = model.prepare(F)
            s0 = sigma_wired(model, rho, ctx)
            rho, _ = model.advance(rho, ctx, DT)
            s1 = sigma_wired(model, rho, ctx)
        cost = cost + 0.5 * DT * (s0 + s1).sum()
        if record:
            rhos.append(rho.detach()); sig.append(0.5 * float((s0 + s1).sum()))
    return rho[0], cost, (torch.cat(rhos, 0) if record else None), sig


def proxy(model, r, ctx):
    """Flux-GNN: sum_e J_e^2 / (k rho_mean,e), J = -k q (its flux convention)."""
    J = -KR * model.flux_q(r, ctx)
    rm = G.mean_e(r)
    return (J * J / (KR * rm)).flatten(1).sum(-1)


def stage_opt(kind, iters=400, lr=0.05, resume=False):
    dev = dev_()
    torch.manual_seed(0)
    b = torch.load(os.path.join(OUT, 'bound.pt'))
    r0, rT = b['r0'].to(dev), b['rT'].to(dev)
    dtype = torch.float32
    if kind == 'fluxgnn':
        model = load_fluxgnn(dev)
    else:
        w0, w16 = load_models(dev)
        model = w0 if kind == 'wired0' else w16
    for p in model.parameters():
        p.requires_grad_(False)
    s = torch.linspace(0, 1, K_NODES, dtype=torch.float64, device=dev)[:, None]
    nodes0 = -torch.log((1 - s) * r0 + s * rT)                                # naive: interpolate the equilibrium potentials
    nodes = (nodes0 - nodes0.mean(-1, keepdim=True)).float().clone().requires_grad_(True)
    r0f, rTf = r0.float(), rT.float()
    bound = b['W2_over_tau']
    beta = bound / (0.01 * float(rTf.norm())) ** 2                               # 1 % terminal error costs bound / 2
    lam = torch.zeros_like(rTf)
    log = []
    pfile = os.path.join(OUT, f'protocol_{kind}.pt')
    if resume and os.path.exists(pfile):                                          # continue a previous run
        prev = torch.load(pfile)
        nodes = prev['nodes'].to(dev).clone().requires_grad_(True)
        lam = prev.get('lam', lam).to(dev); log = prev['log']
    opt = torch.optim.Adam([nodes], lr=lr)
    t0 = time.time()
    for it in range(iters):
        rho, cost, _, _ = rollout(kind, model, U_path(nodes), r0f)
        g = rho - rTf
        loss = cost + (lam * g).sum() + 0.5 * beta * (g * g).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            nodes -= nodes.mean(-1, keepdim=True)
        err = float(g.norm() / rTf.norm())
        if (it + 1) % 50 == 0:
            lam = lam + beta * g.detach()
        if it % 10 == 0 or it == iters - 1:
            log.append(dict(it=it, cost=float(cost), err=err, t=time.time() - t0))
            print(f'{kind} it {it} cost {float(cost):.4f} (bound {bound:.4f}) terminal err {err:.3e} t {time.time() - t0:.0f}s', flush=True)
    torch.save(dict(nodes=nodes.detach().cpu(), log=log, kind=kind, lam=lam.detach().cpu()), os.path.join(OUT, f'protocol_{kind}.pt'))


# ------------------------------------------------------------------------------------------ evaluation in the true system
@torch.no_grad()
def stage_eval():
    dev = dev_()
    b = torch.load(os.path.join(OUT, 'bound.pt'))
    r0, rT = b['r0'].to(dev), b['rT'].to(dev)
    w0, _ = load_models(dev)
    protos = {}
    s = torch.linspace(0, 1, K_NODES, dtype=torch.float64, device=dev)[:, None]
    nodes = -torch.log((1 - s) * r0 + s * rT)
    protos['naive interpolation'] = U_path(nodes)
    protos['quench'] = (-torch.log(rT))[None].expand(NSTEP, -1)
    protos['geodesic (Benamou-Brenier)'] = b['U_geo'].to(dev)
    for kind in ('wired0', 'wired16', 'fluxgnn'):
        p = os.path.join(OUT, f'protocol_{kind}.pt')
        if os.path.exists(p):
            protos[f'optimised on {kind}'] = U_path(torch.load(p)['nodes'].to(dev).double())
    res = {}
    for name, Up in protos.items():
        rho, cost, rhos, sig = rollout('wired', w0.double(), Up.double(), r0, record=True)
        res[name] = dict(sigma_total=float(cost), ratio_to_bound=float(cost) / b['W2_over_tau'],
                         terminal_err=float((rho - rT).norm() / rT.norm()), sigma_t=sig, min_rho=float(rhos.min()))
        key = {'naive interpolation': 'naive', 'quench': 'quench', 'geodesic (Benamou-Brenier)': 'geodesic'}.get(name, name.split()[-1])
        torch.save(dict(U=Up.float().cpu(), rho=rhos.float().cpu(), sigma=sig), os.path.join(OUT, f'traj_{key}.pt'))
        print(f'{name}: true sigma_total {float(cost):.4f} = {res[name]["ratio_to_bound"]:.3f} x bound, terminal err {res[name]["terminal_err"]:.3e}', flush=True)
    # shape of the optimised protocols against the geodesic one (correlation of mean-free potentials over space-time)
    Ug = b['U_geo'].to(dev).double()
    for kind in ('wired0', 'wired16', 'fluxgnn'):
        p = os.path.join(OUT, f'protocol_{kind}.pt')
        if os.path.exists(p):
            Uo = U_path(torch.load(p)['nodes'].to(dev).double())
            Uo = Uo - Uo.mean(-1, keepdim=True)
            res[f'optimised on {kind}']['corr_with_geodesic_U'] = float(torch.corrcoef(torch.stack([Uo.flatten(), Ug.flatten()]))[0, 1])
            res[f'optimised on {kind}']['rel_diff_to_geodesic_U'] = float((Uo - Ug).norm() / Ug.norm())
    res['bound'] = {k: v for k, v in b.items() if not torch.is_tensor(v)}
    json.dump(res, open(os.path.join(OUT, 'ws5.json'), 'w'), indent=1)


# ------------------------------------------------------------------------------------------ inverse design
def stage_inverse(iters=100, horizon=60):
    """target stationary density: two Gaussian bumps.  Wired (A = 0): U = -log rho* is exact.  Flux-GNN: optimise U
    so that its state after `horizon` sampling steps (3 time units) from a uniform start equals rho*, starting from
    U = 0; then compute
    the stationary state of that U in the true system."""
    dev = dev_()
    a, bq = torch.meshgrid(torch.arange(N, dtype=torch.float64), torch.arange(N, dtype=torch.float64), indexing='ij')
    def bump(ca, cb, w):
        da = torch.minimum((a - ca).abs(), N - (a - ca).abs()); db = torch.minimum((bq - cb).abs(), N - (bq - cb).abs())
        return torch.exp(-(da ** 2 + db ** 2) / (2 * w ** 2))
    rs = 0.15 + bump(9, 9, 3.0) + 0.7 * bump(22, 20, 4.0)
    rs = (rs / rs.mean()).to(dev).reshape(-1)
    w0, _ = load_models(dev)
    w0 = w0.double()
    res = {}
    with torch.no_grad():
        from .coarse import model_stationary
        ctx = w0.prepare(wired_drive(-torch.log(rs), dev, torch.float64))
        pi, _ = model_stationary(w0, ctx, torch.ones(1, N * N, dtype=torch.float64, device=dev))
        res['wired_exact_U'] = dict(rel_err=float((pi[0] - rs).norm() / rs.norm()))
    fg = load_fluxgnn(dev)
    for p in fg.parameters():
        p.requires_grad_(False)
    U = torch.zeros_like(rs).float().requires_grad_(True)                      # the black box gets no closed form: start flat
    opt = torch.optim.Adam([U], lr=0.05)
    rsf = rs.float()
    t0 = time.time()
    for it in range(iters):
        ctx = fg.prepare(wired_drive(U, dev, torch.float32))
        r = torch.ones(1, N, N, device=dev)
        for k in range(horizon):
            r = torch.utils.checkpoint.checkpoint(lambda rr: fg.advance(rr, ctx, DT)[0], r, use_reentrant=False)
        loss = ((r.reshape(-1) - rsf) ** 2).sum() / (rsf ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 25 == 0:
            print(f'inverse fluxgnn it {it} loss {float(loss):.3e} t {time.time() - t0:.0f}s', flush=True)
    with torch.no_grad():
        Uf = U.detach().double()
        ctx = w0.prepare(wired_drive(Uf, dev, torch.float64))
        pi, _ = model_stationary(w0, ctx, torch.ones(1, N * N, dtype=torch.float64, device=dev))
        res['fluxgnn_optimised_U'] = dict(model_err_at_horizon=float(loss) ** 0.5, true_stationary_rel_err=float((pi[0] - rs).norm() / rs.norm()),
                                          time_s=time.time() - t0)
        res['wired_exact_U']['true_stationary_rel_err'] = res['wired_exact_U']['rel_err']
    json.dump(res, open(os.path.join(OUT, 'inverse.json'), 'w'), indent=1)
    torch.save(dict(rs=rs.cpu(), U_fluxgnn=Uf.cpu()), os.path.join(OUT, 'inverse.pt'))
    print(json.dumps(res), flush=True)


PAL = dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', grid='#e4e3df', s1='#2a78d6', s2='#eb6834', s3='#1baf7a',
           s4='#eda100', mid='#f0efec', red='#e34948', blue_dark='#1c5cab')
ROWS = [('geodesic', 'Geodesic'), ('wired0', 'Wired, untrained'),
        ('wired16', 'Wired, 16 traj.'), ('fluxgnn', 'Flux-GNN')]


def stage_plot(assets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    seq = LinearSegmentedColormap.from_list('seq', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
    div = LinearSegmentedColormap.from_list('div', [PAL['blue_dark'], '#86b6ef', PAL['mid'], '#f29a99', PAL['red']])
    res = json.load(open(os.path.join(OUT, 'ws5.json')))
    b = torch.load(os.path.join(OUT, 'bound.pt'))
    rT = b['rT'].reshape(N, N).numpy()
    trajs = {k: torch.load(os.path.join(OUT, f'traj_{k}.pt')) for k, _ in ROWS if os.path.exists(os.path.join(OUT, f'traj_{k}.pt'))}
    rows = [(k, lab) for k, lab in ROWS if k in trajs]
    ui = [NSTEP // 4, NSTEP // 2, 3 * NSTEP // 4]
    ri = [0, NSTEP // 2, NSTEP]
    vmax_r = max(float(t['rho'].max()) for t in trajs.values())
    vU = max(float(t['U'][ui].abs().max()) for t in trajs.values())
    sall = np.concatenate([np.asarray(t['sigma']) for t in trajs.values()])
    slim = (0.0, 1.08 * float(sall.max()))                                   # one sigma axis for all rows
    fig = plt.figure(figsize=(17, 2.3 * len(rows) + 0.6), facecolor=PAL['surface'])
    gs = fig.add_gridspec(len(rows), 9, width_ratios=[1, 1, 1, 0.15, 1, 1, 1, 1, 2.2], wspace=0.08, hspace=0.25)
    for r, (k, lab) in enumerate(rows):
        t = trajs[k]
        for j, i in enumerate(ui):
            ax = fig.add_subplot(gs[r, j]); U = t['U'][i].reshape(N, N).numpy(); U = U - U.mean()
            ax.imshow(U, cmap=div, norm=TwoSlopeNorm(0, -vU, vU), origin='lower'); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'U at t = {(i + 0.5) * DT:.2f}', fontsize=9, color=PAL['ink2'])
            if j == 0:
                ax.set_ylabel(lab, fontsize=9, color=PAL['ink'])
        for j, i in enumerate(ri):
            ax = fig.add_subplot(gs[r, 4 + j]); ax.imshow(t['rho'][i].reshape(N, N).numpy(), cmap=seq, vmin=0, vmax=vmax_r, origin='lower')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'ρ at t = {i * DT:.1f}', fontsize=9, color=PAL['ink2'])
        ax = fig.add_subplot(gs[r, 7]); ax.imshow(rT, cmap=seq, vmin=0, vmax=vmax_r, origin='lower'); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title('target ρ_T', fontsize=9, color=PAL['ink2'])
        ax = fig.add_subplot(gs[r, 8])
        ax.set_facecolor(PAL['surface'])
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        tt = (np.arange(NSTEP) + 0.5) * DT
        if k != 'geodesic':
            ax.plot(tt, trajs['geodesic']['sigma'], color=PAL['ink2'], lw=1, ls=(0, (4, 3)), label='geodesic')
        name = {'geodesic': 'geodesic (Benamou-Brenier)'}.get(k, f'optimised on {k}')
        ax.plot(tt, t['sigma'], color=PAL['s1'] if k != 'fluxgnn' else PAL['s2'], lw=2,
                label=f'σ total {res[name]["sigma_total"]:.0f} = {res[name]["ratio_to_bound"]:.2f} × bound')
        ax.tick_params(labelsize=8, colors=PAL['ink2']); ax.grid(True, color=PAL['grid'], lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc='lower left')
        ax.set_ylim(*slim)
        if r == len(rows) - 1:
            ax.set_xlabel('t', fontsize=9, color=PAL['ink2'])
        ax.set_ylabel('true σ(t)', fontsize=9, color=PAL['ink2'])
    fig.savefig(os.path.join(assets, 'ws5_protocol.png'), dpi=150, facecolor=PAL['surface'], bbox_inches='tight')
    plt.close(fig)
    # ---- total entropy production of every protocol in the true system, relative to the bound
    names = [('geodesic (Benamou-Brenier)', 'geodesic'), ('optimised on wired0', 'wired, untrained'),
             ('optimised on wired16', 'wired, 16 traj.'), ('optimised on fluxgnn', 'Flux-GNN'),
             ('naive interpolation', 'naive interpolation'), ('quench', 'quench')]
    names = [(n, l) for n, l in names if n in res]
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=PAL['surface'])
    ax.set_facecolor(PAL['surface'])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    y = np.arange(len(names))[::-1]
    cols = [PAL['ink2'], PAL['s1'], PAL['s1'], PAL['s2'], '#9b9a95', '#9b9a95']
    for yi, (n, l), c in zip(y, names, cols):
        v = res[n]['ratio_to_bound']
        ax.barh(yi, v, color=c, height=0.6)
        ax.text(v + 0.02, yi, f'{v:.2f}  (terminal error {res[n]["terminal_err"]:.1e})', va='center', fontsize=8, color=PAL['ink'])
    ax.axvline(1.0, color=PAL['ink'], lw=1)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in names], fontsize=9, color=PAL['ink'])
    ax.set_xlabel('true ∫σ dt / (W² / τ)', fontsize=9, color=PAL['ink2'])
    ax.tick_params(colors=PAL['ink2'], labelsize=8)
    ax.set_xlim(0, max(res[n]['ratio_to_bound'] for n, _ in names) * 1.45)
    fig.tight_layout()
    fig.savefig(os.path.join(assets, 'ws5_bound.png'), dpi=160, facecolor=PAL['surface'])
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True); p.add_argument('--model', default='wired0')
    p.add_argument('--iters', type=int, default=400); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.05); p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    if dev_() == 'cuda':
        from .gpu_guard import require_free
        require_free('cuda')
        torch.cuda.set_per_process_memory_fraction(min(1.0, 10e9 / torch.cuda.get_device_properties(0).total_memory))   # <= 10 GB on a shared GPU
    if a.stage == 'bound':
        stage_bound()
    elif a.stage == 'opt':
        stage_opt(a.model, a.iters, a.lr, a.resume)
    elif a.stage == 'eval':
        stage_eval()
    elif a.stage == 'inverse':
        stage_inverse()
    elif a.stage == 'plot':
        stage_plot(os.path.join(presets.ROOT, 'assets'))
