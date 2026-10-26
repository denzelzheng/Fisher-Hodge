"""Common task C, our side: HyperME on the 16 x 16 x 2 density of the mixed-species particles.

Hypergraph: transport of species 0 and 1 along the edges of the periodic 16 x 16 grid (h = L / 16 = 1.25).
Wired mode with a non-local interaction family in the chemical potential,
    mu_s = log rho_s + sum_k sum_s' chi^k_{s s'} (W_k * rho_s'),     W_k: periodic Gaussian, widths 0 (on-site), 1, 2 bins
(chi need not be symmetric: chi_01 != chi_10 is the non-reciprocal coupling; then mu is not the gradient of any free
energy), SQRA kinetics k_s = k_ref exp(kappa_s) per species, WiredDrive with one learned coefficient per harmonic
cochain (a uniform drift per species and direction).  Learned: 12 chi + 2 kappa + 4 harmonic + s_A = 19 scalars.
Training: the library's default estimator HyperME.fit() (closed-form NNLS initial value of the log-rate table by
hyperme.linear_fit, then hyperme.train.train: rollout loss over K = 4..8 steps of dt = 1, Adam lr 1e-3 with cosine
schedule, batch 8, gradient clip 1) on overlapping 21-frame windows of the 3 training trajectories (the GNN's split).
python taskC_hyperme.py --work ~/ext_work/taskC"""
import os, sys, json, math, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.nn as nn
from hyperme import Hypergraph, HyperME
from hyperme.model import WiredKinetics, WiredDrive, WiredFreeEnergy, wired_featurizer
from fisher_hodge.ext import nonreciprocal_gnn as ng

NB, LBOX = 16, 20.0
H = LBOX / NB
EPS = 1e-2                        # density floor (particles per unit area; one particle in a bin is 0.64)
WIDTHS = (0.0, 1.0, 2.0)


def gauss_kernels(n=NB, widths=WIDTHS):
    """(K, V, V) symmetric periodic kernels, rows sum to 1; width 0 = identity"""
    ix = torch.arange(n, dtype=torch.float64)
    d = (ix[:, None] - ix[None, :]).abs()
    d = torch.minimum(d, n - d)
    out = []
    for w in widths:
        if w == 0:
            out.append(torch.eye(n * n, dtype=torch.float64)); continue
        g = torch.exp(-0.5 * (d / w) ** 2)
        W = torch.kron(g, g)                              # bins flattened as ix * n + iy
        out.append(W / W.sum(1, keepdim=True))
    return torch.stack(out)


class NonlocalFE(WiredFreeEnergy):
    """mu_s = log c_s + U_s / T + sum_k chi^k_{s s'} (W_k c_s')   (two species).  A WiredFreeEnergy (ideal part and
    potential wired) with the non-local family as its non-ideal correction, so HyperME treats the model as wired and
    fit() starts with the closed-form NNLS step on the log-rate table."""

    def __init__(s, hg, widths=WIDTHS):
        super().__init__(hg, 'none')
        s.register_buffer('W', gauss_kernels(int(round(math.sqrt(hg.V))), widths))
        s.chi = nn.Parameter(torch.zeros(len(widths), s.NS, s.NS, dtype=torch.float64))

    @property
    def c_dependent(s):
        return True

    def correction(s, c, ctx):
        cs = c.view(c.shape[0], s.NS, s.V)
        Wc = torch.einsum('kuv,bsv->bksu', s.W.to(c.dtype), cs)              # (B, K, NS, V)
        return torch.einsum('kst,bktv->bsv', s.chi.to(c.dtype), Wc).reshape(c.shape)

    def phi(s, c, ctx):
        return ctx['Ufull'] + s.correction(c, ctx)

    def value(s, c, ctx):
        """free energy of the symmetric part of chi (exact only when chi is symmetric)"""
        cs = c.view(c.shape[0], s.NS, s.V)
        chis = 0.5 * (s.chi + s.chi.transpose(1, 2)).to(c.dtype)
        Wc = torch.einsum('kuv,bsv->bksu', s.W.to(c.dtype), cs)
        return (c * (torch.log(c) - 1 + ctx['Ufull'])).sum(-1) + 0.5 * torch.einsum('bsv,kst,bktv->b', cs, chis, Wc)


def build_model(k_ref=None):
    hg = Hypergraph.grid_transport(NB, n_species=2)
    torch.set_default_dtype(torch.float64)
    feat = wired_featurizer(hg)
    kin = WiredKinetics(feat.n_edge_sym, hg=hg, k_ref=k_ref if k_ref is not None else 1.0 / H ** 2, table=True)
    dr = WiredDrive(hg, harmonic=True)
    fe = NonlocalFE(hg)
    m = HyperME(hg, fe, kin, dr, feat)
    torch.set_default_dtype(torch.float32)
    return m.double()


def drive(B, dev):
    return dict(U=torch.zeros(B, 2 * NB * NB, dtype=torch.float64, device=dev), T=torch.ones(B, dtype=torch.float64, device=dev))


def step(model, c, Dt=1.0, dt_max=0.05, theta=0.5, flux=False, max_sub=20000):
    """SSP-RK2 (Heun form) with positivity-limited substeps, dt <= min(dt_max, theta / escape ratio) for the whole
    batch (the rule of hyperme.integrators.positive_ssp_rk2); returns c(t + Dt) and, with flux=True, the
    time-averaged hyperedge flux over the step (B, H)"""
    ctx = model.prepare(drive(c.shape[0], c.device))
    Jacc = torch.zeros(c.shape[0], model.hg.H, dtype=c.dtype, device=c.device)
    t, n = 0.0, 0
    while Dt - t > 1e-12 and n < max_sub:
        a0, b0 = model.rates(c, ctx)
        r0 = float(model.escape_ratio(c, a0, b0).max())
        dt = min(dt_max, Dt - t, theta / r0 if r0 > 0 else dt_max)
        while True:
            c1 = c + dt * model.hg.apply_S(a0 - b0)
            if float(c1.min()) > 0:
                a1, b1 = model.rates(c1, ctx)
                if dt * float(model.escape_ratio(c1, a1, b1).max()) <= theta:
                    break
            dt *= 0.5
        Jbar = 0.5 * ((a0 - b0) + (a1 - b1))
        c = c + dt * model.hg.apply_S(Jbar)
        Jacc = Jacc + dt * Jbar
        t += dt; n += 1
    return (c, Jacc / Dt) if flux else c


def density_series(d, idx):
    """binned densities (T+1, M, 2*NB*NB) + EPS, float64, flattened as [species, ix, iy]"""
    rho = ng.bin_density(d['x'][:, idx], d['c'][idx][None].expand(d['x'].shape[0], -1, -1), NB, LBOX)
    return rho.reshape(rho.shape[0], rho.shape[1], -1).double() + EPS


def edge_to_faces(J):
    """grid_transport edge flux (B, 2 * 2V) -> Jx, Jy (B, 2, NB, NB) in the [species, ix, iy] layout of
    nonreciprocal_gnn.face_flux.  The grid edge order is that of fisher_hodge/common.py: per species first the V
    edges (i, j) -> (i, j + 1) (second axis, iy), then the V edges (i, j) -> (i + 1, j) (first axis, ix).
    Units: density per unit time per edge; multiply by h^2 for particles per unit time."""
    B = J.shape[0]
    V = NB * NB
    J = J.view(B, 2, 2, NB, NB)
    return J[:, :, 1], J[:, :, 0]


def windows(R, length=21, stride=10):
    """(T+1, M, V) density series -> (B, length, V) overlapping windows: the default fit() / train() samples rollout
    windows from the first T_TRAIN + 1 = 21 frames of every training sequence"""
    out = []
    for m in range(R.shape[1]):
        for t in range(0, R.shape[0] - length + 1, stride):
            out.append(R[t:t + length, m])
    return torch.stack(out)


@torch.no_grad()
def rollout_loss(model, va, dt=1.0, K=6):
    """the training loss of hyperme.train.train (relative squared error averaged over K rollout steps) on validation
    windows, from every window start that leaves K steps"""
    C = va['c']
    ctx = model.prepare(va['drive'])
    tot, n = 0.0, 0
    for st in range(0, C.shape[1] - K, 4):
        c = C[:, st]
        for k in range(1, K + 1):
            c, _ = model.advance(c, ctx, dt)
            y = C[:, st + k]
            tot += float((((c - y) ** 2).sum(-1) / (y ** 2).sum(-1)).mean()); n += 1
    return tot / n


def fit_auto(model, data, va, iters, dt, seed, pr, probe=50):
    """candidate estimator for the library (not in hyperme yet): build both initial values (closed-form NNLS and the
    default one), train each for `probe` iterations with the default rollout training, keep the one with the lower
    validation rollout loss and run the default fit() with that choice for the full budget."""
    import copy
    from hyperme.train import train
    from hyperme.linear_fit import fit_linear
    init = copy.deepcopy(model.state_dict())
    scores = {}
    for lin in (True, False):
        model.load_state_dict(init)
        if lin:
            fit_linear(model, data, dt, log=pr)
        train(model, data['train'], probe, seed, dt, log=lambda *x: None)
        scores[lin] = rollout_loss(model, va, dt)
        pr(f'probe linear_init={lin}: validation rollout loss after {probe} it = {scores[lin]:.5f}')
    choice = min(scores, key=scores.get)
    model.load_state_dict(init)
    info = model.fit(data, iters, dt=dt, seed=seed, log=pr, linear_init=choice)
    info['auto'] = dict(probe=probe, scores={str(k): v for k, v in scores.items()}, chosen_linear_init=choice)
    return info


def onestep_train(model, Rtr, Rva, iters, seed, pr, lr=0.02, batch=60):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T = Rtr.shape[0] - 1
    g = torch.Generator().manual_seed(seed)

    def loss_on(R, m, t):
        pred = step(model, R[t, m]); tgt = R[t + 1, m]
        return ((pred - tgt) ** 2).sum() / (tgt ** 2).sum()
    best, best_state, t0 = float('inf'), None, time.time()
    mv = torch.arange(Rva.shape[1]).repeat_interleave(T); tv = torch.arange(T).repeat(Rva.shape[1])
    for it in range(iters):
        m = torch.randint(Rtr.shape[1], (batch,), generator=g); t = torch.randint(T, (batch,), generator=g)
        loss = loss_on(Rtr, m, t)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 or it == iters - 1:
            with torch.no_grad():
                vl = float(np.mean([loss_on(Rva, mv[k:k + 150], tv[k:k + 150]).item() for k in range(0, len(mv), 150)]))
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
            pr(f'it {it} train {loss.item():.5f} valid {vl:.5f} best {best:.5f} {time.time() - t0:.0f}s')
    model.load_state_dict(best_state)
    return dict(best_valid=best, train_time_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.path.expanduser('~/ext_work/taskC'))
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_linear_init', action='store_true', help='skip the closed-form NNLS first step of fit()')
    ap.add_argument('--tag', default='')
    ap.add_argument('--auto', action='store_true', help='fit() with an automatic choice of the initial value (closed-form '
                    'NNLS or the default), by the validation rollout loss after a short probe training')
    ap.add_argument('--onestep', action='store_true', help='sensitivity variant: one-step (tau = 1) loss on all frames, '
                    'Adam lr 0.02, batch 60, selection on the validation trajectories (no fit())')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    split = json.load(open(os.path.join(a.work, 'split.json')))
    d = torch.load(os.path.join(a.work, 'train.pt'))
    C = windows(density_series(d, split['train'])).to(dev)                 # the GNN's 3 training trajectories
    data = dict(train=dict(c=C, drive=drive(C.shape[0], dev)))
    model = build_model().to(dev)
    log = open(os.path.join(a.work, f'hyperme_train{a.tag}.log'), 'w')
    pr = lambda *x: print(*x, file=log, flush=True)
    pr(f'training windows {tuple(C.shape)}')
    if a.auto:
        Cva = windows(density_series(d, split['valid'])).to(dev)
        info = fit_auto(model, data, dict(c=Cva, drive=drive(Cva.shape[0], dev)), a.iters, 1.0, a.seed, pr)
    elif a.onestep:
        info = onestep_train(model, density_series(d, split['train']).to(dev), density_series(d, split['valid']).to(dev), a.iters, a.seed, pr)
    else:
        info = model.fit(data, a.iters, dt=1.0, seed=a.seed, log=pr, linear_init=not a.no_linear_init)            # default estimator: fit_linear, then rollout training
    pr(json.dumps({k: v for k, v in info.items() if k != 'loss_curve'}, default=str)[:3000])
    pr(f'logk {model.kinetics.core.tolist()} chi {model.free_energy.chi.detach().cpu().numpy().round(3).tolist()} '
       f'b {model.drive.b.detach().cpu().numpy().round(3).tolist()} s_A {float(model.drive.s_A):.3f}')
    with torch.no_grad():                                                     # validation loss (same 3 trajectories as the GNN), one step
        Rva = density_series(d, split['valid']).to(dev)
        pred = torch.cat([step(model, Rva[t]) for t in range(Rva.shape[0] - 1)])
        tgt = Rva[1:].reshape(-1, Rva.shape[-1])
        vl = float(((pred - tgt) ** 2).sum() / (tgt ** 2).sum())
    pr(f'valid one-step relative MSE {vl:.5f}')
    torch.save(dict(state=model.state_dict(), valid=vl, iters=a.iters, info={k: v for k, v in info.items()}),
               os.path.join(a.work, f'hyperme{a.tag}.pt'))


if __name__ == '__main__':
    main()
