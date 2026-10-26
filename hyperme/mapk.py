"""WS6a: a MAPK cascade with the Huang-Ferrell structure, on a 4 x 4 grid.

Three tiers, dual phosphorylation of MAPKK and MAPK, every enzymatic step written as binding E + S <-> ES followed by
catalysis ES (+ ATP) <-> E + P (+ ADP or Pi); mass action throughout; ATP, ADP and Pi are chemostats.  The rate
constants are dimensionless example values chosen so that the fastest and the slowest processes are within a factor
of about 1e2, and the enzyme concentrations are O(0.1 - 1).  They are NOT the parameters of BIOMD0000000009.
The input kinase E1 sits at the 4 corner sites only; MAPK in its three forms diffuses, so the signal spreads.

    python -m hyperme.mapk --stage data              # truth: ODE trajectories (train / test_id / test_atp_hi / test_atp_lo / test_init)
    python -m hyperme.mapk --stage train --model wired|flexible|crnn_chem|crnn|gnn [--seed 0]
    python -m hyperme.mapk --stage gillespie         # one site, Omega = 200, truth vs the learned rates
    python -m hyperme.mapk --stage figure | report
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as Fn
from . import presets, baselines as BL
from .complex import Hypergraph
from .model import HyperME, Context

OUT = os.path.join(presets.ROOT, 'runs', 'hyperme', 'mapk')
N, DT, T_TRAIN, T_TEST = 4, 0.5, 20, 200
STATE = ['KKK', 'KKKp', 'E1', 'E2', 'KK', 'KKp', 'KKpp', 'P2', 'K', 'Kp', 'Kpp', 'P3',
         'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
CHEM = ['ATP', 'ADP', 'Pi']
# (enzyme, substrate, complex, product, kinase?)
STEPS = [('E1', 'KKK', 'C1', 'KKKp', True), ('E2', 'KKKp', 'C2', 'KKK', False),
         ('KKKp', 'KK', 'C3', 'KKp', True), ('KKKp', 'KKp', 'C4', 'KKpp', True),
         ('P2', 'KKpp', 'C5', 'KKp', False), ('P2', 'KKp', 'C6', 'KK', False),
         ('KKpp', 'K', 'C7', 'Kp', True), ('KKpp', 'Kp', 'C8', 'Kpp', True),
         ('P3', 'Kpp', 'C9', 'Kp', False), ('P3', 'Kp', 'C10', 'K', False)]
DIFFUSING = ['K', 'Kp', 'Kpp']
A_BIND, D_UNBIND, K_KIN, K_PHOS, EPS, D_DIFF = 10.0, 2.0, 2.0, 1.0, 1e-3, 0.5
TOTALS = dict(KKK=1.0, KK=1.0, K=1.0, E2=0.2, P2=0.3, P3=0.3)
INPUT_SITES = [0, 1, 4, 5]                       # the 2 x 2 corner


def reactions():
    rx, kf, kb = [], [], []
    for e, s_, c, p, kin in STEPS:
        rx.append(({e: 1, s_: 1}, {c: 1})); kf.append(A_BIND); kb.append(D_UNBIND)
        if kin:
            rx.append(({c: 1, 'ATP': 1}, {e: 1, p: 1, 'ADP': 1})); kf.append(K_KIN); kb.append(EPS * K_KIN)
        else:
            rx.append(({c: 1}, {e: 1, p: 1, 'Pi': 1})); kf.append(K_PHOS); kb.append(EPS * K_PHOS)
    return rx, np.array(kf), np.array(kb)


def build_hg(n=N, n_sites=None):
    rx, _, _ = reactions()
    if n_sites == 1:
        return Hypergraph.reaction_transport(STATE + CHEM, rx, n_sites=1, chemostats=CHEM)
    return Hypergraph.reaction_transport(STATE + CHEM, rx, grid_n=n, diffusing=DIFFUSING, chemostats=CHEM)


def true_constants(hg):
    """per-hyperedge forward / backward rate constants (chemostats excluded)."""
    _, kf, kb = reactions()
    R, V = len(kf), hg.V
    KF = np.concatenate([np.repeat(kf, V), np.full(hg.H - R * V, D_DIFF)])
    KB = np.concatenate([np.repeat(kb, V), np.full(hg.H - R * V, D_DIFF)])
    return KF, KB


def chem_factors(hg, logc):
    """forward / backward chemostat mass-action factors (B, H) from log chemostat concentrations (B, 3)."""
    Y = torch.as_tensor(hg.meta['chem_S'].toarray(), dtype=logc.dtype, device=logc.device)
    lcv = logc.repeat_interleave(hg.V, 1)
    return torch.exp(lcv @ (-Y).clamp_min(0).T), torch.exp(lcv @ Y.clamp_min(0).T)


def true_rates(hg, logc):
    KF, KB = (torch.as_tensor(x, dtype=logc.dtype, device=logc.device) for x in true_constants(hg))
    ff, fb = chem_factors(hg, logc)
    pa, pb = KF[None] * ff, KB[None] * fb
    return lambda c: (pa * hg.mass_action(c, '+'), pb * hg.mass_action(c, '-'))


# ------------------------------------------------------------------------------------------ data
def initial_state(rng, hg, kind):
    V = hg.V
    c = {s: np.zeros(V) for s in STATE}
    e1 = np.full(V, 1e-4); e1[INPUT_SITES] = 0.3 * rng.uniform(0.5, 1.5, len(INPUT_SITES))
    c['E1'] = e1
    for e in ('E2', 'P2', 'P3'):
        c[e] = TOTALS[e] * rng.uniform(0.8, 1.2, V)
    for tier, forms in (('KKK', ['KKK', 'KKKp']), ('KK', ['KK', 'KKp', 'KKpp']), ('K', ['K', 'Kp', 'Kpp'])):
        alpha = np.array([6.0] + [1.0] * (len(forms) - 1)) if kind != 'init' else np.array([1.0] * (len(forms) - 1) + [6.0])
        fr = rng.dirichlet(alpha, size=V)
        for j, f in enumerate(forms):
            c[f] = TOTALS[tier] * rng.uniform(0.8, 1.2, V) * fr[:, j]
    for cx in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'):
        c[cx] = np.full(V, 1e-3)
    x = np.concatenate([c[s] for s in STATE])
    return np.maximum(x, 1e-6)


def solve(hg, logc, x0, nsteps):
    from scipy.integrate import solve_ivp
    rates = true_rates(hg, torch.tensor(logc[None], dtype=torch.float64))
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
    ts = np.arange(nsteps + 1) * DT
    sol = solve_ivp(f, (0, ts[-1]), x0, method='LSODA', t_eval=ts, rtol=1e-8, atol=1e-11, jac=jac)
    assert sol.success, sol.message
    C = sol.y.T
    a, b = rates(torch.tensor(C))
    a, b = a.numpy(), b.numpy()
    sig = ((a - b) * (np.log(a) - np.log(b))).sum(-1)
    kin = kinase_cat_mask(hg)
    turnover = ((a - b)[:, kin]).sum(-1)                 # ATP consumed per unit time (all sites)
    return C, sig, turnover


def kinase_cat_mask(hg):
    Y = hg.meta['chem_S'].toarray()
    return (Y[:, :hg.V] < 0).any(1)                      # hyperedges that consume ATP


def stage_data(seed=0):
    os.makedirs(OUT, exist_ok=True)
    hg = build_hg()
    rng = np.random.default_rng(seed)
    specs = [('train', 48, (0.5, 2.0), 'id', T_TRAIN), ('test_id', 16, (0.5, 2.0), 'id', T_TEST),
             ('test_atp_hi', 16, (3.0, 5.0), 'id', T_TEST), ('test_atp_lo', 16, (0.15, 0.3), 'id', T_TEST),
             ('test_init', 16, (0.5, 2.0), 'init', T_TEST)]
    data = {}
    t0 = time.time()
    for name, nb, (lo, hi), kind, ns in specs:
        out = dict(c=[], logc=[], sigma=[], turnover=[])
        for _ in range(nb):
            atp = np.exp(rng.uniform(np.log(lo), np.log(hi)))
            logc = np.log(np.array([atp, 1.0, 1.0]))
            x0 = initial_state(rng, hg, kind)
            C, sig, tov = solve(hg, logc, x0, ns)
            out['c'].append(C); out['logc'].append(logc); out['sigma'].append(sig); out['turnover'].append(tov)
        data[name] = {k: torch.tensor(np.array(v)) for k, v in out.items()}
        print(f'{name}: c {tuple(data[name]["c"].shape)} min {float(data[name]["c"].min()):.2e} '
              f'Kpp frac end {float(kpp_fraction(data[name]["c"][:, -1]).mean()):.3f} turnover end {float(data[name]["turnover"][:, -1].mean()):.3f} '
              f't {time.time() - t0:.0f}s', flush=True)
    S = hg.S; L = hg.L
    data['meta'] = dict(dim_kerS=int(hg.Z.shape[1]), dim_kerST=int(L.shape[1]), H=hg.H, M=hg.M,
                        full_rank=int(np.linalg.matrix_rank(np.concatenate([S, hg.meta['chem_S'].toarray().T], 0))))
    print(data['meta'])
    torch.save(data, os.path.join(OUT, 'data.pt'))


def kpp_fraction(c):
    """fraction of MAPK in the doubly phosphorylated form (free Kpp plus Kpp in C9), per site: (..., V)"""
    V = N * N
    sl = lambda nm: c[..., STATE.index(nm) * V:(STATE.index(nm) + 1) * V]
    tot = sl('K') + sl('Kp') + sl('Kpp') + sl('C7') + sl('C8') + sl('C9') + sl('C10')
    return (sl('Kpp') + sl('C9')) / tot


# ------------------------------------------------------------------------------------------ models
def featurizer(hg, knob_features):
    """drive = dict(logc (B, 3) log concentrations of ATP, ADP, Pi).  knob_features=False (wired): the kinetics sees
    only the hyperedge type; True (the other models): also the chemostat force along each hyperedge and log ATP."""
    ntyp = len(STEPS) * 2 + len(DIFFUSING)

    def feat(drive):
        lc = drive['logc']
        B, dt, dev = lc.shape[0], lc.dtype, lc.device
        t = hg.on(dev, dt)
        force = hg.chemostat_force(lc)
        cols = [Fn.one_hot(t['typ'], ntyp).to(dt).expand(B, -1, -1)]
        if knob_features:
            cols.append(force.abs()[..., None])
        return dict(U=torch.zeros(B, hg.V, dtype=dt, device=dev), T=torch.ones(B, dtype=dt, device=dev), line=None,
                    chem_logc=lc, chem_mu0=None, edge_sym=torch.cat(cols, -1), edge_ori=force if knob_features else torch.zeros_like(force),
                    force=force, node=dict(logATP=lc[:, :1].expand(B, hg.V)) if knob_features else {}, raw=drive)
    feat.n_edge_sym = ntyp + (1 if knob_features else 0)
    return feat


class CRNNPos(BL.CRNN):
    """CRNN with the positivity-preserving substeps of HyperME (same integrator for every mass-action model)."""
    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        s.step = dict(dt_max=DT / 8, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9)

    def advance(s, c, drive, Dt, **kw):
        from .integrators import positive_ssp_rk2
        cfg = dict(s.step); cfg.update({k: v for k, v in kw.items() if k == 'vectorized'})
        return positive_ssp_rk2(s, c, s.context(drive), Dt, **cfg)


class GNNCk(BL.GNNODE):
    """GNN-ODE with activation checkpointing over each sampling step (memory on a shared GPU)."""
    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        if torch.is_grad_enabled():
            import torch.utils.checkpoint as ck
            return ck.checkpoint(lambda x: BL.heun_fixed(s, x, ctx, Dt, s.nsub)[0], c, use_reentrant=False), s.nsub
        return BL.heun_fixed(s, c, ctx, Dt, s.nsub)


def make_model(kind, hg=None):
    hg = hg or build_hg()
    step = dict(dt_max=DT / 8, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9)
    with presets.default_dtype(torch.float64):
        if kind == 'wired':
            m = HyperME.build_wired(hg, hidden=32, k_ref=1.0, T_exponent=0.0, featurizer=featurizer(hg, False), step=step,
                                    learn_mu0=True, learn_chem_mu0=True, harmonic=False, kinetics_table=True)
        elif kind == 'flexible':
            m = HyperME.build_flexible(hg, hidden=64, node_inputs=('logATP',), featurizer=featurizer(hg, True), step=step,
                                       force_gain_init=None)
        elif kind in ('crnn', 'crnn_chem'):
            f = featurizer(hg, kind == 'crnn')
            m = CRNNPos(hg, f.n_edge_sym + (1 if kind == 'crnn' else 0), f, hidden=64, oriented=kind == 'crnn', chem=kind == 'crnn_chem',
                        table=kind == 'crnn_chem')
        elif kind == 'gnn':
            f = featurizer(hg, True)
            ex = dict(logc=torch.zeros(1, 3, dtype=torch.float64))
            nn_in, ne_in = BL._Base.n_node_in(hg, f, ex)
            m = GNNCk(hg, nn_in, ne_in, f, hidden=64, nsub=10)
        else:
            raise ValueError(kind)
    return m.double()


def load_data(device='cpu'):
    d = torch.load(os.path.join(OUT, 'data.pt'))
    return {k: (dict(c=v['c'].double().to(device), drive=dict(logc=v['logc'].double().to(device)), sigma=v['sigma'], turnover=v['turnover'])
                if k != 'meta' else v) for k, v in d.items()}


@torch.no_grad()
def evaluate(model, data, hg):
    from .train import relerr
    res = {}
    L = hg.on('cpu', torch.float64)['L']
    kin = torch.as_tensor(kinase_cat_mask(hg))
    has_rates = hasattr(model, 'rates')
    for name in ('test_id', 'test_atp_hi', 'test_atp_lo', 'test_init'):
        sd = data[name]; Y = sd['c']
        ctx = model.prepare(sd['drive'])
        c = Y[:, 0].clone(); R = [c]; sig = []; tov = []; t0 = time.time(); nsub = []
        for k in range(Y.shape[1] - 1):
            if has_rates:
                a, b = model.rates(c, ctx)
                sig.append(((a - b) * (torch.log(a) - torch.log(b))).sum(-1)); tov.append((a - b)[:, kin].sum(-1))
            c, n = model.advance(c, ctx, DT)
            nsub.append(float(n.float().mean()) if torch.is_tensor(n) else float(n))
            bad = ~torch.isfinite(c).all(-1)
            c = torch.where(bad[:, None], torch.full_like(c, float('nan')), c)
            R.append(c)
        R = torch.stack(R, 1)
        e = relerr(R, Y).numpy()
        q = R @ L
        cons = (torch.linalg.vector_norm(q - q[:, :1], dim=-1) / torch.linalg.vector_norm(q[:, :1], dim=-1))
        S = dict(err_T=float(np.nanmedian(e[:, 20])), err_10T=float(np.nanmedian(e[:, -1])), n_blown=int(np.isnan(e[:, -1]).sum()),
                 min_c=float(np.nanmin(R.numpy())), n_neg=int((R < 0).any(-1).any(-1).sum()), cons_max=float(np.nanmax(cons.numpy())),
                 kpp_end_err=float(np.nanmedian(np.abs(kpp_fraction(R[:, -1]) - kpp_fraction(Y[:, -1])).mean(-1).numpy())),
                 substeps=float(np.mean(nsub)), wall_s=time.time() - t0)
        if has_rates:
            sg = torch.stack(sig, 1).numpy(); st = sd['sigma'][:, :-1].numpy()
            tv = torch.stack(tov, 1).numpy(); tt = sd['turnover'][:, :-1].numpy()
            S['sigma_relerr'] = float(np.nanmedian(np.linalg.norm(sg - st, axis=1) / np.linalg.norm(st, axis=1)))
            S['sigma_min'] = float(np.nanmin(sg))
            S['turnover_end_relerr'] = float(np.nanmedian(np.abs(tv[:, -1] - tt[:, -1]) / np.abs(tt[:, -1])))
        res[name] = S
    return res


def warm_start(model, tr, iters, log, lr=2e-2):
    """one-step derivative matching: dc/dt of the model at the midpoint of consecutive snapshots against the finite
    difference (c_{t+1} - c_t) / DT, relative squared error, full batch, no integrator.  Uses only training data."""
    C = tr['c']; B, T1, M = C.shape
    mid = 0.5 * (C[:, 1:] + C[:, :-1]); tgt = (C[:, 1:] - C[:, :-1]) / DT
    drive = {k: v.repeat_interleave(T1 - 1, 0) for k, v in tr['drive'].items()}
    x, y = mid.reshape(-1, M), tgt.reshape(-1, M)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for it in range(iters):
        ctx = model.prepare(drive)
        loss = (((model.rhs(x, ctx) - y) ** 2).sum(-1) / (y ** 2).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if it % 200 == 0 or it == iters - 1:
            log(f'warm {it:5d} derivative loss {float(loss):.4e} t {time.time() - t0:.0f}s')


def stage_train(kind, seed=0, iters=2000, device='cpu', lr=1e-3, tag='', warm=0):
    from .train import train
    torch.manual_seed(seed); np.random.seed(seed)
    hg = build_hg()
    model = make_model(kind, hg).to(device)
    data = load_data(device)
    name = f'mapk_{kind}{tag}_s{seed}'
    logf = open(os.path.join(OUT, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    log(f'{name}: params={sum(p.numel() for p in model.parameters())}')
    if warm:
        warm_start(model, data['train'], warm, log)
    info = train(model, data['train'], iters, seed, DT, lr=lr, log=log)
    torch.save(model.state_dict(), os.path.join(OUT, name + '.pt'))
    model.eval()
    model = model.cpu(); data = load_data('cpu')
    ev = evaluate(model, data, hg)
    for k, v in ev.items():
        log(f'[{k}] ' + ' '.join(f'{a} {b:.3e}' if isinstance(b, float) else f'{a} {b}' for a, b in v.items()))
    json.dump(dict(name=name, kind=kind, seed=seed, iters=iters, lr=lr, params=sum(p.numel() for p in model.parameters()), **info, sets=ev),
              open(os.path.join(OUT, name + '.json'), 'w'), indent=1)


MODELS = ['wired', 'flexible', 'crnn_chem', 'crnn', 'gnn']
LABEL = dict(truth='truth', wired='HyperME wired', flexible='HyperME flexible', crnn_chem='CRNN-chem (oracle class)',
             crnn='CRNN (ATP as input feature)', gnn='GNN-ODE')
SETS = ['test_id', 'test_atp_hi', 'test_atp_lo', 'test_init']


TAGS = ['', '_lr2e-2', '_lr1e-2']          # learning rate 1e-3 (no tag), 2e-2, 1e-2


def best_name(kind):
    """the run of this model with the lower final training loss (never chosen on test data)."""
    runs = []
    for t in TAGS:
        p = os.path.join(OUT, f'mapk_{kind}{t}_s0.json')
        if os.path.exists(p):
            r = json.load(open(p)); runs.append((r.get('final_train_loss') or float('inf'), f'mapk_{kind}{t}_s0'))
    return min(runs)[1] if runs else None


def load_model(kind, hg=None):
    m = make_model(kind, hg)
    m.load_state_dict(torch.load(os.path.join(OUT, best_name(kind) + '.pt'), map_location='cpu'))
    return m.eval()


def local_sigma(hg, a, b):
    """entropy production of the reactions at every site (transport excluded): (B, V)"""
    V = hg.V; R = len(STEPS) * 2
    s = ((a - b) * (torch.log(a) - torch.log(b)))[:, :R * V]
    return s.view(-1, R, V).sum(1)


@torch.no_grad()
def stage_predict(kinds):
    """roll out every model on every test set; save the per-site Kpp fraction over time and the final local EPR."""
    hg = build_hg(); data = load_data()
    for kind in kinds:
        if best_name(kind) is None:
            continue
        m = load_model(kind, hg)
        out = {}
        for name in SETS:
            sd = data[name]; ctx = m.prepare(sd['drive'])
            c = sd['c'][:, 0].clone(); F = [kpp_fraction(c)]
            for k in range(sd['c'].shape[1] - 1):
                c, _ = m.advance(c, ctx, DT)
                F.append(kpp_fraction(c))
            out[f'{name}/kpp'] = torch.stack(F, 1).float()
            if hasattr(m, 'rates'):
                a, b = m.rates(c, ctx)
                out[f'{name}/sigma_loc'] = local_sigma(hg, a, b).float()
        torch.save(out, os.path.join(OUT, f'pred_{kind}.pt'))
        print('predicted', kind, flush=True)
    out = {}
    for name in SETS:                                   # the truth in the same format
        sd = data[name]
        out[f'{name}/kpp'] = kpp_fraction(sd['c']).float()
        a, b = true_rates(hg, sd['drive']['logc'])(sd['c'][:, -1])
        out[f'{name}/sigma_loc'] = local_sigma(hg, a, b).float()
        out[f'{name}/atp'] = sd['drive']['logc'][:, 0].exp()
    torch.save(out, os.path.join(OUT, 'pred_truth.pt'))


# ------------------------------------------------------------------------------------------ one site, Gillespie
@torch.no_grad()
def stage_gillespie(B=16, omega=200, t_end=60.0, t_burn=20.0, dt_rec=0.1):
    from . import stochastic as St
    hg1 = build_hg(n_sites=1)
    rng = np.random.default_rng(7)
    x0g = initial_state(rng, build_hg(), 'id')
    V = N * N
    x0 = torch.tensor(np.array([x0g[i * V + INPUT_SITES[0]] for i in range(len(STATE))]))     # an input site
    logc = torch.zeros(1, 3, dtype=torch.float64)                                             # ATP = ADP = Pi = 1
    drive = dict(logc=logc)
    kin = torch.as_tensor(kinase_cat_mask(hg1))
    res = {}
    for kind in [k for k in ['truth', 'wired', 'crnn_chem', 'crnn'] if k == 'truth' or best_name(k) is not None]:
        if kind == 'truth':
            rates = true_rates(hg1, logc.expand(B, -1))
        else:
            m = make_model(kind, hg1)
            m.load_state_dict(torch.load(os.path.join(OUT, best_name(kind) + '.pt'), map_location='cpu'))
            ctx = m.eval().prepare(drive).expand(B)
            if isinstance(m, HyperME):
                rates = (lambda mm, cc: lambda c: mm.rates(c, cc, zero_safe=True))(m, ctx)
            else:
                rates = (lambda mm, cc: lambda c: mm.rates(c, cc))(m, ctx)
        n0 = torch.round(omega * x0)[None].expand(B, -1).clone()
        t0 = time.time()
        r = St.gillespie(rates, hg1, n0, omega, t_end, dt_rec, generator=torch.Generator().manual_seed(1), t_burn=t_burn)
        st = St.fluctuation_stats(r, t_burn=t_burn)
        ikpp, ic9 = STATE.index('Kpp'), STATE.index('C9')
        sel = r['times'] >= t_burn
        kpp = (r['n'][:, sel, ikpp] + r['n'][:, sel, ic9])
        T = float(r['times'][-1] - t_burn); i0 = int(torch.nonzero(sel)[0])
        Q = (r['Nf'][:, -1] - r['Nb'][:, -1]) - (r['Nf'][:, i0] - r['Nb'][:, i0])
        atp = Q[:, kin].sum(-1)                                                 # net ATP consumed per trajectory
        res[kind] = dict(kpp_mean=float(kpp.mean()), kpp_fano=float(kpp.var() / kpp.mean()),
                         atp_rate_mean=float(atp.mean() / T), atp_var_rate=float(atp.var() / T),
                         events=float(r['events'].float().mean()), wall_s=time.time() - t0)
        print(kind, res[kind], flush=True)
    json.dump(res, open(os.path.join(OUT, 'gillespie.json'), 'w'), indent=1)


# ------------------------------------------------------------------------------------------ figure and report
PAL = dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', grid='#e4e3df', s1='#2a78d6', s2='#eb6834', s3='#1baf7a')


def _ax(ax):
    ax.set_facecolor(PAL['surface'])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors=PAL['ink2']); ax.grid(True, color=PAL['grid'], lw=0.6); ax.set_axisbelow(True)


def stage_figure(assets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    seq = LinearSegmentedColormap.from_list('seq', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
    epr = LinearSegmentedColormap.from_list('epr', ['#fbe3d6', '#f5a37f', '#eb6834', '#b9471b', '#7a2c0e'])
    P = {k: torch.load(os.path.join(OUT, f'pred_{k}.pt')) for k in ['truth'] + MODELS if os.path.exists(os.path.join(OUT, f'pred_{k}.pt'))}
    t = np.arange(T_TEST + 1) * DT
    fig = plt.figure(figsize=(15, 9.2), facecolor=PAL['surface'])
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1.05, 1.1], hspace=0.45, wspace=0.35)
    # (a) signal along the space: per-site Kpp fraction over time, coloured by the local EPR at the end (truth | wired)
    tr = P['truth']; b = 0
    smax = max(float(P[k]['test_id/sigma_loc'][b].max()) for k in ('truth', 'wired') if k in P)
    for j, k in enumerate(['truth', 'wired']):
        if k not in P:
            continue
        ax = fig.add_subplot(gs[0, 3 * j:3 * j + 3]); _ax(ax)
        kp = P[k]['test_id/kpp'][b].numpy(); sl = P[k]['test_id/sigma_loc'][b].numpy()
        for v in np.argsort(sl):
            ax.plot(t, kp[:, v], color=epr(sl[v] / smax), lw=1.6)
        ax.set_xscale('symlog', linthresh=1.0); ax.set_xlim(0, t[-1])
        ax.set_title(f'{LABEL[k]}: active MAPK fraction at each of the 16 sites', fontsize=10, color=PAL['ink'], loc='left')
        ax.set_xlabel('t', fontsize=9, color=PAL['ink2']); ax.set_ylabel('Kpp / K total', fontsize=9, color=PAL['ink2'])
    sm = plt.cm.ScalarMappable(cmap=epr, norm=Normalize(0, smax)); sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes[-1], fraction=0.04, pad=0.01); cb.set_label('local EPR at t = 100', fontsize=8, color=PAL['ink2'])
    cb.ax.tick_params(labelsize=7)
    # (b) the 4 x 4 grid at five times, truth and wired
    times = [0, 4, 10, 20, 200]
    vmax = float(tr['test_id/kpp'][b].max())
    sub =gs[1, :].subgridspec(2, 6, wspace=0.08, hspace=0.12)
    for r, k in enumerate(['truth', 'wired']):
        for j, i in enumerate(times):
            ax = fig.add_subplot(sub[r, j]); ax.imshow(P[k]['test_id/kpp'][b, i].reshape(N, N).numpy(), cmap=seq, vmin=0, vmax=vmax, origin='lower')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f't = {i * DT:g}', fontsize=9, color=PAL['ink2'])
            if j == 0:
                ax.set_ylabel(LABEL[k], fontsize=9, color=PAL['ink'])
        if r == 0:
            ax = fig.add_subplot(sub[:, 5]); ax.axis('off')
            ax.text(0.0, 0.5, 'input kinase at the\nlower-left 2 x 2 corner;\nMAPK diffuses', fontsize=9, color=PAL['ink2'], va='center')
    # (c) drive out of distribution: final active MAPK fraction (all sites) against ATP, every test trajectory
    ax = fig.add_subplot(gs[2, 0:3]); _ax(ax)
    atp = np.concatenate([tr[f'{s}/atp'].numpy() for s in SETS[:3]])
    ytr = np.concatenate([tr[f'{s}/kpp'][:, -1].mean(-1).numpy() for s in SETS[:3]])
    o = np.argsort(atp)
    ax.plot(atp[o], ytr[o], color=PAL['ink'], lw=1.5, label='truth')
    nlow = 0
    for k, col, mk, sz, z in (('gnn', PAL['s3'], '^', 16, 3), ('flexible', PAL['s2'], 's', 16, 4), ('wired', PAL['s1'], 'o', 26, 5)):
        if k in P:
            y = np.concatenate([P[k][f'{s}/kpp'][:, -1].mean(-1).numpy() for s in SETS[:3]])
            if k == 'gnn':
                nlow = int((y < -0.05).sum())
            ax.scatter(atp, y, s=sz, color=col, marker=mk, edgecolor=PAL['surface'], lw=0.5, label=LABEL[k], zorder=z)
    ax.axvspan(0.5, 2.0, color=PAL['grid'], alpha=0.6, lw=0, label='training range of ATP')
    ax.set_ylim(-0.05, 1.0)
    if nlow:
        ax.text(0.99, 0.02, f'{nlow} GNN-ODE points below the axis (negative fractions)', transform=ax.transAxes, ha='right',
                fontsize=8, color=PAL['ink2'])
    ax.set_xscale('log'); ax.set_xlabel('ATP (ADP = Pi = 1)', fontsize=9, color=PAL['ink2'])
    ax.set_ylabel('active MAPK fraction at t = 100', fontsize=9, color=PAL['ink2'])
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_title('Steady response to ATP, inside and outside the training range', fontsize=10, color=PAL['ink'], loc='left')
    # (d) the cascade: mean active fraction of each tier is not stored; show the local EPR map (truth vs wired vs flexible)
    for j, k in enumerate(['truth', 'wired', 'flexible']):
        if k not in P:
            continue
        ax = fig.add_subplot(gs[2, 3 + j]); ax.imshow(P[k]['test_id/sigma_loc'][b].reshape(N, N).numpy(), cmap=epr, vmin=0, vmax=smax, origin='lower')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'local EPR, {LABEL[k]}', fontsize=9, color=PAL['ink2'])
    fig.savefig(os.path.join(assets, 'ws6_mapk_cascade.png'), dpi=150, facecolor=PAL['surface'], bbox_inches='tight')
    plt.close(fig)


def stage_report(path):
    L = []
    L += ['| model | params | 10T ID | 10T ATP high (3-5) | 10T ATP low (0.15-0.3) | 10T initial state OOD | T ID | Kpp error at t = 100, ID / high / low | '
          'moiety drift (max) | min c | sigma rel. err ID / high | ATP turnover rel. err at t = 100, ID / high / low | substeps per DT | train s |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k in MODELS:
        if best_name(k) is None:
            continue
        p = os.path.join(OUT, best_name(k) + '.json')
        if not os.path.exists(p):
            continue
        r = json.load(open(p)); s = r['sets']
        g = lambda st, key, f='.3g': (format(s[st][key], f) if s[st].get(key) is not None else '–')
        L.append(f'| {LABEL[k]} (lr {r.get("lr", 1e-3):g}) | {r.get("params", "")} | {g("test_id", "err_10T")} | {g("test_atp_hi", "err_10T")} | {g("test_atp_lo", "err_10T")} | '
                 f'{g("test_init", "err_10T")} | {g("test_id", "err_T")} | {g("test_id", "kpp_end_err")} / {g("test_atp_hi", "kpp_end_err")} / {g("test_atp_lo", "kpp_end_err")} | '
                 f'{max(s[x]["cons_max"] for x in SETS):.1e} | {min(s[x]["min_c"] for x in SETS):.1e} | '
                 f'{g("test_id", "sigma_relerr")} / {g("test_atp_hi", "sigma_relerr")} | '
                 f'{g("test_id", "turnover_end_relerr")} / {g("test_atp_hi", "turnover_end_relerr")} / {g("test_atp_lo", "turnover_end_relerr")} | '
                 f'{g("test_id", "substeps", ".1f")} | {r["train_time_s"]:.0f} |')
    p = os.path.join(OUT, 'gillespie.json')
    if os.path.exists(p):
        gj = json.load(open(p))
        L += ['', '| one site, Omega = 200 | active MAPK (molecules), mean | Fano | ATP turnover per unit time, mean | Var(N_ATP)/t |', '|---|---|---|---|---|']
        for k, v in gj.items():
            L.append(f'| {LABEL.get(k, k)} | {v["kpp_mean"]:.1f} | {v["kpp_fano"]:.2f} | {v["atp_rate_mean"]:.1f} | {v["atp_var_rate"]:.1f} |')
    open(path, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True); p.add_argument('--model', default='wired')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--threads', type=int, default=4); p.add_argument('--device', default='cpu')
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--tag', default='')
    p.add_argument('--warm', type=int, default=0, help='iterations of one-step derivative matching before the rollout training')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    if a.device.startswith('cuda'):
        from .gpu_guard import require_free
        require_free(a.device)
        torch.cuda.set_per_process_memory_fraction(min(1.0, 10e9 / torch.cuda.get_device_properties(0).total_memory))
    if a.stage == 'data':
        stage_data()
    elif a.stage == 'train':
        stage_train(a.model, a.seed, a.iters, a.device, a.lr, a.tag, a.warm)
    elif a.stage == 'predict':
        stage_predict(a.model.split(','))
    elif a.stage == 'gillespie':
        stage_gillespie()
    elif a.stage == 'figure':
        stage_figure(os.path.join(presets.ROOT, 'assets'))
    elif a.stage == 'report':
        stage_report(os.path.join(OUT, 'tables.md'))
