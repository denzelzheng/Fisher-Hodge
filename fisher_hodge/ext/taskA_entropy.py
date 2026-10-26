"""Common task A (NEEP vs HyperME): entropy production from trajectories.

Systems
  ratchet  NEEP's discrete flashing ratchet (6 states, jump chain), simulated with NEEP's own toy/ratchet.py
           (kdkyum/neep, MIT).  Truth: ratchet.ep_per_step (analytic, per step).  Conditions: base V=2, r=1;
           r x2 (switching rate doubled); V x2.
  c1       our phosphorylation cycle C1 at one site, Omega = 200: X + E + ATP <-> Xp + E + ADP (R1),
           Xp <-> X + Pi (R2), E = 1 catalytic, ATP / ADP / Pi chemostatted.  Mass action with k_R1 = 2, k_R2 = 1;
           base ln(ATP/ADP) = 2 (ATP = e, ADP = 1/e), Pi = 0.02 (so k2'/k2 = 0.02 as in hyper/gen_crn.py).  'ATP x2'
           doubles the affinity, ln(ATP/ADP) = 4, by raising ATP with ADP fixed.  Exact Gillespie (direct method),
           every event recorded with its channel and dwell time.  Truth: exact stationary entropy production of the
           chemical master equation (birth-death chain in n_X with two channels per direction), per jump.
Observation levels for C1: 'chan' (state + which channel fired, i.e. the reservoir exchange is visible: NEEP gets the
token (n_X, net ATP count mod 3)); 'state' (n_X jump chain only).

Our estimators
  counts   plug-in from transition / channel counts (pseudo-count 0.5 on every known transition), no model.
  hyperme  the wired HyperME (hyperme/): mu = log c + U/T, drive from the known potential / chemostats, a log-rate
           table per hyperedge type learned by maximum likelihood on the same trajectories.  sigma from the model's
           rates.  Zero-shot: change the knob (switching prefactor, V, ATP) without new data.

python taskA_entropy.py gen      -> ~/ext_work/taskA/*.npz  (train / valid pools, 64 x L each, 3 seeds)
python taskA_entropy.py ours     -> runs/external/common/taskA_ours.json
"""
import os, sys, json, math, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FH = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, FH)
NEEP = os.path.expanduser('~/ext_baselines/neep')
WORK = os.path.expanduser('~/ext_work/taskA')
OUT = os.path.join(FH, 'runs', 'external', 'common')
L_TRAJ = 10000
MS = (1, 4, 16, 64)
SEEDS = (0, 1, 2)
OMEGA = 200
C1_BASE = dict(k1=2.0, k2=1.0, logATP=1.0, logADP=-1.0, logPi=math.log(0.02), E=1.0)
RATCHET = dict(base=dict(V=2.0, r=1.0), r2=dict(V=2.0, r=2.0), V2=dict(V=4.0, r=1.0))
C1_COND = dict(base=dict(), atp2=dict(logATP=3.0))


def ratchet_mod():
    sys.path.insert(0, NEEP)
    from toy import ratchet
    return ratchet


# ------------------------------------------------------------------------------------------ C1 truth / sampler
def c1_params(cond):
    p = dict(C1_BASE); p.update(C1_COND[cond]); return p


def c1_props(k1, k2, logATP, logADP, logPi, E=1.0, n=None):
    """per-channel propensities at n_X = n (0..Omega): columns R1f (X->Xp), R1b, R2f (Xp->X), R2b; and the change of
    n_X per channel."""
    n = np.arange(OMEGA + 1) if n is None else n
    nX, nXp = n, OMEGA - n
    w = np.stack([k1 * math.exp(logATP) * E * nX, k1 * math.exp(logADP) * E * nXp, k2 * nXp, k2 * math.exp(logPi) * nX], -1)
    return w, np.array([-1, 1, 1, -1])


def chain_ep(w, dn):
    """exact stationary EP of the 1D chain with several channels.  w (Omega+1, C) propensities, dn (C,) +-1.
    Reverse of channel c is channel c ^ 1.  Returns sigma per unit time, mean jump rate, sigma per jump,
    and the EP of the coarse chain that ignores channels (= 0 for a birth-death chain)."""
    N = w.shape[0]
    up = w[:, dn > 0].sum(-1); dw = w[:, dn < 0].sum(-1)
    logpi = np.zeros(N)
    for k in range(1, N):
        logpi[k] = logpi[k - 1] + math.log(up[k - 1]) - math.log(dw[k])
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
    sig = 0.0
    for c in range(w.shape[1]):
        rc = c ^ 1
        for k in range(N):
            k2 = k + dn[c]
            if 0 <= k2 < N and w[k, c] > 0:
                sig += pi[k] * w[k, c] * math.log(pi[k] * w[k, c] / (pi[k2] * w[k2, rc]))
    lam = float((pi * w.sum(-1)).sum())
    return dict(sigma_rate=sig, jump_rate=lam, sigma_jump=sig / lam, coarse_sigma_jump=0.0)


def c1_truth(cond):
    w, dn = c1_props(**c1_params(cond))
    return chain_ep(w, dn)


def c1_gillespie(cond, n_traj, L, rng):
    """returns n (n_traj, L) states before each jump, ch (n_traj, L-1) channel of jump t -> t+1, dwell (n_traj, L-1)"""
    p = c1_params(cond)
    w, dn = c1_props(**p)
    pX = (w[0, 1] + w[0, 2]) / OMEGA                                  # per-molecule rates (Xp -> X) and (X -> Xp)
    q = (w[OMEGA, 0] + w[OMEGA, 3]) / OMEGA
    n = rng.binomial(OMEGA, pX / (pX + q), size=n_traj)               # stationary initial condition
    Ns = np.zeros((n_traj, L), dtype=np.int64); Ch = np.zeros((n_traj, L - 1), dtype=np.int64)
    Dw = np.zeros((n_traj, L - 1))
    for t in range(L):
        Ns[:, t] = n
        if t == L - 1:
            break
        wt = w[n]
        a0 = wt.sum(-1)
        Dw[:, t] = rng.exponential(1.0 / a0)
        u = rng.random(n_traj) * a0
        c = (np.cumsum(wt, -1) < u[:, None]).sum(-1)
        Ch[:, t] = c
        n = n + dn[c]
    return Ns, Ch, Dw


def c1_tokens(Ns, Ch, level):
    """NEEP tokens: 'state' -> n_X; 'chan' -> n_X * 3 + (net R1 firings mod 3)"""
    if level == 'state':
        return Ns
    m = np.zeros_like(Ns)
    step = np.where(Ch == 0, 1, np.where(Ch == 1, -1, 0))
    m[:, 1:] = np.cumsum(step, 1)
    return Ns * 3 + np.mod(m, 3)


# ------------------------------------------------------------------------------------------ data generation
def gen():
    os.makedirs(WORK, exist_ok=True)
    R = ratchet_mod()
    truth = {}
    for cond, pr in RATCHET.items():
        truth[f'ratchet_{cond}'] = float(R.ep_per_step(pr['V'], 1, pr['r']))
        for s in SEEDS:
            np.random.seed(1000 * s + 17 * len(cond))
            tr = R.simulation(64, L_TRAJ, pr['V'], T=1, r=pr['r'])
            va = R.simulation(64, L_TRAJ, pr['V'], T=1, r=pr['r'])
            np.savez_compressed(os.path.join(WORK, f'ratchet_{cond}_s{s}.npz'), train=tr.astype(np.int16), valid=va.astype(np.int16))
    for cond in C1_COND:
        truth[f'c1_{cond}'] = c1_truth(cond)
        for s in SEEDS:
            rng = np.random.default_rng(5000 + 100 * s + len(cond))
            d = {}
            for split in ('train', 'valid'):
                Ns, Ch, Dw = c1_gillespie(cond, 64, L_TRAJ, rng)
                d[f'{split}_n'] = Ns.astype(np.int16); d[f'{split}_ch'] = Ch.astype(np.int8); d[f'{split}_dw'] = Dw.astype(np.float32)
                for lev in ('chan', 'state'):
                    d[f'{split}_tok_{lev}'] = c1_tokens(Ns, Ch, lev).astype(np.int16)
            np.savez_compressed(os.path.join(WORK, f'c1_{cond}_s{s}.npz'), **d)
    # self-check: our own ratchet EP from the transition matrix equals NEEP's analytic value
    chk = {}
    for cond, pr in RATCHET.items():
        P = R.transition_matrix(pr['V'], 1, pr['r'])
        chk[cond] = (jump_chain_ep(P), truth[f'ratchet_{cond}'])
    json.dump(dict(truth=truth, check_ratchet_ep=chk, L=L_TRAJ), open(os.path.join(WORK, 'truth.json'), 'w'), indent=1)
    print(json.dumps(dict(truth=truth, check=chk), indent=1))


def jump_chain_ep(P):
    P = np.asarray(P, float)
    ev, evec = np.linalg.eig(P.T)
    pi = np.real(evec[:, np.argmin(abs(ev - 1))]); pi /= pi.sum()
    s = 0.0
    for i in range(len(P)):
        for j in range(len(P)):
            if P[i, j] > 0 and P[j, i] > 0:
                s += pi[i] * P[i, j] * math.log(pi[i] * P[i, j] / (pi[j] * P[j, i]))
    return float(s)


# ------------------------------------------------------------------------------------------ our estimators: ratchet
RATCHET_EDGES = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]


def transition_counts(trajs, n_states):
    C = np.zeros((n_states, n_states))
    for tr in trajs:
        np.add.at(C, (tr[:-1], tr[1:]), 1)
    return C


def plugin_ep(C, edges, pseudo=0.5):
    """plug-in EP per step from transition counts on the known edges (both directions get the pseudo-count)"""
    N = C.sum()
    s = 0.0
    for i, j in edges:
        a, b = C[i, j] + pseudo, C[j, i] + pseudo
        s += (a - b) / N * math.log(a / b)
    return float(s)


def ratchet_hyperme():
    from hyperme import Hypergraph, HyperME
    hg = Hypergraph.from_graph(RATCHET_EDGES, 6)
    hg.meta['typ'] = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])          # ON transport, OFF transport, switching
    hg._torch = {}
    with torch.random.fork_rng():
        torch.set_default_dtype(torch.float64)
        m = HyperME(hg, kinetics_table=True, harmonic=False)
        torch.set_default_dtype(torch.float32)
    return m.double(), hg


def ratchet_drive(V, B=1):
    U = torch.tensor([[0.0, V, 2 * V, 0, 0, 0]], dtype=torch.float64).expand(B, 6)
    line = torch.zeros(B, 9, dtype=torch.float64)
    for h, (i, j) in enumerate(RATCHET_EDGES[6:], start=6):
        line[:, h] = U[:, j] - U[:, i]                               # energy jump imposed by the flashing
    return dict(U=U, line=line, T=torch.ones(B, dtype=torch.float64))


def ratchet_P(model, V, log_switch_scale=0.0):
    c = torch.ones(1, 6, dtype=torch.float64)
    qp, qm = model.transition_rates(c, ratchet_drive(V))
    qp, qm = qp[0].clone(), qm[0].clone()
    qp[6:] = qp[6:] * math.exp(log_switch_scale); qm[6:] = qm[6:] * math.exp(log_switch_scale)
    W = torch.zeros(6, 6, dtype=torch.float64)
    for h, (i, j) in enumerate(RATCHET_EDGES):
        W[i, j] = qp[h]; W[j, i] = qm[h]
    return W / W.sum(1, keepdim=True)


def fit_ratchet(C, V, iters=3000, lr=0.05):
    model, hg = ratchet_hyperme()
    Ct = torch.tensor(C, dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        P = ratchet_P(model, V)
        nll = -(Ct * torch.log(P.clamp_min(1e-300))).sum() / Ct.sum()
        opt.zero_grad(); nll.backward(); opt.step()
    return model, float(nll)


# ------------------------------------------------------------------------------------------ our estimators: C1
def c1_hyperme():
    from hyperme import Hypergraph, HyperME
    hg = Hypergraph.reaction_transport(['X', 'Xp', 'E', 'ATP', 'ADP', 'Pi'],
                                       [({'X': 1, 'E': 1, 'ATP': 1}, {'Xp': 1, 'E': 1, 'ADP': 1}), ({'Xp': 1}, {'X': 1, 'Pi': 1})],
                                       n_sites=1, chemostats=['ATP', 'ADP', 'Pi'])
    torch.set_default_dtype(torch.float64)
    m = HyperME(hg, kinetics_table=True, harmonic=False)
    torch.set_default_dtype(torch.float32)
    return m.double()


def c1_model_props(model, p):
    """HyperME propensities Omega * a(n / Omega) for n = 0..Omega, as (Omega+1, 4) [R1f, R1b, R2f, R2b]"""
    n = torch.arange(OMEGA + 1, dtype=torch.float64)
    c = torch.stack([n, OMEGA - n, torch.full_like(n, OMEGA * p['E'])], -1) / OMEGA
    drive = dict(chem_logc=torch.tensor([[p['logATP'], p['logADP'], p['logPi']]], dtype=torch.float64).expand(len(n), 3))
    a, b = model.rates(c, drive, zero_safe=True)
    return OMEGA * torch.stack([a[:, 0], b[:, 0], a[:, 1], b[:, 1]], -1)


def c1_stats(Ns, Ch, Dw):
    """sufficient statistics: dwell time per state T (Omega+1,), counts per (state, channel) (Omega+1, 4)"""
    T = np.zeros(OMEGA + 1); Nc = np.zeros((OMEGA + 1, 4))
    n0 = Ns[:, :-1]
    np.add.at(T, n0.ravel(), Dw.ravel())
    np.add.at(Nc, (n0.ravel(), Ch.ravel()), 1)
    return T, Nc


def fit_c1(stats, p, level, iters=3000, lr=0.05):
    model = c1_hyperme()
    T, Nc = (torch.tensor(x, dtype=torch.float64) for x in stats)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        w = c1_model_props(model, p)
        if level == 'chan':                                  # Poisson-process likelihood with channel labels
            ll = (Nc * torch.log(w.clamp_min(1e-300))).sum() - (T[:, None] * w).sum()
        else:                                                # jump chain of n_X only (no times, no channels)
            up = w[:, 1] + w[:, 2]; dw = w[:, 0] + w[:, 3]
            U = Nc[:, 1] + Nc[:, 2]; D = Nc[:, 0] + Nc[:, 3]
            tot = up + dw
            ll = (U * torch.log((up / tot).clamp_min(1e-300)) + D * torch.log((dw / tot).clamp_min(1e-300))).sum()
        loss = -ll / Nc.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def c1_counts_ep(stats, level, pseudo=0.5):
    """'chan': mass-action constants per channel = count / integral of the mass-action factor, then exact EP;
    'state': the n_X chain alone (plug-in on up / down counts), whose EP is that of a birth-death chain (~0)."""
    T, Nc = stats
    n = np.arange(OMEGA + 1)
    if level == 'chan':
        fac = np.stack([n, OMEGA - n, OMEGA - n, n], -1).astype(float)
        kap = (Nc.sum(0) + pseudo) / (T[:, None] * fac).sum(0)
        return chain_ep(kap[None] * fac, np.array([-1, 1, 1, -1]))['sigma_jump']
    Cn = np.zeros((OMEGA + 1, OMEGA + 1))
    for k in range(OMEGA):
        Cn[k, k + 1] = Nc[k, 1] + Nc[k, 2]; Cn[k + 1, k] = Nc[k + 1, 0] + Nc[k + 1, 3]
    return plugin_ep(Cn, [(k, k + 1) for k in range(OMEGA) if Cn[k, k + 1] + Cn[k + 1, k] > 0], pseudo)


def ours():
    truth = json.load(open(os.path.join(WORK, 'truth.json')))['truth']
    res = []
    for s in SEEDS:
        d = np.load(os.path.join(WORK, f'ratchet_base_s{s}.npz'))
        for M in MS:
            C = transition_counts(d['train'][:M].astype(np.int64), 6)
            row = dict(sys='ratchet', seed=s, M=M, transitions=int(C.sum()), counts_base=plugin_ep(C, RATCHET_EDGES))
            model, nll = fit_ratchet(C, 2.0)
            with torch.no_grad():
                row['hyperme_base'] = jump_chain_ep(ratchet_P(model, 2.0).numpy())
                row['hyperme_r2_zeroshot'] = jump_chain_ep(ratchet_P(model, 2.0, math.log(2.0)).numpy())
                row['hyperme_V2_zeroshot'] = jump_chain_ep(ratchet_P(model, 4.0).numpy())
                row['hyperme_logk'] = model.kinetics.core.tolist(); row['hyperme_sA'] = float(model.drive.s_A)
            for cond in ('r2', 'V2'):                         # counts on new data for reference
                dc = np.load(os.path.join(WORK, f'ratchet_{cond}_s{s}.npz'))
                row[f'counts_{cond}_newdata'] = plugin_ep(transition_counts(dc['train'][:M].astype(np.int64), 6), RATCHET_EDGES)
            res.append(row); print(row, flush=True)
    for s in SEEDS:
        d = np.load(os.path.join(WORK, f'c1_base_s{s}.npz'))
        for M in MS:
            st = c1_stats(d['train_n'][:M].astype(np.int64), d['train_ch'][:M].astype(np.int64), d['train_dw'][:M].astype(np.float64))
            for level in ('chan', 'state'):
                row = dict(sys='c1', level=level, seed=s, M=M, transitions=int(st[1].sum()), counts_base=c1_counts_ep(st, level))
                model = fit_c1(st, c1_params('base'), level)
                with torch.no_grad():
                    for cond in ('base', 'atp2'):
                        w = c1_model_props(model, c1_params(cond)).numpy()
                        row[f'hyperme_{cond}' + ('' if cond == 'base' else '_zeroshot')] = chain_ep(w, np.array([-1, 1, 1, -1]))['sigma_jump']
                    row['hyperme_logk'] = model.kinetics.core.tolist()
                dc = np.load(os.path.join(WORK, f'c1_atp2_s{s}.npz'))
                st2 = c1_stats(dc['train_n'][:M].astype(np.int64), dc['train_ch'][:M].astype(np.int64), dc['train_dw'][:M].astype(np.float64))
                row['counts_atp2_newdata'] = c1_counts_ep(st2, level)
                res.append(row); print(row, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(truth=truth, rows=res, true_logk_c1=[math.log(2.0), 0.0]), open(os.path.join(OUT, 'taskA_ours.json'), 'w'), indent=1)


def ours2m():
    """same estimators on the same 2M trajectories NEEP consumes (its M training + M test trajectories), base only"""
    res = []
    for s in SEEDS:
        d = np.load(os.path.join(WORK, f'ratchet_base_s{s}.npz'))
        for M in MS:
            tr = np.concatenate([d['train'][:M], d['valid'][:M]]).astype(np.int64)
            C = transition_counts(tr, 6)
            model, _ = fit_ratchet(C, 2.0)
            with torch.no_grad():
                res.append(dict(sys='ratchet', seed=s, M=M, counts_base_2M=plugin_ep(C, RATCHET_EDGES),
                                hyperme_base_2M=jump_chain_ep(ratchet_P(model, 2.0).numpy()),
                                hyperme_r2_zeroshot_2M=jump_chain_ep(ratchet_P(model, 2.0, math.log(2.0)).numpy()),
                                hyperme_V2_zeroshot_2M=jump_chain_ep(ratchet_P(model, 4.0).numpy())))
        d = np.load(os.path.join(WORK, f'c1_base_s{s}.npz'))
        for M in MS:
            cat = lambda k, dt: np.concatenate([d['train_' + k][:M], d['valid_' + k][:M]]).astype(dt)
            st = c1_stats(cat('n', np.int64), cat('ch', np.int64), cat('dw', np.float64))
            for level in ('chan', 'state'):
                model = fit_c1(st, c1_params('base'), level)
                with torch.no_grad():
                    w = c1_model_props(model, c1_params('base')).numpy()
                    w2 = c1_model_props(model, c1_params('atp2')).numpy()
                res.append(dict(sys='c1', level=level, seed=s, M=M, counts_base_2M=c1_counts_ep(st, level),
                                hyperme_base_2M=chain_ep(w, np.array([-1, 1, 1, -1]))['sigma_jump'],
                                hyperme_atp2_zeroshot_2M=chain_ep(w2, np.array([-1, 1, 1, -1]))['sigma_jump']))
        print(s, flush=True)
    json.dump(res, open(os.path.join(OUT, 'taskA_ours_2M.json'), 'w'), indent=1)


if __name__ == '__main__':
    {'gen': gen, 'ours': ours, 'ours2m': ours2m}[sys.argv[1]]()
