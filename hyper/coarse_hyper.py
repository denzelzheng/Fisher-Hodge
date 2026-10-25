"""Coarse-graining commutation test: 8x8 -> 4x4 block merge (concentrations summed [spec] or averaged; U block-mean).
E_comm = || coarsen(rollout_8x8(c0)) - rollout_4x4(coarsen(c0)) || / || coarsen(rollout_8x8(c0)) ||   (X, Xp channels)
same trained model on both graphs. Reference: the true physics with the same (naive) parameters on the 4x4 graph.
python coarse_hyper.py -> runs/coarse.json, runs/coarse_commutation.png"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyperme import HyperME, DT, T_TEST, RUNS, NS, build_graph_np
from baselines_hyper import CRNN, GNNODE3
from gen_crn import solve_traj

MODELS = {'hyperme': lambda: HyperME(True), 'hyperme_rev': lambda: HyperME(False), 'crnn': CRNN, 'gnn': GNNODE3}
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
data = torch.load(os.path.join(RUNS, 'data', 'crn.pt'))
TS = [20, 200]


def coarsen(c, mode):     # (..., NS, 64) -> (..., NS, 16)
    x = c.reshape(*c.shape[:-1], 4, 2, 4, 2).sum((-3, -1)).reshape(*c.shape[:-1], 16)
    return x if mode == 'sum' else x / 4.0


def coarsenU(U):
    return U.reshape(-1, 4, 2, 4, 2).mean((2, 4)).reshape(-1, 16)


@torch.no_grad()
def rollout(model, U, dG, c0, n):
    ctx = model.prepare(U, dG, n)
    c = c0.clone(); out = [c.cpu()]
    for _ in range(T_TEST):
        c, _ = model.advance(c, ctx, DT)
        bad = (~torch.isfinite(c)).flatten(1).any(1) | (c.abs().flatten(1).amax(1) > 1e4)
        if bad.any():
            c = torch.where(bad[:, None, None], torch.full_like(c, float('nan')), c)
        out.append(c.cpu())
    return torch.stack(out, 1).numpy()


def rel(R, Y):
    B, T1 = Y.shape[:2]
    e = np.linalg.norm((R - Y)[:, :, :2].reshape(B, T1, -1), axis=-1) / np.linalg.norm(Y[:, :, :2].reshape(B, T1, -1), axis=-1)
    return np.where(np.isfinite(e), e, np.inf)


res = {}
sets = ['test_id', 'test_oodG']
g4 = build_graph_np(4)
for mode in ['sum', 'mean']:
    res[mode] = {}
    for setname in sets:
        sd = data[setname]; Y = sd['c'].numpy(); U = sd['U'].numpy(); dG = sd['dG'].numpy()
        Yc = coarsen(Y, mode); Uc = coarsenU(U)
        # truth on coarse graph with the same physical parameters
        Tc = np.stack([solve_traj(g4, Uc[b], dG[b], Yc[b, 0].ravel(), T_TEST)[0].reshape(-1, NS, 16) for b in range(len(Y))])
        et = rel(Tc, Yc)
        res[mode].setdefault('truth', {})[setname] = {str(t): float(np.median(et[:, t])) for t in TS}
        for mname in MODELS:
            for seed in [0, 1]:
                ck = os.path.join(RUNS, 'main', f'{mname}_s{seed}.pt')
                if not os.path.exists(ck):
                    continue
                m = MODELS[mname]().to(dev); m.load_state_dict(torch.load(ck, map_location=dev)); m.eval()
                Ut, Gt = torch.tensor(U, device=dev), torch.tensor(dG, device=dev)
                Rf = rollout(m, Ut, Gt, torch.tensor(Y[:, 0], device=dev), 8)
                Rc = rollout(m, torch.tensor(Uc, device=dev), Gt, torch.tensor(Yc[:, 0], device=dev), 4)
                Rfc = coarsen(Rf, mode)
                ecomm = rel(Rc, Rfc); efine = rel(Rfc, Yc); ecoarse = rel(Rc, Yc)
                d = res[mode].setdefault(mname, {}).setdefault(setname, {})
                d[f's{seed}'] = dict(comm={str(t): float(np.median(ecomm[:, t])) for t in TS},
                                     fine_vs_truth={str(t): float(np.median(efine[:, t])) for t in TS},
                                     coarse_vs_truth={str(t): float(np.median(ecoarse[:, t])) for t in TS},
                                     comm_t=np.median(ecomm, 0).tolist(), neg_coarse=int((Rc < 0).reshape(len(Rc), -1).any(1).sum()))
                print(mode, setname, mname, seed, 'comm T/10T', [f'{np.median(ecomm[:, t]):.3e}' for t in TS],
                      'coarse-vs-truth 10T', f'{np.median(ecoarse[:, 200]):.3e}', flush=True)
        print(mode, setname, 'TRUTH comm T/10T', res[mode]['truth'][setname], flush=True)
json.dump(res, open(os.path.join(RUNS, 'coarse.json'), 'w'), indent=1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
names = {'hyperme': 'HyperME', 'hyperme_rev': 'HyperME (A=0)', 'crnn': 'B1 CRNN', 'gnn': 'B2 GNN-ODE'}
cols = {'hyperme': '#1f77b4', 'hyperme_rev': '#9467bd', 'crnn': '#ff7f0e', 'gnn': '#2ca02c'}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
t = np.arange(T_TEST + 1) * DT
for ax, mode in zip(axs[:2], ['sum', 'mean']):
    for mname in MODELS:
        if mname not in res[mode]:
            continue
        curves = [np.array(res[mode][mname]['test_id'][s]['comm_t']) for s in res[mode][mname]['test_id']]
        for i, cv in enumerate(curves):
            ax.semilogy(t, np.where(np.isfinite(cv), cv, np.nan), color=cols[mname], lw=1.5, ls='-' if i == 0 else '--',
                        label=names[mname] if i == 0 else None)
    ax.axhline(res[mode]['truth']['test_id']['200'], color='k', ls=':', label='true physics, naive coarse params (10T)')
    ax.set_title(f'commutation error, block-{mode} (ID test)'); ax.set_xlabel('t'); ax.set_ylabel('median rel. L2 (X,Xp)')
    ax.legend(fontsize=7)
ax = axs[2]
labs = list(MODELS) + ['truth']
x = np.arange(len(labs)); w = 0.35
for j, mode in enumerate(['sum', 'mean']):
    vals = []
    for mname in labs:
        if mname == 'truth':
            vals.append(res[mode]['truth']['test_id']['200'])
        elif mname in res[mode]:
            vals.append(np.mean([res[mode][mname]['test_id'][s]['comm']['200'] for s in res[mode][mname]['test_id']]))
        else:
            vals.append(np.nan)
    ax.bar(x + (j - 0.5) * w, vals, w, label=f'block-{mode}')
ax.set_xticks(x); ax.set_xticklabels([names.get(l, 'truth (naive)') for l in labs], rotation=20, fontsize=8)
ax.set_yscale('log'); ax.set_title('commutation error at 10T (ID, seed mean)'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RUNS, 'coarse_commutation.png'), dpi=130)
print('saved')
