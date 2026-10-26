"""Common task A: NEEP (Kim et al. PRL 2020, github.com/kdkyum/neep, MIT) on the same trajectories as our estimators.
NEEP's own EmbeddingNEEP, CartesianSampler, train / validate functions and default hyper-parameters of
train_ratchet.py (n_layer 1, n_hidden 128, lr 1e-4, wd 5e-5, batch 4096, 50000 iterations, record every 100,
estimate = 'best_pred_rate', the test-set EP per step at the lowest test loss).  As in NEEP's default the test set has
the same size as the training set (M fresh trajectories); our estimators use only the M training trajectories.
python taskA_neep.py --worker k --n_workers W"""
import os, sys, json, argparse, time, itertools
import numpy as np
import torch
NEEP = os.path.expanduser('~/ext_baselines/neep')
sys.path.insert(0, NEEP)
from model.net import EmbeddingNEEP
from misc.sampler import CartesianSampler
from misc.utils import logging_rneep
import train_ratchet as tr_mod                     # NEEP's train() / validate()

WORK = os.path.expanduser('~/ext_work/taskA')
OUTD = os.path.join(WORK, 'neep')
MS, SEEDS = (1, 4, 16, 64), (0, 1, 2)


def jobs():
    out = []
    for cond, M, s in itertools.product(('base', 'r2', 'V2'), MS, SEEDS):
        out.append(('ratchet', cond, None, M, s))
    for lev, cond, M, s in itertools.product(('chan', 'state'), ('base', 'atp2'), MS, SEEDS):
        out.append(('c1', cond, lev, M, s))
    return out


def run(sysname, cond, lev, M, s, n_iter=50000):
    tag = f'{sysname}_{cond}' + (f'_{lev}' if lev else '') + f'_M{M}_s{s}'
    fout = os.path.join(OUTD, tag + '.json')
    if os.path.exists(fout):
        return
    d = np.load(os.path.join(WORK, f'{sysname}_{cond}_s{s}.npz'))
    if sysname == 'ratchet':
        trn, val, ntok = d['train'][:M], d['valid'][:M], 6
    else:
        trn, val, ntok = d[f'train_tok_{lev}'][:M], d[f'valid_tok_{lev}'][:M], 201 * (3 if lev == 'chan' else 1)
    opt = argparse.Namespace(n_token=ntok, n_hidden=128, n_layer=1, lr=1e-4, wd=5e-5, batch_size=4096,
                             test_batch_size=10000, n_iter=n_iter, record_freq=100, seed=s, seq_len=2)
    torch.manual_seed(s)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmbeddingNEEP(opt).to(opt.device)
    optim = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.wd)
    T = torch.from_numpy(trn.astype(np.int64)).to(opt.device)
    V = torch.from_numpy(val.astype(np.int64)).to(opt.device)
    L = T.shape[1]
    trs = CartesianSampler(M, L, opt.batch_size, device=opt.device)
    tes = CartesianSampler(M, L, opt.test_batch_size, device=opt.device, train=False)
    best_loss, best_rate, hist = None, None, []
    t0 = time.time()
    for i in range(1, n_iter + 1):
        if i % opt.record_freq == 0 or i == 1:
            preds, test_loss = tr_mod.validate(opt, model, V, tes)
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                rate = float(logging_rneep(i, test_loss, opt.seq_len, preds, train=False)['pred_rate'])   # NEEP's estimate
            if best_loss is None or test_loss < best_loss:
                best_loss, best_rate, best_mean = test_loss, rate, float(np.mean(preds))
            hist.append((i, test_loss, rate))
            trs.train()
        tr_mod.train(opt, model, optim, T, trs)
    preds_tr, train_loss = tr_mod.validate(opt, model, T, CartesianSampler(M, L, opt.test_batch_size, device=opt.device, train=False))
    json.dump(dict(sys=sysname, cond=cond, level=lev, M=M, seed=s, best_pred_rate=best_rate, best_pred_mean=best_mean, best_test_loss=best_loss,
                   final_train_rate=float(np.mean(preds_tr)), hist=hist[::10] + [hist[-1]], seconds=time.time() - t0,
                   transitions=int(M * (L - 1))), open(fout, 'w'))
    print(tag, best_rate, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--n_workers', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    for k, j in enumerate(jobs()):
        if k % a.n_workers == a.worker:
            run(*j)
