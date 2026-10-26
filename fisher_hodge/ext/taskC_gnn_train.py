"""Common task C, step 1: particle data (our re-implementation of the mixed-species simulator, parameter set (i))
and training of the re-implemented non-reciprocal GNN with the paper's protocol.
python taskC_gnn_train.py --work ~/ext_work/taskC [--epochs 20000] [--hours 9]"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from fisher_hodge.ext import nonreciprocal_gnn as ng

ap = argparse.ArgumentParser()
ap.add_argument('--work', default=os.path.expanduser('~/ext_work/taskC'))
ap.add_argument('--epochs', type=int, default=20000)
ap.add_argument('--hours', type=float, default=9.0)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
sets = dict(train=(6, (200, 200), 0), test_id=(3, (200, 200), 1), test_ood_ratio=(3, (100, 300), 2))
for name, (M, nps, seed) in sets.items():
    f = os.path.join(a.work, f'{name}.pt')
    if not os.path.exists(f):
        t = time.time()
        torch.save(ng.make_dataset(M, nps, T=300, seed=seed), f)
        print(f'{name}: {M} trajectories {nps} in {time.time() - t:.0f}s', flush=True)
data = torch.load(os.path.join(a.work, 'train.pt'))
perm = torch.randperm(6, generator=torch.Generator().manual_seed(42)).tolist()
tr, va = perm[:3], perm[3:]
json.dump(dict(train=tr, valid=va), open(os.path.join(a.work, 'split.json'), 'w'))
log = open(os.path.join(a.work, 'gnn_train.log'), 'a')
print(f'train trajectories {tr}, valid {va}', file=log, flush=True)
model, hist = ng.train_gnn(data, tr, va, epochs=a.epochs, seed=a.seed, log=log, ckpt=os.path.join(a.work, 'gnn_best.pt'),
                           time_limit=a.hours * 3600)
torch.save(dict(state=model.state_dict(), hist=hist), os.path.join(a.work, 'gnn_final.pt'))
print('done', file=log, flush=True)
