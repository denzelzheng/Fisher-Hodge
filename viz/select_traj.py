"""Print per-trajectory qualitative-difference stats from cached rollouts to choose the showcase trajectory."""
import os, sys, numpy as np, torch
FH_ROOT = os.environ.get('FH_ROOT', os.path.expanduser('~/fisher_hodge'))
sys.path.insert(0, os.path.join(FH_ROOT, 'fisher_hodge'))
from common import RUNS
C = os.path.join(FH_ROOT, 'viz', 'cache')
np.set_printoptions(linewidth=200, precision=3, suppress=True)


def load(k):
    f = os.path.join(C, k + '.npz')
    return np.load(f)['R'] if os.path.exists(f) else None


def stats(sysn, dsys, st, models):
    Y = torch.load(os.path.join(RUNS, 'data', f'{dsys}.pt'))[st]['rho'].numpy()
    print(f'== {sysn} {st}  truth max {Y.max():.2f}')
    hdr = 'traj ' + ' | '.join(f'{m[:14]:>14s} e10T  neg%  minR  mass' for m in models)
    print(hdr)
    for b in range(Y.shape[0]):
        row = f'{b:4d} '
        for m in models:
            R = load(f'{sysn}_{m}_{st}')
            if R is None:
                row += ' ' * 40; continue
            r = R[b]; y = Y[b]
            e = np.linalg.norm(r[200] - y[200]) / np.linalg.norm(y[200])
            neg = np.nanmax((r < 0).mean((1, 2))) * 100
            fb = np.where(~np.isfinite(r).all((1, 2)))[0]
            mn = np.nanmin(r)
            ms = np.nanmax(np.abs(r.mean((1, 2)) - y.mean((1, 2))))
            row += f'{e:14.3f} {neg:5.1f} {mn:6.2f} {ms:5.2f}' + (f' BLOWN@{fb[0]}' if len(fb) else '') + ' | '
        print(row)


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'S1':
        stats('S1', 'S1', 'test_oodpot', ['fisher_hodge', 'euclid_gf', 'gnn_ode'])
    if what == 'S2':
        for st in ['test_oodpot', 'test_id']:
            stats('S2', 'S2', st, ['fisher_hodge', 'euclid_gf', 'gnn_ode', 'jko_amort'])
    if what == 'S3':
        stats('S3', 'S3', 'test_oodpot', ['fisher_hodge', 'fisher_hodge_wfr', 'gnn_ode', 'euclid_gf'])
        for m in ['fisher_hodge', 'fisher_hodge_wfr', 'gnn_ode', 'euclid_gf']:
            R = load(f'S3_{m}_test_oodpot')
            if R is not None:
                print(m, 'max per traj', np.nanmax(R.reshape(16, -1), 1).round(2))
    if what in ('S4', 'S5'):
        stats(what, what, 'test_oodpot', ['fisher_hodge', 'euclid_gf'])
