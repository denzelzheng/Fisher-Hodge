"""Re-run full 201-frame rollouts of trained checkpoints (same advance()/substep logic as train.py / train2.py)
and cache them for the qualitative visualisations.  Usage: python viz/rollout_all.py [job-filter ...]"""
import os, sys, json, time
import numpy as np
import torch
FH_ROOT = os.environ.get('FH_ROOT', os.path.expanduser('~/fisher_hodge'))
sys.path.insert(0, os.path.join(FH_ROOT, 'fisher_hodge'))
from common import *
import models2
from models2 import MODELS2

dev = 'cuda'
VIZ = os.path.join(ROOT, 'viz'); CACHE = os.path.join(VIZ, 'cache'); os.makedirs(CACHE, exist_ok=True)
MAIN = os.path.join(RUNS, 'main'); STRESS = os.path.join(RUNS, 'stress', 'main')
FIELDS = ['U', 'dpsi', 'psin']
SYSCLASS = {'S1': 'S1', 'S2': 'S2', 'S3': 'S3', 'S3b': 'S3', 'S4': 'S1', 'S5': 'S1', 'S6': 'S1'}

# (tag, train-system, model, checkpoint dir, ckpt stem, data system, test sets)
JOBS = []
for s in ['S1', 'S2']:
    for m in ['fisher_hodge', 'euclid_gf', 'gnn_ode']:
        JOBS.append((s, m, MAIN, f'{s}_{m}_s0', s, ['test_oodpot', 'test_id']))
JOBS.append(('S2', 'jko_amort', STRESS, 'S2_jko_amort_s0', 'S2', ['test_oodpot', 'test_id']))
for m in ['fisher_hodge', 'euclid_gf', 'gnn_ode']:
    JOBS.append(('S3', m, MAIN, f'S3_{m}_s0', 'S3', ['test_oodpot']))
JOBS.append(('S3', 'fisher_hodge_wfr', STRESS, 'S3_fisher_hodge_wfr_s0', 'S3', ['test_oodpot']))
JOBS.append(('S3b', 'fisher_hodge_wfr', STRESS, 'S3b_fisher_hodge_wfr_s0', 'S3', ['test_oodpot']))
JOBS.append(('S4', 'fisher_hodge', STRESS, 'S4_fisher_hodge_s0_viz', 'S4', ['test_oodpot']))
JOBS.append(('S4', 'euclid_gf', STRESS, 'S4_euclid_gf_s0', 'S4', ['test_oodpot']))
JOBS.append(('S5', 'fisher_hodge', STRESS, 'S5_fisher_hodge_s0', 'S5', ['test_oodpot']))
JOBS.append(('S5', 'euclid_gf', STRESS, 'S5_euclid_gf_s0', 'S5', ['test_oodpot']))

filt = sys.argv[1:]
DATA = {}


def build(tsys, mname):
    s = tsys if tsys in ('S1', 'S2', 'S3', 'S5') else SYSCLASS[tsys]
    m = MODELS2[mname](s)
    if tsys == 'S5':
        m.sys = 'S1'
    return m


@torch.no_grad()
def rollout(model, F, rho0, nsteps):
    ctx = model.prepare(F)
    r = rho0.clone(); out = [r.cpu()]
    for k in range(nsteps):
        r, ns = model.advance(r, ctx, DT)
        bad = (~torch.isfinite(r)).any(-1).any(-1) | (r.abs().amax((-1, -2)) > 1e4)
        if bad.any():
            r = torch.where(bad[:, None, None], torch.full_like(r, float('nan')), r)
        out.append(r.cpu())
    return torch.stack(out, 1), ctx


for (tsys, mname, d, stem, dsys, sets) in JOBS:
    key = f'{tsys}_{mname}'
    if filt and not any(f in key for f in filt):
        continue
    ck = os.path.join(d, stem + '.pt')
    if not os.path.exists(ck):
        print('MISSING', ck, flush=True); continue
    if dsys not in DATA:
        DATA[dsys] = torch.load(os.path.join(RUNS, 'data', f'{dsys}.pt'))
    data = DATA[dsys]
    model = build(tsys, mname).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    snapf = os.path.join(d, stem + '_snap.npz')
    snap = np.load(snapf) if os.path.exists(snapf) else None
    for st in sets:
        out = os.path.join(CACHE, f'{key}_{st}.npz')
        if os.path.exists(out):
            print('skip', out); continue
        sd = data[st]
        F = {k: sd[k].to(dev) for k in FIELDS}
        t0 = time.time()
        R, ctx = rollout(model, F, sd['rho'][:, 0].to(dev), sd['rho'].shape[1] - 1)
        R = R.numpy().astype(np.float32)
        extra = {}
        if hasattr(model, 'flux_q') and SYSCLASS[tsys] in ('S1', 'S2'):
            # physical flux J = -(D/h) q at every 10th frame (t = 0, 0.5, ..., 10), used for streamlines / particles
            Js = []
            for k in range(0, R.shape[1], 10):
                rr = torch.tensor(R[:, k], device=dev)
                fin = torch.isfinite(rr).all(-1).all(-1)
                rr = torch.where(fin[:, None, None], rr, torch.ones_like(rr))
                with torch.no_grad():
                    q = model.flux_q(rr, ctx).detach()
                Js.append((-(D / HC) * q).float().cpu().numpy())
            extra['J'] = np.stack(Js, 1)          # (B, 21, 2, n, n)
        chk = ''
        if snap is not None and st in snap.files:
            dif = np.nanmax(np.abs(R[:, [0, 20, 40, 100, 200]] - snap[st]))
            chk = f'max |R - snap| = {dif:.2e}'
        np.savez_compressed(out, R=R, **extra)
        print(f'{key} {st}: {time.time() - t0:.0f}s  {chk}', flush=True)
    del model; torch.cuda.empty_cache()
print('done')
