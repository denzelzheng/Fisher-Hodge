"""Diagnostic: is the (rare) learned-Lyapunov increase of FH on S1 OOD-potential a time-discretisation effect?
Re-roll the trained S1 FH models with NSUB = 8 (as trained/evaluated) vs 32 substeps per DT."""
import json, os, numpy as np, torch
import models
from common import *
dev = 'cuda'
data = torch.load(os.path.join(RUNS, 'data', 'S1.pt'))
out = {}
for seed in (0, 1, 2):
    m = models.FisherHodge('S1').to(dev)
    m.load_state_dict(torch.load(os.path.join(RUNS, 'main', f'S1_fisher_hodge_s{seed}.pt'))); m.eval()
    for st in ['test_oodpot', 'test_id']:
        sd = data[st]; F = {k: sd[k].to(dev) for k in ['U', 'dpsi', 'psin']}
        Y = sd['rho'].numpy()
        for nsub in (8, 32):
            models.NSUB = nsub
            with torch.no_grad():
                ctx = m.prepare(F); r = sd['rho'][:, 0].to(dev); L = [m.lyapunov(r, F).cpu().numpy()]; R = [r.cpu().numpy()]
                for k in range(200):
                    r, _ = m.advance(r, ctx, DT); L.append(m.lyapunov(r, F).cpu().numpy()); R.append(r.cpu().numpy())
            L = np.stack(L, 1); R = np.stack(R, 1)
            inc = np.diff(L, axis=1).max(1)
            err = np.linalg.norm((R[:, -1] - Y[:, -1]).reshape(16, -1), axis=1) / np.linalg.norm(Y[:, -1].reshape(16, -1), axis=1)
            res = dict(lyap_monotone_frac=float((inc <= 1e-6 * (1 + np.abs(L[:, :-1]).max(1))).mean()), max_increase=float(inc.max()),
                       relerr10T_median=float(np.median(err)), min_rho=float(R.min()))
            out[f's{seed}_{st}_nsub{nsub}'] = res; print(seed, st, nsub, res, flush=True)
json.dump(out, open(os.path.join(RUNS, 'check_lyap_S1.json'), 'w'), indent=1)
