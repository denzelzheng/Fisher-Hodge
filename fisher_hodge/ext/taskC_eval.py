"""Common task C: evaluation of the re-implemented non-reciprocal GNN (particle level, binned afterwards) and of
HyperME (density level) on the same test trajectories, with the same ruler.

Cases: test trajectory m (3 per set), start frame t0 in {30, 60, ..., 270}; 27 cases per set.  Sets: test_id
(200 + 200 particles) and test_ood_ratio (100 + 300 particles, same interactions; not seen in training).
Metrics (all on the 16 x 16 x 2 binned density, h = 1.25):
  rho@1, rho@10   relative L2 of the density 1 and 10 time units after t0 (10T; T = 1 = the training horizon)
  J@1             relative L2 of the face flux over [t0, t0 + 1] against the flux of the data (one noise realisation)
                  and against the expected flux (mean of 8 re-simulations of the true particle model)
  circ            plaquette circulation of the face flux: correlation with the expected flux's circulation and
                  RMS ratio; and the time-averaged (t in [100, 299]) flux of every test trajectory, its RMS and its
                  circulation RMS
References: 'oracle' = the true particle model re-simulated from the exact particle state with new noise (single
realisation and the mean of 8), 'persist' = rho(t0), 'uniform' = the mean density.
python taskC_eval.py --work ~/ext_work/taskC --out runs/external/common"""
import os, sys, json, math, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from fisher_hodge.ext import nonreciprocal_gnn as ng
from fisher_hodge.ext import taskC_hyperme as th

NB, LBOX, H = th.NB, th.LBOX, th.H
NENS = 8
T0S = list(range(30, 271, 30))


def rel(a, b):
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return ((a - b).norm(dim=1) / b.norm(dim=1)).cpu().numpy()


def circulation(Jx, Jy):
    """counter-clockwise circulation around every plaquette (ix, iy)-(ix+1, iy+1); (..., 2, NB, NB)"""
    return Jx + Jy.roll(-1, -2) - Jx.roll(-1, -1) - Jy


def corr(a, b):
    a = a.reshape(a.shape[0], -1); b = b.reshape(b.shape[0], -1)
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)).cpu().numpy()


def summarize(v):
    v = np.asarray(v)
    return dict(median=float(np.median(v)), mean=float(np.mean(v)), q25=float(np.quantile(v, 0.25)), q75=float(np.quantile(v, 0.75)))


@torch.no_grad()
def eval_set(d, gnn, hm, truth, dev, gen):
    M = d['x'].shape[1]
    cases = [(m, t0) for m in range(M) for t0 in T0S]
    mi = torch.tensor([c[0] for c in cases]); ti = torch.tensor([c[1] for c in cases])
    X, P, C = d['x'], d['phi'], d['c']
    x0, p0, c0 = X[ti, mi].to(dev), P[ti, mi].to(dev), C[mi].to(dev)
    B = len(cases)
    dens = lambda x, c: ng.bin_density(x, c, NB, LBOX)
    true_rho = {k: dens(X[ti + k, mi].to(dev), c0) for k in (1, 10)}
    true_J = ng.face_flux(x0, X[ti + 1, mi].to(dev), c0, NB, LBOX)
    out = {}

    def ensemble(fn):
        """fn(x, p, c) -> (X (11, B', N, 2)); returns mean densities at 1, 10, single member, mean flux"""
        xs = x0.repeat(NENS, 1, 1); ps = p0.repeat(NENS, 1); cs = c0.repeat(NENS, 1)
        Xs = fn(xs, ps, cs)
        r = {k: dens(Xs[k], cs).view(NENS, B, 2, NB, NB) for k in (1, 10)}
        Jx, Jy = ng.face_flux(xs, Xs[1], cs, NB, LBOX)
        return ({k: r[k].mean(0) for k in r}, {k: r[k][0] for k in r},
                (Jx.view(NENS, B, 2, NB, NB).mean(0), Jy.view(NENS, B, 2, NB, NB).mean(0)), (Jx.view(NENS, B, 2, NB, NB)[0], Jy.view(NENS, B, 2, NB, NB)[0]))
    preds = {}
    o_mean, o_one, oJ_mean, oJ_one = ensemble(lambda x, p, c: ng.simulate(truth, x, p, c, 10, generator=gen)[0])
    preds['oracle_mean'] = (o_mean, oJ_mean); preds['oracle_single'] = (o_one, oJ_one)
    g_mean, g_one, gJ_mean, gJ_one = ensemble(lambda x, p, c: ng.rollout(gnn, x, p, c, 10, sigma=truth.p['sigma'], generator=gen)[0])
    preds['gnn_sde_mean'] = (g_mean, gJ_mean); preds['gnn_sde_single'] = (g_one, gJ_one)
    Xo = ng.rollout(gnn, x0, p0, c0, 10, sigma=0.0)[0]
    preds['gnn_ode'] = ({k: dens(Xo[k], c0) for k in (1, 10)}, ng.face_flux(x0, Xo[1], c0, NB, LBOX))
    # HyperME on the density
    r0 = dens(x0, c0)
    c = r0.reshape(B, -1).double() + th.EPS
    rh, Jh = {}, None
    for k in range(1, 11):
        if k == 1:
            c, Jbar = th.step(hm, c, flux=True)
            Jh = tuple(j.float() * H ** 2 for j in th.edge_to_faces(Jbar))
        else:
            c = th.step(hm, c)
        if k in (1, 10):
            rh[k] = (c - th.EPS).float().view(B, 2, NB, NB)
    preds['hyperme'] = (rh, Jh)
    zero = (torch.zeros_like(true_J[0]), torch.zeros_like(true_J[1]))
    preds['persist'] = ({1: r0, 10: r0}, zero)
    preds['uniform'] = ({k: r0.mean((-1, -2), keepdim=True).expand_as(r0) for k in (1, 10)}, zero)
    Jexp = oJ_mean
    circ_exp = circulation(*Jexp)
    for name, (rho, J) in preds.items():
        Jcat = torch.cat([J[0], J[1]], -1); Ttrue = torch.cat([true_J[0], true_J[1]], -1); Texp = torch.cat([Jexp[0], Jexp[1]], -1)
        circ = circulation(*J)
        out[name] = dict(rho1=summarize(rel(rho[1], true_rho[1])), rho10=summarize(rel(rho[10], true_rho[10])),
                         J1_vs_data=summarize(rel(Jcat, Ttrue)), J1_vs_expected=summarize(rel(Jcat, Texp)),
                         circ_corr_expected=summarize(corr(circ, circ_exp)),
                         circ_rms_ratio=float(circ.pow(2).mean().sqrt() / circ_exp.pow(2).mean().sqrt()))
    # time-averaged flux over t in [100, 299] per test trajectory (one-step predictions from the true snapshots)
    ta = {}
    for m in range(M):
        ts = torch.arange(100, 300)
        xs, xs1 = X[ts, m].to(dev), X[ts + 1, m].to(dev)
        cs = C[m][None].expand(len(ts), -1).to(dev)
        Jd = ng.face_flux(xs, xs1, cs, NB, LBOX)
        Xg = ng.rollout(gnn, xs, P[ts, m].to(dev), cs, 1, sigma=0.0)[0]
        Jg = ng.face_flux(xs, Xg[1], cs, NB, LBOX)
        cc = ng.bin_density(xs, cs, NB, LBOX).reshape(len(ts), -1).double() + th.EPS
        _, Jb = th.step(hm, cc, flux=True)
        Jm = tuple(j.float() * H ** 2 for j in th.edge_to_faces(Jb))
        for name, J in (('data', Jd), ('gnn_ode', Jg), ('hyperme', Jm)):
            Jbar = (J[0].mean(0, keepdim=True), J[1].mean(0, keepdim=True))
            ta.setdefault(name, []).append(dict(rms=float(torch.cat(Jbar, -1).pow(2).mean().sqrt()),
                                                circ_rms=float(circulation(*Jbar).pow(2).mean().sqrt()),
                                                Jbar=Jbar))
    tavg = {}
    for name in ta:
        tavg[name] = dict(rms=float(np.mean([e['rms'] for e in ta[name]])), circ_rms=float(np.mean([e['circ_rms'] for e in ta[name]])))
        if name != 'data':
            a = torch.cat([torch.cat(e['Jbar'], -1) for e in ta[name]]); b = torch.cat([torch.cat(e['Jbar'], -1) for e in ta['data']])
            tavg[name]['rel_l2_vs_data'] = float(np.mean(rel(a, b))); tavg[name]['corr_vs_data'] = float(np.mean(corr(a, b)))
    out['time_averaged_flux'] = tavg
    # HyperME-only readouts at the test snapshots
    ro = hm.readouts(r0.reshape(B, -1).double() + th.EPS, th.drive(B, dev))
    out['hyperme_readouts'] = dict(sigma=summarize(ro['sigma'].cpu().numpy()), excess=summarize(ro['excess'].cpu().numpy()),
                                   housekeeping=summarize(ro['housekeeping'].cpu().numpy()))
    out['n_cases'] = B
    return out, preds, true_rho, (x0, p0, c0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.path.expanduser('~/ext_work/taskC'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'runs', 'external', 'common'))
    ap.add_argument('--gnn', default='gnn_best.pt')
    ap.add_argument('--tag', default='')
    ap.add_argument('--hyperme', default='hyperme_fit.pt')
    a = ap.parse_args()
    dev = 'cuda'
    gen = torch.Generator(device=dev).manual_seed(123)
    truth = ng.MixedSpecies().to(dev)
    gnn = ng.NonreciprocalGNN().to(dev)
    ck = torch.load(os.path.join(a.work, a.gnn))
    gnn.load_state_dict(ck['state']); gnn.eval()
    hm = th.build_model().to(dev)
    hk = torch.load(os.path.join(a.work, a.hyperme))
    hm.load_state_dict(hk['state'])
    res = dict(hyperme_checkpoint=a.hyperme, gnn_checkpoint=dict(epoch=ck.get('epoch'), valid=ck.get('valid')),
               hyperme=dict(valid=hk['valid'], logk=hm.kinetics.core.tolist(), chi=hm.free_energy.chi.tolist(),
                            widths=list(th.WIDTHS), harmonic_b=hm.drive.b.tolist(), s_A=float(hm.drive.s_A)))
    # only-GNN: learned two-body functions against the truth (training data snapshots)
    d = torch.load(os.path.join(a.work, 'train.pt'))
    res['gnn_interactions'] = ng.compare_interactions(gnn, truth, d['x'], d['phi'], d['c'])
    keep = {}
    for name in ('test_id', 'test_ood_ratio'):
        dd = torch.load(os.path.join(a.work, f'{name}.pt'))
        t = time.time()
        r, preds, true_rho, st = eval_set(dd, gnn, hm, truth, dev, gen)
        res[name] = r
        keep[name] = dict(true10={k: v[0:3].cpu() for k, v in true_rho.items()},
                          pred10={k: (v[0][10][0:3].cpu() if torch.is_tensor(v[0][10]) else None) for k, v in preds.items()})
        print(name, f'{time.time() - t:.0f}s', json.dumps({k: (v['rho10']['median'], v['J1_vs_expected']['median']) for k, v in r.items() if isinstance(v, dict) and 'rho10' in v}), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f'taskC_results{a.tag}.json'), 'w'), indent=1)
    torch.save(keep, os.path.join(a.work, f'taskC_examples{a.tag}.pt'))


if __name__ == '__main__':
    main()
