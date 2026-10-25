"""E4: closed-form coarse-graining (RG) of a trained TC-ME generator -- "coarse-grain the model, not the data".

Rule (projected generator / m-projection onto block-lumped Markov chains, Esposito PRE 85, 041125 (2012)):
    Qbar_ab = sum_{i in a} (pi_i / pi_a) sum_{j in b} Q_ij ,      pi_a = sum_{i in a} pi_i ,
with pi the stationary state of the learned fine generator (fixed point pi = stat(Q(pi)), since phi = alpha U + g(rho,U)
depends on rho).  Rho-dependence is handled by the local-equilibrium closure rho_i = m^2 rhobar_a pi_i / pi_a (density units;
block mean = rhobar_a), so the coarse model is Qbar(rhobar) = projection of Q(rho_closure(rhobar)).
TC-ME parametric form of Qbar (rates in code use phi = mu - log rho):
    (1) phibar_a = -log sum_{i in a} exp(-phi_i)                  (log-sum-exp free energy)
    (2) kbar_ab  = sqrt(Qbar_ab Qbar_ba) = Qbar_ab exp(-[(phibar_a - phibar_b) + Abar_ab]/2)   (symmetric prefactor)
    (3) Abar_ab  = log(Qbar_ab / Qbar_ba) - (phibar_a - phibar_b);  fbar_ab = log(pi_a Qbar_ab / pi_b Qbar_ba) = mubar_a - mubar_b + Abar_ab
Checked by assertion (float64): Ppi stationary for Qbar; Jbar_ab = sum of crossing fine stationary fluxes; sigmabar <= sigma
(log-sum inequality) along the chain 32 -> 16 -> 8; RG_4 = RG_2 o RG_2 (semigroup); Abar = 0 exactly if A = 0 (detailed
balance preserved, rule (1) is the right free energy); (k, phi, A) reconstructs Qbar.
  python rg_coarse.py            -> runs/rg/E4.json, E4.log
  python rg_coarse.py --summary  -> runs/rg/summary.md, epr_hierarchy.png, steady_flux_match.png"""
import argparse, json, os, sys, time, glob
import numpy as np
import torch
from common import *
import common, models, models2
from models2 import FluxGNN
from models import FisherHodge, GNNODE, logmean_e, NSUB, MAXSUB
from models_tcme import TCME
from coarse_test import set_rate_scale
from train_tcme import jclean, grid_edges
from train_coarse import coarse_set, MODELS
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis'))
import hodge_lib as HL

dev = 'cuda'
ODIR = os.path.join(RUNS, 'rg'); TDIR = os.path.join(RUNS, 'tcme')
FIELDS = ['U', 'dpsi', 'psin']
SETS = ['test_id', 'test_oodpot']
NSS = 400            # extra steps (t = 20) from the t = 10 state to reach the stationary state for rollout-only models


# ============================================================ closed-form RG rule
def up(x, m):
    return x.repeat_interleave(m, -2).repeat_interleave(m, -1)


def block_sum(x, m):
    n = x.shape[-1]; c = n // m
    return x.reshape(*x.shape[:-2], c, m, c, m).sum((-3, -1))


def block_weights(pi, m):
    """w_i = pi_i / pi_a  (pi in any units)"""
    return pi / up(block_sum(pi, m), m)


def cross_sum(J, m):
    """fine edge field (B,2,n,n) -> sum over the m fine edges crossing each coarse edge, (B,2,n/m,n/m)."""
    B, _, n, _ = J.shape; c = n // m; sl = slice(m - 1, None, m)
    Jh = J[:, 0][..., :, sl].reshape(B, c, m, c).sum(2)
    Jv = J[:, 1][..., sl, :].reshape(B, c, c, m).sum(3)
    return torch.stack([Jh, Jv], 1)


def rg_rates(qf, qb, w, m):
    """fine one-way edge rates (qf: tail->head, qb: head->tail; (B,2,n,n)), in-block weights w (B,n,n)
    -> coarse one-way rates for m x m blocks:  Qbar_ab = sum_{i in a} w_i sum_{j in b} Q_ij  (nearest-neighbour blocks only)."""
    B, _, n, _ = qf.shape; c = n // m; sl = slice(m - 1, None, m)
    Qf_h = (w[..., :, sl] * qf[:, 0][..., :, sl]).reshape(B, c, m, c).sum(2)
    Qb_h = (w.roll(-1, -1)[..., :, sl] * qb[:, 0][..., :, sl]).reshape(B, c, m, c).sum(2)
    Qf_v = (w[..., sl, :] * qf[:, 1][..., sl, :]).reshape(B, c, c, m).sum(3)
    Qb_v = (w.roll(-1, -2)[..., sl, :] * qb[:, 1][..., sl, :]).reshape(B, c, c, m).sum(3)
    return torch.stack([Qf_h, Qf_v], 1), torch.stack([Qb_h, Qb_v], 1)


def lse_free_energy(phi, m):
    """rule (1): phibar_a = -log sum_{i in a} exp(-phi_i)"""
    return -torch.log(block_sum(torch.exp(-(phi - phi.amin((-1, -2), keepdim=True))), m)) + phi.amin((-1, -2), keepdim=True)


def param_form(Qf, Qb, phibar):
    """rules (2),(3): (kbar, Abar) such that Qf = k exp((-dphi + A)/2), Qb = k exp((dphi - A)/2)."""
    k = torch.sqrt(Qf * Qb)
    A = torch.log(Qf / Qb) + grad_e(phibar)
    return k, A


def edge_flux(rho, qf, qb):
    rh = torch.stack([rho.roll(-1, -1), rho.roll(-1, -2)], -3)
    a, b = rho.unsqueeze(-3) * qf, rh * qb
    return a, b


def sigma_of(rho, qf, qb):
    """Schnakenberg EPR in probability units: h^2 sum_e (a-b) log(a/b)  (density units -> p = rho h^2)."""
    n = rho.shape[-1]; a, b = edge_flux(rho, qf, qb)
    return ((a - b) * (torch.log(a) - torch.log(b))).sum((-3, -2, -1)) / n ** 2


def escape(qf, qb):
    return qf[:, 0] + qf[:, 1] + qb[:, 0].roll(1, -1) + qb[:, 1].roll(1, -2)


class RGTCME:
    """coarse TC-ME obtained in closed form from a trained fine TC-ME (no coarse data).  frozen=True: generator frozen at
    Q(pi) (linear coarse master equation); frozen=False: local-equilibrium closure rho_i = m^2 rhobar_a pi_i/pi_a."""
    def __init__(s, fine, Ffine, pi, m, frozen=False, kscale=1.0):
        s.fine, s.m, s.frozen, s.kscale = fine, m, frozen, kscale
        with torch.no_grad():
            s.ctx = fine.prepare(Ffine)
        s.w = block_weights(pi, m)
        if frozen:
          with torch.no_grad():
            s.Q0 = rg_rates(*fine.rates_e(pi, s.ctx), s.w, m)

    def rates_e(s, rb, ctx=None):
        if s.frozen:
            return s.Q0
        qf, qb = s.fine.rates_e(up(rb, s.m) * s.w * s.m ** 2, s.ctx)
        Qf, Qb = rg_rates(qf, qb, s.w, s.m)
        return (Qf, Qb) if s.kscale == 1.0 else (s.kscale * Qf, s.kscale * Qb)

    flux_out = TCME.flux_out
    advance = TCME.advance


# ============================================================ stationary states
def dense_from_rates(qf, qb):
    """numpy (2,n,n) one-way rates -> dense generator (n^2, n^2)"""
    n = qf.shape[-1]; E = grid_edges(n)
    Q = np.zeros((n * n, n * n))
    Q[E[:, 0], E[:, 1]] = qf.ravel(); Q[E[:, 1], E[:, 0]] = qb.ravel()
    Q[np.arange(n * n), np.arange(n * n)] = -Q.sum(1)
    return Q


def stat_dense(Q):
    M = Q.T.copy(); M[-1] = 1.0
    r = np.zeros(len(Q)); r[-1] = 1.0
    return np.linalg.solve(M, r)


@torch.no_grad()
def tcme_stationary(model, ctx, B, n, iters=12, tol=1e-6):
    """fixed point pi = stat(Q(pi)).  Returns density pi (B,n,n) float64 and the float64 one-way rates Q(rho_prev) for which pi
    is the exact stationary state (these are 'the' learned generator at the steady state)."""
    rho = torch.ones((B, n, n), device=dev)
    pis = np.zeros((B, n, n)); qfs = np.zeros((B, 2, n, n)); qbs = np.zeros((B, 2, n, n)); its = []
    done = np.zeros(B, bool)
    for it in range(iters):
        qf, qb = model.rates_e(rho, ctx)
        qf = qf.double().cpu().numpy(); qb = qb.double().cpu().numpy()
        for bi in range(B):
            if done[bi]:
                continue
            p = stat_dense(dense_from_rates(qf[bi], qb[bi])) * n * n
            if it > 0 and np.abs(p - pis[bi].ravel()).max() < tol * p.max():
                done[bi] = True; its.append(it)
            pis[bi] = p.reshape(n, n); qfs[bi] = qf[bi]; qbs[bi] = qb[bi]
        if done.all():
            break
        rho = torch.tensor(pis, device=dev, dtype=torch.float32)
    return pis, qfs, qbs, (max(its) if len(its) == B else -1)


@torch.no_grad()
def roll_to_stationary(model, F, rho0, nsteps=NSS):
    ctx = model.ctx if isinstance(model, RGTCME) else model.prepare(F); r = rho0.clone()
    for k in range(nsteps):
        rp = r
        r, _ = model.advance(r, ctx, DT)
    res = (torch.linalg.vector_norm((r - rp).flatten(1), dim=1) / torch.linalg.vector_norm(r.flatten(1), dim=1)).cpu().numpy()
    return r, ctx, res     # res: relative change over the last DT


# ============================================================ readouts for the rollout-type models
@torch.no_grad()
def flux_phys(model, rho, ctx):
    n = rho.shape[-1]
    if isinstance(model, (TCME, RGTCME)):
        return model.flux_out(rho, ctx)[0] / n
    if isinstance(model, GNNODE):
        return None
    return -(D * n) * model.flux_q(rho, ctx)


@torch.no_grad()
def sigma_model(model, rho, ctx):
    """model's own EPR (probability units).  TC-ME: Schnakenberg; Fisher-Hodge: J_nat . f with f = J_nat/(S kappa Theta)
    = -(d mu + c/kappa), i.e. D sum kappa Theta (d mu + c/kappa)^2; Flux-GNN: *formal* Schnakenberg EPR of its directed
    softplus rates (no free energy / LDB behind it); GNN-ODE: none."""
    n = rho.shape[-1]; S = D * n * n
    if isinstance(model, TCME):
        return sigma_of(rho, *model.rates_e(rho, ctx))
    if isinstance(model, FisherHodge):
        mu = model.mu(rho, ctx['F']); th = logmean_e(rho)
        x = model.flux_q(rho, ctx, mu) / (ctx['kap'] * th)          # = d mu + c/kappa
        return D * (ctx['kap'] * th * x * x).sum((-3, -2, -1))
    if isinstance(model, FluxGNN):
        a = model.rates(rho, ctx)
        af = torch.stack([rho * a[:, 0], rho * a[:, 2]], 1) * S
        ab = torch.stack([rho.roll(-1, -1) * a[:, 1].roll(-1, -1), rho.roll(-1, -2) * a[:, 3].roll(-1, -2)], 1) * S
        return ((af - ab) * (torch.log(af) - torch.log(ab))).sum((-3, -2, -1)) / n ** 2
    return None


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return np.linalg.norm((a - b).reshape(a.shape[0], -1), axis=1) / np.linalg.norm(b.reshape(b.shape[0], -1), axis=1)


def rollout_err(model, F, rho0, Y):
    set_n = None
    ctx = model.ctx if isinstance(model, RGTCME) else model.prepare(F)
    r = rho0.clone(); out = [r.cpu()]
    with torch.no_grad():
        for k in range(Y.shape[1] - 1):
            r, _ = model.advance(r, ctx, DT); out.append(r.cpu())
    R = torch.stack(out, 1).double()
    B, T1 = Y.shape[:2]
    err = (torch.linalg.vector_norm((R - Y).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T1, -1), dim=-1)).numpy()
    Rn = R.numpy()
    return dict(relerr_median=np.median(err, 0), relerr_10T_per_traj=err[:, -1], mass_err_max=float(np.abs(Rn.mean((2, 3)) - 1).max()),
                min_rho=float(Rn.min()), n_traj_with_negative=int((Rn < 0).any((1, 2, 3)).sum()))


# ============================================================ main experiment
def run(args):
    data = torch.load(os.path.join(RUNS, 'data', 'S2.pt'))
    LOG = open(os.path.join(ODIR, 'E4.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

    out = dict(rg={}, scratch={}, fine={}, ref={}, asserts={})
    AS = dict(stat_resid=[], stat_dense_relerr=[], flux_exact_relerr=[], sigma16_over_32=[], sigma8_over_16=[], semigroup_relerr=[],
              eq_Abar_max=[], eq_A_fine_max=[], reconstruct_relerr=[], ksym=[], Abar_grad_frac=[], Abar_grad_frac_fine=[],
              g_rho_sensitivity=[], fp_iters=[], kbar_over_Dn2=[], kfine_over_Dn2=[])
    for setname in SETS:
        sd = data[setname]; F32 = {k: sd[k].to(dev) for k in FIELDS}; B = sd['rho'].shape[0]
        s16 = coarse_set(sd, 16); s8 = coarse_set(sd, 8)
        ref = torch.load(os.path.join(TDIR, f'ref_ss_{setname}.pt'))
        rr32 = ref['rho_ss'].double(); jr32 = ref['J_ss'].double()
        rr16, jr16 = coarsen(rr32), coarsen_flux(jr32); rr8, jr8 = coarsen(rr16), coarsen_flux(jr16)
        sref = lambda r, j: ((j ** 2 / (D * mean_e(r))).sum((-3, -2, -1)) / r.shape[-1] ** 2).numpy()
        out['ref'][setname] = dict(sigma64=ref['sigma_ref_fine'], sigma32=sref(rr32, jr32), sigma16=sref(rr16, jr16), sigma8=sref(rr8, jr8))
        # -------------------------------------------------- (a) RG-TC-ME from each fine TC-ME seed
        for seed in args.seeds:
            key = f'tcme_s{seed}'
            fine = TCME('S2').to(dev); fine.load_state_dict(torch.load(os.path.join(TDIR, f'S2_tcme_s{seed}.pt'))); fine.eval()
            with torch.no_grad():
                ctx = fine.prepare(F32)
            t0 = time.time()
            pi32, qf, qb, nit = tcme_stationary(fine, ctx, B, N)
            AS['fp_iters'].append(nit)
            P32 = torch.tensor(pi32); qf = torch.tensor(qf); qb = torch.tensor(qb)
            a32, b32 = edge_flux(P32, qf, qb); J32 = a32 - b32                     # fine native stationary flux (density units)
            sig32 = sigma_of(P32, qf, qb)
            # 32 -> 16
            w2 = block_weights(P32, 2); Qf16, Qb16 = rg_rates(qf, qb, w2, 2); P16 = coarsen(P32)
            a16, b16 = edge_flux(P16, Qf16, Qb16); J16 = a16 - b16
            sig16 = sigma_of(P16, Qf16, Qb16)
            res = div_T(J16).abs().amax((-1, -2)) / (P16 * escape(Qf16, Qb16)).amax((-1, -2))
            AS['stat_resid'] += res.tolist()
            pd = np.stack([stat_dense(dense_from_rates(Qf16[b].numpy(), Qb16[b].numpy())).reshape(16, 16) * 256 for b in range(B)])
            AS['stat_dense_relerr'] += rel(pd, P16.numpy()).tolist()
            AS['flux_exact_relerr'] += rel(J16.numpy(), (cross_sum(J32, 2) / 4).numpy()).tolist()
            # 16 -> 8 (chain) and 32 -> 8 (direct)
            Qf8c, Qb8c = rg_rates(Qf16, Qb16, block_weights(P16, 2), 2)
            Qf8, Qb8 = rg_rates(qf, qb, block_weights(P32, 4), 4); P8 = coarsen(P16)
            AS['semigroup_relerr'] += rel(torch.cat([Qf8c, Qb8c], 1).numpy(), torch.cat([Qf8, Qb8], 1).numpy()).tolist()
            sig8 = sigma_of(P8, Qf8, Qb8)
            a8, b8 = edge_flux(P8, Qf8, Qb8)
            AS['flux_exact_relerr'] += rel((a8 - b8).numpy(), (cross_sum(J32, 4) / 16).numpy()).tolist()
            AS['sigma16_over_32'] += (sig16 / sig32).tolist(); AS['sigma8_over_16'] += (sig8 / sig16).tolist()
            # parametric form: (1)-(3)
            with torch.no_grad():
                rprev = torch.tensor(pi32, device=dev, dtype=torch.float32)   # (pi itself; Q(pi) differs from Q(rho_prev) by < tol)
                phi32 = fine.phi(rprev, F32).double().cpu()
                eps = 1e-3   # g's rho-sensitivity: |phi(1.001 pi) - phi(pi)| relative to |d phi| across edges
                dphi_rho = (fine.phi(rprev * (1 + eps), F32) - fine.phi(rprev, F32)).double().cpu()
            AS['g_rho_sensitivity'] += (dphi_rho.abs().amax((-1, -2)) / eps / grad_e(phi32).abs().amax((-3, -2, -1))).tolist()
            phib = lse_free_energy(phi32, 2)
            kb, Ab = param_form(Qf16, Qb16, phib)
            Qf_r = kb * torch.exp(0.5 * (-grad_e(phib) + Ab)); Qb_r = kb * torch.exp(0.5 * (grad_e(phib) - Ab))
            AS['reconstruct_relerr'] += rel(torch.cat([Qf_r, Qb_r], 1).numpy(), torch.cat([Qf16, Qb16], 1).numpy()).tolist()
            AS['kbar_over_Dn2'] += (kb.flatten(1).median(1).values / (D * 16 ** 2)).tolist()
            AS['kfine_over_Dn2'] += (torch.sqrt(qf * qb).flatten(1).median(1).values / (D * N ** 2)).tolist()
            AS['ksym'] += [float((kb - torch.sqrt(Qb16 * Qf16)).abs().max() / kb.abs().max())] * B
            fbar = torch.log(a16 / b16)            # = mubar_a - mubar_b + Abar,  mubar = log rhobar + phibar
            mub = torch.log(P16) + phib
            assert float((fbar - (-grad_e(mub) + Ab)).abs().max()) < 1e-9 * float(fbar.abs().max() + 1)
            E16 = grid_edges(16); d016 = HL.incidence(E16, 256); E32 = grid_edges(N); d032 = HL.incidence(E32, N * N)
            for b in range(B):
                Av = Ab[b].numpy().ravel(); g, _ = HL.hodge(Av, np.ones_like(Av), d016)
                AS['Abar_grad_frac'].append(float((g ** 2).sum() / (Av ** 2).sum()))
                if b == 0:
                    Af = ctx['A'][b].double().cpu().numpy().ravel(); gf, _ = HL.hodge(Af, np.ones_like(Af), d032)
                    AS['Abar_grad_frac_fine'].append(float((gf ** 2).sum() / (Af ** 2).sum()))
            # equilibrium check: A := 0 in the fine model -> Abar must vanish with phibar = lse(phi)
            if setname == 'test_id':
                ctx0 = dict(ctx); ctx0['A'] = torch.zeros_like(ctx['A'])
                pe, qfe, qbe, _ = tcme_stationary(fine, ctx0, B, N)
                pe = torch.tensor(pe); qfe = torch.tensor(qfe); qbe = torch.tensor(qbe)
                Qfe, Qbe = rg_rates(qfe, qbe, block_weights(pe, 2), 2)
                # phi at the density where the rates were evaluated (rho_prev ~ pi within tol): recover it from the rates exactly
                # (A = 0 => log(qf/qb) = -d phi; integrate on the torus)
                with torch.no_grad():
                    phie = fine.phi(pe.to(dev, torch.float32), F32).double().cpu()
                Abe = param_form(Qfe, Qbe, lse_free_energy(phie, 2))[1]
                lr_f = torch.log(qfe / qbe) + grad_e(phie)          # fine-level analogue (float32 phi evaluation noise floor)
                AS['eq_Abar_max'] += Abe.abs().amax((-3, -2, -1)).tolist()
                AS['eq_A_fine_max'] += lr_f.abs().amax((-3, -2, -1)).tolist()
            log(f'[{setname} {key}] fp iters {nit} | stat resid {res.max():.1e} | sigma32 {sig32.median():.4f} sigma16 {sig16.median():.4f} '
                f'sigma8 {sig8.median():.4f} | {time.time() - t0:.0f}s')
            # rollouts on the 16x16 coarse test set
            Y16 = s16['rho'].double(); r0 = s16['rho'][:, 0].to(dev)
            Pt = torch.tensor(pi32, device=dev, dtype=torch.float32)
            R = {}
            for lab, fr, ks in [('closure', False, 1.0), ('frozen', True, 1.0), ('closure_k_over_m', False, 0.5)]:
                rg = RGTCME(fine, F32, Pt, 2, frozen=fr, kscale=ks)
                R[lab] = rollout_err(rg, None, r0, Y16)
                log(f'   RG-{lab}: relerr@T,2T,5T,10T ' + ', '.join(f'{R[lab]["relerr_median"][t]:.3e}' for t in [20, 40, 100, 200]) +
                    f' | mass {R[lab]["mass_err_max"]:.1e} min {R[lab]["min_rho"]:.2e} neg {R[lab]["n_traj_with_negative"]}')
            # RG stationary also reached by the closure dynamics (from the t=10 coarse state)
            rg = RGTCME(fine, F32, Pt, 2)
            rs, _, rres = roll_to_stationary(rg, None, s16['rho'][:, -1].to(dev))
            jrg = (J16 / 16).numpy()                                     # physical flux per unit length on the 16 grid
            out['rg'][key] = out['rg'].get(key, {})
            out['rg'][key][setname] = dict(
                rollout=R, sigma32=sig32.numpy(), sigma16=sig16.numpy(), sigma8=sig8.numpy(),
                pi_vs_Ppi=rel(pd, P16.numpy()), pi_rollout_vs_Ppi=rel(rs.double().cpu().numpy(), P16.numpy()), pi_rollout_resid=rres,
                J_vs_PJ=rel(jrg, (coarsen_flux(J32 / N)).numpy()),
                pi_vs_true=rel(P16.numpy(), rr16.numpy()), J_vs_true=rel(jrg, jr16.numpy()),
                pi32_vs_true=rel(pi32, rr32.numpy()), J32_vs_true=rel((J32 / N).numpy(), jr32.numpy()),
                pi8_vs_true=rel(P8.numpy(), rr8.numpy()))
            if setname == 'test_id' and seed == args.seeds[0]:
                np.savez(os.path.join(ODIR, 'flux_scatter_rg.npz'), J=jrg, PJ=coarsen_flux(J32 / N).numpy(), Jtrue=jr16.numpy())
            out['fine'][key] = out['fine'].get(key, {}); out['fine'][key][setname] = dict(pi=pi32, J=(J32 / N).numpy())
        # -------------------------------------------------- fine FH / Flux-GNN stationary on 32 (for their own cross-scale check)
        for m in ['fisher_hodge', 'flux_gnn']:
            for seed in args.seeds:
                key = f'{m}_s{seed}'
                set_rate_scale(N)
                mod = MODELS[m]('S2').to(dev); mod.load_state_dict(torch.load(os.path.join(TDIR, f'S2_{m}_s{seed}.pt'))); mod.eval()
                rs, c, rres = roll_to_stationary(mod, F32, sd['rho'][:, -1].to(dev))
                sg = sigma_model(mod, rs, c).double().cpu().numpy()
                out['fine'][key] = out['fine'].get(key, {})
                out['fine'][key][setname] = dict(pi=rs.double().cpu().numpy(), J=flux_phys(mod, rs, c).double().cpu().numpy(), sigma32=sg, resid=rres)
                log(f'[{setname} fine {key}] stationary resid {rres.max():.1e} sigma32 {np.median(sg):.4f}')
        # -------------------------------------------------- (c) scratch models on 16 and 8
        for m in ['tcme', 'fisher_hodge', 'flux_gnn', 'gnn_ode']:
            for seed in args.seeds:
                key = f'{m}_s{seed}'; rec = {}
                for n, sc in [(16, s16), (8, s8)]:
                    ck = os.path.join(ODIR, f'S2_{m}_n{n}_s{seed}.pt')
                    if not os.path.exists(ck):
                        continue
                    set_rate_scale(n)
                    mod = MODELS[m]('S2').to(dev); mod.load_state_dict(torch.load(ck)); mod.eval()
                    Fc = {k: sc[k].to(dev) for k in FIELDS}
                    rs, c, rres = roll_to_stationary(mod, Fc, sc['rho'][:, -1].to(dev))
                    if m == 'tcme':      # exact stationary state of the learned coarse generator
                        pid, qfc, qbc, _ = tcme_stationary(mod, c, B, n)
                        pi_c = pid
                        sg = sigma_of(torch.tensor(pid), torch.tensor(qfc), torch.tensor(qbc)).numpy()
                        J = (edge_flux(torch.tensor(pid), torch.tensor(qfc), torch.tensor(qbc))[0] - edge_flux(torch.tensor(pid), torch.tensor(qfc), torch.tensor(qbc))[1]).numpy() / n
                    else:
                        pi_c = rs.double().cpu().numpy()
                        s_ = sigma_model(mod, rs, c); sg = None if s_ is None else s_.double().cpu().numpy()
                        Jt = flux_phys(mod, rs, c); J = None if Jt is None else Jt.double().cpu().numpy()
                    rr, jr = (rr16, jr16) if n == 16 else (rr8, jr8)
                    d = dict(sigma=sg, pi_resid=rres, pi_vs_true=rel(pi_c, rr.numpy()), J_vs_true=None if J is None else rel(J, jr.numpy()))
                    # cross-scale consistency vs the fine (32x32) model of the same class (gnn_ode: no fine model)
                    fk = out['fine'].get(key, {}).get(setname)
                    if fk is not None:
                        Pp = torch.tensor(fk['pi']); PJ = torch.tensor(fk['J'])
                        f = N // n
                        Pp = block_sum(Pp, f) / f ** 2
                        PJc = PJ
                        for _ in range(int(np.log2(f))):
                            PJc = coarsen_flux(PJc)
                        d['pi_vs_Ppi'] = rel(pi_c, Pp.numpy())
                        d['J_vs_PJ'] = None if J is None else rel(J, PJc.numpy())
                    # also vs P of the fine TC-ME (common yardstick)
                    ft = out['fine'][f'tcme_s{seed}'][setname]; f = N // n
                    d['pi_vs_Ppi_tcme'] = rel(pi_c, (block_sum(torch.tensor(ft['pi']), f) / f ** 2).numpy())
                    if n == 16 and setname == 'test_id' and seed == args.seeds[0] and J is not None:
                        np.savez(os.path.join(ODIR, f'flux_scatter_{m}.npz'), J=J, PJ=PJc.numpy() if fk is not None else np.nan, Jtrue=jr16.numpy())
                    rec[n] = d
                    log(f'[{setname} scratch {key} n{n}] resid {rres.max():.1e} | pi vs Ppi {np.median(d.get("pi_vs_Ppi", [np.nan])):.3e} | '
                        f'J vs PJ {np.median(d["J_vs_PJ"]) if d.get("J_vs_PJ") is not None else float("nan"):.3e} | '
                        f'sigma {np.median(sg) if sg is not None else float("nan"):.4f}')
                set_rate_scale(N)
                out['scratch'][key] = out['scratch'].get(key, {}); out['scratch'][key][setname] = rec
    for k in list(out['fine']):
        for s in out['fine'][k]:
            out['fine'][k][s].pop('pi', None); out['fine'][k][s].pop('J', None)
    out['asserts'] = AS
    mx = lambda k: float(np.max(AS[k])) if AS[k] else float('nan')
    log('ASSERTS: stat_resid max %.2e | stat_dense_relerr max %.2e | flux_exact max %.2e | sigma16/32 max %.4f | sigma8/16 max %.4f | '
        'semigroup max %.2e | eq Abar max %.2e (fine-level floor %.2e) | reconstruct %.2e | ksym %.1e | Abar grad frac median %.3f (fine A %.1e) | '
        'g rho-sensitivity max %.2e | fp iters %s | kbar/(D n16^2) median %.3f, kfine/(D n32^2) %.3f' % (mx('stat_resid'), mx('stat_dense_relerr'), mx('flux_exact_relerr'), mx('sigma16_over_32'),
        mx('sigma8_over_16'), mx('semigroup_relerr'), mx('eq_Abar_max'), mx('eq_A_fine_max'), mx('reconstruct_relerr'), mx('ksym'),
        float(np.median(AS['Abar_grad_frac'])), mx('Abar_grad_frac_fine'), mx('g_rho_sensitivity'), AS['fp_iters'], float(np.median(AS['kbar_over_Dn2'])), float(np.median(AS['kfine_over_Dn2']))))
    json.dump(jclean(out), open(os.path.join(ODIR, 'E4.json'), 'w'))
    assert mx('stat_resid') < 1e-9, 'Ppi not stationary for Qbar'
    assert mx('stat_dense_relerr') < 1e-9, 'dense stationary of Qbar != Ppi'
    assert mx('flux_exact_relerr') < 1e-9, 'coarse stationary flux not exact'
    assert mx('sigma16_over_32') <= 1 + 1e-12 and mx('sigma8_over_16') <= 1 + 1e-12, 'EPR increased under coarse-graining'
    assert mx('semigroup_relerr') < 1e-9, 'RG_4 != RG_2 o RG_2'
    assert mx('reconstruct_relerr') < 1e-12, 'parametric form does not reconstruct Qbar'
    log('all assertions passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--summary', action='store_true')
    args = p.parse_args()
    if args.summary:
        from rg_summary import summary
        summary()
    else:
        run(args)
