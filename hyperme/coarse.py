"""Coarse-graining a learned linear master equation (Markov chain on the state components).

Convention: row generator Q (Q_ij = rate i -> j, rows sum to zero), stationary pi with pi Q = 0, partition of the
fine states into blocks given by labels (M,) with values in [0, Mc).  P = block-sum matrix (M, Mc).

projected_generator   Qbar_ab = sum_{i in a} (pi_i / pi_a) sum_{j in b} Q_ij   (stationary-weighted lumping, the
                      m-projection onto block-lumped chains).  Exact at the steady state: pibar = P^T pi is
                      stationary for Qbar, the block currents equal the summed fine currents, sigmabar <= sigma.
lagged_propagator     Kbar(tau) = Pi exp(tau Q) P with Pi_ai = pi_i / pi_a (Markov-state-model construction).
                      Exact stationary state and lag-tau currents; better transients when the blocks are not lumpable;
                      the per-lag entropy production is again a lower bound of the fine one.
coarse_parameters     the HyperME form of a coarse generator: phibar_a = -log sum_{i in a} exp(-phi_i),
                      kbar_ab = sqrt(Qbar_ab Qbar_ba), Abar_ab = log(Qbar_ab / Qbar_ba) - (phibar_a - phibar_b).
coarse_grain_model    builds the fine generator of a model (frozen at its stationary state), applies either rule and
                      checks the assertions pibar = P pi, Jbar = P J, sigmabar <= sigma.
Each function checks its assertions when check=True and raises AssertionError on failure.
"""
import numpy as np
import scipy.linalg as sla
import torch


def partition_matrix(labels, n_coarse=None):
    labels = np.asarray(labels)
    n_coarse = int(labels.max()) + 1 if n_coarse is None else n_coarse
    P = np.zeros((len(labels), n_coarse)); P[np.arange(len(labels)), labels] = 1.0
    return P


def grid_block_labels(n, m):
    """n x n grid nodes (row-major) -> labels of the (n/m) x (n/m) blocks of size m x m"""
    c = n // m
    return ((np.arange(n)[:, None] // m) * c + (np.arange(n)[None, :] // m)).ravel()


def stationary(Q):
    """pi with pi Q = 0, sum pi = 1 (dense solve)"""
    M = Q.T.copy(); M[-1] = 1.0
    r = np.zeros(len(Q)); r[-1] = 1.0
    return np.linalg.solve(M, r)


def pair_currents(pi, Q):
    """net stationary currents C_ij = pi_i Q_ij - pi_j Q_ji (antisymmetric, zero diagonal)"""
    F = pi[:, None] * Q
    np.fill_diagonal(F, 0.0)
    return F - F.T


def epr(pi, W):
    """sum_{i<j} (pi_i W_ij - pi_j W_ji) log(pi_i W_ij / pi_j W_ji) over pairs with both one-way fluxes > 0;
    W is a generator (sigma per unit time) or a propagator (sigma per lag)."""
    F = pi[:, None] * W
    np.fill_diagonal(F, 0.0)
    ok = (F > 0) & (F.T > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.where(ok, (F - F.T) * (np.log(np.where(ok, F, 1.0)) - np.log(np.where(ok, F.T, 1.0))), 0.0)
    return 0.5 * float(x.sum())


def projected_generator(Q, pi, labels, check=True, tol=1e-9):
    P = partition_matrix(labels)
    pib = P.T @ pi
    Pi = (P * pi[:, None]).T / pib[:, None]                    # (Mc, M), Pi_ai = pi_i / pi_a
    Qb = Pi @ Q @ P
    Jf = pair_currents(pi, Q); Jb = pair_currents(pib, Qb)
    out = dict(Qbar=Qb, pibar=pib, Jbar=Jb, PJ=P.T @ Jf @ P, sigma=epr(pi, Q), sigmabar=epr(pib, Qb), P=P)
    if check:
        scale = np.abs(pi[:, None] * Q).max()
        assert np.abs(Qb.sum(1)).max() <= tol * np.abs(Qb).max(), 'Qbar rows must sum to zero'
        assert np.abs(pib @ Qb).max() <= tol * scale, 'pibar = P pi is not stationary for Qbar'
        assert np.abs(Jb - out['PJ']).max() <= tol * scale, 'Jbar != P J'
        assert out['sigmabar'] <= out['sigma'] * (1 + tol) + tol * scale, 'sigmabar > sigma'
    return out


def lagged_propagator(Q, pi, labels, tau, check=True, tol=1e-9):
    P = partition_matrix(labels)
    pib = P.T @ pi
    Pi = (P * pi[:, None]).T / pib[:, None]
    K = sla.expm(tau * Q)
    Kb = Pi @ K @ P
    Jf = pair_currents(pi, K); Jb = pair_currents(pib, Kb)
    out = dict(Kbar=Kb, K=K, pibar=pib, Jbar=Jb, PJ=P.T @ Jf @ P, sigma=epr(pi, K), sigmabar=epr(pib, Kb), P=P,
               generator=None)
    if check:
        assert np.abs(Kb.sum(1) - 1).max() <= tol, 'Kbar is not row-stochastic'
        assert np.abs(pib @ Kb - pib).max() <= tol * pib.max(), 'pibar = P pi is not stationary for Kbar'
        assert np.abs(Jb - out['PJ']).max() <= tol * pib.max(), 'lag currents: Jbar != P J'
        assert out['sigmabar'] <= out['sigma'] * (1 + tol) + tol, 'sigmabar > sigma'
    try:
        Lg = np.real(sla.logm(Kb)) / tau
        out['generator'] = Lg
        out['generator_legal'] = bool((Lg - np.diag(np.diag(Lg)) >= -1e-10).all())
    except Exception:
        pass
    return out


def coarse_parameters(Qbar, phi_fine=None, labels=None, tol=0.0):
    """HyperME form of a coarse generator on the pairs (a < b) with Qbar_ab > tol and Qbar_ba > tol.
    phibar from the log-sum-exp rule when the fine excess potential phi is given, else phibar = -log pibar."""
    Mc = len(Qbar)
    if phi_fine is not None:
        P = partition_matrix(labels, Mc)
        m = phi_fine.min()
        phib = -np.log(P.T @ np.exp(-(phi_fine - m))) + m
    else:
        phib = -np.log(stationary(Qbar))
    iu, ju = np.nonzero(np.triu((Qbar > tol) & (Qbar.T > tol), 1))
    qf, qb = Qbar[iu, ju], Qbar[ju, iu]
    k = np.sqrt(qf * qb)
    A = np.log(qf / qb) - (phib[iu] - phib[ju])
    return dict(edges=np.stack([iu, ju], 1), k=k, A=A, phibar=phib)


def coarse_hypergraph(edges, n_states):
    from .complex import Hypergraph
    return Hypergraph.from_graph(edges, n_states)


@torch.no_grad()
def model_stationary(model, drive, c0, iters=20, tol=1e-10):
    """stationary state of a model's (frozen) linear generator, iterated to the fixed point pi = stat(Q(pi)) when the
    rates depend on c; the total mass of c0 is kept.  Returns (pi as (B, M) density, row generators (B, M, M) numpy)."""
    ctx = model.context(drive)
    c = c0.clone()
    for _ in range(iters):
        Lg = model.generator(c, ctx).double().cpu().numpy()
        Qs = np.transpose(Lg, (0, 2, 1))
        pis = np.stack([stationary(Qs[b]) for b in range(len(Qs))])
        new = torch.as_tensor(pis, dtype=c.dtype, device=c.device) * c0.sum(-1, keepdim=True)
        done = bool(((new - c).abs().amax(-1) <= tol * new.amax(-1)).all())
        c = new
        if done or not model.free_energy.c_dependent and not model.kinetics.rho_dependent:
            break
    return c, Qs


@torch.no_grad()
def coarse_grain_model(model, drive, c0, labels, method='generator', tau=None, check=True):
    """coarse-grain a model with a unimolecular hypergraph: frozen generator at its stationary state, then the
    projected generator ('generator') or the lag-tau propagator ('lag').  Returns one result dict per sample."""
    ctx = model.context(drive)
    pi_c, Qs = model_stationary(model, ctx, c0)
    phis = model.phi(pi_c, ctx).double().cpu().numpy()
    res = []
    for b in range(len(Qs)):
        pi = pi_c[b].double().cpu().numpy(); pi = pi / pi.sum()
        if method == 'generator':
            r = projected_generator(Qs[b], pi, labels, check=check)
            r['params'] = coarse_parameters(r['Qbar'], phis[b], labels)
        else:
            r = lagged_propagator(Qs[b], pi, labels, tau, check=check)
        r['pi'] = pi
        res.append(r)
    return res
