"""TC-ME: thermodynamically consistent learned master-equation operator (2026-10-24).

    Q_{i->j} = k_e exp( (phi_i - phi_j)/2 + A_e/2 ),   Q_{j->i} = k_e exp( (phi_j - phi_i)/2 - A_e/2 )
    J_e = rho_i Q_{i->j} - rho_j Q_{j->i}   (edge e = (tail i, head j), net flow i->j),   d rho/dt = d^T J

  * k_e > 0 symmetric (edge MLP on orientation-invariant features, softplus, times D/h^2).
  * phi = alpha U + g_theta(rho, U)  (node-local, so phi = dG/drho for G = int g: a local free energy);
    chemical potential mu = log rho + phi.
  * A = d1^T psi_theta + sum_k b_k h_k  (antisymmetric 1-cochain in cycle space): psi_theta lives on faces
    (= dual vertices / corners), h_x, h_y are the two harmonic (constant) 1-cochains of the torus.
  * f_e = log(rho_i Q_ij / rho_j Q_ji) = (mu_i - mu_j) + A_e  (exact part + cycle part), and the Onsager
    coefficient ell_e = Lambda(rho_i Q_ij, rho_j Q_ji) (log mean of the two one-way fluxes) -- the same objects
    as analysis/hodge_lib.py.
  * g = 0 => linear master equation; mass is conserved exactly, and each Euler stage with dt * max_i sum_j Q_ij <= 1
    is a non-negative linear map, so the SSP-RK2 integrator below is positivity preserving without clamping.
  * Resolution independent: n and h = 1/n are read from the input shape (rate scale D n^2).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from common import D, grad_e, div_T, mean_e, dual_dpsi, corner_to_node, curl_f
from models import MLP, static_edge_feats, NSUB, MAXSUB, rho_scale


def node_to_corner(u):
    """node (...,n,n) -> corner/face (...,n,n); corner (a,b) is the centre of the face with lower-left node (a,b)."""
    return 0.25 * (u + u.roll(-1, -1) + u.roll(-1, -2) + u.roll(-1, -1).roll(-1, -2))


def d1T(psi):
    """face 0-cochain (...,n,n) -> edge 1-cochain (...,2,n,n): adjoint of curl_f (= dual_dpsi)."""
    return dual_dpsi(psi)


def stream_from_dpsi(dp):
    """invert dual_dpsi on the periodic grid: edge (...,2,n,n) -> corner stream function (...,n,n), mean zero.
    dp_h[a,b] = P[a,b] - P[a-1,b],  dp_v[a,b] = -(P[a,b] - P[a,b-1])."""
    dh, dv = dp[..., 0, :, :], dp[..., 1, :, :]
    z = torch.zeros_like(dv[..., :1, :1])
    row0 = torch.cat([z[..., 0, :], -torch.cumsum(dv[..., 0, 1:], -1)], -1)            # P[0,b] - P[0,0]
    col = torch.cat([torch.zeros_like(dh[..., :1, :]), torch.cumsum(dh[..., 1:, :], -2)], -2)   # P[a,b] - P[0,b]
    P = row0.unsqueeze(-2) + col
    return P - P.mean((-1, -2), keepdim=True)


class TCME(nn.Module):
    name = 'tcme'

    def __init__(s, sysname='S2', H=128):
        super().__init__()
        s.sys, s.RS = sysname, rho_scale(sysname)
        s.k = MLP(6, H, 1)
        nn.init.zeros_(s.k.net[-1].weight); nn.init.constant_(s.k.net[-1].bias, math.log(math.e - 1))  # k = D/h^2 at init
        s.g = MLP(2, H, 1, zero_last=True)
        s.alpha = nn.Parameter(torch.tensor(1.0))
        s.psi = MLP(2, H, 1, zero_last=True)            # face field from (stream function, U) at the face centre
        s.gam = nn.Parameter(torch.tensor(0.0))         # linear skip psi += gam * P
        s.v = nn.Parameter(torch.zeros(2))              # harmonic part, as a velocity (units D / length): A_h = h v
        s.Wb = nn.Parameter(torch.zeros(2, 2))          # harmonic response to the global mean of dpsi
        s.sub_log = []

    # ---------------------------------------------------------------- static (field-dependent) parts
    def prepare(s, F):
        U = F['U']; n = U.shape[-1]; h = 1.0 / n; S = D * n * n
        sym, _ = static_edge_feats(F)
        k = S * Fn.softplus(s.k(sym).squeeze(-1))
        P = stream_from_dpsi(F['dpsi'])
        Uc = node_to_corner(U)
        psi = s.gam * P + s.psi(torch.stack([P, Uc], -1)).squeeze(-1)
        Acyc = d1T(psi)
        mdp = F['dpsi'].mean((-1, -2))                               # (B,2) global mean of dpsi per direction
        b = h * s.v.view(1, 2) + mdp @ s.Wb.T                        # (B,2) harmonic coefficients (per edge)
        A = Acyc + b[:, :, None, None]
        return dict(F=F, k=k, A=A, Acyc=Acyc, b=b, psi=psi, S=S, h=h, n=n)

    def phi(s, rho, F):
        return s.alpha * F['U'] + s.g(torch.stack([rho / s.RS, F['U']], -1)).squeeze(-1)

    def mu(s, rho, F):
        return torch.log(rho) + s.phi(rho, F)

    def rates_e(s, rho, ctx):
        """one-way edge rates (Q_fwd tail->head, Q_bwd head->tail), each (B,2,n,n)."""
        dphi = grad_e(s.phi(rho, ctx['F']))           # phi_head - phi_tail
        x = 0.5 * (-dphi + ctx['A'])
        return ctx['k'] * torch.exp(x), ctx['k'] * torch.exp(-x)

    def flux_out(s, rho, ctx):
        qf, qb = s.rates_e(rho, ctx)
        rh = torch.stack([rho.roll(-1, -1), rho.roll(-1, -2)], -3)
        J = rho.unsqueeze(-3) * qf - rh * qb
        out = qf[:, 0] + qf[:, 1] + qb[:, 0].roll(1, -1) + qb[:, 1].roll(1, -2)   # total escape rate of each node
        return J, out

    def flux(s, rho, ctx):
        return s.flux_out(rho, ctx)[0]

    def flux_q(s, rho, ctx):
        """same sign/scale convention as FisherHodge.flux_q (rate = -S d^T q): q = -J / S."""
        return -s.flux(rho, ctx) / ctx['S']

    def rate(s, rho, ctx):
        return div_T(s.flux(rho, ctx))

    def edge_thermo(s, rho, ctx):
        """a, b one-way fluxes, J, force f = log(a/b), Onsager coefficient ell = logmean(a, b)."""
        qf, qb = s.rates_e(rho, ctx)
        rh = torch.stack([rho.roll(-1, -1), rho.roll(-1, -2)], -3)
        a, b = rho.unsqueeze(-3) * qf, rh * qb
        f = torch.log(a) - torch.log(b)
        sm = f.abs() < 1e-6
        ell = torch.where(sm, 0.5 * (a + b), (a - b) / torch.where(sm, torch.ones_like(f), f))
        return a, b, a - b, f, ell

    def advance(s, rho, ctx, Dt):
        """SSP-RK2 = average of rho and two Euler stages; each stage uses dt * max escape rate <= 1 (non-negative map)."""
        t, nsub, dt_nom = 0.0, 0, Dt / NSUB
        while Dt - t > 1e-9:
            J0, out0 = s.flux_out(rho, ctx)
            with torch.no_grad():
                dt_safe = 0.9 / out0.max().item()
            dt = min(dt_nom, Dt - t, dt_safe)
            while True:
                r1 = rho + dt * div_T(J0)
                J1, out1 = s.flux_out(r1, ctx)
                with torch.no_grad():
                    ok = dt * out1.max().item() <= 1.0
                if ok or dt < 1e-12:
                    break
                dt *= 0.5
            r2 = r1 + dt * div_T(J1)
            rho = 0.5 * (rho + r2)
            t += dt; nsub += 1
            if nsub >= MAXSUB:
                break
        return rho, nsub
