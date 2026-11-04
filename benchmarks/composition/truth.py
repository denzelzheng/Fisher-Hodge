"""The reference systems of the composition benchmarks: mass action in the activity, solved with DOP853."""
import math

import numpy as np
import torch

from neuralchart import Hypergraph

K1, K2, K2R, D = 2.0, 1.0, 0.02, 1.0               # the phosphorylation cycle
K1B, K2B, K2RB = 1.0, 0.5, 0.01                    # the second cycle; the same K2 / K2R, so that one chemostat drive fits both
CHI, K_I = 0.5, 0.5
CHEM = ('ATP', 'ADP')
CHEM_MU0 = [0.5 * math.log(K2 / K2R), -0.5 * math.log(K2 / K2R)]


def reaction_hg():
    return Hypergraph.from_strings(['X + E + ATP <-> Xp + E + ADP', 'Xp <-> X'], chemostats=CHEM, species=['X', 'Xp', 'E'])


def second_hg():
    return Hypergraph.from_strings(['Y + K + ATP <-> Yp + K + ADP', 'Yp <-> Y'], chemostats=CHEM, species=['Y', 'Yp', 'K'])


def transport_hg(n):
    return Hypergraph.from_strings([], species=['X', 'Xp'], diffusing=['X', 'Xp'], grid_n=n)


class Truth:
    """log a = log kf_c(h) + <nu+, y> + chemostats + dU / 2, log b = log kb_c(h) + <nu-, y> + chemostats - dU / 2, y = log c + excess.
    The potential acts on `moving`, the species that diffuse.  chi: G = chi c_X c_Y.  inhibit: k of class 0 divided by 1 + c_Yp / K_I."""

    def __init__(s, hg, kf, kb, moving=(), chi=0.0, inhibit=False):
        s.hg, s.chi, s.inhibit = hg, chi, inhibit
        typ = torch.as_tensor(np.asarray(hg.meta['typ']))
        s.typ = typ
        s.lkf, s.lkb = torch.log(torch.tensor(kf, dtype=torch.float64))[typ], torch.log(torch.tensor(kb, dtype=torch.float64))[typ]
        s.names = list(hg.species_names)
        s.moving = [s.names.index(n) for n in moving]
        if 'chem_Np' in hg.meta:
            s.Yp, s.Ym = torch.as_tensor(hg.meta['chem_Np'].toarray()), torch.as_tensor(hg.meta['chem_Nm'].toarray())

    def log_constants(s, cond):
        B = cond['dG'].shape[0] if 'dG' in cond else cond['U'].shape[0]
        la, lb = s.lkf[None].expand(B, -1).clone(), s.lkb[None].expand(B, -1).clone()
        if s.moving and 'U' in cond:
            U = cond['U'].double().cpu()
            Uf = torch.zeros(B, s.hg.M, dtype=torch.float64)
            for i in s.moving:
                Uf[:, i * s.hg.V:(i + 1) * s.hg.V] = U
            dU = s.hg.dot(Uf, '+') - s.hg.dot(Uf, '-')
            la, lb = la + 0.5 * dU, lb - 0.5 * dU
        if 'chem_Np' in s.hg.meta:
            lc = cond['chem_logc'].double().cpu().repeat_interleave(s.hg.V, 1)
            la, lb = la + lc @ s.Yp.T, lb + lc @ s.Ym.T
        return la, lb

    def rates(s, cond, c):
        la, lb = s.log_constants(cond)
        c = c.double().cpu()
        y = torch.log(c)
        if s.chi:
            ix, iy = s.names.index('X'), s.names.index('Y')
            ex = torch.zeros_like(c)
            ex[:, ix] = s.chi * c[:, iy]
            ex[:, iy] = s.chi * c[:, ix]
            y = y + ex
        pa, pb = s.hg.dot(y, '+'), s.hg.dot(y, '-')
        if s.inhibit:
            dk = -torch.log1p(c[:, s.names.index('Yp')] / K_I)
            la = torch.where(s.typ == 0, la + dk[:, None], la)
            lb = torch.where(s.typ == 0, lb + dk[:, None], lb)
        return torch.exp(la + pa), torch.exp(lb + pb)

    def solve(s, cond, c0, nsteps, dt):
        """one trajectory (nsteps + 1, M); cond with batch size 1"""
        from scipy.integrate import solve_ivp
        S = torch.as_tensor(s.hg.S)

        def rhs(t, x):
            a, b = s.rates(cond, torch.as_tensor(x)[None])
            return (S @ (a - b)[0]).numpy()
        ts = np.arange(nsteps + 1) * dt
        sol = solve_ivp(rhs, (0, ts[-1]), c0, method='DOP853', t_eval=ts, rtol=1e-10, atol=1e-12)
        assert sol.success and sol.y.min() > 0, sol.message
        return sol.y.T
