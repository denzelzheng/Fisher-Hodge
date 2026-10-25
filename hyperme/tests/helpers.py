"""Random test models (float64) on several hypergraphs."""
import numpy as np
import torch
from hyperme import Hypergraph, HyperME, GridOps, presets
from hyperme.model import NoDrive


def perturb(model, std=0.3, seed=0):
    """give the zero-initialised last layers random values so that G, A and k are non-trivial."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in (model.free_energy, model.kinetics, model.drive):
            mlp = getattr(m, 'mlp', None) or getattr(m, 'psi', None)
            if mlp is not None:
                for p in mlp.net[-1].parameters():
                    p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * std)
            for nm in ('gam', 'v', 'Wb', 'gain', 'chi'):
                if hasattr(m, nm) and isinstance(getattr(m, nm), torch.nn.Parameter):
                    p = getattr(m, nm); p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * std)
    return model


def smooth(B, n, g, amp=0.5):
    x = torch.randn(B, n, n, generator=g, dtype=torch.float64)
    for _ in range(3):
        x = 0.25 * (x.roll(1, -1) + x.roll(-1, -1) + x.roll(1, -2) + x.roll(-1, -2))
    return amp * x / x.std()


def gaussian_kernel(n, width=0.15):
    a = np.arange(n)
    d = np.minimum(np.abs(a[:, None] - a[None, :]), n - np.abs(a[:, None] - a[None, :])) / n
    D2 = d[:, None, :, None] ** 2 + d[None, :, None, :] ** 2
    W = np.exp(-D2.transpose(0, 2, 1, 3).reshape(n * n, n * n) / (2 * width ** 2))
    return W / W.sum(1).mean()


def case(name, B=4, seed=0, reversible=False):
    """returns (model, drive, c0) in float64"""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    with presets.default_dtype(torch.float64):
        if name == 'c1':
            m = presets.c1()
            U = smooth(B, 8, g).reshape(B, -1)
            drive = dict(U=U, dG=1 + 2 * torch.rand(B, generator=g, dtype=torch.float64))
            c0 = torch.rand(B, m.hg.M, generator=g, dtype=torch.float64) * 1.5 + 0.05
        elif name == 'grid_nonlocal':
            n = 6
            hg = Hypergraph.grid_transport(n)
            m = HyperME.build(hg, hidden=32, nonlocal_kernel=gaussian_kernel(n), k_scale=5.0, step=dict(dt_max=0.01))
            P = smooth(B, n, g)
            force = GridOps.flat_e(GridOps.d1T(P)) + 0.3 * torch.randn(B, hg.H, generator=g, dtype=torch.float64)
            drive = dict(U=smooth(B, n, g).reshape(B, -1), force=force)
            c0 = torch.rand(B, hg.M, generator=g, dtype=torch.float64) * 2 + 1e-3
        elif name == 'reactions':
            sp = ['A', 'B', 'C', 'D']
            # one cycle per site (r1 + r3 = r2) and two moieties per site (A atoms, B atoms)
            rx = [({'A': 1, 'B': 1}, {'C': 1}), ({'A': 2}, {'D': 1}), ({'C': 1, 'A': 1}, {'D': 1, 'B': 1})]
            hg = Hypergraph.from_reactions(rx, sp, n_sites=3)
            m = HyperME.build(hg, hidden=32, step=dict(dt_max=0.01))
            drive = dict(U=0.3 * torch.randn(B, hg.M, generator=g, dtype=torch.float64),
                         force=torch.randn(B, hg.H, generator=g, dtype=torch.float64))
            c0 = torch.rand(B, hg.M, generator=g, dtype=torch.float64) + 1e-3
        elif name == 'schlogl':
            # Schloegl model on a 5 x 5 torus: A + 2X <-> 3X, X <-> B with chemostats A, B; X diffuses.
            # No conserved moiety; the chemostats drive the local cycle through chemostat_force().
            hg = Hypergraph.reaction_transport(['X', 'A', 'B'], [({'A': 1, 'X': 2}, {'X': 3}), ({'X': 1}, {'B': 1})],
                                               grid_n=5, diffusing=['X'], chemostats=['A', 'B'])
            m = HyperME.build(hg, hidden=32, step=dict(dt_max=0.01))
            mu = torch.stack([1.0 + torch.rand(B, generator=g, dtype=torch.float64), torch.zeros(B, dtype=torch.float64)], 1)
            drive = dict(U=0.3 * smooth(B, 5, g).reshape(B, -1), force=hg.chemostat_force(mu))
            c0 = torch.rand(B, hg.M, generator=g, dtype=torch.float64) + 0.2
        elif name == 'tcme':
            n = 8
            m = presets.s2_tcme(n=n, hidden=32)
            drive = dict(U=smooth(B, n, g), dpsi=GridOps.d1T(smooth(B, n, g, 0.3)), psin=smooth(B, n, g))
            c0 = torch.rand(B, n * n, generator=g, dtype=torch.float64) * 2 + 1e-3
        else:
            raise ValueError(name)
        if reversible:
            m.drive = NoDrive()
        m = perturb(m.double(), seed=seed)
    return m, drive, c0


CASES = ['c1', 'schlogl', 'reactions', 'grid_nonlocal', 'tcme']
