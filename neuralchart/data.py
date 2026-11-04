"""The data a Neural Chart takes: Conditions and Trajectories."""
import numpy as np
import torch


class Conditions(dict):
    """the conditions of B trajectories, tensors with a leading batch dimension.  A Neural Chart reads
      T          (B,)            temperature (default 1)
      U          (B, V) or (B, M) potential energy, on every species or per species and site
      force      (B, H)          line integral of an external force along every hyperedge (energy)
      chem_logc  (B, n_chem)     log concentrations of the chemostats
      edge       (B, H, F)       local descriptors of the hyperedges
    Any other key is kept and ignored by the chart (the inputs of other models on the same data)."""

    def __init__(s, *a, **kw):
        super().__init__(*a, **{k: v for k, v in kw.items() if v is not None})

    @property
    def batch(s):
        return next(v.shape[0] for v in s.values() if torch.is_tensor(v) and v.dim() >= 1)

    def take(s, idx):
        return Conditions({k: v[idx] if torch.is_tensor(v) and v.dim() >= 1 else v for k, v in s.items()})

    def to(s, *a, **kw):
        return Conditions({k: v.to(*a, **kw) if torch.is_tensor(v) and v.is_floating_point() else
                           v.to(a[0] if a and not isinstance(a[0], torch.dtype) else kw.get('device', v.device)) if torch.is_tensor(v) else v
                           for k, v in s.items()})


class Trajectories:
    """states c (B, T + 1, M) at the sampling times `times` (T + 1,) or with the sampling step `dt`, under `conditions`.  The sampling
    must be uniform: rollouts advance by one sampling step."""

    def __init__(s, c, conditions=None, times=None, dt=None):
        s.c = c
        s.conditions = Conditions(conditions if conditions is not None else {})
        if times is not None:
            t = np.asarray(torch.as_tensor(times).cpu(), dtype=np.float64)
            d = np.diff(t)
            assert len(d) and np.allclose(d, d[0], rtol=1e-6, atol=0.0), 'the sampling times must be uniform'
            dt = float(d[0])
        assert dt is not None, 'give the sampling times or the sampling step'
        s.dt = float(dt)

    @property
    def window(s):
        return s.c.shape[1] - 1

    def as_dict(s):
        return dict(c=s.c, drive=s.conditions)
