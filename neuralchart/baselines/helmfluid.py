"""HelmFluid (Xing et al., "HelmFluid: Learning Helmholtz Dynamics for Interpretable Fluid Prediction", ICML 2024)
with static condition channels.

The HelmFluid code is not part of this repository.  Clone https://github.com/thuml/HelmFluid (MIT licence; we used
commit d8003045425b89d4c2fdd84dced241a77495e50d) and set NEURALCHART_HELMFLUID to the clone.  The model needs CuPy.

The forward below follows models/HelmFluid_2D_corr_potential.py of that repository, which is distributed under the
MIT licence (Copyright (c) 2024 THUML @ Tsinghua University; see THIRD_PARTY.md for the full notice).  The only
change: the input x = [history (T_in frames), cond (n_cond fields)] along the last axis.  The velocity branch sees
prev = [hist[:-1], cond] and next = [hist[1:], cond]; the texture branch sees [hist, cond, grid].  With n_cond = 0
this is the original forward for dataset_name 'NS' (circular padding 16).
"""
import os
import sys


def load_helmfluid():
    """import the HelmFluid model module from $NEURALCHART_HELMFLUID; returns (model module, LpLoss, Adam)"""
    root = os.environ.get('NEURALCHART_HELMFLUID')
    if not root or not os.path.isdir(root):
        raise RuntimeError('set NEURALCHART_HELMFLUID to a clone of https://github.com/thuml/HelmFluid (see THIRD_PARTY.md)')
    if root not in sys.path:
        sys.path.insert(0, root)
    import cupy
    if not hasattr(cupy.cuda, 'compile_with_cache'):      # removed in CuPy 14; RawModule is its replacement
        cupy.cuda.compile_with_cache = lambda src, **kw: cupy.RawModule(code=src)
    import models.HelmFluid_2D_corr_potential as HM
    from utils.utilities3 import LpLoss
    from utils.adam import Adam
    HM.print = lambda *a, **k: None                       # the module prints debug output
    return HM, LpLoss, Adam


def conditioned_helmfluid(args, n_cond):
    """a HelmFluid_2D_corr_potential model whose input carries n_cond static condition fields after the history"""
    import torch
    import torch.nn.functional as F
    HM, _, _ = load_helmfluid()

    class CondHelmFluid(HM.Model):
        def __init__(self, args, n_cond=0):
            super().__init__(args)
            self.n_cond = n_cond

        def forward(self, x, freeze=False):
            B, H, W, C = x.shape
            nc = self.n_cond
            hist, cond = x[..., :C - nc], x[..., C - nc:]
            prev = torch.cat([hist[..., :-1], cond], -1)
            nxt = torch.cat([hist[..., 1:], cond], -1)
            assert self.dataset_name == 'NS'
            prev = F.pad(prev.permute(0, 3, 1, 2), [16, 16, 16, 16], mode='circular').permute(0, 2, 3, 1).contiguous()
            nxt = F.pad(nxt.permute(0, 3, 1, 2), [16, 16, 16, 16], mode='circular').permute(0, 2, 3, 1).contiguous()
            prev = self.feature_fc0(prev).permute(0, 3, 1, 2)
            nxt = self.feature_fc0(nxt).permute(0, 3, 1, 2)
            grid = self.get_grid(x.shape, x.device)
            x = self.texture_fc0(torch.cat((x, grid), dim=-1)).permute(0, 3, 1, 2)
            x1 = self.texture_inc(x)
            x2 = self.texture_down1(x1)
            x3 = self.texture_down2(x2)
            prev_x1 = self.feature_inc(prev); prev_x2 = self.feature_down1(prev_x1); prev_x3 = self.feature_down2(prev_x2)
            next_x1 = self.feature_inc(nxt); next_x2 = self.feature_down1(next_x1); next_x3 = self.feature_down2(next_x2)
            x3, helm, vel, _, _ = self.process3(prev_x3, next_x3, x3, None)
            x2, helm, vel, _, _ = self.process2(prev_x2, next_x2, x2, helm)
            x1, helm, vel, _, _ = self.process1(prev_x1, next_x1, x1, helm)
            x = self.up3(x3, x2)
            x = self.up4(x, x1)
            x = self.outc(x).permute(0, 2, 3, 1)
            x = self.fc2(F.gelu(self.fc1(x)))
            vel = vel.reshape([B, self.groups, 2, H, W])[:, 0]
            helm = helm.reshape([B, self.groups, 2, H + 32, W + 32])[:, 0, ..., 16:-16, 16:-16]
            return x, helm, vel

    return CondHelmFluid(args, n_cond)
