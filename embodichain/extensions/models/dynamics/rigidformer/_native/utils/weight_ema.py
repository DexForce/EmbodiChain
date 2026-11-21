# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""Exponential moving average (EMA) of a model's weights.

The average covers every floating-point entry of the module's state dict (parameters and
floating buffers such as BatchNorm running statistics); integer buffers (e.g. BatchNorm's
num_batches_tracked) are copied. The shadow copy is kept in fp32. The effective decay ramps as
min(decay, (1 + n) / (10 + n)) over the first updates so the average is not biased toward the
initial weights.
"""

from __future__ import annotations
from typing import Dict, Optional

import torch


class WeightEMA:
    def __init__(self, module: torch.nn.Module, decay: float):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1); got {decay}")
        self.module = module
        self.decay = float(decay)
        self.num_updates = 0
        self._keys = list(module.state_dict(keep_vars=True).keys())
        self.shadow: Dict[str, torch.Tensor] = {}
        self.reset()

    def _state(self) -> Dict[str, torch.Tensor]:
        sd = self.module.state_dict(keep_vars=True)
        if list(sd.keys()) != self._keys:
            raise RuntimeError(
                "WeightEMA: the module's state-dict keys changed after construction"
            )
        return sd

    @torch.no_grad()
    def reset(self) -> None:
        """Restart the average from the module's current weights (e.g. after loading a checkpoint)."""
        self.shadow = {
            k: (
                v.detach().float().clone()
                if v.is_floating_point()
                else v.detach().clone()
            )
            for k, v in self._state().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self) -> None:
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        sh, src = [], []
        for k, v in self._state().items():
            if v.is_floating_point():
                sh.append(self.shadow[k])
                src.append(v.detach().float())
            else:
                self.shadow[k].copy_(v.detach())
        torch._foreach_mul_(sh, d)
        torch._foreach_add_(sh, src, alpha=1.0 - d)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Averaged weights in each entry's own dtype (loadable with module.load_state_dict)."""
        return {
            k: (
                self.shadow[k].to(v.dtype)
                if v.is_floating_point()
                else self.shadow[k].clone()
            )
            for k, v in self._state().items()
        }

    @torch.no_grad()
    def load_state_dict(
        self, sd: Dict[str, torch.Tensor], num_updates: Optional[int] = None
    ) -> None:
        missing = [k for k in self._keys if k not in sd]
        if missing:
            raise KeyError(
                f"WeightEMA.load_state_dict: {len(missing)} keys missing, e.g. {missing[:3]}"
            )
        for k in self._keys:
            self.shadow[k].copy_(
                sd[k].to(device=self.shadow[k].device, dtype=self.shadow[k].dtype)
            )
        if num_updates is not None:
            self.num_updates = int(num_updates)
