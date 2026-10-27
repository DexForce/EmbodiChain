# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Author-equivalent EMA as a recoverable shared-trainer state."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from torch import nn

__all__ = ["RigidFormerEMA"]


class RigidFormerEMA:
    """Maintain the author's ramped FP32 EMA, including model buffers.

    Args:
        model: Model whose parameter and buffer layout remains fixed.
        decay: Maximum decay; initial updates use the author's warmup.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0,1)")
        self.model, self.decay, self.num_updates = model, decay, 0
        self.shadow = {
            k: (
                v.detach().float().clone()
                if v.is_floating_point()
                else v.detach().clone()
            )
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def on_update(self, learner: Any) -> None:
        """Update the average once after a successful parameter update."""
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        floating, values = [], []
        for name, value in self.model.state_dict().items():
            if value.is_floating_point():
                floating.append(self.shadow[name])
                values.append(value.detach().float())
            else:
                self.shadow[name].copy_(value)
        torch._foreach_mul_(floating, decay)
        torch._foreach_add_(floating, values, alpha=1 - decay)

    def state_dict(self) -> dict[str, Any]:
        """Return owned FP32 shadows and the exact EMA update cursor."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()},
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate EMA policy and every shadow tensor before restoration."""
        if (
            state.get("decay") != self.decay
            or type(state.get("num_updates")) is not int
            or state["num_updates"] < 0
            or state["shadow"].keys() != self.shadow.keys()
            or any(
                v.shape != state["shadow"][k].shape
                or v.dtype != state["shadow"][k].dtype
                for k, v in self.shadow.items()
            )
        ):
            raise ValueError("RigidFormer EMA state differs")

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a prevalidated EMA without changing live model weights."""
        self.num_updates = state["num_updates"]
        for key, value in self.shadow.items():
            value.copy_(state["shadow"][key])
