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

"""Bounded temporal aggregation of independently predicted action chunks."""

from __future__ import annotations

import math

import torch

__all__ = ["ACTTemporalEnsembler"]


class ACTTemporalEnsembler:
    """Average overlapping predictions for the current action, oldest first.

    The cache retains at most one horizon of chunks. Rows can be reset
    independently when vector environments finish at different times.

    Args:
        horizon: Prediction horizon.
        num_envs: Number of independent rollout rows.
        action_dim: Width of an action.
        device: Device of incoming action chunks.
        decay: Exponential weighting coefficient in generation order.
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        action_dim: int,
        device: str | torch.device = "cpu",
        decay: float = 0.01,
    ) -> None:
        if any(type(v) is not int or v < 1 for v in (horizon, num_envs, action_dim)):
            raise ValueError("ACT ensemble dimensions must be positive integers")
        if not math.isfinite(decay) or decay < 0:
            raise ValueError("ACT ensemble decay must be finite and nonnegative")
        self.horizon, self.num_envs, self.action_dim = horizon, num_envs, action_dim
        self.decay = decay
        self.chunks = torch.zeros(horizon, num_envs, horizon, action_dim, device=device)
        self.valid = torch.zeros(horizon, num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Forget predictions for all or selected rollout rows.

        Args:
            env_ids: Optional integer indices of rows to clear.
        """
        if env_ids is None:
            self.valid.zero_()
        else:
            self.valid[:, env_ids] = False

    def update(self, prediction: torch.Tensor) -> torch.Tensor:
        """Add a chunk and return an owned weighted action for each row.

        Args:
            prediction: Physical commands ``[num_envs, horizon, action_dim]``.

        Returns:
            Weighted action tensor ``[num_envs, action_dim]``.
        """
        if prediction.shape != (self.num_envs, self.horizon, self.action_dim):
            raise ValueError("ACT ensemble received an incompatible chunk")
        if prediction.device != self.chunks.device or not bool(
            torch.isfinite(prediction).all()
        ):
            raise ValueError("ACT ensemble requires finite chunks on its device")
        self.chunks[1:] = self.chunks[:-1].clone()
        self.valid[1:] = self.valid[:-1].clone()
        self.chunks[0].copy_(prediction)
        self.valid[0] = True
        age = torch.arange(self.horizon, device=prediction.device)
        values = self.chunks[age, :, age]
        if self.decay * (self.horizon - 1) < 80:
            weights = torch.exp(self.decay * age.float())[:, None] * self.valid
        else:
            oldest = torch.where(self.valid, age[:, None], 0).max(dim=0).values
            logits = (age[:, None] - oldest[None, :]).to(torch.float64) * self.decay
            weights = torch.exp(torch.where(self.valid, logits, -torch.inf)).to(
                values.dtype
            )
        return (values * weights[..., None]).sum(0) / weights.sum(0)[:, None]
