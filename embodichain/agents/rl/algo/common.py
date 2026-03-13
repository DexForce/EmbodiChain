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

from __future__ import annotations

import torch
from tensordict import TensorDict

__all__ = ["compute_gae"]


def compute_gae(
    rollout: TensorDict, gamma: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over a rollout stored as `[num_envs, time + 1]`.

    Args:
        rollout: Rollout TensorDict where `value[:, -1]` stores the bootstrap
            value for the final observation and transition-only fields reserve
            their last slot as padding.
        gamma: Discount factor.
        gae_lambda: GAE lambda coefficient.

    Returns:
        Tuple of `(advantages, returns)`, both shaped `[num_envs, time]`.
    """
    rewards = rollout["reward"][:, :-1].float()
    dones = rollout["done"][:, :-1].bool()
    values = rollout["value"].float()

    if rewards.ndim != 2:
        raise ValueError(
            f"Expected reward tensor with shape [num_envs, time], got {rewards.shape}."
        )

    num_envs, time_dim = rewards.shape
    if values.shape != (num_envs, time_dim + 1):
        raise ValueError(
            "Expected value tensor with shape [num_envs, time + 1], got "
            f"{values.shape} for rewards shape {rewards.shape}."
        )
    advantages = torch.zeros_like(rollout["reward"].float())
    last_advantage = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(time_dim)):
        not_done = (~dones[:, t]).float()
        delta = rewards[:, t] + gamma * values[:, t + 1] * not_done - values[:, t]
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[:, t] = last_advantage

    returns = torch.zeros_like(advantages)
    returns[:, :-1] = advantages[:, :-1] + values[:, :-1]
    rollout["advantage"] = advantages
    rollout["return"] = returns
    return advantages[:, :-1], returns[:, :-1]
