# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .mlp import MLP
from .policy import Policy


class ActorCritic(Policy):
    """Actor-Critic with learnable log_std for Gaussian policy.

    This is a placeholder implementation of the Policy interface that:
    - Encapsulates MLP networks (actor + critic) that need to be trained by RL algorithms
    - Handles internal computation: MLP output → mean + learnable log_std → Normal distribution
    - Provides a uniform interface for RL algorithms (PPO, SAC, etc.)

    This allows seamless swapping with other policy implementations (e.g., VLAPolicy)
    without modifying RL algorithm code.

    Implements:
      - get_action(obs, deterministic=False) -> (action, log_prob, value)
      - get_value(obs)
      - evaluate_actions(obs, actions) -> (log_prob, entropy, value)
    """

    def __init__(
        self,
        obs_space,
        action_space,
        device: torch.device,
        actor: nn.Module,
        critic: nn.Module,
    ):
        super().__init__()
        self.obs_dim = obs_space.shape[-1]
        self.action_dim = action_space.shape[-1]
        self.device = device

        # Require external injection of actor and critic
        self.actor = actor
        self.critic = critic
        self.actor.to(self.device)
        self.critic.to(self.device)

        # learnable log_std per action dim
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.device))
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    @torch.no_grad()
    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand(mean.shape[0], -1)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand(mean.shape[0], -1)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value
