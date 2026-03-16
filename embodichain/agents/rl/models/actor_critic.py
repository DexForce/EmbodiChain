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
import torch.nn as nn
from torch.distributions.normal import Normal
from tensordict import TensorDict

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

    Implements TensorDict-native interfaces while preserving `get_action()`
    compatibility for evaluation and legacy call-sites.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        actor: nn.Module,
        critic: nn.Module,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
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

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand(mean.shape[0], -1)
        return Normal(mean, std)

    def forward(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        obs = tensordict["obs"]
        dist = self._distribution(obs)
        mean = dist.mean
        action = mean if deterministic else dist.sample()
        tensordict["action"] = action
        tensordict["sample_log_prob"] = dist.log_prob(action).sum(dim=-1)
        tensordict["value"] = self.critic(obs).squeeze(-1)
        return tensordict

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        tensordict["value"] = self.critic(tensordict["obs"]).squeeze(-1)
        return tensordict

    def evaluate_actions(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        obs = tensordict["obs"]
        action = tensordict["action"]
        dist = self._distribution(obs)
        return TensorDict(
            {
                "sample_log_prob": dist.log_prob(action).sum(dim=-1),
                "entropy": dist.entropy().sum(dim=-1),
                "value": self.critic(obs).squeeze(-1),
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )
