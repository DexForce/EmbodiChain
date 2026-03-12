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

from .policy import Policy


class ActorOnly(Policy):
    """Actor-only policy for algorithms that do not use a value function (e.g., GRPO).

    Same interface as ActorCritic: get_action and evaluate_actions return (action, log_prob, value),
    but value is always zeros since no critic is used.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        actor: nn.Module,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.actor = actor
        self.actor.to(self.device)

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
        obs = tensordict["observation"]
        dist = self._distribution(obs)
        mean = dist.mean
        action = mean if deterministic else dist.sample()
        tensordict["action"] = action
        tensordict["sample_log_prob"] = dist.log_prob(action).sum(dim=-1)
        tensordict["value"] = torch.zeros(
            obs.shape[0], device=self.device, dtype=obs.dtype
        )
        return tensordict

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict["observation"]
        tensordict["value"] = torch.zeros(
            obs.shape[0], device=self.device, dtype=obs.dtype
        )
        return tensordict

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict["observation"]
        action = tensordict["action"]
        dist = self._distribution(obs)
        tensordict["sample_log_prob"] = dist.log_prob(action).sum(dim=-1)
        tensordict["entropy"] = dist.entropy().sum(dim=-1)
        tensordict["value"] = torch.zeros(
            obs.shape[0], device=self.device, dtype=obs.dtype
        )
        return tensordict
