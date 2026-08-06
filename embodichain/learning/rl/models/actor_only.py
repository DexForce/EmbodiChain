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

__all__ = ["ActorOnly"]


class ActorOnly(Policy):
    """Actor-only policy for algorithms that do not use a value function (e.g., GRPO).

    Same interface as ActorCritic: get_action and evaluate_actions return (action, log_prob, value),
    but value is always zeros since no critic is used.

    When ``squash_actions`` is enabled, Gaussian samples are transformed with
    tanh into ``[-1, 1]`` and log probabilities include the transform's
    Jacobian correction.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        actor: nn.Module,
        squash_actions: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.actor = actor
        self.squash_actions = squash_actions
        self.actor.to(self.device)

        self.log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.device))
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand(mean.shape[0], -1)
        return Normal(mean, std)

    def _action_log_prob(
        self,
        distribution: Normal,
        action: torch.Tensor,
        pre_squash_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute action log probability with the tanh Jacobian correction."""
        if not self.squash_actions:
            return distribution.log_prob(action).sum(dim=-1)
        epsilon = torch.finfo(action.dtype).eps
        bounded_action = action.clamp(-1.0 + epsilon, 1.0 - epsilon)
        if pre_squash_action is None:
            pre_squash_action = torch.atanh(bounded_action)
        log_det_jacobian = torch.log(1.0 - bounded_action.square() + epsilon)
        return (distribution.log_prob(pre_squash_action) - log_det_jacobian).sum(dim=-1)

    def forward(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        return self._sample_action(
            tensordict,
            deterministic=deterministic,
            reparameterized=False,
        )

    def get_differentiable_action(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Sample an action with pathwise gradients."""
        return self._sample_action(
            tensordict,
            deterministic=deterministic,
            reparameterized=True,
        )

    def _sample_action(
        self,
        tensordict: TensorDict,
        *,
        deterministic: bool,
        reparameterized: bool,
    ) -> TensorDict:
        obs = tensordict["obs"]
        dist = self._distribution(obs)
        mean = dist.mean
        if deterministic:
            pre_squash_action = mean
        elif reparameterized:
            pre_squash_action = dist.rsample()
        else:
            pre_squash_action = dist.sample()
        action = (
            torch.tanh(pre_squash_action) if self.squash_actions else pre_squash_action
        )
        tensordict["action"] = action
        tensordict["sample_log_prob"] = self._action_log_prob(
            dist, action, pre_squash_action
        )
        if reparameterized:
            tensordict["entropy"] = dist.entropy().sum(dim=-1)
        tensordict["value"] = torch.zeros(
            obs.shape[0], device=self.device, dtype=obs.dtype
        )
        return tensordict

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict["obs"]
        tensordict["value"] = torch.zeros(
            obs.shape[0], device=self.device, dtype=obs.dtype
        )
        return tensordict

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict["obs"]
        action = tensordict["action"]
        dist = self._distribution(obs)
        return TensorDict(
            {
                "sample_log_prob": self._action_log_prob(dist, action),
                "entropy": dist.entropy().sum(dim=-1),
                "value": torch.zeros(obs.shape[0], device=self.device, dtype=obs.dtype),
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )
