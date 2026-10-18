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

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tensordict import TensorDict

from .mlp import MLP
from .normalizer import EmpiricalNormalizer
from .policy import Policy

__all__ = ["ActorCritic"]


class ActorCritic(Policy):
    """TensorDict actor-critic with a learnable Gaussian action distribution.

    Args:
        obs_dim: Actor observation dimension.
        action_dim: Action dimension.
        device: Device used by the policy modules and distribution parameters.
        actor: Module that maps actor observations to action means.
        critic: Module that maps critic observations to scalar values.
        critic_obs_dim: Separate critic observation dimension. ``None`` reuses
            actor observations.
        actor_obs_groups: Ordered observation groups consumed by the actor.
        critic_obs_groups: Ordered observation groups consumed by the critic.
        actor_obs_normalization: Whether to normalize actor observations.
        critic_obs_normalization: Whether to normalize critic observations.
        initial_action_std: Initial standard deviation for every action.
        action_std_range: Inclusive clamp range applied to standard deviation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        actor: nn.Module,
        critic: nn.Module,
        critic_obs_dim: int | None = None,
        actor_obs_groups: Sequence[str] | None = None,
        critic_obs_groups: Sequence[str] | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        initial_action_std: float = 1.0,
        action_std_range: tuple[float, float] = (1e-6, 1e6),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.actor_obs_dim = obs_dim
        self.critic_obs_dim = obs_dim if critic_obs_dim is None else int(critic_obs_dim)
        self.uses_separate_critic_obs = critic_obs_dim is not None
        self.actor_obs_groups = (
            None if actor_obs_groups is None else tuple(actor_obs_groups)
        )
        self.critic_obs_groups = (
            self.actor_obs_groups
            if critic_obs_groups is None
            else tuple(critic_obs_groups)
        )
        self.action_dim = action_dim
        self.distribution_param_dim = action_dim
        self.device = device
        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        if initial_action_std <= 0.0:
            raise ValueError("initial_action_std must be positive.")
        if (
            len(action_std_range) != 2
            or action_std_range[0] <= 0.0
            or action_std_range[1] < action_std_range[0]
        ):
            raise ValueError(
                "action_std_range must contain positive increasing bounds."
            )
        self.action_std_range = (
            float(action_std_range[0]),
            float(action_std_range[1]),
        )

        self.actor = actor
        self.critic = critic
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_obs_normalizer: nn.Module = (
            EmpiricalNormalizer(self.actor_obs_dim).to(self.device)
            if self.actor_obs_normalization
            else nn.Identity()
        )
        self.critic_obs_normalizer: nn.Module = (
            EmpiricalNormalizer(self.critic_obs_dim).to(self.device)
            if self.critic_obs_normalization
            else nn.Identity()
        )

        initial_std = torch.full(
            (self.action_dim,),
            float(initial_action_std),
            device=self.device,
        )
        self.log_std = nn.Parameter(initial_std.log())

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(self.actor_obs_normalizer(obs))
        std = self.action_std.expand(mean.shape[0], -1)
        return Normal(mean, std, validate_args=False)

    @property
    def action_std(self) -> torch.Tensor:
        """Return the clamped state-independent action standard deviation."""
        lower, upper = self.action_std_range
        log_lower = math.log(lower)
        log_upper = math.log(upper)
        return self.log_std.clamp(log_lower, log_upper).exp()

    def optimization_parameter_groups(
        self,
    ) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
        """Return actor-side and critic-side trainable parameter groups."""
        actor_parameters = (*self.actor.parameters(), self.log_std)
        critic_parameters = tuple(self.critic.parameters())
        return actor_parameters, critic_parameters

    def _value(self, observation: torch.Tensor) -> torch.Tensor:
        """Evaluate the critic after applying its observation normalizer."""
        return self.critic(self.critic_obs_normalizer(observation)).squeeze(-1)

    @torch.no_grad()
    def update_normalization(self, tensordict: TensorDict) -> None:
        """Update actor and critic observation statistics from environment data."""
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(tensordict["obs"])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self._critic_observation(tensordict))

    def _critic_observation(self, tensordict: TensorDict) -> torch.Tensor:
        """Return the observation selected for value estimation."""
        if self.uses_separate_critic_obs:
            if "critic_obs" not in tensordict.keys():
                raise KeyError(
                    "ActorCritic requires 'critic_obs' because the critic uses "
                    f"{self.critic_obs_dim} inputs while the actor uses "
                    f"{self.actor_obs_dim}."
                )
            return tensordict["critic_obs"]
        return tensordict["obs"]

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
        actor_obs = tensordict["obs"]
        critic_obs = self._critic_observation(tensordict)
        dist = self._distribution(actor_obs)
        mean = dist.mean
        if deterministic:
            action = mean
        elif reparameterized:
            action = dist.rsample()
        else:
            action = dist.sample()
        tensordict["action"] = action
        tensordict["sample_log_prob"] = dist.log_prob(action).sum(dim=-1)
        tensordict["action_mean"] = dist.mean
        tensordict["action_std"] = dist.stddev
        if reparameterized:
            tensordict["entropy"] = dist.entropy().sum(dim=-1)
        tensordict["value"] = self._value(critic_obs)
        return tensordict

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        critic_obs = self._critic_observation(tensordict)
        tensordict["value"] = self._value(critic_obs)
        return tensordict

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        actor_obs = tensordict["obs"]
        critic_obs = self._critic_observation(tensordict)
        action = tensordict["action"]
        dist = self._distribution(actor_obs)
        return TensorDict(
            {
                "sample_log_prob": dist.log_prob(action).sum(dim=-1),
                "entropy": dist.entropy().sum(dim=-1),
                "value": self._value(critic_obs),
                "action_mean": dist.mean,
                "action_std": dist.stddev,
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )
