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

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tensordict import TensorDict

from .policy import Policy, _TrainingInputSpec

__all__ = ["ActorOnly"]


def _validate_normal(
    distribution: Normal, action: torch.Tensor | None = None
) -> Normal:
    """Validate an unchecked Normal, optionally including an existing action."""
    if not Normal._validate_args:
        if action is None:
            del distribution._validate_args
        return distribution
    tensors = (distribution.loc, distribution.scale)
    if action is not None:
        tensors += (action,)
    if all(
        type(value) is torch.Tensor
        and value.device == distribution.loc.device
        and value.layout == torch.strided
        and value.is_floating_point()
        for value in tensors
    ) and (
        action is None
        or all(
            actual == 1 or expected == 1 or actual == expected
            for actual, expected in zip(
                reversed(action.shape), reversed(distribution.batch_shape)
            )
        )
    ):
        valid = (
            distribution.arg_constraints["loc"].check(distribution.loc).all()
            & distribution.arg_constraints["scale"].check(distribution.scale).all()
        )
        if action is not None:
            valid &= distribution.support.check(action).all()
        if bool(valid):
            if action is None:
                # Restore Normal's inherited default, including future changes,
                # so log_prob still validates actions generated after this check.
                del distribution._validate_args
            return distribution
    # Let Torch retain parameter-before-sample validation, broadcasting errors
    # and its original exception messages. Failed/unsupported inputs are cold.
    return Normal(distribution.loc, distribution.scale)


class ActorOnly(Policy):
    """Actor-only policy for algorithms that do not use a value function (e.g., GRPO).

    Returns TensorDict policy outputs with zero values in place of a critic.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        actor: nn.Module,
        actor_obs_groups: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.actor_obs_dim = obs_dim
        self.actor_obs_groups = (
            None if actor_obs_groups is None else tuple(actor_obs_groups)
        )
        self.action_dim = action_dim
        self.device = device

        self.actor = actor
        self.actor.to(self.device)

        self.log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.device))
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    def _training_input_spec(self) -> _TrainingInputSpec:
        if (
            type(self) is not ActorOnly
            or getattr(self.evaluate_actions, "__func__", None)
            is not ActorOnly.evaluate_actions
        ):
            return _TrainingInputSpec()
        return _TrainingInputSpec(fields=("obs", "action"), mutates_inputs=False)

    def _distribution(
        self, obs: torch.Tensor, *, validate_args: bool | None = None
    ) -> Normal:
        mean = self.actor(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand(mean.shape[0], -1)
        if (
            type(self) is ActorOnly
            and type(obs) is torch.Tensor
            and obs.device.type == "cuda"
            and type(mean) is torch.Tensor
            and type(std) is torch.Tensor
            and validate_args is None
        ):
            return _validate_normal(Normal(mean, std, validate_args=False))
        return Normal(mean, std, validate_args=validate_args)

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

    def _collect_differentiable_action(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Collect pathwise actions without unused likelihood or value outputs."""
        if (
            getattr(self.get_differentiable_action, "__func__", None)
            is not ActorOnly.get_differentiable_action
            or getattr(self._sample_action, "__func__", None)
            is not ActorOnly._sample_action
        ):
            return self.get_differentiable_action(
                tensordict, deterministic=deterministic
            )
        return self._sample_action(
            tensordict,
            deterministic=deterministic,
            reparameterized=True,
            action_only=True,
        )

    def _sample_action(
        self,
        tensordict: TensorDict,
        *,
        deterministic: bool,
        reparameterized: bool,
        action_only: bool = False,
    ) -> TensorDict:
        obs = tensordict["obs"]
        dist = self._distribution(obs)
        mean = dist.mean
        if deterministic:
            action = mean
        elif reparameterized:
            action = dist.rsample()
        else:
            action = dist.sample()
        tensordict["action"] = action
        if reparameterized:
            tensordict["entropy"] = dist.entropy().sum(dim=-1)
        if not action_only:
            tensordict["sample_log_prob"] = dist.log_prob(action).sum(dim=-1)
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
        if (
            type(self) is ActorOnly
            and type(obs) is torch.Tensor
            and obs.device.type == "cuda"
            and getattr(self._distribution, "__func__", None) is ActorOnly._distribution
        ):
            # Combine CUDA checks for loc, scale and action validity.
            dist = _validate_normal(
                self._distribution(obs, validate_args=False), action
            )
        else:
            dist = self._distribution(obs)
        return TensorDict(
            {
                "sample_log_prob": dist.log_prob(action).sum(dim=-1),
                "entropy": dist.entropy().sum(dim=-1),
                "value": torch.zeros(obs.shape[0], device=self.device, dtype=obs.dtype),
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )
