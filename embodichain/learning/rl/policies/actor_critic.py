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
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import MappingProxyType

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tensordict import TensorDict

from embodichain.learning.models import EmpiricalNormalizer
from .policy import Policy, _TrainingInputSpec
from .auxiliary import AuxiliaryTask

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
        actor_obs_normalization: Enable the default empirical normalizer and
            update actor input statistics at rollout boundaries.
        critic_obs_normalization: Matching critic normalization/update setting.
        initial_action_std: Initial standard deviation for every action.
        action_std_range: Inclusive clamp range applied to standard deviation.
        auxiliary_tasks: Named auxiliary objectives. The actor must implement
            ``forward_with_features(obs) -> (mean, features)`` when nonempty.
        actor_normalizer: Optional actor input transform. The normalization flag
            controls its running-statistics updates.
        critic_normalizer: Optional critic input transform with the same contract.
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
        *,
        auxiliary_tasks: Mapping[str, AuxiliaryTask] | None = None,
        actor_normalizer: nn.Module | None = None,
        critic_normalizer: nn.Module | None = None,
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
            actor_normalizer
            if actor_normalizer is not None
            else (
                EmpiricalNormalizer(self.actor_obs_dim)
                if self.actor_obs_normalization
                else nn.Identity()
            )
        )
        self.critic_obs_normalizer: nn.Module = (
            critic_normalizer
            if critic_normalizer is not None
            else (
                EmpiricalNormalizer(self.critic_obs_dim)
                if self.critic_obs_normalization
                else nn.Identity()
            )
        )
        self.actor_obs_normalizer.to(self.device)
        self.critic_obs_normalizer.to(self.device)
        self._custom_normalization = (
            actor_normalizer is not None or critic_normalizer is not None
        )
        for enabled, normalizer in (
            (self.actor_obs_normalization, self.actor_obs_normalizer),
            (self.critic_obs_normalization, self.critic_obs_normalizer),
        ):
            if enabled and not callable(getattr(normalizer, "update", None)):
                raise ValueError(
                    "An updating observation normalizer must provide update()."
                )

        initial_std = torch.full(
            (self.action_dim,),
            float(initial_action_std),
            device=self.device,
        )
        self.log_std = nn.Parameter(initial_std.log())
        self._auxiliary_tasks = dict(auxiliary_tasks or {})
        self._auxiliary_buffer_names: dict[str, tuple[str, ...]] = {}
        self.training_observation_groups: dict[str, tuple[str, ...]] = {}
        self.training_field_specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        if self._auxiliary_tasks and not callable(
            getattr(actor, "forward_with_features", None)
        ):
            raise ValueError("Auxiliary tasks require actor.forward_with_features().")
        for name, task in self._auxiliary_tasks.items():
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or name.startswith("_")
            ):
                raise ValueError("Auxiliary task names must be public identifiers.")
            task.validate_observation_groups(self.actor_obs_groups)
            if (
                task.training_observation_groups.keys()
                != task.training_field_specs.keys()
            ):
                raise ValueError(
                    f"Auxiliary task {name!r} has inconsistent training fields."
                )
            for key, module in task.modules.items():
                self._validate_auxiliary_name(name, key)
                self.add_module(f"{name}_{key}", module.to(self.device))
            buffers = task.initial_buffers()
            self._auxiliary_buffer_names[name] = tuple(buffers)
            for key, value in buffers.items():
                self._validate_auxiliary_name(name, key)
                self.register_buffer(
                    f"{name}_{key}", None if value is None else value.to(self.device)
                )
            for key, spec in task.training_field_specs.items():
                field = f"{name}_{key}"
                if (
                    not isinstance(key, str)
                    or not key.isidentifier()
                    or key.startswith("_")
                    or field in self.training_field_specs
                ):
                    raise ValueError(
                        f"Auxiliary training field {field!r} is invalid or duplicated."
                    )
                self.training_observation_groups[field] = tuple(
                    task.training_observation_groups[key]
                )
                self.training_field_specs[field] = spec

    def _validate_auxiliary_name(self, name: str, key: str) -> None:
        if (
            not isinstance(key, str)
            or not key.isidentifier()
            or key.startswith("_")
            or hasattr(self, f"{name}_{key}")
        ):
            raise ValueError(
                f"Auxiliary component {name}_{key!s} conflicts with policy state."
            )

    @property
    def auxiliary_tasks(self) -> Mapping[str, AuxiliaryTask]:
        """Return the read-only objective registry; neural state lives on this policy."""
        return MappingProxyType(self._auxiliary_tasks)

    @property
    def supports_auxiliary_loss(self) -> bool:
        """Whether configured tasks or a custom override supply an objective."""
        return (
            bool(self._auxiliary_tasks)
            or getattr(self.auxiliary_loss, "__func__", None)
            is not ActorCritic.auxiliary_loss
        )

    def _training_input_spec(self) -> _TrainingInputSpec:
        normalization_fields = None
        if (
            type(self) is ActorCritic
            and getattr(self.update_normalization, "__func__", None)
            is ActorCritic.update_normalization
            and getattr(self._critic_observation, "__func__", None)
            is ActorCritic._critic_observation
        ):
            requested = ("obs",) if self.actor_obs_normalization else ()
            if self.critic_obs_normalization:
                requested += ("critic_obs" if self.uses_separate_critic_obs else "obs",)
            normalization_fields = tuple(dict.fromkeys(requested))
        if (
            type(self) is not ActorCritic
            or getattr(self.evaluate_actions, "__func__", None)
            is not ActorCritic.evaluate_actions
        ):
            return _TrainingInputSpec(normalization_fields=normalization_fields)
        if (
            getattr(self.auxiliary_loss, "__func__", None)
            is not ActorCritic.auxiliary_loss
        ):
            return _TrainingInputSpec(
                mutates_inputs=False, normalization_fields=normalization_fields
            )
        fields = ("obs", "action")
        if self.uses_separate_critic_obs:
            fields += ("critic_obs",)
        return _TrainingInputSpec(
            fields=(*fields, *self.training_field_specs),
            mutates_inputs=False,
            normalization_fields=normalization_fields,
        )

    def _actor_outputs(
        self, obs: torch.Tensor, *, return_features: bool = False
    ) -> tuple[Normal, Mapping[str, torch.Tensor]]:
        normalized = self.actor_obs_normalizer(obs)
        if return_features:
            mean, features = self.actor.forward_with_features(normalized)
        else:
            mean, features = self.actor(normalized), {}
        std = self.action_std.expand(mean.shape[0], -1)
        return Normal(mean, std, validate_args=False), features

    def _distribution(self, obs: torch.Tensor) -> Normal:
        return self._actor_outputs(obs)[0]

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
        """Return logical actor/critic groups, retaining any shared parameters."""
        actor_parameters = tuple(
            dict.fromkeys(
                (
                    *self.actor.parameters(),
                    *self.actor_obs_normalizer.parameters(),
                    self.log_std,
                )
            )
        )
        seen = {id(parameter) for parameter in actor_parameters}
        critic_parameters = tuple(
            dict.fromkeys(
                (
                    *self.critic.parameters(),
                    *self.critic_obs_normalizer.parameters(),
                )
            )
        )
        seen.update(id(parameter) for parameter in critic_parameters)
        extra = []
        for task in self._auxiliary_tasks.values():
            for module in task.modules.values():
                for parameter in module.parameters():
                    if id(parameter) not in seen:
                        extra.append(parameter)
                        seen.add(id(parameter))
        actor_parameters = (*actor_parameters, *extra)
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

    def _collect_differentiable_action(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Collect pathwise actions without unused value or likelihood graphs."""
        if (
            getattr(self.get_differentiable_action, "__func__", None)
            is not ActorCritic.get_differentiable_action
            or getattr(self._sample_action, "__func__", None)
            is not ActorCritic._sample_action
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
        actor_obs = tensordict["obs"]
        critic_obs = None if action_only else self._critic_observation(tensordict)
        dist = self._distribution(actor_obs)
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
            tensordict["action_mean"] = dist.mean
            tensordict["action_std"] = dist.stddev
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
        if self._auxiliary_tasks:
            dist, features = self._actor_outputs(actor_obs, return_features=True)
        else:
            dist, features = self._distribution(actor_obs), {}
        fields = {
            "sample_log_prob": dist.log_prob(action).sum(dim=-1),
            "entropy": dist.entropy().sum(dim=-1),
            "value": self._value(critic_obs),
            "action_mean": dist.mean,
            "action_std": dist.stddev,
        }
        for name, task in self._auxiliary_tasks.items():
            outputs = task.evaluate(features)
            if outputs.keys() != set(task.output_names):
                raise ValueError(
                    f"Auxiliary task {name!r} returned undeclared outputs."
                )
            for key, value in outputs.items():
                field = f"{name}_{key}"
                if field in fields:
                    raise ValueError(
                        f"Auxiliary output {field!r} conflicts with policy outputs."
                    )
                fields[field] = value
        return TensorDict(
            fields,
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )

    def auxiliary_loss(self, batch: TensorDict, evaluated: TensorDict) -> torch.Tensor:
        """Sum configured auxiliary objectives using this transition's labels.

        Args:
            batch: State-aligned transition inputs and declared training fields.
            evaluated: Graph-connected outputs from :meth:`evaluate_actions`.

        Returns:
            Scalar loss; PPO owns the overall auxiliary coefficient.

        Raises:
            ValueError: No auxiliary objective is configured.
        """
        total = None
        for name, task in self._auxiliary_tasks.items():
            loss = task.loss(
                {key: batch[f"{name}_{key}"] for key in task.training_field_specs},
                {key: evaluated[f"{name}_{key}"] for key in task.output_names},
                {
                    key: getattr(self, f"{name}_{key}")
                    for key in self._auxiliary_buffer_names[name]
                },
            )
            total = loss if total is None else total + loss
        if total is None:
            raise ValueError("Auxiliary supervision requires configured tasks.")
        return total

    def export_actor(self) -> nn.Module:
        """Copy the deterministic actor and its input transform for inference.

        Returns:
            Independent evaluation-mode module without critic or auxiliary heads.
            The caller owns observation history and action postprocessing.
        """
        return nn.Sequential(
            deepcopy(self.actor_obs_normalizer), deepcopy(self.actor)
        ).eval()

    def checkpoint_contract(self) -> dict[str, object]:
        """Describe configured component semantics beyond the neural state layout.

        Returns:
            Serializable component contracts. Ordinary policies without composed
            components retain their original checkpoint metadata.
        """
        contract = {}
        for name, module in (("actor", self.actor), ("critic", self.critic)):
            describe = getattr(module, "checkpoint_contract", None)
            if callable(describe):
                declared = describe()
                if declared:
                    contract[name] = declared
        if self._auxiliary_tasks:
            contract["auxiliary_tasks"] = {
                name: task.checkpoint_contract()
                for name, task in self._auxiliary_tasks.items()
            }
        if contract or self._custom_normalization:
            contract["normalization_updates"] = {
                "actor": self.actor_obs_normalization,
                "critic": self.critic_obs_normalization,
            }
        return contract
