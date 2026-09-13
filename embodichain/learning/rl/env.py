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

"""Contracts and registration helpers for lightweight learning environments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

import torch
from gymnasium.spaces import Space
from tensordict import TensorDict

__all__ = [
    "DifferentiableObservation",
    "DifferentiableRolloutSpec",
    "DifferentiableVecEnv",
    "LearningVecEnv",
    "build_learning_env",
    "get_registered_learning_env_names",
    "register_learning_env",
    "ScheduledDifferentiableVecEnv",
    "stratified_rollout_value",
]

DifferentiableObservation: TypeAlias = torch.Tensor | TensorDict
LearningEnvFactory: TypeAlias = Callable[..., "LearningVecEnv"]

_LEARNING_ENV_REGISTRY: dict[str, LearningEnvFactory] = {}


@dataclass(frozen=True)
class DifferentiableRolloutSpec:
    """Describe one independent complete rollout used for a gradient microbatch.

    Args:
        num_steps: Full rollout horizon. The trainer must not detach or truncate it.
        objective_scale: Per-environment or scalar multiplier applied to returns.
        metadata: Scalar labels recorded with training metrics.
    """

    num_steps: int
    objective_scale: float | torch.Tensor = 1.0
    metadata: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.num_steps, bool) or int(self.num_steps) != self.num_steps:
            raise TypeError("num_steps must be a positive integer.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")


def stratified_rollout_value(index: int, minimum: int, maximum: int) -> int:
    """Cycle uniformly through an integer range and rotate each cycle's order.

    Args:
        index: Zero-based rollout index.
        minimum: Inclusive minimum scheduled value.
        maximum: Inclusive maximum scheduled value.

    Returns:
        Scheduled integer for ``index``.

    Raises:
        ValueError: If the inclusive range is empty.
    """
    count = int(maximum) - int(minimum) + 1
    if count < 1:
        raise ValueError("minimum must be less than or equal to maximum.")
    cycle, position = divmod(int(index), count)
    return int(minimum) + ((position + cycle) % count)


@runtime_checkable
class LearningVecEnv(Protocol):
    """Structural interface shared by lightweight vector environments."""

    num_envs: int
    device: torch.device
    single_observation_space: Space
    single_action_space: Space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[DifferentiableObservation, dict[str, Any]]:
        """Reset all environments and return the initial observation."""
        ...

    def step(self, action: torch.Tensor) -> tuple[
        DifferentiableObservation,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Advance all environments by one step."""
        ...

    def close(self) -> None:
        """Release owned resources."""
        ...


@runtime_checkable
class DifferentiableVecEnv(LearningVecEnv, Protocol):
    """Batched env that preserves the autograd path through ``step``.

    ``detach_state`` is the truncated-backpropagation boundary: detach
    differentiable internal state and return the current observation without
    resetting or resampling the episode. Finished rows must auto-reset inside
    ``step``, returning the terminal reward/done with the next initial observation.
    """

    def detach_state(self) -> DifferentiableObservation:
        """Detach internal state and return its current detached observation."""
        ...


@runtime_checkable
class ScheduledDifferentiableVecEnv(DifferentiableVecEnv, Protocol):
    """Differentiable env that schedules variable independent rollouts.

    The trainer calls :meth:`prepare_differentiable_rollout` immediately before
    resetting the environment. Implementations may select task difficulty for
    the next reset, such as an ordered-waypoint count, and must return the full
    horizon and objective scaling for that selection.
    """

    def prepare_differentiable_rollout(
        self,
        rollout_index: int,
    ) -> DifferentiableRolloutSpec:
        """Configure the next reset and return its complete-rollout contract.

        Args:
            rollout_index: Zero-based independent-rollout index.

        Returns:
            Full horizon, objective scaling, and optional metric metadata.
        """
        ...


def _is_nested_package_shadow(existing: Any, candidate: Any) -> bool:
    """Return True when ``candidate`` is an editable-install nested duplicate.

    Legacy editable installs can expose both a canonical ``embodichain_tasks``
    module and a nested ``embodichain_tasks.embodichain_tasks`` duplicate.
    Prefer the shorter canonical module path already registered.
    """
    existing_module = getattr(existing, "__module__", "") or ""
    candidate_module = getattr(candidate, "__module__", "") or ""
    if not existing_module or not candidate_module:
        return False
    nested_prefix = existing_module.partition(".")[0] + "." + existing_module
    return candidate_module == nested_prefix or candidate_module.startswith(
        nested_prefix + "."
    )


def register_learning_env(
    name: str,
    factory: LearningEnvFactory | None = None,
    *,
    override: bool = False,
) -> Callable[[LearningEnvFactory], LearningEnvFactory] | LearningEnvFactory:
    """Register a lightweight vector-environment factory.

    The function supports both ``@register_learning_env("Name")`` and direct
    ``register_learning_env("Name", Factory)`` use.
    """

    def decorator(env_factory: LearningEnvFactory) -> LearningEnvFactory:
        key = name.lower()
        if key in _LEARNING_ENV_REGISTRY:
            existing = _LEARNING_ENV_REGISTRY[key]
            if _is_nested_package_shadow(existing, env_factory):
                return env_factory
            if not override:
                raise ValueError(
                    f"Learning environment '{name}' is already registered."
                )
        _LEARNING_ENV_REGISTRY[key] = env_factory
        return env_factory

    if factory is None:
        return decorator
    return decorator(factory)


def get_registered_learning_env_names() -> list[str]:
    """Return registered lightweight environment names."""
    return sorted(_LEARNING_ENV_REGISTRY)


def build_learning_env(
    name: str,
    *,
    num_envs: int,
    device: torch.device | str,
    **cfg: Any,
) -> LearningVecEnv:
    """Build a registered lightweight vector environment."""
    key = name.lower()
    if key not in _LEARNING_ENV_REGISTRY:
        available = ", ".join(get_registered_learning_env_names()) or "<none>"
        raise ValueError(
            f"Learning environment '{name}' is not registered. Available: {available}"
        )
    return _LEARNING_ENV_REGISTRY[key](
        num_envs=num_envs,
        device=torch.device(device),
        **cfg,
    )
