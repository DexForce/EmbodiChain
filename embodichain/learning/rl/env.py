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
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

import torch
from gymnasium.spaces import Space
from tensordict import TensorDict

__all__ = [
    "DifferentiableObservation",
    "DifferentiableVecEnv",
    "LearningVecEnv",
    "build_learning_env",
    "get_registered_learning_env_names",
    "register_learning_env",
]

DifferentiableObservation: TypeAlias = torch.Tensor | TensorDict
LearningEnvFactory: TypeAlias = Callable[..., "LearningVecEnv"]

_LEARNING_ENV_REGISTRY: dict[str, LearningEnvFactory] = {}


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
        if key in _LEARNING_ENV_REGISTRY and not override:
            raise ValueError(f"Learning environment '{name}' is already registered.")
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
