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

"""Contracts for differentiable vector environments."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

import torch
from gymnasium.spaces import Space
from tensordict import TensorDict

__all__ = ["DifferentiableObservation", "DifferentiableVecEnv"]

DifferentiableObservation: TypeAlias = torch.Tensor | TensorDict


@runtime_checkable
class DifferentiableVecEnv(Protocol):
    """Structural interface for batched differentiable environments.

    Implementations must preserve the autograd path from ``action`` through
    both the returned reward and the differentiable portion of the next
    observation. Unlike a regular Gym environment, callers may retain this
    graph across multiple calls to :meth:`step`.

    ``detach_state`` defines an explicit truncated-backpropagation boundary.
    It must detach differentiable internal state and return the corresponding
    detached observation without resetting episode state, randomizing the
    environment, or changing tensor values.

    Implementations must auto-reset each environment that terminates or
    truncates. For those environments, :meth:`step` returns the terminal reward
    and done flags together with the initial observation of the next episode.
    """

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
        """Advance one step and auto-reset environments marked done."""
        ...

    def detach_state(self) -> DifferentiableObservation:
        """Detach internal state and return its current detached observation."""
        ...
