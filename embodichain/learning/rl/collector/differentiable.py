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

"""Graph-preserving rollout collection for differentiable environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from tensordict import TensorDict

from embodichain.learning.rl.env import (
    DifferentiableObservation,
    DifferentiableVecEnv,
)
from embodichain.learning.rl.models import Policy
from embodichain.learning.rl.utils import flatten_dict_observation

__all__ = [
    "DifferentiableCollector",
    "DifferentiableRollout",
    "DifferentiableTransition",
]


@dataclass(frozen=True)
class DifferentiableTransition:
    """One graph-preserving environment transition."""

    observation: torch.Tensor
    policy_output: TensorDict
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_observation: torch.Tensor
    info: dict[str, Any]

    @property
    def action(self) -> torch.Tensor:
        """Return the differentiable policy action."""
        return self.policy_output["action"]

    @property
    def done(self) -> torch.Tensor:
        """Return the combined termination mask."""
        return self.terminated | self.truncated


@dataclass(frozen=True)
class DifferentiableRollout:
    """An immutable sequence of graph-preserving transitions."""

    initial_observation: torch.Tensor
    transitions: tuple[DifferentiableTransition, ...]

    @property
    def num_steps(self) -> int:
        """Return the number of collected transitions."""
        return len(self.transitions)

    @property
    def final_observation(self) -> torch.Tensor:
        """Return the observation after the final transition."""
        if not self.transitions:
            return self.initial_observation
        return self.transitions[-1].next_observation

    @property
    def rewards(self) -> torch.Tensor:
        """Stack rewards as ``[time, num_envs]`` without detaching them."""
        if not self.transitions:
            return self.initial_observation.new_empty(
                (0, self.initial_observation.shape[0])
            )
        return torch.stack([transition.reward for transition in self.transitions])


class DifferentiableCollector:
    """Collect graph-preserving rollouts without a preallocated buffer."""

    def __init__(
        self,
        env: DifferentiableVecEnv,
        policy: Policy,
        device: torch.device,
    ) -> None:
        self.env = env
        self.policy = policy
        self.device = device
        self._observation: DifferentiableObservation | None = None

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[DifferentiableObservation, dict[str, Any]]:
        """Reset the environment and collector state."""
        self._observation, info = self.env.reset(seed=seed)
        return self._observation, info

    def collect(
        self,
        num_steps: int,
        *,
        deterministic: bool = False,
        on_step_callback: Callable[[DifferentiableTransition], None] | None = None,
    ) -> DifferentiableRollout:
        """Collect a graph-preserving rollout segment.

        Args:
            num_steps: Number of differentiable environment steps.
            deterministic: Whether to use deterministic policy actions.
            on_step_callback: Optional callback invoked with each transition.

        Returns:
            An immutable differentiable rollout.

        Raises:
            ValueError: If ``num_steps`` is not positive.
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if self._observation is None:
            self.reset()

        initial_observation = self._flatten_observation(self._observation)
        transitions: list[DifferentiableTransition] = []

        for _ in range(num_steps):
            observation = self._flatten_observation(self._observation)
            policy_input = TensorDict(
                {"obs": observation},
                batch_size=[self.env.num_envs],
                device=self.device,
            )
            policy_output = self.policy.get_differentiable_action(
                policy_input,
                deterministic=deterministic,
            )
            next_observation, reward, terminated, truncated, info = self.env.step(
                policy_output["action"]
            )
            transition = DifferentiableTransition(
                observation=observation,
                policy_output=policy_output,
                reward=reward.to(self.device),
                terminated=terminated.to(self.device),
                truncated=truncated.to(self.device),
                next_observation=self._flatten_observation(next_observation),
                info=info,
            )
            transitions.append(transition)
            self._observation = next_observation

            if on_step_callback is not None:
                on_step_callback(transition)

        return DifferentiableRollout(
            initial_observation=initial_observation,
            transitions=tuple(transitions),
        )

    def detach_state(self) -> torch.Tensor:
        """Start a new truncated-backpropagation segment."""
        self._observation = self.env.detach_state()
        return self._flatten_observation(self._observation)

    def _flatten_observation(
        self, observation: DifferentiableObservation | None
    ) -> torch.Tensor:
        if observation is None:
            raise RuntimeError("The differentiable collector has no observation.")
        if isinstance(observation, torch.Tensor):
            return observation.to(self.device)
        return flatten_dict_observation(observation.to(self.device))
