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

"""Lightweight NeRD learning environments sharing the production RL protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from embodichain_models.dynamics.nerd.codec import reset_history
from embodichain_models.dynamics.nerd.transition import NeRDState, NeRDTransition

from ._task_reset import sample_task_reset
from ._task_values import reference_task_values

__all__ = ["NeRDModelEnv"]


class NeRDModelEnv:
    """Own canonical state/history and task rules without a simulation host.

    Args:
        transition: Independent transition with an injected geometry backend.
        max_episode_steps: Positive control-step episode horizon.
        random_reset: Whether to sample reference initial-state perturbations.
        seed: Optional environment RNG seed, independent of policy sampling.
        reset_source: Optional workflow-owned selected-row initial-state sampler.
        physics_metadata: Validated reference physical identity for reporting.

    Finished rows auto-reset using new storage. Rewards and the complete
    ``terminal_observation`` batch keep the pre-reset graph. Other rows retain
    their histories and graph. ``detach_state`` is the explicit TBPTT boundary;
    it never resets counters, draws random values or changes observations.
    """

    def __init__(
        self,
        transition: NeRDTransition,
        *,
        max_episode_steps: int,
        random_reset: bool = True,
        seed: int | None = None,
        reset_source: (
            Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
        physics_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        if not isinstance(random_reset, bool):
            raise TypeError("random_reset must be boolean")
        self.transition = transition
        self.robot_name = transition.codec.robot
        self.num_envs = transition.geometry.gravity.shape[0]
        self.device = transition.geometry.gravity.device
        self.physics_dt = transition.dt
        self.max_episode_steps = max_episode_steps
        self.random_reset = random_reset
        self._reset_source = reset_source
        self._physics_metadata = deepcopy(dict(physics_metadata or {}))
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(torch.initial_seed() if seed is None else seed)
        self.single_action_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(1 if self.robot_name == "cartpole" else 8,),
            dtype=np.float32,
        )
        self.single_observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(4 if self.robot_name == "cartpole" else 27,),
            dtype=np.float32,
        )
        self._state = transition.initial_state(
            transition.geometry.gravity.new_zeros(
                (self.num_envs, transition.codec.q_dim)
            ),
            transition.geometry.gravity.new_zeros(
                (self.num_envs, transition.codec.qd_dim)
            ),
        )
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._closed = False
        self.reset()

    @property
    def dynamics_state(self) -> NeRDState:
        """Return the graph-connected canonical state; its tensors are read-only."""
        return self._state

    def dynamics_metadata(self) -> dict[str, Any]:
        """Return an owned copy of the validated reference physical identity."""
        return deepcopy(self._physics_metadata)

    def dynamics_parameters(self) -> Mapping[str, torch.Tensor]:
        """Return resident network parameters without copying or detaching."""
        return dict(self.transition.network.named_parameters())

    def get_obs(self) -> torch.Tensor:
        """Return observations from canonical state, preserving its graph."""
        return reference_task_values(self.robot_name, self._state.q, self._state.qd)[0]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset selected rows while retaining all unselected trajectories.

        Args:
            seed: Optional seed for the environment's private generator.
            options: Optional ``reset_ids`` sequence; omission resets all rows.

        Returns:
            A complete observation batch and reset metadata.
        """
        self._check_open()
        if seed is not None:
            self._generator.manual_seed(seed)
        selection = None if options is None else options.get("reset_ids")
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if selection is None
            else torch.as_tensor(
                selection, dtype=torch.long, device=self.device
            ).reshape(-1)
        )
        if ids.unique().numel() != ids.numel() or bool(
            ((ids < 0) | (ids >= self.num_envs)).any()
        ):
            raise ValueError("reset_ids must be unique valid environment rows")
        if not ids.numel():
            return self.get_obs(), {}
        q, qd = (
            self._reset_source(ids)
            if self._reset_source is not None
            else sample_task_reset(
                self.robot_name,
                ids.numel(),
                self.device,
                self._generator,
                self.random_reset,
            )
        )
        if (
            q.shape != (ids.numel(), self.transition.codec.q_dim)
            or qd.shape != (ids.numel(), self.transition.codec.qd_dim)
            or q.device != self.device
            or qd.device != self.device
            or q.dtype != self._state.q.dtype
            or qd.dtype != self._state.qd.dtype
        ):
            raise ValueError("Reset source must return matching canonical q/qd rows")
        if ids.numel() == self.num_envs:
            # Zero-valued index_copy/where derivatives still retain the old
            # episode's graph. A complete reset must not refer to it at all.
            self._state = self.transition.initial_state(
                torch.zeros_like(self._state.q).index_copy(0, ids, q.detach().clone()),
                torch.zeros_like(self._state.qd).index_copy(
                    0, ids, qd.detach().clone()
                ),
            )
            self._elapsed_steps = torch.zeros_like(self._elapsed_steps)
            return self.get_obs(), {}
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device).scatter(
            0, ids, True
        )
        history, lengths = reset_history(
            self._state.history, self._state.valid_lengths, mask
        )
        self._state = NeRDState(
            self._state.q.index_copy(0, ids, q.detach().clone()),
            self._state.qd.index_copy(0, ids, qd.detach().clone()),
            history,
            lengths,
        )
        self._elapsed_steps = torch.where(mask, 0, self._elapsed_steps)
        return self.get_obs(), {}

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Advance the model and return graph-safe terminal/reset observations.

        Args:
            action: Normalized controls, shape ``(num_envs, action_dim)``.

        Returns:
            Observation, reward, termination, truncation and information batch.
        """
        self._check_open()
        action = torch.as_tensor(action, dtype=self._state.q.dtype, device=self.device)
        if action.shape != (self.num_envs, *self.single_action_space.shape):
            raise ValueError("Action shape must match the batched action space")
        effort = action.clamp(-1.0, 1.0)
        if self.robot_name == "cartpole":
            effort = torch.cat((effort * 1500.0, torch.zeros_like(effort)), -1)
        else:
            effort = effort * 400.0
        self._state = self.transition.step(self._state, effort)
        self._elapsed_steps = self._elapsed_steps + 1
        obs, reward, terminated = reference_task_values(
            self.robot_name, self._state.q, self._state.qd
        )
        truncated = self._elapsed_steps >= self.max_episode_steps
        metrics = (
            {"pole_angle": obs[:, 1].abs().detach()}
            if self.robot_name == "cartpole"
            else {
                "height": self._state.q[:, 2].detach(),
                "forward_velocity": self._state.qd[:, 1].detach(),
            }
        )
        info: dict[str, Any] = {"fail": terminated, "metrics": metrics}
        done = terminated | truncated
        if bool(done.any()):
            info["terminal_observation"] = obs
            info["terminal_mask"] = done
            fresh, _ = self.reset(options={"reset_ids": done.nonzero().flatten()})
            obs = torch.where(done[:, None], fresh, obs)
        return obs, reward, terminated, truncated, info

    def detach_state(self) -> torch.Tensor:
        """Cut recurrent gradients without reset, resampling or time advancement.

        Returns:
            The current detached observation used by the next rollout segment.
        """
        self._check_open()
        self._state = self._state.detach()
        return self.get_obs().detach()

    def close(self) -> None:
        """Stop stepping; no World or solver cleanup is required."""
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("NeRD learning environment is closed")
