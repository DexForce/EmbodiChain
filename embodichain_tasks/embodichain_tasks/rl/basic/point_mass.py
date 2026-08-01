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

"""Differentiable two-dimensional point-mass navigation task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from embodichain.learning.rl.env import register_learning_env

__all__ = ["PointMassEnv"]


@register_learning_env("PointMassRL", override=True)
class PointMassEnv:
    """Navigate a damped point mass to a goal while avoiding two obstacles.

    Always differentiable: PPO/GRPO run under ``torch.no_grad()``, APG keeps
    the graph. ``success_bonus`` is optional reward shaping for on-policy
    methods that optimize dense distance rewards but are evaluated on strict
    success; leave it at ``0`` for APG.
    """

    observation_dim = 14
    action_dim = 2

    def __init__(
        self,
        num_envs: int = 64,
        device: torch.device | str = "cpu",
        *,
        max_episode_steps: int = 100,
        workspace_half: float = 1.0,
        mass: float = 1.0,
        force_scale: float = 5.0,
        linear_damping: float = 5.0,
        dt: float = 1.0 / 60.0,
        success_threshold: float = 0.03,
        success_bonus: float = 0.0,
        obstacle_radius_range: Sequence[float] = (0.08, 0.15),
        obstacle_min_goal_distance: float = 0.25,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
        if len(obstacle_radius_range) != 2:
            raise ValueError("obstacle_radius_range must contain two values.")
        radius_min, radius_max = map(float, obstacle_radius_range)
        if radius_min <= 0 or radius_max < radius_min:
            raise ValueError("Invalid obstacle_radius_range.")
        if success_bonus < 0:
            raise ValueError("success_bonus cannot be negative.")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.max_episode_steps = int(max_episode_steps)
        self.workspace_half = float(workspace_half)
        self.mass = float(mass)
        self.force_scale = float(force_scale)
        self.linear_damping = float(linear_damping)
        self.dt = float(dt)
        self.success_threshold = float(success_threshold)
        self.success_bonus = float(success_bonus)
        self.obstacle_radius_range = (radius_min, radius_max)
        self.obstacle_min_goal_distance = float(obstacle_min_goal_distance)

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(torch.seed())
        self.position = torch.zeros(self.num_envs, 2, device=self.device)
        self.velocity = torch.zeros_like(self.position)
        self.goal_position = torch.zeros_like(self.position)
        self.obstacle_position = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self.obstacle_radius = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_action = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self.episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.path_length = torch.zeros(self.num_envs, device=self.device)
        self.collision_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset all environments, or selected IDs from ``options``."""
        if seed is not None:
            self._generator.manual_seed(int(seed))
        env_ids = None if options is None else options.get("env_ids")
        ids = self._normalize_env_ids(env_ids)
        self._reset_ids(ids)
        return self._observation(), {}

    def step(self, action: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Advance the differentiable dynamics and auto-reset finished rows."""
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).clamp(
            -1.0, 1.0
        )
        if tuple(action.shape) != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected action shape {(self.num_envs, self.action_dim)}, "
                f"got {tuple(action.shape)}."
            )

        previous_position = self.position
        damping = 1.0 - self.linear_damping * self.dt
        self.velocity = (
            self.velocity * damping + action * self.force_scale / self.mass * self.dt
        )
        self.position = (self.position + self.velocity * self.dt).clamp(
            -self.workspace_half, self.workspace_half
        )
        self.episode_step = self.episode_step + 1
        self.path_length = (
            self.path_length + (self.position - previous_position).norm(dim=-1).detach()
        )

        center_distance = (self.position[:, None, :] - self.obstacle_position).norm(
            dim=-1
        )
        penetration = torch.clamp(self.obstacle_radius - center_distance, min=0.0)
        collided = (penetration > 0).any(dim=-1)
        self.collision_count = self.collision_count + collided.long()

        distance = (self.position - self.goal_position).norm(dim=-1)
        terminated = distance < self.success_threshold
        truncated = self.episode_step >= self.max_episode_steps
        done = terminated | truncated
        reward = (
            -distance
            + 0.5 * torch.exp(-(distance.square()) / (2.0 * 0.05**2))
            - 2.0 * penetration.square().sum(dim=-1)
            - 0.01 * self.velocity.square().sum(dim=-1)
            - 0.001 * (action - self.last_action).square().sum(dim=-1)
            + self.success_bonus * terminated.float()
        )

        terminal_distance = distance.detach()
        terminal_path_length = self.path_length.detach().clone()
        terminal_collisions = self.collision_count.detach().clone()
        info = {
            "success": terminated.detach(),
            "metrics": {
                "final_distance": terminal_distance,
                "path_length": terminal_path_length,
                "collision_count": terminal_collisions,
            },
        }

        self.last_action = action.detach()
        if bool(done.any()):
            self._reset_ids(torch.nonzero(done, as_tuple=False).squeeze(-1))
        return self._observation(), reward, terminated, truncated, info

    def detach_state(self) -> torch.Tensor:
        """Detach dynamic state at a truncated-backpropagation boundary."""
        self.position = self.position.detach()
        self.velocity = self.velocity.detach()
        self.last_action = self.last_action.detach()
        return self._observation()

    def close(self) -> None:
        """No-op; the tensor-only task owns no external resources."""

    def _observation(self) -> torch.Tensor:
        return torch.cat(
            (
                self.position,
                self.velocity,
                self.goal_position,
                self.last_action,
                self.obstacle_position.reshape(self.num_envs, -1),
                self.obstacle_radius,
            ),
            dim=-1,
        )

    def _normalize_env_ids(
        self, env_ids: torch.Tensor | Sequence[int] | None
    ) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(
            -1
        )

    def _reset_ids(self, ids: torch.Tensor) -> None:
        if ids.numel() == 0:
            return
        count = int(ids.numel())
        goal = self._uniform(count, 2, scale=0.7 * self.workspace_half)
        start = self._uniform(count, 2, scale=0.6 * self.workspace_half)
        obstacle_position = self._sample_obstacles(goal)
        radius_min, radius_max = self.obstacle_radius_range
        obstacle_radius = radius_min + (radius_max - radius_min) * torch.rand(
            count, 2, generator=self._generator, device=self.device
        )

        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        mask_2d = mask[:, None]
        mask_3d = mask[:, None, None]

        full_goal = self.goal_position.detach().clone()
        full_goal[ids] = goal
        full_start = self.position.detach().clone()
        full_start[ids] = start
        full_obstacles = self.obstacle_position.detach().clone()
        full_obstacles[ids] = obstacle_position
        full_radii = self.obstacle_radius.detach().clone()
        full_radii[ids] = obstacle_radius

        self.position = torch.where(mask_2d, full_start, self.position)
        self.velocity = torch.where(
            mask_2d, torch.zeros_like(self.velocity), self.velocity
        )
        self.goal_position = torch.where(mask_2d, full_goal, self.goal_position)
        self.obstacle_position = torch.where(
            mask_3d, full_obstacles, self.obstacle_position
        )
        self.obstacle_radius = torch.where(mask_2d, full_radii, self.obstacle_radius)
        self.last_action = torch.where(
            mask_2d, torch.zeros_like(self.last_action), self.last_action
        )
        self.episode_step = torch.where(
            mask, torch.zeros_like(self.episode_step), self.episode_step
        )
        self.path_length = torch.where(
            mask, torch.zeros_like(self.path_length), self.path_length
        )
        self.collision_count = torch.where(
            mask, torch.zeros_like(self.collision_count), self.collision_count
        )

    def _sample_obstacles(self, goal: torch.Tensor) -> torch.Tensor:
        count = goal.shape[0]
        result = self._uniform(count, 2, 2, scale=0.8 * self.workspace_half)
        valid = (result - goal[:, None, :]).norm(
            dim=-1
        ) > self.obstacle_min_goal_distance
        for _ in range(50):
            if bool(valid.all()):
                break
            candidates = self._uniform(count, 2, 2, scale=0.8 * self.workspace_half)
            result = torch.where(valid[:, :, None], result, candidates)
            valid = (result - goal[:, None, :]).norm(
                dim=-1
            ) > self.obstacle_min_goal_distance
        fallback = -0.5 * goal[:, None, :].expand(-1, 2, -1)
        return torch.where(valid[:, :, None], result, fallback)

    def _uniform(self, *shape: int, scale: float) -> torch.Tensor:
        return (
            torch.rand(*shape, generator=self._generator, device=self.device) * 2.0
            - 1.0
        ) * scale
