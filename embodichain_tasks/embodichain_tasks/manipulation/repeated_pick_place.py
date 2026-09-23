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

"""RLinf-facing repeated pick-and-place environment."""

from __future__ import annotations

from typing import Any

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.types import EnvObs

__all__ = ["RepeatedPickPlaceRlinfEnv", "RepeatedPickPlaceRlinfJointEnv"]


@register_env(
    "RepeatedPickPlaceRlinf-Franka-v1",
    max_episode_steps=240,
    override=True,
    supports_rl=True,
)
class RepeatedPickPlaceRlinfEnv(EmbodiedEnv):
    """Single-cube pick-and-place task with a flat VLA action interface."""

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        super().__init__(cfg or EmbodiedEnvCfg(), **kwargs)
        self._success_hold_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )

    def reset(self, *args: Any, **kwargs: Any) -> tuple[EnvObs, dict[str, Any]]:
        observation, info = super().reset(*args, **kwargs)
        self._success_hold_steps.zero_()
        return observation, info

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        cube = self.sim.get_rigid_object("cube")
        target = self.sim.get_rigid_object("target")
        cube_pos = cube.get_local_pose(to_matrix=True)[:, :3, 3]
        target_pos = target.get_local_pose(to_matrix=True)[:, :3, 3]
        distance = torch.linalg.vector_norm(cube_pos[:, :2] - target_pos[:, :2], dim=1)
        height_error = torch.abs(cube_pos[:, 2] - target_pos[:, 2])
        hand_ids = self.robot.get_joint_ids("hand", remove_mimic=True)
        hand_open = self.robot.get_qpos()[:, hand_ids].mean(dim=-1) > 0.02
        candidate = (distance < 0.08) & (height_error < 0.08) & hand_open
        counters = getattr(
            self,
            "_success_hold_steps",
            torch.zeros(self.num_envs, device=self.device, dtype=torch.int32),
        )
        counters = torch.where(candidate, counters + 1, torch.zeros_like(counters))
        self._success_hold_steps = counters
        success = counters >= 3
        fail = torch.zeros_like(success)
        return (
            success,
            fail,
            {
                "cube_target_distance": distance,
                "success_hold_steps": counters,
            },
        )

    def check_truncated(self, obs: EnvObs, info: dict[str, Any]) -> torch.Tensor:
        del obs, info
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)


@register_env(
    "RepeatedPickPlaceRlinfJoint-Franka-v1",
    max_episode_steps=240,
    override=True,
    supports_rl=True,
)
class RepeatedPickPlaceRlinfJointEnv(RepeatedPickPlaceRlinfEnv):
    """Joint-position action variant of repeated pick-and-place."""
