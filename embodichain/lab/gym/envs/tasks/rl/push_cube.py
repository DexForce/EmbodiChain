# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch
import numpy as np
from typing import Dict, Any, Sequence
from gymnasium import spaces

from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.types import EnvObs, EnvAction
from embodichain.utils import logger


@register_env("PushCubeRL", max_episode_steps=50, override=True)
class PushCubeEnv(EmbodiedEnv):
    """Push cube task for reinforcement learning.

    The task involves pushing a cube to a target goal position using a robotic arm.
    The reward consists of reaching reward, placing reward, action penalty, and success bonus.
    """

    def __init__(self, cfg=None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()

        super().__init__(cfg, **kwargs)

    @property
    def goal_pose(self) -> torch.Tensor:
        """Get current goal poses (4x4 matrices) for all environments."""
        return self._goal_pose

    def _draw_goal_marker(self):
        """Draw axis marker at goal position for visualization."""
        goal_sphere = self.sim.get_rigid_object("goal_sphere")
        if goal_sphere is None:
            return

        num_envs = self.cfg.num_envs

        # Get actual goal positions from each arena
        goal_poses = goal_sphere.get_local_pose(to_matrix=True)  # (num_envs, 4, 4)

        # Draw marker for each arena separately
        for arena_idx in range(num_envs):
            marker_name = f"goal_marker_{arena_idx}"

            self.sim.remove_marker(marker_name)

            goal_pose = goal_poses[arena_idx].detach().cpu().numpy()
            marker_cfg = MarkerCfg(
                name=marker_name,
                marker_type="axis",
                axis_xpos=[goal_pose],
                axis_size=0.003,
                axis_len=0.02,
                arena_index=arena_idx,
            )
            self.sim.draw_marker(cfg=marker_cfg)

    def _initialize_episode(
        self, env_ids: Sequence[int] | None = None, **kwargs
    ) -> None:
        super()._initialize_episode(env_ids=env_ids, **kwargs)

        # Draw marker at goal position
        # self._draw_goal_marker()

    def _step_action(self, action: EnvAction) -> EnvAction:
        scaled_action = action * self.action_scale
        scaled_action = torch.clamp(
            scaled_action, -self.joint_limits, self.joint_limits
        )
        current_qpos = self.robot.body_data.qpos
        target_qpos = current_qpos.clone()
        target_qpos[:, :6] += scaled_action[:, :6]
        self.robot.set_qpos(qpos=target_qpos)
        return scaled_action

    def get_info(self, **kwargs) -> Dict[str, Any]:
        cube = self.sim.get_rigid_object("cube")
        cube_pos = cube.body_data.pose[:, :3]

        # Get goal position from event-managed goal pose
        if self.goal_pose is not None:
            goal_pos = self.goal_pose[:, :3, 3]
            xy_distance = torch.norm(cube_pos[:, :2] - goal_pos[:, :2], dim=1)
            is_success = xy_distance < self.success_threshold
        else:
            # Goal not yet set by randomize_target_pose event (e.g., before first reset)
            xy_distance = torch.zeros(self.cfg.num_envs, device=self.device)
            is_success = torch.zeros(
                self.cfg.num_envs, device=self.device, dtype=torch.bool
            )

        info = {
            "success": is_success,
            "fail": torch.zeros(
                self.cfg.num_envs, device=self.device, dtype=torch.bool
            ),
            "elapsed_steps": self._elapsed_steps,
            "goal_pose": self.goal_pose,
        }
        info["metrics"] = {
            "distance_to_goal": xy_distance,
        }
        return info

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        is_timeout = self._elapsed_steps >= self.episode_length
        cube = self.sim.get_rigid_object("cube")
        cube_pos = cube.body_data.pose[:, :3]
        is_fallen = cube_pos[:, 2] < -0.1
        return is_timeout | is_fallen

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        info = self.get_info(**kwargs)
        return {
            "success": info["success"][0].item(),
            "distance_to_goal": info["distance_to_goal"],
        }
