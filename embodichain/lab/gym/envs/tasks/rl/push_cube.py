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
from typing import Dict, Any, Optional, Sequence
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

        extensions = getattr(cfg, "extensions", {}) or {}

        # cfg.sim_cfg.enable_rt = True

        defaults = {
            "success_threshold": 0.1,
            "reaching_reward_weight": 0.1,
            "place_reward_weight": 2.0,
            "place_penalty_weight": 0.5,
            "action_penalty_weight": 0.01,
            "success_bonus_weight": 10.0,
        }
        for name, default in defaults.items():
            value = extensions.get(name, getattr(cfg, name, default))
            setattr(cfg, name, value)
            setattr(self, name, getattr(cfg, name))

        self.last_cube_goal_dist = None

        super().__init__(cfg, **kwargs)

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

    def _init_sim_state(self, **kwargs):
        super()._init_sim_state(**kwargs)
        self.single_action_space = spaces.Box(
            low=-self.joint_limits,
            high=self.joint_limits,
            shape=(6,),
            dtype=np.float32,
        )
        if self.obs_mode == "state":
            self.single_observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
            )

    def _initialize_episode(
        self, env_ids: Optional[Sequence[int]] = None, **kwargs
    ) -> None:
        super()._initialize_episode(env_ids=env_ids, **kwargs)
        cube = self.sim.get_rigid_object("cube")

        # Calculate previous distance (for incremental reward) based on current (possibly randomized) pose
        cube_pos = cube.body_data.pose[:, :3]
        goal_sphere = self.sim.get_rigid_object("goal_sphere")
        goal_pos = goal_sphere.body_data.pose[
            :, :3
        ]  # Get actual goal positions for each environment
        self.last_cube_goal_dist = torch.norm(cube_pos[:, :2] - goal_pos[:, :2], dim=1)

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

    def get_obs(self, **kwargs) -> EnvObs:
        qpos_all = self.robot.body_data.qpos[:, :6]
        ee_pose_matrix = self.robot.compute_fk(
            name="arm", qpos=qpos_all, to_matrix=True
        )
        ee_pos_all = ee_pose_matrix[:, :3, 3]
        cube = self.sim.get_rigid_object("cube")
        cube_pos_all = cube.body_data.pose[:, :3]
        # Get actual goal positions for each environment
        goal_sphere = self.sim.get_rigid_object("goal_sphere")
        goal_pos_all = goal_sphere.body_data.pose[:, :3]
        if self.obs_mode == "state":
            return torch.cat([qpos_all, ee_pos_all, cube_pos_all, goal_pos_all], dim=1)
        return {
            "robot": {"qpos": qpos_all, "ee_pos": ee_pos_all},
            "object": {"cube_pos": cube_pos_all, "goal_pos": goal_pos_all},
        }

    def get_reward(
        self, obs: EnvObs, action: EnvAction, info: Dict[str, Any]
    ) -> torch.Tensor:
        if self.obs_mode == "state":
            ee_pos = obs[:, 6:9]
            cube_pos = obs[:, 9:12]
            goal_pos = obs[:, 12:15]
        else:
            ee_pos = obs["robot"]["ee_pos"]
            cube_pos = obs["object"]["cube_pos"]
            goal_pos = obs["object"]["goal_pos"]
        push_direction = goal_pos - cube_pos
        push_dir_norm = torch.norm(push_direction, dim=1, keepdim=True) + 1e-6
        push_dir_normalized = push_direction / push_dir_norm
        push_pose = (
            cube_pos
            - 0.015 * push_dir_normalized
            + torch.tensor([0, 0, 0.015], device=self.device, dtype=torch.float32)
        )
        ee_to_push_dist = torch.norm(ee_pos - push_pose, dim=1)
        reaching_reward_raw = 1.0 - torch.tanh(5.0 * ee_to_push_dist)
        reaching_reward = self.reaching_reward_weight * reaching_reward_raw
        cube_to_goal_dist = torch.norm(cube_pos[:, :2] - goal_pos[:, :2], dim=1)
        distance_delta = 10.0 * (self.last_cube_goal_dist - cube_to_goal_dist)
        distance_delta_normalized = torch.tanh(distance_delta)
        place_reward = torch.where(
            distance_delta_normalized >= 0,
            self.place_reward_weight * distance_delta_normalized,
            self.place_penalty_weight * distance_delta_normalized,
        )
        self.last_cube_goal_dist = cube_to_goal_dist
        action_magnitude = torch.norm(action, dim=1)
        action_penalty = -self.action_penalty_weight * action_magnitude
        success_bonus_raw = info["success"].float()
        success_bonus = self.success_bonus_weight * success_bonus_raw
        reward = reaching_reward + place_reward + action_penalty + success_bonus
        # Organize reward components in a dedicated "rewards" dict
        # This allows trainer to easily identify and log reward components
        if "rewards" not in info:
            info["rewards"] = {}
        info["rewards"]["reaching_reward"] = reaching_reward
        info["rewards"]["place_reward"] = place_reward
        info["rewards"]["action_penalty"] = action_penalty
        info["rewards"]["success_bonus"] = success_bonus
        return reward

    def get_info(self, **kwargs) -> Dict[str, Any]:
        cube = self.sim.get_rigid_object("cube")
        cube_pos = cube.body_data.pose[:, :3]
        # Get actual goal positions for each environment
        goal_sphere = self.sim.get_rigid_object("goal_sphere")
        goal_pos = goal_sphere.body_data.pose[:, :3]
        xy_distance = torch.norm(cube_pos[:, :2] - goal_pos[:, :2], dim=1)
        is_success = xy_distance < self.success_threshold
        info = {
            "success": is_success,
            "fail": torch.zeros(
                self.cfg.num_envs, device=self.device, dtype=torch.bool
            ),
            "elapsed_steps": self._elapsed_steps,
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
