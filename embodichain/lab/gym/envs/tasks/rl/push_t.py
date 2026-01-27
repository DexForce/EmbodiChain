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

from typing import Dict, Any, Sequence

import torch

from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.envs.rl_env import RLEnv
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.types import EnvObs, EnvAction


@register_env("PushTRL", max_episode_steps=50, override=True)
class PushTEnv(RLEnv):
    """Push-T task.

    The task requires pushing a T-shaped block to a goal position on the tabletop.
    Success is defined by the object being within a planar distance threshold.
    """

    def __init__(self, cfg=None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()

        super().__init__(cfg, **kwargs)

        self.control_part_name = getattr(self, "control_part_name", "arm")
        self.require_on_table = getattr(self, "require_on_table", True)
        self.table_height = getattr(self, "table_height", 0.0)

    @property
    def goal_pose(self) -> torch.Tensor | None:
        """Get current goal poses (4x4 matrices) for all environments."""
        return getattr(self, "_goal_pose", None)

    def _draw_goal_marker(self) -> None:
        """Draw axis marker at goal position for visualization."""
        if self.goal_pose is None:
            return

        goal_poses = self.goal_pose.detach().cpu().numpy()
        for arena_idx in range(self.cfg.num_envs):
            marker_name = f"goal_marker_{arena_idx}"
            self.sim.remove_marker(marker_name)

            marker_cfg = MarkerCfg(
                name=marker_name,
                marker_type="axis",
                axis_xpos=[goal_poses[arena_idx]],
                axis_size=0.003,
                axis_len=0.02,
                arena_index=arena_idx,
            )
            self.sim.draw_marker(cfg=marker_cfg)

    def _initialize_episode(
        self, env_ids: Sequence[int] | None = None, **kwargs
    ) -> None:
        super()._initialize_episode(env_ids=env_ids, **kwargs)
        # self._draw_goal_marker()

    def _get_eef_pos(self) -> torch.Tensor:
        """Get end-effector position using FK."""
        if self.control_part_name:
            try:
                joint_ids = self.robot.get_joint_ids(self.control_part_name)
                qpos = self.robot.get_qpos()[:, joint_ids]
                ee_pose = self.robot.compute_fk(
                    name=self.control_part_name, qpos=qpos, to_matrix=True
                )
            except (ValueError, KeyError, AttributeError):
                qpos = self.robot.get_qpos()
                ee_pose = self.robot.compute_fk(qpos=qpos, to_matrix=True)
        else:
            qpos = self.robot.get_qpos()
            ee_pose = self.robot.compute_fk(qpos=qpos, to_matrix=True)
        return ee_pose[:, :3, 3]

    def get_info(self, **kwargs) -> Dict[str, Any]:
        t_obj = self.sim.get_rigid_object("t")
        t_pos = t_obj.body_data.pose[:, :3]
        ee_pos = self._get_eef_pos()

        if self.goal_pose is not None:
            goal_pos = self.goal_pose[:, :3, 3]
            xy_distance = torch.norm(t_pos[:, :2] - goal_pos[:, :2], dim=1)
            is_success = xy_distance < self.success_threshold
            if self.require_on_table:
                is_success = is_success & (t_pos[:, 2] >= self.table_height - 1e-3)
        else:
            xy_distance = torch.zeros(self.cfg.num_envs, device=self.device)
            is_success = torch.zeros(
                self.cfg.num_envs, device=self.device, dtype=torch.bool
            )

        ee_to_t = torch.norm(ee_pos - t_pos, dim=1)
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
            "eef_to_t": ee_to_t,
            "t_height": t_pos[:, 2],
        }
        return info

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        is_timeout = self._elapsed_steps >= self.episode_length
        t_obj = self.sim.get_rigid_object("t")
        t_pos = t_obj.body_data.pose[:, :3]
        is_fallen = t_pos[:, 2] < -0.1
        return is_timeout | is_fallen

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        info = self.get_info(**kwargs)
        return {
            "success": info["success"],
            "distance_to_goal": info["metrics"]["distance_to_goal"],
        }
