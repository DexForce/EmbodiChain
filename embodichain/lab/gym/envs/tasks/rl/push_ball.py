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
from typing import Dict, Any, Tuple

from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs.rl_env import RLEnv
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.sim.types import EnvObs


@register_env("PushBallRL", max_episode_steps=100, override=True)
class PushBallEnv(RLEnv):
    """Push Ball Gate Task Environment.

    The robot must push a soccer ball into a goal area.
    Success is defined by the ball being within a distance threshold of the goal.
    """

    def __init__(self, cfg=None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()
        super().__init__(cfg, **kwargs)

    def _initialize_episode(self, env_ids=None, **kwargs):
        super()._initialize_episode(env_ids, **kwargs)
        if self.goal_pose is None:
            return
        if "goal_circle" not in self.sim.get_rigid_object_uid_list():
            return
        goal_circle = self.sim.get_rigid_object("goal_circle")
        goal_pose = self.goal_pose.clone()
        goal_pose[:, 2, 3] = 0.001
        if env_ids is None:
            goal_circle.set_local_pose(goal_pose)
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(list(env_ids), device=self.device)
        env_ids = env_ids.to(dtype=torch.long)
        current_pose = goal_circle.get_local_pose(to_matrix=True)
        current_pose[env_ids] = goal_pose[env_ids]
        goal_circle.set_local_pose(current_pose)

    def compute_task_state(
        self, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compute task-specific state: success, fail, and metrics."""
        ball = self.sim.get_rigid_object("soccer_ball")
        ball_pos = ball.body_data.pose[:, :3]

        if self.goal_pose is not None:
            goal_pos = self.goal_pose[:, :3, 3]
            xy_distance = torch.norm(ball_pos[:, :2] - goal_pos[:, :2], dim=1)
            is_success = xy_distance < self.success_threshold
        else:
            xy_distance = torch.zeros(self.num_envs, device=self.device)
            is_success = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )

        is_fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        metrics = {
            "distance_to_goal": xy_distance,
            "ball_height": ball_pos[:, 2],
        }

        return is_success, is_fail, metrics

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        is_timeout = self._elapsed_steps >= self.episode_length
        ball = self.sim.get_rigid_object("soccer_ball")
        ball_pos = ball.body_data.pose[:, :3]
        is_fallen = ball_pos[:, 2] < -0.1
        return is_timeout | is_fallen
