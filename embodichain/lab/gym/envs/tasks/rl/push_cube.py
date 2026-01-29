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


@register_env("PushCubeRL", max_episode_steps=50, override=True)
class PushCubeEnv(RLEnv):
    """Push cube task for reinforcement learning.

    The task involves pushing a cube to a target goal position using a robotic arm.
    The reward consists of reaching reward, placing reward, action penalty, and success bonus.
    """

    def __init__(self, cfg=None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()
        super().__init__(cfg, **kwargs)

    def compute_task_state(
        self, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        cube = self.sim.get_rigid_object("cube")
        cube_pos = cube.get_local_pose(to_matrix=True)[:, :3, 3]

        # Check if goal_pose is defined (set by randomize_target_pose event)
        goal_pose = getattr(self, "goal_pose", None)
        if goal_pose is not None:
            goal_pos = goal_pose[:, :3, 3]
            xy_distance = torch.norm(cube_pos[:, :2] - goal_pos[:, :2], dim=1)
            is_success = xy_distance < self.success_threshold
        else:
            xy_distance = torch.zeros(self.num_envs, device=self.device)
            is_success = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )

        is_fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        metrics = {"distance_to_goal": xy_distance}

        return is_success, is_fail, metrics

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        is_timeout = self._elapsed_steps >= self.episode_length
        cube = self.sim.get_rigid_object("cube")
        cube_pos = cube.get_local_pose(to_matrix=True)[:, :3, 3]
        is_fallen = cube_pos[:, 2] < -0.1
        return is_timeout | is_fallen
