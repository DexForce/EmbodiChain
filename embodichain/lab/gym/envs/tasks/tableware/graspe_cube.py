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

from typing import Dict, Optional

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["GraspeCubeEnv", "GraspeCubeAgentEnv"]


@register_env("GraspeCube-v1", max_episode_steps=600)
class GraspeCubeEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

    def is_task_success(self, **kwargs) -> torch.Tensor:
        cube = self.sim.get_rigid_object("cube")
        container = self.sim.get_rigid_object("container")

        cube_pose = cube.get_local_pose(to_matrix=True)
        container_pose = container.get_local_pose(to_matrix=True)

        cube_pos = cube_pose[:, :3, 3]
        container_pos = container_pose[:, :3, 3]

        success_cfg: Dict = self.metadata.get("success_params", {})
        tolerance_xy: float = success_cfg.get("tolerance_xy", 0.05)
        min_height_delta: float = success_cfg.get("min_height_delta", 0.02)

        xy_dist = torch.norm(cube_pos[:, :2] - container_pos[:, :2], dim=-1)
        height_ok = cube_pos[:, 2] >= container_pos[:, 2] + min_height_delta

        return (xy_dist <= tolerance_xy) & height_ok


@register_env("GraspeCubeAgent-v1", max_episode_steps=600)
class GraspeCubeAgentEnv(BaseAgentEnv, GraspeCubeEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info

