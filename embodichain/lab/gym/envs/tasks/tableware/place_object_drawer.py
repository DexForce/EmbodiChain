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


from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger

__all__ = ["PlaceObjectDrawerEnv"]


@register_env("PlaceObjectDrawer-v1", max_episode_steps=600)
class PlaceObjectDrawerEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed.

        The task is successful if:
        1. Object is within drawer inner_box area
        2. Drawer has been closed

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """
        try:
            object_obj = self.sim.get_rigid_object("object")
            drawer = self.sim.get_articulation("drawer")
        except Exception as e:
            logger.log_warning(f"Object or drawer not found: {e}, returning False.")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Get poses
        object_pose = object_obj.get_local_pose(to_matrix=True)
        # Get drawer inner_box (drawer interior) pose, not outer_box
        inner_box_pose = drawer.get_link_pose("inner_box", to_matrix=True)

        # Extract positions
        object_pos = object_pose[:, :3, 3]  # (num_envs, 3)
        inner_box_pos = inner_box_pose[:, :3, 3]  # (num_envs, 3)

        # Get drawer joint position
        drawer_qpos = drawer.get_qpos()  # (num_envs, num_joints)
        drawer_joint_pos = drawer_qpos[:, 0]

        # Check if drawer has been closed
        drawer_closed = drawer_joint_pos < 0.05

        # Check if object is within drawer inner_box area
        xy_diff = torch.abs(object_pos[:, :2] - inner_box_pos[:, :2])  # (num_envs, 2)
        xy_tolerance = torch.tensor(
            [0.03, 0.03], dtype=torch.float32, device=self.device
        )
        object_in_drawer_xy = torch.all(xy_diff < xy_tolerance.unsqueeze(0), dim=1)

        # Object must be in drawer and drawer must be closed
        success = drawer_closed & object_in_drawer_xy

        return success
