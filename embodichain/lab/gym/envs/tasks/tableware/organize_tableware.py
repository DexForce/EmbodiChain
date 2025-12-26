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

__all__ = ["OrganizeTablewareEnv"]


@register_env("OrganizeTableware-v1", max_episode_steps=600)
class OrganizeTablewareEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

        # Define target positions for different tableware types
        # Left side: fork, Right side: spoon
        self.fork_target_pos = torch.tensor(
            [0.725, -0.1, 0.86], dtype=torch.float32, device=self.device
        )
        self.spoon_target_pos = torch.tensor(
            [0.825, 0.1, 0.86], dtype=torch.float32, device=self.device
        )

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed.

        The task is successful if:
        1. Fork is placed in the left target area
        2. Spoon is placed in the right target area
        3. Both fork and spoon are oriented towards x-axis

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """
        try:
            fork = self.sim.get_rigid_object("fork")
            spoon = self.sim.get_rigid_object("spoon")
        except Exception as e:
            logger.log_warning(f"Fork or spoon not found: {e}, returning False.")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Get poses
        fork_pose = fork.get_local_pose(to_matrix=True)
        spoon_pose = spoon.get_local_pose(to_matrix=True)

        # Extract positions
        fork_pos = fork_pose[:, :3, 3]  # (num_envs, 3)
        spoon_pos = spoon_pose[:, :3, 3] 

        # Tolerance for checking if objects are in target area
        xy_tolerance = torch.tensor(
            [0.08, 0.05], dtype=torch.float32, device=self.device
        )
        z_tolerance = 0.05

        # Check if fork is in left target area
        fork_target = self.fork_target_pos.unsqueeze(0).repeat(self.num_envs, 1)
        fork_xy_diff = torch.abs(fork_pos[:, :2] - fork_target[:, :2])
        fork_z_diff = torch.abs(fork_pos[:, 2] - fork_target[:, 2])
        fork_in_target = torch.all(fork_xy_diff < xy_tolerance.unsqueeze(0), dim=1) & (
            fork_z_diff < z_tolerance
        )

        # Check if spoon is in right target area
        spoon_target = self.spoon_target_pos.unsqueeze(0).repeat(self.num_envs, 1)
        spoon_xy_diff = torch.abs(spoon_pos[:, :2] - spoon_target[:, :2])
        spoon_z_diff = torch.abs(spoon_pos[:, 2] - spoon_target[:, 2])
        spoon_in_target = torch.all(
            spoon_xy_diff < xy_tolerance.unsqueeze(0), dim=1
        ) & (spoon_z_diff < z_tolerance)

        # Check orientation: both fork and spoon should be oriented towards x-axis
        fork_oriented = self._is_oriented_towards_x(fork_pose)
        spoon_oriented = self._is_oriented_towards_x(spoon_pose)

        # Both must be in their target areas and correctly oriented
        success = fork_in_target & spoon_in_target & fork_oriented & spoon_oriented

        return success

    def _is_oriented_towards_x(self, pose: torch.Tensor) -> torch.Tensor:
        # Extract x-axis from rotation matrix (first column, first 3 elements)
        pose_rx = pose[:, :3, 0]  # (num_envs, 3)
        world_x_axis = torch.tensor([1, 0, 0], dtype=pose.dtype, device=pose.device)

        # Compute dot product for each batch element
        dot_product = torch.sum(pose_rx * world_x_axis, dim=-1)  # (num_envs,)

        # Clamp to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Compute angle between object x-axis and world x-axis
        angle = torch.arccos(dot_product)

        # Angle difference should be less than pi/6
        orientation_tolerance = torch.pi / 6
        return angle < orientation_tolerance
