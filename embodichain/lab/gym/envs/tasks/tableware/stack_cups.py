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

__all__ = ["StackCupsEnv"]


@register_env("StackCups-v1", max_episode_steps=600)
class StackCupsEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed.

        The task is successful if:
        1. Both cups haven't fallen over
        2. Cups are aligned in xy plane
        3. Top cup is only slightly higher than base cup

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """
        try:
            cup1 = self.sim.get_rigid_object("cup_1")
            cup2 = self.sim.get_rigid_object("cup_2")
        except Exception as e:
            logger.log_warning(f"Cups not found: {e}, returning False.")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Get cup poses
        cup1_pose = cup1.get_local_pose(to_matrix=True)
        cup2_pose = cup2.get_local_pose(to_matrix=True)

        # Extract positions
        cup1_pos = cup1_pose[:, :3, 3]  # (num_envs, 3)
        cup2_pos = cup2_pose[:, :3, 3]  # (num_envs, 3)

        # Check if cups haven't fallen
        cup1_fallen = self._is_fall(cup1_pose)
        cup2_fallen = self._is_fall(cup2_pose)

        # Determine which cup is lower (should be cup1)
        cup1_is_lower = cup1_pos[:, 2] < cup2_pos[:, 2]

        # Use the lower cup as base, calculate expected position of top cup
        base_cup_pos = torch.where(
            cup1_is_lower.unsqueeze(1).expand_as(cup1_pos), cup1_pos, cup2_pos
        )
        top_cup_pos = torch.where(
            cup1_is_lower.unsqueeze(1).expand_as(cup2_pos), cup2_pos, cup1_pos
        )

        # Top cup should be slightly higher than base cup
        cup_rim_offset = 0.015
        expected_top_pos = torch.stack(
            [
                base_cup_pos[:, 0],
                base_cup_pos[:, 1],
                base_cup_pos[:, 2] + cup_rim_offset,
            ],
            dim=1,
        )

        # Tolerance
        eps = torch.tensor([0.04, 0.04, 0.02], dtype=torch.float32, device=self.device)

        # Check if top cup is within tolerance of expected position
        position_diff = torch.abs(top_cup_pos - expected_top_pos)  # (num_envs, 3)
        stacked_correctly = torch.all(
            position_diff < eps.unsqueeze(0), dim=1
        )  # (num_envs,)

        # Task succeeds if cups are stacked correctly and haven't fallen
        success = stacked_correctly & ~cup1_fallen & ~cup2_fallen

        return success

    def _is_fall(self, pose: torch.Tensor) -> torch.Tensor:
        # Extract z-axis from rotation matrix (last column, first 3 elements)
        pose_rz = pose[:, :3, 2]
        world_z_axis = torch.tensor([0, 0, 1], dtype=pose.dtype, device=pose.device)

        # Compute dot product for each batch element
        dot_product = torch.sum(pose_rz * world_z_axis, dim=-1)  # Shape: (batch_size,)

        # Clamp to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Compute angle and check if fallen
        angle = torch.arccos(dot_product)
        return angle >= torch.pi / 4
