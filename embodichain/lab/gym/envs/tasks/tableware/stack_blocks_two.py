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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import torch
import numpy as np

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger

__all__ = ["StackBlocksTwoEnv"]


@register_env("StackBlocksTwo-v1", max_episode_steps=600)
class StackBlocksTwoEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed. This is mainly used in the data generation process
        of the imitation learning.

        The task is successful if:
        1. Block2 is stacked on top of Block1
        2. Both blocks haven't fallen over

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """
        try:
            block1 = self.sim.get_rigid_object("block_1")
            block2 = self.sim.get_rigid_object("block_2")
        except:
            logger.log_warning("Block1 or Block2 not found, returning False.")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Get poses
        block1_pose = block1.get_local_pose(to_matrix=True)
        block2_pose = block2.get_local_pose(to_matrix=True)

        # Extract positions
        block1_pos = block1_pose[:, :3, 3]  # (num_envs, 3)
        block2_pos = block2_pose[:, :3, 3]

        # Check if blocks haven't fallen
        block1_fallen = self._is_fall(block1_pose)
        block2_fallen = self._is_fall(block2_pose)

        # Block2 should be on top of block1
        expected_block2_pos = torch.stack(
            [
                block1_pos[:, 0],
                block1_pos[:, 1],
                block1_pos[:, 2] + 0.05,  # block1 z + block height
            ],
            dim=1,
        )

        # Tolerance
        eps = torch.tensor(
            [0.025, 0.025, 0.012], dtype=torch.float32, device=self.device
        )

        # Check if block2 is within tolerance of expected position
        position_diff = torch.abs(block2_pos - expected_block2_pos)  # (num_envs, 3)
        within_tolerance = torch.all(
            position_diff < eps.unsqueeze(0), dim=1
        )  # (num_envs,)

        # Task succeeds if blocks are stacked correctly and haven't fallen
        success = within_tolerance & ~block1_fallen & ~block2_fallen

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
