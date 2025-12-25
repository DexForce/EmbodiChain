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

__all__ = ["BlocksRankingSizeEnv"]


@register_env("BlocksRankingSize-v1", max_episode_steps=600)
class BlocksRankingSizeEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed.

        The task is successful if:
        1. Three blocks are arranged in size order from left to right:
           - Large block (block_1) x < Medium block (block_2) x < Small block (block_3) x
        2. All blocks are close together (within tolerance)

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """
        try:
            block1 = self.sim.get_rigid_object("block_1")  # Large
            block2 = self.sim.get_rigid_object("block_2")  # Medium
            block3 = self.sim.get_rigid_object("block_3")  # Small
        except Exception as e:
            logger.log_warning(f"Blocks not found: {e}, returning False.")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Get block poses
        block1_pose = block1.get_local_pose(to_matrix=True)
        block2_pose = block2.get_local_pose(to_matrix=True)
        block3_pose = block3.get_local_pose(to_matrix=True)

        # Extract positions (x, y, z)
        block1_pos = block1_pose[:, :3, 3]  # (num_envs, 3)
        block2_pos = block2_pose[:, :3, 3]
        block3_pos = block3_pose[:, :3, 3]

        # Tolerance for checking if blocks are close together
        # Same as RoboTwin: eps = [0.13, 0.03]
        eps = torch.tensor([0.13, 0.03], dtype=torch.float32, device=self.device)

        # Check if blocks are close together in x-y plane
        # block1 and block2 should be close
        block1_block2_diff = torch.abs(block1_pos[:, :2] - block2_pos[:, :2])
        blocks_close_12 = torch.all(block1_block2_diff < eps.unsqueeze(0), dim=1)

        # block2 and block3 should be close
        block2_block3_diff = torch.abs(block2_pos[:, :2] - block3_pos[:, :2])
        blocks_close_23 = torch.all(block2_block3_diff < eps.unsqueeze(0), dim=1)

        # Check size order: block1 (large) x < block2 (medium) x < block3 (small) x
        size_order = (block1_pos[:, 0] < block2_pos[:, 0]) & (
            block2_pos[:, 0] < block3_pos[:, 0]
        )

        # All conditions must be satisfied
        success = blocks_close_12 & blocks_close_23 & size_order

        return success
