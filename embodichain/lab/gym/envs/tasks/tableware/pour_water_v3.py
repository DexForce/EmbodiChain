# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import os
import torch
import numpy as np
from copy import deepcopy
from typing import Dict, Union, Optional, Sequence, Tuple, List

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import configclass, logger

from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv

__all__ = ["PourWaterEnv3"]


@register_env("PourWater-v3", max_episode_steps=600)
class PourWaterEnv3(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed. This is mainly used in the data generation process
        of the imitation learning.

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """

        bottle = self.sim.get_rigid_object("bottle")
        cup = self.sim.get_rigid_object("cup")

        bottle_final_xpos = bottle.get_local_pose(to_matrix=True)
        cup_final_xpos = cup.get_local_pose(to_matrix=True)

        bottle_ret = self._is_fall(bottle_final_xpos)
        cup_ret = self._is_fall(cup_final_xpos)

        return ~(bottle_ret | cup_ret)

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


@register_env("PourWaterAgent-v3", max_episode_steps=600)
class PourWaterAgentEnv3(BaseAgentEnv, PourWaterEnv3):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info