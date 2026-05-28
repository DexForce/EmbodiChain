# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

from __future__ import annotations

from typing import Dict, Optional

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.tasks.tableware.single_arm_agent_env import (
    SingleArmAgentEnv,
)
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["UR10CupTransferEnv", "UR10CupTransferAgentEnv"]


@register_env("UR10CupTransfer-v3", max_episode_steps=600)
class UR10CupTransferEnv(EmbodiedEnv):
    """UR10 single-arm task for moving a cup across the table."""

    def is_task_success(self, **kwargs) -> torch.Tensor:
        cup = self.sim.get_rigid_object("cup")
        cup_pose = cup.get_local_pose(to_matrix=True)
        target_xy = torch.as_tensor(
            getattr(self, "target_xy", [0.75, -0.18]),
            dtype=cup_pose.dtype,
            device=cup_pose.device,
        )
        tolerance = float(getattr(self, "success_tolerance", 0.04))

        xy_distance = torch.linalg.norm(cup_pose[:, :2, 3] - target_xy, dim=-1)
        return (xy_distance <= tolerance) & ~self._is_fall(cup_pose)

    def _is_fall(self, pose: torch.Tensor) -> torch.Tensor:
        pose_rz = pose[:, :3, 2]
        world_z_axis = torch.tensor([0, 0, 1], dtype=pose.dtype, device=pose.device)
        dot_product = torch.sum(pose_rz * world_z_axis, dim=-1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle = torch.arccos(dot_product)
        return angle >= torch.pi / 4


@register_env("UR10CupTransferAgent-v3", max_episode_steps=600)
class UR10CupTransferAgentEnv(SingleArmAgentEnv, UR10CupTransferEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info
