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
from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["AtomicActionsAgentEnv"]


@register_env("AtomicActionsAgent-v3", max_episode_steps=600)
class AtomicActionsAgentEnv(BaseAgentEnv, EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info

    def is_task_success(self) -> torch.Tensor:
        target_object_name = getattr(self, "agent_success_object", "mug")
        target_object = self.sim.get_rigid_object(target_object_name)
        target_object_pose = target_object.get_local_pose(to_matrix=True)
        target_position = torch.as_tensor(
            getattr(self, "agent_success_position", [0.2489, 0.3970, 0.24]),
            dtype=target_object_pose.dtype,
            device=target_object_pose.device,
        )
        tolerance = float(getattr(self, "agent_success_tolerance", 0.05))
        distance = torch.linalg.norm(
            target_object_pose[:, :3, 3] - target_position, dim=-1
        )
        return distance <= tolerance
