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

from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.base_agent_env import (
    BaseAgentEnv,
)
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.envs.tasks.tableware.rearrangement import RearrangementEnv
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["RearrangementAgentEnv"]


@register_env("RearrangementAgent-v3", max_episode_steps=600)
class RearrangementAgentEnv(BaseAgentEnv, RearrangementEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info

    def is_task_success(self):
        fork = self.sim.get_rigid_object("fork")
        spoon = self.sim.get_rigid_object("spoon")
        plate = self.sim.get_rigid_object("plate")

        plate_pose = plate.get_local_pose(to_matrix=True)
        spoon_place_target_y = plate_pose[0, 1, 3] - 0.16
        fork_place_target_y = plate_pose[0, 1, 3] + 0.16

        spoon_pose = spoon.get_local_pose(to_matrix=True)
        spoon_y = spoon_pose[0, 1, 3]

        fork_pose = fork.get_local_pose(to_matrix=True)
        fork_y = fork_pose[0, 1, 3]

        tolerance = self.metadata.get("success_params", {}).get("tolerance", 0.02)

        return (
            abs(spoon_y - spoon_place_target_y) <= tolerance
            and abs(fork_y - fork_place_target_y) <= tolerance
        )
