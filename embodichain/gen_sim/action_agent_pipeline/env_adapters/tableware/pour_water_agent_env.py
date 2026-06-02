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
from embodichain.lab.gym.envs.tasks.tableware.pour_water.pour_water import PourWaterEnv
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["PourWaterAgentEnv"]


@register_env("PourWaterAgent-v3", max_episode_steps=600)
class PourWaterAgentEnv(BaseAgentEnv, PourWaterEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info
