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

from typing import Any, Dict

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.utils import configclass


@configclass
class RLEnvCfg(EmbodiedEnvCfg):
    """Extended configuration for RL environments built from gym-style specs."""

    env_id: str = ""
    extensions: Dict[str, Any] = {}

    @classmethod
    def from_dict(cls, d):
        """Create an instance from a dictionary."""
        return cls(**d)
