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

from .base_env import *
from .embodied_env import *
from .tasks import *
from .wrapper import *

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

# Specific task environments
from embodichain.lab.gym.envs.tasks.tableware.pour_water.pour_water import (
    PourWaterEnv,
)
from embodichain.lab.gym.envs.tasks.tableware.scoop_ice import ScoopIce

# Reinforcement learning environments
from embodichain.lab.gym.envs.tasks.rl.push_cube import PushCubeEnv

from embodichain.lab.gym.envs.tasks.special.simple_task import SimpleTaskEnv
