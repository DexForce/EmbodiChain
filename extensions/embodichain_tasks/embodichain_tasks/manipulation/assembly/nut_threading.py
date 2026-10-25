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

"""Native robot-held nut threading task."""

from __future__ import annotations

from embodichain.lab.gym.envs.assembly_env import BlindAssemblyEnv
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["BlindNutThreadingEnv"]


@register_env("BlindNutThreadingRL-v1", max_episode_steps=750, supports_rl=True)
class BlindNutThreadingEnv(BlindAssemblyEnv):
    """Nut threading through contact and proprioceptive feedback."""

    assembly_kind = "nut"
