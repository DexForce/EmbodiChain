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

"""Unitree G1 rough-terrain velocity task."""

from __future__ import annotations

from embodichain.lab.gym.utils.registration import register_env
from .rough.config import load_config
from .rough.runtime import RoughVelocityEnv

__all__ = ["UnitreeG1RoughEnv"]

_CONFIG = load_config("g1_rough")


@register_env(
    "UnitreeG1RoughRL-v1", max_episode_steps=1000, override=True, supports_rl=True
)
class UnitreeG1RoughEnv(RoughVelocityEnv):
    """Track planar velocity commands with the 37-DOF G1 model on rough terrain."""

    velocity_task_config = _CONFIG
    foot_link_names = tuple(_CONFIG.data["feet"])
    foot_offsets = ((0.0, 0.0, 0.0),) * len(foot_link_names)
    illegal_contact_link_names = tuple(_CONFIG.data["illegal_bodies"])
    orientation_link_name = "torso_link"
    explicit_pd_effort_control = False
