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

"""Manager functors shared by native locomotion tasks."""

from __future__ import annotations

from .actions import DefaultJointPositionTerm, DelayedDefaultJointPositionTerm
from .events import push_articulation_by_setting_velocity
from .observations import velocity_locomotion_observation
from .rewards import (
    velocity_locomotion_reward,
    velocity_locomotion_total_reward,
)

__all__ = [
    "DefaultJointPositionTerm",
    "DelayedDefaultJointPositionTerm",
    "push_articulation_by_setting_velocity",
    "velocity_locomotion_observation",
    "velocity_locomotion_reward",
    "velocity_locomotion_total_reward",
]
