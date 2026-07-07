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

from copy import deepcopy
import math
from typing import Any

__all__ = [
    "DUAL_UR5_AGENT_ARM_SLOTS",
    "DUAL_UR5_ARM_AIM_YAW_OFFSET",
    "make_dual_ur5_arm_slot_config",
]


DUAL_UR5_AGENT_ARM_SLOTS = {
    "left": {
        "arm": "right_arm",
        "eef": "right_eef",
    },
    "right": {
        "arm": "left_arm",
        "eef": "left_eef",
    },
}
DUAL_UR5_ARM_AIM_YAW_OFFSET = {
    "left": math.pi,
    "right": 0.0,
}


def make_dual_ur5_arm_slot_config() -> dict[str, Any]:
    """Return the Dual-UR5 semantic-slot to physical-control-part binding."""
    return {
        "agent_arm_slots": deepcopy(DUAL_UR5_AGENT_ARM_SLOTS),
        "arm_aim_yaw_offset": deepcopy(DUAL_UR5_ARM_AIM_YAW_OFFSET),
    }
