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

"""Serialized action vocabulary shared by generation and runtime validation."""

from __future__ import annotations

from typing import Final

__all__ = [
    "ARM_ACTION_KEYS",
    "ATOMIC_ACTION_CLASSES",
    "CONTROL_ARM",
    "CONTROL_HAND",
    "DUAL_ARM_NAME",
    "LEFT_ARM_ACTION_KEY",
    "LEFT_ARM_NAME",
    "MAX_COORDINATED_PAYLOADS",
    "OBJECT_ORIENTATION_AXES",
    "OBJECT_ORIENTATION_GOALS",
    "POSE_REFERENCES",
    "RIGHT_ARM_ACTION_KEY",
    "RIGHT_ARM_NAME",
    "SUPPORTED_CONTROLS",
]

LEFT_ARM_NAME: Final = "left_arm"
RIGHT_ARM_NAME: Final = "right_arm"
DUAL_ARM_NAME: Final = "dual_arm"
LEFT_ARM_ACTION_KEY: Final = "left_arm_action"
RIGHT_ARM_ACTION_KEY: Final = "right_arm_action"
ARM_ACTION_KEYS: Final = frozenset({LEFT_ARM_ACTION_KEY, RIGHT_ARM_ACTION_KEY})

OBJECT_ORIENTATION_GOALS: Final = frozenset(
    {"preserve", "upright", "lay_flat", "axis_align"}
)
OBJECT_ORIENTATION_AXES: Final = frozenset(
    {"none", "x", "y", "long_axis", "short_axis"}
)
POSE_REFERENCES: Final = frozenset({"object", "absolute", "relative"})

# Generation and runtime enforce the same upper bound so a generated action
# cannot pass schema construction and then fail solely because of count drift.
MAX_COORDINATED_PAYLOADS: Final = 4

ATOMIC_ACTION_CLASSES: Final = frozenset(
    {
        "CoordinatedPickment",
        "PickUp",
        "MoveEndEffector",
        "MoveJoints",
        "MoveHeldObject",
        "Place",
    }
)
CONTROL_ARM: Final = "arm"
CONTROL_HAND: Final = "hand"
SUPPORTED_CONTROLS: Final = frozenset({CONTROL_ARM, CONTROL_HAND})
