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

import enum
import re
from typing import TypeVar

__all__ = [
    "DexforceW1Version",
    "DexforceW1HandVersion",
    "DexforceW1ArmSide",
    "DexforceW1Type",
    "DexforceW1HandBrand",
]

_W1EnumT = TypeVar("_W1EnumT", bound="_W1Enum")


class _W1Enum(enum.Enum):
    @classmethod
    def _parse_label(cls) -> str:
        name = cls.__name__.removeprefix("DexforceW1")
        words = re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()
        return f"Dexforce W1 {words}"

    @classmethod
    def parse(cls: type[_W1EnumT], value) -> _W1EnumT:
        """Parse an enum instance, member name, or serialized value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.lower()
            for member in cls:
                if normalized in (member.name.lower(), str(member.value).lower()):
                    return member
        raise ValueError(f"Invalid {cls._parse_label()}: {value!r}")


class DexforceW1Version(_W1Enum):
    """Released version of the W1 robot body and arms."""

    V021 = "v021"
    V022 = "v022"
    V025 = "v025"


class DexforceW1HandVersion(_W1Enum):
    """Released version of an external W1 hand or gripper asset."""

    V021 = "v021"


class DexforceW1ArmSide(_W1Enum):
    """Arm side for DexforceW1: left or right."""

    LEFT = "left"
    RIGHT = "right"


class DexforceW1Type(_W1Enum):
    """Component type for DexforceW1."""

    CHASSIS = "chassis"
    TORSO = "torso"
    EYES = "eyes"
    HEAD = "head"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    FULL_BODY = "full_body"  # Full robot


class DexforceW1HandBrand(_W1Enum):
    BRAINCO_HAND = "BRAINCO_HAND"
    DH_PGC_GRIPPER = "DH_PGC_GRIPPER"
    DH_PGC_GRIPPER_M = "DH_PGC_GRIPPER_M"
