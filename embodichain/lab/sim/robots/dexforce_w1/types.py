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

__all__ = [
    "DexforceW1Version",
    "DexforceW1HandVersion",
    "DexforceW1ArmSide",
    "DexforceW1Type",
    "DexforceW1HandBrand",
    "parse_w1_version",
    "parse_w1_hand_version",
    "parse_w1_arm_side",
    "parse_w1_hand_brand",
]


class DexforceW1Version(enum.Enum):
    """Released version of the W1 robot body and arms."""

    V021 = "v021"
    V022 = "v022"
    V025 = "v025"


class DexforceW1HandVersion(enum.Enum):
    """Released version of an external W1 hand or gripper asset."""

    V021 = "v021"


class DexforceW1ArmSide(enum.Enum):
    """Arm side for DexforceW1: left or right."""

    LEFT = "left"
    RIGHT = "right"


class DexforceW1Type(enum.Enum):
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


class DexforceW1HandBrand(enum.Enum):
    BRAINCO_HAND = "BRAINCO_HAND"
    DH_PGC_GRIPPER = "DH_PGC_GRIPPER"
    DH_PGC_GRIPPER_M = "DH_PGC_GRIPPER_M"


def _parse_enum(value, enum_type, label):
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        for member in enum_type:
            if normalized in (member.name.lower(), str(member.value).lower()):
                return member
    raise ValueError(f"Invalid {label}: {value!r}")


def parse_w1_version(value) -> DexforceW1Version:
    """Parse a W1 robot version from an enum, member name, or value."""
    return _parse_enum(value, DexforceW1Version, "Dexforce W1 version")


def parse_w1_hand_version(value) -> DexforceW1HandVersion:
    """Parse a W1 hand version from an enum, member name, or value."""
    return _parse_enum(value, DexforceW1HandVersion, "Dexforce W1 hand version")


def parse_w1_arm_side(value) -> DexforceW1ArmSide:
    """Parse a W1 arm side from an enum, member name, or serialized value."""
    return _parse_enum(value, DexforceW1ArmSide, "Dexforce W1 arm side")


def parse_w1_hand_brand(value) -> DexforceW1HandBrand:
    """Parse a W1 hand brand from an enum, member name, or serialized value."""
    return _parse_enum(value, DexforceW1HandBrand, "Dexforce W1 hand brand")
