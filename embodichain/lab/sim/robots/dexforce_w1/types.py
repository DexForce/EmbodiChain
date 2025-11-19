# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import enum

all = [
    "DexforceW1Version",
    "DexforceW1ArmKind",
    "DexforceW1ArmSide",
    "DexforceW1Type",
]


class DexforceW1Version(enum.Enum):
    """Versioning for DexforceW1 components."""

    V020 = "v020"


class DexforceW1ArmKind(enum.Enum):
    """Arm type for DexforceW1: anthropomorphic or industrial."""

    ANTHROPOMORPHIC = "anthropomorphic"
    INDUSTRIAL = "industrial"


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
    LEFT_ARM1 = "left_arm"  # Anthropomorphic left arm
    RIGHT_ARM1 = "right_arm"  # Anthropomorphic right arm
    LEFT_ARM2 = "left_arm2"  # Industrial left arm
    RIGHT_ARM2 = "right_arm2"  # Industrial right arm
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    FULL_BODY = "full_body"  # Full robot


class DexforceW1HandBrand(enum.Enum):
    BRAINCO_HAND = "BRAINCO_HAND"
    DH_PGC_GRIPPER = "DH_PGC_GRIPPER"
    DH_PGC_GRIPPER_M = "DH_PGC_GRIPPER_M"
