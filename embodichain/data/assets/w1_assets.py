# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

# ================= Dexforce W1 Asset Dataset Overview =================
# This file provides dataset classes for the Dexforce W1 humanoid robot
# and its individual components.
#
# Main Asset:
#   - DexforceW1V020:
#       Represents the complete humanoid robot asset,
#       including both industrial arms and anthropomorphic arms.
#
# Component Assets:
#   - DexforceW1ChassisV020:   Chassis component
#   - DexforceW1TorsoV020:     Torso component
#   - DexforceW1EyesV020:      Eyes component
#   - DexforceW1HeadV020:      Head component
#
# Arm Assets:
#   - DexforceW1LeftArm1V020 / DexforceW1RightArm1V020:
#       Anthropomorphic (human-like) arms, left and right.
#   - DexforceW1LeftArm2V020 / DexforceW1RightArm2V020:
#       Industrial arms, left and right.
#
# All classes inherit from EmbodiChainDataset and are responsible for
# downloading and managing the data resources for their respective components.
# ======================================================================


class DexforceW1V020(EmbodiChainDataset):
    """Dataset class for the Dexforce W1 V020.

    Directory structure:
        DexforceW1V020/DexforceW1V020.urdf

    Example usage:
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DexforceW1V020/DexforceW1V020.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "W1_v020.zip",
            "776e0ae90c4de5d58464d4259b464843",
        )
        prefix = "DexforceW1V020"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1V020_INDUSTRIAL_DH_PGC_GRIPPER_M(EmbodiChainDataset):
    """Dataset class for the industrial Dexforce W1 V020 with DH_PGC_gripper.

    Directory structure:
        DexforceW1V020_INDUSTRIAL_DH_PGC_GRIPPER_M/DexforceW1V020.urdf

    Example usage:
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DexforceW1V020_INDUSTRIAL_DH_PGC_GRIPPER_M/DexforceW1V020.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX
            + "DexforceW1V020_INDUSTRIAL_DH_PGC_GRIPPER_M.zip",
            "99c7cbddd5c6de142b390f6eb0df6dec",
        )
        prefix = "DexforceW1V020_INDUSTRIAL_DH_PGC_GRIPPER_M"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1V020_ANTHROPOMORPHIC_BRAINCO_HAND(EmbodiChainDataset):
    """Dataset class for the anthropomorphic Dexforce W1 V020 with BrainCo_hand.

    Directory structure:
        DexforceW1V020_ANTHROPOMORPHIC_BRAINCO_HAND/DexforceW1V020.urdf

    Example usage:
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DexforceW1V020_ANTHROPOMORPHIC_BRAINCO_HAND/DexforceW1V020.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX
            + "DexforceW1V020_ANTHROPOMORPHIC_BRAINCO_HAND.zip",
            "2006d060a81a171a4e7a09fc2d013304",
        )
        prefix = "DexforceW1V020_ANTHROPOMORPHIC_BRAINCO_HAND"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1ChassisV020(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "W1_Chassis_v020.zip",
            "efd2a3cef43cb1f37ebc3a776e3bc6e7",
        )
        prefix = "DexforceW1ChassisV020"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1TorsoV020(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "W1_Torso_v020.zip",
            "4f762a3ae6ef2acbe484c915cf80da7b",
        )
        prefix = "DexforceW1TorsoV020"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1EyesV020(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "W1_Eyes_v020.zip",
            "80e0b86ef2e934f439c99b79074f6f3c",
        )
        prefix = "DexforceW1EyesV020"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1HeadV020(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "W1_Head_v020.zip",
            "17d571ff010387078674b5298d6e723f",
        )
        prefix = "DexforceW1HeadV020"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1LeftArm1V020(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "W1_LeftArm_1_v020.zip",
            "c3cacda7bd36389ed98620047bff6216",
        )
        prefix = "DexforceW1LeftArm1V020"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1RightArm1V020(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "W1_RightArm_1_v020.zip",
            "456c9495748171003246a3f6626bb0db",
        )
        prefix = "DexforceW1RightArm2V020"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1LeftArm2V020(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "W1_LeftArm_2_v020.zip",
            "b99bd0587cc9a36fed3cdaa4f9fd62e7",
        )
        prefix = "DexforceW1LeftArm2V020"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1RightArm2V020(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "W1_RightArm_2_v020.zip",
            "d9f25b2d5244ca5a859040327273a99e",
        )
        prefix = "DexforceW1RightArm1V020"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
