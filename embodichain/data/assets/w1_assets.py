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

import os
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
#   - DexforceW1V021:
#       Represents the complete humanoid robot asset,
#       including both industrial arms and anthropomorphic arms.
#
# Component Assets:
#   - DexforceW1ChassisV021:   Chassis component
#   - DexforceW1TorsoV021:     Torso component
#   - DexforceW1EyesV021:      Eyes component
#   - DexforceW1HeadV021:      Head component
#
# Arm Assets:
#   - DexforceW1LeftArm1V021 / DexforceW1RightArm1V021:
#       Anthropomorphic (human-like) arms, left and right.
#   - DexforceW1LeftArm2V021 / DexforceW1RightArm2V021:
#       Industrial arms, left and right.
#
# All classes inherit from EmbodiChainDataset and are responsible for
# downloading and managing the data resources for their respective components.
# ======================================================================


w1_assets = "dexforce_w1"


class DexforceW1V021(EmbodiChainDataset):
    """Dataset class for the Dexforce W1 V021.

    Directory structure:
        DexforceW1V021/DexforceW1V021.urdf

    Example usage:
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf"))
        >>> print(get_data_path("DexforceW1V021/DexforceW1_v02_2.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, w1_assets, "DexforceW1V021.zip"),
            "3cc3a0bfd1c50ebed5bee9dadeee6756",
        )
        prefix = "DexforceW1V021"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1V021_INDUSTRIAL_DH_PGC_GRIPPER_M(EmbodiChainDataset):
    """Dataset class for the industrial Dexforce W1 V021 with DH_PGC_gripper.

    Directory structure:
        DexforceW1V021_INDUSTRIAL_DH_PGC_GRIPPER_M/DexforceW1V021.urdf

    Example usage:
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DexforceW1V021_INDUSTRIAL_DH_PGC_GRIPPER_M/DexforceW1V021.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX,
                w1_assets,
                "DexforceW1V021_INDUSTRIAL_DH_PGC_GRIPPER_M.zip",
            ),
            "06ec5dfa76dc69160d7ff9bc537a6a7b",
        )
        prefix = "DexforceW1V021_INDUSTRIAL_DH_PGC_GRIPPER_M"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1V021_ANTHROPOMORPHIC_BRAINCO_HAND_REVO1(EmbodiChainDataset):
    """Dataset class for the anthropomorphic Dexforce W1 V021 with BrainCo_hand_revo_1.

    Directory structure:
        DexforceW1V021_ANTHROPOMORPHIC_BRAINCO_HAND_REVO1/DexforceW1V021.urdf

    Example usage:
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DexforceW1V021_ANTHROPOMORPHIC_BRAINCO_HAND_REVO1/DexforceW1V021.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX,
                w1_assets,
                "DexforceW1V021_ANTHROPOMORPHIC_BRAINCO_HAND_REVO1.zip",
            ),
            "ef19d247799e79233863b558c47b32cd",
        )
        prefix = "DexforceW1V021_ANTHROPOMORPHIC_BRAINCO_HAND_REVO1"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1ChassisV021(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, w1_assets, "W1_Chassis_v021.zip"),
            "6b0517a4d92a572988641d46269d063f",
        )
        prefix = "DexforceW1ChassisV021"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1TorsoV021(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, w1_assets, "W1_Torso_v021.zip"),
            "4f762a3ae6ef2acbe484c915cf80da7b",
        )
        prefix = "DexforceW1TorsoV021"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1EyesV021(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, w1_assets, "W1_Eyes_v021.zip"),
            "80e0b86ef2e934f439c99b79074f6f3c",
        )
        prefix = "DexforceW1EyesV021"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1HeadV021(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, w1_assets, "W1_Head_v021.zip"),
            "ba72805828c5fd62ad55d6a1458893d0",
        )
        prefix = "DexforceW1HeadV021"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1LeftArm1V021(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, w1_assets, "W1_LeftArm_1_v021.zip"
            ),
            "c3cacda7bd36389ed98620047bff6216",
        )
        prefix = "DexforceW1LeftArm1V021"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1RightArm1V021(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, w1_assets, "W1_RightArm_1_v021.zip"
            ),
            "456c9495748171003246a3f6626bb0db",
        )
        prefix = "DexforceW1RightArm2V021"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1LeftArm2V021(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, w1_assets, "W1_LeftArm_2_v021.zip"
            ),
            "b99bd0587cc9a36fed3cdaa4f9fd62e7",
        )
        prefix = "DexforceW1LeftArm2V021"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexforceW1RightArm2V021(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, w1_assets, "W1_RightArm_2_v021.zip"
            ),
            "d9f25b2d5244ca5a859040327273a99e",
        )
        prefix = "DexforceW1RightArm1V021"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
