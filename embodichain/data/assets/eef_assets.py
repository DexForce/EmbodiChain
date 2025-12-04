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


eef_assets = "eef_assets"


class DH_PGC_140_50(EmbodiChainDataset):
    """Dataset class for the DH Robotics PGC-140-50 end-effector gripper.

    Reference:
        https://www.dh-robotics.com/product/pgc

    Directory structure:
        DH_PGC_140_50/
            DH_PGC_140_50.urdf

    Example usage:
        >>> from embodichain.data.eef_dataset import DH_PGC_140_50
        >>> dataset = DH_PGC_140_50()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DH_PGC_140_50/DH_PGC_140_50.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, eef_assets, "DH_PGC_140_50.zip"),
            "c2a642308a76e99b1b8b7cb3a11c5df3",
        )
        prefix = "DH_PGC_140_50"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DH_PGI_140_80(EmbodiChainDataset):
    """Dataset class for the DH Robotics PGI-140-80 end-effector gripper.

    Reference:
        https://www.dh-robotics.com/product/pgia###

    Directory structure:
        DH_PGI_140_80/
            DH_PGI_140_80.urdf

    Example usage:
        >>> from embodichain.data.eef_dataset import DH_PGI_140_80
        >>> dataset = DH_PGI_140_80()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DH_PGI_140_80/DH_PGI_140_80.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, eef_assets, "DH_PGI_140_80.zip"),
            "05a1a08b13c6250cc12affeeda3a08ba",
        )
        prefix = "DH_PGI_140_80"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DH_PGC_140_50_M(EmbodiChainDataset):
    """Dataset class for the DH Robotics PGC-140-50 end-effector gripper.
       DexForce modified connector and finger.

    Reference:
        https://www.dh-robotics.com/product/pgc

    Directory structure:
        DH_PGC_140_50_M/
            DH_PGC_140_50_M.urdf

    Example usage:
        >>> from embodichain.data.eef_dataset import DH_PGC_140_50_M
        >>> dataset = DH_PGC_140_50_M()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, eef_assets, "DH_PGC_140_50_M.zip"
            ),
            "3a9ab5f32639e03afb38dc033b44bb62",
        )
        prefix = "DH_PGC_140_50_M"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class ZH_CTM2F110(EmbodiChainDataset):
    """Dataset class for the Zhixing Robot Technology CTM2F110 end-effector gripper.

    Reference:
        https://www.changingtek.com/service

    Directory structure:
        ZH_CTM2F110/
            ZH_CTM2F110.urdf

    Example usage:
        >>> from embodichain.data.eef_dataset import ZH_CTM2F110
        >>> dataset = ZH_CTM2F110()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("ZH_CTM2F110/ZH_CTM2F110.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, eef_assets, "ZH_CTM2F110.zip"),
            "0e7c3310425609797fe010b2a76fe465",
        )
        prefix = "ZH_CTM2F110"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class BrainCoHandRevo1(EmbodiChainDataset):
    """Dataset class for the BrainCo Hand Revo 1 robotic hand.

    Reference:
        https://www.brainco-hz.com/docs/revolimb-hand/revo1/parameters.html

    Directory structure:
        BrainCoHandRevo1/
            BrainCoRightHand/BrainCoRightHand.urdf
            BrainCoLeftHand/BrainCoLeftHand.urdf

    Example usage:
        >>> from embodichain.data.eef_dataset import BrainCoHandRevo1
        >>> dataset = BrainCoHandRevo1()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("BrainCoHandRevo1/BrainCoRightHand/BrainCoRightHand.urdf"))
        >>> print(get_data_path("BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, eef_assets, "BrainCoHandRevo01.zip"
            ),
            "ff9ac77e7e1493fd32d40c87fecbee6c",
        )
        prefix = "BrainCoHandRevo1"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class InspireHand(EmbodiChainDataset):
    """Dataset class for the Inspire Hand robotic hand.

    Reference:
        https://en.inspire-robots.com/product/rh56bfx

    Directory structure:
        InspireHand/
            InspireLeftHand/InspireLeftHand.urdf
            InspireRightHand/InspireRightHand.urdf
            inspire_joint_data.csv
            inspire_joint_data.npy

    Example usage:
        >>> from embodichain.data.eef_dataset import InspireHand
        >>> dataset = InspireHand()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("InspireHand/InspireLeftHand/InspireLeftHand.urdf"))
        >>> print(get_data_path("InspireHand/InspireRightHand/InspireRightHand.urdf"))
        >>> print(get_data_path("InspireHand/inspire_joint_data.csv"))
        >>> print(get_data_path("InspireHand/inspire_joint_data.npy"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, eef_assets, "InspireHand.zip"),
            "c60132a6f03866fb021cca5b6d72845e",
        )
        prefix = "InspireHand"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Robotiq2F85(EmbodiChainDataset):
    """Dataset class for the Robotiq 2F85 robotic gripper.

    Reference:
        https://robotiq.com/products/adaptive-grippers#Two-Finger-Gripper

    Directory structure:
        Robotiq2F85/
            Robotiq2F85.urdf

    Example usage:
        >>> from embodichain.data.eef_dataset import Robotiq2F85
        >>> dataset = Robotiq2F85()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Robotiq2F85/Robotiq2F85.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, eef_assets, "Robotiq2F85.zip"),
            "53ecbf2c953f43f1134aa7223e592292",
        )
        prefix = "Robotiq2F85"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class WheelTecFA2F(EmbodiChainDataset):
    """Dataset class for the WheelTec FA 2 fingers robotic gripper.

    Reference:
        https://www.wheeltec.net/

    Directory structure:
        WheelTecFA2F/
            WheelTecFA2F.urdf

    Example usage:
        >>> from embodichain.data.eef_dataset import WheelTecFA2F
        >>> dataset = WheelTecFA2F()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("WheelTecFA2F/WheelTecFA2F.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, eef_assets, "WheelTecFA2F.zip"),
            "feaf13f25b1c6ce58d011b1f2fa72f58",
        )
        prefix = "WheelTecFA2F"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
