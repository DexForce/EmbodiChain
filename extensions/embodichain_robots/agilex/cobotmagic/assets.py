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

"""Asset source and cache identity for CobotMagicArm."""

from __future__ import annotations

import os
import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["CobotMagicArm"]

robot_assets = "robot_assets"


class CobotMagicArm(EmbodiChainDataset):
    """Dataset class for the Cobot Magic Arm robot.

    Reference:
        https://global.agilex.ai/products/cobot-magic

    Directory structure:
        CobotMagicArm/
            CobotMagicNoGripper.urdf
            CobotMagicWithGripperV70.urdf
            CobotMagicWithGripperV70NewUV.urdf
            CobotMagicWithGripperV70NoMaterial.urdf
            CobotMagicWithGripperV100.urdf
            CobotMagicWithGripperV100NewUV.urdf
            CobotMagicWithGripperV100NoMaterial.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import CobotMagicArm
        >>> dataset = CobotMagicArm()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("CobotMagicArm/CobotMagicWithGripperV100.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "CobotMagicArmV4.zip"
            ),
            "8cc54c240c2f26e84e22250c8364b0ec",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
