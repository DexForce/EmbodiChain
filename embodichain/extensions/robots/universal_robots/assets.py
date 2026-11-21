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

"""Asset source and cache identity for UniversalRobots."""

from __future__ import annotations

import os
import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["UniversalRobots"]

robot_assets = "robot_assets"


class UniversalRobots(EmbodiChainDataset):
    """Dataset class for the Universal Robots.

    V4 corrects the legacy UR3/UR3e material palette without changing robot
    geometry, collision meshes, kinematics or inertial properties.

    Reference:
        https://www.universal-robots.com/products/ur-series/

    Directory structure:
        UniversalRobots/
            UR3/UR3.urdf
            UR3e/UR3e.urdf
            UR5/UR5.urdf
            UR5e/UR5e.urdf
            UR10/UR10.urdf
            UR10e/UR10e.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import UniversalRobots
        >>> dataset = UniversalRobots()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("UniversalRobots/UR3/UR3.urdf"))
        >>> print(get_data_path("UniversalRobots/UR3e/UR3e.urdf"))
        >>> print(get_data_path("UniversalRobots/UR5/UR5.urdf"))
        >>> print(get_data_path("UniversalRobots/UR5e/UR5e.urdf"))
        >>> print(get_data_path("UniversalRobots/UR10/UR10.urdf"))
        >>> print(get_data_path("UniversalRobots/UR10e/UR10e.urdf"))
    """

    def __init__(self, data_root: str = None):
        filename = f"{robot_assets}/UniversalRobots_v4.zip"
        hub_prefix = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
        )
        urls = [f"{EMBODICHAIN_DOWNLOAD_PREFIX}{filename}"]
        if EMBODICHAIN_DOWNLOAD_PREFIX != hub_prefix:
            urls.append(f"{hub_prefix}{filename}")
        data_descriptor = o3d.data.DataDescriptor(
            urls, "a92eda28a1d2f9034252cdb2936ceffc"
        )
        # Assembly signatures include URDF paths, not referenced mesh contents.
        # A versioned source directory also invalidates old assembled materials.
        prefix = f"{type(self).__name__}V4"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
