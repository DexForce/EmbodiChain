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

"""Asset source and cache identity for Franka."""

from __future__ import annotations

import os
import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["Franka"]

robot_assets = "robot_assets"


class Franka(EmbodiChainDataset):
    """Dataset class for the Franka robots.

    Reference:
        https://franka.de/franka-research-3

    Directory structure:
        Franka/
            Panda/
                Panda.urdf
                PandaWithHand.urdf
            FR3/
                fr3.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import Franka
        >>> dataset = Franka()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Franka/Panda/Panda.urdf"))
        >>> print(get_data_path("Franka/PandaWithHand/PandaWithHand.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "FrankaV4.zip"),
            "0c613d884798e8b084604381641cc8e1",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
