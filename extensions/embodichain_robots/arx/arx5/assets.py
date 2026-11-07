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

"""Asset source and cache identity for ARX5."""

from __future__ import annotations

import os
import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["ARX5"]

robot_assets = "robot_assets"


class ARX5(EmbodiChainDataset):
    """Dataset class for the ARX5 robot.

    Reference:
        https://arx-x.com/

    Directory structure:
        ARX5/
            X5A.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import ARX5
        >>> dataset = ARX5()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("ARX5/X5A.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "ARX5_2.zip"),
            "6fc1d6a5a35b7926ba6325c0d20f5bc3",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
