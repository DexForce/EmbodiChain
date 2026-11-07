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

"""Download the existing DH_AG95 asset archives."""

from __future__ import annotations

import os

import open3d as o3d

from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["DH_AG95"]

_EEF_ASSET_DIRECTORY = "eef_assets"


class DH_AG95(EmbodiChainDataset):
    """Dataset class for the DH Robotics AG95 end-effector gripper.
       DexForce modified connector and finger.

    Reference:
        https://www.dh-robotics.com/product/ag

    Directory structure:
        DH_AG95/
            dh_ag95.urdf

    Example usage:
        >>> from embodichain_robots.dh_robotics.ag95.assets import DH_AG95
        >>> dataset = DH_AG95()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DH_AG95/dh_ag95.urdf"))

    Args:
        data_root: Optional cache root; defaults to ``EMBODICHAIN_DATA_ROOT``.
    """

    def __init__(self, data_root: str | None = None) -> None:
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, _EEF_ASSET_DIRECTORY, "DH_AG95.zip"
            ),
            "34b6f3c2f649697ea7f12814b6a50529",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
