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

"""Download the existing DH_PGC_140_50, DH_PGC_140_50_M asset archives."""

from __future__ import annotations

import os

import open3d as o3d

from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["DH_PGC_140_50", "DH_PGC_140_50_M"]

_EEF_ASSET_DIRECTORY = "eef_assets"


class DH_PGC_140_50(EmbodiChainDataset):
    """Dataset class for the DH Robotics PGC-140-50 end-effector gripper.

    Reference:
        https://www.dh-robotics.com/product/pgc

    Directory structure:
        DH_PGC_140_50/
            DH_PGC_140_50.urdf

    Example usage:
        >>> from embodichain_robots.dh_robotics.pgc_140_50.assets import DH_PGC_140_50
        >>> dataset = DH_PGC_140_50()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DH_PGC_140_50/DH_PGC_140_50.urdf"))

    Args:
        data_root: Optional cache root; defaults to ``EMBODICHAIN_DATA_ROOT``.
    """

    def __init__(self, data_root: str | None = None) -> None:
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, _EEF_ASSET_DIRECTORY, "DH_PGC_140_50.zip"
            ),
            "c2a642308a76e99b1b8b7cb3a11c5df3",
        )
        prefix = type(self).__name__
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
        >>> from embodichain_robots.dh_robotics.pgc_140_50.assets import DH_PGC_140_50_M
        >>> dataset = DH_PGC_140_50_M()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf"))

    Args:
        data_root: Optional cache root; defaults to ``EMBODICHAIN_DATA_ROOT``.
    """

    def __init__(self, data_root: str | None = None) -> None:
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, _EEF_ASSET_DIRECTORY, "DH_PGC_140_50_M.zip"
            ),
            "3a9ab5f32639e03afb38dc033b44bb62",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
