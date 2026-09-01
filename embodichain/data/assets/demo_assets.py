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

"""Registered downloadable asset bundles for standalone demos."""

from __future__ import annotations

import open3d as o3d
import os

from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

demo_assets = "demo"

__all__ = [
    "CoordinatedPlacementAndPickment",
    "DeformableDemoData",
    "MultiW1Data",
    "ScoopIceNewEnv",
]


class ScoopIceNewEnv(EmbodiChainDataset):
    """Downloadable meshes and robot assets for the scoop-ice demo."""

    def __init__(self, data_root: str | None = None) -> None:
        """Initialize the scoop-ice asset bundle.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, demo_assets, "ScoopIceNewEnv.zip"
            ),
            "e92734a9de0f64be33a11fbda0fbd3b6",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class MultiW1Data(EmbodiChainDataset):
    """Downloadable scene assets for multi-W1 manipulation demos."""

    def __init__(self, data_root: str | None = None) -> None:
        """Initialize the multi-W1 demo asset bundle.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, demo_assets, "multi_w1_demo.zip"),
            "984e8fa3aa05cb36a1fd973a475183ed",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)


class DeformableDemoData(EmbodiChainDataset):
    """Shared cloth-twist and W1 T-shirt-folding demo assets."""

    def __init__(self, data_root: str | None = None) -> None:
        """Initialize the downloadable deformable-demo bundle.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX,
                demo_assets,
                "deformable_demo_assets.zip",
            ),
            "cdb1d1b105f0e96f46945052296da4d3",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)


class CoordinatedPlacementAndPickment(EmbodiChainDataset):
    """Downloadable meshes for coordinated placement and pickment tutorials."""

    def __init__(self, data_root: str | None = None) -> None:
        """Initialize the coordinated manipulation asset bundle.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX,
                demo_assets,
                "coordinated_placement_and_pickment.zip",
            ),
            "297c10b386a4d7a8ccb68926d69425e9",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)
