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

"""Download the existing BrainCoHandRevo1 asset archives."""

from __future__ import annotations

import os

import open3d as o3d

from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["BrainCoHandRevo1"]

_EEF_ASSET_DIRECTORY = "eef_assets"


class BrainCoHandRevo1(EmbodiChainDataset):
    """Dataset class for the BrainCo Hand Revo 1 robotic hand.

    Reference:
        https://www.brainco-hz.com/docs/revolimb-hand/revo1/parameters.html

    Directory structure:
        BrainCoHandRevo1/
            BrainCoRightHand/BrainCoRightHand.urdf
            BrainCoLeftHand/BrainCoLeftHand.urdf

    Example usage:
        >>> from embodichain_robots.brainco.revo1.assets import BrainCoHandRevo1
        >>> dataset = BrainCoHandRevo1()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("BrainCoHandRevo1/BrainCoRightHand/BrainCoRightHand.urdf"))
        >>> print(get_data_path("BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf"))

    Args:
        data_root: Optional cache root; defaults to ``EMBODICHAIN_DATA_ROOT``.
    """

    def __init__(self, data_root: str | None = None) -> None:
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX,
                _EEF_ASSET_DIRECTORY,
                "BrainCoHandRevo01.zip",
            ),
            "ff9ac77e7e1493fd32d40c87fecbee6c",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
