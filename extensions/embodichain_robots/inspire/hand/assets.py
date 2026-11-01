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

"""Download the existing InspireHand asset archives."""

from __future__ import annotations

import os

import open3d as o3d

from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["InspireHand"]

_EEF_ASSET_DIRECTORY = "eef_assets"


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
        >>> from embodichain_robots.inspire.hand.assets import InspireHand
        >>> dataset = InspireHand()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("InspireHand/InspireLeftHand/InspireLeftHand.urdf"))
        >>> print(get_data_path("InspireHand/InspireRightHand/InspireRightHand.urdf"))
        >>> print(get_data_path("InspireHand/inspire_joint_data.csv"))
        >>> print(get_data_path("InspireHand/inspire_joint_data.npy"))

    Args:
        data_root: Optional cache root; defaults to ``EMBODICHAIN_DATA_ROOT``.
    """

    def __init__(self, data_root: str | None = None) -> None:
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, _EEF_ASSET_DIRECTORY, "InspireHand.zip"
            ),
            "c60132a6f03866fb021cca5b6d72845e",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
