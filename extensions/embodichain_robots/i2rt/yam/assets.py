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

"""Versioned YAM robot assets and pinned source attribution."""

from __future__ import annotations

import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["YamI2rt"]

robot_assets = "robot_assets"


class YamI2rt(EmbodiChainDataset):
    """Download the versioned YAM i2rt dual-arm robot and its collision meshes.

    The archive retains the I2RT Robotics MIT license, source provenance and
    attribution notice. It contains the URDF used by both simulation and FK/IK;
    ordinary downloads do not require the mesh-conversion dependencies.

    Example:
        >>> from embodichain.data import get_data_path
        >>> path = get_data_path("YamI2rt/yam_i2rt.urdf")
    """

    archive_md5: str = "23601ea820ff3e6260336c4962ef01a5"

    def __init__(self, data_root: str | None = None) -> None:
        """Download and extract the checked archive into the selected data root.

        Args:
            data_root: Cache root, or the configured default when omitted.
        """
        filename = f"{robot_assets}/YamI2rtV1.zip"
        urls = [f"{EMBODICHAIN_DOWNLOAD_PREFIX}{filename}"]
        hub_url = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/"
            f"resolve/main/{filename}"
        )
        if hub_url not in urls:
            urls.append(hub_url)
        descriptor = o3d.data.DataDescriptor(urls, self.archive_md5)
        super().__init__(
            type(self).__name__,
            descriptor,
            EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root,
        )
