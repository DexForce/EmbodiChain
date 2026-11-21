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

"""Download the existing PikaGripper asset archives."""

from __future__ import annotations

import os

import open3d as o3d

from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["PikaGripper"]

_EEF_ASSET_DIRECTORY = "eef_assets"


class PikaGripper(EmbodiChainDataset):
    """Standalone Pika parallel gripper with visual and collision meshes.

    Resolve the URDF with
    ``get_data_path("PikaGripper/PikaGripper.urdf")``. ``FINGER1_JOINT`` is
    actuated over 0–0.05 m (closed to open); ``FINGER2_JOINT`` mimics it.
    Mount the asset through ``GRIPPER_ROOT_LINK``.

    Args:
        data_root: Optional cache root; defaults to ``EMBODICHAIN_DATA_ROOT``.
    """

    def __init__(self, data_root: str | None = None) -> None:
        filename = f"{_EEF_ASSET_DIRECTORY}/PikaGripper.zip"
        hub_prefix = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
        )
        urls = [f"{EMBODICHAIN_DOWNLOAD_PREFIX}{filename}"]
        # New assets may precede mirror synchronization.
        if EMBODICHAIN_DOWNLOAD_PREFIX != hub_prefix:
            urls.append(f"{hub_prefix}{filename}")
        descriptor = o3d.data.DataDescriptor(urls, "8a9d0779e7dbfa7bf5d4713175df014e")
        super().__init__(
            type(self).__name__,
            descriptor,
            EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root,
        )
