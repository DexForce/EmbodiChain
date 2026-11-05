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

"""Download the pinned official Wuji Hand descriptions and meshes."""

from __future__ import annotations

import open3d as o3d

from embodichain.data.constants import (
    EMBODICHAIN_DEFAULT_DATA_ROOT,
    EMBODICHAIN_DOWNLOAD_PREFIX,
)
from embodichain.data.dataset import EmbodiChainDataset

__all__ = ["WujiHand"]


class WujiHand(EmbodiChainDataset):
    """Left and right Wuji Hand assets with 20 active joints per hand.

    Resolve a URDF with
    ``get_data_path("WujiHand/urdf/right.urdf")``.
    The bundle also retains the upstream MJCF models, mesh dependencies,
    licensing and a source/checksum manifest. It provides no robot preset,
    controller, tactile simulation or trained manipulation policy.

    Args:
        data_root: Optional cache root; defaults to ``EMBODICHAIN_DATA_ROOT``.
    """

    cache_prefix = "WujiHandV1"

    def __init__(self, data_root: str | None = None) -> None:
        filename = "eef_assets/WujiHandV1.zip"
        hub_prefix = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
        )
        urls = [f"{EMBODICHAIN_DOWNLOAD_PREFIX}{filename}"]
        if EMBODICHAIN_DOWNLOAD_PREFIX != hub_prefix:
            urls.append(f"{hub_prefix}{filename}")
        descriptor = o3d.data.DataDescriptor(urls, "898dec6518c33b96cc5a769c19fd9c14")
        super().__init__(
            self.cache_prefix,
            descriptor,
            EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root,
        )
