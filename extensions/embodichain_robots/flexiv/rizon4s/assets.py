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

"""Asset source and cache identity for FlexivRizon4S."""

from __future__ import annotations

import os
import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)
from typing import ClassVar

__all__ = ["FlexivRizon4S"]

robot_assets = "robot_assets"


class FlexivRizon4S(EmbodiChainDataset):
    """Download the Apache-2.0 Flexiv Rizon 4S description from Hugging Face.

    The pinned official URDF, meshes, original parameter files, license and
    provenance share the standard locked download and extraction cache.
    The normal data registry resolves ``FlexivRizon4S/rizon4s.urdf``.

    Args:
        data_root: Optional root for the standard locked asset cache.
    """

    cache_prefix: ClassVar[str] = "FlexivRizon4S_v1"
    """Revision-specific extraction identity used by the data CLI."""

    def __init__(self, data_root: str | None = None) -> None:
        filename = f"{robot_assets}/FlexivRizon4S_v1.zip"
        hub_prefix = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
        )
        urls = [f"{EMBODICHAIN_DOWNLOAD_PREFIX}{filename}"]
        if EMBODICHAIN_DOWNLOAD_PREFIX != hub_prefix:
            urls.append(f"{hub_prefix}{filename}")
        descriptor = o3d.data.DataDescriptor(urls, "5464472433d57328e7711e2a94b46279")
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(self.cache_prefix, descriptor, root)
