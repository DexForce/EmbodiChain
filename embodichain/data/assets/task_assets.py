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

"""Downloadable task meshes and reference demonstration data."""

from __future__ import annotations

import open3d as o3d

from embodichain.data.constants import (
    EMBODICHAIN_DEFAULT_DATA_ROOT,
    EMBODICHAIN_DOWNLOAD_PREFIX,
)
from embodichain.data.dataset import EmbodiChainDataset

__all__ = ["PourCupAssets", "PourCupDemoData"]

_HUB_PREFIX = (
    "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
)


class _TaskAsset(EmbodiChainDataset):
    """Download one task archive through the shared locked dataset cache."""

    archive_md5: str

    def __init__(self, data_root: str | None = None) -> None:
        """Initialize and extract the task bundle.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        prefix = type(self).__name__
        filename = f"task_assets/{prefix}.zip"
        urls = [f"{EMBODICHAIN_DOWNLOAD_PREFIX}{filename}"]
        if EMBODICHAIN_DOWNLOAD_PREFIX != _HUB_PREFIX:
            urls.append(f"{_HUB_PREFIX}{filename}")
        descriptor = o3d.data.DataDescriptor(urls, self.archive_md5)
        super().__init__(
            prefix,
            descriptor,
            EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root,
        )


class PourCupAssets(_TaskAsset):
    """Cup, fluid boundary, receiver, table and floor meshes for cup pouring.

    Resolve individual meshes with ``get_data_path("PourCupAssets/cup.glb")``.
    The bundle contains no initial-fluid particle file.
    """

    archive_md5 = "f97857b639189be4daf717a873ab84cf"


class PourCupDemoData(_TaskAsset):
    """Recorded pouring trajectory, videos and original validation evidence.

    ``trajectory.pt`` is a historical MuJoCo recording for diagnostic replay.
    Online pouring tutorials and tasks do not load it. ``particles_30.npz`` is a final reference snapshot and is not used
    to initialize fluids. Resolve files through ``get_data_path()``.
    """

    archive_md5 = "8eb96cb756ac08efe403417f50ad2bf2"
