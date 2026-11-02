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


"""Model and terrain archives for dexterous manipulation and rough locomotion."""

from __future__ import annotations

import open3d as o3d

from embodichain.data.constants import (
    EMBODICHAIN_DEFAULT_DATA_ROOT,
    EMBODICHAIN_DOWNLOAD_PREFIX,
)
from embodichain.data.dataset import EmbodiChainDataset

__all__ = [
    "UnitreeG1Rough",
    "G1RoughTerrains",
    "ShadowHandPen",
    "LeapHandCubeFaceTurn",
    "LeapHandCube",
]


class _RLAsset(EmbodiChainDataset):
    """Download an archive through the standard asset cache."""

    category: str = "robot_assets"
    archive_md5: str

    def __init__(self, data_root: str | None = None) -> None:
        name = type(self).__name__
        descriptor = o3d.data.DataDescriptor(
            f"{EMBODICHAIN_DOWNLOAD_PREFIX}{self.category}/{name}.zip", self.archive_md5
        )
        super().__init__(
            name,
            descriptor,
            EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root,
        )


class UnitreeG1Rough(_RLAsset):
    """Bundled model inputs for UnitreeG1Rough."""

    archive_md5 = "b2ff7af5456458a38f96c03dc4ae39db"


class G1RoughTerrains(_RLAsset):
    """Bundled model inputs for G1RoughTerrains."""

    category = "scene_assets"
    archive_md5 = "3799830bd532aa7047708f749fe462c3"


class ShadowHandPen(_RLAsset):
    """Bundled model inputs for ShadowHandPen."""

    archive_md5 = "d5642a9fb20bf49be918455826c35a97"


class LeapHandCubeFaceTurn(_RLAsset):
    """Bundled model inputs for LeapHandCubeFaceTurn."""

    archive_md5 = "6394cd7665963f3fdcacbcf170a9af79"


class LeapHandCube(_RLAsset):
    """Bundled model inputs for LeapHandCube."""

    archive_md5 = "1bf309614884c706e225d7cd5165b949"


_ASSET_CATALOG: dict[str, dict[str, str]] = {
    name: {
        "class_path": f"{__name__}:{name}",
        "category": "scene" if name == "G1RoughTerrains" else "robot",
    }
    for name in __all__
}
