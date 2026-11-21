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

from __future__ import annotations

import os
from typing import ClassVar
import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = [
    "SceneData",
    "EmptyRoom",
    "WoodLab",
    "IndustrialWorkshop",
    "DemoStudio",
    "CleanLab",
    "RoboSplatterOfficeGS",
    "DiscoverseLab3GS",
    "RoboSimGSTabletopGS",
]

scene_assets = "scene_assets"

# Immutable Hub snapshot; V1 archives are additionally protected by their
# exact DataDescriptor checksums and version-specific extraction prefixes.
_GAUSSIAN_REVISION = "3ec5b1fb7d320599c54f07bc20d4c49d570a2e2b"


class _GaussianBackground(EmbodiChainDataset):
    """Shared locked transport for versioned visual-only Gaussian packages."""

    cache_prefix: ClassVar[str]
    archive_file: ClassVar[str]
    archive_md5: ClassVar[str]

    def __init__(self, data_root: str | None = None) -> None:
        """Download a complete USD-entry package under a version-specific cache.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        relative = "scene_assets/gaussian_backgrounds_v1/" + self.archive_file
        prefix = EMBODICHAIN_DOWNLOAD_PREFIX.replace(
            "/resolve/main/", f"/resolve/{_GAUSSIAN_REVISION}/", 1
        )
        urls = [prefix + relative]
        official = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/"
            + _GAUSSIAN_REVISION
            + "/"
            + relative
        )
        if official not in urls:
            urls.append(official)
        descriptor = o3d.data.DataDescriptor(urls, self.archive_md5)
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(self.cache_prefix, descriptor, root)


class RoboSplatterOfficeGS(_GaussianBackground):
    """RoboSplatter office scan with a metre/Z-up USD and AssetRevision entry.

    Resolve ``RoboSplatterOfficeGS/asset.usda`` through ``get_data_path``.
    The USD carries the source Y-up rotation; do not apply it a second time.
    This Apache-2.0 scan contains appearance only, with no collision geometry.
    """

    cache_prefix = "RoboSplatterOfficeGSV1"
    archive_file = "robosplatter_office_v1.zip"
    archive_md5 = "c36ff7550a47677479f22a21a8ee3c40"


class DiscoverseLab3GS(_GaussianBackground):
    """DISCOVERSE lab3 with preserved source and decoded Gaussian PLY bytes.

    Resolve ``DiscoverseLab3GS/asset.usda`` through ``get_data_path``.
    This is a visual-only metre/Z-up scene. Upstream model licensing is not
    separately declared; the package retains that status and its provenance.
    Packaging does not infer the project's software license for model data.
    """

    cache_prefix = "DiscoverseLab3GSV1"
    archive_file = "discoverse_lab3_v1.zip"
    archive_md5 = "ec9f17d4610e0dd5347ddb4815c531ad"


class RoboSimGSTabletopGS(_GaussianBackground):
    """RoboSimGS single-view tabletop with a visual-only USD/AssetRevision entry.

    Resolve ``RoboSimGSTabletopGS/asset.usda`` through ``get_data_path``.
    Coordinates are metres/Z-up with the captured tabletop near Z=0. Source
    bytes and the upstream dataset's MIT declaration are retained in the bundle.
    """

    cache_prefix = "RoboSimGSTabletopGSV1"
    archive_file = "robosimgs_tabletop_v1.zip"
    archive_md5 = "b6fc69397d64ccff611bff5712931b1a"


class SceneData(EmbodiChainDataset):
    """Dataset class for the Scene.

    Directory structure:
        SceneData/
            factory.glb
            kitchen.gltf
            office.glb

    Example usage:
        >>> from embodichain.data.assets.scene_assets import SceneData
        >>> data = SceneData()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Scenedata/factory.glb"))
    """

    def __init__(self, data_root: str = None):

        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, scene_assets, "SceneData.zip"),
            "fb46e4694cc88886fc785704e891a68a",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)


class EmptyRoom(o3d.data.DownloadDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, scene_assets, "empty_room.zip"),
            "612ffead4fac95114bec2e3812469f96",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class WoodLab(EmbodiChainDataset):
    """SimReady wood lab with GLB visuals and a static USD floor.

    Resolve the USD entry with
    ``get_data_path("WoodLab/wood_lab/wood_lab.usda")``.
    Preserve the bundle's relative visual and collision references.
    """

    def __init__(self, data_root: str | None = None) -> None:
        """Download and extract the room bundle under the shared cache lock.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        relative_path = "scene_assets/wood_lab.zip"
        urls = [EMBODICHAIN_DOWNLOAD_PREFIX + relative_path]
        hub_url = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
            + relative_path
        )
        if hub_url not in urls:
            urls.append(hub_url)
        descriptor = o3d.data.DataDescriptor(urls, "c549494006de4ca6d0f6cafd914df890")
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(type(self).__name__, descriptor, root)


class IndustrialWorkshop(EmbodiChainDataset):
    """SimReady industrial workshop with GLB visuals and a static USD floor.

    Resolve the USD entry with
    ``get_data_path("IndustrialWorkshop/industrial_workshop/industrial_workshop.usda")``.
    Preserve the bundle's relative visual and collision references.
    """

    def __init__(self, data_root: str | None = None) -> None:
        """Download and extract the room bundle under the shared cache lock.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        relative_path = "scene_assets/industrial_workshop.zip"
        urls = [EMBODICHAIN_DOWNLOAD_PREFIX + relative_path]
        hub_url = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
            + relative_path
        )
        if hub_url not in urls:
            urls.append(hub_url)
        descriptor = o3d.data.DataDescriptor(urls, "a74f3fc598af42f0cba2947fc37ec186")
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(type(self).__name__, descriptor, root)


class DemoStudio(EmbodiChainDataset):
    """SimReady demo studio with GLB visuals and a static USD floor.

    Resolve the USD entry with
    ``get_data_path("DemoStudio/demo_studio/demo_studio.usda")``.
    Preserve the bundle's relative visual and collision references.
    """

    def __init__(self, data_root: str | None = None) -> None:
        """Download and extract the room bundle under the shared cache lock.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        relative_path = "scene_assets/demo_studio.zip"
        urls = [EMBODICHAIN_DOWNLOAD_PREFIX + relative_path]
        hub_url = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
            + relative_path
        )
        if hub_url not in urls:
            urls.append(hub_url)
        descriptor = o3d.data.DataDescriptor(urls, "0e74b55b03f980bcac2c3e254e9c9188")
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(type(self).__name__, descriptor, root)


class CleanLab(EmbodiChainDataset):
    """SimReady clean lab with GLB visuals and a static USD floor.

    Resolve the USD entry with
    ``get_data_path("CleanLab/clean_lab/clean_lab.usda")``.
    Preserve the bundle's relative visual and collision references.
    """

    def __init__(self, data_root: str | None = None) -> None:
        """Download and extract the room bundle under the shared cache lock.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        relative_path = "scene_assets/clean_lab.zip"
        urls = [EMBODICHAIN_DOWNLOAD_PREFIX + relative_path]
        hub_url = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
            + relative_path
        )
        if hub_url not in urls:
            urls.append(hub_url)
        descriptor = o3d.data.DataDescriptor(urls, "3e0c7a820aa5a72bddf6292be85ee6a4")
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(type(self).__name__, descriptor, root)
