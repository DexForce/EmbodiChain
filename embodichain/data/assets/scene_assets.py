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
]

scene_assets = "scene_assets"


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
