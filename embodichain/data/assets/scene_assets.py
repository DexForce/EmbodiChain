# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import os
import open3d as o3d
from pathlib import Path
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)


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
        prefix = "SceneData"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)


class EmptyRoom(o3d.data.DownloadDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, scene_assets, "empty_room.zip"),
            "612ffead4fac95114bec2e3812469f96",
        )
        prefix = "EmptyRoom"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
