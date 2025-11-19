# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import open3d as o3d
from pathlib import Path
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)


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
            EMBODICHAIN_DOWNLOAD_PREFIX + "SceneData.zip",
            "fb46e4694cc88886fc785704e891a68a",
        )
        prefix = "SceneData"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)
