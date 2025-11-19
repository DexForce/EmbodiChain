# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------


import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)


class RealSense_D405(EmbodiChainDataset):
    """Dataset class for the RealSense D405 camera.

    Reference:
        https://www.intel.com/content/www/us/en/developer/tools/oneapi/real-sense-d405.html
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "RealSense_D405.zip",
            "71a20b8e509ba32504a9215633a76b69",
        )
        prefix = "RealSense_D405"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
