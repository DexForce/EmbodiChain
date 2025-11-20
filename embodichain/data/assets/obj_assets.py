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


import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)


class ShopTableSimple(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "shop_table_simple.zip",
            "e3061ee024de7840f773b70140dcd43f",
        )
        prefix = "ShopTableSimple"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class CircleTableSimple(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "circle_table_simple.zip",
            "42ad2be8cd0caddcf9bfbf106b7783f3",
        )
        prefix = "CircleTableSimple"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class PlasticBin(o3d.data.DownloadDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "plastic_bin.glb",
            "6ae6eb88ef9540e03e45e8f774a84689",
        )
        prefix = "PlasticBin"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class EmptyRoom(o3d.data.DownloadDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "empty_room.glb",
            "3281ab5d803546835fc4ece01c22d8f7",
        )
        prefix = "EmptyRoom"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Chair(o3d.data.DownloadDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "chair.glb",
            "a86285f6a2a520b0b61f34d1958e6757",
        )
        prefix = "Chair"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Shelf(o3d.data.DownloadDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "shelf.glb",
            "1f659411082e6e7353cda62fc78e8c20",
        )
        prefix = "Shelf"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Sofa(o3d.data.DownloadDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "sofa.glb", "76fd7285a6f7335e9c7162c6eaf151eb"
        )
        prefix = "Sofa"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class ContainerMetal(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "container_metal.zip",
            "ceafb87f8177609f87aaa6779fcbb9a3",
        )
        prefix = "ContainerMetal"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class SimpleBoxDrawer(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "simple_box_drawer.zip",
            "966b648bca16823ee91525847c183973",
        )
        prefix = "SimpleBoxDrawer"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class AmazonPrimeBox(o3d.data.DownloadDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "amazon_prime_shipping_box.glb",
            "e3e75d131479b780ef8d7b68732cfcfa",
        )
        prefix = "AmazonPrimeBox"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class AdrianoTable(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "adriano_table.zip",
            "8453583a9a1a9d04d50268f8a3da554f",
        )
        prefix = "AdrianoTable"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class CoffeeCup(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "CoffeeCup.zip",
            "f05fce385826414c15e19df3b75dc886",
        )
        prefix = "CoffeeCup"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class SlidingBoxDrawer(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "SlidingBoxDrawer.zip",
            "b03d9006503d27b75ddeb06d31b2c7a5",
        )
        prefix = "SlidingBoxDrawer"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class AiLiMu_BoxDrawer(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "AiLiMu_BoxDrawer_v3.zip",
            "9a2889151a23d482f95f602cce9900c6",
        )
        prefix = "AiLiMu_BoxDrawer"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class AluminumTable(o3d.data.DownloadDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "AluminumTable.glb",
            "02991d36ca9b70f019ed330a61143aa9",
        )
        prefix = "AluminumTable"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class ToyDuck(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "ToyDuck.zip",
            "2f5c00ba487edf34ad668f7257c0264e",
        )
        prefix = "ToyDuck"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class PaperCup(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "PaperCup.zip",
            "359d13af8c5f31ad3226d8994a1a7198",
        )
        prefix = "PaperCup"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
