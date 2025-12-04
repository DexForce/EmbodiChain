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
from typing import List

from embodichain.data.dataset import EmbodiChainDataset
from embodichain.utils import logger
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)


material_assets = "materials"


class SimResources(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, material_assets, "embodisim_resources.zip"
            ),
            "53c054b3ae0857416dc52632eb562c12",
        )
        prefix = "SimResources"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)

    def get_ibl_path(self, name: str) -> str:
        """Get the path of the IBL resource.

        Args:
            name (str): The name of the IBL resource.

        Returns:
            str: The path to the IBL resource.
        """
        ibl_names = self.get_ibl_list()
        if name not in ibl_names:
            logger.log_error(
                f"Invalid IBL name: {name}. Available names are: {ibl_names}"
            )
        return str(Path(self.extract_dir) / "embodysim_resources" / "IBL" / name)

    def get_ibl_list(self) -> List[str]:
        """Get the names of all IBL resources.

        Returns:
            List[str]: The names of all IBL resources.
        """
        return [
            f.name
            for f in Path(self.extract_dir).glob("embodysim_resources/IBL/*")
            if f.is_dir()
        ]

    def get_material_path(self, name: str) -> str:
        """Get the path of the material resource.

        Args:
            name (str): The name of the material resource.

        Returns:
            str: The path to the material resource.
        """
        material_names = self.get_material_list()
        if name not in material_names:
            logger.log_error(
                f"Invalid material name: {name}. Available names are: {material_names}"
            )
        return str(Path(self.extract_dir) / "embodysim_resources" / "materials" / name)

    def get_material_list(self) -> List[str]:
        """Get the names of all material resources.

        Returns:
            List[str]: The names of all material resources.
        """
        return [
            f.name
            for f in Path(self.extract_dir).glob("embodysim_resources/materials/*")
            if f.is_dir()
        ]


class CocoBackground(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, material_assets, "CocoBackground.zip"
            ),
            "fda82404a317281263bd5849e9eb31a1",
        )
        prefix = "CocoBackground"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
