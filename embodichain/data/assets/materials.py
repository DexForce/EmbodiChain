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
import hashlib
import shutil
import zipfile
import open3d as o3d

from pathlib import Path
from typing import ClassVar, List

from embodichain.data.dataset import EmbodiChainDataset
from embodichain.utils import logger
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)
from embodichain.data.default_materials import _catalog, get_default_material_info

__all__ = [
    "SimResources",
    "EnvMapHDR",
    "CocoBackground",
    "BackgroundTexture",
    "DefaultMaterials",
]

material_assets = "materials"


class DefaultMaterials(EmbodiChainDataset):
    """Lazily download the pinned default-material collection.

    Args:
        data_root: Optional data cache root. Each resource revision has its own
            locked cache entry. ``EMBODICHAIN_DOWNLOAD_PREFIX`` selects a mirror
            containing the catalog's versioned archive path; otherwise the
            original DexSim resource server is used (requires network access).
    """

    cache_prefix: ClassVar[str] = f"DefaultMaterials_{_catalog()['bundle']['revision']}"
    """Revision-specific cache identity, also used by the data CLI."""

    def __init__(self, data_root: str | None = None) -> None:
        bundle = _catalog()["bundle"]
        url = (
            EMBODICHAIN_DOWNLOAD_PREFIX + bundle["mirror_path"]
            if os.environ.get("EMBODICHAIN_DOWNLOAD_PREFIX")
            else bundle["upstream_url"]
        )
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        prefix = self.cache_prefix
        super().__init__(prefix, o3d.data.DataDescriptor(url, bundle["md5"]), root)
        archive = Path(self.download_dir) / bundle["archive_name"]
        if self._sha256(archive) != bundle["sha256"]:
            raise ValueError(
                "DefaultMaterials archive does not match its pinned SHA256."
            )
        self._verify_files(Path(self.extract_dir))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _verify_files(cls, root: Path) -> None:
        for name, info in _catalog()["bundle"]["files"].items():
            path = root / name
            if (
                not path.is_file()
                or path.stat().st_size != info["size_bytes"]
                or cls._sha256(path) != info["sha256"]
            ):
                raise ValueError(
                    f"Incomplete or corrupt DefaultMaterials resource: {path}."
                )

    def check_zip(
        self, prefix: str, data_descriptor: o3d.data.DataDescriptor, path: str
    ) -> None:
        """Validate or repair this revision while the dataset lock is held.

        Args:
            prefix: Versioned cache entry owned by this dataset.
            data_descriptor: Native download descriptor with the pinned MD5.
            path: Data cache root, including caller-provided roots.
        """
        # Called under EmbodiChainDataset's existing interprocess lock. Repair
        # only this revision's extraction, including caches with custom roots.
        archive = (
            Path(path) / "download" / prefix / _catalog()["bundle"]["archive_name"]
        )
        extracted = Path(path) / "extract" / prefix
        if not archive.is_file():
            if extracted.exists():
                shutil.rmtree(extracted)
            return
        if self._sha256(archive) != _catalog()["bundle"]["sha256"]:
            archive.unlink()
            if extracted.exists():
                shutil.rmtree(extracted)
            return
        try:
            self._verify_files(extracted)
        except ValueError:
            if extracted.exists():
                shutil.rmtree(extracted)
            # The archive was verified against a built-in immutable digest.
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extracted)

    def get_material_path(self, name: str) -> str:
        """Get an authored preset directory in this verified resource revision.

        Args:
            name: Downloadable resource preset or ``SoapBubbleInterior``.

        Returns:
            Absolute path suitable for DexSim's native material loader.

        Raises:
            ValueError: The name is unknown or denotes a built-in parameter
                preset, which has no directory in this archive.
        """
        info = get_default_material_info(name)
        if not info["requires_download"]:
            raise ValueError(
                f"{name!r} is a built-in parameter preset with no resource directory; "
                "use VisualMaterialCfg.from_preset()."
            )
        return str(Path(self.extract_dir) / "Material" / name)


class SimResources(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, material_assets, "embodisim_resources.zip"
            ),
            "53c054b3ae0857416dc52632eb562c12",
        )
        prefix = type(self).__name__
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


class EnvMapHDR(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, material_assets, "EnvMapHDR.zip"),
            "ea7abc8e955fe64069073d63834da60e",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)

    def get_env_map_path(self, name: str) -> str:
        """Get the path of an HDR environment map.

        Args:
            name (str): The name of the HDR environment map.

        Returns:
            str: The path to the HDR environment map file.
        """
        env_map_names = self.get_env_map_list()
        if name not in env_map_names:
            logger.log_error(
                f"Invalid env map name: {name}. Available names are: {env_map_names}"
            )
        return str(Path(self.extract_dir) / "EnvMapHDR" / name)

    def get_env_map_list(self) -> List[str]:
        """Get the names of all HDR environment maps.

        Returns:
            List[str]: The names of all HDR environment map files.
        """
        return [
            f.name
            for f in Path(self.extract_dir).glob("EnvMapHDR/*.hdr")
            if f.is_file()
        ]


class CocoBackground(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, material_assets, "CocoBackground.zip"
            ),
            "fda82404a317281263bd5849e9eb31a1",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class BackgroundTexture(EmbodiChainDataset):
    """Download the bundled background-texture collection.

    Args:
        data_root: Optional root directory for downloaded and extracted assets.
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, material_assets, "BackgroundTexture.zip"
            ),
            "09cbde19f9e29d09c9aadf93b5dcf45d",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
