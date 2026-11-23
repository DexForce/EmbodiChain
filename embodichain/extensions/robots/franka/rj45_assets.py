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
"""Pinned Franka narrow jaws and RJ45 fixture for the insertion-cycle scene."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import open3d as o3d

from embodichain.data import EmbodiChainDataset, get_data_path
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data.dataset import _dataset_cache_paths, _dataset_download_lock

from .assets import _FR3_WITH_HAND_SHA256, _franka_v4_archive, _write_derivatives
from .rope_pads import apply_rope_pads

__all__ = ["FrankaRj45", "Rj45Fixture", "get_rj45_models"]

_BUNDLE = "dexuni_cable_rj45_assets_v1"
_BUNDLE_MD5 = "fbb618b557b494b83d388e2dcfdb9c6f"
_BUNDLE_SHA256 = "e440f3a6d3c9b2db5b9ee9e16b5e5949fd31e9e7200bb70e66d86290a4c763ea"
_MANIFEST_SHA256 = "1b280e46082329d2bb4c238e1d79376eebd841b10c49b620f09e79ecdfef3767"
_ASSET_BASE = "http://192.168.3.120/CoreEngine/Data/dexsim_data/dexuni_demo"
_ROBOT_MODEL = "Panda/Fr3Rj45V1.urdf"
_FIXTURE_ROOT = f"{_BUNDLE}/assets/rj45"


class FrankaRj45(EmbodiChainDataset):
    """Derive the narrow jaws from the pinned FR3 model while retaining its hand.

    Args:
        data_root: Optional root of the shared EmbodiChain asset cache.
    """

    def __init__(self, data_root: str | None = None) -> None:
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__("Franka", _franka_v4_archive(), root)
        source = Path(self.extract_dir) / "Panda/PandaWithHand.urdf"
        if _sha256(source) != _FR3_WITH_HAND_SHA256:
            raise ValueError("Pinned FR3 with hand differs from the RJ45 scene source.")
        model = ET.parse(source).getroot()
        apply_rope_pads(
            model,
            stem_size=(0.014, 0.008, 0.04),
            center=(0, 0.002, 0.041),
            size=(0.009, 0.004, 0.012),
        )
        _write_derivatives(
            source,
            root,
            {
                Path(_ROBOT_MODEL).name: ET.tostring(
                    model, encoding="utf-8", xml_declaration=True
                )
            },
        )


class Rj45Fixture(EmbodiChainDataset):
    """Resolve the original articulated plug and its fixed socket fixture.

    Args:
        data_root: Optional root of the shared EmbodiChain asset cache.
    """

    def __init__(self, data_root: str | None = None) -> None:
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        prefix = type(self).__name__
        descriptor = o3d.data.DataDescriptor(
            f"{_ASSET_BASE}/{_BUNDLE}.zip", _BUNDLE_MD5
        )
        super().__init__(prefix, descriptor, root)
        fixture = Path(self.extract_dir) / _FIXTURE_ROOT
        with _dataset_download_lock(root, prefix):
            for name in ("plug", "socket"):
                target = fixture / f"rj45_{name}_v1.urdf"
                model = _fixture_urdf(fixture / f"rj45_{name}.urdf", name)
                if target.exists():
                    if target.read_bytes() != model:
                        raise ValueError(f"Cached RJ45 derivative differs: {target}.")
                    continue
                with NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                    pending = Path(temporary.name)
                    try:
                        temporary.write(model)
                        temporary.close()
                        pending.replace(target)
                    finally:
                        pending.unlink(missing_ok=True)

    def check_zip(
        self, prefix: str, data_descriptor: o3d.data.DataDescriptor, path: str
    ) -> None:
        """Install the verified archive under the ordinary dataset cache lock.

        The bundle omits ZIP directory entries, so its extraction needs a
        reader that creates the implicit nested directories.

        Args:
            prefix: Dataset name identifying one cache entry.
            data_descriptor: Archive URL and its native dataset digest.
            path: Root of the EmbodiChain asset cache.
        """
        download, extracted = _dataset_cache_paths(path, prefix)
        archive = download / f"{_BUNDLE}.zip"
        if not archive.is_file() or _sha256(archive) != _BUNDLE_SHA256:
            download.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=download, delete=False) as temporary:
                pending = Path(temporary.name)
                try:
                    with urlopen(data_descriptor.urls[0], timeout=30) as source:
                        shutil.copyfileobj(source, temporary)
                    temporary.close()
                    if _sha256(pending) != _BUNDLE_SHA256:
                        raise ValueError("RJ45 bundle differs from the pinned archive.")
                    pending.replace(archive)
                finally:
                    pending.unlink(missing_ok=True)
        manifest = extracted / _FIXTURE_ROOT / "asset_manifest.json"
        if manifest.is_file() and _sha256(manifest) == _MANIFEST_SHA256:
            return
        extracted.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=extracted.parent) as temporary:
            staging = Path(temporary) / "asset"
            # The archive content is pinned by SHA-256 before extraction.
            with ZipFile(archive) as bundle:
                bundle.extractall(staging)
            if _sha256(staging / _FIXTURE_ROOT / manifest.name) != _MANIFEST_SHA256:
                raise ValueError("RJ45 fixture differs from the pinned scene source.")
            if extracted.exists():
                shutil.rmtree(extracted)
            staging.replace(extracted)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fixture_urdf(source: Path, name: str) -> bytes:
    model = ET.parse(source).getroot()
    if name == "socket":
        ET.SubElement(model, "link", name="socket_mount")
        mount = ET.SubElement(model, "joint", name="socket_mount_joint", type="fixed")
        ET.SubElement(mount, "parent", link="socket_mount")
        ET.SubElement(mount, "child", link="socket")
        ET.SubElement(mount, "origin", xyz="0 0 0", rpy="0 0 0")
    for visual in model.iter("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        glb = Path(mesh.attrib["filename"]).with_suffix(".glb")
        if (source.parent / glb).is_file():
            mesh.set("filename", glb.as_posix())
            material = visual.find("material")
            if material is not None:
                visual.remove(material)
    for mesh in model.iter("mesh"):
        filename = Path(mesh.attrib["filename"])
        if filename.is_absolute() or not (source.parent / filename).is_file():
            raise ValueError(f"Missing relative mesh {filename} for {source.name}.")
    return ET.tostring(model, encoding="utf-8", xml_declaration=True)


def get_rj45_models() -> dict[str, Path]:
    """Resolve the three shared URDFs through the EmbodiChain asset cache.

    Returns:
        Paths for the `robot`, `plug` and `socket` models, each with meshes
        relative to its containing directory.
    """
    return {
        "robot": Path(get_data_path(f"FrankaRj45/{_ROBOT_MODEL}")).resolve(),
        "plug": Path(
            get_data_path(f"Rj45Fixture/{_FIXTURE_ROOT}/rj45_plug_v1.urdf")
        ).resolve(),
        "socket": Path(
            get_data_path(f"Rj45Fixture/{_FIXTURE_ROOT}/rj45_socket_v1.urdf")
        ).resolve(),
    }
