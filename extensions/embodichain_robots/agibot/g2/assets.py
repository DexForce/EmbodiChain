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

"""Pinned official G2 meshes and a portable, pre-expanded URDF."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import open3d as o3d

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data.dataset import EmbodiChainDataset, _dataset_download_lock

__all__ = ["AgibotG2"]

_RESOURCES = Path(__file__).parent
_LOCK = json.loads((_RESOURCES / "source.lock.json").read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_verified(destination: Path, content: bytes, expected: str) -> None:
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"G2 asset checksum mismatch for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class AgibotG2(EmbodiChainDataset):
    """Resolve the official CRS-arm, OmniPicker-equipped G2 at a fixed revision.

    The source archive uses the shared locked download cache. Meshes and source
    evidence are SHA256-checked and repaired from that archive. The packaged
    URDF is expanded beforehand, so normal use requires neither ROS nor Xacro.
    The original archive and its third-party notices remain in the cache;
    this preset references only AgiBot's Apache-2.0 G2 assets.

    Args:
        data_root: Optional asset cache root; defaults to the configured
            ``EMBODICHAIN_DATA_ROOT``.
    """

    cache_prefix = f"AgibotG2_{_LOCK['revision'][:12]}_v1"

    def __init__(self, data_root: str | None = None) -> None:
        root = str(
            Path(EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root)
            .expanduser()
            .resolve()
        )
        descriptor = o3d.data.DataDescriptor(_LOCK["archive_url"], _LOCK["archive_md5"])
        super().__init__(self.cache_prefix, descriptor, root)
        with _dataset_download_lock(root, self.cache_prefix):
            extracted = Path(self.extract_dir)
            archive_path = (
                Path(self.download_dir) / _LOCK["archive_url"].rsplit("/", 1)[1]
            )
            if _sha256(archive_path) != _LOCK["archive_sha256"]:
                raise ValueError("G2 source archive SHA256 mismatch")
            with ZipFile(archive_path) as archive:
                for relative, digest in _LOCK["files"].items():
                    destination = extracted / relative
                    if not destination.is_file() or _sha256(destination) != digest:
                        _write_verified(destination, archive.read(relative), digest)

            template = (_RESOURCES / "g2.urdf").read_bytes()
            if hashlib.sha256(template).hexdigest() != _LOCK["urdf_sha256"]:
                raise ValueError("G2 packaged URDF checksum mismatch")
            # Keep the archive's source tree intact and make mesh paths relative
            # to the public AgibotG2/g2.urdf entry point.
            mesh_prefix = f"{_LOCK['source_directory']}/robots/genie/g2/meshes/"
            derived = template.replace(
                b'filename="meshes/', f'filename="{mesh_prefix}'.encode()
            )
            for name, content in (
                ("g2.urdf", derived),
                (
                    "LICENSE",
                    (extracted / _LOCK["source_directory"] / "LICENSE").read_bytes(),
                ),
                ("source.lock.json", (_RESOURCES / "source.lock.json").read_bytes()),
            ):
                digest = hashlib.sha256(content).hexdigest()
                destination = extracted / name
                if not destination.is_file() or _sha256(destination) != digest:
                    _write_verified(destination, content, digest)
