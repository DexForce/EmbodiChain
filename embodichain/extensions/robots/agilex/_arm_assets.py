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

"""Cache portable URDFs from the pinned AgileX arm description repository."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import open3d as o3d

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data.dataset import EmbodiChainDataset, _dataset_download_lock

__all__ = []

_REVISION = "680fa925cd00db7e2628563d2edc5540963045de"
_SOURCE = f"https://raw.githubusercontent.com/agilexrobotics/agx_arm_urdf/{_REVISION}"
_LICENSE_SHA256 = "247944c63de1573653722ba784f96123da267105b7b48577e0c8826a24b809f6"


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class _AgilexArmAsset(EmbodiChainDataset):
    """Fetch only one model's source files, sharing the project's cache lock."""

    def __init__(
        self,
        data_root: str | None,
        *,
        source_urdf: str,
        urdf_md5: str,
        source_files: dict[str, str],
        output_urdf: str,
    ) -> None:
        root = str(
            Path(data_root or EMBODICHAIN_DEFAULT_DATA_ROOT).expanduser().resolve()
        )
        descriptor = o3d.data.DataDescriptor(f"{_SOURCE}/{source_urdf}", urdf_md5)
        super().__init__(self.cache_prefix, descriptor, root)
        extracted = Path(self.extract_dir)
        with _dataset_download_lock(root, self.cache_prefix):
            for relative, expected in {
                "LICENSE": _LICENSE_SHA256,
                **source_files,
            }.items():
                destination = extracted / relative
                if (
                    destination.is_file()
                    and hashlib.sha256(destination.read_bytes()).hexdigest() == expected
                ):
                    continue
                if relative == source_urdf:
                    content = (
                        Path(self.download_dir) / Path(relative).name
                    ).read_bytes()
                else:
                    with urlopen(f"{_SOURCE}/{relative}", timeout=30) as stream:
                        content = stream.read()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"AgileX asset checksum mismatch for {relative}")
                _write_atomic(destination, content)

            portable = extracted / output_urdf
            content = self._portable_urdf(extracted, source_urdf).replace(
                b"package://agx_arm_description/agx_arm_urdf/", b""
            )
            if not portable.is_file() or portable.read_bytes() != content:
                _write_atomic(portable, content)

    def _portable_urdf(self, extracted: Path, source_urdf: str) -> bytes:
        return (extracted / source_urdf).read_bytes()
