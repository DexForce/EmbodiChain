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

"""Cache pinned RoboParty source archives and prepare portable URDF paths."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ClassVar

import open3d as o3d

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data.dataset import EmbodiChainDataset, _dataset_download_lock

__all__ = []


class _RoboPartySourceAsset(EmbodiChainDataset):
    """Keep the official URDF/MJCF tree and expose root-level portable URDFs."""

    repository: ClassVar[str]
    revision: ClassVar[str]
    archive_md5: ClassVar[str]
    urdf_names: ClassVar[tuple[str, ...]]

    def __init__(self, data_root: str | None = None) -> None:
        """Download and prepare a pinned description.

        Args:
            data_root: Optional cache root. Defaults to the configured data root.
        """
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        url = f"https://github.com/Roboparty/{self.repository}/archive/{self.revision}.zip"
        descriptor = o3d.data.DataDescriptor(url, self.archive_md5)
        super().__init__(self.cache_prefix, descriptor, root)

        with _dataset_download_lock(root, self.cache_prefix):
            directory = Path(self.extract_dir)
            for name in self.urdf_names:
                source = (directory / "urdf" / name).read_bytes()
                portable = source.replace(b'filename="../meshes/', b'filename="meshes/')
                _publish_urdf(directory / name, portable)

    @property
    def extract_dir(self) -> str:
        """Return the extracted model root containing the URDFs and meshes."""
        return str(Path(super().extract_dir) / f"{self.repository}-{self.revision}")


def _publish_urdf(path: Path, content: bytes) -> None:
    """Replace a generated URDF atomically when its content changes."""
    if path.is_file() and path.read_bytes() == content:
        return
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
