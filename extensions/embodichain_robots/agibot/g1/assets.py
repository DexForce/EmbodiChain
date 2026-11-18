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

"""Resolve the original GenieSim USD and the distinct observed-replay URDF."""

from __future__ import annotations

import hashlib
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import ClassVar
from urllib.request import urlopen

import open3d as o3d

from embodichain.data.constants import (
    EMBODICHAIN_DEFAULT_DATA_ROOT,
    EMBODICHAIN_DOWNLOAD_PREFIX,
)
from embodichain.data.dataset import EmbodiChainDataset, _dataset_download_lock

__all__ = ["AgibotG1", "AgibotG1ObservedReplay"]

_REVISION = "a0813aad7c16165daffbe6c2737754e0344809ee"
_SOURCE = (
    f"https://huggingface.co/datasets/agibot-world/GenieSimAssets/resolve/{_REVISION}"
)
_MODEL_SOURCE = f"{_SOURCE}/robot/G1_120s"
_ROBOT_USD = "robot.usda"
_ROBOT_USD_MD5 = "05fad1a0d51b3e2a1782757f66d06986"
_FILES = {
    "LICENSE": "e66c269d4819aaab34b49ef5220c4ddab6756f21bb5180761a4eb8561f2b7bbd",
    "config.yaml": "8c4dd46f3de94e2710a99a5ef5d2bf68292f83918e756572b411b279ffdf0e29",
    _ROBOT_USD: "6ecb91bc40a3e9b2fe883c5f8c63d6bc29186a5ed124025ff9c3811f4df8d603",
    "configuration/G1_120s_base.usd": "90f385a1959ebcd20038ef16d09c0990971f0615e9e8aaacd9943da9b2f4319b",
    "configuration/G1_120s_physics.usd": "a43486eec6424fecc8e22d6278fc734a1beecd2a912852297ff887f7f760a4e9",
    "configuration/G1_120s_sensor.usd": "343212ad7f4fbdf8cf646a4256477733deea96cdffcaac6e6d4a5f2de5217c1d",
    "configuration/camera_stand.usdc": "ea2df2916182f264e2332a4b4a4003f5ccb570386707c5e8bc27f282fbb0f65e",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _ensure_file(url: str, destination: Path, expected: str) -> None:
    if destination.is_file() and _sha256(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=60) as source:
                copyfileobj(source, output)
        if _sha256(temporary) != expected:
            raise ValueError(f"AgiBot G1 source SHA256 mismatch: {destination.name}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class AgibotG1(EmbodiChainDataset):
    """Download and verify the original G1 USD layers and their license.

    The model is provided by AgiBot's GenieSimAssets under CC BY-NC-SA 4.0.
    The cache retains the original files, attribution and relative layer paths.

    Args:
        data_root: Optional asset cache root.
    """

    cache_prefix = f"AgibotG1_{_REVISION[:12]}"

    def __init__(self, data_root: str | None = None) -> None:
        root = str(
            Path(data_root or EMBODICHAIN_DEFAULT_DATA_ROOT).expanduser().resolve()
        )
        descriptor = o3d.data.DataDescriptor(
            f"{_MODEL_SOURCE}/{_ROBOT_USD}", _ROBOT_USD_MD5
        )
        super().__init__(self.cache_prefix, descriptor, root)
        with _dataset_download_lock(root, self.cache_prefix):
            for relative, expected in _FILES.items():
                if relative == _ROBOT_USD:
                    url = (Path(self.download_dir) / _ROBOT_USD).as_uri()
                else:
                    source = _SOURCE if relative == "LICENSE" else _MODEL_SOURCE
                    url = f"{source}/{relative}"
                _ensure_file(url, Path(self.extract_dir) / relative, expected)


class AgibotG1ObservedReplay(EmbodiChainDataset):
    """Download the pinned, collision-bearing observed-replay G1 URDF.

    The model bundle contains ``robot.urdf``, local visual and collision meshes,
    collision provenance, attribution and the CC-BY-NC-SA-4.0 license. It uses
    a separate versioned cache from the original :class:`AgibotG1` USD model.

    Args:
        data_root: Optional asset cache root; defaults to the configured
            ``EMBODICHAIN_DATA_ROOT``.

    Raises:
        ValueError: The pinned archive, checksum manifest, or extracted files differ.

    Example:
        >>> from embodichain.data import get_data_path
        >>> path = get_data_path("AgibotG1ObservedReplay/robot.urdf")
    """

    cache_prefix: ClassVar[str] = "AgibotG1ObservedReplayV2"
    _archive_name: ClassVar[str] = "AgibotG1ObservedReplayV2.zip"
    _revision: ClassVar[str] = "e002d649bb292d3e18c3a97d5b5c22c8ddcc81a3"
    _archive_md5: ClassVar[str] = "b8dbea604576e68c1371cfc9eb55b337"
    _archive_sha256: ClassVar[str] = (
        "a289e751006be59e830119be1629a6a55a07338176d2ab4171e22463fc4e9c9c"
    )
    _checksums_sha256: ClassVar[str] = (
        "4308a768e868127079934ca07ca0ceb541d80f2270c37f7269725a9b6e54fd80"
    )

    def __init__(self, data_root: str | None = None) -> None:
        filename = "robot_assets/" + self._archive_name
        origin = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/"
            f"{self._revision}/"
        )
        prefix = EMBODICHAIN_DOWNLOAD_PREFIX.replace(
            "/resolve/main/", f"/resolve/{self._revision}/", 1
        )
        urls = [prefix + filename]
        if origin != prefix:
            urls.append(origin + filename)
        descriptor = o3d.data.DataDescriptor(urls, self._archive_md5)
        root = str(
            Path(EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root)
            .expanduser()
            .resolve()
        )
        super().__init__(self.cache_prefix, descriptor, root)
        with _dataset_download_lock(root, self.cache_prefix):
            if (
                _sha256(Path(self.download_dir) / self._archive_name)
                != self._archive_sha256
            ):
                raise ValueError("AgiBot observed-replay archive SHA256 mismatch")
            self._verify_files(Path(self.extract_dir))

    @classmethod
    def _verify_files(cls, root: Path) -> None:
        root = root.resolve()
        checksums = root / "SHA256SUMS"
        if not checksums.is_file() or _sha256(checksums) != cls._checksums_sha256:
            raise ValueError(
                "AgiBot observed-replay checksum manifest is missing or changed"
            )
        for line in checksums.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = (root / relative).resolve()
            if (
                not path.is_relative_to(root)
                or not path.is_file()
                or _sha256(path) != expected
            ):
                raise ValueError(
                    f"AgiBot observed-replay resource is missing or changed: {relative}"
                )
