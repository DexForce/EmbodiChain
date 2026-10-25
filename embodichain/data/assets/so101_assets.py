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

"""Pinned official SO101 follower descriptions, meshes and motor references."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import open3d as o3d

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data.dataset import EmbodiChainDataset, _dataset_download_lock

__all__ = ["SO101FollowerAssets"]

_REVISION = "a758567c3978dfeefe282ede0500085a48fe8f78"
_SOURCE_ROOT = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{_REVISION}"
_URDF = "so101_new_calib.urdf"
_URDF_MD5 = "308645ed050642dbbfc21814588a286c"
_FILES = {
    "README.md": "c6df213e471e4573ba6384cb957e9eee9644f5d671272f98f67d28866333f747",
    "assets/base_motor_holder_so101_v1.stl": "8cd2f241037ea377af1191fffe0dd9d9006beea6dcc48543660ed41647072424",
    "assets/base_so101_v2.stl": "bb12b7026575e1f70ccc7240051f9d943553bf34e5128537de6cd86fae33924d",
    "assets/motor_holder_so101_base_v1.stl": "31242ae6fb59d8b15c66617b88ad8e9bded62d57c35d11c0c43a70d2f4caa95b",
    "assets/motor_holder_so101_wrist_v1.stl": "887f92e6013cb64ea3a1ab8675e92da1e0beacfd5e001f972523540545e08011",
    "assets/moving_jaw_so101_v1.stl": "785a9dded2f474bc1d869e0d3dae398a3dcd9c0c345640040472210d2861fa9d",
    "assets/rotation_pitch_so101_v1.stl": "9be900cc2a2bf718102841ef82ef8d2873842427648092c8ed2ca1e2ef4ffa34",
    "assets/sts3215_03a_no_horn_v1.stl": "75ef3781b752e4065891aea855e34dc161a38a549549cd0970cedd07eae6f887",
    "assets/sts3215_03a_v1.stl": "a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0",
    "assets/under_arm_so101_v1.stl": "d01d1f2de365651dcad9d6669e94ff87ff7652b5bb2d10752a66a456a86dbc71",
    "assets/upper_arm_so101_v1.stl": "475056e03a17e71919b82fd88ab9a0b898ab50164f2a7943652a6b2941bb2d4f",
    "assets/waveshare_mounting_plate_so101_v2.stl": "e197e24005a07d01bbc06a8c42311664eaeda415bf859f68fa247884d0f1a6e9",
    "assets/wrist_roll_follower_so101_v1.stl": "4b17b410a12d64ec39554abc3e8054d8a97384b2dc4a8d95a5ecb2a93670f5f4",
    "assets/wrist_roll_pitch_so101_v2.stl": "6c7ec5525b4d8b9e397a30ab4bb0037156a5d5f38a4adf2c7d943d6c56eda5ae",
    "joints_properties.xml": "abbcc92761943befe3a24b1c13e82c62c5b8c2914303818f3c870b1f51cd45ed",
    "so101_new_calib.urdf": "3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_file(url: str, destination: Path, expected: str) -> None:
    if destination.is_file() and _sha256(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=60) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
        actual = _sha256(temporary)
        if actual != expected:
            raise ValueError(
                f"SO101 asset checksum mismatch for {destination.name}: {actual}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SO101FollowerAssets(EmbodiChainDataset):
    """Download the official follower URDF and its dependencies at a fixed revision.

    The original Apache-2.0 license, CAD inertials, joint frames, limits and
    yellow/black URDF materials are preserved. Each source file is checked
    against its pinned SHA256, including when repairing an existing cache.

    Args:
        data_root: Optional asset cache root. Omission uses
            ``EMBODICHAIN_DATA_ROOT`` through the shared data constants.
    """

    cache_prefix = f"SO101Follower_{_REVISION[:12]}"

    def __init__(self, data_root: str | None = None) -> None:
        root = str(
            Path(EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root)
            .expanduser()
            .resolve()
        )
        descriptor = o3d.data.DataDescriptor(
            f"{_SOURCE_ROOT}/Simulation/SO101/{_URDF}", _URDF_MD5
        )
        super().__init__(self.cache_prefix, descriptor, root)
        with _dataset_download_lock(root, self.cache_prefix):
            for relative, digest in _FILES.items():
                if relative == _URDF:
                    url = (Path(self.download_dir) / _URDF).as_uri()
                else:
                    source = (
                        relative
                        if relative == "LICENSE"
                        else f"Simulation/SO101/{relative}"
                    )
                    url = f"{_SOURCE_ROOT}/{source}"
                _ensure_file(url, Path(self.extract_dir) / relative, digest)
