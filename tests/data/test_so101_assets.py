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

"""The SO101 asset cache verifies dependencies and repairs partial downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from embodichain_robots.the_robot_studio.so101 import assets as so101_assets

pytestmark = pytest.mark.no_sim


@pytest.fixture
def upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "upstream"
    files = {
        "so101_new_calib.urdf": b'<robot name="so101"/>',
        "assets/body.stl": b"verified mesh data",
        "LICENSE": b"Apache-2.0 fixture license",
    }
    for relative, content in files.items():
        path = root / (
            relative if relative == "LICENSE" else f"Simulation/SO101/{relative}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(so101_assets, "_SOURCE_ROOT", root.as_uri())
    monkeypatch.setattr(
        so101_assets,
        "_URDF_MD5",
        hashlib.md5(files["so101_new_calib.urdf"]).hexdigest(),
    )
    monkeypatch.setattr(
        so101_assets,
        "_FILES",
        {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    )
    return root


def test_so101_cache_reuses_verified_files_and_repairs_mesh(
    upstream: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = str(tmp_path / "embodichain_data")
    asset = so101_assets.SO101FollowerAssets(root)
    extracted = Path(asset.extract_dir)
    assert (extracted / "LICENSE").read_bytes() == (upstream / "LICENSE").read_bytes()
    assert (extracted / "so101_new_calib.urdf").read_bytes() == (
        upstream / "Simulation/SO101/so101_new_calib.urdf"
    ).read_bytes()
    with monkeypatch.context() as context:
        context.setattr(
            so101_assets,
            "urlopen",
            lambda *args, **kwargs: pytest.fail(
                "Verified cache must not be downloaded again"
            ),
        )
        so101_assets.SO101FollowerAssets(root)
    mesh = extracted / "assets/body.stl"
    mesh.write_bytes(b"truncated download")
    so101_assets.SO101FollowerAssets(root)
    assert mesh.read_bytes() == b"verified mesh data"


def test_so101_rejects_changed_source_without_publishing_unverified_mesh(
    upstream: Path, tmp_path: Path
) -> None:
    (upstream / "Simulation/SO101/assets/body.stl").write_bytes(b"changed source")
    root = tmp_path / "embodichain_data"
    with pytest.raises(ValueError, match="checksum mismatch for body.stl"):
        so101_assets.SO101FollowerAssets(str(root))
    asset_dir = root / "extract" / so101_assets.SO101FollowerAssets.cache_prefix
    assert not (asset_dir / "assets/body.stl").exists()
    assert list((asset_dir / "assets").iterdir()) == []
