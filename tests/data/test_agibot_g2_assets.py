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

"""G2 asset installation and repair use a pinned archive without ROS."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

from embodichain_robots.agibot.g2 import assets

pytestmark = pytest.mark.no_sim


@pytest.fixture
def source_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    archive = tmp_path / "g2.zip"
    source_files = {
        "source/LICENSE": b"Apache-2.0 fixture license",
        "source/robots/genie/g2/meshes/body.stl": b"verified source mesh",
    }
    with ZipFile(archive, "w") as output:
        directories = sorted(
            {
                str(p) + "/"
                for name in source_files
                for p in Path(name).parents
                if str(p) != "."
            }
        )
        for name in directories:
            output.writestr(name, b"")
        for name, content in source_files.items():
            output.writestr(name, content)
    resources = tmp_path / "resources"
    resources.mkdir()
    template = b'<robot name="fixture"><link name="base"><visual><geometry><mesh filename="meshes/body.stl"/></geometry></visual></link></robot>'
    (resources / "g2.urdf").write_bytes(template)
    lock = {
        "archive_url": archive.as_uri(),
        "archive_md5": hashlib.md5(archive.read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_directory": "source",
        "urdf_sha256": hashlib.sha256(template).hexdigest(),
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in source_files.items()
        },
    }
    (resources / "source.lock.json").write_text(json.dumps(lock))
    monkeypatch.setattr(assets, "_RESOURCES", resources)
    monkeypatch.setattr(assets, "_LOCK", lock)
    return archive


def test_g2_installs_relative_meshes_and_repairs_cached_files(
    source_archive: Path, tmp_path: Path
) -> None:
    root = str(tmp_path / "embodichain_data")
    asset = assets.AgibotG2(root)
    extracted = Path(asset.extract_dir)
    urdf = extracted / "g2.urdf"
    mesh = extracted / ET.parse(urdf).find(".//mesh").get("filename")
    assert mesh.read_bytes() == b"verified source mesh"
    assert (extracted / "LICENSE").read_bytes() == b"Apache-2.0 fixture license"
    assert json.loads((extracted / "source.lock.json").read_text()) == assets._LOCK
    # Repair uses the cached verified archive, even when upstream is unavailable.
    source_archive.unlink()
    mesh.write_bytes(b"corrupt")
    urdf.write_bytes(b"corrupt")
    (extracted / "LICENSE").unlink()
    assets.AgibotG2(root)
    assert mesh.read_bytes() == b"verified source mesh"
    assert ET.parse(urdf).getroot().get("name") == "fixture"
    assert (extracted / "LICENSE").is_file()


def test_g2_rejects_modified_packaged_urdf(
    source_archive: Path, tmp_path: Path
) -> None:
    (assets._RESOURCES / "g2.urdf").write_bytes(b"modified resource")
    with pytest.raises(ValueError, match="packaged URDF checksum mismatch"):
        assets.AgibotG2(str(tmp_path / "embodichain_data"))
