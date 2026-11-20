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

"""Portable prepared-robot resource snapshots and ownership boundaries."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from embodichain.data.resources import snapshot_urdf_resources


def test_urdf_snapshot_preserves_transitive_files_and_arbitrary_model_notices(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    source = package / "prepared"
    source.mkdir(parents=True)
    urdf = source / "robot.urdf"
    urdf.write_text('<robot><mesh filename="mesh/model.obj"/></robot>')
    meshes = source / "mesh"
    meshes.mkdir()
    (meshes / "model.obj").write_text("mtllib material.mtl\nv 0 0 0\n")
    (meshes / "material.mtl").write_text("map_Kd texture.png\n")
    (meshes / "texture.png").write_bytes(b"texture-payload")
    (package / "source_manifest.json").write_text('{"source":"pinned"}')
    vendor = package / "another_robot"
    vendor.mkdir()
    (vendor / "LICENSE").write_text("original license")
    (package / "NOTICE").write_text("original notice")
    target = tmp_path / "snapshot"
    entry = snapshot_urdf_resources(urdf, target, provenance_root=package)
    shutil.rmtree(package)
    assert entry.read_text() == '<robot><mesh filename="mesh/model.obj"/></robot>'
    assert (target / "mesh/texture.png").read_bytes() == b"texture-payload"
    assert (target / "mesh/material.mtl").read_text() == "map_Kd texture.png\n"
    assert (target / "licenses/another_robot.LICENSE").read_text() == "original license"
    assert (target / "licenses/NOTICE").read_text() == "original notice"
    assert (target / "source_manifest.json").read_text() == '{"source":"pinned"}'
    assert snapshot_urdf_resources(entry, target) == entry


def test_urdf_snapshot_rejects_recursive_copy_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "robot.urdf"
    entry.write_text("<robot/>")
    target = source / "snapshot"
    with pytest.raises(ValueError, match="outside"):
        snapshot_urdf_resources(entry, target)
    assert not target.exists()
    with pytest.raises(FileNotFoundError):
        snapshot_urdf_resources(tmp_path / "missing.urdf", tmp_path / "destination")
