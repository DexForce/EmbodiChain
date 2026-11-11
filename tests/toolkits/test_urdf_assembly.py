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

"""URDF assembly preserves the resources referenced by component meshes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodichain.toolkits.urdf_assembly import URDFAssemblyManager
from embodichain.toolkits.urdf_assembly.signature import URDFAssemblySignatureManager


def test_dual_arm_assembly_restores_declared_obj_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    robot = source / "X5A.urdf"
    robot.write_text(
        '<robot name="X5A"><link name="base_link"><visual><geometry>'
        '<mesh filename="link6_visual.obj"/></geometry></visual></link></robot>'
    )
    (source / "link6_visual.obj").write_text(
        "mtllib link6.mtl\nusemtl black\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    )
    material = b"newmtl black\nKd 0 0 0\n"
    (source / "link6.mtl").write_bytes(material)

    manager = URDFAssemblyManager()
    assert manager.add_component("left_arm", robot)
    assert manager.add_component("right_arm", robot)
    output = tmp_path / "assembled" / "X5A_X5A.urdf"

    hash_signature = URDFAssemblySignatureManager._calculate_string_md5

    def legacy_signature(
        signature_manager: URDFAssemblySignatureManager, serialized: str
    ) -> str:
        payload = json.loads(serialized)
        payload.pop("mesh_copy_version", None)
        return hash_signature(signature_manager, json.dumps(payload, sort_keys=True))

    with monkeypatch.context() as legacy:
        legacy.setattr(
            URDFAssemblySignatureManager, "_calculate_string_md5", legacy_signature
        )
        old = manager.merge_urdfs(str(output))
    paths = [output.parent / mesh.attrib["filename"] for mesh in old.findall(".//mesh")]
    assert len(paths) == 2 and paths[0] != paths[1]
    for path in paths:
        path.with_name("link6.mtl").unlink(missing_ok=True)

    manager.merge_urdfs(str(output))

    for path in paths:
        assert "mtllib link6.mtl" in path.read_text().splitlines()
        assert path.with_name("link6.mtl").read_bytes() == material
