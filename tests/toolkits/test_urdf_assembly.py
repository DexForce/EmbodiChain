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
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest

from embodichain.toolkits.urdf_assembly import URDFAssemblyManager
from embodichain.toolkits.urdf_assembly.signature import URDFAssemblySignatureManager
from embodichain.toolkits.urdf_assembly.mesh import URDFMeshManager
from embodichain.toolkits.urdf_assembly.component import URDFComponentManager
from embodichain.toolkits.urdf_assembly.sensor import URDFSensorManager


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


@pytest.fixture
def mesh_sources(tmp_path: Path) -> tuple[Path, Path]:
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "source.stl").write_text("solid test\nendsolid test\n")
    paths = []
    for filename, link in (("arm", "ArmBase"), ("camera", "body")):
        path = tmp_path / f"{filename}.urdf"
        path.write_text(
            f'<robot name="{filename}"><link name="{link}"><visual><geometry>'
            '<mesh filename="meshes/source.stl"/></geometry></visual></link></robot>'
        )
        paths.append(path)
    return paths[0], paths[1]


@pytest.mark.parametrize("injection", ["all", "component_only"])
def test_injected_managers_follow_output_and_name_policy_across_merges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mesh_sources: tuple[Path, Path],
    injection: str,
) -> None:
    source, _ = mesh_sources
    mesh = URDFMeshManager(str(tmp_path / "unused"))
    component = URDFComponentManager(mesh)
    sensor = URDFSensorManager(mesh)
    ensure_dirs = Mock(wraps=mesh.ensure_dirs)
    process_component = Mock(wraps=component.process_component)
    process_sensors = Mock(wraps=sensor.process_sensor_attachments)
    monkeypatch.setattr(mesh, "ensure_dirs", ensure_dirs)
    monkeypatch.setattr(component, "process_component", process_component)
    monkeypatch.setattr(sensor, "process_sensor_attachments", process_sensors)
    manager = (
        URDFAssemblyManager(
            mesh_manager=mesh, component_manager=component, sensor_manager=sensor
        )
        if injection == "all"
        else URDFAssemblyManager(component_manager=component)
    )
    assert manager.add_component("arm", str(source))

    for directory, casing, expected_name in (
        ("lower", "lower", "armbase"),
        ("upper", "upper", "ARMBASE"),
    ):
        manager.name_case = {"link": casing, "joint": "original"}
        output = tmp_path / directory / "robot.urdf"

        root = manager.merge_urdfs(str(output), use_signature_check=False)

        assert output.is_file()
        assert root.find(f"link[@name='{expected_name}']") is not None
        assert mesh.output_dir == str(output.parent)
        assert (output.parent / "Visual").is_dir()
        assert (output.parent / "Collision").is_dir()
        for reference in root.findall(".//mesh"):
            assert (output.parent / reference.get("filename")).is_file()

    ensure_dirs.assert_called()
    process_component.assert_called()
    if injection == "all":
        process_sensors.assert_called()


def test_injected_sensor_manager_does_not_retain_removed_registrations(
    tmp_path: Path, mesh_sources: tuple[Path, Path]
) -> None:
    arm, camera = mesh_sources
    sensors = URDFSensorManager(URDFMeshManager(str(tmp_path / "unused")))
    manager = URDFAssemblyManager(sensor_manager=sensors)
    assert manager.add_component("arm", str(arm))
    assert manager.attach_sensor("cam", str(camera), "arm", "ArmBase")
    output = tmp_path / "output" / "robot.urdf"

    root = manager.merge_urdfs(str(output), use_signature_check=False)

    assert root.find("link[@name='cam_body']") is not None
    for reference in root.findall(".//mesh"):
        assert (output.parent / reference.get("filename")).is_file()

    manager.sensor_registry.remove("cam")
    root = manager.merge_urdfs(str(output), use_signature_check=False)

    assert root.find("link[@name='cam_body']") is None
    assert sensors.attached_sensors == {}


def test_preloaded_sensor_templates_survive_repeated_and_failed_merges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mesh_sources: tuple[Path, Path],
) -> None:
    arm, camera = mesh_sources
    mesh = URDFMeshManager(str(tmp_path / "unused"))
    sensors = URDFSensorManager(mesh)
    assert sensors.attach_sensor("cam", str(camera), "arm", "ArmBase")
    sources = sensors.attached_sensors
    source_link = sources["cam"]["links"][0]
    source_xml = ET.tostring(source_link)
    manager = URDFAssemblyManager(mesh_manager=mesh, sensor_manager=sensors)
    assert manager.add_component("arm", str(arm))

    for directory in ("first", "second"):
        output = tmp_path / directory / "robot.urdf"
        root = manager.merge_urdfs(str(output), use_signature_check=False)

        assert root.find("link[@name='cam_body']") is not None
        for reference in root.findall(".//mesh"):
            assert (output.parent / reference.get("filename")).is_file()
        assert sensors.attached_sensors is sources
        assert ET.tostring(source_link) == source_xml

    failure = OSError("test write failure")
    monkeypatch.setattr(manager.file_writer, "write_urdf", Mock(side_effect=failure))

    with pytest.raises(OSError, match="test write failure"):
        manager.merge_urdfs(
            str(tmp_path / "failure" / "robot.urdf"), use_signature_check=False
        )

    assert sensors.attached_sensors is sources
    assert sensors.attached_sensors["cam"]["links"][0] is source_link
    assert ET.tostring(source_link) == source_xml
