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
"""Native compound and articulation conversion from neutral assembly data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embodichain.lab.sim.assembly import (
    to_articulation_configs,
    to_rigid_config,
    to_rigid_configs,
)
from embodichain.toolkits.assembly import AssemblyAsset, import_mjcf

SOURCE = (
    Path(__file__).resolve().parents[2] / "fixtures/assembly/body.xml"
).read_text()


def test_hinge_cannot_silently_become_free_body() -> None:
    pytest.importorskip("mujoco")
    asset = import_mjcf(
        SOURCE.replace(
            '<freejoint name="part_joint"/>', '<joint name="part_joint" type="hinge"/>'
        )
    )
    with pytest.raises(ValueError, match="articulation"):
        to_rigid_configs(
            asset,
        )


def test_hinge_visual_export_keeps_body_local_appearance_and_source_physics(
    tmp_path,
) -> None:
    """One normal-bearing visual must not replace or move the link's colliders."""
    from dataclasses import asdict
    from pathlib import Path
    import xml.etree.ElementTree as ET

    from embodichain.toolkits.assembly import (
        AssemblyBody,
        AssemblyGeometry,
        AssemblyJoint,
    )

    body = AssemblyBody(
        name="gear",
        position=[0.1, 0.2, 0.3],
        quaternion=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
        mass=2.0,
        com=[0.001, 0.002, 0.003],
        inertia=[[0.01, 0.001, 0.002], [0.001, 0.02, 0.003], [0.002, 0.003, 0.025]],
        joint=AssemblyJoint(name="axis", kind="hinge", damping=0.1, armature=0.02),
        geometries=[
            AssemblyGeometry(
                vertices=[[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]],
                triangles=[[0, 1, 2]],
                position=[0.02, -0.03, 0.04],
                quaternion=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
                rgba=[1, 0, 0, 1],
            ),
            AssemblyGeometry(
                kind="cylinder",
                size=[0.005, 0.01],
                position=[-0.02, 0, 0],
                visible=False,
            ),
        ],
    )
    original = asdict(body)
    config = to_articulation_configs(
        AssemblyAsset(bodies=[body]), tmp_path, source_contact=False
    )["gear"]
    robot = ET.parse(config.fpath).getroot()
    link = robot.find("./link[@name='gear']")
    visuals = link.findall("visual")
    assert len(visuals) == 1
    assert visuals[0].find("origin").attrib == {"xyz": "0 0 0", "rpy": "0 0 0"}
    visual_file = Path(config.fpath).parent / visuals[0].find("geometry/mesh").get(
        "filename"
    )
    lines = visual_file.read_text().splitlines()
    vertices = np.array(
        [list(map(float, line.split()[1:])) for line in lines if line.startswith("v ")]
    )
    np.testing.assert_allclose(
        vertices,
        [[0.02, -0.03, 0.04], [0.02, -0.02, 0.04], [0.01, -0.03, 0.04]],
        atol=1e-15,
    )
    normals = np.array(
        [list(map(float, line.split()[1:])) for line in lines if line.startswith("vn ")]
    )
    np.testing.assert_allclose(normals, [[0, 0, 1]], atol=1e-15)
    assert "f 1//1 2//1 3//1" in lines
    assert "Kd 1 0 0" in visual_file.with_suffix(".mtl").read_text()
    collisions = link.findall("collision")
    assert len(collisions) == 2
    np.testing.assert_allclose(
        np.fromstring(collisions[0].find("origin").get("xyz"), sep=" "),
        [0.02, -0.03, 0.04],
    )
    np.testing.assert_allclose(
        np.fromstring(collisions[0].find("origin").get("rpy"), sep=" "),
        [0, 0, np.pi / 2],
        atol=1e-15,
    )
    collision_file = Path(config.fpath).parent / collisions[0].find(
        "geometry/mesh"
    ).get("filename")
    assert "v 0.01 0 0" in collision_file.read_text().splitlines()
    assert collisions[1].find("geometry/cylinder").attrib == {
        "radius": "0.005",
        "length": "0.02",
    }
    inertial = link.find("inertial")
    assert float(inertial.find("mass").get("value")) == body.mass
    np.testing.assert_allclose(
        np.fromstring(inertial.find("origin").get("xyz"), sep=" "), body.com
    )
    assert {key: float(value) for key, value in inertial.find("inertia").items()} == {
        "ixx": 0.01,
        "ixy": 0.001,
        "ixz": 0.002,
        "iyy": 0.02,
        "iyz": 0.003,
        "izz": 0.025,
    }
    joint = robot.find("joint")
    assert joint.get("type") == "continuous"
    assert joint.find("child").get("link") == "gear"
    np.testing.assert_allclose(config.init_local_pose[:3, 3], body.position)
    np.testing.assert_allclose(
        config.init_local_pose[:3, :3],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        atol=1e-15,
    )
    assert asdict(body) == original


def test_usd_hinge_is_a_real_joint_with_a_fixed_anchor(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    from dexsim.kit.usd import parse_usd

    asset = import_mjcf(
        SOURCE.replace(
            '<freejoint name="part_joint"/>',
            '<joint name="part_joint" type="hinge" axis="0 0 1"/>',
        )
    )
    parsed = parse_usd(str(asset.write_usd(tmp_path / "hinge.usda")), root_name="test")
    assert len(parsed.articulations) == 1
    assert len(parsed.articulations[0].links) == 2
    assert len(parsed.articulations[0].joints) == 1
    assert parsed.articulations[0].fixed_base


def test_contact_priority_and_mixing_survive_native_config_conversion() -> None:
    pytest.importorskip("mujoco")
    xml = SOURCE.replace(
        "<worldbody>", '<default><geom priority="3" solmix="0.7"/></default><worldbody>'
    )
    body = import_mjcf(xml).bodies[0]
    native = to_rigid_config(
        body,
    )
    assert native.attrs.collision_props.priority == 3
    assert native.attrs.collision_props.solmix == pytest.approx(0.7)


def test_separate_visual_mesh_never_enters_collision_parts() -> None:
    pytest.importorskip("mujoco")
    from embodichain.lab.sim.shapes import CompoundShapeCfg

    xml = SOURCE.replace(
        "</body>",
        '<geom name="visual" type="box" size="1 1 1" contype="0" conaffinity="0" group="2" density="0"/></body>',
    )
    body = import_mjcf(xml).bodies[0]
    cfg = to_rigid_config(body, source_contact=False)
    assert isinstance(cfg.shape, CompoundShapeCfg)
    assert len(cfg.shape.parts) == 2
    assert len(cfg.shape.visual_parts) == 3
    assert cfg.attrs.mass_props.mass == 2.0


def test_native_visual_obj_preserves_source_triangles_and_body_ownership(
    tmp_path,
) -> None:
    """File loading must preserve separate surfaces rather than filling their gap."""
    from dataclasses import asdict

    from embodichain.toolkits.assembly import AssemblyBody, AssemblyGeometry
    from embodichain.lab.sim.assembly._visual import write_visual_mesh

    source_vertices = [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]]
    body = AssemblyBody(
        name="part",
        mass=2,
        com=[0.001, 0.002, 0.003],
        inertia=[[0.01, 0, 0], [0, 0.02, 0], [0, 0, 0.025]],
        geometries=[
            AssemblyGeometry(
                vertices=source_vertices,
                triangles=[[0, 1, 2]],
                position=[0.02, -0.03, 0.04],
                quaternion=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
            ),
            AssemblyGeometry(
                vertices=source_vertices,
                triangles=[[0, 1, 2]],
                position=[-0.02, 0, 0],
                rgba=[1, 0, 0, 1],
            ),
            AssemblyGeometry(kind="sphere", size=[2], visible=False),
        ],
    )
    original = asdict(body)
    output = write_visual_mesh(body, tmp_path / "part.obj")
    lines = output.read_text().splitlines()
    vertices = np.array(
        [list(map(float, line.split()[1:])) for line in lines if line.startswith("v ")]
    )
    faces = np.array(
        [
            [int(corner.split("//")[0]) for corner in line.split()[1:]]
            for line in lines
            if line.startswith("f ")
        ]
    )
    expected = [
        [0.02, -0.03, 0.04],
        [0.02, -0.02, 0.04],
        [0.01, -0.03, 0.04],
        [-0.02, 0, 0],
        [-0.01, 0, 0],
        [-0.02, 0.01, 0],
    ]
    np.testing.assert_allclose(vertices, expected, atol=1e-15)
    np.testing.assert_array_equal(faces, [[1, 2, 3], [4, 5, 6]])
    assert [line for line in lines if line.startswith("usemtl ")] == [
        "usemtl rgba_0",
        "usemtl rgba_1",
    ]
    assert lines[0] == "mtllib part.mtl"
    assert "Kd 1 0 0" in output.with_suffix(".mtl").read_text()
    assert asdict(body) == original


def test_native_visual_obj_keeps_rotated_sharp_edge_normals(tmp_path) -> None:
    """Shared vertices must not smooth a ninety-degree edge or translate normals."""
    from embodichain.toolkits.assembly import AssemblyBody, AssemblyGeometry
    from embodichain.lab.sim.assembly._visual import write_visual_mesh

    body = AssemblyBody(
        name="hard_corner",
        geometries=[
            AssemblyGeometry(
                vertices=[[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
                triangles=[[0, 1, 2], [0, 3, 1]],
                position=[1, 2, 3],
                quaternion=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
            )
        ],
    )
    lines = write_visual_mesh(body, tmp_path / "corner.obj").read_text().splitlines()
    normals = np.array(
        [list(map(float, line.split()[1:])) for line in lines if line.startswith("vn ")]
    )
    face_normals = [
        normals[[int(corner.split("//")[1]) - 1 for corner in line.split()[1:]]]
        for line in lines
        if line.startswith("f ")
    ]
    # Original +Z and +Y faces become +Z and -X after the authored Z rotation.
    np.testing.assert_allclose(face_normals[0], [[0, 0, 1]] * 3, atol=1e-15)
    np.testing.assert_allclose(face_normals[1], [[-1, 0, 0]] * 3, atol=1e-15)
    assert "s off" in lines


def test_native_visual_obj_uses_an_analytic_cylinder_only_for_appearance(
    tmp_path,
) -> None:
    """The file's tessellated appearance must leave its native collider intact."""
    from embodichain.toolkits.assembly import AssemblyBody, AssemblyGeometry
    from embodichain.lab.sim.assembly._visual import write_visual_mesh
    from embodichain.lab.sim.shapes import CylinderCfg

    body = AssemblyBody(
        name="bolt", geometries=[AssemblyGeometry(kind="cylinder", size=[0.005, 0.01])]
    )
    output = write_visual_mesh(body, tmp_path / "cylinder.obj")
    lines = output.read_text().splitlines()
    vertices = np.array(
        [list(map(float, line.split()[1:])) for line in lines if line.startswith("v ")]
    )
    np.testing.assert_allclose(vertices.min(axis=0), [-0.005, -0.005, -0.01], atol=1e-9)
    np.testing.assert_allclose(vertices.max(axis=0), [0.005, 0.005, 0.01], atol=1e-9)
    normals = np.array(
        [list(map(float, line.split()[1:])) for line in lines if line.startswith("vn ")]
    )
    corners = (
        np.array(
            [
                [list(map(int, corner.split("//"))) for corner in line.split()[1:]]
                for line in lines
                if line.startswith("f ")
            ]
        ).reshape(-1, 2)
        - 1
    )
    positions = vertices[corners[:, 0]]
    directions = normals[corners[:, 1]]
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-7)
    sides = np.abs(directions[:, 2]) < 1e-7
    assert sides.any()
    np.testing.assert_allclose(
        directions[sides, :2], positions[sides, :2] / 0.005, atol=1e-7
    )
    caps = ~sides
    np.testing.assert_allclose(directions[caps, :2], 0, atol=1e-7)
    np.testing.assert_allclose(
        directions[caps, 2], np.sign(positions[caps, 2]), atol=1e-7
    )
    assert isinstance(
        to_rigid_config(body, source_contact=False).shape.parts[0].shape, CylinderCfg
    )


def test_default_source_contacts_require_an_explicit_backend_comparison() -> None:
    pytest.importorskip("mujoco")
    body = import_mjcf(SOURCE).bodies[0]
    with pytest.raises(NotImplementedError, match="cannot preserve MuJoCo solref"):
        to_rigid_config(body, backend="default", source_contact=True)
    cfg = to_rigid_config(body, backend="default", source_contact=False)
    assert cfg.attrs.mass_props.mass == 2.0
