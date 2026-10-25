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
"""Independent geometry, mass ownership and portable manifest regressions."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.assembly import AssemblyAsset, import_mjcf

SOURCE = """<mujoco>
<compiler angle="radian"/>
<worldbody><body name="part" pos=".1 .2 .3" euler=".2 -.3 .4">
<freejoint name="part_joint"/>
<inertial pos=".001 .002 .003" mass="2" diaginertia=".01 .02 .025" euler=".1 .2 .3"/>
<geom name="piece_a" type="box" size=".01 .02 .03" pos="-.02 0 0"/>
<geom name="piece_b" type="cylinder" size=".01 .02" pos=".03 .01 0" euler=".2 .4 -.1"/>
</body></worldbody></mujoco>"""


def test_import_preserves_authored_mass_and_rotated_inertia() -> None:
    pytest.importorskip("mujoco")
    asset = import_mjcf(SOURCE)
    body = asset.bodies[0]
    assert body.mass == 2.0
    np.testing.assert_allclose(body.com, [0.001, 0.002, 0.003], atol=1e-15)
    # MuJoCo's default Euler sequence is intrinsic XYZ, independently formed
    # here rather than reading the converter's compiled inertia back.
    rotation = Rotation.from_euler("XYZ", [0.1, 0.2, 0.3]).as_matrix()
    expected = rotation @ np.diag([0.01, 0.02, 0.025]) @ rotation.T
    np.testing.assert_allclose(body.inertia, expected, atol=1e-14)
    assert len(body.geometries) == 2
    assert body.geometries[1].kind == "cylinder"


def test_compiled_mesh_recentering_preserves_body_local_vertices() -> None:
    """MuJoCo's mesh principal frame must not be applied a second time."""
    pytest.importorskip("mujoco")
    source_vertices = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.11, 0.0, 0.0],
            [0.1, 0.03, 0.0],
            [0.1, 0.0, 0.04],
        ]
    )
    xml = """<mujoco><compiler angle="radian"/>
    <asset><mesh name="offset_tetra" vertex=".1 0 0 .11 0 0 .1 .03 0 .1 0 .04"
      face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset>
    <worldbody><body name="part"><freejoint name="joint"/>
      <geom name="offset_mesh" type="mesh" mesh="offset_tetra" pos=".02 -.03 .04" euler=".2 -.3 .4"/>
    </body></worldbody></mujoco>"""
    geom = import_mjcf(xml).bodies[0].geometries[0]
    expected = source_vertices @ Rotation.from_euler(
        "XYZ", [0.2, -0.3, 0.4]
    ).as_matrix().T + [0.02, -0.03, 0.04]
    actual = (
        np.asarray(geom.vertices) @ Rotation.from_quat(geom.quaternion).as_matrix().T
        + geom.position
    )
    # Convex compilation may permute vertices; compare their body-local set.
    pairwise_distance = np.linalg.norm(actual[:, None] - expected[None], axis=-1)
    assert np.max(np.min(pairwise_distance, axis=1)) < 1e-8
    assert np.max(np.min(pairwise_distance, axis=0)) < 1e-8


def test_manifest_roundtrip_does_not_import_generator_and_detects_tampering(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("mujoco")
    asset = import_mjcf(SOURCE)
    target = asset.save(tmp_path / "asset.json")
    monkeypatch.setitem(sys.modules, "mjfit", None)
    monkeypatch.setitem(sys.modules, "mujoco", None)
    restored = AssemblyAsset.load(target)
    assert restored.content_hash == asset.content_hash
    data = json.loads(target.read_text())
    data["asset"]["bodies"][0]["mass"] = 3.0
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        AssemblyAsset.load(target)


def test_hinge_cannot_silently_become_free_body() -> None:
    pytest.importorskip("mujoco")
    asset = import_mjcf(
        SOURCE.replace(
            '<freejoint name="part_joint"/>', '<joint name="part_joint" type="hinge"/>'
        )
    )
    with pytest.raises(ValueError, match="articulation"):
        asset.to_rigid_configs()


def test_peg_task_floor_is_plate_attached_and_preserves_free_part() -> None:
    pytest.importorskip("mjfit")
    from embodichain.gen_sim.assembly.task_assets import build_task_assembly

    closed = build_task_assembly("peg")
    opened = build_task_assembly("peg", {"bottom_thickness": 0})
    bodies = {body.name: body for body in closed.bodies}
    original = {body.name: body for body in opened.bodies}
    assert set(bodies) == set(original) == {"plate", "peg"}
    plate, peg = bodies["plate"], bodies["peg"]
    assert plate.joint.kind == "fixed" and peg.joint.kind == "free"
    assert len(plate.geometries) == len(original["plate"].geometries) + 1
    floor = next(geometry for geometry in plate.geometries if geometry.name == "bottom")
    assert floor.collision and floor.visible
    vertices = (
        np.asarray(floor.vertices) @ Rotation.from_quat(floor.quaternion).as_matrix().T
        + floor.position
    )
    # The nominal plate is 18 mm deep. Its local underside is -9 mm, and
    # the public generator's 2 mm floor must terminate exactly there.
    np.testing.assert_allclose(vertices[:, 2].max(), -0.009, atol=2e-8)
    np.testing.assert_allclose(np.ptp(vertices[:, 2]), 0.002, atol=2e-8)
    np.testing.assert_allclose(vertices[:, :2].min(axis=0), [-0.07, -0.07], atol=2e-8)
    np.testing.assert_allclose(vertices[:, :2].max(axis=0), [0.07, 0.07], atol=2e-8)
    opening = np.asarray(closed.config["mating_outline"])
    assert np.all(opening >= vertices[:, :2].min(axis=0))
    assert np.all(opening <= vertices[:, :2].max(axis=0))
    assert peg.mass == original["peg"].mass == 0.5
    np.testing.assert_array_equal(peg.inertia, original["peg"].inertia)
    np.testing.assert_array_equal(peg.com, original["peg"].com)
    assert len(peg.geometries) == len(original["peg"].geometries) == 1
    np.testing.assert_array_equal(
        peg.geometries[0].vertices, original["peg"].geometries[0].vertices
    )
    assert not any(
        geometry.name == "bottom" for geometry in original["plate"].geometries
    )


def test_elevated_peg_task_seats_on_local_floor_not_global_ground() -> None:
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("mjfit")
    import xml.etree.ElementTree as ET
    from embodichain.gen_sim.assembly.task_assets import build_task_assembly

    asset = build_task_assembly("peg")
    model_xml = ET.fromstring(asset.source_mjcf)
    # Move the whole assembly 25 cm above the unrelated global ground.
    # This is scene preparation, before constructing or stepping the model.
    for name in ("plate", "peg"):
        body = model_xml.find(f"./worldbody/body[@name='{name}']")
        position = np.fromstring(body.attrib["pos"], sep=" ")
        position[2] += 0.25
        body.set("pos", " ".join(map(str, position)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(model_xml, encoding="unicode"))
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    peg_id, plate_id = model.body("peg").id, model.body("plate").id
    expected_floor_height = model.body_pos[plate_id, 2] - 0.009
    peg = next(body for body in asset.bodies if body.name == "peg")
    geometry = peg.geometries[0]
    vertices = (
        np.asarray(geometry.vertices)
        @ Rotation.from_quat(geometry.quaternion).as_matrix().T
        + geometry.position
    )
    bottom = vertices[np.isclose(vertices[:, 2], vertices[:, 2].min(), atol=1e-8)]
    world_bottom = bottom @ data.xmat[peg_id].reshape(3, 3).T + data.xpos[peg_id]
    assert np.isfinite(data.qpos).all()
    np.testing.assert_allclose(world_bottom[:, 2], expected_floor_height, atol=5e-5)
    floor_id = model.geom("bottom").id
    assert any(floor_id in contact.geom for contact in data.contact[: data.ncon])


def test_hinge_visual_export_keeps_body_local_appearance_and_source_physics(
    tmp_path,
) -> None:
    """One normal-bearing visual must not replace or move the link's colliders."""
    from dataclasses import asdict
    from pathlib import Path
    import xml.etree.ElementTree as ET

    from embodichain.gen_sim.assembly import (
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
    config = AssemblyAsset(bodies=[body]).to_articulation_configs(
        tmp_path, source_contact=False
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


def test_usd_preserves_body_ownership_vertices_and_mass(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    from embodichain.gen_sim.assembly.precision import compare_usd_geometry

    asset = import_mjcf(SOURCE)
    path = asset.write_usd(tmp_path / "part.usda")
    measured = compare_usd_geometry(asset, path)
    assert measured["passed"]
    assert measured["colliders"] == 2
    assert measured["mass_relative_error"] < 1e-6
    assert measured["inertia_relative_error"] < 1e-6


def test_usd_static_compound_has_one_body_not_one_body_per_collider(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    from dexsim.kit.usd import parse_usd

    asset = import_mjcf(SOURCE.replace('<freejoint name="part_joint"/>', ""))
    path = asset.write_usd(tmp_path / "static.usda")
    parsed = parse_usd(str(path), root_name="test")
    assert len(parsed.mesh_objects) == 1
    assert len(parsed.mesh_objects[0].collisions) == 2


def test_usd_precision_detects_changed_primitive_dimensions(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    from embodichain.gen_sim.assembly.precision import compare_usd_geometry

    asset = import_mjcf(SOURCE)
    path = asset.write_usd(tmp_path / "wrong_radius.usda")
    stage = Usd.Stage.Open(str(path))
    cylinder = UsdGeom.Cylinder(stage.GetPrimAtPath("/Assembly/part/geom_1"))
    cylinder.GetRadiusAttr().Set(0.012)  # Source radius is 0.010 m.
    stage.GetRootLayer().Save()

    measured = compare_usd_geometry(asset, path)

    assert not measured["passed"]
    assert measured["geometry_error_m"] >= 0.0019


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


def test_unsupported_equality_constraints_fail_before_conversion() -> None:
    pytest.importorskip("mujoco")
    xml = SOURCE.replace(
        "</mujoco>", '<equality><weld body1="part"/></equality></mujoco>'
    )
    with pytest.raises(ValueError, match="equality"):
        import_mjcf(xml)


def test_nonstandard_collision_masks_cannot_silently_change_filtering() -> None:
    pytest.importorskip("mujoco")
    xml = SOURCE.replace('name="piece_a"', 'name="piece_a" contype="2" conaffinity="2"')
    with pytest.raises(ValueError, match="collision masks"):
        import_mjcf(xml)


def test_contact_priority_and_mixing_survive_native_config_conversion() -> None:
    pytest.importorskip("mujoco")
    xml = SOURCE.replace(
        "<worldbody>", '<default><geom priority="3" solmix="0.7"/></default><worldbody>'
    )
    body = import_mjcf(xml).bodies[0]
    native = body.to_rigid_config()
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
    cfg = body.to_rigid_config(source_contact=False)
    assert isinstance(cfg.shape, CompoundShapeCfg)
    assert len(cfg.shape.parts) == 2
    assert len(cfg.shape.visual_parts) == 3
    assert cfg.attrs.mass_props.mass == 2.0


def test_native_visual_obj_preserves_source_triangles_and_body_ownership(
    tmp_path,
) -> None:
    """File loading must preserve separate surfaces rather than filling their gap."""
    from dataclasses import asdict

    from embodichain.gen_sim.assembly import AssemblyBody, AssemblyGeometry
    from embodichain.gen_sim.assembly._visual import write_visual_mesh

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
    from embodichain.gen_sim.assembly import AssemblyBody, AssemblyGeometry
    from embodichain.gen_sim.assembly._visual import write_visual_mesh

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
    from embodichain.gen_sim.assembly import AssemblyBody, AssemblyGeometry
    from embodichain.gen_sim.assembly._visual import write_visual_mesh
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
        body.to_rigid_config(source_contact=False).shape.parts[0].shape, CylinderCfg
    )


def test_usd_mesh_keeps_sharp_face_normals_in_geometry_local_frame(tmp_path) -> None:
    """USD shading must retain hard edges without applying the geom pose twice."""
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    from embodichain.gen_sim.assembly import AssemblyBody, AssemblyGeometry

    geometry = AssemblyGeometry(
        vertices=[[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
        triangles=[[0, 1, 2], [0, 3, 1]],
        position=[1, 2, 3],
        quaternion=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
    )
    body = AssemblyBody(name="hard_corner", geometries=[geometry])
    path = AssemblyAsset(bodies=[body]).write_usd(tmp_path / "corner.usda")
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Assembly/hard_corner/geom_0"))

    np.testing.assert_allclose(mesh.GetPointsAttr().Get(), geometry.vertices)
    np.testing.assert_array_equal(
        mesh.GetFaceVertexIndicesAttr().Get(), [0, 1, 2, 0, 3, 1]
    )
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
    normals = np.asarray(mesh.GetNormalsAttr().Get())
    # The two source planes face +Z and +Y; their common edge stays sharp.
    np.testing.assert_allclose(normals, [[0, 0, 1]] * 3 + [[0, 1, 0]] * 3)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    world_normals = [
        transform.TransformDir(Gf.Vec3d(*normal.tolist())) for normal in normals
    ]
    # Geometry rotation, not authored normal data, maps +Y to world -X.
    np.testing.assert_allclose(
        world_normals, [[0, 0, 1]] * 3 + [[-1, 0, 0]] * 3, atol=1e-6
    )


def test_default_source_contacts_require_an_explicit_backend_comparison() -> None:
    pytest.importorskip("mujoco")
    body = import_mjcf(SOURCE).bodies[0]
    with pytest.raises(NotImplementedError, match="cannot preserve MuJoCo solref"):
        body.to_rigid_config(backend="default", source_contact=True)
    cfg = body.to_rigid_config(backend="default", source_contact=False)
    assert cfg.attrs.mass_props.mass == 2.0


def test_precision_host_matches_source_floor_and_disables_missing_floor() -> None:
    pytest.importorskip("mujoco")
    from embodichain.gen_sim.assembly.precision import (
        PrecisionRunCfg,
        _source_floor_attributes,
    )

    cfg = PrecisionRunCfg(backend="newton", source_contact=True)
    enabled, attrs = _source_floor_attributes(import_mjcf(SOURCE), cfg)
    assert not enabled and attrs is None
    floor_source = SOURCE.replace(
        "<worldbody>",
        '<worldbody><geom name="floor" type="plane" size="1 1 .01" friction="1 .01 .001" solref=".002 1" solimp=".99 .999 .001 .5 2"/>',
    )
    asset = import_mjcf(floor_source)
    enabled, attrs = _source_floor_attributes(asset, cfg)
    assert len(asset.bodies) == 1  # Ground remains a host-owned scene property.
    assert enabled
    assert attrs.material_props.dynamic_friction == pytest.approx(1)
    assert attrs.material_props.torsional_friction == pytest.approx(0.01)
    assert attrs.collision_props.solref == pytest.approx([0.002, 1])
    assert attrs.collision_props.solimp == pytest.approx([0.99, 0.999, 0.001, 0.5, 2])
    assert attrs.collision_props.contact_offset == 0


@pytest.mark.parametrize("backend", ["unknown", "fake"])
def test_invalid_probe_backend_is_not_reported_as_skipped(backend) -> None:
    from embodichain.gen_sim.assembly.precision import (
        PrecisionRunCfg,
        run_passive_probe,
    )

    with pytest.raises(ValueError, match="Unknown precision backend"):
        run_passive_probe(AssemblyAsset(), PrecisionRunCfg(backend=backend))


@pytest.mark.parametrize("scale,bottom", [(1.0, 0.0), (2.0, 0.004)])
def test_peg_drop_distinguishes_seating_from_rim_and_outside_falls(
    scale: float, bottom: float
) -> None:
    """A tiny penetration after the spawn gap must not become drop success."""
    from embodichain.gen_sim.assembly.precision import PrecisionRunCfg, _score

    # Pinned peg() defaults: 10 mm plate, 30 mm insertion section, 5 mm
    # spawn gap. The blind-hole bottom and all dimensions scale together.
    thickness, length, gap = np.array([0.01, 0.03, 0.005]) * scale
    bottom *= scale
    plate = np.array([0.0, 0.0, bottom + thickness / 2])
    initial_peg = np.array([0.0, 0.0, bottom + thickness + gap + length / 2])
    seated = [0.0, 0.0, bottom + length / 2]
    rim_stop = [0.02 * scale, 0.0, bottom + thickness + length / 2 - 1e-7]
    outside_fall = [0.05 * scale, 0.0, bottom + length / 2]
    asset = AssemblyAsset(config={"kind": "peg", "scale": scale})

    measured = _score(
        asset,
        {"peg": initial_peg, "plate": plate},
        {"peg": np.array([seated, rim_stop, outside_fall]), "plate": plate},
        PrecisionRunCfg(),
    )

    # Every endpoint fell farther than 5 mm, but only the centered peg reached
    # the bottom of the bore. This expected count is independent of scoring.
    assert measured["success_rate"] == pytest.approx(1 / 3)
    assert measured["required_insertion_depth_m"] == pytest.approx(thickness)


def test_many_pegs_preserves_official_sampling_prefixes_and_grid_placements(
    tmp_path,
) -> None:
    """A grid is distinct bodies with their own source geometry and inertia."""
    mujoco = pytest.importorskip("mujoco")
    mjfit = pytest.importorskip("mjfit")
    from embodichain.gen_sim.assembly import build_many_pegs

    asset = build_many_pegs(2, seed=0, family="primitive", cache_dir=tmp_path)
    configs = mjfit.sample_pegs(2, seed=0, family="primitive")
    source = mjfit.grid(
        [mjfit.build(config, standalone=False) for config in configs], spacing=0.3
    )
    source.compile()
    model = mujoco.MjModel.from_xml_string(source.to_xml())

    assert [item["config"] for item in asset.config["assemblies"]] == configs
    assert [body.name for body in asset.bodies] == [
        "p0_plate",
        "p0_peg",
        "p1_plate",
        "p1_peg",
    ]
    # Two cells in the source's square grid occupy the top row, not two
    # coincident rigid facades or independent simulated arenas.
    np.testing.assert_allclose(
        [item["position"] for item in asset.config["assemblies"]],
        [[-0.15, 0.15, 0], [0.15, 0.15, 0]],
        atol=1e-15,
    )
    assert sum(len(body.geometries) for body in asset.bodies) == model.ngeom - 1
    for body in asset.bodies:
        source_body = model.body(body.name)
        assert body.mass == source_body.mass[0]
        np.testing.assert_allclose(body.position, source_body.pos, atol=1e-15)
        assert body.joint.kind == ("free" if body.name.endswith("peg") else "fixed")
    assert AssemblyAsset.load(next(tmp_path.glob("*.json"))).content_hash == (
        asset.content_hash
    )

    # The primitive sampler does not accept max_error; it belongs to the
    # public curved shape spec. Its tighter curve approximation must
    # produce more collider pieces while retaining the same sampled outline.
    precise = build_many_pegs(
        2, seed=0, family="primitive", geometry_parameters={"max_error": 1e-5}
    )
    assert precise.config["geometry_parameters"] == {"max_error": 1e-5}
    assert precise.config["assemblies"][0]["sampled_config"] == configs[0]
    assert precise.config["assemblies"][0]["config"]["hole"] == {
        **configs[0]["hole"],
        "max_error": 1e-5,
    }
    assert sum(len(body.geometries) for body in precise.bodies) > model.ngeom - 1


def test_many_pegs_scores_each_translated_cell_and_rejects_rim_or_floor_passage() -> (
    None
):
    """Heterogeneous cells cannot use one global plate position or thickness."""
    from embodichain.gen_sim.assembly.many import _score_many_pegs

    cells = [
        ([-0.3, 0.3, 0.005], 0.01, 0.03),
        ([0.3, -0.3, 0.01], 0.02, 0.06),
    ]
    assemblies, initial, final = [], {}, {}
    for index, (plate, thickness, length) in enumerate(cells):
        plate_name, peg_name = f"p{index}_plate", f"p{index}_peg"
        assemblies.append(
            {
                "index": index,
                "plate_body": plate_name,
                "peg_body": peg_name,
                "config": {"plate_thickness": thickness, "peg_length": length},
            }
        )
        initial[plate_name] = final[plate_name] = np.array(plate)
        initial[peg_name] = np.array(
            [plate[0], plate[1], thickness + 0.005 + length / 2]
        )
        # Four independent endpoint cases: fully seated, on the rim, fallen
        # outside the plate, and tunneled through the floor.
        final[peg_name] = np.array(
            [
                [plate[0], plate[1], length / 2],
                [plate[0] + 0.02, plate[1], thickness + length / 2],
                [plate[0] + 0.10, plate[1], length / 2],
                [plate[0], plate[1], length / 2 - 0.01],
            ]
        )
    measured = _score_many_pegs(
        AssemblyAsset(config={"kind": "many_pegs", "assemblies": assemblies}),
        initial,
        final,
    )

    assert measured["assembly_count"] == 2
    assert measured["success_rate"] == pytest.approx(0.25)
    assert [row["success_rate"] for row in measured["per_assembly"]] == [0.25, 0.25]
    assert [row["required_insertion_depth_m"] for row in measured["per_assembly"]] == [
        0.01,
        0.02,
    ]


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"backend": "default", "solver": "cg"}, "Default does not accept"),
        ({"warp_reference_initialization": "unknown"}, "initialization must be"),
        (
            {
                "backend": "mujoco_cpu",
                "warp_reference_initialization": "warp_recomputed",
            },
            "only for MuJoCo Warp references",
        ),
        (
            {"backend": "newton", "warp_reference_initialization": "warp_recomputed"},
            "only for MuJoCo Warp references",
        ),
        ({"backend": "default", "solver_tolerance": 1e-6}, "Default does not accept"),
        ({"backend": "default", "enable_multiccd": False}, "Default does not accept"),
        ({"solver_tolerance": -1.0}, "tolerance must be finite and non-negative"),
        (
            {"solver_tolerance": float("inf")},
            "tolerance must be finite and non-negative",
        ),
        (
            {"solver_tolerance": float("nan")},
            "tolerance must be finite and non-negative",
        ),
        ({"enable_multiccd": "false"}, "enable_multiccd must be a bool"),
        ({"backend": "mujoco_cpu", "use_mujoco_cpu": True}, "Newton CPU integration"),
        ({"backend": "newton", "use_mujoco_cpu": True, "num_envs": 2}, "one world"),
    ],
)
def test_precision_solver_comparisons_reject_ignored_or_unsupported_settings(
    settings, message
) -> None:
    from embodichain.gen_sim.assembly.precision import (
        PrecisionRunCfg,
        run_passive_probe,
    )

    # These errors must be diagnosed before importing or starting a simulator.
    with pytest.raises(ValueError, match=message):
        run_passive_probe(AssemblyAsset(), PrecisionRunCfg(**settings))


@pytest.mark.parametrize(
    ("source_multiccd", "overrides", "expected_tolerance", "expected_multiccd"),
    [
        (True, {}, 7e-9, True),
        (False, {}, 7e-9, False),
        (False, {"solver_tolerance": 0.0, "enable_multiccd": True}, 0.0, True),
        (True, {"solver_tolerance": 1e-6, "enable_multiccd": False}, 1e-6, False),
    ],
)
def test_precision_reference_preserves_source_options_unless_explicitly_overridden(
    source_multiccd, overrides, expected_tolerance, expected_multiccd
) -> None:
    """Unspecified calibration knobs must not rewrite the recorded source baseline."""
    mujoco = pytest.importorskip("mujoco")
    from embodichain.gen_sim.assembly.precision import PrecisionRunCfg, _run_mujoco

    flag = "enable" if source_multiccd else "disable"
    source = SOURCE.replace(
        "<mujoco>",
        f'<mujoco><option tolerance="7e-9"><flag multiccd="{flag}"/></option>',
    )
    asset = import_mjcf(source, config={"kind": "options_inspection"})

    class OptionsInspected(Exception):
        pass

    def inspect(model, data, step):
        assert step == 0
        assert model.opt.tolerance == expected_tolerance
        multiccd_disabled = bool(
            model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
        )
        assert multiccd_disabled is not expected_multiccd
        raise OptionsInspected

    with pytest.raises(OptionsInspected):
        _run_mujoco(asset, PrecisionRunCfg(**overrides), on_step=inspect)


def test_failed_precision_measurements_preserve_json_and_failure_reason(
    monkeypatch,
) -> None:
    from embodichain.gen_sim.assembly import precision

    # Real sub-micron meshes can fail backend cooking; the matrix must retain
    # that diagnostic instead of terminating without a result file.
    def unsupported(*args):
        raise ValueError("planar mesh collider")

    monkeypatch.setattr(precision, "_run_mujoco", unsupported)
    result = precision.run_passive_probe(AssemblyAsset(), precision.PrecisionRunCfg())
    assert result["status"] == "failed"
    assert "planar mesh collider" in result["reason"]

    def nonfinite(*args):
        return {
            "status": "failed",
            "metrics": {"finite": False, "distance": float("nan")},
            "final_state": {"peg": [0.0, float("inf"), 0.0]},
        }

    monkeypatch.setattr(precision, "_run_mujoco", nonfinite)
    result = precision.run_passive_probe(AssemblyAsset(), precision.PrecisionRunCfg())
    json.dumps(result, allow_nan=False)
    assert result["status"] == "failed"
    assert result["metrics"]["finite"] is False
    assert result["metrics"]["distance"] is None
    assert result["final_state"]["peg"] == [0.0, None, 0.0]
    assert result["nonfinite_measurement_count"] == 2
