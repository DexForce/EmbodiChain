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
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from embodichain.gen_sim.task_engine._task_program import twist_geometry
from embodichain.gen_sim.task_engine._task_program.twist_binding import (
    KnobBinding,
    discover_twist,
)

_POINTS = np.asarray(
    [(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.02, 0.0), (0.0, 0.0, 0.03)]
)
_FACES = np.asarray([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)])
_ROTATE_Z_90 = np.asarray([(0, -1, 0), (1, 0, 0), (0, 0, 1)])
_GRIP_LOCAL_TRANSLATION = np.asarray([0.005, -0.004, 0.003])
_GRIP_GROUP_TRANSLATION = np.asarray([0.04, 0.05, 0.06])
_PANEL_LOCAL_TRANSLATION = np.asarray([0.001, 0.002, 0.003])
_PANEL_GROUP_TRANSLATION = np.asarray([-0.04, 0.03, -0.02])
_SOURCE_CALIBRATION = (
    "knob_rotation",
    "grip",
    "pointer",
    {0: "label0", 90: "label90"},
)


def _mesh(
    stage: Usd.Stage,
    path: str,
    *,
    points: np.ndarray = _POINTS,
    collision: bool = True,
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(_FACES))
    mesh.CreateFaceVertexIndicesAttr(_FACES.reshape(-1).tolist())
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def _qualify(fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> KnobBinding:
    fixture.stage.GetRootLayer().Save()
    digest = hashlib.sha256(fixture.path.read_bytes()).hexdigest()
    monkeypatch.setitem(twist_geometry._CALIBRATIONS, digest, _SOURCE_CALIBRATION)
    return replace(fixture.binding, source_sha256=digest)


@pytest.fixture
def geometry_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    path = tmp_path / "knob.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Control")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root.AddTranslateOp().Set(Gf.Vec3d(2.0, 4.0, 6.0))
    root.AddRotateZOp().Set(15.0)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name, translation, angle in (
        ("base", (0.1, 0.2, 0.3), 30.0),
        ("knob", (-0.1, -0.2, 0.4), -25.0),
    ):
        body = UsdGeom.Xform.Define(stage, "/Control/" + name)
        body.AddTranslateOp().Set(Gf.Vec3d(*translation))
        body.AddRotateZOp().Set(angle)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

    grip_group = UsdGeom.Xform.Define(stage, "/Control/knob/grip_group")
    grip_group.AddTranslateOp().Set(Gf.Vec3d(*_GRIP_GROUP_TRANSLATION))
    grip_group.AddRotateZOp().Set(90.0)
    grip_group.AddScaleOp().Set(Gf.Vec3f(2.0))
    grip = _mesh(stage, "/Control/knob/grip_group/grip")
    grip.AddTranslateOp().Set(Gf.Vec3d(*_GRIP_LOCAL_TRANSLATION))
    panel_group = UsdGeom.Xform.Define(stage, "/Control/base/panel_group")
    panel_group.AddTranslateOp().Set(Gf.Vec3d(*_PANEL_GROUP_TRANSLATION))
    panel_group.AddRotateZOp().Set(-90.0)
    panel_group.AddScaleOp().Set(Gf.Vec3f(0.5))
    panel = _mesh(stage, "/Control/base/panel_group/panel")
    panel.AddTranslateOp().Set(Gf.Vec3d(*_PANEL_LOCAL_TRANSLATION))
    cap = _mesh(stage, "/Control/knob/cap")
    cap.AddTranslateOp().Set(Gf.Vec3d(0.3, 0.0, 0.0))
    render = _mesh(stage, "/Control/knob/render_only", collision=False)
    render.AddTranslateOp().Set(Gf.Vec3d(100.0, 100.0, 100.0))
    disabled = _mesh(stage, "/Control/knob/disabled_collider")
    UsdPhysics.CollisionAPI(disabled.GetPrim()).CreateCollisionEnabledAttr(False)
    nested = UsdGeom.Xform.Define(stage, "/Control/knob/other_body")
    UsdPhysics.RigidBodyAPI.Apply(nested.GetPrim())
    _mesh(stage, "/Control/knob/other_body/unowned_collider")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Control/knob_rotation")
    joint.CreateBody0Rel().SetTargets(["/Control/base"])
    joint.CreateBody1Rel().SetTargets(["/Control/knob"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-180.0)
    joint.CreateUpperLimitAttr(180.0)
    binding = KnobBinding(
        object_id="control",
        joint="knob_rotation",
        link="knob",
        parent="base",
        source_sha256="",
        scale=1.0,
        axis=(0.0, 0.0, -1.0),
        axis_sign=-1.0,
        origin=(0.0, 0.0, 0.0),
        outer_point=(0.0, 0.0, 0.1),
        grip_depth=0.03,
        grip_width=0.04,
        limits=(-np.pi, np.pi),
        target_setting=90,
        target_qpos=0.0,
    )
    fixture = SimpleNamespace(path=path, stage=stage, binding=binding)
    fixture.binding = _qualify(fixture, monkeypatch)
    return fixture


def _load(fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch):
    return twist_geometry.load_twist_geometry(
        fixture.path, _qualify(fixture, monkeypatch)
    )


@pytest.mark.parametrize("scale", [0.4, 1.0, 1.736111111, 2.5])
def test_authored_mesh_transform_and_body_scale_apply_once(geometry_source, scale):
    binding = replace(geometry_source.binding, scale=scale)
    geometry = twist_geometry.load_twist_geometry(geometry_source.path, binding)
    expected = (
        (2.0 * (_POINTS + _GRIP_LOCAL_TRANSLATION)) @ _ROTATE_Z_90.T
        + _GRIP_GROUP_TRANSLATION
    ) * scale
    np.testing.assert_allclose(geometry.grip.vertices, expected, atol=1e-9)
    assert geometry.scale == scale


def test_parent_collision_retains_its_own_rigid_body_frame(geometry_source):
    geometry = twist_geometry.load_twist_geometry(
        geometry_source.path, geometry_source.binding
    )
    expected = (
        0.5 * (_POINTS + _PANEL_LOCAL_TRANSLATION)
    ) @ _ROTATE_Z_90 + _PANEL_GROUP_TRANSLATION
    assert len(geometry.parent_collisions) == 1
    np.testing.assert_allclose(
        geometry.parent_collisions[0].vertices, expected, atol=1e-9
    )
    assert geometry.parent_collisions[0].owner_path == "/Control/base"


def test_root_authored_world_placement_does_not_rebase_link_geometry(
    geometry_source, monkeypatch
):
    original = _load(geometry_source, monkeypatch)
    root = UsdGeom.Xformable(geometry_source.stage.GetPrimAtPath("/Control"))
    root.GetOrderedXformOps()[0].Set(Gf.Vec3d(-17.0, 23.0, 4.0))
    root.GetOrderedXformOps()[1].Set(-65.0)
    changed = _load(geometry_source, monkeypatch)
    np.testing.assert_allclose(changed.grip.vertices, original.grip.vertices, atol=1e-9)
    np.testing.assert_allclose(
        changed.parent_collisions[0].vertices,
        original.parent_collisions[0].vertices,
        atol=1e-9,
    )


def test_collision_inventory_excludes_render_disabled_and_other_rigid_owners(
    geometry_source,
):
    geometry = twist_geometry.load_twist_geometry(
        geometry_source.path, geometry_source.binding
    )
    assert {mesh.path for mesh in geometry.target_collisions} == {
        "/Control/knob/grip_group/grip",
        "/Control/knob/cap",
    }
    assert all(
        mesh.owner_path == "/Control/knob" for mesh in geometry.target_collisions
    )
    assert geometry.target_body_path == "/Control/knob"
    assert geometry.parent_body_path == "/Control/base"


def test_articulation_clearance_inventory_preserves_other_link_owners(
    geometry_source,
):
    geometry = twist_geometry.load_twist_geometry(
        geometry_source.path, geometry_source.binding
    )
    by_path = {mesh.path: mesh for mesh in geometry.articulation_collisions}
    assert set(by_path) == {
        "/Control/knob/grip_group/grip",
        "/Control/knob/cap",
        "/Control/base/panel_group/panel",
        "/Control/knob/other_body/unowned_collider",
    }
    nested = by_path["/Control/knob/other_body/unowned_collider"]
    assert nested.owner_path == "/Control/knob/other_body"
    np.testing.assert_allclose(nested.vertices, _POINTS, atol=1e-9)


def test_grip_proxy_is_calibrated_mesh_not_full_collision_union(geometry_source):
    geometry = twist_geometry.load_twist_geometry(
        geometry_source.path, geometry_source.binding
    )
    assert geometry.grip.path == "/Control/knob/grip_group/grip"
    assert len(geometry.grip.vertices) == len(_POINTS)
    assert geometry.grip.vertices[:, 0].max() < 0.1
    assert max(mesh.vertices[:, 0].max() for mesh in geometry.target_collisions) > 0.3


def test_loaded_arrays_are_readonly_and_preserve_valid_triangles(geometry_source):
    geometry = twist_geometry.load_twist_geometry(
        geometry_source.path, geometry_source.binding
    )
    for mesh in (
        geometry.grip,
        *geometry.target_collisions,
        *geometry.parent_collisions,
    ):
        assert mesh.vertices.dtype == np.float64
        assert mesh.faces.dtype == np.int64
        assert not mesh.vertices.flags.writeable
        assert not mesh.faces.flags.writeable
        np.testing.assert_array_equal(mesh.faces, _FACES)


def test_source_sha_is_checked_before_cached_or_loaded_geometry(geometry_source):
    binding = replace(geometry_source.binding, source_sha256="0" * 64)
    with pytest.raises(ValueError):
        twist_geometry.load_twist_geometry(geometry_source.path, binding)


def test_changed_source_bytes_invalidate_previously_loaded_binding(geometry_source):
    twist_geometry.load_twist_geometry(geometry_source.path, geometry_source.binding)
    geometry_source.stage.SetMetadata("comment", "Source changed after binding.")
    geometry_source.stage.GetRootLayer().Save()
    with pytest.raises(ValueError):
        twist_geometry.load_twist_geometry(
            geometry_source.path, geometry_source.binding
        )


def test_matching_source_sha_without_audited_calibration_is_rejected(
    geometry_source, monkeypatch
):
    monkeypatch.delitem(
        twist_geometry._CALIBRATIONS, geometry_source.binding.source_sha256
    )
    with pytest.raises(ValueError):
        twist_geometry.load_twist_geometry(
            geometry_source.path, geometry_source.binding
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf, True])
def test_invalid_deployment_scale_fails_closed(geometry_source, scale):
    with pytest.raises(ValueError):
        twist_geometry.load_twist_geometry(
            geometry_source.path, replace(geometry_source.binding, scale=scale)
        )


@pytest.mark.parametrize(
    "field,value", [("joint", "other"), ("link", "base"), ("parent", "knob")]
)
def test_binding_joint_and_body_identity_cannot_be_substituted(
    geometry_source, field, value
):
    with pytest.raises(ValueError):
        twist_geometry.load_twist_geometry(
            geometry_source.path, replace(geometry_source.binding, **{field: value})
        )


@pytest.mark.parametrize("body", ["base", "knob"])
def test_selected_body_requires_rigid_ownership(geometry_source, monkeypatch, body):
    geometry_source.stage.GetPrimAtPath("/Control/" + body).RemoveAPI(
        UsdPhysics.RigidBodyAPI
    )
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


@pytest.mark.parametrize("kind", ["missing", "render_only", "disabled", "other_owner"])
def test_calibrated_grip_must_be_enabled_target_owned_collision_mesh(
    geometry_source, monkeypatch, kind
):
    path = "/Control/knob/grip_group/grip"
    prim = geometry_source.stage.GetPrimAtPath(path)
    if kind == "missing":
        geometry_source.stage.RemovePrim(path)
    elif kind == "render_only":
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
    elif kind == "disabled":
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    else:
        UsdPhysics.RigidBodyAPI.Apply(prim.GetParent())
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


def test_empty_enabled_parent_collision_inventory_fails_closed(
    geometry_source, monkeypatch
):
    prim = geometry_source.stage.GetPrimAtPath("/Control/base/panel_group/panel")
    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


def test_unsupported_enabled_collision_primitive_cannot_be_silently_omitted(
    geometry_source, monkeypatch
):
    cube = UsdGeom.Cube.Define(geometry_source.stage, "/Control/base/cube")
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


@pytest.mark.parametrize("role", ["joint", "target_body", "parent_body"])
def test_disabled_bound_joint_or_rigid_body_fails_closed(
    geometry_source, monkeypatch, role
):
    if role == "joint":
        joint = UsdPhysics.RevoluteJoint(
            geometry_source.stage.GetPrimAtPath("/Control/knob_rotation")
        )
        joint.CreateJointEnabledAttr(False)
    else:
        name = "knob" if role == "target_body" else "base"
        api = UsdPhysics.RigidBodyAPI(
            geometry_source.stage.GetPrimAtPath("/Control/" + name)
        )
        api.CreateRigidBodyEnabledAttr(False)
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


def test_duplicate_calibrated_grip_name_is_rejected(geometry_source, monkeypatch):
    _mesh(geometry_source.stage, "/Control/knob/duplicate/grip")
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


def test_collision_approximation_metadata_is_not_silently_reinterpreted(
    geometry_source, monkeypatch
):
    prim = geometry_source.stage.GetPrimAtPath("/Control/knob/grip_group/grip")
    api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    api.CreateApproximationAttr("convexDecomposition")
    geometry = _load(geometry_source, monkeypatch)
    assert geometry.grip.approximation == "convexDecomposition"


def test_unhashed_external_layer_cannot_supply_collision_geometry(
    geometry_source, monkeypatch, tmp_path
):
    external_path = tmp_path / "external.usda"
    external = Usd.Stage.CreateNew(str(external_path))
    _mesh(external, "/Control/base/external_collider")
    external.GetRootLayer().Save()
    geometry_source.stage.GetRootLayer().subLayerPaths.append(str(external_path))
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


def test_non_metre_authored_stage_units_fail_closed(geometry_source, monkeypatch):
    UsdGeom.SetStageMetersPerUnit(geometry_source.stage, 0.01)
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


def test_stage_up_axis_does_not_rotate_native_link_local_mesh_again(
    geometry_source, monkeypatch
):
    original = _load(geometry_source, monkeypatch)
    UsdGeom.SetStageUpAxis(geometry_source.stage, "Y")
    changed = _load(geometry_source, monkeypatch)
    np.testing.assert_allclose(changed.grip.vertices, original.grip.vertices)
    np.testing.assert_allclose(
        changed.parent_collisions[0].vertices, original.parent_collisions[0].vertices
    )
    # Runtime native poses already carry the deployment's stage-axis conversion.
    native_pose = np.eye(4)
    native_pose[:3, :3] = np.asarray([(1, 0, 0), (0, 0, -1), (0, 1, 0)])
    native_pose[:3, 3] = [0.4, 0.5, 0.6]
    world = changed.grip.vertices @ native_pose[:3, :3].T + native_pose[:3, 3]
    expected = original.grip.vertices @ native_pose[:3, :3].T + native_pose[:3, 3]
    np.testing.assert_allclose(world, expected)


@pytest.mark.parametrize("scale", [(2.0, 1.0, 2.0), (-1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
def test_unsupported_mesh_transform_does_not_receive_guessed_geometry(
    geometry_source, monkeypatch, scale
):
    group = UsdGeom.Xformable(
        geometry_source.stage.GetPrimAtPath("/Control/knob/grip_group")
    )
    group.GetOrderedXformOps()[2].Set(Gf.Vec3f(*scale))
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


@pytest.mark.parametrize("kind", ["shear", "body_scale"])
def test_non_rigid_link_frame_or_mesh_shear_is_rejected(
    geometry_source, monkeypatch, kind
):
    if kind == "body_scale":
        body = UsdGeom.Xformable(geometry_source.stage.GetPrimAtPath("/Control/knob"))
        body.AddScaleOp().Set(Gf.Vec3f(2.0))
    else:
        group = UsdGeom.Xformable(
            geometry_source.stage.GetPrimAtPath("/Control/knob/grip_group")
        )
        matrix = Gf.Matrix4d(1.0)
        matrix[0, 1] = 0.2
        group.AddTransformOp().Set(matrix)
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


@pytest.mark.parametrize(
    "points,counts,indices",
    [
        ([], [], []),
        (_POINTS.tolist(), [], []),
        (_POINTS.tolist(), [3], [0, 1]),
        (_POINTS.tolist(), [2], [0, 1]),
        (_POINTS.tolist(), [3], [0, 1, 8]),
        (_POINTS.tolist(), [3], [0, 1, -1]),
        (_POINTS.tolist(), [3], [0, 1, 1]),
        ([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [3], [0, 1, 2]),
        ([[0, 0, 0], [1, 0, 0], [0, np.nan, 0]], [3], [0, 1, 2]),
    ],
)
def test_invalid_collision_topology_fails_closed(
    geometry_source, monkeypatch, points, counts, indices
):
    mesh = UsdGeom.Mesh(
        geometry_source.stage.GetPrimAtPath("/Control/knob/grip_group/grip")
    )
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


def test_convex_polygon_is_triangulated_without_changing_its_bounds(
    geometry_source, monkeypatch
):
    mesh = UsdGeom.Mesh(
        geometry_source.stage.GetPrimAtPath("/Control/knob/grip_group/grip")
    )
    points = np.asarray([(0, 0, 0), (0.02, 0, 0), (0.02, 0.03, 0), (0, 0.03, 0)])
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    geometry = _load(geometry_source, monkeypatch)
    assert geometry.grip.faces.shape == (2, 3)
    triangles = geometry.grip.vertices[geometry.grip.faces]
    area = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        ).sum()
        / 2.0
    )
    assert area == pytest.approx(0.02 * 0.03 * 2.0**2)


def test_concave_polygon_cannot_use_unvalidated_fan_triangulation(
    geometry_source, monkeypatch
):
    mesh = UsdGeom.Mesh(
        geometry_source.stage.GetPrimAtPath("/Control/knob/grip_group/grip")
    )
    mesh.CreatePointsAttr([(0, 0, 0), (2, 0, 0), (1, 0.5, 0), (2, 2, 0), (0, 2, 0)])
    mesh.CreateFaceVertexCountsAttr([5])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


def test_authored_mesh_holes_cannot_be_filled_by_collision_proxy(
    geometry_source, monkeypatch
):
    mesh = UsdGeom.Mesh(
        geometry_source.stage.GetPrimAtPath("/Control/knob/grip_group/grip")
    )
    mesh.CreateHoleIndicesAttr([0])
    with pytest.raises(ValueError):
        _load(geometry_source, monkeypatch)


@pytest.fixture
def robot_source(tmp_path: Path) -> Path:
    path = tmp_path / "gripper.urdf"
    path.write_text(
        """<robot name="test_gripper">
  <link name="pad">
    <visual><origin xyz="100 100 100"/>
      <geometry><box size="20 20 20"/></geometry>
    </visual>
    <collision><origin xyz="0.01 0.02 0.03" rpy="0 0 1.5707963267948966"/>
      <geometry><box size="0.02 0.04 0.06"/></geometry>
    </collision>
  </link>
  <link name="visual_only">
    <visual><geometry><box size="1 1 1"/></geometry></visual>
  </link>
</robot>""",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("scale", [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0)])
def test_robot_collision_origin_rotation_and_body_scale_apply_once(robot_source, scale):
    meshes = twist_geometry.load_robot_link_meshes(robot_source, ("pad",), scale)
    assert set(meshes) == {"pad"}
    assert len(meshes["pad"]) == 1
    vertices = meshes["pad"][0].vertices
    # Ninety degrees swaps X/Y half extents before collision-origin/body scaling.
    expected_min = np.asarray([-0.02, -0.01, -0.03]) + [0.01, 0.02, 0.03]
    expected_max = np.asarray([0.02, 0.01, 0.03]) + [0.01, 0.02, 0.03]
    np.testing.assert_allclose(vertices.min(0), expected_min * scale, atol=1e-12)
    np.testing.assert_allclose(vertices.max(0), expected_max * scale, atol=1e-12)
    assert meshes["pad"][0].owner_path == "pad"
    assert meshes["pad"][0].approximation == "urdfBoxInput"
    assert not vertices.flags.writeable


def test_robot_mesh_scale_precedes_origin_and_body_scale(tmp_path):
    import trimesh

    shape = trimesh.creation.box(extents=[0.02, 0.04, 0.06])
    shape.export(tmp_path / "pad.stl")
    path = tmp_path / "gripper.urdf"
    path.write_text(
        """<robot name="test_gripper"><link name="pad"><collision>
  <origin xyz="0.01 0.02 0.03" rpy="0 0 1.5707963267948966"/>
  <geometry><mesh filename="pad.stl" scale="2 3 4"/></geometry>
</collision></link></robot>""",
        encoding="utf-8",
    )
    body_scale = (2.0, 3.0, 4.0)
    geometry = twist_geometry.load_robot_link_meshes(path, ("pad",), body_scale)
    vertices = geometry["pad"][0].vertices
    expected_min = (np.asarray([-0.06, -0.02, -0.12]) + [0.01, 0.02, 0.03]) * body_scale
    expected_max = (np.asarray([0.06, 0.02, 0.12]) + [0.01, 0.02, 0.03]) * body_scale
    np.testing.assert_allclose(vertices.min(0), expected_min, atol=1e-8)
    np.testing.assert_allclose(vertices.max(0), expected_max, atol=1e-8)
    assert geometry["pad"][0].approximation == "urdfMeshInput"


@pytest.mark.parametrize("names", [("visual_only",), ("missing",), ("pad", "pad"), ()])
def test_robot_geometry_does_not_fallback_to_visual_or_substitute_links(
    robot_source, names
):
    with pytest.raises(ValueError):
        twist_geometry.load_robot_link_meshes(robot_source, names)


@pytest.mark.parametrize("scale", [(0, 1, 1), (-1, 1, 1), (np.nan, 1, 1), (1, 1)])
def test_invalid_robot_body_scale_is_rejected(robot_source, scale):
    with pytest.raises(ValueError):
        twist_geometry.load_robot_link_meshes(robot_source, ("pad",), scale)


@pytest.mark.parametrize(
    "geometry",
    [
        '<sphere radius="0.01"/>',
        '<cylinder radius="0.01" length="0.02"/>',
        '<mesh filename="package://gripper/pad.stl"/>',
        '<box size="0 -1 1"/>',
        '<box size="1 nan 1"/>',
        '<box size="1 1"/>',
        '<box size="1 1 1"/><box size="1 1 1"/>',
        "",
    ],
)
def test_invalid_or_unsupported_robot_collision_geometry_fails_closed(
    tmp_path, geometry
):
    path = tmp_path / "gripper.urdf"
    path.write_text(
        '<robot name="test"><link name="pad"><collision><geometry>'
        + geometry
        + "</geometry></collision></link></robot>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        twist_geometry.load_robot_link_meshes(path, ("pad",))


def test_cached_dual_franka_pads_load_collision_inputs_without_world():
    from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

    assembly = "Panda_robotiq_arg2f_140_Panda_robotiq_arg2f_140"
    path = (
        Path(EMBODICHAIN_DEFAULT_DATA_ROOT)
        / "assembled"
        / assembly
        / (assembly + ".urdf")
    )
    if not path.is_file():
        pytest.skip("Optional assembled dual-Franka robot cache is not present.")
    names = ("left_inner_finger_pad", "left_right_inner_finger_pad")
    meshes = twist_geometry.load_robot_link_meshes(path, names)
    for name in names:
        assert len(meshes[name]) > 0
        for mesh in meshes[name]:
            assert mesh.owner_path == name
            assert mesh.approximation in {"urdfMeshInput", "urdfBoxInput"}
            assert np.isfinite(mesh.vertices).all()
            assert not mesh.vertices.flags.writeable
            assert mesh.faces.shape[1] == 3


@pytest.mark.parametrize(
    "task,asset,setting,scale",
    [
        ("task1130", "knob_panel_001", 90, 2.0 / 3.0),
        ("task1139", "hot_plate_001", 1, 0.470352381468),
        ("task1149", "dimmer_switch_001", 2, 12.5 / 7.2),
    ],
)
def test_cached_knob_assets_have_scale_consistent_grip_and_collision_frames(
    task, asset, setting, scale
):
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "gym_project/task100/new2"
        / task
        / "scene_export/articulated_assets"
        / asset
        / (asset + ".usdc")
    )
    if not path.is_file():
        pytest.skip("Optional external knob asset cache is not present.")
    route = discover_twist(
        {"uid": asset, "fpath": str(path), "fix_base": True, "body_scale": [scale] * 3},
        setting,
    )
    geometry = twist_geometry.load_twist_geometry(path, route.binding)
    unit_geometry = twist_geometry.load_twist_geometry(
        path, replace(route.binding, scale=1.0)
    )
    np.testing.assert_allclose(
        geometry.grip.vertices, unit_geometry.grip.vertices * scale, atol=1e-10
    )
    assert geometry.grip.owner_path == geometry.target_body_path
    assert geometry.grip.path in {mesh.path for mesh in geometry.target_collisions}
    assert len(geometry.parent_collisions) > 0
    depth = np.ptp(geometry.grip.vertices @ np.asarray(route.binding.axis))
    assert depth == pytest.approx(route.binding.grip_depth, abs=1e-9)
    for actual, unit in zip(
        geometry.parent_collisions, unit_geometry.parent_collisions
    ):
        assert actual.path == unit.path
        np.testing.assert_allclose(actual.vertices, unit.vertices * scale, atol=1e-10)


@pytest.fixture
def candidate_score_case(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import twist_runtime

    root_pose = torch.eye(4).unsqueeze(0)
    root_pose[0, :3, :3] = torch.tensor(_ROTATE_Z_90, dtype=torch.float32)
    root_pose[0, :3, 3] = torch.tensor([1.0, 2.0, 0.7])
    names = ("left_inner_finger_pad", "left_right_inner_finger_pad")
    fk = torch.eye(4).expand(1, 2, 4, 4).clone()
    fk[0, 0, 0, 3] = -0.02
    fk[0, 1, 0, 3] = 0.02

    def reject_render(*args, **kwargs):
        pytest.fail("E8 candidate score must not request native render vertices.")

    robot = SimpleNamespace(
        link_names=names,
        joint_names=("joint",),
        compute_fk=lambda **kwargs: fk,
        get_local_pose=lambda **kwargs: root_pose,
        get_link_vert_face=reject_render,
    )
    monkeypatch.setattr(twist_runtime, "resolve_pose_goal", lambda *a, **kw: root_pose)
    monkeypatch.setattr(
        twist_runtime, "resolve_pose_target", lambda *a, **kw: root_pose
    )
    target_to_parent = torch.eye(4)
    target_to_parent[0, 3] = 0.02
    geometry = {
        "gen_sim_twist_arm": "left",
        "gen_sim_twist_grip_points": [[-0.02, 0, 0], [0.02, 0, 0]],
        "gen_sim_twist_parent_collision_points": [[0, 0, 0]],
        "gen_sim_twist_pad_collision_points": {name: [[0, 0, 0]] for name in names},
        "gen_sim_twist_target_to_parent": target_to_parent.tolist(),
        "gen_sim_twist_geometry_contract": "scaled_owning_link_collision_inputs/v1",
        # Legacy render fields deliberately disagree with the collider inputs.
        "gen_sim_twist_target_points": [[99, 99, 99]],
        "gen_sim_twist_parent_points": [[99, 99, 99]],
    }
    request = SimpleNamespace(
        goal=SimpleNamespace(
            semantics=SimpleNamespace(geometry=geometry), target_pose=None
        )
    )
    plan = SimpleNamespace(
        segments=[SimpleNamespace(name="close", stop=1)],
        joint_trajectory=SimpleNamespace(positions=torch.zeros(1, 1, 1)),
    )
    return SimpleNamespace(
        runtime=twist_runtime,
        owner=SimpleNamespace(robot=robot),
        request=request,
        plan=plan,
        context=SimpleNamespace(robot=SimpleNamespace(qpos=torch.zeros(1, 1))),
        geometry=geometry,
        pad_names=names,
    )


def _candidate_score(case):
    return case.runtime.GenSimTwist._candidate_geometry_score(
        case.owner, case.plan, case.request, case.context
    )


def test_candidate_score_uses_source_grip_and_distinct_parent_frame_not_render(
    candidate_score_case,
):
    # One pad coincides with the transformed parent collider; neither is far from grip.
    assert _candidate_score(candidate_score_case) == pytest.approx(
        (1.0, 0.5, 0.0, 0.0), abs=1e-6
    )


def test_candidate_score_uses_pad_collision_points_in_native_fk_frame(
    candidate_score_case,
):
    name = candidate_score_case.pad_names[0]
    candidate_score_case.geometry["gen_sim_twist_pad_collision_points"][name] = [
        [0, 0, 0.01]
    ]
    score = _candidate_score(candidate_score_case)
    assert score[:3] == pytest.approx((1.0, 0.5, 0.01), abs=1e-6)


@pytest.mark.parametrize(
    "key",
    [
        "gen_sim_twist_grip_points",
        "gen_sim_twist_parent_collision_points",
        "gen_sim_twist_pad_collision_points",
        "gen_sim_twist_target_to_parent",
    ],
)
def test_candidate_score_missing_collision_fields_cannot_use_legacy_render_fields(
    candidate_score_case, key
):
    candidate_score_case.geometry.pop(key)
    with pytest.raises(ValueError, match="source-bound collision geometry"):
        _candidate_score(candidate_score_case)


def test_candidate_score_requires_collision_points_for_each_selected_pad(
    candidate_score_case,
):
    candidate_score_case.geometry["gen_sim_twist_pad_collision_points"].pop(
        candidate_score_case.pad_names[0]
    )
    with pytest.raises(ValueError, match="pad collision geometry is missing"):
        _candidate_score(candidate_score_case)


def _pose(translation, rotation=None):
    result = torch.eye(4).unsqueeze(0)
    result[0, :3, 3] = torch.tensor(translation, dtype=torch.float32)
    if rotation is not None:
        result[0, :3, :3] = torch.tensor(rotation, dtype=torch.float32)
    return result


@pytest.fixture
def park_clearance_case():
    from unittest.mock import Mock

    from embodichain.gen_sim.task_engine._task_program import twist_runtime

    # Source local points are already scaled; native poses must not scale them again.
    finger_points = [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]
    pad_poses = {"pad_a": _pose([0, 0, 0]), "pad_b": _pose([-0.1, 0, 0])}
    collider_points = np.asarray([[0, 0, 0], [0.02, 0.04, 0.06]]) * 2.0
    meshes = tuple(
        SimpleNamespace(owner_path="/Control/" + name, vertices=collider_points.copy())
        for name in ("base", "knob", "other")
    )
    owner_poses = {
        "base": _pose([1.0, 0, 0]),
        "knob": _pose([0.8, 0, 0]),
        "other": _pose([0.2, 0, 0], _ROTATE_Z_90),
    }
    art = SimpleNamespace(
        link_names=("base", "knob", "other"),
        get_link_pose=Mock(side_effect=lambda name, **kwargs: owner_poses[name]),
        get_link_vert_face=Mock(side_effect=AssertionError("No render mesh fallback.")),
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(body_scale=[10, 10, 10]),
        get_link_pose=Mock(side_effect=lambda name, **kwargs: pad_poses[name]),
        get_link_vert_face=Mock(side_effect=AssertionError("No render mesh fallback.")),
    )
    sensor = SimpleNamespace(
        finger_names=("pad_a", "pad_b"),
        finger_collision_points={name: finger_points for name in pad_poses},
        geometry=SimpleNamespace(articulation_collisions=meshes, scale=2.0),
        art=art,
    )
    return SimpleNamespace(
        runtime=twist_runtime,
        owner=SimpleNamespace(sensor=sensor, robot=robot),
        art=art,
        robot=robot,
        meshes=meshes,
        owner_poses=owner_poses,
        pad_poses=pad_poses,
    )


def _park_clearance(case):
    return case.runtime.TwistAcceptancePort._clearance(case.owner)


def test_park_clearance_includes_nonselected_owner_and_applies_rigid_pose_once(
    park_clearance_case,
):
    assert _park_clearance(park_clearance_case) == pytest.approx(0.11, abs=1e-7)
    assert [
        call.args[0] for call in park_clearance_case.art.get_link_pose.call_args_list
    ] == [
        "base",
        "knob",
        "other",
    ]
    park_clearance_case.art.get_link_vert_face.assert_not_called()
    park_clearance_case.robot.get_link_vert_face.assert_not_called()


def test_park_clearance_rejects_missing_native_owner_instead_of_selected_link_fallback(
    park_clearance_case,
):
    park_clearance_case.art.link_names = ("base", "knob")
    with pytest.raises(ValueError, match="ownership differs"):
        _park_clearance(park_clearance_case)


@pytest.mark.parametrize(
    "kind",
    ["collider_vertices", "native_owner_pose", "native_pad_pose", "pad_vertices"],
)
def test_park_clearance_rejects_nonfinite_geometry_and_native_poses(
    park_clearance_case, kind
):
    if kind == "collider_vertices":
        park_clearance_case.meshes[-1].vertices[0, 0] = np.nan
    elif kind == "native_owner_pose":
        park_clearance_case.owner_poses["other"][0, 0, 3] = torch.nan
    elif kind == "native_pad_pose":
        park_clearance_case.pad_poses["pad_a"][0, 0, 3] = torch.nan
    else:
        park_clearance_case.owner.sensor.finger_collision_points["pad_a"] = [
            [np.nan, 0, 0]
        ]
    with pytest.raises(ValueError, match="finite"):
        _park_clearance(park_clearance_case)


def test_park_clearance_rejects_empty_articulation_collision_inventory(
    park_clearance_case,
):
    park_clearance_case.owner.sensor.geometry.articulation_collisions = ()
    with pytest.raises(ValueError, match="complete finger and articulation"):
        _park_clearance(park_clearance_case)


def test_sensor_geometry_evidence_declares_frame_scale_and_uncertified_cooking(
    geometry_source,
):
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
        TwistContactSensor,
    )

    geometry = twist_geometry.load_twist_geometry(
        geometry_source.path, replace(geometry_source.binding, scale=2.0)
    )
    pad = replace(geometry.grip, path="pad/mesh", owner_path="pad")
    sensor = SimpleNamespace(geometry=geometry, finger_collisions={"pad": (pad,)})
    evidence = TwistContactSensor.geometry_evidence(sensor)
    assert evidence["schema"] == "scaled_owning_link_collision_inputs/v1"
    assert evidence["frame"] == "scaled owning rigid-link local"
    assert evidence["units"] == "m"
    assert evidence["asset_scale_applied"] == 2.0
    assert evidence["native_pose_scaled_again"] is False
    assert evidence["source_sha256"] == geometry_source.binding.source_sha256
    assert evidence["backend_cooked_shapes_verified"] is False
    assert evidence["continuous_collision_certificate"] is False
    assert len(evidence["articulation_collisions"]) == len(
        geometry.articulation_collisions
    )
    assert evidence["fingers"]["pad"][0]["owner"] == "pad"
    np.testing.assert_allclose(
        evidence["grip"]["bounds"],
        [geometry.grip.vertices.min(0), geometry.grip.vertices.max(0)],
    )
