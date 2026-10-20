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

from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.task_engine._task_program import twist_support
from embodichain.gen_sim.task_engine.scene.articulation_geometry import (
    ArticulationGeometry,
)


def test_sweep_finds_interior_minimum_without_angle_sampling():
    # Rotating +X around Y attains Z=-1 inside the interval at pi/2.
    result = twist_support.revolute_swept_min_z(
        np.array([[1.0, 0.0, 0.0]]),
        np.eye(3),
        np.zeros(3),
        np.array([0.0, 1.0, 0.0]),
        (0.0, math.pi),
        1.0,
    )
    assert result == pytest.approx(-1.0)


@pytest.mark.parametrize("limits", [(0.0, math.pi / 6), (0.0, 0.0)])
def test_sweep_respects_limited_joint_interval(limits):
    result = twist_support.revolute_swept_min_z(
        np.array([[1.0, 0.0, 0.0]]),
        np.eye(3),
        np.zeros(3),
        np.array([0.0, 1.0, 0.0]),
        limits,
        1.0,
    )
    assert result == pytest.approx(-math.sin(limits[1]))


def test_sweep_includes_tilt_scale_and_nonzero_pivot():
    pivot = np.array([0.0, 0.0, 0.25])
    rotation = Rotation.from_euler("Y", 30, degrees=True).as_matrix()
    result = twist_support.revolute_swept_min_z(
        np.array([[1.0, 0.0, 0.25]]),
        rotation,
        pivot,
        np.array([0.0, 1.0, 0.0]),
        (0.0, math.pi),
        2.0,
    )
    assert result == pytest.approx(2 * (0.25 * math.cos(math.pi / 6) - 1))


def test_sweep_handles_many_turns_and_axis_parallel_vertex():
    result = twist_support.revolute_swept_min_z(
        np.array([[0.0, 0.0, -2.0], [1.0, 0.0, 0.0]]),
        np.eye(3),
        np.zeros(3),
        np.array([0.0, 0.0, 2.0]),
        (-20 * math.pi, 20 * math.pi),
        0.5,
    )
    assert result == pytest.approx(-1.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), True])
def test_sweep_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="positive uniform scale"):
        twist_support.revolute_swept_min_z(
            np.array([[0.0, 0.0, 0.0]]),
            np.eye(3),
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            (0.0, 1.0),
            scale,
        )


@pytest.mark.parametrize(
    "points,axis,limits",
    [
        (np.empty((0, 3)), [0, 0, 1], (0, 1)),
        (np.zeros((1, 3)), [0, 0, 0], (0, 1)),
        (np.zeros((1, 3)), [0, 0, 1], (1, 0)),
        (np.zeros((1, 3)), [0, 0, 1], (0, float("inf"))),
    ],
)
def test_sweep_rejects_unmeasured_geometry_or_invalid_joint(points, axis, limits):
    with pytest.raises(ValueError):
        twist_support.revolute_swept_min_z(
            points, np.eye(3), np.zeros(3), np.array(axis), limits, 1.0
        )


@pytest.fixture
def measured_control(monkeypatch):
    points = np.array([[0.0, 0.0, -0.016], [0.0, 0.0, 0.006]])
    geometry = ArticulationGeometry(
        "base", points, {"base": np.eye(4), "knob": np.eye(4)}, {"knob": points}
    )
    monkeypatch.setattr(twist_support, "read_articulation_geometry", lambda _: geometry)
    monkeypatch.setattr(
        twist_support,
        "_moving_collision_envelopes",
        lambda *_: [
            {"collision_index": 0, "contact_offset": 0.006, "rest_offset": 0.001}
        ],
    )
    config = {
        "uid": "control",
        "fpath": "control.usda",
        "root_props": {"fixed_base": True},
        "init_pos": [0.2, -0.1, 0.709],
        "body_scale": [0.5] * 3,
    }
    binding = SimpleNamespace(
        object_id="control",
        joint="knob_axis",
        link="knob",
        origin=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, -1.0),
        axis_sign=-1.0,
        limits=(-math.pi / 2, 0.0),
        scale=0.5,
        source_sha256="fixture",
    )
    return config, binding, geometry


def test_adjust_support_changes_only_root_z_and_records_inputs(measured_control):
    config, binding, _ = measured_control
    original = deepcopy(config)
    adjusted, audit = twist_support.adjust_twist_support(
        config, binding, {"uid": "table"}, table_top_z=0.7
    )
    assert config == original
    assert adjusted["init_pos"][:2] == original["init_pos"][:2]
    assert adjusted["body_scale"] == original["body_scale"]
    assert audit["translation_z"] == pytest.approx(0.0111)
    assert audit["support_gap_after"] == pytest.approx(0.0121)
    assert np.asarray(adjusted["init_local_pose"])[2, 3] == adjusted["init_pos"][2]
    assert audit["joint_axis_root"] == [0.0, 0.0, 1.0]
    assert audit["geometry_frame"] == "unscaled_native_root_link"
    assert audit["source_edited"] is False


def test_adjust_support_preserves_sufficient_existing_gap(measured_control):
    config, binding, _ = measured_control
    config["init_pos"][2] += 0.1
    adjusted, audit = twist_support.adjust_twist_support(
        config, binding, {"uid": "table"}, table_top_z=0.7
    )
    assert audit["translation_z"] == 0
    assert adjusted["init_pos"] == config["init_pos"]


@pytest.mark.parametrize("millimeters", [1, 2, 3, 4])
def test_explicit_total_lift_records_actual_gap_without_no_contact_claim(
    measured_control, millimeters
):
    config, binding, _ = measured_control
    adjusted, audit = twist_support.adjust_twist_support(
        config,
        binding,
        {"uid": "table"},
        table_top_z=0.7,
        requested_total_lift_m=millimeters / 1000,
    )
    assert adjusted["init_pos"][2] == pytest.approx(0.709 + millimeters / 1000)
    assert audit["translation_z"] == pytest.approx(millimeters / 1000)
    assert audit["conservative_translation_z"] == pytest.approx(0.0111)
    assert audit["support_gap_after"] == pytest.approx(0.001 + millimeters / 1000)
    assert audit["geometric_penetration_free"] is True
    assert audit["no_contact_envelope_satisfied"] is False
    assert (
        audit["qualification"]
        == "runtime_initial_stability_and_terminal_checks_required"
    )


def test_explicit_total_lift_cannot_hide_swept_penetration(measured_control):
    config, binding, _ = measured_control
    config["init_pos"][2] -= 0.003
    with pytest.raises(
        twist_support.TwistSupportPenetrationError, match="below the table"
    ):
        twist_support.adjust_twist_support(
            config,
            binding,
            {"uid": "table"},
            table_top_z=0.7,
            requested_total_lift_m=0.001,
        )
    _, audit = twist_support.adjust_twist_support(
        config,
        binding,
        {"uid": "table"},
        table_top_z=0.7,
        requested_total_lift_m=0.003,
    )
    assert audit["geometric_penetration_free"] is True


@pytest.mark.parametrize("lift", [True, 0, 0.0005, 0.0025, 0.005, float("nan")])
def test_explicit_total_lift_is_limited_to_approved_candidates(measured_control, lift):
    config, binding, _ = measured_control
    with pytest.raises(ValueError, match="1, 2, 3 or 4 mm"):
        twist_support.adjust_twist_support(
            config,
            binding,
            {"uid": "table"},
            table_top_z=0.7,
            requested_total_lift_m=lift,
        )


def test_adjust_support_respects_table_configured_envelope(measured_control):
    config, binding, _ = measured_control
    table = {"uid": "table", "attrs": {"collision_props": {"contact_offset": 0.002}}}
    _, audit = twist_support.adjust_twist_support(
        config, binding, table, table_top_z=0.7
    )
    assert audit["table_contact_offset"] == pytest.approx(0.002)
    assert audit["translation_z"] == pytest.approx(0.0071)


def test_adjust_support_rejects_missing_moving_geometry(measured_control):
    config, binding, geometry = measured_control
    geometry.link_vertices.clear()
    with pytest.raises(ValueError, match="no measured moving collision"):
        twist_support.adjust_twist_support(config, binding, {}, table_top_z=0.7)


def test_adjust_support_rejects_nonfixed_base(measured_control):
    config, binding, _ = measured_control
    config["root_props"]["fixed_base"] = False
    with pytest.raises(ValueError, match="fixed base"):
        twist_support.adjust_twist_support(config, binding, {}, table_top_z=0.7)


def test_adjust_support_converts_scaled_child_binding_into_root_frame(measured_control):
    config, binding, geometry = measured_control
    link_pose = np.eye(4)
    link_pose[:3, :3] = Rotation.from_euler("X", 90, degrees=True).as_matrix()
    link_pose[:3, 3] = [0.2, 0.0, 0.4]
    geometry.link_poses["knob"] = link_pose
    binding.scale = 0.25
    binding.origin = (0.0, 0.0, 0.025)
    pivot_root = np.array([0.2, -0.1, 0.4])
    geometry.link_vertices["knob"] = np.array([pivot_root + [1.0, 0.0, 0.0]])
    _, audit = twist_support.adjust_twist_support(config, binding, {}, table_top_z=0.7)
    assert audit["pivot_root"] == pytest.approx(pivot_root.tolist())
    assert audit["joint_axis_root"] == pytest.approx([0.0, -1.0, 0.0])
    # Negative rotation around -Y reaches root-local pivot_z - radius.
    assert audit["moving_swept_min_z_before"] == pytest.approx(
        config["init_pos"][2] + 0.5 * (pivot_root[2] - 1.0)
    )


def test_adjust_support_uses_explicit_intrinsic_pose_and_preserves_orientation(
    measured_control,
):
    config, binding, geometry = measured_control
    rotation = Rotation.from_euler("XYZ", [20, 15, 40], degrees=True).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = config["init_pos"]
    config["init_local_pose"] = pose.tolist()
    geometry.link_vertices["knob"] = np.array([[0.1, 0.0, 0.0]])
    adjusted, audit = twist_support.adjust_twist_support(
        config, binding, {}, table_top_z=0.7
    )
    expected = (
        twist_support.revolute_swept_min_z(
            geometry.link_vertices["knob"],
            rotation,
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            binding.limits,
            0.5,
        )
        + pose[2, 3]
    )
    assert audit["moving_swept_min_z_before"] == pytest.approx(expected)
    assert np.asarray(adjusted["init_local_pose"])[:3, :3] == pytest.approx(rotation)


@pytest.fixture
def source_control(tmp_path):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "control.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Control")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for name in ("base", "knob"):
        body = UsdGeom.Xform.Define(stage, "/Control/" + name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.1)
        mesh = UsdGeom.Mesh.Define(stage, "/Control/" + name + "/collision")
        mesh.CreatePointsAttr(
            [(-0.01, -0.01, 0), (0.01, -0.01, 0), (0, 0.01, 0), (0, 0, 0.02)]
        )
        mesh.CreateFaceVertexCountsAttr([3] * 4)
        mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
            "convexHull"
        )
        mesh.GetPrim().CreateAttribute(
            "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
        ).Set(0.004)
        mesh.GetPrim().CreateAttribute(
            "physxCollision:restOffset", Sdf.ValueTypeNames.Float
        ).Set(0.0005)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Control/axis")
    joint.CreateBody0Rel().SetTargets(["/Control/base"])
    joint.CreateBody1Rel().SetTargets(["/Control/knob"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-90)
    joint.CreateUpperLimitAttr(0)
    stage.GetRootLayer().Save()
    return {"uid": "control", "fpath": str(path), "root_props": {"fixed_base": True}}


def test_collision_envelopes_preserve_source_authored_values(source_control):
    source_control["asset_physics_mode"] = "preserve"
    result = twist_support._moving_collision_envelopes(source_control, "knob")
    assert result[0]["contact_offset"] == pytest.approx(0.004)
    assert result[0]["rest_offset"] == pytest.approx(0.0005)


def test_collision_envelopes_apply_global_and_exact_link_overlays(source_control):
    source_control["asset_physics_mode"] = "overlay"
    source_control["attrs"] = {"collision_props": {"contact_offset": 0.008}}
    source_control["link_attrs"] = {
        "selected": {
            "link_names_expr": ["knob"],
            "attrs": {"collision_props": {"contact_offset": 0.003}},
        }
    }
    result = twist_support._moving_collision_envelopes(source_control, "knob")
    assert result[0]["contact_offset"] == pytest.approx(0.003)
    assert result[0]["rest_offset"] == pytest.approx(0.0)


def test_collision_envelopes_normalize_prepared_scene_metadata(source_control):
    source_control.pop("root_props")
    source_control["fix_base"] = True
    source_control["runtime_uid"] = "control"
    source_control["metadata"] = {"source": "prepared_scene"}
    source_control["asset_physics_mode"] = "overlay"
    source_control["attrs"] = {"collision_props": {"contact_offset": 0.008}}
    result = twist_support._moving_collision_envelopes(source_control, "knob")
    assert result[0]["contact_offset"] == pytest.approx(0.008)


def test_collision_mesh_can_own_its_rigid_body(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "mesh_bodies.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Control").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    points = [(-0.01, -0.01, 0), (0.01, -0.01, 0), (0, 0.01, 0), (0, 0, 0.02)]
    for name in ("base", "knob"):
        mesh = UsdGeom.Mesh.Define(stage, "/Control/" + name)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3] * 4)
        mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
        UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Control/axis")
    joint.CreateBody0Rel().SetTargets(["/Control/base"])
    joint.CreateBody1Rel().SetTargets(["/Control/knob"])
    stage.GetRootLayer().Save()

    geometry = twist_support.read_articulation_geometry(path)

    assert geometry.root_link == "base"
    assert set(geometry.link_vertices) == {"base", "knob"}
    assert geometry.link_vertices["base"] == pytest.approx(np.asarray(points))
    assert geometry.link_vertices["knob"] == pytest.approx(np.asarray(points))
