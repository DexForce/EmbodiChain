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

"""Provider-free tabletop release geometry and restricted-target regressions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Point, box
import trimesh

from embodichain.gen_sim.task_engine import task_program_bundle as bundle


def _mesh_object(
    tmp_path: Path,
    uid: str,
    extents: list[float],
    position: list[float],
    *,
    local_center: list[float] | None = None,
) -> dict:
    mesh = trimesh.creation.box(extents=extents)
    if local_center is not None:
        mesh.vertices += np.asarray(local_center)
    mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, 1, -1]
    path = tmp_path / f"{uid}.glb"
    mesh.export(path)
    return {
        "runtime_uid": uid,
        "uid": uid,
        "role": "rigid_object",
        "shape": {"shape_type": "Mesh", "fpath": str(path)},
        "init_pos": position,
        "init_rot": [0.0, 0.0, 0.0],
        "body_scale": [1.0, 1.0, 1.0],
    }


def _scene_objects(tmp_path: Path) -> dict[str, dict]:
    return {
        "table": _mesh_object(tmp_path, "table", [1.0, 1.0, 0.05], [0, 0, 0.725]),
        "payload": _mesh_object(tmp_path, "payload", [0.06, 0.06, 0.06], [0, 0, 1.0]),
    }


def _displacement(objects: dict[str, dict], **kwargs) -> list[float]:
    return bundle._relative_world_displacement(
        "on",
        object_id="payload",
        reference_id="table",
        scene_objects=objects,
        axis_align_objects=set(),
        table_top_z=0.75,
        **kwargs,
    )


def test_table_landing_relocates_occupied_source_projection(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    objects["cabinet"] = _mesh_object(
        tmp_path, "cabinet", [0.24, 0.24, 0.4], [0, 0, 0.95]
    )
    original = deepcopy(objects)

    displacement = _displacement(objects)
    landing = np.asarray(displacement) + np.asarray(objects["table"]["init_pos"])
    footprint = MultiPoint(bundle._mesh_vertices(objects["payload"])[:, :2]).convex_hull
    footprint = translate(footprint, *landing[:2])

    assert not footprint.intersects(box(-0.14, -0.14, 0.14, 0.14))
    assert box(-0.5, -0.5, 0.5, 0.5).covers(footprint)
    assert landing[2] == pytest.approx(0.79)
    assert objects == original


@pytest.mark.parametrize("xy", [[-0.18, -0.28], [0.21, 0.29], [0.46, 0.0]])
def test_valid_original_landing_is_preserved_exactly(
    tmp_path: Path, xy: list[float]
) -> None:
    objects = _scene_objects(tmp_path)
    objects["payload"]["init_pos"][:2] = xy
    objects["obstacle"] = _mesh_object(
        tmp_path, "obstacle", [0.1, 0.1, 0.1], [0, 0, 0.8]
    )
    assert _displacement(objects)[:2] == xy


def test_held_subject_is_not_its_own_obstacle(tmp_path: Path) -> None:
    assert _displacement(_scene_objects(tmp_path))[:2] == [0.0, 0.0]


def test_full_table_has_no_permissive_landing_fallback(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    objects["cabinet"] = _mesh_object(
        tmp_path, "cabinet", [1.0, 1.0, 0.4], [0, 0, 0.95]
    )
    with pytest.raises(ValueError, match="remaining.*landing|landing.*space"):
        _displacement(objects)


def test_unmeasured_obstacle_does_not_silently_admit_a_target(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    objects["unknown"] = {"runtime_uid": "unknown", "init_pos": [0, 0, 0.8]}
    with pytest.raises(ValueError, match="geometry"):
        _displacement(objects)


def test_unmeasured_table_does_not_use_guessed_bounds(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    objects["table"].pop("shape")
    with pytest.raises(ValueError, match="geometry"):
        _displacement(objects)


def test_table_landing_keeps_holes_in_the_measured_support(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    table = trimesh.creation.annulus(r_min=0.08, r_max=0.45, height=0.05)
    table.vertices = np.asarray(table.vertices)[:, [0, 2, 1]] * [1, 1, -1]
    table.export(objects["table"]["shape"]["fpath"])
    with pytest.raises(ValueError, match="support"):
        _displacement(objects)


def test_small_mesh_top_variation_keeps_the_authored_release_height(
    tmp_path: Path,
) -> None:
    objects = _scene_objects(tmp_path)
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.05])
    corner = trimesh.creation.box(extents=[0.005, 0.005, 0.0005])
    corner.apply_translation([0.49, 0.49, 0.02525])
    mesh = trimesh.util.concatenate([mesh, corner])
    mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, 1, -1]
    mesh.export(objects["table"]["shape"]["fpath"])
    top = float(bundle._table_world_vertices(objects["table"])[:, 2].max())
    result = bundle._relative_world_displacement(
        "on",
        object_id="payload",
        reference_id="table",
        scene_objects=objects,
        axis_align_objects=set(),
        table_top_z=top,
    )
    assert result[:2] == [0.0, 0.0]
    assert result[2] + 0.725 == pytest.approx(top + 0.04)


def test_unsupported_original_projection_is_not_silently_relocated(
    tmp_path: Path,
) -> None:
    objects = _scene_objects(tmp_path)
    objects["payload"]["init_pos"][0] = 0.49
    with pytest.raises(ValueError, match="support|footprint"):
        _displacement(objects)


def test_obstacle_uses_scaled_off_origin_authoritative_pose(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    objects["obstacle"] = _mesh_object(
        tmp_path,
        "obstacle",
        [0.2, 0.1, 0.4],
        [0, 0, 0.95],
        local_center=[0.2, 0, 0],
    )
    objects["obstacle"]["body_scale"] = [1.5, 1.5, 1]
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [0, -0.3, 0.95]
    objects["obstacle"]["init_local_pose"] = pose.tolist()
    objects["obstacle"]["init_pos"] = [0, -0.3, 0.95]

    landing = np.asarray(_displacement(objects))[:2]
    assert not box(-0.075 - 0.05, -0.15 - 0.05, 0.075 + 0.05, 0.15 + 0.05).contains(
        Point(landing)
    )


def test_named_region_is_not_relocated_to_another_support(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    objects["blue_region"] = _mesh_object(
        tmp_path, "blue_region", [0.1, 0.1, 0.01], [0.2, 0.3, 0.755]
    )
    objects["unknown"] = {"runtime_uid": "unknown"}
    result = bundle._relative_world_displacement(
        "on",
        object_id="payload",
        reference_id="blue_region",
        scene_objects=objects,
        axis_align_objects=set(),
        table_top_z=0.75,
    )
    assert result[:2] == [0.0, 0.0]


def _relative_node(task_type: str = "E1") -> dict:
    return {
        "id": "place",
        "task_instance_id": "placement",
        "task_type": task_type,
        "call": {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {"object": "payload", "reference": "table", "relation": "on"},
        },
    }


def test_e2_upright_keeps_its_explicit_in_place_target(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    graph = {
        "nodes": [_relative_node("E2")],
        "targets": {
            "placement_upright_target": {
                "kind": "pose",
                "values": [
                    {"position": [0.12, 0.23, 0.82], "quaternion_xyzw": [0, 0, 0, 1]}
                ],
            }
        },
    }
    objects["unknown"] = {"runtime_uid": "unknown"}
    scene = SimpleNamespace(planner_objects=tuple(objects.values()), table_top_z=0.75)
    route = bundle._relative_place_route_payloads(graph, scene)[0]
    assert route["world_displacement"] == pytest.approx([0.12, 0.23, 0.095])


def test_explicit_original_pose_is_not_replaced_by_table_landing(
    tmp_path: Path,
) -> None:
    objects = _scene_objects(tmp_path)
    node = _relative_node("E3")
    node["call"] = {
        "kind": "place",
        "object": "payload",
        "target": "original_pose",
    }
    graph = {"nodes": [node], "targets": {"original_pose": {"values": []}}}
    scene = SimpleNamespace(planner_objects=tuple(objects.values()), table_top_z=0.75)
    assert bundle._relative_place_route_payloads(graph, scene) == []


def test_above_table_retains_its_previous_target_policy(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    objects["unknown"] = {"runtime_uid": "unknown"}
    result = bundle._relative_world_displacement(
        "above",
        object_id="payload",
        reference_id="table",
        scene_objects=objects,
        axis_align_objects=set(),
        table_top_z=0.75,
    )
    assert result[:2] == [0.0, 0.0]


def _articulation_obstacle(
    tmp_path: Path, *, joint_kind: str = "prismatic", handle: bool = True
) -> dict:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "cabinet.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/cabinet").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for name, extents, position in (
        ("base", [0.2, 0.2, 0.3], [0, 0, 0.15]),
        ("drawer", [0.16, 0.16, 0.1], [0, 0, 0.2]),
    ):
        body = UsdGeom.Xform.Define(stage, f"/cabinet/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mesh = trimesh.creation.box(extents=extents)
        mesh_name = "handle" if handle else "surface"
        usd_mesh = UsdGeom.Mesh.Define(stage, f"/cabinet/{name}/{mesh_name}")
        usd_mesh.CreatePointsAttr(mesh.vertices.tolist())
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
        UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim()).CreateCollisionEnabledAttr(
            True
        )
    joint_class = (
        UsdPhysics.PrismaticJoint
        if joint_kind == "prismatic"
        else UsdPhysics.RevoluteJoint
    )
    joint = joint_class.Define(stage, "/cabinet/slide")
    joint.CreateBody0Rel().SetTargets(["/cabinet/base"])
    joint.CreateBody1Rel().SetTargets(["/cabinet/drawer"])
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.05))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLowerLimitAttr(-0.2)
    joint.CreateUpperLimitAttr(0.0)
    stage.GetRootLayer().Save()
    return {
        "uid": "cabinet",
        "runtime_uid": "cabinet",
        "role": "articulation",
        "fpath": str(path),
        "fix_base": True,
        "init_pos": [0.2, 0, 0.75],
        "init_rot": [0, 0, 0],
        "body_scale": [1, 1, 1],
    }


def test_table_landing_excludes_prismatic_full_travel_envelope(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    objects["cabinet"] = _articulation_obstacle(tmp_path)
    landing = np.asarray(_displacement(objects))[:2]
    # The authored drawer is beside this point, but can slide across it.
    assert not box(-0.15, -0.15, 0.55, 0.15).contains(Point(landing))


def test_unqualified_moving_articulation_is_not_assumed_static(
    tmp_path: Path,
) -> None:
    objects = _scene_objects(tmp_path)
    objects["cabinet"] = _articulation_obstacle(tmp_path, joint_kind="revolute")
    with pytest.raises(ValueError, match="[Aa]rticulation|prismatic|handle|joint"):
        _displacement(objects)


def test_handleless_obstacle_is_measured_without_inventing_e6_grasp(
    tmp_path: Path,
) -> None:
    objects = _scene_objects(tmp_path)
    objects["cabinet"] = _articulation_obstacle(tmp_path, handle=False)
    landing = np.asarray(_displacement(objects))[:2]
    assert not box(-0.15, -0.15, 0.55, 0.15).contains(Point(landing))
    with pytest.raises(ValueError, match="handle"):
        bundle.discover_prismatic_parts(objects["cabinet"])


def test_floating_articulation_obstacle_has_no_static_fallback(tmp_path: Path) -> None:
    objects = _scene_objects(tmp_path)
    objects["cabinet"] = _articulation_obstacle(tmp_path)
    objects["cabinet"]["fix_base"] = False
    with pytest.raises(ValueError, match="fixed"):
        _displacement(objects)


def test_slider_travel_follows_scaled_authoritative_world_axes(tmp_path: Path) -> None:
    source = _articulation_obstacle(tmp_path)
    source["body_scale"] = [2.0, 2.0, 2.0]
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = source["init_pos"]
    source["init_local_pose"] = pose.tolist()
    bounds = bundle._table_obstacle_bounds(source)
    assert bounds[:, 0] == pytest.approx([0.0, 0.4], abs=1e-7)
    assert bounds[:, 1] == pytest.approx([-0.6, 0.6], abs=1e-7)


def test_missing_slider_limits_are_not_guessed(tmp_path: Path) -> None:
    from pxr import Usd, UsdPhysics

    objects = _scene_objects(tmp_path)
    objects["cabinet"] = _articulation_obstacle(tmp_path)
    stage = Usd.Stage.Open(objects["cabinet"]["fpath"])
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/cabinet/slide"))
    joint.GetLowerLimitAttr().Clear()
    joint.GetUpperLimitAttr().Clear()
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="limits"):
        _displacement(objects)


def _region_graph_scene(tmp_path: Path):
    objects = _scene_objects(tmp_path)
    node = _relative_node()
    node["call"]["resources"] = {"primary": "left"}
    graph = {
        "nodes": [node],
        "targets": {},
        "task_groups": [{"id": "placement", "task_type": "E1", "node_ids": ["place"]}],
    }
    scene = SimpleNamespace(
        planner_objects=tuple(objects.values()),
        rigid_objects=(objects["payload"],),
        articulations=(),
        table_top_z=0.75,
    )
    embodiment = {
        "skill_profile": {
            "resources": [
                {
                    "resource_id": "left",
                    "endpoints": [
                        {"endpoint_id": "motion", "control_part": "left_arm"}
                    ],
                }
            ]
        }
    }
    return graph, scene, embodiment


def test_bare_table_on_emits_region_not_proposal_point_acceptance(tmp_path: Path):
    graph, scene, embodiment = _region_graph_scene(tmp_path)
    presets = bundle._task_stability_payload(graph, scene, embodiment)["presets"]
    cfg = presets["gen_sim.place.stable"]
    assert cfg["kind"] == "supported_region"
    assert cfg["reference"] == "table"
    assert "displacement" not in cfg and "target_position" not in cfg
    item = bundle._program_payload(graph, "region", scene, stability_presets=presets)[
        "program"
    ]["items"][0]
    assert not item.get("validators")
    assert any(p.get("preset") == "gen_sim.place.stable" for p in item["post"])


@pytest.mark.parametrize(
    "presets",
    [
        None,
        set(),
        {"gen_sim.place.stable"},
        {
            "gen_sim.place.stable": {
                "kind": "supported_placement",
                "entity": "payload",
                "reference": "table",
            }
        },
    ],
)
def test_missing_actual_region_preset_keeps_strict_point_validator(
    tmp_path: Path, presets
):
    graph, scene, _ = _region_graph_scene(tmp_path)
    item = bundle._program_payload(graph, "strict", scene, stability_presets=presets)[
        "program"
    ]["items"][0]
    assert item["validators"][0]["kind"] == "object_near_relative_target"


def test_named_support_does_not_acquire_region_acceptance(tmp_path: Path):
    graph, scene, embodiment = _region_graph_scene(tmp_path)
    graph["nodes"][0]["call"]["arguments"]["reference"] = "payload"
    presets = bundle._task_stability_payload(graph, scene, embodiment)["presets"]
    assert presets["gen_sim.place.stable"]["kind"] == "supported_placement"
    assert "displacement" in presets["gen_sim.place.stable"]


@pytest.mark.parametrize(
    "extra", [{"target": "original_pose"}, {"reference_part": "blue_region"}]
)
def test_restrictive_extra_argument_is_not_a_bare_region(tmp_path: Path, extra):
    graph, scene, embodiment = _region_graph_scene(tmp_path)
    graph["nodes"][0]["call"]["arguments"].update(extra)
    presets = bundle._task_stability_payload(graph, scene, embodiment)["presets"]
    assert presets["gen_sim.place.stable"]["kind"] == "supported_placement"


def test_region_preset_for_a_different_entity_cannot_drop_point_validator(
    tmp_path: Path,
):
    graph, scene, _ = _region_graph_scene(tmp_path)
    presets = {
        "gen_sim.place.stable": {
            "kind": "supported_region",
            "entity": "another_object",
            "reference": "table",
        }
    }
    item = bundle._program_payload(graph, "strict", scene, stability_presets=presets)[
        "program"
    ]["items"][0]
    assert item["validators"][0]["kind"] == "object_near_relative_target"
