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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import trimesh
import numpy as np

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from embodichain.gen_sim.task_engine.agent import TaskAgent
from embodichain.gen_sim.task_engine.interpretation import _INTENT_FIELD_DEFAULTS
from embodichain.gen_sim.task_engine.orchestration.contracts import ROLE_BINDINGS_SCHEMA
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine.task_program_bundle import (
    _scene_payload,
    generate_task_program_bundle,
)
from embodichain.gen_sim.task_engine._task_program.articulation_binding import (
    PARK_CALL,
    PrismaticBinding,
    SLIDE_CALL,
    WITHDRAW_CALL,
    discover_prismatic_parts,
    inspect_prismatic,
    validate_placement,
)
from embodichain.gen_sim.task_engine._task_program.articulation_slide import (
    ArticulationSlideFactory,
    ArticulationWithdrawFactory,
    ArticulationStabilityPort,
    preset_id,
    synchronize_joint_limits,
)
from embodichain.gen_sim.task_engine._task_program.assembly import load_deployment
from embodichain.gen_sim.task_engine._bundle_runner import _verify_program_projection
from embodichain.gen_sim.task_engine._task_program.configured import decode_task_lowerer
from embodichain.lab.sim.atomic_actions import (
    JointPositionCommand,
    MoveEndEffectorOptions,
    SlideGoal,
    SlideOptions,
)
from embodichain.lab.sim.cfg import ArticulationCfg
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    SceneArticulationRef,
)
from embodichain.utils.utility import load_config
from embodichain.gen_sim.task_engine.scene.articulation_geometry import (
    read_articulation_geometry,
    fit_articulation_to_proxy,
)
from embodichain.gen_sim.task_engine.scene.final_inspection import _measure_geometry


@pytest.fixture
def scene(tmp_path: Path) -> PreparedScene:
    path = tmp_path / "drawer.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/fixture")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    for name in ("cabinet", "drawer"):
        body = UsdGeom.Xform.Define(stage, f"/fixture/{name}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    shape = trimesh.creation.box(extents=(0.15, 0.04, 0.04))
    mesh = UsdGeom.Mesh.Define(stage, "/fixture/drawer/handle")
    mesh.CreatePointsAttr(shape.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(shape.faces))
    mesh.CreateFaceVertexIndicesAttr(shape.faces.flatten().tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    for name in ("handle_mount_left", "handle_post_left", "handle_leg_left"):
        support = UsdGeom.Mesh.Define(stage, f"/fixture/drawer/{name}")
        support.CreatePointsAttr(shape.vertices.tolist())
        support.CreateFaceVertexCountsAttr([3] * len(shape.faces))
        support.CreateFaceVertexIndicesAttr(shape.faces.flatten().tolist())
        UsdPhysics.CollisionAPI.Apply(support.GetPrim())
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/fixture/slide")
    joint.CreateBody0Rel().SetTargets(["/fixture/cabinet"])
    joint.CreateBody1Rel().SetTargets(["/fixture/drawer"])
    joint.CreateAxisAttr("Y")
    joint.CreateLowerLimitAttr(-0.4)
    joint.CreateUpperLimitAttr(0.0)
    stage.GetRootLayer().Save()
    drawer = {
        "uid": "drawer",
        "fpath": str(path),
        "fix_base": True,
        "body_scale": [0.5] * 3,
        "init_pos": [0.1, 0.0, 0.8],
        "init_rot": [0.0, 0.0, 0.0],
    }
    table = {
        "uid": "table",
        "shape": {"shape_type": "Cube", "size": [1.0, 1.0, 0.72]},
        "init_pos": [0.0, 0.0, 0.36],
    }
    return PreparedScene(
        source_config_path=tmp_path / "scene.json",
        scene_dir=tmp_path,
        planner_objects=(
            {**drawer, "runtime_uid": "drawer", "role": "articulation"},
            {**table, "runtime_uid": "table", "role": "background"},
        ),
        background=(table,),
        rigid_objects=(),
        articulations=(drawer,),
        uid_map={"drawer": "drawer", "table": "table"},
        table_top_z=0.72,
        z_rotation_degrees=0.0,
        body_scale_policy="preserve",
        body_scale=(1.0, 1.0, 1.0),
        asset_hashes={},
    )


@pytest.mark.parametrize("fixed", [True, False])
def test_handle_discovery_follows_only_fixed_child_links(
    scene: PreparedScene, fixed: bool
) -> None:
    stage = Usd.Stage.Open(scene.articulations[0]["fpath"])
    child = UsdGeom.Xform.Define(stage, "/fixture/grip_link")
    UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
    Sdf.CopySpec(
        stage.GetRootLayer(),
        "/fixture/drawer/handle",
        stage.GetRootLayer(),
        "/fixture/grip_link/handle",
    )
    stage.RemovePrim("/fixture/drawer/handle")
    joint_type = UsdPhysics.FixedJoint if fixed else UsdPhysics.RevoluteJoint
    joint = joint_type.Define(stage, "/fixture/grip_joint")
    joint.CreateBody0Rel().SetTargets(["/fixture/drawer"])
    joint.CreateBody1Rel().SetTargets(["/fixture/grip_link"])
    stage.GetRootLayer().Save()
    if not fixed:
        with pytest.raises(ValueError, match="handle candidate"):
            discover_prismatic_parts(scene.articulations[0])
        return
    parts = discover_prismatic_parts(scene.articulations[0])
    assert len(parts) == 1
    assert parts[0].link == "drawer"
    assert parts[0].handle_paths == ("/fixture/grip_link/handle",)
    binding = inspect_prismatic(scene.articulations[0])
    assert binding.link == "grip_link"
    assert binding.joint == "slide"
    assert binding.parent == "cabinet"


def test_composite_legacy_handle_prefers_grip_over_structural_members(
    scene: PreparedScene,
) -> None:
    stage = Usd.Stage.Open(scene.articulations[0]["fpath"])
    source = "/fixture/drawer/handle"
    for name in (
        "handle_left_standoff",
        "handle_left_drop",
        "handle_right_standoff",
        "handle_right_drop",
        "handle_grip",
    ):
        Sdf.CopySpec(
            stage.GetRootLayer(),
            source,
            stage.GetRootLayer(),
            f"/fixture/drawer/{name}",
        )
    stage.RemovePrim(source)
    stage.GetRootLayer().Save()

    part = discover_prismatic_parts(scene.articulations[0])[0]
    assert part.handle_paths == ("/fixture/drawer/handle_grip",)
    binding = inspect_prismatic(scene.articulations[0])
    assert binding.handle_path == "/fixture/drawer/handle_grip"
    for name in ("handle_left_standoff", "handle_right_drop"):
        prim = stage.GetPrimAtPath(f"/fixture/drawer/{name}")
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()


def test_composite_legacy_handle_keeps_multiple_real_grips_ambiguous(
    scene: PreparedScene,
) -> None:
    stage = Usd.Stage.Open(scene.articulations[0]["fpath"])
    source = "/fixture/drawer/handle"
    for name in ("handle_grip_left", "handle_grip_right"):
        Sdf.CopySpec(
            stage.GetRootLayer(),
            source,
            stage.GetRootLayer(),
            f"/fixture/drawer/{name}",
        )
    stage.RemovePrim(source)
    stage.GetRootLayer().Save()

    part = discover_prismatic_parts(scene.articulations[0])[0]
    assert len(part.handle_paths) == 2
    with pytest.raises(ValueError, match="one unambiguous handle"):
        inspect_prismatic(scene.articulations[0])


def test_fixed_handle_binding_expresses_axis_in_handle_frame(
    scene: PreparedScene,
) -> None:
    stage = Usd.Stage.Open(scene.articulations[0]["fpath"])
    child = UsdGeom.Xform.Define(stage, "/fixture/grip_link")
    child.AddRotateZOp().Set(90.0)
    UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
    Sdf.CopySpec(
        stage.GetRootLayer(),
        "/fixture/drawer/handle",
        stage.GetRootLayer(),
        "/fixture/grip_link/handle",
    )
    stage.RemovePrim("/fixture/drawer/handle")
    joint = UsdPhysics.FixedJoint.Define(stage, "/fixture/grip_joint")
    joint.CreateBody0Rel().SetTargets(["/fixture/drawer"])
    joint.CreateBody1Rel().SetTargets(["/fixture/grip_link"])
    stage.GetRootLayer().Save()
    binding = inspect_prismatic(scene.articulations[0])
    assert binding.link == "grip_link"
    assert np.allclose(binding.axis, (1.0, 0.0, 0.0), atol=1e-7)


@pytest.mark.parametrize(
    "limits,closed", [((0.0, 0.4), 0.4), ((-0.4, 0.0), -0.4), ((-0.1, 0.4), 0.4)]
)
def test_explicit_closed_endpoint_controls_state_semantics(
    scene: PreparedScene, limits: tuple[float, float], closed: float
) -> None:
    stage = Usd.Stage.Open(scene.articulations[0]["fpath"])
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/fixture/slide"))
    joint.GetLowerLimitAttr().Set(limits[0])
    joint.GetUpperLimitAttr().Set(limits[1])
    joint.GetPrim().CreateAttribute(
        "gen_sim:closedPosition", Sdf.ValueTypeNames.Double
    ).Set(closed)
    stage.GetRootLayer().Save()
    binding = inspect_prismatic(scene.articulations[0])
    assert binding.target("closed") == pytest.approx(closed * 0.5)
    opened = max(limits, key=lambda value: abs(value - closed))
    assert binding.target("open") == pytest.approx(opened * 0.5)
    assert PrismaticBinding.decode(binding.payload()) == binding


@pytest.mark.parametrize("closed", [0.1, -0.1, float("nan"), True])
def test_invalid_explicit_closed_endpoint_is_rejected(
    scene: PreparedScene, closed
) -> None:
    binding = inspect_prismatic(scene.articulations[0])
    with pytest.raises(ValueError, match="closed_position"):
        replace(binding, closed_position=closed)


def test_unannotated_binding_keeps_legacy_payload(scene: PreparedScene) -> None:
    binding = inspect_prismatic(scene.articulations[0])
    assert "closed_position" not in binding.payload()
    assert PrismaticBinding.decode(binding.payload()) == binding


@pytest.mark.parametrize("state", ["open", "closed"])
@pytest.mark.parametrize(
    "limits,expected",
    [((0.0, 0.07367365696480928), 0.03), ((0.0, 0.04), 0.0196)],
)
def test_articulation_state_tolerance_is_metric_but_keeps_endpoints_disjoint(
    scene: PreparedScene,
    state: str,
    limits: tuple[float, float],
    expected: float,
) -> None:
    binding = replace(
        inspect_prismatic(scene.articulations[0]),
        limits=limits,
        closed_position=limits[1],
    )
    assert binding.tolerance(state) == pytest.approx(expected)
    assert binding.tolerance(state) < (limits[1] - limits[0]) / 2


def _graph(scene: PreparedScene, states: tuple[str, ...] = ("open",)) -> dict:
    empty = {
        "kind": "none",
        "reference": "",
        "step_id": "",
        "quantifier": "one",
        "count": 0,
    }
    steps = [
        {
            **deepcopy(_INTENT_FIELD_DEFAULTS),
            "id": f"step_{i+1:02d}",
            "task_type": "E6",
            "object": {**empty, "kind": "scene_ref", "reference": "drawer"},
            "target": empty,
            "required_arm": "right_arm" if i == 0 else "left_arm",
            "target_state": state,
            "depends_on": [] if i == 0 else [f"step_{i:02d}"],
        }
        for i, state in enumerate(states)
    ]
    candidate = TaskAgent(caller=lambda **kw: {"steps": steps}).generate(
        "slide", "Open and close the drawer", candidate_count=1
    )["candidates"][0]
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": "slide",
        "candidate_id": candidate["candidate_id"],
        "reference_bindings": {
            f"step_{i+1:02d}.object": ["drawer"] for i in range(len(states))
        },
        "role_bindings": {},
    }
    return SemanticTaskPlanner().plan(candidate, bindings, scene.planner_objects)


def test_discover_prismatic_parts_returns_all_enabled_parts(
    scene: PreparedScene,
) -> None:
    stage = Usd.Stage.Open(scene.articulations[0]["fpath"])
    right = UsdGeom.Xform.Define(stage, "/fixture/right_drawer")
    UsdPhysics.RigidBodyAPI.Apply(right.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/fixture/right_drawer/right_handle")
    shape = trimesh.creation.box(extents=(0.15, 0.04, 0.04))
    mesh.CreatePointsAttr(shape.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(shape.faces))
    mesh.CreateFaceVertexIndicesAttr(shape.faces.flatten().tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/fixture/right_slide")
    joint.CreateBody0Rel().SetTargets(["/fixture/cabinet"])
    joint.CreateBody1Rel().SetTargets(["/fixture/right_drawer"])
    joint.CreateAxisAttr("Y")
    joint.CreateLowerLimitAttr(-0.2)
    joint.CreateUpperLimitAttr(0.0)
    stage.GetRootLayer().Save()

    parts = discover_prismatic_parts(scene.articulations[0])
    assert len(parts) == 2
    assert {part.joint for part in parts} == {"slide", "right_slide"}
    assert all(part.part_id.startswith("part_") for part in parts)
    assert all(len(part.handle_paths) == 1 for part in parts)


def test_multi_part_bundle_keeps_distinct_bindings(
    scene: PreparedScene, tmp_path: Path
) -> None:
    stage = Usd.Stage.Open(scene.articulations[0]["fpath"])
    right = UsdGeom.Xform.Define(stage, "/fixture/right_drawer")
    UsdPhysics.RigidBodyAPI.Apply(right.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/fixture/right_drawer/right_handle")
    shape = trimesh.creation.box(extents=(0.15, 0.04, 0.04))
    mesh.CreatePointsAttr(shape.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(shape.faces))
    mesh.CreateFaceVertexIndicesAttr(shape.faces.flatten().tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/fixture/right_slide")
    joint.CreateBody0Rel().SetTargets(["/fixture/cabinet"])
    joint.CreateBody1Rel().SetTargets(["/fixture/right_drawer"])
    joint.CreateAxisAttr("Y")
    joint.CreateLowerLimitAttr(-0.2)
    joint.CreateUpperLimitAttr(0.0)
    joint.GetPrim().CreateAttribute(
        "gen_sim:closedPosition", Sdf.ValueTypeNames.Double
    ).Set(0.0)
    stage.GetRootLayer().Save()
    parts = discover_prismatic_parts(scene.articulations[0])
    part_ids = {part.joint: part.part_id for part in parts}
    empty = {
        "kind": "none",
        "reference": "",
        "step_id": "",
        "quantifier": "one",
        "count": 0,
    }
    steps = [
        {
            **deepcopy(_INTENT_FIELD_DEFAULTS),
            "id": "step_01",
            "task_type": "E6",
            "object": {**empty, "kind": "scene_ref", "reference": "drawer"},
            "target": empty,
            "required_arm": "right_arm",
            "target_state": "open",
            "depends_on": [],
        },
        {
            **deepcopy(_INTENT_FIELD_DEFAULTS),
            "id": "step_02",
            "task_type": "E6",
            "object": {**empty, "kind": "scene_ref", "reference": "drawer"},
            "target": empty,
            "required_arm": "left_arm",
            "target_state": "closed",
            "depends_on": ["step_01"],
        },
    ]
    candidate = TaskAgent(caller=lambda **kw: {"steps": steps}).generate(
        "multi", "operate both drawers", candidate_count=1
    )["candidates"][0]
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": "multi",
        "candidate_id": candidate["candidate_id"],
        "reference_bindings": {
            "step_01.object": ["drawer"],
            "step_02.object": ["drawer"],
        },
        "role_bindings": {
            "step_01.object": part_ids["slide"],
            "step_02.object": part_ids["right_slide"],
        },
    }
    graph = SemanticTaskPlanner().plan(candidate, bindings, scene.planner_objects)
    _, paths = generate_task_program_bundle(
        graph, scene, tmp_path / "multi_bundle", robot_profile="dual_franka"
    )
    integration = load_config(paths.integration)
    assert len(integration["scene_binding"]["links"]) == 2
    lowerers = integration["runtime_services"]["registered_semantic_lowerers"]
    assert len(lowerers[0]["bindings"]) == 2


@pytest.mark.parametrize("states", [("open",), ("closed",), ("open", "closed")])
def test_e6_bundle_uses_standard_registration_and_complete_recipe(
    scene: PreparedScene, tmp_path: Path, states: tuple
) -> None:
    if "closed" in states:
        stage = Usd.Stage.Open(scene.articulations[0]["fpath"])
        stage.GetPrimAtPath("/fixture/slide").CreateAttribute(
            "gen_sim:closedPosition", Sdf.ValueTypeNames.Double
        ).Set(0.0)
        stage.GetRootLayer().Save()
    graph, paths = generate_task_program_bundle(
        _graph(scene, states), scene, tmp_path / "bundle", robot_profile="dual_franka"
    )
    _verify_program_projection(paths.program, graph)
    assert [n["call"]["call_id"] for n in graph["nodes"]] == [
        SLIDE_CALL,
        WITHDRAW_CALL,
        PARK_CALL,
    ] * len(states)
    assert [n["role"] for n in graph["nodes"]] == [
        "primary",
        "cleanup",
        "cleanup",
    ] * len(states)
    integration = load_config(paths.integration)
    assert integration["scene_binding"]["articulations"] == [
        {"entity_id": "drawer", "simulation_uid": "drawer"}
    ]
    options = integration["profile"]["action_options"][SLIDE_CALL]
    assert options["preshape_fraction"] == 0.85
    assert options["approach_along_grasp_axis"] is True
    assert options["release_retreat_distance"] == 0.04
    embodiment = load_config(paths.embodiment)["simulation"]
    assert embodiment["joint_drive_props"]["stiffness"]["left_arm"] == 50000.0
    assert embodiment["joint_drive_props"]["stiffness"]["left_eef"] == 50.0
    assert "mimic_pros" not in embodiment
    articulation = load_config(paths.scene)["simulation"]["articulation"][0]
    assert articulation["joint_drive_props"] == {
        "drive_type": "none",
        "friction": {"slide": 0.01},
    }
    assert articulation["root_props"] == {"fixed_base": True}
    assert articulation["asset_physics_mode"] == "overlay"
    assert articulation["link_attrs"] == {
        "gen_sim_prismatic_link_mass": {
            "link_names_expr": ["drawer"],
            "attrs": {"mass_props": {"mass": 1.0}},
        }
    }
    assert "fix_base" not in articulation
    assert "proxy_init_pos" not in articulation
    assert "joint_drive_props" not in scene.articulations[0]
    deployment = load_deployment(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.root,
    )
    assert deployment.integration.registration.catalog is not None
    assert len(deployment.integration.adapter_factory.articulation_bindings) == 1
    items = load_config(paths.program)["program"]["items"]
    assert all(
        item["validators"][0]["kind"] == "articulation_joint_position" for item in items
    )
    if len(states) == 2:
        assert graph["nodes"][3]["depends_on"] == [graph["nodes"][2]["id"]]
        assert graph["nodes"][3]["call"]["resources"] == {"primary": "left"}


def test_e6_closed_target_rejects_unannotated_endpoint(
    scene: PreparedScene, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="explicit closed endpoint"):
        generate_task_program_bundle(
            _graph(scene, ("closed",)),
            scene,
            tmp_path / "bundle",
            robot_profile="dual_franka",
        )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("friction", [0.0, 0.03, {"slide": 0.02}])
def test_e6_preserves_authored_passive_friction(
    scene: PreparedScene, tmp_path: Path, friction: object
) -> None:
    config = deepcopy(scene.articulations[0])
    config["joint_drive_props"] = {"drive_type": "none", "friction": friction}
    prepared = replace(scene, articulations=(config,))
    _, paths = generate_task_program_bundle(
        _graph(prepared), prepared, tmp_path / "bundle", robot_profile="dual_franka"
    )
    assert (
        load_config(paths.scene)["simulation"]["articulation"][0]["joint_drive_props"][
            "friction"
        ]
        == friction
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "wrong_type",
        "handle",
        "scale",
        "base",
        "incomplete",
        "dependency",
        "penetration",
        "tilt",
    ],
)
def test_e6_rejects_invalid_assets_or_recipes_before_publication(
    scene: PreparedScene, tmp_path: Path, mutation: str
) -> None:
    graph = _graph(scene)
    cfg = deepcopy(scene.articulations[0])
    if mutation == "missing":
        scene = replace(scene, articulations=())
    elif mutation == "scale":
        cfg["body_scale"] = [1.0, 2.0, 1.0]
    elif mutation == "base":
        cfg["fix_base"] = False
    elif mutation == "penetration":
        cfg["init_pos"][2] = 0.70
    elif mutation == "tilt":
        cfg["init_rot"][0] = 2.0
    elif mutation == "incomplete":
        graph["nodes"][1]["call"]["call_id"] = PARK_CALL
    elif mutation == "dependency":
        graph["nodes"][1]["depends_on"] = []
    else:
        stage = Usd.Stage.Open(cfg["fpath"])
        if mutation == "wrong_type":
            stage.GetPrimAtPath("/fixture/slide").SetTypeName("PhysicsRevoluteJoint")
        else:
            extra = UsdGeom.Mesh.Define(stage, "/fixture/drawer/second_handle")
            UsdPhysics.CollisionAPI.Apply(extra.GetPrim())
        stage.GetRootLayer().Save()
    if mutation != "missing":
        scene = replace(scene, articulations=(cfg,))
    with pytest.raises(ValueError):
        generate_task_program_bundle(
            graph, scene, tmp_path / "bundle", robot_profile="dual_franka"
        )
    assert not (tmp_path / "bundle").exists()


def test_articulation_geometry_matches_runtime_root_not_usd_world(
    scene: PreparedScene,
) -> None:
    cfg = scene.articulations[0]
    before = read_articulation_geometry(cfg["fpath"])
    stage = Usd.Stage.Open(cfg["fpath"])
    UsdGeom.Xformable(stage.GetDefaultPrim()).AddTranslateOp().Set((0, 0, -12))
    stage.GetRootLayer().Save()
    after = read_articulation_geometry(cfg["fpath"])
    assert before.root_link == after.root_link == "cabinet"
    np.testing.assert_allclose(before.vertices, after.vertices)
    measured = _measure_geometry(cfg, convert_y_up=True)
    assert measured is not None
    assert measured["bounds"][0, 2] == pytest.approx(0.79)
    assert measured["bounds"][1, 2] == pytest.approx(0.81)


@pytest.mark.parametrize("axis,units", [("Y", 1.0), ("Z", 0.01)])
def test_articulation_geometry_rejects_unsupported_frame(
    scene: PreparedScene, axis: str, units: float
) -> None:
    path = scene.articulations[0]["fpath"]
    stage = Usd.Stage.Open(path)
    UsdGeom.SetStageUpAxis(stage, axis)
    UsdGeom.SetStageMetersPerUnit(stage, units)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="Z-up.*metre"):
        read_articulation_geometry(path)


def test_explicit_proxy_fit_preserves_source_and_uniform_shape(
    scene: PreparedScene,
) -> None:
    cfg = scene.articulations[0]
    original = deepcopy(cfg)
    source_bytes = Path(cfg["fpath"]).read_bytes()
    geometry = read_articulation_geometry(cfg["fpath"])
    proxy = geometry.vertices * 2
    proxy[:, 0] += 0.1
    repaired = fit_articulation_to_proxy(cfg, proxy, table_top_z=0.72)
    assert cfg == original and Path(cfg["fpath"]).read_bytes() == source_bytes
    assert repaired["body_scale"] == pytest.approx([2, 2, 2])
    measured = _measure_geometry(repaired, convert_y_up=True)
    assert measured["bounds"][0, 2] == pytest.approx(0.721)
    assert measured["bounds"][:, 0].mean() == pytest.approx(0.2)
    with pytest.raises(ValueError):
        fit_articulation_to_proxy(cfg, proxy * 0, table_top_z=0.72)


def test_articulation_geometry_rejects_nonfinite_transform(
    scene: PreparedScene,
) -> None:
    path = scene.articulations[0]["fpath"]
    stage = Usd.Stage.Open(path)
    UsdGeom.Xformable(stage.GetDefaultPrim()).AddTranslateOp().Set((0, 0, float("nan")))
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="finite"):
        read_articulation_geometry(path)


def test_explicit_tabletop_fit_levels_only_small_pose_errors(
    scene: PreparedScene,
) -> None:
    cfg = deepcopy(scene.articulations[0])
    cfg["init_rot"] = [0.02, -0.01, 90]
    proxy = read_articulation_geometry(cfg["fpath"]).vertices
    result = fit_articulation_to_proxy(cfg, proxy, table_top_z=0.72)
    assert result["init_rot"][:2] == [0.0, 0.0]
    assert result["init_rot"][2] == pytest.approx(90, abs=0.01)
    assert cfg["init_rot"] == [0.02, -0.01, 90]
    cfg["init_rot"] = [20, 0, 90]
    with pytest.raises(ValueError, match="one degree"):
        fit_articulation_to_proxy(cfg, proxy, table_top_z=0.72)


@pytest.mark.parametrize("target_index", [0, 1])
def test_limit_sync_does_not_overwrite_other_joints(target_index: int) -> None:
    names = ["other", "other"]
    names[target_index] = "selected"
    cached = torch.tensor([[[-0.105, 0.0], [-0.105, 0.0]]])
    native = torch.tensor([[[-0.0735, 0.0], [-0.0735, 0.0]]])
    before = native.clone()
    calls = []

    def write_limits(values, joint_ids=None):
        ids = list(range(2)) if joint_ids is None else list(joint_ids)
        calls.append(ids)
        native[:, ids, :] = values
        cached[:, ids, :] = values

    art = SimpleNamespace(
        joint_names=names,
        get_qpos_limits=lambda: cached.clone(),
        set_qpos_limits=write_limits,
        get_parent_joint_chain=lambda _: [
            SimpleNamespace(
                name="selected",
                joint_type="prismatic",
                parent_link_name="cabinet",
                joint_limits=(-0.105, 0.0),
            )
        ],
    )
    binding = SimpleNamespace(
        joint="selected",
        link="handle",
        parent="cabinet",
        limits=(-0.0735, 0.0),
        scale=0.7,
        part_id="part_selected",
    )
    synchronize_joint_limits(binding, art)
    assert calls == [[target_index]]
    assert torch.equal(native, before)
    synchronize_joint_limits(binding, art)
    assert len(calls) == 1


def _runtime(scene: PreparedScene):
    cfg = scene.articulations[0]
    binding = inspect_prismatic(cfg)
    position = torch.tensor([[binding.target("open")]])
    cached_limits = torch.tensor([[[-0.4, 0.0]]])
    runtime_cfg = dict(cfg)
    runtime_cfg.setdefault("init_local_pose", None)
    fixed_base = runtime_cfg.pop("fix_base")
    art = SimpleNamespace(
        cfg=SimpleNamespace(
            **runtime_cfg,
            root_props=SimpleNamespace(fixed_base=fixed_base),
        ),
        joint_names=["slide"],
        get_qpos=lambda: position.clone(),
        get_qpos_limits=lambda: cached_limits.clone(),
        set_qpos_limits=lambda values, joint_ids=None: cached_limits.copy_(values),
        get_parent_joint_chain=lambda name: [
            SimpleNamespace(
                name="slide",
                joint_type="prismatic",
                parent_link_name="cabinet",
                joint_limits=binding.limits,
            )
        ],
        get_link_pose=lambda *a, **kw: torch.eye(4)[None],
    )
    robot = SimpleNamespace(
        get_qpos=lambda: torch.zeros(1, 8), compute_fk=lambda **kw: torch.eye(4)[None]
    )
    table = SimpleNamespace(
        get_local_pose=lambda **kw: torch.eye(4)[None],
        get_vertices=lambda **kw: torch.tensor([[[0.0, 0.0, 0.72]]]),
    )
    simulation = SimpleNamespace(
        get_articulation=lambda uid: art, num_envs=1, get_rigid_object=lambda uid: table
    )
    registry = SimpleNamespace(
        lookup=lambda *a, **kw: SimpleNamespace(
            parent=SceneArticulationRef("drawer"), native_name="drawer"
        )
    )
    return (
        binding,
        position,
        art,
        robot,
        dict(
            simulation=simulation,
            robot=robot,
            scene_registry=registry,
            engine=SimpleNamespace(robot=robot, device="cpu"),
        ),
    )


def test_e6_runtime_binding_preserves_decoded_matrix(scene: PreparedScene) -> None:
    """Bundle decode must not feed extrinsic angles back to intrinsic validation."""
    from scipy.spatial.transform import Rotation

    config = deepcopy(scene.articulations[0])
    roll, pitch = np.deg2rad([30.0, 20.0])
    # This compound rotation leaves the local Y slide rail horizontal.
    yaw = np.arctan2(-np.sin(roll), np.cos(roll) * np.sin(pitch))
    config["init_rot"] = [30.0, 20.0, float(np.rad2deg(yaw))]
    scene = replace(scene, articulations=(config,))
    binding, _, art, _, kwargs = _runtime(scene)
    validate_placement(binding, config, scene.table_top_z)
    item = _scene_payload(scene, program_id="pose_binding")["simulation"][
        "articulation"
    ][0]
    art.cfg = ArticulationCfg.from_dict(item)
    expected = Rotation.from_euler("XYZ", config["init_rot"], degrees=True).as_matrix()
    np.testing.assert_allclose(art.cfg.init_local_pose[:3, :3], expected, atol=1e-12)
    assert not np.allclose(
        Rotation.from_euler("XYZ", art.cfg.init_rot, degrees=True).as_matrix(),
        expected,
    )

    assert ArticulationSlideFactory((binding,)).create(**kwargs) is not None


@pytest.mark.parametrize("as_list", [True, False])
def test_articulation_world_vertices_prefer_explicit_pose(
    scene: PreparedScene, as_list: bool
) -> None:
    config = deepcopy(scene.articulations[0])
    geometry = read_articulation_geometry(config["fpath"])
    expected = geometry.world_vertices(config)
    pose = np.eye(4)
    pose[:3, 3] = config["init_pos"]
    config["init_local_pose"] = pose.tolist() if as_list else pose
    config["init_pos"] = [9.0, 9.0, -9.0]
    config["init_rot"] = [40.0, 50.0, 60.0]

    np.testing.assert_allclose(geometry.world_vertices(config), expected)
    validate_placement(inspect_prismatic(config), config, scene.table_top_z)


@pytest.mark.parametrize(
    "pose",
    [
        np.eye(3),
        np.full((4, 4), np.nan),
        np.diag([1.0, 1.0, 1.0, 0.0]),
        np.diag([2.0, 1.0, 1.0, 1.0]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
    ],
    ids=["shape", "nonfinite", "nonhomogeneous", "scale", "reflection"],
)
def test_articulation_geometry_rejects_invalid_root_matrix(
    scene: PreparedScene, pose: np.ndarray
) -> None:
    config = {**scene.articulations[0], "init_local_pose": pose.tolist()}
    with pytest.raises(ValueError, match="init_local_pose"):
        read_articulation_geometry(config["fpath"]).world_vertices(config)


@pytest.mark.parametrize("violation", ["penetration", "tilt"])
def test_e6_binding_rejects_invalid_matrix_placement(
    scene: PreparedScene, violation: str
) -> None:
    from scipy.spatial.transform import Rotation

    binding, _, art, _, kwargs = _runtime(scene)
    item = _scene_payload(scene, program_id="invalid_pose")["simulation"][
        "articulation"
    ][0]
    art.cfg = ArticulationCfg.from_dict(item)
    if violation == "penetration":
        art.cfg.init_local_pose[2, 3] = 0.6
        message = "intersects the tabletop"
    else:
        art.cfg.init_local_pose[:3, :3] = Rotation.from_euler(
            "X", 10.0, degrees=True
        ).as_matrix()
        message = "one degree"
    # Leave cfg.init_pos/init_rot safe to prove validation reads the matrix.
    with pytest.raises(ValueError, match=message):
        ArticulationSlideFactory((binding,)).create(**kwargs)


def test_proxy_fit_updates_explicit_root_matrix(scene: PreparedScene) -> None:
    from scipy.spatial.transform import Rotation

    config = deepcopy(scene.articulations[0])
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler(
        "XYZ", [0.02, -0.01, 90], degrees=True
    ).as_matrix()
    pose[:3, 3] = config["init_pos"]
    config["init_local_pose"] = pose.tolist()
    before = deepcopy(config)
    geometry = read_articulation_geometry(config["fpath"])

    repaired = fit_articulation_to_proxy(config, geometry.vertices, table_top_z=0.72)

    np.testing.assert_allclose(
        np.asarray(repaired["init_local_pose"])[:3, 3], repaired["init_pos"]
    )
    assert geometry.world_vertices(repaired)[:, 2].min() == pytest.approx(0.721)
    assert np.asarray(repaired["init_local_pose"])[2, 2] == pytest.approx(1.0)
    assert config == before


def test_lowerers_keep_exact_joint_goal_and_require_open_hand_for_withdrawal(
    scene: PreparedScene,
) -> None:
    binding, position, _, robot, kwargs = _runtime(scene)
    factory = ArticulationSlideFactory((binding,))
    first, second = factory.create(**kwargs), factory.create(**kwargs)
    assert first is not second
    assert kwargs["simulation"].get_articulation("drawer").get_qpos_limits().tolist()[
        0
    ][0] == pytest.approx(binding.limits)
    endpoint = SimpleNamespace(
        task_state_key="right",
        require_target=lambda cls: SimpleNamespace(
            joint_ids=tuple(range(6)), control_part="arm"
        ),
    )
    hand = SimpleNamespace(
        runtime_target=SimpleNamespace(joint_ids=(6, 7)),
        commands={"open": JointPositionCommand(torch.zeros(2))},
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            action_binding=SimpleNamespace(endpoint=lambda *a: endpoint),
            resources={"primary": SimpleNamespace(endpoints={"grasp": hand})},
        )
    )
    context = SimpleNamespace(
        task=SimpleNamespace(
            get_held_object=lambda key: None, coordinated_held_objects={}
        ),
        robot=SimpleNamespace(qpos=torch.zeros(1, 8)),
        env_ids=torch.arange(1),
        batch_size=1,
    )
    call = RegisteredSemanticCall(
        call_id=SLIDE_CALL, arguments={"object": "drawer", "state": "closed"}
    )
    lowering = first.lower(
        call, context=context, bound=bound, option_template=SlideOptions()
    )
    assert type(lowering.goal) is SlideGoal and lowering.skill_options is None
    assert lowering.goal.joint_target.position == 0.0
    assert lowering.goal.joint_target.articulation_id == "drawer"
    withdraw = ArticulationWithdrawFactory((binding,)).create(**kwargs)
    call = RegisteredSemanticCall(
        call_id=WITHDRAW_CALL, arguments={"object": "drawer", "state": "open"}
    )
    withdrawal = withdraw.lower(
        call, context=context, bound=bound, option_template=MoveEndEffectorOptions()
    )
    assert withdrawal.goal.xpos.shape == (1, 3, 4, 4)
    torch.testing.assert_close(withdrawal.goal.xpos[0, 1, 2, 3], torch.tensor(0.12))
    assert torch.linalg.vector_norm(withdrawal.goal.xpos[0, 2, :2, 3]) > 0.09
    context.robot.qpos[:, 6:] = 0.1
    with pytest.raises(ValueError, match="open posture"):
        withdraw.lower(
            call, context=context, bound=bound, option_template=MoveEndEffectorOptions()
        )
    context.robot.qpos[:, 6:] = 0.0
    position += 0.05
    with pytest.raises(ValueError, match="target was lost"):
        withdraw.lower(
            call, context=context, bound=bound, option_template=MoveEndEffectorOptions()
        )
    position[:] = binding.target("open")
    hand.commands.clear()
    with pytest.raises(ValueError, match="hand-open command"):
        withdraw.lower(
            call, context=context, bound=bound, option_template=MoveEndEffectorOptions()
        )
    context.task.coordinated_held_objects[("left", "right")] = SimpleNamespace(
        env_mask=torch.tensor([True])
    )
    with pytest.raises(ValueError, match="coordinated holding"):
        first.lower(call, context=context, bound=bound, option_template=SlideOptions())
    context.task.coordinated_held_objects.clear()
    context.task.get_held_object = lambda key: SimpleNamespace(env_mask=None)
    with pytest.raises(ValueError, match="holding another object"):
        first.lower(call, context=context, bound=bound, option_template=SlideOptions())


@pytest.mark.parametrize(
    "mutation",
    [
        "multiple_envs",
        "asset_changed",
        "native_joint",
        "native_limits",
        "runtime_table",
    ],
)
def test_articulation_factory_rejects_runtime_binding_drift(
    scene: PreparedScene, mutation: str
) -> None:
    binding, _, art, _, kwargs = _runtime(scene)
    if mutation == "multiple_envs":
        kwargs["simulation"].num_envs = 2
    elif mutation == "native_joint":
        art.joint_names = ["another_joint"]
    elif mutation == "native_limits":
        art.get_parent_joint_chain = lambda name: [
            SimpleNamespace(
                name="slide",
                joint_type="prismatic",
                parent_link_name="cabinet",
                joint_limits=(-0.4, 0.0),
            )
        ]
    elif mutation == "runtime_table":
        kwargs["simulation"].get_rigid_object("table").get_vertices = (
            lambda **kw: torch.tensor([[[0.0, 0.0, 0.9]]])
        )
    else:
        stage = Usd.Stage.Open(art.cfg.fpath)
        stage.SetMetadata("comment", "Changed after bundle generation")
        stage.GetRootLayer().Save()
    with pytest.raises(ValueError):
        ArticulationSlideFactory((binding,)).create(**kwargs)


@pytest.mark.parametrize("drift,expected", [(0.0, True), (0.002, False)])
def test_articulation_post_policy_measures_retention_not_execution_state(
    scene: PreparedScene, drift: float, expected: bool
) -> None:
    binding, qpos, _, robot, kwargs = _runtime(scene)
    port = ArticulationStabilityPort(
        None, kwargs["simulation"], robot, (binding,), 0.04
    )
    policy = SimpleNamespace(
        cfg=SimpleNamespace(kind="wait_stable", preset=preset_id(binding, "open")),
        entity=SimpleNamespace(entity_id="drawer"),
    )
    segment = SimpleNamespace(post_policies=(policy,))
    for _ in port.actions(policy, segment=segment, active_mask=torch.tensor([True])):
        qpos += drift
    assert port.post_policy_result(policy, segment=segment).tolist() == [expected]


def test_articulation_declarations_reject_unknown_fields(scene: PreparedScene) -> None:
    value = inspect_prismatic(scene.articulations[0]).payload()
    with pytest.raises(ValueError):
        PrismaticBinding.decode({**value, "class_type": "untrusted.Code"})
    with pytest.raises(ValueError):
        decode_task_lowerer(
            {"kind": "articulation_slide", "bindings": [value], "options": {}},
            path="test",
        )


@pytest.fixture
def drawer_container_scene(scene: PreparedScene, tmp_path: Path) -> PreparedScene:
    stage = Usd.Stage.Open(scene.articulations[0]["fpath"])
    stage.GetPrimAtPath("/fixture/slide").CreateAttribute(
        "gen_sim:closedPosition", Sdf.ValueTypeNames.Double
    ).Set(0.0)
    for name, size, center in (
        ("floor", (0.32, 0.42, 0.02), (0, 0.25, 0)),
        ("left", (0.02, 0.42, 0.14), (-0.16, 0.25, 0.08)),
        ("right", (0.02, 0.42, 0.14), (0.16, 0.25, 0.08)),
        ("front", (0.30, 0.02, 0.14), (0, 0.04, 0.08)),
        ("back", (0.30, 0.02, 0.14), (0, 0.46, 0.08)),
    ):
        shape = trimesh.creation.box(extents=size)
        shape.apply_translation(center)
        mesh = UsdGeom.Mesh.Define(stage, f"/fixture/drawer/{name}")
        mesh.CreatePointsAttr(shape.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(shape.faces))
        mesh.CreateFaceVertexIndicesAttr(shape.faces.flatten().tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    cube_path = tmp_path / "cube.glb"
    trimesh.creation.box(extents=[0.03] * 3).export(cube_path)
    cube = {
        "uid": "cube",
        "runtime_uid": "cube",
        "role": "rigid_object",
        "category": "cube",
        "shape": {"shape_type": "Mesh", "fpath": str(cube_path)},
        "init_pos": [0.1, 0.2, 0.75],
        "init_rot": [0.0] * 3,
        "body_scale": [1.0] * 3,
    }
    return replace(
        scene,
        planner_objects=(*scene.planner_objects, cube),
        rigid_objects=(cube,),
        uid_map={**scene.uid_map, "cube": "cube"},
    )


def _drawer_composite_graph(scene: PreparedScene) -> dict:
    empty = {
        "kind": "none",
        "reference": "",
        "step_id": "",
        "quantifier": "one",
        "count": 0,
    }
    opened = {
        **deepcopy(_INTENT_FIELD_DEFAULTS),
        "id": "open",
        "task_type": "E6",
        "object": {**empty, "kind": "scene_ref", "reference": "drawer"},
        "target": empty,
        "required_arm": "right_arm",
        "target_state": "open",
        "depends_on": [],
    }
    result = {**empty, "kind": "step_result", "step_id": "open"}
    placed = {
        **deepcopy(_INTENT_FIELD_DEFAULTS),
        "id": "place",
        "task_type": "E1",
        "object": {**empty, "kind": "scene_ref", "reference": "cube"},
        "target": result,
        "relation": "inside",
        "required_arm": "right_arm",
        "depends_on": ["open"],
    }
    closed = {
        **opened,
        "id": "close",
        "object": result,
        "target_state": "closed",
        "depends_on": ["place"],
    }
    candidate = TaskAgent(
        caller=lambda **kw: {"steps": [opened, placed, closed]}
    ).generate(
        "drawer_composite",
        "Open the drawer, put cube inside, close it",
        candidate_count=1,
    )[
        "candidates"
    ][
        0
    ]
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": "drawer_composite",
        "candidate_id": candidate["candidate_id"],
        "reference_bindings": {
            "step_01.object": ["drawer"],
            "step_02.object": ["cube"],
        },
        "role_bindings": {},
    }
    return SemanticTaskPlanner().plan(candidate, bindings, scene.planner_objects)


def test_drawer_open_place_close_builds_one_link_owned_container(
    drawer_container_scene: PreparedScene, tmp_path: Path
) -> None:
    graph, paths = generate_task_program_bundle(
        _drawer_composite_graph(drawer_container_scene),
        drawer_container_scene,
        tmp_path / "composite",
        robot_profile="dual_franka",
    )
    _verify_program_projection(paths.program, graph)
    deployment = load_deployment(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.root,
    )
    binding = deployment.integration.registration.scene_binding
    assert "drawer" not in {b.entity_id for b in binding.rigid_objects}
    assert len(binding.articulations) == len(binding.containers) == 1
    route = deployment.integration.adapter_factory.drawer_routes[0]
    assert load_config(paths.execution_policy)["motion"]["sample_count"] >= 260
    assert binding.containers[0].parent_id == route.binding.link_id
    assert route.binding.part_id
    transport = [
        n
        for n in graph["nodes"]
        if n["call"].get("call_id") == "simulation.move_held_object"
        and n["call"]["arguments"]["target"] == route.affordance
    ]
    assert len(transport) == 1
    place_node = next(n for n in graph["nodes"] if n["call"].get("kind") == "place")
    assert place_node["depends_on"] == [transport[0]["id"]]
    slides = [
        n["call"]["arguments"]
        for n in graph["nodes"]
        if n["call"].get("call_id") == SLIDE_CALL
    ]
    assert [a["part"] for a in slides] == [route.binding.part_id] * 2
    items = load_config(paths.program)["program"]["items"]
    placement = next(i for i in items if i["steps"]["call"].get("kind") == "place")
    assert not any(
        p["preset"].startswith("gen_sim.drawer.")
        for item in items
        for p in item.get("post", [])
    )
    assert placement["post"][0]["preset"] == "contained_rigid_object"
    assert items[-1]["validators"][0]["kind"] == "articulation_joint_position"


def test_composite_unifies_implicit_and_explicit_same_part(
    drawer_container_scene: PreparedScene,
) -> None:
    from embodichain.gen_sim.task_engine._task_program.drawer_binding import (
        prepare_drawer_graph,
    )

    graph = _drawer_composite_graph(drawer_container_scene)
    part = discover_prismatic_parts(drawer_container_scene.articulations[0])[0].part_id
    for node in graph["nodes"]:
        args = node["call"].get("arguments", {})
        if args.get("state") == "closed":
            args["part"] = part
    before = deepcopy(graph)
    result = prepare_drawer_graph(graph, drawer_container_scene)
    assert graph == before
    assert next(
        n["call"]["inside"] for n in result["nodes"] if "inside" in n["call"]
    ).endswith(f"__{part}")


def _live_drawer(scene: PreparedScene):
    from embodichain.gen_sim.task_engine._task_program.drawer_binding import (
        prepare_drawer_graph,
        drawer_routes,
    )
    from embodichain.gen_sim.task_engine._task_program.drawer_runtime import (
        DrawerObservation,
    )

    graph = prepare_drawer_graph(_drawer_composite_graph(scene), scene)
    route = drawer_routes(graph, scene)[0]
    link = torch.eye(4).unsqueeze(0)
    obj_pose = link.clone()
    obj_pose[0, :3, 3] = torch.tensor([0.0, 0.125, 0.025])
    vertices = torch.tensor(
        trimesh.creation.box(extents=[0.03] * 3).vertices, dtype=torch.float32
    )
    cfg = scene.articulations[0]
    art = SimpleNamespace(
        cfg=SimpleNamespace(
            fpath=cfg["fpath"],
            body_scale=cfg["body_scale"],
            root_props=SimpleNamespace(fixed_base=True),
        ),
        joint_names=[route.binding.joint],
        get_link_pose=lambda *a, **kw: link.clone(),
        get_qpos=lambda: torch.tensor([[route.binding.target("open")]]),
    )
    obj = SimpleNamespace(
        get_vertices=lambda **kw: vertices.unsqueeze(0),
        get_local_pose=lambda **kw: obj_pose.clone(),
    )
    sim = SimpleNamespace(
        get_articulation=lambda uid: art,
        get_rigid_object=lambda uid: obj,
        get_articulation_uid_list=lambda: [route.binding.object_id],
        get_rigid_object_uid_list=lambda: [],
    )
    return DrawerObservation(route, sim), sim, link, obj_pose


def test_handle_only_geometry_is_not_a_task_rejection(scene: PreparedScene) -> None:
    from embodichain.gen_sim.task_engine._task_program.drawer_geometry import (
        drawer_region,
    )

    low, high = drawer_region(
        scene.articulations[0], inspect_prismatic(scene.articulations[0])
    )
    assert np.isfinite([low, high]).all()


def test_drawer_runtime_target_follows_link(
    drawer_container_scene: PreparedScene,
) -> None:
    obs, _, link, _ = _live_drawer(drawer_container_scene)
    initial = obs.target_pose()
    link[0, 0, 3] += 0.12
    moved = obs.target_pose()
    torch.testing.assert_close(
        moved[0, :3, 3] - initial[0, :3, 3], torch.tensor([0.12, 0, 0])
    )


def test_oversize_drawer_payload_bundle_has_no_containment_gate(
    drawer_container_scene: PreparedScene, tmp_path: Path
) -> None:
    cube = {**drawer_container_scene.rigid_objects[0], "body_scale": [10.0] * 3}
    scene = replace(
        drawer_container_scene,
        rigid_objects=(cube,),
        planner_objects=(*drawer_container_scene.planner_objects[:-1], cube),
    )
    _, paths = generate_task_program_bundle(
        _drawer_composite_graph(scene),
        scene,
        tmp_path / "oversize",
        robot_profile="dual_franka",
    )
    assert paths.deployment.is_file()
    items = load_config(paths.program)["program"]["items"]
    assert not any(
        p["preset"].startswith("gen_sim.drawer.")
        for item in items
        for p in item.get("post", [])
    )


def test_drawer_place_preparation_uses_live_target_without_fit_gate(
    drawer_container_scene: PreparedScene,
) -> None:
    from dataclasses import dataclass
    from embodichain.gen_sim.task_engine._task_program.drawer_runtime import (
        prepare_place,
    )
    from embodichain.lab.sim.atomic_actions import (
        PlaceGoal,
        PlaceOptions,
        SceneEntityPose,
    )

    obs, _, link, _ = _live_drawer(drawer_container_scene)
    obs.vertices *= 10
    qpos = torch.zeros(1, 1)
    motion = SimpleNamespace(
        task_state_key="arm",
    )
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id=obs.route.object_id),
        object_to_eef=torch.eye(4).unsqueeze(0),
    )
    context = SimpleNamespace(
        task=SimpleNamespace(get_held_object=lambda key: held),
        robot=SimpleNamespace(qpos=qpos),
        env_ids=torch.tensor([0]),
    )

    @dataclass
    class Request:
        goal: object
        skill_options: object
        binding: object

    request = Request(
        PlaceGoal(SceneEntityPose(obs.route.affordance)),
        PlaceOptions(),
        SimpleNamespace(endpoint=lambda slot, name: motion),
    )
    prepared = prepare_place(request, context, (obs,))
    assert prepared.goal.xpos.shape == (1, 3, 4, 4)
    torch.testing.assert_close(
        prepared.goal.xpos[:, 2], obs.entry_poses()[2] @ held.object_to_eef
    )
    assert prepared.goal.xpos[0, 0, 1, 3] < prepared.goal.xpos[0, 1, 1, 3]
    assert isinstance(request.goal.xpos, SceneEntityPose)
    link[0, 0, 3] += 0.12
    changed = prepare_place(request, context, (obs,))
    torch.testing.assert_close(
        changed.goal.xpos[0, 0, :3, 3] - prepared.goal.xpos[0, 0, :3, 3],
        torch.tensor([0.12, 0, 0]),
    )
