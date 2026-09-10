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
from pxr import Usd, UsdGeom, UsdPhysics

from embodichain.gen_sim.task_engine.agent import TaskAgent
from embodichain.gen_sim.task_engine.interpretation import _INTENT_FIELD_DEFAULTS
from embodichain.gen_sim.task_engine.orchestration.contracts import ROLE_BINDINGS_SCHEMA
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)
from embodichain.gen_sim.task_engine._task_program.articulation_binding import (
    PrismaticBinding,
    SLIDE_CALL,
    WITHDRAW_CALL,
    inspect_prismatic,
)
from embodichain.gen_sim.task_engine._task_program.articulation_slide import (
    ArticulationSlideFactory,
    ArticulationWithdrawFactory,
    ArticulationStabilityPort,
    preset_id,
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


@pytest.mark.parametrize("states", [("open",), ("closed",), ("open", "closed")])
def test_e6_bundle_uses_standard_registration_and_complete_recipe(
    scene: PreparedScene, tmp_path: Path, states: tuple
) -> None:
    graph, paths = generate_task_program_bundle(
        _graph(scene, states), scene, tmp_path / "bundle", robot_profile="dual_franka"
    )
    _verify_program_projection(paths.program, graph)
    assert [n["call"]["call_id"] for n in graph["nodes"]] == [
        SLIDE_CALL,
        WITHDRAW_CALL,
        "simulation.park",
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
    elif mutation == "incomplete":
        graph["nodes"][1]["call"]["call_id"] = "simulation.park"
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


def _runtime(scene: PreparedScene):
    cfg = scene.articulations[0]
    binding = inspect_prismatic(cfg)
    position = torch.tensor([[binding.target("open")]])
    cached_limits = torch.tensor([[[-0.4, 0.0]]])
    art = SimpleNamespace(
        cfg=SimpleNamespace(**cfg),
        joint_names=["slide"],
        get_qpos=lambda: position.clone(),
        get_qpos_limits=lambda: cached_limits.clone(),
        set_qpos_limits=lambda values: cached_limits.copy_(values),
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
    withdraw.lower(
        call, context=context, bound=bound, option_template=MoveEndEffectorOptions()
    )
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
