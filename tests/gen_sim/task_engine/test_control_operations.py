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
from types import SimpleNamespace
import math

import pytest

from .test_press import button
from .test_press_integration import scene as press_scene
from .test_twist import knob_scene
from .test_task_spec_generation import candidate, selector, step
from embodichain.gen_sim.task_engine._task_program.assembly import load_deployment
from embodichain.gen_sim.task_engine._task_program.twist_binding import (
    enrich_twist_inventory,
)
from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
    evaluate_twist,
    _initial_twist_state_stable,
)
from embodichain.gen_sim.task_engine._task_program.press import PressSample
from embodichain.gen_sim.task_engine.orchestration.contracts import ROLE_BINDINGS_SCHEMA
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine.semantic_graph_validation import (
    verify_graph_contract,
)
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)
from embodichain.utils.utility import load_config


@pytest.fixture
def mixed_scene(knob_scene, press_scene, tmp_path):
    import trimesh

    cube_path = tmp_path / "cube.glb"
    trimesh.creation.box(extents=[0.04] * 3).export(cube_path)
    art = deepcopy(press_scene.articulations[0])
    art["init_pos"] = [0.35, 0.0, 0.8]
    cube = {
        "uid": "cube",
        "shape": {"shape_type": "Mesh", "fpath": str(cube_path)},
        "init_pos": [-0.25, 0.15, 0.75],
        "body_scale": [1.0] * 3,
        "attrs": {"mass_props": {"mass": 0.04}},
    }
    result = replace(
        knob_scene,
        articulations=(*knob_scene.articulations, art),
        rigid_objects=(cube,),
        planner_objects=(
            *knob_scene.planner_objects,
            {**art, "runtime_uid": "button", "role": "articulation"},
            {**cube, "runtime_uid": "cube", "role": "rigid_object"},
        ),
    )
    enrich_twist_inventory(list(result.planner_objects), list(result.articulations))
    return result


def control_plan(scene, *, placements=True):
    values = (
        step("E1", "cube", target=selector("table"), relation="on"),
        step("E8", "control", target_setting=90, required_arm="right_arm"),
        step("E9", "button", target_state="activated", required_arm="left_arm"),
        step("E8", "control", target_setting=0, required_arm="right_arm"),
        step("E9", "button", target_state="activated", required_arm="left_arm"),
        step("E1", "cube", target=selector("table"), relation="on"),
    )
    selected = candidate(*(values if placements else values[1:-1]))
    refs = {}
    for item in selected["draft"]["steps"]:
        refs[f"{item['id']}.object"] = [item["object"]["reference"]]
        if item["target"]["kind"] != "none":
            refs[f"{item['id']}.target"] = [item["target"]["reference"]]
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": "demo",
        "candidate_id": selected["candidate_id"],
        "reference_bindings": refs,
        "role_bindings": {},
    }
    return SemanticTaskPlanner().plan_spec(selected, bindings, scene.planner_objects)


def test_serial_mixed_repeated_controls_compose_with_task_spec(mixed_scene, tmp_path):
    plan = control_plan(mixed_scene)
    graph, paths = generate_task_program_bundle(
        plan.graph,
        mixed_scene,
        tmp_path / "bundle",
        robot_profile="dual_franka",
        task_spec=plan.task_spec,
        task_candidate=plan.candidate,
        role_bindings=plan.role_bindings,
        step_lineage=plan.lineage,
    )
    assert verify_graph_contract(paths.root, graph) is not None
    deployment = load_deployment(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.root,
    )
    adapter = deployment.integration.adapter_factory
    routes = (*adapter.twist_routes, *adapter.press_routes)
    assert {r.operation_id for r in routes} == {
        g["id"] for g in graph["task_groups"] if g["task_type"] in {"E8", "E9"}
    }
    assert len({r.sensor_uid for r in routes}) == 4
    assert len({r.preset("ready") for r in routes}) == 4
    scene = load_config(paths.scene)["simulation"]
    assert [obj["uid"] for obj in scene["rigid_object"]] == ["cube"]
    assert all(obj["uid"] != "cube" for obj in scene["background"])
    policy = load_config(paths.execution_policy)
    assert policy["effect_assurance"] == "verified"
    for preset in deployment.integration.registration.robot_profile_binding.presets:
        assert preset.motion_policy.sample_count == policy["motion"]["sample_count"]
        assert preset.recovery_policy.max_replans == policy["recovery"]["max_replans"]
        assert (
            preset.workflow_recovery_policy.max_recovery_attempts
            == policy["workflow_recovery"]["max_recovery_attempts"]
        )


@pytest.mark.parametrize("source_scale", [0.5, 0.25, 1.0])
def test_shared_knob_and_button_keep_one_gripper_qualified_scale(
    knob_scene, tmp_path, source_scale
):
    import hashlib
    import json
    from pathlib import Path

    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    art = deepcopy(knob_scene.articulations[0])
    path = Path(art["fpath"])
    stage = Usd.Stage.Open(str(path))
    body = UsdGeom.Xform.Define(stage, "/Control/button").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(0.02)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.02, 0.03))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1, Gf.Vec3f(0)))
    geometry = trimesh.creation.box(extents=[0.03, 0.03, 0.01])
    geometry.apply_translation([0.07, -0.06, 0.01])
    mesh = UsdGeom.Mesh.Define(stage, "/Control/button/button_cap")
    mesh.CreatePointsAttr(geometry.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(geometry.faces))
    mesh.CreateFaceVertexIndicesAttr(geometry.faces.reshape(-1).tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Control/press_joint")
    joint.CreateBody0Rel().SetTargets(["/Control/base"])
    joint.CreateBody1Rel().SetTargets(["/Control/button"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-0.005)
    joint.CreateUpperLimitAttr(0)
    stage.GetRootLayer().Save()
    sidecar = path.with_suffix(".twist.json")
    declaration = json.loads(sidecar.read_text())
    declaration["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(declaration))
    original = path.read_bytes()
    art["body_scale"] = [source_scale] * 3
    scene = replace(
        knob_scene,
        articulations=(art,),
        planner_objects=(
            {**deepcopy(art), "runtime_uid": "control", "role": "articulation"},
            *[
                item
                for item in knob_scene.planner_objects
                if item["runtime_uid"] != "control"
            ],
        ),
    )
    enrich_twist_inventory(list(scene.planner_objects), list(scene.articulations))
    selected = candidate(
        step("E8", "control", target_setting=90, required_arm="right_arm"),
        step("E9", "control", target_state="activated", required_arm="left_arm"),
    )
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": "demo",
        "candidate_id": selected["candidate_id"],
        "reference_bindings": {
            f"{item['id']}.object": ["control"] for item in selected["draft"]["steps"]
        },
        "role_bindings": {},
    }
    plan = SemanticTaskPlanner().plan_spec(selected, bindings, scene.planner_objects)
    _, paths = generate_task_program_bundle(
        plan.graph,
        scene,
        tmp_path / "shared_control",
        robot_profile="dual_franka",
        task_spec=plan.task_spec,
        task_candidate=plan.candidate,
        role_bindings=plan.role_bindings,
        step_lineage=plan.lineage,
    )
    deployment = load_deployment(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.root,
    )
    adapter = deployment.integration.adapter_factory
    expected_scale = max(0.5, source_scale)
    assert adapter.twist_routes[0].binding.scale == pytest.approx(expected_scale)
    assert adapter.press_routes[0].binding.scale == pytest.approx(expected_scale)
    runtime = load_config(paths.scene)["simulation"]["articulation"]
    assert len(runtime) == 1
    assert runtime[0]["body_scale"] == pytest.approx([expected_scale] * 3)
    assert "gen_sim_press_density" not in runtime[0].get("link_attrs", {})
    calibration = load_config(paths.root / "press_calibration.json")["routes"][0]
    assert calibration["mass_policy"] == "qualified_fixed_mass_source"
    assert calibration["runtime_mass"] == pytest.approx(0.02)
    assert calibration["calibration_scale"] == source_scale
    assert art["body_scale"] == [source_scale] * 3
    assert path.read_bytes() == original

    from embodichain.gen_sim.task_engine._task_program.press_binding import (
        calibrate_scene,
        discover_press,
    )
    from pxr import Sdf

    press = replace(
        adapter.press_routes[0],
        binding=replace(
            adapter.press_routes[0].binding, source_path=runtime[0]["fpath"]
        ),
    )
    source_press = discover_press(art)
    lineage = adapter.twist_routes[0].binding.mass_lineage_sha256
    wrong_scale = replace(
        press, binding=replace(press.binding, scale=expected_scale + 0.1)
    )
    payload = {"simulation": {"articulation": [{"uid": "control"}]}}
    with pytest.raises(ValueError, match="source changed"):
        calibrate_scene(
            payload,
            (wrong_scale,),
            source_routes=(source_press,),
            fixed_mass_lineages={"control": lineage},
        )
    with pytest.raises(ValueError, match="source changed"):
        calibrate_scene(
            payload,
            (press,),
            source_routes=(
                replace(
                    source_press,
                    binding=replace(source_press.binding, source_sha256="0" * 64),
                ),
            ),
        )

    # Re-sealing hashes cannot authorize a changed button joint in a mass copy.
    runtime_path = Path(runtime[0]["fpath"])
    changed = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(str(runtime_path)))
    slider = UsdPhysics.PrismaticJoint(changed.GetPrimAtPath("/Control/press_joint"))
    slider.GetLowerLimitAttr().Set(-0.004)
    changed.GetRootLayer().Export(str(runtime_path))
    runtime_sidecar = runtime_path.with_suffix(".twist.json")
    roles = json.loads(runtime_sidecar.read_text())
    roles["source_sha256"] = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    runtime_sidecar.write_text(json.dumps(roles))
    manifest_path = runtime_path.with_suffix(".mass.json")
    mass_manifest = json.loads(manifest_path.read_text())
    mass_manifest["runtime_sha256"] = roles["source_sha256"]
    mass_manifest["runtime_sidecar_sha256"] = hashlib.sha256(
        runtime_sidecar.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(mass_manifest))
    forged = discover_press(
        {**art, "fpath": str(runtime_path), "body_scale": runtime[0]["body_scale"]}
    )
    with pytest.raises(ValueError, match="allowed MassAPI"):
        calibrate_scene(
            payload,
            (forged,),
            source_routes=(source_press,),
            fixed_mass_lineages={
                "control": hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            },
        )
    assert path.read_bytes() == original


def test_repeated_twist_uses_nonzero_initial_state_and_truthful_coarse_metrics():
    route = SimpleNamespace(
        binding=SimpleNamespace(target_qpos=1.0, limits=(-2.0, 2.0)),
        success_criterion="contact_backed_coarse_turn",
    )
    samples = [
        PressSample(37.0, 0.4, "prepare", False),
        PressSample(37.1, 0.4, "twist", True),
        PressSample(37.2, 0.1, "twist", True),
        PressSample(37.3, 0.4, "cleanup", False),
    ]
    result = evaluate_twist(route, samples, final_qpos=0.4)
    assert result["accepted"] is True
    assert result["target_qpos_reached"] is False
    assert _initial_twist_state_stable([0.4, 0.4], [0.0, 0.0])
    assert not _initial_twist_state_stable([0.4, 0.7], [0.0, 0.0])
    route.success_criterion = "calibrated_setting"
    assert evaluate_twist(route, samples, final_qpos=0.4)["accepted"] is False


@pytest.fixture
def control_bundle(mixed_scene, tmp_path):
    from embodichain.utils.utility import save_config

    plan = control_plan(mixed_scene, placements=False)
    root = tmp_path / "controls"
    root.mkdir()
    save_config(
        root / "static_scene_manifest.json", {"source": {"review_fixture": True}}
    )
    graph, paths = generate_task_program_bundle(
        plan.graph,
        mixed_scene,
        tmp_path / "controls",
        robot_profile="dual_franka",
        task_spec=plan.task_spec,
        task_candidate=plan.candidate,
        role_bindings=plan.role_bindings,
        step_lineage=plan.lineage,
    )
    dep = load_deployment(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.root,
    )
    return plan, graph, paths, dep


def control_receipts(graph, paths, dep):
    from embodichain.gen_sim.task_engine._task_program.control_evidence import (
        stamp_control_result,
    )

    routes = {
        r.operation_id: r
        for r in (
            *dep.integration.adapter_factory.twist_routes,
            *dep.integration.adapter_factory.press_routes,
        )
    }
    groups = {g["id"]: g for g in graph["task_groups"]}
    runtime = {"segments": []}
    for index, node in enumerate(graph["nodes"]):
        group = groups[node["task_instance_id"]]
        route = routes[group["id"]]
        policies = []
        if node["id"] == group["node_ids"][-1]:
            samples = [
                PressSample(float(index), 0.4, "prepare", False),
                PressSample(index + 0.1, 0.7, "twist", True),
                PressSample(index + 0.2, 0.4, "cleanup", False),
            ]
            result = {
                "accepted": True,
                "terminal": True,
                "contact_released": True,
                "stable": True,
                "others_unchanged": True,
                "contact_path": 0.3,
                "target_qpos_reached": False,
                "target_error": 1.0,
                "clearance": 0.1,
                "event_accepted": True,
                "retreat_verified": True,
                "housing_contact_released": True,
                "terminal_clearance": 0.1,
            }
            preset = route.preset("turned" if group["task_type"] == "E8" else "pressed")
            policies = [
                {
                    "policy_index": 0,
                    "source_path": ["program", "items", index, "post", 0],
                    "result_mask": [True],
                    "result": stamp_control_result(
                        result, route, SimpleNamespace(samples=samples), preset
                    ),
                }
            ]
        runtime["segments"].append(
            {
                "name": node["id"],
                "successes": [True],
                "active": [True],
                "end_steps": [index + 1],
                "metadata": {"runtime": {"env_ids": [0]}, "post_policies": policies},
            }
        )
    return runtime


def test_control_task_spec_goals_and_all_repeated_milestones(control_bundle):
    from embodichain.gen_sim.task_engine._task_program.control_evidence import (
        capture_control_operations,
    )
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        capture_final_goals,
        evaluate_runtime_evidence,
    )

    plan, graph, paths, dep = control_bundle
    runtime = control_receipts(graph, paths, dep)
    contract = verify_graph_contract(paths.root, graph)
    program = load_config(paths.program)
    controls = capture_control_operations(dep, graph, runtime, program, 1)
    final = capture_final_goals(
        SimpleNamespace(sim=SimpleNamespace()),
        dep,
        paths.root,
        graph,
        contract,
        1,
        runtime=runtime,
    )
    evaluation = evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        graph,
        runtime,
        final,
        num_envs=1,
        run_id="synthetic-control-receipts",
        program=program,
        constraints=load_config(paths.root / "task_program/constraints.json"),
        control_milestones=controls,
    )
    row = evaluation["environments"][0]
    assert row["certificate_ready"] is True
    assert all(value["status"] == "satisfied" for value in row["milestones"])
    assert len(controls) == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "operation",
        "digest",
        "source",
        "window",
        "release",
        "clearance",
        "duplicate",
        "policy_path",
        "inactive",
        "wrong_env",
        "missing_segment",
        "numeric_mask",
        "bool_env",
    ],
)
def test_control_receipts_fail_closed_on_foreign_or_incomplete_evidence(
    control_bundle, mutation
):
    from embodichain.gen_sim.task_engine._task_program.control_evidence import (
        capture_control_operations,
    )

    plan, graph, paths, dep = control_bundle
    runtime = control_receipts(graph, paths, dep)
    group = graph["task_groups"][0]
    terminal = next(
        s for s in runtime["segments"] if s["name"] == group["node_ids"][-1]
    )
    receipt = terminal["metadata"]["post_policies"][0]
    result = receipt["result"]
    if mutation == "operation":
        result["control"]["operation_id"] = "another_operation"
    elif mutation == "digest":
        result["control"]["route_digest"] = "0" * 64
    elif mutation == "source":
        result["control"]["source_sha256"] = "0" * 64
    elif mutation == "window":
        result["control"]["window"] = [0.0, float("nan")]
    elif mutation == "release":
        result["contact_released"] = False
    elif mutation == "clearance":
        result["clearance"] = 0.0
    elif mutation == "duplicate":
        terminal["metadata"]["post_policies"].append(deepcopy(receipt))
    elif mutation == "policy_path":
        receipt["source_path"][-1] = 1
    elif mutation == "inactive":
        terminal["active"] = [False]
    elif mutation == "wrong_env":
        terminal["metadata"]["runtime"]["env_ids"] = [1]
    elif mutation == "missing_segment":
        runtime["segments"] = runtime["segments"][1:]
    elif mutation == "numeric_mask":
        terminal["successes"] = [1]
    elif mutation == "bool_env":
        terminal["metadata"]["runtime"]["env_ids"] = [False]
    controls = capture_control_operations(
        dep, graph, runtime, load_config(paths.program), 1
    )
    assert "unavailable" in controls[group["id"]]
    assert all("accepted" in v for k, v in controls.items() if k != group["id"])


def test_twist_candidate_dispatch_is_occurrence_owned_and_restored(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import twist_runtime as module

    action = module.GenSimTwist()
    first, second = module._TwistCandidateState(), module._TwistCandidateState()
    first.goal_complete = True
    action._operation_states = {"first": first, "second": second}
    action._operation_guards = {"first": object(), "second": object()}

    def scoped(self, request, context):
        return module._candidate_state(), self._geometry_guard

    monkeypatch.setattr(module.GenSimTwist, "_plan_scoped", scoped)
    previous = module._CANDIDATE_STATE.get()
    request = SimpleNamespace(
        goal=SimpleNamespace(
            semantics=SimpleNamespace(geometry={"gen_sim_twist_operation_id": "second"})
        )
    )
    state, guard = action._plan(request, None)
    assert state is second and not state.goal_complete
    assert guard is action._operation_guards["second"]
    assert module._CANDIDATE_STATE.get() is previous
    assert action._geometry_guard is None
    monkeypatch.setattr(
        module.GenSimTwist,
        "_plan_scoped",
        lambda *args: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(RuntimeError, match="injected"):
        action._plan(request, None)
    assert module._CANDIDATE_STATE.get() is previous
    assert action._geometry_guard is None


def test_delayed_control_epoch_preserves_clock_and_discards_prior_startup():
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
        TwistContactSensor,
    )

    sensor = object.__new__(TwistContactSensor)
    sensor.clock, sensor._startup_epoch = 37.0, 1
    sensor.startup_trace = [{"prior_operation": True}]
    sensor._capture_startup = lambda: sensor.startup_trace.append(
        {"timestamp": sensor.clock}
    )
    sensor._begin_startup()
    assert sensor.clock == 37.0 and sensor._startup_started_at == 37.0
    assert sensor._startup_epoch == 2 and sensor.startup_trace == [{"timestamp": 37.0}]


def test_default_engine_start_keeps_public_plan_transform(monkeypatch):
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )
    from unittest.mock import Mock

    engine = object.__new__(GenSimActionEngine)
    engine._twist_session_controller, engine._twist_session_controllers = None, {}
    original = Mock(return_value=object())
    monkeypatch.setattr(AtomicActionEngine, "start", original)
    transform = lambda *args: None
    engine.start((), plan_transform=transform)
    original.assert_called_once_with(
        (), None, eligible_mask=None, plan_transform=transform
    )


def test_control_height_retry_rebuilds_task_spec_and_selected_identity(
    control_bundle, mixed_scene, tmp_path
):
    from .test_twist_attempts import Preparation
    from embodichain.gen_sim.task_engine._task_program.twist_attempts import (
        execute_support_candidates,
        is_twist_bundle,
    )
    from embodichain.gen_sim.task_engine.config import TaskEnginePlanningCfg
    from embodichain.gen_sim.task_engine.workflow import _headless_renderer_for_bundle
    from embodichain.utils.utility import save_config

    plan, graph, paths, dep = control_bundle
    prep = Preparation(
        paths.root,
        graph,
        SimpleNamespace(
            prepared_scene=mixed_scene, scene_manifest={"robot_profile": "dual_franka"}
        ),
        paths,
    )
    calls = []

    def executor(bundle, output, **kwargs):
        assert (
            verify_graph_contract(
                bundle, load_config(bundle / "semantic_task_graph.json")
            )
            is not None
        )
        assert (
            is_twist_bundle(bundle)
            and _headless_renderer_for_bundle(bundle) == "hybrid"
        )
        calls.append(bundle)
        output.mkdir(parents=True)
        if len(calls) == 1:
            save_config(
                output / "twist_evidence.json",
                {
                    "acceptance": {
                        "phase": "initial_stable",
                        "accepted": False,
                        "startup_check": {"reason": "startup_table_contact"},
                    }
                },
            )
            return {"status": "failed"}
        return {"status": "succeeded"}

    report, selected, execution, attempts = execute_support_candidates(
        prep, tmp_path / "execution", executor, {"seed": 0}, TaskEnginePlanningCfg()
    )
    assert report["status"] == "succeeded" and len(calls) == 2
    assert selected.output_dir != paths.root
    assert load_config(selected.output_dir / "task_spec.json") == load_config(
        paths.root / "task_spec.json"
    )
    assert (
        verify_graph_contract(selected.output_dir, selected.semantic_task_graph)
        is not None
    )
    assert (
        verify_graph_contract(selected.output_dir, selected.semantic_task_graph)[
            "source_manifest_digest"
        ]
        is not None
    )


@pytest.mark.parametrize(
    "skill,marker",
    [("twist", "gen_sim_twist_operation_id"), ("press", "gen_sim_press_operation_id")],
)
def test_control_resolver_preserves_caller_timeout_and_thresholds(
    monkeypatch, skill, marker
):
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        MotionPolicy,
        RecoveryPolicy,
    )
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Invocation:
        skill_id: str
        goal: object
        motion_policy: object
        recovery_policy: object
        invocation_id: str | None = None

    engine = object.__new__(GenSimActionEngine)
    engine._motion_samples = {}
    engine._pour_return_calls = frozenset()
    monkeypatch.setattr(
        AtomicActionEngine, "_resolve", lambda self, invocation: invocation
    )
    policy = RecoveryPolicy(
        action_timeout=3.0, goal_translation_threshold=0.007, max_replans=3
    )
    request = Invocation(
        skill,
        SimpleNamespace(semantics=SimpleNamespace(geometry={marker: "owned"})),
        MotionPolicy(sample_count=140),
        policy,
    )
    resolved = engine._resolve(request)
    assert resolved.recovery_policy == replace(
        policy, max_replans=0, max_action_retries=0
    )
    assert request.recovery_policy.action_timeout == 3.0


def test_multiple_assembly_dimension_failure_returns_original_report(
    control_bundle, mixed_scene, tmp_path, monkeypatch
):
    from .test_twist_attempts import Preparation
    from embodichain.gen_sim.task_engine._task_program import twist_attempts as module
    from embodichain.gen_sim.task_engine.config import TaskEnginePlanningCfg
    from embodichain.utils.utility import save_config

    _, graph, paths, _ = control_bundle
    adaptation = load_config(paths.root / "twist_adaptation.json")
    adaptation["scale_candidate"] = None
    adaptation["records"] *= 2
    save_config(paths.root / "twist_adaptation.json", adaptation)
    prep = Preparation(
        paths.root,
        graph,
        SimpleNamespace(
            prepared_scene=mixed_scene, scene_manifest={"robot_profile": "dual_franka"}
        ),
        paths,
    )
    report = {"status": "failed"}
    monkeypatch.setattr(module, "size_failure", lambda *args: True)
    result, selected, _, _ = module.execute_support_candidates(
        prep,
        tmp_path / "failed",
        lambda *args, **kwargs: report,
        {"seed": 0},
        TaskEnginePlanningCfg(),
    )
    assert result is report and selected is prep


@pytest.mark.parametrize("sensor_kind", ["press", "twist"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_control_contact_sensor_preserves_simulation_device(
    monkeypatch, sensor_kind, device
):
    import torch
    from embodichain.gen_sim.task_engine._task_program.press_runtime import (
        PressContactSensor,
    )
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
        TwistContactSensor,
    )
    from embodichain.lab.sim.sensors import ContactSensor

    captured = {}
    owner = object()
    config = object()

    def initialize(self, cfg, output_device, *, owner):
        captured.update(cfg=cfg, device=output_device, owner=owner)

    monkeypatch.setattr(ContactSensor, "__init__", initialize)
    sensor = PressContactSensor if sensor_kind == "press" else TwistContactSensor
    sensor(config, torch.device(device), owner=owner)
    assert captured == {"cfg": config, "device": torch.device(device), "owner": owner}
