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

from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import pick_continuation as gate
from embodichain.gen_sim.task_engine._task_program._search_cost import search_cost_scope
from embodichain.lab.sim.atomic_actions import (
    AffordancePoseCandidates,
    JointPositionTarget,
    PlannerDiagnostics,
)
from embodichain.lab.sim.atomic_actions.state import EntityState
from embodichain.lab.task_program.semantics import (
    Pick,
    Place,
    RegisteredSemanticCall,
    SceneObjectRef,
    SceneAffordanceRef,
    EffectAssurance,
    SemanticPose,
)


@pytest.mark.parametrize(
    "damage", [None, "other", "middle", "parallel", "projected", "monitor"]
)
def test_only_owned_adjacent_verified_calls_can_preview(damage) -> None:
    calls = [
        Pick(SceneObjectRef("item"), resources={"primary": "arm"}),
        Place(
            SceneObjectRef("other" if damage == "other" else "item"),
            inside=SceneAffordanceRef("bin"),
            resources={"primary": "arm"},
        ),
    ]
    if damage == "middle":
        calls.insert(
            1,
            RegisteredSemanticCall(
                call_id="simulation.park", resources={"primary": "arm"}
            ),
        )
    segments = [
        SimpleNamespace(
            segment_id=str(i),
            parallel_block=object() if damage == "parallel" else None,
            calls=(SimpleNamespace(call=c, segment_call_index=0),),
        )
        for i, c in enumerate(calls)
    ]
    program = SimpleNamespace(program_id="p", iter_segments=lambda: iter(segments))
    preset = SimpleNamespace(
        effect_assurance=(
            EffectAssurance.PROJECTED
            if damage == "projected"
            else EffectAssurance.VERIFIED
        ),
        effect_monitors={} if damage == "monitor" else {"pick": None, "place": None},
    )
    assert bool(gate.bind_calls(program, preset)) is (damage is None)


@pytest.mark.parametrize("destination", ["at", "on", "inside"])
def test_adjacent_place_keeps_its_exact_destination_and_resource(destination) -> None:
    target = (
        SemanticPose((0.3, 0.2, 0.1), (0.0, 0.0, 0.0, 1.0))
        if destination == "at"
        else SceneAffordanceRef("measured_target")
    )
    pick = Pick(SceneObjectRef("item"), resources={"primary": "arm"})
    place = Place(
        SceneObjectRef("item"),
        **{destination: target},
        resources={"primary": "arm"},
    )
    segments = tuple(
        SimpleNamespace(
            segment_id=str(index),
            parallel_block=None,
            calls=(SimpleNamespace(call=call, segment_call_index=0),),
        )
        for index, call in enumerate((pick, place))
    )
    program = SimpleNamespace(program_id="p", iter_segments=lambda: iter(segments))
    preset = SimpleNamespace(
        effect_assurance=EffectAssurance.VERIFIED,
        effect_monitors={"pick": None, "place": None},
    )
    (route,) = gate.bind_calls(program, preset)
    assert route.place_call is place
    assert getattr(route.place_call, destination) is getattr(place, destination)
    assert dict(route.place_call.resources) == {"primary": "arm"}


@pytest.mark.parametrize(
    "damage", [None, "unknown", "object", "resource", "part", "arguments", "monitor"]
)
def test_registered_relative_place_preserves_its_exact_call_and_identity(
    damage,
) -> None:
    arguments = {
        "object": "other" if damage == "object" else "item",
        "reference": "table",
        "relation": "on",
        "stage_id": "place_step",
    }
    if damage == "part":
        arguments["reference_part"] = "drawer"
    elif damage == "arguments":
        arguments["unowned_target"] = "other"
    pick = Pick(SceneObjectRef("item"), resources={"primary": "arm"})
    place = RegisteredSemanticCall(
        call_id="unknown.place" if damage == "unknown" else "simulation.place_relative",
        arguments=arguments,
        resources={"primary": "other" if damage == "resource" else "arm"},
    )
    segments = tuple(
        SimpleNamespace(
            segment_id=str(index),
            parallel_block=None,
            calls=(SimpleNamespace(call=call, segment_call_index=0),),
        )
        for index, call in enumerate((pick, place))
    )
    program = SimpleNamespace(program_id="p", iter_segments=lambda: iter(segments))
    preset = SimpleNamespace(
        effect_assurance=EffectAssurance.VERIFIED,
        effect_monitors=(
            {"pick": None}
            if damage == "monitor"
            else {"pick": None, "simulation.place_relative": None}
        ),
    )
    routes = gate.bind_calls(program, preset)
    assert bool(routes) is (damage is None)
    if routes:
        assert routes[0].place_call is place
        assert dict(routes[0].place_call.arguments) == arguments


def test_bind_previews_analyzes_original_registered_target_without_rewrite() -> None:
    pick = Pick(SceneObjectRef("item"), resources={"primary": "arm"})
    place = RegisteredSemanticCall(
        call_id="simulation.place_relative",
        arguments={"object": "item", "reference": "table", "relation": "on"},
        resources={"primary": "arm"},
    )
    route = gate.Continuation("p/pick:0", "p/place:0", pick, place)
    compiler = SimpleNamespace(analyze=Mock(side_effect=lambda calls, **_kw: calls))
    preview = gate.bind_previews(compiler, (route,))[route.pick_id]
    assert preview.pick_workflow == (pick,)
    assert preview.place_workflow == (place,)
    assert preview.place_workflow[0] is place


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "unknown",
        "object",
        "resource",
        "role",
        "target",
        "arguments",
        "parallel",
        "monitor",
        "extra",
    ],
)
def test_only_one_explicit_same_object_transport_can_join_continuation(damage) -> None:
    pick = Pick(SceneObjectRef("item"), resources={"primary": "arm"})
    transport = RegisteredSemanticCall(
        call_id=(
            "other.transport" if damage == "unknown" else "simulation.move_held_object"
        ),
        arguments=(
            {
                "object": "other" if damage == "object" else "item",
                "target": "declared_pose",
            }
            if damage != "target"
            else {"object": "item"}
        ),
        resources={
            "destination" if damage == "role" else "primary": (
                "other" if damage == "resource" else "arm"
            )
        },
    )
    if damage == "arguments":
        transport = replace(
            transport, arguments={**transport.arguments, "unowned_pose": 1}
        )
    place = Place(
        SceneObjectRef("item"), on=SceneObjectRef("table"), resources={"primary": "arm"}
    )
    calls = [pick, transport, place]
    if damage == "extra":
        calls.insert(2, transport)
    segments = tuple(
        SimpleNamespace(
            segment_id=str(index),
            parallel_block=object() if damage == "parallel" and index == 1 else None,
            calls=(SimpleNamespace(call=call, segment_call_index=0),),
        )
        for index, call in enumerate(calls)
    )
    program = SimpleNamespace(program_id="p", iter_segments=lambda: iter(segments))
    monitors = {"pick": None, "place": None}
    if damage != "monitor":
        monitors["simulation.move_held_object"] = None
    preset = SimpleNamespace(
        effect_assurance=EffectAssurance.VERIFIED, effect_monitors=monitors
    )
    routes = gate.bind_calls(program, preset)
    assert bool(routes) is (damage is None)
    if routes:
        assert routes[0].transport_id == "p/1:0"
        assert routes[0].transport_call is transport
        compiler = SimpleNamespace(analyze=Mock(side_effect=lambda calls, **_kw: calls))
        preview = gate.bind_previews(compiler, routes)[routes[0].pick_id]
        assert preview.transport_workflow == (transport,)
        assert preview.place_workflow == (place,)


def test_unrelated_calls_outside_continuation_do_not_change_binding() -> None:
    pick = Pick(SceneObjectRef("item"), resources={"primary": "arm"})
    place = Place(
        SceneObjectRef("item"), on=SceneObjectRef("table"), resources={"primary": "arm"}
    )
    segments = [
        SimpleNamespace(
            segment_id=name,
            parallel_block=None,
            calls=(SimpleNamespace(call=call, segment_call_index=0),),
        )
        for name, call in (("pick", pick), ("place", place))
    ]
    program = SimpleNamespace(program_id="p", iter_segments=lambda: iter(segments))
    preset = SimpleNamespace(
        effect_assurance=EffectAssurance.VERIFIED,
        effect_monitors={"pick": None, "place": None},
    )
    baseline = gate.bind_calls(program, preset)
    unrelated = SimpleNamespace(
        segment_id="unrelated",
        parallel_block=None,
        calls=(
            SimpleNamespace(
                call=RegisteredSemanticCall(call_id="other.capability"),
                segment_call_index=0,
            ),
        ),
    )
    segments.insert(0, unrelated)
    segments.append(unrelated)
    assert gate.bind_calls(program, preset) == baseline


@dataclass(frozen=True)
class Goal:
    grasp_xpos: object = None


@dataclass(frozen=True)
class Options:
    approach_direction: object


@dataclass(frozen=True)
class Request:
    goal: Goal
    skill_options: object
    motion_policy: object


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    expected_effects: object
    diagnostics: PlannerDiagnostics
    commands: tuple = ()


def _fixture(monkeypatch):
    poses = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
    poses[0, 1, 2, 3] = 0.3
    poses[0, 2, 2, 3] = 0.1
    plan = Plan(
        torch.tensor([True]),
        SimpleNamespace(
            held_object_updates={"arm": SimpleNamespace(grasp_xpos=poses[:, 0])}
        ),
        PlannerDiagnostics(backend="test"),
    )
    robot = SimpleNamespace(device=torch.device("cpu"))
    action = SimpleNamespace(
        robot=robot,
        failed_plan=Mock(
            return_value=replace(plan, plan_success=torch.tensor([False]))
        ),
    )
    request = Request(
        Goal(),
        Options(approach_direction=torch.tensor([0.0, 0.0, -1.0])),
        object(),
    )
    context = SimpleNamespace(robot=SimpleNamespace(qpos=torch.zeros(1, 1)))
    selection = SimpleNamespace(
        direction=torch.tensor([[0.0, 0.0, -1.0]]),
        candidates=AffordancePoseCandidates(
            poses, torch.zeros(1, 3), torch.tensor([[True, False, True]])
        ),
    )
    built = []

    def build(selected, current):
        assert current is context
        built.append(selected)
        torch.rand(1)
        return replace(
            plan,
            expected_effects=SimpleNamespace(
                held_object_updates={
                    "arm": SimpleNamespace(grasp_xpos=selected.goal.grasp_xpos)
                }
            ),
            commands=(selected.goal.grasp_xpos,),
        )

    preview = SimpleNamespace(
        route=SimpleNamespace(place_id="p/place:0"),
        compiler=SimpleNamespace(
            integration=SimpleNamespace(engine=SimpleNamespace(plan_request=build))
        ),
    )
    monkeypatch.setattr(
        gate,
        "_validate_final_plan_velocity",
        lambda *_a: {"valid_mask": torch.tensor([True])},
    )
    monkeypatch.setattr(gate, "_inspect_pick_plan", lambda *_a: {"valid": True})
    return action, request, context, plan, selection, preview, built


def test_qualified_original_retains_plan_rng_and_never_rebuilds(monkeypatch) -> None:
    data = _fixture(monkeypatch)
    before = torch.random.get_rng_state()

    def preview(*_a, **_kw):
        torch.rand(1)
        return {"valid": True}

    monkeypatch.setattr(gate, "preview_place", preview)
    assert gate.recover(*data[:6]) is data[3]
    assert torch.equal(before, torch.random.get_rng_state())
    assert not data[6]


@pytest.mark.parametrize(
    "reason", [{"valid": False}, {"valid": False, "unreachable_waypoints": False}]
)
def test_uncertain_place_prediction_fails_closed_without_reselecting_grasp(
    monkeypatch, reason
) -> None:
    data = _fixture(monkeypatch)
    monkeypatch.setattr(gate, "preview_place", lambda *_a, **_kw: reason)
    result = gate.recover(*data[:6])
    assert not result.plan_success.any()
    assert result.diagnostics.failure.code == "pick_continuation_unavailable"
    assert not data[6]


@pytest.mark.parametrize(
    "reason",
    [
        "place_planning_failed",
        "place_velocity_rejected",
        "robot_proxy_rejected",
        "transport_planning_failed",
        "transport_velocity_rejected",
    ],
)
def test_non_kinematic_rejection_reselects_a_legal_grasp(monkeypatch, reason) -> None:
    data = _fixture(monkeypatch)
    monkeypatch.setattr(
        gate,
        "preview_place",
        Mock(
            side_effect=[
                {"valid": False, "status": "unqualified", "reason": reason},
                {"valid": True},
            ]
        ),
    )
    result = gate.recover(*data[:6])
    chosen = data[6][0].goal.grasp_xpos
    assert result.plan_success.all()
    assert torch.equal(
        result.expected_effects.held_object_updates["arm"].grasp_xpos, chosen
    )
    assert torch.equal(result.commands[0], chosen)
    assert data[6][0].motion_policy is data[1].motion_policy


def test_unavailable_geometry_cannot_be_repaired_by_grasp_search(monkeypatch) -> None:
    data = _fixture(monkeypatch)
    monkeypatch.setattr(
        gate,
        "preview_place",
        lambda *_a, **_kw: {
            "valid": False,
            "status": "unavailable",
            "reason": "held_object_scene_geometry_unavailable",
        },
    )
    result = gate.recover(*data[:6])
    assert result.diagnostics.failure.code == "pick_continuation_unavailable"
    assert not data[6]


def test_failed_continuation_rebuilds_only_legal_grasp_not_task_goal_or_limits(
    monkeypatch,
) -> None:
    data = _fixture(monkeypatch)
    before = torch.random.get_rng_state()
    monkeypatch.setattr(
        gate,
        "preview_place",
        Mock(
            side_effect=[
                {"valid": False, "unreachable_waypoints": True},
                {"valid": True},
            ]
        ),
    )
    result = gate.recover(*data[:6])
    assert len(data[6]) == 1
    assert torch.equal(data[6][0].goal.grasp_xpos, data[4].candidates.poses[:, 2])
    assert data[6][0].motion_policy is data[1].motion_policy
    assert result.diagnostics.metadata["gen_sim_pick_continuation"]["candidate_id"] == 2
    assert torch.equal(before, torch.random.get_rng_state())


def test_exhausted_or_unsafe_pool_fails_closed_without_acquired_state(
    monkeypatch,
) -> None:
    data = _fixture(monkeypatch)
    monkeypatch.setattr(
        gate,
        "preview_place",
        lambda *_a, **_kw: {"valid": False, "unreachable_waypoints": True},
    )
    monkeypatch.setattr(gate, "_inspect_pick_plan", lambda *_a: {"valid": False})
    result = gate.recover(*data[:6])
    assert not result.plan_success.any()
    assert result.diagnostics.failure.code == "pick_continuation_unqualified"


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "batch",
        "env",
        "robot",
        "identity",
        "cartesian",
        "explicit",
        "fixed",
        "upright",
        "axis",
        "part",
        "constrained",
        "pool",
        "direction",
        "strategy",
        "dynamic",
        "pour",
        "preview",
    ],
)
def test_pick_recovery_does_not_capture_other_action_contracts(damage) -> None:
    robot = object()
    action = SimpleNamespace(robot=robot)
    engine = SimpleNamespace(robot=robot, _cartesian_calls=())
    preview = SimpleNamespace(
        route=SimpleNamespace(pick_id="p/pick:0"),
        compiler=SimpleNamespace(integration=SimpleNamespace(engine=engine)),
    )
    request = SimpleNamespace(
        invocation_id="p/pick:0",
        goal=SimpleNamespace(grasp_xpos=None),
        skill_options=SimpleNamespace(
            fixed_object_to_eef=None,
            rotate_upright=None,
            obj_upright_direction=None,
            pick_object_part="center",
        ),
        motion_policy=SimpleNamespace(
            strategy="ik_interp", dynamic_collision_mode="off"
        ),
    )
    context = SimpleNamespace(batch_size=1, env_ids=torch.tensor([0]))
    selection = SimpleNamespace(
        constrained=False, candidates=object(), direction=object()
    )
    if damage == "batch":
        context.batch_size = 2
    elif damage == "env":
        context.env_ids = torch.tensor([1])
    elif damage == "robot":
        action.robot = object()
    elif damage == "identity":
        request.invocation_id = "other"
    elif damage == "pour":
        action.pick_purposes = {request.invocation_id: "pour"}
    elif damage == "cartesian":
        engine._cartesian_calls = (request.invocation_id,)
    elif damage == "explicit":
        request.goal.grasp_xpos = object()
    elif damage in {"fixed", "upright", "axis", "part"}:
        field = {
            "fixed": "fixed_object_to_eef",
            "upright": "rotate_upright",
            "axis": "obj_upright_direction",
            "part": "pick_object_part",
        }[damage]
        setattr(request.skill_options, field, "other")
    elif damage == "constrained":
        selection.constrained = True
    elif damage in {"pool", "direction"}:
        setattr(selection, "candidates" if damage == "pool" else "direction", None)
    elif damage in {"strategy", "dynamic"}:
        setattr(
            request.motion_policy,
            "strategy" if damage == "strategy" else "dynamic_collision_mode",
            "other",
        )
    token = gate._PREVIEW.set(damage == "preview")
    try:
        assert gate.eligible(action, request, context, selection, preview) is (
            damage is None
        )
    finally:
        gate._PREVIEW.reset(token)


@dataclass(frozen=True)
class Scene:
    entities: dict


@dataclass(frozen=True)
class Context:
    robot: object
    task: object
    scene: Scene

    @property
    def last_qpos(self):
        return self.robot.qpos

    def project(self, *, qpos, task):
        return replace(self, robot=SimpleNamespace(qpos=qpos), task=task)


def _preview_fixture(monkeypatch):
    arm, hand = JointPositionTarget("arm", (0, 1)), JointPositionTarget("hand", (2,))
    endpoints = [
        SimpleNamespace(
            task_state_key="arm",
            target=t,
            require_target=lambda _cls, t=t: t,
            joint_positions=lambda *_a, **_kw: torch.zeros(1, 1),
        )
        for t in (arm, hand)
    ]
    binding = SimpleNamespace(endpoint=lambda _slot, name: endpoints[name == "grasp"])
    pose, grasp = torch.eye(4)[None], torch.eye(4)[None]
    grasp[:, 2, 3] = 2.0
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="item"),
        object_to_eef=pose.clone(),
        grasp_xpos=grasp,
    )
    initial_task, projected_task = object(), object()
    context = Context(
        SimpleNamespace(qpos=torch.zeros(1, 3)),
        initial_task,
        Scene({"item": EntityState(pose.clone())}),
    )
    picked = SimpleNamespace(
        plan_success=torch.tensor([True]),
        joint_trajectory=SimpleNamespace(positions=torch.zeros(1, 2, 3)),
        expected_effects=SimpleNamespace(
            held_object_updates={"arm": held}, apply=lambda task, mask: projected_task
        ),
    )
    request = SimpleNamespace(
        binding=binding,
        goal=SimpleNamespace(semantics=held.semantics),
        skill_options=SimpleNamespace(lift_height=10.0),
    )
    place_request = SimpleNamespace(binding=binding)
    placed = SimpleNamespace(
        plan_success=torch.tensor([True]),
        invocation_id="p/place:0",
        skill_id="place",
        joint_trajectory=SimpleNamespace(positions=torch.zeros(1, 2, 3)),
        expected_effects=SimpleNamespace(
            held_object_updates={"arm": None},
            coordinated_held_object_updates={},
            articulation_joint_updates={},
        ),
    )
    fk = pose.clone()
    fk[:, 2, 3] = 0.42
    robot = SimpleNamespace(
        device=torch.device("cpu"), compute_fk=Mock(return_value=fk)
    )
    action = SimpleNamespace(robot=robot)
    engine = SimpleNamespace(
        resolve=Mock(return_value=place_request),
        plan=Mock(return_value=placed),
        actions={"place": object()},
    )
    compiler = SimpleNamespace(
        ground=Mock(return_value=SimpleNamespace(invocation=object())),
        integration=SimpleNamespace(engine=engine),
    )
    preview = gate.Preview(
        SimpleNamespace(pick_id="p/pick:0", place_id="p/place:0"),
        compiler,
        object(),
        object(),
    )
    monkeypatch.setattr(gate, "_require_monitor", Mock())
    monkeypatch.setattr(
        gate,
        "_validate_final_plan_velocity",
        lambda *_a: {"valid_mask": torch.tensor([True])},
    )
    return SimpleNamespace(
        action=action,
        request=request,
        context=context,
        picked=picked,
        placed=placed,
        place_request=place_request,
        preview=preview,
        compiler=compiler,
        engine=engine,
        binding=binding,
        endpoints=endpoints,
        initial_task=initial_task,
        projected_task=projected_task,
        pose=pose,
        fk=fk,
    )


def test_preview_uses_emitted_pick_fk_and_private_state_not_intended_lift(
    monkeypatch,
) -> None:
    data = _preview_fixture(monkeypatch)
    result = gate.preview_place(
        data.action, data.request, data.context, data.picked, data.preview
    )
    assert result["valid"]
    assert result["physical_evidence"] is False
    assert result["withdrawal_qualification"] == "unavailable"
    predicted = data.compiler.ground.call_args_list[1].args[2]
    assert (
        predicted.task is data.projected_task and data.context.task is data.initial_task
    )
    assert torch.equal(predicted.scene.entities["item"].pose, data.fk)
    assert torch.equal(data.context.scene.entities["item"].pose, data.pose)
    assert torch.equal(data.context.last_qpos, torch.zeros(1, 3))


@pytest.mark.parametrize("damage", ["key", "target"])
def test_preview_rejects_resolved_resource_identity_changes(
    monkeypatch, damage
) -> None:
    data = _preview_fixture(monkeypatch)
    changed = SimpleNamespace(
        task_state_key="other" if damage == "key" else "arm",
        target=(
            JointPositionTarget("other", (0, 1))
            if damage == "target"
            else data.endpoints[0].target
        ),
    )
    data.place_request.binding = SimpleNamespace(
        endpoint=lambda _slot, name: changed if name == "motion" else data.endpoints[1]
    )
    result = gate.preview_place(
        data.action, data.request, data.context, data.picked, data.preview
    )
    assert result["status"] == "unavailable"
    assert result["reason"] == "continuation_resource_identity_mismatch"
    data.engine.plan.assert_not_called()


@pytest.mark.parametrize("damage", ["release", "coordinated", "articulation", "hand"])
def test_preview_rejects_unsupported_release_effects(monkeypatch, damage) -> None:
    data = _preview_fixture(monkeypatch)
    if damage == "release":
        data.placed.expected_effects.held_object_updates = {"other": None}
    elif damage == "coordinated":
        data.placed.expected_effects.coordinated_held_object_updates = {"arm": object()}
    elif damage == "articulation":
        data.placed.expected_effects.articulation_joint_updates = {"drawer": object()}
    else:
        data.placed.joint_trajectory.positions[:, -1, 2] = 1.0
    result = gate.preview_place(
        data.action, data.request, data.context, data.picked, data.preview
    )
    assert result["status"] == "unavailable"
    assert result["reason"] == "continuation_release_contract_mismatch"


def test_preview_keeps_caller_rng_and_pick_trajectory_private(monkeypatch) -> None:
    data = _preview_fixture(monkeypatch)
    before = torch.random.get_rng_state()

    def plan(_invocation, future):
        torch.rand(1)
        future.robot.qpos.add_(1.0)
        return data.placed

    data.engine.plan.side_effect = plan
    with search_cost_scope() as cost:
        assert gate.preview_place(
            data.action, data.request, data.context, data.picked, data.preview
        )["valid"]
    assert cost.snapshot()["counters"] == {"place_preview": 1}
    assert torch.equal(before, torch.random.get_rng_state())
    assert not data.picked.joint_trajectory.positions.any()
    assert not data.context.robot.qpos.any()
    assert not gate._PREVIEW.get()


def _transport_preview_fixture(monkeypatch):
    data = _preview_fixture(monkeypatch)
    retained = data.picked.expected_effects.held_object_updates["arm"]
    moved_qpos = data.picked.joint_trajectory.positions.clone()
    moved_qpos[:, -1, :2] = 0.7
    data.transported_task = object()
    data.moved = SimpleNamespace(
        plan_success=torch.tensor([True]),
        invocation_id="p/transport:0",
        skill_id="move_held_object",
        joint_trajectory=SimpleNamespace(positions=moved_qpos),
        expected_effects=SimpleNamespace(
            held_object_updates={"arm": retained},
            coordinated_held_object_updates={},
            articulation_joint_updates={},
            apply=lambda *_a: data.transported_task,
        ),
        diagnostics=PlannerDiagnostics(
            backend="test",
            metadata={},
        ),
    )
    data.transport_request = SimpleNamespace(
        binding=data.binding,
        goal=SimpleNamespace(object_target_pose=torch.eye(4)[None]),
        motion_policy=object(),
        recovery_policy=object(),
    )
    data.preview = replace(
        data.preview,
        route=SimpleNamespace(
            pick_id="p/pick:0", place_id="p/place:0", transport_id="p/transport:0"
        ),
        transport_workflow=object(),
    )
    data.engine.resolve.side_effect = [data.transport_request, data.place_request]
    data.engine.plan.side_effect = [data.moved, data.placed]
    data.transported_fk = data.fk.clone()
    data.transported_fk[:, 0, 3] = 0.6
    data.action.robot.compute_fk.side_effect = [data.fk, data.transported_fk]
    return data


def test_transport_prediction_projects_emitted_state_before_exact_place(
    monkeypatch,
) -> None:
    data = _transport_preview_fixture(monkeypatch)
    monitor = Mock()
    monkeypatch.setattr(gate, "_require_monitor", monitor)
    before = torch.random.get_rng_state()
    target = data.transport_request.goal.object_target_pose.clone()
    policy, recovery = (
        data.transport_request.motion_policy,
        data.transport_request.recovery_policy,
    )
    with search_cost_scope() as cost:
        result = gate.preview_place(
            data.action, data.request, data.context, data.picked, data.preview
        )
    assert result["valid"] and result["physical_evidence"] is False
    assert result["transport_preview"]["invocation_id"] == "p/transport:0"
    assert result["transport_preview"]["path_qualification"] == "not_checked"
    assert cost.snapshot()["counters"] == {"place_preview": 1, "transport_preview": 1}
    predicted = data.compiler.ground.call_args_list[2].args[2]
    assert predicted.task is data.transported_task
    assert torch.equal(predicted.scene.entities["item"].pose, data.transported_fk)
    assert torch.equal(
        predicted.robot.qpos, data.moved.joint_trajectory.positions[:, -1]
    )
    assert data.context.task is data.initial_task
    assert torch.equal(data.context.scene.entities["item"].pose, data.pose)
    assert torch.equal(target, data.transport_request.goal.object_target_pose)
    assert data.transport_request.motion_policy is policy
    assert data.transport_request.recovery_policy is recovery
    assert monitor.call_args_list[1].args[1] is gate.SemanticEffectKind.ATTACH
    assert monitor.call_args_list[1].args[2] == "p/transport:0"
    assert torch.equal(before, torch.random.get_rng_state())
    assert not gate._PREVIEW.get()


@pytest.mark.parametrize(
    "damage, status",
    [
        ("planning", "unqualified"),
        ("velocity", "unqualified"),
        ("resource", "unavailable"),
        ("transform", "unavailable"),
        ("object", "unavailable"),
        ("other_role", "unavailable"),
        ("coordinated", "unavailable"),
        ("articulation", "unavailable"),
        ("hand", "unavailable"),
        ("identity", "unavailable"),
    ],
)
def test_transport_prediction_fails_closed_before_place(
    monkeypatch, damage, status
) -> None:
    data = _transport_preview_fixture(monkeypatch)
    effects = data.moved.expected_effects
    held = effects.held_object_updates["arm"]
    if damage == "planning":
        data.moved.plan_success[:] = False
    elif damage == "velocity":
        monkeypatch.setattr(
            gate,
            "_validate_final_plan_velocity",
            lambda *_a: {"valid_mask": torch.tensor([False])},
        )
    elif damage == "resource":
        data.transport_request.binding = SimpleNamespace(
            endpoint=lambda *_a: SimpleNamespace(
                task_state_key="other", target=object()
            )
        )
    elif damage in {"transform", "object"}:
        changed = SimpleNamespace(**vars(held))
        if damage == "transform":
            changed.object_to_eef = held.object_to_eef.clone()
            changed.object_to_eef[:, 0, 3] += 0.1
        else:
            changed.semantics = SimpleNamespace(entity_id="other")
        effects.held_object_updates = {"arm": changed}
    elif damage == "other_role":
        effects.held_object_updates["other"] = held
    elif damage == "coordinated":
        effects.coordinated_held_object_updates = {"arm": held}
    elif damage == "articulation":
        effects.articulation_joint_updates = {"drawer": 1.0}
    elif damage == "hand":
        data.moved.joint_trajectory.positions[:, -1, 2] = 1.0
    else:
        data.moved.invocation_id = "other"
    result = gate.preview_place(
        data.action, data.request, data.context, data.picked, data.preview
    )
    assert not result["valid"]
    assert result["status"] == status
    assert data.compiler.ground.call_count == 2
    assert data.engine.plan.call_count <= 1
    assert data.context.task is data.initial_task


@pytest.mark.parametrize(
    "inspection, status",
    [
        (
            {"valid": False, "reason": "held_object_scene_geometry_unavailable"},
            "unavailable",
        ),
        (
            {"valid": False, "reason": "held_object_scene_envelope_rejected"},
            "unqualified",
        ),
        (
            {
                "valid": False,
                "reason": "robot_proxy_rejected",
                "collision": {"valid": False, "unavailable": "missing_mesh"},
            },
            "unavailable",
        ),
    ],
)
def test_safety_preview_distinguishes_missing_geometry_from_rejected_path(
    monkeypatch, inspection, status
) -> None:
    data = _preview_fixture(monkeypatch)
    monkeypatch.setattr(gate, "inspect_place_plan", lambda *_a: inspection)
    result = gate.preview_place(
        data.action, data.request, data.context, data.picked, data.preview, safety=True
    )
    assert result["status"] == status
    assert result["physical_evidence"] is False


def test_continuation_search_retains_its_existing_candidate_cap(monkeypatch) -> None:
    data = _fixture(monkeypatch)
    poses = torch.eye(4)[None, None].repeat(1, 20, 1, 1)
    poses[0, :, 2, 3] = torch.arange(20) * 0.01
    data[4].candidates = AffordancePoseCandidates(
        poses, torch.zeros(1, 20), torch.ones(1, 20, dtype=torch.bool)
    )
    monkeypatch.setattr(
        gate,
        "preview_place",
        lambda *_a, **_kw: {"valid": False, "status": "unqualified"},
    )
    with search_cost_scope() as cost:
        result = gate.recover(*data[:6])
    assert not result.plan_success.any()
    assert 0 < len(data[6]) <= 12
    assert cost.snapshot()["counters"]["continuation_grasp_candidate"] == len(data[6])
    assert cost.snapshot()["counters"]["continuation_pick_plan"] == len(data[6])


def test_failed_preview_restores_rng_and_recursion_guard(monkeypatch) -> None:
    data = _fixture(monkeypatch)
    before = torch.random.get_rng_state()

    def fail(*_args):
        torch.rand(1)
        raise ValueError("canonical_effect_monitor_unavailable")

    monkeypatch.setattr(gate, "preview_place", fail)
    with pytest.raises(ValueError, match="canonical_effect_monitor_unavailable"):
        gate.recover(*data[:6])
    assert torch.equal(before, torch.random.get_rng_state())
    assert not gate._PREVIEW.get()
