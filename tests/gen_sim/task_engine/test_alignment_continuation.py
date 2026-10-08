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

"""Only an owned, failed adjacent continuation may change a free-yaw heading."""

from __future__ import annotations

from dataclasses import dataclass, replace
import pickle
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import alignment_continuation as gate
from embodichain.gen_sim.task_engine._task_program import (
    actions,
    articulation_collision,
    held_alignment_recovery,
    motion,
    planning_probe,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    PlannerDiagnostics,
    StateDelta,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    MoveHeldObjectOptions,
)
from embodichain.lab.sim.atomic_actions.state import (
    HeldObjectState,
    RobotObservation,
    TaskState,
)
from embodichain.lab.task_program.semantics import (
    EffectAssurance,
    EffectMonitorRef,
    RegisteredSemanticCall,
    SemanticEffectKind,
    SkillPolicyPreset,
)


def _program(*, preserve=False, other=False, parallel=False, middle=False):
    align = RegisteredSemanticCall(
        call_id="gen_sim.align_held",
        arguments={"object": "item", "target": "goal", "preserve_yaw": preserve},
        resources={"primary": "active"},
    )
    place = RegisteredSemanticCall(
        call_id="gen_sim.place_upright",
        arguments={"object": "other" if other else "item", "target": "goal"},
        resources={"primary": "active"},
    )
    calls = [align, place]
    if middle:
        calls.insert(
            1,
            RegisteredSemanticCall(
                call_id="simulation.park", resources={"primary": "active"}
            ),
        )
    segments = tuple(
        SimpleNamespace(
            segment_id=f"s{index}",
            parallel_block=object() if parallel and index == 1 else None,
            calls=(SimpleNamespace(call=call, segment_call_index=0),),
        )
        for index, call in enumerate(calls)
    )
    return SimpleNamespace(program_id="program", iter_segments=lambda: iter(segments))


def _preset(*, verified=True, monitor=True):
    return SkillPolicyPreset(
        preset_id="verified",
        action_option_templates={},
        effect_assurance=(
            EffectAssurance.VERIFIED if verified else EffectAssurance.PROJECTED
        ),
        effect_monitors={
            name: EffectMonitorRef("builtin.composite_effect", "1")
            for name in (
                ("gen_sim.align_held", "gen_sim.place_upright")
                if monitor and verified
                else ()
            )
        },
    )


@pytest.mark.parametrize(
    "invalid",
    [None, "exact", "object", "parallel", "gap", "projected", "monitor", "unowned"],
)
def test_only_actual_adjacent_verified_owned_yaw_free_calls_bind(invalid):
    program = _program(
        preserve=invalid == "exact",
        other=invalid == "object",
        parallel=invalid == "parallel",
        middle=invalid == "gap",
    )
    routes = gate.bind_calls(
        program,
        () if invalid == "unowned" else ("program/s0:0",),
        _preset(verified=invalid != "projected", monitor=invalid != "monitor"),
    )
    assert bool(routes) is (invalid is None)
    if routes:
        assert routes[0].alignment_id == "program/s0:0"
        assert routes[0].place_id == "program/s1:0"


def _state():
    return (
        pickle.dumps((random.getstate(), np.random.get_state())),
        torch.random.get_rng_state().clone(),
    )


@dataclass(frozen=True)
class _Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics


@pytest.mark.parametrize("accepted", [True, False])
def test_original_joint_safe_plan_identity_and_rng_are_preserved(monkeypatch, accepted):
    original = _Plan(torch.tensor([True]), PlannerDiagnostics("test"))
    request, context, engine, preview = (
        object(),
        SimpleNamespace(robot=SimpleNamespace(qpos=torch.zeros(1, 4))),
        SimpleNamespace(robot=SimpleNamespace(device="cpu")),
        SimpleNamespace(route=SimpleNamespace(place_id="program/s1:0")),
    )
    monkeypatch.setattr(gate, "eligible", lambda *args: True)
    monkeypatch.setattr(gate, "inspect_alignment", lambda *args: {"valid": True})

    def inspect(*args):
        random.random()
        np.random.rand()
        torch.rand(())
        return {"valid": accepted}

    monkeypatch.setattr(gate, "inspect_place", inspect)
    monkeypatch.setattr(gate, "exact_requests", lambda *args: ())
    build = Mock(side_effect=AssertionError("safe original must not search"))
    before = _state()
    result = gate.recover(
        engine, request, context, original, preview=preview, build=build
    )
    after = _state()
    assert before[0] == after[0] and torch.equal(before[1], after[1])
    assert (result is original) is accepted
    assert bool(result.plan_success.all()) is accepted
    build.assert_not_called()


def test_unowned_plan_is_exact_identity_without_preview_or_rng(monkeypatch):
    plan = _Plan(torch.tensor([True]), PlannerDiagnostics("test"))
    monkeypatch.setattr(gate, "eligible", lambda *args: False)
    monkeypatch.setattr(
        gate, "inspect_place", lambda *args: pytest.fail("unowned preview")
    )
    assert gate.recover(None, None, None, plan, preview=None, build=None) is plan


@dataclass(frozen=True)
class _Scene:
    entities: dict


@dataclass(frozen=True)
class _Entity:
    pose: torch.Tensor
    confidence: float = 1.0


@dataclass(frozen=True)
class _Context:
    robot: RobotObservation
    task: TaskState
    scene: _Scene
    env_ids: torch.Tensor
    batch_size: int = 1
    control_dt: float = 0.04


@dataclass(frozen=True)
class _Request:
    goal: object
    binding: object
    motion_policy: MotionPolicy
    skill_options: object
    skill_id: str = "move_held_object"
    invocation_id: str = "program/s0:0"
    revision: int = 0
    tracking_policy: object = None
    recovery_policy: object = None
    phase_effect_gates: tuple = ()

    def snapshot(self):
        return replace(
            self,
            goal=replace(
                self.goal, object_target_pose=self.goal.object_target_pose.clone()
            ),
        )


@dataclass(frozen=True)
class _FullPlan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics
    joint_trajectory: TimedTrajectory
    commands: object
    expected_effects: StateDelta
    invocation_id: str
    skill_id: str


@pytest.fixture
def full_inputs(monkeypatch):
    qpos = torch.tensor([[0.0, 0.0, 0.02, 0.5]])
    arm, hand = JointPositionTarget("active", (0, 1)), JointPositionTarget("hand", (2,))
    endpoints = tuple(
        SimpleNamespace(
            task_state_key="active",
            resource_id="active",
            target=target,
            require_target=lambda cls, target=target: target,
            joint_positions=lambda name, **kwargs: torch.tensor(
                [[0.0 if name == "open" else 0.02]]
            ),
        )
        for target in (arm, hand)
    )
    binding = SimpleNamespace(
        endpoints=endpoints, endpoint=lambda slot, name: endpoints[name == "grasp"]
    )
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[:, 0, 2, 3], poses[:, 1, 0, 3] = 0.1, 0.004
    request = _Request(
        gate.HeldObjectPoseGoal(poses, world_yaw_free=True),
        binding,
        MotionPolicy(
            strategy="ik_interp", dynamic_collision_mode="off", sample_count=260
        ),
        MoveHeldObjectOptions(),
    )
    semantics = ObjectSemantics(
        AntipodalAffordance(
            mesh_vertices=torch.tensor(
                [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
            ),
            mesh_triangles=torch.tensor([[0, 1, 2]]),
        ),
        {},
        entity_id="item",
    )
    held = HeldObjectState(
        semantics, object_to_eef=torch.eye(4)[None], grasp_xpos=torch.eye(4)[None]
    )
    task = TaskState(1, "cpu", held_objects={"active": held})
    context = _Context(
        RobotObservation(qpos=qpos, qvel=torch.zeros_like(qpos), timestamp=1.0),
        task,
        _Scene({"item": _Entity(torch.eye(4)[None])}),
        torch.tensor([0]),
    )
    desired = qpos.clone()

    def fk(*, qpos, **unused):
        pose = torch.eye(4)[None].to(qpos)
        pose[:, :2, 3] = qpos[:, :2]
        return pose

    robot = SimpleNamespace(
        device=torch.device("cpu"),
        get_joint_ids=lambda name: (0, 1) if name == "active" else (2,),
        get_qpos_limits=lambda name, **kwargs: torch.tensor([[[-1.0, 1.0]]]).expand(
            1, 2 if name == "active" else 1, 2
        ),
        get_qpos=lambda target=False: desired if target else context.robot.qpos,
        get_qvel=lambda target=False: torch.zeros_like(qpos),
        compute_fk=fk,
        set_qpos=Mock(side_effect=AssertionError("preview cannot write Native")),
    )
    action = actions.GenSimMoveHeldObject()
    generator = SimpleNamespace(direct_transport_planning=False)
    action._planning_services = SimpleNamespace(
        robot=robot, device=robot.device, motion_generator=generator
    )
    action._staging_rigid_entity_ids = frozenset({"item"})
    action._staging_invocation_ids = frozenset({request.invocation_id})
    action._staging_retention_sources = {
        request.invocation_id: {
            "entity_id": "item",
            "resource_id": "active",
            "preset_id": "verified",
        }
    }
    engine = SimpleNamespace(
        robot=robot,
        motion_generator=generator,
        _actions={request.skill_id: action, "place": actions.GenSimPlace()},
    )
    for module in (motion, held_alignment_recovery, planning_probe, gate):
        monkeypatch.setattr(
            module, "_joint_velocity_limits", lambda *args: torch.ones(1, 4)
        )
    monkeypatch.setattr(
        articulation_collision, "check_robot_motion", lambda *args: {"valid": True}
    )
    monkeypatch.setattr(
        held_alignment_recovery, "check_robot_motion", lambda *args: {"valid": True}
    )

    def make_plan(selected=request, current=context, *, success=True, release=False):
        positions = current.robot.qpos[:, None].repeat(1, 3, 1)
        positions[:, 1:, 0] += torch.tensor([0.002, 0.004])
        if release:
            positions[:, 1:, 2] = torch.tensor([0.01, 0.0])
        else:
            positions[:, :, 2] = selected.binding.endpoint(
                "primary", "grasp"
            ).joint_positions("grasp")[:, None, 0]
        trajectory = TimedTrajectory.from_uniform_step(
            positions, env_ids=current.env_ids, step_dt=0.04
        )
        return _FullPlan(
            torch.tensor([success]),
            PlannerDiagnostics(
                "test", metadata={"gen_sim_invocation_policy": {"sample_count": 260}}
            ),
            trajectory,
            action._joint_command_sequence(
                selected, trajectory, active_mask=torch.tensor([success])
            ),
            StateDelta(held_object_updates={"active": None if release else held}),
            selected.invocation_id,
            selected.skill_id,
        )

    route = gate.bind_calls(_program(), (request.invocation_id,), _preset())[0]
    workflows = []

    def analyze(calls, *, workflow_id):
        workflows.append(workflow_id)
        return SimpleNamespace(calls=calls, workflow_id=workflow_id)

    def ground(workflow, index, current):
        alignment = workflow.calls[0].call_id == "gen_sim.align_held"
        identity = f"{workflow.workflow_id}:{index}"
        return SimpleNamespace(
            invocation=SimpleNamespace(invocation_id=identity),
            analyzed=SimpleNamespace(effect_assurance=EffectAssurance.VERIFIED),
            effect_monitor=object(),
            effect_spec=SimpleNamespace(
                effect_kind=(
                    SemanticEffectKind.ATTACH
                    if alignment
                    else SemanticEffectKind.RELEASE
                ),
                invocation_id=identity,
                state_expectations=(
                    SimpleNamespace(object_id="item", task_state_key="active"),
                ),
            ),
        )

    compiler = SimpleNamespace(
        integration=SimpleNamespace(
            engine=engine, link_call=lambda call: SimpleNamespace(preset=_preset())
        ),
        analyze=analyze,
        ground=ground,
    )
    preview = gate.bind_previews(compiler, (route,))[request.invocation_id]
    place_request = replace(
        request, skill_id="place", invocation_id=route.place_id, goal=SimpleNamespace()
    )
    engine.resolve = Mock(return_value=place_request)
    engine.plan_request = Mock(
        side_effect=lambda selected, current: make_plan(selected, current, release=True)
    )
    return SimpleNamespace(
        request=request,
        context=context,
        robot=robot,
        action=action,
        engine=engine,
        preview=preview,
        make_plan=make_plan,
        place_request=place_request,
        desired=desired,
        workflows=workflows,
    )


def test_exact_candidates_are_only_original_eight_offsets_and_keep_all_controls(
    full_inputs,
):
    p = full_inputs
    candidates = gate.exact_requests(p.request)
    assert len(candidates) == 8
    for angle, selected in candidates:
        assert selected.goal.world_yaw_free is False
        assert torch.equal(
            selected.goal.object_target_pose[:, :-1],
            p.request.goal.object_target_pose[:, :-1],
        )
        assert torch.equal(
            selected.goal.object_target_pose[..., :3, 3],
            p.request.goal.object_target_pose[..., :3, 3],
        )
        assert selected.motion_policy == p.request.motion_policy
        assert (
            selected.binding is p.request.binding
            and selected.invocation_id == p.request.invocation_id
        )


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "exact",
        "batch",
        "object",
        "unowned",
        "projected",
        "sampling",
        "direct",
        "foreign_engine",
        "foreign_place",
    ],
)
def test_runtime_scope_requires_real_factory_authority(full_inputs, invalid):
    p = full_inputs
    request, context = p.request, p.context
    if invalid == "exact":
        request = replace(request, goal=replace(request.goal, world_yaw_free=False))
    elif invalid == "batch":
        context = replace(context, env_ids=torch.tensor([1]))
    elif invalid == "object":
        p.action._staging_rigid_entity_ids = frozenset()
    elif invalid == "unowned":
        p.action._staging_invocation_ids = frozenset()
    elif invalid == "projected":
        p.action._staging_retention_sources = {}
    elif invalid == "sampling":
        request = replace(
            request, motion_policy=replace(request.motion_policy, sample_count=320)
        )
    elif invalid == "direct":
        p.engine.motion_generator.direct_transport_planning = True
    elif invalid == "foreign_engine":
        p.preview.compiler.integration.engine = object()
    elif invalid == "foreign_place":
        p.engine._actions["place"] = object()
    assert gate.eligible(p.engine, request, context, p.preview) is (invalid is None)


def test_canonical_place_prediction_keeps_real_context_task_and_native_untouched(
    full_inputs,
):
    p = full_inputs
    plan = p.make_plan()
    before = p.context.robot.qpos.clone()
    assert gate.inspect_place(p.engine, p.request, p.context, plan, p.preview)["valid"]
    selected, predicted = p.engine.plan_request.call_args.args
    assert selected is p.place_request
    assert predicted is not p.context and predicted.task is p.context.task
    assert torch.equal(predicted.robot.qpos, plan.joint_trajectory.positions[:, -1])
    assert predicted.scene.entities["item"] is not p.context.scene.entities["item"]
    assert torch.equal(p.context.robot.qpos, before) and torch.equal(
        p.robot.get_qpos(), before
    )
    p.robot.set_qpos.assert_not_called()
    assert len(p.workflows) == 2
    gate.inspect_place(p.engine, p.request, p.context, plan, p.preview)
    assert len(p.workflows) == 2


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "payload",
        "passive",
        "velocity",
        "release",
        "closed",
        "policy",
        "wrong_id",
        "monitor",
    ],
)
def test_place_preview_rejects_unqualified_full_commands_and_release(
    full_inputs, invalid
):
    p = full_inputs

    def build(selected, current):
        plan = p.make_plan(selected, current, release=True)
        positions = plan.joint_trajectory.positions.clone()
        if invalid == "payload":
            frame = plan.commands.frames[1]
            command = frame.commands[0]
            commands = replace(
                plan.commands,
                frames=(
                    plan.commands.frames[0],
                    replace(
                        frame,
                        commands=(
                            replace(
                                command,
                                payload=replace(
                                    command.payload,
                                    positions=command.payload.positions + 0.1,
                                ),
                            ),
                            frame.commands[1],
                        ),
                    ),
                    plan.commands.frames[2],
                ),
            )
            return replace(plan, commands=commands)
        if invalid == "passive":
            positions[:, -1, 3] += 0.001
        elif invalid == "velocity":
            positions[:, -1, 0] = 0.9
        elif invalid == "closed":
            positions[:, -1, 2] = 0.01
        if invalid in {"passive", "velocity", "closed"}:
            trajectory = replace(plan.joint_trajectory, positions=positions)
            plan = replace(
                plan,
                joint_trajectory=trajectory,
                commands=p.action._joint_command_sequence(
                    selected, trajectory, active_mask=plan.plan_success
                ),
            )
        if invalid == "release":
            plan = replace(plan, expected_effects=StateDelta())
        elif invalid == "policy":
            plan = replace(
                plan,
                diagnostics=PlannerDiagnostics(
                    "test",
                    metadata={"gen_sim_invocation_policy": {"sample_count": 320}},
                ),
            )
        elif invalid == "wrong_id":
            plan = replace(plan, invocation_id="foreign")
        return plan

    p.engine.plan_request.side_effect = build
    if invalid == "monitor":
        original = p.preview.compiler.ground

        def ground(*args):
            grounded = original(*args)
            grounded.effect_monitor = None
            return grounded

        p.preview.compiler.ground = ground
    assert gate.inspect_place(p.engine, p.request, p.context, p.make_plan(), p.preview)[
        "valid"
    ] is (invalid is None)


@pytest.mark.parametrize(
    "invalid",
    [None, "payload", "passive", "velocity", "held", "policy", "observed_collision"],
)
def test_alignment_final_certificate_is_strict_after_any_transform(
    monkeypatch, full_inputs, invalid
):
    p = full_inputs
    plan = p.make_plan()
    selected = gate.exact_requests(p.request)[5][1]
    if invalid == "payload":
        frame = plan.commands.frames[1]
        command = frame.commands[0]
        plan = replace(
            plan,
            commands=replace(
                plan.commands,
                frames=(
                    plan.commands.frames[0],
                    replace(
                        frame,
                        commands=(
                            replace(
                                command,
                                payload=replace(
                                    command.payload,
                                    positions=command.payload.positions + 0.1,
                                ),
                            ),
                            frame.commands[1],
                        ),
                    ),
                    plan.commands.frames[2],
                ),
            ),
        )
    elif invalid in {"passive", "velocity"}:
        positions = plan.joint_trajectory.positions.clone()
        positions[:, -1, 3 if invalid == "passive" else 0] += 0.1
        trajectory = replace(plan.joint_trajectory, positions=positions)
        plan = replace(
            plan,
            joint_trajectory=trajectory,
            commands=p.action._joint_command_sequence(
                selected, trajectory, active_mask=plan.plan_success
            ),
        )
    elif invalid == "held":
        plan = replace(plan, expected_effects=StateDelta())
    elif invalid == "policy":
        selected = replace(
            selected, motion_policy=replace(selected.motion_policy, sample_count=320)
        )
    elif invalid == "observed_collision":
        monkeypatch.setattr(
            gate, "_inspect_pick_motion", lambda *args: {"valid": False}
        )
    assert gate.inspect_alignment(p.engine, p.request, selected, p.context, plan)[
        "valid"
    ] is (invalid is None)


@pytest.mark.parametrize("changed", [False, True])
def test_exact_yaw_never_fakes_retained_command_authority(full_inputs, changed):
    p = full_inputs
    p.context.robot.qpos[:, 2] = 0.0
    p.desired[:, 2] = 0.3 + (0.001 if changed else 0.0)
    p.request.binding.endpoints[1].joint_positions = (
        lambda *args, **kwargs: torch.tensor([[0.3]])
    )
    selected = gate.exact_requests(p.request)[5][1]
    certificate = gate.inspect_alignment(
        p.engine, p.request, selected, p.context, p.make_plan(selected)
    )
    assert certificate["valid"] is (not changed)
    if not changed:
        assert certificate["retained_hand_boundary"]["accepted"]
    assert p.request.goal.world_yaw_free and not selected.goal.world_yaw_free
    p.robot.set_qpos.assert_not_called()


@pytest.mark.parametrize("outcome", ["accept", "reject", "error"])
def test_failed_joint_route_search_is_bounded_and_restores_all_rng(
    monkeypatch, full_inputs, outcome
):
    p = full_inputs
    original = p.make_plan()
    calls = []
    inspected = []
    monkeypatch.setattr(gate, "inspect_alignment", lambda *args: {"valid": True})

    def inspect(engine, selected, *args):
        inspected.append(selected)
        return {
            "valid": selected is not p.request and outcome == "accept",
            "reason": "place_failed",
        }

    monkeypatch.setattr(gate, "inspect_place", inspect)

    def build(selected):
        calls.append((random.random(), float(np.random.rand()), float(torch.rand(()))))
        assert gate._PREVIEW.get()
        if outcome == "error":
            raise RuntimeError("injected")
        return original

    before = _state()
    result = gate.recover(
        p.engine, p.request, p.context, original, preview=p.preview, build=build
    )
    after = _state()
    assert before[0] == after[0] and torch.equal(before[1], after[1])
    assert len(calls) == (1 if outcome == "accept" else 8)
    assert all(value == calls[0] for value in calls)
    assert bool(result.plan_success.all()) is (outcome == "accept")
    assert not gate._PREVIEW.get()
    assert result.expected_effects is original.expected_effects


def test_original_full_safe_route_keeps_exact_plan_object(full_inputs):
    p = full_inputs
    plan = p.make_plan()
    build = Mock(side_effect=AssertionError("no extra heading"))
    before = _state()
    result = gate.recover(
        p.engine, p.request, p.context, plan, preview=p.preview, build=build
    )
    after = _state()
    assert result is plan and result.commands is plan.commands
    assert before[0] == after[0] and torch.equal(before[1], after[1])
    build.assert_not_called()


def test_candidate_search_uses_preoriginal_rng_and_preserves_legacy_stop(
    monkeypatch, full_inputs
):
    p = full_inputs
    start = gate._RandomState.capture(set())
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    stopped = _state()
    monkeypatch.setattr(gate, "inspect_alignment", lambda *args: {"valid": True})
    monkeypatch.setattr(gate, "inspect_place", lambda *args: {"valid": False})
    draws = []

    def build(selected):
        draws.append((random.random(), float(np.random.rand()), float(torch.rand(()))))
        return p.make_plan(selected)

    gate.recover(
        p.engine,
        p.request,
        p.context,
        p.make_plan(),
        preview=p.preview,
        build=build,
        before_original=start,
    )
    after = _state()
    assert draws == [expected] * 8
    assert stopped[0] == after[0] and torch.equal(stopped[1], after[1])


def test_all_place_previews_start_from_exact_legacy_stop_even_after_candidate_draws(
    monkeypatch, full_inputs
):
    p = full_inputs
    start = gate._RandomState.capture(set())
    for _ in range(3):
        random.random()
        np.random.rand()
        torch.rand(())
    legacy = _state()
    previews = []
    builds = []
    monkeypatch.setattr(gate, "inspect_alignment", lambda *args: {"valid": True})

    def inspect(*args):
        observed = _state()
        previews.append(observed)
        random.random()
        np.random.rand()
        torch.rand(())
        return {"valid": False}

    def build(selected):
        random.random()
        np.random.rand()
        torch.rand(())
        builds.append(_state())
        return p.make_plan(selected)

    monkeypatch.setattr(gate, "inspect_place", inspect)
    gate.recover(
        p.engine,
        p.request,
        p.context,
        p.make_plan(),
        preview=p.preview,
        build=build,
        before_original=start,
    )
    assert len(previews) == 9 and len(builds) == 8
    assert all(
        state[0] == legacy[0] and torch.equal(state[1], legacy[1]) for state in previews
    )
    assert all(state[0] != legacy[0] for state in builds)
    after = _state()
    assert after[0] == legacy[0] and torch.equal(after[1], legacy[1])


@pytest.mark.parametrize("invalid", [None, "payload", "timing", "mutate_request"])
def test_engine_joint_gate_runs_after_transform_and_keeps_preview_out_of_initial_capture(
    monkeypatch, full_inputs, invalid
):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine

    p = full_inputs
    engine = object.__new__(GenSimActionEngine)
    engine._planning_services = p.action._planning_services
    engine._actions = p.engine._actions
    engine._motion_samples, engine._articulation_calls = {}, {}
    engine._cartesian_calls = frozenset()
    engine._alignment_continuations = {p.request.invocation_id: p.preview}
    p.preview.compiler.integration.engine = engine
    engine.resolve = p.engine.resolve
    engine.plan_request = p.engine.plan_request
    transforms = []

    def transform(selected, current, plan):
        transforms.append(selected.invocation_id)
        if invalid == "mutate_request":
            selected.goal.object_target_pose[:, -1, 0, 3] += 0.1
            return plan
        if invalid == "timing":
            trajectory = replace(plan.joint_trajectory, dt=plan.joint_trajectory.dt * 2)
            return replace(
                plan,
                joint_trajectory=trajectory,
                commands=p.action._joint_command_sequence(
                    selected, trajectory, active_mask=plan.plan_success
                ),
            )
        if invalid == "payload":
            frame = plan.commands.frames[0]
            command = frame.commands[0]
            return replace(
                plan,
                commands=replace(
                    plan.commands,
                    frames=(
                        replace(
                            frame,
                            commands=(
                                replace(
                                    command,
                                    payload=replace(
                                        command.payload,
                                        positions=command.payload.positions + 0.01,
                                    ),
                                ),
                                frame.commands[1],
                            ),
                        ),
                        *plan.commands.frames[1:],
                    ),
                ),
            )
        return plan

    def original(self, selected, current, *, plan_transform=None):
        plan = p.make_plan(selected, current)
        return (
            plan if plan_transform is None else plan_transform(selected, current, plan)
        )

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", original)
    result = engine._plan_request(p.request, p.context, plan_transform=transform)
    assert bool(result.plan_success.all()) is (invalid is None)
    assert transforms == [p.request.invocation_id]
    assert p.engine.plan_request.call_count == (1 if invalid is None else 0)


def test_owned_continuation_preserves_accepted_320_retime_without_preview(
    monkeypatch, full_inputs
):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine

    p = full_inputs
    engine = object.__new__(GenSimActionEngine)
    engine._planning_services = p.action._planning_services
    engine._actions = p.engine._actions
    engine._motion_samples = {p.request.invocation_id: 260}
    engine._articulation_calls, engine._cartesian_calls = {}, frozenset()
    p.preview.compiler.integration.engine = engine
    plan = p.make_plan()
    positions = plan.joint_trajectory.positions.clone()
    positions[:, 1:, 0] = torch.tensor([0.1, 0.2])
    trajectory = replace(plan.joint_trajectory, positions=positions)
    invalid = replace(
        plan,
        joint_trajectory=trajectory,
        commands=p.action._joint_command_sequence(
            p.request, trajectory, active_mask=plan.plan_success
        ),
    )
    calls = []

    def build(self, selected, current, *, plan_transform=None):
        calls.append(selected.motion_policy.sample_count)
        random.random()
        np.random.rand()
        torch.rand(())
        return invalid if selected.motion_policy.sample_count == 260 else plan

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", build)
    preview = Mock(side_effect=AssertionError("320 is outside continuation scope"))
    monkeypatch.setattr(gate, "inspect_place", preview)
    start = gate._RandomState.capture(set())
    outcomes = []
    for owned in (False, True):
        start.restore()
        calls.clear()
        engine._alignment_continuations = (
            {p.request.invocation_id: p.preview} if owned else {}
        )
        result = engine._plan_request(p.request, p.context)
        assert calls == [260, 320]
        assert result.plan_success.all()
        assert result.commands is plan.commands
        assert (
            result.diagnostics.metadata["gen_sim_invocation_policy"]["sample_count"]
            == 320
        )
        assert "gen_sim_alignment_continuation" not in result.diagnostics.metadata
        outcomes.append(_state())
    assert outcomes[0][0] == outcomes[1][0] and torch.equal(
        outcomes[0][1], outcomes[1][1]
    )
    preview.assert_not_called()


@pytest.mark.parametrize("exact", [False, True])
def test_real_canonical_task1102_composer_binds_monitored_place_and_conditional_fingerprint(
    monkeypatch, tmp_path, exact
):
    from copy import deepcopy
    import trimesh
    from embodichain.gen_sim.task_engine.agent import TaskAgent
    from embodichain.gen_sim.task_engine.interpretation import _INTENT_FIELD_DEFAULTS
    from embodichain.gen_sim.task_engine.orchestration.contracts import (
        ROLE_BINDINGS_SCHEMA,
    )
    from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
    from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.gen_sim.task_engine._task_program.assembly import load_deployment
    from embodichain.utils.utility import load_config, save_config

    mesh = tmp_path / "can.glb"
    trimesh.creation.box(extents=(0.06, 0.24, 0.06)).export(mesh)
    item = {
        "uid": "can",
        "shape": {"shape_type": "Mesh", "fpath": str(mesh)},
        "init_pos": [0.1, -0.2, 0.85],
        "init_rot": [90.0, 0.0, 0.0],
    }
    table = {
        "uid": "table",
        "shape": {"shape_type": "Cube", "size": [1.0, 1.0, 0.72]},
        "init_pos": [0.0, 0.0, 0.36],
    }
    scene = PreparedScene(
        source_config_path=tmp_path / "scene.yaml",
        scene_dir=tmp_path,
        planner_objects=(
            {**item, "runtime_uid": "can", "role": "rigid_object"},
            {**table, "runtime_uid": "table", "role": "background"},
        ),
        background=(table,),
        rigid_objects=(item,),
        articulations=(),
        uid_map={"can": "can", "table": "table"},
        table_top_z=0.72,
        z_rotation_degrees=0.0,
        body_scale_policy="preserve",
        body_scale=(1.0, 1.0, 1.0),
        asset_hashes={},
    )
    none = {
        "kind": "none",
        "reference": "",
        "step_id": "",
        "quantifier": "one",
        "count": 0,
    }
    step = {
        **deepcopy(_INTENT_FIELD_DEFAULTS),
        "id": "step_01",
        "task_type": "E2",
        "object": {**none, "kind": "scene_ref", "reference": "largest can"},
        "target": none,
        "required_arm": "right_arm",
        "orientation_goal": "upright",
        "depends_on": [],
    }
    candidate = TaskAgent(caller=lambda **kwargs: {"steps": [step]}).generate(
        "task1102", "Use right arm to upright the largest can", candidate_count=1
    )["candidates"][0]
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": "task1102",
        "candidate_id": candidate["candidate_id"],
        "reference_bindings": {"step_01.object": ["can"]},
        "role_bindings": {},
    }
    graph = SemanticTaskPlanner().plan(candidate, bindings, scene.planner_objects)
    original = deepcopy(graph)
    _, paths = generate_task_program_bundle(
        graph, scene, tmp_path / "bundle", robot_profile="dual_franka"
    )
    if exact:
        program = load_config(paths.program)
        for entry in program["program"]["items"]:
            call = entry["steps"]["call"]
            if call.get("call_id") == "gen_sim.align_held":
                call["arguments"]["preserve_yaw"] = True
        save_config(paths.program, program)
    args = dict(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.deployment.parent,
    )
    profile_before = deepcopy(args["skill_profile"])
    first = load_deployment(**args)
    factory = first.integration.adapter_factory
    routes = factory.alignment_continuation_routes
    assert bool(routes) is (not exact)
    if routes:
        assert (
            len(routes) == 1
            and routes[0].place_call.call_id == "simulation.place_relative"
        )
        assert (
            ".items[2]:0" in routes[0].alignment_id
            and ".items[3]:0" in routes[0].place_id
        )
        preset = next(
            p
            for p in factory.registration.robot_profile_binding.presets
            if p.preset_id == factory.runtime_preset
        )
        assert preset.effect_assurance is EffectAssurance.VERIFIED
        assert (
            preset.effect_monitors[routes[0].place_call.call_id].monitor_id
            == "builtin.composite_effect"
        )
        assert (
            factory.registration.call_catalog.descriptors[
                routes[0].place_call.call_id
            ].target_descriptor.skill_id
            == "place"
        )
        from embodichain.gen_sim.task_engine._task_program import assembly
        from embodichain.lab.task_program.runtime import assembly as runtime_assembly

        engine = Mock()
        engine._clearance_profile = object()
        compiler = SimpleNamespace(
            integration=SimpleNamespace(
                engine=engine, link_call=lambda call: SimpleNamespace(preset=preset)
            ),
            analyze=Mock(
                side_effect=lambda calls, **kwargs: SimpleNamespace(
                    calls=calls, **kwargs
                )
            ),
        )
        production_factory = object.__new__(assembly._TaskFactory)
        production_factory._clearance_routes = ()
        production_factory._alignment_continuation_routes = routes
        production_factory._runtime_preset = factory.runtime_preset
        production_factory._registration = factory.registration
        production_factory._robot_profile_binding = (
            factory.registration.robot_profile_binding
        )
        production_factory._scene_binding = factory.registration.scene_binding
        lowerers = (object(),)
        shared = Mock(return_value=lowerers)
        canonical = Mock(return_value=SimpleNamespace(compiler=compiler))
        monkeypatch.setattr(
            assembly.SimulationTaskProgramFactory,
            "create_registered_semantic_lowerers",
            shared,
        )
        monkeypatch.setattr(
            runtime_assembly, "assemble_semantic_executor_components", canonical
        )
        assert (
            production_factory.create_registered_semantic_lowerers(
                scene_registry=object(), engine=engine
            )
            is lowerers
        )
        previews = engine.bind_alignment_continuation.call_args.args[0]
        assert set(previews) == {routes[0].alignment_id}
        assert previews[routes[0].alignment_id].place_workflow.calls == (
            routes[0].place_call,
        )
        assert (
            compiler.analyze.call_count == 2
            and shared.call_count == 1
            and canonical.call_count == 1
        )
    monkeypatch.setattr(gate, "ALIGNMENT_CONTINUATION_REVISION", 999)
    changed = load_deployment(**args)
    assert (
        first.integration.integration_fingerprint
        != changed.integration.integration_fingerprint
    ) is (not exact)
    assert graph == original and args["skill_profile"] == profile_before
