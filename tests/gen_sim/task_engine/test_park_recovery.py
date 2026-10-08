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

"""Only a canonical released-hand Park may recover a rejected scene path."""

from __future__ import annotations

from dataclasses import dataclass, replace
import pickle
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import park_recovery as recovery
from embodichain.gen_sim.task_engine._task_program import (
    clearance_recovery,
    motion,
    planning_probe,
)
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    JointPositionGoal,
    JointPositionTarget,
    JointPositionCommand,
    MoveJoints,
    MotionPolicy,
    PlannerDiagnostics,
    StateDelta,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.control import PARK_COMMAND
from embodichain.lab.task_program.semantics import (
    EffectAssurance,
    RegisteredSemanticCall,
)


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics


def state():
    return (
        pickle.dumps((random.getstate(), np.random.get_state())),
        torch.random.get_rng_state().clone(),
    )


def test_unowned_park_keeps_exact_plan_without_search(monkeypatch):
    plan = Plan(torch.tensor([True]), PlannerDiagnostics("fixture"))
    monkeypatch.setattr(recovery, "eligible", lambda *args: False)
    assert (
        recovery.recover(
            None,
            None,
            None,
            plan,
            acceptance={"valid": False},
            build=None,
            simulation=None,
        )
        is plan
    )


def test_qualified_original_is_same_identity_without_ik_or_rng(monkeypatch):
    plan = Plan(torch.tensor([True]), PlannerDiagnostics("fixture"))
    monkeypatch.setattr(recovery, "eligible", lambda *args: True)
    monkeypatch.setattr(
        recovery, "basic_check", lambda *args, **kwargs: {"valid": True}
    )
    request = SimpleNamespace()
    engine = SimpleNamespace(robot=Mock())
    before = state()
    result = recovery.recover(
        engine,
        request,
        None,
        plan,
        acceptance={"valid": True},
        build=Mock(side_effect=AssertionError("safe route search")),
        simulation=None,
    )
    after = state()
    assert result is plan and before[0] == after[0] and torch.equal(before[1], after[1])
    engine.robot.compute_ik.assert_not_called()


@dataclass(frozen=True)
class Observation:
    qpos: torch.Tensor
    qvel: torch.Tensor


@dataclass(frozen=True)
class Context:
    robot: Observation
    task: object
    env_ids: torch.Tensor
    batch_size: int = 1
    control_dt: float = 0.04


@dataclass(frozen=True)
class Request:
    goal: JointPositionGoal
    binding: object
    motion_policy: MotionPolicy
    skill_id: str = "move_joints"
    invocation_id: str = "program/park:0"


@dataclass(frozen=True)
class FullPlan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics
    joint_trajectory: TimedTrajectory
    commands: object
    expected_effects: StateDelta
    invocation_id: str
    skill_id: str


@pytest.fixture
def inputs(monkeypatch):
    qpos = torch.tensor([[0.0, 0.0, 0.0, 0.2]])
    ctx = Context(
        Observation(qpos, torch.zeros_like(qpos)),
        SimpleNamespace(
            held_object_mask=lambda key: torch.tensor([False]),
            get_held_object=lambda key: None,
            coordinated_held_objects={},
        ),
        torch.tensor([0]),
    )
    ready = torch.tensor([[0.03, 0.03, 0.02]])
    target = JointPositionTarget("arm", (0, 1, 2))
    endpoint = SimpleNamespace(
        target=target,
        task_state_key="right",
        resource_id="right",
        commands={PARK_COMMAND: JointPositionCommand(ready)},
        require_target=lambda cls: target,
        joint_positions=lambda *args, **kwargs: ready.clone(),
    )
    binding = SimpleNamespace(endpoint=lambda *args: endpoint, endpoints=(endpoint,))
    req = Request(
        JointPositionGoal(PARK_COMMAND),
        binding,
        MotionPolicy(
            strategy="ik_interp", dynamic_collision_mode="off", sample_count=260
        ),
    )
    seeds = []

    def fk(*, qpos, **kwargs):
        pose = torch.eye(4)[None].to(qpos)
        pose[:, :3, 3] = qpos
        return pose

    def ik(*, pose, joint_seed, **kwargs):
        seeds.append(joint_seed.clone())
        random.random()
        np.random.rand()
        torch.rand(())
        return torch.tensor([True]), pose[:, :3, 3].clone()

    robot = SimpleNamespace(
        device=torch.device("cpu"),
        get_qpos=lambda: qpos,
        get_joint_ids=lambda **kwargs: (0, 1, 2),
        get_qpos_limits=lambda **kwargs: torch.tensor([[[-1.0, 1.0]]]).repeat(1, 3, 1),
        compute_fk=fk,
        compute_ik=Mock(side_effect=ik),
    )
    action = MoveJoints()
    action._planning_services = SimpleNamespace(device=robot.device)
    engine = SimpleNamespace(robot=robot, _actions={"move_joints": action})
    bound = SimpleNamespace(
        linked=SimpleNamespace(
            call=RegisteredSemanticCall(
                call_id="simulation.park", resources={"primary": "right"}
            )
        ),
        preset=SimpleNamespace(effect_assurance=EffectAssurance.VERIFIED),
        binding=SimpleNamespace(action_binding=binding),
    )
    preview = clearance_recovery._ParkPreview(
        SimpleNamespace(integration=SimpleNamespace(engine=engine)),
        SimpleNamespace(calls=(SimpleNamespace(bound=bound),)),
        0,
        req.invocation_id,
    )
    engine._park_recovery_previews = {req.invocation_id: preview}
    for module in (motion, clearance_recovery, recovery, planning_probe):
        monkeypatch.setattr(
            module,
            "_joint_velocity_limits",
            lambda robot, name=None: torch.full((1, 3 if name else 4), 2.0),
        )
    monkeypatch.setattr(
        clearance_recovery,
        "check_free_motion",
        lambda *args: {"valid": True, "frames": len(args[2])},
    )

    def make(selected=req, current=ctx):
        goal = (
            ready[:, None]
            if isinstance(selected.goal.target, str)
            else selected.goal.target
        )
        if goal.shape[1] == 1:
            active = torch.stack(
                [
                    torch.lerp(
                        current.robot.qpos[:, :3], goal[:, 0], torch.tensor(i / 4)
                    )
                    for i in range(5)
                ],
                1,
            )
        else:
            active = torch.cat(
                (
                    torch.stack(
                        [
                            torch.lerp(
                                current.robot.qpos[:, :3],
                                goal[:, 0],
                                torch.tensor(i / 2),
                            )
                            for i in range(3)
                        ],
                        1,
                    ),
                    torch.stack(
                        [
                            torch.lerp(goal[:, 0], goal[:, 1], torch.tensor(i / 2))
                            for i in range(1, 3)
                        ],
                        1,
                    ),
                ),
                1,
            )
        full = current.robot.qpos[:, None].repeat(1, 5, 1)
        full[:, :, :3] = active
        trajectory = TimedTrajectory.from_uniform_step(
            full, env_ids=current.env_ids, step_dt=0.04
        )
        return FullPlan(
            torch.tensor([True]),
            PlannerDiagnostics(
                "fixture", metadata={"gen_sim_clearance_park": {"valid": True}}
            ),
            trajectory,
            action._joint_command_sequence(
                selected, trajectory, active_mask=torch.tensor([True])
            ),
            StateDelta(),
            selected.invocation_id,
            selected.skill_id,
        )

    collision = {
        "valid": False,
        "joint_limits": {"valid": True},
        "velocity": {"valid": True},
        "collision": {
            "valid": False,
            "kind": "scene",
            "frame": 2,
            "pairs": [["hand", "obstacle"]],
        },
    }
    return SimpleNamespace(
        request=req,
        context=ctx,
        engine=engine,
        robot=robot,
        action=action,
        preview=preview,
        bound=bound,
        make=make,
        original=make(),
        collision=collision,
        ready=ready,
        seeds=seeds,
    )


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "unbound",
        "foreign",
        "unverified",
        "held",
        "batch",
        "tensor",
        "command",
        "parallel_resource",
        "strategy",
    ],
)
def test_only_owned_verified_literal_park_and_free_resource_are_eligible(
    inputs, invalid
):
    p = inputs
    request, context = p.request, p.context
    if invalid == "unbound":
        p.engine._park_recovery_previews = {}
    elif invalid == "foreign":
        p.preview.compiler.integration.engine = object()
    elif invalid == "unverified":
        p.bound.preset.effect_assurance = EffectAssurance.PROJECTED
    elif invalid == "held":
        context.task.held_object_mask = lambda key: torch.tensor([True])
    elif invalid == "batch":
        context = replace(context, env_ids=torch.tensor([1]))
    elif invalid == "tensor":
        request = replace(request, goal=JointPositionGoal(p.ready))
    elif invalid == "command":
        request.binding.endpoints[0].commands = {
            PARK_COMMAND: JointPositionCommand(p.ready + 0.1)
        }
        p.bound.binding.action_binding = SimpleNamespace(
            endpoint=lambda *args: SimpleNamespace(
                target=p.request.binding.endpoints[0].target,
                task_state_key="right",
                commands={PARK_COMMAND: JointPositionCommand(p.ready)},
            )
        )
    elif invalid == "parallel_resource":
        p.bound.linked.call = RegisteredSemanticCall(
            call_id="simulation.park", resources={"primary": "other"}
        )
    elif invalid == "strategy":
        request = replace(
            request, motion_policy=replace(request.motion_policy, strategy="motion_gen")
        )
    assert recovery.eligible(p.engine, request, context) is (invalid is None)


@pytest.mark.parametrize("failure", [None, "ik", "scene", "exception"])
def test_bounded_four_routes_preserve_ready_controls_native_and_rng(
    monkeypatch, inputs, failure
):
    p = inputs
    built = []
    if failure == "ik":
        p.robot.compute_ik.side_effect = lambda **kwargs: (
            torch.tensor([False]),
            torch.zeros(1, 3),
        )
    if failure == "scene":
        monkeypatch.setattr(
            clearance_recovery,
            "check_free_motion",
            lambda *args: {"valid": False, "kind": "scene", "frame": 2},
        )

    def build(selected, current):
        built.append(selected)
        assert not recovery.eligible(p.engine, selected, current)
        assert torch.equal(selected.goal.target[:, -1], p.ready)
        assert (
            selected.motion_policy is p.request.motion_policy
            and selected.binding is p.request.binding
        )
        if failure == "exception":
            raise RuntimeError("injected")
        return p.make(selected, current)

    before = state()
    qpos = p.robot.get_qpos().clone()
    result = recovery.recover(
        p.engine,
        p.request,
        p.context,
        p.original,
        acceptance=p.collision,
        build=build,
        simulation=object(),
    )
    after = state()
    assert (
        before[0] == after[0]
        and torch.equal(before[1], after[1])
        and torch.equal(qpos, p.robot.get_qpos())
    )
    assert result.plan_success.all().item() is (failure is None)
    assert len(built) <= (1 if failure is None else 4)
    if failure:
        assert result.diagnostics.failure.code == "park_clearance_failed"


@pytest.mark.parametrize(
    "bad",
    [
        "initial",
        "ready_collision",
        "self",
        "limits",
        "velocity",
        "end",
        "dt",
        "passive",
        "payload",
    ],
)
def test_invalid_original_never_searches(monkeypatch, inputs, bad):
    p = inputs
    acceptance = {**p.collision, "collision": dict(p.collision["collision"])}
    plan = p.original
    if bad == "initial":
        acceptance["collision"]["frame"] = 0
    elif bad == "ready_collision":
        acceptance["collision"]["frame"] = 5
    elif bad == "self":
        acceptance["collision"]["kind"] = "self"
    elif bad in {"limits", "velocity"}:
        acceptance["joint_limits" if bad == "limits" else "velocity"] = {"valid": False}
    elif bad in {"end", "passive"}:
        positions = plan.joint_trajectory.positions.clone()
        positions[:, -1, 0 if bad == "end" else 3] += 0.001
        trajectory = replace(plan.joint_trajectory, positions=positions)
        plan = replace(
            plan,
            joint_trajectory=trajectory,
            commands=p.action._joint_command_sequence(
                p.request, trajectory, active_mask=plan.plan_success
            ),
        )
        if bad == "passive":
            acceptance = {"valid": False, "reason": "uncommanded_joint_changed"}
    elif bad == "dt":
        plan = replace(
            plan,
            joint_trajectory=replace(
                plan.joint_trajectory, dt=plan.joint_trajectory.dt * 2
            ),
        )
    else:
        frame = plan.commands.frames[1]
        cmd = frame.commands[0]
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
                                cmd,
                                payload=replace(
                                    cmd.payload, positions=cmd.payload.positions + 0.01
                                ),
                            ),
                        ),
                    ),
                    *plan.commands.frames[2:],
                ),
            ),
        )
    build = Mock(side_effect=AssertionError("must not search"))
    result = recovery.recover(
        p.engine,
        p.request,
        p.context,
        plan,
        acceptance=acceptance,
        build=build,
        simulation=object(),
    )
    assert not result.plan_success.any()
    build.assert_not_called()
    p.robot.compute_ik.assert_not_called()


def test_complete_safe_original_is_exact_identity_with_real_basic_check(inputs):
    p = inputs
    before = state()
    acceptance = clearance_recovery.validate_motion(
        p.original, p.context, p.robot, object(), "arm"
    )
    build = Mock(side_effect=AssertionError("original route is safe"))
    result = recovery.recover(
        p.engine,
        p.request,
        p.context,
        p.original,
        acceptance=acceptance,
        build=build,
        simulation=object(),
    )
    after = state()
    assert result is p.original and result.commands is p.original.commands
    assert before[0] == after[0] and torch.equal(before[1], after[1])
    build.assert_not_called()
    p.robot.compute_ik.assert_not_called()


@pytest.mark.parametrize("bad", ["end", "grid", "payload", "bounds", "passive"])
def test_no_invalid_emitted_candidate_can_be_accepted(inputs, bad):
    p = inputs

    def build(selected, current):
        plan = p.make(selected, current)
        if bad == "payload":
            frame = plan.commands.frames[1]
            cmd = frame.commands[0]
            return replace(
                plan,
                commands=replace(
                    plan.commands,
                    frames=(
                        plan.commands.frames[0],
                        replace(
                            frame,
                            commands=(
                                replace(
                                    cmd,
                                    payload=replace(
                                        cmd.payload,
                                        positions=cmd.payload.positions + 0.01,
                                    ),
                                ),
                            ),
                        ),
                        *plan.commands.frames[2:],
                    ),
                ),
            )
        positions = plan.joint_trajectory.positions.clone()
        if bad == "end":
            positions[:, -1, 0] += 0.001
        elif bad == "bounds":
            positions[:, 2, 0] = 1.2
        elif bad == "passive":
            positions[:, 2, 3] += 0.001
        trajectory = replace(
            plan.joint_trajectory,
            positions=positions,
            dt=plan.joint_trajectory.dt * (2 if bad == "grid" else 1),
        )
        return replace(
            plan,
            joint_trajectory=trajectory,
            commands=p.action._joint_command_sequence(
                selected, trajectory, active_mask=plan.plan_success
            ),
        )

    result = recovery.recover(
        p.engine,
        p.request,
        p.context,
        p.original,
        acceptance=p.collision,
        build=build,
        simulation=object(),
    )
    assert (
        not result.plan_success.any()
        and result.diagnostics.failure.code == "park_clearance_failed"
    )
    assert len(result.diagnostics.metadata["gen_sim_park_recovery"]["attempts"]) == 4


@pytest.mark.parametrize("transformed_end", [False, True])
def test_actual_gen_sim_engine_keeps_nested_scene_gate_and_rechecks_transform(
    monkeypatch, inputs, transformed_end
):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )

    p = inputs
    engine = object.__new__(GenSimActionEngine)
    engine._planning_services = SimpleNamespace(
        robot=p.robot, motion_generator=object()
    )
    engine._actions = p.engine._actions
    engine._motion_samples, engine._articulation_calls = {}, {}
    engine._cartesian_calls = frozenset()
    engine._clearance_requests = {}
    p.preview.compiler.integration.engine = engine
    engine.bind_release_clearance({"clear": (p.preview,)}, simulation=object())
    calls = []

    def base(self, selected, current, *, plan_transform=None):
        calls.append(selected.goal.target)
        plan = p.make(selected, current)
        plan = replace(
            plan,
            diagnostics=PlannerDiagnostics(
                "fixture",
                metadata={
                    "tensor_goal": isinstance(selected.goal.target, torch.Tensor)
                },
            ),
        )
        return (
            plan if plan_transform is None else plan_transform(selected, current, plan)
        )

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", base)

    def scene(plan, *args):
        return (
            {
                "valid": True,
                "joint_limits": {"valid": True},
                "velocity": {"valid": True},
            }
            if plan.diagnostics.metadata["tensor_goal"]
            else p.collision
        )

    monkeypatch.setattr(clearance_recovery, "validate_motion", scene)
    monkeypatch.setattr(recovery, "validate_motion", scene)
    transformed = []

    def transform(selected, current, plan):
        if isinstance(selected.goal.target, torch.Tensor):
            transformed.append(selected.goal.target.clone())
            if transformed_end:
                positions = plan.joint_trajectory.positions.clone()
                positions[:, -1, 0] += 0.001
                trajectory = replace(plan.joint_trajectory, positions=positions)
                return replace(
                    plan,
                    joint_trajectory=trajectory,
                    commands=p.action._joint_command_sequence(
                        selected, trajectory, active_mask=plan.plan_success
                    ),
                )
        return plan

    result = engine._plan_request(p.request, p.context, plan_transform=transform)
    assert result.plan_success.all().item() is (not transformed_end)
    assert calls[0] == PARK_COMMAND and all(
        isinstance(goal, torch.Tensor) for goal in calls[1:]
    )
    assert len(transformed) == (4 if transformed_end else 1)
    assert result.diagnostics.metadata["gen_sim_park_recovery"]["accepted"] is (
        not transformed_end
    )
    if transformed_end:
        assert result.diagnostics.failure.code == "park_clearance_failed"


def test_clearance_owned_preview_uses_same_park_engine_path(monkeypatch, inputs):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )

    p = inputs
    engine = object.__new__(GenSimActionEngine)
    engine._planning_services = SimpleNamespace(
        robot=p.robot, motion_generator=object()
    )
    engine._actions = p.engine._actions
    engine._motion_samples, engine._articulation_calls = {}, {}
    engine._cartesian_calls = frozenset()
    engine._clearance_requests = {}
    p.preview.compiler.integration.engine = engine
    engine.bind_release_clearance({"clear": (p.preview,)}, simulation=object())
    clear = replace(p.request, invocation_id="clear", skill_id="move_end_effector")
    monkeypatch.setattr(
        AtomicActionEngine,
        "_plan_request",
        lambda self, selected, current, **kwargs: replace(
            p.make(selected, current),
            diagnostics=PlannerDiagnostics(
                "fixture",
                metadata={
                    "tensor_goal": isinstance(selected.goal.target, torch.Tensor)
                },
            ),
        ),
    )
    scene = lambda plan, *args: (
        {"valid": True, "joint_limits": {"valid": True}, "velocity": {"valid": True}}
        if plan.diagnostics.metadata["tensor_goal"]
        else p.collision
    )
    monkeypatch.setattr(clearance_recovery, "validate_motion", scene)
    monkeypatch.setattr(recovery, "validate_motion", scene)
    parks = []

    def preview(request, current, original, *, build, **kwargs):
        predicted = replace(
            current, robot=replace(current.robot, qpos=current.robot.qpos.clone())
        )
        parks.append(build(p.request, predicted))
        return original

    monkeypatch.setattr(clearance_recovery, "recover", preview)
    engine._plan_request(clear, p.context)
    assert (
        parks[0].plan_success.all()
        and parks[0].diagnostics.metadata["gen_sim_park_recovery"]["accepted"]
    )
