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

"""Invocation policy ownership, independent of recipe suffixes and call types."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
    GenSimActionEngine,
    bind_cartesian_calls,
    bind_motion_samples,
    cartesian_call_declarations,
    motion_sample_declarations,
    bind_pour_receivers,
    pour_receiver_declarations,
)
from embodichain.lab.sim.atomic_actions import AtomicActionEngine, PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy


def node(family: str, name: str, *, on: bool = False) -> dict:
    call = {"kind": "pick", "object": "payload"}
    if family == "E5":
        call = {
            "kind": "registered",
            "call_id": "simulation.coordinated_transport",
            "arguments": {"object": "payload", "target": "destination"},
        }
        if on:
            call["arguments"].update(reference="support", relation="on")
    return {"id": name, "task_type": family, "task_instance_id": name, "call": call}


@pytest.mark.parametrize("first", ["E1", "E2", "E3", "E4", "E5", "E6"])
@pytest.mark.parametrize("later", ["E1", "E2", "E3", "E4", "E5", "E6"])
def test_appending_any_recipe_preserves_prefix_motion_budget(first, later):
    initial = node(first, "first", on=True)
    alone = motion_sample_declarations({"nodes": [initial]})
    combined = motion_sample_declarations(
        {"nodes": [initial, node(later, "later", on=True)]}
    )
    assert alone["first"] == combined["first"]


def test_only_current_recipe_drawer_and_on_requirements_raise_the_budget():
    graph = {
        "nodes": [
            node("E1", "ordinary"),
            node("E1", "drawer"),
            node("E5", "directional"),
            node("E5", "on", on=True),
        ]
    }
    assert motion_sample_declarations(graph, drawer_groups=frozenset({"drawer"})) == {
        "ordinary": 140,
        "drawer": 260,
        "directional": 140,
        "on": 260,
    }


@pytest.mark.parametrize(
    "marker", ["hand_over", "gen_sim.clear_released", "gen_sim.articulation_withdraw"]
)
def test_later_recipe_does_not_change_ordinary_cartesian_mode(marker):
    ordinary = node("E1", "ordinary")
    later = node("E4", "later")
    later["call"] = (
        {"kind": "hand_over"}
        if marker == "hand_over"
        else {"kind": "registered", "call_id": marker, "arguments": {}}
    )
    assert cartesian_call_declarations({"nodes": [ordinary]}) == []
    assert cartesian_call_declarations({"nodes": [ordinary, later]}) == ["later"]


def test_cartesian_scope_binding_rejects_stale_or_duplicate_names():
    segments = [
        SimpleNamespace(
            name="one",
            segment_id="segment-0",
            calls=(SimpleNamespace(segment_call_index=0),),
        ),
        SimpleNamespace(
            name="two",
            segment_id="segment-1",
            calls=(SimpleNamespace(segment_call_index=0),),
        ),
    ]
    program = SimpleNamespace(program_id="test", iter_segments=lambda: iter(segments))
    assert bind_cartesian_calls(["two"], program) == ("test/segment-1:0",)
    for value in (["absent"], ["two", "two"], [True], {"two": True}):
        with pytest.raises(ValueError):
            bind_cartesian_calls(value, program)


def test_cartesian_planning_scope_restores_on_nested_failure(monkeypatch):
    from dataclasses import dataclass, field
    from embodichain.gen_sim.task_engine._task_program import motion

    @dataclass
    class Diagnostics:
        metadata: dict = field(default_factory=dict)

    @dataclass
    class Plan:
        diagnostics: Diagnostics = field(default_factory=Diagnostics)

    observed = []

    def plan(self, request, context=None, *, plan_transform=None):
        observed.append(motion._CARTESIAN_APPROACH.get())
        if request.invocation_id == "fails":
            raise RuntimeError("injected")
        return Plan()

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", plan)
    engine = object.__new__(GenSimActionEngine)
    engine._cartesian_calls = frozenset({"special", "fails"})
    engine._articulation_calls = {}

    def request(name):
        return SimpleNamespace(
            invocation_id=name,
            motion_policy=MotionPolicy(sample_count=140),
            skill_id="move_end_effector",
            goal=None,
        )

    with motion.cartesian_approach_scope(True):
        result = engine._plan_request(request("ordinary"))
        assert motion._CARTESIAN_APPROACH.get() is True
        with pytest.raises(RuntimeError, match="injected"):
            engine._plan_request(request("fails"))
        assert motion._CARTESIAN_APPROACH.get() is True
    engine._plan_request(request("special"))
    assert observed == [False, True, True]
    assert motion._CARTESIAN_APPROACH.get() is False
    assert (
        result.diagnostics.metadata["gen_sim_invocation_policy"]["cartesian_approach"]
        is False
    )


@dataclass(frozen=True)
class _VelocityPlan:
    plan_success: torch.Tensor
    joint_trajectory: object
    diagnostics: PlannerDiagnostics


@pytest.mark.parametrize(
    ("accepted_count", "expected_calls", "expected_attempts"),
    [
        (260, [140, 260], [260]),
        (320, [140, 260, 320], [260, 320]),
    ],
)
def test_velocity_failure_retries_only_this_gensim_invocation_with_denser_samples(
    monkeypatch, tmp_path, accepted_count, expected_calls, expected_attempts
):
    """A coarse full-robot path retries at denser budgets without changing limits."""

    robot_file = tmp_path / "robot.urdf"
    robot_file.write_text(
        '<robot><joint name="arm"><limit velocity="1"/></joint></robot>'
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=str(robot_file)),
        joint_names=["arm"],
        get_joint_ids=lambda **kwargs: [0],
        get_qvel_limits=lambda **kwargs: torch.ones(1, 1),
    )
    calls = []

    def plan(_self, request, context=None, *, plan_transform=None):
        del context, plan_transform
        sample_count = request.motion_policy.sample_count
        calls.append(sample_count)
        positions = torch.zeros(1, 2, 1)
        if sample_count < accepted_count:
            # 0.2 rad in 0.04 s is 5x the 1 rad/s limit.
            positions[0, 1, 0] = 0.2
        return _VelocityPlan(
            plan_success=torch.tensor([True]),
            joint_trajectory=SimpleNamespace(
                positions=positions,
                dt=torch.tensor([[0.0, 0.04]]),
            ),
            diagnostics=PlannerDiagnostics(backend="fixture"),
        )

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", plan)
    engine = object.__new__(GenSimActionEngine)
    engine._motion_samples = {"task/segment:0": 140}
    engine._cartesian_calls = frozenset()
    engine._articulation_calls = {}
    engine._planning_services = SimpleNamespace(robot=robot)

    @dataclass(frozen=True)
    class Request:
        invocation_id: str
        motion_policy: MotionPolicy
        skill_id: str = "move_joints"
        goal: object | None = None

    request = Request("task/segment:0", MotionPolicy(sample_count=140))
    result = engine._plan_request(request)

    assert calls == expected_calls
    assert request.motion_policy.sample_count == 140
    assert (
        result.diagnostics.metadata["gen_sim_invocation_policy"]["sample_count"]
        == accepted_count
    )
    assert (
        result.diagnostics.metadata["gen_sim_invocation_policy"][
            "velocity_retime_attempts"
        ]
        == expected_attempts
    )
    assert robot.get_qvel_limits()[0, 0] == 1


def test_valid_gensim_plan_does_not_pay_for_velocity_retry(monkeypatch, tmp_path):
    """A valid default path is planned once and keeps the 140-sample budget."""

    robot_file = tmp_path / "robot.urdf"
    robot_file.write_text(
        '<robot><joint name="arm"><limit velocity="1"/></joint></robot>'
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=str(robot_file)),
        joint_names=["arm"],
        get_joint_ids=lambda **kwargs: [0],
        get_qvel_limits=lambda **kwargs: torch.ones(1, 1),
    )
    calls = []

    def plan(_self, request, context=None, *, plan_transform=None):
        del context, plan_transform
        calls.append(request.motion_policy.sample_count)
        return _VelocityPlan(
            plan_success=torch.tensor([True]),
            joint_trajectory=SimpleNamespace(
                positions=torch.zeros(1, 2, 1),
                dt=torch.tensor([[0.0, 0.04]]),
            ),
            diagnostics=PlannerDiagnostics(backend="fixture"),
        )

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", plan)
    engine = object.__new__(GenSimActionEngine)
    engine._motion_samples = {"task/segment:0": 140}
    engine._cartesian_calls = frozenset()
    engine._articulation_calls = {}
    engine._planning_services = SimpleNamespace(robot=robot)

    @dataclass(frozen=True)
    class Request:
        invocation_id: str
        motion_policy: MotionPolicy
        skill_id: str = "move_joints"
        goal: object | None = None

    result = engine._plan_request(
        Request("task/segment:0", MotionPolicy(sample_count=140))
    )

    assert calls == [140]
    assert (
        result.diagnostics.metadata["gen_sim_invocation_policy"]["sample_count"] == 140
    )


def test_compiled_call_binding_is_exact_and_does_not_use_object_identity():
    def segment(name, index):
        return SimpleNamespace(
            name=name,
            segment_id=f"segment-{index}",
            calls=(SimpleNamespace(segment_call_index=0),),
        )

    segments = [segment("first", 0), segment("second", 1)]
    program = SimpleNamespace(program_id="test", iter_segments=lambda: iter(segments))
    assert bind_motion_samples({"first": 140, "second": 260}, program) == (
        ("test/segment-0:0", 140),
        ("test/segment-1:0", 260),
    )
    for bad in (
        {"first": 140},
        {"first": 140, "second": 260, "absent": 140},
        {"first": True, "second": 260},
    ):
        with pytest.raises(ValueError):
            bind_motion_samples(bad, program)


def test_resolution_owns_policies_and_cannot_leak_between_same_skill_calls(monkeypatch):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Invocation:
        invocation_id: str
        motion_policy: MotionPolicy

    monkeypatch.setattr(AtomicActionEngine, "_resolve", lambda self, value: value)
    engine = object.__new__(GenSimActionEngine)
    engine._motion_samples = {"first": 140, "second": 260}
    base = MotionPolicy(sample_count=140)
    first = Invocation("first", base)
    second = Invocation("second", base)
    assert engine._resolve(second).motion_policy.sample_count == 260
    assert engine._resolve(first).motion_policy.sample_count == 140
    assert base.sample_count == 140
    assert second.motion_policy is base
    larger = Invocation("second", MotionPolicy(sample_count=400))
    assert engine._resolve(larger).motion_policy.sample_count == 400
    with pytest.raises(ValueError, match="invocation"):
        engine._resolve(Invocation("unknown", base))


def test_resolution_failure_does_not_mutate_another_calls_policy(monkeypatch):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Invocation:
        invocation_id: str
        motion_policy: MotionPolicy

    def resolve(self, value):
        if value.invocation_id == "failed":
            raise RuntimeError("injected resolution failure")
        return value

    monkeypatch.setattr(AtomicActionEngine, "_resolve", resolve)
    engine = object.__new__(GenSimActionEngine)
    engine._motion_samples = {"failed": 260, "next": 140}
    policy = MotionPolicy(sample_count=140)
    with pytest.raises(RuntimeError, match="injected"):
        engine._resolve(Invocation("failed", policy))
    assert engine._resolve(Invocation("next", policy)).motion_policy.sample_count == 140
    assert policy.sample_count == 140


def test_two_pours_of_the_same_object_keep_distinct_receiver_bindings():
    from embodichain.gen_sim.task_engine._task_program.actions import GenSimPour

    def recipe(name, receiver):
        return [
            {
                "id": name + "_stage",
                "task_instance_id": name,
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.move_held_object",
                    "arguments": {"object": "cup", "reference": receiver},
                },
            },
            {
                "id": name + "_pour",
                "task_instance_id": name,
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.pour",
                    "arguments": {"object": "cup"},
                },
            },
        ]

    first = recipe("first", "receiver_a")
    graph = {"nodes": first + recipe("second", "receiver_b")}
    assert pour_receiver_declarations({"nodes": first}) == {"first_pour": "receiver_a"}
    declarations = pour_receiver_declarations(graph)
    assert declarations == {"first_pour": "receiver_a", "second_pour": "receiver_b"}
    segments = [
        SimpleNamespace(
            name=n["id"],
            segment_id=f"s{i}",
            calls=(
                SimpleNamespace(
                    segment_call_index=0,
                    call=SimpleNamespace(call_id=n["call"]["call_id"]),
                ),
            ),
        )
        for i, n in enumerate(graph["nodes"])
    ]
    program = SimpleNamespace(program_id="test", iter_segments=lambda: iter(segments))
    bindings = dict(bind_pour_receivers(declarations, program))
    action = GenSimPour(bindings)
    assert action._receiver_id("test/s1:0") == "receiver_a"
    assert action._receiver_id("test/s3:0") == "receiver_b"
    bindings["test/s1:0"] = "mutated"
    assert action._receiver_id("test/s1:0") == "receiver_a"
    with pytest.raises(ValueError, match="Unbound"):
        action._receiver_id("unknown")
    with pytest.raises(ValueError, match="exact receiver"):
        bind_pour_receivers({"first_pour": "receiver_a"}, program)


@dataclass(frozen=True)
class _PickTransformPlan:
    plan_success: torch.Tensor
    joint_trajectory: object
    commands: object
    diagnostics: PlannerDiagnostics


@pytest.fixture
def pick_transform_inputs(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import (
        actions,
        articulation_collision,
        motion,
    )
    from embodichain.lab.sim.atomic_actions import (
        JointPositionTarget,
        ObjectSemantics,
        PickUpOptions,
        TimedTrajectory,
    )
    from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import GraspGoal, PickUp

    affordance = AntipodalAffordance()
    affordance.set_custom_config(actions._PICK_GRASP_RULE, ("can", "upright"))
    targets = (JointPositionTarget("arm", (0,)), JointPositionTarget("hand", (1,)))
    endpoints = tuple(
        SimpleNamespace(task_state_key="held", require_target=lambda _, t=t: t)
        for t in targets
    )
    request = SimpleNamespace(
        invocation_id="test/pick:0",
        skill_id="pick_up",
        goal=GraspGoal(
            ObjectSemantics(affordance=affordance, geometry={}, entity_id="can")
        ),
        skill_options=PickUpOptions(),
        motion_policy=MotionPolicy(strategy="ik_interp", dynamic_collision_mode="off"),
        binding=SimpleNamespace(
            endpoint=lambda _, name: endpoints[name == "grasp"], endpoints=endpoints
        ),
    )
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=SimpleNamespace(qpos=torch.zeros(1, 2)),
    )
    robot = SimpleNamespace(
        device=torch.device("cpu"),
        get_qpos_limits=lambda **kwargs: torch.tensor([[[-1.0, 1.0]]]),
    )
    engine = object.__new__(GenSimActionEngine)
    engine._motion_samples = {}
    engine._cartesian_calls = frozenset()
    engine._articulation_calls = {}
    engine._actions = {"pick_up": actions.GenSimPickUp()}
    engine._planning_services = SimpleNamespace(
        robot=robot, motion_generator=SimpleNamespace(direct_transport_planning=False)
    )
    engine.initial_context = Mock(return_value=context)

    def make_plan(positions):
        trajectory = TimedTrajectory.from_uniform_step(
            positions, env_ids=context.env_ids, step_dt=0.04
        )
        commands = PickUp._joint_command_sequence(
            object.__new__(PickUp),
            request,
            trajectory,
            active_mask=torch.tensor([True]),
        )
        return _PickTransformPlan(
            torch.tensor([True]),
            trajectory,
            commands,
            PlannerDiagnostics(backend="transform_fixture"),
        )

    first = make_plan(torch.zeros(1, 3, 2))

    def shared_plan(self, selected, current=None, *, plan_transform=None):
        return (
            first
            if plan_transform is None
            else plan_transform(
                selected, context if current is None else current, first
            )
        )

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", shared_plan)
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.ones(1, 2)
    )
    collision = Mock(
        side_effect=lambda robot, positions, part: {
            "valid": bool((positions[:, 0] == 0).all())
        }
    )
    monkeypatch.setattr(articulation_collision, "check_robot_motion", collision)
    return SimpleNamespace(
        engine=engine,
        request=request,
        context=context,
        first=first,
        make_plan=make_plan,
        collision=collision,
    )


@pytest.mark.parametrize("failure", ["other_arm", "velocity"])
@pytest.mark.parametrize("implicit_context", [False, True])
def test_pick_transform_rechecks_final_full_commands_and_fails_closed(
    pick_transform_inputs, failure, implicit_context
):
    probe = pick_transform_inputs

    def transform(request, context, plan):
        positions = plan.joint_trajectory.positions.clone()
        positions[:, 1:, 0] = 0.01 if failure == "other_arm" else 0.2
        return probe.make_plan(positions)

    result = probe.engine._plan_request(
        probe.request,
        None if implicit_context else probe.context,
        plan_transform=transform,
    )
    assert result.plan_success.tolist() == [False]
    assert result.diagnostics.failure.code == "pick_robot_safety_failed"
    assert result.diagnostics.failure.retryable is False
    assert probe.engine.initial_context.call_count == int(implicit_context)


def test_pick_without_transform_does_not_repeat_robotproxy_inspection(
    monkeypatch, pick_transform_inputs
):
    from embodichain.gen_sim.task_engine._task_program import pick_safety

    probe = pick_transform_inputs
    monkeypatch.setattr(
        pick_safety,
        "_inspect_pick_plan",
        lambda *args: pytest.fail("unexpected second inspection"),
    )
    result = probe.engine._plan_request(probe.request)
    assert result.plan_success.tolist() == [True]
    assert result.joint_trajectory is probe.first.joint_trajectory
    probe.engine.initial_context.assert_not_called()


@pytest.mark.parametrize(
    "exemption",
    [
        "shared_action",
        "explicit",
        "fixed",
        "upright",
        "downstream",
        "unmarked",
        "batch",
        "collision",
        "motion_gen",
    ],
)
def test_pick_transform_keeps_other_owner_contracts_unchanged(
    monkeypatch, pick_transform_inputs, exemption
):
    from embodichain.gen_sim.task_engine._task_program import pick_safety
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp

    probe = pick_transform_inputs
    request = probe.request
    if exemption == "shared_action":
        probe.engine._actions["pick_up"] = PickUp()
    elif exemption == "explicit":
        request.goal = replace(request.goal, grasp_xpos=torch.eye(4))
    elif exemption == "unmarked":
        request.goal.semantics.affordance.custom_config.clear()
    elif exemption == "batch":
        probe.context.batch_size = 2
    elif exemption in {"collision", "motion_gen"}:
        request.motion_policy = replace(
            request.motion_policy,
            **(
                {"dynamic_collision_mode": "required"}
                if exemption == "collision"
                else {"strategy": "motion_gen"}
            ),
        )
    else:
        field, value = {
            "fixed": ("fixed_object_to_eef", torch.eye(4)),
            "upright": ("rotate_upright", 0.0),
            "downstream": ("downstream_object_target_poses", (torch.eye(4),)),
        }[exemption]
        request.skill_options = replace(request.skill_options, **{field: value})
    monkeypatch.setattr(
        pick_safety,
        "_inspect_pick_plan",
        lambda *args: pytest.fail("owner exemption was ignored"),
    )
    result = probe.engine._plan_request(
        request, probe.context, plan_transform=lambda *args: probe.first
    )
    assert result.plan_success.tolist() == [True]
    assert result.joint_trajectory is probe.first.joint_trajectory
