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
"""Feedback-only endpoint holds without changing ordinary E8 trajectories."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import numpy as np
import torch

from embodichain.gen_sim.task_engine._task_program import motion, twist_runtime
from embodichain.gen_sim.task_engine._task_program.twist_runtime import GenSimTwist
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    AtomicAction,
    EndpointBinding,
    EntityState,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionRequest,
    RobotObservation,
    SceneEntityPose,
    SceneSnapshot,
    StateDelta,
    TaskState,
    TimedTrajectory,
    TrackingPolicy,
    Twist,
    TwistAffordance,
    TwistGoal,
    TwistOptions,
)
from embodichain.gen_sim.task_engine._task_program.twist_session import (
    DueObservationRequest,
)


@dataclass(frozen=True)
class _Semantics:
    geometry: dict


@dataclass(frozen=True)
class _Goal:
    semantics: _Semantics


@dataclass(frozen=True)
class _Request:
    goal: _Goal
    skill_options: TwistOptions


@dataclass(frozen=True)
class _Diagnostics:
    metadata: dict


@dataclass(frozen=True)
class _Plan:
    plan_success: torch.Tensor
    diagnostics: _Diagnostics | None
    joint_trajectory: TimedTrajectory
    segments: tuple
    expected_effects: StateDelta = StateDelta()
    scene_dependencies: tuple = ("control::knob",)
    scene_dependency_end_segment: str | None = None


def _segments(lengths):
    result, start = [], 0
    for name, length in lengths.items():
        result.append(NS(name=name, start=start, stop=start + length))
        start += length
    return tuple(result)


@pytest.fixture
def case(monkeypatch: pytest.MonkeyPatch):
    lengths = dict(approach=4, reach=5, close=3, twist=4, open=3, retract=5)
    count = sum(lengths.values())
    positions = torch.zeros(1, count, 3)
    positions[0, :, 0] = torch.arange(count) * 0.001
    positions[0, :, 1] = torch.arange(count) * 0.002
    positions[0, :, 2] = 0.42
    trajectory = TimedTrajectory.from_uniform_step(
        positions, env_ids=torch.tensor([0]), step_dt=0.04
    )
    plan = _Plan(torch.tensor([True]), _Diagnostics({}), trajectory, _segments(lengths))
    request = _Request(
        _Goal(
            _Semantics(
                {
                    "gen_sim_twist_angle": 0.1,
                    "gen_sim_twist_measured_qpos": 0.0,
                    "gen_sim_twist_target_qpos": 1.0,
                    "gen_sim_twist_axis_sign": 1.0,
                    "gen_sim_twist_chunk_angle": 0.1,
                    "gen_sim_twist_bite_depth": 0.006,
                    "gen_sim_twist_settle_steps": 2,
                }
            )
        ),
        TwistOptions(),
    )
    context = NS(
        robot=NS(qpos=torch.zeros(1, 3)),
        batch_size=1,
        env_ids=torch.tensor([0]),
        require_control_dt=lambda: 0.04,
    )
    events = []
    generated = []

    def public_plan(action, candidate, context):
        generated.append(candidate)
        return plan

    def score(action, value, request, context):
        index = value.diagnostics.metadata["gen_sim_twist_roll_candidate"]
        events.append(("score", index))
        return (0.0, 0.0, 0.0, float(index))

    def filter_candidate(action, value, request, context):
        events.append(("table", value.joint_trajectory.waypoint_count))
        if case.reject_final and value.joint_trajectory.waypoint_count > count:
            return replace(value, plan_success=torch.tensor([False]))
        return value

    def root_dependency(value, request, context):
        events.append(("alias", value.joint_trajectory.waypoint_count))
        return value, {"qualified": True}

    def validity(*args):
        events.append(("velocity", args[0].shape[1]))
        return torch.tensor([case.velocity_pass])

    def build_plan(
        request,
        context,
        *,
        success,
        trajectory,
        diagnostics=None,
        segment_lengths,
        expected_effects,
        **kwargs,
    ):
        return _Plan(
            success,
            diagnostics,
            trajectory,
            _segments(segment_lengths),
            expected_effects,
            scene_dependency_end_segment=kwargs.get("scene_dependency_end_segment"),
        )

    feedback = NS(
        enabled=True,
        apply_bias=Mock(side_effect=lambda request, context: request),
        record_plan=Mock(
            side_effect=lambda *args: events.append(
                ("record", args[-1].joint_trajectory.waypoint_count)
            )
        ),
        plan_continuation=Mock(),
    )
    guard = NS(
        filter_candidate=filter_candidate,
        root_dependency=root_dependency,
        filter_continuation=Mock(
            side_effect=lambda action, plan, request, context: plan
        ),
    )
    action = GenSimTwist(geometry_guard=guard, feedback_state=feedback)
    action._planning_services = NS(
        device=torch.device("cpu"),
        robot=NS(
            device=torch.device("cpu"), get_qpos=lambda **kwargs: context.robot.qpos
        ),
    )
    action.build_plan = Mock(side_effect=build_plan)
    action.failed_plan = Mock(
        return_value=replace(plan, plan_success=torch.tensor([False]))
    )
    monkeypatch.setattr(Twist, "_plan", public_plan)
    monkeypatch.setattr(
        GenSimTwist,
        "_candidate_request",
        staticmethod(lambda request, roll, shift: request),
    )
    monkeypatch.setattr(GenSimTwist, "_candidate_geometry_score", score)
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.ones(1, 3)
    )
    monkeypatch.setattr(motion, "_velocity_validity", validity)
    token = twist_runtime._CANDIDATE_STATE.set(twist_runtime._TwistCandidateState())
    case = NS(
        action=action,
        feedback=feedback,
        guard=guard,
        request=request,
        context=context,
        plan=plan,
        lengths=lengths,
        events=events,
        generated=generated,
        reject_final=False,
        velocity_pass=True,
    )
    yield case
    twist_runtime._CANDIDATE_STATE.reset(token)


def _expected(case, *, alignment):
    pieces = []
    settle = case.request.goal.semantics.geometry["gen_sim_twist_settle_steps"]
    for segment in case.plan.segments:
        piece = case.plan.joint_trajectory.positions[:, segment.start : segment.stop]
        hold = (
            12
            if alignment and segment.name in {"approach", "reach"}
            else settle if segment.name in {"close", "twist"} else 0
        )
        if hold:
            piece = torch.cat((piece, piece[:, -1:].expand(-1, hold, -1)), 1)
        pieces.append(piece)
    return torch.cat(pieces, 1)


def test_enabled_feedback_appends_exactly_twelve_endpoint_frames_to_both_free_segments(
    case,
):
    result = case.action._plan(case.request, case.context)
    expected = _expected(case, alignment=True)
    torch.testing.assert_close(
        result.joint_trajectory.positions, expected, atol=0, rtol=0
    )
    lengths = {
        segment.name: segment.stop - segment.start for segment in result.segments
    }
    assert lengths == dict(approach=16, reach=17, close=5, twist=6, open=3, retract=5)
    assert result.scene_dependency_end_segment == "reach"
    assert (
        result.joint_trajectory.waypoint_count
        == case.plan.joint_trajectory.waypoint_count + 28
    )
    assert torch.all(result.joint_trajectory.positions[:, :, 2] == 0.42)
    torch.testing.assert_close(
        result.joint_trajectory.dt[:, 1:],
        torch.full_like(result.joint_trajectory.dt[:, 1:], 0.04),
        atol=0,
        rtol=0,
    )
    assert result.joint_trajectory.dt[0, 0] == 0
    baseline = TimedTrajectory.from_uniform_step(
        _expected(case, alignment=False), env_ids=case.context.env_ids, step_dt=0.04
    )
    assert float(result.joint_trajectory.dt.sum() - baseline.dt.sum()) == pytest.approx(
        0.96, abs=1e-6
    )


@pytest.mark.parametrize("enabled", [None, False])
def test_none_or_disabled_feedback_keeps_existing_trajectory_and_guard_calls(
    case, enabled
):
    if enabled is None:
        case.action._feedback_state = None
    else:
        case.feedback.enabled = False
    result = case.action._plan(case.request, case.context)
    torch.testing.assert_close(
        result.joint_trajectory.positions,
        _expected(case, alignment=False),
        atol=0,
        rtol=0,
    )
    expected_timed = TimedTrajectory.from_uniform_step(
        _expected(case, alignment=False), env_ids=case.context.env_ids, step_dt=0.04
    )
    torch.testing.assert_close(
        result.joint_trajectory.dt, expected_timed.dt, atol=0, rtol=0
    )
    assert len([event for event in case.events if event[0] == "table"]) == 12
    expected_tail = [("velocity", 28), ("alias", 28)]
    if enabled is False:
        expected_tail.append(("record", 28))
    assert case.events[-len(expected_tail) :] == expected_tail


def test_alignment_holds_do_not_require_a_nonzero_contact_settle(case):
    geometry = {**case.request.goal.semantics.geometry, "gen_sim_twist_settle_steps": 0}
    case.request = replace(
        case.request, goal=replace(case.request.goal, semantics=_Semantics(geometry))
    )
    result = case.action._plan(case.request, case.context)
    torch.testing.assert_close(
        result.joint_trajectory.positions,
        _expected(case, alignment=True),
        atol=0,
        rtol=0,
    )
    assert result.joint_trajectory.waypoint_count == 48


def test_default_zero_settle_keeps_the_original_trajectory_object(case):
    geometry = {**case.request.goal.semantics.geometry, "gen_sim_twist_settle_steps": 0}
    case.request = replace(
        case.request, goal=replace(case.request.goal, semantics=_Semantics(geometry))
    )
    case.action._feedback_state = None
    result = case.action._plan(case.request, case.context)
    assert result.joint_trajectory is case.plan.joint_trajectory
    assert len([event for event in case.events if event[0] == "table"]) == 12


def test_final_full_geometry_filter_and_velocity_cannot_be_skipped_by_holds(case):
    result = case.action._plan(case.request, case.context)
    assert case.events[-4:] == [
        ("velocity", 52),
        ("table", 52),
        ("alias", 52),
        ("record", 52),
    ]
    assert result.diagnostics.metadata["gen_sim_twist_roll_candidate"] == 0
    assert len([event for event in case.events if event[0] == "score"]) == 12
    for index in range(12):
        assert case.events[2 * index : 2 * index + 2] == [
            ("table", 24),
            ("score", index),
        ]


def test_final_table_rejection_cannot_alias_or_record(case):
    case.reject_final = True
    result = case.action._plan(case.request, case.context)
    assert not result.plan_success.any()
    assert ("table", 52) in case.events
    assert not any(event[0] in {"alias", "record"} for event in case.events)


def test_velocity_rejection_cannot_alias_or_record(case):
    case.velocity_pass = False
    result = case.action._plan(case.request, case.context)
    assert not result.plan_success.any()
    assert not any(event[0] in {"alias", "record"} for event in case.events)


def test_failed_candidates_do_not_gain_alignment_holds(case):
    case.plan = replace(case.plan, plan_success=torch.tensor([False]))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Twist, "_plan", lambda *args: case.plan)
        result = case.action._plan(case.request, case.context)
    assert not result.plan_success.any()
    assert result.joint_trajectory is case.plan.joint_trajectory
    case.action.build_plan.assert_not_called()
    case.feedback.record_plan.assert_not_called()


def test_goal_complete_single_frame_hold_is_not_extended(case):
    twist_runtime._candidate_state().goal_complete = True
    result = case.action._plan(case.request, case.context)
    assert result.joint_trajectory.waypoint_count == 1
    assert len(result.segments) == 1 and result.segments[0].stop == 1
    assert not any(event[0] == "table" for event in case.events)
    case.feedback.record_plan.assert_called_once_with(
        case.action, case.request, case.context, result
    )


@pytest.mark.parametrize("enabled", [None, False])
def test_goal_complete_default_or_disabled_feedback_does_not_gain_recording(
    case, enabled
):
    if enabled is None:
        case.action._feedback_state = None
    else:
        case.feedback.enabled = False
    twist_runtime._candidate_state().goal_complete = True
    result = case.action._plan(case.request, case.context)
    assert result.joint_trajectory.waypoint_count == 1
    case.feedback.record_plan.assert_not_called()


def test_goal_complete_guard_rejection_is_not_recorded(case):
    case.guard.root_dependency = lambda plan, request, context: (
        replace(plan, plan_success=torch.tensor([False])),
        {"qualified": False},
    )
    twist_runtime._candidate_state().goal_complete = True
    result = case.action._plan(case.request, case.context)
    assert not result.plan_success.any()
    case.feedback.record_plan.assert_not_called()


def test_recorded_stationary_goal_actual_due_returns_none_without_changing_correction_ledger(
    case,
):
    from embodichain.gen_sim.task_engine._task_program.twist_feedback import (
        TwistFeedbackState,
    )

    robot = case.action.robot
    art = NS(get_link_pose=Mock(return_value=torch.eye(4)[None]))
    route = NS(
        feedback_enabled=True,
        grip_extra_m=0.0,
        binding=NS(object_id="control", link="knob"),
    )
    simulation = NS(get_articulation=Mock(return_value=art))
    sensor = NS(route=route, robot=robot, art=art)
    guard = case.guard
    guard.route, guard.simulation, guard.robot, guard.sensor = (
        route,
        simulation,
        robot,
        sensor,
    )
    feedback = TwistFeedbackState(
        route, simulation, robot, sensor, guard, wall_clock=lambda: 0.0
    )
    case.action._feedback_state = feedback
    case.action._planning_services.motion_generator = NS(planner=NS())
    case.action._planning_services.planner_name = "cpu"
    case.context = PlanningContext(
        robot=RobotObservation(
            timestamp=1.0, qpos=torch.zeros(1, 3), qvel=torch.zeros(1, 3)
        ),
        task=TaskState(batch_size=1, device=torch.device("cpu")),
        scene=SceneSnapshot(
            timestamp=1.0,
            version=0,
            entities={"control::knob": EntityState(torch.eye(4)[None])},
        ),
        env_ids=torch.tensor([0]),
        control_dt=0.04,
    )
    binding = ActionBinding(
        "robot",
        endpoints=(
            EndpointBinding(
                "primary",
                "motion",
                "arm",
                "robot",
                JointPositionTarget("left_arm", (0, 2)),
            ),
            EndpointBinding(
                "primary",
                "grasp",
                "hand",
                "robot",
                JointPositionTarget("left_eef", (1,)),
            ),
        ),
    )
    request = ResolvedActionRequest(
        skill_id="twist",
        goal=TwistGoal(
            ObjectSemantics(
                entity_id="control::knob",
                geometry=case.request.goal.semantics.geometry,
                affordance=TwistAffordance(
                    grasp_position=(0.0, 0.0, 0.0),
                    axis_origin=(0.0, 0.0, 0.0),
                    joint_name="knob",
                    joint_limits=(-2.0, 2.0),
                ),
            ),
            SceneEntityPose("control::knob"),
        ),
        binding=binding,
        motion_policy=MotionPolicy(strategy="ik_interp"),
        tracking_policy=TrackingPolicy.timed(settle_duration=0.2),
        recovery_policy=RecoveryPolicy(),
        skill_options=TwistOptions(),
        invocation_id="stationary",
        revision=0,
    )
    case.action.build_plan = lambda *args, **kwargs: AtomicAction.build_plan(
        case.action, *args, **kwargs
    )
    twist_runtime._candidate_state().goal_complete = True
    result = case.action._plan(request, case.context)
    assert (
        result.joint_trajectory.waypoint_count == 1 and result.commands.frame_count == 1
    )
    due = DueObservationRequest(
        skill_id="twist",
        invocation_id="stationary",
        invocation_revision=0,
        invocation_index=0,
        attempt_generation=0,
        next_waypoint_index=0,
        segment_name="approach",
        env_mask=torch.tensor([True]),
        original_planned_at=1.0,
        original_action_timeout=30.0,
    )
    assert feedback.propose(case.context, due, previous_command=None) is None
    chunk = feedback.chunks["stationary"]
    assert chunk.desired is None and chunk.closed_qpos is None
    assert (
        chunk.corrections == 0
        and chunk.path_length == 0
        and feedback.episode_corrections == 0
    )
    np.testing.assert_array_equal(feedback.bias, np.zeros(3))
    assert not feedback.pending and not feedback.audit and not feedback.samples
    assert not case.generated


def test_continuation_with_its_own_holds_is_not_extended_again(case):
    positions = _expected(case, alignment=True)
    already = replace(
        case.plan,
        diagnostics=_Diagnostics({"continuation_measured_qpos": 0.0}),
        joint_trajectory=TimedTrajectory.from_uniform_step(
            positions, env_ids=case.context.env_ids, step_dt=0.04
        ),
    )
    case.feedback.plan_continuation.return_value = already
    geometry = {
        **case.request.goal.semantics.geometry,
        "gen_sim_twist_continuation": True,
    }
    request = replace(
        case.request, goal=replace(case.request.goal, semantics=_Semantics(geometry))
    )
    result = case.action._plan(request, case.context)
    assert result.joint_trajectory is already.joint_trajectory
    case.action.build_plan.assert_not_called()
    case.guard.filter_continuation.assert_called_once()
    assert not any(event[0] == "table" for event in case.events)
