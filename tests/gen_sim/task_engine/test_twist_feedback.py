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
"""CPU contracts for explicit native E8 position-feedback ownership."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
from types import MappingProxyType
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import twist_feedback as helper
from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
    GenSimTwist,
    _candidate_table,
)
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    ActionPlan,
    AtomicAction,
    EndpointBinding,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    PlannerDiagnostics,
    RecoveryPolicy,
    ResolvedActionRequest,
    TrackingPolicy,
    TimedTrajectory,
    TrajectorySegment,
    TwistAffordance,
    TwistGoal,
    TwistOptions,
)
from embodichain.gen_sim.task_engine._task_program.twist_session import (
    DueObservationRequest,
    RevisionControllerStop,
)
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
)


class _Plan(NS):
    def snapshot(self):
        return deepcopy(self)

    def segment(self, name):
        return next(segment for segment in self.segments if segment.name == name)


@pytest.fixture
def case() -> NS:
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
                JointPositionTarget("left_eef", (4,)),
            ),
        ),
    )
    goal = TwistGoal(
        ObjectSemantics(
            entity_id="control::knob",
            geometry={"gen_sim_twist_measured_qpos": 0.0},
            affordance=TwistAffordance(
                grasp_position=(0.0, 0.0, 0.1),
                axis_origin=(0.0, 0.0, 0.0),
                twist_axis=torch.tensor([0.0, 0.0, 1.0]),
                joint_name="knob",
                joint_limits=(-2.0, 2.0),
            ),
        ),
        torch.eye(4),
    )
    request = ResolvedActionRequest(
        skill_id="twist",
        invocation_id="chunk0",
        goal=goal,
        binding=binding,
        motion_policy=MotionPolicy(sample_count=80),
        tracking_policy=TrackingPolicy.timed(settle_duration=0.2),
        recovery_policy=RecoveryPolicy(),
        skill_options=TwistOptions(),
    ).snapshot()
    route = NS(
        feedback_enabled=True,
        grip_extra_m=0.002,
        arm="left",
        binding=NS(
            object_id="control",
            joint="knob",
            link="rotor",
            parent="panel",
            grip_depth=0.0125,
            grip_width=0.1,
        ),
    )
    native = torch.zeros(1, 6)
    target = torch.full((1, 6), 0.01)
    native_pose = torch.eye(4)[None]
    native_pose[0, 2, 3] = 0.97

    def fk(*, qpos, link_names, qpos_joint_names):
        assert qpos_joint_names == robot.joint_names
        result = torch.eye(4).repeat(len(qpos), len(link_names), 1, 1)
        result[:, :, 0, 3] = qpos[:, 0, None]
        result[:, :, 1, 3] = qpos[:, 2, None]
        result[:, :, 2, 3] = 1
        return result

    robot = NS(
        cfg=NS(solver_cfg={"left_arm": NS(end_link_name="tcp", tcp=np.eye(4))}),
        joint_names=tuple(f"q{i}" for i in range(6)),
        num_instances=1,
        get_qpos=Mock(
            side_effect=lambda target=False: (
                target_q.clone() if target else native.clone()
            )
        ),
        get_link_pose=Mock(side_effect=lambda name, to_matrix: native_pose.clone()),
        get_local_pose=Mock(return_value=torch.eye(4)[None]),
        compute_fk=Mock(side_effect=fk),
    )
    target_q = target
    art = NS(
        joint_names=("knob",),
        get_qpos=Mock(return_value=torch.tensor([[0.25]])),
        get_link_pose=Mock(return_value=torch.eye(4)[None]),
    )
    simulation = NS(
        get_articulation=Mock(return_value=art), sim_config=NS(physics_dt=0.01)
    )
    sensor = NS(
        route=route,
        robot=robot,
        art=art,
        trace=[],
        finger_names=("left_pad", "right_pad"),
    )
    guard = NS(route=route, robot=robot, simulation=simulation, sensor=sensor)
    wall = [0.0]
    state = helper.TwistFeedbackState(
        route, simulation, robot, sensor, guard, wall_clock=lambda: wall[0]
    )
    context = NS(
        batch_size=1, env_ids=torch.tensor([0]), robot=NS(qpos=native, timestamp=1.0)
    )
    plan = _Plan(
        plan_success=torch.tensor([True]),
        segments=(
            NS(name="approach", start=0, stop=2),
            NS(name="reach", start=2, stop=4),
            NS(name="close", start=4, stop=6),
        ),
        joint_trajectory=NS(positions=torch.zeros(1, 6, 6)),
        diagnostics=NS(
            metadata={"gen_sim_twist_roll_candidate": 3, "nested": {"value": 7}}
        ),
    )
    action = NS(robot=robot)
    state.record_plan(action, request, context, plan)
    state._gaps = Mock(return_value=([0.01, 0.01], [0.001, 0.001]))
    result = NS(
        state=state,
        route=route,
        robot=robot,
        art=art,
        sensor=sensor,
        guard=guard,
        simulation=simulation,
        context=context,
        request=request,
        plan=plan,
        action=action,
        native=native,
        target=target_q,
        native_pose=native_pose,
        wall=wall,
    )
    samples(result)
    return result


def samples(case: NS, *, phase: str = "close", offset: float = 0.03) -> None:
    chunk = case.state.chunks["chunk0"]
    pose = (chunk.desired_standoff if phase == "reach" else chunk.desired).copy()
    pose[2, 3] -= offset
    case.state.samples.clear()
    for timestamp in (
        case.context.robot.timestamp - 0.02,
        case.context.robot.timestamp - 0.01,
        case.context.robot.timestamp,
    ):
        case.state.samples.append(
            dict(
                timestamp=timestamp,
                tcp=pose.copy(),
                valid=True,
                dropped=0,
                parent_contact=False,
                target_contact=False,
            )
        )


def due(*, phase="close", revision=0, generation=0, planned_at=0.0, timeout=30.0):
    return DueObservationRequest(
        skill_id="twist",
        invocation_id="chunk0",
        invocation_revision=revision,
        invocation_index=0,
        attempt_generation=generation,
        next_waypoint_index=4,
        segment_name=phase,
        env_mask=torch.tensor([True]),
        original_planned_at=planned_at,
        original_action_timeout=timeout,
    )


def frame(case: NS) -> RuntimeCommandFrame:
    return RuntimeCommandFrame(
        commands=tuple(
            EndpointCommand(
                target, JointPositionPayload(case.target[:, list(target.joint_ids)])
            )
            for target in (
                endpoint.require_target(JointPositionTarget)
                for endpoint in case.request.binding.endpoints
            )
        ),
        active_mask=torch.tensor([True]),
        env_ids=case.context.env_ids,
        hold_duration=torch.tensor([0.04]),
    )


def propose(case: NS, **values):
    return case.state.propose(case.context, due(**values), previous_command=frame(case))


def record_revision(case: NS, invocation: ActionInvocation) -> None:
    request = replace(case.request, revision=invocation.revision, goal=invocation.goal)
    case.state.record_plan(case.action, request, case.context, case.plan)


def test_controller_identity_matches_factory(case):
    assert case.state.controller_id == "gen_sim.twist.precontact"
    assert case.state.revision == "1"
    assert case.state.supported_skill_ids == ("twist",)


def test_proposal_and_staged_plan_do_not_commit_until_installed_generation(case):
    before = case.native.clone()
    result = propose(case)
    assert isinstance(result, ActionInvocation) and result.revision == 1
    assert result.goal.semantics.geometry["gen_sim_twist_continuation"] is True
    assert result.goal.semantics.geometry["gen_sim_twist_resume_phase"] == "close"
    assert result.goal.semantics.geometry["gen_sim_twist_measured_qpos"] == 0.25
    assert case.state.episode_corrections == 0
    assert not np.any(case.state.bias)
    reservation = case.state.pending["chunk0"]
    np.testing.assert_allclose(reservation.increment, [0, 0, 0.008], atol=0, rtol=0)
    torch.testing.assert_close(reservation.command_seed, case.target, atol=0, rtol=0)
    record_revision(case, result)
    assert case.state.episode_corrections == 0
    assert case.state.pending["chunk0"].planned_plan is not case.plan
    assert propose(case, phase="twist", revision=1, generation=1) is None
    np.testing.assert_array_equal(case.state.bias, reservation.bias)
    chunk = case.state.chunks["chunk0"]
    assert chunk.corrections == case.state.episode_corrections == 1
    assert chunk.path_length == pytest.approx(0.008)
    assert chunk.active_revision == 1 and chunk.active_generation == 1
    assert "chunk0" not in case.state.pending
    torch.testing.assert_close(case.native, before, atol=0, rtol=0)


@pytest.mark.parametrize("recorded,generation", [(False, 1), (True, 0)])
def test_unplanned_or_uninstalled_revision_fails_without_commit(
    case, recorded, generation
):
    result = propose(case)
    if recorded:
        record_revision(case, result)
    stopped = propose(case, phase="twist", revision=1, generation=generation)
    assert isinstance(stopped, RevisionControllerStop)
    assert case.state.episode_corrections == 0 and not np.any(case.state.bias)


def test_nominal_plan_and_metadata_are_owned_once(case):
    chunk = case.state.chunks["chunk0"]
    desired = chunk.desired.copy()
    case.plan.joint_trajectory.positions[:] = 42
    case.plan.diagnostics.metadata["nested"]["value"] = 99
    case.state.record_plan(case.action, case.request, case.context, case.plan)
    np.testing.assert_array_equal(chunk.desired, desired)
    assert chunk.selected_metadata["nested"]["value"] == 7
    assert not bool(chunk.plan.joint_trajectory.positions.any())


def test_record_real_framework_action_plan_owns_mappingproxy_diagnostics(case):
    request = replace(case.request, invocation_id="typed_chunk")
    trajectory = TimedTrajectory.from_uniform_step(
        case.plan.joint_trajectory.positions,
        env_ids=case.context.env_ids,
        step_dt=0.04,
    )
    commands = AtomicAction._joint_command_sequence(
        case.action, request, trajectory, active_mask=torch.tensor([True])
    )
    plan = ActionPlan(
        skill_id="twist",
        plan_success=torch.tensor([True]),
        commands=commands,
        recovery_policy=request.recovery_policy,
        tracking_policy=request.tracking_policy,
        planned_scene_version=0,
        planned_collision_world_revision=(0,),
        diagnostics=PlannerDiagnostics(
            backend="mock",
            metadata={"gen_sim_twist_roll_candidate": 3, "nested": {"value": 7}},
        ),
        joint_trajectory=trajectory,
        segments=(
            TrajectorySegment("approach", 0, 2),
            TrajectorySegment("reach", 2, 4),
            TrajectorySegment("close", 4, 6),
        ),
        invocation_id=request.invocation_id,
    )
    assert isinstance(plan.diagnostics.metadata, MappingProxyType)
    case.state.record_plan(case.action, request, case.context, plan)
    chunk = case.state.chunks[request.invocation_id]
    assert type(chunk.selected_metadata) is dict
    assert chunk.selected_metadata == dict(plan.diagnostics.metadata)
    assert chunk.plan is not plan
    plan.diagnostics.metadata["nested"]["value"] = 99
    plan.joint_trajectory.positions[:] = 42
    assert chunk.selected_metadata["nested"]["value"] == 7
    assert chunk.plan.diagnostics.metadata["nested"]["value"] == 7
    assert not bool(chunk.plan.joint_trajectory.positions.any())


def test_default_off_has_no_observations_commands_or_requests(case):
    case.state.enabled = False
    case.robot.get_qpos.reset_mock()
    case.robot.get_link_pose.reset_mock()
    case.state.capture_native()
    assert case.state.propose(None, None, previous_command=None) is None
    assert case.state.apply_bias(case.request, None) is case.request
    case.robot.get_qpos.assert_not_called()
    case.robot.get_link_pose.assert_not_called()


@pytest.mark.parametrize("field", ["route", "robot", "simulation", "sensor"])
def test_guard_instance_scope_cannot_be_substituted(case, field):
    setattr(case.guard, field, object())
    with pytest.raises(ValueError, match="exact route/robot/sensor"):
        helper.TwistFeedbackState(
            case.route, case.simulation, case.robot, case.sensor, case.guard
        )


def test_apply_bias_owns_only_command_goal_with_child_frame_transform(case):
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose = torch.eye(4)[None]
    pose[0, :3, :3] = rotation
    case.art.get_link_pose.return_value = pose
    case.state.bias[:] = [0.008, 0, 0]
    before = case.native.clone()
    original_point = case.request.goal.semantics.affordance.grasp_position
    result = case.state.apply_bias(case.request, case.context)
    assert result.goal.semantics.affordance.grasp_position == pytest.approx(
        (0, -0.008, 0.1)
    )
    assert case.request.goal.semantics.affordance.grasp_position == original_point
    assert result.goal.semantics.geometry == case.request.goal.semantics.geometry
    torch.testing.assert_close(case.native, before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "rotation",
    [
        np.eye(3),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
@pytest.mark.parametrize("explicit_point", [True, False])
def test_installed_bias_survives_every_normal_next_chunk_candidate_chain(
    case, rotation, explicit_point
):
    invocation = propose(case)
    record_revision(case, invocation)
    assert propose(case, phase="twist", revision=1, generation=1) is None
    assert case.state.episode_corrections == 1 and not case.state.pending
    np.testing.assert_array_equal(case.state.bias, [0, 0, 0.008])
    point = case.request.goal.semantics.affordance.grasp_position
    axis = (0.0, 0.0, 1.0)
    bite = min(0.008, case.route.binding.grip_depth / 2)
    outer = tuple(p - (bite - 0.020) * a for p, a in zip(point, axis))
    geometry = {
        **case.request.goal.semantics.geometry,
        "gen_sim_twist_outer_point": outer,
        "gen_sim_twist_axis": axis,
        "gen_sim_twist_bite_depth": bite,
        "gen_sim_twist_grip_points": [[0.01, 0.02, 0.03]],
        "gen_sim_twist_parent_collision_points": [[0.04, 0.05, 0.06]],
        "gen_sim_twist_target_to_parent": np.eye(4).tolist(),
    }
    if explicit_point:
        geometry["gen_sim_twist_grasp_point"] = point
    request = replace(
        case.request,
        invocation_id="next_chunk",
        revision=0,
        goal=replace(
            case.request.goal,
            semantics=replace(case.request.goal.semantics, geometry=geometry),
        ),
    ).snapshot()
    pose = torch.eye(4)[None]
    pose[0, :3, :3] = torch.from_numpy(rotation).float()
    case.art.get_link_pose.return_value = pose
    native_pose = pose.clone()
    native_qpos = case.native.clone()
    command_qpos = case.target.clone()
    before_geometry = deepcopy(dict(request.goal.semantics.geometry))
    case.art.set_qpos = Mock(side_effect=AssertionError("Native writes are forbidden"))
    case.robot.set_qpos = Mock(
        side_effect=AssertionError("Native writes are forbidden")
    )
    biased = case.state.apply_bias(request, case.context)
    repeated = case.state.apply_bias(request, case.context)
    assert (
        repeated.goal.semantics.affordance.grasp_position
        == biased.goal.semantics.affordance.grasp_position
    )
    assert request.goal.semantics.affordance.grasp_position == point
    assert dict(request.goal.semantics.geometry) == before_geometry
    for key in geometry:
        if key != "gen_sim_twist_grasp_point":
            assert biased.goal.semantics.geometry[key] == before_geometry[key]
    for _, roll, shift in _candidate_table(bite, fixed_roll=None):
        original = GenSimTwist._candidate_request(request, roll, shift)
        candidate = GenSimTwist._candidate_request(biased, roll, shift)
        delta = np.asarray(
            candidate.goal.semantics.affordance.grasp_position
        ) - np.asarray(original.goal.semantics.affordance.grasp_position)
        np.testing.assert_allclose(rotation @ delta, case.state.bias, atol=1e-8, rtol=0)
        assert (
            candidate.goal.semantics.affordance.grasp_roll
            == original.goal.semantics.affordance.grasp_roll
        )
        assert candidate.goal.semantics.affordance.grasp_roll_range is None
        assert candidate.goal.semantics.geometry["gen_sim_twist_bite_depth"] == bite
    torch.testing.assert_close(pose, native_pose, atol=0, rtol=0)
    torch.testing.assert_close(case.native, native_qpos, atol=0, rtol=0)
    torch.testing.assert_close(case.target, command_qpos, atol=0, rtol=0)
    case.art.set_qpos.assert_not_called()
    case.robot.set_qpos.assert_not_called()


def test_disabled_bias_preserves_candidate_request_identity_geometry_and_commands(case):
    geometry = {
        "gen_sim_twist_outer_point": (0.0, 0.0, 0.03),
        "gen_sim_twist_grasp_point": (0.0, 0.0, 0.02),
        "gen_sim_twist_axis": (0.0, 0.0, 1.0),
    }
    request = replace(
        case.request,
        goal=replace(
            case.request.goal,
            semantics=replace(case.request.goal.semantics, geometry=geometry),
        ),
    ).snapshot()
    case.state.enabled = False
    case.state.bias[:] = [0, 0, 0.008]
    case.art.get_link_pose.reset_mock()
    result = case.state.apply_bias(request, case.context)
    assert result is request
    for _, roll, shift in _candidate_table(0.00625, fixed_roll=None):
        expected = GenSimTwist._candidate_request(request, roll, shift)
        candidate = GenSimTwist._candidate_request(result, roll, shift)
        assert (
            candidate.goal.semantics.affordance.grasp_position
            == expected.goal.semantics.affordance.grasp_position
        )
        assert (
            candidate.goal.semantics.affordance.grasp_roll
            == expected.goal.semantics.affordance.grasp_roll
        )
    assert dict(result.goal.semantics.geometry) == dict(request.goal.semantics.geometry)
    case.art.get_link_pose.assert_not_called()


def test_seed_uses_owned_full_command_baseline_not_native_fill(case):
    case.native[:] = -0.02
    result = case.state._seed(case.state.chunks["chunk0"], case.context, frame(case))
    torch.testing.assert_close(result, case.target, atol=0, rtol=0)
    assert not torch.equal(result[:, [1, 3, 5]], case.native[:, [1, 3, 5]])


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing", "matching accepted SEND"),
        ("env", "matching accepted SEND"),
        ("mask", "matching accepted SEND"),
        ("address", "addresses differ"),
        ("payload", "readback"),
        ("untouched", "Unaddressed"),
        ("deviation", "native observation bound"),
        ("nonfinite", "native observation bound"),
    ],
)
def test_seed_invalid_send_or_native_difference_is_rejected(case, change, reason):
    command = frame(case)
    if change == "missing":
        command = None
    elif change == "env":
        command = replace(command, env_ids=torch.tensor([1]))
    elif change == "mask":
        command = replace(command, active_mask=torch.tensor([False]))
    elif change == "address":
        command = replace(command, commands=(command.commands[0],))
    elif change == "payload":
        case.target[:, 0] += 0.001
    elif change == "untouched":
        case.target[:, 1] += 0.001
    elif change == "deviation":
        case.native[:, 0] = -0.041
    else:
        case.native[:, 0] = torch.nan
    with pytest.raises(ValueError, match=reason):
        case.state._seed(case.state.chunks["chunk0"], case.context, command)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing", "invalid/incomplete"),
        ("drop", "invalid/incomplete"),
        ("invalid", "invalid/incomplete"),
        ("nonfinite", "invalid/incomplete"),
        ("stale", "stale"),
        ("unordered", "stale"),
        ("future", "stale"),
        ("parent", "parent"),
        ("gap", "parent clearance"),
        ("rotation", "stable/aligned"),
        ("spread", "stable/aligned"),
        ("target", "established target contact"),
    ],
)
def test_invalid_or_unsafe_native_window_never_reserves(case, change, reason):
    latest = case.state.samples[-1]
    if change == "missing":
        case.state.samples.popleft()
    elif change == "drop":
        latest["dropped"] = 1
    elif change == "invalid":
        latest["valid"] = False
    elif change == "nonfinite":
        latest["tcp"][0, 3] = np.nan
    elif change == "stale":
        case.context.robot.timestamp += 0.1
    elif change == "unordered":
        latest["timestamp"] = case.state.samples[-2]["timestamp"]
    elif change == "future":
        latest["timestamp"] += 0.1
    elif change == "parent":
        latest["parent_contact"] = True
    elif change == "gap":
        case.state._gaps.return_value = ([0.002, 0.01], [0.001, 0.001])
    elif change == "rotation":
        theta = np.deg2rad(4)
        latest["tcp"][:3, :3] = [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    elif change == "spread":
        latest["tcp"][0, 3] += 0.002
    else:
        latest["target_contact"] = True
    result = propose(case)
    assert isinstance(result, RevisionControllerStop) and reason in result.reason
    assert not case.state.pending and case.state.episode_corrections == 0


@pytest.mark.parametrize("change", ["angle", "spread", "both"])
def test_tracking_window_failure_records_exact_measurements_due_and_original_limits(
    case, change
):
    samples(case, phase="reach")
    latest = case.state.samples[-1]
    if change in {"angle", "both"}:
        theta = np.deg2rad(4)
        latest["tcp"][:3, :3] = [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    if change in {"spread", "both"}:
        latest["tcp"][0, 3] += 0.002
    observation = replace(due(phase="reach"), next_waypoint_index=2)
    result = case.state.propose(case.context, observation, previous_command=frame(case))
    assert isinstance(result, RevisionControllerStop)
    assert result.reason.startswith("Native E8 tracking window is not stable/aligned.")
    for field in (
        "angle_deg=",
        "spread_mm=",
        "position_error_mm=",
        "window_timestamps=",
        "due_phase=reach",
        "due_revision=0",
        "due_waypoint=2",
        "angle_limit_deg=3",
        "spread_limit_mm=0.5",
    ):
        assert field in result.reason
    audit = case.state.audit[-1]
    assert audit["decision"] == "stopped"
    assert audit["reason"] == result.reason
    assert audit["code"] == "native_tracking_window_not_stable_aligned"
    assert audit["angle_deg"] == pytest.approx(4.0 if change != "spread" else 0.0)
    assert audit["spread_mm"] == pytest.approx(4 / 3 if change != "angle" else 0.0)
    expected_error = np.linalg.norm([0.002 if change != "angle" else 0, 0, 0.03]) * 1000
    assert audit["position_error_mm"] == pytest.approx(expected_error)
    assert audit["window_timestamps"] == [0.98, 0.99, 1.0]
    assert audit["invocation_id"] == "chunk0" and audit["attempt_generation"] == 0
    assert audit["due_phase"] == "reach" and audit["due_revision"] == 0
    assert audit["due_waypoint"] == 2
    assert audit["angle_limit_deg"] == pytest.approx(3.0)
    assert audit["spread_limit_mm"] == 0.5
    assert audit["alignment_accepted"] is False
    assert not case.state.pending and case.state.episode_corrections == 0


def test_successful_and_default_off_paths_do_not_add_failure_diagnostics(case):
    samples(case, offset=0.01)
    assert propose(case) is None
    assert case.state.audit[-1]["decision"] == "tracking_window"
    assert case.state.audit[-1]["alignment_accepted"] is True
    assert not any(row["decision"] == "stopped" for row in case.state.audit)
    case.state.enabled = False
    before = deepcopy(list(case.state.audit))
    assert case.state.propose(None, None, previous_command=None) is None
    assert list(case.state.audit) == before


def test_healthy_window_audit_does_not_change_native_commands_bias_or_seed(case):
    samples(case, offset=0.01)
    before = case.native.clone()
    target = case.target.clone()
    case.robot.get_qpos.reset_mock()
    case.robot.get_link_pose.reset_mock()
    assert propose(case) is None
    row = case.state.audit[-1]
    assert row["decision"] == "tracking_window" and row["alignment_accepted"] is True
    assert row["angle_deg"] == pytest.approx(0)
    assert row["spread_mm"] == pytest.approx(0)
    assert row["position_error_mm"] == pytest.approx(10)
    assert row["window_timestamps"] == [0.98, 0.99, 1.0]
    assert row["invocation_id"] == "chunk0" and row["attempt_generation"] == 0
    assert row["due_phase"] == "close" and row["due_revision"] == 0
    assert row["due_waypoint"] == 4
    assert not case.state.pending and not np.any(case.state.bias)
    assert case.state.episode_corrections == 0
    torch.testing.assert_close(case.native, before, atol=0, rtol=0)
    torch.testing.assert_close(case.target, target, atol=0, rtol=0)
    case.robot.get_qpos.assert_not_called()
    case.robot.get_link_pose.assert_not_called()


@pytest.mark.parametrize("kind", ["sized", "safe", "other_phase"])
def test_safe_original_or_noninspection_phase_is_not_forced_through_feedback(
    case, kind
):
    if kind == "sized":
        samples(case, offset=0.01)
    elif kind == "safe":
        case.state._gaps.return_value = ([0.01, 0.01], [0.003, 0.003])
    assert propose(case, phase="twist" if kind == "other_phase" else "close") is None
    assert not case.state.pending


def test_each_due_phase_inspected_once_and_close_continuation_skips_repeat_reach(case):
    revision = propose(case)
    assert propose(case) is None
    record_revision(case, revision)
    samples(case, phase="reach")
    assert propose(case, phase="reach", revision=1, generation=1) is None
    assert case.state.episode_corrections == 1
    assert not case.state.pending


@pytest.mark.parametrize("kind", ["logical", "wall", "identity", "nonfinite"])
def test_original_deadline_not_reset_by_revision(case, kind):
    result = propose(case)
    record_revision(case, result)
    options = dict(phase="twist", revision=1, generation=1)
    if kind == "logical":
        case.context.robot.timestamp = 30.001
    elif kind == "wall":
        case.wall[0] = 180.001
    elif kind == "identity":
        options["planned_at"] = 1.0
    else:
        case.context.robot.timestamp = np.nan
    stopped = propose(case, **options)
    assert isinstance(stopped, RevisionControllerStop)
    chunk = case.state.chunks["chunk0"]
    assert chunk.original_planned_at == 0 and chunk.original_timeout == 30


@pytest.mark.parametrize("kind", ["chunk", "episode", "path"])
def test_correction_budget_is_not_relaxed(case, kind):
    chunk = case.state.chunks["chunk0"]
    chunk.repair_required = True
    if kind == "chunk":
        chunk.corrections = 3
    elif kind == "episode":
        case.state.episode_corrections = 12
    else:
        chunk.path_length = 0.029
    assert isinstance(propose(case), RevisionControllerStop)
    assert not case.state.pending


@pytest.mark.parametrize(
    "field,value",
    [
        ("timestamp", np.nan),
        ("parent_contact", None),
        ("target_contact", None),
        ("dropped_contacts", None),
    ],
)
def test_capture_invalid_native_contact_fields_stop_without_sensor_update(
    case, field, value
):
    row = dict(
        timestamp=1.0,
        valid=True,
        dropped_contacts=0,
        parent_contact=False,
        target_contact=False,
    )
    row[field] = value
    case.sensor.trace.append(row)
    case.sensor.update = Mock(side_effect=AssertionError("No update allowed"))
    case.state.capture_native()
    assert case.state.sample_error
    assert isinstance(propose(case), RevisionControllerStop)
    case.sensor.update.assert_not_called()


def test_capture_reads_native_tcp_once_and_reset_epoch_clears_ledger(case):
    case.state.episode_corrections = 2
    case.state.bias[:] = 0.001
    case.sensor.trace.append(
        dict(
            timestamp=0.0,
            valid=True,
            dropped_contacts=0,
            parent_contact=False,
            target_contact=False,
        )
    )
    case.robot.get_link_pose.reset_mock()
    case.state.capture_native()
    case.robot.get_link_pose.assert_called_once_with("tcp", to_matrix=True)
    assert case.state.episode_corrections == 0 and not np.any(case.state.bias)
    assert len(case.state.samples) == 1 and not case.state.chunks
    assert case.state.samples[0]["tcp"][2, 3] == pytest.approx(0.97)


def test_bounded_update_clips_bias_step_and_never_changes_inputs():
    bias = np.array([0.019, 0, 0])
    error = np.array([1.0, 0, 0])
    result, increment, path = helper._bounded_update(bias, error, 0, 0)
    assert np.linalg.norm(result) <= 0.020
    assert np.linalg.norm(increment) <= 0.008 and path == pytest.approx(0.001)
    np.testing.assert_array_equal(bias, [0.019, 0, 0])
    np.testing.assert_array_equal(error, [1, 0, 0])


def test_continuation_requires_owned_reservation_and_passes_separate_seed(
    case, monkeypatch
):
    from embodichain.gen_sim.task_engine._task_program import twist_continuation

    with pytest.raises(ValueError, match="owned staged"):
        case.state.plan_continuation(case.action, case.request, case.context)
    invocation = propose(case)
    request = replace(case.request, revision=1, goal=invocation.goal)
    callback = Mock(return_value=object())
    monkeypatch.setattr(twist_continuation, "plan_continuation", callback)
    result = case.state.plan_continuation(case.action, request, case.context)
    assert result is callback.return_value
    args = callback.call_args.args
    assert args[2] is case.context and args[3].resume_phase == "close"
    torch.testing.assert_close(
        callback.call_args.kwargs["command_seed"], case.target, atol=0, rtol=0
    )
    assert case.state.episode_corrections == 0


@pytest.mark.parametrize("revision,generation", [(1, 0), (0, -1)])
def test_first_due_cannot_claim_foreign_revision_or_invalid_generation(
    case, revision, generation
):
    if generation < 0:
        with pytest.raises(ValueError, match="attempt_generation"):
            propose(case, revision=revision, generation=generation)
        return
    assert isinstance(
        propose(case, revision=revision, generation=generation), RevisionControllerStop
    )
    assert not case.state.pending


def test_generation_cannot_move_backward_without_a_revision(case):
    assert propose(case, phase="twist", generation=2) is None
    assert isinstance(propose(case, generation=1), RevisionControllerStop)


def test_slow_finalized_revision_cannot_stage_or_send_past_original_wall_bound(case):
    invocation = propose(case)
    case.wall[0] = 180.001
    with pytest.raises(ValueError, match="wall deadline exhausted during planning"):
        record_revision(case, invocation)
    reservation = case.state.pending["chunk0"]
    assert reservation.planned_request is None and reservation.planned_plan is None
    assert case.state.episode_corrections == 0 and not np.any(case.state.bias)


def test_parent_contact_during_long_control_hold_is_not_lost_outside_pose_window(case):
    assert propose(case, phase="approach") is None
    for index in range(1, 28):
        case.sensor.trace.append(
            dict(
                timestamp=1.0 + index * 0.01,
                valid=True,
                dropped_contacts=0,
                parent_contact=index == 1,
                target_contact=False,
            )
        )
        case.state.capture_native()
    assert len(case.state.samples) == 12
    assert not any(sample["parent_contact"] for sample in case.state.samples)
    case.context.robot.timestamp = 1.27
    result = propose(case)
    assert (
        isinstance(result, RevisionControllerStop)
        and "touched the parent" in result.reason
    )
    assert not case.state.pending


def test_previous_chunk_parent_contact_does_not_leak_into_new_call_epoch(case):
    case.state.last_parent_contact_at = 10.0
    case.context.robot.timestamp = 20.0
    request = replace(case.request, invocation_id="chunk1")
    case.state.record_plan(case.action, request, case.context, case.plan)
    next_due = replace(due(phase="approach", planned_at=20.0), invocation_id="chunk1")
    assert (
        case.state.propose(case.context, next_due, previous_command=frame(case)) is None
    )
    assert case.state.chunks["chunk1"].last_contact_time == 20.0
    case.state.last_parent_contact_at = 20.01
    case.context.robot.timestamp = 20.20
    result = case.state.propose(case.context, next_due, previous_command=frame(case))
    assert (
        isinstance(result, RevisionControllerStop)
        and "touched the parent" in result.reason
    )


def test_actual_pad_gaps_apply_robot_root_and_parent_pose_exactly_once(case):
    del case.state._gaps
    local = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]])
    case.sensor.finger_collision_points = {"left_pad": local, "right_pad": local}
    case.sensor.geometry = NS(parent_collisions=(NS(vertices=local),))
    parent = torch.eye(4)[None]
    parent[0, 2, 3] = 0.5
    case.art.get_link_pose.return_value = parent
    root = torch.eye(4)[None]
    root[0, 2, 3] = 2
    case.robot.get_local_pose.return_value = root
    native, predicted = case.state._gaps(case.state.chunks["chunk0"], "close")
    assert native == pytest.approx([0.46, 0.46])
    assert predicted == pytest.approx([2.49, 2.49])
    assert case.robot.compute_fk.call_args.kwargs["link_names"] == [
        "left_pad",
        "right_pad",
    ]
    assert (
        case.robot.compute_fk.call_args.kwargs["qpos_joint_names"]
        == case.robot.joint_names
    )
    torch.testing.assert_close(
        case.native, torch.zeros_like(case.native), atol=0, rtol=0
    )


@pytest.mark.parametrize("kind", ["pads", "cloud", "parent"])
def test_actual_gap_requires_two_complete_pad_and_parent_collision_inputs(case, kind):
    del case.state._gaps
    case.sensor.finger_collision_points = {
        name: np.zeros((1, 3)) for name in case.sensor.finger_names
    }
    case.sensor.geometry = NS(parent_collisions=(NS(vertices=np.zeros((1, 3))),))
    if kind == "pads":
        case.sensor.finger_names = ("left_pad",)
    elif kind == "cloud":
        case.sensor.finger_collision_points["left_pad"] = np.empty((0, 3))
    else:
        case.sensor.geometry.parent_collisions = ()
    assert isinstance(propose(case), RevisionControllerStop)
    assert not case.state.pending


@pytest.mark.parametrize(
    "bias,error,count,path",
    [
        ([0.021, 0, 0], [0.001, 0, 0], 0, 0),
        ([0.020, 0, 0], [0.001, 0, 0], 0, 0),
        ([0, 0, 0], [0, 0, 0], 0, 0),
        ([0, 0, 0], [0.001, 0, 0], 3, 0),
        ([0, 0, 0], [0.001, 0, 0], 0, 0.030),
        ([0, 0, 0], [np.nan, 0, 0], 0, 0),
    ],
)
def test_bounded_update_exhausted_or_malformed_values_fail_closed(
    bias, error, count, path
):
    with pytest.raises(ValueError):
        helper._bounded_update(bias, error, count, path)
