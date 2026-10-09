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
"""CPU contracts for pure E8 continuation suffix construction."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import twist_continuation as helper
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    AtomicAction,
    ArticulationJointState,
    EndpointBinding,
    JointPositionCommand,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    ObservedArticulationJointState,
    OPEN_COMMAND,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionRequest,
    RobotObservation,
    SceneSnapshot,
    TaskState,
    TrackingPolicy,
    Twist,
    TwistAffordance,
    TwistGoal,
    TwistOptions,
)


@pytest.fixture
def case(monkeypatch: pytest.MonkeyPatch) -> NS:
    geometry = {
        "gen_sim_twist_angle": 0.2,
        "gen_sim_twist_measured_qpos": 0.0,
        "gen_sim_twist_target_qpos": 1.0,
        "gen_sim_twist_axis_sign": 1.0,
        "gen_sim_twist_chunk_angle": 0.1,
        "gen_sim_twist_settle_steps": 5,
        "gen_sim_twist_articulation_id": "control",
        "gen_sim_twist_joint": "knob",
    }
    goal = TwistGoal(
        ObjectSemantics(
            entity_id="control::knob",
            geometry=geometry,
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
                commands={OPEN_COMMAND: JointPositionCommand(torch.zeros(1))},
            ),
        ),
    )
    request = ResolvedActionRequest(
        skill_id="Twist",
        goal=goal,
        binding=binding,
        motion_policy=MotionPolicy(sample_count=80),
        tracking_policy=TrackingPolicy.timed(settle_duration=0.2),
        recovery_policy=RecoveryPolicy(),
        skill_options=TwistOptions(hand_interp_steps=4),
        revision=1,
    ).snapshot()
    native = torch.zeros(1, 6)
    seed = torch.tensor([[0.01, 0.02, 0.01, 0.02, 0.0, 0.02]])
    context = _native_context(native)
    state = context.scene.get_articulation_joint_state("control", "knob")
    planned = []

    def plan_pose(target, start, control, request, count, *, interpolation_dt):
        planned.append(
            (target.clone(), start.clone(), request, count, interpolation_dt)
        )
        path = start[:, None].expand(-1, count, -1).clone()
        path += torch.linspace(0, 0.01, count)[None, :, None]
        return torch.tensor([True]), path

    def build_plan(request, context, **values):
        return NS(request=request, context=context, **values)

    action = NS(
        _motion_segment_lengths=Twist._motion_segment_lengths,
        _require_twist_affordance=Twist._require_twist_affordance,
        _twisted_grasp_poses=Mock(return_value=torch.eye(4)[None, None]),
        _plan_pose_segment=plan_pose,
        build_plan=Mock(side_effect=build_plan),
        robot=object(),
        planning_services=NS(planner_name="mock"),
    )
    limits = Mock(return_value=torch.ones(1, 6))
    monkeypatch.setattr(helper, "_joint_velocity_limits", limits)
    closed = seed.clone()
    closed[:, 4] = 0.02
    desired = np.eye(4)
    desired[:3, 3] = [0.1, 0.2, 0.3]
    standoff = desired.copy()
    standoff[2, 3] += 0.05
    chunk = NS(
        resume_phase="reach",
        desired=desired,
        desired_standoff=standoff,
        closed_qpos=closed,
        hand_ids=(4,),
        selected_metadata={"gen_sim_twist_roll_candidate": 3, "nested": {"value": 7}},
    )
    return NS(
        request=request,
        context=context,
        action=action,
        chunk=chunk,
        seed=seed,
        bias=np.array([0.002, 0, 0]),
        planned=planned,
        state=state,
        limits=limits,
    )


def run(case: NS):
    return helper.plan_continuation(
        case.action,
        case.request,
        case.context,
        case.chunk,
        command_seed=case.seed,
        bias=case.bias,
    )


def _native_context(qpos, value=0.97, *, scene_timestamp=1.0, valid=True):
    joints = (
        {}
        if value is None
        else {
            ("control", "knob"): ObservedArticulationJointState(
                torch.tensor([[value]]), torch.tensor([valid])
            )
        }
    )
    return PlanningContext(
        robot=RobotObservation(timestamp=1.0, qpos=qpos, qvel=torch.zeros_like(qpos)),
        task=TaskState(batch_size=1, device=torch.device("cpu")),
        scene=SceneSnapshot(
            timestamp=scene_timestamp, version=1, articulation_joints=joints
        ),
        env_ids=torch.tensor([0]),
        control_dt=0.04,
    )


def test_real_planning_context_live_joint_survives_owned_due_copy_without_verified_state(
    case,
):
    original = _native_context(case.context.robot.qpos)
    case.context = replace(
        original,
        robot=replace(original.robot),
        task=replace(original.task),
        scene=replace(original.scene),
    )
    assert case.context.get_articulation_joint_state("control", "knob") is None
    expected = float(
        case.context.scene.get_articulation_joint_state("control", "knob").position[
            0, 0
        ]
    )
    result = run(case)
    assert result.diagnostics.metadata["continuation_measured_qpos"] == expected
    assert (
        result.diagnostics.metadata["continuation_angle_source"]
        == "scene_observed_joint"
    )
    assert result.context is case.context
    assert case.context.task.articulation_joints == {}


def test_simulation_registry_reads_native_joint_once_then_continuation_uses_owned_scene(
    case,
):
    from embodichain.lab.task_program.semantics import SceneRegistry

    measured = torch.tensor([[0.97]])
    articulation = NS(
        joint_names=("knob",),
        get_qpos=Mock(return_value=measured),
        get_local_pose=Mock(return_value=torch.eye(4)[None]),
    )
    simulation = NS(get_articulation=Mock(return_value=articulation))
    registry = SceneRegistry.from_simulation(
        simulation, articulations={"control": "native_uid"}
    )
    scene = registry.make_scene_provider(batch_size=1).snapshot(
        timestamp=1.0, env_ids=torch.tensor([0])
    )
    case.context = replace(case.context, scene=scene)
    result = run(case)
    articulation.get_qpos.assert_called_once_with(target=False)
    simulation.get_articulation.assert_called_once_with("native_uid")
    assert result.diagnostics.metadata["continuation_measured_qpos"] == float(
        measured[0, 0]
    )
    assert result.diagnostics.metadata["continuation_joint_observed_at"] == 1.0


@pytest.mark.parametrize("scene_timestamp", [0.96, 1.04, float("nan"), float("inf")])
def test_scene_clock_must_match_same_native_robot_observation(case, scene_timestamp):
    case.context = _native_context(
        case.context.robot.qpos, scene_timestamp=scene_timestamp
    )
    with pytest.raises(ValueError, match="same-tick"):
        run(case)
    assert case.planned == []


def test_verified_task_joint_and_geometry_measurement_cannot_replace_missing_scene_joint(
    case,
):
    empty = _native_context(case.context.robot.qpos, None)
    task = replace(
        empty.task,
        articulation_joints={
            ("control", "knob"): ArticulationJointState(torch.tensor([[0.98]]))
        },
    )
    case.context = replace(empty, task=task)
    assert case.context.get_articulation_joint_state("control", "knob") is not None
    with pytest.raises(ValueError, match="fresh scene joint"):
        run(case)


@pytest.mark.parametrize(
    "kind",
    [
        "invalid",
        "multiple_values",
        "outside_limits",
        "foreign_joint",
        "foreign_articulation",
        "source_joint_mismatch",
        "missing_limits",
    ],
)
def test_scene_joint_qualification_is_fail_closed(case, kind):
    if kind == "invalid":
        case.context = _native_context(case.context.robot.qpos, valid=False)
    elif kind == "multiple_values":
        scene = replace(
            case.context.scene,
            articulation_joints={
                ("control", "knob"): ObservedArticulationJointState(
                    torch.tensor([[0.97, 0.98]]), torch.tensor([True])
                )
            },
        )
        case.context = replace(case.context, scene=scene)
    elif kind == "outside_limits":
        case.context = _native_context(case.context.robot.qpos, 2.01)
    else:
        geometry = dict(case.request.goal.semantics.geometry)
        affordance = case.request.goal.semantics.affordance
        if kind == "foreign_joint":
            geometry["gen_sim_twist_joint"] = "other"
            affordance = replace(affordance, joint_name="other")
        elif kind == "foreign_articulation":
            geometry["gen_sim_twist_articulation_id"] = "other"
        elif kind == "source_joint_mismatch":
            geometry["gen_sim_twist_joint"] = "other"
        else:
            affordance = replace(affordance, joint_limits=None)
        case.request = replace(
            case.request,
            goal=replace(
                case.request.goal,
                semantics=replace(
                    case.request.goal.semantics,
                    geometry=geometry,
                    affordance=affordance,
                ),
            ),
        )
    with pytest.raises(ValueError):
        run(case)
    assert case.planned == []


def test_native_dtype_limit_roundoff_is_accepted_without_mutating_context(case):
    observed = float(torch.nextafter(torch.tensor(2.0), torch.tensor(float("inf"))))
    case.context = _native_context(case.context.robot.qpos, observed)
    result = run(case)
    assert result.diagnostics.metadata["continuation_measured_qpos"] == 2.0
    assert (
        float(
            case.context.scene.get_articulation_joint_state("control", "knob").position[
                0, 0
            ]
        )
        == observed
    )


@pytest.mark.parametrize(
    "phase,segments",
    [
        ("reach", ("approach", "reach", "close", "twist", "open", "retract")),
        ("close", ("reach", "close", "twist", "open", "retract")),
    ],
)
def test_complete_suffix_preserves_full_first_command_native_context_and_deadline(
    case, phase, segments
):
    case.chunk.resume_phase = phase
    before = case.context.robot.qpos.clone()
    seed = case.seed.clone()
    result = run(case)
    assert tuple(result.segment_lengths) == segments
    torch.testing.assert_close(result.trajectory.positions[:, 0], seed, atol=0, rtol=0)
    torch.testing.assert_close(case.context.robot.qpos, before, atol=0, rtol=0)
    torch.testing.assert_close(case.seed, seed, atol=0, rtol=0)
    for index in (1, 3, 5):
        torch.testing.assert_close(
            result.trajectory.positions[:, :, index],
            seed[:, index : index + 1].expand(-1, result.trajectory.waypoint_count),
            atol=0,
            rtol=0,
        )
    torch.testing.assert_close(
        result.trajectory.dt[:, 1:],
        torch.full_like(result.trajectory.dt[:, 1:], 0.04),
        atol=0,
        rtol=0,
    )
    assert result.trajectory.dt[0, 0] == 0
    assert result.request.recovery_policy == case.request.recovery_policy
    assert result.request.tracking_policy == case.request.tracking_policy
    assert result.request.motion_policy == case.request.motion_policy
    assert result.scene_dependency_end_segment == "reach"
    assert (
        result.segment_lengths["close"] == 9 and result.segment_lengths["twist"] == 23
    )
    assert result.diagnostics.metadata["gen_sim_twist_continuation"]
    assert (
        result.diagnostics.metadata["continuation_start_kind"]
        == "separate_last_issued_command_seed"
    )
    case.limits.assert_called_once_with(case.action.robot, None)


def test_revision_uses_fresh_native_joint_not_old_measurement_and_retains_it(case):
    result = run(case)
    expected = float(case.state.position[0])
    assert (
        result.request.goal.semantics.geometry["gen_sim_twist_measured_qpos"]
        == expected
    )
    assert result.diagnostics.metadata["continuation_measured_qpos"] == expected
    assert (
        result.diagnostics.metadata["continuation_angle_source"]
        == "scene_observed_joint"
    )
    assert result.request.skill_options.twist_angle == pytest.approx(1.0 - expected)
    assert case.action._twisted_grasp_poses.call_args.args[-2] == pytest.approx(
        1.0 - expected
    )
    assert case.request.goal.semantics.geometry["gen_sim_twist_measured_qpos"] == 0.0
    assert (
        case.request.skill_options.twist_angle
        != result.request.skill_options.twist_angle
    )


def test_public_command_lowering_preserves_first_bound_endpoint_payload(case):
    result = run(case)
    commands = AtomicAction._joint_command_sequence(
        case.action, result.request, result.trajectory, active_mask=torch.tensor([True])
    )
    assert commands.frame_count == result.trajectory.waypoint_count
    for command in commands.frames[0].commands:
        torch.testing.assert_close(
            command.payload.positions,
            case.seed[:, list(command.target.joint_ids)],
            atol=0,
            rtol=0,
        )


def test_issued_seed_values_are_not_cast_through_native_observation(case):
    case.seed = case.seed.double()
    case.seed[:, 1] = 0.020000000123
    result = run(case)
    torch.testing.assert_close(
        result.trajectory.positions[:, 0], case.seed, atol=0, rtol=0
    )


@pytest.mark.parametrize("value", [0.0, float("nan")])
def test_invalid_axis_sign_cannot_generate_a_suffix(case, value):
    goal = replace(
        case.request.goal,
        semantics=replace(
            case.request.goal.semantics,
            geometry={
                **case.request.goal.semantics.geometry,
                "gen_sim_twist_axis_sign": value,
            },
        ),
    )
    case.request = replace(case.request, goal=goal)
    with pytest.raises(ValueError, match="nonzero axis sign"):
        run(case)
    assert case.planned == []


@pytest.mark.parametrize("value", [None, float("nan")])
def test_no_fresh_native_joint_cannot_fall_back_to_old_precontact_measurement(
    case, value, monkeypatch
):
    if value is None:
        case.context = _native_context(case.context.robot.qpos, None)
    else:
        monkeypatch.setattr(
            SceneSnapshot,
            "get_articulation_joint_state",
            lambda *args: NS(
                position=torch.tensor([[value]]), valid_mask=torch.tensor([True])
            ),
        )
    with pytest.raises(ValueError, match="fresh scene joint|invalid/incomplete"):
        run(case)
    assert case.planned == []
    case.action.build_plan.assert_not_called()


def test_existing_nominal_pose_bias_and_closed_target_are_preserved(case):
    original = case.chunk.desired.copy()
    closed = case.chunk.closed_qpos.clone()
    metadata = case.chunk.selected_metadata.copy()
    result = run(case)
    np.testing.assert_array_equal(case.chunk.desired, original)
    torch.testing.assert_close(case.chunk.closed_qpos, closed, atol=0, rtol=0)
    np.testing.assert_allclose(
        case.planned[1][0][0, :3, 3].numpy(), original[:3, 3] + case.bias, atol=1e-7
    )
    starts = np.cumsum([0, *result.segment_lengths.values()])
    close_index = list(result.segment_lengths).index("close")
    torch.testing.assert_close(
        result.trajectory.positions[:, starts[close_index + 1] - 1, 4], closed[:, 4]
    )
    assert result.diagnostics.metadata["gen_sim_twist_roll_candidate"] == 3
    result.diagnostics.metadata["nested"]["value"] = 8
    assert metadata["nested"]["value"] == 7


@pytest.mark.parametrize("phase", ["grip", "twist", "retract"])
def test_non_precontact_phase_is_rejected(case, phase):
    case.chunk.resume_phase = phase
    with pytest.raises(ValueError, match="reach/close"):
        run(case)


@pytest.mark.parametrize("kind", ["far", "nan", "shape"])
def test_separate_command_seed_is_bounded_and_finite(case, kind):
    if kind == "far":
        case.seed[:, 1] = 0.05001
    elif kind == "nan":
        case.seed[:, 1] = float("nan")
    else:
        case.seed = torch.zeros(1, 5)
    with pytest.raises(ValueError, match="native observation bounds"):
        run(case)


@pytest.mark.parametrize("kind", ["bias", "revision", "batch", "endpoint", "closed"])
def test_invalid_continuation_scope_is_rejected(case, kind):
    if kind == "bias":
        case.bias = np.array([0.021, 0, 0])
    elif kind == "revision":
        case.request = replace(case.request, revision=0)
    elif kind == "batch":
        case.context = NS(batch_size=2)
    elif kind == "endpoint":
        case.chunk.hand_ids = (3,)
    else:
        case.chunk.closed_qpos = torch.full_like(case.seed, float("nan"))
    with pytest.raises(ValueError):
        run(case)
    case.action.build_plan.assert_not_called()


def test_remaining_turn_retains_original_chunk_cap_and_sign(case):
    case.context = _native_context(case.context.robot.qpos, 1.4)
    result = run(case)
    assert result.request.skill_options.twist_angle == pytest.approx(-0.1)


def test_original_velocity_guard_rejects_without_retiming(case, monkeypatch):
    guard = Mock(return_value=torch.tensor([False]))
    monkeypatch.setattr(helper, "_velocity_validity", guard)
    with pytest.raises(ValueError, match="joint velocity limits"):
        run(case)
    case.action.build_plan.assert_not_called()
    assert guard.call_args.args[0].shape[-1] == 6
    torch.testing.assert_close(
        guard.call_args.args[1][:, 1:],
        torch.full_like(guard.call_args.args[1][:, 1:], 0.04),
    )


@pytest.mark.parametrize("kind", ["jump", "shape", "ik"])
def test_suffix_planner_contract_is_fail_closed(case, kind):
    original = case.action._plan_pose_segment

    def fail(*args, **kwargs):
        success, arm = original(*args, **kwargs)
        if kind == "jump":
            arm[:, 0] += 0.01
        elif kind == "shape":
            arm = arm[:, :-1]
        else:
            success = torch.tensor([False])
        return success, arm

    case.action._plan_pose_segment = fail
    with pytest.raises(ValueError):
        run(case)
    case.action.build_plan.assert_not_called()
