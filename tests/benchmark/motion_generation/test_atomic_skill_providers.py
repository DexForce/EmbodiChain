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

"""Tests for the OpenDoor, Pour, AxisAlign and PushObject benchmark providers."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    AxisAlign,
    JointPositionGoal,
    MotionPolicy,
    OpenDoor,
    OpenDoorAffordance,
)
from embodichain.lab.sim.motion.planners import MoveType, ToppraPlanOptions
from embodichain.lab.sim.motion.planners.utils import TrajectorySampleMethod
from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv
from scripts.benchmark.motion_generation.config import PlannerSpecCfg, load_suite
from scripts.benchmark.motion_generation.models import (
    AlgorithmRole,
    BenchmarkCase,
    CaseOutcome,
)
from scripts.benchmark.motion_generation.planners.base import PlannerContext
from scripts.benchmark.motion_generation.planners.toppra import ToppraAdapter
from scripts.benchmark.motion_generation.scenarios import (
    atomic_skill_providers as providers,
)
from scripts.benchmark.motion_generation.scenarios.atomic_task import (
    AtomicTaskScenario,
    _ActiveCaseGraspPoseGenerator,
    _ExecutionObservation,
    _signed_rotation_about_axis,
    _with_plan_options,
    create_atomic_skill_provider,
)

CPU = torch.device("cpu")


def _case(skill_id: str, batch_size: int, **parameters: object) -> BenchmarkCase:
    return BenchmarkCase(
        suite_version="test_v1",
        track="atomic-task",
        scenario_id=skill_id,
        case_id=f"atomic-task:{skill_id}:case:b{batch_size}:s11",
        seed=11,
        batch_size=batch_size,
        num_waypoints=1,
        path_shape="test",
        start_state_bin="pre_action",
        start_qpos=torch.zeros(batch_size, 7),
        target_waypoints=torch.eye(4).repeat(batch_size, 1, 1, 1),
        reference_qpos=torch.zeros(batch_size, 1, 7),
        robot_id="franka_pgi",
        skill_id=skill_id,
        object_id="cube",
        task_difficulty="simple",
        primary_success="task_success",
        full_start_qpos=torch.zeros(batch_size, 9),
        case_parameters=dict(parameters),
    )


def _outcomes(batch_size: int, *, motion_valid: bool = True) -> tuple[CaseOutcome, ...]:
    return tuple(
        CaseOutcome(
            env_index=index,
            planning_success=True,
            finite=True,
            ordered_waypoints_reached=motion_valid,
            motion_valid=motion_valid,
            completed_waypoint_ratio=1.0,
            final_translation_err_mm=0.0,
            final_rotation_err_deg=0.0,
            waypoint_translation_err_mm_mean=0.0,
            waypoint_translation_err_mm_p95=0.0,
            waypoint_translation_err_mm_max=0.0,
            waypoint_rotation_err_deg_mean=0.0,
            waypoint_rotation_err_deg_p95=0.0,
            waypoint_rotation_err_deg_max=0.0,
            joint_limit_violation=False,
            max_normalized_joint_violation=0.0,
            joint_path_length_rad=0.0,
            cartesian_path_length_m=0.0,
            path_efficiency=1.0,
        )
        for index in range(batch_size)
    )


def _observation(batch_size: int, **values: object) -> _ExecutionObservation:
    return _ExecutionObservation(
        execution_success=torch.ones(batch_size, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(batch_size, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(batch_size),
        execution_time_ms=1.0,
        task_completion_time_s=0.1,
        **values,
    )


def _rotation_z(angle: float) -> torch.Tensor:
    return axis_angle_to_rotation_matrix(torch.tensor([0.0, 0.0, angle]))


def _door_affordance() -> OpenDoorAffordance:
    """A handle at local +X around a local +Z hinge through the origin."""
    return OpenDoorAffordance(
        mesh_vertices=torch.tensor(
            [[0.9, -0.1, 0.0], [1.1, -0.1, 0.0], [1.0, 0.2, 0.0]]
        ),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        rotation_axis=torch.tensor([0.0, 0.0, 1.0]),
        axis_origin=(0.0, 0.0, 0.0),
        joint_name="door_hinge",
        joint_limits=(0.0, math.pi / 2),
    )


def test_registry_exposes_the_four_added_skills() -> None:
    for skill_id in ("open_door", "pour", "axis_align", "push_object"):
        provider = create_atomic_skill_provider(skill_id)
        assert provider.skill_id == skill_id
        assert provider.requires_gripper


def test_door_geometry_matches_the_open_door_primitive() -> None:
    affordance = _door_affordance()
    link_pose = torch.eye(4).repeat(2, 1, 1)
    link_pose[1, :3, 3] = torch.tensor([0.2, -0.1, 0.3])
    grasp_pose = link_pose.clone()
    grasp_pose[:, :3, 3] += torch.tensor([1.0, 0.0, 0.0])
    rotation = torch.tensor([math.pi / 6, math.pi / 3])
    action = SimpleNamespace(device=CPU)

    expected_link, expected_eef = OpenDoor._opened_link_and_eef_poses(
        action,
        link_pose,
        grasp_pose,
        affordance.rotation_axis,
        (0.0, 0.0, 0.0),
        rotation,
        5,
    )
    opened_link, opened_eef = providers._door_opened_poses(
        link_pose, grasp_pose, affordance.rotation_axis, (0.0, 0.0, 0.0), rotation, 5
    )

    assert torch.allclose(opened_link, expected_link, atol=1.0e-6)
    assert torch.allclose(opened_eef, expected_eef, atol=1.0e-6)
    assert torch.allclose(
        providers._door_approach_direction_local(affordance),
        OpenDoor._approach_direction_local(affordance),
    )


def test_door_poses_cover_approach_grasp_every_arc_keyframe_and_retract() -> None:
    from embodichain.lab.sim.atomic_actions import OpenDoorOptions

    affordance = _door_affordance()
    link_pose = torch.eye(4).unsqueeze(0)
    grasp_pose = link_pose.clone()
    grasp_pose[:, 0, 3] = 1.0
    options = OpenDoorOptions(
        hand_interp_steps=3,
        door_waypoint_count=4,
        approach_distance=0.1,
        retract_distance=0.1,
    )

    poses = providers._door_poses(
        link_pose, grasp_pose, affordance, torch.tensor([math.pi / 6]), options
    )

    assert poses.shape == (1, 4 + 3, 4, 4)
    assert torch.allclose(poses[0, 0, :3, 3], torch.tensor([1.0, 0.1, 0.0]), atol=1e-6)
    assert torch.allclose(poses[0, 1], grasp_pose[0])
    angle = math.pi / 6
    assert torch.allclose(
        poses[0, -2, :3, 3],
        torch.tensor([math.cos(angle), math.sin(angle), 0.0]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        poses[0, -1, :3, 3],
        torch.tensor(
            [
                math.cos(angle) - 0.1 * math.sin(angle),
                math.sin(angle) + 0.1 * math.cos(angle),
                0.0,
            ]
        ),
        atol=1.0e-6,
    )
    screened = providers._door_screening_poses(poses)
    assert len(screened) == 5
    assert torch.equal(screened[-2], poses[:, -2])


def test_axis_align_geometry_matches_the_axis_align_primitive() -> None:
    object_pose = torch.eye(4).repeat(2, 1, 1)
    object_pose[:, :3, 3] = torch.tensor([-0.4, 0.0, 0.03])
    object_pose[1, :3, :3] = _rotation_z(0.3)
    internal = torch.tensor([1.0, 0.0, 0.0])
    action = SimpleNamespace(device=CPU, num_envs=2)
    action._axis_alignment_parameters = (
        lambda *args: AxisAlign._axis_alignment_parameters(action, *args)
    )

    for target in (torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 0.0])):
        expected = AxisAlign._axis_alignment_parameters(
            action, object_pose, internal, target
        )
        actual = providers._axis_alignment_parameters(object_pose, internal, target)
        for left, right in zip(actual, expected, strict=True):
            assert torch.allclose(left, right, atol=1.0e-6)

    grasp = object_pose.clone()
    grasp[:, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    reference = grasp.clone()
    reference[:, :3, :3] = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    assert torch.allclose(
        providers._symmetric_nearest(grasp, reference),
        AxisAlign._find_symmetric_nearest_xpos(action, grasp, reference),
    )

    upright = torch.tensor([0.0, 0.0, 1.0])
    _, _, rotation_axis, _ = providers._axis_alignment_parameters(
        object_pose, internal, upright
    )
    expected_grasp = AxisAlign._apply_upright_grasp_pre_rotation(
        SimpleNamespace(_UPRIGHT_GRASP_PRE_ROTATION=math.pi / 4),
        grasp,
        rotation_axis,
        torch.ones(2, dtype=torch.bool),
    )
    poses = providers._axis_align_poses(
        object_pose,
        grasp,
        torch.tensor([0.0, 0.0, -1.0]),
        pre_grasp_distance=0.15,
        lift_height=0.16,
        internal_axis=internal,
        target_axis=upright,
    )
    assert torch.allclose(poses[:, 1], expected_grasp, atol=1.0e-6)
    object_to_eef = torch.bmm(pose_inv(object_pose), expected_grasp)
    lifted_object = torch.bmm(poses[:, 2], pose_inv(object_to_eef))
    expected_aligned = AxisAlign._axis_alignment_eef_keyframes(
        action, lifted_object, object_to_eef, internal, upright, waypoint_count=1
    )[:, -1]
    assert torch.allclose(poses[:, 3], expected_aligned, atol=1.0e-5)
    aligned_object = torch.bmm(poses[:, 3], pose_inv(object_to_eef))
    assert torch.allclose(
        aligned_object[:, :3, :3] @ internal, upright.expand(2, -1), atol=1.0e-5
    )


def test_pour_poses_rotate_the_held_object_about_its_axis_and_return() -> None:
    object_pose = torch.eye(4).unsqueeze(0)
    object_pose[:, :3, :3] = _rotation_z(0.4)
    grasp = object_pose.clone()
    grasp[:, :3, 3] += torch.tensor([0.0, 0.0, 0.02])
    internal = torch.tensor([1.0, 0.0, 0.0])

    poses = providers._pour_poses(
        object_pose,
        grasp,
        torch.tensor([0.0, 0.0, -1.0]),
        pre_grasp_distance=0.15,
        lift_height=0.16,
        internal_axis=internal,
        rotate_angle=-0.7,
    )

    assert poses.shape == (1, 5, 4, 4)
    assert torch.allclose(poses[:, 4], poses[:, 2])
    object_to_eef = torch.bmm(pose_inv(object_pose), grasp)
    lifted = torch.bmm(poses[:, 2], pose_inv(object_to_eef))
    poured = torch.bmm(poses[:, 3], pose_inv(object_to_eef))
    world_axis = lifted[0, :3, :3] @ internal
    assert torch.allclose(poured[0, :3, :3] @ internal, world_axis, atol=1.0e-6)
    assert _signed_rotation_about_axis(
        lifted[:, :3, :3], poured[:, :3, :3], world_axis.unsqueeze(0)
    ).item() == pytest.approx(-0.7, abs=1.0e-5)
    assert torch.allclose(poured[:, :3, 3], lifted[:, :3, 3], atol=1.0e-6)


def test_push_poses_place_the_tcp_behind_the_object_and_push_to_target() -> None:
    object_pose = torch.eye(4).repeat(2, 1, 1)
    object_pose[:, 0, 3] = torch.tensor([0.4, 0.5])
    target_pose = object_pose.clone()
    target_pose[:, 1, 3] += torch.tensor([0.12, -0.12])

    poses = providers._push_poses(
        object_pose,
        target_pose,
        contact_distance=0.03,
        approach_height=0.1,
        retract_height=0.08,
        push_overshoot=0.0,
        contact_frame_to_eef=torch.eye(4),
        object_contact_offset=torch.tensor([0.0, 0.0, -0.01]),
    )

    direction = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    contact = object_pose[:, :3, 3] - direction * 0.03 + torch.tensor([0.0, 0.0, -0.01])
    assert torch.allclose(poses[:, 0, :3, 3], contact + torch.tensor([0.0, 0.0, 0.1]))
    assert torch.allclose(poses[:, 1, :3, 3], contact)
    assert torch.allclose(poses[:, 2, :3, 3], contact + direction * 0.12)
    assert torch.allclose(
        poses[:, 3, :3, 3], contact + direction * 0.12 + torch.tensor([0.0, 0.0, 0.08])
    )
    assert torch.allclose(
        poses[:, 1, :3, 2], torch.tensor([0.0, 0.0, -1.0]).expand(2, -1)
    )
    with pytest.raises(ValueError, match="non-zero in-plane displacement"):
        providers._push_poses(
            object_pose,
            object_pose,
            contact_distance=0.03,
            approach_height=0.1,
            retract_height=0.1,
            push_overshoot=0.0,
            contact_frame_to_eef=torch.eye(4),
            object_contact_offset=torch.zeros(3),
        )


def test_open_door_scores_the_hinge_at_the_end_of_the_opening_segment() -> None:
    provider = create_atomic_skill_provider("open_door")
    case = _case(
        "open_door",
        3,
        target_joint_position=1.0,
        hinge_angle_tolerance_rad=0.05,
    )
    observation = _observation(
        3,
        articulation_joint_effect_end=torch.tensor([0.98, 0.9, 1.0]),
        articulation_joint_peak_signed_delta=torch.tensor([1.0, 1.0, 1.0]),
    )
    outcomes = _outcomes(3)
    outcomes = (*outcomes[:2], replace(outcomes[2], motion_valid=False))

    success, failure_code = provider.task_result(
        Mock(), case, Mock(), observation, outcomes
    )

    assert success.tolist() == [True, False, False]
    assert failure_code == "articulation_effect_miss"
    missing, _ = provider.task_result(
        Mock(), case, Mock(), _observation(3), _outcomes(3)
    )
    assert missing.tolist() == [False, False, False]


def test_pour_requires_the_commanded_rotation_while_still_holding() -> None:
    provider = create_atomic_skill_provider("pour")
    case = _case(
        "pour",
        3,
        rotate_angle_rad=-0.8,
        pour_angle_tolerance_rad=0.05,
        minimum_object_lift_m=0.04,
    )
    observation = _observation(
        3,
        object_rotation_peak_signed_rad=torch.tensor([-0.78, -0.6, -0.8]),
        object_lift_delta_m=torch.tensor([0.16, 0.16, 0.01]),
    )

    success, failure_code = provider.task_result(
        Mock(), case, Mock(), observation, _outcomes(3)
    )

    assert success.tolist() == [True, False, False]
    assert failure_code == "object_rotation_miss"


def test_axis_align_requires_the_aligned_axis_while_still_holding() -> None:
    provider = create_atomic_skill_provider("axis_align")
    case = _case(
        "axis_align",
        3,
        internal_axis=[1.0, 0.0, 0.0],
        target_axis=[0.0, 0.0, 1.0],
        axis_angle_tolerance_rad=0.0873,
        minimum_object_lift_m=0.04,
    )
    upright = axis_angle_to_rotation_matrix(torch.tensor([0.0, -math.pi / 2, 0.0]))
    final_pose = torch.eye(4).repeat(3, 1, 1)
    final_pose[0, :3, :3] = upright
    final_pose[2, :3, :3] = upright
    observation = _observation(
        3,
        final_object_pose=final_pose,
        object_lift_delta_m=torch.tensor([0.16, 0.16, 0.0]),
    )

    success, failure_code = provider.task_result(
        Mock(), case, Mock(), observation, _outcomes(3)
    )

    assert torch.allclose(
        upright @ torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        atol=1e-6,
    )
    assert success.tolist() == [True, False, False]
    assert failure_code == "axis_alignment_miss"


def test_push_object_requires_the_measured_object_pose_at_the_target() -> None:
    provider = create_atomic_skill_provider("push_object")
    target = torch.eye(4).repeat(2, 1, 1)
    final_pose = target.clone()
    final_pose[1, 1, 3] += 0.02
    case = _case(
        "push_object",
        2,
        target_object_pose=target.tolist(),
        object_position_threshold_m=0.01,
        object_tilt_threshold_rad=0.0873,
    )
    scenario = AtomicTaskScenario()
    scenario.suite = Mock(protocol=Mock(rotation_threshold_rad=0.1))

    success, failure_code = provider.task_result(
        scenario,
        case,
        Mock(),
        _observation(2, final_object_pose=final_pose),
        _outcomes(2),
    )

    assert success.tolist() == [True, False]
    assert failure_code == "object_goal_miss"


def test_signed_rotation_about_axis_recovers_both_directions() -> None:
    reference = _rotation_z(0.2).unsqueeze(0).repeat(3, 1, 1)
    current = torch.stack(
        (_rotation_z(0.2 + 1.1), _rotation_z(0.2 - 2.5), _rotation_z(0.2))
    )
    axis = torch.tensor([0.0, 0.0, 1.0]).expand(3, -1)

    angle = _signed_rotation_about_axis(reference, current, axis)

    assert torch.allclose(angle, torch.tensor([1.1, -2.5, 0.0]), atol=1.0e-5)
    assert torch.allclose(
        _signed_rotation_about_axis(reference, current, -axis),
        torch.tensor([-1.1, 2.5, 0.0]),
        atol=1.0e-5,
    )


def test_active_case_grasp_generator_replays_the_frozen_target_local_grasp() -> None:
    fallback = Mock()
    generator = _ActiveCaseGraspPoseGenerator(fallback)
    target = torch.eye(4).unsqueeze(0)
    grasp = target.clone()
    grasp[:, :3, 3] = torch.tensor([0.1, 0.0, 0.0])
    moved_target = target.clone()
    moved_target[:, :3, :3] = _rotation_z(math.pi / 2)
    moved_target[:, :3, 3] = torch.tensor([1.0, 0.0, 0.0])

    generator.activate(target, grasp)
    rows = generator.get_valid_grasp_poses(
        mesh_vertices=torch.zeros(3, 3),
        mesh_triangles=torch.zeros(1, 3, dtype=torch.long),
        obj_poses=moved_target.repeat(2, 1, 1),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )
    success, best, cost = generator.get_best_grasp_poses(
        mesh_vertices=torch.zeros(3, 3),
        mesh_triangles=torch.zeros(1, 3, dtype=torch.long),
        obj_poses=moved_target,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )

    assert len(rows) == 2
    assert torch.allclose(
        rows[1][0][0, :3, 3], torch.tensor([1.0, 0.1, 0.0]), atol=1e-6
    )
    assert success.tolist() == [True] and cost.tolist() == [0.0]
    assert torch.allclose(best[0, :3, 3], torch.tensor([1.0, 0.1, 0.0]), atol=1e-6)
    fallback.get_valid_grasp_poses.assert_not_called()

    generator.deactivate()
    generator.get_valid_grasp_poses(
        mesh_vertices=torch.zeros(3, 3),
        mesh_triangles=torch.zeros(1, 3, dtype=torch.long),
        obj_poses=target,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )
    fallback.get_valid_grasp_poses.assert_called_once()
    with pytest.raises(RuntimeError, match="No frozen benchmark grasp"):
        _ActiveCaseGraspPoseGenerator().get_best_grasp_poses(
            mesh_vertices=torch.zeros(3, 3),
            mesh_triangles=torch.zeros(1, 3, dtype=torch.long),
            obj_poses=target,
            approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        )


def test_backend_plan_options_fill_only_unset_motion_generation_policies() -> None:
    options = ToppraPlanOptions(sample_interval=40)
    explicit = ToppraPlanOptions(sample_interval=10)

    def invocation(policy: MotionPolicy) -> ActionInvocation:
        return ActionInvocation(
            skill_id="move_joints",
            goal=JointPositionGoal(torch.zeros(1, 7)),
            binding=Mock(spec=ActionBinding),
            motion_policy=policy,
        )

    filled = _with_plan_options(
        invocation(MotionPolicy(strategy="motion_gen")), options
    )
    kept = _with_plan_options(
        invocation(MotionPolicy(strategy="motion_gen", plan_opts=explicit)), options
    )
    interpolated = _with_plan_options(invocation(MotionPolicy()), options)

    assert filled.motion_policy.plan_opts.sample_interval == 40
    assert kept.motion_policy.plan_opts.sample_interval == 10
    assert interpolated.motion_policy.plan_opts is None


def test_toppra_adapter_serves_atomic_skills_and_joint_cases() -> None:
    adapter = ToppraAdapter(
        PlannerSpecCfg(
            id="toppra",
            adapter="toppra",
            role=AlgorithmRole.DIAGNOSTIC_BASELINE.value,
            config={"velocity": 0.3, "acceleration": 0.6},
        ),
        PlannerContext(
            robot=Mock(), control_part="arm", device=CPU, sample_interval=40
        ),
    )
    adapter.motion_generator = Mock()
    case = replace(
        _case("move_joints", 1, motion_validity="ordered_joint_waypoints"),
        reference_qpos=torch.full((1, 1, 7), 0.25),
    )

    adapter.plan(case)

    targets, options = adapter.motion_generator.generate.call_args.args
    assert {"atomic_action", "joint_waypoint"} <= adapter.capabilities
    assert targets[0].move_type is MoveType.JOINT_MOVE
    assert torch.equal(targets[0].qpos, case.reference_qpos[:, 0])
    assert options.plan_opts.constraints == {"velocity": 0.3, "acceleration": 0.6}
    atomic = adapter.atomic_plan_options()
    assert atomic.sample_method is TrajectorySampleMethod.QUANTITY
    assert atomic.sample_interval == 40
    assert atomic.constraints == {"velocity": 0.3, "acceleration": 0.6}


def test_replay_measures_hinge_at_segment_end_and_peak_object_rotation() -> None:
    state = {"step": 0}
    scenario = AtomicTaskScenario()
    scenario.simulation = Mock()
    scenario.simulation.update.side_effect = lambda step: state.update(
        step=state["step"] + step
    )
    scenario.robot = Mock()
    scenario.robot.get_qpos.side_effect = lambda: torch.zeros(1, 7)
    scenario.track = Mock(
        config={
            "physics": {"steps_per_waypoint": 1, "hold_steps": 0, "hold_sim_steps": 1}
        }
    )
    entity = Mock()

    def object_pose(to_matrix: bool) -> torch.Tensor:
        pose = torch.eye(4).unsqueeze(0)
        pose[:, :3, :3] = _rotation_z(-0.1 * state["step"])
        return pose

    entity.get_local_pose.side_effect = object_pose
    scenario._objects = {"cube": SimpleNamespace(entity=entity)}
    compiled = SimpleNamespace(
        trajectory=SimpleNamespace(positions=torch.zeros(1, 5, 7)),
        plan_success=torch.tensor([True]),
    )
    provider = Mock()
    provider.lift_segment_start.return_value = None
    provider.object_rotation_segment.return_value = (1, 3)
    case = _case(
        "pour",
        1,
        object_rotation_axis=[0.0, 0.0, 1.0],
        object_rotation_sign=-1.0,
    )

    metrics = scenario._replay_physics(compiled, case, provider, collect_metrics=True)

    # The reference is read after waypoint 0; waypoints 1 and 2 add two steps.
    assert metrics.object_rotation_peak_signed_rad.item() == pytest.approx(
        -0.2, abs=1e-5
    )

    state["step"] = 0
    articulation = Mock()
    articulation.joint_qpos.side_effect = lambda name: torch.tensor(
        [0.1 * state["step"]]
    )
    scenario._objects = {}
    scenario._articulations = {"microwave": articulation}
    compiled.segment = Mock(return_value=SimpleNamespace(start=1, stop=3))
    provider.articulation_effect_segment.return_value = "open"
    door_case = replace(
        _case("open_door", 1, target_joint="door_hinge"), object_id="microwave"
    )

    metrics = scenario._replay_physics(
        compiled, door_case, provider, collect_metrics=True
    )

    compiled.segment.assert_called_once_with(0, "open")
    assert metrics.articulation_joint_effect_end.item() == pytest.approx(0.3)
    assert metrics.articulation_joint_final.item() == pytest.approx(0.5)


def test_door_pour_align_push_suite_declares_the_four_skills() -> None:
    suite = load_suite("atomic_franka_pgi_door_pour_align_push")
    track = suite.enabled_tracks()[0]

    assert [spec.id for spec in suite.planners if spec.enabled] == ["curobo", "toppra"]
    assert [skill["id"] for skill in track.config["skills"]] == [
        "open_door",
        "pour",
        "axis_align",
        "push_object",
    ]
    assert track.config["physics"]["steps_per_waypoint"] == 16
    assert suite.protocol.validation_samples == 512
    assert all(
        "fit_type" not in spec.config.get("auto_gen", {}) for spec in suite.planners
    )
