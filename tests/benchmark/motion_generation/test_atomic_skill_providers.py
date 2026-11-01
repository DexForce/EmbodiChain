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
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    find_symmetric_nearest_xpos,
)
from embodichain.lab.sim.motion.planners import MoveType, ToppraPlanOptions
from embodichain.lab.sim.motion.planners.utils import TrajectorySampleMethod
from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv
from embodichain_benchmarks.motion_generation.config import PlannerSpecCfg, load_suite
from embodichain_benchmarks.motion_generation.contracts import (
    ReplayObservationTrace,
    ReplayObservationWindow,
)
from embodichain_benchmarks.motion_generation.models import (
    AlgorithmRole,
    BenchmarkCase,
    CaseOutcome,
)
from embodichain_benchmarks.motion_generation.planners.base import PlannerContext
from embodichain_benchmarks.motion_generation.planners.toppra import ToppraAdapter
from embodichain_benchmarks.motion_generation.scenarios import (
    atomic_skill_providers as providers,
)
from embodichain_benchmarks.motion_generation.scenarios.atomic_skill import (
    AtomicSkillScenario,
)
from embodichain_benchmarks.motion_generation.scenarios.atomic_grasps import (
    ActiveCaseGraspPoseGenerator,
)
from embodichain_benchmarks.motion_generation.contracts import ExecutionObservation
from embodichain_benchmarks.motion_generation.scenarios.atomic_evaluation import (
    signed_rotation_about_axis,
)
from embodichain_benchmarks.motion_generation.scenarios.atomic_policy import (
    with_plan_options,
)
from embodichain_benchmarks.motion_generation.atomic_skill import (
    create_atomic_skill_provider,
)

CPU = torch.device("cpu")


def _case(skill_id: str, batch_size: int, **parameters: object) -> BenchmarkCase:
    return BenchmarkCase(
        suite_version="test_v1",
        track="atomic-skill",
        scenario_id=skill_id,
        case_id=f"atomic-skill:{skill_id}:case:b{batch_size}:s11",
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


def _observation(batch_size: int, **values: object) -> ExecutionObservation:
    return ExecutionObservation(
        execution_success=torch.ones(batch_size, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(batch_size, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(batch_size),
        execution_time_ms=1.0,
        task_completion_time_s=0.1,
        **values,
    )


def _rotation_z(angle: float) -> torch.Tensor:
    return axis_angle_to_rotation_matrix(torch.tensor([0.0, 0.0, angle]))


def test_open_door_physical_evaluator_uses_hinge_and_release_windows() -> None:
    case = _case("open_door", 2, target_joint_position=0.5)
    observation = _observation(
        2,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "door_actuation": ReplayObservationTrace(
                articulation_joint=torch.tensor([[0.0, 0.0], [0.5, 0.5], [0.5, 0.3]])
            ),
            "door_release": ReplayObservationTrace(
                gripper_qpos={"hand": torch.tensor([[[1.0], [1.0]], [[0.0], [0.0]]])}
            ),
        },
    )

    evaluator = create_atomic_skill_provider("open_door").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, _outcomes(2))

    assert [result.task_success for result in results] == [True, False]
    assert results[1].failure_stage == "opened"
    missing = evaluator.evaluate(case, _observation(2), _outcomes(2))
    assert all(result.failure_code == "evidence_unavailable" for result in missing)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_pour_physical_evaluator_uses_object_pose_interval(sign: float) -> None:
    initial = torch.eye(4).repeat(2, 1, 1)
    poses = torch.eye(4).repeat(3, 2, 1, 1)
    poses[:, :, 2, 3] = 0.1
    poses[1, :, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[0.0, 0.0, 0.5 * sign], [0.0, 0.0, 0.2 * sign]])
    )
    case = _case(
        "pour",
        2,
        object_initial_pose=initial.tolist(),
        object_rotation_axis=[0.0, 0.0, 1.0],
        object_rotation_sign=sign,
        rotate_angle_rad=0.5 * sign,
    )
    settle = poses[-1:].repeat(2, 1, 1, 1)
    observation = _observation(
        2,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "pour_actuation": ReplayObservationTrace(object_pose=poses),
            "pour_settle": ReplayObservationTrace(
                object_pose=settle,
                gripper_qpos={"hand": torch.full((2, 2, 1), 0.5)},
            ),
        },
    )

    evaluator = create_atomic_skill_provider("pour").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, _outcomes(2))

    assert [result.task_success for result in results] == [True, False]
    assert results[0].stages[0].measurements[
        "peak_signed_axis_angle_error_rad"
    ] == pytest.approx(0.0, abs=1.0e-6)


def test_pour_requires_terminal_lift_and_retention() -> None:
    initial = torch.eye(4).repeat(2, 1, 1)
    actuation = torch.eye(4).repeat(3, 2, 1, 1)
    actuation[:, :, 2, 3] = 0.1
    actuation[1, :, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    )
    settle = torch.eye(4).repeat(2, 2, 1, 1)
    settle[:, 1, 2, 3] = 0.1
    terminal_gripper = torch.full((2, 2, 1), 0.5)
    terminal_gripper[:, 1] = 1.0
    case = _case(
        "pour",
        2,
        object_initial_pose=initial.tolist(),
        object_rotation_axis=[0.0, 0.0, 1.0],
        object_rotation_sign=1.0,
        rotate_angle_rad=0.5,
    )
    observation = _observation(
        2,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "pour_actuation": ReplayObservationTrace(object_pose=actuation),
            "pour_settle": ReplayObservationTrace(
                object_pose=settle,
                gripper_qpos={"hand": terminal_gripper},
            ),
        },
    )

    evaluator = create_atomic_skill_provider("pour").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, _outcomes(2))

    assert [result.failure_stage for result in results] == ["settled", "held"]


def test_axis_align_physical_evaluator_keeps_terminal_retention_separate() -> None:
    initial = torch.eye(4).repeat(2, 1, 1)
    actuation = torch.eye(4).repeat(2, 2, 1, 1)
    actuation[:, :, 2, 3] = 0.1
    actuation[-1, :, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[0.0, 0.0, math.pi / 2], [0.0, 0.0, math.pi / 4]])
    )
    settle = actuation[-1:].clone().repeat(2, 1, 1, 1)
    case = _case(
        "axis_align",
        2,
        object_initial_pose=initial.tolist(),
        internal_axis=[1.0, 0.0, 0.0],
        target_axis=[0.0, 1.0, 0.0],
    )
    observation = _observation(
        2,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "axis_align_grasp": ReplayObservationTrace(
                gripper_qpos={"hand": torch.full((2, 2, 1), 0.5)},
            ),
            "axis_align_actuation": ReplayObservationTrace(object_pose=actuation),
            "axis_align_settle": ReplayObservationTrace(
                object_pose=settle,
                gripper_qpos={"hand": torch.full((2, 2, 1), 0.5)},
            ),
        },
    )

    evaluator = create_atomic_skill_provider("axis_align").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, _outcomes(2))

    assert [result.task_success for result in results] == [True, False]
    assert results[0].stages[-1].stage_id == "held"
    assert results[1].failure_stage == "aligned"


def test_axis_align_requires_lift_and_terminal_alignment() -> None:
    initial = torch.eye(4).repeat(2, 1, 1)
    actuation = torch.eye(4).repeat(2, 2, 1, 1)
    actuation[-1, :, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[0.0, 0.0, math.pi / 2], [0.0, 0.0, math.pi / 2]])
    )
    actuation[:, 1, 2, 3] = 0.1
    settle = actuation[-1:].repeat(2, 1, 1, 1)
    settle[:, 1, :3, :3] = torch.eye(3)
    case = _case(
        "axis_align",
        2,
        object_initial_pose=initial.tolist(),
        internal_axis=[1.0, 0.0, 0.0],
        target_axis=[0.0, 1.0, 0.0],
    )
    observation = _observation(
        2,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "axis_align_grasp": ReplayObservationTrace(
                gripper_qpos={"hand": torch.full((2, 2, 1), 0.5)},
            ),
            "axis_align_actuation": ReplayObservationTrace(object_pose=actuation),
            "axis_align_settle": ReplayObservationTrace(
                object_pose=settle,
                gripper_qpos={"hand": torch.full((2, 2, 1), 0.5)},
            ),
        },
    )

    evaluator = create_atomic_skill_provider("axis_align").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, _outcomes(2))

    assert [result.failure_stage for result in results] == ["aligned", "settled"]


def test_push_object_physical_evaluator_uses_contact_and_support_windows() -> None:
    initial = torch.eye(4).repeat(2, 1, 1)
    target = initial.clone()
    target[:, 1, 3] = 0.1
    contact = initial.unsqueeze(0).repeat(2, 1, 1, 1)
    contact[-1, 0, 1, 3] = 0.001
    actuation = initial.unsqueeze(0).repeat(2, 1, 1, 1)
    actuation[-1] = target
    settle = target.unsqueeze(0).repeat(2, 1, 1, 1)
    case = _case(
        "push_object",
        2,
        object_initial_pose=initial.tolist(),
        target_object_pose=target.tolist(),
        contact_motion_threshold_m=1.0e-5,
        support_plane_tolerance_m=0.01,
    )
    observation = _observation(
        2,
        replay_traces={
            "push_contact": ReplayObservationTrace(object_pose=contact),
            "push_actuation": ReplayObservationTrace(object_pose=actuation),
            "push_settle": ReplayObservationTrace(object_pose=settle),
        },
    )

    evaluator = create_atomic_skill_provider("push_object").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, _outcomes(2))

    assert [result.task_success for result in results] == [True, False]
    assert results[0].stages[0].evidence_source == (
        "measured_proxy:object_motion_during_contact"
    )
    assert results[1].failure_stage == "contacted"


def test_push_object_rechecks_in_plane_goal_after_settle() -> None:
    initial = torch.eye(4).reshape(1, 4, 4)
    target = initial.clone()
    target[:, 1, 3] = 0.1
    contact = initial.unsqueeze(0).repeat(2, 1, 1, 1)
    contact[-1, :, 1, 3] = 0.001
    actuation = torch.stack((initial, target))
    settle = target.unsqueeze(0).repeat(2, 1, 1, 1)
    settle[:, :, 0, 3] = 0.05
    case = _case(
        "push_object",
        1,
        object_initial_pose=initial.tolist(),
        target_object_pose=target.tolist(),
        contact_motion_threshold_m=1.0e-5,
        support_plane_tolerance_m=0.01,
    )
    observation = _observation(
        1,
        replay_traces={
            "push_contact": ReplayObservationTrace(object_pose=contact),
            "push_actuation": ReplayObservationTrace(object_pose=actuation),
            "push_settle": ReplayObservationTrace(object_pose=settle),
        },
    )

    evaluator = create_atomic_skill_provider("push_object").physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, _outcomes(1))[0]

    assert not result.task_success
    assert result.failure_stage == "on_support_plane"


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
    action._opened_link_and_eef_poses_at_fractions = lambda *args: (
        OpenDoor._opened_link_and_eef_poses_at_fractions(action, *args)
    )

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


@pytest.mark.parametrize("twist", [0.0, torch.tensor([0.4, -0.7])])
@pytest.mark.parametrize(
    "offset", [None, torch.tensor([[0.1, -0.02, 0.0], [-0.05, 0.03, 0.02]])]
)
def test_axis_align_geometry_matches_the_axis_align_primitive(twist, offset) -> None:
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
        providers.nearest_half_turn_pose(grasp, reference),
        find_symmetric_nearest_xpos(grasp, reference),
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
        alignment_twist_rad=twist,
        alignment_offset_m=offset,
    )
    assert torch.allclose(poses[:, 1], expected_grasp, atol=1.0e-6)
    object_to_eef = torch.bmm(pose_inv(object_pose), expected_grasp)
    lifted_object = torch.bmm(poses[:, 2], pose_inv(object_to_eef))
    expected_aligned = AxisAlign._axis_alignment_eef_keyframes(
        action,
        lifted_object,
        object_to_eef,
        internal,
        upright,
        waypoint_count=1,
        alignment_twist_rad=twist,
        alignment_offset_m=offset,
    )[:, -1]
    assert torch.allclose(poses[:, 3], expected_aligned, atol=1.0e-5)
    aligned_object = torch.bmm(poses[:, 3], pose_inv(object_to_eef))
    expected_position = lifted_object[:, :3, 3] + (0 if offset is None else offset)
    torch.testing.assert_close(aligned_object[:, :3, 3], expected_position)
    assert torch.allclose(
        aligned_object[:, :3, :3] @ internal, upright.expand(2, -1), atol=1.0e-5
    )


@pytest.mark.parametrize("early_filter", [False, True])
def test_axis_inward_rule_keeps_original_envs_and_smallest_reachable_offset(
    early_filter,
) -> None:
    """Two independent rows select literal minimal offsets in one shared batch."""
    objects = torch.eye(4).repeat(2, 1, 1)
    objects[:, :3, 3] = torch.tensor([[2.0, 1.0, 0.03], [-1.0, 3.0, 0.03]])
    base = torch.eye(4).repeat(2, 1, 1)
    base[:, :3, 3] = torch.tensor([[1.0, 1.0, 0.0], [-1.0, 2.0, 0.0]])
    grasps = objects.clone()
    grasps[:, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    calls = []

    def solve(*, pose, joint_seed, **kwargs):
        assert "env_ids" not in kwargs
        success = torch.ones(2, dtype=torch.bool)
        aligned = pose[:, 1, 2].abs() > 0.9
        distance = (pose[:, :2, 3] - objects[:, :2, 3]).norm(dim=-1)
        success &= ~aligned | (distance >= torch.tensor([0.024, 0.099]))
        if aligned.any():
            calls.append(distance.clone())
        return success, joint_seed + 0.01

    scenario = AtomicSkillScenario()
    scenario.robot = SimpleNamespace(
        device=CPU,
        get_control_part_base_pose=lambda **kwargs: base,
        compute_ik=solve,
    )
    scenario.control_part = "arm"
    sampler = Mock(return_value=[grasps[i : i + 1].clone() for i in range(2)])
    scenario._ranked_antipodal_grasp_pools = sampler

    def poses(obj, grasp, *, alignment_offset_m):
        return providers._axis_align_poses(
            obj,
            grasp,
            torch.tensor([0.0, 0.0, -1.0]),
            pre_grasp_distance=0.15,
            lift_height=0.16,
            internal_axis=torch.tensor([0.0, 0.0, 1.0]),
            target_axis=torch.tensor([0.0, 1.0, 0.0]),
            alignment_offset_m=alignment_offset_m,
        )

    chosen, offsets, evidence = providers._select_inward_axis_grasp(
        scenario,
        SimpleNamespace(object_id="cube"),
        objects,
        torch.tensor([0.0, 0.0, -1.0]),
        torch.zeros(2, 7),
        grasps,
        torch.tensor([[-1.0, 0.0, 0.0]]).repeat(2, 1),
        (0.0, 0.025, 0.1),
        grasp_parameters={
            "grasp_sample_count": 10000,
            "grasp_max_candidates": 128,
            "grasp_alignment_max_angle_deg": 10.0,
            "grasp_filter_before_ranking": early_filter,
        },
        build_poses=poses,
        seed=43001,
    )
    sampler.assert_called_once()
    assert sampler.call_args.kwargs["alignment_max_angle_deg"] == (
        10.0 if early_filter else None
    )
    torch.testing.assert_close(chosen, grasps)
    torch.testing.assert_close(
        offsets, torch.tensor([[-0.025, 0.0, 0.0], [0.0, -0.1, 0.0]])
    )
    assert [row["selected_inward_distance_m"] for row in evidence["rows"]] == [
        0.025,
        0.1,
    ]
    assert [len(row["rejected"]) for row in evidence["rows"]] == [1, 2]
    # Earlier accepted rows retain their own offset while another row searches.
    torch.testing.assert_close(calls[-1], torch.tensor([0.025, 0.1]))
    final = poses(objects, chosen, alignment_offset_m=offsets)[:, -1]
    aligned = final @ pose_inv(pose_inv(objects) @ chosen)
    torch.testing.assert_close(
        aligned[:, :3, 3], torch.tensor([[1.975, 1.0, 0.19], [-1.0, 2.9, 0.19]])
    )
    torch.testing.assert_close(
        aligned[:, :3, 2],
        torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1),
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("backend_error", [False, True])
def test_axis_inward_rule_retains_unknowns_and_propagates_backend_errors(
    backend_error,
) -> None:
    objects = torch.eye(4)[None]
    objects[:, 0, 3] = 1.0
    grasps = objects.clone()
    grasps[:, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    calls = []

    def solve(**kwargs):
        calls.append(kwargs["pose"])
        if backend_error:
            raise RuntimeError("broken backend")
        return torch.zeros(1, dtype=torch.bool), kwargs["joint_seed"]

    scenario = AtomicSkillScenario()
    scenario.robot = SimpleNamespace(
        device=CPU,
        get_control_part_base_pose=lambda **kwargs: torch.eye(4)[None],
        compute_ik=solve,
    )
    scenario.control_part = "arm"
    scenario._ranked_antipodal_grasp_pools = Mock(return_value=[grasps])
    expected = RuntimeError if backend_error else providers._AntipodalGraspSearchError
    match = "broken backend" if backend_error else "environment rows.*0.*0.0, 0.1"
    with pytest.raises(expected, match=match):
        providers._select_inward_axis_grasp(
            scenario,
            SimpleNamespace(object_id="cube"),
            objects,
            torch.tensor([0.0, 0.0, -1.0]),
            torch.zeros(1, 7),
            grasps,
            torch.tensor([[-1.0, 0.0, 0.0]]),
            (0.0, 0.1),
            grasp_parameters={
                "grasp_sample_count": 10000,
                "grasp_max_candidates": 128,
                "grasp_alignment_max_angle_deg": 10.0,
            },
            build_poses=lambda obj, grasp, **kwargs: grasp[:, None],
            seed=43001,
        )
    assert len(calls) == (1 if backend_error else 4)


@pytest.mark.parametrize("values", [[0.1], [0.0, 0.1, 0.1], [0.0, float("nan")]])
def test_axis_inward_rule_requires_an_explicit_ordered_bounded_search(values) -> None:
    with pytest.raises(ValueError, match="alignment_inward_candidates_m"):
        providers._axis_align_inward_distances(
            {"alignment_inward_candidates_m": values}
        )
    with pytest.raises(ValueError, match="Choose"):
        providers._axis_align_inward_distances(
            {
                "alignment_inward_candidates_m": [0.0, 0.1],
                "alignment_offset_m": [0, 0, 0],
            }
        )


def test_pour_approach_choices_preserve_elevation_budget_and_frozen_rows() -> None:
    objects = torch.eye(4).repeat(2, 1, 1)
    objects[:, 0, 3] = torch.tensor([0.3, 0.4])
    base = torch.tensor([-1.0, 0.0, -1.0]) / math.sqrt(2)
    direction90 = torch.tensor([0.0, -1.0, -1.0]) / math.sqrt(2)

    def make_grasp(direction, position):
        x = (
            torch.tensor([0.0, 1.0, 0.0])
            if abs(float(direction[0])) > 0.5
            else torch.tensor([1.0, 0.0, 0.0])
        )
        result = torch.eye(4)
        result[:3, :3] = torch.stack(
            (x, torch.linalg.cross(direction, x), direction), dim=1
        )
        result[:3, 3] = position
        return result

    first = make_grasp(base, objects[0, :3, 3])
    second = make_grasp(direction90, objects[1, :3, 3])
    samples = []

    def pools(handle, pose, directions, **kwargs):
        torch.testing.assert_close(pose, objects)
        samples.append((directions.clone(), kwargs["max_candidates"]))
        if len(samples) == 1:
            return [first[None], first.new_empty((0, 4, 4))]
        return [second.new_empty((0, 4, 4)), second[None]]

    scenario = AtomicSkillScenario()
    scenario.control_part = "arm"
    scenario.robot = SimpleNamespace(
        device=CPU,
        compute_ik=lambda **kw: (
            torch.ones(2, dtype=torch.bool),
            kw["joint_seed"] + 0.01,
        ),
    )
    scenario._ranked_antipodal_grasp_pools = pools

    def build(obj, grasp, *, approach_direction):
        return providers._pour_poses(
            obj,
            grasp,
            approach_direction,
            pre_grasp_distance=0.15,
            lift_height=0.16,
            internal_axis=torch.tensor([1.0, 0.0, 0.0]),
            rotate_angle=math.pi / 4,
        )

    grasp, directions, evidence = providers._select_pour_approach_grasp(
        scenario,
        SimpleNamespace(object_id="cube"),
        objects,
        base,
        torch.zeros(2, 7),
        ((0.0, 2), (math.pi / 2, 1)),
        grasp_parameters={
            "grasp_sample_count": 10000,
            "grasp_alignment_max_angle_deg": 10.0,
            "grasp_filter_before_ranking": True,
        },
        build_poses=build,
        seed=43001,
    )
    torch.testing.assert_close(grasp, torch.stack((first, second)))
    torch.testing.assert_close(
        directions, torch.stack((base, direction90)), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(directions[:, 2], torch.full((2,), -1 / math.sqrt(2)))
    assert [n for _, n in samples] == [2, 1] and evidence[
        "total_candidate_allocation"
    ] == 3
    torch.testing.assert_close(samples[1][0][0], base)
    assert [r["selected_yaw_offset_rad"] for r in evidence["rows"]] == [
        0.0,
        math.pi / 2,
    ]
    assert evidence["rows"][1]["rejected"][0]["available_candidates"] == 0
    targets = build(objects, grasp, approach_direction=directions)
    relative = pose_inv(objects) @ grasp
    poured = targets[:, 3] @ pose_inv(relative)
    returned = targets[:, 4] @ pose_inv(relative)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1 / math.sqrt(2), -1 / math.sqrt(2)],
            [0.0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
        ]
    )
    torch.testing.assert_close(
        poured[:, :3, :3], expected.expand(2, -1, -1), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        returned[:, :3, :3], torch.eye(3).expand(2, -1, -1), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize(
    "choices,total",
    [
        ([{"yaw_offset_rad": 0.0, "max_candidates": 4}], 3),
        (
            [
                {"yaw_offset_rad": 0.0, "max_candidates": 1},
                {"yaw_offset_rad": 0.0, "max_candidates": 1},
            ],
            3,
        ),
        ([{"yaw_offset_rad": math.pi / 2, "max_candidates": 1}], 3),
    ],
)
def test_pour_approach_choices_require_nominal_priority_and_total_budget(
    choices, total
) -> None:
    with pytest.raises(ValueError):
        providers._pour_approach_candidates(
            {"grasp_approach_candidates": choices, "grasp_max_candidates": total}
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
    assert signed_rotation_about_axis(
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


def test_signed_rotation_about_axis_recovers_both_directions() -> None:
    reference = _rotation_z(0.2).unsqueeze(0).repeat(3, 1, 1)
    current = torch.stack(
        (_rotation_z(0.2 + 1.1), _rotation_z(0.2 - 2.5), _rotation_z(0.2))
    )
    axis = torch.tensor([0.0, 0.0, 1.0]).expand(3, -1)

    angle = signed_rotation_about_axis(reference, current, axis)

    assert torch.allclose(angle, torch.tensor([1.1, -2.5, 0.0]), atol=1.0e-5)
    assert torch.allclose(
        signed_rotation_about_axis(reference, current, -axis),
        torch.tensor([-1.1, 2.5, 0.0]),
        atol=1.0e-5,
    )


def test_active_case_grasp_generator_replays_the_frozen_target_local_grasp() -> None:
    fallback = Mock()
    generator = ActiveCaseGraspPoseGenerator(fallback)
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
        ActiveCaseGraspPoseGenerator().get_best_grasp_poses(
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

    filled = with_plan_options(invocation(MotionPolicy(strategy="motion_gen")), options)
    kept = with_plan_options(
        invocation(
            MotionPolicy(
                strategy="motion_gen", plan_opts=explicit, acceleration_limit=0.7
            )
        ),
        options,
    )
    interpolated = with_plan_options(invocation(MotionPolicy()), options)

    assert filled.motion_policy.plan_opts.sample_interval == 40
    assert filled.motion_policy.acceleration_limit is None
    assert kept.motion_policy.plan_opts.sample_interval == 10
    assert kept.motion_policy.acceleration_limit == 0.7
    assert interpolated.motion_policy.plan_opts is None


def test_toppra_adapter_serves_atomic_skills_and_joint_cases() -> None:
    robot = Mock()
    robot.get_qvel_limits.return_value = torch.ones(1, 7)
    adapter = ToppraAdapter(
        PlannerSpecCfg(
            id="toppra",
            adapter="toppra",
            role=AlgorithmRole.DIAGNOSTIC_BASELINE.value,
            config={},
        ),
        PlannerContext(
            robot=robot,
            control_part="arm",
            device=CPU,
            sample_interval=40,
            joint_velocity_limit_rad_s=0.3,
            joint_acceleration_limit_rad_s2=0.6,
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
    assert options.plan_opts.constraints == {
        "velocity": [0.3] * 7,
        "acceleration": [0.6] * 7,
    }
    atomic = adapter.atomic_plan_options()
    on_clock = adapter.atomic_plan_options(control_dt=0.04)
    assert atomic.stop_at_waypoints is False
    assert on_clock.stop_at_waypoints is True
    assert on_clock.sample_method is TrajectorySampleMethod.TIME
    assert on_clock.sample_interval == 0.04
    assert on_clock.constraints == atomic.constraints
    with pytest.raises(ValueError, match="control_dt"):
        adapter.atomic_plan_options(control_dt=0.0)
    assert atomic.sample_method is TrajectorySampleMethod.QUANTITY
    assert atomic.sample_interval == 40
    assert atomic.constraints == {"velocity": [0.3] * 7, "acceleration": [0.6] * 7}


def test_replay_measures_hinge_at_segment_end_and_peak_object_rotation() -> None:
    state = {"step": 0}
    scenario = AtomicSkillScenario()
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
    provider.replay_observation_windows.return_value = (
        ReplayObservationWindow("pour_trace", 1, 3, frozenset({"object_pose"})),
    )
    case = _case(
        "pour",
        1,
        object_rotation_axis=[0.0, 0.0, 1.0],
        object_rotation_sign=-1.0,
    )

    metrics = scenario._replay_runtime().replay_physics(
        compiled, case, provider, collect_metrics=True
    )

    # The reference is read after waypoint 0; waypoints 1 and 2 add two steps.
    assert metrics.object_rotation_peak_signed_rad.item() == pytest.approx(
        -0.2, abs=1e-5
    )
    assert metrics.replay_traces["pour_trace"].object_pose.shape == (3, 1, 4, 4)

    state["step"] = 0
    articulation = Mock()
    articulation.joint_qpos.side_effect = lambda name: torch.tensor(
        [0.1 * state["step"]]
    )
    scenario._objects = {}
    scenario._articulations = {"microwave": articulation}
    compiled.segment = Mock(return_value=SimpleNamespace(start=1, stop=3))
    provider.articulation_effect_segment.return_value = "open"
    provider.replay_observation_windows.return_value = ()
    door_case = replace(
        _case("open_door", 1, target_joint="door_hinge"), object_id="microwave"
    )

    metrics = scenario._replay_runtime().replay_physics(
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


def test_axis_align_accepts_already_aligned_measured_grasp() -> None:
    """A correctly grasped/lifted object need not rotate to be aligned."""
    initial = torch.eye(4).reshape(1, 4, 4)
    lifted = initial.clone()
    lifted[:, 2, 3] = 0.1
    poses = lifted.unsqueeze(0).repeat(2, 1, 1, 1)
    holding = torch.full((2, 1, 1), 0.5)
    case = _case(
        "axis_align",
        1,
        object_initial_pose=initial.tolist(),
        internal_axis=[1.0, 0.0, 0.0],
        target_axis=[1.0, 0.0, 0.0],
    )
    observation = _observation(
        1,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "axis_align_grasp": ReplayObservationTrace(gripper_qpos={"hand": holding}),
            "axis_align_actuation": ReplayObservationTrace(object_pose=poses),
            "axis_align_settle": ReplayObservationTrace(
                object_pose=poses, gripper_qpos={"hand": holding}
            ),
        },
    )
    result = (
        create_atomic_skill_provider("axis_align")
        .physical_evaluator()
        .evaluate(case, observation, ())[0]
    )
    assert result.task_success
    assert result.stages[0].measurements["grasp_established"] is True


def test_push_succeeds_when_contact_starts_during_planar_motion() -> None:
    """The clearance/descent prefix can leave the object stationary."""
    initial = torch.eye(4).reshape(1, 4, 4)
    target = initial.clone()
    target[:, 1, 3] = 0.1
    case = _case(
        "push_object",
        1,
        object_initial_pose=initial.tolist(),
        target_object_pose=target.tolist(),
    )
    observation = _observation(
        1,
        replay_traces={
            "push_contact": ReplayObservationTrace(
                object_pose=torch.stack((initial, initial, target))
            ),
            "push_actuation": ReplayObservationTrace(
                object_pose=torch.stack((initial, target))
            ),
            "push_settle": ReplayObservationTrace(
                object_pose=torch.stack((target, target))
            ),
        },
    )
    result = (
        create_atomic_skill_provider("push_object")
        .physical_evaluator()
        .evaluate(case, observation, ())[0]
    )
    assert result.task_success


def test_custom_gripper_provider_initializes_configured_commands(monkeypatch):
    from embodichain_benchmarks.motion_generation import atomic_skill as registry
    from embodichain_benchmarks.motion_generation.config import SuiteCfg, TrackCfg
    from embodichain_benchmarks.motion_generation.scenarios.atomic_skill import (
        AtomicSkillScenario,
    )

    class Custom(registry.AtomicSkillCaseProvider):
        skill_id = "custom_gripper_regression"
        requires_gripper = True

        def generate_case(self, *args, **kwargs):
            raise NotImplementedError

        def build_invocation(self, *args, **kwargs):
            raise NotImplementedError

    monkeypatch.setitem(registry._ATOMIC_SKILL_PROVIDERS, Custom.skill_id, Custom)
    scenario = AtomicSkillScenario()
    simulation = Mock()
    robot = Mock()
    robot.get_qpos.return_value = torch.zeros(1, 9)
    scenario.simulation = simulation
    scenario._runtime_entities_created = True
    expected_open, expected_grasp = torch.zeros(1, 2), torch.ones(1, 2)
    resolve = Mock(return_value=("custom_hand", expected_open, expected_grasp))
    monkeypatch.setattr(scenario, "_resolve_gripper", resolve)
    track = TrackCfg(
        id="custom",
        scenario="atomic_skill",
        config={
            "skills": [{"id": Custom.skill_id}],
            "gripper": {"control_part": "custom_hand"},
        },
    )
    scenario.configure_runtime(simulation, robot, SuiteCfg(), track, "arm")
    resolve.assert_called_once()
    assert scenario.end_effector_part == "custom_hand"
    torch.testing.assert_close(
        scenario._configured_grippers()["custom_hand"][0], expected_open
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA unavailable"
                ),
            ],
        ),
    ],
)
def test_door_approach_uses_mesh_device_for_axis_and_origin(device):
    affordance = SimpleNamespace(
        mesh_vertices=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], device=device),
        rotation_axis=torch.tensor([0.0, 0.0, 1.0]),
        axis_origin=[0.0, 0.0, 0.0],
        opening_direction=1,
    )
    actual = providers._door_approach_direction_local(affordance)
    assert actual.device == torch.device(device)
    assert torch.allclose(actual, torch.tensor([0.0, -1.0, 0.0], device=device))


@pytest.mark.parametrize("early_filter", [False, True])
@pytest.mark.parametrize("skill_id,waypoint_count", [("pour", 5), ("axis_align", 4)])
def test_object_rotation_providers_screen_every_pose_row(
    monkeypatch, skill_id, waypoint_count, early_filter
):
    """Freeze independently feasible row grasps and match their reference chains."""
    from types import MethodType

    from embodichain_benchmarks.motion_generation import _atomic_fixtures

    from embodichain.lab.sim.objects.robot import Robot

    batch_size = 8
    object_pose = torch.eye(4).repeat(batch_size, 1, 1)
    object_pose[:, 0, 3] = torch.arange(batch_size) * 0.03
    object_pose[:, 2, 3] = 0.03
    object_pose[:, :3, :3] = torch.stack(
        [_rotation_z(i * 0.25) for i in range(batch_size)]
    )
    candidate = object_pose[0].clone()
    candidate[:3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    alternative = candidate.clone()
    alternative[0, 3] += 0.02
    generator = Mock()
    pools = []
    for row in range(batch_size):
        row_candidate = object_pose[row].clone()
        row_candidate[:3, :3] = object_pose[row, :3, :3] @ candidate[:3, :3]
        row_candidate[1, 3] += row * 0.001
        row_alternative = row_candidate.clone()
        row_alternative[0, 3] += 0.02
        pools.append(
            (torch.stack([row_candidate, row_alternative]), torch.tensor([0.1, 0.2]))
        )
    generator.get_valid_grasp_poses.return_value = pools
    generator_factory = Mock(return_value=generator)
    monkeypatch.setattr(
        _atomic_fixtures, "create_parallel_jaw_grasp_pose_generator", generator_factory
    )
    monkeypatch.setattr(
        _atomic_fixtures,
        "create_antipodal_semantics",
        lambda *args, **kwargs: SimpleNamespace(
            affordance=SimpleNamespace(
                mesh_vertices=torch.zeros(3, 3), mesh_triangles=torch.zeros(1, 3)
            )
        ),
    )
    full_qpos = torch.zeros(batch_size, 9)
    initial = torch.arange(batch_size).unsqueeze(1).expand(-1, 7) * 0.01
    full_qpos[:, :7] = initial
    inside_screen = False
    calls = []

    def solve(*, target_xpos, qpos_seed, **kwargs):
        success = torch.ones(len(qpos_seed), dtype=torch.bool)
        if inside_screen:
            assert target_xpos.shape[0] == batch_size
            # Environment four alone rejects both half-turns of the cheapest
            # candidate. The second candidate passes every requested waypoint.
            if (
                torch.linalg.vector_norm(target_xpos[4, :2, 3] - object_pose[4, :2, 3])
                < 0.01
            ):
                success[4] = False
        calls.append((inside_screen, target_xpos.clone(), qpos_seed.clone(), success))
        return success, qpos_seed + 0.01

    robot = SimpleNamespace(
        cfg=SimpleNamespace(
            solver_cfg={
                "arm": SimpleNamespace(
                    tcp=[
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.15],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            }
        ),
        device=CPU,
        _all_indices=torch.arange(batch_size),
        _solvers={"arm": SimpleNamespace(root_link_name="base", get_ik=solve)},
        get_link_pose=lambda **kwargs: torch.eye(4).repeat(
            len(kwargs["env_ids"]), 1, 1
        ),
        get_qpos=lambda name=None: (
            full_qpos[:, :7].clone() if name else full_qpos.clone()
        ),
        get_joint_ids=lambda name: list(range(7)),
        compute_fk=lambda *args, **kwargs: candidate.unsqueeze(0).repeat(
            batch_size, 1, 1
        ),
    )
    robot.compute_ik = MethodType(Robot.compute_ik, robot)
    scenario = AtomicSkillScenario()
    scenario.robot = robot
    handle = SimpleNamespace(object_id="cube", entity=Mock(), config={})
    scenario.activate_object = lambda object_id: handle
    scenario.restore_base_robot = lambda: None
    scenario.randomize_object_pose = lambda *args, **kwargs: object_pose.clone()

    def set_start(qpos, **kwargs):
        full_qpos[:, :7] = qpos

    scenario.set_robot_start = set_start
    screen = scenario._screen_antipodal_grasp_rows

    def checked_screen(*args, **kwargs):
        nonlocal inside_screen
        inside_screen = True
        try:
            return screen(*args, **kwargs)
        finally:
            inside_screen = False

    scenario._screen_antipodal_grasp_rows = checked_screen
    sample_pools = scenario._ranked_antipodal_grasp_pools

    def checked_pools(*args, **kwargs):
        if skill_id == "axis_align":
            ranking_probes = torch.stack([candidate, candidate.clone()])
            ranking_probes[1, :3, :3] = _rotation_z(torch.pi / 2) @ candidate[:3, :3]
            costs = torch.tensor([0.1, 0.2])
            assert kwargs["row_rank"](0, ranking_probes, costs).tolist() == [1, 0]
            assert kwargs["row_rank"](7, ranking_probes, costs).tolist() == [0, 1]
        return sample_pools(*args, **kwargs)

    scenario._ranked_antipodal_grasp_pools = checked_pools
    config = {
        "name": "batched_rotation",
        "object": "cube",
        "grasp_source": "antipodal",
        "grasp_sample_count": 321,
        "grasp_max_candidates": 2,
        "approach_direction": [0.0, 0.0, -1.0],
        "internal_axis": [1.0, 0.0, 0.0],
        "target_axis": [0.0, 0.0, 1.0],
        "pre_action_settle_steps": 0,
    }
    if early_filter:
        config["grasp_filter_before_ranking"] = True
        if skill_id == "pour":
            config["grasp_approach_candidates"] = [
                {"yaw_offset_rad": 0.0, "max_candidates": 2}
            ]
    case = create_atomic_skill_provider(skill_id).generate_case(
        scenario,
        SimpleNamespace(
            suite_version="test_v1", robot=SimpleNamespace(id="franka_pgi")
        ),
        SimpleNamespace(id="atomic-test"),
        config,
        seed=43001,
        batch_size=batch_size,
    )
    screening = [call for call in calls if call[0]]
    assert len(screening) == 2 * waypoint_count + 1
    # The first candidate finishes for seven rows even though row four fails.
    # Those seven frozen choices survive row four's mirror and next candidate.
    accepted = screening[:waypoint_count]
    last_accepted = screening[-waypoint_count:]
    accepted_poses = torch.stack([call[1] for call in accepted], dim=1)
    accepted_poses[4] = torch.stack([call[1][4] for call in last_accepted])
    torch.testing.assert_close(accepted_poses, case.target_waypoints)
    for index, call in enumerate(accepted):
        torch.testing.assert_close(call[2][0], accepted[0][2][0] + index * 0.01)
    for index, call in enumerate(last_accepted):
        torch.testing.assert_close(call[2][4], last_accepted[0][2][4] + index * 0.01)
    references = [call for call in calls if not call[0]][-waypoint_count:]
    torch.testing.assert_close(
        torch.stack([call[1] for call in references], dim=1), case.target_waypoints
    )
    for index, call in enumerate(references):
        torch.testing.assert_close(call[2], case.start_qpos + index * 0.01)
    generator_factory.assert_called_once_with(
        gripper_model=None,
        root_from_tcp=robot.cfg.solver_cfg["arm"].tcp,
        n_sample=321,
        force_refresh=False,
        max_candidates=2,
        **({"approach_alignment_angle": math.radians(10)} if early_filter else {}),
    )
    assert case.seed == 43001 and case.batch_size == 8

    assert (
        case.case_parameters.get("grasp_filter_before_ranking", False) is early_filter
    )

    if early_filter and skill_id == "pour":
        assert (
            case.case_parameters["approach_realisation"]["total_candidate_allocation"]
            == 2
        )
        torch.testing.assert_close(
            torch.tensor(case.case_parameters["approach_direction"]),
            torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch_size, 1),
        )


def test_pour_requires_full_return_orientation_after_successful_tilt() -> None:
    initial = torch.eye(4).repeat(2, 1, 1)
    actuation = torch.eye(4).repeat(3, 2, 1, 1)
    actuation[:, :, 2, 3] = 0.1
    actuation[1, :, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    )
    settle = actuation[-1:].repeat(2, 1, 1, 1)
    # A drift around another axis must fail full return despite correct peak,
    # terminal lift and retention.
    settle[:, 1, :3, :3] = _rotation_z(0.3)
    case = _case(
        "pour",
        2,
        object_initial_pose=initial.tolist(),
        object_rotation_axis=[1.0, 0.0, 0.0],
        object_rotation_sign=1.0,
        rotate_angle_rad=0.5,
    )
    observation = _observation(
        2,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "pour_actuation": ReplayObservationTrace(object_pose=actuation),
            "pour_settle": ReplayObservationTrace(
                object_pose=settle,
                gripper_qpos={"hand": torch.full((2, 2, 1), 0.5)},
            ),
        },
    )
    evaluator = create_atomic_skill_provider("pour").physical_evaluator()
    results = evaluator.evaluate(case, observation, _outcomes(2))
    assert [result.task_success for result in results] == [True, False]
    assert results[1].failure_stage == "settled"
    assert results[1].stages[1].measurements[
        "return_rotation_error_rad"
    ] == pytest.approx(0.3)
