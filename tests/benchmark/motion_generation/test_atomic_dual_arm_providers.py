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

"""Tests for the dual-arm benchmark providers and their scenario support."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import numpy as np

from embodichain_benchmarks.motion_generation.scenarios._atomic_pose_sampling import (
    apply_task_frame_randomization,
)

from embodichain.lab.sim.motion.planners import PlanState
from embodichain.toolkits.graspkit import (
    ParallelJawGraspPoseGenerator,
    ParallelJawGripperModelCfg,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv
from embodichain_benchmarks.motion_generation.config import (
    AtomicPoseRandomizationCfg,
    load_suite,
)
from embodichain_benchmarks.motion_generation.contracts import (
    ReplayObservationTrace,
    ReplayObservationWindow,
)
from embodichain_benchmarks.motion_generation.models import BenchmarkCase, CaseOutcome
from embodichain_benchmarks.motion_generation import robots as _robots  # noqa: F401
from embodichain_benchmarks.motion_generation.registry import create_robot_provider
from embodichain_benchmarks.motion_generation.scenarios import (
    atomic_dual_arm_providers as dual,
)
from embodichain_benchmarks.motion_generation.scenarios import atomic_case_utils
from embodichain_benchmarks.motion_generation.scenarios import atomic_skill
from embodichain_benchmarks.motion_generation.scenarios.atomic_skill import (
    AtomicSkillScenario,
)
from embodichain_benchmarks.motion_generation.scenarios.atomic_grasps import (
    ActiveCaseGraspPoseGenerator,
    CaseSampleCountGraspPoseGenerator,
)
from embodichain_benchmarks.motion_generation.contracts import ExecutionObservation
from embodichain_benchmarks.motion_generation.atomic_skill import (
    create_atomic_skill_provider,
)

CPU = torch.device("cpu")


def _pose(x: float, y: float = 0.0, z: float = 0.0) -> torch.Tensor:
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([x, y, z])
    return pose


def _case(skill_id: str, **parameters: object) -> BenchmarkCase:
    return BenchmarkCase(
        suite_version="test_v1",
        track="atomic-skill",
        scenario_id=skill_id,
        case_id=f"atomic-skill:{skill_id}:case:b1:s11",
        seed=11,
        batch_size=1,
        num_waypoints=1,
        path_shape="test",
        start_state_bin="pre_action",
        start_qpos=torch.zeros(1, 3),
        target_waypoints=_pose(0.1).reshape(1, 1, 4, 4),
        reference_qpos=torch.zeros(1, 1, 3),
        robot_id="dual_arm",
        skill_id=skill_id,
        object_id="object",
        task_difficulty="simple",
        primary_success="task_success",
        full_start_qpos=torch.zeros(1, 6),
        case_parameters=dict(parameters),
    )


def _outcome(*, motion_valid: bool = True) -> CaseOutcome:
    return CaseOutcome(
        env_index=0,
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


def _observation(**values: object) -> ExecutionObservation:
    return ExecutionObservation(
        execution_success=torch.ones(1, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).unsqueeze(0),
        joint_tracking_rmse_rad=torch.zeros(1),
        execution_time_ms=1.0,
        task_completion_time_s=0.1,
        **values,
    )


@pytest.mark.parametrize(
    ("xy_offset", "z_offset", "success"),
    [
        (0.02, -0.05, True),
        (0.05, -0.05, False),
        (0.0, -0.15, False),
        (0.0, 0.05, False),
    ],
)
def test_hand_over_physical_evaluator_uses_phase_local_observations(
    xy_offset: float, z_offset: float, success: bool
) -> None:
    target = _pose(0.2, -0.1, 0.1).unsqueeze(0)
    initial = _pose(0.0).unsqueeze(0)
    object_trace = torch.stack((target, target))
    settle_trace = object_trace.clone()
    settle_trace[:, :, 0, 3] += xy_offset
    settle_trace[:, :, 2, 3] += z_offset
    case = _case(
        "hand_over",
        control_parts={
            "source": {"motion": "left_arm", "grasp": "left_hand"},
            "destination": {"motion": "right_arm", "grasp": "right_hand"},
        },
        object_initial_pose=initial.tolist(),
        target_object_pose=target.tolist(),
    )
    holding = torch.full((2, 1, 1), 0.5)
    released = torch.zeros(2, 1, 1)
    observation = _observation(
        gripper_open_qpos={
            "left_hand": torch.tensor([0.0]),
            "right_hand": torch.tensor([0.0]),
        },
        gripper_grasp_qpos={
            "left_hand": torch.tensor([1.0]),
            "right_hand": torch.tensor([1.0]),
        },
        replay_traces={
            "handover_pickup": ReplayObservationTrace(
                gripper_qpos={"left_hand": holding, "right_hand": released}
            ),
            "handover_transfer": ReplayObservationTrace(
                object_pose=object_trace,
                gripper_qpos={"left_hand": holding, "right_hand": holding},
            ),
            "handover_exchange": ReplayObservationTrace(
                object_pose=object_trace,
                gripper_qpos={"left_hand": released, "right_hand": holding},
            ),
            "handover_settle": ReplayObservationTrace(
                object_pose=settle_trace,
                gripper_qpos={"left_hand": released, "right_hand": released},
            ),
        },
    )

    evaluator = create_atomic_skill_provider("hand_over").physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_outcome(),))[0]

    assert result.task_success is success
    if not success:
        assert result.failure_stage == "placed"
    assert [stage.stage_id for stage in result.stages] == [
        "grasped",
        "transferred",
        "handed_over",
        "receiver_released",
        "placed",
    ]


def test_hand_over_requires_receiver_release_at_terminal_target() -> None:
    target = _pose(0.2, -0.1, 0.1).unsqueeze(0)
    initial = _pose(0.0).unsqueeze(0)
    object_trace = torch.stack((target, target))
    holding = torch.full((2, 1, 1), 0.5)
    receiver_holding = torch.full((2, 1, 1), 0.75)
    released = torch.zeros(2, 1, 1)
    case = _case(
        "hand_over",
        control_parts={
            "source": {"motion": "left_arm", "grasp": "left_hand"},
            "destination": {"motion": "right_arm", "grasp": "right_hand"},
        },
        object_initial_pose=initial.tolist(),
        target_object_pose=target.tolist(),
    )
    observation = _observation(
        gripper_open_qpos={
            "left_hand": torch.tensor([0.0]),
            "right_hand": torch.tensor([0.0]),
        },
        gripper_grasp_qpos={
            "left_hand": torch.tensor([1.0]),
            "right_hand": torch.tensor([1.0]),
        },
        replay_traces={
            "handover_pickup": ReplayObservationTrace(
                gripper_qpos={"left_hand": holding, "right_hand": released}
            ),
            "handover_transfer": ReplayObservationTrace(
                object_pose=object_trace,
                gripper_qpos={"left_hand": holding, "right_hand": holding},
            ),
            "handover_exchange": ReplayObservationTrace(
                object_pose=object_trace,
                gripper_qpos={"left_hand": released, "right_hand": holding},
            ),
            "handover_settle": ReplayObservationTrace(
                object_pose=object_trace,
                gripper_qpos={
                    "left_hand": released,
                    "right_hand": receiver_holding,
                },
            ),
        },
    )

    evaluator = create_atomic_skill_provider("hand_over").physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_outcome(),))[0]

    assert not result.task_success
    assert result.failure_stage == "receiver_released"


@pytest.mark.parametrize(
    ("offset", "yaw", "left_qpos", "right_qpos", "failure_stage"),
    [
        (0.005, 0.0, 0.02, 0.02, None),
        (0.02, 0.0, 0.02, 0.02, "moved"),
        (0.0, math.pi / 2, 0.02, 0.02, "moved"),
        (0.0, 0.0, 0.02, 0.026, "dual_grasped"),
        (0.0, 0.0, 0.0255, 0.02, "dual_grasped"),
        (0.0, 0.0, 0.0, 0.02, "dual_grasped"),
        (0.0, 0.0, None, 0.02, "dual_grasped"),
    ],
)
def test_coordinated_pickment_evaluator_uses_all_physical_stages(
    offset: float,
    yaw: float,
    left_qpos: float | None,
    right_qpos: float,
    failure_stage: str | None,
) -> None:
    initial = _pose(0.0).unsqueeze(0)
    target = _pose(0.1, 0.0, 0.1).unsqueeze(0)
    lift = torch.stack((initial, _pose(0.0, 0.0, 0.05).unsqueeze(0)))
    final = target.clone()
    final[:, 0, 3] += offset
    final[:, :3, :3] = axis_angle_to_rotation_matrix(torch.tensor([0.0, 0.0, yaw]))
    move = torch.stack((lift[-1], final))
    grippers = {"right_hand": torch.full((2, 1, 1), right_qpos)}
    if left_qpos is not None:
        grippers["left_hand"] = torch.full((2, 1, 1), left_qpos)
    case = _case(
        "coordinated_pickment",
        control_parts={
            "left": {"motion": "left_arm", "grasp": "left_hand"},
            "right": {"motion": "right_arm", "grasp": "right_hand"},
        },
        object_initial_pose=initial.tolist(),
        target_object_pose=target.tolist(),
    )
    observation = _observation(
        gripper_open_qpos={
            "left_hand": torch.tensor([0.0]),
            "right_hand": torch.tensor([0.0]),
        },
        gripper_grasp_qpos={
            "left_hand": torch.tensor([0.026]),
            "right_hand": torch.tensor([0.026]),
        },
        replay_traces={
            "dual_pick_grasp": ReplayObservationTrace(gripper_qpos=grippers),
            "dual_pick_lift": ReplayObservationTrace(object_pose=lift),
            "dual_pick_move": ReplayObservationTrace(object_pose=move),
            "dual_pick_settle": ReplayObservationTrace(
                object_pose=final.unsqueeze(0).repeat(2, 1, 1, 1),
                gripper_qpos=grippers,
            ),
        },
    )

    evaluator = create_atomic_skill_provider(
        "coordinated_pickment"
    ).physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_outcome(),))[0]

    assert result.task_success is (failure_stage is None)
    assert result.failure_stage == failure_stage
    assert [stage.stage_id for stage in result.stages] == [
        "dual_grasped",
        "lifted",
        "moved",
        "settled",
        "held",
    ]


def test_coordinated_pickment_rechecks_target_after_settle() -> None:
    initial = _pose(0.0).unsqueeze(0)
    target = _pose(0.1, 0.0, 0.1).unsqueeze(0)
    lift = torch.stack((initial, _pose(0.0, 0.0, 0.05).unsqueeze(0)))
    move = torch.stack((lift[-1], target))
    settled_wrong = _pose(0.4, 0.0, 0.1).reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)
    holding = torch.full((2, 1, 1), 0.5)
    case = _case(
        "coordinated_pickment",
        control_parts={
            "left": {"motion": "left_arm", "grasp": "left_hand"},
            "right": {"motion": "right_arm", "grasp": "right_hand"},
        },
        object_initial_pose=initial.tolist(),
        target_object_pose=target.tolist(),
    )
    observation = _observation(
        gripper_open_qpos={
            "left_hand": torch.tensor([0.0]),
            "right_hand": torch.tensor([0.0]),
        },
        gripper_grasp_qpos={
            "left_hand": torch.tensor([1.0]),
            "right_hand": torch.tensor([1.0]),
        },
        replay_traces={
            "dual_pick_grasp": ReplayObservationTrace(
                gripper_qpos={"left_hand": holding, "right_hand": holding}
            ),
            "dual_pick_lift": ReplayObservationTrace(object_pose=lift),
            "dual_pick_move": ReplayObservationTrace(object_pose=move),
            "dual_pick_settle": ReplayObservationTrace(
                object_pose=settled_wrong,
                gripper_qpos={"left_hand": holding, "right_hand": holding},
            ),
        },
    )

    evaluator = create_atomic_skill_provider(
        "coordinated_pickment"
    ).physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_outcome(),))[0]

    assert not result.task_success
    assert result.failure_stage == "settled"


@pytest.mark.parametrize(
    ("position_offset", "rotation", "support_present", "failure_stage"),
    [
        (0.0, (0.0, 0.0, 0.0), True, None),
        (-0.1, (0.0, 0.0, 0.0), True, "aligned"),
        (0.0, (0.3, 0.0, 0.0), True, "aligned"),
        (0.0, (0.0, 0.0, 0.3), True, "aligned"),
        (0.0, (0.0, 0.0, 0.0), False, "aligned"),
    ],
)
def test_coordinated_placement_evaluator_uses_named_support_pose_trace(
    position_offset: float,
    rotation: tuple[float, float, float],
    support_present: bool,
    failure_stage: str | None,
) -> None:
    support = _pose(0.5, 0.2, 0.7)
    support[:3, :3] = axis_angle_to_rotation_matrix(torch.tensor([0.0, 0.0, 0.4]))
    relative = _pose(0.04, -0.1, 0.1)
    pose = (support @ relative).unsqueeze(0)
    actual = relative.clone()
    actual[2, 3] += position_offset
    actual[:3, :3] = axis_angle_to_rotation_matrix(torch.tensor(rotation))
    pose_trace = (support @ actual).reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)
    support_traces = (
        {"support": support.reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)}
        if support_present
        else {}
    )
    placing_released = torch.zeros(2, 1, 1)
    support_held = torch.full((2, 1, 1), 0.5)
    case = _case(
        "coordinated_placement",
        control_parts={
            "placing": {"motion": "left_arm", "grasp": "left_hand"},
            "support": {"motion": "right_arm", "grasp": "right_hand"},
        },
        support_object_id="support",
        object_initial_pose=pose.tolist(),
        placing_target_pose=pose.tolist(),
        support_target_pose=support.unsqueeze(0).tolist(),
        placing_height_offset_m=0.0,
        support_height_offset_m=0.0,
        placing_relative_target_pose=relative.unsqueeze(0).tolist(),
    )
    observation = _observation(
        gripper_open_qpos={
            "left_hand": torch.tensor([0.0]),
            "right_hand": torch.tensor([0.0]),
        },
        gripper_grasp_qpos={
            "left_hand": torch.tensor([1.0]),
            "right_hand": torch.tensor([1.0]),
        },
        replay_traces={
            "coordinated_place_alignment": ReplayObservationTrace(
                object_pose=pose_trace,
                scene_object_poses=support_traces,
            ),
            "coordinated_place_release": ReplayObservationTrace(
                gripper_qpos={
                    "left_hand": placing_released,
                    "right_hand": support_held,
                }
            ),
            "coordinated_place_settle": ReplayObservationTrace(
                object_pose=pose_trace,
                scene_object_poses=support_traces,
            ),
        },
    )

    evaluator = create_atomic_skill_provider(
        "coordinated_placement"
    ).physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_outcome(),))[0]

    assert result.task_success is (failure_stage is None)
    assert result.failure_stage == failure_stage
    if failure_stage is None:
        assert result.stages[1].measurements == {
            "placing_release_established": True,
            "support_retained": True,
        }
    elif not support_present:
        assert result.failure_code == "evidence_unavailable"


def test_coordinated_placement_alignment_uses_world_height_offsets() -> None:
    """Alignment follows raised commands while stability uses resting poses."""
    support_target = _pose(0.3, -0.2, 0.4)
    support_target[:3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([0.0, math.pi / 2, 0.0])
    )
    placing_target = support_target.clone()
    placing_height_offset = 0.10
    support_height_offset = 0.02
    aligned_placing = placing_target.clone()
    aligned_support = support_target.clone()
    aligned_placing[2, 3] += placing_height_offset
    aligned_support[2, 3] += support_height_offset
    aligned_placing_trace = aligned_placing.reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)
    aligned_support_trace = aligned_support.reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)
    resting_placing_trace = placing_target.reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)
    resting_support_trace = support_target.reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)
    placing_released = torch.zeros(2, 1, 1)
    support_held = torch.full((2, 1, 1), 0.5)
    case = _case(
        "coordinated_placement",
        control_parts={
            "placing": {"motion": "left_arm", "grasp": "left_hand"},
            "support": {"motion": "right_arm", "grasp": "right_hand"},
        },
        support_object_id="support",
        object_initial_pose=placing_target.unsqueeze(0).tolist(),
        placing_target_pose=placing_target.unsqueeze(0).tolist(),
        support_target_pose=support_target.unsqueeze(0).tolist(),
        placing_height_offset_m=placing_height_offset,
        support_height_offset_m=support_height_offset,
        placing_relative_target_pose=torch.eye(4).unsqueeze(0).tolist(),
    )
    observation = _observation(
        gripper_open_qpos={
            "left_hand": torch.tensor([0.0]),
            "right_hand": torch.tensor([0.0]),
        },
        gripper_grasp_qpos={
            "left_hand": torch.tensor([1.0]),
            "right_hand": torch.tensor([1.0]),
        },
        replay_traces={
            "coordinated_place_alignment": ReplayObservationTrace(
                object_pose=aligned_placing_trace,
                scene_object_poses={"support": aligned_support_trace},
            ),
            "coordinated_place_release": ReplayObservationTrace(
                gripper_qpos={
                    "left_hand": placing_released,
                    "right_hand": support_held,
                }
            ),
            "coordinated_place_settle": ReplayObservationTrace(
                object_pose=resting_placing_trace,
                scene_object_poses={"support": resting_support_trace},
            ),
        },
    )

    evaluator = create_atomic_skill_provider(
        "coordinated_placement"
    ).physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_outcome(),))[0]

    assert result.task_success
    assert [stage.stage_id for stage in result.stages] == [
        "aligned",
        "released",
        "stable",
    ]
    assert result.stages[0].measurements["relative_position_error_m"] < 1.0e-6
    assert result.stages[2].measurements["relative_position_error_m"] < 1.0e-6


def test_coordinated_placement_rechecks_relative_target_after_release() -> None:
    target = _pose(0.0).unsqueeze(0)
    aligned_trace = target.unsqueeze(0).repeat(2, 1, 1, 1)
    support_trace = target.unsqueeze(0).repeat(2, 1, 1, 1)
    settled_wrong = _pose(0.5).reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)
    placing_released = torch.zeros(2, 1, 1)
    support_held = torch.full((2, 1, 1), 0.5)
    case = _case(
        "coordinated_placement",
        control_parts={
            "placing": {"motion": "left_arm", "grasp": "left_hand"},
            "support": {"motion": "right_arm", "grasp": "right_hand"},
        },
        support_object_id="support",
        object_initial_pose=target.tolist(),
        placing_target_pose=target.tolist(),
        support_target_pose=target.tolist(),
        placing_height_offset_m=0.0,
        support_height_offset_m=0.0,
        placing_relative_target_pose=target.tolist(),
    )
    observation = _observation(
        gripper_open_qpos={
            "left_hand": torch.tensor([0.0]),
            "right_hand": torch.tensor([0.0]),
        },
        gripper_grasp_qpos={
            "left_hand": torch.tensor([1.0]),
            "right_hand": torch.tensor([1.0]),
        },
        replay_traces={
            "coordinated_place_alignment": ReplayObservationTrace(
                object_pose=aligned_trace,
                scene_object_poses={"support": support_trace},
            ),
            "coordinated_place_release": ReplayObservationTrace(
                gripper_qpos={
                    "left_hand": placing_released,
                    "right_hand": support_held,
                }
            ),
            "coordinated_place_settle": ReplayObservationTrace(
                object_pose=settled_wrong,
                scene_object_poses={"support": support_trace},
            ),
        },
    )

    evaluator = create_atomic_skill_provider(
        "coordinated_placement"
    ).physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_outcome(),))[0]

    assert not result.task_success
    assert result.failure_stage == "stable"


def test_replay_collects_named_scene_object_poses_in_sync() -> None:
    state = {"step": 0}
    scenario = AtomicSkillScenario()
    scenario.simulation = Mock()
    scenario.simulation.update.side_effect = lambda step: state.update(
        step=state["step"] + step
    )
    scenario.robot = Mock()
    scenario.robot.get_qpos.side_effect = lambda: torch.zeros(1, 6)
    scenario.track = Mock(
        config={
            "physics": {
                "steps_per_waypoint": 1,
                "hold_steps": 0,
                "hold_sim_steps": 1,
            }
        }
    )

    def moving_pose(to_matrix: bool) -> torch.Tensor:
        del to_matrix
        pose = torch.eye(4).unsqueeze(0)
        pose[:, 0, 3] = 0.01 * state["step"]
        return pose

    placing = Mock()
    placing.entity.get_local_pose.side_effect = moving_pose
    support = Mock()
    support.entity.get_local_pose.return_value = _pose(0.2).unsqueeze(0)
    scenario._objects = {"placing": placing, "support": support}
    compiled = SimpleNamespace(
        trajectory=SimpleNamespace(positions=torch.zeros(1, 3, 6)),
        plan_success=torch.tensor([True]),
    )
    provider = Mock()
    provider.lift_segment_start.return_value = None
    provider.object_rotation_segment.return_value = None
    provider.replay_observation_windows.return_value = (
        ReplayObservationWindow(
            "relative",
            0,
            2,
            frozenset({"object_pose", "scene_object_pose"}),
            scene_object_ids=("support",),
        ),
    )
    case = replace(
        _case("coordinated_placement", support_object_id="support"),
        object_id="placing",
    )

    metrics = scenario._replay_runtime().replay_physics(
        compiled, case, provider, collect_metrics=True
    )

    assert metrics is not None
    trace = metrics.replay_traces["relative"]
    assert trace.object_pose is not None
    assert trace.object_pose.shape == (3, 1, 4, 4)
    assert trace.scene_object_poses["support"].shape == (3, 1, 4, 4)
    torch.testing.assert_close(
        trace.scene_object_poses["support"][:, 0, 0, 3], torch.full((3,), 0.2)
    )


def _two_arm_robot() -> Mock:
    """A 6-joint robot whose arm FK maps the first three joints to x, y, z."""
    robot = Mock(device=CPU)
    robot.get_joint_ids.side_effect = lambda name: (
        [0, 1, 2] if name == "left_arm" else [3, 4, 5]
    )
    robot.get_qpos_limits.side_effect = lambda name: torch.tensor([[[-5.0, 5.0]] * 3])
    robot.get_qvel_limits.return_value = torch.full((1, 3), 2.0)

    def batch_fk(*, qpos: torch.Tensor, name: str, to_matrix: bool) -> torch.Tensor:
        del name, to_matrix
        poses = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        poses[..., :3, 3] = qpos[..., :3]
        return poses

    robot.compute_batch_fk.side_effect = batch_fk
    return robot


def _protocol_suite() -> Mock:
    return Mock(
        protocol=Mock(
            validation_samples=32,
            position_threshold_m=0.01,
            rotation_threshold_rad=0.1,
            joint_limit_tolerance_rad=1.0e-5,
            start_state_tolerance_rad=1.0e-3,
            joint_velocity_limit_rad_s=None,
            joint_acceleration_limit_rad_s2=None,
            joint_jerk_limit_rad_s3=None,
            dynamic_limit_tolerance=1.0e-5,
        )
    )


@pytest.mark.parametrize(("right_x", "valid"), [(0.2, True), (0.05, False)])
def test_other_arm_must_pass_its_own_frozen_waypoints(right_x: float, valid: bool):
    case = _case("hand_over", arm_waypoints={"right_arm": [[_pose(0.2).tolist()]]})
    positions = torch.zeros(1, 5, 6)
    positions[0, :, 0] = torch.linspace(0.0, 0.1, 5)
    positions[0, :, 3] = torch.linspace(0.0, right_x, 5)
    result = SimpleNamespace(
        plan_success=torch.tensor([True]),
        trajectory=SimpleNamespace(
            positions=positions, dt=torch.tensor([[0.0, 0.1, 0.1, 0.1, 0.1]])
        ),
    )

    merged = AtomicSkillScenario._with_other_arm_validity(
        result, case, _two_arm_robot(), _protocol_suite(), (_outcome(),)
    )

    assert merged[0].motion_valid is valid
    assert merged[0].ordered_waypoints_reached is valid
    assert merged[0].failure_code == (None if valid else "waypoint_miss")
    single = replace(case, case_parameters={})
    assert AtomicSkillScenario._with_other_arm_validity(
        result, single, _two_arm_robot(), _protocol_suite(), (_outcome(),)
    ) == (_outcome(),)


def test_reference_recorder_keeps_requested_poses_but_not_internal_ik():
    robot = SimpleNamespace()
    robot.compute_ik = Mock(return_value=(torch.tensor([True]), torch.zeros(1, 3)))
    robot.compute_fk = Mock(return_value=_pose(0.9).unsqueeze(0))

    def original_generate(target_states, options=None):
        # A generator that interpolates through IK must not add IK targets.
        robot.compute_ik(pose=_pose(0.5).unsqueeze(0), name=options.control_part)
        return "planned"

    generator = SimpleNamespace(generate=original_generate)
    scenario = SimpleNamespace(
        robot=robot, reference_motion_generator=lambda: generator
    )
    plan = dual._ReferencePlan(compiled=None)

    with dual._record_requests(scenario, plan):
        assert (
            generator.generate(
                [
                    PlanState.from_xpos(_pose(0.1).unsqueeze(0)),
                    PlanState.from_qpos(torch.zeros(1, 3)),
                ],
                options=SimpleNamespace(control_part="left_arm"),
            )
            == "planned"
        )
        robot.compute_ik(pose=_pose(0.3).unsqueeze(0), name="right_arm")

    assert generator.generate is original_generate
    assert [pose[0, 0, 3].item() for pose in plan.planned_poses["left_arm"]] == [
        pytest.approx(0.1),
        pytest.approx(0.9),
    ]
    assert list(plan.solved_poses) == ["right_arm"]
    assert plan.arm_poses("left_arm").shape == (1, 2, 4, 4)
    assert plan.arm_poses("right_arm")[0, 0, 0, 3].item() == pytest.approx(0.3)
    with pytest.raises(RuntimeError, match="requested no pose"):
        plan.arm_poses("third_arm")


class _EndpointRobot:
    """Two prismatic chains with distinct rotated bases and TCP offsets."""

    def __init__(self, batch: int, failure: str | None = None) -> None:
        self.batch = batch
        self.failure = failure
        self.seeds = {"left_arm": [], "right_arm": []}
        self.bases = {}
        self.tcp = {}
        for arm, sign in (("left_arm", 1.0), ("right_arm", -1.0)):
            base = _pose(sign * 0.5, 0.0, 0.2).repeat(batch, 1, 1)
            base[:, :3, :3] = axis_angle_to_rotation_matrix(
                torch.tensor([0.0, 0.0, sign * math.pi / 2])
            )
            base[:, 1, 3] = torch.arange(batch) * 2.0
            self.bases[arm] = base
            self.tcp[arm] = _pose(0.0, 0.0, sign * 0.1)

    def get_joint_ids(self, *, name):
        return [0, 2, 4] if name == "left_arm" else [1, 3, 5]

    def get_qpos_limits(self, *, name, env_ids=None):
        count = self.batch if env_ids is None else len(env_ids)
        return torch.tensor([[[-1.0, 1.0]]]).repeat(count, 3, 1)

    def forward(self, qpos, name, env_ids=None):
        ids = list(range(self.batch)) if env_ids is None else env_ids
        local = torch.eye(4).repeat(len(qpos), 1, 1)
        local[:, :3, 3] = qpos
        return self.bases[name][ids] @ local @ self.tcp[name]

    def compute_ik(self, *, pose, joint_seed, name, env_ids=None):
        ids = list(range(self.batch)) if env_ids is None else env_ids
        self.seeds[name].append(joint_seed.clone())
        local = pose_inv(self.bases[name][ids]) @ pose @ pose_inv(self.tcp[name])
        qpos = local[:, :3, 3].clone()
        success = torch.ones(len(ids), dtype=torch.bool)
        if name == "right_arm" and 1 in ids:
            row = ids.index(1)
            if self.failure == "ik":
                success[row] = False
            elif self.failure == "joint_bounds":
                qpos[row, 0] = 2.0
            elif self.failure == "nonfinite":
                qpos[row, 0] = float("nan")
        return success, qpos

    def compute_fk(self, *, qpos, name, to_matrix, env_ids=None):
        assert to_matrix
        result = self.forward(qpos, name, env_ids)
        if name == "right_arm" and self.failure == "position":
            result[1, 0, 3] += 0.02  # Outside the unchanged 1cm arrival ball.
        elif name == "right_arm" and self.failure == "rotation":
            result[1, :3, :3] = result[1, :3, :3] @ axis_angle_to_rotation_matrix(
                torch.tensor([0.2, 0.0, 0.0])  # Outside the 0.1rad threshold.
            )
        return result


def _prepared_endpoint_case(batch=8, failure=None):
    suite = load_suite("atomic_dual_franka_pgi_hand_over")
    scenario = AtomicSkillScenario()
    scenario.robot = robot = _EndpointRobot(batch, failure)
    scenario.control_part = "left_arm"
    scenario.suite = suite
    full_start = torch.arange(batch * 6, dtype=torch.float32).reshape(batch, 6) / 100
    left_qpos = torch.arange(batch * 2 * 3).reshape(batch, 2, 3).float() / 100
    right_qpos = -torch.arange(batch * 3 * 3).reshape(batch, 3, 3).float() / 100
    left = torch.stack([robot.forward(q, "left_arm") for q in left_qpos.unbind(1)], 1)
    right = torch.stack(
        [robot.forward(q, "right_arm") for q in right_qpos.unbind(1)], 1
    )
    before = (full_start.clone(), left.clone(), right.clone())
    case = dual._case(
        create_atomic_skill_provider("hand_over"),
        scenario,
        suite,
        suite.tracks[0],
        {"name": "endpoint_contract"},
        seed=11,
        batch_size=batch,
        path_shape="test",
        object_id="object",
        targets=left,
        start_qpos=full_start[:, [0, 2, 4]].clone(),
        full_start_qpos=full_start,
        parameters={"arm_waypoints": {"right_arm": right.tolist()}},
    )
    for actual, original in zip((full_start, left, right), before, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    return case, robot, left_qpos, right_qpos


def test_dual_case_freezes_both_arm_endpoint_witnesses_from_the_full_start():
    """B8 rows retain each arm's own frame, TCP, joint indices and waypoint count."""
    case, robot, left, right = _prepared_endpoint_case()
    evidence = case.case_parameters["endpoint_qualification"]
    assert evidence["scope"] == "endpoint_reference_only_not_path_or_physical_success"
    assert set(evidence["arms"]) == {"left_arm", "right_arm"}
    torch.testing.assert_close(case.reference_qpos, left)
    for arm, expected in (("left_arm", left), ("right_arm", right)):
        row = evidence["arms"][arm]
        torch.testing.assert_close(torch.tensor(row["reference_qpos"]), expected)
        torch.testing.assert_close(
            torch.tensor(row["start_qpos"]),
            case.full_start_qpos[:, robot.get_joint_ids(name=arm)],
        )
        torch.testing.assert_close(robot.seeds[arm][0], torch.tensor(row["start_qpos"]))
        torch.testing.assert_close(robot.seeds[arm][1], expected[:, 0])
        assert torch.tensor(row["qualified"]).shape == expected.shape[:2]
        assert torch.tensor(row["qualified"]).all()
        assert torch.tensor(row["position_error_m"]).max() < 1e-5
        assert torch.tensor(row["rotation_error_rad"]).max() < 1e-5


@pytest.mark.parametrize(
    "failure", ["position", "rotation", "joint_bounds", "nonfinite", "ik"]
)
def test_dual_case_does_not_freeze_when_only_the_primary_arm_is_qualified(failure):
    """A secondary-arm failure keeps the case incomplete, never dropping a row."""
    with pytest.raises(RuntimeError, match="right_arm") as error:
        _prepared_endpoint_case(batch=2, failure=failure)
    assert "1" in str(error.value)
    if failure == "ik":
        assert "does not establish unreachability" in str(error.value)


def test_control_parts_default_to_left_then_right_and_validate():
    assert dual._control_parts({}, ("source", "destination")) == {
        "source": {"motion": "left_arm", "grasp": "left_eef"},
        "destination": {"motion": "right_arm", "grasp": "right_eef"},
    }
    swapped = dual._control_parts(
        {"control_parts": {"source": {"motion": "right_arm", "grasp": "right_eef"}}},
        ("source", "destination"),
    )
    assert swapped["source"]["motion"] == "right_arm"
    with pytest.raises(ValueError, match="exactly motion and grasp"):
        dual._control_parts(
            {"control_parts": {"left": {"extra": "x"}}}, ("left", "right")
        )


def test_dual_arm_target_randomization_is_batched_and_deterministic():
    """HandOver and Pickment receive one stable target sample per env row."""
    batch_size = 4
    object_pose = torch.eye(4).repeat(batch_size, 1, 1)
    config = {
        "_pose_batch": AtomicPoseRandomizationCfg(
            target_translation_jitter_m=(
                0.03,
                0.02,
                0.01,
            ),
            seed_stride=1,
        )
    }

    first = dual._randomized_target_pose(
        object_pose,
        config,
        default_offset=(0.0, -0.2, 0.1),
        seed=11,
        stream=703,
    )
    second = dual._randomized_target_pose(
        object_pose,
        config,
        default_offset=(0.0, -0.2, 0.1),
        seed=11,
        stream=703,
    )

    offsets = first[:, :3, 3] - torch.tensor([0.0, -0.2, 0.1])
    torch.testing.assert_close(first, second)
    assert torch.unique(offsets, dim=0).shape[0] == batch_size
    assert torch.all(offsets.abs() <= torch.tensor([0.03, 0.02, 0.01]))


def test_coordinated_placement_randomizes_both_objects_before_one_settle():
    """The placing and support batches share a settle window but separate streams."""
    batch_size = 4
    placing_pose = torch.eye(4).repeat(batch_size, 1, 1)
    support_pose = torch.eye(4).repeat(batch_size, 1, 1)
    placing = Mock(object_id="bread", entity=Mock())
    support = Mock(object_id="pan", entity=Mock())
    placing.entity.get_local_pose.return_value = placing_pose
    support.entity.get_local_pose.return_value = support_pose
    scenario = Mock(simulation=Mock())
    config = {
        "pre_action_settle_steps": 12,
        "object_position_jitter_m": [0.01, 0.01, 0.0],
        "support_object_position_jitter_m": [0.02, 0.03, 0.0],
        "support_object_yaw_jitter_rad": 0.2,
        "_pose_batch": AtomicPoseRandomizationCfg(
            object_translation_jitter_m=(
                0.04,
                0.04,
                0.0,
            )
        ),
    }

    actual = dual._settled_placement_poses(scenario, placing, support, config, seed=17)

    torch.testing.assert_close(actual[0], placing_pose)
    torch.testing.assert_close(actual[1], support_pose)
    assert actual[2] == 12
    placing_call, support_call = scenario.randomize_object_pose.call_args_list
    assert placing_call.kwargs == {"seed": 17, "stream": 901, "settle_steps": 0}
    assert placing_call.args[1] is config
    assert support_call.kwargs == {"seed": 17, "stream": 903, "settle_steps": 0}
    assert support_call.args[1]["object_position_jitter_m"] == [0.02, 0.03, 0.0]
    assert support_call.args[1]["object_yaw_jitter_rad"] == pytest.approx(0.2)
    scenario.simulation.update.assert_called_once_with(step=12)
    placing.entity.clear_dynamics.assert_called_once_with()
    support.entity.clear_dynamics.assert_called_once_with()


def test_coordinated_placement_builds_each_pose_row_independently(monkeypatch):
    """Placement keeps row geometry and applies one shared target shift per row."""
    batch_size = 4
    placing_pose = torch.eye(4).repeat(batch_size, 1, 1)
    support_pose = torch.eye(4).repeat(batch_size, 1, 1)
    placing_pose[:, 0, 3] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    support_pose[:, 1, 3] = torch.tensor([-0.1, -0.2, -0.3, -0.4])
    geometry = (torch.zeros(1, 3),) * 4
    monkeypatch.setattr(dual, "_placement_geometry", lambda *args: geometry)

    def row_poses(placing, support, *args, **kwargs):
        del args, kwargs
        return {
            "placing_grasp_pose": placing.clone(),
            "support_grasp_pose": support.clone(),
            "placing_target_pose": placing.clone(),
            "support_target_pose": support.clone(),
        }

    monkeypatch.setattr(dual, "_placement_pose_row", row_poses)

    config = {
        "target_object_offset_jitter_m": [0.03, 0.02, 0.01],
        "_pose_batch": AtomicPoseRandomizationCfg(seed_stride=1),
    }
    support = Mock()
    support.entity.get_triangles.return_value = torch.tensor([[[0, 1, 2], [1, 2, 3]]])
    poses = dual._placement_poses(
        Mock(), support, placing_pose, support_pose, config, seed=11
    )
    repeated = dual._placement_poses(
        Mock(), support, placing_pose, support_pose, config, seed=11
    )

    torch.testing.assert_close(poses["placing_grasp_pose"], placing_pose)
    torch.testing.assert_close(poses["support_grasp_pose"], support_pose)
    placing_shift = poses["placing_target_pose"][:, :3, 3] - placing_pose[:, :3, 3]
    support_shift = poses["support_target_pose"][:, :3, 3] - support_pose[:, :3, 3]
    torch.testing.assert_close(placing_shift, support_shift)
    assert torch.unique(placing_shift, dim=0).shape[0] == batch_size
    assert torch.all(placing_shift.abs() <= torch.tensor([0.03, 0.02, 0.01]))
    for name, pose in poses.items():
        torch.testing.assert_close(pose, repeated[name])


def test_coordinated_placement_transforms_fixed_targets_with_task_frame(monkeypatch):
    """The shared task frame moves both fixed tutorial targets with the objects."""
    batch_size = 4
    nominal_placing_pose = torch.eye(4).repeat(batch_size, 1, 1)
    nominal_support_pose = torch.eye(4).repeat(batch_size, 1, 1)
    nominal_placing_pose[:, :3, 3] = torch.tensor([0.2, -0.1, 0.2])
    nominal_support_pose[:, :3, 3] = torch.tensor([-0.1, 0.2, 0.2])
    config = {
        "_pose_batch": AtomicPoseRandomizationCfg(
            task_frame_translation_jitter_m=(
                0.03,
                0.02,
                0.0,
            ),
            task_frame_yaw_jitter_rad=0.3,
            seed_stride=1,
        )
    }
    placing_pose = nominal_placing_pose.clone()
    support_pose = nominal_support_pose.clone()
    apply_task_frame_randomization(placing_pose, config, seed=11)
    apply_task_frame_randomization(support_pose, config, seed=11)
    placing_vertices = torch.tensor(
        [
            [-0.04, -0.03, -0.01],
            [0.04, -0.03, -0.01],
            [-0.04, 0.03, 0.01],
            [0.04, 0.03, 0.01],
        ]
    )
    support_vertices = torch.tensor(
        [
            [-0.08, -0.08, 0.0],
            [0.08, -0.08, 0.0],
            [-0.08, 0.08, 0.0],
            [0.08, 0.08, 0.0],
            [0.15, 0.0, 0.06],
        ]
    )
    geometry = (
        placing_vertices,
        support_vertices,
        placing_vertices.min(dim=0).values,
        placing_vertices.max(dim=0).values,
    )
    monkeypatch.setattr(dual, "_placement_geometry", lambda *args: geometry)

    support = Mock()
    support.entity.get_triangles.return_value = torch.tensor([[[0, 1, 2], [1, 2, 3]]])
    poses = dual._placement_poses(
        Mock(), support, placing_pose, support_pose, config, seed=11
    )
    nominal = dual._placement_poses(
        Mock(), support, nominal_placing_pose, nominal_support_pose, {}, seed=11
    )
    task_frame_pose = torch.eye(4).repeat(batch_size, 1, 1)
    apply_task_frame_randomization(task_frame_pose, config, seed=11)
    expected_placing = torch.bmm(task_frame_pose, nominal["placing_target_pose"])
    expected_support = torch.bmm(task_frame_pose, nominal["support_target_pose"])

    torch.testing.assert_close(poses["placing_target_pose"], expected_placing)
    torch.testing.assert_close(poses["support_target_pose"], expected_support)
    torch.testing.assert_close(
        torch.bmm(pose_inv(poses["support_target_pose"]), poses["placing_target_pose"]),
        torch.bmm(
            pose_inv(nominal["support_target_pose"]),
            nominal["placing_target_pose"],
        ),
    )


def test_scenario_configures_every_gripper_and_freezes_each_hand_over_grasp():
    robot = Mock(device=CPU, dof=16)
    robot.control_parts = {
        "left_arm": object(),
        "right_arm": object(),
        "left_eef": object(),
        "right_eef": object(),
    }
    robot.get_qpos.return_value = torch.zeros(1, 16)
    robot.get_qvel.return_value = torch.zeros(1, 16)
    robot.get_qpos_limits.side_effect = lambda name: torch.tensor([[[0.0, 0.04]]])
    robot.get_joint_ids.side_effect = lambda name: {
        "left_arm": list(range(6)),
        "right_arm": list(range(6, 12)),
        "left_eef": [12],
        "right_eef": [13],
    }[name]
    simulation = Mock()
    scenario = AtomicSkillScenario()
    scenario.simulation = simulation
    scenario._runtime_entities_created = True
    track = Mock(
        config={
            "skills": [{"id": "hand_over"}],
            "grippers": [
                {"control_part": "left_eef", "open_qpos": [0.0], "grasp_qpos": [0.03]},
                {"control_part": "right_eef", "open_qpos": [0.0], "grasp_qpos": [0.04]},
            ],
        }
    )

    scenario.configure_runtime(simulation, robot, Mock(), track, "left_arm")

    assert list(scenario._configured_grippers()) == ["left_eef", "right_eef"]
    assert scenario.end_effector_part == "left_eef"
    assert scenario._gripper_states["right_eef"][1].tolist() == [pytest.approx(0.04)]

    target = _pose(0.0).unsqueeze(0)
    scenario._case_providers["c"] = create_atomic_skill_provider("hand_over")
    scenario._cases_by_id["c"] = replace(
        _case("hand_over"),
        case_id="c",
        case_parameters={
            "frozen_grasps": {
                "left_eef": [target.tolist(), _pose(0.1).unsqueeze(0).tolist()],
                "right_eef": [target.tolist(), _pose(0.2).unsqueeze(0).tolist()],
            }
        },
    )
    motion_generator = Mock(robot=robot, device=CPU)
    motion_generator.planner.cfg.planner_type = "stub"
    adapter = Mock()
    adapter.require_motion_generator.return_value = motion_generator

    scenario.prepare_planner(adapter, Mock())

    engine = scenario.require_engine()
    assert set(engine.control_profiles) == {"left_eef", "right_eef"}
    assert all(
        isinstance(engine.grasp_pose_generators[part], ActiveCaseGraspPoseGenerator)
        for part in ("left_eef", "right_eef")
    )


class _CountingSampler(ParallelJawGraspPoseGenerator):
    """Report the sample count it was built with instead of sampling."""

    def __init__(
        self, n_sample: int, *, gripper_model: ParallelJawGripperModelCfg | None = None
    ) -> None:
        super().__init__(gripper_model or ParallelJawGripperModelCfg())
        self.n_sample = n_sample

    def get_valid_grasp_poses(self, **kwargs: object) -> list:
        return []

    def get_best_grasp_poses(self, **kwargs: object) -> tuple:
        return ()

    def get_dual_arm_valid_grasp_poses(self, **kwargs: object) -> list:
        return [self.n_sample]


@pytest.mark.parametrize("model_id", [None, "franka_panda_hand"])
def test_each_self_sampling_case_compiles_with_its_own_sample_count(
    monkeypatch, model_id
):
    monkeypatch.setattr(atomic_skill, "parallel_jaw_sampler", _CountingSampler)
    robot = Mock(device=CPU)
    robot.control_parts = {"left_eef": object(), "right_eef": object()}
    robot.get_joint_ids.side_effect = lambda name: {
        "left_eef": [12],
        "right_eef": [13],
    }[name]
    robot.get_qpos_limits.side_effect = lambda name: torch.tensor([[[0.0, 0.04]]])
    scenario = AtomicSkillScenario()
    scenario.track = SimpleNamespace(config={"grasp_model": model_id})
    scenario.robot = robot
    scenario.simulation = Mock(sim_config=Mock(physics_dt=0.01))
    scenario._gripper_states = {
        part: (torch.tensor([0.0]), torch.tensor([0.026]))
        for part in ("left_eef", "right_eef")
    }
    provider = Mock(
        samples_grasps=True,
        requires_gripper=True,
        skill_id="coordinated_pickment",
    )
    provider.frozen_grasps.return_value = {}
    provider.prefix_invocations.return_value = ()
    provider.build_invocation.return_value = Mock(
        motion_policy=Mock(strategy="motion_gen")
    )
    cases = {}
    for name, count in (("small", 2_000), ("large", 10_000)):
        cases[name] = replace(
            _case("coordinated_pickment"),
            case_id=name,
            case_parameters={"grasp_sample_count": count},
        )
        scenario._case_providers[name] = provider
        scenario._cases_by_id[name] = cases[name]
    motion_generator = Mock(robot=robot, device=CPU)
    motion_generator.planner.cfg.planner_type = "stub"
    adapter = Mock()
    adapter.require_motion_generator.return_value = motion_generator
    adapter.atomic_plan_options.return_value = None

    scenario.prepare_planner(adapter, Mock())

    sampler = scenario.require_engine().grasp_pose_generators["left_eef"]
    assert isinstance(sampler, CaseSampleCountGraspPoseGenerator)
    if model_id is not None:
        assert sampler.gripper_model.model_id == model_id
        assert sampler.gripper_model.max_opening_width == pytest.approx(0.08)
    assert scenario.require_engine().grasp_pose_generators["right_eef"] is sampler
    engine = Mock(motion_generator=motion_generator)
    engine.compile.side_effect = lambda *args, **kwargs: (
        sampler.get_dual_arm_valid_grasp_poses()[0]
    )
    scenario._engine = engine

    assert scenario.plan_case(adapter, cases["small"]) == 2_000
    assert scenario.plan_case(adapter, cases["large"]) == 10_000
    # Outside a case compile the sampler falls back to the largest count.
    assert sampler.get_dual_arm_valid_grasp_poses() == [10_000]


def test_scenario_rejects_single_and_multiple_gripper_declarations_together():
    robot = Mock(device=CPU)
    robot.get_qpos.return_value = torch.zeros(1, 8)
    scenario = AtomicSkillScenario()
    scenario.simulation = Mock()
    scenario._runtime_entities_created = True
    track = Mock(
        config={
            "skills": [{"id": "hand_over"}],
            "gripper": {"control_part": "eef", "grasp_qpos": [0.04]},
            "grippers": [{"control_part": "eef", "grasp_qpos": [0.04]}],
        }
    )

    with pytest.raises(ValueError, match="either gripper or grippers"):
        scenario.configure_runtime(scenario.simulation, robot, Mock(), track, "arm")


def test_fixtures_are_static_and_scene_objects_stay_in_the_case():
    simulation = Mock(device=CPU, num_envs=1)
    entities = {}

    def add_rigid_object(cfg):
        entity = Mock()
        entities[cfg.uid] = (cfg, entity)
        return entity

    simulation.add_rigid_object.side_effect = add_rigid_object
    scenario = AtomicSkillScenario()
    track = Mock(
        config={
            "fixtures": [{"id": "table", "kind": "cube", "size": [1.0, 1.0, 0.02]}],
            "objects": [
                {"id": "bread", "kind": "cube"},
                {"id": "pan", "kind": "cube"},
                {"id": "spare", "kind": "cube"},
            ],
        }
    )

    scenario.create_runtime_entities(simulation, Mock(), track)

    table_cfg, table = entities["atomic_benchmark_table"]
    assert table_cfg.body_type == "static"
    assert entities["atomic_benchmark_bread"][0].body_type == "dynamic"
    assert set(scenario._fixtures) == {"table"}
    assert "table" not in scenario._objects

    robot = Mock(device=CPU)
    pan_pose = _pose(0.3, 0.1, 0.7).unsqueeze(0)
    case = replace(
        _case("coordinated_placement"),
        object_id="bread",
        case_parameters={
            "object_initial_pose": _pose(0.1).unsqueeze(0).tolist(),
            "scene_object_poses": {"pan": pan_pose.tolist()},
        },
    )

    scenario.reset_case(simulation, robot, case, "left_arm")

    pan = scenario._objects["pan"].entity
    spare = scenario._objects["spare"].entity
    assert all(
        torch.equal(call.args[0], pan_pose)
        for call in pan.set_local_pose.call_args_list
    )
    assert pan.set_local_pose.call_count == 2
    assert spare.set_local_pose.call_args.args[0][0, 0, 3].item() >= 8.0
    table.set_local_pose.assert_not_called()


def test_reset_case_restores_every_row_of_a_frozen_placement_scene():
    """A replay restores the placing and support pose batches exactly."""
    batch_size = 4
    placing_pose = torch.eye(4).repeat(batch_size, 1, 1)
    support_pose = torch.eye(4).repeat(batch_size, 1, 1)
    placing_pose[:, 0, 3] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    support_pose[:, 1, 3] = torch.tensor([-0.1, -0.2, -0.3, -0.4])
    placing = Mock(object_id="bread", entity=Mock())
    support = Mock(object_id="pan", entity=Mock())
    scenario = AtomicSkillScenario()
    scenario._objects = {"bread": placing, "pan": support}
    case = Mock(
        full_start_qpos=torch.zeros(batch_size, 12),
        object_id="bread",
        case_parameters={
            "object_initial_pose": placing_pose.tolist(),
            "scene_object_poses": {"pan": support_pose.tolist()},
            "pre_action_settle_steps": 10,
        },
    )

    scenario.reset_case(Mock(), Mock(device=CPU), case, "left_arm")

    torch.testing.assert_close(
        placing.entity.set_local_pose.call_args.args[0], placing_pose
    )
    torch.testing.assert_close(
        support.entity.set_local_pose.call_args.args[0], support_pose
    )


def test_dual_arm_provider_builds_both_arms_and_grippers():
    suite = load_suite("atomic_dual_ur5_pgi_hand_over")
    cfg = create_robot_provider(suite.robot).build_cfg()

    assert {"left_arm", "right_arm", "left_eef", "right_eef"} <= set(cfg.control_parts)
    assert cfg.uid == "benchmark_dual_ur5_hand_over"
    assert cfg.solver_cfg["left_arm"].tcp[2][3] == pytest.approx(0.15)
    assert create_robot_provider(suite.robot).control_part == "left_arm"


def test_native_panda_dual_provider_preserves_real_fingers_and_stock_tcp(tmp_path):
    """Check actual assembled geometry/kinematics, including the native mimic."""
    from embodichain.utils.string import resolve_matching_names
    from embodichain_robots import FrankaPandaCfg

    suite = load_suite("atomic_dual_franka_pgi_hand_over")
    values = {
        **suite.robot.config,
        "gripper_type": "panda_hand",
        "uid": "native_panda_dual_test",
        "urdf_name": "native_panda_dual_test",
    }
    values.pop("tcp_z")
    cfg = create_robot_provider(replace(suite.robot, config=values)).build_cfg()
    assert Path(cfg.urdf_cfg.fpath).name == "native_panda_dual_test.urdf"
    cfg.urdf_cfg.fpath = str(tmp_path / "native_panda_dual.urdf")
    assembled_path = cfg.urdf_cfg.assemble_urdf()
    tree = ET.parse(assembled_path)
    links = [link.get("name") for link in tree.findall("link")]
    joints = {
        joint.get("name"): joint
        for joint in tree.findall("joint")
        if joint.get("type") != "fixed"
    }
    assert len(joints) == 18
    assert len(cfg.init_qpos) == 18
    assert cfg.init_qpos[-4:] == [0.04] * 4
    assert [Path(c["urdf_path"]).name for c in cfg.urdf_cfg.components.values()] == [
        "PandaWithHand.urdf",
        "PandaWithHand.urdf",
    ]
    contact_links = resolve_matching_names(
        cfg.link_attrs["newton_gripper_contacts"].link_names_expr, links
    )[1]
    assert set(contact_links) == {
        f"{side}_fr3_{finger}finger"
        for side in ("left", "right")
        for finger in ("left", "right")
    }
    for rules in (
        cfg.joint_drive_props.stiffness,
        cfg.joint_drive_props.damping,
        cfg.joint_drive_props.max_effort,
    ):
        resolve_matching_names(list(rules), list(joints))

    stock = FrankaPandaCfg.from_dict({})
    stock_chain = stock.build_pk_serial_chain(device=CPU)["arm"]
    qpos = torch.tensor(
        [stock.init_qpos[:7], [0.3, -0.2, 0.1, -2.0, 0.2, 2.5, 0.4]],
        dtype=torch.float32,
    )
    expected_fk = stock_chain.forward_kinematics(qpos).get_matrix()
    chains = cfg.build_pk_serial_chain(device=CPU)
    for side in ("left", "right"):
        arm = f"{side}_arm"
        fingers = [f"{side}_fr3_finger_joint{i}" for i in (1, 2)]
        assert cfg.control_parts[f"{side}_eef"] == fingers
        assert joints[fingers[1]].find("mimic").get("joint") == fingers[0]
        limits = []
        for name in fingers:
            assert joints[name].get("type") == "prismatic"
            limit = joints[name].find("limit")
            limits.append([float(limit.get("lower")), float(limit.get("upper"))])
        robot = Mock(device=CPU)
        robot.get_qpos_limits.return_value = torch.tensor([limits])
        _, opened, closed = AtomicSkillScenario._resolve_gripper(
            robot,
            {
                "control_part": f"{side}_eef",
                "open_qpos": [0.04, 0.04],
                "grasp_qpos": [0.0, 0.0],
            },
        )
        torch.testing.assert_close(opened, torch.full((2,), 0.04))
        torch.testing.assert_close(closed, torch.zeros(2))
        solver_cfg = cfg.solver_cfg[arm]
        assert solver_cfg.end_link_name == f"{side}_fr3_hand_tcp"
        np.testing.assert_allclose(solver_cfg.tcp, np.eye(4))
        solver_cfg.urdf_path = assembled_path
        solver_cfg.joint_names = cfg.control_parts[arm]
        solver = solver_cfg.init_solver(device=CPU)
        torch.testing.assert_close(solver.get_fk(qpos), expected_fk)
        torch.testing.assert_close(
            chains[arm].forward_kinematics(qpos).get_matrix(), expected_fk
        )

    # Hardware replacement retains both arm base transforms in the world.
    root = np.eye(4)
    root[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    root[:3, 3] = cfg.init_pos
    for side, y in (("left", 0.3), ("right", -0.3)):
        world_from_arm = root @ cfg.urdf_cfg.components[f"{side}_arm"]["transform"]
        expected = np.diag([-1.0, -1.0, 1.0, 1.0])
        expected[:3, 3] = [0.5, y, 0.5]
        np.testing.assert_allclose(world_from_arm, expected, atol=1e-6)


@pytest.mark.parametrize(
    "overrides, message",
    [({"robot_type": "ur5"}, "requires robot_type"), ({"tcp_z": 0.15}, "omit tcp_z")],
)
def test_native_panda_dual_rejects_incompatible_arm_or_tool_override(
    overrides, message
):
    suite = load_suite("atomic_dual_franka_pgi_hand_over")
    values = {**suite.robot.config, "gripper_type": "panda_hand"}
    values.pop("tcp_z")
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        create_robot_provider(replace(suite.robot, config=values)).build_cfg()


def test_native_grasp_sampling_maps_closing_axis_once_before_frozen_replay(monkeypatch):
    """Every sampler API preserves metadata and native replay retains the pose."""
    from embodichain.toolkits.graspkit import get_parallel_jaw_gripper_model
    from embodichain_benchmarks.motion_generation import _atomic_fixtures

    canonical = _pose(0.2, 0.3, 0.4).unsqueeze(0)
    canonical[:, :3, :3] = axis_angle_to_rotation_matrix(torch.tensor([0.2, -0.1, 0.3]))
    before = canonical.clone()
    costs, widths, success = (
        torch.tensor([0.25]),
        torch.tensor([0.03]),
        torch.tensor([True]),
    )
    model = get_parallel_jaw_gripper_model("franka_panda_hand")
    source = Mock(gripper_model=model)
    source.get_valid_grasp_poses.return_value = [(canonical, costs)]
    source.get_grasp_candidates.return_value = [(canonical, widths, costs)]
    source.get_best_grasp_poses.return_value = (success, canonical, widths)
    source.get_dual_arm_valid_grasp_poses.return_value = [
        {
            part: {"grasp_poses": canonical, "open_lengths": widths, "costs": costs}
            for part in ("left", "right")
        },
        None,
    ]
    monkeypatch.setattr(
        _atomic_fixtures, "AntipodalGraspPoseGenerator", Mock(return_value=source)
    )
    sampler = _atomic_fixtures.create_parallel_jaw_grasp_pose_generator(
        n_sample=32, force_refresh=False, gripper_model=model
    )
    valid, actual_costs = sampler.get_valid_grasp_poses()[0]
    candidates, actual_widths, candidate_costs = sampler.get_grasp_candidates()[0]
    actual_success, best, best_widths = sampler.get_best_grasp_poses()
    paired = sampler.get_dual_arm_valid_grasp_poses()
    assert paired[1] is None
    for actual in (
        valid,
        candidates,
        best,
        *(v["grasp_poses"] for v in paired[0].values()),
    ):
        # Native finger translation follows TCP +Y; sampler closing follows +X.
        torch.testing.assert_close(actual[:, :3, 1], canonical[:, :3, 0])
        torch.testing.assert_close(actual[:, :3, 2], canonical[:, :3, 2])
        torch.testing.assert_close(actual[:, :3, 3], canonical[:, :3, 3])
    for actual in (actual_costs, candidate_costs):
        torch.testing.assert_close(actual, costs)
    for actual in (actual_widths, best_widths):
        torch.testing.assert_close(actual, widths)
    torch.testing.assert_close(actual_success, success)
    torch.testing.assert_close(canonical, before, rtol=0, atol=0)

    frozen = ActiveCaseGraspPoseGenerator(sampler)
    object_pose = torch.eye(4).unsqueeze(0)
    frozen.activate(object_pose, best)
    _, replayed, _ = frozen.get_best_grasp_poses(
        mesh_vertices=torch.zeros(3, 3),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        obj_poses=object_pose,
        approach_direction=canonical[:, :3, 2],
    )
    torch.testing.assert_close(replayed, best, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("suite_name", "skill_id"),
    [
        ("atomic_dual_ur5_pgi_hand_over", "hand_over"),
        ("atomic_dual_ur5_pgi_coordinated_pickment", "coordinated_pickment"),
        ("atomic_dual_ur5_pgi_coordinated_placement", "coordinated_placement"),
    ],
)
def test_dual_arm_suites_plan_with_toppra_and_declare_both_grippers(
    suite_name: str, skill_id: str
):
    suite = load_suite(suite_name)
    track = suite.enabled_tracks()[0]

    assert [spec.adapter for spec in suite.planners if spec.enabled] == ["toppra"]
    assert suite.robot.provider == "dual_arm"
    assert [item["id"] for item in track.config["skills"]] == [skill_id]
    assert [item["control_part"] for item in track.config["grippers"]] == [
        "left_eef",
        "right_eef",
    ]
    assert track.config["physics"]["steps_per_waypoint"] == 16
    assert track.config["fixtures"]


@pytest.mark.parametrize("skill_id", ["hand_over", "coordinated_placement"])
def test_franka_dual_suites_use_independent_seven_joint_arms(skill_id: str) -> None:
    """Decode the published suites through their production robot provider."""
    suite = load_suite(f"atomic_dual_franka_pgi_{skill_id}")
    cfg = create_robot_provider(suite.robot).build_cfg()
    for part in ("left_arm", "right_arm"):
        assert len(cfg.control_parts[part]) == 7
        assert cfg.solver_cfg[part].root_link_name.startswith(part.split("_")[0])
    assert set(cfg.control_parts["left_arm"]).isdisjoint(cfg.control_parts["right_arm"])
    assert {spec.id for spec in suite.planners if spec.enabled} == {
        "toppra",
        "curobo",
        "nmg",
        "nmg_retimed",
    }
    track = suite.enabled_tracks()[0]
    assert [skill["id"] for skill in track.config["skills"]] == [skill_id]
    assert track.config["batch_sizes"] == [1]
