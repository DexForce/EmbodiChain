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

"""Pure-logic tests for the planner-oriented Atomic Task benchmark track."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ArticulationAffordanceGeometry,
    MotionPolicy,
)
from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory, TimedTrajectory
from embodichain.utils.math import axis_angle_to_rotation_matrix
from scripts.benchmark.motion_generation.aggregation import aggregate_results
from scripts.benchmark.motion_generation.artifacts import (
    TrialJsonlWriter,
    write_case_manifest,
)
from scripts.benchmark.motion_generation.config import load_suite
from scripts.benchmark.motion_generation.contracts import ReplayObservationTrace
from scripts.benchmark.motion_generation.models import (
    AlgorithmRole,
    BenchmarkCase,
    CaseOutcome,
    PlannerMetadata,
    TrialPhase,
    TrialRecord,
)
from scripts.benchmark.motion_generation.registry import create_robot_provider
from scripts.benchmark.motion_generation.runner import BenchmarkRunner
from scripts.benchmark.motion_generation import robots as _robots  # noqa: F401
from scripts.benchmark.motion_generation.scenarios.atomic_objects import (
    atomic_object_kind_names,
    create_atomic_object,
)
from scripts.benchmark.motion_generation.scenarios.atomic_task import (
    AtomicTaskScenario,
    _ExecutionObservation,
    _atomic_case_id,
    _articulation_effect_success,
    _benchmark_motion_attribution,
    _benchmark_motion_policy_strategy,
    _canonical_case_qpos,
    _case_generation_seed,
    _randomization_parameters,
    _seeded_jitter,
    _snapshot_case_qpos,
    atomic_skill_provider_names,
    create_atomic_skill_provider,
)
from scripts.benchmark.motion_generation.scenarios.base import ScenarioProvider
from scripts.benchmark.motion_generation.scenarios.free_space import FreeSpaceScenario
from scripts.benchmark.motion_generation.video import (
    VideoRecordCfg,
    build_video_path,
    record_with_window,
    should_record_case,
    summarize_video_recording,
    video_cfg_from_args,
)


def _atomic_case() -> BenchmarkCase:
    target = torch.eye(4).reshape(1, 1, 4, 4)
    return BenchmarkCase(
        suite_version="atomic_test_v1",
        track="atomic-task",
        scenario_id="move_end_effector",
        case_id="atomic-task:move_end_effector:simple:s11",
        seed=11,
        batch_size=1,
        num_waypoints=1,
        path_shape="robot_relative_waypoints",
        start_state_bin="pre_action",
        start_qpos=torch.zeros(1, 7),
        target_waypoints=target,
        reference_qpos=torch.zeros(1, 1, 7),
        robot_id="franka_pgi",
        skill_id="move_end_effector",
        task_difficulty="simple",
        primary_success="task_success",
        full_start_qpos=torch.zeros(1, 9),
        case_parameters={"sample_count": 80, "target_offsets_m": [[0.1, 0.0, 0.0]]},
    )


def _atomic_outcome() -> CaseOutcome:
    return CaseOutcome(
        env_index=0,
        planning_success=True,
        finite=True,
        ordered_waypoints_reached=True,
        motion_valid=True,
        completed_waypoint_ratio=1.0,
        final_translation_err_mm=1.0,
        final_rotation_err_deg=1.0,
        waypoint_translation_err_mm_mean=1.0,
        waypoint_translation_err_mm_p95=1.0,
        waypoint_translation_err_mm_max=1.0,
        waypoint_rotation_err_deg_mean=1.0,
        waypoint_rotation_err_deg_p95=1.0,
        waypoint_rotation_err_deg_max=1.0,
        joint_limit_violation=False,
        max_normalized_joint_violation=0.0,
        joint_path_length_rad=0.2,
        cartesian_path_length_m=0.1,
        path_efficiency=1.0,
        execution_success=True,
        task_success=True,
        task_completion_time_s=1.5,
        joint_tracking_rmse_rad=0.002,
        replan_count=0,
        articulation_joint_initial=0.0,
        articulation_joint_final=0.001,
        articulation_joint_delta=0.001,
        articulation_joint_peak_signed_delta=0.006,
    )


def test_atomic_case_identity_includes_simulator_batch_size() -> None:
    case_b1 = _atomic_case_id("atomic-task", "pick_up", "cube", batch_size=1, seed=11)
    case_b8 = _atomic_case_id("atomic-task", "pick_up", "cube", batch_size=8, seed=11)

    assert case_b1 == "atomic-task:pick_up:cube:b1:s11"
    assert case_b8 == "atomic-task:pick_up:cube:b8:s11"
    assert case_b1 != case_b8


@pytest.mark.parametrize(
    ("skill_id", "parameters", "expected_ids"),
    [
        (
            "pick_up",
            {},
            ("pickup_grasp", "pickup_lift", "pickup_settle"),
        ),
        (
            "move_held_object",
            {},
            ("held_object_transport", "held_object_settle"),
        ),
        ("place", {}, ("place_release", "place_settle")),
        ("press", {}, ("press_actuation",)),
        ("slide", {"direction": "pull"}, ("slide_actuation", "slide_release")),
        ("twist", {}, ("twist_actuation", "twist_release")),
        ("open_door", {}, ("door_actuation", "door_release")),
        ("pour", {}, ("pour_actuation", "pour_settle")),
        (
            "axis_align",
            {},
            ("axis_align_grasp", "axis_align_actuation", "axis_align_settle"),
        ),
        (
            "hand_over",
            {},
            (
                "handover_pickup",
                "handover_transfer",
                "handover_exchange",
                "handover_settle",
            ),
        ),
        (
            "push_object",
            {},
            ("push_contact", "push_actuation", "push_settle"),
        ),
        (
            "coordinated_pickment",
            {},
            (
                "dual_pick_grasp",
                "dual_pick_lift",
                "dual_pick_move",
                "dual_pick_settle",
            ),
        ),
        (
            "coordinated_placement",
            {"support_object_id": "support"},
            (
                "coordinated_place_alignment",
                "coordinated_place_release",
                "coordinated_place_settle",
            ),
        ),
    ],
)
def test_migrated_skill_providers_declare_replay_observation_windows(
    skill_id: str,
    parameters: dict[str, object],
    expected_ids: tuple[str, ...],
) -> None:
    segments = {
        name: SimpleNamespace(start=index * 5, stop=(index + 1) * 5)
        for index, name in enumerate(
            (
                "press",
                "pull",
                "open",
                "twist",
                "release",
                "pour",
                "manipulate",
                "pickup_close",
                "pickup_transport",
                "receive_close",
                "handover_release",
                "close",
                "lift",
                "transport",
                "contact",
                "push",
                "move",
                "approach",
            )
        )
    }
    compiled = Mock(trajectory=SimpleNamespace(waypoint_count=60))
    compiled.segment.side_effect = lambda action, name: segments[name]
    case = replace(
        _atomic_case(),
        scenario_id=skill_id,
        skill_id=skill_id,
        case_parameters=parameters,
    )

    windows = create_atomic_skill_provider(skill_id).replay_observation_windows(
        compiled, case
    )

    assert tuple(window.observation_id for window in windows) == expected_ids


def test_every_atomic_skill_provider_uses_the_versioned_physical_evaluator() -> None:
    assert all(
        create_atomic_skill_provider(skill_id).physical_evaluator() is not None
        for skill_id in atomic_skill_provider_names()
    )


def test_move_end_effector_provider_uses_settled_tcp_standard_evidence() -> None:
    case = replace(
        _atomic_case(),
        batch_size=2,
        target_waypoints=torch.eye(4).repeat(2, 1, 1, 1),
    )
    final_tcp = torch.eye(4).repeat(2, 1, 1)
    final_tcp[:, 0, 3] = torch.tensor([0.002, 0.004])
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=final_tcp,
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
    )
    outcomes = (_atomic_outcome(), replace(_atomic_outcome(), env_index=1))

    evaluator = create_atomic_skill_provider("move_end_effector").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, outcomes)

    assert [result.task_success for result in results] == [True, False]
    assert results[0].protocol_version == "atomic-skill-v1"
    assert results[0].stages[0].evidence_source == (
        "simulation_state:terminal_tcp_pose"
    )


def test_move_end_effector_rejects_wrong_terminal_orientation() -> None:
    case = replace(
        _atomic_case(),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
    )
    final_tcp = torch.eye(4).reshape(1, 4, 4)
    final_tcp[:, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[0.0, 0.0, math.pi]])
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(1, dtype=torch.bool),
        final_tcp_pose=final_tcp,
        joint_tracking_rmse_rad=torch.zeros(1),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
    )

    evaluator = create_atomic_skill_provider("move_end_effector").physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_atomic_outcome(),))[0]

    assert not result.task_success
    assert result.failure_stage == "reached"


def test_move_end_effector_keeps_non_finite_failures_row_local() -> None:
    case = replace(
        _atomic_case(),
        batch_size=2,
        target_waypoints=torch.eye(4).repeat(2, 1, 1, 1),
    )
    final_tcp = torch.eye(4).repeat(2, 1, 1)
    final_tcp[1, 0, 3] = torch.nan
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=final_tcp,
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
    )

    evaluator = create_atomic_skill_provider("move_end_effector").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(
        case,
        observation,
        (_atomic_outcome(), replace(_atomic_outcome(), env_index=1)),
    )

    assert results[0].task_success
    assert results[1].failure_code == "evidence_unavailable"


def test_move_joints_provider_uses_l2_settled_joint_error() -> None:
    case = replace(
        _atomic_case(),
        batch_size=2,
        scenario_id="move_joints",
        skill_id="move_joints",
        case_parameters={
            "sample_count": 80,
            "target_qpos": [[[0.0, 0.0]], [[0.0, 0.0]]],
        },
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        final_arm_qpos=torch.tensor([[0.003, 0.003], [0.004, 0.004]]),
    )
    outcomes = (_atomic_outcome(), replace(_atomic_outcome(), env_index=1))

    evaluator = create_atomic_skill_provider("move_joints").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, outcomes)

    assert [result.task_success for result in results] == [True, False]
    assert results[1].failure_code == "task_goal_miss"
    assert results[0].stages[0].measurements["joint_error_l2_rad"] == pytest.approx(
        2**0.5 * 0.003
    )


def test_move_joints_missing_terminal_observation_is_unavailable() -> None:
    case = replace(
        _atomic_case(),
        scenario_id="move_joints",
        skill_id="move_joints",
        case_parameters={"target_qpos": [[[0.0, 0.0]]]},
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(1, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).reshape(1, 4, 4),
        joint_tracking_rmse_rad=torch.zeros(1),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
    )

    evaluator = create_atomic_skill_provider("move_joints").physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_atomic_outcome(),))[0]

    assert result.stages[0].status == "unavailable"
    assert result.failure_code == "evidence_unavailable"


def test_pick_up_provider_uses_grasp_lift_and_retention_windows() -> None:
    case = replace(
        _atomic_case(),
        batch_size=2,
        scenario_id="pick_up",
        skill_id="pick_up",
        case_parameters={"object_initial_pose": torch.eye(4).repeat(2, 1, 1).tolist()},
    )
    lift = torch.eye(4).repeat(2, 2, 1, 1)
    lift[-1, :, 2, 3] = torch.tensor([0.05, 0.03])
    held = torch.full((2, 2, 1), 0.5)
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "pickup_grasp": ReplayObservationTrace(gripper_qpos={"hand": held}),
            "pickup_lift": ReplayObservationTrace(object_pose=lift, tcp_pose=lift),
            "pickup_settle": ReplayObservationTrace(
                object_pose=lift[-1:].repeat(2, 1, 1, 1),
                tcp_pose=lift[-1:].repeat(2, 1, 1, 1),
                gripper_qpos={"hand": held},
            ),
        },
    )

    evaluator = create_atomic_skill_provider("pick_up").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(
        case,
        observation,
        (_atomic_outcome(), replace(_atomic_outcome(), env_index=1)),
    )

    assert [result.task_success for result in results] == [True, False]
    assert results[1].failure_stage == "lifted"
    assert results[0].stages[0].evidence_source == (
        "measured_proxy:gripper_joint_state"
    )


def test_move_held_object_provider_uses_measured_transport_pose() -> None:
    target = torch.eye(4).repeat(2, 1, 1)
    target[:, 2, 3] = 0.1
    transport = target.unsqueeze(0).repeat(2, 1, 1, 1)
    poses = target.unsqueeze(0).repeat(2, 1, 1, 1)
    poses[-1, :, 0, 3] = torch.tensor([0.005, 0.02])
    case = replace(
        _atomic_case(),
        batch_size=2,
        scenario_id="move_held_object",
        skill_id="move_held_object",
        case_parameters={
            "object_initial_pose": torch.eye(4).repeat(2, 1, 1).tolist(),
            "target_object_pose": target.tolist(),
        },
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "held_object_transport": ReplayObservationTrace(object_pose=transport),
            "held_object_settle": ReplayObservationTrace(
                object_pose=poses,
                gripper_qpos={"hand": torch.full((2, 2, 1), 0.5)},
            ),
        },
    )

    evaluator = create_atomic_skill_provider("move_held_object").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(
        case,
        observation,
        (_atomic_outcome(), replace(_atomic_outcome(), env_index=1)),
    )

    assert [result.task_success for result in results] == [True, False]
    assert results[1].failure_stage == "settled"


def test_move_held_object_requires_terminal_retention() -> None:
    target = torch.eye(4).reshape(1, 4, 4)
    target[:, 2, 3] = 0.1
    poses = target.unsqueeze(0).repeat(2, 1, 1, 1)
    case = replace(
        _atomic_case(),
        scenario_id="move_held_object",
        skill_id="move_held_object",
        case_parameters={
            "object_initial_pose": torch.eye(4).reshape(1, 4, 4).tolist(),
            "target_object_pose": target.tolist(),
        },
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(1, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).reshape(1, 4, 4),
        joint_tracking_rmse_rad=torch.zeros(1),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "held_object_transport": ReplayObservationTrace(object_pose=poses),
            "held_object_settle": ReplayObservationTrace(
                object_pose=poses,
                gripper_qpos={"hand": torch.ones(2, 1, 1)},
            ),
        },
    )

    evaluator = create_atomic_skill_provider("move_held_object").physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_atomic_outcome(),))[0]

    assert not result.task_success
    assert result.failure_stage == "held"
    assert result.failure_code == "object_dropped"


def test_place_provider_separates_release_goal_and_stability() -> None:
    target = torch.eye(4).repeat(2, 1, 1)
    target[:, 2, 3] = 0.02
    settle = target.unsqueeze(0).repeat(3, 1, 1, 1)
    settle[-1, :, 0, 3] = torch.tensor([0.005, 0.02])
    released = torch.zeros(2, 2, 1)
    case = replace(
        _atomic_case(),
        batch_size=2,
        scenario_id="place",
        skill_id="place",
        case_parameters={
            "object_initial_pose": torch.eye(4).repeat(2, 1, 1).tolist(),
            "target_object_pose": target.tolist(),
        },
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "place_release": ReplayObservationTrace(gripper_qpos={"hand": released}),
            "place_settle": ReplayObservationTrace(object_pose=settle),
        },
    )

    evaluator = create_atomic_skill_provider("place").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(
        case,
        observation,
        (_atomic_outcome(), replace(_atomic_outcome(), env_index=1)),
    )

    assert [result.task_success for result in results] == [True, False]
    assert results[1].failure_stage == "placed"


def test_place_rejects_wrong_terminal_height_and_orientation() -> None:
    target = torch.eye(4).repeat(2, 1, 1)
    settle = target.unsqueeze(0).repeat(2, 1, 1, 1)
    settle[:, 0, 2, 3] = 0.1
    settle[:, 1, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[math.pi / 2, 0.0, 0.0]])
    )
    released = torch.zeros(2, 2, 1)
    case = replace(
        _atomic_case(),
        batch_size=2,
        scenario_id="place",
        skill_id="place",
        case_parameters={
            "object_initial_pose": torch.eye(4).repeat(2, 1, 1).tolist(),
            "target_object_pose": target.tolist(),
        },
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "place_release": ReplayObservationTrace(gripper_qpos={"hand": released}),
            "place_settle": ReplayObservationTrace(object_pose=settle),
        },
    )

    evaluator = create_atomic_skill_provider("place").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(
        case,
        observation,
        (_atomic_outcome(), replace(_atomic_outcome(), env_index=1)),
    )

    assert [result.task_success for result in results] == [False, False]
    assert [result.failure_stage for result in results] == ["placed", "placed"]


def test_press_provider_scores_only_the_requested_actuation_trace() -> None:
    case = replace(
        _atomic_case(),
        batch_size=2,
        scenario_id="press",
        skill_id="press",
        case_parameters={
            "articulation_joint_stroke": 0.006,
            "effect_joint_sign": 1.0,
        },
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        replay_traces={
            "press_actuation": ReplayObservationTrace(
                articulation_joint=torch.tensor(
                    [[0.0, 0.0], [0.003, 0.003], [0.005, 0.004]]
                )
            )
        },
    )
    outcomes = (_atomic_outcome(), replace(_atomic_outcome(), env_index=1))

    evaluator = create_atomic_skill_provider("press").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, outcomes)

    assert [result.task_success for result in results] == [True, False]
    assert results[1].failure_stage == "pressed"


def test_slide_provider_uses_actuation_and_release_windows() -> None:
    case = replace(
        _atomic_case(),
        batch_size=2,
        scenario_id="slide",
        skill_id="slide",
        case_parameters={
            "translation_distance_m": 0.1,
            "direction": "pull",
            "effect_joint_sign": 1.0,
        },
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "slide_actuation": ReplayObservationTrace(
                articulation_joint=torch.tensor([[0.0, 0.0], [0.1, 0.07]])
            ),
            "slide_release": ReplayObservationTrace(
                gripper_qpos={"hand": torch.tensor([[[1.0], [1.0]], [[0.0], [0.0]]])}
            ),
        },
    )
    outcomes = (_atomic_outcome(), replace(_atomic_outcome(), env_index=1))

    evaluator = create_atomic_skill_provider("slide").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, outcomes)

    assert [result.task_success for result in results] == [True, False]
    assert results[0].stages[-1].stage_id == "released"
    assert results[1].failure_stage == "slid"


def test_slide_rejects_opposite_joint_direction() -> None:
    case = replace(
        _atomic_case(),
        scenario_id="slide",
        skill_id="slide",
        case_parameters={
            "translation_distance_m": 0.1,
            "direction": "pull",
            "effect_joint_sign": 1.0,
        },
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(1, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).reshape(1, 4, 4),
        joint_tracking_rmse_rad=torch.zeros(1),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "slide_actuation": ReplayObservationTrace(
                articulation_joint=torch.tensor([[0.0], [-0.1]])
            ),
            "slide_release": ReplayObservationTrace(
                gripper_qpos={"hand": torch.tensor([[[1.0]], [[0.0]]])}
            ),
        },
    )

    evaluator = create_atomic_skill_provider("slide").physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_atomic_outcome(),))[0]

    assert not result.task_success
    assert result.failure_stage == "grasped"


def test_twist_provider_scores_executed_tcp_rotation_not_knob_angle() -> None:
    case = replace(
        _atomic_case(),
        batch_size=2,
        scenario_id="twist",
        skill_id="twist",
        case_parameters={
            "twist_target_pose": torch.eye(4).repeat(2, 1, 1).tolist(),
            "twist_axis": [0.0, 0.0, 1.0],
            "twist_angle_rad": 0.5,
        },
    )
    tcp = torch.eye(4).repeat(2, 2, 1, 1)
    tcp[-1, :, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.2]])
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "twist_actuation": ReplayObservationTrace(
                tcp_pose=tcp,
                articulation_joint=torch.tensor([[0.0, 0.0], [0.1, 0.1]]),
            ),
            "twist_release": ReplayObservationTrace(
                gripper_qpos={"hand": torch.tensor([[[1.0], [1.0]], [[0.0], [0.0]]])}
            ),
        },
    )
    outcomes = (_atomic_outcome(), replace(_atomic_outcome(), env_index=1))

    evaluator = create_atomic_skill_provider("twist").physical_evaluator()
    assert evaluator is not None
    results = evaluator.evaluate(case, observation, outcomes)

    assert [result.task_success for result in results] == [True, False]
    assert results[1].failure_stage == "twisted"


def test_twist_rejects_opposite_tcp_rotation() -> None:
    case = replace(
        _atomic_case(),
        scenario_id="twist",
        skill_id="twist",
        case_parameters={
            "twist_target_pose": torch.eye(4).reshape(1, 4, 4).tolist(),
            "twist_axis": [0.0, 0.0, 1.0],
            "twist_angle_rad": -0.5,
        },
    )
    tcp = torch.eye(4).repeat(2, 1, 1, 1)
    tcp[-1, :, :3, :3] = axis_angle_to_rotation_matrix(torch.tensor([[0.0, 0.0, 0.5]]))
    observation = _ExecutionObservation(
        execution_success=torch.ones(1, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).reshape(1, 4, 4),
        joint_tracking_rmse_rad=torch.zeros(1),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "twist_actuation": ReplayObservationTrace(
                tcp_pose=tcp,
                articulation_joint=torch.tensor([[0.0], [0.1]]),
            ),
            "twist_release": ReplayObservationTrace(
                gripper_qpos={"hand": torch.tensor([[[1.0]], [[0.0]]])}
            ),
        },
    )

    evaluator = create_atomic_skill_provider("twist").physical_evaluator()
    assert evaluator is not None
    result = evaluator.evaluate(case, observation, (_atomic_outcome(),))[0]

    assert not result.task_success
    assert result.failure_stage == "twisted"


def _fake_articulation_geometry(*, joint_type: str) -> dict[str, torch.Tensor]:
    """Build non-degenerate link-local geometry for articulation providers."""
    target_points = torch.tensor(
        [
            [-0.01, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, -0.01, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, -0.01],
            [0.0, 0.0, 0.01],
        ],
        dtype=torch.float32,
    )
    non_target_points = torch.tensor(
        [
            [-0.002, 0.0, 0.018],
            [0.002, 0.0, 0.018],
            [0.0, -0.002, 0.018],
            [0.0, 0.002, 0.018],
        ],
        dtype=torch.float32,
    )
    kwargs: dict[str, torch.Tensor] = {}
    if joint_type == "prismatic":
        kwargs["prismatic_joint_axis"] = torch.tensor([0.0, 0.0, 1.0])
    elif joint_type == "revolute":
        kwargs["revolute_joint_axis"] = torch.tensor([0.0, 0.0, 1.0])
        kwargs["revolute_axis_origin"] = torch.zeros(3)
    else:
        raise ValueError(f"Unsupported fake articulation joint type: {joint_type}")
    return ArticulationAffordanceGeometry(
        target_link_point_cloud=target_points,
        articulation_point_cloud=torch.cat((target_points, non_target_points)),
        non_target_articulation_point_cloud=non_target_points,
        **kwargs,
    ).to_object_geometry()


def test_atomic_suite_is_franka_pgi_and_curobo_only():
    suite = load_suite("atomic_franka_pgi_curobo")

    assert suite.suite_version == "atomic_franka_pgi_curobo_smoke_v3"
    assert suite.robot.id == "franka_pgi"
    assert suite.robot.provider == "franka_pgi"
    assert [spec.id for spec in suite.planners if spec.enabled] == ["curobo"]
    assert [(track.id, track.scenario) for track in suite.enabled_tracks()] == [
        ("atomic-task", "atomic_task")
    ]
    skills = suite.enabled_tracks()[0].config["skills"]
    assert [item["id"] for item in skills] == [
        "move_end_effector",
        "move_joints",
        "pick_up",
        "move_held_object",
        "place",
        "press",
        "slide",
        "twist",
    ]
    skills_by_id = {item["id"]: item for item in skills}
    assert {
        key: skills_by_id["press"][key]
        for key in (
            "sample_count",
            "hand_interp_steps",
            "approach_distance_m",
            "press_distance_m",
        )
    } == {
        "sample_count": 256,
        "hand_interp_steps": 12,
        "approach_distance_m": 0.12,
        "press_distance_m": 0.03,
    }
    assert {
        key: skills_by_id["slide"][key]
        for key in (
            "sample_count",
            "hand_interp_steps",
            "approach_distance_m",
            "translation_distance_m",
            "direction",
        )
    } == {
        "sample_count": 140,
        "hand_interp_steps": 12,
        "approach_distance_m": 0.10,
        "translation_distance_m": 0.18,
        "direction": "pull",
    }
    assert {
        key: skills_by_id["twist"][key]
        for key in (
            "sample_count",
            "hand_interp_steps",
            "twist_waypoint_count",
            "pre_grasp_distance_m",
            "twist_angle_rad",
        )
    } == {
        "sample_count": 256,
        "hand_interp_steps": 20,
        "twist_waypoint_count": 8,
        "pre_grasp_distance_m": 0.12,
        "twist_angle_rad": pytest.approx(-0.7853981634),
    }
    gripper = suite.enabled_tracks()[0].config["gripper"]
    assert gripper == {
        "control_part": "eef",
        "open_qpos": [0.0],
        "grasp_qpos": [0.040],
    }
    articulations = {
        item["id"]: item for item in suite.enabled_tracks()[0].config["articulations"]
    }
    assert set(articulations) == {"microwave", "drawer"}
    assert articulations["microwave"] == {
        "id": "microwave",
        "asset_path": "Microwave/microwave.urdf",
        "position": [-1.0, -0.30, 0.4],
        "rotation_deg": [0.0, 0.0, 90.0],
        "init_qpos": [0.0, 0.0, 0.0, 0.0],
        "drive": {
            "stiffness": 0.001,
            "damping": 100.0,
            "max_effort": 0.01,
        },
        "fix_base": True,
        "settle_steps": 10,
    }
    assert articulations["drawer"] == {
        "id": "drawer",
        "asset_path": "Drawer/model_split_links_with_inertials.urdf",
        "position": [-1.1, 0.0, 0.0],
        "rotation_deg": [0.0, 0.0, 90.0],
        "init_qpos": [0.0],
        "drive": {"drive_type": "none"},
        "attrs": {"material_props": {"static_friction": 1.0, "dynamic_friction": 1.0}},
        "fix_base": True,
        "settle_steps": 10,
        "reset_settle_steps": 0,
    }
    assert {
        key: skills_by_id["press"][key]
        for key in ("articulation", "target_link", "target_joint")
    } == {
        "articulation": "microwave",
        "target_link": "button_link",
        "target_joint": "button_joint",
    }
    assert {
        key: skills_by_id["slide"][key]
        for key in ("articulation", "target_link", "target_joint")
    } == {
        "articulation": "drawer",
        "target_link": "large_handle_bar",
        "target_joint": "cabinet_to_drawer",
    }
    assert {
        key: skills_by_id["twist"][key]
        for key in ("articulation", "target_link", "target_joint")
    } == {
        "articulation": "microwave",
        "target_link": "knob_link",
        "target_joint": "knob_joint",
    }


def test_randomized_atomic_suite_covers_six_skills_with_fixed_seed_sweep():
    suite = load_suite("atomic_franka_pgi_curobo_randomized")
    track = suite.enabled_tracks()[0]

    assert suite.suite_version == "atomic_franka_pgi_curobo_randomized_v1"
    assert track.id == "atomic-task-randomized"
    assert track.config["seeds"] == list(range(101, 117))
    assert [item["id"] for item in track.config["skills"]] == [
        "move_end_effector",
        "move_joints",
        "pick_up",
        "move_held_object",
        "place",
        "press",
    ]
    assert all(
        any(str(key).endswith(("_jitter_m", "_jitter_rad")) for key in item)
        for item in track.config["skills"]
    )
    assert track.config["articulations"][0]["id"] == "microwave"
    assert track.config["articulations"][0]["asset_path"] == "Microwave/microwave.urdf"
    press = next(item for item in track.config["skills"] if item["id"] == "press")
    assert press["sample_count"] == 256
    assert press["hand_interp_steps"] == 12
    assert {
        key: press[key] for key in ("articulation", "target_link", "target_joint")
    } == {
        "articulation": "microwave",
        "target_link": "button_link",
        "target_joint": "button_joint",
    }
    assert press["articulation_position_jitter_m"] == [0.03, 0.03, 0.015]


def test_ur5_atomic_suite_loads_nmg_and_six_skills():
    suite = load_suite("ur5_atomic_randomized")
    track = suite.enabled_tracks()[0]

    assert suite.robot.id == "ur5_pgi"
    assert suite.robot.provider == "ur5_pgi"
    assert [spec.id for spec in suite.planners if spec.enabled] == ["nmg"]
    assert track.scenario == "atomic_task"
    assert [item["id"] for item in track.config["skills"]] == [
        "move_end_effector",
        "move_joints",
        "pick_up",
        "move_held_object",
        "place",
        "press",
    ]


def test_ur5_pgi_robot_provider_exposes_six_joint_arm_and_hand():
    suite = load_suite("ur5_atomic_randomized")
    cfg = create_robot_provider(suite.robot).build_cfg()

    assert cfg.uid == "benchmark_ur5_pgi"
    assert cfg.control_parts["arm"] == [f"joint{index}" for index in range(1, 7)]
    assert cfg.control_parts["eef"] == ["gripper_finger1_joint_1"]
    assert cfg.solver_cfg["arm"].end_link_name == "ee_link"
    assert cfg.solver_cfg["arm"].tcp[2][3] == pytest.approx(0.17)
    assert len(cfg.init_qpos) == 8


def test_seeded_jitter_is_reproducible_bounded_and_stream_separated():
    kwargs = {
        "amplitude": (0.1, 0.2, 0.0),
        "seed": 107,
        "dtype": torch.float64,
        "device": torch.device("cpu"),
    }
    first = _seeded_jitter(stream=5, **kwargs)
    repeated = _seeded_jitter(stream=5, **kwargs)
    different_seed = _seeded_jitter(stream=5, **{**kwargs, "seed": 108})
    different_stream = _seeded_jitter(stream=6, **kwargs)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different_seed)
    assert not torch.equal(first, different_stream)
    assert torch.all(torch.abs(first) <= torch.tensor((0.1, 0.2, 0.0)))
    assert first.dtype == torch.float64


def test_case_generation_seed_is_stable_and_stratified():
    value = _case_generation_seed(107, skill_index=2, case_index=3)

    assert value == _case_generation_seed(107, skill_index=2, case_index=3)
    assert value != _case_generation_seed(108, skill_index=2, case_index=3)
    assert value != _case_generation_seed(107, skill_index=3, case_index=3)
    assert value != _case_generation_seed(107, skill_index=2, case_index=4)


def test_case_qpos_canonicalization_removes_sub_resolution_noise():
    first = torch.tensor([[0.123421, -0.456779]], dtype=torch.float32)
    second = torch.tensor([[0.123419, -0.456781]], dtype=torch.float32)

    assert torch.equal(_canonical_case_qpos(first), _canonical_case_qpos(second))


def test_randomization_manifest_records_contract_and_seed():
    parameters = _randomization_parameters(
        {
            "target_offset_jitter_m": [0.03, 0.02, 0.01],
            "object_yaw_jitter_rad": 0.4,
            "sample_count": 80,
        },
        seed=113,
    )

    assert parameters == {
        "enabled": True,
        "seed": 113,
        "distribution": "independent_uniform",
        "ranges": {
            "target_offset_jitter_m": [0.03, 0.02, 0.01],
            "object_yaw_jitter_rad": 0.4,
        },
    }


def test_franka_pgi_robot_provider_exposes_arm_hand_and_tcp():
    from embodichain_robots import FrankaPandaPgiCfg

    suite = load_suite("atomic_franka_pgi_curobo")
    cfg = create_robot_provider(suite.robot).build_cfg()

    assert isinstance(cfg, FrankaPandaPgiCfg)
    assert cfg.uid == "benchmark_franka_pgi"
    # The suites were authored with the tutorial scenes' rotated base.
    assert list(cfg.init_rot) == [0.0, 0.0, 180.0]
    assert len(cfg.control_parts["arm"]) == 7
    assert cfg.control_parts["eef"] == ["gripper_finger1_joint_1"]
    assert cfg.solver_cfg["arm"].end_link_name == "fr3_link8"
    # Suite overrides still apply on top of the shared configuration.
    assert cfg.solver_cfg["arm"].tcp[2][3] == pytest.approx(0.15)
    assert cfg.attrs.material_props.static_friction == pytest.approx(1.0)
    assert len(cfg.init_qpos) == 9
    # The PGI assembly has no fr3 finger links for a stock contact override.
    assert all(
        "fr3_leftfinger" not in str(override.link_names_expr)
        for override in cfg.link_attrs.values()
    )


def test_atomic_invocation_uses_motion_generator_policy():
    provider = create_atomic_skill_provider("move_end_effector")
    scenario = Mock(control_part="manipulator")

    provider.build_invocation(
        scenario,
        _atomic_case(),
        Mock(),
    )

    args, kwargs = scenario.require_engine().make_invocation.call_args
    assert args[0] == "move_end_effector"
    assert kwargs["motion_policy"].strategy == "motion_gen"
    assert kwargs["control_parts"] == {"primary": {"motion": "manipulator"}}


@pytest.mark.parametrize(
    "skill_id", ["open_door", "press", "push_object", "slide", "twist"]
)
def test_contact_benchmarks_use_attributed_ik_interpolation(skill_id: str) -> None:
    assert _benchmark_motion_policy_strategy(skill_id) == "ik_interp"
    expected = {
        "motion_policy_strategy": "ik_interp",
        "selected_planner_backend_used": False,
        "effective_motion_algorithm": "sequential_ik_interpolation",
    }
    assert _benchmark_motion_attribution(skill_id) == expected
    assert (
        AtomicTaskScenario().trial_metadata(replace(_atomic_case(), skill_id=skill_id))
        == expected
    )


def test_noncontact_benchmarks_retain_selected_planner_attribution() -> None:
    assert _benchmark_motion_policy_strategy("move_end_effector") == "motion_gen"
    assert _benchmark_motion_attribution("move_end_effector") == {
        "motion_policy_strategy": "motion_gen",
        "selected_planner_backend_used": True,
        "effective_motion_algorithm": "selected_planner_backend",
    }


@pytest.mark.parametrize(
    "skill_id", ["open_door", "press", "push_object", "slide", "twist"]
)
def test_atomic_scenario_accepts_required_contact_strategy(skill_id: str) -> None:
    case = replace(
        _atomic_case(),
        case_id=f"atomic-task:{skill_id}:simple:s11",
        scenario_id=skill_id,
        skill_id=skill_id,
    )
    motion_generator = Mock()
    adapter = Mock()
    adapter.require_motion_generator.return_value = motion_generator
    adapter.atomic_plan_options.return_value = None
    invocation = SimpleNamespace(
        motion_policy=MotionPolicy(strategy="ik_interp", sample_count=80)
    )
    provider = Mock(samples_grasps=False)
    provider.build_invocation.return_value = invocation
    provider.initial_task_state.return_value = None
    provider.prefix_invocations.return_value = ()
    provider.initial_scene.return_value = None
    provider.frozen_grasps.return_value = {}
    compiled = object()
    engine = Mock(motion_generator=motion_generator)
    engine.compile.return_value = compiled
    scenario = AtomicTaskScenario()
    scenario.simulation = SimpleNamespace(sim_config=SimpleNamespace(physics_dt=0.01))
    scenario.robot = SimpleNamespace(device=torch.device("cpu"))
    scenario._engine = engine
    scenario._case_providers[case.case_id] = provider

    assert scenario.plan_case(adapter, case) is compiled
    engine.compile.assert_called_once()


def test_atomic_skill_and_object_extensions_are_registry_driven():
    assert atomic_skill_provider_names() == (
        "axis_align",
        "coordinated_pickment",
        "coordinated_placement",
        "hand_over",
        "move_end_effector",
        "move_held_object",
        "move_joints",
        "open_door",
        "pick_up",
        "place",
        "pour",
        "press",
        "push_object",
        "slide",
        "twist",
    )
    assert atomic_object_kind_names() == ("cube", "mesh")
    with pytest.raises(ValueError, match="Unknown atomic object kind"):
        create_atomic_object(Mock(), {"id": "new_object", "kind": "not_registered"})


def test_atomic_cube_uses_current_rigid_object_physics_api():
    simulation = Mock()
    simulation.device = torch.device("cpu")
    simulation.num_envs = 1
    entity = Mock()
    simulation.add_rigid_object.return_value = entity

    handle = create_atomic_object(
        simulation,
        {"id": "cube", "kind": "cube", "size": [0.05, 0.05, 0.05]},
        initialize=False,
    )

    cfg = simulation.add_rigid_object.call_args.kwargs["cfg"]
    assert handle.entity is entity
    assert cfg.asset_physics_mode == "overlay"


@pytest.mark.parametrize(
    ("skill_id", "config", "joint_type", "expected_waypoints"),
    [
        (
            "press",
            {
                "name": "microwave_start_button",
                "sample_count": 256,
                "hand_interp_steps": 12,
                "approach_distance_m": 0.12,
                "press_distance_m": 0.03,
                "articulation": "microwave",
                "target_link": "button_link",
                "target_joint": "button_joint",
                "minimum_joint_delta_m": 0.004,
            },
            "prismatic",
            4,
        ),
        (
            "slide",
            {
                "name": "drawer_pull",
                "sample_count": 140,
                "hand_interp_steps": 12,
                "approach_distance_m": 0.10,
                "translation_distance_m": 0.18,
                "direction": "pull",
                "articulation": "drawer",
                "target_link": "large_handle_bar",
                "target_joint": "cabinet_to_drawer",
                "minimum_joint_delta_m": 0.12,
            },
            "prismatic",
            3,
        ),
        (
            "twist",
            {
                "name": "microwave_power_knob",
                "sample_count": 256,
                "hand_interp_steps": 20,
                "twist_waypoint_count": 8,
                "pre_grasp_distance_m": 0.12,
                "twist_angle_rad": -0.7853981634,
                "articulation": "microwave",
                "target_link": "knob_link",
                "target_joint": "knob_joint",
                "minimum_joint_delta_rad": 0.5,
            },
            "revolute",
            11,
        ),
    ],
)
def test_new_atomic_skill_cases_freeze_reference_waypoints(
    skill_id,
    config,
    joint_type,
    expected_waypoints,
):
    batch_size = 8
    robot = Mock(device=torch.device("cpu"))
    raw_start = torch.full((batch_size, 7), 0.002798617584630847)
    robot.get_qpos.return_value = raw_start
    robot.get_joint_ids.return_value = list(range(7))
    robot.compute_fk.return_value = torch.eye(4).repeat(batch_size, 1, 1)
    scenario = Mock(robot=robot, control_part="arm")
    entity = Mock(uid=f"benchmark_{config['articulation']}")
    entity.joint_names = [config["target_joint"]]
    entity.get_qpos_limits.return_value = torch.tensor([[[0.0, 0.006]]])
    articulation_dof = 4 if config["articulation"] == "microwave" else 1
    articulation_pose = torch.eye(4).repeat(batch_size, 1, 1)
    articulation_qpos = torch.zeros(batch_size, articulation_dof)
    entity.get_qpos.return_value = articulation_qpos
    handle = Mock(
        object_id=config["articulation"],
        entity=entity,
        initial_pose=articulation_pose,
        initial_qpos=articulation_qpos,
    )
    handle.link_pose.return_value = articulation_pose
    handle.link_mesh.return_value = (
        torch.tensor(
            [
                [-0.01, -0.01, -0.01],
                [0.01, -0.01, -0.01],
                [0.0, 0.01, 0.01],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([[0, 1, 2]], dtype=torch.long),
    )
    scenario.activate_articulation.return_value = handle
    scenario.randomize_articulation_pose.return_value = articulation_pose
    scenario.sample_articulation_geometry.return_value = _fake_articulation_geometry(
        joint_type=joint_type
    )
    scenario.resolve_articulation_grasp.return_value = articulation_pose
    scenario.solve_reference_qpos.side_effect = lambda start, targets: torch.zeros(
        start.shape[0], targets.shape[1], start.shape[1]
    )
    suite = Mock(
        suite_version="test_v1",
        robot=Mock(id="franka_pgi"),
        protocol=Mock(rotation_threshold_rad=0.1),
    )
    track = Mock(id="atomic-task")

    case = create_atomic_skill_provider(skill_id).generate_case(
        scenario,
        suite,
        track,
        config,
        seed=11,
        batch_size=batch_size,
    )

    assert case.skill_id == skill_id
    assert case.case_id == f"atomic-task:{skill_id}:{config['name']}:b8:s11"
    assert torch.equal(case.full_start_qpos, _canonical_case_qpos(raw_start))
    assert torch.equal(case.start_qpos, case.full_start_qpos[:, list(range(7))])
    assert not torch.equal(case.start_qpos, raw_start)
    assert case.num_waypoints == expected_waypoints
    assert case.target_waypoints.shape == (batch_size, expected_waypoints, 4, 4)
    assert case.reference_qpos.shape == (batch_size, expected_waypoints, 7)
    assert torch.isfinite(case.target_waypoints).all()
    assert case.case_parameters["sample_count"] == config["sample_count"]
    assert case.case_parameters["hand_interp_steps"] == config["hand_interp_steps"]
    assert case.object_id == config["articulation"]
    assert case.case_parameters["target_link"] == config["target_link"]
    assert case.case_parameters["target_joint"] == config["target_joint"]
    assert case.case_parameters["target_entity_id"] == (
        f"benchmark_{config['articulation']}:{config['target_link']}"
    )
    scenario.validate_articulation_target.assert_called_once_with(
        handle,
        target_link=config["target_link"],
        target_joint=config["target_joint"],
        joint_type=joint_type,
    )
    scenario.randomize_articulation_pose.assert_called_once_with(
        handle,
        config,
        seed=11,
        stream={"press": 32, "slide": 42, "twist": 52}[skill_id],
    )
    assert len(case.case_parameters["articulation_translation_offsets_m"]) == (
        batch_size
    )


def test_fixed_held_object_case_keeps_pickup_and_transport_at_table_pose() -> None:
    """Pre-pick composite cases must not reuse the legacy held-height offset."""
    batch_size = 2
    table_pose = torch.eye(4).repeat(batch_size, 1, 1)
    table_pose[:, :3, 3] = torch.tensor(
        [[-0.42, -0.08, 0.05], [-0.38, -0.04, 0.05]], dtype=torch.float32
    )

    robot = Mock(device=torch.device("cpu"))
    robot.get_qpos.side_effect = lambda name=None: torch.zeros(batch_size, 7)
    robot.get_joint_ids.return_value = list(range(7))
    robot.compute_ik.side_effect = lambda pose, joint_seed, name: (
        torch.ones(batch_size, dtype=torch.bool),
        joint_seed.clone(),
    )
    handle = Mock(
        object_id="cube",
        entity=Mock(),
        initial_pose=table_pose.clone(),
        config={"id": "cube"},
    )
    scenario = Mock(
        robot=robot,
        control_part="arm",
        simulation=None,
    )
    scenario.activate_object.return_value = handle
    scenario.randomize_object_pose.return_value = table_pose.clone()
    scenario.solve_reference_qpos.side_effect = lambda start, targets: torch.zeros(
        targets.shape[0], targets.shape[1], start.shape[-1]
    )
    suite = Mock(
        suite_version="test_v1",
        robot=Mock(id="franka_pgi"),
        protocol=Mock(rotation_threshold_rad=0.1),
    )
    track = Mock(id="atomic-task")
    config = {
        "name": "fixed_table_pick",
        "object": "cube",
        "grasp_source": "fixed",
        # This legacy value used to move the object and TCP up by 18 cm.
        "held_object_offset_m": [0.0, 0.0, 0.18],
        "target_object_offset_m": [0.10, 0.10, 0.0],
        "lift_height_m": 0.16,
        "pre_action_settle_steps": 0,
    }

    case = create_atomic_skill_provider("move_held_object").generate_case(
        scenario,
        suite,
        track,
        config,
        seed=11,
        batch_size=batch_size,
    )

    object_initial = torch.tensor(case.case_parameters["object_initial_pose"])
    grasp_pose = torch.tensor(case.case_parameters["grasp_pose"])
    target_object = torch.tensor(case.case_parameters["target_object_pose"])
    torch.testing.assert_close(object_initial, table_pose)
    torch.testing.assert_close(grasp_pose[:, 2, 3], table_pose[:, 2, 3])
    torch.testing.assert_close(target_object[:, 2, 3], table_pose[:, 2, 3] + 0.16)
    assert case.target_waypoints.shape == (batch_size, 1, 4, 4)


def test_held_object_composites_clear_dynamics_at_pickup_lift() -> None:
    """Composite transport/place reuses the PickUp lift boundary."""
    compiled = Mock()
    compiled.segment.return_value = Mock(start=17)

    for skill_id in ("move_held_object", "place"):
        provider = create_atomic_skill_provider(skill_id)
        assert provider.lift_segment_start(compiled) == 17

    compiled.segment.assert_called_with(0, "lift")


def _compiled_replay_batch(
    plan_success: list[bool], positions: torch.Tensor | None = None
) -> CompiledTrajectory:
    """Build a small finite compiled batch for replay-mask tests."""
    batch_size = len(plan_success)
    if positions is None:
        positions = torch.zeros(batch_size, 2, 7)
    trajectory = TimedTrajectory.from_uniform_step(
        positions,
        env_ids=torch.arange(batch_size, dtype=torch.long),
        step_dt=0.01,
    )
    return CompiledTrajectory(
        plan_success=torch.tensor(plan_success, dtype=torch.bool),
        trajectory=trajectory,
        action_plans=(),
        projected_context=Mock(),
    )


def test_partial_compiled_batch_is_replayable() -> None:
    """One successful row should not be blocked by another failed row."""
    assert AtomicTaskScenario._is_replayable(_compiled_replay_batch([True, False]))
    assert not AtomicTaskScenario._is_replayable(_compiled_replay_batch([False, False]))


def test_replay_holds_failed_rows_at_their_frozen_start() -> None:
    """A failed planner row must never send its rollout samples to the robot."""
    positions = torch.tensor(
        [
            [[1.0] * 7, [2.0] * 7],
            [[8.0] * 7, [9.0] * 7],
        ]
    )
    compiled = _compiled_replay_batch([True, False], positions)
    start_qpos = torch.tensor([[0.0] * 7, [0.25] * 7])
    case = replace(
        _atomic_case(),
        batch_size=2,
        full_start_qpos=start_qpos,
    )
    scenario = AtomicTaskScenario()
    scenario.simulation = Mock()
    scenario.robot = Mock()
    scenario.track = Mock(
        config={
            "physics": {
                "steps_per_waypoint": 1,
                "hold_steps": 0,
                "hold_sim_steps": 1,
            }
        }
    )

    scenario._replay_physics(compiled, case, None, collect_metrics=False)

    commands = [item.args[0] for item in scenario.robot.set_qpos.call_args_list]
    assert len(commands) == 2
    assert torch.equal(commands[0][0], positions[0, 0])
    assert torch.equal(commands[1][0], positions[0, 1])
    assert torch.equal(commands[0][1], start_qpos[1])
    assert torch.equal(commands[1][1], start_qpos[1])


def test_execute_masks_failed_plan_rows_from_execution_success() -> None:
    """Controller tracking cannot turn a planner-failed row into success."""
    batch_size = 2
    scenario = AtomicTaskScenario()
    scenario.simulation = Mock(sim_config=Mock(physics_dt=0.01))
    scenario.robot = Mock(device=torch.device("cpu"))
    scenario.robot.get_qpos.side_effect = lambda name=None: torch.zeros(batch_size, 7)
    scenario.robot.compute_fk.return_value = torch.eye(4).repeat(batch_size, 1, 1)
    scenario.track = Mock(
        config={
            "physics": {
                "steps_per_waypoint": 1,
                "hold_steps": 0,
                "hold_sim_steps": 1,
                "joint_tracking_tolerance_rad": 0.05,
            }
        }
    )
    scenario.control_part = "arm"
    scenario._replay_physics = Mock(
        return_value=Mock(
            squared_tracking_error=torch.zeros(batch_size),
            tracking_value_count=7,
            articulation_joint_initial=None,
            articulation_joint_final=None,
            articulation_joint_peak_signed_delta=None,
        )
    )
    case = replace(
        _atomic_case(),
        batch_size=batch_size,
        start_qpos=torch.zeros(batch_size, 7),
        target_waypoints=torch.eye(4).repeat(batch_size, 1, 1, 1),
        reference_qpos=torch.zeros(batch_size, 1, 7),
        full_start_qpos=torch.zeros(batch_size, 9),
    )

    observation = scenario._execute(_compiled_replay_batch([True, False]), case, Mock())

    assert observation is not None
    assert observation.execution_success.tolist() == [True, False]


def test_articulation_effect_uses_peak_displacement_when_joint_rebounds():
    case = replace(
        _atomic_case(),
        case_parameters={"minimum_articulation_joint_delta": 0.004},
    )
    observation = _ExecutionObservation(
        execution_success=torch.tensor([True]),
        final_tcp_pose=torch.eye(4).unsqueeze(0),
        joint_tracking_rmse_rad=torch.zeros(1),
        execution_time_ms=1.0,
        task_completion_time_s=0.1,
        articulation_joint_initial=torch.tensor([0.0]),
        articulation_joint_final=torch.tensor([0.0]),
        articulation_joint_peak_signed_delta=torch.tensor([0.005]),
    )

    success = _articulation_effect_success(
        case,
        observation,
        (_atomic_outcome(),),
    )

    assert observation.articulation_joint_final.item() == pytest.approx(
        observation.articulation_joint_initial.item()
    )
    assert success.tolist() == [True]


def test_pick_up_requires_measured_object_lift() -> None:
    provider = create_atomic_skill_provider("pick_up")
    case = replace(
        _atomic_case(),
        batch_size=2,
        skill_id="pick_up",
        case_parameters={"minimum_object_lift_m": 0.04},
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.1,
        object_lift_delta_m=torch.tensor([0.05, 0.01]),
    )
    outcomes = tuple(replace(_atomic_outcome(), env_index=index) for index in range(2))

    success, failure_code = provider.task_result(
        Mock(), case, Mock(), observation, outcomes
    )

    assert success.tolist() == [True, False]
    assert failure_code == "object_not_lifted"


@pytest.mark.parametrize(
    ("skill_id", "failure_code"),
    [
        ("move_held_object", "object_goal_miss"),
        ("place", "object_not_placed"),
    ],
)
def test_object_motion_requires_measured_target_pose(
    skill_id: str, failure_code: str
) -> None:
    provider = create_atomic_skill_provider(skill_id)
    target_pose = torch.eye(4).repeat(3, 1, 1)
    final_pose = target_pose.clone()
    final_pose[1, 0, 3] = 0.20
    angle = torch.tensor(0.20)
    final_pose[2, 1, 1] = torch.cos(angle)
    final_pose[2, 1, 2] = -torch.sin(angle)
    final_pose[2, 2, 1] = torch.sin(angle)
    final_pose[2, 2, 2] = torch.cos(angle)
    case = replace(
        _atomic_case(),
        batch_size=3,
        skill_id=skill_id,
        case_parameters={
            "target_object_pose": target_pose.tolist(),
            "object_position_threshold_m": 0.05,
            "object_tilt_threshold_rad": 0.10,
        },
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(3, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(3, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(3),
        execution_time_ms=1.0,
        task_completion_time_s=0.1,
        final_object_pose=final_pose,
    )
    scenario = AtomicTaskScenario()
    scenario.suite = Mock(protocol=Mock(rotation_threshold_rad=0.10))
    outcomes = tuple(replace(_atomic_outcome(), env_index=index) for index in range(3))

    success, actual_failure_code = provider.task_result(
        scenario, case, Mock(), observation, outcomes
    )

    assert success.tolist() == [True, False, False]
    assert actual_failure_code == failure_code


@pytest.mark.parametrize("skill_id", ["move_held_object", "place"])
def test_target_yaw_domain_requires_measured_full_rotation(skill_id: str) -> None:
    """A target-yaw case rejects a yaw miss even when object tilt is zero."""
    provider = create_atomic_skill_provider(skill_id)
    target_pose = torch.eye(4).repeat(2, 1, 1)
    final_pose = target_pose.clone()
    yaw = torch.tensor(0.20)
    final_pose[1, 0, 0] = torch.cos(yaw)
    final_pose[1, 0, 1] = -torch.sin(yaw)
    final_pose[1, 1, 0] = torch.sin(yaw)
    final_pose[1, 1, 1] = torch.cos(yaw)
    case = replace(
        _atomic_case(),
        batch_size=2,
        skill_id=skill_id,
        case_parameters={
            "target_object_pose": target_pose.tolist(),
            "object_position_threshold_m": 0.05,
            "object_tilt_threshold_rad": 0.10,
            "object_rotation_threshold_rad": 0.10,
        },
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.1,
        final_object_pose=final_pose,
    )
    scenario = AtomicTaskScenario()
    scenario.suite = Mock(protocol=Mock(rotation_threshold_rad=0.10))
    outcomes = tuple(replace(_atomic_outcome(), env_index=index) for index in range(2))

    success, _ = provider.task_result(scenario, case, Mock(), observation, outcomes)

    assert success.tolist() == [True, False]


@pytest.mark.parametrize("skill_id", ["press", "slide", "twist"])
def test_articulation_tasks_require_measured_joint_displacement(skill_id: str) -> None:
    provider = create_atomic_skill_provider(skill_id)
    case = replace(
        _atomic_case(),
        batch_size=2,
        skill_id=skill_id,
        case_parameters={"minimum_articulation_joint_delta": 0.004},
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.1,
        articulation_joint_peak_signed_delta=torch.tensor([0.005, 0.003]),
    )
    outcomes = tuple(replace(_atomic_outcome(), env_index=index) for index in range(2))

    success, failure_code = provider.task_result(
        Mock(), case, Mock(), observation, outcomes
    )

    assert success.tolist() == [True, False]
    assert failure_code == "articulation_effect_miss"


def test_case_outcome_records_articulation_joint_measurements():
    outcome = _atomic_outcome()

    assert outcome.articulation_joint_initial == pytest.approx(0.0)
    assert outcome.articulation_joint_final == pytest.approx(0.001)
    assert outcome.articulation_joint_delta == pytest.approx(0.001)
    assert outcome.articulation_joint_peak_signed_delta == pytest.approx(0.006)


def test_atomic_primary_success_and_execution_efficiency_aggregate():
    case = _atomic_case()
    metadata = [
        PlannerMetadata(
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="curobo",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint", "atomic_action"}),
            supported_robots=("franka_pgi",),
        )
    ]
    record = TrialRecord(
        suite_version=case.suite_version,
        track=case.track,
        scenario_id=case.scenario_id,
        case_id=case.case_id,
        algorithm_id="curobo",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        model_revision="curobo-v2",
        planner_config_hash="abc",
        seed=case.seed,
        repeat=0,
        batch_size=1,
        waypoint_count=1,
        path_shape=case.path_shape,
        start_state_bin=case.start_state_bin,
        phase=TrialPhase.MEASURED,
        cost_time_ms=20.0,
        robot_id=case.robot_id,
        skill_id=case.skill_id,
        task_difficulty=case.task_difficulty,
        primary_success=case.primary_success,
        execution_time_ms=30.0,
        end_to_end_time_ms=50.0,
        trajectory_duration_s=1.5,
        trajectory_waypoints=80,
        outcomes=(_atomic_outcome(),),
    )

    aggregates = aggregate_results([record], metadata, [case], measured_trials=1)
    metrics = aggregates["success_and_metrics"][0]
    performance = aggregates["time_and_memory"][0]
    leaderboard = aggregates["leaderboard"][0]

    assert metrics["primary_success"] == "task_success"
    assert metrics["success_rate"] == pytest.approx(1.0)
    assert metrics["execution_success_rate"] == pytest.approx(1.0)
    assert metrics["task_success_rate"] == pytest.approx(1.0)
    assert metrics["articulation_joint_delta"] == pytest.approx(0.001)
    assert metrics["articulation_joint_peak_signed_delta"] == pytest.approx(0.006)
    assert performance["execution_time_ms"] == pytest.approx(30.0)
    assert performance["end_to_end_time_ms"] == pytest.approx(50.0)
    assert leaderboard["overall_success_rate"] == pytest.approx(1.0)
    assert leaderboard["task_success_rate"] == pytest.approx(1.0)


def test_planner_aggregates_exclude_atomic_cases_that_bypass_backend():
    selected_case = _atomic_case()
    bypassed_case = replace(
        selected_case,
        scenario_id="press",
        case_id="atomic-task:press:simple:s11",
        skill_id="press",
    )
    metadata = [
        PlannerMetadata(
            algorithm_id="curobo",
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter="curobo",
            config_hash="abc",
            capabilities=frozenset({"eef_waypoint", "atomic_action"}),
            supported_robots=("franka_pgi",),
        )
    ]
    selected_record = TrialRecord(
        suite_version=selected_case.suite_version,
        track=selected_case.track,
        scenario_id=selected_case.scenario_id,
        case_id=selected_case.case_id,
        algorithm_id="curobo",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        model_revision="curobo-v2",
        planner_config_hash="abc",
        seed=selected_case.seed,
        repeat=0,
        batch_size=selected_case.batch_size,
        waypoint_count=selected_case.num_waypoints,
        path_shape=selected_case.path_shape,
        start_state_bin=selected_case.start_state_bin,
        phase=TrialPhase.MEASURED,
        cost_time_ms=20.0,
        robot_id=selected_case.robot_id,
        skill_id=selected_case.skill_id,
        task_difficulty=selected_case.task_difficulty,
        primary_success=selected_case.primary_success,
        metadata=_benchmark_motion_attribution(selected_case.skill_id),
        outcomes=(replace(_atomic_outcome(), task_success=False),),
    )
    bypassed_record = replace(
        selected_record,
        scenario_id=bypassed_case.scenario_id,
        case_id=bypassed_case.case_id,
        skill_id=bypassed_case.skill_id,
        cost_time_ms=1.0,
        metadata=_benchmark_motion_attribution(bypassed_case.skill_id),
        outcomes=(_atomic_outcome(),),
    )

    aggregates = aggregate_results(
        [selected_record, bypassed_record],
        metadata,
        [selected_case, bypassed_case],
        measured_trials=1,
    )

    performance = aggregates["time_and_memory"]
    metrics = aggregates["success_and_metrics"]
    leaderboard = aggregates["leaderboard"][0]
    assert [row["skill"] for row in performance] == ["move_end_effector"]
    assert [row["skill"] for row in metrics] == ["move_end_effector"]
    assert leaderboard["attributed_cases"] == 1
    assert leaderboard["bypassed_cases"] == 1
    assert leaderboard["coverage_rate"] == pytest.approx(1.0)
    assert leaderboard["overall_success_rate"] == pytest.approx(0.0)
    assert leaderboard["latency_p95_ms"] == pytest.approx(20.0)

    bypass_only = aggregate_results(
        [bypassed_record], metadata, [bypassed_case], measured_trials=1
    )
    bypass_leaderboard = bypass_only["leaderboard"][0]
    assert bypass_only["success_and_metrics"] == []
    assert bypass_leaderboard["eligible"] is False
    assert bypass_leaderboard["attributed_cases"] == 0
    assert bypass_leaderboard["bypassed_cases"] == 1
    assert bypass_leaderboard["coverage_rate"] == pytest.approx(0.0)
    assert bypass_leaderboard["overall_success_rate"] == pytest.approx(0.0)
    assert bypass_leaderboard["latency_p95_ms"] is None


def test_runner_preserves_preplan_attribution_after_planner_error(tmp_path):
    case = replace(
        _atomic_case(),
        scenario_id="press",
        case_id="atomic-task:press:simple:s11",
        skill_id="press",
    )
    provider = Mock(spec=ScenarioProvider)
    provider.trial_metadata.return_value = _benchmark_motion_attribution(case.skill_id)
    provider.plan_case.side_effect = RuntimeError("planner failed")
    provider.failure_outcomes.return_value = ()
    runner = object.__new__(BenchmarkRunner)
    runner.control_part = "arm"
    runner.suite = Mock()
    runner.video = VideoRecordCfg()
    runner.records = []
    runner._run_dir = None
    runner._video_paths = []
    planner_metadata = PlannerMetadata(
        algorithm_id="curobo",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        adapter="curobo",
        config_hash="abc",
        capabilities=frozenset({"eef_waypoint", "atomic_action"}),
    )

    runner._run_plan_call(
        TrialJsonlWriter(tmp_path / "trials.jsonl"),
        Mock(),
        Mock(),
        Mock(),
        planner_metadata,
        case,
        provider,
        TrialPhase.MEASURED,
        repeat=0,
    )

    record = runner.records[0]
    assert record.status == "error"
    assert record.failure_code == "planner_exception"
    assert record.metadata["selected_planner_backend_used"] is False
    assert (
        record.metadata["effective_motion_algorithm"] == "sequential_ik_interpolation"
    )


def test_atomic_case_manifest_retains_robot_skill_object_and_parameters(tmp_path):
    case = _atomic_case()
    path = write_case_manifest(tmp_path / "case_manifest.json", [case])
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = payload["cases"][0]

    assert payload["case_schema_version"] == 3
    assert serialized["domain"] is None
    assert serialized["robot_id"] == "franka_pgi"
    assert serialized["skill_id"] == "move_end_effector"
    assert serialized["primary_success"] == "task_success"
    assert serialized["case_parameters"]["sample_count"] == 80
    assert serialized["validity_evidence"]["method"] == "independent_sequential_ik"


def test_should_record_case_respects_enable_failure_and_limit():
    disabled = VideoRecordCfg()
    enabled = VideoRecordCfg(enabled=True)
    with_failed = VideoRecordCfg(enabled=True, record_failed=True)
    limited = VideoRecordCfg(enabled=True, case_limit=1)

    assert should_record_case(disabled, 0, True) is False
    assert should_record_case(enabled, 0, True) is True
    assert should_record_case(enabled, 0, False) is False
    assert should_record_case(with_failed, 0, False) is True
    assert should_record_case(limited, 0, True) is True
    assert should_record_case(limited, 1, True) is False


def test_build_video_path_sanitizes_case_id(tmp_path):
    path = build_video_path(
        tmp_path,
        "curobo",
        "pick_up",
        "atomic-task:pick_up:cube_top_center:s11",
    )

    assert path.parent == tmp_path
    assert path.name == "curobo_pick_up_atomic-task_pick_up_cube_top_center_s11.mp4"
    assert tmp_path.is_dir()


def test_default_scenario_record_replay_is_noop(tmp_path):
    provider = FreeSpaceScenario()
    path = provider.record_replay(
        None,
        _atomic_case(),
        None,
        output_dir=tmp_path,
        algorithm_id="curobo",
        video=VideoRecordCfg(enabled=True),
    )

    assert path is None
    assert isinstance(provider, ScenarioProvider)


def test_atomic_record_replay_without_runtime_returns_none(tmp_path):
    scenario = AtomicTaskScenario()
    path = scenario.record_replay(
        None,
        _atomic_case(),
        None,
        output_dir=tmp_path,
        algorithm_id="curobo",
        video=VideoRecordCfg(enabled=True),
    )
    assert path is None


def test_record_with_window_swallows_exceptions_and_does_not_raise(tmp_path):
    sim = Mock()
    sim.sim_config.width = 64
    sim.sim_config.height = 64
    sim.start_window_record.side_effect = RuntimeError("recorder failed")
    sim.is_window_recording.return_value = False

    path = record_with_window(
        sim,
        VideoRecordCfg(enabled=True),
        tmp_path / "failed.mp4",
        lambda: None,
    )

    assert path is None
    sim.wait_window_record_saves.assert_called()


def test_atomic_record_replay_swallows_recorder_errors(tmp_path):
    scenario = AtomicTaskScenario()
    sim = Mock()
    sim.sim_config.width = 64
    sim.sim_config.height = 64
    sim.start_window_record.side_effect = RuntimeError("boom")
    sim.is_window_recording.return_value = False
    scenario.simulation = sim
    scenario.robot = Mock()
    scenario.reset_case = Mock()

    path = scenario.record_replay(
        None,
        _atomic_case(),
        None,
        output_dir=tmp_path,
        algorithm_id="curobo",
        video=VideoRecordCfg(enabled=True, record_failed=True),
    )

    assert path is None
    scenario.reset_case.assert_called_once()


def test_atomic_failed_plan_records_static_hold(tmp_path):
    scenario = AtomicTaskScenario()
    sim = Mock()
    sim.sim_config.width = 64
    sim.sim_config.height = 64
    sim.start_window_record.return_value = True
    sim.is_window_recording.return_value = True
    sim.stop_window_record.return_value = True
    scenario.simulation = sim
    scenario.robot = Mock()
    scenario.track = Mock(
        config={
            "physics": {"hold_steps": 2, "hold_sim_steps": 1, "steps_per_waypoint": 4}
        }
    )
    scenario.reset_case = Mock()

    path = scenario.record_replay(
        None,
        _atomic_case(),
        None,
        output_dir=tmp_path,
        algorithm_id="curobo",
        video=VideoRecordCfg(enabled=True, record_failed=True),
    )

    assert path is not None
    assert path.name.startswith("curobo_move_end_effector_")
    assert sim.update.call_count == 2


def test_video_cfg_from_args_and_summary_notes():
    args = argparse.Namespace(
        record_video=True,
        record_failed_video=False,
        video_case_limit=0,
        video_fps=20,
        video_width=640,
        video_height=480,
        video_max_memory=2048,
        video_dir=None,
    )
    cfg = video_cfg_from_args(args)
    notes = summarize_video_recording(cfg, ())

    assert cfg.enabled is True
    assert cfg.record_failed is False
    assert cfg.output_dir is None
    assert "Video policy: disabled." not in notes
    assert "videos=0" in notes
    assert summarize_video_recording(VideoRecordCfg(), ()) == [
        "Video policy: disabled."
    ]
    with pytest.raises(ValueError, match="case_limit"):
        VideoRecordCfg(enabled=True, case_limit=-1)


def test_runner_skips_video_outside_measured_phase(tmp_path):
    suite = load_suite("atomic_franka_pgi_curobo")
    specs = [spec for spec in suite.planners if spec.enabled]
    runner = BenchmarkRunner(
        suite,
        specs,
        device="cpu",
        output_root=tmp_path,
        video=VideoRecordCfg(enabled=True, record_failed=True),
    )
    runner._run_dir = tmp_path
    provider = Mock()
    provider.record_replay.return_value = tmp_path / "should_not_write.mp4"

    path = runner._maybe_record_replay(
        provider,
        None,
        _atomic_case(),
        None,
        "curobo",
        TrialPhase.WARMUP,
    )

    assert path is None
    provider.record_replay.assert_not_called()


@pytest.mark.parametrize(
    "execution_ok,motion_ok", [(True, True), (False, True), (True, False)]
)
def test_physical_evaluator_dispatch_preserves_row_results_and_execution_gates(
    monkeypatch, execution_ok: bool, motion_ok: bool
) -> None:
    """Physical evidence stays row-local and cannot override failed execution."""
    from scripts.benchmark.motion_generation import contracts
    from scripts.benchmark.motion_generation.scenarios import atomic_task

    case = replace(
        _atomic_case(), batch_size=2, target_waypoints=torch.eye(4).repeat(2, 1, 1, 1)
    )
    motion = (
        replace(
            _atomic_outcome(),
            motion_valid=motion_ok,
            failure_code=None if motion_ok else "motion_failed",
        ),
        replace(_atomic_outcome(), env_index=1),
    )
    monkeypatch.setattr(
        atomic_task, "compute_case_outcomes", lambda *args, **kwargs: motion
    )
    observation = contracts.ExecutionObservation(
        execution_success=torch.tensor([execution_ok, True]),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        object_lift_delta_m=torch.tensor([0.1, 0.0]),
    )

    class LiftEvaluator:
        def evaluate(self, case, observation, motion_outcomes):
            # Only measured evidence is available; the compiled plan has no input slot.
            return tuple(
                contracts.PhysicalEvaluation(
                    env_index=index,
                    stages=(
                        contracts.StageOutcome(
                            stage_id="lift",
                            status="passed" if height > 0.05 else "failed",
                            failure_code=None if height > 0.05 else "object_not_lifted",
                            measurements={"height_m": height},
                        ),
                    ),
                )
                for index, height in reversed(
                    list(enumerate(observation.object_lift_delta_m.tolist()))
                )
            )

    provider = create_atomic_skill_provider("pick_up")
    monkeypatch.setattr(provider, "physical_evaluator", lambda: LiftEvaluator())

    def forbid_legacy(*args, **kwargs):
        raise AssertionError("A physical evaluator must bypass legacy task_result")

    monkeypatch.setattr(provider, "task_result", forbid_legacy)
    scenario = AtomicTaskScenario()
    scenario._case_providers[case.case_id] = provider
    monkeypatch.setattr(scenario, "_execute", lambda *args: observation)
    robot = Mock(device=torch.device("cpu"))
    robot.get_joint_ids.return_value = list(range(7))
    result = scenario.evaluate_case(
        _compiled_replay_batch([True, True]),
        case,
        robot,
        "arm",
        load_suite("atomic_franka_pgi_curobo"),
        planning_time_ms=2.0,
    )
    assert [row.task_success for row in result.outcomes] == [
        execution_ok and motion_ok,
        False,
    ]
    assert result.outcomes[1].failure_code == "object_not_lifted"
    assert result.outcomes[1].failure_stage == "lift"
    assert result.outcomes[0].stages[0].measurements["height_m"] == pytest.approx(0.1)
    if not motion_ok:
        assert result.outcomes[0].failure_code == "motion_failed"
    elif not execution_ok:
        assert result.outcomes[0].failure_code == "controller_tracking_failure"


def test_runner_freezes_domain_manifest_before_each_planner(
    monkeypatch, tmp_path
) -> None:
    """The same case population and provenance reach all candidate records."""
    from scripts.benchmark.motion_generation import runner as runner_module
    from scripts.benchmark.motion_generation.config import DomainCfg

    suite = load_suite("atomic_franka_pgi_curobo")
    suite.tracks = [suite.enabled_tracks()[0]]
    suite.tracks[0].domain = DomainCfg(id="nominal", version="v1", kind="nominal")
    specs = [suite.planners[0], replace(suite.planners[0], id="second")]
    runner = BenchmarkRunner(suite, specs, device="cpu", output_root=tmp_path)
    provider = Mock()
    provider.batch_sizes.return_value = [1]
    provider.generate_cases.return_value = [_atomic_case()]
    provider.required_capabilities = frozenset()
    monkeypatch.setattr(
        runner_module, "create_scenario_provider", lambda name: provider
    )
    monkeypatch.setattr(runner_module, "environment_metadata", lambda: {})
    monkeypatch.setattr(
        runner_module.SimulationManager, "flush_cleanup_queue", lambda: None
    )
    monkeypatch.setattr(
        runner, "_create_simulation", lambda batch_size: (Mock(), Mock())
    )
    manifests = []

    def run_adapter(writer, sim, robot, spec, cases, capabilities, scenario):
        manifest = json.loads((runner._run_dir / "case_manifest.json").read_text())
        manifests.append(manifest)
        assert manifest["cases"][0]["domain"] == {
            "id": "nominal",
            "version": "v1",
            "kind": "nominal",
        }
        metadata = PlannerMetadata(
            algorithm_id=spec.id,
            algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
            adapter=spec.adapter,
            config_hash=spec.id,
            capabilities=frozenset(),
        )
        runner.metadata[spec.id] = metadata
        runner._append(
            writer,
            TrialRecord(
                **runner._base_record(
                    metadata, cases[0], TrialPhase.MEASURED, repeat=0
                ),
                outcomes=(_atomic_outcome(),),
            ),
        )

    monkeypatch.setattr(runner, "_run_adapter", run_adapter)
    result = runner.run()
    assert len(manifests) == 2
    assert manifests[0] == manifests[1]
    assert all(record.domain.version == "v1" for record in result.records)
    assert provider.generate_cases.return_value[0].domain is None


def test_case_snapshot_uses_one_read_and_runtime_joint_order() -> None:
    """Arm/full states share a snapshot, including batched noncontiguous joints."""
    raw = torch.tensor(
        [[0.123421, 0.4, -0.456779, 0.8], [0.234521, 0.5, -0.567879, 0.9]],
        dtype=torch.float32,
    )
    before = raw.clone()
    robot = Mock()
    robot.get_qpos.return_value = raw
    robot.get_joint_ids.return_value = [2, 0]

    start, full = _snapshot_case_qpos(robot, "noncontiguous_arm")

    robot.get_qpos.assert_called_once_with()
    robot.get_joint_ids.assert_called_once_with(name="noncontiguous_arm")
    assert torch.equal(full, _canonical_case_qpos(before))
    assert torch.equal(start, full[:, [2, 0]])
    full.fill_(9.0)
    assert torch.equal(raw, before)
    assert torch.equal(start, _canonical_case_qpos(before)[:, [2, 0]])


def test_ur5_atomic_move_joints_case_uses_six_controlled_joints() -> None:
    suite = load_suite("ur5_atomic_randomized")
    track = suite.enabled_tracks()[0]
    skill = next(item for item in track.config["skills"] if item["id"] == "move_joints")
    config = {
        **{key: value for key, value in skill.items() if key not in {"id", "cases"}},
        **skill["cases"][0],
    }
    robot = Mock(device=torch.device("cpu"))
    robot.get_qpos.return_value = torch.zeros(1, 8)  # Six arm and two hand joints.
    robot.get_joint_ids.return_value = list(range(6))
    robot.get_qpos_limits.return_value = torch.tensor([[[-6.28, 6.28]] * 6])
    robot.compute_fk.return_value = torch.eye(4).unsqueeze(0)
    scenario = Mock(robot=robot, control_part="arm")

    case = create_atomic_skill_provider("move_joints").generate_case(
        scenario, suite, track, config, seed=101, batch_size=1
    )

    robot.get_qpos.assert_called_once_with()
    robot.get_joint_ids.assert_called_once_with(name="arm")
    assert case.robot_id == "ur5_pgi"
    assert case.start_qpos.shape == (1, 6)
    assert case.full_start_qpos.shape == (1, 8)
    assert case.reference_qpos.shape == (1, 1, 6)


def test_move_joints_case_uses_canonical_snapshot_for_targets(tmp_path: Path) -> None:
    """The generated case, FK inputs and manifest share one rounded start."""
    joint_ids = [6, 0, 4, 2, 7, 1, 5]
    raw = torch.tensor(
        [
            [
                0.0027986176,
                0.123421,
                -0.456779,
                0.02,
                -0.0193351433,
                0.003220673,
                0.234521,
                -0.345621,
                0.03,
            ]
        ],
        dtype=torch.float32,
    ).repeat(2, 1)
    raw[1] += 0.001
    alternate = torch.nextafter(raw, torch.full_like(raw, float("inf")))
    canonical = _canonical_case_qpos(raw)
    assert torch.equal(canonical, _canonical_case_qpos(alternate))
    config = {
        "name": "canonical_joint_offsets",
        "target_offsets_rad": [[0.1] * 7, [-0.05] * 7],
    }
    expected_start = canonical[:, joint_ids]
    expected_targets = expected_start[:, None] + torch.cumsum(
        torch.tensor(config["target_offsets_rad"], dtype=raw.dtype), dim=0
    )
    manifests = []
    for index, observed in enumerate((raw, alternate)):
        robot = Mock(device=torch.device("cpu"))
        robot.get_qpos.return_value = observed
        robot.get_joint_ids.return_value = joint_ids
        robot.get_qpos_limits.return_value = torch.tensor(
            [[[-3.0, 3.0]] * len(joint_ids)], dtype=torch.float32
        )
        robot.compute_fk.return_value = torch.eye(4).repeat(2, 1, 1)
        scenario = Mock(robot=robot, control_part="arm")
        case = create_atomic_skill_provider("move_joints").generate_case(
            scenario,
            Mock(suite_version="snapshot_test_v1", robot=Mock(id="franka_pgi")),
            Mock(id="atomic-task"),
            config,
            seed=103,
            batch_size=2,
        )
        robot.get_qpos.assert_called_once_with()
        assert torch.equal(case.full_start_qpos, canonical)
        assert torch.equal(case.start_qpos, expected_start)
        assert torch.equal(case.reference_qpos, expected_targets)
        assert all(
            torch.equal(call.args[0], expected_targets[:, waypoint])
            for waypoint, call in enumerate(robot.compute_fk.call_args_list)
        )
        manifest = write_case_manifest(tmp_path / f"case-{index}.json", [case])
        manifests.append(json.loads(manifest.read_text()))
    assert manifests[0] == manifests[1]


@pytest.mark.parametrize(
    ("object_x", "object_z", "tcp_x", "expected"),
    [(0.0, 0.0, 0.0, False), (0.03, 0.1, 0.0, False), (0.03, 0.1, 0.03, True)],
)
def test_pickup_retention_tracks_the_target_object(
    object_x: float,
    object_z: float,
    tcp_x: float,
    expected: bool,
) -> None:
    """Reject falling/slipping objects, while allowing rigid hand/object motion."""
    initial = torch.eye(4).reshape(1, 4, 4)
    lifted = initial.clone()
    lifted[:, 2, 3] = 0.1
    final = lifted.clone()
    final[:, 0, 3], final[:, 2, 3] = object_x, object_z
    final_tcp = lifted.clone()
    final_tcp[:, 0, 3] = tcp_x
    holding = torch.full((2, 1, 1), 0.5)
    case = replace(
        _atomic_case(),
        skill_id="pick_up",
        case_parameters={"object_initial_pose": initial.tolist()},
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(1, dtype=torch.bool),
        final_tcp_pose=final_tcp,
        joint_tracking_rmse_rad=torch.zeros(1),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": torch.tensor([0.0])},
        gripper_grasp_qpos={"hand": torch.tensor([1.0])},
        replay_traces={
            "pickup_grasp": ReplayObservationTrace(gripper_qpos={"hand": holding}),
            "pickup_lift": ReplayObservationTrace(
                object_pose=torch.stack((initial, lifted)),
                tcp_pose=torch.stack((initial, lifted)),
            ),
            "pickup_settle": ReplayObservationTrace(
                object_pose=torch.stack((lifted, final)),
                tcp_pose=torch.stack((lifted, final_tcp)),
                gripper_qpos={"hand": holding},
            ),
        },
    )
    result = (
        create_atomic_skill_provider("pick_up")
        .physical_evaluator()
        .evaluate(case, observation, ())[0]
    )
    assert result.task_success is expected
    if not expected:
        assert result.failure_stage == "held"
        assert result.failure_code == "object_dropped"


@pytest.mark.parametrize(
    "invalid_channel", ["gripper", "open_command", "grasp_command"]
)
def test_nonfinite_grasp_evidence_is_unavailable_per_row(invalid_channel: str) -> None:
    """Invalid raw measurements must not become a measured grasp failure."""
    initial = torch.eye(4).repeat(2, 1, 1)
    lifted = initial.clone()
    lifted[:, 2, 3] = 0.1
    holding = torch.full((2, 2, 1), 0.5)
    opened = torch.zeros(2, 1)
    grasped = torch.ones(2, 1)
    if invalid_channel == "gripper":
        holding[-1, 1, 0] = torch.nan
    elif invalid_channel == "open_command":
        opened[1, 0] = torch.inf
    else:
        grasped[1, 0] = torch.nan
    case = replace(
        _atomic_case(),
        batch_size=2,
        skill_id="pick_up",
        case_parameters={"object_initial_pose": initial.tolist()},
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=lifted,
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        gripper_open_qpos={"hand": opened},
        gripper_grasp_qpos={"hand": grasped},
        replay_traces={
            "pickup_grasp": ReplayObservationTrace(gripper_qpos={"hand": holding}),
            "pickup_lift": ReplayObservationTrace(
                object_pose=torch.stack((initial, lifted)),
                tcp_pose=torch.stack((initial, lifted)),
            ),
            "pickup_settle": ReplayObservationTrace(
                object_pose=torch.stack((lifted, lifted)),
                tcp_pose=torch.stack((lifted, lifted)),
                gripper_qpos={"hand": holding},
            ),
        },
    )
    results = (
        create_atomic_skill_provider("pick_up")
        .physical_evaluator()
        .evaluate(case, observation, ())
    )
    assert results[0].task_success
    assert results[1].stages[0].status == "unavailable"
    assert results[1].failure_code == "evidence_unavailable"


def test_push_contact_window_includes_contact_made_during_push() -> None:
    provider = create_atomic_skill_provider("push_object")
    compiled = Mock(trajectory=SimpleNamespace(waypoint_count=30))
    segments = {
        "contact": SimpleNamespace(start=5, stop=10),
        "push": SimpleNamespace(start=10, stop=20),
    }
    compiled.segment.side_effect = lambda index, name: segments[name]
    window = provider.replay_observation_windows(compiled, _atomic_case())[0]
    assert (window.start, window.stop) == (5, 20)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_nonfinite_press_joint_evidence_does_not_count_as_contact_failure(
    invalid_value: float,
) -> None:
    """A corrupt sample in one environment cannot become a false contact label."""
    case = replace(
        _atomic_case(),
        batch_size=2,
        skill_id="press",
        case_parameters={"articulation_joint_stroke": 0.006},
    )
    observation = _ExecutionObservation(
        execution_success=torch.ones(2, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(2, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(2),
        execution_time_ms=1.0,
        task_completion_time_s=0.5,
        replay_traces={
            "press_actuation": ReplayObservationTrace(
                articulation_joint=torch.tensor(
                    [[0.0, 0.0], [0.003, invalid_value], [0.006, 0.006]]
                )
            ),
        },
    )
    results = (
        create_atomic_skill_provider("press")
        .physical_evaluator()
        .evaluate(case, observation, ())
    )
    assert results[0].task_success
    assert results[1].stages[0].status == "unavailable"
    assert results[1].stages[1].status == "not_reached"
    assert results[1].failure_code == "evidence_unavailable"
