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
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from scripts.benchmark.motion_generation.aggregation import aggregate_results
from scripts.benchmark.motion_generation.artifacts import write_case_manifest
from scripts.benchmark.motion_generation.config import load_suite
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
    )


def test_atomic_suite_is_franka_pgi_and_curobo_only():
    suite = load_suite("atomic_franka_pgi_curobo")

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
    ]
    gripper = suite.enabled_tracks()[0].config["gripper"]
    assert gripper == {
        "control_part": "hand",
        "open_qpos": [0.0],
        "grasp_qpos": [0.024],
    }


def test_franka_pgi_robot_provider_exposes_arm_hand_and_tcp():
    suite = load_suite("atomic_franka_pgi_curobo")
    cfg = create_robot_provider(suite.robot).build_cfg()

    assert cfg.uid == "benchmark_franka_pgi"
    assert len(cfg.control_parts["arm"]) == 7
    assert cfg.control_parts["hand"] == ["gripper_finger1_joint_1"]
    assert cfg.solver_cfg["arm"].end_link_name == "fr3_link8"
    assert cfg.solver_cfg["arm"].tcp[2][3] == pytest.approx(0.15)
    assert len(cfg.init_qpos) == 9


def test_atomic_invocation_pins_motion_generator_and_selected_planner():
    provider = create_atomic_skill_provider("move_end_effector")
    invocation = provider.build_invocation(
        Mock(control_part="manipulator"),
        _atomic_case(),
        Mock(motion_policy_planner="curobo"),
    )

    assert invocation.motion_policy.strategy == "motion_gen"
    assert invocation.motion_policy.planner == "curobo"
    assert invocation.skill_id == "move_end_effector"
    assert invocation.binding.manipulators == {"primary": "manipulator"}


def test_atomic_skill_and_object_extensions_are_registry_driven():
    assert atomic_skill_provider_names() == (
        "move_end_effector",
        "move_held_object",
        "move_joints",
        "pick_up",
        "place",
        "press",
    )
    assert atomic_object_kind_names() == ("cube", "mesh")
    with pytest.raises(ValueError, match="Unknown atomic object kind"):
        create_atomic_object(Mock(), {"id": "new_object", "kind": "not_registered"})


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
    assert performance["execution_time_ms"] == pytest.approx(30.0)
    assert performance["end_to_end_time_ms"] == pytest.approx(50.0)
    assert leaderboard["overall_success_rate"] == pytest.approx(1.0)
    assert leaderboard["task_success_rate"] == pytest.approx(1.0)


def test_atomic_case_manifest_retains_robot_skill_object_and_parameters(tmp_path):
    case = _atomic_case()
    path = write_case_manifest(tmp_path / "case_manifest.json", [case])
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = payload["cases"][0]

    assert payload["case_schema_version"] == 2
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
