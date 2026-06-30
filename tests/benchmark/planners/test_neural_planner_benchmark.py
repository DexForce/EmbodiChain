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

"""Tests for NeuralPlanner benchmark helpers."""

from __future__ import annotations

import math

import pytest
import torch
from unittest.mock import Mock

from embodichain.lab.sim.planners.utils import PlanResult
from scripts.benchmark.planners.neural_planner.run_benchmark import (
    IMPL_NEURAL,
    PERFORMANCE_SUMMARY_COLUMNS,
    QUALITY_SUMMARY_COLUMNS,
    _aggregate_rows,
    _build_performance_leaderboard_rows,
    _build_quality_leaderboard_rows,
    _compute_result_metrics,
    _make_waypoints,
    _project_table_rows,
    compute_waypoint_errors,
    get_pose_err,
)


def test_make_waypoints_clamps_to_available_offsets():
    start_pose = torch.eye(4)
    waypoints = _make_waypoints(start_pose, num_waypoints=99)
    assert waypoints.shape == (8, 4, 4)
    assert torch.allclose(waypoints[0, :3, 3], torch.tensor([0.10, 0.00, 0.00]))

    single = _make_waypoints(start_pose, num_waypoints=1)
    assert single.shape == (1, 4, 4)


def test_get_pose_err_identical_poses():
    pose = torch.eye(4)
    t_err, r_err = get_pose_err(pose, pose)
    assert t_err == pytest.approx(0.0)
    assert r_err == pytest.approx(0.0)


def test_get_pose_err_translation_only():
    pose_a = torch.eye(4)
    pose_b = torch.eye(4)
    pose_b[0, 3] = 0.05
    t_err, r_err = get_pose_err(pose_a, pose_b)
    assert t_err == pytest.approx(0.05)
    assert r_err == pytest.approx(0.0)


def test_compute_waypoint_errors_uses_best_trajectory_hit():
    waypoints = torch.stack(
        [
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ]
    )
    trajectory_poses = [
        torch.eye(4),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ]
    errors = compute_waypoint_errors(trajectory_poses, waypoints)
    assert errors["mean_waypoint_pos_err_mm"] == pytest.approx(0.0)
    assert errors["max_waypoint_pos_err_mm"] == pytest.approx(0.0)


def test_compute_result_metrics_success():
    robot = Mock()
    robot.compute_fk.return_value = torch.eye(4).unsqueeze(0)
    result = PlanResult(
        success=True,
        positions=torch.zeros(10, 7),
        xpos_list=torch.eye(4).unsqueeze(0).repeat(10, 1, 1),
        duration=0.12,
    )
    waypoints = torch.eye(4).unsqueeze(0)
    metrics = _compute_result_metrics(result, waypoints, robot)
    assert metrics["success"] is True
    assert metrics["rollout_steps"] == 10
    assert metrics["duration_s"] == pytest.approx(0.12)
    assert metrics["translation_err_mm"] == pytest.approx(0.0)
    assert metrics["mean_waypoint_pos_err_mm"] == pytest.approx(0.0)


def test_compute_result_metrics_failure():
    robot = Mock()
    metrics = _compute_result_metrics(
        PlanResult(success=False, positions=None),
        torch.eye(4).unsqueeze(0),
        robot,
    )
    assert metrics["success"] is False
    assert metrics["rollout_steps"] == 0
    assert metrics["translation_err_mm"] == math.inf
    assert metrics["mean_waypoint_pos_err_mm"] == math.inf


def test_aggregate_rows_excludes_warmup_and_computes_p95():
    trial_rows = [
        {
            "impl": IMPL_NEURAL,
            "num_waypoints": 3,
            "trial_id": 0,
            "warmup": True,
            "cost_time_ms": "999.0",
            "success": True,
            "final_translation_err_mm": "0.0",
            "final_rotation_err_deg": "0.0",
            "mean_waypoint_pos_err_mm": "0.0",
            "max_waypoint_pos_err_mm": "0.0",
            "mean_waypoint_rot_err_deg": "0.0",
            "max_waypoint_rot_err_deg": "0.0",
            "rollout_steps": 1,
            "cpu_delta_mb": "0.0",
            "gpu_delta_mb": "0.0",
            "peak_gpu_mb": "1.0",
        },
        {
            "impl": IMPL_NEURAL,
            "num_waypoints": 3,
            "trial_id": 1,
            "warmup": False,
            "cost_time_ms": "10.0",
            "success": True,
            "final_translation_err_mm": "1.0",
            "final_rotation_err_deg": "2.0",
            "mean_waypoint_pos_err_mm": "3.0",
            "max_waypoint_pos_err_mm": "4.0",
            "mean_waypoint_rot_err_deg": "5.0",
            "max_waypoint_rot_err_deg": "6.0",
            "rollout_steps": 5,
            "cpu_delta_mb": "1.0",
            "gpu_delta_mb": "2.0",
            "peak_gpu_mb": "3.0",
        },
        {
            "impl": IMPL_NEURAL,
            "num_waypoints": 3,
            "trial_id": 2,
            "warmup": False,
            "cost_time_ms": "20.0",
            "success": False,
            "final_translation_err_mm": "7.0",
            "final_rotation_err_deg": "8.0",
            "mean_waypoint_pos_err_mm": "9.0",
            "max_waypoint_pos_err_mm": "10.0",
            "mean_waypoint_rot_err_deg": "11.0",
            "max_waypoint_rot_err_deg": "12.0",
            "rollout_steps": 6,
            "cpu_delta_mb": "3.0",
            "gpu_delta_mb": "4.0",
            "peak_gpu_mb": "5.0",
        },
    ]
    summary = _aggregate_rows(trial_rows)
    assert len(summary) == 1
    row = summary[0]
    assert row["num_trials"] == 2
    assert row["success_rate"] == "50.00%"
    assert float(row["cost_time_ms_mean"]) == pytest.approx(15.0)
    assert float(row["cost_time_ms_p95"]) == pytest.approx(20.0)
    assert float(row["rollout_steps_mean"]) == pytest.approx(5.5)
    assert float(row["cpu_delta_mb_mean"]) == pytest.approx(2.0)
    assert float(row["gpu_delta_mb_mean"]) == pytest.approx(3.0)
    assert float(row["peak_gpu_mb_mean"]) == pytest.approx(4.0)
    assert float(row["peak_gpu_mb_max"]) == pytest.approx(5.0)
    assert float(row["mean_waypoint_pos_err_mm_mean"]) == pytest.approx(6.0)

    quality = _project_table_rows(summary, QUALITY_SUMMARY_COLUMNS)
    performance = _project_table_rows(summary, PERFORMANCE_SUMMARY_COLUMNS)
    assert set(quality[0]) == set(QUALITY_SUMMARY_COLUMNS)
    assert set(performance[0]) == set(PERFORMANCE_SUMMARY_COLUMNS)
    assert "cost_time_ms_mean" not in quality[0]
    assert "success_rate" not in performance[0]


def test_build_quality_leaderboard_rows_ranks_by_success():
    summary_rows = [
        {
            "impl": "ik_interpolate",
            "num_waypoints": 3,
            "success_rate": "100.00%",
            "final_translation_err_mm_mean": "1.0",
            "final_rotation_err_deg_mean": "1.0",
            "mean_waypoint_pos_err_mm_mean": "1.0",
        },
        {
            "impl": IMPL_NEURAL,
            "num_waypoints": 3,
            "success_rate": "50.00%",
            "final_translation_err_mm_mean": "2.0",
            "final_rotation_err_deg_mean": "2.0",
            "mean_waypoint_pos_err_mm_mean": "2.0",
        },
    ]
    leaderboard = _build_quality_leaderboard_rows(summary_rows)
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[0]["algorithm"] == "ik_interpolate"
    assert leaderboard[1]["algorithm"] == IMPL_NEURAL


def test_build_performance_leaderboard_rows_ranks_by_latency():
    summary_rows = [
        {
            "impl": "ik_interpolate",
            "num_waypoints": 3,
            "cost_time_ms_mean": "5.0",
            "cost_time_ms_p95": "6.0",
            "cpu_delta_mb_mean": "0.1",
            "gpu_delta_mb_mean": "0.2",
            "peak_gpu_mb_mean": "1.0",
        },
        {
            "impl": IMPL_NEURAL,
            "num_waypoints": 3,
            "cost_time_ms_mean": "2.0",
            "cost_time_ms_p95": "3.0",
            "cpu_delta_mb_mean": "0.3",
            "gpu_delta_mb_mean": "0.4",
            "peak_gpu_mb_mean": "2.0",
        },
    ]
    leaderboard = _build_performance_leaderboard_rows(summary_rows)
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[0]["algorithm"] == IMPL_NEURAL
    assert leaderboard[1]["algorithm"] == "ik_interpolate"
