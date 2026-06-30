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

"""Smoke tests for NeuralPlanner benchmark aggregation and reporting."""

from __future__ import annotations

import pytest
import torch

from scripts.benchmark.planners.neural_planner.run_benchmark import (
    IMPL_IK,
    IMPL_NEURAL,
    IMPL_TOPPRA,
    QUALITY_SUMMARY_COLUMNS,
    _aggregate_rows,
    _format_waypoint_grouped_tables,
    compute_waypoint_errors,
)


def test_compute_waypoint_errors_uses_best_trajectory_hit():
    waypoints = torch.stack(
        [
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
    )
    trajectory_poses = [torch.eye(4), waypoints[1]]
    errors = compute_waypoint_errors(trajectory_poses, waypoints)
    assert errors["mean_waypoint_pos_err_mm"] == pytest.approx(0.0)
    assert errors["max_waypoint_pos_err_mm"] == pytest.approx(0.0)


def test_aggregate_rows_excludes_warmup_and_computes_p95():
    trial_rows = [
        {
            "impl": IMPL_NEURAL,
            "num_waypoints": 3,
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
    row = _aggregate_rows(trial_rows)[0]
    assert row["num_trials"] == 2
    assert row["success_rate"] == "50.00%"
    assert float(row["cost_time_ms_mean"]) == pytest.approx(15.0)
    assert float(row["cost_time_ms_p95"]) == pytest.approx(20.0)
    assert float(row["mean_waypoint_pos_err_mm_mean"]) == pytest.approx(6.0)


def test_format_waypoint_grouped_tables_splits_by_num_waypoints():
    summary_rows = [
        {
            "impl": IMPL_IK,
            "num_waypoints": 3,
            "num_trials": 8,
            "success_rate": "100.00%",
            "final_translation_err_mm_mean": "1.0",
            "final_rotation_err_deg_mean": "0.0",
            "mean_waypoint_pos_err_mm_mean": "1.0",
            "max_waypoint_pos_err_mm_mean": "1.0",
            "mean_waypoint_rot_err_deg_mean": "0.0",
            "max_waypoint_rot_err_deg_mean": "0.0",
        },
        {
            "impl": IMPL_NEURAL,
            "num_waypoints": 1,
            "num_trials": 8,
            "success_rate": "100.00%",
            "final_translation_err_mm_mean": "2.0",
            "final_rotation_err_deg_mean": "0.0",
            "mean_waypoint_pos_err_mm_mean": "2.0",
            "max_waypoint_pos_err_mm_mean": "2.0",
            "mean_waypoint_rot_err_deg_mean": "0.0",
            "max_waypoint_rot_err_deg_mean": "0.0",
        },
        {
            "impl": IMPL_TOPPRA,
            "num_waypoints": 3,
            "num_trials": 8,
            "success_rate": "100.00%",
            "final_translation_err_mm_mean": "0.5",
            "final_rotation_err_deg_mean": "0.0",
            "mean_waypoint_pos_err_mm_mean": "0.5",
            "max_waypoint_pos_err_mm_mean": "0.5",
            "mean_waypoint_rot_err_deg_mean": "0.0",
            "max_waypoint_rot_err_deg_mean": "0.0",
        },
    ]
    text = "\n".join(
        _format_waypoint_grouped_tables(summary_rows, QUALITY_SUMMARY_COLUMNS)
    )
    assert text.index("### num_waypoints = 1") < text.index("### num_waypoints = 3")
    assert "| num_waypoints |" not in text
    assert text.index(IMPL_NEURAL) < text.index(IMPL_IK)
