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

"""Tests for the Franka planner NMG benchmark helpers."""

from __future__ import annotations

import torch

from scripts.benchmark.robotics.nmg.franka_planner import (
    PlannerOutcome,
    PlannerTrial,
    build_trial_row,
    expand_planner_selection,
    make_leaderboard_rows,
    make_metric_rows,
    make_perf_rows,
    make_skipped_rows,
    parse_args,
    simulation_requires_cuda,
    write_markdown_report,
)


class _FakeRobot:
    device = torch.device("cpu")


def _trial() -> PlannerTrial:
    return PlannerTrial(
        trial_id=0,
        start_qpos=torch.zeros(7),
        target_qpos=torch.zeros(1, 7),
        waypoints=[torch.eye(4)],
    )


def _rows() -> list[dict[str, object]]:
    return [
        {
            "script": "franka_planner_nmg",
            "planner": "ik_toppra",
            "trial_id": 0,
            "warmup": True,
            "num_waypoints": 1,
            "action_success": False,
            "strict_pose_success": False,
            "nmg_threshold_success": False,
            "planning_time_sec": 10.0,
            "cpu_delta_mb": 0.0,
            "gpu_delta_mb": 0.0,
            "peak_gpu_mb": 0.0,
            "trajectory_steps": 0,
            "final_tcp_pos_error": None,
            "final_tcp_rot_error": None,
            "joint_path_length": 0.0,
            "max_joint_step": 0.0,
            "mean_target_qpos_error": None,
            "final_qpos": None,
        },
        {
            "script": "franka_planner_nmg",
            "planner": "ik_toppra",
            "trial_id": 1,
            "warmup": False,
            "num_waypoints": 1,
            "action_success": True,
            "strict_pose_success": True,
            "nmg_threshold_success": True,
            "planning_time_sec": 0.8,
            "cpu_delta_mb": 1.0,
            "gpu_delta_mb": 0.0,
            "peak_gpu_mb": 0.0,
            "trajectory_steps": 120,
            "final_tcp_pos_error": 0.0005,
            "final_tcp_rot_error": 0.01,
            "joint_path_length": 0.5,
            "max_joint_step": 0.1,
            "mean_target_qpos_error": 0.0,
            "final_qpos": [0.0] * 7,
        },
        {
            "script": "franka_planner_nmg",
            "planner": "neural",
            "trial_id": 1,
            "warmup": False,
            "num_waypoints": 1,
            "action_success": True,
            "strict_pose_success": False,
            "nmg_threshold_success": True,
            "planning_time_sec": 0.2,
            "cpu_delta_mb": 0.5,
            "gpu_delta_mb": 2.0,
            "peak_gpu_mb": 10.0,
            "trajectory_steps": 120,
            "final_tcp_pos_error": 0.02,
            "final_tcp_rot_error": 0.2,
            "joint_path_length": 0.7,
            "max_joint_step": 0.2,
            "mean_target_qpos_error": 0.1,
            "final_qpos": [0.1] * 7,
        },
    ]


def test_cli_defaults_and_selection_expansion():
    args = parse_args([])

    assert args.planner == "all"
    assert args.num_trials == 8
    assert expand_planner_selection("all") == [
        "ik_toppra",
        "neural",
        "neural_refine",
    ]
    assert expand_planner_selection("neural") == ["neural"]


def test_simulation_cuda_preflight():
    args = parse_args(["--device", "cpu", "--renderer", "fast-rt"])

    assert simulation_requires_cuda(args)
    skipped = make_skipped_rows(["ik_toppra", "neural"], reason="cuda missing")

    assert len(skipped) == 2
    assert {row["planner"] for row in skipped} == {"ik_toppra", "neural"}
    assert all(row["skip_reason"] == "cuda missing" for row in skipped)


def test_report_builders_ignore_warmup_rows():
    perf_rows = make_perf_rows(_rows())
    metric_rows = make_metric_rows(_rows())
    leaderboard_rows = make_leaderboard_rows(_rows())

    assert {row["planner"] for row in perf_rows} == {"ik_toppra", "neural"}
    assert all(row["repeat_count"] == 1 for row in perf_rows)
    assert metric_rows[0]["strict_pose_success_rate"] == "100.0%"
    assert leaderboard_rows[0]["planner"] == "ik_toppra"


def test_write_markdown_report_has_exactly_three_tables(tmp_path):
    report_path = tmp_path / "report.md"

    written = write_markdown_report(_rows(), str(report_path))

    text = written.read_text(encoding="utf-8")
    assert written == report_path
    assert text.count("## Time & Memory") == 1
    assert text.count("## Success & Other Metrics") == 1
    assert text.count("## Leaderboard") == 1
    assert text.count("| planner |") == 3


def test_build_trial_row_marks_failed_empty_trajectory():
    outcome = PlannerOutcome(
        action_success=False,
        positions=None,
        planning_time_sec=0.1,
        cpu_delta_mb=0.0,
        gpu_delta_mb=0.0,
        peak_gpu_mb=0.0,
    )

    row = build_trial_row(
        planner="neural",
        trial=_trial(),
        warmup=False,
        outcome=outcome,
        robot=_FakeRobot(),
        args=parse_args([]),
    )

    assert row["action_success"] is False
    assert row["strict_pose_success"] is False
    assert row["nmg_threshold_success"] is False
    assert row["trajectory_steps"] == 0
