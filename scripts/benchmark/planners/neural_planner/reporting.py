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

"""Render the free-space benchmark as exactly three Markdown tables."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

from .config import SuiteCfg

__all__ = ["write_markdown_report"]

TIME_COLUMNS = (
    "track",
    "algorithm",
    "algorithm_role",
    "batch_size",
    "waypoint_count",
    "num_trials",
    "planner_construct_ms",
    "backend_prepare_ms",
    "cold_plan_ms",
    "cost_time_ms",
    "warm_plan_ms_p50",
    "warm_plan_ms_p95",
    "latency_per_env_ms",
    "cost_time_per_segment_ms",
    "trajectories_per_second",
    "cpu_delta_mb",
    "gpu_delta_mb",
    "peak_gpu_mb",
)

METRIC_COLUMNS = (
    "track",
    "scenario",
    "algorithm",
    "algorithm_role",
    "batch_size",
    "waypoint_count",
    "path_shape",
    "start_state_bin",
    "cases",
    "coverage_rate",
    "success_rate",
    "planning_success_rate",
    "ordered_waypoint_success_rate",
    "motion_valid_rate",
    "waypoint_completion_rate",
    "final_pos_err_mm",
    "final_rot_err_deg",
    "waypoint_pos_err_mm_p95",
    "waypoint_rot_err_deg_p95",
    "joint_violation_rate",
    "joint_path_length_rad",
    "cartesian_path_length_m",
    "path_efficiency",
    "top_failure",
)

LEADERBOARD_COLUMNS = (
    "rank",
    "track",
    "algorithm",
    "algorithm_role",
    "model_revision",
    "planner_config_hash",
    "eligible",
    "coverage_rate",
    "overall_success_rate",
    "planning_success_rate",
    "motion_valid_rate",
    "task_success_rate",
    "latency_p95_ms",
    "peak_gpu_mb",
)


def _format_value(column: str, value: object) -> str:
    """Format one display value without mutating raw aggregate artifacts."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        if column.endswith("_rate") or column in {"path_efficiency"}:
            return f"{value:.2%}"
        return f"{value:.6f}"
    return str(value)


def _format_table(rows: list[dict[str, object]], columns: tuple[str, ...]) -> list[str]:
    """Render one table with a stable schema even when rows are empty."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_value(column, row.get(column)) for column in columns)
            + " |"
        )
    return lines


def write_markdown_report(
    path: str | Path,
    suite: SuiteCfg,
    aggregates: dict[str, list[dict[str, object]]],
    notes: list[str] | None = None,
) -> Path:
    """Write one report containing exactly the required three tables."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Motion Generation Benchmark Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"- suite: `{suite.name}`",
        f"- suite_version: `{suite.suite_version}`",
        f"- profile: `{suite.profile}`",
        f"- external position threshold: `{suite.protocol.position_threshold_m} m`",
        f"- external rotation threshold: `{suite.protocol.rotation_threshold_rad} rad`",
        "",
        "## Time & Memory",
        "",
    ]
    lines.extend(_format_table(aggregates["time_and_memory"], TIME_COLUMNS))
    lines.extend(["", "## Success & Other Metrics", ""])
    lines.extend(_format_table(aggregates["success_and_metrics"], METRIC_COLUMNS))
    lines.extend(["", "## Leaderboard", ""])
    lines.extend(_format_table(aggregates["leaderboard"], LEADERBOARD_COLUMNS))
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
