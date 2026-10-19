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

"""Render separate planning and skill views with three tables per view."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .config import SuiteCfg

__all__ = ["write_markdown_report"]

TIME_COLUMNS = (
    "track",
    "algorithm",
    "algorithm_role",
    "robot",
    "skill",
    "object",
    "task_difficulty",
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
    "cost_time_per_waypoint_ms",
    "trajectories_per_second",
    "cpu_delta_mb",
    "gpu_delta_mb",
    "peak_gpu_mb",
    "execution_time_ms",
    "end_to_end_time_ms",
    "timing_is_solved",
    "trajectory_duration_s",
    "trajectory_waypoints",
)

METRIC_COLUMNS = (
    "track",
    "scenario",
    "algorithm",
    "algorithm_role",
    "robot",
    "skill",
    "object",
    "task_difficulty",
    "primary_success",
    "batch_size",
    "waypoint_count",
    "path_shape",
    "start_state_bin",
    "cases",
    "n_valid",
    "coverage_rate",
    "success_rate",
    "planning_success_rate",
    "motion_valid_rate",
    "execution_success_rate",
    "task_success_rate",
    "physical_skill_success_rate",
    "end_to_end_success_rate",
    "evaluation_protocols",
    "physical_evaluation_coverage_rate",
    "ordered_waypoint_success_rate",
    "waypoint_completion_rate",
    "final_pos_err_mm",
    "final_rot_err_deg",
    "waypoint_pos_err_mm_p95",
    "waypoint_rot_err_deg_p95",
    "joint_violation_rate",
    "velocity_violation_rate",
    "acceleration_violation_rate",
    "jerk_violation_rate",
    "velocity_utilization",
    "acceleration_utilization",
    "jerk_utilization",
    "dynamic_measurement_status",
    "max_joint_velocity_rad_s",
    "max_joint_acceleration_rad_s2",
    "max_joint_jerk_rad_s3",
    "joint_path_length_rad",
    "cartesian_path_length_m",
    "path_efficiency",
    "task_completion_time_s",
    "joint_tracking_rmse_rad",
    "object_lift_delta_m",
    "articulation_joint_delta",
    "articulation_joint_peak_signed_delta",
    "replan_count",
    "stage_success_rates",
    "stage_counts",
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
    "attributed_cases",
    "bypassed_cases",
    "coverage_rate",
    "overall_success_rate",
    "planning_success_rate",
    "motion_valid_rate",
    "execution_success_rate",
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
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
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


_MOTION_METRIC_COLUMNS = (
    "track",
    "scenario",
    "algorithm",
    "algorithm_role",
    "robot",
    "batch_size",
    "waypoint_count",
    "path_shape",
    "start_state_bin",
    "cases",
    "n_valid",
    "coverage_rate",
    "success_rate",
    "planning_success_rate",
    "geometry_valid_rate",
    "timed_motion_valid_rate",
    "ordered_waypoint_success_rate",
    "waypoint_completion_rate",
    "final_pos_err_mm",
    "final_rot_err_deg",
    "waypoint_pos_err_mm_p95",
    "waypoint_rot_err_deg_p95",
    "joint_violation_rate",
    "velocity_violation_rate",
    "acceleration_violation_rate",
    "jerk_violation_rate",
    "velocity_utilization",
    "acceleration_utilization",
    "jerk_utilization",
    "dynamic_measurement_status",
    "joint_path_length_rad",
    "cartesian_path_length_m",
    "path_efficiency",
    "top_failure",
)
_ATOMIC_METRIC_COLUMNS = (
    "track",
    "algorithm",
    "algorithm_role",
    "robot",
    "skill",
    "object",
    "task_difficulty",
    "primary_success",
    "cases",
    "coverage_rate",
    "success_rate",
    "physical_skill_success_rate",
    "end_to_end_success_rate",
    "planning_success_rate",
    "geometry_valid_rate",
    "timed_motion_valid_rate",
    "execution_success_rate",
    "physical_evaluation_coverage_rate",
    "evaluation_protocols",
    "task_completion_time_s",
    "joint_tracking_rmse_rad",
    "object_lift_delta_m",
    "articulation_joint_delta",
    "articulation_joint_peak_signed_delta",
    "replan_count",
    "stage_success_rates",
    "stage_counts",
    "top_failure",
)
_VIEW_LEADERBOARD_COLUMNS = (
    "rank",
    "ranking",
    "track",
    "algorithm",
    "algorithm_role",
    "eligible",
    "attributed_cases",
    "bypassed_cases",
    "coverage_rate",
    "overall_success_rate",
    "geometry_valid_rate",
    "timed_motion_valid_rate",
    "physical_skill_success_rate",
    "end_to_end_success_rate",
    "latency_p95_ms",
    "compile_latency_p95_ms",
    "peak_gpu_mb",
)


def write_markdown_report(
    path: str | Path,
    suite: SuiteCfg,
    aggregates: dict[str, list[dict[str, object]]],
    notes: list[str] | None = None,
) -> Path:
    """Write one report, keeping each scenario family's results and ranks separate."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Motion Planning & Atomic Skills Benchmark Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"- suite: `{suite.name}`",
        f"- suite_version: `{suite.suite_version}`",
        f"- report_schema_version: `{suite.protocol.report_schema_version}`",
        f"- profile: `{suite.profile}`",
        f"- external position threshold: `{suite.protocol.position_threshold_m} m` (external motion-validity gate)",
        f"- external rotation threshold: `{suite.protocol.rotation_threshold_rad} rad`",
        "",
        "Geometry validity checks finite positions, ordered targets, and joint-position limits independently of timing. "
        "`timed_motion_valid_rate` additionally requires a solved, valid clock and the applicable dynamic limits. "
        "Nominal or unknown clocks are N/A in timed comparisons. Missing expected outcomes count against success and coverage.",
        "",
        "Continuous error/path columns use geometry-valid outcomes (`n_valid`) for every method. "
        "Dynamic peaks and violations are diagnostic measurements, including nominal clocks; they are not timed success scores.",
        "",
    ]
    views = aggregates.get("report_views", [])
    if not views:
        # Legacy callers still identify a view from the suite's explicit scenario,
        # never from skill names or friendly scenario ids in the artifacts.
        track_scenarios = {
            track.id: track.scenario for track in suite.evaluation_tracks()
        }
        views = [
            {
                "view": view,
                "title": title,
                "tracks": sorted(
                    track
                    for track, owner in track_scenarios.items()
                    if owner == scenario
                ),
            }
            for scenario, view, title in (
                ("free_space", "motion_planning", "Motion Planning"),
                ("atomic_skill", "atomic_skills", "Atomic Skills"),
            )
            if scenario in track_scenarios.values()
        ]
    if not views:
        views = [
            {
                "view": "unclassified",
                "title": "Unclassified scenario",
                "tracks": sorted(
                    {str(row["track"]) for row in aggregates["leaderboard"]}
                ),
            }
        ]
    for descriptor in views:
        view = str(descriptor["view"])
        tracks = set(descriptor["tracks"])
        rows = {
            key: aggregates.get(
                f"{view}_{key}",
                [row for row in aggregates[key] if row["track"] in tracks],
            )
            for key in ("time_and_memory", "success_and_metrics", "leaderboard")
        }
        lines.extend([f"## {descriptor['title']}", ""])
        if view == "atomic_skills":
            lines.extend(
                [
                    "`cost_time_ms` measures Atomic Action compilation; it is reported as `compile_time_ms` and never ranked with bare planner-call latency. "
                    "Physical execution, end-to-end latency, physical stage success, and end-to-end success remain distinct. "
                    "Scenario-owned motion is attributed to its effective algorithm.",
                    "",
                ]
            )
            time_columns = tuple(
                column
                for column in TIME_COLUMNS
                if column
                not in {
                    "warm_plan_ms_p50",
                    "warm_plan_ms_p95",
                    "latency_per_env_ms",
                    "cost_time_per_waypoint_ms",
                    "trajectories_per_second",
                }
            ) + (
                "timing_scope",
                "compile_time_ms",
                "compile_time_ms_p50",
                "compile_time_ms_p95",
            )
            metric_columns = _ATOMIC_METRIC_COLUMNS
        else:
            lines.extend(
                [
                    "Geometry and solved-timing rankings are independent. Raw nominal-timing methods can rank for geometry but receive no timed rank. "
                    "Each track ranks separately; there is no combined score across the two benchmark views.",
                    "",
                ]
            )
            time_columns = tuple(
                column
                for column in TIME_COLUMNS
                if column
                not in {
                    "skill",
                    "object",
                    "task_difficulty",
                    "execution_time_ms",
                    "end_to_end_time_ms",
                }
            ) + ("timing_scope", "planning_time_ms")
            metric_columns = _MOTION_METRIC_COLUMNS
        lines.extend(["### Time & Memory", ""])
        lines.extend(_format_table(rows["time_and_memory"], time_columns))
        lines.extend(["", "### Success & Other Metrics", ""])
        lines.extend(_format_table(rows["success_and_metrics"], metric_columns))
        lines.extend(["", "### Leaderboard", ""])
        leaderboard = rows["leaderboard"] + (
            aggregates.get("motion_timing_leaderboard", [])
            if view == "motion_planning"
            else []
        )
        lines.extend(_format_table(leaderboard, _VIEW_LEADERBOARD_COLUMNS))
        lines.append("")
    domain_rows = aggregates.get("generalization", [])
    if domain_rows:
        lines.extend(
            [
                "## Generalization summary",
                "",
                "Domain populations stay separate by track and algorithm; unsupported sources affect coverage but do not erase another source's measured evidence.",
                "",
            ]
        )
        for row in domain_rows:
            lines.append(
                "- "
                + "; ".join(
                    f"{name}={_format_value(name, row.get(name))}"
                    for name in (
                        "track",
                        "algorithm",
                        "domain_id",
                        "domain_version",
                        "domain_kind",
                        "cases",
                        "attributed_cases",
                        "bypassed_cases",
                        "coverage_rate",
                        "success_rate",
                        "spatial_bucket_success_rates",
                        "spatial_bucket_counts",
                        "spatial_bucket_failure_stages",
                        "domain_mean_success_rate",
                        "domain_worst_success_rate",
                        "domain_success_variance",
                        "domain_retention",
                        "generalization_eligible",
                    )
                )
            )
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
