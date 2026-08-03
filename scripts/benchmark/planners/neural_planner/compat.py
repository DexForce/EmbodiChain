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

"""Temporary helpers retained for callers of the pre-refactor benchmark module."""

from __future__ import annotations

import math
from collections import defaultdict

__all__ = [
    "IMPL_IK",
    "IMPL_NEURAL",
    "IMPL_TOPPRA",
    "QUALITY_SUMMARY_COLUMNS",
    "aggregate_legacy_rows",
    "format_waypoint_grouped_tables",
]

IMPL_NEURAL = "neural_planner"
IMPL_IK = "ik_interpolate"
IMPL_TOPPRA = "ik_toppra"

QUALITY_SUMMARY_COLUMNS = (
    "impl",
    "num_trials",
    "success_rate",
    "final_translation_err_mm_mean",
    "final_rotation_err_deg_mean",
    "mean_waypoint_pos_err_mm_mean",
    "max_waypoint_pos_err_mm_mean",
    "mean_waypoint_rot_err_deg_mean",
    "max_waypoint_rot_err_deg_mean",
)

_IMPL_REPORT_ORDER = {IMPL_NEURAL: 0, IMPL_IK: 1, IMPL_TOPPRA: 2}


def _percentile(values: list[float], percentile: float) -> float:
    """Return the legacy nearest-rank percentile."""
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile / 100.0 * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def _mean_finite(rows: list[dict[str, object]], key: str) -> str:
    """Format the mean of legacy finite numeric values."""
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return f"{sum(values) / len(values):.6f}" if values else "inf"


def aggregate_legacy_rows(
    trial_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate the legacy row schema during the CLI migration window."""
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        if not bool(row["warmup"]):
            groups[(str(row["impl"]), int(row["num_waypoints"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (impl, waypoint_count), rows in groups.items():
        costs = [float(row["cost_time_ms"]) for row in rows]
        summaries.append(
            {
                "impl": impl,
                "num_waypoints": waypoint_count,
                "num_trials": len(rows),
                "success_rate": f"{sum(bool(row['success']) for row in rows) / len(rows):.2%}",
                "cost_time_ms_mean": f"{sum(costs) / len(costs):.6f}",
                "cost_time_ms_p95": f"{_percentile(costs, 95.0):.6f}",
                "rollout_steps_mean": f"{sum(int(row['rollout_steps']) for row in rows) / len(rows):.2f}",
                "cpu_delta_mb_mean": f"{sum(float(row['cpu_delta_mb']) for row in rows) / len(rows):.6f}",
                "gpu_delta_mb_mean": f"{sum(float(row['gpu_delta_mb']) for row in rows) / len(rows):.6f}",
                "peak_gpu_mb_mean": f"{sum(float(row['peak_gpu_mb']) for row in rows) / len(rows):.6f}",
                "peak_gpu_mb_max": f"{max(float(row['peak_gpu_mb']) for row in rows):.6f}",
                "final_translation_err_mm_mean": _mean_finite(
                    rows, "final_translation_err_mm"
                ),
                "final_rotation_err_deg_mean": _mean_finite(
                    rows, "final_rotation_err_deg"
                ),
                "mean_waypoint_pos_err_mm_mean": _mean_finite(
                    rows, "mean_waypoint_pos_err_mm"
                ),
                "max_waypoint_pos_err_mm_mean": _mean_finite(
                    rows, "max_waypoint_pos_err_mm"
                ),
                "mean_waypoint_rot_err_deg_mean": _mean_finite(
                    rows, "mean_waypoint_rot_err_deg"
                ),
                "max_waypoint_rot_err_deg_mean": _mean_finite(
                    rows, "max_waypoint_rot_err_deg"
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            int(row["num_waypoints"]),
            _IMPL_REPORT_ORDER.get(str(row["impl"]), 99),
        ),
    )


def _format_table(rows: list[dict[str, object]]) -> list[str]:
    """Render the small legacy table used by compatibility tests."""
    if not rows:
        return ["No data."]
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[header]) for header in headers) + " |" for row in rows
    )
    return lines


def format_waypoint_grouped_tables(
    summary_rows: list[dict[str, object]], columns: tuple[str, ...]
) -> list[str]:
    """Render legacy summaries grouped by waypoint count."""
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        groups[int(row["num_waypoints"])].append(row)
    lines: list[str] = []
    for group_index, waypoint_count in enumerate(sorted(groups)):
        if group_index:
            lines.append("")
        rows = sorted(
            groups[waypoint_count],
            key=lambda row: _IMPL_REPORT_ORDER.get(str(row["impl"]), 99),
        )
        lines.extend([f"### num_waypoints = {waypoint_count}", ""])
        lines.extend(
            _format_table([{column: row[column] for column in columns} for row in rows])
        )
    return lines or ["No data."]
