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

"""Aggregate raw planner trials and build a complete benchmark leaderboard."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from .models import BenchmarkCase, CaseOutcome, PlannerMetadata, TrialPhase, TrialRecord

__all__ = ["aggregate_results"]


def _mean(values: Iterable[float | None]) -> float | None:
    """Return the mean of finite values or ``None`` when unavailable."""
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(finite) / len(finite) if finite else None


def _percentile(values: Iterable[float | None], percentile: float) -> float | None:
    """Return a nearest-rank percentile over finite values."""
    finite = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return None
    index = max(
        0, min(len(finite) - 1, math.ceil(percentile / 100.0 * len(finite)) - 1)
    )
    return finite[index]


def _rate(values: Iterable[bool]) -> float | None:
    """Return a boolean rate or ``None`` for an empty sequence."""
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _top_failure(outcomes: list[CaseOutcome]) -> str | None:
    """Return the most frequent non-empty failure code."""
    failures = Counter(
        outcome.failure_code for outcome in outcomes if outcome.failure_code
    )
    return failures.most_common(1)[0][0] if failures else None


def _lifecycle_value(
    records: list[TrialRecord],
    algorithm_id: str,
    batch_size: int,
    phase: TrialPhase,
) -> float | None:
    """Return the first lifecycle cost for one algorithm and batch size."""
    for record in records:
        if (
            record.algorithm_id == algorithm_id
            and record.batch_size == batch_size
            and record.phase is phase
        ):
            return record.cost_time_ms
    return None


def _performance_rows(
    records: list[TrialRecord], metadata: list[PlannerMetadata]
) -> list[dict[str, object]]:
    """Aggregate steady-state time and memory by algorithm and input shape."""
    measured_groups: dict[tuple[str, int, int], list[TrialRecord]] = defaultdict(list)
    for record in records:
        if record.phase is TrialPhase.MEASURED:
            measured_groups[
                (record.algorithm_id, record.batch_size, record.waypoint_count)
            ].append(record)

    metadata_by_id = {item.algorithm_id: item for item in metadata}
    rows: list[dict[str, object]] = []
    for key in sorted(measured_groups):
        algorithm_id, batch_size, waypoint_count = key
        group = measured_groups[key]
        info = metadata_by_id[algorithm_id]
        costs = [record.cost_time_ms for record in group]
        mean_cost = _mean(costs)
        rows.append(
            {
                "track": "free-space-common",
                "algorithm": algorithm_id,
                "algorithm_role": info.algorithm_role.value,
                "batch_size": batch_size,
                "waypoint_count": waypoint_count,
                "num_trials": len(group),
                "planner_construct_ms": _lifecycle_value(
                    records, algorithm_id, batch_size, TrialPhase.CONSTRUCT
                ),
                "backend_prepare_ms": _lifecycle_value(
                    records, algorithm_id, batch_size, TrialPhase.PREPARE
                ),
                "cold_plan_ms": _lifecycle_value(
                    records, algorithm_id, batch_size, TrialPhase.COLD
                ),
                "cost_time_ms": mean_cost,
                "warm_plan_ms_p50": _percentile(costs, 50.0),
                "warm_plan_ms_p95": _percentile(costs, 95.0),
                "latency_per_env_ms": (
                    mean_cost / batch_size if mean_cost is not None else None
                ),
                "cost_time_per_segment_ms": (
                    mean_cost / waypoint_count if mean_cost is not None else None
                ),
                "trajectories_per_second": (
                    batch_size * 1000.0 / mean_cost
                    if mean_cost is not None and mean_cost > 0.0
                    else None
                ),
                "cpu_delta_mb": _mean(record.cpu_delta_mb for record in group),
                "gpu_delta_mb": _mean(record.gpu_delta_mb for record in group),
                "peak_gpu_mb": max(
                    (record.peak_gpu_mb or 0.0 for record in group), default=0.0
                ),
            }
        )

    present_algorithms = {row["algorithm"] for row in rows}
    for info in metadata:
        if info.algorithm_id in present_algorithms:
            continue
        rows.append(
            {
                "track": "free-space-common",
                "algorithm": info.algorithm_id,
                "algorithm_role": info.algorithm_role.value,
                "batch_size": None,
                "waypoint_count": None,
                "num_trials": 0,
                "planner_construct_ms": None,
                "backend_prepare_ms": None,
                "cold_plan_ms": None,
                "cost_time_ms": None,
                "warm_plan_ms_p50": None,
                "warm_plan_ms_p95": None,
                "latency_per_env_ms": None,
                "cost_time_per_segment_ms": None,
                "trajectories_per_second": None,
                "cpu_delta_mb": None,
                "gpu_delta_mb": None,
                "peak_gpu_mb": None,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["algorithm"]),
            int(row["batch_size"] or 0),
            int(row["waypoint_count"] or 0),
        ),
    )


def _metric_rows(
    records: list[TrialRecord],
    metadata: list[PlannerMetadata],
    cases: list[BenchmarkCase],
    measured_trials: int,
) -> list[dict[str, object]]:
    """Aggregate external success and quality metrics by scenario condition."""
    outcome_groups: dict[tuple[str, str, int, int, str], list[CaseOutcome]] = (
        defaultdict(list)
    )
    for record in records:
        if record.phase is not TrialPhase.MEASURED:
            continue
        key = (
            record.algorithm_id,
            record.scenario_id,
            record.batch_size,
            record.waypoint_count,
            record.path_shape,
        )
        outcome_groups[key].extend(record.outcomes)

    expected_by_group: Counter[tuple[str, int, int, str]] = Counter()
    unique_cases_by_group: Counter[tuple[str, int, int, str]] = Counter()
    for case in cases:
        key = (case.scenario_id, case.batch_size, case.num_waypoints, case.path_shape)
        expected_by_group[key] += case.batch_size * measured_trials
        unique_cases_by_group[key] += case.batch_size

    rows: list[dict[str, object]] = []
    for info in metadata:
        for group_key in sorted(expected_by_group):
            scenario_id, batch_size, waypoint_count, path_shape = group_key
            outcomes = outcome_groups.get(
                (
                    info.algorithm_id,
                    scenario_id,
                    batch_size,
                    waypoint_count,
                    path_shape,
                ),
                [],
            )
            valid_outcomes = [outcome for outcome in outcomes if outcome.motion_valid]
            expected = expected_by_group[group_key]
            rows.append(
                {
                    "track": "free-space-common",
                    "scenario": scenario_id,
                    "algorithm": info.algorithm_id,
                    "algorithm_role": info.algorithm_role.value,
                    "batch_size": batch_size,
                    "waypoint_count": waypoint_count,
                    "path_shape": path_shape,
                    "cases": unique_cases_by_group[group_key],
                    "coverage_rate": min(1.0, len(outcomes) / max(expected, 1)),
                    "success_rate": _rate(outcome.motion_valid for outcome in outcomes),
                    "planning_success_rate": _rate(
                        outcome.planning_success for outcome in outcomes
                    ),
                    "ordered_waypoint_success_rate": _rate(
                        outcome.ordered_waypoints_reached for outcome in outcomes
                    ),
                    "motion_valid_rate": _rate(
                        outcome.motion_valid for outcome in outcomes
                    ),
                    "waypoint_completion_rate": _mean(
                        outcome.completed_waypoint_ratio for outcome in outcomes
                    ),
                    "final_pos_err_mm": _mean(
                        outcome.final_translation_err_mm for outcome in valid_outcomes
                    ),
                    "final_rot_err_deg": _mean(
                        outcome.final_rotation_err_deg for outcome in valid_outcomes
                    ),
                    "waypoint_pos_err_mm_p95": _mean(
                        outcome.waypoint_translation_err_mm_p95
                        for outcome in valid_outcomes
                    ),
                    "waypoint_rot_err_deg_p95": _mean(
                        outcome.waypoint_rotation_err_deg_p95
                        for outcome in valid_outcomes
                    ),
                    "joint_violation_rate": _rate(
                        outcome.joint_limit_violation for outcome in outcomes
                    ),
                    "joint_path_length_rad": _mean(
                        outcome.joint_path_length_rad for outcome in valid_outcomes
                    ),
                    "cartesian_path_length_m": _mean(
                        outcome.cartesian_path_length_m for outcome in valid_outcomes
                    ),
                    "path_efficiency": _mean(
                        outcome.path_efficiency for outcome in valid_outcomes
                    ),
                    "top_failure": _top_failure(outcomes),
                }
            )
    return rows


def _leaderboard_rows(
    records: list[TrialRecord],
    metadata: list[PlannerMetadata],
    cases: list[BenchmarkCase],
    measured_trials: int,
) -> list[dict[str, object]]:
    """Build a complete success/coverage/latency ordered leaderboard."""
    expected_outcomes = sum(case.batch_size for case in cases) * measured_trials
    entries: list[dict[str, object]] = []
    for info in metadata:
        measured = [
            record
            for record in records
            if record.algorithm_id == info.algorithm_id
            and record.phase is TrialPhase.MEASURED
        ]
        outcomes = [outcome for record in measured for outcome in record.outcomes]
        coverage = min(1.0, len(outcomes) / max(expected_outcomes, 1))
        motion_rate = _rate(outcome.motion_valid for outcome in outcomes) or 0.0
        planning_rate = _rate(outcome.planning_success for outcome in outcomes) or 0.0
        latency_p95 = _percentile((record.cost_time_ms for record in measured), 95.0)
        peak_gpu = max((record.peak_gpu_mb or 0.0 for record in measured), default=None)
        entries.append(
            {
                "track": "free-space-common",
                "algorithm": info.algorithm_id,
                "algorithm_role": info.algorithm_role.value,
                "model_revision": info.model_revision,
                "planner_config_hash": info.config_hash[:12],
                "eligible": coverage >= 1.0 - 1.0e-12,
                "coverage_rate": coverage,
                "overall_success_rate": motion_rate,
                "planning_success_rate": planning_rate,
                "motion_valid_rate": motion_rate,
                "task_success_rate": None,
                "latency_p95_ms": latency_p95,
                "peak_gpu_mb": peak_gpu,
            }
        )

    entries.sort(
        key=lambda row: (
            not bool(row["eligible"]),
            -float(row["overall_success_rate"]),
            -float(row["coverage_rate"]),
            (
                float(row["latency_p95_ms"])
                if row["latency_p95_ms"] is not None
                else math.inf
            ),
            str(row["algorithm"]),
        )
    )
    return [{"rank": rank, **entry} for rank, entry in enumerate(entries, start=1)]


def aggregate_results(
    records: list[TrialRecord],
    metadata: list[PlannerMetadata],
    cases: list[BenchmarkCase],
    measured_trials: int,
) -> dict[str, list[dict[str, object]]]:
    """Build all three report datasets from raw numeric records."""
    return {
        "time_and_memory": _performance_rows(records, metadata),
        "success_and_metrics": _metric_rows(records, metadata, cases, measured_trials),
        "leaderboard": _leaderboard_rows(records, metadata, cases, measured_trials),
    }
