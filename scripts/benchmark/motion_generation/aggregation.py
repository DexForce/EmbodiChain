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

from .metrics.stats import nearest_rank_percentile
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


def _case_macro_rate(
    measured: list[TrialRecord],
    track_cases: list[BenchmarkCase],
    attribute: str,
) -> float:
    """Macro-average a boolean outcome attribute equally across mandatory cases.

    Within each case, env rows (and measured repeats) are micro-averaged first.
    Cases then receive equal weight regardless of ``batch_size``, so a B=64
    case cannot dominate a B=1 case on the leaderboard. Missing cases
    contribute ``0.0`` so selective skipping cannot inflate the rate.
    """
    if not track_cases:
        return 0.0

    outcomes_by_case: dict[str, list[CaseOutcome]] = defaultdict(list)
    for record in measured:
        outcomes_by_case[record.case_id].extend(record.outcomes)

    case_rates: list[float] = []
    for case in track_cases:
        outcomes = outcomes_by_case.get(case.case_id, [])
        if not outcomes:
            case_rates.append(0.0)
            continue
        case_rates.append(
            sum(bool(getattr(outcome, attribute)) for outcome in outcomes)
            / len(outcomes)
        )
    return sum(case_rates) / len(case_rates)


def _case_macro_mean(
    measured: list[TrialRecord],
    track_cases: list[BenchmarkCase],
    attribute: str,
) -> float | None:
    """Macro-average a numeric outcome attribute across cases with values."""
    if not track_cases:
        return None

    outcomes_by_case: dict[str, list[CaseOutcome]] = defaultdict(list)
    for record in measured:
        outcomes_by_case[record.case_id].extend(record.outcomes)

    case_means: list[float] = []
    for case in track_cases:
        values = [
            float(getattr(outcome, attribute))
            for outcome in outcomes_by_case.get(case.case_id, [])
            if getattr(outcome, attribute) is not None
            and math.isfinite(float(getattr(outcome, attribute)))
        ]
        if values:
            case_means.append(sum(values) / len(values))
    return sum(case_means) / len(case_means) if case_means else None


def _case_macro_latency_p95(
    measured: list[TrialRecord],
    track_cases: list[BenchmarkCase],
) -> float | None:
    """Return a case-macro warm-latency p95 for leaderboard ranking.

    Each mandatory case first collapses its measured repeats to a mean
    ``cost_time_ms``. The leaderboard then takes the nearest-rank p95 over
    those case means so every case has equal weight regardless of repeat
    count, ``batch_size``, or waypoint difficulty. Missing cases are omitted
    from the percentile (coverage / ``eligible`` already penalize skips);
    stratified absolute latency remains in the Time & Memory table.
    """
    if not track_cases:
        return None

    costs_by_case: dict[str, list[float]] = defaultdict(list)
    for record in measured:
        if record.cost_time_ms is None or not math.isfinite(float(record.cost_time_ms)):
            continue
        costs_by_case[record.case_id].append(float(record.cost_time_ms))

    case_means = [
        sum(costs_by_case[case.case_id]) / len(costs_by_case[case.case_id])
        for case in track_cases
        if case.case_id in costs_by_case
    ]
    return nearest_rank_percentile(case_means, 95.0)


def _top_failure(outcomes: list[CaseOutcome]) -> str | None:
    """Return the most frequent non-empty external failure code."""
    failures = Counter(
        outcome.failure_code for outcome in outcomes if outcome.failure_code
    )
    return failures.most_common(1)[0][0] if failures else None


def _peak_gpu(records: Iterable[TrialRecord]) -> float | None:
    """Return the maximum observed peak GPU MB, or ``None`` when unavailable."""
    peaks = [
        float(record.peak_gpu_mb)
        for record in records
        if record.peak_gpu_mb is not None and math.isfinite(float(record.peak_gpu_mb))
    ]
    return max(peaks) if peaks else None


def _track_ids(records: list[TrialRecord], cases: list[BenchmarkCase]) -> list[str]:
    """Return deterministic track ids observed in cases or records."""
    tracks = {case.track for case in cases}
    tracks.update(record.track for record in records)
    return sorted(tracks)


def _lifecycle_value(
    records: list[TrialRecord],
    track: str,
    algorithm_id: str,
    batch_size: int,
    phase: TrialPhase,
    waypoint_count: int | None = None,
) -> float | None:
    """Return the first lifecycle cost for one track, algorithm, and batch size.

    When ``waypoint_count`` is provided, only records for that waypoint shape
    match. This keeps first-case ``cold_plan_ms`` from being reused on every
    waypoint row in the Time & Memory table.
    """
    for record in records:
        if (
            record.track == track
            and record.algorithm_id == algorithm_id
            and record.batch_size == batch_size
            and record.phase is phase
            and (waypoint_count is None or record.waypoint_count == waypoint_count)
        ):
            return record.cost_time_ms
    return None


def _performance_rows(
    records: list[TrialRecord],
    metadata: list[PlannerMetadata],
    cases: list[BenchmarkCase],
) -> list[dict[str, object]]:
    """Aggregate steady-state time and memory by track, algorithm, and input shape."""
    measured_groups: dict[tuple[str, str, int, int], list[TrialRecord]] = defaultdict(
        list
    )
    for record in records:
        if record.phase is TrialPhase.MEASURED:
            measured_groups[
                (
                    record.track,
                    record.algorithm_id,
                    record.batch_size,
                    record.waypoint_count,
                )
            ].append(record)

    metadata_by_id = {item.algorithm_id: item for item in metadata}
    rows: list[dict[str, object]] = []
    for key in sorted(measured_groups):
        track, algorithm_id, batch_size, waypoint_count = key
        group = measured_groups[key]
        info = metadata_by_id[algorithm_id]
        costs = [record.cost_time_ms for record in group]
        mean_cost = _mean(costs)
        rows.append(
            {
                "track": track,
                "algorithm": algorithm_id,
                "algorithm_role": info.algorithm_role.value,
                "batch_size": batch_size,
                "waypoint_count": waypoint_count,
                "num_trials": len(group),
                "planner_construct_ms": _lifecycle_value(
                    records, track, algorithm_id, batch_size, TrialPhase.CONSTRUCT
                ),
                "backend_prepare_ms": _lifecycle_value(
                    records, track, algorithm_id, batch_size, TrialPhase.PREPARE
                ),
                "cold_plan_ms": _lifecycle_value(
                    records,
                    track,
                    algorithm_id,
                    batch_size,
                    TrialPhase.COLD,
                    waypoint_count=waypoint_count,
                ),
                "cost_time_ms": mean_cost,
                "warm_plan_ms_p50": nearest_rank_percentile(costs, 50.0),
                "warm_plan_ms_p95": nearest_rank_percentile(costs, 95.0),
                "latency_per_env_ms": (
                    mean_cost / batch_size if mean_cost is not None else None
                ),
                "cost_time_per_waypoint_ms": (
                    mean_cost / waypoint_count if mean_cost is not None else None
                ),
                "trajectories_per_second": (
                    batch_size * 1000.0 / mean_cost
                    if mean_cost is not None and mean_cost > 0.0
                    else None
                ),
                "cpu_delta_mb": _mean(record.cpu_delta_mb for record in group),
                "gpu_delta_mb": _mean(record.gpu_delta_mb for record in group),
                "peak_gpu_mb": _peak_gpu(group),
            }
        )

    present = {(row["track"], row["algorithm"]) for row in rows}
    for track in _track_ids(records, cases):
        for info in metadata:
            if (track, info.algorithm_id) in present:
                continue
            rows.append(
                {
                    "track": track,
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
                    "cost_time_per_waypoint_ms": None,
                    "trajectories_per_second": None,
                    "cpu_delta_mb": None,
                    "gpu_delta_mb": None,
                    "peak_gpu_mb": None,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row["track"]),
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
    """Aggregate external success and quality metrics by scenario condition.

    Boolean success columns use the same case-level macro average as the
    leaderboard (missing cases contribute ``0.0``). Continuous quality metrics
    remain conditioned on externally motion-valid outcomes.
    """
    measured_by_key: dict[
        tuple[str, str, str, int, int, str, str], list[TrialRecord]
    ] = defaultdict(list)
    for record in records:
        if record.phase is not TrialPhase.MEASURED:
            continue
        key = (
            record.track,
            record.algorithm_id,
            record.scenario_id,
            record.batch_size,
            record.waypoint_count,
            record.path_shape,
            record.start_state_bin,
        )
        measured_by_key[key].append(record)

    expected_by_group: Counter[tuple[str, str, int, int, str, str]] = Counter()
    cases_by_group: dict[tuple[str, str, int, int, str, str], list[BenchmarkCase]] = (
        defaultdict(list)
    )
    for case in cases:
        key = (
            case.track,
            case.scenario_id,
            case.batch_size,
            case.num_waypoints,
            case.path_shape,
            case.start_state_bin,
        )
        expected_by_group[key] += case.batch_size * measured_trials
        cases_by_group[key].append(case)

    rows: list[dict[str, object]] = []
    for info in metadata:
        for group_key in sorted(expected_by_group):
            (
                track,
                scenario_id,
                batch_size,
                waypoint_count,
                path_shape,
                start_state_bin,
            ) = group_key
            group_cases = cases_by_group[group_key]
            measured = measured_by_key.get(
                (
                    track,
                    info.algorithm_id,
                    scenario_id,
                    batch_size,
                    waypoint_count,
                    path_shape,
                    start_state_bin,
                ),
                [],
            )
            outcomes = [outcome for record in measured for outcome in record.outcomes]
            valid_outcomes = [outcome for outcome in outcomes if outcome.motion_valid]
            expected = expected_by_group[group_key]
            rows.append(
                {
                    "track": track,
                    "scenario": scenario_id,
                    "algorithm": info.algorithm_id,
                    "algorithm_role": info.algorithm_role.value,
                    "batch_size": batch_size,
                    "waypoint_count": waypoint_count,
                    "path_shape": path_shape,
                    "start_state_bin": start_state_bin,
                    "cases": len(group_cases),
                    "coverage_rate": min(1.0, len(outcomes) / max(expected, 1)),
                    # Free-space primary success is external motion validity.
                    "success_rate": _case_macro_rate(
                        measured, group_cases, "motion_valid"
                    ),
                    "planning_success_rate": _case_macro_rate(
                        measured, group_cases, "planning_success"
                    ),
                    "ordered_waypoint_success_rate": _case_macro_rate(
                        measured, group_cases, "ordered_waypoints_reached"
                    ),
                    "waypoint_completion_rate": _case_macro_mean(
                        measured, group_cases, "completed_waypoint_ratio"
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
                    "joint_violation_rate": _case_macro_rate(
                        measured, group_cases, "joint_limit_violation"
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
    """Build a complete success/coverage/latency ordered leaderboard per track.

    Success rates are macro-averaged over mandatory cases (equal case weight).
    ``latency_p95_ms`` uses the same case-equal weighting: per-case mean warm
    latency, then nearest-rank p95 across cases. ``coverage_rate`` remains an
    outcome-count completeness check used for eligibility.
    """
    entries: list[dict[str, object]] = []
    for track in _track_ids(records, cases):
        track_cases = [case for case in cases if case.track == track]
        expected_outcomes = (
            sum(case.batch_size for case in track_cases) * measured_trials
        )
        track_entries: list[dict[str, object]] = []
        for info in metadata:
            measured = [
                record
                for record in records
                if record.algorithm_id == info.algorithm_id
                and record.track == track
                and record.phase is TrialPhase.MEASURED
            ]
            outcomes = [outcome for record in measured for outcome in record.outcomes]
            coverage = min(1.0, len(outcomes) / max(expected_outcomes, 1))
            motion_rate = _case_macro_rate(measured, track_cases, "motion_valid")
            planning_rate = _case_macro_rate(measured, track_cases, "planning_success")
            latency_p95 = _case_macro_latency_p95(measured, track_cases)
            peak_gpu = _peak_gpu(measured)
            track_entries.append(
                {
                    "track": track,
                    "algorithm": info.algorithm_id,
                    "algorithm_role": info.algorithm_role.value,
                    "model_revision": info.model_revision,
                    "planner_config_hash": info.config_hash[:12],
                    "eligible": coverage >= 1.0 - 1.0e-12,
                    "coverage_rate": coverage,
                    # free-space v1: primary_success == motion_valid
                    "overall_success_rate": motion_rate,
                    "planning_success_rate": planning_rate,
                    "motion_valid_rate": motion_rate,
                    "task_success_rate": None,
                    "latency_p95_ms": latency_p95,
                    "peak_gpu_mb": peak_gpu,
                }
            )

        track_entries.sort(
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
        entries.extend(
            {"rank": rank, **entry} for rank, entry in enumerate(track_entries, start=1)
        )
    return entries


def aggregate_results(
    records: list[TrialRecord],
    metadata: list[PlannerMetadata],
    cases: list[BenchmarkCase],
    measured_trials: int,
) -> dict[str, list[dict[str, object]]]:
    """Build all three report datasets from raw numeric records."""
    return {
        "time_and_memory": _performance_rows(records, metadata, cases),
        "success_and_metrics": _metric_rows(records, metadata, cases, measured_trials),
        "leaderboard": _leaderboard_rows(records, metadata, cases, measured_trials),
    }
