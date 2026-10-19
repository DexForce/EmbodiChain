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
from dataclasses import replace

from .metrics.stats import nearest_rank_percentile
from .generalization import aggregate_generalization
from .models import BenchmarkCase, CaseOutcome, PlannerMetadata, TrialPhase, TrialRecord

__all__ = ["aggregate_generalization", "aggregate_results"]

_EFFECTIVE_MOTION_ALGORITHM_ROLE = "effective_motion_algorithm"


def _mean(values: Iterable[float | None]) -> float | None:
    """Return the mean of finite values or ``None`` when unavailable."""
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(finite) / len(finite) if finite else None


def _mean_unbounded(values: Iterable[float | None]) -> float | None:
    """Return the mean of measured values, keeping an unbounded one unbounded.

    A position change across a zero interval is recorded as an infinite peak.
    :func:`_mean` would drop it as non-finite and summarize the remaining
    trajectories as if it had never happened, so here any infinite value makes
    the mean infinite. NaN still carries no measurement and is ignored.
    """
    measured = [
        float(value)
        for value in values
        if value is not None and not math.isnan(float(value))
    ]
    if not measured:
        return None
    if any(math.isinf(value) for value in measured):
        return math.inf
    return sum(measured) / len(measured)


def _mean_dynamic(
    outcomes: Iterable[CaseOutcome],
    attribute: str,
    *,
    violation_attribute: str | None = None,
) -> float | None:
    """Average one dynamic metric while restoring JSON-safe unbounded rows.

    Raw artifacts encode non-finite numbers as ``null`` and retain
    ``dynamic_measurement_status="unbounded"``. Peak fields are therefore
    unbounded whenever any such row is present. Utilization fields additionally
    require their matching limit to apply; otherwise ``null`` means N/A rather
    than an infinite utilization.
    """
    rows = list(outcomes)
    if any(
        outcome.dynamic_measurement_status == "unbounded"
        and (
            violation_attribute is None or getattr(outcome, violation_attribute) is True
        )
        for outcome in rows
    ):
        return math.inf
    return _mean_unbounded(getattr(outcome, attribute) for outcome in rows)


def _dynamic_measurement_status(outcomes: Iterable[CaseOutcome]) -> str:
    """Summarize whether dynamic measurements are finite or unbounded."""
    rows = list(outcomes)
    numeric_fields = (
        "max_joint_velocity_rad_s",
        "max_joint_acceleration_rad_s2",
        "max_joint_jerk_rad_s3",
        "velocity_utilization",
        "acceleration_utilization",
        "jerk_utilization",
    )
    if any(
        outcome.dynamic_measurement_status == "unbounded"
        or any(
            value is not None and math.isinf(float(value))
            for value in (getattr(outcome, field) for field in numeric_fields)
        )
        for outcome in rows
    ):
        return "unbounded"
    if any(
        outcome.dynamic_measurement_status == "finite"
        or any(getattr(outcome, field) is not None for field in numeric_fields)
        for outcome in rows
    ):
        return "finite"
    return "unavailable"


def _case_macro_rate(
    measured: list[TrialRecord],
    track_cases: list[BenchmarkCase],
    attribute: str,
    *,
    measured_trials: int = 1,
    excluded_case_ids: frozenset[str] = frozenset(),
) -> float:
    """Macro-average a boolean outcome attribute equally across mandatory cases.

    Within each case, env rows (and measured repeats) are micro-averaged first.
    Cases then receive equal weight regardless of ``batch_size``, so a B=64
    case cannot dominate a B=1 case on the leaderboard. Missing cases
    contribute ``0.0`` so selective skipping cannot inflate the rate.
    """
    track_cases = [
        case for case in track_cases if case.case_id not in excluded_case_ids
    ]
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
            / max(len(outcomes), case.batch_size * measured_trials)
        )
    return sum(case_rates) / len(case_rates)


def _case_macro_rate_optional(
    measured: list[TrialRecord],
    track_cases: list[BenchmarkCase],
    attribute: str,
) -> float | None:
    """Macro-average a tri-state outcome attribute, reporting N/A as ``None``.

    Unlike :func:`_case_macro_rate`, a ``None`` outcome means the check did not
    apply rather than that it passed, so it is excluded from the average and a
    case with no applicable outcome is skipped entirely. When no case applies
    the metric is ``None``, which keeps an unconfigured limit from being
    reported as a zero violation rate.
    """
    if not track_cases:
        return None

    outcomes_by_case: dict[str, list[CaseOutcome]] = defaultdict(list)
    for record in measured:
        outcomes_by_case[record.case_id].extend(record.outcomes)

    case_rates: list[float] = []
    for case in track_cases:
        applicable = [
            outcome
            for outcome in outcomes_by_case.get(case.case_id, [])
            if getattr(outcome, attribute) is not None
        ]
        if not applicable:
            continue
        case_rates.append(
            sum(bool(getattr(outcome, attribute)) for outcome in applicable)
            / len(applicable)
        )
    return sum(case_rates) / len(case_rates) if case_rates else None


def _timed_motion_rate(
    measured: list[TrialRecord],
    cases: list[BenchmarkCase],
    *,
    measured_trials: int,
    fallback_timing: bool,
    timing_records: list[TrialRecord] | None = None,
    excluded_case_ids: frozenset[str] = frozenset(),
) -> float | None:
    """Score solved-clock cases against their full expected row/repeat population."""
    by_case: dict[str, list[TrialRecord]] = defaultdict(list)
    for record in measured:
        by_case[record.case_id].append(record)
    timing_by_case: dict[str, list[TrialRecord]] = defaultdict(list)
    for record in measured if timing_records is None else timing_records:
        timing_by_case[record.case_id].append(record)
    rates = []
    for case in cases:
        if case.case_id in excluded_case_ids:
            continue
        rows = by_case.get(case.case_id, [])
        solved = [
            record
            for record in rows
            if record.metadata.get("timing_is_solved", fallback_timing) is True
        ]
        known = timing_by_case.get(case.case_id, [])
        timing_applies = (
            any(
                record.metadata.get("timing_is_solved", fallback_timing) is True
                for record in known
            )
            if known
            else fallback_timing
        )
        if not timing_applies:
            continue
        outcomes = [outcome for record in solved for outcome in record.outcomes]
        rates.append(
            sum(outcome.timed_motion_valid is True for outcome in outcomes)
            / max(len(outcomes), case.batch_size * measured_trials)
        )
    return sum(rates) / len(rates) if rates else None


def _case_supports_attribute(case: BenchmarkCase, attribute: str) -> bool:
    """Return whether a staged outcome applies to one case protocol."""
    if attribute in {"task_success", "physical_skill_success", "end_to_end_success"}:
        return case.primary_success in {
            "task_success",
            "physical_skill_success",
            "end_to_end_success",
        }
    if attribute == "execution_success":
        return case.primary_success in {
            "execution_success",
            "task_success",
            "physical_skill_success",
            "end_to_end_success",
        }
    return True


def _case_macro_optional_rate(
    measured: list[TrialRecord],
    track_cases: list[BenchmarkCase],
    attribute: str,
    *,
    measured_trials: int = 1,
    excluded_case_ids: frozenset[str] = frozenset(),
) -> float | None:
    """Macro-average an applicable staged boolean, otherwise return ``None``."""
    applicable = [
        case
        for case in track_cases
        if case.case_id not in excluded_case_ids
        and _case_supports_attribute(case, attribute)
    ]
    if not applicable:
        return None
    if attribute in {"physical_skill_success", "end_to_end_success"} and not any(
        getattr(outcome, attribute) is not None
        for record in measured
        if record.case_id in {case.case_id for case in applicable}
        for outcome in record.outcomes
    ):
        return None
    return _case_macro_rate(
        measured,
        applicable,
        attribute,
        excluded_case_ids=excluded_case_ids,
        measured_trials=measured_trials,
    )


def _case_macro_primary_rate(
    measured: list[TrialRecord],
    track_cases: list[BenchmarkCase],
    *,
    measured_trials: int = 1,
    excluded_case_ids: frozenset[str] = frozenset(),
) -> float:
    """Macro-average each case's explicitly declared primary success stage."""
    track_cases = [
        case for case in track_cases if case.case_id not in excluded_case_ids
    ]
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
            sum(bool(getattr(outcome, case.primary_success)) for outcome in outcomes)
            / max(len(outcomes), case.batch_size * measured_trials)
        )
    return sum(case_rates) / len(case_rates)


def _case_macro_mean(
    measured: list[TrialRecord],
    track_cases: list[BenchmarkCase],
    attribute: str,
    *,
    excluded_case_ids: frozenset[str] = frozenset(),
) -> float | None:
    """Macro-average a numeric outcome attribute across cases with values."""
    track_cases = [
        case for case in track_cases if case.case_id not in excluded_case_ids
    ]
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


def _unsupported_case_ids(records: Iterable[TrialRecord]) -> frozenset[str]:
    """Return capability-gated cases excluded from success denominators."""
    by_case_source: dict[str, dict[str, list[TrialRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_case_source[record.case_id][record.algorithm_id].append(record)
    return frozenset(
        case_id
        for case_id, sources in by_case_source.items()
        if all(_source_unsupported(rows) for rows in sources.values())
    )


def _source_unsupported(records: list[TrialRecord]) -> bool:
    """Return whether a source was explicitly gated without measured evidence."""
    return not any(record.phase is TrialPhase.MEASURED for record in records) and any(
        record.phase is TrialPhase.AVAILABILITY
        and record.status == "unsupported"
        and record.failure_code in {"unsupported_capability", "unsupported_capacity"}
        for record in records
    )


def _rate_cases(
    cases: list[BenchmarkCase], records: list[TrialRecord], *, effective: bool
) -> list[BenchmarkCase]:
    """Retain each case's weight while counting every applicable source repeat."""
    if not effective:
        return cases
    by_case_source: dict[str, dict[str, list[TrialRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_case_source[record.case_id][record.algorithm_id].append(record)
    return [
        replace(
            case,
            batch_size=case.batch_size
            * max(
                1,
                sum(
                    not _source_unsupported(rows)
                    for rows in by_case_source[case.case_id].values()
                ),
            ),
        )
        for case in cases
    ]


def _stage_success_rates(
    measured: list[TrialRecord],
    cases: list[BenchmarkCase],
    *,
    excluded_case_ids: frozenset[str],
) -> dict[str, float | None] | None:
    """Compute chained physical stage rates from measured observations."""
    counts = _stage_counts(measured, cases, excluded_case_ids=excluded_case_ids)
    if counts is None:
        return None
    return {
        name: values["passed"] / values["reached"] if values["reached"] else None
        for name, values in counts.items()
    }


def _stage_counts(
    measured: list[TrialRecord],
    cases: list[BenchmarkCase],
    *,
    excluded_case_ids: frozenset[str],
) -> dict[str, dict[str, int]] | None:
    """Count row/repeat observations; missing evidence never enters reached."""
    outcomes_by_case: dict[str, list[CaseOutcome]] = defaultdict(list)
    for record in measured:
        outcomes_by_case[record.case_id].extend(record.outcomes)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        if case.case_id in excluded_case_ids:
            continue
        for outcome in outcomes_by_case.get(case.case_id, []):
            prefix_passed = True
            for stage in outcome.stages:
                counts[stage.stage_id][stage.status] += 1
                if not prefix_passed or stage.status in {
                    "not_applicable",
                    "not_reached",
                    "unavailable",
                }:
                    if stage.status != "not_applicable":
                        prefix_passed = False
                    continue
                counts[stage.stage_id]["reached"] += 1
                if stage.status != "passed":
                    prefix_passed = False
    if not counts:
        return None
    return {
        stage_id: {
            key: counts[stage_id][key]
            for key in (
                "reached",
                "passed",
                "failed",
                "not_reached",
                "not_applicable",
                "unavailable",
            )
        }
        for stage_id in sorted(counts)
    }


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


def _bypassed_case_ids(
    records: Iterable[TrialRecord], algorithm_id: str, track: str
) -> frozenset[str]:
    """Return cases whose scenario did not call the selected planner backend."""
    return frozenset(
        record.case_id
        for record in records
        if record.algorithm_id == algorithm_id
        and record.track == track
        and not record.uses_selected_planner_backend
    )


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
            and record.uses_selected_planner_backend
            and (waypoint_count is None or record.waypoint_count == waypoint_count)
        ):
            return record.cost_time_ms
    return None


def _performance_rows(
    records: list[TrialRecord],
    metadata: list[PlannerMetadata],
    cases: list[BenchmarkCase],
) -> list[dict[str, object]]:
    """Aggregate planner-attributed time and memory by track and input shape."""
    measured_groups: dict[
        tuple[str, str, str, str, str | None, str | None, int, int],
        list[TrialRecord],
    ] = defaultdict(list)
    for record in records:
        if record.phase is TrialPhase.MEASURED and record.uses_selected_planner_backend:
            measured_groups[
                (
                    record.track,
                    record.algorithm_id,
                    record.robot_id,
                    record.skill_id,
                    record.object_id,
                    record.task_difficulty,
                    record.batch_size,
                    record.waypoint_count,
                )
            ].append(record)

    metadata_by_id = {item.algorithm_id: item for item in metadata}
    rows: list[dict[str, object]] = []
    for key in sorted(
        measured_groups,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            item[3],
            item[4] or "",
            item[5] or "",
            item[6],
            item[7],
        ),
    ):
        (
            track,
            algorithm_id,
            robot_id,
            skill_id,
            object_id,
            task_difficulty,
            batch_size,
            waypoint_count,
        ) = key
        group = measured_groups[key]
        info = metadata_by_id[algorithm_id]
        costs = [record.cost_time_ms for record in group]
        mean_cost = _mean(costs)
        rows.append(
            {
                "track": track,
                "algorithm": algorithm_id,
                "algorithm_role": info.algorithm_role.value,
                "robot": robot_id,
                "skill": skill_id,
                "object": object_id,
                "task_difficulty": task_difficulty,
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
                "execution_time_ms": _mean(
                    record.execution_time_ms for record in group
                ),
                "end_to_end_time_ms": _mean(
                    record.end_to_end_time_ms for record in group
                ),
                "timing_is_solved": info.timing_is_solved,
                # A planner that never solves timing reports a nominal dt, so
                # its duration is not comparable with a solved one.
                "trajectory_duration_s": (
                    _mean(record.trajectory_duration_s for record in group)
                    if info.timing_is_solved
                    else None
                ),
                "trajectory_waypoints": _mean(
                    (
                        float(record.trajectory_waypoints)
                        if record.trajectory_waypoints is not None
                        else None
                    )
                    for record in group
                ),
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
                    "robot": None,
                    "skill": None,
                    "object": None,
                    "task_difficulty": None,
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
                    "execution_time_ms": None,
                    "end_to_end_time_ms": None,
                    "trajectory_duration_s": None,
                    "trajectory_waypoints": None,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row["track"]),
            str(row["algorithm"]),
            int(row["batch_size"] or 0),
            int(row["waypoint_count"] or 0),
            str(row["skill"] or ""),
            str(row["object"] or ""),
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
    remain conditioned on externally geometry-valid outcomes. Cases whose
    scenario bypasses the selected planner are excluded from planner metrics
    and reported under the algorithm that actually produced their motion.
    """
    measured_by_key: dict[
        tuple[
            str,
            tuple[str, bool],
            str,
            str,
            str,
            str | None,
            str | None,
            str,
            int,
            int,
            str,
            str,
        ],
        list[TrialRecord],
    ] = defaultdict(list)
    bypassed_by_algorithm_track: dict[tuple[str, str], set[str]] = defaultdict(set)
    effective_case_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    effective_algorithms: set[str] = set()
    for record in records:
        is_effective = not record.uses_selected_planner_backend
        report_key = (record.effective_motion_algorithm, is_effective)
        if is_effective:
            bypassed_by_algorithm_track[(record.algorithm_id, record.track)].add(
                record.case_id
            )
            effective_algorithms.add(record.effective_motion_algorithm)
            effective_case_sources[
                (
                    record.effective_motion_algorithm,
                    record.track,
                    record.case_id,
                )
            ].add(record.algorithm_id)
        if record.phase is not TrialPhase.MEASURED:
            continue
        key = (
            record.track,
            report_key,
            record.scenario_id,
            record.robot_id,
            record.skill_id,
            record.object_id,
            record.task_difficulty,
            record.primary_success,
            record.batch_size,
            record.waypoint_count,
            record.path_shape,
            record.start_state_bin,
        )
        measured_by_key[key].append(record)

    expected_by_group: Counter[
        tuple[
            str,
            str,
            str,
            str,
            str | None,
            str | None,
            str,
            int,
            int,
            str,
            str,
        ]
    ] = Counter()
    cases_by_group: dict[
        tuple[
            str,
            str,
            str,
            str,
            str | None,
            str | None,
            str,
            int,
            int,
            str,
            str,
        ],
        list[BenchmarkCase],
    ] = defaultdict(list)
    for case in cases:
        key = (
            case.track,
            case.scenario_id,
            case.robot_id,
            case.skill_id,
            case.object_id,
            case.task_difficulty,
            case.primary_success,
            case.batch_size,
            case.num_waypoints,
            case.path_shape,
            case.start_state_bin,
        )
        expected_by_group[key] += case.batch_size * measured_trials
        cases_by_group[key].append(case)

    rows: list[dict[str, object]] = []
    report_algorithms = [
        ((info.algorithm_id, False), info.algorithm_role.value) for info in metadata
    ]
    report_algorithms.extend(
        ((algorithm_id, True), _EFFECTIVE_MOTION_ALGORITHM_ROLE)
        for algorithm_id in sorted(effective_algorithms)
    )
    for report_key, algorithm_role in report_algorithms:
        algorithm_id, is_effective = report_key
        for group_key in sorted(
            expected_by_group,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
                item[4] or "",
                item[5] or "",
                item[6],
                item[7],
                item[8],
                item[9],
                item[10],
            ),
        ):
            (
                track,
                scenario_id,
                robot_id,
                skill_id,
                object_id,
                task_difficulty,
                primary_success,
                batch_size,
                waypoint_count,
                path_shape,
                start_state_bin,
            ) = group_key
            if is_effective:
                group_cases = [
                    case
                    for case in cases_by_group[group_key]
                    if effective_case_sources.get((algorithm_id, track, case.case_id))
                ]
            else:
                bypassed_case_ids = bypassed_by_algorithm_track[(algorithm_id, track)]
                group_cases = [
                    case
                    for case in cases_by_group[group_key]
                    if case.case_id not in bypassed_case_ids
                ]
            if not group_cases:
                continue
            measured = measured_by_key.get(
                (
                    track,
                    report_key,
                    scenario_id,
                    robot_id,
                    skill_id,
                    object_id,
                    task_difficulty,
                    primary_success,
                    batch_size,
                    waypoint_count,
                    path_shape,
                    start_state_bin,
                ),
                [],
            )
            population_records = [
                record
                for record in records
                if record.track == track
                and (
                    record.effective_motion_algorithm,
                    not record.uses_selected_planner_backend,
                )
                == report_key
            ]
            excluded_case_ids = _unsupported_case_ids(population_records)
            rate_cases = _rate_cases(
                group_cases, population_records, effective=is_effective
            )
            outcomes = [outcome for record in measured for outcome in record.outcomes]
            valid_outcomes = [outcome for outcome in outcomes if outcome.geometry_valid]
            expected = sum(
                case.batch_size
                * measured_trials
                * (
                    len(effective_case_sources[(algorithm_id, track, case.case_id)])
                    if is_effective
                    else 1
                )
                for case in group_cases
            )
            rows.append(
                {
                    "track": track,
                    "scenario": scenario_id,
                    "algorithm": algorithm_id,
                    "algorithm_role": algorithm_role,
                    "robot": robot_id,
                    "skill": skill_id,
                    "object": object_id,
                    "task_difficulty": task_difficulty,
                    "primary_success": primary_success,
                    "batch_size": batch_size,
                    "waypoint_count": waypoint_count,
                    "path_shape": path_shape,
                    "start_state_bin": start_state_bin,
                    "cases": len(group_cases),
                    "n_valid": len(valid_outcomes),
                    "coverage_rate": min(1.0, len(outcomes) / max(expected, 1)),
                    "success_rate": _case_macro_primary_rate(
                        measured,
                        rate_cases,
                        excluded_case_ids=excluded_case_ids,
                        measured_trials=measured_trials,
                    ),
                    "planning_success_rate": _case_macro_rate(
                        measured,
                        rate_cases,
                        "planning_success",
                        excluded_case_ids=excluded_case_ids,
                        measured_trials=measured_trials,
                    ),
                    "geometry_valid_rate": _case_macro_rate(
                        measured,
                        rate_cases,
                        "geometry_valid",
                        measured_trials=measured_trials,
                        excluded_case_ids=excluded_case_ids,
                    ),
                    "timed_motion_valid_rate": _timed_motion_rate(
                        measured,
                        rate_cases,
                        measured_trials=measured_trials,
                        timing_records=population_records,
                        fallback_timing=next(
                            (
                                info.timing_is_solved
                                for info in metadata
                                if info.algorithm_id == algorithm_id
                            ),
                            False,
                        ),
                        excluded_case_ids=excluded_case_ids,
                    ),
                    "motion_valid_rate": _case_macro_rate(
                        measured,
                        rate_cases,
                        "motion_valid",
                        excluded_case_ids=excluded_case_ids,
                        measured_trials=measured_trials,
                    ),
                    "execution_success_rate": _case_macro_optional_rate(
                        measured,
                        rate_cases,
                        "execution_success",
                        excluded_case_ids=excluded_case_ids,
                        measured_trials=measured_trials,
                    ),
                    "task_success_rate": _case_macro_optional_rate(
                        measured,
                        rate_cases,
                        "task_success",
                        excluded_case_ids=excluded_case_ids,
                        measured_trials=measured_trials,
                    ),
                    "ordered_waypoint_success_rate": _case_macro_rate(
                        measured,
                        rate_cases,
                        "ordered_waypoints_reached",
                        excluded_case_ids=excluded_case_ids,
                        measured_trials=measured_trials,
                    ),
                    "physical_skill_success_rate": _case_macro_optional_rate(
                        measured,
                        rate_cases,
                        "physical_skill_success",
                        excluded_case_ids=excluded_case_ids,
                        measured_trials=measured_trials,
                    ),
                    "end_to_end_success_rate": _case_macro_optional_rate(
                        measured,
                        rate_cases,
                        "end_to_end_success",
                        excluded_case_ids=excluded_case_ids,
                        measured_trials=measured_trials,
                    ),
                    "evaluation_protocols": sorted(
                        {
                            outcome.evaluation_protocol or "legacy/unversioned"
                            for outcome in outcomes
                        }
                    ),
                    "physical_evaluation_coverage_rate": (
                        sum(
                            outcome.physical_skill_success is not None
                            for outcome in outcomes
                        )
                        / max(expected, 1)
                    ),
                    "stage_counts": _stage_counts(
                        measured,
                        rate_cases,
                        excluded_case_ids=excluded_case_ids,
                    ),
                    "waypoint_completion_rate": _case_macro_mean(
                        measured,
                        rate_cases,
                        "completed_waypoint_ratio",
                        excluded_case_ids=excluded_case_ids,
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
                        measured,
                        rate_cases,
                        "joint_limit_violation",
                        excluded_case_ids=excluded_case_ids,
                        measured_trials=measured_trials,
                    ),
                    "velocity_violation_rate": _case_macro_rate_optional(
                        measured, group_cases, "velocity_limit_violation"
                    ),
                    "acceleration_violation_rate": _case_macro_rate_optional(
                        measured, group_cases, "acceleration_limit_violation"
                    ),
                    "jerk_violation_rate": _case_macro_rate_optional(
                        measured, group_cases, "jerk_limit_violation"
                    ),
                    # Dynamics are summarized over every measured outcome, not
                    # the motion-valid subset: exceeding a limit can make an
                    # outcome invalid, so restricting the population here would
                    # drop exactly the trajectories with the largest peaks and
                    # pair a modest peak with a nonzero violation rate.
                    "max_joint_velocity_rad_s": _mean_dynamic(
                        outcomes, "max_joint_velocity_rad_s"
                    ),
                    "max_joint_acceleration_rad_s2": _mean_dynamic(
                        outcomes, "max_joint_acceleration_rad_s2"
                    ),
                    "max_joint_jerk_rad_s3": _mean_dynamic(
                        outcomes, "max_joint_jerk_rad_s3"
                    ),
                    "velocity_utilization": _mean_dynamic(
                        outcomes,
                        "velocity_utilization",
                        violation_attribute="velocity_limit_violation",
                    ),
                    "acceleration_utilization": _mean_dynamic(
                        outcomes,
                        "acceleration_utilization",
                        violation_attribute="acceleration_limit_violation",
                    ),
                    "jerk_utilization": _mean_dynamic(
                        outcomes,
                        "jerk_utilization",
                        violation_attribute="jerk_limit_violation",
                    ),
                    "dynamic_measurement_status": _dynamic_measurement_status(outcomes),
                    "joint_path_length_rad": _mean(
                        outcome.joint_path_length_rad for outcome in valid_outcomes
                    ),
                    "cartesian_path_length_m": _mean(
                        outcome.cartesian_path_length_m for outcome in valid_outcomes
                    ),
                    "path_efficiency": _mean(
                        outcome.path_efficiency for outcome in valid_outcomes
                    ),
                    "task_completion_time_s": _mean(
                        outcome.task_completion_time_s
                        for outcome in outcomes
                        if outcome.task_success
                    ),
                    "joint_tracking_rmse_rad": _mean(
                        outcome.joint_tracking_rmse_rad for outcome in outcomes
                    ),
                    "object_lift_delta_m": _mean(
                        outcome.object_lift_delta_m for outcome in outcomes
                    ),
                    "articulation_joint_delta": _mean(
                        outcome.articulation_joint_delta for outcome in outcomes
                    ),
                    "articulation_joint_peak_signed_delta": _mean(
                        outcome.articulation_joint_peak_signed_delta
                        for outcome in outcomes
                    ),
                    "replan_count": _mean(
                        (
                            float(outcome.replan_count)
                            if outcome.replan_count is not None
                            else None
                        )
                        for outcome in outcomes
                    ),
                    "stage_success_rates": _stage_success_rates(
                        measured,
                        rate_cases,
                        excluded_case_ids=excluded_case_ids,
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
    outcome-count completeness check used for eligibility. Cases explicitly
    attributed to another motion algorithm are excluded from both ranking
    numerators and denominators.
    """
    entries: list[dict[str, object]] = []
    for track in _track_ids(records, cases):
        all_track_cases = [case for case in cases if case.track == track]
        track_entries: list[dict[str, object]] = []
        for info in metadata:
            bypassed_case_ids = _bypassed_case_ids(records, info.algorithm_id, track)
            track_cases = [
                case
                for case in all_track_cases
                if case.case_id not in bypassed_case_ids
            ]
            expected_outcomes = (
                sum(case.batch_size for case in track_cases) * measured_trials
            )
            measured = [
                record
                for record in records
                if record.algorithm_id == info.algorithm_id
                and record.track == track
                and record.phase is TrialPhase.MEASURED
                and record.uses_selected_planner_backend
            ]
            outcomes = [outcome for record in measured for outcome in record.outcomes]
            excluded_case_ids = frozenset(
                record.case_id
                for record in records
                if record.algorithm_id == info.algorithm_id
                and record.track == track
                and record.phase is TrialPhase.AVAILABILITY
                and record.status == "unsupported"
                and record.failure_code
                in {"unsupported_capability", "unsupported_capacity"}
            )
            coverage = min(1.0, len(outcomes) / max(expected_outcomes, 1))
            motion_rate = _case_macro_rate(
                measured,
                track_cases,
                "motion_valid",
                excluded_case_ids=excluded_case_ids,
                measured_trials=measured_trials,
            )
            planning_rate = _case_macro_rate(
                measured,
                track_cases,
                "planning_success",
                excluded_case_ids=excluded_case_ids,
                measured_trials=measured_trials,
            )
            execution_rate = _case_macro_optional_rate(
                measured,
                track_cases,
                "execution_success",
                excluded_case_ids=excluded_case_ids,
                measured_trials=measured_trials,
            )
            task_rate = _case_macro_optional_rate(
                measured,
                track_cases,
                "task_success",
                excluded_case_ids=excluded_case_ids,
                measured_trials=measured_trials,
            )
            primary_rate = _case_macro_primary_rate(
                measured,
                track_cases,
                excluded_case_ids=excluded_case_ids,
                measured_trials=measured_trials,
            )
            latency_p95 = _case_macro_latency_p95(measured, track_cases)
            peak_gpu = _peak_gpu(measured)
            track_entries.append(
                {
                    "track": track,
                    "algorithm": info.algorithm_id,
                    "algorithm_role": info.algorithm_role.value,
                    "model_revision": info.model_revision,
                    "planner_config_hash": info.config_hash[:12],
                    "eligible": bool(track_cases) and coverage >= 1.0 - 1.0e-12,
                    "attributed_cases": len(track_cases),
                    "bypassed_cases": len(all_track_cases) - len(track_cases),
                    "coverage_rate": coverage,
                    "overall_success_rate": primary_rate,
                    "planning_success_rate": planning_rate,
                    "motion_valid_rate": motion_rate,
                    "execution_success_rate": execution_rate,
                    "task_success_rate": task_rate,
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


def _view_leaderboard(
    records: list[TrialRecord],
    metadata: list[PlannerMetadata],
    cases: list[BenchmarkCase],
    measured_trials: int,
    *,
    view: str,
    timing: bool = False,
) -> list[dict[str, object]]:
    """Rank one explicit scenario family without mixing clocks or timing scopes."""
    entries: list[dict[str, object]] = []
    info_by_id = {info.algorithm_id: info for info in metadata}
    identities = [(info.algorithm_id, False) for info in metadata]
    identities += sorted(
        {
            (record.effective_motion_algorithm, True)
            for record in records
            if not record.uses_selected_planner_backend
        }
    )
    for track in _track_ids(records, cases):
        track_cases = [case for case in cases if case.track == track]
        ranked = []
        for algorithm, effective in identities:
            population = [
                record
                for record in records
                if record.track == track
                and (
                    record.effective_motion_algorithm,
                    not record.uses_selected_planner_backend,
                )
                == (algorithm, effective)
            ]
            if effective:
                attributed_cases = [
                    case
                    for case in track_cases
                    if any(record.case_id == case.case_id for record in population)
                ]
            else:
                bypassed = _bypassed_case_ids(records, algorithm, track)
                attributed_cases = [
                    case for case in track_cases if case.case_id not in bypassed
                ]
            if effective and not attributed_cases:
                continue
            rate_cases = _rate_cases(attributed_cases, population, effective=effective)
            measured = [
                record for record in population if record.phase is TrialPhase.MEASURED
            ]
            excluded = _unsupported_case_ids(population)
            info = info_by_id.get(algorithm)
            solved = False if effective or info is None else info.timing_is_solved
            outcomes = [outcome for record in measured for outcome in record.outcomes]
            expected = sum(
                case.batch_size
                * measured_trials
                * (
                    len(
                        {
                            record.algorithm_id
                            for record in population
                            if record.case_id == case.case_id
                        }
                    )
                    if effective
                    else 1
                )
                for case in attributed_cases
            )
            coverage = min(1.0, len(outcomes) / max(1, expected))
            geometry = _case_macro_rate(
                measured,
                rate_cases,
                "geometry_valid",
                measured_trials=measured_trials,
                excluded_case_ids=excluded,
            )
            timed = _timed_motion_rate(
                measured,
                rate_cases,
                measured_trials=measured_trials,
                fallback_timing=solved,
                timing_records=population,
                excluded_case_ids=excluded,
            )
            primary = _case_macro_primary_rate(
                measured,
                rate_cases,
                measured_trials=measured_trials,
                excluded_case_ids=excluded,
            )
            success = (
                (timed if timing else geometry)
                if view == "motion_planning"
                else primary
            )
            if not any(case.case_id not in excluded for case in attributed_cases):
                success = None
            latency = _case_macro_latency_p95(measured, rate_cases)
            row = {
                "view": view,
                "ranking": (
                    "timed_motion"
                    if timing
                    else ("geometry" if view == "motion_planning" else "physical_skill")
                ),
                "track": track,
                "algorithm": algorithm,
                "algorithm_role": (
                    _EFFECTIVE_MOTION_ALGORITHM_ROLE
                    if effective
                    else info.algorithm_role.value
                ),
                "model_revision": None if effective else info.model_revision,
                "planner_config_hash": None if effective else info.config_hash[:12],
                "eligible": bool(attributed_cases)
                and coverage >= 1.0 - 1.0e-12
                and success is not None,
                "attributed_cases": len(attributed_cases),
                "bypassed_cases": len(track_cases) - len(attributed_cases),
                "coverage_rate": coverage,
                "overall_success_rate": success,
                "geometry_valid_rate": geometry,
                "timed_motion_valid_rate": timed,
                "planning_success_rate": _case_macro_rate(
                    measured,
                    rate_cases,
                    "planning_success",
                    measured_trials=measured_trials,
                    excluded_case_ids=excluded,
                ),
                "latency_p95_ms": latency if view == "motion_planning" else None,
                "compile_latency_p95_ms": latency if view == "atomic_skills" else None,
                "peak_gpu_mb": _peak_gpu(measured),
            }
            for attribute in (
                "motion_valid",
                "execution_success",
                "task_success",
                "physical_skill_success",
                "end_to_end_success",
            ):
                row[f"{attribute}_rate"] = _case_macro_optional_rate(
                    measured,
                    rate_cases,
                    attribute,
                    measured_trials=measured_trials,
                    excluded_case_ids=excluded,
                )
            ranked.append(row)
        ranked.sort(
            key=lambda row: (
                not row["eligible"],
                -(
                    float(row["overall_success_rate"])
                    if row["overall_success_rate"] is not None
                    else -1.0
                ),
                -float(row["coverage_rate"]),
                float(
                    row["latency_p95_ms"]
                    if row["latency_p95_ms"] is not None
                    else (
                        row["compile_latency_p95_ms"]
                        if row["compile_latency_p95_ms"] is not None
                        else math.inf
                    )
                ),
                str(row["algorithm"]),
            )
        )
        rank = 0
        for row in ranked:
            if row["overall_success_rate"] is not None:
                rank += 1
                row["rank"] = rank
            else:
                row["rank"] = None
            entries.append(row)
    return entries


def _view_performance(
    records: list[TrialRecord],
    metadata: list[PlannerMetadata],
    cases: list[BenchmarkCase],
    *,
    view: str,
) -> list[dict[str, object]]:
    """Attribute measured costs to the actual algorithm and label their scope."""
    metadata_by_id = {info.algorithm_id: info for info in metadata}
    identities = [(info.algorithm_id, False) for info in metadata]
    identities.extend(
        sorted(
            {
                (record.effective_motion_algorithm, True)
                for record in records
                if not record.uses_selected_planner_backend
            }
        )
    )
    internal_ids = {
        identity: f"report_identity_{index}"
        for index, identity in enumerate(identities)
    }
    display_ids = {internal: identity for identity, internal in internal_ids.items()}
    projected_metadata = []
    for algorithm, effective in identities:
        if effective:
            source = next(
                record
                for record in records
                if not record.uses_selected_planner_backend
                and record.effective_motion_algorithm == algorithm
            )
            info = metadata_by_id[source.algorithm_id]
            timing_is_solved = source.metadata.get("timing_is_solved") is True
        else:
            info = metadata_by_id[algorithm]
            timing_is_solved = info.timing_is_solved
        projected_metadata.append(
            replace(
                info,
                algorithm_id=internal_ids[(algorithm, effective)],
                timing_is_solved=timing_is_solved,
            )
        )

    projected = []
    for record in records:
        # Construction and backend preparation belong to the configured
        # adapter even when a later skill bypasses it. Do not charge these
        # lifecycle costs to the effective scenario-owned motion algorithm.
        identity = (
            (record.algorithm_id, False)
            if record.phase in (TrialPhase.CONSTRUCT, TrialPhase.PREPARE)
            else (
                record.effective_motion_algorithm,
                not record.uses_selected_planner_backend,
            )
        )
        projected.append(
            replace(
                record,
                algorithm_id=internal_ids[identity],
                metadata={**record.metadata, "selected_planner_backend_used": True},
            )
        )
    rows = _performance_rows(projected, projected_metadata, cases)
    for row in rows:
        algorithm, effective = display_ids[row["algorithm"]]
        row["algorithm"] = algorithm
        row["view"] = view
        group = [
            record
            for record in records
            if record.phase is TrialPhase.MEASURED
            and record.track == row["track"]
            and (
                record.effective_motion_algorithm,
                not record.uses_selected_planner_backend,
            )
            == (algorithm, effective)
            and record.robot_id == row["robot"]
            and record.skill_id == row["skill"]
            and record.object_id == row["object"]
            and record.task_difficulty == row["task_difficulty"]
            and record.batch_size == row["batch_size"]
            and record.waypoint_count == row["waypoint_count"]
        ]
        if group:
            solved = all(
                record.metadata.get(
                    "timing_is_solved",
                    metadata_by_id[record.algorithm_id].timing_is_solved,
                )
                is True
                for record in group
            )
            row["timing_is_solved"] = solved
            row["trajectory_duration_s"] = (
                _mean(record.trajectory_duration_s for record in group)
                if solved
                else None
            )
        if effective:
            row["algorithm_role"] = _EFFECTIVE_MOTION_ALGORITHM_ROLE
            row["planner_construct_ms"] = None
            row["backend_prepare_ms"] = None
        row["timing_scope"] = (
            "atomic_action_compile" if view == "atomic_skills" else "planner_call"
        )
        row["planning_time_ms"] = (
            row["cost_time_ms"] if view == "motion_planning" else None
        )
        row["compile_time_ms"] = (
            row["cost_time_ms"] if view == "atomic_skills" else None
        )
        row["compile_time_ms_p50"] = (
            row["warm_plan_ms_p50"] if view == "atomic_skills" else None
        )
        row["compile_time_ms_p95"] = (
            row["warm_plan_ms_p95"] if view == "atomic_skills" else None
        )
        if view == "atomic_skills":
            for name in (
                "warm_plan_ms_p50",
                "warm_plan_ms_p95",
                "latency_per_env_ms",
                "cost_time_per_waypoint_ms",
                "trajectories_per_second",
            ):
                row[name] = None
    return rows


def aggregate_results(
    records: list[TrialRecord],
    metadata: list[PlannerMetadata],
    cases: list[BenchmarkCase],
    measured_trials: int,
    *,
    track_scenarios: dict[str, str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build raw aggregates and scenario-owned report views with separate ranks.

    ``track_scenarios`` is the suite's expanded track-to-scenario mapping.
    Omitting it preserves legacy aggregates; it never infers a family from a
    skill id, scenario display name, or track naming convention.
    """
    result = {
        "time_and_memory": _performance_rows(records, metadata, cases),
        "success_and_metrics": _metric_rows(records, metadata, cases, measured_trials),
        "leaderboard": _leaderboard_rows(records, metadata, cases, measured_trials),
        "generalization": aggregate_generalization(
            records, metadata, cases, measured_trials
        ),
    }
    if track_scenarios is None:
        return result
    result["report_views"] = []
    for scenario, view, title in (
        ("free_space", "motion_planning", "Motion Planning"),
        ("atomic_skill", "atomic_skills", "Atomic Skills"),
    ):
        tracks = {
            track for track, owner in track_scenarios.items() if owner == scenario
        }
        if not tracks:
            continue
        view_cases = [case for case in cases if case.track in tracks]
        view_records = [record for record in records if record.track in tracks]
        result["report_views"].append(
            {
                "view": view,
                "title": title,
                "tracks": sorted(tracks),
                "schema_version": 3,
            }
        )
        result[f"{view}_time_and_memory"] = _view_performance(
            view_records, metadata, view_cases, view=view
        )
        result[f"{view}_success_and_metrics"] = [
            {
                **row,
                "view": view,
                "success_rate": (
                    row["geometry_valid_rate"]
                    if view == "motion_planning"
                    else row["success_rate"]
                ),
            }
            for row in result["success_and_metrics"]
            if row["track"] in tracks
        ]
        result[f"{view}_leaderboard"] = _view_leaderboard(
            view_records, metadata, view_cases, measured_trials, view=view
        )
        if view == "motion_planning":
            result["motion_timing_leaderboard"] = _view_leaderboard(
                view_records,
                metadata,
                view_cases,
                measured_trials,
                view=view,
                timing=True,
            )
            result["motion_geometry_leaderboard"] = result[f"{view}_leaderboard"]
        else:
            result["atomic_leaderboard"] = result[f"{view}_leaderboard"]
    return result
