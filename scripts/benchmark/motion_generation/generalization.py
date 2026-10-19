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

"""Domain-aware aggregation for configuration-driven benchmark suites."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from .contracts import EvaluationDomain
from .models import BenchmarkCase, PlannerMetadata, TrialPhase, TrialRecord

__all__ = ["aggregate_generalization"]

_EFFECTIVE_MOTION_ALGORITHM_ROLE = "effective_motion_algorithm"


def _track_group(case: BenchmarkCase) -> str:
    """Return the stable track identity shared by all domain populations."""
    return case.track_group or case.track


def _domain_key(domain: EvaluationDomain) -> tuple[str, str, str]:
    """Return a deterministic domain identity tuple."""
    return domain.id, domain.version, domain.kind


def _unsupported_case_ids(records: Iterable[TrialRecord]) -> set[str]:
    """Return cases excluded from success denominators by capability gates."""
    return {
        record.case_id
        for record in records
        if record.phase is TrialPhase.AVAILABILITY
        and record.status == "unsupported"
        and record.failure_code in {"unsupported_capability", "unsupported_capacity"}
    }


def _case_success(
    records: Iterable[TrialRecord],
    case: BenchmarkCase,
) -> tuple[float, int]:
    """Return a case-macro success rate and the number of observed rows."""
    outcomes = [
        outcome
        for record in records
        if record.phase is TrialPhase.MEASURED
        for outcome in record.outcomes
    ]
    if not outcomes:
        return 0.0, 0
    attribute = case.primary_success
    values = [bool(getattr(outcome, attribute, False)) for outcome in outcomes]
    return sum(values) / len(values), len(outcomes)


def _spatial_bucket_statistics(
    cases: Iterable[BenchmarkCase],
    records_by_case: dict[str, list[TrialRecord]],
    expected_repeats_by_case: dict[str, int],
) -> tuple[dict[str, float], dict[str, int], dict[str, dict[str, int]]]:
    """Aggregate row-level success and failure stages for spatial buckets."""
    successes: dict[str, int] = defaultdict(int)
    observed_counts: dict[str, int] = defaultdict(int)
    expected_counts: dict[str, int] = defaultdict(int)
    failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in cases:
        randomization = case.case_parameters.get("randomization", {})
        if not isinstance(randomization, dict):
            continue
        buckets = randomization.get("spatial_buckets")
        if buckets is None:
            continue
        if not isinstance(buckets, list) or len(buckets) != case.batch_size:
            raise ValueError(
                f"Case {case.case_id!r} spatial_buckets must match batch_size."
            )
        for bucket in buckets:
            expected_counts[str(bucket)] += expected_repeats_by_case.get(
                case.case_id, 0
            )
        for record in records_by_case.get(case.case_id, []):
            if record.phase is not TrialPhase.MEASURED:
                continue
            for outcome in record.outcomes:
                if not 0 <= outcome.env_index < len(buckets):
                    raise ValueError(
                        f"Case {case.case_id!r} outcome env_index is outside "
                        "spatial_buckets."
                    )
                bucket = str(buckets[outcome.env_index])
                observed_counts[bucket] += 1
                succeeded = bool(getattr(outcome, case.primary_success, False))
                successes[bucket] += int(succeeded)
                if not succeeded:
                    stage = outcome.failure_stage or outcome.failure_code or "unknown"
                    failures[bucket][stage] += 1
    counts = {
        bucket: max(expected_counts[bucket], observed_counts[bucket])
        for bucket in sorted(expected_counts.keys() | observed_counts.keys())
    }
    for bucket, count in counts.items():
        missing = count - observed_counts[bucket]
        if missing:
            failures[bucket]["missing_outcome"] += missing
    rates = {bucket: successes[bucket] / count for bucket, count in counts.items()}
    return (
        rates,
        dict(sorted(counts.items())),
        {
            bucket: dict(sorted(stage_counts.items()))
            for bucket, stage_counts in sorted(failures.items())
        },
    )


def _population_stats(
    rows: list[dict[str, object]],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return mean, worst, population variance and retention over OOD domains."""
    non_nominal = [
        float(row["success_rate"])
        for row in rows
        if row["domain_kind"] != "nominal"
        and row["success_rate"] is not None
        and math.isfinite(float(row["success_rate"]))
    ]
    nominal = [
        float(row["success_rate"])
        for row in rows
        if row["domain_kind"] == "nominal"
        and row["success_rate"] is not None
        and math.isfinite(float(row["success_rate"]))
    ]
    if not non_nominal:
        return None, None, None, None
    mean = sum(non_nominal) / len(non_nominal)
    worst = min(non_nominal)
    variance = sum((value - mean) ** 2 for value in non_nominal) / len(non_nominal)
    nominal_rate = sum(nominal) / len(nominal) if nominal else None
    retention = None if nominal_rate in (None, 0.0) else worst / nominal_rate
    return mean, worst, variance, retention


def _population_row(
    *,
    track: str,
    algorithm: str,
    algorithm_role: str,
    domain_id: str,
    domain_version: str,
    domain_kind: str,
    cases: list[BenchmarkCase],
    total_cases: int,
    bypassed_cases: int,
    records_by_case: dict[str, list[TrialRecord]],
    expected_repeats_by_case: dict[str, int],
) -> dict[str, object]:
    """Aggregate one planner- or effective-algorithm domain population."""
    population_records = [
        record for case in cases for record in records_by_case.get(case.case_id, [])
    ]
    unsupported = _unsupported_case_ids(population_records)
    applicable_cases = [case for case in cases if case.case_id not in unsupported]
    expected_rows = sum(
        case.batch_size * expected_repeats_by_case.get(case.case_id, 0)
        for case in cases
    )
    observed_rows = sum(
        len(record.outcomes)
        for record in population_records
        if record.phase is TrialPhase.MEASURED
    )
    case_rates = [
        _case_success(records_by_case.get(case.case_id, []), case)[0]
        for case in applicable_cases
    ]
    success_rate = sum(case_rates) / len(case_rates) if case_rates else None
    bucket_rates, bucket_counts, bucket_failures = _spatial_bucket_statistics(
        applicable_cases,
        records_by_case,
        expected_repeats_by_case,
    )
    return {
        "track": track,
        "algorithm": algorithm,
        "algorithm_role": algorithm_role,
        "domain_id": domain_id,
        "domain_version": domain_version,
        "domain_kind": domain_kind,
        "cases": total_cases,
        "attributed_cases": len(cases),
        "applicable_cases": len(applicable_cases),
        "bypassed_cases": bypassed_cases,
        "unsupported_cases": len(unsupported),
        "coverage_rate": min(1.0, observed_rows / max(expected_rows, 1)),
        "success_rate": success_rate,
        "spatial_bucket_success_rates": bucket_rates,
        "spatial_bucket_counts": bucket_counts,
        "spatial_bucket_failure_stages": bucket_failures,
        "nominal_success_rate": None,
        "domain_mean_success_rate": None,
        "domain_worst_success_rate": None,
        "domain_success_variance": None,
        "domain_retention": None,
        "domain_mean": None,
        "domain_worst": None,
        "domain_variance": None,
        "generalization_eligible": False,
    }


def aggregate_generalization(
    records: Iterable[TrialRecord],
    metadata: Iterable[PlannerMetadata],
    cases: Iterable[BenchmarkCase],
    measured_trials: int,
) -> list[dict[str, object]]:
    """Aggregate versioned nominal, robustness, and held-out populations.

    Unsupported capability cases lower coverage and are excluded from the
    success denominator. Cases that bypass the selected planner are excluded
    from planner attribution and reported under their effective motion
    algorithm. Runtime errors and ordinary planner failures remain applicable
    cases and therefore count as failures. The returned rows are sufficient
    to recompute the domain summary from raw trial records.

    Args:
        records: Raw lifecycle and measured trial records.
        metadata: Planner metadata for every algorithm in the run.
        cases: Planner-independent frozen case manifest.
        measured_trials: Number of measured repeats expected per case.

    Returns:
        One row per attributed planner or effective algorithm and evaluation
        domain, including generalization summary statistics shared by that
        algorithm's non-nominal domains.
    """
    if measured_trials < 1:
        raise ValueError("measured_trials must be positive.")
    record_list = list(records)
    case_list = [case for case in cases if case.domain is not None]
    if not case_list:
        return []
    metadata_list = list(metadata)
    by_algorithm_case: dict[tuple[str, str], list[TrialRecord]] = defaultdict(list)
    by_effective_algorithm_case: dict[tuple[str, str], list[TrialRecord]] = defaultdict(
        list
    )
    effective_case_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in record_list:
        by_algorithm_case[(record.algorithm_id, record.case_id)].append(record)
        if not record.uses_selected_planner_backend:
            by_effective_algorithm_case[
                (record.effective_motion_algorithm, record.case_id)
            ].append(record)
            effective_case_sources[
                (record.effective_motion_algorithm, record.case_id)
            ].add(record.algorithm_id)

    cases_by_population: dict[tuple[str, str, str, str], list[BenchmarkCase]] = (
        defaultdict(list)
    )
    for case in case_list:
        assert case.domain is not None
        domain_id, domain_version, domain_kind = _domain_key(case.domain)
        cases_by_population[
            (_track_group(case), domain_id, domain_version, domain_kind)
        ].append(case)

    rows: list[dict[str, object]] = []
    for planner in metadata_list:
        for population_key, population_cases in sorted(cases_by_population.items()):
            track, domain_id, domain_version, domain_kind = population_key
            bypassed = {
                case.case_id
                for case in population_cases
                if any(
                    not record.uses_selected_planner_backend
                    for record in by_algorithm_case[
                        (planner.algorithm_id, case.case_id)
                    ]
                )
            }
            attributed_cases = [
                case for case in population_cases if case.case_id not in bypassed
            ]
            rows.append(
                _population_row(
                    track=track,
                    algorithm=planner.algorithm_id,
                    algorithm_role=planner.algorithm_role.value,
                    domain_id=domain_id,
                    domain_version=domain_version,
                    domain_kind=domain_kind,
                    cases=attributed_cases,
                    total_cases=len(population_cases),
                    bypassed_cases=len(bypassed),
                    records_by_case={
                        case.case_id: [
                            record
                            for record in by_algorithm_case[
                                (planner.algorithm_id, case.case_id)
                            ]
                            if record.uses_selected_planner_backend
                        ]
                        for case in attributed_cases
                    },
                    expected_repeats_by_case={
                        case.case_id: measured_trials for case in attributed_cases
                    },
                )
            )

    effective_algorithms = sorted(
        {algorithm for algorithm, _ in by_effective_algorithm_case}
    )
    for algorithm in effective_algorithms:
        for population_key, population_cases in sorted(cases_by_population.items()):
            track, domain_id, domain_version, domain_kind = population_key
            attributed_cases = [
                case
                for case in population_cases
                if (algorithm, case.case_id) in by_effective_algorithm_case
            ]
            if not attributed_cases:
                continue
            rows.append(
                _population_row(
                    track=track,
                    algorithm=algorithm,
                    algorithm_role=_EFFECTIVE_MOTION_ALGORITHM_ROLE,
                    domain_id=domain_id,
                    domain_version=domain_version,
                    domain_kind=domain_kind,
                    cases=attributed_cases,
                    total_cases=len(attributed_cases),
                    bypassed_cases=0,
                    records_by_case={
                        case.case_id: by_effective_algorithm_case[
                            (algorithm, case.case_id)
                        ]
                        for case in attributed_cases
                    },
                    expected_repeats_by_case={
                        case.case_id: measured_trials
                        * len(effective_case_sources[(algorithm, case.case_id)])
                        for case in attributed_cases
                    },
                )
            )

    by_summary_group: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        by_summary_group[
            (
                str(row["track"]),
                str(row["algorithm"]),
                str(row["algorithm_role"]),
            )
        ].append(row)
    for population_rows in by_summary_group.values():
        mean, worst, variance, retention = _population_stats(population_rows)
        nominal_rates = [
            float(row["success_rate"])
            for row in population_rows
            if row["domain_kind"] == "nominal" and row["success_rate"] is not None
        ]
        nominal_rate = (
            sum(nominal_rates) / len(nominal_rates) if nominal_rates else None
        )
        has_non_nominal = any(
            row["domain_kind"] != "nominal" for row in population_rows
        )
        eligible = has_non_nominal and all(
            float(row["coverage_rate"]) >= 1.0 - 1.0e-12
            for row in population_rows
            if row["domain_kind"] != "nominal"
        )
        for row in population_rows:
            row["nominal_success_rate"] = nominal_rate
            row["domain_mean_success_rate"] = mean
            row["domain_worst_success_rate"] = worst
            row["domain_success_variance"] = variance
            row["domain_retention"] = retention
            row["domain_mean"] = mean
            row["domain_worst"] = worst
            row["domain_variance"] = variance
            row["generalization_eligible"] = eligible
    return rows
