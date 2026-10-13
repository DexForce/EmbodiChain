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
from collections.abc import Iterable, Mapping

from .contracts import EvaluationDomain
from .models import BenchmarkCase, PlannerMetadata, TrialPhase, TrialRecord

__all__ = ["aggregate_generalization", "spatial_breakdown"]

_SPLIT_KINDS = ("robustness", "held_out")


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


def _population_stats(
    rows: list[dict[str, object]],
    kinds: tuple[str, ...] | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return mean, worst, population variance and retention over OOD domains.

    Args:
        rows: Domain rows of one track and planner.
        kinds: Domain kinds to summarize; ``None`` pools every non-nominal kind.
    """
    non_nominal = [
        float(row["success_rate"])
        for row in rows
        if row["domain_kind"] != "nominal"
        and (kinds is None or row["domain_kind"] in kinds)
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


def _spatial_feasibility(
    cases: Iterable[BenchmarkCase],
) -> tuple[int | None, int | None, float | None]:
    """Return draws, infeasible draws and their rate for a sampled population.

    Every case of one template carries the same template summary, so each
    template is counted once.  Populations without spatial sampling return
    ``None`` values.
    """
    templates: dict[str, Mapping[str, object]] = {}
    for case in cases:
        summary = case.case_parameters.get("spatial_feasibility")
        if isinstance(summary, Mapping):
            templates[str(summary["template"])] = summary
    if not templates:
        return None, None, None
    draws = sum(int(summary["draws"]) for summary in templates.values())
    infeasible = sum(int(summary["infeasible"]) for summary in templates.values())
    return draws, infeasible, infeasible / draws if draws else None


def aggregate_generalization(
    records: Iterable[TrialRecord],
    metadata: Iterable[PlannerMetadata],
    cases: Iterable[BenchmarkCase],
    measured_trials: int,
) -> list[dict[str, object]]:
    """Aggregate versioned nominal, robustness, and held-out populations.

    Unsupported capability cases lower coverage and are excluded from the
    success denominator.  Runtime errors and ordinary planner failures remain
    applicable cases and therefore count as failures.  The returned rows are
    sufficient to recompute the domain summary from raw trial records.

    Args:
        records: Raw lifecycle and measured trial records.
        metadata: Planner metadata for every algorithm in the run.
        cases: Planner-independent frozen case manifest.
        measured_trials: Number of measured repeats expected per case.

    Returns:
        One row per planner and evaluation domain, including generalization
        summary statistics shared by that planner's non-nominal domains.
    """
    if measured_trials < 1:
        raise ValueError("measured_trials must be positive.")
    record_list = list(records)
    case_list = [case for case in cases if case.domain is not None]
    if not case_list:
        return []
    metadata_list = list(metadata)
    by_algorithm_case: dict[tuple[str, str], list[TrialRecord]] = defaultdict(list)
    for record in record_list:
        by_algorithm_case[(record.algorithm_id, record.case_id)].append(record)

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
            population_records = [
                record
                for case in population_cases
                for record in by_algorithm_case[(planner.algorithm_id, case.case_id)]
            ]
            unsupported = _unsupported_case_ids(population_records)
            applicable_cases = [
                case for case in population_cases if case.case_id not in unsupported
            ]
            expected_rows = (
                sum(case.batch_size for case in population_cases) * measured_trials
            )
            observed_rows = sum(
                len(record.outcomes)
                for record in population_records
                if record.phase is TrialPhase.MEASURED
            )
            case_rates: list[float] = []
            for case in applicable_cases:
                rate, _ = _case_success(
                    by_algorithm_case[(planner.algorithm_id, case.case_id)], case
                )
                case_rates.append(rate)
            success_rate = sum(case_rates) / len(case_rates) if case_rates else None
            draws, infeasible, infeasible_rate = _spatial_feasibility(population_cases)
            rows.append(
                {
                    "track": track,
                    "algorithm": planner.algorithm_id,
                    "algorithm_role": planner.algorithm_role.value,
                    "domain_id": domain_id,
                    "domain_version": domain_version,
                    "domain_kind": domain_kind,
                    "cases": len(population_cases),
                    "applicable_cases": len(applicable_cases),
                    "unsupported_cases": len(unsupported),
                    "coverage_rate": min(1.0, observed_rows / max(expected_rows, 1)),
                    "success_rate": success_rate,
                    "nominal_success_rate": None,
                    "domain_mean_success_rate": None,
                    "domain_worst_success_rate": None,
                    "domain_success_variance": None,
                    "domain_retention": None,
                    "domain_mean": None,
                    "domain_worst": None,
                    "domain_variance": None,
                    "generalization_eligible": False,
                    **{
                        f"{kind}_{field}": None
                        for kind in _SPLIT_KINDS
                        for field in (
                            "mean_success_rate",
                            "worst_success_rate",
                            "retention",
                        )
                    },
                    "spatial_draws": draws,
                    "spatial_infeasible_draws": infeasible,
                    "spatial_infeasible_rate": infeasible_rate,
                }
            )

    by_summary_group: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_summary_group[(str(row["track"]), str(row["algorithm"]))].append(row)
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
        # Robustness perturbs the nominal domain while held-out domains are new
        # populations; pooling them would hide which one a planner fails.
        for kind in _SPLIT_KINDS:
            kind_mean, kind_worst, _, kind_retention = _population_stats(
                population_rows, (kind,)
            )
            for row in population_rows:
                row[f"{kind}_mean_success_rate"] = kind_mean
                row[f"{kind}_worst_success_rate"] = kind_worst
                row[f"{kind}_retention"] = kind_retention
    return rows


def spatial_breakdown(
    records: Iterable[TrialRecord],
    metadata: Iterable[PlannerMetadata],
    cases: Iterable[BenchmarkCase],
    *,
    radius_bin_m: float = 0.05,
) -> list[dict[str, object]]:
    """Slice sampled domains by the base-frame radius of each position.

    Reachability changes mostly with distance from the arm base, so success
    is reported per radius band of every sampled ``cylinder`` parameter.

    Args:
        records: Raw lifecycle and measured trial records.
        metadata: Planner metadata for every algorithm in the run.
        cases: Frozen case manifest; only spatially sampled cases are used.
        radius_bin_m: Width of each radius band in meters.

    Returns:
        One row per planner, domain, parameter and radius band.
    """
    if not math.isfinite(radius_bin_m) or radius_bin_m <= 0.0:
        raise ValueError("radius_bin_m must be positive.")
    by_algorithm_case: dict[tuple[str, str], list[TrialRecord]] = defaultdict(list)
    for record in records:
        by_algorithm_case[(record.algorithm_id, record.case_id)].append(record)
    bands: dict[tuple[str, str, str, int], list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        sample = case.case_parameters.get("spatial_sample")
        if case.domain is None or not isinstance(sample, Mapping):
            continue
        for name, values in sample["parameters"].items():
            radius = values.get("radius_m")
            if radius is None:
                continue
            band = int(math.floor(float(radius) / radius_bin_m))
            bands[(_track_group(case), case.domain.id, name, band)].append(case)
    rows: list[dict[str, object]] = []
    for planner in metadata:
        for (track, domain_id, name, band), band_cases in sorted(bands.items()):
            unsupported = _unsupported_case_ids(
                record
                for case in band_cases
                for record in by_algorithm_case[(planner.algorithm_id, case.case_id)]
            )
            rates = [
                _case_success(
                    by_algorithm_case[(planner.algorithm_id, case.case_id)], case
                )[0]
                for case in band_cases
                if case.case_id not in unsupported
            ]
            rows.append(
                {
                    "track": track,
                    "algorithm": planner.algorithm_id,
                    "domain_id": domain_id,
                    "parameter": name,
                    "radius_low_m": band * radius_bin_m,
                    "radius_high_m": (band + 1) * radius_bin_m,
                    "cases": len(band_cases),
                    "success_rate": sum(rates) / len(rates) if rates else None,
                }
            )
    return rows
