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

"""Shared applicability rules for benchmark success denominators."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import BenchmarkCase, PlannerMetadata, TrialPhase, TrialRecord

__all__: list[str] = []


def _report_identity(record: TrialRecord) -> tuple[str, bool]:
    """Keep configured and scenario-owned algorithms distinct even if IDs match."""
    return record.effective_motion_algorithm, not record.uses_selected_planner_backend


class _PopulationIndex:
    """Index attribution facts without choosing a view's grouping or denominator."""

    def __init__(self, records: Iterable[TrialRecord]) -> None:
        self.by_algorithm_case: dict[tuple[str, str], list[TrialRecord]] = defaultdict(
            list
        )
        self.by_effective_algorithm_case: dict[tuple[str, str], list[TrialRecord]] = (
            defaultdict(list)
        )
        self.by_identity_track: dict[
            tuple[tuple[str, bool], str], list[TrialRecord]
        ] = defaultdict(list)
        self.effective_case_sources: dict[tuple[str, str, str], set[str]] = defaultdict(
            set
        )
        self.bypassed_by_algorithm_track: dict[tuple[str, str], set[str]] = defaultdict(
            set
        )
        for record in records:
            identity = _report_identity(record)
            self.by_identity_track[(identity, record.track)].append(record)
            self.by_algorithm_case[(record.algorithm_id, record.case_id)].append(record)
            if identity[1]:
                self.by_effective_algorithm_case[(identity[0], record.case_id)].append(
                    record
                )
                self.effective_case_sources[
                    (identity[0], record.track, record.case_id)
                ].add(record.algorithm_id)
                self.bypassed_by_algorithm_track[
                    (record.algorithm_id, record.track)
                ].add(record.case_id)

    @property
    def effective_algorithms(self) -> list[str]:
        return sorted({algorithm for algorithm, _ in self.by_effective_algorithm_case})

    def identities(self, metadata: Iterable[PlannerMetadata]) -> list[tuple[str, bool]]:
        return [(info.algorithm_id, False) for info in metadata] + [
            (algorithm, True) for algorithm in self.effective_algorithms
        ]

    def attributed_cases(
        self, cases: Iterable[BenchmarkCase], algorithm: str, *, effective: bool
    ) -> list[BenchmarkCase]:
        if effective:
            return [
                case
                for case in cases
                if (algorithm, case.track, case.case_id) in self.effective_case_sources
            ]
        return [
            case
            for case in cases
            if case.case_id
            not in self.bypassed_by_algorithm_track.get((algorithm, case.track), set())
        ]


def _case_success(
    records: Iterable[TrialRecord],
    case: BenchmarkCase,
    measured_trials: int,
    attribute: str | None = None,
) -> tuple[float, int]:
    """Return measured success with missing mandatory rows counted as failures."""
    outcomes = [
        outcome
        for record in records
        if record.phase is TrialPhase.MEASURED
        for outcome in record.outcomes
    ]
    if not outcomes:
        return 0.0, 0
    attribute = attribute or case.primary_success
    return (
        sum(bool(getattr(outcome, attribute)) for outcome in outcomes)
        / max(len(outcomes), case.batch_size * measured_trials),
        len(outcomes),
    )


def _case_rates(
    records: Iterable[TrialRecord],
    cases: Iterable[BenchmarkCase],
    attribute: str | None = None,
    *,
    measured_trials: int = 1,
    excluded_case_ids: frozenset[str] = frozenset(),
) -> list[tuple[BenchmarkCase, float]]:
    """Calculate the common case weights for rates, rankings and seed intervals."""
    by_case: dict[str, list[TrialRecord]] = defaultdict(list)
    for record in records:
        by_case[record.case_id].append(record)
    return [
        (
            case,
            _case_success(by_case[case.case_id], case, measured_trials, attribute)[0],
        )
        for case in cases
        if case.case_id not in excluded_case_ids
    ]


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
