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

"""Eligibility checks; a completed run is not automatically comparable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import statistics
from typing import Any

__all__ = ["ComparisonResult", "compare_metric", "comparison_reasons"]


@dataclass(frozen=True)
class ComparisonResult:
    """Paired comparison result with explicit eligibility and source count."""

    eligible: bool
    reasons: tuple[str, ...]
    baseline: str
    candidate: str
    metric: str
    pairs: int
    ratio: float | None
    delta: float | None


def _field(row: Mapping[str, Any], path: str) -> Any:
    value = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def comparison_reasons(
    rows: Sequence[Mapping[str, Any]], *, invariant_paths: Sequence[str]
) -> tuple[str, ...]:
    """Return missing/mismatched conditions, leaving raw results available.

    Domains declare invariants and supply independent quality qualification.
    This function never promotes task completion or image non-emptiness into
    quality qualification, and never invents a speedup from incompatible data.
    """
    complete = [r for r in rows if r.get("status") == "completed"]
    reasons = []
    if len({r.get("backend") for r in complete}) < 2:
        reasons.append("need_at_least_two_completed_backends")
    if not complete or any(r.get("quality_status") != "qualified" for r in complete):
        reasons.append("quality_status")
    for path in invariant_paths:
        values = [_field(row, path) for row in complete]
        if (
            not values
            or any(v is None for v in values)
            or any(v != values[0] for v in values)
        ):
            reasons.append(path)
    return tuple(reasons)


def compare_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    baseline: str,
    candidate: str,
    invariant_paths: Sequence[str],
) -> ComparisonResult:
    """Compare one metric over matched case/repeat rows.

    ``ratio`` is baseline divided by candidate, so a lower candidate latency
    produces a value above one. It is returned only when all declared
    invariants and independent quality checks pass.
    """
    complete = [
        row
        for row in rows
        if row.get("status") == "completed"
        and row.get("backend") in {baseline, candidate}
    ]
    reasons = list(comparison_reasons(complete, invariant_paths=invariant_paths))
    backends = {row.get("backend") for row in complete}
    if baseline not in backends or candidate not in backends:
        reasons.append("requested_backends")
    if len({row.get("case_id") for row in complete}) > 1:
        reasons.append("multiple_cases")
    pairs: list[tuple[float, float]] = []
    left: dict[tuple[object, object], Mapping[str, Any]] = {}
    right: dict[tuple[object, object], Mapping[str, Any]] = {}
    for row in complete:
        key = (row.get("case_id"), row.get("repeat"))
        target = left if row.get("backend") == baseline else right
        if key in target:
            reasons.append("duplicate_pair_identity")
        target[key] = row
    if set(left) != set(right):
        reasons.append("unmatched_pairs")
    for key in sorted(set(left) & set(right), key=str):
        left_value = left[key].get("metrics", {}).get(metric)
        right_value = right[key].get("metrics", {}).get(metric)
        if (
            type(left_value) in (int, float)
            and type(right_value) in (int, float)
            and math.isfinite(left_value)
            and math.isfinite(right_value)
        ):
            pairs.append((float(left_value), float(right_value)))
    if not pairs:
        reasons.append("no_matched_metric_values")
    ratios = [
        left_value / right_value
        for left_value, right_value in pairs
        if right_value != 0
    ]
    if pairs and len(ratios) != len(pairs):
        reasons.append("candidate_metric_zero")
    eligible = not reasons
    return ComparisonResult(
        eligible=eligible,
        reasons=tuple(dict.fromkeys(reasons)),
        baseline=baseline,
        candidate=candidate,
        metric=metric,
        pairs=len(pairs),
        ratio=statistics.median(ratios) if eligible else None,
        delta=(
            statistics.median(
                right_value - left_value for left_value, right_value in pairs
            )
            if eligible
            else None
        ),
    )
