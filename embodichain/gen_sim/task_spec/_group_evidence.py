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

"""Shared final aggregation of already verified finite-member facts."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .expressions import PredicateEvaluation

__all__: list[str] = []


def aggregate_member_outcomes(
    predicate: Any,
    outcomes: list[Any],
    *,
    size: int,
    refs: tuple[str, ...],
    reason: str | None,
) -> tuple[PredicateEvaluation, ...]:
    """Preserve unavailable members and complete measured negatives per row."""
    from .expressions import PredicateEvaluation, EvaluationStatus

    if not outcomes and reason is None:
        reason = "No complete verified member evidence was provided."
    result = []
    for row in range(size):
        missing = (
            None
            if reason
            else next(
                (
                    values[row]
                    for values in outcomes
                    if values[row].status is EvaluationStatus.UNAVAILABLE
                ),
                None,
            )
        )
        unavailable = reason is not None or missing is not None
        accepted = not unavailable and all(
            values[row].status is EvaluationStatus.SATISFIED for values in outcomes
        )
        member_refs = tuple(
            ref for values in outcomes for ref in values[row].evidence_refs
        )
        result.append(
            PredicateEvaluation(
                predicate.id,
                predicate.name,
                (
                    EvaluationStatus.UNAVAILABLE
                    if unavailable
                    else (
                        EvaluationStatus.SATISFIED
                        if accepted
                        else EvaluationStatus.FAILED
                    )
                ),
                observed=None if unavailable else accepted,
                expected=predicate.expected,
                reason=reason
                or (
                    missing.reason
                    if missing
                    else "Every finite-count member was checked against its original state criterion."
                ),
                evidence_refs=tuple(dict.fromkeys((*refs, *member_refs))),
            )
        )
    return tuple(result)
