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

"""Backend-neutral predicates, evaluation graphs and bounded temporal state."""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "EvaluationStatus",
    "PredicateSpec",
    "BooleanRule",
    "TemporalRule",
    "MilestoneRule",
    "EvaluationSpec",
    "PredicateMeasurement",
    "PredicateDefinition",
    "PredicateRegistry",
    "CompiledEvaluation",
    "RuleResult",
    "EvaluationSnapshot",
    "RuleSession",
    "builtin_predicates",
    "evaluate_recorded_rules",
]


class EvaluationStatus(IntEnum):
    """Codes used in the per-row predicate and window status tensors."""

    NOT_RUN = 0
    UNKNOWN = 1
    UNAVAILABLE = 2
    FAILED = 3
    SATISFIED = 4

    @property
    def reason(self) -> str:
        """Describe why the result can or cannot establish satisfaction."""
        return {
            EvaluationStatus.NOT_RUN: "not_evaluated",
            EvaluationStatus.UNKNOWN: "window_pending",
            EvaluationStatus.UNAVAILABLE: "insufficient_qualified_evidence",
            EvaluationStatus.FAILED: "criteria_not_met",
            EvaluationStatus.SATISFIED: "criteria_met",
        }[self]


from ._graph import (
    BooleanRule,
    CompiledEvaluation,
    EvaluationSpec,
    MilestoneRule,
    PredicateDefinition,
    PredicateMeasurement,
    PredicateRegistry,
    PredicateSpec,
    RuleResult,
    TemporalRule,
)
from ._predicates import builtin_predicates
from ._session import EvaluationSnapshot, RuleSession
from ._recording import evaluate_recorded_rules
