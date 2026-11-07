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

"""Normative receiver hold over normalized physical facts."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from .spec import TaskSpec
from .expressions import PredicateEvaluation
from .manipulation_evidence import (
    ManipulationEvidenceBinding,
    evaluate_manipulation_evidence,
)

__all__ = ["evaluate_handover_evidence"]


def evaluate_handover_evidence(
    spec: TaskSpec | Mapping[str, Any],
    binding: ManipulationEvidenceBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> tuple[PredicateEvaluation, ...]:
    """Compare normalized facts with sealed task criteria.

    Args:
        spec: Original normative specification.
        binding: Provider-independent criteria and observation scope.
        evidence: Normalized facts with per-row validity and provenance.
        program: Original program, used only for its content identity.

    Returns:
        Per-environment normative predicate results.
    """
    selected = ManipulationEvidenceBinding.from_dict(binding)
    if selected.kind not in {
        "receiver_hold",
        "receiver_orientation",
        "handover_complete",
    }:
        raise ValueError("Evidence kind differs from this normative evaluator.")
    return evaluate_manipulation_evidence(spec, selected, evidence, program=program)
