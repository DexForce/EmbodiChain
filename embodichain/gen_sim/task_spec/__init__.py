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

"""GenSim-owned, simulator-independent TaskSpec protocol and cache.

The core object is :class:`TaskSpec`. Legacy artifact exports are resolved
from :mod:`.compat` only when requested; normal generation and cache access
do not load them. Runtime evidence belongs to the execution integrations.
"""

from __future__ import annotations

from typing import Any

from .canonicalization import (
    CANONICALIZATION_VERSION,
    canonical_hash,
    canonical_json,
    canonicalize,
    json_snapshot,
)
from .contracts import RoleSpec
from .expressions import (
    EVALUATION_STATUSES,
    KNOWN_PREDICATES,
    PREDICATE_OPERATORS,
    EvaluationStatus,
    Predicate,
    PredicateEvaluation,
    PredicateSpec,
    TemporalCondition,
    TemporalSpec,
    evaluate_predicate,
    evaluate_temporal_condition,
    lookup_observation,
)
from .registry import TaskSpecCache, TaskSpecCacheError, TaskSpecRegistry
from .spec import (
    MILESTONE_SCHEMA,
    TASK_SPEC_SCHEMA,
    MilestoneSpec,
    TaskRequirement,
    TaskSpec,
    TaskTemplate,
)
from .validation import (
    EvaluationReport,
    TaskSpecValidationError,
    evaluate_task_spec,
    evaluate_task_template,
    validate_task_spec,
    validate_task_template,
)

__all__ = [
    "CANONICALIZATION_VERSION",
    "EVALUATION_STATUSES",
    "KNOWN_PREDICATES",
    "MILESTONE_SCHEMA",
    "PREDICATE_OPERATORS",
    "TASK_SPEC_SCHEMA",
    "EvaluationReport",
    "EvaluationStatus",
    "MilestoneSpec",
    "Predicate",
    "PredicateEvaluation",
    "PredicateSpec",
    "RoleSpec",
    "TaskRequirement",
    "TaskSpec",
    "TaskSpecCache",
    "TaskSpecCacheError",
    "TaskSpecRegistry",
    "TaskSpecValidationError",
    "TaskTemplate",
    "TemporalCondition",
    "TemporalSpec",
    "canonical_hash",
    "canonical_json",
    "canonicalize",
    "evaluate_predicate",
    "evaluate_task_spec",
    "evaluate_task_template",
    "evaluate_temporal_condition",
    "json_snapshot",
    "lookup_observation",
    "validate_task_spec",
    "validate_task_template",
]
