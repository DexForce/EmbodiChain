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

"""Pure protocol objects for generated, cacheable Task Specifications.

``embodichain.task_spec`` intentionally has no simulator dependency.  Task
Engine produces :class:`TaskTemplate`; scene/runtime integrations append
``SceneInstance``, ``ActionWitness``, and ``ValidationCertificate`` evidence;
trajectory expansion records lineage in ``ExpansionManifest``.
"""

from __future__ import annotations

from .canonicalization import (
    CANONICALIZATION_VERSION,
    canonical_hash,
    canonical_json,
    canonicalize,
    json_snapshot,
)
from .contracts import (
    ACTION_WITNESS_SCHEMA,
    EXPANSION_MANIFEST_SCHEMA,
    SCENE_INSTANCE_SCHEMA,
    TASK_TEMPLATE_SCHEMA,
    VALIDATION_CERTIFICATE_SCHEMA,
    ActionWitness,
    ExpansionManifest,
    RequirementSpec,
    RoleSpec,
    SceneInstance,
    TaskSpec,
    TaskTemplate,
    ValidationCertificate,
)
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
from .validation import (
    EvaluationReport,
    TaskSpecValidationError,
    build_validation_certificate,
    evaluate_task_template,
    validate_action_witness,
    validate_expansion_manifest,
    validate_scene_instance,
    validate_task_template,
    validate_validation_certificate,
)

__all__ = [
    "ACTION_WITNESS_SCHEMA",
    "CANONICALIZATION_VERSION",
    "EVALUATION_STATUSES",
    "EXPANSION_MANIFEST_SCHEMA",
    "KNOWN_PREDICATES",
    "PREDICATE_OPERATORS",
    "SCENE_INSTANCE_SCHEMA",
    "TASK_TEMPLATE_SCHEMA",
    "VALIDATION_CERTIFICATE_SCHEMA",
    "ActionWitness",
    "EvaluationReport",
    "EvaluationStatus",
    "ExpansionManifest",
    "Predicate",
    "PredicateEvaluation",
    "PredicateSpec",
    "RequirementSpec",
    "RoleSpec",
    "SceneInstance",
    "TaskSpec",
    "TaskSpecCache",
    "TaskSpecCacheError",
    "TaskSpecRegistry",
    "TaskSpecValidationError",
    "TaskTemplate",
    "TemporalCondition",
    "TemporalSpec",
    "ValidationCertificate",
    "build_validation_certificate",
    "canonical_hash",
    "canonical_json",
    "canonicalize",
    "evaluate_predicate",
    "evaluate_task_template",
    "evaluate_temporal_condition",
    "json_snapshot",
    "lookup_observation",
    "validate_action_witness",
    "validate_expansion_manifest",
    "validate_scene_instance",
    "validate_task_template",
    "validate_validation_certificate",
]
