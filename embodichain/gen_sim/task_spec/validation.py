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

"""Validation and provider-free evaluation for TaskSpec contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .canonicalization import json_snapshot
from .contracts import (
    ActionWitness,
    ExpansionManifest,
    SceneInstance,
    ValidationCertificate,
)
from .expressions import (
    EvaluationStatus,
    PredicateEvaluation,
    evaluate_predicate,
    evaluate_temporal_condition,
)
from .spec import TaskSpec

__all__ = [
    "EvaluationReport",
    "TaskSpecValidationError",
    "build_validation_certificate",
    "evaluate_task_spec",
    "evaluate_task_template",
    "validate_action_witness",
    "validate_expansion_manifest",
    "validate_scene_instance",
    "validate_task_spec",
    "validate_task_template",
    "validate_validation_certificate",
]


class TaskSpecValidationError(ValueError):
    """Raised when a TaskSpec violates its versioned protocol contract."""


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Complete evaluation of init, goal, invariants, and temporal checks."""

    status: EvaluationStatus
    init: tuple[PredicateEvaluation, ...]
    goal: tuple[PredicateEvaluation, ...]
    invariants: tuple[PredicateEvaluation, ...]
    temporal: tuple[PredicateEvaluation, ...]

    @property
    def certificate_ready(self) -> bool:
        """Whether every required check is satisfied."""
        return self.status is EvaluationStatus.SATISFIED

    def to_dict(self) -> dict[str, Any]:
        """Return an evidence-friendly report mapping."""
        return {
            "status": self.status.value,
            "certificate_ready": self.certificate_ready,
            "init": [item.to_dict() for item in self.init],
            "goal": [item.to_dict() for item in self.goal],
            "invariants": [item.to_dict() for item in self.invariants],
            "temporal": [item.to_dict() for item in self.temporal],
        }


def validate_task_spec(
    value: TaskSpec | Mapping[str, Any],
    *,
    require_observations: bool = False,
) -> dict[str, Any]:
    """Validate and normalize one compact TaskSpec.

    Args:
        value: Dataclass or JSON mapping to validate.
        require_observations: Require every goal, invariant, and temporal
            predicate to declare a runtime observation key.

    Returns:
        A detached canonical mapping including the verified semantic hash.

    Raises:
        TaskSpecValidationError: If a schema, predicate, or observation
            declaration is invalid.
    """
    try:
        template = TaskSpec.from_dict(value)
        if not template.roles:
            raise ValueError("TaskSpec.roles must contain at least one role.")
        if not template.goal:
            raise ValueError("TaskSpec.goal must contain at least one predicate.")
        if require_observations:
            missing = [
                item.predicate_id or item.name
                for item in (*template.goal, *template.invariants)
                if item.observation is None
            ] + [
                item.condition_id or item.predicate.name
                for item in template.temporal
                if item.predicate.observation is None
            ]
            if missing:
                raise ValueError(
                    "TaskSpec predicates require observation keys: "
                    + ", ".join(missing)
                )
        return template.to_dict()
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TaskSpecValidationError):
            raise
        raise TaskSpecValidationError(str(exc)) from exc


def validate_scene_instance(value: SceneInstance | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one grounded SceneInstance."""
    try:
        return SceneInstance.from_dict(value).to_dict()
    except (TypeError, ValueError) as exc:
        raise TaskSpecValidationError(str(exc)) from exc


def validate_action_witness(value: ActionWitness | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one ActionWitness."""
    try:
        return ActionWitness.from_dict(value).to_dict()
    except (TypeError, ValueError) as exc:
        raise TaskSpecValidationError(str(exc)) from exc


def validate_expansion_manifest(
    value: ExpansionManifest | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one ExpansionManifest."""
    try:
        return ExpansionManifest.from_dict(value).to_dict()
    except (TypeError, ValueError) as exc:
        raise TaskSpecValidationError(str(exc)) from exc


def validate_validation_certificate(
    value: ValidationCertificate | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one ValidationCertificate."""
    try:
        return ValidationCertificate.from_dict(value).to_dict()
    except (TypeError, ValueError) as exc:
        raise TaskSpecValidationError(str(exc)) from exc


def evaluate_task_template(
    template: TaskSpec | Mapping[str, Any],
    observations: Mapping[str, Any],
    *,
    trajectory: Sequence[Mapping[str, Any]] | None = None,
) -> EvaluationReport:
    """Evaluate a template against explicit runtime observations.

    An adapter may provide section-specific mappings under ``init``, ``goal``,
    and ``invariants``.  When absent, the supplied mapping is used for all
    snapshot predicates.  Temporal conditions use ``trajectory`` or an
    explicit ``observations['trajectory']`` sequence.  Missing measurements are
    reported as ``unavailable`` and never treated as success.
    """
    normalized = TaskSpec.from_dict(template)
    validate_task_spec(normalized)
    if not isinstance(observations, Mapping):
        raise TypeError("observations must be a mapping.")
    root = json_snapshot(dict(observations))
    if not isinstance(root, dict):  # pragma: no cover - guarded by snapshot
        raise TypeError("observations must be a mapping.")
    init_observations = _section(root, "init")
    goal_observations = _section(root, "goal")
    invariant_observations = _section(root, "invariants")
    init = tuple(
        evaluate_predicate(item, init_observations, predicate_id=item.predicate_id)
        for item in normalized.init
    )
    goal = tuple(
        evaluate_predicate(item, goal_observations, predicate_id=item.predicate_id)
        for item in normalized.goal
    )
    invariants = tuple(
        evaluate_predicate(item, invariant_observations, predicate_id=item.predicate_id)
        for item in normalized.invariants
    )
    temporal_source: Any = trajectory
    if temporal_source is None:
        temporal_source = root.get("trajectory")
    if temporal_source is None:
        temporal = tuple(
            PredicateEvaluation(
                predicate_id=item.condition_id or item.predicate.name,
                predicate=item.predicate.name,
                status=EvaluationStatus.UNAVAILABLE,
                observed=None,
                expected=item.predicate.expected,
                reason="no trajectory was supplied for temporal condition",
            )
            for item in normalized.temporal
        )
    else:
        temporal = tuple(
            evaluate_temporal_condition(item, temporal_source)
            for item in normalized.temporal
        )
    status = _aggregate_status((*init, *goal, *invariants, *temporal))
    return EvaluationReport(
        status=status,
        init=init,
        goal=goal,
        invariants=invariants,
        temporal=temporal,
    )


def validate_task_template(
    value: TaskSpec | Mapping[str, Any],
    *,
    require_observations: bool = False,
) -> dict[str, Any]:
    """Compatibility alias for :func:`validate_task_spec`."""
    return validate_task_spec(value, require_observations=require_observations)


def evaluate_task_spec(
    template: TaskSpec | Mapping[str, Any],
    observations: Mapping[str, Any],
    *,
    trajectory: Sequence[Mapping[str, Any]] | None = None,
) -> EvaluationReport:
    """Evaluate one TaskSpec; kept separate as the primary public spelling."""
    return evaluate_task_template(template, observations, trajectory=trajectory)


def build_validation_certificate(
    report: EvaluationReport,
    *,
    certificate_id: str,
    task_template_hash: str,
    scene_instance_hash: str,
    witness_id: str | None = None,
    checker: str = "task_spec.evaluator",
    checker_version: str = "task_spec/v0.1",
    metrics: Mapping[str, Any] | None = None,
    evidence_refs: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ValidationCertificate:
    """Build a certificate while preserving each section's evidence records."""
    results = (*report.init, *report.goal, *report.invariants, *report.temporal)
    return ValidationCertificate(
        certificate_id=certificate_id,
        task_template_hash=task_template_hash,
        scene_instance_hash=scene_instance_hash,
        witness_id=witness_id,
        checker=checker,
        status=report.status.value,
        predicate_results=results,
        metrics={} if metrics is None else metrics,
        evidence_refs=tuple(evidence_refs),
        checker_version=checker_version,
        metadata={} if metadata is None else metadata,
    )


def _section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = root.get(name)
    if isinstance(value, Mapping):
        return value
    return root


def _aggregate_status(results: Sequence[PredicateEvaluation]) -> EvaluationStatus:
    if not results:
        return EvaluationStatus.SATISFIED
    statuses = {item.status for item in results}
    if EvaluationStatus.FAILED in statuses:
        return EvaluationStatus.FAILED
    if EvaluationStatus.UNAVAILABLE in statuses:
        return EvaluationStatus.UNAVAILABLE
    if EvaluationStatus.UNKNOWN in statuses:
        return EvaluationStatus.UNKNOWN
    if EvaluationStatus.NOT_RUN in statuses:
        return EvaluationStatus.NOT_RUN
    return EvaluationStatus.SATISFIED
