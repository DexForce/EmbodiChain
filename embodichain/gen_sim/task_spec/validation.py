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
from .expressions import (
    EvaluationStatus,
    PredicateEvaluation,
    evaluate_predicate,
    evaluate_temporal_condition,
)
from .spec import MilestoneSpec, TaskSpec

__all__ = [
    "EvaluationReport",
    "TaskSpecValidationError",
    "evaluate_task_spec",
    "evaluate_task_template",
    "validate_task_spec",
    "validate_task_template",
]

_LEGACY_EXPORTS = {
    "build_validation_certificate",
    "validate_action_witness",
    "validate_expansion_manifest",
    "validate_scene_instance",
    "validate_validation_certificate",
}


def __getattr__(name: str) -> Any:
    """Resolve moved evidence validators only for explicit legacy imports."""
    if name in _LEGACY_EXPORTS:
        from . import compat

        return getattr(compat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class TaskSpecValidationError(ValueError):
    """Raised when a TaskSpec violates its versioned protocol contract."""


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Evaluation of snapshot checks, temporal windows, and ordered milestones."""

    status: EvaluationStatus
    init: tuple[PredicateEvaluation, ...]
    goal: tuple[PredicateEvaluation, ...]
    invariants: tuple[PredicateEvaluation, ...]
    temporal: tuple[PredicateEvaluation, ...]
    milestones: tuple[PredicateEvaluation, ...] = ()

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
            "milestones": [item.to_dict() for item in self.milestones],
        }


def validate_task_spec(
    value: TaskSpec | Mapping[str, Any],
    *,
    require_observations: bool = False,
) -> dict[str, Any]:
    """Validate and normalize one compact TaskSpec.

    Args:
        value: Dataclass or JSON mapping to validate.
        require_observations: Require every goal, invariant, temporal, and milestone
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
            missing = (
                [
                    item.predicate_id or item.name
                    for item in (*template.goal, *template.invariants)
                    if item.observation is None
                ]
                + [
                    item.condition_id or item.predicate.name
                    for item in template.temporal
                    if item.predicate.observation is None
                ]
                + [
                    f"milestone.{milestone.id}.{item.predicate_id or item.name}"
                    for milestone in template.milestones
                    for item in milestone.achieve
                    if item.observation is None
                ]
            )
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


def evaluate_task_spec(
    template: TaskSpec | Mapping[str, Any],
    observations: Mapping[str, Any],
    *,
    trajectory: Sequence[Mapping[str, Any]] | None = None,
) -> EvaluationReport:
    """Evaluate a TaskSpec against explicit runtime observations.

    An adapter may provide section-specific mappings under ``init``, ``goal``,
    and ``invariants``.  When absent, the supplied mapping is used for all
    snapshot predicates.  Temporal conditions use ``trajectory`` or an
    explicit ``observations['trajectory']`` sequence. Milestones require all
    achievement predicates at one snapshot strictly after their predecessors.
    Missing measurements are reported as ``unavailable`` and never treated as success.
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
    milestones = _evaluate_milestones(normalized.milestones, temporal_source)
    status = _aggregate_status((*init, *goal, *invariants, *temporal, *milestones))
    return EvaluationReport(
        status=status,
        init=init,
        goal=goal,
        invariants=invariants,
        temporal=temporal,
        milestones=milestones,
    )


def _evaluate_milestones(
    milestones: Sequence[MilestoneSpec],
    trajectory: Sequence[Mapping[str, Any]] | None,
) -> tuple[PredicateEvaluation, ...]:
    if not milestones:
        return ()
    if trajectory is not None and (
        not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes))
    ):
        raise TypeError("trajectory must be a sequence of observation mappings.")
    snapshots = () if trajectory is None else trajectory
    by_id = {item.id: item for item in milestones}
    results: dict[str, PredicateEvaluation] = {}
    achieved_at: dict[str, int] = {}

    def evaluate(milestone_id: str) -> PredicateEvaluation:
        if milestone_id in results:
            return results[milestone_id]
        milestone = by_id[milestone_id]
        predecessors = [evaluate(item) for item in milestone.after]
        status = EvaluationStatus.UNAVAILABLE
        reason = "no trajectory evidence was supplied for milestone"
        observed: dict[str, Any] | None = None
        evidence_refs: tuple[str, ...] = ()
        if any(item.status is not EvaluationStatus.SATISFIED for item in predecessors):
            reason = "milestone predecessors have not been verified"
        else:
            start = max((achieved_at[item] + 1 for item in milestone.after), default=0)
            statuses = []
            for step in range(start, len(snapshots)):
                checks = tuple(
                    evaluate_predicate(
                        predicate,
                        snapshots[step],
                        predicate_id=predicate.predicate_id
                        or f"{milestone_id}.{index}",
                    )
                    for index, predicate in enumerate(milestone.achieve)
                )
                frame_status = _aggregate_status(checks)
                statuses.append(frame_status)
                observed = {
                    "step": step,
                    "predicates": [item.to_dict() for item in checks],
                }
                evidence_refs = tuple(
                    ref for item in checks for ref in item.evidence_refs
                )
                if frame_status is EvaluationStatus.SATISFIED:
                    achieved_at[milestone_id] = step
                    status = EvaluationStatus.SATISFIED
                    reason = f"milestone achieved at step {step}"
                    break
            else:
                if statuses:
                    status = next(
                        (
                            item
                            for item in (
                                EvaluationStatus.UNAVAILABLE,
                                EvaluationStatus.UNKNOWN,
                                EvaluationStatus.NOT_RUN,
                            )
                            if item in statuses
                        ),
                        EvaluationStatus.FAILED,
                    )
                    reason = "milestone was not verified after its predecessors"
        result = PredicateEvaluation(
            predicate_id=f"milestone:{milestone_id}",
            predicate="semantic_goal",
            status=status,
            observed=observed,
            expected=True,
            reason=reason,
            evidence_refs=evidence_refs,
        )
        results[milestone_id] = result
        return result

    return tuple(evaluate(item.id) for item in milestones)


# Deprecated spellings kept for one transition release.
validate_task_template = validate_task_spec
evaluate_task_template = evaluate_task_spec


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
