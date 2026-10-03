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

"""The small, declarative expression language used by TaskSpec v0.1.

Expressions are intentionally data, rather than Python callbacks or strings
that are interpreted at runtime.  A TaskSpec can therefore be serialized and
cached without importing a simulator.  Runtime adapters provide measurements
under the observation key declared by each predicate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from .canonicalization import json_snapshot

__all__ = [
    "EVALUATION_STATUSES",
    "KNOWN_PREDICATES",
    "PREDICATE_OPERATORS",
    "EvaluationStatus",
    "Predicate",
    "PredicateEvaluation",
    "PredicateSpec",
    "TemporalCondition",
    "TemporalSpec",
    "evaluate_predicate",
    "evaluate_temporal_condition",
    "lookup_observation",
]


class EvaluationStatus(str, Enum):
    """Outcome vocabulary shared by predicates and certificates."""

    SATISFIED = "satisfied"
    FAILED = "failed"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    NOT_RUN = "not_run"


EVALUATION_STATUSES: Final = frozenset(item.value for item in EvaluationStatus)
PREDICATE_OPERATORS: Final = frozenset(
    {"eq", "ne", "gt", "gte", "lt", "lte", "in", "contains", "approx"}
)

# This is a deliberately finite v0.1 vocabulary.  The values describe the
# semantic measurement a backend must provide; they are not Python evaluator
# callbacks.  New names require a protocol version change or an explicit
# registry extension in a later version.
KNOWN_PREDICATES: Final = frozenset(
    {
        "exists",
        "relation",
        "state",
        "attribute",
        "orientation",
        "contact",
        "distance",
        "held",
        "semantic_goal",
        "object_upright",
        "poured",
        "handover_complete",
        "held_by_both_grippers",
        "articulation_joint_near",
        "pressed",
    }
)

_TEMPORAL_MODES: Final = frozenset({"eventually", "always"})
_PREDICATE_KEYS: Final = frozenset(
    {"id", "predicate", "arguments", "operator", "expected", "observation"}
)
_TEMPORAL_KEYS: Final = frozenset(
    {"id", "predicate", "within_steps", "start_step", "mode"}
)


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    try:
        result = json_snapshot(dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain JSON-safe values.") from exc
    if not isinstance(result, dict):  # pragma: no cover - guarded by snapshot
        raise TypeError(f"{context} must be a mapping.")
    return result


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, context)


@dataclass(frozen=True, slots=True)
class Predicate:
    """One typed, observation-backed TaskSpec predicate.

    Args:
        name: One of :data:`KNOWN_PREDICATES`.
        arguments: JSON arguments such as ``subject`` and ``reference``.
        operator: Comparison to apply to the observed measurement.
        expected: JSON value used by ``operator``.
        observation: Direct or dotted key supplied by a runtime observer.
        predicate_id: Stable local evidence identifier.  TaskTemplate assigns
            one when omitted.

    A missing observation key is reported as ``unavailable`` by the evaluator;
    it is never interpreted as a Python expression.
    """

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    operator: str = "eq"
    expected: Any = True
    observation: str | None = None
    predicate_id: str | None = None

    def __post_init__(self) -> None:
        name = _nonempty(self.name, "Predicate.name")
        if name not in KNOWN_PREDICATES:
            raise ValueError(
                f"Unknown TaskSpec predicate {name!r}; supported values are "
                f"{sorted(KNOWN_PREDICATES)}."
            )
        if self.operator not in PREDICATE_OPERATORS:
            raise ValueError(
                f"Predicate.operator must be one of {sorted(PREDICATE_OPERATORS)}."
            )
        arguments = _mapping(self.arguments, "Predicate.arguments")
        expected = json_snapshot(self.expected)
        observation = _optional_string(self.observation, "Predicate.observation")
        predicate_id = _optional_string(self.predicate_id, "Predicate.predicate_id")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "expected", expected)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "predicate_id", predicate_id)

    @property
    def predicate(self) -> str:
        """Return the serialized predicate name."""
        return self.name

    @property
    def args(self) -> Mapping[str, Any]:
        """Compatibility alias for callers that use the shorter AST name."""
        return self.arguments

    @property
    def id(self) -> str | None:
        """Return the local evidence identifier, when one is assigned."""
        return self.predicate_id

    def with_id(self, predicate_id: str) -> Predicate:
        """Return this predicate with a validated local evidence identifier."""
        return Predicate(
            name=self.name,
            arguments=self.arguments,
            operator=self.operator,
            expected=self.expected,
            observation=self.observation,
            predicate_id=predicate_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        return {
            "id": self.predicate_id,
            "predicate": self.name,
            "arguments": json_snapshot(self.arguments),
            "operator": self.operator,
            "expected": json_snapshot(self.expected),
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | Predicate) -> Predicate:
        """Decode one strict predicate mapping."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Predicate must be a mapping.")
        raw = dict(value)
        # ``name``/``args``/``predicate_id`` are accepted as read-side aliases
        # so adapters can consume compact hand-authored specs.  Serialization
        # always uses the canonical names above.
        if "predicate" not in raw and "name" in raw:
            raw["predicate"] = raw.pop("name")
        if "arguments" not in raw and "args" in raw:
            raw["arguments"] = raw.pop("args")
        if "id" not in raw and "predicate_id" in raw:
            raw["id"] = raw.pop("predicate_id")
        if "expected" not in raw and "value" in raw:
            raw["expected"] = raw.pop("value")
        unknown = set(raw) - _PREDICATE_KEYS
        if unknown:
            raise ValueError(f"Predicate has unknown fields {sorted(unknown)}.")
        return cls(
            name=raw.get("predicate"),
            arguments=raw.get("arguments", {}),
            operator=raw.get("operator", "eq"),
            expected=raw.get("expected", True),
            observation=raw.get("observation"),
            predicate_id=raw.get("id"),
        )


@dataclass(frozen=True, slots=True)
class TemporalCondition:
    """A bounded ``eventually`` or ``always`` predicate condition."""

    predicate: Predicate
    within_steps: int
    start_step: int = 0
    mode: str = "eventually"
    condition_id: str | None = None

    def __post_init__(self) -> None:
        predicate = Predicate.from_dict(self.predicate)
        if (
            isinstance(self.within_steps, bool)
            or not isinstance(self.within_steps, int)
            or self.within_steps < 0
        ):
            raise ValueError("TemporalCondition.within_steps must be >= 0.")
        if (
            isinstance(self.start_step, bool)
            or not isinstance(self.start_step, int)
            or self.start_step < 0
        ):
            raise ValueError("TemporalCondition.start_step must be >= 0.")
        if self.mode not in _TEMPORAL_MODES:
            raise ValueError(
                f"TemporalCondition.mode must be one of {sorted(_TEMPORAL_MODES)}."
            )
        condition_id = _optional_string(self.condition_id, "TemporalCondition.id")
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "condition_id", condition_id)

    @property
    def id(self) -> str | None:
        """Return the local temporal evidence identifier."""
        return self.condition_id

    @property
    def end_step(self) -> int:
        """Return the inclusive end of this bounded observation window."""
        return self.start_step + self.within_steps

    def with_id(self, condition_id: str) -> TemporalCondition:
        """Return this condition with a validated local evidence identifier."""
        return TemporalCondition(
            predicate=self.predicate,
            within_steps=self.within_steps,
            start_step=self.start_step,
            mode=self.mode,
            condition_id=condition_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        return {
            "id": self.condition_id,
            "predicate": self.predicate.to_dict(),
            "within_steps": self.within_steps,
            "start_step": self.start_step,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any] | TemporalCondition
    ) -> TemporalCondition:
        """Decode one bounded temporal condition."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("TemporalCondition must be a mapping.")
        raw = dict(value)
        unknown = set(raw) - _TEMPORAL_KEYS
        if unknown:
            raise ValueError(f"TemporalCondition has unknown fields {sorted(unknown)}.")
        return cls(
            predicate=Predicate.from_dict(raw.get("predicate")),
            within_steps=raw.get("within_steps"),
            start_step=raw.get("start_step", 0),
            mode=raw.get("mode", "eventually"),
            condition_id=raw.get("id"),
        )


@dataclass(frozen=True, slots=True)
class PredicateEvaluation:
    """One auditable predicate measurement outcome."""

    predicate_id: str
    predicate: str
    status: EvaluationStatus
    observed: Any
    expected: Any
    reason: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.predicate_id, "PredicateEvaluation.predicate_id")
        _nonempty(self.predicate, "PredicateEvaluation.predicate")
        if not isinstance(self.status, EvaluationStatus):
            object.__setattr__(self, "status", EvaluationStatus(str(self.status)))
        object.__setattr__(self, "observed", json_snapshot(self.observed))
        object.__setattr__(self, "expected", json_snapshot(self.expected))
        _nonempty(self.reason, "PredicateEvaluation.reason")
        refs = tuple(self.evidence_refs)
        if any(not isinstance(item, str) or not item for item in refs):
            raise ValueError("PredicateEvaluation.evidence_refs must be strings.")
        object.__setattr__(self, "evidence_refs", refs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe evaluation record."""
        return {
            "predicate_id": self.predicate_id,
            "predicate": self.predicate,
            "status": self.status.value,
            "observed": self.observed,
            "expected": self.expected,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
        }


def lookup_observation(observations: Mapping[str, Any], key: str) -> tuple[bool, Any]:
    """Read a direct or dotted observation key without evaluating strings."""
    if not isinstance(observations, Mapping):
        raise TypeError("observations must be a mapping.")
    if key in observations:
        return True, observations[key]
    current: Any = observations
    for part in key.split("."):
        if not part or not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def evaluate_predicate(
    predicate: Predicate,
    observations: Mapping[str, Any],
    *,
    predicate_id: str | None = None,
) -> PredicateEvaluation:
    """Evaluate one predicate against declared observations.

    Missing observations and explicit ``unknown``/``unavailable`` records are
    preserved as statuses instead of being converted into a false success.
    """
    expression = Predicate.from_dict(predicate)
    local_id = predicate_id or expression.predicate_id or expression.name
    observation_key = expression.observation or local_id
    found, observed = lookup_observation(observations, observation_key)
    if not found:
        return PredicateEvaluation(
            predicate_id=local_id,
            predicate=expression.name,
            status=EvaluationStatus.UNAVAILABLE,
            observed=None,
            expected=expression.expected,
            reason=f"observation {observation_key!r} was not provided",
        )
    if isinstance(observed, Mapping) and "status" in observed:
        marker = observed.get("status")
        if marker in EVALUATION_STATUSES:
            return PredicateEvaluation(
                predicate_id=local_id,
                predicate=expression.name,
                status=EvaluationStatus(marker),
                observed=observed.get("value"),
                expected=expression.expected,
                reason=str(observed.get("reason") or f"measurement is {marker}"),
                evidence_refs=_string_refs(observed.get("evidence_refs", ())),
            )
        if "value" in observed:
            observed = observed["value"]
    if observed is None:
        return PredicateEvaluation(
            predicate_id=local_id,
            predicate=expression.name,
            status=EvaluationStatus.UNKNOWN,
            observed=None,
            expected=expression.expected,
            reason="observation value is null",
        )
    try:
        satisfied = _compare(observed, expression.expected, expression.operator)
    except (TypeError, ValueError) as exc:
        return PredicateEvaluation(
            predicate_id=local_id,
            predicate=expression.name,
            status=EvaluationStatus.UNKNOWN,
            observed=observed,
            expected=expression.expected,
            reason=f"measurement could not be compared: {exc}",
        )
    return PredicateEvaluation(
        predicate_id=local_id,
        predicate=expression.name,
        status=(EvaluationStatus.SATISFIED if satisfied else EvaluationStatus.FAILED),
        observed=observed,
        expected=expression.expected,
        reason="predicate satisfied" if satisfied else "predicate comparison failed",
    )


def evaluate_temporal_condition(
    condition: TemporalCondition,
    trajectory: Sequence[Mapping[str, Any]],
) -> PredicateEvaluation:
    """Evaluate one bounded temporal condition over observation snapshots."""
    expression = TemporalCondition.from_dict(condition)
    local_id = (
        expression.condition_id or expression.predicate.predicate_id or "temporal"
    )
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        raise TypeError("trajectory must be a sequence of observation mappings.")
    window = trajectory[
        expression.start_step : expression.start_step + expression.within_steps + 1
    ]
    if not window:
        return PredicateEvaluation(
            predicate_id=local_id,
            predicate=expression.predicate.name,
            status=EvaluationStatus.UNAVAILABLE,
            observed=None,
            expected=expression.predicate.expected,
            reason="bounded temporal window has no observations",
        )
    evaluations = [
        evaluate_predicate(
            expression.predicate,
            snapshot,
            predicate_id=expression.predicate.predicate_id or local_id,
        )
        for snapshot in window
    ]
    statuses = [item.status for item in evaluations]
    if expression.mode == "eventually":
        if EvaluationStatus.SATISFIED in statuses:
            status = EvaluationStatus.SATISFIED
            reason = "predicate was satisfied within the bounded window"
        elif EvaluationStatus.FAILED in statuses and all(
            item in {EvaluationStatus.FAILED, EvaluationStatus.SATISFIED}
            for item in statuses
        ):
            status = EvaluationStatus.FAILED
            reason = "predicate was never satisfied within the bounded window"
        elif EvaluationStatus.UNKNOWN in statuses:
            status = EvaluationStatus.UNKNOWN
            reason = "temporal window contains an unknown measurement"
        elif EvaluationStatus.NOT_RUN in statuses:
            status = EvaluationStatus.NOT_RUN
            reason = "temporal window was not run"
        else:
            status = EvaluationStatus.UNAVAILABLE
            reason = "temporal window lacks a usable measurement"
    else:  # always
        if EvaluationStatus.FAILED in statuses:
            status = EvaluationStatus.FAILED
            reason = "predicate failed within the bounded window"
        elif EvaluationStatus.UNAVAILABLE in statuses:
            status = EvaluationStatus.UNAVAILABLE
            reason = "temporal window lacks a required measurement"
        elif EvaluationStatus.UNKNOWN in statuses:
            status = EvaluationStatus.UNKNOWN
            reason = "temporal window contains an unknown measurement"
        elif EvaluationStatus.NOT_RUN in statuses:
            status = EvaluationStatus.NOT_RUN
            reason = "temporal window was not run"
        else:
            status = EvaluationStatus.SATISFIED
            reason = "predicate held throughout the bounded window"
    return PredicateEvaluation(
        predicate_id=local_id,
        predicate=expression.predicate.name,
        status=status,
        observed=[item.observed for item in evaluations],
        expected=expression.predicate.expected,
        reason=reason,
        evidence_refs=tuple(ref for item in evaluations for ref in item.evidence_refs),
    )


def _compare(observed: Any, expected: Any, operator: str) -> bool:
    if operator == "eq":
        return observed == expected
    if operator == "ne":
        return observed != expected
    if operator == "gt":
        return observed > expected
    if operator == "gte":
        return observed >= expected
    if operator == "lt":
        return observed < expected
    if operator == "lte":
        return observed <= expected
    if operator == "in":
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
            raise TypeError("operator='in' expects a sequence as expected value")
        return observed in expected
    if operator == "contains":
        return expected in observed
    if operator == "approx":
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            raise TypeError("operator='approx' requires a numeric observation")
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            raise TypeError("operator='approx' requires a numeric expected value")
        return math.isclose(
            float(observed), float(expected), rel_tol=1e-6, abs_tol=1e-6
        )
    raise ValueError(f"unsupported predicate operator {operator!r}")


def _string_refs(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


# Protocol-facing aliases keep the vocabulary readable for integrations that
# call every expression a ``*Spec`` while retaining the compact class names.
PredicateSpec = Predicate
TemporalSpec = TemporalCondition
