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

"""The compact, normative TaskSpec contract.

TaskSpec is deliberately smaller than the runtime artifact model.  It owns
roles, conditions, milestones, downstream requirements, and semantic identity.
Scene bindings, trajectories, execution attempts, certificates, and expansion
receipts remain owned by their existing integrations and only reference the
TaskSpec hash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .canonicalization import (
    CANONICALIZATION_VERSION,
    canonical_hash,
    canonical_json,
    json_snapshot,
)
from .contracts import RequirementSpec, RoleSpec
from .expressions import Predicate, TemporalCondition

__all__ = [
    "MILESTONE_SCHEMA",
    "TASK_SPEC_SCHEMA",
    "MilestoneSpec",
    "TaskRequirement",
    "TaskSpec",
    "TaskTemplate",
]

TASK_SPEC_SCHEMA: Final = "task_spec/v0.1"
# Kept for readers of the pre-simplification schema.
MILESTONE_SCHEMA: Final = "task_spec_milestone/v0.1"

_CARDINALITIES = frozenset({"one", "all", "count"})
_CONSUMERS = frozenset({"asset", "scene", "embodiment", "task_program", "expansion"})
_LEGACY_SCHEMA = "task_template/v0.1"


def _text(value: Any, context: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _sequence(value: Any, context: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence.")
    return tuple(value)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    result = json_snapshot(dict(value))
    if not isinstance(result, dict):  # pragma: no cover - guarded by snapshot
        raise TypeError(f"{context} must be a mapping.")
    return result


def _strings(value: Any, context: str) -> tuple[str, ...]:
    values = _sequence(value, context)
    result = tuple(
        _text(item, f"{context}[{index}]") for index, item in enumerate(values)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{context} must not contain duplicates.")
    return result  # type: ignore[return-value]


def _predicate(value: Predicate | Mapping[str, Any]) -> Predicate:
    raw = dict(value) if isinstance(value, Mapping) else value
    if isinstance(raw, dict):
        # The compact TaskSpec form keeps common arguments beside the predicate
        # rather than forcing every consumer to know the internal AST shape.
        arguments = dict(raw.get("arguments", {}))
        for key in ("subject", "reference", "key", "relation", "count"):
            if key in raw:
                if key in arguments and arguments[key] != raw[key]:
                    raise ValueError(f"Predicate argument {key!r} is duplicated.")
                arguments[key] = raw.pop(key)
        if arguments:
            raw["arguments"] = arguments
    return Predicate.from_dict(raw)


def _predicate_dict(predicate: Predicate) -> dict[str, Any]:
    raw = predicate.to_dict()
    arguments = dict(raw.pop("arguments", {}))
    # Preserve the compact public spelling for the common role references.
    for key in ("subject", "reference", "key", "relation", "count"):
        if key in arguments:
            raw[key] = arguments.pop(key)
    if arguments:
        raw["arguments"] = arguments
    if raw.get("id") is None:
        raw.pop("id", None)
    if raw.get("observation") is None:
        raw.pop("observation", None)
    return raw


def _semantic_predicate_dict(predicate: Predicate) -> dict[str, Any]:
    result = _predicate_dict(predicate)
    result.pop("id", None)
    result.pop("observation", None)
    return result


def _semantic_temporal_dict(condition: TemporalCondition) -> dict[str, Any]:
    return {
        "predicate": _semantic_predicate_dict(condition.predicate),
        "within_steps": condition.within_steps,
        "start_step": condition.start_step,
        "mode": condition.mode,
    }


def _semantic_milestone_dict(milestone: MilestoneSpec) -> dict[str, Any]:
    """Return a milestone projection without runtime evidence identifiers."""
    result: dict[str, Any] = {
        "id": milestone.id,
        "achieve": [
            _semantic_predicate_dict(predicate) for predicate in milestone.achieve
        ],
    }
    if milestone.after:
        result["after"] = list(milestone.after)
    return result


def _role(value: RoleSpec | Mapping[str, Any]) -> RoleSpec:
    if isinstance(value, RoleSpec):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("TaskSpec.roles values must be mappings.")
    raw = dict(value)
    if "name" not in raw:
        raw["name"] = raw.pop("id", raw.pop("role", None))
    if "cardinality" in raw:
        raw["quantifier"] = raw.pop("cardinality")
    if "properties" in raw and "attributes" not in raw:
        raw["attributes"] = raw.pop("properties")
    allowed = {
        "id",
        "role",
        "name",
        "kind",
        "required",
        "quantifier",
        "count",
        "cardinality",
        "source_structure",
        "affordances",
        "initial_state",
        "attributes",
        "properties",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"TaskSpec role has unknown fields {sorted(unknown)}.")
    return RoleSpec.from_dict(raw)


def _role_dict(role: RoleSpec) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": role.name,
        "kind": role.kind,
        "cardinality": role.quantifier,
        "affordances": list(role.affordances),
        "properties": json_snapshot(role.attributes),
    }
    if role.quantifier == "count":
        result["count"] = role.count
    if not role.required:
        result["required"] = False
    if role.source_structure is not None:
        result["source_structure"] = role.source_structure
    return result


@dataclass(frozen=True, slots=True)
class TaskRequirement:
    """A derived, consumer-specific requirement projection."""

    consumer: str
    key: str
    value: Any = True
    required: bool = True

    def __post_init__(self) -> None:
        consumer = _text(self.consumer, "TaskRequirement.consumer")
        if consumer not in _CONSUMERS:
            raise ValueError(
                f"TaskRequirement.consumer must be one of {sorted(_CONSUMERS)}."
            )
        key = _text(self.key, "TaskRequirement.key")
        if type(self.required) is not bool:
            raise TypeError("TaskRequirement.required must be a boolean.")
        object.__setattr__(self, "consumer", consumer)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "value", json_snapshot(self.value))

    @classmethod
    def from_value(
        cls, value: TaskRequirement | RequirementSpec | Mapping[str, Any]
    ) -> TaskRequirement:
        if isinstance(value, cls):
            return value
        if isinstance(value, RequirementSpec):
            return cls(
                consumer=value.kind,
                key=value.name,
                value=value.value,
                required=value.required,
            )
        if not isinstance(value, Mapping):
            raise TypeError("TaskSpec requirements must be mappings.")
        raw = dict(value)
        if "consumer" not in raw:
            raw["consumer"] = raw.pop("kind", None)
        if "key" not in raw:
            raw["key"] = raw.pop("name", None)
        allowed = {"consumer", "key", "value", "required", "kind", "name"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"TaskSpec requirement has unknown fields {sorted(unknown)}."
            )
        return cls(
            consumer=raw.get("consumer"),
            key=raw.get("key"),
            value=raw.get("value", True),
            required=raw.get("required", True),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "consumer": self.consumer,
            "key": self.key,
            "value": json_snapshot(self.value),
        }
        if not self.required:
            result["required"] = False
        return result


@dataclass(frozen=True, slots=True)
class MilestoneSpec:
    """A required long-horizon achievement with explicit predecessors."""

    milestone_id: str
    achieve: tuple[Predicate | Mapping[str, Any], ...]
    after: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        milestone_id = _text(self.milestone_id, "MilestoneSpec.id")
        achieve = tuple(_predicate(item) for item in self.achieve)
        if not achieve:
            raise ValueError("MilestoneSpec.achieve must not be empty.")
        after = _strings(self.after, "MilestoneSpec.after")
        object.__setattr__(self, "milestone_id", milestone_id)
        object.__setattr__(self, "achieve", achieve)
        object.__setattr__(self, "after", after)

    @property
    def id(self) -> str:
        """Return the stable milestone identifier."""
        return self.milestone_id

    @classmethod
    def from_value(cls, value: MilestoneSpec | Mapping[str, Any]) -> MilestoneSpec:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("TaskSpec milestones must be mappings.")
        raw = dict(value)
        allowed = {"id", "milestone_id", "achieve", "after"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"TaskSpec milestone has unknown fields {sorted(unknown)}."
            )
        return cls(
            milestone_id=raw.get("id", raw.get("milestone_id")),
            achieve=tuple(raw.get("achieve", ())),
            after=tuple(raw.get("after", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.milestone_id,
            "achieve": [_predicate_dict(item) for item in self.achieve],
        }
        if self.after:
            result["after"] = list(self.after)
        return result


def _validate_milestone_order(milestones: tuple[MilestoneSpec, ...]) -> None:
    """Reject cyclic milestone dependencies before hashing the specification."""
    dependencies = {item.milestone_id: set(item.after) for item in milestones}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(milestone_id: str) -> None:
        if milestone_id in visiting:
            raise ValueError("TaskSpec milestones must form an acyclic order.")
        if milestone_id in visited:
            return
        visiting.add(milestone_id)
        for dependency in dependencies[milestone_id]:
            visit(dependency)
        visiting.remove(milestone_id)
        visited.add(milestone_id)

    for milestone_id in dependencies:
        visit(milestone_id)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Compact, action-sequence-independent normative task specification."""

    task_id: str | None = None
    instruction: str | None = None
    roles: tuple[RoleSpec | Mapping[str, Any], ...] = ()
    init: tuple[Predicate | Mapping[str, Any], ...] = ()
    goal: tuple[Predicate | Mapping[str, Any], ...] = ()
    invariants: tuple[Predicate | Mapping[str, Any], ...] = ()
    temporal: tuple[TemporalCondition | Mapping[str, Any], ...] = ()
    milestones: tuple[MilestoneSpec | Mapping[str, Any], ...] = ()
    requirements: tuple[TaskRequirement | RequirementSpec | Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = TASK_SPEC_SCHEMA
    canonicalization_version: str = CANONICALIZATION_VERSION
    semantic_hash: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        schema_version = _text(self.schema_version, "TaskSpec.schema_version")
        if schema_version == _LEGACY_SCHEMA:
            schema_version = TASK_SPEC_SCHEMA
        if schema_version != TASK_SPEC_SCHEMA:
            raise ValueError(f"TaskSpec.schema_version must be {TASK_SPEC_SCHEMA!r}.")
        if self.canonicalization_version != CANONICALIZATION_VERSION:
            raise ValueError(
                f"TaskSpec.canonicalization_version must be {CANONICALIZATION_VERSION!r}."
            )
        task_id = _text(self.task_id, "TaskSpec.task_id", optional=True)
        instruction = _text(self.instruction, "TaskSpec.instruction", optional=True)
        roles = tuple(_role(item) for item in self.roles)
        if len({item.name for item in roles}) != len(roles):
            raise ValueError("TaskSpec.roles must have unique IDs.")
        init = tuple(_predicate(item) for item in self.init)
        roles = _hydrate_role_initial_state(roles, init)
        goal = tuple(_predicate(item) for item in self.goal)
        invariants = tuple(_predicate(item) for item in self.invariants)
        temporal = tuple(TemporalCondition.from_dict(item) for item in self.temporal)
        milestones = tuple(MilestoneSpec.from_value(item) for item in self.milestones)
        milestone_ids = {item.milestone_id for item in milestones}
        if len(milestone_ids) != len(milestones):
            raise ValueError("TaskSpec.milestones must have unique IDs.")
        if any(set(item.after) - milestone_ids for item in milestones):
            raise ValueError(
                "TaskSpec milestone.after references an unknown milestone."
            )
        _validate_milestone_order(milestones)
        requirements = tuple(
            TaskRequirement.from_value(item) for item in self.requirements
        )
        metadata = _mapping(self.metadata, "TaskSpec.metadata")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(
            self, "roles", tuple(sorted(roles, key=lambda item: item.name))
        )
        object.__setattr__(self, "init", _sort_predicates(init, "init"))
        object.__setattr__(self, "goal", _sort_predicates(goal, "goal"))
        object.__setattr__(
            self, "invariants", _sort_predicates(invariants, "invariant")
        )
        object.__setattr__(self, "temporal", temporal)
        object.__setattr__(
            self, "milestones", tuple(sorted(milestones, key=lambda item: item.id))
        )
        object.__setattr__(
            self,
            "requirements",
            tuple(sorted(requirements, key=lambda item: (item.consumer, item.key))),
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self, "semantic_hash", canonical_hash(self.semantic_payload())
        )

    def semantic_payload(self) -> dict[str, Any]:
        """Return only normative fields used for semantic identity."""
        return {
            "schema_version": self.schema_version,
            "canonicalization_version": self.canonicalization_version,
            "roles": [_role_dict(item) for item in self.roles],
            "init": [_semantic_predicate_dict(item) for item in self.init],
            "goal": [_semantic_predicate_dict(item) for item in self.goal],
            "invariants": [_semantic_predicate_dict(item) for item in self.invariants],
            "temporal": [_semantic_temporal_dict(item) for item in self.temporal],
            "milestones": [_semantic_milestone_dict(item) for item in self.milestones],
            "requirements": [item.to_dict() for item in self.requirements],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the compact cacheable TaskSpec mapping."""
        result = {
            "schema_version": self.schema_version,
            "canonicalization_version": self.canonicalization_version,
            "roles": [_role_dict(item) for item in self.roles],
            "init": [_predicate_dict(item) for item in self.init],
            "goal": [_predicate_dict(item) for item in self.goal],
            "invariants": [_predicate_dict(item) for item in self.invariants],
            "temporal": [item.to_dict() for item in self.temporal],
            "milestones": [item.to_dict() for item in self.milestones],
            "requirements": [item.to_dict() for item in self.requirements],
        }
        if self.task_id is not None:
            result["task_id"] = self.task_id
        if self.instruction is not None:
            result["instruction"] = self.instruction
        if self.metadata:
            result["metadata"] = json_snapshot(self.metadata)
        result["semantic_hash"] = self.semantic_hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | TaskSpec) -> TaskSpec:
        """Decode a compact TaskSpec or a compatible legacy template document."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("TaskSpec must be a mapping.")
        raw = dict(value)
        allowed = {
            "schema_version",
            "canonicalization_version",
            "task_id",
            "instruction",
            "roles",
            "init",
            "goal",
            "invariants",
            "temporal",
            "milestones",
            "requirements",
            "metadata",
            "provenance",
            "semantic_hash",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"TaskSpec has unknown fields {sorted(unknown)}.")
        metadata = dict(raw.get("metadata", {}))
        if "provenance" in raw:
            metadata.setdefault("provenance", raw["provenance"])
        spec = cls(
            task_id=raw.get("task_id"),
            instruction=raw.get("instruction"),
            roles=tuple(raw.get("roles", ())),
            init=tuple(raw.get("init", ())),
            goal=tuple(raw.get("goal", ())),
            invariants=tuple(raw.get("invariants", ())),
            temporal=tuple(raw.get("temporal", ())),
            milestones=tuple(raw.get("milestones", ())),
            requirements=tuple(raw.get("requirements", ())),
            metadata=metadata,
            schema_version=raw.get("schema_version", TASK_SPEC_SCHEMA),
            canonicalization_version=raw.get(
                "canonicalization_version", CANONICALIZATION_VERSION
            ),
        )
        if (
            raw.get("semantic_hash") is not None
            and raw["semantic_hash"] != spec.semantic_hash
        ):
            raise ValueError(
                "TaskSpec.semantic_hash does not match its semantic payload."
            )
        return spec


def _sort_predicates(values: Sequence[Predicate], prefix: str) -> tuple[Predicate, ...]:
    ordered = sorted(values, key=lambda item: canonical_json(item.to_dict()))
    used = {item.predicate_id for item in ordered if item.predicate_id is not None}
    result: list[Predicate] = []
    index = 1
    for item in ordered:
        predicate_id = item.predicate_id
        if predicate_id is None:
            while f"{prefix}_{index:03d}" in used:
                index += 1
            predicate_id = f"{prefix}_{index:03d}"
            used.add(predicate_id)
            index += 1
        result.append(item.with_id(predicate_id))
    if len(used) != len(result):
        raise ValueError(f"TaskSpec.{prefix} predicates must have unique IDs.")
    return tuple(result)


def _hydrate_role_initial_state(
    roles: Sequence[RoleSpec], init: Sequence[Predicate]
) -> tuple[RoleSpec, ...]:
    """Keep the compact ``init`` section as the canonical state source."""
    states: dict[str, dict[str, Any]] = {
        role.name: dict(role.initial_state) for role in roles
    }
    for predicate in init:
        if predicate.name != "state":
            continue
        subject = predicate.arguments.get("subject")
        key = predicate.arguments.get("key")
        if isinstance(subject, str) and isinstance(key, str) and subject in states:
            states[subject][key] = predicate.expected
    return tuple(
        RoleSpec(
            name=role.name,
            kind=role.kind,
            required=role.required,
            quantifier=role.quantifier,
            count=role.count,
            source_structure=role.source_structure,
            affordances=role.affordances,
            initial_state=states[role.name],
            attributes=role.attributes,
        )
        for role in roles
    )


# Compatibility name for callers that used the pre-simplification model.
TaskTemplate = TaskSpec
