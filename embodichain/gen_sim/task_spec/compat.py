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

"""Compatibility contracts for legacy TaskSpec runtime artifacts.

New production code should use :class:`TaskSpec` and
:class:`TaskRequirement`; these artifact types remain only for legacy
documents and migration readers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .canonicalization import (
    CANONICALIZATION_VERSION,
    canonical_hash,
    canonical_json,
    json_snapshot,
)
from .contracts import RoleSpec

from .expressions import (
    EVALUATION_STATUSES,
    EvaluationStatus,
    Predicate,
    PredicateEvaluation,
    TemporalCondition,
)

__all__ = [
    "ACTION_WITNESS_SCHEMA",
    "EXPANSION_MANIFEST_SCHEMA",
    "SCENE_INSTANCE_SCHEMA",
    "TASK_TEMPLATE_SCHEMA",
    "VALIDATION_CERTIFICATE_SCHEMA",
    "ActionWitness",
    "ExpansionManifest",
    "RequirementSpec",
    "SceneInstance",
    "TaskTemplate",
    "ValidationCertificate",
]

TASK_TEMPLATE_SCHEMA: Final = "task_template/v0.1"
REQUIREMENT_KINDS: Final = frozenset(
    {"asset", "scene", "embodiment", "task_program", "expansion"}
)
SCENE_INSTANCE_SCHEMA: Final = "scene_instance/v0.1"
ACTION_WITNESS_SCHEMA: Final = "action_witness/v0.1"
EXPANSION_MANIFEST_SCHEMA: Final = "expansion_manifest/v0.1"
VALIDATION_CERTIFICATE_SCHEMA: Final = "validation_certificate/v0.1"

_WITNESS_STATUSES: Final = frozenset(
    {"not_run", "prepared", "succeeded", "failed", "cancelled"}
)
_HASH_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _optional_text(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _text(value, context)


def _bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{context} must be a boolean.")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}.")
    return value


def _strings(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence of strings.")
    result = tuple(
        _text(item, f"{context}[{index}]") for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{context} must not contain duplicates.")
    return result


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    snapshot = json_snapshot(dict(value))
    if not isinstance(snapshot, dict):  # pragma: no cover - guarded by snapshot
        raise TypeError(f"{context} must be a mapping.")
    return snapshot


def _hash(value: Any, context: str) -> str:
    result = _text(value, context)
    if _HASH_RE.fullmatch(result) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest.")
    return result


def _sequence(value: Any, context: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence.")
    return tuple(value)


def _predicate(value: Any) -> Predicate:
    return Predicate.from_dict(value)


def _predicates(value: Any, context: str) -> tuple[Predicate, ...]:
    return tuple(_predicate(item) for item in _sequence(value, context))


def _temporal(value: Any) -> tuple[TemporalCondition, ...]:
    return tuple(
        TemporalCondition.from_dict(item)
        for item in _sequence(value, "TaskTemplate.temporal")
    )


@dataclass(frozen=True, slots=True)
class RequirementSpec:
    """A cross-engine requirement derived from normative task semantics."""

    kind: str
    name: str
    value: Any = True
    required: bool = True

    def __post_init__(self) -> None:
        kind = _text(self.kind, "RequirementSpec.kind")
        if kind not in REQUIREMENT_KINDS:
            raise ValueError(
                f"RequirementSpec.kind must be one of {sorted(REQUIREMENT_KINDS)}."
            )
        name = _text(self.name, "RequirementSpec.name")
        required = _bool(self.required, "RequirementSpec.required")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", json_snapshot(self.value))
        object.__setattr__(self, "required", required)

    @property
    def domain(self) -> str:
        """Compatibility alias for ``kind``."""
        return self.kind

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical requirement mapping."""
        return {
            "kind": self.kind,
            "name": self.name,
            "value": json_snapshot(self.value),
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | RequirementSpec) -> RequirementSpec:
        """Decode one requirement mapping."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("RequirementSpec must be a mapping.")
        raw = dict(value)
        if "kind" not in raw and "domain" in raw:
            raw["kind"] = raw.pop("domain")
        allowed = {"kind", "name", "value", "required"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"RequirementSpec has unknown fields {sorted(unknown)}.")
        return cls(
            kind=raw.get("kind"),
            name=raw.get("name"),
            value=raw.get("value", True),
            required=raw.get("required", True),
        )


@dataclass(frozen=True, slots=True)
class TaskTemplate:
    """Normative, action-sequence-independent TaskSpec source."""

    task_id: str
    instruction: str
    roles: tuple[RoleSpec, ...] = ()
    init: tuple[Predicate, ...] = ()
    goal: tuple[Predicate, ...] = ()
    invariants: tuple[Predicate, ...] = ()
    temporal: tuple[TemporalCondition, ...] = ()
    requirements: tuple[RequirementSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = TASK_TEMPLATE_SCHEMA
    canonicalization_version: str = CANONICALIZATION_VERSION
    semantic_hash: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        task_id = _text(self.task_id, "TaskTemplate.task_id")
        instruction = _text(self.instruction, "TaskTemplate.instruction")
        if self.schema_version != TASK_TEMPLATE_SCHEMA:
            raise ValueError(
                f"TaskTemplate.schema_version must be {TASK_TEMPLATE_SCHEMA!r}."
            )
        if self.canonicalization_version != CANONICALIZATION_VERSION:
            raise ValueError(
                "TaskTemplate.canonicalization_version must be "
                f"{CANONICALIZATION_VERSION!r}."
            )
        roles = tuple(RoleSpec.from_dict(item) for item in self.roles)
        if len({item.name for item in roles}) != len(roles):
            raise ValueError("TaskTemplate.roles must have unique names.")
        init = tuple(Predicate.from_dict(item) for item in self.init)
        goal = tuple(Predicate.from_dict(item) for item in self.goal)
        invariants = tuple(Predicate.from_dict(item) for item in self.invariants)
        temporal = tuple(TemporalCondition.from_dict(item) for item in self.temporal)
        requirements = tuple(
            RequirementSpec.from_dict(item) for item in self.requirements
        )
        metadata = _mapping(self.metadata, "TaskTemplate.metadata")
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
        object.__setattr__(self, "temporal", _sort_temporal(temporal))
        object.__setattr__(
            self,
            "requirements",
            tuple(sorted(requirements, key=lambda item: (item.kind, item.name))),
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self, "semantic_hash", canonical_hash(self.semantic_payload())
        )

    def semantic_payload(self) -> dict[str, Any]:
        """Return the fields that define normative task identity.

        Generation metadata is intentionally excluded, so rerunning Task
        Engine with a different model or candidate vote does not change the
        reusable TaskSpec identity.
        """
        return {
            "schema_version": self.schema_version,
            "canonicalization_version": self.canonicalization_version,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "roles": [item.to_dict() for item in self.roles],
            "init": [item.to_dict() for item in self.init],
            "goal": [item.to_dict() for item in self.goal],
            "invariants": [item.to_dict() for item in self.invariants],
            "temporal": [item.to_dict() for item in self.temporal],
            "requirements": [item.to_dict() for item in self.requirements],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete cacheable TaskTemplate mapping."""
        result = self.semantic_payload()
        result.update(
            {
                "metadata": json_snapshot(self.metadata),
                "semantic_hash": self.semantic_hash,
            }
        )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | TaskTemplate) -> TaskTemplate:
        """Decode and verify one cacheable TaskTemplate mapping."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("TaskTemplate must be a mapping.")
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
            "requirements",
            "metadata",
            "semantic_hash",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"TaskTemplate has unknown fields {sorted(unknown)}.")
        template = cls(
            task_id=raw.get("task_id"),
            instruction=raw.get("instruction"),
            roles=tuple(RoleSpec.from_dict(item) for item in raw.get("roles", ())),
            init=_predicates(raw.get("init", ()), "TaskTemplate.init"),
            goal=_predicates(raw.get("goal", ()), "TaskTemplate.goal"),
            invariants=_predicates(
                raw.get("invariants", ()), "TaskTemplate.invariants"
            ),
            temporal=_temporal(raw.get("temporal", ())),
            requirements=tuple(
                RequirementSpec.from_dict(item)
                for item in _sequence(
                    raw.get("requirements", ()), "TaskTemplate.requirements"
                )
            ),
            metadata=raw.get("metadata", {}),
            schema_version=raw.get("schema_version", TASK_TEMPLATE_SCHEMA),
            canonicalization_version=raw.get(
                "canonicalization_version", CANONICALIZATION_VERSION
            ),
        )
        provided_hash = raw.get("semantic_hash")
        if provided_hash is not None and provided_hash != template.semantic_hash:
            raise ValueError(
                "TaskTemplate.semantic_hash does not match its canonical semantic payload."
            )
        return template


@dataclass(frozen=True, slots=True)
class SceneInstance:
    """Grounded scene identity and measured initial state for a template."""

    scene_id: str
    task_template_hash: str
    assets: tuple[Mapping[str, Any], ...] = ()
    role_bindings: Mapping[str, Any] = field(default_factory=dict)
    embodiment: str | None = None
    initial_state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCENE_INSTANCE_SCHEMA
    content_hash: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        scene_id = _text(self.scene_id, "SceneInstance.scene_id")
        task_hash = _hash(self.task_template_hash, "SceneInstance.task_template_hash")
        if self.schema_version != SCENE_INSTANCE_SCHEMA:
            raise ValueError(
                f"SceneInstance.schema_version must be {SCENE_INSTANCE_SCHEMA!r}."
            )
        assets = tuple(_mapping(item, "SceneInstance.assets[]") for item in self.assets)
        bindings = _mapping(self.role_bindings, "SceneInstance.role_bindings")
        embodiment = _optional_text(self.embodiment, "SceneInstance.embodiment")
        initial_state = _mapping(self.initial_state, "SceneInstance.initial_state")
        metadata = _mapping(self.metadata, "SceneInstance.metadata")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "task_template_hash", task_hash)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "role_bindings", bindings)
        object.__setattr__(self, "embodiment", embodiment)
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self, "content_hash", canonical_hash(self.identity_payload())
        )

    @property
    def task_hash(self) -> str:
        """Compatibility alias for the referenced TaskTemplate hash."""
        return self.task_template_hash

    def identity_payload(self) -> dict[str, Any]:
        """Return scene identity fields excluding derived and audit metadata."""
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "task_template_hash": self.task_template_hash,
            "assets": [json_snapshot(item) for item in self.assets],
            "role_bindings": json_snapshot(self.role_bindings),
            "embodiment": self.embodiment,
            "initial_state": json_snapshot(self.initial_state),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the cacheable scene instance mapping."""
        result = self.identity_payload()
        result.update(
            {
                "metadata": json_snapshot(self.metadata),
                "content_hash": self.content_hash,
            }
        )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | SceneInstance) -> SceneInstance:
        """Decode and verify one scene instance."""
        if isinstance(value, cls):
            return value
        raw = _document_mapping(
            value,
            "SceneInstance",
            {
                "schema_version",
                "scene_id",
                "task_template_hash",
                "task_hash",
                "assets",
                "role_bindings",
                "embodiment",
                "initial_state",
                "metadata",
                "content_hash",
            },
        )
        item = cls(
            scene_id=raw.get("scene_id"),
            task_template_hash=raw.get("task_template_hash", raw.get("task_hash")),
            assets=tuple(raw.get("assets", ())),
            role_bindings=raw.get("role_bindings", {}),
            embodiment=raw.get("embodiment"),
            initial_state=raw.get("initial_state", {}),
            metadata=raw.get("metadata", {}),
            schema_version=raw.get("schema_version", SCENE_INSTANCE_SCHEMA),
        )
        if (
            raw.get("content_hash") is not None
            and raw["content_hash"] != item.content_hash
        ):
            raise ValueError(
                "SceneInstance.content_hash does not match its identity payload."
            )
        return item


@dataclass(frozen=True, slots=True)
class ActionWitness:
    """One concrete program/trajectory attempt for a TaskTemplate and scene."""

    witness_id: str
    task_template_hash: str
    scene_instance_hash: str
    program: Mapping[str, Any] = field(default_factory=dict)
    integration: Mapping[str, Any] = field(default_factory=dict)
    trajectory: Mapping[str, Any] | None = None
    execution_status: str = "not_run"
    program_completed: bool = False
    goal_status: str = EvaluationStatus.NOT_RUN.value
    evidence_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTION_WITNESS_SCHEMA
    content_hash: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        witness_id = _text(self.witness_id, "ActionWitness.witness_id")
        task_hash = _hash(self.task_template_hash, "ActionWitness.task_template_hash")
        scene_hash = _hash(
            self.scene_instance_hash, "ActionWitness.scene_instance_hash"
        )
        if self.schema_version != ACTION_WITNESS_SCHEMA:
            raise ValueError(
                f"ActionWitness.schema_version must be {ACTION_WITNESS_SCHEMA!r}."
            )
        execution_status = _text(
            self.execution_status, "ActionWitness.execution_status"
        )
        if execution_status not in _WITNESS_STATUSES:
            raise ValueError(
                f"ActionWitness.execution_status must be one of {sorted(_WITNESS_STATUSES)}."
            )
        goal_status = _text(self.goal_status, "ActionWitness.goal_status")
        if goal_status not in EVALUATION_STATUSES:
            raise ValueError(
                f"ActionWitness.goal_status must be one of {sorted(EVALUATION_STATUSES)}."
            )
        evidence_refs = _strings(self.evidence_refs, "ActionWitness.evidence_refs")
        program = _mapping(self.program, "ActionWitness.program")
        integration = _mapping(self.integration, "ActionWitness.integration")
        trajectory = (
            None
            if self.trajectory is None
            else _mapping(self.trajectory, "ActionWitness.trajectory")
        )
        metadata = _mapping(self.metadata, "ActionWitness.metadata")
        object.__setattr__(self, "witness_id", witness_id)
        object.__setattr__(self, "task_template_hash", task_hash)
        object.__setattr__(self, "scene_instance_hash", scene_hash)
        object.__setattr__(self, "program", program)
        object.__setattr__(self, "integration", integration)
        object.__setattr__(self, "trajectory", trajectory)
        object.__setattr__(self, "execution_status", execution_status)
        object.__setattr__(
            self,
            "program_completed",
            _bool(self.program_completed, "ActionWitness.program_completed"),
        )
        object.__setattr__(self, "goal_status", goal_status)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self, "content_hash", canonical_hash(self.identity_payload())
        )

    @property
    def task_hash(self) -> str:
        """Compatibility alias for the referenced template hash."""
        return self.task_template_hash

    def identity_payload(self) -> dict[str, Any]:
        """Return witness identity and outcome fields excluding metadata."""
        return {
            "schema_version": self.schema_version,
            "witness_id": self.witness_id,
            "task_template_hash": self.task_template_hash,
            "scene_instance_hash": self.scene_instance_hash,
            "program": json_snapshot(self.program),
            "integration": json_snapshot(self.integration),
            "trajectory": json_snapshot(self.trajectory),
            "execution_status": self.execution_status,
            "program_completed": self.program_completed,
            "goal_status": self.goal_status,
            "evidence_refs": list(self.evidence_refs),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe witness mapping."""
        result = self.identity_payload()
        result.update(
            {
                "metadata": json_snapshot(self.metadata),
                "content_hash": self.content_hash,
            }
        )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | ActionWitness) -> ActionWitness:
        """Decode and verify one witness."""
        if isinstance(value, cls):
            return value
        raw = _document_mapping(
            value,
            "ActionWitness",
            {
                "schema_version",
                "witness_id",
                "task_template_hash",
                "task_hash",
                "scene_instance_hash",
                "program",
                "integration",
                "trajectory",
                "execution_status",
                "program_completed",
                "goal_status",
                "evidence_refs",
                "metadata",
                "content_hash",
            },
        )
        item = cls(
            witness_id=raw.get("witness_id"),
            task_template_hash=raw.get("task_template_hash", raw.get("task_hash")),
            scene_instance_hash=raw.get("scene_instance_hash"),
            program=raw.get("program", {}),
            integration=raw.get("integration", {}),
            trajectory=raw.get("trajectory"),
            execution_status=raw.get("execution_status", "not_run"),
            program_completed=raw.get("program_completed", False),
            goal_status=raw.get("goal_status", EvaluationStatus.NOT_RUN.value),
            evidence_refs=raw.get("evidence_refs", ()),
            metadata=raw.get("metadata", {}),
            schema_version=raw.get("schema_version", ACTION_WITNESS_SCHEMA),
        )
        if (
            raw.get("content_hash") is not None
            and raw["content_hash"] != item.content_hash
        ):
            raise ValueError(
                "ActionWitness.content_hash does not match its identity payload."
            )
        return item


@dataclass(frozen=True, slots=True)
class ExpansionManifest:
    """Lineage and invalidation scope for trajectory/data expansion."""

    expansion_id: str
    task_template_hash: str
    parent_witness_id: str | None
    scope: Mapping[str, Any]
    operator: str
    seed: int
    child_witness_ids: tuple[str, ...] = ()
    invalidation_checks: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EXPANSION_MANIFEST_SCHEMA
    content_hash: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        expansion_id = _text(self.expansion_id, "ExpansionManifest.expansion_id")
        task_hash = _hash(
            self.task_template_hash, "ExpansionManifest.task_template_hash"
        )
        parent = _optional_text(
            self.parent_witness_id, "ExpansionManifest.parent_witness_id"
        )
        operator = _text(self.operator, "ExpansionManifest.operator")
        seed = _integer(self.seed, "ExpansionManifest.seed", minimum=0)
        if self.schema_version != EXPANSION_MANIFEST_SCHEMA:
            raise ValueError(
                f"ExpansionManifest.schema_version must be {EXPANSION_MANIFEST_SCHEMA!r}."
            )
        scope = _mapping(self.scope, "ExpansionManifest.scope")
        child_ids = _strings(
            self.child_witness_ids, "ExpansionManifest.child_witness_ids"
        )
        checks = tuple(
            _mapping(item, "ExpansionManifest.invalidation_checks[]")
            for item in self.invalidation_checks
        )
        metadata = _mapping(self.metadata, "ExpansionManifest.metadata")
        object.__setattr__(self, "expansion_id", expansion_id)
        object.__setattr__(self, "task_template_hash", task_hash)
        object.__setattr__(self, "parent_witness_id", parent)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "child_witness_ids", child_ids)
        object.__setattr__(self, "invalidation_checks", checks)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self, "content_hash", canonical_hash(self.identity_payload())
        )

    @property
    def task_hash(self) -> str:
        """Compatibility alias for the referenced template hash."""
        return self.task_template_hash

    def identity_payload(self) -> dict[str, Any]:
        """Return lineage fields excluding audit metadata and derived hash."""
        return {
            "schema_version": self.schema_version,
            "expansion_id": self.expansion_id,
            "task_template_hash": self.task_template_hash,
            "parent_witness_id": self.parent_witness_id,
            "scope": json_snapshot(self.scope),
            "operator": self.operator,
            "seed": self.seed,
            "child_witness_ids": list(self.child_witness_ids),
            "invalidation_checks": [
                json_snapshot(item) for item in self.invalidation_checks
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe expansion manifest."""
        result = self.identity_payload()
        result.update(
            {
                "metadata": json_snapshot(self.metadata),
                "content_hash": self.content_hash,
            }
        )
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any] | ExpansionManifest
    ) -> ExpansionManifest:
        """Decode and verify one expansion manifest."""
        if isinstance(value, cls):
            return value
        raw = _document_mapping(
            value,
            "ExpansionManifest",
            {
                "schema_version",
                "expansion_id",
                "task_template_hash",
                "task_hash",
                "parent_witness_id",
                "scope",
                "operator",
                "seed",
                "child_witness_ids",
                "invalidation_checks",
                "metadata",
                "content_hash",
            },
        )
        item = cls(
            expansion_id=raw.get("expansion_id"),
            task_template_hash=raw.get("task_template_hash", raw.get("task_hash")),
            parent_witness_id=raw.get("parent_witness_id"),
            scope=raw.get("scope", {}),
            operator=raw.get("operator"),
            seed=raw.get("seed"),
            child_witness_ids=raw.get("child_witness_ids", ()),
            invalidation_checks=raw.get("invalidation_checks", ()),
            metadata=raw.get("metadata", {}),
            schema_version=raw.get("schema_version", EXPANSION_MANIFEST_SCHEMA),
        )
        if (
            raw.get("content_hash") is not None
            and raw["content_hash"] != item.content_hash
        ):
            raise ValueError(
                "ExpansionManifest.content_hash does not match its identity payload."
            )
        return item


@dataclass(frozen=True, slots=True)
class ValidationCertificate:
    """Versioned evidence summary for one TaskSpec evaluation."""

    certificate_id: str
    task_template_hash: str
    scene_instance_hash: str
    witness_id: str | None
    checker: str
    status: str
    predicate_results: tuple[PredicateEvaluation | Mapping[str, Any], ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    checker_version: str = "task_spec/v0.1"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = VALIDATION_CERTIFICATE_SCHEMA
    content_hash: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        certificate_id = _text(
            self.certificate_id, "ValidationCertificate.certificate_id"
        )
        task_hash = _hash(
            self.task_template_hash, "ValidationCertificate.task_template_hash"
        )
        scene_hash = _hash(
            self.scene_instance_hash, "ValidationCertificate.scene_instance_hash"
        )
        witness_id = _optional_text(self.witness_id, "ValidationCertificate.witness_id")
        checker = _text(self.checker, "ValidationCertificate.checker")
        status = _text(self.status, "ValidationCertificate.status")
        if status not in EVALUATION_STATUSES:
            raise ValueError(
                f"ValidationCertificate.status must be one of {sorted(EVALUATION_STATUSES)}."
            )
        checker_version = _text(
            self.checker_version, "ValidationCertificate.checker_version"
        )
        if self.schema_version != VALIDATION_CERTIFICATE_SCHEMA:
            raise ValueError(
                f"ValidationCertificate.schema_version must be {VALIDATION_CERTIFICATE_SCHEMA!r}."
            )
        results = tuple(_evaluation(item) for item in self.predicate_results)
        if results:
            expected_status = _aggregate_evaluation_status(results)
            if status != expected_status:
                raise ValueError(
                    "ValidationCertificate.status must match the aggregate "
                    f"predicate status {expected_status!r}."
                )
        metrics = _mapping(self.metrics, "ValidationCertificate.metrics")
        evidence_refs = _strings(
            self.evidence_refs, "ValidationCertificate.evidence_refs"
        )
        metadata = _mapping(self.metadata, "ValidationCertificate.metadata")
        object.__setattr__(self, "certificate_id", certificate_id)
        object.__setattr__(self, "task_template_hash", task_hash)
        object.__setattr__(self, "scene_instance_hash", scene_hash)
        object.__setattr__(self, "witness_id", witness_id)
        object.__setattr__(self, "checker", checker)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "checker_version", checker_version)
        object.__setattr__(self, "predicate_results", results)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self, "content_hash", canonical_hash(self.identity_payload())
        )

    @property
    def task_hash(self) -> str:
        """Compatibility alias for the referenced template hash."""
        return self.task_template_hash

    def identity_payload(self) -> dict[str, Any]:
        """Return certificate fields excluding metadata and derived hash."""
        return {
            "schema_version": self.schema_version,
            "certificate_id": self.certificate_id,
            "task_template_hash": self.task_template_hash,
            "scene_instance_hash": self.scene_instance_hash,
            "witness_id": self.witness_id,
            "checker": self.checker,
            "status": self.status,
            "predicate_results": [item.to_dict() for item in self.predicate_results],
            "metrics": json_snapshot(self.metrics),
            "evidence_refs": list(self.evidence_refs),
            "checker_version": self.checker_version,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe certificate."""
        result = self.identity_payload()
        result.update(
            {
                "metadata": json_snapshot(self.metadata),
                "content_hash": self.content_hash,
            }
        )
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any] | ValidationCertificate
    ) -> ValidationCertificate:
        """Decode and verify one validation certificate."""
        if isinstance(value, cls):
            return value
        raw = _document_mapping(
            value,
            "ValidationCertificate",
            {
                "schema_version",
                "certificate_id",
                "task_template_hash",
                "task_hash",
                "scene_instance_hash",
                "witness_id",
                "checker",
                "status",
                "predicate_results",
                "metrics",
                "evidence_refs",
                "checker_version",
                "metadata",
                "content_hash",
            },
        )
        item = cls(
            certificate_id=raw.get("certificate_id"),
            task_template_hash=raw.get("task_template_hash", raw.get("task_hash")),
            scene_instance_hash=raw.get("scene_instance_hash"),
            witness_id=raw.get("witness_id"),
            checker=raw.get("checker"),
            status=raw.get("status"),
            predicate_results=tuple(raw.get("predicate_results", ())),
            metrics=raw.get("metrics", {}),
            evidence_refs=raw.get("evidence_refs", ()),
            checker_version=raw.get("checker_version", "task_spec/v0.1"),
            metadata=raw.get("metadata", {}),
            schema_version=raw.get("schema_version", VALIDATION_CERTIFICATE_SCHEMA),
        )
        if (
            raw.get("content_hash") is not None
            and raw["content_hash"] != item.content_hash
        ):
            raise ValueError(
                "ValidationCertificate.content_hash does not match its identity payload."
            )
        return item


def _sort_temporal(
    values: Sequence[TemporalCondition],
) -> tuple[TemporalCondition, ...]:
    """Normalize independent temporal declarations into stable order."""
    ordered = sorted(
        values,
        key=lambda item: canonical_json(
            {
                "predicate": {
                    "predicate": item.predicate.name,
                    "arguments": item.predicate.arguments,
                    "operator": item.predicate.operator,
                    "expected": item.predicate.expected,
                    "observation": item.predicate.observation,
                },
                "within_steps": item.within_steps,
                "start_step": item.start_step,
                "mode": item.mode,
            }
        ),
    )
    used = {item.condition_id for item in ordered if item.condition_id is not None}
    result: list[TemporalCondition] = []
    generated = 1
    for item in ordered:
        condition_id = item.condition_id
        if condition_id is None:
            while f"temporal_{generated:03d}" in used:
                generated += 1
            condition_id = f"temporal_{generated:03d}"
            generated += 1
            used.add(condition_id)
        result.append(item.with_id(condition_id))
    if len(used) != len(result):
        raise ValueError("TaskTemplate.temporal must have unique IDs.")
    return tuple(result)


def _sort_predicates(values: Sequence[Predicate], prefix: str) -> tuple[Predicate, ...]:
    # Sort by semantic body, then assign only missing IDs. Explicit IDs remain
    # available to downstream evidence references.
    ordered = sorted(
        values,
        key=lambda item: canonical_json(
            {
                "predicate": item.name,
                "arguments": item.arguments,
                "operator": item.operator,
                "expected": item.expected,
                "observation": item.observation,
            }
        ),
    )
    used = {item.predicate_id for item in ordered if item.predicate_id is not None}
    result: list[Predicate] = []
    generated = 1
    for item in ordered:
        predicate_id = item.predicate_id
        if predicate_id is None:
            while f"{prefix}_{generated:03d}" in used:
                generated += 1
            predicate_id = f"{prefix}_{generated:03d}"
            generated += 1
            used.add(predicate_id)
        result.append(item.with_id(predicate_id))
    if len(used) != len(result):
        raise ValueError(f"TaskTemplate.{prefix} predicates must have unique IDs.")
    return tuple(result)


def _evaluation(value: PredicateEvaluation | Mapping[str, Any]) -> PredicateEvaluation:
    if isinstance(value, PredicateEvaluation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("ValidationCertificate.predicate_results must be mappings.")
    return PredicateEvaluation(
        predicate_id=value.get("predicate_id"),
        predicate=value.get("predicate"),
        status=value.get("status"),
        observed=value.get("observed"),
        expected=value.get("expected"),
        reason=value.get("reason"),
        evidence_refs=value.get("evidence_refs", ()),
    )


def _aggregate_evaluation_status(
    results: Sequence[PredicateEvaluation],
) -> str:
    statuses = {item.status for item in results}
    if EvaluationStatus.FAILED in statuses:
        return EvaluationStatus.FAILED.value
    if EvaluationStatus.UNAVAILABLE in statuses:
        return EvaluationStatus.UNAVAILABLE.value
    if EvaluationStatus.UNKNOWN in statuses:
        return EvaluationStatus.UNKNOWN.value
    if EvaluationStatus.NOT_RUN in statuses:
        return EvaluationStatus.NOT_RUN.value
    return EvaluationStatus.SATISFIED.value


def _document_mapping(
    value: Any, context: str, allowed: set[str] | frozenset[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    result = dict(value)
    unknown = set(result) - set(allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields {sorted(unknown)}.")
    return result


def validate_scene_instance(value: SceneInstance | Mapping[str, Any]) -> dict[str, Any]:
    """Validate one legacy grounded SceneInstance document."""
    try:
        return SceneInstance.from_dict(value).to_dict()
    except (TypeError, ValueError) as exc:
        from .validation import TaskSpecValidationError

        raise TaskSpecValidationError(str(exc)) from exc


def validate_action_witness(value: ActionWitness | Mapping[str, Any]) -> dict[str, Any]:
    """Validate one legacy ActionWitness document."""
    try:
        return ActionWitness.from_dict(value).to_dict()
    except (TypeError, ValueError) as exc:
        from .validation import TaskSpecValidationError

        raise TaskSpecValidationError(str(exc)) from exc


def validate_expansion_manifest(
    value: ExpansionManifest | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one legacy ExpansionManifest document."""
    try:
        return ExpansionManifest.from_dict(value).to_dict()
    except (TypeError, ValueError) as exc:
        from .validation import TaskSpecValidationError

        raise TaskSpecValidationError(str(exc)) from exc


def validate_validation_certificate(
    value: ValidationCertificate | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one legacy ValidationCertificate document."""
    try:
        return ValidationCertificate.from_dict(value).to_dict()
    except (TypeError, ValueError) as exc:
        from .validation import TaskSpecValidationError

        raise TaskSpecValidationError(str(exc)) from exc


def build_validation_certificate(
    report: Any,
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
    """Build one legacy certificate from a core evaluation report."""
    results = (
        *report.init,
        *report.goal,
        *report.invariants,
        *report.temporal,
        *getattr(report, "milestones", ()),
    )
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


__all__ += [
    "build_validation_certificate",
    "validate_action_witness",
    "validate_expansion_manifest",
    "validate_scene_instance",
    "validate_validation_certificate",
]
