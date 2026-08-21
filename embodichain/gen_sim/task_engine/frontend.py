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

"""Strict task interpretation and canonical provider-free scene binding."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType

from embodichain.gen_sim.task_engine.contracts import (
    FORBIDDEN_SEMANTIC_GRAPH_FIELDS,
    TaskSpec,
    decode_task_spec,
)
from embodichain.lab.gym.envs.expert_program.cfg import (
    HandOverCfg,
    OperateArticulationCfg,
    PickCfg,
    PlaceCfg,
    RegisteredSemanticCallCfg,
    SemanticCallCfg,
)
from embodichain.lab.gym.envs.expert_program.decoder import (
    decode_semantic_call,
    encode_semantic_call,
)
from embodichain.lab.sim.skills import (
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRef,
    SceneLinkRef,
    SceneManifest,
    SceneObjectRef,
)

__all__ = [
    "SCENE_REQUIREMENTS_SCHEMA",
    "TASK_DRAFT_OUTPUT_SCHEMA",
    "TASK_DRAFT_SCHEMA",
    "BoundTaskDraft",
    "RoleRequirement",
    "SceneRequirements",
    "SemanticCallCandidate",
    "TaskDraft",
    "TaskDraftCaller",
    "TaskInterpretationError",
    "TaskInterpretationResult",
    "bind_task_draft",
    "decode_scene_requirements",
    "decode_task_draft",
    "interpret_task_candidates",
    "validate_planner_projection",
]

SCENE_REQUIREMENTS_SCHEMA = "scene_requirements/v1"
TASK_DRAFT_SCHEMA = "task_draft/v1"
_PLANNER_PROJECTION_SCHEMA = "semantic_integration_planner_projection/v1"
_ROLE_TYPES = frozenset({"object", "articulation", "link", "affordance"})
_MAX_CANDIDATES = 16

TaskDraftCaller = Callable[..., object]

TASK_DRAFT_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "candidate_id",
        "integration_fingerprint",
        "task_spec",
        "scene_requirements",
        "semantic_call_candidates",
        "model_provenance",
    ],
    "properties": {
        "schema_version": {"const": TASK_DRAFT_SCHEMA},
        "candidate_id": {"type": "string"},
        "integration_fingerprint": {"type": "string"},
        "task_spec": {"type": "object"},
        "scene_requirements": {"type": "object"},
        "semantic_call_candidates": {"type": "array", "minItems": 1},
        "model_provenance": {"type": "object"},
    },
}


def _identifier(value: object, path: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{path} must be a non-empty, trimmed string.")
    return value


def _owned_json(value: object, path: str) -> object:
    return _clone_json(value, path=path, active=set(), depth=0)


def _clone_json(
    value: object,
    *,
    path: str,
    active: set[int],
    depth: int,
) -> object:
    if depth > 32:
        raise ValueError(f"{path} exceeds the maximum JSON depth.")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a cycle.")
        active.add(identity)
        result: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str or not key or key != key.strip():
                raise ValueError(f"{path} keys must be non-empty, trimmed strings.")
            result[key] = _clone_json(
                nested,
                path=f"{path}.{key}",
                active=active,
                depth=depth + 1,
            )
        active.remove(identity)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a cycle.")
        active.add(identity)
        result = [
            _clone_json(
                nested,
                path=f"{path}[{index}]",
                active=active,
                depth=depth + 1,
            )
            for index, nested in enumerate(value)
        ]
        active.remove(identity)
        return result
    raise TypeError(f"{path} contains non-JSON type {type(value).__name__}.")


def _mapping(value: object, path: str) -> dict[str, object]:
    result = _owned_json(value, path)
    if type(result) is not dict:
        raise TypeError(f"{path} must be a JSON object.")
    if any(type(key) is not str or not key or key != key.strip() for key in result):
        raise ValueError(f"{path} keys must be non-empty, trimmed strings.")
    return result


def _sequence(value: object, path: str) -> list[object]:
    result = _owned_json(value, path)
    if type(result) is not list:
        raise TypeError(f"{path} must be a JSON array.")
    return result


def _keys(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    path: str,
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}.")
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {unknown}.")


def _freeze_mapping(value: object, path: str) -> Mapping[str, object]:
    return MappingProxyType(_mapping(value, path))


def _fingerprint(value: object, path: str) -> str:
    result = _identifier(value, path)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest.")
    return result


def _reject_forbidden(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_SEMANTIC_GRAPH_FIELDS:
                raise ValueError(f"{path}.{key} is forbidden in task interpretation.")
            _reject_forbidden(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_forbidden(nested, f"{path}[{index}]")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_planner_projection(value: object) -> dict[str, object]:
    """Validate and own the canonical catalog planner projection.

    Args:
        value: Projection returned by ``ExpertProgramIntegrationCatalog``.

    Returns:
        Detached JSON-safe planner projection.
    """
    result = _mapping(value, "PlannerProjection")
    if result.get("schema_version") != _PLANNER_PROJECTION_SCHEMA:
        raise ValueError("PlannerProjection.schema_version is unsupported.")
    _fingerprint(
        result.get("integration_fingerprint"),
        "PlannerProjection.integration_fingerprint",
    )
    calls = _sequence(result.get("semantic_calls"), "PlannerProjection.semantic_calls")
    call_ids: list[str] = []
    for index, raw_call in enumerate(calls):
        call = _mapping(raw_call, f"PlannerProjection.semantic_calls[{index}]")
        call_ids.append(
            _identifier(
                call.get("call_id"),
                f"PlannerProjection.semantic_calls[{index}].call_id",
            )
        )
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("PlannerProjection semantic call IDs must be unique.")
    return result


@dataclass(frozen=True, slots=True)
class RoleRequirement:
    """One language role that must resolve to a canonical typed scene ref."""

    role_id: str
    reference: str
    expected_type: str
    capability: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.role_id, "RoleRequirement.role_id")
        _identifier(self.reference, "RoleRequirement.reference")
        if self.expected_type not in _ROLE_TYPES:
            raise ValueError(f"expected_type must be one of {sorted(_ROLE_TYPES)}.")
        if self.capability is not None:
            _identifier(self.capability, "RoleRequirement.capability")
            if self.expected_type != "affordance":
                raise ValueError("capability is valid only for affordance roles.")

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe role requirement."""
        result: dict[str, object] = {
            "role_id": self.role_id,
            "reference": self.reference,
            "expected_type": self.expected_type,
        }
        if self.capability is not None:
            result["capability"] = self.capability
        return result


@dataclass(frozen=True, slots=True)
class SceneRequirements:
    """Provider-free typed scene identities required by one TaskSpec."""

    task_id: str
    roles: tuple[RoleRequirement, ...]
    schema_version: str = SCENE_REQUIREMENTS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCENE_REQUIREMENTS_SCHEMA:
            raise ValueError("SceneRequirements.schema_version is unsupported.")
        _identifier(self.task_id, "SceneRequirements.task_id")
        if not self.roles or not all(
            type(role) is RoleRequirement for role in self.roles
        ):
            raise ValueError("SceneRequirements.roles must contain typed roles.")
        role_ids = [role.role_id for role in self.roles]
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("SceneRequirements role IDs must be unique.")

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe scene requirement mapping."""
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "roles": [role.to_dict() for role in self.roles],
        }


def decode_scene_requirements(value: object) -> SceneRequirements:
    """Decode strict provider-free scene requirements."""
    result = _mapping(value, "SceneRequirements")
    _keys(
        result,
        required=frozenset({"schema_version", "task_id", "roles"}),
        path="SceneRequirements",
    )
    roles: list[RoleRequirement] = []
    for index, raw_role in enumerate(
        _sequence(result["roles"], "SceneRequirements.roles")
    ):
        path = f"SceneRequirements.roles[{index}]"
        role = _mapping(raw_role, path)
        _keys(
            role,
            required=frozenset({"role_id", "reference", "expected_type"}),
            optional=frozenset({"capability"}),
            path=path,
        )
        roles.append(
            RoleRequirement(
                role_id=_identifier(role["role_id"], f"{path}.role_id"),
                reference=_identifier(role["reference"], f"{path}.reference"),
                expected_type=_identifier(
                    role["expected_type"], f"{path}.expected_type"
                ),
                capability=(
                    None
                    if role.get("capability") is None
                    else _identifier(role["capability"], f"{path}.capability")
                ),
            )
        )
    return SceneRequirements(
        task_id=_identifier(result["task_id"], "SceneRequirements.task_id"),
        roles=tuple(roles),
        schema_version=_identifier(
            result["schema_version"], "SceneRequirements.schema_version"
        ),
    )


@dataclass(frozen=True, slots=True)
class SemanticCallCandidate:
    """One candidate semantic call sequence for a TaskSpec instance."""

    task_instance_id: str
    _call_payloads: tuple[Mapping[str, object], ...]
    confidence: float

    def __post_init__(self) -> None:
        _identifier(self.task_instance_id, "SemanticCallCandidate.task_instance_id")
        if not self._call_payloads:
            raise ValueError("SemanticCallCandidate requires at least one call.")
        object.__setattr__(
            self,
            "_call_payloads",
            tuple(
                _freeze_mapping(payload, "SemanticCallCandidate.call")
                for payload in self._call_payloads
            ),
        )
        if (
            type(self.confidence) not in (int, float)
            or isinstance(self.confidence, bool)
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("SemanticCallCandidate.confidence must be in [0, 1].")

    @property
    def calls(self) -> tuple[SemanticCallCfg, ...]:
        """Return fresh canonical semantic-call configs."""
        return tuple(
            decode_semantic_call(_mapping(payload, "SemanticCallCandidate.call"))
            for payload in self._call_payloads
        )

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe call candidate."""
        return {
            "task_instance_id": self.task_instance_id,
            "calls": [
                _mapping(payload, "SemanticCallCandidate.call")
                for payload in self._call_payloads
            ],
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True, slots=True)
class TaskDraft:
    """One locally validated interpretation candidate before scene binding."""

    candidate_id: str
    integration_fingerprint: str
    task_spec: TaskSpec
    scene_requirements: SceneRequirements
    semantic_call_candidates: tuple[SemanticCallCandidate, ...]
    model_provenance: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schema_version: str = TASK_DRAFT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TASK_DRAFT_SCHEMA:
            raise ValueError("TaskDraft.schema_version is unsupported.")
        _identifier(self.candidate_id, "TaskDraft.candidate_id")
        _fingerprint(
            self.integration_fingerprint,
            "TaskDraft.integration_fingerprint",
        )
        if type(self.task_spec) is not TaskSpec:
            raise TypeError("TaskDraft.task_spec must be exactly TaskSpec.")
        if type(self.scene_requirements) is not SceneRequirements:
            raise TypeError(
                "TaskDraft.scene_requirements must be exactly SceneRequirements."
            )
        if self.task_spec.task_id != self.scene_requirements.task_id:
            raise ValueError("TaskDraft task IDs must agree across contracts.")
        instance_ids = {instance.id for instance in self.task_spec.task_instances}
        candidate_ids = [
            candidate.task_instance_id for candidate in self.semantic_call_candidates
        ]
        if set(candidate_ids) != instance_ids or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValueError(
                "TaskDraft must contain one semantic call candidate per task instance."
            )
        object.__setattr__(
            self,
            "model_provenance",
            _freeze_mapping(self.model_provenance, "TaskDraft.model_provenance"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe task draft."""
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "integration_fingerprint": self.integration_fingerprint,
            "task_spec": self.task_spec.to_dict(),
            "scene_requirements": self.scene_requirements.to_dict(),
            "semantic_call_candidates": [
                candidate.to_dict() for candidate in self.semantic_call_candidates
            ],
            "model_provenance": _mapping(
                self.model_provenance, "TaskDraft.model_provenance"
            ),
        }


def _call_id(call: SemanticCallCfg) -> str:
    return call.call_id if type(call) is RegisteredSemanticCallCfg else call.kind


def decode_task_draft(value: object, *, planner_projection: object) -> TaskDraft:
    """Decode one model or deterministic draft through the same local contracts."""
    _reject_forbidden(value, "TaskDraft")
    projection = validate_planner_projection(planner_projection)
    result = _mapping(value, "TaskDraft")
    _keys(
        result,
        required=frozenset(
            {
                "schema_version",
                "candidate_id",
                "integration_fingerprint",
                "task_spec",
                "scene_requirements",
                "semantic_call_candidates",
                "model_provenance",
            }
        ),
        path="TaskDraft",
    )
    fingerprint = _fingerprint(
        result["integration_fingerprint"], "TaskDraft.integration_fingerprint"
    )
    if fingerprint != projection["integration_fingerprint"]:
        raise ValueError("TaskDraft integration fingerprint does not match catalog.")
    known_call_ids = {
        str(item["call_id"])
        for item in projection["semantic_calls"]  # type: ignore[union-attr]
    }
    candidates: list[SemanticCallCandidate] = []
    for index, raw_candidate in enumerate(
        _sequence(
            result["semantic_call_candidates"],
            "TaskDraft.semantic_call_candidates",
        )
    ):
        path = f"TaskDraft.semantic_call_candidates[{index}]"
        candidate = _mapping(raw_candidate, path)
        _keys(
            candidate,
            required=frozenset({"task_instance_id", "calls", "confidence"}),
            path=path,
        )
        payloads: list[dict[str, object]] = []
        for call_index, raw_call in enumerate(
            _sequence(candidate["calls"], f"{path}.calls")
        ):
            call = decode_semantic_call(
                raw_call,
                path=("semantic_call_candidates", index, "calls", call_index),
            )
            if _call_id(call) not in known_call_ids:
                raise ValueError(
                    f"{path}.calls[{call_index}] is unavailable in the canonical catalog."
                )
            payloads.append(encode_semantic_call(call))
        candidates.append(
            SemanticCallCandidate(
                task_instance_id=_identifier(
                    candidate["task_instance_id"], f"{path}.task_instance_id"
                ),
                _call_payloads=tuple(payloads),
                confidence=candidate["confidence"],  # type: ignore[arg-type]
            )
        )
    task_spec = decode_task_spec(result["task_spec"])
    requirements = decode_scene_requirements(result["scene_requirements"])
    return TaskDraft(
        candidate_id=_identifier(result["candidate_id"], "TaskDraft.candidate_id"),
        integration_fingerprint=fingerprint,
        task_spec=task_spec,
        scene_requirements=requirements,
        semantic_call_candidates=tuple(candidates),
        model_provenance=_mapping(
            result["model_provenance"], "TaskDraft.model_provenance"
        ),
    )


@dataclass(frozen=True, slots=True)
class BoundTaskDraft:
    """Task interpretation bound only to canonical provider-free scene refs."""

    draft: TaskDraft
    role_bindings: Mapping[str, SceneEntityRef]
    semantic_call_candidates: tuple[SemanticCallCandidate, ...]

    def __post_init__(self) -> None:
        if type(self.draft) is not TaskDraft:
            raise TypeError("BoundTaskDraft.draft must be exactly TaskDraft.")
        bindings = dict(self.role_bindings)
        if not all(type(value) in _REF_TYPES.values() for value in bindings.values()):
            raise TypeError(
                "BoundTaskDraft bindings must contain canonical scene refs."
            )
        object.__setattr__(self, "role_bindings", MappingProxyType(bindings))


_REF_TYPES: Mapping[str, type[SceneEntityRef]] = MappingProxyType(
    {
        "object": SceneObjectRef,
        "articulation": SceneArticulationRef,
        "link": SceneLinkRef,
        "affordance": SceneAffordanceRef,
    }
)


def _replace_roles(value: object, bindings: Mapping[str, SceneEntityRef]) -> object:
    if type(value) is str and value in bindings:
        return bindings[value].entity_id
    if isinstance(value, Mapping):
        return {
            str(key): _replace_roles(nested, bindings) for key, nested in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_replace_roles(nested, bindings) for nested in value]
    return value


def _validate_call_scene_refs(call: SemanticCallCfg, scene: SceneManifest) -> None:
    if type(call) in {PickCfg, PlaceCfg, HandOverCfg}:
        scene.resolve(call.object, expected_type=SceneObjectRef)
    if type(call) is PickCfg and call.grasp is not None:
        scene.resolve(call.grasp, expected_type=SceneAffordanceRef)
    if type(call) is PlaceCfg:
        for reference in (call.on, call.inside):
            if reference is not None:
                scene.resolve(reference)
    if type(call) is OperateArticulationCfg:
        scene.resolve(call.articulation, expected_type=SceneArticulationRef)
        if call.handle is not None:
            scene.resolve(call.handle, expected_type=SceneAffordanceRef)


def bind_task_draft(draft: TaskDraft, scene: SceneManifest) -> BoundTaskDraft:
    """Resolve language roles once through the canonical SceneManifest.

    Args:
        draft: Locally validated task interpretation candidate.
        scene: Canonical provider-free scene manifest.

    Returns:
        Draft with typed canonical role bindings and rebound semantic calls.

    Raises:
        SemanticValidationError: If a reference is unknown, ambiguous, or mistyped.
    """
    if type(draft) is not TaskDraft:
        raise TypeError("draft must be exactly TaskDraft.")
    if type(scene) is not SceneManifest:
        raise TypeError("scene must be exactly SceneManifest.")
    bindings: dict[str, SceneEntityRef] = {}
    for requirement in draft.scene_requirements.roles:
        ref = scene.resolve(
            requirement.reference,
            expected_type=_REF_TYPES[requirement.expected_type],
            path=("scene_requirements", requirement.role_id),
        )
        if requirement.capability is not None:
            entry = next(item for item in scene.entries if item.ref == ref)
            if requirement.capability not in entry.affordance_capabilities:
                raise ValueError(
                    f"Scene role {requirement.role_id!r} lacks affordance capability "
                    f"{requirement.capability!r}."
                )
        bindings[requirement.role_id] = ref

    rebound: list[SemanticCallCandidate] = []
    for candidate in draft.semantic_call_candidates:
        payloads: list[dict[str, object]] = []
        for payload in candidate._call_payloads:
            call = decode_semantic_call(_replace_roles(payload, bindings))
            _validate_call_scene_refs(call, scene)
            payloads.append(encode_semantic_call(call))
        rebound.append(
            SemanticCallCandidate(
                task_instance_id=candidate.task_instance_id,
                _call_payloads=tuple(payloads),
                confidence=candidate.confidence,
            )
        )
    return BoundTaskDraft(
        draft=draft,
        role_bindings=bindings,
        semantic_call_candidates=tuple(rebound),
    )


class TaskInterpretationError(ValueError):
    """Raised when every independent interpretation candidate is invalid."""


@dataclass(frozen=True, slots=True)
class TaskInterpretationResult:
    """Valid unique candidates plus candidate-local validation failures."""

    candidates: tuple[TaskDraft, ...]
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("TaskInterpretationResult requires a valid candidate.")


def interpret_task_candidates(
    instruction: str,
    *,
    caller: TaskDraftCaller,
    planner_projection: object,
    candidate_count: int = 3,
) -> TaskInterpretationResult:
    """Generate independent drafts through one strict local validation path.

    Args:
        instruction: Original task instruction.
        caller: Structured model or deterministic caller.
        planner_projection: Canonical semantic integration planner projection.
        candidate_count: Number of independent candidates to request.

    Returns:
        Deduplicated valid drafts and candidate-local errors.

    Raises:
        TaskInterpretationError: If every candidate fails local validation.
    """
    normalized_instruction = _identifier(instruction, "instruction")
    if not callable(caller):
        raise TypeError("caller must be callable.")
    if type(candidate_count) is not int or not 1 <= candidate_count <= _MAX_CANDIDATES:
        raise ValueError(f"candidate_count must be in [1, {_MAX_CANDIDATES}].")
    projection = validate_planner_projection(planner_projection)
    unique: dict[str, TaskDraft] = {}
    errors: list[str] = []
    for index in range(candidate_count):
        try:
            raw = caller(
                instruction=normalized_instruction,
                schema=deepcopy(TASK_DRAFT_OUTPUT_SCHEMA),
                planner_projection=deepcopy(projection),
                candidate_index=index,
            )
            draft = decode_task_draft(raw, planner_projection=projection)
            if draft.task_spec.instruction != normalized_instruction:
                raise ValueError("TaskDraft instruction does not match the request.")
            canonical = draft.to_dict()
            canonical.pop("candidate_id")
            unique.setdefault(_canonical_hash(canonical), draft)
        except Exception as exc:  # Candidate-local structured boundary.
            errors.append(f"candidate_{index + 1:02d}: {type(exc).__name__}: {exc}")
    if not unique:
        raise TaskInterpretationError(
            "All task interpretation candidates failed validation: " + "; ".join(errors)
        )
    return TaskInterpretationResult(
        candidates=tuple(unique.values()), errors=tuple(errors)
    )
