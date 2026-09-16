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

"""Strict, JSON-safe public contracts owned by Task Engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import Any, TypeAlias

from .interpretation import validate_instruction_intent
from .ontology import TASK_CONTRACTS, task_success_type

__all__ = [
    "SCENE_REQUEST_SCHEMA",
    "SUCCESS_SPEC_SCHEMA",
    "TASK_CANDIDATE_SET_SCHEMA",
    "TASK_DRAFT_SCHEMA",
    "SceneRequest",
    "SuccessSpec",
    "TaskCandidate",
    "TaskCandidateSet",
    "TaskDraft",
    "canonical_hash",
    "validate_scene_request",
    "validate_success_spec",
    "validate_task_candidate",
    "validate_task_candidate_set",
    "validate_task_draft",
]

TASK_DRAFT_SCHEMA = "action_engine_task_draft_v1"
SCENE_REQUEST_SCHEMA = "action_engine_scene_request_v1"
SUCCESS_SPEC_SCHEMA = "action_engine_success_spec_v1"
TASK_CANDIDATE_SET_SCHEMA = "action_engine_task_candidate_set_v1"

TaskDraft: TypeAlias = dict[str, Any]
SceneRequest: TypeAlias = dict[str, Any]
SuccessSpec: TypeAlias = dict[str, Any]
TaskCandidate: TypeAlias = dict[str, Any]
TaskCandidateSet: TypeAlias = dict[str, Any]

_SUCCESS_TYPES = frozenset(
    {contract.success_type for contract in TASK_CONTRACTS.values()} | {"semantic_goal"}
)
_DRAFT_KEYS = frozenset({"schema_version", "task_id", "instruction", "steps"})
_SCENE_REQUEST_KEYS = frozenset({"schema_version", "task_id", "references"})
_REFERENCE_KEYS = frozenset(
    {
        "reference_id",
        "step_id",
        "role",
        "reference",
        "quantifier",
        "count",
        "source_structure",
        "affordances",
        "initial_state",
        "attributes",
    }
)
_SUCCESS_KEYS = frozenset({"schema_version", "task_id", "op", "terms"})
_SUCCESS_TERM_KEYS = frozenset({"step_id", "type"})
_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "draft",
        "scene_request",
        "success_spec",
        "semantic_hash",
        "vote_count",
        "attempts",
        "normalizations",
    }
)
_CANDIDATE_SET_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "instruction",
        "candidates",
        "requested_candidate_count",
        "valid_response_count",
        "errors",
    }
)


def canonical_hash(value: Any) -> str:
    """Return the stable SHA-256 of one JSON-safe protocol value."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_task_draft(value: Mapping[str, Any]) -> TaskDraft:
    result = _mapping(value, "TaskDraft")
    _keys(result, _DRAFT_KEYS, "TaskDraft")
    _schema(result, TASK_DRAFT_SCHEMA, "TaskDraft")
    result["task_id"] = _nonempty(result.get("task_id"), "TaskDraft.task_id")
    result["instruction"] = _nonempty(
        result.get("instruction"), "TaskDraft.instruction"
    )
    intent = validate_instruction_intent({"steps": result.get("steps")})
    result["steps"] = intent["steps"]
    return result


def validate_scene_request(value: Mapping[str, Any]) -> SceneRequest:
    result = _mapping(value, "SceneRequest")
    _keys(result, _SCENE_REQUEST_KEYS, "SceneRequest")
    _schema(result, SCENE_REQUEST_SCHEMA, "SceneRequest")
    task_id = _nonempty(result.get("task_id"), "SceneRequest.task_id")
    references: list[dict[str, Any]] = []
    for index, raw in enumerate(
        _sequence(result.get("references"), "SceneRequest.references")
    ):
        context = f"SceneRequest.references[{index}]"
        reference = _mapping(raw, context)
        _keys(reference, _REFERENCE_KEYS, context)
        for key in ("reference_id", "step_id", "role", "reference", "source_structure"):
            reference[key] = _nonempty(reference.get(key), f"{context}.{key}")
        reference["role"] = _enum(
            reference["role"], {"object", "target"}, f"{context}.role"
        )
        reference["quantifier"] = _enum(
            reference.get("quantifier"),
            {"one", "all", "count"},
            f"{context}.quantifier",
        )
        reference["count"] = _integer(
            reference.get("count"), f"{context}.count", minimum=0
        )
        if reference["quantifier"] in {"one", "all"} and reference["count"] != 0:
            raise ValueError(
                f"{context} quantifier={reference['quantifier']} requires count=0."
            )
        if reference["quantifier"] == "count" and reference["count"] < 1:
            raise ValueError(f"{context} quantifier=count requires count>=1.")
        reference["affordances"] = _strings(
            reference.get("affordances"), f"{context}.affordances"
        )
        reference["initial_state"] = _mapping(
            reference.get("initial_state"), f"{context}.initial_state"
        )
        reference["attributes"] = _mapping(
            reference.get("attributes"), f"{context}.attributes"
        )
        references.append(reference)
    _unique([item["reference_id"] for item in references], "SceneRequest reference IDs")
    result["task_id"] = task_id
    result["references"] = references
    _json_safe(result, "SceneRequest")
    return result


def validate_success_spec(
    value: Mapping[str, Any],
    *,
    draft: Mapping[str, Any] | None = None,
) -> SuccessSpec:
    result = _mapping(value, "SuccessSpec")
    _keys(result, _SUCCESS_KEYS, "SuccessSpec")
    _schema(result, SUCCESS_SPEC_SCHEMA, "SuccessSpec")
    task_id = _nonempty(result.get("task_id"), "SuccessSpec.task_id")
    if result.get("op") != "all":
        raise ValueError("SuccessSpec.op must be 'all'.")
    terms: list[dict[str, str]] = []
    for index, raw in enumerate(_sequence(result.get("terms"), "SuccessSpec.terms")):
        context = f"SuccessSpec.terms[{index}]"
        term = _mapping(raw, context)
        _keys(term, _SUCCESS_TERM_KEYS, context)
        terms.append(
            {
                "step_id": _nonempty(term.get("step_id"), f"{context}.step_id"),
                "type": _enum(term.get("type"), set(_SUCCESS_TYPES), f"{context}.type"),
            }
        )
    if not terms:
        raise ValueError("SuccessSpec.terms must not be empty.")
    _unique([term["step_id"] for term in terms], "SuccessSpec step IDs")
    if draft is not None:
        normalized_draft = validate_task_draft(draft)
        if normalized_draft["task_id"] != task_id:
            raise ValueError("SuccessSpec.task_id must match TaskDraft.task_id.")
        expected = [
            {
                "step_id": step["id"],
                "type": task_success_type(step["task_type"], step),
            }
            for step in normalized_draft["steps"]
        ]
        if terms != expected:
            raise ValueError(
                "SuccessSpec terms must be ordered, complete, and derived from "
                "task_success_type."
            )
    result["task_id"] = task_id
    result["terms"] = terms
    return result


def validate_task_candidate(value: Mapping[str, Any]) -> TaskCandidate:
    result = _mapping(value, "TaskCandidate")
    _keys(result, _CANDIDATE_KEYS, "TaskCandidate")
    result["candidate_id"] = _nonempty(
        result.get("candidate_id"), "TaskCandidate.candidate_id"
    )
    result["draft"] = validate_task_draft(result.get("draft"))
    result["scene_request"] = validate_scene_request(result.get("scene_request"))
    result["success_spec"] = validate_success_spec(
        result.get("success_spec"), draft=result["draft"]
    )
    for name in ("scene_request", "success_spec"):
        if result[name]["task_id"] != result["draft"]["task_id"]:
            raise ValueError(f"TaskCandidate {name}.task_id must match its draft.")
    from .agent import derive_scene_request

    if result["scene_request"] != derive_scene_request(result["draft"]):
        raise ValueError(
            "TaskCandidate.scene_request must be derived exactly from its draft."
        )
    result["semantic_hash"] = _digest(
        result.get("semantic_hash"), "TaskCandidate.semantic_hash"
    )
    if result["semantic_hash"] != canonical_hash(result["draft"]["steps"]):
        raise ValueError(
            "TaskCandidate.semantic_hash does not match its canonical steps."
        )
    result["vote_count"] = _integer(
        result.get("vote_count"), "TaskCandidate.vote_count", minimum=1
    )
    result["attempts"] = _integer(
        result.get("attempts"), "TaskCandidate.attempts", minimum=1, maximum=2
    )
    result["normalizations"] = _mapping_sequence(
        result.get("normalizations"), "TaskCandidate.normalizations"
    )
    return result


def validate_task_candidate_set(value: Mapping[str, Any]) -> TaskCandidateSet:
    result = _mapping(value, "TaskCandidateSet")
    _keys(result, _CANDIDATE_SET_KEYS, "TaskCandidateSet")
    _schema(result, TASK_CANDIDATE_SET_SCHEMA, "TaskCandidateSet")
    task_id = _nonempty(result.get("task_id"), "TaskCandidateSet.task_id")
    instruction = _nonempty(result.get("instruction"), "TaskCandidateSet.instruction")
    requested = _integer(
        result.get("requested_candidate_count"),
        "TaskCandidateSet.requested_candidate_count",
        minimum=1,
    )
    valid = _integer(
        result.get("valid_response_count"),
        "TaskCandidateSet.valid_response_count",
        minimum=1,
        maximum=requested,
    )
    candidates = [
        validate_task_candidate(item)
        for item in _sequence(result.get("candidates"), "TaskCandidateSet.candidates")
    ]
    if not candidates:
        raise ValueError("TaskCandidateSet.candidates must not be empty.")
    _unique([item["candidate_id"] for item in candidates], "TaskCandidate IDs")
    _unique(
        [item["semantic_hash"] for item in candidates], "TaskCandidate semantic hashes"
    )
    if sum(item["vote_count"] for item in candidates) != valid:
        raise ValueError(
            "TaskCandidate vote_count values must sum to valid_response_count."
        )
    for candidate in candidates:
        if (
            candidate["draft"]["task_id"] != task_id
            or candidate["draft"]["instruction"] != instruction
        ):
            raise ValueError("Every TaskCandidate draft must match its candidate set.")
    errors = _strings(result.get("errors"), "TaskCandidateSet.errors", allow_empty=True)
    if valid + len(errors) != requested:
        raise ValueError(
            "Valid responses plus errors must equal requested_candidate_count."
        )
    result.update(
        {
            "task_id": task_id,
            "instruction": instruction,
            "requested_candidate_count": requested,
            "valid_response_count": valid,
            "candidates": candidates,
            "errors": errors,
        }
    )
    return result


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return deepcopy(dict(value))


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a list.")
    return list(value)


def _keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], context: str
) -> None:
    if set(value) != set(expected):
        raise ValueError(
            f"{context} requires exactly fields {sorted(expected)}; received {sorted(value)}."
        )


def _schema(value: Mapping[str, Any], expected: str, context: str) -> None:
    if value.get("schema_version") != expected:
        raise ValueError(f"{context}.schema_version must be {expected!r}.")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string.")
    return value.strip()


def _nonempty(value: Any, context: str) -> str:
    result = _string(value, context)
    if not result:
        raise ValueError(f"{context} must not be empty.")
    return result


def _enum(value: Any, choices: set[str], context: str) -> str:
    result = _string(value, context)
    if result not in choices:
        raise ValueError(f"{context} must be one of {sorted(choices)}.")
    return result


def _integer(
    value: Any, context: str, *, minimum: int, maximum: int | None = None
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{context} must be an integer in the allowed range.")
    return value


def _strings(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    result = [_string(item, context) for item in _sequence(value, context)]
    if not allow_empty and any(not item for item in result):
        raise ValueError(f"{context} values must not be empty.")
    if len(result) != len(set(result)):
        raise ValueError(f"{context} values must be unique.")
    return result


def _mapping_sequence(value: Any, context: str) -> list[dict[str, Any]]:
    result = [_mapping(item, context) for item in _sequence(value, context)]
    _json_safe(result, context)
    return result


def _digest(value: Any, context: str) -> str:
    result = _string(value, context)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest.")
    return result


def _unique(values: Sequence[str], context: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{context} must be unique.")


def _json_safe(value: Any, context: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be finite and JSON serializable.") from error
