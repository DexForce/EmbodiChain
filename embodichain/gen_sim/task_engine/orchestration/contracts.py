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

"""Strict cross-engine contracts for scene binding and orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from typing import Any, TypeAlias

from embodichain.gen_sim.action_engine.domain import (
    validate_scene_requirements,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.runtime import (
    EXECUTION_REPORT_SCHEMA,
    validate_execution_report,
)
from embodichain.gen_sim.task_engine import (
    SCENE_REQUEST_SCHEMA,
    SUCCESS_SPEC_SCHEMA,
    TASK_CANDIDATE_SET_SCHEMA,
    TASK_DRAFT_SCHEMA,
    SceneRequest,
    SuccessSpec,
    TaskCandidate,
    TaskCandidateSet,
    TaskDraft,
    canonical_hash,
    task_success_type,
    validate_scene_request,
    validate_success_spec,
    validate_task_candidate,
    validate_task_candidate_set,
    validate_task_draft,
)

__all__ = [
    "BINDING_REPORT_SCHEMA",
    "EXECUTION_REPORT_SCHEMA",
    "GROUNDED_TASK_PLAN_SCHEMA",
    "ROLE_BINDINGS_SCHEMA",
    "SCENE_MANIFEST_SCHEMA",
    "SCENE_REQUEST_SCHEMA",
    "SUCCESS_SPEC_SCHEMA",
    "TASK_CANDIDATE_SET_SCHEMA",
    "TASK_DRAFT_SCHEMA",
    "BindingReport",
    "ExecutionReport",
    "GroundedTaskPlan",
    "RoleBindings",
    "SceneManifest",
    "SceneRequest",
    "SuccessSpec",
    "TaskCandidate",
    "TaskCandidateSet",
    "TaskDraft",
    "canonical_hash",
    "validate_binding_report",
    "validate_execution_report",
    "validate_grounded_task_plan",
    "validate_role_bindings",
    "validate_scene_manifest",
    "validate_scene_request",
    "validate_success_spec",
    "validate_task_candidate",
    "validate_task_candidate_set",
    "validate_task_draft",
]

SCENE_MANIFEST_SCHEMA = "action_engine_scene_manifest_v1"
ROLE_BINDINGS_SCHEMA = "action_engine_role_bindings_v1"
BINDING_REPORT_SCHEMA = "action_engine_binding_report_v1"
GROUNDED_TASK_PLAN_SCHEMA = "action_engine_grounded_task_plan_v1"
SceneManifest: TypeAlias = dict[str, Any]
RoleBindings: TypeAlias = dict[str, Any]
BindingReport: TypeAlias = dict[str, Any]
GroundedTaskPlan: TypeAlias = dict[str, Any]
ExecutionReport: TypeAlias = dict[str, Any]


def validate_scene_manifest(value: Mapping[str, Any]) -> SceneManifest:
    result = _mapping(value, "SceneManifest")
    _keys(
        result,
        {"schema_version", "scene_id", "source_format", "robot_profile", "objects"},
        "SceneManifest",
    )
    _schema(result, SCENE_MANIFEST_SCHEMA, "SceneManifest")
    for key in ("scene_id", "source_format", "robot_profile"):
        result[key] = _nonempty(result.get(key), f"SceneManifest.{key}")
    object_keys = {
        "uid",
        "role",
        "name",
        "description",
        "category",
        "color",
        "affordances",
        "initial_state",
        "attributes",
    }
    objects = []
    for index, raw in enumerate(
        _sequence(result.get("objects"), "SceneManifest.objects")
    ):
        context = f"SceneManifest.objects[{index}]"
        item = _mapping(raw, context)
        _keys(item, object_keys, context)
        item["uid"] = _nonempty(item.get("uid"), f"{context}.uid")
        for key in ("role", "name", "description", "category"):
            item[key] = _string(item.get(key), f"{context}.{key}")
        if item.get("color") is not None:
            item["color"] = _string(item.get("color"), f"{context}.color")
        item["affordances"] = _strings(
            item.get("affordances"), f"{context}.affordances"
        )
        item["initial_state"] = _mapping(
            item.get("initial_state"), f"{context}.initial_state"
        )
        item["attributes"] = _mapping(item.get("attributes"), f"{context}.attributes")
        objects.append(item)
    _unique([item["uid"] for item in objects], "SceneManifest object UIDs")
    result["objects"] = objects
    _json_safe(result, "SceneManifest")
    return result


def validate_role_bindings(value: Mapping[str, Any]) -> RoleBindings:
    result = _mapping(value, "RoleBindings")
    _keys(
        result,
        {
            "schema_version",
            "task_id",
            "candidate_id",
            "reference_bindings",
            "role_bindings",
        },
        "RoleBindings",
    )
    _schema(result, ROLE_BINDINGS_SCHEMA, "RoleBindings")
    for key in ("task_id", "candidate_id"):
        result[key] = _nonempty(result.get(key), f"RoleBindings.{key}")
    result["reference_bindings"] = _string_lists(
        result.get("reference_bindings"), "RoleBindings.reference_bindings"
    )
    if any(not uids for uids in result["reference_bindings"].values()):
        raise ValueError("RoleBindings.reference_bindings values must not be empty.")
    result["role_bindings"] = _string_map(
        result.get("role_bindings"), "RoleBindings.role_bindings"
    )
    return result


def validate_binding_report(value: Mapping[str, Any]) -> BindingReport:
    result = _mapping(value, "BindingReport")
    _keys(
        result,
        {
            "schema_version",
            "task_id",
            "status",
            "selected_candidate_id",
            "selection_reason",
            "candidates",
        },
        "BindingReport",
    )
    _schema(result, BINDING_REPORT_SCHEMA, "BindingReport")
    result["task_id"] = _nonempty(result.get("task_id"), "BindingReport.task_id")
    result["status"] = _enum(
        result.get("status"),
        {"bound", "ambiguous", "unsatisfied"},
        "BindingReport.status",
    )
    result["selected_candidate_id"] = _string(
        result.get("selected_candidate_id"), "BindingReport.selected_candidate_id"
    )
    result["selection_reason"] = _string(
        result.get("selection_reason"), "BindingReport.selection_reason"
    )
    if result["status"] == "bound" and not result["selected_candidate_id"]:
        raise ValueError("A bound BindingReport requires selected_candidate_id.")
    candidate_keys = {
        "candidate_id",
        "semantic_hash",
        "status",
        "references",
        "reasons",
    }
    reference_keys = {
        "reference_id",
        "status",
        "confidence",
        "candidate_uids",
        "selected_uids",
        "reasons",
    }
    candidates = []
    for index, raw in enumerate(
        _sequence(result.get("candidates"), "BindingReport.candidates")
    ):
        context = f"BindingReport.candidates[{index}]"
        candidate = _mapping(raw, context)
        _keys(candidate, candidate_keys, context)
        candidate["candidate_id"] = _nonempty(
            candidate.get("candidate_id"), f"{context}.candidate_id"
        )
        candidate["semantic_hash"] = _digest(
            candidate.get("semantic_hash"), f"{context}.semantic_hash"
        )
        candidate["status"] = _enum(
            candidate.get("status"),
            {"resolved", "ambiguous", "not_found", "incompatible"},
            f"{context}.status",
        )
        references = []
        for ref_index, ref_raw in enumerate(
            _sequence(candidate.get("references"), f"{context}.references")
        ):
            ref_context = f"{context}.references[{ref_index}]"
            reference = _mapping(ref_raw, ref_context)
            _keys(reference, reference_keys, ref_context)
            reference["reference_id"] = _nonempty(
                reference.get("reference_id"), f"{ref_context}.reference_id"
            )
            reference["status"] = _enum(
                reference.get("status"),
                {"resolved", "ambiguous", "not_found", "incompatible"},
                f"{ref_context}.status",
            )
            reference["confidence"] = _number(
                reference.get("confidence"),
                f"{ref_context}.confidence",
                minimum=0.0,
                maximum=1.0,
            )
            reference["candidate_uids"] = _strings(
                reference.get("candidate_uids"),
                f"{ref_context}.candidate_uids",
                allow_empty=True,
            )
            reference["selected_uids"] = _strings(
                reference.get("selected_uids"),
                f"{ref_context}.selected_uids",
                allow_empty=True,
            )
            reference["reasons"] = _strings(
                reference.get("reasons"), f"{ref_context}.reasons", allow_empty=True
            )
            selected = set(reference["selected_uids"])
            candidates_for_reference = set(reference["candidate_uids"])
            if not selected <= candidates_for_reference:
                raise ValueError(
                    f"{ref_context}.selected_uids must be a subset of candidate_uids."
                )
            if reference["status"] == "resolved" and not selected:
                raise ValueError(
                    f"{ref_context} status=resolved requires selected_uids."
                )
            if reference["status"] != "resolved" and selected:
                raise ValueError(
                    f"{ref_context} non-resolved status cannot select UIDs."
                )
            if reference["status"] == "not_found" and candidates_for_reference:
                raise ValueError(
                    f"{ref_context} status=not_found cannot carry candidate_uids."
                )
            references.append(reference)
        if not references:
            raise ValueError(f"{context}.references must not be empty.")
        _unique(
            [item["reference_id"] for item in references],
            f"{context} reference IDs",
        )
        expected_status = _candidate_binding_status(references)
        if candidate["status"] != expected_status:
            raise ValueError(
                f"{context}.status must be {expected_status!r} for its references."
            )
        candidate["references"] = references
        candidate["reasons"] = _strings(
            candidate.get("reasons"), f"{context}.reasons", allow_empty=True
        )
        candidates.append(candidate)
    if not candidates:
        raise ValueError("BindingReport.candidates must not be empty.")
    _unique(
        [item["candidate_id"] for item in candidates], "BindingReport candidate IDs"
    )
    if result["selected_candidate_id"] and result["selected_candidate_id"] not in {
        item["candidate_id"] for item in candidates
    }:
        raise ValueError("BindingReport.selected_candidate_id is unknown.")
    if result["status"] != "bound" and result["selected_candidate_id"]:
        raise ValueError(
            "A non-bound BindingReport cannot carry selected_candidate_id."
        )
    selected = next(
        (
            candidate
            for candidate in candidates
            if candidate["candidate_id"] == result["selected_candidate_id"]
        ),
        None,
    )
    if result["status"] == "bound" and (
        selected is None or selected["status"] != "resolved"
    ):
        raise ValueError(
            "A bound BindingReport must select a resolved candidate audit."
        )
    if result["status"] == "unsatisfied" and any(
        candidate["status"] in {"resolved", "ambiguous"} for candidate in candidates
    ):
        raise ValueError(
            "An unsatisfied BindingReport cannot contain resolved or ambiguous candidates."
        )
    result["candidates"] = candidates
    return result


def validate_grounded_task_plan(value: Mapping[str, Any]) -> GroundedTaskPlan:
    result = _mapping(value, "GroundedTaskPlan")
    keys = {
        "schema_version",
        "task_id",
        "instruction",
        "selected_candidate_id",
        "task_draft",
        "task_spec",
        "scene_requirements",
        "success_spec",
        "scene_manifest",
        "role_bindings",
        "binding_report",
        "hashes",
    }
    _keys(result, keys, "GroundedTaskPlan")
    _schema(result, GROUNDED_TASK_PLAN_SCHEMA, "GroundedTaskPlan")
    task_id = _nonempty(result.get("task_id"), "GroundedTaskPlan.task_id")
    result["instruction"] = _nonempty(
        result.get("instruction"), "GroundedTaskPlan.instruction"
    )
    result["selected_candidate_id"] = _nonempty(
        result.get("selected_candidate_id"), "GroundedTaskPlan.selected_candidate_id"
    )
    result["task_draft"] = validate_task_draft(result.get("task_draft"))
    # Candidate SuccessSpec terms use draft step IDs.  Once count/all selectors
    # are lowered, the grounded plan carries one term per concrete TaskSpec
    # instance instead, and is checked against the v2 recipe below.
    result["success_spec"] = validate_success_spec(result.get("success_spec"))
    result["scene_manifest"] = validate_scene_manifest(result.get("scene_manifest"))
    result["role_bindings"] = validate_role_bindings(result.get("role_bindings"))
    result["binding_report"] = validate_binding_report(result.get("binding_report"))
    result["task_spec"] = validate_task_spec(
        _mapping(result.get("task_spec"), "GroundedTaskPlan.task_spec")
    )
    result["scene_requirements"] = validate_scene_requirements(
        _mapping(
            result.get("scene_requirements"),
            "GroundedTaskPlan.scene_requirements",
        )
    )
    hashes = _mapping(result.get("hashes"), "GroundedTaskPlan.hashes")
    _keys(
        hashes,
        {"task_draft", "task_spec", "scene_manifest", "role_bindings", "plan"},
        "GroundedTaskPlan.hashes",
    )
    for key in hashes:
        hashes[key] = _digest(hashes[key], f"GroundedTaskPlan.hashes.{key}")
    if any(
        part["task_id"] != task_id
        for part in (
            result["task_draft"],
            result["task_spec"],
            result["scene_requirements"],
            result["success_spec"],
            result["role_bindings"],
            result["binding_report"],
        )
    ):
        raise ValueError("GroundedTaskPlan task IDs must agree.")
    if result["task_draft"]["instruction"] != result["instruction"]:
        raise ValueError("GroundedTaskPlan instruction must match TaskDraft.")
    if result["task_spec"]["instruction"] != result["instruction"]:
        raise ValueError("GroundedTaskPlan instruction must match TaskSpec.")
    if (
        result["role_bindings"]["candidate_id"] != result["selected_candidate_id"]
        or result["binding_report"]["selected_candidate_id"]
        != result["selected_candidate_id"]
    ):
        raise ValueError("GroundedTaskPlan selected candidate IDs must agree.")
    if result["binding_report"]["status"] != "bound":
        raise ValueError("GroundedTaskPlan requires a bound BindingReport.")
    selected_audit = next(
        candidate
        for candidate in result["binding_report"]["candidates"]
        if candidate["candidate_id"] == result["selected_candidate_id"]
    )
    if selected_audit["semantic_hash"] != canonical_hash(result["task_draft"]["steps"]):
        raise ValueError(
            "GroundedTaskPlan selected candidate hash must match TaskDraft."
        )
    task_metadata = result["task_spec"].get("metadata", {})
    task_oracle = result["task_spec"].get("oracle", {})
    serialized_bindings = (
        task_metadata.get("role_bindings")
        if isinstance(task_metadata, Mapping)
        and task_metadata.get("role_bindings") is not None
        else (
            task_oracle.get("role_bindings")
            if isinstance(task_oracle, Mapping)
            else None
        )
    )
    if serialized_bindings != result["role_bindings"]["role_bindings"]:
        raise ValueError(
            "GroundedTaskPlan RoleBindings must match the TaskSpec binding hand-off."
        )
    requirement_roles = {
        str(item["role_id"]) for item in result["scene_requirements"]["objects"]
    }
    if requirement_roles != set(result["role_bindings"]["role_bindings"]):
        raise ValueError(
            "GroundedTaskPlan SceneRequirements roles must match RoleBindings."
        )
    manifest_uids = {item["uid"] for item in result["scene_manifest"]["objects"]}
    missing_uids = sorted(
        set(result["role_bindings"]["role_bindings"].values()) - manifest_uids
    )
    if missing_uids:
        raise ValueError(
            f"GroundedTaskPlan RoleBindings reference unknown scene UIDs {missing_uids}."
        )
    reference_ids = {
        f"{step['id']}.{role}"
        for step in result["task_draft"]["steps"]
        for role in ("object", "target")
        if step[role]["kind"] == "scene_ref"
    }
    reference_bindings = result["role_bindings"]["reference_bindings"]
    if set(reference_bindings) != reference_ids:
        raise ValueError(
            "GroundedTaskPlan reference bindings must cover every draft scene_ref exactly."
        )
    bound_uids = {uid for uids in reference_bindings.values() for uid in uids} | set(
        result["role_bindings"]["role_bindings"].values()
    )
    unknown_uids = sorted(bound_uids - manifest_uids)
    if unknown_uids:
        raise ValueError(
            "GroundedTaskPlan bindings reference unknown SceneManifest UIDs: "
            f"{unknown_uids}."
        )
    audited_bindings = {
        reference["reference_id"]: reference["selected_uids"]
        for reference in selected_audit["references"]
    }
    if audited_bindings != reference_bindings:
        raise ValueError(
            "GroundedTaskPlan RoleBindings must match the selected candidate audit."
        )
    ontology_success = [
        {
            "step_id": instance["id"],
            "type": task_success_type(instance["task_type"], instance["params"]),
        }
        for instance in result["task_spec"]["task_instances"]
    ]
    recipe_success = [
        {
            "step_id": term["task_instance_id"],
            "type": term["type"],
        }
        for term in result["task_spec"]["success"]["terms"]
    ]
    if recipe_success != ontology_success:
        raise ValueError(
            "GroundedTaskPlan TaskSpec success recipe must be derived from "
            "task_success_type."
        )
    if result["success_spec"]["terms"] != ontology_success:
        raise ValueError(
            "GroundedTaskPlan SuccessSpec must exactly match the lowered "
            "TaskSpec success recipe."
        )
    expected_hashes = {
        "task_draft": canonical_hash(result["task_draft"]),
        "task_spec": canonical_hash(result["task_spec"]),
        "scene_manifest": canonical_hash(result["scene_manifest"]),
        "role_bindings": canonical_hash(result["role_bindings"]),
    }
    base = {key: value for key, value in result.items() if key != "hashes"}
    expected_hashes["plan"] = canonical_hash(base)
    if hashes != expected_hashes:
        raise ValueError("GroundedTaskPlan hashes do not match their contents.")
    result["task_id"] = task_id
    result["hashes"] = hashes
    _json_safe(result, "GroundedTaskPlan")
    return result


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return deepcopy(dict(value))


def _candidate_binding_status(references: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(reference["status"]) for reference in references}
    if statuses == {"resolved"}:
        return "resolved"
    if "ambiguous" in statuses:
        return "ambiguous"
    if "incompatible" in statuses:
        return "incompatible"
    return "not_found"


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


def _number(value: Any, context: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(
            f"{context} must be a finite number between {minimum} and {maximum}."
        )
    return float(value)


def _strings(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    result = [_string(item, context) for item in _sequence(value, context)]
    if not allow_empty and any(not item for item in result):
        raise ValueError(f"{context} values must not be empty.")
    if len(result) != len(set(result)):
        raise ValueError(f"{context} values must be unique.")
    return result


def _string_map(value: Any, context: str) -> dict[str, str]:
    result = _mapping(value, context)
    return {
        _nonempty(key, context): _nonempty(item, context)
        for key, item in result.items()
    }


def _string_lists(value: Any, context: str) -> dict[str, list[str]]:
    result = _mapping(value, context)
    return {
        _nonempty(key, context): _strings(item, context) for key, item in result.items()
    }


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
