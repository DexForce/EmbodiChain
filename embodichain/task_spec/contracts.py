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

"""Strict data records and evidence acceptance; no execution or storage ownership."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from ._json import digest, fields, hash_json, items, json_copy, text
from .canonicalization import canonical_template, semantic_hash
from .registry import registry_snapshot
from .validation import validate_template_structure

__all__ = [
    "TASK_TEMPLATE_SCHEMA",
    "SCENE_INSTANCE_SCHEMA",
    "ACTION_WITNESS_SCHEMA",
    "EXPANSION_MANIFEST_SCHEMA",
    "VALIDATION_CERTIFICATE_SCHEMA",
    "TaskTemplate",
    "SceneInstance",
    "ActionWitness",
    "ExpansionManifest",
    "ValidationCertificate",
    "validate_task_template",
    "scene_instance_hash",
    "validate_scene_instance",
    "validate_action_witness",
    "validate_expansion_manifest",
    "validate_validation_certificate",
    "certificate_passed",
]

TASK_TEMPLATE_SCHEMA = "taskspec/template/v0.1"
SCENE_INSTANCE_SCHEMA = "taskspec/scene_instance/v0.1"
ACTION_WITNESS_SCHEMA = "taskspec/action_witness/v0.1"
EXPANSION_MANIFEST_SCHEMA = "taskspec/expansion/v0.1"
VALIDATION_CERTIFICATE_SCHEMA = "taskspec/certificate/v0.1"

TaskTemplate: TypeAlias = dict[str, Any]
SceneInstance: TypeAlias = dict[str, Any]
ActionWitness: TypeAlias = dict[str, Any]
ExpansionManifest: TypeAlias = dict[str, Any]
ValidationCertificate: TypeAlias = dict[str, Any]

_SCOPES = frozenset(
    {
        "initial",
        "preflight",
        "planning",
        "segment",
        "task_goal",
        "invariant",
        "robustness",
        "execution",
        "capability",
        "grounding",
    }
)
_INSTANCE_FIELDS = {
    "template_hash",
    "roles",
    "scene",
    "embodiment",
    "initial_state",
    "evidence",
}


def _record(
    value: object, schema: str, required: set[str], optional: set[str] | None = None
) -> dict[str, Any]:
    result = fields(
        json_copy(value), {"schema_version"} | required, optional or set(), schema
    )
    if result["schema_version"] != schema:
        raise ValueError(f"unsupported schema_version: expected {schema}")
    return result


def _reference(value: object, path: str) -> None:
    result = fields(value, {"uri", "content_hash"}, set(), path)
    text(result["uri"], f"{path}.uri")
    digest(result["content_hash"], f"{path}.content_hash")


def _references(value: object, path: str, *, nonempty: bool = False) -> None:
    for index, ref in enumerate(items(value, path, nonempty=nonempty)):
        _reference(ref, f"{path}[{index}]")


def _version(value: object, path: str) -> None:
    result = fields(value, {"name", "version"}, set(), path)
    text(result["name"], f"{path}.name")
    text(result["version"], f"{path}.version")


def _unique_strings(value: object, path: str, *, nonempty: bool = False) -> list[str]:
    result = [text(item, path) for item in items(value, path, nonempty=nonempty)]
    if len(set(result)) != len(result):
        raise ValueError(f"{path} must not contain duplicates")
    return result


def _episode(value: dict[str, Any], path: str, *, required: bool = True) -> None:
    env_id, episode_id = value["env_id"], value["episode_id"]
    if not required and env_id is None and episode_id is None:
        return
    if type(env_id) is not int or env_id < 0:
        raise ValueError(f"{path}.env_id must be a nonnegative integer")
    text(episode_id, f"{path}.episode_id")


def validate_task_template(value: Mapping[str, Any]) -> TaskTemplate:
    """Validate normative semantics and their stored identity.

    Args:
        value: Complete TaskTemplate including semantic_hash.

    Returns:
        Detached, validated record preserving author labels and input units.

    Raises:
        ValueError: Invalid semantics, absent or stale semantic hash.
    """
    result = validate_template_structure(value)
    digest(result.get("semantic_hash"), "TaskTemplate.semantic_hash")
    if result["semantic_hash"] != semantic_hash(result):
        raise ValueError(
            "TaskTemplate.semantic_hash does not match canonical semantics"
        )
    return result


def _instance(value: Mapping[str, Any]) -> SceneInstance:
    result = _record(value, SCENE_INSTANCE_SCHEMA, _INSTANCE_FIELDS, {"content_hash"})
    digest(result["template_hash"], "SceneInstance.template_hash")
    roles = result["roles"]
    if not isinstance(roles, dict) or not roles:
        raise ValueError("SceneInstance.roles must be a nonempty object")
    if len(roles) > 6 or set(roles) != {f"role_{i}" for i in range(len(roles))}:
        raise ValueError("SceneInstance.roles must use contiguous canonical role IDs")
    entities = []
    for role, binding in roles.items():
        text(role, "SceneInstance role")
        fields(binding, {"entity_id", "asset"}, set(), "role binding")
        entities.append(text(binding["entity_id"], "role binding.entity_id"))
        _reference(binding["asset"], "role binding.asset")
    if len(set(entities)) != len(entities):
        raise ValueError("SceneInstance roles must bind distinct entities")
    for field in ("scene", "embodiment", "initial_state"):
        _reference(result[field], f"SceneInstance.{field}")
    _references(result["evidence"], "SceneInstance.evidence", nonempty=True)
    if "content_hash" in result:
        digest(result["content_hash"], "SceneInstance.content_hash")
    return result


def scene_instance_hash(value: Mapping[str, Any]) -> str:
    """Return the exact content identity of an instance manifest.

    Args:
        value: Valid instance payload with optional content_hash.

    Returns:
        SHA-256 covering grounding, asset and component references, observed
        initial-state and evidence references, excluding the top-level hash.
        Referenced bytes must still be verified by the consuming host.
    """
    result = _instance(value)
    result.pop("content_hash", None)
    return hash_json(result)


def validate_scene_instance(
    value: Mapping[str, Any],
    *,
    template: Mapping[str, Any] | None = None,
) -> SceneInstance:
    """Validate instance grounding, content references and stored identity.

    Args:
        value: SceneInstance JSON with a content_hash.
        template: Optional owner used to verify semantic identity and complete roles.

    Returns:
        Detached instance manifest. No file is read and no initial state is
        inferred from requested generator poses.
    """
    result = _instance(value)
    digest(result.get("content_hash"), "SceneInstance.content_hash")
    if result["content_hash"] != scene_instance_hash(result):
        raise ValueError("SceneInstance.content_hash does not match its content")
    if template is not None:
        owner = validate_task_template(template)
        if owner["semantic_hash"] != result["template_hash"]:
            raise ValueError("SceneInstance template identity does not match its owner")
        if result["roles"].keys() != canonical_template(owner)["roles"].keys():
            raise ValueError("SceneInstance roles must exactly match its template")
    return result


def validate_action_witness(
    value: Mapping[str, Any],
    *,
    instance: Mapping[str, Any] | None = None,
) -> ActionWitness:
    """Validate one candidate or executed solution's artifact references.

    Args:
        value: Witness JSON; candidate execution must be null.
        instance: Optional owner used to verify instance and template identities.

    Returns:
        Detached record. Succeeded status requires affirmative program/task
        results and execution/evaluation evidence references; this structural
        validation does not certify the referenced reports' truth.
    """
    result = _record(
        value,
        ACTION_WITNESS_SCHEMA,
        {
            "template_hash",
            "instance_hash",
            "status",
            "plan",
            "execution",
            "evidence",
        },
    )
    for key in ("template_hash", "instance_hash"):
        digest(result[key], f"ActionWitness.{key}")
    if instance is not None:
        owner = validate_scene_instance(instance)
        if (
            result["instance_hash"] != owner["content_hash"]
            or result["template_hash"] != owner["template_hash"]
        ):
            raise ValueError(
                "ActionWitness instance/template identities do not match its owner"
            )
    status = text(result["status"], "ActionWitness.status")
    if status not in {"candidate", "succeeded", "failed"}:
        raise ValueError("unsupported ActionWitness status")
    plan_refs = {"graph", "program", "integration", "execution_policy", "constraints"}
    plan = fields(
        result["plan"],
        plan_refs | {"integration_fingerprint"},
        set(),
        "ActionWitness.plan",
    )
    for key in plan_refs:
        _reference(plan[key], f"ActionWitness.plan.{key}")
    digest(plan["integration_fingerprint"], "integration_fingerprint")
    _references(
        result["evidence"], "ActionWitness.evidence", nonempty=status != "candidate"
    )
    if status == "candidate":
        if result["execution"] is not None:
            raise ValueError("candidate must not claim an execution result")
        return result
    execution = fields(
        result["execution"],
        {
            "env_id",
            "episode_id",
            "program_success",
            "task_success",
            "runtime_result",
            "task_evaluation",
            "trajectory",
        },
        set(),
        "ActionWitness.execution",
    )
    _episode(execution, "ActionWitness.execution")
    for key in ("program_success", "task_success"):
        if type(execution[key]) is not bool:
            raise ValueError(f"execution.{key} must be a boolean")
    for key in ("runtime_result", "task_evaluation"):
        _reference(execution[key], f"ActionWitness.execution.{key}")
    _references(
        execution["trajectory"],
        "ActionWitness.execution.trajectory",
        nonempty=status == "succeeded",
    )
    succeeded = execution["program_success"] and execution["task_success"]
    if (status == "succeeded") != succeeded:
        raise ValueError("witness status disagrees with program/task success")
    return result


def validate_expansion_manifest(value: Mapping[str, Any]) -> ExpansionManifest:
    """Validate lineage with independent change scope and semantic effect.

    Args:
        value: Expansion record with parent/child content references.

    Returns:
        Detached manifest. Invalidation declarations remain evidence to be
        enforced by the host; this decoder neither expands nor runs checks.
    """
    result = _record(
        value,
        EXPANSION_MANIFEST_SCHEMA,
        {
            "parent",
            "changes",
            "operator",
            "seed",
            "semantic_effect",
            "invalidates",
        },
    )
    _reference(result["parent"], "ExpansionManifest.parent")
    for change in items(result["changes"], "changes", nonempty=True):
        fields(change, {"scope", "child"}, set(), "change")
        scope = text(change["scope"], "change.scope")
        if scope not in {"language", "scene", "embodiment", "trajectory", "episode"}:
            raise ValueError("unsupported expansion scope")
        _reference(change["child"], "change.child")
    operator = fields(
        result["operator"], {"name", "version", "parameters"}, set(), "operator"
    )
    _version({key: operator[key] for key in ("name", "version")}, "operator")
    if not isinstance(operator["parameters"], dict):
        raise ValueError("operator.parameters must be an object")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if text(result["semantic_effect"], "semantic_effect") not in {
        "preserve",
        "mutate",
        "unknown",
    }:
        raise ValueError("unsupported semantic_effect")
    for scope in _unique_strings(result["invalidates"], "invalidates"):
        if scope not in _SCOPES:
            raise ValueError(f"unsupported invalidated check scope {scope!r}")
    return result


def validate_validation_certificate(value: Mapping[str, Any]) -> ValidationCertificate:
    """Validate evidence records without promoting unrun checks to passes.

    Args:
        value: Certificate with explicit required check IDs and evidence.

    Returns:
        Detached certificate, including failed, unavailable and absent checks.
        Use certificate_passed for policy acceptance, after the host verifies
        referenced content. No checker or simulator is invoked here.
    """
    result = _record(
        value,
        VALIDATION_CERTIFICATE_SCHEMA,
        {
            "template_hash",
            "instance_hash",
            "witness_id",
            "checks",
            "policy",
        },
    )
    for key in ("template_hash", "instance_hash"):
        digest(result[key], key)
    _reference(result["witness_id"], "witness_id")
    policy = fields(
        result["policy"], {"id", "version", "required_checks"}, set(), "policy"
    )
    text(policy["id"], "policy.id")
    text(policy["version"], "policy.version")
    _unique_strings(policy["required_checks"], "required_checks", nonempty=True)
    ids = set()
    episodes = set()
    for check in items(result["checks"], "checks"):
        fields(
            check,
            {
                "id",
                "status",
                "scope",
                "inputs",
                "checker",
                "predicate",
                "parameters",
                "metrics",
                "env_id",
                "episode_id",
                "evidence",
            },
            set(),
            "check",
        )
        identifier = text(check["id"], "check.id")
        if identifier in ids:
            raise ValueError("duplicate certificate check id")
        ids.add(identifier)
        status = text(check["status"], "check.status")
        scope = text(check["scope"], "check.scope")
        if status not in {"pass", "failed", "not_run", "unavailable", "unsupported"}:
            raise ValueError("unsupported check status")
        if scope not in _SCOPES:
            raise ValueError("unsupported check scope")
        inputs = check["inputs"]
        if (
            not isinstance(inputs, dict)
            or not {"template", "instance", "witness"} <= inputs.keys()
        ):
            raise ValueError(
                "check.inputs requires template, instance and witness identities"
            )
        for name, content in inputs.items():
            text(name, "input name")
            digest(content, "check input content hash")
        if (
            inputs["template"] != result["template_hash"]
            or inputs["instance"] != result["instance_hash"]
            or inputs["witness"] != result["witness_id"]["content_hash"]
        ):
            raise ValueError("check inputs disagree with certificate identities")
        _version(check["checker"], "check.checker")
        if check["predicate"] is not None:
            _version(check["predicate"], "check.predicate")
            if status == "pass":
                spec = registry_snapshot()["predicates"].get(check["predicate"]["name"])
                if spec is None or spec["revision"] != check["predicate"]["version"]:
                    raise ValueError("pass requires a supported predicate revision")
        elif status == "pass" and scope in {"initial", "task_goal", "invariant"}:
            raise ValueError("predicate checks require a predicate revision")
        for key in ("parameters", "metrics"):
            if not isinstance(check[key], dict):
                raise ValueError(f"check.{key} must be an object")
        _episode(
            check,
            "check",
            required=scope
            in {"initial", "segment", "task_goal", "invariant", "execution"}
            and status == "pass",
        )
        if check["episode_id"] is not None:
            episodes.add((check["env_id"], check["episode_id"]))
        _references(check["evidence"], "check.evidence", nonempty=status == "pass")
    if len(episodes) > 1:
        raise ValueError("one certificate cannot mix env/episode evidence")
    return result


def certificate_passed(value: Mapping[str, Any]) -> bool:
    """Apply the certificate's explicit required-check acceptance policy.

    Args:
        value: Certificate with valid input/checker/evidence references.

    Returns:
        True only if every required check exists and has status pass. This is
        report acceptance, not independent verification of referenced bytes.
    """
    certificate = validate_validation_certificate(value)
    statuses = {check["id"]: check["status"] for check in certificate["checks"]}
    return all(
        statuses.get(key) == "pass" for key in certificate["policy"]["required_checks"]
    )
