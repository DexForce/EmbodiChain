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

"""JSON contracts owned by the Scene Engine anti-corruption boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from typing import Any, TypeAlias

__all__ = [
    "ASSESSMENT_STATUSES",
    "FEASIBILITY_REPORT_SCHEMA",
    "REMEDIATION_CLASSES",
    "STATIC_SCENE_MANIFEST_SCHEMA",
    "FeasibilityReport",
    "StaticSceneManifest",
    "validate_feasibility_report",
    "validate_static_scene_manifest",
]


STATIC_SCENE_MANIFEST_SCHEMA = "embodichain.static-scene-manifest/v1"
FEASIBILITY_REPORT_SCHEMA = "embodichain.scene-action-feasibility/v2"
ASSESSMENT_STATUSES = frozenset({"proven", "runtime_probe", "unknown", "contradicted"})
REMEDIATION_CLASSES = frozenset(
    {"none", "scene_remediable", "action_capability", "input_conflict", "terminal"}
)
_EVIDENCE_STATUSES = frozenset({"declared", "inferred", "verified", "contradicted"})

StaticSceneManifest: TypeAlias = dict[str, Any]
FeasibilityReport: TypeAlias = dict[str, Any]


def validate_static_scene_manifest(value: Mapping[str, Any]) -> StaticSceneManifest:
    """Validate and detach one static scene manifest."""
    result = _mapping(value, "StaticSceneManifest")
    _exact_keys(
        result,
        {
            "schema_version",
            "scene_id",
            "source_format",
            "robot_profile",
            "source",
            "adapter_capabilities",
            "objects",
        },
        "StaticSceneManifest",
    )
    _schema(result, STATIC_SCENE_MANIFEST_SCHEMA, "StaticSceneManifest")
    for key in ("scene_id", "source_format", "robot_profile"):
        result[key] = _nonempty(result.get(key), f"StaticSceneManifest.{key}")
    result["source"] = _mapping(result.get("source"), "StaticSceneManifest.source")
    result["adapter_capabilities"] = _bool_mapping(
        result.get("adapter_capabilities"),
        "StaticSceneManifest.adapter_capabilities",
    )

    objects: list[dict[str, Any]] = []
    for index, raw in enumerate(_sequence(result.get("objects"), "objects")):
        context = f"StaticSceneManifest.objects[{index}]"
        item = _mapping(raw, context)
        _exact_keys(
            item,
            {
                "uid",
                "source_uid",
                "role",
                "name",
                "description",
                "category",
                "color",
                "geometry",
                "initial_pose",
                "physics",
                "articulation",
                "affordances",
                "initial_state",
                "attributes",
                "provenance",
            },
            context,
        )
        item["uid"] = _nonempty(item.get("uid"), f"{context}.uid")
        item["source_uid"] = _string(item.get("source_uid"), f"{context}.source_uid")
        item["role"] = _nonempty(item.get("role"), f"{context}.role")
        for key in ("name", "description", "category"):
            item[key] = _string(item.get(key), f"{context}.{key}")
        color = item.get("color")
        if color is not None:
            color = _string(color, f"{context}.color")
        item["color"] = color
        for key in (
            "geometry",
            "initial_pose",
            "physics",
            "articulation",
            "initial_state",
            "attributes",
            "provenance",
        ):
            item[key] = _mapping(item.get(key), f"{context}.{key}")
        item["affordances"] = [
            _validate_affordance(evidence, f"{context}.affordances[{evidence_index}]")
            for evidence_index, evidence in enumerate(
                _sequence(item.get("affordances"), f"{context}.affordances")
            )
        ]
        objects.append(item)
    uids = [item["uid"] for item in objects]
    if len(set(uids)) != len(uids):
        raise ValueError("StaticSceneManifest object UIDs must be unique.")
    result["objects"] = objects
    _json_safe(result, "StaticSceneManifest")
    return result


def validate_feasibility_report(value: Mapping[str, Any]) -> FeasibilityReport:
    """Validate and detach one scene/action feasibility report."""
    result = _mapping(value, "FeasibilityReport")
    _exact_keys(
        result,
        {
            "schema_version",
            "task_id",
            "candidate_id",
            "scene_id",
            "status",
            "remediation_class",
            "checks",
            "blockers",
            "summary",
        },
        "FeasibilityReport",
    )
    _schema(result, FEASIBILITY_REPORT_SCHEMA, "FeasibilityReport")
    for key in ("task_id", "candidate_id", "scene_id"):
        result[key] = _nonempty(result.get(key), f"FeasibilityReport.{key}")
    result["status"] = _status(result.get("status"), "FeasibilityReport.status")
    remediation_class = result.get("remediation_class")
    if remediation_class not in REMEDIATION_CLASSES:
        raise ValueError(
            "FeasibilityReport.remediation_class must be one of "
            f"{sorted(REMEDIATION_CLASSES)}."
        )
    result["remediation_class"] = str(remediation_class)
    if result["status"] != "contradicted" and remediation_class != "none":
        raise ValueError(
            "A non-contradicted FeasibilityReport requires remediation_class=none."
        )
    checks: list[dict[str, Any]] = []
    for index, raw in enumerate(_sequence(result.get("checks"), "checks")):
        context = f"FeasibilityReport.checks[{index}]"
        item = _mapping(raw, context)
        _exact_keys(
            item,
            {"kind", "subject", "status", "reason", "evidence"},
            context,
        )
        item["kind"] = _nonempty(item.get("kind"), f"{context}.kind")
        item["subject"] = _nonempty(item.get("subject"), f"{context}.subject")
        item["status"] = _status(item.get("status"), f"{context}.status")
        item["reason"] = _nonempty(item.get("reason"), f"{context}.reason")
        item["evidence"] = _mapping(item.get("evidence"), f"{context}.evidence")
        checks.append(item)
    result["checks"] = checks
    blockers = _sequence(result.get("blockers"), "FeasibilityReport.blockers")
    if any(not isinstance(item, str) or not item for item in blockers):
        raise ValueError("FeasibilityReport.blockers must contain non-empty strings.")
    result["blockers"] = list(blockers)
    summary = _mapping(result.get("summary"), "FeasibilityReport.summary")
    expected = set(ASSESSMENT_STATUSES)
    if set(summary) != expected:
        raise ValueError(
            "FeasibilityReport.summary must count every assessment status."
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in summary.values()
    ):
        raise ValueError(
            "FeasibilityReport.summary counts must be non-negative integers."
        )
    if sum(summary.values()) != len(checks):
        raise ValueError("FeasibilityReport.summary must match the check count.")
    result["summary"] = dict(summary)
    _json_safe(result, "FeasibilityReport")
    return result


def _validate_affordance(value: Any, context: str) -> dict[str, Any]:
    item = _mapping(value, context)
    _exact_keys(
        item,
        {
            "type",
            "status",
            "confidence",
            "source",
            "link_uid",
            "frame",
            "parameters",
        },
        context,
    )
    item["type"] = _nonempty(item.get("type"), f"{context}.type")
    status = item.get("status")
    if status not in _EVIDENCE_STATUSES:
        raise ValueError(
            f"{context}.status must be one of {sorted(_EVIDENCE_STATUSES)}."
        )
    confidence = item.get("confidence")
    if confidence is not None:
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError(f"{context}.confidence must be null or in [0, 1].")
        confidence = float(confidence)
    item["confidence"] = confidence
    item["source"] = _nonempty(item.get("source"), f"{context}.source")
    item["link_uid"] = _string(item.get("link_uid"), f"{context}.link_uid")
    item["frame"] = _mapping(item.get("frame"), f"{context}.frame")
    item["parameters"] = _mapping(item.get("parameters"), f"{context}.parameters")
    return item


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    return deepcopy(dict(value))


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence.")
    return list(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{context} fields differ; missing={missing}, extra={extra}.")


def _schema(value: Mapping[str, Any], expected: str, context: str) -> None:
    if value.get("schema_version") != expected:
        raise ValueError(f"{context}.schema_version must be {expected!r}.")


def _nonempty(value: Any, context: str) -> str:
    result = _string(value, context).strip()
    if not result:
        raise ValueError(f"{context} must not be empty.")
    return result


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string.")
    return value


def _status(value: Any, context: str) -> str:
    if value not in ASSESSMENT_STATUSES:
        raise ValueError(f"{context} must be one of {sorted(ASSESSMENT_STATUSES)}.")
    return str(value)


def _bool_mapping(value: Any, context: str) -> dict[str, bool]:
    result = _mapping(value, context)
    if any(
        not isinstance(key, str) or not isinstance(item, bool)
        for key, item in result.items()
    ):
        raise TypeError(f"{context} must map strings to booleans.")
    return result


def _json_safe(value: Any, context: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain strict JSON data.") from exc
