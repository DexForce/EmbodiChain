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

"""Deterministic task, scene, robot, and action-capability intersection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    ASSESSMENT_STATUSES,
    FEASIBILITY_REPORT_SCHEMA,
    FeasibilityReport,
    validate_feasibility_report,
    validate_static_scene_manifest,
)

__all__ = ["FeasibilityBroker"]


_STATUS_PRIORITY = {
    "proven": 0,
    "runtime_probe": 1,
    "unknown": 2,
    "contradicted": 3,
}


class FeasibilityBroker:
    """Produce an auditable compatibility report without repairing inputs."""

    def assess(
        self,
        candidate: Mapping[str, Any],
        role_bindings: Mapping[str, Any],
        scene_manifest: Mapping[str, Any],
        *,
        capability_catalog: Mapping[str, Mapping[str, Any]],
        task_actions: Mapping[str, Sequence[str]],
    ) -> FeasibilityReport:
        """Assess one grounded candidate against static and runtime capabilities."""
        manifest = validate_static_scene_manifest(scene_manifest)
        draft = _mapping(candidate.get("draft"), "candidate.draft")
        scene_request = _mapping(
            candidate.get("scene_request"), "candidate.scene_request"
        )
        bindings = role_bindings.get("reference_bindings", role_bindings)
        bindings = _mapping(bindings, "role_bindings.reference_bindings")
        objects = {item["uid"]: item for item in manifest["objects"]}
        steps = {
            str(item["id"]): item
            for item in _sequence(draft.get("steps"), "candidate.draft.steps")
        }
        checks: list[dict[str, Any]] = []

        for step_id, step in steps.items():
            task_type = str(step.get("task_type", ""))
            actions = task_actions.get(task_type)
            if not actions:
                checks.append(
                    _check(
                        "task_capability",
                        step_id,
                        "contradicted",
                        f"Task type {task_type!r} has no registered action recipe.",
                    )
                )
                continue
            for action_name in actions:
                capability = capability_catalog.get(str(action_name))
                if capability is None:
                    checks.append(
                        _check(
                            "atomic_capability",
                            f"{step_id}:{action_name}",
                            "contradicted",
                            f"AtomicAction {action_name!r} is not registered.",
                        )
                    )
                elif not bool(capability.get("runtime_available", False)):
                    checks.append(
                        _check(
                            "atomic_capability",
                            f"{step_id}:{action_name}",
                            "contradicted",
                            str(
                                capability.get("unavailable_reason")
                                or "Action is planning-only."
                            ),
                            evidence={"action": str(action_name)},
                        )
                    )
                else:
                    checks.append(
                        _check(
                            "atomic_capability",
                            f"{step_id}:{action_name}",
                            "proven",
                            "AtomicAction is registered and executable.",
                            evidence={"action": str(action_name)},
                        )
                    )

        for request in _sequence(
            scene_request.get("references"), "candidate.scene_request.references"
        ):
            reference_id = str(request.get("reference_id", ""))
            raw_uids = bindings.get(reference_id, ())
            if not isinstance(raw_uids, Sequence) or isinstance(raw_uids, (str, bytes)):
                raw_uids = ()
            uids = [str(uid) for uid in raw_uids]
            if not uids:
                checks.append(
                    _check(
                        "binding",
                        reference_id,
                        "contradicted",
                        "Reference has no grounded scene entity.",
                    )
                )
                continue
            for uid in uids:
                entity = objects.get(uid)
                if entity is None:
                    checks.append(
                        _check(
                            "binding",
                            f"{reference_id}:{uid}",
                            "contradicted",
                            "Binding references an entity absent from the static manifest.",
                        )
                    )
                    continue
                checks.extend(self._entity_checks(request, entity, reference_id))

        statuses = Counter(check["status"] for check in checks)
        status = max(
            (check["status"] for check in checks),
            key=_STATUS_PRIORITY.__getitem__,
            default="unknown",
        )
        blockers = sorted(
            {
                f"{check['subject']}: {check['reason']}"
                for check in checks
                if check["status"] == "contradicted"
            }
        )
        return validate_feasibility_report(
            {
                "schema_version": FEASIBILITY_REPORT_SCHEMA,
                "task_id": str(draft.get("task_id", "")),
                "candidate_id": str(candidate.get("candidate_id", "")),
                "scene_id": manifest["scene_id"],
                "status": status,
                "checks": checks,
                "blockers": blockers,
                "summary": {
                    name: int(statuses.get(name, 0))
                    for name in sorted(ASSESSMENT_STATUSES)
                },
            }
        )

    def _entity_checks(
        self,
        request: Mapping[str, Any],
        entity: Mapping[str, Any],
        reference_id: str,
    ) -> list[dict[str, Any]]:
        uid = str(entity["uid"])
        subject = f"{reference_id}:{uid}"
        checks = [self._structure_check(request, entity, subject)]
        evidence_by_type: dict[str, list[Mapping[str, Any]]] = {}
        for evidence in entity["affordances"]:
            evidence_by_type.setdefault(str(evidence["type"]), []).append(evidence)
        for affordance in request.get("affordances", ()):
            name = str(affordance)
            checks.append(
                self._affordance_check(name, evidence_by_type.get(name, ()), subject)
            )
        for field_name in ("initial_state", "attributes"):
            required = request.get(field_name, {})
            actual = entity.get(field_name, {})
            if isinstance(required, Mapping) and isinstance(actual, Mapping):
                for key, expected in required.items():
                    if key not in actual:
                        status = "unknown"
                        reason = f"Required {field_name} field {key!r} is not declared."
                    elif actual[key] != expected:
                        status = "contradicted"
                        reason = f"Required {field_name} field {key!r} conflicts with the scene."
                    else:
                        status = "proven"
                        reason = f"Required {field_name} field {key!r} matches."
                    checks.append(
                        _check(
                            field_name,
                            subject,
                            status,
                            reason,
                            evidence={"field": str(key)},
                        )
                    )
        if str(request.get("role")) == "object":
            checks.append(
                _check(
                    "runtime_reachability",
                    subject,
                    "runtime_probe",
                    "Reachability, collision, and grasp geometry require live planning.",
                )
            )
        return checks

    @staticmethod
    def _structure_check(
        request: Mapping[str, Any],
        entity: Mapping[str, Any],
        subject: str,
    ) -> dict[str, Any]:
        expected = str(request.get("source_structure", ""))
        role = str(entity.get("role", ""))
        accepted = {
            "articulation": {"articulation"},
            "rigid_object": {"object", "rigid_object"},
            "movable": {"object", "rigid_object"},
            "support_surface": {"background", "support_surface", "table"},
        }.get(expected, {expected})
        if role in accepted:
            return _check(
                "structure",
                subject,
                "proven",
                f"Scene role {role!r} satisfies structure {expected!r}.",
            )
        return _check(
            "structure",
            subject,
            "contradicted",
            f"Scene role {role!r} does not satisfy structure {expected!r}.",
        )

    @staticmethod
    def _affordance_check(
        name: str,
        evidence: Sequence[Mapping[str, Any]],
        subject: str,
    ) -> dict[str, Any]:
        if not evidence:
            return _check(
                "affordance",
                subject,
                "unknown",
                f"Affordance {name!r} has no evidence.",
                evidence={"affordance": name},
            )
        statuses = {str(item.get("status")) for item in evidence}
        if statuses == {"contradicted"}:
            status = "contradicted"
            reason = f"Affordance {name!r} is explicitly contradicted."
        elif "verified" in statuses:
            status = "proven"
            reason = f"Affordance {name!r} has verified evidence."
        else:
            status = "runtime_probe"
            reason = (
                f"Affordance {name!r} is declared but requires physical validation."
            )
        return _check(
            "affordance",
            subject,
            status,
            reason,
            evidence={
                "affordance": name,
                "sources": sorted({str(item.get("source")) for item in evidence}),
            },
        )


def _check(
    kind: str,
    subject: str,
    status: str,
    reason: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "subject": subject,
        "status": status,
        "reason": reason,
        "evidence": dict(evidence or {}),
    }


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    return dict(value)


def _sequence(value: Any, context: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence.")
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError(f"{context} must contain mappings.")
    return list(value)
