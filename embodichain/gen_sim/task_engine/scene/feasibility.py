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
import math
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

        checks.extend(self._workspace_checks(steps, bindings, objects))

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
                "remediation_class": _remediation_class(checks),
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
        if (
            str(request.get("role")) == "target"
            and str(request.get("source_structure")) == "physical_entity"
        ):
            checks.append(
                _check(
                    "placement_support",
                    subject,
                    "runtime_probe",
                    "Support depends on the payload, candidate pose, live geometry, "
                    "and post-release stability.",
                    evidence={
                        "runtime_obligations": [
                            "placement_candidates",
                            "object_supported_by",
                            "stable_for",
                            "final_support_revalidation",
                        ]
                    },
                )
            )
        return checks

    def _workspace_checks(
        self,
        steps: Mapping[str, Mapping[str, Any]],
        bindings: Mapping[str, Any],
        objects: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Defer arm-side compatibility to the live robot frame."""
        checks: list[dict[str, Any]] = []
        object_uids_by_step: dict[str, tuple[str, ...]] = {}
        phases: list[dict[str, Any]] = []
        for step_id, step in steps.items():
            object_uids = _step_selector_uids(
                step_id,
                "object",
                step.get("object"),
                bindings,
                object_uids_by_step,
            )
            object_uids_by_step[step_id] = object_uids
            target_uids = _step_selector_uids(
                step_id,
                "target",
                step.get("target"),
                bindings,
                object_uids_by_step,
            )
            task_type = str(step.get("task_type", ""))
            required_arm = str(step.get("required_arm", "auto"))
            if task_type == "E4":
                required_arm = str(step.get("transfer_arm", "none"))
            if required_arm in {"left_arm", "right_arm"}:
                for uid in object_uids:
                    entity = objects.get(uid)
                    position = (
                        entity.get("initial_pose", {}).get("position", ())
                        if isinstance(entity, Mapping)
                        and isinstance(entity.get("initial_pose"), Mapping)
                        else ()
                    )
                    if (
                        not isinstance(position, Sequence)
                        or isinstance(position, (str, bytes, bytearray))
                        or len(position) < 2
                    ):
                        continue
                    checks.append(
                        _check(
                            "arm_layout_risk",
                            f"{step_id}:{uid}",
                            "runtime_probe",
                            "Arm-side compatibility requires live left/right arm-base "
                            "poses and workspace geometry.",
                            evidence={
                                "required_arm": required_arm,
                                "object_world_position": [
                                    float(position[0]),
                                    float(position[1]),
                                ],
                                "arm_side_frame": "live_robot",
                                "mismatch_risk": None,
                                "geometry_certificate": False,
                            },
                        )
                    )

            phases.extend(
                _workflow_phases(
                    step_id,
                    task_type,
                    object_uids,
                    target_uids,
                    transfer_arm=str(step.get("transfer_arm", "none")),
                    receive_arm=str(step.get("receive_arm", "none")),
                )
            )
        if phases:
            checks.append(
                _check(
                    "task_workspace",
                    "task_workflow",
                    "runtime_probe",
                    "Scene layout must satisfy pickup, transfer, placement, and "
                    "safety-clearance phases across the complete task workflow.",
                    evidence={
                        "arm_side_frame": "live_robot",
                        "phases": phases,
                        "geometry_certificate": False,
                    },
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
        if expected in {"scene_entity", "spatial_reference"}:
            if role in {"camera", "light", "robot", "sensor"}:
                return _check(
                    "structure",
                    subject,
                    "contradicted",
                    f"Scene entity role {role!r} cannot be a spatial action target.",
                    evidence={"static_pose": _has_static_pose(entity)},
                )
            if not _has_static_pose(entity):
                return _check(
                    "structure",
                    subject,
                    "unknown",
                    "Static scene evidence does not provide a finite spatial pose.",
                    evidence={"static_pose": False},
                )
            if role == "articulation":
                return _check(
                    "structure",
                    subject,
                    "runtime_probe",
                    "Articulation has a static pose, but live spatial target lookup "
                    "must be validated at runtime.",
                    evidence={
                        "static_pose": True,
                        "runtime_entity_kind": "articulation",
                    },
                )
            has_runtime_body = bool(entity.get("physics"))
            if (
                role
                in {
                    "background",
                    "object",
                    "rigid_object",
                    "support_surface",
                    "table",
                }
                and has_runtime_body
            ):
                return _check(
                    "structure",
                    subject,
                    "proven",
                    "Scene entity has a static pose and a rigid runtime body.",
                    evidence={
                        "static_pose": True,
                        "runtime_entity_kind": "rigid_object",
                    },
                )
            return _check(
                "structure",
                subject,
                "runtime_probe",
                "Scene entity has a static pose, but its live target interface is "
                "not proven by the static manifest.",
                evidence={"static_pose": True, "runtime_entity_kind": "unknown"},
            )
        if expected == "physical_entity":
            geometry = entity.get("geometry", {})
            shape = geometry.get("shape", {}) if isinstance(geometry, Mapping) else {}
            asset_sha256 = (
                geometry.get("asset_sha256", "")
                if isinstance(geometry, Mapping)
                else ""
            )
            physics = entity.get("physics", {})
            articulation = entity.get("articulation", {})
            has_physical_geometry = bool(shape) or bool(asset_sha256)
            has_runtime_body = bool(physics) or bool(articulation)
            if role in {"camera", "light", "sensor"}:
                return _check(
                    "structure",
                    subject,
                    "contradicted",
                    f"Scene entity role {role!r} is not a physical collision body.",
                    evidence={"physical_geometry": False, "runtime_body": False},
                )
            if role == "articulation" or bool(articulation):
                return _check(
                    "structure",
                    subject,
                    "contradicted",
                    "Placement on an articulation requires a link-level target "
                    "interface that the current runtime does not provide.",
                    evidence={
                        "physical_geometry": has_physical_geometry,
                        "runtime_body": bool(articulation),
                        "runtime_entity_kind": "articulation",
                        "runtime_target_interface": False,
                    },
                )
            if has_physical_geometry and has_runtime_body:
                return _check(
                    "structure",
                    subject,
                    "proven",
                    "Scene entity has physical geometry and a runtime body.",
                    evidence={"physical_geometry": True, "runtime_body": True},
                )
            return _check(
                "structure",
                subject,
                "unknown",
                "Static scene evidence does not prove physical geometry and a "
                "runtime body required for placement.",
                evidence={
                    "physical_geometry": has_physical_geometry,
                    "runtime_body": has_runtime_body,
                },
            )
        accepted_by_structure = {
            "articulation": {"articulation"},
            "rigid_object": {"object", "rigid_object"},
            "movable": {"object", "rigid_object"},
            "support_surface": {"background", "support_surface", "table"},
        }
        accepted = accepted_by_structure.get(expected)
        if accepted is None:
            return _check(
                "structure",
                subject,
                "unknown",
                f"Structure contract {expected!r} is not recognized by the broker.",
                evidence={"scene_role": role},
            )
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


def _has_static_pose(entity: Mapping[str, Any]) -> bool:
    pose = entity.get("initial_pose")
    if not isinstance(pose, Mapping):
        return False
    position = pose.get("position")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes, bytearray))
        or len(position) != 3
    ):
        return False
    return all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for value in position
    )


def _step_selector_uids(
    step_id: str,
    role: str,
    selector: Any,
    bindings: Mapping[str, Any],
    object_uids_by_step: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Resolve direct and prior-step selectors for static workspace advice."""
    raw = bindings.get(f"{step_id}.{role}", ())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        direct = tuple(str(uid) for uid in raw if str(uid))
        if direct:
            return direct
    if not isinstance(selector, Mapping) or selector.get("kind") != "step_result":
        return ()
    source_step = str(selector.get("step_id", ""))
    return tuple(object_uids_by_step.get(source_step, ()))


def _workflow_phases(
    step_id: str,
    task_type: str,
    object_uids: Sequence[str],
    target_uids: Sequence[str],
    *,
    transfer_arm: str,
    receive_arm: str,
) -> list[dict[str, Any]]:
    """Describe whole-task layout anchors without inventing geometry bounds."""
    phases: list[dict[str, Any]] = []
    if object_uids:
        phases.append(
            {
                "step_id": step_id,
                "phase": "pickup",
                "object_uids": list(object_uids),
            }
        )
    if task_type == "E4":
        phases.append(
            {
                "step_id": step_id,
                "phase": "handover_shared_workspace",
                "object_uids": list(object_uids),
                "transfer_arm": transfer_arm,
                "receive_arm": receive_arm,
            }
        )
    if target_uids:
        phases.append(
            {
                "step_id": step_id,
                "phase": "target_interaction",
                "object_uids": list(object_uids),
                "target_uids": list(target_uids),
            }
        )
    if task_type in {"E1", "E2", "E3", "E4", "E5"}:
        phases.append(
            {
                "step_id": step_id,
                "phase": "safety_clearance",
                "object_uids": list(object_uids),
            }
        )
    return phases


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


def _remediation_class(checks: Sequence[Mapping[str, Any]]) -> str:
    """Classify contradictions by the subsystem capable of changing them."""
    contradicted = [check for check in checks if check.get("status") == "contradicted"]
    if not contradicted:
        return "none"
    kinds = {str(check.get("kind", "")) for check in contradicted}
    if kinds.intersection({"task_capability", "atomic_capability"}):
        return "action_capability"
    # A new materialization seed can change observed pose/orientation, but it
    # cannot change task semantics, entity roles, bindings, or declared affordances.
    if kinds <= {"initial_state"}:
        return "scene_remediable"
    if kinds.intersection({"binding", "structure", "affordance", "attributes"}):
        return "input_conflict"
    return "terminal"


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
