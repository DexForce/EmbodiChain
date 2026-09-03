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

"""Deterministic TaskCandidate lowering to provider-free Semantic Calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Final

from .contracts import TaskCandidate, validate_task_candidate
from .orchestration.contracts import RoleBindings, validate_role_bindings
from .semantic_graph import SemanticTaskGraph, validate_semantic_task_graph

__all__ = ["SemanticTaskPlanner", "UnsupportedSemanticCapabilityError"]

_COORDINATED_TRANSPORT_CALL_ID: Final = "simulation.coordinated_transport"
_AXIS_ALIGN_CALL_ID: Final = "simulation.axis_align"
_PARK_CALL_ID: Final = "simulation.park"
_PLACE_RELATIVE_CALL_ID: Final = "simulation.place_relative"


class UnsupportedSemanticCapabilityError(ValueError):
    """Raised when a TaskSpec has no executable canonical Semantic Call route."""


class SemanticTaskPlanner:
    """Lower Task Engine ontology steps into one immutable semantic task DAG."""

    def plan(
        self,
        candidate: Mapping[str, Any],
        role_bindings: Mapping[str, Any],
        scene_objects: Sequence[Mapping[str, Any]],
        *,
        planner_route: str = "offline",
        integration_fingerprint: str = "0" * 64,
    ) -> SemanticTaskGraph:
        """Build one semantic graph without importing or materializing actions.

        Args:
            candidate: Valid selected TaskCandidate.
            role_bindings: Canonical scene IDs selected by Scene Adapter.
            scene_objects: Provider-free normalized scene metadata.
            planner_route: Candidate route provenance.
            integration_fingerprint: Exact integration fingerprint, or the
                all-zero placeholder used before bundle preflight.

        Returns:
            A validated ``semantic_task_graph/v1`` value.
        """
        selected: TaskCandidate = validate_task_candidate(candidate)
        bindings: RoleBindings = validate_role_bindings(role_bindings)
        if bindings["task_id"] != selected["draft"]["task_id"]:
            raise ValueError("RoleBindings.task_id must match the TaskCandidate.")
        if bindings["candidate_id"] != selected["candidate_id"]:
            raise ValueError("RoleBindings.candidate_id must match the TaskCandidate.")

        objects = {
            str(item.get("runtime_uid", item.get("uid", ""))): deepcopy(dict(item))
            for item in scene_objects
            if str(item.get("runtime_uid", item.get("uid", ""))).strip()
        }
        steps = selected["draft"]["steps"]
        steps_by_id = {str(step["id"]): step for step in steps}
        result_objects: dict[str, str] = {}
        held_by: dict[str, str] = {}
        targets: dict[str, Any] = {}
        nodes: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        group_terminal: dict[str, str] = {}

        for step in steps:
            step_id = str(step["id"])
            task_type = str(step["task_type"])
            object_id = self._resolve_step_entity(
                step,
                role="object",
                bindings=bindings,
                result_objects=result_objects,
                steps_by_id=steps_by_id,
            )
            calls: list[dict[str, Any]]
            cleanup_resources: tuple[str, ...] = ()
            if task_type == "E2":
                if str(step.get("orientation_goal")) != "upright":
                    raise UnsupportedSemanticCapabilityError(
                        f"Step {step_id!r} E2 currently requires "
                        "orientation_goal='upright'."
                    )
                requested = str(step.get("required_arm", "auto"))
                resource = (
                    self._nearest_resource(object_id, objects)
                    if requested == "auto"
                    else _resource(requested, field="required_arm")
                )
                calls = [
                    {
                        "kind": "registered",
                        "call_id": _AXIS_ALIGN_CALL_ID,
                        "arguments": {"object": object_id},
                        "resources": {"primary": resource},
                    },
                    _relative_place_call(
                        object_id,
                        reference_id="table",
                        relation="on",
                        resource=resource,
                    ),
                ]
                held_by.pop(object_id, None)
                cleanup_resources = (resource,)
            elif task_type == "E4":
                source = _resource(step.get("transfer_arm"), field="transfer_arm")
                destination = _resource(step.get("receive_arm"), field="receive_arm")
                calls = [
                    {
                        "kind": "pick",
                        "object": object_id,
                        "resources": {"primary": source},
                    },
                    {
                        "kind": "hand_over",
                        "object": object_id,
                        "resources": {"source": source, "destination": destination},
                    },
                ]
                held_by[object_id] = destination
                # HandOver already clears and retreats the source participant.
                # Parking it as a separate call delays the receiver's next
                # operation while a friction-held object can drift, and also
                # leaves the following Place with a stale attachment transform.
                # Defer parking until that resource next owns explicit work.
                cleanup_resources = ()
            elif task_type == "E1":
                target_id = self._resolve_step_entity(
                    step,
                    role="target",
                    bindings=bindings,
                    result_objects=result_objects,
                    steps_by_id=steps_by_id,
                )
                requested = str(step.get("required_arm", "auto"))
                resource = (
                    self._nearest_resource(object_id, objects)
                    if requested == "auto"
                    else _resource(requested, field="required_arm")
                )
                calls = []
                if held_by.get(object_id) != resource:
                    if object_id in held_by:
                        raise UnsupportedSemanticCapabilityError(
                            f"Step {step_id!r} requires {resource!r} while {object_id!r} "
                            f"is held by {held_by[object_id]!r}; add an explicit hand_over."
                        )
                    calls.append(
                        {
                            "kind": "pick",
                            "object": object_id,
                            "resources": {"primary": resource},
                        }
                    )
                relation = str(step.get("relation", "none"))
                if relation == "inside":
                    placement: dict[str, Any] = {
                        "inside": _inside_affordance(target_id, object_id)
                    }
                elif relation in {
                    "above",
                    "behind",
                    "front_of",
                    "left_of",
                    "on",
                    "right_of",
                }:
                    calls.append(
                        _relative_place_call(
                            object_id,
                            reference_id=target_id,
                            relation=relation,
                            resource=resource,
                        )
                    )
                    held_by.pop(object_id, None)
                    cleanup_resources = (resource,)
                    placement = None
                else:
                    raise UnsupportedSemanticCapabilityError(
                        f"Step {step_id!r} E1 relation {relation!r} has no canonical "
                        "Semantic Call route."
                    )
                if placement is not None:
                    calls.append(
                        {
                            "kind": "place",
                            "object": object_id,
                            **placement,
                            "resources": {"primary": resource},
                        }
                    )
                held_by.pop(object_id, None)
                cleanup_resources = (resource,)
            elif task_type == "E5":
                if str(step.get("terminal_behavior")) != "place":
                    raise UnsupportedSemanticCapabilityError(
                        f"Step {step_id!r} E5 currently requires terminal_behavior='place'."
                    )
                calls = [
                    {
                        "kind": "registered",
                        "call_id": _COORDINATED_TRANSPORT_CALL_ID,
                        "arguments": {
                            "object": object_id,
                            "target": f"{object_id}_forward",
                        },
                        "resources": {"left": "left", "right": "right"},
                    }
                ]
                held_by.pop(object_id, None)
                cleanup_resources = ("left", "right")
            else:
                raise UnsupportedSemanticCapabilityError(
                    f"Task type {task_type!r} has no phase-one Semantic Call route."
                )

            call_roles = [(call, "primary") for call in calls]
            call_roles.extend(
                (_park_call(resource), "cleanup") for resource in cleanup_resources
            )
            dependencies = [str(value) for value in step["depends_on"]]
            first_dependencies = [group_terminal[value] for value in dependencies]
            group_node_ids: list[str] = []
            previous: str | None = None
            for call_index, (call, role) in enumerate(call_roles, start=1):
                node_id = f"{step_id}__call_{call_index:02d}"
                node_dependencies = (
                    [previous] if previous is not None else first_dependencies
                )
                nodes.append(
                    {
                        "id": node_id,
                        "call": call,
                        "depends_on": node_dependencies,
                        "task_instance_id": step_id,
                        "task_type": task_type,
                        "role": role,
                    }
                )
                group_node_ids.append(node_id)
                previous = node_id
            assert previous is not None
            group_terminal[step_id] = previous
            groups.append(
                {
                    "id": step_id,
                    "task_type": task_type,
                    "node_ids": group_node_ids,
                    "depends_on": dependencies,
                    "success": _step_success(selected["success_spec"], step_id),
                }
            )
            result_objects[step_id] = object_id

        return validate_semantic_task_graph(
            {
                "schema_version": "semantic_task_graph/v1",
                "task_id": selected["draft"]["task_id"],
                "instruction": selected["draft"]["instruction"],
                "planner_route": str(planner_route),
                "integration_fingerprint": str(integration_fingerprint),
                "targets": targets,
                "nodes": nodes,
                "task_groups": groups,
                "success": {
                    "kind": "all_task_groups",
                    "source": deepcopy(selected["success_spec"]),
                },
            }
        )

    def _resolve_step_entity(
        self,
        step: Mapping[str, Any],
        *,
        role: str,
        bindings: RoleBindings,
        result_objects: Mapping[str, str],
        steps_by_id: Mapping[str, Mapping[str, Any]],
    ) -> str:
        reference = step[role]
        kind = str(reference["kind"])
        if kind == "scene_ref":
            reference_id = f"{step['id']}.{role}"
            values = bindings["reference_bindings"].get(reference_id)
            if not values or len(values) != 1:
                raise ValueError(
                    f"{reference_id} must resolve to exactly one scene entity."
                )
            return str(values[0])
        if kind == "step_result":
            source_step = str(reference["step_id"])
            if source_step not in steps_by_id or source_step not in result_objects:
                raise ValueError(
                    f"Step {step['id']!r} references unavailable result {source_step!r}."
                )
            return result_objects[source_step]
        raise UnsupportedSemanticCapabilityError(
            f"Step {step['id']!r} {role} reference kind {kind!r} is unsupported."
        )

    def _nearest_resource(
        self,
        object_id: str,
        objects: Mapping[str, Mapping[str, Any]],
    ) -> str:
        item = objects.get(object_id)
        if item is None:
            raise ValueError(f"Scene metadata is missing object {object_id!r}.")
        position = item.get("init_pos")
        if not isinstance(position, Sequence) or len(position) != 3:
            raise ValueError(f"Scene object {object_id!r} has no three-value init_pos.")
        # The canonical dual-Franka embodiment faces world -X: its right arm
        # base is on +Y and its left arm base is on -Y.
        return "right" if float(position[1]) >= 0.0 else "left"


def _resource(value: Any, *, field: str) -> str:
    normalized = str(value).strip()
    mapping = {
        "left": "left",
        "left_arm": "left",
        "right": "right",
        "right_arm": "right",
    }
    if normalized not in mapping:
        raise UnsupportedSemanticCapabilityError(
            f"{field}={normalized!r} is not a dual-Franka semantic resource."
        )
    return mapping[normalized]


def _park_call(resource: str) -> dict[str, Any]:
    """Build one payload-free semantic cleanup call for a logical resource."""
    return {
        "kind": "registered",
        "call_id": _PARK_CALL_ID,
        "arguments": {},
        "resources": {"primary": resource},
    }


def _relative_place_call(
    object_id: str,
    *,
    reference_id: str,
    relation: str,
    resource: str,
) -> dict[str, Any]:
    """Build one provider-free object relation for JIT runtime grounding."""
    return {
        "kind": "registered",
        "call_id": _PLACE_RELATIVE_CALL_ID,
        "arguments": {
            "object": object_id,
            "reference": reference_id,
            "relation": relation,
        },
        "resources": {"primary": resource},
    }


def _inside_affordance(container_id: str, object_id: str) -> str:
    return f"inside__{container_id}__{object_id}"


def _step_success(success_spec: Mapping[str, Any], step_id: str) -> dict[str, Any]:
    term = next(
        (
            deepcopy(dict(item))
            for item in success_spec["terms"]
            if item["step_id"] == step_id
        ),
        None,
    )
    if term is None:
        raise ValueError(f"SuccessSpec has no term for TaskGroup {step_id!r}.")
    return {"kind": "semantic_task_term", **term}
