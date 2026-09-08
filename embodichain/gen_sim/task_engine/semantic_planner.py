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
import math
from typing import Any, Final

from .contracts import TaskCandidate, validate_task_candidate
from .orchestration.contracts import RoleBindings, validate_role_bindings
from .semantic_graph import SemanticTaskGraph, validate_semantic_task_graph

__all__ = ["SemanticTaskPlanner", "UnsupportedSemanticCapabilityError"]

_COORDINATED_TRANSPORT_CALL_ID: Final = "simulation.coordinated_transport"
_COORDINATED_HOLD_CALL_ID: Final = "simulation.coordinated_hold"
_PARK_CALL_ID: Final = "simulation.park"
_PLACE_RELATIVE_CALL_ID: Final = "simulation.place_relative"
_STACK_PLACE_CALL_ID: Final = "gen_sim.stack_place"
_STACK_PICK_CALL_ID: Final = "gen_sim.stack_pick"
_MOVE_HELD_OBJECT_CALL_ID: Final = "simulation.move_held_object"
_ALIGN_HELD_CALL_ID: Final = "gen_sim.align_held"
_CLEAR_RELEASED_CALL_ID: Final = "gen_sim.clear_released"
_PICK_CALL_ID: Final = "simulation.pick"
_POUR_CALL_ID: Final = "simulation.pour"


class UnsupportedSemanticCapabilityError(ValueError):
    """Raised when a TaskSpec has no executable canonical Semantic Call route."""


class SemanticTaskPlanner:
    """Lower Task Engine ontology steps into one immutable semantic task DAG.

    Args:
        lateral_relation_distance: World-frame offset used for ``right_of``.
        front_relation_distance: World-frame offset used for ``front_of``.

    Raises:
        ValueError: If either relation distance is not positive.
    """

    def __init__(
        self,
        *,
        lateral_relation_distance: float = 0.10,
        front_relation_distance: float = 0.18,
    ) -> None:
        for field_name, value in (
            ("lateral_relation_distance", lateral_relation_distance),
            ("front_relation_distance", front_relation_distance),
        ):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{field_name} must be positive.")
        self.lateral_relation_distance = float(lateral_relation_distance)
        self.front_relation_distance = float(front_relation_distance)

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
        upright_objects: set[str] = set()
        upright_staging_targets: dict[str, str] = {}
        targets: dict[str, Any] = {}
        nodes: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        group_terminal: dict[str, str] = {}

        for step in steps:
            step_id = str(step["id"])
            task_type = str(step["task_type"])
            if task_type not in {"E1", "E2", "E3", "E4", "E5"}:
                raise UnsupportedSemanticCapabilityError(
                    f"Task Engine currently supports only E1-E5, not {task_type}."
                )
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
                    held_by.get(object_id) or self._nearest_resource(object_id, objects)
                    if requested in {"auto", "none"}
                    else _resource(requested, field="required_arm")
                )
                target_name = f"{step_id}_upright_target"
                staging_target_name = f"{step_id}_upright_staging_target"
                upright_position = self._upright_target_position(
                    object_id,
                    objects,
                )
                targets.update(_upright_targets(step_id, upright_position))
                calls = []
                if held_by.get(object_id) != resource:
                    if object_id in held_by:
                        raise UnsupportedSemanticCapabilityError(
                            f"Step {step_id!r} requires {resource!r} while "
                            f"{object_id!r} is held by {held_by[object_id]!r}."
                        )
                    calls.append(
                        {
                            "kind": "registered",
                            "call_id": _PICK_CALL_ID,
                            "arguments": {
                                "object": object_id,
                                "target": target_name,
                            },
                            "resources": {"primary": resource},
                        }
                    )
                calls.extend(
                    (
                        {
                            "kind": "registered",
                            "call_id": _ALIGN_HELD_CALL_ID,
                            "arguments": {
                                "object": object_id,
                                "target": staging_target_name,
                                "preserve_yaw": False,
                            },
                            "resources": {"primary": resource},
                        },
                        {
                            "kind": "registered",
                            "call_id": _PLACE_RELATIVE_CALL_ID,
                            "arguments": {
                                "object": object_id,
                                "reference": "table",
                                "relation": "on",
                            },
                            "resources": {"primary": resource},
                        },
                    )
                )
                calls.insert(
                    -1,
                    {
                        "kind": "registered",
                        "call_id": _ALIGN_HELD_CALL_ID,
                        "arguments": {
                            "object": object_id,
                            "target": staging_target_name,
                            "preserve_yaw": True,
                        },
                        "resources": {"primary": resource},
                    },
                )
                held_by.pop(object_id, None)
                objects[object_id]["init_pos"] = upright_position
                upright_objects.add(object_id)
                upright_staging_targets[object_id] = staging_target_name
                cleanup_resources = (resource,)
            elif task_type == "E3":
                target_id = self._resolve_step_entity(
                    step,
                    role="target",
                    bindings=bindings,
                    result_objects=result_objects,
                    steps_by_id=steps_by_id,
                )
                requested = str(step.get("required_arm", "auto"))
                resource = (
                    held_by.get(object_id) or self._nearest_resource(object_id, objects)
                    if requested in {"auto", "none"}
                    else _resource(requested, field="required_arm")
                )
                calls = []
                if held_by.get(object_id) != resource:
                    if object_id in held_by:
                        raise UnsupportedSemanticCapabilityError(
                            f"Step {step_id!r} Pour requires {resource!r} while "
                            f"{object_id!r} is held by {held_by[object_id]!r}."
                        )
                    calls.append(
                        {
                            "kind": "pick",
                            "object": object_id,
                            "resources": {"primary": resource},
                        }
                    )
                pour_target = f"{step_id}_pour_target"
                return_target = f"{step_id}_return_target"
                targets[return_target] = {
                    "kind": "cyclic_pose",
                    "values": [
                        {
                            "position": [
                                float(value) for value in objects[object_id]["init_pos"]
                            ],
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        }
                    ],
                }
                calls.extend(
                    (
                        {
                            "kind": "registered",
                            "call_id": _MOVE_HELD_OBJECT_CALL_ID,
                            "arguments": {
                                "object": object_id,
                                "target": pour_target,
                                "reference": target_id,
                            },
                            "resources": {"primary": resource},
                        },
                        {
                            "kind": "registered",
                            "call_id": _POUR_CALL_ID,
                            "arguments": {"object": object_id},
                            "resources": {"primary": resource},
                        },
                        {
                            "kind": "place",
                            "object": object_id,
                            "at": {"kind": "target_ref", "target": return_target},
                            "resources": {"primary": resource},
                        },
                    )
                )
                held_by.pop(object_id, None)
                cleanup_resources = (resource,)
            elif task_type == "E4":
                source = _resource(step.get("transfer_arm"), field="transfer_arm")
                destination = _resource(step.get("receive_arm"), field="receive_arm")
                calls = []
                if held_by.get(object_id) != source:
                    if object_id in held_by:
                        raise UnsupportedSemanticCapabilityError(
                            f"Step {step_id!r} HandOver requires {source!r} but "
                            f"{object_id!r} is held by {held_by[object_id]!r}."
                        )
                    calls.append(
                        {
                            "kind": "pick",
                            "object": object_id,
                            "resources": {"primary": source},
                        }
                    )
                calls.append(
                    {
                        "kind": "hand_over",
                        "object": object_id,
                        "resources": {"source": source, "destination": destination},
                    }
                )
                held_by[object_id] = destination
                # HandOver already retreats the source; do not delay a held receiver.
                cleanup_resources = ()
                if str(step.get("terminal_behavior")) == "place":
                    target_id = self._resolve_step_entity(
                        step,
                        role="target",
                        bindings=bindings,
                        result_objects=result_objects,
                        steps_by_id=steps_by_id,
                    )
                    place_call = self._placement_call(
                        step_id=step_id,
                        object_id=object_id,
                        target_id=target_id,
                        relation=str(step.get("relation", "none")),
                        resource=destination,
                    )
                    calls.append(place_call)
                    held_by.pop(object_id, None)
                    cleanup_resources = (source, destination)
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
                    held_by.get(object_id) or self._nearest_resource(object_id, objects)
                    if requested in {"auto", "none"}
                    else _resource(requested, field="required_arm")
                )
                relation = str(step.get("relation", "none"))
                calls = []
                if (
                    held_by.get(object_id) == resource
                    and relation in {"on", "above"}
                    and object_id in upright_objects
                    and target_id in upright_objects
                ):
                    # A low receiving grasp can intersect the support during
                    # stacking. Expose the regrasp as ordinary semantic calls.
                    calls.append(
                        self._placement_call(
                            step_id=step_id,
                            object_id=object_id,
                            target_id="table",
                            relation="on",
                            resource=resource,
                        )
                    )
                    held_by.pop(object_id)
                    other_resource = "left" if resource == "right" else "right"
                    if other_resource not in held_by.values():
                        # Clear the former source hand only after the receiver
                        # has safely staged its object, not during a live hold.
                        calls.append(_park_call(other_resource))
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
                place_call = self._placement_call(
                    step_id=step_id,
                    object_id=object_id,
                    target_id=target_id,
                    relation=relation,
                    resource=resource,
                )
                if (
                    relation in {"on", "above"}
                    and object_id in upright_objects
                    and target_id in upright_objects
                ):
                    if calls and calls[-1]["kind"] == "pick":
                        calls[-1] = {
                            "kind": "registered",
                            "call_id": _STACK_PICK_CALL_ID,
                            "arguments": {"object": object_id, "target": target_id},
                            "resources": {"primary": resource},
                        }
                    place_call["call_id"] = _STACK_PLACE_CALL_ID
                    calls.append(
                        {
                            "kind": "registered",
                            "call_id": _ALIGN_HELD_CALL_ID,
                            "arguments": {
                                "object": object_id,
                                "target": upright_staging_targets[object_id],
                                "preserve_yaw": True,
                            },
                            "resources": {"primary": resource},
                        }
                    )
                calls.append(place_call)
                held_by.pop(object_id, None)
                cleanup_resources = (resource,)
            elif task_type == "E5":
                terminal_behavior = str(step.get("terminal_behavior"))
                target_reference = step.get("target", {})
                target_id = None
                if isinstance(target_reference, Mapping) and target_reference.get(
                    "kind"
                ) not in {None, "none"}:
                    target_id = self._resolve_step_entity(
                        step,
                        role="target",
                        bindings=bindings,
                        result_objects=result_objects,
                        steps_by_id=steps_by_id,
                    )
                displacement = self._transport_world_displacement(
                    step,
                    object_id=object_id,
                    target_id=target_id,
                    objects=objects,
                )
                calls = [
                    {
                        "kind": "registered",
                        "call_id": (
                            _COORDINATED_TRANSPORT_CALL_ID
                            if terminal_behavior == "place"
                            else _COORDINATED_HOLD_CALL_ID
                        ),
                        "arguments": {
                            "object": object_id,
                            "target": f"{step_id}_coordinated_target",
                            "world_displacement": displacement,
                        },
                        "resources": {"left": "left", "right": "right"},
                    }
                ]
                if terminal_behavior == "place":
                    held_by.pop(object_id, None)
                    cleanup_resources = ("left", "right")
                else:
                    held_by[object_id] = "coordinated"
            else:
                raise UnsupportedSemanticCapabilityError(
                    f"Task type {task_type!r} has no phase-one Semantic Call route."
                )

            requested_upright = (
                task_type in {"E1", "E4"} and step.get("orientation_goal") == "upright"
            )
            if requested_upright:
                orientation_resource = resource if task_type == "E1" else destination
                position = self._upright_target_position(object_id, objects)
                targets.update(_upright_targets(step_id, position))
                staging_target = f"{step_id}_upright_staging_target"
                upright_staging_targets[object_id] = staging_target
                if task_type == "E4":
                    handover_index = next(
                        i for i, call in enumerate(calls) if call["kind"] == "hand_over"
                    )
                    calls.insert(
                        handover_index,
                        {
                            "kind": "registered",
                            "call_id": _ALIGN_HELD_CALL_ID,
                            "arguments": {
                                "object": object_id,
                                "target": "current_object_pose",
                                "preserve_yaw": False,
                            },
                            "resources": {"primary": source},
                        },
                    )
                if not any(
                    call.get("call_id") == _ALIGN_HELD_CALL_ID
                    and call.get("resources", {}).get("primary") == orientation_resource
                    for call in calls
                ):
                    if task_type == "E1":
                        for index, call in enumerate(calls):
                            if call["kind"] == "pick":
                                calls[index] = {
                                    "kind": "registered",
                                    "call_id": _PICK_CALL_ID,
                                    "arguments": {
                                        "object": object_id,
                                        "target": f"{step_id}_upright_target",
                                    },
                                    "resources": {"primary": orientation_resource},
                                }
                        calls.insert(
                            -1,
                            {
                                "kind": "registered",
                                "call_id": _ALIGN_HELD_CALL_ID,
                                "arguments": {
                                    "object": object_id,
                                    "target": staging_target,
                                    "preserve_yaw": False,
                                },
                                "resources": {"primary": orientation_resource},
                            },
                        )
                    alignment = {
                        "kind": "registered",
                        "call_id": _ALIGN_HELD_CALL_ID,
                        "arguments": {
                            "object": object_id,
                            "target": "current_object_pose",
                            "preserve_yaw": True,
                        },
                        "resources": {"primary": orientation_resource},
                    }
                    if cleanup_resources:
                        calls.insert(-1, alignment)
                    else:
                        calls.append(alignment)
                upright_objects.add(object_id)
            call_roles = [(call, "primary") for call in calls]
            if cleanup_resources and (
                task_type == "E2"
                or requested_upright
                or calls[-1].get("call_id") == _STACK_PLACE_CALL_ID
            ):
                call_roles.append(
                    (
                        {
                            "kind": "registered",
                            "call_id": _CLEAR_RELEASED_CALL_ID,
                            "arguments": {
                                "object": object_id,
                                "target": upright_staging_targets[object_id],
                            },
                            "resources": {
                                "primary": (
                                    orientation_resource
                                    if requested_upright
                                    else resource
                                )
                            },
                        },
                        "cleanup",
                    )
                )
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

    def _relation_world_offset(
        self,
        relation: str,
        *,
        object_id: str,
        target_id: str,
        objects: Mapping[str, Mapping[str, Any]],
    ) -> list[float]:
        """Return a world-frame relation displacement including placement height."""
        target = objects.get(target_id)
        obj = objects.get(object_id)
        if target is None or obj is None:
            raise ValueError(
                f"Scene metadata is missing relation participants {object_id!r}, {target_id!r}."
            )
        target_position = target.get("init_pos")
        object_position = obj.get("init_pos")
        if (
            not isinstance(target_position, Sequence)
            or len(target_position) != 3
            or not isinstance(object_position, Sequence)
            or len(object_position) != 3
        ):
            raise ValueError(
                "Relation participants require three-value init_pos fields."
            )
        directions = {
            "left_of": (0.0, -self.lateral_relation_distance),
            "right_of": (0.0, self.lateral_relation_distance),
            "front_of": (-self.front_relation_distance, 0.0),
            "behind": (self.front_relation_distance, 0.0),
            "front_left_of": (
                -self.front_relation_distance,
                -self.lateral_relation_distance,
            ),
            "front_right_of": (
                -self.front_relation_distance,
                self.lateral_relation_distance,
            ),
            "back_left_of": (
                self.front_relation_distance,
                -self.lateral_relation_distance,
            ),
            "back_right_of": (
                self.front_relation_distance,
                self.lateral_relation_distance,
            ),
        }
        try:
            x_offset, y_offset = directions[relation]
        except KeyError as exc:
            raise ValueError(f"Unsupported directional relation {relation!r}.") from exc
        return [
            x_offset,
            y_offset,
            float(object_position[2]) - float(target_position[2]),
        ]

    def _placement_call(
        self,
        *,
        step_id: str,
        object_id: str,
        target_id: str,
        relation: str,
        resource: str,
    ) -> dict[str, Any]:
        """Build one canonical built-in or live-relative Place call."""
        if relation == "inside":
            return {
                "kind": "place",
                "object": object_id,
                "inside": _inside_affordance(target_id, object_id),
                "resources": {"primary": resource},
            }
        if relation not in {
            "on",
            "above",
            "left_of",
            "right_of",
            "front_of",
            "behind",
            "front_left_of",
            "front_right_of",
            "back_left_of",
            "back_right_of",
        }:
            raise UnsupportedSemanticCapabilityError(
                f"Step {step_id!r} relation {relation!r} has no canonical "
                "Semantic Call route."
            )
        return {
            "kind": "registered",
            "call_id": _PLACE_RELATIVE_CALL_ID,
            "arguments": {
                "object": object_id,
                "reference": target_id,
                "relation": relation,
            },
            "resources": {"primary": resource},
        }

    def _transport_world_displacement(
        self,
        step: Mapping[str, Any],
        *,
        object_id: str,
        target_id: str | None,
        objects: Mapping[str, Mapping[str, Any]],
    ) -> list[float]:
        """Resolve an E5 direction or target relation to one world displacement."""
        if target_id is not None:
            offset = self._relation_world_offset(
                str(step.get("relation", "none")),
                object_id=object_id,
                target_id=target_id,
                objects=objects,
            )
            return [
                float(objects[target_id]["init_pos"][index])
                + offset[index]
                - float(objects[object_id]["init_pos"][index])
                for index in range(3)
            ]
        distance = 0.14
        bounds = objects[object_id].get("attributes", {}).get("final_world_aabb")
        if isinstance(bounds, Mapping) and "min" in bounds and "max" in bounds:
            horizontal_extent = max(
                float(bounds["max"][axis]) - float(bounds["min"][axis])
                for axis in (0, 1)
            )
            distance = min(distance, max(0.05, horizontal_extent * 0.3))
        vectors = {
            "world_x": (1.0, 0.0, 0.0),
            "world_y": (0.0, 1.0, 0.0),
            "front": (-1.0, 0.0, 0.0),
            "back": (1.0, 0.0, 0.0),
            "left": (0.0, -1.0, 0.0),
            "right": (0.0, 1.0, 0.0),
            "front_left": (-1.0, -1.0, 0.0),
            "front_right": (-1.0, 1.0, 0.0),
            "back_left": (1.0, -1.0, 0.0),
            "back_right": (1.0, 1.0, 0.0),
            "up": (0.0, 0.0, 1.0),
            "down": (0.0, 0.0, -1.0),
        }
        direction = str(step.get("direction", "none"))
        try:
            vector = vectors[direction]
        except KeyError as exc:
            raise UnsupportedSemanticCapabilityError(
                f"E5 direction {direction!r} has no coordinated transport route."
            ) from exc
        magnitude = math.sqrt(sum(component * component for component in vector))
        return [distance * component / magnitude for component in vector]

    @staticmethod
    def _upright_target_position(
        object_id: str,
        objects: Mapping[str, Mapping[str, Any]],
    ) -> list[float]:
        """Place an upright object at its original XY and measured support height."""
        item = objects.get(object_id)
        if item is None:
            raise ValueError(f"Scene metadata is missing object {object_id!r}.")
        position = item.get("init_pos")
        if not isinstance(position, Sequence) or len(position) != 3:
            raise ValueError(f"Scene object {object_id!r} has no three-value init_pos.")
        result = [float(value) for value in position]
        attributes = item.get("attributes", {})
        object_aabb = (
            attributes.get("final_world_aabb")
            if isinstance(attributes, Mapping)
            else None
        )
        table = objects.get("table")
        table_attributes = table.get("attributes", {}) if table is not None else {}
        table_aabb = (
            table_attributes.get("final_world_aabb")
            if isinstance(table_attributes, Mapping)
            else None
        )
        if isinstance(object_aabb, Mapping) and isinstance(table_aabb, Mapping):
            object_min = object_aabb.get("min")
            object_max = object_aabb.get("max")
            table_max = table_aabb.get("max")
            if (
                isinstance(object_min, Sequence)
                and len(object_min) == 3
                and isinstance(object_max, Sequence)
                and len(object_max) == 3
                and isinstance(table_max, Sequence)
                and len(table_max) == 3
            ):
                longest_extent = max(
                    float(maximum) - float(minimum)
                    for minimum, maximum in zip(object_min, object_max, strict=True)
                )
                result[2] = float(table_max[2]) + 0.01 + longest_extent / 2.0
        return result


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


def _upright_targets(step_id: str, position: list[float]) -> dict[str, Any]:
    """Share the upright acquisition recipe; bundle geometry refines its heights."""
    return {
        f"{step_id}_{suffix}": {
            "kind": "cyclic_pose",
            "values": [
                {"position": list(position), "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
            ],
        }
        for suffix in ("upright_target", "upright_staging_target")
    }


def _park_call(resource: str) -> dict[str, Any]:
    """Build one payload-free semantic cleanup call for a logical resource."""
    return {
        "kind": "registered",
        "call_id": _PARK_CALL_ID,
        "arguments": {},
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
