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

"""Coordinate-free task and execution program contracts.

The task agent is the only structure an LLM is allowed to influence. The
execution program is produced deterministically and contains the complete
symbolic action DAG consumed by runtime. Neither representation may contain
poses, trajectories, joint values, or other environment-specific geometry.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from typing import Any

from embodichain.gen_sim.action_engine.protocol import (
    EXECUTION_PROGRAM_SCHEMA,
    TASK_AGENT_SCHEMA,
)

__all__ = [
    "EXECUTION_PROGRAM_SCHEMA",
    "MOTION_POLICY_VERSION",
    "TASK_AGENT_SCHEMA",
    "execution_program_hash",
    "validate_execution_program",
    "validate_task_agent",
]

MOTION_POLICY_VERSION = "action_engine_motion_policy_v2"

_ACTOR_MODES = frozenset({"auto", "required", "coordinated"})
_CONTROL_MODES = frozenset({"arm", "hand", "coordinated"})
_TASK_KEYS = frozenset(
    {"schema_version", "task", "goal", "semantic_steps", "allocation_groups"}
)
_TASK_STEP_KEYS = frozenset(
    {"id", "operator", "object", "objects", "actor", "goal", "depends_on"}
)
_EXECUTION_KEYS = frozenset(
    {
        "schema_version",
        "task",
        "goal_description",
        "start",
        "goal",
        "nodes",
        "edges",
        "semantic_steps",
        "allocation_groups",
        "motion_policy_version",
    }
)
_EXECUTION_STEP_KEYS = frozenset(
    {
        "id",
        "parent_step_id",
        "operator",
        "object",
        "actor",
        "goal",
        "depends_on",
        "postcondition",
        "edge_ids",
    }
)
_EDGE_KEYS = frozenset(
    {
        "id",
        "source",
        "target",
        "semantic_step_id",
        "actions",
        "depends_on",
        "resources",
    }
)
_ACTION_KEYS = frozenset(
    {
        "atomic_action_class",
        "actor",
        "control",
        "target_binding",
        "motion_policy",
    }
)
_TASK_ALLOCATION_GROUP_KEYS = frozenset({"id", "semantic_step_ids", "arm_constraint"})
_BINDING_REQUIREMENTS = {
    "coordinated_goal": frozenset({"object"}),
    "coordinated_placement_goal": frozenset({"placing_object", "support_object"}),
    "current_held_pose": frozenset(),
    "joint_state": frozenset({"source"}),
    "object": frozenset({"object"}),
    "policy_pose": frozenset(),
    "semantic_goal": frozenset({"semantic_step"}),
}
_POSTCONDITION_TYPES = frozenset(
    {
        "both_arms_at_initial_qpos",
        "both_grippers_open",
        "coordinated_placed",
        "grippers_clear_of_object",
        "held_by_both_grippers",
        "line_member_placed",
        "object_axis_near",
        "object_axis_offset_near",
        "object_held",
        "object_held_by_both_grippers",
        "object_held_by_gripper",
        "object_in_container",
        "object_lifted",
        "object_not_fallen",
        "object_on_object",
        "object_position_near",
        "object_upright",
        "object_xy_near",
        "objects_collinear",
        "objects_ordered",
        "pressed",
        "semantic_goal",
        "stack_layer_supported",
    }
)
_OBJECT_REFERENCE_KEYS = frozenset(
    {
        "anchor",
        "container",
        "object",
        "object_uid",
        "orientation_reference_object",
        "placing_object",
        "reference",
        "reference_object",
        "support",
        "support_object",
    }
)

# These fields indicate that planning-time or runtime geometry leaked into a
# symbolic program. Integers such as slot and layer indices remain valid.
_GROUNDED_FIELD_NAMES = frozenset(
    {
        "absolute_position",
        "coordinates",
        "joint_positions",
        "object_target_pose",
        "position",
        "positions",
        "pose",
        "qpos",
        "release_position",
        "staging_position",
        "target_pose",
        "trajectory",
        "waypoints",
        "xpos",
    }
)


def validate_task_agent(
    program: Mapping[str, Any],
    *,
    known_objects: Collection[str] | None = None,
) -> dict[str, Any]:
    """Validate and return a detached canonical TaskAgent mapping.

    Defaults are added only for structural fields that have one unambiguous
    meaning: ``actor={"mode": "auto"}``, an empty goal, and no dependencies.
    Operator-specific semantics are validated by the capability registry.

    Args:
        program: Candidate route-free task agent.
        known_objects: Optional runtime scene UIDs. When supplied, every object
            reference is validated before compilation.

    Returns:
        A deep-copied, canonical mapping safe for compilation.

    Raises:
        ValueError: If the program violates the TaskAgent contract.
    """
    value = _mapping_copy(program, "TaskAgent")
    _reject_unknown_keys(value, _TASK_KEYS, "TaskAgent")
    _require_schema(value, TASK_AGENT_SCHEMA, "TaskAgent")
    _require_nonempty_string(value.get("task"), "TaskAgent.task")
    _require_nonempty_string(value.get("goal"), "TaskAgent.goal")

    raw_steps = _sequence(value.get("semantic_steps"), "TaskAgent.semantic_steps")
    if not raw_steps:
        raise ValueError("TaskAgent.semantic_steps must not be empty.")

    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        context = f"TaskAgent.semantic_steps[{index}]"
        step = _mapping_copy(raw_step, context)
        _reject_unknown_keys(step, _TASK_STEP_KEYS, context)
        _require_nonempty_string(step.get("id"), f"{context}.id")
        _require_nonempty_string(step.get("operator"), f"{context}.operator")

        has_object = "object" in step
        has_objects = "objects" in step
        if has_object == has_objects:
            raise ValueError(
                f"{context} must contain exactly one of 'object' or 'objects'."
            )
        if has_object:
            _require_nonempty_string(step["object"], f"{context}.object")
        else:
            objects = _string_list(step["objects"], f"{context}.objects")
            if not objects:
                raise ValueError(f"{context}.objects must not be empty.")
            _require_unique(objects, f"{context}.objects")
            step["objects"] = objects

        step["actor"] = _validate_actor(
            step.get("actor", {"mode": "auto"}),
            f"{context}.actor",
        )
        step["goal"] = _mapping_copy(step.get("goal", {}), f"{context}.goal")
        step["depends_on"] = _string_list(
            step.get("depends_on", []),
            f"{context}.depends_on",
        )
        _require_unique(step["depends_on"], f"{context}.depends_on")
        steps.append(step)

    step_ids = [step["id"] for step in steps]
    _require_unique(step_ids, "TaskAgent semantic step IDs")
    dependencies = {step["id"]: step["depends_on"] for step in steps}
    _validate_dependency_dag(dependencies, "TaskAgent semantic steps")
    value["semantic_steps"] = steps
    value["allocation_groups"] = _validate_task_allocation_groups(
        value.get("allocation_groups", []),
        set(step_ids),
    )
    if known_objects is not None:
        _validate_known_objects(value, known_objects)
    _reject_grounded_values(value)
    return value


def validate_execution_program(program: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached canonical ExecutionProgram mapping.

    Validation covers both dependency DAGs: semantic-step dependencies and
    executable edge dependencies. It also proves node reachability, edge
    ownership, resource declarations, and the symbolic action envelope.

    Args:
        program: Candidate deterministic execution program.

    Returns:
        A deep-copied mapping safe for runtime consumption or hashing.

    Raises:
        ValueError: If the program violates the ExecutionProgram contract.
    """
    value = _mapping_copy(program, "ExecutionProgram")
    _reject_unknown_keys(value, _EXECUTION_KEYS, "ExecutionProgram")
    _require_schema(value, EXECUTION_PROGRAM_SCHEMA, "ExecutionProgram")
    _require_nonempty_string(value.get("task"), "ExecutionProgram.task")
    _require_nonempty_string(
        value.get("goal_description"),
        "ExecutionProgram.goal_description",
    )
    _require_nonempty_string(value.get("start"), "ExecutionProgram.start")
    _require_nonempty_string(value.get("goal"), "ExecutionProgram.goal")
    if value.get("motion_policy_version") != MOTION_POLICY_VERSION:
        raise ValueError(
            "ExecutionProgram.motion_policy_version must be "
            f"{MOTION_POLICY_VERSION!r}."
        )

    nodes = _validate_nodes(value.get("nodes"))
    node_ids = {node["id"] for node in nodes}
    if value["start"] not in node_ids or value["goal"] not in node_ids:
        raise ValueError("ExecutionProgram start and goal must reference nodes.")

    edges = _validate_edges(value.get("edges"), node_ids)
    edge_by_id = {edge["id"]: edge for edge in edges}
    _validate_dependency_dag(
        {edge_id: edge["depends_on"] for edge_id, edge in edge_by_id.items()},
        "ExecutionProgram edges",
    )
    _validate_node_reachability(
        start=value["start"],
        goal=value["goal"],
        node_ids=node_ids,
        edges=edges,
    )

    semantic_steps = _validate_execution_steps(
        value.get("semantic_steps"),
        edge_by_id,
    )
    step_ids = {step["id"] for step in semantic_steps}
    _validate_allocation_groups(value.get("allocation_groups"), step_ids)

    value["nodes"] = nodes
    value["edges"] = edges
    value["semantic_steps"] = semantic_steps
    value["allocation_groups"] = deepcopy(list(value.get("allocation_groups", [])))
    _reject_grounded_values(value)
    return value


def execution_program_hash(program: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 hash of a validated ExecutionProgram."""
    canonical = validate_execution_program(program)
    try:
        payload = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "ExecutionProgram must contain JSON-serializable values."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _validate_nodes(value: Any) -> list[dict[str, Any]]:
    raw_nodes = _sequence(value, "ExecutionProgram.nodes")
    if len(raw_nodes) < 2:
        raise ValueError("ExecutionProgram.nodes must contain start and goal nodes.")
    nodes: list[dict[str, Any]] = []
    for index, raw_node in enumerate(raw_nodes):
        context = f"ExecutionProgram.nodes[{index}]"
        node = _mapping_copy(raw_node, context)
        _reject_unknown_keys(node, frozenset({"id", "semantic"}), context)
        _require_nonempty_string(node.get("id"), f"{context}.id")
        _require_nonempty_string(node.get("semantic"), f"{context}.semantic")
        nodes.append(node)
    _require_unique([node["id"] for node in nodes], "ExecutionProgram node IDs")
    return nodes


def _validate_edges(value: Any, node_ids: set[str]) -> list[dict[str, Any]]:
    raw_edges = _sequence(value, "ExecutionProgram.edges")
    if not raw_edges:
        raise ValueError("ExecutionProgram.edges must not be empty.")
    edges: list[dict[str, Any]] = []
    for index, raw_edge in enumerate(raw_edges):
        context = f"ExecutionProgram.edges[{index}]"
        edge = _mapping_copy(raw_edge, context)
        _reject_unknown_keys(edge, _EDGE_KEYS, context)
        for key in ("id", "source", "target", "semantic_step_id"):
            _require_nonempty_string(edge.get(key), f"{context}.{key}")
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            raise ValueError(f"{context} references an unknown graph node.")

        edge["depends_on"] = _string_list(
            edge.get("depends_on", []),
            f"{context}.depends_on",
        )
        edge["resources"] = _string_list(
            edge.get("resources", []),
            f"{context}.resources",
        )
        _require_unique(edge["depends_on"], f"{context}.depends_on")
        _require_unique(edge["resources"], f"{context}.resources")
        edge["actions"] = _validate_actions(edge.get("actions"), context)
        edges.append(edge)

    edge_ids = [edge["id"] for edge in edges]
    _require_unique(edge_ids, "ExecutionProgram edge IDs")
    known_edges = set(edge_ids)
    for edge in edges:
        unknown = set(edge["depends_on"]) - known_edges
        if unknown:
            raise ValueError(
                f"Edge {edge['id']!r} depends on unknown edges: {sorted(unknown)}."
            )
    return edges


def _validate_actions(value: Any, edge_context: str) -> list[dict[str, Any]]:
    raw_actions = _sequence(value, f"{edge_context}.actions")
    if not raw_actions:
        raise ValueError(f"{edge_context}.actions must not be empty.")
    actions: list[dict[str, Any]] = []
    for index, raw_action in enumerate(raw_actions):
        context = f"{edge_context}.actions[{index}]"
        action = _mapping_copy(raw_action, context)
        _reject_unknown_keys(action, _ACTION_KEYS, context)
        _require_nonempty_string(
            action.get("atomic_action_class"),
            f"{context}.atomic_action_class",
        )
        action["actor"] = _validate_actor(action.get("actor"), f"{context}.actor")
        control = _require_nonempty_string(action.get("control"), f"{context}.control")
        if control not in _CONTROL_MODES:
            raise ValueError(
                f"{context}.control must be one of {sorted(_CONTROL_MODES)}."
            )
        binding = _mapping_copy(
            action.get("target_binding"),
            f"{context}.target_binding",
        )
        _require_nonempty_string(
            binding.get("kind"),
            f"{context}.target_binding.kind",
        )
        kind = binding["kind"]
        required = _BINDING_REQUIREMENTS.get(kind)
        if required is None:
            raise ValueError(f"{context}.target_binding.kind {kind!r} is unsupported.")
        missing = sorted(
            key for key in required if not _is_present_binding_value(binding.get(key))
        )
        if missing:
            raise ValueError(
                f"{context}.target_binding is missing required fields: {missing}."
            )
        action["target_binding"] = binding
        _require_nonempty_string(
            action.get("motion_policy"),
            f"{context}.motion_policy",
        )
        actions.append(action)
    return actions


def _validate_execution_steps(
    value: Any,
    edge_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_steps = _sequence(value, "ExecutionProgram.semantic_steps")
    if not raw_steps:
        raise ValueError("ExecutionProgram.semantic_steps must not be empty.")
    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        context = f"ExecutionProgram.semantic_steps[{index}]"
        step = _mapping_copy(raw_step, context)
        _reject_unknown_keys(step, _EXECUTION_STEP_KEYS, context)
        for key in ("id", "parent_step_id", "operator", "object"):
            _require_nonempty_string(step.get(key), f"{context}.{key}")
        step["actor"] = _validate_actor(step.get("actor"), f"{context}.actor")
        step["goal"] = _mapping_copy(step.get("goal"), f"{context}.goal")
        step["postcondition"] = _mapping_copy(
            step.get("postcondition"),
            f"{context}.postcondition",
        )
        _require_nonempty_string(
            step["postcondition"].get("type"),
            f"{context}.postcondition.type",
        )
        if step["postcondition"]["type"] not in _POSTCONDITION_TYPES:
            raise ValueError(
                f"{context}.postcondition.type "
                f"{step['postcondition']['type']!r} is unsupported."
            )
        step["depends_on"] = _string_list(
            step.get("depends_on", []),
            f"{context}.depends_on",
        )
        step["edge_ids"] = _string_list(
            step.get("edge_ids"),
            f"{context}.edge_ids",
        )
        if not step["edge_ids"]:
            raise ValueError(f"{context}.edge_ids must not be empty.")
        _require_unique(step["depends_on"], f"{context}.depends_on")
        _require_unique(step["edge_ids"], f"{context}.edge_ids")
        steps.append(step)

    step_ids = [step["id"] for step in steps]
    _require_unique(step_ids, "ExecutionProgram semantic step IDs")
    _validate_dependency_dag(
        {step["id"]: step["depends_on"] for step in steps},
        "ExecutionProgram semantic steps",
    )

    covered_edges: list[str] = []
    for step in steps:
        for edge_id in step["edge_ids"]:
            edge = edge_by_id.get(edge_id)
            if edge is None:
                raise ValueError(
                    f"Semantic step {step['id']!r} owns unknown edge {edge_id!r}."
                )
            if edge["semantic_step_id"] != step["id"]:
                raise ValueError(
                    f"Edge {edge_id!r} is assigned to {edge['semantic_step_id']!r}, "
                    f"not {step['id']!r}."
                )
            covered_edges.append(edge_id)
    _require_unique(covered_edges, "ExecutionProgram semantic edge ownership")
    if set(covered_edges) != set(edge_by_id):
        missing = sorted(set(edge_by_id) - set(covered_edges))
        raise ValueError(f"ExecutionProgram has unowned edges: {missing}.")
    return steps


def _validate_allocation_groups(value: Any, step_ids: set[str]) -> None:
    groups = _sequence(value, "ExecutionProgram.allocation_groups")
    group_ids: list[str] = []
    for index, raw_group in enumerate(groups):
        context = f"ExecutionProgram.allocation_groups[{index}]"
        group = _mapping_copy(raw_group, context)
        allowed = frozenset(
            {
                "id",
                "semantic_step_ids",
                "arm_constraint",
                "execution_policy",
                "parallel_action_classes",
                "workspace_policy",
            }
        )
        _reject_unknown_keys(group, allowed, context)
        group_ids.append(_require_nonempty_string(group.get("id"), f"{context}.id"))
        members = _string_list(
            group.get("semantic_step_ids"),
            f"{context}.semantic_step_ids",
        )
        if len(members) < 2:
            raise ValueError(f"{context} must contain at least two semantic steps.")
        _require_unique(members, f"{context}.semantic_step_ids")
        unknown = set(members) - step_ids
        if unknown:
            raise ValueError(f"{context} references unknown steps: {sorted(unknown)}.")
        for key in ("arm_constraint", "execution_policy", "workspace_policy"):
            _require_nonempty_string(group.get(key), f"{context}.{key}")
        if group["arm_constraint"] != "distinct_arms":
            raise ValueError(f"{context}.arm_constraint must be 'distinct_arms'.")
        if group["execution_policy"] != "parallel_if_feasible":
            raise ValueError(
                f"{context}.execution_policy must be 'parallel_if_feasible'."
            )
        if group["workspace_policy"] != "shared_target_serial":
            raise ValueError(
                f"{context}.workspace_policy must be 'shared_target_serial'."
            )
        action_classes = _string_list(
            group.get("parallel_action_classes"),
            f"{context}.parallel_action_classes",
        )
        if not action_classes:
            raise ValueError(f"{context}.parallel_action_classes must not be empty.")
    _require_unique(group_ids, "ExecutionProgram allocation group IDs")


def _validate_task_allocation_groups(
    value: Any,
    step_ids: set[str],
) -> list[dict[str, Any]]:
    groups = _sequence(value, "TaskAgent.allocation_groups")
    result: list[dict[str, Any]] = []
    ids: list[str] = []
    members_seen: set[str] = set()
    for index, raw_group in enumerate(groups):
        context = f"TaskAgent.allocation_groups[{index}]"
        group = _mapping_copy(raw_group, context)
        _reject_unknown_keys(group, _TASK_ALLOCATION_GROUP_KEYS, context)
        group_id = _require_nonempty_string(group.get("id"), f"{context}.id")
        members = _string_list(
            group.get("semantic_step_ids"),
            f"{context}.semantic_step_ids",
        )
        if len(members) < 2:
            raise ValueError(f"{context} must contain at least two semantic steps.")
        _require_unique(members, f"{context}.semantic_step_ids")
        unknown = set(members) - step_ids
        if unknown:
            raise ValueError(f"{context} references unknown steps: {sorted(unknown)}.")
        overlap = set(members) & members_seen
        if overlap:
            raise ValueError(
                f"TaskAgent allocation groups overlap at steps: {sorted(overlap)}."
            )
        constraint = _require_nonempty_string(
            group.get("arm_constraint"),
            f"{context}.arm_constraint",
        )
        if constraint != "distinct_arms":
            raise ValueError(f"{context}.arm_constraint must be 'distinct_arms'.")
        ids.append(group_id)
        members_seen.update(members)
        result.append(
            {
                "id": group_id,
                "semantic_step_ids": members,
                "arm_constraint": constraint,
            }
        )
    _require_unique(ids, "TaskAgent allocation group IDs")
    return result


def _validate_known_objects(
    program: Mapping[str, Any],
    known_objects: Collection[str],
) -> None:
    known = {str(uid) for uid in known_objects}
    if not known:
        raise ValueError("known_objects must not be empty when supplied.")
    allowed_sentinels = {"self", "table", "table_center"}
    references: list[tuple[str, str]] = []
    for index, step in enumerate(program["semantic_steps"]):
        if "object" in step:
            references.append((f"semantic_steps[{index}].object", step["object"]))
        for item_index, uid in enumerate(step.get("objects", [])):
            references.append((f"semantic_steps[{index}].objects[{item_index}]", uid))
        _collect_object_references(
            step["goal"],
            f"semantic_steps[{index}].goal",
            references,
        )
    unknown = [
        f"{path}={uid!r}"
        for path, uid in references
        if uid not in known and uid not in allowed_sentinels
    ]
    if unknown:
        raise ValueError(
            "TaskAgent references objects not present in the scene: "
            + ", ".join(unknown)
            + "."
        )


def _collect_object_references(
    value: Any,
    path: str,
    output: list[tuple[str, str]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in _OBJECT_REFERENCE_KEYS and isinstance(child, str):
                output.append((child_path, child))
            elif (
                key in {"objects", "object_uids", "payloads"}
                and isinstance(child, Sequence)
                and not isinstance(child, (str, bytes, bytearray))
            ):
                for index, item in enumerate(child):
                    uid = item.get("object") if isinstance(item, Mapping) else item
                    if isinstance(uid, str):
                        output.append((f"{child_path}[{index}]", uid))
            else:
                _collect_object_references(child, child_path, output)


def _is_present_binding_value(value: Any) -> bool:
    return value is not None and value != "" and value != []


def _validate_actor(value: Any, context: str) -> dict[str, Any]:
    actor = _mapping_copy(value, context)
    mode = _require_nonempty_string(actor.get("mode"), f"{context}.mode")
    if mode not in _ACTOR_MODES:
        raise ValueError(f"{context}.mode must be one of {sorted(_ACTOR_MODES)}.")
    if mode == "auto":
        _reject_unknown_keys(actor, frozenset({"mode", "allocation_group"}), context)
    elif mode == "required":
        _reject_unknown_keys(
            actor,
            frozenset({"mode", "arm", "allocation_group"}),
            context,
        )
        _require_nonempty_string(actor.get("arm"), f"{context}.arm")
    else:
        _reject_unknown_keys(actor, frozenset({"mode", "arms"}), context)
        arms = _string_list(actor.get("arms"), f"{context}.arms")
        if len(arms) < 2:
            raise ValueError(f"{context}.arms must contain at least two arms.")
        _require_unique(arms, f"{context}.arms")
        actor["arms"] = arms
    if "allocation_group" in actor:
        _require_nonempty_string(
            actor["allocation_group"],
            f"{context}.allocation_group",
        )
    return actor


def _validate_dependency_dag(
    dependencies: Mapping[str, Sequence[str]],
    context: str,
) -> None:
    known = set(dependencies)
    outgoing: dict[str, list[str]] = {item_id: [] for item_id in known}
    indegree = {item_id: 0 for item_id in known}
    for item_id, required_ids in dependencies.items():
        unknown = set(required_ids) - known
        if unknown:
            raise ValueError(f"{context} reference unknown IDs: {sorted(unknown)}.")
        if item_id in required_ids:
            raise ValueError(f"{context} contain a self-dependency at {item_id!r}.")
        for required_id in required_ids:
            outgoing[required_id].append(item_id)
            indegree[item_id] += 1

    ready = deque(
        sorted(item_id for item_id, degree in indegree.items() if degree == 0)
    )
    visited = 0
    while ready:
        item_id = ready.popleft()
        visited += 1
        for dependent_id in sorted(outgoing[item_id]):
            indegree[dependent_id] -= 1
            if indegree[dependent_id] == 0:
                ready.append(dependent_id)
    if visited != len(known):
        cyclic = sorted(item_id for item_id, degree in indegree.items() if degree > 0)
        raise ValueError(f"{context} contain a dependency cycle: {cyclic}.")


def _validate_node_reachability(
    *,
    start: str,
    goal: str,
    node_ids: set[str],
    edges: Sequence[Mapping[str, Any]],
) -> None:
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        outgoing[edge["source"]].append(edge["target"])
    reachable = {start}
    ready = deque([start])
    while ready:
        node_id = ready.popleft()
        for target_id in outgoing[node_id]:
            if target_id not in reachable:
                reachable.add(target_id)
                ready.append(target_id)
    if goal not in reachable:
        raise ValueError("ExecutionProgram goal is unreachable from start.")
    unreachable = sorted(node_ids - reachable)
    if unreachable:
        raise ValueError(f"ExecutionProgram contains unreachable nodes: {unreachable}.")


def _reject_grounded_values(value: Any, path: str = "program") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _GROUNDED_FIELD_NAMES:
                raise ValueError(
                    f"{path}.{key} is grounded runtime data and is not allowed."
                )
            _reject_grounded_values(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_grounded_values(child, f"{path}[{index}]")
        return
    if isinstance(value, float):
        raise ValueError(
            f"{path} contains a floating-point runtime value; use a named "
            "motion policy or symbolic relation instead."
        )


def _mapping_copy(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return deepcopy(dict(value))


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{context} must be a list.")
    return list(value)


def _string_list(value: Any, context: str) -> list[str]:
    items = _sequence(value, context)
    result: list[str] = []
    for index, item in enumerate(items):
        result.append(_require_nonempty_string(item, f"{context}[{index}]"))
    return result


def _require_schema(value: Mapping[str, Any], expected: str, context: str) -> None:
    if value.get("schema_version") != expected:
        raise ValueError(f"{context}.schema_version must be {expected!r}.")


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def _require_unique(values: Sequence[str], context: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{context} must not contain duplicates.")


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {unknown}.")
