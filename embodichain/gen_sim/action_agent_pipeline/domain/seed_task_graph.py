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

"""Schema ownership for environment-independent executable Seed graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

__all__ = [
    "MOTION_POLICY_VERSION",
    "SEED_TASK_GRAPH_SCHEMA_VERSION",
    "SEMANTIC_STEP_SCHEMA_VERSION",
    "seed_task_graph_hash",
    "validate_seed_task_graph",
]

SEED_TASK_GRAPH_SCHEMA_VERSION = "seed_task_graph_v5"
SEMANTIC_STEP_SCHEMA_VERSION = "semantic_steps_v4"
MOTION_POLICY_VERSION = "action_agent_motion_policy_v4"

_ROUTES = {"arrangement_line", "object_manipulation", "stacking"}
_ACTOR_MODES = {"auto", "coordinated", "required"}
_ARM_NAMES = {"left_arm", "right_arm"}
_BINDING_KINDS = {
    "coordinated_goal",
    "current_held_pose",
    "joint_state",
    "object",
    "policy_pose",
    "semantic_goal",
}
_ATOMIC_ACTION_CLASSES = {
    "CoordinatedPickment",
    "MoveEndEffector",
    "MoveHeldObject",
    "MoveJoints",
    "PickUp",
    "Place",
}
_CONTROLS = {"arm", "coordinated", "hand"}
_MOTION_POLICIES = {
    "default_home",
    "default_pickup",
    "default_release",
    "default_retreat",
    "default_transport",
    "upright_in_place_pickup",
    "upright_in_place_release",
    "upright_in_place_retreat",
    "upright_in_place_transport",
}
_BINDING_FIELDS = {
    "coordinated_goal": {"kind", "object"},
    "current_held_pose": {"kind"},
    "joint_state": {"kind", "source"},
    "object": {"affordance", "kind", "object"},
    "policy_pose": {"kind"},
    "semantic_goal": {"kind", "semantic_step"},
}
_ACTION_CONTRACTS = {
    "CoordinatedPickment": ("coordinated_goal", "coordinated"),
    "MoveEndEffector": ("policy_pose", "arm"),
    "MoveHeldObject": ("semantic_goal", "arm"),
    "MoveJoints": ("joint_state", None),
    "PickUp": ("object", "arm"),
    "Place": ("current_held_pose", "arm"),
}
_ACTION_POLICIES = {
    "CoordinatedPickment": "default_transport",
    "MoveEndEffector": {"default_retreat", "upright_in_place_retreat"},
    "MoveHeldObject": {"default_transport", "upright_in_place_transport"},
    "PickUp": {"default_pickup", "upright_in_place_pickup"},
    "Place": {"default_release", "upright_in_place_release"},
}
_TOP_LEVEL_KEYS = {
    "allocation_groups",
    "edges",
    "goal",
    "motion_policy_version",
    "nodes",
    "program",
    "route",
    "schema_version",
    "semantic_step_schema_version",
    "semantic_steps",
    "start",
    "task",
}
_ALLOCATION_GROUP_KEYS = {
    "arm_constraint",
    "execution_policy",
    "id",
    "parallel_action_classes",
    "semantic_step_ids",
    "workspace_policy",
}
_EDGE_KEYS = {
    "actions",
    "depends_on",
    "id",
    "resources",
    "source",
    "target",
}
_STEP_KEYS = {
    "actor",
    "depends_on",
    "edge_ids",
    "goal",
    "id",
    "object",
    "operator",
    "postcondition",
}
_ACTION_KEYS = {
    "actor",
    "atomic_action_class",
    "control",
    "motion_policy",
    "target_binding",
}
_GROUNDED_FIELD_NAMES = {
    "active_arm",
    "cfg",
    "distance",
    "execution",
    "failure_reason",
    "high_position",
    "ik",
    "joint_state",
    "lift_height",
    "offset",
    "position",
    "planning",
    "pre_grasp_distance",
    "qpos",
    "release_position",
    "resolved_eef_pose",
    "resolved_motion_policy",
    "resolved_target_object_pose",
    "resolved_target_position",
    "rotation_matrix",
    "runtime",
    "runtime_status",
    "status",
    "surface_clearance",
    "target_position",
    "target_qpos",
    "target_xy",
    "tolerance",
    "trajectory",
}
_GROUNDED_FIELD_FRAGMENTS = {
    "distance",
    "offset",
    "position",
    "qpos",
    "tolerance",
    "trajectory",
}
_DEFERRED_POLICY_FIELDS = {
    "clearance",
    "hover_height",
    "line_spacing",
    "post_hold_steps",
    "relation_distance",
    "retreat_height",
    "sample_interval",
}


def seed_task_graph_hash(seed_graph: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 identity of one validated executable Seed."""
    validate_seed_task_graph(seed_graph)
    canonical = json.dumps(
        seed_graph,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_seed_task_graph(
    seed_graph: Mapping[str, Any],
    *,
    task_name: str | None = None,
    route: str | None = None,
) -> None:
    """Validate Seed v5 topology and reject runtime-grounded data leakage."""
    if not isinstance(seed_graph, Mapping):
        raise TypeError("Seed task graph must be a mapping.")
    unknown_graph_keys = set(seed_graph) - _TOP_LEVEL_KEYS
    if unknown_graph_keys:
        raise ValueError(
            "Seed task graph contains unsupported top-level fields: "
            f"{sorted(unknown_graph_keys)}."
        )
    if seed_graph.get("schema_version") != SEED_TASK_GRAPH_SCHEMA_VERSION:
        actual = seed_graph.get("schema_version")
        if actual in {
            "seed_task_graph_v1",
            "seed_task_graph_v2",
            "seed_task_graph_v3",
            "seed_task_graph_v4",
        }:
            raise ValueError(
                f"{actual} is no longer supported. Regenerate the action-agent "
                "config with --overwrite."
            )
        raise ValueError(
            "Seed task graph schema_version must be "
            f"{SEED_TASK_GRAPH_SCHEMA_VERSION!r}. Legacy/precomputed graphs are "
            "not supported; regenerate the action-agent config with --overwrite."
        )
    if seed_graph.get("semantic_step_schema_version") != SEMANTIC_STEP_SCHEMA_VERSION:
        raise ValueError(
            "Seed task graph semantic_step_schema_version must be "
            f"{SEMANTIC_STEP_SCHEMA_VERSION!r}."
        )
    if seed_graph.get("motion_policy_version") != MOTION_POLICY_VERSION:
        raise ValueError(
            f"Seed task graph motion_policy_version must be {MOTION_POLICY_VERSION!r}."
        )
    graph_task = seed_graph.get("task")
    if not isinstance(graph_task, str) or not graph_task.strip():
        raise ValueError("Seed task graph requires a non-empty task name.")
    if task_name is not None and graph_task != task_name:
        raise ValueError(
            f"Seed task graph task {graph_task!r} does not match {task_name!r}."
        )
    graph_route = seed_graph.get("route")
    if graph_route not in _ROUTES:
        raise ValueError(f"Unsupported seed task graph route: {graph_route!r}.")
    if route is not None and graph_route != route:
        raise ValueError(
            f"Seed task graph route {graph_route!r} does not match {route!r}."
        )
    if not isinstance(seed_graph.get("program"), str) or not seed_graph["program"]:
        raise ValueError("Seed task graph requires a non-empty program.")

    _reject_grounded_fields(seed_graph)
    nodes = seed_graph.get("nodes")
    edges = seed_graph.get("edges")
    steps = seed_graph.get("semantic_steps")
    allocation_groups = seed_graph.get("allocation_groups")
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError("Seed task graph requires at least two nodes.")
    if not isinstance(edges, list) or not edges:
        raise ValueError("Seed task graph requires a non-empty edges list.")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Seed task graph requires non-empty semantic_steps.")
    if not isinstance(allocation_groups, list):
        raise TypeError("Seed task graph allocation_groups must be a list.")

    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise TypeError(f"Seed node {index} must be a mapping.")
        if set(node) != {"id", "semantic"}:
            raise ValueError(f"Seed node {index} must contain id and semantic.")
        node_id = node["id"]
        if not isinstance(node_id, str) or not node_id or node_id in node_ids:
            raise ValueError(f"Invalid or duplicate seed node id: {node_id!r}.")
        node_ids.add(node_id)
    for endpoint in ("start", "goal"):
        if seed_graph.get(endpoint) not in node_ids:
            raise ValueError(f"Seed graph {endpoint} must reference a defined node.")

    edge_ids: set[str] = set()
    incoming = {node_id: 0 for node_id in node_ids}
    outgoing = {node_id: 0 for node_id in node_ids}
    known_edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            raise TypeError(f"Seed edge {index} must be a mapping.")
        if set(edge) != _EDGE_KEYS:
            raise ValueError(
                "Seed edge "
                f"{index} must contain id/source/target/actions/depends_on/resources."
            )
        edge_id = edge["id"]
        if not isinstance(edge_id, str) or not edge_id or edge_id in edge_ids:
            raise ValueError(f"Invalid or duplicate seed edge id: {edge_id!r}.")
        edge_ids.add(edge_id)
        dependencies = edge["depends_on"]
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) for item in dependencies
        ):
            raise TypeError(f"Seed edge {edge_id!r} depends_on must be a list.")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"Seed edge {edge_id!r} has duplicate dependencies.")
        unknown_dependencies = set(dependencies) - known_edge_ids
        if unknown_dependencies:
            raise ValueError(
                f"Seed edge {edge_id!r} depends on non-prior edges: "
                f"{sorted(unknown_dependencies)}."
            )
        resources = edge["resources"]
        if (
            not isinstance(resources, list)
            or not all(isinstance(item, str) and item for item in resources)
            or len(resources) != len(set(resources))
        ):
            raise ValueError(
                f"Seed edge {edge_id!r} resources must be unique non-empty strings."
            )
        source = edge["source"]
        target = edge["target"]
        if source not in node_ids or target not in node_ids:
            raise ValueError(f"Seed edge {edge_id!r} references an unknown node.")
        outgoing[source] += 1
        incoming[target] += 1
        actions = edge["actions"]
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"Seed edge {edge_id!r} requires symbolic actions.")
        for action in actions:
            _validate_symbolic_action(edge_id, action)
        known_edge_ids.add(edge_id)

    start = str(seed_graph["start"])
    goal = str(seed_graph["goal"])
    if incoming[start] != 0 or outgoing[goal] != 0:
        raise ValueError("Seed start/goal topology is invalid.")
    _validate_action_dag(start, goal, node_ids, edges)

    known_steps: set[str] = set()
    covered_edges: list[str] = []
    edge_by_id = {str(edge["id"]): edge for edge in edges}
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or set(step) != _STEP_KEYS:
            raise ValueError(f"Seed semantic step {index} has invalid fields.")
        step_id = step["id"]
        if not isinstance(step_id, str) or not step_id or step_id in known_steps:
            raise ValueError(f"Invalid or duplicate semantic step id: {step_id!r}.")
        dependencies = step["depends_on"]
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) for item in dependencies
        ):
            raise TypeError(f"Semantic step {step_id!r} depends_on must be a list.")
        unknown_dependencies = set(dependencies) - known_steps
        if unknown_dependencies:
            raise ValueError(
                f"Semantic step {step_id!r} depends on non-prior steps: "
                f"{sorted(unknown_dependencies)}."
            )
        _validate_seed_actor(step_id, step["actor"])
        if not isinstance(step["operator"], str) or not step["operator"]:
            raise ValueError(f"Semantic step {step_id!r} requires an operator.")
        if not isinstance(step["object"], str) or not step["object"]:
            raise ValueError(f"Semantic step {step_id!r} requires an object.")
        if not isinstance(step["goal"], Mapping) or not isinstance(
            step["postcondition"], Mapping
        ):
            raise TypeError(f"Semantic step {step_id!r} goals must be mappings.")
        assigned = step["edge_ids"]
        if not isinstance(assigned, list) or not assigned:
            raise ValueError(f"Semantic step {step_id!r} requires edge_ids.")
        if any(edge_id not in edge_ids for edge_id in assigned):
            raise ValueError(f"Semantic step {step_id!r} references unknown edges.")
        for edge_id in assigned:
            edge_record = edge_by_id[str(edge_id)]
            edge_actions = edge_record["actions"]
            _validate_edge_resources(step_id, step, edge_record)
            for action in edge_actions:
                actor_matches_step = action["actor"] == step["actor"]
                coordinated_child = (
                    step["actor"]["mode"] == "coordinated"
                    and action["actor"].get("mode") == "required"
                    and action["actor"].get("arm") in _ARM_NAMES
                )
                if not actor_matches_step and not coordinated_child:
                    raise ValueError(
                        f"Seed edge {edge_id!r} actor does not match semantic step "
                        f"{step_id!r}."
                    )
                binding = action["target_binding"]
                if (
                    binding["kind"] == "semantic_goal"
                    and binding.get("semantic_step") != step_id
                ):
                    raise ValueError(
                        f"Seed edge {edge_id!r} references the wrong semantic goal."
                    )
            if len(edge_actions) > 1:
                child_arms = {
                    action["actor"].get("arm")
                    for action in edge_actions
                    if action["actor"].get("mode") == "required"
                }
                if (
                    step["actor"]["mode"] != "coordinated"
                    or len(edge_actions) != 2
                    or child_arms != _ARM_NAMES
                ):
                    raise ValueError(
                        f"Seed edge {edge_id!r} has an invalid multi-arm action list."
                    )
            elif step["actor"]["mode"] == "coordinated" and (
                edge_actions[0]["atomic_action_class"] != "CoordinatedPickment"
                or edge_actions[0]["actor"] != step["actor"]
            ):
                raise ValueError(
                    f"Seed edge {edge_id!r} has an invalid coordinated action."
                )
        covered_edges.extend(assigned)
        known_steps.add(step_id)
    if len(covered_edges) != len(set(covered_edges)) or set(covered_edges) != edge_ids:
        raise ValueError(
            "Semantic step edge_ids must cover every Seed edge exactly once."
        )
    _validate_allocation_groups(allocation_groups, steps, edge_by_id)
    if graph_route == "arrangement_line":
        _validate_arrangement_steps(steps, edge_by_id)


def _validate_allocation_groups(
    groups: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    edge_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate group-level arm allocation without resolving a runtime arm."""
    step_by_id = {str(step["id"]): step for step in steps}
    grouped_steps: set[str] = set()
    group_ids: set[str] = set()
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or set(group) != _ALLOCATION_GROUP_KEYS:
            raise ValueError(f"Seed allocation group {index} has invalid fields.")
        group_id = group["id"]
        if not isinstance(group_id, str) or not group_id or group_id in group_ids:
            raise ValueError(f"Invalid or duplicate allocation group id: {group_id!r}.")
        step_ids = group["semantic_step_ids"]
        if (
            not isinstance(step_ids, list)
            or len(step_ids) != 2
            or len(set(step_ids)) != 2
            or any(step_id not in step_by_id for step_id in step_ids)
        ):
            raise ValueError(
                f"Allocation group {group_id!r} requires two known semantic steps."
            )
        if grouped_steps.intersection(step_ids):
            raise ValueError("A semantic step may belong to only one allocation group.")
        if group["arm_constraint"] != "distinct_arms":
            raise ValueError(f"Allocation group {group_id!r} requires distinct_arms.")
        if group["execution_policy"] != "parallel_if_feasible":
            raise ValueError(
                f"Allocation group {group_id!r} requires parallel_if_feasible."
            )
        if group["parallel_action_classes"] != ["PickUp"]:
            raise ValueError(
                f"Allocation group {group_id!r} may parallelize only PickUp."
            )
        if group["workspace_policy"] != "shared_target_serial":
            raise ValueError(
                f"Allocation group {group_id!r} requires shared_target_serial."
            )
        objects = {str(step_by_id[step_id]["object"]) for step_id in step_ids}
        if len(objects) != 2:
            raise ValueError(
                f"Allocation group {group_id!r} requires two distinct objects."
            )
        for step_id in step_ids:
            step = step_by_id[step_id]
            if step["actor"]["mode"] not in {"auto", "required"}:
                raise ValueError(
                    f"Allocation group {group_id!r} has an unsupported actor."
                )
            first_edge = edge_by_id[str(step["edge_ids"][0])]
            if first_edge["actions"][0]["atomic_action_class"] != "PickUp":
                raise ValueError(
                    f"Allocation group {group_id!r} steps must begin with PickUp."
                )
        grouped_steps.update(step_ids)
        group_ids.add(group_id)


def _validate_seed_actor(step_id: str, actor: Any) -> None:
    if not isinstance(actor, Mapping) or actor.get("mode") not in _ACTOR_MODES:
        raise ValueError(f"Seed task graph step {step_id!r} has an invalid actor.")
    mode = actor["mode"]
    if mode == "required" and (
        set(actor) != {"arm", "mode"} or actor.get("arm") not in _ARM_NAMES
    ):
        raise ValueError(
            f"Seed task graph step {step_id!r} required actor needs a valid arm."
        )
    if mode == "auto" and set(actor) != {"mode"}:
        raise ValueError(
            f"Seed task graph step {step_id!r} auto actor must remain unresolved."
        )
    if mode == "coordinated":
        arms = actor.get("arms")
        if (
            set(actor) != {"arms", "mode"}
            or not isinstance(arms, list)
            or set(arms) != _ARM_NAMES
        ):
            raise ValueError(
                f"Seed task graph step {step_id!r} coordinated actor needs both arms."
            )


def _validate_edge_resources(
    step_id: str,
    step: Mapping[str, Any],
    edge: Mapping[str, Any],
) -> None:
    resources = set(edge["resources"])
    required = {f"object:{step['object']}"}
    actor = step["actor"]
    if actor["mode"] == "required":
        required.add(f"arm:{actor['arm']}")
    elif actor["mode"] == "coordinated":
        required.update({"arm:left_arm", "arm:right_arm"})
    else:
        required.add("arm:auto")
    missing = required - resources
    if missing:
        raise ValueError(
            f"Seed edge {edge['id']!r} for semantic step {step_id!r} is "
            f"missing required resources: {sorted(missing)}."
        )


def _validate_symbolic_action(edge_id: str, action: Any) -> None:
    if not isinstance(action, Mapping) or set(action) != _ACTION_KEYS:
        raise ValueError(f"Seed edge {edge_id!r} has an invalid symbolic action.")
    if action["atomic_action_class"] not in _ATOMIC_ACTION_CLASSES:
        raise ValueError(f"Seed edge {edge_id!r} has an invalid action class.")
    _validate_seed_actor(edge_id, action["actor"])
    if action["control"] not in _CONTROLS:
        raise ValueError(f"Seed edge {edge_id!r} has an invalid control mode.")
    binding = action["target_binding"]
    if not isinstance(binding, Mapping) or binding.get("kind") not in _BINDING_KINDS:
        raise ValueError(f"Seed edge {edge_id!r} has an invalid target binding.")
    binding_kind = str(binding["kind"])
    valid_binding_fields = set(binding) == _BINDING_FIELDS[binding_kind]
    if binding_kind == "semantic_goal":
        valid_binding_fields = valid_binding_fields or set(binding) == {
            *_BINDING_FIELDS[binding_kind],
            "phase",
        }
    if not valid_binding_fields:
        raise ValueError(
            f"Seed edge {edge_id!r} target binding {binding_kind!r} has "
            "invalid fields."
        )
    expected_binding, expected_control = _ACTION_CONTRACTS[
        str(action["atomic_action_class"])
    ]
    if binding_kind != expected_binding or (
        expected_control is not None and action["control"] != expected_control
    ):
        raise ValueError(
            f"Seed edge {edge_id!r} action/binding/control contract is invalid."
        )
    if binding_kind in {"coordinated_goal", "object"} and not isinstance(
        binding.get("object"), str
    ):
        raise ValueError(f"Seed edge {edge_id!r} requires an object binding.")
    if binding_kind == "semantic_goal" and not isinstance(
        binding.get("semantic_step"), str
    ):
        raise ValueError(f"Seed edge {edge_id!r} requires a semantic step binding.")
    if binding_kind == "semantic_goal" and "phase" in binding:
        if binding["phase"] not in {"staging", "final"}:
            raise ValueError(
                f"Seed edge {edge_id!r} has an invalid semantic-goal phase."
            )
    if binding_kind == "joint_state":
        source = binding.get("source")
        expected_control = (
            "hand" if source in {"gripper_closed", "gripper_open"} else "arm"
        )
        if source not in {"gripper_closed", "gripper_open", "initial"} or action[
            "control"
        ] != (expected_control):
            raise ValueError(
                f"Seed edge {edge_id!r} has an invalid joint-state binding."
            )
    policy = action["motion_policy"]
    if policy not in _MOTION_POLICIES:
        raise ValueError(f"Seed edge {edge_id!r} requires a motion policy ID.")
    action_class = str(action["atomic_action_class"])
    expected_policy = _ACTION_POLICIES.get(action_class)
    if action_class == "MoveJoints":
        expected_policy = (
            "default_home" if binding.get("source") == "initial" else "default_release"
        )
    policy_matches = (
        policy in expected_policy
        if isinstance(expected_policy, set)
        else policy == expected_policy
    )
    if not policy_matches:
        raise ValueError(
            f"Seed edge {edge_id!r} action {action_class!r} requires motion "
            f"policy {expected_policy!r}."
        )


def _validate_action_dag(
    start: str,
    goal: str,
    node_ids: set[str],
    edges: Sequence[Mapping[str, Any]],
) -> None:
    """Validate both the displayed state DAG and the action dependency DAG."""
    successors = {node_id: [] for node_id in node_ids}
    predecessors = {node_id: [] for node_id in node_ids}
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        successors[source].append(target)
        predecessors[target].append(source)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("Seed v5 state topology contains a cycle.")
        if node_id in visited:
            return
        visiting.add(node_id)
        for successor in successors[node_id]:
            visit(successor)
        visiting.remove(node_id)
        visited.add(node_id)

    visit(start)
    if visited != node_ids:
        raise ValueError(
            "Seed v5 state topology requires every node to be reachable from start."
        )

    reverse_reachable = {goal}
    pending = [goal]
    while pending:
        node_id = pending.pop()
        for predecessor in predecessors[node_id]:
            if predecessor not in reverse_reachable:
                reverse_reachable.add(predecessor)
                pending.append(predecessor)
    if reverse_reachable != node_ids:
        raise ValueError(
            "Seed v5 state topology requires every node to be able to reach goal."
        )


def _validate_arrangement_steps(
    steps: Sequence[Mapping[str, Any]],
    edge_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate the coordinate-free arrangement contract and six-edge skeleton."""
    expected_actions = [
        "PickUp",
        "MoveHeldObject",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ]
    objects: list[str] | None = None
    slots: list[int] = []
    constraints: set[str] = set()
    for step in steps:
        step_id = str(step["id"])
        if step["operator"] != "place_in_line":
            raise ValueError(
                f"Arrangement semantic step {step_id!r} must use place_in_line."
            )
        goal = step["goal"]
        required_goal_fields = {
            "anchor",
            "axis",
            "layout",
            "nominal_slot_index",
            "objects",
            "order_constraint",
            "order_by",
            "order_direction",
            "orientation_axis",
            "orientation_goal",
            "slot_constraint",
        }
        if set(goal) != required_goal_fields:
            raise ValueError(
                f"Arrangement semantic step {step_id!r} has invalid goal fields."
            )
        if goal["layout"] != "line":
            raise ValueError(
                f"Arrangement semantic step {step_id!r} must use line layout."
            )
        if goal["anchor"] != "table_center":
            raise ValueError(
                f"Arrangement semantic step {step_id!r} must use table_center."
            )
        if goal["axis"] not in {"world_x", "world_y"}:
            raise ValueError(
                f"Arrangement semantic step {step_id!r} has an invalid world axis."
            )
        if goal["slot_constraint"] not in {"free_reassignable", "required"}:
            raise ValueError(
                f"Arrangement semantic step {step_id!r} has invalid slot constraint."
            )
        expected_constraint = (
            "required" if goal["order_constraint"] == "ordered" else "free_reassignable"
        )
        if goal["slot_constraint"] != expected_constraint:
            raise ValueError(
                f"Arrangement semantic step {step_id!r} has inconsistent constraints."
            )
        step_objects = goal["objects"]
        if (
            not isinstance(step_objects, list)
            or not step_objects
            or not all(isinstance(uid, str) and uid for uid in step_objects)
        ):
            raise ValueError(
                f"Arrangement semantic step {step_id!r} requires object UIDs."
            )
        if objects is None:
            objects = list(step_objects)
        elif step_objects != objects:
            raise ValueError("Arrangement steps must share one nominal object order.")
        slot = goal["nominal_slot_index"]
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise ValueError(
                f"Arrangement semantic step {step_id!r} needs an integer slot."
            )
        slots.append(slot)
        constraints.add(str(goal["slot_constraint"]))

        assigned_edges = [edge_by_id[str(edge_id)] for edge_id in step["edge_ids"]]
        action_classes = [
            str(edge["actions"][0]["atomic_action_class"]) for edge in assigned_edges
        ]
        if action_classes != expected_actions:
            raise ValueError(
                f"Arrangement semantic step {step_id!r} must use the six-edge "
                "pickup/staging/final/place/retreat/home topology."
            )
        phases = [
            assigned_edges[index]["actions"][0]["target_binding"].get("phase")
            for index in (1, 2)
        ]
        if phases != ["staging", "final"]:
            raise ValueError(
                f"Arrangement semantic step {step_id!r} must bind staging then final."
            )
        postcondition = step["postcondition"]
        if (
            postcondition.get("nominal_slot_index") != slot
            or postcondition.get("slot_constraint") != goal["slot_constraint"]
        ):
            raise ValueError(
                f"Arrangement semantic step {step_id!r} postcondition is inconsistent."
            )

    expected_slots = list(range(len(steps)))
    if slots != expected_slots:
        raise ValueError(
            "Arrangement nominal slots must cover [0, count) in spatial order."
        )
    if objects != [str(step["object"]) for step in steps]:
        raise ValueError(
            "Arrangement objects and semantic steps must share nominal spatial order."
        )
    if len(constraints) != 1:
        raise ValueError("Arrangement steps must use one consistent slot constraint.")


def _reject_grounded_fields(value: Any, path: str = "seed_task_graph") -> None:
    if isinstance(value, float):
        raise ValueError(
            f"Seed task graph must not contain geometric float value at {path}."
        )
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            normalized_key = key_text.lower()
            is_grounded_field = normalized_key != "execution_policy" and (
                normalized_key in _GROUNDED_FIELD_NAMES
                or normalized_key in _DEFERRED_POLICY_FIELDS
                or normalized_key == "absolute_axis_target"
                or normalized_key == "axis_target"
                or normalized_key.endswith("_pose")
                or normalized_key.startswith(("execution_", "ik_", "planning_"))
                or any(
                    fragment in normalized_key for fragment in _GROUNDED_FIELD_FRAGMENTS
                )
            )
            if is_grounded_field:
                raise ValueError(
                    f"Seed task graph must not contain grounded field "
                    f"{path}.{key_text}."
                )
            _reject_grounded_fields(child, f"{path}.{key_text}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_grounded_fields(child, f"{path}[{index}]")
