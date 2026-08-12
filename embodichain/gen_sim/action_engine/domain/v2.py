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

"""Strict coordinate-free contracts for Action Engine SeedGraph v3."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from embodichain.gen_sim.action_engine.protocol import (
    SCENE_REQUIREMENTS_SCHEMA,
    SEED_GRAPH_SCHEMA,
    TASK_SPEC_SCHEMA,
)
from .motion import validate_motion_policy

__all__ = [
    "REASONING_TYPES",
    "TASK_LEVELS",
    "TASK_TYPES",
    "public_task_spec",
    "seed_graph_hash",
    "validate_public_task_spec",
    "validate_scene_requirements",
    "validate_seed_graph",
    "validate_task_spec",
]

TASK_LEVELS = frozenset({"L1", "L2", "L3", "L4"})
TASK_TYPES = frozenset({f"E{index}" for index in range(1, 10)})
REASONING_TYPES = frozenset(
    {
        "none",
        "memory",
        "visual_semantics",
        "pattern",
        "logic",
        "common_sense",
        "constraint",
    }
)

_TASK_SPEC_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "level",
        "instruction",
        "reasoning_type",
        "task_instances",
        "success",
        "oracle",
        "metadata",
    }
)
_TASK_INSTANCE_KEYS = frozenset({"id", "task_type", "params", "depends_on", "role"})
_SCENE_REQUIREMENTS_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "objects",
        "cameras",
        "spatial_constraints",
        "distractor_count",
        "metadata",
    }
)
_OBJECT_REQUIREMENT_KEYS = frozenset(
    {
        "role_id",
        "category",
        "count",
        "affordances",
        "initial_state",
        "attributes",
    }
)
_SEED_GRAPH_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "instruction",
        "level",
        "reasoning_type",
        "planner_route",
        "nodes",
        "task_groups",
        "success",
        "capability_catalog_hash",
        "metadata",
    }
)
_ACTION_NODE_KEYS = frozenset(
    {
        "id",
        "atomic_action",
        "object_uid",
        "actor",
        "control",
        "target_binding",
        "depends_on",
        "contract",
        "task_instance_id",
        "task_type",
        "role",
        "precondition",
        "postcondition",
        "motion_policy",
        "sync_group",
    }
)
_TASK_GROUP_KEYS = frozenset(
    {
        "id",
        "task_type",
        "role",
        "operator",
        "object_uid",
        "actor",
        "goal",
        "depends_on",
        "parent_task_instance_id",
        "node_ids",
        "success",
        "contract",
    }
)
_ACTOR_MODES = frozenset({"auto", "required", "preferred", "coordinated"})
_NODE_ROLES = frozenset({"primary", "recovery", "cleanup"})
_GROUP_ROLES = frozenset({"primary", "recovery"})
_PLANNER_ROUTES = frozenset({"offline", "online", "selected", "fused"})
_ACTION_CONTRACT_KEYS = frozenset(
    {"version", "requires", "effects", "claims", "completion"}
)
_TASK_GROUP_CONTRACT_KEYS = frozenset(
    {
        "entry_requires",
        "exit_effects",
        "claims",
        "entry_node_ids",
        "terminal_node_ids",
        "completion",
    }
)
_STATE_ATOM_KEYS = frozenset({"predicate", "object_uid", "arm"})
_STATE_PREDICATES = frozenset(
    {
        "arm_free",
        "object_free",
        "object_held",
        "object_coordinated_held",
        "handover_complete",
        "arm_clear",
        "arm_home",
    }
)
_EFFECT_KEYS = frozenset({"op", "atom"})
_EFFECT_OPERATIONS = frozenset({"add", "delete"})
_CLAIM_KEYS = frozenset({"resource", "access", "lifetime"})
_CLAIM_ACCESS = frozenset({"shared_read", "exclusive"})
_CLAIM_LIFETIMES = frozenset({"action", "until_release"})
_ACTION_COMPLETION = frozenset({"ordinary", "cleanup", "terminal_barrier"})
_GROUP_COMPLETION = frozenset({"ordinary", "terminal_barrier"})
_OBJECT_REFERENCE_KEYS = frozenset(
    {
        "anchor",
        "container",
        "object",
        "object_uid",
        "placing_object",
        "reference",
        "reference_object",
        "support",
        "support_object",
    }
)
_GROUNDED_FIELD_NAMES = frozenset(
    {
        "absolute_position",
        "coordinates",
        "grasp_pose",
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


def validate_task_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one task-first, scene-independent task specification."""
    result = _mapping(value, "TaskSpec")
    _keys(result, _TASK_SPEC_KEYS, "TaskSpec")
    _schema(result, TASK_SPEC_SCHEMA, "TaskSpec")
    _string(result.get("task_id"), "TaskSpec.task_id")
    level = _enum(result.get("level"), TASK_LEVELS, "TaskSpec.level")
    _string(result.get("instruction"), "TaskSpec.instruction")
    reasoning = _enum(
        result.get("reasoning_type", "none"),
        REASONING_TYPES,
        "TaskSpec.reasoning_type",
    )
    if level == "L4" and reasoning == "none":
        raise ValueError("TaskSpec L4 tasks require a non-'none' reasoning_type.")
    if level != "L4" and reasoning != "none":
        raise ValueError("Only TaskSpec L4 tasks may declare reasoning_type.")

    instances: list[dict[str, Any]] = []
    for index, item in enumerate(
        _sequence(result.get("task_instances"), "TaskSpec.task_instances")
    ):
        context = f"TaskSpec.task_instances[{index}]"
        instance = _mapping(item, context)
        _keys(instance, _TASK_INSTANCE_KEYS, context)
        _string(instance.get("id"), f"{context}.id")
        _enum(instance.get("task_type"), TASK_TYPES, f"{context}.task_type")
        instance["params"] = _mapping(instance.get("params", {}), f"{context}.params")
        instance["depends_on"] = _strings(
            instance.get("depends_on", []), f"{context}.depends_on"
        )
        instance["role"] = _enum(
            instance.get("role", "primary"), _GROUP_ROLES, f"{context}.role"
        )
        instances.append(instance)
    if not instances:
        raise ValueError("TaskSpec.task_instances must not be empty.")
    _unique([item["id"] for item in instances], "TaskSpec task instance IDs")
    _dag(
        {item["id"]: item["depends_on"] for item in instances},
        "TaskSpec task instances",
    )
    _validate_level_shape(level, instances)
    result["task_instances"] = instances
    result["success"] = _mapping(result.get("success"), "TaskSpec.success")
    result["oracle"] = _mapping(result.get("oracle", {}), "TaskSpec.oracle")
    result["metadata"] = _mapping(result.get("metadata", {}), "TaskSpec.metadata")
    _reject_grounded(result)
    _finite(result)
    return result


def public_task_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated TaskSpec view safe to expose to an online agent."""
    if "task_instances" not in value and value.get("level") == "L4":
        result = dict(value)
        _strip_public_private_metadata(result)
        return validate_public_task_spec(result)
    result = validate_task_spec(value)
    result.pop("oracle", None)
    if result["level"] == "L4":
        # L4 task instances are the hidden reference plan, not public intent.
        result.pop("task_instances", None)
    _strip_public_private_metadata(result)
    return validate_public_task_spec(result)


def _strip_public_private_metadata(value: dict[str, Any]) -> None:
    """Remove role/UID bindings that would turn the public view into an oracle."""
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        return
    private_keys = {
        "role_bindings",
        "uid_map",
        "source_uid_map",
        "reference_seed_graph",
        "oracle",
    }

    def strip(child: Any) -> Any:
        if isinstance(child, Mapping):
            return {
                key: strip(nested)
                for key, nested in child.items()
                if str(key).lower() not in private_keys
            }
        if isinstance(child, list):
            return [strip(item) for item in child]
        if isinstance(child, tuple):
            return [strip(item) for item in child]
        return deepcopy(child)

    value["metadata"] = strip(metadata)


def validate_public_task_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the oracle-free TaskSpec projection consumed online."""
    result = _mapping(value, "PublicTaskSpec")
    allowed = _TASK_SPEC_KEYS - {"oracle"}
    _keys(result, allowed, "PublicTaskSpec")
    if "oracle" in result:
        raise ValueError("PublicTaskSpec must not contain oracle data.")
    _schema(result, TASK_SPEC_SCHEMA, "PublicTaskSpec")
    _string(result.get("task_id"), "PublicTaskSpec.task_id")
    level = _enum(result.get("level"), TASK_LEVELS, "PublicTaskSpec.level")
    _string(result.get("instruction"), "PublicTaskSpec.instruction")
    reasoning = _enum(
        result.get("reasoning_type", "none"),
        REASONING_TYPES,
        "PublicTaskSpec.reasoning_type",
    )
    if (level == "L4") != (reasoning != "none"):
        raise ValueError(
            "PublicTaskSpec reasoning_type must be non-'none' exactly for L4."
        )
    if level == "L4":
        if "task_instances" in result:
            raise ValueError("Public L4 TaskSpec must hide reference task instances.")
    else:
        # Reuse the complete structural validator for explicit L1-L3 tasks.
        normalized = validate_task_spec({**result, "oracle": {}})
        normalized.pop("oracle", None)
        return normalized
    result["success"] = _mapping(result.get("success"), "PublicTaskSpec.success")
    result["metadata"] = _mapping(result.get("metadata", {}), "PublicTaskSpec.metadata")
    _reject_grounded(result)
    _finite(result)
    return result


def validate_scene_requirements(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the structured hand-off contract consumed by a Scene Engine."""
    result = _mapping(value, "SceneRequirements")
    _keys(result, _SCENE_REQUIREMENTS_KEYS, "SceneRequirements")
    _schema(result, SCENE_REQUIREMENTS_SCHEMA, "SceneRequirements")
    _string(result.get("task_id"), "SceneRequirements.task_id")
    objects: list[dict[str, Any]] = []
    for index, item in enumerate(
        _sequence(result.get("objects"), "SceneRequirements.objects")
    ):
        context = f"SceneRequirements.objects[{index}]"
        requirement = _mapping(item, context)
        _keys(requirement, _OBJECT_REQUIREMENT_KEYS, context)
        _string(requirement.get("role_id"), f"{context}.role_id")
        _string(requirement.get("category"), f"{context}.category")
        count = requirement.get("count", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{context}.count must be a positive integer.")
        requirement["count"] = count
        requirement["affordances"] = _strings(
            requirement.get("affordances", []), f"{context}.affordances"
        )
        requirement["initial_state"] = _mapping(
            requirement.get("initial_state", {}), f"{context}.initial_state"
        )
        requirement["attributes"] = _mapping(
            requirement.get("attributes", {}), f"{context}.attributes"
        )
        objects.append(requirement)
    if not objects:
        raise ValueError("SceneRequirements.objects must not be empty.")
    _unique([item["role_id"] for item in objects], "SceneRequirements role IDs")
    result["objects"] = objects
    result["cameras"] = [
        _mapping(item, f"SceneRequirements.cameras[{index}]")
        for index, item in enumerate(
            _sequence(result.get("cameras", []), "SceneRequirements.cameras")
        )
    ]
    result["spatial_constraints"] = [
        _mapping(item, f"SceneRequirements.spatial_constraints[{index}]")
        for index, item in enumerate(
            _sequence(
                result.get("spatial_constraints", []),
                "SceneRequirements.spatial_constraints",
            )
        )
    ]
    distractors = result.get("distractor_count", 0)
    if (
        not isinstance(distractors, int)
        or isinstance(distractors, bool)
        or distractors < 0
    ):
        raise ValueError("SceneRequirements.distractor_count must be non-negative.")
    result["distractor_count"] = distractors
    result["metadata"] = _mapping(
        result.get("metadata", {}), "SceneRequirements.metadata"
    )
    _finite(result)
    return result


def validate_seed_graph(
    value: Mapping[str, Any],
    *,
    known_objects: Collection[str] | None = None,
    known_actions: Collection[str] | None = None,
    executable_actions: Collection[str] | None = None,
    require_executable: bool = False,
) -> dict[str, Any]:
    """Validate a direct, coordinate-free AtomicAction DAG."""
    result = _mapping(value, "SeedGraph")
    _keys(result, _SEED_GRAPH_KEYS, "SeedGraph")
    _schema(result, SEED_GRAPH_SCHEMA, "SeedGraph")
    _string(result.get("task_id"), "SeedGraph.task_id")
    _string(result.get("instruction"), "SeedGraph.instruction")
    level = _enum(result.get("level"), TASK_LEVELS, "SeedGraph.level")
    reasoning = _enum(
        result.get("reasoning_type", "none"),
        REASONING_TYPES,
        "SeedGraph.reasoning_type",
    )
    if (level == "L4") != (reasoning != "none"):
        raise ValueError("SeedGraph reasoning_type must be non-'none' exactly for L4.")
    result["planner_route"] = _enum(
        result.get("planner_route"), _PLANNER_ROUTES, "SeedGraph.planner_route"
    )
    _string(result.get("capability_catalog_hash"), "SeedGraph.capability_catalog_hash")

    nodes: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(result.get("nodes"), "SeedGraph.nodes")):
        context = f"SeedGraph.nodes[{index}]"
        node = _mapping(item, context)
        _keys(node, _ACTION_NODE_KEYS, context)
        _string(node.get("id"), f"{context}.id")
        action = _string(node.get("atomic_action"), f"{context}.atomic_action")
        if known_actions is not None and action not in set(known_actions):
            raise ValueError(f"{context} references unknown AtomicAction {action!r}.")
        if (
            require_executable
            and executable_actions is not None
            and action not in set(executable_actions)
        ):
            raise ValueError(
                f"AtomicAction {action!r} is planning-only and cannot be executed."
            )
        object_uid = _string(node.get("object_uid"), f"{context}.object_uid")
        if known_objects is not None and object_uid not in set(known_objects):
            raise ValueError(f"{context} references unknown object {object_uid!r}.")
        node["actor"] = _actor(node.get("actor", {"mode": "auto"}), f"{context}.actor")
        node["control"] = _string(node.get("control", "arm"), f"{context}.control")
        binding = _mapping(node.get("target_binding"), f"{context}.target_binding")
        _string(binding.get("kind"), f"{context}.target_binding.kind")
        node["target_binding"] = binding
        node["depends_on"] = _strings(
            node.get("depends_on", []), f"{context}.depends_on"
        )
        node["contract"] = _action_contract(node.get("contract"), f"{context}.contract")
        node["task_instance_id"] = _string(
            node.get("task_instance_id"), f"{context}.task_instance_id"
        )
        node["task_type"] = _enum(
            node.get("task_type"), TASK_TYPES, f"{context}.task_type"
        )
        node["role"] = _enum(
            node.get("role", "primary"), _NODE_ROLES, f"{context}.role"
        )
        node["precondition"] = _mapping(
            node.get("precondition", {}), f"{context}.precondition"
        )
        node["postcondition"] = _mapping(
            node.get("postcondition", {}), f"{context}.postcondition"
        )
        node["motion_policy"] = validate_motion_policy(
            node.get("motion_policy"), f"{context}.motion_policy"
        )
        if "sync_group" in node:
            node["sync_group"] = _string(node["sync_group"], f"{context}.sync_group")
        nodes.append(node)
    if not nodes:
        raise ValueError("SeedGraph.nodes must not be empty.")
    node_ids = [node["id"] for node in nodes]
    _unique(node_ids, "SeedGraph node IDs")
    _dag({node["id"]: node["depends_on"] for node in nodes}, "SeedGraph nodes")

    groups: list[dict[str, Any]] = []
    for index, item in enumerate(
        _sequence(result.get("task_groups"), "SeedGraph.task_groups")
    ):
        context = f"SeedGraph.task_groups[{index}]"
        group = _mapping(item, context)
        _keys(group, _TASK_GROUP_KEYS, context)
        group["id"] = _string(group.get("id"), f"{context}.id")
        group["task_type"] = _enum(
            group.get("task_type"), TASK_TYPES, f"{context}.task_type"
        )
        group["role"] = _enum(
            group.get("role", "primary"), _GROUP_ROLES, f"{context}.role"
        )
        group["operator"] = _string(group.get("operator"), f"{context}.operator")
        group["object_uid"] = _string(group.get("object_uid"), f"{context}.object_uid")
        group["actor"] = _actor(
            group.get("actor", {"mode": "auto"}), f"{context}.actor"
        )
        group["goal"] = _mapping(group.get("goal", {}), f"{context}.goal")
        group["depends_on"] = _strings(
            group.get("depends_on", []), f"{context}.depends_on"
        )
        if "parent_task_instance_id" in group:
            group["parent_task_instance_id"] = _string(
                group["parent_task_instance_id"],
                f"{context}.parent_task_instance_id",
            )
        group["node_ids"] = _strings(group.get("node_ids"), f"{context}.node_ids")
        if not group["node_ids"]:
            raise ValueError(f"{context}.node_ids must not be empty.")
        group["success"] = _mapping(group.get("success"), f"{context}.success")
        group["contract"] = _task_group_contract(
            group.get("contract"), f"{context}.contract"
        )
        groups.append(group)
    if not groups:
        raise ValueError("SeedGraph.task_groups must not be empty.")
    _validate_groups(nodes, groups)
    _validate_group_contract_topology(nodes, groups)
    _validate_cleanup_barriers(nodes, groups)
    _validate_task_group_semantics(nodes, groups)
    _validate_ownership_transitions(nodes, groups)
    _validate_resource_conflicts(nodes, groups, result.get("metadata", {}))
    _dag(
        {group["id"]: group["depends_on"] for group in groups},
        "SeedGraph task groups",
    )
    _validate_group_dependency_alignment(nodes, groups)
    result["nodes"] = nodes
    result["task_groups"] = groups
    result["success"] = _mapping(result.get("success"), "SeedGraph.success")
    result["metadata"] = _mapping(result.get("metadata", {}), "SeedGraph.metadata")
    _reject_grounded(result)
    _finite(result)
    if known_objects is not None:
        _known_object_references(result, set(known_objects))
    return result


def seed_graph_hash(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for one validated SeedGraph."""
    canonical = validate_seed_graph(value)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_level_shape(level: str, instances: Sequence[Mapping[str, Any]]) -> None:
    primary = [item for item in instances if item["role"] == "primary"]
    types = {str(item["task_type"]) for item in primary}
    if level == "L1" and len(primary) != 1:
        raise ValueError("L1 requires exactly one primary task instance.")
    if level == "L2" and (len(primary) < 2 or len(types) != 1):
        raise ValueError("L2 requires at least two primary instances of one E type.")
    if level == "L3" and (len(primary) < 2 or len(types) < 2):
        raise ValueError(
            "L3 requires at least two primary instances of different E types."
        )


def _validate_groups(
    nodes: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> None:
    node_by_id = {str(node["id"]): node for node in nodes}
    _unique([str(group["id"]) for group in groups], "SeedGraph task group IDs")
    memberships: dict[str, str] = {}
    for group in groups:
        group_id = str(group["id"])
        for node_id in group["node_ids"]:
            if node_id not in node_by_id:
                raise ValueError(
                    f"SeedGraph task group {group_id!r} references unknown node {node_id!r}."
                )
            if node_id in memberships:
                raise ValueError(
                    f"SeedGraph node {node_id!r} belongs to multiple task groups."
                )
            node = node_by_id[node_id]
            if node["task_instance_id"] != group_id:
                raise ValueError(
                    f"SeedGraph node {node_id!r} task_instance_id does not match {group_id!r}."
                )
            if node["task_type"] != group["task_type"]:
                raise ValueError(
                    f"SeedGraph node {node_id!r} task_type does not match its group."
                )
            memberships[node_id] = group_id
    missing = sorted(set(node_by_id) - set(memberships))
    if missing:
        raise ValueError(
            f"SeedGraph nodes are missing task group membership: {missing}."
        )


def _validate_task_group_semantics(
    nodes: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> None:
    node_by_id = {str(node["id"]): node for node in nodes}
    required_actions = {
        "E1": set(),
        "E2": {"MoveHeldObject", "Place"},
        "E3": {"Pour"},
        "E4": {"HandOver"},
        "E5": set(),
        "E6": {"PullArticulatedPart"},
        "E7": {"PushArticulatedPart"},
        "E8": {"TurnKnob"},
        "E9": {"Press"},
    }
    for group in groups:
        task_type = str(group["task_type"])
        group_nodes = [node_by_id[node_id] for node_id in group["node_ids"]]
        actions = {str(node["atomic_action"]) for node in group_nodes}
        missing = required_actions[task_type] - actions
        if task_type == "E1":
            if not actions.intersection({"MoveHeldObject", "Place"}):
                missing = {"MoveHeldObject|Place"}
            elif "PickUp" not in actions:
                first = group_nodes[0]
                precondition = first.get("precondition", {})
                if precondition.get("type") != "object_held":
                    missing = {"PickUp|object_held precondition"}
        if task_type == "E2" and "PickUp" not in actions:
            first = group_nodes[0]
            precondition = first.get("precondition", {})
            if precondition.get("type") != "object_held":
                missing = {"PickUp|object_held precondition"}
        # Recovery may explicitly preserve a verified downstream hold. Ordinary
        # E2 groups always complete their supported world state with Place.
        if (
            task_type == "E2"
            and "Place" not in actions
            and group.get("goal", {}).get("terminal_behavior") == "hold"
            and "MoveHeldObject" in actions
            and group.get("role") == "recovery"
        ):
            missing.discard("Place")
        if task_type == "E5" and not actions.intersection(
            {"CoordinatedPickment", "CoordinatedPlacement"}
        ):
            missing = {"CoordinatedPickment|CoordinatedPlacement"}
        if missing:
            raise ValueError(
                f"SeedGraph TaskGroup {group['id']!r} is missing {task_type} "
                f"core actions: {sorted(missing)}."
            )


def _validate_ownership_transitions(
    nodes: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> None:
    """Check release/reacquire and explicit single-arm hold transitions.

    An ordinary E2 -> E4 transition persists the supported upright state, ends
    the predecessor resource lease, and lets E4 acquire a fresh transfer grasp.
    E4 -> E1 keeps receiver ownership because the exchanged object is not yet
    supported. Recovery groups may preserve an explicitly requested hold.
    """
    node_by_id = {str(node["id"]): node for node in nodes}
    group_by_id = {str(group["id"]): group for group in groups}
    nodes_by_group = {
        group_id: [node_by_id[node_id] for node_id in group["node_ids"]]
        for group_id, group in group_by_id.items()
    }

    def direct_predecessor(
        group: Mapping[str, Any], task_type: str
    ) -> Mapping[str, Any] | None:
        for dependency in group.get("depends_on", []):
            candidate = group_by_id.get(str(dependency))
            if candidate is not None and candidate.get("task_type") == task_type:
                return candidate
        return None

    def held_arm(node: Mapping[str, Any]) -> str | None:
        precondition = node.get("precondition", {})
        if (
            isinstance(precondition, Mapping)
            and precondition.get("type") == "object_held"
        ):
            arm = str(precondition.get("arm", ""))
            if arm in {"left_arm", "right_arm"}:
                return arm
        actor = node.get("actor", {})
        if isinstance(actor, Mapping) and actor.get("mode") == "required":
            arm = str(actor.get("arm", ""))
            if arm in {"left_arm", "right_arm"}:
                return arm
        return None

    for group_id, group in group_by_id.items():
        task_type = str(group.get("task_type"))
        group_nodes = nodes_by_group[group_id]
        actions = [str(node.get("atomic_action")) for node in group_nodes]
        object_uid = str(group.get("object_uid"))

        if task_type == "E2":
            handover = next(
                (
                    candidate
                    for candidate in groups
                    if candidate.get("task_type") == "E4"
                    and group_id
                    in {str(item) for item in candidate.get("depends_on", [])}
                    and str(candidate.get("object_uid")) == object_uid
                ),
                None,
            )
            if handover is None:
                continue
            if (
                group.get("goal", {}).get("terminal_behavior") == "hold"
                and group.get("role") != "recovery"
            ):
                raise ValueError(
                    f"SeedGraph E2 group {group_id!r} may not preserve a holder "
                    "across an ordinary E2->E4 TaskGroup boundary."
                )
            if group.get("role") != "recovery" and "Place" not in actions:
                raise ValueError(
                    f"SeedGraph E2 group {group_id!r} must release its supported "
                    "object before E4 reacquires it."
                )

        if task_type == "E4":
            predecessor = direct_predecessor(group, "E2")
            if (
                predecessor is not None
                and str(predecessor.get("object_uid")) == object_uid
            ):
                predecessor_nodes = nodes_by_group[str(predecessor["id"])]
                preserves_hold = (
                    predecessor.get("role") == "recovery"
                    and predecessor.get("goal", {}).get("terminal_behavior") == "hold"
                )
                if preserves_hold:
                    if "PickUp" in actions or not group_nodes:
                        raise ValueError(
                            f"SeedGraph E4 group {group_id!r} must consume the "
                            "recovery-held object without PickUp."
                        )
                    first = group_nodes[0]
                    holder_arm = next(
                        (
                            held_arm(node)
                            for node in reversed(predecessor_nodes)
                            if held_arm(node) is not None
                        ),
                        None,
                    )
                    if (
                        str(first.get("atomic_action")) != "MoveHeldObject"
                        or held_arm(first) is None
                        or held_arm(first) != holder_arm
                    ):
                        raise ValueError(
                            f"SeedGraph E2->E4 recovery holder mismatch for object "
                            f"{object_uid!r}."
                        )
                else:
                    predecessor_actions = {
                        str(node.get("atomic_action")) for node in predecessor_nodes
                    }
                    if "Place" not in predecessor_actions:
                        raise ValueError(
                            f"SeedGraph E2 predecessor {predecessor['id']!r} must "
                            "release its object before E4."
                        )
                    if (
                        not group_nodes
                        or str(group_nodes[0].get("atomic_action")) != "PickUp"
                    ):
                        raise ValueError(
                            f"SeedGraph E4 group {group_id!r} must reacquire the "
                            "supported E2 object with PickUp."
                        )

        if task_type == "E1":
            predecessor = direct_predecessor(group, "E4")
            if (
                predecessor is not None
                and str(predecessor.get("object_uid")) == object_uid
            ):
                if "PickUp" in actions or not group_nodes:
                    raise ValueError(
                        f"SeedGraph E1 group {group_id!r} must preserve the E4 receiver hold "
                        "without PickUp."
                    )
                first = group_nodes[0]
                if (
                    str(first.get("atomic_action")) != "MoveHeldObject"
                    or held_arm(first) is None
                ):
                    raise ValueError(
                        f"SeedGraph E1 group {group_id!r} must start with MoveHeldObject "
                        "from the receiver hold."
                    )


def _validate_group_dependency_alignment(
    nodes: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> None:
    group_by_node = {
        str(node_id): str(group["id"])
        for group in groups
        for node_id in group["node_ids"]
    }
    group_dependencies = {
        str(group["id"]): set(str(parent) for parent in group["depends_on"])
        for group in groups
    }

    def group_reaches(child: str, parent: str) -> bool:
        pending = list(group_dependencies[child])
        visited = set()
        while pending:
            current = pending.pop()
            if current == parent:
                return True
            if current not in visited:
                visited.add(current)
                pending.extend(group_dependencies[current])
        return False

    for node in nodes:
        child_group = group_by_node[str(node["id"])]
        for dependency in node["depends_on"]:
            parent_group = group_by_node[str(dependency)]
            if parent_group != child_group and not group_reaches(
                child_group, parent_group
            ):
                raise ValueError(
                    f"SeedGraph node {node['id']!r} depends on TaskGroup "
                    f"{parent_group!r}, but TaskGroup {child_group!r} does not."
                )


def _validate_resource_conflicts(
    nodes: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    metadata: Any,
) -> None:
    dependencies = {
        str(node["id"]): set(str(item) for item in node["depends_on"]) for node in nodes
    }

    def reaches(child: str, parent: str) -> bool:
        pending = list(dependencies[child])
        visited = set()
        while pending:
            current = pending.pop()
            if current == parent:
                return True
            if current not in visited:
                visited.add(current)
                pending.extend(dependencies[current])
        return False

    group_by_node = {
        str(node_id): str(group["id"])
        for group in groups
        for node_id in group["node_ids"]
    }
    distinct_arm_pairs = _distinct_arm_pairs(metadata)
    for index, first in enumerate(nodes):
        for second in nodes[index + 1 :]:
            first_id = str(first["id"])
            second_id = str(second["id"])
            if reaches(first_id, second_id) or reaches(second_id, first_id):
                continue
            if (
                first.get("sync_group") == second.get("sync_group")
                and first.get("sync_group") is not None
            ):
                continue
            first_claims = {
                str(claim["resource"]): str(claim["access"])
                for claim in first["contract"]["claims"]
            }
            second_claims = {
                str(claim["resource"]): str(claim["access"])
                for claim in second["contract"]["claims"]
            }
            conflicts = sorted(
                resource
                for resource in set(first_claims) & set(second_claims)
                if "exclusive" in {first_claims[resource], second_claims[resource]}
            )
            if (
                frozenset({group_by_node[first_id], group_by_node[second_id]})
                in distinct_arm_pairs
            ):
                conflicts = [item for item in conflicts if item != "arm:auto"]
            if conflicts:
                raise ValueError(
                    f"SeedGraph concurrent nodes {first_id!r} and {second_id!r} "
                    f"have resource conflicts: {conflicts}."
                )


def _distinct_arm_pairs(value: Any) -> set[frozenset[str]]:
    if not isinstance(value, Mapping):
        return set()
    groups = value.get("legacy_allocation_groups", value.get("allocation_groups", ()))
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes, bytearray)):
        return set()
    result: set[frozenset[str]] = set()
    for group in groups:
        if (
            not isinstance(group, Mapping)
            or group.get("arm_constraint") != "distinct_arms"
        ):
            continue
        members = group.get("semantic_step_ids", group.get("task_instance_ids", ()))
        if not isinstance(members, Sequence) or isinstance(
            members, (str, bytes, bytearray)
        ):
            continue
        member_ids = [str(item) for item in members]
        for index, first in enumerate(member_ids):
            for second in member_ids[index + 1 :]:
                result.add(frozenset({first, second}))
    return result


def _action_contract(value: Any, context: str) -> dict[str, Any]:
    contract = _mapping(value, context)
    _keys(contract, _ACTION_CONTRACT_KEYS, context)
    if set(contract) != _ACTION_CONTRACT_KEYS:
        missing = sorted(_ACTION_CONTRACT_KEYS - set(contract))
        raise ValueError(f"{context} is missing required fields: {missing}.")
    if contract["version"] != "action_contract_v1":
        raise ValueError(f"{context}.version must be 'action_contract_v1'.")
    contract["requires"] = [
        _state_atom(item, f"{context}.requires[{index}]")
        for index, item in enumerate(
            _sequence(contract["requires"], f"{context}.requires")
        )
    ]
    contract["effects"] = [
        _state_effect(item, f"{context}.effects[{index}]")
        for index, item in enumerate(
            _sequence(contract["effects"], f"{context}.effects")
        )
    ]
    contract["claims"] = [
        _resource_claim(item, f"{context}.claims[{index}]")
        for index, item in enumerate(_sequence(contract["claims"], f"{context}.claims"))
    ]
    _unique(
        [str(item["resource"]) for item in contract["claims"]],
        f"{context}.claims resources",
    )
    contract["completion"] = _enum(
        contract["completion"], _ACTION_COMPLETION, f"{context}.completion"
    )
    return contract


def _task_group_contract(value: Any, context: str) -> dict[str, Any]:
    contract = _mapping(value, context)
    _keys(contract, _TASK_GROUP_CONTRACT_KEYS, context)
    if set(contract) != _TASK_GROUP_CONTRACT_KEYS:
        missing = sorted(_TASK_GROUP_CONTRACT_KEYS - set(contract))
        raise ValueError(f"{context} is missing required fields: {missing}.")
    contract["entry_requires"] = [
        _state_atom(item, f"{context}.entry_requires[{index}]")
        for index, item in enumerate(
            _sequence(contract["entry_requires"], f"{context}.entry_requires")
        )
    ]
    contract["exit_effects"] = [
        _state_effect(item, f"{context}.exit_effects[{index}]")
        for index, item in enumerate(
            _sequence(contract["exit_effects"], f"{context}.exit_effects")
        )
    ]
    contract["claims"] = [
        _resource_claim(item, f"{context}.claims[{index}]")
        for index, item in enumerate(_sequence(contract["claims"], f"{context}.claims"))
    ]
    _unique(
        [str(item["resource"]) for item in contract["claims"]],
        f"{context}.claims resources",
    )
    contract["entry_node_ids"] = _strings(
        contract["entry_node_ids"], f"{context}.entry_node_ids"
    )
    contract["terminal_node_ids"] = _strings(
        contract["terminal_node_ids"], f"{context}.terminal_node_ids"
    )
    if not contract["entry_node_ids"] or not contract["terminal_node_ids"]:
        raise ValueError(f"{context} requires entry and terminal node IDs.")
    contract["completion"] = _enum(
        contract["completion"], _GROUP_COMPLETION, f"{context}.completion"
    )
    return contract


def _state_atom(value: Any, context: str) -> dict[str, str]:
    atom = _mapping(value, context)
    _keys(atom, _STATE_ATOM_KEYS, context)
    predicate = _enum(atom.get("predicate"), _STATE_PREDICATES, f"{context}.predicate")
    required = {
        "arm_free": {"arm"},
        "object_free": {"object_uid"},
        "object_held": {"object_uid", "arm"},
        "object_coordinated_held": {"object_uid"},
        "handover_complete": {"object_uid"},
        "arm_clear": {"arm"},
        "arm_home": {"arm"},
    }[predicate]
    present = set(atom) - {"predicate"}
    if present != required:
        raise ValueError(
            f"{context} predicate {predicate!r} requires exactly {sorted(required)}."
        )
    for field in required:
        atom[field] = _string(atom.get(field), f"{context}.{field}")
    return atom


def _state_effect(value: Any, context: str) -> dict[str, Any]:
    effect = _mapping(value, context)
    _keys(effect, _EFFECT_KEYS, context)
    if set(effect) != _EFFECT_KEYS:
        raise ValueError(f"{context} requires op and atom.")
    effect["op"] = _enum(effect["op"], _EFFECT_OPERATIONS, f"{context}.op")
    effect["atom"] = _state_atom(effect["atom"], f"{context}.atom")
    return effect


def _resource_claim(value: Any, context: str) -> dict[str, str]:
    claim = _mapping(value, context)
    _keys(claim, _CLAIM_KEYS, context)
    if set(claim) != _CLAIM_KEYS:
        raise ValueError(f"{context} requires resource, access, and lifetime.")
    claim["resource"] = _string(claim["resource"], f"{context}.resource")
    claim["access"] = _enum(claim["access"], _CLAIM_ACCESS, f"{context}.access")
    claim["lifetime"] = _enum(
        claim["lifetime"], _CLAIM_LIFETIMES, f"{context}.lifetime"
    )
    return claim


def _validate_group_contract_topology(
    nodes: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> None:
    node_by_id = {str(node["id"]): node for node in nodes}
    children: dict[str, set[str]] = {node_id: set() for node_id in node_by_id}
    for node in nodes:
        for dependency in node["depends_on"]:
            children[str(dependency)].add(str(node["id"]))
    for group in groups:
        group_id = str(group["id"])
        node_ids = {str(item) for item in group["node_ids"]}
        expected_entries = {
            node_id
            for node_id in node_ids
            if not any(
                str(parent) in node_ids for parent in node_by_id[node_id]["depends_on"]
            )
        }
        expected_terminals = {
            node_id for node_id in node_ids if not (children[node_id] & node_ids)
        }
        contract = group["contract"]
        if set(contract["entry_node_ids"]) != expected_entries:
            raise ValueError(
                f"SeedGraph TaskGroup {group_id!r} contract entry_node_ids do not "
                "match its internal topology."
            )
        if set(contract["terminal_node_ids"]) != expected_terminals:
            raise ValueError(
                f"SeedGraph TaskGroup {group_id!r} contract terminal_node_ids do not "
                "match its internal topology."
            )
        if contract["completion"] == "terminal_barrier" and not all(
            node_by_id[node_id]["contract"]["completion"] == "terminal_barrier"
            for node_id in expected_terminals
        ):
            raise ValueError(
                f"SeedGraph TaskGroup {group_id!r} terminal barrier must end in "
                "terminal_barrier AtomicActions."
            )


def _validate_cleanup_barriers(
    nodes: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> None:
    node_by_id = {str(node["id"]): node for node in nodes}
    for group in groups:
        group_id = str(group["id"])
        group_nodes = [node_by_id[str(node_id)] for node_id in group["node_ids"]]
        cleanup = [
            node for node in group_nodes if node["contract"]["completion"] == "cleanup"
        ]
        has_handover = any(node["atomic_action"] == "HandOver" for node in group_nodes)
        recovery_successor = any(
            candidate.get("role") == "recovery"
            and candidate.get("parent_task_instance_id") == group_id
            and group_id in candidate.get("depends_on", ())
            for candidate in groups
        )
        if has_handover and not cleanup and recovery_successor:
            continue
        if has_handover and not cleanup:
            raise ValueError(
                f"SeedGraph HandOver TaskGroup {group_id!r} is missing retreat cleanup."
            )
        if not cleanup and not has_handover:
            continue
        terminal_ids = group["contract"]["terminal_node_ids"]
        if group["contract"]["completion"] != "terminal_barrier" or not all(
            node_by_id[node_id]["contract"]["completion"] == "terminal_barrier"
            for node_id in terminal_ids
        ):
            raise ValueError(
                f"SeedGraph TaskGroup {group_id!r} cleanup must end at a home "
                "terminal barrier."
            )


def _actor(value: Any, context: str) -> dict[str, Any]:
    actor = _mapping(value, context)
    mode = _enum(actor.get("mode"), _ACTOR_MODES, f"{context}.mode")
    allowed = {"mode"}
    if mode in {"required", "preferred"}:
        allowed.add("arm")
        _string(actor.get("arm"), f"{context}.arm")
    elif mode == "coordinated":
        allowed.add("arms")
        arms = _strings(actor.get("arms"), f"{context}.arms")
        if len(arms) < 2:
            raise ValueError(f"{context}.arms must contain at least two arms.")
        actor["arms"] = arms
    _keys(actor, frozenset(allowed), context)
    return actor


def _known_object_references(
    value: Any, known: set[str], path: str = "SeedGraph"
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in _OBJECT_REFERENCE_KEYS and isinstance(child, str):
                if child not in known and child not in {"table_center", "world"}:
                    raise ValueError(
                        f"{child_path} references unknown object {child!r}."
                    )
            _known_object_references(child, known, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _known_object_references(child, known, f"{path}[{index}]")


def _reject_grounded(value: Any, path: str = "document") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _GROUNDED_FIELD_NAMES:
                raise ValueError(f"{path}.{key} contains grounded motion data.")
            _reject_grounded(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_grounded(child, f"{path}[{index}]")


def _finite(value: Any, path: str = "document") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite.")


def _dag(dependencies: Mapping[str, Sequence[str]], context: str) -> None:
    known = set(dependencies)
    outgoing = {item_id: [] for item_id in known}
    indegree = {item_id: 0 for item_id in known}
    for item_id, required in dependencies.items():
        unknown = set(required) - known
        if unknown:
            raise ValueError(f"{context} reference unknown IDs: {sorted(unknown)}.")
        if item_id in required:
            raise ValueError(f"{context} contain a self-dependency at {item_id!r}.")
        for parent in required:
            outgoing[parent].append(item_id)
            indegree[item_id] += 1
    ready = deque(
        sorted(item_id for item_id, degree in indegree.items() if degree == 0)
    )
    visited = 0
    while ready:
        item_id = ready.popleft()
        visited += 1
        for child in sorted(outgoing[item_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(known):
        cyclic = sorted(item_id for item_id, degree in indegree.items() if degree)
        raise ValueError(f"{context} contain a dependency cycle: {cyclic}.")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    return deepcopy(dict(value))


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{context} must be a list.")
    return list(value)


def _strings(value: Any, context: str) -> list[str]:
    result = [
        _string(item, f"{context}[{index}]")
        for index, item in enumerate(_sequence(value, context))
    ]
    _unique(result, context)
    return result


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _enum(value: Any, allowed: Collection[str], context: str) -> str:
    result = _string(value, context)
    if result not in allowed:
        raise ValueError(f"{context} must be one of {sorted(allowed)}.")
    return result


def _unique(values: Sequence[str], context: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{context} must be unique.")


def _keys(value: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} contains unsupported fields: {unknown}.")


def _schema(value: Mapping[str, Any], expected: str, context: str) -> None:
    if value.get("schema_version") != expected:
        raise ValueError(f"{context}.schema_version must be {expected!r}.")
