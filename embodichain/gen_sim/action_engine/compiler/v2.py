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

"""Bridge mature v1 task recipes to the direct AtomicAction SeedGraph v3."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
import re
from typing import Any

from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapabilityRegistry,
    build_atomic_capability_registry,
    capability_precondition,
)
from embodichain.gen_sim.action_engine.domain import (
    EXECUTION_PROGRAM_SCHEMA,
    MOTION_POLICY_VERSION,
    validate_execution_program,
    validate_seed_graph,
)
from embodichain.gen_sim.action_engine.protocol import SEED_GRAPH_SCHEMA
from embodichain.gen_sim.action_engine.planning.linker import (
    link_seed_graph,
    validate_persisted_contracts,
)

__all__ = [
    "compile_task_agent_v2",
    "execution_program_to_seed_graph",
    "seed_graph_to_execution_program",
]

_UNSAFE_ID_RE = re.compile(r"[^0-9a-z]+")
_OPERATOR_TASK_TYPES = {
    "arrange_line": "E1",
    "build_stack": "E1",
    "coordinated_place": "E5",
    "coordinated_transport": "E5",
    "hold_hover": "E1",
    "orient_object": "E2",
    "place_in_line": "E1",
    "place_relative": "E1",
    "press": "E9",
}


def compile_task_agent_v2(
    program: Mapping[str, Any],
    *,
    known_objects: Collection[str] | None = None,
    registry: AtomicCapabilityRegistry | None = None,
) -> dict[str, Any]:
    """Compile a mature semantic recipe directly to the persisted v3 graph."""
    from .core import compile_task_agent

    legacy = compile_task_agent(program, known_objects=known_objects)
    return execution_program_to_seed_graph(
        legacy,
        known_objects=known_objects,
        registry=registry,
    )


def execution_program_to_seed_graph(
    program: Mapping[str, Any],
    *,
    planner_route: str = "offline",
    known_objects: Collection[str] | None = None,
    registry: AtomicCapabilityRegistry | None = None,
) -> dict[str, Any]:
    """Convert a mature v1 result without changing its AtomicAction topology."""
    legacy = validate_execution_program(program)
    capabilities = registry or build_atomic_capability_registry()
    steps = {str(step["id"]): step for step in legacy["semantic_steps"]}
    node_ids_by_edge: dict[str, list[str]] = {}
    nodes: list[dict[str, Any]] = []

    for edge in legacy["edges"]:
        edge_id = str(edge["id"])
        step = steps[str(edge["semantic_step_id"])]
        task_type = _task_type(str(step["operator"]))
        dependencies = [
            node_id
            for dependency in edge.get("depends_on", [])
            for node_id in node_ids_by_edge[str(dependency)]
        ]
        edge_nodes: list[str] = []
        actions = list(edge["actions"])
        for action_index, action in enumerate(actions):
            action_name = str(action["atomic_action_class"])
            descriptor_view = {
                "atomic_action": action_name,
                "control": action.get("control", "arm"),
                "target_binding": action["target_binding"],
            }
            capabilities.validate_binding(descriptor_view)
            capability = capabilities.get(action_name)
            node_id = _node_id(edge_id, action_name, action_index, len(actions))
            postcondition = (
                deepcopy(step["postcondition"])
                if edge_id == step["edge_ids"][-1]
                else {}
            )
            node = {
                "id": node_id,
                "atomic_action": action_name,
                "object_uid": str(step["object"]),
                "actor": _v2_actor(action["actor"]),
                "control": str(action.get("control", "arm")),
                "target_binding": deepcopy(dict(action["target_binding"])),
                "depends_on": list(dict.fromkeys(dependencies)),
                "task_instance_id": str(step["id"]),
                "task_type": task_type,
                "role": _node_role(action_name, action["target_binding"]),
                "precondition": capability_precondition(
                    capability,
                    object_uid=str(step["object"]),
                    actor=_v2_actor(action["actor"]),
                    target_binding=action["target_binding"],
                ),
                "postcondition": postcondition,
                "motion_policy": deepcopy(dict(action["motion_policy"])),
            }
            if len(actions) > 1:
                node["sync_group"] = edge_id
            nodes.append(node)
            edge_nodes.append(node_id)
        node_ids_by_edge[edge_id] = edge_nodes

    groups = []
    for step in legacy["semantic_steps"]:
        group_node_ids = [
            node_id
            for edge_id in step["edge_ids"]
            for node_id in node_ids_by_edge[str(edge_id)]
        ]
        groups.append(
            {
                "id": str(step["id"]),
                "task_type": _task_type(str(step["operator"])),
                "role": "primary",
                "operator": str(step["operator"]),
                "object_uid": str(step["object"]),
                "actor": _v2_actor(step["actor"]),
                "goal": deepcopy(dict(step.get("goal", {}))),
                "depends_on": list(step.get("depends_on", [])),
                "parent_task_instance_id": str(step.get("parent_step_id", step["id"])),
                "node_ids": group_node_ids,
                "success": deepcopy(dict(step["postcondition"])),
            }
        )

    level = _level(groups)
    graph = {
        "schema_version": SEED_GRAPH_SCHEMA,
        "task_id": str(legacy["task"]),
        "instruction": str(legacy["goal_description"]),
        "level": level,
        "reasoning_type": "none",
        "planner_route": planner_route,
        "nodes": nodes,
        "task_groups": groups,
        "success": {
            "op": "all",
            "terms": [deepcopy(group["success"]) for group in groups],
        },
        "capability_catalog_hash": capabilities.catalog_hash(),
        "metadata": {
            "source_schema": EXECUTION_PROGRAM_SCHEMA,
            "legacy_allocation_groups": deepcopy(legacy.get("allocation_groups", [])),
            "planning_latency_seconds": 0.0,
            "vlm_call_count": 0,
        },
    }
    return link_seed_graph(
        graph,
        registry=capabilities,
        task_order=[str(step["id"]) for step in legacy["semantic_steps"]],
        known_objects=known_objects,
    )


def seed_graph_to_execution_program(
    graph: Mapping[str, Any],
    *,
    known_objects: Collection[str] | None = None,
    registry: AtomicCapabilityRegistry | None = None,
    require_executable: bool = True,
) -> dict[str, Any]:
    """Materialize the v3 DAG as the existing in-memory runtime view."""
    capabilities = registry or build_atomic_capability_registry()
    seed = validate_seed_graph(
        graph,
        known_objects=known_objects,
        known_actions=capabilities.names(),
        executable_actions=capabilities.executable_names(),
        require_executable=require_executable,
    )
    if seed["capability_catalog_hash"] != capabilities.catalog_hash():
        raise ValueError(
            "SeedGraph capability_catalog_hash does not match the runtime catalog."
        )
    validate_persisted_contracts(seed, capabilities)
    for node in seed["nodes"]:
        capabilities.validate_binding(node)

    node_by_id = {str(node["id"]): node for node in seed["nodes"]}
    unit_by_node, units = _execution_units(seed["nodes"])
    ordered_units = _topological_units(units)
    edge_id_by_unit = {unit_id: f"edge_{_slug(unit_id)}" for unit_id in ordered_units}
    target_by_unit = {
        unit_id: f"state_{index + 1:04d}_{_slug(unit_id)}"
        for index, unit_id in enumerate(ordered_units)
    }
    start = "state_start"
    edges = []
    graph_nodes = [{"id": start, "semantic": "Initial live simulator state"}]
    for unit_id in ordered_units:
        unit = units[unit_id]
        dependencies = sorted(unit["depends_on"])
        source = start if not dependencies else target_by_unit[dependencies[0]]
        target = target_by_unit[unit_id]
        graph_nodes.append(
            {
                "id": target,
                "semantic": f"Completed AtomicAction unit {unit_id}",
            }
        )
        unit_nodes = [node_by_id[node_id] for node_id in unit["node_ids"]]
        edges.append(
            {
                "id": edge_id_by_unit[unit_id],
                "source": source,
                "target": target,
                "semantic_step_id": str(unit_nodes[0]["task_instance_id"]),
                "actions": [
                    {
                        "atomic_action_class": node["atomic_action"],
                        "actor": deepcopy(node["actor"]),
                        "control": node["control"],
                        "target_binding": deepcopy(node["target_binding"]),
                        "motion_policy": node["motion_policy"],
                        "seed_node_id": node["id"],
                        "failure_policy": node["contract"]["failure_policy"],
                    }
                    for node in unit_nodes
                ],
                "depends_on": [edge_id_by_unit[item] for item in dependencies],
                "resources": sorted(
                    {
                        str(claim["resource"])
                        for node in unit_nodes
                        for claim in node["contract"]["claims"]
                    }
                ),
            }
        )

    group_by_id = {str(group["id"]): group for group in seed["task_groups"]}
    semantic_steps = []
    for group_id in _topological_groups(seed["task_groups"]):
        group = group_by_id[group_id]
        group_units = [
            unit_id
            for unit_id in ordered_units
            if any(
                node_by_id[node_id]["task_instance_id"] == group_id
                for node_id in units[unit_id]["node_ids"]
            )
        ]
        semantic_steps.append(
            {
                "id": group_id,
                "parent_step_id": str(group.get("parent_task_instance_id", group_id)),
                "operator": str(group["operator"]),
                "object": str(group["object_uid"]),
                "actor": deepcopy(group["actor"]),
                "goal": deepcopy(group["goal"]),
                "depends_on": list(group["depends_on"]),
                "postcondition": deepcopy(group["success"]),
                "edge_ids": [edge_id_by_unit[unit_id] for unit_id in group_units],
            }
        )

    metadata = seed.get("metadata", {})
    allocation_groups = (
        deepcopy(metadata.get("legacy_allocation_groups", []))
        if isinstance(metadata, Mapping)
        else []
    )
    program = {
        "schema_version": EXECUTION_PROGRAM_SCHEMA,
        "task": seed["task_id"],
        "goal_description": seed["instruction"],
        "start": start,
        "goal": target_by_unit[ordered_units[-1]],
        "nodes": graph_nodes,
        "edges": edges,
        "semantic_steps": semantic_steps,
        "allocation_groups": allocation_groups,
        "motion_policy_version": MOTION_POLICY_VERSION,
    }
    return validate_execution_program(program)


def _execution_units(
    nodes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    unit_by_node = {
        str(node["id"]): str(node.get("sync_group", node["id"])) for node in nodes
    }
    units: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = str(node["id"])
        unit_id = unit_by_node[node_id]
        unit = units.setdefault(unit_id, {"node_ids": [], "depends_on": set()})
        unit["node_ids"].append(node_id)
        for dependency in node["depends_on"]:
            dependency_unit = unit_by_node[str(dependency)]
            if dependency_unit == unit_id:
                raise ValueError(
                    f"Synchronized unit {unit_id!r} has an internal dependency."
                )
            unit["depends_on"].add(dependency_unit)
    for unit_id, unit in units.items():
        groups = {
            str(
                next(node for node in nodes if node["id"] == node_id)[
                    "task_instance_id"
                ]
            )
            for node_id in unit["node_ids"]
        }
        if len(groups) != 1:
            raise ValueError(f"Synchronized unit {unit_id!r} crosses task groups.")
    return unit_by_node, units


def _topological_units(units: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return _topological_ids(
        {unit_id: list(unit["depends_on"]) for unit_id, unit in units.items()}
    )


def _topological_groups(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    return _topological_ids(
        {str(group["id"]): list(group["depends_on"]) for group in groups}
    )


def _topological_ids(dependencies: Mapping[str, Sequence[str]]) -> list[str]:
    outgoing = {item_id: [] for item_id in dependencies}
    indegree = {item_id: 0 for item_id in dependencies}
    for item_id, parents in dependencies.items():
        for parent in parents:
            outgoing[parent].append(item_id)
            indegree[item_id] += 1
    ready = deque(
        sorted(item_id for item_id, degree in indegree.items() if degree == 0)
    )
    ordered = []
    while ready:
        item_id = ready.popleft()
        ordered.append(item_id)
        for child in sorted(outgoing[item_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(ordered) != len(dependencies):
        raise ValueError("Graph contains a dependency cycle.")
    return ordered


def _v2_actor(value: Mapping[str, Any]) -> dict[str, Any]:
    actor = deepcopy(dict(value))
    actor.pop("allocation_group", None)
    if actor.get("mode") == "required" and actor.get("arm") in {"left", "right"}:
        actor["arm"] = f"{actor['arm']}_arm"
    return actor


def _task_type(operator: str) -> str:
    try:
        return _OPERATOR_TASK_TYPES[operator]
    except KeyError as exc:
        raise ValueError(
            f"Semantic operator {operator!r} has no registered task contract."
        ) from exc


def _level(groups: Sequence[Mapping[str, Any]]) -> str:
    types = {str(group["task_type"]) for group in groups}
    if len(groups) == 1:
        return "L1"
    return "L2" if len(types) == 1 else "L3"


def _node_role(action_name: str, binding: Mapping[str, Any]) -> str:
    if (
        action_name == "MoveJoints" and binding.get("source") == "initial"
    ) or binding.get("kind") == "policy_pose":
        return "cleanup"
    return "primary"


def _node_id(edge_id: str, action: str, index: int, count: int) -> str:
    base = f"{_slug(edge_id)}_{_slug(action)}"
    return base if count == 1 else f"{base}_{index + 1}"


def _slug(value: str) -> str:
    return _UNSAFE_ID_RE.sub("_", value.lower()).strip("_") or "node"
