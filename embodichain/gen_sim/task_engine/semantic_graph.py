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

"""Immutable, execution-free task graphs built from canonical Semantic Calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import re
from typing import Any, Final, TypeAlias

from embodichain.lab.task_program.language import decode_task_program

from .contracts import canonical_hash

__all__ = [
    "SEMANTIC_TASK_GRAPH_SCHEMA",
    "SemanticTaskGraph",
    "semantic_task_graph_hash",
    "validate_semantic_task_graph",
]

SEMANTIC_TASK_GRAPH_SCHEMA: Final = "semantic_task_graph/v1"
SemanticTaskGraph: TypeAlias = dict[str, Any]

_GRAPH_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "instruction",
        "planner_route",
        "integration_fingerprint",
        "targets",
        "nodes",
        "task_groups",
        "success",
    }
)
_NODE_KEYS = frozenset(
    {
        "id",
        "call",
        "depends_on",
        "task_instance_id",
        "task_type",
        "role",
    }
)
_GROUP_KEYS = frozenset({"id", "task_type", "node_ids", "depends_on", "success"})
_FORBIDDEN_KEYS = frozenset(
    {
        "action",
        "action_invocation",
        "action_options",
        "atomic_action",
        "atomic_goal",
        "command",
        "command_frame",
        "control_part",
        "controller",
        "eef_pose",
        "goal",
        "grasp_pose",
        "held_object",
        "motion_policy",
        "planner_backend",
        "qpos",
        "resource_claim",
        "robot_part",
        "solver",
        "trajectory",
        "waypoint",
    }
)
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


def validate_semantic_task_graph(value: Mapping[str, Any]) -> SemanticTaskGraph:
    """Validate and detach one provider-free semantic task graph.

    Every node payload is decoded through the Task Program language decoder.
    Consequently Task Engine cannot silently grow a second Semantic Call JSON
    dialect while planning remains independent from grounded robot actions.

    Args:
        value: JSON-compatible task graph mapping.

    Returns:
        A detached, normalized graph dictionary.

    Raises:
        TypeError: If the graph is not an exact JSON mapping.
        ValueError: If topology, schema, or a Semantic Call is invalid.
    """
    graph = _json_snapshot(value)
    _exact_keys(graph, _GRAPH_KEYS, "SemanticTaskGraph")
    if graph["schema_version"] != SEMANTIC_TASK_GRAPH_SCHEMA:
        raise ValueError(
            "SemanticTaskGraph.schema_version must be "
            f"{SEMANTIC_TASK_GRAPH_SCHEMA!r}."
        )
    for field in ("task_id", "instruction", "planner_route"):
        graph[field] = _nonempty(graph[field], f"SemanticTaskGraph.{field}")
    fingerprint = _nonempty(
        graph["integration_fingerprint"],
        "SemanticTaskGraph.integration_fingerprint",
    )
    if _FINGERPRINT.fullmatch(fingerprint) is None:
        raise ValueError(
            "SemanticTaskGraph.integration_fingerprint must be a lowercase "
            "SHA-256 digest."
        )

    targets = graph["targets"]
    if type(targets) is not dict:
        raise TypeError("SemanticTaskGraph.targets must be an exact mapping.")
    _reject_execution_data(targets, path="SemanticTaskGraph.targets")

    raw_nodes = graph["nodes"]
    if type(raw_nodes) is not list or not raw_nodes:
        raise ValueError("SemanticTaskGraph.nodes must be a non-empty list.")
    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    for index, raw in enumerate(raw_nodes):
        context = f"SemanticTaskGraph.nodes[{index}]"
        if type(raw) is not dict:
            raise TypeError(f"{context} must be an exact mapping.")
        _exact_keys(raw, _NODE_KEYS, context)
        node = deepcopy(raw)
        for field in ("id", "task_instance_id", "task_type", "role"):
            node[field] = _nonempty(node[field], f"{context}.{field}")
        if node["id"] in node_ids:
            raise ValueError(f"Duplicate SemanticTaskGraph node ID {node['id']!r}.")
        node_ids.add(node["id"])
        node["depends_on"] = _string_list(node["depends_on"], f"{context}.depends_on")
        if type(node["call"]) is not dict:
            raise TypeError(f"{context}.call must be an exact mapping.")
        _reject_execution_data(node["call"], path=f"{context}.call")
        nodes.append(node)

    _validate_dependencies(nodes, node_ids, owner="SemanticTaskGraph.nodes")
    _decode_calls(targets, [node["call"] for node in nodes])

    raw_groups = graph["task_groups"]
    if type(raw_groups) is not list or not raw_groups:
        raise ValueError("SemanticTaskGraph.task_groups must be a non-empty list.")
    groups: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    assigned_nodes: set[str] = set()
    for index, raw in enumerate(raw_groups):
        context = f"SemanticTaskGraph.task_groups[{index}]"
        if type(raw) is not dict:
            raise TypeError(f"{context} must be an exact mapping.")
        _exact_keys(raw, _GROUP_KEYS, context)
        group = deepcopy(raw)
        for field in ("id", "task_type"):
            group[field] = _nonempty(group[field], f"{context}.{field}")
        if group["id"] in group_ids:
            raise ValueError(f"Duplicate TaskGroup ID {group['id']!r}.")
        group_ids.add(group["id"])
        group["node_ids"] = _string_list(group["node_ids"], f"{context}.node_ids")
        if not group["node_ids"]:
            raise ValueError(f"{context}.node_ids must not be empty.")
        unknown = sorted(set(group["node_ids"]) - node_ids)
        if unknown:
            raise ValueError(f"{context} references unknown node IDs {unknown}.")
        overlap = assigned_nodes.intersection(group["node_ids"])
        if overlap:
            raise ValueError(
                f"SemanticTaskGraph nodes belong to multiple groups: {sorted(overlap)}."
            )
        assigned_nodes.update(group["node_ids"])
        group["depends_on"] = _string_list(group["depends_on"], f"{context}.depends_on")
        if type(group["success"]) is not dict:
            raise TypeError(f"{context}.success must be an exact mapping.")
        _reject_execution_data(group["success"], path=f"{context}.success")
        groups.append(group)

    if assigned_nodes != node_ids:
        raise ValueError(
            "Every SemanticTaskGraph node must belong to exactly one TaskGroup; "
            f"unassigned={sorted(node_ids - assigned_nodes)}."
        )
    _validate_dependencies(groups, group_ids, owner="SemanticTaskGraph.task_groups")
    node_group = {
        node_id: group["id"] for group in groups for node_id in group["node_ids"]
    }
    for node in nodes:
        if node_group[node["id"]] != node["task_instance_id"]:
            raise ValueError(
                f"Node {node['id']!r} task_instance_id does not match TaskGroup membership."
            )
    if type(graph["success"]) is not dict:
        raise TypeError("SemanticTaskGraph.success must be an exact mapping.")
    _reject_execution_data(graph["success"], path="SemanticTaskGraph.success")

    graph["nodes"] = nodes
    graph["task_groups"] = groups
    return graph


def semantic_task_graph_hash(value: Mapping[str, Any]) -> str:
    """Return the deterministic content hash of a valid semantic task graph.

    Args:
        value: JSON-compatible semantic task graph.

    Returns:
        Lowercase SHA-256 content digest.

    Raises:
        TypeError: If the graph is not an exact JSON mapping.
        ValueError: If graph topology, schema, or a Semantic Call is invalid.
    """
    return canonical_hash(validate_semantic_task_graph(value))


def _decode_calls(
    targets: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]
) -> None:
    items = [{"kind": "invoke", "call": deepcopy(dict(call))} for call in calls]
    decode_task_program(
        {
            "program_id": "semantic_task_graph_validation",
            "integration": {
                "robot_profile": "provider_free_profile",
                "scene_registry": "provider_free_scene",
                "runtime_preset": "provider_free_runtime",
            },
            "targets": deepcopy(dict(targets)),
            "program": {"kind": "sequence", "items": items},
        }
    )


def _validate_dependencies(
    items: Sequence[Mapping[str, Any]],
    known_ids: set[str],
    *,
    owner: str,
) -> None:
    positions = {str(item["id"]): index for index, item in enumerate(items)}
    for item in items:
        item_id = str(item["id"])
        dependencies = list(item["depends_on"])
        unknown = sorted(set(dependencies) - known_ids)
        if unknown:
            raise ValueError(f"{owner} {item_id!r} has unknown dependencies {unknown}.")
        if item_id in dependencies:
            raise ValueError(f"{owner} {item_id!r} cannot depend on itself.")
        later = [
            dependency
            for dependency in dependencies
            if positions[dependency] >= positions[item_id]
        ]
        if later:
            raise ValueError(
                f"{owner} must be topologically ordered; {item_id!r} depends on {later}."
            )


def _reject_execution_data(value: Any, *, path: str) -> None:
    if type(value) is dict:
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError(
                    f"{path}.{key} is grounded execution data and is forbidden."
                )
            _reject_execution_data(child, path=f"{path}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _reject_execution_data(child, path=f"{path}[{index}]")


def _json_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("SemanticTaskGraph must be an exact mapping.")
    try:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "SemanticTaskGraph must contain only finite JSON values."
        ) from exc
    assert type(decoded) is dict
    return decoded


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _string_list(value: Any, context: str) -> list[str]:
    if type(value) is not list:
        raise TypeError(f"{context} must be a list.")
    result = [
        _nonempty(item, f"{context}[{index}]") for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise ValueError(f"{context} must not contain duplicates.")
    return result
