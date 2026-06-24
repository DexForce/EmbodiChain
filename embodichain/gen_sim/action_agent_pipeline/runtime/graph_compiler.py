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

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import extract_json_object

__all__ = [
    "compile_agent_graph_from_file",
    "compile_agent_graph_spec",
    "load_agent_graph_bundle",
]

_RECOVERY_KEYS = {
    "recovery_graph",
    "recovery_spec",
    "recovery_bindings",
    "recovery_nodes",
    "recovery_edges",
    "recovery_branches",
    "recoveries",
}
_COMPILED_BUNDLE_KEYS = {"task_graph", "metadata"}
_EDGE_KEYS = {"id", "source", "target", "left_arm_action", "right_arm_action"}


def load_agent_graph_bundle(path: str | Path) -> dict[str, Any]:
    """Load a compiled graph JSON bundle from disk."""
    return extract_json_object(Path(path).read_text(encoding="utf-8"))


def compile_agent_graph_from_file(
    path: str | Path,
    *,
    graph_cls: type | None = None,
    action_module: Any = None,
) -> Any:
    """Compile a graph JSON bundle from disk into an executable graph."""
    bundle = load_agent_graph_bundle(path)
    if "task_graph" in bundle:
        unknown_bundle_keys = set(bundle) - _COMPILED_BUNDLE_KEYS
        if unknown_bundle_keys:
            raise ValueError(
                "Compiled graph artifact contains unsupported top-level fields: "
                f"{', '.join(sorted(unknown_bundle_keys))}."
            )
        task_graph = bundle["task_graph"]
    else:
        task_graph = bundle
    return compile_agent_graph_spec(
        task_graph,
        graph_cls=graph_cls,
        action_module=action_module,
    )


def compile_agent_graph_spec(
    task_graph: str | Mapping[str, Any],
    *,
    graph_cls: type | None = None,
    action_module: Any = None,
) -> Any:
    """Compile a nominal JSON graph into ``AgentTaskGraph``."""
    task_spec = extract_json_object(task_graph)
    _reject_recovery_keys(task_spec)
    _validate_task_spec(task_spec)
    graph_cls, action_module = _resolve_runtime(
        graph_cls=graph_cls,
        action_module=action_module,
    )

    graph = graph_cls(
        start=task_spec["start"],
        goal=task_spec["goal"],
        max_transitions=int(task_spec.get("max_transitions", 1000)),
    )

    for node in task_spec.get("nodes", []):
        graph.add_node(node["id"], node.get("semantic", ""))

    for edge in task_spec.get("edges", []):
        graph.add_edge(
            edge["id"],
            edge["source"],
            edge["target"],
            left_arm_action=_compile_action(edge.get("left_arm_action"), action_module),
            right_arm_action=_compile_action(
                edge.get("right_arm_action"), action_module
            ),
        )

    return graph


def _resolve_runtime(
    *,
    graph_cls: type | None,
    action_module: Any,
) -> tuple[type, Any]:
    if graph_cls is None:
        graph_cls = _resolve_attr(
            importlib.import_module(
                "embodichain.gen_sim.action_agent_pipeline.runtime.task_graph"
            ),
            "AgentTaskGraph",
        )
    if action_module is None:
        action_module = importlib.import_module(
            "embodichain.gen_sim.action_agent_pipeline.runtime.atom_actions"
        )
    return graph_cls, action_module


def _validate_task_spec(task_spec: Mapping[str, Any]) -> None:
    node_ids = set()
    for node in task_spec.get("nodes", []):
        node_id = node["id"]
        if node_id in node_ids:
            raise ValueError(f"Duplicate graph node id '{node_id}'.")
        node_ids.add(node_id)

    for required_node in (task_spec["start"], task_spec["goal"]):
        if required_node not in node_ids:
            raise ValueError(f"Graph node '{required_node}' is not defined.")

    edge_specs = list(task_spec.get("edges", []))
    edge_ids = set()
    for edge in edge_specs:
        unknown_edge_keys = set(edge) - _EDGE_KEYS
        if unknown_edge_keys:
            raise ValueError(
                f"Nominal edge '{edge.get('id', '<unknown>')}' contains unsupported "
                f"fields: {', '.join(sorted(unknown_edge_keys))}."
            )
        edge_id = edge["id"]
        if edge_id in edge_ids:
            raise ValueError(f"Duplicate graph edge id '{edge_id}'.")
        edge_ids.add(edge_id)
        if _is_empty_action_spec(edge.get("left_arm_action")) and _is_empty_action_spec(
            edge.get("right_arm_action")
        ):
            raise ValueError(f"Nominal edge '{edge_id}' must define an arm action.")

        for node_key in ("source", "target"):
            node_id = edge[node_key]
            if node_id not in node_ids:
                raise ValueError(
                    f"Edge '{edge_id}' references unknown {node_key} node '{node_id}'."
                )

    _validate_nominal_path(task_spec, edge_specs)


def _validate_nominal_path(
    task_spec: Mapping[str, Any],
    edge_specs: list[Mapping[str, Any]],
) -> None:
    outgoing_edges: dict[str, Mapping[str, Any]] = {}
    for edge in edge_specs:
        source = edge["source"]
        if source in outgoing_edges:
            raise ValueError(
                f"Nominal node '{source}' has multiple outgoing edges. "
                "The current graph executor expects one deterministic nominal path."
            )
        outgoing_edges[source] = edge

    current = task_spec["start"]
    goal = task_spec["goal"]
    visited_edges = set()
    visited_nodes = {current}

    while current != goal:
        edge = outgoing_edges.get(current)
        if edge is None:
            raise ValueError(
                f"Nominal graph has no start-to-goal path from node '{current}'."
            )
        edge_id = edge["id"]
        if edge_id in visited_edges:
            raise ValueError("Nominal graph contains a cycle.")

        visited_edges.add(edge_id)
        current = edge["target"]
        if current in visited_nodes and current != goal:
            raise ValueError("Nominal graph contains a cycle.")
        visited_nodes.add(current)

    all_edge_ids = {edge["id"] for edge in edge_specs}
    unused_edge_ids = all_edge_ids - visited_edges
    if unused_edge_ids:
        unused = ", ".join(sorted(unused_edge_ids))
        raise ValueError(
            f"Nominal graph contains edges outside the start-to-goal path: {unused}."
        )


def _compile_action(spec: Any, action_module: Any) -> Any:
    if _is_empty_action_spec(spec):
        return None
    if not isinstance(spec, Mapping):
        raise TypeError(f"Action spec must be a mapping or null, but got {type(spec)}.")
    if "fn" in spec:
        raise ValueError(
            "Legacy fn/kwargs action schema is not supported. Use atomic action "
            "class JSON spec with atomic_action_class, robot_name, control, cfg, "
            "and exactly one of target_object, target_pose, or target_qpos."
        )
    if "action" in spec:
        raise ValueError(
            "Legacy action schema is not supported. Use atomic_action_class with "
            "PickUpAction, MoveAction, or PlaceAction."
        )
    if spec.get("atomic_action_class") is None:
        raise ValueError(
            "Atomic action class schema requires atomic_action_class, robot_name, "
            "control, cfg, and exactly one of target_object, target_pose, or "
            "target_qpos."
        )

    normalized = action_module.normalize_atomic_action_spec(spec)
    spec_cls = getattr(action_module, "AtomicActionSpec", None)
    if spec_cls is None:
        return normalized
    return spec_cls.from_normalized(normalized)


def _is_empty_action_spec(spec: Any) -> bool:
    return spec is None or (
        isinstance(spec, str) and spec.strip().lower() in {"", "none", "null"}
    )


def _reject_recovery_keys(task_spec: Mapping[str, Any]) -> None:
    present = _RECOVERY_KEYS & set(task_spec)
    if present:
        raise ValueError(
            "Recovery graph fields are no longer supported: "
            f"{', '.join(sorted(present))}."
        )


def _resolve_attr(namespace: Any, name: str) -> Any:
    if isinstance(namespace, Mapping):
        return namespace[name]
    return getattr(namespace, name)
