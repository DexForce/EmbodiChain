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

"""Score, select, and conservatively fuse whole TaskGroups."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapabilityRegistry,
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.domain import (
    validate_seed_graph,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.runtime.motion_policy import (
    resolve_motion_policy,
)

from .linker import link_seed_graph, validate_persisted_contracts

__all__ = [
    "CandidateEvaluation",
    "evaluate_candidate",
    "fuse_seed_graphs",
    "select_seed_graph",
]


@dataclass(frozen=True)
class CandidateEvaluation:
    """Auditable static candidate score before any physical execution."""

    route: str
    valid: bool
    executable: bool
    coverage: float
    visual_confidence: float
    estimated_cost: float
    score: float
    errors: tuple[str, ...] = ()


def evaluate_candidate(
    graph: Mapping[str, Any],
    task_spec: Mapping[str, Any],
    *,
    known_objects: Collection[str],
    visual_confidence: float = 1.0,
    exact_template_match: bool = False,
    registry: AtomicCapabilityRegistry | None = None,
    robot_profile: str = "dual_ur10",
) -> CandidateEvaluation:
    """Apply schema, capabilities, object identity, coverage, and cost scoring."""
    task = validate_task_spec(task_spec)
    capabilities = registry or build_atomic_capability_registry()
    errors = []
    try:
        seed = validate_seed_graph(
            graph,
            known_objects=known_objects,
            known_actions=capabilities.names(),
        )
        if seed["capability_catalog_hash"] != capabilities.catalog_hash():
            raise ValueError("SeedGraph capability catalog does not match runtime.")
        validate_persisted_contracts(seed, capabilities)
        for node in seed["nodes"]:
            capabilities.validate_binding(node)
            if capabilities.get(str(node["atomic_action"])).runtime_available:
                resolve_motion_policy(
                    robot_profile,
                    node["atomic_action"],
                    node["motion_policy"],
                )
    except (TypeError, ValueError) as error:
        return CandidateEvaluation(
            route=str(graph.get("planner_route", "unknown")),
            valid=False,
            executable=False,
            coverage=0.0,
            visual_confidence=0.0,
            estimated_cost=float("inf"),
            score=float("-inf"),
            errors=(str(error),),
        )

    required = {str(item["id"]) for item in task["task_instances"]}
    provided = {str(group["id"]) for group in seed["task_groups"]}
    if task["level"] == "L4":
        coverage = 1.0 if provided and seed["success"] else 0.0
        unexpected = set()
        mismatched_types = {}
    else:
        coverage = len(required & provided) / max(len(required), 1)
        unexpected = provided - required
        if unexpected:
            errors.append(f"unexpected task groups: {sorted(unexpected)}")
        expected_types = {
            str(item["id"]): str(item["task_type"]) for item in task["task_instances"]
        }
        mismatched_types = {
            str(group["id"]): str(group["task_type"])
            for group in seed["task_groups"]
            if group["id"] in expected_types
            and group["task_type"] != expected_types[group["id"]]
        }
        if mismatched_types:
            errors.append(f"task group type mismatches: {mismatched_types}")
    unavailable = sorted(
        {
            str(node["atomic_action"])
            for node in seed["nodes"]
            if not capabilities.get(str(node["atomic_action"])).runtime_available
        }
    )
    executable = not unavailable
    if unavailable:
        errors.append(f"planning-only actions: {unavailable}")
    confidence = min(max(float(visual_confidence), 0.0), 1.0)
    estimated_cost = float(len(seed["nodes"]))
    score = coverage * 100.0 - estimated_cost
    route = str(seed["planner_route"])
    if exact_template_match and route == "offline":
        score += 15.0
    if task["level"] == "L4" and route == "online":
        score += 20.0 * confidence
    if not executable:
        score -= 30.0
    return CandidateEvaluation(
        route=route,
        valid=not unexpected and not mismatched_types and coverage == 1.0,
        executable=executable,
        coverage=coverage,
        visual_confidence=confidence,
        estimated_cost=estimated_cost,
        score=score,
        errors=tuple(errors),
    )


def select_seed_graph(
    offline: Mapping[str, Any],
    online: Mapping[str, Any],
    task_spec: Mapping[str, Any],
    *,
    known_objects: Collection[str],
    visual_confidence: float = 1.0,
    exact_template_match: bool = False,
    registry: AtomicCapabilityRegistry | None = None,
    robot_profile: str = "dual_ur10",
) -> tuple[dict[str, Any], dict[str, CandidateEvaluation]]:
    """Choose one complete candidate; ties prefer mature offline templates."""
    evaluations = {
        "offline": evaluate_candidate(
            offline,
            task_spec,
            known_objects=known_objects,
            visual_confidence=1.0,
            exact_template_match=exact_template_match,
            registry=registry,
            robot_profile=robot_profile,
        ),
        "online": evaluate_candidate(
            online,
            task_spec,
            known_objects=known_objects,
            visual_confidence=visual_confidence,
            registry=registry,
            robot_profile=robot_profile,
        ),
    }
    valid = [item for item in evaluations.items() if item[1].valid]
    if not valid:
        messages = {name: evaluation.errors for name, evaluation in evaluations.items()}
        raise ValueError(f"Neither SeedGraph candidate is valid: {messages}.")
    valid.sort(
        key=lambda item: (
            item[1].score,
            item[0] == "offline",
        ),
        reverse=True,
    )
    selected = deepcopy(dict(offline if valid[0][0] == "offline" else online))
    selected["planner_route"] = "selected"
    selected.setdefault("metadata", {})["selected_from"] = valid[0][0]
    return selected, evaluations


def fuse_seed_graphs(
    offline: Mapping[str, Any],
    online: Mapping[str, Any],
    group_routes: Mapping[str, str],
    *,
    registry: AtomicCapabilityRegistry | None = None,
) -> dict[str, Any]:
    """Fuse candidates only at complete TaskGroup boundaries."""
    capabilities = registry or build_atomic_capability_registry()
    if offline.get("task_id") != online.get("task_id"):
        raise ValueError("Cannot fuse graphs for different tasks.")
    for field in ("instruction", "level", "reasoning_type", "capability_catalog_hash"):
        if offline.get(field) != online.get(field):
            raise ValueError(f"Cannot fuse graphs with different {field} values.")
    by_route = {
        "offline": validate_seed_graph(offline, known_actions=capabilities.names()),
        "online": validate_seed_graph(online, known_actions=capabilities.names()),
    }
    for graph in by_route.values():
        validate_persisted_contracts(graph, capabilities)
    groups_by_route = {
        route: {str(group["id"]): group for group in graph["task_groups"]}
        for route, graph in by_route.items()
    }
    expected = set(groups_by_route["offline"])
    if set(groups_by_route["online"]) != expected or set(group_routes) != expected:
        raise ValueError(
            "Fusion requires the same complete TaskGroup set in both graphs."
        )
    if set(group_routes.values()) - {"offline", "online"}:
        raise ValueError("Every fused TaskGroup route must be offline or online.")

    selected_groups = {
        group_id: deepcopy(groups_by_route[route][group_id])
        for group_id, route in group_routes.items()
    }
    _reject_state_conflicts(selected_groups)
    source_nodes = {
        route: {str(node["id"]): node for node in graph["nodes"]}
        for route, graph in by_route.items()
    }
    selected_nodes_by_group: dict[str, list[dict[str, Any]]] = {}
    id_map: dict[tuple[str, str], str] = {}
    for group_id, route in group_routes.items():
        group = selected_groups[group_id]
        group.pop("contract", None)
        selected_nodes_by_group[group_id] = []
        for node_id in group["node_ids"]:
            node = deepcopy(source_nodes[route][node_id])
            fused_id = f"{route}_{node_id}"
            id_map[(route, node_id)] = fused_id
            node["id"] = fused_id
            selected_nodes_by_group[group_id].append(node)

    terminals = {}
    for group_id, route in group_routes.items():
        original_ids = set(selected_groups[group_id]["node_ids"])
        referenced = {
            dependency
            for node_id in original_ids
            for dependency in source_nodes[route][node_id]["depends_on"]
            if dependency in original_ids
        }
        terminals[group_id] = [
            id_map[(route, node_id)]
            for node_id in selected_groups[group_id]["node_ids"]
            if node_id not in referenced
        ]
    nodes = []
    groups = []
    for group_id in _topological_groups(selected_groups):
        route = group_routes[group_id]
        group = selected_groups[group_id]
        own_original_ids = set(group["node_ids"])
        group_nodes = selected_nodes_by_group[group_id]
        for node in group_nodes:
            original_id = node["id"][len(route) + 1 :]
            original = source_nodes[route][original_id]
            internal = [
                id_map[(route, dependency)]
                for dependency in original["depends_on"]
                if dependency in own_original_ids
            ]
            external = [
                terminal
                for parent in group["depends_on"]
                for terminal in terminals[parent]
            ]
            node["depends_on"] = list(dict.fromkeys([*internal, *external]))
            nodes.append(node)
        group["node_ids"] = [node["id"] for node in group_nodes]
        groups.append(group)

    fused = deepcopy(by_route["offline"])
    fused["planner_route"] = "fused"
    fused["nodes"] = nodes
    fused["task_groups"] = groups
    fused["success"] = {"op": "all", "terms": [group["success"] for group in groups]}
    fused["metadata"] = {
        "fusion_routes": dict(sorted(group_routes.items())),
        "fusion_boundary": "task_group",
    }
    return link_seed_graph(
        fused,
        registry=capabilities,
        task_order=[str(group["id"]) for group in groups],
    )


def _reject_state_conflicts(groups: Mapping[str, Mapping[str, Any]]) -> None:
    by_object: dict[str, list[str]] = defaultdict(list)
    for group_id, group in groups.items():
        by_object[str(group["object_uid"])].append(group_id)
    dependencies = {
        group_id: set(group["depends_on"]) for group_id, group in groups.items()
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

    for object_uid, group_ids in by_object.items():
        for index, first in enumerate(group_ids):
            for second in group_ids[index + 1 :]:
                if not reaches(first, second) and not reaches(second, first):
                    raise ValueError(
                        f"Fusion has unordered state changes for object {object_uid!r}."
                    )


def _topological_groups(groups: Mapping[str, Mapping[str, Any]]) -> list[str]:
    outgoing = {group_id: [] for group_id in groups}
    indegree = {group_id: 0 for group_id in groups}
    for group_id, group in groups.items():
        for parent in group["depends_on"]:
            outgoing[parent].append(group_id)
            indegree[group_id] += 1
    ready = deque(
        sorted(group_id for group_id, degree in indegree.items() if degree == 0)
    )
    result = []
    while ready:
        group_id = ready.popleft()
        result.append(group_id)
        for child in sorted(outgoing[group_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return result
