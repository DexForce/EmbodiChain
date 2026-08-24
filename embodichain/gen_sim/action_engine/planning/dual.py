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

"""Parallel offline/online candidate planning with isolated task views."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapabilityRegistry,
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.domain import (
    public_task_spec,
    seed_graph_hash,
    validate_seed_graph,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.runtime.motion_policy import (
    resolve_motion_policy,
)

from .linker import validate_persisted_contracts

__all__ = ["CandidatePair", "plan_candidates_parallel"]

CandidatePlanner = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class CandidatePair:
    """Two independently planned graphs and branch-local planning metrics."""

    offline: dict[str, Any]
    online: dict[str, Any]
    planning_metrics: dict[str, dict[str, Any]]


def plan_candidates_parallel(
    task_spec: Mapping[str, Any],
    *,
    offline_planner: CandidatePlanner,
    online_planner: CandidatePlanner,
    known_objects: set[str] | None = None,
    robot_profile: str = "dual_ur10",
    registry: AtomicCapabilityRegistry | None = None,
    require_executable: bool = False,
) -> CandidatePair:
    """Plan both routes concurrently while hiding the oracle from online.

    Both returned graphs are validated against the same capability catalog and
    motion-policy table before the pair is published.  ``require_executable``
    is intentionally opt-in here: product planning may retain planning-only
    candidates for inspection, while strict A/B execution enables the flag in
    its final preflight.
    """
    task = validate_task_spec(task_spec)
    online_view = public_task_spec(task)
    _reject_private_or_live_fields(online_view, "PublicTaskSpec")
    capabilities = registry or build_atomic_capability_registry()

    def invoke(route: str) -> tuple[dict[str, Any], float]:
        planner = offline_planner if route == "offline" else online_planner
        # A planner is user/LLM supplied code.  Give each route a detached
        # copy so accidental mutation cannot change the other route's input or
        # reintroduce private oracle fields after validation.
        planner_input = deepcopy(task if route == "offline" else online_view)
        started = perf_counter()
        try:
            result = planner(task_spec=planner_input)
        except Exception as exc:
            raise RuntimeError(f"{route} planner failed: {exc}") from exc
        elapsed = perf_counter() - started
        _reject_private_or_live_fields(result, f"{route} SeedGraph")
        graph = validate_seed_graph(
            result,
            known_objects=known_objects,
            known_actions=capabilities.names(),
            executable_actions=capabilities.executable_names(),
            require_executable=require_executable,
        )
        if graph["planner_route"] != route:
            raise ValueError(
                f"{route} planner returned route {graph['planner_route']!r}."
            )
        if graph["task_id"] != task["task_id"]:
            raise ValueError(f"{route} planner returned a graph for another task.")
        if graph["level"] != task["level"]:
            raise ValueError(f"{route} planner returned a graph for another level.")
        if graph["reasoning_type"] != task["reasoning_type"]:
            raise ValueError(
                f"{route} planner returned a graph with incompatible reasoning_type."
            )
        if graph["capability_catalog_hash"] != capabilities.catalog_hash():
            raise ValueError(
                f"{route} SeedGraph capability catalog does not match runtime."
            )
        validate_persisted_contracts(graph, capabilities)
        _validate_task_group_coverage(task, graph, route=route)
        for node in graph["nodes"]:
            capabilities.validate_binding(node)
            if capabilities.get(str(node["atomic_action"])).runtime_available:
                resolve_motion_policy(
                    robot_profile,
                    node["atomic_action"],
                    node["motion_policy"],
                )
        return graph, elapsed

    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="action-engine-plan"
    ) as pool:
        futures = {route: pool.submit(invoke, route) for route in ("offline", "online")}
        results: dict[str, tuple[dict[str, Any], float]] = {}
        for route, future in futures.items():
            try:
                results[route] = future.result()
            except Exception as exc:
                # Do not expose a bare Future exception; callers need to know
                # which route invalidated the pair before any environment is
                # allowed to move.
                for other_route, other in futures.items():
                    if other_route != route:
                        other.cancel()
                raise RuntimeError(
                    f"A/B {route} planning/preflight failed: {exc}"
                ) from exc

    metrics = {
        route: {
            "planning_seconds": elapsed,
            "vlm_call_count": int(graph.get("metadata", {}).get("vlm_call_count", 0)),
            "seed_graph_hash": seed_graph_hash(graph),
            "node_count": len(graph["nodes"]),
            "task_group_count": len(graph["task_groups"]),
        }
        for route, (graph, elapsed) in results.items()
    }
    return CandidatePair(
        offline=results["offline"][0],
        online=results["online"][0],
        planning_metrics=metrics,
    )


def _validate_task_group_coverage(
    task: Mapping[str, Any], graph: Mapping[str, Any], *, route: str
) -> None:
    """Ensure every explicit L1-L3 task instance has one complete group."""
    if task.get("level") == "L4":
        # L4's reference instances are intentionally hidden from the online
        # route; the graph validator still enforces non-empty, coherent groups.
        return
    expected = {
        str(item["id"])
        for item in task.get("task_instances", ())
        if isinstance(item, Mapping)
    }
    actual = {str(group["id"]) for group in graph.get("task_groups", ())}
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            f"{route} SeedGraph TaskGroup coverage mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )


_PRIVATE_OR_LIVE_KEYS = frozenset(
    {
        "absolute_position",
        "coordinates",
        "grasp_pose",
        "joint_positions",
        "live_pose",
        "live_transform",
        "object_pose",
        "oracle",
        "pose",
        "positions",
        "qpos",
        "target_pose",
        "trajectory",
        "waypoints",
        "xpos",
    }
)


def _reject_private_or_live_fields(value: Any, context: str) -> None:
    """Reject private-oracle and grounded state fields in online inputs/outputs."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _PRIVATE_OR_LIVE_KEYS:
                raise ValueError(f"{context} contains private/live field {key!r}.")
            _reject_private_or_live_fields(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_private_or_live_fields(child, f"{context}[{index}]")
