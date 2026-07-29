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

"""Convert deterministic task plans into executable nominal task graphs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    ArrangementSpecLike,
    RelativeSpecLike,
    StackingSpecLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.nominal_graph import (
    build_nominal_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    compile_seed_graph_metadata,
    make_arrangement_seed_task_graph,
    make_relative_seed_task_graph,
    make_stacking_seed_task_graph,
    validate_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_plan_builders import (
    _arrangement_step_edge_blocks,
    _coordinated_pickment_graph_steps,
    _dual_relative_graph_steps,
    _hold_hover_graph_steps,
    _nominal_step,
    _single_relative_graph_steps,
    _stacking_step_edge_blocks,
    _uses_serial_dual_sequence,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_offsets import (
    _side_relation_xy_offsets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.success_specs import (
    _make_relative_placement_success_spec,
)

__all__ = [
    "compile_arrangement_task_graph",
    "compile_relative_task_graph",
    "compile_stacking_task_graph",
    "make_arrangement_task_graph",
    "make_stacking_task_graph",
    "make_relative_task_graph",
]


def make_arrangement_task_graph(
    task_name: str,
    spec: ArrangementSpecLike,
) -> dict[str, Any]:
    seed_graph = make_arrangement_seed_task_graph(task_name, spec)
    return compile_arrangement_task_graph(task_name, seed_graph, spec)


def compile_arrangement_task_graph(
    task_name: str,
    seed_graph: Mapping[str, Any],
    spec: ArrangementSpecLike,
) -> dict[str, Any]:
    """Compile a symbolic arrangement seed into its grounded atomic graph."""
    validate_seed_task_graph(
        seed_graph,
        task_name=task_name,
        route="arrangement_line",
    )
    _validate_arrangement_seed(seed_graph, spec)
    steps = []
    # The seed owns the desired ordered layout. The deterministic arrangement
    # planner remains free to choose a collision-safe execution schedule.
    for step in spec.steps:
        steps.extend(
            _nominal_step(title, actions)
            for title, actions in _arrangement_step_edge_blocks(step)
        )
    graph = build_nominal_task_graph(task_name=task_name, steps=steps)
    return compile_seed_graph_metadata(graph, seed_graph)


def make_stacking_task_graph(
    task_name: str,
    spec: StackingSpecLike,
) -> dict[str, Any]:
    seed_graph = make_stacking_seed_task_graph(task_name, spec)
    return compile_stacking_task_graph(task_name, seed_graph, spec)


def compile_stacking_task_graph(
    task_name: str,
    seed_graph: Mapping[str, Any],
    spec: StackingSpecLike,
) -> dict[str, Any]:
    """Compile a symbolic stacking seed into its grounded atomic graph."""
    validate_seed_task_graph(seed_graph, task_name=task_name, route="stacking")
    ordered_steps = _unique_specs_in_seed_order(seed_graph, spec.steps, "runtime_uid")
    steps = []
    for step in ordered_steps:
        steps.extend(
            _nominal_step(title, actions)
            for title, actions in _stacking_step_edge_blocks(
                step,
                object_anchored=spec.anchor == "object",
                stack_mode=spec.stack_mode,
            )
        )
    graph = build_nominal_task_graph(task_name=task_name, steps=steps)
    return compile_seed_graph_metadata(graph, seed_graph)


def make_relative_task_graph(
    task_name: str,
    spec: RelativeSpecLike,
) -> dict[str, Any]:
    seed_graph = make_relative_seed_task_graph(task_name, spec)
    return compile_relative_task_graph(task_name, seed_graph, spec)


def compile_relative_task_graph(
    task_name: str,
    seed_graph: Mapping[str, Any],
    spec: RelativeSpecLike,
) -> dict[str, Any]:
    """Compile ordered manipulation semantics with deterministic geometry."""
    validate_seed_task_graph(
        seed_graph,
        task_name=task_name,
        route="object_manipulation",
    )
    seed_steps = list(seed_graph["steps"])
    _validate_relative_seed_order(seed_steps, spec)
    semantic_groups = None
    if spec.intent == "coordinated_pickment":
        steps = _coordinated_pickment_graph_steps(spec)
    elif spec.intent == "hold_hover":
        steps = _hold_hover_graph_steps(spec)
    elif len(spec.placements) > 1:
        if _uses_serial_dual_sequence(spec):
            semantic_groups = [
                (placement, _single_relative_graph_steps(placement))
                for placement in spec.placements
            ]
            steps = [step for _, group_steps in semantic_groups for step in group_steps]
        else:
            steps = _dual_relative_graph_steps(spec)
    else:
        steps = _single_relative_graph_steps(spec)
        semantic_groups = [(spec.placements[0], steps)]
    graph = build_nominal_task_graph(task_name=task_name, steps=steps)
    if semantic_groups is not None:
        graph["semantic_step_schema_version"] = "semantic_steps_v1"
        graph["semantic_steps"] = _relative_semantic_steps(
            graph,
            semantic_groups,
            seed_steps,
        )
    return compile_seed_graph_metadata(graph, seed_graph)


def _relative_semantic_steps(
    graph: dict[str, Any],
    groups: list[tuple[Any, list[Any]]],
    seed_steps: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind ordered semantic operations to their generated atomic edge ranges."""
    if len(groups) != len(seed_steps):
        raise ValueError(
            "Semantic seed steps do not match the compiled relative edge groups."
        )
    graph_edges = graph["edges"]
    result: list[dict[str, Any]] = []
    edge_cursor = 0
    for (placement, atomic_steps), seed_step in zip(groups, seed_steps):
        edge_count = len(atomic_steps)
        edge_ids = [
            edge["id"] for edge in graph_edges[edge_cursor : edge_cursor + edge_count]
        ]
        edge_cursor += edge_count
        result.append(
            {
                "id": seed_step["id"],
                "operator": seed_step["operator"],
                "object": seed_step["object"],
                "actor": _compiled_relative_actor(seed_step["actor"], placement),
                "goal": deepcopy(dict(seed_step["goal"])),
                "depends_on": list(seed_step["depends_on"]),
                "postcondition": _make_relative_placement_success_spec(
                    placement,
                    side_relation_xy_offsets=_side_relation_xy_offsets,
                ),
                "edge_ids": edge_ids,
            }
        )
    if edge_cursor != len(graph_edges):
        raise ValueError("Semantic-step edge groups do not cover the generated graph.")
    return result


def _compiled_relative_actor(
    seed_actor: Mapping[str, Any],
    placement: Any,
) -> dict[str, Any]:
    active_arm = f"{placement.active_side}_arm"
    if seed_actor["mode"] == "required":
        required_arm = seed_actor["arm"]
        if required_arm != active_arm:
            raise ValueError(
                f"Required seed actor {required_arm!r} was changed to "
                f"{active_arm!r} during deterministic compilation."
            )
        return {"mode": "required", "arm": required_arm}
    return {"mode": "assigned", "arm": active_arm}


def _validate_relative_seed_order(
    seed_steps: list[Mapping[str, Any]],
    spec: RelativeSpecLike,
) -> None:
    placements = list(spec.placements)
    if len(seed_steps) != len(placements):
        raise ValueError(
            "Relative seed step count does not match the normalized LLM program."
        )
    for index, (seed_step, placement) in enumerate(zip(seed_steps, placements)):
        expected_relation = (
            "held_above_initial"
            if placement.intent == "hold_hover"
            else placement.relation
        )
        expected_reference = (
            placement.moved_runtime_uid
            if placement.intent == "hold_hover"
            else placement.reference_runtime_uid
        )
        expected = (
            placement.intent,
            placement.moved_runtime_uid,
            expected_relation,
            expected_reference,
        )
        actual_goal = seed_step["goal"]
        actual = (
            seed_step["operator"],
            seed_step["object"],
            actual_goal.get("relation"),
            actual_goal.get("reference_object"),
        )
        if actual != expected:
            raise ValueError(
                f"Relative seed step {index} no longer matches its normalized "
                "LLM semantic step."
            )


def _validate_arrangement_seed(
    seed_graph: Mapping[str, Any],
    spec: ArrangementSpecLike,
) -> None:
    seed_steps = seed_graph["steps"]
    if len(seed_steps) != 1 or seed_steps[0]["operator"] != "arrange_in_line":
        raise ValueError("Arrangement seed graph requires one arrange_in_line step.")
    goal_objects = seed_steps[0]["goal"].get("objects")
    expected_objects = [
        step.runtime_uid
        for step in sorted(spec.steps, key=lambda item: int(item.slot_index))
    ]
    if goal_objects != expected_objects:
        raise ValueError(
            "Arrangement seed object order does not match the normalized LLM goal."
        )


def _unique_specs_in_seed_order(
    seed_graph: Mapping[str, Any],
    specs: Any,
    uid_attribute: str,
) -> list[Any]:
    spec_list = list(specs)
    by_uid = {str(getattr(spec, uid_attribute)): spec for spec in spec_list}
    seed_uids = [str(step["object"]) for step in seed_graph["steps"]]
    if len(by_uid) != len(spec_list) or set(seed_uids) != set(by_uid):
        raise ValueError(
            "Seed graph objects do not match the deterministic route specification."
        )
    return [by_uid[uid] for uid in seed_uids]
