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

from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    ArrangementSpecLike,
    RelativeSpecLike,
    StackingSpecLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.nominal_graph import (
    build_nominal_task_graph,
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
    "make_arrangement_task_graph",
    "make_stacking_task_graph",
    "make_relative_task_graph",
]


def make_arrangement_task_graph(
    task_name: str,
    spec: ArrangementSpecLike,
) -> dict[str, Any]:
    steps = []
    for step in spec.steps:
        steps.extend(
            _nominal_step(title, actions)
            for title, actions in _arrangement_step_edge_blocks(step)
        )
    return build_nominal_task_graph(task_name=task_name, steps=steps)


def make_stacking_task_graph(
    task_name: str,
    spec: StackingSpecLike,
) -> dict[str, Any]:
    steps = []
    for step in spec.steps:
        steps.extend(
            _nominal_step(title, actions)
            for title, actions in _stacking_step_edge_blocks(
                step,
                object_anchored=spec.anchor == "object",
                stack_mode=spec.stack_mode,
            )
        )
    return build_nominal_task_graph(task_name=task_name, steps=steps)


def make_relative_task_graph(
    task_name: str,
    spec: RelativeSpecLike,
) -> dict[str, Any]:
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
        graph["semantic_steps"] = _relative_semantic_steps(graph, semantic_groups)
    return graph


def _relative_semantic_steps(
    graph: dict[str, Any],
    groups: list[tuple[Any, list[Any]]],
) -> list[dict[str, Any]]:
    """Bind ordered semantic operations to their generated atomic edge ranges."""
    graph_edges = graph["edges"]
    result: list[dict[str, Any]] = []
    edge_cursor = 0
    previous_step_id: str | None = None
    for index, (placement, atomic_steps) in enumerate(groups, start=1):
        edge_count = len(atomic_steps)
        edge_ids = [
            edge["id"] for edge in graph_edges[edge_cursor : edge_cursor + edge_count]
        ]
        edge_cursor += edge_count
        step_id = getattr(placement, "step_id", "") or f"s{index:02d}_step"
        configured_dependencies = tuple(getattr(placement, "depends_on", ()))
        depends_on = list(
            configured_dependencies
            or ((previous_step_id,) if previous_step_id is not None else ())
        )
        arm_request = str(getattr(placement, "arm_request", "auto"))
        result.append(
            {
                "id": step_id,
                "operator": placement.intent,
                "object": placement.moved_runtime_uid,
                "actor": {
                    "mode": "required" if arm_request != "auto" else "assigned",
                    "arm": f"{placement.active_side}_arm",
                },
                "goal": {
                    "relation": placement.relation,
                    "reference_object": placement.reference_runtime_uid,
                    "reference_state": (
                        "initial" if placement.reference_is_initial_pose else "live"
                    ),
                },
                "depends_on": depends_on,
                "postcondition": _make_relative_placement_success_spec(
                    placement,
                    side_relation_xy_offsets=_side_relation_xy_offsets,
                ),
                "edge_ids": edge_ids,
            }
        )
        previous_step_id = step_id
    if edge_cursor != len(graph_edges):
        raise ValueError("Semantic-step edge groups do not cover the generated graph.")
    return result
