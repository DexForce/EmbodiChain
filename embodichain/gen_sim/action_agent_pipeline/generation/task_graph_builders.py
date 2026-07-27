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
    if spec.intent == "coordinated_pickment":
        steps = _coordinated_pickment_graph_steps(spec)
    elif spec.intent == "hold_hover":
        steps = _hold_hover_graph_steps(spec)
    elif len(spec.placements) > 1:
        steps = _dual_relative_graph_steps(spec)
    else:
        steps = _single_relative_graph_steps(spec)
    return build_nominal_task_graph(task_name=task_name, steps=steps)
