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

import pytest

from embodichain.gen_sim.action_engine.compiler import (
    compile_task_agent,
    compile_task_agent_v2,
    seed_graph_to_execution_program,
)
from embodichain.gen_sim.action_engine.domain import TASK_AGENT_SCHEMA
from embodichain.gen_sim.action_engine.protocol import SEED_GRAPH_SCHEMA


def _task_agent() -> dict:
    return {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "place-cup",
        "goal": "Place the cup in the tray.",
        "semantic_steps": [
            {
                "id": "s01_place_cup",
                "operator": "place_relative",
                "object": "cup",
                "actor": {"mode": "auto"},
                "goal": {
                    "relation": "inside",
                    "reference_object": "tray",
                    "reference_state": "live",
                    "slot": "auto",
                    "orientation_goal": "preserve",
                    "orientation_axis": "none",
                },
                "depends_on": [],
            }
        ],
        "allocation_groups": [],
    }


def test_v2_compiler_preserves_mature_atomic_action_topology() -> None:
    known = {"cup", "tray"}
    legacy = compile_task_agent(_task_agent(), known_objects=known)
    seed = compile_task_agent_v2(_task_agent(), known_objects=known)
    materialized = seed_graph_to_execution_program(seed, known_objects=known)

    legacy_actions = [
        action["atomic_action_class"]
        for edge in legacy["edges"]
        for action in edge["actions"]
    ]
    seed_actions = [node["atomic_action"] for node in seed["nodes"]]
    materialized_actions = [
        action["atomic_action_class"]
        for edge in materialized["edges"]
        for action in edge["actions"]
    ]
    assert seed["schema_version"] == SEED_GRAPH_SCHEMA
    assert seed_actions == legacy_actions
    assert materialized_actions == legacy_actions
    assert seed["task_groups"][0]["task_type"] == "E1"


@pytest.mark.parametrize(
    ("operator", "objects", "goal", "actor"),
    [
        (
            "orient_object",
            ["can"],
            {
                "orientation_goal": "upright",
                "orientation_axis": "long_axis",
                "position_anchor": "initial_xy",
                "support_object": "table",
                "upright_local_axis": "long_axis",
            },
            {"mode": "auto"},
        ),
        (
            "coordinated_transport",
            ["tray"],
            {
                "direction": "up",
                "terminal_behavior": "hold",
                "orientation_goal": "preserve",
                "orientation_axis": "none",
            },
            {"mode": "coordinated", "arms": ["left_arm", "right_arm"]},
        ),
        (
            "build_stack",
            ["cube_a", "cube_b"],
            {
                "anchor": "table_center",
                "stack_mode": "on_top",
                "orientation_goal": "preserve",
                "orientation_axis": "none",
            },
            {"mode": "auto"},
        ),
        (
            "arrange_line",
            ["cube_a", "cube_b"],
            {
                "anchor": "table_center",
                "axis": "world_y",
                "order_by": "explicit",
                "order_constraint": "ordered",
                "order_direction": "given",
                "orientation_goal": "preserve",
                "orientation_axis": "none",
                "participation": "auto",
            },
            {"mode": "auto"},
        ),
    ],
)
def test_v2_preserves_all_current_task_recipe_topologies(
    operator: str,
    objects: list[str],
    goal: dict,
    actor: dict,
) -> None:
    step = {
        "id": "task_01",
        "operator": operator,
        "actor": actor,
        "goal": goal,
        "depends_on": [],
    }
    if operator in {"build_stack", "arrange_line"}:
        step["objects"] = objects
    else:
        step["object"] = objects[0]
    task = {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": f"regression-{operator}",
        "goal": f"Regression task for {operator}.",
        "semantic_steps": [step],
        "allocation_groups": [],
    }
    known = {*objects, "table"}
    legacy = compile_task_agent(task, known_objects=known)
    seed = compile_task_agent_v2(task, known_objects=known)
    rematerialized = seed_graph_to_execution_program(seed, known_objects=known)

    def signature(program: dict) -> dict[str, list[list[str]]]:
        edges = {edge["id"]: edge for edge in program["edges"]}
        return {
            step["id"]: [
                [action["atomic_action_class"] for action in edges[edge_id]["actions"]]
                for edge_id in step["edge_ids"]
            ]
            for step in program["semantic_steps"]
        }

    assert signature(rematerialized) == signature(legacy)
