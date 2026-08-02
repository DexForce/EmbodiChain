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

from collections.abc import Mapping
from typing import Any

import pytest

from embodichain.gen_sim.action_engine.compiler import compile_task_agent
from embodichain.gen_sim.action_engine.domain import (
    EXECUTION_PROGRAM_SCHEMA,
    TASK_AGENT_SCHEMA,
)


def _program(step: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "operator_demo",
        "goal": "Exercise one semantic operator.",
        "semantic_steps": [dict(step)],
    }


@pytest.mark.parametrize(
    ("step", "expected_action"),
    [
        (
            {
                "id": "s01_line",
                "operator": "arrange_line",
                "objects": ["can_a", "can_b"],
                "goal": {"axis": "world_y", "anchor": "table_center"},
            },
            "MoveHeldObject",
        ),
        (
            {
                "id": "s01_stack",
                "operator": "build_stack",
                "objects": ["block_a", "block_b"],
                "goal": {"stack_mode": "on_top", "anchor": "table_center"},
            },
            "PickUp",
        ),
        (
            {
                "id": "s01_place",
                "operator": "place_relative",
                "object": "cup",
                "goal": {"reference_object": "tray", "relation": "on"},
            },
            "Place",
        ),
        (
            {
                "id": "s01_hover",
                "operator": "hold_hover",
                "object": "cup",
                "goal": {},
            },
            "MoveJoints",
        ),
        (
            {
                "id": "s01_transport",
                "operator": "coordinated_transport",
                "object": "tray",
                "goal": {"direction": "front", "terminal_behavior": "place"},
            },
            "CoordinatedPickment",
        ),
        (
            {
                "id": "s01_orient",
                "operator": "orient_object",
                "object": "cup",
                "goal": {
                    "orientation_goal": "upright",
                    "orientation_axis": "none",
                },
            },
            "MoveHeldObject",
        ),
        (
            {
                "id": "s01_press",
                "operator": "press",
                "object": "button",
                "goal": {"terminal_state": "activated"},
            },
            "Press",
        ),
        (
            {
                "id": "s01_coordinated_place",
                "operator": "coordinated_place",
                "object": "cup",
                "goal": {"support_object": "tray", "relation": "on"},
            },
            "CoordinatedPlacement",
        ),
    ],
)
def test_every_builtin_operator_compiles(
    step: Mapping[str, Any],
    expected_action: str,
) -> None:
    execution = compile_task_agent(_program(step))
    action_classes = {
        action["atomic_action_class"]
        for edge in execution["edges"]
        for action in edge["actions"]
    }

    assert execution["schema_version"] == EXECUTION_PROGRAM_SCHEMA
    assert expected_action in action_classes
    assert execution["nodes"][0]["id"] == execution["start"]
    assert execution["goal"] in {node["id"] for node in execution["nodes"]}


def test_collective_operator_expands_and_composes_with_press() -> None:
    program = {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "arrange_then_press",
        "goal": "Arrange both cans, then press the button.",
        "semantic_steps": [
            {
                "id": "s01_line",
                "operator": "arrange_line",
                "objects": ["can_a", "can_b"],
                "goal": {"axis": "world_y", "anchor": "table_center"},
                "depends_on": [],
            },
            {
                "id": "s02_press",
                "operator": "press",
                "object": "button",
                "goal": {},
                "depends_on": ["s01_line"],
            },
        ],
    }

    execution = compile_task_agent(program)
    steps = {step["id"]: step for step in execution["semantic_steps"]}

    assert list(steps) == ["s01_line__01", "s01_line__02", "s02_press"]
    assert steps["s02_press"]["depends_on"] == [
        "s01_line__01",
        "s01_line__02",
    ]
    assert execution["edges"][-1]["depends_on"] == [
        steps["s01_line__01"]["edge_ids"][-1],
        steps["s01_line__02"]["edge_ids"][-1],
    ]
    assert "route" not in repr(execution)


def test_coordinated_place_picks_both_objects_before_placement() -> None:
    execution = compile_task_agent(
        _program(
            {
                "id": "s01_coordinated_place",
                "operator": "coordinated_place",
                "object": "cup",
                "goal": {"support_object": "tray", "relation": "on"},
            }
        )
    )
    step = execution["semantic_steps"][0]
    first_edge, placement_edge = [
        next(edge for edge in execution["edges"] if edge["id"] == edge_id)
        for edge_id in step["edge_ids"]
    ]

    assert [action["atomic_action_class"] for action in first_edge["actions"]] == [
        "PickUp",
        "PickUp",
    ]
    assert [action["actor"] for action in first_edge["actions"]] == [
        {"mode": "required", "arm": "left_arm"},
        {"mode": "required", "arm": "right_arm"},
    ]
    assert [action["target_binding"]["object"] for action in first_edge["actions"]] == [
        "cup",
        "tray",
    ]
    assert [action["motion_policy"] for action in first_edge["actions"]] == [
        "default_pickup",
        "default_pickup",
    ]
    assert placement_edge["actions"][0]["atomic_action_class"] == (
        "CoordinatedPlacement"
    )
    assert placement_edge["depends_on"] == [first_edge["id"]]


def test_independent_required_arms_create_allocation_group() -> None:
    program = {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "parallel_place",
        "goal": "Place two objects with opposite arms.",
        "semantic_steps": [
            {
                "id": "s01_left",
                "operator": "place_relative",
                "object": "left_object",
                "actor": {"mode": "required", "arm": "left_arm"},
                "goal": {"reference_object": "left_tray", "relation": "on"},
                "depends_on": [],
            },
            {
                "id": "s02_right",
                "operator": "place_relative",
                "object": "right_object",
                "actor": {"mode": "required", "arm": "right_arm"},
                "goal": {"reference_object": "right_tray", "relation": "on"},
                "depends_on": [],
            },
        ],
    }

    execution = compile_task_agent(program)

    assert execution["allocation_groups"] == [
        {
            "id": "g01_distinct_arms",
            "semantic_step_ids": ["s01_left", "s02_right"],
            "arm_constraint": "distinct_arms",
            "execution_policy": "parallel_if_feasible",
            "parallel_action_classes": ["PickUp"],
            "workspace_policy": "shared_target_serial",
        }
    ]


def test_auto_pickups_require_shared_explicit_allocation_group() -> None:
    program = {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "dual_arm_basket",
        "goal": "Use both arms to place the cube and cup in the basket.",
        "semantic_steps": [
            {
                "id": "s01_cube",
                "operator": "place_relative",
                "object": "cube",
                "actor": {
                    "mode": "auto",
                    "allocation_group": "dual_arms_1",
                },
                "goal": {"reference_object": "basket", "relation": "inside"},
                "depends_on": [],
            },
            {
                "id": "s02_cup",
                "operator": "place_relative",
                "object": "cup",
                "actor": {
                    "mode": "auto",
                    "allocation_group": "dual_arms_1",
                },
                "goal": {"reference_object": "basket", "relation": "inside"},
                "depends_on": [],
            },
        ],
    }

    execution = compile_task_agent(program)
    pickup_edges = [
        next(
            edge
            for edge in execution["edges"]
            if edge["semantic_step_id"] == step_id
            and edge["actions"][0]["atomic_action_class"] == "PickUp"
        )
        for step_id in ("s01_cube", "s02_cup")
    ]
    transport_edges = [
        next(
            edge
            for edge in execution["edges"]
            if edge["semantic_step_id"] == step_id
            and edge["actions"][0]["atomic_action_class"] == "MoveHeldObject"
        )
        for step_id in ("s01_cube", "s02_cup")
    ]

    assert execution["allocation_groups"] == [
        {
            "id": "g01_distinct_arms",
            "semantic_step_ids": ["s01_cube", "s02_cup"],
            "arm_constraint": "distinct_arms",
            "execution_policy": "parallel_if_feasible",
            "parallel_action_classes": ["PickUp"],
            "workspace_policy": "shared_target_serial",
        }
    ]
    assert all("workspace:basket" not in edge["resources"] for edge in pickup_edges)
    assert all("workspace:basket" in edge["resources"] for edge in transport_edges)

    for step in program["semantic_steps"]:
        step["actor"].pop("allocation_group")
    assert compile_task_agent(program)["allocation_groups"] == []


def test_unrelated_dependent_is_allowed_while_hold_reserves_arm() -> None:
    program = {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "hold_then_press",
        "goal": "Hold the cube and then press the button.",
        "semantic_steps": [
            {
                "id": "s01_hold",
                "operator": "hold_hover",
                "object": "cube",
                "actor": {"mode": "auto"},
                "goal": {},
                "depends_on": [],
            },
            {
                "id": "s02_press",
                "operator": "press",
                "object": "button",
                "actor": {"mode": "auto"},
                "goal": {},
                "depends_on": ["s01_hold"],
            },
        ],
    }

    execution = compile_task_agent(program)
    steps = {step["id"]: step for step in execution["semantic_steps"]}

    assert steps["s01_hold"]["postcondition"] == {
        "type": "object_held",
        "object": "cube",
    }
    assert steps["s02_press"]["depends_on"] == ["s01_hold"]


def test_hold_may_follow_an_ancestor_that_previously_used_the_same_object() -> None:
    program = {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "place_then_hold",
        "goal": "Place the cube, then pick it up and keep holding it.",
        "semantic_steps": [
            {
                "id": "s01_place",
                "operator": "place_relative",
                "object": "cube",
                "actor": {"mode": "auto"},
                "goal": {"reference_object": "tray", "relation": "on"},
                "depends_on": [],
            },
            {
                "id": "s02_hold",
                "operator": "hold_hover",
                "object": "cube",
                "actor": {"mode": "required", "arm": "left_arm"},
                "goal": {},
                "depends_on": ["s01_place"],
            },
        ],
    }

    execution = compile_task_agent(program)

    assert execution["semantic_steps"][-1]["postcondition"]["type"] == "object_held"


@pytest.mark.parametrize(
    ("operator", "actor", "goal"),
    [
        ("press", {"mode": "required", "arm": "left_arm"}, {}),
        (
            "coordinated_transport",
            {"mode": "coordinated", "arms": ["left_arm", "right_arm"]},
            {"direction": "none", "terminal_behavior": "hold"},
        ),
    ],
)
def test_required_hold_rejects_later_steps_that_need_its_arm(
    operator: str,
    actor: dict,
    goal: dict,
) -> None:
    program = {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "occupied_arm",
        "goal": "Keep holding the cube, then operate the button.",
        "semantic_steps": [
            {
                "id": "s01_hold",
                "operator": "hold_hover",
                "object": "cube",
                "actor": {"mode": "required", "arm": "left_arm"},
                "goal": {},
                "depends_on": [],
            },
            {
                "id": "s02_other",
                "operator": operator,
                "object": "button",
                "actor": actor,
                "goal": goal,
                "depends_on": ["s01_hold"],
            },
        ],
    }

    with pytest.raises(ValueError, match="reserves arm 'left_arm'"):
        compile_task_agent(program)


def test_held_object_cannot_be_reused_by_an_independent_step() -> None:
    program = {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "conflicting_object_ownership",
        "goal": "Hold and place the same cube.",
        "semantic_steps": [
            {
                "id": "s01_hold",
                "operator": "hold_hover",
                "object": "cube",
                "actor": {"mode": "auto"},
                "goal": {},
                "depends_on": [],
            },
            {
                "id": "s02_place",
                "operator": "place_relative",
                "object": "cube",
                "actor": {"mode": "auto"},
                "goal": {"reference_object": "tray", "relation": "on"},
                "depends_on": [],
            },
        ],
    }

    with pytest.raises(ValueError, match="reserves object 'cube'"):
        compile_task_agent(program)


def test_unknown_operator_is_rejected_before_graph_construction() -> None:
    with pytest.raises(ValueError, match="Unknown semantic operator"):
        compile_task_agent(
            _program(
                {
                    "id": "s01_unknown",
                    "operator": "teleport",
                    "object": "cube",
                    "goal": {},
                }
            )
        )


def test_coordinated_transport_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        compile_task_agent(
            _program(
                {
                    "id": "s01_transport",
                    "operator": "coordinated_transport",
                    "object": "shared_box",
                    "actor": {
                        "mode": "coordinated",
                        "arms": ["left_arm", "right_arm"],
                    },
                    "goal": {
                        "direction": "somewhere_vague",
                        "terminal_behavior": "hold",
                    },
                    "depends_on": [],
                }
            )
        )
