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

from embodichain.gen_sim.action_engine.orientation import (
    AlignAxisConstraint,
    MatchRotationConstraint,
    compile_orientation_constraint,
)
from embodichain.gen_sim.action_engine.compiler import compile_task_agent
from embodichain.gen_sim.action_engine.protocol import (
    TASK_AGENT_SCHEMA,
    TASK_SPEC_SCHEMA,
)
from embodichain.gen_sim.action_engine.tasks import instantiate_seed_graph


def test_unspecified_orientation_has_no_hard_constraint() -> None:
    constraint = compile_orientation_constraint({})

    assert constraint.terms == ()
    assert constraint.planning_preference == "minimize_rotation_from_current"
    assert not constraint.requires_reference


def test_explicit_preserve_compiles_to_rotation_match() -> None:
    constraint = compile_orientation_constraint({"orientation_goal": "preserve"})

    assert constraint.terms == (MatchRotationConstraint(reference="step_start"),)
    assert constraint.requires_reference


def test_match_rotation_requires_an_explicit_serialized_term() -> None:
    constraint = compile_orientation_constraint(
        {
            "orientation_constraint": {
                "terms": [
                    {
                        "type": "match_rotation",
                        "reference": "target_pose",
                    }
                ]
            }
        }
    )

    assert constraint.terms == (MatchRotationConstraint(reference="target_pose"),)
    assert not constraint.allows_yaw_search


def test_upright_compiles_to_directed_axis_when_requested() -> None:
    constraint = compile_orientation_constraint(
        {
            "orientation_goal": "upright",
            "upright_local_axis": "z",
            "orientation_directed": True,
        }
    )

    assert constraint.terms == (
        AlignAxisConstraint(
            local_axis="z",
            target_axis="world_up",
            directed=True,
        ),
    )


def test_upright_rejects_non_boolean_directed_flag() -> None:
    with pytest.raises(ValueError, match="orientation_directed must be a boolean"):
        compile_orientation_constraint(
            {
                "orientation_goal": "upright",
                "orientation_directed": "false",
            }
        )


def test_legacy_long_axis_upright_remains_undirected() -> None:
    constraint = compile_orientation_constraint(
        {
            "orientation_goal": "upright",
            "upright_local_axis": "long_axis",
        }
    )

    assert constraint.terms == (
        AlignAxisConstraint(
            local_axis="long_axis",
            target_axis="world_up",
            directed=False,
        ),
    )
    assert constraint.allows_yaw_search


def test_lay_flat_compiles_to_short_axis_alignment_with_free_yaw() -> None:
    constraint = compile_orientation_constraint({"orientation_goal": "lay_flat"})

    assert constraint.terms == (
        AlignAxisConstraint(
            local_axis="short_axis",
            target_axis="world_up",
            directed=False,
        ),
    )
    assert constraint.allows_yaw_search


def test_hold_hover_without_orientation_request_has_no_rotation_match() -> None:
    compiled = compile_task_agent(
        {
            "schema_version": TASK_AGENT_SCHEMA,
            "task": "hold",
            "goal": "Hold the can above its initial position.",
            "semantic_steps": [
                {
                    "id": "hold",
                    "operator": "hold_hover",
                    "object": "can",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {},
                    "depends_on": [],
                }
            ],
        }
    )

    goal = compiled["semantic_steps"][0]["goal"]
    assert goal["orientation_goal"] == "none"
    assert compile_orientation_constraint(goal).terms == ()


def test_serialized_constraint_keeps_term_local_tolerance() -> None:
    constraint = compile_orientation_constraint(
        {
            "orientation_constraint": {
                "terms": [
                    {
                        "type": "align_axis",
                        "local_axis": "z",
                        "target_axis": "world_up",
                        "directed": True,
                        "tolerance": 0.1,
                        "scope": "terminal",
                    }
                ]
            }
        }
    )

    assert constraint.terms == (
        AlignAxisConstraint(
            local_axis="z",
            target_axis="world_up",
            directed=True,
            tolerance=0.1,
        ),
    )


def test_new_placement_without_orientation_request_has_no_hard_constraint() -> None:
    task = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "place_can",
        "level": "L1",
        "instruction": "Place the can beside the box.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "place",
                "task_type": "E1",
                "params": {
                    "object_role": "can",
                    "target_role": "box",
                    "relation": "left_of",
                },
                "depends_on": [],
                "role": "primary",
            }
        ],
        "success": {"type": "semantic_goal"},
        "oracle": {},
        "metadata": {},
    }

    graph = instantiate_seed_graph(task, {"can": "can", "box": "box"})

    assert graph["task_groups"][0]["goal"]["orientation_goal"] == "none"
