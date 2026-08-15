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

from copy import deepcopy

import pytest

from embodichain.gen_sim.action_engine.compiler import compile_task_agent
from embodichain.gen_sim.action_engine.domain import (
    TASK_AGENT_SCHEMA,
    execution_program_hash,
    validate_execution_program,
    validate_task_agent,
)


def _task_agent() -> dict:
    return {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "place_demo",
        "goal": "Place the cup on the tray.",
        "semantic_steps": [
            {
                "id": "s01_place",
                "operator": "place_relative",
                "object": "cup",
                "actor": {"mode": "auto"},
                "goal": {"reference_object": "tray", "relation": "on"},
                "depends_on": [],
            }
        ],
    }


def test_task_validation_is_detached_and_adds_unambiguous_defaults() -> None:
    source = _task_agent()
    del source["semantic_steps"][0]["actor"]
    del source["semantic_steps"][0]["depends_on"]

    validated = validate_task_agent(source)
    validated["semantic_steps"][0]["goal"]["relation"] = "inside"

    assert source["semantic_steps"][0]["goal"]["relation"] == "on"
    assert validated["semantic_steps"][0]["actor"] == {"mode": "auto"}
    assert validated["semantic_steps"][0]["depends_on"] == []


@pytest.mark.parametrize(
    "actor",
    [
        {"mode": "auto", "allocation_group": "dual_arms_1"},
        {
            "mode": "required",
            "arm": "left_arm",
            "allocation_group": "dual_arms_1",
        },
    ],
)
def test_single_arm_allocation_group_is_validated_and_preserved(actor: dict) -> None:
    source = _task_agent()
    source["semantic_steps"][0]["actor"] = actor

    validated = validate_task_agent(source)
    execution = compile_task_agent(validated)

    assert validated["semantic_steps"][0]["actor"] == actor
    assert execution["semantic_steps"][0]["actor"] == actor
    assert all(
        action["actor"] == actor
        for edge in execution["edges"]
        for action in edge["actions"]
    )


def test_allocation_group_must_be_nonempty_and_single_arm_only() -> None:
    source = _task_agent()
    source["semantic_steps"][0]["actor"]["allocation_group"] = " "
    with pytest.raises(ValueError, match="allocation_group"):
        validate_task_agent(source)

    source["semantic_steps"][0]["actor"] = {
        "mode": "coordinated",
        "arms": ["left_arm", "right_arm"],
        "allocation_group": "dual_arms_1",
    }
    with pytest.raises(ValueError, match="unknown fields"):
        validate_task_agent(source)


def test_task_validation_rejects_cycles_and_grounded_values() -> None:
    cyclic = _task_agent()
    cyclic["semantic_steps"].extend(
        [
            {
                "id": "s02",
                "operator": "press",
                "object": "button",
                "depends_on": ["s03"],
            },
            {
                "id": "s03",
                "operator": "press",
                "object": "button",
                "depends_on": ["s02"],
            },
        ]
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_task_agent(cyclic)

    grounded = _task_agent()
    grounded["semantic_steps"][0]["goal"]["target_pose"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="grounded runtime data"):
        validate_task_agent(grounded)


def test_execution_hash_is_stable_and_validation_is_strict() -> None:
    execution = compile_task_agent(_task_agent())
    reordered = {key: execution[key] for key in reversed(list(execution))}

    assert execution_program_hash(execution) == execution_program_hash(reordered)
    assert len(execution_program_hash(execution)) == 64

    broken = deepcopy(execution)
    broken["edges"][0]["target_binding"] = {}
    with pytest.raises(ValueError, match="unknown fields"):
        validate_execution_program(broken)


def test_execution_validation_rejects_unowned_edges() -> None:
    execution = compile_task_agent(_task_agent())
    execution["semantic_steps"][0]["edge_ids"].pop()

    with pytest.raises(ValueError, match="unowned edges"):
        validate_execution_program(execution)
