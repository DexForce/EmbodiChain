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

from embodichain.gen_sim.action_agent_pipeline.generation.nominal_graph import (
    NominalGraphStep,
    build_nominal_task_graph,
)


def test_nominal_graph_builds_single_chain_with_goal_node() -> None:
    first_action = {
        "atomic_action_class": "PickUp",
        "robot_name": "left_arm",
        "control": "arm",
        "cfg": {"sample_interval": 45},
    }
    second_action = {
        "atomic_action_class": "Place",
        "robot_name": "left_arm",
        "control": "arm",
        "cfg": {"sample_interval": 80},
    }

    graph = build_nominal_task_graph(
        task_name="UnitTask",
        steps=[
            NominalGraphStep("Pick up the object", left_arm_action=first_action),
            NominalGraphStep("Place the object", left_arm_action=second_action),
        ],
    )

    assert graph["start"] == "v0_start"
    assert graph["goal"] == "v2_done"
    assert graph["nodes"][-1]["id"] == "v2_done"
    assert len(graph["nodes"]) == len(graph["edges"]) + 1
    assert [edge["source"] for edge in graph["edges"]] == [
        "v0_start",
        "v1_pick_up_the_object",
    ]
    assert [edge["target"] for edge in graph["edges"]] == [
        "v1_pick_up_the_object",
        "v2_done",
    ]


def test_nominal_graph_rejects_empty_or_actionless_steps() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        build_nominal_task_graph(task_name="EmptyTask", steps=[])

    with pytest.raises(ValueError, match="has no arm action"):
        build_nominal_task_graph(
            task_name="ActionlessTask",
            steps=[NominalGraphStep("No-op")],
        )


def test_nominal_graph_copies_action_specs() -> None:
    action = {
        "atomic_action_class": "MoveJoints",
        "robot_name": "left_arm",
        "control": "arm",
        "target_qpos": {"source": "initial"},
        "cfg": {"sample_interval": 30},
    }

    graph = build_nominal_task_graph(
        task_name="CopyTask",
        steps=[NominalGraphStep("Return arm", left_arm_action=action)],
    )
    action["cfg"]["sample_interval"] = 1

    assert graph["edges"][0]["left_arm_action"]["cfg"]["sample_interval"] == 30
