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

from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
    compile_agent_graph_spec,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atom_actions import (
    AtomicActionSpec,
)


class _FakeGraph:
    def __init__(self, start: str, goal: str, max_transitions: int = 1000) -> None:
        self.start = start
        self.goal = goal
        self.max_transitions = max_transitions
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id: str, semantic: str = ""):
        self.nodes[node_id] = semantic
        return self

    def add_edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        *,
        left_arm_action=None,
        right_arm_action=None,
    ):
        self.edges[edge_id] = {
            "source": source,
            "target": target,
            "left_arm_action": left_arm_action,
            "right_arm_action": right_arm_action,
        }
        return self


def _pick_up_spec(robot_name: str, obj_name: str) -> dict:
    return {
        "atomic_action_class": "PickUp",
        "robot_name": robot_name,
        "control": "arm",
        "target_object": {
            "obj_name": obj_name,
            "affordance": "antipodal",
        },
        "cfg": {
            "pre_grasp_distance": 0.08,
            "sample_interval": 45,
        },
    }


def _task_graph(action: dict) -> dict:
    return {
        "task": "unit",
        "start": "v0_start",
        "goal": "v1_done",
        "nodes": [
            {"id": "v0_start"},
            {"id": "v1_done"},
        ],
        "edges": [
            {
                "id": "e01",
                "source": "v0_start",
                "target": "v1_done",
                "left_arm_action": action,
                "right_arm_action": None,
            }
        ],
    }


def test_compile_agent_graph_accepts_atomic_action_class_spec() -> None:
    action = _pick_up_spec("left_arm", "apple")
    graph = compile_agent_graph_spec(
        _task_graph(action),
        graph_cls=_FakeGraph,
    )

    left_arm_action = graph.edges["e01"]["left_arm_action"]
    assert isinstance(left_arm_action, AtomicActionSpec)
    assert left_arm_action.to_dict() == action


def test_compile_agent_graph_rejects_legacy_action_schema() -> None:
    task_graph = _task_graph({"action": "pick_up", "robot_name": "left_arm"})

    with pytest.raises(ValueError, match="Legacy action schema"):
        compile_agent_graph_spec(
            task_graph,
            graph_cls=_FakeGraph,
        )


def test_compile_agent_graph_rejects_extra_edge_fields() -> None:
    task_graph = _task_graph(_pick_up_spec("left_arm", "apple"))
    task_graph["edges"][0]["monitor"] = {"condition": "object visible"}

    with pytest.raises(ValueError, match="unsupported fields: monitor"):
        compile_agent_graph_spec(
            task_graph,
            graph_cls=_FakeGraph,
        )


def test_compile_agent_graph_rejects_empty_string_arm_actions() -> None:
    task_graph = _task_graph(_pick_up_spec("left_arm", "apple"))
    task_graph["edges"][0]["left_arm_action"] = ""
    task_graph["edges"][0]["right_arm_action"] = "none"

    with pytest.raises(ValueError, match="must define an arm action"):
        compile_agent_graph_spec(
            task_graph,
            graph_cls=_FakeGraph,
        )
