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

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_graph_spec_module():
    source_path = REPO_ROOT / "embodichain" / "lab" / "sim" / "agent" / "graph_spec.py"
    spec = importlib.util.spec_from_file_location(
        "agent_graph_spec_under_test",
        source_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


graph_spec_module = _load_graph_spec_module()
compile_agent_graph_spec = graph_spec_module.compile_agent_graph_spec
expand_recovery_spec = graph_spec_module.expand_recovery_spec
normalize_recovery_spec = graph_spec_module.normalize_recovery_spec


@dataclass
class FakeEdge:
    id: str
    source: str
    target: str
    left_arm_action: Any = None
    right_arm_action: Any = None
    monitor_sequences: list[list[Any]] | None = None
    monitor_labels: list[str] | None = None
    is_recovery: bool = False


class FakeGraph:
    def __init__(self, start: str, goal: str, max_transitions: int = 1000) -> None:
        self.start = start
        self.goal = goal
        self.max_transitions = max_transitions
        self.nodes = {}
        self.edges = {}
        self.recovery_branches = {}

    def add_node(self, node_id: str, semantic: str = "") -> "FakeGraph":
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
        monitor_sequences=None,
        monitor_labels=None,
        is_recovery: bool = False,
    ) -> "FakeGraph":
        self.edges[edge_id] = FakeEdge(
            id=edge_id,
            source=source,
            target=target,
            left_arm_action=left_arm_action,
            right_arm_action=right_arm_action,
            monitor_sequences=monitor_sequences,
            monitor_labels=monitor_labels,
            is_recovery=is_recovery,
        )
        return self

    def add_recovery(
        self,
        edge_id: str,
        monitor_index: int,
        recovery_edges: list[str],
    ) -> "FakeGraph":
        self.recovery_branches[(edge_id, monitor_index)] = recovery_edges
        return self


def _nominal_task_graph() -> dict[str, Any]:
    return {
        "task": "grasp cup",
        "start": "v0_start",
        "goal": "v2_done",
        "nodes": [
            {"id": "v0_start", "semantic": "Cup is on table."},
            {"id": "v1_grasped", "semantic": "Cup is grasped."},
            {"id": "v2_done", "semantic": "Cup has been moved."},
        ],
        "edges": [
            {
                "id": "e01_grasp",
                "source": "v0_start",
                "target": "v1_grasped",
                "left_arm_action": {
                    "fn": "grasp",
                    "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                },
                "right_arm_action": None,
            },
            {
                "id": "e12_move",
                "source": "v1_grasped",
                "target": "v2_done",
                "left_arm_action": {"fn": "move", "kwargs": {"robot_name": "left_arm"}},
                "right_arm_action": None,
            },
        ],
    }


def test_expand_recovery_spec_generates_reusable_recovery_graph() -> None:
    recovery_graph = expand_recovery_spec(
        _nominal_task_graph(),
        {
            "task": "grasp cup",
            "recovery_bindings": [
                {
                    "edge_id": "e01_grasp",
                    "failure_name": "cup_moved",
                    "monitors": [
                        {
                            "type": "object_moved",
                            "objects": ["cup"],
                            "threshold": 0.02,
                        }
                    ],
                    "recovery": [
                        {
                            "type": "regrasp",
                            "robot_name": "left_arm",
                            "obj_name": "cup",
                            "pre_grasp_dis": 0.1,
                            "force_valid": True,
                        }
                    ],
                    "merge": "target",
                    "repeat_until_success": True,
                }
            ],
        },
    )

    assert set(recovery_graph) == {
        "task",
        "recovery_nodes",
        "recovery_edges",
        "recovery_branches",
    }
    assert recovery_graph["recovery_nodes"] == []
    assert recovery_graph["recovery_edges"] == [
        {
            "id": "re_e01_grasp_1_regrasp_cup",
            "source": "v0_start",
            "target": "v1_grasped",
            "left_arm_action": {
                "fn": "grasp",
                "kwargs": {
                    "robot_name": "left_arm",
                    "obj_name": "cup",
                    "pre_grasp_dis": 0.1,
                    "force_valid": True,
                },
            },
            "right_arm_action": None,
        }
    ]
    assert recovery_graph["recovery_branches"] == [
        {
            "edge_id": "e01_grasp",
            "failure_name": "cup_moved",
            "monitor_sequence": [
                {
                    "fn": "monitor_object_moved",
                    "kwargs": {"obj_name": "cup", "threshold": 0.02},
                }
            ],
            "recovery_edges": ["re_e01_grasp_1_regrasp_cup"],
        },
        {
            "edge_id": "re_e01_grasp_1_regrasp_cup",
            "failure_name": "cup_moved_during_re_e01_grasp_1_regrasp_cup",
            "monitor_sequence": [
                {
                    "fn": "monitor_object_moved",
                    "kwargs": {"obj_name": "cup", "threshold": 0.02},
                }
            ],
            "recovery_edges": ["re_e01_grasp_1_regrasp_cup"],
        },
    ]


def test_normalize_recovery_spec_infers_simple_aliases() -> None:
    task_graph = _nominal_task_graph()
    recovery_spec = {
        "task": "grasp cup",
        "bindings": [
            {
                "edge": "e01_grasp",
                "failure": "cup_moved",
                "monitor": [{"type": "object shifted"}],
                "recovery": [{"type": "grasp again"}],
                "merge": "complete",
            }
        ],
    }

    normalized, issues = normalize_recovery_spec(task_graph, recovery_spec)

    assert issues == []
    assert normalized == {
        "task": "grasp cup",
        "recovery_bindings": [
            {
                "edge_id": "e01_grasp",
                "failure_name": "cup_moved",
                "monitors": [
                    {"type": "object_moved", "objects": ["cup"], "threshold": 0.02}
                ],
                "recovery": [
                    {
                        "type": "regrasp",
                        "robot_name": "left_arm",
                        "obj_name": "cup",
                        "pre_grasp_dis": 0.1,
                        "force_valid": False,
                    }
                ],
                "merge": "target",
                "repeat_until_success": True,
            }
        ],
    }


def test_expand_recovery_spec_accepts_canonicalized_direct_monitor() -> None:
    recovery_graph = expand_recovery_spec(
        _nominal_task_graph(),
        {
            "task": "grasp cup",
            "recovery_bindings": [
                {
                    "edge_id": "e01_grasp",
                    "failure_name": "custom_monitor",
                    "monitors": [{"fn": "monitor", "kwargs": {"value": 1}}],
                    "recovery": [{"type": "retry_failed_edge"}],
                    "merge": "target",
                    "repeat_until_success": False,
                }
            ],
        },
    )

    assert recovery_graph["recovery_branches"] == [
        {
            "edge_id": "e01_grasp",
            "failure_name": "custom_monitor",
            "monitor_sequence": [{"fn": "monitor", "kwargs": {"value": 1}}],
            "recovery_edges": ["re_e01_grasp_1_retry_e01_grasp"],
        }
    ]


def test_expand_recovery_spec_rejects_compiled_recovery_graph_input() -> None:
    with pytest.raises(ValueError, match="recovery_bindings"):
        expand_recovery_spec(
            _nominal_task_graph(),
            {
                "task": "grasp cup",
                "recovery_nodes": [],
                "recovery_edges": [],
                "recovery_branches": [],
            },
        )


def test_expand_recovery_spec_uses_step_specific_recovery_monitors() -> None:
    recovery_graph = expand_recovery_spec(
        _nominal_task_graph(),
        {
            "task": "grasp cup",
            "recovery_bindings": [
                {
                    "edge_id": "e12_move",
                    "failure_name": "cup_hold_lost",
                    "monitors": [
                        {
                            "type": "hold_lost",
                            "robot_name": "left_arm",
                            "obj_name": "cup",
                            "threshold": 0.05,
                        }
                    ],
                    "recovery": [
                        {
                            "type": "regrasp",
                            "robot_name": "left_arm",
                            "obj_name": "cup",
                        },
                        {"type": "retry_failed_edge"},
                    ],
                    "merge": "target",
                    "repeat_until_success": True,
                }
            ],
        },
    )

    branches = {
        branch["edge_id"]: branch for branch in recovery_graph["recovery_branches"]
    }

    assert branches["re_e12_move_1_regrasp_cup"]["monitor_sequence"] == [
        {
            "fn": "monitor_object_moved",
            "kwargs": {"obj_name": "cup", "threshold": 0.02},
        }
    ]
    assert branches["re_e12_move_2_retry_e12_move"]["monitor_sequence"] == [
        {
            "fn": "monitor_object_held",
            "kwargs": {
                "robot_name": "left_arm",
                "obj_name": "cup",
                "threshold": 0.05,
            },
        }
    ]
    assert branches["re_e12_move_2_retry_e12_move_repair_regrasp_cup"][
        "monitor_sequence"
    ] == [
        {
            "fn": "monitor_object_moved",
            "kwargs": {"obj_name": "cup", "threshold": 0.02},
        }
    ]


def test_compile_agent_graph_spec_rejects_authoring_recovery_spec() -> None:
    with pytest.raises(ValueError, match="expand_recovery_spec"):
        compile_agent_graph_spec(
            _nominal_task_graph(),
            {"task": "grasp cup", "recovery_bindings": []},
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={},
        )


def test_compile_agent_graph_spec_adds_explicit_recovery_branch() -> None:
    env = object()

    def grasp(**kwargs):
        return kwargs

    def move(**kwargs):
        return kwargs

    def monitor_object_moved(**kwargs):
        return kwargs

    recovery_graph = {
        "task": "grasp cup",
        "recovery_nodes": [],
        "recovery_edges": [
            {
                "id": "r_e01_regrasp",
                "source": "v0_start",
                "target": "v1_grasped",
                "left_arm_action": {
                    "fn": "grasp",
                    "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                },
                "right_arm_action": None,
            }
        ],
        "recovery_branches": [
            {
                "edge_id": "e01_grasp",
                "failure_name": "cup_moved",
                "monitor_sequence": [
                    {
                        "fn": "monitor_object_moved",
                        "kwargs": {"obj_name": "cup", "threshold": 0.02},
                    }
                ],
                "recovery_edges": ["r_e01_regrasp"],
            }
        ],
    }

    graph = compile_agent_graph_spec(
        _nominal_task_graph(),
        recovery_graph,
        env=env,
        graph_cls=FakeGraph,
        action_module={"grasp": grasp, "move": move},
        monitor_module={"monitor_object_moved": monitor_object_moved},
    )

    nominal_edge = graph.edges["e01_grasp"]
    recovery_edge = graph.edges["r_e01_regrasp"]

    assert graph.start == "v0_start"
    assert graph.goal == "v2_done"
    assert nominal_edge.left_arm_action.keywords == {
        "robot_name": "left_arm",
        "obj_name": "cup",
    }
    assert nominal_edge.monitor_labels == ["cup_moved"]
    assert nominal_edge.monitor_sequences[0][0].func is monitor_object_moved
    assert nominal_edge.monitor_sequences[0][0].keywords == {
        "obj_name": "cup",
        "threshold": 0.02,
        "env": env,
    }
    assert recovery_edge.source == "v0_start"
    assert recovery_edge.target == "v1_grasped"
    assert recovery_edge.left_arm_action.func is grasp
    assert recovery_edge.is_recovery is True
    assert graph.recovery_branches[("e01_grasp", 0)] == ["r_e01_regrasp"]


def test_compile_agent_graph_spec_allows_recovery_branch_on_recovery_edge() -> None:
    env = object()

    def grasp(**kwargs):
        return kwargs

    def move(**kwargs):
        return kwargs

    def monitor_object_held(**kwargs):
        return kwargs

    recovery_graph = {
        "task": "grasp cup",
        "recovery_nodes": [],
        "recovery_edges": [
            {
                "id": "r_e01_regrasp",
                "source": "v0_start",
                "target": "v1_grasped",
                "left_arm_action": {
                    "fn": "grasp",
                    "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                },
                "right_arm_action": None,
            },
            {
                "id": "r_e01_fix_regrasp",
                "source": "v0_start",
                "target": "v1_grasped",
                "left_arm_action": {
                    "fn": "grasp",
                    "kwargs": {
                        "robot_name": "left_arm",
                        "obj_name": "cup",
                        "force_valid": True,
                    },
                },
                "right_arm_action": None,
            },
        ],
        "recovery_branches": [
            {
                "edge_id": "e01_grasp",
                "failure_name": "cup_moved",
                "monitor_sequence": [
                    {
                        "fn": "monitor_object_held",
                        "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                    }
                ],
                "recovery_edges": ["r_e01_regrasp"],
            },
            {
                "edge_id": "r_e01_regrasp",
                "failure_name": "cup_lost_during_regrasp",
                "monitor_sequence": [
                    {
                        "fn": "monitor_object_held",
                        "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                    }
                ],
                "recovery_edges": ["r_e01_fix_regrasp"],
            },
        ],
    }

    graph = compile_agent_graph_spec(
        _nominal_task_graph(),
        recovery_graph,
        env=env,
        graph_cls=FakeGraph,
        action_module={"grasp": grasp, "move": move},
        monitor_module={"monitor_object_held": monitor_object_held},
    )

    recovery_edge = graph.edges["r_e01_regrasp"]

    assert recovery_edge.monitor_labels == ["cup_lost_during_regrasp"]
    assert recovery_edge.monitor_sequences[0][0].func is monitor_object_held
    assert recovery_edge.monitor_sequences[0][0].keywords == {
        "robot_name": "left_arm",
        "obj_name": "cup",
        "env": env,
    }
    assert graph.recovery_branches[("r_e01_regrasp", 0)] == ["r_e01_fix_regrasp"]


def test_compile_agent_graph_spec_allows_self_recovering_recovery_edge() -> None:
    env = object()

    def grasp(**kwargs):
        return kwargs

    def move(**kwargs):
        return kwargs

    def monitor_object_held(**kwargs):
        return kwargs

    recovery_graph = {
        "task": "grasp cup",
        "recovery_nodes": [],
        "recovery_edges": [
            {
                "id": "r_e01_regrasp",
                "source": "v0_start",
                "target": "v1_grasped",
                "left_arm_action": {
                    "fn": "grasp",
                    "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                },
                "right_arm_action": None,
            }
        ],
        "recovery_branches": [
            {
                "edge_id": "e01_grasp",
                "failure_name": "cup_moved",
                "monitor_sequence": [
                    {
                        "fn": "monitor_object_held",
                        "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                    }
                ],
                "recovery_edges": ["r_e01_regrasp"],
            },
            {
                "edge_id": "r_e01_regrasp",
                "failure_name": "cup_lost_during_regrasp",
                "monitor_sequence": [
                    {
                        "fn": "monitor_object_held",
                        "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                    }
                ],
                "recovery_edges": ["r_e01_regrasp"],
            },
        ],
    }

    graph = compile_agent_graph_spec(
        _nominal_task_graph(),
        recovery_graph,
        env=env,
        graph_cls=FakeGraph,
        action_module={"grasp": grasp, "move": move},
        monitor_module={"monitor_object_held": monitor_object_held},
    )

    recovery_edge = graph.edges["r_e01_regrasp"]

    assert recovery_edge.monitor_labels == ["cup_lost_during_regrasp"]
    assert recovery_edge.monitor_sequences[0][0].func is monitor_object_held
    assert graph.recovery_branches[("r_e01_regrasp", 0)] == ["r_e01_regrasp"]


def test_compile_agent_graph_spec_supports_intermediate_recovery_nodes() -> None:
    def grasp(**kwargs):
        return kwargs

    def move(**kwargs):
        return kwargs

    graph = compile_agent_graph_spec(
        _nominal_task_graph(),
        {
            "task": "grasp cup",
            "recovery_nodes": {
                "r_e12_regrasped": {"semantic": "Cup has been re-grasped."}
            },
            "recovery_edges": [
                {
                    "id": "r_e12_regrasp",
                    "source": "v1_grasped",
                    "target": "r_e12_regrasped",
                    "left_arm_action": {
                        "fn": "grasp",
                        "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                    },
                    "right_arm_action": None,
                },
                {
                    "id": "r_e12_move",
                    "source": "r_e12_regrasped",
                    "target": "v2_done",
                    "left_arm_action": {
                        "fn": "move",
                        "kwargs": {"robot_name": "left_arm"},
                    },
                    "right_arm_action": None,
                },
            ],
            "recovery_branches": [
                {
                    "edge_id": "e12_move",
                    "failure_name": "cup_not_held",
                    "monitor_sequence": [{"fn": "monitor", "kwargs": {}}],
                    "recovery_edges": ["r_e12_regrasp", "r_e12_move"],
                }
            ],
        },
        graph_cls=FakeGraph,
        action_module={"grasp": grasp, "move": move},
        monitor_module={"monitor": lambda **kwargs: kwargs},
    )

    assert graph.nodes["r_e12_regrasped"] == "Cup has been re-grasped."
    assert graph.edges["r_e12_regrasp"].target == "r_e12_regrasped"
    assert graph.edges["r_e12_move"].target == "v2_done"
    assert graph.recovery_branches[("e12_move", 0)] == [
        "r_e12_regrasp",
        "r_e12_move",
    ]


def test_compile_agent_graph_spec_allows_retry_by_merging_to_source() -> None:
    def grasp(**kwargs):
        return kwargs

    def move(**kwargs):
        return kwargs

    graph = compile_agent_graph_spec(
        _nominal_task_graph(),
        {
            "task": "grasp cup",
            "recovery_nodes": [],
            "recovery_edges": [
                {
                    "id": "r_e01_restore_precondition",
                    "source": "v0_start",
                    "target": "v0_start",
                    "left_arm_action": None,
                    "right_arm_action": {"fn": "move", "kwargs": {}},
                }
            ],
            "recovery_branches": [
                {
                    "edge_id": "e01_grasp",
                    "failure_name": "retry_grasp",
                    "monitor_sequence": [{"fn": "monitor", "kwargs": {}}],
                    "recovery_edges": ["r_e01_restore_precondition"],
                }
            ],
        },
        graph_cls=FakeGraph,
        action_module={"grasp": grasp, "move": move},
        monitor_module={"monitor": lambda **kwargs: kwargs},
    )

    assert graph.edges["e01_grasp"].monitor_labels == ["retry_grasp"]
    assert graph.edges["r_e01_restore_precondition"].target == "v0_start"
    assert graph.recovery_branches[("e01_grasp", 0)] == ["r_e01_restore_precondition"]


def test_compile_agent_graph_spec_rejects_recovery_without_monitor() -> None:
    with pytest.raises(ValueError, match="monitor_sequence"):
        compile_agent_graph_spec(
            _nominal_task_graph(),
            {
                "recovery_nodes": [],
                "recovery_edges": [
                    {
                        "id": "r_e01_regrasp",
                        "source": "v0_start",
                        "target": "v1_grasped",
                        "left_arm_action": None,
                        "right_arm_action": {"fn": "move", "kwargs": {}},
                    }
                ],
                "recovery_branches": [
                    {
                        "edge_id": "e01_grasp",
                        "failure_name": "invalid",
                        "recovery_edges": ["r_e01_regrasp"],
                    }
                ],
            },
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={},
        )


def test_compile_agent_graph_spec_rejects_unknown_recovery_edge() -> None:
    with pytest.raises(ValueError, match="unknown recovery edge"):
        compile_agent_graph_spec(
            _nominal_task_graph(),
            {
                "recovery_nodes": [],
                "recovery_edges": [],
                "recovery_branches": [
                    {
                        "edge_id": "e01_grasp",
                        "failure_name": "invalid",
                        "monitor_sequence": [{"fn": "monitor", "kwargs": {}}],
                        "recovery_edges": ["r_missing"],
                    }
                ],
            },
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={"monitor": lambda **kwargs: kwargs},
        )


def test_compile_agent_graph_spec_rejects_non_contiguous_recovery_path() -> None:
    with pytest.raises(ValueError, match="path-contiguous"):
        compile_agent_graph_spec(
            _nominal_task_graph(),
            {
                "recovery_nodes": [{"id": "r_mid", "semantic": "Intermediate state."}],
                "recovery_edges": [
                    {
                        "id": "r_first",
                        "source": "v0_start",
                        "target": "r_mid",
                        "left_arm_action": None,
                        "right_arm_action": {"fn": "move", "kwargs": {}},
                    },
                    {
                        "id": "r_second",
                        "source": "v1_grasped",
                        "target": "v1_grasped",
                        "left_arm_action": None,
                        "right_arm_action": {"fn": "move", "kwargs": {}},
                    },
                ],
                "recovery_branches": [
                    {
                        "edge_id": "e01_grasp",
                        "failure_name": "invalid",
                        "monitor_sequence": [{"fn": "monitor", "kwargs": {}}],
                        "recovery_edges": ["r_first", "r_second"],
                    }
                ],
            },
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={"monitor": lambda **kwargs: kwargs},
        )


def test_compile_agent_graph_spec_rejects_recovery_path_that_ends_off_failed_edge() -> (
    None
):
    with pytest.raises(ValueError, match="source or target node"):
        compile_agent_graph_spec(
            _nominal_task_graph(),
            {
                "recovery_nodes": [{"id": "r_mid", "semantic": "Intermediate state."}],
                "recovery_edges": [
                    {
                        "id": "r_first",
                        "source": "v0_start",
                        "target": "r_mid",
                        "left_arm_action": None,
                        "right_arm_action": {"fn": "move", "kwargs": {}},
                    }
                ],
                "recovery_branches": [
                    {
                        "edge_id": "e01_grasp",
                        "failure_name": "invalid",
                        "monitor_sequence": [{"fn": "monitor", "kwargs": {}}],
                        "recovery_edges": ["r_first"],
                    }
                ],
            },
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={"monitor": lambda **kwargs: kwargs},
        )


def test_compile_agent_graph_spec_rejects_unrelated_nominal_merge_node() -> None:
    with pytest.raises(ValueError, match="source or target node"):
        compile_agent_graph_spec(
            _nominal_task_graph(),
            {
                "recovery_nodes": [],
                "recovery_edges": [
                    {
                        "id": "r_skip",
                        "source": "v0_start",
                        "target": "v2_done",
                        "left_arm_action": None,
                        "right_arm_action": {"fn": "move", "kwargs": {}},
                    }
                ],
                "recovery_branches": [
                    {
                        "edge_id": "e01_grasp",
                        "failure_name": "invalid",
                        "monitor_sequence": [{"fn": "monitor", "kwargs": {}}],
                        "recovery_edges": ["r_skip"],
                    }
                ],
            },
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={"monitor": lambda **kwargs: kwargs},
        )


def test_compile_agent_graph_spec_rejects_recovery_edge_without_action() -> None:
    with pytest.raises(ValueError, match="must define an arm action"):
        compile_agent_graph_spec(
            _nominal_task_graph(),
            {
                "recovery_nodes": [],
                "recovery_edges": [
                    {
                        "id": "r_noop",
                        "source": "v0_start",
                        "target": "v1_grasped",
                        "left_arm_action": None,
                        "right_arm_action": None,
                    }
                ],
                "recovery_branches": [],
            },
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={},
        )


def test_compile_agent_graph_spec_rejects_branched_nominal_graph() -> None:
    task_graph = _nominal_task_graph()
    task_graph["nodes"].append({"id": "v_alt", "semantic": "Alternative state."})
    task_graph["edges"].append(
        {
            "id": "e0_alt",
            "source": "v0_start",
            "target": "v_alt",
            "left_arm_action": None,
            "right_arm_action": {"fn": "move", "kwargs": {}},
        }
    )

    with pytest.raises(ValueError, match="multiple outgoing"):
        compile_agent_graph_spec(
            task_graph,
            {},
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={},
        )


def test_compile_agent_graph_spec_rejects_legacy_recoveries_schema() -> None:
    with pytest.raises(ValueError, match="recoveries"):
        compile_agent_graph_spec(
            _nominal_task_graph(),
            {"recoveries": []},
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={},
        )


def test_compile_agent_graph_spec_rejects_branch_error_functions() -> None:
    with pytest.raises(ValueError, match="error_functions"):
        compile_agent_graph_spec(
            _nominal_task_graph(),
            {
                "recovery_nodes": [],
                "recovery_edges": [],
                "recovery_branches": [
                    {
                        "edge_id": "e01_grasp",
                        "failure_name": "invalid",
                        "error_functions": [],
                        "monitor_sequence": [{"fn": "monitor", "kwargs": {}}],
                        "recovery_edges": [],
                    }
                ],
            },
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={"monitor": lambda **kwargs: kwargs},
        )


def test_compile_agent_graph_spec_rejects_legacy_branch_actions_and_merge() -> None:
    with pytest.raises(ValueError, match="recovery_actions"):
        compile_agent_graph_spec(
            _nominal_task_graph(),
            {
                "recovery_nodes": [],
                "recovery_edges": [],
                "recovery_branches": [
                    {
                        "edge_id": "e01_grasp",
                        "failure_name": "invalid",
                        "monitor_sequence": [{"fn": "monitor", "kwargs": {}}],
                        "recovery_actions": [],
                        "merge": "target",
                    }
                ],
            },
            graph_cls=FakeGraph,
            action_module={
                "grasp": lambda **kwargs: kwargs,
                "move": lambda **kwargs: kwargs,
            },
            monitor_module={"monitor": lambda **kwargs: kwargs},
        )
