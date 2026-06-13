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

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.runtime.atom_actions import (
    execute_parallel_atomic_actions,
)

__all__ = [
    "AgentGraphEdge",
    "AgentGraphNode",
    "AgentTaskGraph",
    "ExecutedActionList",
]


@dataclass
class AgentGraphNode:
    """Semantic keyframe in an atomic-action task graph."""

    id: str
    semantic: str = ""


@dataclass
class AgentGraphEdge:
    """Executable transition between two graph nodes."""

    id: str
    source: str
    target: str
    left_arm_action: Any = None
    right_arm_action: Any = None


class ExecutedActionList:
    """Action sequence already executed online by the graph runtime."""

    already_executed = True

    def __init__(self, actions: list[Any]) -> None:
        self.actions = actions

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __getitem__(self, index):
        return self.actions[index]


class AgentTaskGraph:
    """Deterministic atomic-action graph with one nominal start-to-goal path."""

    def __init__(self, start: str, goal: str, max_transitions: int = 1000) -> None:
        self.start = start
        self.goal = goal
        self.max_transitions = max_transitions
        self.nodes: dict[str, AgentGraphNode] = {}
        self.edges: dict[str, AgentGraphEdge] = {}
        self.outgoing: dict[str, list[str]] = defaultdict(list)

    def add_node(self, node_id: str, semantic: str = "") -> "AgentTaskGraph":
        self.nodes[node_id] = AgentGraphNode(node_id, semantic)
        return self

    def add_edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        *,
        left_arm_action=None,
        right_arm_action=None,
    ) -> "AgentTaskGraph":
        self.edges[edge_id] = AgentGraphEdge(
            id=edge_id,
            source=source,
            target=target,
            left_arm_action=left_arm_action,
            right_arm_action=right_arm_action,
        )
        self.outgoing[source].append(edge_id)
        return self

    def run(self, env=None, **kwargs) -> ExecutedActionList:
        current = self.start
        executed_actions: list[Any] = []
        transitions = 0

        while current != self.goal:
            transitions += 1
            if transitions > self.max_transitions:
                raise RuntimeError("Agent task graph exceeded max_transitions.")

            edge = self.edges[self._next_edge(current)]
            actions = execute_parallel_atomic_actions(
                left_arm_action=edge.left_arm_action,
                right_arm_action=edge.right_arm_action,
                env=env,
                **kwargs,
            )
            executed_actions.extend(actions)
            current = edge.target

        return ExecutedActionList(executed_actions)

    def _next_edge(self, node_id: str) -> str:
        outgoing_edges = self.outgoing[node_id]
        if len(outgoing_edges) != 1:
            raise RuntimeError(
                f"Nominal node '{node_id}' must have exactly one outgoing edge."
            )
        return outgoing_edges[0]
