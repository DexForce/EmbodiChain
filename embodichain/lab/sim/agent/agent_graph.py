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
from dataclasses import dataclass, field
from typing import Any

from embodichain.lab.sim.agent.atom_actions import drive

__all__ = [
    "AgentGraphEdge",
    "AgentGraphNode",
    "AgentRecoveryBranch",
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
    monitor_sequences: list[list[Any]] | None = None
    monitor_labels: list[str] = field(default_factory=list)
    is_recovery: bool = False


@dataclass
class AgentRecoveryBranch:
    """Mapping from one edge monitor trigger to recovery edge transitions."""

    edge_id: str
    monitor_index: int
    recovery_edges: list[str]


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
    """Atomic-action graph with monitor-triggered recovery edge switching."""

    def __init__(self, start: str, goal: str, max_transitions: int = 1000) -> None:
        self.start = start
        self.goal = goal
        self.max_transitions = max_transitions
        self.nodes: dict[str, AgentGraphNode] = {}
        self.edges: dict[str, AgentGraphEdge] = {}
        self.outgoing: dict[str, list[str]] = defaultdict(list)
        self.recovery_branches: dict[tuple[str, int], AgentRecoveryBranch] = {}

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
        monitor_sequences=None,
        monitor_labels=None,
        is_recovery: bool = False,
    ) -> "AgentTaskGraph":
        self.edges[edge_id] = AgentGraphEdge(
            id=edge_id,
            source=source,
            target=target,
            left_arm_action=left_arm_action,
            right_arm_action=right_arm_action,
            monitor_sequences=monitor_sequences,
            monitor_labels=list(monitor_labels or []),
            is_recovery=is_recovery,
        )
        self.outgoing[source].append(edge_id)
        return self

    def add_recovery(
        self,
        edge_id: str,
        monitor_index: int,
        recovery_edges: list[str],
    ) -> "AgentTaskGraph":
        self.recovery_branches[(edge_id, monitor_index)] = AgentRecoveryBranch(
            edge_id=edge_id,
            monitor_index=monitor_index,
            recovery_edges=list(recovery_edges),
        )
        return self

    def run(self, env=None, **kwargs) -> ExecutedActionList:
        current = self.start
        pending_edges: list[str] = []
        continuation_stack: list[list[str]] = []
        executed_actions: list[Any] = []
        transitions = 0

        while current != self.goal or pending_edges or continuation_stack:
            transitions += 1
            if transitions > self.max_transitions:
                raise RuntimeError("Agent task graph exceeded max_transitions.")

            if not pending_edges and continuation_stack:
                pending_edges = continuation_stack.pop()

            edge_id = (
                pending_edges.pop(0) if pending_edges else self._next_edge(current)
            )
            edge = self.edges[edge_id]
            result = drive(
                left_arm_action=edge.left_arm_action,
                right_arm_action=edge.right_arm_action,
                monitor_sequences=edge.monitor_sequences,
                env=env,
                return_result=True,
                **kwargs,
            )
            executed_actions.extend(result["actions"])

            monitor_index = result["monitor_index"]
            if monitor_index is not None:
                branch = self.recovery_branches.get((edge.id, monitor_index))
                if branch is None:
                    raise RuntimeError(
                        f"No recovery branch for edge '{edge.id}' monitor {monitor_index}."
                    )
                branch_final_target = self.edges[branch.recovery_edges[-1]].target
                continuation_edges = list(pending_edges)
                if branch_final_target == edge.source and edge.source != edge.target:
                    continuation_edges = [edge.id, *continuation_edges]
                elif branch_final_target != edge.target:
                    raise RuntimeError(
                        f"Recovery branch for edge '{edge.id}' must merge to its source or target."
                    )

                if continuation_edges:
                    continuation_stack.append(continuation_edges)
                pending_edges = list(branch.recovery_edges)
                current = edge.source
                continue

            current = edge.target

        return ExecutedActionList(executed_actions)

    def _next_edge(self, node_id: str) -> str:
        for edge_id in self.outgoing[node_id]:
            if not self.edges[edge_id].is_recovery:
                return edge_id
        raise RuntimeError(f"No nominal outgoing edge from node '{node_id}'.")
