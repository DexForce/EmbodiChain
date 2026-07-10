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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.runtime.atom_actions import (
    execute_parallel_atomic_actions,
    init_parallel_world_states,
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


class ExecutedActionList(Sequence[Any]):
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

    def run(self, *, env, **kwargs) -> ExecutedActionList:
        if env is None:
            raise ValueError("env is required to run an agent task graph.")
        current = self.start
        executed_actions: list[Any] = []
        transitions = 0
        world_states = init_parallel_world_states(env)
        failed_env_mask = None

        while current != self.goal:
            transitions += 1
            if transitions > self.max_transitions:
                raise RuntimeError("Agent task graph exceeded max_transitions.")

            edge = self.edges[self._next_edge(current)]
            result = execute_parallel_atomic_actions(
                left_arm_action=edge.left_arm_action,
                right_arm_action=edge.right_arm_action,
                env=env,
                world_states=world_states,
                failed_env_mask=failed_env_mask,
                return_result=True,
                pickup_downstream_object_target_specs=self._pickup_downstream_targets(
                    edge
                ),
                **kwargs,
            )
            actions = result["actions"]
            world_states = result["world_states"]
            failed_env_mask = result["failed_env_mask"]
            executed_actions.extend(actions)
            current = edge.target

        return ExecutedActionList(executed_actions)

    def _pickup_downstream_targets(
        self, edge: AgentGraphEdge
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        """Collect future object targets needed to choose a feasible pickup grasp."""
        targets: dict[str, tuple[dict[str, Any], ...]] = {}
        for action in (edge.left_arm_action, edge.right_arm_action):
            if (
                not isinstance(action, Mapping)
                or action.get("atomic_action_class") != "PickUp"
            ):
                continue
            robot_name = action.get("robot_name")
            if not isinstance(robot_name, str):
                continue
            targets[robot_name] = self._future_move_held_object_targets(
                edge.target, robot_name
            )
        return targets

    def _future_move_held_object_targets(
        self, node_id: str, robot_name: str
    ) -> tuple[dict[str, Any], ...]:
        """Return the held-object targets before this arm next releases or regraspes."""
        targets: list[dict[str, Any]] = []
        while node_id != self.goal:
            edge = self.edges[self._next_edge(node_id)]
            action = self._action_for_robot(edge, robot_name)
            if isinstance(action, Mapping):
                action_class = action.get("atomic_action_class")
                if action_class in {"MoveHeldObject", "Place"}:
                    target = action.get("target_object_pose")
                    # Relative targets depend on the runtime EEF pose after the
                    # preceding action and cannot be screened during PickUp.
                    if isinstance(target, Mapping) and target.get(
                        "reference", "object"
                    ) in {"object", "absolute"}:
                        targets.append(dict(target))
                    if action_class == "Place":
                        break
                elif action_class == "PickUp":
                    break
            node_id = edge.target
        return tuple(targets)

    @staticmethod
    def _action_for_robot(edge: AgentGraphEdge, robot_name: str) -> Any:
        for action in (edge.left_arm_action, edge.right_arm_action):
            if isinstance(action, Mapping) and action.get("robot_name") == robot_name:
                return action
        return None

    def _next_edge(self, node_id: str) -> str:
        outgoing_edges = self.outgoing[node_id]
        if len(outgoing_edges) != 1:
            raise RuntimeError(
                f"Nominal node '{node_id}' must have exactly one outgoing edge."
            )
        return outgoing_edges[0]
