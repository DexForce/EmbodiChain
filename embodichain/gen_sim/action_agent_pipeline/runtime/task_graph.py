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
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from embodichain.gen_sim.action_agent_pipeline.config.defaults import defaults_section
from embodichain.gen_sim.action_agent_pipeline.runtime.atom_actions import (
    execute_parallel_atomic_actions,
    init_parallel_world_states,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.success_evaluator import (
    evaluate_configured_success,
)
from embodichain.utils.logger import log_info, log_warning

__all__ = [
    "AgentGraphEdge",
    "AgentGraphNode",
    "AgentSemanticStep",
    "AgentTaskGraph",
    "ExecutedActionList",
]

_CLOSED_LOOP_DEFAULTS = defaults_section("closed_loop")
_SEMANTIC_STEP_SETTLE_STEPS = int(_CLOSED_LOOP_DEFAULTS["semantic_step_settle_steps"])


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


@dataclass(frozen=True)
class AgentSemanticStep:
    """One high-level operation bound to a contiguous atomic edge range."""

    id: str
    operator: str
    object_uid: str
    actor: dict[str, Any]
    goal: dict[str, Any]
    depends_on: tuple[str, ...]
    postcondition: dict[str, Any]
    edge_ids: tuple[str, ...]


class ExecutedActionList(Sequence[Any]):
    """Action sequence already executed online by the graph runtime."""

    already_executed = True

    def __init__(self, actions: list[Any]) -> None:
        self.actions = actions
        self.semantic_step_success: dict[str, torch.Tensor] = {}

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
        self.semantic_steps: dict[str, AgentSemanticStep] = {}
        self.semantic_step_by_edge: dict[str, AgentSemanticStep] = {}

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

    def add_semantic_step(
        self,
        step_id: str,
        *,
        operator: str,
        object_uid: str,
        actor: Mapping[str, Any],
        goal: Mapping[str, Any],
        depends_on: Sequence[str],
        postcondition: Mapping[str, Any],
        edge_ids: Sequence[str],
    ) -> "AgentTaskGraph":
        """Register closed-loop metadata without changing atomic graph edges."""
        step = AgentSemanticStep(
            id=step_id,
            operator=operator,
            object_uid=object_uid,
            actor=deepcopy(dict(actor)),
            goal=deepcopy(dict(goal)),
            depends_on=tuple(str(item) for item in depends_on),
            postcondition=deepcopy(dict(postcondition)),
            edge_ids=tuple(str(item) for item in edge_ids),
        )
        self.semantic_steps[step_id] = step
        for edge_id in step.edge_ids:
            self.semantic_step_by_edge[edge_id] = step
        return self

    def run(self, *, env, **kwargs) -> ExecutedActionList:
        if env is None:
            raise ValueError("env is required to run an agent task graph.")
        current = self.start
        executed_actions: list[Any] = []
        transitions = 0
        world_states = init_parallel_world_states(env)
        failed_env_mask = None
        semantic_step_success: dict[str, torch.Tensor] = {}
        started_semantic_steps: set[str] = set()

        while current != self.goal:
            transitions += 1
            if transitions > self.max_transitions:
                raise RuntimeError("Agent task graph exceeded max_transitions.")

            edge = self.edges[self._next_edge(current)]
            semantic_step = self.semantic_step_by_edge.get(edge.id)
            if (
                semantic_step is not None
                and semantic_step.id not in started_semantic_steps
            ):
                failed_env_mask = self._begin_semantic_step(
                    semantic_step,
                    env=env,
                    failed_env_mask=failed_env_mask,
                    semantic_step_success=semantic_step_success,
                )
                started_semantic_steps.add(semantic_step.id)
            log_info(
                f"Executing task graph edge {transitions}/{len(self.edges)}: "
                f"{edge.id} ({edge.source} -> {edge.target}); "
                f"left={self._action_log_label(edge.left_arm_action)}, "
                f"right={self._action_log_label(edge.right_arm_action)}."
            )
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
            # Failure is monotonic across graph edges: an environment that
            # failed once receives hold commands for every remaining edge.
            failed_env_mask = result["failed_env_mask"]
            failed_count = (
                int(failed_env_mask.sum().item()) if failed_env_mask is not None else 0
            )
            log_info(
                f"Completed task graph edge {edge.id}: "
                f"action_steps={len(actions)}, failed_envs={failed_count}."
            )
            executed_actions.extend(actions)
            current = edge.target
            if semantic_step is not None and edge.id == semantic_step.edge_ids[-1]:
                failed_env_mask, step_success = self._complete_semantic_step(
                    semantic_step,
                    env=env,
                    failed_env_mask=failed_env_mask,
                    settle_steps=int(
                        kwargs.get(
                            "semantic_step_settle_steps",
                            _SEMANTIC_STEP_SETTLE_STEPS,
                        )
                    ),
                )
                semantic_step_success[semantic_step.id] = step_success

        result = ExecutedActionList(executed_actions)
        result.semantic_step_success = semantic_step_success
        return result

    def _begin_semantic_step(
        self,
        step: AgentSemanticStep,
        *,
        env: Any,
        failed_env_mask: torch.Tensor | None,
        semantic_step_success: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Validate dependencies and live entity bindings before planning."""
        failed = self._normalized_failure_mask(env, failed_env_mask)
        dependencies_satisfied = torch.ones_like(failed, dtype=torch.bool)
        for dependency in step.depends_on:
            dependency_success = semantic_step_success.get(dependency)
            if dependency_success is None:
                raise RuntimeError(
                    f"Semantic step {step.id!r} started before dependency "
                    f"{dependency!r} completed."
                )
            dependencies_satisfied &= dependency_success.to(
                device=failed.device,
                dtype=torch.bool,
            )
        unmet = ~failed & ~dependencies_satisfied
        if bool(unmet.any()):
            log_warning(
                f"Semantic step {step.id} has unsatisfied dependencies in "
                f"{int(unmet.sum().item())} environment(s)."
            )
            failed |= unmet
        self._validate_live_step_entities(env, step)
        log_info(
            f"Grounding semantic step {step.id} from live scene state: "
            f"operator={step.operator}, object={step.object_uid}, "
            f"actor={step.actor.get('arm')}, goal={step.goal}."
        )
        return failed

    @staticmethod
    def _validate_live_step_entities(env: Any, step: AgentSemanticStep) -> None:
        """Fail early when a symbolic live-scene binding cannot be resolved."""
        required_uids = [step.object_uid]
        reference_uid = step.goal.get("reference_object")
        if step.goal.get("reference_state") == "live" and isinstance(
            reference_uid, str
        ):
            required_uids.append(reference_uid)
        for uid in required_uids:
            entity = env.sim.get_rigid_object(uid)
            if entity is None:
                raise ValueError(
                    f"Semantic step {step.id!r} references unknown live object "
                    f"{uid!r}."
                )
            # Reading the pose here establishes the step snapshot boundary.
            # Atomic target resolvers read it again immediately before planning.
            entity.get_local_pose(to_matrix=True)

    def _complete_semantic_step(
        self,
        step: AgentSemanticStep,
        *,
        env: Any,
        failed_env_mask: torch.Tensor | None,
        settle_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Settle physics and verify the step postcondition per environment."""
        failed = self._normalized_failure_mask(env, failed_env_mask)
        if settle_steps < 0:
            raise ValueError("semantic_step_settle_steps must be non-negative.")
        if settle_steps and bool((~failed).any()):
            env.sim.update(step=settle_steps)
        observed_success = evaluate_configured_success(env, step.postcondition).to(
            device=failed.device,
            dtype=torch.bool,
        )
        active = ~failed
        step_success = active & observed_success
        postcondition_failed = active & ~observed_success
        failed |= postcondition_failed
        log_info(
            f"Verified semantic step {step.id}: "
            f"succeeded={int(step_success.sum().item())}/{len(step_success)}, "
            f"postcondition_failed={int(postcondition_failed.sum().item())}."
        )
        return failed, step_success

    @staticmethod
    def _normalized_failure_mask(
        env: Any,
        failed_env_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if failed_env_mask is not None:
            return failed_env_mask.to(device=env.device, dtype=torch.bool).clone()
        return torch.zeros(
            (int(env.num_envs),),
            dtype=torch.bool,
            device=env.device,
        )

    def _pickup_downstream_targets(
        self, edge: AgentGraphEdge
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        """Collect future object targets needed to choose a feasible pickup grasp."""
        targets: dict[str, tuple[dict[str, Any], ...]] = {}
        for action in (edge.left_arm_action, edge.right_arm_action):
            action = self._action_mapping(action)
            if action is None or action.get("atomic_action_class") != "PickUp":
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
            if action is not None:
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
            action_mapping = AgentTaskGraph._action_mapping(action)
            if (
                action_mapping is not None
                and action_mapping.get("robot_name") == robot_name
            ):
                return action_mapping
        return None

    @staticmethod
    def _action_mapping(action: Any) -> Mapping[str, Any] | None:
        if isinstance(action, Mapping):
            return action
        to_dict = getattr(action, "to_dict", None)
        if not callable(to_dict):
            return None
        value = to_dict()
        return value if isinstance(value, Mapping) else None

    @classmethod
    def _action_log_label(cls, action: Any) -> str:
        """Return a compact action label without serializing the full spec."""
        action_mapping = cls._action_mapping(action)
        if action_mapping is None:
            return "null"
        action_class = str(action_mapping.get("atomic_action_class", "unknown"))
        robot_name = str(action_mapping.get("robot_name", "unknown"))
        return f"{action_class}({robot_name})"

    def _next_edge(self, node_id: str) -> str:
        outgoing_edges = self.outgoing[node_id]
        if len(outgoing_edges) != 1:
            raise RuntimeError(
                f"Nominal node '{node_id}' must have exactly one outgoing edge."
            )
        return outgoing_edges[0]
