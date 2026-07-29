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

"""Execute Seed v2 one semantic step at a time against live environment state."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.config.defaults import defaults_section
from embodichain.gen_sim.action_agent_pipeline.runtime.action_execution import (
    _execute_atomic_action_result,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atom_actions import (
    execute_parallel_atomic_actions,
    init_parallel_world_states,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.symbolic_grounding import (
    ground_symbolic_action,
    select_auto_arm_from_candidates,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.task_graph_artifact import (
    RuntimeTaskGraphRecorder,
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
    """One symbolic state in the executable Seed graph."""

    id: str
    semantic: str = ""


@dataclass
class AgentGraphEdge:
    """One transition containing ungrounded symbolic actions."""

    id: str
    source: str
    target: str
    symbolic_actions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AgentSemanticStep:
    """One closed-loop operation bound to a contiguous symbolic edge range."""

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
        self.runtime_graph_output_dir: str | None = None

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __getitem__(self, index):
        return self.actions[index]


class AgentTaskGraph:
    """Runtime executor for one immutable executable Seed Graph v2."""

    def __init__(
        self,
        start: str,
        goal: str,
        max_transitions: int = 1000,
        *,
        seed_graph: Mapping[str, Any],
    ) -> None:
        self.start = start
        self.goal = goal
        self.max_transitions = max_transitions
        self.seed_graph = deepcopy(dict(seed_graph))
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
        symbolic_actions: Sequence[Mapping[str, Any]],
    ) -> "AgentTaskGraph":
        self.edges[edge_id] = AgentGraphEdge(
            id=edge_id,
            source=source,
            target=target,
            symbolic_actions=tuple(
                deepcopy(dict(action)) for action in symbolic_actions
            ),
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
        """Ground, execute, verify, and record every semantic step per env."""
        if env is None:
            raise ValueError("env is required to run an agent task graph.")
        recorder = RuntimeTaskGraphRecorder(
            self.seed_graph,
            env=env,
            run_id=kwargs.get("runtime_run_id"),
            episode_index=int(kwargs.get("episode_index", 0)),
        )
        current = self.start
        executed_actions: list[Any] = []
        world_states = init_parallel_world_states(env)
        failed = torch.zeros(int(env.num_envs), dtype=torch.bool, device=env.device)
        semantic_success: dict[str, torch.Tensor] = {}
        step_assignments: dict[str, list[str | None]] = {}
        step_selection_failures: dict[str, torch.Tensor] = {}
        step_target_positions: dict[str, torch.Tensor] = {}
        step_motion_policies: dict[str, dict[str, Any]] = {}
        transitions = 0
        aborted_reason = None
        try:
            while current != self.goal:
                transitions += 1
                if transitions > self.max_transitions:
                    raise RuntimeError("Agent task graph exceeded max_transitions.")
                edge = self.edges[self._next_edge(current)]
                step = self.semantic_step_by_edge[edge.id]
                if step.id not in step_assignments:
                    failed = self._check_dependencies(
                        step,
                        failed=failed,
                        semantic_success=semantic_success,
                    )
                    eligible_for_step = ~failed
                    assignments, selection_failed = self._select_step_arms(
                        step,
                        env=env,
                        world_states=world_states,
                        failed=failed,
                        runtime_kwargs=kwargs,
                    )
                    failed |= selection_failed
                    step_assignments[step.id] = assignments
                    step_selection_failures[step.id] = selection_failed
                    preview = ground_symbolic_action(
                        edge.symbolic_actions[0],
                        step,
                        env=env,
                        arm=_representative_arm(assignments, step.actor),
                    )
                    recorder.begin_step(
                        step,
                        assignments=assignments,
                        object_pose=preview.object_pose,
                        reference_pose=preview.reference_pose,
                        active_mask=eligible_for_step,
                        selection_failed_mask=selection_failed,
                    )
                    log_info(
                        f"Grounded semantic step {step.id}: operator={step.operator}, "
                        f"object={step.object_uid}, assignments={assignments}."
                    )

                assignments = step_assignments[step.id]
                failed_before = failed.clone()
                result, grounded_actions = self._execute_symbolic_edge(
                    edge,
                    step,
                    assignments=assignments,
                    env=env,
                    world_states=world_states,
                    failed=failed,
                    runtime_kwargs=kwargs,
                )
                world_states = result["world_states"]
                failed = result["failed_env_mask"]
                actions = result["actions"]
                executed_actions.extend(actions)
                grounded_target = next(
                    (
                        grounded.target_object_pose
                        for grounded in grounded_actions
                        if grounded.target_object_pose is not None
                    ),
                    None,
                )
                resolved_target_positions = _resolved_object_target_positions(
                    result["arm_actions"],
                    fallback=grounded_target,
                )
                if resolved_target_positions is not None:
                    step_target_positions[step.id] = resolved_target_positions
                    for grounded in grounded_actions:
                        if "postcondition_tolerance" in grounded.motion_policy:
                            step_motion_policies[step.id] = grounded.motion_policy
                            break
                recorder.record_edge(
                    edge.id,
                    assignments=assignments,
                    grounded_actions=grounded_actions,
                    failed_before=failed_before,
                    failed_after=failed,
                    grounding_failed=(
                        step_selection_failures[step.id]
                        if edge.id == step.edge_ids[0]
                        else torch.zeros_like(failed)
                    ),
                    action_steps=len(actions),
                    arm_actions=result["arm_actions"],
                )
                current = edge.target
                log_info(
                    f"Completed symbolic edge {edge.id}: action_steps={len(actions)}, "
                    f"failed_envs={int(failed.sum().item())}."
                )
                if edge.id == step.edge_ids[-1]:
                    (
                        failed,
                        success,
                        observed_positions,
                        position_error,
                        tolerance,
                    ) = self._complete_semantic_step(
                        step,
                        env=env,
                        failed=failed,
                        target_positions=step_target_positions.get(step.id),
                        motion_policy=step_motion_policies.get(step.id),
                        settle_steps=int(
                            kwargs.get(
                                "semantic_step_settle_steps",
                                _SEMANTIC_STEP_SETTLE_STEPS,
                            )
                        ),
                    )
                    semantic_success[step.id] = success
                    recorder.complete_step(
                        step.id,
                        success=success,
                        failed_mask=failed,
                        observed_positions=observed_positions,
                        target_positions=step_target_positions.get(step.id),
                        position_error=position_error,
                        tolerance=tolerance,
                    )
        except BaseException as error:
            aborted_reason = f"{type(error).__name__}: {error}"
            raise
        finally:
            recorder.finalize(failed, aborted_reason=aborted_reason)

        result = ExecutedActionList(executed_actions)
        result.semantic_step_success = semantic_success
        result.runtime_graph_output_dir = str(recorder.output_dir)
        return result

    def _execute_symbolic_edge(
        self,
        edge: AgentGraphEdge,
        step: AgentSemanticStep,
        *,
        assignments: Sequence[str | None],
        env: Any,
        world_states: Mapping[str, Any],
        failed: torch.Tensor,
        runtime_kwargs: Mapping[str, Any],
    ) -> tuple[dict[str, Any], tuple[Any, ...]]:
        if step.actor["mode"] == "coordinated":
            return self._execute_coordinated_edge(
                edge,
                step,
                env=env,
                world_states=world_states,
                failed=failed,
                runtime_kwargs=runtime_kwargs,
            )

        if len(edge.symbolic_actions) != 1:
            raise ValueError(
                f"Non-coordinated Seed edge {edge.id!r} must contain one action."
            )
        symbolic = edge.symbolic_actions[0]
        left_mask = (
            torch.tensor(
                [arm == "left_arm" for arm in assignments],
                dtype=torch.bool,
                device=env.device,
            )
            & ~failed
        )
        right_mask = (
            torch.tensor(
                [arm == "right_arm" for arm in assignments],
                dtype=torch.bool,
                device=env.device,
            )
            & ~failed
        )
        left_grounded = (
            ground_symbolic_action(symbolic, step, env=env, arm="left_arm")
            if bool(left_mask.any())
            else None
        )
        right_grounded = (
            ground_symbolic_action(symbolic, step, env=env, arm="right_arm")
            if bool(right_mask.any())
            else None
        )
        grounded = left_grounded or right_grounded
        if grounded is None:
            grounded = ground_symbolic_action(
                symbolic,
                step,
                env=env,
                arm=_representative_arm(assignments, step.actor),
            )
        execution_kwargs = _execution_kwargs(runtime_kwargs)
        if str(symbolic["atomic_action_class"]) == "PickUp":
            execution_kwargs["pickup_downstream_object_target_specs"] = (
                self._pickup_downstream_targets(
                    step,
                    env=env,
                    arms=("left_arm", "right_arm"),
                )
            )
        result = execute_parallel_atomic_actions(
            left_arm_action=(
                left_grounded.action_spec if left_grounded is not None else None
            ),
            right_arm_action=(
                right_grounded.action_spec if right_grounded is not None else None
            ),
            left_active_env_mask=left_mask,
            right_active_env_mask=right_mask,
            env=env,
            world_states=dict(world_states),
            failed_env_mask=failed,
            return_result=True,
            **execution_kwargs,
        )
        return result, (grounded,)

    def _execute_coordinated_edge(
        self,
        edge: AgentGraphEdge,
        step: AgentSemanticStep,
        *,
        env: Any,
        world_states: Mapping[str, Any],
        failed: torch.Tensor,
        runtime_kwargs: Mapping[str, Any],
    ) -> tuple[dict[str, Any], tuple[Any, ...]]:
        if len(edge.symbolic_actions) == 1:
            symbolic = edge.symbolic_actions[0]
            if symbolic["atomic_action_class"] != "CoordinatedPickment":
                raise ValueError(
                    f"Coordinated Seed edge {edge.id!r} has an invalid action."
                )
            grounded = ground_symbolic_action(
                symbolic,
                step,
                env=env,
                arm="left_arm",
            )
            result = execute_parallel_atomic_actions(
                left_arm_action=grounded.action_spec,
                right_arm_action=None,
                env=env,
                world_states=dict(world_states),
                failed_env_mask=failed,
                return_result=True,
                **_execution_kwargs(runtime_kwargs),
            )
            return result, (grounded,)

        if len(edge.symbolic_actions) != 2:
            raise ValueError(
                f"Coordinated Seed edge {edge.id!r} requires one dual action "
                "or one action per arm."
            )
        grounded_by_arm = {}
        for symbolic in edge.symbolic_actions:
            arm = symbolic["actor"].get("arm")
            if arm not in {"left_arm", "right_arm"} or arm in grounded_by_arm:
                raise ValueError(
                    f"Coordinated Seed edge {edge.id!r} must bind each arm once."
                )
            grounded_by_arm[arm] = ground_symbolic_action(
                symbolic,
                step,
                env=env,
                arm=arm,
            )
        result = execute_parallel_atomic_actions(
            left_arm_action=grounded_by_arm["left_arm"].action_spec,
            right_arm_action=grounded_by_arm["right_arm"].action_spec,
            left_active_env_mask=~failed,
            right_active_env_mask=~failed,
            env=env,
            world_states=dict(world_states),
            failed_env_mask=failed,
            return_result=True,
            **_execution_kwargs(runtime_kwargs),
        )
        return result, (
            grounded_by_arm["left_arm"],
            grounded_by_arm["right_arm"],
        )

    def _select_step_arms(
        self,
        step: AgentSemanticStep,
        *,
        env: Any,
        world_states: Mapping[str, Any],
        failed: torch.Tensor,
        runtime_kwargs: Mapping[str, Any],
    ) -> tuple[list[str | None], torch.Tensor]:
        mode = step.actor["mode"]
        if mode == "required":
            arm = str(step.actor["arm"])
            side = arm.removesuffix("_arm")
            try:
                feasible, _ = self._plan_arm_candidate(
                    step,
                    arm=arm,
                    env=env,
                    initial_state=world_states.get(side),
                    failed=failed,
                    runtime_kwargs=runtime_kwargs,
                )
            except Exception as error:
                log_warning(f"Required-arm {arm} planning failed: {error}")
                feasible = torch.zeros_like(failed)
            selection_failed = ~feasible & ~failed
            assignments = [
                (
                    arm
                    if not (
                        bool(failed[index].item())
                        or bool(selection_failed[index].item())
                    )
                    else None
                )
                for index in range(len(failed))
            ]
            return assignments, selection_failed
        if mode == "coordinated":
            assignments = [None if bool(value) else "coordinated" for value in failed]
            return assignments, torch.zeros_like(failed)

        candidates = {}
        for side in ("left", "right"):
            arm = f"{side}_arm"
            try:
                feasible, cost = self._plan_arm_candidate(
                    step,
                    arm=arm,
                    env=env,
                    initial_state=world_states.get(side),
                    failed=failed,
                    runtime_kwargs=runtime_kwargs,
                )
            except Exception as error:
                log_warning(f"Auto-arm {arm} candidate planning failed: {error}")
                feasible = torch.zeros_like(failed)
                cost = torch.full_like(failed, float("inf"), dtype=torch.float32)
            candidates[side] = (feasible & ~failed, cost)
        assignments, selection_failed = select_auto_arm_from_candidates(
            candidates["left"][0],
            candidates["right"][0],
            candidates["left"][1],
            candidates["right"][1],
        )
        return assignments, selection_failed & ~failed

    def _plan_arm_candidate(
        self,
        step: AgentSemanticStep,
        *,
        arm: str,
        env: Any,
        initial_state: Any,
        failed: torch.Tensor,
        runtime_kwargs: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan pickup and semantic transport before fixing one arm assignment."""
        candidate_edges = [
            self.edges[edge_id]
            for edge_id in step.edge_ids
            if self.edges[edge_id].symbolic_actions[0]["atomic_action_class"]
            in {"PickUp", "MoveHeldObject"}
        ]
        if not candidate_edges:
            candidate_edges = [self.edges[step.edge_ids[0]]]
        downstream = self._pickup_downstream_targets(
            step,
            env=env,
            arms=(arm,),
        )
        planner_kwargs = _execution_kwargs(runtime_kwargs)
        planner_kwargs["pickup_downstream_object_target_specs"] = downstream
        feasible = ~failed
        total_cost = torch.zeros(
            int(env.num_envs),
            dtype=torch.float32,
            device=failed.device,
        )
        state = initial_state
        for edge in candidate_edges:
            grounded = ground_symbolic_action(
                edge.symbolic_actions[0],
                step,
                env=env,
                arm=arm,
            )
            executed = _execute_atomic_action_result(
                grounded.action_spec,
                env=env,
                state=state,
                **planner_kwargs,
            )
            edge_feasible = (
                torch.ones_like(failed)
                if executed.failed_env_mask is None
                else ~executed.failed_env_mask.to(
                    device=failed.device,
                    dtype=torch.bool,
                )
            )
            feasible &= edge_feasible
            total_cost += _trajectory_cost(executed.action, env, failed.device)
            state = executed.next_state
        return feasible, total_cost

    def _pickup_downstream_targets(
        self,
        step: AgentSemanticStep,
        *,
        env: Any,
        arms: Sequence[str],
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        """Resolve future held-object targets used during grasp selection."""
        result: dict[str, tuple[dict[str, Any], ...]] = {}
        for arm in arms:
            target_specs = []
            for edge_id in step.edge_ids:
                action = self.edges[edge_id].symbolic_actions[0]
                if action["atomic_action_class"] != "MoveHeldObject":
                    continue
                grounded = ground_symbolic_action(
                    action,
                    step,
                    env=env,
                    arm=arm,
                )
                target_pose = grounded.action_spec.get("target_object_pose")
                if isinstance(target_pose, Mapping):
                    target_specs.append(deepcopy(dict(target_pose)))
            result[arm] = tuple(target_specs)
        return result

    def _check_dependencies(
        self,
        step: AgentSemanticStep,
        *,
        failed: torch.Tensor,
        semantic_success: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        result = failed.clone()
        dependencies_satisfied = torch.ones_like(result)
        for dependency in step.depends_on:
            completed = semantic_success.get(dependency)
            if completed is None:
                raise RuntimeError(
                    f"Semantic step {step.id!r} started before dependency "
                    f"{dependency!r} completed."
                )
            dependencies_satisfied &= completed
        result |= ~dependencies_satisfied
        return result

    def _complete_semantic_step(
        self,
        step: AgentSemanticStep,
        *,
        env: Any,
        failed: torch.Tensor,
        target_positions: torch.Tensor | None,
        motion_policy: Mapping[str, Any] | None,
        settle_steps: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        float | None,
    ]:
        if settle_steps < 0:
            raise ValueError("semantic_step_settle_steps must be non-negative.")
        if settle_steps and bool((~failed).any()):
            env.sim.update(step=settle_steps)
        observed = env.sim.get_rigid_object(step.object_uid).get_local_pose(
            to_matrix=True
        )[:, :3, 3]
        if target_positions is None:
            postcondition_success = ~failed
            distance = None
            tolerance = None
        else:
            if motion_policy is None or "postcondition_tolerance" not in motion_policy:
                raise ValueError(
                    f"Semantic step {step.id!r} has no resolved postcondition "
                    "tolerance in its motion policy."
                )
            tolerance = float(motion_policy["postcondition_tolerance"])
            target = target_positions.to(device=observed.device, dtype=observed.dtype)
            distance = torch.linalg.norm(observed - target, dim=-1)
            postcondition_success = distance <= tolerance
        active = ~failed
        success = active & postcondition_success
        failed = failed | (active & ~postcondition_success)
        log_info(
            f"Verified semantic step {step.id}: "
            f"succeeded={int(success.sum().item())}/{len(success)}."
        )
        return failed, success, observed, distance, tolerance

    def _next_edge(self, node_id: str) -> str:
        outgoing_edges = self.outgoing[node_id]
        if len(outgoing_edges) != 1:
            raise RuntimeError(
                f"Seed node {node_id!r} must have exactly one outgoing edge."
            )
        return outgoing_edges[0]


def _trajectory_cost(
    action: np.ndarray,
    env: Any,
    device: torch.device,
) -> torch.Tensor:
    trajectory = np.asarray(action, dtype=np.float32)
    if trajectory.ndim == 2:
        trajectory = trajectory[None, ...]
    if trajectory.shape[0] != int(env.num_envs):
        raise ValueError("Candidate trajectory batch does not match env.num_envs.")
    diffs = np.diff(trajectory, axis=1)
    cost = np.linalg.norm(diffs, axis=-1).sum(axis=-1)
    return torch.as_tensor(cost, dtype=torch.float32, device=device)


def _resolved_object_target_positions(
    arm_actions: Mapping[str, Any],
    *,
    fallback: torch.Tensor | None,
) -> torch.Tensor | None:
    """Read the geometry-adjusted object target produced by the planner."""
    for executed in arm_actions.values():
        pose = getattr(executed, "resolved_object_target_pose", None)
        if not isinstance(pose, torch.Tensor):
            continue
        if pose.ndim == 2 and tuple(pose.shape) == (4, 4):
            return pose[:3, 3].unsqueeze(0)
        if pose.ndim == 3 and tuple(pose.shape[-2:]) == (4, 4):
            return pose[:, :3, 3]
        raise ValueError(
            f"Resolved object target pose has invalid shape {tuple(pose.shape)}."
        )
    return fallback


def _representative_arm(
    assignments: Sequence[str | None],
    actor: Mapping[str, Any],
) -> str:
    for arm in assignments:
        if arm in {"left_arm", "right_arm"}:
            return arm
    configured = actor.get("arm")
    return str(configured) if configured in {"left_arm", "right_arm"} else "left_arm"


def _execution_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    internal = {
        "env",
        "episode_index",
        "runtime_run_id",
        "semantic_step_settle_steps",
    }
    return {key: value for key, value in kwargs.items() if key not in internal}
