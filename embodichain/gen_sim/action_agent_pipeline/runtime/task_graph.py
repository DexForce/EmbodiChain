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

"""Execute a Seed v5 action DAG against live environment state."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import WorldState
from embodichain.gen_sim.action_agent_pipeline.config.defaults import defaults_section
from embodichain.gen_sim.action_agent_pipeline.domain.success_policy import (
    upright_in_place_success_spec,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_execution import (
    _execute_atomic_action_result,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.arrangement_runtime import (
    ArrangementRuntimePlan,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atom_actions import (
    execute_parallel_atomic_actions,
    init_parallel_world_states,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.symbolic_grounding import (
    ArmCandidateScore,
    ground_symbolic_action,
    score_arm_candidate,
    select_auto_arm_from_candidates,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.pose_utils import (
    _object_world_vertices,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.success_evaluator import (
    evaluate_configured_success,
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
_UPRIGHT_IN_PLACE_XY_TOLERANCE = float(
    _CLOSED_LOOP_DEFAULTS["upright_in_place_xy_tolerance"]
)
_UPRIGHT_MAX_TILT = float(_CLOSED_LOOP_DEFAULTS["upright_max_tilt"])
_ARM_SELECTION_DEFAULTS = defaults_section("arm_selection")
_ARM_CROSSING_DEADBAND_RATIO = float(_ARM_SELECTION_DEFAULTS["crossing_deadband_ratio"])
_ARM_PICKUP_CROSSING_WEIGHT = float(_ARM_SELECTION_DEFAULTS["pickup_crossing_weight"])
_ARM_PLACEMENT_CROSSING_WEIGHT = float(
    _ARM_SELECTION_DEFAULTS["placement_crossing_weight"]
)
_ARM_MOTION_COST_SCALE = float(_ARM_SELECTION_DEFAULTS["motion_cost_scale"])
_ARM_FALLBACK_WORKSPACE_HALF_WIDTH = float(
    _ARM_SELECTION_DEFAULTS["fallback_workspace_half_width"]
)


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
    depends_on: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()


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
        self.runtime_success: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __getitem__(self, index):
        return self.actions[index]


class AgentTaskGraph:
    """Runtime executor for one immutable executable Seed Graph v5."""

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
        self.allocation_groups = tuple(
            deepcopy(group) for group in seed_graph.get("allocation_groups", ())
        )
        self.allocation_group_by_step = {
            str(step_id): group
            for group in self.allocation_groups
            for step_id in group["semantic_step_ids"]
        }
        self._candidate_failure_phases: dict[tuple[str, str], list[str | None]] = {}
        self._candidate_scores: dict[tuple[str, str], ArmCandidateScore] = {}
        self._candidate_feasible: dict[tuple[str, str], torch.Tensor] = {}
        self._step_arm_world_states: dict[tuple[str, str], Any] = {}

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
        depends_on: Sequence[str] = (),
        resources: Sequence[str] = (),
    ) -> "AgentTaskGraph":
        self.edges[edge_id] = AgentGraphEdge(
            id=edge_id,
            source=source,
            target=target,
            symbolic_actions=tuple(
                deepcopy(dict(action)) for action in symbolic_actions
            ),
            depends_on=tuple(str(edge_id) for edge_id in depends_on),
            resources=tuple(str(resource) for resource in resources),
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
        """Execute all ready Seed actions with conservative resource packing."""
        if env is None:
            raise ValueError("env is required to run an agent task graph.")
        recorder = RuntimeTaskGraphRecorder(
            self.seed_graph,
            env=env,
            run_id=kwargs.get("runtime_run_id"),
            episode_index=int(kwargs.get("episode_index", 0)),
            graph_renderer=kwargs.get("runtime_graph_renderer"),
        )
        arrangement_plan = (
            ArrangementRuntimePlan(
                env=env,
                semantic_steps=list(self.semantic_steps.values()),
            )
            if self.seed_graph.get("route") == "arrangement_line"
            else None
        )
        executed_actions: list[Any] = []
        self._step_arm_world_states.clear()
        self._candidate_failure_phases.clear()
        self._candidate_scores.clear()
        self._candidate_feasible.clear()
        world_states = init_parallel_world_states(env)
        failed = torch.zeros(int(env.num_envs), dtype=torch.bool, device=env.device)
        unsafe_failed = torch.zeros_like(failed)
        cleanup_failed = torch.zeros_like(failed)
        step_cleanup_failed: dict[str, torch.Tensor] = {}
        semantic_success: dict[str, torch.Tensor] = {}
        step_assignments: dict[str, list[str | None]] = {}
        step_selection_failures: dict[str, torch.Tensor] = {}
        step_target_positions: dict[str, torch.Tensor] = {}
        step_motion_policies: dict[str, dict[str, Any]] = {}
        dependency_checked_steps: set[str] = set()
        recorded_steps: set[str] = set()
        step_active_masks: dict[str, torch.Tensor] = {}
        completed_edges: set[str] = set()
        remaining_edges = list(self.edges)
        transitions = 0
        aborted_reason = None
        relation_success = None
        try:
            while remaining_edges:
                ready_edges = [
                    self.edges[edge_id]
                    for edge_id in remaining_edges
                    if set(self.edges[edge_id].depends_on) <= completed_edges
                ]
                if not ready_edges:
                    raise RuntimeError(
                        "Seed action DAG has no ready edge; dependencies are invalid."
                    )
                batch = self._pack_ready_edges(
                    ready_edges,
                    strict_serial=bool(kwargs.get("strict_serial", False)),
                )
                transitions += len(batch)
                if transitions > self.max_transitions:
                    raise RuntimeError("Agent task graph exceeded max_transitions.")

                # ``failed`` is local to the currently executing semantic branch.
                # A completed sibling branch must not suppress an independent
                # ready step; only declared semantic dependencies propagate failure.
                new_steps = []
                for edge in batch:
                    step = self.semantic_step_by_edge[edge.id]
                    if (
                        step.id in dependency_checked_steps
                        or self._is_prefetched_pickup(
                            edge,
                            step,
                        )
                    ):
                        continue
                    new_steps.append(step)
                if new_steps:
                    failed = unsafe_failed.clone()
                    for step in new_steps:
                        failed = self._check_dependencies(
                            step,
                            failed=failed,
                            semantic_success=semantic_success,
                        )
                        dependency_checked_steps.add(step.id)

                if len(batch) == 2 and self._batch_has_distinct_arm_group(batch):
                    eligible_for_group = ~failed
                    assignments_by_step, selection_failed = (
                        self._select_parallel_pickup_arms(
                            batch,
                            env=env,
                            world_states=world_states,
                            failed=failed,
                            runtime_kwargs=kwargs,
                            arrangement_plan=arrangement_plan,
                        )
                    )
                    failed |= selection_failed
                    for step_id, assignments in assignments_by_step.items():
                        step_assignments[step_id] = assignments
                        step_selection_failures[step_id] = selection_failed
                        step_active_masks[step_id] = eligible_for_group.clone()

                for edge in batch:
                    step = self.semantic_step_by_edge[edge.id]
                    if step.id not in step_assignments:
                        eligible_for_step = ~failed
                        assignments, selection_failed = self._select_step_arms(
                            step,
                            env=env,
                            world_states=world_states,
                            failed=failed,
                            runtime_kwargs=kwargs,
                            arrangement_plan=arrangement_plan,
                        )
                        failed |= selection_failed
                        step_assignments[step.id] = assignments
                        step_selection_failures[step.id] = selection_failed
                        step_active_masks[step.id] = eligible_for_step
                    if step.id in recorded_steps:
                        continue
                    assignments = step_assignments[step.id]
                    selection_failed = step_selection_failures[step.id]
                    preview = ground_symbolic_action(
                        edge.symbolic_actions[0],
                        step,
                        env=env,
                        arm=_representative_arm(assignments, step.actor),
                        arrangement_plan=arrangement_plan,
                    )
                    recorder.begin_step(
                        step,
                        assignments=assignments,
                        object_pose=preview.object_pose,
                        reference_pose=preview.reference_pose,
                        active_mask=step_active_masks[step.id],
                        selection_failed_mask=selection_failed,
                        physical_control_parts=_physical_control_parts_for_assignments(
                            env,
                            assignments,
                        ),
                        arrangement_metadata=(
                            arrangement_plan.metadata(step)
                            if arrangement_plan is not None
                            else None
                        ),
                        candidate_failures=self._step_candidate_failures(
                            step,
                            int(env.num_envs),
                        ),
                        candidate_scores=self._step_candidate_scores(
                            step,
                            int(env.num_envs),
                        ),
                    )
                    physical_control_parts = _physical_control_parts_for_assignments(
                        env,
                        assignments,
                    )
                    log_info(
                        f"Grounded semantic step {step.id}: operator={step.operator}, "
                        f"object={step.object_uid}, semantic_arms={assignments}, "
                        f"physical_control_parts={physical_control_parts}."
                    )
                    recorded_steps.add(step.id)

                failed_before = failed.clone()
                if len(batch) == 2:
                    result, grounded_by_edge = self._execute_parallel_pickup_edges(
                        batch,
                        assignments_by_step=step_assignments,
                        env=env,
                        world_states=world_states,
                        failed=failed,
                        runtime_kwargs=kwargs,
                        arrangement_plan=arrangement_plan,
                    )
                else:
                    edge = batch[0]
                    step = self.semantic_step_by_edge[edge.id]
                    result, grounded_actions = self._execute_symbolic_edge(
                        edge,
                        step,
                        assignments=step_assignments[step.id],
                        env=env,
                        world_states=world_states,
                        failed=failed,
                        runtime_kwargs=kwargs,
                        arrangement_plan=arrangement_plan,
                    )
                    grounded_by_edge = {edge.id: grounded_actions}
                world_states = result["world_states"]
                actions = result["actions"]
                executed_actions.extend(actions)
                execution_failed = result["failed_env_mask"]
                batch_is_cleanup = all(_is_cleanup_edge(edge) for edge in batch)
                failed, newly_cleanup_failed = _classify_execution_failure(
                    failed_before,
                    execution_failed,
                    cleanup=batch_is_cleanup,
                )
                if batch_is_cleanup:
                    unsafe_failed |= newly_cleanup_failed
                    cleanup_failed |= newly_cleanup_failed
                    for edge in batch:
                        step_id = self.semantic_step_by_edge[edge.id].id
                        current_cleanup = step_cleanup_failed.setdefault(
                            step_id,
                            torch.zeros_like(failed),
                        )
                        current_cleanup |= newly_cleanup_failed
                else:
                    unsafe_failed |= execution_failed & ~failed_before

                for edge in batch:
                    step = self.semantic_step_by_edge[edge.id]
                    grounded_actions = grounded_by_edge[edge.id]
                    assignments = step_assignments[step.id]
                    resolved_target_positions = _postcondition_target_positions(
                        edge,
                        arm_actions=result["arm_actions"],
                        grounded_actions=grounded_actions,
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
                        failed_after=execution_failed,
                        grounding_failed=(
                            step_selection_failures[step.id]
                            if edge.id == step.edge_ids[0]
                            else torch.zeros_like(failed)
                        ),
                        action_steps=len(actions),
                        arm_actions=result.get(
                            "arm_actions_by_env",
                            result["arm_actions"],
                        ),
                        failure_class=(
                            "cleanup" if _is_cleanup_edge(edge) else "fatal"
                        ),
                    )
                    completed_edges.add(edge.id)
                    remaining_edges.remove(edge.id)
                    log_info(
                        f"Completed symbolic edge {edge.id}: "
                        f"action_steps={len(actions)}, "
                        f"failed_envs={int(execution_failed.sum().item())}."
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
                        if arrangement_plan is not None:
                            arrangement_plan.mark_completed(step.id, success)
                        recorder.complete_step(
                            step.id,
                            success=success,
                            failed_mask=failed,
                            observed_positions=observed_positions,
                            target_positions=step_target_positions.get(step.id),
                            position_error=position_error,
                            tolerance=tolerance,
                            cleanup_failed_mask=step_cleanup_failed.get(step.id),
                        )
            semantic_all = torch.ones_like(failed)
            for success in semantic_success.values():
                semantic_all &= success
            failed = ~semantic_all
            if arrangement_plan is not None:
                relation_success = evaluate_configured_success(env)
                failed |= ~relation_success
        except BaseException as error:
            aborted_reason = f"{type(error).__name__}: {error}"
            raise
        finally:
            recorder.finalize(
                failed,
                aborted_reason=aborted_reason,
                relation_success=relation_success,
                cleanup_failed_mask=cleanup_failed,
            )

        result = ExecutedActionList(executed_actions)
        result.semantic_step_success = semantic_success
        result.runtime_success = ~failed
        result.runtime_graph_output_dir = str(recorder.output_dir)
        return result

    def _pack_ready_edges(
        self,
        ready_edges: Sequence[AgentGraphEdge],
        *,
        strict_serial: bool = False,
    ) -> tuple[AgentGraphEdge, ...]:
        """Pack one declared, resource-safe dual-PickUp group."""
        first = ready_edges[0]
        if strict_serial:
            return (first,)
        if not _is_parallel_pickup_candidate(first, self.semantic_step_by_edge):
            return (first,)
        first_step = self.semantic_step_by_edge[first.id]
        first_resources = set(first.resources)
        for second in ready_edges[1:]:
            if not _is_parallel_pickup_candidate(
                second,
                self.semantic_step_by_edge,
            ):
                continue
            second_step = self.semantic_step_by_edge[second.id]
            if first_step.object_uid == second_step.object_uid:
                continue
            if (
                first_step.actor.get("mode") == "required"
                and second_step.actor.get("mode") == "required"
                and first_step.actor["arm"] == second_step.actor["arm"]
            ):
                continue
            shared_resources = first_resources & set(second.resources)
            if shared_resources - {"arm:auto"}:
                continue
            if not self._steps_share_parallel_group(first_step.id, second_step.id):
                if (
                    first_step.actor.get("mode") != "required"
                    or second_step.actor.get("mode") != "required"
                ):
                    continue
            return (first, second)
        return (first,)

    def _steps_share_parallel_group(self, first_id: str, second_id: str) -> bool:
        first_group = self.allocation_group_by_step.get(first_id)
        second_group = self.allocation_group_by_step.get(second_id)
        return first_group is not None and first_group is second_group

    def _batch_has_distinct_arm_group(
        self,
        edges: Sequence[AgentGraphEdge],
    ) -> bool:
        if len(edges) != 2:
            return False
        first = self.semantic_step_by_edge[edges[0].id]
        second = self.semantic_step_by_edge[edges[1].id]
        return self._steps_share_parallel_group(first.id, second.id) and any(
            step.actor.get("mode") == "auto" for step in (first, second)
        )

    def _select_parallel_pickup_arms(
        self,
        edges: Sequence[AgentGraphEdge],
        *,
        env: Any,
        world_states: Mapping[str, Any],
        failed: torch.Tensor,
        runtime_kwargs: Mapping[str, Any],
        arrangement_plan: ArrangementRuntimePlan | None,
    ) -> tuple[dict[str, list[str | None]], torch.Tensor]:
        """Jointly choose one distinct-arm assignment for a dual-PickUp group.

        The selected permutation may differ by environment. Held-object states
        are isolated by semantic step and arm so one vectorized arm can safely
        carry different step objects in disjoint environment masks.
        """
        steps = [self.semantic_step_by_edge[edge.id] for edge in edges]
        candidates: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for step in steps:
            for arm in ("left_arm", "right_arm"):
                if (
                    step.actor.get("mode") == "required"
                    and step.actor.get("arm") != arm
                ):
                    candidates[(step.id, arm)] = (
                        torch.zeros_like(failed),
                        torch.full_like(failed, float("inf"), dtype=torch.float32),
                    )
                    continue
                side = arm.removesuffix("_arm")
                try:
                    feasible, cost = self._plan_arm_candidate(
                        step,
                        arm=arm,
                        env=env,
                        initial_state=world_states.get(side),
                        failed=failed,
                        runtime_kwargs=runtime_kwargs,
                        arrangement_plan=arrangement_plan,
                    )
                except Exception as error:
                    log_info(
                        f"Rejected {arm} candidate for {step.id}: {error}",
                    )
                    self._candidate_scores.pop((step.id, arm), None)
                    self._candidate_feasible.pop((step.id, arm), None)
                    feasible = torch.zeros_like(failed)
                    cost = torch.full_like(failed, float("inf"), dtype=torch.float32)
                    self._candidate_failure_phases[(step.id, arm)] = [
                        (
                            _candidate_exception_phase(error)
                            if not bool(value.item())
                            else None
                        )
                        for value in failed
                    ]
                candidates[(step.id, arm)] = (feasible & ~failed, cost)

        permutations = (
            ("left_arm", "right_arm"),
            ("right_arm", "left_arm"),
        )
        assignments = {step.id: [None] * int(env.num_envs) for step in steps}
        selection_failed = torch.zeros_like(failed)
        selected_counts = {permutation: 0 for permutation in permutations}
        for env_id in range(int(env.num_envs)):
            if bool(failed[env_id].item()):
                continue
            ranked = []
            for permutation in permutations:
                first_arm, second_arm = permutation
                first_feasible, first_cost = candidates[(steps[0].id, first_arm)]
                second_feasible, second_cost = candidates[(steps[1].id, second_arm)]
                feasible = bool(
                    first_feasible[env_id].item() and second_feasible[env_id].item()
                )
                cost = float(first_cost[env_id].item() + second_cost[env_id].item())
                ranked.append((not feasible, cost, permutation))
            ranked.sort(key=lambda item: (item[0], item[1], item[2]))
            infeasible, _, selected = ranked[0]
            if infeasible:
                selection_failed[env_id] = True
                continue
            assignments[steps[0].id][env_id] = selected[0]
            assignments[steps[1].id][env_id] = selected[1]
            selected_counts[selected] += 1
        log_info(
            "Selected per-environment dual-PickUp assignments: "
            + ", ".join(
                f"{permutation[0]}/{permutation[1]}={count}"
                for permutation, count in selected_counts.items()
            )
            + f"; failed={int(selection_failed.sum().item())}."
        )
        return assignments, selection_failed

    def _is_prefetched_pickup(
        self,
        edge: AgentGraphEdge,
        step: AgentSemanticStep,
    ) -> bool:
        if not _edge_has_action_class(edge, "PickUp") or not step.depends_on:
            return False
        dependency_tails = {
            self.semantic_steps[step_id].edge_ids[-1] for step_id in step.depends_on
        }
        return dependency_tails.isdisjoint(edge.depends_on)

    def _execute_parallel_pickup_edges(
        self,
        edges: Sequence[AgentGraphEdge],
        *,
        assignments_by_step: Mapping[str, Sequence[str | None]],
        env: Any,
        world_states: Mapping[str, Any],
        failed: torch.Tensor,
        runtime_kwargs: Mapping[str, Any],
        arrangement_plan: ArrangementRuntimePlan | None,
    ) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]]]:
        """Ground and execute one selected PickUp per semantic arm."""
        grounded_lists: dict[str, list[Any]] = {edge.id: [] for edge in edges}
        result: dict[str, Any] = {
            "actions": [],
            "world_states": dict(world_states),
            "arm_actions": {},
            "failed_env_mask": failed.clone(),
            "arm_actions_by_env": [{} for _ in range(int(env.num_envs))],
        }
        fell_back_to_serial = False
        steps = [self.semantic_step_by_edge[edge.id] for edge in edges]
        permutations = (
            ("left_arm", "right_arm"),
            ("right_arm", "left_arm"),
        )
        for permutation in permutations:
            partition_mask = torch.tensor(
                [
                    assignments_by_step[steps[0].id][env_id] == permutation[0]
                    and assignments_by_step[steps[1].id][env_id] == permutation[1]
                    for env_id in range(int(env.num_envs))
                ],
                dtype=torch.bool,
                device=env.device,
            )
            if not bool(partition_mask.any()):
                continue
            edge_by_arm = {
                permutation[index]: edges[index] for index in range(len(edges))
            }
            grounded_by_arm = {}
            downstream: dict[str, tuple[dict[str, Any], ...]] = {}
            for arm, edge in edge_by_arm.items():
                step = self.semantic_step_by_edge[edge.id]
                grounded = ground_symbolic_action(
                    edge.symbolic_actions[0],
                    step,
                    env=env,
                    arm=arm,
                    arrangement_plan=arrangement_plan,
                )
                grounded_lists[edge.id].append(grounded)
                grounded_by_arm[arm] = grounded
                downstream.update(
                    self._pickup_downstream_targets(
                        step,
                        env=env,
                        arms=(arm,),
                        arrangement_plan=arrangement_plan,
                    )
                )
            execution_kwargs = _execution_kwargs(runtime_kwargs)
            execution_kwargs["pickup_downstream_object_target_specs"] = downstream
            execution_kwargs["require_joint_safety"] = True
            partition_world_states = dict(result["world_states"])
            for side in ("left", "right"):
                state = partition_world_states.get(side)
                if state is not None:
                    partition_world_states[side] = WorldState(
                        last_qpos=state.last_qpos.clone()
                    )
            partition_result = execute_parallel_atomic_actions(
                left_arm_action=grounded_by_arm["left_arm"].action_spec,
                right_arm_action=grounded_by_arm["right_arm"].action_spec,
                left_active_env_mask=partition_mask,
                right_active_env_mask=partition_mask,
                env=env,
                world_states=partition_world_states,
                failed_env_mask=result["failed_env_mask"],
                return_result=True,
                **execution_kwargs,
            )
            if partition_result.get("parallel_rejected", False):
                fell_back_to_serial = True
                reason = partition_result.get("parallel_safety", {}).get(
                    "reason", "unknown"
                )
                log_info(
                    "Parallel PickUp preflight rejected; falling back to serial "
                    f"execution before stepping: {reason}."
                )
                for edge in edges:
                    step = self.semantic_step_by_edge[edge.id]
                    serial_result, serial_grounded = self._execute_symbolic_edge(
                        edge,
                        step,
                        assignments=assignments_by_step[step.id],
                        env=env,
                        world_states=result["world_states"],
                        failed=result["failed_env_mask"],
                        runtime_kwargs=runtime_kwargs,
                        arrangement_plan=arrangement_plan,
                    )
                    result["actions"].extend(serial_result["actions"])
                    result["world_states"] = serial_result["world_states"]
                    result["failed_env_mask"] = serial_result["failed_env_mask"]
                    grounded_lists[edge.id].extend(serial_grounded)
                break
            result["actions"].extend(partition_result["actions"])
            result["world_states"] = partition_result["world_states"]
            result["arm_actions"] = partition_result["arm_actions"]
            result["failed_env_mask"] = partition_result["failed_env_mask"]
            for env_id in torch.nonzero(partition_mask).flatten().tolist():
                result["arm_actions_by_env"][env_id] = partition_result["arm_actions"]
            for arm, edge in edge_by_arm.items():
                executed = partition_result["arm_actions"][arm.removesuffix("_arm")]
                next_state = getattr(executed, "next_state", None)
                if next_state is not None:
                    step = self.semantic_step_by_edge[edge.id]
                    self._step_arm_world_states[(step.id, arm)] = next_state
        grounded_by_edge = {
            edge_id: tuple(grounded) for edge_id, grounded in grounded_lists.items()
        }
        log_info(
            (
                "Executed serial fallback for distinct-arm PickUps: "
                if fell_back_to_serial
                else "Executed parallel distinct-arm PickUps: "
            )
            + ", ".join(
                f"{self.semantic_step_by_edge[edge.id].id}="
                f"{assignments_by_step[self.semantic_step_by_edge[edge.id].id]}"
                for edge in edges
            )
        )
        return result, grounded_by_edge

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
        arrangement_plan: ArrangementRuntimePlan | None = None,
    ) -> tuple[dict[str, Any], tuple[Any, ...]]:
        if step.actor["mode"] == "coordinated":
            return self._execute_coordinated_edge(
                edge,
                step,
                env=env,
                world_states=world_states,
                failed=failed,
                runtime_kwargs=runtime_kwargs,
                arrangement_plan=arrangement_plan,
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
            ground_symbolic_action(
                symbolic,
                step,
                env=env,
                arm="left_arm",
                arrangement_plan=arrangement_plan,
            )
            if bool(left_mask.any())
            else None
        )
        right_grounded = (
            ground_symbolic_action(
                symbolic,
                step,
                env=env,
                arm="right_arm",
                arrangement_plan=arrangement_plan,
            )
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
                arrangement_plan=arrangement_plan,
            )
        execution_kwargs = _execution_kwargs(runtime_kwargs)
        if str(symbolic["atomic_action_class"]) == "PickUp":
            execution_kwargs["pickup_downstream_object_target_specs"] = (
                self._pickup_downstream_targets(
                    step,
                    env=env,
                    arms=("left_arm", "right_arm"),
                    arrangement_plan=arrangement_plan,
                )
            )
        execution_world_states = dict(world_states)
        for arm in ("left_arm", "right_arm"):
            step_state = self._step_arm_world_states.get((step.id, arm))
            if step_state is not None:
                execution_world_states[arm.removesuffix("_arm")] = step_state
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
            world_states=execution_world_states,
            failed_env_mask=failed,
            return_result=True,
            **execution_kwargs,
        )
        for arm in ("left_arm", "right_arm"):
            executed = result["arm_actions"].get(arm.removesuffix("_arm"))
            next_state = getattr(executed, "next_state", None)
            if next_state is not None:
                self._step_arm_world_states[(step.id, arm)] = next_state
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
        arrangement_plan: ArrangementRuntimePlan | None = None,
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
                arrangement_plan=arrangement_plan,
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
                arrangement_plan=arrangement_plan,
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
        arrangement_plan: ArrangementRuntimePlan | None = None,
        allow_reassignment: bool = True,
    ) -> tuple[list[str | None], torch.Tensor]:
        if bool(failed.all()):
            return [None] * len(failed), torch.zeros_like(failed)

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
                    arrangement_plan=arrangement_plan,
                )
            except Exception as error:
                log_warning(f"Required-arm {arm} planning failed: {error}")
                self._candidate_scores.pop((step.id, arm), None)
                self._candidate_feasible.pop((step.id, arm), None)
                feasible = torch.zeros_like(failed)
                self._candidate_failure_phases[(step.id, arm)] = [
                    (
                        _candidate_exception_phase(error)
                        if not bool(value.item())
                        else None
                    )
                    for value in failed
                ]
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
                    arrangement_plan=arrangement_plan,
                )
            except Exception as error:
                log_warning(f"Auto-arm {arm} candidate planning failed: {error}")
                self._candidate_scores.pop((step.id, arm), None)
                self._candidate_feasible.pop((step.id, arm), None)
                feasible = torch.zeros_like(failed)
                cost = torch.full_like(failed, float("inf"), dtype=torch.float32)
                self._candidate_failure_phases[(step.id, arm)] = [
                    (
                        _candidate_exception_phase(error)
                        if not bool(value.item())
                        else None
                    )
                    for value in failed
                ]
            candidates[side] = (feasible & ~failed, cost)
        assignments, selection_failed = select_auto_arm_from_candidates(
            candidates["left"][0],
            candidates["right"][0],
            candidates["left"][1],
            candidates["right"][1],
        )
        selection_failed &= ~failed
        if (
            allow_reassignment
            and arrangement_plan is not None
            and step.goal.get("slot_constraint") == "free_reassignable"
            and bool(selection_failed.any())
        ):
            initial_failures = {
                arm: list(
                    self._candidate_failure_phases.get(
                        (step.id, arm),
                        [None] * int(env.num_envs),
                    )
                )
                for arm in ("left_arm", "right_arm")
            }
            reassigned = self._reassign_free_slots(
                env=env,
                world_states=world_states,
                failed=failed,
                trigger_mask=selection_failed,
                runtime_kwargs=runtime_kwargs,
                arrangement_plan=arrangement_plan,
            )
            if bool(reassigned.any()):
                log_info(
                    "Reassigned unexecuted arrangement slots in environments "
                    f"{torch.nonzero(reassigned).flatten().tolist()}."
                )
                return self._select_step_arms(
                    step,
                    env=env,
                    world_states=world_states,
                    failed=failed,
                    runtime_kwargs=runtime_kwargs,
                    arrangement_plan=arrangement_plan,
                    allow_reassignment=False,
                )
            for arm, phases in initial_failures.items():
                self._candidate_failure_phases[(step.id, arm)] = phases
        return assignments, selection_failed

    def _reassign_free_slots(
        self,
        *,
        env: Any,
        world_states: Mapping[str, Any],
        failed: torch.Tensor,
        trigger_mask: torch.Tensor,
        runtime_kwargs: Mapping[str, Any],
        arrangement_plan: ArrangementRuntimePlan,
    ) -> torch.Tensor:
        """Globally rematch only unfinished objects and slots in failing envs."""
        from scipy.optimize import linear_sum_assignment

        reassigned = torch.zeros_like(trigger_mask)
        for env_id in torch.nonzero(trigger_mask).flatten().tolist():
            remaining_ids = arrangement_plan.remaining_step_ids(env_id)
            occupied = arrangement_plan.occupied_slots(env_id)
            available_slots = [
                slot
                for slot in range(arrangement_plan.slot_count)
                if slot not in occupied
            ]
            if len(remaining_ids) != len(available_slots):
                raise RuntimeError(
                    "Arrangement unfinished steps and unoccupied slots diverged."
                )
            original_slots = {
                step_id: int(arrangement_plan.resolved_slots[step_id][env_id].item())
                for step_id in remaining_ids
            }
            costs = np.full(
                (len(remaining_ids), len(available_slots)),
                np.inf,
                dtype=np.float64,
            )
            only_env_failed = torch.ones_like(failed)
            only_env_failed[env_id] = bool(failed[env_id].item())
            for row, step_id in enumerate(remaining_ids):
                candidate_step = self.semantic_steps[step_id]
                for column, slot in enumerate(available_slots):
                    arrangement_plan.resolved_slots[step_id][env_id] = int(slot)
                    arm_costs = []
                    for side in ("left", "right"):
                        try:
                            feasible, cost = self._plan_arm_candidate(
                                candidate_step,
                                arm=f"{side}_arm",
                                env=env,
                                initial_state=world_states.get(side),
                                failed=only_env_failed,
                                runtime_kwargs=runtime_kwargs,
                                arrangement_plan=arrangement_plan,
                            )
                        except Exception as error:
                            log_warning(
                                "Arrangement rematch candidate failed for "
                                f"env={env_id}, step={step_id}, slot={slot}, "
                                f"arm={side}_arm: {error}"
                            )
                            key = (step_id, f"{side}_arm")
                            self._candidate_scores.pop(key, None)
                            self._candidate_feasible.pop(key, None)
                            continue
                        if bool(feasible[env_id].item()):
                            arm_costs.append(float(cost[env_id].item()))
                    if arm_costs:
                        costs[row, column] = min(arm_costs)
                arrangement_plan.resolved_slots[step_id][env_id] = original_slots[
                    step_id
                ]

            if not np.isfinite(costs).any(axis=1).all():
                continue
            finite_costs = np.where(np.isfinite(costs), costs, 1.0e12)
            rows, columns = linear_sum_assignment(finite_costs)
            selected = costs[rows, columns]
            if len(rows) != len(remaining_ids) or not np.isfinite(selected).all():
                continue
            assignment = {
                remaining_ids[int(row)]: available_slots[int(column)]
                for row, column in zip(rows, columns)
            }
            arrangement_plan.set_assignment(
                env_id,
                assignment,
                reason="nominal slot path infeasible; globally rematched unfinished slots",
                cost=float(selected.sum()),
            )
            reassigned[env_id] = True
        return reassigned

    def _plan_arm_candidate(
        self,
        step: AgentSemanticStep,
        *,
        arm: str,
        env: Any,
        initial_state: Any,
        failed: torch.Tensor,
        runtime_kwargs: Mapping[str, Any],
        arrangement_plan: ArrangementRuntimePlan | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan mandatory grasp and held transport before assigning one arm."""
        log_info(
            f"Evaluating non-executing arm candidate: step={step.id}, arm={arm}. "
            "Any IK rejection before the selection summary is diagnostic."
        )
        candidate_edges = [
            self.edges[edge_id]
            for edge_id in step.edge_ids
            if _edge_has_action_class(
                self.edges[edge_id],
                "PickUp",
            )
            or _edge_has_action_class(
                self.edges[edge_id],
                "MoveHeldObject",
            )
        ]
        if not candidate_edges:
            candidate_edges = [self.edges[step.edge_ids[0]]]
        downstream = self._pickup_downstream_targets(
            step,
            env=env,
            arms=(arm,),
            arrangement_plan=arrangement_plan,
        )
        planner_kwargs = _execution_kwargs(runtime_kwargs)
        planner_kwargs["pickup_downstream_object_target_specs"] = downstream
        planner_kwargs["_diagnostic_context"] = "arm_candidate"
        feasible = ~failed
        total_cost = torch.zeros(
            int(env.num_envs),
            dtype=torch.float32,
            device=failed.device,
        )
        state = initial_state
        previous_eef_target: torch.Tensor | None = None
        source_pose: torch.Tensor | None = None
        target_pose: torch.Tensor | None = None
        failure_phases: list[str | None] = [None] * int(env.num_envs)
        for edge in candidate_edges:
            grounded = ground_symbolic_action(
                edge.symbolic_actions[0],
                step,
                env=env,
                arm=arm,
                arrangement_plan=arrangement_plan,
                policy_reference_pose=previous_eef_target,
            )
            if source_pose is None and grounded.object_pose is not None:
                source_pose = grounded.object_pose
            if grounded.target_object_pose is not None:
                binding = edge.symbolic_actions[0].get("target_binding", {})
                phase = str(binding.get("phase", "final"))
                if target_pose is None or phase == "final":
                    target_pose = grounded.target_object_pose
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
            phase = _candidate_edge_phase(edge)
            newly_failed = feasible & ~edge_feasible
            for env_id in torch.nonzero(newly_failed).flatten().tolist():
                failure_phases[env_id] = phase
            feasible &= edge_feasible
            total_cost += _trajectory_cost(executed.action, env, failed.device)
            state = executed.next_state
            if isinstance(executed.resolved_eef_target_pose, torch.Tensor):
                previous_eef_target = executed.resolved_eef_target_pose
            if not bool((feasible & ~failed).any()):
                break
        self._candidate_failure_phases[(step.id, arm)] = failure_phases
        workspace_center_y, workspace_half_width = _arm_selection_workspace(
            env,
            device=failed.device,
        )
        score = score_arm_candidate(
            arm=arm,
            motion_cost=total_cost,
            source_pose=source_pose,
            target_pose=target_pose,
            workspace_center_y=workspace_center_y,
            workspace_half_width=workspace_half_width,
            crossing_deadband_ratio=_ARM_CROSSING_DEADBAND_RATIO,
            pickup_crossing_weight=_ARM_PICKUP_CROSSING_WEIGHT,
            placement_crossing_weight=_ARM_PLACEMENT_CROSSING_WEIGHT,
            motion_cost_scale=_ARM_MOTION_COST_SCALE,
        )
        self._candidate_scores[(step.id, arm)] = score
        self._candidate_feasible[(step.id, arm)] = feasible.clone()
        log_info(
            f"Scored arm candidate: step={step.id}, arm={arm}, "
            f"motion={score.normalized_motion_cost.detach().cpu().tolist()}, "
            f"pickup_crossing={score.pickup_crossing_penalty.detach().cpu().tolist()}, "
            f"placement_crossing={score.placement_crossing_penalty.detach().cpu().tolist()}, "
            f"total={score.total_cost.detach().cpu().tolist()}."
        )
        return feasible, score.total_cost

    def _step_candidate_failures(
        self,
        step: AgentSemanticStep,
        count: int,
    ) -> list[dict[str, str | None]]:
        left = self._candidate_failure_phases.get(
            (step.id, "left_arm"),
            [None] * count,
        )
        right = self._candidate_failure_phases.get(
            (step.id, "right_arm"),
            [None] * len(left),
        )
        return [
            {
                "left_arm": left[index] if index < len(left) else None,
                "right_arm": right[index] if index < len(right) else None,
            }
            for index in range(count)
        ]

    def _step_candidate_scores(
        self,
        step: AgentSemanticStep,
        count: int,
    ) -> list[dict[str, dict[str, float | bool] | None]]:
        scores = {
            arm: self._candidate_scores.get((step.id, arm))
            for arm in ("left_arm", "right_arm")
        }
        feasible = {
            arm: self._candidate_feasible.get((step.id, arm))
            for arm in ("left_arm", "right_arm")
        }
        result: list[dict[str, dict[str, float | bool] | None]] = []
        for env_id in range(count):
            env_scores: dict[str, dict[str, float | bool] | None] = {}
            for arm, score in scores.items():
                if score is None or env_id >= len(score.total_cost):
                    env_scores[arm] = None
                    continue
                env_scores[arm] = {
                    "feasible": bool(
                        feasible[arm] is not None
                        and env_id < len(feasible[arm])
                        and feasible[arm][env_id].item()
                    ),
                    "motion_cost": float(score.motion_cost[env_id].item()),
                    "normalized_motion_cost": float(
                        score.normalized_motion_cost[env_id].item()
                    ),
                    "pickup_crossing_penalty": float(
                        score.pickup_crossing_penalty[env_id].item()
                    ),
                    "placement_crossing_penalty": float(
                        score.placement_crossing_penalty[env_id].item()
                    ),
                    "total_cost": float(score.total_cost[env_id].item()),
                }
            result.append(env_scores)
        return result

    def _pickup_downstream_targets(
        self,
        step: AgentSemanticStep,
        *,
        env: Any,
        arms: Sequence[str],
        arrangement_plan: ArrangementRuntimePlan | None = None,
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
                    arrangement_plan=arrangement_plan,
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
        # The validated postcondition is the runtime acceptance contract.
        # Goal fields still provide geometry and orientation details.
        postcondition_type = str(step.postcondition.get("type", ""))
        relation = str(step.postcondition.get("relation", ""))
        reference_object = step.postcondition.get("reference_object")
        if step.goal.get("placement_mode") == "upright_in_place":
            initial_pose = getattr(env, "agent_initial_object_poses", {}).get(
                step.object_uid
            )
            if initial_pose is None:
                raise ValueError(
                    f"Upright-in-place step {step.id!r} requires the initial "
                    f"pose of {step.object_uid!r}."
                )
            target = initial_pose[:, :3, 3].to(
                device=observed.device,
                dtype=observed.dtype,
            )
            distance = torch.linalg.norm(observed[:, :2] - target[:, :2], dim=-1)
            tolerance = _UPRIGHT_IN_PLACE_XY_TOLERANCE
            postcondition_success = evaluate_configured_success(
                env,
                upright_in_place_success_spec(
                    step.object_uid,
                    local_axis=str(step.goal.get("upright_local_axis", "z")),
                    xy_tolerance=_UPRIGHT_IN_PLACE_XY_TOLERANCE,
                    max_tilt=_UPRIGHT_MAX_TILT,
                ),
            )
        elif (
            postcondition_type == "semantic_goal"
            and relation == "inside"
            and isinstance(step.goal.get("reference_object"), str)
        ):
            postcondition_success = evaluate_configured_success(
                env,
                {
                    "type": "object_in_container",
                    "object": step.object_uid,
                    "container": step.goal["reference_object"],
                },
            )
            distance = None
            tolerance = None
        elif (
            postcondition_type == "stack_layer_supported"
            and isinstance(reference_object, str)
        ) or (
            postcondition_type == "semantic_goal"
            and relation in {"on", "on_top", "on_top_of"}
            and isinstance(step.goal.get("reference_object"), str)
        ):
            support = (
                reference_object
                if postcondition_type == "stack_layer_supported"
                else step.goal["reference_object"]
            )
            postcondition_success = evaluate_configured_success(
                env,
                {
                    "type": "object_on_object",
                    "object": step.object_uid,
                    "support": support,
                },
            )
            distance = None
            tolerance = None
        elif target_positions is None:
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


def _arm_selection_workspace(
    env: Any,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return live table center and half-width along robot-view world-y."""
    count = int(env.num_envs)
    centers = torch.zeros(count, dtype=torch.float32, device=device)
    half_widths = torch.full(
        (count,),
        _ARM_FALLBACK_WORKSPACE_HALF_WIDTH,
        dtype=torch.float32,
        device=device,
    )
    sim = getattr(env, "sim", None)
    table = sim.get_rigid_object("table") if sim is not None else None
    if table is None or not hasattr(table, "get_vertices"):
        return centers, half_widths

    for env_id in range(count):
        vertices = _object_world_vertices(table, device, env_id=env_id)
        minimum = vertices[:, 1].min()
        maximum = vertices[:, 1].max()
        half_width = (maximum - minimum) * 0.5
        if float(half_width) <= 1.0e-6:
            continue
        centers[env_id] = (minimum + maximum) * 0.5
        half_widths[env_id] = half_width
    return centers, half_widths


def _candidate_edge_phase(edge: AgentGraphEdge) -> str:
    action = edge.symbolic_actions[0]
    action_class = str(action["atomic_action_class"])
    if action_class == "PickUp":
        return "pickup"
    if action_class == "Place":
        return "release"
    if action_class == "MoveEndEffector":
        return "retreat"
    if action_class == "MoveJoints":
        return "home"
    binding = action.get("target_binding", {})
    phase = binding.get("phase")
    return str(phase) if phase in {"staging", "final"} else "transport"


def _candidate_exception_phase(error: Exception) -> str:
    """Keep the concrete candidate failure without allowing multiline artifacts."""
    message = " ".join(str(error).split())
    detail = f": {message}" if message else ""
    return f"candidate_exception:{type(error).__name__}{detail}"


def _edge_has_action_class(edge: AgentGraphEdge, action_class: str) -> bool:
    return any(
        str(action["atomic_action_class"]) == action_class
        for action in edge.symbolic_actions
    )


def _is_parallel_pickup_candidate(
    edge: AgentGraphEdge,
    step_by_edge: Mapping[str, AgentSemanticStep],
) -> bool:
    step = step_by_edge[edge.id]
    return (
        len(edge.symbolic_actions) == 1
        and _edge_has_action_class(edge, "PickUp")
        and step.actor.get("mode") in {"auto", "required"}
        and (
            step.actor.get("mode") == "auto"
            or step.actor.get("arm") in {"left_arm", "right_arm"}
        )
    )


def _is_cleanup_edge(edge: AgentGraphEdge) -> bool:
    """Classify best-effort post-release retreat and home actions."""
    for action in edge.symbolic_actions:
        binding = action.get("target_binding", {})
        if binding.get("kind") == "policy_pose":
            continue
        if (
            action.get("atomic_action_class") == "MoveJoints"
            and binding.get("kind") == "joint_state"
            and binding.get("source") == "initial"
        ):
            continue
        return False
    return True


def _classify_execution_failure(
    failed_before: torch.Tensor,
    execution_failed: torch.Tensor,
    *,
    cleanup: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep cleanup failures observable without poisoning task semantics."""
    newly_failed = execution_failed & ~failed_before
    if cleanup:
        return failed_before.clone(), newly_failed
    return execution_failed, torch.zeros_like(execution_failed)


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


def _postcondition_target_positions(
    edge: AgentGraphEdge,
    *,
    arm_actions: Mapping[str, Any],
    grounded_actions: Sequence[Any],
) -> torch.Tensor | None:
    """Return a target only when this edge owns the semantic goal geometry."""
    target_kinds = {
        str(action.get("target_binding", {}).get("kind", ""))
        for action in edge.symbolic_actions
    }
    if target_kinds.isdisjoint({"semantic_goal", "coordinated_goal"}):
        return None
    semantic_phases = {
        action.get("target_binding", {}).get("phase")
        for action in edge.symbolic_actions
        if action.get("target_binding", {}).get("kind") == "semantic_goal"
    }
    if semantic_phases == {"staging"}:
        return None

    fallback = next(
        (
            grounded.target_object_pose
            for grounded in grounded_actions
            if grounded.target_object_pose is not None
        ),
        None,
    )
    return _resolved_object_target_positions(
        arm_actions,
        fallback=fallback,
    )


def _physical_control_parts_for_assignments(
    env: Any,
    assignments: Sequence[str | None],
) -> list[str | None]:
    """Expose the physical control part behind each semantic arm assignment."""
    physical_parts: list[str | None] = []
    for assignment in assignments:
        if assignment not in {"left_arm", "right_arm"}:
            physical_parts.append(assignment)
            continue
        resolver = getattr(env, "get_agent_arm_control_part", None)
        if resolver is None:
            physical_parts.append(assignment)
            continue
        physical_parts.append(str(resolver(assignment == "left_arm")))
    return physical_parts


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
    """Remove scheduler and artifact controls before invoking atomic actions."""
    internal = {
        "action_module",
        "env",
        "episode_index",
        "observations",
        "regenerate",
        "runtime_graph_renderer",
        "runtime_run_id",
        "seed_task_graph",
        "semantic_step_settle_steps",
        "strict_serial",
    }
    return {key: value for key, value in kwargs.items() if key not in internal}
