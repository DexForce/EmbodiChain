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

"""Closed-loop executor for action-engine execution-program DAGs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
import logging
from threading import RLock
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import WorldState
from embodichain.utils import logger as project_logger
from embodichain.utils.logger import log_info, log_warning

from .actions import AtomicActionAdapter
from .grounding import ActionGrounder, LiveArrangementPlan, LivePlacementPlan
from .models import (
    ActionOutcome,
    ExecutionEdge,
    ExecutionProgram,
    ExecutionResult,
    GroundedAction,
    SemanticStep,
)
from .predicates import evaluate_predicate
from .recording import RuntimeRecorder

__all__ = ["ProgramExecutor"]


@dataclass
class _Candidate:
    feasible: torch.Tensor
    cost: torch.Tensor
    plans: dict[str, tuple[GroundedAction, ActionOutcome]]
    warnings: tuple[str, ...] = ()


@dataclass
class _EdgeResult:
    actions: list[torch.Tensor]
    failed: torch.Tensor
    grounded: list[GroundedAction]


_SPECULATIVE_LOG_LOCK = RLock()


@contextmanager
def _capture_speculative_warnings() -> Iterator[list[str]]:
    """Temporarily capture project warnings without changing its log level."""
    messages: list[str] = []
    collector = logging.Handler(level=logging.WARNING)
    collector.emit = lambda record: messages.append(record.getMessage())
    logger = project_logger.logger
    with _SPECULATIVE_LOG_LOCK:
        handlers = list(logger.handlers)
        propagate = logger.propagate
        try:
            logger.handlers[:] = [collector]
            logger.propagate = False
            yield messages
        finally:
            logger.handlers[:] = handlers
            logger.propagate = propagate


class ProgramExecutor:
    """Schedule, ground, plan, execute, and verify one immutable program."""

    def __init__(
        self,
        program: ExecutionProgram,
        env: Any,
        *,
        max_transitions: int = 1000,
        settle_steps: int = 10,
        record_runtime: bool = True,
        record_root: str | None = None,
    ) -> None:
        self.program = program
        self.env = env
        self.max_transitions = int(max_transitions)
        self.settle_steps = int(settle_steps)
        self.record_runtime = bool(record_runtime)
        self.record_root = record_root
        self.edges = {edge.id: edge for edge in program.edges}
        self.steps = {step.id: step for step in program.semantic_steps}
        self.step_by_edge = {
            edge_id: step
            for step in program.semantic_steps
            for edge_id in step.edge_ids
        }
        missing = set(self.edges) - set(self.step_by_edge)
        if missing:
            raise ValueError(
                "Every execution edge must belong to one semantic step; missing "
                f"{sorted(missing)}."
            )
        self.group_by_step = {
            str(step_id): group
            for group in program.allocation_groups
            for step_id in group.get("semantic_step_ids", ())
        }
        arrangement_steps = [
            step
            for step in program.semantic_steps
            if step.operator in {"arrange_line", "place_in_line"}
        ]
        arrangement_groups: dict[str, list[SemanticStep]] = {}
        for step in arrangement_steps:
            arrangement_groups.setdefault(step.parent_step_id, []).append(step)
        plans = [
            LiveArrangementPlan(env, steps) for steps in arrangement_groups.values()
        ]
        self.arrangements = {step.id: plan for plan in plans for step in plan.steps}
        # Retain the singular attribute as a convenient introspection hook for
        # the common one-arrangement case.
        self.arrangement = plans[0] if len(plans) == 1 else None
        placement_groups: dict[str, list[SemanticStep]] = {}
        for step in program.semantic_steps:
            if (
                step.operator == "place_relative"
                and step.goal.get("relation") == "inside"
                and isinstance(step.goal.get("reference_object"), str)
            ):
                placement_groups.setdefault(
                    str(step.goal["reference_object"]),
                    [],
                ).append(step)
        placement_plans = [
            LivePlacementPlan(env, steps)
            for steps in placement_groups.values()
            if len(steps) > 1
        ]
        self.placements = {
            step.id: plan for plan in placement_plans for step in plan.steps
        }
        self.adapter = AtomicActionAdapter(env)
        self.grounder = ActionGrounder(
            program,
            env,
            self.adapter.semantics,
            self.arrangements,
            self.placements,
        )
        self._step_states: dict[tuple[str, str], WorldState] = {}
        self._object_states: dict[tuple[str, str], WorldState] = {}
        self._object_owners: dict[str, list[str | None]] = {}
        self._arm_owners: dict[str, list[str | None]] = {
            "left_arm": [None] * int(env.num_envs),
            "right_arm": [None] * int(env.num_envs),
        }
        self._assignments: dict[str, list[str | None]] = {}
        self._candidate_cache: dict[tuple[str, str], _Candidate] = {}
        self._candidate_failures: dict[tuple[str, str], str] = {}
        self._candidate_diagnostics: dict[str, tuple[str, ...]] = {}
        self._reported_candidates: set[str] = set()
        self._targets: dict[str, torch.Tensor] = {}
        self._policies: dict[str, dict[str, Any]] = {}
        self._payload_initial: dict[str, dict[str, torch.Tensor]] = {}

    def run(
        self,
        *,
        run_id: str | None = None,
        episode_index: int = 0,
    ) -> ExecutionResult:
        """Execute ready edges until the DAG completes or raises a structural error."""
        self._reset_runtime_state()
        recorder = RuntimeRecorder(
            self.program,
            num_envs=int(self.env.num_envs),
            run_id=run_id,
            episode_index=episode_index,
            output_root=self.record_root,
            enabled=self.record_runtime,
        )
        aggregate_failed = torch.zeros(
            int(self.env.num_envs),
            dtype=torch.bool,
            device=self.env.device,
        )
        edge_failures: dict[str, torch.Tensor] = {}
        semantic_success: dict[str, torch.Tensor] = {}
        completed: set[str] = set()
        remaining = [edge.id for edge in self.program.edges]
        executed_actions: list[torch.Tensor] = []
        transitions = 0
        error_message = None
        try:
            while remaining:
                ready = [
                    self.edges[edge_id]
                    for edge_id in remaining
                    if set(self.edges[edge_id].depends_on) <= completed
                ]
                if not ready:
                    raise RuntimeError(
                        "Execution program is deadlocked: no remaining edge is ready."
                    )
                batch = self._pack_ready_edges(ready)
                blocked = {
                    edge.id: self._dependency_failures(edge, edge_failures)
                    for edge in batch
                }
                # A synchronized pair needs the same active rows. Execute a
                # healthy independent branch separately when its peer is blocked.
                if len(batch) == 2 and not torch.equal(
                    blocked[batch[0].id], blocked[batch[1].id]
                ):
                    batch = (batch[0],)
                transitions += len(batch)
                if transitions > self.max_transitions:
                    raise RuntimeError("Execution exceeded max_transitions.")

                if len(batch) == 2:
                    edge_results, _ = self._execute_parallel_pickups(
                        batch,
                        failed=blocked[batch[0].id],
                    )
                    for edge in batch:
                        result = edge_results[edge.id]
                        step = self.step_by_edge[edge.id]
                        active = ~blocked[edge.id]
                        recorder.edge(
                            edge.id,
                            step,
                            assignments=self._assignments[step.id],
                            grounded=result.grounded,
                            active=active,
                            failed=result.failed,
                            action_steps=len(result.actions),
                            diagnostics=self._edge_diagnostics(
                                step,
                                edge,
                                result.failed,
                            ),
                        )
                    # Both edge records describe the same synchronized command
                    # stream. Store it once in the returned execution trace.
                    executed_actions.extend(edge_results[batch[0].id].actions)
                    for edge in batch:
                        edge_failures[edge.id] = edge_results[edge.id].failed.clone()
                else:
                    edge = batch[0]
                    step = self.step_by_edge[edge.id]
                    branch_failed = blocked[edge.id]
                    self._ensure_assignment(step, branch_failed)
                    active = ~branch_failed
                    edge_result = self._execute_edge(edge, step, failed=branch_failed)
                    if self._is_cleanup_edge(edge):
                        # Cleanup degradation is observable in the record but does
                        # not invalidate an already achieved semantic relation.
                        next_failed = branch_failed
                    else:
                        next_failed = edge_result.failed
                    recorder.edge(
                        edge.id,
                        step,
                        assignments=self._assignments[step.id],
                        grounded=edge_result.grounded,
                        active=active,
                        failed=edge_result.failed,
                        action_steps=len(edge_result.actions),
                        diagnostics=self._edge_diagnostics(
                            step,
                            edge,
                            edge_result.failed,
                        ),
                    )
                    executed_actions.extend(edge_result.actions)
                    edge_failures[edge.id] = next_failed

                for edge in batch:
                    completed.add(edge.id)
                    remaining.remove(edge.id)
                    step = self.step_by_edge[edge.id]
                    if edge.id != step.edge_ids[-1]:
                        continue
                    verified_failed, step_success, observed = self._verify_step(
                        step, edge_failures[edge.id]
                    )
                    edge_failures[edge.id] = verified_failed
                    aggregate_failed |= ~step_success
                    semantic_success[step.id] = step_success
                    arrangement = self.arrangements.get(step.id)
                    if arrangement is not None:
                        arrangement.mark_completed(step.id, step_success)
                    recorder.step(
                        step,
                        step_success,
                        observed=observed,
                        target=self._targets.get(step.id),
                    )
            record_dir = recorder.finalize(~aggregate_failed)
        except BaseException as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            recorder.finalize(~aggregate_failed, error=error_message)
            raise
        finally:
            if error_message is not None:
                log_warning(f"Action Engine execution aborted: {error_message}")

        return ExecutionResult(
            actions=executed_actions,
            success=~aggregate_failed,
            semantic_success=semantic_success,
            record_dir=record_dir,
        )

    def _dependency_failures(
        self,
        edge: ExecutionEdge,
        failures: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return only failures that can reach this edge through the DAG."""
        result = torch.zeros(
            int(self.env.num_envs), dtype=torch.bool, device=self.env.device
        )
        for dependency in edge.depends_on:
            result |= failures[dependency]
        return result

    def _reset_runtime_state(self) -> None:
        self._step_states.clear()
        self._object_states.clear()
        self._object_owners.clear()
        for owners in self._arm_owners.values():
            owners[:] = [None] * int(self.env.num_envs)
        self._assignments.clear()
        self._candidate_cache.clear()
        self._candidate_failures.clear()
        self._candidate_diagnostics.clear()
        self._reported_candidates.clear()
        self._targets.clear()
        self._policies.clear()
        self._payload_initial.clear()

    def _pack_ready_edges(
        self,
        ready: Sequence[ExecutionEdge],
    ) -> tuple[ExecutionEdge, ...]:
        """Pack only a declared, resource-safe pair of independent PickUps."""
        first = ready[0]
        if not self._parallel_pickup_candidate(first):
            return (first,)
        first_step = self.step_by_edge[first.id]
        for second in ready[1:]:
            if not self._parallel_pickup_candidate(second):
                continue
            second_step = self.step_by_edge[second.id]
            if first_step.object_uid == second_step.object_uid:
                continue
            shared = set(first.resources) & set(second.resources)
            same_group = self.group_by_step.get(first_step.id) is not None and (
                self.group_by_step.get(first_step.id)
                is self.group_by_step.get(second_step.id)
            )
            if same_group:
                # A shared destination workspace constrains transport/place,
                # not two independent pickups declared by this group.
                conflicts = {
                    item
                    for item in shared
                    if item != "arm:auto" and not item.startswith("workspace:")
                }
            else:
                conflicts = shared - {"arm:auto"}
            if conflicts:
                continue
            required_opposite = (
                first_step.actor.get("mode") == "required"
                and second_step.actor.get("mode") == "required"
                and first_step.actor.get("arm") != second_step.actor.get("arm")
            )
            if same_group or required_opposite:
                return first, second
        return (first,)

    def _parallel_pickup_candidate(self, edge: ExecutionEdge) -> bool:
        if len(edge.actions) != 1:
            return False
        step = self.step_by_edge[edge.id]
        return edge.actions[0].get(
            "atomic_action_class"
        ) == "PickUp" and step.actor.get("mode") in {"auto", "required"}

    def _ensure_assignment(
        self,
        step: SemanticStep,
        failed: torch.Tensor,
        *,
        allow_rematch: bool = True,
    ) -> None:
        if step.id in self._assignments:
            return
        mode = str(step.actor.get("mode", "auto"))
        if mode == "coordinated":
            self._assignments[step.id] = [
                (
                    None
                    if bool(failed[index])
                    or self._arm_owners["left_arm"][index] is not None
                    or self._arm_owners["right_arm"][index] is not None
                    else "coordinated"
                )
                for index in range(len(failed))
            ]
            return
        if mode == "required":
            arm = str(step.actor["arm"])
            candidate = self._candidate(step, arm, failed)
            self._assignments[step.id] = [
                (
                    arm
                    if not bool(failed[index]) and bool(candidate.feasible[index])
                    else None
                )
                for index in range(len(failed))
            ]
            self._report_candidates(step, (candidate,))
            return

        left = self._candidate(step, "left_arm", failed)
        right = self._candidate(step, "right_arm", failed)
        owners = self._object_owners.get(step.object_uid, [None] * len(failed))
        assignments: list[str | None] = []
        selection_failed = torch.zeros_like(failed)
        for env_id in range(len(failed)):
            if bool(failed[env_id]):
                assignments.append(None)
                continue
            if owners[env_id] is not None:
                owner = str(owners[env_id])
                owned = left if owner == "left_arm" else right
                if bool(owned.feasible[env_id]):
                    assignments.append(owner)
                else:
                    assignments.append(None)
                    selection_failed[env_id] = True
                continue
            left_ok = bool(left.feasible[env_id])
            right_ok = bool(right.feasible[env_id])
            if left_ok and (
                not right_ok or float(left.cost[env_id]) <= float(right.cost[env_id])
            ):
                assignments.append("left_arm")
            elif right_ok:
                assignments.append("right_arm")
            else:
                assignments.append(None)
                selection_failed[env_id] = True

        if (
            allow_rematch
            and bool(selection_failed.any())
            and step.id in self.arrangements
            and step.goal.get("slot_constraint") == "free_reassignable"
            and bool(self._rematch_arrangement(step, selection_failed, failed).any())
        ):
            self._assignments.pop(step.id, None)
            self._ensure_assignment(step, failed, allow_rematch=False)
            return
        self._assignments[step.id] = assignments
        self._report_candidates(step, (left, right))

    def _candidate(
        self,
        step: SemanticStep,
        arm: str,
        failed: torch.Tensor,
    ) -> _Candidate:
        """Plan the complete semantic suffix before fixing an arm."""
        if step.actor.get("mode") == "required" and str(step.actor.get("arm")) != arm:
            return _Candidate(
                feasible=torch.zeros_like(failed),
                cost=torch.full(
                    failed.shape,
                    torch.inf,
                    dtype=torch.float32,
                    device=self.env.device,
                ),
                plans={},
            )
        cached = self._candidate_cache.get((step.id, arm))
        if cached is not None:
            return _Candidate(
                feasible=cached.feasible & ~failed,
                cost=cached.cost,
                plans=cached.plans,
                warnings=cached.warnings,
            )
        feasible = ~failed.clone() & ~self._resource_conflicts(step, arm)
        cost = torch.zeros(
            int(self.env.num_envs),
            dtype=torch.float32,
            device=self.env.device,
        )
        state = self._state_for(step, arm)
        reference_eef_pose = None
        plans: dict[str, tuple[GroundedAction, ActionOutcome]] = {}
        warnings: list[str] = []
        try:
            with _capture_speculative_warnings() as captured:
                for edge_id in step.edge_ids:
                    edge = self.edges[edge_id]
                    if len(edge.actions) != 1:
                        raise ValueError(
                            "Auto/required arm candidates require one action per edge."
                        )
                    grounded = self.grounder.ground(
                        edge.actions[0],
                        step,
                        arm=arm,
                        state=state,
                        reference_eef_pose=reference_eef_pose,
                    )
                    if grounded.action_class == "PickUp":
                        grounded = self._with_downstream_targets(
                            step, edge_id, arm, state, grounded
                        )
                    outcome = self.adapter.plan(grounded, state)
                    plans[edge_id] = (grounded, outcome)
                    feasible &= outcome.success
                    cost += outcome.cost
                    state = outcome.next_state
                    target = outcome.grounded.target_object_pose
                    if isinstance(target, torch.Tensor):
                        reference_eef_pose = self._eef_target(outcome)
                    if not bool((feasible & ~failed).any()):
                        break
                warnings.extend(captured)
        except Exception as exc:
            self._candidate_failures[(step.id, arm)] = f"{type(exc).__name__}: {exc}"
            feasible = torch.zeros_like(failed)
            cost[:] = torch.inf
        candidate = _Candidate(
            feasible=feasible,
            cost=cost,
            plans=plans,
            warnings=tuple(warnings),
        )
        self._candidate_cache[(step.id, arm)] = candidate
        return _Candidate(
            feasible=feasible & ~failed,
            cost=cost,
            plans=plans,
            warnings=tuple(warnings),
        )

    def _report_candidates(
        self,
        step: SemanticStep,
        candidates: Sequence[_Candidate],
    ) -> None:
        if step.id in self._reported_candidates:
            return
        warning_count = sum(len(item.warnings) for item in candidates)
        failures = [
            message
            for (step_id, _), message in self._candidate_failures.items()
            if step_id == step.id
        ]
        diagnostics = tuple(
            dict.fromkeys(message for item in candidates for message in item.warnings)
        ) + tuple(dict.fromkeys(failures))
        diagnostics = tuple(dict.fromkeys(diagnostics))
        if diagnostics:
            self._candidate_diagnostics[step.id] = diagnostics
        if warning_count or failures:
            feasible = ", ".join(
                f"{int(item.feasible.sum())}/{len(item.feasible)}"
                for item in candidates
            )
            log_info(
                f"Speculative arm candidates for {step.id}: feasible=[{feasible}], "
                f"suppressed_warnings={warning_count}, exceptions={len(failures)}."
            )
            for message in diagnostics[:3]:
                log_warning(f"Candidate planning for {step.id}: {message}")
        self._reported_candidates.add(step.id)

    def _edge_diagnostics(
        self,
        step: SemanticStep,
        edge: ExecutionEdge,
        failed: torch.Tensor,
    ) -> tuple[str, ...]:
        if edge.id != step.edge_ids[0] or not bool(failed.any()):
            return ()
        return self._candidate_diagnostics.get(step.id, ())

    def _with_downstream_targets(
        self,
        step: SemanticStep,
        pickup_edge_id: str,
        arm: str,
        state: WorldState,
        grounded: GroundedAction,
    ) -> GroundedAction:
        """Screen grasp poses against every later held-object target."""
        targets = []
        start = step.edge_ids.index(pickup_edge_id) + 1
        for edge_id in step.edge_ids[start:]:
            edge = self.edges[edge_id]
            if len(edge.actions) != 1:
                continue
            action = edge.actions[0]
            if action.get("atomic_action_class") != "MoveHeldObject":
                continue
            future = self.grounder.ground(action, step, arm=arm, state=state)
            if future.target_object_pose is not None:
                targets.append(future.target_object_pose)
        if not targets:
            return grounded
        return replace(
            grounded,
            cfg={
                **grounded.cfg,
                "downstream_object_target_poses": tuple(targets),
            },
        )

    def _eef_target(self, outcome: ActionOutcome) -> torch.Tensor | None:
        state = outcome.next_state
        object_target = outcome.grounded.target_object_pose
        if object_target is not None and state.held_object is not None:
            object_to_eef = state.held_object.object_to_eef.to(
                device=object_target.device,
                dtype=object_target.dtype,
            )
            return torch.bmm(object_target, object_to_eef)
        if state.held_object is not None:
            return state.held_object.grasp_xpos
        target = outcome.grounded.target
        return getattr(target, "xpos", None)

    def _state_for(self, step: SemanticStep, arm: str) -> WorldState:
        """Refresh qpos while retaining state inside one semantic step."""
        cached = self._step_states.get((step.id, arm))
        live_qpos = self.env.robot.get_qpos().clone()
        if cached is None:
            return WorldState(last_qpos=live_qpos)
        return WorldState(
            last_qpos=live_qpos,
            held_object=cached.held_object,
            coordinated_held_object=cached.coordinated_held_object,
        )

    def _resource_conflicts(
        self,
        step: SemanticStep,
        arm: str,
    ) -> torch.Tensor:
        object_owners = self._object_owners.get(
            step.object_uid, [None] * int(self.env.num_envs)
        )
        arm_owners = self._arm_owners[arm]
        return torch.tensor(
            [
                (object_owner not in {None, arm})
                or (arm_owner not in {None, step.object_uid})
                for object_owner, arm_owner in zip(object_owners, arm_owners)
            ],
            dtype=torch.bool,
            device=self.env.device,
        )

    def _update_ownership(
        self,
        step: SemanticStep,
        arm: str,
        action_class: str,
        state: WorldState,
        successful: torch.Tensor,
    ) -> None:
        owners = self._object_owners.setdefault(
            step.object_uid, [None] * int(self.env.num_envs)
        )
        if action_class == "Place":
            for env_id in torch.nonzero(successful, as_tuple=False).flatten().tolist():
                if owners[env_id] == arm:
                    owners[env_id] = None
                if self._arm_owners[arm][env_id] == step.object_uid:
                    self._arm_owners[arm][env_id] = None
            if arm not in owners:
                self._object_states.pop((step.object_uid, arm), None)
            return
        if state.held_object is None or (
            action_class == "PickUp" and not bool(successful.any())
        ):
            return
        self._object_states[(step.object_uid, arm)] = state
        if action_class == "PickUp":
            for env_id in torch.nonzero(successful, as_tuple=False).flatten().tolist():
                owners[env_id] = arm
                self._arm_owners[arm][env_id] = step.object_uid

    def _rematch_arrangement(
        self,
        trigger_step: SemanticStep,
        trigger: torch.Tensor,
        failed: torch.Tensor,
    ) -> torch.Tensor:
        """Globally rematch unfinished objects to feasible free slots."""
        from scipy.optimize import linear_sum_assignment

        arrangement = self.arrangements[trigger_step.id]
        changed = torch.zeros_like(trigger)
        for env_id in torch.nonzero(trigger, as_tuple=False).flatten().tolist():
            step_ids = arrangement.remaining(env_id)
            slots = arrangement.available_slots(env_id)
            if len(step_ids) != len(slots):
                continue
            original = {
                step_id: int(arrangement.assignments[step_id][env_id])
                for step_id in step_ids
            }
            costs = np.full((len(step_ids), len(slots)), np.inf, dtype=np.float64)
            isolate = torch.ones_like(failed)
            isolate[env_id] = failed[env_id]
            for row, step_id in enumerate(step_ids):
                for column, slot_id in enumerate(slots):
                    arrangement.assignments[step_id][env_id] = slot_id
                    self._candidate_cache.pop((step_id, "left_arm"), None)
                    self._candidate_cache.pop((step_id, "right_arm"), None)
                    arm_costs = []
                    for arm in ("left_arm", "right_arm"):
                        candidate = self._candidate(self.steps[step_id], arm, isolate)
                        if bool(candidate.feasible[env_id]):
                            arm_costs.append(float(candidate.cost[env_id]))
                    if arm_costs:
                        costs[row, column] = min(arm_costs)
                arrangement.assignments[step_id][env_id] = original[step_id]
                self._candidate_cache.pop((step_id, "left_arm"), None)
                self._candidate_cache.pop((step_id, "right_arm"), None)
            if not np.isfinite(costs).any(axis=1).all():
                continue
            rows, columns = linear_sum_assignment(
                np.where(np.isfinite(costs), costs, 1.0e12)
            )
            if not np.isfinite(costs[rows, columns]).all():
                continue
            arrangement.assign(
                env_id,
                {
                    step_ids[int(row)]: slots[int(column)]
                    for row, column in zip(rows, columns)
                },
            )
            changed[env_id] = True
        return changed

    def _execute_edge(
        self,
        edge: ExecutionEdge,
        step: SemanticStep,
        *,
        failed: torch.Tensor,
    ) -> _EdgeResult:
        if len(edge.actions) == 1 and edge.actions[0].get("atomic_action_class") in {
            "CoordinatedPickment",
            "CoordinatedPlacement",
        }:
            return self._execute_coordinated(edge, step, failed)
        if len(edge.actions) == 2:
            return self._execute_explicit_dual(edge, step, failed)
        if len(edge.actions) != 1:
            raise ValueError(
                f"Edge {edge.id!r} must contain one action or an explicit dual pair."
            )
        assignments = self._assignments[step.id]
        outcomes: dict[str, ActionOutcome | None] = {
            "left_arm": None,
            "right_arm": None,
        }
        masks = {
            arm: torch.tensor(
                [assignment == arm for assignment in assignments],
                dtype=torch.bool,
                device=self.env.device,
            )
            & ~failed
            for arm in outcomes
        }
        grounded_items: list[GroundedAction] = []
        action_class = str(edge.actions[0]["atomic_action_class"])
        for arm in outcomes:
            if not bool(masks[arm].any()):
                continue
            state = self._state_for(step, arm)
            if action_class == "PickUp":
                candidate = self._candidate_cache.get((step.id, arm))
                planned = None if candidate is None else candidate.plans.get(edge.id)
                if planned is None:
                    raise RuntimeError(
                        f"Selected arm {arm!r} for {step.id!r} has no cached "
                        f"PickUp plan for edge {edge.id!r}."
                    )
                grounded, outcome = planned
            else:
                # Re-ground transport and placement from live simulator state;
                # only the expensive, immediately executed PickUp is reusable.
                grounded = self.grounder.ground(
                    edge.actions[0], step, arm=arm, state=state
                )
                outcome = self.adapter.plan(grounded, state)
            outcomes[arm] = outcome
            grounded_items.append(grounded)
            self._step_states[(step.id, arm)] = outcome.next_state
            self._remember_target(step, grounded)
        if not grounded_items:
            return _EdgeResult([], torch.ones_like(failed), [])
        trajectory, action_success = self.adapter.combine(outcomes, masks)
        assigned = masks["left_arm"] | masks["right_arm"]
        active = assigned & action_success & ~failed
        actions = self.adapter.execute_trajectory(trajectory, active=active)
        physical_failed = torch.zeros_like(failed)
        for arm, outcome in outcomes.items():
            if outcome is not None:
                successful = masks[arm] & outcome.success & active
                if action_class == "PickUp":
                    physical = self._physical_pickup(
                        step.object_uid, arm, outcome.next_state, successful
                    )
                    physical_failed |= successful & ~physical
                    successful = physical
                self._update_ownership(
                    step,
                    arm,
                    action_class,
                    outcome.next_state,
                    successful,
                )
                if action_class == "Press":
                    semantic_states = getattr(
                        self.env,
                        "action_engine_semantic_states",
                        None,
                    )
                    if semantic_states is None:
                        semantic_states = {}
                        self.env.action_engine_semantic_states = semantic_states
                    semantic_states[(step.object_uid, "pressed")] = successful.clone()
        edge_failed = (
            failed
            | (~failed & ~assigned)
            | (assigned & ~action_success)
            | physical_failed
        )
        return _EdgeResult(actions, edge_failed, grounded_items)

    def _physical_pickup(
        self,
        uid: str,
        arm: str,
        state: WorldState,
        attempted: torch.Tensor,
    ) -> torch.Tensor:
        """Confirm a planned grasp against live geometry before reserving an arm."""
        owners = list(self._object_owners.get(uid, [None] * int(self.env.num_envs)))
        for env_id in torch.nonzero(attempted, as_tuple=False).flatten().tolist():
            owners[env_id] = arm
        states = dict(self._object_states)
        states[(uid, arm)] = state
        return attempted & evaluate_predicate(
            self.env,
            {"type": "object_held", "object": uid},
            held_owners={**self._object_owners, uid: owners},
            held_states=states,
        )

    def _execute_coordinated(
        self,
        edge: ExecutionEdge,
        step: SemanticStep,
        failed: torch.Tensor,
    ) -> _EdgeResult:
        assigned = torch.tensor(
            [item == "coordinated" for item in self._assignments[step.id]],
            dtype=torch.bool,
            device=self.env.device,
        )
        active = assigned & ~failed
        if not bool(active.any()):
            return _EdgeResult([], failed | (~failed & ~assigned), [])
        if edge.actions[0]["atomic_action_class"] == "CoordinatedPickment":
            self._capture_payloads(step)
        state = self._state_for(step, "coordinated")
        grounded = self.grounder.ground(
            edge.actions[0],
            step,
            arm="coordinated",
            state=state,
        )
        outcome = self.adapter.plan(grounded, state)
        self._step_states[(step.id, "coordinated")] = outcome.next_state
        self._remember_target(step, grounded)
        successful = active & outcome.success
        actions = self.adapter.execute_trajectory(
            outcome.trajectory,
            active=successful,
        )
        return _EdgeResult(
            actions,
            failed | (~failed & ~assigned) | (active & ~outcome.success),
            [grounded],
        )

    def _execute_explicit_dual(
        self,
        edge: ExecutionEdge,
        step: SemanticStep,
        failed: torch.Tensor,
    ) -> _EdgeResult:
        assigned = torch.tensor(
            [item == "coordinated" for item in self._assignments[step.id]],
            dtype=torch.bool,
            device=self.env.device,
        )
        if not bool((assigned & ~failed).any()):
            return _EdgeResult([], failed | (~failed & ~assigned), [])
        outcomes: dict[str, ActionOutcome | None] = {
            "left_arm": None,
            "right_arm": None,
        }
        masks = {arm: assigned & ~failed for arm in outcomes}
        grounded_items = []
        coordinated_state = self._state_for(step, "coordinated")
        for action in edge.actions:
            actor = action.get("actor", {})
            arm = str(actor.get("arm", ""))
            if arm not in outcomes or outcomes[arm] is not None:
                raise ValueError(
                    f"Explicit dual edge {edge.id!r} must bind each arm once."
                )
            state = self._step_states.get((step.id, arm))
            if state is None:
                state = coordinated_state
            else:
                state = self._state_for(step, arm)
            grounded = self.grounder.ground(
                action,
                step,
                arm=arm,
                state=state,
            )
            outcome = self.adapter.plan(grounded, state)
            outcomes[arm] = outcome
            self._step_states[(step.id, arm)] = outcome.next_state
            grounded_items.append(grounded)
        trajectory, action_success = self.adapter.combine(outcomes, masks)
        active = assigned & ~failed & action_success
        actions = self.adapter.execute_trajectory(trajectory, active=active)
        return _EdgeResult(
            actions,
            failed | (~failed & ~assigned) | (assigned & ~action_success),
            grounded_items,
        )

    def _execute_parallel_pickups(
        self,
        edges: Sequence[ExecutionEdge],
        *,
        failed: torch.Tensor,
    ) -> tuple[dict[str, _EdgeResult], torch.Tensor]:
        steps = [self.step_by_edge[edge.id] for edge in edges]
        candidates = {
            (step.id, arm): self._candidate(step, arm, failed)
            for step in steps
            for arm in ("left_arm", "right_arm")
        }
        for step in steps:
            self._report_candidates(
                step,
                (
                    candidates[(step.id, "left_arm")],
                    candidates[(step.id, "right_arm")],
                ),
            )
        assignments = {step.id: [None] * len(failed) for step in steps}
        selection_failed = torch.zeros_like(failed)
        permutations = (
            ("left_arm", "right_arm"),
            ("right_arm", "left_arm"),
        )
        for env_id in range(len(failed)):
            if bool(failed[env_id]):
                continue
            ranked = []
            for first_arm, second_arm in permutations:
                first = candidates[(steps[0].id, first_arm)]
                second = candidates[(steps[1].id, second_arm)]
                feasible = bool(first.feasible[env_id] and second.feasible[env_id])
                cost = float(first.cost[env_id] + second.cost[env_id])
                ranked.append((not feasible, cost, first_arm, second_arm))
            ranked.sort()
            infeasible, _, first_arm, second_arm = ranked[0]
            if infeasible:
                selection_failed[env_id] = True
                continue
            assignments[steps[0].id][env_id] = first_arm
            assignments[steps[1].id][env_id] = second_arm
        self._assignments.update(assignments)

        base_failed = failed | selection_failed
        results = {edge.id: _EdgeResult([], base_failed.clone(), []) for edge in edges}
        for first_arm, second_arm in permutations:
            partition = torch.tensor(
                [
                    assignments[steps[0].id][env_id] == first_arm
                    and assignments[steps[1].id][env_id] == second_arm
                    for env_id in range(len(failed))
                ],
                dtype=torch.bool,
                device=self.env.device,
            )
            if not bool(partition.any()):
                continue
            outcomes: dict[str, ActionOutcome | None] = {
                "left_arm": None,
                "right_arm": None,
            }
            masks = {
                "left_arm": partition,
                "right_arm": partition,
            }
            edge_by_arm = {first_arm: edges[0], second_arm: edges[1]}
            for arm, edge in edge_by_arm.items():
                step = self.step_by_edge[edge.id]
                grounded, outcome = candidates[(step.id, arm)].plans[edge.id]
                outcomes[arm] = outcome
                self._step_states[(step.id, arm)] = outcome.next_state
                results[edge.id].grounded.append(grounded)
            trajectory, action_success = self.adapter.combine(outcomes, masks)
            active = partition & ~base_failed & action_success
            commands = self.adapter.execute_trajectory(trajectory, active=active)
            for arm, edge in edge_by_arm.items():
                step = self.step_by_edge[edge.id]
                outcome = outcomes[arm]
                assert outcome is not None
                attempted = active & outcome.success
                physical = self._physical_pickup(
                    step.object_uid, arm, outcome.next_state, attempted
                )
                results[edge.id].failed |= partition & ~physical
                self._update_ownership(
                    step,
                    arm,
                    "PickUp",
                    outcome.next_state,
                    physical,
                )
            for edge in edges:
                # Both edge records refer to this one synchronized stream. The
                # run loop adds only the first copy to its returned trace.
                results[edge.id].actions.extend(commands)
        aggregate_failed = torch.zeros_like(failed)
        for result in results.values():
            aggregate_failed |= result.failed
        return results, aggregate_failed

    def _remember_target(
        self,
        step: SemanticStep,
        grounded: GroundedAction,
    ) -> None:
        target = grounded.target_object_pose
        if target is not None:
            self._targets[step.id] = target[:, :3, 3].clone()
            self._policies[step.id] = grounded.motion_policy

    def _verify_step(
        self,
        step: SemanticStep,
        failed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be non-negative.")
        if self.settle_steps and bool((~failed).any()):
            self.env.sim.update(step=self.settle_steps)
        entity = self.env.sim.get_rigid_object(step.object_uid)
        if entity is None:
            raise ValueError(f"Unknown semantic object {step.object_uid!r}.")
        observed = entity.get_local_pose(to_matrix=True)[:, :3, 3]
        active = ~failed
        if not bool(active.any()):
            success = torch.zeros_like(failed)
            log_info(f"Skipped verification for {step.id}: no active environments.")
            return failed, success, observed
        relation = str(step.goal.get("relation", ""))
        reference = step.goal.get("reference_object")
        postcondition_type = step.postcondition.get("type")
        if postcondition_type == "object_held":
            # A planned hover target is not evidence that the object remains
            # grasped. Verify live TCP/object geometry and gripper closure.
            satisfied = evaluate_predicate(
                self.env,
                step.postcondition,
                held_owners=self._object_owners,
                held_states=self._object_states,
            )
        elif postcondition_type in {
            "held_by_both_grippers",
            "object_held_by_both_grippers",
        }:
            satisfied = evaluate_predicate(
                self.env,
                step.postcondition,
                coordinated_state=self._step_states.get((step.id, "coordinated")),
            )
        elif postcondition_type == "pressed":
            satisfied = evaluate_predicate(self.env, step.postcondition)
        elif relation == "inside" and isinstance(reference, str):
            satisfied = evaluate_predicate(
                self.env,
                {
                    "type": "object_in_container",
                    "object": step.object_uid,
                    "container": reference,
                },
            )
        elif relation in {"on", "on_top", "on_top_of"} and isinstance(reference, str):
            satisfied = evaluate_predicate(
                self.env,
                {
                    "type": "object_on_object",
                    "object": step.object_uid,
                    "support": reference,
                },
            )
        elif step.id in self._targets:
            target = self._targets[step.id].to(
                device=observed.device,
                dtype=observed.dtype,
            )
            policy = self._policies.get(step.id, {})
            tolerance = float(
                policy.get(
                    "postcondition_tolerance",
                    0.05,
                )
            )
            arrangement = self.arrangements.get(step.id)
            if arrangement is not None:
                # Line membership is a planar relation. Height changes after
                # release (for example a can settling onto another stable face)
                # must not invalidate an otherwise correct row placement.
                delta = torch.abs(observed - target)
                axis_tolerance = float(
                    policy.get("line_axis_tolerance", max(tolerance, 0.06))
                )
                perpendicular_tolerance = float(
                    policy.get(
                        "line_perpendicular_tolerance",
                        max(tolerance, 0.06),
                    )
                )
                satisfied = (delta[:, arrangement.axis_index] <= axis_tolerance) & (
                    delta[:, arrangement.perpendicular_index] <= perpendicular_tolerance
                )
            else:
                satisfied = (
                    torch.linalg.vector_norm(observed - target, dim=-1) <= tolerance
                )
        else:
            satisfied = evaluate_predicate(self.env, step.postcondition)
        if step.goal.get("payloads"):
            satisfied &= self._verify_payloads(step)
        success = active & satisfied
        failed = failed | (active & ~satisfied)
        log_info(
            f"Verified {step.id}: {int(success.sum())}/{len(success)} envs succeeded."
        )
        return failed, success, observed

    def _capture_payloads(self, step: SemanticStep) -> None:
        if step.id in self._payload_initial:
            return
        carrier = self._entity_pose(step.object_uid)
        record = {"carrier_rotation": carrier[:, :3, :3].clone()}
        for payload in step.goal.get("payloads", []):
            uid = str(payload["object"])
            record[uid] = torch.bmm(torch.linalg.inv(carrier), self._entity_pose(uid))
        self._payload_initial[step.id] = record

    def _verify_payloads(self, step: SemanticStep) -> torch.Tensor:
        record = self._payload_initial.get(step.id)
        if record is None:
            return torch.zeros(
                int(self.env.num_envs),
                dtype=torch.bool,
                device=self.env.device,
            )
        carrier = self._entity_pose(step.object_uid)
        initial_up = record["carrier_rotation"][:, :3, 2]
        live_up = carrier[:, :3, 2]
        tilt_ok = torch.sum(initial_up * live_up, dim=-1) >= 0.94
        result = tilt_ok
        carrier_entity = self.env.sim.get_rigid_object(step.object_uid)
        for payload in step.goal["payloads"]:
            uid = str(payload["object"])
            expected = torch.bmm(carrier, record[uid])
            observed = self._entity_pose(uid)
            drift_ok = (
                torch.linalg.vector_norm(
                    observed[:, :3, 3] - expected[:, :3, 3],
                    dim=-1,
                )
                <= 0.08
            )
            support_ok = torch.ones_like(drift_ok)
            for env_id in range(int(self.env.num_envs)):
                vertices = carrier_entity.get_vertices(
                    env_ids=[env_id],
                    scale=True,
                )
                if isinstance(vertices, (list, tuple)):
                    vertices = vertices[0]
                vertices = torch.as_tensor(
                    vertices,
                    dtype=carrier.dtype,
                    device=carrier.device,
                )
                world = (
                    vertices @ carrier[env_id, :3, :3].transpose(0, 1)
                    + carrier[env_id, :3, 3]
                )
                position = observed[env_id, :2, 3]
                lower = world[:, :2].min(dim=0).values - 0.015
                upper = world[:, :2].max(dim=0).values + 0.015
                support_ok[env_id] = bool(
                    torch.all(position >= lower) and torch.all(position <= upper)
                )
            result &= drift_ok & support_ok
        return result

    def _entity_pose(self, uid: str) -> torch.Tensor:
        entity = self.env.sim.get_rigid_object(uid)
        if entity is None:
            raise ValueError(f"Unknown rigid object {uid!r}.")
        pose = torch.as_tensor(
            entity.get_local_pose(to_matrix=True),
            dtype=torch.float32,
            device=self.env.device,
        )
        if pose.ndim == 2:
            pose = pose.unsqueeze(0).repeat(int(self.env.num_envs), 1, 1)
        return pose

    @staticmethod
    def _is_cleanup_edge(edge: ExecutionEdge) -> bool:
        for action in edge.actions:
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
