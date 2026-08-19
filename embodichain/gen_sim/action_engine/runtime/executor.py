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
from copy import deepcopy
from dataclasses import dataclass, field, replace
import logging
from threading import RLock
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_engine.config import (
    ArmSelectionPolicyCfg,
    RuntimePolicyCfg,
    default_runtime_policy,
    runtime_policy_hash,
)
from embodichain.gen_sim.action_engine.domain import normalize_placement_relation
from embodichain.gen_sim.action_engine.orientation import (
    AlignAxisConstraint,
    MatchRotationConstraint,
    compile_orientation_constraint,
)
from embodichain.lab.sim.atomic_actions import (
    HeldObjectState,
    SceneProvider,
    StateDelta,
)
from embodichain.utils import logger as project_logger
from embodichain.utils.logger import log_info, log_warning

from .actions import AtomicActionAdapter
from .frames import DIRECTIONAL_RELATIONS, robot_frame_axes
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
from .recovery import RuntimeGraph
from .robot_parts import arm_control_part
from .state import ExecutionState

__all__ = ["ProgramExecutor"]


@dataclass
class _Candidate:
    feasible: torch.Tensor
    cost: torch.Tensor
    plans: dict[str, tuple[GroundedAction, ActionOutcome]]
    score_components: dict[str, torch.Tensor] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    blockers: tuple[dict[str, Any], ...] = ()


@dataclass
class _EdgeResult:
    actions: list[torch.Tensor]
    failed: torch.Tensor
    grounded: list[GroundedAction]
    planner_traces: list[dict[str, Any]] = field(default_factory=list)
    executed: torch.Tensor | None = None


@dataclass(frozen=True)
class _SupportRelation:
    support_uid: str
    semantic_step_id: str


@dataclass
class _PlacementRecoveryResult:
    failed: torch.Tensor
    succeeded: torch.Tensor
    observed: torch.Tensor
    actions: list[torch.Tensor]
    failure_events: list[dict[str, Any]] = field(default_factory=list)
    covered_failures: torch.Tensor | None = None


def _score_arm_candidate(
    *,
    arm: str,
    motion_cost: torch.Tensor,
    source_pose: torch.Tensor,
    target_pose: torch.Tensor | None,
    workspace_center_xy: torch.Tensor,
    workspace_half_width: torch.Tensor,
    robot_lateral_axis: torch.Tensor,
    policy: ArmSelectionPolicyCfg,
) -> dict[str, torch.Tensor]:
    """Combine motion length with soft, table-normalized cross-zone costs."""
    arm_sign = 1.0 if arm == "left_arm" else -1.0
    deadband = workspace_half_width * float(policy.crossing_deadband_ratio)

    def crossing(pose: torch.Tensor | None, weight: float) -> torch.Tensor:
        if pose is None:
            return torch.zeros_like(motion_cost)
        lateral = torch.sum(
            (pose[:, :2, 3] - workspace_center_xy) * robot_lateral_axis,
            dim=1,
        )
        wrong_side_depth = torch.clamp(
            -arm_sign * lateral - deadband,
            min=0.0,
        )
        return weight * torch.square(wrong_side_depth / workspace_half_width)

    normalized_motion = motion_cost / float(policy.motion_cost_scale)
    pickup_penalty = crossing(source_pose, float(policy.pickup_crossing_weight))
    placement_penalty = crossing(
        target_pose,
        float(policy.placement_crossing_weight),
    )
    return {
        "motion_cost": motion_cost,
        "normalized_motion_cost": normalized_motion,
        "pickup_crossing_penalty": pickup_penalty,
        "placement_crossing_penalty": placement_penalty,
        "total_cost": normalized_motion + pickup_penalty + placement_penalty,
    }


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
        max_transitions: int | None = None,
        settle_steps: int | None = None,
        record_runtime: bool = True,
        record_root: str | None = None,
        runtime_policy: RuntimePolicyCfg | None = None,
        capability_registry: Any | None = None,
        scene_provider: SceneProvider | None = None,
    ) -> None:
        self.program = program
        self.env = env
        self.record_runtime = bool(record_runtime)
        self.record_root = record_root
        if runtime_policy is None:
            profile = str(getattr(env, "agent_robot_profile", "dual_ur10"))
            runtime_policy = default_runtime_policy(profile)
        if not isinstance(runtime_policy, RuntimePolicyCfg):
            raise TypeError("ProgramExecutor runtime_policy must be RuntimePolicyCfg.")
        self.runtime_policy = runtime_policy
        self.capability_registry = capability_registry
        self.env.runtime_policy = runtime_policy
        execution = runtime_policy.execution
        self.max_transitions = int(
            execution["max_transitions"] if max_transitions is None else max_transitions
        )
        self.settle_steps = int(
            execution["semantic_step_settle_steps"]
            if settle_steps is None
            else settle_steps
        )
        self.max_retries_per_action = int(execution["max_retries_per_action"])
        self.support_stability_samples = int(execution["support_stability_samples"])
        self.support_stability_interval_steps = int(
            execution["support_stability_interval_steps"]
        )
        self.support_linear_velocity_tolerance = float(
            execution["support_linear_velocity_tolerance"]
        )
        self.support_angular_velocity_tolerance = float(
            execution["support_angular_velocity_tolerance"]
        )
        self.placement_recovery_attempts = int(
            runtime_policy.grounding["placement"]["recovery_attempts"]
        )
        self.runtime_graph = (
            RuntimeGraph(
                program.seed_graph,
                num_envs=int(env.num_envs),
                max_retries=self.max_retries_per_action,
                max_revisions=int(execution["max_graph_revisions"]),
                max_recovery_actions=int(execution["max_recovery_actions"]),
                registry=capability_registry,
            )
            if program.seed_graph is not None
            else None
        )
        self.retry_count = 0
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
        self._completion_only_dependencies = self._completion_only_dependency_edges()
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
        arrangement_policy = runtime_policy.grounding["arrangement"]
        plans = [
            LiveArrangementPlan(
                env,
                steps,
                slot_margin=float(arrangement_policy["slot_margin"]),
                minimum_spacing=float(arrangement_policy["minimum_spacing"]),
                clearance=float(arrangement_policy["layout_clearance"]),
                row_search_step=float(arrangement_policy["row_search_step"]),
                row_search_radius=float(arrangement_policy["row_search_radius"]),
            )
            for steps in arrangement_groups.values()
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
            LivePlacementPlan(
                env,
                steps,
                clearance=float(runtime_policy.grounding["placement"]["clearance"]),
            )
            for steps in placement_groups.values()
            if len(steps) > 1
        ]
        self.placements = {
            step.id: plan for plan in placement_plans for step in plan.steps
        }
        self.adapter = AtomicActionAdapter(
            env,
            grasp_policy=runtime_policy.grasp,
            planner_policy=runtime_policy.planner,
            capability_registry=capability_registry,
            scene_provider=scene_provider,
        )
        self.grounder = ActionGrounder(
            program,
            env,
            self.adapter.semantics,
            self.arrangements,
            self.placements,
            runtime_policy=runtime_policy,
            capability_registry=capability_registry,
        )
        self._step_states: dict[tuple[str, str], ExecutionState] = {}
        self._object_states: dict[tuple[str, str], ExecutionState] = {}
        self._object_owners: dict[str, list[str | None]] = {}
        self._arm_owners: dict[str, list[str | None]] = {
            "left_arm": [None] * int(env.num_envs),
            "right_arm": [None] * int(env.num_envs),
        }
        self._assignments: dict[str, list[str | None]] = {}
        self._candidate_cache: dict[tuple[str, str], _Candidate] = {}
        self._candidate_failures: dict[tuple[str, str], str] = {}
        self._candidate_diagnostics: dict[str, tuple[str, ...]] = {}
        self._candidate_blockers: dict[str, tuple[dict[str, Any], ...]] = {}
        self._reported_candidates: set[str] = set()
        self._pickup_retry_exclusions: dict[tuple[str, int], set[str]] = {}
        self._targets: dict[str, torch.Tensor] = {}
        self._target_poses: dict[str, torch.Tensor] = {}
        self._orientation_references: dict[str, torch.Tensor] = {}
        self._orientation_errors: dict[str, torch.Tensor] = {}
        self._policies: dict[str, dict[str, Any]] = {}
        self._payload_initial: dict[str, dict[str, torch.Tensor]] = {}
        self._support_relations: dict[str, list[_SupportRelation | None]] = {}
        self._placement_candidate_history: dict[tuple[str, str], set[int]] = {}
        self._robot_lateral_axis_cache: torch.Tensor | None = None
        self._transition_count = 0
        self._retry_counts = [0] * int(self.env.num_envs)

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
            runtime_policy=self.runtime_policy.as_mapping(),
            runtime_policy_hash=runtime_policy_hash(self.runtime_policy),
        )
        aggregate_failed = torch.zeros(
            int(self.env.num_envs),
            dtype=torch.bool,
            device=self.env.device,
        )
        edge_failures: dict[str, torch.Tensor] = {}
        semantic_success: dict[str, torch.Tensor] = {}
        failure_events: list[dict[str, Any]] = []
        completed: set[str] = set()
        remaining = [edge.id for edge in self.program.edges]
        executed_actions: list[torch.Tensor] = []
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
                ready_blocked = {
                    edge.id: self._dependency_failures(edge, edge_failures)
                    for edge in ready
                }
                batch = self._pack_ready_edges(
                    ready,
                    inactive=ready_blocked,
                    completed=completed,
                )
                blocked = {edge.id: ready_blocked[edge.id] for edge in batch}
                # A synchronized pair needs the same active rows. Execute a
                # healthy independent branch separately when its peer is blocked.
                if len(batch) == 2 and not torch.equal(
                    blocked[batch[0].id], blocked[batch[1].id]
                ):
                    batch = (batch[0],)
                self._consume_transitions(len(batch))

                if len(batch) == 2:
                    posture_before = {
                        edge.id: self._object_not_fallen(self.step_by_edge[edge.id])
                        for edge in batch
                    }
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
                            planner_traces=getattr(result, "planner_traces", ()),
                            diagnostics=self._edge_diagnostics(
                                step,
                                edge,
                                result.failed,
                            ),
                        )
                        failure_events.extend(
                            self._failure_events(
                                edge,
                                step,
                                result.failed & ~blocked[edge.id],
                                postcondition=False,
                                executed=result.executed,
                                fallen_transition=self._fallen_transition(
                                    step,
                                    posture_before[edge.id],
                                    result,
                                ),
                                planner_traces=result.planner_traces,
                            )
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
                    posture_before = self._object_not_fallen(step)
                    failure_policy = self._edge_failure_policy(edge)
                    try:
                        primary_result = self._execute_edge_with_retries(
                            edge,
                            step,
                            failed=branch_failed,
                        )
                    except Exception as exc:
                        if failure_policy != "best_effort":
                            raise
                        primary_result = self._edge_exception_result(
                            edge,
                            step,
                            branch_failed,
                            exc,
                        )
                    newly_failed = primary_result.failed & ~branch_failed
                    fallen_transition = self._fallen_transition(
                        step,
                        posture_before,
                        primary_result,
                    )
                    recorder.edge(
                        edge.id,
                        step,
                        assignments=self._assignments[step.id],
                        grounded=primary_result.grounded,
                        active=active,
                        failed=primary_result.failed,
                        action_steps=len(primary_result.actions),
                        planner_traces=getattr(primary_result, "planner_traces", ()),
                        diagnostics=self._edge_diagnostics(
                            step,
                            edge,
                            primary_result.failed,
                        ),
                        phase="primary",
                    )
                    edge_result = primary_result
                    if failure_policy == "task_required":
                        edge_result = self._recover_object_fallen(
                            edge,
                            step,
                            edge_result,
                            inherited_failed=branch_failed,
                            fallen_transition=fallen_transition,
                            recorder=recorder,
                        )
                    if failure_policy == "best_effort":
                        # Best-effort parking is observable but cannot invalidate
                        # an already verified task or safety condition.
                        next_failed = branch_failed
                    else:
                        next_failed = edge_result.failed
                    executed_actions.extend(edge_result.actions)
                    edge_failures[edge.id] = next_failed
                    failure_events.extend(
                        self._failure_events(
                            edge,
                            step,
                            newly_failed & edge_result.failed,
                            postcondition=False,
                            executed=getattr(primary_result, "executed", None),
                            fallen_transition=fallen_transition,
                            planner_traces=getattr(
                                primary_result, "planner_traces", ()
                            ),
                        )
                    )

                for edge in batch:
                    completed.add(edge.id)
                    remaining.remove(edge.id)
                    step = self.step_by_edge[edge.id]
                    if edge.id != step.edge_ids[-1]:
                        continue
                    prior_failed = edge_failures[edge.id]
                    verified_failed, step_success, observed = self._verify_step(
                        step, prior_failed
                    )
                    postcondition_failed = verified_failed & ~prior_failed
                    recovery_covered = torch.zeros_like(verified_failed)
                    primary_step_recorded = False
                    if (
                        self.placement_recovery_attempts
                        and bool(postcondition_failed.any())
                        and step.operator == "place_relative"
                        and normalize_placement_relation(
                            step.goal.get("relation", "on")
                        )
                        == "on"
                    ):
                        recorder.step(
                            step,
                            step_success,
                            observed=observed,
                            target=self._targets.get(step.id),
                            metadata=(
                                self._step_runtime_metadata(step)
                                if self.record_runtime
                                else None
                            ),
                            phase="primary",
                        )
                        primary_step_recorded = True
                        recovery = self._recover_unstable_placement(
                            step,
                            postcondition_failed,
                            recorder=recorder,
                        )
                        executed_actions.extend(recovery.actions)
                        failure_events.extend(recovery.failure_events)
                        recovery_covered = (
                            torch.zeros_like(verified_failed)
                            if recovery.covered_failures is None
                            else recovery.covered_failures
                        )
                        verified_failed = (
                            verified_failed & ~postcondition_failed
                        ) | recovery.failed
                        step_success |= recovery.succeeded
                        observed = recovery.observed
                    failure_events.extend(
                        self._failure_events(
                            edge,
                            step,
                            verified_failed & ~prior_failed & ~recovery_covered,
                            postcondition=True,
                            executed=~prior_failed,
                            fallen_transition=None,
                        )
                    )
                    edge_failures[edge.id] = verified_failed
                    aggregate_failed |= ~step_success
                    semantic_success[step.id] = step_success
                    arrangement = self.arrangements.get(step.id)
                    if arrangement is not None:
                        arrangement.mark_completed(step.id, step_success)
                    if not primary_step_recorded:
                        recorder.step(
                            step,
                            step_success,
                            observed=observed,
                            target=self._targets.get(step.id),
                            metadata=(
                                self._step_runtime_metadata(step)
                                if self.record_runtime
                                else None
                            ),
                        )
            revalidation_failures = self._revalidate_support_relations()
            for step_id, lost in revalidation_failures.items():
                step = self.steps[step_id]
                edge = self.edges[step.edge_ids[-1]]
                aggregate_failed |= lost
                semantic_success[step_id] = semantic_success[step_id] & ~lost
                edge_failures[edge.id] |= lost
                failure_events.extend(
                    self._failure_events(
                        edge,
                        step,
                        lost,
                        postcondition=True,
                        executed=torch.ones_like(lost),
                        fallen_transition=None,
                    )
                )
                recorder.step(
                    step,
                    semantic_success[step_id],
                    observed=self._entity_pose(step.object_uid)[:, :3, 3],
                    target=self._targets.get(step.id),
                    metadata=(
                        self._step_runtime_metadata(step)
                        if self.record_runtime
                        else None
                    ),
                    phase="final_revalidation",
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
            retry_count=self.retry_count,
            retry_counts=list(self._retry_counts),
            recovery_count=(
                0
                if self.runtime_graph is None
                else sum(
                    revision.kind == "insert_recovery"
                    for revision in self.runtime_graph.revisions
                )
            ),
            revision_count=(
                0 if self.runtime_graph is None else len(self.runtime_graph.revisions)
            ),
            failure_events=failure_events,
            runtime_revisions=(
                []
                if self.runtime_graph is None
                else [
                    {
                        "revision": revision.revision,
                        "kind": revision.kind,
                        "reason": revision.reason,
                        "failed_node_id": revision.failed_node_id,
                        "inserted_group_ids": list(revision.inserted_group_ids),
                        "replaced_group_ids": list(revision.replaced_group_ids),
                        "active_env_ids": list(revision.active_env_ids),
                    }
                    for revision in self.runtime_graph.revisions
                ]
            ),
        )

    def _failure_events(
        self,
        edge: ExecutionEdge,
        step: SemanticStep,
        failed: torch.Tensor,
        *,
        postcondition: bool,
        executed: torch.Tensor | None,
        fallen_transition: torch.Tensor | None,
        planner_traces: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        if not bool(failed.any()):
            return []
        action = edge.actions[-1]
        action_name = str(action["atomic_action_class"])
        capability = self.adapter.capabilities.get(action_name)
        executed_mask = (
            torch.zeros_like(failed)
            if executed is None
            else torch.as_tensor(
                executed,
                dtype=torch.bool,
                device=failed.device,
            ).reshape(-1)
        )
        if executed_mask.shape != failed.shape:
            raise ValueError("Failure provenance mask must match failed rows.")
        transitioned = (
            torch.zeros_like(failed)
            if fallen_transition is None
            else torch.as_tensor(
                fallen_transition,
                dtype=torch.bool,
                device=failed.device,
            ).reshape(-1)
        )
        if transitioned.shape != failed.shape:
            raise ValueError("Fallen-transition mask must match failed rows.")
        failure_policy = (
            "task_required" if postcondition else self._edge_failure_policy(edge)
        )
        fatal = failure_policy != "best_effort"
        if postcondition:
            classified = (("postcondition_failed", failed),)
        else:
            fallen = failed & executed_mask & transitioned
            planning = failed & ~executed_mask
            execution = failed & executed_mask & ~fallen
            if capability.failure_classifier == "grasp":
                execution_type = "grasp_missed"
            elif capability.state_effect in {"preserve_hold", "transfer_hold"}:
                execution_type = "object_dropped"
            else:
                execution_type = "plan_failed"
            classified = (
                ("object_fallen", fallen),
                ("search_exhausted", planning),
                (execution_type, execution),
            )
        result: list[dict[str, Any]] = []
        for failure_type, mask in classified:
            env_ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            if not env_ids:
                continue
            if failure_type == "search_exhausted":
                covered: set[int] = set()
                for blocker in getattr(self, "_candidate_blockers", {}).get(
                    step.id, ()
                ):
                    env_id = int(blocker["env_id"])
                    if env_id not in env_ids:
                        continue
                    assignment = self._assignments.get(step.id, [None] * len(failed))[
                        env_id
                    ]
                    if assignment is not None and blocker.get("arm") != assignment:
                        continue
                    blocker_policy = str(blocker.get("failure_policy", failure_policy))
                    result.append(
                        {
                            "node_id": blocker.get("node_id"),
                            "edge_id": edge.id,
                            "origin_edge_id": edge.id,
                            "blocking_edge_id": blocker["blocking_edge_id"],
                            "task_instance_id": step.id,
                            "atomic_action": blocker["atomic_action"],
                            "object_uid": step.object_uid,
                            "arm": blocker.get("arm"),
                            "failure_type": "search_exhausted",
                            "failure_policy": blocker_policy,
                            "fatal": blocker_policy != "best_effort",
                            "planning_stage": blocker["planning_stage"],
                            "search_strategy": blocker["search_strategy"],
                            "search_budget": deepcopy(blocker["search_budget"]),
                            "reason": (
                                "Bounded candidate search exhausted without a "
                                "valid plan; this is not a geometric proof of "
                                "unreachability."
                            ),
                            "evidence": deepcopy(blocker["evidence"]),
                            "env_ids": [env_id],
                        }
                    )
                    covered.add(env_id)
                for env_id in (item for item in env_ids if item not in covered):
                    trace = next(
                        (
                            item
                            for item in planner_traces
                            if str(item.get("arm", ""))
                            == str(self._assignments.get(step.id, [None])[env_id])
                        ),
                        planner_traces[0] if planner_traces else {},
                    )
                    result.append(
                        {
                            "node_id": action.get("seed_node_id"),
                            "edge_id": edge.id,
                            "blocking_edge_id": edge.id,
                            "task_instance_id": step.id,
                            "atomic_action": action_name,
                            "arm": self._assignments.get(step.id, [None])[env_id],
                            "failure_type": "search_exhausted",
                            "failure_policy": failure_policy,
                            "fatal": fatal,
                            "planning_stage": "runtime_planning",
                            **self._planner_failure_details(trace, env_id),
                            "reason": (
                                "Bounded runtime search exhausted without a valid "
                                "plan; this is not a geometric proof of "
                                "unreachability."
                            ),
                            "env_ids": [env_id],
                        }
                    )
                continue
            result.append(
                {
                    "node_id": action.get("seed_node_id"),
                    "edge_id": edge.id,
                    "blocking_edge_id": edge.id,
                    "task_instance_id": step.id,
                    "atomic_action": action_name,
                    "object_uid": step.object_uid,
                    "failure_type": failure_type,
                    "failure_policy": failure_policy,
                    "fatal": fatal,
                    "planning_stage": (
                        "postcondition" if postcondition else "execution"
                    ),
                    "env_ids": env_ids,
                }
            )
        return result

    def _edge_exception_result(
        self,
        edge: ExecutionEdge,
        step: SemanticStep,
        inherited_failed: torch.Tensor,
        exc: Exception,
    ) -> _EdgeResult:
        """Convert a planning exception into an auditable failed edge result."""
        action = edge.actions[0]
        assignments = self._assignments.get(step.id, [None] * int(self.env.num_envs))
        arm = next((item for item in assignments if item is not None), None)
        trace = {
            "action_class": str(action.get("atomic_action_class")),
            "arm": arm,
            "primary_strategy": "planner_exception",
            "primary_success": torch.zeros_like(inherited_failed),
            "fallback_attempted": torch.zeros_like(inherited_failed),
            "fallback_success": torch.zeros_like(inherited_failed),
            "search_budget": self._planner_search_budget(),
            "exception": f"{type(exc).__name__}: {exc}",
        }
        return _EdgeResult(
            [],
            torch.ones_like(inherited_failed),
            [],
            [trace],
            torch.zeros_like(inherited_failed),
        )

    def _object_not_fallen(self, step: SemanticStep) -> torch.Tensor | None:
        """Return the live posture predicate when the object supports it."""
        try:
            return evaluate_predicate(
                self.env,
                {"type": "object_not_fallen", "object": step.object_uid},
            )
        except (TypeError, ValueError):
            return None

    def _fallen_transition(
        self,
        step: SemanticStep,
        before: torch.Tensor | None,
        result: _EdgeResult,
    ) -> torch.Tensor:
        """Identify rows where an executed action changed upright to fallen."""
        result_executed = getattr(result, "executed", None)
        if before is None or result_executed is None:
            return torch.zeros_like(result.failed)
        after = self._object_not_fallen(step)
        if after is None:
            return torch.zeros_like(result.failed)
        executed = torch.as_tensor(
            result_executed,
            dtype=torch.bool,
            device=result.failed.device,
        ).reshape(-1)
        return (
            executed & before.to(result.failed.device) & ~after.to(result.failed.device)
        )

    def _execute_edge_with_retries(
        self,
        edge: ExecutionEdge,
        step: SemanticStep,
        *,
        failed: torch.Tensor,
    ) -> _EdgeResult:
        """Retry a complete AtomicAction with fresh Grounding on failed rows."""
        result = self._execute_edge(edge, step, failed=failed)
        if self.runtime_graph is None or len(edge.actions) != 1:
            return result
        action = edge.actions[0]
        node_id = action.get("seed_node_id")
        if not isinstance(node_id, str):
            return result
        aggregate_actions = list(result.actions)
        grounded = list(result.grounded)
        planner_traces = list(getattr(result, "planner_traces", ()))
        executed = (
            torch.zeros_like(result.failed)
            if getattr(result, "executed", None) is None
            else result.executed.clone()
        )
        current_failed = result.failed.clone()
        attempted_failure = current_failed & ~failed
        while bool(attempted_failure.any()):
            precondition = self._retry_precondition(node_id, attempted_failure)
            decision = self.runtime_graph.record_failure(
                node_id,
                attempted_failure,
                precondition_holds=precondition,
            )
            if not bool(decision.retry.any()):
                break
            self.retry_count += int(decision.retry.sum())
            for env_id in (
                torch.nonzero(decision.retry, as_tuple=False).flatten().tolist()
            ):
                self._retry_counts[env_id] += 1
            self._consume_transitions(1)
            for arm in ("left_arm", "right_arm"):
                self._candidate_cache.pop((step.id, arm), None)
                self._candidate_failures.pop((step.id, arm), None)
            capability = self.adapter.capabilities.get(
                str(action.get("atomic_action_class"))
            )
            if capability.state_effect == "hold":
                previous = list(self._assignments[step.id])
                for env_id in (
                    torch.nonzero(decision.retry, as_tuple=False).flatten().tolist()
                ):
                    arm = previous[env_id]
                    if step.actor.get("mode") == "auto" and arm in {
                        "left_arm",
                        "right_arm",
                    }:
                        self._pickup_retry_exclusions.setdefault(
                            (step.id, env_id), set()
                        ).add(str(arm))
                for arm in ("left_arm", "right_arm"):
                    self._step_states.pop((step.id, arm), None)
                if step.actor.get("mode") == "auto":
                    self._assignments.pop(step.id, None)
                    self._ensure_assignment(step, ~decision.retry)
                    refreshed = self._assignments[step.id]
                    self._assignments[step.id] = [
                        refreshed[index] if bool(decision.retry[index]) else assignment
                        for index, assignment in enumerate(previous)
                    ]
            retry_result = self._execute_edge(
                edge,
                step,
                failed=~decision.retry,
            )
            aggregate_actions.extend(retry_result.actions)
            grounded.extend(retry_result.grounded)
            planner_traces.extend(getattr(retry_result, "planner_traces", ()))
            if getattr(retry_result, "executed", None) is not None:
                executed |= retry_result.executed
            succeeded = decision.retry & ~retry_result.failed
            current_failed &= ~succeeded
            attempted_failure = decision.retry & retry_result.failed
        return _EdgeResult(
            aggregate_actions,
            current_failed,
            grounded,
            planner_traces,
            executed,
        )

    def _recover_object_fallen(
        self,
        edge: ExecutionEdge,
        step: SemanticStep,
        result: _EdgeResult,
        *,
        inherited_failed: torch.Tensor,
        fallen_transition: torch.Tensor,
        recorder: RuntimeRecorder,
    ) -> _EdgeResult:
        """Run the bounded E2 repair and replay only the failed vector rows."""
        if self.runtime_graph is None or len(edge.actions) != 1:
            return result
        node_id = edge.actions[0].get("seed_node_id")
        if not isinstance(node_id, str) or not node_id:
            return result
        newly_failed = result.failed & ~inherited_failed
        if not bool(newly_failed.any()):
            return result
        executed = (
            torch.zeros_like(result.failed)
            if getattr(result, "executed", None) is None
            else torch.as_tensor(
                result.executed,
                dtype=torch.bool,
                device=result.failed.device,
            ).reshape(-1)
        )
        transition = torch.as_tensor(
            fallen_transition,
            dtype=torch.bool,
            device=result.failed.device,
        ).reshape(-1)
        if (
            executed.shape != result.failed.shape
            or transition.shape != result.failed.shape
        ):
            raise ValueError("Recovery provenance masks must match failed rows.")
        fallen = newly_failed & executed & transition
        if not bool(fallen.any()):
            return result

        env_ids = torch.nonzero(fallen, as_tuple=False).flatten().tolist()
        original_assignment = list(
            self._assignments.get(step.id, [None] * int(self.env.num_envs))
        )
        try:
            patched = self.runtime_graph.insert_default_recovery(
                failed_node_id=node_id,
                failure_type="object_fallen",
                active_env_ids=env_ids,
                resume_failed_group=True,
            )
            revision = self.runtime_graph.revisions[-1]
            recovery_group_id = revision.inserted_group_ids[0]
            from .loader import load_execution_program

            recovery_program = load_execution_program(
                patched,
                registry=self.capability_registry,
                require_executable=True,
            )
            recovery_step = next(
                item
                for item in recovery_program.semantic_steps
                if item.id == recovery_group_id
            )
            recovery_spec = next(
                item
                for item in recovery_program.raw["semantic_steps"]
                if str(item["id"]) == recovery_group_id
            )
            recorder.register_step(recovery_step, recovery_spec)
            recovery_edges = {
                item.id: item
                for item in recovery_program.edges
                if item.id in set(recovery_step.edge_ids)
            }
            if set(recovery_edges) != set(recovery_step.edge_ids):
                raise RuntimeError("Compiled recovery group is incomplete.")
        except Exception as exc:
            recorder.recovery(
                failure_type="object_fallen",
                failed_node_id=node_id,
                active=fallen,
                status="rejected",
                error=f"{type(exc).__name__}: {exc}",
                semantic_step_id=step.id,
            )
            return result

        recorder.recovery(
            failure_type="object_fallen",
            failed_node_id=node_id,
            active=fallen,
            status="started",
            recovery_group_id=recovery_group_id,
            semantic_step_id=step.id,
        )
        aggregate_actions = list(result.actions)
        grounded = list(result.grounded)
        planner_traces = list(getattr(result, "planner_traces", ()))
        self._clear_recovery_rows(step, fallen)

        # Recovery edges are compiled from the revised graph but execute through
        # this executor so live ownership, recorder, and simulator state remain
        # continuous. They are removed from the scheduling maps afterwards.
        installed_edge_ids: list[str] = []
        self.steps[recovery_step.id] = recovery_step
        for recovery_edge in recovery_edges.values():
            self.edges[recovery_edge.id] = recovery_edge
            self.step_by_edge[recovery_edge.id] = recovery_step
            installed_edge_ids.append(recovery_edge.id)
        recovery_failed = ~fallen
        try:
            self._assignments.pop(recovery_step.id, None)
            self._ensure_assignment(recovery_step, recovery_failed)
            for recovery_edge_id in recovery_step.edge_ids:
                self._consume_transitions(1)
                recovery_edge = recovery_edges[recovery_edge_id]
                recovery_result = self._execute_edge_with_retries(
                    recovery_edge,
                    recovery_step,
                    failed=recovery_failed,
                )
                recorder.edge(
                    recovery_edge.id,
                    recovery_step,
                    assignments=self._assignments[recovery_step.id],
                    grounded=recovery_result.grounded,
                    active=~recovery_failed,
                    failed=recovery_result.failed,
                    action_steps=len(recovery_result.actions),
                    planner_traces=recovery_result.planner_traces,
                    phase="recovery",
                )
                aggregate_actions.extend(recovery_result.actions)
                grounded.extend(recovery_result.grounded)
                planner_traces.extend(recovery_result.planner_traces)
                recovery_failed = recovery_result.failed
            _, recovery_success, observed = self._verify_step(
                recovery_step,
                recovery_failed,
            )
            recorder.step(
                recovery_step,
                recovery_success,
                observed=observed,
                target=self._targets.get(recovery_step.id),
                metadata=(
                    self._step_runtime_metadata(recovery_step)
                    if self.record_runtime
                    else None
                ),
                phase="recovery",
            )
        except Exception as exc:
            recorder.recovery(
                failure_type="object_fallen",
                failed_node_id=node_id,
                active=fallen,
                status="failed",
                recovery_group_id=recovery_group_id,
                error=f"{type(exc).__name__}: {exc}",
                semantic_step_id=step.id,
            )
            return _EdgeResult(
                aggregate_actions,
                result.failed,
                grounded,
                planner_traces,
                result.executed,
            )
        finally:
            for recovery_edge_id in installed_edge_ids:
                self.edges.pop(recovery_edge_id, None)
                self.step_by_edge.pop(recovery_edge_id, None)
            self.steps.pop(recovery_step.id, None)

        recovered = fallen & recovery_success
        if not bool(recovered.any()):
            recorder.recovery(
                failure_type="object_fallen",
                failed_node_id=node_id,
                active=fallen,
                status="failed",
                recovery_group_id=recovery_group_id,
                semantic_step_id=step.id,
            )
            return _EdgeResult(
                aggregate_actions,
                result.failed,
                grounded,
                planner_traces,
                result.executed,
            )

        # Recompute this TaskGroup's assignment for recovered rows, retaining
        # the untouched assignments of healthy vector rows. Replay the prefix
        # through the failed edge; the ordinary main loop will then continue at
        # the next edge and verify the TaskGroup exactly once.
        try:
            self._assignments.pop(step.id, None)
            for arm in ("left_arm", "right_arm"):
                self._candidate_cache.pop((step.id, arm), None)
                self._candidate_failures.pop((step.id, arm), None)
            self._ensure_assignment(step, ~recovered)
            replay_assignment = self._assignments[step.id]
            self._assignments[step.id] = [
                (
                    replay_assignment[index]
                    if bool(recovered[index])
                    else original_assignment[index]
                )
                for index in range(int(self.env.num_envs))
            ]
            replay_failed = ~recovered
            for prefix_edge_id in step.edge_ids:
                self._consume_transitions(1)
                prefix_edge = self.edges[prefix_edge_id]
                replay_active = ~replay_failed
                prefix_result = self._execute_edge_with_retries(
                    prefix_edge,
                    step,
                    failed=replay_failed,
                )
                recorder.edge(
                    prefix_edge.id,
                    step,
                    assignments=self._assignments[step.id],
                    grounded=prefix_result.grounded,
                    active=replay_active,
                    failed=prefix_result.failed,
                    action_steps=len(prefix_result.actions),
                    planner_traces=prefix_result.planner_traces,
                    diagnostics=self._edge_diagnostics(
                        step,
                        prefix_edge,
                        prefix_result.failed,
                    ),
                    phase="replay",
                )
                aggregate_actions.extend(prefix_result.actions)
                grounded.extend(prefix_result.grounded)
                planner_traces.extend(prefix_result.planner_traces)
                replay_failed = prefix_result.failed
                if prefix_edge_id == edge.id:
                    break
        except Exception as exc:
            recorder.recovery(
                failure_type="object_fallen",
                failed_node_id=node_id,
                active=fallen,
                status="failed",
                recovery_group_id=recovery_group_id,
                error=f"{type(exc).__name__}: {exc}",
                semantic_step_id=step.id,
            )
            return _EdgeResult(
                aggregate_actions,
                result.failed,
                grounded,
                planner_traces,
                result.executed,
            )
        final_failed = result.failed.clone()
        final_failed[fallen] = replay_failed[fallen]
        recorder.recovery(
            failure_type="object_fallen",
            failed_node_id=node_id,
            active=fallen,
            status=("succeeded" if not bool(final_failed[fallen].any()) else "failed"),
            recovery_group_id=recovery_group_id,
            semantic_step_id=step.id,
        )
        return _EdgeResult(
            aggregate_actions,
            final_failed,
            grounded,
            planner_traces,
            result.executed,
        )

    def _clear_recovery_rows(
        self,
        step: SemanticStep,
        mask: torch.Tensor,
    ) -> None:
        """Discard stale hold projections only for rows entering recovery."""
        owners = self._object_owners.setdefault(
            step.object_uid, [None] * int(self.env.num_envs)
        )
        for env_id in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            owner = owners[env_id]
            owners[env_id] = None
            if owner in self._arm_owners and (
                self._arm_owners[str(owner)][env_id] == step.object_uid
            ):
                self._arm_owners[str(owner)][env_id] = None
            for arm in ("left_arm", "right_arm"):
                if self._arm_owners[arm][env_id] == step.object_uid:
                    self._arm_owners[arm][env_id] = None

        candidate_keys = [
            key for key in self._object_states if key[0] == step.object_uid
        ]
        step_keys = [key for key in self._step_states if key[0] == step.id]
        for cache, keys in (
            (self._object_states, candidate_keys),
            (self._step_states, step_keys),
        ):
            for key in keys:
                state = cache[key]
                delta = StateDelta(
                    held_object_updates={name: None for name in state.held_objects},
                )
                if delta.is_empty:
                    continue
                cache[key] = ExecutionState.from_task_state(
                    delta.apply(state.to_task_state(), mask),
                    last_qpos=self.env.robot.get_qpos().clone(),
                )

    def _recover_unstable_placement(
        self,
        step: SemanticStep,
        failed: torch.Tensor,
        *,
        recorder: RuntimeRecorder,
    ) -> _PlacementRecoveryResult:
        """Regrasp after release, then retry unused placement poses only."""
        pending = failed.clone()
        recovered = torch.zeros_like(failed)
        observed = self._entity_pose(step.object_uid)[:, :3, 3]
        actions: list[torch.Tensor] = []
        blocking_failures: list[tuple[ExecutionEdge, _EdgeResult, torch.Tensor]] = []
        terminal_edge = self.edges[step.edge_ids[-1]]
        failed_node_id = str(
            terminal_edge.actions[-1].get("seed_node_id", terminal_edge.id)
        )
        recorder.recovery(
            failure_type="placement_unstable",
            failed_node_id=failed_node_id,
            active=failed,
            status="started",
            semantic_step_id=step.id,
        )
        for _attempt in range(self.placement_recovery_attempts):
            if not bool(pending.any()):
                break
            self._consume_transitions(len(step.edge_ids))
            attempt_active = pending.clone()
            self._clear_recovery_rows(step, attempt_active)
            self._assignments.pop(step.id, None)
            for arm in ("left_arm", "right_arm"):
                self._candidate_cache.pop((step.id, arm), None)
                self._candidate_failures.pop((step.id, arm), None)
            try:
                self._ensure_assignment(step, ~attempt_active)
            except Exception as exc:
                blocking_edge = self.edges[step.edge_ids[0]]
                blocking_result = self._edge_exception_result(
                    blocking_edge,
                    step,
                    ~attempt_active,
                    exc,
                )
                blocking_failures.append(
                    (blocking_edge, blocking_result, attempt_active)
                )
                recorder.edge(
                    blocking_edge.id,
                    step,
                    assignments=self._assignments.get(
                        step.id,
                        [None] * int(self.env.num_envs),
                    ),
                    grounded=(),
                    active=attempt_active,
                    failed=blocking_result.failed,
                    action_steps=0,
                    planner_traces=blocking_result.planner_traces,
                    diagnostics=self._edge_diagnostics(
                        step,
                        blocking_edge,
                        blocking_result.failed,
                    ),
                    phase="recovery",
                )
                break

            replay_failed = ~attempt_active
            for edge_id in step.edge_ids:
                edge = self.edges[edge_id]
                edge_active = ~replay_failed
                try:
                    result = self._execute_edge_with_retries(
                        edge,
                        step,
                        failed=replay_failed,
                    )
                except Exception as exc:
                    result = self._edge_exception_result(
                        edge,
                        step,
                        replay_failed,
                        exc,
                    )
                actions.extend(result.actions)
                recorder.edge(
                    edge.id,
                    step,
                    assignments=self._assignments[step.id],
                    grounded=result.grounded,
                    active=edge_active,
                    failed=result.failed,
                    action_steps=len(result.actions),
                    planner_traces=result.planner_traces,
                    diagnostics=self._edge_diagnostics(step, edge, result.failed),
                    phase="recovery",
                )
                newly_failed = edge_active & result.failed
                if bool(newly_failed.any()):
                    blocking_failures.append((edge, result, newly_failed))
                replay_failed = result.failed
                if not bool((attempt_active & ~replay_failed).any()):
                    break

            execution_succeeded = attempt_active & ~replay_failed
            if not bool(execution_succeeded.any()):
                break
            verified_failed, verified_success, observed = self._verify_step(
                step,
                ~execution_succeeded,
            )
            del verified_failed
            recovered_now = attempt_active & verified_success
            recovered |= recovered_now
            recorder.step(
                step,
                verified_success,
                observed=observed,
                target=self._targets.get(step.id),
                metadata=(
                    self._step_runtime_metadata(step) if self.record_runtime else None
                ),
                phase="recovery",
            )
            action_failed = attempt_active & replay_failed
            pending &= ~recovered_now
            if bool(action_failed.any()):
                break

        final_failed = failed & ~recovered
        recovery_events: list[dict[str, Any]] = []
        covered_failures = torch.zeros_like(failed)
        for blocking_edge, blocking_result, blocking_rows in blocking_failures:
            event_rows = final_failed & blocking_rows & ~covered_failures
            if not bool(event_rows.any()):
                continue
            events = self._failure_events(
                blocking_edge,
                step,
                event_rows,
                postcondition=False,
                executed=blocking_result.executed,
                fallen_transition=None,
                planner_traces=blocking_result.planner_traces,
            )
            for event in events:
                event["phase"] = "recovery"
                event["origin_edge_id"] = terminal_edge.id
            recovery_events.extend(events)
            covered_failures |= event_rows
        recorder.recovery(
            failure_type="placement_unstable",
            failed_node_id=failed_node_id,
            active=failed,
            status="failed" if bool(final_failed.any()) else "succeeded",
            semantic_step_id=step.id,
        )
        return _PlacementRecoveryResult(
            failed=final_failed,
            succeeded=recovered,
            observed=observed,
            actions=actions,
            failure_events=recovery_events,
            covered_failures=covered_failures,
        )

    def _retry_precondition(
        self,
        node_id: str,
        failed: torch.Tensor,
    ) -> torch.Tensor:
        assert self.runtime_graph is not None
        node = next(
            item for item in self.runtime_graph.graph["nodes"] if item["id"] == node_id
        )
        predicate = node.get("precondition", {})
        if not predicate:
            return failed.clone()
        try:
            return failed & evaluate_predicate(
                self.env,
                predicate,
                held_owners=self._object_owners,
                held_states=self._object_states,
                coordinated_state=self._step_states.get(
                    (str(node.get("task_instance_id", "")), "coordinated")
                ),
            )
        except (TypeError, ValueError):
            return torch.zeros_like(failed)

    def _dependency_failures(
        self,
        edge: ExecutionEdge,
        failures: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return success-required failures that can reach this edge."""
        result = torch.zeros(
            int(self.env.num_envs), dtype=torch.bool, device=self.env.device
        )
        for dependency in edge.depends_on:
            if (dependency, edge.id) in self._completion_only_dependencies:
                continue
            result |= failures[dependency]
        return result

    def _completion_only_dependency_edges(self) -> frozenset[tuple[str, str]]:
        """Resolve linker-added resource ordering to executable edge pairs."""
        graph = self.program.seed_graph
        if not isinstance(graph, Mapping):
            return frozenset()
        metadata = graph.get("metadata", {})
        if not isinstance(metadata, Mapping):
            return frozenset()

        reasons_by_pair: dict[tuple[str, str], set[str]] = {}
        sources = (
            ("action_contract_task_linker", "linked_dependencies"),
            ("action_contract_linker", "group_dependencies"),
        )
        for metadata_key, dependency_key in sources:
            provenance = metadata.get(metadata_key, {})
            if not isinstance(provenance, Mapping):
                continue
            dependencies = provenance.get(dependency_key, ())
            if not isinstance(dependencies, Sequence) or isinstance(
                dependencies, (str, bytes, bytearray)
            ):
                continue
            for dependency in dependencies:
                if not isinstance(dependency, Mapping):
                    continue
                parent = dependency.get("from")
                child = dependency.get("to")
                reason = dependency.get("reason")
                if not all(
                    isinstance(value, str) and value for value in (parent, child)
                ):
                    continue
                if reason not in {"causal", "resource"}:
                    continue
                reasons_by_pair.setdefault((parent, child), set()).add(reason)

        completion_only_steps = {
            pair for pair, reasons in reasons_by_pair.items() if reasons == {"resource"}
        }
        return frozenset(
            (dependency, edge.id)
            for edge in self.program.edges
            for dependency in edge.depends_on
            if (
                self.step_by_edge[dependency].id,
                self.step_by_edge[edge.id].id,
            )
            in completion_only_steps
        )

    def _reset_runtime_state(self) -> None:
        self.retry_count = 0
        if self.program.seed_graph is not None:
            execution = self.runtime_policy.execution
            self.runtime_graph = RuntimeGraph(
                self.program.seed_graph,
                num_envs=int(self.env.num_envs),
                max_retries=self.max_retries_per_action,
                max_revisions=int(execution["max_graph_revisions"]),
                max_recovery_actions=int(execution["max_recovery_actions"]),
                registry=self.capability_registry,
            )
        self._step_states.clear()
        self._object_states.clear()
        self._object_owners.clear()
        for owners in self._arm_owners.values():
            owners[:] = [None] * int(self.env.num_envs)
        self._assignments.clear()
        self._candidate_cache.clear()
        self._candidate_failures.clear()
        self._candidate_diagnostics.clear()
        self._candidate_blockers.clear()
        self._reported_candidates.clear()
        self._pickup_retry_exclusions.clear()
        self._targets.clear()
        self._target_poses.clear()
        self._orientation_references.clear()
        self._orientation_errors.clear()
        self._policies.clear()
        self._payload_initial.clear()
        self._support_relations.clear()
        self._placement_candidate_history.clear()
        self._robot_lateral_axis_cache = None
        self._transition_count = 0
        self._retry_counts = [0] * int(self.env.num_envs)

    def _consume_transitions(self, count: int) -> None:
        """Charge ordinary, retry, and recovery edges to one runtime budget."""
        self._transition_count += int(count)
        if self._transition_count > self.max_transitions:
            raise RuntimeError("Execution exceeded max_transitions.")

    def _pack_ready_edges(
        self,
        ready: Sequence[ExecutionEdge],
        *,
        inactive: Mapping[str, torch.Tensor] | None = None,
        completed: set[str] | None = None,
    ) -> tuple[ExecutionEdge, ...]:
        """Prefer progress on held payloads and pack only resource-safe pickups."""
        inactive = inactive or {}
        completed = completed or set()
        schedulable = [
            edge
            for edge in ready
            if not self._temporarily_resource_blocked(
                edge,
                inactive.get(edge.id),
            )
        ]
        started = [
            edge
            for edge in schedulable
            if any(
                edge_id in completed for edge_id in self.step_by_edge[edge.id].edge_ids
            )
        ]
        candidates = started or schedulable or list(ready)
        first = candidates[0]
        if not self._parallel_pickup_candidate(first):
            return (first,)
        if not self._two_arms_available(inactive.get(first.id)):
            return (first,)
        first_step = self.step_by_edge[first.id]
        for second in candidates[1:]:
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

    def _temporarily_resource_blocked(
        self,
        edge: ExecutionEdge,
        inactive: torch.Tensor | None,
    ) -> bool:
        """Defer a new pickup while its arm is carrying another payload."""
        if not self._parallel_pickup_candidate(edge):
            return False
        step = self.step_by_edge[edge.id]
        mode = str(step.actor.get("mode", "auto"))
        inactive_mask = (
            torch.zeros(
                int(self.env.num_envs), dtype=torch.bool, device=self.env.device
            )
            if inactive is None
            else inactive
        )
        for env_id in range(int(self.env.num_envs)):
            if bool(inactive_mask[env_id]):
                continue
            if mode == "required":
                arms = (str(step.actor["arm"]),)
            else:
                arms = ("left_arm", "right_arm")
            if not any(
                self._arm_owners[arm][env_id] in {None, step.object_uid} for arm in arms
            ):
                return True
        return False

    def _two_arms_available(self, inactive: torch.Tensor | None) -> bool:
        inactive_mask = (
            torch.zeros(
                int(self.env.num_envs), dtype=torch.bool, device=self.env.device
            )
            if inactive is None
            else inactive
        )
        for env_id in range(int(self.env.num_envs)):
            if bool(inactive_mask[env_id]):
                continue
            free = sum(
                self._arm_owners[arm][env_id] is None
                for arm in ("left_arm", "right_arm")
            )
            if free < 2:
                return False
        return True

    def _parallel_pickup_candidate(self, edge: ExecutionEdge) -> bool:
        if len(edge.actions) != 1:
            return False
        step = self.step_by_edge[edge.id]
        capability = self.adapter.capabilities.get(
            str(edge.actions[0].get("atomic_action_class"))
        )
        return (
            capability.state_effect == "hold"
            and capability.resource_mode == "single_arm_object"
            and step.actor.get("mode") in {"auto", "required"}
            and step.operator != "orient_object"
        )

    def _preferred_in_place_arm(
        self,
        step: SemanticStep,
        env_id: int,
    ) -> str | None:
        """Map a clearly sided in-place object to the robot-view arm slot."""
        if step.operator != "orient_object":
            return None
        initial = getattr(self.env, "agent_initial_object_poses", {}).get(
            step.object_uid
        )
        if initial is None:
            entity = self.env.sim.get_rigid_object(step.object_uid)
            if entity is None:
                return None
            initial = entity.get_local_pose(to_matrix=True)
        pose = torch.as_tensor(initial, device=self.env.device)
        if pose.ndim == 2:
            pose = pose.unsqueeze(0)
        center, _, lateral_axis = self._arm_selection_workspace(step)
        index = min(env_id, pose.shape[0] - 1)
        lateral = float(
            torch.sum((pose[index, :2, 3] - center[index]) * lateral_axis[index])
        )
        if (
            abs(lateral)
            <= self.runtime_policy.arm_selection.orient_object_preferred_arm_deadband
        ):
            return None
        return "left_arm" if lateral > 0.0 else "right_arm"

    def _preferred_live_pickup_arm(
        self,
        step: SemanticStep,
        env_id: int,
    ) -> str | None:
        """Choose the arm on the object's current side when estimates fail."""
        pose = self._entity_pose(step.object_uid)
        center, _, lateral_axis = self._arm_selection_workspace(step)
        index = min(env_id, pose.shape[0] - 1)
        lateral = float(
            torch.sum((pose[index, :2, 3] - center[index]) * lateral_axis[index])
        )
        if (
            abs(lateral)
            <= self.runtime_policy.arm_selection.orient_object_preferred_arm_deadband
        ):
            return None
        return "left_arm" if lateral > 0.0 else "right_arm"

    def _ensure_assignment(
        self,
        step: SemanticStep,
        failed: torch.Tensor,
        *,
        allow_rematch: bool = True,
    ) -> None:
        self._capture_orientation_reference(step)
        if step.id in self._assignments:
            return
        if bool(failed.all()):
            self._assignments[step.id] = [None] * int(self.env.num_envs)
            return
        mode = str(step.actor.get("mode", "auto"))
        group = self.group_by_step.get(step.id)
        if mode == "auto" and group is not None:
            self._ensure_serial_group_assignments(group, failed)
            if step.id in self._assignments:
                return
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
            first_action = self.edges[step.edge_ids[0]].actions[0]
            first_capability = self.adapter.capabilities.get(
                str(first_action.get("atomic_action_class"))
            )
            if step.operator == "handover" and first_capability.state_effect != "hold":
                # A coordinated handover has an internal, multi-arm planner.
                # Do not let a speculative single-arm suffix plan veto the
                # real execution (or create a misleading downstream pickup
                # error) once a predecessor already established the transfer
                # hold. A standalone E4 starts with PickUp and still needs its
                # cached candidate plan for that first action.
                source_state = self._state_for(step, arm)
                has_hold = (
                    source_state.get_held_object(arm_control_part(self.env, arm))
                    is not None
                )
                self._assignments[step.id] = [
                    arm if has_hold and not bool(failed[index]) else None
                    for index in range(len(failed))
                ]
                return
            candidate = self._candidate(step, arm, failed)
            conflicts = self._resource_conflicts(step, arm)
            self._assignments[step.id] = [
                (
                    arm
                    if not bool(failed[index]) and not bool(conflicts[index])
                    else None
                )
                for index in range(len(failed))
            ]
            self._report_candidates(step, (candidate,))
            return

        left = self._candidate(step, "left_arm", failed)
        right = self._candidate(step, "right_arm", failed)
        candidates = {"left_arm": left, "right_arm": right}
        conflicts = {
            arm: self._resource_conflicts(step, arm)
            for arm in ("left_arm", "right_arm")
        }
        owners = self._object_owners.get(step.object_uid, [None] * len(failed))
        assignments: list[str | None] = []
        selection_failed = torch.zeros_like(failed)
        for env_id in range(len(failed)):
            if bool(failed[env_id]):
                assignments.append(None)
                continue
            if owners[env_id] is not None:
                owner = str(owners[env_id])
                excluded = self._pickup_retry_exclusions.get((step.id, env_id), set())
                if owner not in excluded and not bool(conflicts[owner][env_id]):
                    assignments.append(owner)
                else:
                    assignments.append(None)
                    selection_failed[env_id] = True
                continue
            excluded = self._pickup_retry_exclusions.get((step.id, env_id), set())
            available = [
                arm
                for arm in ("left_arm", "right_arm")
                if arm not in excluded and not bool(conflicts[arm][env_id])
            ]
            if not available:
                assignments.append(None)
                selection_failed[env_id] = True
                continue
            preferred = self._preferred_in_place_arm(step, env_id)
            feasible = [
                arm for arm in available if bool(candidates[arm].feasible[env_id])
            ]
            if preferred in feasible:
                assignments.append(preferred)
            elif feasible:
                assignments.append(
                    min(feasible, key=lambda arm: float(candidates[arm].cost[env_id]))
                )
            else:
                live_preferred = self._preferred_live_pickup_arm(step, env_id)
                assignments.append(
                    live_preferred if live_preferred in available else available[0]
                )

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

    def _ensure_serial_group_assignments(
        self,
        group: Mapping[str, Any],
        failed: torch.Tensor,
    ) -> None:
        """Bind a distinct-arm pair even when its operators execute serially."""
        step_ids = [str(value) for value in group.get("semantic_step_ids", ())]
        if len(step_ids) != 2 or any(
            step_id in self._assignments for step_id in step_ids
        ):
            return
        steps = [self.steps[step_id] for step_id in step_ids]
        for candidate_step in steps:
            self._capture_orientation_reference(candidate_step)
        candidates = {
            (candidate_step.id, arm): self._candidate(candidate_step, arm, failed)
            for candidate_step in steps
            for arm in ("left_arm", "right_arm")
        }
        for candidate_step in steps:
            self._report_candidates(
                candidate_step,
                (
                    candidates[(candidate_step.id, "left_arm")],
                    candidates[(candidate_step.id, "right_arm")],
                ),
            )
        assignments = {
            candidate_step.id: [None] * len(failed) for candidate_step in steps
        }
        permutations = (("left_arm", "right_arm"), ("right_arm", "left_arm"))
        for env_id in range(len(failed)):
            if bool(failed[env_id]):
                continue
            ranked: list[tuple[bool, bool, float, float, str, str]] = []
            for first_arm, second_arm in permutations:
                first = candidates[(steps[0].id, first_arm)]
                second = candidates[(steps[1].id, second_arm)]
                feasible = bool(first.feasible[env_id] and second.feasible[env_id])
                required_match = all(
                    candidate_step.actor.get("mode") != "required"
                    or str(candidate_step.actor.get("arm")) == candidate_arm
                    for candidate_step, candidate_arm in (
                        (steps[0], first_arm),
                        (steps[1], second_arm),
                    )
                )
                available = required_match and not bool(
                    self._resource_conflicts(steps[0], first_arm)[env_id]
                    or self._resource_conflicts(steps[1], second_arm)[env_id]
                )
                preferred = (
                    self._preferred_in_place_arm(steps[0], env_id),
                    self._preferred_in_place_arm(steps[1], env_id),
                )
                side_penalty = float(first_arm != preferred[0]) if preferred[0] else 0.0
                side_penalty += (
                    float(second_arm != preferred[1]) if preferred[1] else 0.0
                )
                ranked.append(
                    (
                        not available,
                        not feasible,
                        side_penalty,
                        float(first.cost[env_id] + second.cost[env_id]),
                        first_arm,
                        second_arm,
                    )
                )
            ranked.sort()
            unavailable, _, _, _, first_arm, second_arm = ranked[0]
            if unavailable:
                continue
            assignments[steps[0].id][env_id] = first_arm
            assignments[steps[1].id][env_id] = second_arm
        self._assignments.update(assignments)

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
                score_components=cached.score_components,
                warnings=cached.warnings,
                blockers=cached.blockers,
            )
        feasible = ~failed.clone() & ~self._resource_conflicts(step, arm)
        motion_cost = torch.zeros(
            int(self.env.num_envs),
            dtype=torch.float32,
            device=self.env.device,
        )
        source_pose = self._entity_pose(step.object_uid)
        target_pose = None
        state = self._state_for(step, arm)
        reference_eef_pose = None
        plans: dict[str, tuple[GroundedAction, ActionOutcome]] = {}
        warnings: list[str] = []
        blockers: list[dict[str, Any]] = []
        try:
            with _capture_speculative_warnings() as captured:
                for edge_id in step.edge_ids:
                    edge = self.edges[edge_id]
                    if len(edge.actions) != 1:
                        raise ValueError(
                            "Auto/required arm candidates require one action per edge."
                        )
                    action = edge.actions[0]
                    capability = self.adapter.capabilities.get(
                        str(action.get("atomic_action_class"))
                    )
                    if (
                        step.operator == "handover"
                        and capability.resource_mode == "coordinated_object"
                    ):
                        # A standalone E4 needs a speculative PickUp/staging
                        # prefix to choose and cache its transfer arm. The
                        # actual HandOver is coordinated, however, and must
                        # only be planned from the live post-staging state.
                        break
                    failure_policy = self._edge_failure_policy(edge)
                    try:
                        if capability.state_effect == "hold":
                            grounded = self.grounder.ground(
                                action,
                                step,
                                arm=arm,
                                state=state,
                                reference_eef_pose=reference_eef_pose,
                                orientation_reference_pose=self._orientation_references.get(
                                    step.id
                                ),
                            )
                            grounded = self._with_downstream_targets(
                                step, edge_id, arm, state, grounded
                            )
                            outcome = self.adapter.plan(grounded, state)
                        else:
                            grounded, outcome = self._ground_and_plan_candidates(
                                action,
                                step,
                                arm=arm,
                                state=state,
                                active=feasible & ~failed,
                                reference_eef_pose=reference_eef_pose,
                                orientation_reference_pose=self._orientation_references.get(
                                    step.id
                                ),
                            )
                    except Exception as exc:
                        if failure_policy != "best_effort":
                            blockers.extend(
                                self._candidate_exception_blockers(
                                    step,
                                    edge,
                                    arm,
                                    failed,
                                    exc,
                                )
                            )
                            raise
                        warnings.append(
                            f"{arm} best-effort action could not be planned at "
                            f"{edge_id} ({capability.name}): "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                    plans[edge_id] = (grounded, outcome)
                    if failure_policy != "best_effort":
                        feasible &= outcome.success
                        motion_cost += outcome.cost
                        blockers.extend(
                            self._candidate_outcome_blockers(
                                step,
                                edge,
                                arm,
                                failed,
                                outcome,
                            )
                        )
                    elif not bool(outcome.success.all()):
                        warnings.append(
                            f"{arm} best-effort action degraded at {edge_id} "
                            f"({capability.name}); required suffix remains feasible."
                        )
                    state = outcome.next_state
                    target = outcome.grounded.target_object_pose
                    if isinstance(target, torch.Tensor):
                        reference_eef_pose = self._eef_target(outcome)
                        binding = edge.actions[0].get("target_binding", {})
                        if (
                            binding.get("kind")
                            in {
                                "semantic_goal",
                                "coordinated_goal",
                            }
                            and binding.get("phase", "final") != "staging"
                        ):
                            target_pose = target
                    if not bool((feasible & ~failed).any()):
                        target = getattr(grounded.target, "xpos", None)
                        target_detail = ""
                        if isinstance(target, torch.Tensor) and target.shape[-2:] == (
                            4,
                            4,
                        ):
                            target_z = target[..., 2, 3]
                            target_detail = (
                                f" target_z=[{float(target_z.min()):.3f}, "
                                f"{float(target_z.max()):.3f}]"
                            )
                        warnings.append(
                            f"{arm} candidate became infeasible at {edge_id} "
                            f"({capability.name}).{target_detail}"
                        )
                        break
                warnings.extend(captured)
        except Exception as exc:
            self._candidate_failures[(step.id, arm)] = f"{type(exc).__name__}: {exc}"
            feasible = torch.zeros_like(failed)
            motion_cost[:] = torch.inf
        center_xy, half_width, lateral_axis = self._arm_selection_workspace(step)
        score_components = _score_arm_candidate(
            arm=arm,
            motion_cost=motion_cost,
            source_pose=source_pose,
            target_pose=target_pose,
            workspace_center_xy=center_xy,
            workspace_half_width=half_width,
            robot_lateral_axis=lateral_axis,
            policy=self.runtime_policy.arm_selection,
        )
        cost = score_components["total_cost"]
        candidate = _Candidate(
            feasible=feasible,
            cost=cost,
            plans=plans,
            score_components=score_components,
            warnings=tuple(warnings),
            blockers=tuple(blockers),
        )
        self._candidate_cache[(step.id, arm)] = candidate
        return _Candidate(
            feasible=feasible & ~failed,
            cost=cost,
            plans=plans,
            score_components=score_components,
            warnings=tuple(warnings),
            blockers=tuple(blockers),
        )

    def _ground_and_plan_candidates(
        self,
        action: Mapping[str, Any],
        step: SemanticStep,
        *,
        arm: str,
        state: ExecutionState,
        active: torch.Tensor,
        reference_eef_pose: torch.Tensor | None = None,
        orientation_reference_pose: torch.Tensor | None = None,
    ) -> tuple[GroundedAction, ActionOutcome]:
        """Plan live grounding candidates and retain the best bounded attempt."""
        groundings = self.grounder.ground_candidates(
            action,
            step,
            arm=arm,
            state=state,
            reference_eef_pose=reference_eef_pose,
            orientation_reference_pose=orientation_reference_pose,
        )
        used = self._placement_candidate_history.get((step.id, arm), set())
        selected: tuple[GroundedAction, ActionOutcome] | None = None
        selected_rank: tuple[int, float, int] | None = None
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for ordinal, grounded in enumerate(groundings):
            candidate_index = int(
                grounded.motion_policy.get("placement_candidate_index", ordinal)
            )
            is_placement = "placement_candidate_index" in grounded.motion_policy
            if is_placement and candidate_index in used:
                attempts.append(
                    {
                        "candidate_index": candidate_index,
                        "status": "previously_released",
                    }
                )
                continue
            try:
                outcome = self.adapter.plan(grounded, state)
            except Exception as exc:
                last_error = exc
                attempts.append(
                    {
                        "candidate_index": candidate_index,
                        "status": "planning_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            failed_count = int((active & ~outcome.success).sum())
            active_cost = (
                float(outcome.cost[active].sum()) if bool(active.any()) else 0.0
            )
            rank = (failed_count, active_cost, candidate_index)
            attempts.append(
                {
                    "candidate_index": candidate_index,
                    "status": "planned",
                    "failed_rows": failed_count,
                    "cost": active_cost,
                }
            )
            if selected is None or rank < selected_rank:
                selected = (grounded, outcome)
                selected_rank = rank
            if failed_count == 0:
                break
        if selected is None:
            if last_error is not None:
                raise RuntimeError(
                    "All grounding candidates raised during planning."
                ) from last_error
            raise RuntimeError("No unused grounding candidate remains.")
        grounded, outcome = selected
        outcome = replace(
            outcome,
            planner_trace={
                **outcome.planner_trace,
                "grounding_candidates": attempts,
                "selected_grounding_candidate": int(
                    grounded.motion_policy.get("placement_candidate_index", 0)
                ),
            },
        )
        return grounded, outcome

    def _arm_selection_workspace(
        self,
        step: SemanticStep,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return workspace geometry along the robot's live lateral axis."""
        lateral_axis = self._robot_view_lateral_axis()
        arrangement = self.arrangements.get(step.id)
        if arrangement is not None:
            minimum = arrangement.table_bounds[:, 0, :2]
            maximum = arrangement.table_bounds[:, 1, :2]
            center = (minimum + maximum) * 0.5
            half_extents = (maximum - minimum) * 0.5
            half_width = torch.sum(torch.abs(lateral_axis) * half_extents, dim=1)
            return center, half_width, lateral_axis
        count = int(self.env.num_envs)
        centers = torch.zeros((count, 2), dtype=torch.float32, device=self.env.device)
        half_widths = torch.full(
            (count,),
            float(self.runtime_policy.arm_selection.fallback_workspace_half_width),
            dtype=torch.float32,
            device=self.env.device,
        )
        table = self.env.sim.get_rigid_object("table")
        if table is None or not hasattr(table, "get_vertices"):
            return centers, half_widths, lateral_axis
        table_pose = self._entity_pose("table")
        for env_id in range(count):
            value = table.get_vertices(env_ids=[env_id], scale=True)
            if isinstance(value, (list, tuple)):
                value = value[0]
            vertices = torch.as_tensor(
                value,
                dtype=torch.float32,
                device=self.env.device,
            )
            if vertices.ndim == 3 and vertices.shape[0] == 1:
                vertices = vertices[0]
            if vertices.ndim != 2 or vertices.shape[-1] != 3:
                continue
            world = (
                vertices @ table_pose[env_id, :3, :3].transpose(0, 1)
                + table_pose[env_id, :3, 3]
            )
            minimum = world[:, :2].min(dim=0).values
            maximum = world[:, :2].max(dim=0).values
            center = (minimum + maximum) * 0.5
            lateral = torch.sum((world[:, :2] - center) * lateral_axis[env_id], dim=1)
            half_width = torch.max(torch.abs(lateral))
            if float(half_width) > 1.0e-6:
                centers[env_id] = center
                half_widths[env_id] = half_width
        return centers, half_widths, lateral_axis

    def _robot_view_lateral_axis(self) -> torch.Tensor:
        """Return the normalized world-space axis pointing right-arm to left-arm."""
        if self._robot_lateral_axis_cache is not None:
            return self._robot_lateral_axis_cache
        _, self._robot_lateral_axis_cache = robot_frame_axes(self.env)
        return self._robot_lateral_axis_cache

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
        blockers = tuple(
            deepcopy(item)
            for candidate in candidates
            for item in getattr(candidate, "blockers", ())
        )
        if blockers:
            self._candidate_blockers[step.id] = blockers
        if warning_count or failures:
            feasible = ", ".join(
                f"{int(item.feasible.sum())}/{len(item.feasible)}"
                for item in candidates
            )
            log_info(
                f"Speculative arm candidates for {step.id}: feasible=[{feasible}], "
                f"suppressed_warnings={warning_count}, exceptions={len(failures)}."
            )
            edge_failures = tuple(
                message
                for message in diagnostics
                if "candidate became infeasible" in message
            )
            prioritized = tuple(
                dict.fromkeys((*failures, *edge_failures, *diagnostics))
            )
            for message in prioritized[:3]:
                log_warning(f"Candidate planning for {step.id}: {message}")
        self._reported_candidates.add(step.id)

    def _candidate_outcome_blockers(
        self,
        step: SemanticStep,
        edge: ExecutionEdge,
        arm: str,
        inherited_failed: torch.Tensor,
        outcome: ActionOutcome,
    ) -> list[dict[str, Any]]:
        """Capture the real suffix edge that exhausted bounded planning."""
        failed = ~outcome.success & ~inherited_failed
        action = edge.actions[0]
        return [
            {
                "env_id": int(env_id),
                "node_id": action.get("seed_node_id"),
                "blocking_edge_id": edge.id,
                "atomic_action": str(action.get("atomic_action_class")),
                "arm": arm,
                "failure_policy": self._edge_failure_policy(edge),
                "planning_stage": "candidate_suffix",
                **self._planner_failure_details(outcome.planner_trace, env_id),
            }
            for env_id in torch.nonzero(failed, as_tuple=False).flatten().tolist()
        ]

    def _candidate_exception_blockers(
        self,
        step: SemanticStep,
        edge: ExecutionEdge,
        arm: str,
        inherited_failed: torch.Tensor,
        exc: Exception,
    ) -> list[dict[str, Any]]:
        """Record a bounded candidate-planning exception without claiming proof."""
        del step
        action = edge.actions[0]
        budget = self._planner_search_budget()
        return [
            {
                "env_id": int(env_id),
                "node_id": action.get("seed_node_id"),
                "blocking_edge_id": edge.id,
                "atomic_action": str(action.get("atomic_action_class")),
                "arm": arm,
                "failure_policy": self._edge_failure_policy(edge),
                "planning_stage": "candidate_suffix",
                "search_strategy": "planner_exception",
                "search_budget": budget,
                "evidence": {"exception": f"{type(exc).__name__}: {exc}"},
            }
            for env_id in torch.nonzero(~inherited_failed, as_tuple=False)
            .flatten()
            .tolist()
        ]

    def _planner_search_budget(self) -> dict[str, Any]:
        """Return the configured finite search budget used by motion planning."""
        runtime_policy = getattr(self, "runtime_policy", None)
        planner = getattr(runtime_policy, "planner", {})
        curobo = planner.get("curobo", {}) if isinstance(planner, Mapping) else {}
        return {
            "primary_max_attempts": int(curobo.get("max_attempts", 1)),
            "fallback_enabled": bool(planner.get("allow_fallback", False)),
        }

    def _planner_failure_details(
        self,
        trace: Mapping[str, Any],
        env_id: int,
    ) -> dict[str, Any]:
        """Extract compact row-local evidence from one planner trace."""
        reachability = trace.get("reachability_search")
        reachability = reachability if isinstance(reachability, Mapping) else {}
        strategy = str(
            reachability.get("strategy") or trace.get("primary_strategy") or "unknown"
        )
        budget = deepcopy(
            dict(trace.get("search_budget", self._planner_search_budget()))
        )
        attempts = reachability.get("attempts", ())
        evidence: dict[str, Any] = {
            "primary_success": bool(
                self._row_trace_value(trace.get("primary_success", False), env_id)
            ),
            "fallback_attempted": bool(
                self._row_trace_value(trace.get("fallback_attempted", False), env_id)
            ),
            "fallback_success": bool(
                self._row_trace_value(trace.get("fallback_success", False), env_id)
            ),
        }
        if trace.get("exception") is not None:
            evidence["exception"] = str(trace["exception"])
        if isinstance(attempts, Sequence) and not isinstance(
            attempts, (str, bytes, bytearray)
        ):
            evidence["reachability_attempts"] = [
                {
                    "candidate": str(item.get("candidate", "")),
                    "target_z": self._row_trace_value(item.get("target_z"), env_id),
                    "success": bool(
                        self._row_trace_value(item.get("success", False), env_id)
                    ),
                }
                for item in attempts
                if isinstance(item, Mapping)
            ]
            budget["reachability_candidate_count"] = len(
                evidence["reachability_attempts"]
            )
        return {
            "search_strategy": strategy,
            "search_budget": budget,
            "evidence": evidence,
        }

    @staticmethod
    def _row_trace_value(value: Any, env_id: int) -> Any:
        """Detach one environment row from JSON-like or tensor trace data."""
        if isinstance(value, torch.Tensor):
            detached = value.detach().cpu()
            if detached.ndim == 0:
                return detached.item()
            row = detached[min(env_id, detached.shape[0] - 1)]
            return row.item() if row.ndim == 0 else row.tolist()
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if not value:
                return None
            return deepcopy(value[min(env_id, len(value) - 1)])
        return deepcopy(value)

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
        state: ExecutionState,
        grounded: GroundedAction,
    ) -> GroundedAction:
        """Screen grasp poses against every later held-object target.

        A handover is split across semantic steps: its staging ``MoveHeldObject``
        edge is not part of the pickup step's local edge suffix.  Include that
        first exchange pose here so ``PickUp`` can reject a grasp whose
        ``object_to_eef`` transform makes the later transfer arm unreachable.
        This keeps the screening speculative and bounded; no simulator steps
        are sent while a candidate is being built.
        """
        targets: list[torch.Tensor] = []
        start = step.edge_ids.index(pickup_edge_id) + 1
        for edge_id in step.edge_ids[start:]:
            edge = self.edges[edge_id]
            if len(edge.actions) != 1:
                continue
            action = edge.actions[0]
            if (
                self.adapter.capabilities.get(
                    str(action.get("atomic_action_class"))
                ).target_materializer
                != "semantic_held_object"
            ):
                continue
            future = self.grounder.ground(
                action,
                step,
                arm=arm,
                state=state,
                orientation_reference_pose=self._orientation_references.get(step.id),
            )
            if future.target_object_pose is not None:
                targets.append(future.target_object_pose)
        targets.extend(self._handover_successor_targets(step, arm, state))
        if not targets:
            return grounded
        existing = tuple(grounded.cfg.get("downstream_object_target_poses", ()))
        return replace(
            grounded,
            cfg={
                **grounded.cfg,
                "downstream_object_target_poses": existing + tuple(targets),
            },
        )

    def _handover_successor_targets(
        self,
        step: SemanticStep,
        arm: str,
        state: ExecutionState,
    ) -> list[torch.Tensor]:
        """Return staging poses for handovers downstream of a pickup.

        ``SemanticStep.depends_on`` contains semantic IDs rather than edge IDs,
        so walk the small dependency graph instead of assuming the handover is
        an immediate child.  Only a handover that transfers this object from
        the selected pickup arm is relevant to the grasp screen.
        """
        reachable = {step.id}
        changed = True
        while changed:
            changed = False
            for candidate in self.steps.values():
                if candidate.id in reachable:
                    continue
                if any(dependency in reachable for dependency in candidate.depends_on):
                    reachable.add(candidate.id)
                    changed = True

        targets: list[torch.Tensor] = []
        for successor in self.steps.values():
            if (
                successor.id not in reachable
                or successor.id == step.id
                or successor.operator != "handover"
                or successor.object_uid != step.object_uid
            ):
                continue
            for edge_id in successor.edge_ids:
                edge = self.edges[edge_id]
                if len(edge.actions) != 1:
                    continue
                action = edge.actions[0]
                binding = action.get("target_binding", {})
                if not isinstance(binding, Mapping):
                    continue
                if binding.get("kind") != "handover_staging":
                    continue
                transfer_arm = str(
                    binding.get(
                        "transfer_arm",
                        successor.goal.get("transfer_arm", ""),
                    )
                )
                if transfer_arm != arm:
                    break
                try:
                    grounded = self.grounder.ground(
                        action,
                        successor,
                        arm=arm,
                        state=state,
                        orientation_reference_pose=self._orientation_references.get(
                            successor.id,
                            self._orientation_references.get(step.id),
                        ),
                    )
                except (AttributeError, KeyError, ValueError):
                    # A malformed/incomplete successor must not make an
                    # otherwise valid pickup candidate disappear. The normal
                    # successor execution will report that grounding error.
                    break
                if grounded.target_object_pose is not None:
                    targets.append(grounded.target_object_pose)
                break
        return targets

    def _eef_target(self, outcome: ActionOutcome) -> torch.Tensor | None:
        state = outcome.next_state
        held_object = state.get_held_object(
            arm_control_part(self.env, outcome.grounded.arm)
        )
        object_target = outcome.grounded.target_object_pose
        if object_target is not None and held_object is not None:
            object_to_eef = held_object.object_to_eef.to(
                device=object_target.device,
                dtype=object_target.dtype,
            )
            return torch.bmm(object_target, object_to_eef)
        if held_object is not None:
            return held_object.grasp_xpos
        target = outcome.grounded.target
        return getattr(target, "xpos", None)

    def _state_for(self, step: SemanticStep, arm: str) -> ExecutionState:
        """Refresh qpos while retaining holds across TaskGroup boundaries."""
        cached = self._step_states.get((step.id, arm))
        if cached is None:
            cached = self._object_states.get((step.object_uid, arm))
        live_qpos = self.env.robot.get_qpos().clone()
        if cached is None:
            return ExecutionState(last_qpos=live_qpos)
        return cached.with_updates(last_qpos=live_qpos)

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
        state: ExecutionState,
        successful: torch.Tensor,
    ) -> None:
        capability = self.adapter.capabilities.get(action_class)
        owners = self._object_owners.setdefault(
            step.object_uid, [None] * int(self.env.num_envs)
        )
        if capability.state_effect == "release":
            for env_id in torch.nonzero(successful, as_tuple=False).flatten().tolist():
                if owners[env_id] == arm:
                    owners[env_id] = None
                if self._arm_owners[arm][env_id] == step.object_uid:
                    self._arm_owners[arm][env_id] = None
            if arm not in owners:
                self._object_states.pop((step.object_uid, arm), None)
            return
        held_object = state.get_held_object(arm_control_part(self.env, arm))
        if held_object is None or not bool(successful.any()):
            return
        self._object_states[(step.object_uid, arm)] = state
        if capability.state_effect == "hold":
            self._clear_support_relation(step.object_uid, successful)
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
        if step.goal.get("payloads"):
            # Capture before the first physical action, including an ordinary
            # single-arm pickup. Verification then measures whether every
            # direct payload stayed fixed relative to its carrier.
            self._capture_payloads(step)
        if (
            len(edge.actions) == 1
            and self.adapter.capabilities.get(
                str(edge.actions[0].get("atomic_action_class"))
            ).resource_mode
            == "coordinated_object"
        ):
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
        planner_traces: list[dict[str, Any]] = []
        planning_failed = torch.zeros_like(failed)
        action_class = str(edge.actions[0]["atomic_action_class"])
        capability = self.adapter.capabilities.get(action_class)
        for arm in outcomes:
            if not bool(masks[arm].any()):
                continue
            state = self._state_for(step, arm)
            if capability.state_effect == "hold":
                try:
                    grounded, outcome = self._plan_live_hold(edge, step, arm)
                except Exception as exc:
                    planning_failed |= masks[arm]
                    planner_traces.append(
                        self._live_hold_failure_trace(edge, step, arm, exc)
                    )
                    continue
            else:
                # Re-ground transport and placement from live simulator state.
                grounded, outcome = self._ground_and_plan_candidates(
                    edge.actions[0],
                    step,
                    arm=arm,
                    state=state,
                    active=masks[arm],
                    orientation_reference_pose=self._orientation_references.get(
                        step.id
                    ),
                )
            grounded = outcome.grounded
            outcomes[arm] = outcome
            grounded_items.append(grounded)
            planner_traces.append(outcome.planner_trace)
            self._remember_target(step, grounded)
            placement_index = grounded.motion_policy.get("placement_candidate_index")
            if placement_index is not None and bool(
                (masks[arm] & outcome.success).any()
            ):
                self._placement_candidate_history.setdefault((step.id, arm), set()).add(
                    int(placement_index)
                )
        assigned = masks["left_arm"] | masks["right_arm"]
        if not grounded_items:
            return _EdgeResult(
                [],
                failed | (~failed & ~assigned) | planning_failed,
                [],
                planner_traces,
                executed=torch.zeros_like(failed),
            )
        trajectory, action_success = self.adapter.combine(outcomes, masks)
        active = assigned & action_success & ~failed & ~planning_failed
        actions = self.adapter.execute_trajectory(trajectory, active=active)
        physical_failed = torch.zeros_like(failed)
        for arm, outcome in outcomes.items():
            if outcome is not None:
                successful = masks[arm] & outcome.success & active
                if capability.state_effect == "hold":
                    physical = self._physical_pickup(
                        step.object_uid, arm, outcome.next_state, successful
                    )
                    physical_failed |= successful & ~physical
                    successful = physical
                elif capability.state_effect == "preserve_hold":
                    physical = self._physical_hold(
                        step.object_uid, arm, outcome.next_state, successful
                    )
                    lost = successful & ~physical
                    physical_failed |= lost
                    self._release_ownership(step.object_uid, arm, lost)
                    successful = physical
                if capability.verifier_hook is not None:
                    verified = torch.as_tensor(
                        capability.verifier_hook(
                            executor=self,
                            step=step,
                            arm=arm,
                            outcome=outcome,
                            attempted=successful,
                        ),
                        dtype=torch.bool,
                        device=self.env.device,
                    ).reshape(-1)
                    if verified.numel() != int(self.env.num_envs):
                        raise ValueError(
                            f"AtomicAction {action_class!r} verifier returned "
                            "an invalid vectorized mask."
                        )
                    physical_failed |= successful & ~verified
                    successful &= verified
                committed_state = outcome.state_after(successful)
                if capability.state_effect in {"hold", "preserve_hold"}:
                    committed_state = self._rebase_held_state(
                        step.object_uid,
                        arm,
                        committed_state,
                        successful,
                        from_planned_qpos=capability.state_effect == "preserve_hold",
                    )
                self._step_states[(step.id, arm)] = committed_state
                self._update_ownership(
                    step,
                    arm,
                    action_class,
                    committed_state,
                    successful,
                )
                if capability.verifier == "pressed":
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
            | planning_failed
            | physical_failed
        )
        return _EdgeResult(
            actions,
            edge_failed,
            grounded_items,
            planner_traces,
            active,
        )

    def _plan_live_hold(
        self,
        edge: ExecutionEdge,
        step: SemanticStep,
        arm: str,
    ) -> tuple[GroundedAction, ActionOutcome]:
        """Replace a speculative hold plan with one grounded at execution time."""
        candidate = self._candidate_cache.get((step.id, arm))
        cached_plan_available = candidate is not None and edge.id in candidate.plans
        update_obj_info = getattr(self.env, "update_obj_info", None)
        if callable(update_obj_info):
            update_obj_info()
        object_pose = self._entity_pose(step.object_uid).detach().clone()
        state = self._state_for(step, arm)
        grounded = self.grounder.ground(
            edge.actions[0],
            step,
            arm=arm,
            state=state,
            orientation_reference_pose=self._orientation_references.get(step.id),
        )
        grounded = self._with_downstream_targets(step, edge.id, arm, state, grounded)
        outcome = self.adapter.plan(grounded, state)
        return grounded, replace(
            outcome,
            planner_trace={
                **outcome.planner_trace,
                "execution_replanned_from_live_state": True,
                "speculative_candidate_available": cached_plan_available,
                "speculative_candidate_replaced": cached_plan_available,
                "execution_object_pose": object_pose,
            },
        )

    def _live_hold_failure_trace(
        self,
        edge: ExecutionEdge,
        step: SemanticStep,
        arm: str,
        exc: Exception,
    ) -> dict[str, Any]:
        """Describe a live PickUp planning exception without aborting the task."""
        candidate = self._candidate_cache.get((step.id, arm))
        return {
            "action_class": str(edge.actions[0].get("atomic_action_class")),
            "arm": arm,
            "primary_strategy": "live_pickup_replan",
            "primary_success": torch.zeros(
                int(self.env.num_envs),
                dtype=torch.bool,
                device=self.env.device,
            ),
            "execution_replanned_from_live_state": True,
            "speculative_candidate_available": (
                candidate is not None and edge.id in candidate.plans
            ),
            "speculative_candidate_replaced": False,
            "execution_object_pose": self._entity_pose(step.object_uid)
            .detach()
            .clone(),
            "exception": f"{type(exc).__name__}: {exc}",
        }

    def _physical_pickup(
        self,
        uid: str,
        arm: str,
        state: ExecutionState,
        attempted: torch.Tensor,
    ) -> torch.Tensor:
        owners = list(self._object_owners.get(uid, [None] * int(self.env.num_envs)))
        for env_id in torch.nonzero(attempted, as_tuple=False).flatten().tolist():
            owners[env_id] = arm
        states = dict(self._object_states)
        states[(uid, arm)] = state
        physical = attempted & evaluate_predicate(
            self.env,
            {
                "type": "object_held",
                "object": uid,
                "position_tolerance": self.runtime_policy.predicate_fallbacks[
                    "held_position_tolerance"
                ],
            },
            held_owners={**self._object_owners, uid: owners},
            held_states=states,
        )
        return physical

    def _physical_hold(
        self,
        uid: str,
        arm: str,
        state: ExecutionState,
        attempted: torch.Tensor,
        *,
        owners: Mapping[str, Sequence[str | None]] | None = None,
        states: Mapping[tuple[str, str], ExecutionState] | None = None,
        position_tolerance: float | None = None,
    ) -> torch.Tensor:
        candidate_states = dict(self._object_states if states is None else states)
        candidate_states[(uid, arm)] = state
        return attempted & evaluate_predicate(
            self.env,
            {
                "type": "object_held",
                "object": uid,
                "position_tolerance": (
                    self.runtime_policy.predicate_fallbacks["held_position_tolerance"]
                    if position_tolerance is None
                    else float(position_tolerance)
                ),
                "arm": arm,
            },
            held_owners=self._object_owners if owners is None else owners,
            held_states=candidate_states,
        )

    def _release_ownership(
        self,
        uid: str,
        arm: str,
        lost: torch.Tensor,
    ) -> None:
        owners = self._object_owners.get(uid)
        if owners is None:
            return
        for env_id in torch.nonzero(lost, as_tuple=False).flatten().tolist():
            if owners[env_id] == arm:
                owners[env_id] = None
            if self._arm_owners[arm][env_id] == uid:
                self._arm_owners[arm][env_id] = None
        if arm not in owners:
            self._object_states.pop((uid, arm), None)

    def _execute_coordinated(
        self,
        edge: ExecutionEdge,
        step: SemanticStep,
        failed: torch.Tensor,
    ) -> _EdgeResult:
        action = edge.actions[0]
        action_name = str(action["atomic_action_class"])
        capability = self.adapter.capabilities.get(action_name)
        binding = action.get("target_binding", {})
        transfer_arm = str(binding.get("transfer_arm", "left_arm"))
        accepted_assignments = (
            {"coordinated", transfer_arm}
            if capability.state_effect == "transfer_hold"
            else {"coordinated"}
        )
        assigned = torch.tensor(
            [item in accepted_assignments for item in self._assignments[step.id]],
            dtype=torch.bool,
            device=self.env.device,
        )
        receiver_arm = str(binding.get("receive_arm", "right_arm"))
        receiver_conflict = torch.tensor(
            [
                owner not in {None, step.object_uid}
                for owner in self._arm_owners[receiver_arm]
            ],
            dtype=torch.bool,
            device=self.env.device,
        )
        active = assigned & ~failed & ~receiver_conflict
        if not bool(active.any()):
            return _EdgeResult(
                [],
                failed | (~failed & ~assigned) | receiver_conflict,
                [],
                executed=torch.zeros_like(failed),
            )
        state_key = (
            transfer_arm
            if capability.state_effect == "transfer_hold"
            else "coordinated"
        )
        state = self._state_for(step, state_key)
        if capability.state_effect == "coordinated_release":
            held_objects = dict(state.held_objects)
            for arm in ("left_arm", "right_arm"):
                arm_state = self._step_states.get((step.id, arm))
                if arm_state is None:
                    continue
                control_part = arm_control_part(self.env, arm)
                held_object = arm_state.get_held_object(control_part)
                if held_object is not None:
                    held_objects[control_part] = held_object
            state = state.with_updates(held_objects=held_objects)
        update_obj_info = getattr(self.env, "update_obj_info", None)
        if callable(update_obj_info):
            update_obj_info()
        groundings = self.grounder.ground_candidates(
            action,
            step,
            arm="coordinated",
            state=state,
            orientation_reference_pose=self._orientation_references.get(step.id),
        )
        selected: tuple[GroundedAction, ActionOutcome] | None = None
        selected_warnings: tuple[str, ...] = ()
        best_failure_count = int(active.sum()) + 1
        rejected_warning_count = 0
        for candidate in groundings:
            with _capture_speculative_warnings() as captured:
                candidate_outcome = self.adapter.plan(candidate, state)
            failure_count = int((active & ~candidate_outcome.success).sum())
            if selected is None or failure_count < best_failure_count:
                selected = (candidate, candidate_outcome)
                selected_warnings = tuple(captured)
                best_failure_count = failure_count
            if failure_count == 0:
                if rejected_warning_count:
                    log_info(
                        "Selected a feasible coordinated grounding after "
                        f"suppressing {rejected_warning_count} warnings from "
                        "rejected candidates."
                    )
                break
            rejected_warning_count += len(captured)
        if selected is None:
            raise RuntimeError("Coordinated action grounding produced no candidates.")
        if best_failure_count:
            for message in dict.fromkeys(selected_warnings):
                log_warning(message)
        grounded, outcome = selected
        if capability.state_effect == "transfer_hold":
            outcome = replace(
                outcome,
                planner_trace={
                    **outcome.planner_trace,
                    "execution_replanned_from_live_state": True,
                    "execution_object_pose": self._entity_pose(step.object_uid)
                    .detach()
                    .clone(),
                },
            )
        self._remember_target(step, grounded)
        successful = active & outcome.success
        actions = self.adapter.execute_trajectory(
            outcome.trajectory,
            active=successful,
        )
        physical_failed = torch.zeros_like(failed)
        committed_state = outcome.state_after(successful)
        if capability.state_effect == "coordinated_hold":
            self._clear_support_relation(step.object_uid, successful)
        if capability.state_effect == "transfer_hold":
            if bool(successful.any()):
                current_owners = list(
                    self._object_owners.get(
                        step.object_uid,
                        [None] * int(self.env.num_envs),
                    )
                )
                tentative_owners = list(current_owners)
                for env_id in (
                    torch.nonzero(successful, as_tuple=False).flatten().tolist()
                ):
                    tentative_owners[env_id] = receiver_arm
                tentative_states = dict(self._object_states)
                tentative_states[(step.object_uid, receiver_arm)] = outcome.next_state
                physical = self._physical_hold(
                    step.object_uid,
                    receiver_arm,
                    outcome.next_state,
                    successful,
                    owners={
                        **self._object_owners,
                        step.object_uid: tentative_owners,
                    },
                    states=tentative_states,
                    position_tolerance=min(
                        float(
                            self.runtime_policy.predicate_fallbacks[
                                "held_position_tolerance"
                            ]
                        ),
                        float(
                            grounded.motion_policy.get(
                                "held_position_tolerance",
                                self.runtime_policy.predicate_fallbacks[
                                    "held_position_tolerance"
                                ],
                            )
                        ),
                    ),
                )
                lost = successful & ~physical
                physical_failed |= lost
                successful = physical
                committed_state = outcome.state_after(successful)
                committed_state = self._rebase_held_state(
                    step.object_uid,
                    receiver_arm,
                    committed_state,
                    successful,
                    from_planned_qpos=True,
                )
                committed_owners = list(current_owners)
                for env_id in (
                    torch.nonzero(
                        active & outcome.success,
                        as_tuple=False,
                    )
                    .flatten()
                    .tolist()
                ):
                    if bool(physical[env_id]):
                        committed_owners[env_id] = receiver_arm
                        self._arm_owners[receiver_arm][env_id] = step.object_uid
                    else:
                        committed_owners[env_id] = None
                    self._arm_owners[transfer_arm][env_id] = None
                self._object_owners[step.object_uid] = committed_owners
                if any(owner == receiver_arm for owner in committed_owners):
                    self._step_states[(step.id, receiver_arm)] = committed_state
                    self._object_states[(step.object_uid, receiver_arm)] = (
                        committed_state
                    )
                else:
                    self._object_states.pop((step.object_uid, receiver_arm), None)
                if not any(owner == transfer_arm for owner in committed_owners):
                    self._object_states.pop((step.object_uid, transfer_arm), None)
        self._step_states[(step.id, "coordinated")] = committed_state
        return _EdgeResult(
            actions,
            failed
            | (~failed & ~assigned)
            | (active & ~outcome.success)
            | physical_failed,
            [grounded],
            [outcome.planner_trace],
            active & outcome.success,
        )

    def _rebase_held_state(
        self,
        uid: str,
        arm: str,
        state: ExecutionState,
        mask: torch.Tensor,
        *,
        from_planned_qpos: bool = True,
    ) -> ExecutionState:
        """Refresh a held object's object-to-EEF transform after execution."""
        control_part = arm_control_part(self.env, arm)
        held = state.get_held_object(control_part)
        entity = self.env.sim.get_rigid_object(uid)
        if held is None or entity is None or not bool(mask.any()):
            return state
        if from_planned_qpos:
            # Preserve-hold planning must stay in its terminal qpos/FK frame;
            # get_current_xpos_agent() may still expose the previous command.
            joint_ids = self.env.robot.get_joint_ids(name=control_part)
            eef_pose = self.env.robot.compute_fk(
                state.last_qpos[:, joint_ids],
                name=control_part,
                to_matrix=True,
            )
        else:
            eef_poses = self.env.get_current_xpos_agent()
            eef_pose = eef_poses[0 if arm == "left_arm" else 1]
        eef_pose = torch.as_tensor(
            eef_pose,
            dtype=held.object_to_eef.dtype,
            device=held.object_to_eef.device,
        )
        object_pose = torch.as_tensor(
            entity.get_local_pose(to_matrix=True),
            dtype=eef_pose.dtype,
            device=eef_pose.device,
        )
        if eef_pose.ndim == 2:
            eef_pose = eef_pose.unsqueeze(0).repeat(int(self.env.num_envs), 1, 1)
        if object_pose.ndim == 2:
            object_pose = object_pose.unsqueeze(0).repeat(int(self.env.num_envs), 1, 1)
        selector = mask[:, None, None]
        live_object_to_eef = torch.bmm(torch.linalg.inv(object_pose), eef_pose)
        rebased = HeldObjectState(
            semantics=held.semantics,
            object_to_eef=torch.where(
                selector,
                live_object_to_eef,
                held.object_to_eef,
            ),
            grasp_xpos=torch.where(selector, eef_pose, held.grasp_xpos),
            env_mask=held.env_mask,
        )
        held_objects = dict(state.held_objects)
        held_objects[control_part] = rebased
        return state.with_updates(held_objects=held_objects)

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
            return _EdgeResult(
                [],
                failed | (~failed & ~assigned),
                [],
                executed=torch.zeros_like(failed),
            )
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
                orientation_reference_pose=self._orientation_references.get(step.id),
            )
            outcome = self.adapter.plan(grounded, state)
            outcomes[arm] = outcome
            grounded_items.append(grounded)
        trajectory, action_success = self.adapter.combine(outcomes, masks)
        active = assigned & ~failed & action_success
        actions = self.adapter.execute_trajectory(trajectory, active=active)
        is_coordinated_release = {
            str(
                action.get("target_binding", {}).get(
                    "coordinated_release_role",
                    "",
                )
            )
            for action in edge.actions
        } == {"participant", "commit"} and all(
            action.get("control") == "hand"
            and action.get("target_binding", {}).get("kind") == "joint_state"
            and action.get("target_binding", {}).get("source") == "gripper_open"
            for action in edge.actions
        )
        physical_failed = torch.zeros_like(failed)
        if is_coordinated_release:
            opened = evaluate_predicate(self.env, {"type": "both_grippers_open"})
            released = active & opened
            physical_failed = active & ~opened
            control_parts = (
                arm_control_part(self.env, "left_arm"),
                arm_control_part(self.env, "right_arm"),
            )
            released_task = StateDelta(
                held_object_updates={name: None for name in control_parts}
            ).apply(coordinated_state.to_task_state(), released)
            released_state = ExecutionState.from_task_state(
                released_task,
                last_qpos=self.env.robot.get_qpos().clone(),
            )
            for key in ("coordinated", "left_arm", "right_arm"):
                self._step_states[(step.id, key)] = released_state
        else:
            for arm, outcome in outcomes.items():
                if outcome is not None:
                    self._step_states[(step.id, arm)] = outcome.state_after(
                        active & outcome.success
                    )
        return _EdgeResult(
            actions,
            failed
            | (~failed & ~assigned)
            | (assigned & ~action_success)
            | physical_failed,
            grounded_items,
            [
                outcome.planner_trace
                for outcome in outcomes.values()
                if outcome is not None
            ],
            active,
        )

    def _execute_parallel_pickups(
        self,
        edges: Sequence[ExecutionEdge],
        *,
        failed: torch.Tensor,
    ) -> tuple[dict[str, _EdgeResult], torch.Tensor]:
        steps = [self.step_by_edge[edge.id] for edge in edges]
        for step in steps:
            self._capture_orientation_reference(step)
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
            ranked: list[tuple[bool, bool, float, float, str, str]] = []
            for first_arm, second_arm in permutations:
                first = candidates[(steps[0].id, first_arm)]
                second = candidates[(steps[1].id, second_arm)]
                feasible = bool(first.feasible[env_id] and second.feasible[env_id])
                required_match = all(
                    candidate_step.actor.get("mode") != "required"
                    or str(candidate_step.actor.get("arm")) == candidate_arm
                    for candidate_step, candidate_arm in (
                        (steps[0], first_arm),
                        (steps[1], second_arm),
                    )
                )
                available = required_match and not bool(
                    self._resource_conflicts(steps[0], first_arm)[env_id]
                    or self._resource_conflicts(steps[1], second_arm)[env_id]
                )
                first_preferred = self._preferred_in_place_arm(steps[0], env_id)
                second_preferred = self._preferred_in_place_arm(steps[1], env_id)
                side_penalty = (
                    float(first_arm != first_preferred) if first_preferred else 0.0
                )
                side_penalty += (
                    float(second_arm != second_preferred) if second_preferred else 0.0
                )
                cost = float(first.cost[env_id] + second.cost[env_id])
                ranked.append(
                    (
                        not available,
                        not feasible,
                        side_penalty,
                        cost,
                        first_arm,
                        second_arm,
                    )
                )
            ranked.sort()
            unavailable, _, _, _, first_arm, second_arm = ranked[0]
            if unavailable:
                selection_failed[env_id] = True
                continue
            assignments[steps[0].id][env_id] = first_arm
            assignments[steps[1].id][env_id] = second_arm
        self._assignments.update(assignments)

        base_failed = failed | selection_failed
        results = {
            edge.id: _EdgeResult(
                [],
                base_failed.clone(),
                [],
                executed=torch.zeros_like(failed),
            )
            for edge in edges
        }
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
            parallel_planning_failed = False
            for arm, edge in edge_by_arm.items():
                step = self.step_by_edge[edge.id]
                try:
                    grounded, outcome = self._plan_live_hold(edge, step, arm)
                except Exception as exc:
                    parallel_planning_failed = True
                    results[edge.id].planner_traces.append(
                        self._live_hold_failure_trace(edge, step, arm, exc)
                    )
                    continue
                outcomes[arm] = outcome
                results[edge.id].grounded.append(outcome.grounded)
                results[edge.id].planner_traces.append(outcome.planner_trace)
                if bool((partition & ~outcome.success).any()):
                    parallel_planning_failed = True
            if parallel_planning_failed:
                serial_actions: list[torch.Tensor] = []
                for edge in edges:
                    step = self.step_by_edge[edge.id]
                    serial = self._execute_edge_with_retries(
                        edge,
                        step,
                        failed=~partition,
                    )
                    serial_actions.extend(serial.actions)
                    results[edge.id].grounded.extend(serial.grounded)
                    results[edge.id].planner_traces.extend(serial.planner_traces)
                    results[edge.id].failed = torch.where(
                        partition,
                        serial.failed,
                        results[edge.id].failed,
                    )
                    assert results[edge.id].executed is not None
                    if serial.executed is not None:
                        results[edge.id].executed |= serial.executed
                for edge in edges:
                    results[edge.id].actions.extend(serial_actions)
                continue
            trajectory, action_success = self.adapter.combine(outcomes, masks)
            active = partition & ~base_failed & action_success
            commands = self.adapter.execute_trajectory(trajectory, active=active)
            for edge in edges:
                assert results[edge.id].executed is not None
                results[edge.id].executed |= active
            for arm, edge in edge_by_arm.items():
                step = self.step_by_edge[edge.id]
                outcome = outcomes[arm]
                assert outcome is not None
                attempted = active & outcome.success
                physical = self._physical_pickup(
                    step.object_uid, arm, outcome.next_state, attempted
                )
                committed_state = outcome.state_after(physical)
                committed_state = self._rebase_held_state(
                    step.object_uid,
                    arm,
                    committed_state,
                    physical,
                    from_planned_qpos=False,
                )
                self._step_states[(step.id, arm)] = committed_state
                results[edge.id].failed |= partition & ~physical
                self._update_ownership(
                    step,
                    arm,
                    str(edge.actions[0]["atomic_action_class"]),
                    committed_state,
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
            self._target_poses[step.id] = target.clone()
            self._policies[step.id] = grounded.motion_policy

    def _capture_orientation_reference(self, step: SemanticStep) -> None:
        """Freeze preserve orientation before speculative pickup can disturb it."""
        if (
            compile_orientation_constraint(step.goal).requires_reference
            and step.id not in self._orientation_references
        ):
            predecessor_references = [
                self._orientation_references[predecessor.id]
                for dependency in step.depends_on
                if (predecessor := self.steps.get(dependency)) is not None
                and predecessor.object_uid == step.object_uid
                and predecessor.id in self._orientation_references
            ]
            if predecessor_references:
                self._orientation_references[step.id] = predecessor_references[
                    0
                ].clone()
                return
            self._orientation_references[step.id] = self._entity_pose(
                step.object_uid
            ).clone()

    def _step_runtime_metadata(self, step: SemanticStep) -> list[dict[str, Any]]:
        """Expose the live grounding and allocation decisions for diagnosis."""
        observed_pose = self._entity_pose(step.object_uid)
        assignments = self._assignments.get(
            step.id,
            [None] * int(self.env.num_envs),
        )
        target_pose = self._target_poses.get(step.id)
        orientation_reference = self._orientation_references.get(step.id)
        orientation_error = self._orientation_errors.get(step.id)
        arrangement = self.arrangements.get(step.id)
        result = []
        for env_id, assignment in enumerate(assignments):
            physical_part = assignment
            if assignment in {"left_arm", "right_arm"}:
                physical_part = arm_control_part(self.env, assignment)
            candidate_scores = {}
            for arm in ("left_arm", "right_arm"):
                candidate = self._candidate_cache.get((step.id, arm))
                if candidate is None:
                    candidate_scores[arm] = None
                    continue
                scores = {
                    name: float(values[env_id])
                    for name, values in candidate.score_components.items()
                }
                candidate_scores[arm] = {
                    "feasible": bool(candidate.feasible[env_id]),
                    **scores,
                    "failure": self._candidate_failures.get((step.id, arm)),
                }
            item: dict[str, Any] = {
                "assigned_arm": assignment,
                "physical_control_part": physical_part,
                "observed_object_pose": observed_pose[env_id],
                "final_target_pose": (
                    None if target_pose is None else target_pose[env_id]
                ),
                "orientation_reference_pose": (
                    None
                    if orientation_reference is None
                    else orientation_reference[env_id]
                ),
                "orientation_error": (
                    None
                    if orientation_error is None
                    else float(orientation_error[env_id])
                ),
                "candidate_scores": candidate_scores,
            }
            if arrangement is not None:
                item["arrangement"] = arrangement.metadata(step, env_id)
            result.append(item)
        return result

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
        observed_pose = torch.as_tensor(
            entity.get_local_pose(to_matrix=True),
            dtype=torch.float32,
            device=self.env.device,
        )
        observed = observed_pose[:, :3, 3]
        active = ~failed
        if not bool(active.any()):
            success = torch.zeros_like(failed)
            log_info(f"Skipped verification for {step.id}: no active environments.")
            return failed, success, observed
        relation = (
            normalize_placement_relation(step.goal.get("relation", "on"))
            if step.operator == "place_relative"
            else str(step.goal.get("relation", ""))
        )
        reference = self._support_reference_uid(step)
        postcondition_type = step.postcondition.get("type")
        if postcondition_type in {"object_held", "handover_complete"}:
            # A planned hover target is not evidence that the object remains
            # grasped. Verify live TCP/object geometry and gripper closure.
            satisfied = evaluate_predicate(
                self.env,
                step.postcondition,
                held_owners=self._object_owners,
                held_states=self._object_states,
            )
            satisfied &= self._placement_orientation_satisfied(step, observed_pose)
        elif postcondition_type in {
            "held_by_both_grippers",
            "object_held_by_both_grippers",
        }:
            satisfied = evaluate_predicate(
                self.env,
                step.postcondition,
                coordinated_state=self._step_states.get((step.id, "coordinated")),
            )
            target = self._targets.get(step.id)
            if target is not None:
                policy = self._policies.get(step.id, {})
                tolerance = float(
                    policy.get(
                        "postcondition_tolerance",
                        self.runtime_policy.predicate_fallbacks["position_tolerance"],
                    )
                )
                target = target.to(device=observed.device, dtype=observed.dtype)
                satisfied &= (
                    torch.linalg.vector_norm(observed - target, dim=1) <= tolerance
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
            satisfied = self._support_stable_for(step, reference, active)
            satisfied &= self._support_cycle_free(
                step.object_uid,
                reference,
                active,
            )
        elif step.operator == "orient_object":
            position_anchor = str(step.goal.get("position_anchor", "initial_xy"))
            anchor_pose = None
            if position_anchor == "initial_xy":
                anchor_pose = getattr(self.env, "agent_initial_object_poses", {}).get(
                    step.object_uid
                )
            if anchor_pose is None:
                anchor_pose = self._targets.get(step.id)
            if anchor_pose is None:
                raise ValueError(
                    f"orient_object step {step.id!r} has no {position_anchor} anchor."
                )
            anchor_pose = torch.as_tensor(
                anchor_pose,
                dtype=observed.dtype,
                device=observed.device,
            )
            if anchor_pose.ndim == 2 and anchor_pose.shape == (4, 4):
                anchor_pose = anchor_pose.unsqueeze(0).repeat(
                    int(self.env.num_envs), 1, 1
                )
            target_xy = (
                anchor_pose[:, :2, 3] if anchor_pose.ndim == 3 else anchor_pose[:, :2]
            )
            policy = self._policies.get(step.id, {})
            fallbacks = self.runtime_policy.predicate_fallbacks
            upright = evaluate_predicate(
                self.env,
                {
                    "type": "object_upright",
                    "object": step.object_uid,
                    "local_axis": policy.get("upright_local_axis", "long_axis"),
                    "max_tilt": float(
                        policy.get("upright_max_tilt", fallbacks["upright_max_tilt"])
                    ),
                },
            )
            xy_near_initial = evaluate_predicate(
                self.env,
                {
                    "type": "object_xy_near",
                    "object": step.object_uid,
                    "target_xy": target_xy,
                    "tolerance": float(
                        policy.get("upright_xy_tolerance", fallbacks["xy_tolerance"])
                    ),
                },
            )
            satisfied = upright & xy_near_initial
        elif step.id in self._targets:
            target = self._targets[step.id].to(
                device=observed.device,
                dtype=observed.dtype,
            )
            policy = self._policies.get(step.id, {})
            fallbacks = self.runtime_policy.predicate_fallbacks
            tolerance = float(
                policy.get("postcondition_tolerance", fallbacks["position_tolerance"])
            )
            arrangement = self.arrangements.get(step.id)
            if arrangement is not None:
                # Line membership is a planar relation. Height changes after
                # release (for example a can settling onto another stable face)
                # must not invalidate an otherwise correct row placement.
                delta = torch.abs(observed - target)
                axis_tolerance = float(
                    policy.get(
                        "line_axis_tolerance",
                        fallbacks["line_axis_tolerance"],
                    )
                )
                perpendicular_tolerance = float(
                    policy.get(
                        "line_perpendicular_tolerance",
                        fallbacks["line_perpendicular_tolerance"],
                    )
                )
                satisfied = (delta[:, arrangement.axis_index] <= axis_tolerance) & (
                    delta[:, arrangement.perpendicular_index] <= perpendicular_tolerance
                )
            elif relation in DIRECTIONAL_RELATIONS:
                # Left/right/front/behind constrain the support plane. The
                # grounded release height is a transport target and may differ
                # from the stable height after the object settles.
                satisfied = (
                    torch.linalg.vector_norm(observed[:, :2] - target[:, :2], dim=-1)
                    <= tolerance
                )
            else:
                satisfied = (
                    torch.linalg.vector_norm(observed - target, dim=-1) <= tolerance
                )
        else:
            satisfied = evaluate_predicate(self.env, step.postcondition)
        if relation in DIRECTIONAL_RELATIONS and isinstance(reference, str):
            policy = self._policies.get(step.id, {})
            satisfied &= evaluate_predicate(
                self.env,
                {
                    "type": "object_relative_position",
                    "object": step.object_uid,
                    "reference_object": reference,
                    "relation": relation,
                    "relation_frame": step.goal.get("relation_frame", "world"),
                    "minimum_distance": float(policy.get("relation_clearance", 0.01)),
                },
            )
        verifies_placement_orientation = bool(
            compile_orientation_constraint(step.goal).terms
        ) and (
            postcondition_type == "semantic_goal"
            or self.arrangements.get(step.id) is not None
        )
        if verifies_placement_orientation:
            satisfied &= self._placement_orientation_satisfied(step, observed_pose)
        if step.goal.get("payloads"):
            satisfied &= self._verify_payloads(step)
        success = active & satisfied
        failed = failed | (active & ~satisfied)
        if relation in {"on", "on_top", "on_top_of"} and isinstance(reference, str):
            self._commit_support_relation(step, reference, success)
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
        fallbacks = self.runtime_policy.predicate_fallbacks
        tilt_ok = torch.sum(initial_up * live_up, dim=-1) >= float(
            fallbacks["payload_minimum_upright_cosine"]
        )
        result = tilt_ok
        carrier_entity = self.env.sim.get_rigid_object(step.object_uid)
        for payload in step.goal["payloads"]:
            uid = str(payload["object"])
            expected = torch.bmm(carrier, record[uid])
            observed = self._entity_pose(uid)
            drift_ok = torch.linalg.vector_norm(
                observed[:, :3, 3] - expected[:, :3, 3],
                dim=-1,
            ) <= float(fallbacks["payload_position_tolerance"])
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
                if vertices.ndim == 3:
                    vertices = vertices[0]
                world = (
                    vertices @ carrier[env_id, :3, :3].transpose(0, 1)
                    + carrier[env_id, :3, 3]
                )
                position = observed[env_id, :2, 3]
                margin = float(fallbacks["payload_support_margin"])
                lower = world[:, :2].min(dim=0).values - margin
                upper = world[:, :2].max(dim=0).values + margin
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

    def _entity_motion_stable(self, uid: str) -> torch.Tensor:
        entity = self.env.sim.get_rigid_object(uid)
        if entity is None:
            raise ValueError(f"Unknown rigid object {uid!r}.")

        def velocity(value: Any, name: str) -> torch.Tensor | None:
            if callable(value):
                value = value()
            if value is None:
                return None
            tensor = torch.as_tensor(
                value,
                dtype=torch.float32,
                device=self.env.device,
            )
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0).repeat(int(self.env.num_envs), 1)
            if tensor.shape != (int(self.env.num_envs), 3):
                raise ValueError(
                    f"Rigid object {uid!r} {name} must have shape "
                    f"({int(self.env.num_envs)}, 3)."
                )
            return tensor

        linear = velocity(getattr(entity, "lin_vel", None), "lin_vel")
        angular = velocity(getattr(entity, "ang_vel", None), "ang_vel")
        if linear is None or angular is None:
            body_state = getattr(entity, "body_state", None)
            if callable(body_state):
                body_state = body_state()
            if body_state is not None:
                state = torch.as_tensor(
                    body_state,
                    dtype=torch.float32,
                    device=self.env.device,
                )
                if state.ndim == 1:
                    state = state.unsqueeze(0).repeat(int(self.env.num_envs), 1)
                if state.shape == (int(self.env.num_envs), 13):
                    linear = state[:, 7:10]
                    angular = state[:, 10:13]
        if linear is None or angular is None:
            body_data = getattr(entity, "body_data", None)
            if body_data is not None:
                if linear is None:
                    linear = velocity(getattr(body_data, "lin_vel", None), "lin_vel")
                if angular is None:
                    angular = velocity(getattr(body_data, "ang_vel", None), "ang_vel")
        if linear is None or angular is None:
            return torch.zeros(
                int(self.env.num_envs),
                dtype=torch.bool,
                device=self.env.device,
            )
        return (
            torch.linalg.vector_norm(linear, dim=1)
            <= self.support_linear_velocity_tolerance
        ) & (
            torch.linalg.vector_norm(angular, dim=1)
            <= self.support_angular_velocity_tolerance
        )

    def _support_stable_for(
        self,
        step: SemanticStep,
        support_uid: str,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Require the support relation and low motion across a time window."""
        stable = active.clone()
        for sample_index in range(self.support_stability_samples):
            supported = evaluate_predicate(
                self.env,
                {
                    "type": "object_supported_by",
                    "object": step.object_uid,
                    "support": support_uid,
                },
            )
            stable &= (
                supported
                & self._entity_motion_stable(step.object_uid)
                & self._entity_motion_stable(support_uid)
            )
            if (
                sample_index + 1 < self.support_stability_samples
                and self.support_stability_interval_steps
                and bool(active.any())
            ):
                self.env.sim.update(step=self.support_stability_interval_steps)
        return stable

    def _clear_support_relation(self, object_uid: str, mask: torch.Tensor) -> None:
        relations = self._support_relations.get(object_uid)
        if relations is None:
            return
        for env_id in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            relations[env_id] = None
        if not any(relation is not None for relation in relations):
            self._support_relations.pop(object_uid, None)

    def _support_cycle_free(
        self,
        object_uid: str,
        support_uid: str,
        active: torch.Tensor,
    ) -> torch.Tensor:
        result = active.clone()
        for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
            current = support_uid
            visited: set[str] = set()
            while current and current not in visited:
                if current == object_uid:
                    result[env_id] = False
                    break
                visited.add(current)
                relations = self._support_relations.get(current)
                relation = None if relations is None else relations[env_id]
                current = "" if relation is None else relation.support_uid
        return result

    def _commit_support_relation(
        self,
        step: SemanticStep,
        support_uid: str,
        successful: torch.Tensor,
    ) -> None:
        relations = self._support_relations.setdefault(
            step.object_uid,
            [None] * int(self.env.num_envs),
        )
        relation = _SupportRelation(
            support_uid=support_uid,
            semantic_step_id=step.id,
        )
        for env_id in torch.nonzero(successful, as_tuple=False).flatten().tolist():
            relations[env_id] = relation

    def _placement_orientation_satisfied(
        self,
        step: SemanticStep,
        observed_pose: torch.Tensor,
    ) -> torch.Tensor:
        constraint = compile_orientation_constraint(step.goal)
        satisfied = torch.ones(
            int(self.env.num_envs),
            dtype=torch.bool,
            device=self.env.device,
        )
        if not constraint.terms or (
            step.goal.get("orientation_goal") == "preserve"
            and step.goal.get("relation") == "inside"
        ):
            return satisfied
        policy = self._policies.get(step.id, {})
        fallbacks = self.runtime_policy.predicate_fallbacks
        errors = []
        for term in constraint.terms:
            if isinstance(term, AlignAxisConstraint):
                if term.target_axis != "world_up":
                    raise ValueError(
                        f"Unsupported orientation target axis {term.target_axis!r}."
                    )
                satisfied &= evaluate_predicate(
                    self.env,
                    {
                        "type": "object_upright",
                        "object": step.object_uid,
                        "local_axis": term.local_axis,
                        "directed": term.directed,
                        "max_tilt": float(
                            term.tolerance
                            if term.tolerance is not None
                            else policy.get(
                                "upright_max_tilt", fallbacks["upright_max_tilt"]
                            )
                        ),
                    },
                )
                continue
            if not isinstance(term, MatchRotationConstraint):
                raise TypeError(f"Unsupported orientation term {type(term)!r}.")
            reference_pose = (
                self._orientation_references.get(step.id)
                if term.reference == "step_start"
                else self._target_poses.get(step.id)
            )
            if reference_pose is None:
                satisfied &= False
                continue
            reference_rotation = reference_pose[:, :3, :3].to(
                device=observed_pose.device,
                dtype=observed_pose.dtype,
            )
            relative = torch.bmm(
                reference_rotation.transpose(1, 2),
                observed_pose[:, :3, :3],
            )
            cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
            error = torch.acos(cosine.clamp(-1.0, 1.0))
            errors.append(error)
            satisfied &= error <= float(
                term.tolerance
                if term.tolerance is not None
                else policy.get(
                    "preserve_orientation_tolerance",
                    fallbacks["preserve_orientation_tolerance"],
                )
            )
        if errors:
            self._orientation_errors[step.id] = torch.stack(errors).amax(dim=0)
        return satisfied

    def _revalidate_support_relations(self) -> dict[str, torch.Tensor]:
        active_by_step: dict[str, torch.Tensor] = {}
        for relations in self._support_relations.values():
            for env_id, relation in enumerate(relations):
                if relation is None:
                    continue
                active = active_by_step.setdefault(
                    relation.semantic_step_id,
                    torch.zeros(
                        int(self.env.num_envs),
                        dtype=torch.bool,
                        device=self.env.device,
                    ),
                )
                active[env_id] = True
        failures: dict[str, torch.Tensor] = {}
        for step_id, active in active_by_step.items():
            step = self.steps[step_id]
            support_uid = self._support_reference_uid(step)
            if support_uid is None:
                failures[step_id] = active
                continue
            observed_pose = self._entity_pose(step.object_uid)
            valid = self._support_stable_for(step, support_uid, active)
            valid &= self._placement_orientation_satisfied(step, observed_pose)
            lost = active & ~valid
            if bool(lost.any()):
                failures[step_id] = lost
        return failures

    @staticmethod
    def _support_reference_uid(step: SemanticStep) -> str | None:
        value = step.goal.get("reference_object", step.goal.get("support_object"))
        if isinstance(value, str) and value:
            return value
        if (
            step.postcondition.get("type") == "stack_layer_supported"
            and int(step.goal.get("layer_index", -1)) == 0
        ):
            return "table"
        return None

    @staticmethod
    def _edge_failure_policy(edge: ExecutionEdge) -> str:
        """Return the persisted node policy for one synchronized edge."""
        policies = {
            str(action.get("failure_policy", "task_required"))
            for action in edge.actions
        }
        if not policies <= {"task_required", "safety_required", "best_effort"}:
            raise ValueError(
                f"Edge {edge.id!r} contains unknown failure policies {policies}."
            )
        if len(policies) != 1:
            raise ValueError(
                f"Edge {edge.id!r} mixes incompatible failure policies {policies}."
            )
        return next(iter(policies))
