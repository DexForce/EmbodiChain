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

"""Closed-loop execution session for dynamic atomic-action plans."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import TYPE_CHECKING

import torch

from .effects import StateDelta
from .invocation import (
    ActionInvocation,
    PhaseEffectGateRequirement,
    ResolvedActionRequest,
)
from .bindings import RuntimeEndpointTarget
from .plans import (
    ActionPlan,
    EffectVerificationRequirement,
    TrajectorySegment,
)
from .policies import RecoveryPolicy
from .runtime_commands import RuntimeCommandFrame, TimedCommandSequence
from .state import EntityState, PlanningContext, SceneSnapshot, TaskState
from .tracking import (
    FeedbackTerminalAcceptance,
    TimedTerminalAcceptance,
    TrackingEvaluation,
    TrackingFrame,
    TrackingMetricCfg,
)

if TYPE_CHECKING:
    from .engine import AtomicActionEngine


class ExecutionStatus(str, Enum):
    """Lifecycle status of an execution session."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionEventKind(str, Enum):
    """Structured event categories emitted by :meth:`ExecutionSession.tick`."""

    ACTION_PLANNED = "action_planned"
    INVOCATION_REVISED = "invocation_revised"
    REPLANNED = "replanned"
    TRACKING_DIVERGED = "tracking_diverged"
    TRACKING_FEEDBACK_FAILED = "tracking_feedback_failed"
    TERMINAL_ACCEPTANCE_PENDING = "terminal_acceptance_pending"
    TERMINAL_ACCEPTANCE_FAILED = "terminal_acceptance_failed"
    DYNAMIC_GOAL_CHANGED = "dynamic_goal_changed"
    COLLISION_WORLD_CHANGED = "collision_world_changed"
    ACTION_PLANNING_FAILED = "action_planning_failed"
    ACTION_TIMEOUT = "action_timeout"
    TRAJECTORY_SEGMENT_ENTERED = "trajectory_segment_entered"
    TRAJECTORY_COMPLETED = "trajectory_completed"
    EFFECT_VERIFICATION_REQUIRED = "effect_verification_required"
    EFFECT_VERIFICATION_FAILED = "effect_verification_failed"
    EFFECT_VERIFICATION_TIMEOUT = "effect_verification_timeout"
    PHASE_EFFECT_GATE_REQUIRED = "phase_effect_gate_required"
    PHASE_EFFECT_GATE_SATISFIED = "phase_effect_gate_satisfied"
    PHASE_EFFECT_GATE_FAILED = "phase_effect_gate_failed"
    HELD_OBJECT_LOST = "held_object_lost"
    ACTION_RETRY = "action_retry"
    ACTION_COMPLETED = "action_completed"
    RECOVERY_REQUIRED = "recovery_required"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    ROWS_DEACTIVATED = "rows_deactivated"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"


@dataclass(frozen=True, slots=True, eq=False)
class ExecutionEvent:
    """One timestamped execution or recovery event."""

    kind: ExecutionEventKind
    timestamp: float
    skill_id: str | None
    invocation_id: str | None
    invocation_revision: int
    invocation_index: int
    env_mask: torch.Tensor
    message: str = ""
    segment_name: str | None = None
    failure_code: str | None = None
    retryable: bool | None = None

    def __post_init__(self) -> None:
        if self.timestamp < 0.0:
            raise ValueError("ExecutionEvent.timestamp must be non-negative.")
        if self.invocation_index < 0:
            raise ValueError("invocation_index must be non-negative.")
        if self.invocation_revision < 0:
            raise ValueError("invocation_revision must be non-negative.")
        if not isinstance(self.env_mask, torch.Tensor):
            raise TypeError("env_mask must be a torch.Tensor.")
        if self.env_mask.dtype != torch.bool or self.env_mask.dim() != 1:
            raise ValueError("ExecutionEvent.env_mask must be a 1D bool tensor.")
        for name in ("segment_name", "failure_code"):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not str or not value or value != value.strip()
            ):
                raise ValueError(f"ExecutionEvent.{name} must be non-empty or None.")
        if self.retryable is not None and type(self.retryable) is not bool:
            raise TypeError("ExecutionEvent.retryable must be a bool or None.")
        if (self.failure_code is None) != (self.retryable is None):
            raise ValueError(
                "ExecutionEvent.failure_code and retryable must be set together."
            )
        object.__setattr__(self, "env_mask", self.env_mask.clone())


@dataclass(frozen=True, slots=True, eq=False)
class ExecutionPlanAttempt:
    """Owned inspection snapshot for one installed action plan.

    Recovery can install several plans for one logical invocation.  This value
    preserves the exact scene/collision revisions and trajectory structure of
    every installation, correlated with the session-local attempt generation
    and row-local recovery counters.
    """

    attempt_generation: int
    event_kind: ExecutionEventKind
    planned_at: float
    invocation_index: int
    planned_mask: torch.Tensor
    action_retry_counts: tuple[int, ...]
    replan_counts: tuple[int, ...]
    request: ResolvedActionRequest
    plan: ActionPlan

    def __post_init__(self) -> None:
        if type(self.attempt_generation) is not int or self.attempt_generation < 0:
            raise ValueError("attempt_generation must be a non-negative integer.")
        if self.event_kind not in {
            ExecutionEventKind.ACTION_PLANNED,
            ExecutionEventKind.INVOCATION_REVISED,
            ExecutionEventKind.REPLANNED,
        }:
            raise ValueError("event_kind must describe an installed action plan.")
        if not math.isfinite(self.planned_at) or self.planned_at < 0.0:
            raise ValueError("planned_at must be finite and non-negative.")
        if type(self.invocation_index) is not int or self.invocation_index < 0:
            raise ValueError("invocation_index must be a non-negative integer.")
        if (
            not isinstance(self.planned_mask, torch.Tensor)
            or self.planned_mask.dtype != torch.bool
            or self.planned_mask.dim() != 1
        ):
            raise ValueError("planned_mask must be a one-dimensional bool tensor.")
        retries = tuple(self.action_retry_counts)
        replans = tuple(self.replan_counts)
        batch_size = int(self.planned_mask.numel())
        if len(retries) != batch_size or len(replans) != batch_size:
            raise ValueError("Recovery counters must contain one value per row.")
        if any(type(value) is not int or value < 0 for value in (*retries, *replans)):
            raise ValueError("Recovery counters must be non-negative integers.")
        if not isinstance(self.request, ResolvedActionRequest):
            raise TypeError("request must be a ResolvedActionRequest.")
        if not isinstance(self.plan, ActionPlan):
            raise TypeError("plan must be an ActionPlan.")
        if (
            self.request.skill_id != self.plan.skill_id
            or self.request.invocation_id != self.plan.invocation_id
            or self.request.revision != self.plan.invocation_revision
        ):
            raise ValueError("request identity must match the installed plan.")
        if self.plan.plan_success.shape != self.planned_mask.shape:
            raise ValueError("plan and planned_mask batch shapes must match.")
        if self.plan.plan_success.device != self.planned_mask.device:
            raise ValueError("plan and planned_mask must share a device.")
        object.__setattr__(self, "planned_mask", self.planned_mask.clone())
        object.__setattr__(self, "action_retry_counts", retries)
        object.__setattr__(self, "replan_counts", replans)
        object.__setattr__(self, "request", self.request.snapshot())
        object.__setattr__(self, "plan", self.plan.snapshot())

    def snapshot(self) -> ExecutionPlanAttempt:
        """Return an independently owned plan-attempt trace."""
        return ExecutionPlanAttempt(
            attempt_generation=self.attempt_generation,
            event_kind=self.event_kind,
            planned_at=self.planned_at,
            invocation_index=self.invocation_index,
            planned_mask=self.planned_mask,
            action_retry_counts=self.action_retry_counts,
            replan_counts=self.replan_counts,
            request=self.request,
            plan=self.plan,
        )


@dataclass(frozen=True, slots=True, eq=False)
class EffectVerificationRequest:
    """Typed boundary describing a physical effect awaiting verification.

    ``requested_at`` and ``deadline`` use the same timestamp domain as
    :class:`RobotObservation`. Request-mask shrinkage retains both values;
    only a newly installed plan starts a new attempt deadline.
    ``attempt_generation`` is session-local and remains stable when partial
    resolution or row deactivation replaces only the request ID.
    ``failure_invalidation`` is a core-owned removal-only delta; verification
    results may select failed rows on which to apply it but cannot replace it.
    """

    verification_id: int
    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    invocation_index: int
    attempt_generation: int
    terminal_segment: str | None
    requested_at: float
    deadline: float
    env_mask: torch.Tensor
    expected_effects: StateDelta
    effect_verification: EffectVerificationRequirement | None = None
    failure_invalidation: StateDelta = field(default_factory=StateDelta)

    def __post_init__(self) -> None:
        if type(self.verification_id) is not int or self.verification_id < 0:
            raise ValueError("verification_id must be a non-negative integer.")
        if not isinstance(self.skill_id, str) or not self.skill_id:
            raise ValueError("skill_id must be a non-empty string.")
        if self.invocation_id is not None and (
            not isinstance(self.invocation_id, str) or not self.invocation_id
        ):
            raise ValueError("invocation_id must be a non-empty string or None.")
        if self.invocation_revision < 0:
            raise ValueError("invocation_revision must be non-negative.")
        if self.invocation_index < 0:
            raise ValueError("invocation_index must be non-negative.")
        if type(self.attempt_generation) is not int or self.attempt_generation < 0:
            raise ValueError("attempt_generation must be a non-negative integer.")
        if self.terminal_segment is not None and (
            not isinstance(self.terminal_segment, str) or not self.terminal_segment
        ):
            raise ValueError("terminal_segment must be a non-empty string or None.")
        if not math.isfinite(self.requested_at) or self.requested_at < 0.0:
            raise ValueError("requested_at must be finite and non-negative.")
        if not math.isfinite(self.deadline) or self.deadline < self.requested_at:
            raise ValueError(
                "deadline must be finite and no earlier than requested_at."
            )
        if not isinstance(self.env_mask, torch.Tensor):
            raise TypeError("env_mask must be a torch.Tensor.")
        if self.env_mask.dtype != torch.bool or self.env_mask.dim() != 1:
            raise ValueError("env_mask must be a 1D bool tensor.")
        if not self.env_mask.any():
            raise ValueError("env_mask must contain at least one requested row.")
        if not isinstance(self.expected_effects, StateDelta):
            raise TypeError("expected_effects must be a StateDelta.")
        if (
            self.effect_verification is not None
            and type(self.effect_verification) is not EffectVerificationRequirement
        ):
            raise TypeError(
                "effect_verification must be exactly "
                "EffectVerificationRequirement or None."
            )
        if self.expected_effects.is_empty and self.effect_verification is None:
            raise ValueError(
                "Effect verification requires expected symbolic effects or an "
                "explicit physical-effect requirement."
            )
        if not isinstance(self.failure_invalidation, StateDelta):
            raise TypeError("failure_invalidation must be a StateDelta.")
        if (
            any(
                value is not None
                for value in self.failure_invalidation.held_object_updates.values()
            )
            or any(
                value is not None
                for value in self.failure_invalidation.coordinated_held_object_updates.values()
            )
            or any(
                value is not None
                for value in self.failure_invalidation.articulation_joint_updates.values()
            )
        ):
            raise ValueError(
                "failure_invalidation may only remove previously verified state."
            )
        object.__setattr__(self, "env_mask", self.env_mask.clone())
        object.__setattr__(self, "expected_effects", self.expected_effects.snapshot())
        object.__setattr__(
            self,
            "failure_invalidation",
            self.failure_invalidation.snapshot(),
        )
        object.__setattr__(
            self,
            "effect_verification",
            (
                None
                if self.effect_verification is None
                else self.effect_verification.snapshot()
            ),
        )

    def snapshot(self) -> EffectVerificationRequest:
        """Return a request snapshot with an independently owned row mask."""
        return EffectVerificationRequest(
            verification_id=self.verification_id,
            skill_id=self.skill_id,
            invocation_id=self.invocation_id,
            invocation_revision=self.invocation_revision,
            invocation_index=self.invocation_index,
            attempt_generation=self.attempt_generation,
            terminal_segment=self.terminal_segment,
            requested_at=self.requested_at,
            deadline=self.deadline,
            env_mask=self.env_mask,
            expected_effects=self.expected_effects,
            effect_verification=self.effect_verification,
            failure_invalidation=self.failure_invalidation,
        )


@dataclass(frozen=True, slots=True, eq=False)
class EffectExpectationResult:
    """Current-observation outcome for one physical state expectation.

    ``inverse_satisfied_mask`` is stronger than contradiction: every clause
    must have reached its explicit inverse band for the monitor's complete
    hysteresis window.  It may therefore be used to retain a pre-existing
    relation during failure reconciliation, while a single contradictory
    clause may not.
    """

    expectation_id: str
    satisfied_mask: torch.Tensor
    contradicted_mask: torch.Tensor
    inverse_satisfied_mask: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.expectation_id) is not str
            or not self.expectation_id
            or self.expectation_id != self.expectation_id.strip()
        ):
            raise ValueError(
                "expectation_id must be a non-empty string without outer whitespace."
            )
        for name in (
            "satisfied_mask",
            "contradicted_mask",
            "inverse_satisfied_mask",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        masks = (
            self.satisfied_mask,
            self.contradicted_mask,
            self.inverse_satisfied_mask,
        )
        if any(mask.shape != masks[0].shape for mask in masks[1:]):
            raise ValueError("Expectation-result masks must have equal shapes.")
        if any(mask.device != masks[0].device for mask in masks[1:]):
            raise ValueError("Expectation-result masks must use the same device.")
        if (self.satisfied_mask & self.contradicted_mask).any():
            raise ValueError("satisfied_mask and contradicted_mask must not overlap.")
        if (self.inverse_satisfied_mask & ~self.contradicted_mask).any():
            raise ValueError(
                "inverse_satisfied_mask must be a subset of contradicted_mask."
            )
        for name in (
            "satisfied_mask",
            "contradicted_mask",
            "inverse_satisfied_mask",
        ):
            object.__setattr__(self, name, getattr(self, name).clone())

    def snapshot(self) -> EffectExpectationResult:
        """Return an independently owned expectation outcome."""
        return EffectExpectationResult(
            expectation_id=self.expectation_id,
            satisfied_mask=self.satisfied_mask,
            contradicted_mask=self.contradicted_mask,
            inverse_satisfied_mask=self.inverse_satisfied_mask,
        )


@dataclass(frozen=True, slots=True, eq=False)
class EffectVerificationResult:
    """Correlated per-environment update for one effect boundary.

    Rows absent from both ``success_mask`` and ``failure_mask`` remain
    unresolved. ``invalidation_mask`` and ``retry_mask`` classify only failed
    rows: the former selects the request's core-owned removal delta, while the
    latter authorizes replay of the same invocation. Failed rows outside the
    retry mask require external recovery. This lets one shared batch barrier
    commit verified rows while other rows continue observing the same physical
    effect.
    """

    verification_id: int
    success_mask: torch.Tensor
    failure_mask: torch.Tensor
    invalidation_mask: torch.Tensor
    retry_mask: torch.Tensor
    expectation_results: tuple[EffectExpectationResult, ...] = ()

    def __post_init__(self) -> None:
        if type(self.verification_id) is not int or self.verification_id < 0:
            raise ValueError("verification_id must be a non-negative integer.")
        for name in (
            "success_mask",
            "failure_mask",
            "invalidation_mask",
            "retry_mask",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        masks = (
            self.success_mask,
            self.failure_mask,
            self.invalidation_mask,
            self.retry_mask,
        )
        if any(mask.shape != masks[0].shape for mask in masks[1:]):
            raise ValueError("Effect-result masks must have equal shapes.")
        if any(mask.device != masks[0].device for mask in masks[1:]):
            raise ValueError("Effect-result masks must use the same device.")
        if (self.success_mask & self.failure_mask).any():
            raise ValueError("success_mask and failure_mask must not overlap.")
        if (self.invalidation_mask & ~self.failure_mask).any():
            raise ValueError("invalidation_mask must be a subset of failure_mask.")
        if (self.retry_mask & ~self.failure_mask).any():
            raise ValueError("retry_mask must be a subset of failure_mask.")
        expectation_results = tuple(self.expectation_results)
        if not all(
            type(value) is EffectExpectationResult for value in expectation_results
        ):
            raise TypeError(
                "expectation_results must contain exact EffectExpectationResult values."
            )
        expectation_ids = [value.expectation_id for value in expectation_results]
        if len(set(expectation_ids)) != len(expectation_ids):
            raise ValueError("Effect expectation-result IDs must be unique.")
        if expectation_results:
            expected_success = torch.ones_like(self.success_mask)
            expected_failure = torch.zeros_like(self.failure_mask)
            for value in expectation_results:
                if value.satisfied_mask.shape != self.success_mask.shape:
                    raise ValueError(
                        "Expectation and aggregate result masks must have equal shapes."
                    )
                if value.satisfied_mask.device != self.success_mask.device:
                    raise ValueError(
                        "Expectation and aggregate result masks must use the same device."
                    )
                expected_success &= value.satisfied_mask
                expected_failure |= value.contradicted_mask
            if not torch.equal(self.success_mask, expected_success):
                raise ValueError(
                    "success_mask must equal the conjunction of expectation results."
                )
            if not torch.equal(self.failure_mask, expected_failure):
                raise ValueError(
                    "failure_mask must equal the union of expectation results."
                )
        for name in (
            "success_mask",
            "failure_mask",
            "invalidation_mask",
            "retry_mask",
        ):
            object.__setattr__(self, name, getattr(self, name).clone())
        object.__setattr__(
            self,
            "expectation_results",
            tuple(value.snapshot() for value in expectation_results),
        )


@dataclass(frozen=True, slots=True, eq=False)
class PhaseEffectGateRequest:
    """Correlate a blocking physical-effect check with a segment entry.

    The action's preceding command remains active while the gate is unresolved.
    A gate is scoped to the enclosing action attempt and does not create a
    separate planning, recovery, or timeout budget.

    Args:
        verification_id: Session-local single-use request identity.
        gate_id: Invocation-local stable gate identity.
        skill_id: Registered action skill identity.
        invocation_id: Optional logical invocation correlation identity.
        invocation_revision: Active invocation revision.
        invocation_index: Active invocation position in the session.
        attempt_generation: Installed action-plan attempt generation.
        next_waypoint_index: First command frame blocked by the gate.
        segment_name: Named trajectory segment blocked by the gate.
        requested_at: Request creation time in the observation timestamp domain.
        deadline: Enclosing action deadline in that same timestamp domain.
        env_mask: Active rows that must satisfy the gate together.
    """

    verification_id: int
    gate_id: str
    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    invocation_index: int
    attempt_generation: int
    next_waypoint_index: int
    segment_name: str
    requested_at: float
    deadline: float
    env_mask: torch.Tensor

    def __post_init__(self) -> None:
        for name in (
            "verification_id",
            "invocation_revision",
            "invocation_index",
            "attempt_generation",
            "next_waypoint_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in ("gate_id", "skill_id", "segment_name"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    f"{name} must be a non-empty string without outer whitespace."
                )
        if self.invocation_id is not None and (
            type(self.invocation_id) is not str or not self.invocation_id
        ):
            raise ValueError("invocation_id must be a non-empty string or None.")
        if not math.isfinite(self.requested_at) or self.requested_at < 0.0:
            raise ValueError("requested_at must be finite and non-negative.")
        if not math.isfinite(self.deadline) or self.deadline < self.requested_at:
            raise ValueError(
                "deadline must be finite and no earlier than requested_at."
            )
        if not isinstance(self.env_mask, torch.Tensor):
            raise TypeError("env_mask must be a torch.Tensor.")
        if self.env_mask.dtype != torch.bool or self.env_mask.dim() != 1:
            raise ValueError("env_mask must be a one-dimensional bool tensor.")
        if not self.env_mask.any():
            raise ValueError("env_mask must contain at least one gated row.")
        object.__setattr__(self, "env_mask", self.env_mask.clone())

    def snapshot(self) -> PhaseEffectGateRequest:
        """Return an independently owned gate request."""
        return PhaseEffectGateRequest(
            verification_id=self.verification_id,
            gate_id=self.gate_id,
            skill_id=self.skill_id,
            invocation_id=self.invocation_id,
            invocation_revision=self.invocation_revision,
            invocation_index=self.invocation_index,
            attempt_generation=self.attempt_generation,
            next_waypoint_index=self.next_waypoint_index,
            segment_name=self.segment_name,
            requested_at=self.requested_at,
            deadline=self.deadline,
            env_mask=self.env_mask,
        )


@dataclass(frozen=True, slots=True, eq=False)
class PhaseEffectGateResult:
    """Current-observation decision for one blocking segment-entry gate.

    Rows absent from both decision masks remain unresolved. ``retry_mask`` is
    a subset of failed rows for which replaying the enclosing action remains
    valid; no gate outcome mutates verified task state.

    Args:
        verification_id: Identity copied from the consumed gate request.
        gate_id: Stable gate identity copied from the request.
        attempt_generation: Action attempt copied from the request.
        invocation_index: Session invocation index copied from the request.
        next_waypoint_index: Blocked waypoint copied from the request.
        success_mask: Rows whose current evidence satisfies the gate.
        failure_mask: Rows whose current evidence contradicts the gate.
        retry_mask: Failed rows allowed to retry the enclosing action.
        message: Optional physical-failure diagnostic.
    """

    verification_id: int
    gate_id: str
    attempt_generation: int
    invocation_index: int
    next_waypoint_index: int
    success_mask: torch.Tensor
    failure_mask: torch.Tensor
    retry_mask: torch.Tensor
    message: str = ""

    def __post_init__(self) -> None:
        for name in (
            "verification_id",
            "attempt_generation",
            "invocation_index",
            "next_waypoint_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if (
            type(self.gate_id) is not str
            or not self.gate_id
            or self.gate_id != self.gate_id.strip()
        ):
            raise ValueError(
                "gate_id must be a non-empty string without outer whitespace."
            )
        masks = (self.success_mask, self.failure_mask, self.retry_mask)
        for name, value in zip(
            ("success_mask", "failure_mask", "retry_mask"),
            masks,
            strict=True,
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        if any(value.shape != masks[0].shape for value in masks[1:]):
            raise ValueError("Phase-effect gate masks must have equal shapes.")
        if any(value.device != masks[0].device for value in masks[1:]):
            raise ValueError("Phase-effect gate masks must use the same device.")
        if (self.success_mask & self.failure_mask).any():
            raise ValueError("Gate success and failure masks must not overlap.")
        if (self.retry_mask & ~self.failure_mask).any():
            raise ValueError("retry_mask must be a subset of failure_mask.")
        if type(self.message) is not str:
            raise TypeError("message must be a string.")
        for name in ("success_mask", "failure_mask", "retry_mask"):
            object.__setattr__(self, name, getattr(self, name).clone())


@dataclass(frozen=True, slots=True, eq=False)
class HeldObjectGuardRequest:
    """Describe the next in-flight command boundary for held-object checks.

    A request is correlated to one installed action-plan attempt and one next
    waypoint. The named segment lets an external verifier select phase-aware
    physical evidence without teaching the execution core skill-specific
    phases. ``deadline`` uses the observation timestamp domain.
    """

    verification_id: int
    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    invocation_index: int
    attempt_generation: int
    next_waypoint_index: int
    segment_name: str
    env_mask: torch.Tensor
    allowed_held_object_relations: tuple[tuple[str, str], ...]
    allowed_coordinated_held_object_relations: tuple[tuple[str, str, str], ...]
    deadline: float

    def __post_init__(self) -> None:
        if type(self.verification_id) is not int or self.verification_id < 0:
            raise ValueError("verification_id must be a non-negative integer.")
        if type(self.skill_id) is not str or not self.skill_id:
            raise ValueError("skill_id must be a non-empty string.")
        if self.invocation_id is not None and (
            type(self.invocation_id) is not str or not self.invocation_id
        ):
            raise ValueError("invocation_id must be a non-empty string or None.")
        for name in (
            "invocation_revision",
            "invocation_index",
            "attempt_generation",
            "next_waypoint_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if type(self.segment_name) is not str or not self.segment_name:
            raise ValueError("segment_name must be a non-empty string.")
        if not isinstance(self.env_mask, torch.Tensor):
            raise TypeError("env_mask must be a torch.Tensor.")
        if self.env_mask.dtype != torch.bool or self.env_mask.dim() != 1:
            raise ValueError("env_mask must be a one-dimensional bool tensor.")
        if not self.env_mask.any():
            raise ValueError("env_mask must contain at least one guarded row.")
        held_relations = tuple(self.allowed_held_object_relations)
        if len(set(held_relations)) != len(held_relations) or not all(
            type(value) is tuple
            and len(value) == 2
            and all(type(item) is str and item for item in value)
            for value in held_relations
        ):
            raise ValueError(
                "allowed_held_object_relations must contain unique "
                "(task_state_key, object_id) pairs."
            )
        coordinated_relations = tuple(self.allowed_coordinated_held_object_relations)
        if len(set(coordinated_relations)) != len(coordinated_relations) or not all(
            type(value) is tuple
            and len(value) == 3
            and all(type(item) is str and item for item in value)
            for value in coordinated_relations
        ):
            raise ValueError(
                "allowed_coordinated_held_object_relations must contain unique "
                "(first_key, second_key, object_id) triples."
            )
        if not math.isfinite(self.deadline) or self.deadline < 0.0:
            raise ValueError("deadline must be finite and non-negative.")
        object.__setattr__(self, "env_mask", self.env_mask.clone())
        object.__setattr__(self, "allowed_held_object_relations", held_relations)
        object.__setattr__(
            self,
            "allowed_coordinated_held_object_relations",
            coordinated_relations,
        )

    def snapshot(self) -> HeldObjectGuardRequest:
        """Return an independently owned guard request.

        Returns:
            Request with an independently owned environment mask.
        """
        return HeldObjectGuardRequest(
            verification_id=self.verification_id,
            skill_id=self.skill_id,
            invocation_id=self.invocation_id,
            invocation_revision=self.invocation_revision,
            invocation_index=self.invocation_index,
            attempt_generation=self.attempt_generation,
            next_waypoint_index=self.next_waypoint_index,
            segment_name=self.segment_name,
            env_mask=self.env_mask,
            allowed_held_object_relations=self.allowed_held_object_relations,
            allowed_coordinated_held_object_relations=(
                self.allowed_coordinated_held_object_relations
            ),
            deadline=self.deadline,
        )


@dataclass(frozen=True, slots=True, eq=False)
class HeldObjectGuardResult:
    """Correlated in-flight held-object loss and recovery decision.

    ``state_invalidation`` may only remove single-resource or coordinated
    held-object relations. It is applied to ``failure_mask`` before recovery
    planning, so a retry always observes reconciled symbolic state.
    """

    verification_id: int
    object_id: str
    attempt_generation: int
    invocation_index: int
    next_waypoint_index: int
    failure_mask: torch.Tensor
    state_invalidation: StateDelta
    retry_mask: torch.Tensor
    message: str = ""

    def __post_init__(self) -> None:
        if type(self.verification_id) is not int or self.verification_id < 0:
            raise ValueError("verification_id must be a non-negative integer.")
        if type(self.object_id) is not str or not self.object_id:
            raise ValueError("object_id must be a non-empty string.")
        for name in (
            "attempt_generation",
            "invocation_index",
            "next_waypoint_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in ("failure_mask", "retry_mask"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        if self.failure_mask.shape != self.retry_mask.shape:
            raise ValueError("failure_mask and retry_mask must have equal shapes.")
        if self.failure_mask.device != self.retry_mask.device:
            raise ValueError("failure_mask and retry_mask must use the same device.")
        if (self.retry_mask & ~self.failure_mask).any():
            raise ValueError("retry_mask must be a subset of failure_mask.")
        if not isinstance(self.state_invalidation, StateDelta):
            raise TypeError("state_invalidation must be a StateDelta.")
        if any(
            value is not None
            for value in self.state_invalidation.held_object_updates.values()
        ) or any(
            value is not None
            for value in self.state_invalidation.coordinated_held_object_updates.values()
        ):
            raise ValueError(
                "state_invalidation may only remove held-object relations."
            )
        if self.state_invalidation.articulation_joint_updates:
            raise ValueError(
                "state_invalidation cannot update articulation-joint state."
            )
        has_invalidation = bool(
            self.state_invalidation.held_object_updates
            or self.state_invalidation.coordinated_held_object_updates
        )
        if bool(self.failure_mask.any().item()) != has_invalidation:
            raise ValueError(
                "state_invalidation must contain relation removals exactly when "
                "failure_mask contains failed rows."
            )
        if type(self.message) is not str:
            raise TypeError("message must be a string.")
        object.__setattr__(self, "failure_mask", self.failure_mask.clone())
        object.__setattr__(self, "retry_mask", self.retry_mask.clone())
        object.__setattr__(
            self,
            "state_invalidation",
            self.state_invalidation.snapshot(),
        )


@dataclass(frozen=True, slots=True, eq=False)
class ExecutionTick:
    """Result returned after one closed-loop execution update."""

    status: ExecutionStatus
    eligible_mask: torch.Tensor
    command: RuntimeCommandFrame | None
    hold_targets: tuple[RuntimeEndpointTarget, ...]
    events: tuple[ExecutionEvent, ...]
    task_state: TaskState
    pending_effect: EffectVerificationRequest | None = None
    pending_phase_effect_gate: PhaseEffectGateRequest | None = None

    def __post_init__(self) -> None:
        if self.eligible_mask.dtype != torch.bool or self.eligible_mask.dim() != 1:
            raise ValueError("eligible_mask must be a 1D bool tensor.")
        if self.pending_effect is not None and not isinstance(
            self.pending_effect, EffectVerificationRequest
        ):
            raise TypeError(
                "pending_effect must be an EffectVerificationRequest or None."
            )
        if self.pending_phase_effect_gate is not None and not isinstance(
            self.pending_phase_effect_gate,
            PhaseEffectGateRequest,
        ):
            raise TypeError(
                "pending_phase_effect_gate must be a PhaseEffectGateRequest or None."
            )
        if (
            self.pending_effect is not None
            and self.pending_phase_effect_gate is not None
        ):
            raise ValueError(
                "Terminal effect verification and a phase-effect gate cannot be "
                "pending together."
            )
        if self.command is not None and not isinstance(
            self.command,
            RuntimeCommandFrame,
        ):
            raise TypeError("command must be a RuntimeCommandFrame or None.")
        if isinstance(self.hold_targets, (str, bytes)) or not all(
            isinstance(target, RuntimeEndpointTarget) for target in self.hold_targets
        ):
            raise TypeError("hold_targets must contain RuntimeEndpointTarget values.")
        if self.command is not None and self.hold_targets:
            raise ValueError("A tick cannot send commands and request a hold together.")
        if self.pending_effect is not None:
            if not isinstance(self.pending_effect, EffectVerificationRequest):
                raise TypeError(
                    "pending_effect must be an EffectVerificationRequest or None."
                )
            object.__setattr__(
                self,
                "pending_effect",
                self.pending_effect.snapshot(),
            )
        if self.pending_phase_effect_gate is not None:
            object.__setattr__(
                self,
                "pending_phase_effect_gate",
                self.pending_phase_effect_gate.snapshot(),
            )
        hold_targets: list[RuntimeEndpointTarget] = []
        for target in self.hold_targets:
            snapshot = target.snapshot()
            if type(snapshot) is not type(target) or snapshot is target:
                raise TypeError(
                    "RuntimeEndpointTarget.snapshot() must return an independently "
                    "owned value of the same target type."
                )
            hold_targets.append(snapshot)
        object.__setattr__(self, "eligible_mask", self.eligible_mask.clone())
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "hold_targets", tuple(hold_targets))


class ExecutionSession:
    """Execute grounded invocations incrementally with bounded local recovery.

    The session never steps a simulator itself. Each :meth:`tick` consumes the
    latest observation and scene snapshot and emits at most one synchronized
    endpoint-command frame. A declared physical-effect boundary resolves only
    after the caller supplies a correlated :class:`EffectVerificationResult`,
    and non-empty expected symbolic effects are committed for accepted rows
    only. Higher-level runtimes decide how to produce that result from their
    configured monitor selection.

    Environment eligibility and recovery budgets are tracked per row. The
    waypoint cursor is batch-synchronized: a recoverable row replans the active
    cohort from the latest observation and restarts the action trajectory.
    Calls that mutate the session must be serialized by its owner; the session
    does not provide thread synchronization.
    """

    def __init__(
        self,
        engine: AtomicActionEngine,
        invocations: tuple[ActionInvocation, ...],
        context: PlanningContext,
        *,
        eligible_mask: torch.Tensor | None = None,
    ) -> None:
        if not invocations:
            raise ValueError("ExecutionSession requires at least one invocation.")
        engine._validate_context(context)
        self._engine = engine
        self._requests: tuple[ResolvedActionRequest, ...] = tuple(
            engine._resolve(invocation) for invocation in invocations
        )
        self._task_state = context.task
        self._context = context
        self._invocation_index = 0
        self._waypoint_index = 0
        self._plan: ActionPlan | None = None
        self._active_targets: dict[
            tuple[str, str],
            RuntimeEndpointTarget,
        ] = {}
        self._active_tracking_routes: dict[
            tuple[str, str, str],
            tuple[object, str, str],
        ] = {}
        self._planned_scene = context.scene
        self._action_started_at = context.robot.timestamp
        self._attempt_generation = -1
        self._last_tracking_frame: TrackingFrame | None = None
        self._last_command_mask = torch.zeros(
            context.batch_size, dtype=torch.bool, device=context.robot.qpos.device
        )
        self._tracking_violation_counts = torch.zeros(
            context.batch_size,
            dtype=torch.long,
            device=context.robot.qpos.device,
        )
        self._terminal_acceptance_counts = torch.zeros_like(
            self._tracking_violation_counts
        )
        self._terminal_started_at: float | None = None
        self._terminal_pending_reported = False
        self._eligible = (
            torch.ones_like(self._last_command_mask)
            if eligible_mask is None
            else self._normalize_mask(eligible_mask, "eligible_mask")
        )
        self._pending = self._eligible.clone()
        self._action_retries = torch.zeros(
            context.batch_size, dtype=torch.long, device=context.robot.qpos.device
        )
        self._replans = torch.zeros_like(self._action_retries)
        self._pending_effect: EffectVerificationRequest | None = None
        self._effect_failures = torch.zeros_like(self._eligible)
        self._effect_requested_at: float | None = None
        self._next_effect_verification_id = 0
        self._next_held_object_guard_verification_id = 0
        self._pending_phase_effect_gate: PhaseEffectGateRequest | None = None
        self._satisfied_phase_effect_gates: set[str] = set()
        self._reported_phase_effect_gates: set[str] = set()
        self._next_phase_effect_gate_verification_id = 0
        self._plan_attempts: list[ExecutionPlanAttempt] = []
        self._status = (
            ExecutionStatus.RUNNING if self._eligible.any() else ExecutionStatus.FAILED
        )
        self._queued_events: list[ExecutionEvent] = []
        if self._status is ExecutionStatus.RUNNING:
            self._plan_current(context, ExecutionEventKind.ACTION_PLANNED)
        else:
            self._queued_events.append(
                self._event(
                    ExecutionEventKind.SESSION_FAILED,
                    self._eligible,
                    "No environment was initially eligible for execution.",
                )
            )

    @property
    def status(self) -> ExecutionStatus:
        """Current session status."""
        return self._status

    @property
    def eligible_mask(self) -> torch.Tensor:
        """Rows still eligible to complete the full invocation sequence.

        This is deliberately not named ``success_mask``: while the session is
        running, eligibility does not imply that execution or semantic effects
        have succeeded.
        """
        return self._eligible.clone()

    @property
    def task_state(self) -> TaskState:
        """Verified symbolic task state accumulated by this session."""
        return self._task_state

    @property
    def effect_verification_pending(self) -> bool:
        """Whether the current physical effect still requires verification."""
        return self._pending_effect is not None

    @property
    def pending_effect(self) -> EffectVerificationRequest | None:
        """Owned snapshot of the current effect boundary, when present."""
        return None if self._pending_effect is None else self._pending_effect.snapshot()

    @property
    def phase_effect_gate_request(self) -> PhaseEffectGateRequest | None:
        """Return the blocking gate at the next trajectory-segment entry.

        Returns:
            Owned request snapshot, or ``None`` when the next command is not
            blocked by a physical-effect gate.
        """
        request = self._phase_effect_gate_request()
        return None if request is None else request.snapshot()

    @property
    def held_object_guard_request(self) -> HeldObjectGuardRequest | None:
        """Describe the phase that must be checked before the next command.

        The request remains available while terminal acceptance is settling,
        using the final waypoint and segment identity. Once terminal physical
        effect verification begins, that verifier owns the boundary and this
        property returns ``None``.

        Returns:
            Owned phase-aware guard request, or ``None`` when no command-phase
            guard is active.
        """
        request = self._held_object_guard_request()
        return None if request is None else request.snapshot()

    def deactivate_rows(
        self,
        env_mask: torch.Tensor,
        *,
        reason: str,
    ) -> torch.Tensor:
        """Permanently remove selected rows from this invocation sequence.

        Deactivation is sticky across action barriers and recovery replans.
        The next emitted command frame marks those rows inactive so the command
        sink can apply target-specific safe hold behavior.

        Args:
            env_mask: Rows requested for deactivation.
            reason: Human-readable event message.

        Returns:
            Owned mask of rows that changed from eligible to inactive.

        Raises:
            RuntimeError: If the session is already terminal.
            ValueError: If ``reason`` is empty or the mask shape is invalid.
        """
        if self._status is not ExecutionStatus.RUNNING:
            raise RuntimeError("Only a running execution session can deactivate rows.")
        if type(reason) is not str or not reason:
            raise ValueError("reason must be a non-empty string.")
        requested = self._normalize_mask(env_mask, "env_mask")
        changed = requested & self._eligible
        if not changed.any():
            return changed
        self._eligible &= ~changed
        self._pending &= ~changed
        self._effect_failures &= ~changed
        self._last_command_mask &= ~changed
        self._queued_events.append(
            self._event(ExecutionEventKind.ROWS_DEACTIVATED, changed, reason)
        )
        if self._pending_effect is not None:
            assert self._plan is not None
            previous_effect = self._pending_effect
            remaining_effect = (
                previous_effect.env_mask & self._pending & self._plan.plan_success
            )
            if torch.equal(remaining_effect, previous_effect.env_mask):
                self._pending_effect = previous_effect
            elif remaining_effect.any():
                self._pending_effect = self._effect_verification_request(
                    remaining_effect
                )
            else:
                self._pending_effect = None
        if self._pending_phase_effect_gate is not None:
            self._pending_phase_effect_gate = None
            self._next_phase_effect_gate_verification_id += 1
        terminal_event = self._update_terminal_status()
        if terminal_event is not None:
            self._queued_events.append(terminal_event)
        return changed.clone()

    def revise_current(
        self,
        invocation: ActionInvocation,
        *,
        context: PlanningContext | None = None,
    ) -> None:
        """Replace and replan the current invocation with a newer revision.

        The replacement is resolved into a new immutable request snapshot from
        ``context`` or the session's latest observation. Retry and replan
        budgets restart for the new revision, while verified task state, the
        current batch barrier, and per-environment eligibility are preserved.
        Ordinary recovery replans continue to reuse this snapshot until another
        explicit revision. Once the action owns runtime destinations, the
        replacement must preserve their exact address fingerprints; changing
        controllers or safe-hold footprints requires a new invocation.

        Args:
            invocation: Grounded replacement for the currently active skill.
                Its ``revision`` must be strictly greater than the active one,
                and its ``skill_id`` and ``invocation_id`` must identify the
                same logical call.
            context: Optional fresh observation used to ground the replacement.
                A manually ticked caller may omit it to reuse
                :attr:`latest_context`. Runner-driven code stages revisions on
                :class:`ExecutionRunner`, which supplies a due-time observation.

        Raises:
            TypeError: If ``invocation`` is not an ActionInvocation.
            RuntimeError: If the session is no longer running or a physical
                effect is awaiting verification.
            ValueError: If the replacement identifies another invocation or
                does not advance the revision, or if its plan changes the
                active runtime target addresses.
        """
        replacement = self._prepare_revision(invocation)
        replacement_context = self._context if context is None else context
        self._install_prepared_revision(replacement, replacement_context)

    def _prepare_revision(
        self,
        invocation: ActionInvocation,
    ) -> ResolvedActionRequest:
        """Validate and snapshot one revision without planning or installing it."""
        if not isinstance(invocation, ActionInvocation):
            raise TypeError("invocation must be an ActionInvocation.")
        if self._status is not ExecutionStatus.RUNNING:
            raise RuntimeError("Only a running execution session can be revised.")
        if (
            self._pending_effect is not None
            or self._pending_phase_effect_gate is not None
            or self._effect_failures.any()
        ):
            raise RuntimeError(
                "Cannot revise while physical-effect resolution is pending; "
                "resolve it or cancel and start a new invocation."
            )
        self._validate_revision_identity(
            skill_id=invocation.skill_id,
            invocation_id=invocation.invocation_id,
            revision=invocation.revision,
        )
        return self._engine.resolve(invocation)

    def _install_prepared_revision(
        self,
        replacement: ResolvedActionRequest,
        context: PlanningContext,
    ) -> None:
        """Plan and transactionally install a previously snapshotted revision."""
        if not isinstance(replacement, ResolvedActionRequest):
            raise TypeError("replacement must be a ResolvedActionRequest.")
        if self._status is not ExecutionStatus.RUNNING:
            raise RuntimeError("Only a running execution session can be revised.")
        if (
            self._pending_effect is not None
            or self._pending_phase_effect_gate is not None
            or self._effect_failures.any()
        ):
            raise RuntimeError(
                "Cannot revise while physical-effect resolution is pending; "
                "resolve it or cancel and start a new invocation."
            )
        self._validate_revision_identity(
            skill_id=replacement.skill_id,
            invocation_id=replacement.invocation_id,
            revision=replacement.revision,
        )
        replacement_context = self._validated_context(context)
        replacement_plan = self._engine.plan_request(
            replacement,
            replacement_context,
        )
        self._validate_destination_continuity(
            replacement_plan,
            ExecutionEventKind.INVOCATION_REVISED,
        )
        self._validate_tracking_continuity(
            replacement_plan,
            ExecutionEventKind.INVOCATION_REVISED,
        )
        requests = list(self._requests)
        requests[self._invocation_index] = replacement
        self._requests = tuple(requests)
        self._context = replacement_context
        self._waypoint_index = 0
        self._action_retries.zero_()
        self._replans.zero_()
        self._install_plan(
            replacement_plan,
            replacement_context,
            ExecutionEventKind.INVOCATION_REVISED,
            destination_continuity_validated=True,
        )

    def _validate_revision_identity(
        self,
        *,
        skill_id: str,
        invocation_id: str | None,
        revision: int,
    ) -> None:
        """Validate identity and ordering shared by staged and direct revisions."""
        current = self._requests[self._invocation_index]
        if skill_id != current.skill_id:
            raise ValueError(
                f"Revision skill_id {skill_id!r} does not match "
                f"the active skill {current.skill_id!r}."
            )
        if invocation_id != current.invocation_id:
            raise ValueError(
                "Revision invocation_id must match the active invocation_id."
            )
        if revision <= current.revision:
            raise ValueError(
                f"Revision must advance beyond {current.revision}, got " f"{revision}."
            )

    @property
    def latest_context(self) -> PlanningContext:
        """Latest validated context with the session's verified task state."""
        return self._context

    @property
    def active_commands(self) -> TimedCommandSequence:
        """Return an owned snapshot of the active action command sequence.

        This inspection surface is intended for diagnostics and visualization.
        Mutating the returned tensors cannot affect execution state.
        """
        assert self._plan is not None
        return self._plan.commands.snapshot()

    @property
    def active_plan(self) -> ActionPlan:
        """Return an independently owned snapshot of the active action plan.

        This is a read-only diagnostics boundary for runtime metadata,
        visualization, and tests.  Planning and recovery remain session-owned;
        mutating any tensor in the returned value cannot affect execution.
        """
        assert self._plan is not None
        return self._plan.snapshot()

    @property
    def plan_attempts(self) -> tuple[ExecutionPlanAttempt, ...]:
        """Return every installed plan in deterministic recovery order.

        The initial plan has generation zero.  Each invocation revision,
        recovery replan, or whole-action retry appends a new generation instead
        of replacing earlier scene/collision evidence.
        """
        return tuple(attempt.snapshot() for attempt in self._plan_attempts)

    def trajectory_segment(self, name: str) -> TrajectorySegment:
        """Return named segment metadata for the active action plan.

        Segment ranges are action-local and may change after a replan when a
        backend preserves its own sample count.
        """
        assert self._plan is not None
        return self._plan.segment(name)

    def tick(
        self,
        context: PlanningContext,
        *,
        effect_result: EffectVerificationResult | None = None,
        phase_effect_gate_result: PhaseEffectGateResult | None = None,
        held_object_guard_result: HeldObjectGuardResult | None = None,
    ) -> ExecutionTick:
        """Advance execution by one observation/command cycle.

        Args:
            context: Latest measured robot and versioned scene state. Its task
                state is replaced by the session's verified task state.
            effect_result: Optional correlated semantic-effect result for an
                action waiting at its terminal waypoint.
            phase_effect_gate_result: Optional correlated physical-effect
                decision for a blocked trajectory-segment entry.
            held_object_guard_result: Optional correlated in-flight held-object
                loss result for the current waypoint phase. ``None`` means the
                verifier found no applicable guard for this phase or no result
                was supplied.

        Returns:
            Status, optional command, events, and current verified task state.
        """
        self._context = self._validated_context(context)
        events = self._drain_events()
        if effect_result is not None:
            if type(effect_result) is not EffectVerificationResult:
                raise TypeError(
                    "effect_result must be exactly EffectVerificationResult or None."
                )
            if self._pending_effect is None:
                raise ValueError("No physical effect is awaiting verification.")
            if effect_result.verification_id != self._pending_effect.verification_id:
                raise ValueError(
                    "effect_result verification_id does not match the pending "
                    "effect boundary."
                )
        phase_gate_request = self._phase_effect_gate_request()
        if phase_effect_gate_result is not None:
            if type(phase_effect_gate_result) is not PhaseEffectGateResult:
                raise TypeError(
                    "phase_effect_gate_result must be exactly "
                    "PhaseEffectGateResult or None."
                )
            if phase_gate_request is None:
                raise ValueError("No phase-effect gate is awaiting verification.")
            if phase_effect_gate_result.verification_id != (
                phase_gate_request.verification_id
            ):
                raise ValueError(
                    "phase_effect_gate_result verification_id does not match "
                    "the pending gate."
                )
            for name in (
                "gate_id",
                "attempt_generation",
                "invocation_index",
                "next_waypoint_index",
            ):
                if getattr(phase_effect_gate_result, name) != getattr(
                    phase_gate_request,
                    name,
                ):
                    raise ValueError(
                        f"phase_effect_gate_result {name} does not match the "
                        "pending gate."
                    )
            self._next_phase_effect_gate_verification_id += 1
            self._pending_phase_effect_gate = None
        guard_request = self._held_object_guard_request()
        if held_object_guard_result is not None:
            if type(held_object_guard_result) is not HeldObjectGuardResult:
                raise TypeError(
                    "held_object_guard_result must be exactly "
                    "HeldObjectGuardResult or None."
                )
            if guard_request is None:
                raise ValueError("No held-object guard is active for this phase.")
            if held_object_guard_result.verification_id != (
                guard_request.verification_id
            ):
                raise ValueError(
                    "held_object_guard_result verification_id does not match the "
                    "active guard request."
                )
            for name in (
                "attempt_generation",
                "invocation_index",
                "next_waypoint_index",
            ):
                if getattr(held_object_guard_result, name) != getattr(
                    guard_request,
                    name,
                ):
                    raise ValueError(
                        f"held_object_guard_result {name} does not match the "
                        "active guard request."
                    )
        if guard_request is not None:
            self._next_held_object_guard_verification_id += 1
        if self._status is not ExecutionStatus.RUNNING:
            return self._tick_result(command=None, events=events)

        assert self._plan is not None
        if phase_effect_gate_result is not None:
            assert phase_gate_request is not None
            events.extend(
                self._apply_phase_effect_gate_result(
                    phase_effect_gate_result,
                    phase_gate_request,
                )
            )
            if self._status is not ExecutionStatus.RUNNING:
                return self._tick_result(command=None, events=events)
            assert self._plan is not None
        if held_object_guard_result is not None:
            assert guard_request is not None
            events.extend(
                self._apply_held_object_guard_result(
                    held_object_guard_result,
                    guard_request,
                )
            )
            if self._status is not ExecutionStatus.RUNNING:
                return self._tick_result(command=None, events=events)
            assert self._plan is not None
        if not self._pending.any():
            return self._finish_action_tick(self._pending, None, events)

        if self._pending_effect is not None:
            execution_mask = (
                self._pending_effect.env_mask & self._pending & self._plan.plan_success
            )
            if self._action_timed_out(self._plan, execution_mask):
                pending_request = self._pending_effect
                timed_out = execution_mask.clone()
                known_failures = self._effect_failures.clone()
                planning_failed = self._pending & ~self._plan.plan_success
                invalidation_presence = self._failure_invalidation_presence_mask(
                    pending_request.failure_invalidation
                )
                external_recovery = timed_out & invalidation_presence
                retry_mask = (
                    (timed_out & ~external_recovery) | known_failures | planning_failed
                )
                self._apply_effect_failure_invalidation(
                    pending_request.failure_invalidation,
                    timed_out,
                )
                self._pending_effect = None
                self._effect_failures.zero_()
                if external_recovery.any():
                    self._eligible &= ~external_recovery
                    self._pending &= ~external_recovery
                    self._last_command_mask &= ~external_recovery
                    events.append(
                        self._event(
                            ExecutionEventKind.RECOVERY_REQUIRED,
                            external_recovery,
                            "Effect evidence remained unresolved at the action "
                            "deadline, so previously verified state was "
                            "invalidated before external recovery.",
                        )
                    )
                if known_failures.any():
                    events.append(
                        self._event(
                            ExecutionEventKind.EFFECT_VERIFICATION_FAILED,
                            known_failures,
                            "Required physical effects were not observed.",
                        )
                    )
                if planning_failed.any():
                    events.append(
                        self._event(
                            ExecutionEventKind.ACTION_PLANNING_FAILED,
                            planning_failed,
                            "Planning failed for pending environments.",
                        )
                    )
                events.extend(
                    self._attempt_action_retry(
                        retry_mask,
                        ExecutionEventKind.EFFECT_VERIFICATION_TIMEOUT,
                        "Effect verification exceeded the action attempt timeout.",
                        reason_mask=timed_out,
                    )
                )
                if self._status is not ExecutionStatus.RUNNING:
                    return self._tick_result(command=None, events=events)
                if not self._pending.any():
                    return self._finish_action_tick(self._pending, None, events)
                assert self._plan is not None
                effect_result = None
            else:
                return self._finish_action_tick(
                    execution_mask,
                    effect_result,
                    events,
                )

        if self._effect_failures.any():
            failed_effect = self._effect_failures.clone()
            planning_failed = self._pending & ~self._plan.plan_success
            retry_mask = failed_effect | planning_failed
            self._effect_failures.zero_()
            if planning_failed.any():
                events.append(
                    self._event(
                        ExecutionEventKind.ACTION_PLANNING_FAILED,
                        planning_failed,
                        "Planning failed for pending environments.",
                    )
                )
            events.extend(
                self._attempt_action_retry(
                    retry_mask,
                    ExecutionEventKind.EFFECT_VERIFICATION_FAILED,
                    "Required physical effects were not observed.",
                    reason_mask=failed_effect,
                )
            )
            if self._status is not ExecutionStatus.RUNNING:
                return self._tick_result(command=None, events=events)
            if not self._pending.any():
                return self._finish_action_tick(self._pending, None, events)
            assert self._plan is not None

        plan = self._plan
        execution_mask = self._pending & plan.plan_success
        recovery_events = self._recover_if_needed(plan, execution_mask)
        events.extend(recovery_events)
        if self._status is not ExecutionStatus.RUNNING:
            return self._tick_result(command=None, events=events)
        if recovery_events and any(
            event.kind
            in {
                ExecutionEventKind.REPLANNED,
                ExecutionEventKind.RECOVERY_EXHAUSTED,
                ExecutionEventKind.TRACKING_FEEDBACK_FAILED,
            }
            for event in recovery_events
        ):
            assert self._plan is not None
            plan = self._plan
            execution_mask = self._pending & self._plan.plan_success
        if not self._pending.any():
            return self._finish_action_tick(self._pending, None, events)

        if not execution_mask.any():
            return self._finish_action_tick(execution_mask, None, events)

        phase_gate_request = self._phase_effect_gate_request()
        if phase_gate_request is not None:
            events.extend(self._phase_effect_gate_required_events(phase_gate_request))
            preceding_waypoint = phase_gate_request.next_waypoint_index - 1
            command = self._command_at(plan, preceding_waypoint, execution_mask)
            return self._tick_result(command=command, events=events)

        commands = plan.commands
        if self._waypoint_index < commands.frame_count:
            segment_event = self._segment_entry_event(
                plan,
                self._waypoint_index,
                execution_mask,
            )
            if segment_event is not None:
                events.append(segment_event)
            command = self._command_at(plan, self._waypoint_index, execution_mask)
            self._waypoint_index += 1
            return self._tick_result(command=command, events=events)

        terminal = plan.tracking_policy.terminal
        if self._terminal_started_at is None:
            self._terminal_started_at = self._context.robot.timestamp
        elapsed_terminal = self._context.robot.timestamp - self._terminal_started_at
        terminal_pending = torch.zeros_like(execution_mask)
        if isinstance(terminal, TimedTerminalAcceptance):
            if elapsed_terminal < terminal.settle_duration:
                terminal_pending = execution_mask.clone()
        elif isinstance(terminal, FeedbackTerminalAcceptance):
            if plan.tracking is None or not plan.tracking.frames:
                raise RuntimeError(
                    "Feedback terminal acceptance requires a terminal tracking "
                    "frame."
                )
            try:
                accepted, valid, normalized_error = self._evaluate_tracking_frame(
                    plan.tracking.frames[-1],
                    terminal.metrics,
                )
            except Exception as exc:  # noqa: BLE001 - fail required feedback closed
                events.extend(
                    self._fail_tracking_feedback(
                        execution_mask,
                        "Terminal tracking feedback evaluation failed: "
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                return self._tick_result(command=None, events=events)
            invalid = execution_mask & ~valid
            if invalid.any():
                events.extend(
                    self._fail_tracking_feedback(
                        invalid,
                        "Required terminal tracking feedback was invalid.",
                    )
                )
                if self._status is not ExecutionStatus.RUNNING:
                    return self._tick_result(command=None, events=events)
                execution_mask = self._pending & plan.plan_success
            accepted_now = execution_mask & valid & accepted
            self._terminal_acceptance_counts[accepted_now] += 1
            self._terminal_acceptance_counts[execution_mask & ~accepted_now] = 0
            terminal_pending = execution_mask & (
                self._terminal_acceptance_counts < terminal.consecutive_acceptances
            )
            if terminal_pending.any() and elapsed_terminal >= terminal.settle_timeout:
                max_error = float(normalized_error[terminal_pending].amax().item())
                events.extend(
                    self._attempt_action_retry(
                        terminal_pending,
                        ExecutionEventKind.TERMINAL_ACCEPTANCE_FAILED,
                        "Terminal feedback did not satisfy the acceptance "
                        "contract before its settle timeout "
                        f"(max_normalized_error={max_error:.6f}).",
                    )
                )
                if self._status is not ExecutionStatus.RUNNING:
                    return self._tick_result(command=None, events=events)
                assert self._plan is not None
                plan = self._plan
                execution_mask = self._pending & plan.plan_success
                if not self._pending.any():
                    return self._finish_action_tick(self._pending, None, events)
                if plan.commands.frame_count > 0 and execution_mask.any():
                    segment_event = self._segment_entry_event(
                        plan,
                        0,
                        execution_mask,
                    )
                    if segment_event is not None:
                        events.append(segment_event)
                    command = self._command_at(plan, 0, execution_mask)
                    self._waypoint_index = 1
                    return self._tick_result(command=command, events=events)
                events.append(
                    self._event(
                        ExecutionEventKind.TRAJECTORY_COMPLETED,
                        execution_mask,
                        "Replanned action has no executable command frame.",
                    )
                )
                return self._finish_action_tick(
                    execution_mask,
                    effect_result,
                    events=events,
                )
        else:  # pragma: no cover - TrackingPolicy validates exact alternatives
            raise AssertionError(
                f"Unsupported terminal acceptance {type(terminal).__name__}."
            )

        if terminal_pending.any():
            if not self._terminal_pending_reported:
                events.append(
                    self._event(
                        ExecutionEventKind.TERMINAL_ACCEPTANCE_PENDING,
                        terminal_pending,
                        "Maintaining the terminal command while acceptance is "
                        "pending.",
                    )
                )
                self._terminal_pending_reported = True
            if plan.commands.frame_count == 0:
                raise RuntimeError(
                    "Terminal settling requires an executable terminal command "
                    "frame."
                )
            terminal_command = plan.commands.frames[-1].with_active_mask(
                plan.commands.frames[-1].active_mask & terminal_pending
            )
            return self._tick_result(command=terminal_command, events=events)

        events.append(
            self._event(
                ExecutionEventKind.TRAJECTORY_COMPLETED,
                execution_mask,
                "Action trajectory completed.",
            )
        )

        return self._finish_action_tick(
            execution_mask,
            effect_result,
            events=events,
        )

    def _finish_action_tick(
        self,
        execution_mask: torch.Tensor,
        effect_result: EffectVerificationResult | None,
        events: list[ExecutionEvent],
    ) -> ExecutionTick:
        """Finish the active action and construct its tick result."""
        command, hold_targets, completion_events = self._finish_action(
            execution_mask,
            effect_result,
        )
        events.extend(completion_events)
        return self._tick_result(
            command=command,
            hold_targets=hold_targets,
            events=events,
        )

    def _validated_context(self, context: PlanningContext) -> PlanningContext:
        """Validate one monotonic observation and attach verified task state."""
        self._engine._validate_context(context)
        if context.robot.timestamp < self._context.robot.timestamp:
            raise ValueError("Execution tick timestamps must be monotonic.")
        if context.scene.timestamp < self._context.scene.timestamp:
            raise ValueError("Scene snapshot timestamps must be monotonic.")
        if context.scene.version < self._context.scene.version:
            raise ValueError("Scene snapshot versions must be monotonic.")
        previous_collision_revision = torch.tensor(
            self._context.scene.collision_world_revisions(context.batch_size),
            dtype=torch.long,
            device=context.robot.qpos.device,
        )
        current_collision_revision = torch.tensor(
            context.scene.collision_world_revisions(context.batch_size),
            dtype=torch.long,
            device=context.robot.qpos.device,
        )
        if (current_collision_revision < previous_collision_revision).any():
            raise ValueError("Collision-world revisions must be monotonic.")
        if not torch.equal(context.env_ids, self._context.env_ids):
            raise ValueError("Execution tick env_ids must remain stable and ordered.")
        return replace(context, task=self._task_state)

    def _plan_current(
        self,
        context: PlanningContext,
        event_kind: ExecutionEventKind,
    ) -> None:
        """Plan the current invocation from the latest observation."""
        request = self._requests[self._invocation_index]
        plan = self._engine._plan_request(request, context)
        self._install_plan(plan, context, event_kind)

    def _install_plan(
        self,
        plan: ActionPlan,
        context: PlanningContext,
        event_kind: ExecutionEventKind,
        *,
        destination_continuity_validated: bool = False,
    ) -> None:
        """Install a plan, checking target continuity unless already checked."""
        replacement_targets = {
            (target.transport_id, target.target_id): target
            for target in plan.commands.targets
        }
        replacement_destinations = frozenset(replacement_targets)
        replacement_tracking_routes = self._tracking_routes(plan)
        if not destination_continuity_validated:
            self._validate_destination_continuity(plan, event_kind)
        self._validate_tracking_continuity(plan, event_kind)
        self._validate_phase_effect_gates(plan)
        if (
            event_kind
            not in (
                ExecutionEventKind.REPLANNED,
                ExecutionEventKind.INVOCATION_REVISED,
            )
            or replacement_destinations
        ):
            self._active_targets = replacement_targets
        if (
            event_kind
            not in (
                ExecutionEventKind.REPLANNED,
                ExecutionEventKind.INVOCATION_REVISED,
            )
            or replacement_tracking_routes
        ):
            self._active_tracking_routes = replacement_tracking_routes
        self._plan = plan
        self._attempt_generation += 1
        self._waypoint_index = 0
        self._planned_scene = context.scene
        self._action_started_at = context.robot.timestamp
        self._last_tracking_frame = None
        self._last_command_mask.zero_()
        self._tracking_violation_counts.zero_()
        self._terminal_acceptance_counts.zero_()
        self._terminal_started_at = None
        self._terminal_pending_reported = False
        self._pending_effect = None
        self._effect_failures.zero_()
        self._effect_requested_at = None
        self._pending_phase_effect_gate = None
        self._satisfied_phase_effect_gates.clear()
        self._reported_phase_effect_gates.clear()
        planned_mask = self._pending & plan.plan_success
        self._plan_attempts.append(
            ExecutionPlanAttempt(
                attempt_generation=self._attempt_generation,
                event_kind=event_kind,
                planned_at=context.robot.timestamp,
                invocation_index=self._invocation_index,
                planned_mask=planned_mask,
                action_retry_counts=tuple(
                    int(value) for value in self._action_retries.detach().cpu().tolist()
                ),
                replan_counts=tuple(
                    int(value) for value in self._replans.detach().cpu().tolist()
                ),
                request=self._requests[self._invocation_index],
                plan=plan,
            )
        )
        self._queued_events.append(
            self._event(event_kind, planned_mask, "Planned from the latest context.")
        )
        planning_failed = self._pending & ~plan.plan_success
        failure = plan.diagnostics.failure
        if planning_failed.any() and failure is not None and not failure.retryable:
            self._queued_events.append(
                self._event(
                    ExecutionEventKind.ACTION_PLANNING_FAILED,
                    planning_failed,
                    "Planning failed with a non-retryable classification: "
                    f"{failure.code!r}.",
                    failure_code=failure.code,
                    retryable=False,
                )
            )
            self._eligible &= ~planning_failed
            self._pending &= ~planning_failed
            terminal_event = self._update_terminal_status()
            if terminal_event is not None:
                self._queued_events.append(terminal_event)

    def _validate_phase_effect_gates(self, plan: ActionPlan) -> None:
        """Bind invocation-owned gates to non-initial named plan segments."""
        request = self._requests[self._invocation_index]
        for requirement in request.phase_effect_gates:
            if type(requirement) is not PhaseEffectGateRequirement:
                raise TypeError(
                    "Resolved phase-effect gates must be exact "
                    "PhaseEffectGateRequirement values."
                )
            try:
                segment = plan.segment(requirement.segment_name)
            except KeyError as exc:
                raise ValueError(
                    f"Phase-effect gate {requirement.gate_id!r} references "
                    f"missing segment {requirement.segment_name!r}."
                ) from exc
            if segment.start == 0:
                raise ValueError(
                    f"Phase-effect gate {requirement.gate_id!r} cannot block the "
                    "first trajectory segment because no preceding command exists "
                    "to preserve while evidence is acquired."
                )

    def _validate_destination_continuity(
        self,
        plan: ActionPlan,
        event_kind: ExecutionEventKind,
    ) -> None:
        """Reject in-place plans that change controller or safe-hold ownership."""
        if event_kind not in (
            ExecutionEventKind.REPLANNED,
            ExecutionEventKind.INVOCATION_REVISED,
        ):
            return
        replacement_targets = {
            (target.transport_id, target.target_id): target
            for target in plan.commands.targets
        }
        active_destinations = frozenset(self._active_targets)
        replacement_destinations = frozenset(replacement_targets)
        if not active_destinations:
            return
        if not replacement_destinations:
            if event_kind is ExecutionEventKind.REPLANNED:
                return
            raise ValueError(
                "Invocation revisions must declare the active runtime destination "
                "set; an empty replacement plan cannot prove target continuity."
            )
        if replacement_destinations == active_destinations:
            mismatched_fingerprints = sorted(
                destination
                for destination in active_destinations
                if replacement_targets[destination].address_fingerprint
                != self._active_targets[destination].address_fingerprint
            )
            if not mismatched_fingerprints:
                return
            prefix = (
                "Recovery replans"
                if event_kind is ExecutionEventKind.REPLANNED
                else "Invocation revisions"
            )
            guidance = (
                ""
                if event_kind is ExecutionEventKind.REPLANNED
                else " Start a new invocation to change runtime target addresses."
            )
            raise ValueError(
                f"{prefix} must preserve each runtime target address fingerprint; "
                f"changed={mismatched_fingerprints}.{guidance}"
            )
        if event_kind is ExecutionEventKind.REPLANNED:
            prefix = "Recovery replans"
            guidance = ""
        else:
            prefix = "Invocation revisions"
            guidance = " Start a new invocation to change runtime destinations."
        raise ValueError(
            f"{prefix} must preserve the active runtime destination set; "
            f"previous={sorted(active_destinations)}, "
            f"replacement={sorted(replacement_destinations)}.{guidance}"
        )

    def _validate_tracking_continuity(
        self,
        plan: ActionPlan,
        event_kind: ExecutionEventKind,
    ) -> None:
        """Reject in-place replacement of feedback ownership or projection."""
        if event_kind not in (
            ExecutionEventKind.REPLANNED,
            ExecutionEventKind.INVOCATION_REVISED,
        ):
            return
        if self._plan is None:
            return
        previous_routes = self._active_tracking_routes
        replacement_routes = self._tracking_routes(plan)
        if (
            event_kind is ExecutionEventKind.REPLANNED
            and not plan.commands.targets
            and not replacement_routes
        ):
            return
        if previous_routes == replacement_routes:
            return
        prefix = (
            "Recovery replans"
            if event_kind is ExecutionEventKind.REPLANNED
            else "Invocation revisions"
        )
        raise ValueError(
            f"{prefix} must preserve endpoint tracking source fingerprints and "
            "projector routes; start a new invocation to change feedback "
            "ownership."
        )

    @staticmethod
    def _tracking_routes(
        plan: ActionPlan,
    ) -> dict[tuple[str, str, str], tuple[object, str, str]]:
        """Return the complete feedback/projector route owned by one plan."""
        if plan.tracking is None or not plan.tracking.frames:
            return {}
        return {
            setpoint.key: (
                setpoint.binding.source.source_fingerprint,
                setpoint.binding.projector.projector_id,
                setpoint.binding.projector.revision,
            )
            for setpoint in plan.tracking.frames[0].setpoints
        }

    def _recover_if_needed(
        self,
        plan: ActionPlan,
        execution_mask: torch.Tensor,
    ) -> list[ExecutionEvent]:
        """Detect tracking, scene, and timeout invalidation."""
        events: list[ExecutionEvent] = []
        if not execution_mask.any():
            return events
        if self._action_timed_out(plan, execution_mask):
            return self._attempt_action_retry(
                execution_mask,
                ExecutionEventKind.ACTION_TIMEOUT,
                "Action attempt timeout exceeded.",
            )
        collision_mask = execution_mask & self._collision_world_change_mask(plan)
        if collision_mask.any():
            return self._attempt_replan(
                collision_mask,
                ExecutionEventKind.COLLISION_WORLD_CHANGED,
                "The collision world changed after this trajectory was planned.",
            )
        in_flight = plan.tracking_policy.in_flight
        if (
            in_flight is not None
            and self._last_tracking_frame is not None
            and self._waypoint_index < plan.commands.frame_count
        ):
            tracking_mask = execution_mask & self._last_command_mask
            if (
                tracking_mask.any()
                and self._context.robot.timestamp - self._action_started_at
                >= in_flight.grace_period
            ):
                try:
                    accepted, valid, normalized_error = self._evaluate_tracking_frame(
                        self._last_tracking_frame,
                        in_flight.metrics,
                    )
                except Exception as exc:  # noqa: BLE001 - fail required feedback closed
                    return self._fail_tracking_feedback(
                        tracking_mask,
                        "In-flight tracking feedback evaluation failed: "
                        f"{type(exc).__name__}: {exc}",
                    )
                invalid = tracking_mask & ~valid
                if invalid.any():
                    return self._fail_tracking_feedback(
                        invalid,
                        "Required in-flight tracking feedback was invalid.",
                    )
                violated = tracking_mask & valid & ~accepted
                self._tracking_violation_counts[violated] += 1
                self._tracking_violation_counts[tracking_mask & ~violated] = 0
                diverged = tracking_mask & (
                    self._tracking_violation_counts >= in_flight.consecutive_violations
                )
                if diverged.any():
                    max_error = float(normalized_error[diverged].amax().item())
                    return self._attempt_replan(
                        diverged,
                        ExecutionEventKind.TRACKING_DIVERGED,
                        "Observed in-flight tracking diverged from the commanded "
                        f"setpoint (max_normalized_error={max_error:.6f}).",
                    )
        scene_mask, scene_message = self._dynamic_scene_change(
            plan,
            execution_mask,
        )
        if scene_mask.any():
            assert scene_message is not None
            return self._attempt_replan(
                scene_mask,
                ExecutionEventKind.DYNAMIC_GOAL_CHANGED,
                scene_message,
            )
        return events

    def _action_timed_out(
        self,
        plan: ActionPlan,
        execution_mask: torch.Tensor,
    ) -> bool:
        """Return whether an active action attempt exceeded its deadline."""
        return bool(
            execution_mask.any()
            and self._context.robot.timestamp - self._action_started_at
            > plan.recovery_policy.action_timeout
        )

    def _attempt_replan(
        self,
        trigger_mask: torch.Tensor,
        reason: ExecutionEventKind,
        message: str,
    ) -> list[ExecutionEvent]:
        """Apply per-row budgets and replan the synchronized active cohort."""
        assert self._plan is not None
        plan = self._plan
        events = [self._event(reason, trigger_mask, message)]
        allowed = (
            trigger_mask
            & plan.replannable
            & (self._replans < plan.recovery_policy.max_replans)
        )
        exhausted = trigger_mask & ~allowed
        if exhausted.any():
            self._eligible &= ~exhausted
            self._pending &= ~exhausted
            events.append(
                self._event(
                    ExecutionEventKind.RECOVERY_EXHAUSTED,
                    exhausted,
                    "Local replan budget exhausted.",
                )
            )
        if allowed.any():
            self._replans[allowed] += 1
            self._plan_current(self._context, ExecutionEventKind.REPLANNED)
            events.extend(self._drain_events())
        terminal_event = self._update_terminal_status()
        if terminal_event is not None:
            events.append(terminal_event)
        return events

    def _attempt_action_retry(
        self,
        trigger_mask: torch.Tensor,
        reason: ExecutionEventKind,
        message: str,
        *,
        reason_mask: torch.Tensor | None = None,
    ) -> list[ExecutionEvent]:
        """Retry the current action or permanently fail exhausted rows."""
        assert self._plan is not None
        policy = self._plan.recovery_policy
        cause_mask = trigger_mask if reason_mask is None else reason_mask
        events = [self._event(reason, cause_mask, message)]
        self._pending_effect = None
        self._effect_failures &= ~trigger_mask
        allowed = trigger_mask & (self._action_retries < policy.max_action_retries)
        exhausted = trigger_mask & ~allowed
        if exhausted.any():
            self._eligible &= ~exhausted
            self._pending &= ~exhausted
            events.append(
                self._event(
                    ExecutionEventKind.RECOVERY_EXHAUSTED,
                    exhausted,
                    "Action retry budget exhausted.",
                )
            )
        if allowed.any():
            self._action_retries[allowed] += 1
            self._replans[allowed] = 0
            events.append(
                self._event(
                    ExecutionEventKind.ACTION_RETRY,
                    allowed,
                    "Retrying the action from the latest observation.",
                )
            )
            self._plan_current(self._context, ExecutionEventKind.REPLANNED)
            events.extend(self._drain_events())
        terminal_event = self._update_terminal_status()
        if terminal_event is not None:
            events.append(terminal_event)
        return events

    def _finish_action(
        self,
        execution_mask: torch.Tensor,
        effect_result: EffectVerificationResult | None,
    ) -> tuple[
        RuntimeCommandFrame | None,
        tuple[RuntimeEndpointTarget, ...],
        list[ExecutionEvent],
    ]:
        """Verify effects, update symbolic state, and advance the action barrier."""
        assert self._plan is not None
        plan_targets = self._plan.commands.targets
        active_targets = (
            plan_targets
            if plan_targets
            else tuple(target.snapshot() for target in self._active_targets.values())
        )
        orphaned_targets = bool(active_targets) and not plan_targets
        events: list[ExecutionEvent] = []
        if not self._pending.any():
            hold_targets, barrier_events = self._advance_action_barrier(
                active_targets,
                orphaned_targets=orphaned_targets,
            )
            return None, hold_targets, barrier_events

        planning_failed = self._pending & ~self._plan.plan_success
        if not execution_mask.any() and planning_failed.any():
            events.extend(
                self._attempt_action_retry(
                    planning_failed,
                    ExecutionEventKind.ACTION_PLANNING_FAILED,
                    "Planning failed for every pending environment.",
                )
            )
            if self._status is not ExecutionStatus.RUNNING:
                return None, active_targets, events
            if not self._pending.any():
                hold_targets, barrier_events = self._advance_action_barrier(
                    active_targets,
                    orphaned_targets=orphaned_targets,
                )
                events.extend(barrier_events)
                return None, hold_targets, events
            return None, active_targets, events

        failed_effect = torch.zeros_like(execution_mask)
        unresolved = torch.zeros_like(execution_mask)
        made_progress = False
        if not self._plan.requires_effect_verification:
            verified = execution_mask
        elif effect_result is None:
            if self._pending_effect is None:
                self._pending_effect = self._effect_verification_request(execution_mask)
                events.append(
                    self._event(
                        ExecutionEventKind.EFFECT_VERIFICATION_REQUIRED,
                        execution_mask,
                        "The action requires external physical-effect verification.",
                    )
                )
            return None, active_targets, events
        else:
            assert self._pending_effect is not None
            pending_request = self._pending_effect
            success_input = self._normalize_mask(
                effect_result.success_mask,
                "effect_result.success_mask",
            )
            failure_input = self._normalize_mask(
                effect_result.failure_mask,
                "effect_result.failure_mask",
            )
            invalidation_input = self._normalize_mask(
                effect_result.invalidation_mask,
                "effect_result.invalidation_mask",
            )
            retry_input = self._normalize_mask(
                effect_result.retry_mask,
                "effect_result.retry_mask",
            )
            reported = success_input | failure_input
            if (reported & ~execution_mask).any():
                raise ValueError(
                    "Effect verification masks must be subsets of the pending "
                    "effect request env_mask."
                )
            for outcome in effect_result.expectation_results:
                for name in (
                    "satisfied_mask",
                    "contradicted_mask",
                    "inverse_satisfied_mask",
                ):
                    outcome_mask = self._normalize_mask(
                        getattr(outcome, name),
                        f"effect_result.expectation_results.{name}",
                    )
                    if (outcome_mask & ~execution_mask).any():
                        raise ValueError(
                            "Effect expectation-result masks must be subsets "
                            "of the pending effect request env_mask."
                        )
            verified = execution_mask & success_input
            failed_effect = execution_mask & failure_input
            unresolved = execution_mask & ~reported
            made_progress = bool(reported.any().item())
            invalidated = failed_effect & invalidation_input
            retryable_failure = failed_effect & retry_input
            external_recovery = failed_effect & ~retry_input
            self._apply_effect_failure_invalidation(
                pending_request.failure_invalidation,
                invalidated,
            )
            self._effect_failures |= retryable_failure
            if external_recovery.any():
                self._eligible &= ~external_recovery
                self._pending &= ~external_recovery
                self._effect_failures &= ~external_recovery
                self._last_command_mask &= ~external_recovery
                events.extend(
                    (
                        self._event(
                            ExecutionEventKind.EFFECT_VERIFICATION_FAILED,
                            external_recovery,
                            "Required physical effects were contradicted.",
                        ),
                        self._event(
                            ExecutionEventKind.RECOVERY_REQUIRED,
                            external_recovery,
                            "The reconciled effect failure cannot safely replay "
                            "the current invocation.",
                        ),
                    )
                )
            if not unresolved.any():
                self._pending_effect = None

        if verified.any():
            if not self._plan.expected_effects.is_empty:
                self._task_state = self._plan.expected_effects.apply(
                    self._task_state, verified
                )
                self._context = replace(self._context, task=self._task_state)
            self._pending &= ~verified
        if unresolved.any():
            if made_progress:
                self._pending_effect = self._effect_verification_request(unresolved)
            return None, active_targets, events
        terminal_event = self._update_terminal_status()
        if terminal_event is not None:
            events.append(terminal_event)
            return None, active_targets, events
        retry_candidates = self._effect_failures | planning_failed
        if retry_candidates.any():
            effect_failure_mask = self._effect_failures.clone()
            self._effect_failures.zero_()
            reason = (
                ExecutionEventKind.EFFECT_VERIFICATION_FAILED
                if effect_failure_mask.any()
                else ExecutionEventKind.ACTION_PLANNING_FAILED
            )
            reason_mask = (
                effect_failure_mask if effect_failure_mask.any() else retry_candidates
            )
            if effect_failure_mask.any() and planning_failed.any():
                events.append(
                    self._event(
                        ExecutionEventKind.ACTION_PLANNING_FAILED,
                        planning_failed,
                        "Planning failed for pending environments.",
                    )
                )
            events.extend(
                self._attempt_action_retry(
                    retry_candidates,
                    reason,
                    "Planning or expected-effect verification failed.",
                    reason_mask=reason_mask,
                )
            )
            if self._status is not ExecutionStatus.RUNNING:
                return None, active_targets, events
            if self._pending.any():
                return None, active_targets, events

        if self._pending.any():
            return None, active_targets, events
        hold_targets, barrier_events = self._advance_action_barrier(
            active_targets,
            orphaned_targets=orphaned_targets,
        )
        events.extend(barrier_events)
        return None, hold_targets, events

    def _advance_action_barrier(
        self,
        active_targets: tuple[RuntimeEndpointTarget, ...],
        *,
        orphaned_targets: bool,
    ) -> tuple[tuple[RuntimeEndpointTarget, ...], list[ExecutionEvent]]:
        """Complete an empty action cohort and install the next invocation."""
        if self._status is not ExecutionStatus.RUNNING or self._plan is None:
            raise RuntimeError("Only a running planned action can cross its barrier.")
        if self._pending.any():
            raise RuntimeError("The action barrier cannot advance with pending rows.")
        self._pending_effect = None
        self._effect_failures.zero_()
        self._effect_requested_at = None
        events: list[ExecutionEvent] = []
        events.append(
            self._event(
                ExecutionEventKind.ACTION_COMPLETED,
                self._eligible,
                "Action completed at the batch barrier.",
            )
        )
        self._invocation_index += 1
        if self._invocation_index >= len(self._requests):
            self._status = (
                ExecutionStatus.COMPLETED
                if self._eligible.any()
                else ExecutionStatus.FAILED
            )
            terminal_kind = (
                ExecutionEventKind.SESSION_COMPLETED
                if self._status is ExecutionStatus.COMPLETED
                else ExecutionEventKind.SESSION_FAILED
            )
            events.append(
                self._event(
                    terminal_kind,
                    self._eligible,
                    "Invocation sequence completed.",
                )
            )
            return (active_targets if orphaned_targets else ()), events

        self._pending = self._eligible.clone()
        self._pending_effect = None
        self._effect_failures.zero_()
        self._action_retries.zero_()
        self._replans.zero_()
        self._plan_current(self._context, ExecutionEventKind.ACTION_PLANNED)
        events.extend(self._drain_events())
        return active_targets, events

    def _command_at(
        self,
        plan: ActionPlan,
        waypoint_index: int,
        active_mask: torch.Tensor,
    ) -> RuntimeCommandFrame:
        """Return one frame and retain its generic typed tracking targets."""
        frame = plan.commands.frames[waypoint_index]
        frame = frame.with_active_mask(frame.active_mask & active_mask)
        self._last_tracking_frame = (
            None
            if plan.tracking is None
            else plan.tracking.frames[waypoint_index].snapshot()
        )
        self._last_command_mask = frame.active_mask.clone()
        if waypoint_index == plan.commands.frame_count - 1:
            self._terminal_started_at = self._context.robot.timestamp
            self._terminal_acceptance_counts.zero_()
            self._terminal_pending_reported = False
        return frame

    def _segment_entry_event(
        self,
        plan: ActionPlan,
        waypoint_index: int,
        env_mask: torch.Tensor,
    ) -> ExecutionEvent | None:
        """Report entry into a named trajectory segment exactly once per pass."""
        segment = plan.segment_at(waypoint_index)
        if waypoint_index != segment.start:
            return None
        return self._event(
            ExecutionEventKind.TRAJECTORY_SEGMENT_ENTERED,
            env_mask,
            f"Entered trajectory segment {segment.name!r}.",
            segment_name=segment.name,
        )

    def _evaluate_tracking_frame(
        self,
        frame: TrackingFrame,
        metrics: tuple[TrackingMetricCfg, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate typed endpoint predicates without mixing physical units."""
        evaluations = self._engine.tracking_runtime.evaluate_frame(
            frame,
            metrics,
            self._context,
        )
        accepted = torch.ones_like(self._eligible)
        valid = torch.ones_like(self._eligible)
        normalized_error = torch.zeros(
            self._context.batch_size,
            dtype=self._context.robot.qpos.dtype,
            device=self._context.robot.qpos.device,
        )
        for evaluation in evaluations.values():
            if not isinstance(evaluation, TrackingEvaluation):
                raise TypeError(
                    "TrackingRuntime.evaluate_frame() must return "
                    "TrackingEvaluation values."
                )
            accepted &= evaluation.accepted_mask
            valid &= evaluation.valid_mask
            normalized_error = torch.maximum(
                normalized_error,
                evaluation.normalized_error.to(normalized_error.dtype),
            )
        return accepted, valid, normalized_error

    def _fail_tracking_feedback(
        self,
        failed_mask: torch.Tensor,
        message: str,
    ) -> list[ExecutionEvent]:
        """Fail affected rows closed when required feedback is unavailable."""
        self._eligible &= ~failed_mask
        self._pending &= ~failed_mask
        events = [
            self._event(
                ExecutionEventKind.TRACKING_FEEDBACK_FAILED,
                failed_mask,
                message,
            )
        ]
        terminal_event = self._update_terminal_status()
        if terminal_event is not None:
            events.append(terminal_event)
        return events

    def _dynamic_scene_change(
        self,
        plan: ActionPlan,
        execution_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, str | None]:
        """Detect and describe material scene-dependency invalidation."""
        dependencies = plan.scene_dependencies
        changed = torch.zeros_like(self._eligible)
        dependency_end = plan.scene_dependency_end_segment
        if (
            not dependencies
            or (
                dependency_end is not None
                and self._waypoint_index >= plan.segment(dependency_end).stop
            )
            or self._context.scene.version == self._planned_scene.version
        ):
            return changed, None
        policy = plan.recovery_policy
        details: list[str] = []
        for entity_id in sorted(dependencies):
            monitor_until = plan.scene_dependency_monitor_until.get(entity_id)
            if monitor_until is not None and self._waypoint_index >= monitor_until:
                continue
            previous = self._planned_scene.entities.get(entity_id)
            current = self._context.scene.entities.get(entity_id)
            if previous is None or current is None:
                entity_changed = execution_mask.clone()
                if not entity_changed.any():
                    continue
                changed |= entity_changed
                missing = []
                if previous is None:
                    missing.append("planned_scene")
                if current is None:
                    missing.append("current_scene")
                details.append(
                    self._scene_dependency_change_detail(
                        entity_id=entity_id,
                        monitor_until=monitor_until,
                        policy=policy,
                        max_translation=None,
                        max_rotation=None,
                        missing=",".join(missing),
                    )
                )
                continue
            previous_pose = self._batched_entity_pose(previous)
            current_pose = self._batched_entity_pose(current)
            translation = torch.linalg.vector_norm(
                current_pose[:, :3, 3] - previous_pose[:, :3, 3], dim=1
            )
            relative_rotation = torch.bmm(
                previous_pose[:, :3, :3].transpose(1, 2),
                current_pose[:, :3, :3],
            )
            cosine = (
                (relative_rotation.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0) / 2.0
            ).clamp(-1.0, 1.0)
            rotation = torch.acos(cosine)
            entity_changed = execution_mask & (
                (translation > policy.goal_translation_threshold)
                | (rotation > policy.goal_rotation_threshold)
            )
            if not entity_changed.any():
                continue
            changed |= entity_changed
            details.append(
                self._scene_dependency_change_detail(
                    entity_id=entity_id,
                    monitor_until=monitor_until,
                    policy=policy,
                    max_translation=float(translation[entity_changed].amax().item()),
                    max_rotation=float(rotation[entity_changed].amax().item()),
                    missing=None,
                )
            )
        if not details:
            return changed, None
        return (
            changed,
            "Scene dependency invalidated the active plan at "
            f"waypoint_index={self._waypoint_index}: " + " | ".join(details) + ".",
        )

    @staticmethod
    def _scene_dependency_change_detail(
        *,
        entity_id: str,
        monitor_until: int | None,
        policy: RecoveryPolicy,
        max_translation: float | None,
        max_rotation: float | None,
        missing: str | None,
    ) -> str:
        """Return one stable scene-dependency diagnostic fragment."""
        cutoff = "none" if monitor_until is None else str(monitor_until)
        translation = (
            "unavailable" if max_translation is None else f"{max_translation:.6f}"
        )
        rotation = "unavailable" if max_rotation is None else f"{max_rotation:.6f}"
        missing_detail = "" if missing is None else f", missing={missing}"
        return (
            f"entity_id={entity_id!r}, monitor_cutoff={cutoff}{missing_detail}, "
            f"max_translation={translation}, "
            f"translation_threshold={policy.goal_translation_threshold:.6f}, "
            f"max_rotation={rotation}, "
            f"rotation_threshold={policy.goal_rotation_threshold:.6f}"
        )

    def _collision_world_change_mask(self, plan: ActionPlan) -> torch.Tensor:
        """Detect collision revisions newer than the active action plan."""
        if not plan.collision_world_sensitive:
            return torch.zeros_like(self._eligible)
        current = torch.tensor(
            self._context.scene.collision_world_revisions(self._context.batch_size),
            dtype=torch.long,
            device=self._eligible.device,
        )
        planned = torch.tensor(
            plan.planned_collision_world_revision,
            dtype=torch.long,
            device=self._eligible.device,
        )
        return current > planned

    def _batched_entity_pose(self, state: EntityState) -> torch.Tensor:
        """Broadcast an entity pose to the session batch."""
        pose = state.pose.to(
            device=self._context.robot.qpos.device,
            dtype=self._context.robot.qpos.dtype,
        )
        if pose.shape == (4, 4):
            return pose.unsqueeze(0).expand(self._context.batch_size, -1, -1)
        if pose.shape != (self._context.batch_size, 4, 4):
            raise ValueError("Scene entity pose batch does not match the session.")
        return pose

    def _phase_effect_gate_requirement(
        self,
    ) -> PhaseEffectGateRequirement | None:
        """Resolve a gate exactly at the next named segment's first frame."""
        if (
            self._status is not ExecutionStatus.RUNNING
            or self._plan is None
            or self._pending_effect is not None
            or self._effect_failures.any()
            or self._plan.commands.frame_count == 0
            or self._waypoint_index >= self._plan.commands.frame_count
        ):
            return None
        segment = self._plan.segment_at(self._waypoint_index)
        if self._waypoint_index != segment.start:
            return None
        request = self._requests[self._invocation_index]
        return next(
            (
                value
                for value in request.phase_effect_gates
                if value.segment_name == segment.name
                and value.gate_id not in self._satisfied_phase_effect_gates
            ),
            None,
        )

    def _phase_effect_gate_request(self) -> PhaseEffectGateRequest | None:
        """Build or retain the current blocking segment-entry gate request."""
        requirement = self._phase_effect_gate_requirement()
        if requirement is None:
            self._pending_phase_effect_gate = None
            return None
        assert self._plan is not None
        env_mask = self._pending & self._plan.plan_success
        if not env_mask.any():
            self._pending_phase_effect_gate = None
            return None
        current = self._pending_phase_effect_gate
        if (
            current is not None
            and current.gate_id == requirement.gate_id
            and current.attempt_generation == self._attempt_generation
            and current.next_waypoint_index == self._waypoint_index
            and torch.equal(current.env_mask, env_mask)
        ):
            return current
        invocation = self._requests[self._invocation_index]
        deadline = self._action_started_at + self._plan.recovery_policy.action_timeout
        current = PhaseEffectGateRequest(
            verification_id=self._next_phase_effect_gate_verification_id,
            gate_id=requirement.gate_id,
            skill_id=invocation.skill_id,
            invocation_id=invocation.invocation_id,
            invocation_revision=invocation.revision,
            invocation_index=self._invocation_index,
            attempt_generation=self._attempt_generation,
            next_waypoint_index=self._waypoint_index,
            segment_name=requirement.segment_name,
            requested_at=min(self._context.robot.timestamp, deadline),
            deadline=deadline,
            env_mask=env_mask,
        )
        self._pending_phase_effect_gate = current
        return current

    def _phase_effect_gate_required_events(
        self,
        request: PhaseEffectGateRequest,
    ) -> list[ExecutionEvent]:
        """Emit the gate boundary once per installed action attempt."""
        if request.gate_id in self._reported_phase_effect_gates:
            return []
        self._reported_phase_effect_gates.add(request.gate_id)
        return [
            self._event(
                ExecutionEventKind.PHASE_EFFECT_GATE_REQUIRED,
                request.env_mask,
                f"Physical-effect gate {request.gate_id!r} blocks segment "
                f"{request.segment_name!r} until current evidence succeeds.",
            )
        ]

    def _apply_phase_effect_gate_result(
        self,
        result: PhaseEffectGateResult,
        request: PhaseEffectGateRequest,
    ) -> list[ExecutionEvent]:
        """Resolve one gate observation without mutating verified task state."""
        success = self._normalize_mask(
            result.success_mask,
            "phase_effect_gate_result.success_mask",
        )
        failure = self._normalize_mask(
            result.failure_mask,
            "phase_effect_gate_result.failure_mask",
        )
        retry = self._normalize_mask(
            result.retry_mask,
            "phase_effect_gate_result.retry_mask",
        )
        request_mask = request.env_mask.to(self._eligible.device)
        if ((success | failure | retry) & ~request_mask).any():
            raise ValueError(
                "Phase-effect gate result masks must be subsets of the pending "
                "request env_mask."
            )
        events = self._phase_effect_gate_required_events(request)
        message = result.message or (
            f"Physical evidence contradicted gate {request.gate_id!r} before "
            f"segment {request.segment_name!r}."
        )
        non_retry = failure & ~retry
        if non_retry.any():
            self._eligible &= ~non_retry
            self._pending &= ~non_retry
            self._last_command_mask &= ~non_retry
            events.extend(
                (
                    self._event(
                        ExecutionEventKind.PHASE_EFFECT_GATE_FAILED,
                        non_retry,
                        message,
                    ),
                    self._event(
                        ExecutionEventKind.RECOVERY_REQUIRED,
                        non_retry,
                        "The failed segment-entry gate requires recovery outside "
                        "the current action retry policy.",
                    ),
                )
            )
        previous_generation = self._attempt_generation
        if retry.any():
            events.extend(
                self._attempt_action_retry(
                    retry,
                    ExecutionEventKind.PHASE_EFFECT_GATE_FAILED,
                    message,
                )
            )
        else:
            terminal_event = self._update_terminal_status()
            if terminal_event is not None:
                events.append(terminal_event)
        if (
            self._status is not ExecutionStatus.RUNNING
            or self._attempt_generation != previous_generation
        ):
            return events
        assert self._plan is not None
        remaining = request_mask & self._pending & self._plan.plan_success
        if remaining.any() and torch.equal(success & remaining, remaining):
            self._satisfied_phase_effect_gates.add(request.gate_id)
            events.append(
                self._event(
                    ExecutionEventKind.PHASE_EFFECT_GATE_SATISFIED,
                    remaining,
                    f"Physical-effect gate {request.gate_id!r} released segment "
                    f"{request.segment_name!r}.",
                )
            )
        return events

    def _held_object_guard_request(self) -> HeldObjectGuardRequest | None:
        """Build the current command-phase held-object guard request."""
        if (
            self._status is not ExecutionStatus.RUNNING
            or self._plan is None
            or self._pending_effect is not None
            or self._phase_effect_gate_request() is not None
            or self._effect_failures.any()
            or self._plan.commands.frame_count == 0
        ):
            return None
        env_mask = self._pending & self._plan.plan_success
        if not env_mask.any():
            return None
        next_waypoint_index = min(
            self._waypoint_index,
            self._plan.commands.frame_count - 1,
        )
        segment = self._plan.segment_at(next_waypoint_index)
        invocation = self._requests[self._invocation_index]
        (
            allowed_held_object_relations,
            allowed_coordinated_held_object_relations,
        ) = self._authorized_held_object_invalidation_relations(
            invocation=invocation,
        )
        return HeldObjectGuardRequest(
            verification_id=self._next_held_object_guard_verification_id,
            skill_id=invocation.skill_id,
            invocation_id=invocation.invocation_id,
            invocation_revision=invocation.revision,
            invocation_index=self._invocation_index,
            attempt_generation=self._attempt_generation,
            next_waypoint_index=next_waypoint_index,
            segment_name=segment.name,
            env_mask=env_mask,
            allowed_held_object_relations=allowed_held_object_relations,
            allowed_coordinated_held_object_relations=(
                allowed_coordinated_held_object_relations
            ),
            deadline=(
                self._action_started_at + self._plan.recovery_policy.action_timeout
            ),
        )

    def _apply_held_object_guard_result(
        self,
        result: HeldObjectGuardResult,
        request: HeldObjectGuardRequest,
    ) -> list[ExecutionEvent]:
        """Reconcile lost relations and enter row-local bounded recovery."""
        failure_mask = self._normalize_mask(
            result.failure_mask,
            "held_object_guard_result.failure_mask",
        )
        retry_mask = self._normalize_mask(
            result.retry_mask,
            "held_object_guard_result.retry_mask",
        )
        request_mask = request.env_mask.to(self._eligible.device)
        if (failure_mask & ~request_mask).any():
            raise ValueError(
                "Held-object guard failure_mask must be a subset of the active "
                "request env_mask."
            )
        if (retry_mask & ~request_mask).any():
            raise ValueError(
                "Held-object guard retry_mask must be a subset of the active "
                "request env_mask."
            )
        self._validate_held_object_invalidation_authorization(
            result.state_invalidation,
            object_id=result.object_id,
            allowed_held_object_relations=request.allowed_held_object_relations,
            allowed_coordinated_held_object_relations=(
                request.allowed_coordinated_held_object_relations
            ),
        )
        if not failure_mask.any():
            return []

        self._task_state = result.state_invalidation.apply(
            self._task_state,
            failure_mask,
        )
        self._context = PlanningContext(
            robot=self._context.robot,
            task=self._task_state,
            scene=self._context.scene,
            env_ids=self._context.env_ids,
        )
        message = result.message or (
            "Physical evidence contradicted the verified held-object relation."
        )
        non_retry_mask = failure_mask & ~retry_mask
        events: list[ExecutionEvent] = []
        if non_retry_mask.any():
            self._eligible &= ~non_retry_mask
            self._pending &= ~non_retry_mask
            self._effect_failures &= ~non_retry_mask
            self._last_command_mask &= ~non_retry_mask
            events.extend(
                (
                    self._event(
                        ExecutionEventKind.HELD_OBJECT_LOST,
                        non_retry_mask,
                        message,
                    ),
                    self._event(
                        ExecutionEventKind.RECOVERY_REQUIRED,
                        non_retry_mask,
                        "Held-object loss requires recovery outside the current "
                        "action retry policy.",
                    ),
                )
            )
        if retry_mask.any():
            events.extend(
                self._attempt_action_retry(
                    retry_mask,
                    ExecutionEventKind.HELD_OBJECT_LOST,
                    message,
                )
            )
        else:
            terminal_event = self._update_terminal_status()
            if terminal_event is not None:
                events.append(terminal_event)
        return events

    def _authorized_held_object_invalidation_relations(
        self,
        *,
        invocation: ResolvedActionRequest | None = None,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]]:
        """Return action-owned key/object identities eligible for removal."""
        assert self._plan is not None
        active_invocation = (
            self._requests[self._invocation_index] if invocation is None else invocation
        )
        binding_task_state_keys = {
            endpoint.task_state_key for endpoint in active_invocation.binding.endpoints
        }
        held_relations: set[tuple[str, str]] = set()
        for key, candidate in self._task_state.held_objects.items():
            object_id = candidate.semantics.entity_id
            if key in binding_task_state_keys and object_id is not None:
                held_relations.add((key, object_id))
        for key, candidate in self._plan.expected_effects.held_object_updates.items():
            if candidate is not None and candidate.semantics.entity_id is not None:
                held_relations.add((key, candidate.semantics.entity_id))
        for key, candidate in self._plan.effect_candidates.held_object_updates.items():
            if candidate is not None and candidate.semantics.entity_id is not None:
                held_relations.add((key, candidate.semantics.entity_id))

        related_keys = {key for key, _ in held_relations}
        coordinated_relations: set[tuple[str, str, str]] = set()
        for resources, candidate in self._task_state.coordinated_held_objects.items():
            object_id = candidate.semantics.entity_id
            if not set(resources).isdisjoint(related_keys) and object_id is not None:
                coordinated_relations.add((*resources, object_id))
        for (
            resources,
            candidate,
        ) in self._plan.expected_effects.coordinated_held_object_updates.items():
            if candidate is not None and candidate.semantics.entity_id is not None:
                coordinated_relations.add((*resources, candidate.semantics.entity_id))
        return tuple(sorted(held_relations)), tuple(sorted(coordinated_relations))

    def _validate_held_object_invalidation_authorization(
        self,
        state_invalidation: StateDelta,
        *,
        object_id: str,
        allowed_held_object_relations: tuple[tuple[str, str], ...],
        allowed_coordinated_held_object_relations: tuple[tuple[str, str, str], ...],
    ) -> None:
        """Reject removals outside the action-owned key/object identity set."""
        invalidated_held_relations = {
            (key, object_id) for key in state_invalidation.held_object_updates
        }
        if not invalidated_held_relations.issubset(allowed_held_object_relations):
            raise ValueError(
                "Held-object state invalidation contains a key/object identity "
                "outside the active action's authorized relation set."
            )
        invalidated_coordinated_relations = {
            (*resources, object_id)
            for resources in state_invalidation.coordinated_held_object_updates
        }
        if not invalidated_coordinated_relations.issubset(
            allowed_coordinated_held_object_relations
        ):
            raise ValueError(
                "Held-object state invalidation contains a coordinated key/object "
                "identity outside the active action's authorized relation set."
            )

    def _normalize_mask(self, value: torch.Tensor, name: str) -> torch.Tensor:
        """Validate and copy a per-environment boolean mask."""
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.dtype != torch.bool or value.shape != (self._context.batch_size,):
            raise ValueError(
                f"{name} must be bool with shape ({self._context.batch_size},)."
            )
        return value.to(self._context.robot.qpos.device).clone()

    def _effect_verification_request(
        self,
        env_mask: torch.Tensor,
    ) -> EffectVerificationRequest:
        """Describe the current action's pending semantic-effect boundary."""
        assert self._plan is not None
        request = self._requests[self._invocation_index]
        verification_id = self._next_effect_verification_id
        self._next_effect_verification_id += 1
        if self._effect_requested_at is None:
            self._effect_requested_at = self._context.robot.timestamp
        return EffectVerificationRequest(
            verification_id=verification_id,
            skill_id=request.skill_id,
            invocation_id=request.invocation_id,
            invocation_revision=request.revision,
            invocation_index=self._invocation_index,
            attempt_generation=self._attempt_generation,
            terminal_segment=(
                self._plan.segments[-1].name if self._plan.segments else None
            ),
            requested_at=self._effect_requested_at,
            deadline=(
                self._action_started_at + self._plan.recovery_policy.action_timeout
            ),
            env_mask=env_mask,
            expected_effects=self._plan.expected_effects,
            effect_verification=self._plan.effect_verification,
            failure_invalidation=self._effect_failure_invalidation(),
        )

    def _effect_failure_invalidation(self) -> StateDelta:
        """Build the core-owned fail-closed state removal for this effect."""
        assert self._plan is not None
        expected = self._plan.expected_effects
        held_keys = set(expected.held_object_updates)
        coordinated_keys = set(expected.coordinated_held_object_updates)
        coordinated_keys.update(
            resources
            for resources in self._task_state.coordinated_held_objects
            if not set(resources).isdisjoint(held_keys)
        )
        return StateDelta(
            held_object_updates={key: None for key in held_keys},
            coordinated_held_object_updates={
                resources: None for resources in coordinated_keys
            },
            articulation_joint_updates={
                key: None for key in expected.articulation_joint_updates
            },
        )

    def _apply_effect_failure_invalidation(
        self,
        state_invalidation: StateDelta,
        env_mask: torch.Tensor,
    ) -> None:
        """Apply a request-owned failure delta and refresh planning context."""
        if not env_mask.any() or state_invalidation.is_empty:
            return
        self._task_state = state_invalidation.apply(self._task_state, env_mask)
        self._context = PlanningContext(
            robot=self._context.robot,
            task=self._task_state,
            scene=self._context.scene,
            env_ids=self._context.env_ids,
        )

    def _failure_invalidation_presence_mask(
        self,
        state_invalidation: StateDelta,
    ) -> torch.Tensor:
        """Return rows whose verified state would actually be removed."""
        present = torch.zeros_like(self._eligible)
        for key in state_invalidation.held_object_updates:
            value = self._task_state.held_objects.get(key)
            if value is not None:
                assert value.env_mask is not None
                present |= value.env_mask.to(present.device)
        for key in state_invalidation.coordinated_held_object_updates:
            value = self._task_state.coordinated_held_objects.get(key)
            if value is not None:
                assert value.env_mask is not None
                present |= value.env_mask.to(present.device)
        for key in state_invalidation.articulation_joint_updates:
            value = self._task_state.articulation_joints.get(key)
            if value is not None:
                assert value.env_mask is not None
                present |= value.env_mask.to(present.device)
        return present

    def _event(
        self,
        kind: ExecutionEventKind,
        env_mask: torch.Tensor,
        message: str,
        *,
        segment_name: str | None = None,
        failure_code: str | None = None,
        retryable: bool | None = None,
    ) -> ExecutionEvent:
        """Create an event correlated with the current invocation."""
        skill_id = (
            self._requests[self._invocation_index].skill_id
            if self._invocation_index < len(self._requests)
            else None
        )
        invocation_id = (
            self._requests[self._invocation_index].invocation_id
            if self._invocation_index < len(self._requests)
            else None
        )
        invocation_revision = (
            self._requests[self._invocation_index].revision
            if self._invocation_index < len(self._requests)
            else 0
        )
        if segment_name is None and self._plan is not None and self._plan.segments:
            waypoint_index = min(
                self._waypoint_index,
                self._plan.commands.frame_count - 1,
            )
            segment_name = self._plan.segment_at(waypoint_index).name
        if (
            failure_code is None
            and kind is ExecutionEventKind.ACTION_PLANNING_FAILED
            and self._plan is not None
            and self._plan.diagnostics.failure is not None
        ):
            failure = self._plan.diagnostics.failure
            failure_code = failure.code
            retryable = failure.retryable
        return ExecutionEvent(
            kind=kind,
            timestamp=self._context.robot.timestamp,
            skill_id=skill_id,
            invocation_id=invocation_id,
            invocation_revision=invocation_revision,
            invocation_index=min(self._invocation_index, len(self._requests) - 1),
            env_mask=env_mask,
            message=message,
            segment_name=segment_name,
            failure_code=failure_code,
            retryable=retryable,
        )

    def _drain_events(self) -> list[ExecutionEvent]:
        """Return and clear events queued during planning."""
        events = self._queued_events
        self._queued_events = []
        return events

    def _update_terminal_status(self) -> ExecutionEvent | None:
        """Mark and report failure when no environment can continue."""
        if not self._eligible.any() and self._status is ExecutionStatus.RUNNING:
            self._status = ExecutionStatus.FAILED
            self._pending_effect = None
            self._pending_phase_effect_gate = None
            self._effect_failures.zero_()
            self._effect_requested_at = None
            return self._event(
                ExecutionEventKind.SESSION_FAILED,
                self._eligible,
                "No environment remains eligible for execution.",
            )
        return None

    def _tick_result(
        self,
        *,
        command: RuntimeCommandFrame | None,
        events: list[ExecutionEvent],
        hold_targets: tuple[RuntimeEndpointTarget, ...] = (),
    ) -> ExecutionTick:
        """Build an immutable tick result."""
        phase_gate = self._phase_effect_gate_request()
        if phase_gate is not None:
            events.extend(self._phase_effect_gate_required_events(phase_gate))
        return ExecutionTick(
            status=self._status,
            eligible_mask=self._eligible,
            command=command,
            hold_targets=hold_targets,
            events=tuple(events),
            task_state=self._task_state,
            pending_effect=self._pending_effect,
            pending_phase_effect_gate=phase_gate,
        )


__all__ = [
    "EffectExpectationResult",
    "EffectVerificationRequest",
    "EffectVerificationResult",
    "ExecutionEvent",
    "ExecutionEventKind",
    "ExecutionPlanAttempt",
    "ExecutionSession",
    "ExecutionStatus",
    "ExecutionTick",
    "HeldObjectGuardRequest",
    "HeldObjectGuardResult",
    "PhaseEffectGateRequest",
    "PhaseEffectGateResult",
]
