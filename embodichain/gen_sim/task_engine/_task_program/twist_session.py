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
"""E8-only feedback using public session APIs and the unchanged shared runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Protocol, TYPE_CHECKING, runtime_checkable

import torch

from embodichain.lab.sim.atomic_actions import (
    ExecutionSession,
    ExecutionStatus,
    ExecutionTick,
)
from embodichain.lab.sim.atomic_actions.invocation import ActionInvocation
from embodichain.lab.sim.atomic_actions.runtime_commands import RuntimeCommandFrame
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.verification import (
    EffectVerificationResult,
    HeldObjectGuardResult,
    PhaseEffectGateResult,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class DueObservationRequest:
    """Owned E8 phase identity and the first plan's unchanged logical deadline.

    The public phase view is scheduling metadata, not a held-object assertion.
    """

    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    invocation_index: int
    attempt_generation: int
    next_waypoint_index: int
    segment_name: str
    env_mask: torch.Tensor
    original_planned_at: float
    original_action_timeout: float

    def __post_init__(self) -> None:
        for name in ("skill_id", "segment_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        for name in (
            "invocation_revision",
            "invocation_index",
            "attempt_generation",
            "next_waypoint_index",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if (
            not isinstance(self.env_mask, torch.Tensor)
            or self.env_mask.dtype != torch.bool
            or self.env_mask.ndim != 1
        ):
            raise ValueError("env_mask must be a one-dimensional boolean tensor.")
        if (
            not math.isfinite(self.original_planned_at)
            or self.original_planned_at < 0
            or math.isnan(self.original_action_timeout)
            or self.original_action_timeout <= 0
        ):
            raise ValueError(
                "Original time must be finite and timeout positive or unbounded."
            )
        object.__setattr__(self, "env_mask", self.env_mask.clone())


@dataclass(frozen=True, slots=True)
class RevisionControllerStop:
    """Refuse further E8 commands through the original runner's safe stop."""

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Revision controller stop reason must be non-empty.")


@runtime_checkable
class DueObservationRevisionController(Protocol):
    """Instance-owned E8 policy; it neither dispatches nor steps a simulator."""

    controller_id: str
    revision: str
    supported_skill_ids: tuple[str, ...]

    def propose(
        self,
        context: PlanningContext,
        due: DueObservationRequest,
        *,
        previous_command: RuntimeCommandFrame | None,
    ) -> ActionInvocation | RevisionControllerStop | None:
        """Return one revision, refusal, or retain the active plan."""


class TwistFeedbackSession(ExecutionSession):
    """Specialize only the pre-tick E8 policy, delegating execution to Lab.

    Production ownership is the unchanged :class:`ExecutionRunner`: it observes
    once when due, calls this public ``tick``, dispatches, and stops on rejection.
    A stored frame is merely emitted, never an acknowledgement. At the next
    runner tick, the E8 policy additionally checks native command-target readback
    before using that frame as a continuation seed. Direct manual ticking does
    not provide an accepted-transport guarantee.

    Non-E8/default sessions must be created by the original engine route. This
    specialization admits one effectless, empty-hand Twist invocation only;
    semantic effects, gates, held guards, and parallel lanes are not supported.
    """

    def __init__(
        self,
        engine: AtomicActionEngine,
        invocations: tuple[ActionInvocation, ...],
        context: PlanningContext,
        *,
        revision_controller: DueObservationRevisionController,
        eligible_mask: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(revision_controller, DueObservationRevisionController):
            raise TypeError("E8 feedback requires its typed revision controller.")
        self._feedback_controller = revision_controller
        self._feedback_identity = (
            id(revision_controller),
            revision_controller.controller_id,
            revision_controller.revision,
            revision_controller.supported_skill_ids,
        )
        for name in ("controller_id", "revision"):
            value = getattr(revision_controller, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(
                    f"E8 controller {name} must be canonical and non-empty."
                )
        if revision_controller.supported_skill_ids != ("twist",):
            raise ValueError("E8 feedback requires the exact one-skill Twist scope.")
        if (
            not isinstance(context, PlanningContext)
            or context.batch_size != 1
            or len(invocations) != 1
            or not isinstance(invocations[0], ActionInvocation)
            or invocations[0].skill_id != "twist"
            or not invocations[0].invocation_id
        ):
            raise ValueError(
                "E8 feedback requires one identified Twist in one environment."
            )
        if invocations[0].phase_effect_gates:
            raise ValueError("E8 feedback does not support phase-effect gates.")
        self._assert_empty_hand(context)
        self._last_emitted_frame: RuntimeCommandFrame | None = None
        self._validated_generation = -1
        super().__init__(engine, invocations, context, eligible_mask=eligible_mask)
        attempts = self.plan_attempts
        if attempts:
            first = attempts[0]
            self._original_timing = (
                first.planned_at,
                first.plan.recovery_policy.action_timeout,
            )
            self._assert_effectless_plan()
            self._validated_generation = first.attempt_generation
        else:
            self._original_timing = (
                context.robot.timestamp,
                invocations[0].recovery_policy.action_timeout,
            )

    @staticmethod
    def _assert_empty_hand(context: PlanningContext) -> None:
        if context.task.held_objects or context.task.coordinated_held_objects:
            raise ValueError("E8 feedback requires empty verified held-object state.")

    def _assert_identity(self) -> None:
        controller = self._feedback_controller
        identity = (
            id(controller),
            controller.controller_id,
            controller.revision,
            controller.supported_skill_ids,
        )
        if identity != self._feedback_identity:
            raise RuntimeError("E8 feedback controller identity or scope changed.")

    def _assert_effectless_plan(self) -> None:
        plan = self.active_plan
        if not plan.expected_effects.is_empty or plan.requires_effect_verification:
            raise ValueError("E8 feedback requires an effectless active plan.")

    def tick(
        self,
        context: PlanningContext,
        *,
        effect_result: EffectVerificationResult | None = None,
        phase_effect_gate_result: PhaseEffectGateResult | None = None,
        held_object_guard_result: HeldObjectGuardResult | None = None,
    ) -> ExecutionTick:
        """Propose before the shared tick without changing command scheduling."""
        if any(
            value is not None
            for value in (
                effect_result,
                phase_effect_gate_result,
                held_object_guard_result,
            )
        ):
            raise RuntimeError(
                "E8 feedback does not consume effect, gate, or held results."
            )
        if self.status is ExecutionStatus.RUNNING:
            self._assert_identity()
            if not isinstance(context, PlanningContext) or context.batch_size != 1:
                raise ValueError("E8 feedback observation must retain one environment.")
            self._assert_empty_hand(context)
            if (
                self.pending_effect is not None
                or self.phase_effect_gate_request is not None
            ):
                raise RuntimeError(
                    "E8 feedback cannot revise an effect or gate boundary."
                )
            phase = self.held_object_guard_request
            if phase is not None:
                if phase.skill_id != "twist" or phase.invocation_index != 0:
                    raise RuntimeError(
                        "E8 feedback phase changed its invocation scope."
                    )
                if phase.attempt_generation != self._validated_generation:
                    self._assert_effectless_plan()
                    self._validated_generation = phase.attempt_generation
                planned_at, timeout = self._original_timing
                now = context.robot.timestamp
                if not math.isfinite(now) or now < planned_at:
                    raise ValueError("E8 feedback observation clock is invalid.")
                remaining = planned_at + timeout - now
                if remaining < 0:
                    raise RuntimeError(
                        "E8 feedback original invocation deadline expired."
                    )
                due = DueObservationRequest(
                    skill_id=phase.skill_id,
                    invocation_id=phase.invocation_id,
                    invocation_revision=phase.invocation_revision,
                    invocation_index=phase.invocation_index,
                    attempt_generation=phase.attempt_generation,
                    next_waypoint_index=phase.next_waypoint_index,
                    segment_name=phase.segment_name,
                    env_mask=phase.env_mask,
                    original_planned_at=planned_at,
                    original_action_timeout=timeout,
                )
                decision = self._feedback_controller.propose(
                    replace(
                        context,
                        robot=replace(context.robot),
                        task=replace(context.task),
                        scene=replace(context.scene),
                    ),
                    due,
                    previous_command=(
                        None
                        if self._last_emitted_frame is None
                        else self._last_emitted_frame.snapshot()
                    ),
                )
                self._assert_identity()
                if type(decision) is RevisionControllerStop:
                    raise RuntimeError(f"E8 feedback stopped: {decision.reason}")
                if decision is not None:
                    if not isinstance(decision, ActionInvocation):
                        raise TypeError(
                            "E8 feedback must return an invocation, stop, or None."
                        )
                    if decision.phase_effect_gates:
                        raise ValueError("E8 feedback cannot add phase-effect gates.")
                    if remaining <= 0:
                        raise RuntimeError(
                            "E8 feedback has no original deadline budget."
                        )
                    if math.isnan(decision.recovery_policy.action_timeout):
                        raise ValueError("E8 feedback proposed a NaN timeout.")
                    decision = replace(
                        decision,
                        recovery_policy=replace(
                            decision.recovery_policy,
                            action_timeout=min(
                                decision.recovery_policy.action_timeout, remaining
                            ),
                        ),
                    )
                    self.revise_current(decision, context=context)
                    self._assert_effectless_plan()
        tick = super().tick(context)
        if tick.status is not ExecutionStatus.RUNNING:
            self._last_emitted_frame = None
        elif tick.command is not None:
            self._last_emitted_frame = (
                tick.command.snapshot()
                if bool(tick.command.active_mask.any())
                else None
            )
        elif tick.hold_targets:
            self._last_emitted_frame = None
        return tick
