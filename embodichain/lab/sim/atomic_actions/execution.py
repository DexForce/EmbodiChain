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

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

from .invocation import ActionInvocation, ResolvedActionRequest
from .plans import ActionPlan, PlannedPhase
from .state import EntityState, PlanningContext, SceneSnapshot, TaskState

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
    TRACKING_ERROR = "tracking_error"
    DYNAMIC_GOAL_CHANGED = "dynamic_goal_changed"
    PHASE_TIMEOUT = "phase_timeout"
    PHASE_COMPLETED = "phase_completed"
    EFFECT_VERIFICATION_REQUIRED = "effect_verification_required"
    ACTION_RETRY = "action_retry"
    ACTION_COMPLETED = "action_completed"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    SESSION_COMPLETED = "session_completed"


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

    def __post_init__(self) -> None:
        if self.timestamp < 0.0:
            raise ValueError("ExecutionEvent.timestamp must be non-negative.")
        if self.invocation_index < 0:
            raise ValueError("invocation_index must be non-negative.")
        if self.invocation_revision < 0:
            raise ValueError("invocation_revision must be non-negative.")
        if self.env_mask.dtype != torch.bool or self.env_mask.dim() != 1:
            raise ValueError("ExecutionEvent.env_mask must be a 1D bool tensor.")
        object.__setattr__(self, "env_mask", self.env_mask.clone())


@dataclass(frozen=True, slots=True, eq=False)
class JointCommand:
    """Full-robot command produced by one session tick."""

    positions: torch.Tensor
    velocities: torch.Tensor | None
    active_mask: torch.Tensor
    env_ids: torch.Tensor
    hold_duration: torch.Tensor
    """Per-environment delay before the next observation/command cycle."""

    def __post_init__(self) -> None:
        if self.positions.dim() != 2:
            raise ValueError("JointCommand.positions must have shape (B, robot_dof).")
        if (
            self.velocities is not None
            and self.velocities.shape != self.positions.shape
        ):
            raise ValueError("JointCommand.velocities must match positions shape.")
        if self.active_mask.dtype != torch.bool or self.active_mask.shape != (
            self.positions.shape[0],
        ):
            raise ValueError("JointCommand.active_mask must be bool with shape (B,).")
        if self.env_ids.dtype != torch.long or self.env_ids.shape != (
            self.positions.shape[0],
        ):
            raise ValueError("JointCommand.env_ids must be int64 with shape (B,).")
        if not isinstance(self.hold_duration, torch.Tensor):
            raise TypeError("JointCommand.hold_duration must be a torch.Tensor.")
        if self.hold_duration.shape != (self.positions.shape[0],):
            raise ValueError("JointCommand.hold_duration must have shape (B,).")
        if (
            not torch.isfinite(self.hold_duration).all()
            or (self.hold_duration < 0.0).any()
        ):
            raise ValueError(
                "JointCommand.hold_duration must contain finite non-negative values."
            )
        if self.active_mask.device != self.positions.device:
            raise ValueError("JointCommand tensors must share a device.")
        if self.env_ids.device != self.positions.device:
            raise ValueError("JointCommand tensors must share a device.")
        if self.hold_duration.device != self.positions.device:
            raise ValueError("JointCommand tensors must share a device.")
        object.__setattr__(self, "positions", self.positions.clone())
        if self.velocities is not None:
            object.__setattr__(self, "velocities", self.velocities.clone())
        object.__setattr__(self, "active_mask", self.active_mask.clone())
        object.__setattr__(self, "env_ids", self.env_ids.clone())
        object.__setattr__(self, "hold_duration", self.hold_duration.clone())


@dataclass(frozen=True, slots=True, eq=False)
class ExecutionTick:
    """Result returned after one closed-loop execution update."""

    status: ExecutionStatus
    eligible_mask: torch.Tensor
    command: JointCommand | None
    events: tuple[ExecutionEvent, ...]
    task_state: TaskState

    def __post_init__(self) -> None:
        if self.eligible_mask.dtype != torch.bool or self.eligible_mask.dim() != 1:
            raise ValueError("eligible_mask must be a 1D bool tensor.")
        object.__setattr__(self, "eligible_mask", self.eligible_mask.clone())
        object.__setattr__(self, "events", tuple(self.events))


class ExecutionSession:
    """Execute grounded invocations incrementally with bounded local recovery.

    The session never steps a simulator itself. Each :meth:`tick` consumes the
    latest observation and scene snapshot and emits at most one full-robot
    command. Expected symbolic effects are committed only after the caller
    supplies ``effect_success`` for a non-empty :class:`StateDelta`.

    Environment eligibility and recovery budgets are tracked per row. Phase and
    waypoint cursors are batch-synchronized: a recoverable row replans the active
    cohort from the latest observation and restarts its shared phase cursor.
    """

    def __init__(
        self,
        engine: AtomicActionEngine,
        invocations: tuple[ActionInvocation, ...],
        context: PlanningContext,
    ) -> None:
        if not invocations:
            raise ValueError("ExecutionSession requires at least one invocation.")
        engine._validate_context(context)
        self._engine = engine
        self._requests: tuple[ResolvedActionRequest, ...] = tuple(
            engine.resolve(invocation) for invocation in invocations
        )
        self._task_state = context.task
        self._context = context
        self._invocation_index = 0
        self._phase_index = 0
        self._waypoint_index = 0
        self._plan: ActionPlan | None = None
        self._planned_scene = context.scene
        self._phase_started_at = context.robot.timestamp
        self._last_command: torch.Tensor | None = None
        self._last_command_mask = torch.zeros(
            context.batch_size, dtype=torch.bool, device=context.robot.qpos.device
        )
        self._eligible = torch.ones_like(self._last_command_mask)
        self._pending = self._eligible.clone()
        self._action_retries = torch.zeros(
            context.batch_size, dtype=torch.long, device=context.robot.qpos.device
        )
        self._replans = torch.zeros_like(self._action_retries)
        self._effect_wait_emitted = False
        self._status = ExecutionStatus.RUNNING
        self._queued_events: list[ExecutionEvent] = []
        self._plan_current(context, ExecutionEventKind.ACTION_PLANNED)

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

    def revise_current(self, invocation: ActionInvocation) -> None:
        """Replace and replan the current invocation with a newer revision.

        The replacement is resolved into a new immutable request snapshot from
        the latest observation. Retry and replan budgets restart for the new
        revision, while verified task state, the current batch barrier, and
        per-environment eligibility are preserved. Ordinary recovery replans
        continue to reuse this snapshot until another explicit revision.

        Args:
            invocation: Grounded replacement for the currently active skill.
                Its ``revision`` must be strictly greater than the active one,
                and its ``skill_id`` and ``invocation_id`` must identify the
                same logical call.

        Raises:
            TypeError: If ``invocation`` is not an ActionInvocation.
            RuntimeError: If the session is no longer running.
            ValueError: If the replacement identifies another invocation or
                does not advance the revision.
        """
        if not isinstance(invocation, ActionInvocation):
            raise TypeError("invocation must be an ActionInvocation.")
        if self._status is not ExecutionStatus.RUNNING:
            raise RuntimeError("Only a running execution session can be revised.")
        current = self._requests[self._invocation_index]
        if invocation.skill_id != current.skill_id:
            raise ValueError(
                f"Revision skill_id {invocation.skill_id!r} does not match "
                f"the active skill {current.skill_id!r}."
            )
        if invocation.invocation_id != current.invocation_id:
            raise ValueError(
                "Revision invocation_id must match the active invocation_id."
            )
        if invocation.revision <= current.revision:
            raise ValueError(
                f"Revision must advance beyond {current.revision}, got "
                f"{invocation.revision}."
            )

        replacement = self._engine.resolve(invocation)
        replacement_plan = self._engine.plan_request(replacement, self._context)
        requests = list(self._requests)
        requests[self._invocation_index] = replacement
        self._requests = tuple(requests)
        self._phase_index = 0
        self._waypoint_index = 0
        self._action_retries.zero_()
        self._replans.zero_()
        self._install_plan(
            replacement_plan,
            self._context,
            ExecutionEventKind.INVOCATION_REVISED,
        )

    @property
    def latest_context(self) -> PlanningContext:
        """Latest validated context with the session's verified task state."""
        return self._context

    def tick(
        self,
        context: PlanningContext,
        *,
        effect_success: torch.Tensor | None = None,
    ) -> ExecutionTick:
        """Advance execution by one observation/command cycle.

        Args:
            context: Latest measured robot and versioned scene state. Its task
                state is replaced by the session's verified task state.
            effect_success: Optional per-environment semantic-effect verification
                for an action waiting at its terminal waypoint.

        Returns:
            Status, optional command, events, and current verified task state.
        """
        self._engine._validate_context(context)
        if context.robot.timestamp < self._context.robot.timestamp:
            raise ValueError("Execution tick timestamps must be monotonic.")
        if context.scene.timestamp < self._context.scene.timestamp:
            raise ValueError("Scene snapshot timestamps must be monotonic.")
        if context.scene.version < self._context.scene.version:
            raise ValueError("Scene snapshot versions must be monotonic.")
        if not torch.equal(context.env_ids, self._context.env_ids):
            raise ValueError("Execution tick env_ids must remain stable and ordered.")
        self._context = PlanningContext(
            robot=context.robot,
            task=self._task_state,
            scene=context.scene,
            env_ids=context.env_ids,
        )
        events = self._drain_events()
        if self._status is not ExecutionStatus.RUNNING:
            return self._tick_result(command=None, events=events)

        assert self._plan is not None
        phase = self._current_phase()
        execution_mask = self._pending & self._plan.plan_success
        recovery_events = self._recover_if_needed(phase, execution_mask)
        events.extend(recovery_events)
        if self._status is not ExecutionStatus.RUNNING:
            return self._tick_result(command=None, events=events)
        if recovery_events and any(
            event.kind
            in {
                ExecutionEventKind.REPLANNED,
                ExecutionEventKind.RECOVERY_EXHAUSTED,
            }
            for event in recovery_events
        ):
            assert self._plan is not None
            phase = self._current_phase()
            execution_mask = self._pending & self._plan.plan_success

        trajectory = phase.trajectory
        if self._waypoint_index < trajectory.waypoint_count:
            command = self._command_at(phase, self._waypoint_index, execution_mask)
            self._waypoint_index += 1
            return self._tick_result(command=command, events=events)

        terminal_error = self._terminal_error(phase)
        not_reached = execution_mask & (
            terminal_error > phase.spec.recovery_policy.tracking_error_threshold
        )
        if not_reached.any():
            events.extend(
                self._attempt_replan(
                    not_reached,
                    ExecutionEventKind.TRACKING_ERROR,
                    "Terminal command has not been reached.",
                )
            )
            if self._status is not ExecutionStatus.RUNNING:
                return self._tick_result(command=None, events=events)
            phase = self._current_phase()
            execution_mask = self._pending & self._plan.plan_success
            command = self._command_at(phase, 0, execution_mask)
            self._waypoint_index = 1
            return self._tick_result(command=command, events=events)

        events.append(
            self._event(
                ExecutionEventKind.PHASE_COMPLETED,
                execution_mask,
                f"Phase {phase.spec.name!r} completed.",
            )
        )
        if self._phase_index + 1 < len(self._plan.phases):
            self._phase_index += 1
            self._waypoint_index = 0
            self._replans.zero_()
            self._phase_started_at = self._context.robot.timestamp
            self._last_command = None
            return self._tick_result(command=self._hold_command(), events=events)

        command, completion_events = self._finish_action(
            execution_mask,
            effect_success,
        )
        events.extend(completion_events)
        return self._tick_result(command=command, events=events)

    def _plan_current(
        self,
        context: PlanningContext,
        event_kind: ExecutionEventKind,
    ) -> None:
        """Plan the current invocation from the latest observation."""
        request = self._requests[self._invocation_index]
        plan = self._engine.plan_request(request, context)
        self._install_plan(plan, context, event_kind)

    def _install_plan(
        self,
        plan: ActionPlan,
        context: PlanningContext,
        event_kind: ExecutionEventKind,
    ) -> None:
        """Install an already validated plan as the current execution plan."""
        self._plan = plan
        self._phase_index = min(self._phase_index, len(plan.phases) - 1)
        self._waypoint_index = 0
        self._planned_scene = context.scene
        self._phase_started_at = context.robot.timestamp
        self._last_command = None
        self._last_command_mask.zero_()
        self._effect_wait_emitted = False
        planned_mask = self._pending & plan.plan_success
        self._queued_events.append(
            self._event(event_kind, planned_mask, "Planned from the latest context.")
        )

    def _current_phase(self) -> PlannedPhase:
        """Return the currently active planned phase."""
        assert self._plan is not None
        return self._plan.phases[self._phase_index]

    def _recover_if_needed(
        self,
        phase: PlannedPhase,
        execution_mask: torch.Tensor,
    ) -> list[ExecutionEvent]:
        """Detect tracking, scene, and timeout invalidation."""
        events: list[ExecutionEvent] = []
        if not execution_mask.any():
            return events
        if (
            self._context.robot.timestamp - self._phase_started_at
            > phase.spec.recovery_policy.phase_timeout
        ):
            return self._attempt_action_retry(
                execution_mask,
                ExecutionEventKind.PHASE_TIMEOUT,
                "Phase timeout exceeded.",
            )
        if self._last_command is not None:
            tracking_error = torch.amax(
                torch.abs(self._context.robot.qpos - self._last_command), dim=1
            )
            tracking_mask = (
                execution_mask
                & self._last_command_mask
                & (tracking_error > phase.spec.recovery_policy.tracking_error_threshold)
            )
            if tracking_mask.any():
                return self._attempt_replan(
                    tracking_mask,
                    ExecutionEventKind.TRACKING_ERROR,
                    "Observed joint tracking error exceeded the policy threshold.",
                )
        scene_mask = self._dynamic_scene_change_mask(phase)
        if (execution_mask & scene_mask).any():
            return self._attempt_replan(
                execution_mask & scene_mask,
                ExecutionEventKind.DYNAMIC_GOAL_CHANGED,
                "A referenced scene entity moved beyond the policy threshold.",
            )
        return events

    def _attempt_replan(
        self,
        trigger_mask: torch.Tensor,
        reason: ExecutionEventKind,
        message: str,
    ) -> list[ExecutionEvent]:
        """Apply per-row budgets and replan the synchronized active cohort."""
        phase = self._current_phase()
        events = [self._event(reason, trigger_mask, message)]
        allowed = (
            trigger_mask
            & phase.spec.replannable
            & (self._replans < phase.spec.recovery_policy.max_replans)
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
        self._update_terminal_status()
        return events

    def _attempt_action_retry(
        self,
        trigger_mask: torch.Tensor,
        reason: ExecutionEventKind,
        message: str,
    ) -> list[ExecutionEvent]:
        """Retry the current action or permanently fail exhausted rows."""
        policy = self._current_phase().spec.recovery_policy
        events = [self._event(reason, trigger_mask, message)]
        allowed = trigger_mask & (self._action_retries < policy.max_phase_retries)
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
            self._replans.zero_()
            events.append(
                self._event(
                    ExecutionEventKind.ACTION_RETRY,
                    allowed,
                    "Retrying the action from the latest observation.",
                )
            )
            self._phase_index = 0
            self._plan_current(self._context, ExecutionEventKind.REPLANNED)
            events.extend(self._drain_events())
        self._update_terminal_status()
        return events

    def _finish_action(
        self,
        execution_mask: torch.Tensor,
        effect_success: torch.Tensor | None,
    ) -> tuple[JointCommand | None, list[ExecutionEvent]]:
        """Verify effects, update symbolic state, and advance the action barrier."""
        assert self._plan is not None
        events: list[ExecutionEvent] = []
        if self._plan.expected_effects.is_empty:
            verified = execution_mask
        elif effect_success is None:
            if not self._effect_wait_emitted:
                events.append(
                    self._event(
                        ExecutionEventKind.EFFECT_VERIFICATION_REQUIRED,
                        execution_mask,
                        "Expected symbolic effects require external verification.",
                    )
                )
                self._effect_wait_emitted = True
            return self._hold_command(), events
        else:
            verified_input = self._normalize_mask(effect_success, "effect_success")
            verified = execution_mask & verified_input

        if verified.any():
            self._task_state = self._plan.expected_effects.apply(
                self._task_state, verified
            )
            self._context = PlanningContext(
                robot=self._context.robot,
                task=self._task_state,
                scene=self._context.scene,
                env_ids=self._context.env_ids,
            )
            self._pending &= ~verified
        failed_effect = execution_mask & ~verified
        planning_failed = self._pending & ~self._plan.plan_success
        retry_mask = failed_effect | planning_failed
        if retry_mask.any():
            events.extend(
                self._attempt_action_retry(
                    retry_mask,
                    ExecutionEventKind.ACTION_RETRY,
                    "Planning or expected-effect verification failed.",
                )
            )
            if self._status is not ExecutionStatus.RUNNING:
                return None, events
            return self._hold_command(), events

        if self._pending.any():
            return self._hold_command(), events
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
            events.append(
                self._event(
                    ExecutionEventKind.SESSION_COMPLETED,
                    self._eligible,
                    "Invocation sequence completed.",
                )
            )
            return None, events

        self._pending = self._eligible.clone()
        self._action_retries.zero_()
        self._replans.zero_()
        self._phase_index = 0
        self._plan_current(self._context, ExecutionEventKind.ACTION_PLANNED)
        events.extend(self._drain_events())
        return self._hold_command(), events

    def _command_at(
        self,
        phase: PlannedPhase,
        waypoint_index: int,
        active_mask: torch.Tensor,
    ) -> JointCommand:
        """Build one command and retain it for tracking-error monitoring."""
        positions = phase.trajectory.positions[:, waypoint_index]
        hold = self._context.robot.qpos
        positions = torch.where(active_mask[:, None], positions, hold)
        velocities = None
        if phase.trajectory.velocities is not None:
            values = phase.trajectory.velocities[:, waypoint_index]
            velocities = torch.where(
                active_mask[:, None], values, torch.zeros_like(values)
            )
        self._last_command = positions.clone()
        self._last_command_mask = active_mask.clone()
        # ``dt[:, i]`` leads to waypoint ``i``. After dispatching waypoint
        # ``i``, wait for ``dt[:, i + 1]`` before the next dispatch. Reuse the
        # final arrival interval as its terminal settling window.
        next_waypoint_index = min(
            waypoint_index + 1,
            phase.trajectory.waypoint_count - 1,
        )
        hold_duration = phase.trajectory.dt[:, next_waypoint_index]
        return JointCommand(
            positions=positions,
            velocities=velocities,
            active_mask=active_mask,
            env_ids=phase.trajectory.env_ids,
            hold_duration=hold_duration,
        )

    def _hold_command(self) -> JointCommand:
        """Build a passive hold command from the latest observation."""
        return JointCommand(
            positions=self._context.robot.qpos,
            velocities=torch.zeros_like(self._context.robot.qpos),
            active_mask=torch.zeros_like(self._eligible),
            env_ids=self._context.env_ids,
            hold_duration=torch.zeros(
                self._context.batch_size,
                dtype=torch.float32,
                device=self._context.robot.qpos.device,
            ),
        )

    def _terminal_error(self, phase: PlannedPhase) -> torch.Tensor:
        """Return per-row max joint error to the phase terminal command."""
        if phase.trajectory.waypoint_count == 0:
            return torch.full_like(self._eligible, float("inf"), dtype=torch.float32)
        return torch.amax(
            torch.abs(self._context.robot.qpos - phase.trajectory.positions[:, -1]),
            dim=1,
        )

    def _dynamic_scene_change_mask(self, phase: PlannedPhase) -> torch.Tensor:
        """Detect material motion of entities referenced by the phase goal."""
        dependencies = phase.spec.scene_dependencies
        changed = torch.zeros_like(self._eligible)
        if (
            not dependencies
            or self._context.scene.version == self._planned_scene.version
        ):
            return changed
        policy = phase.spec.recovery_policy
        for entity_id in dependencies:
            previous = self._planned_scene.entities.get(entity_id)
            current = self._context.scene.entities.get(entity_id)
            if previous is None or current is None:
                changed |= self._eligible
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
            changed |= (translation > policy.goal_translation_threshold) | (
                rotation > policy.goal_rotation_threshold
            )
        return changed

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

    def _normalize_mask(self, value: torch.Tensor, name: str) -> torch.Tensor:
        """Validate and copy a per-environment boolean mask."""
        if value.dtype != torch.bool or value.shape != (self._context.batch_size,):
            raise ValueError(
                f"{name} must be bool with shape ({self._context.batch_size},)."
            )
        return value.to(self._context.robot.qpos.device).clone()

    def _event(
        self,
        kind: ExecutionEventKind,
        env_mask: torch.Tensor,
        message: str,
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
        return ExecutionEvent(
            kind=kind,
            timestamp=self._context.robot.timestamp,
            skill_id=skill_id,
            invocation_id=invocation_id,
            invocation_revision=invocation_revision,
            invocation_index=min(self._invocation_index, len(self._requests) - 1),
            env_mask=env_mask,
            message=message,
        )

    def _drain_events(self) -> list[ExecutionEvent]:
        """Return and clear events queued during planning."""
        events = self._queued_events
        self._queued_events = []
        return events

    def _update_terminal_status(self) -> None:
        """Mark the session failed when no environment can continue."""
        if not self._eligible.any():
            self._status = ExecutionStatus.FAILED

    def _tick_result(
        self,
        *,
        command: JointCommand | None,
        events: list[ExecutionEvent],
    ) -> ExecutionTick:
        """Build an immutable tick result."""
        return ExecutionTick(
            status=self._status,
            eligible_mask=self._eligible,
            command=command,
            events=tuple(events),
            task_state=self._task_state,
        )


__all__ = [
    "ExecutionEvent",
    "ExecutionEventKind",
    "ExecutionSession",
    "ExecutionStatus",
    "ExecutionTick",
    "JointCommand",
]
