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

"""Controller-independent scheduling for closed-loop atomic-action execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
import math
import time
from typing import Protocol, runtime_checkable

import torch

from embodichain.utils import configclass

from .bindings import RuntimeEndpointTarget
from .execution import (
    EffectVerificationResult,
    ExecutionSession,
    ExecutionStatus,
    ExecutionTick,
)
from .invocation import ActionInvocation, ResolvedActionRequest
from .runtime_commands import RuntimeCommandFrame
from .state import PlanningContext, TaskState


class CommandAckStatus(str, Enum):
    """Outcome reported by a command transport or controller."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class CommandAcknowledgement:
    """Synchronous acknowledgement returned by a :class:`CommandSink`."""

    status: CommandAckStatus
    """Transport/controller acknowledgement status."""

    message: str = ""
    """Human-readable diagnostic intended for logs, not policy branching."""

    def __post_init__(self) -> None:
        if not isinstance(self.status, CommandAckStatus):
            raise TypeError("status must be a CommandAckStatus.")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string.")

    @property
    def accepted(self) -> bool:
        """Whether the controller accepted the requested operation."""
        return self.status is CommandAckStatus.ACCEPTED

    @classmethod
    def accepted_ack(cls, message: str = "") -> CommandAcknowledgement:
        """Build an accepted acknowledgement.

        Args:
            message: Optional controller diagnostic.

        Returns:
            Accepted acknowledgement.
        """
        return cls(CommandAckStatus.ACCEPTED, message)


class CommandOperation(str, Enum):
    """Command-sink operation recorded by an execution runner."""

    SEND = "send"
    HOLD = "hold"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class CommandDispatch:
    """Auditable record of one controller operation and acknowledgement."""

    operation: CommandOperation
    acknowledgement: CommandAcknowledgement

    def __post_init__(self) -> None:
        if not isinstance(self.operation, CommandOperation):
            raise TypeError("operation must be a CommandOperation.")
        if not isinstance(self.acknowledgement, CommandAcknowledgement):
            raise TypeError("acknowledgement must be a CommandAcknowledgement.")


@runtime_checkable
class ObservationProvider(Protocol):
    """Source of fresh planning contexts for feedback-driven execution."""

    def observe(self, task_state: TaskState) -> PlanningContext:
        """Capture the latest robot and scene state.

        Args:
            task_state: Runner-owned, externally verified symbolic task state.

        Returns:
            Fresh context with stable, ordered environment IDs.
        """


@runtime_checkable
class CommandSink(Protocol):
    """Controller boundary used by :class:`ExecutionRunner`."""

    def send(
        self,
        command: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Submit one synchronized endpoint-command frame.

        Args:
            command: Transport-neutral command frame with an active-row mask.
                The sink must actively neutralize inactive rows for every
                addressed target; omission is not a safe state for persistent
                controllers.
            timeout: Maximum acknowledgement latency in seconds.

        Returns:
            Transport or controller acknowledgement.
        """

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Apply transport-specific safe state to the supplied targets.

        Args:
            targets: Runtime targets that may retain controller state.
            context: Latest observation used by position-hold transports.
            timeout: Maximum acknowledgement latency in seconds.

        Returns:
            Transport or controller acknowledgement.
        """

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Cancel any controller-side command that has not completed.

        Args:
            targets: Runtime targets whose queued work must be cancelled.
            timeout: Maximum acknowledgement latency in seconds.

        Returns:
            Transport or controller acknowledgement.
        """


@runtime_checkable
class ExecutionClock(Protocol):
    """Clock abstraction used for deterministic and simulation scheduling."""

    def now(self) -> float:
        """Return a monotonic timestamp in seconds.

        Returns:
            Monotonic timestamp in seconds.
        """

    def sleep(self, duration: float) -> None:
        """Wait or advance the execution backend by ``duration`` seconds.

        Args:
            duration: Non-negative duration in seconds.
        """


class MonotonicExecutionClock:
    """Wall-clock implementation backed by :mod:`time`."""

    def now(self) -> float:
        """Return the current monotonic wall-clock time.

        Returns:
            Monotonic wall-clock timestamp in seconds.
        """
        return time.monotonic()

    def sleep(self, duration: float) -> None:
        """Sleep for a non-negative wall-clock duration.

        Args:
            duration: Requested duration in seconds.
        """
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("duration must be finite and non-negative.")
        time.sleep(duration)


@configclass
class ExecutionRunnerCfg:
    """Transport and scheduling policy for an :class:`ExecutionRunner`."""

    command_timeout: float = 1.0
    """Maximum time allowed for a command acknowledgement."""

    safe_stop_timeout: float = 1.0
    """Maximum time allowed for each cancel or hold acknowledgement."""

    minimum_cycle_time: float = 1.0e-3
    """Minimum delay between feedback cycles, including passive hold cycles."""

    hold_on_completion: bool = True
    """Whether to issue a final hold after the session completes."""

    def __post_init__(self) -> None:
        for name in ("command_timeout", "safe_stop_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero.")
        if not math.isfinite(self.minimum_cycle_time) or self.minimum_cycle_time < 0.0:
            raise ValueError("minimum_cycle_time must be finite and non-negative.")
        if not isinstance(self.hold_on_completion, bool):
            raise TypeError("hold_on_completion must be a bool.")


class RunnerStatus(str, Enum):
    """Lifecycle status owned by an :class:`ExecutionRunner`."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True, eq=False)
class RunnerStep:
    """Result of one non-blocking execution-runner update."""

    status: RunnerStatus
    timestamp: float
    wait_duration: float
    context: PlanningContext | None
    tick: ExecutionTick | None
    dispatches: tuple[CommandDispatch, ...]
    command_count: int
    message: str | None = None
    """Terminal or failure diagnostic, when available."""

    def __post_init__(self) -> None:
        if not isinstance(self.status, RunnerStatus):
            raise TypeError("status must be a RunnerStatus.")
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("timestamp must be finite and non-negative.")
        if not math.isfinite(self.wait_duration) or self.wait_duration < 0.0:
            raise ValueError("wait_duration must be finite and non-negative.")
        if self.command_count < 0:
            raise ValueError("command_count must be non-negative.")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be a string or None.")
        object.__setattr__(self, "dispatches", tuple(self.dispatches))

    @property
    def is_waiting(self) -> bool:
        """Whether no session tick was due during this update."""
        return (
            self.status is RunnerStatus.RUNNING
            and self.tick is None
            and self.wait_duration > 0.0
        )


EffectVerifier = Callable[
    [PlanningContext, ExecutionTick],
    EffectVerificationResult | None,
]
"""Callback that verifies a pending semantic effect for each environment."""

RunnerStepCallback = Callable[[RunnerStep], None]
"""Optional observer called after every blocking runner-loop iteration."""


class ExecutionRunner:
    """Connect an execution session to observation, controller, and time ports.

    :meth:`step` is non-blocking. It observes and advances the session only when
    the next command is due according to
    :attr:`RuntimeCommandFrame.hold_duration`.
    :meth:`run_until_blocked` supplies the blocking loop for tutorials and simple
    applications. Controller rejection, timeout, observation failure, and
    session exceptions all trigger a best-effort cancel-then-hold sequence.
    Runner methods are designed for serialized event-loop use and are not
    thread-safe.

    Args:
        session: Stateful atomic-action execution session.
        observation_provider: Source of fresh robot and scene observations.
        command_sink: Controller or simulation command boundary.
        clock: Optional scheduler clock. Defaults to monotonic wall time.
        cfg: Optional acknowledgement, scheduling, and completion policy.
    """

    def __init__(
        self,
        session: ExecutionSession,
        observation_provider: ObservationProvider,
        command_sink: CommandSink,
        *,
        clock: ExecutionClock | None = None,
        cfg: ExecutionRunnerCfg | None = None,
    ) -> None:
        if not isinstance(session, ExecutionSession):
            raise TypeError("session must be an ExecutionSession.")
        if not isinstance(observation_provider, ObservationProvider):
            raise TypeError("observation_provider must implement ObservationProvider.")
        if not isinstance(command_sink, CommandSink):
            raise TypeError("command_sink must implement CommandSink.")
        if clock is not None and not isinstance(clock, ExecutionClock):
            raise TypeError("clock must implement ExecutionClock.")
        if cfg is not None and not isinstance(cfg, ExecutionRunnerCfg):
            raise TypeError("cfg must be an ExecutionRunnerCfg.")
        self._session = session
        self._observation_provider = observation_provider
        self._command_sink = command_sink
        self._clock = clock or MonotonicExecutionClock()
        self.cfg = cfg or ExecutionRunnerCfg()
        self._status = RunnerStatus.RUNNING
        self._next_step_at = self._clock_now()
        self._last_context: PlanningContext | None = session.latest_context
        self._command_count = 0
        self._message: str | None = None
        self._effect_context: PlanningContext | None = None
        self._effect_tick: ExecutionTick | None = None
        self._armed_targets: dict[tuple[str, str], RuntimeEndpointTarget] = {}
        self._pending_revision: ResolvedActionRequest | None = None

    @property
    def session(self) -> ExecutionSession:
        """Execution session advanced by this runner.

        Call :meth:`revise_current` or :meth:`deactivate_rows` on the runner,
        rather than mutating the session directly, while this runner owns
        scheduling.
        """
        return self._session

    @property
    def status(self) -> RunnerStatus:
        """Current runner lifecycle status."""
        return self._status

    @property
    def command_count(self) -> int:
        """Number of active commands accepted by the sink."""
        return self._command_count

    @property
    def effect_verification_pending(self) -> bool:
        """Whether execution is waiting for an external semantic-effect result."""
        return (
            self._effect_tick is not None
            and self._effect_tick.pending_effect is not None
        )

    def revise_current(self, invocation: ActionInvocation) -> None:
        """Stage a newer revision for the next scheduled observation boundary.

        Staging preserves the active frame deadline. When that deadline is due,
        :meth:`step` observes fresh state, atomically plans and installs the
        replacement, and dispatches its first command. The submitted invocation
        is resolved into an owned snapshot immediately, so later caller
        mutation cannot alter the staged revision.

        Args:
            invocation: Strictly newer revision of the active logical call.

        Raises:
            TypeError: If ``invocation`` is not an ActionInvocation.
            RuntimeError: If this runner or its session is no longer running,
                or if a physical effect is awaiting verification.
            ValueError: If session-level revision invariants are violated.
        """
        if not isinstance(invocation, ActionInvocation):
            raise TypeError("invocation must be an ActionInvocation.")
        if self._status is not RunnerStatus.RUNNING:
            raise RuntimeError("Only a running execution runner can be revised.")
        prepared = self._session._prepare_revision(invocation)
        if (
            self._pending_revision is not None
            and prepared.revision <= self._pending_revision.revision
        ):
            raise ValueError(
                "A staged revision must advance beyond the pending revision "
                f"{self._pending_revision.revision}, got {prepared.revision}."
            )
        self._pending_revision = prepared

    def deactivate_rows(
        self,
        env_mask: torch.Tensor,
        *,
        reason: str,
    ) -> torch.Tensor:
        """Permanently deactivate environment rows owned by this runner.

        The runner refreshes its cached effect boundary so a verifier cannot
        submit a result correlated with a request that deactivation replaced.
        In-flight controller work is neutralized for those rows by the next
        due command frame according to the :class:`CommandSink` contract.

        Args:
            env_mask: Rows requested for deactivation.
            reason: Human-readable event message.

        Returns:
            Owned mask of rows that changed from eligible to inactive.

        Raises:
            RuntimeError: If the runner is already terminal.
            TypeError: If ``env_mask`` is not a tensor.
            ValueError: If the mask or reason is invalid.
        """
        if self._status is not RunnerStatus.RUNNING:
            raise RuntimeError("Only a running execution runner can deactivate rows.")
        changed = self._session.deactivate_rows(env_mask, reason=reason)
        if self._session.status is not ExecutionStatus.RUNNING:
            self._pending_revision = None
        pending_effect = self._session.pending_effect
        if pending_effect is None:
            self._clear_effect_boundary()
        elif self._effect_tick is not None:
            self._effect_tick = replace(
                self._effect_tick,
                status=self._session.status,
                eligible_mask=self._session.eligible_mask,
                task_state=self._session.task_state,
                pending_effect=pending_effect,
            )
        return changed

    def step(
        self,
        *,
        effect_result: EffectVerificationResult | None = None,
    ) -> RunnerStep:
        """Perform one due observation/session/controller update without sleeping.

        Args:
            effect_result: Optional correlated effect result. If this call
                occurs before the next cycle is due, it is not consumed and
                must be supplied again on a later call.

        Returns:
            Runner status, optional session tick, controller acknowledgements,
            and time remaining before another update is due.
        """
        now = self._clock_now()
        if self._status is not RunnerStatus.RUNNING:
            return self._result(timestamp=now)
        wait_duration = self._remaining_wait(now)
        if wait_duration > 0.0:
            return self._result(
                timestamp=now,
                wait_duration=wait_duration,
            )

        try:
            context = self._observation_provider.observe(self._session.task_state)
            if not isinstance(context, PlanningContext):
                raise TypeError(
                    "ObservationProvider.observe() must return PlanningContext."
                )
        except Exception as exc:
            return self._fail(
                f"Observation provider failed: {type(exc).__name__}: {exc}",
                context=self._last_context,
            )
        self._last_context = context

        try:
            if self._pending_revision is not None:
                self._session._install_prepared_revision(
                    self._pending_revision,
                    context,
                )
                self._pending_revision = None
            tick = self._session.tick(context, effect_result=effect_result)
            context = self._session.latest_context
            self._last_context = context
        except Exception as exc:
            return self._fail(
                f"Execution session failed: {type(exc).__name__}: {exc}",
                context=context,
            )
        self._update_effect_boundary(context, tick)

        dispatches: list[CommandDispatch] = []
        if tick.command is not None:
            self._remember_targets(tick.command.targets)
            operation = (
                CommandOperation.SEND
                if bool(tick.command.active_mask.any().item())
                else CommandOperation.HOLD
            )
            dispatch = self._dispatch(
                operation,
                command=(tick.command if operation is CommandOperation.SEND else None),
                targets=tick.command.targets,
                context=context,
            )
            dispatches.append(dispatch)
            if not dispatch.acknowledgement.accepted:
                failure = dispatch.acknowledgement
                message = (
                    "Controller did not accept the requested command: "
                    f"{failure.status.value}."
                )
                if failure.message:
                    message += f" {failure.message}"
                return self._fail(
                    message,
                    context=context,
                    tick=tick,
                    dispatches=dispatches,
                )
            if operation is CommandOperation.SEND:
                self._command_count += 1
            interval = self._command_interval(tick.command)
            self._next_step_at = self._clock_now() + interval
        elif tick.hold_targets:
            self._remember_targets(tick.hold_targets)
            hold_dispatch = self._dispatch(
                CommandOperation.HOLD,
                targets=tick.hold_targets,
                context=context,
            )
            dispatches.append(hold_dispatch)
            if not hold_dispatch.acknowledgement.accepted:
                failure = hold_dispatch.acknowledgement
                message = (
                    "Controller did not accept the requested hold: "
                    f"{failure.status.value}."
                )
                if failure.message:
                    message += f" {failure.message}"
                return self._fail(
                    message,
                    context=context,
                    tick=tick,
                    dispatches=dispatches,
                )
            self._next_step_at = self._clock_now() + self.cfg.minimum_cycle_time
        elif tick.pending_effect is not None:
            self._next_step_at = self._clock_now() + self.cfg.minimum_cycle_time
        else:
            self._next_step_at = self._clock_now()

        if tick.status is ExecutionStatus.COMPLETED:
            if self.cfg.hold_on_completion:
                hold_dispatch = self._dispatch(
                    CommandOperation.HOLD,
                    targets=self._armed_target_snapshots(),
                    context=context,
                )
                dispatches.append(hold_dispatch)
                if not hold_dispatch.acknowledgement.accepted:
                    failure = hold_dispatch.acknowledgement
                    message = (
                        "Final safety hold was not accepted: "
                        f"{failure.status.value}."
                    )
                    if failure.message:
                        message += f" {failure.message}"
                    return self._fail(
                        message,
                        context=context,
                        tick=tick,
                        dispatches=dispatches,
                    )
            self._status = RunnerStatus.COMPLETED
            self._next_step_at = self._clock_now()
        elif tick.status is ExecutionStatus.FAILED:
            return self._fail(
                "Execution session failed; inspect its terminal events for the cause.",
                context=context,
                tick=tick,
                dispatches=dispatches,
            )

        return self._result(
            timestamp=self._clock_now(),
            context=context,
            tick=tick,
            dispatches=dispatches,
            wait_duration=self._remaining_wait(self._clock_now()),
        )

    def cancel(self, reason: str = "Execution cancelled by caller.") -> RunnerStep:
        """Cancel controller work and hold the latest observed position.

        Args:
            reason: Human-readable cancellation reason.

        Returns:
            Terminal runner step. The status is ``cancelled`` only when both
            cancel and hold are acknowledged; otherwise it is ``failed``.
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string.")
        now = self._clock_now()
        if self._status is not RunnerStatus.RUNNING:
            return self._result(timestamp=now)
        context = self._observe_for_stop()
        dispatches = self._safe_stop(context)
        if all(item.acknowledgement.accepted for item in dispatches):
            self._status = RunnerStatus.CANCELLED
            self._message = reason
        else:
            self._status = RunnerStatus.FAILED
            self._message = f"{reason} Safe stop acknowledgement failed."
        self._clear_effect_boundary()
        self._pending_revision = None
        self._next_step_at = self._clock_now()
        return self._result(
            timestamp=self._clock_now(),
            context=context,
            dispatches=dispatches,
        )

    def run_until_blocked(
        self,
        *,
        effect_verifier: EffectVerifier | None = None,
        on_step: RunnerStepCallback | None = None,
        max_steps: int = 100_000,
    ) -> RunnerStep:
        """Run with clock-driven waiting until terminal or effect verification blocks.

        Args:
            effect_verifier: Optional callback used after an
                ``effect_verification_required`` event. Without one, the method
                returns the running step so the caller can verify externally.
            on_step: Optional callback for tracing or tutorial visualization.
            max_steps: Hard bound on loop iterations.

        Returns:
            Terminal step, or a running step blocked on external verification.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero.")
        effect_result: EffectVerificationResult | None = None
        now = self._clock_now()
        last_result = self._result(
            timestamp=now,
            wait_duration=self._remaining_wait(now),
            context=self._effect_context,
            tick=self._effect_tick,
        )
        if self.effect_verification_pending and effect_verifier is None:
            return last_result
        for _ in range(max_steps):
            result = self.step(effect_result=effect_result)
            if result.tick is not None:
                effect_result = None
            if on_step is not None:
                try:
                    on_step(result)
                except Exception as exc:
                    return self._fail(
                        f"Runner step callback failed: {type(exc).__name__}: {exc}",
                        context=result.context or self._last_context,
                        tick=result.tick,
                        dispatches=list(result.dispatches),
                    )
            last_result = result
            if result.status is not RunnerStatus.RUNNING:
                return result
            verification_required = (
                result.tick is not None and result.tick.pending_effect is not None
            )
            if verification_required:
                if effect_verifier is None or result.context is None:
                    return result
                try:
                    effect_result = effect_verifier(result.context, result.tick)
                except Exception as exc:
                    return self._fail(
                        f"Effect verifier failed: {type(exc).__name__}: {exc}",
                        context=result.context,
                        tick=result.tick,
                        dispatches=list(result.dispatches),
                    )
                if effect_result is None:
                    return result
            if result.wait_duration > 0.0:
                try:
                    self._clock.sleep(result.wait_duration)
                except Exception as exc:
                    return self._fail(
                        f"Execution clock failed: {type(exc).__name__}: {exc}",
                        context=result.context or self._last_context,
                        tick=result.tick,
                        dispatches=list(result.dispatches),
                    )
        return self._fail(
            f"Execution runner exceeded max_steps={max_steps}.",
            context=last_result.context or self._last_context,
            tick=last_result.tick,
            dispatches=list(last_result.dispatches),
        )

    def _update_effect_boundary(
        self,
        context: PlanningContext,
        tick: ExecutionTick,
    ) -> None:
        """Remember or clear the external effect-verification boundary."""
        if tick.pending_effect is not None:
            self._effect_context = context
            self._effect_tick = tick
        else:
            self._clear_effect_boundary()

    def _clear_effect_boundary(self) -> None:
        """Clear a remembered external effect-verification boundary."""
        self._effect_context = None
        self._effect_tick = None

    def _clock_now(self) -> float:
        """Read and validate the injected monotonic clock."""
        value = float(self._clock.now())
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("ExecutionClock.now() must be finite and non-negative.")
        return value

    def _command_interval(self, command: RuntimeCommandFrame) -> float:
        """Resolve a synchronized batch interval from per-environment durations."""
        durations = (
            command.hold_duration[command.active_mask]
            if command.active_mask.any()
            else command.hold_duration
        )
        requested = float(durations.max().item()) if durations.numel() else 0.0
        return max(requested, self.cfg.minimum_cycle_time)

    def _remaining_wait(self, now: float) -> float:
        """Return scheduled wait while absorbing float32 timing roundoff."""
        remaining = self._next_step_at - now
        tolerance = max(1.0e-9, self.cfg.minimum_cycle_time * 1.0e-6)
        return remaining if remaining > tolerance else 0.0

    def _dispatch(
        self,
        operation: CommandOperation,
        command: RuntimeCommandFrame | None = None,
        *,
        targets: tuple[RuntimeEndpointTarget, ...] = (),
        context: PlanningContext | None = None,
    ) -> CommandDispatch:
        """Call one sink operation and convert exceptions to rejection acks."""
        try:
            if operation is CommandOperation.SEND:
                if command is None:
                    raise ValueError("SEND requires a RuntimeCommandFrame.")
                acknowledgement = self._command_sink.send(
                    command,
                    timeout=self.cfg.command_timeout,
                )
            elif operation is CommandOperation.HOLD:
                if context is None:
                    raise ValueError("HOLD requires a PlanningContext.")
                acknowledgement = self._command_sink.hold(
                    targets,
                    context,
                    timeout=self.cfg.safe_stop_timeout,
                )
            else:
                acknowledgement = self._command_sink.cancel(
                    targets, timeout=self.cfg.safe_stop_timeout
                )
            if not isinstance(acknowledgement, CommandAcknowledgement):
                raise TypeError(
                    "CommandSink methods must return CommandAcknowledgement."
                )
        except Exception as exc:
            acknowledgement = CommandAcknowledgement(
                CommandAckStatus.REJECTED,
                f"{type(exc).__name__}: {exc}",
            )
        return CommandDispatch(operation, acknowledgement)

    def _remember_targets(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
    ) -> None:
        """Remember every controller destination armed during this run."""
        for target in targets:
            key = (target.transport_id, target.target_id)
            self._armed_targets[key] = target.snapshot()

    def _armed_target_snapshots(self) -> tuple[RuntimeEndpointTarget, ...]:
        """Return owned armed targets in first-use order."""
        return tuple(target.snapshot() for target in self._armed_targets.values())

    def _observe_for_stop(self) -> PlanningContext | None:
        """Best-effort observation used to build a cancellation hold command."""
        try:
            context = self._observation_provider.observe(self._session.task_state)
            if not isinstance(context, PlanningContext):
                return self._last_context
            self._last_context = context
            return context
        except Exception:
            return self._last_context

    def _safe_stop(
        self,
        context: PlanningContext | None,
    ) -> list[CommandDispatch]:
        """Attempt controller cancellation followed by an observed-position hold."""
        targets = self._armed_target_snapshots()
        dispatches = [self._dispatch(CommandOperation.CANCEL, targets=targets)]
        if context is not None:
            dispatches.append(
                self._dispatch(
                    CommandOperation.HOLD,
                    targets=targets,
                    context=context,
                )
            )
        return dispatches

    def _fail(
        self,
        message: str,
        *,
        context: PlanningContext | None,
        tick: ExecutionTick | None = None,
        dispatches: list[CommandDispatch] | None = None,
    ) -> RunnerStep:
        """Enter failed state after a best-effort cancel-then-hold sequence."""
        records = list(dispatches or ())
        records.extend(self._safe_stop(context))
        self._status = RunnerStatus.FAILED
        self._message = message
        self._clear_effect_boundary()
        self._pending_revision = None
        self._next_step_at = self._clock_now()
        return self._result(
            timestamp=self._clock_now(),
            context=context,
            tick=tick,
            dispatches=records,
        )

    def _result(
        self,
        *,
        timestamp: float,
        wait_duration: float = 0.0,
        context: PlanningContext | None = None,
        tick: ExecutionTick | None = None,
        dispatches: list[CommandDispatch] | tuple[CommandDispatch, ...] = (),
    ) -> RunnerStep:
        """Build an immutable runner result."""
        return RunnerStep(
            status=self._status,
            timestamp=timestamp,
            wait_duration=wait_duration,
            context=context,
            tick=tick,
            dispatches=tuple(dispatches),
            command_count=self._command_count,
            message=self._message,
        )


__all__ = [
    "CommandAckStatus",
    "CommandAcknowledgement",
    "CommandDispatch",
    "CommandOperation",
    "CommandSink",
    "EffectVerifier",
    "ExecutionClock",
    "ExecutionRunner",
    "ExecutionRunnerCfg",
    "MonotonicExecutionClock",
    "ObservationProvider",
    "RunnerStatus",
    "RunnerStep",
    "RunnerStepCallback",
]
