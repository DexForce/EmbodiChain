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

"""Canonical execution service and convenience facade for semantic skills."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import torch

from embodichain.utils.math import pose_inv

from embodichain.lab.sim.atomic_actions.bindings import EndpointBinding
from embodichain.lab.sim.atomic_actions.engine import AtomicActionEngine
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.execution import (
    ExecutionEvent,
    ExecutionEventKind,
)
from embodichain.lab.sim.atomic_actions.verification import (
    EffectExpectationResult,
    EffectVerificationRequest,
    EffectVerificationResult,
    HeldObjectGuardRequest,
    HeldObjectGuardResult,
    PhaseEffectGateRequest,
    PhaseEffectGateResult,
)
from embodichain.lab.sim.atomic_actions.plans import ActionPlan
from embodichain.lab.sim.atomic_actions.requirements import (
    FORWARD_KINEMATICS_CAPABILITY,
)
from embodichain.lab.sim.atomic_actions.runner import (
    CommandSink,
    ExecutionClock,
    ExecutionRunner,
    ExecutionRunnerCfg,
    MonotonicExecutionClock,
    ObservationProvider,
    RunnerStatus,
    RunnerStep,
)
from embodichain.lab.sim.atomic_actions.state import (
    HeldObjectState,
    PlanningContext,
    TaskState,
)
from embodichain.lab.semantic_skills.calls import (
    HandOver,
    Pick,
    Place,
    SemanticCallSpec,
)
from embodichain.lab.semantic_skills.effects import (
    EffectEvidenceBatch,
    EffectExpectationDecision,
    EffectMonitor,
    EffectMonitorDecision,
    EffectMonitorRef,
    HeldObjectRelation,
    HeldObjectStateExpectation,
    SemanticEffectSpec,
)
from embodichain.lab.semantic_skills.integration import SemanticValidationError
from embodichain.lab.semantic_skills.profiles import (
    EffectAssurance,
    WorkflowRecoveryPolicy,
)
from embodichain.lab.semantic_skills.scene import SceneObjectRef, SceneRegistry

from ._semantic_results import (
    ResolvedCorePolicyTrace,
    SemanticExecutionResult,
    SemanticExecutionStatus,
    SkillCallTrace,
    SkillEffectTrace,
    SkillFailure,
    SkillPlanAttemptTrace,
    SkillWorkflowRecoveryRole,
    SkillWorkflowRecoveryTrace,
    _snapshot_event,
    _snapshot_task_state,
)
from ._semantic_compiler import (
    GroundedHeldObjectGuard,
    GroundedPhaseEffectGate,
    GroundedSemanticCall,
    HeldObjectGuardBaseline,
    SemanticCallCompiler,
)


@dataclass(frozen=True, slots=True, eq=False)
class _WorkflowRecoveryWorkItem:
    """One cohort scheduled at a shared semantic-call recovery barrier."""

    role: SkillWorkflowRecoveryRole
    call: SemanticCallSpec
    env_mask: torch.Tensor
    attempt_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.role, SkillWorkflowRecoveryRole):
            raise TypeError("role must be a SkillWorkflowRecoveryRole.")
        if not isinstance(self.call, SemanticCallSpec):
            raise TypeError("call must be a SemanticCallSpec.")
        if (
            not isinstance(self.env_mask, torch.Tensor)
            or self.env_mask.dtype != torch.bool
            or self.env_mask.dim() != 1
            or not self.env_mask.any()
        ):
            raise ValueError(
                "env_mask must be a non-empty one-dimensional bool tensor."
            )
        if type(self.attempt_index) is not int or self.attempt_index <= 0:
            raise ValueError("attempt_index must be a positive integer.")
        object.__setattr__(self, "env_mask", self.env_mask.clone())


@dataclass(slots=True)
class _WorkflowRecoveryBarrier:
    """Mutable per-call barrier while failed rows recover and rejoin."""

    trigger_call_index: int
    trigger_call: SemanticCallSpec
    policy: WorkflowRecoveryPolicy
    source_resource_id: str
    source_task_state_key: str
    entered_mask: torch.Tensor
    success_mask: torch.Tensor
    final_failure_mask: torch.Tensor
    attempt_counts: torch.Tensor
    work_items: deque[_WorkflowRecoveryWorkItem]
    failure_messages: list[str]


@dataclass(frozen=True, slots=True)
class _FinishedCallAttempt:
    """Internal terminal projection of one execution session."""

    trace: SkillCallTrace
    completed_mask: torch.Tensor
    failed_mask: torch.Tensor
    status: RunnerStatus
    message: str | None


@dataclass(frozen=True, slots=True)
class _WorkflowRecoveryTrigger:
    """Resolved workflow policy and source identity for one original call."""

    policy: WorkflowRecoveryPolicy
    source_resource_id: str
    source_task_state_key: str


@runtime_checkable
class EffectEvidenceCollectorPort(Protocol):
    """Minimal collector surface consumed by :class:`SemanticCallExecutor`."""

    def collect(
        self,
        spec: SemanticEffectSpec,
        *,
        timestamp: float,
        observation_revision: int,
        env_ids: torch.Tensor | None = None,
    ) -> Mapping[str, EffectEvidenceBatch]:
        """Acquire synchronized raw evidence for one grounded effect."""


class _PrimedObservationProvider:
    """Return a JIT-grounding observation once before delegating fresh reads."""

    def __init__(
        self,
        context: PlanningContext,
        delegate: ObservationProvider,
    ) -> None:
        self._context: PlanningContext | None = context
        self._delegate = delegate

    def observe(self, task_state: TaskState) -> PlanningContext:
        """Reuse the grounding snapshot for the session's first due cycle."""
        context = self._context
        if context is None:
            return self._delegate.observe(task_state)
        self._context = None
        return PlanningContext(
            robot=context.robot,
            task=task_state,
            scene=context.scene,
            env_ids=context.env_ids,
            control_dt=context.control_dt,
        )


class SemanticCallExecutor:
    """JIT-ground and execute semantic calls through one runner per call.

    Static workflow analysis occurs once in :meth:`start`. Each call then gets
    a fresh observation, one grounded invocation, one execution session, and
    one :class:`ExecutionRunner`. Verified task state and row eligibility cross
    call barriers; execution sessions never do.
    """

    def __init__(
        self,
        compiler: SemanticCallCompiler,
        observation_provider: ObservationProvider,
        command_sink: CommandSink,
        evidence_collector: EffectEvidenceCollectorPort,
        *,
        task_state: TaskState | None = None,
        clock: ExecutionClock | None = None,
        runner_cfg: ExecutionRunnerCfg | None = None,
    ) -> None:
        if not isinstance(compiler, SemanticCallCompiler):
            raise TypeError("compiler must be a SemanticCallCompiler.")
        if not isinstance(observation_provider, ObservationProvider):
            raise TypeError("observation_provider must implement ObservationProvider.")
        if not isinstance(command_sink, CommandSink):
            raise TypeError("command_sink must implement CommandSink.")
        if not isinstance(evidence_collector, EffectEvidenceCollectorPort):
            raise TypeError(
                "evidence_collector must implement EffectEvidenceCollectorPort."
            )
        if clock is not None and not isinstance(clock, ExecutionClock):
            raise TypeError("clock must implement ExecutionClock.")
        if runner_cfg is not None and not isinstance(runner_cfg, ExecutionRunnerCfg):
            raise TypeError("runner_cfg must be an ExecutionRunnerCfg or None.")
        integration = compiler.integration
        engine = integration.engine
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError(
                "compiler.integration.engine must be an AtomicActionEngine."
            )
        initial_task = (
            engine.initial_context().task if task_state is None else task_state
        )
        if not isinstance(initial_task, TaskState):
            raise TypeError("task_state must be a TaskState or None.")
        if initial_task.device != engine.device:
            raise ValueError("task_state and compiler engine must share a device.")

        self._compiler = compiler
        self._engine = engine
        self._observation_provider = observation_provider
        self._command_sink = command_sink
        self._evidence_collector = evidence_collector
        self._clock = clock or MonotonicExecutionClock()
        self._runner_cfg_override = runner_cfg
        self._step_observer: Callable[[RunnerStep], None] | None = None
        self._task_state = _snapshot_task_state(initial_task)
        self._env_ids = torch.arange(
            self._task_state.batch_size,
            dtype=torch.long,
            device=self._task_state.device,
        )
        self._has_observed_env_ids = False
        self._status = SemanticExecutionStatus.IDLE
        self._workflow: object | None = None
        self._workflow_id: str | None = None
        self._calls: tuple[SemanticCallSpec, ...] = ()
        self._execution_prefix_length = 0
        self._current_call_index: int | None = None
        self._runner: ExecutionRunner | None = None
        self._grounded: GroundedSemanticCall | None = None
        self._active_call: SemanticCallSpec | None = None
        self._active_recovery_item: _WorkflowRecoveryWorkItem | None = None
        self._recovery_barrier: _WorkflowRecoveryBarrier | None = None
        self._call_entered_mask = torch.zeros(
            self._task_state.batch_size,
            dtype=torch.bool,
            device=self._task_state.device,
        )
        self._eligible = torch.ones_like(self._call_entered_mask)
        self._success = torch.zeros_like(self._eligible)
        self._failed = torch.zeros_like(self._eligible)
        self._cancelled = torch.zeros_like(self._eligible)
        self._events: list[ExecutionEvent] = []
        self._call_traces: list[SkillCallTrace] = []
        self._effect_traces: list[SkillEffectTrace] = []
        self._workflow_recovery_traces: list[SkillWorkflowRecoveryTrace] = []
        self._failures: list[SkillFailure] = []
        self._call_event_offset = 0
        self._call_effect_offset = 0
        self._observation_revision = 0
        self._next_guard_verification_id = 0
        self._next_gate_verification_id = 0
        self._next_workflow_recovery_id = 0
        self._wait_duration = 0.0
        self._message: str | None = None

    @property
    def compiler(self) -> SemanticCallCompiler:
        """Return the installed semantic compiler."""
        return self._compiler

    @property
    def engine(self) -> AtomicActionEngine:
        """Return the atomic-action engine owned by this runtime."""
        return self._engine

    @property
    def observation_provider(self) -> ObservationProvider:
        """Return the observation port used for just-in-time grounding."""
        return self._observation_provider

    @property
    def clock(self) -> ExecutionClock:
        """Return the shared execution clock used by this runtime.

        Parallel coordinators use the same clock for every derived lane so a
        branch cannot advance independently of the environment step grid.
        """
        return self._clock

    @property
    def scene_registry(self) -> SceneRegistry:
        """Return the authoritative semantic scene registry."""
        return self._compiler.integration.scene_registry

    def validate(
        self,
        calls: Iterable[SemanticCallSpec],
        *,
        workflow_id: str = "semantic_workflow",
    ) -> object:
        """Analyze a workflow without executing it."""
        return self.compiler.analyze(calls, workflow_id=workflow_id)

    @property
    def task_state(self) -> TaskState:
        """Return an owned snapshot of persistent verified task state."""
        return _snapshot_task_state(self._task_state)

    def fork(
        self,
        command_sink: CommandSink,
        *,
        task_state: TaskState | None = None,
    ) -> SemanticCallExecutor:
        """Create an independent execution lane from the same runtime ports.

        The derived runtime shares the immutable compiler integration,
        observation/evidence providers, clock, and runner policy, but owns its
        workflow, runner, masks, and verified task state.  Its command sink is
        supplied explicitly so a parallel coordinator can buffer commands
        until all lanes have reached the same environment tick.

        Args:
            command_sink: Lane-local command sink.
            task_state: Optional verified barrier state.  The current owned
                task state is used when omitted.

        Returns:
            A new idle semantic runtime for one independent lane.
        """
        if not isinstance(command_sink, CommandSink):
            raise TypeError("command_sink must implement CommandSink.")
        initial_state = self.task_state if task_state is None else task_state
        if not isinstance(initial_state, TaskState):
            raise TypeError("task_state must be a TaskState or None.")
        return SemanticCallExecutor(
            self._compiler,
            self._observation_provider,
            command_sink,
            self._evidence_collector,
            task_state=initial_state,
            clock=self._clock,
            runner_cfg=self._runner_cfg_override,
        )

    @property
    def status(self) -> SemanticExecutionStatus:
        """Return the current workflow status."""
        return self._status

    @property
    def result(self) -> SemanticExecutionResult:
        """Return an immutable snapshot of the current workflow."""
        return SemanticExecutionResult(
            status=self._status,
            workflow_id=self._workflow_id,
            current_call_index=self._current_call_index,
            env_ids=self._env_ids,
            success_mask=self._success,
            failure_mask=self._failed,
            cancelled_mask=self._cancelled,
            eligible_mask=self._eligible,
            task_state=self._task_state,
            events=tuple(self._events),
            calls=tuple(self._call_traces),
            effects=tuple(self._effect_traces),
            workflow_recoveries=tuple(self._workflow_recovery_traces),
            failures=tuple(self._failures),
            wait_duration=self._wait_duration,
            message=self._message,
        )

    def start(
        self,
        *calls: SemanticCallSpec | Iterable[SemanticCallSpec],
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
    ) -> SemanticExecutionResult:
        """Analyze once and prepare the first call without blocking on motion.

        Args:
            *calls: Complete ordered semantic analysis window.  Calls after the
                execution prefix participate in static look-ahead but are not
                grounded or executed by this run.
            workflow_id: Stable workflow identifier used in diagnostics.
            eligible_mask: Optional row-local execution eligibility.
            execution_prefix_length: Number of leading calls to execute.  When
                omitted, the complete analysis window is executed.

        Returns:
            Immutable initial runtime result.
        """
        if self._status is SemanticExecutionStatus.RUNNING:
            raise RuntimeError("A semantic workflow is already running.")
        normalized = self._normalize_calls(calls)
        if type(workflow_id) is not str or not workflow_id:
            raise ValueError("workflow_id must be a non-empty string.")
        prefix_length = self._normalize_execution_prefix_length(
            execution_prefix_length,
            call_count=len(normalized),
        )
        workflow = self._compiler.analyze(normalized, workflow_id=workflow_id)
        self._reset_workflow(
            normalized,
            workflow,
            workflow_id=workflow_id,
            eligible_mask=eligible_mask,
            execution_prefix_length=prefix_length,
        )
        try:
            self._prepare_call(0)
        except Exception as exc:  # noqa: BLE001 - return one uniform result
            self._fail_preparation(0, exc)
        return self.result

    def step(self) -> SemanticExecutionResult:
        """Advance the current call by at most one due runner cycle."""
        if self._status is not SemanticExecutionStatus.RUNNING:
            return self.result
        runner = self._require_runner()
        grounded = self._require_grounded()
        monitor = grounded.effect_monitor
        if monitor is not None:
            verifier = self._effect_verifier
        elif grounded.analyzed.effect_assurance is EffectAssurance.PROJECTED:
            verifier = self._project_unverified_effect
        else:  # pragma: no cover - compiler rejects this before execution
            raise RuntimeError(
                "A verified semantic call reached execution without an effect "
                "monitor."
            )
        guards = grounded.effect_guards
        guard_verifier = self._held_object_guard_verifier if guards else None
        gates = grounded.effect_gates
        gate_verifier = self._phase_effect_gate_verifier if gates else None
        runner_step = runner.step(
            effect_verifier=verifier,
            phase_effect_gate_verifier=gate_verifier,
            held_object_guard_verifier=guard_verifier,
        )
        if self._step_observer is not None:
            self._step_observer(runner_step)
        self._consume_runner_step(runner_step)
        if (
            runner_step.status is RunnerStatus.RUNNING
            and runner_step.tick is not None
            and runner_step.tick.pending_phase_effect_gate is not None
            and not gates
        ):
            self._abort(
                "The atomic invocation requested a phase-effect gate, but the "
                "grounded semantic call did not install its monitor."
            )
            return self.result
        if runner_step.status is RunnerStatus.RUNNING:
            return self.result
        recovery_item = self._active_recovery_item
        trigger = (
            self._workflow_recovery_trigger()
            if recovery_item is None and self._active_call_requires_workflow_recovery()
            else None
        )
        finished = self._finish_active_call(runner_step)
        if recovery_item is None:
            self._call_traces.append(finished.trace)
            self._handle_original_call_finished(finished, trigger=trigger)
        else:
            self._handle_recovery_call_finished(recovery_item, finished)
        return self.result

    def run(
        self,
        *calls: SemanticCallSpec | Iterable[SemanticCallSpec],
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
        max_steps: int = 100_000,
        on_step: Callable[[RunnerStep], None] | None = None,
    ) -> SemanticExecutionResult:
        """Synchronously execute an analyzed semantic-call prefix."""
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        if on_step is not None and not callable(on_step):
            raise TypeError("on_step must be callable or None.")
        previous_observer = self._step_observer
        self._step_observer = on_step
        try:
            result = self.start(
                *calls,
                workflow_id=workflow_id,
                eligible_mask=eligible_mask,
                execution_prefix_length=execution_prefix_length,
            )
            for _ in range(max_steps):
                if result.terminal:
                    return result
                if result.wait_duration > 0.0:
                    self._clock.sleep(result.wait_duration)
                result = self.step()
            self._abort(f"Semantic runtime exceeded max_steps={max_steps}.")
            return self.result
        finally:
            self._step_observer = previous_observer

    def cancel(
        self, reason: str = "Semantic workflow cancelled by caller."
    ) -> SemanticExecutionResult:
        """Cancel the active runner and inherit its cancel-then-hold behavior."""
        if type(reason) is not str or not reason:
            raise ValueError("reason must be a non-empty string.")
        if self._status is not SemanticExecutionStatus.RUNNING:
            return self.result
        pending = self._eligible.clone()
        recovery_item = self._active_recovery_item
        runner_step = self._require_runner().cancel(reason)
        self._consume_runner_step(runner_step)
        self._message = runner_step.message or reason
        finished = self._finish_active_call(runner_step)
        if recovery_item is None:
            self._call_traces.append(finished.trace)
        else:
            self._append_workflow_recovery_trace(
                recovery_item,
                call=finished.trace,
                completed_mask=finished.completed_mask,
                failed_mask=finished.failed_mask,
                message=finished.message,
            )
        self._status = (
            SemanticExecutionStatus.CANCELLED
            if runner_step.status is RunnerStatus.CANCELLED
            else SemanticExecutionStatus.FAILED
        )
        if self._status is SemanticExecutionStatus.FAILED:
            self._failed |= pending
            self._cancelled &= ~pending
        else:
            self._cancelled |= pending
        self._eligible &= ~pending
        self._recovery_barrier = None
        self._current_call_index = None
        self._wait_duration = 0.0
        return self.result

    def deactivate_rows(
        self,
        env_mask: torch.Tensor,
        *,
        reason: str,
    ) -> SemanticExecutionResult:
        """Cancel selected rows while the remaining shared call keeps running.

        This is the row-local cancellation boundary used by a parallel
        fail-fast coordinator. The active runner remains the sole owner of
        controller neutralization and effect-request correlation.

        Args:
            env_mask: Rows to remove permanently from this workflow.
            reason: Human-readable cancellation reason.

        Returns:
            Updated immutable workflow result.
        """
        if self._status is not SemanticExecutionStatus.RUNNING:
            return self.result
        if not isinstance(env_mask, torch.Tensor):
            raise TypeError("env_mask must be a torch.Tensor.")
        if (
            env_mask.dtype != torch.bool
            or env_mask.shape != self._eligible.shape
            or env_mask.device != self._eligible.device
        ):
            raise ValueError(
                "env_mask must be bool and match the runtime batch/device."
            )
        if type(reason) is not str or not reason:
            raise ValueError("reason must be a non-empty string.")
        changed = env_mask & self._eligible
        self._require_runner().deactivate_rows(
            changed,
            reason=reason,
        )
        self._cancelled |= changed
        self._eligible &= ~changed
        barrier = self._recovery_barrier
        if barrier is not None and changed.any():
            barrier.success_mask &= ~changed
            retained_items: deque[_WorkflowRecoveryWorkItem] = deque()
            for item in barrier.work_items:
                retained = item.env_mask & ~changed
                if retained.any():
                    retained_items.append(
                        _WorkflowRecoveryWorkItem(
                            role=item.role,
                            call=item.call,
                            env_mask=retained,
                            attempt_index=item.attempt_index,
                        )
                    )
            barrier.work_items = retained_items
        if not self._eligible.any():
            recovery_item = self._active_recovery_item
            runner_step = self._require_runner().cancel(reason)
            self._consume_runner_step(runner_step)
            finished = self._finish_active_call(runner_step)
            if recovery_item is None:
                self._call_traces.append(finished.trace)
            else:
                self._append_workflow_recovery_trace(
                    recovery_item,
                    call=finished.trace,
                    completed_mask=finished.completed_mask,
                    failed_mask=finished.failed_mask,
                    message=finished.message,
                )
            self._status = (
                SemanticExecutionStatus.CANCELLED
                if runner_step.status is RunnerStatus.CANCELLED
                else SemanticExecutionStatus.FAILED
            )
            if self._status is SemanticExecutionStatus.FAILED:
                failed = self._call_entered_mask & ~self._cancelled
                self._failed |= failed
            self._recovery_barrier = None
            self._current_call_index = None
            self._wait_duration = 0.0
        return self.result

    def adopt_verified_task_state(
        self, task_state: TaskState
    ) -> SemanticExecutionResult:
        """Install a verified state snapshot between independent workflows.

        Parallel coordinators use this explicit barrier operation after
        deterministically merging branch-local effects. Running workflows
        cannot replace their runner-owned state.
        """
        if self._status is SemanticExecutionStatus.RUNNING:
            raise RuntimeError("Cannot replace task state while a workflow is running.")
        if not isinstance(task_state, TaskState):
            raise TypeError("task_state must be a TaskState.")
        if (
            task_state.batch_size != self._task_state.batch_size
            or task_state.device != self._task_state.device
        ):
            raise ValueError("task_state must match the runtime batch and device.")
        self._task_state = _snapshot_task_state(task_state)
        return self.result

    @property
    def _has_next_call(self) -> bool:
        assert self._current_call_index is not None
        return self._current_call_index + 1 < self._execution_prefix_length

    @staticmethod
    def _normalize_execution_prefix_length(
        value: int | None,
        *,
        call_count: int,
    ) -> int:
        """Normalize a non-empty execution prefix inside one analysis window."""
        if value is None:
            return call_count
        if type(value) is not int:
            raise TypeError("execution_prefix_length must be an integer or None.")
        if not 1 <= value <= call_count:
            raise ValueError(
                "execution_prefix_length must be in " f"[1, {call_count}], got {value}."
            )
        return value

    def _normalize_calls(
        self,
        supplied: tuple[SemanticCallSpec | Iterable[SemanticCallSpec], ...],
    ) -> tuple[SemanticCallSpec, ...]:
        """Normalize varargs and one explicit iterable to the same compiler path."""
        if len(supplied) == 1 and not isinstance(supplied[0], SemanticCallSpec):
            candidate = supplied[0]
            if isinstance(candidate, (str, bytes)):
                raise TypeError("calls must contain SemanticCallSpec values.")
            try:
                calls = tuple(candidate)
            except TypeError as exc:
                raise TypeError(
                    "A single run argument must be a SemanticCallSpec or iterable."
                ) from exc
        else:
            calls = tuple(supplied)
        if not calls:
            raise ValueError("A semantic workflow requires at least one call.")
        if not all(isinstance(call, SemanticCallSpec) for call in calls):
            raise TypeError("calls must contain SemanticCallSpec values.")
        return calls

    def _reset_workflow(
        self,
        calls: tuple[SemanticCallSpec, ...],
        workflow: object,
        *,
        workflow_id: str,
        eligible_mask: torch.Tensor | None,
        execution_prefix_length: int,
    ) -> None:
        """Reset per-run state while retaining verified symbolic state."""
        if eligible_mask is None:
            eligible = torch.ones(
                self._task_state.batch_size,
                dtype=torch.bool,
                device=self._task_state.device,
            )
        else:
            if not isinstance(eligible_mask, torch.Tensor):
                raise TypeError("eligible_mask must be a torch.Tensor or None.")
            if eligible_mask.dtype != torch.bool or eligible_mask.shape != (
                self._task_state.batch_size,
            ):
                raise ValueError(
                    "eligible_mask must be bool with shape "
                    f"({self._task_state.batch_size},)."
                )
            eligible = eligible_mask.to(self._task_state.device).clone()
        if not eligible.any():
            raise ValueError("eligible_mask must contain at least one active row.")
        self._workflow = workflow
        self._workflow_id = workflow_id
        self._calls = calls
        self._execution_prefix_length = execution_prefix_length
        self._current_call_index = 0
        self._runner = None
        self._grounded = None
        self._active_call = None
        self._active_recovery_item = None
        self._recovery_barrier = None
        self._eligible = eligible
        self._success = torch.zeros_like(eligible)
        self._failed = torch.zeros_like(eligible)
        self._cancelled = torch.zeros_like(eligible)
        self._events = []
        self._call_traces = []
        self._effect_traces = []
        self._workflow_recovery_traces = []
        self._failures = []
        self._call_event_offset = 0
        self._call_effect_offset = 0
        self._observation_revision = 0
        self._next_guard_verification_id = 0
        self._next_gate_verification_id = 0
        self._next_workflow_recovery_id = 0
        self._wait_duration = 0.0
        self._message = None
        self._status = SemanticExecutionStatus.RUNNING

    def _observe_for_grounding(self) -> PlanningContext:
        """Capture and normalize one fresh context for JIT lowering."""
        context = self._observation_provider.observe(self._task_state)
        if not isinstance(context, PlanningContext):
            raise TypeError(
                "ObservationProvider.observe() must return PlanningContext."
            )
        normalized = PlanningContext(
            robot=context.robot,
            task=self._task_state,
            scene=context.scene,
            env_ids=context.env_ids,
            control_dt=context.control_dt,
        )
        if normalized.batch_size != self._task_state.batch_size:
            raise ValueError(
                "Observation batch size changed during semantic execution."
            )
        if normalized.robot.qpos.device != self._task_state.device:
            raise ValueError("Observation and verified TaskState must share a device.")
        if self._has_observed_env_ids:
            if normalized.env_ids.device != self._env_ids.device or not torch.equal(
                normalized.env_ids,
                self._env_ids,
            ):
                raise ValueError(
                    "Observation env_ids must remain stable across call barriers."
                )
        else:
            self._env_ids = normalized.env_ids.clone()
            self._has_observed_env_ids = True
        return normalized

    def _prepare_call(self, call_index: int) -> None:
        """Freshly ground and create exactly one session and runner."""
        assert self._workflow is not None
        self._prepare_grounded_call(
            self._workflow,
            analysis_call_index=call_index,
            workflow_call_index=call_index,
            call=self._calls[call_index],
            active_mask=self._eligible,
            recovery_item=None,
        )

    def _prepare_recovery_work_item(
        self,
        item: _WorkflowRecoveryWorkItem,
    ) -> None:
        """Analyze and ground one real recovery call with fresh observation."""
        barrier = self._require_recovery_barrier()
        suffix = self._calls[barrier.trigger_call_index :]
        analysis_calls = (
            (item.call, *suffix)
            if item.role is SkillWorkflowRecoveryRole.REACQUIRE
            else suffix
        )
        workflow = self._compiler.analyze(
            analysis_calls,
            workflow_id=(
                f"{self._workflow_id}:workflow_recovery:"
                f"{self._next_workflow_recovery_id}"
            ),
        )
        self._prepare_grounded_call(
            workflow,
            analysis_call_index=0,
            workflow_call_index=barrier.trigger_call_index,
            call=item.call,
            active_mask=item.env_mask,
            recovery_item=item,
        )

    def _prepare_grounded_call(
        self,
        workflow: object,
        *,
        analysis_call_index: int,
        workflow_call_index: int,
        call: SemanticCallSpec,
        active_mask: torch.Tensor,
        recovery_item: _WorkflowRecoveryWorkItem | None,
    ) -> None:
        """Install one original or recovery semantic call in a fresh session."""
        context = self._observe_for_grounding()
        grounded = self._compiler.ground(
            workflow,
            analysis_call_index,
            context,
            eligible_mask=active_mask,
        )
        invocation = grounded.invocation
        grounded_eligible = grounded.eligible_mask
        effect_spec = grounded.effect_spec
        effect_monitor = grounded.effect_monitor
        effect_guards = grounded.effect_guards
        effect_gates = grounded.effect_gates
        if not isinstance(grounded_eligible, torch.Tensor) or not torch.equal(
            grounded_eligible,
            active_mask,
        ):
            raise ValueError("Grounded call must preserve runtime eligibility.")
        if (effect_spec is None) != (effect_monitor is None):
            raise ValueError(
                "Grounded effect_spec and effect_monitor must be set together."
            )
        if effect_spec is not None:
            if not isinstance(effect_spec, SemanticEffectSpec):
                raise TypeError("Grounded effect_spec must be a SemanticEffectSpec.")
            if not isinstance(effect_monitor, EffectMonitor):
                raise TypeError("Grounded effect_monitor must be an EffectMonitor.")
            if effect_spec.env_ids.device != context.env_ids.device or not torch.equal(
                effect_spec.env_ids,
                context.env_ids,
            ):
                raise ValueError("Grounded effect env_ids must match the call context.")
        if not all(type(value) is GroundedHeldObjectGuard for value in effect_guards):
            raise TypeError(
                "Grounded effect_guards must contain exact "
                "GroundedHeldObjectGuard values."
            )
        if effect_guards and effect_spec is None:
            raise ValueError("Grounded held-object guards require an effect spec.")
        if not all(type(value) is GroundedPhaseEffectGate for value in effect_gates):
            raise TypeError(
                "Grounded effect_gates must contain exact "
                "GroundedPhaseEffectGate values."
            )
        if effect_gates and effect_spec is None:
            raise ValueError("Grounded phase-effect gates require an effect spec.")

        self._grounded = grounded
        runner_cfg = self._runner_cfg_override
        if runner_cfg is None:
            runner_cfg = grounded.analyzed.bound.preset.runner_cfg
            if not isinstance(runner_cfg, ExecutionRunnerCfg):
                raise TypeError(
                    "Grounded semantic call preset must own an ExecutionRunnerCfg."
                )
        session = self._engine.start(
            (invocation,),
            context,
            eligible_mask=active_mask,
        )
        primed = _PrimedObservationProvider(context, self._observation_provider)
        runner = ExecutionRunner(
            session,
            primed,
            self._command_sink,
            clock=self._clock,
            cfg=runner_cfg,
        )
        self._current_call_index = workflow_call_index
        self._runner = runner
        self._active_call = call
        self._active_recovery_item = recovery_item
        self._call_entered_mask = active_mask.clone()
        self._call_event_offset = len(self._events)
        self._call_effect_offset = len(self._effect_traces)
        self._wait_duration = 0.0

    def _effect_verifier(
        self,
        context: PlanningContext,
        request: EffectVerificationRequest,
    ) -> EffectVerificationResult:
        """Collect raw evidence and feed the grounded call's monitor."""
        grounded = self._require_grounded()
        spec = grounded.effect_spec
        monitor = grounded.effect_monitor
        if not isinstance(spec, SemanticEffectSpec) or not isinstance(
            monitor,
            EffectMonitor,
        ):
            raise RuntimeError(
                "The active atomic plan requested effect verification, but its "
                "semantic call has no grounded effect monitor."
            )
        if request.skill_id != spec.skill_id:
            raise ValueError("Effect request skill_id does not match the effect spec.")
        if request.invocation_id != spec.invocation_id:
            raise ValueError(
                "Effect request invocation_id does not match the effect spec."
            )
        if request.invocation_revision != spec.invocation_revision:
            raise ValueError("Effect request revision does not match the effect spec.")
        decision = self._observe_effect_monitor(
            context,
            request,
            spec=spec,
            monitor=monitor,
        )
        expectation_decisions = self._validated_expectation_decisions(
            spec,
            decision,
        )
        invalidation_mask, retry_mask = self._terminal_failure_policy(
            grounded,
            decision.failure_mask,
            expectation_decisions,
        )
        return EffectVerificationResult(
            verification_id=request.verification_id,
            success_mask=decision.success_mask,
            failure_mask=decision.failure_mask,
            invalidation_mask=invalidation_mask,
            retry_mask=retry_mask,
            expectation_results=tuple(
                EffectExpectationResult(
                    expectation_id=value.expectation_id,
                    satisfied_mask=value.satisfied_mask,
                    contradicted_mask=value.contradicted_mask,
                    inverse_satisfied_mask=value.inverse_satisfied_mask,
                )
                for value in expectation_decisions
            ),
        )

    @staticmethod
    def _project_unverified_effect(
        context: PlanningContext,
        request: EffectVerificationRequest,
    ) -> EffectVerificationResult:
        """Advance projected symbolic state without claiming physical evidence."""
        del context
        accepted = request.env_mask.clone()
        rejected = torch.zeros_like(accepted)
        return EffectVerificationResult(
            verification_id=request.verification_id,
            success_mask=accepted,
            failure_mask=rejected,
            invalidation_mask=rejected,
            retry_mask=rejected,
        )

    @staticmethod
    def _validated_expectation_decisions(
        spec: SemanticEffectSpec,
        decision: EffectMonitorDecision,
    ) -> tuple[EffectExpectationDecision, ...]:
        """Require one current-observation outcome per physical expectation."""
        physical_ids = tuple(
            expectation.expectation_id
            for expectation in spec.state_expectations
            if any(
                clause.expectation_id == expectation.expectation_id
                for clause in spec.clauses
            )
        )
        outcomes = tuple(decision.expectation_decisions)
        outcome_ids = tuple(value.expectation_id for value in outcomes)
        if outcome_ids != physical_ids:
            raise ValueError(
                "Effect monitor must return one ordered outcome for every "
                f"physical expectation; expected={physical_ids}, got={outcome_ids}."
            )
        return outcomes

    @staticmethod
    def _terminal_failure_policy(
        grounded: GroundedSemanticCall,
        failure_mask: torch.Tensor,
        expectation_decisions: tuple[EffectExpectationDecision, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select fail-closed invalidation and safe local retry rows."""
        call = grounded.analyzed.call
        invalidation = failure_mask.clone()
        retry = failure_mask.clone()
        if type(call) is Pick:
            return invalidation, retry
        if type(call) is Place:
            source = next(
                value
                for value in expectation_decisions
                if value.expectation_id == "source"
            )
            retained = failure_mask & source.inverse_satisfied_mask
            return failure_mask & ~retained, retained
        if type(call) is HandOver:
            source = next(
                value
                for value in expectation_decisions
                if value.expectation_id == "source"
            )
            retained = failure_mask & source.inverse_satisfied_mask
            return failure_mask & ~retained, torch.zeros_like(failure_mask)
        return invalidation, retry

    def _phase_effect_gate_verifier(
        self,
        context: PlanningContext,
        request: PhaseEffectGateRequest,
    ) -> PhaseEffectGateResult:
        """Observe one blocking segment-entry effect on a fresh due cycle."""
        grounded = self._require_grounded()
        gates = grounded.effect_gates
        matches = tuple(value for value in gates if value.gate_id == request.gate_id)
        if len(matches) != 1:
            raise RuntimeError(
                f"Grounded call must own exactly one phase-effect gate "
                f"{request.gate_id!r}."
            )
        gate = matches[0]
        if gate.segment_name != request.segment_name:
            raise ValueError(
                "Phase-effect gate request segment does not match its grounded "
                "monitor."
            )
        session = self._require_runner().session
        monitor_request = EffectVerificationRequest(
            verification_id=self._next_gate_verification_id,
            skill_id=request.skill_id,
            invocation_id=request.invocation_id,
            invocation_revision=request.invocation_revision,
            invocation_index=request.invocation_index,
            attempt_generation=request.attempt_generation,
            terminal_segment=request.segment_name,
            requested_at=request.requested_at,
            deadline=request.deadline,
            env_mask=request.env_mask,
            expected_effects=self._phase_effect_gate_expected_effects(
                gate,
                session.active_plan,
            ),
        )
        self._next_gate_verification_id += 1
        decision = self._observe_effect_monitor(
            context,
            monitor_request,
            spec=gate.effect_spec,
            monitor=gate.effect_monitor,
            boundary_kind="phase_effect_gate",
            gate_id=gate.gate_id,
            segment_name=gate.segment_name,
        )
        return PhaseEffectGateResult(
            verification_id=request.verification_id,
            gate_id=request.gate_id,
            attempt_generation=request.attempt_generation,
            invocation_index=request.invocation_index,
            next_waypoint_index=request.next_waypoint_index,
            success_mask=decision.success_mask,
            failure_mask=decision.failure_mask,
            retry_mask=(
                decision.failure_mask
                if gate.retry_action
                else torch.zeros_like(decision.failure_mask)
            ),
            message=(
                f"Physical evidence contradicted gate {gate.gate_id!r} before "
                f"segment {gate.segment_name!r}."
                if decision.failure_mask.any()
                else ""
            ),
        )

    @staticmethod
    def _phase_effect_gate_expected_effects(
        gate: GroundedPhaseEffectGate,
        action_plan: ActionPlan,
    ) -> StateDelta:
        """Project the action-owned held relation required by one gate."""
        expectation = gate.effect_spec.state_expectations[0]
        if type(expectation) is not HeldObjectStateExpectation:
            raise TypeError("Built-in phase-effect gates require held-object state.")
        key = expectation.task_state_key
        if expectation.relation is HeldObjectRelation.ATTACHED:
            candidate = action_plan.effect_candidates.held_object_updates.get(key)
            if candidate is None:
                candidate = action_plan.expected_effects.held_object_updates.get(key)
            if not isinstance(candidate, HeldObjectState):
                raise ValueError(
                    f"Attached gate {gate.gate_id!r} requires an action-owned "
                    "HeldObjectState candidate."
                )
        else:
            if key not in action_plan.expected_effects.held_object_updates:
                raise ValueError(
                    f"Active action does not declare gate state key {key!r}."
                )
            candidate = action_plan.expected_effects.held_object_updates[key]
            if candidate is not None:
                raise ValueError(
                    f"Detached gate {gate.gate_id!r} requires an action-owned "
                    "removal."
                )
        return StateDelta(held_object_updates={key: candidate})

    def _held_object_guard_verifier(
        self,
        context: PlanningContext,
        request: HeldObjectGuardRequest,
    ) -> HeldObjectGuardResult | None:
        """Observe a phase-scoped held-object invariant before dispatch.

        Args:
            context: Fresh due-cycle physical observation.
            request: Core-owned phase and correlation identity.

        Returns:
            Correlated row-local loss decision, or ``None`` when this named
            action segment has no held-object invariant.
        """
        if context.robot.timestamp > request.deadline:
            return None
        grounded = self._require_grounded()
        guards = grounded.effect_guards
        active = tuple(
            guard for guard in guards if request.segment_name in guard.active_segments
        )
        if not active:
            return None
        if len(active) != 1:
            raise RuntimeError(
                "At most one held-object guard may own an action segment; "
                f"segment={request.segment_name!r}, guards="
                f"{[guard.guard_id for guard in active]}."
            )
        guard = active[0]
        session = self._require_runner().session
        if guard.baseline is HeldObjectGuardBaseline.VERIFIED_TASK_STATE:
            candidate = session.task_state.get_held_object(guard.task_state_key)
        else:
            candidate = session.active_plan.effect_candidates.held_object_updates.get(
                guard.task_state_key
            )
            if candidate is None:
                candidate = (
                    session.active_plan.expected_effects.held_object_updates.get(
                        guard.task_state_key
                    )
                )
        covered = torch.zeros_like(request.env_mask)
        if isinstance(candidate, HeldObjectState):
            covered = (
                torch.ones_like(request.env_mask)
                if candidate.env_mask is None
                else candidate.env_mask.to(request.env_mask.device)
            )
            if candidate.semantics.entity_id != self._guard_object_id(
                guard.effect_spec
            ):
                covered.zero_()
        observed_mask = request.env_mask & covered
        failure_mask = request.env_mask & ~covered
        if observed_mask.any():
            assert isinstance(candidate, HeldObjectState)
            verification_id = self._next_guard_verification_id
            self._next_guard_verification_id += 1
            monitor_request = EffectVerificationRequest(
                verification_id=verification_id,
                skill_id=request.skill_id,
                invocation_id=request.invocation_id,
                invocation_revision=request.invocation_revision,
                invocation_index=request.invocation_index,
                attempt_generation=request.attempt_generation,
                terminal_segment=request.segment_name,
                requested_at=context.robot.timestamp,
                deadline=request.deadline,
                env_mask=observed_mask,
                expected_effects=StateDelta(
                    held_object_updates={guard.task_state_key: candidate}
                ),
            )
            decision = self._observe_effect_monitor(
                context,
                monitor_request,
                spec=guard.effect_spec,
                monitor=guard.effect_monitor,
                boundary_kind="in_flight_guard",
                guard_id=guard.guard_id,
                segment_name=request.segment_name,
            )
            failure_mask |= decision.failure_mask
        invalidation = self._held_object_invalidation(
            guard.invalidation_task_state_keys,
            failure_mask,
            session.task_state,
        )
        retry_mask = (
            failure_mask.clone()
            if guard.retry_action
            else torch.zeros_like(failure_mask)
        )
        return HeldObjectGuardResult(
            verification_id=request.verification_id,
            object_id=self._guard_object_id(guard.effect_spec),
            attempt_generation=request.attempt_generation,
            invocation_index=request.invocation_index,
            next_waypoint_index=request.next_waypoint_index,
            failure_mask=failure_mask,
            state_invalidation=invalidation,
            retry_mask=retry_mask,
            message=(
                f"Held-object invariant {guard.guard_id!r} failed during "
                f"segment {request.segment_name!r}."
                if failure_mask.any()
                else ""
            ),
        )

    @staticmethod
    def _guard_object_id(spec: SemanticEffectSpec) -> str:
        """Return the canonical object ID from a single guard expectation."""
        expectation = spec.state_expectations[0]
        object_id = getattr(expectation, "object_id", None)
        if type(object_id) is not str or not object_id:
            raise TypeError("Held-object guard expectation must own an object_id.")
        return object_id

    @staticmethod
    def _held_object_invalidation(
        task_state_keys: tuple[str, ...],
        failure_mask: torch.Tensor,
        task_state: TaskState,
    ) -> StateDelta:
        """Build conservative removal-only reconciliation for failed rows."""
        if not failure_mask.any():
            return StateDelta()
        related = set(task_state_keys)
        return StateDelta(
            held_object_updates={key: None for key in task_state_keys},
            coordinated_held_object_updates={
                resources: None
                for resources in task_state.coordinated_held_objects
                if not set(resources).isdisjoint(related)
            },
        )

    def _observe_effect_monitor(
        self,
        context: PlanningContext,
        request: EffectVerificationRequest,
        *,
        spec: SemanticEffectSpec,
        monitor: EffectMonitor,
        boundary_kind: str = "terminal",
        guard_id: str | None = None,
        gate_id: str | None = None,
        segment_name: str | None = None,
    ) -> EffectMonitorDecision:
        """Collect evidence, run one monitor, and append an auditable trace."""
        grounded = self._require_grounded()
        observation_revision = self._observation_revision
        self._observation_revision += 1
        selected_env_ids = spec.env_ids[request.env_mask.to(spec.env_ids.device)]
        evidence = self._evidence_collector.collect(
            spec,
            timestamp=context.robot.timestamp,
            observation_revision=observation_revision,
            env_ids=selected_env_ids,
        )
        observed = monitor.observe(request, evidence)
        expectation_decisions = self._validated_expectation_decisions(
            spec,
            observed,
        )
        decision = EffectMonitorDecision(
            success_mask=observed.success_mask,
            failure_mask=observed.failure_mask,
            expectation_decisions=expectation_decisions,
        )
        analyzed = grounded.analyzed
        monitor_ref = analyzed.effect_monitor_ref
        if monitor_ref is not None and not isinstance(monitor_ref, EffectMonitorRef):
            raise TypeError("Grounded effect monitor reference must be typed.")
        if monitor_ref is None:
            monitor_id = f"{type(monitor).__module__}.{type(monitor).__qualname__}"
            monitor_revision = None
            configured_monitor_params: Mapping[str, object] = {}
        else:
            monitor_id = monitor_ref.monitor_id
            monitor_revision = monitor_ref.revision
            configured_monitor_params = monitor_ref.params
        resolved_monitor_params = monitor.resolved_params
        if not isinstance(resolved_monitor_params, Mapping):
            raise TypeError("EffectMonitor.resolved_params must return a mapping.")
        trace = SkillEffectTrace(
            call_index=self._require_call_index(),
            verification_id=request.verification_id,
            observation_revision=observation_revision,
            timestamp=context.robot.timestamp,
            success_mask=decision.success_mask,
            failure_mask=decision.failure_mask,
            expectation_decisions=decision.expectation_decisions,
            effect_spec=spec,
            monitor_id=monitor_id,
            monitor_revision=monitor_revision,
            configured_monitor_params=configured_monitor_params,
            resolved_monitor_params=resolved_monitor_params,
            evidence=evidence,
            boundary_kind=boundary_kind,
            guard_id=guard_id,
            gate_id=gate_id,
            segment_name=segment_name,
        )
        self._effect_traces.append(trace)
        return decision

    def _consume_runner_step(self, runner_step: RunnerStep) -> None:
        """Merge one runner update into workflow-level traces."""
        self._wait_duration = runner_step.wait_duration
        if runner_step.tick is not None:
            self._task_state = _snapshot_task_state(runner_step.tick.task_state)
            self._events.extend(
                _snapshot_event(event) for event in runner_step.tick.events
            )
        if runner_step.message:
            self._message = runner_step.message

    def _finish_active_call(self, runner_step: RunnerStep) -> _FinishedCallAttempt:
        """Project one terminal session without deciding workflow eligibility."""
        runner = self._require_runner()
        grounded = self._require_grounded()
        call_index = self._require_call_index()
        call = self._active_call
        if not isinstance(call, SemanticCallSpec):
            raise RuntimeError("No semantic call is associated with the active runner.")
        self._task_state = _snapshot_task_state(runner.session.task_state)
        after = runner.session.eligible_mask
        invocation = grounded.invocation
        if runner_step.status is RunnerStatus.COMPLETED:
            completed = self._call_entered_mask & after
            failed = self._call_entered_mask & ~after & ~self._cancelled
        elif runner_step.status is RunnerStatus.CANCELLED:
            completed = torch.zeros_like(self._call_entered_mask)
            failed = torch.zeros_like(self._call_entered_mask)
        else:
            completed = torch.zeros_like(self._call_entered_mask)
            failed = self._call_entered_mask & ~self._cancelled
        self._reconcile_observed_held_relations(
            grounded,
            runner.session.latest_context,
            completed,
        )
        plan_attempts = tuple(
            SkillPlanAttemptTrace.from_execution_attempt(
                attempt,
                profile_id=grounded.analyzed.bound.robot_profile.profile_id,
                preset_id=grounded.analyzed.bound.preset.preset_id,
            )
            for attempt in runner.session.plan_attempts
        )
        trace = SkillCallTrace(
            call_index=call_index,
            semantic_id=call.semantic_id,
            call_metadata=call.to_metadata(),
            skill_id=invocation.skill_id,
            invocation_id=invocation.invocation_id,
            invocation_revision=invocation.revision,
            status=runner_step.status,
            entered_mask=self._call_entered_mask,
            completed_mask=completed,
            failed_mask=failed,
            command_count=runner_step.command_count,
            resolved_core_policy=plan_attempts[-1].resolved_core_policy,
            plan_attempts=plan_attempts,
            events=tuple(self._events[self._call_event_offset :]),
            effects=tuple(self._effect_traces[self._call_effect_offset :]),
        )
        self._runner = None
        self._grounded = None
        self._active_call = None
        self._active_recovery_item = None
        return _FinishedCallAttempt(
            trace=trace,
            completed_mask=completed,
            failed_mask=failed,
            status=runner_step.status,
            message=runner_step.message,
        )

    def _reconcile_observed_held_relations(
        self,
        grounded: GroundedSemanticCall,
        context: PlanningContext,
        completed_mask: torch.Tensor,
    ) -> None:
        """Replace projected attachment transforms with terminal measurements.

        Grasp execution can move an object relative to its planned contact frame.
        A downstream object-space action must therefore use the relation measured
        after the preceding call, rather than continuing to project the original
        grasp candidate.  Only completed rows and individually held relations are
        updated; missing or zero-confidence scene observations retain their last
        verified projection.
        """
        if not completed_mask.any():
            return

        endpoints_by_state_key: dict[str, list[EndpointBinding]] = {}
        for endpoint in grounded.invocation.binding.endpoints:
            if FORWARD_KINEMATICS_CAPABILITY not in endpoint.capabilities:
                continue
            task_state_key = endpoint.task_state_key
            if task_state_key is None:
                continue
            endpoints_by_state_key.setdefault(task_state_key, []).append(endpoint)

        for task_state_key, endpoints in endpoints_by_state_key.items():
            held = self._task_state.get_held_object(task_state_key)
            if not isinstance(held, HeldObjectState):
                continue
            entity_id = held.semantics.entity_id
            if entity_id is None:
                continue
            entity = context.scene.entities.get(entity_id)
            if entity is None or entity.confidence <= 0.0:
                continue

            unique_endpoints = {
                endpoint.destination_key: endpoint for endpoint in endpoints
            }
            if len(unique_endpoints) != 1:
                raise ValueError(
                    "A held-object task-state key must resolve to exactly one "
                    "forward-kinematics endpoint."
                )
            endpoint = next(iter(unique_endpoints.values()))
            active = completed_mask & held.env_mask
            if not active.any():
                continue

            joint_ids = torch.tensor(
                endpoint.joint_ids,
                dtype=torch.long,
                device=context.robot.qpos.device,
            )
            endpoint_qpos = context.robot.qpos.index_select(1, joint_ids)
            endpoint_pose = self._engine.robot.compute_fk(
                qpos=endpoint_qpos,
                name=endpoint.target.target_id,
                env_ids=context.env_ids.detach().cpu().tolist(),
                to_matrix=True,
            )
            if not isinstance(endpoint_pose, torch.Tensor) or endpoint_pose.shape != (
                context.batch_size,
                4,
                4,
            ):
                raise ValueError(
                    "Forward kinematics must return one 4x4 endpoint pose per "
                    "environment when reconciling a held object."
                )
            endpoint_pose = endpoint_pose.to(
                device=context.robot.qpos.device,
                dtype=context.robot.qpos.dtype,
            )
            object_pose = entity.pose.to(
                device=endpoint_pose.device,
                dtype=endpoint_pose.dtype,
            )
            if object_pose.shape == (4, 4):
                object_pose = object_pose.unsqueeze(0).expand(
                    context.batch_size,
                    -1,
                    -1,
                )
            if object_pose.shape != endpoint_pose.shape:
                raise ValueError(
                    "A held object's observed pose must be one 4x4 transform per "
                    "environment."
                )

            observed = HeldObjectState(
                semantics=held.semantics,
                object_to_eef=torch.bmm(pose_inv(object_pose), endpoint_pose),
                grasp_xpos=endpoint_pose,
                env_mask=held.env_mask,
            )
            self._task_state = StateDelta(
                held_object_updates={task_state_key: observed}
            ).apply(self._task_state, active)

    def _workflow_recovery_trigger(self) -> _WorkflowRecoveryTrigger | None:
        """Resolve preset policy and the failed call's physical source endpoint."""
        call = self._active_call
        if type(call) is Place:
            source_slot = "primary"
        elif type(call) is HandOver:
            source_slot = "source"
        else:
            return None
        grounded = self._require_grounded()
        preset = grounded.analyzed.bound.preset
        policy = preset.workflow_recovery_policy
        if type(policy) is not WorkflowRecoveryPolicy:
            raise TypeError("Grounded preset workflow_recovery_policy must be exact.")
        if policy.max_recovery_attempts == 0:
            return None
        endpoints = tuple(
            endpoint
            for endpoint in grounded.invocation.binding.endpoints
            if endpoint.slot_id == source_slot
        )
        resource_ids = {endpoint.resource_id for endpoint in endpoints}
        task_state_keys = {endpoint.task_state_key for endpoint in endpoints}
        if not endpoints or len(resource_ids) != 1 or len(task_state_keys) != 1:
            raise RuntimeError(
                f"Workflow recovery requires one physical source resource and "
                f"task-state key for slot {source_slot!r}."
            )
        return _WorkflowRecoveryTrigger(
            policy=policy,
            source_resource_id=next(iter(resource_ids)),
            source_task_state_key=next(iter(task_state_keys)),
        )

    def _active_call_requires_workflow_recovery(self) -> bool:
        """Whether the active call emitted a row-local external-recovery hand-off."""
        entered = self._call_entered_mask
        return any(
            event.kind is ExecutionEventKind.RECOVERY_REQUIRED
            and bool((event.env_mask.to(entered.device) & entered).any().item())
            for event in self._events[self._call_event_offset :]
        )

    @staticmethod
    def _recovery_required_mask(trace: SkillCallTrace) -> torch.Tensor:
        """Return failed rows explicitly handed to workflow recovery by core."""
        required = torch.zeros_like(trace.failed_mask)
        for event in trace.events:
            if event.kind is ExecutionEventKind.RECOVERY_REQUIRED:
                required |= event.env_mask.to(required.device)
        return required & trace.failed_mask

    def _handle_original_call_finished(
        self,
        finished: _FinishedCallAttempt,
        *,
        trigger: _WorkflowRecoveryTrigger | None,
    ) -> None:
        """Either advance one call barrier or start bounded row-local recovery."""
        if finished.status is RunnerStatus.CANCELLED:
            cancelled = finished.trace.entered_mask & self._eligible
            self._cancelled |= cancelled
            self._eligible &= ~cancelled
            self._finish_workflow_terminal()
            return
        recovery_required = self._recovery_required_mask(finished.trace)
        recoverable = (
            torch.zeros_like(recovery_required)
            if trigger is None
            else recovery_required
        )
        if not recoverable.any():
            self._complete_original_call_barrier(
                success_mask=finished.completed_mask,
                failure_mask=finished.failed_mask,
                message=finished.message,
            )
            return
        assert trigger is not None
        permanent_failure = finished.failed_mask & ~recoverable
        call_index = self._require_call_index()
        barrier = _WorkflowRecoveryBarrier(
            trigger_call_index=call_index,
            trigger_call=self._calls[call_index],
            policy=trigger.policy,
            source_resource_id=trigger.source_resource_id,
            source_task_state_key=trigger.source_task_state_key,
            entered_mask=finished.trace.entered_mask.clone(),
            success_mask=finished.completed_mask.clone(),
            final_failure_mask=permanent_failure.clone(),
            attempt_counts=torch.zeros_like(
                finished.trace.entered_mask,
                dtype=torch.long,
            ),
            work_items=deque(),
            failure_messages=(
                [finished.message or "Semantic call failed for some rows."]
                if permanent_failure.any()
                else []
            ),
        )
        self._recovery_barrier = barrier
        self._eligible = (barrier.success_mask | recoverable) & ~self._cancelled
        self._failed |= permanent_failure
        self._schedule_recovery_cycle(recoverable)
        self._start_next_recovery_work_item_or_finish()

    def _handle_recovery_call_finished(
        self,
        item: _WorkflowRecoveryWorkItem,
        finished: _FinishedCallAttempt,
    ) -> None:
        """Update one recovery cohort and retain the shared call barrier."""
        barrier = self._require_recovery_barrier()
        self._append_workflow_recovery_trace(
            item,
            call=finished.trace,
            completed_mask=finished.completed_mask,
            failed_mask=finished.failed_mask,
            message=finished.message,
        )
        if finished.status is RunnerStatus.CANCELLED:
            cancelled = item.env_mask & self._eligible
            self._cancelled |= cancelled
            self._eligible &= ~cancelled
        elif item.role is SkillWorkflowRecoveryRole.REACQUIRE:
            if finished.completed_mask.any():
                barrier.work_items.append(
                    _WorkflowRecoveryWorkItem(
                        role=SkillWorkflowRecoveryRole.RETRY_REACQUIRED,
                        call=barrier.trigger_call,
                        env_mask=finished.completed_mask,
                        attempt_index=item.attempt_index,
                    )
                )
            if finished.failed_mask.any():
                self._schedule_recovery_cycle(finished.failed_mask)
        else:
            barrier.success_mask |= finished.completed_mask
            if finished.failed_mask.any():
                recovery_required = self._recovery_required_mask(finished.trace)
                permanent = finished.failed_mask & ~recovery_required
                self._record_permanent_recovery_failure(
                    permanent,
                    finished.message
                    or "The retried semantic call failed without a recovery hand-off.",
                )
                self._schedule_recovery_cycle(recovery_required)
        self._start_next_recovery_work_item_or_finish()

    def _schedule_recovery_cycle(self, requested_mask: torch.Tensor) -> None:
        """Consume one per-row budget and enqueue retained/reacquire cohorts."""
        barrier = self._require_recovery_barrier()
        requested = requested_mask & self._eligible & ~self._cancelled
        allowed = requested & (
            barrier.attempt_counts < barrier.policy.max_recovery_attempts
        )
        exhausted = requested & ~allowed
        self._record_permanent_recovery_failure(
            exhausted,
            "Workflow recovery exhausted its per-row attempt budget.",
        )
        if not allowed.any():
            return
        barrier.attempt_counts[allowed] += 1
        for attempt_index in range(1, barrier.policy.max_recovery_attempts + 1):
            cohort = allowed & (barrier.attempt_counts == attempt_index)
            if not cohort.any():
                continue
            retained = self._retained_source_mask(cohort)
            reacquire = cohort & ~retained
            if retained.any():
                barrier.work_items.append(
                    _WorkflowRecoveryWorkItem(
                        role=SkillWorkflowRecoveryRole.RETRY_RETAINED,
                        call=barrier.trigger_call,
                        env_mask=retained,
                        attempt_index=attempt_index,
                    )
                )
            if reacquire.any():
                barrier.work_items.append(
                    _WorkflowRecoveryWorkItem(
                        role=SkillWorkflowRecoveryRole.REACQUIRE,
                        call=self._reacquisition_call(barrier),
                        env_mask=reacquire,
                        attempt_index=attempt_index,
                    )
                )

    def _retained_source_mask(self, env_mask: torch.Tensor) -> torch.Tensor:
        """Return rows whose reconciled symbolic state proves source retention."""
        barrier = self._require_recovery_barrier()
        held = self._task_state.get_held_object(barrier.source_task_state_key)
        if not isinstance(held, HeldObjectState):
            return torch.zeros_like(env_mask)
        trigger_object = getattr(barrier.trigger_call, "object", None)
        object_id = getattr(trigger_object, "entity_id", None)
        if held.semantics.entity_id != object_id:
            return torch.zeros_like(env_mask)
        active = (
            torch.ones_like(env_mask)
            if held.env_mask is None
            else held.env_mask.to(env_mask.device)
        )
        return env_mask & active

    def _reacquisition_call(self, barrier: _WorkflowRecoveryBarrier) -> Pick:
        """Derive a real Pick using the failed call's resolved source resource."""
        trigger_object = getattr(barrier.trigger_call, "object", None)
        if type(trigger_object) is not SceneObjectRef:
            raise TypeError("Curated workflow recovery requires a SceneObjectRef.")
        grasp: SceneAffordanceRef | None = None
        for candidate in reversed(self._calls[: barrier.trigger_call_index]):
            if (
                type(candidate) is Pick
                and candidate.object.entity_id == trigger_object.entity_id
            ):
                grasp = candidate.grasp
                break
        return Pick(
            object=SceneObjectRef(trigger_object.entity_id),
            grasp=(None if grasp is None else SceneAffordanceRef(grasp.entity_id)),
            resources={"primary": barrier.source_resource_id},
        )

    def _start_next_recovery_work_item_or_finish(self) -> None:
        """Start the next non-empty cohort or close the recovered call barrier."""
        barrier = self._require_recovery_barrier()
        while barrier.work_items:
            queued = barrier.work_items.popleft()
            active = queued.env_mask & self._eligible & ~self._cancelled
            if not active.any():
                continue
            item = _WorkflowRecoveryWorkItem(
                role=queued.role,
                call=queued.call,
                env_mask=active,
                attempt_index=queued.attempt_index,
            )
            try:
                self._prepare_recovery_work_item(item)
            except Exception as exc:  # noqa: BLE001 - row-local recovery failure
                message = (
                    f"Could not prepare workflow recovery call "
                    f"{item.call.semantic_id!r}: {type(exc).__name__}: {exc}"
                )
                self._append_workflow_recovery_trace(
                    item,
                    call=None,
                    completed_mask=torch.zeros_like(item.env_mask),
                    failed_mask=item.env_mask,
                    message=message,
                )
                self._record_permanent_recovery_failure(item.env_mask, message)
                self._runner = None
                self._grounded = None
                self._active_call = None
                self._active_recovery_item = None
                continue
            return
        self._finish_recovery_barrier()

    def _append_workflow_recovery_trace(
        self,
        item: _WorkflowRecoveryWorkItem,
        *,
        call: SkillCallTrace | None,
        completed_mask: torch.Tensor,
        failed_mask: torch.Tensor,
        message: str | None,
    ) -> None:
        """Append one immutable recovery-call trace with stable correlation."""
        barrier = self._require_recovery_barrier()
        self._workflow_recovery_traces.append(
            SkillWorkflowRecoveryTrace(
                recovery_id=self._next_workflow_recovery_id,
                trigger_call_index=barrier.trigger_call_index,
                trigger_semantic_id=barrier.trigger_call.semantic_id,
                attempt_index=item.attempt_index,
                max_recovery_attempts=barrier.policy.max_recovery_attempts,
                role=item.role,
                source_resource_id=barrier.source_resource_id,
                source_task_state_key=barrier.source_task_state_key,
                entered_mask=item.env_mask,
                completed_mask=completed_mask,
                failed_mask=failed_mask,
                call=call,
                message=message,
            )
        )
        self._next_workflow_recovery_id += 1

    def _record_permanent_recovery_failure(
        self,
        env_mask: torch.Tensor,
        message: str,
    ) -> None:
        """Remove exhausted rows while leaving other recovery cohorts active."""
        if not env_mask.any():
            return
        barrier = self._require_recovery_barrier()
        barrier.final_failure_mask |= env_mask
        barrier.failure_messages.append(message)
        self._failed |= env_mask
        self._eligible &= ~env_mask

    def _finish_recovery_barrier(self) -> None:
        """Rejoin recovered rows and advance the original program counter once."""
        barrier = self._require_recovery_barrier()
        unresolved = (
            barrier.entered_mask
            & ~barrier.success_mask
            & ~barrier.final_failure_mask
            & ~self._cancelled
        )
        if unresolved.any():
            self._record_permanent_recovery_failure(
                unresolved,
                "Workflow recovery ended with unresolved rows.",
            )
        success = barrier.success_mask & ~self._cancelled
        failure = barrier.final_failure_mask & ~self._cancelled
        message = (
            None
            if not failure.any()
            else "; ".join(dict.fromkeys(barrier.failure_messages))
        )
        self._recovery_barrier = None
        self._complete_original_call_barrier(
            success_mask=success,
            failure_mask=failure,
            message=message,
        )

    def _complete_original_call_barrier(
        self,
        *,
        success_mask: torch.Tensor,
        failure_mask: torch.Tensor,
        message: str | None,
    ) -> None:
        """Commit final row outcomes and advance exactly one original call."""
        call_index = self._require_call_index()
        call = self._calls[call_index]
        failure = failure_mask & ~self._cancelled
        self._failed |= failure
        self._eligible = success_mask & ~self._failed & ~self._cancelled
        if failure.any():
            failure_message = message or "Semantic call failed for these rows."
            self._failures.append(
                SkillFailure(
                    call_index=call_index,
                    semantic_id=call.semantic_id,
                    env_mask=failure,
                    message=failure_message,
                )
            )
            self._message = failure_message
        elif self._workflow_recovery_traces:
            self._message = None
        if self._eligible.any() and self._has_next_call:
            next_index = call_index + 1
            try:
                self._prepare_call(next_index)
            except Exception as exc:  # noqa: BLE001 - preserve workflow trace
                self._fail_preparation(next_index, exc)
        elif self._eligible.any():
            self._success = self._eligible.clone()
            self._status = SemanticExecutionStatus.COMPLETED
            self._current_call_index = None
            self._wait_duration = 0.0
        else:
            self._finish_workflow_terminal()

    def _finish_workflow_terminal(self) -> None:
        """Choose one terminal status from final row-local outcomes."""
        self._status = (
            SemanticExecutionStatus.CANCELLED
            if self._cancelled.any() and not self._failed.any()
            else SemanticExecutionStatus.FAILED
        )
        self._current_call_index = None
        self._wait_duration = 0.0

    def _fail_preparation(self, call_index: int, exc: Exception) -> None:
        """Convert a post-barrier grounding failure to a terminal result."""
        failed = self._eligible.clone()
        self._failed |= failed
        self._eligible &= ~failed
        semantic_id = self._calls[call_index].semantic_id
        message = (
            f"Could not prepare semantic call {call_index} ({semantic_id!r}): "
            f"{type(exc).__name__}: {exc}"
        )
        diagnostic = (
            exc.diagnostic if isinstance(exc, SemanticValidationError) else None
        )
        self._failures.append(
            SkillFailure(
                call_index=call_index,
                semantic_id=semantic_id,
                env_mask=failed,
                message=message,
                code=(
                    "semantic_call_preparation_failed"
                    if diagnostic is None
                    else diagnostic.code
                ),
                phase="preparation",
                diagnostic=diagnostic,
            )
        )
        self._append_preparation_failure_trace(call_index, failed)
        self._message = message
        self._status = SemanticExecutionStatus.FAILED
        self._current_call_index = None
        self._runner = None
        self._grounded = None
        self._active_call = None
        self._active_recovery_item = None
        self._recovery_barrier = None
        self._wait_duration = 0.0

    def _append_preparation_failure_trace(
        self,
        call_index: int,
        failed_mask: torch.Tensor,
    ) -> None:
        """Record statically resolved policy choices when planning never starts."""
        grounded = self._grounded
        analyzed = getattr(grounded, "analyzed", None)
        invocation = getattr(grounded, "invocation", None)
        if analyzed is None:
            workflow_calls = getattr(self._workflow, "calls", ())
            if call_index < len(workflow_calls):
                analyzed = workflow_calls[call_index]
        bound = getattr(analyzed, "bound", None)
        if bound is None:
            return
        try:
            profile = bound.robot_profile
            preset = bound.preset
            action_binding = (
                bound.binding.action_binding
                if invocation is None
                else invocation.binding
            )
            resolved = ResolvedCorePolicyTrace.from_resolved_binding(
                profile_id=profile.profile_id,
                preset_id=preset.preset_id,
                motion_policy=(
                    preset.motion_policy
                    if invocation is None
                    else invocation.motion_policy
                ),
                tracking_policy=(
                    preset.tracking_policy
                    if invocation is None
                    else invocation.tracking_policy
                ),
                recovery_policy=(
                    preset.recovery_policy
                    if invocation is None
                    else invocation.recovery_policy
                ),
                endpoints=action_binding.endpoints,
            )
            skill_id = bound.linked.descriptor.skill_id
        except (AttributeError, TypeError, ValueError):
            return
        self._call_traces.append(
            SkillCallTrace(
                call_index=call_index,
                semantic_id=self._calls[call_index].semantic_id,
                call_metadata=self._calls[call_index].to_metadata(),
                skill_id=skill_id,
                invocation_id=(
                    None if invocation is None else invocation.invocation_id
                ),
                invocation_revision=(0 if invocation is None else invocation.revision),
                status=RunnerStatus.FAILED,
                entered_mask=failed_mask,
                completed_mask=torch.zeros_like(failed_mask),
                failed_mask=failed_mask,
                command_count=0,
                resolved_core_policy=resolved,
                plan_attempts=(),
            )
        )

    def _abort(self, reason: str) -> None:
        """Safe-stop the active runner and mark remaining rows failed."""
        if self._runner is not None:
            recovery_item = self._active_recovery_item
            safe_stop_step = self._runner.cancel(reason)
            runner_step = replace(
                safe_stop_step,
                status=RunnerStatus.FAILED,
                message=reason,
            )
            self._consume_runner_step(runner_step)
            finished = self._finish_active_call(runner_step)
            if recovery_item is None:
                self._call_traces.append(finished.trace)
            elif self._recovery_barrier is not None:
                self._append_workflow_recovery_trace(
                    recovery_item,
                    call=finished.trace,
                    completed_mask=finished.completed_mask,
                    failed_mask=finished.failed_mask,
                    message=reason,
                )
        failed = self._eligible.clone()
        self._failed |= failed
        self._eligible &= ~failed
        if failed.any() and self._calls:
            call_index = min(
                self._current_call_index or 0,
                len(self._calls) - 1,
            )
            self._failures.append(
                SkillFailure(
                    call_index=call_index,
                    semantic_id=self._calls[call_index].semantic_id,
                    env_mask=failed,
                    message=reason,
                    code="semantic_runtime_aborted",
                    phase="runtime",
                )
            )
        self._message = reason
        self._status = SemanticExecutionStatus.FAILED
        self._recovery_barrier = None
        self._current_call_index = None
        self._wait_duration = 0.0

    def _require_runner(self) -> ExecutionRunner:
        if self._runner is None:
            raise RuntimeError("No semantic call runner is active.")
        return self._runner

    def _require_grounded(self) -> GroundedSemanticCall:
        if self._grounded is None:
            raise RuntimeError("No grounded semantic call is active.")
        return self._grounded

    def _require_recovery_barrier(self) -> _WorkflowRecoveryBarrier:
        if self._recovery_barrier is None:
            raise RuntimeError("No workflow-recovery barrier is active.")
        return self._recovery_barrier

    def _require_call_index(self) -> int:
        if self._current_call_index is None:
            raise RuntimeError("No semantic call is active.")
        return self._current_call_index


__all__: list[str] = []
