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

"""High-level orchestration for semantic skill workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING

import torch

from ..atomic_actions.engine import AtomicActionEngine
from ..atomic_actions.execution import (
    EffectVerificationRequest,
    ExecutionEvent,
    ExecutionTick,
)
from ..atomic_actions.runner import (
    CommandSink,
    ExecutionClock,
    ExecutionRunner,
    ExecutionRunnerCfg,
    MonotonicExecutionClock,
    ObservationProvider,
    RunnerStatus,
    RunnerStep,
    RunnerStepCallback,
)
from ..atomic_actions.sim_adapter import SimulationExecutionAdapter
from ..atomic_actions.state import PlanningContext, TaskState
from .calls import (
    SemanticCallCatalog,
    SemanticCallDescriptor,
    SemanticCallSpec,
    builtin_semantic_call_catalog,
)
from .compiler import (
    GroundedSemanticCall,
    HandOverPoseProvider,
    RegisteredSemanticLowerer,
    RelationTargetGrounder,
    SemanticSkillCompiler,
    SemanticWorkflow,
)
from .integration import SceneManifest, SemanticIntegrationManifest
from .profiles import (
    ResourceEndpoint,
    ResourceEndpointAdapter,
    RobotSkillProfile,
)
from .scene import SceneRegistry

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.planners import MotionGenerator
    from embodichain.lab.sim.sim_manager import SimulationManager


SemanticEffectVerifier = Callable[
    [SemanticCallSpec, EffectVerificationRequest, PlanningContext],
    torch.Tensor,
]
"""Verify one semantic effect and return a per-environment success mask."""


class SemanticExecutionStatus(str, Enum):
    """Lifecycle status of one semantic workflow segment."""

    RUNNING = "running"
    WAITING_FOR_EFFECT = "waiting_for_effect"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SemanticTaskStatus(str, Enum):
    """Terminal status of one semantic task."""

    SUCCEEDED = "succeeded"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True, eq=False)
class SemanticCallRecord:
    """Terminal execution record for one grounded semantic call."""

    call_index: int
    semantic_id: str
    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    status: RunnerStatus
    eligible_mask: torch.Tensor
    events: tuple[ExecutionEvent, ...]
    command_count: int
    message: str | None = None

    def __post_init__(self) -> None:
        if self.call_index < 0:
            raise ValueError("call_index must be non-negative.")
        for name in ("semantic_id", "skill_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if self.invocation_revision < 0:
            raise ValueError("invocation_revision must be non-negative.")
        if self.invocation_id is not None and (
            not isinstance(self.invocation_id, str) or not self.invocation_id
        ):
            raise ValueError("invocation_id must be a non-empty string or None.")
        if not isinstance(self.status, RunnerStatus):
            raise TypeError("status must be a RunnerStatus.")
        if (
            not isinstance(self.eligible_mask, torch.Tensor)
            or self.eligible_mask.dtype != torch.bool
            or self.eligible_mask.dim() != 1
        ):
            raise ValueError("eligible_mask must be a one-dimensional bool tensor.")
        if not all(isinstance(event, ExecutionEvent) for event in self.events):
            raise TypeError("events must contain ExecutionEvent values.")
        if self.command_count < 0:
            raise ValueError("command_count must be non-negative.")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be a string or None.")
        object.__setattr__(self, "eligible_mask", self.eligible_mask.clone())
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True, slots=True, eq=False)
class SemanticSegmentResult:
    """Terminal result of one statically analyzed workflow segment."""

    segment_id: str
    workflow_id: str
    status: SemanticExecutionStatus
    eligible_mask: torch.Tensor
    task_state: TaskState
    calls: tuple[SemanticCallRecord, ...]
    message: str | None = None

    def __post_init__(self) -> None:
        for name in ("segment_id", "workflow_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.status, SemanticExecutionStatus) or self.status not in {
            SemanticExecutionStatus.COMPLETED,
            SemanticExecutionStatus.FAILED,
            SemanticExecutionStatus.CANCELLED,
        }:
            raise ValueError("SemanticSegmentResult status must be terminal.")
        if (
            not isinstance(self.eligible_mask, torch.Tensor)
            or self.eligible_mask.dtype != torch.bool
            or self.eligible_mask.dim() != 1
        ):
            raise ValueError("eligible_mask must be a one-dimensional bool tensor.")
        if not isinstance(self.task_state, TaskState):
            raise TypeError("task_state must be a TaskState.")
        if not all(isinstance(call, SemanticCallRecord) for call in self.calls):
            raise TypeError("calls must contain SemanticCallRecord values.")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be a string or None.")
        object.__setattr__(self, "eligible_mask", self.eligible_mask.clone())
        object.__setattr__(self, "calls", tuple(self.calls))

    @property
    def events(self) -> tuple[ExecutionEvent, ...]:
        """Return all structured execution events in call order."""
        return tuple(event for call in self.calls for event in call.events)


@dataclass(frozen=True, slots=True, eq=False)
class SemanticTaskResult:
    """Terminal result of a complete static or dynamically segmented task."""

    task_id: str
    status: SemanticTaskStatus
    initial_eligible_mask: torch.Tensor
    eligible_mask: torch.Tensor
    task_state: TaskState
    latest_context: PlanningContext
    segments: tuple[SemanticSegmentResult, ...]
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string.")
        if not isinstance(self.status, SemanticTaskStatus):
            raise TypeError("status must be a SemanticTaskStatus.")
        for name in ("initial_eligible_mask", "eligible_mask"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.bool
                or value.dim() != 1
            ):
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
            object.__setattr__(self, name, value.clone())
        if self.initial_eligible_mask.shape != self.eligible_mask.shape:
            raise ValueError("Task eligibility masks must share a shape.")
        if not isinstance(self.task_state, TaskState):
            raise TypeError("task_state must be a TaskState.")
        if not isinstance(self.latest_context, PlanningContext):
            raise TypeError("latest_context must be a PlanningContext.")
        if not all(isinstance(item, SemanticSegmentResult) for item in self.segments):
            raise TypeError("segments must contain SemanticSegmentResult values.")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be a string or None.")
        object.__setattr__(self, "segments", tuple(self.segments))

    @property
    def events(self) -> tuple[ExecutionEvent, ...]:
        """Return all structured execution events in segment order."""
        return tuple(event for segment in self.segments for event in segment.events)

    def require_all_succeeded(self) -> None:
        """Require every initially eligible environment to have succeeded.

        Raises:
            RuntimeError: If the task failed, was cancelled, or retained only a
                subset of its initial environment cohort.
        """
        if self.status is not SemanticTaskStatus.SUCCEEDED:
            detail = "" if self.message is None else f" {self.message}"
            raise RuntimeError(
                f"Semantic task {self.task_id!r} finished with "
                f"{self.status.value!r}.{detail}"
            )


@dataclass(frozen=True, slots=True, eq=False)
class SemanticExecutionStep:
    """Latest high-level state after advancing a semantic segment."""

    status: SemanticExecutionStatus
    task_id: str
    segment_id: str
    call_index: int
    eligible_mask: torch.Tensor
    runner_step: RunnerStep | None = None
    pending_effect: EffectVerificationRequest | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, SemanticExecutionStatus):
            raise TypeError("status must be a SemanticExecutionStatus.")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string.")
        if not isinstance(self.segment_id, str) or not self.segment_id:
            raise ValueError("segment_id must be a non-empty string.")
        if self.call_index < 0:
            raise ValueError("call_index must be non-negative.")
        if (
            not isinstance(self.eligible_mask, torch.Tensor)
            or self.eligible_mask.dtype != torch.bool
            or self.eligible_mask.dim() != 1
        ):
            raise ValueError("eligible_mask must be a one-dimensional bool tensor.")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be a string or None.")
        object.__setattr__(self, "eligible_mask", self.eligible_mask.clone())


class SemanticSkillRuntime:
    """Bind semantic declarations to closed-loop planning and execution ports.

    The runtime is intentionally thin. It owns one compiler and the controller
    ports used to construct existing :class:`ExecutionRunner` instances. A
    :class:`SemanticTask` owns verified symbolic state across workflow segments.

    Args:
        compiler: Bound compiler used for static analysis and JIT grounding.
        observation_provider: Source of fresh robot and scene observations.
        command_sink: Destination for controller commands and safe-stop requests.
        clock: Optional execution clock. Wall-clock time is used when omitted.
        effect_verifier: Optional default callback for physical effect checks.
        runner_cfg: Optional transport and scheduling policy overriding each
            call's skill preset.
    """

    def __init__(
        self,
        compiler: SemanticSkillCompiler,
        observation_provider: ObservationProvider,
        command_sink: CommandSink,
        *,
        clock: ExecutionClock | None = None,
        effect_verifier: SemanticEffectVerifier | None = None,
        runner_cfg: ExecutionRunnerCfg | None = None,
    ) -> None:
        if not isinstance(compiler, SemanticSkillCompiler):
            raise TypeError("compiler must be a SemanticSkillCompiler.")
        if not isinstance(observation_provider, ObservationProvider):
            raise TypeError("observation_provider must implement ObservationProvider.")
        if not isinstance(command_sink, CommandSink):
            raise TypeError("command_sink must implement CommandSink.")
        if clock is not None and not isinstance(clock, ExecutionClock):
            raise TypeError("clock must implement ExecutionClock.")
        if effect_verifier is not None and not callable(effect_verifier):
            raise TypeError("effect_verifier must be callable or None.")
        if runner_cfg is not None and not isinstance(runner_cfg, ExecutionRunnerCfg):
            raise TypeError("runner_cfg must be an ExecutionRunnerCfg or None.")
        self.compiler = compiler
        self.observation_provider = observation_provider
        self.command_sink = command_sink
        self.clock = MonotonicExecutionClock() if clock is None else clock
        self.effect_verifier = effect_verifier
        self.runner_cfg: ExecutionRunnerCfg | None = (
            None if runner_cfg is None else deepcopy(runner_cfg)
        )
        self._active_task: SemanticTask | None = None

    @classmethod
    def bind(
        cls,
        *,
        manifest: SemanticIntegrationManifest,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        observation_provider: ObservationProvider,
        command_sink: CommandSink,
        clock: ExecutionClock | None = None,
        effect_verifier: SemanticEffectVerifier | None = None,
        registered_lowerers: Iterable[RegisteredSemanticLowerer] = (),
        relation_grounders: Iterable[RelationTargetGrounder] = (),
        handover_pose_providers: Iterable[HandOverPoseProvider] = (),
        endpoint_adapters: (
            Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
        ) = None,
        runner_cfg: ExecutionRunnerCfg | None = None,
    ) -> SemanticSkillRuntime:
        """Bind an explicit integration to generic execution ports.

        Args:
            manifest: Provider-free scene, robot, and semantic-call declaration.
            scene_registry: Live registry matching ``manifest.scene``.
            engine: Atomic-action engine owning the target robot and planners.
            observation_provider: Source of fresh execution observations.
            command_sink: Destination for runtime commands and safe stops.
            clock: Optional scheduler clock shared by every per-call runner.
            effect_verifier: Optional default semantic effect verifier.
            registered_lowerers: Lowerers for registered extension calls.
            relation_grounders: Providers for late-bound relation targets.
            handover_pose_providers: Named handover-pose providers.
            endpoint_adapters: Optional adapters for custom resource endpoints.
            runner_cfg: Optional runner policy overriding per-skill presets.

        Returns:
            A validated runtime ready to analyze or execute semantic calls.

        Raises:
            TypeError: If an integration object or execution port is invalid.
            SemanticValidationError: If the live registry, engine, and manifest
                do not describe the same scene and robot capabilities.
        """
        if type(manifest) is not SemanticIntegrationManifest:
            raise TypeError("manifest must be exactly SemanticIntegrationManifest.")
        if not isinstance(scene_registry, SceneRegistry):
            raise TypeError("scene_registry must be a SceneRegistry.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        integration = manifest.bind(
            scene_registry,
            engine,
            endpoint_adapters=endpoint_adapters,
        )
        compiler = SemanticSkillCompiler(
            integration,
            registered_lowerers=registered_lowerers,
            relation_grounders=relation_grounders,
            handover_pose_providers=handover_pose_providers,
        )
        return cls(
            compiler,
            observation_provider,
            command_sink,
            clock=clock,
            effect_verifier=effect_verifier,
            runner_cfg=runner_cfg,
        )

    @classmethod
    def from_simulation(
        cls,
        *,
        simulation: SimulationManager,
        robot: Robot,
        motion_generator: MotionGenerator,
        scene_registry: SceneRegistry,
        robot_profile: RobotSkillProfile,
        call_catalog: SemanticCallCatalog | None = None,
        effect_verifier: SemanticEffectVerifier | None = None,
        registered_lowerers: Iterable[RegisteredSemanticLowerer] = (),
        relation_grounders: Iterable[RelationTargetGrounder] = (),
        handover_pose_providers: Iterable[HandOverPoseProvider] = (),
        endpoint_adapters: (
            Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
        ) = None,
        runner_cfg: ExecutionRunnerCfg | None = None,
        control_dt: float | None = None,
        scene_translation_threshold: float = 1.0e-4,
        scene_rotation_threshold: float = 1.0e-3,
    ) -> SemanticSkillRuntime:
        """Build the standard joint-position semantic runtime for simulation.

        This factory validates the registry against the motion generator,
        constructs a :class:`SimulationExecutionAdapter`, installs built-in
        atomic actions, and delegates the remaining binding to :meth:`bind`.

        Args:
            simulation: Simulation advanced by the runtime execution clock.
            robot: Robot observed and controlled through joint-position ports.
            motion_generator: Planner used by built-in atomic actions.
            scene_registry: Canonical live scene and affordance registry.
            robot_profile: Embodiment-specific skill and resource declaration.
            call_catalog: Optional semantic-call catalog. Built-ins are used by
                default.
            effect_verifier: Optional default semantic effect verifier.
            registered_lowerers: Lowerers for registered extension calls.
            relation_grounders: Providers for late-bound relation targets.
            handover_pose_providers: Named handover-pose providers.
            endpoint_adapters: Optional adapters for custom resource endpoints.
            runner_cfg: Optional runner policy overriding per-skill presets.
            control_dt: Optional semantic command period. The simulation physics
                period is used when omitted.
            scene_translation_threshold: Translation needed to advance the
                registry-backed scene version.
            scene_rotation_threshold: Rotation needed to advance the
                registry-backed scene version.

        Returns:
            A runtime using one simulation adapter for observation, commands,
            and deterministic simulated time.

        Raises:
            TypeError: If a registry, profile, catalog, or port is invalid.
            ValueError: If robot state or collision integration is inconsistent.
        """
        if not isinstance(scene_registry, SceneRegistry):
            raise TypeError("scene_registry must be a SceneRegistry.")
        if type(robot_profile) is not RobotSkillProfile:
            raise TypeError("robot_profile must be exactly RobotSkillProfile.")
        if call_catalog is not None and type(call_catalog) is not SemanticCallCatalog:
            raise TypeError("call_catalog must be exactly SemanticCallCatalog or None.")
        qpos = robot.get_qpos()
        if not isinstance(qpos, torch.Tensor) or qpos.dim() != 2:
            raise ValueError("robot.get_qpos() must return shape (B, robot_dof).")
        batch_size = int(qpos.shape[0])
        scene_provider = scene_registry.make_planning_scene_provider(
            motion_generator,
            batch_size=batch_size,
            translation_threshold=scene_translation_threshold,
            rotation_threshold=scene_rotation_threshold,
        )
        adapter = SimulationExecutionAdapter(
            simulation,
            robot,
            control_dt=control_dt,
            scene_provider=scene_provider,
        )
        engine = AtomicActionEngine(
            motion_generator,
            control_profiles=robot_profile.action_control_profiles(),
        )
        manifest = SemanticIntegrationManifest(
            scene=SceneManifest.from_registry(scene_registry),
            robot_profile=robot_profile,
            call_catalog=(
                builtin_semantic_call_catalog()
                if call_catalog is None
                else call_catalog
            ),
        )
        return cls.bind(
            manifest=manifest,
            scene_registry=scene_registry,
            engine=engine,
            observation_provider=adapter,
            command_sink=adapter,
            clock=adapter,
            effect_verifier=effect_verifier,
            registered_lowerers=registered_lowerers,
            relation_grounders=relation_grounders,
            handover_pose_providers=handover_pose_providers,
            endpoint_adapters=endpoint_adapters,
            runner_cfg=runner_cfg,
        )

    @property
    def engine(self) -> AtomicActionEngine:
        """Return the bound atomic-action engine."""
        return self.compiler.integration.engine

    @property
    def scene_registry(self) -> SceneRegistry:
        """Return the bound canonical scene registry."""
        return self.compiler.integration.scene_registry

    @property
    def available_calls(self) -> Mapping[str, SemanticCallDescriptor]:
        """Return semantic calls executable by the currently bound profile."""
        supported_skills = set(self.compiler.integration.robot_profile.skills)
        return MappingProxyType(
            {
                call_id: descriptor
                for call_id, descriptor in self.compiler.integration.manifest.call_catalog.descriptors.items()
                if descriptor.skill_id in supported_skills
            }
        )

    @property
    def active_task(self) -> SemanticTask | None:
        """Return the task currently owning this runtime."""
        return self._active_task

    def validate(
        self,
        calls: Iterable[SemanticCallSpec],
        *,
        workflow_id: str = "semantic_workflow",
    ) -> SemanticWorkflow:
        """Analyze one workflow without observing, planning, or executing it.

        Args:
            calls: Ordered robot-independent semantic calls.
            workflow_id: Stable identifier used for diagnostics and revisions.

        Returns:
            Provider-free linked workflow accepted by the current integration.
        """
        return self.compiler.analyze(calls, workflow_id=workflow_id)

    def open_task(
        self,
        task_id: str,
        *,
        initial_task_state: TaskState | None = None,
        eligible_mask: torch.Tensor | None = None,
    ) -> SemanticTask:
        """Open one exclusive task that may execute several workflow segments.

        Args:
            task_id: Stable task identifier without outer whitespace.
            initial_task_state: Optional previously verified symbolic state.
            eligible_mask: Optional initial per-environment execution cohort.

        Returns:
            A task retaining verified state across dynamic segment boundaries.

        Raises:
            RuntimeError: If another task already owns this runtime.
            ValueError: If the identifier or eligibility mask is invalid.
        """
        _validate_identifier(task_id, name="task_id")
        if self._active_task is not None:
            raise RuntimeError(
                f"Semantic task {self._active_task.task_id!r} already owns this runtime."
            )
        task = SemanticTask(
            self,
            task_id,
            initial_task_state=initial_task_state,
            eligible_mask=eligible_mask,
        )
        self._active_task = task
        return task

    def start(
        self,
        calls: Iterable[SemanticCallSpec],
        *,
        task_id: str = "semantic_task",
        segment_id: str = "main",
        initial_task_state: TaskState | None = None,
        eligible_mask: torch.Tensor | None = None,
    ) -> SemanticExecution:
        """Start a one-segment task and return a non-blocking execution handle.

        Args:
            calls: Ordered semantic calls analyzed before controller work starts.
            task_id: Stable identifier for the one-shot task.
            segment_id: Stable identifier for its only workflow segment.
            initial_task_state: Optional previously verified symbolic state.
            eligible_mask: Optional initial per-environment execution cohort.

        Returns:
            A handle advanced through :meth:`SemanticExecution.step` or
            :meth:`SemanticExecution.run_until_blocked`.
        """
        task = self.open_task(
            task_id,
            initial_task_state=initial_task_state,
            eligible_mask=eligible_mask,
        )
        try:
            return task._start_segment(
                calls,
                segment_id=segment_id,
                finish_task_on_completion=True,
            )
        except Exception:
            task.cancel("Semantic task could not be started.")
            raise

    def run(
        self,
        calls: Iterable[SemanticCallSpec],
        *,
        task_id: str = "semantic_task",
        segment_id: str = "main",
        initial_task_state: TaskState | None = None,
        eligible_mask: torch.Tensor | None = None,
        effect_verifier: SemanticEffectVerifier | None = None,
        on_step: RunnerStepCallback | None = None,
        max_steps_per_call: int = 100_000,
    ) -> SemanticTaskResult:
        """Run one semantic workflow to a terminal task result.

        Args:
            calls: Ordered semantic calls analyzed before execution.
            task_id: Stable identifier for the one-shot task.
            segment_id: Stable identifier for its only workflow segment.
            initial_task_state: Optional previously verified symbolic state.
            eligible_mask: Optional initial per-environment execution cohort.
            effect_verifier: Per-call effect verifier overriding the runtime
                default.
            on_step: Optional observer for each low-level runner step.
            max_steps_per_call: Hard loop bound applied separately to each call.

        Returns:
            Terminal result containing verified state, eligibility, and events.

        Raises:
            ValueError: If no effect verifier is available or a bound is invalid.
        """
        verifier = self.effect_verifier if effect_verifier is None else effect_verifier
        if verifier is None:
            raise ValueError(
                "run() requires an effect_verifier; use start() for manual "
                "effect submission."
            )
        execution = self.start(
            calls,
            task_id=task_id,
            segment_id=segment_id,
            initial_task_state=initial_task_state,
            eligible_mask=eligible_mask,
        )
        execution.run_until_blocked(
            effect_verifier=verifier,
            on_step=on_step,
            max_steps_per_call=max_steps_per_call,
        )
        result = execution.task_result
        if result is None:
            execution.cancel("Blocking semantic execution did not terminate.")
            result = execution.task_result
        assert result is not None
        return result

    def _release_task(self, task: SemanticTask) -> None:
        """Release task ownership after an exact active-task match."""
        if self._active_task is task:
            self._active_task = None


class SemanticTask:
    """Own verified state across one or more semantic workflow segments.

    Tasks are created by :meth:`SemanticSkillRuntime.open_task`. Successful
    segments leave the task open for a later application or agent decision.
    Failed or cancelled segments are terminal and release runtime ownership.

    Args:
        runtime: Runtime exclusively owned until this task terminates.
        task_id: Stable task identifier.
        initial_task_state: Optional externally verified symbolic state.
        eligible_mask: Optional initial per-environment execution cohort.
    """

    def __init__(
        self,
        runtime: SemanticSkillRuntime,
        task_id: str,
        *,
        initial_task_state: TaskState | None,
        eligible_mask: torch.Tensor | None,
    ) -> None:
        self.runtime = runtime
        self.task_id = _validate_identifier(task_id, name="task_id")
        if initial_task_state is not None and not isinstance(
            initial_task_state, TaskState
        ):
            raise TypeError("initial_task_state must be a TaskState or None.")
        self._task_state = (
            runtime.engine.initial_context().task
            if initial_task_state is None
            else initial_task_state
        )
        self._latest_context: PlanningContext | None = None
        self._latest_context = self._observe()
        self._initial_eligible_mask = _normalize_eligible_mask(
            eligible_mask,
            self._latest_context,
        )
        self._eligible_mask = self._initial_eligible_mask.clone()
        self._segments: list[SemanticSegmentResult] = []
        self._segment_ids: set[str] = set()
        self._active_execution: SemanticExecution | None = None
        self._failed = False
        self._cancelled = False
        self._message: str | None = None
        self._result: SemanticTaskResult | None = None

    def __enter__(self) -> SemanticTask:
        """Return this task for scoped dynamic execution."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Cancel unfinished work on scope exit and release runtime ownership."""
        del exc_type, traceback
        if self._result is not None:
            return
        if exc is not None or self._active_execution is not None or not self._segments:
            self.cancel(
                "Semantic task scope exited before normal completion."
                if exc is None
                else f"Semantic task scope exited with {type(exc).__name__}."
            )
        else:
            self.finish()

    @property
    def task_state(self) -> TaskState:
        """Return the latest externally verified symbolic state."""
        return self._task_state

    @property
    def latest_context(self) -> PlanningContext:
        """Return the latest observation carrying verified task state."""
        assert self._latest_context is not None
        return self._latest_context

    @property
    def eligible_mask(self) -> torch.Tensor:
        """Return environments still eligible to finish this task."""
        return self._eligible_mask.clone()

    @property
    def active_execution(self) -> SemanticExecution | None:
        """Return the currently active workflow segment, if any."""
        return self._active_execution

    @property
    def segments(self) -> tuple[SemanticSegmentResult, ...]:
        """Return terminal segment results in execution order."""
        return tuple(self._segments)

    @property
    def result(self) -> SemanticTaskResult | None:
        """Return the terminal task result, when finished."""
        return self._result

    def start_segment(
        self,
        calls: Iterable[SemanticCallSpec],
        *,
        segment_id: str | None = None,
    ) -> SemanticExecution:
        """Analyze and start one non-blocking workflow segment.

        Args:
            calls: Semantic calls known at the current decision boundary.
            segment_id: Optional stable segment identifier. A deterministic
                task-local identifier is generated when omitted.

        Returns:
            A non-blocking execution handle for this segment.

        Raises:
            RuntimeError: If the task is terminal or another segment is active.
            ValueError: If the segment identifier is invalid or already used.
        """
        return self._start_segment(
            calls,
            segment_id=segment_id,
            finish_task_on_completion=False,
        )

    def _start_segment(
        self,
        calls: Iterable[SemanticCallSpec],
        *,
        segment_id: str | None,
        finish_task_on_completion: bool,
    ) -> SemanticExecution:
        if self._result is not None:
            raise RuntimeError("A finished semantic task cannot start another segment.")
        if self._failed or self._cancelled:
            raise RuntimeError("A failed or cancelled task cannot start a segment.")
        if self._active_execution is not None:
            raise RuntimeError("Only one semantic segment may execute at a time.")
        selected_segment_id = (
            f"segment_{len(self._segments)}" if segment_id is None else segment_id
        )
        _validate_identifier(selected_segment_id, name="segment_id")
        if selected_segment_id in self._segment_ids:
            raise ValueError(f"Duplicate segment_id {selected_segment_id!r}.")
        workflow = self.runtime.compiler.analyze(
            calls,
            workflow_id=f"{self.task_id}.{selected_segment_id}",
        )
        execution = SemanticExecution(
            self,
            workflow,
            selected_segment_id,
            finish_task_on_completion=finish_task_on_completion,
        )
        self._segment_ids.add(selected_segment_id)
        self._active_execution = execution
        return execution

    def run_segment(
        self,
        calls: Iterable[SemanticCallSpec],
        *,
        segment_id: str | None = None,
        effect_verifier: SemanticEffectVerifier | None = None,
        on_step: RunnerStepCallback | None = None,
        max_steps_per_call: int = 100_000,
    ) -> SemanticSegmentResult:
        """Run one segment while retaining successful task state for more work.

        Args:
            calls: Semantic calls known at the current decision boundary.
            segment_id: Optional stable segment identifier.
            effect_verifier: Verifier overriding the runtime default.
            on_step: Optional observer for low-level runner steps.
            max_steps_per_call: Hard loop bound applied separately to each call.

        Returns:
            Terminal segment result. A successful task remains open; a failed
            or cancelled task closes automatically.

        Raises:
            ValueError: If no effect verifier is available or a bound is invalid.
            RuntimeError: If this task cannot start another segment.
        """
        verifier = (
            self.runtime.effect_verifier if effect_verifier is None else effect_verifier
        )
        if verifier is None:
            raise ValueError(
                "run_segment() requires an effect_verifier; use start_segment() "
                "for manual effect submission."
            )
        execution = self.start_segment(calls, segment_id=segment_id)
        execution.run_until_blocked(
            effect_verifier=verifier,
            on_step=on_step,
            max_steps_per_call=max_steps_per_call,
        )
        result = execution.segment_result
        if result is None:
            execution.cancel("Blocking semantic segment did not terminate.")
            result = execution.segment_result
        assert result is not None
        return result

    def cancel(
        self, reason: str = "Semantic task cancelled by caller."
    ) -> SemanticTaskResult:
        """Cancel active controller work and release runtime ownership.

        Args:
            reason: Non-empty cancellation diagnostic.

        Returns:
            Idempotent terminal task result after best-effort safe stop.
        """
        _validate_identifier(reason, name="reason")
        if self._result is not None:
            return self._result
        if self._active_execution is not None:
            self._active_execution.cancel(reason)
        else:
            self._cancelled = True
            self._message = reason
        return self.finish()

    def finish(self) -> SemanticTaskResult:
        """Finalize this task and release its exclusive runtime ownership.

        Returns:
            Idempotent terminal result derived from sticky eligibility and
            segment outcomes.

        Raises:
            RuntimeError: If a segment is active or no segment has run.
        """
        if self._result is not None:
            return self._result
        if self._active_execution is not None:
            raise RuntimeError("Cannot finish while a semantic segment is running.")
        if not self._segments and not self._cancelled and not self._failed:
            raise RuntimeError("Cannot finish a semantic task with no segments.")
        if self._cancelled:
            status = SemanticTaskStatus.CANCELLED
        elif self._failed or not self._eligible_mask.any():
            status = SemanticTaskStatus.FAILED
        else:
            initial = self._initial_eligible_mask
            retained = self._eligible_mask & initial
            status = (
                SemanticTaskStatus.SUCCEEDED
                if torch.equal(retained, initial)
                else SemanticTaskStatus.PARTIAL_SUCCESS
            )
        result = SemanticTaskResult(
            task_id=self.task_id,
            status=status,
            initial_eligible_mask=self._initial_eligible_mask,
            eligible_mask=self._eligible_mask,
            task_state=self._task_state,
            latest_context=self.latest_context,
            segments=tuple(self._segments),
            message=self._message,
        )
        self._result = result
        self.runtime._release_task(self)
        return result

    def _observe(self) -> PlanningContext:
        """Capture and validate a fresh context carrying verified task state."""
        observed = self.runtime.observation_provider.observe(self._task_state)
        if type(observed) is not PlanningContext:
            raise TypeError(
                "ObservationProvider.observe() must return PlanningContext."
            )
        context = PlanningContext(
            robot=observed.robot,
            task=self._task_state,
            scene=observed.scene,
            env_ids=observed.env_ids,
            control_dt=observed.control_dt,
        )
        self.runtime.engine._validate_context(context)
        previous = self._latest_context
        if previous is not None:
            _validate_context_progress(previous, context)
        self._latest_context = context
        return context

    def _adopt_runner(self, runner: ExecutionRunner) -> None:
        """Carry verified state and sticky eligibility across call boundaries."""
        session = runner.session
        self._task_state = session.task_state
        self._eligible_mask = session.eligible_mask
        self._latest_context = session.latest_context

    def _accept_segment(
        self,
        execution: SemanticExecution,
        result: SemanticSegmentResult,
    ) -> None:
        """Install one exact active segment result."""
        if self._active_execution is not execution:
            raise RuntimeError("Semantic segment no longer owns this task.")
        self._segments.append(result)
        self._active_execution = None
        if result.status is SemanticExecutionStatus.FAILED:
            self._failed = True
            self._message = result.message
        elif result.status is SemanticExecutionStatus.CANCELLED:
            self._cancelled = True
            self._message = result.message


class SemanticExecution:
    """Drive one analyzed workflow through one JIT-grounded call at a time.

    Instances are created by :meth:`SemanticSkillRuntime.start` or
    :meth:`SemanticTask.start_segment`; direct construction is not required.

    Args:
        task: Task retaining verified state and sticky eligibility.
        workflow: Statically analyzed semantic workflow.
        segment_id: Stable identifier of the owning segment.
        finish_task_on_completion: Whether a successful segment also finalizes
            its task. Failures and cancellations always finalize the task.
    """

    def __init__(
        self,
        task: SemanticTask,
        workflow: SemanticWorkflow,
        segment_id: str,
        *,
        finish_task_on_completion: bool,
    ) -> None:
        self.task = task
        self.workflow = workflow
        self.segment_id = segment_id
        self._finish_task_on_completion = finish_task_on_completion
        self._call_index = 0
        self._runner: ExecutionRunner | None = None
        self._grounded: GroundedSemanticCall | None = None
        self._current_events: list[ExecutionEvent] = []
        self._current_event_ids: set[int] = set()
        self._call_records: list[SemanticCallRecord] = []
        self._status = SemanticExecutionStatus.RUNNING
        self._message: str | None = None
        self._segment_result: SemanticSegmentResult | None = None
        self._task_result: SemanticTaskResult | None = None
        self._last_step: SemanticExecutionStep | None = None
        self._start_current_call()

    @property
    def status(self) -> SemanticExecutionStatus:
        """Return the current segment lifecycle status."""
        return self._status

    @property
    def call_index(self) -> int:
        """Return the currently active or last call index."""
        return self._call_index

    @property
    def segment_result(self) -> SemanticSegmentResult | None:
        """Return the terminal segment result, when available."""
        return self._segment_result

    @property
    def task_result(self) -> SemanticTaskResult | None:
        """Return the terminal task result for one-shot runtime execution."""
        return self._task_result

    @property
    def pending_effect(self) -> EffectVerificationRequest | None:
        """Return the effect currently awaiting external verification."""
        runner = self._runner
        if runner is None or not runner.effect_verification_pending:
            return None
        step = self._last_step
        return None if step is None else step.pending_effect

    def step(
        self,
        *,
        effect_success: torch.Tensor | None = None,
    ) -> SemanticExecutionStep:
        """Advance the active call without sleeping.

        Args:
            effect_success: Optional per-environment result for a currently
                pending effect request. Premature submissions are rejected.

        Returns:
            Latest semantic status and its underlying runner step.

        Raises:
            RuntimeError: If an effect result is submitted before the explicit
                verification boundary.
        """
        if self._segment_result is not None:
            assert self._last_step is not None
            return self._last_step
        assert self._runner is not None
        if effect_success is not None and not self._runner.effect_verification_pending:
            raise RuntimeError(
                "effect_success may only be submitted for pending effect "
                "verification."
            )
        runner_step = self._runner.step(effect_success=effect_success)
        self._record_runner_step(runner_step)
        return self._consume_runner_step(runner_step)

    def run_until_blocked(
        self,
        *,
        effect_verifier: SemanticEffectVerifier | None = None,
        on_step: RunnerStepCallback | None = None,
        max_steps_per_call: int = 100_000,
    ) -> SemanticExecutionStep:
        """Run until the segment terminates or external verification is needed.

        Args:
            effect_verifier: Optional callback used at every physical effect
                boundary. Without one, execution returns ``WAITING_FOR_EFFECT``.
            on_step: Optional observer for each low-level runner step.
            max_steps_per_call: Hard loop bound reset for every semantic call.

        Returns:
            Terminal segment step or an external-verification boundary.

        Raises:
            ValueError: If ``max_steps_per_call`` is not positive.
        """
        if max_steps_per_call <= 0:
            raise ValueError("max_steps_per_call must be greater than zero.")
        if self._segment_result is not None:
            assert self._last_step is not None
            return self._last_step
        while True:
            assert self._runner is not None
            runner = self._runner
            callback_step: RunnerStep | None = None

            def record_step(step: RunnerStep) -> None:
                nonlocal callback_step
                callback_step = step
                self._record_runner_step(step)
                if on_step is not None:
                    on_step(step)

            runner_step = runner.run_until_blocked(
                effect_verifier=(
                    None
                    if effect_verifier is None
                    else self._adapt_effect_verifier(effect_verifier)
                ),
                on_step=record_step,
                max_steps=max_steps_per_call,
            )
            if callback_step is not runner_step:
                self._record_runner_step(runner_step)
            result = self._consume_runner_step(runner_step)
            if result.status is not SemanticExecutionStatus.RUNNING:
                return result

    def revise_current(self, replacement: SemanticCallSpec) -> None:
        """Reanalyze and stage a compatible revision of the active call.

        The low-level runner still enforces the same semantic skill, logical
        invocation ID, and runtime endpoint addresses. This method is for a
        newer target or policy revision, not task-level skill replacement.

        Args:
            replacement: Replacement semantic call for the active workflow slot.

        Raises:
            TypeError: If ``replacement`` is not a semantic call.
            RuntimeError: If the segment is not running or awaits verification.
            ValueError: If the replacement violates compiler or runner revision
                invariants.
        """
        if not isinstance(replacement, SemanticCallSpec):
            raise TypeError("replacement must be a SemanticCallSpec.")
        if self._status is not SemanticExecutionStatus.RUNNING:
            raise RuntimeError("Only a running semantic call can be revised.")
        assert self._runner is not None and self._grounded is not None
        calls = [item.call for item in self.workflow.calls]
        calls[self._call_index] = replacement
        revised_workflow = self.task.runtime.compiler.analyze(
            calls,
            workflow_id=self.workflow.workflow_id,
        )
        context = self.task._observe()
        grounded = self.task.runtime.compiler.ground(
            revised_workflow,
            self._call_index,
            context,
            eligible_mask=self.task.eligible_mask,
            revision=self._grounded.invocation.revision + 1,
        )
        self._runner.revise_current(grounded.invocation)
        self.workflow = revised_workflow
        self._grounded = grounded

    def cancel(
        self,
        reason: str = "Semantic execution cancelled by caller.",
    ) -> SemanticExecutionStep:
        """Cancel the active low-level runner and finalize this segment.

        Args:
            reason: Non-empty cancellation diagnostic.

        Returns:
            Terminal semantic step after best-effort cancel and safe hold.
        """
        _validate_identifier(reason, name="reason")
        if self._segment_result is not None:
            assert self._last_step is not None
            return self._last_step
        assert self._runner is not None
        runner_step = self._runner.cancel(reason)
        self._record_runner_step(runner_step)
        return self._consume_runner_step(runner_step)

    def _start_current_call(self) -> None:
        """Observe, ground, plan, and install one semantic call runner."""
        context = self.task._observe()
        grounded = self.task.runtime.compiler.ground(
            self.workflow,
            self._call_index,
            context,
            eligible_mask=self.task.eligible_mask,
        )
        session = self.task.runtime.engine.start(
            (grounded.invocation,),
            context,
            eligible_mask=grounded.eligible_mask,
        )
        self._grounded = grounded
        self._runner = ExecutionRunner(
            session,
            self.task.runtime.observation_provider,
            self.task.runtime.command_sink,
            clock=self.task.runtime.clock,
            cfg=deepcopy(
                self._grounded.analyzed.bound.preset.runner_cfg
                if self.task.runtime.runner_cfg is None
                else self.task.runtime.runner_cfg
            ),
        )
        self._current_events = []
        self._current_event_ids = set()

    def _adapt_effect_verifier(
        self,
        verifier: SemanticEffectVerifier,
    ) -> Callable[[PlanningContext, ExecutionTick], torch.Tensor]:
        """Adapt the semantic verifier to the low-level runner callback."""

        def verify(context: PlanningContext, tick: ExecutionTick) -> torch.Tensor:
            pending = tick.pending_effect
            if not isinstance(pending, EffectVerificationRequest):
                raise RuntimeError("Effect verifier was called without a request.")
            call = self.workflow.calls[self._call_index].call
            result = verifier(call, pending, context)
            if not isinstance(result, torch.Tensor):
                raise TypeError("SemanticEffectVerifier must return a torch.Tensor.")
            return result

        return verify

    def _record_runner_step(self, step: RunnerStep) -> None:
        """Retain each structured event exactly once."""
        if step.tick is None:
            return
        for event in step.tick.events:
            event_id = id(event)
            if event_id in self._current_event_ids:
                continue
            self._current_event_ids.add(event_id)
            self._current_events.append(event)

    def _consume_runner_step(self, runner_step: RunnerStep) -> SemanticExecutionStep:
        """Advance the semantic call barrier from one low-level result."""
        assert self._runner is not None
        pending = None if runner_step.tick is None else runner_step.tick.pending_effect
        if runner_step.status is RunnerStatus.RUNNING:
            self._status = (
                SemanticExecutionStatus.WAITING_FOR_EFFECT
                if pending is not None
                else SemanticExecutionStatus.RUNNING
            )
            return self._make_step(runner_step, pending_effect=pending)

        self.task._adopt_runner(self._runner)
        self._record_call(runner_step)
        if runner_step.status is RunnerStatus.COMPLETED:
            if self._call_index + 1 < len(self.workflow.calls):
                self._call_index += 1
                try:
                    self._start_current_call()
                except Exception as exc:  # noqa: BLE001 - normalize call boundary
                    self._message = (
                        f"Could not start semantic call {self._call_index}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._finish_segment(SemanticExecutionStatus.FAILED)
                    return self._make_step(None, message=self._message)
                self._status = SemanticExecutionStatus.RUNNING
                return self._make_step(runner_step)
            self._finish_segment(SemanticExecutionStatus.COMPLETED)
            return self._make_step(runner_step)

        terminal_status = (
            SemanticExecutionStatus.CANCELLED
            if runner_step.status is RunnerStatus.CANCELLED
            else SemanticExecutionStatus.FAILED
        )
        self._message = runner_step.message
        self._finish_segment(terminal_status)
        return self._make_step(runner_step, message=self._message)

    def _record_call(self, runner_step: RunnerStep) -> None:
        """Snapshot the terminal state of the current grounded call."""
        assert self._runner is not None and self._grounded is not None
        invocation = self._grounded.invocation
        call = self.workflow.calls[self._call_index].call
        self._call_records.append(
            SemanticCallRecord(
                call_index=self._call_index,
                semantic_id=call.semantic_id,
                skill_id=invocation.skill_id,
                invocation_id=invocation.invocation_id,
                invocation_revision=invocation.revision,
                status=runner_step.status,
                eligible_mask=self.task.eligible_mask,
                events=tuple(self._current_events),
                command_count=self._runner.command_count,
                message=runner_step.message,
            )
        )

    def _finish_segment(self, status: SemanticExecutionStatus) -> None:
        """Install one terminal segment and optionally finalize its task."""
        self._status = status
        result = SemanticSegmentResult(
            segment_id=self.segment_id,
            workflow_id=self.workflow.workflow_id,
            status=status,
            eligible_mask=self.task.eligible_mask,
            task_state=self.task.task_state,
            calls=tuple(self._call_records),
            message=self._message,
        )
        self._segment_result = result
        self.task._accept_segment(self, result)
        if (
            self._finish_task_on_completion
            or status is not SemanticExecutionStatus.COMPLETED
        ):
            self._task_result = self.task.finish()

    def _make_step(
        self,
        runner_step: RunnerStep | None,
        *,
        pending_effect: EffectVerificationRequest | None = None,
        message: str | None = None,
    ) -> SemanticExecutionStep:
        """Build and retain the latest high-level execution step."""
        step = SemanticExecutionStep(
            status=self._status,
            task_id=self.task.task_id,
            segment_id=self.segment_id,
            call_index=self._call_index,
            eligible_mask=self.task.eligible_mask,
            runner_step=runner_step,
            pending_effect=pending_effect,
            message=message,
        )
        self._last_step = step
        return step


def _validate_identifier(value: str, *, name: str) -> str:
    """Return one exact non-empty identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without outer whitespace.")
    return value


def _normalize_eligible_mask(
    eligible_mask: torch.Tensor | None,
    context: PlanningContext,
) -> torch.Tensor:
    """Return one owned eligibility mask matching an observed context."""
    if eligible_mask is None:
        return torch.ones(
            context.batch_size,
            dtype=torch.bool,
            device=context.robot.qpos.device,
        )
    if not isinstance(eligible_mask, torch.Tensor):
        raise TypeError("eligible_mask must be a torch.Tensor or None.")
    if eligible_mask.dtype != torch.bool or eligible_mask.shape != (
        context.batch_size,
    ):
        raise ValueError("eligible_mask must be bool with one value per environment.")
    if eligible_mask.device != context.robot.qpos.device:
        raise ValueError("eligible_mask and the planning context must share a device.")
    return eligible_mask.clone()


def _validate_context_progress(
    previous: PlanningContext,
    current: PlanningContext,
) -> None:
    """Validate monotonic observations across semantic call sessions."""
    if not torch.equal(previous.env_ids, current.env_ids):
        raise ValueError("Semantic task env_ids must remain stable and ordered.")
    if current.robot.timestamp < previous.robot.timestamp:
        raise ValueError("Semantic task robot timestamps must be monotonic.")
    if current.scene.timestamp < previous.scene.timestamp:
        raise ValueError("Semantic task scene timestamps must be monotonic.")
    if current.scene.version < previous.scene.version:
        raise ValueError("Semantic task scene versions must be monotonic.")
    previous_revisions = previous.scene.collision_world_revisions(previous.batch_size)
    current_revisions = current.scene.collision_world_revisions(current.batch_size)
    if any(
        current_revision < previous_revision
        for previous_revision, current_revision in zip(
            previous_revisions,
            current_revisions,
            strict=True,
        )
    ):
        raise ValueError("Semantic task collision-world revisions must be monotonic.")


__all__ = [
    "SemanticCallRecord",
    "SemanticEffectVerifier",
    "SemanticExecution",
    "SemanticExecutionStatus",
    "SemanticExecutionStep",
    "SemanticSegmentResult",
    "SemanticSkillRuntime",
    "SemanticTask",
    "SemanticTaskResult",
    "SemanticTaskStatus",
]
