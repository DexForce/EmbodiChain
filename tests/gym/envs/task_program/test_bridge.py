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

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.envs.task_program.bridge import (
    TaskProgramDemoBridge,
    BufferedGymCommandSink,
    TaskProgramBridgeError,
    EnvironmentStepClock,
    EnvironmentStepTimingError,
    GymPlanningObservationProvider,
    RuntimeCommandFrameEncoder,
    UnsupportedRuntimeTransportError,
)
import embodichain.lab.gym.envs.task_program.bridge as bridge_module
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.sim.atomic_actions.bindings import (
    JointPositionTarget,
    RuntimeEndpointTarget,
)
from embodichain.lab.sim.atomic_actions.execution import (
    ExecutionEvent,
    ExecutionEventKind,
)
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy, RecoveryPolicy
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUpOptions
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceOptions
from embodichain.lab.sim.atomic_actions.runner import ExecutionRunnerCfg
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
    RuntimeCommandPayload,
)
from embodichain.lab.sim.atomic_actions.state import (
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.atomic_actions.tracking import TrackingPolicy
from embodichain.lab.task_program.semantics.calls import (
    Pick,
    Place,
    RegisteredSemanticCall,
)
from embodichain.lab.task_program.semantics.scene import SceneObjectRef
from embodichain.lab.task_program.runtime.parallel import ParallelTimingPolicy
from embodichain.lab.task_program.runtime.results import (
    SemanticExecutionResult,
    SemanticExecutionStatus,
)
from embodichain.lab.task_program.runtime.parallel_executor import (
    ParallelLaneCommandSink,
    ParallelExecutorBranch,
    ParallelSemanticExecutionResult,
    ParallelSemanticExecutor,
)
from embodichain.lab.task_program.semantics.profiles import (
    EffectAssurance,
    ResourceClaim,
    SkillPolicyPreset,
)

STEP_DT = 0.02
BATCH_SIZE = 2
ROBOT_DOF = 5
PROGRESS_SAMPLE_COUNT = 40


class _QposProvider:
    """Return an owned fixed full-qpos snapshot."""

    def __init__(self, qpos: torch.Tensor) -> None:
        self.qpos = qpos.clone()

    def current_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        assert env_ids.numel() == self.qpos.shape[0]
        return self.qpos.clone()


def _context(
    *,
    qpos: torch.Tensor | None = None,
    env_ids: torch.Tensor | None = None,
) -> PlanningContext:
    qpos = torch.zeros(BATCH_SIZE, ROBOT_DOF) if qpos is None else qpos
    env_ids = torch.tensor([7, 3], dtype=torch.long) if env_ids is None else env_ids
    return PlanningContext(
        robot=RobotObservation(
            timestamp=0.0,
            qpos=qpos,
            qvel=torch.zeros_like(qpos),
        ),
        task=TaskState.empty(qpos.shape[0], qpos.device),
        scene=SceneSnapshot.empty(),
        env_ids=env_ids,
    )


def _joint_frame(
    *,
    duration: float,
    active_mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> RuntimeCommandFrame:
    active_mask = torch.tensor([True, True]) if active_mask is None else active_mask
    positions = (
        torch.tensor([[10.0, 30.0], [11.0, 31.0]]) if positions is None else positions
    )
    return RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                target=JointPositionTarget(
                    control_part="arm",
                    joint_ids=(1, 3),
                ),
                payload=JointPositionPayload(positions=positions),
            ),
        ),
        active_mask=active_mask,
        env_ids=torch.tensor([7, 3], dtype=torch.long),
        hold_duration=torch.full((BATCH_SIZE,), duration),
    )


@dataclass(frozen=True, slots=True)
class _DummyTarget(RuntimeEndpointTarget):
    """Test-only non-joint runtime target."""

    name: str

    @property
    def transport_id(self) -> str:
        return "test.transport"

    @property
    def target_id(self) -> str:
        return self.name

    def snapshot(self) -> _DummyTarget:
        return _DummyTarget(self.name)


@dataclass(frozen=True, slots=True, eq=False)
class _DummyPayload(RuntimeCommandPayload):
    """Test-only scalar payload."""

    values: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", self.values.clone())

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def transport_id(self) -> str:
        return "test.transport"

    def snapshot(self) -> _DummyPayload:
        return _DummyPayload(self.values)


class _DummyTransportEncoder:
    """Test registration proving the frame encoder is transport-extensible."""

    transport_id = "test.transport"
    target_types = (_DummyTarget,)
    payload_types = (_DummyPayload,)

    def encode(
        self,
        command: EndpointCommand,
        *,
        base_action: Any,
        active_mask: torch.Tensor,
    ) -> Any:
        assert isinstance(command.payload, _DummyPayload)
        action = base_action.clone()
        action[active_mask, 0] = command.payload.values[active_mask]
        return action

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        base_action: Any,
        context: PlanningContext,
    ) -> Any:
        del targets, context
        return base_action.clone()


def _dummy_frame() -> RuntimeCommandFrame:
    return RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                target=_DummyTarget("base"),
                payload=_DummyPayload(torch.tensor([4.0, 5.0])),
            ),
        ),
        active_mask=torch.tensor([True, False]),
        env_ids=torch.tensor([7, 3], dtype=torch.long),
        hold_duration=torch.full((BATCH_SIZE,), STEP_DT),
    )


@dataclass(frozen=True, slots=True)
class _FakeCompiledCall:
    call_index: int
    call: object


@dataclass(frozen=True, slots=True)
class _FakeSegment:
    segment_index: int = 0
    segment_id: str = "segment-0"
    name: str = "pick-and-place"
    calls: tuple[_FakeCompiledCall, ...] = (
        _FakeCompiledCall(0, "pick"),
        _FakeCompiledCall(1, "place"),
    )
    source_path: tuple[object, ...] = ("program", "steps", 0)
    post_policies: tuple[object, ...] = ()
    validators: tuple[object, ...] = ()
    parallel_block: object | None = None
    implicit: bool = False


@dataclass(frozen=True, slots=True)
class _FakeParallelBranch:
    branch_index: int
    calls: tuple[_FakeCompiledCall, ...]


@dataclass(frozen=True, slots=True)
class _FakeBarrier:
    timeout_steps: int = 17
    failure_policy: str = "fail_fast"


@dataclass(frozen=True, slots=True)
class _FakeParallelBlock:
    branches: tuple[_FakeParallelBranch, ...]
    barrier: _FakeBarrier = _FakeBarrier()


@dataclass(frozen=True, slots=True)
class _FakeProgramAnalysis:
    calls: tuple[object, ...]
    execution_prefix_length: int


class _FakeProgram:
    program_id = "demo-program"

    def __init__(self, *segments: _FakeSegment) -> None:
        self.segments = segments

    def iter_segments(self):
        yield from self.segments

    def sequential_execution_analysis(
        self,
        segment_index: int,
    ) -> _FakeProgramAnalysis:
        current = self.segments[segment_index]
        if current.parallel_block is not None:
            raise ValueError("Parallel segments have no sequential analysis.")
        calls: list[object] = []
        for segment in self.segments[segment_index:]:
            if segment.parallel_block is not None:
                break
            calls.extend(compiled.call for compiled in segment.calls)
        return _FakeProgramAnalysis(tuple(calls), len(current.calls))


def _skill_result(
    status: SemanticExecutionStatus,
    *,
    wait_duration: float = 0.0,
    workflow_id: str = "demo-program/segment-0",
) -> SemanticExecutionResult:
    env_ids = torch.tensor([7, 3], dtype=torch.long)
    eligible = torch.ones(BATCH_SIZE, dtype=torch.bool)
    success = (
        torch.ones(BATCH_SIZE, dtype=torch.bool)
        if status is SemanticExecutionStatus.COMPLETED
        else torch.zeros(BATCH_SIZE, dtype=torch.bool)
    )
    failure = (
        torch.ones(BATCH_SIZE, dtype=torch.bool)
        if status is SemanticExecutionStatus.FAILED
        else torch.zeros(BATCH_SIZE, dtype=torch.bool)
    )
    if status is SemanticExecutionStatus.FAILED:
        eligible = torch.zeros_like(eligible)
    return SemanticExecutionResult(
        status=status,
        workflow_id=workflow_id,
        current_call_index=0 if status is SemanticExecutionStatus.RUNNING else None,
        env_ids=env_ids,
        success_mask=success,
        failure_mask=failure,
        cancelled_mask=torch.zeros(BATCH_SIZE, dtype=torch.bool),
        eligible_mask=eligible,
        task_state=TaskState.empty(BATCH_SIZE, "cpu"),
        wait_duration=wait_duration,
    )


class _FakeRuntime:
    """Clock-aware nonblocking runtime used to test the Gym boundary only."""

    def __init__(
        self,
        sink: BufferedGymCommandSink,
        clock: EnvironmentStepClock,
        frame: RuntimeCommandFrame,
    ) -> None:
        self.sink = sink
        self.clock = clock
        self.frame = frame
        self._status = SemanticExecutionStatus.IDLE
        self._result = _skill_result(SemanticExecutionStatus.IDLE)
        self._due_at = 0.0
        self._sent = False
        self.start_count = 0
        self.step_count = 0
        self.cancel_count = 0
        self.calls: tuple[object, ...] = ()
        self.execution_prefix_lengths: list[int | None] = []
        self.eligible_masks: list[torch.Tensor | None] = []
        self.adopted_states: list[TaskState] = []

    @property
    def result(self) -> SemanticExecutionResult:
        return self._result

    @property
    def status(self) -> SemanticExecutionStatus:
        return self._status

    def start(
        self,
        *calls: object,
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
    ) -> SemanticExecutionResult:
        self.start_count += 1
        self.calls = tuple(calls[0]) if len(calls) == 1 else calls
        self.execution_prefix_lengths.append(execution_prefix_length)
        self.eligible_masks.append(
            None if eligible_mask is None else eligible_mask.clone()
        )
        self._sent = False
        self._due_at = 0.0
        self._status = SemanticExecutionStatus.RUNNING
        self._result = _skill_result(
            SemanticExecutionStatus.RUNNING,
            workflow_id=workflow_id,
        )
        return self._result

    def adopt_verified_task_state(
        self, task_state: TaskState
    ) -> SemanticExecutionResult:
        self.adopted_states.append(task_state)
        return self._result

    def step(self) -> SemanticExecutionResult:
        self.step_count += 1
        if not self._sent:
            self.sink.send(self.frame, timeout=1.0)
            self._sent = True
            self._due_at = self.clock.now() + float(
                self.frame.hold_duration.max().item()
            )
        remaining = max(self._due_at - self.clock.now(), 0.0)
        if remaining > 1.0e-9:
            self._result = _skill_result(
                SemanticExecutionStatus.RUNNING,
                wait_duration=remaining,
                workflow_id=self._result.workflow_id or "semantic_workflow",
            )
            return self._result
        self._status = SemanticExecutionStatus.COMPLETED
        self._result = _skill_result(
            SemanticExecutionStatus.COMPLETED,
            workflow_id=self._result.workflow_id or "semantic_workflow",
        )
        return self._result

    def cancel(self, reason: str) -> SemanticExecutionResult:
        del reason
        self.cancel_count += 1
        self.sink.cancel(self.frame.targets, timeout=1.0)
        self.sink.hold(self.frame.targets, _context(), timeout=1.0)
        self._status = SemanticExecutionStatus.CANCELLED
        self._result = SemanticExecutionResult(
            status=SemanticExecutionStatus.CANCELLED,
            workflow_id=self._result.workflow_id,
            current_call_index=None,
            env_ids=torch.tensor([7, 3], dtype=torch.long),
            success_mask=torch.zeros(BATCH_SIZE, dtype=torch.bool),
            failure_mask=torch.zeros(BATCH_SIZE, dtype=torch.bool),
            cancelled_mask=torch.ones(BATCH_SIZE, dtype=torch.bool),
            eligible_mask=torch.zeros(BATCH_SIZE, dtype=torch.bool),
            task_state=TaskState.empty(BATCH_SIZE, "cpu"),
        )
        return self._result


class _StartFailingRuntime(_FakeRuntime):
    """Fail semantic preflight before accepting any controller command."""

    def start(
        self,
        *calls: object,
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
    ) -> SemanticExecutionResult:
        del calls, workflow_id, eligible_mask, execution_prefix_length
        self.start_count += 1
        raise RuntimeError("semantic runtime preflight failed")


class _TerminalHoldRuntime(_FakeRuntime):
    """Emit a terminal safe hold after one consumed command."""

    def step(self) -> SemanticExecutionResult:
        self.step_count += 1
        if not self._sent:
            self.sink.send(self.frame, timeout=1.0)
            self._sent = True
            self._status = SemanticExecutionStatus.RUNNING
            self._result = _skill_result(
                SemanticExecutionStatus.RUNNING,
                workflow_id=self._result.workflow_id or "semantic_workflow",
            )
            return self._result
        self.sink.hold(self.frame.targets, _context(), timeout=1.0)
        self._status = SemanticExecutionStatus.COMPLETED
        self._result = _skill_result(
            SemanticExecutionStatus.COMPLETED,
            workflow_id=self._result.workflow_id or "semantic_workflow",
        )
        return self._result


class _TerminalFailedRuntime(_FakeRuntime):
    """Fail terminally during planning without accepting a command."""

    def start(
        self,
        *calls: object,
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
    ) -> SemanticExecutionResult:
        running = super().start(
            *calls,
            workflow_id=workflow_id,
            eligible_mask=eligible_mask,
            execution_prefix_length=execution_prefix_length,
        )
        failed_mask = torch.ones(BATCH_SIZE, dtype=torch.bool)
        self._status = SemanticExecutionStatus.FAILED
        self._result = SemanticExecutionResult(
            status=SemanticExecutionStatus.FAILED,
            workflow_id=running.workflow_id,
            current_call_index=None,
            env_ids=running.env_ids,
            success_mask=torch.zeros_like(failed_mask),
            failure_mask=failed_mask,
            cancelled_mask=torch.zeros_like(failed_mask),
            eligible_mask=torch.zeros_like(failed_mask),
            task_state=running.task_state,
            events=(
                ExecutionEvent(
                    kind=ExecutionEventKind.ACTION_PLANNING_FAILED,
                    timestamp=self.clock.now(),
                    skill_id="slide",
                    invocation_id="open-drawer-call",
                    invocation_revision=0,
                    invocation_index=0,
                    env_mask=failed_mask,
                    message="Slide contact motion failed.",
                ),
            ),
            message="Motion planning failed before the first command.",
        )
        return self._result


class _PartialSuccessRuntime(_FakeRuntime):
    """Complete the workflow while retaining one failed environment row."""

    def step(self) -> SemanticExecutionResult:
        result = super().step()
        if result.status is SemanticExecutionStatus.COMPLETED:
            active_mask = torch.tensor([True, False])
            self._result = SemanticExecutionResult(
                status=SemanticExecutionStatus.COMPLETED,
                workflow_id=result.workflow_id,
                current_call_index=None,
                env_ids=result.env_ids,
                success_mask=active_mask,
                failure_mask=~active_mask,
                cancelled_mask=torch.zeros_like(active_mask),
                eligible_mask=active_mask,
                task_state=result.task_state,
            )
        return self._result


def _parallel_result(
    status: SemanticExecutionStatus,
    *,
    wait_duration: float = 0.0,
) -> ParallelSemanticExecutionResult:
    env_ids = torch.tensor([7, 3], dtype=torch.long)
    terminal = status in {
        SemanticExecutionStatus.COMPLETED,
        SemanticExecutionStatus.FAILED,
        SemanticExecutionStatus.CANCELLED,
    }
    return ParallelSemanticExecutionResult(
        status=status,
        env_ids=env_ids,
        success_mask=(
            torch.ones(BATCH_SIZE, dtype=torch.bool)
            if status is SemanticExecutionStatus.COMPLETED
            else torch.zeros(BATCH_SIZE, dtype=torch.bool)
        ),
        failure_mask=(
            torch.ones(BATCH_SIZE, dtype=torch.bool)
            if status is SemanticExecutionStatus.FAILED
            else torch.zeros(BATCH_SIZE, dtype=torch.bool)
        ),
        cancelled_mask=(
            torch.ones(BATCH_SIZE, dtype=torch.bool)
            if status is SemanticExecutionStatus.CANCELLED
            else torch.zeros(BATCH_SIZE, dtype=torch.bool)
        ),
        pending_mask=(
            torch.zeros(BATCH_SIZE, dtype=torch.bool)
            if terminal
            else torch.ones(BATCH_SIZE, dtype=torch.bool)
        ),
        task_state=TaskState.empty(BATCH_SIZE, "cpu"),
        branch_results={},
        elapsed_steps=0,
        command_count=0,
        wait_duration=wait_duration,
    )


class _FakeParallelRuntime:
    """One-grid-frame parallel coordinator used at the bridge boundary."""

    def __init__(
        self,
        sink: BufferedGymCommandSink,
        clock: EnvironmentStepClock,
    ) -> None:
        self.sink = sink
        self.clock = clock
        self._result = _parallel_result(SemanticExecutionStatus.IDLE)
        self._sent = False
        self._due_at = 0.0
        self.eligible_mask: torch.Tensor | None = None

    @property
    def result(self) -> ParallelSemanticExecutionResult:
        return self._result

    def start(
        self,
        *,
        workflow_id: str = "parallel_workflow",
        eligible_mask: torch.Tensor | None = None,
    ) -> ParallelSemanticExecutionResult:
        del workflow_id
        self.eligible_mask = None if eligible_mask is None else eligible_mask.clone()
        self._result = _parallel_result(SemanticExecutionStatus.RUNNING)
        return self._result

    def step(self) -> ParallelSemanticExecutionResult:
        if not self._sent:
            self.sink.send(_joint_frame(duration=STEP_DT), timeout=1.0)
            self._sent = True
            self._due_at = self.clock.now() + STEP_DT
        remaining = max(self._due_at - self.clock.now(), 0.0)
        self._result = (
            _parallel_result(SemanticExecutionStatus.RUNNING, wait_duration=remaining)
            if remaining > 1.0e-9
            else _parallel_result(SemanticExecutionStatus.COMPLETED)
        )
        return self._result

    def cancel(self, reason: str) -> ParallelSemanticExecutionResult:
        del reason
        self._result = _parallel_result(SemanticExecutionStatus.CANCELLED)
        return self._result


class _GridLaneRuntime:
    """Small branch runtime used with the real parallel coordinator and sink."""

    def __init__(
        self,
        sink: ParallelLaneCommandSink,
        script: tuple[tuple[SemanticExecutionStatus, RuntimeCommandFrame | None], ...],
    ) -> None:
        self.sink = sink
        self.script = script
        self.step_count = 0
        self._result = _skill_result(SemanticExecutionStatus.IDLE)

    @property
    def result(self) -> SemanticExecutionResult:
        return self._result

    def start(
        self,
        *calls: object,
        workflow_id: str,
        eligible_mask: torch.Tensor | None = None,
    ) -> SemanticExecutionResult:
        del calls, eligible_mask
        self._result = _skill_result(
            SemanticExecutionStatus.RUNNING, workflow_id=workflow_id
        )
        return self._result

    def step(self) -> SemanticExecutionResult:
        status, frame = self.script[min(self.step_count, len(self.script) - 1)]
        self.step_count += 1
        if frame is not None:
            self.sink.send(frame, timeout=1.0)
        if status is not SemanticExecutionStatus.RUNNING:
            last_frame = frame or self.sink.last_frame
            assert last_frame is not None
            self.sink.hold(last_frame.targets, _context(), timeout=1.0)
        self._result = _skill_result(
            status, workflow_id=self._result.workflow_id or "lane"
        )
        return self._result

    def deactivate_rows(
        self,
        env_mask: torch.Tensor,
        *,
        reason: str,
    ) -> SemanticExecutionResult:
        del env_mask, reason
        return self._result

    def cancel(self, reason: str) -> SemanticExecutionResult:
        del reason
        last_frame = self.sink.last_frame
        if last_frame is not None:
            self.sink.cancel(last_frame.targets, timeout=1.0)
            self.sink.hold(last_frame.targets, _context(), timeout=1.0)
        self._result = SemanticExecutionResult(
            status=SemanticExecutionStatus.CANCELLED,
            workflow_id=self._result.workflow_id,
            current_call_index=None,
            env_ids=torch.tensor([7, 3], dtype=torch.long),
            success_mask=torch.zeros(BATCH_SIZE, dtype=torch.bool),
            failure_mask=torch.zeros(BATCH_SIZE, dtype=torch.bool),
            cancelled_mask=torch.ones(BATCH_SIZE, dtype=torch.bool),
            eligible_mask=torch.zeros(BATCH_SIZE, dtype=torch.bool),
            task_state=TaskState.empty(BATCH_SIZE, "cpu"),
        )
        return self._result


def _grid_frame(
    control_part: str,
    joint_id: int,
    value: float,
) -> RuntimeCommandFrame:
    return RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                JointPositionTarget(control_part, (joint_id,)),
                JointPositionPayload(
                    torch.full((BATCH_SIZE, 1), value, dtype=torch.float32)
                ),
            ),
        ),
        active_mask=torch.ones(BATCH_SIZE, dtype=torch.bool),
        env_ids=torch.tensor([7, 3], dtype=torch.long),
        hold_duration=torch.full((BATCH_SIZE,), STEP_DT),
    )


class _PostPolicyPort:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action
        self.seen: list[object] = []
        self.active_masks: list[torch.Tensor] = []

    def validate_policy(self, policy: object, *, segment: object) -> None:
        del policy, segment

    def actions(
        self,
        policy: object,
        *,
        segment: object,
        active_mask: torch.Tensor,
    ):
        del segment
        self.seen.append(policy)
        self.active_masks.append(active_mask.clone())
        yield self.action

    def post_policy_result(
        self,
        policy: object,
        *,
        segment: object,
    ) -> torch.Tensor:
        del policy, segment
        return self.active_masks[-1].clone()

    def post_policy_metadata(
        self,
        policy: object,
        *,
        segment: object,
    ) -> dict[str, object]:
        del policy, segment
        return {}


class _ValidatorPort:
    def __init__(self, result: torch.Tensor) -> None:
        self.result = result
        self.seen: list[object] = []

    def validate_validator(self, validator: object, *, segment: object) -> None:
        del validator, segment

    def validate(self, validator: object, *, segment: object) -> torch.Tensor:
        del segment
        self.seen.append(validator)
        return self.result

    def validator_metadata(
        self,
        validator: object,
        *,
        segment: object,
    ) -> dict[str, object]:
        del validator, segment
        return {}


class _MetadataPostPolicyPort(_PostPolicyPort):
    """Post-policy test port exposing a deterministic result trace."""

    def post_policy_metadata(
        self,
        policy: object,
        *,
        segment: object,
    ) -> dict[str, object]:
        del policy, segment
        return {
            "status": "timed_out",
            "state": {
                "elapsed_steps": 1,
                "settled_mask": [True, False],
                "timeout_mask": [False, True],
            },
        }

    def post_policy_result(
        self,
        policy: object,
        *,
        segment: object,
    ) -> torch.Tensor:
        del policy, segment
        return torch.tensor([True, False])


class _MetadataValidatorPort(_ValidatorPort):
    """Validator test port exposing observed error metadata."""

    def validator_metadata(
        self,
        validator: object,
        *,
        segment: object,
    ) -> dict[str, object]:
        del validator, segment
        return {"position_error": [0.01, 0.10]}


class _FailingPostPolicyPort:
    """Raise from lazy policy iteration after the runtime reached a safe hold."""

    def validate_policy(self, policy: object, *, segment: object) -> None:
        del policy, segment

    def actions(
        self,
        policy: object,
        *,
        segment: object,
        active_mask: torch.Tensor,
    ):
        del policy, segment, active_mask
        if False:
            yield torch.empty(0)
        raise RuntimeError("post-policy observation failed")

    def post_policy_result(
        self,
        policy: object,
        *,
        segment: object,
    ) -> bool:
        del policy, segment
        return True

    def post_policy_metadata(
        self,
        policy: object,
        *,
        segment: object,
    ) -> dict[str, object]:
        del policy, segment
        return {}


class _AcceptParallelSafety:
    """Test-only authoritative gate that accepts the supplied merged frame."""

    def validate(
        self,
        *,
        branch_frames: dict[str, RuntimeCommandFrame],
        merged_frame: RuntimeCommandFrame,
    ) -> None:
        assert branch_frames
        assert isinstance(merged_frame, RuntimeCommandFrame)


def _bridge(
    *,
    duration: float,
    segment: _FakeSegment | None = None,
    post_policy_port: object | None = None,
    validator_port: object | None = None,
    parallel_safety_validator: object | None = None,
    runner_cfg: ExecutionRunnerCfg | None = None,
) -> tuple[TaskProgramDemoBridge, _FakeRuntime, EnvironmentStepClock]:
    clock = EnvironmentStepClock(STEP_DT)
    encoder = RuntimeCommandFrameEncoder(
        _QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))
    )
    sink = BufferedGymCommandSink(encoder, clock)
    runtime = _FakeRuntime(sink, clock, _joint_frame(duration=duration))
    bridge = TaskProgramDemoBridge(
        _FakeProgram(_FakeSegment() if segment is None else segment),
        runtime,
        sink,
        clock,
        post_policy_port=post_policy_port,
        validator_port=validator_port,
        runner_cfg=runner_cfg,
        parallel_safety_validator=parallel_safety_validator,
    )
    return bridge, runtime, clock


@pytest.mark.parametrize(
    "overrides, expected_total",
    [
        ({}, 2 * PROGRESS_SAMPLE_COUNT),
        ({"motion_policy": MotionPolicy(strategy="motion_gen")}, None),
        ({"recovery_policy": RecoveryPolicy()}, None),
        ({"tracking_policy": TrackingPolicy.joint_position()}, None),
        ({"tracking_policy": TrackingPolicy.timed(settle_duration=STEP_DT)}, None),
    ],
)
def test_bridge_progress_counts_only_fixed_pick_place_calls(
    overrides, expected_total
) -> None:
    """Only fixed paths expose a total, without starting or analyzing a workflow."""
    cube = SceneObjectRef("cube")
    calls = (Pick(cube), Place(cube, on=SceneObjectRef("table")))
    bridge, runtime, _ = _bridge(
        duration=STEP_DT,
        segment=_FakeSegment(
            calls=tuple(_FakeCompiledCall(i, call) for i, call in enumerate(calls))
        ),
        runner_cfg=ExecutionRunnerCfg(
            minimum_cycle_time=0.0,
            hold_on_completion=False,
            hold_during_effect_verification=False,
        ),
    )
    policies = {
        "motion_policy": MotionPolicy(sample_count=PROGRESS_SAMPLE_COUNT),
        "recovery_policy": RecoveryPolicy(max_replans=0, max_action_retries=0),
        "tracking_policy": TrackingPolicy.timed(),
        **overrides,
    }
    compiler = Mock()
    compiler.integration.link_call.return_value.preset = SkillPolicyPreset(
        "progress",
        effect_assurance=EffectAssurance.PROJECTED,
        action_option_templates={"pick": PickUpOptions(), "place": PlaceOptions()},
        **policies,
    )
    runtime.compiler = compiler

    demo_segment = next(bridge.iter_segments())

    assert demo_segment.progress_total_steps == expected_total
    compiler.analyze.assert_not_called()
    assert runtime.start_count == 0


def test_bridge_snapshots_runner_cfg_before_lazy_parallel_creation() -> None:
    """Later advanced-path config mutation cannot change lazy bridge policy."""
    runner_cfg = ExecutionRunnerCfg(command_timeout=0.25)
    bridge, _, _ = _bridge(duration=STEP_DT, runner_cfg=runner_cfg)

    runner_cfg.command_timeout = 9.0

    assert bridge._runner_cfg is not runner_cfg
    assert bridge._runner_cfg.command_timeout == pytest.approx(0.25)


def test_environment_step_clock_advances_only_explicitly() -> None:
    clock = EnvironmentStepClock(STEP_DT)

    assert clock.now() == 0.0
    assert clock.steps_for_duration(3 * STEP_DT) == 3
    with pytest.raises(EnvironmentStepTimingError, match="not an integer multiple"):
        clock.steps_for_duration(0.03)
    with pytest.raises(RuntimeError, match="cannot sleep"):
        clock.sleep(STEP_DT)
    assert clock.now() == 0.0

    clock.advance_after_env_step()
    assert clock.step_index == 1
    assert clock.now() == pytest.approx(STEP_DT)


def test_observation_provider_reorders_qpos_by_stable_env_id() -> None:
    context = _context(
        qpos=torch.tensor([[7.0, 7.1, 7.2, 7.3, 7.4], [3.0, 3.1, 3.2, 3.3, 3.4]])
    )
    provider = GymPlanningObservationProvider(lambda task_state: context)

    observed = provider.observe(context.task)
    reordered = provider.current_qpos(torch.tensor([3, 7], dtype=torch.long))

    assert observed is context
    assert torch.equal(reordered[0], context.robot.qpos[1])
    assert torch.equal(reordered[1], context.robot.qpos[0])


def test_joint_encoder_emits_full_qpos_and_holds_inactive_rows() -> None:
    qpos = torch.arange(BATCH_SIZE * ROBOT_DOF, dtype=torch.float32).reshape(
        BATCH_SIZE, ROBOT_DOF
    )
    encoder = RuntimeCommandFrameEncoder(_QposProvider(qpos))
    frame = _joint_frame(
        duration=STEP_DT,
        active_mask=torch.tensor([True, False]),
    )

    action = encoder.encode(frame)

    assert isinstance(action, torch.Tensor)
    assert action.shape == qpos.shape
    assert torch.equal(action[0, torch.tensor([1, 3])], torch.tensor([10.0, 30.0]))
    assert torch.equal(action[0, torch.tensor([0, 2, 4])], qpos[0, [0, 2, 4]])
    assert torch.equal(action[1], qpos[1])


def test_frame_encoder_supports_registered_future_transport() -> None:
    encoder = RuntimeCommandFrameEncoder(
        _QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))
    )
    frame = _dummy_frame()
    with pytest.raises(UnsupportedRuntimeTransportError, match="test.transport"):
        encoder.encode(frame)

    encoder.register_transport(_DummyTransportEncoder())
    action = encoder.encode(frame)

    assert isinstance(action, torch.Tensor)
    assert action[0, 0].item() == 4.0
    assert action[1, 0].item() == 0.0


def test_frame_encoder_composes_in_registration_not_frame_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport registration is the stable controller composition order."""
    calls: list[str] = []
    original_joint_encode = bridge_module.JointPositionGymTransportEncoder.encode
    original_dummy_encode = _DummyTransportEncoder.encode

    def record_joint(self: object, *args: object, **kwargs: object) -> object:
        calls.append("joint")
        return original_joint_encode(self, *args, **kwargs)

    def record_dummy(self: object, *args: object, **kwargs: object) -> object:
        calls.append("dummy")
        return original_dummy_encode(self, *args, **kwargs)

    monkeypatch.setattr(
        bridge_module.JointPositionGymTransportEncoder,
        "encode",
        record_joint,
    )
    monkeypatch.setattr(_DummyTransportEncoder, "encode", record_dummy)
    joint = _joint_frame(duration=STEP_DT).commands[0]
    dummy = _dummy_frame().commands[0]
    frame = RuntimeCommandFrame(
        commands=(dummy, joint),
        active_mask=torch.tensor([True, False]),
        env_ids=torch.tensor([7, 3], dtype=torch.long),
        hold_duration=torch.full((BATCH_SIZE,), STEP_DT),
    )
    encoder = RuntimeCommandFrameEncoder(
        _QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF)),
        transports=(_DummyTransportEncoder(),),
    )

    encoder.encode(frame)

    assert calls == ["joint", "dummy"]


def test_hold_encoder_composes_in_registration_not_target_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safe-hold transport composition uses the same registered ordering."""
    calls: list[str] = []
    original_joint_hold = bridge_module.JointPositionGymTransportEncoder.hold
    original_dummy_hold = _DummyTransportEncoder.hold

    def record_joint(self: object, *args: object, **kwargs: object) -> object:
        calls.append("joint")
        return original_joint_hold(self, *args, **kwargs)

    def record_dummy(self: object, *args: object, **kwargs: object) -> object:
        calls.append("dummy")
        return original_dummy_hold(self, *args, **kwargs)

    monkeypatch.setattr(
        bridge_module.JointPositionGymTransportEncoder,
        "hold",
        record_joint,
    )
    monkeypatch.setattr(_DummyTransportEncoder, "hold", record_dummy)
    encoder = RuntimeCommandFrameEncoder(
        _QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF)),
        transports=(_DummyTransportEncoder(),),
    )
    dummy_target = _dummy_frame().targets[0]
    joint_target = _joint_frame(duration=STEP_DT).targets[0]

    encoder.encode_hold((dummy_target, joint_target), _context())

    assert calls == ["joint", "dummy"]


def test_frame_encoder_rejects_transport_without_static_type_declarations() -> None:
    """Every runtime transport declares its exact pre-sim routing surface."""

    class MissingDeclarations:
        transport_id = "test.missing"

        def encode(self, *args: object, **kwargs: object) -> object:
            raise AssertionError

        def hold(self, *args: object, **kwargs: object) -> object:
            raise AssertionError

    encoder = RuntimeCommandFrameEncoder(
        _QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))
    )

    with pytest.raises(TypeError, match="RuntimeTransportActionEncoder"):
        encoder.register_transport(MissingDeclarations())  # type: ignore[arg-type]


def test_frame_encoder_requires_exact_declared_target_coverage() -> None:
    """Transport routing never widens a declaration through subclass checks."""

    class WrongTargetCoverage(_DummyTransportEncoder):
        target_types = (JointPositionTarget,)

    encoder = RuntimeCommandFrameEncoder(
        _QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF)),
        transports=(WrongTargetCoverage(),),
    )

    with pytest.raises(TypeError, match="does not declare exact target type"):
        encoder.encode(_dummy_frame())

    with pytest.raises(TypeError, match="does not declare exact hold target type"):
        encoder.encode_hold(_dummy_frame().targets, _context())


def test_frame_encoder_requires_exact_declared_payload_coverage() -> None:
    """Payload declarations are enforced independently of target coverage."""

    class WrongPayloadCoverage(_DummyTransportEncoder):
        payload_types = (JointPositionPayload,)

    encoder = RuntimeCommandFrameEncoder(
        _QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF)),
        transports=(WrongPayloadCoverage(),),
    )

    with pytest.raises(TypeError, match="does not declare exact payload type"):
        encoder.encode(_dummy_frame())


def test_buffered_sink_rejects_off_grid_frame_before_buffering() -> None:
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )

    with pytest.raises(EnvironmentStepTimingError, match="hold_duration"):
        sink.send(_joint_frame(duration=0.03), timeout=1.0)

    assert sink.pending_count == 0
    assert clock.step_index == 0


def test_buffered_sink_buffers_command_hold_and_cancel_without_stepping() -> None:
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    frame = _joint_frame(duration=STEP_DT)

    acknowledgement = sink.send(frame, timeout=1.0)
    assert acknowledgement.accepted
    assert sink.pending_count == 1
    action = sink.pop()
    assert isinstance(action, ControllerAction)
    assert action.metadata["bridge_action_kind"] == "runtime_command"
    assert clock.step_index == 0

    sink.hold(frame.targets, _context(), timeout=1.0)
    assert sink.pending_count == 1
    sink.cancel(frame.targets, timeout=1.0)
    assert sink.pending_count == 0
    assert clock.step_index == 0


def test_atomic_demo_bridge_is_lazy_and_waits_with_hold_actions() -> None:
    bridge, runtime, clock = _bridge(duration=2 * STEP_DT)

    demo_segment = next(bridge.iter_segments())
    assert runtime.start_count == 0
    with pytest.raises(RuntimeError, match="before its action iterable"):
        demo_segment.validator()

    actions = iter(demo_segment.actions)
    command = next(actions)
    assert runtime.start_count == 1
    assert runtime.calls == ("pick", "place")
    assert clock.step_index == 0
    assert command.metadata["bridge_action_kind"] == "runtime_command"
    assert command.metadata["environment_step"] == 0

    wait_hold = next(actions)
    assert clock.step_index == 1
    assert wait_hold.metadata["bridge_action_kind"] == "runtime_wait_hold"
    assert torch.equal(wait_hold.value, command.value)

    with pytest.raises(StopIteration):
        next(actions)
    assert clock.step_index == 2
    assert runtime.status is SemanticExecutionStatus.COMPLETED
    assert runtime.cancel_count == 0
    assert demo_segment.validator().tolist() == [True, True]


def test_bridge_publishes_completion_only_after_normal_program_exhaustion() -> None:
    """Task success cannot observe a partial segment lifecycle as completion."""
    bridge, _, _ = _bridge(duration=STEP_DT)
    segments = iter(bridge.iter_segments())

    demo_segment = next(segments)
    assert bridge.program_completed is False
    with pytest.raises(RuntimeError, match="before all segments"):
        bridge.completion_mask

    tuple(demo_segment.actions)
    assert demo_segment.validator().tolist() == [True, True]
    assert bridge.program_completed is False

    with pytest.raises(StopIteration):
        next(segments)
    assert bridge.program_completed is True
    completion = bridge.completion_mask
    completion[0] = False
    assert bridge.completion_mask.tolist() == [True, True]


def test_closing_without_abort_handshake_fails_loudly_and_does_not_ack() -> None:
    bridge, runtime, clock = _bridge(duration=2 * STEP_DT)
    actions = iter(next(bridge.iter_segments()).actions)

    next(actions)
    with pytest.raises(TaskProgramBridgeError, match="abort_actions"):
        actions.close()

    assert clock.step_index == 0
    assert runtime.cancel_count == 1


def test_abort_handshake_discards_unconsumed_command_and_yields_safe_hold() -> None:
    bridge, runtime, clock = _bridge(duration=2 * STEP_DT)
    segment = next(bridge.iter_segments())
    actions = iter(segment.actions)

    command = next(actions)
    assert command.metadata["bridge_action_kind"] == "runtime_command"
    assert segment.abort_actions is not None
    emergency = iter(segment.abort_actions("operator stop", last_action_consumed=False))
    hold = next(emergency)

    assert hold.metadata["bridge_action_kind"] == "runtime_abort_safe_hold"
    assert clock.step_index == 0
    with pytest.raises(StopIteration):
        next(emergency)
    assert clock.step_index == 1
    assert runtime.cancel_count == 1
    assert runtime.sink.pending_count == 0
    assert segment.metadata["runtime"]["status"] == "cancelled"
    assert segment.metadata["runtime"]["masks"]["cancelled"] == [True, True]
    with pytest.raises(RuntimeError, match="already started"):
        next(
            iter(
                segment.abort_actions(
                    "duplicate stop",
                    last_action_consumed=False,
                )
            )
        )
    actions.close()


def test_abort_handshake_acknowledges_consumed_command_exactly_once() -> None:
    bridge, runtime, clock = _bridge(duration=2 * STEP_DT)
    segment = next(bridge.iter_segments())
    actions = iter(segment.actions)

    next(actions)
    assert segment.abort_actions is not None
    emergency = iter(
        segment.abort_actions("environment failure", last_action_consumed=True)
    )
    hold = next(emergency)

    assert clock.step_index == 1
    assert hold.metadata["bridge_action_kind"] == "runtime_abort_safe_hold"
    with pytest.raises(StopIteration):
        next(emergency)
    assert clock.step_index == 2
    assert runtime.cancel_count == 1
    actions.close()


def test_abort_replays_unconsumed_terminal_safe_hold_without_recancelling() -> None:
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    runtime = _TerminalHoldRuntime(sink, clock, _joint_frame(duration=STEP_DT))
    bridge = TaskProgramDemoBridge(_FakeProgram(_FakeSegment()), runtime, sink, clock)
    segment = next(bridge.iter_segments())
    actions = iter(segment.actions)

    command = next(actions)
    assert command.metadata["bridge_action_kind"] == "runtime_command"
    terminal_hold = next(actions)
    assert terminal_hold.metadata["bridge_action_kind"] == "runtime_safe_hold"
    assert runtime.status is SemanticExecutionStatus.COMPLETED
    assert clock.step_index == 1

    assert segment.abort_actions is not None
    emergency = iter(
        segment.abort_actions("stop before hold", last_action_consumed=False)
    )
    replay = next(emergency)
    assert replay.metadata["bridge_action_kind"] == "runtime_abort_safe_hold"
    assert torch.equal(replay.value, terminal_hold.value)
    with pytest.raises(StopIteration):
        next(emergency)

    assert clock.step_index == 2
    assert runtime.cancel_count == 0
    assert sink.pending_count == 0
    actions.close()


def test_post_policy_interruption_replays_last_runtime_safe_hold() -> None:
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    runtime = _TerminalHoldRuntime(sink, clock, _joint_frame(duration=STEP_DT))
    post_policy = object()
    segment_spec = _FakeSegment(post_policies=(post_policy,))
    bridge = TaskProgramDemoBridge(
        _FakeProgram(segment_spec),
        runtime,
        sink,
        clock,
        post_policy_port=_PostPolicyPort(torch.ones(BATCH_SIZE, ROBOT_DOF)),
    )
    segment = next(bridge.iter_segments())
    actions = iter(segment.actions)

    assert next(actions).metadata["bridge_action_kind"] == "runtime_command"
    assert next(actions).metadata["bridge_action_kind"] == "runtime_safe_hold"
    post_action = next(actions)
    assert post_action.metadata["bridge_action_kind"] == "program_post_policy"
    assert clock.step_index == 2

    assert segment.abort_actions is not None
    emergency = iter(
        segment.abort_actions("post policy stop", last_action_consumed=False)
    )
    replay = next(emergency)
    assert replay.metadata["bridge_action_kind"] == "runtime_abort_safe_hold"
    assert torch.equal(replay.value, torch.zeros(BATCH_SIZE, ROBOT_DOF))
    with pytest.raises(StopIteration):
        next(emergency)

    assert clock.step_index == 3
    assert runtime.cancel_count == 0
    actions.close()


class _BridgeExecutorEnv:
    """Minimal demo executor proving abort actions cross the Gym boundary."""

    def __init__(
        self,
        bridge: TaskProgramDemoBridge,
        *,
        fail_first_mask: torch.Tensor | None = None,
        raise_first_step: bool = False,
    ) -> None:
        self.bridge = bridge
        self.fail_first_mask = (
            torch.zeros(BATCH_SIZE, dtype=torch.bool)
            if fail_first_mask is None
            else fail_first_mask.clone()
        )
        self.raise_first_step = raise_first_step
        self.num_envs = BATCH_SIZE
        self.steps: list[ControllerAction] = []
        self._demo_no_auto_reset = False

    @property
    def unwrapped(self) -> _BridgeExecutorEnv:
        return self

    def create_demo_segments(self):
        return self.bridge.iter_segments()

    def step(self, action: ControllerAction):
        assert isinstance(action, ControllerAction)
        self.steps.append(action.snapshot())
        if self.raise_first_step and len(self.steps) == 1:
            raise RuntimeError("simulated environment failure")
        failed = (
            self.fail_first_mask
            if len(self.steps) == 1
            else torch.zeros(BATCH_SIZE, dtype=torch.bool)
        )
        return (
            None,
            torch.zeros(BATCH_SIZE),
            torch.zeros(BATCH_SIZE, dtype=torch.bool),
            torch.zeros(BATCH_SIZE, dtype=torch.bool),
            {"fail": failed},
        )

    def _mask_demo_action(
        self,
        action: ControllerAction,
        active_mask: tuple[bool, ...],
    ) -> ControllerAction:
        del active_mask
        return action.snapshot()


def test_zero_command_terminal_runtime_failure_preserves_trace_and_validates_once() -> (
    None
):
    validator = object()
    first = _FakeSegment(
        calls=(_FakeCompiledCall(0, "slide"),),
        validators=(validator,),
    )
    second = _FakeSegment(
        segment_index=1,
        segment_id="segment-1",
        name="must-not-start",
        calls=(_FakeCompiledCall(1, "place"),),
        source_path=("program", "steps", 1),
    )
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    runtime = _TerminalFailedRuntime(
        sink,
        clock,
        _joint_frame(duration=STEP_DT),
    )
    validator_port = _ValidatorPort(torch.ones(BATCH_SIZE, dtype=torch.bool))
    bridge = TaskProgramDemoBridge(
        _FakeProgram(first, second),
        runtime,
        sink,
        clock,
        validator_port=validator_port,
    )
    env = _BridgeExecutorEnv(bridge)

    result = execute_demo_episode(env)

    assert env.steps == []
    assert clock.step_index == 0
    assert runtime.start_count == 1
    assert runtime.step_count == 0
    assert validator_port.seen == [validator]
    assert not result.completed
    assert result.terminal_reason == "segment_validation_failed"
    assert len(result.segments) == 1
    segment_result = result.segments[0]
    assert segment_result.failure_reason == "segment_validation_failed"
    assert segment_result.outcome_kinds == ("runtime_failed", "runtime_failed")
    runtime_trace = segment_result.metadata["runtime"]
    assert runtime_trace["status"] == "failed"
    assert (
        runtime_trace["message"] == "Motion planning failed before the first command."
    )
    assert runtime_trace["events"] == [
        {
            "kind": "action_planning_failed",
            "timestamp": 0.0,
            "skill_id": "slide",
            "invocation_id": "open-drawer-call",
            "invocation_revision": 0,
            "invocation_index": 0,
            "env_mask": [True, True],
            "message": "Slide contact motion failed.",
        }
    ]
    assert segment_result.metadata["validation"] == {
        "env_ids": [7, 3],
        "runtime_success_mask": [False, False],
        "eligible_mask_before_validation": [False, False],
        "post_policy_success_mask": None,
        "validators": [
            {
                "validator_index": 0,
                "kind": "object",
                "source_path": [],
                "result_mask": [True, True],
                "result": {},
            }
        ],
        "accepted_mask": [False, False],
    }


def test_sequential_start_failure_before_first_command_preserves_cause() -> None:
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    runtime = _StartFailingRuntime(sink, clock, _joint_frame(duration=STEP_DT))
    bridge = TaskProgramDemoBridge(_FakeProgram(_FakeSegment()), runtime, sink, clock)
    env = _BridgeExecutorEnv(bridge)

    with pytest.raises(RuntimeError, match="action generation") as error:
        execute_demo_episode(env)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "semantic runtime preflight failed"
    assert env.steps == []
    assert clock.step_index == 0
    assert runtime.start_count == 1
    assert runtime.cancel_count == 0
    assert sink.pending_count == 0


def test_parallel_construction_failure_before_first_command_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _FakeParallelBlock(
        branches=(
            _FakeParallelBranch(0, (_FakeCompiledCall(0, "left"),)),
            _FakeParallelBranch(1, (_FakeCompiledCall(1, "right"),)),
        )
    )
    segment = _FakeSegment(parallel_block=block)
    bridge, runtime, clock = _bridge(
        duration=STEP_DT,
        segment=segment,
        parallel_safety_validator=_AcceptParallelSafety(),
    )
    env = _BridgeExecutorEnv(bridge)

    def fail_construction(*args: object, **kwargs: object) -> _FakeParallelRuntime:
        del args, kwargs
        raise RuntimeError("parallel runtime construction failed")

    monkeypatch.setattr(bridge_module, "SemanticCallExecutor", _FakeRuntime)
    monkeypatch.setattr(
        ParallelSemanticExecutor,
        "from_template",
        classmethod(fail_construction),
    )

    with pytest.raises(RuntimeError, match="action generation") as error:
        execute_demo_episode(env)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "parallel runtime construction failed"
    assert env.steps == []
    assert clock.step_index == 0
    assert runtime.start_count == 0
    assert runtime.cancel_count == 0
    assert runtime.sink.pending_count == 0


def test_post_policy_timeout_is_row_local_and_preserved_in_segment_result() -> None:
    segment = _FakeSegment(post_policies=(object(),))
    bridge, runtime, clock = _bridge(
        duration=STEP_DT,
        segment=segment,
        post_policy_port=_MetadataPostPolicyPort(torch.ones(BATCH_SIZE, ROBOT_DOF)),
    )
    env = _BridgeExecutorEnv(bridge)

    result = execute_demo_episode(env)

    assert result.segments[0].successes == (True, False)
    assert result.segments[0].failure_reasons == (
        None,
        "segment_validation_failed",
    )
    assert result.segments[0].outcome_kinds == (
        "succeeded",
        "post_policy_failed",
    )
    assert result.segments[0].metadata["post_policies"][0]["result_mask"] == [
        True,
        False,
    ]
    assert result.segments[0].metadata["validation"]["accepted_mask"] == [
        True,
        False,
    ]
    assert [action.metadata["bridge_action_kind"] for action in env.steps] == [
        "runtime_command",
        "program_post_policy",
    ]
    assert runtime.cancel_count == 0
    assert clock.step_index == 2


def test_post_policy_generator_error_replays_safe_hold_before_propagating() -> None:
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    runtime = _TerminalHoldRuntime(sink, clock, _joint_frame(duration=STEP_DT))
    segment_spec = _FakeSegment(post_policies=(object(),))
    bridge = TaskProgramDemoBridge(
        _FakeProgram(segment_spec),
        runtime,
        sink,
        clock,
        post_policy_port=_FailingPostPolicyPort(),
    )
    env = _BridgeExecutorEnv(bridge)

    with pytest.raises(RuntimeError, match="action generation") as error:
        execute_demo_episode(env)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert [action.metadata["bridge_action_kind"] for action in env.steps] == [
        "runtime_command",
        "runtime_safe_hold",
        "runtime_abort_safe_hold",
    ]
    assert runtime.cancel_count == 0
    assert sink.pending_count == 0
    assert clock.step_index == 3


def test_demo_executor_pre_step_stop_consumes_only_abort_hold() -> None:
    bridge, runtime, clock = _bridge(duration=2 * STEP_DT)
    env = _BridgeExecutorEnv(bridge)
    checks = iter((False, True))

    result = execute_demo_episode(env, should_stop=lambda: next(checks, True))

    assert result.terminal_reason == "interrupted"
    assert result.length == 1
    assert [action.metadata["bridge_action_kind"] for action in env.steps] == [
        "runtime_abort_safe_hold"
    ]
    assert clock.step_index == 1
    assert runtime.cancel_count == 1
    assert runtime.sink.pending_count == 0


def test_demo_executor_post_step_failure_acknowledges_then_safe_stops() -> None:
    bridge, runtime, clock = _bridge(duration=2 * STEP_DT)
    env = _BridgeExecutorEnv(
        bridge,
        fail_first_mask=torch.ones(BATCH_SIZE, dtype=torch.bool),
    )

    result = execute_demo_episode(env)

    assert result.terminal_reason == "failure"
    assert result.length == 2
    assert [action.metadata["bridge_action_kind"] for action in env.steps] == [
        "runtime_command",
        "runtime_abort_safe_hold",
    ]
    assert clock.step_index == 2
    assert runtime.cancel_count == 1
    assert runtime.sink.pending_count == 0


def test_demo_executor_safe_stops_when_regular_env_step_raises() -> None:
    bridge, runtime, clock = _bridge(duration=2 * STEP_DT)
    env = _BridgeExecutorEnv(bridge, raise_first_step=True)

    with pytest.raises(RuntimeError, match="emergency safe-stop"):
        execute_demo_episode(env)

    assert [action.metadata["bridge_action_kind"] for action in env.steps] == [
        "runtime_command",
        "runtime_abort_safe_hold",
    ]
    assert clock.step_index == 1
    assert runtime.cancel_count == 1
    assert runtime.sink.pending_count == 0


def test_row_independent_partial_failure_does_not_abort_healthy_peer() -> None:
    bridge, runtime, clock = _bridge(duration=2 * STEP_DT)
    env = _BridgeExecutorEnv(
        bridge,
        fail_first_mask=torch.tensor([True, False]),
    )

    result = execute_demo_episode(env)

    assert result.lengths == (1, 2)
    assert [action.metadata["bridge_action_kind"] for action in env.steps] == [
        "runtime_command",
        "runtime_wait_hold",
    ]
    assert clock.step_index == 2
    assert runtime.cancel_count == 0
    assert runtime.sink.pending_count == 0


def test_post_policy_and_validator_ports_stay_at_demo_boundary() -> None:
    post_policy = object()
    validator = object()
    segment = _FakeSegment(
        post_policies=(post_policy,),
        validators=(validator,),
    )
    post_port = _PostPolicyPort(torch.ones(BATCH_SIZE, ROBOT_DOF))
    validator_port = _ValidatorPort(torch.tensor([True, False]))
    bridge, _, clock = _bridge(
        duration=STEP_DT,
        segment=segment,
        post_policy_port=post_port,
        validator_port=validator_port,
    )
    demo_segment = next(bridge.iter_segments())
    actions = iter(demo_segment.actions)

    runtime_action = next(actions)
    assert runtime_action.metadata["bridge_action_kind"] == "runtime_command"
    post_action = next(actions)
    assert clock.step_index == 1
    assert post_action.metadata["bridge_action_kind"] == "program_post_policy"
    with pytest.raises(StopIteration):
        next(actions)

    assert clock.step_index == 2
    assert post_port.seen == [post_policy]
    assert len(post_port.active_masks) == 1
    assert post_port.active_masks[0].tolist() == [True, True]
    assert demo_segment.validator().tolist() == [True, False]
    assert validator_port.seen == [validator]


def test_post_policy_receives_only_rows_surviving_partial_runtime_failure() -> None:
    """Post-policy completion cannot be blocked by a runtime-failed row."""
    post_policy = object()
    segment = _FakeSegment(post_policies=(post_policy,))
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    runtime = _PartialSuccessRuntime(
        sink,
        clock,
        _joint_frame(duration=STEP_DT),
    )
    post_port = _PostPolicyPort(torch.ones(BATCH_SIZE, ROBOT_DOF))
    bridge = TaskProgramDemoBridge(
        _FakeProgram(segment),
        runtime,
        sink,
        clock,
        post_policy_port=post_port,
    )
    demo_segment = next(bridge.iter_segments())

    tuple(demo_segment.actions)

    assert len(post_port.active_masks) == 1
    assert post_port.active_masks[0].tolist() == [True, False]
    assert demo_segment.metadata["post_policies"][0]["result_mask"] == [True, False]
    assert demo_segment.validator().tolist() == [True, False]


def test_later_post_policy_receives_only_rows_passing_prior_policy() -> None:
    """Sequential post-policies monotonically narrow their active cohort."""
    segment = _FakeSegment(post_policies=(object(), object()))
    post_port = _MetadataPostPolicyPort(
        torch.ones(BATCH_SIZE, ROBOT_DOF),
    )
    bridge, _, _ = _bridge(
        duration=STEP_DT,
        segment=segment,
        post_policy_port=post_port,
    )

    tuple(next(bridge.iter_segments()).actions)

    assert [mask.tolist() for mask in post_port.active_masks] == [
        [True, True],
        [True, False],
    ]


def test_segment_lifecycle_metadata_records_runtime_post_and_validation() -> None:
    post_policy = object()
    validator = object()
    segment = _FakeSegment(
        post_policies=(post_policy,),
        validators=(validator,),
    )
    bridge, _, _ = _bridge(
        duration=STEP_DT,
        segment=segment,
        post_policy_port=_MetadataPostPolicyPort(torch.ones(BATCH_SIZE, ROBOT_DOF)),
        validator_port=_MetadataValidatorPort(torch.tensor([True, False])),
    )
    demo_segment = next(bridge.iter_segments())

    tuple(demo_segment.actions)
    assert demo_segment.metadata["runtime"]["status"] == "completed"
    assert demo_segment.metadata["post_policies"] == [
        {
            "policy_index": 0,
            "kind": "object",
            "source_path": [],
            "result_mask": [True, False],
            "result": {
                "status": "timed_out",
                "state": {
                    "elapsed_steps": 1,
                    "settled_mask": [True, False],
                    "timeout_mask": [False, True],
                },
            },
        }
    ]

    assert demo_segment.validator().tolist() == [True, False]
    assert demo_segment.metadata["validation"] == {
        "env_ids": [7, 3],
        "runtime_success_mask": [True, True],
        "eligible_mask_before_validation": [True, True],
        "post_policy_success_mask": [True, False],
        "validators": [
            {
                "validator_index": 0,
                "kind": "object",
                "source_path": [],
                "result_mask": [True, False],
                "result": {"position_error": [0.01, 0.1]},
            }
        ],
        "accepted_mask": [True, False],
    }
    json.dumps(demo_segment.metadata, allow_nan=False, sort_keys=True)


def test_declared_post_policy_requires_explicit_port() -> None:
    segment = _FakeSegment(post_policies=(object(),))
    bridge, _, _ = _bridge(duration=STEP_DT, segment=segment)

    with pytest.raises(TaskProgramBridgeError, match="no SegmentPostPolicyPort"):
        tuple(next(bridge.iter_segments()).actions)


def test_bridge_marks_segments_row_independent_and_retains_failed_rows() -> None:
    first_validator = object()
    first = _FakeSegment(validators=(first_validator,))
    second = _FakeSegment(
        segment_index=1,
        segment_id="segment-1",
        name="place-next",
        calls=(_FakeCompiledCall(2, "place-next"),),
        source_path=("program", "steps", 1),
    )
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    runtime = _FakeRuntime(sink, clock, _joint_frame(duration=STEP_DT))
    bridge = TaskProgramDemoBridge(
        _FakeProgram(first, second),
        runtime,
        sink,
        clock,
        validator_port=_ValidatorPort(torch.tensor([True, False])),
    )
    segments = iter(bridge.iter_segments())

    first_demo = next(segments)
    assert first_demo.failure_policy == "row_independent"
    tuple(first_demo.actions)
    assert first_demo.validator().tolist() == [True, False]

    second_demo = next(segments)
    tuple(second_demo.actions)
    assert runtime.eligible_masks[0] is None
    assert runtime.eligible_masks[1].tolist() == [True, False]
    assert second_demo.validator().tolist() == [True, False]


def test_bridge_refuses_next_segment_when_validator_was_skipped() -> None:
    first = _FakeSegment(calls=(_FakeCompiledCall(0, "pick"),))
    second = _FakeSegment(
        segment_index=1,
        segment_id="segment-1",
        name="place",
        calls=(_FakeCompiledCall(1, "place"),),
        source_path=("program", "steps", 1),
    )
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    runtime = _FakeRuntime(sink, clock, _joint_frame(duration=STEP_DT))
    segments = iter(
        TaskProgramDemoBridge(_FakeProgram(first, second), runtime, sink, clock)
    )

    first_demo = next(segments)
    tuple(first_demo.actions)

    with pytest.raises(TaskProgramBridgeError, match="validator must be called"):
        next(segments)
    assert runtime.start_count == 1


def test_demo_executor_consumes_validation_before_requesting_next_segment() -> None:
    first_validator = object()
    first = _FakeSegment(
        calls=(_FakeCompiledCall(0, "pick"),),
        validators=(first_validator,),
    )
    second = _FakeSegment(
        segment_index=1,
        segment_id="segment-1",
        name="place",
        calls=(_FakeCompiledCall(1, "place"),),
        source_path=("program", "steps", 1),
    )
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    runtime = _FakeRuntime(sink, clock, _joint_frame(duration=STEP_DT))
    validator_port = _ValidatorPort(torch.ones(BATCH_SIZE, dtype=torch.bool))
    bridge = TaskProgramDemoBridge(
        _FakeProgram(first, second),
        runtime,
        sink,
        clock,
        validator_port=validator_port,
    )

    result = execute_demo_episode(_BridgeExecutorEnv(bridge))

    assert result.completed
    assert len(result.segments) == 2
    assert [segment.success for segment in result.segments] == [True, True]
    assert runtime.start_count == 2
    assert validator_port.seen == [first_validator]


def test_sequential_segment_analyzes_downstream_calls_but_executes_own_prefix() -> None:
    first = _FakeSegment(
        calls=(_FakeCompiledCall(0, "pick"),),
    )
    second = _FakeSegment(
        segment_index=1,
        segment_id="segment-1",
        name="place",
        calls=(_FakeCompiledCall(1, "place"),),
        source_path=("program", "steps", 1),
    )
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    runtime = _FakeRuntime(sink, clock, _joint_frame(duration=STEP_DT))
    bridge = TaskProgramDemoBridge(_FakeProgram(first, second), runtime, sink, clock)
    segments = iter(bridge.iter_segments())

    first_demo = next(segments)
    tuple(first_demo.actions)
    assert first_demo.validator().tolist() == [True, True]

    assert runtime.calls == ("pick", "place")
    assert runtime.execution_prefix_lengths == [1]

    tuple(next(segments).actions)
    assert runtime.calls == ("place",)
    assert runtime.execution_prefix_lengths == [1, 1]


def test_parallel_segment_preserves_branches_barrier_and_adopts_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_calls = (_FakeCompiledCall(0, "left-pick"),)
    right_calls = (
        _FakeCompiledCall(1, "right-pick"),
        _FakeCompiledCall(2, "right-place"),
    )
    block = _FakeParallelBlock(
        branches=(
            _FakeParallelBranch(0, left_calls),
            _FakeParallelBranch(1, right_calls),
        )
    )
    segment = _FakeSegment(
        calls=left_calls + right_calls,
        parallel_block=block,
    )
    safety_validator = _AcceptParallelSafety()
    bridge, runtime, clock = _bridge(
        duration=STEP_DT,
        segment=segment,
        parallel_safety_validator=safety_validator,
    )
    captured: dict[str, Any] = {}
    fake_parallel = _FakeParallelRuntime(runtime.sink, clock)

    def from_template(
        cls: type[ParallelSemanticExecutor],
        template_runtime: object,
        branch_calls: dict[str, tuple[object, ...]],
        command_sink: object,
        timing_policy: object,
        supplied_safety_validator: object,
        *,
        timeout_steps: int,
        failure_policy: str,
        runner_cfg: object,
        workflow_id: str,
        branch_paths: dict[str, tuple[object, ...]],
    ) -> _FakeParallelRuntime:
        del cls
        captured.update(
            {
                "template_runtime": template_runtime,
                "branch_calls": branch_calls,
                "command_sink": command_sink,
                "timing_policy": timing_policy,
                "safety_validator": supplied_safety_validator,
                "timeout_steps": timeout_steps,
                "failure_policy": failure_policy,
                "runner_cfg": runner_cfg,
                "workflow_id": workflow_id,
                "branch_paths": branch_paths,
            }
        )
        return fake_parallel

    monkeypatch.setattr(bridge_module, "SemanticCallExecutor", _FakeRuntime)
    monkeypatch.setattr(
        ParallelSemanticExecutor,
        "from_template",
        classmethod(from_template),
    )

    demo_segment = next(bridge.iter_segments())
    actions = tuple(demo_segment.actions)

    assert len(actions) == 1
    assert captured["template_runtime"] is runtime
    assert captured["command_sink"] is runtime.sink
    assert captured["branch_calls"] == {
        "branch_0": ("left-pick",),
        "branch_1": ("right-pick", "right-place"),
    }
    assert captured["timing_policy"].step_dt == STEP_DT
    assert captured["safety_validator"] is safety_validator
    assert captured["timeout_steps"] == 17
    assert captured["failure_policy"] == "fail_fast"
    assert captured["runner_cfg"] is not None
    assert captured["workflow_id"].endswith(":parallel_analysis")
    assert captured["branch_paths"] == {
        "branch_0": segment.source_path,
        "branch_1": segment.source_path,
    }
    assert len(runtime.adopted_states) == 1
    assert demo_segment.failure_policy == "row_independent"
    assert demo_segment.metadata["runtime"]["kind"] == "parallel_skill_result"
    assert demo_segment.metadata["runtime"]["status"] == "completed"
    assert demo_segment.metadata["runtime"]["masks"]["success"] == [True, True]
    assert demo_segment.validator().tolist() == [True, True]


def test_real_parallel_coordinator_buffers_one_ordered_gym_action_per_step() -> None:
    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_QposProvider(torch.zeros(BATCH_SIZE, ROBOT_DOF))),
        clock,
    )
    left_sink = ParallelLaneCommandSink()
    right_sink = ParallelLaneCommandSink()
    left_runtime = _GridLaneRuntime(
        left_sink,
        (
            (SemanticExecutionStatus.RUNNING, _grid_frame("left_arm", 0, 1.0)),
            (SemanticExecutionStatus.COMPLETED, None),
        ),
    )
    right_runtime = _GridLaneRuntime(
        right_sink,
        (
            (SemanticExecutionStatus.RUNNING, _grid_frame("right_arm", 1, 2.0)),
            (SemanticExecutionStatus.RUNNING, _grid_frame("right_arm", 1, 3.0)),
            (SemanticExecutionStatus.COMPLETED, None),
        ),
    )
    runtime = ParallelSemanticExecutor(
        (
            ParallelExecutorBranch(
                "left",
                (RegisteredSemanticCall("test.left"),),
                ResourceClaim(frozenset({"left_arm"}), (0,)),
                left_runtime,
                left_sink,
            ),
            ParallelExecutorBranch(
                "right",
                (RegisteredSemanticCall("test.right"),),
                ResourceClaim(frozenset({"right_arm"}), (1,)),
                right_runtime,
                right_sink,
            ),
        ),
        sink,
        clock,
        ParallelTimingPolicy(STEP_DT),
        _AcceptParallelSafety(),
        timeout_steps=8,
    )

    runtime.start()
    first = runtime.step()
    assert first.status is SemanticExecutionStatus.RUNNING
    assert sink.pending_count == 1
    first_action = sink.pop()
    assert first_action.metadata["bridge_action_kind"] == "runtime_command"
    assert torch.equal(first_action.value[:, 0], torch.ones(BATCH_SIZE))
    assert torch.equal(first_action.value[:, 1], torch.full((BATCH_SIZE,), 2.0))
    clock.advance_after_env_step()

    padded = runtime.step()
    assert padded.status is SemanticExecutionStatus.RUNNING
    assert sink.pending_count == 1
    padding_action = sink.pop()
    assert padding_action.metadata["bridge_action_kind"] == "runtime_safe_hold"
    assert torch.equal(padding_action.value, torch.zeros(BATCH_SIZE, ROBOT_DOF))
    lane_steps = (left_runtime.step_count, right_runtime.step_count)
    clock.advance_after_env_step()

    deferred = runtime.step()
    assert deferred.status is SemanticExecutionStatus.RUNNING
    assert sink.pending_count == 1
    deferred_action = sink.pop()
    assert deferred_action.metadata["bridge_action_kind"] == "runtime_command"
    assert torch.equal(deferred_action.value[:, 0], torch.zeros(BATCH_SIZE))
    assert torch.equal(
        deferred_action.value[:, 1],
        torch.full((BATCH_SIZE,), 3.0),
    )
    assert (left_runtime.step_count, right_runtime.step_count) == lane_steps
    clock.advance_after_env_step()

    completed = runtime.step()
    assert completed.status is SemanticExecutionStatus.COMPLETED
    assert sink.pending_count == 1
    terminal_hold = sink.pop()
    assert terminal_hold.metadata["bridge_action_kind"] == "runtime_safe_hold"
    assert completed.command_count == 2
    clock.advance_after_env_step()
    assert clock.step_index == 4


def test_parallel_segment_fails_closed_without_safety_validator() -> None:
    block = _FakeParallelBlock(
        branches=(
            _FakeParallelBranch(0, (_FakeCompiledCall(0, "left"),)),
            _FakeParallelBranch(1, (_FakeCompiledCall(1, "right"),)),
        )
    )
    segment = _FakeSegment(parallel_block=block)
    bridge, runtime, _ = _bridge(duration=STEP_DT, segment=segment)

    with pytest.raises(TaskProgramBridgeError, match="requires an explicit"):
        tuple(next(bridge.iter_segments()).actions)

    assert runtime.start_count == 0
