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

"""Tests for branch-local semantic execution at a parallel barrier."""

from __future__ import annotations

from dataclasses import dataclass
import json

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ArticulationJointState,
    CommandAcknowledgement,
    EndpointCommand,
    ExecutionRunnerCfg,
    JointPositionPayload,
    JointPositionTarget,
    PlanningContext,
    RobotObservation,
    RuntimeCommandFrame,
    SceneSnapshot,
    StateDelta,
    TaskState,
)
from embodichain.lab.semantic_skills.calls import RegisteredSemanticCall
from embodichain.lab.expert_program._parallel import ParallelTimingPolicy
from embodichain.lab.expert_program._parallel_executor import (
    ParallelLaneCommandSink,
    ParallelExecutorBranch,
    ParallelSemanticExecutor,
)
from embodichain.lab.semantic_skills.profiles import ResourceClaim
from embodichain.lab.expert_program._semantic_results import (
    SemanticExecutionResult,
    SemanticExecutionStatus,
)

ENV_IDS = torch.tensor([4, 9], dtype=torch.long)


class _Clock:
    """Deterministic environment-grid clock."""

    def __init__(self) -> None:
        self.time = 0.0

    def now(self) -> float:
        return self.time

    def sleep(self, duration: float) -> None:
        self.time += duration


class _OutboundSink:
    """Record the coordinator's one merged transport transaction."""

    def __init__(
        self,
        *,
        reject: bool = False,
        reject_cancel: bool = False,
        reject_hold: bool = False,
        raise_send: bool = False,
    ) -> None:
        self.reject = reject
        self.reject_cancel = reject_cancel
        self.reject_hold = reject_hold
        self.raise_send = raise_send
        self.frames: list[RuntimeCommandFrame] = []
        self.hold_targets: list[tuple[str, ...]] = []
        self.hold_fingerprints: list[tuple[object, ...]] = []
        self.operations: list[str] = []
        self.timeouts: list[tuple[str, float]] = []
        self.holds = 0
        self.cancels = 0

    def send(
        self,
        command: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        self.timeouts.append(("send", timeout))
        if self.raise_send:
            raise RuntimeError("send exploded")
        self.operations.append("send")
        self.frames.append(command.snapshot())
        if self.reject:
            return CommandAcknowledgement.rejected_ack("test rejection")
        return CommandAcknowledgement.accepted_ack()

    def hold(
        self,
        targets: tuple[object, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        del context
        self.timeouts.append(("hold", timeout))
        self.operations.append("hold")
        self.holds += 1
        self.hold_targets.append(
            tuple(getattr(target, "target_id") for target in targets)
        )
        self.hold_fingerprints.append(
            tuple(getattr(target, "address_fingerprint") for target in targets)
        )
        if self.reject_hold:
            return CommandAcknowledgement.rejected_ack("hold rejected")
        return CommandAcknowledgement.accepted_ack()

    def cancel(
        self,
        targets: tuple[object, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        del targets
        self.timeouts.append(("cancel", timeout))
        self.operations.append("cancel")
        self.cancels += 1
        if self.reject_cancel:
            return CommandAcknowledgement.rejected_ack("cancel rejected")
        return CommandAcknowledgement.accepted_ack()


class _AcceptSafety:
    """Accept fake joint commands while recording validation calls."""

    def __init__(self) -> None:
        self.calls = 0

    def validate(
        self,
        *,
        branch_frames: dict[str, RuntimeCommandFrame],
        merged_frame: RuntimeCommandFrame,
    ) -> None:
        assert branch_frames
        assert merged_frame.commands
        self.calls += 1


class _RejectSafety:
    """Reject every synchronized motion as physically unsafe."""

    def validate(
        self,
        *,
        branch_frames: dict[str, RuntimeCommandFrame],
        merged_frame: RuntimeCommandFrame,
    ) -> None:
        del branch_frames, merged_frame
        raise RuntimeError("predicted self collision")


@dataclass(frozen=True, slots=True)
class _ScriptStep:
    """One fake lane cycle."""

    status: SemanticExecutionStatus
    eligible: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    cancelled: torch.Tensor
    frame: RuntimeCommandFrame | None = None
    task_state: TaskState | None = None
    wait_duration: float = 0.0
    emit_hold: bool = False
    hold_targets: tuple[JointPositionTarget, ...] = ()


class _BranchRuntime:
    """Small deterministic implementation of the parallel runtime protocol."""

    def __init__(
        self,
        script: tuple[_ScriptStep, ...],
        sink: ParallelLaneCommandSink,
        *,
        initial_state: TaskState | None = None,
        emit_terminal_hold: bool = True,
    ) -> None:
        self._script = script
        self._sink = sink
        self._index = 0
        self._state = initial_state or TaskState.empty(2, "cpu")
        self._emit_terminal_hold = emit_terminal_hold
        self._result = self._make_result(
            SemanticExecutionStatus.IDLE,
            eligible=torch.ones(2, dtype=torch.bool),
        )

    @property
    def result(self) -> SemanticExecutionResult:
        return self._result

    @property
    def step_count(self) -> int:
        return self._index

    def start(
        self,
        *calls: RegisteredSemanticCall,
        workflow_id: str,
        eligible_mask: torch.Tensor | None = None,
    ) -> SemanticExecutionResult:
        del calls
        eligible = (
            torch.ones(2, dtype=torch.bool)
            if eligible_mask is None
            else eligible_mask.clone()
        )
        self._result = self._make_result(
            SemanticExecutionStatus.RUNNING,
            workflow_id=workflow_id,
            eligible=eligible,
        )
        return self._result

    def step(self) -> SemanticExecutionResult:
        scripted = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        if scripted.frame is not None:
            self._sink.send(scripted.frame, timeout=1.0)
        self._state = scripted.task_state or self._state
        if scripted.hold_targets:
            self._sink.hold(
                scripted.hold_targets,
                _context(self._state),
                timeout=1.0,
            )
        elif scripted.emit_hold or (
            scripted.status is not SemanticExecutionStatus.RUNNING
            and self._emit_terminal_hold
        ):
            last_frame = scripted.frame or self._sink.last_frame
            targets = () if last_frame is None else last_frame.targets
            self._sink.hold(targets, _context(self._state), timeout=1.0)
        self._result = self._make_result(
            scripted.status,
            workflow_id=self._result.workflow_id,
            eligible=scripted.eligible & ~self._result.cancelled_mask,
            success=scripted.success & ~self._result.cancelled_mask,
            failure=scripted.failure,
            cancelled=self._result.cancelled_mask | scripted.cancelled,
            wait_duration=scripted.wait_duration,
        )
        return self._result

    def deactivate_rows(
        self,
        env_mask: torch.Tensor,
        *,
        reason: str,
    ) -> SemanticExecutionResult:
        del reason
        changed = env_mask & self._result.eligible_mask
        self._result = self._make_result(
            self._result.status,
            workflow_id=self._result.workflow_id,
            eligible=self._result.eligible_mask & ~changed,
            success=self._result.success_mask & ~changed,
            failure=self._result.failure_mask,
            cancelled=self._result.cancelled_mask | changed,
            wait_duration=self._result.wait_duration,
        )
        return self._result

    def cancel(self, reason: str) -> SemanticExecutionResult:
        del reason
        active = self._result.eligible_mask & ~self._result.failure_mask
        last_frame = self._sink.last_frame
        targets = () if last_frame is None else last_frame.targets
        self._sink.cancel(targets, timeout=1.0)
        self._sink.hold(targets, _context(self._state), timeout=1.0)
        self._result = self._make_result(
            SemanticExecutionStatus.CANCELLED,
            workflow_id=self._result.workflow_id,
            eligible=self._result.eligible_mask & ~active,
            failure=self._result.failure_mask,
            cancelled=self._result.cancelled_mask | active,
        )
        return self._result

    def _make_result(
        self,
        status: SemanticExecutionStatus,
        *,
        workflow_id: str | None = None,
        eligible: torch.Tensor | None = None,
        success: torch.Tensor | None = None,
        failure: torch.Tensor | None = None,
        cancelled: torch.Tensor | None = None,
        wait_duration: float = 0.0,
    ) -> SemanticExecutionResult:
        zeros = torch.zeros(2, dtype=torch.bool)
        return SemanticExecutionResult(
            status=status,
            workflow_id=workflow_id,
            current_call_index=0 if status is SemanticExecutionStatus.RUNNING else None,
            env_ids=ENV_IDS,
            success_mask=zeros if success is None else success,
            failure_mask=zeros if failure is None else failure,
            cancelled_mask=zeros if cancelled is None else cancelled,
            eligible_mask=(
                torch.ones(2, dtype=torch.bool) if eligible is None else eligible
            ),
            task_state=self._state,
            wait_duration=wait_duration,
        )


def _mask(first: bool, second: bool) -> torch.Tensor:
    return torch.tensor([first, second], dtype=torch.bool)


def _context(task_state: TaskState) -> PlanningContext:
    return PlanningContext(
        robot=RobotObservation(
            timestamp=1.0,
            qpos=torch.zeros(2, 3),
            qvel=torch.zeros(2, 3),
        ),
        task=task_state,
        scene=SceneSnapshot.empty(),
        env_ids=ENV_IDS,
    )


def _frame(joint_id: int, values: tuple[float, float]) -> RuntimeCommandFrame:
    target = JointPositionTarget(f"resource_{joint_id}", (joint_id,))
    return RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                target,
                JointPositionPayload(torch.tensor(values).reshape(2, 1)),
            ),
        ),
        active_mask=_mask(True, True),
        env_ids=ENV_IDS,
        hold_duration=torch.full((2,), 0.1),
    )


def _branch(
    branch_id: str,
    joint_id: int,
    script: tuple[_ScriptStep, ...],
    *,
    initial_state: TaskState | None = None,
    emit_terminal_hold: bool = True,
) -> ParallelExecutorBranch:
    sink = ParallelLaneCommandSink()
    return ParallelExecutorBranch(
        branch_id=branch_id,
        calls=(RegisteredSemanticCall(f"test.{branch_id}"),),
        claim=ResourceClaim(frozenset({f"resource_{joint_id}"}), (joint_id,)),
        executor=_BranchRuntime(
            script,
            sink,
            initial_state=initial_state,
            emit_terminal_hold=emit_terminal_hold,
        ),
        command_sink=sink,
    )


def _running_step(
    *,
    frame: RuntimeCommandFrame | None = None,
    eligible: torch.Tensor | None = None,
    failure: torch.Tensor | None = None,
    task_state: TaskState | None = None,
    wait_duration: float = 0.0,
    emit_hold: bool = False,
    hold_targets: tuple[JointPositionTarget, ...] = (),
) -> _ScriptStep:
    return _ScriptStep(
        SemanticExecutionStatus.RUNNING,
        _mask(True, True) if eligible is None else eligible,
        _mask(False, False),
        _mask(False, False) if failure is None else failure,
        _mask(False, False),
        frame,
        task_state,
        wait_duration=wait_duration,
        emit_hold=emit_hold,
        hold_targets=hold_targets,
    )


def _completed_step(
    *,
    frame: RuntimeCommandFrame | None = None,
    success: torch.Tensor | None = None,
    failure: torch.Tensor | None = None,
    task_state: TaskState | None = None,
) -> _ScriptStep:
    succeeded = _mask(True, True) if success is None else success
    failed = _mask(False, False) if failure is None else failure
    return _ScriptStep(
        SemanticExecutionStatus.COMPLETED,
        succeeded,
        succeeded,
        failed,
        _mask(False, False),
        frame,
        task_state,
    )


def test_parallel_runtime_merges_one_frame_and_hold_pads_short_lane() -> None:
    left_state = TaskState.empty(2, "cpu")
    left_state = StateDelta(
        articulation_joint_updates={
            ("left_fixture", "joint"): ArticulationJointState(torch.full((2, 1), 0.5))
        }
    ).apply(left_state, _mask(True, True))
    right_state = TaskState.empty(2, "cpu")
    right_state = StateDelta(
        articulation_joint_updates={
            ("right_fixture", "joint"): ArticulationJointState(torch.full((2, 1), 1.0))
        }
    ).apply(right_state, _mask(True, True))
    left = _branch(
        "left",
        0,
        (
            _running_step(frame=_frame(0, (1.0, 1.0))),
            _completed_step(task_state=left_state),
        ),
    )
    right = _branch(
        "right",
        1,
        (
            _running_step(frame=_frame(1, (2.0, 2.0))),
            _running_step(frame=_frame(1, (3.0, 3.0))),
            _completed_step(task_state=right_state),
        ),
    )
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (left, right),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=8,
    )

    result = runtime.start()
    assert result.status is SemanticExecutionStatus.RUNNING
    result = runtime.step()
    assert len(outbound.frames) == 1
    assert len(outbound.frames[0].commands) == 2
    assert outbound.operations == ["send"]
    assert isinstance(left.executor, _BranchRuntime)
    assert isinstance(right.executor, _BranchRuntime)
    first_lane_steps = (left.executor.step_count, right.executor.step_count)
    same_tick = runtime.step()
    assert same_tick.wait_duration == pytest.approx(0.1)
    assert outbound.operations == ["send"]
    assert (left.executor.step_count, right.executor.step_count) == first_lane_steps

    clock.time = 0.1
    result = runtime.step()
    assert result.status is SemanticExecutionStatus.RUNNING
    assert outbound.operations == ["send", "hold"]
    assert len(outbound.frames) == 1
    branch_steps = (left.executor.step_count, right.executor.step_count)
    same_tick = runtime.step()
    assert same_tick.wait_duration == pytest.approx(0.1)
    assert outbound.operations == ["send", "hold"]
    assert (left.executor.step_count, right.executor.step_count) == branch_steps

    clock.time = 0.2
    result = runtime.step()
    assert result.status is SemanticExecutionStatus.RUNNING
    assert outbound.operations == ["send", "hold", "send"]
    assert len(outbound.frames[1].commands) == 1
    assert outbound.frames[1].commands[0].target.target_id == "resource_1"
    assert (left.executor.step_count, right.executor.step_count) == branch_steps

    clock.time = 0.3
    result = runtime.step()

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert result.command_count == 2
    assert outbound.holds == 2
    assert outbound.operations == ["send", "hold", "send", "hold"]
    assert (
        result.task_state.get_articulation_joint_state("left_fixture", "joint")
        is not None
    )
    assert (
        result.task_state.get_articulation_joint_state("right_fixture", "joint")
        is not None
    )
    metadata = result.to_metadata()
    json.dumps(metadata, allow_nan=False, sort_keys=True)
    assert metadata["kind"] == "parallel_skill_result"
    assert list(metadata["branches"]) == ["left", "right"]
    assert metadata["elapsed_steps"] == 3


def test_deferred_command_waits_for_clock_after_padding_hold() -> None:
    left = _branch(
        "left",
        0,
        (
            _running_step(
                frame=_frame(0, (1.0, 1.0)),
                emit_hold=True,
            ),
        ),
    )
    right = _branch(
        "right",
        1,
        (_running_step(frame=_frame(1, (2.0, 2.0))),),
    )
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (left, right),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )

    runtime.start()
    padded = runtime.step()
    lane_steps = (left.executor.step_count, right.executor.step_count)
    same_tick = runtime.step()

    assert padded.status is SemanticExecutionStatus.RUNNING
    assert same_tick.wait_duration == pytest.approx(0.1)
    assert outbound.operations == ["hold"]
    assert not outbound.frames
    assert (left.executor.step_count, right.executor.step_count) == lane_steps

    clock.time = 0.1
    runtime.step()

    assert outbound.operations == ["hold", "send"]
    assert len(outbound.frames) == 1
    assert (left.executor.step_count, right.executor.step_count) == lane_steps


def test_completion_hold_waits_for_clock_after_accepted_command() -> None:
    left = _branch(
        "left",
        0,
        (
            _running_step(frame=_frame(0, (1.0, 1.0))),
            _completed_step(),
        ),
    )
    right = _branch(
        "right",
        1,
        (
            _running_step(frame=_frame(1, (2.0, 2.0))),
            _completed_step(),
        ),
    )
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (left, right),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )

    runtime.start()
    runtime.step()
    lane_steps = (left.executor.step_count, right.executor.step_count)
    same_tick = runtime.step()

    assert same_tick.status is SemanticExecutionStatus.RUNNING
    assert same_tick.wait_duration == pytest.approx(0.1)
    assert outbound.operations == ["send"]
    assert (left.executor.step_count, right.executor.step_count) == lane_steps

    clock.time = 0.1
    completed = runtime.step()

    assert completed.status is SemanticExecutionStatus.COMPLETED
    assert outbound.operations == ["send", "hold"]


def test_parallel_runtime_uses_runner_transport_timeouts() -> None:
    """Merged sends and safe stops share the selected preset runner policy."""
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (
            _branch("left", 0, (_running_step(frame=_frame(0, (1.0, 1.0))),)),
            _branch("right", 1, (_running_step(frame=_frame(1, (2.0, 2.0))),)),
        ),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
        runner_cfg=ExecutionRunnerCfg(
            command_timeout=0.25,
            safe_stop_timeout=0.75,
            hold_on_completion=False,
        ),
    )

    runtime.start()
    runtime.step()
    runtime.cancel("operator stop")

    assert outbound.timeouts == [
        ("send", pytest.approx(0.25)),
        ("cancel", pytest.approx(0.75)),
        ("hold", pytest.approx(0.75)),
    ]


def test_parallel_failure_safe_holds_when_completion_hold_is_disabled() -> None:
    """Failure policy always cancels and holds independently of success policy."""
    outbound = _OutboundSink()
    runtime = ParallelSemanticExecutor(
        (
            _branch("left", 0, (_running_step(frame=_frame(0, (1.0, 1.0))),)),
            _branch("right", 1, (_running_step(frame=_frame(1, (2.0, 2.0))),)),
        ),
        outbound,
        _Clock(),
        ParallelTimingPolicy(0.1),
        _RejectSafety(),
        timeout_steps=5,
        runner_cfg=ExecutionRunnerCfg(hold_on_completion=False),
    )

    runtime.start()
    result = runtime.step()

    assert result.status is SemanticExecutionStatus.FAILED
    assert outbound.operations == ["cancel", "hold"]


def test_parallel_completion_respects_disabled_completion_hold() -> None:
    """Successful completion does not synthesize a hold when policy disables it."""
    left = _branch(
        "left",
        0,
        (
            _running_step(frame=_frame(0, (1.0, 1.0))),
            _completed_step(),
        ),
        emit_terminal_hold=False,
    )
    right = _branch(
        "right",
        1,
        (
            _running_step(frame=_frame(1, (2.0, 2.0))),
            _completed_step(),
        ),
        emit_terminal_hold=False,
    )
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (left, right),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
        runner_cfg=ExecutionRunnerCfg(hold_on_completion=False),
    )

    runtime.start()
    runtime.step()
    clock.time = 0.1
    result = runtime.step()

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert outbound.operations == ["send"]


def test_parallel_minimum_cycle_time_limits_coordinator_cadence() -> None:
    """Coordinator dispatches no faster than the preset's minimum cycle time."""
    left = _branch(
        "left",
        0,
        (
            _running_step(frame=_frame(0, (1.0, 1.0))),
            _running_step(frame=_frame(0, (3.0, 3.0))),
        ),
    )
    right = _branch(
        "right",
        1,
        (
            _running_step(frame=_frame(1, (2.0, 2.0))),
            _running_step(frame=_frame(1, (4.0, 4.0))),
        ),
    )
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (left, right),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
        runner_cfg=ExecutionRunnerCfg(minimum_cycle_time=0.25),
    )

    runtime.start()
    first = runtime.step()
    clock.time = 0.1
    waiting = runtime.step()
    clock.time = 0.25
    runtime.step()

    assert first.wait_duration == pytest.approx(0.25)
    assert waiting.wait_duration == pytest.approx(0.15)
    assert len(outbound.frames) == 2


def test_parallel_runtime_fail_fast_is_row_local() -> None:
    left = _branch(
        "left",
        0,
        (
            _running_step(
                frame=_frame(0, (1.0, 1.0)),
                eligible=_mask(False, True),
                failure=_mask(True, False),
            ),
            _completed_step(
                success=_mask(False, True),
                failure=_mask(True, False),
            ),
        ),
    )
    right = _branch(
        "right",
        1,
        (
            _running_step(frame=_frame(1, (3.0, 3.0))),
            _completed_step(
                success=_mask(False, True),
            ),
        ),
    )
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (left, right),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )

    runtime.start()
    first = runtime.step()

    assert torch.equal(first.failure_mask, _mask(True, False))
    assert torch.equal(
        first.branch_results["right"].cancelled_mask,
        _mask(True, False),
    )
    assert torch.equal(outbound.frames[0].active_mask, _mask(False, True))

    clock.time = 0.1
    result = runtime.step()
    assert result.status is SemanticExecutionStatus.FAILED
    assert torch.equal(result.failure_mask, _mask(True, False))
    assert torch.equal(result.success_mask, _mask(False, True))


def test_parallel_failure_without_fresh_peer_frame_forces_masked_dispatch() -> None:
    left = _branch(
        "left",
        0,
        (
            _running_step(frame=_frame(0, (1.0, 1.0))),
            _running_step(
                eligible=_mask(False, True),
                failure=_mask(True, False),
            ),
        ),
    )
    right = _branch(
        "right",
        1,
        (
            _running_step(frame=_frame(1, (2.0, 2.0))),
            _running_step(wait_duration=0.1),
        ),
    )
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (left, right),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )

    runtime.start()
    runtime.step()
    assert torch.equal(outbound.frames[-1].active_mask, _mask(True, True))

    # The failure update has no fresh frame from either lane. The coordinator
    # still replays the last transaction with row 0 inactive.
    clock.time = 0.1
    runtime.step()
    assert len(outbound.frames) == 2
    assert torch.equal(outbound.frames[-1].active_mask, _mask(False, True))


def test_parallel_timeout_counts_completed_environment_steps() -> None:
    left = _branch("left", 0, (_running_step(wait_duration=0.1),))
    right = _branch("right", 1, (_running_step(wait_duration=0.1),))
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (left, right),
        _OutboundSink(),
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=1,
    )

    runtime.start()
    before_step = runtime.step()
    assert before_step.status is SemanticExecutionStatus.RUNNING
    assert before_step.elapsed_steps == 0

    clock.time = 0.1
    timed_out = runtime.step()
    assert timed_out.status is SemanticExecutionStatus.FAILED
    assert timed_out.elapsed_steps == 1
    assert torch.equal(timed_out.failure_mask, _mask(True, True))


def test_parallel_timeout_does_not_execute_deadline_tick() -> None:
    left_runtime_steps = (
        _running_step(frame=_frame(0, (1.0, 1.0)), wait_duration=0.1),
        _running_step(frame=_frame(0, (2.0, 2.0))),
    )
    right_runtime_steps = (
        _running_step(frame=_frame(1, (3.0, 3.0)), wait_duration=0.1),
        _running_step(frame=_frame(1, (4.0, 4.0))),
    )
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (
            _branch("left", 0, left_runtime_steps),
            _branch("right", 1, right_runtime_steps),
        ),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=1,
    )

    runtime.start()
    runtime.step()
    assert len(outbound.frames) == 1
    clock.time = 0.1
    result = runtime.step()

    assert result.status is SemanticExecutionStatus.FAILED
    assert len(outbound.frames) == 1
    assert outbound.cancels == 1
    assert outbound.holds == 1


def test_parallel_timeout_discards_frame_deferred_behind_completion_hold() -> None:
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (
            _branch(
                "left",
                0,
                (
                    _running_step(
                        frame=_frame(0, (1.0, 1.0)),
                        emit_hold=True,
                    ),
                ),
            ),
            _branch(
                "right",
                1,
                (_running_step(frame=_frame(1, (2.0, 2.0))),),
            ),
        ),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=1,
    )

    runtime.start()
    padded = runtime.step()
    assert padded.status is SemanticExecutionStatus.RUNNING
    assert outbound.operations == ["hold"]
    assert not outbound.frames

    clock.time = 0.1
    timed_out = runtime.step()

    assert timed_out.status is SemanticExecutionStatus.FAILED
    assert torch.equal(timed_out.failure_mask, _mask(True, True))
    assert not timed_out.success_mask.any()
    assert not outbound.frames
    assert outbound.operations == ["hold", "cancel", "hold"]


def test_parallel_cancel_discards_deferred_frame_and_covers_started_rows() -> None:
    outbound = _OutboundSink()
    runtime = ParallelSemanticExecutor(
        (
            _branch(
                "left",
                0,
                (
                    _running_step(
                        frame=_frame(0, (1.0, 1.0)),
                        emit_hold=True,
                    ),
                ),
            ),
            _branch(
                "right",
                1,
                (_running_step(frame=_frame(1, (2.0, 2.0))),),
            ),
        ),
        outbound,
        _Clock(),
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )

    runtime.start()
    runtime.step()
    cancelled = runtime.cancel("operator stop during padding")

    assert cancelled.status is SemanticExecutionStatus.CANCELLED
    assert torch.equal(cancelled.cancelled_mask, _mask(True, True))
    assert not cancelled.success_mask.any()
    assert not cancelled.failure_mask.any()
    assert not outbound.frames
    assert outbound.operations == ["hold", "cancel", "hold"]


def test_deferred_frame_validation_failure_does_not_advance_lanes() -> None:
    outbound = _OutboundSink()
    clock = _Clock()
    left = _branch(
        "left",
        0,
        (
            _running_step(
                frame=_frame(0, (1.0, 1.0)),
                emit_hold=True,
            ),
        ),
    )
    right = _branch(
        "right",
        1,
        (_running_step(frame=_frame(1, (2.0, 2.0))),),
    )
    runtime = ParallelSemanticExecutor(
        (left, right),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _RejectSafety(),
        timeout_steps=5,
    )

    runtime.start()
    runtime.step()
    assert isinstance(left.executor, _BranchRuntime)
    assert isinstance(right.executor, _BranchRuntime)
    steps_before_dispatch = (left.executor.step_count, right.executor.step_count)

    clock.time = 0.1
    failed = runtime.step()

    assert failed.status is SemanticExecutionStatus.FAILED
    assert (
        left.executor.step_count,
        right.executor.step_count,
    ) == steps_before_dispatch
    assert not outbound.frames
    assert outbound.operations == ["hold", "cancel", "hold"]


def test_terminal_fresh_frames_fail_closed_without_post_command_observation() -> None:
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (
            _branch(
                "left",
                0,
                (_completed_step(frame=_frame(0, (1.0, 1.0))),),
            ),
            _branch(
                "right",
                1,
                (_completed_step(frame=_frame(1, (2.0, 2.0))),),
            ),
        ),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )

    runtime.start()
    result = runtime.step()

    assert result.status is SemanticExecutionStatus.FAILED
    assert torch.equal(result.failure_mask, _mask(True, True))
    assert not result.success_mask.any()
    assert not outbound.frames
    assert outbound.operations == ["cancel", "hold"]
    assert "post-command observation" in (result.message or "")


def test_hold_aggregation_preserves_same_destination_distinct_fingerprints() -> None:
    target_a = JointPositionTarget("shared_arm", (0,))
    target_b = JointPositionTarget("shared_arm", (1,))
    lane_sink = ParallelLaneCommandSink()
    lane_sink.hold(
        (target_a, target_b),
        _context(TaskState.empty(2, "cpu")),
        timeout=1.0,
    )
    pending_targets, _ = lane_sink.hold_request
    assert tuple(target.address_fingerprint for target in pending_targets) == (
        target_a.address_fingerprint,
        target_b.address_fingerprint,
    )

    outbound = _OutboundSink()
    runtime = ParallelSemanticExecutor(
        (
            _branch(
                "left",
                0,
                (_running_step(hold_targets=(target_a, target_b)),),
            ),
            _branch("right", 2, (_running_step(wait_duration=0.1),)),
        ),
        outbound,
        _Clock(),
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )

    runtime.start()
    runtime.step()

    assert outbound.hold_targets == [("shared_arm", "shared_arm")]
    assert outbound.hold_fingerprints == [
        (target_a.address_fingerprint, target_b.address_fingerprint)
    ]


def test_parallel_lane_does_not_drop_prior_call_completion_hold() -> None:
    left_sink = ParallelLaneCommandSink()
    left = ParallelExecutorBranch(
        branch_id="left",
        calls=(
            RegisteredSemanticCall("test.left_first"),
            RegisteredSemanticCall("test.left_second"),
        ),
        claim=ResourceClaim(frozenset({"left"}), (0, 2)),
        executor=_BranchRuntime(
            (
                _running_step(
                    frame=_frame(0, (1.0, 1.0)),
                    emit_hold=True,
                ),
                _running_step(frame=_frame(2, (2.0, 2.0))),
                _completed_step(),
            ),
            left_sink,
        ),
        command_sink=left_sink,
    )
    right = _branch(
        "right",
        1,
        (
            _running_step(frame=_frame(1, (3.0, 3.0))),
            _running_step(frame=_frame(1, (4.0, 4.0))),
            _completed_step(),
        ),
    )
    outbound = _OutboundSink()
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (left, right),
        outbound,
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )

    runtime.start()
    runtime.step()
    result = runtime.result
    for step_index in range(1, 8):
        if result.terminal:
            break
        clock.time = step_index * 0.1
        result = runtime.step()

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert any("resource_0" in targets for targets in outbound.hold_targets)
    assert any("resource_2" in targets for targets in outbound.hold_targets)


def test_parallel_runtime_rejects_overlapping_claims_before_start() -> None:
    script = (_running_step(),)
    left = _branch("left", 0, script)
    right_sink = ParallelLaneCommandSink()
    right = ParallelExecutorBranch(
        branch_id="right",
        calls=(RegisteredSemanticCall("test.right"),),
        claim=ResourceClaim(frozenset({"different_name"}), (0,)),
        executor=_BranchRuntime(script, right_sink),
        command_sink=right_sink,
    )

    with pytest.raises(ValueError, match="overlapping resource claims"):
        ParallelSemanticExecutor(
            (left, right),
            _OutboundSink(),
            _Clock(),
            ParallelTimingPolicy(0.1),
            _AcceptSafety(),
            timeout_steps=5,
        )


def test_parallel_runtime_requires_equal_branch_barrier_state() -> None:
    changed = StateDelta(
        articulation_joint_updates={
            ("fixture", "joint"): ArticulationJointState(torch.ones(2, 1))
        }
    ).apply(TaskState.empty(2, "cpu"), _mask(True, True))

    with pytest.raises(ValueError, match="same verified TaskState"):
        ParallelSemanticExecutor(
            (
                _branch("left", 0, (_running_step(),)),
                _branch(
                    "right",
                    1,
                    (_running_step(),),
                    initial_state=changed,
                ),
            ),
            _OutboundSink(),
            _Clock(),
            ParallelTimingPolicy(0.1),
            _AcceptSafety(),
            timeout_steps=5,
        )


def test_terminal_targets_without_hold_context_fail_closed() -> None:
    clock = _Clock()
    runtime = ParallelSemanticExecutor(
        (
            _branch(
                "left",
                0,
                (
                    _running_step(frame=_frame(0, (1.0, 1.0))),
                    _completed_step(),
                ),
                emit_terminal_hold=False,
            ),
            _branch(
                "right",
                1,
                (
                    _running_step(frame=_frame(1, (2.0, 2.0))),
                    _completed_step(),
                ),
                emit_terminal_hold=False,
            ),
        ),
        _OutboundSink(),
        clock,
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )

    runtime.start()
    result = runtime.step()
    assert result.status is SemanticExecutionStatus.RUNNING
    clock.time = 0.1
    result = runtime.step()

    assert result.status is SemanticExecutionStatus.FAILED
    assert torch.equal(result.failure_mask, _mask(True, True))
    assert "no synchronized planning context" in (result.message or "")


@pytest.mark.parametrize(
    ("sink_kwargs", "expected_status"),
    [
        ({}, SemanticExecutionStatus.CANCELLED),
        ({"reject_cancel": True}, SemanticExecutionStatus.FAILED),
        ({"reject_hold": True}, SemanticExecutionStatus.FAILED),
    ],
)
def test_parallel_caller_cancel_checks_cancel_and_hold_acknowledgements(
    sink_kwargs: dict[str, bool],
    expected_status: SemanticExecutionStatus,
) -> None:
    outbound = _OutboundSink(**sink_kwargs)
    runtime = ParallelSemanticExecutor(
        (
            _branch("left", 0, (_running_step(frame=_frame(0, (1.0, 1.0))),)),
            _branch("right", 1, (_running_step(frame=_frame(1, (2.0, 2.0))),)),
        ),
        outbound,
        _Clock(),
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )
    runtime.start()
    runtime.step()

    result = runtime.cancel("operator stop")

    assert result.status is expected_status
    assert outbound.cancels == 1
    assert outbound.holds >= 1
    if expected_status is SemanticExecutionStatus.CANCELLED:
        assert torch.equal(result.cancelled_mask, _mask(True, True))
        assert not result.failure_mask.any()
    else:
        assert torch.equal(result.failure_mask, _mask(True, True))
        assert not result.cancelled_mask.any()


@pytest.mark.parametrize(
    ("safety", "sink"),
    [
        (_RejectSafety(), _OutboundSink()),
        (_AcceptSafety(), _OutboundSink(raise_send=True)),
    ],
)
def test_parallel_tick_exception_safe_stops_with_disjoint_failure_masks(
    safety: object,
    sink: _OutboundSink,
) -> None:
    runtime = ParallelSemanticExecutor(
        (
            _branch("left", 0, (_running_step(frame=_frame(0, (1.0, 1.0))),)),
            _branch("right", 1, (_running_step(frame=_frame(1, (2.0, 2.0))),)),
        ),
        sink,
        _Clock(),
        ParallelTimingPolicy(0.1),
        safety,
        timeout_steps=5,
    )
    runtime.start()

    result = runtime.step()

    assert result.status is SemanticExecutionStatus.FAILED
    assert torch.equal(result.failure_mask, _mask(True, True))
    assert not result.success_mask.any()
    assert not result.cancelled_mask.any()
    assert sink.cancels == 1
    assert sink.holds == 1


def test_cancel_preserves_verified_state_from_an_earlier_branch_call() -> None:
    changed = StateDelta(
        articulation_joint_updates={
            ("fixture", "joint"): ArticulationJointState(torch.ones(2, 1))
        }
    ).apply(TaskState.empty(2, "cpu"), _mask(True, True))
    runtime = ParallelSemanticExecutor(
        (
            _branch(
                "left",
                0,
                (
                    _running_step(
                        frame=_frame(0, (1.0, 1.0)),
                        task_state=changed,
                    ),
                ),
            ),
            _branch("right", 1, (_running_step(frame=_frame(1, (2.0, 2.0))),)),
        ),
        _OutboundSink(),
        _Clock(),
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )
    runtime.start()
    runtime.step()

    result = runtime.cancel()

    assert result.status is SemanticExecutionStatus.CANCELLED
    assert (
        result.task_state.get_articulation_joint_state("fixture", "joint") is not None
    )


def test_disjoint_intrinsic_rows_still_conflict_on_same_unpartitioned_key() -> None:
    initial = TaskState.empty(2, "cpu")
    left_state = StateDelta(
        articulation_joint_updates={
            ("fixture", "joint"): ArticulationJointState(torch.ones(2, 1))
        }
    ).apply(initial, _mask(True, False))
    right_state = StateDelta(
        articulation_joint_updates={
            ("fixture", "joint"): ArticulationJointState(torch.full((2, 1), 2.0))
        }
    ).apply(initial, _mask(False, True))
    runtime = ParallelSemanticExecutor(
        (
            _branch(
                "left",
                0,
                (_running_step(frame=_frame(0, (1.0, 1.0)), task_state=left_state),),
            ),
            _branch(
                "right",
                1,
                (_running_step(frame=_frame(1, (2.0, 2.0)), task_state=right_state),),
            ),
        ),
        _OutboundSink(),
        _Clock(),
        ParallelTimingPolicy(0.1),
        _AcceptSafety(),
        timeout_steps=5,
    )
    runtime.start()
    runtime.step()

    result = runtime.cancel()

    assert result.status is SemanticExecutionStatus.FAILED
    assert torch.equal(result.failure_mask, _mask(True, True))
    assert not result.cancelled_mask.any()


__all__: list[str] = []
