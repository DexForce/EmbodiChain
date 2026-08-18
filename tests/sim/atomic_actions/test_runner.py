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

"""Tests for controller-independent atomic-action execution scheduling."""

from __future__ import annotations

from collections import deque
from typing import ClassVar
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    ActionOptions,
    ActionPlan,
    Affordance,
    AtomicAction,
    AtomicActionEngine,
    CommandAcknowledgement,
    CommandAckStatus,
    CommandOperation,
    EndEffectorPoseGoal,
    ExecutionEventKind,
    ExecutionRunner,
    ExecutionRunnerCfg,
    HeldObjectState,
    JointCommand,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionRequest,
    RobotObservation,
    RunnerStatus,
    SceneSnapshot,
    StateDelta,
    TaskState,
    TimedTrajectory,
)

BATCH_SIZE = 1
ROBOT_DOF = 2
FIRST_INTERVAL = 0.1
SECOND_INTERVAL = 0.2
MINIMUM_CYCLE_TIME = 0.01
TARGET_POSITION = 1.0


class FakeClock:
    """Deterministic clock used by non-blocking runner tests."""

    def __init__(self) -> None:
        self.time = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        """Return deterministic time."""
        return self.time

    def sleep(self, duration: float) -> None:
        """Advance deterministic time."""
        self.sleeps.append(duration)
        self.time += duration

    def advance(self, duration: float) -> None:
        """Advance time outside the runner's blocking loop."""
        self.time += duration


class FakeObservationProvider:
    """In-memory robot observation provider."""

    def __init__(self, clock: FakeClock, batch_size: int = BATCH_SIZE) -> None:
        self.clock = clock
        self.qpos = torch.zeros(batch_size, ROBOT_DOF)
        self.fail = False

    def observe(self, task_state: TaskState) -> PlanningContext:
        """Return the current in-memory robot state."""
        if self.fail:
            raise RuntimeError("observation unavailable")
        return PlanningContext(
            robot=RobotObservation(
                timestamp=self.clock.now(),
                qpos=self.qpos,
                qvel=torch.zeros_like(self.qpos),
            ),
            task=task_state,
            scene=SceneSnapshot(timestamp=self.clock.now(), version=0),
            env_ids=torch.arange(self.qpos.shape[0], dtype=torch.long),
        )


class FakeCommandSink:
    """Recording command sink with configurable acknowledgements and tracking."""

    def __init__(self, provider: FakeObservationProvider) -> None:
        self.provider = provider
        self.send_statuses: deque[CommandAckStatus] = deque()
        self.follow_commands: deque[bool] = deque()
        self.sent: list[JointCommand] = []
        self.send_times: list[float] = []
        self.held: list[JointCommand] = []
        self.cancel_count = 0

    def send(
        self,
        command: JointCommand,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Record an active command and optionally update observed qpos."""
        self.sent.append(command)
        self.send_times.append(self.provider.clock.now())
        status = (
            self.send_statuses.popleft()
            if self.send_statuses
            else CommandAckStatus.ACCEPTED
        )
        follows = self.follow_commands.popleft() if self.follow_commands else True
        if status is CommandAckStatus.ACCEPTED and follows:
            self.provider.qpos = command.positions.clone()
        return CommandAcknowledgement(status)

    def hold(
        self,
        command: JointCommand,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Record and apply a hold command."""
        self.held.append(command)
        self.provider.qpos = command.positions.clone()
        return CommandAcknowledgement.accepted_ack()

    def cancel(self, *, timeout: float) -> CommandAcknowledgement:
        """Record controller cancellation."""
        self.cancel_count += 1
        return CommandAcknowledgement.accepted_ack()


class TimedAction(AtomicAction[EndEffectorPoseGoal, ActionOptions]):
    """Test action with explicit non-uniform command intervals."""

    skill_id: ClassVar[str] = "timed"
    GoalType: ClassVar[type] = EndEffectorPoseGoal
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(self, *, with_effect: bool = False) -> None:
        super().__init__()
        self.with_effect = with_effect
        self.plan_count = 0

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan three samples with intervals 0.1 s and 0.2 s."""
        goal = self.require_goal(request)
        self.plan_count += 1
        assert isinstance(goal.xpos, torch.Tensor)
        target_value = float(goal.xpos[0, 3])
        target = torch.full_like(context.robot.qpos, target_value)
        midpoint = torch.lerp(context.robot.qpos, target, 0.5)
        positions = torch.stack([context.robot.qpos, midpoint, target], dim=1)
        dt = torch.tensor(
            [[0.0, FIRST_INTERVAL, SECOND_INTERVAL]],
            dtype=torch.float32,
        ).repeat(context.batch_size, 1)
        if context.batch_size > 1:
            dt[1, 1:] *= 2.0
        trajectory = TimedTrajectory.from_positions(
            positions,
            env_ids=context.env_ids,
            control_dt=request.motion_policy.control_dt,
            dt=dt,
        )
        effects = StateDelta()
        if self.with_effect:
            semantics = ObjectSemantics(
                affordance=Affordance(), geometry={}, label="runner-object"
            )
            held = HeldObjectState(
                semantics=semantics,
                object_to_eef=torch.eye(4),
                grasp_xpos=torch.eye(4),
            )
            effects = StateDelta(held_object_updates={"arm": held})
        return self.build_plan(
            request,
            context,
            success=True,
            trajectory=trajectory,
            expected_effects=effects,
        )


def _make_runner(
    *,
    with_effect: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tuple[
    ExecutionRunner,
    FakeClock,
    FakeObservationProvider,
    FakeCommandSink,
    TimedAction,
]:
    clock = FakeClock()
    provider = FakeObservationProvider(clock, batch_size)
    sink = FakeCommandSink(provider)
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = ROBOT_DOF
    robot.control_parts = {"arm": object()}
    robot.get_qpos.return_value = torch.zeros(batch_size, ROBOT_DOF)
    robot.get_joint_ids.return_value = list(range(ROBOT_DOF))
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub"
    action = TimedAction(with_effect=with_effect)
    engine = AtomicActionEngine(generator)
    engine.register(action)
    initial_task = TaskState.empty(batch_size, "cpu")
    initial_context = provider.observe(initial_task)
    goal_pose = torch.eye(4)
    goal_pose[0, 3] = TARGET_POSITION
    invocation = ActionInvocation(
        skill_id="timed",
        goal=EndEffectorPoseGoal(goal_pose),
        binding=ActionBinding(manipulators={"primary": "arm"}),
        motion_policy=MotionPolicy(sample_count=3, control_dt=FIRST_INTERVAL),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            tracking_error_threshold=0.05,
            action_timeout=10.0,
        ),
    )
    session = engine.start((invocation,), initial_context)
    runner = ExecutionRunner(
        session,
        provider,
        sink,
        clock=clock,
        cfg=ExecutionRunnerCfg(minimum_cycle_time=MINIMUM_CYCLE_TIME),
    )
    return runner, clock, provider, sink, action


def test_runner_dispatches_only_when_timed_waypoint_is_due() -> None:
    runner, clock, _, sink, _ = _make_runner()

    first = runner.step()
    early = runner.step()
    clock.advance(FIRST_INTERVAL)
    second = runner.step()
    clock.advance(SECOND_INTERVAL)
    third = runner.step()

    assert first.command_count == 1
    assert first.wait_duration == pytest.approx(FIRST_INTERVAL)
    assert early.is_waiting
    assert len(sink.sent) == 3
    assert sink.send_times == pytest.approx(
        [0.0, FIRST_INTERVAL, FIRST_INTERVAL + SECOND_INTERVAL]
    )
    assert second.command_count == 2
    assert second.wait_duration == pytest.approx(SECOND_INTERVAL)
    assert third.command_count == 3
    assert third.wait_duration == pytest.approx(SECOND_INTERVAL)


def test_session_active_trajectory_returns_an_owned_snapshot() -> None:
    runner, _, _, _, _ = _make_runner()

    trajectory = runner.session.active_trajectory
    trajectory.positions.fill_(-1.0)

    assert torch.all(runner.session.active_trajectory.positions >= 0.0)


def test_runner_uses_the_longest_active_batch_interval_as_a_barrier() -> None:
    runner, clock, _, _, _ = _make_runner(batch_size=2)

    first = runner.step()
    clock.advance(2.0 * FIRST_INTERVAL)
    second = runner.step()

    assert first.wait_duration == pytest.approx(2.0 * FIRST_INTERVAL)
    assert second.wait_duration == pytest.approx(2.0 * SECOND_INTERVAL)


def test_runner_completes_and_holds_after_last_command_settles() -> None:
    runner, clock, _, sink, _ = _make_runner()

    runner.step()
    clock.advance(FIRST_INTERVAL)
    runner.step()
    clock.advance(SECOND_INTERVAL)
    runner.step()
    clock.advance(SECOND_INTERVAL)
    completed = runner.step()

    assert completed.status is RunnerStatus.COMPLETED
    assert completed.command_count == 3
    assert [item.operation for item in completed.dispatches] == [CommandOperation.HOLD]
    assert len(sink.held) == 1


@pytest.mark.parametrize(
    "status",
    [CommandAckStatus.REJECTED, CommandAckStatus.TIMED_OUT],
)
def test_runner_safely_stops_when_command_is_not_accepted(
    status: CommandAckStatus,
) -> None:
    runner, _, _, sink, _ = _make_runner()
    sink.send_statuses.append(status)

    failed = runner.step()

    assert failed.status is RunnerStatus.FAILED
    assert [item.operation for item in failed.dispatches] == [
        CommandOperation.SEND,
        CommandOperation.CANCEL,
        CommandOperation.HOLD,
    ]
    assert sink.cancel_count == 1
    assert failed.message is not None and status.value in failed.message


def test_runner_cancel_performs_cancel_then_hold() -> None:
    runner, _, _, sink, _ = _make_runner()

    cancelled = runner.cancel("operator stop")
    repeated = runner.step()

    assert cancelled.status is RunnerStatus.CANCELLED
    assert [item.operation for item in cancelled.dispatches] == [
        CommandOperation.CANCEL,
        CommandOperation.HOLD,
    ]
    assert cancelled.message == "operator stop"
    assert repeated.status is RunnerStatus.CANCELLED
    assert repeated.dispatches == ()
    assert sink.cancel_count == 1


def test_runner_replans_from_observation_after_tracking_error() -> None:
    runner, clock, _, sink, action = _make_runner()
    sink.follow_commands.extend([True, False, True])

    runner.step()
    clock.advance(FIRST_INTERVAL)
    runner.step()
    clock.advance(SECOND_INTERVAL)
    recovered = runner.step()

    assert action.plan_count == 2
    assert recovered.tick is not None
    event_kinds = {event.kind for event in recovered.tick.events}
    assert ExecutionEventKind.TRACKING_ERROR in event_kinds
    assert ExecutionEventKind.REPLANNED in event_kinds
    assert recovered.status is RunnerStatus.RUNNING


def test_runner_surfaces_explicit_invocation_revision() -> None:
    runner, _, _, _, action = _make_runner()
    revised_pose = torch.eye(4)
    revised_pose[0, 3] = 2.0 * TARGET_POSITION
    revised = ActionInvocation(
        skill_id="timed",
        goal=EndEffectorPoseGoal(revised_pose),
        binding=ActionBinding(manipulators={"primary": "arm"}),
        motion_policy=MotionPolicy(sample_count=3, control_dt=FIRST_INTERVAL),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            tracking_error_threshold=0.05,
            action_timeout=10.0,
        ),
        revision=1,
    )

    runner.session.revise_current(revised)
    result = runner.step()

    assert action.plan_count == 2
    assert result.tick is not None
    assert any(
        event.kind is ExecutionEventKind.INVOCATION_REVISED
        and event.invocation_revision == 1
        for event in result.tick.events
    )


def test_runner_fails_safely_when_observation_provider_raises() -> None:
    runner, _, provider, sink, _ = _make_runner()
    provider.fail = True

    failed = runner.step()

    assert failed.status is RunnerStatus.FAILED
    assert [item.operation for item in failed.dispatches] == [
        CommandOperation.CANCEL,
        CommandOperation.HOLD,
    ]
    assert len(sink.held) == 1
    assert sink.cancel_count == 1
    assert failed.message is not None and "observation unavailable" in failed.message


def test_blocking_runner_uses_clock_and_completes() -> None:
    runner, clock, _, _, _ = _make_runner()

    completed = runner.run_until_blocked()

    assert completed.status is RunnerStatus.COMPLETED
    assert completed.command_count == 3
    assert clock.sleeps == pytest.approx(
        [FIRST_INTERVAL, SECOND_INTERVAL, SECOND_INTERVAL]
    )


def test_blocking_runner_safely_stops_when_the_clock_fails() -> None:
    runner, clock, _, sink, _ = _make_runner()

    def fail_sleep(duration: float) -> None:
        raise RuntimeError("clock backend unavailable")

    clock.sleep = fail_sleep

    failed = runner.run_until_blocked()

    assert failed.status is RunnerStatus.FAILED
    assert [item.operation for item in failed.dispatches[-2:]] == [
        CommandOperation.CANCEL,
        CommandOperation.HOLD,
    ]
    assert sink.cancel_count == 1
    assert failed.message is not None and "clock backend unavailable" in failed.message


def test_blocking_runner_verifies_effect_before_committing_task_state() -> None:
    runner, _, _, _, _ = _make_runner(with_effect=True)

    completed = runner.run_until_blocked(
        effect_verifier=lambda context, tick: torch.ones(
            context.batch_size, dtype=torch.bool
        )
    )

    assert completed.status is RunnerStatus.COMPLETED
    assert completed.tick is not None
    assert completed.tick.task_state.get_held_object("arm") is not None


def test_blocking_runner_resumes_a_stored_effect_verification_boundary() -> None:
    runner, _, _, _, _ = _make_runner(with_effect=True)

    blocked = runner.run_until_blocked()

    assert blocked.status is RunnerStatus.RUNNING
    assert blocked.tick is not None
    assert blocked.tick.pending_effect is not None
    assert runner.effect_verification_pending is True

    completed = runner.run_until_blocked(
        effect_verifier=lambda context, tick: torch.ones(
            context.batch_size, dtype=torch.bool
        )
    )

    assert runner.effect_verification_pending is False
    assert completed.status is RunnerStatus.COMPLETED
    assert completed.tick is not None
    assert completed.tick.task_state.get_held_object("arm") is not None
