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
    EffectVerificationRequest,
    EffectVerificationResult,
    ExecutionEventKind,
    ExecutionRunner,
    ExecutionRunnerCfg,
    HeldObjectGuardRequest,
    HeldObjectState,
    JOINT_POSITION_CAPABILITY,
    JointPositionPayload,
    JointPositionTrackingMetric,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    PlanningContextTrackingFeedbackProvider,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionRequest,
    RobotObservation,
    RuntimeCommandFrame,
    RuntimeEndpointTarget,
    RunnerStatus,
    RunnerStep,
    SceneSnapshot,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    StateDelta,
    TaskState,
    TimedTrajectory,
    TrackingEvaluation,
    TrackingEvaluatorRegistry,
    TrackingFeedbackBatch,
    TrackingFeedbackProviderRegistry,
    TrackingFeedbackSourceRef,
    TrackingMetricCfg,
    TrackingRuntime,
    TrackingState,
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
        self.sent: list[RuntimeCommandFrame] = []
        self.send_times: list[float] = []
        self.held: list[tuple[tuple[RuntimeEndpointTarget, ...], PlanningContext]] = []
        self.cancelled: list[tuple[RuntimeEndpointTarget, ...]] = []
        self.cancel_count = 0

    def send(
        self,
        command: RuntimeCommandFrame,
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
            positions = self.provider.qpos.clone()
            for endpoint_command in command.commands:
                target = endpoint_command.target
                payload = endpoint_command.payload
                assert isinstance(target, JointPositionTarget)
                assert isinstance(payload, JointPositionPayload)
                joint_ids = list(target.joint_ids)
                positions[:, joint_ids] = torch.where(
                    command.active_mask[:, None],
                    payload.positions,
                    positions[:, joint_ids],
                )
            self.provider.qpos = positions
        return CommandAcknowledgement(status)

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Record targets and apply the supplied observed-state hold."""
        self.held.append((tuple(targets), context))
        self.provider.qpos = context.robot.qpos.clone()
        return CommandAcknowledgement.accepted_ack()

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Record controller cancellation."""
        self.cancelled.append(tuple(targets))
        self.cancel_count += 1
        return CommandAcknowledgement.accepted_ack()


class RaisingFeedbackProvider:
    """Built-in-source replacement that simulates a provider failure."""

    provider_id = "planning_context.robot"
    revision = "1"

    def observe(
        self,
        source: TrackingFeedbackSourceRef,
        context: PlanningContext,
    ) -> TrackingFeedbackBatch:
        """Raise instead of returning required feedback."""
        del source, context
        raise RuntimeError("provider unavailable")


class RaisingJointTrackingEvaluator:
    """Joint evaluator replacement that simulates an evaluation failure."""

    metric_id = JointPositionTrackingMetric.metric_id
    revision = JointPositionTrackingMetric.revision
    metric_type = JointPositionTrackingMetric

    def evaluate(
        self,
        desired: TrackingState,
        observed: TrackingState,
        valid_mask: torch.Tensor,
        metric: TrackingMetricCfg,
    ) -> TrackingEvaluation:
        """Raise instead of evaluating required feedback."""
        del desired, observed, valid_mask, metric
        raise RuntimeError("evaluator unavailable")


class MaskedFeedbackProvider(PlanningContextTrackingFeedbackProvider):
    """Context provider exposing a deterministic per-row validity mask."""

    def __init__(self, valid_mask: tuple[bool, ...]) -> None:
        self.valid_mask = valid_mask

    def observe(
        self,
        source: TrackingFeedbackSourceRef,
        context: PlanningContext,
    ) -> TrackingFeedbackBatch:
        """Return built-in feedback with selected rows marked invalid."""
        feedback = super().observe(source, context)
        return TrackingFeedbackBatch(
            source=feedback.source,
            state=feedback.state,
            valid_mask=torch.tensor(
                self.valid_mask,
                dtype=torch.bool,
                device=feedback.state.device,
            ),
            timestamp=feedback.timestamp,
        )


class TimedAction(AtomicAction[EndEffectorPoseGoal, ActionOptions]):
    """Test action with explicit non-uniform command intervals."""

    skill_id: ClassVar[str] = "timed"
    GoalType: ClassVar[type] = EndEffectorPoseGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(
                    SkillEndpointRequirement(
                        endpoint_id="motion",
                        capabilities=frozenset({JOINT_POSITION_CAPABILITY}),
                    ),
                ),
            ),
        )
    )

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


def _timed_action_binding(action: TimedAction) -> ActionBinding:
    """Bind the timed action's generic motion endpoint to the fake arm."""
    return action.planning_services.bind_control_parts(
        TimedAction.binding_contract,
        {"primary": {"motion": "arm"}},
    )


def _make_runner(
    *,
    with_effect: bool = False,
    batch_size: int = BATCH_SIZE,
    control_joint_ids: tuple[int, ...] | None = None,
    max_action_retries: int = 2,
    action_timeout: float = 10.0,
    tracking_runtime: TrackingRuntime | None = None,
    hold_on_completion: bool = True,
    hold_during_effect_verification: bool = True,
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
    robot.get_joint_ids.return_value = list(
        range(ROBOT_DOF) if control_joint_ids is None else control_joint_ids
    )
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub"
    action = TimedAction(with_effect=with_effect)
    engine = AtomicActionEngine(generator, tracking_runtime=tracking_runtime)
    engine.register(action)
    initial_task = TaskState.empty(batch_size, "cpu")
    initial_context = provider.observe(initial_task)
    goal_pose = torch.eye(4)
    goal_pose[0, 3] = TARGET_POSITION
    invocation = ActionInvocation(
        skill_id="timed",
        goal=EndEffectorPoseGoal(goal_pose),
        binding=_timed_action_binding(action),
        motion_policy=MotionPolicy(sample_count=3, control_dt=FIRST_INTERVAL),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            max_action_retries=max_action_retries,
            action_timeout=action_timeout,
        ),
    )
    session = engine.start((invocation,), initial_context)
    runner = ExecutionRunner(
        session,
        provider,
        sink,
        clock=clock,
        cfg=ExecutionRunnerCfg(
            minimum_cycle_time=MINIMUM_CYCLE_TIME,
            hold_on_completion=hold_on_completion,
            hold_during_effect_verification=hold_during_effect_verification,
        ),
    )
    return runner, clock, provider, sink, action


def _successful_effect_result(
    context: PlanningContext,
    request: EffectVerificationRequest,
) -> EffectVerificationResult:
    """Correlate a successful result with the pending effect boundary."""
    return EffectVerificationResult(
        verification_id=request.verification_id,
        success_mask=torch.ones(
            context.batch_size,
            dtype=torch.bool,
            device=context.robot.qpos.device,
        ),
        failure_mask=torch.zeros(
            context.batch_size,
            dtype=torch.bool,
            device=context.robot.qpos.device,
        ),
    )


def _unresolved_effect_result(
    context: PlanningContext,
    request: EffectVerificationRequest,
) -> EffectVerificationResult:
    """Keep every row pending at the current effect boundary."""
    return EffectVerificationResult(
        verification_id=request.verification_id,
        success_mask=torch.zeros(
            context.batch_size,
            dtype=torch.bool,
            device=context.robot.qpos.device,
        ),
        failure_mask=torch.zeros(
            context.batch_size,
            dtype=torch.bool,
            device=context.robot.qpos.device,
        ),
    )


def test_joint_feedback_ignores_motion_outside_bound_endpoint() -> None:
    runner, clock, provider, sink, action = _make_runner(control_joint_ids=(0,))

    runner.step()
    provider.qpos[:, 1] = 42.0
    clock.advance(FIRST_INTERVAL)
    second = runner.step()
    clock.advance(SECOND_INTERVAL)
    runner.step()
    clock.advance(SECOND_INTERVAL)
    completed = runner.step()

    assert action.plan_count == 1
    assert len(sink.sent) == 3
    assert not any(
        event.kind is ExecutionEventKind.TRACKING_DIVERGED
        for step in (second, completed)
        if step.tick is not None
        for event in step.tick.events
    )
    assert completed.status is RunnerStatus.COMPLETED
    assert provider.qpos[0, 1].item() == 42.0


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


def test_runner_calls_held_object_guard_with_fresh_command_phase() -> None:
    runner, _, _, sink, _ = _make_runner()
    observed: list[tuple[float, HeldObjectGuardRequest]] = []

    def verifier(
        context: PlanningContext,
        request: HeldObjectGuardRequest,
    ) -> None:
        observed.append((context.robot.timestamp, request))
        return None

    first = runner.step(held_object_guard_verifier=verifier)

    assert first.status is RunnerStatus.RUNNING
    assert len(sink.sent) == 1
    assert len(observed) == 1
    timestamp, request = observed[0]
    assert timestamp == 0.0
    assert request.verification_id == 0
    assert request.segment_name == "timed"
    assert request.attempt_generation == 0
    assert request.invocation_index == 0
    assert request.next_waypoint_index == 0


def test_runner_guard_exception_performs_cancel_then_observed_hold() -> None:
    runner, clock, _, sink, _ = _make_runner()
    runner.step()
    clock.advance(FIRST_INTERVAL)

    def verifier(
        context: PlanningContext,
        request: HeldObjectGuardRequest,
    ) -> None:
        del context, request
        raise RuntimeError("guard evidence unavailable")

    failed = runner.step(held_object_guard_verifier=verifier)

    assert failed.status is RunnerStatus.FAILED
    assert [dispatch.operation for dispatch in failed.dispatches] == [
        CommandOperation.CANCEL,
        CommandOperation.HOLD,
    ]
    assert sink.cancel_count == 1
    assert [target.target_id for target in sink.cancelled[0]] == ["arm"]
    assert failed.message is not None
    assert "guard evidence unavailable" in failed.message


def test_runner_dispatches_transport_neutral_endpoint_frames() -> None:
    runner, _, _, sink, _ = _make_runner()

    runner.step()

    frame = sink.sent[0]
    assert isinstance(frame, RuntimeCommandFrame)
    assert len(frame.commands) == 1
    endpoint_command = frame.commands[0]
    assert isinstance(endpoint_command.target, JointPositionTarget)
    assert endpoint_command.target.transport_id == "robot.joint_position"
    assert endpoint_command.target.target_id == "arm"
    assert endpoint_command.target.joint_ids == (0, 1)
    assert isinstance(endpoint_command.payload, JointPositionPayload)
    assert endpoint_command.payload.transport_id == endpoint_command.target.transport_id


def test_session_active_commands_return_an_owned_endpoint_snapshot() -> None:
    runner, _, _, _, _ = _make_runner()

    commands = runner.session.active_commands
    payload = commands.frames[0].commands[0].payload
    assert isinstance(payload, JointPositionPayload)
    payload.positions.fill_(-1.0)

    current_payload = runner.session.active_commands.frames[0].commands[0].payload
    assert isinstance(current_payload, JointPositionPayload)
    assert torch.all(current_payload.positions >= 0.0)


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
    held_targets, hold_context = sink.held[0]
    assert [(target.transport_id, target.target_id) for target in held_targets] == [
        ("robot.joint_position", "arm")
    ]
    assert torch.equal(hold_context.robot.qpos, sink.provider.qpos)


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
    assert [target.target_id for target in sink.cancelled[0]] == ["arm"]
    assert [target.target_id for target in sink.held[0][0]] == ["arm"]
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
    assert sink.cancelled == [()]
    assert sink.held[0][0] == ()


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
    assert ExecutionEventKind.TRACKING_DIVERGED in event_kinds
    assert ExecutionEventKind.REPLANNED in event_kinds
    assert recovered.status is RunnerStatus.RUNNING


@pytest.mark.parametrize("failure_kind", ["provider", "evaluator"])
def test_runner_fails_closed_when_required_tracking_runtime_raises(
    failure_kind: str,
) -> None:
    builtins = TrackingRuntime.with_builtins()
    if failure_kind == "provider":
        tracking_runtime = TrackingRuntime(
            TrackingFeedbackProviderRegistry((RaisingFeedbackProvider(),)),
            builtins.projectors,
            builtins.evaluators,
        )
    else:
        tracking_runtime = TrackingRuntime(
            builtins.providers,
            builtins.projectors,
            TrackingEvaluatorRegistry((RaisingJointTrackingEvaluator(),)),
        )
    runner, clock, _, sink, action = _make_runner(tracking_runtime=tracking_runtime)

    runner.step()
    clock.advance(FIRST_INTERVAL)
    failed = runner.step()

    assert failed.status is RunnerStatus.FAILED
    assert failed.tick is not None
    event_kinds = {event.kind for event in failed.tick.events}
    assert ExecutionEventKind.TRACKING_FEEDBACK_FAILED in event_kinds
    assert ExecutionEventKind.REPLANNED not in event_kinds
    assert action.plan_count == 1
    assert sink.cancel_count == 1


def test_runner_deactivates_only_rows_with_invalid_required_feedback() -> None:
    builtins = TrackingRuntime.with_builtins()
    tracking_runtime = TrackingRuntime(
        TrackingFeedbackProviderRegistry((MaskedFeedbackProvider((True, False)),)),
        builtins.projectors,
        builtins.evaluators,
    )
    runner, clock, _, _, _ = _make_runner(
        batch_size=2,
        tracking_runtime=tracking_runtime,
    )

    runner.step()
    clock.advance(2.0 * FIRST_INTERVAL)
    partial = runner.step()

    assert partial.status is RunnerStatus.RUNNING
    assert partial.tick is not None
    assert partial.tick.command is not None
    assert partial.tick.command.active_mask.tolist() == [True, False]
    feedback_failure = next(
        event
        for event in partial.tick.events
        if event.kind is ExecutionEventKind.TRACKING_FEEDBACK_FAILED
    )
    assert feedback_failure.env_mask.tolist() == [False, True]


def test_runner_maintains_final_target_while_terminal_acceptance_is_pending() -> None:
    runner, clock, _, sink, action = _make_runner()
    sink.follow_commands.extend([True, True, False])

    runner.step()
    clock.advance(FIRST_INTERVAL)
    runner.step()
    clock.advance(SECOND_INTERVAL)
    runner.step()
    final_command = sink.sent[-1]

    clock.advance(SECOND_INTERVAL)
    settling = runner.step()

    assert action.plan_count == 1
    assert settling.status is RunnerStatus.RUNNING
    assert settling.tick is not None
    assert settling.tick.command is not None
    assert len(sink.sent) == 4
    assert sink.sent[-1] is settling.tick.command
    assert torch.equal(sink.sent[-1].active_mask, final_command.active_mask)
    final_payload = final_command.commands[0].payload
    settling_payload = sink.sent[-1].commands[0].payload
    assert isinstance(final_payload, JointPositionPayload)
    assert isinstance(settling_payload, JointPositionPayload)
    assert torch.equal(settling_payload.positions, final_payload.positions)
    event_kinds = {event.kind for event in settling.tick.events}
    assert ExecutionEventKind.TERMINAL_ACCEPTANCE_PENDING in event_kinds
    assert ExecutionEventKind.REPLANNED not in event_kinds


def test_terminal_settle_reemits_final_target_only_for_pending_rows() -> None:
    runner, clock, provider, sink, action = _make_runner(batch_size=2)

    runner.step()
    clock.advance(2.0 * FIRST_INTERVAL)
    runner.step()
    clock.advance(2.0 * SECOND_INTERVAL)
    runner.step()
    provider.qpos[1].zero_()

    clock.advance(2.0 * SECOND_INTERVAL)
    settling = runner.step()

    assert action.plan_count == 1
    assert settling.status is RunnerStatus.RUNNING
    assert settling.tick is not None
    assert settling.tick.command is not None
    assert settling.tick.command.active_mask.tolist() == [False, True]
    pending = next(
        event
        for event in settling.tick.events
        if event.kind is ExecutionEventKind.TERMINAL_ACCEPTANCE_PENDING
    )
    assert pending.env_mask.tolist() == [False, True]
    assert not any(
        event.kind is ExecutionEventKind.REPLANNED for event in settling.tick.events
    )


def test_runner_revision_waits_for_deadline_and_plans_from_fresh_observation() -> None:
    runner, clock, provider, sink, action = _make_runner()
    first = runner.step()
    assert first.wait_duration == pytest.approx(FIRST_INTERVAL)

    revised_pose = torch.eye(4)
    revised_pose[0, 3] = 2.0 * TARGET_POSITION
    revised = ActionInvocation(
        skill_id="timed",
        goal=EndEffectorPoseGoal(revised_pose),
        binding=_timed_action_binding(action),
        motion_policy=MotionPolicy(sample_count=3, control_dt=FIRST_INTERVAL),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            action_timeout=10.0,
        ),
        revision=1,
    )

    runner.revise_current(revised)
    provider.qpos.fill_(0.4)
    waiting = runner.step()

    assert waiting.is_waiting is True
    assert action.plan_count == 1
    assert sink.send_times == [0.0]

    clock.advance(FIRST_INTERVAL)
    result = runner.step()

    assert action.plan_count == 2
    assert result.command_count == 2
    assert sink.send_times == pytest.approx([0.0, FIRST_INTERVAL])
    assert result.tick is not None
    revised_payload = result.tick.command.commands[0].payload
    assert isinstance(revised_payload, JointPositionPayload)
    assert torch.allclose(revised_payload.positions, torch.full((1, 2), 0.4))
    assert any(
        event.kind is ExecutionEventKind.INVOCATION_REVISED
        and event.invocation_revision == 1
        for event in result.tick.events
    )


def test_runner_revision_rejects_pending_effect_verification() -> None:
    runner, _, _, _, action = _make_runner(with_effect=True)
    blocked = runner.run_until_blocked()
    assert blocked.tick is not None
    assert blocked.tick.pending_effect is not None
    assert runner.effect_verification_pending is True

    revised_pose = torch.eye(4)
    revised_pose[0, 3] = 2.0 * TARGET_POSITION
    revised = ActionInvocation(
        skill_id="timed",
        goal=EndEffectorPoseGoal(revised_pose),
        binding=_timed_action_binding(action),
        motion_policy=MotionPolicy(sample_count=3, control_dt=FIRST_INTERVAL),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            action_timeout=10.0,
        ),
        revision=1,
    )

    with pytest.raises(RuntimeError, match="physical-effect resolution"):
        runner.revise_current(revised)

    assert runner.effect_verification_pending is True
    completed = runner.run_until_blocked(
        effect_verifier=_successful_effect_result,
    )
    assert completed.status is RunnerStatus.COMPLETED


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
    assert sink.cancelled == [()]
    assert sink.held[0][0] == ()
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
        effect_verifier=_successful_effect_result,
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
        effect_verifier=_successful_effect_result,
    )

    assert runner.effect_verification_pending is False
    assert completed.status is RunnerStatus.COMPLETED
    assert completed.tick is not None
    assert completed.tick.task_state.get_held_object("arm") is not None


def test_runner_holds_while_effect_verification_is_pending_by_default() -> None:
    runner, _, _, sink, _ = _make_runner(with_effect=True)

    blocked = runner.run_until_blocked()

    assert blocked.status is RunnerStatus.RUNNING
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    assert [item.operation for item in blocked.dispatches] == [CommandOperation.HOLD]
    assert len(sink.held) == 1
    assert [target.target_id for target in sink.held[0][0]] == ["arm"]


def test_runner_skips_all_effect_pending_holds_when_disabled() -> None:
    runner, clock, _, sink, _ = _make_runner(
        with_effect=True,
        hold_on_completion=False,
        hold_during_effect_verification=False,
    )

    blocked = runner.run_until_blocked()
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    assert blocked.dispatches == ()

    polls: list[RunnerStep] = []
    for _ in range(2):
        clock.advance(MINIMUM_CYCLE_TIME)
        polls.append(runner.step(effect_verifier=_unresolved_effect_result))

    assert all(step.status is RunnerStatus.RUNNING for step in polls)
    assert all(
        step.tick is not None and step.tick.pending_effect is not None for step in polls
    )
    assert all(step.dispatches == () for step in polls)
    assert sink.held == []


def test_effect_success_adds_no_hold_when_pending_and_completion_holds_are_disabled() -> (
    None
):
    runner, clock, _, sink, _ = _make_runner(
        with_effect=True,
        hold_on_completion=False,
        hold_during_effect_verification=False,
    )
    blocked = runner.run_until_blocked()
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    clock.advance(MINIMUM_CYCLE_TIME)

    completed = runner.step(effect_verifier=_successful_effect_result)

    assert completed.status is RunnerStatus.COMPLETED
    assert completed.tick is not None and completed.tick.pending_effect is None
    assert completed.dispatches == ()
    assert sink.held == []


def test_resumed_effect_verifier_uses_a_fresh_observation() -> None:
    runner, clock, _, _, _ = _make_runner(with_effect=True)
    blocked = runner.run_until_blocked()
    assert blocked.context is not None
    blocked_at = blocked.context.robot.timestamp
    clock.advance(0.5)
    resumed_at = clock.now()
    observed_at: list[float] = []

    def record_fresh_context(
        context: PlanningContext,
        request: EffectVerificationRequest,
    ) -> EffectVerificationResult:
        observed_at.append(context.robot.timestamp)
        return _successful_effect_result(context, request)

    completed = runner.run_until_blocked(effect_verifier=record_fresh_context)

    assert completed.status is RunnerStatus.COMPLETED
    assert observed_at and observed_at[0] >= resumed_at
    assert observed_at[0] > blocked_at


def test_due_effect_verifier_consumes_fresh_observation_in_the_same_step() -> None:
    runner, clock, _, _, _ = _make_runner(with_effect=True)
    blocked = runner.run_until_blocked()
    assert blocked.context is not None
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    blocked_at = blocked.context.robot.timestamp
    clock.advance(MINIMUM_CYCLE_TIME)
    observed_at: list[float] = []

    def verify_fresh_observation(
        context: PlanningContext,
        request: EffectVerificationRequest,
    ) -> EffectVerificationResult:
        observed_at.append(context.robot.timestamp)
        return _successful_effect_result(context, request)

    completed = runner.step(effect_verifier=verify_fresh_observation)

    assert completed.status is RunnerStatus.COMPLETED
    assert completed.tick is not None and completed.tick.pending_effect is None
    assert completed.tick.task_state.get_held_object("arm") is not None
    assert completed.context is not None
    assert observed_at == [completed.context.robot.timestamp]
    assert observed_at[0] > blocked_at


def test_effect_verifier_runs_and_succeeds_at_the_request_deadline() -> None:
    runner, clock, _, _, _ = _make_runner(
        with_effect=True,
        action_timeout=2.0,
    )
    blocked = runner.run_until_blocked()
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    request = blocked.tick.pending_effect
    clock.advance(request.deadline - clock.now())
    observed_at: list[float] = []

    def verify_at_deadline(
        context: PlanningContext,
        current_request: EffectVerificationRequest,
    ) -> EffectVerificationResult:
        observed_at.append(context.robot.timestamp)
        return _successful_effect_result(context, current_request)

    completed = runner.step(effect_verifier=verify_at_deadline)

    assert completed.status is RunnerStatus.COMPLETED
    assert observed_at == pytest.approx([request.deadline])


def test_effect_verifier_is_not_called_after_deadline_and_session_retries() -> None:
    runner, clock, _, _, action = _make_runner(
        with_effect=True,
        max_action_retries=1,
        action_timeout=2.0,
    )
    blocked = runner.run_until_blocked()
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    request = blocked.tick.pending_effect
    plan_count = action.plan_count
    clock.advance(request.deadline - clock.now() + MINIMUM_CYCLE_TIME)
    verifier = Mock()

    retry = runner.step(effect_verifier=verifier)

    verifier.assert_not_called()
    assert retry.status is RunnerStatus.RUNNING
    assert retry.tick is not None and retry.tick.command is not None
    assert action.plan_count == plan_count + 1
    assert {
        ExecutionEventKind.EFFECT_VERIFICATION_TIMEOUT,
        ExecutionEventKind.ACTION_RETRY,
        ExecutionEventKind.REPLANNED,
    }.issubset({event.kind for event in retry.tick.events})


def test_effect_result_and_effect_verifier_are_mutually_exclusive() -> None:
    runner, _, _, sink, action = _make_runner(with_effect=True)
    result = EffectVerificationResult(
        verification_id=0,
        success_mask=torch.tensor([True]),
        failure_mask=torch.tensor([False]),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.step(
            effect_result=result,
            effect_verifier=_successful_effect_result,
        )

    assert action.plan_count == 1
    assert sink.sent == []


@pytest.mark.parametrize(
    "invalid_result",
    [None, True],
    ids=["none", "wrong-type"],
)
def test_effect_verifier_invalid_result_fails_with_cancel_then_hold(
    invalid_result: object | None,
) -> None:
    runner, clock, _, sink, _ = _make_runner(with_effect=True)
    blocked = runner.run_until_blocked()
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    clock.advance(MINIMUM_CYCLE_TIME)

    def invalid_verifier(
        context: PlanningContext,
        request: EffectVerificationRequest,
    ) -> object | None:
        del context, request
        return invalid_result

    failed = runner.step(effect_verifier=invalid_verifier)

    assert failed.status is RunnerStatus.FAILED
    assert [item.operation for item in failed.dispatches] == [
        CommandOperation.CANCEL,
        CommandOperation.HOLD,
    ]
    assert sink.cancel_count == 1
    assert failed.message is not None
    assert "must return exactly EffectVerificationResult" in failed.message


def test_all_false_effect_updates_keep_polling_the_same_request() -> None:
    runner, clock, _, sink, _ = _make_runner(with_effect=True)
    blocked = runner.run_until_blocked()
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    initial_request = blocked.tick.pending_effect
    observed_requests: list[tuple[int, int]] = []

    def report_no_progress(
        context: PlanningContext,
        request: EffectVerificationRequest,
    ) -> EffectVerificationResult:
        observed_requests.append((request.verification_id, request.attempt_generation))
        return EffectVerificationResult(
            verification_id=request.verification_id,
            success_mask=torch.zeros(context.batch_size, dtype=torch.bool),
            failure_mask=torch.zeros(context.batch_size, dtype=torch.bool),
        )

    clock.advance(MINIMUM_CYCLE_TIME)
    first_poll = runner.step(effect_verifier=report_no_progress)
    clock.advance(MINIMUM_CYCLE_TIME)
    second_poll = runner.step(effect_verifier=report_no_progress)

    assert first_poll.status is RunnerStatus.RUNNING
    assert second_poll.status is RunnerStatus.RUNNING
    assert first_poll.tick is not None and first_poll.tick.pending_effect is not None
    assert second_poll.tick is not None and second_poll.tick.pending_effect is not None
    assert observed_requests == [
        (initial_request.verification_id, initial_request.attempt_generation),
        (initial_request.verification_id, initial_request.attempt_generation),
    ]
    assert sink.cancel_count == 0
    assert second_poll.tick.task_state.get_held_object("arm") is None


def test_partial_effect_verifier_receives_the_committed_task_state() -> None:
    runner, _, _, _, _ = _make_runner(with_effect=True, batch_size=2)
    observations: list[list[bool] | None] = []

    def verify_in_two_updates(
        context: PlanningContext,
        request: EffectVerificationRequest,
    ) -> EffectVerificationResult:
        held = context.task.get_held_object("arm")
        observations.append(
            None if held is None or held.env_mask is None else held.env_mask.tolist()
        )
        if request.env_mask.tolist() == [True, True]:
            return EffectVerificationResult(
                verification_id=request.verification_id,
                success_mask=torch.tensor([True, False]),
                failure_mask=torch.tensor([False, False]),
            )
        assert request.env_mask.tolist() == [False, True]
        assert held is not None and held.env_mask is not None
        assert held.env_mask.tolist() == [True, False]
        assert torch.equal(context.task.held_objects["arm"].env_mask, held.env_mask)
        return EffectVerificationResult(
            verification_id=request.verification_id,
            success_mask=torch.tensor([False, True]),
            failure_mask=torch.tensor([False, False]),
        )

    completed = runner.run_until_blocked(effect_verifier=verify_in_two_updates)

    assert completed.status is RunnerStatus.COMPLETED
    assert observations == [None, [True, False]]
    assert completed.context is not None and completed.tick is not None
    assert completed.context.task is completed.tick.task_state


def test_runner_effect_timeout_replans_and_invalidates_cached_request() -> None:
    runner, clock, _, _, action = _make_runner(
        with_effect=True,
        max_action_retries=1,
        action_timeout=2.0,
    )
    blocked = runner.run_until_blocked()
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    request = blocked.tick.pending_effect
    plan_count = action.plan_count
    clock.advance(request.deadline - clock.now() + 0.01)

    retry = runner.step()

    assert retry.status is RunnerStatus.RUNNING
    assert retry.tick is not None and retry.tick.command is not None
    assert runner.effect_verification_pending is False
    assert action.plan_count == plan_count + 1
    kinds = {event.kind for event in retry.tick.events}
    assert ExecutionEventKind.EFFECT_VERIFICATION_TIMEOUT in kinds
    assert ExecutionEventKind.ACTION_RETRY in kinds
    assert ExecutionEventKind.REPLANNED in kinds


def test_runner_effect_timeout_exhaustion_cancels_and_holds() -> None:
    runner, clock, _, sink, _ = _make_runner(
        with_effect=True,
        max_action_retries=0,
        action_timeout=2.0,
    )
    blocked = runner.run_until_blocked()
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    request = blocked.tick.pending_effect
    clock.advance(request.deadline - clock.now() + 0.01)

    failed = runner.step()

    assert failed.status is RunnerStatus.FAILED
    assert runner.effect_verification_pending is False
    assert failed.tick is not None and failed.tick.pending_effect is None
    assert failed.tick.task_state.get_held_object("arm") is None
    assert [item.operation for item in failed.dispatches[-2:]] == [
        CommandOperation.CANCEL,
        CommandOperation.HOLD,
    ]
    assert sink.cancel_count == 1


def test_effect_timeout_still_cancels_and_holds_when_pending_holds_are_disabled() -> (
    None
):
    runner, clock, _, sink, _ = _make_runner(
        with_effect=True,
        max_action_retries=0,
        action_timeout=2.0,
        hold_on_completion=False,
        hold_during_effect_verification=False,
    )
    blocked = runner.run_until_blocked()
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    assert sink.held == []
    request = blocked.tick.pending_effect
    clock.advance(request.deadline - clock.now() + MINIMUM_CYCLE_TIME)

    failed = runner.step()

    assert failed.status is RunnerStatus.FAILED
    assert [item.operation for item in failed.dispatches] == [
        CommandOperation.CANCEL,
        CommandOperation.HOLD,
    ]
    assert sink.cancel_count == 1
    assert len(sink.held) == 1
    assert [target.target_id for target in sink.held[0][0]] == ["arm"]


def test_runner_deactivation_refreshes_cached_effect_request() -> None:
    runner, _, _, _, _ = _make_runner(with_effect=True, batch_size=2)
    blocked = runner.run_until_blocked()
    assert blocked.tick is not None and blocked.tick.pending_effect is not None
    old_id = blocked.tick.pending_effect.verification_id
    old_generation = blocked.tick.pending_effect.attempt_generation

    changed = runner.deactivate_rows(
        torch.tensor([False, True]),
        reason="environment terminated",
    )
    refreshed = runner.run_until_blocked()

    assert changed.tolist() == [False, True]
    assert refreshed.tick is not None and refreshed.tick.pending_effect is not None
    assert refreshed.tick.pending_effect.env_mask.tolist() == [True, False]
    assert refreshed.tick.pending_effect.verification_id != old_id
    assert refreshed.tick.pending_effect.attempt_generation == old_generation

    def verify_remaining(
        context: PlanningContext,
        request: EffectVerificationRequest,
    ) -> EffectVerificationResult:
        return EffectVerificationResult(
            verification_id=request.verification_id,
            success_mask=torch.tensor([True, False]),
            failure_mask=torch.tensor([False, False]),
        )

    completed = runner.run_until_blocked(effect_verifier=verify_remaining)

    assert completed.status is RunnerStatus.COMPLETED
    assert completed.tick is not None
    assert completed.tick.eligible_mask.tolist() == [True, False]


def test_blocking_runner_fails_safely_for_a_mismatched_effect_result() -> None:
    runner, _, _, sink, _ = _make_runner(with_effect=True)

    def mismatched_effect_result(
        context: PlanningContext,
        request: EffectVerificationRequest,
    ) -> EffectVerificationResult:
        return EffectVerificationResult(
            verification_id=request.verification_id + 1,
            success_mask=torch.ones(context.batch_size, dtype=torch.bool),
            failure_mask=torch.zeros(context.batch_size, dtype=torch.bool),
        )

    failed = runner.run_until_blocked(effect_verifier=mismatched_effect_result)

    assert failed.status is RunnerStatus.FAILED
    assert [item.operation for item in failed.dispatches[-2:]] == [
        CommandOperation.CANCEL,
        CommandOperation.HOLD,
    ]
    assert sink.cancel_count == 1
    assert failed.message is not None
    assert "verification_id does not match" in failed.message
