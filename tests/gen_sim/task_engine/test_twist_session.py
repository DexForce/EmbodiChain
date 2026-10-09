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
"""CPU contracts for GenSim feedback around the unchanged runner and session."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.twist_session import (
    DueObservationRequest,
    RevisionControllerStop,
    TwistFeedbackSession,
)
from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    ActionOptions,
    ActionPlan,
    AtomicAction,
    AtomicActionEngine,
    CommandAcknowledgement,
    CommandAckStatus,
    EndEffectorPoseGoal,
    EntityState,
    ExecutionEventKind,
    ExecutionRunner,
    ExecutionRunnerCfg,
    JOINT_POSITION_CAPABILITY,
    JointPositionPayload,
    MotionPolicy,
    PhaseEffectGateRequirement,
    PlanningContext,
    RecoveryPolicy,
    RobotObservation,
    RunnerStatus,
    SceneSnapshot,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    StateDelta,
    TaskState,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.execution import ExecutionSession

INTERVAL = 0.1


class _Clock:
    time = 0.0

    def now(self) -> float:
        return self.time

    def sleep(self, duration: float) -> None:
        self.time += duration


class _Observation:
    def __init__(self, clock: _Clock, batch_size: int) -> None:
        self.clock = clock
        self.qpos = torch.zeros(batch_size, 2)
        self.count = 0
        self.fail = False

    def observe(self, task_state: TaskState) -> PlanningContext:
        self.count += 1
        if self.fail:
            raise RuntimeError("observation unavailable")
        return PlanningContext(
            RobotObservation(
                timestamp=self.clock.time,
                qpos=self.qpos,
                qvel=torch.zeros_like(self.qpos),
            ),
            task_state,
            SceneSnapshot(timestamp=self.clock.time, version=0),
            env_ids=torch.arange(len(self.qpos), dtype=torch.long),
        )


class _Sink:
    def __init__(self, observation: _Observation) -> None:
        self.observation = observation
        self.status = CommandAckStatus.ACCEPTED
        self.sent: list[Any] = []
        self.holds = 0
        self.cancels = 0

    def send(self, command: Any, *, timeout: float) -> CommandAcknowledgement:
        self.sent.append(command.snapshot())
        if self.status is CommandAckStatus.ACCEPTED:
            for endpoint in command.commands:
                assert isinstance(endpoint.payload, JointPositionPayload)
                self.observation.qpos[:, list(endpoint.target.joint_ids)] = (
                    endpoint.payload.positions
                )
        return CommandAcknowledgement(self.status)

    def hold(self, targets: Any, context: Any, *, timeout: float) -> Any:
        self.holds += 1
        return CommandAcknowledgement.accepted_ack()

    def cancel(self, targets: Any, *, timeout: float) -> Any:
        self.cancels += 1
        return CommandAcknowledgement.accepted_ack()


class _Twist(AtomicAction[EndEffectorPoseGoal, ActionOptions]):
    skill_id: ClassVar[str] = "twist"
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

    def __init__(self) -> None:
        super().__init__()
        self.plan_count = 0

    def _plan(self, request: Any, context: PlanningContext) -> ActionPlan:
        self.plan_count += 1
        goal = self.require_goal(request)
        target = torch.full_like(context.robot.qpos, float(goal.xpos[0, 3]))
        positions = torch.stack(
            [context.robot.qpos, torch.lerp(context.robot.qpos, target, 0.5), target],
            dim=1,
        )
        trajectory = TimedTrajectory.from_positions(
            positions,
            env_ids=context.env_ids,
            dt=torch.tensor([[0.0, INTERVAL, INTERVAL]]).repeat(context.batch_size, 1),
        )
        return self.build_plan(
            request,
            context,
            success=True,
            trajectory=trajectory,
            segment_lengths={"approach": 1, "reach": 1, "close": 1},
        )


class _Controller:
    controller_id = "gen_sim.test.precontact"
    revision = "1"
    supported_skill_ids = ("twist",)

    def __init__(self) -> None:
        self.proposal: Any = None
        self.observed: list[Any] = []

    def propose(self, context: Any, due: Any, *, previous_command: Any) -> Any:
        self.observed.append((context, due, previous_command))
        return (
            self.proposal(context, due, previous_command)
            if callable(self.proposal)
            else self.proposal
        )


def _case(
    *, timeout: float = 10.0, batch_size: int = 1, gensim_engine: bool = False
) -> Any:
    clock = _Clock()
    observation = _Observation(clock, batch_size)
    sink = _Sink(observation)
    robot = Mock()
    robot.device, robot.dof = torch.device("cpu"), 2
    robot.control_parts = {"arm": object()}
    robot.get_qpos.return_value = observation.qpos
    robot.get_joint_ids.return_value = [0, 1]
    generator = Mock(robot=robot, device=torch.device("cpu"))
    generator.planner.cfg.planner_type = "stub"
    if gensim_engine:
        from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
            GenSimActionEngine,
        )

        engine = GenSimActionEngine(generator)
    else:
        engine = AtomicActionEngine(generator)
    action = _Twist()
    engine.register(action, replace=True)
    context = observation.observe(TaskState.empty(batch_size, "cpu"))
    pose = torch.eye(4)
    pose[0, 3] = 1.0
    invocation = ActionInvocation(
        skill_id="twist",
        invocation_id="e8/chunk0",
        goal=EndEffectorPoseGoal(pose),
        binding=action.planning_services.bind_control_parts(
            action.binding_contract, {"primary": {"motion": "arm"}}
        ),
        motion_policy=MotionPolicy(sample_count=3),
        recovery_policy=RecoveryPolicy(action_timeout=timeout),
    )
    return SimpleNamespace(
        clock=clock,
        observation=observation,
        sink=sink,
        engine=engine,
        action=action,
        context=context,
        invocation=invocation,
        controller=_Controller(),
    )


def _runner(case: Any, *, feedback: bool = True) -> ExecutionRunner:
    session = (
        TwistFeedbackSession(
            case.engine,
            (case.invocation,),
            case.context,
            revision_controller=case.controller,
        )
        if feedback
        else case.engine.start((case.invocation,), case.context)
    )
    return ExecutionRunner(
        session,
        case.observation,
        case.sink,
        clock=case.clock,
        cfg=ExecutionRunnerCfg(minimum_cycle_time=0.01),
    )


def test_feedback_uses_shared_session_and_due_runner_observation() -> None:
    case = _case()
    runner = _runner(case)
    assert isinstance(runner.session, ExecutionSession)
    before = case.observation.count
    runner.step()
    assert case.observation.count == before + 1
    assert len(case.controller.observed) == 1
    assert case.controller.observed[0][2] is None
    runner.step()
    assert case.observation.count == before + 1
    assert len(case.controller.observed) == 1
    case.clock.sleep(INTERVAL)
    runner.step()
    assert case.observation.count == before + 2
    assert len(case.controller.observed) == 2
    context, due, emitted = case.controller.observed[-1]
    assert context.robot.timestamp == INTERVAL
    assert due.segment_name == "reach"
    assert torch.equal(
        emitted.commands[0].payload.positions,
        case.sink.sent[0].commands[0].payload.positions,
    )


@pytest.mark.parametrize(
    "status", [CommandAckStatus.REJECTED, CommandAckStatus.TIMED_OUT]
)
def test_unaccepted_send_never_reaches_another_feedback_cycle(status: Any) -> None:
    case = _case()
    case.sink.status = status
    runner = _runner(case)
    assert runner.step().status is RunnerStatus.FAILED
    case.clock.sleep(INTERVAL)
    runner.step()
    assert len(case.controller.observed) == 1
    assert case.action.plan_count == 1
    assert case.sink.cancels == case.sink.holds == 1


def test_cancel_does_not_reuse_emitted_seed() -> None:
    case = _case()
    runner = _runner(case)
    runner.step()
    runner.cancel()
    case.clock.sleep(INTERVAL)
    runner.step()
    assert len(case.controller.observed) == 1


@pytest.mark.parametrize(
    "field", ["effect_result", "phase_effect_gate_result", "held_object_guard_result"]
)
def test_correlated_results_fail_before_proposal_or_revision(field: str) -> None:
    case = _case()
    session = _runner(case).session
    case.controller.proposal = replace(case.invocation, revision=1)
    with pytest.raises(RuntimeError, match="effect|gate|held"):
        session.tick(case.context, **{field: object()})
    assert case.controller.observed == []
    assert case.action.plan_count == 1


def test_revision_replans_once_from_fresh_observation_without_renewing_timeout() -> (
    None
):
    case = _case(timeout=0.3)
    runner = _runner(case)
    runner.step()
    case.controller.proposal = replace(case.invocation, revision=1)
    case.clock.sleep(INTERVAL)
    case.observation.qpos.fill_(0.4)
    step = runner.step()
    assert case.action.plan_count == 2
    assert step.tick is not None
    assert any(
        event.kind is ExecutionEventKind.INVOCATION_REVISED
        for event in step.tick.events
    )
    assert runner.session.active_plan.recovery_policy.action_timeout == pytest.approx(
        0.2
    )
    assert torch.equal(
        case.sink.sent[-1].commands[0].payload.positions, torch.full((1, 2), 0.4)
    )
    case.controller.proposal = None
    case.clock.sleep(0.21)
    assert runner.step().status is RunnerStatus.FAILED
    assert len(case.sink.sent) == 2
    assert case.sink.cancels == case.sink.holds == 1


def test_controller_deadline_identity_survives_multiple_revisions() -> None:
    case = _case()
    runner = _runner(case)
    runner.step()
    for revision in (1, 2, 3):
        case.controller.proposal = replace(case.invocation, revision=revision)
        case.clock.sleep(INTERVAL)
        assert runner.step().status is RunnerStatus.RUNNING
    assert case.action.plan_count == 4
    assert [
        (due.original_planned_at, due.original_action_timeout)
        for _, due, _ in case.controller.observed
    ] == [(0.0, 10.0)] * 4
    assert runner.session.active_plan.recovery_policy.action_timeout == pytest.approx(
        9.7
    )


@pytest.mark.parametrize("kind", ["stop", "wrong_type", "wrong_skill", "exception"])
def test_proposal_refusal_uses_shared_safe_stop(kind: str) -> None:
    case = _case()
    runner = _runner(case)
    runner.step()
    if kind == "stop":
        case.controller.proposal = RevisionControllerStop("bounded refusal")
    elif kind == "wrong_type":
        case.controller.proposal = object()
    elif kind == "wrong_skill":
        case.controller.proposal = replace(
            case.invocation, revision=1, skill_id="foreign"
        )
    else:
        case.controller.proposal = Mock(side_effect=ValueError("controller failed"))
    case.clock.sleep(INTERVAL)
    result = runner.step()
    assert result.status is RunnerStatus.FAILED
    assert len(case.sink.sent) == 1
    assert case.sink.cancels == case.sink.holds == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "2"),
        ("supported_skill_ids", ("foreign",)),
        ("controller_id", "changed"),
    ],
)
def test_controller_identity_drift_stops_before_dispatch(
    field: str, value: Any
) -> None:
    case = _case()
    runner = _runner(case)
    runner.step()
    setattr(case.controller, field, value)
    case.clock.sleep(INTERVAL)
    assert runner.step().status is RunnerStatus.FAILED
    assert len(case.sink.sent) == 1


def test_proposal_context_and_emitted_frame_are_owned_snapshots() -> None:
    case = _case()
    runner = _runner(case)
    runner.step()

    def mutate(context: Any, due: Any, previous: Any) -> None:
        context.robot.qpos.fill_(999)
        due.env_mask.fill_(False)
        previous.commands[0].payload.positions.fill_(999)

    case.controller.proposal = mutate
    case.clock.sleep(INTERVAL)
    step = runner.step()
    assert not step.context.robot.qpos.eq(999).any()
    assert case.sink.sent[-1].active_mask.all()
    assert not case.sink.sent[0].commands[0].payload.positions.eq(999).any()


def test_observation_failure_never_proposes_or_revises() -> None:
    case = _case()
    runner = _runner(case)
    case.observation.fail = True
    assert runner.step().status is RunnerStatus.FAILED
    assert case.controller.observed == []


def test_retaining_plan_matches_original_session_commands_and_events_exactly() -> None:
    records = []
    for feedback in (False, True):
        case = _case()
        runner = _runner(case, feedback=feedback)
        events = []
        while runner.status is RunnerStatus.RUNNING:
            step = runner.step()
            if step.tick is not None:
                events += [
                    (event.kind, event.timestamp, event.message)
                    for event in step.tick.events
                ]
            case.clock.sleep(step.wait_duration)
        records.append(
            (
                runner.status,
                events,
                case.observation.count,
                case.sink.sent,
                case.sink.holds,
            )
        )
    assert records[0][:3] == records[1][:3]
    assert records[0][4] == records[1][4]
    for left, right in zip(records[0][3], records[1][3], strict=True):
        assert torch.equal(
            left.commands[0].payload.positions, right.commands[0].payload.positions
        )
        assert torch.equal(left.active_mask, right.active_mask)
        assert torch.equal(left.hold_duration, right.hold_duration)


def test_feedback_rejects_multiple_environments_before_planning() -> None:
    case = _case(batch_size=2)
    with pytest.raises(ValueError, match="one|single"):
        _runner(case)
    assert case.action.plan_count == 0


def test_feedback_rejects_declared_phase_gate_before_planning() -> None:
    case = _case()
    case.invocation = replace(
        case.invocation,
        phase_effect_gates=(PhaseEffectGateRequirement("ready", "close"),),
    )
    with pytest.raises(ValueError, match="gate"):
        _runner(case)
    assert case.action.plan_count == 0


def test_manual_tick_frame_is_emitted_not_a_transport_acknowledgement() -> None:
    case = _case()
    session = _runner(case).session
    first = session.tick(case.context)
    assert first.command is not None
    assert case.sink.sent == []
    assert case.controller.observed[0][2] is None
    assert not hasattr(session, "accepted_command")


def test_manual_tick_without_native_target_update_cannot_qualify_seed() -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_feedback import (
        TwistFeedbackState,
    )

    case = _case()
    session = _runner(case).session
    session.tick(case.context)
    second = session.tick(
        replace(case.context, robot=replace(case.context.robot, timestamp=INTERVAL))
    )
    frame = second.command
    assert frame is not None and case.sink.sent == []
    target = frame.commands[0].target
    assert not torch.equal(frame.commands[0].payload.positions, case.observation.qpos)
    state = SimpleNamespace(
        robot=SimpleNamespace(get_qpos=Mock(return_value=case.observation.qpos.clone()))
    )
    chunk = SimpleNamespace(
        request=SimpleNamespace(
            binding=SimpleNamespace(
                endpoint=lambda *_: SimpleNamespace(require_target=lambda _: target)
            )
        ),
        command_baseline=case.observation.qpos.clone(),
    )
    with pytest.raises(ValueError, match="command-target readback"):
        TwistFeedbackState._seed(state, chunk, case.context, frame)


def test_hold_tick_clears_emitted_frame() -> None:
    case = _case()
    runner = _runner(case)
    while runner.status is RunnerStatus.RUNNING:
        step = runner.step()
        case.clock.sleep(step.wait_duration)
    assert case.sink.holds > 0
    assert runner.session._last_emitted_frame is None


def test_controller_instance_swap_is_not_an_identity_match() -> None:
    case = _case()
    runner = _runner(case)
    runner.step()
    runner.session._feedback_controller = _Controller()
    case.clock.sleep(INTERVAL)
    assert runner.step().status is RunnerStatus.FAILED
    assert len(case.sink.sent) == 1


def test_identity_changed_inside_propose_fails_before_dispatch() -> None:
    case = _case()
    runner = _runner(case)

    def mutate(context: Any, due: Any, previous: Any) -> None:
        case.controller.revision = "changed"

    case.controller.proposal = mutate
    assert runner.step().status is RunnerStatus.FAILED
    assert case.sink.sent == []


def test_scene_snapshot_mutation_cannot_change_live_context() -> None:
    case = _case()
    case.context = replace(
        case.context,
        scene=SceneSnapshot(
            timestamp=0.0,
            version=0,
            entities={"fixture": EntityState(torch.eye(4))},
        ),
    )
    session = _runner(case).session

    def mutate(context: Any, due: Any, previous: Any) -> None:
        context.scene.entities["fixture"].pose.fill_(999)

    case.controller.proposal = mutate
    session.tick(case.context)
    assert not case.context.scene.entities["fixture"].pose.eq(999).any()
    assert not session.latest_context.scene.entities["fixture"].pose.eq(999).any()


def test_effectful_replacement_cannot_dispatch_from_specialized_session() -> None:
    case = _case()
    runner = _runner(case)
    runner.step()
    original = case.action._plan

    def effectful(request: Any, context: Any) -> ActionPlan:
        from embodichain.lab.sim.atomic_actions import ArticulationJointState

        plan = original(request, context)
        return replace(
            plan,
            expected_effects=StateDelta(
                articulation_joint_updates={
                    ("fixture", "joint"): ArticulationJointState(torch.zeros(1, 1))
                }
            ),
        )

    case.action._plan = effectful
    case.controller.proposal = replace(case.invocation, revision=1)
    case.clock.sleep(INTERVAL)
    assert runner.step().status is RunnerStatus.FAILED
    assert len(case.sink.sent) == 1


def test_due_observation_owns_mask_and_preserves_unbounded_timeout() -> None:
    mask = torch.tensor([True])
    due = DueObservationRequest(
        "twist", "chunk", 0, 0, 0, 0, "reach", mask, 0.0, float("inf")
    )
    mask.fill_(False)
    assert due.env_mask.tolist() == [True]
    assert due.original_action_timeout == float("inf")


def test_empty_eligible_cohort_preserves_failed_original_session() -> None:
    from embodichain.lab.sim.atomic_actions.execution import ExecutionStatus

    case = _case()
    session = TwistFeedbackSession(
        case.engine,
        (case.invocation,),
        case.context,
        revision_controller=case.controller,
        eligible_mask=torch.tensor([False]),
    )
    assert session.status is ExecutionStatus.FAILED
    assert session.plan_attempts == ()
    assert session.tick(case.context).status is ExecutionStatus.FAILED
    assert case.action.plan_count == 0
    assert case.controller.observed == []


def test_unsuccessful_initial_plan_does_not_open_feedback_boundary() -> None:
    case = _case()
    original = case.action._plan

    def unsuccessful(request: Any, context: Any) -> ActionPlan:
        plan = original(request, context)
        return case.action.build_plan(
            request,
            context,
            success=False,
            trajectory=plan.joint_trajectory,
            segment_lengths={"approach": 1, "reach": 1, "close": 1},
        )

    case.action._plan = unsuccessful
    runner = _runner(case)
    runner.step()
    assert case.controller.observed == []
    assert case.sink.sent == []


def _feedback_wiring() -> Any:
    from embodichain.gen_sim.task_engine._task_program.twist_feedback import (
        TwistFeedbackState,
    )
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import GenSimTwist

    case = _case(gensim_engine=True)
    route = SimpleNamespace(
        feedback_enabled=True,
        grip_extra_m=0.002,
        binding=SimpleNamespace(object_id="fixture"),
    )
    art = object()
    simulation = SimpleNamespace(get_articulation=lambda _: art)
    sensor = SimpleNamespace(route=route, robot=case.engine.robot, art=art)
    guard = SimpleNamespace(
        route=route, simulation=simulation, robot=case.engine.robot, sensor=sensor
    )
    controller = TwistFeedbackState(
        route, simulation, case.engine.robot, sensor, guard, wall_clock=lambda: 0.0
    )
    action = GenSimTwist(feedback_state=controller)
    case.engine.register(action, replace=True)
    case.controller, case.action = controller, action
    case.route, case.simulation, case.sensor, case.guard = (
        route,
        simulation,
        sensor,
        guard,
    )
    return case


@pytest.mark.parametrize(
    "mismatch", ["controller_type", "disabled", "robot", "action_type", "action_state"]
)
def test_feedback_configuration_rejects_unowned_runtime_parts(mismatch: str) -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import GenSimTwist

    case = _feedback_wiring()
    controller = case.controller
    if mismatch == "controller_type":
        controller = object()
    elif mismatch == "disabled":
        controller.enabled = False
    elif mismatch == "robot":
        controller.robot = object()
    elif mismatch == "action_type":
        case.engine.register(_Twist(), replace=True)
    else:
        case.engine.register(GenSimTwist(feedback_state=object()), replace=True)
    with pytest.raises(ValueError, match="exact action and robot"):
        case.engine.configure_twist_feedback(controller)
    assert case.engine._twist_session_controller is None


def test_feedback_configuration_binds_exact_instance_and_rejects_reinstallation() -> (
    None
):
    case = _feedback_wiring()
    case.engine.configure_twist_feedback(case.controller)
    assert case.engine._twist_session_controller is case.controller
    with pytest.raises(ValueError, match="exact action and robot"):
        case.engine.configure_twist_feedback(case.controller)
    assert case.engine._twist_session_controller is case.controller


def test_gensim_engine_without_feedback_returns_original_session() -> None:
    case = _case(gensim_engine=True)
    session = case.engine.start(iter((case.invocation,)), case.context)
    assert type(session) is ExecutionSession
    assert case.action.plan_count == 1


def test_gensim_engine_non_twist_keeps_original_session_with_feedback_installed() -> (
    None
):
    class NonTwist(_Twist):
        skill_id = "test.motion"
        binding_contract = _Twist.binding_contract

    case = _feedback_wiring()
    case.engine.configure_twist_feedback(case.controller)
    action = NonTwist()
    case.engine.register(action)
    invocation = replace(
        case.invocation,
        skill_id=action.skill_id,
        binding=action.planning_services.bind_control_parts(
            action.binding_contract, {"primary": {"motion": "arm"}}
        ),
    )
    session = case.engine.start((invocation,), case.context)
    assert type(session) is ExecutionSession
    assert action.plan_count == 1
    assert case.controller.chunks == {}


def test_empty_eligible_engine_route_skips_feedback_session(monkeypatch: Any) -> None:
    from embodichain.gen_sim.task_engine._task_program import twist_session

    case = _feedback_wiring()
    case.engine.configure_twist_feedback(case.controller)
    original_session = object()
    base_start = Mock(return_value=original_session)
    selected_start = Mock(side_effect=AssertionError("feedback must not start"))
    monkeypatch.setattr(AtomicActionEngine, "start", base_start)
    monkeypatch.setattr(twist_session, "TwistFeedbackSession", selected_start)
    eligible = torch.tensor([False])
    result = case.engine.start((case.invocation,), case.context, eligible_mask=eligible)
    assert result is original_session
    base_start.assert_called_once_with(
        (case.invocation,), case.context, eligible_mask=eligible
    )
    selected_start.assert_not_called()
    assert case.controller.chunks == {}


def test_opted_in_engine_routes_exact_owned_controller_and_context(
    monkeypatch: Any,
) -> None:
    from embodichain.gen_sim.task_engine._task_program import twist_session

    case = _feedback_wiring()
    case.engine.configure_twist_feedback(case.controller)
    selected_session = object()
    constructor = Mock(return_value=selected_session)
    monkeypatch.setattr(twist_session, "TwistFeedbackSession", constructor)
    eligible = torch.tensor([True])
    result = case.engine.start(
        iter((case.invocation,)), case.context, eligible_mask=eligible
    )
    assert result is selected_session
    constructor.assert_called_once_with(
        case.engine,
        (case.invocation,),
        case.context,
        revision_controller=case.controller,
        eligible_mask=eligible,
    )


def test_reassembling_engine_allocates_fresh_feedback_state(monkeypatch: Any) -> None:
    from embodichain.gen_sim.task_engine._task_program import twist_geometry_guard
    from embodichain.gen_sim.task_engine._task_program.assembly import _TaskFactory
    from embodichain.lab.task_program.semantics import RobotSkillProfile

    case = _feedback_wiring()
    profile = Mock(spec=RobotSkillProfile)
    profile.profile_id = "test.robot"
    profile.action_control_profiles.return_value = {}
    factory = object.__new__(_TaskFactory)
    factory._robot, factory._simulation = case.engine.robot, case.simulation
    factory._robot_profile_binding = profile
    factory._registration = Mock()
    factory._create_motion_generator = Mock(return_value=case.engine.motion_generator)
    factory._drawers = ()
    factory._grasp_pose_generators = {}
    factory._motion_samples = ()
    factory._cartesian_calls = ()
    factory._articulation_calls = ()
    factory._adaptive_pick = False
    factory._pick_purposes = ()
    factory._pour_receivers = {}
    factory._coordinated_motion = False
    factory._press_routes = ()
    factory._twist_routes = (case.route,)
    factory._task_post_port = SimpleNamespace(sensor=case.sensor)
    monkeypatch.setattr(
        twist_geometry_guard,
        "E8GeometryGuard",
        lambda route, simulation, robot, sensor: SimpleNamespace(
            route=route, simulation=simulation, robot=robot, sensor=sensor
        ),
    )
    first = factory.create_atomic_action_engine(profile)
    second = factory.create_atomic_action_engine(profile)
    left, right = first._twist_session_controller, second._twist_session_controller
    assert first is not second
    assert left is not right and left is not case.controller
    assert first.actions["twist"]._feedback_state is left
    assert second.actions["twist"]._feedback_state is right
    assert case.sensor._feedback_state is right
    left.episode_corrections = 3
    left.samples.append({"timestamp": 1.0})
    assert right.episode_corrections == 0
    assert len(right.samples) == 0
    factory._registration.validate_engine.assert_any_call(first)
    factory._registration.validate_engine.assert_any_call(second)
