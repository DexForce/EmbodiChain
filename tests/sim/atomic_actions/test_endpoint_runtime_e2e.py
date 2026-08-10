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

"""End-to-end coverage for generic atomic-action runtime endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import Mock

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    ActionOptions,
    ActionPlan,
    AtomicAction,
    AtomicActionEngine,
    CommandAcknowledgement,
    EndpointCommand,
    EndpointCommandRouter,
    ExecutionRunner,
    ExecutionStatus,
    JOINT_POSITION_CAPABILITY,
    JointPositionGoal,
    JointPositionPayload,
    JointPositionTarget,
    MoveJoints,
    PlanningContext,
    RobotObservation,
    RunnerStatus,
    RuntimeCommandFrame,
    RuntimeCommandPayload,
    RuntimeEndpointTarget,
    SceneSnapshot,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    TaskState,
    TimedCommandSequence,
)
from embodichain.lab.sim.atomic_actions.invocation import ResolvedActionRequest
from embodichain.lab.sim.planners import PlanResult
from embodichain.lab.sim.skills import (
    EndpointResolution,
    ResourceBinding,
    ResourceEndpoint,
    ResourceEndpointAdapter,
    RobotResource,
    RobotSkillProfile,
)


class _Clock:
    """Deterministic clock used by the runner."""

    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        """Return simulated time."""
        return self.value

    def sleep(self, duration: float) -> None:
        """Advance simulated time."""
        self.value += duration


class _Robot:
    """Small stateful robot with one whole-body control part."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dof = 4
        self.control_parts = {"whole_body": object()}
        self.qpos = torch.zeros(2, self.dof)

    def get_qpos(self, name: str | None = None) -> torch.Tensor:
        """Return observed joint positions."""
        if name is not None and name != "whole_body":
            raise KeyError(name)
        return self.qpos.clone()

    def get_qvel(self, name: str | None = None) -> torch.Tensor:
        """Return zero joint velocities."""
        return torch.zeros_like(self.get_qpos(name))

    def get_joint_ids(self, name: str) -> list[int]:
        """Resolve the whole-body control part."""
        if name != "whole_body":
            raise KeyError(name)
        return list(range(self.dof))


class _Provider:
    """Observe the stateful robot at the injected clock time."""

    def __init__(self, robot: _Robot, clock: _Clock) -> None:
        self.robot = robot
        self.clock = clock
        self.env_ids = torch.tensor([3, 7], dtype=torch.long)

    def observe(self, task_state: TaskState) -> PlanningContext:
        """Return one fresh, correlated planning context."""
        qpos = self.robot.get_qpos()
        timestamp = self.clock.now()
        return PlanningContext(
            robot=RobotObservation(
                timestamp=timestamp,
                qpos=qpos,
                qvel=torch.zeros_like(qpos),
            ),
            task=task_state,
            scene=SceneSnapshot(timestamp=timestamp, version=0),
            env_ids=self.env_ids,
        )


def _engine(robot: _Robot) -> AtomicActionEngine:
    """Build a core engine around a controllable planning stub."""
    generator = Mock()
    generator.robot = robot
    generator.device = robot.device
    generator.planner.cfg.planner_type = "stub"

    def generate(states: list[object], *, options: object) -> PlanResult:
        target = states[-1].qpos
        assert isinstance(target, torch.Tensor)
        start = options.start_qpos
        assert isinstance(start, torch.Tensor)
        positions = torch.stack((start, target), dim=1)
        dt = torch.zeros(positions.shape[:2], dtype=torch.float32)
        dt[:, 1] = 0.01
        return PlanResult(
            success=torch.ones(positions.shape[0], dtype=torch.bool),
            positions=positions,
            dt=dt,
            duration=dt.sum(dim=1),
        )

    generator.generate.side_effect = generate
    return AtomicActionEngine(generator, load_builtins=False)


class _JointTransport:
    """Apply joint endpoint payloads to the stateful test robot."""

    transport_id = JointPositionTarget.TRANSPORT_ID
    payload_type = JointPositionPayload

    def __init__(self, robot: _Robot) -> None:
        self.robot = robot
        self.sent: list[RuntimeCommandFrame] = []
        self.held: list[tuple[RuntimeEndpointTarget, ...]] = []

    def send(
        self,
        frame: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Apply each addressed joint subset."""
        del timeout
        self.sent.append(frame.snapshot())
        for command in frame.commands:
            assert isinstance(command.target, JointPositionTarget)
            assert isinstance(command.payload, JointPositionPayload)
            joint_ids = list(command.target.joint_ids)
            self.robot.qpos[:, joint_ids] = torch.where(
                frame.active_mask[:, None],
                command.payload.positions,
                self.robot.qpos[:, joint_ids],
            )
        return CommandAcknowledgement.accepted_ack()

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Hold only the joint subsets addressed by the runner."""
        del timeout
        self.held.append(tuple(target.snapshot() for target in targets))
        for target in targets:
            assert isinstance(target, JointPositionTarget)
            joint_ids = list(target.joint_ids)
            self.robot.qpos[:, joint_ids] = context.robot.qpos[:, joint_ids]
        return CommandAcknowledgement.accepted_ack()

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Acknowledge synchronous cancellation."""
        del targets, timeout
        return CommandAcknowledgement.accepted_ack()


def test_whole_body_joint_endpoint_executes_without_arm_or_tool_roles() -> None:
    robot = _Robot()
    engine = _engine(robot)
    engine.register(MoveJoints())
    binding = engine.bind_control_parts(
        "move_joints",
        {"primary": {"motion": "whole_body"}},
    )
    invocation = ActionInvocation(
        skill_id="move_joints",
        goal=JointPositionGoal(torch.full((2, robot.dof), 0.5)),
        binding=binding,
    )
    clock = _Clock()
    provider = _Provider(robot, clock)
    context = provider.observe(TaskState.empty(batch_size=2, device="cpu"))

    plan = engine.plan(invocation, context)
    target = binding.endpoint("primary", "motion").require_target(JointPositionTarget)
    assert target.control_part == "whole_body"
    assert plan.joint_trajectory is not None
    assert plan.commands.targets[0].target_id == "whole_body"

    transport = _JointTransport(robot)
    runner = ExecutionRunner(
        engine.start((invocation,), context),
        provider,
        EndpointCommandRouter((transport,)),
        clock=clock,
    )
    result = runner.run_until_blocked()

    assert result.status is RunnerStatus.COMPLETED
    assert result.tick is not None
    assert result.tick.status is ExecutionStatus.COMPLETED
    assert len(transport.sent) == 2
    assert len(transport.held) == 1
    assert transport.held[0][0].target_id == "whole_body"
    assert torch.allclose(robot.qpos, torch.full((2, robot.dof), 0.5))


@dataclass(frozen=True, slots=True)
class _PlanarVelocityTarget(RuntimeEndpointTarget):
    """Address one planar velocity controller."""

    controller_id: str

    @property
    def transport_id(self) -> str:
        """Return the custom transport identifier."""
        return "test.planar_velocity"

    @property
    def target_id(self) -> str:
        """Return the controller-local target identifier."""
        return self.controller_id


@dataclass(frozen=True, slots=True, eq=False)
class _PlanarVelocityPayload(RuntimeCommandPayload):
    """Batched ``(vx, vy, yaw_rate)`` commands."""

    twist: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.twist, torch.Tensor) or self.twist.dim() != 2:
            raise ValueError("twist must have shape (batch_size, 3).")
        if self.twist.shape[0] < 1 or self.twist.shape[1] != 3:
            raise ValueError("twist must have shape (batch_size, 3).")
        object.__setattr__(self, "twist", self.twist.clone())

    @property
    def batch_size(self) -> int:
        """Return the number of environment rows."""
        return int(self.twist.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the payload device."""
        return self.twist.device

    @property
    def transport_id(self) -> str:
        """Return the custom transport identifier."""
        return "test.planar_velocity"

    def snapshot(self) -> _PlanarVelocityPayload:
        """Return an independently owned payload."""
        return _PlanarVelocityPayload(self.twist)


@dataclass(frozen=True, slots=True)
class _PlanarVelocityEndpoint(ResourceEndpoint):
    """Profile declaration for a planar velocity controller."""

    controller_id: str


class _PlanarVelocityAdapter(ResourceEndpointAdapter):
    """Resolve the custom profile endpoint to a runtime target."""

    adapter_id: ClassVar[str] = "test.planar_velocity"
    endpoint_type: ClassVar[type[ResourceEndpoint]] = _PlanarVelocityEndpoint

    def resolve(
        self,
        endpoint: ResourceEndpoint,
        *,
        engine: AtomicActionEngine,
    ) -> EndpointResolution:
        """Resolve immutable addressing and an exclusive controller claim."""
        del engine
        assert isinstance(endpoint, _PlanarVelocityEndpoint)
        return EndpointResolution(
            runtime_target=_PlanarVelocityTarget(endpoint.controller_id),
            claim_tokens=frozenset({f"controller:{endpoint.controller_id}"}),
        )


@dataclass(frozen=True, slots=True, eq=False)
class _DriveGoal:
    """Planar velocity command used by the custom atomic action."""

    goal_kind: ClassVar[str] = "planar_velocity"
    twist: torch.Tensor


class _DriveVelocity(AtomicAction[_DriveGoal, ActionOptions]):
    """Custom action proving non-joint commands cross the full runtime."""

    skill_id: ClassVar[str] = "drive_velocity"
    GoalType: ClassVar[type] = _DriveGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                "body",
                endpoints=(
                    SkillEndpointRequirement(
                        "motion",
                        capabilities=frozenset({"motion.base.planar_velocity"}),
                    ),
                ),
            ),
        )
    )

    def _plan(
        self,
        request: ResolvedActionRequest[_DriveGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Emit one drive frame followed by an explicit zero-velocity frame."""
        goal = self.require_goal(request)
        target = request.binding.endpoint("body", "motion").require_target(
            _PlanarVelocityTarget
        )
        active = torch.ones(context.batch_size, dtype=torch.bool, device=self.device)
        frames = tuple(
            RuntimeCommandFrame(
                commands=(
                    EndpointCommand(
                        target=target,
                        payload=_PlanarVelocityPayload(twist),
                    ),
                ),
                active_mask=active,
                env_ids=context.env_ids,
                hold_duration=torch.full(
                    (context.batch_size,),
                    duration,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            for twist, duration in (
                (goal.twist.to(self.device), 0.02),
                (torch.zeros_like(goal.twist, device=self.device), 0.0),
            )
        )
        return self.build_command_plan(
            request,
            context,
            success=True,
            commands=TimedCommandSequence(frames, context.env_ids),
            segment_lengths={"drive": 1, "stop": 1},
        )


class _PlanarVelocityTransport:
    """Record velocity frames and own the zero-velocity safe state."""

    transport_id = "test.planar_velocity"
    payload_type = _PlanarVelocityPayload

    def __init__(self) -> None:
        self.sent: list[torch.Tensor] = []
        self.hold_targets: tuple[RuntimeEndpointTarget, ...] = ()
        self.last_twist: torch.Tensor | None = None

    def send(
        self,
        frame: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Record active twists and neutralize every inactive row."""
        del timeout
        payload = frame.commands[0].payload
        assert isinstance(payload, _PlanarVelocityPayload)
        self.last_twist = torch.where(
            frame.active_mask[:, None],
            payload.twist,
            torch.zeros_like(payload.twist),
        )
        self.sent.append(self.last_twist)
        return CommandAcknowledgement.accepted_ack()

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Apply the velocity transport's safe zero command."""
        del context, timeout
        self.hold_targets = tuple(target.snapshot() for target in targets)
        assert self.last_twist is not None
        self.last_twist = torch.zeros_like(self.last_twist)
        return CommandAcknowledgement.accepted_ack()

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Acknowledge cancellation."""
        del targets, timeout
        return CommandAcknowledgement.accepted_ack()


def test_custom_planar_velocity_endpoint_runs_from_profile_through_router() -> None:
    robot = _Robot()
    engine = _engine(robot)
    engine.register(_DriveVelocity())
    profile = RobotSkillProfile(
        profile_id="mobile",
        resources={
            "mobile_base": RobotResource(
                "mobile_base",
                endpoints={
                    "motion": _PlanarVelocityEndpoint(
                        "base_controller",
                        capabilities=frozenset({"motion.base.planar_velocity"}),
                    )
                },
            )
        },
        defaults={"drive_velocity": ResourceBinding({"body": "mobile_base"})},
    )
    bound = engine.bind_skill_profile(
        profile,
        endpoint_adapters={_PlanarVelocityEndpoint: _PlanarVelocityAdapter()},
    )
    binding = bound.resolve("drive_velocity").action_binding
    goal_twist = torch.tensor([[0.5, 0.0, 0.1], [0.2, 0.0, -0.1]])
    invocation = ActionInvocation(
        skill_id="drive_velocity",
        goal=_DriveGoal(goal_twist),
        binding=binding,
    )
    clock = _Clock()
    provider = _Provider(robot, clock)
    context = provider.observe(TaskState.empty(batch_size=2, device="cpu"))

    plan = engine.plan(invocation, context)
    assert plan.joint_trajectory is None
    assert plan.segment("drive").waypoint_count == 1
    assert plan.commands.targets[0].transport_id == "test.planar_velocity"

    transport = _PlanarVelocityTransport()
    runner = ExecutionRunner(
        engine.start((invocation,), context),
        provider,
        EndpointCommandRouter((transport,)),
        clock=clock,
    )
    result = runner.run_until_blocked()

    assert result.status is RunnerStatus.COMPLETED
    assert len(transport.sent) == 2
    assert torch.allclose(transport.sent[0], goal_twist)
    assert torch.count_nonzero(transport.sent[1]) == 0
    assert transport.last_twist is not None
    assert torch.count_nonzero(transport.last_twist) == 0
    assert transport.hold_targets[0].target_id == "base_controller"


def test_planar_velocity_transport_neutralizes_inactive_rows() -> None:
    transport = _PlanarVelocityTransport()
    frame = RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                target=_PlanarVelocityTarget("base_controller"),
                payload=_PlanarVelocityPayload(torch.ones(2, 3)),
            ),
        ),
        active_mask=torch.tensor([True, False]),
        env_ids=torch.tensor([0, 1]),
        hold_duration=torch.zeros(2),
    )

    acknowledgement = transport.send(frame, timeout=1.0)

    assert acknowledgement.accepted
    assert transport.last_twist is not None
    assert torch.equal(transport.last_twist[0], torch.ones(3))
    assert torch.count_nonzero(transport.last_twist[1]) == 0
