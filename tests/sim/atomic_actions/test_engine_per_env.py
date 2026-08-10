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

"""Tests for dynamic goals and closed-loop execution recovery."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
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
    DynamicCollisionMode,
    EndEffectorPoseGoal,
    EndpointBinding,
    EndpointCommand,
    EntityState,
    ExecutionEventKind,
    ExecutionSession,
    ExecutionStatus,
    ExecutionTick,
    EffectVerificationResult,
    GraspGoal,
    HeldObjectState,
    JointPositionPayload,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionRequest,
    RobotObservation,
    RuntimeCommandFrame,
    SceneEntityPose,
    SceneSnapshot,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    StateDelta,
    TaskState,
    TimedCommandSequence,
    TimedTrajectory,
)
from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.planners import PlanOptions


class DynamicAction(AtomicAction[EndEffectorPoseGoal, ActionOptions]):
    """Test action whose terminal joint command follows a scene entity x pose."""

    skill_id: ClassVar[str] = "dynamic"
    GoalType: ClassVar[type] = EndEffectorPoseGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(SkillEndpointRequirement(endpoint_id="motion"),),
            ),
        )
    )

    def __init__(self) -> None:
        super().__init__()
        self.plan_count = 0
        self.requests: list[ResolvedActionRequest] = []

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(request)
        self.plan_count += 1
        self.requests.append(request)
        pose = resolve_pose_goal(goal.xpos, context, name="xpos")
        target = pose[:, 0, 3].unsqueeze(1).expand_as(context.robot.qpos)
        return self.build_plan(
            request,
            context,
            success=True,
            trajectory=torch.stack([context.robot.qpos, target], dim=1),
        )


class EffectAction(DynamicAction):
    """Dynamic test action that declares an attachment effect."""

    skill_id: ClassVar[str] = "effect"
    binding_contract: ClassVar[SkillBindingContract] = DynamicAction.binding_contract

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(request)
        pose = resolve_pose_goal(goal.xpos, context, name="xpos")
        target = pose[:, 0, 3].unsqueeze(1).expand_as(context.robot.qpos)
        semantics = ObjectSemantics(
            affordance=Affordance(), geometry={}, label="object"
        )
        held = HeldObjectState(
            semantics=semantics,
            object_to_eef=torch.eye(4),
            grasp_xpos=torch.eye(4),
        )
        return self.build_plan(
            request,
            context,
            success=True,
            trajectory=torch.stack([context.robot.qpos, target], dim=1),
            expected_effects=StateDelta(held_object_updates={"arm": held}),
        )


class FailedEffectAction(EffectAction):
    """Effect-declaring action whose planner fails for every environment."""

    skill_id: ClassVar[str] = "failed_effect"
    binding_contract: ClassVar[SkillBindingContract] = DynamicAction.binding_contract

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        plan = super()._plan(request, context)
        return replace(plan, plan_success=torch.zeros_like(plan.plan_success))


class MixedEffectAction(EffectAction):
    """Effect action whose final environment row always fails planning."""

    skill_id: ClassVar[str] = "mixed_effect"
    binding_contract: ClassVar[SkillBindingContract] = DynamicAction.binding_contract

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        plan = super()._plan(request, context)
        plan_success = torch.ones_like(plan.plan_success)
        plan_success[-1] = False
        return replace(plan, plan_success=plan_success)


class NonuniformTimingAction(DynamicAction):
    """Test action with explicit nonuniform waypoint arrival intervals."""

    skill_id: ClassVar[str] = "nonuniform_timing"
    binding_contract: ClassVar[SkillBindingContract] = DynamicAction.binding_contract

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(request)
        pose = resolve_pose_goal(goal.xpos, context, name="xpos")
        target = pose[:, 0, 3].unsqueeze(1).expand_as(context.robot.qpos)
        positions = torch.stack(
            [context.robot.qpos, torch.lerp(context.robot.qpos, target, 0.5), target],
            dim=1,
        )
        dt = torch.tensor(
            [0.0, 0.1, 0.3], dtype=torch.float32, device=positions.device
        ).expand(context.batch_size, -1)
        trajectory = TimedTrajectory.from_positions(
            positions,
            env_ids=context.env_ids,
            control_dt=request.motion_policy.control_dt,
            dt=dt,
            duration=dt.sum(dim=1),
        )
        return self.build_plan(
            request,
            context,
            success=True,
            trajectory=trajectory,
        )


class DestinationSequenceAction(AtomicAction[EndEffectorPoseGoal, ActionOptions]):
    """Emit a configured destination sequence across recovery plans."""

    skill_id: ClassVar[str] = "destination_sequence"
    GoalType: ClassVar[type] = EndEffectorPoseGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(
                    SkillEndpointRequirement(endpoint_id="first"),
                    SkillEndpointRequirement(endpoint_id="second"),
                ),
            ),
        )
    )

    def __init__(self, destinations: tuple[str | None, ...]) -> None:
        super().__init__()
        self.destinations = destinations
        self.plan_count = 0

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        self.require_goal(request)
        index = min(self.plan_count, len(self.destinations) - 1)
        endpoint_id = self.destinations[index]
        self.plan_count += 1
        if endpoint_id is None:
            commands = TimedCommandSequence(frames=(), env_ids=context.env_ids)
            return self.build_command_plan(
                request,
                context,
                success=False,
                commands=commands,
            )

        target = request.binding.endpoint("primary", endpoint_id).require_target(
            JointPositionTarget
        )
        joint_ids = list(target.joint_ids)
        frame = RuntimeCommandFrame(
            commands=(
                EndpointCommand(
                    target=target,
                    payload=JointPositionPayload(
                        positions=context.robot.qpos[:, joint_ids]
                    ),
                ),
            ),
            active_mask=torch.ones(
                context.batch_size,
                dtype=torch.bool,
                device=context.robot.qpos.device,
            ),
            env_ids=context.env_ids,
            hold_duration=torch.zeros(
                context.batch_size,
                dtype=torch.float32,
                device=context.robot.qpos.device,
            ),
        )
        return self.build_command_plan(
            request,
            context,
            success=True,
            commands=TimedCommandSequence(
                frames=(frame,),
                env_ids=context.env_ids,
            ),
        )


class UncopyableEntity(BatchEntity):
    """Minimal live entity whose simulator identity must not be copied."""

    def __init__(self) -> None:
        self._pose = torch.eye(4).unsqueeze(0)

    def __deepcopy__(self, memo: dict[int, object]) -> UncopyableEntity:
        raise AssertionError("Live simulator entities must not be deep-copied.")

    def set_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        self._pose = pose.clone()

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        return self._pose.clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        return None


def _engine(batch_size: int = 1) -> tuple[AtomicActionEngine, DynamicAction]:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 2
    robot.control_parts = {"arm": object()}
    robot.get_qpos.return_value = torch.zeros(batch_size, 2)
    robot.get_qvel.return_value = torch.zeros(batch_size, 2)
    robot.get_joint_ids.return_value = [0, 1]
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub"
    generator.supports_dynamic_collision_world = False
    engine = AtomicActionEngine(generator, load_builtins=False)
    action = DynamicAction()
    engine.register(action)
    return engine, action


def _destination_engine(
    destinations: tuple[str | None, ...],
) -> tuple[AtomicActionEngine, DestinationSequenceAction]:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 2
    robot.control_parts = {"arm_a": object(), "arm_b": object()}
    robot.get_qpos.return_value = torch.zeros(1, 2)
    robot.get_qvel.return_value = torch.zeros(1, 2)
    robot.get_joint_ids.side_effect = lambda *, name: {
        "arm_a": [0],
        "arm_b": [1],
    }[name]
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub"
    generator.supports_dynamic_collision_world = False
    engine = AtomicActionEngine(generator, load_builtins=False)
    action = DestinationSequenceAction(destinations)
    engine.register(action)
    return engine, action


def _context(
    timestamp: float,
    qpos: float | tuple[float, ...],
    entity_x: float | tuple[float, ...],
    version: int,
) -> PlanningContext:
    qpos_values = torch.as_tensor(qpos, dtype=torch.float32).reshape(-1)
    entity_x_values = torch.as_tensor(entity_x, dtype=torch.float32).reshape(-1)
    if qpos_values.shape != entity_x_values.shape:
        raise ValueError("qpos and entity_x must describe the same batch.")
    batch_size = int(qpos_values.shape[0])
    positions = qpos_values[:, None].expand(-1, 2).clone()
    pose = torch.eye(4).repeat(batch_size, 1, 1)
    pose[:, 0, 3] = entity_x_values
    return PlanningContext(
        robot=RobotObservation(
            timestamp=timestamp,
            qpos=positions,
            qvel=torch.zeros_like(positions),
        ),
        task=TaskState.empty(batch_size=batch_size, device="cpu"),
        scene=SceneSnapshot(
            timestamp=timestamp,
            version=version,
            entities={"target": EntityState(pose)},
        ),
        env_ids=torch.arange(batch_size, dtype=torch.long),
    )


def _collision_context(
    timestamp: float,
    qpos: torch.Tensor,
    obstacle_x: torch.Tensor,
    collision_revision: int | tuple[int, ...],
) -> PlanningContext:
    """Build a scene whose obstacle is independent from the action goal."""
    batch_size = int(qpos.shape[0])
    target_pose = torch.eye(4).repeat(batch_size, 1, 1)
    target_pose[:, 0, 3] = 0.2
    obstacle_pose = torch.eye(4).repeat(batch_size, 1, 1)
    obstacle_pose[:, 0, 3] = obstacle_x
    return PlanningContext(
        robot=RobotObservation(
            timestamp=timestamp,
            qpos=qpos,
            qvel=torch.zeros_like(qpos),
        ),
        task=TaskState.empty(batch_size=batch_size, device="cpu"),
        scene=SceneSnapshot(
            timestamp=timestamp,
            version=int(timestamp > 0.0),
            entities={
                "target": EntityState(target_pose),
                "obstacle": EntityState(obstacle_pose),
            },
            collision_world_revision=collision_revision,
            collision_entity_ids=("obstacle",),
        ),
        env_ids=torch.arange(batch_size, dtype=torch.long),
    )


def _invocation(
    engine: AtomicActionEngine,
    *,
    skill_id: str = "dynamic",
    max_replans: int = 2,
    max_action_retries: int = 2,
    action_timeout: float = 30.0,
    control_dt: float = 1.0 / 60.0,
    strategy: str = "ik_interp",
    dynamic_collision_mode: DynamicCollisionMode = DynamicCollisionMode.AUTO,
) -> ActionInvocation[EndEffectorPoseGoal]:
    return ActionInvocation(
        skill_id=skill_id,
        goal=EndEffectorPoseGoal(SceneEntityPose("target")),
        binding=engine.planning_services.bind_control_parts(
            DynamicAction.binding_contract,
            {"primary": {"motion": "arm"}},
        ),
        motion_policy=MotionPolicy(
            sample_count=2,
            control_dt=control_dt,
            strategy=strategy,
            dynamic_collision_mode=dynamic_collision_mode,
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=max_replans,
            max_action_retries=max_action_retries,
            tracking_error_threshold=0.05,
            goal_translation_threshold=0.02,
            action_timeout=action_timeout,
        ),
        invocation_id="dynamic-call",
    )


def _destination_invocation(
    engine: AtomicActionEngine,
) -> ActionInvocation[EndEffectorPoseGoal]:
    return ActionInvocation(
        skill_id=DestinationSequenceAction.skill_id,
        goal=EndEffectorPoseGoal(SceneEntityPose("target")),
        binding=engine.bind_control_parts(
            DestinationSequenceAction.skill_id,
            {
                "primary": {
                    "first": "arm_a",
                    "second": "arm_b",
                }
            },
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            max_action_retries=1,
            goal_translation_threshold=0.02,
        ),
        invocation_id="destination-call",
    )


def _effect_session(
    *,
    batch_size: int = 1,
    max_action_retries: int = 2,
    action_timeout: float = 30.0,
    eligible_mask: torch.Tensor | None = None,
    action: EffectAction | None = None,
) -> tuple[ExecutionSession, ExecutionTick]:
    """Advance a test effect action to its verification boundary."""
    engine, _ = _engine(batch_size=batch_size)
    selected_action = EffectAction() if action is None else action
    engine.register(selected_action)
    base = _invocation(
        engine,
        max_action_retries=max_action_retries,
        action_timeout=action_timeout,
    )
    invocation = ActionInvocation(
        skill_id=selected_action.skill_id,
        goal=base.goal,
        binding=base.binding,
        motion_policy=base.motion_policy,
        recovery_policy=base.recovery_policy,
    )
    qpos = tuple(0.0 for _ in range(batch_size))
    target = tuple(0.2 for _ in range(batch_size))
    session = engine.start(
        (invocation,),
        _context(0.0, qpos, target, 0),
        eligible_mask=eligible_mask,
    )
    session.tick(_context(0.0, qpos, target, 0))
    session.tick(_context(0.1, qpos, target, 0))
    waiting = session.tick(_context(0.2, target, target, 0))
    assert waiting.pending_effect is not None
    return session, waiting


def _joint_positions(command: RuntimeCommandFrame | None) -> torch.Tensor:
    """Return the only joint-position payload emitted by the test action."""
    assert command is not None
    assert len(command.commands) == 1
    payload = command.commands[0].payload
    assert isinstance(payload, JointPositionPayload)
    return payload.positions


def test_session_completes_incremental_command_sequence() -> None:
    engine, _ = _engine()
    session = engine.start((_invocation(engine),), _context(0.0, 0.0, 0.2, 0))

    first = session.tick(_context(0.0, 0.0, 0.2, 0))
    second = session.tick(_context(0.1, 0.0, 0.2, 0))
    final = session.tick(_context(0.2, 0.2, 0.2, 0))

    assert torch.all(_joint_positions(first.command) == 0.0)
    assert all(event.invocation_id == "dynamic-call" for event in first.events)
    assert torch.all(_joint_positions(second.command) == 0.2)
    assert final.status is ExecutionStatus.COMPLETED
    assert final.eligible_mask.tolist() == [True]


def test_initial_eligibility_is_sticky_across_invocation_barriers() -> None:
    engine, _ = _engine(batch_size=2)
    invocation = _invocation(engine)
    supplied_mask = torch.tensor([True, False])
    session = engine.start(
        (invocation, invocation),
        _context(0.0, (0.0, 0.0), (0.2, 0.2), 0),
        eligible_mask=supplied_mask,
    )
    supplied_mask.fill_(True)

    first = session.tick(_context(0.0, (0.0, 0.0), (0.2, 0.2), 0))
    session.tick(_context(0.1, (0.0, 0.0), (0.2, 0.2), 0))
    barrier = session.tick(_context(0.2, (0.2, 7.0), (0.2, 0.2), 0))
    second_action = session.tick(_context(0.3, (0.2, 7.0), (0.2, 0.2), 0))

    assert first.command is not None
    assert first.command.active_mask.tolist() == [True, False]
    assert barrier.status is ExecutionStatus.RUNNING
    assert second_action.command is not None
    assert second_action.command.active_mask.tolist() == [True, False]
    assert second_action.eligible_mask.tolist() == [True, False]


def test_empty_initial_eligibility_fails_without_planning() -> None:
    engine, action = _engine(batch_size=2)
    initial = _context(0.0, (0.0, 0.0), (0.2, 0.2), 0)

    session = engine.start(
        (_invocation(engine),),
        initial,
        eligible_mask=torch.tensor([False, False]),
    )
    terminal = session.tick(initial)

    assert action.plan_count == 0
    assert terminal.status is ExecutionStatus.FAILED
    assert terminal.eligible_mask.tolist() == [False, False]
    assert any(
        event.kind is ExecutionEventKind.SESSION_FAILED for event in terminal.events
    )


def test_initial_eligibility_is_owned_and_validated() -> None:
    engine, _ = _engine(batch_size=2)
    initial = _context(0.0, (0.0, 0.0), (0.2, 0.2), 0)

    with pytest.raises(TypeError, match="eligible_mask must be a torch.Tensor"):
        engine.start((_invocation(engine),), initial, eligible_mask=[True, False])
    with pytest.raises(ValueError, match="bool with shape"):
        engine.start(
            (_invocation(engine),),
            initial,
            eligible_mask=torch.tensor([1, 0]),
        )
    with pytest.raises(ValueError, match="bool with shape"):
        engine.start(
            (_invocation(engine),),
            initial,
            eligible_mask=torch.tensor([True]),
        )

    supplied = torch.tensor([True, False])
    session = engine.start(
        (_invocation(engine),),
        initial,
        eligible_mask=supplied,
    )
    supplied.fill_(False)
    observed = session.eligible_mask
    observed.fill_(False)

    assert session.eligible_mask.tolist() == [True, False]


def test_deactivate_rows_is_sticky_and_masks_the_next_command() -> None:
    engine, _ = _engine(batch_size=2)
    initial = _context(0.0, (0.0, 0.0), (0.2, 0.2), 0)
    session = engine.start((_invocation(engine),), initial)
    session.tick(initial)

    changed = session.deactivate_rows(
        torch.tensor([False, True]),
        reason="environment terminated",
    )
    unchanged = session.deactivate_rows(
        torch.tensor([False, True]),
        reason="duplicate termination",
    )
    tick = session.tick(_context(0.1, (0.0, 0.0), (0.2, 0.2), 0))

    assert changed.tolist() == [False, True]
    assert unchanged.tolist() == [False, False]
    assert tick.command is not None
    assert tick.command.active_mask.tolist() == [True, False]
    assert tick.eligible_mask.tolist() == [True, False]
    deactivated = [
        event
        for event in tick.events
        if event.kind is ExecutionEventKind.ROWS_DEACTIVATED
    ]
    assert len(deactivated) == 1
    assert deactivated[0].env_mask.tolist() == [False, True]
    assert deactivated[0].message == "environment terminated"


def test_session_commands_schedule_arrivals_and_final_settling() -> None:
    engine, _ = _engine()
    engine.register(NonuniformTimingAction())
    session = engine.start(
        (_invocation(engine, skill_id="nonuniform_timing"),),
        _context(0.0, 0.0, 0.2, 0),
    )

    first = session.tick(_context(0.0, 0.0, 0.2, 0))
    second = session.tick(_context(0.0, 0.0, 0.2, 0))
    third = session.tick(_context(0.1, 0.1, 0.2, 0))

    assert first.command is not None
    assert second.command is not None
    assert third.command is not None
    command_durations = torch.stack(
        [
            first.command.hold_duration,
            second.command.hold_duration,
            third.command.hold_duration,
        ],
        dim=1,
    )
    assert torch.allclose(command_durations, torch.tensor([[0.1, 0.3, 0.3]]))
    assert torch.allclose(command_durations[:, :-1].sum(dim=1), torch.tensor([0.4]))


def test_request_snapshot_preserves_live_entity_identity() -> None:
    entity = UncopyableEntity()
    grasp_xpos = torch.eye(4).unsqueeze(0)
    geometry_extent = torch.tensor([0.1, 0.2, 0.3])
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={"extent": geometry_extent},
        label="object",
        entity=entity,
    )
    goal = GraspGoal(semantics=semantics, grasp_xpos=grasp_xpos)

    request = ResolvedActionRequest(
        skill_id="pick_up",
        goal=goal,
        binding=ActionBinding(owner_id="snapshot-test"),
        motion_policy=MotionPolicy(),
        recovery_policy=RecoveryPolicy(),
        skill_options=ActionOptions(),
    )
    grasp_xpos.fill_(9.0)
    geometry_extent.fill_(9.0)

    assert request.goal is not goal
    assert request.goal.semantics is not semantics
    assert request.goal.semantics.entity is entity
    assert torch.equal(request.goal.grasp_xpos, torch.eye(4).unsqueeze(0))
    assert torch.equal(
        request.goal.semantics.geometry["extent"],
        torch.tensor([0.1, 0.2, 0.3]),
    )


def test_scene_motion_replans_late_bound_goal() -> None:
    engine, action = _engine()
    session = engine.start((_invocation(engine),), _context(0.0, 0.0, 0.1, 0))
    session.tick(_context(0.0, 0.0, 0.1, 0))

    tick = session.tick(_context(0.1, 0.0, 0.3, 1))

    kinds = {event.kind for event in tick.events}
    assert ExecutionEventKind.DYNAMIC_GOAL_CHANGED in kinds
    assert ExecutionEventKind.REPLANNED in kinds
    assert action.plan_count == 2
    assert action.requests[0] is action.requests[1]
    assert tick.command is not None


def test_recovery_replan_rejects_runtime_destination_change() -> None:
    engine, action = _destination_engine(("first", "second"))
    invocation = _destination_invocation(engine)
    initial = _context(0.0, 0.0, 0.1, 0)
    session = engine.start((invocation,), initial)

    activated = session.tick(initial)
    assert activated.command is not None
    assert activated.command.commands[0].target.target_id == "arm_a"

    with pytest.raises(
        ValueError,
        match="Recovery replans must preserve the active runtime destination set",
    ) as exc_info:
        session.tick(_context(0.1, 0.0, 0.3, 1))

    assert "arm_a" in str(exc_info.value)
    assert "arm_b" in str(exc_info.value)
    assert action.plan_count == 2


def test_empty_failed_replan_preserves_destination_for_same_target_retry() -> None:
    engine, action = _destination_engine(("first", None, "first"))
    invocation = _destination_invocation(engine)
    initial = _context(0.0, 0.0, 0.1, 0)
    session = engine.start((invocation,), initial)

    activated = session.tick(initial)
    assert activated.command is not None
    assert activated.command.commands[0].target.target_id == "arm_a"

    recovered = session.tick(_context(0.1, 0.0, 0.3, 1))

    kinds = [event.kind for event in recovered.events]
    assert ExecutionEventKind.DYNAMIC_GOAL_CHANGED in kinds
    assert ExecutionEventKind.ACTION_RETRY in kinds
    assert kinds.count(ExecutionEventKind.REPLANNED) == 2
    assert action.plan_count == 3
    assert recovered.command is None
    assert [target.target_id for target in recovered.hold_targets] == ["arm_a"]

    resumed = session.tick(_context(0.2, 0.0, 0.3, 1))
    assert resumed.command is not None
    assert resumed.command.commands[0].target.target_id == "arm_a"


def test_collision_world_change_replans_with_latest_obstacle_pose() -> None:
    engine, action = _engine()
    generator = engine.motion_generator
    generator.supports_dynamic_collision_world = True
    generator.bind_collision_world.side_effect = (
        lambda options, *, obstacle_poses: options or PlanOptions()
    )
    initial_qpos = torch.zeros(1, 2)
    initial = _collision_context(
        0.0,
        initial_qpos,
        torch.tensor([0.4]),
        (0,),
    )
    session = engine.start(
        (_invocation(engine, strategy="motion_gen"),),
        initial,
    )
    session.tick(initial)

    changed = _collision_context(
        0.1,
        initial_qpos,
        torch.tensor([0.6]),
        (1,),
    )
    tick = session.tick(changed)

    kinds = {event.kind for event in tick.events}
    assert ExecutionEventKind.COLLISION_WORLD_CHANGED in kinds
    assert ExecutionEventKind.REPLANNED in kinds
    assert action.plan_count == 2
    latest_obstacles = generator.bind_collision_world.call_args.kwargs["obstacle_poses"]
    assert latest_obstacles["obstacle"][0, 0, 3] == pytest.approx(0.6)
    assert tick.command is not None


def test_collision_world_exhaustion_only_disables_changed_environment() -> None:
    engine, _ = _engine(batch_size=2)
    generator = engine.motion_generator
    generator.supports_dynamic_collision_world = True
    generator.bind_collision_world.side_effect = (
        lambda options, *, obstacle_poses: options or PlanOptions()
    )
    qpos = torch.zeros(2, 2)
    initial = _collision_context(
        0.0,
        qpos,
        torch.tensor([0.4, 0.4]),
        (0, 0),
    )
    session = engine.start(
        (
            _invocation(
                engine,
                max_replans=0,
                strategy="motion_gen",
            ),
        ),
        initial,
    )
    session.tick(initial)

    changed = _collision_context(
        0.1,
        qpos,
        torch.tensor([0.4, 0.6]),
        (0, 1),
    )
    tick = session.tick(changed)

    collision_event = next(
        event
        for event in tick.events
        if event.kind is ExecutionEventKind.COLLISION_WORLD_CHANGED
    )
    exhausted_event = next(
        event
        for event in tick.events
        if event.kind is ExecutionEventKind.RECOVERY_EXHAUSTED
    )
    assert collision_event.env_mask.tolist() == [False, True]
    assert exhausted_event.env_mask.tolist() == [False, True]
    assert tick.status is ExecutionStatus.RUNNING
    assert tick.eligible_mask.tolist() == [True, False]
    assert tick.command is not None
    assert tick.command.active_mask.tolist() == [True, False]


def test_dynamic_collision_off_skips_binding_and_revision_recovery() -> None:
    engine, action = _engine()
    generator = engine.motion_generator
    generator.supports_dynamic_collision_world = True
    initial_qpos = torch.zeros(1, 2)
    initial = _collision_context(
        0.0,
        initial_qpos,
        torch.tensor([0.4]),
        (0,),
    )
    session = engine.start(
        (
            _invocation(
                engine,
                strategy="motion_gen",
                dynamic_collision_mode=DynamicCollisionMode.OFF,
            ),
        ),
        initial,
    )
    session.tick(initial)

    changed = _collision_context(
        0.1,
        initial_qpos,
        torch.tensor([0.6]),
        (1,),
    )
    tick = session.tick(changed)

    assert ExecutionEventKind.COLLISION_WORLD_CHANGED not in {
        event.kind for event in tick.events
    }
    assert action.plan_count == 1
    generator.bind_collision_world.assert_not_called()


def test_required_dynamic_collision_rejects_incompatible_strategy() -> None:
    engine, _ = _engine()
    engine.motion_generator.supports_dynamic_collision_world = True

    with pytest.raises(ValueError, match="strategy='motion_gen'"):
        engine.plan(
            _invocation(
                engine,
                dynamic_collision_mode=DynamicCollisionMode.REQUIRED,
            ),
            _collision_context(
                0.0,
                torch.zeros(1, 2),
                torch.tensor([0.4]),
                (0,),
            ),
        )


def test_required_dynamic_collision_rejects_missing_scene_entities() -> None:
    engine, _ = _engine()
    engine.motion_generator.supports_dynamic_collision_world = True

    with pytest.raises(ValueError, match="scene collision entities"):
        engine.plan(
            _invocation(
                engine,
                strategy="motion_gen",
                dynamic_collision_mode=DynamicCollisionMode.REQUIRED,
            ),
            _context(0.0, 0.0, 0.2, 0),
        )


def test_required_dynamic_collision_rejects_unsupported_planner() -> None:
    engine, _ = _engine()

    with pytest.raises(ValueError, match="dynamic collision-world support"):
        engine.plan(
            _invocation(
                engine,
                strategy="motion_gen",
                dynamic_collision_mode=DynamicCollisionMode.REQUIRED,
            ),
            _collision_context(
                0.0,
                torch.zeros(1, 2),
                torch.tensor([0.4]),
                (0,),
            ),
        )


def test_required_dynamic_collision_binds_supported_scene() -> None:
    engine, _ = _engine()
    generator = engine.motion_generator
    generator.supports_dynamic_collision_world = True
    generator.bind_collision_world.side_effect = (
        lambda options, *, obstacle_poses: options or PlanOptions()
    )

    plan = engine.plan(
        _invocation(
            engine,
            strategy="motion_gen",
            dynamic_collision_mode=DynamicCollisionMode.REQUIRED,
        ),
        _collision_context(
            0.0,
            torch.zeros(1, 2),
            torch.tensor([0.4]),
            (0,),
        ),
    )

    assert plan.collision_world_sensitive is True
    generator.bind_collision_world.assert_called_once()


def test_resolved_goal_snapshot_is_reused_during_recovery() -> None:
    engine, action = _engine()
    target = torch.eye(4).unsqueeze(0)
    target[:, 0, 3] = 0.2
    base = _invocation(engine)
    invocation = ActionInvocation(
        skill_id=base.skill_id,
        goal=EndEffectorPoseGoal(target),
        binding=base.binding,
        motion_policy=base.motion_policy,
        recovery_policy=base.recovery_policy,
        invocation_id=base.invocation_id,
    )
    session = engine.start((invocation,), _context(0.0, 0.0, 0.0, 0))
    target[:, 0, 3] = 0.8
    session.tick(_context(0.0, 0.0, 0.0, 0))

    session.tick(_context(0.1, 1.0, 0.0, 0))

    assert action.plan_count == 2
    assert action.requests[0] is action.requests[1]
    snapshot = action.requests[1].goal.xpos
    assert isinstance(snapshot, torch.Tensor)
    assert torch.equal(snapshot[:, 0, 3], torch.tensor([0.2]))


def test_subset_replan_restarts_synchronized_active_cohort() -> None:
    engine, action = _engine(batch_size=2)
    session = engine.start(
        (_invocation(engine),),
        _context(0.0, (0.0, 0.0), (0.1, 0.2), 0),
    )
    session.tick(_context(0.0, (0.0, 0.0), (0.1, 0.2), 0))

    replanned = session.tick(_context(0.1, (0.0, 0.0), (0.4, 0.2), 1))
    next_command = session.tick(_context(0.2, (0.0, 0.0), (0.4, 0.2), 1))

    changed = next(
        event
        for event in replanned.events
        if event.kind is ExecutionEventKind.DYNAMIC_GOAL_CHANGED
    )
    cohort = next(
        event
        for event in replanned.events
        if event.kind is ExecutionEventKind.REPLANNED
    )
    assert changed.env_mask.tolist() == [True, False]
    assert cohort.env_mask.tolist() == [True, True]
    assert replanned.eligible_mask.tolist() == [True, True]
    assert torch.all(_joint_positions(replanned.command) == 0.0)
    assert torch.equal(
        _joint_positions(next_command.command)[:, 0],
        torch.tensor([0.4, 0.2]),
    )
    assert action.plan_count == 2


def test_replan_exhaustion_disables_only_triggering_row() -> None:
    engine, _ = _engine(batch_size=2)
    session = engine.start(
        (_invocation(engine, max_replans=1),),
        _context(0.0, (0.0, 0.0), (0.1, 0.2), 0),
    )
    session.tick(_context(0.0, (0.0, 0.0), (0.1, 0.2), 0))
    session.tick(_context(0.1, (0.0, 0.0), (0.4, 0.2), 1))

    exhausted = session.tick(_context(0.2, (0.0, 0.0), (0.6, 0.2), 2))

    event = next(
        event
        for event in exhausted.events
        if event.kind is ExecutionEventKind.RECOVERY_EXHAUSTED
    )
    assert event.env_mask.tolist() == [True, False]
    assert exhausted.eligible_mask.tolist() == [False, True]
    assert exhausted.status is ExecutionStatus.RUNNING
    assert exhausted.command is not None
    assert exhausted.command.active_mask.tolist() == [False, True]


def test_action_retry_resets_replan_budget_only_for_allowed_rows() -> None:
    engine, _ = _engine(batch_size=2)
    session = engine.start(
        (
            _invocation(
                engine,
                max_replans=1,
                max_action_retries=1,
            ),
        ),
        _context(0.0, (0.0, 0.0), (0.1, 0.2), 0),
    )
    session.tick(_context(0.0, (0.0, 0.0), (0.1, 0.2), 0))

    row_b_replan = session.tick(_context(0.1, (0.0, 0.0), (0.1, 0.4), 1))
    changed = next(
        event
        for event in row_b_replan.events
        if event.kind is ExecutionEventKind.DYNAMIC_GOAL_CHANGED
    )
    assert changed.env_mask.tolist() == [False, True]

    retry_events = session._attempt_action_retry(
        torch.tensor([True, False]),
        ExecutionEventKind.ACTION_TIMEOUT,
        "Row A starts a new action attempt.",
    )
    retried = next(
        event for event in retry_events if event.kind is ExecutionEventKind.ACTION_RETRY
    )
    assert retried.env_mask.tolist() == [True, False]

    row_b_exhausted = session.tick(_context(0.2, (0.0, 0.0), (0.1, 0.6), 2))
    exhausted = next(
        event
        for event in row_b_exhausted.events
        if event.kind is ExecutionEventKind.RECOVERY_EXHAUSTED
    )
    assert exhausted.env_mask.tolist() == [False, True]
    assert row_b_exhausted.eligible_mask.tolist() == [True, False]


def test_session_revision_replans_from_latest_context() -> None:
    engine, action = _engine()
    original = _invocation(engine)
    session = engine.start((original,), _context(0.0, 0.0, 0.1, 0))
    revised_pose = torch.eye(4).unsqueeze(0)
    revised_pose[:, 0, 3] = 0.8
    revised = ActionInvocation(
        skill_id=original.skill_id,
        goal=EndEffectorPoseGoal(revised_pose),
        binding=original.binding,
        motion_policy=original.motion_policy,
        recovery_policy=original.recovery_policy,
        invocation_id=original.invocation_id,
        revision=1,
    )

    session.revise_current(revised)
    first = session.tick(_context(0.0, 0.0, 0.1, 0))
    second = session.tick(_context(0.1, 0.0, 0.1, 0))

    assert action.plan_count == 2
    assert action.requests[0] is not action.requests[1]
    assert [request.revision for request in action.requests] == [0, 1]
    assert any(
        event.kind is ExecutionEventKind.INVOCATION_REVISED
        and event.invocation_revision == 1
        for event in first.events
    )
    assert torch.all(_joint_positions(second.command) == 0.8)


def test_session_revision_must_advance_same_invocation() -> None:
    engine, _ = _engine()
    original = _invocation(engine)
    session = engine.start((original,), _context(0.0, 0.0, 0.1, 0))

    with pytest.raises(ValueError, match="must advance"):
        session.revise_current(original)

    with pytest.raises(ValueError, match="invocation_id"):
        session.revise_current(
            ActionInvocation(
                skill_id=original.skill_id,
                goal=original.goal,
                binding=original.binding,
                motion_policy=original.motion_policy,
                recovery_policy=original.recovery_policy,
                invocation_id="another-call",
                revision=1,
            )
        )


def test_session_revision_rejects_runtime_destination_change() -> None:
    engine, action = _destination_engine(("first", "second"))
    invocation = _destination_invocation(engine)
    initial = _context(0.0, 0.0, 0.1, 0)
    session = engine.start((invocation,), initial)

    with pytest.raises(
        ValueError,
        match="Invocation revisions must preserve the active runtime destination set",
    ) as exc_info:
        session.revise_current(replace(invocation, revision=1))

    assert "Start a new invocation" in str(exc_info.value)
    assert "arm_a" in str(exc_info.value)
    assert "arm_b" in str(exc_info.value)
    assert action.plan_count == 2

    active = session.tick(initial)
    assert active.command is not None
    assert active.command.commands[0].target.target_id == "arm_a"


def test_session_revision_rejects_empty_target_plan() -> None:
    engine, action = _destination_engine(("first", None))
    invocation = _destination_invocation(engine)
    initial = _context(0.0, 0.0, 0.1, 0)
    session = engine.start((invocation,), initial)

    with pytest.raises(ValueError, match="empty replacement plan"):
        session.revise_current(replace(invocation, revision=1))

    assert action.plan_count == 2
    active = session.tick(initial)
    assert active.command is not None
    assert active.command.commands[0].target.target_id == "arm_a"


def test_session_revision_rejects_changed_target_address_fingerprint() -> None:
    engine, action = _engine()
    invocation = _invocation(engine)
    initial = _context(0.0, 0.0, 0.1, 0)
    session = engine.start((invocation,), initial)
    endpoint = invocation.binding.endpoint("primary", "motion")
    changed_endpoint = EndpointBinding(
        slot_id=endpoint.slot_id,
        endpoint_id=endpoint.endpoint_id,
        resource_id=endpoint.resource_id,
        adapter_id=endpoint.adapter_id,
        target=JointPositionTarget(control_part="arm", joint_ids=(0,)),
        capabilities=endpoint.capabilities,
        commands=endpoint.commands,
        claim_tokens=endpoint.claim_tokens,
        joint_ids=(0,),
    )
    revised = replace(
        invocation,
        binding=ActionBinding(
            owner_id=invocation.binding.owner_id,
            endpoints=(changed_endpoint,),
        ),
        revision=1,
    )

    with pytest.raises(ValueError, match="address fingerprint"):
        session.revise_current(revised)

    assert action.plan_count == 2
    active = session.tick(initial)
    assert active.command is not None
    target = active.command.commands[0].target
    assert isinstance(target, JointPositionTarget)
    assert target.joint_ids == (0, 1)


def test_tracking_error_fails_when_replan_budget_is_zero() -> None:
    engine, _ = _engine()
    session = engine.start(
        (_invocation(engine, max_replans=0),),
        _context(0.0, 0.0, 0.2, 0),
    )
    session.tick(_context(0.0, 0.0, 0.2, 0))

    tick = session.tick(_context(0.1, 1.0, 0.2, 0))

    kinds = {event.kind for event in tick.events}
    assert ExecutionEventKind.TRACKING_ERROR in kinds
    assert ExecutionEventKind.RECOVERY_EXHAUSTED in kinds
    assert tick.status is ExecutionStatus.FAILED
    assert tick.eligible_mask.tolist() == [False]


def test_action_timeout_retry_budget_is_bounded() -> None:
    engine, action = _engine()
    session = engine.start(
        (
            _invocation(
                engine,
                max_action_retries=1,
                action_timeout=0.05,
            ),
        ),
        _context(0.0, 0.0, 0.2, 0),
    )
    session.tick(_context(0.0, 0.0, 0.2, 0))

    retry = session.tick(_context(0.1, 0.0, 0.2, 0))
    exhausted = session.tick(_context(0.2, 0.0, 0.2, 0))

    retry_kinds = {event.kind for event in retry.events}
    assert ExecutionEventKind.ACTION_TIMEOUT in retry_kinds
    assert ExecutionEventKind.ACTION_RETRY in retry_kinds
    assert ExecutionEventKind.REPLANNED in retry_kinds
    assert action.plan_count == 2
    exhausted_kinds = {event.kind for event in exhausted.events}
    assert ExecutionEventKind.ACTION_TIMEOUT in exhausted_kinds
    assert ExecutionEventKind.RECOVERY_EXHAUSTED in exhausted_kinds
    assert exhausted.status is ExecutionStatus.FAILED
    assert exhausted.eligible_mask.tolist() == [False]


def test_session_rejects_changed_environment_identity() -> None:
    engine, _ = _engine()
    initial = _context(0.0, 0.0, 0.2, 0)
    session = engine.start((_invocation(engine),), initial)
    changed = PlanningContext(
        robot=initial.robot,
        task=initial.task,
        scene=initial.scene,
        env_ids=torch.tensor([7], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="env_ids must remain stable"):
        session.tick(changed)


def test_session_rejects_regressing_scene_snapshot() -> None:
    engine, _ = _engine()
    session = engine.start((_invocation(engine),), _context(1.0, 0.0, 0.2, 2))

    with pytest.raises(ValueError, match="versions must be monotonic"):
        session.tick(_context(1.0, 0.0, 0.2, 1))


def test_session_rejects_regressing_collision_world_revision() -> None:
    engine, _ = _engine()
    qpos = torch.zeros(1, 2)
    initial = _collision_context(0.0, qpos, torch.tensor([0.4]), (2,))
    session = engine.start(
        (_invocation(engine, strategy="motion_gen"),),
        initial,
    )
    regressed = _collision_context(0.1, qpos, torch.tensor([0.4]), (1,))

    with pytest.raises(ValueError, match="Collision-world revisions"):
        session.tick(regressed)


def test_nonempty_effect_is_committed_only_after_external_verification() -> None:
    engine, _ = _engine()
    effect = EffectAction()
    engine.register(effect)
    invocation = _invocation(engine)
    invocation = ActionInvocation(
        skill_id="effect",
        goal=invocation.goal,
        binding=invocation.binding,
        motion_policy=invocation.motion_policy,
        recovery_policy=invocation.recovery_policy,
    )
    session = engine.start((invocation,), _context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.1, 0.0, 0.2, 0))

    waiting = session.tick(_context(0.2, 0.2, 0.2, 0))
    still_waiting = session.tick(_context(0.25, 0.2, 0.2, 0))
    completed = session.tick(
        _context(0.3, 0.2, 0.2, 0),
        effect_result=EffectVerificationResult(
            waiting.pending_effect.verification_id,
            torch.tensor([True]),
            torch.tensor([False]),
        ),
    )

    assert waiting.status is ExecutionStatus.RUNNING
    assert waiting.task_state.get_held_object("arm") is None
    assert waiting.pending_effect is not None
    assert waiting.pending_effect.skill_id == "effect"
    assert waiting.pending_effect.terminal_segment == "effect"
    assert waiting.pending_effect.env_mask.tolist() == [True]
    assert not waiting.pending_effect.expected_effects.is_empty
    assert any(
        event.kind is ExecutionEventKind.EFFECT_VERIFICATION_REQUIRED
        for event in waiting.events
    )
    assert still_waiting.pending_effect is not None
    assert not any(
        event.kind
        in {
            ExecutionEventKind.TRAJECTORY_COMPLETED,
            ExecutionEventKind.EFFECT_VERIFICATION_REQUIRED,
        }
        for event in still_waiting.events
    )
    assert completed.status is ExecutionStatus.COMPLETED
    assert completed.pending_effect is None
    assert completed.task_state.get_held_object("arm") is not None


def test_initially_ineligible_rows_never_receive_effects() -> None:
    session, waiting = _effect_session(
        batch_size=2,
        eligible_mask=torch.tensor([True, False]),
    )
    request = waiting.pending_effect
    assert request is not None
    assert request.env_mask.tolist() == [True, False]

    completed = session.tick(
        _context(0.21, (0.2, 0.2), (0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            request.verification_id,
            success_mask=torch.tensor([True, False]),
            failure_mask=torch.tensor([False, False]),
        ),
    )

    held = completed.task_state.get_held_object("arm")
    assert completed.status is ExecutionStatus.COMPLETED
    assert completed.eligible_mask.tolist() == [True, False]
    assert held is not None and held.env_mask is not None
    assert held.env_mask.tolist() == [True, False]


def test_partial_effect_success_commits_resolved_rows_and_shrinks_request() -> None:
    session, waiting = _effect_session(batch_size=2)
    first_request = waiting.pending_effect
    assert first_request is not None

    no_progress = session.tick(
        _context(0.205, (0.2, 0.2), (0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            first_request.verification_id,
            success_mask=torch.tensor([False, False]),
            failure_mask=torch.tensor([False, False]),
        ),
    )
    assert no_progress.pending_effect is not None
    assert no_progress.pending_effect.verification_id == first_request.verification_id

    partial = session.tick(
        _context(0.21, (0.2, 0.2), (0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            first_request.verification_id,
            success_mask=torch.tensor([True, False]),
            failure_mask=torch.tensor([False, False]),
        ),
    )

    held = partial.task_state.get_held_object("arm")
    assert held is not None and held.env_mask is not None
    assert held.env_mask.tolist() == [True, False]
    assert partial.pending_effect is not None
    assert partial.pending_effect.env_mask.tolist() == [False, True]
    assert partial.pending_effect.verification_id != first_request.verification_id
    assert partial.pending_effect.attempt_generation == first_request.attempt_generation
    assert partial.pending_effect.requested_at == first_request.requested_at
    assert partial.pending_effect.deadline == first_request.deadline
    assert not any(
        event.kind is ExecutionEventKind.ACTION_RETRY for event in partial.events
    )

    with pytest.raises(ValueError, match="verification_id"):
        session.tick(
            _context(0.22, (0.2, 0.2), (0.2, 0.2), 0),
            effect_result=EffectVerificationResult(
                first_request.verification_id,
                success_mask=torch.tensor([False, True]),
                failure_mask=torch.tensor([False, False]),
            ),
        )

    current_request = partial.pending_effect
    completed = session.tick(
        _context(0.23, (0.2, 0.2), (0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            current_request.verification_id,
            success_mask=torch.tensor([False, True]),
            failure_mask=torch.tensor([False, False]),
        ),
    )

    completed_held = completed.task_state.get_held_object("arm")
    assert completed.status is ExecutionStatus.COMPLETED
    assert completed_held is not None and completed_held.env_mask is not None
    assert completed_held.env_mask.tolist() == [True, True]


def test_effect_result_masks_are_owned_disjoint_and_request_scoped() -> None:
    success = torch.tensor([True, False])
    failure = torch.tensor([False, True])
    result = EffectVerificationResult(0, success, failure)
    success.fill_(False)
    failure.fill_(False)
    assert result.success_mask.tolist() == [True, False]
    assert result.failure_mask.tolist() == [False, True]

    with pytest.raises(ValueError, match="must not overlap"):
        EffectVerificationResult(
            0,
            torch.tensor([True, False]),
            torch.tensor([True, False]),
        )

    session, waiting = _effect_session(batch_size=2)
    request = waiting.pending_effect
    assert request is not None
    request.env_mask.fill_(False)
    published_effect = request.expected_effects.held_object_updates["arm"]
    assert published_effect is not None
    published_effect.object_to_eef.fill_(9.0)
    published_effect.grasp_xpos.fill_(8.0)
    published_effect.semantics.affordance.set_custom_config("mutated", True)
    preserved = session.pending_effect
    assert preserved is not None
    assert preserved.env_mask.tolist() == [True, True]
    preserved_effect = preserved.expected_effects.held_object_updates["arm"]
    assert preserved_effect is not None
    assert torch.equal(preserved_effect.object_to_eef, torch.eye(4))
    assert torch.equal(preserved_effect.grasp_xpos, torch.eye(4))
    assert preserved_effect.semantics.affordance.get_custom_config("mutated") is None

    partial = session.tick(
        _context(0.21, (0.2, 0.2), (0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            preserved.verification_id,
            success_mask=torch.tensor([True, False]),
            failure_mask=torch.tensor([False, False]),
        ),
    )
    current = partial.pending_effect
    assert current is not None
    held = partial.task_state.get_held_object("arm")
    assert held is not None
    assert torch.equal(held.object_to_eef[0], torch.eye(4))

    with pytest.raises(ValueError, match="subsets"):
        session.tick(
            _context(0.22, (0.2, 0.2), (0.2, 0.2), 0),
            effect_result=EffectVerificationResult(
                current.verification_id,
                success_mask=torch.tensor([True, False]),
                failure_mask=torch.tensor([False, False]),
            ),
        )


def test_state_delta_snapshot_owns_effect_data_and_preserves_live_entity() -> None:
    entity = UncopyableEntity()
    semantics = ObjectSemantics(
        affordance=Affordance(custom_config={"threshold": [1.0]}),
        geometry={"size": torch.ones(3)},
        properties={"mass": torch.tensor(1.0)},
        label="snapshot-object",
        entity=entity,
    )
    held = HeldObjectState(
        semantics=semantics,
        object_to_eef=torch.eye(4),
        grasp_xpos=torch.eye(4),
    )
    delta = StateDelta(held_object_updates={"arm": held})

    snapshot = delta.snapshot()
    copied = snapshot.held_object_updates["arm"]
    assert copied is not None
    assert copied is not held
    assert copied.semantics is not semantics
    assert copied.semantics.entity is entity
    assert copied.semantics.affordance is not semantics.affordance
    assert copied.object_to_eef.data_ptr() != held.object_to_eef.data_ptr()
    assert copied.grasp_xpos.data_ptr() != held.grasp_xpos.data_ptr()

    copied.object_to_eef.fill_(7.0)
    copied.semantics.affordance.custom_config["threshold"].append(2.0)
    copied.semantics.geometry["size"].zero_()
    assert torch.equal(held.object_to_eef, torch.eye(4))
    assert semantics.affordance.custom_config["threshold"] == [1.0]
    assert torch.equal(semantics.geometry["size"], torch.ones(3))


def test_partial_effect_failure_waits_for_unresolved_rows_then_retries_failure() -> (
    None
):
    session, waiting = _effect_session(batch_size=2, max_action_retries=1)
    request = waiting.pending_effect
    assert request is not None

    partial = session.tick(
        _context(0.21, (0.2, 0.2), (0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            request.verification_id,
            success_mask=torch.tensor([False, False]),
            failure_mask=torch.tensor([True, False]),
        ),
    )

    assert partial.pending_effect is not None
    assert partial.pending_effect.env_mask.tolist() == [False, True]
    assert not any(
        event.kind is ExecutionEventKind.ACTION_RETRY for event in partial.events
    )

    unresolved_request = partial.pending_effect
    resolved = session.tick(
        _context(0.22, (0.2, 0.2), (0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            unresolved_request.verification_id,
            success_mask=torch.tensor([False, True]),
            failure_mask=torch.tensor([False, False]),
        ),
    )

    held = resolved.task_state.get_held_object("arm")
    assert held is not None and held.env_mask is not None
    assert held.env_mask.tolist() == [False, True]
    failed_event = next(
        event
        for event in resolved.events
        if event.kind is ExecutionEventKind.EFFECT_VERIFICATION_FAILED
    )
    retry_event = next(
        event
        for event in resolved.events
        if event.kind is ExecutionEventKind.ACTION_RETRY
    )
    assert failed_event.env_mask.tolist() == [True, False]
    assert retry_event.env_mask.tolist() == [True, False]

    retry_command = session.tick(_context(0.23, (0.2, 0.2), (0.2, 0.2), 0))
    assert retry_command.command is not None
    assert retry_command.command.active_mask.tolist() == [True, False]


def test_effect_failure_exhaustion_advances_completed_rows_without_empty_request() -> (
    None
):
    session, waiting = _effect_session(batch_size=2, max_action_retries=0)
    request = waiting.pending_effect
    assert request is not None

    terminal = session.tick(
        _context(0.21, (0.2, 0.2), (0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            request.verification_id,
            success_mask=torch.tensor([True, False]),
            failure_mask=torch.tensor([False, True]),
        ),
    )

    held = terminal.task_state.get_held_object("arm")
    assert terminal.status is ExecutionStatus.COMPLETED
    assert terminal.eligible_mask.tolist() == [True, False]
    assert terminal.pending_effect is None
    assert terminal.command is None
    assert held is not None and held.env_mask is not None
    assert held.env_mask.tolist() == [True, False]
    assert any(
        event.kind is ExecutionEventKind.RECOVERY_EXHAUSTED
        and event.env_mask.tolist() == [False, True]
        for event in terminal.events
    )
    assert any(
        event.kind is ExecutionEventKind.SESSION_COMPLETED for event in terminal.events
    )


def test_deactivating_last_unresolved_effect_row_advances_barrier() -> None:
    session, waiting = _effect_session(batch_size=2)
    request = waiting.pending_effect
    assert request is not None
    partial = session.tick(
        _context(0.21, (0.2, 0.2), (0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            request.verification_id,
            success_mask=torch.tensor([True, False]),
            failure_mask=torch.tensor([False, False]),
        ),
    )
    assert partial.pending_effect is not None

    session.deactivate_rows(
        torch.tensor([False, True]),
        reason="effect observation terminated",
    )
    terminal = session.tick(_context(0.22, (0.2, 0.2), (0.2, 0.2), 0))

    assert terminal.status is ExecutionStatus.COMPLETED
    assert terminal.eligible_mask.tolist() == [True, False]
    assert terminal.pending_effect is None
    assert not any(
        event.kind is ExecutionEventKind.EFFECT_VERIFICATION_REQUIRED
        for event in terminal.events
    )


def test_deactivating_all_effect_rows_is_terminal_and_clears_request() -> None:
    session, _ = _effect_session(batch_size=2)

    changed = session.deactivate_rows(
        torch.tensor([True, True]),
        reason="all environments terminated",
    )
    terminal = session.tick(_context(0.21, (0.2, 0.2), (0.2, 0.2), 0))

    assert changed.tolist() == [True, True]
    assert terminal.status is ExecutionStatus.FAILED
    assert terminal.pending_effect is None
    assert any(
        event.kind is ExecutionEventKind.SESSION_FAILED for event in terminal.events
    )
    assert not any(
        event.kind is ExecutionEventKind.EFFECT_VERIFICATION_REQUIRED
        for event in terminal.events
    )


def test_effect_request_deadline_is_stable_and_accepts_result_at_boundary() -> None:
    session, waiting = _effect_session(action_timeout=0.25)
    request = waiting.pending_effect
    assert request is not None
    assert request.requested_at == pytest.approx(0.2)
    assert request.deadline == pytest.approx(0.25)

    polled = session.tick(_context(0.24, 0.2, 0.2, 0))
    assert polled.pending_effect is not None
    assert polled.pending_effect.verification_id == request.verification_id
    assert polled.pending_effect.requested_at == request.requested_at
    assert polled.pending_effect.deadline == request.deadline

    completed = session.tick(
        _context(0.25, 0.2, 0.2, 0),
        effect_result=EffectVerificationResult(
            request.verification_id,
            success_mask=torch.tensor([True]),
            failure_mask=torch.tensor([False]),
        ),
    )
    assert completed.status is ExecutionStatus.COMPLETED


def test_session_revision_cannot_abandon_pending_effect_verification() -> None:
    engine, _ = _engine()
    engine.register(EffectAction())
    base = _invocation(engine)
    invocation = ActionInvocation(
        skill_id="effect",
        goal=base.goal,
        binding=base.binding,
        motion_policy=base.motion_policy,
        recovery_policy=base.recovery_policy,
    )
    session = engine.start((invocation,), _context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.1, 0.0, 0.2, 0))
    waiting = session.tick(_context(0.2, 0.2, 0.2, 0))
    assert waiting.pending_effect is not None

    with pytest.raises(RuntimeError, match="physical-effect resolution"):
        session.revise_current(replace(invocation, revision=1))

    assert session.effect_verification_pending is True
    completed = session.tick(
        _context(0.3, 0.2, 0.2, 0),
        effect_result=EffectVerificationResult(
            waiting.pending_effect.verification_id,
            torch.tensor([True]),
            torch.tensor([False]),
        ),
    )
    assert completed.status is ExecutionStatus.COMPLETED
    assert completed.task_state.get_held_object("arm") is not None


def test_effect_failure_does_not_commit_and_exhausts_retry_budget() -> None:
    engine, _ = _engine()
    engine.register(EffectAction())
    base = _invocation(engine, max_action_retries=0)
    invocation = ActionInvocation(
        skill_id="effect",
        goal=base.goal,
        binding=base.binding,
        motion_policy=base.motion_policy,
        recovery_policy=base.recovery_policy,
    )
    session = engine.start((invocation,), _context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.1, 0.0, 0.2, 0))

    waiting = session.tick(_context(0.2, 0.2, 0.2, 0))
    assert waiting.pending_effect is not None
    failed = session.tick(
        _context(0.3, 0.2, 0.2, 0),
        effect_result=EffectVerificationResult(
            waiting.pending_effect.verification_id,
            torch.tensor([False]),
            torch.tensor([True]),
        ),
    )

    assert failed.status is ExecutionStatus.FAILED
    assert failed.task_state.get_held_object("arm") is None
    assert any(
        event.kind is ExecutionEventKind.RECOVERY_EXHAUSTED for event in failed.events
    )


def test_pending_effect_timeout_exhausts_without_committing_late_result() -> None:
    engine, _ = _engine()
    engine.register(EffectAction())
    base = _invocation(
        engine,
        max_action_retries=0,
        action_timeout=0.25,
    )
    invocation = ActionInvocation(
        skill_id="effect",
        goal=base.goal,
        binding=base.binding,
        motion_policy=base.motion_policy,
        recovery_policy=base.recovery_policy,
    )
    session = engine.start((invocation,), _context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.1, 0.0, 0.2, 0))
    waiting = session.tick(_context(0.2, 0.2, 0.2, 0))
    assert waiting.pending_effect is not None

    timed_out = session.tick(
        _context(0.3, 0.2, 0.2, 0),
        effect_result=EffectVerificationResult(
            waiting.pending_effect.verification_id,
            torch.tensor([True]),
            torch.tensor([False]),
        ),
    )

    kinds = {event.kind for event in timed_out.events}
    assert ExecutionEventKind.EFFECT_VERIFICATION_TIMEOUT in kinds
    assert ExecutionEventKind.RECOVERY_EXHAUSTED in kinds
    assert timed_out.status is ExecutionStatus.FAILED
    assert timed_out.pending_effect is None
    assert timed_out.task_state.get_held_object("arm") is None


def test_effect_timeout_exhaustion_advances_rows_already_verified() -> None:
    session, waiting = _effect_session(
        batch_size=2,
        max_action_retries=0,
        action_timeout=0.25,
    )
    request = waiting.pending_effect
    assert request is not None
    partial = session.tick(
        _context(0.21, (0.2, 0.2), (0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            request.verification_id,
            success_mask=torch.tensor([True, False]),
            failure_mask=torch.tensor([False, False]),
        ),
    )
    assert partial.pending_effect is not None

    terminal = session.tick(_context(0.3, (0.2, 0.2), (0.2, 0.2), 0))

    held = terminal.task_state.get_held_object("arm")
    assert terminal.status is ExecutionStatus.COMPLETED
    assert terminal.eligible_mask.tolist() == [True, False]
    assert terminal.pending_effect is None
    assert held is not None and held.env_mask is not None
    assert held.env_mask.tolist() == [True, False]
    timeout_event = next(
        event
        for event in terminal.events
        if event.kind is ExecutionEventKind.EFFECT_VERIFICATION_TIMEOUT
    )
    assert timeout_event.env_mask.tolist() == [False, True]


def test_effect_timeout_charges_concurrent_planning_failures() -> None:
    session, _ = _effect_session(
        batch_size=2,
        max_action_retries=1,
        action_timeout=0.25,
        action=MixedEffectAction(),
    )

    first_retry = session.tick(_context(0.3, (0.2, 0.2), (0.2, 0.2), 0))
    retry_event = next(
        event
        for event in first_retry.events
        if event.kind is ExecutionEventKind.ACTION_RETRY
    )
    planning_event = next(
        event
        for event in first_retry.events
        if event.kind is ExecutionEventKind.ACTION_PLANNING_FAILED
    )
    assert retry_event.env_mask.tolist() == [True, True]
    assert planning_event.env_mask.tolist() == [False, True]

    session.tick(_context(0.4, (0.2, 0.2), (0.2, 0.2), 0))
    second_wait = session.tick(_context(0.5, (0.2, 0.2), (0.2, 0.2), 0))
    assert second_wait.pending_effect is not None
    terminal = session.tick(_context(0.6, (0.2, 0.2), (0.2, 0.2), 0))

    assert terminal.status is ExecutionStatus.FAILED
    assert terminal.eligible_mask.tolist() == [False, False]
    exhausted = next(
        event
        for event in terminal.events
        if event.kind is ExecutionEventKind.RECOVERY_EXHAUSTED
    )
    assert exhausted.env_mask.tolist() == [True, True]


def test_deferred_effect_failure_charges_concurrent_planning_failures() -> None:
    session, waiting = _effect_session(
        batch_size=3,
        max_action_retries=0,
        action=MixedEffectAction(),
    )
    request = waiting.pending_effect
    assert request is not None
    partial = session.tick(
        _context(0.21, (0.2, 0.2, 0.2), (0.2, 0.2, 0.2), 0),
        effect_result=EffectVerificationResult(
            request.verification_id,
            success_mask=torch.tensor([False, False, False]),
            failure_mask=torch.tensor([True, False, False]),
        ),
    )
    assert partial.pending_effect is not None
    assert partial.pending_effect.env_mask.tolist() == [False, True, False]

    session.deactivate_rows(
        torch.tensor([False, True, False]),
        reason="unresolved effect row terminated",
    )
    terminal = session.tick(_context(0.22, (0.2, 0.2, 0.2), (0.2, 0.2, 0.2), 0))

    assert terminal.status is ExecutionStatus.FAILED
    assert terminal.eligible_mask.tolist() == [False, False, False]
    planning_event = next(
        event
        for event in terminal.events
        if event.kind is ExecutionEventKind.ACTION_PLANNING_FAILED
    )
    exhausted = next(
        event
        for event in terminal.events
        if event.kind is ExecutionEventKind.RECOVERY_EXHAUSTED
    )
    assert planning_event.env_mask.tolist() == [False, False, True]
    assert exhausted.env_mask.tolist() == [True, False, True]


def test_effect_retry_invalidates_previous_verification_id() -> None:
    engine, _ = _engine()
    engine.register(EffectAction())
    base = _invocation(
        engine,
        max_action_retries=1,
        action_timeout=0.25,
    )
    invocation = ActionInvocation(
        skill_id="effect",
        goal=base.goal,
        binding=base.binding,
        motion_policy=base.motion_policy,
        recovery_policy=base.recovery_policy,
    )
    session = engine.start((invocation,), _context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.1, 0.0, 0.2, 0))
    first_wait = session.tick(_context(0.2, 0.2, 0.2, 0))
    assert first_wait.pending_effect is not None
    old_id = first_wait.pending_effect.verification_id
    old_deadline = first_wait.pending_effect.deadline
    old_generation = first_wait.pending_effect.attempt_generation

    retry = session.tick(_context(0.3, 0.2, 0.2, 0))
    assert retry.command is not None
    assert any(event.kind is ExecutionEventKind.ACTION_RETRY for event in retry.events)
    session.tick(_context(0.4, 0.2, 0.2, 0))
    second_wait = session.tick(_context(0.5, 0.2, 0.2, 0))
    assert second_wait.pending_effect is not None
    assert second_wait.pending_effect.verification_id != old_id
    assert second_wait.pending_effect.attempt_generation == old_generation + 1
    assert second_wait.pending_effect.deadline > old_deadline

    with pytest.raises(ValueError, match="verification_id"):
        session.tick(
            _context(0.55, 0.2, 0.2, 0),
            effect_result=EffectVerificationResult(
                old_id,
                torch.tensor([True]),
                torch.tensor([False]),
            ),
        )


def test_effect_request_generation_advances_after_tracking_replan() -> None:
    engine, _ = _engine()
    effect = EffectAction()
    engine.register(effect)
    base = _invocation(engine)
    invocation = ActionInvocation(
        skill_id=effect.skill_id,
        goal=base.goal,
        binding=base.binding,
        motion_policy=base.motion_policy,
        recovery_policy=base.recovery_policy,
    )
    session = engine.start((invocation,), _context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.0, 0.0, 0.2, 0))

    replanned = session.tick(_context(0.1, 1.0, 0.2, 0))
    session.tick(_context(0.2, 1.0, 0.2, 0))
    waiting = session.tick(_context(0.3, 0.2, 0.2, 0))

    assert any(event.kind is ExecutionEventKind.REPLANNED for event in replanned.events)
    assert waiting.pending_effect is not None
    assert waiting.pending_effect.attempt_generation == 1


def test_failed_effect_plan_retries_without_requesting_effect_verification() -> None:
    engine, _ = _engine()
    engine.register(FailedEffectAction())
    base = _invocation(engine, max_action_retries=0)
    invocation = ActionInvocation(
        skill_id="failed_effect",
        goal=base.goal,
        binding=base.binding,
        motion_policy=base.motion_policy,
        recovery_policy=base.recovery_policy,
    )
    session = engine.start((invocation,), _context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.0, 0.0, 0.2, 0))
    session.tick(_context(0.1, 0.0, 0.2, 0))

    failed = session.tick(_context(0.2, 0.0, 0.2, 0))

    assert failed.status is ExecutionStatus.FAILED
    assert failed.pending_effect is None
    assert not any(
        event.kind is ExecutionEventKind.EFFECT_VERIFICATION_REQUIRED
        for event in failed.events
    )
    assert any(
        event.kind is ExecutionEventKind.RECOVERY_EXHAUSTED for event in failed.events
    )
