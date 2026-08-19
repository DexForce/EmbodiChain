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
    EntityState,
    ExecutionEventKind,
    ExecutionStatus,
    GraspGoal,
    HeldObjectState,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionBinding,
    ResolvedActionRequest,
    RobotObservation,
    SceneEntityPose,
    SceneSnapshot,
    StateDelta,
    TaskState,
    TimedTrajectory,
)
from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.planners import PlanOptions


class DynamicAction(AtomicAction[EndEffectorPoseGoal, ActionOptions]):
    """Test action whose terminal joint command follows a scene entity x pose."""

    skill_id: ClassVar[str] = "dynamic"
    GoalType: ClassVar[type] = EndEffectorPoseGoal
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

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
        trajectory = TimedTrajectory.from_uniform_step(
            torch.stack([context.robot.qpos, target], dim=1),
            env_ids=context.env_ids,
            step_dt=0.1,
        )
        return self.build_plan(
            request,
            context,
            success=True,
            trajectory=trajectory,
        )


class EffectAction(DynamicAction):
    """Dynamic test action that declares an attachment effect."""

    skill_id: ClassVar[str] = "effect"

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
        trajectory = TimedTrajectory.from_uniform_step(
            torch.stack([context.robot.qpos, target], dim=1),
            env_ids=context.env_ids,
            step_dt=0.1,
        )
        return self.build_plan(
            request,
            context,
            success=True,
            trajectory=trajectory,
            expected_effects=StateDelta(held_object_updates={"arm": held}),
        )


class FailedEffectAction(EffectAction):
    """Effect-declaring action whose planner fails for every environment."""

    skill_id: ClassVar[str] = "failed_effect"

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        plan = super()._plan(request, context)
        return replace(plan, plan_success=torch.zeros_like(plan.plan_success))


class NonuniformTimingAction(DynamicAction):
    """Test action with explicit nonuniform waypoint arrival intervals."""

    skill_id: ClassVar[str] = "nonuniform_timing"

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
            dt=dt,
        )
        return self.build_plan(
            request,
            context,
            success=True,
            trajectory=trajectory,
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
    *,
    skill_id: str = "dynamic",
    max_replans: int = 2,
    max_action_retries: int = 2,
    action_timeout: float = 30.0,
    strategy: str = "ik_interp",
    dynamic_collision_mode: DynamicCollisionMode = DynamicCollisionMode.AUTO,
) -> ActionInvocation[EndEffectorPoseGoal]:
    return ActionInvocation(
        skill_id=skill_id,
        goal=EndEffectorPoseGoal(SceneEntityPose("target")),
        binding=ActionBinding(manipulators={"primary": "arm"}),
        motion_policy=MotionPolicy(
            sample_count=2,
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


def test_session_completes_incremental_command_sequence() -> None:
    engine, _ = _engine()
    session = engine.start((_invocation(),), _context(0.0, 0.0, 0.2, 0))

    first = session.tick(_context(0.0, 0.0, 0.2, 0))
    second = session.tick(_context(0.1, 0.0, 0.2, 0))
    final = session.tick(_context(0.2, 0.2, 0.2, 0))

    assert first.command is not None and torch.all(first.command.positions == 0.0)
    assert all(event.invocation_id == "dynamic-call" for event in first.events)
    assert second.command is not None and torch.all(second.command.positions == 0.2)
    assert final.status is ExecutionStatus.COMPLETED
    assert final.eligible_mask.tolist() == [True]


def test_session_commands_schedule_arrivals_and_final_settling() -> None:
    engine, _ = _engine()
    engine.register(NonuniformTimingAction())
    session = engine.start(
        (_invocation(skill_id="nonuniform_timing"),),
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
        binding=ResolvedActionBinding(),
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
    session = engine.start((_invocation(),), _context(0.0, 0.0, 0.1, 0))
    session.tick(_context(0.0, 0.0, 0.1, 0))

    tick = session.tick(_context(0.1, 0.0, 0.3, 1))

    kinds = {event.kind for event in tick.events}
    assert ExecutionEventKind.DYNAMIC_GOAL_CHANGED in kinds
    assert ExecutionEventKind.REPLANNED in kinds
    assert action.plan_count == 2
    assert action.requests[0] is action.requests[1]
    assert tick.command is not None


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
        (_invocation(strategy="motion_gen"),),
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
    base = _invocation()
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
        (_invocation(),),
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
    assert replanned.command is not None
    assert torch.all(replanned.command.positions == 0.0)
    assert next_command.command is not None
    assert torch.equal(next_command.command.positions[:, 0], torch.tensor([0.4, 0.2]))
    assert action.plan_count == 2


def test_replan_exhaustion_disables_only_triggering_row() -> None:
    engine, _ = _engine(batch_size=2)
    session = engine.start(
        (_invocation(max_replans=1),),
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


def test_session_revision_replans_from_latest_context() -> None:
    engine, action = _engine()
    original = _invocation()
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
    assert second.command is not None
    assert torch.all(second.command.positions == 0.8)


def test_session_revision_must_advance_same_invocation() -> None:
    engine, _ = _engine()
    original = _invocation()
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


def test_tracking_error_fails_when_replan_budget_is_zero() -> None:
    engine, _ = _engine()
    session = engine.start(
        (_invocation(max_replans=0),),
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
    session = engine.start((_invocation(),), initial)
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
    session = engine.start((_invocation(),), _context(1.0, 0.0, 0.2, 2))

    with pytest.raises(ValueError, match="versions must be monotonic"):
        session.tick(_context(1.0, 0.0, 0.2, 1))


def test_session_rejects_regressing_collision_world_revision() -> None:
    engine, _ = _engine()
    qpos = torch.zeros(1, 2)
    initial = _collision_context(0.0, qpos, torch.tensor([0.4]), (2,))
    session = engine.start(
        (_invocation(strategy="motion_gen"),),
        initial,
    )
    regressed = _collision_context(0.1, qpos, torch.tensor([0.4]), (1,))

    with pytest.raises(ValueError, match="Collision-world revisions"):
        session.tick(regressed)


def test_nonempty_effect_is_committed_only_after_external_verification() -> None:
    engine, _ = _engine()
    effect = EffectAction()
    engine.register(effect)
    invocation = _invocation()
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
        effect_success=torch.tensor([True]),
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


def test_effect_failure_does_not_commit_and_exhausts_retry_budget() -> None:
    engine, _ = _engine()
    engine.register(EffectAction())
    base = _invocation(max_action_retries=0)
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

    failed = session.tick(
        _context(0.2, 0.2, 0.2, 0),
        effect_success=torch.tensor([False]),
    )

    assert failed.status is ExecutionStatus.FAILED
    assert failed.task_state.get_held_object("arm") is None
    assert any(
        event.kind is ExecutionEventKind.RECOVERY_EXHAUSTED for event in failed.events
    )


def test_failed_effect_plan_retries_without_requesting_effect_verification() -> None:
    engine, _ = _engine()
    engine.register(FailedEffectAction())
    base = _invocation(max_action_retries=0)
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
