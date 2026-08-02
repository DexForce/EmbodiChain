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

from typing import ClassVar
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionCfg,
    ActionInvocation,
    ActionPlan,
    Affordance,
    AtomicAction,
    AtomicActionEngine,
    EndEffectorPoseGoal,
    EntityState,
    ExecutionEventKind,
    ExecutionStatus,
    HeldObjectState,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    RecoveryPolicy,
    RobotObservation,
    SceneEntityPose,
    SceneSnapshot,
    StateDelta,
    TaskState,
)
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.planners import PlanOptions


class DynamicAction(AtomicAction[EndEffectorPoseGoal]):
    """Test action whose terminal joint command follows a scene entity x pose."""

    skill_id: ClassVar[str] = "dynamic"
    GoalType: ClassVar[type] = EndEffectorPoseGoal
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(self, motion_generator) -> None:
        super().__init__(motion_generator, ActionCfg(name="dynamic"))
        self.plan_count = 0

    def _plan(
        self,
        invocation: ActionInvocation[EndEffectorPoseGoal],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(invocation)
        self.plan_count += 1
        pose = resolve_pose_goal(goal.xpos, context, name="xpos")
        target = pose[:, 0, 3].unsqueeze(1).expand_as(context.robot.qpos)
        return self.build_plan(
            invocation,
            context,
            success=True,
            trajectory=torch.stack([context.robot.qpos, target], dim=1),
        )


class EffectAction(DynamicAction):
    """Dynamic test action that declares an attachment effect."""

    skill_id: ClassVar[str] = "effect"

    def _plan(
        self,
        invocation: ActionInvocation[EndEffectorPoseGoal],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(invocation)
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
            invocation,
            context,
            success=True,
            trajectory=torch.stack([context.robot.qpos, target], dim=1),
            expected_effects=StateDelta(held_object_updates={"arm": held}),
        )


def _engine(batch_size: int = 1) -> tuple[AtomicActionEngine, DynamicAction]:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 2
    robot.get_qpos.return_value = torch.zeros(batch_size, 2)
    robot.get_qvel.return_value = torch.zeros(batch_size, 2)
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub"
    engine = AtomicActionEngine(generator)
    action = DynamicAction(generator)
    engine.register(action)
    return engine, action


def _context(
    timestamp: float, qpos: float, entity_x: float, version: int
) -> PlanningContext:
    positions = torch.full((1, 2), qpos)
    pose = torch.eye(4).unsqueeze(0)
    pose[:, 0, 3] = entity_x
    return PlanningContext(
        robot=RobotObservation(
            timestamp=timestamp,
            qpos=positions,
            qvel=torch.zeros_like(positions),
        ),
        task=TaskState.empty(batch_size=1, device="cpu"),
        scene=SceneSnapshot(
            timestamp=timestamp,
            version=version,
            entities={"target": EntityState(pose)},
        ),
        env_ids=torch.tensor([0], dtype=torch.long),
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
    max_replans: int = 2,
    max_phase_retries: int = 2,
    phase_timeout: float = 30.0,
    motion_source: str = "ik_interp",
) -> ActionInvocation[EndEffectorPoseGoal]:
    return ActionInvocation(
        skill_id="dynamic",
        goal=EndEffectorPoseGoal(SceneEntityPose("target")),
        binding=ActionBinding(manipulators={"primary": "arm"}),
        motion_policy=MotionPolicy(sample_count=2, motion_source=motion_source),
        recovery_policy=RecoveryPolicy(
            max_replans=max_replans,
            max_phase_retries=max_phase_retries,
            tracking_error_threshold=0.05,
            goal_translation_threshold=0.02,
            phase_timeout=phase_timeout,
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


def test_scene_motion_replans_late_bound_goal() -> None:
    engine, action = _engine()
    session = engine.start((_invocation(),), _context(0.0, 0.0, 0.1, 0))
    session.tick(_context(0.0, 0.0, 0.1, 0))

    tick = session.tick(_context(0.1, 0.0, 0.3, 1))

    kinds = {event.kind for event in tick.events}
    assert ExecutionEventKind.DYNAMIC_GOAL_CHANGED in kinds
    assert ExecutionEventKind.REPLANNED in kinds
    assert action.plan_count == 2
    assert tick.command is not None


def test_collision_world_change_replans_with_latest_obstacle_pose() -> None:
    engine, action = _engine()
    planner = engine.motion_generator.planner
    planner.supports_collision_world_updates = True
    planner.default_plan_options.return_value = PlanOptions()
    planner.with_collision_world.side_effect = (
        lambda options, *, obstacle_poses: options
    )
    initial_qpos = torch.zeros(1, 2)
    initial = _collision_context(
        0.0,
        initial_qpos,
        torch.tensor([0.4]),
        (0,),
    )
    session = engine.start(
        (_invocation(motion_source="motion_gen"),),
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
    latest_obstacles = planner.with_collision_world.call_args.kwargs["obstacle_poses"]
    assert latest_obstacles["obstacle"][0, 0, 3] == pytest.approx(0.6)
    assert tick.command is not None


def test_collision_world_exhaustion_only_disables_changed_environment() -> None:
    engine, _ = _engine(batch_size=2)
    planner = engine.motion_generator.planner
    planner.supports_collision_world_updates = True
    planner.default_plan_options.return_value = PlanOptions()
    planner.with_collision_world.side_effect = (
        lambda options, *, obstacle_poses: options
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
                motion_source="motion_gen",
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


def test_phase_timeout_retry_budget_is_bounded() -> None:
    engine, action = _engine()
    session = engine.start(
        (
            _invocation(
                max_phase_retries=1,
                phase_timeout=0.05,
            ),
        ),
        _context(0.0, 0.0, 0.2, 0),
    )
    session.tick(_context(0.0, 0.0, 0.2, 0))

    retry = session.tick(_context(0.1, 0.0, 0.2, 0))
    exhausted = session.tick(_context(0.2, 0.0, 0.2, 0))

    retry_kinds = {event.kind for event in retry.events}
    assert ExecutionEventKind.PHASE_TIMEOUT in retry_kinds
    assert ExecutionEventKind.ACTION_RETRY in retry_kinds
    assert ExecutionEventKind.REPLANNED in retry_kinds
    assert action.plan_count == 2
    exhausted_kinds = {event.kind for event in exhausted.events}
    assert ExecutionEventKind.PHASE_TIMEOUT in exhausted_kinds
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
        (_invocation(motion_source="motion_gen"),),
        initial,
    )
    regressed = _collision_context(0.1, qpos, torch.tensor([0.4]), (1,))

    with pytest.raises(ValueError, match="Collision-world revisions"):
        session.tick(regressed)


def test_nonempty_effect_is_committed_only_after_external_verification() -> None:
    engine, _ = _engine()
    effect = EffectAction(engine.motion_generator)
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
    completed = session.tick(
        _context(0.3, 0.2, 0.2, 0),
        effect_success=torch.tensor([True]),
    )

    assert waiting.status is ExecutionStatus.RUNNING
    assert waiting.task_state.get_held_object("arm") is None
    assert any(
        event.kind is ExecutionEventKind.EFFECT_VERIFICATION_REQUIRED
        for event in waiting.events
    )
    assert completed.status is ExecutionStatus.COMPLETED
    assert completed.task_state.get_held_object("arm") is not None


def test_effect_failure_does_not_commit_and_exhausts_retry_budget() -> None:
    engine, _ = _engine()
    engine.register(EffectAction(engine.motion_generator))
    base = _invocation(max_phase_retries=0)
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
