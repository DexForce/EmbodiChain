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
    ActionInvocation,
    ActionOptions,
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
    ResolvedActionRequest,
    RobotObservation,
    SceneEntityPose,
    SceneSnapshot,
    StateDelta,
    TaskState,
)
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal


class DynamicAction(AtomicAction[EndEffectorPoseGoal, ActionOptions]):
    """Test action whose terminal joint command follows a scene entity x pose."""

    skill_id: ClassVar[str] = "dynamic"
    GoalType: ClassVar[type] = EndEffectorPoseGoal
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(self) -> None:
        super().__init__()
        self.plan_count = 0
        self.requests: list[ResolvedActionRequest] = []

    def plan(
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

    def plan(
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


def _engine() -> tuple[AtomicActionEngine, DynamicAction]:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 2
    robot.control_parts = {"arm": object()}
    robot.get_qpos.return_value = torch.zeros(1, 2)
    robot.get_qvel.return_value = torch.zeros(1, 2)
    robot.get_joint_ids.return_value = [0, 1]
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub"
    engine = AtomicActionEngine(generator, load_builtins=False)
    action = DynamicAction()
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


def _invocation(
    *,
    max_replans: int = 2,
    max_phase_retries: int = 2,
    phase_timeout: float = 30.0,
) -> ActionInvocation[EndEffectorPoseGoal]:
    return ActionInvocation(
        skill_id="dynamic",
        goal=EndEffectorPoseGoal(SceneEntityPose("target")),
        binding=ActionBinding(manipulators={"primary": "arm"}),
        motion_policy=MotionPolicy(sample_count=2),
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
    assert action.requests[0] is action.requests[1]
    assert tick.command is not None


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
    engine.register(EffectAction())
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
