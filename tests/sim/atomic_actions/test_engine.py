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

"""Tests for the atomic-action registry and static compiler."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionCfg,
    ActionInvocation,
    ActionPlan,
    AtomicAction,
    AtomicActionEngine,
    JointPositionGoal,
    MotionPolicy,
    PlanningContext,
    register_action,
    get_registered_actions,
    unregister_action,
)


class StubAction(AtomicAction[JointPositionGoal]):
    """Deterministic test action that commands every robot joint."""

    skill_id: ClassVar[str] = "stub"
    GoalType: ClassVar[type] = JointPositionGoal
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def plan(
        self,
        invocation: ActionInvocation[JointPositionGoal],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(invocation)
        target = goal.qpos.to(context.robot.qpos)
        if target.dim() == 1:
            target = target.unsqueeze(0).expand(context.batch_size, -1)
        success = torch.ones(
            context.batch_size, dtype=torch.bool, device=context.robot.qpos.device
        )
        if torch.isnan(target).any(dim=1).any():
            success &= ~torch.isnan(target).any(dim=1)
            target = torch.nan_to_num(target)
        trajectory = torch.stack([context.robot.qpos, target], dim=1)
        return self.build_plan(
            invocation,
            context,
            success=success,
            trajectory=trajectory,
        )


def _engine(batch_size: int = 2, robot_dof: int = 3) -> AtomicActionEngine:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = robot_dof
    robot.get_qpos.return_value = torch.zeros(batch_size, robot_dof)
    robot.get_qvel.return_value = torch.zeros(batch_size, robot_dof)
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub_planner"
    return AtomicActionEngine(generator)


def _invocation(qpos: torch.Tensor) -> ActionInvocation[JointPositionGoal]:
    return ActionInvocation(
        skill_id="stub",
        goal=JointPositionGoal(qpos),
        binding=ActionBinding(manipulators={"primary": "all"}),
        motion_policy=MotionPolicy(sample_count=2),
    )


def test_global_registry_uses_stable_skill_id() -> None:
    unregister_action("stub")
    register_action(StubAction)
    try:
        assert get_registered_actions()["stub"] is StubAction
        register_action(StubAction)
    finally:
        unregister_action("stub")


def test_engine_compile_projects_terminal_state_between_actions() -> None:
    engine = _engine()
    engine.register(StubAction(engine.motion_generator, ActionCfg(name="stub")))
    first = torch.ones(2, 3)
    second = torch.full((2, 3), 2.0)

    compiled = engine.compile((_invocation(first), _invocation(second)))

    assert compiled.plan_success.tolist() == [True, True]
    assert compiled.trajectory.positions.shape == (2, 4, 3)
    assert torch.equal(compiled.action_plans[1].trajectory.positions[:, 0], first)
    assert torch.equal(compiled.projected_context.robot.qpos, second)
    assert torch.count_nonzero(engine.robot.get_qpos()) == 0


def test_engine_compile_holds_failed_rows_for_remaining_actions() -> None:
    engine = _engine()
    engine.register(StubAction(engine.motion_generator, ActionCfg(name="stub")))
    first = torch.tensor([[1.0, 1.0, 1.0], [float("nan"), 2.0, 2.0]])
    second = torch.full((2, 3), 4.0)

    compiled = engine.compile((_invocation(first), _invocation(second)))

    assert compiled.plan_success.tolist() == [True, False]
    assert torch.all(compiled.projected_context.robot.qpos[0] == 4.0)
    assert torch.all(compiled.projected_context.robot.qpos[1] == 0.0)
    assert torch.all(compiled.trajectory.positions[1] == 0.0)


def test_engine_compile_empty_sequence_is_successful_noop() -> None:
    engine = _engine()
    context = engine.initial_context()

    compiled = engine.compile((), context)

    assert compiled.plan_success.tolist() == [True, True]
    assert compiled.trajectory.positions.shape == (2, 0, 3)
    assert compiled.projected_context is context


def test_engine_rejects_unknown_skill() -> None:
    engine = _engine()
    with pytest.raises(KeyError, match="stub"):
        engine.compile((_invocation(torch.zeros(2, 3)),))


def test_engine_rejects_duplicate_instance_registration() -> None:
    engine = _engine()
    first = StubAction(engine.motion_generator, ActionCfg(name="first"))
    second = StubAction(engine.motion_generator, ActionCfg(name="second"))
    engine.register(first)
    with pytest.raises(ValueError, match="already registered"):
        engine.register(second)


def test_engine_rejects_plan_for_a_different_skill() -> None:
    engine = _engine()
    action = StubAction(engine.motion_generator, ActionCfg(name="stub"))
    original_plan = action.plan

    def wrong_skill_plan(
        invocation: ActionInvocation[JointPositionGoal],
        context: PlanningContext,
    ) -> ActionPlan:
        return replace(original_plan(invocation, context), skill_id="other")

    action.plan = wrong_skill_plan  # type: ignore[method-assign]
    engine.register(action)

    with pytest.raises(ValueError, match="must match its invocation"):
        engine.compile((_invocation(torch.zeros(2, 3)),))
