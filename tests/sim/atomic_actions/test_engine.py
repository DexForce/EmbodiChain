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
    ActionControlOverrides,
    ActionInvocation,
    ActionOptions,
    ActionPlan,
    AtomicAction,
    AtomicActionEngine,
    ControlPartCommandProfile,
    JointPositionCommand,
    JointPositionGoal,
    MotionPolicy,
    PlanningContext,
    ResolvedActionRequest,
    register_action,
    get_registered_actions,
    unregister_action,
)


class StubAction(AtomicAction[JointPositionGoal, ActionOptions]):
    """Deterministic test action that commands every robot joint."""

    skill_id: ClassVar[str] = "stub"
    GoalType: ClassVar[type] = JointPositionGoal
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def plan(
        self,
        request: ResolvedActionRequest[JointPositionGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(request)
        assert isinstance(goal.target, torch.Tensor)
        target = goal.target.to(context.robot.qpos)
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
            request,
            context,
            success=success,
            trajectory=trajectory,
        )


class OtherStubAction(StubAction):
    """Second configured skill used to verify shared engine resources."""

    skill_id: ClassVar[str] = "other_stub"


def _engine(
    batch_size: int = 2,
    robot_dof: int = 3,
    control_profiles: dict[str, ControlPartCommandProfile] | None = None,
) -> AtomicActionEngine:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = robot_dof
    robot.control_parts = {"all": object()}
    robot.get_qpos.return_value = torch.zeros(batch_size, robot_dof)
    robot.get_qvel.return_value = torch.zeros(batch_size, robot_dof)
    robot.get_joint_ids.return_value = list(range(robot_dof))
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub_planner"
    return AtomicActionEngine(generator, control_profiles=control_profiles)


def _invocation(
    qpos: torch.Tensor,
) -> ActionInvocation[JointPositionGoal, ActionOptions]:
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
    engine.register(StubAction())
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
    engine.register(StubAction())
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
    first = StubAction()
    second = StubAction()
    engine.register(first)
    with pytest.raises(ValueError, match="already registered"):
        engine.register(second)


def test_engine_binds_one_planning_service_to_every_action() -> None:
    engine = _engine()
    first = StubAction()
    second = OtherStubAction()

    engine.register(first)
    engine.register(second)

    assert first.motion_generator is engine.motion_generator
    assert second.motion_generator is engine.motion_generator
    assert first.builder is engine.planning_services.trajectory_builder
    assert second.builder is first.builder


def test_engine_resolves_action_binding_from_robot_control_parts() -> None:
    engine = _engine(robot_dof=3)

    resolved = engine.planning_services.resolve_binding(
        ActionBinding(manipulators={"primary": "all"})
    )

    assert resolved.manipulator().name == "all"
    assert resolved.manipulator().joint_ids == (0, 1, 2)
    assert resolved.manipulator().dof == 3


def test_engine_resolves_invocation_control_override_into_request() -> None:
    engine = _engine(
        robot_dof=3,
        control_profiles={
            "all": ControlPartCommandProfile.joint_positions(ready=torch.zeros(3))
        },
    )
    engine.register(StubAction())
    invocation = replace(
        _invocation(torch.ones(2, 3)),
        control_overrides=ActionControlOverrides(
            manipulators={
                "primary": {"ready": JointPositionCommand(torch.full((3,), 0.4))}
            }
        ),
        revision=2,
    )

    request = engine.resolve(invocation)

    assert request.revision == 2
    assert torch.allclose(
        request.binding.manipulator().joint_positions("ready", n_envs=2, device="cpu"),
        torch.full((2, 3), 0.4),
    )


def test_engine_rejects_binding_outside_robot_control_parts() -> None:
    engine = _engine()
    engine.register(StubAction())
    invocation = ActionInvocation(
        skill_id="stub",
        goal=JointPositionGoal(torch.zeros(2, 3)),
        binding=ActionBinding(manipulators={"primary": "missing_arm"}),
        motion_policy=MotionPolicy(sample_count=2),
    )

    with pytest.raises(ValueError, match="Robot.control_parts"):
        engine.plan(invocation)


def test_engine_motion_generator_is_read_only() -> None:
    engine = _engine()

    with pytest.raises(AttributeError):
        engine.motion_generator = Mock()  # type: ignore[misc]


def test_engine_plan_action_supports_unregistered_configured_instance() -> None:
    engine = _engine()
    action = StubAction()

    plan = engine.plan_action(
        action,
        _invocation(torch.ones(2, 3)),
        engine.initial_context(),
    )

    assert plan.plan_success.tolist() == [True, True]
    assert action.is_bound
    assert engine.actions == {}


def test_action_cannot_be_rebound_to_another_engine() -> None:
    action = StubAction()
    _engine().register(action)

    with pytest.raises(ValueError, match="another AtomicActionEngine"):
        _engine().register(action)


def test_unbound_action_rejects_direct_planning() -> None:
    action = StubAction()

    with pytest.raises(RuntimeError, match="not bound"):
        action.resolve_request(_invocation(torch.ones(2, 3)))


def test_engine_rejects_plan_for_a_different_skill() -> None:
    engine = _engine()
    action = StubAction()
    original_plan = action.plan

    def wrong_skill_plan(
        request: ResolvedActionRequest[JointPositionGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        return replace(original_plan(request, context), skill_id="other")

    action.plan = wrong_skill_plan  # type: ignore[method-assign]
    engine.register(action)

    with pytest.raises(ValueError, match="must match its request"):
        engine.compile((_invocation(torch.zeros(2, 3)),))
