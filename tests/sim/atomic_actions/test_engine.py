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
    BUILTIN_ACTION_TYPES,
    ControlPartCommandProfile,
    JointPositionCommand,
    JointPositionGoal,
    JointPositionTarget,
    JOINT_POSITION_CAPABILITY,
    MotionPolicy,
    PlanningContext,
    PressGoal,
    PressOptions,
    ResolvedActionRequest,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    TimedTrajectory,
)

ACTION_DT = 0.02


class StubAction(AtomicAction[JointPositionGoal, ActionOptions]):
    """Deterministic test action that commands every robot joint."""

    skill_id: ClassVar[str] = "stub"
    GoalType: ClassVar[type] = JointPositionGoal
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
        ),
    )

    def _plan(
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
        trajectory = TimedTrajectory.from_uniform_step(
            torch.stack([context.robot.qpos, target], dim=1),
            env_ids=context.env_ids,
            step_dt=ACTION_DT,
        )
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=trajectory,
        )


class OtherStubAction(StubAction):
    """Second configured skill used to verify shared engine resources."""

    skill_id: ClassVar[str] = "other_stub"


def _motion_generator(
    batch_size: int = 2,
    robot_dof: int = 3,
) -> Mock:
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
    return generator


def _engine(
    batch_size: int = 2,
    robot_dof: int = 3,
    control_profiles: dict[str, ControlPartCommandProfile] | None = None,
    *,
    load_builtins: bool = False,
) -> AtomicActionEngine:
    return AtomicActionEngine(
        _motion_generator(batch_size, robot_dof),
        control_profiles=control_profiles,
        load_builtins=load_builtins,
    )


def _invocation(
    engine: AtomicActionEngine,
    qpos: torch.Tensor,
) -> ActionInvocation[JointPositionGoal, ActionOptions]:
    return ActionInvocation(
        skill_id="stub",
        goal=JointPositionGoal(qpos),
        binding=engine.bind_control_parts(
            "stub",
            {"primary": {"motion": "all"}},
        ),
        motion_policy=MotionPolicy(sample_count=2),
    )


def test_action_subclass_cannot_override_framework_plan() -> None:
    with pytest.raises(TypeError, match="must implement _plan"):

        class InvalidAction(AtomicAction[JointPositionGoal, ActionOptions]):
            skill_id: ClassVar[str] = "invalid"
            GoalType: ClassVar[type] = JointPositionGoal

            def plan(
                self,
                request: ResolvedActionRequest[JointPositionGoal, ActionOptions],
                context: PlanningContext,
            ) -> ActionPlan:
                raise NotImplementedError

            def _plan(
                self,
                request: ResolvedActionRequest[JointPositionGoal, ActionOptions],
                context: PlanningContext,
            ) -> ActionPlan:
                raise NotImplementedError


def test_engine_loads_fresh_builtin_instances_by_default() -> None:
    first = AtomicActionEngine(_motion_generator())
    second = AtomicActionEngine(_motion_generator())
    expected_ids = tuple(action_type.skill_id for action_type in BUILTIN_ACTION_TYPES)

    assert tuple(first.actions) == expected_ids
    assert all(action.is_bound for action in first.actions.values())
    assert all(
        first.actions[skill_id] is not second.actions[skill_id]
        for skill_id in expected_ids
    )


def test_engine_can_disable_builtin_loading() -> None:
    assert _engine(load_builtins=False).actions == {}


def test_auto_registered_builtin_accepts_per_invocation_options() -> None:
    generator = _motion_generator(robot_dof=3)
    generator.robot.control_parts = {"arm": object(), "hand": object()}
    generator.robot.get_joint_ids.side_effect = lambda name: (
        [0, 1] if name == "arm" else [2]
    )
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(grasp=torch.ones(1))
        },
    )
    options = PressOptions(hand_interp_steps=7)
    invocation = ActionInvocation(
        skill_id="press",
        goal=PressGoal(torch.eye(4)),
        binding=engine.bind_control_parts(
            "press",
            {"primary": {"motion": "arm", "grasp": "hand"}},
        ),
        motion_policy=MotionPolicy(sample_count=20),
        skill_options=options,
    )

    request = engine.actions["press"].resolve_request(invocation)

    assert request.skill_options.hand_interp_steps == 7
    assert request.skill_options is not options


def test_engine_compile_projects_terminal_state_between_actions() -> None:
    engine = _engine()
    engine.register(StubAction())
    first = torch.ones(2, 3)
    second = torch.full((2, 3), 2.0)

    compiled = engine.compile((_invocation(engine, first), _invocation(engine, second)))

    assert compiled.plan_success.tolist() == [True, True]
    assert compiled.trajectory.positions.shape == (2, 4, 3)
    second_trajectory = compiled.action_plans[1].joint_trajectory
    assert second_trajectory is not None
    assert torch.equal(second_trajectory.positions[:, 0], first)
    assert torch.equal(compiled.projected_context.robot.qpos, second)
    assert torch.count_nonzero(engine.robot.get_qpos()) == 0
    assert compiled.action_waypoint_offset(1) == 2
    assert compiled.segment(1, "stub").start == 2
    assert compiled.segment(1, "stub").stop == 4


def test_engine_compile_holds_failed_rows_for_remaining_actions() -> None:
    engine = _engine()
    engine.register(StubAction())
    first = torch.tensor([[1.0, 1.0, 1.0], [float("nan"), 2.0, 2.0]])
    second = torch.full((2, 3), 4.0)

    compiled = engine.compile((_invocation(engine, first), _invocation(engine, second)))

    assert compiled.plan_success.tolist() == [True, False]
    assert torch.all(compiled.projected_context.robot.qpos[0] == 4.0)
    assert torch.all(compiled.projected_context.robot.qpos[1] == 0.0)
    first_trajectory = compiled.action_plans[0].joint_trajectory
    assert first_trajectory is not None
    assert torch.all(first_trajectory.positions[1] == 0.0)
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
    invocation = ActionInvocation(
        skill_id="stub",
        goal=JointPositionGoal(torch.zeros(2, 3)),
        binding=ActionBinding(owner_id=engine.binding_owner_id),
        motion_policy=MotionPolicy(sample_count=2),
    )
    with pytest.raises(KeyError, match="stub"):
        engine.compile((invocation,))


def test_engine_rejects_duplicate_instance_registration() -> None:
    engine = _engine()
    first = StubAction()
    second = StubAction()
    engine.register(first)
    with pytest.raises(ValueError, match="already registered"):
        engine.register(second)


def test_engine_replaces_registered_action_only_when_explicit() -> None:
    engine = _engine()
    first = StubAction()
    replacement = StubAction()
    engine.register(first)

    engine.register(replacement, replace=True)

    assert engine.actions["stub"] is replacement
    assert replacement.is_bound


def test_engine_binds_one_planning_service_to_every_action() -> None:
    engine = _engine()
    first = StubAction()
    second = OtherStubAction()

    engine.register(first)
    engine.register(second)

    assert first.motion_generator is engine.motion_generator
    assert second.motion_generator is engine.motion_generator
    assert first.planning_services is engine.planning_services
    assert second.planning_services is engine.planning_services


def test_engine_preserves_custom_action_timing() -> None:
    engine = _engine()
    engine.register(StubAction())

    plan = engine.plan(_invocation(engine, torch.ones(2, 3)))

    assert plan.joint_trajectory is not None
    assert torch.allclose(
        plan.joint_trajectory.dt,
        torch.tensor([[0.0, ACTION_DT], [0.0, ACTION_DT]]),
    )


def test_engine_resolves_action_binding_from_robot_control_parts() -> None:
    engine = _engine(robot_dof=3)
    engine.register(StubAction())

    resolved = engine.bind_control_parts(
        "stub",
        {"primary": {"motion": "all"}},
    )
    target = resolved.endpoint("primary", "motion").require_target(JointPositionTarget)

    assert target.control_part == "all"
    assert target.joint_ids == (0, 1, 2)


def test_engine_resolves_invocation_control_override_into_request() -> None:
    engine = _engine(
        robot_dof=3,
        control_profiles={
            "all": ControlPartCommandProfile.joint_positions(ready=torch.zeros(3))
        },
    )
    engine.register(StubAction())
    invocation = replace(
        _invocation(engine, torch.ones(2, 3)),
        control_overrides=ActionControlOverrides(
            endpoints={
                "primary": {
                    "motion": {"ready": JointPositionCommand(torch.full((3,), 0.4))}
                }
            }
        ),
        revision=2,
    )

    request = engine.actions["stub"].resolve_request(invocation)

    assert request.revision == 2
    assert torch.allclose(
        request.binding.endpoint("primary", "motion").joint_positions(
            "ready", num_envs=2, device="cpu"
        ),
        torch.full((2, 3), 0.4),
    )


def test_engine_rejects_binding_outside_robot_control_parts() -> None:
    engine = _engine()
    engine.register(StubAction())

    with pytest.raises(ValueError, match="Robot.control_parts"):
        engine.bind_control_parts(
            "stub",
            {"primary": {"motion": "missing_arm"}},
        )


def test_engine_motion_generator_is_read_only() -> None:
    engine = _engine()

    with pytest.raises(AttributeError):
        engine.motion_generator = Mock()  # type: ignore[misc]


def test_engine_plan_action_supports_unregistered_configured_instance() -> None:
    engine = _engine()
    action = StubAction()
    binding = engine.bind_control_parts(
        action,
        {"primary": {"motion": "all"}},
    )
    invocation = ActionInvocation(
        skill_id="stub",
        goal=JointPositionGoal(torch.ones(2, 3)),
        binding=binding,
        motion_policy=MotionPolicy(sample_count=2),
    )

    plan = engine.plan_action(
        action,
        invocation,
        engine.initial_context(),
    )

    assert plan.plan_success.tolist() == [True, True]
    assert action.is_bound
    assert engine.actions == {}


def test_engine_cannot_build_binding_for_action_owned_by_another_engine() -> None:
    action = StubAction()
    first = _engine()
    first.register(action)

    with pytest.raises(ValueError, match="belongs to another engine"):
        _engine().bind_control_parts(
            action,
            {"primary": {"motion": "all"}},
        )


def test_action_cannot_be_rebound_to_another_engine() -> None:
    action = StubAction()
    _engine().register(action)

    with pytest.raises(ValueError, match="another AtomicActionEngine"):
        _engine().register(action)


def test_bound_action_exposes_num_envs_property() -> None:
    engine = _engine(batch_size=3)
    action = StubAction()
    engine.register(action)

    assert action.num_envs == 3


def test_unbound_action_rejects_direct_planning() -> None:
    action = StubAction()
    donor_engine = _engine()
    donor_engine.register(StubAction())

    with pytest.raises(RuntimeError, match="not bound"):
        action.resolve_request(_invocation(donor_engine, torch.ones(2, 3)))


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
        engine.compile((_invocation(engine, torch.zeros(2, 3)),))
