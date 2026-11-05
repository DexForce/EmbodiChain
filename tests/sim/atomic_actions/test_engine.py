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
    ActionPlanTemplateAdapter,
    ActionControlOverrides,
    ActionInvocation,
    ActionOptions,
    ActionPlan,
    ArticulationJointState,
    AffordanceSamplingContext,
    AtomicAction,
    AtomicActionEngine,
    BUILTIN_ACTION_TYPES,
    ControlPartCommandProfile,
    EntityState,
    JointPositionCommand,
    JointPositionGoal,
    JointPositionTarget,
    JOINT_POSITION_CAPABILITY,
    MotionPolicy,
    ObjectSemantics,
    PlannerDiagnostics,
    PlanQualification,
    PlanningRecovery,
    PlanningRecoveryOutcome,
    RecoveryPolicy,
    PlanningBudget,
    PlanningContext,
    PressAffordance,
    PressGoal,
    PressOptions,
    ResolvedActionRequest,
    SceneSnapshot,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    StateDelta,
    TimedTrajectory,
    TrackingPolicy,
)
from embodichain.lab.sim.motion.expansion import (
    SceneCase,
    SourceAdapter,
    SourceContext,
    TrajectoryVariant,
    apply_trajectory_variant,
    default_variant_factors,
)

ACTION_DT = 0.02


class StubAction(AtomicAction[JointPositionGoal, ActionOptions]):
    """Deterministic test action that commands every robot joint."""

    skill_id: ClassVar[str] = "stub"
    sample_count: ClassVar[int] = 2
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
        fractions = torch.linspace(
            0.0,
            1.0,
            self.sample_count,
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )
        trajectory = TimedTrajectory.from_uniform_step(
            torch.lerp(
                context.robot.qpos[:, None, :],
                target[:, None, :],
                fractions[None, :, None],
            ),
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


class ThreeFrameStubAction(StubAction):
    """Stub action with an interior free-phase sample for variant tests."""

    sample_count: ClassVar[int] = 3
    binding_contract: ClassVar[SkillBindingContract] = StubAction.binding_contract


class RetimedStubAction(StubAction):
    """Long deterministic motion using the built-in move-joints phase name."""

    skill_id: ClassVar[str] = "move_joints"
    sample_count: ClassVar[int] = 20
    binding_contract: ClassVar[SkillBindingContract] = StubAction.binding_contract


class StubSceneProvider:
    """Record scene captures and return one configured snapshot."""

    def __init__(self, scene: SceneSnapshot) -> None:
        self.scene = scene
        self.calls: list[tuple[float, torch.Tensor]] = []

    def snapshot(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> SceneSnapshot:
        """Return the configured scene and retain owned call arguments."""
        self.calls.append((timestamp, env_ids.clone()))
        return self.scene


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


@pytest.mark.parametrize(
    "name",
    [
        "free_motion_query",
        "pour_grasp_projection_provider",
        "joint_prefix_planner",
        "pick_closing_query_provider",
        "place_scene_provider",
        "place_continuation_provider",
        "measured_rigid_provider",
    ],
)
def test_pick_place_services_reject_untyped_ports_before_binding(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        AtomicActionEngine(_motion_generator(), **{name: object()})


@pytest.mark.parametrize(
    "name, method",
    [
        ("free_motion_query", "inspect"),
        ("joint_prefix_planner", "plan"),
        ("place_continuation_provider", "prepare"),
        ("measured_rigid_provider", "prepare"),
    ],
)
def test_pick_place_services_reject_another_robot_before_binding(
    name: str, method: str
) -> None:
    from types import SimpleNamespace

    calls: list[object] = []
    provider = SimpleNamespace(
        robot=object(), **{method: lambda *args, **kwargs: calls.append(args)}
    )
    with pytest.raises(ValueError, match="exact robot"):
        AtomicActionEngine(_motion_generator(), **{name: provider})
    assert calls == []


def test_initial_context_uses_configured_scene_provider() -> None:
    scene = SceneSnapshot(
        timestamp=1.5,
        version=3,
        entities={"target": EntityState(torch.eye(4))},
    )
    provider = StubSceneProvider(scene)
    engine = AtomicActionEngine(
        _motion_generator(),
        load_builtins=False,
        scene_provider=provider,
    )

    context = engine.initial_context(timestamp=1.5)

    assert context.scene is scene
    assert len(provider.calls) == 1
    timestamp, env_ids = provider.calls[0]
    assert timestamp == pytest.approx(1.5)
    assert torch.equal(env_ids, torch.tensor([0, 1]))


def test_initial_context_explicit_scene_overrides_configured_provider() -> None:
    provider = StubSceneProvider(SceneSnapshot(timestamp=0.0, version=1))
    explicit_scene = SceneSnapshot(timestamp=0.0, version=2)
    engine = AtomicActionEngine(
        _motion_generator(),
        load_builtins=False,
        scene_provider=provider,
    )

    context = engine.initial_context(scene=explicit_scene)

    assert context.scene is explicit_scene
    assert provider.calls == []


def test_initial_context_accepts_direct_affordance_sampling() -> None:
    engine = _engine()
    sampling = AffordanceSamplingContext(count=2, seed=11, attempt_id=4)

    context = engine.initial_context(affordance_sampling=sampling)

    assert context.affordance_sampling is sampling
    assert torch.equal(context.env_ids, torch.tensor([0, 1]))


def test_engine_rejects_invalid_scene_provider() -> None:
    with pytest.raises(TypeError, match="scene_provider must implement SceneProvider"):
        AtomicActionEngine(
            _motion_generator(),
            load_builtins=False,
            scene_provider=object(),  # type: ignore[arg-type]
        )


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
    semantics = ObjectSemantics(
        affordance=PressAffordance(press_position=(0.0, 0.0, 0.0)),
        geometry={},
        entity_id="button",
    )
    invocation = ActionInvocation(
        skill_id="press",
        goal=PressGoal(semantics, torch.eye(4)),
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


def test_engine_compile_preserves_opted_in_failed_trajectory_for_diagnostics() -> None:
    engine = _engine()
    engine.motion_generator.planner.preserve_failed_plan_positions = True
    engine.register(StubAction())
    target = torch.tensor([[1.0, 1.0, 1.0], [float("nan"), 2.0, 2.0]])

    compiled = engine.compile((_invocation(engine, target),))

    assert compiled.plan_success.tolist() == [True, False]
    assert torch.equal(
        compiled.trajectory.positions[1, -1], torch.tensor([0.0, 2.0, 2.0])
    )
    assert torch.all(compiled.projected_context.robot.qpos[1] == 0.0)


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


def test_engine_make_invocation_binds_direct_control_parts() -> None:
    engine = _engine(robot_dof=3)
    engine.register(StubAction())
    goal = JointPositionGoal(torch.ones(2, 3))
    motion_policy = MotionPolicy(sample_count=2)
    tracking_policy = TrackingPolicy.timed()

    invocation = engine.make_invocation(
        "stub",
        goal,
        control_parts={"primary": {"motion": "all"}},
        motion_policy=motion_policy,
        tracking_policy=tracking_policy,
        invocation_id="direct-call",
        revision=1,
    )
    target = invocation.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )

    assert invocation.skill_id == "stub"
    assert invocation.goal is goal
    assert invocation.motion_policy is motion_policy
    assert invocation.tracking_policy is tracking_policy
    assert invocation.invocation_id == "direct-call"
    assert invocation.revision == 1
    assert target.control_part == "all"
    assert engine.plan(invocation).plan_success.tolist() == [True, True]


def test_engine_make_invocation_requires_direct_binding() -> None:
    engine = _engine()
    engine.register(StubAction())

    with pytest.raises(ValueError, match="control_parts is required"):
        engine.make_invocation(
            "stub",
            JointPositionGoal(torch.ones(2, 3)),
        )


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


def test_engine_requires_registered_skill_id_for_direct_binding() -> None:
    engine = _engine()
    action = StubAction()
    with pytest.raises(TypeError, match="skill_id must be a string"):
        engine.bind_control_parts(
            action,  # type: ignore[arg-type]
            {"primary": {"motion": "all"}},
        )

    with pytest.raises(KeyError, match="No atomic action registered"):
        engine.bind_control_parts("stub", {"primary": {"motion": "all"}})

    engine.register(action)
    binding = engine.bind_control_parts("stub", {"primary": {"motion": "all"}})
    assert binding.endpoint("primary", "motion").target.control_part == "all"


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


def test_plan_transform_runs_after_normal_plan_validation() -> None:
    calls = []

    def transform(request, context, plan):
        calls.append((request, context, plan))
        return plan.snapshot()

    engine = _engine()
    engine.register(StubAction())

    plan = engine.plan(
        _invocation(engine, torch.ones(2, 3)),
        plan_transform=transform,
    )

    assert plan.success_all
    assert len(calls) == 1
    assert calls[0][2].skill_id == "stub"


def test_unsupported_nominal_source_never_falls_back_to_native_planning() -> None:
    engine = _engine()
    action = StubAction()
    native = Mock(wraps=action._plan)
    action._plan = native  # type: ignore[method-assign]
    engine.register(action)

    class Source:
        def instantiate(self, request, context):
            raise AssertionError(
                "unsupported skill must reject before invoking the source"
            )

    with pytest.raises(ValueError, match="does not support external nominal"):
        engine.plan(_invocation(engine, torch.zeros(2, 3)), nominal_source=Source())
    native.assert_not_called()


def test_plan_transform_must_return_a_validated_action_plan() -> None:
    engine = _engine()
    engine.register(StubAction())

    with pytest.raises(TypeError, match="must return an ActionPlan"):
        engine.plan(
            _invocation(engine, torch.ones(2, 3)),
            plan_transform=lambda request, context, plan: object(),
        )


def test_plan_transform_revalidates_mutated_tensor_invariants() -> None:
    engine = _engine()
    engine.register(StubAction())

    def transform(request, context, plan):
        del request, context
        plan.plan_success.resize_(1)
        return plan

    with pytest.raises(ValueError, match="plan_success batch"):
        engine.plan(
            _invocation(engine, torch.ones(2, 3)),
            plan_transform=transform,
        )


def test_plan_request_exposes_the_same_call_scoped_transform() -> None:
    engine = _engine()
    action = StubAction()
    engine.register(action)
    invocation = _invocation(engine, torch.ones(2, 3))
    request = action.resolve_request(invocation)
    calls = []

    result = engine.plan_request(
        request,
        plan_transform=lambda resolved, context, plan: calls.append(resolved) or plan,
    )

    assert result.success_all
    assert calls == [request]


def test_plan_transform_rejects_mutated_expected_effect_tensor() -> None:
    engine = _engine()
    engine.register(StubAction())

    def transform(request, context, plan):
        del request, context
        transformed = replace(
            plan,
            expected_effects=StateDelta(
                articulation_joint_updates={
                    ("drawer", "joint"): ArticulationJointState(torch.tensor([0.5]))
                }
            ),
        )
        effect = transformed.expected_effects.articulation_joint_updates[
            ("drawer", "joint")
        ]
        assert effect is not None
        effect.position.fill_(torch.nan)
        return transformed

    with pytest.raises(ValueError, match="position must be finite"):
        engine.plan(
            _invocation(engine, torch.ones(2, 3)),
            plan_transform=transform,
        )


def test_plan_transform_owns_returned_expected_effect_tensors() -> None:
    engine = _engine()
    engine.register(StubAction())
    retained: list[ArticulationJointState] = []

    def transform(request, context, plan):
        del request, context
        transformed = replace(
            plan,
            expected_effects=StateDelta(
                articulation_joint_updates={
                    ("drawer", "joint"): ArticulationJointState(torch.tensor([0.5]))
                }
            ),
        )
        effect = transformed.expected_effects.articulation_joint_updates[
            ("drawer", "joint")
        ]
        assert effect is not None
        retained.append(effect)
        return transformed

    result = engine.plan(
        _invocation(engine, torch.ones(2, 3)),
        plan_transform=transform,
    )
    retained[0].position.fill_(torch.nan)

    result_effect = result.expected_effects.articulation_joint_updates[
        ("drawer", "joint")
    ]
    assert result_effect is not None
    assert torch.equal(result_effect.position, torch.tensor([0.5]))


def test_action_plan_template_adapter_rebuilds_same_grid_commands() -> None:
    engine = _engine(batch_size=1)
    action = ThreeFrameStubAction()
    engine.register(action)
    invocation = _invocation(engine, torch.ones(1, 3))
    context = engine.initial_context()
    plan = engine.plan(invocation, context)
    request = action.resolve_request(invocation)
    adapter = ActionPlanTemplateAdapter(
        joint_names=("j0", "j1", "j2"),
        phase_permissions={"stub": ("joint_residual",)},
        phase_kinds={"stub": "free"},
    )
    source_context = SourceContext(
        "atomic_stub",
        "test:r1",
        "stub:0",
        SceneCase("case", "initial", "signature", "task", "robot"),
        ACTION_DT,
    )

    template = adapter.export_template(plan, context=source_context)
    variant = apply_trajectory_variant(
        template,
        TrajectoryVariant(1, "joint_residual", 1.0, "uniform"),
        cfg=default_variant_factors(redundancy=False),
        joint_limits=torch.tensor([[-2.0, 2.0]] * 3),
        control_dt=ACTION_DT,
    )
    rebuilt = engine.rebuild_plan_from_trajectory(
        request,
        context,
        plan,
        TimedTrajectory.from_positions(
            variant.positions.unsqueeze(0),
            env_ids=context.env_ids,
            dt=variant.dt.unsqueeze(0),
        ),
    )

    assert rebuilt.success_all
    assert rebuilt.commands.frame_count == plan.commands.frame_count
    assert not torch.equal(
        rebuilt.joint_trajectory.positions,
        plan.joint_trajectory.positions,
    )
    assert tuple(segment.name for segment in rebuilt.segments) == ("stub",)


def test_action_plan_template_adapter_implements_source_contract() -> None:
    engine = _engine(batch_size=1)
    engine.register(ThreeFrameStubAction())
    plan = engine.plan(_invocation(engine, torch.ones(1, 3)))
    adapter = ActionPlanTemplateAdapter(
        joint_names=("j0", "j1", "j2"),
        phase_permissions={"stub": ("joint_residual",)},
        phase_kinds={"stub": "free"},
    )
    context = SourceContext(
        source_id="atomic_stub",
        source_revision="test:r1",
        unit_id="stub:0",
        scene_case=SceneCase(
            "case",
            "initial",
            "signature",
            "task",
            "robot",
        ),
        control_dt=ACTION_DT,
    )

    assert isinstance(adapter, SourceAdapter)
    template = adapter.export_template(plan, context=context)

    assert (template.source_id, template.source_revision, template.template_id) == (
        "atomic_stub",
        "test:r1",
        "stub:0",
    )


def test_action_plan_template_adapter_rejects_inexact_phase_declarations() -> None:
    engine = _engine(batch_size=1)
    engine.register(ThreeFrameStubAction())
    plan = engine.plan(_invocation(engine, torch.ones(1, 3)))
    context = SourceContext(
        "atomic_stub",
        "test:r1",
        "stub:0",
        SceneCase("case", "initial", "signature", "task", "robot"),
        ACTION_DT,
    )

    for permissions, kinds in (
        ({}, {}),
        (
            {"stub": (), "extra": ()},
            {"stub": "free", "extra": "free"},
        ),
    ):
        adapter = ActionPlanTemplateAdapter(
            joint_names=("j0", "j1", "j2"),
            phase_permissions=permissions,
            phase_kinds=kinds,
        )
        with pytest.raises(ValueError, match="exact phase declaration"):
            adapter.export_template(plan, context=context)


def test_action_plan_template_adapter_rejects_batch_and_failed_plan() -> None:
    context = SourceContext(
        "atomic_stub",
        "test:r1",
        "stub:0",
        SceneCase("case", "initial", "signature", "task", "robot"),
        ACTION_DT,
    )
    adapter = ActionPlanTemplateAdapter(
        joint_names=("j0", "j1", "j2"),
        phase_permissions={"stub": ()},
        phase_kinds={"stub": "free"},
    )
    batched_engine = _engine(batch_size=2)
    batched_engine.register(ThreeFrameStubAction())
    batched = batched_engine.plan(_invocation(batched_engine, torch.ones(2, 3)))
    failed_engine = _engine(batch_size=1)
    failed_engine.register(ThreeFrameStubAction())
    failed = failed_engine.plan(
        _invocation(failed_engine, torch.full((1, 3), torch.nan))
    )

    with pytest.raises(ValueError, match="one successful row"):
        adapter.export_template(batched, context=context)
    with pytest.raises(ValueError, match="one successful row"):
        adapter.export_template(failed, context=context)


def test_mapped_retime_rebuilds_commands_and_preserves_terminal_velocity() -> None:
    from embodichain.lab.sim.motion.expansion import retime_with_time_map

    engine = _engine(batch_size=1)
    engine.robot.joint_names = ["j0", "j1", "j2"]
    action = RetimedStubAction()
    action._scene_dependencies = lambda request: ("object",)  # type: ignore[method-assign]
    engine.register(action)
    invocation = ActionInvocation(
        skill_id="move_joints",
        goal=JointPositionGoal(torch.ones(1, 3)),
        binding=engine.bind_control_parts(
            "move_joints", {"primary": {"motion": "all"}}
        ),
    )
    context = replace(engine.initial_context(), control_dt=ACTION_DT)
    plan = engine.plan(invocation, context)
    request = action.resolve_request(invocation)
    adapter = ActionPlanTemplateAdapter(
        joint_names=("j0", "j1", "j2"),
        phase_permissions={"move_joints": ("retime",)},
        phase_kinds={"move_joints": "free"},
    )
    template = adapter.export_template(
        plan,
        context=SourceContext(
            "stub",
            "1",
            "unit",
            SceneCase("case", "initial", "scene", "task", "robot"),
            ACTION_DT,
        ),
    )
    result = retime_with_time_map(template, duration_scale=1.5, control_dt=ACTION_DT)
    rebuilt = engine.rebuild_plan_from_mapped_trajectory(
        request,
        context,
        plan,
        template,
        result,
        joint_limits=torch.tensor([[-2.0, 2.0]] * 3),
        velocity_limits=torch.full((3,), 100.0),
        acceleration_limits=torch.full((3,), 1000.0),
    )
    assert rebuilt.commands.frame_count > plan.commands.frame_count
    assert rebuilt.segments[0].stop == rebuilt.commands.frame_count
    assert torch.equal(
        rebuilt.joint_trajectory.positions[:, -1],
        plan.joint_trajectory.positions[:, -1],
    )
    assert torch.equal(
        rebuilt.joint_trajectory.velocities[:, -1],
        plan.joint_trajectory.velocities[:, -1],
    )

    dependency_plan = replace(
        plan,
        scene_dependencies=("object",),
        scene_dependency_monitor_until={"object": 8},
    )
    with pytest.raises(ValueError, match="protected events"):
        engine.rebuild_plan_from_mapped_trajectory(
            request,
            context,
            dependency_plan,
            template,
            result,
            joint_limits=torch.tensor([[-2.0, 2.0]] * 3),
            velocity_limits=torch.full((3,), 100.0),
            acceleration_limits=torch.full((3,), 1000.0),
        )
    protected = retime_with_time_map(
        template, duration_scale=1.5, control_dt=ACTION_DT, protected_indices=(8,)
    )
    remapped = engine.rebuild_plan_from_mapped_trajectory(
        request,
        context,
        dependency_plan,
        template,
        protected,
        joint_limits=torch.tensor([[-2.0, 2.0]] * 3),
        velocity_limits=torch.full((3,), 100.0),
        acceleration_limits=torch.full((3,), 1000.0),
    )
    mapping = dict(
        zip(
            protected.time_map.source_event_indices,
            protected.time_map.execution_event_indices,
        )
    )
    assert remapped.scene_dependency_monitor_until["object"] == mapping[8]


def test_rebuild_plan_rejects_changed_same_grid_dt() -> None:
    engine = _engine(batch_size=1)
    action = ThreeFrameStubAction()
    engine.register(action)
    invocation = _invocation(engine, torch.ones(1, 3))
    context = engine.initial_context()
    plan = engine.plan(invocation, context)
    request = action.resolve_request(invocation)
    changed_timing = replace(
        plan.joint_trajectory,
        dt=torch.tensor([[0.0, 0.01, 0.03]]),
    )

    with pytest.raises(ValueError, match="same-grid.*dt"):
        engine.rebuild_plan_from_trajectory(
            request,
            context,
            plan,
            changed_timing,
        )


def test_rebuild_plan_validates_source_plan_identity() -> None:
    engine = _engine(batch_size=1)
    action = ThreeFrameStubAction()
    engine.register(action)
    invocation = _invocation(engine, torch.ones(1, 3))
    context = engine.initial_context()
    plan = engine.plan(invocation, context)
    request = action.resolve_request(invocation)

    with pytest.raises(ValueError, match="invocation_id"):
        engine.rebuild_plan_from_trajectory(
            request,
            context,
            replace(plan, invocation_id="different"),
            plan.joint_trajectory,
        )


def test_compile_transforms_plans_before_projecting_following_invocations() -> None:
    engine = _engine(batch_size=1)
    action = ThreeFrameStubAction()
    engine.register(action)
    observed_starts = []

    def transform(request, context, plan):
        observed_starts.append(context.robot.qpos.clone())
        positions = plan.joint_trajectory.positions.clone()
        if len(observed_starts) == 1:
            positions[:, -1] = 2.0
        return engine.rebuild_plan_from_trajectory(
            request,
            context,
            plan,
            TimedTrajectory.from_positions(
                positions, dt=plan.joint_trajectory.dt, env_ids=context.env_ids
            ),
        )

    compiled = engine.compile(
        (
            _invocation(engine, torch.ones(1, 3)),
            _invocation(engine, torch.full((1, 3), 3.0)),
        ),
        plan_transform=transform,
    )
    assert compiled.plan_success.all()
    assert torch.equal(observed_starts[1], torch.full((1, 3), 2.0))
    assert torch.equal(
        compiled.action_plans[0].joint_trajectory.positions[:, -1], observed_starts[1]
    )
    assert torch.equal(compiled.projected_context.robot.qpos, torch.full((1, 3), 3.0))


def test_adapter_can_export_an_explicit_successful_batch_row() -> None:
    engine = _engine(batch_size=2)
    engine.register(ThreeFrameStubAction())
    plan = engine.plan(
        _invocation(engine, torch.tensor([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]))
    )
    adapter = ActionPlanTemplateAdapter(
        joint_names=("j0", "j1", "j2"),
        phase_permissions={"stub": ("joint_residual",)},
        phase_kinds={"stub": "free"},
    )
    source = SourceContext(
        "source",
        "revision",
        "logical_variant_7",
        SceneCase("case", "initial", "signature", "task", "robot"),
        ACTION_DT,
    )
    template = adapter.export_template(plan, context=source, row_index=1)
    assert torch.equal(template.positions, plan.joint_trajectory.positions[1])
    assert template.template_id == "logical_variant_7"


def test_adapter_restores_protected_phases_gripper_and_near_equal_zero_time_joins() -> (
    None
):
    from dataclasses import replace
    from embodichain.lab.sim.atomic_actions.plans import TrajectorySegment

    class NineFrameAction(ThreeFrameStubAction):
        sample_count = 9
        binding_contract = ThreeFrameStubAction.binding_contract

    engine = _engine(batch_size=1)
    engine.register(NineFrameAction())
    plan = engine.plan(_invocation(engine, torch.ones(1, 3)))
    positions = plan.joint_trajectory.positions.clone()
    positions[:, 4] = positions[:, 3] + 5e-6
    dt = plan.joint_trajectory.dt.clone()
    dt[:, 4] = 0
    source = replace(
        plan,
        joint_trajectory=replace(plan.joint_trajectory, positions=positions, dt=dt),
        segments=(
            TrajectorySegment("approach", 0, 3),
            TrajectorySegment("release", 3, 6),
            TrajectorySegment("retract", 6, 9),
        ),
    )
    adapter = ActionPlanTemplateAdapter(
        joint_names=("j0", "j1", "j2"),
        controlled_joint_indices=(0, 1),
        phase_permissions={
            "approach": ("joint_residual",),
            "release": (),
            "retract": ("joint_residual",),
        },
        phase_kinds={"approach": "free", "release": "contact", "retract": "free"},
    )
    identity = SourceContext(
        "source",
        "revision",
        "variant_1",
        SceneCase("case", "initial", "signature", "task", "robot"),
        ACTION_DT,
    )
    template = adapter.export_template(source, context=identity)
    assert len(template.dt) == 8
    restored = adapter.restore_positions(source, template.positions + 0.1)
    assert torch.equal(restored[:, 2], positions[0, :, 2])
    assert torch.equal(restored[3:6], positions[0, 3:6])
    for endpoint in (0, 2, 3, 5, 6, 8):
        assert torch.equal(restored[endpoint], positions[0, endpoint])
    assert not torch.equal(restored[1, :2], positions[0, 1, :2])
    assert not torch.equal(restored[7, :2], positions[0, 7, :2])


def test_complete_request_budget_inherits_parent_and_preserves_default(monkeypatch):
    from embodichain.lab.sim.atomic_actions import PlanningBudget, RecoveryPolicy
    from embodichain.lab.sim.motion import search

    clock = [0.0]
    monkeypatch.setattr(search, "perf_counter", lambda: clock[0])

    class Measured(StubAction):
        binding_contract = StubAction.binding_contract

        def _plan(self, request, context):
            clock[0] += 1.0
            search.record_search("native_operation")
            return super()._plan(request, context)

    engine = _engine()
    engine.register(Measured())
    invocation = _invocation(engine, torch.ones(2, 3))
    default = engine.plan(invocation)
    assert "search_cost" not in default.diagnostics.metadata
    bounded = replace(
        invocation, recovery_policy=RecoveryPolicy(planning_budget=PlanningBudget(0.1))
    )
    exhausted = engine.plan(bounded)
    assert not exhausted.plan_success.any()
    assert exhausted.diagnostics.failure.code == "planning_wall_budget_exhausted"
    assert exhausted.diagnostics.failure.retryable is False
    assert search.current_search_cost() is None
    for parent_deadline in (None, 50.0):
        with search.search_cost_scope(wall_seconds=parent_deadline) as parent:
            plan = engine.plan(bounded)
            assert plan.plan_success.all()
            assert parent.counters["native_operation"] == 1
            assert plan.diagnostics.metadata["search_cost"]["budget_enforced"] is (
                parent_deadline is not None
            )


@pytest.mark.parametrize("retry_all_failed", [False, True])
def test_explicit_all_failed_retry_policy_preserves_failure_reason(retry_all_failed):
    from embodichain.lab.sim.atomic_actions import RecoveryPolicy

    class Failed(StubAction):
        binding_contract = StubAction.binding_contract

        def _plan(self, request, context):
            return self.failed_plan(
                request,
                context,
                message="original rejection",
                failure_code="original",
                retryable=True,
            )

    engine = _engine()
    engine.register(Failed())
    invocation = replace(
        _invocation(engine, torch.ones(2, 3)),
        recovery_policy=RecoveryPolicy(retry_all_failed_plans=retry_all_failed),
    )
    failure = engine.plan(invocation).diagnostics.failure
    assert failure.code == "original" and failure.retryable is retry_all_failed


def test_unsupported_planning_recovery_rejects_before_planner():
    from embodichain.lab.sim.atomic_actions import PlanningRecovery, RecoveryPolicy

    engine = _engine()
    action = StubAction()
    engine.register(action)
    action._plan = Mock(side_effect=AssertionError("unsupported planner ran"))
    invocation = replace(
        _invocation(engine, torch.ones(2, 3)),
        recovery_policy=RecoveryPolicy(
            planning_recovery=PlanningRecovery("not_installed", "stub", 1)
        ),
    )
    with pytest.raises(TypeError, match="does not support planning recovery"):
        engine.resolve(invocation)
    action._plan.assert_not_called()


def test_typed_recovery_proposal_owns_snapshot_and_rejects_foreign_invocation():
    from embodichain.lab.sim.atomic_actions import (
        PlanningRecovery,
        PlanningRecoveryOutcome,
        RecoveryPolicy,
    )

    class Proposed(StubAction):
        binding_contract = StubAction.binding_contract
        PlanningRecoveryType = PlanningRecovery

        def _plan(self, request, context):
            plan = super()._plan(request, context)
            rule = request.recovery_policy.planning_recovery
            proposal = PlanningRecoveryOutcome(
                self.skill_id, request.invocation_id, request.revision, rule.kind
            )
            return replace(plan, planning_recovery_outcome=proposal)

    engine = _engine()
    engine.register(Proposed())
    invocation = replace(
        _invocation(engine, torch.ones(2, 3)),
        recovery_policy=RecoveryPolicy(
            planning_recovery=PlanningRecovery("proposal", "stub", 1)
        ),
        invocation_id="unit/declared-occurrence",
    )
    plan = engine.plan(invocation)
    snapshot = plan.snapshot()
    assert snapshot.planning_recovery_outcome == plan.planning_recovery_outcome
    assert snapshot.planning_recovery_outcome is not plan.planning_recovery_outcome
    assert plan.plan_success.tolist() == [True, True]
    with pytest.raises(ValueError, match="does not identify this plan"):
        replace(
            plan,
            planning_recovery_outcome=replace(
                plan.planning_recovery_outcome, invocation_id="foreign"
            ),
        )


def test_final_qualification_can_only_reject_and_runs_before_projection() -> None:
    class QualifiedStub(StubAction):
        binding_contract = StubAction.binding_contract

        def _plan(self, request, context):
            observed_starts.append(context.robot.qpos.clone())
            return super()._plan(request, context)

        def _qualify_final_plan(self, request, context, plan):
            inspected.append(plan.joint_trajectory.positions.clone())
            # Even mutation of the supplied snapshot cannot alter accepted paths.
            plan.joint_trajectory.positions.fill_(99)
            return PlanQualification(
                torch.tensor([True, True, False]),
                PlannerDiagnostics("test_qualification", metadata={"checked": True}),
            )

    observed_starts, inspected = [], []
    engine = _engine(batch_size=3)
    engine.register(QualifiedStub())
    invocation = _invocation(
        engine,
        torch.tensor([[float("nan"), 0, 0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    compiled = engine.compile(
        [invocation, invocation],
        engine.initial_context(),
        plan_transform=lambda request, context, plan: plan,
    )
    first = compiled.action_plans[0]
    assert first.plan_success.tolist() == [False, True, False]
    assert first.diagnostics.failure.code == "final_plan_unqualified"
    torch.testing.assert_close(first.joint_trajectory.positions[1], inspected[0][1])
    torch.testing.assert_close(
        first.joint_trajectory.positions[[0, 2]], torch.zeros(2, 2, 3)
    )
    torch.testing.assert_close(
        first.joint_trajectory.dt, torch.tensor([[0.0, ACTION_DT]]).expand(3, -1)
    )
    assert all(
        frame.active_mask.tolist() == [False, True, False]
        for frame in first.commands.frames
    )
    torch.testing.assert_close(
        observed_starts[1],
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
    )


def test_successive_qualifications_keep_original_rejected_samples(tmp_path) -> None:
    from embodichain.lab.sim.motion.planners.utils import RejectedMotionCandidate

    class GeometryQualified(StubAction):
        binding_contract = StubAction.binding_contract

        def _qualify_final_plan(self, request, context, plan):
            # This is an owned inspection snapshot, including earlier rejection
            # evidence. Mutation cannot replace the engine's original samples.
            plan.rejected_candidate.positions.fill_(99)
            return PlanQualification(torch.tensor([True, False, True]))

    engine = _engine(batch_size=3)
    engine.register(GeometryQualified())
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="fixture">'
        + "".join(
            f'<joint name="j{i}" type="prismatic"><limit velocity="1"/></joint>'
            for i in range(3)
        )
        + "</robot>"
    )
    engine.robot.cfg.fpath = str(path)
    engine.robot.joint_names = ["j0", "j1", "j2"]
    engine.robot.get_qvel_limits.return_value = torch.ones(3, 3)
    target = torch.tensor([[0.2, 0, 0], [0.01, 0, 0], [0.0, 0, 0]])
    invocation = _invocation(engine, target)
    invocation = replace(
        invocation,
        motion_policy=replace(invocation.motion_policy, require_final_velocity=True),
    )
    context = engine.initial_context()
    plan = engine.plan(
        invocation, context, plan_transform=lambda request, current, value: value
    )
    assert plan.plan_success.tolist() == [False, False, True]
    assert plan.rejected_candidate.env_mask.tolist() == [True, True, False]
    torch.testing.assert_close(plan.rejected_candidate.positions[:, -1], target)
    torch.testing.assert_close(plan.joint_trajectory.positions, torch.zeros(3, 2, 3))
    torch.testing.assert_close(plan.rejected_candidate.dt, plan.joint_trajectory.dt)
    assert plan.diagnostics.metadata["final_velocity_qualification"]["valid_mask"] == [
        False,
        True,
        True,
    ]
    with pytest.raises(ValueError, match="cannot be a successful"):
        replace(plan, plan_success=torch.ones(3, dtype=torch.bool))
    active_commands = replace(
        plan.commands,
        frames=tuple(
            replace(frame, active_mask=torch.ones(3, dtype=torch.bool))
            for frame in plan.commands.frames
        ),
    )
    with pytest.raises(ValueError, match="active commands"):
        replace(plan, commands=active_commands)
    with pytest.raises(ValueError, match="time grid"):
        replace(
            plan,
            rejected_candidate=RejectedMotionCandidate(
                plan.rejected_candidate.positions,
                plan.rejected_candidate.dt + 0.01,
                plan.rejected_candidate.env_mask,
            ),
        )
    with pytest.raises(ValueError, match="DOF"):
        replace(
            plan,
            rejected_candidate=RejectedMotionCandidate(
                plan.rejected_candidate.positions[:, :, :1],
                plan.rejected_candidate.dt,
                plan.rejected_candidate.env_mask,
            ),
        )
    rebuilt = engine.rebuild_plan_from_trajectory(
        engine.resolve(invocation), context, plan, plan.joint_trajectory
    )
    torch.testing.assert_close(
        rebuilt.rejected_candidate.positions, plan.rejected_candidate.positions
    )


@pytest.mark.parametrize("mode", ["identity", "rebuild", "reject", "foreign"])
def test_planning_outcome_survives_transform_rebuild_and_final_rejection(
    mode: str,
) -> None:
    class OutcomeAction(StubAction):
        binding_contract = StubAction.binding_contract
        PlanningRecoveryType = PlanningRecovery

        def _plan(self, request, context):
            return replace(
                super()._plan(request, context),
                planning_recovery_outcome=PlanningRecoveryOutcome(
                    "stub", request.invocation_id, request.revision, "current-attempt"
                ),
            )

        def _qualify_final_plan(self, request, context, plan):
            return PlanQualification(torch.tensor([mode != "reject"]))

    engine = _engine(batch_size=1)
    action = OutcomeAction()
    engine.register(action)
    invocation = replace(
        _invocation(engine, torch.ones(1, 3)),
        invocation_id="invocation",
        recovery_policy=RecoveryPolicy(
            planning_recovery=PlanningRecovery("fixture", "stub", 1)
        ),
    )
    context = engine.initial_context()

    def transform(request, current, plan):
        if mode == "foreign":
            return replace(
                plan,
                planning_recovery_outcome=replace(
                    plan.planning_recovery_outcome, plan_id="historical-attempt"
                ),
            )
        if mode == "rebuild":
            # A host reconstructing commands need not manufacture an outcome.
            return action.build_plan(
                request,
                current,
                success=plan.plan_success,
                trajectory=plan.joint_trajectory,
            )
        return plan

    if mode == "foreign":
        with pytest.raises(ValueError, match="cannot replace.*recovery outcome"):
            engine.plan(invocation, context, plan_transform=transform)
        return
    plan = engine.plan(invocation, context, plan_transform=transform)
    assert plan.plan_success.tolist() == [mode != "reject"]
    outcome = plan.planning_recovery_outcome
    assert outcome.plan_id == "current-attempt"
    assert plan.snapshot().planning_recovery_outcome is not outcome
    rebuilt = engine.rebuild_plan_from_trajectory(
        action.resolve_request(invocation), context, plan, plan.joint_trajectory
    )
    assert rebuilt.planning_recovery_outcome == outcome
    assert rebuilt.planning_recovery_outcome is not outcome


def test_complete_request_hook_retains_sampling_transform_and_final_order(monkeypatch):
    events = []

    class CompleteStub(StubAction):
        binding_contract = StubAction.binding_contract

        def plan_complete_request(
            self, request, context, *, build, external_nominal=False
        ):
            assert not external_nominal
            events.append("complete")

            def candidate(selected, current):
                plan = self._plan(selected, current)
                events.append(("raw", plan.diagnostics.metadata["count"]))
                return plan

            result = build(request, planner=candidate)
            events.append(("returned", result.plan_success.tolist()))
            return result

        def _plan(self, request, context):
            events.append(("plan", request.motion_policy.sample_count))
            result = super()._plan(request, context)
            return replace(
                result,
                diagnostics=PlannerDiagnostics(
                    "stub", metadata={"count": request.motion_policy.sample_count}
                ),
            )

        def _qualify_final_plan(self, request, context, plan):
            events.append(("qualify", request.motion_policy.sample_count))
            return PlanQualification(
                torch.zeros_like(plan.plan_success), PlannerDiagnostics("stub")
            )

    engine = _engine()
    engine.register(CompleteStub())
    invocation = _invocation(engine, torch.ones(2, 3))
    invocation = replace(
        invocation,
        motion_policy=replace(
            invocation.motion_policy,
            sample_count_retries=(4,),
            require_final_velocity=True,
        ),
    )
    monkeypatch.setattr(
        engine,
        "_final_velocity_mask",
        lambda plan: torch.full_like(
            plan.plan_success, plan.diagnostics.metadata["count"] >= 4
        ),
    )

    def transformed(request, context, plan):
        events.append(("transform", request.motion_policy.sample_count))
        return plan

    result = engine.plan(
        invocation, engine.initial_context(), plan_transform=transformed
    )
    assert not result.plan_success.any()
    assert events == [
        "complete",
        ("plan", 2),
        ("raw", 2),
        ("transform", 2),
        ("plan", 4),
        ("raw", 4),
        ("transform", 4),
        ("qualify", 4),
        ("returned", [False, False]),
    ]


def test_default_complete_request_hook_delegates_once_and_preserves_result():
    action = StubAction()
    request, context, result = object(), object(), object()
    calls = []

    def build(value):
        calls.append(value)
        return result

    assert action.plan_complete_request(request, context, build=build) is result
    assert calls == [request]


def test_selected_candidate_transform_runs_once_without_replanning(monkeypatch):
    events = []

    class SelectingStub(StubAction):
        binding_contract = StubAction.binding_contract

        def plan_complete_request(
            self, request, context, *, build, external_nominal=False
        ):
            build(request, apply_transform=False)
            selected = build(request, apply_transform=False)
            events.append("select")
            return build.finalize_selected(request, selected)

        def _plan(self, request, context):
            events.append("plan")
            return super()._plan(request, context)

        def _qualify_final_plan(self, request, context, plan):
            events.append("qualify")
            return PlanQualification(
                torch.ones_like(plan.plan_success), PlannerDiagnostics("stub")
            )

    engine = _engine()
    engine.register(SelectingStub())
    invocation = _invocation(engine, torch.ones(2, 3))
    invocation = replace(
        invocation,
        motion_policy=replace(invocation.motion_policy, require_final_velocity=True),
    )

    def velocity(plan):
        events.append("velocity")
        return torch.full_like(plan.plan_success, "transform" not in events)

    monkeypatch.setattr(engine, "_final_velocity_mask", velocity)

    def transformed(request, context, plan):
        events.append("transform")
        return plan

    result = engine.plan(
        invocation, engine.initial_context(), plan_transform=transformed
    )
    assert not result.plan_success.any()
    assert events == [
        "plan",
        "velocity",
        "plan",
        "velocity",
        "select",
        "transform",
        "velocity",
        "qualify",
    ]


def test_complete_candidate_qualifier_is_local_and_keeps_generic_checks(monkeypatch):
    events = []

    class CandidateOwner(StubAction):
        binding_contract = StubAction.binding_contract

        def plan_complete_request(
            self, request, context, *, build, external_nominal=False
        ):
            def selected_qualifier(request, context, plan):
                events.append("selected_qualifier")
                return None

            build(request, action_qualifier=selected_qualifier)
            return build(request)

        def _qualify_final_plan(self, request, context, plan):
            events.append("normal_qualifier")
            return None

    engine = _engine()
    engine.register(CandidateOwner())
    invocation = _invocation(engine, torch.ones(2, 3))
    invocation = replace(
        invocation,
        motion_policy=replace(invocation.motion_policy, require_final_velocity=True),
    )

    def velocity(plan):
        events.append("velocity")
        return torch.ones_like(plan.plan_success)

    monkeypatch.setattr(engine, "_final_velocity_mask", velocity)

    def transform(request, context, plan):
        events.append("transform")
        return plan

    assert engine.plan(
        invocation, engine.initial_context(), plan_transform=transform
    ).plan_success.all()
    assert events == [
        "transform",
        "velocity",
        "selected_qualifier",
        "transform",
        "velocity",
        "normal_qualifier",
    ]


@pytest.mark.parametrize("retry_all_failed", [True, False])
@pytest.mark.parametrize("partial", [True, False])
def test_complete_plan_retry_policy_preserves_code_budgets_and_partial_rows(
    retry_all_failed, partial
):
    engine = _engine()
    engine.register(StubAction())
    targets = torch.full((2, 3), torch.nan)
    if partial:
        targets[0] = 0.0
    policy = RecoveryPolicy(
        max_action_retries=5, max_replans=4, retry_all_failed_plans=retry_all_failed
    )
    invocation = replace(_invocation(engine, targets), recovery_policy=policy)
    plan = engine.plan(invocation)
    assert plan.plan_success.tolist() == [partial, False]
    assert plan.diagnostics.failure.code == "planning_failed"
    assert plan.diagnostics.failure.retryable is (partial or retry_all_failed)
    assert plan.recovery_policy.max_action_retries == 5
    assert plan.recovery_policy.max_replans == 4


def test_terminal_complete_plan_policy_does_not_disable_internal_candidate_recovery():
    failures = []

    class RecoveringStub(StubAction):
        binding_contract = StubAction.binding_contract

        def plan_complete_request(
            self, request, context, *, build, external_nominal=False
        ):
            failed = build(request)
            failures.append(failed.diagnostics.failure.retryable)
            return build(replace(request, goal=JointPositionGoal(torch.zeros(2, 3))))

    engine = _engine()
    engine.register(RecoveringStub())
    invocation = replace(
        _invocation(engine, torch.full((2, 3), torch.nan)),
        recovery_policy=RecoveryPolicy(retry_all_failed_plans=False),
    )
    assert engine.plan(invocation).plan_success.all()
    assert failures == [True]


@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("representation", ["hold", "mask_only"])
def test_final_velocity_rejection_with_no_prior_failure_preserves_source_mask(
    tmp_path, partial, representation
):
    engine = _engine()
    engine.register(StubAction())
    robot = engine.robot
    robot.joint_names = ["a", "b", "c"]
    robot.mimic_ids = []
    robot.mimic_parents = []
    robot.mimic_multipliers = []
    path = tmp_path / "limits.urdf"
    path.write_text(
        '<robot name="limits">'
        + "".join(
            f'<joint name="{name}" type="prismatic"><limit velocity="1"/></joint>'
            for name in robot.joint_names
        )
        + "</robot>"
    )
    robot.cfg.fpath = str(path)
    robot.get_qvel_limits.return_value = torch.ones(2, 3)
    target = torch.ones(2, 3)
    if partial:
        target[0] = 0.005
    plain = _invocation(engine, target)
    original = engine.plan(plain)
    assert original.plan_success.all() and original.diagnostics.failure is None
    invocation = replace(
        plain,
        motion_policy=replace(
            plain.motion_policy,
            require_final_velocity=True,
            final_velocity_rejection=representation,
        ),
        recovery_policy=RecoveryPolicy(
            retry_all_failed_plans=False, max_action_retries=5
        ),
    )
    checked = engine.plan(invocation)
    # B0 InitialPlanCapture creates initial_plan_velocity_limit/False when
    # there was no prior failure. Shared Motion uses final_velocity_limit.
    assert checked.plan_success.tolist() == [partial, False]
    assert checked.diagnostics.failure.code == "final_velocity_limit"
    assert checked.diagnostics.failure.retryable is False
    assert torch.equal(checked.joint_trajectory.dt, original.joint_trajectory.dt)
    if representation == "mask_only":
        assert torch.equal(
            checked.joint_trajectory.positions, original.joint_trajectory.positions
        )
        assert checked.rejected_candidate is None
        for old, new in zip(
            original.commands.frames, checked.commands.frames, strict=True
        ):
            for left, right in zip(old.commands, new.commands, strict=True):
                assert torch.equal(left.payload.positions, right.payload.positions)
            assert torch.equal(old.active_mask, new.active_mask)
            assert torch.equal(old.hold_duration, new.hold_duration)
    else:
        assert torch.equal(
            checked.rejected_candidate.positions, original.joint_trajectory.positions
        )
        assert torch.equal(checked.rejected_candidate.dt, original.joint_trajectory.dt)
        if partial:
            assert torch.equal(
                checked.joint_trajectory.positions[0],
                original.joint_trajectory.positions[0],
            )
        assert not checked.joint_trajectory.positions[~checked.plan_success].any()


def test_mask_only_velocity_keeps_existing_partial_failure_classification(tmp_path):
    engine = _engine(batch_size=3)
    engine.register(StubAction())
    robot = engine.robot
    robot.joint_names = ["a", "b", "c"]
    robot.mimic_ids = []
    robot.mimic_parents = []
    robot.mimic_multipliers = []
    path = tmp_path / "limits.urdf"
    path.write_text(
        '<robot name="limits">'
        + "".join(
            f'<joint name="{name}" type="prismatic"><limit velocity="1"/></joint>'
            for name in robot.joint_names
        )
        + "</robot>"
    )
    robot.cfg.fpath = str(path)
    robot.get_qvel_limits.return_value = torch.ones(3, 3)
    target = torch.tensor([[0.005] * 3, [1.0] * 3, [float("nan")] * 3])
    plain = _invocation(engine, target)
    original = engine.plan(plain)
    invocation = replace(
        plain,
        motion_policy=replace(
            plain.motion_policy,
            require_final_velocity=True,
            final_velocity_rejection="mask_only",
        ),
        recovery_policy=RecoveryPolicy(retry_all_failed_plans=False),
    )
    checked = engine.plan(invocation)
    assert checked.plan_success.tolist() == [True, False, False]
    assert checked.diagnostics.failure.code == "planning_failed"
    assert checked.diagnostics.failure.retryable is True
    assert torch.equal(
        checked.joint_trajectory.positions, original.joint_trajectory.positions
    )


@pytest.mark.parametrize("representation", ["hold", "mask_only"])
def test_complete_mask_only_velocity_gate_does_not_preempt_action_recovery(
    tmp_path, representation
):
    observations = []

    class CompleteRecoveryStub(StubAction):
        binding_contract = StubAction.binding_contract

        def _plan(self, request, context):
            count = request.motion_policy.sample_count
            trajectory = TimedTrajectory.from_uniform_step(
                torch.lerp(
                    context.robot.qpos[:, None],
                    request.goal.target[:, None],
                    torch.linspace(0, 1, count)[None, :, None],
                ),
                env_ids=context.env_ids,
                step_dt=ACTION_DT,
            )
            return self.build_plan(
                request,
                context,
                success=torch.ones(2, dtype=torch.bool),
                trajectory=trajectory,
            )

        def plan_complete_request(
            self, request, context, *, build, external_nominal=False
        ):
            original = build(request)
            observations.append(original.plan_success.tolist())
            if not original.plan_success.all():
                return original
            # The action's continuation search must see the original candidate,
            # as B0 defers initial acceptance until its complete recovery returns.
            return build(
                replace(
                    request,
                    motion_policy=replace(request.motion_policy, sample_count=101),
                )
            )

    engine = _engine()
    engine.register(CompleteRecoveryStub())
    robot = engine.robot
    robot.joint_names = ["a", "b", "c"]
    robot.mimic_ids = []
    robot.mimic_parents = []
    robot.mimic_multipliers = []
    path = tmp_path / "limits.urdf"
    path.write_text(
        '<robot name="limits">'
        + "".join(
            f'<joint name="{name}" type="prismatic"><limit velocity="1"/></joint>'
            for name in robot.joint_names
        )
        + "</robot>"
    )
    robot.cfg.fpath = str(path)
    robot.get_qvel_limits.return_value = torch.ones(2, 3)
    plain = _invocation(engine, torch.ones(2, 3))
    selected = replace(
        plain,
        motion_policy=replace(
            plain.motion_policy,
            require_final_velocity=True,
            final_velocity_rejection=representation,
        ),
        recovery_policy=RecoveryPolicy(retry_all_failed_plans=False),
    )
    result = engine.plan(selected)
    assert observations == (
        [[True, True]] if representation == "mask_only" else [[False, False]]
    )
    assert result.plan_success.all().item() is (representation == "mask_only")
    if representation == "mask_only":
        assert torch.equal(result.joint_trajectory.positions[:, -1], torch.ones(2, 3))

    if representation == "mask_only":
        assert result.joint_trajectory.waypoint_count == 101


@pytest.mark.parametrize(
    "invalid",
    [None, "preview", "observed", "world", "rows", "engine", "request", "scene"],
)
def test_complete_suffix_preview_keeps_identity_and_call_scoped_transform(invalid):
    from embodichain.lab.sim.evidence import PhysicalObservationStamp

    observed, transforms = [], []

    class Complete(StubAction):
        binding_contract = StubAction.binding_contract

        def plan_complete_request(
            self, request, context, *, build, external_nominal=False
        ):
            if request.invocation_id == "suffix":
                return build(request)
            first = build(request)
            projected = replace(
                context.project(
                    qpos=first.joint_trajectory.positions[:, -1], task=context.task
                ),
                is_preview=True,
            )
            suffix = replace(request, invocation_id="suffix")
            if invalid == "preview":
                projected = replace(projected, is_preview=False)
            elif invalid == "observed":
                projected = replace(
                    projected,
                    observed_robot=projected.observed_robot.with_qpos(
                        projected.observed_robot.qpos
                    ),
                )
            elif invalid == "world":
                projected = replace(
                    projected,
                    world_source_stamp=PhysicalObservationStamp(
                        "other", "unit", 1, 0.0
                    ),
                )
            elif invalid == "rows":
                projected = replace(projected, env_ids=torch.tensor([1, 0]))
            elif invalid == "engine":
                other = _engine()
                other.register(StubAction())
                suffix = other.resolve(
                    replace(
                        _invocation(other, torch.ones(2, 3)), invocation_id="suffix"
                    )
                )
            elif invalid == "request":
                suffix = replace(suffix, invocation_id=request.invocation_id)
            elif invalid == "scene":
                projected = replace(projected, scene=SceneSnapshot.empty())
            if invalid is not None:
                with pytest.raises(ValueError, match="preview changed"):
                    build.plan_preview(suffix, projected)
            else:
                result = build.plan_preview(suffix, projected)
                assert result.plan_success.tolist() == [True, True]
            return first

        def _plan(self, request, context):
            observed.append((request.invocation_id, context.is_preview))
            return super()._plan(request, context)

    engine = _engine()
    engine.register(Complete())
    invocation = replace(_invocation(engine, torch.ones(2, 3)), invocation_id="primary")
    context = engine.initial_context()

    def transform(request, current, plan):
        transforms.append((request.invocation_id, current.is_preview))
        return plan

    assert engine.plan(invocation, context, plan_transform=transform).plan_success.all()
    expected = [("primary", False)] + ([] if invalid else [("suffix", True)])
    assert observed == expected and transforms == expected
    transforms.clear()
    engine.plan(invocation, context)
    assert transforms == []


def test_generic_planning_preserves_existing_cooperative_cancel_boundary():
    from embodichain.lab.sim.motion._budget import _planning_budget_scope

    calls = []

    class Budget:
        checks = 0

        def check(self):
            self.checks += 1
            if self.checks >= 2:
                raise RuntimeError("cancelled before skill planning")

        def charge(self, kind, amount=1):
            raise AssertionError("the default stub should not charge native work")

    class Measured(StubAction):
        binding_contract = StubAction.binding_contract

        def _plan(self, request, context):
            calls.append(1)
            return super()._plan(request, context)

    engine = _engine()
    engine.register(Measured())
    invocation = _invocation(engine, torch.ones(2, 3))
    context = engine.initial_context()
    before_rng = torch.random.get_rng_state()
    with pytest.raises(RuntimeError, match="cancelled before skill"):
        with _planning_budget_scope(Budget()):
            engine.plan(invocation, context)
    assert calls == []
    assert torch.equal(torch.random.get_rng_state(), before_rng)


def test_unsupported_segment_policy_rejects_before_request_preparation(monkeypatch):
    engine = _engine()
    action = StubAction()
    engine.register(action)
    preparation = Mock(
        side_effect=AssertionError("unsupported provider preparation ran")
    )
    monkeypatch.setattr(action, "_prepare_request", preparation)
    invocation = replace(
        _invocation(engine, torch.ones(2, 3)),
        motion_policy=MotionPolicy(
            strategy="ik_interp",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            segment_sample_count_retries=(4, 6),
        ),
    )
    with pytest.raises(ValueError, match="does not support Cartesian segment recovery"):
        engine.plan(invocation, engine.initial_context())
    preparation.assert_not_called()


def test_segment_capability_requires_actual_handler_and_effective_fingerprint():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value

    class CopiedMarker(StubAction):
        supports_segment_motion_recovery = True

    class Implemented(StubAction):
        def _validate_segment_motion_recovery(self, request, context):
            pass

    ordinary = StubAction.descriptor()
    copied = CopiedMarker.descriptor()
    capable = Implemented.descriptor()
    assert not ordinary.supports_segment_motion_recovery
    assert not copied.supports_segment_motion_recovery
    assert capable.supports_segment_motion_recovery
    assert (
        "supports_segment_motion_recovery" not in _canonical_value(ordinary)["fields"]
    )
    assert (
        _canonical_value(capable)["fields"]["supports_segment_motion_recovery"] is True
    )
    assert not replace(capable, skill_id="other").supports_segment_motion_recovery
    with pytest.raises(ValueError, match="init=False"):
        replace(ordinary, supports_segment_motion_recovery=True)
