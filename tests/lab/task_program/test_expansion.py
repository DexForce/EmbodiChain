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

"""Source-neutral Task Program expansion integration tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    ActionOptions,
    ActionPlan,
    ActionPlanTemplateAdapter,
    AtomicAction,
    AtomicActionEngine,
    JointPositionGoal,
    JOINT_POSITION_CAPABILITY,
    MotionPolicy,
    PlanningContext,
    ResolvedActionRequest,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    TimedTrajectory,
)
from embodichain.lab.sim.motion.expansion import (
    CombinedExpansionProfile,
    SceneCase,
    SourceAdapter,
    SourceContext,
    TrajectoryTemplate,
)
from embodichain.lab.task_program.integrations import (
    CombinedTaskProgramExpansionFactory,
    TaskProgramSourceAdapter,
)
from embodichain.lab.task_program.runtime import TaskProgramPlanRequest
from embodichain.lab.task_program.semantics.calls import RegisteredSemanticCall

CONTROL_DT = 0.1


class _ThreePhasePlaceAction(AtomicAction[JointPositionGoal, ActionOptions]):
    """Seven-sample Place stand-in with one editable retract interior sample."""

    skill_id: ClassVar[str] = "place"
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
        )
    )

    def _plan(
        self,
        request: ResolvedActionRequest[JointPositionGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        samples = torch.tensor(
            [0.0, 0.2, 0.4, 0.4, 0.6, 0.8, 1.0],
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )
        positions = samples[None, :, None].expand(
            context.batch_size,
            -1,
            context.robot.robot_dof,
        )
        trajectory = TimedTrajectory.from_uniform_step(
            positions,
            env_ids=context.env_ids,
            step_dt=CONTROL_DT,
        )
        return self.build_plan(
            request,
            context,
            success=True,
            trajectory=trajectory,
            segment_lengths={"approach": 2, "release": 2, "retract": 3},
        )


class _WaypointHandOverAction(_ThreePhasePlaceAction):
    """Two opposite arm assignments with the production Handover phases."""

    skill_id: ClassVar[str] = "hand_over"
    binding_contract: ClassVar[SkillBindingContract] = (
        _ThreePhasePlaceAction.binding_contract
    )

    def _plan(
        self,
        request: ResolvedActionRequest[JointPositionGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        lengths = {
            "pickup_approach": 25,
            "pickup_reach": 10,
            "pickup_close": 5,
            "pickup_transport": 25,
            "receive_approach": 25,
            "receive_close": 5,
            "handover_release": 5,
            "place": 25,
            "receive_release": 5,
        }
        positions = torch.zeros(2, sum(lengths.values()), 5)
        positions[0, :, 0] = torch.linspace(0, 0.8, positions.shape[1])
        positions[0, :, 1] = positions[0, :, 0] * 0.5
        positions[1, :, 2:4] = positions[0, :, :2]
        # The unused arm is stationary throughout the empty-hand approach.
        # Later phases contain motion of both arms and a closed hand.
        positions[:, 25:, :4] += 0.3
        positions[:, 35:125, 4] = 0.4
        return self.build_plan(
            request,
            context,
            success=True,
            trajectory=TimedTrajectory.from_uniform_step(
                positions, env_ids=context.env_ids, step_dt=CONTROL_DT
            ),
            segment_lengths=lengths,
        )


def _engine(batch_size: int = 1) -> tuple[AtomicActionEngine, _ThreePhasePlaceAction]:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 3
    robot.joint_names = ("j0", "j1", "j2")
    robot.control_parts = {"all": object()}
    robot.get_qpos.return_value = torch.zeros(batch_size, 3)
    robot.get_qvel.return_value = torch.zeros(batch_size, 3)
    robot.get_joint_ids.return_value = [0, 1, 2]
    robot.body_data = SimpleNamespace(qpos_limits=torch.tensor([[[-2.0, 2.0]] * 3]))
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub_planner"
    engine = AtomicActionEngine(generator, load_builtins=False)
    action = _ThreePhasePlaceAction()
    engine.register(action)
    return engine, action


def _place_invocation(
    engine: AtomicActionEngine, batch_size: int = 1
) -> ActionInvocation:
    return ActionInvocation(
        skill_id="place",
        goal=JointPositionGoal(torch.ones(batch_size, 3)),
        binding=engine.bind_control_parts(
            "place",
            {"primary": {"motion": "all"}},
        ),
        motion_policy=MotionPolicy(sample_count=7),
        invocation_id="place-call",
    )


def _planned_place() -> tuple[
    AtomicActionEngine,
    ActionInvocation,
    ResolvedActionRequest,
    PlanningContext,
    ActionPlan,
]:
    engine, action = _engine()
    invocation = _place_invocation(engine)
    context = engine.initial_context(control_dt=CONTROL_DT)
    plan = engine.plan(invocation, context)
    return engine, invocation, action.resolve_request(invocation), context, plan


def _plan_request(
    invocation: ActionInvocation,
    *,
    skill_id: str = "place",
) -> TaskProgramPlanRequest:
    return TaskProgramPlanRequest(
        workflow_id="repeated_cube_pick_place/move_cube",
        workflow_call_index=0,
        analysis_call_index=0,
        call=RegisteredSemanticCall(call_id="demo.place"),
        invocation=replace(invocation, skill_id=skill_id),
    )


def _combined_profile() -> CombinedExpansionProfile:
    return CombinedExpansionProfile.from_mapping(
        {
            "source": {
                "kind": "task_program",
                "source_id": "repeated_cube_pick_place",
                "source_revision": "config:repeated_pick_place_v1",
                "unit_scope": "episode",
                "template_id": "repeated_pick_place_episode",
            }
        }
    )


def _combined_profile_with_spatial(
    *,
    scale: float,
    tolerance: float | None,
) -> CombinedExpansionProfile:
    """Return the canonical profile with one explicit spatial amplitude."""
    payload = _combined_profile().to_dict()
    spatial = payload["trajectory"]["spatial"]
    spatial["enabled"] = True
    spatial["joint_offset_scale"] = scale
    spatial["dedup_tolerance"] = tolerance
    return CombinedExpansionProfile.from_mapping(payload)


def test_combined_factory_pins_recipe_affordance_before_atomic_planning() -> None:
    engine, invocation, _, context, _ = _planned_place()
    factory = CombinedTaskProgramExpansionFactory(
        _combined_profile(),
        candidate_index=7,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="franka_panda",
    )
    request = _plan_request(invocation, skill_id="pick_up")
    prepared = factory.prepare_planning_context(request, context, engine=engine)

    assert prepared.affordance_sampling is not None
    assert prepared.affordance_sampling.branch_overrides == (1,)
    assert prepared.affordance_sampling.episode_id == 7


def test_combined_factory_expands_place_on_requested_cycle_variant() -> None:
    engine, invocation, resolved, context, plan = _planned_place()
    factory = CombinedTaskProgramExpansionFactory(
        _combined_profile(),
        candidate_index=1,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="franka_panda",
    )
    request = replace(
        _plan_request(invocation), workflow_call_index=1, analysis_call_index=1
    )
    transform = factory.create_plan_transform(request, engine=engine)
    assert callable(transform)

    rebuilt = transform(resolved, context, plan)
    record = factory.records[0]

    assert record.recipe_index == 1
    assert record.cycle_index == 0
    assert record.trajectory_requested == 1
    assert record.trajectory_selected == 1
    assert torch.equal(rebuilt.joint_trajectory.dt, plan.joint_trajectory.dt)


def test_combined_factory_forwards_the_profile_dedup_tolerance() -> None:
    factory = CombinedTaskProgramExpansionFactory(
        _combined_profile_with_spatial(scale=0.005, tolerance=0.002),
        candidate_index=1,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="ur5_dh_pgi_140_80",
    )

    payload = factory._augmentation_payload(7)

    assert payload["coverage"] == {"joint_dedup_normalized_tol": 0.002}
    assert payload["factors"]["spatial"]["joint_offset_scale"] == 0.005

    # A profile that names no tolerance keeps the contract default, which then
    # has to stay below the profile's own offset scale.
    default_factory = CombinedTaskProgramExpansionFactory(
        _combined_profile(),
        candidate_index=1,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="ur5_dh_pgi_140_80",
    )
    assert "coverage" not in default_factory._augmentation_payload(7)


def test_combined_factory_keeps_a_conservative_amplitude_effective() -> None:
    # The packaged task declares a small offset scale, so the profile has to
    # carry a smaller deduplication tolerance. Otherwise every proposal measures
    # as a duplicate of the reference and the requested variant falls back to
    # the unmodified nominal trajectory.
    engine, invocation, resolved, context, plan = _planned_place()
    factory = CombinedTaskProgramExpansionFactory(
        _combined_profile_with_spatial(scale=0.005, tolerance=0.002),
        candidate_index=1,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="ur5_dh_pgi_140_80",
    )
    request = replace(
        _plan_request(invocation), workflow_call_index=1, analysis_call_index=1
    )
    transform = factory.create_plan_transform(request, engine=engine)
    assert transform is not None

    rebuilt = transform(resolved, context, plan)
    record = factory.records[0]

    assert record.trajectory_requested == 1
    assert record.trajectory_selected == 1
    assert record.fallback_count == 0
    assert not torch.equal(
        rebuilt.joint_trajectory.positions,
        plan.joint_trajectory.positions,
    )
    # Retiming stays disabled, so the variant keeps the reference arrival times.
    assert torch.equal(rebuilt.joint_trajectory.dt, plan.joint_trajectory.dt)


def test_combined_factory_expands_multi_environment_batch() -> None:
    engine, invocation, resolved, context, plan = _planned_place_batch(batch_size=2)
    factory = CombinedTaskProgramExpansionFactory(
        _combined_profile(),
        candidate_index=1,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="franka_panda",
    )
    request = replace(
        _plan_request(invocation), workflow_call_index=1, analysis_call_index=1
    )
    transform = factory.create_plan_transform(request, engine=engine)
    assert transform is not None

    rebuilt = transform(resolved, context, plan)

    assert rebuilt.success_all
    assert rebuilt.joint_trajectory is not None
    assert rebuilt.joint_trajectory.batch_size == 2
    assert len(factory.records) == 2


@pytest.mark.parametrize("generic", [False, True])
def test_combined_factory_preserves_reordered_environment_identity(
    generic: bool,
) -> None:
    engine, invocation, resolved, context, _ = _planned_place_batch(batch_size=2)
    stable_ids = torch.tensor([7, 3], dtype=torch.long)
    context = replace(context, env_ids=stable_ids)
    plan = engine.plan(invocation, context)
    payload = {
        "source": {
            "kind": "task_program",
            "source_id": "repeated_cube_pick_place",
            "source_revision": "config:repeated_pick_place_v1",
            "unit_scope": "episode",
            "template_id": "repeated_pick_place_episode",
        }
    }
    if generic:
        payload["source"].update(
            phase_permissions={
                "approach": [],
                "release": [],
                "retract": ["joint_residual", "via_points"],
            },
            phase_kinds={"approach": "free", "release": "contact", "retract": "free"},
        )
    factory = CombinedTaskProgramExpansionFactory(
        CombinedExpansionProfile.from_mapping(payload),
        candidate_index=1,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="franka_panda",
    )
    request = replace(
        _plan_request(invocation), workflow_call_index=1, analysis_call_index=1
    )
    rebuilt = factory.create_plan_transform(request, engine=engine)(
        resolved, context, plan
    )
    assert rebuilt.success_all
    assert torch.equal(rebuilt.joint_trajectory.env_ids, stable_ids)
    assert [record.env_id for record in factory.records] == stable_ids.tolist()
    assert [record.to_metadata()["env_id"] for record in factory.records] == [7, 3]


def test_combined_factory_uses_explicit_recipe_indices_per_row() -> None:
    engine, invocation, resolved, context, plan = _planned_place_batch(batch_size=2)
    factory = CombinedTaskProgramExpansionFactory(
        _combined_profile(),
        candidate_index=7,
        recipe_indices=(7, 9),
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="franka_panda",
    )
    request = replace(
        _plan_request(invocation), workflow_call_index=1, analysis_call_index=1
    )
    transform = factory.create_plan_transform(request, engine=engine)
    assert transform is not None

    transform(resolved, context, plan)

    assert [record.recipe_index for record in factory.records] == [7, 9]


def test_combined_factory_allows_padded_recipe_indices_for_discarded_rows() -> None:
    engine, invocation, resolved, context, plan = _planned_place_batch(batch_size=3)
    factory = CombinedTaskProgramExpansionFactory(
        _combined_profile(),
        candidate_index=7,
        recipe_indices=(7, 9, 9),
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="franka_panda",
    )
    request = replace(
        _plan_request(invocation), workflow_call_index=1, analysis_call_index=1
    )
    transform = factory.create_plan_transform(request, engine=engine)
    assert transform is not None

    transform(resolved, context, plan)

    assert [record.recipe_index for record in factory.records] == [7, 9, 9]


def _planned_place_batch(*, batch_size: int) -> tuple[
    AtomicActionEngine,
    ActionInvocation,
    ResolvedActionRequest,
    PlanningContext,
    ActionPlan,
]:
    engine, action = _engine(batch_size=batch_size)
    invocation = _place_invocation(engine, batch_size=batch_size)
    context = engine.initial_context(control_dt=CONTROL_DT)
    plan = engine.plan(invocation, context)
    return engine, invocation, action.resolve_request(invocation), context, plan


def test_expansion_uses_episode_call_index_across_segment_boundaries() -> None:
    engine, invocation, resolved, context, plan = _planned_place()
    factory = CombinedTaskProgramExpansionFactory(
        _combined_profile(),
        candidate_index=1,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="ur5_dh_pgi_140_80",
    )
    # A new segment's executor starts at local index zero; analysis retains the
    # episode's second call, which is Place rather than the preceding Pick.
    request = replace(
        _plan_request(invocation), workflow_call_index=0, analysis_call_index=1
    )
    assert factory.prepare_planning_context(request, context, engine=engine) is context
    factory.create_plan_transform(request, engine=engine)(resolved, context, plan)
    assert factory.records[0].workflow_call_index == 1
    assert factory.records[0].affordance_requested is None
    assert factory.records[0].trajectory_requested == 1


def test_expansion_preserves_hand_columns_phase_endpoints_and_zero_time_joins() -> None:
    engine, action = _engine()
    engine.robot.get_joint_ids.return_value = [0, 1]
    invocation = _place_invocation(engine)
    context = engine.initial_context(control_dt=CONTROL_DT)
    plan = engine.plan(invocation, context)
    resolved = action.resolve_request(invocation)
    source = plan.joint_trajectory
    positions, dt = source.positions.clone(), source.dt.clone()
    dt[:, 2] = 0.0
    positions[:, 2] = positions[:, 1]
    source = TimedTrajectory(
        positions=positions,
        dt=dt,
        env_ids=context.env_ids,
        velocities=None,
        accelerations=None,
    )
    plan = replace(plan, joint_trajectory=source)
    factory = CombinedTaskProgramExpansionFactory(
        _combined_profile_with_spatial(scale=0.005, tolerance=0.002),
        candidate_index=1,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="custom_gripper",
    )
    request = replace(
        _plan_request(invocation), workflow_call_index=0, analysis_call_index=1
    )
    rebuilt = factory.create_plan_transform(request, engine=engine)(
        resolved, context, plan
    )
    result = rebuilt.joint_trajectory
    assert torch.equal(result.dt, source.dt)
    assert torch.equal(result.positions[:, :, 2], source.positions[:, :, 2])
    assert torch.equal(result.positions[:, 2:4], source.positions[:, 2:4])
    for segment in plan.segments:
        for index in (segment.start, segment.stop - 1):
            assert torch.equal(result.positions[:, index], source.positions[:, index])
    assert not torch.equal(result.positions[:, 5, :2], source.positions[:, 5, :2])


def test_expansion_rejects_missing_variants_instead_of_repeating_reference() -> None:
    engine, invocation, resolved, context, plan = _planned_place()
    profile = _combined_profile()
    profile = replace(
        profile,
        trajectory=replace(
            profile.trajectory,
            spatial=replace(profile.trajectory.spatial, enabled=False),
        ),
    )
    factory = CombinedTaskProgramExpansionFactory(
        profile,
        candidate_index=1,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="ur5_dh_pgi_140_80",
    )
    request = replace(_plan_request(invocation), analysis_call_index=1)
    transform = factory.create_plan_transform(request, engine=engine)
    with pytest.raises(
        ValueError, match="requests trajectory variant 1, but only 1 survived"
    ):
        transform(resolved, context, plan)


def test_nominal_recipe_replays_exact_source_samples() -> None:
    engine, invocation, resolved, context, plan = _planned_place()
    factory = CombinedTaskProgramExpansionFactory(
        _combined_profile(),
        candidate_index=0,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="ur5_dh_pgi_140_80",
    )
    request = replace(_plan_request(invocation), analysis_call_index=1)
    rebuilt = factory.create_plan_transform(request, engine=engine)(
        resolved, context, plan
    )
    assert torch.equal(
        rebuilt.joint_trajectory.positions, plan.joint_trajectory.positions
    )
    assert torch.equal(rebuilt.joint_trajectory.dt, plan.joint_trajectory.dt)


@pytest.mark.parametrize("operator", ("joint_residual", "via_points"))
def test_handover_variants_fix_waypoints_held_phases_and_stationary_arm(
    operator: str,
) -> None:
    engine, _ = _engine(batch_size=2)
    engine.robot.dof = 5
    engine.robot.joint_names = ("l0", "l1", "r0", "r1", "hand")
    engine.robot.get_qpos.return_value = torch.zeros(2, 5)
    engine.robot.get_qvel.return_value = torch.zeros(2, 5)
    engine.robot.get_joint_ids.return_value = [0, 1, 2, 3]
    engine.robot.body_data.qpos_limits = torch.tensor([[[-2.0, 2.0]] * 5])
    action = _WaypointHandOverAction()
    engine.register(action)
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=JointPositionGoal(torch.ones(2, 5)),
        binding=engine.bind_control_parts("hand_over", {"primary": {"motion": "all"}}),
        motion_policy=MotionPolicy(sample_count=130),
        invocation_id="handover-call",
    )
    context = engine.initial_context(control_dt=CONTROL_DT)
    plan = engine.plan(invocation, context)
    payload = _combined_profile_with_spatial(scale=0.005, tolerance=0.002).to_dict()
    payload["trajectory"]["spatial"]["operators"] = [operator]
    factory = CombinedTaskProgramExpansionFactory(
        CombinedExpansionProfile.from_mapping(payload),
        candidate_index=1,
        recipe_indices=(1, 1),
        program_id="repeated_cube_pick_place",
        integration_id="handover_v1",
        robot_profile_id="dual_arm",
    )
    request = replace(
        _plan_request(invocation, skill_id="hand_over"), analysis_call_index=1
    )
    rebuilt = factory.create_plan_transform(request, engine=engine)(
        action.resolve_request(invocation), context, plan
    )
    source, expanded = plan.joint_trajectory, rebuilt.joint_trajectory
    pre_grasp = plan.segment("pickup_approach").stop - 1
    assert torch.equal(expanded.dt, source.dt)
    assert torch.equal(
        expanded.positions[:, pre_grasp:], source.positions[:, pre_grasp:]
    )
    assert torch.equal(expanded.positions[:, :, 4], source.positions[:, :, 4])
    for row, moving_joints in enumerate(([0, 1], [2, 3])):
        stationary_joints = [index for index in range(4) if index not in moving_joints]
        assert torch.equal(
            expanded.positions[row, :, stationary_joints],
            source.positions[row, :, stationary_joints],
        )
        assert not torch.equal(
            expanded.positions[row, 1:pre_grasp, moving_joints],
            source.positions[row, 1:pre_grasp, moving_joints],
        )
        assert torch.equal(expanded.positions[row, 0], source.positions[row, 0])
