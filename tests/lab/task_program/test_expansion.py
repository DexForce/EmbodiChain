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
    request = replace(_plan_request(invocation), workflow_call_index=1)
    transform = factory.create_plan_transform(request, engine=engine)
    assert callable(transform)

    rebuilt = transform(resolved, context, plan)
    record = factory.records[0]

    assert record.recipe_index == 1
    assert record.cycle_index == 0
    assert record.trajectory_requested == 1
    assert record.trajectory_selected == 1
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
    request = replace(_plan_request(invocation), workflow_call_index=1)
    transform = factory.create_plan_transform(request, engine=engine)
    assert transform is not None

    rebuilt = transform(resolved, context, plan)

    assert rebuilt.success_all
    assert rebuilt.joint_trajectory is not None
    assert rebuilt.joint_trajectory.batch_size == 2
    assert len(factory.records) == 2


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
    request = replace(_plan_request(invocation), workflow_call_index=1)
    transform = factory.create_plan_transform(request, engine=engine)
    assert transform is not None

    transform(resolved, context, plan)

    assert [record.recipe_index for record in factory.records] == [7, 9]


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
