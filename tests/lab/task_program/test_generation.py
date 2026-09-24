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

"""Source-neutral Task Program generation integration tests."""

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
    SceneCase,
    SourceAdapter,
    SourceContext,
    TrajectoryGenerationJobCfg,
    TrajectoryTemplate,
)
from embodichain.lab.task_program.integrations import (
    TaskProgramCandidatePlanTransformFactory,
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


def _engine() -> tuple[AtomicActionEngine, _ThreePhasePlaceAction]:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 3
    robot.joint_names = ("j0", "j1", "j2")
    robot.control_parts = {"all": object()}
    robot.get_qpos.return_value = torch.zeros(1, 3)
    robot.get_qvel.return_value = torch.zeros(1, 3)
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


def _place_invocation(engine: AtomicActionEngine) -> ActionInvocation:
    return ActionInvocation(
        skill_id="place",
        goal=JointPositionGoal(torch.ones(1, 3)),
        binding=engine.bind_control_parts(
            "place",
            {"primary": {"motion": "all"}},
        ),
        motion_policy=MotionPolicy(sample_count=7),
        invocation_id="place-call",
    )


def _profile(**overrides: object) -> TrajectoryGenerationJobCfg:
    payload: dict[str, object] = {
        "source": {
            "kind": "task_program",
            "source_id": "repeated_cube_pick_place",
            "source_revision": "config:repeated_pick_place_v1",
            "unit_scope": "action",
            "template_id": "place",
            "phase_permissions": {
                "approach": [],
                "release": [],
                "retract": ["joint_residual", "via_points"],
            },
            "phase_kinds": {
                "approach": "free",
                "release": "contact",
                "retract": "free",
            },
        },
        "augmentation": {
            "seed": 7,
            "max_variants_per_reference": 3,
            "factors": {
                "spatial": {
                    "enabled": True,
                    "method": ["joint_residual", "via_points"],
                    "joint_offset_scale": 0.05,
                },
                "timing": {"enabled": False},
            },
        },
        "affordance": {"enabled": False},
        "scheduling": {"policy": "fifo", "candidate_budget": 3},
        "execution": {"max_inflight": 1},
        "observation": {"enabled": False, "profiles": []},
        "collection": {"max_proposals": 3, "max_rollout_attempts": 3},
    }
    payload.update(overrides)
    return TrajectoryGenerationJobCfg.from_mapping(payload)


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


def _trajectory_template() -> TrajectoryTemplate:
    return TrajectoryTemplate(
        source_id="program",
        source_revision="revision",
        template_id="place:0",
        joint_names=("j0", "j1", "j2"),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        dt=torch.tensor([0.0, 0.1]),
    )


def test_task_program_source_adapter_delegates_to_atomic_adapter() -> None:
    atomic = ActionPlanTemplateAdapter(
        joint_names=("j0", "j1", "j2"),
        phase_permissions={"place": ("joint_residual",)},
        phase_kinds={"place": "free"},
    )
    adapter = TaskProgramSourceAdapter(atomic)
    context = SourceContext(
        "program",
        "revision",
        "place:0",
        SceneCase("case", "initial", "signature", "task", "robot"),
        0.1,
    )
    expected = _trajectory_template()
    source = Mock(spec=ActionPlan)

    assert adapter.kind == "task_program"
    assert isinstance(adapter, SourceAdapter)
    with patch.object(
        ActionPlanTemplateAdapter,
        "export_template",
        return_value=expected,
    ) as export_template:
        actual = adapter.export_template(source, context=context)

    assert actual is expected
    export_template.assert_called_once_with(source, context=context)


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


def _factory(
    profile: TrajectoryGenerationJobCfg,
    *,
    candidate_index: int,
) -> TaskProgramCandidatePlanTransformFactory:
    return TaskProgramCandidatePlanTransformFactory(
        profile,
        candidate_index=candidate_index,
        program_id="repeated_cube_pick_place",
        integration_id="repeated_pick_place_v1",
        robot_profile_id="franka_panda",
    )


def test_candidate_transform_builds_three_same_grid_place_plans() -> None:
    engine, invocation, resolved, context, plan = _planned_place()
    request = _plan_request(invocation)
    rebuilt = []
    selected_ids = []

    for index in range(3):
        factory = _factory(_profile(), candidate_index=index)
        transform = factory.create_plan_transform(request, engine=engine)
        assert callable(transform)
        candidate_plan = transform(resolved, context, plan)
        record = factory.records[0]
        rebuilt.append(candidate_plan)
        selected_ids.append(record.selected_candidate_id)
        assert record.candidate_index == index
        assert len(record.candidates) == 3
        assert len(record.templates) == 3
        assert torch.equal(candidate_plan.joint_trajectory.dt, plan.joint_trajectory.dt)
        release = plan.segment("release")
        assert torch.equal(
            candidate_plan.joint_trajectory.positions[:, release.start : release.stop],
            plan.joint_trajectory.positions[:, release.start : release.stop],
        )
        assert torch.equal(
            candidate_plan.joint_trajectory.positions[:, 0],
            plan.joint_trajectory.positions[:, 0],
        )
        assert torch.equal(
            candidate_plan.joint_trajectory.positions[:, -1],
            plan.joint_trajectory.positions[:, -1],
        )

    assert torch.equal(
        rebuilt[0].joint_trajectory.positions,
        plan.joint_trajectory.positions,
    )
    assert not torch.equal(
        rebuilt[1].joint_trajectory.positions,
        plan.joint_trajectory.positions,
    )
    assert not torch.equal(
        rebuilt[2].joint_trajectory.positions,
        plan.joint_trajectory.positions,
    )
    assert not torch.equal(
        rebuilt[1].joint_trajectory.positions,
        rebuilt[2].joint_trajectory.positions,
    )
    assert len(set(selected_ids)) == 3

    replay = _factory(_profile(), candidate_index=1)
    replay_transform = replay.create_plan_transform(request, engine=engine)
    assert callable(replay_transform)
    replayed = replay_transform(resolved, context, plan)
    assert torch.equal(
        replayed.joint_trajectory.positions,
        rebuilt[1].joint_trajectory.positions,
    )
    assert replay.records[0].selected_candidate_id == selected_ids[1]


def test_candidate_transform_ignores_unselected_skills() -> None:
    engine, invocation, _, _, _ = _planned_place()
    factory = _factory(_profile(), candidate_index=0)

    transform = factory.create_plan_transform(
        _plan_request(invocation, skill_id="pick"),
        engine=engine,
    )

    assert transform is None
    assert factory.records == ()


def test_candidate_transform_rejects_out_of_range_ordinal_atomically() -> None:
    with pytest.raises(ValueError, match="candidate_index"):
        _factory(_profile(), candidate_index=3)


@pytest.mark.parametrize(
    "override",
    [
        {"affordance": {"enabled": True}},
        {
            "augmentation": {
                "max_variants_per_reference": 3,
                "factors": {
                    "spatial": {
                        "enabled": True,
                        "method": ["joint_residual", "via_points"],
                    },
                    "timing": {"enabled": True},
                },
            }
        },
    ],
)
def test_candidate_transform_rejects_unsupported_profile_modes(
    override: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _factory(_profile(**override), candidate_index=0)


def test_candidate_transform_rejects_program_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="source_id"):
        TaskProgramCandidatePlanTransformFactory(
            _profile(),
            candidate_index=0,
            program_id="different_program",
            integration_id="repeated_pick_place_v1",
            robot_profile_id="franka_panda",
        )


def test_generation_records_own_template_tensors() -> None:
    engine, invocation, resolved, context, plan = _planned_place()
    factory = _factory(_profile(), candidate_index=1)
    transform = factory.create_plan_transform(
        _plan_request(invocation),
        engine=engine,
    )
    assert callable(transform)
    transform(resolved, context, plan)
    expected = factory.records[0].templates[0].positions.clone()

    factory.records[0].templates[0].positions.zero_()

    torch.testing.assert_close(
        factory.records[0].templates[0].positions,
        expected,
    )
