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

"""Only actual installed plans may authorize prismatic cleanup continuation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import uuid4
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicAction,
    AtomicActionEngine,
    ActionOptions,
    JointPositionGoal,
    RecoveryPolicy,
    TimedTrajectory,
    SkillBindingContract,
    SkillResourceSlot,
    SkillEndpointRequirement,
    JOINT_POSITION_CAPABILITY,
)
from embodichain.lab.sim.atomic_actions.articulation_continuation import (
    ArticulationPlanningOutcome,
    SlidePlanningRecovery,
    ArticulationWithdrawalPlanningRecovery,
)
from embodichain.lab.task_program.integrations.simulation.articulation_session import (
    ArticulationInvocationBinding,
    ArticulationContinuationRegistry,
)
from ._ordinary_pick_fixture import ordinary_fixture


@dataclass(frozen=True, slots=True, eq=False)
class _Options(ActionOptions):
    planned: bool = True
    changed: bool = True
    accepted: bool = True


class _Slide(AtomicAction):
    skill_id = "slide"
    GoalType = JointPositionGoal
    OptionsType = _Options
    PlanningRecoveryType = SlidePlanningRecovery
    binding_contract = SkillBindingContract(
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

    def _plan(self, request, context):
        scope = self.planning_services.articulation_continuation_provider.prepare(
            request, context
        )
        plan = self.build_plan(
            request,
            context,
            success=torch.tensor([request.skill_options.planned]),
            trajectory=TimedTrajectory.from_uniform_step(
                context.robot.qpos[:, None].repeat(1, 2, 1),
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
        )
        outcome = ArticulationPlanningOutcome(
            request.skill_id,
            request.invocation_id,
            request.revision,
            uuid4().hex,
            scope.scope_id,
            scope.history_version,
            request.skill_options.changed,
            request.skill_options.accepted,
        )
        return replace(plan, planning_recovery_outcome=outcome)


class _Withdraw(_Slide):
    skill_id = "move_end_effector"
    PlanningRecoveryType = ArticulationWithdrawalPlanningRecovery
    binding_contract = _Slide.binding_contract


@pytest.mark.parametrize("planned", [True, False])
def test_planning_preview_does_not_grant_installed_continuation(tmp_path, planned):
    generator, context, _, _, _ = ordinary_fixture(tmp_path)
    bindings = (
        ArticulationInvocationBinding(
            "slide0", "gen_sim.articulation_slide", "slide", withdrawal_sample_count=40
        ),
        ArticulationInvocationBinding(
            "withdraw0",
            "gen_sim.articulation_withdraw",
            "withdraw",
            predecessor_id="slide0",
        ),
        ArticulationInvocationBinding(
            "park0", "gen_sim.articulation_park", "park", predecessor_id="withdraw0"
        ),
    )
    registry = ArticulationContinuationRegistry(
        generator.robot,
        bindings,
        program_fingerprint="program",
        episode_id="episode",
        validate_scope=lambda: None,
        validate_observation=lambda value: None,
    )
    engine = AtomicActionEngine(
        generator, load_builtins=False, articulation_continuation_provider=registry
    )
    engine.register(_Slide())
    engine.register(_Withdraw())
    invocation = engine.make_invocation(
        "slide",
        JointPositionGoal(torch.zeros(1, 3)),
        control_parts={"primary": {"motion": "arm"}},
        skill_options=_Options(planned=planned),
        recovery_policy=RecoveryPolicy(planning_recovery=SlidePlanningRecovery()),
        invocation_id="slide0",
    )
    withdrawal = SimpleNamespace(
        invocation_id="withdraw0", skill_id="move_end_effector"
    )
    park = SimpleNamespace(invocation_id="park0", skill_id="move_joints")
    proposal = engine.plan(invocation, context)
    assert proposal.planning_recovery_outcome.withdrawal_selection_changed
    assert not registry.prepare(withdrawal, context).withdrawal_reselected
    assert registry.installed_facts == ()
    session = engine.start((invocation,), context, plan_lifecycle=registry)
    assert len(registry.installed_facts) == 1
    assert registry.prepare(withdrawal, context).withdrawal_reselected is planned
    with pytest.raises(ValueError, match="stale"):
        registry.on_plan_installed(session.plan_attempts[0])
    other = ArticulationContinuationRegistry(
        generator.robot,
        bindings,
        program_fingerprint="program",
        episode_id="episode",
        validate_scope=lambda: None,
        validate_observation=lambda value: None,
    )
    with pytest.raises(ValueError, match="another scope"):
        other.on_plan_installed(session.plan_attempts[0])
    assert not registry.prepare(park, context).park_after_withdrawal
    withdrawn = engine.make_invocation(
        "move_end_effector",
        JointPositionGoal(torch.zeros(1, 3)),
        control_parts={"primary": {"motion": "arm"}},
        skill_options=_Options(changed=False, accepted=False),
        recovery_policy=RecoveryPolicy(
            planning_recovery=ArticulationWithdrawalPlanningRecovery()
        ),
        invocation_id="withdraw0",
    )
    engine.plan(withdrawn, context)
    assert not registry.prepare(park, context).park_after_withdrawal
    engine.start((withdrawn,), context, plan_lifecycle=registry)
    assert not registry.prepare(park, context).park_after_withdrawal
    accepted = replace(
        withdrawn, skill_options=_Options(changed=False, accepted=True), revision=1
    )
    engine.start((accepted,), context, plan_lifecycle=registry)
    assert registry.prepare(park, context).park_after_withdrawal
    registry.close()
    with pytest.raises(RuntimeError, match="closed"):
        registry.prepare(park, context)
    assert len(registry.installed_facts) == 3


def test_withdrawal_targets_keep_source_world_up_and_short_outward_geometry():
    from embodichain.lab.sim.atomic_actions.articulation_continuation import (
        articulation_withdrawal_targets,
        short_articulation_withdrawal_targets,
    )

    current = torch.eye(4)[None]
    current[:, :3, 3] = torch.tensor([0.2, 0.1, 0.5])
    original = current.clone()
    targets = articulation_withdrawal_targets(current, torch.tensor([[0.0, -1.0, 0.0]]))
    torch.testing.assert_close(
        targets[0, :, :3, 3],
        torch.tensor([[0.2, 0.1, 0.5], [0.2, 0.1, 0.62], [0.2, 0.0, 0.62]]),
    )
    short = short_articulation_withdrawal_targets(targets)
    torch.testing.assert_close(
        short[0, :, :3, 3], torch.tensor([[0.2, 0.1, 0.5], [0.2, 0.08, 0.5]])
    )
    assert torch.equal(current, original)
    assert torch.equal(
        short[:, :, :3, :3], current[:, None, :3, :3].expand(-1, 2, -1, -1)
    )


def test_withdrawal_start_keeps_narrow_joint_band_and_open_hand_boundary():
    from embodichain.lab.sim.atomic_actions.articulation_continuation import (
        require_articulation_withdrawal_state,
    )

    require_articulation_withdrawal_state(
        torch.tensor([0.0098]),
        target=0.0,
        joint_span=0.02,
        observed_hand=torch.tensor([[0.08]]),
        open_hand=torch.zeros(1, 1),
    )
    with pytest.raises(ValueError, match="target was lost"):
        require_articulation_withdrawal_state(
            torch.tensor([0.0099]), target=0.0, joint_span=0.02
        )
    with pytest.raises(ValueError, match="open posture"):
        require_articulation_withdrawal_state(
            torch.zeros(1),
            target=0.0,
            joint_span=0.2,
            observed_hand=torch.tensor([[0.0801]]),
            open_hand=torch.zeros(1, 1),
        )


def test_lifecycle_composition_keeps_one_exact_owner():
    from embodichain.lab.task_program.runtime.lifecycle import (
        CompositeTaskProgramPlanLifecycleFactory,
    )

    owner = SimpleNamespace(
        on_plan_installed=lambda value: None, on_execution_aborted=lambda reason: None
    )

    class Factory:
        def __init__(self, value):
            self.value = value

        def create_plan_lifecycle(self, request, *, engine):
            return self.value

    composite = CompositeTaskProgramPlanLifecycleFactory(
        (Factory(None), Factory(owner))
    )
    assert composite.create_plan_lifecycle(object(), engine=object()) is owner
    conflict = CompositeTaskProgramPlanLifecycleFactory(
        (Factory(owner), Factory(owner))
    )
    with pytest.raises(ValueError, match="More than one"):
        conflict.create_plan_lifecycle(object(), engine=object())


@pytest.mark.parametrize(
    "kind", ["slide_bounded", "articulation_withdrawal", "articulation_park"]
)
def test_configured_articulation_recovery_is_closed(kind):
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
        _decode_action_options,
    )

    declaration = {"kind": kind, "revision": 1}
    config = {
        "max_replans": 0,
        "max_action_retries": 0,
        "planning_recovery": declaration,
    }
    rule = _decode_recovery_policy(config, path="recovery").planning_recovery
    assert rule.kind == kind
    with pytest.raises(ValueError, match="unsupported"):
        _decode_recovery_policy(
            {**config, "planning_recovery": {**declaration, "max_endpoints": 5}},
            path="recovery",
        )
    for action, flag in (
        ("place", "drawer_placement"),
        ("move_held_object", "drawer_transport"),
    ):
        options = _decode_action_options(
            {"kind": action, flag: True, "preserve_vertical": True}, path="options"
        )
        assert getattr(options, flag) is True and options.preserve_vertical is True
