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

"""Public Press declaration and bounded planning contracts."""

from __future__ import annotations

import pytest
import math
import random
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    ActionOptions,
    MotionPolicy,
    ObjectSemantics,
    PlanningRecovery,
    Press,
    PressAffordance,
    PressGoal,
    PressOptions,
    PressPlanningRecovery,
    RecoveryPolicy,
    ResolvedActionRequest,
    TrackingPolicy,
    validate_press_extra_distance,
)
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.lab.sim.motion.search import PlanningRandomState
from embodichain.lab.sim.atomic_actions.primitives import press as press_module


@pytest.mark.parametrize("value", [None, 0, 0.0, 0.005, 0.010])
def test_extra_press_distance_preserves_nullable_closed_bounds(value):
    assert validate_press_extra_distance(value) is None


@pytest.mark.parametrize(
    "value", [True, False, -1e-12, 0.0100001, float("nan"), float("inf"), "0.005"]
)
def test_extra_press_distance_rejects_unsupported_declarations(value):
    with pytest.raises(ValueError, match="press_extra_distance"):
        validate_press_extra_distance(value)


def test_planning_recovery_is_owned_complete_and_rejects_other_skills():
    rule = PressPlanningRecovery()
    policy = RecoveryPolicy(planning_recovery=rule)
    assert policy.planning_recovery is not rule
    assert asdict(policy)["planning_recovery"] == {
        "kind": "press_bounded",
        "skill_id": "press",
        "revision": 1,
        "max_complete_plans": 2,
        "retry_failed_cartesian_phases": True,
        "single_environment_only": True,
        "qualify_emitted_velocity": True,
    }
    binding = ActionBinding(owner_id="test", endpoints={})
    with pytest.raises(ValueError, match="cannot be used"):
        ActionInvocation("move_joints", object(), binding, recovery_policy=policy)
    with pytest.raises(ValueError, match="cannot be used"):
        ResolvedActionRequest(
            "move_joints",
            object(),
            binding,
            MotionPolicy(),
            TrackingPolicy.timed(),
            policy,
            ActionOptions(),
        )
    unsupported = RecoveryPolicy(
        planning_recovery=PlanningRecovery("unknown", "press", 1)
    )
    with pytest.raises(TypeError, match="does not support"):
        Press.descriptor().validate_planning_recovery(unsupported)
    Press.descriptor().validate_planning_recovery(policy)


def test_recovery_declaration_rejects_mutable_or_callable_data():
    @dataclass(frozen=True, slots=True)
    class InvalidRecovery(PlanningRecovery):
        value: object = None

    for value in ([1], {"retry": 1}, lambda: None):
        with pytest.raises(TypeError, match="immutable"):
            InvalidRecovery("invalid", "press", 1, value)


@pytest.mark.parametrize(
    "nominal,recovered", [(True, True), (False, True), (False, False)]
)
def test_phase_continuation_preserves_targets_seed_time_and_failure(
    monkeypatch, nominal, recovered
):
    action = Press()
    fk = Mock(return_value=torch.eye(4)[None])
    generator = object()
    action._planning_services = NS(robot=NS(compute_fk=fk), motion_generator=generator)
    seed = torch.zeros(1, 2)
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[0, 0, 0, 3], poses[0, 1, 0, 3] = 0.01, 0.02
    original = (torch.tensor([nominal]), (torch.zeros(1, 8, 2), torch.zeros(1, 8)))
    monkeypatch.setattr(action, "_plan_pose_segment", lambda *args, **kwargs: original)
    calls = []

    def solve(actual, targets, options, **kwargs):
        calls.append(options)
        assert actual is generator and len(targets) == 7
        torch.testing.assert_close(targets[-1].xpos, poses[:, -1], rtol=0, atol=0)
        torch.testing.assert_close(options.start_qpos, seed, rtol=0, atol=0)
        assert options.control_part == "arm" and options.interpolation_dt == 0.04
        assert options.preserve_cartesian_samples is True
        return (
            PlanResult(
                success=torch.tensor([True]),
                positions=torch.zeros(1, 8, 2),
                dt=torch.tensor([[0.0] + [0.04] * 7]),
            )
            if recovered
            else None
        )

    monkeypatch.setattr(press_module, "resolve_continuous_ik_path", solve)
    request = NS(
        motion_policy=MotionPolicy(strategy="ik_interp", sample_count=8),
        recovery_policy=RecoveryPolicy(planning_recovery=PressPlanningRecovery()),
    )
    records = []
    result = action._plan_press_phase(
        poses,
        seed,
        "arm",
        request,
        8,
        interpolation_dt=0.04,
        cartesian_linear=True,
        phase_name="press",
        recovery_records=records,
    )
    assert len(calls) == (0 if nominal else 1)
    if nominal or not recovered:
        assert result is original
    else:
        assert result[0].tolist() == [True]
        assert 0.28 <= float(result[1][1].sum()) <= 0.320001
    assert records == (
        []
        if nominal
        else [
            {
                "phase": "press",
                "sample_count": 8,
                "control_dt": 0.04,
                "accepted": recovered,
                "scope": "failure_only_seeded_ik_same_targets_and_time_not_physical_evidence",
            }
        ]
    )


@dataclass
class _Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics = field(
        default_factory=lambda: PlannerDiagnostics(backend="test")
    )


@dataclass
class _Request:
    goal: PressGoal
    recovery_policy: RecoveryPolicy
    skill_options: PressOptions


@pytest.mark.parametrize(
    "nominal,recovered", [(True, False), (False, True), (False, False)]
)
def test_complete_plan_budget_roll_and_rng_follow_source(
    monkeypatch, nominal, recovered
):
    action = Press()
    affordance = PressAffordance(
        press_axis=torch.tensor([1.0, 0, 0]), press_position=(0.1, 0.2, 0.3)
    )
    goal = PressGoal(
        ObjectSemantics(entity_id="button", geometry={}, affordance=affordance),
        torch.eye(4)[None],
    )
    request = _Request(
        goal,
        RecoveryPolicy(planning_recovery=PressPlanningRecovery()),
        PressOptions(contact_roll_alternatives=(math.pi / 2,)),
    )
    original, alternate = _Plan(torch.tensor([nominal])), _Plan(
        torch.tensor([recovered])
    )
    seen, draws = [], []

    def plan(candidate, context):
        seen.append(candidate)
        draws.append(
            (random.random(), float(np.random.random()), float(torch.rand(())))
        )
        return original if len(seen) == 1 else alternate

    monkeypatch.setattr(action, "_plan_checked", plan)
    state = PlanningRandomState.capture((torch.device("cpu"),))
    try:
        random.seed(73)
        np.random.seed(73)
        torch.manual_seed(73)
        expected_python = random.Random(73)
        expected_numpy = np.random.RandomState(73)
        expected_torch = torch.Generator().manual_seed(73)
        result = action._plan(request, NS(batch_size=1))
        expected = [
            (
                expected_python.random(),
                float(expected_numpy.random_sample()),
                float(torch.rand((), generator=expected_torch)),
            )
            for _ in seen
        ]
        assert draws == expected
        assert (random.random(), float(np.random.random()), float(torch.rand(()))) == (
            expected_python.random(),
            float(expected_numpy.random_sample()),
            float(torch.rand((), generator=expected_torch)),
        )
    finally:
        state.restore()
    assert len(seen) == (1 if nominal else 2)
    assert request.goal is goal and goal.semantics.affordance is affordance
    if not nominal:
        first = affordance.get_press_pose(torch.eye(4)[None])
        second = seen[1].goal.semantics.affordance.get_press_pose(torch.eye(4)[None])
        assert torch.equal(first[:, :3, 2:], second[:, :3, 2:])
        assert torch.equal(second[:, :3, 0], first[:, :3, 1])
        assert torch.equal(second[:, :3, 1], -first[:, :3, 0])
    assert result.plan_success.tolist() == [nominal or recovered]


def test_default_and_source_budget_do_not_enable_extra_complete_plans(monkeypatch):
    action = Press()
    goal = PressGoal(
        ObjectSemantics(
            entity_id="button",
            geometry={},
            affordance=PressAffordance(press_axis=torch.tensor([1.0, 0, 0])),
        ),
        torch.eye(4)[None],
    )
    original = _Plan(torch.tensor([False]))
    nominal = Mock(return_value=original)
    checked = Mock(
        side_effect=AssertionError("Default Press must not activate recovery.")
    )
    monkeypatch.setattr(action, "_plan_nominal", nominal)
    monkeypatch.setattr(action, "_plan_checked", checked)
    request = _Request(goal, RecoveryPolicy(), PressOptions())
    assert action._plan(request, NS(batch_size=1)) is original
    checked.assert_not_called()
    request.skill_options = PressOptions(contact_roll_alternatives=(math.pi / 2,))
    with pytest.raises(ValueError, match="require PressPlanningRecovery"):
        action._plan(request, NS(batch_size=1))
    request.recovery_policy = RecoveryPolicy(planning_recovery=PressPlanningRecovery())
    request.skill_options = PressOptions(
        contact_roll_alternatives=(math.pi / 2, -math.pi / 2)
    )
    with pytest.raises(ValueError, match="including nominal"):
        action._plan(request, NS(batch_size=1))
    assert nominal.call_count == 1


def test_non_singleton_phase_retains_original_failure(monkeypatch):
    action = Press()
    original = (torch.tensor([False, False]), (object(), object()))
    monkeypatch.setattr(action, "_plan_pose_segment", lambda *args, **kwargs: original)
    monkeypatch.setattr(
        press_module,
        "resolve_continuous_ik_path",
        lambda *a, **kw: pytest.fail("No B2 continuation"),
    )
    records = []
    result = action._plan_press_phase(
        torch.eye(4).repeat(2, 1, 1),
        torch.zeros(2, 2),
        "arm",
        NS(motion_policy=NS(strategy="ik_interp")),
        8,
        interpolation_dt=0.04,
        phase_name="press",
        recovery_records=records,
    )
    assert result is original and records == []


def test_transformed_full_hand_velocity_is_masked_without_replanning(monkeypatch):
    action = Press()
    action._planning_services = NS(robot=object(), planner_name="test")
    monkeypatch.setattr(
        press_module,
        "authored_joint_velocity_limits",
        lambda robot, part: torch.ones(1, 2),
    )
    positions = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]])
    before = positions.clone()
    plan = NS(
        plan_success=torch.tensor([True]),
        joint_trajectory=NS(positions=positions, dt=torch.tensor([[0.0, 0.04]])),
    )
    request = NS(
        recovery_policy=RecoveryPolicy(planning_recovery=PressPlanningRecovery())
    )
    result = action._qualify_final_plan(request, NS(), plan)
    assert result.valid_mask.tolist() == [False]
    assert torch.equal(positions, before)
    assert result.diagnostics.failure.code == "press_final_velocity_invalid"
    request.recovery_policy = RecoveryPolicy()
    assert action._qualify_final_plan(request, NS(), plan) is None


def test_successful_transform_without_joint_motion_is_rejected_explicitly(monkeypatch):
    action = Press()
    action._planning_services = NS(robot=object(), planner_name="test")
    limits = Mock(return_value=torch.ones(1, 2))
    monkeypatch.setattr(press_module, "authored_joint_velocity_limits", limits)
    plan = NS(plan_success=torch.tensor([True]), joint_trajectory=None)
    request = NS(
        recovery_policy=RecoveryPolicy(planning_recovery=PressPlanningRecovery())
    )

    result = action._qualify_final_plan(request, NS(), plan)

    assert result.valid_mask.tolist() == [False]
    assert result.diagnostics.failure.code == "press_final_joint_motion_missing"
    assert result.diagnostics.failure.retryable is False
    assert "joint trajectory" in result.diagnostics.messages[0]
    limits.assert_not_called()


def test_configured_recovery_is_closed_and_fully_fingerprinted():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
        _decode_action_options,
    )
    from embodichain.lab.task_program.integrations import (
        SimulationTaskProgramRegistration,
        SimulationSceneBinding,
        SimulationRobotSkillProfileBinding,
        ControlPartResourceBinding,
        ControlPartEndpointBinding,
    )
    from embodichain.lab.task_program.semantics import (
        SkillPolicyPreset,
        EffectAssurance,
    )

    config = {
        "max_replans": 0,
        "max_action_retries": 0,
        "planning_recovery": {"kind": "press_bounded", "revision": 1},
    }
    policy = _decode_recovery_policy(config, path="recovery")
    assert type(policy.planning_recovery) is PressPlanningRecovery
    options = _decode_action_options(
        {"kind": "press", "contact_roll_alternatives": [math.pi / 2]}, path="options"
    )
    assert options.contact_roll_alternatives == (math.pi / 2,)
    for extra in ({"callable": "module.factory"}, {"max_complete_plans": 3}):
        with pytest.raises(ValueError, match="unsupported"):
            _decode_recovery_policy(
                {
                    **config,
                    "planning_recovery": {**config["planning_recovery"], **extra},
                },
                path="recovery",
            )

    def registration(recovery):
        return SimulationTaskProgramRegistration(
            SimulationSceneBinding(registry_id="scene"),
            SimulationRobotSkillProfileBinding(
                profile_id="robot",
                resources=(
                    ControlPartResourceBinding(
                        resource_id="arm",
                        endpoints=(
                            ControlPartEndpointBinding(
                                endpoint_id="motion",
                                control_part="arm",
                                capabilities=frozenset({"motion.joint_position"}),
                            ),
                        ),
                    ),
                ),
                presets=(
                    SkillPolicyPreset(
                        "policy",
                        action_option_templates={},
                        effect_assurance=EffectAssurance.PROJECTED,
                        recovery_policy=recovery,
                    ),
                ),
                default_preset="policy",
            ),
        )

    registered = registration(policy)
    assert (
        registered.fingerprint
        != registration(RecoveryPolicy(max_replans=0, max_action_retries=0)).fingerprint
    )
    text = str(registered.fingerprint_payload())
    for field_name in asdict(policy.planning_recovery):
        assert field_name in text
