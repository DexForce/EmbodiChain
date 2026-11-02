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

"""E9 failure-only recovery preserves phase targets, incoming seed and time."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import motion
from embodichain.gen_sim.task_engine._task_program.press_runtime import GenSimPress
from embodichain.lab.sim.atomic_actions import (
    ObjectSemantics,
    Press,
    PressAffordance,
    PressGoal,
)
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy


@pytest.mark.parametrize(
    "nominal,recovered", [(True, True), (False, True), (False, False)]
)
def test_press_phase_retries_only_failure_without_changing_targets(
    monkeypatch, nominal, recovered
):
    action = GenSimPress()
    fk = Mock(return_value=torch.eye(4)[None])
    generator = object()
    action._planning_services = NS(robot=NS(compute_fk=fk), motion_generator=generator)
    action._press_recovery_records = []
    action._press_phase_index = 0
    seed = torch.zeros(1, 2)
    poses = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
    poses[0, 0, 0, 3], poses[0, 1, 0, 3] = 0.01, 0.02
    original = (torch.tensor([nominal]), (torch.zeros(1, 8, 2), torch.zeros(1, 8)))
    monkeypatch.setattr(Press, "_plan_pose_segment", lambda *args, **kwargs: original)
    calls = []

    def solve(actual_generator, targets, options, **kwargs):
        calls.append(options)
        assert actual_generator is generator
        assert len(targets) == 7
        assert torch.equal(targets[-1].xpos, poses[:, -1])
        assert torch.equal(options.start_qpos, seed)
        assert options.control_part == "arm"
        assert options.interpolation_dt == 0.04
        assert options.preserve_cartesian_samples is True
        if not recovered:
            return None
        return NS(
            success=torch.tensor([True]),
            positions=torch.zeros(1, 8, 2),
            dt=torch.tensor([[0.0] + [0.04] * 7]),
        )

    monkeypatch.setattr(motion, "resolve_continuous_ik_path", solve)
    request = NS(motion_policy=MotionPolicy(strategy="ik_interp", sample_count=8))
    result = action._plan_pose_segment(
        poses, seed, "arm", request, 8, interpolation_dt=0.04, cartesian_linear=True
    )
    assert len(calls) == (0 if nominal else 1)
    if nominal or not recovered:
        assert result is original
    else:
        assert result[0].tolist() == [True]
        # The shared retimer may round a floating-point duration up one tick.
        assert 0.28 <= float(result[1][1].sum()) <= 0.320001
        assert torch.allclose(
            result[1][1][:, 1:], torch.full_like(result[1][1][:, 1:], 0.04)
        )
    if nominal:
        fk.assert_not_called()
    else:
        fk.assert_called_once()
        assert action._press_recovery_records[0]["accepted"] is recovered


def test_non_singleton_press_keeps_original_failure(monkeypatch):
    action = GenSimPress()
    action._press_recovery_records = []
    action._press_phase_index = 0
    original = (torch.tensor([False, False]), (object(), object()))
    monkeypatch.setattr(Press, "_plan_pose_segment", lambda *args, **kwargs: original)
    monkeypatch.setattr(
        motion,
        "resolve_continuous_ik_path",
        lambda *a, **kw: pytest.fail("no batch fallback"),
    )
    result = action._plan_pose_segment(
        torch.eye(4)[None].repeat(2, 1, 1),
        torch.zeros(2, 2),
        "arm",
        NS(motion_policy=NS(strategy="ik_interp")),
        8,
        interpolation_dt=0.04,
    )
    assert result is original


@dataclass
class _Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics = field(
        default_factory=lambda: PlannerDiagnostics(backend="test")
    )


@dataclass
class _Request:
    goal: object


@pytest.mark.parametrize(
    "nominal,recovered", [(True, False), (False, True), (False, False)]
)
def test_failure_only_quarter_roll_preserves_contact_point_axis_and_request(
    monkeypatch, nominal, recovered
):
    action = GenSimPress()
    affordance = PressAffordance(
        press_axis=torch.tensor([1.0, 0.0, 0.0]), press_position=(0.1, 0.2, 0.3)
    )
    goal = PressGoal(
        ObjectSemantics(entity_id="button", geometry={}, affordance=affordance),
        torch.eye(4)[None],
    )
    request = _Request(goal)
    original = _Plan(torch.tensor([nominal]))
    selected = _Plan(torch.tensor([recovered]))
    seen = []

    def plan(candidate, context):
        seen.append(candidate)
        return original if len(seen) == 1 else selected

    monkeypatch.setattr(action, "_plan_checked", plan, raising=False)
    result = action._plan(request, NS(batch_size=1))
    assert len(seen) == (1 if nominal else 2)
    assert request.goal is goal and goal.semantics.affordance is affordance
    if nominal:
        assert result is original
    else:
        first = affordance.get_press_pose(torch.eye(4)[None])
        second = seen[1].goal.semantics.affordance.get_press_pose(torch.eye(4)[None])
        assert torch.equal(first[:, :3, 2:], second[:, :3, 2:])
        assert torch.equal(second[:, :3, 0], first[:, :3, 1])
        assert torch.equal(second[:, :3, 1], -first[:, :3, 0])
        assert result.plan_success.tolist() == [recovered]
