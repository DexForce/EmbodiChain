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

"""Atomic export preserves source ownership, phase boundaries and passive geometry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.trajectory_generation.integrations.atomic import (
    export_pickup_templates,
)


def _inputs():
    q = torch.zeros(2, 9, 3)
    q[:, :3, 0] = torch.tensor([0.0, 0.1, 0.2])
    q[:, 3:, 0] = 0.2
    q[:, 5:, 1] = 0.03
    dt = torch.full((2, 9), 0.05, dtype=torch.float64)
    dt[:, (0, 3)] = 0.0
    compiled = SimpleNamespace(
        action_plans=(
            SimpleNamespace(skill_id="move_end_effector"),
            SimpleNamespace(skill_id="pick_up"),
        ),
        plan_success=torch.ones(2, dtype=torch.bool),
        trajectory=SimpleNamespace(positions=q, dt=dt, env_ids=torch.arange(2)),
        action_waypoint_offset=lambda index: 3,
        segment=lambda index, name: {
            "approach": SimpleNamespace(start=3, stop=5),
            "close": SimpleNamespace(start=5, stop=7),
            "lift": SimpleNamespace(start=7, stop=9),
        }[name],
    )
    robot = SimpleNamespace(
        num_instances=2,
        dof=3,
        joint_names=("arm", "finger", "mimic"),
        mimic_ids=(2,),
        mimic_parents=(1,),
        mimic_multipliers=(-1.0,),
        mimic_offsets=(0.0,),
        get_joint_ids=lambda name: (0,),
    )
    return compiled, robot


def test_export_keeps_atomic_commands_and_declares_real_hold_and_mimic_geometry():
    compiled, robot = _inputs()
    originals = compiled.trajectory.positions.clone(), compiled.trajectory.dt.clone()
    templates = export_pickup_templates(compiled, robot, control_dt=0.05, hold_steps=3)
    assert len(templates) == 2
    for template in templates:
        assert template.positions.shape == (12, 3)
        torch.testing.assert_close(template.positions[:9, :2], originals[0][0, :, :2])
        torch.testing.assert_close(template.positions[:, 2], -template.positions[:, 1])
        assert template.dt[0] == 0 and torch.all(template.dt[1:] == 0.05)
        assert template.phases[0].allowed_operators == ("joint_residual",)
        assert all(not p.allowed_operators for p in template.phases[1:])
        assert template.phases[-1].start_index == 9
    assert torch.equal(compiled.trajectory.positions, originals[0])
    assert torch.equal(compiled.trajectory.dt, originals[1])
    templates[0].positions.zero_()
    assert templates[1].positions.abs().sum() > 0


@pytest.mark.parametrize(
    "failure",
    ["failed_plan", "boundary", "clock", "row_order", "skill", "missing_segment"],
)
def test_export_rejects_incompatible_atomic_plan(failure):
    compiled, robot = _inputs()
    if failure == "failed_plan":
        compiled.plan_success[0] = False
    if failure == "boundary":
        compiled.trajectory.positions[:, 3, 0] += 0.1
    if failure == "clock":
        compiled.trajectory.dt[:, 5] = 0.1
    if failure == "row_order":
        compiled.trajectory.env_ids = torch.tensor([1, 0])
    if failure == "skill":
        compiled.action_plans[1].skill_id = "place"
    if failure == "missing_segment":
        compiled.segment = lambda index, name: SimpleNamespace(start=0, stop=1)
    with pytest.raises(ValueError):
        export_pickup_templates(compiled, robot, control_dt=0.05)
