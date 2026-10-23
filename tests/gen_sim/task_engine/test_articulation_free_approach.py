# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    articulation_recovery as recovery,
    motion,
)
from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics


@dataclass
class Plan:
    plan_success: torch.Tensor
    joint_trajectory: object
    diagnostics: PlannerDiagnostics


@pytest.mark.parametrize(
    "planner,robot_clear,scene_clear,frames,expected",
    [
        (True, True, True, 3, True),
        (False, True, True, 3, False),
        (True, False, True, 3, False),
        (True, True, False, 3, False),
        (True, True, True, 6, False),
    ],
)
def test_push_free_prefix_keeps_contact_suffix_hand_commands_and_all_gates(
    monkeypatch,
    planner,
    robot_clear,
    scene_clear,
    frames,
    expected,
) -> None:
    full = torch.zeros(1, 8, 4)
    full[:, :4, :2] = torch.linspace(0.0, 0.003, 4)[None, :, None]
    full[:, 4:, :2] = 0.003
    full[:, :, 2] = torch.linspace(0.0, 0.5, 8)
    before = full.clone()
    dt = torch.full((1, 8), 0.04)
    dt[:, 0] = 0.0
    robot = NS(get_qpos_limits=lambda **kw: torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]))
    generator = NS(robot=robot)
    scope = recovery.PlanningScope(generator, "slide")
    tracks = (full[0, :4, :2].clone(), *[full[0, 4:5, :2].clone() for _ in range(4)])
    scope.tracks = tracks
    request = NS(
        binding=NS(
            endpoint=lambda *a: NS(
                require_target=lambda cls: JointPositionTarget("arm", (0, 1))
            )
        )
    )
    context = NS(
        robot=NS(qpos=torch.zeros(1, 4)),
        env_ids=torch.tensor([0]),
        require_control_dt=lambda: 0.04,
    )
    original = Plan(
        torch.tensor([True]),
        NS(positions=full, dt=dt),
        PlannerDiagnostics(backend="test"),
    )
    built = []

    def prefix(owner, simulation, current, part, start, target, locked):
        assert owner is generator and part == "arm"
        assert torch.equal(start, context.robot.qpos[:, :2])
        assert torch.equal(target, full[:, 3, :2])
        assert torch.equal(locked, context.robot.qpos)
        positions = (
            torch.linspace(0.0, 0.003, frames)[None, :, None]
            .expand(1, frames, 2)
            .clone()
        )
        times = torch.full((1, frames), 0.04)
        times[:, 0] = 0
        return NS(success=torch.tensor([planner]), positions=positions, dt=times)

    def build(selected):
        assert selected is request
        built.append(True)
        values = full.clone()
        values[:, :4, :2] = scope.tracks[0]
        scope.cursor = len(tracks)
        return Plan(
            torch.tensor([True]),
            NS(positions=values, dt=dt),
            PlannerDiagnostics(backend="test"),
        )

    monkeypatch.setattr(motion, "_curobo_joint_prefix", prefix)
    monkeypatch.setattr(
        recovery, "_joint_velocity_limits", lambda *a: torch.full((1, 4), 100.0)
    )
    monkeypatch.setattr(
        recovery, "check_robot_motion", lambda *a: {"valid": robot_clear}
    )
    monkeypatch.setattr(
        recovery, "check_free_motion", lambda *a: {"valid": scene_clear}
    )
    result, report = recovery._repair_push_approach(
        scope, request, context, original, tracks, build, object()
    )
    assert (result is not None) is expected
    if expected:
        assert torch.equal(result.joint_trajectory.positions[:, 4:], full[:, 4:])
        assert torch.equal(result.joint_trajectory.positions[:, :, 2:], full[:, :, 2:])
        assert report["valid"] is True
    assert torch.equal(full, before)
    assert scope.tracks is tracks and scope.cursor == 0
