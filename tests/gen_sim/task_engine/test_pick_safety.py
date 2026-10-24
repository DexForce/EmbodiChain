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

from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import motion
from embodichain.gen_sim.task_engine._task_program.pick_safety import _inspect_pick_plan
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget


@pytest.mark.parametrize("joint,phase", [(0, "lift"), (1, "close")])
def test_velocity_rejection_reports_emitted_joint_frame_and_phase(
    monkeypatch, joint: int, phase: str
) -> None:
    positions = torch.zeros(1, 4, 2)
    frame = 3 if phase == "lift" else 2
    positions[:, frame:, joint] = 0.2
    dt = torch.tensor([[0.0, 0.04, 0.04, 0.04]])
    endpoints = {
        name: SimpleNamespace(
            task_state_key="arm",
            require_target=lambda cls, name=name: JointPositionTarget(
                name, (0,) if name == "motion" else (1,)
            ),
        )
        for name in ("motion", "grasp")
    }
    request = SimpleNamespace(
        binding=SimpleNamespace(endpoint=lambda slot, name: endpoints[name])
    )
    context = SimpleNamespace(
        robot=SimpleNamespace(qpos=torch.zeros(1, 2)), env_ids=torch.tensor([0])
    )
    plan = SimpleNamespace(
        joint_trajectory=SimpleNamespace(positions=positions, dt=dt),
        segments=tuple(
            SimpleNamespace(name=name, start=start, stop=stop)
            for name, start, stop in (
                ("approach", 0, 2),
                ("close", 2, 3),
                ("lift", 3, 4),
            )
        ),
    )
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.ones(1, 2)
    )
    result = _inspect_pick_plan(object(), request, context, plan)
    assert result["valid"] is False and result["reason"] == "final_velocity_rejected"
    evidence = result["velocity"][0]
    assert evidence["control_joint_index"] == joint and evidence["frame"] == frame
    assert evidence["phase"] == phase
    assert evidence["speed_ratio"] == pytest.approx(5.0)
