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

"""Bounded E6 opening changes retain geometry checks and one complete rebuild."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import e6_preshape as module
from embodichain.gen_sim.task_engine._task_program import grasp_filter, planning_probe
from embodichain.gen_sim.task_engine._task_program.e6_clearance import HandClearance
from embodichain.lab.sim.atomic_actions import MotionPolicy, SlideOptions


@pytest.mark.parametrize(
    "reason",
    [
        "opening",
        "nominal_success",
        "already_fits",
        "table",
        "too_wide",
        "rebuild_failure",
    ],
)
def test_call_local_opening_band_keeps_one_rebuild_and_all_geometry(
    monkeypatch, reason
):
    @dataclass
    class Diagnostics:
        metadata: dict = field(default_factory=dict)
        failure: Any = None
        messages: tuple = ()

    @dataclass
    class Plan:
        plan_success: torch.Tensor
        diagnostics: Diagnostics = field(default_factory=Diagnostics)

    @dataclass
    class Request:
        skill_options: SlideOptions
        binding: Any
        invocation_id: str = "e6"
        skill_id: str = "slide"
        motion_policy: MotionPolicy = field(default_factory=MotionPolicy)

    measured = []

    def clearance(fraction):
        measured.append(fraction)
        return HandClearance(
            torch.tensor([[0.0, 0.0, 0.0]]),
            0.016782 + (0.85 - fraction) * 0.145,
            0.03,
            0.0,
        )

    class Provider:
        _clearance = clearance(0.85)
        _clearance_factory = staticmethod(clearance)

    provider = Provider()
    monkeypatch.setattr(grasp_filter, "E6ApproachGraspPoseGenerator", Provider)
    monkeypatch.setattr(planning_probe, "initial_plan_capture", lambda _: None)
    request = Request(
        SlideOptions(preshape_fraction=0.85),
        NS(endpoint=lambda *a: NS(require_target=lambda _: NS(target_id="hand"))),
    )
    engine = NS(
        _articulation_calls={"e6": "slide"},
        robot=NS(device="cpu"),
        grasp_pose_generators={"hand": provider},
    )
    context = NS(batch_size=1, robot=NS(qpos=torch.zeros(1, 2)))
    calls = []
    width = (
        0.04
        if reason == "too_wide"
        else 0.01 if reason in {"nominal_success", "already_fits", "table"} else 0.02466
    )

    def build(selected):
        calls.append(selected.skill_options.preshape_fraction)
        pose = torch.eye(4)[None]
        pose[0, 2, 3] = 0.0 if reason == "table" else 0.2
        _, _, costs = module.filter_candidates(
            provider,
            provider._clearance,
            pose,
            torch.tensor([width]),
            torch.tensor([0.0]),
        )
        success = bool(torch.isfinite(costs).all())
        if reason in {"already_fits", "rebuild_failure"}:
            success = False
        return Plan(torch.tensor([success]))

    plan = module.plan_with_preshape(engine, request, context, build)
    if reason in {"opening", "rebuild_failure"}:
        assert calls == [0.85, 0.79]
        assert measured[-1] == 0.79
        detail = plan.diagnostics.metadata["gen_sim_e6_preshape"]
        assert detail["selected_fraction"] == 0.79
        assert detail["physical_success"] is None
        assert detail["accepted_plan"] is (reason == "opening")
        assert bool(plan.plan_success.all()) is (reason == "opening")
    else:
        assert calls == [0.85]
        assert bool(plan.plan_success.all()) is (reason == "nominal_success")
        if reason in {"nominal_success", "already_fits", "table"}:
            assert measured == [0.85]
        else:
            assert len(measured) <= 11 and measured[-1] >= 0.75
