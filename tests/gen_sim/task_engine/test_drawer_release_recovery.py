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

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import drawer_runtime as drawer
from embodichain.lab.sim.atomic_actions import PlaceGoal, SceneEntityPose
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics


def test_reduced_release_keeps_outside_rim_and_payload_target_unchanged() -> None:
    observation = object.__new__(drawer.DrawerObservation)
    observation.route = NS(
        binding=NS(axis=(1.0, 0.0, 0.0), axis_sign=1),
        lower=(-0.1, -0.1, 0.0),
        upper=(0.1, 0.1, 0.1),
    )
    observation.vertices = [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]
    observation.poses = lambda: (torch.eye(4)[None], torch.eye(4)[None])
    target = torch.eye(4)[None]
    target[:, 2, 3] = 0.015
    before = target.clone()
    nominal = observation._entry_poses_for_target(target)
    lower = observation._entry_poses_for_target(target, release_lift=0.0)
    assert torch.equal(nominal[0], lower[0]) and torch.equal(nominal[1], lower[1])
    assert torch.equal(lower[2], lower[1])
    assert nominal[2][0, 2, 3] - lower[2][0, 2, 3] == pytest.approx(0.12)
    assert torch.equal(target, before)


@dataclass
class Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics


@pytest.mark.parametrize(
    "nominal_success,nominal_count,expected_lifts",
    [
        (True, 1, [0.12]),
        (False, 1, [0.12, 0.0]),
        (False, 2, [0.12]),
    ],
)
def test_drawer_release_fallback_preserves_nominal_and_total_full_plan_budget(
    monkeypatch,
    nominal_success,
    nominal_count,
    expected_lifts,
) -> None:
    generated, planned = [], []
    generator = NS(placement_path=lambda target: nullcontext())

    class Engine(drawer.DrawerPlacementEngine):
        def __init__(self):
            self._drawer_observations = (NS(route=NS(affordance="drawer")),)

        @property
        def robot(self):
            return object()

        @property
        def motion_generator(self):
            return generator

    def prepare(request, context, observations, *, release_lift=0.12):
        generated.append(release_lift)
        return tuple(
            NS(goal=NS(xpos=torch.full((1, 3, 4, 4), release_lift)))
            for _ in range(nominal_count)
        )

    def prefilter(request, context, candidates, robot):
        return tuple(enumerate(candidates)), []

    def build(engine, request, context, **kwargs):
        lift = float(request.goal.xpos[0, 0, 0, 0])
        planned.append(lift)
        return Plan(
            torch.tensor([nominal_success or lift == 0.0]),
            PlannerDiagnostics(backend="test"),
        )

    monkeypatch.setattr(drawer, "prepare_place_candidates", prepare)
    monkeypatch.setattr(drawer, "_prefilter_place_candidates", prefilter)
    monkeypatch.setattr(
        drawer, "_validate_drawer_plan_velocity", lambda plan, robot: plan
    )
    monkeypatch.setattr(drawer.GenSimActionEngine, "_plan_request", build)
    request = NS(goal=PlaceGoal(SceneEntityPose("drawer")))
    result = Engine()._plan_request(request, NS(batch_size=1))
    assert generated == expected_lifts
    assert len(planned) <= drawer.DRAWER_MAX_FULL_PLAN_CANDIDATES
    assert result.plan_success.tolist() == [nominal_success or 0.0 in expected_lifts]
    assert isinstance(request.goal.xpos, SceneEntityPose)
