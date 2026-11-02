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

"""Drawer uprighting selects existing headings without sweeping an articulation."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    articulation_collision as collision,
)
from embodichain.gen_sim.task_engine._task_program import (
    held_alignment_recovery as alignment,
)
from embodichain.lab.sim.motion.planners.utils import PlanResult


@pytest.mark.parametrize("scene_clear", [None, False, True])
def test_alignment_reuses_heading_search_for_articulation_collision(
    monkeypatch, scene_clear
):
    robot = NS(device="cpu")
    context = NS(robot=NS(qpos=torch.zeros(1, 2)))
    scope = alignment.PlanningScope(
        robot, context, "arm", (0,), (1,), torch.tensor([[0.1]])
    )
    monkeypatch.setattr(alignment, "check_robot_motion", lambda *a: {"valid": True})
    checked = []
    if scene_clear is not None:

        def inspect(positions):
            checked.append(positions.clone())
            assert torch.equal(positions[:, 1], torch.full((3,), 0.1))
            return {"valid": scene_clear, "kind": "scene"}

        scope.articulation_checker = inspect
    original = PlanResult(
        success=torch.tensor([True]),
        positions=torch.zeros(1, 3, 1),
        dt=torch.tensor([[0.0, 0.04, 0.04]]),
    )
    token = alignment._SCOPE.set(scope)
    before = torch.random.get_rng_state()
    try:
        result = alignment.screen_alignment_result(
            NS(robot=robot), original, NS(control_part="arm")
        )
    finally:
        alignment._SCOPE.reset(token)
    assert bool(result.success.all()) is (scene_clear is not False)
    assert scope.rejected_raw_success == (1 if scene_clear is False else 0)
    assert len(checked) == (0 if scene_clear is None else 1)
    if scene_clear is not False:
        assert result is original
    assert torch.equal(before, torch.random.get_rng_state())


def test_articulation_checker_declares_its_limited_geometry_scope(monkeypatch):
    seen = []

    def build(robot, simulation, part, **kwargs):
        seen.append((robot, simulation, part, kwargs))
        return lambda positions: {"valid": False, "pairs": [["finger", "drawer"]]}

    monkeypatch.setattr(collision, "_make_checker", build)
    checker = collision.articulation_motion_checker("robot", "sim", "arm")
    assert seen == [
        ("robot", "sim", "arm", {"include_scene": True, "include_rigids": False})
    ]
    result = checker(torch.zeros(3, 2))
    assert result["valid"] is False
    assert result["scope"] == "discrete_robot_articulation_meshes"
