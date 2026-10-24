# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations


from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from embodichain.gen_sim.task_engine._task_program import actions
from embodichain.gen_sim.task_engine._task_program.pick_closing_scene import (
    check_closing_scene,
)
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics


def test_closing_world_excludes_only_source_after_fixed_arm_check(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import articulation_collision

    source, table, other = object(), object(), object()
    world = SimpleNamespace(
        get_rigid_object_uid_list=lambda: ["physical_payload", "table", "other"],
        get_rigid_object=lambda uid: {
            "physical_payload": source,
            "table": table,
            "other": other,
        }[uid],
        get_articulation_uid_list=lambda: ["cabinet"],
        get_articulation=lambda uid: "native_cabinet",
    )
    path = torch.zeros(3, 4)
    path[:, 2] = torch.tensor([0.0, 0.35, 0.7])
    result = {"valid": False, "kind": "scene", "pairs": [["pad", "table:0"]]}

    def check(robot, snapshot, positions, part):
        assert snapshot.get_rigid_object_uid_list() == ["table", "other"]
        assert snapshot.get_rigid_object("table") is table
        assert snapshot.get_articulation_uid_list() == ["cabinet"]
        assert snapshot.get_articulation("cabinet") == "native_cabinet"
        assert torch.equal(positions, path) and part == "arm"
        return result

    monkeypatch.setattr(articulation_collision, "check_free_motion", check)
    assert (
        check_closing_scene(
            object(),
            world,
            path,
            "arm",
            (0, 1),
            "physical_payload",
            contact_fraction=1.0,
            grid_samples=3,
        )
        is result
    )
    assert world.get_rigid_object_uid_list() == ["physical_payload", "table", "other"]


@pytest.mark.parametrize("moving_arm", [False, True])
def test_closing_world_rejects_unresolved_source_or_moving_arm(monkeypatch, moving_arm):
    from embodichain.gen_sim.task_engine._task_program import articulation_collision

    check = Mock(side_effect=AssertionError("unsafe source exclusion"))
    monkeypatch.setattr(articulation_collision, "check_free_motion", check)
    world = SimpleNamespace(get_rigid_object_uid_list=lambda: ["source"])
    path = torch.zeros(3, 4)
    if moving_arm:
        path[-1, 0] = 0.01
    with pytest.raises(ValueError):
        check_closing_scene(
            object(),
            world,
            path,
            "arm",
            (0, 1),
            "source" if moving_arm else "semantic_alias",
            contact_fraction=1.0,
            grid_samples=3,
        )
    check.assert_not_called()


@dataclass
class Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics
    joint_trajectory: object
    segments: tuple


@pytest.mark.parametrize(
    "revision,purpose,expected",
    [
        (7, "ordinary", False),
        (8, "ordinary", False),
        (6, "ordinary", True),
        (7, "pour", True),
        (8, "pour", True),
        (7, "ends", True),
        (8, "ends", False),
    ],
)
def test_final_closure_cannot_bypass_revision_seven_scene_check(
    monkeypatch, revision, purpose, expected
):
    action = actions.GenSimPickUp()
    action.pick_purposes = {"pick": purpose}
    request = SimpleNamespace(
        invocation_id="pick",
        goal=SimpleNamespace(
            semantics=SimpleNamespace(
                affordance=SimpleNamespace(
                    get_custom_config=lambda key: "ends" if purpose == "ends" else None
                )
            )
        ),
    )
    screen = SimpleNamespace(revision=revision, report=lambda: {"revision": revision})
    selection = actions._PickSelection(request=request, closing_geometry=screen)
    plan = Plan(
        torch.tensor([True]),
        PlannerDiagnostics(backend="test"),
        SimpleNamespace(positions=torch.zeros(1, 4, 2)),
        (SimpleNamespace(name="close", start=1, stop=3),),
    )
    action.inspect_closing_plan = lambda *a, **kw: {"valid": True}
    checker = Mock(
        return_value={"valid": False, "kind": "scene", "pairs": [["pad", "table:0"]]}
    )
    action._inspect_closing_scene = checker
    token = actions._PICK_SELECTION.set(selection)
    try:
        result = action._qualify_closing_plan(request, object(), plan)
    finally:
        actions._PICK_SELECTION.reset(token)
    assert result.plan_success.tolist() == [expected]
    assert checker.call_count == (0 if expected else 1)
    if not expected:
        assert result.diagnostics.failure.code == "gen_sim_pick_contact_unqualified"
        assert (
            result.diagnostics.metadata["gen_sim_pick_contact_geometry"][
                "closing_scene"
            ]["valid"]
            is False
        )
