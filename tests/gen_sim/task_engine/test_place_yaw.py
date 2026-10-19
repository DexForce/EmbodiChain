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

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import place_yaw as yaw
from embodichain.gen_sim.task_engine._task_program._search_cost import (
    PlanningBudgetExceeded,
)
from embodichain.lab.sim.atomic_actions import PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal, PlaceOptions
from embodichain.lab.task_program.semantics import (
    Pick,
    Place,
    SceneAffordanceRef,
    SceneObjectRef,
)
from embodichain.utils.math import matrix_from_euler, pose_inv


def test_yaw_rotates_world_object_not_tcp_and_preserves_origin():
    obj, held = torch.eye(4)[None], torch.eye(4)[None]
    obj[0, :3, :3] = matrix_from_euler(torch.tensor([0.4, 0.2, 0.1]), "XYZ")
    obj[0, :3, 3] = torch.tensor([0.1, -0.2, 0.8])
    held[0, :3, :3] = matrix_from_euler(torch.tensor([0.2, -0.3, 0.5]), "XYZ")
    held[0, :3, 3] = torch.tensor([0.04, 0.02, 0.15])
    original = obj @ held
    changed, rotated = yaw.rotated_target(original, held, -math.pi / 4)
    world_yaw = matrix_from_euler(torch.tensor([0.0, 0.0, -math.pi / 4]), "XYZ")
    torch.testing.assert_close(rotated[:, :3, 3], obj[:, :3, 3])
    torch.testing.assert_close(rotated[:, :3, :3], world_yaw @ obj[:, :3, :3])
    torch.testing.assert_close(changed @ pose_inv(held), rotated)
    torch.testing.assert_close(rotated[:, 2], obj[:, 2])
    torch.testing.assert_close(original, obj @ held)


@pytest.mark.parametrize(
    "bad", [None, "place", {}, [True], [""], [" place"], ["p", "p"]]
)
def test_yaw_permission_requires_exact_unique_names(bad):
    with pytest.raises(ValueError, match="yaw"):
        yaw.declarations(bad)


def test_declared_yaw_calls_require_exact_inside_place():
    call = SimpleNamespace(
        call=Place(SceneObjectRef("item"), inside=SceneAffordanceRef("inside")),
        segment_call_index=0,
    )
    segment = SimpleNamespace(name="place", segment_id="s", calls=(call,))
    program = SimpleNamespace(program_id="p", iter_segments=lambda: iter((segment,)))
    assert yaw.bind_calls([], program) == ()
    assert yaw.bind_calls(["place"], program) == ("p/s:0",)
    with pytest.raises(ValueError, match="missing"):
        yaw.bind_calls(["absent"], program)
    call.call = Place(SceneObjectRef("item"), on=SceneObjectRef("table"))
    with pytest.raises(ValueError, match="inside Place"):
        yaw.bind_calls(["place"], program)


def test_factory_binds_only_declared_same_resource_rigid_route(monkeypatch):
    item, tray = SimpleNamespace(entity_id="item"), SimpleNamespace(entity_id="tray")
    container = SimpleNamespace(
        entity_id="inside", parent_id="tray", release_clearance=0.12
    )
    route = SimpleNamespace(
        pick_id="pick",
        place_id="place",
        pick_call=Pick(SceneObjectRef("item"), resources={"primary": "left"}),
        place_call=Place(
            SceneObjectRef("item"),
            inside=SceneAffordanceRef("inside"),
            resources={"primary": "left"},
        ),
    )
    factory = SimpleNamespace(
        _world_yaw_free_places=("place",),
        _pick_continuations=(route,),
        _pick_purposes=(("pick", "ordinary"),),
        _simulation=object(),
        task_program_registration=SimpleNamespace(
            scene_binding=SimpleNamespace(
                rigid_objects=(item, tray), containers=(container,)
            )
        ),
    )
    geometry = object()
    monkeypatch.setattr(yaw, "_prepare_placement_geometry", lambda *_: geometry)
    assert yaw.bind_routes(factory) == {
        "place": ("item", container, item, tray, geometry)
    }
    factory._world_yaw_free_places = ("missing",)
    with pytest.raises(ValueError, match="bound ordinary"):
        yaw.bind_routes(factory)
    factory._world_yaw_free_places = ("place",)
    factory._pick_purposes = (("pick", "pour"),)
    with pytest.raises(ValueError, match="bound ordinary"):
        yaw.bind_routes(factory)


@dataclass(frozen=True)
class Request:
    invocation_id: str
    goal: PlaceGoal
    skill_options: PlaceOptions
    binding: Any
    motion_policy: Any


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics


def fixture(monkeypatch):
    container = SimpleNamespace(
        entity_id="inside", parent_id="tray", release_clearance=0.12
    )
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="item"), object_to_eef=torch.eye(4)[None]
    )
    action = SimpleNamespace(
        world_yaw_free_routes={
            "place": ("item", container, object(), object(), object())
        },
        world_yaw_simulation=object(),
        robot=SimpleNamespace(device=torch.device("cpu")),
        motion_generator=SimpleNamespace(),
    )
    request = Request(
        "place",
        PlaceGoal(SceneEntityPose("inside")),
        PlaceOptions(lift_height=0.05, preserve_current_object_orientation=True),
        SimpleNamespace(endpoint=lambda *_: SimpleNamespace(task_state_key="held")),
        SimpleNamespace(strategy="ik_interp", dynamic_collision_mode="off"),
    )
    context = SimpleNamespace(
        batch_size=1, env_ids=torch.tensor([0]), get_held_object=lambda _: held
    )
    pose = torch.eye(4)[None]
    pose[0, 2, 3] = 0.85
    monkeypatch.setattr(
        yaw,
        "resolve_pose_goal",
        lambda *_a, **kw: (
            pose.clone() if kw["name"] == "yaw_nominal_release" else torch.eye(4)[None]
        ),
    )
    monkeypatch.setattr(
        yaw, "measure_planar_placement", lambda *_a, **_kw: ([True], {})
    )
    first = Plan(torch.tensor([False]), PlannerDiagnostics(backend="test"))
    return action, request, context, first


@pytest.mark.parametrize(
    "protected",
    [
        "success",
        "undeclared",
        "preview",
        "cartesian",
        "world_orientation",
        "collision_policy",
        "height_cap",
        "wrong_held",
    ],
)
def test_protected_requests_never_enter_yaw_search(monkeypatch, protected):
    action, request, context, first = fixture(monkeypatch)
    token = yaw._PREVIEW.set(protected == "preview")
    cartesian = yaw._CARTESIAN_APPROACH.set(protected == "cartesian")
    try:
        if protected == "success":
            first = replace(first, plan_success=torch.tensor([True]))
        elif protected == "undeclared":
            action.world_yaw_free_routes = None
        elif protected == "world_orientation":
            request = replace(
                request,
                goal=PlaceGoal(
                    SceneEntityPose("inside", world_orientation=torch.eye(3)[None])
                ),
            )
        elif protected == "collision_policy":
            request.motion_policy.dynamic_collision_mode = "replan"
        elif protected == "height_cap":
            request = replace(
                request,
                skill_options=replace(
                    request.skill_options, max_approach_retract_z=0.9
                ),
            )
        elif protected == "wrong_held":
            context.get_held_object = lambda _: None
        assert (
            yaw.recover(
                action,
                request,
                context,
                first,
                lambda *_: pytest.fail("Protected request searched"),
            )
            is first
        )
    finally:
        yaw._PREVIEW.reset(token)
        yaw._CARTESIAN_APPROACH.reset(cartesian)


def test_yaw_reuses_place_and_accepts_only_final_inspector(monkeypatch):
    action, request, context, first = fixture(monkeypatch)
    rng = torch.get_rng_state().clone()
    targets = []

    def build(selected, current):
        assert current is context
        assert selected.skill_options is request.skill_options
        targets.append(selected.goal.xpos.relative_pose.clone())
        torch.rand(1)
        return replace(first, plan_success=torch.tensor([True]))

    monkeypatch.setattr(yaw, "inspect_emitted", lambda *_: {"valid": len(targets) == 2})
    result = yaw.recover(action, request, context, first, build)
    receipt = result.diagnostics.metadata["gen_sim_inside_world_yaw"]
    assert result.plan_success.all() and receipt["accepted"]
    assert [r["angle_deg"] for r in receipt["attempts"]] == [45, -45]
    assert receipt["release_clearance_m"] == 0.12 and receipt["lift_height_m"] == 0.05
    assert receipt["physical_evidence"] is False
    assert torch.equal(rng, torch.get_rng_state())
    for target in targets:
        assert target[0, 2, 3] == pytest.approx(0.85)


def test_invalid_footprint_cannot_reach_planner(monkeypatch):
    action, request, context, first = fixture(monkeypatch)
    monkeypatch.setattr(
        yaw, "measure_planar_placement", lambda *_a, **_kw: ([False], {})
    )
    result = yaw.recover(
        action,
        request,
        context,
        first,
        lambda *_: pytest.fail("Invalid footprint planned"),
    )
    assert not result.plan_success.any()
    assert len(result.diagnostics.metadata["gen_sim_inside_world_yaw"]["attempts"]) == 7


def test_budget_failure_propagates_and_restores_rng(monkeypatch):
    action, request, context, first = fixture(monkeypatch)
    rng = torch.get_rng_state().clone()

    def expired(*_):
        torch.rand(1)
        raise PlanningBudgetExceeded("yaw", 60.0, 60.1)

    with pytest.raises(PlanningBudgetExceeded):
        yaw.recover(action, request, context, first, expired)
    assert torch.equal(rng, torch.get_rng_state())
