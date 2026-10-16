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

from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import trimesh

from embodichain.gen_sim.task_engine._task_program import pick_contact, release_contact
from embodichain.gen_sim.task_engine._task_program.pick_contact import (
    ClosingContactScreen,
    ContactPart,
)
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose
from embodichain.lab.task_program.semantics import (
    Pick,
    Place,
    RegisteredSemanticCall,
    SceneAffordanceRef,
    SceneObjectRef,
)


def _poses(axis, values):
    result = np.tile(np.eye(4), (len(values), 1, 1))
    result[:, axis, 3] = values
    return result


def _closing(revision=3):
    pad = trimesh.creation.box((0.004, 0.02, 0.02))
    parts = tuple(
        ContactPart(name, (pad,), _poses(0, values), 0.001, True, 0.006)
        for name, values in (
            ("negative_pad", [-0.02, -0.013, -0.01]),
            ("positive_pad", [0.02, 0.013, 0.01]),
        )
    )
    screen = ClosingContactScreen(
        parts,
        (trimesh.creation.box((0.02, 0.02, 0.02)),),
        0.001,
        revision=revision,
        target_contact_offset=0.003,
    )
    screen.identity = {"native_robot_path": "arena_0/robot"}
    return screen


def _opening(closing, *, height=0.025, support_pose=None):
    target = np.eye(4)
    target[2, 3] = height
    reference = np.eye(4) if support_pose is None else support_pose
    return release_contact.OpeningContactScreen(
        closing,
        (trimesh.creation.box((0.1, 0.1, 0.02)),),
        reference,
        reference @ target,
        0.001,
        0.003,
        identity={"reference_entity_id": "support"},
    )


def test_source_qualified_grasps_are_decided_by_cooked_mesh_not_axis_label():
    closing = _closing()
    opening = _opening(closing)
    side = np.eye(4)
    vertical = np.eye(4)
    vertical[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    assert closing.inspect(side)["valid"]
    assert closing.inspect(vertical)["valid"]
    assert opening.inspect(side, closing.inspect(side))["valid"]
    blocked = opening.inspect(vertical, closing.inspect(vertical))
    assert not blocked["valid"]
    assert blocked["reason"] == "hand_support_rest_envelope_overlap"
    assert blocked["link"] == "negative_pad"
    assert _opening(closing, height=0.1).inspect(vertical, closing.inspect(vertical))[
        "valid"
    ]
    assert not _opening(closing, height=0.012).inspect(side, closing.inspect(side))[
        "valid"
    ]


def test_contact_halo_is_recorded_but_never_used_as_rest_rejection():
    closing = _closing()
    result = _opening(closing).inspect(np.eye(4), closing.inspect(np.eye(4)))
    assert result["valid"]
    assert result["contact_halo_links"] == ["negative_pad", "positive_pad"]
    assert result["minimum_distance_m"] == pytest.approx(0.005)


def test_mask_keeps_source_records_rng_shape_and_candidate_order():
    closing = _closing()
    closing.opening_screen = _opening(closing)
    poses = torch.eye(4, dtype=torch.float64)[None, None].repeat(1, 2, 1, 1)
    poses[0, 1, :3, :3] = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    before, rng = poses.clone(), torch.random.get_rng_state().clone()
    assert closing.mask(poses, torch.eye(4)[None]).tolist() == [[True, False]]
    assert torch.equal(poses, before) and torch.equal(torch.random.get_rng_state(), rng)
    assert all(item["valid"] for item in closing.report()["reasons"])
    opening = closing.report()["opening"]
    assert opening["accepted_queries"] == opening["rejected_queries"] == 1
    assert opening["physical_evidence"] is False


@pytest.mark.parametrize("revision", [1, 2])
def test_old_revisions_never_consult_opening_or_change_report(revision):
    closing = _closing(revision)
    closing.opening_screen = Mock()
    assert closing.mask(torch.eye(4)[None, None], torch.eye(4)[None]).tolist() == [
        [True]
    ]
    closing.opening_screen.inspect.assert_not_called()
    assert "opening" not in closing.report()


def test_source_rejected_grasp_never_runs_opening():
    closing = _closing()
    closing.opening_screen = Mock()
    far = torch.eye(4)[None, None]
    far[:, :, 2, 3] = 1.0
    assert closing.mask(far, torch.eye(4)[None]).tolist() == [[False]]
    closing.opening_screen.inspect.assert_not_called()


def test_opening_uses_only_pre_bilateral_grid_and_returns_owned_reports():
    closing = _closing()
    part = ContactPart(
        "body",
        (trimesh.creation.box((0.01,) * 3),),
        _poses(2, [0.2, 0.2, -0.025]),
        0.001,
        False,
        0.003,
    )
    closing = ClosingContactScreen(
        (*closing.parts, part),
        (trimesh.creation.box((0.02,) * 3),),
        0.001,
        revision=3,
        target_contact_offset=0.003,
    )
    opening = _opening(closing)
    source = closing.inspect(np.eye(4))
    assert source["checked_samples"] == 2
    result = opening.inspect(np.eye(4), source)
    assert result["valid"] and result["checked_queries"] == 6
    result["contact_halo_links"].clear()
    assert opening.inspect(np.eye(4), source)["contact_halo_links"]


def _factory(monkeypatch, registered=False):
    closing = _closing()
    robot = NS(uid="robot")
    native_robot = NS(
        path="arena_0/robot", arena_name="arena_0", get_name=lambda: "robot"
    )
    body = NS(get_rest_offset=lambda: 0.001, get_contact_offset=lambda: 0.003)
    handle = NS(
        path="arena_0/support",
        arena_name="arena_0",
        get_name=lambda: "native_support",
        native=lambda: NS(get_physical_body=lambda: body),
    )
    handles = {native_robot.path: native_robot, handle.path: handle}
    simulation = NS(
        num_envs=1,
        is_default_backend=True,
        spawn_result=NS(get=handles.get, get_paths=lambda: tuple(handles)),
    )
    monkeypatch.setattr(
        pick_contact,
        "_native_meshes",
        lambda _: (trimesh.creation.box((0.1, 0.1, 0.02)),),
    )
    scene = NS(
        rigid_objects=(NS(entity_id="support", simulation_uid="native_support"),),
        support_surfaces=(NS(entity_id="on__support__item", parent_id="support"),),
    )
    pick = Pick(object=SceneObjectRef("item"), resources={"primary": "left"})
    place = (
        RegisteredSemanticCall(
            call_id="simulation.place_relative",
            arguments={"object": "item", "reference": "support", "relation": "on"},
            resources={"primary": "left"},
        )
        if registered
        else Place(
            object=SceneObjectRef("item"),
            on=SceneAffordanceRef("on__support__item"),
            resources={"primary": "left"},
        )
    )
    preview = NS(
        route=NS(
            pick_call=pick,
            place_call=place,
            pick_id="pick",
            place_id="place",
            transport_id="move",
        )
    )
    target = torch.eye(4)
    target[2, 3] = 0.025
    request = NS(
        invocation_id="pick",
        goal=NS(semantics=NS(entity_id="item")),
        skill_options=NS(
            downstream_object_target_poses=(
                torch.eye(4),
                SceneEntityPose("support", relative_pose=target),
            )
        ),
    )
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=NS(qpos=torch.zeros(1, 1)),
        scene=NS(entities={"support": NS(pose=torch.eye(4)[None], confidence=1.0)}),
    )
    return closing, robot, simulation, scene, request, context, preview


@pytest.mark.parametrize("registered", [False, True])
def test_factory_resolves_exact_live_reference_and_final_place_not_middle_move(
    monkeypatch, registered
):
    args = _factory(monkeypatch, registered)
    screen = release_contact.build_opening_screen(*args, purpose="ordinary")
    assert screen is not None
    assert screen.object_target_pose[2, 3] == pytest.approx(0.025)
    assert screen.identity["terminal_target_index"] == 1
    args[5].scene.entities["support"].pose[:, 0, 3] = 2.0
    fresh = release_contact.build_opening_screen(*args, purpose="ordinary")
    assert fresh.object_target_pose[0, 3] == 2.0
    assert screen.object_target_pose[0, 3] == 0.0


@pytest.mark.parametrize("purpose", [None, "pour", "ends"])
def test_nonordinary_and_standalone_are_not_owned(purpose):
    assert (
        release_contact.build_opening_screen(
            None, None, None, None, None, None, None, purpose=purpose
        )
        is None
    )


@pytest.mark.parametrize("mode", ["pour", "ends"])
def test_explicit_grasp_semantics_are_not_overridden_by_an_ordinary_marker(
    monkeypatch, mode
):
    args = _factory(monkeypatch)
    args[4].goal.semantics.affordance = NS(get_custom_config=lambda _: mode)
    assert release_contact.build_opening_screen(*args, purpose="ordinary") is None


def test_explicit_part_affordance_is_not_owned(monkeypatch):
    args = _factory(monkeypatch)
    target = "on__support__item__part_0"
    args[3].support_surfaces = (NS(entity_id=target, parent_id="support"),)
    args[-1].route.place_call = replace(
        args[-1].route.place_call, on=SceneAffordanceRef(target)
    )
    assert release_contact.build_opening_screen(*args, purpose="ordinary") is None


def test_wrong_arena_and_ambiguous_native_support_fail_closed(monkeypatch):
    args = _factory(monkeypatch)
    handle = args[2].spawn_result.get("arena_0/support")
    handle.arena_name = "arena_1"
    with pytest.raises(ValueError, match="native support"):
        release_contact.build_opening_screen(*args, purpose="ordinary")
    handle.arena_name = "arena_0"
    args[2].spawn_result.get_paths = lambda: ("arena_0/support", "arena_0/support")
    with pytest.raises(ValueError, match="native support"):
        release_contact.build_opening_screen(*args, purpose="ordinary")


def test_world_transform_preserves_native_geometric_decision():
    closing = _closing()
    world = np.eye(4)
    world[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    world[:3, 3] = (2.0, -3.0, 0.7)
    source = closing.inspect(np.eye(4))
    plain = _opening(closing).inspect(np.eye(4), source)
    transformed = _opening(closing, support_pose=world).inspect(np.eye(4), source)
    assert plain["valid"] == transformed["valid"]
    assert plain["minimum_distance_m"] == pytest.approx(
        transformed["minimum_distance_m"], abs=1e-7
    )


@pytest.mark.parametrize(
    "kind", ["inside", "part", "foreign_resource", "foreign_object"]
)
def test_other_continuation_contracts_remain_unowned(monkeypatch, kind):
    args = _factory(monkeypatch, registered=kind == "part")
    preview = args[-1]
    if kind == "inside":
        preview.route.place_call = Place(
            object=SceneObjectRef("item"),
            inside=SceneObjectRef("support"),
            resources={"primary": "left"},
        )
    elif kind == "part":
        call = preview.route.place_call
        preview.route.place_call = replace(
            call, arguments={**call.arguments, "reference_part": "part_0"}
        )
    elif kind == "foreign_resource":
        preview.route.place_call = replace(
            preview.route.place_call, resources={"primary": "right"}
        )
    else:
        preview.route.place_call = replace(
            preview.route.place_call, object=SceneObjectRef("other")
        )
    assert release_contact.build_opening_screen(*args, purpose="ordinary") is None


@pytest.mark.parametrize(
    "damage",
    ["reference", "affordance", "targets", "pose", "native", "offset", "context"],
)
def test_owned_missing_evidence_fails_closed(monkeypatch, damage):
    args = list(_factory(monkeypatch))
    if damage == "reference":
        args[3].rigid_objects = ()
    elif damage == "affordance":
        args[3].support_surfaces = ()
    elif damage == "targets":
        args[4].skill_options.downstream_object_target_poses = ()
    elif damage == "pose":
        args[5].scene.entities["support"].pose[0, 0, 0] = float("nan")
    elif damage == "native":
        args[2].spawn_result.get_paths = lambda: ()
    elif damage == "offset":
        handle = args[2].spawn_result.get("arena_0/support")
        handle.native().get_physical_body().get_rest_offset = lambda: float("nan")
    else:
        args[5] = None
    with pytest.raises((ValueError, KeyError)):
        release_contact.build_opening_screen(*args, purpose="ordinary")
