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

"""Shared rigid-support reservations preserve targets and existing free space."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Polygon, box
import trimesh

from embodichain.gen_sim.task_engine import task_program_bundle as bundle
from embodichain.gen_sim.task_engine._task_program.placement_proposals import (
    generate_on_targets,
)


def _mesh_object(
    tmp_path: Path, uid: str, size: list[float], position: list[float]
) -> dict:
    mesh = trimesh.creation.box(extents=size)
    mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, 1, -1]
    path = tmp_path / f"{uid}.glb"
    mesh.export(path)
    return {
        "runtime_uid": uid,
        "uid": uid,
        "role": "rigid_object",
        "shape": {"shape_type": "Mesh", "fpath": str(path)},
        "init_pos": position,
        "init_rot": [0.0, 0.0, 0.0],
        "body_scale": [1.0] * 3,
    }


def _scene(tmp_path: Path) -> SimpleNamespace:
    objects = [
        _mesh_object(tmp_path, "dish", [0.162, 0.162, 0.075], [0.26, -0.38, 0.84]),
        _mesh_object(tmp_path, "utensil", [0.192, 0.046, 0.022], [0.26, 0.38, 0.81]),
        _mesh_object(tmp_path, "pad", [0.44, 0.30, 0.002], [0.12, 0.0, 0.8]),
    ]
    return SimpleNamespace(planner_objects=objects, table_top_z=0.79)


def _on(subject: str, stage: str) -> dict:
    return {
        "id": stage,
        "task_instance_id": stage,
        "task_type": "E1",
        "call": {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {"object": subject, "reference": "pad", "relation": "on"},
        },
    }


def test_sequential_landings_reserve_space_on_the_same_support(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    original = deepcopy(scene.planner_objects)
    graph = {"nodes": [_on("dish", "a"), _on("utensil", "b")]}
    routes = bundle._relative_place_route_payloads(graph, scene)
    footprints = []
    for route in routes:
        subject = next(
            o for o in scene.planner_objects if o["runtime_uid"] == route["object_id"]
        )
        position = np.asarray(route["world_displacement"]) + [0.12, 0.0, 0.8]
        footprint = MultiPoint(bundle._mesh_vertices(subject)[:, :2]).convex_hull
        placed = translate(footprint, *position[:2])
        assert box(-0.10, -0.15, 0.34, 0.15).buffer(1e-8).covers(placed)
        footprints.append(placed)
    assert routes[0]["world_displacement"][:2] == [0.0, 0.0]
    assert footprints[0].distance(footprints[1]) >= 0.02
    assert routes == bundle._relative_place_route_payloads(graph, scene, settled=True)
    assert scene.planner_objects == original


@pytest.mark.parametrize(
    "kind,call_id,subject,stationary",
    [
        ("pick", "", "pad", False),
        ("hand_over", "", "pad", False),
        ("registered", "simulation.coordinated_hold", "pad", False),
        ("registered", "simulation.coordinated_transport", "pad", False),
        ("registered", "simulation.move_held_object", "pad", False),
        ("pick", "", "dish", True),
        ("registered", "simulation.park", None, True),
    ],
)
def test_only_stationary_supports_receive_shared_footprint_reservations(
    tmp_path: Path, kind: str, call_id: str, subject: str | None, stationary: bool
) -> None:
    scene = _scene(tmp_path)
    if kind == "registered":
        call = {
            "kind": kind,
            "call_id": call_id,
            "arguments": {"object": subject} if subject is not None else {},
        }
    else:
        call = {"kind": kind, "object": subject}
    graph = {"nodes": [_on("dish", "a"), _on("utensil", "b"), {"call": call}]}
    routes = bundle._relative_place_route_payloads(graph, scene)
    assert len(routes) == 2
    relocated = any(route["world_displacement"][:2] != [0.0, 0.0] for route in routes)
    assert relocated is stationary


@pytest.mark.parametrize("repick", [False, True])
def test_unoccupied_targets_remain_unchanged(tmp_path: Path, repick: bool) -> None:
    scene = _scene(tmp_path)
    nodes = [_on("utensil", "b")]
    if repick:
        nodes = [_on("dish", "a"), {"call": {"kind": "pick", "object": "dish"}}, *nodes]
    assert all(
        route["world_displacement"][:2] == [0.0, 0.0]
        for route in bundle._relative_place_route_payloads({"nodes": nodes}, scene)
    )


@pytest.mark.parametrize("occupied_size, expected", [(0.162, True), (0.44, False)])
def test_proposals_reject_overlap_without_leaving_support(
    occupied_size: float, expected: bool
) -> None:
    subject = trimesh.creation.box(extents=(0.192, 0.046, 0.022))
    support = trimesh.creation.box(extents=(0.44, 0.30, 0.002))
    support.apply_translation((0.0, 0.0, 0.8))
    occupied = box(
        -occupied_size / 2, -occupied_size / 2, occupied_size / 2, occupied_size / 2
    )
    proposals = generate_on_targets(
        subject,
        support,
        [np.eye(3)],
        (0.26, 0.38),
        clearance_m=0.01,
        contact_band_m=0.02,
        occupied_footprints=[occupied],
        separation_m=0.02,
    )
    assert bool(proposals) == expected
    for proposal in proposals:
        footprint = Polygon(proposal["footprint"])
        assert box(-0.22, -0.15, 0.22, 0.15).covers(footprint)
        assert footprint.distance(occupied) >= 0.02


@pytest.mark.parametrize(
    "upright, explicit, task_type",
    [
        (False, False, "E1"),
        (True, False, "E1"),
        (False, True, "E1"),
        (False, False, "E3"),
    ],
)
def test_native_depth_permission_is_local_to_horizontal_slender_on_pick(
    tmp_path: Path, upright: bool, explicit: bool, task_type: str
) -> None:
    scene = _scene(tmp_path)
    objects = {item["runtime_uid"]: item for item in scene.planner_objects}
    objects["utensil"]["init_rot"] = [0.0, 90.0 if upright else 0.0, 0.0]
    pick = {
        "id": "acquire",
        "task_type": task_type,
        "call": {
            "kind": "pick",
            "object": "utensil",
            "resources": {"primary": "right"},
        },
    }
    if explicit:
        pick["call"]["grasp"] = "authored_grasp"
    place = _on("utensil", "release")
    place["call"]["resources"] = {"primary": "right"}
    expected = (
        {"acquire": 8} if not upright and not explicit and task_type == "E1" else {}
    )
    assert (
        bundle._horizontal_on_pick_revisions({"nodes": [pick, place]}, objects)
        == expected
    )


def test_inside_release_uses_bounded_placement_clearance(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    scene.articulations = ()
    scene.planner_objects.append(
        {
            **_mesh_object(tmp_path, "table", [1.0, 1.0, 0.05], [0.0, 0.0, 0.765]),
            "role": "background",
        }
    )
    graph = {
        "targets": {},
        "nodes": [
            {
                "id": "acquire",
                "task_type": "E1",
                "task_instance_id": "transfer",
                "role": "primary",
                "depends_on": [],
                "call": {
                    "kind": "pick",
                    "object": "dish",
                    "resources": {"primary": "right"},
                },
            },
            {
                "id": "release",
                "task_type": "E1",
                "task_instance_id": "transfer",
                "role": "primary",
                "depends_on": ["acquire"],
                "call": {
                    "kind": "place",
                    "object": "dish",
                    "inside": "inside__pad__dish",
                    "resources": {"primary": "right"},
                },
            },
        ],
        "task_groups": [
            {"id": "transfer", "task_type": "E1", "node_ids": ["acquire", "release"]}
        ],
    }
    embodiment = {
        "simulation": {
            "init_pos": [0.0, 0.0, 0.0],
            "init_rot": [0.0, 0.0, 0.0],
            "urdf_cfg": {
                "components": [
                    {"component_type": "right_arm", "transform": np.eye(4).tolist()}
                ]
            },
        }
    }
    payload = bundle._integration_payload(
        graph, scene, program_id="test", scene_contract="test", embodiment=embodiment
    )
    containers = [
        a
        for item in payload["scene_binding"]["rigid_objects"]
        for a in item.get("affordances", [])
        if a["kind"] == "container"
    ]
    assert len(containers) == 1
    assert containers[0]["release_clearance"] == 0.12


@pytest.mark.parametrize("preceding", [False, True])
@pytest.mark.parametrize("container_moved", [False, True])
@pytest.mark.parametrize(
    "reacquire", [None, "pick", "source", "horizontal_source", "hand_over"]
)
def test_lateral_reference_tracks_preceding_inside_landing(
    monkeypatch: pytest.MonkeyPatch,
    preceding: bool,
    container_moved: bool,
    reacquire: str | None,
) -> None:
    objects = [
        {"runtime_uid": "item", "role": "rigid_object", "init_pos": [0, 0, 0.75]},
        {"runtime_uid": "anchor", "role": "rigid_object", "init_pos": [0, 0, 1.2]},
    ]
    scene = SimpleNamespace(planner_objects=objects, table_top_z=0.75)
    inside = {
        "id": "inside",
        "task_type": "E1",
        "call": {"kind": "place", "object": "anchor", "inside": "inside__tray__anchor"},
    }
    side = {
        "id": "side",
        "task_type": "E1",
        "task_instance_id": "side",
        "call": {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {
                "object": "item",
                "reference": "anchor",
                "relation": "right_of",
            },
        },
    }
    monkeypatch.setattr(
        bundle, "_inside_reference_height", lambda *args: 0.80, raising=False
    )
    monkeypatch.setattr(
        bundle, "_vertical_mesh_bounds", lambda *args, **kwargs: (0, 0.1)
    )
    monkeypatch.setattr(
        bundle, "_horizontal_relation_distance", lambda *args, **kwargs: 0.2
    )
    nodes = [inside, side] if preceding else [side, inside]
    if preceding and reacquire is not None:
        if reacquire in {"pick", "hand_over"}:
            call = {"kind": reacquire, "object": "anchor"}
        else:
            call = {
                "kind": "registered",
                "call_id": (
                    "gen_sim.pick.handover_source"
                    if reacquire == "source"
                    else "gen_sim.pick.handover_horizontal_source.anchor"
                ),
                "arguments": {"object": "anchor"},
            }
        nodes.insert(1, {"id": "reacquire", "task_type": "E4", "call": call})
    if container_moved:
        nodes.insert(0, {"id": "move_tray", "call": {"kind": "pick", "object": "tray"}})
    routes = bundle._relative_place_route_payloads({"nodes": nodes}, scene)
    reference_z = (
        0.80 if preceding and not container_moved and reacquire is None else 1.2
    )
    assert routes[0]["world_displacement"][2] == pytest.approx(
        0.75 - reference_z + 0.03
    )
    settled = bundle._relative_place_route_payloads(
        {"nodes": nodes}, scene, settled=True
    )
    assert settled[0]["world_displacement"][2] == pytest.approx(
        0.75 - reference_z + bundle._PLACEMENT_CLEARANCE
    )
    assert routes[0]["world_displacement"][:2] == settled[0]["world_displacement"][:2]
    assert objects[1]["init_pos"][2] == 1.2


@pytest.mark.parametrize("unsupported", [None, "tilted", "articulation", "unmeasured"])
def test_inside_height_uses_existing_measured_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsupported: str | None
) -> None:
    mesh = trimesh.Trimesh(
        vertices=[[-1, -1, 0.02], [1, -1, 0.02], [1, 1, 0.02], [-1, 1, 0.02]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    objects = {
        "tray": {
            "role": "rigid_object",
            "init_pos": [0, 0, 0.7],
            "shape": {"fpath": "unused.glb"},
        },
        "anchor": {"role": "rigid_object"},
    }
    monkeypatch.setattr(bundle, "_rigid_mesh_zup", lambda *args: mesh.copy())
    monkeypatch.setattr(
        bundle, "_vertical_mesh_bounds", lambda *args, **kwargs: (-0.01, 0.1)
    )
    if unsupported == "tilted":
        objects["tray"]["init_rot"] = [10, 0, 0]
    elif unsupported == "articulation":
        objects["tray"]["role"] = "articulation"
    elif unsupported == "unmeasured":
        from embodichain.gen_sim.task_engine._task_program import container_targets

        monkeypatch.setattr(container_targets, "_floor_region", lambda mesh: None)
    result = bundle._inside_reference_height(
        {"inside": "inside__tray__anchor"}, objects
    )
    assert result == (pytest.approx(0.731) if unsupported is None else None)


@pytest.mark.parametrize(
    "task_type,relation,reference_role,release_z",
    [
        ("E1", "right_of", "rigid_object", 0.04),
        ("E1", "left_of", "rigid_object", 0.04),
        ("E1", "front_of", "rigid_object", 0.04),
        ("E1", "behind", "rigid_object", 0.04),
        ("E4", "right_of", "rigid_object", 0.02),
        ("E1", "on", "rigid_object", 0.02),
        ("E1", "above", "rigid_object", 0.02),
        ("E1", "front_right_of", "rigid_object", 0.02),
        ("E1", "right_of", "articulation", 0.02),
    ],
)
def test_lateral_release_reserve_changes_no_xy_yaw_or_settled_target(
    monkeypatch, task_type, relation, reference_role, release_z
) -> None:
    scene = SimpleNamespace(
        planner_objects=[
            {"runtime_uid": "item", "role": "rigid_object"},
            {"runtime_uid": "anchor", "role": reference_role},
        ],
        table_top_z=0.7,
    )
    node = {
        "id": "side",
        "task_instance_id": "stage",
        "task_type": task_type,
        "call": {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {
                "object": "item",
                "reference": "anchor",
                "relation": relation,
            },
        },
    }
    monkeypatch.setattr(
        bundle, "_relative_world_displacement", lambda *args, **kwargs: [0.1, 0.2, 0.02]
    )
    release = bundle._relative_place_route_payloads({"nodes": [node]}, scene)[0]
    settled = bundle._relative_place_route_payloads(
        {"nodes": [node]}, scene, settled=True
    )[0]
    assert release["world_displacement"] == pytest.approx([0.1, 0.2, release_z])
    assert settled["world_displacement"] == [0.1, 0.2, 0.02]
    assert {k: v for k, v in release.items() if k != "world_displacement"} == {
        k: v for k, v in settled.items() if k != "world_displacement"
    }
