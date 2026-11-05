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

"""Semantic support reservations retain source ordering and release meaning."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Polygon, box
import trimesh
from embodichain.gen_sim.task_engine import placement_declarations as bundle
from embodichain.lab.sim.scene.source_geometry import read_source_vertices


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
    routes = bundle.resolve_relative_placement_routes(graph, scene)
    footprints = []
    for route in routes:
        subject = next(
            o for o in scene.planner_objects if o["runtime_uid"] == route["object_id"]
        )
        position = np.asarray(route["world_displacement"]) + [0.12, 0.0, 0.8]
        footprint = MultiPoint(read_source_vertices(subject)[:, :2]).convex_hull
        placed = translate(footprint, *position[:2])
        assert box(-0.10, -0.15, 0.34, 0.15).buffer(1e-8).covers(placed)
        footprints.append(placed)
    assert routes[0]["world_displacement"][:2] == [0.0, 0.0]
    assert footprints[0].distance(footprints[1]) >= 0.02
    assert routes == bundle.resolve_relative_placement_routes(
        graph, scene, settled=True
    )
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
    routes = bundle.resolve_relative_placement_routes(graph, scene)
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
        for route in bundle.resolve_relative_placement_routes({"nodes": nodes}, scene)
    )


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
    routes = bundle.resolve_relative_placement_routes({"nodes": nodes}, scene)
    reference_z = (
        0.80 if preceding and not container_moved and reacquire is None else 1.2
    )
    assert routes[0]["world_displacement"][2] == pytest.approx(
        0.75 - reference_z + 0.03
    )
    settled = bundle.resolve_relative_placement_routes(
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
        monkeypatch.setattr(bundle, "planar_floor_region", lambda mesh: None)
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
    release = bundle.resolve_relative_placement_routes({"nodes": [node]}, scene)[0]
    settled = bundle.resolve_relative_placement_routes(
        {"nodes": [node]}, scene, settled=True
    )[0]
    assert release["world_displacement"] == pytest.approx([0.1, 0.2, release_z])
    assert settled["world_displacement"] == [0.1, 0.2, 0.02]
    assert {k: v for k, v in release.items() if k != "world_displacement"} == {
        k: v for k, v in settled.items() if k != "world_displacement"
    }


def test_repeated_object_routes_keep_stage_identity_and_reject_conflicting_alias(
    tmp_path,
):
    scene = _scene(tmp_path)
    graph = {
        "nodes": [
            _on("utensil", "z"),
            {"call": {"kind": "pick", "object": "utensil"}},
            _on("utensil", "a"),
        ]
    }
    routes = bundle.resolve_relative_placement_routes(graph, scene)
    assert [route["stage_id"] for route in routes] == ["a", "z"]
    assert routes[0]["world_displacement"] == routes[1]["world_displacement"]
    graph["nodes"] = [
        _on("utensil", "same"),
        {
            "call": {
                "kind": "registered",
                "call_id": "simulation.axis_align",
                "arguments": {"object": "utensil"},
            }
        },
        _on("utensil", "same"),
    ]
    with pytest.raises(ValueError, match="Repeated placement"):
        bundle.resolve_relative_placement_routes(graph, scene)


def test_stability_conditions_use_explicit_owned_sensor_declarations(tmp_path):
    from embodichain.gen_sim.task_engine.stability_declarations import (
        resolve_stability_conditions,
    )
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationStabilityConditionCfg,
        SupportSensorDeclarationCfg,
    )

    scene = _scene(tmp_path)
    graph = {
        "nodes": [_on("dish", "release")],
        "task_groups": [{"node_ids": ["release"]}],
    }
    key = "gen_sim.release.stable"
    with pytest.raises(ValueError, match="missing=.*gen_sim.release.stable"):
        resolve_stability_conditions(graph, scene, {})
    sensor = SupportSensorDeclarationCfg(
        uid="explicit_support", max_contacts_per_env=512
    )
    result = resolve_stability_conditions(
        graph, scene, {}, support_sensors={key: sensor}
    )
    condition = result[key]
    assert type(condition) is SimulationStabilityConditionCfg
    assert condition.kind == "supported_placement"
    assert condition.entity == "dish" and condition.reference == "pad"
    assert condition.position_tolerance == 0.05 and condition.duration == 3.0
    assert condition.timeout == 12.0 and condition.support_tolerance == 0.01
    assert condition.displacement == pytest.approx((0.0, 0.0, 0.0485), abs=1e-7)
    assert condition.support_contract_revision is None
    sensor.uid = "changed_after_binding"
    assert condition.support_sensor.uid == "explicit_support"
    assert "support_sensor" not in condition.to_condition_dict()
    strong = resolve_stability_conditions(
        graph, scene, {}, support_contract_revision=1, support_sensors={key: sensor}
    )
    assert strong[key].support_contract_revision == 1
    assert strong[key].displacement == condition.displacement
    with pytest.raises(ValueError, match="unused=.*extra"):
        resolve_stability_conditions(
            graph, scene, {}, support_sensors={key: sensor, "extra": sensor}
        )


def test_coordinated_stability_accumulates_declared_goals_with_custom_control_parts(
    tmp_path,
):
    from embodichain.gen_sim.task_engine.stability_declarations import (
        resolve_stability_conditions,
    )

    scene = _scene(tmp_path)
    nodes = [
        {
            "id": name,
            "task_instance_id": "carry",
            "task_type": "E5",
            "call": {
                "kind": "registered",
                "call_id": "simulation.coordinated_hold",
                "arguments": {"object": "dish", "world_displacement": delta},
                "resources": {"left": "resource_a", "right": "resource_b"},
            },
        }
        for name, delta in [("first", [0.14, 0.0, 0.0]), ("second", [0.0, -0.1, 0.02])]
    ]
    graph = {"nodes": nodes, "task_groups": [{"node_ids": ["first", "second"]}]}
    conditions = resolve_stability_conditions(
        graph, scene, {"resource_a": "custom_motion_a", "resource_b": "custom_motion_b"}
    )
    first, second = (
        conditions[f"gen_sim.{name}.stable"] for name in ("first", "second")
    )
    assert first.kind == second.kind == "hold"
    assert first.target_position == pytest.approx((0.40, -0.38, 0.84))
    assert second.target_position == pytest.approx((0.40, -0.48, 0.86))
    assert (
        first.motion_parts
        == second.motion_parts
        == ("custom_motion_a", "custom_motion_b")
    )
    assert first.support_sensor is second.support_sensor is None
    assert scene.planner_objects[0]["init_pos"] == [0.26, -0.38, 0.84]


@pytest.fixture(autouse=True)
def forbid_runtime_construction(monkeypatch):
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.motion.solvers import BaseSolver

    def forbidden(*args, **kwargs):
        pytest.fail("placement declarations constructed a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(BaseSolver, "__init__", forbidden)
