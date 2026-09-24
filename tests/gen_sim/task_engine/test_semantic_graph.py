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

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
from types import SimpleNamespace

import trimesh
from embodichain.gen_sim.task_engine._task_program.configured import (
    decode_task_lowerer,
)

import pytest

import embodichain.gen_sim.task_engine.task_program_bundle as task_program_bundle
from embodichain.gen_sim.task_engine._bundle_runner import (
    _semantic_success_by_env,
    _verify_program_projection,
)
from embodichain.gen_sim.task_engine.semantic_graph import (
    semantic_task_graph_hash,
    validate_semantic_task_graph,
)
from embodichain.gen_sim.task_engine.task_program_bundle import (
    _bind_embodiment_to_scene,
    _program_node,
    generate_task_program_bundle,
    _integration_payload,
    _program_payload,
    _refine_upright_targets,
    _scene_payload,
    _support_target_pose,
    _task_settle_rigid_objects,
)
from embodichain.gen_sim.task_engine.semantic_planner import _initial_quaternion_xyzw
from embodichain.utils.utility import load_config, save_config
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene


@pytest.mark.parametrize("height", [0.12, 0.25])
def test_pour_target_uses_upright_geometry_not_receiver_yaw(
    monkeypatch: pytest.MonkeyPatch,
    height: float,
) -> None:
    source = {"runtime_uid": "vessel", "init_rot": [0.0, 0.0, 30.0]}
    receiver = {"runtime_uid": "receiver", "init_rot": [0.0, 0.0, 0.0]}
    vertices = {
        "vessel": np.array([[-0.03, -0.03, 0.0], [0.03, 0.03, height]]),
        "receiver": np.array([[-0.05, -0.05, 0.0], [0.05, 0.05, 0.15]]),
    }
    monkeypatch.setattr(
        task_program_bundle, "_mesh_vertices", lambda obj: vertices[obj["runtime_uid"]]
    )
    model = {
        "max_opening_width": 0.128,
        "finger_thickness": 0.01,
        "finger_width": 0.02,
        "finger_length": 0.13,
    }
    embodiment = {
        "skill_profile": {
            "runtime_services": {
                "grasp_pose_generators": {"left_eef": {"model": model}}
            }
        }
    }
    first, semantics = task_program_bundle._pour_target_geometry(
        source, receiver, embodiment, "left"
    )
    receiver["init_rot"] = [0.0, 0.0, 125.0]
    second, _ = task_program_bundle._pour_target_geometry(
        source, receiver, embodiment, "left"
    )
    assert first["relative_pose"] == pytest.approx(second["relative_pose"])
    assert first["world_orientation"] == pytest.approx(second["world_orientation"])
    assert semantics["upright_axis"] == [0.0, 0.0, 1.0]
    assert semantics["pivot_local"] == pytest.approx([0.0, 0.0, height / 2])
    rotation = np.asarray(first["world_orientation"]).reshape(3, 3)
    relative = np.asarray(first["relative_pose"]).reshape(4, 4)
    pivot_height = (rotation @ (relative[:3, 3] + semantics["pivot_local"]))[2]
    radius = np.linalg.norm(vertices["vessel"] - semantics["pivot_local"], axis=1).max()
    assert pivot_height - radius > 0.15 + 0.015


def _graph() -> dict:
    return {
        "schema_version": "semantic_task_graph/v1",
        "task_id": "place_cube",
        "instruction": "Place the cube",
        "planner_route": "offline",
        "integration_fingerprint": "0" * 64,
        "targets": {},
        "nodes": [
            {
                "id": "pick_cube",
                "call": {"kind": "pick", "object": "cube"},
                "depends_on": [],
                "task_instance_id": "pick_group",
                "task_type": "E1",
                "role": "primary",
            },
            {
                "id": "place_cube",
                "call": {
                    "kind": "place",
                    "object": "cube",
                    "inside": "tray_inside",
                },
                "depends_on": ["pick_cube"],
                "task_instance_id": "place_group",
                "task_type": "E1",
                "role": "primary",
            },
        ],
        "task_groups": [
            {
                "id": "pick_group",
                "task_type": "E1",
                "node_ids": ["pick_cube"],
                "depends_on": [],
                "success": {"kind": "call_completed"},
            },
            {
                "id": "place_group",
                "task_type": "E1",
                "node_ids": ["place_cube"],
                "depends_on": ["pick_group"],
                "success": {"kind": "object_inside", "object": "cube"},
            },
        ],
        "success": {"kind": "all_task_groups"},
    }


def _fresh_handover_graph() -> dict:
    graph = _graph()
    graph["nodes"][0]["call"] = {
        "kind": "registered",
        "call_id": "gen_sim.pick.handover_source",
        "arguments": {"object": "cube", "target": "source_grasp"},
        "resources": {"primary": "left"},
    }
    graph["nodes"][1]["call"] = {
        "kind": "hand_over",
        "object": "cube",
        "resources": {"source": "left", "destination": "right"},
    }
    for node in graph["nodes"]:
        node["task_type"] = "E4"
        node["task_instance_id"] = "exchange"
    graph["task_groups"] = [
        {
            "id": "exchange",
            "task_type": "E4",
            "node_ids": [node["id"] for node in graph["nodes"]],
            "depends_on": [],
            "success": {"kind": "call_completed"},
        }
    ]
    return graph


def test_handover_staging_preserves_dependencies_and_is_idempotent(monkeypatch) -> None:
    from copy import deepcopy
    from embodichain.gen_sim.task_engine import task_program_bundle as bundle

    graph = _fresh_handover_graph()
    original = deepcopy(graph)
    scene = SimpleNamespace(
        planner_objects=[
            {
                "runtime_uid": "cube",
                "init_pos": [2.4, 3.2, 0.7],
                "init_rot": [0.0, 0.0, 90.0],
            }
        ],
        background=[
            {"uid": "table", "init_pos": [2.0, 3.0, 0.0], "init_rot": [0.0, 0.0, 0.0]}
        ],
    )
    monkeypatch.setattr(
        bundle,
        "_mesh_vertices",
        lambda table: np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]),
    )
    result = bundle._add_handover_staging(graph, scene)
    stage = result["nodes"][1]
    pose = result["targets"][stage["call"]["arguments"]["target"]]["values"][0]
    assert pose["position"] == pytest.approx([2.2, 3.1, 0.86])
    assert pose["quaternion_xyzw"] == pytest.approx([0.0, 0.0, 2**-0.5, 2**-0.5])
    assert stage["depends_on"] == [graph["nodes"][0]["id"]]
    assert result["nodes"][2]["depends_on"] == [stage["id"]]
    assert result["task_groups"][0]["node_ids"] == [
        node["id"] for node in result["nodes"]
    ]
    assert result["success"] == original["success"]
    assert graph == original
    assert bundle._add_handover_staging(result, None) == result


def test_handover_staging_does_not_predict_cross_task_holds() -> None:
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _add_handover_staging,
    )

    graph = _fresh_handover_graph()
    pick = graph["nodes"][0]
    pick["task_instance_id"] = "earlier_pick"
    graph["task_groups"][0]["node_ids"].remove(pick["id"])
    graph["task_groups"][0]["depends_on"] = ["earlier_pick"]
    graph["task_groups"].insert(
        0,
        {
            "id": "earlier_pick",
            "task_type": "E4",
            "node_ids": [pick["id"]],
            "depends_on": [],
            "success": {"kind": "call_completed"},
        },
    )
    assert _add_handover_staging(graph, None) == graph


def test_e3_return_pose_preserves_scene_rotation() -> None:
    quaternion = _initial_quaternion_xyzw([90.0, 0.0, 0.0])
    assert quaternion == pytest.approx([2**-0.5, 0.0, 0.0, 2**-0.5], abs=1.0e-6)


def _program(graph: dict) -> dict:
    return {
        "program_id": "place_cube",
        "targets": deepcopy(graph["targets"]),
        "program": {
            "kind": "sequence",
            "items": [
                {
                    "kind": "segment",
                    "name": node["id"],
                    "steps": {"kind": "invoke", "call": deepcopy(node["call"])},
                }
                for node in graph["nodes"]
            ],
        },
    }


@pytest.mark.parametrize(
    "relation", ["right_of", "left_of", "front_of", "behind", "front_left_of"]
)
def test_tabletop_region_cannot_be_lowered_outside_the_table(relation: str) -> None:
    graph = _graph()
    graph["nodes"] = [
        {
            "call": {
                "kind": "registered",
                "call_id": "simulation.place_relative",
                "arguments": {
                    "object": "cube",
                    "reference": "table",
                    "relation": relation,
                },
            }
        }
    ]
    scene = SimpleNamespace(planner_objects=())
    with pytest.raises(ValueError, match="named relation anchor"):
        task_program_bundle._relative_place_route_payloads(graph, scene)


@pytest.mark.parametrize("table_height", [0.5, 0.9])
def test_handover_minimum_height_preserves_retreat_headroom(
    table_height: float,
) -> None:
    assert task_program_bundle._handover_position_z(table_height) == pytest.approx(
        table_height + 0.10
    )


def test_relative_place_geometry_uses_only_preceding_alignment(monkeypatch) -> None:
    place = {
        "id": "place",
        "task_type": "E1",
        "call": {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {"object": "cup", "reference": "mat", "relation": "on"},
        },
    }
    align = {
        "id": "align",
        "task_type": "E2",
        "call": {
            "kind": "registered",
            "call_id": task_program_bundle._MOVE_HELD_OBJECT_CALL_ID,
            "arguments": {"object": "cup"},
        },
    }
    scene = SimpleNamespace(planner_objects=(), table_top_z=0.7)
    monkeypatch.setattr(
        task_program_bundle,
        "_relative_world_displacement",
        lambda *a, axis_align_objects, **kw: [
            0.0,
            0.0,
            float("cup" in axis_align_objects),
        ],
    )
    routes = task_program_bundle._relative_place_route_payloads
    assert routes({"nodes": [place, align]}, scene) == routes({"nodes": [place]}, scene)
    assert routes({"nodes": [align, place]}, scene)[0]["world_displacement"][2] == 1.0
    with pytest.raises(ValueError, match="stage-specific"):
        routes({"nodes": [place, align, place]}, scene)


def test_future_upright_does_not_change_earlier_placement_validation(monkeypatch):
    place = {
        "id": "place",
        "task_instance_id": "place_group",
        "task_type": "E1",
        "call": {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {"object": "cup", "reference": "table", "relation": "on"},
        },
    }
    align = {
        "id": "align",
        "task_instance_id": "upright_group",
        "task_type": "E2",
        "call": {
            "kind": "registered",
            "call_id": task_program_bundle._MOVE_HELD_OBJECT_CALL_ID,
            "arguments": {"object": "cup"},
        },
    }
    scene = SimpleNamespace(planner_objects=({"runtime_uid": "cup"},), table_top_z=0.7)
    monkeypatch.setattr(
        task_program_bundle, "_relative_world_displacement", lambda *a, **k: [0, 0, 0.1]
    )
    monkeypatch.setattr(
        task_program_bundle, "_longest_local_axis", lambda *a: [1, 0, 0]
    )
    for nodes, expected in (([place, align], "placement"), ([align, place], "upright")):
        graph = {"nodes": nodes, "task_groups": []}
        result = task_program_bundle._task_stability_payload(
            graph, scene, {"skill_profile": {"resources": []}}
        )
        assert result["presets"]["gen_sim.place.stable"]["kind"] == expected


@pytest.mark.parametrize("table_height", [None, float("nan"), float("inf")])
def test_handover_minimum_requires_finite_table_height(table_height) -> None:
    with pytest.raises(ValueError, match="tabletop height"):
        task_program_bundle._handover_position_z(table_height)


def test_collision_world_declares_unreferenced_obstacles_without_changing_default() -> (
    None
):
    graph = _graph()
    graph["nodes"] = [{"call": {"kind": "pick", "object": "bottle"}}]
    scene = SimpleNamespace(
        table_top_z=0.7,
        planner_objects=[
            {"runtime_uid": "bottle", "role": "rigid_object"},
            {"runtime_uid": "obstacle", "role": "rigid_object"},
            {"runtime_uid": "table", "role": "background"},
        ],
    )
    default = task_program_bundle._integration_payload(
        graph, scene, program_id="probe", scene_contract="scene"
    )
    assert {v["entity_id"] for v in default["scene_binding"]["rigid_objects"]} == {
        "bottle",
        "table",
    }
    assert all(
        "collision_role" not in v for v in default["scene_binding"]["rigid_objects"]
    )
    configured = task_program_bundle._integration_payload(
        graph, scene, program_id="probe", scene_contract="scene", collision_world=True
    )
    assert {
        v["entity_id"]: v["collision_role"]
        for v in configured["scene_binding"]["rigid_objects"]
    } == {
        "bottle": "dynamic",
        "obstacle": "dynamic",
        "table": "static",
    }


def test_generic_repick_uses_end_grasp_only_after_all_objects_were_uprighted() -> None:
    scene = SimpleNamespace(
        table_top_z=0.7,
        planner_objects=[
            {"runtime_uid": "can", "role": "rigid_object"},
            {"runtime_uid": "box", "role": "rigid_object"},
            {"runtime_uid": "table", "role": "background"},
        ],
    )
    graph = {
        "nodes": [
            {
                "task_type": "E2",
                "call": {
                    "kind": "registered",
                    "call_id": "gen_sim.clear_released",
                    "arguments": {"object": "can"},
                },
            },
            {"task_type": "E1", "call": {"kind": "pick", "object": "can"}},
        ]
    }
    options = _integration_payload(
        graph, scene, program_id="probe", scene_contract="scene"
    )["profile"]["action_options"]
    assert options["pick"]["pick_object_part"] == "top"

    graph["nodes"].append(
        {"task_type": "E1", "call": {"kind": "pick", "object": "box"}}
    )
    options = _integration_payload(
        graph, scene, program_id="probe", scene_contract="scene"
    )["profile"]["action_options"]
    assert options["pick"]["pick_object_part"] == "center"


def test_semantic_graph_uses_canonical_calls_and_has_stable_hash() -> None:
    graph = _graph()
    validated = validate_semantic_task_graph(graph)
    expected_hash = semantic_task_graph_hash(graph)

    graph["nodes"][0]["call"]["object"] = "mutated"

    assert validated["nodes"][0]["call"] == {"kind": "pick", "object": "cube"}
    assert semantic_task_graph_hash(validated) == expected_hash


def test_semantic_graph_rejects_nested_grounded_execution_data() -> None:
    graph = _graph()
    graph["nodes"][0]["call"]["metadata"] = {"fallback": {"trajectory": [[0.0, 1.0]]}}

    with pytest.raises(ValueError, match="trajectory.*forbidden"):
        validate_semantic_task_graph(graph)


def test_program_must_be_exact_projection_of_semantic_graph(tmp_path: Path) -> None:
    graph = validate_semantic_task_graph(_graph())
    program_path = tmp_path / "program.yaml"
    program = _program(graph)
    save_config(program_path, program)
    _verify_program_projection(program_path, graph)

    program["program"]["items"][1]["steps"]["call"]["object"] = "apple"
    save_config(program_path, program)
    with pytest.raises(ValueError, match="exact SemanticTaskGraph call projection"):
        _verify_program_projection(program_path, graph)


def test_completed_task_groups_survive_later_runtime_failure() -> None:
    graph = validate_semantic_task_graph(_graph())
    runtime_result = {
        "segments": [
            {
                "name": "pick_cube",
                "active": [True],
                "successes": [True],
            },
            {
                "name": "place_cube",
                "active": [True],
                "successes": [False],
            },
        ]
    }

    assert _semantic_success_by_env(graph, runtime_result, num_envs=1) == [
        {"pick_group": True, "place_group": False}
    ]


@pytest.mark.parametrize(
    "active,successes",
    [
        (None, [True]),
        ([], [True]),
        (["false"], [True]),
        ([True], ["false"]),
        ([1], [True]),
        ([True], [1]),
        ([False], [True]),
    ],
)
def test_semantic_projection_rejects_unproven_segment_masks(active, successes) -> None:
    graph = _graph()
    runtime = {
        "segments": [{"name": "pick_cube", "active": active, "successes": successes}]
    }
    assert _semantic_success_by_env(graph, runtime, num_envs=1) == [
        {"pick_group": False, "place_group": False}
    ]


def test_segment_failure_prevents_whole_group_success() -> None:
    graph = _graph()
    graph["task_groups"] = [{"id": "task", "node_ids": ["pick_cube", "place_cube"]}]
    runtime = {
        "segments": [
            {"name": "pick_cube", "active": [True], "successes": [True]},
            {"name": "place_cube", "active": [True], "successes": [False]},
        ]
    }
    assert _semantic_success_by_env(graph, runtime, num_envs=1) == [{"task": False}]


def test_inside_place_waits_for_released_object() -> None:
    node = {
        "id": "place_cube",
        "call": {
            "kind": "place",
            "object": "cube",
            "inside": "inside__tray__cube",
        },
    }

    program_node = _program_node(node, relative_routes={})

    assert program_node["post"] == [
        {
            "kind": "wait_stable",
            "entity": "cube",
            "preset": "contained_rigid_object",
        },
    ]


def test_coordinated_transport_waits_for_observed_object_motion() -> None:
    node = {
        "id": "move_tray",
        "call": {
            "kind": "registered",
            "call_id": "simulation.coordinated_transport",
            "arguments": {
                "object": "tray",
                "target": "tray_forward",
            },
        },
    }

    program_node = _program_node(node, relative_routes={})

    assert program_node["post"] == [
        {
            "kind": "wait_stable",
            "entity": "tray",
            "preset": "transported_rigid_object",
        },
    ]


@pytest.mark.parametrize("with_clearance", [False, True])
def test_relative_placement_checks_stability_again_after_park(
    monkeypatch, with_clearance
) -> None:
    graph = _graph()
    graph["nodes"] = [
        {
            "id": "place",
            "task_type": "E1",
            "task_instance_id": "move",
            "call": {
                "kind": "registered",
                "call_id": "simulation.place_relative",
                "arguments": {
                    "object": "cube",
                    "reference": "block",
                    "relation": "near",
                },
                "resources": {"primary": "right"},
            },
        },
        {
            "id": "park",
            "task_type": "E1",
            "task_instance_id": "move",
            "call": {
                "kind": "registered",
                "call_id": "simulation.park",
                "arguments": {},
            },
        },
    ]
    graph["task_groups"] = [{"node_ids": ["place", "park"]}]
    if with_clearance:
        graph["nodes"].insert(
            1,
            {
                "id": "clear",
                "task_type": "E1",
                "task_instance_id": "move",
                "call": {
                    "kind": "registered",
                    "call_id": "gen_sim.clear_released",
                    "arguments": {"object": "cube", "target": "clear_target"},
                },
            },
        )
        graph["task_groups"][0]["node_ids"].insert(1, "clear")
    route = {
        "object_id": "cube",
        "reference_entity_id": "block",
        "relation": "near",
        "world_displacement": [0.1, 0.0, 0.0],
    }
    monkeypatch.setattr(
        task_program_bundle, "_relative_place_route_payloads", lambda *a, **kw: [route]
    )
    scene = SimpleNamespace(planner_objects=())
    payload = task_program_bundle._task_stability_payload(
        graph, scene, {"skill_profile": {"resources": []}}
    )
    cfg = payload["presets"]["gen_sim.place.stable"]
    assert cfg["kind"] == "placement"
    assert cfg["reference"] == "block"
    assert cfg["displacement"] == [0.1, 0.0, 0.0]
    program = _program_payload(
        graph, "move", scene=scene, stability_presets=set(payload["presets"])
    )
    place, *_, park = program["program"]["items"]
    if with_clearance:
        assert "post" not in place and "validators" not in place
        assert len(park["post"]) == 2
        assert park["validators"]
    else:
        assert park["post"] == [place["post"][-1]]
    assert park["post"][-1]["preset"] == "gen_sim.place.stable"


def test_coordinated_placement_checks_destination_after_release_and_cleanup() -> None:
    graph = _graph()
    graph["nodes"] = [
        {
            "id": "transport",
            "task_type": "E5",
            "call": {
                "kind": "registered",
                "call_id": "simulation.coordinated_transport",
                "arguments": {
                    "object": "tray",
                    "target": "forward",
                    "world_displacement": [-0.14, 0.0, 0.0],
                },
                "resources": {"left": "left", "right": "right"},
            },
        },
        {
            "id": "park",
            "task_type": "E5",
            "call": {
                "kind": "registered",
                "call_id": "simulation.park",
                "arguments": {},
            },
        },
    ]
    graph["task_groups"] = [{"task_type": "E5", "node_ids": ["transport", "park"]}]
    scene = SimpleNamespace(
        planner_objects=({"runtime_uid": "tray", "init_pos": [0.0, 0.0, 0.75]},),
        table_top_z=0.72,
    )
    embodiment = {"skill_profile": {"resources": []}}
    constraints = task_program_bundle._task_stability_payload(graph, scene, embodiment)
    cfg = constraints["presets"]["gen_sim.transport.stable"]
    assert cfg["kind"] == "placement"
    assert cfg["target_position"] == [-0.14, 0.0, 0.75]
    assert cfg["motion_parts"] == []
    program = _program_payload(
        graph, "carry", stability_presets=set(constraints["presets"])
    )
    transport, park = program["program"]["items"]
    assert park["post"] == [transport["post"][-1]]
    assert park["post"][0]["preset"] == "gen_sim.transport.stable"


def test_cleanup_rechecks_public_position_validators_without_a_local_preset() -> None:
    graph = _graph()
    graph["nodes"] = [
        {
            "id": "restore",
            "task_type": "E3",
            "call": {
                "kind": "place",
                "object": "bottle",
                "at": {"kind": "target_ref", "target": "original"},
            },
        },
        {
            "id": "park",
            "task_type": "E3",
            "call": {
                "kind": "registered",
                "call_id": "simulation.park",
                "arguments": {},
            },
        },
    ]
    graph["task_groups"] = [{"task_type": "E3", "node_ids": ["restore", "park"]}]
    restore, park = _program_payload(graph, "pour")["program"]["items"]
    assert park["post"] == restore["post"]
    assert park["validators"] == restore["validators"]
    assert park["validators"][0]["kind"] == "object_near_target"


def test_relative_place_waits_and_validates_fresh_reference_pose() -> None:
    node = {
        "id": "place_cube_left_of_tray",
        "call": {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {
                "object": "cube",
                "reference": "tray",
                "relation": "left_of",
            },
        },
    }
    route = {
        "object_id": "cube",
        "reference_entity_id": "tray",
        "relation": "left_of",
        "world_displacement": [0.0, -0.12, 0.04],
    }

    program_node = _program_node(
        node,
        relative_routes={("cube", "tray", "left_of"): route},
    )

    assert program_node["post"][0]["kind"] == "wait_stable"
    assert program_node["validators"] == [
        {
            "kind": "object_near_relative_target",
            "object": "cube",
            "reference": "tray",
            "displacement": [0.0, -0.12, 0.04],
            "position_tolerance": 0.05,
        }
    ]


def test_phase_one_bundle_rejects_unsupported_robot_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supports only dual_franka"):
        generate_task_program_bundle(
            _graph(),
            object(),
            tmp_path,
            robot_profile="franka",
        )


@pytest.mark.parametrize("task_type", ["E7", "E8", "E9"])
def test_bundle_rejects_out_of_scope_tasks_before_writing_assets(
    tmp_path: Path, task_type: str
) -> None:
    graph = _graph()
    for node in graph["nodes"]:
        node["task_type"] = task_type
    for group in graph["task_groups"]:
        group["task_type"] = task_type
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="only E1-E6"):
        generate_task_program_bundle(graph, None, output, robot_profile="dual_franka")
    assert not output.exists()


def test_dual_franka_mount_is_bound_to_the_scene_table() -> None:
    embodiment = {"simulation": {"init_pos": [-0.7, 0.0, 0.322894]}}

    _bind_embodiment_to_scene(embodiment, table_top_z=1.054499)

    assert embodiment["simulation"]["init_pos"] == pytest.approx([-0.7, 0.0, 0.704499])


def test_reset_settles_only_task_referenced_rigid_objects() -> None:
    scene = SimpleNamespace(
        rigid_objects=(
            {"uid": "cube"},
            {"uid": "wood_cube"},
            {"uid": "smiley_ball"},
        )
    )
    graph = _graph()
    graph["nodes"][1]["call"] = {
        "kind": "place",
        "object": "cube",
        "reference": "wood_cube",
    }

    selected = _task_settle_rigid_objects(graph, scene)

    assert [item["uid"] for item in selected] == ["cube", "wood_cube"]


def test_horizontal_relation_distance_keeps_gripper_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scene extents retain free space for a later parallel-jaw Pick."""
    meshes = {
        "can": np.asarray(
            [
                [-0.03, -0.03, -0.06],
                [0.03, 0.03, 0.06],
            ]
        ),
        "notebook": np.asarray(
            [
                [-0.09, -0.07, -0.01],
                [0.09, 0.07, 0.01],
            ]
        ),
    }
    monkeypatch.setattr(
        task_program_bundle,
        "_mesh_vertices",
        lambda source: meshes[source["runtime_uid"]],
    )

    distance = task_program_bundle._horizontal_relation_distance(
        {"runtime_uid": "can"},
        {"runtime_uid": "notebook", "init_rot": [0.0, 0.0, 0.0]},
        world_axis=1,
        object_axis_aligned=True,
        reference_axis_aligned=False,
        minimum=0.10,
    )

    assert distance == pytest.approx(0.14)

    ordinary_distance = task_program_bundle._horizontal_relation_distance(
        {"runtime_uid": "can", "init_rot": [0.0, 0.0, 0.0]},
        {"runtime_uid": "notebook", "init_rot": [0.0, 0.0, 0.0]},
        world_axis=1,
        object_axis_aligned=False,
        reference_axis_aligned=False,
        minimum=0.10,
    )

    assert ordinary_distance == pytest.approx(0.12)


def test_e2_place_requires_task_owned_stability_and_shared_position_validation() -> (
    None
):
    node = {
        "id": "upright_can__call_04",
        "task_instance_id": "upright_can",
        "task_type": "E2",
        "call": {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {"object": "can", "reference": "table", "relation": "on"},
        },
    }
    route = {
        "object_id": "can",
        "reference_entity_id": "table",
        "world_displacement": [0.0, 0.0, 0.1],
    }
    routes = {("can", "table", "on"): route}
    with pytest.raises(ValueError, match="task-owned upright stability"):
        _program_node(node, relative_routes=routes)
    program_node = _program_node(node, relative_routes=routes, task_stability=True)

    assert program_node["post"] == [
        {"kind": "wait_stable", "entity": "can", "preset": "rigid_object"},
        {
            "kind": "wait_stable",
            "entity": "can",
            "preset": "gen_sim.upright_can__call_04.stable",
        },
    ]
    assert program_node["validators"] == [
        {
            "kind": "object_near_relative_target",
            "object": "can",
            "reference": "table",
            "displacement": [0.0, 0.0, 0.1],
            "position_tolerance": 0.05,
        }
    ]


def test_relative_place_waits_then_validates_live_relation() -> None:
    node = {
        "id": "place_behind",
        "task_type": "E1",
        "call": {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {
                "object": "can",
                "reference": "bottle",
                "relation": "behind",
            },
            "resources": {"primary": "right"},
        },
    }

    route = {
        "object_id": "can",
        "reference_entity_id": "bottle",
        "relation": "behind",
        "world_displacement": [0.18, 0.0, 0.05],
    }
    program_node = _program_node(
        node, relative_routes={("can", "bottle", "behind"): route}
    )

    assert program_node["post"] == [
        {"kind": "wait_stable", "entity": "can", "preset": "rigid_object"}
    ]
    assert program_node["validators"] == [
        {
            "kind": "object_near_relative_target",
            "object": "can",
            "reference": "bottle",
            "displacement": [0.18, 0.0, 0.05],
            "position_tolerance": 0.05,
        }
    ]


def test_upright_target_uses_the_normalized_mesh_origin_offset(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=[0.06, 0.24, 0.06])
    mesh.apply_translation([0.0, 0.12, 0.0])
    mesh_path = tmp_path / "bottle.glb"
    mesh.export(mesh_path)
    graph = {
        "schema_version": "semantic_task_graph/v1",
        "task_id": "upright",
        "instruction": "upright bottle",
        "planner_route": "offline",
        "integration_fingerprint": "0" * 64,
        "targets": {
            "step_upright_target": {
                "kind": "cyclic_pose",
                "values": [
                    {
                        "position": [0.1, 0.2, 0.85],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            },
            "step_upright_staging_target": {
                "kind": "cyclic_pose",
                "values": [
                    {
                        "position": [0.1, 0.2, 0.85],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            },
        },
        "nodes": [
            {
                "id": "pick",
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.pick",
                    "arguments": {"object": "bottle", "target": "step_upright_target"},
                },
                "depends_on": [],
                "task_instance_id": "step",
                "task_type": "E2",
                "role": "primary",
            },
            {
                "id": "stage",
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.move_held_object",
                    "arguments": {
                        "object": "bottle",
                        "target": "step_upright_staging_target",
                    },
                },
                "depends_on": ["pick"],
                "task_instance_id": "step",
                "task_type": "E2",
                "role": "primary",
            },
            {
                "id": "move",
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.move_held_object",
                    "arguments": {
                        "object": "bottle",
                        "target": "step_upright_target",
                    },
                },
                "depends_on": ["stage"],
                "task_instance_id": "step",
                "task_type": "E2",
                "role": "primary",
            },
        ],
        "task_groups": [
            {
                "id": "step",
                "task_type": "E2",
                "node_ids": ["pick", "stage", "move"],
                "depends_on": [],
                "success": {
                    "kind": "semantic_task_term",
                    "step_id": "step",
                    "type": "object_upright",
                },
            }
        ],
        "success": {"kind": "all_task_groups"},
    }
    scene = SimpleNamespace(
        table_top_z=0.72,
        articulations=(),
        planner_objects=(
            {
                "runtime_uid": "bottle",
                "role": "rigid_object",
                "shape": {"shape_type": "Mesh", "fpath": str(mesh_path)},
                "init_rot": [90.0, 0.0, 0.0],
            },
            {
                "runtime_uid": "table",
                "role": "background",
                "init_pos": [0.0, 0.0, 0.0],
                "attributes": {
                    "final_world_aabb": {
                        "min": [-0.5, -0.5, 0.0],
                        "max": [0.5, 0.5, 0.72],
                    }
                },
            },
        ),
    )

    refined = _refine_upright_targets(graph, scene)

    assert refined["targets"]["step_upright_target"]["values"][0][
        "position"
    ] == pytest.approx([0.1, 0.2, 0.73])
    assert refined["targets"]["step_upright_staging_target"]["values"][0][
        "position"
    ] == pytest.approx([0.1, 0.2, 0.93])

    integration = _integration_payload(
        refined,
        scene,
        program_id="upright",
        scene_contract="upright_scene_v1",
    )
    release_safe_options = integration["profile"]["action_options"]["simulation.pick"]
    assert release_safe_options["kind"] == "pick_up"
    assert release_safe_options["grasp_settle_steps"] == 16
    assert release_safe_options["pick_object_part"] == "center"
    assert "object_axis_approach_weight" not in release_safe_options
    lowerer = next(
        item
        for item in integration["runtime_services"]["registered_semantic_lowerers"]
        if item["kind"] == "pick"
    )
    route = lowerer["routes"][0]
    assert route["grasp_region"] == "upper_half"
    assert "approach_axis_weight" not in route
    assert release_safe_options["approach_direction"] == pytest.approx(
        [0.0, 2**-0.5, -(2**-0.5)]
    )
    assert "required_object_target_poses" not in route
    assert route["release_clearance_object_pose"]["position"] == pytest.approx(
        [0.1, 0.2, 0.73]
    )
    assert route["release_clearance_plane_z"] == pytest.approx(0.72)
    assert route["release_clearance_safety_margin"] == pytest.approx(0.02)
    bottle_binding = next(
        item
        for item in integration["scene_binding"]["rigid_objects"]
        if item["entity_id"] == "bottle"
    )
    assert bottle_binding["affordances"][0]["internal_axis"] == [0.0, -0.0, 1.0]
    assert decode_task_lowerer(lowerer, path="lowerer").call_id == ("simulation.pick")


def test_generated_handover_uses_only_baseline_release_and_retreat_options() -> None:
    graph = {
        "schema_version": "semantic_task_graph/v1",
        "task_id": "handover",
        "instruction": "hand over can",
        "planner_route": "offline",
        "integration_fingerprint": "0" * 64,
        "targets": {},
        "nodes": [
            {
                "id": "handover",
                "call": {
                    "kind": "hand_over",
                    "object": "can",
                    "resources": {"source": "left", "destination": "right"},
                },
                "depends_on": [],
                "task_instance_id": "step",
                "task_type": "E3",
                "role": "primary",
            }
        ],
        "task_groups": [],
        "success": {"kind": "all_task_groups"},
    }
    scene = SimpleNamespace(
        table_top_z=0.72,
        articulations=(),
        planner_objects=(
            {
                "runtime_uid": "can",
                "role": "rigid_object",
                "shape": {"shape_type": "Cube", "size": [0.06, 0.06, 0.12]},
            },
            {"runtime_uid": "table", "role": "background"},
        ),
    )

    integration = _integration_payload(
        graph,
        scene,
        program_id="handover",
        scene_contract="handover_scene_v1",
    )
    options = integration["profile"]["action_options"]["hand_over"]

    assert options["hand_interp_steps"] == 16
    assert options["hold_steps"] == 8
    assert "source_release_settle_steps" not in options
    assert options["retreat_steps"] == 36
    assert options["retreat_distance"] == pytest.approx(0.10)
    assert integration["runtime_services"]["handover_pose_providers"][0][
        "final_position"
    ] == pytest.approx([0.0, -0.08, 0.82])


def test_task_gripper_opening_respects_pad_clearance_and_stricter_models() -> None:
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _calibrate_task_gripper_opening,
    )

    generators = {
        "left": {"model": {"model_id": "robotiq_arg2f_140", "max_opening_width": 0.15}},
        "right": {
            "model": {"model_id": "robotiq_arg2f_140", "max_opening_width": 0.12}
        },
        "other": {"model": {"model_id": "other", "max_opening_width": 0.2}},
    }
    payload = {
        "skill_profile": {"runtime_services": {"grasp_pose_generators": generators}}
    }
    _calibrate_task_gripper_opening(payload)
    assert generators["left"]["model"]["max_opening_width"] == 0.128
    assert generators["right"]["model"]["max_opening_width"] == 0.12
    assert generators["other"]["model"]["max_opening_width"] == 0.2


def test_task_pick_directions_are_declared_per_policy_not_injected_by_lowerers() -> (
    None
):
    graph = _graph()
    graph["targets"] = {
        name: {
            "kind": "cyclic_pose",
            "values": [
                {"position": [0.0, 0.0, 0.9], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
            ],
        }
        for name in ("one", "two")
    }
    graph["nodes"] = [
        {
            "id": name,
            "task_type": "E2",
            "call": {
                "kind": "registered",
                "call_id": f"gen_sim.pick.{name}",
                "arguments": {"object": name, "target": name},
            },
        }
        for name in ("one", "two")
    ]
    scene = SimpleNamespace(
        table_top_z=0.72,
        articulations=(),
        planner_objects=(
            {
                "runtime_uid": "one",
                "role": "rigid_object",
                "init_pos": [0.0, -0.2, 0.8],
                "init_rot": [90.0, 0.0, 0.0],
            },
            {
                "runtime_uid": "two",
                "role": "rigid_object",
                "init_pos": [0.0, 0.2, 0.8],
                "init_rot": [-90.0, 0.0, 0.0],
            },
            {"runtime_uid": "table", "role": "background", "init_pos": [0.0, 0.0, 0.0]},
        ),
    )
    integration = _integration_payload(
        graph, scene, program_id="directions", scene_contract="directions_scene"
    )
    options = integration["profile"]["action_options"]
    for key in ("pick", "gen_sim.pick.one", "gen_sim.pick.two"):
        intervals = options[key]["hand_interp_steps"] - 1
        assert 0.7 / (intervals * 0.04) <= 2.0
    assert options["gen_sim.pick.one"]["approach_direction"] == pytest.approx(
        [0.0, 2**-0.5, -(2**-0.5)]
    )
    assert options["gen_sim.pick.two"]["approach_direction"] == pytest.approx(
        [0.0, -(2**-0.5), -(2**-0.5)]
    )
    factories = integration["runtime_services"]["registered_semantic_lowerers"]
    assert {f["call_id"] for f in factories if f["kind"] == "pick"} == {
        "gen_sim.pick.one",
        "gen_sim.pick.two",
    }


def test_upright_support_target_uses_the_mesh_top_not_half_extent(
    tmp_path: Path,
) -> None:
    support_mesh = trimesh.creation.box(extents=[0.06, 0.12, 0.06])
    support_mesh.apply_translation([0.0, 0.06, 0.0])
    support_path = tmp_path / "can.glb"
    support_mesh.export(support_path)
    child_mesh = trimesh.creation.box(extents=[0.10, 0.10, 0.10])
    child_path = tmp_path / "apple.glb"
    child_mesh.export(child_path)

    pose = _support_target_pose(
        {"shape": {"shape_type": "Mesh", "fpath": str(support_path)}},
        {"shape": {"shape_type": "Mesh", "fpath": str(child_path)}},
        axis_aligned=True,
    )

    assert pose[3] == pytest.approx(0.0)
    assert pose[7] == pytest.approx(0.0)
    assert pose[11] == pytest.approx(0.18)


@pytest.mark.parametrize("max_hulls", [1, 8])
@pytest.mark.parametrize("method", [None, "coacd", "visacd"])
def test_scene_payload_defaults_to_visacd_and_preserves_explicit_algorithm(
    max_hulls: int, method: str | None
) -> None:
    from embodichain.lab.sim.cfg import RigidObjectCfg
    from embodichain.lab.sim.spawn.descriptors import rigid_desc_from_cfg

    source = {
        "uid": "mesh",
        "shape": {"shape_type": "Mesh", "fpath": "/tmp/mesh.glb"},
        "max_convex_hull_num": max_hulls,
    }
    if method is not None:
        source["acd_method"] = method
    scene = SimpleNamespace(background=(), rigid_objects=(source,), articulations=())

    payload = _scene_payload(scene, program_id="collision_policy")

    runtime = payload["simulation"]["rigid_object"][0]
    cfg = RigidObjectCfg.from_dict(runtime)
    descriptor, _ = rigid_desc_from_cfg(cfg)
    if max_hulls > 1:
        assert cfg.shape.collision.acd_method == (method or "visacd")
        assert descriptor.collisions[0].decomp_algorithm == (method or "visacd")
        assert descriptor.collisions[0].decomp_max_hulls == max_hulls
    else:
        assert runtime["shape"]["collision"] == {"approximation": "convex_hull"}
    assert "collision" not in source["shape"]


def test_grasp_contact_clearances_follow_physics_overrides() -> None:
    from copy import deepcopy
    from dexsim.spawn import DexsimCollisionDesc

    embodiment = {
        "simulation": {
            "attrs": {"collision_props": {"contact_offset": 0.004}},
            "link_attrs": {
                "pads": {"attrs": {"collision_props": {"contact_offset": 0.006}}},
                "mass_only": {"attrs": {"mass_props": {"mass": 1.0}}},
            },
        }
    }
    before = deepcopy(embodiment)
    scene = SimpleNamespace(
        rigid_objects=(
            {"uid": "cup", "attrs": {"collision_props": {"contact_offset": 0.003}}},
            {"uid": "default"},
            {"uid": "zero", "attrs": {"collision_props": {"contact_offset": 0.0}}},
        )
    )
    values = task_program_bundle._grasp_contact_clearances(scene, embodiment)
    assert values["cup"] == pytest.approx(0.009)
    assert values["default"] == pytest.approx(
        0.006 + DexsimCollisionDesc().contact_offset
    )
    assert values["zero"] == pytest.approx(0.006)
    assert embodiment == before


@pytest.mark.parametrize("offset", [-0.001, float("nan"), True])
def test_grasp_contact_clearances_reject_invalid_physics(offset) -> None:
    with pytest.raises(ValueError, match="contact_offset"):
        task_program_bundle._grasp_contact_clearances(
            SimpleNamespace(rigid_objects=()),
            {"simulation": {"attrs": {"collision_props": {"contact_offset": offset}}}},
        )


def test_generated_usd_articulation_disables_urdf_only_pk_chain() -> None:
    scene = SimpleNamespace(
        table_top_z=0.72,
        background=(),
        rigid_objects=(),
        articulations=(
            {
                "uid": "button",
                "fpath": "/tmp/button.usdc",
                "category": "button_box",
                "name": "red button",
                "proxy_glb_fpath": "/tmp/button.glb",
            },
        ),
    )

    payload = _scene_payload(scene, program_id="press")

    articulation = payload["simulation"]["articulation"][0]
    assert articulation["build_pk_chain"] is False
    assert "category" not in articulation
    assert "name" not in articulation
    assert "proxy_glb_fpath" not in articulation


@pytest.mark.parametrize("section", ["background", "rigid_object", "articulation"])
def test_scene_payload_preserves_intrinsic_xyz_through_config_decode(
    section: str,
) -> None:
    from scipy.spatial.transform import Rotation
    from embodichain.lab.sim.cfg import ArticulationCfg, RigidObjectCfg
    from embodichain.lab.sim.spawn.descriptors import _pose_from_cfg

    position = [0.1, -0.2, 0.8]
    rotation = [70.0, -25.0, 15.0]
    config = {"uid": "target", "init_pos": position, "init_rot": rotation}
    scene = SimpleNamespace(
        table_top_z=0.72,
        background=(config,) if section == "background" else (),
        rigid_objects=(config,) if section == "rigid_object" else (),
        articulations=(config,) if section == "articulation" else (),
    )
    payload = _scene_payload(scene, program_id="pose_decode")
    item = payload["simulation"][section][0]
    cfg_type = ArticulationCfg if section == "articulation" else RigidObjectCfg
    decoded = cfg_type.from_dict(item)
    pose = _pose_from_cfg(decoded)
    expected_rotation = Rotation.from_euler("XYZ", rotation, degrees=True).as_matrix()

    assert np.allclose(pose[:3, :3], expected_rotation, atol=1e-6)
    assert np.allclose(pose[:3, 3], position)
    assert "init_local_pose" in item
    assert "init_local_pose" not in config


def test_scene_payload_preserves_explicit_pose_without_mutating_source() -> None:
    from scipy.spatial.transform import Rotation

    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("XYZ", [20, 30, 40], degrees=True).as_matrix()
    pose[:3, 3] = [0.1, 0.2, 0.3]
    config = {"uid": "object", "init_local_pose": pose.tolist()}
    scene = SimpleNamespace(
        table_top_z=0.72, background=(), rigid_objects=(config,), articulations=()
    )
    result = _scene_payload(scene, program_id="explicit_pose")["simulation"][
        "rigid_object"
    ][0]
    assert np.allclose(result["init_local_pose"], pose)
    result["init_local_pose"][0][3] = 9.0
    assert config["init_local_pose"][0][3] == pytest.approx(0.1)


def _prepared_axis_scene(
    tmp_path: Path, *, source_extents: tuple[float, float, float] = (0.06, 0.24, 0.06)
) -> PreparedScene:
    mesh_path = tmp_path / "bottle.glb"
    trimesh.creation.box(extents=source_extents).export(mesh_path)
    bottle = {
        "uid": "bottle",
        "shape": {"shape_type": "Mesh", "fpath": str(mesh_path)},
        "init_pos": [0.1, -0.2, 0.85],
        "init_rot": [90.0, 0.0, 0.0],
    }
    table = {
        "uid": "table",
        "shape": {"shape_type": "Cube", "size": [1.0, 1.0, 0.72]},
        "init_pos": [0.0, 0.0, 0.36],
    }
    scene = PreparedScene(
        source_config_path=tmp_path / "scene.yaml",
        scene_dir=tmp_path,
        planner_objects=(
            {**bottle, "runtime_uid": "bottle", "role": "rigid_object"},
            {
                **table,
                "runtime_uid": "table",
                "role": "background",
                # PreparedScene already owns the measured tabletop; no duplicate AABB is needed.
            },
        ),
        background=(table,),
        rigid_objects=(bottle,),
        articulations=(),
        uid_map={"bottle": "bottle", "table": "table"},
        table_top_z=0.72,
        z_rotation_degrees=0.0,
        body_scale_policy="preserve",
        body_scale=(1.0, 1.0, 1.0),
        asset_hashes={},
    )
    return scene


@pytest.mark.parametrize(
    "extents,rotation,horizontal,axis",
    [
        ((0.06, 0.24, 0.06), [0.0, 0.0, 0.0], False, [0.0, 0.0, 1.0]),
        ((0.06, 0.24, 0.06), [90.0, 0.0, 0.0], True, [0.0, 0.0, 1.0]),
        ((0.24, 0.06, 0.06), [0.0, 0.0, 90.0], True, [1.0, 0.0, 0.0]),
    ],
)
def test_handover_source_region_follows_measured_axis(
    tmp_path, extents, rotation, horizontal, axis
):
    scene = _prepared_axis_scene(tmp_path, source_extents=extents)
    scene.planner_objects[0]["init_rot"][:] = rotation
    graph = _fresh_handover_graph()
    graph["nodes"][0]["call"] = {
        "kind": "pick",
        "object": "bottle",
        "resources": {"primary": "left"},
    }
    graph["nodes"][1]["call"]["object"] = "bottle"
    generated, paths = generate_task_program_bundle(
        graph, scene, tmp_path / "bundle", robot_profile="dual_franka"
    )
    _verify_program_projection(paths.program, generated)
    integration = load_config(paths.integration)
    call_id = generated["nodes"][0]["call"]["call_id"]
    assert call_id == (
        "gen_sim.pick.handover_horizontal_source.bottle"
        if horizontal
        else "gen_sim.pick.handover_source"
    )
    options = integration["profile"]["action_options"][call_id]
    assert options["pick_object_part"] == "top"
    assert "pick_object_local_axis" not in options
    factory = next(
        x
        for x in integration["runtime_services"]["registered_semantic_lowerers"]
        if x.get("call_id") == call_id
    )
    assert "grasp_region" not in factory["routes"][0]
    assert len(generated["nodes"]) == 3
    if horizontal:
        assert np.linalg.norm(options["approach_direction"]) == pytest.approx(1.0)
        binding = next(
            x
            for x in integration["scene_binding"]["rigid_objects"]
            if x["entity_id"] == "bottle"
        )
        assert binding["affordances"][0]["internal_axis"] == axis
        constraints = load_config(paths.program.parent / "constraints.json")["presets"]
        final = constraints[f"gen_sim.{generated['nodes'][-1]['id']}.stable"]
        assert final["kind"] == "hold"
        assert abs(final["world_axis"][2]) < 1e-6
        assert final["motion_parts"] == ["right_arm"]


@pytest.mark.parametrize(
    ("task_type", "terminal"), [("E1", "place"), ("E4", "hold"), ("E4", "place")]
)
def test_explicit_orientation_bundle_uses_shared_preflight_and_terminal_post(
    tmp_path: Path, task_type: str, terminal: str
) -> None:
    scene = _prepared_axis_scene(tmp_path)
    graph = _graph()
    resource = "left" if task_type == "E1" else "right"
    calls = [{"kind": "pick", "object": "bottle", "resources": {"primary": "left"}}]
    if task_type == "E4":
        calls.append(
            {
                "kind": "hand_over",
                "object": "bottle",
                "resources": {"source": "left", "destination": "right"},
            }
        )
    calls.append(
        {
            "kind": "registered",
            "call_id": "gen_sim.align_held",
            "arguments": {
                "object": "bottle",
                "target": "current_object_pose",
                "preserve_yaw": True,
            },
            "resources": {"primary": resource},
        }
    )
    if terminal == "place":
        graph["targets"] = {
            "orientation_upright_staging_target": {
                "kind": "cyclic_pose",
                "values": [
                    {
                        "position": [0.1, -0.2, 1.05],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            }
        }
        calls.extend(
            [
                {
                    "kind": "registered",
                    "call_id": "simulation.place_relative",
                    "arguments": {
                        "object": "bottle",
                        "reference": "table",
                        "relation": "on",
                    },
                    "resources": {"primary": resource},
                },
                {
                    "kind": "registered",
                    "call_id": "gen_sim.clear_released",
                    "arguments": {
                        "object": "bottle",
                        "target": "orientation_upright_staging_target",
                    },
                    "resources": {"primary": resource},
                },
                {
                    "kind": "registered",
                    "call_id": "simulation.park",
                    "arguments": {},
                    "resources": {"primary": resource},
                },
            ]
        )
    graph["nodes"] = [
        {
            "id": f"call_{i}",
            "task_type": task_type,
            "task_instance_id": "orientation",
            "role": "primary",
            "depends_on": [] if i == 0 else [f"call_{i-1}"],
            "call": call,
        }
        for i, call in enumerate(calls)
    ]
    graph["task_groups"] = [
        {
            "id": "orientation",
            "task_type": task_type,
            "node_ids": [n["id"] for n in graph["nodes"]],
            "depends_on": [],
            "success": {"kind": "call_completed"},
        }
    ]
    generated, paths = generate_task_program_bundle(
        graph, scene, tmp_path / "bundle", robot_profile="dual_franka"
    )
    _verify_program_projection(paths.program, generated)
    policy = load_config(paths.execution_policy)
    assert policy["tracking"]["terminal_max_abs_error"] == pytest.approx(0.25)
    if task_type == "E4":
        assert policy["motion"]["sample_count"] >= 260
    integration = load_config(paths.integration)
    if any(
        node["call"].get("call_id") == "simulation.move_held_object"
        for node in generated["nodes"]
    ):
        transport_monitor = integration["profile"]["effect_monitors"][
            "simulation.move_held_object"
        ]
        assert transport_monitor["monitor_id"] == "builtin.composite_effect"
        assert transport_monitor["params"]["attached_translation_threshold"] == 0.06
    if task_type == "E4":
        source_call = generated["nodes"][0]["call"]
        assert (
            source_call["call_id"] == "gen_sim.pick.handover_horizontal_source.bottle"
        )
        options = integration["profile"]["action_options"]
        assert options[source_call["call_id"]]["pick_object_part"] == "top"
        assert "pick_object_local_axis" not in options[source_call["call_id"]]
        assert options["hand_over"]["receive_pick_object_part"] == "center"
        assert source_call["call_id"] in integration["profile"]["effect_monitors"]
    assert integration["profile"]["action_options"]["place"][
        "lift_height"
    ] == pytest.approx(0.05)
    if terminal == "place":
        place_options = integration["profile"]["action_options"][
            "simulation.place_relative"
        ]
        assert "max_approach_retract_z" not in place_options
        assert place_options["lift_height"] == pytest.approx(0.10)
        place_params = integration["profile"]["effect_monitors"][
            "simulation.place_relative"
        ]["params"]
        assert place_params["attached_translation_threshold"] == pytest.approx(0.02)
        assert place_params["detached_translation_threshold"] == pytest.approx(0.03)
    program = load_config(paths.program)
    post = program["program"]["items"][-1]["post"][-1]
    constraints = load_config(paths.program.parent / "constraints.json")["presets"]
    assert constraints[post["preset"]]["kind"] == (
        "hold" if terminal == "hold" else "upright"
    )
    assert constraints[post["preset"]]["local_axis"] == [0.0, -0.0, 1.0]
    if terminal == "hold":
        assert constraints[post["preset"]]["motion_parts"] == ["right_arm"]
        assert "target_position" not in constraints[post["preset"]]


def test_stack_alignment_does_not_replace_support_acceptance_with_upright_only(
    tmp_path: Path,
) -> None:
    source_scene = _prepared_axis_scene(tmp_path)
    objects = source_scene.planner_objects
    scene = SimpleNamespace(
        planner_objects=(*objects, {**objects[0], "runtime_uid": "support"}),
        table_top_z=0.72,
    )
    graph = _graph()
    calls = [
        ("E2", "support_group", "simulation.axis_align", {"object": "support"}),
        (
            "E1",
            "stack_group",
            "gen_sim.align_held",
            {"object": "bottle", "target": "staging", "preserve_yaw": True},
        ),
        (
            "E1",
            "stack_group",
            "gen_sim.stack_place",
            {"object": "bottle", "reference": "support", "relation": "on"},
        ),
    ]
    graph["nodes"] = [
        {
            "id": f"node_{i}",
            "task_type": kind,
            "task_instance_id": group,
            "call": {"kind": "registered", "call_id": call_id, "arguments": args},
        }
        for i, (kind, group, call_id, args) in enumerate(calls)
    ]
    graph["task_groups"] = [
        {"node_ids": ["node_0"]},
        {"node_ids": ["node_1", "node_2"]},
    ]
    constraints = task_program_bundle._task_stability_payload(
        graph, scene, {"skill_profile": {"resources": []}}
    )
    assert constraints["presets"]["gen_sim.node_2.stable"]["kind"] == "stack"


def test_ordinary_on_placement_does_not_invent_an_orientation_requirement(tmp_path):
    source_scene = _prepared_axis_scene(tmp_path)
    objects = source_scene.planner_objects
    scene = SimpleNamespace(
        planner_objects=(*objects, {**objects[0], "runtime_uid": "support"}),
        table_top_z=0.72,
    )
    graph = _graph()
    graph["nodes"] = [
        {
            "id": "place",
            "task_type": "E1",
            "task_instance_id": "place_group",
            "call": {
                "kind": "registered",
                "call_id": "simulation.place_relative",
                "arguments": {
                    "object": "bottle",
                    "reference": "support",
                    "relation": "on",
                },
            },
        }
    ]
    graph["task_groups"] = [{"node_ids": ["place"]}]
    cfg = task_program_bundle._task_stability_payload(
        graph, scene, {"skill_profile": {"resources": []}}
    )["presets"]["gen_sim.place.stable"]
    assert cfg["kind"] == "supported_placement"
    assert (
        not {"local_axis", "reference_axis", "object_bottom", "reference_top"}
        & cfg.keys()
    )


@pytest.mark.parametrize(
    "call_id", ["simulation.coordinated_hold", "simulation.coordinated_transport"]
)
@pytest.mark.parametrize("displacement", [[-0.1, 0.0, 0.05], [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("max_episode_steps", [None, 5000])
def test_coordinated_bundle_composes_against_unmodified_public_options(
    tmp_path: Path,
    call_id: str,
    displacement: list[float],
    max_episode_steps: int | None,
) -> None:
    scene = _prepared_axis_scene(tmp_path)
    graph = _graph()
    graph["nodes"] = [
        {
            "id": "carry",
            "task_instance_id": "carry",
            "task_type": "E5",
            "role": "primary",
            "depends_on": [],
            "call": {
                "kind": "registered",
                "call_id": call_id,
                "arguments": {
                    "object": "bottle",
                    "target": "forward",
                    "world_displacement": displacement,
                },
                "resources": {"left": "left", "right": "right"},
            },
        }
    ]
    graph["task_groups"] = [
        {
            "id": "carry",
            "task_type": "E5",
            "node_ids": ["carry"],
            "depends_on": [],
            "success": {"kind": "call_completed"},
        }
    ]
    generated, paths = generate_task_program_bundle(
        graph,
        scene,
        tmp_path / "bundle",
        robot_profile="dual_franka",
        max_episode_steps=max_episode_steps,
    )
    deployment = load_config(paths.deployment)
    assert deployment["max_episode_steps"] == (
        10000 if max_episode_steps is None else max_episode_steps
    )
    assert (
        deployment["env"]["events"]["settle_objects_on_reset"]["params"][
            "restore_initial_xy"
        ]
        is False
    )
    _verify_program_projection(paths.program, generated)
    integration = load_config(paths.integration)
    assert (
        "source_release_settle_steps"
        not in integration["profile"]["action_options"]["hand_over"]
    )
    assert (
        load_config(paths.integration_fingerprint)["schema_version"]
        == "semantic_integration_fingerprint/v2"
    )


@pytest.mark.parametrize(
    ("source_extents", "expected_axis"),
    [
        ((0.06, 0.24, 0.06), [0.0, -0.0, 1.0]),
        ((0.06, 0.06, 0.24), [0.0, -1.0, 0.0]),
    ],
)
@pytest.mark.parametrize("resource", ["left", "right"])
def test_generated_e2_bundle_shares_release_route_with_axis_acceptance(
    tmp_path: Path,
    source_extents: tuple[float, float, float],
    expected_axis: list[float],
    resource: str,
) -> None:
    """The merged E2 route passes real configured composition and preflight."""
    from embodichain.lab.sim.cfg import RobotCfg

    scene = _prepared_axis_scene(tmp_path, source_extents=source_extents)
    scene.rigid_objects[0]["attrs"] = {"mass_props": {"mass": 0.01}}
    graph = _graph()
    graph["task_id"] = "upright"
    graph["targets"] = {
        name: {
            "kind": "cyclic_pose",
            "values": [
                {
                    "position": [0.1, -0.2, 0.85],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ],
        }
        for name in ("upright_upright_target", "upright_upright_staging_target")
    }
    calls = [
        (
            "simulation.pick",
            {"object": "bottle", "target": "upright_upright_target"},
        ),
        (
            "gen_sim.align_held",
            {
                "object": "bottle",
                "target": "upright_upright_staging_target",
                "preserve_yaw": False,
            },
        ),
        (
            "gen_sim.align_held",
            {
                "object": "bottle",
                "target": "upright_upright_staging_target",
                "preserve_yaw": True,
            },
        ),
        (
            "simulation.place_relative",
            {"object": "bottle", "reference": "table", "relation": "on"},
        ),
        (
            "gen_sim.clear_released",
            {"object": "bottle", "target": "upright_upright_staging_target"},
        ),
        ("simulation.park", {}),
    ]
    graph["nodes"] = [
        {
            "id": f"call_{index}",
            "task_instance_id": "upright",
            "task_type": "E2",
            "role": "primary",
            "depends_on": [] if index == 0 else [f"call_{index - 1}"],
            "call": {
                "kind": "registered",
                "call_id": call_id,
                "arguments": arguments,
                "resources": {"primary": resource},
            },
        }
        for index, (call_id, arguments) in enumerate(calls)
    ]
    graph["task_groups"] = [
        {
            "id": "upright",
            "task_type": "E2",
            "depends_on": [],
            "node_ids": [node["id"] for node in graph["nodes"]],
            "success": {"kind": "call_completed"},
        }
    ]

    generated, paths = generate_task_program_bundle(
        graph,
        scene,
        tmp_path / "bundle",
        robot_profile="dual_franka",
    )

    _verify_program_projection(paths.program, generated)
    integration = load_config(paths.integration)
    segments = load_config(paths.program)["program"]["items"]
    robot_cfg = RobotCfg.from_dict(load_config(paths.embodiment)["simulation"])
    assert "mimic_compliance" not in load_config(paths.embodiment)["simulation"]
    assert robot_cfg.joint_drive_props.max_effort[f"{resource}_eef"] == 0.5
    assert generated["nodes"][0]["call"]["call_id"] == "gen_sim.pick.upright"
    assert "gen_sim.pick.upright" in integration["profile"]["action_options"]
    release = segments[-3]
    terminal = segments[-1]
    route = next(
        item
        for item in integration["runtime_services"]["registered_semantic_lowerers"]
        if item["kind"] == "place_relative"
    )["routes"][0]
    # Upright yaw is selected during alignment, not changed again on descent.
    assert "world_yaw_offset" not in route
    assert "post" not in release and "validators" not in release
    assert terminal["validators"][0]["kind"] == "object_near_relative_target"
    # Planning includes 1 cm release clearance; acceptance uses the support surface.
    expected_displacement = list(route["world_displacement"])
    expected_displacement[2] -= 0.01
    assert terminal["validators"][0]["displacement"] == pytest.approx(
        expected_displacement
    )
    constraints = load_config(paths.program.parent / "constraints.json")["presets"]
    preset = terminal["post"][-1]["preset"]
    assert constraints[preset]["kind"] == "upright"
    assert constraints[preset]["position_tolerance"] == pytest.approx(0.05)
    assert constraints[preset]["local_axis"] == expected_axis
    assert constraints[preset]["displacement"] == pytest.approx(expected_displacement)
    assert len(terminal["validators"]) == 1
    assert terminal["steps"]["call"]["call_id"] == "simulation.park"
    assert len(terminal["post"]) == 2
    assert set(release["steps"]["call"]["arguments"]) == {
        "object",
        "reference",
        "relation",
    }
    assert generated["integration_fingerprint"] != "0" * 64

    from embodichain.gen_sim.task_engine._task_program.assembly import (
        compose_deployment,
        load_deployment,
    )
    from embodichain.gen_sim.task_engine._task_program.align_held import (
        with_held_alignment,
    )

    deployment = load_deployment(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.deployment.parent,
    )
    registration = deployment.integration.registration
    for preset in registration.robot_profile_binding.presets:
        monitor = preset.effect_monitors["gen_sim.align_held"]
        assert monitor.monitor_id == "builtin.composite_effect"
        assert monitor.params["attached_translation_threshold"] == 0.06
    factory = next(
        factory
        for factory in registration.registered_semantic_lowerer_factories
        if factory.call_id == "gen_sim.align_held"
    )
    assert factory.verify_retention

    base = compose_deployment(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.deployment.parent,
    ).integration.registration
    effectless = with_held_alignment(
        base,
        program=load_config(paths.program),
        constraints={
            "upright": SimpleNamespace(entity="bottle", local_axis=tuple(expected_axis))
        },
        verify_retention=False,
    )
    for before, drawer, retained in zip(
        base.robot_profile_binding.presets,
        effectless.robot_profile_binding.presets,
        registration.robot_profile_binding.presets,
        strict=True,
    ):
        assert drawer.effect_monitors == before.effect_monitors
        assert {
            key: value
            for key, value in retained.effect_monitors.items()
            if key != "gen_sim.align_held"
        } == before.effect_monitors
