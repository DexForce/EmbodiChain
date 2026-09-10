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
)
from embodichain.utils.utility import load_config, save_config
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene


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
            "position_tolerance": 0.04,
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
            "position_tolerance": 0.04,
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
            "position_tolerance": 0.04,
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
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                ],
            },
            "step_upright_staging_target": {
                "kind": "cyclic_pose",
                "values": [
                    {
                        "position": [0.1, 0.2, 0.85],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
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
    ] == pytest.approx([0.0, -0.08, 0.913])


def test_task_pick_directions_are_declared_per_policy_not_injected_by_lowerers() -> (
    None
):
    graph = _graph()
    graph["targets"] = {
        name: {
            "kind": "cyclic_pose",
            "values": [
                {"position": [0.0, 0.0, 0.9], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
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


def _prepared_axis_scene(tmp_path: Path) -> PreparedScene:
    mesh_path = tmp_path / "bottle.glb"
    trimesh.creation.box(extents=[0.06, 0.24, 0.06]).export(mesh_path)
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
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
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


@pytest.mark.parametrize(
    "call_id", ["simulation.coordinated_hold", "simulation.coordinated_transport"]
)
@pytest.mark.parametrize("max_episode_steps", [None, 5000])
def test_coordinated_bundle_composes_against_unmodified_public_options(
    tmp_path: Path, call_id: str, max_episode_steps: int | None
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
                    "world_displacement": [-0.1, 0.0, 0.05],
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
    assert load_config(paths.deployment)["max_episode_steps"] == (
        10000 if max_episode_steps is None else max_episode_steps
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


def test_generated_e2_bundle_shares_release_route_with_axis_acceptance(
    tmp_path: Path,
) -> None:
    """The merged E2 route passes real configured composition and preflight."""
    scene = _prepared_axis_scene(tmp_path)
    graph = _graph()
    graph["task_id"] = "upright"
    graph["targets"] = {
        name: {
            "kind": "cyclic_pose",
            "values": [
                {
                    "position": [0.1, -0.2, 0.85],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
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
                "resources": {"primary": "left"},
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
    assert generated["nodes"][0]["call"]["call_id"] == "gen_sim.pick.upright"
    assert "gen_sim.pick.upright" in integration["profile"]["action_options"]
    release = segments[-3]
    terminal = segments[-1]
    route = next(
        item
        for item in integration["runtime_services"]["registered_semantic_lowerers"]
        if item["kind"] == "place_relative"
    )["routes"][0]
    assert release["validators"][0]["kind"] == "object_near_relative_target"
    # Planning includes 1 cm release clearance; acceptance uses the support surface.
    expected_displacement = list(route["world_displacement"])
    expected_displacement[2] -= 0.01
    assert release["validators"][0]["displacement"] == pytest.approx(
        expected_displacement
    )
    constraints = load_config(paths.program.parent / "constraints.json")["presets"]
    preset = release["post"][-1]["preset"]
    assert constraints[preset]["kind"] == "upright"
    # The imported rotation is baked into the normalized mesh before binding.
    assert constraints[preset]["local_axis"] == [0.0, -0.0, 1.0]
    assert constraints[preset]["displacement"] == pytest.approx(expected_displacement)
    assert len(release["validators"]) == 1
    assert terminal["steps"]["call"]["call_id"] == "simulation.park"
    assert terminal["post"] == [release["post"][-1]]
    assert terminal["validators"] == release["validators"]
    assert set(release["steps"]["call"]["arguments"]) == {
        "object",
        "reference",
        "relation",
    }
    assert generated["integration_fingerprint"] != "0" * 64
