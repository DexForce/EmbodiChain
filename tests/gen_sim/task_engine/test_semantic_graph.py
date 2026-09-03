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
)
from embodichain.utils.utility import save_config


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
