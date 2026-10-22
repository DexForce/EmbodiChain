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
import trimesh
import pytest
from embodichain.gen_sim.task_engine._bundle_runner import (
    _verify_integration_fingerprint,
    _verify_program_projection,
)
from embodichain.gen_sim.task_engine.semantic_graph import validate_semantic_task_graph
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
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


@pytest.mark.parametrize("task_type", ["E7"])
def test_bundle_rejects_out_of_scope_tasks_before_writing_assets(
    tmp_path: Path, task_type: str
) -> None:
    graph = _graph()
    for node in graph["nodes"]:
        node["task_type"] = task_type
    for group in graph["task_groups"]:
        group["task_type"] = task_type
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="E1-E6 and E8-E9"):
        generate_task_program_bundle(graph, None, output, robot_profile="dual_franka")
    assert not output.exists()


@pytest.mark.parametrize("robot_profile", ["dual_franka", "dual_ur10"])
def test_selected_robot_bundle_composes_and_keeps_identity(
    tmp_path: Path,
    robot_profile: str,
) -> None:
    graph = _graph()
    graph["nodes"][0]["call"] = {
        "kind": "pick",
        "object": "bottle",
        "resources": {"primary": "left"},
    }
    graph["nodes"][1]["call"] = {
        "kind": "registered",
        "call_id": "simulation.place_relative",
        "arguments": {"object": "bottle", "reference": "table", "relation": "on"},
        "resources": {"primary": "left"},
    }
    for group in graph["task_groups"]:
        group["success"] = {"kind": "call_completed"}
    generated, paths = generate_task_program_bundle(
        graph,
        _prepared_axis_scene(tmp_path),
        tmp_path / "bundle",
        robot_profile=robot_profile,
    )
    record = load_config(paths.integration_fingerprint)
    _verify_program_projection(paths.program, generated)
    integration = _verify_integration_fingerprint(
        paths.root,
        paths.deployment,
        generated,
        record,
    )
    integration.integration.registration.assert_unchanged()
    embodiment = load_config(paths.embodiment)
    assert embodiment["embodiment_id"].startswith(robot_profile)
    assert len(embodiment["simulation"]["control_parts"]["left_arm"]) == (
        6 if robot_profile == "dual_ur10" else 7
    )


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
