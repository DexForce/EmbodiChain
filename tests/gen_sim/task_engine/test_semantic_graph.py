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
                "id": f"{kind}_cube",
                "call": {"kind": kind, "object": "cube", **extra},
                "depends_on": [] if kind == "pick" else ["pick_cube"],
                "task_instance_id": f"{kind}_group",
                "task_type": "E1",
                "role": "primary",
            }
            for kind, extra in (("pick", {}), ("place", {"inside": "tray_inside"}))
        ],
        "task_groups": [
            {
                "id": f"{group}_group",
                "task_type": "E1",
                "node_ids": [f"{group}_cube"],
                "depends_on": [] if group == "pick" else ["pick_group"],
                "success": (
                    {"kind": "call_completed"}
                    if group == "pick"
                    else {"kind": "object_inside", "object": "cube"}
                ),
            }
            for group in ("pick", "place")
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


def test_bundle_rejects_out_of_scope_tasks_before_writing_assets(
    tmp_path: Path,
) -> None:
    graph = _graph()
    for node in graph["nodes"]:
        node["task_type"] = "E7"
    for group in graph["task_groups"]:
        group["task_type"] = "E7"
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="E1-E6 and E8-E9"):
        generate_task_program_bundle(graph, None, output, robot_profile="dual_franka")
    assert not output.exists()


@pytest.mark.parametrize(
    "robot_profile,native_exploration",
    [
        ("dual_franka", False),
        ("dual_ur10", False),
        ("dual_tianji", False),
        ("dual_ur10", True),
    ],
)
def test_selected_robot_bundle_composes_and_keeps_identity(
    tmp_path: Path,
    robot_profile: str,
    native_exploration: bool,
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
    graph["task_groups"][1]["success"] = {"kind": "call_completed"}
    generated, paths = generate_task_program_bundle(
        graph,
        _prepared_axis_scene(tmp_path),
        tmp_path / "bundle",
        robot_profile=robot_profile,
        **({"native_pick_exploration": True} if native_exploration else {}),
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
    assert (
        getattr(
            integration.integration.adapter_factory, "native_pick_exploration", False
        )
        is native_exploration
    )
    embodiment = load_config(paths.embodiment)
    prefix = "tianji_marvin" if robot_profile == "dual_tianji" else robot_profile
    assert embodiment["embodiment_id"].startswith(prefix)
    assert len(embodiment["simulation"]["control_parts"]["left_arm"]) == (
        6 if robot_profile == "dual_ur10" else 7
    )
    if robot_profile == "dual_ur10":
        # The upper-wrist seed avoids the below-table branch at low flanges.
        simulation = embodiment["simulation"]
        assert simulation["init_qpos"][8:10] == [pytest.approx(1.5707963267948966)] * 2
        presets = embodiment["skill_profile"]["command_presets"]
        parks = [
            preset["commands"]["park"]
            for preset in presets
            if preset["preset_id"] in {"left_arm_postures", "right_arm_postures"}
        ]
        assert len(parks) == 2
        assert all(park == simulation["init_qpos"][:12:2] for park in parks)


@pytest.mark.parametrize("profile", ["dual_franka", "dual_tianji"])
def test_native_pick_exploration_rejects_other_robots(
    tmp_path: Path, profile: str
) -> None:
    with pytest.raises(ValueError, match="UR10"):
        generate_task_program_bundle(
            _graph(),
            None,
            tmp_path / "bundle",
            robot_profile=profile,
            native_pick_exploration=True,
        )


def _prepared_axis_scene(tmp_path: Path) -> PreparedScene:
    mesh_path = tmp_path / "bottle.glb"
    trimesh.creation.box(extents=(0.06, 0.24, 0.06)).export(mesh_path)
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
    return PreparedScene(
        source_config_path=tmp_path / "scene.yaml",
        scene_dir=tmp_path,
        planner_objects=tuple(
            {**item, "runtime_uid": item["uid"], "role": role}
            for item, role in ((bottle, "rigid_object"), (table, "background"))
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


@pytest.mark.parametrize("object_id", ["apple", "renamed_object"])
def test_container_without_measured_floor_never_uses_name_fallback(
    tmp_path: Path, object_id: str
) -> None:
    from embodichain.gen_sim.task_engine.task_program_bundle import _integration_payload

    scene = _prepared_axis_scene(tmp_path)
    scene.planner_objects[0]["runtime_uid"] = object_id
    graph = _graph()
    for node in graph["nodes"]:
        node["call"]["object"] = object_id
    graph["nodes"][1]["call"]["inside"] = f"inside__table__{object_id}"
    with pytest.raises(ValueError, match="measured container floor"):
        _integration_payload(graph, scene, program_id="test", scene_contract="test")
