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

from pathlib import Path
import json

from embodichain.gen_sim.action_agent_pipeline.ur5_basket_config_generation import (
    generate_ur5_basket_config_from_project,
)


def test_ur5_basket_generator_uses_parallel_handoff(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_agent",
        target_body_scale=0.6,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}

    assert set(rigid_objects) == {"left_apple", "right_apple", "wicker_basket"}
    assert rigid_objects["left_apple"]["body_scale"] == [0.6, 0.6, 0.6]
    assert rigid_objects["right_apple"]["body_scale"] == [0.6, 0.6, 0.6]

    success_terms = gym_config["env"]["extensions"]["agent_success"]["terms"]
    assert {term["object"] for term in success_terms} == {"left_apple", "right_apple"}
    assert {term["container"] for term in success_terms} == {"wicker_basket"}

    registry = gym_config["env"]["events"]["register_info_to_env"]["params"]["registry"]
    registered_uids = {entry["entity_cfg"]["uid"] for entry in registry}
    assert registered_uids == {"left_apple", "right_apple", "wicker_basket"}

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    basic_background = paths.basic_background.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    normalized_task_prompt = " ".join(task_prompt.split())

    assert "Generate exactly 10 nominal edges" in normalized_task_prompt
    assert "Generate exactly 11 nominal edges" not in normalized_task_prompt
    assert "parallel handoff" in task_prompt
    assert "parallel handoff" in basic_background
    assert "parallel handoff" in atom_actions

    handoff_edge = task_prompt.split("6. After the left gripper", maxsplit=1)[1].split(
        "\n7. Lower the held right target object",
        maxsplit=1,
    )[0]
    assert 'back_to_initial_pose(robot_name="left_arm"' in handoff_edge
    assert 'move_relative_to_object(robot_name="right_arm"' in handoff_edge
    assert 'close_gripper(robot_name="right_arm"' not in handoff_edge
    assert "left_arm_action: null" not in handoff_edge


def _write_project(project_dir: Path) -> None:
    for rel_path in (
        "mesh_assets/table/table_0.glb",
        "mesh_assets/basket/basket_3/basket_3.glb",
        "mesh_assets/apple/apple_1/apple_1.glb",
        "mesh_assets/apple/apple_2/apple_2.glb",
    ):
        mesh_path = project_dir / rel_path
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.write_bytes(b"")

    gym_config = {
        "id": "Image2Tabletop-1790000000-v0",
        "background": [
            _mesh_object(
                "table",
                "mesh_assets/table/table_0.glb",
                [0.0, 0.0, 0.36],
                [0.0, 0.0, 180.0],
            )
        ],
        "rigid_object": [
            _mesh_object(
                "basket_3",
                "mesh_assets/basket/basket_3/basket_3.glb",
                [0.0, 0.08, 0.75],
                [0.0, 0.0, 180.0],
            ),
            _mesh_object(
                "apple_1",
                "mesh_assets/apple/apple_1/apple_1.glb",
                [0.38, -0.11, 0.76],
                [0.0, 0.0, 140.0],
            ),
            _mesh_object(
                "apple_2",
                "mesh_assets/apple/apple_2/apple_2.glb",
                [-0.39, -0.12, 0.76],
                [0.0, 0.0, 160.0],
            ),
        ],
    }
    (project_dir / "gym_config.json").write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )


def _mesh_object(
    uid: str,
    fpath: str,
    init_pos: list[float],
    init_rot: list[float],
) -> dict:
    return {
        "uid": uid,
        "shape": {
            "shape_type": "Mesh",
            "fpath": fpath,
            "compute_uv": False,
        },
        "init_pos": init_pos,
        "init_rot": init_rot,
        "body_scale": [1.0, 1.0, 1.0],
    }
