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
import struct

import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline import ur5_basket_config_generation
from embodichain.gen_sim.action_agent_pipeline.ur5_basket_config_generation import (
    TargetReplacementSpec,
    generate_ur5_basket_config_from_project,
)
from embodichain.lab.gym.envs.tasks.tableware.configurable_success import (
    evaluate_configured_success,
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
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}

    assert set(rigid_objects) == {"left_apple", "right_apple", "wicker_basket"}
    assert rigid_objects["left_apple"]["body_scale"] == [0.6, 0.6, 0.6]
    assert rigid_objects["right_apple"]["body_scale"] == [0.6, 0.6, 0.6]
    assert rigid_objects["wicker_basket"]["body_scale"] == [1.0, 1.0, 1.0]
    assert background_objects["table"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["left_apple"]["shape"]["fpath"].endswith(
        "mesh_assets/apple/apple_2/apple_2.glb"
    )
    assert rigid_objects["right_apple"]["shape"]["fpath"].endswith(
        "mesh_assets/apple/apple_1/apple_1.glb"
    )
    assert gym_config["robot"]["init_pos"] == [-2.0, 0.0, 0.5]
    assert gym_config["robot"]["init_rot"] == [0.0, 0.0, 90.0]

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
    assert "negative-y side" in basic_background
    assert "positive-y side" in basic_background
    assert "negative-x side" not in basic_background
    assert "positive-x side" not in basic_background
    assert "x_offset=0.0, y_offset=-0.04" in task_prompt
    assert "x_offset=0.0, y_offset=0.04" in task_prompt
    assert "x_offset=-0.04, y_offset=0.0" not in task_prompt
    assert "x_offset=0.04, y_offset=0.0" not in task_prompt
    assert "x_offset=0.0, y_offset=-0.04" in atom_actions
    assert "x_offset=0.0, y_offset=0.04" in atom_actions
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
    assert paths.summary["mode"] == "basket_template"


def test_target_replacements_generate_meshes_and_replace_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)
    calls = _patch_prompt2geometry(monkeypatch)

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_agent",
        target_replacements=[
            TargetReplacementSpec("apple_1", "A orange", "new1"),
            TargetReplacementSpec("apple_2", "A apple", "new2"),
        ],
    )

    assert calls == [
        ("A orange", project_dir / "mesh_assets" / "new1", "orange.glb"),
        ("A apple", project_dir / "mesh_assets" / "new2", "apple.glb"),
    ]

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}

    assert set(rigid_objects) == {"left_apple", "right_apple", "wicker_basket"}
    assert rigid_objects["right_apple"]["shape"]["fpath"].endswith(
        "mesh_assets/new1/orange.glb"
    )
    assert rigid_objects["left_apple"]["shape"]["fpath"].endswith(
        "mesh_assets/new2/apple.glb"
    )
    assert paths.summary["target_replacements"][0]["source_uid"] == "apple_1"
    assert paths.summary["target_replacements"][1]["source_uid"] == "apple_2"


def test_target_replacements_can_sync_runtime_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)
    _patch_prompt2geometry(monkeypatch)

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_agent",
        target_replacements=[
            TargetReplacementSpec("apple_2", "A orange", "new1"),
            TargetReplacementSpec("apple_1", "A apple", "new2"),
        ],
        sync_replacement_names=True,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}

    assert set(rigid_objects) == {"left_orange", "right_apple", "wicker_basket"}
    assert rigid_objects["left_orange"]["shape"]["fpath"].endswith(
        "mesh_assets/new1/orange.glb"
    )
    assert rigid_objects["right_apple"]["shape"]["fpath"].endswith(
        "mesh_assets/new2/apple.glb"
    )

    success_terms = gym_config["env"]["extensions"]["agent_success"]["terms"]
    assert {term["object"] for term in success_terms} == {
        "left_orange",
        "right_apple",
    }

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    basic_background = paths.basic_background.read_text(encoding="utf-8")
    assert "the left orange and right apple into the wicker_basket" in task_prompt
    assert "left_arm must only manipulate `left_orange`" in task_prompt
    assert "- left_orange: the orange mesh initially" in basic_background
    assert "- right_apple: the apple mesh initially" in basic_background


def test_directory_input_prefers_merged_config_and_preserves_extra_scene_scale(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)
    background_mesh = project_dir / "mesh_assets/backgrounds/vase_0.glb"
    background_mesh.parent.mkdir(parents=True, exist_ok=True)
    background_mesh.write_bytes(b"")

    merged_config_path = project_dir / "gym_config_merged.json"
    source_config = json.loads(
        (project_dir / "gym_config.json").read_text(encoding="utf-8")
    )
    extra_scene_object = _mesh_object(
        "vase_0",
        "mesh_assets/backgrounds/vase_0.glb",
        [0.16, -0.44, 0.77],
        [0.0, 0.0, -90.0],
    )
    extra_scene_object["body_scale"] = [1.2, 1.1, 0.9]
    source_config["rigid_object"].append(extra_scene_object)
    merged_config_path.write_text(
        json.dumps(source_config, indent=2),
        encoding="utf-8",
    )

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_agent",
        target_body_scale=0.8,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}

    assert set(rigid_objects) == {
        "left_apple",
        "right_apple",
        "wicker_basket",
        "vase_0",
    }
    assert rigid_objects["left_apple"]["body_scale"] == [0.8, 0.8, 0.8]
    assert rigid_objects["right_apple"]["body_scale"] == [0.8, 0.8, 0.8]
    assert rigid_objects["vase_0"]["body_scale"] == [1.2, 1.1, 0.9]
    assert rigid_objects["vase_0"]["shape"]["fpath"].endswith(
        "mesh_assets/backgrounds/vase_0.glb"
    )


def test_task_description_generates_relative_left_of_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        assert kwargs["task_description"] == "把 apple_2 放到 basket_3 左边"
        return {
            "moved_object": "apple_2",
            "reference_object": "basket_3",
            "goal_relation": "left_of",
            "task_prompt_summary": "Move apple_2 to the left of basket_3.",
            "basic_background_notes": "The basket is the spatial reference.",
            "action_sketch": [
                "grasp apple_2",
                "move to the left side of basket_3",
                "release on the table",
            ],
        }

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_relative_agent",
        task_name="AppleLeftOfBasket",
        task_description="把 apple_2 放到 basket_3 左边",
        target_body_scale=0.5,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}
    assert set(rigid_objects) == {"apple_1", "apple_2", "wicker_basket"}
    assert rigid_objects["apple_2"]["body_scale"] == [0.5, 0.5, 0.5]
    assert rigid_objects["apple_1"]["body_scale"] == [0.5, 0.5, 0.5]
    assert rigid_objects["wicker_basket"]["body_scale"] == [1.0, 1.0, 1.0]
    assert background_objects["table"]["body_scale"] == [1.0, 1.0, 1.0]

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["op"] == "all"
    axis_terms = {
        (term.get("axis"), term.get("offset"))
        for term in success["terms"]
        if term["type"] == "object_axis_offset_near"
    }
    assert ("y", -0.16) in axis_terms
    assert ("x", 0.0) in axis_terms

    grasp_overrides = gym_config["env"]["extensions"]["agent_grasp_pose_overrides"]
    assert grasp_overrides == [
        {
            "type": "top_down",
            "object": "apple_2",
            "side": "left",
            "height_offset": 0.036,
        }
    ]

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert "Move apple_2 to the left of basket_3." in task_prompt
    assert (
        "Generate one deterministic nominal graph with exactly 6 nominal edges"
        in task_prompt
    )
    assert 'grasp(robot_name="left_arm",\n     obj_name="apple_2"' in task_prompt
    assert "right_arm_action: null" in task_prompt
    assert "Generate exactly 10 nominal edges" not in task_prompt

    assert paths.summary == {
        "mode": "relative_placement",
        "moved_object": "apple_2",
        "reference_object": "wicker_basket",
        "relation": "left_of",
        "active_arm": "left_arm",
        "release_offset": [0.0, -0.16, 0.12],
    }


def test_task_description_generates_relative_front_of_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        assert kwargs["task_description"] == "用右臂把 apple_1 放到 apple_2 前边"
        return {
            "moved_object": "apple_1",
            "reference_object": "apple_2",
            "goal_relation": "front_of",
            "arm": "right",
            "task_prompt_summary": "Move apple_1 in front of apple_2.",
            "basic_background_notes": "The apple_2 object is the spatial reference.",
        }

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_front_relative_agent",
        task_name="AppleFrontOfApple",
        task_description="用右臂把 apple_1 放到 apple_2 前边",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["op"] == "all"
    axis_terms = {
        (term.get("axis"), term.get("offset"))
        for term in success["terms"]
        if term["type"] == "object_axis_offset_near"
    }
    assert ("x", -0.16) in axis_terms
    assert ("y", 0.0) in axis_terms

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert "x_offset=-0.16" in task_prompt
    assert "y_offset=0" in task_prompt
    assert "x_offset=-0.16" in atom_actions
    assert "y_offset=0" in atom_actions

    assert paths.summary == {
        "mode": "relative_placement",
        "moved_object": "apple_1",
        "reference_object": "apple_2",
        "relation": "front_of",
        "active_arm": "right_arm",
        "release_offset": [-0.16, 0.0, 0.12],
    }


def test_task_description_on_container_is_compiled_as_inside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "apple_1",
            "reference_object": "basket_3",
            "goal_relation": "on",
            "task_prompt_summary": "Release apple_1 above basket_3.",
        }

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_above_container_agent",
        task_description="把 apple_1 放到 basket_3 上方然后松手",
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["type"] == "object_in_container"
    assert success["object"] == "apple_1"
    assert success["container"] == "wicker_basket"
    assert paths.summary["relation"] == "inside"
    assert paths.summary["active_arm"] == "right_arm"

    grasp_overrides = gym_config["env"]["extensions"]["agent_grasp_pose_overrides"]
    assert grasp_overrides == [
        {
            "type": "top_down",
            "object": "apple_1",
            "side": "right",
            "height_offset": 0.036,
        }
    ]


def test_task_description_respects_explicit_left_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "apple_1",
            "reference_object": "basket_3",
            "goal_relation": "left_of",
            "arm": "left",
            "task_prompt_summary": "Use the left arm to move apple_1.",
        }

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_left_arm_agent",
        task_description="左臂把 apple_1 放到 basket_3 左边",
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    grasp_overrides = gym_config["env"]["extensions"]["agent_grasp_pose_overrides"]
    assert grasp_overrides[0]["object"] == "apple_1"
    assert grasp_overrides[0]["side"] == "left"
    assert paths.summary["active_arm"] == "left_arm"

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert 'grasp(robot_name="left_arm",\n     obj_name="apple_1"' in task_prompt
    assert "right_arm_action: null" in task_prompt


def test_task_description_respects_explicit_right_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "apple_2",
            "reference_object": "basket_3",
            "goal_relation": "right_of",
            "arm": "right",
            "task_prompt_summary": "Use the right arm to move apple_2.",
        }

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_right_arm_agent",
        task_description="右臂把 apple_2 放到 basket_3 右边",
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    grasp_overrides = gym_config["env"]["extensions"]["agent_grasp_pose_overrides"]
    assert grasp_overrides[0]["object"] == "apple_2"
    assert grasp_overrides[0]["side"] == "right"
    assert paths.summary["active_arm"] == "right_arm"

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert 'grasp(robot_name="right_arm",\n     obj_name="apple_2"' in task_prompt
    assert "left_arm_action: null" in task_prompt


def test_task_description_generates_dual_arm_relative_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        assert kwargs["task_description"] == (
            "左臂把 apple_2 放到 basket_3 左边，右臂把 apple_1 放到 basket_3 右边"
        )
        return {
            "placements": [
                {
                    "moved_object": "apple_2",
                    "reference_object": "basket_3",
                    "goal_relation": "left_of",
                    "arm": "left",
                },
                {
                    "moved_object": "apple_1",
                    "reference_object": "basket_3",
                    "goal_relation": "right_of",
                    "arm": "right",
                },
            ],
            "task_prompt_summary": "Use both arms for two side placements.",
            "basic_background_notes": "Both arms have explicit work.",
        }

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_dual_relative_agent",
        task_description=(
            "左臂把 apple_2 放到 basket_3 左边，右臂把 apple_1 放到 basket_3 右边"
        ),
        target_body_scale=0.7,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    grasp_overrides = gym_config["env"]["extensions"]["agent_grasp_pose_overrides"]
    assert grasp_overrides == [
        {
            "type": "top_down",
            "object": "apple_2",
            "side": "left",
            "height_offset": 0.036,
        },
        {
            "type": "top_down",
            "object": "apple_1",
            "side": "right",
            "height_offset": 0.036,
        },
    ]

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["op"] == "all"
    assert len(success["terms"]) == 2
    axis_terms = {
        (term["object"], term["axis"], term["offset"])
        for placement_success in success["terms"]
        for term in placement_success["terms"]
        if term["type"] == "object_axis_offset_near"
    }
    assert ("apple_2", "y", -0.16) in axis_terms
    assert ("apple_1", "y", 0.16) in axis_terms

    grasp_pose_attr = next(
        attr
        for attr in gym_config["env"]["events"]["prepare_extra_attr"]["params"]["attrs"]
        if attr["name"] == "grasp_pose_object"
    )
    assert grasp_pose_attr["entity_uids"] == ["apple_2", "apple_1"]
    assert len(grasp_pose_attr["value"]) == 1

    assert paths.summary == {
        "mode": "dual_arm_relative_placement",
        "placements": [
            {
                "moved_object": "apple_2",
                "reference_object": "wicker_basket",
                "relation": "left_of",
                "active_arm": "left_arm",
                "release_offset": [0.0, -0.16, 0.12],
            },
            {
                "moved_object": "apple_1",
                "reference_object": "wicker_basket",
                "relation": "right_of",
                "active_arm": "right_arm",
                "release_offset": [0.0, 0.16, 0.12],
            },
        ],
    }

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    basic_background = paths.basic_background.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert "Generate one deterministic nominal graph with exactly 10 nominal edges" in (
        task_prompt
    )
    assert 'left_arm_action: grasp(robot_name="left_arm"' in task_prompt
    assert 'right_arm_action: grasp(robot_name="right_arm"' in task_prompt
    assert 'close_gripper(robot_name="right_arm", sample_num=10)' in task_prompt
    assert "The inactive arm must remain null" not in task_prompt
    assert "Both arms participate" in basic_background
    assert "left_arm moves `apple_2`" in basic_background
    assert "right_arm moves `apple_1`" in basic_background
    assert 'grasp(robot_name="left_arm", obj_name="apple_2"' in atom_actions
    assert 'grasp(robot_name="right_arm", obj_name="apple_1"' in atom_actions


def test_task_description_rejects_dual_relative_same_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "placements": [
                {
                    "moved_object": "apple_2",
                    "reference_object": "basket_3",
                    "goal_relation": "left_of",
                    "arm": "left",
                },
                {
                    "moved_object": "apple_1",
                    "reference_object": "basket_3",
                    "goal_relation": "right_of",
                    "arm": "left",
                },
            ],
        }

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    with pytest.raises(ValueError, match="one left arm and one right arm"):
        generate_ur5_basket_config_from_project(
            project_dir,
            tmp_path / "bad_dual_relative_agent",
            task_description="双臂分别移动两个苹果",
        )


def test_task_description_dual_auto_assigns_complementary_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    gym_config_path = project_dir / "gym_config.json"
    source_config = json.loads(gym_config_path.read_text(encoding="utf-8"))
    for obj_config in source_config["rigid_object"]:
        if obj_config["uid"] == "apple_1":
            obj_config["init_pos"][1] = -0.03
    gym_config_path.write_text(
        json.dumps(source_config, indent=2),
        encoding="utf-8",
    )

    def fake_call_relative_task_llm(**kwargs):
        return {
            "placements": [
                {
                    "moved_object": "apple_2",
                    "reference_object": "basket_3",
                    "goal_relation": "left_of",
                    "arm": "auto",
                },
                {
                    "moved_object": "apple_1",
                    "reference_object": "basket_3",
                    "goal_relation": "right_of",
                    "arm": "auto",
                },
            ],
        }

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_dual_auto_relative_agent",
        task_description="双臂分别移动两个苹果",
        prewarm_coacd_cache=False,
    )

    active_arms = [placement["active_arm"] for placement in paths.summary["placements"]]
    assert active_arms == ["left_arm", "right_arm"]

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    grasp_overrides = gym_config["env"]["extensions"]["agent_grasp_pose_overrides"]
    assert [override["side"] for override in grasp_overrides] == ["left", "right"]


def test_task_description_on_object_uses_object_on_object_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "apple_2",
            "reference_object": "apple_1",
            "goal_relation": "on",
            "task_prompt_summary": "Stack apple_2 on apple_1.",
        }

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_stack_agent",
        task_description="把 apple_2 放到 apple_1 上方并松手",
        target_body_scale=0.6,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    assert rigid_objects["apple_2"]["body_scale"] == [0.6, 0.6, 0.6]
    assert rigid_objects["apple_1"]["body_scale"] == [0.6, 0.6, 0.6]
    assert rigid_objects["wicker_basket"]["body_scale"] == [1.0, 1.0, 1.0]

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["type"] == "object_on_object"
    assert success["object"] == "apple_2"
    assert success["support"] == "apple_1"

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert "on top of `apple_1`" in task_prompt


def test_task_description_rejects_unknown_llm_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "missing_bread",
            "reference_object": "basket_3",
            "goal_relation": "left_of",
        }

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    with pytest.raises(ValueError, match="unknown moved_object"):
        generate_ur5_basket_config_from_project(
            project_dir,
            tmp_path / "bad_agent",
            task_description="把 missing_bread 放到 basket_3 左边",
        )


def test_high_tabletop_scene_adjusts_robot_height_and_light(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    gym_config_path = project_dir / "gym_config.json"
    source_config = json.loads(gym_config_path.read_text(encoding="utf-8"))
    for obj_config in source_config["rigid_object"]:
        obj_config["init_pos"][2] = 0.12
    gym_config_path.write_text(
        json.dumps(source_config, indent=2),
        encoding="utf-8",
    )

    def fake_resolve_table_mesh_world_zmax(
        scene_dir: Path,
        table_obj,
    ) -> float:
        assert scene_dir == project_dir
        assert table_obj.source_uid == "table"
        return 1.18

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_resolve_table_mesh_world_zmax",
        fake_resolve_table_mesh_world_zmax,
    )

    paths = generate_ur5_basket_config_from_project(
        project_dir,
        tmp_path / "generated_high_table_agent",
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    expected_init_z = (
        1.18
        + ur5_basket_config_generation._DUAL_UR5_TABLETOP_CLEARANCE
        - ur5_basket_config_generation._DUAL_UR5_ARM_COMPONENT_Z
    )
    assert gym_config["robot"]["init_pos"][2] == pytest.approx(
        expected_init_z
    )
    assert gym_config["light"]["direct"][0]["intensity"] == 40.0


def test_table_mesh_world_zmax_reads_glb_vertices(tmp_path: Path) -> None:
    scene_dir = tmp_path / "1790000000_gym_project"
    mesh_path = scene_dir / "mesh_assets/table/table_0.glb"
    _write_minimal_glb(
        mesh_path,
        [(-0.5, -0.5, 0.0), (0.5, -0.5, 1.2), (0.0, 0.5, 0.4)],
    )
    table_obj = ur5_basket_config_generation._SceneObject(
        source_uid="table",
        source_role="background",
        config=_mesh_object(
            "table",
            "mesh_assets/table/table_0.glb",
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0],
        ),
    )
    table_obj.config["body_scale"] = [1.0, 1.0, 2.0]

    assert ur5_basket_config_generation._resolve_table_mesh_world_zmax(
        scene_dir,
        table_obj,
    ) == pytest.approx(2.5)


def test_object_on_object_success_predicate() -> None:
    env = _FakeEnv(
        {
            "apple_2": [0.0, 0.0, 0.15],
            "apple_1": [0.02, 0.01, 0.0],
        }
    )

    success = evaluate_configured_success(
        env,
        {
            "type": "object_on_object",
            "object": "apple_2",
            "support": "apple_1",
            "xy_radius": 0.08,
            "min_z_offset": 0.02,
            "max_z_offset": 0.35,
        },
    )

    assert bool(success.item()) is True


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
                [0.38, 0.11, 0.76],
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


def _write_minimal_glb(
    path: Path,
    vertices: list[tuple[float, float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = b"".join(struct.pack("<fff", *vertex) for vertex in vertices)
    mins = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maxs = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(binary),
                "target": 34962,
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": mins,
                "max": maxs,
            }
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_chunk = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    binary_chunk = binary + b"\x00" * ((4 - len(binary) % 4) % 4)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary_chunk), 0x004E4942)
        + binary_chunk
    )


def _patch_prompt2geometry(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []

    def fake_run_prompt2geometry_replacement(
        *,
        prompt: str,
        output_root: Path,
        output_name: str,
    ) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        mesh_path = output_root / output_name
        mesh_path.write_bytes(b"glb")
        calls.append((prompt, output_root, output_name))
        return {"scaled_mesh_path": str(mesh_path)}

    monkeypatch.setattr(
        ur5_basket_config_generation,
        "_run_prompt2geometry_replacement",
        fake_run_prompt2geometry_replacement,
    )
    return calls


class _FakeEnv:
    num_envs = 1
    device = torch.device("cpu")

    def __init__(self, positions: dict[str, list[float]]) -> None:
        self.sim = _FakeSim(positions)


class _FakeSim:
    def __init__(self, positions: dict[str, list[float]]) -> None:
        self._objects = {
            uid: _FakeRigidObject(position) for uid, position in positions.items()
        }

    def get_rigid_object(self, uid: str):
        return self._objects[uid]


class _FakeRigidObject:
    def __init__(self, position: list[float]) -> None:
        self._position = torch.tensor(position, dtype=torch.float32)

    def get_local_pose(self, to_matrix: bool = True) -> torch.Tensor:
        pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        pose[:, :3, 3] = self._position.reshape(1, 3)
        return pose
