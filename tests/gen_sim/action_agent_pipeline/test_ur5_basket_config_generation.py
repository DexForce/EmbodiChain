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
import base64
import hashlib
import json
import struct

import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline.cli import (
    target_replacements as target_replacements_cli,
)
from embodichain.gen_sim.action_agent_pipeline.generation import (
    action_agent_config as action_agent_config_generation,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_templates import (
    make_dual_ur5_robot_config,
    make_light_config,
    make_sensor_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_frame_normalization import (
    MESH_FRAME_NORMALIZATION_POLICY_VERSION,
    MeshFrameNormalizer,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_config import (
    TargetReplacementSpec,
    generate_action_agent_config_from_project,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_spec import (
    _apply_arrangement_task_response,
    _arrangement_line_slot_positions,
)
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.success import (
    evaluate_configured_success,
)


def test_action_agent_templates_load_fresh_json_copies() -> None:
    first_robot = make_dual_ur5_robot_config(robot_init_z=0.42)
    second_robot = make_dual_ur5_robot_config(robot_init_z=0.84)
    first_sensors = make_sensor_config()
    second_sensors = make_sensor_config()
    first_lights = make_light_config()
    second_lights = make_light_config()

    first_robot["control_parts"]["left_arm"].append("MUTATED_JOINT")
    first_sensors[0]["uid"] = "mutated_camera"
    first_lights["direct"][0]["uid"] = "mutated_light"

    assert second_robot["init_pos"] == pytest.approx([-2.0, 0.0, 0.84])
    assert first_robot["init_pos"] == pytest.approx([-2.0, 0.0, 0.42])
    assert second_robot["control_parts"]["left_arm"] == ["LEFT_JOINT[1-6]"]
    assert second_sensors[0]["uid"] == "cam_high"
    assert second_lights["direct"][0]["uid"] == "main_light"


def test_action_agent_config_generator_uses_parallel_handoff(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_agent",
        target_body_scale=0.6,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}

    assert set(rigid_objects) == {"left_apple", "right_apple"}
    assert rigid_objects["left_apple"]["body_scale"] == [0.6, 0.6, 0.6]
    assert rigid_objects["right_apple"]["body_scale"] == [0.6, 0.6, 0.6]
    assert rigid_objects["left_apple"]["body_type"] == "dynamic"
    assert rigid_objects["right_apple"]["body_type"] == "dynamic"
    assert background_objects["table"]["body_scale"] == [1.0, 1.0, 1.0]
    assert background_objects["wicker_basket"]["body_scale"] == [1.0, 1.0, 1.0]
    assert background_objects["wicker_basket"]["body_type"] == "kinematic"
    _assert_normalized_obj_path(rigid_objects["left_apple"]["shape"]["fpath"])
    _assert_normalized_obj_path(rigid_objects["right_apple"]["shape"]["fpath"])
    _assert_normalized_obj_path(background_objects["table"]["shape"]["fpath"])
    _assert_normalized_obj_path(background_objects["wicker_basket"]["shape"]["fpath"])
    table_top_z = action_agent_config_generation._mesh_config_world_zmax(
        background_objects["table"]
    )
    expected_robot_init_z = (
        table_top_z
        + action_agent_config_generation._DUAL_UR5_TABLETOP_CLEARANCE
        - action_agent_config_generation._DUAL_UR5_ARM_COMPONENT_Z
    )
    assert gym_config["robot"]["init_pos"] == pytest.approx(
        [-2.0, 0.0, expected_robot_init_z]
    )
    assert gym_config["robot"]["init_rot"] == [0.0, 0.0, 90.0]
    extensions = gym_config["env"]["extensions"]
    assert extensions["agent_arm_slots"]["left"] == {
        "arm": "right_arm",
        "eef": "right_eef",
    }
    assert extensions["agent_arm_slots"]["right"] == {
        "arm": "left_arm",
        "eef": "left_eef",
    }
    assert extensions["arm_aim_yaw_offset"]["left"] == pytest.approx(3.141592653589793)
    assert extensions["arm_aim_yaw_offset"]["right"] == pytest.approx(0.0)

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

    assert "Generate exactly 6 nominal edges" in normalized_task_prompt
    assert "Generate exactly 10 nominal edges" not in normalized_task_prompt
    assert "positive-y side" in basic_background
    assert "negative-y side" in basic_background
    assert "negative-x side" not in basic_background
    assert "positive-x side" not in basic_background
    left_high_offset_spec = (
        '"atomic_action_class":"MoveHeldObject","robot_name":"left_arm",'
        '"control":"arm","target_object_pose":{"reference":"object",'
        '"obj_name":"wicker_basket","offset":[0.0,0.04,0.22],'
        '"orientation_goal":"preserve","orientation_axis":"none"}'
    )
    right_high_offset_spec = (
        '"atomic_action_class":"MoveHeldObject","robot_name":"right_arm",'
        '"control":"arm","target_object_pose":{"reference":"object",'
        '"obj_name":"wicker_basket","offset":[0.0,-0.04,0.22],'
        '"orientation_goal":"preserve","orientation_axis":"none"}'
    )
    assert left_high_offset_spec in task_prompt
    assert right_high_offset_spec in task_prompt
    assert (
        '"atomic_action_class":"Place","robot_name":"left_arm","control":"arm",'
        '"target_pose":{"reference":"object","obj_name":"wicker_basket",'
        '"offset":[0.0,0.04,0.12]}' in task_prompt
    )
    assert (
        '"atomic_action_class":"Place","robot_name":"right_arm","control":"arm",'
        '"target_pose":{"reference":"object","obj_name":"wicker_basket",'
        '"offset":[0.0,-0.04,0.12]}' in task_prompt
    )
    assert '"offset":[-0.04,0.0,0.22]' not in task_prompt
    assert '"offset":[0.04,0.0,0.22]' not in task_prompt
    assert left_high_offset_spec in atom_actions
    assert right_high_offset_spec in atom_actions
    assert "parallel handoff" in task_prompt
    assert "parallel handoff" in basic_background
    assert "parallel handoff" in atom_actions
    assert len(paths.summary["normalized_meshes"]) == 4

    handoff_edge = task_prompt.split("4. After the left gripper", maxsplit=1)[1].split(
        "\n5. Place the held right target object",
        maxsplit=1,
    )[0]
    assert (
        '"atomic_action_class":"MoveJoints","robot_name":"left_arm","control":"arm",'
        '"target_qpos":{"source":"initial"}' in handoff_edge
    )
    assert (
        '"atomic_action_class":"MoveHeldObject","robot_name":"right_arm",'
        '"control":"arm","target_object_pose":{"reference":"object"' in handoff_edge
    )
    assert '"state":"close"' not in handoff_edge
    assert "left_arm_action: null" not in handoff_edge
    assert paths.summary["mode"] == "basket_template"


def test_generator_normalizes_glb_meshes_and_preserves_source_rot(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_agent",
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}

    assert background_objects["table"]["init_rot"] == [0.0, 0.0, 180.0]
    assert background_objects["wicker_basket"]["init_rot"] == [0.0, 0.0, 180.0]
    assert rigid_objects["left_apple"]["init_rot"] == [0.0, 0.0, 140.0]
    assert rigid_objects["right_apple"]["init_rot"] == [0.0, 0.0, 160.0]
    for obj_config in [
        background_objects["table"],
        background_objects["wicker_basket"],
        rigid_objects["right_apple"],
        rigid_objects["left_apple"],
    ]:
        _assert_normalized_obj_path(obj_config["shape"]["fpath"])

    source_paths = {
        Path(entry["source_path"]).name for entry in paths.summary["normalized_meshes"]
    }
    assert source_paths == {
        "table_0.glb",
        "basket_3.glb",
        "apple_1.glb",
        "apple_2.glb",
    }


def test_mesh_frame_normalizer_bakes_glb_scene_transform_to_obj(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "source" / "triangle.glb"
    _write_minimal_glb(
        mesh_path,
        [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        node_translation=(1.0, 0.0, 0.0),
    )
    source_sha256 = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    normalizer = MeshFrameNormalizer(output_dir=tmp_path / "normalized")

    normalized_path = normalizer.normalize_path(mesh_path)
    repeated_path = normalizer.normalize_path(mesh_path)

    assert repeated_path == normalized_path
    assert normalized_path.suffix == ".obj"
    assert MESH_FRAME_NORMALIZATION_POLICY_VERSION not in normalized_path.name
    assert len(normalized_path.name) <= 64
    obj_text = normalized_path.read_text(encoding="utf-8")
    assert f"policy_version: {MESH_FRAME_NORMALIZATION_POLICY_VERSION}" in obj_text
    assert f"source_sha256: {source_sha256}" in obj_text
    assert "dexsim_engine_version:" in obj_text
    assert (
        "transform: [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],"
        "[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]"
    ) in obj_text
    assert "mtllib material.mtl" in obj_text
    material_text = (normalized_path.parent / "material.mtl").read_text(
        encoding="utf-8"
    )
    material_name = _single_obj_material_name(obj_text)
    assert material_name != "material_0"
    assert f"newmtl {material_name}" in material_text
    assert "map_Kd " not in material_text
    assert _rounded_vertex_set(_obj_vertices(normalized_path)) == {
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
    }


def test_mesh_frame_normalizer_extracts_embedded_base_color_texture(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "source" / "textured_triangle.glb"
    texture_png = _tiny_png()
    _write_minimal_glb(
        mesh_path,
        _default_mesh_vertices(),
        embedded_base_color_png=texture_png,
    )
    output_dir = tmp_path / "normalized"

    normalized_path = MeshFrameNormalizer(output_dir=output_dir).normalize_path(
        mesh_path
    )

    obj_text = normalized_path.read_text(encoding="utf-8")
    material_name = _single_obj_material_name(obj_text)
    material_text = (output_dir / "material.mtl").read_text(encoding="utf-8")
    assert f"newmtl {material_name}" in material_text
    assert "Kd 1.0 1.0 1.0" in material_text
    map_kd = _single_map_kd_path(material_text, material_name)
    assert map_kd.startswith("textures/")
    assert map_kd.endswith("_basecolor.png")
    assert (output_dir / map_kd).read_bytes() == texture_png

    material_path = output_dir / "material.mtl"
    texture_path = output_dir / map_kd
    material_path.unlink()
    texture_path.unlink()

    reused_path = MeshFrameNormalizer(output_dir=output_dir).normalize_path(mesh_path)

    assert reused_path == normalized_path
    assert material_path.is_file()
    assert texture_path.read_bytes() == texture_png


def test_mesh_frame_normalizer_recreates_material_library_for_reused_obj(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "source" / "triangle.glb"
    _write_minimal_glb(mesh_path, _default_mesh_vertices())
    output_dir = tmp_path / "normalized"
    normalized_path = MeshFrameNormalizer(output_dir=output_dir).normalize_path(
        mesh_path
    )
    material_path = normalized_path.parent / "material.mtl"
    material_path.unlink()

    reused_path = MeshFrameNormalizer(output_dir=output_dir).normalize_path(mesh_path)

    assert reused_path == normalized_path
    assert material_path.is_file()
    material_text = material_path.read_text(encoding="utf-8")
    reused_material_name = _single_obj_material_name(
        reused_path.read_text(encoding="utf-8")
    )
    assert f"newmtl {reused_material_name}" in material_text


def test_target_replacements_generate_meshes_and_replace_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)
    calls = _patch_prompt2geometry(monkeypatch)

    paths = generate_action_agent_config_from_project(
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

    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}

    assert set(rigid_objects) == {"left_apple", "right_apple"}
    assert "wicker_basket" in background_objects
    assert background_objects["wicker_basket"]["body_type"] == "kinematic"
    _assert_normalized_obj_path(rigid_objects["right_apple"]["shape"]["fpath"])
    _assert_normalized_obj_path(rigid_objects["left_apple"]["shape"]["fpath"])
    normalized_sources = {
        Path(entry["source_path"]).as_posix()
        for entry in paths.summary["normalized_meshes"]
    }
    assert (
        project_dir / "mesh_assets" / "new1" / "orange.glb"
    ).as_posix() in normalized_sources
    assert (
        project_dir / "mesh_assets" / "new2" / "apple.glb"
    ).as_posix() in normalized_sources
    assert paths.summary["target_replacements"][0]["source_uid"] == "apple_1"
    assert paths.summary["target_replacements"][1]["source_uid"] == "apple_2"


def test_target_replacements_can_sync_runtime_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)
    _patch_prompt2geometry(monkeypatch)

    paths = generate_action_agent_config_from_project(
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

    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}

    assert set(rigid_objects) == {"left_apple", "right_orange"}
    assert "wicker_basket" in background_objects
    assert background_objects["wicker_basket"]["body_type"] == "kinematic"
    _assert_normalized_obj_path(rigid_objects["left_apple"]["shape"]["fpath"])
    _assert_normalized_obj_path(rigid_objects["right_orange"]["shape"]["fpath"])

    success_terms = gym_config["env"]["extensions"]["agent_success"]["terms"]
    assert {term["object"] for term in success_terms} == {
        "left_apple",
        "right_orange",
    }

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    basic_background = paths.basic_background.read_text(encoding="utf-8")
    assert "the left apple and right orange into the wicker_basket" in task_prompt
    assert "right_arm must only manipulate `right_orange`" in task_prompt
    assert "- left_apple: the apple mesh initially" in basic_background
    assert "- right_orange: the orange mesh initially" in basic_background


def test_pipeline_auto_replacement_uses_rotated_robot_view_order() -> None:
    gym_config = {
        "rigid_object": [
            {"uid": "bread_1", "init_pos": [0.0, 0.2, 0.76]},
            {"uid": "bread_2", "init_pos": [0.0, -0.1, 0.76]},
        ],
    }

    assert (
        target_replacements_cli._auto_replacement_source_uid(
            gym_config,
            replacement_number=1,
            option_name="--target_replacement1",
        )
        == "bread_1"
    )
    assert (
        target_replacements_cli._auto_replacement_source_uid(
            gym_config,
            replacement_number=2,
            option_name="--target_replacement2",
        )
        == "bread_2"
    )


def test_directory_input_prefers_merged_config_and_preserves_extra_scene_scale(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)
    background_mesh = project_dir / "mesh_assets/backgrounds/vase_0.glb"
    _write_minimal_glb(background_mesh, _default_mesh_vertices())

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

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_agent",
        target_body_scale=0.8,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}

    assert set(rigid_objects) == {
        "left_apple",
        "right_apple",
        "vase_0",
    }
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}
    assert "wicker_basket" in background_objects
    assert background_objects["wicker_basket"]["body_type"] == "kinematic"
    assert rigid_objects["left_apple"]["body_scale"] == [0.8, 0.8, 0.8]
    assert rigid_objects["right_apple"]["body_scale"] == [0.8, 0.8, 0.8]
    assert rigid_objects["vase_0"]["body_scale"] == [1.2, 1.1, 0.9]
    _assert_normalized_obj_path(rigid_objects["vase_0"]["shape"]["fpath"])


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
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        action_agent_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_action_agent_config_from_project(
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
    assert set(rigid_objects) == {"apple_2"}
    assert rigid_objects["apple_2"]["body_scale"] == [0.5, 0.5, 0.5]
    assert rigid_objects["apple_2"]["body_type"] == "dynamic"
    assert background_objects["apple_1"]["body_scale"] == [1.0, 1.0, 1.0]
    assert background_objects["apple_1"]["body_type"] == "kinematic"
    assert background_objects["table"]["body_scale"] == [1.0, 1.0, 1.0]
    assert background_objects["wicker_basket"]["body_scale"] == [1.0, 1.0, 1.0]
    assert background_objects["wicker_basket"]["body_type"] == "kinematic"
    assert background_objects["wicker_basket"]["init_rot"] == [0.0, 0.0, 180.0]

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["op"] == "all"
    axis_terms = {
        (term.get("axis"), term.get("offset"))
        for term in success["terms"]
        if term["type"] == "object_axis_offset_near"
    }
    assert ("y", 0.16) in axis_terms
    assert ("x", 0.0) in axis_terms

    assert "agent_grasp_pose_overrides" not in gym_config["env"]["extensions"]

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert "Move apple_2 to the left of basket_3." in task_prompt
    assert (
        "Generate one deterministic nominal graph with exactly 4 nominal edges"
        in task_prompt
    )
    assert '"atomic_action_class":"PickUp","robot_name":"right_arm"' in task_prompt
    assert '"atomic_action_class":"Place","robot_name":"right_arm"' in task_prompt
    assert '"obj_name":"apple_2"' in task_prompt
    assert "left_arm_action: null" in task_prompt
    assert "Generate exactly 10 nominal edges" not in task_prompt

    assert _stable_summary(paths.summary) == {
        "mode": "relative_placement",
        "moved_object": "apple_2",
        "reference_object": "wicker_basket",
        "relation": "left_of",
        "active_arm": "right_arm",
        "release_offset": [0.0, 0.16, 0.12],
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
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        action_agent_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_front_relative_agent",
        task_name="AppleFrontOfApple",
        task_description="用右臂把 apple_1 放到 apple_2 前边",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}
    assert set(rigid_objects) == {"apple_1"}
    assert rigid_objects["apple_1"]["body_type"] == "dynamic"
    assert background_objects["apple_2"]["body_type"] == "kinematic"
    assert background_objects["wicker_basket"]["body_type"] == "kinematic"

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["op"] == "all"
    axis_terms = {
        (term.get("axis"), term.get("offset"))
        for term in success["terms"]
        if term["type"] == "object_axis_offset_near"
    }
    assert ("x", 0.16) in axis_terms
    assert ("y", 0.0) in axis_terms

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert '"offset":[0.16,0.0,0.22]' in task_prompt
    assert '"offset":[0.16,0.0,0.22]' in atom_actions

    assert _stable_summary(paths.summary) == {
        "mode": "relative_placement",
        "moved_object": "apple_1",
        "reference_object": "apple_2",
        "relation": "front_of",
        "active_arm": "right_arm",
        "release_offset": [0.16, 0.0, 0.12],
    }


def test_task_description_generates_self_relative_front_left_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    for rel_path in (
        "mesh_assets/table/table_0.glb",
        "mesh_assets/chip_bag/chip_bag_1.glb",
    ):
        _write_minimal_glb(project_dir / rel_path, _default_mesh_vertices())

    gym_config = {
        "id": "Image2Tabletop-1790000000-v0",
        "background": [
            _mesh_object(
                "table",
                "mesh_assets/table/table_0.glb",
                [0.0, 0.0, 0.36],
                [0.0, 0.0, 180.0],
            ),
        ],
        "rigid_object": [
            _mesh_object(
                "chip_bag_1",
                "mesh_assets/chip_bag/chip_bag_1.glb",
                [0.18, 0.22, 0.76],
                [0.0, 0.0, 25.0],
            )
        ],
    }
    (project_dir / "gym_config.json").write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "chip_bag_1",
            "reference_object": "chip_bag_1",
            "goal_relation": "front_left_of",
            "arm": "left",
            "task_prompt_summary": "Move the chip bag front-left from its start.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_self_relative_agent",
        task_description="用左臂把薯片袋子往左前移动",
        target_body_scale=0.5,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    assert set(rigid_objects) == {"chip_bag"}
    initial_position = rigid_objects["chip_bag"]["init_pos"]
    expected_x = round(initial_position[0] + 0.16, 6)
    expected_y = round(initial_position[1] + 0.16, 6)

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["op"] == "all"
    axis_terms = {
        (term.get("axis"), term.get("target"))
        for term in success["terms"]
        if term["type"] == "object_axis_near"
    }
    assert ("x", expected_x) in axis_terms
    assert ("y", expected_y) in axis_terms

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert '"reference":"absolute"' in task_prompt
    assert '"reference":"absolute"' in atom_actions
    assert f'"position":[{expected_x},{expected_y},' in task_prompt

    assert _stable_summary(paths.summary) == {
        "mode": "relative_placement",
        "moved_object": "chip_bag",
        "reference_object": "chip_bag",
        "relation": "front_left_of",
        "active_arm": "left_arm",
        "release_offset": [0.16, 0.16, 0.12],
    }


def test_task_description_generates_relative_front_right_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "apple_1",
            "reference_object": "basket_3",
            "goal_relation": "front_right_of",
            "arm": "right",
            "task_prompt_summary": "Move apple_1 to the front-right of basket_3.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        action_agent_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_front_right_relative_agent",
        task_description="用右臂把 apple_1 放到 basket_3 右前",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    success = gym_config["env"]["extensions"]["agent_success"]
    axis_terms = {
        (term.get("axis"), term.get("offset"))
        for term in success["terms"]
        if term["type"] == "object_axis_offset_near"
    }
    assert ("x", 0.16) in axis_terms
    assert ("y", -0.16) in axis_terms

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert '"offset":[0.16,-0.16,0.12]' in task_prompt
    assert _stable_summary(paths.summary)["release_offset"] == [0.16, -0.16, 0.12]


def test_side_relation_offsets_use_robot_view_front_back_convention() -> None:
    assert action_agent_config_generation._side_relation_xy_offsets("front_of") == (
        0.16,
        0.0,
    )
    assert action_agent_config_generation._side_relation_xy_offsets("behind") == (
        -0.16,
        0.0,
    )
    assert action_agent_config_generation._side_relation_xy_offsets(
        "front_left_of"
    ) == (
        0.16,
        0.16,
    )
    assert action_agent_config_generation._side_relation_xy_offsets(
        "back_right_of"
    ) == (
        -0.16,
        -0.16,
    )


@pytest.mark.parametrize(
    ("raw_relation", "normalized"),
    [
        ("左前", "front_left_of"),
        ("左后", "back_left_of"),
        ("右前", "front_right_of"),
        ("右后", "back_right_of"),
    ],
)
def test_relative_relation_aliases_include_diagonal_chinese_directions(
    raw_relation: str,
    normalized: str,
) -> None:
    assert action_agent_config_generation._normalize_relative_relation(
        raw_relation
    ) == (normalized)


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
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        action_agent_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_action_agent_config_from_project(
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
    assert paths.summary["active_arm"] == "left_arm"

    assert "agent_grasp_pose_overrides" not in gym_config["env"]["extensions"]


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
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        action_agent_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_left_arm_agent",
        task_description="左臂把 apple_1 放到 basket_3 左边",
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}
    assert set(rigid_objects) == {"apple_1"}
    assert rigid_objects["apple_1"]["body_type"] == "dynamic"
    assert background_objects["apple_2"]["body_type"] == "kinematic"
    assert background_objects["wicker_basket"]["body_type"] == "kinematic"
    assert background_objects["wicker_basket"]["init_rot"] == [0.0, 0.0, 180.0]
    assert "agent_grasp_pose_overrides" not in gym_config["env"]["extensions"]
    assert paths.summary["active_arm"] == "left_arm"

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert '"atomic_action_class":"PickUp","robot_name":"left_arm"' in task_prompt
    assert '"obj_name":"apple_1"' in task_prompt
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
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_right_arm_agent",
        task_description="右臂把 apple_2 放到 basket_3 右边",
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}
    assert set(rigid_objects) == {"apple_2"}
    assert rigid_objects["apple_2"]["body_type"] == "dynamic"
    assert background_objects["apple_1"]["body_type"] == "kinematic"
    assert background_objects["wicker_basket"]["body_type"] == "kinematic"
    assert "agent_grasp_pose_overrides" not in gym_config["env"]["extensions"]
    assert paths.summary["active_arm"] == "right_arm"

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert '"atomic_action_class":"PickUp","robot_name":"right_arm"' in task_prompt
    assert '"obj_name":"apple_2"' in task_prompt
    assert "left_arm_action: null" in task_prompt


def test_demo3_relative_placement_uses_role_aware_scene_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_demo3_role_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        assert kwargs["task_description"] == "用右臂把咖啡杯子放到垫子上"
        return {
            "moved_object": "cup_1",
            "reference_object": "pad_1",
            "goal_relation": "on",
            "arm": "right",
            "task_prompt_summary": "Place the cup on the pad.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_demo3_relative_agent",
        task_description="用右臂把咖啡杯子放到垫子上",
        target_body_scale=0.8,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}
    assert set(rigid_objects) == {"cup"}
    assert rigid_objects["cup"]["body_type"] == "dynamic"
    assert rigid_objects["cup"]["body_scale"] == [0.8, 0.8, 0.8]
    assert background_objects["pad"]["body_type"] == "kinematic"
    assert background_objects["pad"]["body_scale"] == [1.2, 1.0, 0.4]
    assert background_objects["fork"]["body_type"] == "kinematic"
    assert background_objects["fork"]["body_scale"] == [0.7, 0.7, 0.7]

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["type"] == "object_on_object"
    assert success["object"] == "cup"
    assert success["support"] == "pad"

    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert atom_actions.count('"atomic_action_class":"PickUp"') == 1
    assert '"atomic_action_class":"PickUp","robot_name":"right_arm"' in atom_actions
    assert '"obj_name":"cup"' in atom_actions
    assert _stable_summary(paths.summary)["relation"] == "on"


def test_relative_orientation_intent_generates_axis_align_move_held_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "15_Move Stapler Pad_gym_project"
    for rel_path in (
        "mesh_assets/table/table_0.glb",
        "mesh_assets/pad/colored_pad_1.glb",
        "mesh_assets/stapler/stapler_1.glb",
    ):
        _write_minimal_glb(project_dir / rel_path, _default_mesh_vertices())

    gym_config = {
        "id": "Image2Tabletop-15-v0",
        "background": [
            _mesh_object(
                "table_0",
                "mesh_assets/table/table_0.glb",
                [0.0, 0.0, -0.05],
                [0.0, 0.0, 0.0],
            ),
            _mesh_object(
                "colored_pad_1",
                "mesh_assets/pad/colored_pad_1.glb",
                [0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ),
        ],
        "rigid_object": [
            _mesh_object(
                "stapler_1",
                "mesh_assets/stapler/stapler_1.glb",
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0],
            )
        ],
    }
    gym_config["background"][1]["body_scale"] = [1.2, 1.0, 0.3]
    (project_dir / "gym_config.json").write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )

    def fake_call_relative_task_llm(**kwargs):
        assert (
            kwargs["task_description"]
            == "使用合适的机械臂将订书机水平摆正到彩色垫子上。"
        )
        return {
            "moved_object": "stapler_1",
            "reference_object": "colored_pad_1",
            "goal_relation": "on",
            "arm": "auto",
            "orientation_goal": "axis_align",
            "orientation_reference": "reference_object",
            "orientation_axis": "long_axis",
            "task_prompt_summary": "Place the stapler horizontally on the colored pad.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_stapler_pad_agent",
        task_name="Demo3_Text",
        task_description="使用合适的机械臂将订书机水平摆正到彩色垫子上。",
        target_body_scale=0.8,
        prewarm_coacd_cache=False,
    )

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    summary = _stable_summary(paths.summary)
    active_arm = summary["active_arm"]
    release_offset_json = json.dumps(
        summary["release_offset"], ensure_ascii=False, separators=(",", ":")
    )
    high_offset = list(summary["release_offset"])
    high_offset[2] = round(float(high_offset[2]) + 0.25, 6)
    high_offset_json = json.dumps(
        high_offset, ensure_ascii=False, separators=(",", ":")
    )
    assert (
        "Generate one deterministic nominal graph with exactly 7 nominal edges"
        in task_prompt
    )
    for text in (task_prompt, atom_actions):
        assert '"atomic_action_class":"MoveHeldObject"' in text
        assert '"target_object_pose":{"reference":"object"' in text
        assert '"obj_name":"colored_pad"' in text
        assert (
            f'"offset":{high_offset_json},"orientation_goal":"preserve",'
            '"orientation_axis":"none"}' in text
        )
        assert (
            f'"offset":{high_offset_json},"orientation_goal":"axis_align",'
            '"orientation_axis":"long_axis","align_to":"colored_pad"' in text
        )
        assert f'"offset":{release_offset_json}' in text
        assert '"orientation_goal":"axis_align"' in text
        assert '"orientation_axis":"long_axis"' in text
        assert '"align_to":"colored_pad"' in text
        assert (
            f'"atomic_action_class":"Place","robot_name":"{active_arm}",'
            '"control":"arm","target_pose":{"reference":"relative",'
            '"offset":[0.0,0.0,0.0],"frame":"world"},'
            '"cfg":{"sample_interval":10,"lift_height":0.0}' in text
        )
        assert (
            f'"atomic_action_class":"MoveEndEffector","robot_name":"{active_arm}",'
            '"control":"arm","target_pose":{"reference":"relative",'
            '"offset":[0.0,0.0,0.1],"frame":"world"}' in text
        )

    assert summary["orientation_goal"] == "axis_align"
    assert summary["orientation_axis"] == "long_axis"
    assert summary["orientation_align_to"] == "colored_pad"


def test_relative_orientation_upright_does_not_emit_align_to(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "apple_2",
            "reference_object": "basket_3",
            "goal_relation": "left_of",
            "orientation_goal": "upright",
            "orientation_reference": "none",
            "task_prompt_summary": "Move apple_2 upright to the left of basket_3.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_upright_relative_agent",
        task_description="把 apple_2 扶正后放到 basket_3 左边",
        prewarm_coacd_cache=False,
    )

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert (
        "Generate one deterministic nominal graph with exactly 7 nominal edges"
        in task_prompt
    )
    for text in (task_prompt, atom_actions):
        assert '"orientation_goal":"preserve","orientation_axis":"none"' in text
        assert '"orientation_goal":"upright"' in text
        assert '"orientation_axis":"none"' in text
        assert '"align_to"' not in text
        assert (
            '"target_pose":{"reference":"relative","offset":[0.0,0.0,0.0],'
            '"frame":"world"},"cfg":{"sample_interval":10,"lift_height":0.0}' in text
        )
        assert '"atomic_action_class":"MoveEndEffector"' in text
    assert _stable_summary(paths.summary)["orientation_goal"] == "upright"
    assert paths.summary["orientation_axis"] == "none"
    assert paths.summary["orientation_align_to"] is None


def test_relative_on_table_release_offset_uses_tabletop_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "43_Shake Bottle_gym_project"
    table_vertices = [
        (-0.5, -0.4, 0.0),
        (0.5, -0.4, 0.0),
        (0.0, 0.4, 0.0),
        (-0.5, -0.4, 0.36),
        (0.5, -0.4, 0.36),
        (0.0, 0.4, 0.36),
    ]
    bottle_vertices = [
        (-0.02, 0.0, 0.0),
        (0.02, 0.0, 0.0),
        (0.0, 0.02, 0.16),
    ]
    _write_minimal_glb(project_dir / "mesh_assets/table/table_0.glb", table_vertices)
    _write_minimal_glb(
        project_dir / "mesh_assets/bottle/bottle_1.glb",
        bottle_vertices,
    )
    gym_config = {
        "id": "Image2Tabletop-43-v0",
        "background": [
            _mesh_object(
                "table",
                "mesh_assets/table/table_0.glb",
                [0.0, 0.0, -0.02],
                [0.0, 0.0, 0.0],
            )
        ],
        "rigid_object": [
            _mesh_object(
                "interact_bottle_1",
                "mesh_assets/bottle/bottle_1.glb",
                [0.05, 0.05, 0.36],
                [90.0, 0.0, 0.0],
            )
        ],
    }
    (project_dir / "gym_config.json").write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "interact_bottle_1",
            "reference_object": "table",
            "goal_relation": "on",
            "arm": "left",
            "orientation_goal": "upright",
            "orientation_reference": "none",
            "task_prompt_summary": "Use the left arm to stand the bottle on table.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_bottle_on_table_agent",
        task_name="Demo43",
        task_description="用左臂把瓶子扶正放到桌面上",
        target_body_scale=0.8,
        prewarm_coacd_cache=False,
    )

    summary = _stable_summary(paths.summary)
    expected_release_z = 0.36 + 0.003
    assert summary["release_offset"][2] == pytest.approx(expected_release_z)
    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    release_offset_json = json.dumps(
        summary["release_offset"], ensure_ascii=False, separators=(",", ":")
    )
    assert f'"offset":{release_offset_json},"orientation_goal":"upright"' in task_prompt
    assert '"offset":[0.0,0.0,0.2]' not in task_prompt


def test_relative_on_preserve_uses_object_pose_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "2_Beat Block Hammer_gym_project"
    cube_vertices = [
        (-0.03, -0.03, 0.0),
        (0.03, -0.03, 0.0),
        (0.03, 0.03, 0.0),
        (-0.03, -0.03, 0.06),
        (0.03, -0.03, 0.06),
        (0.03, 0.03, 0.06),
    ]
    hammer_vertices = [
        (-0.08, -0.01, -0.01),
        (0.08, -0.01, -0.01),
        (0.08, 0.01, -0.01),
        (-0.08, -0.01, 0.02),
        (0.08, -0.01, 0.02),
        (0.08, 0.01, 0.02),
    ]
    _write_minimal_glb(project_dir / "mesh_assets/table/table_0.glb", cube_vertices)
    _write_minimal_glb(project_dir / "mesh_assets/cube/cube_1.glb", cube_vertices)
    _write_minimal_glb(
        project_dir / "mesh_assets/hammer/hammer_1.glb",
        hammer_vertices,
    )
    gym_config = {
        "id": "Image2Tabletop-2-v0",
        "background": [
            _mesh_object(
                "table",
                "mesh_assets/table/table_0.glb",
                [0.0, 0.0, 0.30],
                [0.0, 0.0, 0.0],
            ),
            _mesh_object(
                "cube_1",
                "mesh_assets/cube/cube_1.glb",
                [0.0, 0.05, 0.40],
                [0.0, 0.0, 0.0],
            ),
        ],
        "rigid_object": [
            _mesh_object(
                "hammer_1",
                "mesh_assets/hammer/hammer_1.glb",
                [0.1, -0.05, 0.40],
                [0.0, 0.0, 0.0],
            )
        ],
    }
    (project_dir / "gym_config.json").write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "hammer_1",
            "reference_object": "cube_1",
            "goal_relation": "on",
            "arm": "auto",
            "task_prompt_summary": "Place the hammer on the cube.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_hammer_on_cube_agent",
        task_name="Demo02",
        task_description="桌上有一把锤子和一个方块，用机械臂抓住锤子放到方块上",
        target_body_scale=0.8,
        prewarm_coacd_cache=False,
    )

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    summary = _stable_summary(paths.summary)
    active_arm = summary["active_arm"]
    moved_object = summary["moved_object"]
    reference_object = summary["reference_object"]
    release_offset_json = json.dumps(
        summary["release_offset"], ensure_ascii=False, separators=(",", ":")
    )

    assert (
        "Generate one deterministic nominal graph with exactly 6 nominal edges"
        in task_prompt
    )
    for text in (task_prompt, atom_actions):
        assert "Place at the release pose" not in text
        assert (
            f'"atomic_action_class":"MoveHeldObject","robot_name":"{active_arm}",'
            '"control":"arm","target_object_pose":{"reference":"object",'
            f'"obj_name":"{reference_object}","offset":{release_offset_json},'
            '"orientation_goal":"preserve","orientation_axis":"none"}' in text
        )
        assert (
            f'"atomic_action_class":"Place","robot_name":"{active_arm}",'
            '"control":"arm","target_pose":{"reference":"relative",'
            '"offset":[0.0,0.0,0.0],"frame":"world"},'
            '"cfg":{"sample_interval":10,"lift_height":0.0}' in text
        )
        assert (
            f'"atomic_action_class":"MoveEndEffector","robot_name":"{active_arm}",'
            '"control":"arm","target_pose":{"reference":"relative",'
            '"offset":[0.0,0.0,0.1],"frame":"world"}' in text
        )

    assert summary["relation"] == "on"
    assert moved_object == "hammer"
    assert reference_object == "cube"


def test_relative_orientation_rejects_invalid_enum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "apple_2",
            "reference_object": "basket_3",
            "goal_relation": "left_of",
            "orientation_goal": "diagonal",
            "task_prompt_summary": "Invalid orientation.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    with pytest.raises(ValueError, match="Unsupported orientation_goal"):
        generate_action_agent_config_from_project(
            project_dir,
            tmp_path / "generated_invalid_orientation_agent",
            task_description="把 apple_2 斜着放到 basket_3 左边",
            prewarm_coacd_cache=False,
        )


def test_relative_orientation_rejects_invalid_reference_axis_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "apple_2",
            "reference_object": "basket_3",
            "goal_relation": "left_of",
            "orientation_goal": "axis_align",
            "orientation_reference": "reference_object",
            "orientation_axis": "x",
            "task_prompt_summary": "Invalid axis pairing.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    with pytest.raises(ValueError, match="reference_object"):
        generate_action_agent_config_from_project(
            project_dir,
            tmp_path / "generated_invalid_axis_pairing_agent",
            task_description="把 apple_2 水平摆正到 basket_3 左边",
            prewarm_coacd_cache=False,
        )


def test_relative_orientation_rejects_legacy_horizontal_goal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "apple_2",
            "reference_object": "basket_3",
            "goal_relation": "left_of",
            "orientation_goal": "horizontal",
            "orientation_reference": "reference_object",
            "orientation_axis": "long_axis",
            "task_prompt_summary": "Legacy orientation.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    with pytest.raises(ValueError, match="Unsupported orientation_goal"):
        generate_action_agent_config_from_project(
            project_dir,
            tmp_path / "generated_legacy_horizontal_agent",
            task_description="把 apple_2 水平摆正到 basket_3 左边",
            prewarm_coacd_cache=False,
        )


def test_task_description_allows_single_rigid_with_background_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    for rel_path in (
        "mesh_assets/table/table_0.glb",
        "mesh_assets/pad/pad_1.glb",
        "mesh_assets/chip_bag/chip_bag_1.glb",
    ):
        _write_minimal_glb(project_dir / rel_path, _default_mesh_vertices())

    gym_config = {
        "id": "Image2Tabletop-1790000000-v0",
        "background": [
            _mesh_object(
                "table",
                "mesh_assets/table/table_0.glb",
                [0.0, 0.0, 0.36],
                [0.0, 0.0, 180.0],
            ),
            _mesh_object(
                "pad_1",
                "mesh_assets/pad/pad_1.glb",
                [-0.1, -0.15, 0.74],
                [0.0, 0.0, 0.0],
            ),
        ],
        "rigid_object": [
            _mesh_object(
                "chip_bag_1",
                "mesh_assets/chip_bag/chip_bag_1.glb",
                [0.18, 0.22, 0.76],
                [0.0, 0.0, 25.0],
            )
        ],
    }
    (project_dir / "gym_config.json").write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )

    def fake_call_relative_task_llm(**kwargs):
        scene_roles = {
            item["source_uid"]: item["role"] for item in kwargs["scene_summary"]
        }
        assert scene_roles["chip_bag_1"] == "rigid_object"
        assert scene_roles["pad_1"] == "background"
        return {
            "moved_object": "chip_bag_1",
            "reference_object": "pad_1",
            "goal_relation": "on",
            "arm": "left",
            "task_prompt_summary": "Place the chip bag on the pad.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_single_rigid_agent",
        task_description="用左臂抓薯片袋子放到垫子上",
        target_body_scale=0.5,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}
    assert set(rigid_objects) == {"chip_bag"}
    assert rigid_objects["chip_bag"]["body_type"] == "dynamic"
    assert rigid_objects["chip_bag"]["body_scale"] == [0.5, 0.5, 0.5]
    assert background_objects["pad"]["body_type"] == "static"

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["type"] == "object_on_object"
    assert success["object"] == "chip_bag"
    assert success["support"] == "pad"

    registry = gym_config["env"]["events"]["register_info_to_env"]["params"]["registry"]
    registered_uids = {entry["entity_cfg"]["uid"] for entry in registry}
    assert {"chip_bag", "pad"}.issubset(registered_uids)


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
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        action_agent_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_dual_relative_agent",
        task_description=(
            "左臂把 apple_2 放到 basket_3 左边，右臂把 apple_1 放到 basket_3 右边"
        ),
        target_body_scale=0.7,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}
    assert set(rigid_objects) == {"apple_1", "apple_2"}
    assert rigid_objects["apple_1"]["body_type"] == "dynamic"
    assert rigid_objects["apple_2"]["body_type"] == "dynamic"
    assert background_objects["wicker_basket"]["body_type"] == "kinematic"
    assert "agent_grasp_pose_overrides" not in gym_config["env"]["extensions"]

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["op"] == "all"
    assert len(success["terms"]) == 2
    axis_terms = {
        (term["object"], term["axis"], term["offset"])
        for placement_success in success["terms"]
        for term in placement_success["terms"]
        if term["type"] == "object_axis_offset_near"
    }
    assert ("apple_2", "y", 0.16) in axis_terms
    assert ("apple_1", "y", -0.16) in axis_terms

    attr_names = {
        attr["name"]
        for attr in gym_config["env"]["events"]["prepare_extra_attr"]["params"]["attrs"]
    }
    assert "grasp_pose_object" not in attr_names

    assert _stable_summary(paths.summary) == {
        "mode": "dual_arm_relative_placement",
        "placements": [
            {
                "moved_object": "apple_2",
                "reference_object": "wicker_basket",
                "relation": "left_of",
                "active_arm": "left_arm",
                "release_offset": [0.0, 0.16, 0.12],
            },
            {
                "moved_object": "apple_1",
                "reference_object": "wicker_basket",
                "relation": "right_of",
                "active_arm": "right_arm",
                "release_offset": [0.0, -0.16, 0.12],
            },
        ],
    }

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    basic_background = paths.basic_background.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert "Generate one deterministic nominal graph with exactly 6 nominal edges" in (
        task_prompt
    )
    assert (
        'left_arm_action: {"atomic_action_class":"PickUp","robot_name":"left_arm"'
        in task_prompt
    )
    assert (
        'right_arm_action: {"atomic_action_class":"PickUp","robot_name":"right_arm"'
        in task_prompt
    )
    assert (
        '"robot_name":"right_arm","control":"hand","target_qpos":{"source":"gripper_state","state":"close"}'
        in task_prompt
    )
    assert '"atomic_action_class":"Place","robot_name":"left_arm"' in task_prompt
    assert '"atomic_action_class":"Place","robot_name":"right_arm"' in task_prompt
    assert "The inactive arm must remain null" not in task_prompt
    assert "Both arms participate" in basic_background
    assert "left_arm moves `apple_2`" in basic_background
    assert "right_arm moves `apple_1`" in basic_background
    assert '"atomic_action_class":"PickUp","robot_name":"left_arm"' in atom_actions
    assert '"obj_name":"apple_2"' in atom_actions
    assert '"atomic_action_class":"PickUp","robot_name":"right_arm"' in atom_actions
    assert '"obj_name":"apple_1"' in atom_actions


def test_arrangement_response_orders_explicit_color_sequence(tmp_path: Path) -> None:
    _write_minimal_glb(
        tmp_path / "mesh_assets/table/table_0.glb",
        [(-0.60, -0.40, 0.0), (0.60, -0.40, 0.0), (0.0, 0.40, 0.0)],
    )
    for uid in ("cube_red", "cube_blue", "cube_green"):
        _write_minimal_glb(
            tmp_path / f"mesh_assets/cube/{uid}.glb",
            [(-0.02, -0.02, 0.0), (0.02, -0.02, 0.0), (0.0, 0.02, 0.04)],
        )

    scene_objects = [
        action_agent_config_generation._SceneObject(
            source_uid="table",
            source_role="background",
            config=_mesh_object(
                "table",
                "mesh_assets/table/table_0.glb",
                [0.0, 0.0, 0.36],
                [0.0, 0.0, 0.0],
            ),
        ),
        action_agent_config_generation._SceneObject(
            source_uid="cube_red",
            source_role="rigid_object",
            config=_mesh_object(
                "cube_red",
                "mesh_assets/cube/cube_red.glb",
                [0.0, 0.20, 0.76],
                [0.0, 0.0, 0.0],
            ),
        ),
        action_agent_config_generation._SceneObject(
            source_uid="cube_blue",
            source_role="rigid_object",
            config=_mesh_object(
                "cube_blue",
                "mesh_assets/cube/cube_blue.glb",
                [0.0, -0.10, 0.76],
                [0.0, 0.0, 0.0],
            ),
        ),
        action_agent_config_generation._SceneObject(
            source_uid="cube_green",
            source_role="rigid_object",
            config=_mesh_object(
                "cube_green",
                "mesh_assets/cube/cube_green.glb",
                [0.0, 0.00, 0.76],
                [0.0, 0.0, 0.0],
            ),
        ),
    ]
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]

    spec = _apply_arrangement_task_response(
        response={
            "objects": ["cube_red", "cube_green", "cube_blue"],
            "order_by": "color",
            "ordered_attributes": ["red", "green", "blue"],
            "object_attributes": {
                "cube_blue": {"color": "blue"},
                "cube_red": {"color": "red"},
                "cube_green": {"color": "green"},
            },
            "task_prompt_summary": "Arrange the cubes red, green, blue.",
        },
        table_source_uid="table",
        scene_objects=scene_objects,
        rigid_objects=rigid_objects,
        scene_dir=tmp_path,
        task_description="将红、绿、蓝三个方块按从左到右红、绿、蓝的顺序排成一行",
    )

    assert [step.source_uid for step in spec.steps] == [
        "cube_red",
        "cube_green",
        "cube_blue",
    ]
    assert [step.color for step in spec.steps] == ["red", "green", "blue"]
    assert [step.slot_index for step in spec.steps] == [0, 1, 2]
    assert [step.target_xy[1] for step in spec.steps] == sorted(
        step.target_xy[1] for step in spec.steps
    )
    assert [step.orientation_goal for step in spec.steps] == [
        "axis_align",
        "axis_align",
        "axis_align",
    ]
    assert [step.orientation_axis for step in spec.steps] == ["y", "y", "y"]


def test_arrangement_line_slot_positions_are_centered_left_to_right() -> None:
    slots = _arrangement_line_slot_positions(
        anchor_xy=[0.10, -0.20],
        count=3,
        spacing=0.08,
        line_axis="left_to_right",
    )

    assert slots == [
        [0.10, -0.28],
        [0.10, -0.20],
        [0.10, -0.12],
    ]


def test_task_description_generates_size_order_arrangement_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_arrangement_project(project_dir)

    def fake_call_arrangement_task_llm(**kwargs):
        size_by_uid = {
            item["source_uid"]: item["size_score"]
            for item in kwargs["scene_summary"]
            if item["role"] == "rigid_object"
        }
        assert size_by_uid["cube_2"] > size_by_uid["cube_1"] > size_by_uid["cube_3"]
        return {
            "objects": ["cube_1", "cube_2", "cube_3"],
            "order_by": "size",
            "order_direction": "descending",
            "anchor": "table_center",
            "line_axis": "left_to_right",
            "task_prompt_summary": (
                "Move the three cubes to the table center and arrange them "
                "from large to small left-to-right."
            ),
            "basic_background_notes": "All three cubes are movable task objects.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_arrangement_task_llm",
        fake_call_arrangement_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_arrangement_agent",
        task_name="BlocksRankingSize",
        task_description="桌上有三个颜色随机的方块，将它们移动到桌子中央，并按从左到右由大到小排列。",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    assert set(rigid_objects) == {"cube_1", "cube_2", "cube_3"}
    assert {obj["body_type"] for obj in rigid_objects.values()} == {"dynamic"}
    assert rigid_objects["cube_2"]["body_scale"] == [1.0, 1.0, 1.0]

    assert _stable_summary(paths.summary) == {
        "mode": "arrangement_line",
        "axis": "left_to_right",
        "anchor": "table_center",
        "order_by": "size",
        "order_direction": "descending",
        "line_origin_xy": paths.summary["line_origin_xy"],
        "spacing": paths.summary["spacing"],
        "layout_clearance": paths.summary["layout_clearance"],
        "placements": [
            {
                "object": "cube_2",
                "source_uid": "cube_2",
                "slot_index": 0,
                "active_arm": "right_arm",
                "target_xy": paths.summary["placements"][0]["target_xy"],
                "orientation_goal": "axis_align",
                "orientation_axis": "y",
            },
            {
                "object": "cube_1",
                "source_uid": "cube_1",
                "slot_index": 1,
                "active_arm": "left_arm",
                "target_xy": paths.summary["placements"][1]["target_xy"],
                "orientation_goal": "axis_align",
                "orientation_axis": "y",
            },
            {
                "object": "cube_3",
                "source_uid": "cube_3",
                "slot_index": 2,
                "active_arm": "left_arm",
                "target_xy": paths.summary["placements"][2]["target_xy"],
                "orientation_goal": "axis_align",
                "orientation_axis": "y",
            },
        ],
    }
    target_x_values = [
        placement["target_xy"][0] for placement in paths.summary["placements"]
    ]
    target_y_values = [
        placement["target_xy"][1] for placement in paths.summary["placements"]
    ]
    assert len({round(value, 6) for value in target_x_values}) == 1
    assert target_y_values == sorted(target_y_values)
    assert paths.summary["spacing"] >= 0.07
    assert paths.summary["layout_clearance"] == pytest.approx(0.025)
    _assert_arrangement_slots_avoid_initial_objects(
        paths.summary,
        gym_config,
    )

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["op"] == "all"
    xy_targets = {
        (term["object"], tuple(term["target_xy"]))
        for term in success["terms"]
        if term["type"] == "object_xy_near"
    }
    expected_xy_targets = {
        (placement["object"], tuple(placement["target_xy"]))
        for placement in paths.summary["placements"]
    }
    assert xy_targets == expected_xy_targets
    expected_xy_tolerance = min(0.03, paths.summary["spacing"] * 0.35)
    xy_tolerances = {
        term["tolerance"]
        for term in success["terms"]
        if term["type"] == "object_xy_near"
    }
    assert len(xy_tolerances) == 1
    assert next(iter(xy_tolerances)) == pytest.approx(expected_xy_tolerance)

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert "Generate one deterministic nominal graph with exactly 18 nominal edges" in (
        task_prompt
    )
    assert task_prompt.count('"atomic_action_class":"PickUp"') == 3
    assert task_prompt.count('"atomic_action_class":"Place"') == 3
    assert task_prompt.count('"reference":"absolute"') >= 9
    assert task_prompt.count('"orientation_goal":"axis_align"') == 6
    assert task_prompt.count('"orientation_axis":"y"') == 6
    assert task_prompt.count('"orientation_goal":"preserve"') == 3
    assert task_prompt.count('"target_pose":{"reference":"relative"') == 3
    assert "Collision-aware line origin xy" in task_prompt
    assert atom_actions.count('"atomic_action_class":"PickUp"') == 3
    assert atom_actions.count('"orientation_goal":"axis_align"') == 6
    assert atom_actions.count('"orientation_axis":"y"') == 6
    assert atom_actions.count('"atomic_action_class":"Place"') == 3


def test_arrangement_collision_aware_layout_scales_to_six_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "six_blocks_gym_project"
    _write_arrangement_project_with_count(project_dir, count=6, cube_size=0.035)

    def fake_call_arrangement_task_llm(**kwargs):
        return {
            "objects": [f"cube_{index}" for index in range(1, 7)],
            "order_by": "explicit",
            "order_direction": "given",
            "anchor": "table_center",
            "line_axis": "left_to_right",
            "task_prompt_summary": "Arrange six cubes left to right.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_arrangement_task_llm",
        fake_call_arrangement_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_six_arrangement_agent",
        task_description="把六个方块从左到右排成一行",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    summary = paths.summary
    assert len(summary["placements"]) == 6
    assert summary["spacing"] >= 0.07
    assert summary["layout_clearance"] == pytest.approx(0.025)
    assert all(
        placement["orientation_goal"] == "axis_align"
        and placement["orientation_axis"] == "y"
        for placement in summary["placements"]
    )
    x_values = [placement["target_xy"][0] for placement in summary["placements"]]
    y_values = [placement["target_xy"][1] for placement in summary["placements"]]
    assert len({round(value, 6) for value in x_values}) == 1
    assert y_values == sorted(y_values)
    _assert_arrangement_slots_avoid_initial_objects(summary, gym_config)

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert "Generate one deterministic nominal graph with exactly 36 nominal edges" in (
        task_prompt
    )


def test_arrangement_layout_fails_when_row_cannot_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "crowded_blocks_gym_project"
    _write_arrangement_project_with_count(
        project_dir,
        count=6,
        cube_size=0.12,
        table_half_x=0.18,
        table_half_y=0.22,
    )

    def fake_call_arrangement_task_llm(**kwargs):
        return {
            "objects": [f"cube_{index}" for index in range(1, 7)],
            "order_by": "explicit",
            "order_direction": "given",
            "anchor": "table_center",
            "line_axis": "left_to_right",
            "task_prompt_summary": "Arrange six oversized cubes left to right.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_arrangement_task_llm",
        fake_call_arrangement_task_llm,
    )

    with pytest.raises(ValueError, match="collision-free one-line arrangement"):
        generate_action_agent_config_from_project(
            project_dir,
            tmp_path / "generated_crowded_arrangement_agent",
            task_description="把六个大方块从左到右排成一行",
            prewarm_coacd_cache=False,
        )


def test_dual_inside_same_container_uses_container_long_axis_slots(
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
                    "goal_relation": "inside",
                    "arm": "left",
                },
                {
                    "moved_object": "apple_1",
                    "reference_object": "basket_3",
                    "goal_relation": "inside",
                    "arm": "right",
                },
            ],
            "task_prompt_summary": "Use both arms to put both apples into basket_3.",
            "basic_background_notes": "Both apples share the same target container.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )
    monkeypatch.setattr(
        action_agent_config_generation,
        "_resolve_table_mesh_world_zmax",
        lambda scene_dir, table_obj: None,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_dual_inside_agent",
        task_description="双臂把两个 apple 放进 basket_3",
        prewarm_coacd_cache=False,
    )

    assert _stable_summary(paths.summary) == {
        "mode": "dual_arm_relative_placement",
        "placements": [
            {
                "moved_object": "apple_2",
                "reference_object": "wicker_basket",
                "relation": "inside",
                "active_arm": "left_arm",
                "release_offset": [-0.04, 0.0, 0.12],
            },
            {
                "moved_object": "apple_1",
                "reference_object": "wicker_basket",
                "relation": "inside",
                "active_arm": "right_arm",
                "release_offset": [0.04, 0.0, 0.12],
            },
        ],
    }

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["op"] == "all"
    assert {
        (term["type"], term["object"], term["container"]) for term in success["terms"]
    } == {
        ("object_in_container", "apple_2", "wicker_basket"),
        ("object_in_container", "apple_1", "wicker_basket"),
    }

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    for text in (task_prompt, atom_actions):
        assert '"offset":[-0.04,0.0,0.22]' in text
        assert '"offset":[0.04,0.0,0.22]' in text
        assert (
            '"atomic_action_class":"Place","robot_name":"left_arm",'
            '"control":"arm","target_pose":{"reference":"object",'
            '"obj_name":"wicker_basket","offset":[-0.04,0.0,0.12]}' in text
        )
        assert (
            '"atomic_action_class":"Place","robot_name":"right_arm",'
            '"control":"arm","target_pose":{"reference":"object",'
            '"obj_name":"wicker_basket","offset":[0.04,0.0,0.12]}' in text
        )
    assert "container XY long axis" in task_prompt


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
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    with pytest.raises(ValueError, match="one left arm and one right arm"):
        generate_action_agent_config_from_project(
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
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_dual_auto_relative_agent",
        task_description="双臂分别移动两个苹果",
        prewarm_coacd_cache=False,
    )

    active_arms = [placement["active_arm"] for placement in paths.summary["placements"]]
    assert active_arms == ["right_arm", "left_arm"]

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    assert "agent_grasp_pose_overrides" not in gym_config["env"]["extensions"]


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
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_stack_agent",
        task_description="把 apple_2 放到 apple_1 上方并松手",
        target_body_scale=0.6,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}
    assert set(rigid_objects) == {"apple_2"}
    assert rigid_objects["apple_2"]["body_scale"] == [0.6, 0.6, 0.6]
    assert rigid_objects["apple_2"]["body_type"] == "dynamic"
    assert background_objects["apple_1"]["body_scale"] == [1.0, 1.0, 1.0]
    assert background_objects["apple_1"]["body_type"] == "kinematic"
    assert background_objects["wicker_basket"]["body_scale"] == [1.0, 1.0, 1.0]
    assert background_objects["wicker_basket"]["body_type"] == "kinematic"

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
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    with pytest.raises(ValueError, match="unknown moved_object"):
        generate_action_agent_config_from_project(
            project_dir,
            tmp_path / "bad_agent",
            task_description="把 missing_bread 放到 basket_3 左边",
        )


def test_high_tabletop_scene_adjusts_robot_height_and_light(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)
    _write_minimal_glb(
        project_dir / "mesh_assets/table/table_0.glb",
        [(-0.5, 0.0, 0.82), (0.5, 0.0, 0.82), (0.0, -0.82, 0.82)],
    )

    gym_config_path = project_dir / "gym_config.json"
    source_config = json.loads(gym_config_path.read_text(encoding="utf-8"))
    for obj_config in source_config["rigid_object"]:
        obj_config["init_pos"][2] = 0.12
    gym_config_path.write_text(
        json.dumps(source_config, indent=2),
        encoding="utf-8",
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_high_table_agent",
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    expected_init_z = (
        1.18
        + action_agent_config_generation._DUAL_UR5_TABLETOP_CLEARANCE
        - action_agent_config_generation._DUAL_UR5_ARM_COMPONENT_Z
    )
    assert gym_config["robot"]["init_pos"][2] == pytest.approx(expected_init_z)
    assert gym_config["light"]["direct"][0]["intensity"] == 40.0


def test_tabletop_z_placement_uses_normalized_mesh_bounds(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_z_agent",
        target_body_scale=0.8,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    table_config = next(
        obj for obj in gym_config["background"] if obj["uid"] == "table"
    )
    table_top_z = action_agent_config_generation._mesh_config_world_zmax(table_config)
    expected_min_z = (
        table_top_z + action_agent_config_generation._TABLETOP_OBJECT_CLEARANCE
    )
    for obj_config in [
        *[obj for obj in gym_config["background"] if obj["uid"] != "table"],
        *gym_config["rigid_object"],
    ]:
        min_z, _ = action_agent_config_generation._mesh_config_world_z_bounds(
            obj_config
        )
        assert min_z == pytest.approx(expected_min_z)


def test_table_mesh_world_zmax_reads_glb_vertices(tmp_path: Path) -> None:
    scene_dir = tmp_path / "1790000000_gym_project"
    mesh_path = scene_dir / "mesh_assets/table/table_0.glb"
    _write_minimal_glb(
        mesh_path,
        [(-0.5, -0.5, 0.0), (0.5, -0.5, 1.2), (0.0, 0.5, 0.4)],
    )
    table_obj = action_agent_config_generation._SceneObject(
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

    assert action_agent_config_generation._resolve_table_mesh_world_zmax(
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
        _write_minimal_glb(mesh_path, _default_mesh_vertices())

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


def _write_demo3_role_project(project_dir: Path) -> None:
    for rel_path in (
        "mesh_assets/table/table_0.glb",
        "mesh_assets/cup/cup_1/cup_1.glb",
        "mesh_assets/pad/pad_1/pad_1.glb",
        "mesh_assets/fork/fork_1/fork_1.glb",
    ):
        _write_minimal_glb(project_dir / rel_path, _default_mesh_vertices())

    cup = _mesh_object(
        "cup_1",
        "mesh_assets/cup/cup_1/cup_1.glb",
        [0.18, 0.22, 0.76],
        [0.0, 0.0, 25.0],
    )
    pad = _mesh_object(
        "pad_1",
        "mesh_assets/pad/pad_1/pad_1.glb",
        [-0.1, -0.15, 0.74],
        [0.0, 0.0, -10.0],
    )
    pad["body_scale"] = [1.2, 1.0, 0.4]
    fork = _mesh_object(
        "fork_1",
        "mesh_assets/fork/fork_1/fork_1.glb",
        [0.32, -0.18, 0.75],
        [0.0, 0.0, 90.0],
    )
    fork["body_scale"] = [0.7, 0.7, 0.7]

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
        "rigid_object": [cup, pad, fork],
    }
    (project_dir / "gym_config.json").write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )


def _write_arrangement_project(project_dir: Path) -> None:
    _write_minimal_glb(
        project_dir / "mesh_assets/table/table_0.glb",
        [(-0.60, -0.40, 0.0), (0.60, -0.40, 0.0), (0.0, 0.40, 0.0)],
    )
    for uid, size in {
        "cube_1": 0.04,
        "cube_2": 0.06,
        "cube_3": 0.03,
    }.items():
        _write_minimal_glb(
            project_dir / f"mesh_assets/cube/{uid}/{uid}.glb",
            [
                (-size / 2.0, -size / 2.0, 0.0),
                (size / 2.0, -size / 2.0, 0.0),
                (0.0, size / 2.0, size),
            ],
        )

    gym_config = {
        "id": "Image2Tabletop-1790000000-v0",
        "background": [
            _mesh_object(
                "table",
                "mesh_assets/table/table_0.glb",
                [0.0, 0.0, 0.36],
                [0.0, 0.0, 0.0],
            )
        ],
        "rigid_object": [
            _mesh_object(
                "cube_1",
                "mesh_assets/cube/cube_1/cube_1.glb",
                [0.0, 0.08, 0.76],
                [0.0, 0.0, 0.0],
            ),
            _mesh_object(
                "cube_2",
                "mesh_assets/cube/cube_2/cube_2.glb",
                [0.0, -0.08, 0.76],
                [0.0, 0.0, 0.0],
            ),
            _mesh_object(
                "cube_3",
                "mesh_assets/cube/cube_3/cube_3.glb",
                [0.0, 0.16, 0.76],
                [0.0, 0.0, 0.0],
            ),
        ],
    }
    (project_dir / "gym_config.json").write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )


def _write_arrangement_project_with_count(
    project_dir: Path,
    *,
    count: int,
    cube_size: float,
    table_half_x: float = 0.60,
    table_half_y: float = 0.40,
) -> None:
    _write_minimal_glb(
        project_dir / "mesh_assets/table/table_0.glb",
        [
            (-table_half_x, -table_half_y, 0.0),
            (table_half_x, -table_half_y, 0.0),
            (0.0, table_half_y, 0.0),
        ],
    )
    rigid_objects = []
    for index in range(count):
        uid = f"cube_{index + 1}"
        _write_minimal_glb(
            project_dir / f"mesh_assets/cube/{uid}/{uid}.glb",
            [
                (-cube_size / 2.0, -cube_size / 2.0, 0.0),
                (cube_size / 2.0, -cube_size / 2.0, 0.0),
                (0.0, cube_size / 2.0, cube_size),
            ],
        )
        y = (index - (count - 1) / 2.0) * (cube_size + 0.01)
        rigid_objects.append(
            _mesh_object(
                uid,
                f"mesh_assets/cube/{uid}/{uid}.glb",
                [0.0, round(float(y), 6), 0.76],
                [0.0, 0.0, 0.0],
            )
        )

    gym_config = {
        "id": "Image2Tabletop-arrangement-v0",
        "background": [
            _mesh_object(
                "table",
                "mesh_assets/table/table_0.glb",
                [0.0, 0.0, 0.36],
                [0.0, 0.0, 0.0],
            )
        ],
        "rigid_object": rigid_objects,
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


def _assert_normalized_obj_path(fpath: str) -> None:
    path = Path(fpath)
    assert path.suffix == ".obj"
    assert "mesh_assets/normalized" in path.as_posix()
    assert MESH_FRAME_NORMALIZATION_POLICY_VERSION not in path.name
    assert len(path.name) <= 64
    assert path.is_file()
    assert (path.parent / "material.mtl").is_file()


def _stable_summary(summary: dict) -> dict:
    stable = {
        key: value
        for key, value in summary.items()
        if key not in {"normalized_meshes", "coacd_cache"}
    }
    if stable.get("orientation_goal") == "preserve":
        stable.pop("orientation_goal", None)
    if stable.get("orientation_axis") == "none":
        stable.pop("orientation_axis", None)
    if stable.get("orientation_align_to") is None:
        stable.pop("orientation_align_to", None)
    for placement in stable.get("placements", []):
        if placement.get("orientation_goal") == "preserve":
            placement.pop("orientation_goal", None)
        if placement.get("orientation_axis") == "none":
            placement.pop("orientation_axis", None)
        if placement.get("orientation_align_to") is None:
            placement.pop("orientation_align_to", None)
    return stable


def _assert_arrangement_slots_avoid_initial_objects(
    summary: dict,
    gym_config: dict,
) -> None:
    clearance = float(summary["layout_clearance"])
    spacing = float(summary["spacing"])
    half_extent = max(0.035, (spacing - clearance) / 2.0)
    initial_bounds = [
        _xy_bounds_around(obj["init_pos"][:2], half_extent)
        for obj in gym_config["rigid_object"]
    ]
    for placement in summary["placements"]:
        slot_bounds = _xy_bounds_around(placement["target_xy"], half_extent)
        assert all(
            not _xy_bounds_overlap_for_test(
                slot_bounds,
                init_bound,
                clearance=clearance,
            )
            for init_bound in initial_bounds
        )


def _xy_bounds_around(
    xy: list[float],
    half_extent: float,
) -> tuple[list[float], list[float]]:
    return (
        [float(xy[0]) - half_extent, float(xy[1]) - half_extent],
        [float(xy[0]) + half_extent, float(xy[1]) + half_extent],
    )


def _xy_bounds_overlap_for_test(
    first: tuple[list[float], list[float]],
    second: tuple[list[float], list[float]],
    *,
    clearance: float,
) -> bool:
    first_min, first_max = first
    second_min, second_max = second
    return not (
        first_max[0] + clearance <= second_min[0]
        or second_max[0] + clearance <= first_min[0]
        or first_max[1] + clearance <= second_min[1]
        or second_max[1] + clearance <= first_min[1]
    )


def _obj_vertices(path: Path) -> list[tuple[float, float, float]]:
    vertices = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("v "):
            continue
        _, x, y, z = line.split(maxsplit=3)
        vertices.append((float(x), float(y), float(z)))
    return vertices


def _single_obj_material_name(obj_text: str) -> str:
    names = {
        line.split(maxsplit=1)[1].strip()
        for line in obj_text.splitlines()
        if line.startswith("usemtl ")
    }
    assert len(names) == 1
    return next(iter(names))


def _single_map_kd_path(material_text: str, material_name: str) -> str:
    current_material = None
    texture_paths = []
    for line in material_text.splitlines():
        if line.startswith("newmtl "):
            current_material = line.split(maxsplit=1)[1].strip()
            continue
        if current_material == material_name and line.startswith("map_Kd "):
            texture_paths.append(line.split(maxsplit=1)[1].strip())
    assert len(texture_paths) == 1
    return texture_paths[0]


def _rounded_vertex_set(
    vertices: list[tuple[float, float, float]],
) -> set[tuple[float, float, float]]:
    return {
        (round(vertex[0], 6), round(vertex[1], 6), round(vertex[2], 6))
        for vertex in vertices
    }


def _default_mesh_vertices() -> list[tuple[float, float, float]]:
    return [(-0.05, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, -0.04, 0.0)]


def _tiny_png() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8DwHwAF"
        "gAJ/l7p7YwAAAABJRU5ErkJggg=="
    )


def _write_minimal_glb(
    path: Path,
    vertices: list[tuple[float, float, float]],
    *,
    node_translation: tuple[float, float, float] | None = None,
    embedded_base_color_png: bytes | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(vertices) < 3:
        raise ValueError("Minimal GLB test mesh requires at least three vertices.")
    position_binary = b"".join(struct.pack("<fff", *vertex) for vertex in vertices)
    position_binary_padded = position_binary + b"\x00" * (
        (4 - len(position_binary) % 4) % 4
    )
    indices = (0, 1, 2)
    index_binary = b"".join(struct.pack("<H", index) for index in indices)
    index_binary_padded = index_binary + b"\x00" * ((4 - len(index_binary) % 4) % 4)
    texcoord_binary = b"".join(
        struct.pack("<ff", *uv) for uv in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    )
    texcoord_binary_padded = texcoord_binary + b"\x00" * (
        (4 - len(texcoord_binary) % 4) % 4
    )
    image_binary = embedded_base_color_png or b""
    image_binary_padded = image_binary + b"\x00" * ((4 - len(image_binary) % 4) % 4)
    binary_parts = [position_binary_padded, index_binary_padded]
    index_offset = len(position_binary_padded)
    texcoord_offset = index_offset + len(index_binary_padded)
    image_offset = texcoord_offset + len(texcoord_binary_padded)
    if embedded_base_color_png is not None:
        binary_parts.extend([texcoord_binary_padded, image_binary_padded])
    binary = b"".join(binary_parts)
    mins = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maxs = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    node = {"mesh": 0}
    if node_translation is not None:
        node["translation"] = [float(value) for value in node_translation]
    buffer_views = [
        {
            "buffer": 0,
            "byteOffset": 0,
            "byteLength": len(position_binary),
            "target": 34962,
        },
        {
            "buffer": 0,
            "byteOffset": index_offset,
            "byteLength": len(index_binary),
            "target": 34963,
        },
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": len(vertices),
            "type": "VEC3",
            "min": mins,
            "max": maxs,
        },
        {
            "bufferView": 1,
            "componentType": 5123,
            "count": len(indices),
            "type": "SCALAR",
            "min": [min(indices)],
            "max": [max(indices)],
        },
    ]
    primitive = {"attributes": {"POSITION": 0}, "indices": 1}
    if embedded_base_color_png is not None:
        buffer_views.extend(
            [
                {
                    "buffer": 0,
                    "byteOffset": texcoord_offset,
                    "byteLength": len(texcoord_binary),
                    "target": 34962,
                },
                {
                    "buffer": 0,
                    "byteOffset": image_offset,
                    "byteLength": len(image_binary),
                },
            ]
        )
        accessors.append(
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": len(indices),
                "type": "VEC2",
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            }
        )
        primitive["attributes"]["TEXCOORD_0"] = 2
        primitive["material"] = 0

    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [{"primitives": [primitive]}],
        "nodes": [node],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if embedded_base_color_png is not None:
        doc["materials"] = [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "roughnessFactor": 1.0,
                }
            }
        ]
        doc["textures"] = [{"source": 0}]
        doc["images"] = [{"bufferView": 3, "mimeType": "image/png"}]
    json_chunk = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
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
        _write_minimal_glb(mesh_path, _default_mesh_vertices())
        calls.append((prompt, output_root, output_name))
        return {"scaled_mesh_path": str(mesh_path)}

    monkeypatch.setattr(
        action_agent_config_generation,
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
