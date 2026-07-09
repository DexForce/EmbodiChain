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
from types import SimpleNamespace
import base64
import hashlib
import json
import re
import struct

import pytest
import torch

from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.solvers import URSolverCfg
from embodichain.gen_sim.action_agent_pipeline.cli import (
    target_replacements as target_replacements_cli,
)
from embodichain.gen_sim.action_agent_pipeline.generation import (
    action_agent_config as action_agent_config_generation,
    relative_geometry,
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
from embodichain.gen_sim.action_agent_pipeline.generation.body_scale_baking import (
    BODY_SCALE_BAKE_POLICY_VERSION,
    bake_body_scale_into_meshes,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_blocks import (
    _make_observations_config,
    _record_camera_event_configs,
    _rotate_camera_extrinsics_around_target_z,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _RelativePlacementSpec,
    _RelativePlacementStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _TABLETOP_OBJECT_CLEARANCE,
    _mesh_config_local_zmin_after_rotation,
    _mesh_config_world_xy_bounds,
    _mesh_config_world_xy_center,
    _mesh_config_world_z_bounds,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _arm_side_for_position,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_config import (
    TargetReplacementSpec,
    generate_action_agent_config_from_project,
)
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_relative_task_prompt,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_spec import (
    _apply_arrangement_task_response,
    _arrangement_line_slot_positions,
)
from embodichain.gen_sim.action_agent_pipeline.generation.stacking_spec import (
    _is_stacking_task_description,
)
from embodichain.gen_sim.action_agent_pipeline.generation.success_specs import (
    _validate_success_uids,
)
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.success import (
    evaluate_configured_success,
)


@pytest.fixture(autouse=True)
def _patch_task_router_for_config_generation_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_task_router_llm(**kwargs):
        task_description = str(kwargs["task_description"])
        route = "object_manipulation"
        if any(keyword in task_description for keyword in ("叠", "堆叠", "摞")):
            route = "stacking"
        elif any(
            keyword in task_description
            for keyword in (
                "排成",
                "摆成一排",
                "排列",
                "排序",
                "从左到右",
                "由大到小",
                "由小到大",
            )
        ):
            route = "arrangement_line"
        return {
            "route": route,
            "confidence": 1.0,
            "reason": "Unit-test router stub.",
            "candidate_objects": [],
            "warnings": [],
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_task_router_llm",
        fake_call_task_router_llm,
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
    first_lights["direct"].append({"uid": "mutated_light"})

    assert second_robot["init_pos"] == pytest.approx([-2.0, 0.0, 0.84])
    assert first_robot["init_pos"] == pytest.approx([-2.0, 0.0, 0.42])
    assert second_robot["control_parts"]["left_arm"] == [
        f"left_joint{i}" for i in range(1, 7)
    ]
    assert second_sensors[0]["uid"] == "cam_high"
    assert second_lights["direct"] == []


def test_record_camera_events_generate_audience_view_name() -> None:
    events = _record_camera_event_configs(make_sensor_config, task_name="Demo111")

    assert set(events) == {"record_camera"}

    params = events["record_camera"]["params"]

    assert params["name"] == "record_cam_audience_view"
    assert params["video_name"] == "Demo111_audience_view"
    assert params["eye"] == pytest.approx([0.6, 0.0, 1.8])
    assert params["target"] == pytest.approx([0.0, 0.0, 0.75])
    assert params["up"] == pytest.approx([-1.0, 0.0, 0.0])


def test_camera_rotation_preserves_target_and_height() -> None:
    rotated = _rotate_camera_extrinsics_around_target_z(
        {
            "eye": [2.0, 1.0, 3.0],
            "target": [1.0, 1.0, 0.5],
            "up": [1.0, 0.0, 0.0],
        },
        degrees=90.0,
    )

    assert rotated["eye"] == pytest.approx([1.0, 2.0, 3.0])
    assert rotated["target"] == pytest.approx([1.0, 1.0, 0.5])
    assert rotated["up"] == pytest.approx([0.0, 1.0, 0.0])


def test_dual_ur5_template_uses_ur_solver_config() -> None:
    robot = make_dual_ur5_robot_config(robot_init_z=0.42)

    left_solver = robot["solver_cfg"]["left_arm"]
    right_solver = robot["solver_cfg"]["right_arm"]

    assert left_solver["class_type"] == "URSolver"
    assert right_solver["class_type"] == "URSolver"
    assert left_solver["ur_type"] == "ur5"
    assert right_solver["ur_type"] == "ur5"
    assert left_solver["urdf_path"] is None
    assert right_solver["urdf_path"] is None
    assert left_solver["root_link_name"] == "left_base_link"
    assert left_solver["end_link_name"] == "left_ee_link"
    assert right_solver["root_link_name"] == "right_base_link"
    assert right_solver["end_link_name"] == "right_ee_link"
    assert left_solver["tcp"][2][3] == pytest.approx(0.16)
    assert right_solver["tcp"][2][3] == pytest.approx(0.21)


def test_dual_ur5_template_uses_robotiq_arg2f_140_grippers() -> None:
    robot = make_dual_ur5_robot_config(robot_init_z=0.42)

    components = robot["urdf_cfg"]["components"]
    left_hand = next(
        component
        for component in components
        if component["component_type"] == "left_hand"
    )
    right_hand = next(
        component
        for component in components
        if component["component_type"] == "right_hand"
    )

    assert left_hand["urdf_path"] == "Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf"
    assert right_hand["urdf_path"] == left_hand["urdf_path"]
    assert robot["urdf_cfg"]["fname"] == "dual_ur5_robotiq_arg2f_140_basket"
    assert robot["control_parts"]["left_eef"] == [
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "left_inner_finger_joint",
        "left_right_outer_knuckle_joint",
        "left_right_inner_knuckle_joint",
        "left_right_inner_finger_joint",
    ]
    assert robot["control_parts"]["right_eef"] == [
        "right_finger_joint",
        "right_left_inner_knuckle_joint",
        "right_left_inner_finger_joint",
        "right_outer_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_inner_finger_joint",
    ]
    assert len(robot["init_qpos"]) == 24


def test_dual_ur5_template_deserializes_to_ur5_solver_cfg() -> None:
    robot_cfg = RobotCfg.from_dict(make_dual_ur5_robot_config(robot_init_z=0.42))

    for arm_name, root_link_name, end_link_name in (
        ("left_arm", "left_base_link", "left_ee_link"),
        ("right_arm", "right_base_link", "right_ee_link"),
    ):
        solver_cfg = robot_cfg.solver_cfg[arm_name]
        assert isinstance(solver_cfg, URSolverCfg)
        assert solver_cfg.class_type == "URSolver"
        assert solver_cfg.ur_type == "ur5"
        assert solver_cfg.urdf_path is None
        assert solver_cfg.root_link_name == root_link_name
        assert solver_cfg.end_link_name == end_link_name
        assert solver_cfg.d1 == pytest.approx(0.089159)
        assert solver_cfg.a2 == pytest.approx(-0.425)
        assert solver_cfg.a3 == pytest.approx(-0.39225)
        assert solver_cfg.tcp[2][3] == pytest.approx(0.16)


def test_observation_joint_ids_derive_from_dual_ur5_robot_config() -> None:
    robot = make_dual_ur5_robot_config(robot_init_z=0.42)

    observations = _make_observations_config(robot)

    assert observations["norm_robot_eef_joint"]["params"]["joint_ids"] == list(
        range(12, 24)
    )


def test_action_agent_config_generator_uses_parallel_handoff(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_agent",
        task_name="Demo111",
        target_body_scale=0.6,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in gym_config["background"]}

    assert set(rigid_objects) == {"left_apple", "right_apple", "wicker_basket"}
    assert rigid_objects["left_apple"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["right_apple"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["wicker_basket"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["left_apple"]["body_type"] == "dynamic"
    assert rigid_objects["right_apple"]["body_type"] == "dynamic"
    assert rigid_objects["wicker_basket"]["body_type"] == "dynamic"
    assert background_objects["table"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["left_apple"]["convex_decomposition_method"] == "vhacd"
    assert rigid_objects["right_apple"]["convex_decomposition_method"] == "vhacd"
    assert rigid_objects["wicker_basket"]["convex_decomposition_method"] == "vhacd"
    assert paths.summary["convex_decomposition_method"] == "vhacd"
    assert paths.summary["coacd_cache"][0]["status"] == "skipped"
    _assert_body_scaled_obj_path(rigid_objects["left_apple"]["shape"]["fpath"])
    _assert_body_scaled_obj_path(rigid_objects["right_apple"]["shape"]["fpath"])
    _assert_normalized_obj_path(background_objects["table"]["shape"]["fpath"])
    _assert_normalized_obj_path(rigid_objects["wicker_basket"]["shape"]["fpath"])
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
    for arm_name, root_link_name, end_link_name in (
        ("left_arm", "left_base_link", "left_ee_link"),
        ("right_arm", "right_base_link", "right_ee_link"),
    ):
        solver = gym_config["robot"]["solver_cfg"][arm_name]
        assert solver["class_type"] == "URSolver"
        assert solver["ur_type"] == "ur5"
        assert solver["urdf_path"] is None
        assert solver["root_link_name"] == root_link_name
        assert solver["end_link_name"] == end_link_name

    robot_cfg = RobotCfg.from_dict(gym_config["robot"])
    for arm_name, root_link_name, end_link_name in (
        ("left_arm", "left_base_link", "left_ee_link"),
        ("right_arm", "right_base_link", "right_ee_link"),
    ):
        solver_cfg = robot_cfg.solver_cfg[arm_name]
        assert isinstance(solver_cfg, URSolverCfg)
        assert solver_cfg.ur_type == "ur5"
        assert solver_cfg.urdf_path is None
        assert solver_cfg.root_link_name == root_link_name
        assert solver_cfg.end_link_name == end_link_name
        assert solver_cfg.d1 == pytest.approx(0.089159)
        assert solver_cfg.a2 == pytest.approx(-0.425)
        assert solver_cfg.a3 == pytest.approx(-0.39225)

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
    record_events = gym_config["env"]["events"]
    assert (
        record_events["record_camera"]["params"]["video_name"]
        == "Demo111_audience_view"
    )
    assert "record_camera_audience" not in record_events

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    task_graph = json.loads(paths.task_graph.read_text(encoding="utf-8"))
    agent_config = json.loads(paths.agent_config.read_text(encoding="utf-8"))
    basic_background = paths.basic_background.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    normalized_task_prompt = " ".join(task_prompt.split())

    assert agent_config["TaskAgent"]["precomputed_task_graph"] == "task_graph.json"
    assert task_graph["start"] == "v0_start"
    assert task_graph["goal"] == "v6_done"
    assert task_graph["nodes"][-1]["id"] == "v6_done"
    assert len(task_graph["nodes"]) == len(task_graph["edges"]) + 1
    assert len(task_graph["edges"]) == 6
    assert task_graph["edges"][3]["left_arm_action"]["target_qpos"] == {
        "source": "initial"
    }
    assert (
        task_graph["edges"][3]["right_arm_action"]["target_object_pose"]["obj_name"]
        == "wicker_basket"
    )

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
    body_scaled_meshes = _body_scaled_meshes_by_uid(paths.summary)
    assert body_scaled_meshes["left_apple"]["body_scale"] == [0.6, 0.6, 0.6]
    assert body_scaled_meshes["right_apple"]["body_scale"] == [0.6, 0.6, 0.6]

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


@pytest.mark.parametrize(
    (
        "robot_profile",
        "expected_uid",
        "expected_solver_type",
        "expected_meta_type",
        "expected_wrist_parents",
    ),
    [
        (
            "ur10",
            "DualUR10",
            "URSolver",
            "DualUR10",
            {
                "cam_wrist_left": "left_ee_link",
                "cam_wrist_right": "right_ee_link",
            },
        ),
        (
            "franka",
            "DualFrankaPanda",
            "PinocchioSolver",
            "DualFrankaPanda",
            {
                "cam_wrist_left": "left_ee_link",
                "cam_wrist_right": "right_ee_link",
            },
        ),
        (
            "franka_v3",
            "DualFrankaV3",
            "PytorchSolver",
            "DualFrankaV3",
            {
                "cam_wrist_left": "left_fr3_hand_tcp",
                "cam_wrist_right": "right_fr3_hand_tcp",
            },
        ),
    ],
)
def test_action_agent_config_generator_uses_selected_robot_profile(
    tmp_path: Path,
    robot_profile: str,
    expected_uid: str,
    expected_solver_type: str,
    expected_meta_type: str,
    expected_wrist_parents: dict[str, str],
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / f"generated_agent_{robot_profile}",
        robot_profile=robot_profile,
        target_body_scale=0.6,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    robot = gym_config["robot"]
    extensions = gym_config["env"]["extensions"]
    dataset_params = gym_config["env"]["dataset"]["lerobot"]["params"]
    basic_background = paths.basic_background.read_text(encoding="utf-8")
    wrist_parents = {
        sensor["uid"]: sensor["extrinsics"]["parent"]
        for sensor in gym_config["sensor"]
        if sensor["uid"] in expected_wrist_parents
    }

    assert robot["uid"] == expected_uid
    assert robot["solver_cfg"]["left_arm"]["class_type"] == expected_solver_type
    assert robot["solver_cfg"]["right_arm"]["class_type"] == expected_solver_type
    assert wrist_parents == expected_wrist_parents
    assert extensions["agent_robot_profile"] == paths.summary["robot_profile"]["id"]
    assert dataset_params["robot_meta"]["robot_type"] == expected_meta_type
    assert paths.summary["robot_profile"]["robot_meta_type"] == expected_meta_type
    assert paths.summary["robot_profile"]["display_name"] in basic_background


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
    assert rigid_objects["wicker_basket"]["init_rot"] == [0.0, 0.0, 180.0]
    assert rigid_objects["left_apple"]["init_rot"] == [0.0, 0.0, 140.0]
    assert rigid_objects["right_apple"]["init_rot"] == [0.0, 0.0, 160.0]
    for obj_config in [background_objects["table"], rigid_objects["wicker_basket"]]:
        _assert_normalized_obj_path(obj_config["shape"]["fpath"])
    for obj_config in [rigid_objects["right_apple"], rigid_objects["left_apple"]]:
        _assert_body_scaled_obj_path(obj_config["shape"]["fpath"])

    source_paths = {
        Path(entry["source_path"]).name for entry in paths.summary["normalized_meshes"]
    }
    assert source_paths == {
        "table_0.glb",
        "basket_3.glb",
        "apple_1.glb",
        "apple_2.glb",
    }
    for entry in paths.summary["normalized_meshes"]:
        assert _flatten_matrix(entry["transform"]) == pytest.approx(
            _flatten_matrix(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )


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
    transform = _obj_header_json_value(obj_text, "transform")
    assert _flatten_matrix(transform) == pytest.approx(
        _flatten_matrix(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
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


def test_body_scale_bake_writes_scaled_obj_and_clears_runtime_scale(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "normalized"
    source_dir.mkdir()
    mesh_path = source_dir / "triangle.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "mtllib material.mtl",
                "usemtl material_test",
                "v 1 2 3",
                "v -1 0 2",
                "v 0.5 1 0",
                "f 1 2 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (source_dir / "material.mtl").write_text(
        "newmtl material_test\nKd 1.0 1.0 1.0\n",
        encoding="utf-8",
    )
    gym_config = {
        "background": [],
        "rigid_object": [
            {
                "uid": "scaled_triangle",
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": mesh_path.as_posix(),
                },
                "body_scale": [2.0, 0.5, 3.0],
            }
        ],
    }

    reports = bake_body_scale_into_meshes(
        gym_config,
        output_dir=tmp_path / "body_scaled",
    )

    obj_config = gym_config["rigid_object"][0]
    scaled_path = Path(obj_config["shape"]["fpath"])
    assert obj_config["body_scale"] == [1.0, 1.0, 1.0]
    assert scaled_path.parent == tmp_path / "body_scaled"
    assert reports == [
        {
            "uid": "scaled_triangle",
            "section": "rigid_object",
            "source_path": mesh_path.as_posix(),
            "scaled_path": scaled_path.as_posix(),
            "source_sha256": reports[0]["source_sha256"],
            "body_scale": [2.0, 0.5, 3.0],
            "status": "generated",
            "policy_version": BODY_SCALE_BAKE_POLICY_VERSION,
        }
    ]
    assert _rounded_vertex_set(_obj_vertices(scaled_path)) == {
        (2.0, 1.0, 9.0),
        (-2.0, 0.0, 6.0),
        (1.0, 0.5, 0.0),
    }
    assert (scaled_path.parent / "material.mtl").is_file()


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

    assert set(rigid_objects) == {"left_apple", "right_apple", "wicker_basket"}
    assert rigid_objects["wicker_basket"]["body_type"] == "dynamic"
    _assert_body_scaled_obj_path(rigid_objects["right_apple"]["shape"]["fpath"])
    _assert_body_scaled_obj_path(rigid_objects["left_apple"]["shape"]["fpath"])
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

    assert set(rigid_objects) == {"left_apple", "right_orange", "wicker_basket"}
    assert rigid_objects["wicker_basket"]["body_type"] == "dynamic"
    _assert_body_scaled_obj_path(rigid_objects["left_apple"]["shape"]["fpath"])
    _assert_body_scaled_obj_path(rigid_objects["right_orange"]["shape"]["fpath"])

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


def test_target_replacements_accept_repeated_zero_to_n_objects(tmp_path: Path) -> None:
    project_dir = tmp_path / "gym_project"
    project_dir.mkdir()
    (project_dir / "gym_config.json").write_text(
        json.dumps(
            {
                "rigid_object": [
                    {"uid": "bread_1", "init_pos": [0.0, 0.2, 0.76]},
                    {"uid": "bread_2", "init_pos": [0.0, 0.0, 0.76]},
                    {"uid": "bread_3", "init_pos": [0.0, -0.2, 0.76]},
                ],
            }
        ),
        encoding="utf-8",
    )

    replacements = target_replacements_cli.resolve_target_replacements(
        SimpleNamespace(
            target_replacement=[
                ["bread_1", "red apple"],
                ["bread_2", "green pear"],
                ["bread_3", "yellow lemon"],
            ],
            target_replacement1=None,
            target_replacement2=None,
        ),
        TargetReplacementSpec,
        project_dir,
    )

    assert [replacement.source_uid for replacement in replacements] == [
        "bread_1",
        "bread_2",
        "bread_3",
    ]
    assert [replacement.output_dir_name for replacement in replacements] == [
        "new1",
        "new2",
        "new3",
    ]


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
        "wicker_basket",
        "vase_0",
    }
    assert rigid_objects["wicker_basket"]["body_type"] == "dynamic"
    assert rigid_objects["left_apple"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["right_apple"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["vase_0"]["body_scale"] == [1.0, 1.0, 1.0]
    _assert_body_scaled_obj_path(rigid_objects["vase_0"]["shape"]["fpath"])
    body_scaled_meshes = _body_scaled_meshes_by_uid(paths.summary)
    assert body_scaled_meshes["left_apple"]["body_scale"] == [0.8, 0.8, 0.8]
    assert body_scaled_meshes["right_apple"]["body_scale"] == [0.8, 0.8, 0.8]
    assert body_scaled_meshes["vase_0"]["body_scale"] == [1.2, 1.1, 0.9]


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
    assert set(rigid_objects) == {"apple_1", "apple_2", "wicker_basket"}
    assert set(background_objects) == {"table"}
    assert rigid_objects["apple_2"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["apple_2"]["body_type"] == "dynamic"
    assert rigid_objects["apple_1"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["apple_1"]["body_type"] == "dynamic"
    assert background_objects["table"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["wicker_basket"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["wicker_basket"]["body_type"] == "dynamic"
    assert rigid_objects["wicker_basket"]["init_rot"] == [0.0, 0.0, 180.0]
    assert _body_scaled_meshes_by_uid(paths.summary)["apple_2"]["body_scale"] == [
        0.5,
        0.5,
        0.5,
    ]

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
        "Generate one deterministic nominal graph with exactly 6 nominal edges"
        in task_prompt
    )
    assert '"atomic_action_class":"PickUp","robot_name":"right_arm"' in task_prompt
    assert '"offset":[0.0,0.16,0.01]' in task_prompt
    assert '"atomic_action_class":"Place","robot_name":"right_arm"' in task_prompt
    assert '"target_pose":{"reference":"relative","offset":[0.0,0.0,0.0]' in task_prompt
    assert '"obj_name":"apple_2"' in task_prompt
    assert "left_arm_action: null" in task_prompt
    assert "Generate exactly 10 nominal edges" not in task_prompt

    assert _stable_summary(paths.summary) == {
        "mode": "object_manipulation",
        "moved_object": "apple_2",
        "reference_object": "wicker_basket",
        "relation": "left_of",
        "active_arm": "right_arm",
        "release_offset": [0.0, 0.16, 0.01],
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
    assert set(rigid_objects) == {"apple_1", "apple_2", "wicker_basket"}
    assert set(background_objects) == {"table"}
    assert rigid_objects["apple_1"]["body_type"] == "dynamic"
    assert rigid_objects["apple_2"]["body_type"] == "dynamic"
    assert rigid_objects["wicker_basket"]["body_type"] == "dynamic"

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
    assert '"offset":[0.16,0.0,0.1]' in task_prompt
    assert '"offset":[0.16,0.0,0.1]' in atom_actions

    assert _stable_summary(paths.summary) == {
        "mode": "object_manipulation",
        "moved_object": "apple_1",
        "reference_object": "apple_2",
        "relation": "front_of",
        "active_arm": "right_arm",
        "release_offset": [0.16, 0.0, 0.0],
    }


def test_task_description_generates_coordinated_pickment_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        assert kwargs["task_description"] == "用双臂将 apple_1 往前移动"
        return {
            "manipulations": [
                {
                    "intent": "place_relative",
                    "moved_object": "apple_1",
                    "reference_object": "apple_1",
                    "goal_relation": "front_of",
                    "arm": "auto",
                }
            ],
            "task_prompt_summary": "Use both arms to move apple_1 forward.",
            "basic_background_notes": "One shared object requires both arms.",
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
        tmp_path / "generated_coordinated_pickment_agent",
        task_name="MoveAppleForwardWithBothArms",
        task_description="用双臂将 apple_1 往前移动",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    assert "left_arm" in gym_config["robot"]["control_parts"]
    assert "right_arm" in gym_config["robot"]["control_parts"]
    success_spec = gym_config["env"]["extensions"]["agent_success"]
    assert "object_held_by_gripper" not in json.dumps(success_spec)
    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    for text in (task_prompt, atom_actions):
        assert '"atomic_action_class":"CoordinatedPickment"' in text
        assert '"robot_name":"dual_arm"' in text
        assert '"target_object":{"obj_name":"apple_1","affordance":"antipodal"}' in text
        assert '"robot_name":"left_arm","control":"hand"' in text
        assert '"robot_name":"right_arm","control":"hand"' in text
        assert '"target_qpos":{"source":"gripper_state","state":"open"}' in text
        assert '"robot_name":"left_arm","control":"arm"' in text
        assert '"robot_name":"right_arm","control":"arm"' in text
        assert '"target_qpos":{"source":"initial"}' in text
        assert '"atomic_action_class":"PickUp"' not in text
    assert "exactly 3 nominal edges" in task_prompt
    assert "must not remain held" in task_prompt
    assert "Both arms must be back at their initial" in task_prompt
    assert "may remain held" not in task_prompt
    assert '"position":[0.54,0.11' in task_prompt

    assert _stable_summary(paths.summary) == {
        "mode": "coordinated_pickment",
        "intent": "coordinated_pickment",
        "moved_object": "apple_1",
        "reference_object": "apple_1",
        "relation": "front_of",
        "active_arm": "dual_arm",
        "release_offset": [0.16, 0.0, 0.12],
        "target_position": paths.summary["target_position"],
    }


def test_coordinated_pickment_side_relation_preserves_object_height(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        assert kwargs["task_description"] == "用双臂将 apple_1 往右移动"
        return {
            "manipulations": [
                {
                    "intent": "place_relative",
                    "moved_object": "apple_1",
                    "reference_object": "table",
                    "goal_relation": "right_of",
                    "arm": "auto",
                }
            ],
            "task_prompt_summary": "Use both arms to move apple_1 right.",
            "basic_background_notes": "One shared object requires both arms.",
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
        tmp_path / "generated_coordinated_pickment_right_agent",
        task_name="MoveAppleRightWithBothArms",
        task_description="用双臂将 apple_1 往右移动",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    object_configs = {
        obj["uid"]: obj
        for group in ("rigid_object", "background")
        for obj in gym_config[group]
    }
    expected_z_offset = round(
        object_configs["apple_1"]["init_pos"][2]
        - object_configs["table"]["init_pos"][2],
        6,
    )
    task_prompt = paths.task_prompt.read_text(encoding="utf-8")

    assert paths.summary["release_offset"] == pytest.approx(
        [0.0, -0.16, expected_z_offset]
    )
    assert '"reference":"object","obj_name":"table"' in task_prompt
    assert f'"offset":[0.0,-0.16,{expected_z_offset}]' in task_prompt


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
        "mode": "object_manipulation",
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
    assert '"offset":[0.16,-0.16,0.01]' in task_prompt
    assert _stable_summary(paths.summary)["release_offset"] == [0.16, -0.16, 0.01]


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
    assert set(rigid_objects) == {"apple_1", "apple_2", "wicker_basket"}
    assert set(background_objects) == {"table"}
    assert rigid_objects["apple_1"]["body_type"] == "dynamic"
    assert rigid_objects["apple_2"]["body_type"] == "dynamic"
    assert rigid_objects["wicker_basket"]["body_type"] == "dynamic"
    assert rigid_objects["wicker_basket"]["init_rot"] == [0.0, 0.0, 180.0]
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
    assert set(rigid_objects) == {"apple_1", "apple_2", "wicker_basket"}
    assert set(background_objects) == {"table"}
    assert rigid_objects["apple_2"]["body_type"] == "dynamic"
    assert rigid_objects["apple_1"]["body_type"] == "dynamic"
    assert rigid_objects["wicker_basket"]["body_type"] == "dynamic"
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
    assert set(rigid_objects) == {"cup", "pad", "fork"}
    assert set(background_objects) == {"table"}
    assert rigid_objects["cup"]["body_type"] == "dynamic"
    assert rigid_objects["cup"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["pad"]["body_type"] == "dynamic"
    assert rigid_objects["pad"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["fork"]["body_type"] == "dynamic"
    assert rigid_objects["fork"]["body_scale"] == [1.0, 1.0, 1.0]
    body_scaled_meshes = _body_scaled_meshes_by_uid(paths.summary)
    assert body_scaled_meshes["cup"]["body_scale"] == [0.8, 0.8, 0.8]
    assert body_scaled_meshes["pad"]["body_scale"] == [1.2, 1.0, 0.4]
    assert body_scaled_meshes["fork"]["body_scale"] == [0.7, 0.7, 0.7]

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["type"] == "object_on_object"
    assert success["object"] == "cup"
    assert success["support"] == "pad"

    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert atom_actions.count('"atomic_action_class":"PickUp"') == 1
    assert '"atomic_action_class":"PickUp","robot_name":"right_arm"' in atom_actions
    assert '"obj_name":"cup"' in atom_actions
    assert _stable_summary(paths.summary)["relation"] == "on"


def test_prompt2scene_relative_placement_preserves_metric_source_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "prompt2scene_demo/gym_export"
    _write_demo3_role_project(project_dir)
    gym_config_path = project_dir / "gym_config.json"
    gym_config = json.loads(gym_config_path.read_text(encoding="utf-8"))
    gym_config["id"] = "Prompt2Scene-test-v0"
    gym_config["background"][0]["body_scale"] = [1.31, 1.32, 1.0]
    gym_config["rigid_object"][0]["body_scale"] = [0.11, 0.12, 0.13]
    gym_config["rigid_object"][1]["body_scale"] = [0.21, 0.22, 0.23]
    gym_config["rigid_object"][2]["body_scale"] = [0.31, 0.32, 0.33]
    source_cup_pos = list(gym_config["rigid_object"][0]["init_pos"])
    source_pad_pos = list(gym_config["rigid_object"][1]["init_pos"])
    source_cup_rot = list(gym_config["rigid_object"][0]["init_rot"])
    source_pad_rot = list(gym_config["rigid_object"][1]["init_rot"])
    source_cup_z = gym_config["rigid_object"][0]["init_pos"][2]
    source_pad_z = gym_config["rigid_object"][1]["init_pos"][2]
    gym_config_path.write_text(json.dumps(gym_config, indent=2), encoding="utf-8")

    def fake_call_relative_task_llm(**kwargs):
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
        gym_config_path,
        tmp_path / "generated_prompt2scene_relative_agent",
        task_description="用右臂把咖啡杯子放到垫子上",
        target_body_scale=0.8,
        source_scene_body_scale_mode="preserve",
        preserve_source_scene_geometry=True,
        source_scene_z_rotation_degrees=-90.0,
        source_mesh_x_rotation_degrees=90.0,
        prewarm_coacd_cache=False,
    )

    generated = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in generated["rigid_object"]}
    background_objects = {obj["uid"]: obj for obj in generated["background"]}

    assert set(background_objects) == {"table"}
    assert rigid_objects["cup"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["pad"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["fork"]["body_scale"] == [1.0, 1.0, 1.0]
    assert background_objects["table"]["body_scale"] == [1.0, 1.0, 1.0]
    assert Path(rigid_objects["cup"]["shape"]["fpath"]).suffix == ".obj"
    assert Path(rigid_objects["pad"]["shape"]["fpath"]).suffix == ".obj"
    assert "mesh_assets/body_scaled" in rigid_objects["cup"]["shape"]["fpath"]
    assert "mesh_assets/body_scaled" in rigid_objects["pad"]["shape"]["fpath"]
    body_scaled_meshes = _body_scaled_meshes_by_uid(paths.summary)
    assert body_scaled_meshes["cup"]["body_scale"] == [0.11, 0.12, 0.13]
    assert body_scaled_meshes["table"]["body_scale"] == [1.31, 1.32, 1.0]
    assert body_scaled_meshes["pad"]["body_scale"] == [0.21, 0.22, 0.23]
    assert body_scaled_meshes["fork"]["body_scale"] == [0.31, 0.32, 0.33]
    assert rigid_objects["cup"]["init_pos"][2] == source_cup_z
    assert rigid_objects["pad"]["init_pos"][2] == source_pad_z
    assert rigid_objects["cup"]["init_pos"] == pytest.approx(
        [source_cup_pos[1], -source_cup_pos[0], source_cup_pos[2]]
    )
    assert rigid_objects["pad"]["init_pos"] == pytest.approx(
        [source_pad_pos[1], -source_pad_pos[0], source_pad_pos[2]]
    )
    assert rigid_objects["cup"]["init_rot"] == pytest.approx(
        [source_cup_rot[0], source_cup_rot[1], source_cup_rot[2] - 90.0]
    )
    assert rigid_objects["pad"]["init_rot"] == pytest.approx(
        [source_pad_rot[0], source_pad_rot[1], source_pad_rot[2] - 90.0]
    )
    assert paths.summary["mode"] == "object_manipulation"
    assert "normalized_meshes" in paths.summary
    for entry in paths.summary["normalized_meshes"]:
        assert _flatten_matrix(entry["transform"]) == pytest.approx(
            _flatten_matrix(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

    scaled_paths = generate_action_agent_config_from_project(
        gym_config_path,
        tmp_path / "generated_prompt2scene_scaled_relative_agent",
        task_description="用右臂把咖啡杯子放到垫子上",
        target_body_scale=0.8,
        source_scene_body_scale_mode="multiply",
        preserve_source_scene_geometry=True,
        source_scene_z_rotation_degrees=-90.0,
        source_mesh_x_rotation_degrees=90.0,
        prewarm_coacd_cache=False,
    )
    scaled_generated = json.loads(scaled_paths.gym_config.read_text(encoding="utf-8"))
    scaled_rigid_objects = {obj["uid"]: obj for obj in scaled_generated["rigid_object"]}
    scaled_background_objects = {
        obj["uid"]: obj for obj in scaled_generated["background"]
    }
    assert scaled_rigid_objects["cup"]["body_scale"] == [1.0, 1.0, 1.0]
    assert scaled_rigid_objects["pad"]["body_scale"] == [1.0, 1.0, 1.0]
    assert scaled_rigid_objects["fork"]["body_scale"] == [1.0, 1.0, 1.0]
    assert scaled_background_objects["table"]["body_scale"] == [1.0, 1.0, 1.0]
    scaled_body_meshes = _body_scaled_meshes_by_uid(scaled_paths.summary)
    assert scaled_body_meshes["cup"]["body_scale"] == pytest.approx(
        [value * 0.8 for value in [0.11, 0.12, 0.13]]
    )
    assert scaled_body_meshes["table"]["body_scale"] == pytest.approx(
        [value * 0.8 for value in [1.31, 1.32, 1.0]]
    )
    assert scaled_body_meshes["pad"]["body_scale"] == pytest.approx(
        [value * 0.8 for value in [0.21, 0.22, 0.23]]
    )
    assert scaled_body_meshes["fork"]["body_scale"] == pytest.approx(
        [value * 0.8 for value in [0.31, 0.32, 0.33]]
    )
    assert _mesh_config_world_z_bounds(scaled_rigid_objects["cup"])[0] == pytest.approx(
        _mesh_config_world_z_bounds(rigid_objects["cup"])[0]
    )
    assert _mesh_config_world_z_bounds(scaled_rigid_objects["pad"])[0] == pytest.approx(
        _mesh_config_world_z_bounds(rigid_objects["pad"])[0]
    )
    assert _mesh_config_world_z_bounds(scaled_rigid_objects["fork"])[
        0
    ] == pytest.approx(_mesh_config_world_z_bounds(rigid_objects["fork"])[0])
    assert _mesh_config_world_z_bounds(scaled_background_objects["table"])[
        1
    ] == pytest.approx(_mesh_config_world_z_bounds(background_objects["table"])[1])

    absolute_paths = generate_action_agent_config_from_project(
        gym_config_path,
        tmp_path / "generated_prompt2scene_absolute_relative_agent",
        task_description="用右臂把咖啡杯子放到垫子上",
        target_body_scale=1.0,
        source_scene_body_scale_mode="absolute",
        preserve_source_scene_geometry=True,
        source_scene_z_rotation_degrees=-90.0,
        source_mesh_x_rotation_degrees=90.0,
        prewarm_coacd_cache=False,
    )
    absolute_generated = json.loads(
        absolute_paths.gym_config.read_text(encoding="utf-8")
    )
    absolute_rigid_objects = {
        obj["uid"]: obj for obj in absolute_generated["rigid_object"]
    }
    absolute_background_objects = {
        obj["uid"]: obj for obj in absolute_generated["background"]
    }
    assert absolute_rigid_objects["cup"]["body_scale"] == [1.0, 1.0, 1.0]
    assert absolute_rigid_objects["pad"]["body_scale"] == [1.0, 1.0, 1.0]
    assert absolute_rigid_objects["fork"]["body_scale"] == [1.0, 1.0, 1.0]
    assert absolute_background_objects["table"]["body_scale"] == [1.0, 1.0, 1.0]


def test_apply_scene_z_rotation_rotates_scene_object_poses() -> None:
    from scipy.spatial.transform import Rotation

    gym_config = {
        "background": [
            {
                "uid": "table",
                "init_pos": [0.2, -0.3, 0.4],
                "init_rot": [10.0, 20.0, 30.0],
            }
        ],
        "rigid_object": [
            {
                "uid": "cup",
                "init_pos": [-0.1, 0.5, 0.7],
                "init_rot": [0.0, 0.0, -45.0],
            }
        ],
    }

    action_agent_config_generation._apply_scene_z_rotation(gym_config, 90.0)

    assert gym_config["background"][0]["init_pos"] == pytest.approx([0.3, 0.2, 0.4])
    expected_table_rot = (
        Rotation.from_euler("z", 90.0, degrees=True)
        * Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True)
    ).as_euler("xyz", degrees=True)
    assert gym_config["background"][0]["init_rot"] == pytest.approx(expected_table_rot)
    assert gym_config["rigid_object"][0]["init_pos"] == pytest.approx([-0.5, -0.1, 0.7])
    assert gym_config["rigid_object"][0]["init_rot"] == pytest.approx([0.0, 0.0, 45.0])


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


def test_relative_cube_preserves_orientation_when_llm_preserves_orientation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "21_Place_A2B_Right_gym_project"
    for rel_path in (
        "mesh_assets/table/table_0.glb",
        "mesh_assets/cube/cube_1.glb",
        "mesh_assets/cube/cube_2.glb",
    ):
        _write_minimal_glb(project_dir / rel_path, _default_mesh_vertices())

    gym_config = {
        "id": "Image2Tabletop-21-v0",
        "background": [
            _mesh_object(
                "table_0",
                "mesh_assets/table/table_0.glb",
                [0.0, 0.0, -0.05],
                [0.0, 0.0, 0.0],
            )
        ],
        "rigid_object": [
            _mesh_object(
                "interact_cube_1",
                "mesh_assets/cube/cube_1.glb",
                [0.0, 0.12, 0.0],
                [0.0, 0.0, 0.0],
            ),
            _mesh_object(
                "interact_cube_2",
                "mesh_assets/cube/cube_2.glb",
                [0.0, -0.12, 0.0],
                [0.0, 0.0, 0.0],
            ),
        ],
    }
    (project_dir / "gym_config.json").write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )

    def fake_call_relative_task_llm(**kwargs):
        return {
            "moved_object": "interact_cube_1",
            "reference_object": "interact_cube_2",
            "goal_relation": "right_of",
            "arm": "auto",
            "orientation_goal": "preserve",
            "orientation_reference": "none",
            "orientation_axis": "none",
            "task_prompt_summary": "Move cube_1 to the right of cube_2.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_cube_relative_agent",
        task_name="Demo21",
        task_description="将一个方块移动到另一个方块右边",
        target_body_scale=0.8,
        prewarm_coacd_cache=False,
    )

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    summary = _stable_summary(paths.summary)
    high_offset = list(summary["release_offset"])
    high_offset[2] = round(float(high_offset[2]) + 0.25, 6)
    high_offset_json = json.dumps(
        high_offset, ensure_ascii=False, separators=(",", ":")
    )
    release_offset_json = json.dumps(
        summary["release_offset"], ensure_ascii=False, separators=(",", ":")
    )

    assert (
        "Generate one deterministic nominal graph with exactly 6 nominal edges"
        in task_prompt
    )
    for text in (task_prompt, atom_actions):
        assert (
            f'"offset":{high_offset_json},"orientation_goal":"preserve",'
            '"orientation_axis":"none"}' in text
        )
        assert (
            f'"offset":{release_offset_json},"orientation_goal":"preserve",'
            '"orientation_axis":"none"}' in text
        )
        assert '"orientation_goal":"axis_align"' not in text
        assert '"align_to"' not in text
        assert '"atomic_action_class":"Place"' in text
        assert '"atomic_action_class":"MoveEndEffector"' in text

    assert summary["moved_object"] == "interact_cube_1"
    assert summary["reference_object"] == "interact_cube_2"
    assert paths.summary["orientation_goal"] == "preserve"
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
    bottle_half_height = 0.08
    bottle_semantic_z_half_height = 0.01
    bottle_vertices = [
        (-0.02, -bottle_half_height, -0.01),
        (0.02, -bottle_half_height, -0.01),
        (0.0, bottle_half_height, 0.01),
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

    target_body_scale = 0.8
    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_bottle_on_table_agent",
        task_name="Demo43",
        task_description="用左臂把瓶子扶正放到桌面上",
        target_body_scale=target_body_scale,
        prewarm_coacd_cache=False,
    )

    summary = _stable_summary(paths.summary)
    expected_release_offset_z = (
        0.36 + 0.003 + bottle_semantic_z_half_height * target_body_scale
    )
    expected_release_position = [
        0.05,
        0.05,
        -0.02 + expected_release_offset_z,
    ]
    expected_release_offset = [0.05, 0.05, expected_release_offset_z]
    assert summary["upright_in_place"] is True
    assert summary["release_offset"][2] == pytest.approx(expected_release_offset_z)
    assert summary["release_offset"] == pytest.approx(expected_release_offset)
    assert summary["reference_object"] == "table"
    assert summary["pickup_rotate_upright"] == pytest.approx(0.7853981633974483)
    assert sum(abs(value) for value in summary["pickup_upright_direction"]) == 1.0

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    moved_object = summary["moved_object"]
    events = gym_config["env"]["events"]
    registry = events["register_info_to_env"]["params"]["registry"]
    registered_uids = {entry["entity_cfg"]["uid"] for entry in registry}
    assert {moved_object, "table"}.issubset(registered_uids)

    extensions = gym_config["env"]["extensions"]
    assert (
        extensions["agent_grasp_pose_overrides"][moved_object]["mode"]
        == "upright_bottle_side_grasp"
    )
    success = extensions["agent_success"]
    assert success["op"] == "all"
    assert {term["type"] for term in success["terms"]} == {
        "object_axis_near",
        "object_not_fallen",
    }
    axis_targets = {
        term["axis"]: term["target"]
        for term in success["terms"]
        if term["type"] == "object_axis_near"
    }
    assert axis_targets == pytest.approx({"x": 0.05, "y": 0.05})

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    release_position_json = json.dumps(
        expected_release_position, ensure_ascii=False, separators=(",", ":")
    )
    pickup_direction_json = json.dumps(
        summary["pickup_upright_direction"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert f'"obj_upright_direction":{pickup_direction_json}' in task_prompt
    assert '"rotate_upright":0.7853981633974483' in task_prompt
    assert (
        f'"position":{release_position_json},"orientation_goal":"upright"'
        in task_prompt
    )
    assert '"reference":"object","obj_name":"table"' not in task_prompt
    assert '"offset":[0.0,0.0,0.2]' not in task_prompt


def test_dual_upright_in_place_supports_cup_like_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "44_Upright Cup And Can_gym_project"
    _write_dual_upright_cup_can_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "manipulations": [
                {
                    "moved_object": "paper_cup_1",
                    "reference_object": "table",
                    "goal_relation": "on",
                    "arm": "left",
                    "orientation_goal": "upright",
                    "orientation_reference": "none",
                },
                {
                    "moved_object": "soda_can_1",
                    "reference_object": "table",
                    "goal_relation": "on",
                    "arm": "right",
                    "orientation_goal": "upright",
                    "orientation_reference": "none",
                },
            ],
            "task_prompt_summary": "Use both arms to stand the cup and can upright.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_dual_upright_agent",
        task_name="Demo44",
        task_description="左臂扶正纸杯放回桌面，右臂扶正罐头放回桌面",
        target_body_scale=0.8,
        prewarm_coacd_cache=False,
    )

    assert paths.summary["mode"] == "dual_arm_object_manipulation"
    manipulations = {
        item["moved_object"]: item for item in paths.summary["manipulations"]
    }
    assert set(manipulations) == {"paper_cup", "soda_can"}
    assert manipulations["paper_cup"]["upright_in_place"] is True
    assert manipulations["paper_cup"]["orientation_goal"] == "upright"
    assert manipulations["paper_cup"]["active_arm"] == "left_arm"
    assert manipulations["paper_cup"]["release_offset"][:2] == pytest.approx(
        [0.06, 0.10]
    )
    assert manipulations["soda_can"]["upright_in_place"] is True
    assert manipulations["soda_can"]["orientation_goal"] == "upright"
    assert manipulations["soda_can"]["active_arm"] == "right_arm"
    assert manipulations["soda_can"]["release_offset"][:2] == pytest.approx(
        [0.05, -0.10]
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    events = gym_config["env"]["events"]
    registry = events["register_info_to_env"]["params"]["registry"]
    registered_uids = {entry["entity_cfg"]["uid"] for entry in registry}
    assert {"paper_cup", "soda_can", "table"}.issubset(registered_uids)

    extensions = gym_config["env"]["extensions"]
    assert set(extensions["agent_grasp_pose_overrides"]) == {
        "paper_cup",
        "soda_can",
    }
    assert {
        override["mode"]
        for override in extensions["agent_grasp_pose_overrides"].values()
    } == {"upright_bottle_side_grasp"}

    success = extensions["agent_success"]
    assert success["op"] == "all"
    per_object_terms = {
        placement_success["terms"][0]["object"]: {
            term["type"] for term in placement_success["terms"]
        }
        for placement_success in success["terms"]
    }
    assert per_object_terms == {
        "paper_cup": {"object_axis_near", "object_not_fallen"},
        "soda_can": {"object_axis_near", "object_not_fallen"},
    }

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert task_prompt.count('"orientation_goal":"upright"') >= 2
    assert '"obj_name":"table"' not in task_prompt


def test_generation_upright_known_z_normalized_asset_uses_local_z(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "wide_soda_can.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v -0.15 -0.12 0.00",
                "v 0.15 -0.12 0.00",
                "v -0.15 0.12 0.00",
                "v 0.15 0.12 0.00",
                "v -0.15 -0.12 0.08",
                "v 0.15 -0.12 0.08",
                "v -0.15 0.12 0.08",
                "v 0.15 0.12 0.08",
            ]
        ),
        encoding="utf-8",
    )
    obj_config = {
        "uid": "soda_can",
        "shape": {"shape_type": "Mesh", "fpath": str(mesh_path)},
        "body_scale": [1.0, 1.0, 1.0],
        "init_pos": [0.0, 0.0, 0.0],
        "init_rot": [90.0, 0.0, 0.0],
    }

    assert relative_geometry._pickup_upright_direction(obj_config) == [0.0, 0.0, 1.0]
    assert relative_geometry._upright_local_zmin(obj_config) == pytest.approx(0.0)


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
            '"orientation_goal":"preserve","orientation_axis":"none",'
            '"z_policy":"object_on_surface",'
            f'"support":"{reference_object}","surface_clearance":0.015' in text
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
    assert set(rigid_objects) == {"chip_bag", "pad"}
    assert set(background_objects) == {"table"}
    assert rigid_objects["chip_bag"]["body_type"] == "dynamic"
    assert rigid_objects["chip_bag"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["pad"]["body_type"] == "dynamic"
    assert _body_scaled_meshes_by_uid(paths.summary)["chip_bag"]["body_scale"] == [
        0.5,
        0.5,
        0.5,
    ]
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
    assert set(rigid_objects) == {"apple_1", "apple_2", "wicker_basket"}
    assert set(background_objects) == {"table"}
    assert rigid_objects["apple_1"]["body_type"] == "dynamic"
    assert rigid_objects["apple_2"]["body_type"] == "dynamic"
    assert rigid_objects["wicker_basket"]["body_type"] == "dynamic"
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
        "mode": "dual_arm_object_manipulation",
        "manipulations": [
            {
                "moved_object": "apple_2",
                "reference_object": "wicker_basket",
                "relation": "left_of",
                "active_arm": "left_arm",
                "release_offset": [0.0, 0.16, 0.01],
            },
            {
                "moved_object": "apple_1",
                "reference_object": "wicker_basket",
                "relation": "right_of",
                "active_arm": "right_arm",
                "release_offset": [0.0, -0.16, 0.01],
            },
        ],
    }

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    basic_background = paths.basic_background.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert "Generate one deterministic nominal graph with exactly 10 nominal edges" in (
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
    assert '"offset":[0.0,0.16,0.01]' in task_prompt
    assert '"offset":[0.0,-0.16,0.01]' in task_prompt
    assert '"atomic_action_class":"Place","robot_name":"left_arm"' in task_prompt
    assert '"atomic_action_class":"Place","robot_name":"right_arm"' in task_prompt
    assert '"target_pose":{"reference":"relative","offset":[0.0,0.0,0.0]' in task_prompt
    assert "The inactive arm must remain null" not in task_prompt
    assert "Both arms participate" in basic_background
    assert "left_arm moves `apple_2`" in basic_background
    assert "right_arm moves `apple_1`" in basic_background
    assert '"atomic_action_class":"PickUp","robot_name":"left_arm"' in atom_actions
    assert '"obj_name":"apple_2"' in atom_actions
    assert '"atomic_action_class":"PickUp","robot_name":"right_arm"' in atom_actions
    assert '"obj_name":"apple_1"' in atom_actions


def test_dual_upright_prompt_preserves_before_orientation_adjustment() -> None:
    rotate_upright = 0.7853981633974483
    left = _RelativePlacementStepSpec(
        intent="place_relative",
        moved_source_uid="left_bottle_src",
        reference_source_uid="table_src",
        moved_runtime_uid="left_bottle",
        reference_runtime_uid="table",
        relation="on",
        active_side="left",
        release_offset=[0.1, 0.2, 0.3],
        high_offset=[0.1, 0.2, 0.55],
        release_position=[0.1, 0.2, 0.3],
        high_position=[0.1, 0.2, 0.55],
        orientation_goal="upright",
        orientation_axis="none",
        upright_in_place=True,
        pickup_upright_direction=[1.0, 0.0, 0.0],
        pickup_rotate_upright=rotate_upright,
    )
    right = _RelativePlacementStepSpec(
        intent="place_relative",
        moved_source_uid="right_bottle_src",
        reference_source_uid="table_src",
        moved_runtime_uid="right_bottle",
        reference_runtime_uid="table",
        relation="on",
        active_side="right",
        release_offset=[0.1, -0.2, 0.3],
        high_offset=[0.1, -0.2, 0.55],
        release_position=[0.1, -0.2, 0.3],
        high_position=[0.1, -0.2, 0.55],
        orientation_goal="upright",
        orientation_axis="none",
        upright_in_place=True,
        pickup_upright_direction=[0.0, 1.0, 0.0],
        pickup_rotate_upright=rotate_upright,
    )
    spec = _RelativePlacementSpec(
        intent="place_relative",
        table_source_uid="table_src",
        moved_source_uid=left.moved_source_uid,
        reference_source_uid=left.reference_source_uid,
        moved_runtime_uid=left.moved_runtime_uid,
        reference_runtime_uid=left.reference_runtime_uid,
        relation=left.relation,
        active_side=left.active_side,
        task_description="用双臂分别将双边的瓶子扶正",
        task_prompt_summary="Stand both bottles upright in place.",
        basic_background_notes="Both bottles start on their sides.",
        action_sketch=["Pick both bottles", "Stand each bottle upright"],
        release_offset=left.release_offset,
        high_offset=left.high_offset,
        placements=(left, right),
        release_position=left.release_position,
        high_position=left.high_position,
        orientation_goal=left.orientation_goal,
        orientation_axis=left.orientation_axis,
        upright_in_place=True,
        pickup_upright_direction=left.pickup_upright_direction,
        pickup_rotate_upright=left.pickup_rotate_upright,
    )

    prompt = make_relative_task_prompt("DemoUpright", "demo_project", spec)

    assert "Generate one deterministic nominal graph with exactly 12 nominal edges" in (
        prompt
    )
    assert '"obj_upright_direction":[1.0,0.0,0.0]' in prompt
    assert '"obj_upright_direction":[0.0,1.0,0.0]' in prompt
    assert f'"rotate_upright":{rotate_upright}' in prompt
    left_high_preserve = (
        '"position":[0.1,0.2,0.55],'
        '"orientation_goal":"preserve","orientation_axis":"none"'
    )
    left_high_upright = (
        '"position":[0.1,0.2,0.55],'
        '"orientation_goal":"upright","orientation_axis":"none"'
    )
    assert left_high_preserve in prompt
    assert left_high_upright in prompt
    assert prompt.index(left_high_preserve) < prompt.index(left_high_upright)


def test_task_description_generates_dual_hold_hover_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "19_Pick Dual Bottles_gym_project"
    _write_dual_bottles_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        assert kwargs["task_description"] == (
            "用一只机械臂拿起一个瓶子悬空，并用另一只机械臂拿起另一个瓶子也悬空。"
        )
        return {
            "manipulations": [
                {
                    "intent": "hold_hover",
                    "moved_object": "interact_soda_bottle_1",
                    "arm": "left",
                    "hover_height": 0.10,
                },
                {
                    "intent": "hold_hover",
                    "moved_object": "interact_soda_bottle_2",
                    "arm": "right",
                    "hover_height": 0.10,
                },
            ],
            "task_prompt_summary": "Pick up both bottles and keep them hovering.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fake_call_relative_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_dual_hold_hover_agent",
        task_name="Demo19",
        task_description=(
            "用一只机械臂拿起一个瓶子悬空，并用另一只机械臂拿起另一个瓶子也悬空。"
        ),
        target_body_scale=0.8,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    summary = paths.summary
    assert summary["mode"] == "dual_arm_object_manipulation"
    assert [item["intent"] for item in summary["manipulations"]] == [
        "hold_hover",
        "hold_hover",
    ]

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert "exactly 3 nominal edges" in task_prompt
    assert task_prompt.count('"atomic_action_class":"PickUp"') == 2
    assert task_prompt.count('"atomic_action_class":"MoveHeldObject"') == 2
    assert (
        task_prompt.count('"target_qpos":{"source":"gripper_state","state":"close"}')
        == 2
    )
    assert '"atomic_action_class":"Place"' not in task_prompt
    assert '"source":"initial"' not in task_prompt
    assert '"atomic_action_class":"Place"' not in atom_actions
    assert '"source":"initial"' not in atom_actions

    success_terms = gym_config["env"]["extensions"]["agent_success"]["terms"]
    lifted = [term for term in success_terms if term["type"] == "object_lifted"]
    held = [term for term in success_terms if term["type"] == "object_held_by_gripper"]
    assert {term["object"] for term in lifted} == {
        "interact_soda_bottle_1",
        "interact_soda_bottle_2",
    }
    assert {(term["object"], term["arm"]) for term in held} == {
        ("interact_soda_bottle_1", "left_arm"),
        ("interact_soda_bottle_2", "right_arm"),
    }


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
    assert [step.target_xy[0] for step in spec.steps] == sorted(
        step.target_xy[0] for step in spec.steps
    )
    assert [step.orientation_goal for step in spec.steps] == [
        "preserve",
        "preserve",
        "preserve",
    ]
    assert [step.orientation_axis for step in spec.steps] == ["none", "none", "none"]


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


def test_arrangement_line_slot_positions_support_world_x() -> None:
    slots = _arrangement_line_slot_positions(
        anchor_xy=[0.10, -0.20],
        count=3,
        spacing=0.08,
        line_axis="world_x",
    )

    assert slots == [
        [0.02, -0.20],
        [0.10, -0.20],
        [0.18, -0.20],
    ]


def test_arrangement_table_long_axis_resolves_from_table_bounds() -> None:
    wide_slots = _arrangement_line_slot_positions(
        anchor_xy=[0.0, 0.0],
        count=3,
        spacing=0.10,
        line_axis="table_long_axis",
        table_bounds=([-0.60, -0.20], [0.60, 0.20]),
    )
    tall_slots = _arrangement_line_slot_positions(
        anchor_xy=[0.0, 0.0],
        count=3,
        spacing=0.10,
        line_axis="table_long_axis",
        table_bounds=([-0.20, -0.60], [0.20, 0.60]),
    )
    square_slots = _arrangement_line_slot_positions(
        anchor_xy=[0.0, 0.0],
        count=3,
        spacing=0.10,
        line_axis="table_long_axis",
        table_bounds=([-0.50, -0.50], [0.50, 0.50]),
    )

    assert [slot[0] for slot in wide_slots] == [-0.10, 0.0, 0.10]
    assert len({slot[1] for slot in wide_slots}) == 1
    assert len({slot[0] for slot in tall_slots}) == 1
    assert [slot[1] for slot in tall_slots] == [-0.10, 0.0, 0.10]
    assert len({slot[0] for slot in square_slots}) == 1
    assert [slot[1] for slot in square_slots] == [-0.10, 0.0, 0.10]


def test_demo153_like_cans_can_reuse_moved_initial_positions(
    tmp_path: Path,
) -> None:
    scene_objects = _write_narrow_four_can_arrangement_scene(tmp_path)
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]

    spec = _apply_arrangement_task_response(
        response={
            "objects": ["can_1", "can_2", "can_3", "can_4"],
            "order_by": "explicit",
            "order_direction": "given",
            "anchor": "table_center",
            "line_axis": "table_long_axis",
            "task_prompt_summary": "Arrange four cans in a straight row.",
        },
        table_source_uid="table",
        scene_objects=scene_objects,
        rigid_objects=rigid_objects,
        scene_dir=tmp_path,
        task_description="将桌面上的罐头摆成一排",
    )

    target_x_values = [step.target_xy[0] for step in spec.steps]
    target_y_values = [step.target_xy[1] for step in spec.steps]
    assert target_x_values == sorted(target_x_values)
    assert len({round(value, 6) for value in target_y_values}) == 1
    assert [step.orientation_goal for step in spec.steps] == [
        "preserve",
        "preserve",
        "preserve",
        "preserve",
    ]
    assert [step.orientation_axis for step in spec.steps] == [
        "none",
        "none",
        "none",
        "none",
    ]


def test_arrangement_keeps_axis_alignment_for_elongated_objects(
    tmp_path: Path,
) -> None:
    _write_minimal_glb(
        tmp_path / "mesh_assets/table/table_0.glb",
        [(-0.35, -0.20, 0.0), (0.35, -0.20, 0.0), (0.0, 0.20, 0.0)],
    )
    bar_vertices = [(-0.06, -0.01, 0.0), (0.06, -0.01, 0.0), (0.0, 0.01, 0.03)]
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
        )
    ]
    for index, y_value in enumerate([-0.04, 0.04], start=1):
        uid = f"bar_{index}"
        _write_minimal_glb(tmp_path / f"mesh_assets/bar/{uid}.glb", bar_vertices)
        scene_objects.append(
            action_agent_config_generation._SceneObject(
                source_uid=uid,
                source_role="rigid_object",
                config=_mesh_object(
                    uid,
                    f"mesh_assets/bar/{uid}.glb",
                    [0.0, y_value, 0.76],
                    [0.0, 0.0, 0.0],
                ),
            )
        )

    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]
    spec = _apply_arrangement_task_response(
        response={
            "objects": ["bar_1", "bar_2"],
            "order_by": "explicit",
            "order_direction": "given",
            "anchor": "table_center",
            "line_axis": "world_x",
            "task_prompt_summary": "Arrange two elongated bars in a row.",
        },
        table_source_uid="table",
        scene_objects=scene_objects,
        rigid_objects=rigid_objects,
        scene_dir=tmp_path,
        task_description="arrange two elongated bars",
    )

    assert [step.orientation_goal for step in spec.steps] == [
        "axis_align",
        "axis_align",
    ]
    assert [step.orientation_axis for step in spec.steps] == ["x", "x"]
    assert all(
        step.high_position[2] - step.release_position[2] == pytest.approx(0.15)
        for step in spec.steps
    )


def test_demo153_like_cans_keep_row_near_table_center_for_reachability(
    tmp_path: Path,
) -> None:
    scene_objects = _write_demo153_like_can_arrangement_scene(tmp_path)
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]

    spec = _apply_arrangement_task_response(
        response={
            "objects": ["can_1", "can_2", "can_3", "can_4"],
            "order_by": "explicit",
            "order_direction": "given",
            "anchor": "table_center",
            "line_axis": "table_long_axis",
            "task_prompt_summary": "Arrange four cans in a straight row.",
        },
        table_source_uid="table",
        scene_objects=scene_objects,
        rigid_objects=rigid_objects,
        scene_dir=tmp_path,
        task_description="用机械臂将桌面上的所有罐头从左往右摆成一排",
    )

    assert spec.line_origin_xy == pytest.approx([0.0, 0.0])
    assert [step.orientation_goal for step in spec.steps] == [
        "preserve",
        "preserve",
        "preserve",
        "preserve",
    ]
    assert [step.orientation_axis for step in spec.steps] == [
        "none",
        "none",
        "none",
        "none",
    ]
    assert len({round(step.target_xy[0], 6) for step in spec.steps}) == 1
    assert spec.steps[0].target_xy[0] == pytest.approx(0.0)
    for step in spec.steps:
        assert step.high_position[2] - step.release_position[2] == pytest.approx(0.10)


def test_arrangement_static_unmoved_object_blocks_line_layout(
    tmp_path: Path,
) -> None:
    scene_objects = _write_narrow_four_can_arrangement_scene(
        tmp_path,
        include_static_blocker=True,
    )
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]

    with pytest.raises(ValueError, match="collision-free one-line arrangement"):
        _apply_arrangement_task_response(
            response={
                "objects": ["can_1", "can_2", "can_3", "can_4"],
                "order_by": "explicit",
                "order_direction": "given",
                "anchor": "table_center",
                "line_axis": "table_long_axis",
                "task_prompt_summary": "Arrange four cans in a straight row.",
            },
            table_source_uid="table",
            scene_objects=scene_objects,
            rigid_objects=rigid_objects,
            scene_dir=tmp_path,
            task_description="将桌面上的罐头摆成一排",
        )


def test_task_router_routes_chinese_row_task_to_arrangement_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_arrangement_project(project_dir)
    router_calls = []

    def fake_call_task_router_llm(**kwargs):
        router_calls.append(kwargs)
        assert kwargs["task_description"] == "将三个方块摆成一排"
        return {
            "route": "arrangement_line",
            "confidence": 0.96,
            "reason": "The task asks for a global row arrangement.",
            "candidate_objects": ["cube_1", "cube_2", "cube_3"],
        }

    def fail_relative_task_llm(**kwargs):
        raise AssertionError("row arrangement task must not use relative routing")

    def fake_call_arrangement_task_llm(**kwargs):
        assert kwargs["task_description"] == "将三个方块摆成一排"
        return {
            "objects": ["cube_1", "cube_2", "cube_3"],
            "order_by": "explicit",
            "order_direction": "given",
            "anchor": "table_center",
            "line_axis": "left_to_right",
            "task_prompt_summary": "Arrange three cubes in one row.",
            "basic_background_notes": "All three cubes are line-arrangement targets.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_task_router_llm",
        fake_call_task_router_llm,
    )
    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        fail_relative_task_llm,
    )
    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_arrangement_task_llm",
        fake_call_arrangement_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_arrangement_agent",
        task_name="BlocksRow",
        task_description="将三个方块摆成一排",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    assert {obj["uid"] for obj in gym_config["rigid_object"]} == {
        "cube_1",
        "cube_2",
        "cube_3",
    }
    assert paths.summary["mode"] == "arrangement_line"
    assert paths.summary["task_route"] == {
        "route": "arrangement_line",
        "confidence": 0.96,
        "reason": "The task asks for a global row arrangement.",
        "candidate_objects": ["cube_1", "cube_2", "cube_3"],
        "warnings": [],
    }
    assert len(router_calls) == 1
    assert "Arrangement plan:" in paths.task_prompt.read_text(encoding="utf-8")


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
        "axis": "table_long_axis",
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
            },
            {
                "object": "cube_1",
                "source_uid": "cube_1",
                "slot_index": 1,
                "active_arm": "left_arm",
                "target_xy": paths.summary["placements"][1]["target_xy"],
            },
            {
                "object": "cube_3",
                "source_uid": "cube_3",
                "slot_index": 2,
                "active_arm": "left_arm",
                "target_xy": paths.summary["placements"][2]["target_xy"],
            },
        ],
    }
    target_x_values = [
        placement["target_xy"][0] for placement in paths.summary["placements"]
    ]
    target_y_values = [
        placement["target_xy"][1] for placement in paths.summary["placements"]
    ]
    assert target_x_values == sorted(target_x_values)
    assert len({round(value, 6) for value in target_y_values}) == 1
    assert paths.summary["spacing"] >= 0.07
    assert paths.summary["layout_clearance"] == pytest.approx(0.025)
    _assert_arrangement_slots_avoid_initial_objects(
        paths.summary,
        gym_config,
    )

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["op"] == "all"
    ordered_objects = [placement["object"] for placement in paths.summary["placements"]]
    assert {
        (term["type"], tuple(term["objects"]), term["axis"])
        for term in success["terms"]
        if term["type"] in {"objects_collinear", "objects_ordered"}
    } == {
        ("objects_collinear", tuple(ordered_objects), "x"),
        ("objects_ordered", tuple(ordered_objects), "x"),
    }
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
    assert task_prompt.count('"lift_height":0.30') == 3
    assert task_prompt.count('"atomic_action_class":"Place"') == 3
    assert task_prompt.count('"atomic_action_class":"MoveEndEffector"') == 3
    assert task_prompt.count('"reference":"absolute"') >= 6
    assert task_prompt.count('"orientation_goal":"axis_align"') == 0
    assert task_prompt.count('"orientation_axis":"x"') == 0
    assert task_prompt.count('"orientation_goal":"preserve"') == 6
    assert task_prompt.count('"target_pose":{"reference":"relative"') == 6
    assert task_prompt.count('"z_policy":"object_on_surface"') == 3
    assert task_prompt.count('"support":"table"') == 3
    assert task_prompt.count('"surface_clearance":0.015') == 3
    assert "Collision-aware line origin xy" in task_prompt
    assert atom_actions.count('"atomic_action_class":"PickUp"') == 3
    assert atom_actions.count('"lift_height":0.30') == 3
    assert atom_actions.count('"orientation_goal":"axis_align"') == 0
    assert atom_actions.count('"orientation_axis":"x"') == 0
    assert atom_actions.count('"orientation_goal":"preserve"') == 6
    assert atom_actions.count('"atomic_action_class":"Place"') == 3
    assert atom_actions.count('"atomic_action_class":"MoveEndEffector"') == 3
    assert atom_actions.count('"z_policy":"object_on_surface"') == 3


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
        placement["orientation_goal"] == "preserve"
        and placement["orientation_axis"] == "none"
        for placement in summary["placements"]
    )
    x_values = [placement["target_xy"][0] for placement in summary["placements"]]
    y_values = [placement["target_xy"][1] for placement in summary["placements"]]
    assert x_values == sorted(x_values)
    assert len({round(value, 6) for value in y_values}) == 1
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


def test_arrangement_uses_table_mesh_bounds_center_when_table_origin_is_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "offset_table_arrangement_gym_project"
    _write_offset_table_center_project(project_dir, count=3)

    def fake_call_arrangement_task_llm(**kwargs):
        return {
            "objects": ["cube_1", "cube_2", "cube_3"],
            "order_by": "explicit",
            "order_direction": "given",
            "anchor": "table_center",
            "line_axis": "left_to_right",
            "task_prompt_summary": "Arrange three cubes on the real table center.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_arrangement_task_llm",
        fake_call_arrangement_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_offset_table_arrangement_agent",
        task_description="把三个方块摆成一排",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    summary = paths.summary
    assert summary["mode"] == "arrangement_line"
    assert summary["line_origin_xy"][1] == pytest.approx(0.24)
    assert all(
        0.0 < placement["target_xy"][1] < 0.48 for placement in summary["placements"]
    )
    _assert_arrangement_slots_avoid_initial_objects(summary, gym_config)


def test_arrangement_recomputes_targets_after_scene_rotation_and_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "rotated_offset_table_arrangement_gym_project"
    _write_offset_table_center_project(project_dir, count=3)

    def fake_call_arrangement_task_llm(**kwargs):
        return {
            "objects": ["cube_1", "cube_2", "cube_3"],
            "order_by": "explicit",
            "order_direction": "given",
            "anchor": "table_center",
            "line_axis": "left_to_right",
            "task_prompt_summary": "Arrange three cubes after scene conversion.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_arrangement_task_llm",
        fake_call_arrangement_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_rotated_arrangement_agent",
        task_description="把三个方块摆成一排",
        target_body_scale=0.8,
        source_scene_body_scale_mode="multiply",
        preserve_source_scene_geometry=True,
        source_scene_z_rotation_degrees=-90.0,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    table_config = next(
        obj for obj in gym_config["background"] if obj["uid"] == "table"
    )
    table_center = _mesh_config_world_xy_center(table_config)
    table_bounds = _mesh_config_world_xy_bounds(table_config)
    assert table_center == pytest.approx([0.192, 0.0])
    assert paths.summary["axis"] == "table_long_axis"
    assert paths.summary["line_origin_xy"] == pytest.approx(table_center)
    assert all(
        placement["orientation_goal"] == "preserve"
        and placement["orientation_axis"] == "none"
        for placement in paths.summary["placements"]
    )
    assert table_bounds is not None
    table_min, table_max = table_bounds
    for placement in paths.summary["placements"]:
        target_xy = placement["target_xy"]
        assert table_min[0] < target_xy[0] < table_max[0]
        assert table_min[1] < target_xy[1] < table_max[1]

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert "resolved to world `y`" in task_prompt
    release_positions = _arrangement_release_positions_from_prompt(task_prompt)
    assert len(release_positions) == len(paths.summary["placements"])
    rigid_by_uid = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    table_top_z = _mesh_config_world_z_bounds(table_config)[1]
    for placement, release_position in zip(
        paths.summary["placements"], release_positions
    ):
        assert release_position[:2] == pytest.approx(placement["target_xy"])
        object_config = rigid_by_uid[placement["object"]]
        object_bottom_offset = _mesh_config_local_zmin_after_rotation(object_config)
        expected_z = table_top_z + _TABLETOP_OBJECT_CLEARANCE - object_bottom_offset
        assert release_position[2] == pytest.approx(expected_z)


def test_task_description_generates_three_block_stacking_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "44_Stack Blocks Three_gym_project"
    _write_stacking_blocks_project(project_dir, count=3)

    def fake_call_stacking_task_llm(**kwargs):
        return {
            "objects": ["red_cube_1", "green_cube_2", "blue_cube_3"],
            "stack_mode": "on_top",
            "bottom_to_top": ["red_cube_1", "green_cube_2", "blue_cube_3"],
            "order_by": "explicit",
            "anchor": "table_center",
            "object_attributes": {
                "red_cube_1": {"color": "red"},
                "green_cube_2": {"color": "green"},
                "blue_cube_3": {"color": "blue"},
            },
            "task_prompt_summary": (
                "Move the cubes to the table center and stack blue on green "
                "and green on red."
            ),
            "basic_background_notes": "Table with red, green, and blue cubes.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_stacking_task_llm",
        fake_call_stacking_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_three_block_stacking_agent",
        task_name="Demo44",
        task_description=(
            "桌上有红、绿、蓝三个方块，将它们移动到桌子中央，并把蓝色方块叠到绿色方块上、绿色方块叠到红色方块上。"
        ),
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    summary = paths.summary
    assert summary["mode"] == "stacking"
    assert summary["stack_mode"] == "on_top"
    assert summary["bottom_to_top"] == ["red_cube", "green_cube", "blue_cube"]
    assert [placement["support"] for placement in summary["placements"]] == [
        None,
        "red_cube",
        "green_cube",
    ]
    assert all(
        placement["orientation_goal"] == "preserve"
        and placement["orientation_axis"] == "none"
        for placement in summary["placements"]
    )
    target_xy = [
        placement["target_position"][:2] for placement in summary["placements"]
    ]
    assert target_xy == [summary["anchor_xy"]] * 3

    success = gym_config["env"]["extensions"]["agent_success"]
    object_on_terms = [
        term for term in success["terms"] if term["type"] == "object_on_object"
    ]
    assert {(term["object"], term["support"]) for term in object_on_terms} == {
        ("green_cube", "red_cube"),
        ("blue_cube", "green_cube"),
    }
    assert any(
        term["type"] == "object_xy_near" and term["object"] == "red_cube"
        for term in success["terms"]
    )

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert "Generate one deterministic nominal graph with exactly 18 nominal edges" in (
        task_prompt
    )
    assert "Pick up both" not in task_prompt
    assert task_prompt.count('"atomic_action_class":"PickUp"') == 3
    assert task_prompt.count('"atomic_action_class":"MoveEndEffector"') == 3
    assert task_prompt.count('"atomic_action_class":"Place"') == 3
    assert atom_actions.count('"orientation_goal":"axis_align"') == 0
    assert atom_actions.count('"orientation_goal":"preserve"') == 6


def test_stacking_uses_table_mesh_bounds_center_when_table_origin_is_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "offset_table_stacking_gym_project"
    _write_offset_table_center_project(project_dir, count=3)

    def fake_call_stacking_task_llm(**kwargs):
        return {
            "objects": ["cube_1", "cube_2", "cube_3"],
            "stack_mode": "on_top",
            "bottom_to_top": ["cube_1", "cube_2", "cube_3"],
            "order_by": "explicit",
            "anchor": "table_center",
            "task_prompt_summary": "Stack three cubes at the real table center.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_stacking_task_llm",
        fake_call_stacking_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_offset_table_stacking_agent",
        task_description="把三个方块叠放到桌面中央",
        prewarm_coacd_cache=False,
    )

    summary = paths.summary
    assert summary["mode"] == "stacking"
    assert summary["anchor_xy"] == pytest.approx([0.0, 0.24])
    assert [
        placement["target_position"][:2] for placement in summary["placements"]
    ] == [summary["anchor_xy"]] * 3
    task_graph = json.loads(paths.task_graph.read_text(encoding="utf-8"))
    assert task_graph["goal"] == "v18_done"
    assert task_graph["nodes"][-1]["id"] == "v18_done"
    assert len(task_graph["edges"]) == 18
    assert len(task_graph["nodes"]) == 19

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    success = gym_config["env"]["extensions"]["agent_success"]
    bottom_xy_terms = [
        term
        for term in success["terms"]
        if term["type"] == "object_xy_near" and term["object"] == "cube_1"
    ]
    assert len(bottom_xy_terms) == 1
    assert bottom_xy_terms[0]["target_xy"] == pytest.approx(summary["anchor_xy"])


def test_stacking_recomputes_anchor_after_scene_rotation_and_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "rotated_offset_table_stacking_gym_project"
    _write_offset_table_center_project(project_dir, count=3)

    def fake_call_stacking_task_llm(**kwargs):
        return {
            "objects": ["cube_1", "cube_2", "cube_3"],
            "stack_mode": "on_top",
            "bottom_to_top": ["cube_1", "cube_2", "cube_3"],
            "order_by": "explicit",
            "anchor": "table_center",
            "task_prompt_summary": "Stack three cubes after scene conversion.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_stacking_task_llm",
        fake_call_stacking_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_rotated_stacking_agent",
        task_description="把三个方块叠放到桌面中央",
        target_body_scale=0.8,
        source_scene_body_scale_mode="multiply",
        preserve_source_scene_geometry=True,
        source_scene_z_rotation_degrees=-90.0,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    table_config = next(
        obj for obj in gym_config["background"] if obj["uid"] == "table"
    )
    table_center = _mesh_config_world_xy_center(table_config)
    assert table_center == pytest.approx([0.192, 0.0])
    assert paths.summary["anchor_xy"] == pytest.approx(table_center)
    assert [
        placement["target_position"][:2] for placement in paths.summary["placements"]
    ] == [paths.summary["anchor_xy"]] * 3

    rigid_by_uid = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    table_top_z = _mesh_config_world_z_bounds(table_config)[1]
    bottom_placement = paths.summary["placements"][0]
    bottom_config = rigid_by_uid[bottom_placement["object"]]
    bottom_offset = _mesh_config_local_zmin_after_rotation(bottom_config)
    expected_bottom_z = table_top_z + _TABLETOP_OBJECT_CLEARANCE - bottom_offset
    assert bottom_placement["target_position"][2] == pytest.approx(expected_bottom_z)

    success = gym_config["env"]["extensions"]["agent_success"]
    bottom_xy_terms = [
        term
        for term in success["terms"]
        if term["type"] == "object_xy_near"
        and term["object"] == bottom_placement["object"]
    ]
    assert len(bottom_xy_terms) == 1
    assert bottom_xy_terms[0]["target_xy"] == pytest.approx(paths.summary["anchor_xy"])


def test_task_description_generates_two_block_stacking_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "45_Stack Blocks Two_gym_project"
    _write_stacking_blocks_project(project_dir, count=2)

    def fake_call_stacking_task_llm(**kwargs):
        return {
            "objects": ["red_cube_1", "green_cube_2"],
            "stack_mode": "on_top",
            "bottom_to_top": ["red_cube_1", "green_cube_2"],
            "order_by": "explicit",
            "anchor": "table_center",
            "task_prompt_summary": "Stack green on red at the table center.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_stacking_task_llm",
        fake_call_stacking_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_two_block_stacking_agent",
        task_name="Demo45",
        task_description="桌上有红、绿两个方块，将它们移动到桌子中央，并把绿色方块叠到红色方块上。",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    summary = paths.summary
    assert summary["mode"] == "stacking"
    assert summary["bottom_to_top"] == ["red_cube", "green_cube"]
    success_terms = gym_config["env"]["extensions"]["agent_success"]["terms"]
    assert any(
        term.get("type") == "object_on_object"
        and term.get("object") == "green_cube"
        and term.get("support") == "red_cube"
        for term in success_terms
    )
    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert "exactly 12 nominal edges" in task_prompt
    assert "Pick up both" not in task_prompt


def test_task_description_generates_nested_bowl_stacking_by_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "46_Stack Bowls Three_gym_project"
    _write_stacking_bowls_project(project_dir, count=3)

    def fake_call_stacking_task_llm(**kwargs):
        return {
            "objects": ["interact_bowl_1", "interact_bowl_2", "interact_bowl_3"],
            "stack_mode": "nested",
            "bottom_to_top": [],
            "order_by": "size",
            "anchor": "table_center",
            "task_prompt_summary": "Nest the three bowls at the table center.",
        }

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_stacking_task_llm",
        fake_call_stacking_task_llm,
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_three_bowl_stacking_agent",
        task_name="Demo46",
        task_description="将三个碗相互叠放。",
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    summary = paths.summary
    assert summary["mode"] == "stacking"
    assert summary["stack_mode"] == "nested"
    assert summary["order_by"] == "size"
    assert summary["bottom_to_top"] == [
        "interact_bowl_1",
        "interact_bowl_2",
        "interact_bowl_3",
    ]
    assert all(
        placement["orientation_goal"] == "preserve"
        and placement["orientation_axis"] == "none"
        for placement in summary["placements"]
    )
    success_terms = gym_config["env"]["extensions"]["agent_success"]["terms"]
    in_container_terms = [
        term for term in success_terms if term["type"] == "object_in_container"
    ]
    assert {(term["object"], term["container"]) for term in in_container_terms} == {
        ("interact_bowl_2", "interact_bowl_1"),
        ("interact_bowl_3", "interact_bowl_2"),
    }
    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    assert "Stack mode: `nested`" in task_prompt
    assert "exactly 18 nominal edges" in task_prompt
    assert "High staging orientation" not in atom_actions
    assert "Align `interact_bowl" not in task_prompt


def test_stacking_keyword_routes_before_arrangement() -> None:
    assert _is_stacking_task_description("将红、绿、蓝三个方块叠成一列")
    assert _is_stacking_task_description("Stack the bowls at the table center")


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
        "mode": "dual_arm_object_manipulation",
        "manipulations": [
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
            '"atomic_action_class":"MoveHeldObject","robot_name":"left_arm",'
            '"control":"arm","target_object_pose":{"reference":"object",'
            '"obj_name":"wicker_basket","offset":[-0.04,0.0,0.12],'
            '"orientation_goal":"preserve","orientation_axis":"none"}' in text
        )
        assert (
            '"atomic_action_class":"MoveHeldObject","robot_name":"right_arm",'
            '"control":"arm","target_object_pose":{"reference":"object",'
            '"obj_name":"wicker_basket","offset":[0.04,0.0,0.12],'
            '"orientation_goal":"preserve","orientation_axis":"none"}' in text
        )
        assert (
            '"atomic_action_class":"Place","robot_name":"left_arm",'
            '"control":"arm","target_pose":{"reference":"relative",'
            '"offset":[0.0,0.0,0.0],"frame":"world"}' in text
        )
        assert (
            '"atomic_action_class":"Place","robot_name":"right_arm",'
            '"control":"arm","target_pose":{"reference":"relative",'
            '"offset":[0.0,0.0,0.0],"frame":"world"}' in text
        )
    assert "container XY long axis" in task_prompt


def test_dual_inside_y_axis_slots_keep_each_arm_on_its_container_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_y_axis_container_project(project_dir)

    def fake_call_relative_task_llm(**kwargs):
        return {
            "placements": [
                {
                    "moved_object": "apple_left",
                    "reference_object": "basket_3",
                    "goal_relation": "inside",
                    "arm": "left",
                },
                {
                    "moved_object": "apple_right",
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

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_y_axis_dual_inside_agent",
        task_description="左臂把左边苹果放进篮子，右臂把右边苹果放进篮子",
        inside_container_slot_distance_scale=0.5,
        prewarm_coacd_cache=False,
    )

    assert _stable_summary(paths.summary) == {
        "mode": "dual_arm_object_manipulation",
        "manipulations": [
            {
                "moved_object": "apple_left",
                "reference_object": "wicker_basket",
                "relation": "inside",
                "active_arm": "left_arm",
                "release_offset": [0.0, 0.02, 0.12],
            },
            {
                "moved_object": "apple_right",
                "reference_object": "wicker_basket",
                "relation": "inside",
                "active_arm": "right_arm",
                "release_offset": [0.0, -0.02, 0.12],
            },
        ],
    }

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    atom_actions = paths.atom_actions.read_text(encoding="utf-8")
    for text in (task_prompt, atom_actions):
        assert (
            '"robot_name":"left_arm","control":"arm","target_object_pose":'
            '{"reference":"object","obj_name":"wicker_basket",'
            '"offset":[0.0,0.02,0.12],'
            '"orientation_goal":"preserve","orientation_axis":"none"}' in text
        )
        assert (
            '"robot_name":"right_arm","control":"arm","target_object_pose":'
            '{"reference":"object","obj_name":"wicker_basket",'
            '"offset":[0.0,-0.02,0.12],'
            '"orientation_goal":"preserve","orientation_axis":"none"}' in text
        )
        assert (
            '"target_pose":{"reference":"relative","offset":[0.0,0.0,0.0],'
            '"frame":"world"}' in text
        )


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

    active_arms = [
        placement["active_arm"] for placement in paths.summary["manipulations"]
    ]
    assert active_arms == ["right_arm", "left_arm"]

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    assert "agent_grasp_pose_overrides" not in gym_config["env"]["extensions"]


def test_task_description_dual_auto_reassigns_after_scene_rotation(
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
        tmp_path / "generated_dual_auto_rotated_relative_agent",
        task_description="双臂分别移动两个苹果",
        source_scene_z_rotation_degrees=-90.0,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    manipulations = paths.summary["manipulations"]
    active_arms = [placement["active_arm"] for placement in manipulations]
    expected_arms = []
    for placement in manipulations:
        moved_object = rigid_objects[placement["moved_object"]]
        expected_arms.append(f"{_arm_side_for_position(moved_object['init_pos'])}_arm")

    assert active_arms == expected_arms
    assert active_arms == ["left_arm", "right_arm"]


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
    assert set(rigid_objects) == {"apple_1", "apple_2", "wicker_basket"}
    assert set(background_objects) == {"table"}
    assert rigid_objects["apple_2"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["apple_2"]["body_type"] == "dynamic"
    assert rigid_objects["apple_1"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["apple_1"]["body_type"] == "dynamic"
    assert rigid_objects["wicker_basket"]["body_scale"] == [1.0, 1.0, 1.0]
    assert rigid_objects["wicker_basket"]["body_type"] == "dynamic"
    assert _body_scaled_meshes_by_uid(paths.summary)["apple_2"]["body_scale"] == [
        0.6,
        0.6,
        0.6,
    ]

    success = gym_config["env"]["extensions"]["agent_success"]
    assert success["type"] == "object_on_object"
    assert success["object"] == "apple_2"
    assert success["support"] == "apple_1"

    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert "on top of `apple_1`" in task_prompt


def test_single_moved_object_on_support_is_not_generated_as_stacking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "1790000000_gym_project"
    _write_project(project_dir)

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_task_router_llm",
        lambda **_: {
            "route": "stacking",
            "confidence": 0.9,
            "reason": "The final state is vertical contact.",
            "candidate_objects": ["apple_2", "apple_1"],
        },
    )

    def fail_if_stacking_is_used(**kwargs):
        raise AssertionError("Single-object on-support tasks must not use stacking.")

    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_stacking_task_llm",
        fail_if_stacking_is_used,
    )
    monkeypatch.setattr(
        action_agent_config_generation,
        "_call_relative_task_llm",
        lambda **_: {
            "moved_object": "apple_2",
            "reference_object": "apple_1",
            "goal_relation": "on",
            "arm": "left",
            "task_prompt_summary": "Use the left arm to place apple_2 on apple_1.",
        },
    )

    paths = generate_action_agent_config_from_project(
        project_dir,
        tmp_path / "generated_single_object_on_support_agent",
        task_description="用左臂把 apple_2 放到 apple_1 上方",
        target_body_scale=0.6,
        prewarm_coacd_cache=False,
    )

    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    rigid_objects = {obj["uid"]: obj for obj in gym_config["rigid_object"]}
    success = gym_config["env"]["extensions"]["agent_success"]

    assert paths.summary["mode"] == "object_manipulation"
    assert paths.summary["active_arm"] == "left_arm"
    assert set(rigid_objects) == {"apple_1", "apple_2", "wicker_basket"}
    assert success["type"] == "object_on_object"
    assert success["object"] == "apple_2"
    assert success["support"] == "apple_1"


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
    direct_lights = gym_config["light"]["direct"]
    assert direct_lights == [
        {
            "uid": "main_light",
            "light_type": "point",
            "color": [1.0, 1.0, 1.0],
            "intensity": 40.0,
            "init_pos": [0.0, -0.4, 2.2],
            "radius": 10.0,
        }
    ]


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


def test_objects_collinear_success_predicate_accepts_straight_row() -> None:
    env = _FakeEnv(
        {
            "can_1": [-0.10, 0.002, 0.10],
            "can_2": [0.00, -0.001, 0.10],
            "can_3": [0.10, 0.003, 0.10],
        }
    )

    success = evaluate_configured_success(
        env,
        {
            "type": "objects_collinear",
            "objects": ["can_1", "can_2", "can_3"],
            "axis": "x",
            "tolerance": 0.01,
        },
    )

    assert bool(success.item()) is True


def test_objects_collinear_success_predicate_rejects_bent_row() -> None:
    env = _FakeEnv(
        {
            "can_1": [-0.10, 0.00, 0.10],
            "can_2": [0.00, 0.04, 0.10],
            "can_3": [0.10, 0.00, 0.10],
        }
    )

    success = evaluate_configured_success(
        env,
        {
            "type": "objects_collinear",
            "objects": ["can_1", "can_2", "can_3"],
            "axis": "x",
            "tolerance": 0.01,
        },
    )

    assert bool(success.item()) is False


def test_objects_ordered_success_predicate_accepts_monotonic_row() -> None:
    env = _FakeEnv(
        {
            "can_1": [-0.10, 0.00, 0.10],
            "can_2": [0.00, 0.00, 0.10],
            "can_3": [0.10, 0.00, 0.10],
        }
    )

    success = evaluate_configured_success(
        env,
        {
            "type": "objects_ordered",
            "objects": ["can_1", "can_2", "can_3"],
            "axis": "x",
            "direction": "ascending",
            "tolerance": 0.01,
        },
    )

    assert bool(success.item()) is True


def test_objects_ordered_success_predicate_rejects_inverted_row() -> None:
    env = _FakeEnv(
        {
            "can_1": [-0.10, 0.00, 0.10],
            "can_2": [0.10, 0.00, 0.10],
            "can_3": [0.00, 0.00, 0.10],
        }
    )

    success = evaluate_configured_success(
        env,
        {
            "type": "objects_ordered",
            "objects": ["can_1", "can_2", "can_3"],
            "axis": "x",
            "direction": "ascending",
            "tolerance": 0.01,
        },
    )

    assert bool(success.item()) is False


def test_success_uid_validation_checks_object_lists() -> None:
    _validate_success_uids(
        {
            "type": "objects_collinear",
            "objects": ["can_1", "can_2"],
            "axis": "x",
        },
        rigid_uids={"can_1", "can_2"},
        scene_uids={"table", "can_1", "can_2"},
    )

    with pytest.raises(ValueError, match="missing_can"):
        _validate_success_uids(
            {
                "type": "objects_ordered",
                "objects": ["can_1", "missing_can"],
                "axis": "x",
            },
            rigid_uids={"can_1", "can_2"},
            scene_uids={"table", "can_1", "can_2"},
        )


def test_object_held_by_gripper_success_predicate() -> None:
    env = _FakeEnv(
        {
            "bottle": [0.0, 0.18, 0.24],
        }
    )
    env.left_eef_pose = _pose_at([0.0, 0.18, 0.25])
    env.left_gripper_state = torch.tensor([0.04], dtype=torch.float32)
    env.close_state = torch.tensor([0.04], dtype=torch.float32)

    success = evaluate_configured_success(
        env,
        {
            "type": "object_held_by_gripper",
            "object": "bottle",
            "arm": "left_arm",
            "max_distance": 0.12,
        },
    )

    assert bool(success.item()) is True


def test_object_held_by_gripper_returns_false_before_agent_state_init() -> None:
    env = _FakeEnvWithoutAgentState(
        {
            "bottle": [0.0, 0.18, 0.24],
        }
    )

    success = evaluate_configured_success(
        env,
        {
            "type": "object_held_by_gripper",
            "object": "bottle",
            "arm": "left_arm",
            "max_distance": 0.12,
        },
    )

    assert bool(success.item()) is False


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


def _write_y_axis_container_project(project_dir: Path) -> None:
    mesh_specs = {
        "mesh_assets/table/table_0.glb": _default_mesh_vertices(),
        "mesh_assets/basket/basket_3/basket_3.glb": [
            (-0.03, -0.08, 0.0),
            (0.03, -0.08, 0.0),
            (0.0, 0.08, 0.0),
        ],
        "mesh_assets/apple/apple_left/apple_left.glb": _default_mesh_vertices(),
        "mesh_assets/apple/apple_right/apple_right.glb": _default_mesh_vertices(),
    }
    for rel_path, vertices in mesh_specs.items():
        _write_minimal_glb(project_dir / rel_path, vertices)

    gym_config = {
        "id": "Image2Tabletop-y-axis-container-v0",
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
                "basket_3",
                "mesh_assets/basket/basket_3/basket_3.glb",
                [0.0, 0.0, 0.75],
                [0.0, 0.0, 0.0],
            ),
            _mesh_object(
                "apple_left",
                "mesh_assets/apple/apple_left/apple_left.glb",
                [0.0, 0.24, 0.76],
                [0.0, 0.0, 0.0],
            ),
            _mesh_object(
                "apple_right",
                "mesh_assets/apple/apple_right/apple_right.glb",
                [0.0, -0.24, 0.76],
                [0.0, 0.0, 0.0],
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


def _write_narrow_four_can_arrangement_scene(
    scene_dir: Path,
    *,
    include_static_blocker: bool = False,
) -> list:
    _write_minimal_glb(
        scene_dir / "mesh_assets/table/table_0.glb",
        [(-0.18, -0.08, 0.0), (0.18, -0.08, 0.0), (0.0, 0.08, 0.0)],
    )
    can_vertices = [(-0.02, -0.02, 0.0), (0.02, -0.02, 0.0), (0.0, 0.02, 0.08)]
    can_slot_x = [-0.105, -0.035, 0.035, 0.105]
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
        )
    ]
    for index, x_value in enumerate(can_slot_x, start=1):
        uid = f"can_{index}"
        _write_minimal_glb(scene_dir / f"mesh_assets/can/{uid}.glb", can_vertices)
        scene_objects.append(
            action_agent_config_generation._SceneObject(
                source_uid=uid,
                source_role="rigid_object",
                config=_mesh_object(
                    uid,
                    f"mesh_assets/can/{uid}.glb",
                    [x_value, 0.0, 0.76],
                    [0.0, 0.0, 0.0],
                ),
            )
        )
    if include_static_blocker:
        _write_minimal_glb(
            scene_dir / "mesh_assets/blocker/blocker.glb",
            can_vertices,
        )
        scene_objects.append(
            action_agent_config_generation._SceneObject(
                source_uid="blocker",
                source_role="rigid_object",
                config=_mesh_object(
                    "blocker",
                    "mesh_assets/blocker/blocker.glb",
                    [0.0, 0.0, 0.76],
                    [0.0, 0.0, 0.0],
                ),
            )
        )
    return scene_objects


def _write_demo153_like_can_arrangement_scene(scene_dir: Path) -> list:
    _write_minimal_glb(
        scene_dir / "mesh_assets/table/table_0.glb",
        [(-0.36, -0.59, 0.0), (0.36, -0.59, 0.0), (0.0, 0.59, 0.0)],
    )
    can_vertices = [(-0.032, -0.032, 0.0), (0.032, -0.032, 0.0), (0.0, 0.032, 0.16)]
    init_xy = [
        (0.07173, 0.276622),
        (-0.093168, 0.132612),
        (0.091263, -0.086257),
        (-0.024065, -0.277453),
    ]
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
        )
    ]
    for index, (x_value, y_value) in enumerate(init_xy, start=1):
        uid = f"can_{index}"
        _write_minimal_glb(scene_dir / f"mesh_assets/can/{uid}.glb", can_vertices)
        scene_objects.append(
            action_agent_config_generation._SceneObject(
                source_uid=uid,
                source_role="rigid_object",
                config=_mesh_object(
                    uid,
                    f"mesh_assets/can/{uid}.glb",
                    [x_value, y_value, 0.76],
                    [0.0, 0.0, 0.0],
                ),
            )
        )
    return scene_objects


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


def _write_offset_table_center_project(project_dir: Path, *, count: int) -> None:
    _write_minimal_glb(
        project_dir / "mesh_assets/table/table_0.glb",
        [(-0.45, 0.0, 0.0), (0.45, 0.0, 0.0), (0.0, 0.48, 0.0)],
    )
    rigid_objects = []
    for index in range(count):
        uid = f"cube_{index + 1}"
        _write_minimal_glb(
            project_dir / f"mesh_assets/cube/{uid}/{uid}.glb",
            [(-0.02, -0.02, 0.0), (0.02, -0.02, 0.0), (0.0, 0.02, 0.04)],
        )
        rigid_objects.append(
            _mesh_object(
                uid,
                f"mesh_assets/cube/{uid}/{uid}.glb",
                [round(-0.12 + index * 0.12, 6), 0.06, 0.76],
                [0.0, 0.0, 0.0],
            )
        )

    gym_config = {
        "id": "Image2Tabletop-offset-table-center-v0",
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


def _write_stacking_blocks_project(project_dir: Path, *, count: int) -> None:
    _write_minimal_glb(
        project_dir / "mesh_assets/table/table_0.glb",
        [(-0.60, -0.40, 0.0), (0.60, -0.40, 0.0), (0.0, 0.40, 0.0)],
    )
    cube_specs = [
        ("red_cube_1", "cube_1", "red", [-0.08, -0.10, 0.76]),
        ("green_cube_2", "cube_2", "green", [0.05, -0.04, 0.76]),
        ("blue_cube_3", "cube_3", "blue", [-0.04, 0.12, 0.76]),
    ][:count]
    rigid_objects = []
    for index, (uid, mesh_uid, _color, init_pos) in enumerate(cube_specs, start=1):
        size = 0.035 + index * 0.005
        _write_minimal_glb(
            project_dir / f"mesh_assets/cube/{mesh_uid}/{mesh_uid}.glb",
            [
                (-size / 2.0, -size / 2.0, 0.0),
                (size / 2.0, -size / 2.0, 0.0),
                (0.0, size / 2.0, size),
            ],
        )
        rigid_objects.append(
            _mesh_object(
                uid,
                f"mesh_assets/cube/{mesh_uid}/{mesh_uid}.glb",
                init_pos,
                [0.0, 0.0, 0.0],
            )
        )

    gym_config = {
        "id": "Image2Tabletop-stacking-blocks-v0",
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


def _write_stacking_bowls_project(project_dir: Path, *, count: int) -> None:
    _write_minimal_glb(
        project_dir / "mesh_assets/table/table_0.glb",
        [(-0.60, -0.40, 0.0), (0.60, -0.40, 0.0), (0.0, 0.40, 0.0)],
    )
    rigid_objects = []
    for index in range(1, count + 1):
        uid = f"interact_bowl_{index}"
        radius = 0.025 + (count - index + 1) * 0.008
        height = 0.025 + (count - index + 1) * 0.006
        _write_minimal_glb(
            project_dir / f"mesh_assets/bowl/bowl_{index}/bowl_{index}.glb",
            [
                (-radius, -radius, 0.0),
                (radius, -radius, 0.0),
                (0.0, radius, height),
            ],
        )
        rigid_objects.append(
            _mesh_object(
                uid,
                f"mesh_assets/bowl/bowl_{index}/bowl_{index}.glb",
                [round(0.08 * index, 6), round(-0.04 * index, 6), 0.76],
                [0.0, 0.0, 0.0],
            )
        )

    gym_config = {
        "id": "Image2Tabletop-stacking-bowls-v0",
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


def _write_dual_upright_cup_can_project(project_dir: Path) -> None:
    table_vertices = [
        (-0.5, -0.4, 0.0),
        (0.5, -0.4, 0.0),
        (0.0, 0.4, 0.0),
        (-0.5, -0.4, 0.36),
        (0.5, -0.4, 0.36),
        (0.0, 0.4, 0.36),
    ]
    uprightable_vertices = [
        (-0.025, -0.025, -0.04),
        (0.025, -0.025, -0.04),
        (0.0, 0.025, 0.06),
    ]
    _write_minimal_glb(project_dir / "mesh_assets/table/table_0.glb", table_vertices)
    _write_minimal_glb(
        project_dir / "mesh_assets/paper_cup/paper_cup_1/paper_cup_1.glb",
        uprightable_vertices,
    )
    _write_minimal_glb(
        project_dir / "mesh_assets/soda_can/soda_can_1/soda_can_1.glb",
        uprightable_vertices,
    )

    gym_config = {
        "id": "Image2Tabletop-upright-cup-can-v0",
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
                "paper_cup_1",
                "mesh_assets/paper_cup/paper_cup_1/paper_cup_1.glb",
                [0.06, 0.10, 0.36],
                [0.0, 0.0, 90.0],
            ),
            _mesh_object(
                "soda_can_1",
                "mesh_assets/soda_can/soda_can_1/soda_can_1.glb",
                [0.05, -0.10, 0.36],
                [0.0, 0.0, -90.0],
            ),
        ],
    }
    (project_dir / "gym_config.json").write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )


def _write_dual_bottles_project(project_dir: Path) -> None:
    for rel_path in (
        "mesh_assets/table/table_0.glb",
        "mesh_assets/soda_bottle/soda_bottle_1/soda_bottle_1.glb",
        "mesh_assets/soda_bottle/soda_bottle_2/soda_bottle_2.glb",
    ):
        _write_minimal_glb(project_dir / rel_path, _default_mesh_vertices())

    gym_config = {
        "id": "Image2Tabletop-dual-bottles-v0",
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
                "interact_soda_bottle_1",
                "mesh_assets/soda_bottle/soda_bottle_1/soda_bottle_1.glb",
                [0.0, 0.18, 0.76],
                [0.0, 0.0, -90.0],
            ),
            _mesh_object(
                "interact_soda_bottle_2",
                "mesh_assets/soda_bottle/soda_bottle_2/soda_bottle_2.glb",
                [0.0, -0.18, 0.76],
                [0.0, 0.0, -90.0],
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


def _assert_normalized_obj_path(fpath: str) -> None:
    path = Path(fpath)
    assert path.suffix == ".obj"
    assert "mesh_assets/normalized" in path.as_posix()
    assert MESH_FRAME_NORMALIZATION_POLICY_VERSION not in path.name
    assert len(path.name) <= 64
    assert path.is_file()
    assert (path.parent / "material.mtl").is_file()


def _assert_body_scaled_obj_path(fpath: str) -> None:
    path = Path(fpath)
    assert path.suffix == ".obj"
    assert "mesh_assets/body_scaled" in path.as_posix()
    assert BODY_SCALE_BAKE_POLICY_VERSION not in path.name
    assert path.is_file()
    assert (path.parent / "material.mtl").is_file()


def _body_scaled_meshes_by_uid(summary: dict) -> dict[str, dict]:
    entries = summary.get("body_scaled_meshes", [])
    return {entry["uid"]: entry for entry in entries}


def _stable_summary(summary: dict) -> dict:
    stable = {
        key: value
        for key, value in summary.items()
        if key
        not in {
            "normalized_meshes",
            "body_scaled_meshes",
            "coacd_cache",
            "convex_decomposition_method",
            "task_route",
        }
    }
    if stable.get("orientation_goal") == "preserve":
        stable.pop("orientation_goal", None)
    if stable.get("orientation_axis") == "none":
        stable.pop("orientation_axis", None)
    if stable.get("orientation_align_to") is None:
        stable.pop("orientation_align_to", None)
    for placement in [*stable.get("placements", []), *stable.get("manipulations", [])]:
        if placement.get("intent") == "place_relative":
            placement.pop("intent", None)
        if placement.get("hover_height") == 0.1:
            placement.pop("hover_height", None)
        if placement.get("orientation_goal") == "preserve":
            placement.pop("orientation_goal", None)
        if placement.get("orientation_axis") == "none":
            placement.pop("orientation_axis", None)
        if placement.get("orientation_align_to") is None:
            placement.pop("orientation_align_to", None)
    if stable.get("intent") == "place_relative":
        stable.pop("intent", None)
    if stable.get("hover_height") == 0.1:
        stable.pop("hover_height", None)
    return stable


def _assert_arrangement_slots_avoid_initial_objects(
    summary: dict,
    gym_config: dict,
) -> None:
    clearance = float(summary["layout_clearance"])
    spacing = float(summary["spacing"])
    half_extent = max(0.035, (spacing - clearance) / 2.0)
    moved_uids = {placement["object"] for placement in summary["placements"]}
    initial_bounds_by_uid = {
        obj["uid"]: _xy_bounds_around(obj["init_pos"][:2], half_extent)
        for obj in gym_config["rigid_object"]
        if obj["uid"] not in moved_uids
    }
    for placement in summary["placements"]:
        slot_bounds = _xy_bounds_around(placement["target_xy"], half_extent)
        assert all(
            not _xy_bounds_overlap_for_test(
                slot_bounds,
                init_bound,
                clearance=clearance,
            )
            for init_bound in initial_bounds_by_uid.values()
        )


def _arrangement_release_positions_from_prompt(prompt: str) -> list[list[float]]:
    positions = [
        json.loads(f"[{match.group(1)}]")
        for match in re.finditer(r'"position":\[(.*?)\]', prompt)
    ]
    return positions[2::3]


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


def _obj_header_json_value(obj_text: str, key: str):
    prefix = f"# {key}: "
    for line in obj_text.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise AssertionError(f"Missing OBJ header key: {key}")


def _flatten_matrix(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


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
        self.left_eef_pose = _pose_at([0.0, 0.0, 0.0])
        self.right_eef_pose = _pose_at([0.0, 0.0, 0.0])
        self.left_gripper_state = torch.tensor([0.0], dtype=torch.float32)
        self.right_gripper_state = torch.tensor([0.0], dtype=torch.float32)
        self.close_state = torch.tensor([0.04], dtype=torch.float32)

    def get_current_xpos_agent(self):
        return self.left_eef_pose, self.right_eef_pose

    def get_current_gripper_state_agent(self):
        return self.left_gripper_state, self.right_gripper_state


class _FakeEnvWithoutAgentState:
    num_envs = 1
    device = torch.device("cpu")

    def __init__(self, positions: dict[str, list[float]]) -> None:
        self.sim = _FakeSim(positions)

    def get_current_xpos_agent(self):
        return self.left_arm_current_xpos, self.right_arm_current_xpos

    def get_current_gripper_state_agent(self):
        return (
            self.left_arm_current_gripper_state,
            self.right_arm_current_gripper_state,
        )


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


def _pose_at(position: list[float]) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose[:, :3, 3] = torch.tensor(position, dtype=torch.float32).reshape(1, 3)
    return pose
