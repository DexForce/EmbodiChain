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

"""Focused tests for the independent Action Engine generation boundary."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from embodichain.gen_sim.action_engine.domain import (
    TASK_AGENT_SCHEMA,
)
from embodichain.gen_sim.action_engine.protocol import (
    SCENE_REQUIREMENTS_SCHEMA,
    TASK_SPEC_SCHEMA,
)
from embodichain.gen_sim.action_engine.cli import (
    generate_action_agent_config as cli_module,
)
from embodichain.gen_sim.action_engine.cli.generate_action_agent_config import (
    build_parser,
)
from embodichain.gen_sim.action_engine.compiler import compile_task_agent
from embodichain.gen_sim.action_engine.generation.artifacts import (
    artifact_paths,
    write_generation_artifacts,
)
from embodichain.gen_sim.action_engine.generation.config_builder import (
    build_agent_config,
    build_fast_gym_config,
)
from embodichain.gen_sim.action_engine.generation.generator import (
    _add_ab_camera_requirements,
    _task_spec_role_bindings,
    generate_action_engine_config,
)
from embodichain.gen_sim.action_engine.generation.source_scene import (
    prepare_scene,
    resolve_gym_config_path,
    resolve_source_scene,
)
from embodichain.gen_sim.action_engine.planning import plan_task


@pytest.fixture
def gym_export(tmp_path: Path) -> Path:
    export = tmp_path / "gym_export"
    assets = export / "mesh_assets"
    assets.mkdir(parents=True)
    (assets / "table.glb").write_bytes(b"not-a-real-glb")
    (assets / "can.glb").write_bytes(b"not-a-real-glb")
    state = export / "scene_state"
    state.mkdir()
    (state / "result.json").write_text("{}\n", encoding="utf-8")

    config = {
        "id": "Prompt2Scene-test-v0",
        "env": {"events": {}, "observations": {}, "dataset": {}},
        "robot": {},
        "sensor": [],
        "light": {},
        "background": [
            {
                "uid": "table_0",
                "description": "A white table.",
                "shape": {"shape_type": "Mesh", "fpath": "mesh_assets/table.glb"},
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
        "rigid_object": [
            {
                "uid": "interact_can_0",
                "description": "A red soda can.",
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": "mesh_assets/can.glb",
                    "acd_method": "coacd",
                    "max_convex_hull_num": 32,
                },
                "attrs": {"mass": 0.01},
                "init_pos": [1.0, 2.0, 0.7],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
                "max_convex_hull_num": 32,
            }
        ],
    }
    (export / "gym_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    return export


@pytest.fixture
def scene_export(tmp_path: Path) -> Path:
    export = tmp_path / "scene_export"
    assets = export / "mesh_assets"
    assets.mkdir(parents=True)
    (assets / "table.glb").write_bytes(b"not-a-real-glb")
    (assets / "bottle_001.glb").write_bytes(b"not-a-real-glb")
    (assets / "bottle_002.glb").write_bytes(b"not-a-real-glb")

    config = {
        "format": "embodichain.scene-export/v1",
        "scene_id": "scene-export-test",
        "background": [
            {
                "uid": "table",
                "description": "A white table.",
                "shape": {"shape_type": "Mesh", "fpath": "mesh_assets/table.glb"},
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
        "rigid_object": [
            {
                "uid": uid,
                "name": f"Bottle {index}",
                "description": f"Bottle instance {index}.",
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": f"mesh_assets/{uid}.glb",
                },
                "init_pos": [float(index), float(index + 1), 0.7],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
            for index, uid in enumerate(("bottle_001", "bottle_002"), start=1)
        ],
    }
    (export / "scene_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    return export


def _existing_v2_task_spec(task_id: str = "direct_task") -> dict[str, object]:
    return {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": task_id,
        "level": "L1",
        "instruction": "扶正这个红色易拉罐。",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "task_01",
                "task_type": "E2",
                "params": {"object_role": "object_01"},
                "depends_on": [],
                "role": "primary",
            }
        ],
        "success": {"type": "semantic_goal", "task_instance_id": "task_01"},
        "oracle": {},
        "metadata": {"role_bindings": {"object_01": "interact_can"}},
    }


def test_prepare_scene_normalizes_uid_paths_and_prompt2scene_transform(
    gym_export: Path,
) -> None:
    scene = prepare_scene(gym_export)

    assert scene.uid_map == {
        "table_0": "table",
        "interact_can_0": "interact_can",
    }
    assert scene.z_rotation_degrees == -90.0
    assert scene.rigid_objects[0]["init_pos"] == [2.0, -1.0, 0.7]
    assert scene.rigid_objects[0]["max_convex_hull_num"] == 16
    assert scene.rigid_objects[0]["acd_method"] == "vhacd"
    assert scene.rigid_objects[0]["shape"]["acd_method"] == "vhacd"
    assert scene.rigid_objects[0]["shape"]["max_convex_hull_num"] == 16
    mesh_path = Path(scene.rigid_objects[0]["shape"]["fpath"])
    assert mesh_path.is_absolute()
    assert mesh_path.is_file()
    assert scene.planner_objects[1]["source_uid"] == "interact_can_0"
    assert scene.planner_objects[1]["uid"] == "interact_can"


def test_prepare_scene_supports_scene_export_v1(scene_export: Path) -> None:
    scene = prepare_scene(scene_export.parent)

    assert scene.source_config_path == scene_export / "scene_config.json"
    assert scene.uid_map == {
        "table": "table",
        "bottle_001": "bottle_001",
        "bottle_002": "bottle_002",
    }
    assert scene.planner_objects[1]["name"] == "Bottle 1"
    assert scene.z_rotation_degrees == -90.0
    assert scene.rigid_objects[0]["init_pos"] == [2.0, -1.0, 0.7]
    assert all(
        Path(config["shape"]["fpath"]).is_file()
        for config in (*scene.background, *scene.rigid_objects)
    )


@pytest.mark.parametrize(
    "companion_relative_path",
    (
        Path("gym_export/scene_config.json"),
        Path("scene_export/scene_config.json"),
    ),
)
def test_source_scene_resolution_prefers_gym_config_in_mixed_export(
    tmp_path: Path,
    companion_relative_path: Path,
) -> None:
    gym_export = tmp_path / "gym_export"
    gym_export.mkdir(parents=True)
    gym_config = gym_export / "gym_config.json"
    gym_config.write_text("{}", encoding="utf-8")
    companion = tmp_path / companion_relative_path
    companion.parent.mkdir(parents=True, exist_ok=True)
    companion.write_text(
        json.dumps({"format": "embodichain.scene-export/v1"}), encoding="utf-8"
    )

    resolved = resolve_source_scene(tmp_path)

    assert resolved.path == gym_config
    assert resolved.source_format == "legacy_gym_config"
    assert resolved.is_prompt2scene is True
    assert resolve_gym_config_path(tmp_path) == resolved.path


def test_explicit_scene_export_config_overrides_mixed_layout(
    gym_export: Path,
    scene_export: Path,
) -> None:
    resolved = resolve_source_scene(scene_export / "scene_config.json")

    assert resolved.path == scene_export / "scene_config.json"
    assert resolved.source_format == "embodichain.scene-export/v1"
    assert resolved.is_prompt2scene is True


def test_scene_export_config_rejects_unknown_format(scene_export: Path) -> None:
    config_path = scene_export / "scene_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["format"] = "embodichain.scene-export/v2"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported format"):
        resolve_source_scene(config_path)


def test_fast_gym_config_has_runnable_franka_contract(gym_export: Path) -> None:
    scene = prepare_scene(gym_export)
    config = build_fast_gym_config(
        scene,
        task_name="line_task",
        task_description="Arrange the can.",
        robot_profile="franka",
        execution_program_hash="a" * 64,
        max_episodes=1,
        max_episode_steps=2000,
        randomize_scene=True,
    )

    assert config["id"] == "ActionEngine-v1"
    assert config["robot"]["uid"] == "DualFrankaPanda"
    assert config["robot"]["init_pos"][2] == pytest.approx(0.35)
    assert config["sensor"][0]["uid"] == "cam_high"
    assert config["env"]["extensions"]["agent_robot_profile"] == "dual_franka"
    assert config["env"]["extensions"]["agent_static_obstacle_uids"] == ["table"]
    assert config["env"]["extensions"]["agent_dynamic_obstacle_uids"] == [
        "interact_can"
    ]
    assert "agent_grasp_runtime_defaults" not in config["env"]["extensions"]
    assert config["env"]["extensions"]["agent_arm_slots"] == {
        "left": {"arm": "left_arm", "eef": "left_eef"},
        "right": {"arm": "right_arm", "eef": "right_eef"},
    }
    assert config["env"]["extensions"]["arm_aim_yaw_offset"] == {
        "left": pytest.approx(0.0),
        "right": pytest.approx(0.0),
    }
    assert config["env"]["extensions"]["action_engine"]["seed_task_graph"] == (
        "seed_task_graph.json"
    )
    assert (
        config["env"]["extensions"]["action_engine"]["defaults_schema_version"]
        == "action_engine_defaults_v1"
    )
    registry = config["env"]["events"]["register_info_to_env"]["params"]["registry"]
    assert [entry["entity_cfg"]["uid"] for entry in registry] == ["interact_can"]
    assert "randomize_interact_can_pose" in config["env"]["events"]
    assert "randomize_table_height" in config["env"]["events"]
    object_length = config["env"]["events"]["prepare_extra_attr"]["params"]["attrs"][0]
    assert object_length["func_kwargs"]["sample_points"] == 5000
    assert (
        config["env"]["dataset"]["lerobot"]["params"]["robot_meta"]["control_freq"]
        == 25
    )
    assert config["env"]["observations"]["norm_robot_eef_joint"]["params"][
        "joint_ids"
    ] == list(range(14, 26))


def test_fast_gym_config_uses_task_name_for_lerobot_directory_label(
    gym_export: Path,
) -> None:
    scene = prepare_scene(gym_export)
    task_name = "task1000"
    task_description = (
        "先用左臂把番茄放到砧板上，然后用左臂把黄瓜放到砧板右边；"
        "再用左臂把胡萝卜放进碗里。"
    )

    config = build_fast_gym_config(
        scene,
        task_name=task_name,
        task_description=task_description,
        robot_profile="franka",
        execution_program_hash="a" * 64,
        max_episodes=1,
        max_episode_steps=2000,
    )

    params = config["env"]["dataset"]["lerobot"]["params"]
    assert params["instruction"]["lang"] == task_description
    assert params["extra"]["task_name"] == task_name
    assert params["extra"]["task_description"] == task_name


def test_ab_config_uses_offline_branch_and_four_vlm_cameras(
    gym_export: Path,
    tmp_path: Path,
) -> None:
    scene = prepare_scene(gym_export)
    graph_path = "offline/seed_task_graph.json"
    config = build_fast_gym_config(
        scene,
        task_name="ab_task",
        task_description="扶正易拉罐。",
        robot_profile="ur10",
        execution_program_hash="d" * 64,
        max_episodes=1,
        max_episode_steps=100,
        planning_mode="ab",
        seed_task_graph_path=graph_path,
    )
    agent = build_agent_config(
        task_name="ab_task",
        robot_profile="ur10",
        execution_program_hash="d" * 64,
        source_config_path=scene.source_config_path,
        uid_map=scene.uid_map,
        planning_mode="ab",
        seed_task_graph_path=graph_path,
        vlm_model="mimo-vlm",
        vlm_camera_uids=[
            "vlm_front",
            "vlm_left",
            "vlm_rear",
            "vlm_right",
        ],
    )
    paths = artifact_paths(tmp_path, planning_mode="ab")

    assert paths.seed_task_graph == tmp_path.resolve() / graph_path
    assert config["env"]["extensions"]["action_engine"]["planning_mode"] == "ab"
    assert config["env"]["extensions"]["action_engine"]["seed_task_graph"] == (
        graph_path
    )
    vlm_sensors = [
        sensor for sensor in config["sensor"] if sensor["uid"].startswith("vlm_")
    ]
    assert [sensor["uid"] for sensor in vlm_sensors] == [
        "vlm_front",
        "vlm_left",
        "vlm_rear",
        "vlm_right",
    ]
    assert all(
        sensor["enable_color"] and sensor["enable_depth"] for sensor in vlm_sensors
    )
    assert agent["planning_mode"] == "ab"
    assert agent["offline_seed_task_graph"] == graph_path
    assert agent["vlm_model"] == "mimo-vlm"
    assert agent["vlm_camera_uids"] == [
        "vlm_front",
        "vlm_left",
        "vlm_rear",
        "vlm_right",
    ]
    assert agent["online_planning"] == {
        "vlm_model": "mimo-vlm",
        "camera_uids": ["vlm_front", "vlm_left", "vlm_rear", "vlm_right"],
    }


def test_ab_builders_default_to_the_offline_graph_path(gym_export: Path) -> None:
    scene = prepare_scene(gym_export)
    config = build_fast_gym_config(
        scene,
        task_name="ab_default_path",
        task_description="A/B path smoke test.",
        robot_profile="ur10",
        execution_program_hash="e" * 64,
        max_episodes=1,
        max_episode_steps=10,
        planning_mode="ab",
    )
    agent = build_agent_config(
        task_name="ab_default_path",
        robot_profile="ur10",
        execution_program_hash="e" * 64,
        source_config_path=scene.source_config_path,
        uid_map=scene.uid_map,
        planning_mode="ab",
    )

    expected = "offline/seed_task_graph.json"
    assert config["env"]["extensions"]["action_engine"]["seed_task_graph"] == expected
    assert agent["seed_task_graph"] == expected
    assert agent["online_planning"]["camera_uids"] == [
        "vlm_front",
        "vlm_left",
        "vlm_rear",
        "vlm_right",
    ]


def test_ab_scene_requirements_declare_four_vlm_views() -> None:
    requirements = {
        "schema_version": "action_engine_scene_requirements_v2",
        "task_id": "ab",
        "objects": [
            {
                "role_id": "object_01",
                "category": "can",
                "count": 1,
                "affordances": ["graspable"],
                "initial_state": {},
                "attributes": {},
            }
        ],
        "cameras": [],
        "spatial_constraints": [],
        "distractor_count": 0,
        "metadata": {},
    }
    output = _add_ab_camera_requirements(requirements)
    assert [item["uid"] for item in output["cameras"]] == [
        "vlm_front",
        "vlm_left",
        "vlm_rear",
        "vlm_right",
    ]
    assert all(item["modalities"] == ["rgb", "depth"] for item in output["cameras"])


def test_ab_builder_rejects_noncanonical_vlm_camera_ids(gym_export: Path) -> None:
    scene = prepare_scene(gym_export)
    with pytest.raises(ValueError, match="canonical"):
        build_agent_config(
            task_name="ab_invalid_cameras",
            robot_profile="ur10",
            execution_program_hash="f" * 64,
            source_config_path=scene.source_config_path,
            uid_map=scene.uid_map,
            planning_mode="ab",
            vlm_camera_uids=["front", "left", "rear", "right"],
        )


@pytest.mark.parametrize(
    ("profile", "robot_uid", "solver_type"),
    [
        ("dual_ur3", "DualUR3", "ur3"),
        ("dual_ur5", "DualUR5", "ur5"),
        ("dual_ur10", "DualUR10", "ur10"),
        ("dual_franka", "DualFrankaPanda", None),
    ],
)
def test_fast_gym_config_supports_all_robot_profiles(
    gym_export: Path,
    profile: str,
    robot_uid: str,
    solver_type: str | None,
) -> None:
    scene = prepare_scene(gym_export)
    config = build_fast_gym_config(
        scene,
        task_name="profile_task",
        task_description="Profile smoke test.",
        robot_profile=profile,
        execution_program_hash="b" * 64,
        max_episodes=1,
        max_episode_steps=20,
    )

    assert config["robot"]["uid"] == robot_uid
    assert config["env"]["extensions"]["agent_robot_profile"] == profile
    if solver_type is not None:
        assert config["robot"]["solver_cfg"]["left_arm"]["ur_type"] == solver_type


@pytest.mark.parametrize(
    (
        "profile",
        "expected_position_xy",
        "expected_rotation",
        "expected_world_x",
    ),
    [
        ("ur10", [2.0, 0.0], [0.0, 0.0, 0.0], 0.9),
        ("franka", [-0.7, 0.0], [0.0, 0.0, 180.0], 0.55),
    ],
)
def test_dual_robot_profiles_use_identity_mounts_and_same_side_arm_names(
    gym_export: Path,
    profile: str,
    expected_position_xy: list[float],
    expected_rotation: list[float],
    expected_world_x: float,
) -> None:
    scene = prepare_scene(gym_export)
    config = build_fast_gym_config(
        scene,
        task_name="dual_ur_frame_task",
        task_description="Verify the Dual-UR world frame.",
        robot_profile=profile,
        execution_program_hash="c" * 64,
        max_episodes=1,
        max_episode_steps=20,
    )

    robot = config["robot"]
    robot_yaw = np.deg2rad(float(robot["init_rot"][2]))
    robot_rotation = np.array(
        [
            [np.cos(robot_yaw), -np.sin(robot_yaw), 0.0],
            [np.sin(robot_yaw), np.cos(robot_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    robot_position = np.asarray(robot["init_pos"], dtype=np.float64)
    components = {
        component["component_type"]: np.asarray(
            component["transform"], dtype=np.float64
        )
        for component in robot["urdf_cfg"]["components"]
        if component["component_type"] in {"left_arm", "right_arm"}
    }
    world_transforms = {}
    for side, component in components.items():
        world = np.eye(4)
        world[:3, :3] = robot_rotation @ component[:3, :3]
        world[:3, 3] = robot_position + robot_rotation @ component[:3, 3]
        world_transforms[side] = world

    assert robot["init_pos"][:2] == pytest.approx(expected_position_xy)
    assert robot["init_rot"] == pytest.approx(expected_rotation)
    assert world_transforms["left_arm"][:3, 3] == pytest.approx(
        [expected_world_x, -0.3, world_transforms["left_arm"][2, 3]]
    )
    assert world_transforms["right_arm"][:3, 3] == pytest.approx(
        [expected_world_x, 0.3, world_transforms["right_arm"][2, 3]]
    )
    np.testing.assert_allclose(components["left_arm"][:3, :3], np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(components["right_arm"][:3, :3], np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(
        world_transforms["left_arm"][:3, :3], robot_rotation, atol=1.0e-12
    )
    np.testing.assert_allclose(
        world_transforms["right_arm"][:3, :3], robot_rotation, atol=1.0e-12
    )


def test_fast_gym_config_keeps_scene_deterministic_by_default(
    gym_export: Path,
) -> None:
    scene = prepare_scene(gym_export)
    config = build_fast_gym_config(
        scene,
        task_name="deterministic_task",
        task_description="Keep the source scene fixed.",
        robot_profile="ur10",
        execution_program_hash="d" * 64,
        max_episodes=1,
        max_episode_steps=20,
    )

    events = config["env"]["events"]
    assert "randomize_interact_can_pose" not in events
    assert "randomize_table_height" not in events


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("franka", "dual_franka"),
        ("ur5", "dual_ur5"),
        ("ur10", "dual_ur10"),
    ],
)
def test_required_cli_robot_aliases_build_runnable_profiles(
    gym_export: Path,
    alias: str,
    canonical: str,
) -> None:
    scene = prepare_scene(gym_export)
    config = build_fast_gym_config(
        scene,
        task_name="profile_alias_task",
        task_description="Profile alias smoke test.",
        robot_profile=alias,
        execution_program_hash="c" * 64,
        max_episodes=1,
        max_episode_steps=20,
    )

    assert config["env"]["extensions"]["agent_robot_profile"] == canonical


def test_source_scene_scale_policies_are_deterministic(gym_export: Path) -> None:
    preserved = prepare_scene(gym_export)
    multiplied = prepare_scene(
        gym_export,
        body_scale_policy="multiply",
        body_scale=(2.0, 3.0, 4.0),
    )
    absolute = prepare_scene(
        gym_export,
        body_scale_policy="absolute",
        body_scale=(2.0, 3.0, 4.0),
    )

    assert preserved.body_scale_policy == "preserve"
    assert multiplied.rigid_objects[0]["body_scale"] == [2.0, 3.0, 4.0]
    assert absolute.rigid_objects[0]["body_scale"] == [2.0, 3.0, 4.0]
    assert multiplied.asset_hashes == absolute.asset_hashes


def test_artifact_writer_refuses_implicit_overwrite(tmp_path: Path) -> None:
    payload = {"value": 1}
    paths = write_generation_artifacts(
        tmp_path,
        gym_config=payload,
        agent_config=payload,
        task_spec=payload,
        scene_requirements=payload,
        seed_task_graph=payload,
        seed_task_graph_png=b"\x89PNG\r\n\x1a\nold",
        overwrite=False,
    )
    assert json.loads(paths.gym_config.read_text(encoding="utf-8")) == payload
    assert paths.seed_task_graph_png.read_bytes().startswith(b"\x89PNG")

    # A leftover PNG participates in the same preflight as every JSON artifact.
    for path in (
        paths.gym_config,
        paths.agent_config,
        paths.task_spec,
        paths.scene_requirements,
        paths.execution_program,
    ):
        path.unlink()
    with pytest.raises(FileExistsError, match="--overwrite"):
        write_generation_artifacts(
            tmp_path,
            gym_config=payload,
            agent_config=payload,
            task_spec=payload,
            scene_requirements=payload,
            seed_task_graph=payload,
            seed_task_graph_png=b"\x89PNG\r\n\x1a\nnew",
            overwrite=False,
        )

    replaced = write_generation_artifacts(
        tmp_path,
        gym_config=payload,
        agent_config=payload,
        task_spec=payload,
        scene_requirements=payload,
        seed_task_graph=payload,
        seed_task_graph_png=b"\x89PNG\r\n\x1a\nnew",
        overwrite=True,
    )
    assert replaced.seed_task_graph_png.read_bytes().endswith(b"new")


def test_artifact_writer_creates_ab_branch_directory(tmp_path: Path) -> None:
    payload = {"value": "ab"}
    paths = write_generation_artifacts(
        tmp_path,
        gym_config=payload,
        agent_config=payload,
        task_spec=payload,
        scene_requirements=payload,
        seed_task_graph=payload,
        seed_task_graph_png=b"\x89PNG\r\n\x1a\nab",
        overwrite=False,
        planning_mode="ab",
    )

    assert paths.seed_task_graph.parent == tmp_path / "offline"
    assert json.loads(paths.seed_task_graph.read_text(encoding="utf-8")) == payload


def test_generation_calls_planner_compiler_and_renderer_once(
    gym_export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_engine import compiler, tasks
    from embodichain.gen_sim.action_engine.generation import generator

    planner_call: dict[str, object] = {}
    rendered: dict[str, object] = {}
    published: dict[str, object] = {}

    real_plan = tasks.plan_grounded_task_spec

    def fake_plan_task_spec(**kwargs):
        planner_call.update(kwargs)
        return real_plan(**kwargs)

    monkeypatch.setattr(tasks, "plan_grounded_task_spec", fake_plan_task_spec)
    renderer_module = ModuleType(
        "embodichain.gen_sim.action_engine.graph_visualization"
    )

    def fake_renderer(program):
        rendered["program"] = program
        return b"\x89PNG\r\n\x1a\nseed"

    renderer_module.render_seed_task_graph_png = fake_renderer
    monkeypatch.setitem(sys.modules, renderer_module.__name__, renderer_module)
    real_writer = generator.write_generation_artifacts

    def capture_writer(*args, **kwargs):
        published["program"] = kwargs["seed_task_graph"]
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(generator, "write_generation_artifacts", capture_writer)
    assert callable(compiler.compile_task_agent)
    output_dir = tmp_path / "configs"
    paths = generate_action_engine_config(
        gym_export,
        output_dir,
        task_name="line_task",
        task_description="扶正红色易拉罐。",
        robot_profile="franka",
        instruction_parser="deterministic",
    )

    assert planner_call["task_name"] == "line_task"
    assert planner_call["task_description"] == "扶正红色易拉罐。"
    assert planner_call["robot_profile"] == "franka"
    planner_objects = planner_call["scene_objects"]
    assert isinstance(planner_objects, list)
    assert {obj["uid"] for obj in planner_objects} == {"table", "interact_can"}
    assert {path.name for path in output_dir.iterdir()} == {
        "fast_gym_config.json",
        "agent_config.json",
        "task_spec.json",
        "scene_requirements.json",
        "seed_task_graph.json",
        "seed_task_graph.png",
    }
    assert paths.seed_task_graph_png.read_bytes() == b"\x89PNG\r\n\x1a\nseed"
    assert rendered["program"] is published["program"]

    agent_config = json.loads(paths.agent_config.read_text(encoding="utf-8"))
    assert agent_config["schema_version"] == "action_engine_config_v2"
    assert agent_config["task_spec"] == "task_spec.json"
    assert agent_config["scene_requirements"] == "scene_requirements.json"
    assert agent_config["seed_task_graph"] == "seed_task_graph.json"
    assert len(agent_config["seed_task_graph_hash"]) == 64
    assert agent_config["runtime_policy"]["schema_version"] == (
        "action_engine_runtime_policy_v4"
    )
    assert agent_config["runtime_policy"]["planner"]["dynamic_collision"] is True
    assert agent_config["runtime_policy"]["planner"]["static_obstacle_uids"] == [
        "table"
    ]
    assert agent_config["runtime_policy"]["planner"]["dynamic_obstacle_uids"] == [
        "interact_can"
    ]
    assert len(agent_config["runtime_policy_hash"]) == 64
    assert "png" not in json.dumps(agent_config).lower()

    from embodichain.gen_sim.action_engine.runtime import (
        load_agent_execution_program,
    )

    regenerated = load_agent_execution_program(
        agent_config,
        agent_config_path=paths.agent_config,
        regenerate=True,
    )
    assert regenerated.task == "line_task"
    assert regenerated.seed_graph is not None


def test_existing_v2_task_spec_bypasses_text_planner_and_derives_scene_requirements(
    gym_export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_engine import tasks

    renderer_module = ModuleType(
        "embodichain.gen_sim.action_engine.graph_visualization"
    )
    renderer_module.render_seed_task_graph_png = lambda _program: b"direct-task-png"
    monkeypatch.setitem(sys.modules, renderer_module.__name__, renderer_module)

    def unexpected_text_planner(**_kwargs):
        raise AssertionError("an existing TaskSpec must not invoke text planning")

    monkeypatch.setattr(
        tasks, "interpret_and_ground_task_spec", unexpected_text_planner
    )
    input_path = tmp_path / "task_spec.json"
    input_path.write_text(
        json.dumps(_existing_v2_task_spec()),
        encoding="utf-8",
    )

    paths = generate_action_engine_config(
        gym_export,
        tmp_path / "generated",
        task_name="direct_task",
        task_spec=input_path,
        robot_profile="ur10",
    )

    persisted_task = json.loads(paths.task_spec.read_text(encoding="utf-8"))
    persisted_requirements = json.loads(
        paths.scene_requirements.read_text(encoding="utf-8")
    )
    assert persisted_task["metadata"]["role_bindings"] == {"object_01": "interact_can"}
    assert [item["role_id"] for item in persisted_requirements["objects"]] == [
        "object_01"
    ]
    assert persisted_requirements["metadata"]["source"] == ("task_spec_role_bindings")
    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    assert (
        gym_config["env"]["dataset"]["lerobot"]["params"]["instruction"]["lang"]
        == "扶正这个红色易拉罐。"
    )


def test_existing_v2_task_spec_uses_validated_scene_requirements_sidecar(
    gym_export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer_module = ModuleType(
        "embodichain.gen_sim.action_engine.graph_visualization"
    )
    renderer_module.render_seed_task_graph_png = lambda _program: b"sidecar-png"
    monkeypatch.setitem(sys.modules, renderer_module.__name__, renderer_module)

    input_dir = tmp_path / "task-first"
    input_dir.mkdir()
    task = _existing_v2_task_spec("sidecar_task")
    requirements = {
        "schema_version": SCENE_REQUIREMENTS_SCHEMA,
        "task_id": "sidecar_task",
        "objects": [
            {
                "role_id": "object_01",
                "category": "can",
                "count": 1,
                "affordances": ["graspable", "orientable"],
                "initial_state": {"orientation": "fallen"},
                "attributes": {"color": "red"},
            }
        ],
        "cameras": [],
        "spatial_constraints": [],
        "distractor_count": 0,
        "metadata": {"task_first": True},
    }
    (input_dir / "task_spec.json").write_text(
        json.dumps(task),
        encoding="utf-8",
    )
    (input_dir / "scene_requirements.json").write_text(
        json.dumps(requirements),
        encoding="utf-8",
    )

    paths = generate_action_engine_config(
        gym_export,
        tmp_path / "generated-sidecar",
        task_name="sidecar_task",
        task_spec=input_dir / "task_spec.json",
        robot_profile="ur10",
    )

    assert json.loads(paths.scene_requirements.read_text(encoding="utf-8")) == (
        requirements
    )


def test_task_factory_style_sidecar_binds_roles_without_text_llm(
    gym_export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer_module = ModuleType(
        "embodichain.gen_sim.action_engine.graph_visualization"
    )
    renderer_module.render_seed_task_graph_png = lambda _program: b"task-first-png"
    monkeypatch.setitem(sys.modules, renderer_module.__name__, renderer_module)
    source_path = gym_export / "gym_config.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["rigid_object"][0]["affordances"] = ["graspable", "orientable"]
    source["rigid_object"][0]["initial_state"] = {"orientation": "fallen"}
    source_path.write_text(json.dumps(source), encoding="utf-8")

    input_dir = tmp_path / "task-first-unbound"
    input_dir.mkdir()
    task = _existing_v2_task_spec("task_first_unbound")
    task["metadata"] = {"generator": "TaskFactory-v2"}
    requirements = {
        "schema_version": SCENE_REQUIREMENTS_SCHEMA,
        "task_id": "task_first_unbound",
        "objects": [
            {
                "role_id": "object_01",
                "category": "can",
                "count": 1,
                "affordances": ["graspable", "orientable"],
                "initial_state": {"orientation": "fallen"},
                "attributes": {"color": "red"},
            }
        ],
        "cameras": [],
        "spatial_constraints": [],
        "distractor_count": 0,
        "metadata": {"task_first": True},
    }
    (input_dir / "task_spec.json").write_text(json.dumps(task), encoding="utf-8")
    (input_dir / "scene_requirements.json").write_text(
        json.dumps(requirements), encoding="utf-8"
    )

    paths = generate_action_engine_config(
        gym_export,
        tmp_path / "generated-unbound-sidecar",
        task_name="task_first_unbound",
        task_spec=input_dir / "task_spec.json",
        robot_profile="ur10",
    )

    task_artifact = json.loads(paths.task_spec.read_text(encoding="utf-8"))
    assert task_artifact["metadata"]["role_bindings"] == {"object_01": "interact_can"}


def test_task_spec_input_rejects_natural_language_and_task_agent_conflicts(
    gym_export: Path,
    tmp_path: Path,
) -> None:
    task = _existing_v2_task_spec()
    with pytest.raises(ValueError, match="task_spec cannot be combined"):
        generate_action_engine_config(
            gym_export,
            tmp_path / "conflict-description",
            task_name="direct_task",
            task_description="do something",
            task_spec=task,
            robot_profile="ur10",
        )
    with pytest.raises(ValueError, match="task_spec cannot be combined"):
        generate_action_engine_config(
            gym_export,
            tmp_path / "conflict-agent",
            task_name="direct_task",
            task_agent={"schema_version": TASK_AGENT_SCHEMA},
            task_spec=task,
            robot_profile="ur10",
        )


def test_task_spec_role_binding_accepts_legacy_oracle_and_rejects_conflicts() -> None:
    task = _existing_v2_task_spec()
    task["metadata"] = {}
    task["oracle"] = {"role_bindings": {"object_01": "interact_can"}}
    assert _task_spec_role_bindings(task, ["table", "interact_can"]) == {
        "object_01": "interact_can"
    }

    task["metadata"] = {"role_bindings": {"object_01": "table"}}
    with pytest.raises(ValueError, match="Conflicting role_bindings"):
        _task_spec_role_bindings(task, ["table", "interact_can"])


def test_task_spec_role_binding_merges_non_overlapping_handoffs() -> None:
    task = _existing_v2_task_spec()
    task["task_instances"][0]["params"]["target_role"] = "object_02"
    task["metadata"] = {"role_bindings": {"object_01": "interact_can"}}
    task["oracle"] = {"role_bindings": {"object_02": "interact_target"}}

    assert _task_spec_role_bindings(
        task,
        ["table", "interact_can", "interact_target"],
    ) == {"object_01": "interact_can", "object_02": "interact_target"}


def test_task_factory_sidecar_requires_static_affordance_and_state_evidence() -> None:
    task = _existing_v2_task_spec("missing-static-evidence")
    task["metadata"] = {}
    requirements = {
        "objects": [
            {
                "role_id": "object_01",
                "category": "can",
                "count": 1,
                "affordances": ["graspable", "orientable"],
                "initial_state": {"orientation": "fallen"},
                "attributes": {"color": "red"},
            }
        ]
    }
    scene = [
        {
            "runtime_uid": "interact_can",
            "description": "A red soda can.",
            "init_pos": [0.0, 0.0, 0.7],
        }
    ]

    with pytest.raises(ValueError, match="requires one unambiguous scene match"):
        _task_spec_role_bindings(
            task,
            ["interact_can"],
            scene_requirements=requirements,
            scene_objects=scene,
            robot_profile="ur10",
        )


def test_ab_generation_writes_shared_and_offline_branch_artifacts(
    gym_export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer_module = ModuleType(
        "embodichain.gen_sim.action_engine.graph_visualization"
    )
    renderer_module.render_seed_task_graph_png = lambda _program: b"ab-seed-png"
    monkeypatch.setitem(sys.modules, renderer_module.__name__, renderer_module)

    output_dir = tmp_path / "ab-config"
    paths = generate_action_engine_config(
        gym_export,
        output_dir,
        task_name="ab_task",
        task_description="扶正红色易拉罐。",
        robot_profile="ur10",
        instruction_parser="deterministic",
        planning_mode="ab",
        vlm_model="mimo-vlm",
    )

    assert paths.seed_task_graph == output_dir / "offline/seed_task_graph.json"
    assert paths.seed_task_graph_png == output_dir / "offline/seed_task_graph.png"
    assert not (output_dir / "seed_task_graph.json").exists()
    agent_config = json.loads(paths.agent_config.read_text(encoding="utf-8"))
    assert agent_config["planning_mode"] == "ab"
    assert agent_config["offline_seed_task_graph"] == "offline/seed_task_graph.json"
    assert agent_config["online_planning"]["vlm_model"] == "mimo-vlm"
    scene_requirements = json.loads(
        paths.scene_requirements.read_text(encoding="utf-8")
    )
    assert [camera["uid"] for camera in scene_requirements["cameras"]] == [
        "vlm_front",
        "vlm_left",
        "vlm_rear",
        "vlm_right",
    ]
    gym_config = json.loads(paths.gym_config.read_text(encoding="utf-8"))
    assert [
        sensor["uid"]
        for sensor in gym_config["sensor"]
        if sensor["uid"].startswith("vlm_")
    ] == ["vlm_front", "vlm_left", "vlm_rear", "vlm_right"]


def test_invalid_explicit_task_fails_before_output_asset_materialization(
    gym_export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_engine import tasks
    from embodichain.gen_sim.action_engine.generation import generator

    normalized = False

    def reject_task(**_kwargs):
        raise ValueError("object selector is ambiguous")

    def record_normalization(*_args, **_kwargs):
        nonlocal normalized
        normalized = True
        raise AssertionError("normalization must not run after planning failure")

    monkeypatch.setattr(tasks, "plan_grounded_task_spec", reject_task)
    monkeypatch.setattr(generator, "normalize_scene_assets", record_normalization)
    output_dir = tmp_path / "invalid"

    with pytest.raises(ValueError, match="ambiguous"):
        generate_action_engine_config(
            gym_export,
            output_dir,
            task_name="invalid_task",
            task_description="扶正黄色瓶子。",
            robot_profile="franka",
            instruction_parser="deterministic",
        )

    assert normalized is False
    assert not output_dir.exists()


def test_agent_config_uses_relative_program_paths(gym_export: Path) -> None:
    scene = prepare_scene(gym_export)
    config = build_agent_config(
        task_name="line_task",
        robot_profile="franka",
        execution_program_hash="b" * 64,
        source_config_path=scene.source_config_path,
        uid_map=scene.uid_map,
    )
    assert config["task_spec"] == "task_spec.json"
    assert config["scene_requirements"] == "scene_requirements.json"
    assert config["seed_task_graph"] == "seed_task_graph.json"
    assert config["runtime_policy"]["arm_selection"]["pickup_crossing_weight"] == 1.0
    assert config["runtime_policy"]["motion_defaults"]["PickUp"][
        "lift_height"
    ] == pytest.approx(0.30)
    assert config["runtime_policy"]["grasp"]["max_open_length"] == pytest.approx(0.115)
    assert len(config["runtime_policy_hash"]) == 64


def test_documented_cli_accepts_franka_profile() -> None:
    args = build_parser().parse_args(
        [
            "--gym_project",
            "gym_export",
            "--output_dir",
            "configs/task4_2",
            "--task_name",
            "task4_2",
            "--task_description",
            "Arrange the cans in a line.",
            "--robot-profile",
            "franka",
            "--overwrite",
        ]
    )
    assert args.robot_profile == "franka"
    assert args.overwrite is True


def test_generation_cli_defaults_to_mature_robot_without_scene_randomization() -> None:
    args = build_parser().parse_args(
        [
            "--gym_project",
            "gym_export",
            "--output_dir",
            "configs/task2_3",
            "--task_name",
            "task2_3",
            "--task_description",
            "Upright both objects.",
        ]
    )

    assert args.robot_profile == "ur10"
    assert args.randomize_scene is False
    assert args.planning_mode == "offline"
    assert args.instruction_parser == "llm"


def test_generation_cli_accepts_ab_models_and_deterministic_compatibility() -> None:
    args = build_parser().parse_args(
        [
            "--gym_project",
            "gym_export",
            "--output_dir",
            "configs/ab",
            "--task_name",
            "ab",
            "--task_description",
            "递给另一只手。",
            "--planning-mode",
            "ab",
            "--instruction-parser",
            "deterministic",
            "--llm-model",
            "text-model",
            "--vlm-model",
            "vision-model",
        ]
    )

    assert args.planning_mode == "ab"
    assert args.instruction_parser == "deterministic"
    assert args.llm_model == "text-model"
    assert args.vlm_model == "vision-model"


def test_generation_cli_accepts_existing_task_spec_without_description() -> None:
    args = build_parser().parse_args(
        [
            "--gym_project",
            "gym_export",
            "--output_dir",
            "configs/direct",
            "--task_name",
            "direct_task",
            "--task-spec",
            "tasks/direct_task/task_spec.json",
        ]
    )

    assert args.task_spec == "tasks/direct_task/task_spec.json"
    assert cli_module._resolve_task_description(args) == ""


def test_generation_cli_reports_seed_png_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = artifact_paths(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "generate_action_engine_config",
        lambda *_args, **_kwargs: paths,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_action_agent_config",
            "--gym_project",
            "gym_export",
            "--output_dir",
            str(tmp_path),
            "--task_name",
            "task4_2",
            "--task_description",
            "Arrange cans.",
        ],
    )

    cli_module.cli()

    assert (
        f"Generated Seed graph PNG: {paths.seed_task_graph_png}"
        in capsys.readouterr().out
    )


def test_task4_line_fallback_preserves_seed_capability() -> None:
    can_uids = [
        "interact_pepsi_can",
        "interact_fanta_can",
        "interact_coca_cola_can",
        "interact_sprite_can",
        "interact_yellow_soda_can",
    ]
    scene_objects = [
        {
            "uid": "table",
            "runtime_uid": "table",
            "role": "background",
            "description": "A table.",
        },
        *[
            {
                "uid": uid,
                "runtime_uid": uid,
                "role": "rigid_object",
                "description": "A soda can.",
            }
            for uid in can_uids
        ],
    ]
    task_agent = plan_task(
        task_name="task4_2",
        task_description="将罐头摆成一排",
        scene_objects=scene_objects,
        deterministic_fallback=True,
    )
    execution_program = compile_task_agent(task_agent)

    assert len(task_agent["semantic_steps"][0]["objects"]) == 5
    assert len(execution_program["semantic_steps"]) == 5
    assert len(execution_program["edges"]) == 30
    assert {step["object"] for step in execution_program["semantic_steps"]} == set(
        can_uids
    )


def _task_agent() -> dict:
    return {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "line_task",
        "goal": "Arrange the can.",
        "semantic_steps": [
            {
                "id": "s1",
                "operator": "hold_hover",
                "object": "interact_can",
                "actor": {"mode": "auto"},
                "goal": {},
                "depends_on": [],
            }
        ],
    }
