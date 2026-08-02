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

import pytest

from embodichain.gen_sim.action_engine.domain import (
    TASK_AGENT_SCHEMA,
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
    generate_action_engine_config,
)
from embodichain.gen_sim.action_engine.generation.source_scene import prepare_scene
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
                "shape": {"shape_type": "Mesh", "fpath": "mesh_assets/can.glb"},
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
    mesh_path = Path(scene.rigid_objects[0]["shape"]["fpath"])
    assert mesh_path.is_absolute()
    assert mesh_path.is_file()
    assert scene.planner_objects[1]["source_uid"] == "interact_can_0"
    assert scene.planner_objects[1]["uid"] == "interact_can"


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
    )

    assert config["id"] == "ActionEngine-v1"
    assert config["robot"]["uid"] == "DualFrankaPanda"
    assert config["robot"]["init_pos"][2] == pytest.approx(0.35)
    assert config["sensor"][0]["uid"] == "cam_high"
    assert config["env"]["extensions"]["agent_robot_profile"] == "dual_franka"
    assert config["env"]["extensions"]["agent_grasp_runtime_defaults"][
        "finger_length"
    ] == pytest.approx(0.13)
    assert config["env"]["extensions"]["action_engine"]["execution_program"] == (
        "seed_task_graph.json"
    )
    registry = config["env"]["events"]["register_info_to_env"]["params"]["registry"]
    assert [entry["entity_cfg"]["uid"] for entry in registry] == ["interact_can"]
    assert "randomize_interact_can_pose" in config["env"]["events"]
    assert "randomize_table_height" in config["env"]["events"]
    assert config["env"]["observations"]["norm_robot_eef_joint"]["params"][
        "joint_ids"
    ] == list(range(14, 26))


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
        task_agent=payload,
        execution_program=payload,
        seed_task_graph_png=b"\x89PNG\r\n\x1a\nold",
        overwrite=False,
    )
    assert json.loads(paths.gym_config.read_text(encoding="utf-8")) == payload
    assert paths.seed_task_graph_png.read_bytes().startswith(b"\x89PNG")

    # A leftover PNG participates in the same preflight as every JSON artifact.
    for path in (
        paths.gym_config,
        paths.agent_config,
        paths.task_agent,
        paths.execution_program,
    ):
        path.unlink()
    with pytest.raises(FileExistsError, match="--overwrite"):
        write_generation_artifacts(
            tmp_path,
            gym_config=payload,
            agent_config=payload,
            task_agent=payload,
            execution_program=payload,
            seed_task_graph_png=b"\x89PNG\r\n\x1a\nnew",
            overwrite=False,
        )

    replaced = write_generation_artifacts(
        tmp_path,
        gym_config=payload,
        agent_config=payload,
        task_agent=payload,
        execution_program=payload,
        seed_task_graph_png=b"\x89PNG\r\n\x1a\nnew",
        overwrite=True,
    )
    assert replaced.seed_task_graph_png.read_bytes().endswith(b"new")


def test_generation_calls_planner_compiler_and_renderer_once(
    gym_export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_engine import compiler, planning
    from embodichain.gen_sim.action_engine.generation import generator

    task_agent = _task_agent()
    planner_call: dict[str, object] = {}
    rendered: dict[str, object] = {}
    published: dict[str, object] = {}

    def fake_plan_task(**kwargs):
        planner_call.update(kwargs)
        return task_agent

    monkeypatch.setattr(planning, "plan_task", fake_plan_task)
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
        published["program"] = kwargs["execution_program"]
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(generator, "write_generation_artifacts", capture_writer)
    assert callable(compiler.compile_task_agent)
    output_dir = tmp_path / "configs"
    paths = generate_action_engine_config(
        gym_export,
        output_dir,
        task_name="line_task",
        task_description="Arrange the can.",
        robot_profile="franka",
    )

    assert planner_call["task_name"] == "line_task"
    assert planner_call["task_description"] == "Arrange the can."
    assert planner_call["model"] is None
    planner_objects = planner_call["scene_objects"]
    assert isinstance(planner_objects, list)
    assert {obj["uid"] for obj in planner_objects} == {"table", "interact_can"}
    assert {path.name for path in output_dir.iterdir()} == {
        "fast_gym_config.json",
        "agent_config.json",
        "task_agent.json",
        "seed_task_graph.json",
        "seed_task_graph.png",
    }
    assert paths.seed_task_graph_png.read_bytes() == b"\x89PNG\r\n\x1a\nseed"
    assert rendered["program"] is published["program"]

    agent_config = json.loads(paths.agent_config.read_text(encoding="utf-8"))
    assert agent_config["schema_version"] == "action_engine_config_v1"
    assert agent_config["task_agent"] == "task_agent.json"
    assert agent_config["execution_program"] == "seed_task_graph.json"
    assert len(agent_config["execution_program_hash"]) == 64
    assert "png" not in json.dumps(agent_config).lower()


def test_agent_config_uses_relative_program_paths(gym_export: Path) -> None:
    scene = prepare_scene(gym_export)
    config = build_agent_config(
        task_name="line_task",
        robot_profile="franka",
        execution_program_hash="b" * 64,
        source_config_path=scene.source_config_path,
        uid_map=scene.uid_map,
    )
    assert config["task_agent"] == "task_agent.json"
    assert config["execution_program"] == "seed_task_graph.json"


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
