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

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from embodichain.gen_sim.action_agent_pipeline.cli import (
    agent_run_stage,
    generate_action_agent_config,
    project_resolution,
)
from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_args import (
    build_parser as build_pipeline_parser,
)
from embodichain.gen_sim.action_agent_pipeline.cli.project_resolution import (
    resolve_task_description_for_generation,
)
from embodichain.gen_sim.action_agent_pipeline.cli.run_agent import (
    build_parser as build_run_agent_parser,
)
from embodichain.gen_sim.action_agent_pipeline.cli import run_agent
from embodichain.gen_sim.action_agent_pipeline.defaults import (
    DEFAULT_MAX_EPISODE_STEPS,
)
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    make_relative_seed_task_graph,
)

_RETIRED_SCENE_SOURCE_OPTIONS = {
    "--background",
    "--gym-project-root",
    "--image2scene-client-url",
    "--image2scene-download-dir",
    "--image2scene-extract-dir",
    "--image2scene-gen-config",
    "--image2scene-llm-config",
    "--image2scene-merged-output",
    "--image2scene-output-root",
    "--image2scene-root",
    "--job-timeout-s",
    "--job_timeout_s",
    "--overwrite-gym-project",
    "--poll-interval",
    "--server",
    "--skip-health-check",
    "--use-image2scene",
}


def test_documented_config_generation_command_remains_accepted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    received = {}
    output_dir = tmp_path / "task4_2"

    def fake_generate(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(
            gym_config=output_dir / "fast_gym_config.json",
            agent_config=output_dir / "agent_config.json",
            task_prompt=output_dir / "task_prompt.txt",
            task_graph=output_dir / "task_graph.json",
            basic_background=output_dir / "basic_background.txt",
            atom_actions=output_dir / "atom_actions.txt",
            summary={},
        )

    monkeypatch.setattr(
        generate_action_agent_config,
        "generate_action_agent_config_from_project",
        fake_generate,
    )
    monkeypatch.setattr(
        generate_action_agent_config,
        "_resolve_source_alignment",
        lambda args: {
            "preserve_source_scene_geometry": True,
            "load_source_meshes_directly": True,
            "source_scene_z_rotation_degrees": -90.0,
            "source_mesh_x_rotation_degrees": 0.0,
        },
    )
    monkeypatch.setattr(
        generate_action_agent_config,
        "_resolve_source_scene_body_scale_mode",
        lambda args: "multiply",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_action_agent_config",
            "--gym_project",
            "gym_project/prompt2scene/task4_2/gym_export",
            "--output_dir",
            str(output_dir),
            "--task_name",
            "task4_2",
            "--task_description",
            "将罐头摆成一排",
            "--robot-profile",
            "franka",
            "--render-graphs",
            "--overwrite",
        ],
    )

    generate_action_agent_config.cli()

    assert received["task_name"] == "task4_2"
    assert received["task_description"] == "将罐头摆成一排"
    assert received["robot_profile"] == "franka"
    assert received["render_graphs"] is True
    assert received["overwrite"] is True
    assert received["max_episode_steps"] == DEFAULT_MAX_EPISODE_STEPS == 2000


def test_documented_run_agent_command_remains_accepted() -> None:
    args = build_run_agent_parser().parse_args(
        [
            "--task_name",
            "task4_2",
            "--gym_config",
            "/tmp/task4_2/fast_gym_config.json",
            "--agent_config",
            "/tmp/task4_2/agent_config.json",
            "--regenerate",
            "--headless",
            "--render-graphs",
        ]
    )

    assert args.task_name == "task4_2"
    assert args.regenerate is True
    assert args.headless is True
    assert args.render_graphs is True


def test_graph_rendering_is_disabled_by_default() -> None:
    run_args = build_run_agent_parser().parse_args(
        [
            "--task_name",
            "task",
            "--gym_config",
            "/tmp/fast_gym_config.json",
            "--agent_config",
            "/tmp/agent_config.json",
        ]
    )
    pipeline_args = build_pipeline_parser().parse_args(
        ["--task_description", "扶正物体"]
    )

    assert run_args.render_graphs is False
    assert pipeline_args.render_graphs is False


def test_run_agent_render_graphs_writes_seed_png(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seed_task_graph.json"
    placement = SimpleNamespace(
        intent="place_relative",
        moved_runtime_uid="object",
        reference_runtime_uid="table",
        relation="on",
        reference_is_initial_pose=False,
        orientation_goal="preserve",
        orientation_axis="none",
        orientation_align_to_runtime_uid=None,
        arm_request="left",
        step_id="s01_place",
        depends_on=(),
    )
    seed = make_relative_seed_task_graph(
        "task/render",
        SimpleNamespace(
            intent="place_relative",
            placements=(placement,),
            coordinated_direction=None,
            coordinated_terminal_behavior=None,
        ),
    )
    seed_path.write_text(
        json.dumps(seed, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_agent, "_graph_output_root", lambda: tmp_path / "graphs")
    env = SimpleNamespace(seed_task_graph_path=seed_path)

    renderer = run_agent._configure_graph_rendering(
        SimpleNamespace(render_graphs=True, task_name="fallback"),
        env,
    )

    output_path = tmp_path / "graphs" / "task_render" / "seed_task_graph.png"
    assert renderer is not None
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_run_agent_render_graphs_injects_runtime_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    renderer = lambda graph: b"png"
    monkeypatch.setattr(
        run_agent,
        "_configure_graph_rendering",
        lambda args, env: renderer,
    )
    monkeypatch.setattr(
        run_agent,
        "run_action_agent",
        lambda **kwargs: captured.update(kwargs),
    )

    run_agent._run_action_agent(
        SimpleNamespace(
            task_name="task",
            render_graphs=True,
            regenerate=False,
            headless=True,
        ),
        SimpleNamespace(),
        {"max_episodes": 1},
    )

    assert captured["runtime_graph_renderer"] is renderer


def test_pipeline_subprocess_forwards_render_graphs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(agent_run_stage.subprocess, "run", fake_run)

    result = agent_run_stage.run_agent_command(
        task_name="task",
        gym_config=tmp_path / "fast_gym_config.json",
        agent_config=tmp_path / "agent_config.json",
        regenerate=False,
        render_graphs=True,
    )

    assert result == 0
    assert "--render-graphs" in captured["command"]


def test_config_generation_requires_a_task_description() -> None:
    """There is no default task template, so an empty goal must fail loudly."""
    args = SimpleNamespace(task_description=None, task_file=None)

    with pytest.raises(ValueError, match="--task_description"):
        generate_action_agent_config._resolve_task_description(args)


def test_config_generation_reads_task_description_from_task_file(
    tmp_path: Path,
) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_text("  将罐头摆成一排  ", encoding="utf-8")
    args = SimpleNamespace(task_description=None, task_file=str(task_file))

    assert (
        generate_action_agent_config._resolve_task_description(args) == "将罐头摆成一排"
    )


def test_pipeline_requires_a_task_description() -> None:
    """The full pipeline fails before scene resolution rather than mid-run."""
    with pytest.raises(ValueError, match="--task_description"):
        resolve_task_description_for_generation(SimpleNamespace(task_description=""))

    assert (
        resolve_task_description_for_generation(
            SimpleNamespace(task_description="  stack the cans  ")
        )
        == "stack the cans"
    )


@pytest.mark.parametrize(
    "use_prompt2scene",
    [False, True],
    ids=["default", "compatibility-flag"],
)
def test_pipeline_resolves_prompt2scene_as_default_source(
    monkeypatch,
    tmp_path: Path,
    use_prompt2scene: bool,
) -> None:
    gym_config_path = tmp_path / "gym_config.json"
    monkeypatch.setattr(
        project_resolution,
        "run_prompt2scene_stage",
        lambda args: gym_config_path,
    )
    args = SimpleNamespace(
        base_history_index=None,
        base_task_name=None,
        use_existing_gym_project=False,
        use_prompt2scene=use_prompt2scene,
    )

    resolution = project_resolution.resolve_gym_project(args)

    assert resolution == project_resolution.ProjectResolution(
        path=gym_config_path,
        mode="prompt2scene",
    )


def test_pipeline_parser_does_not_accept_retired_scene_source_options() -> None:
    parser = build_pipeline_parser()

    assert _RETIRED_SCENE_SOURCE_OPTIONS.isdisjoint(parser._option_string_actions)
