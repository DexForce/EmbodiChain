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
import sys

from embodichain.gen_sim.action_agent_pipeline.cli import (
    generate_action_agent_config,
)
from embodichain.gen_sim.action_agent_pipeline.cli.run_agent import (
    build_parser as build_run_agent_parser,
)
from embodichain.gen_sim.action_agent_pipeline.defaults import (
    DEFAULT_MAX_EPISODE_STEPS,
)


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
        "_resolve_target_replacements",
        lambda args: [],
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
            "--overwrite",
        ],
    )

    generate_action_agent_config.cli()

    assert received["task_name"] == "task4_2"
    assert received["task_description"] == "将罐头摆成一排"
    assert received["robot_profile"] == "franka"
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
        ]
    )

    assert args.task_name == "task4_2"
    assert args.regenerate is True
    assert args.headless is True
