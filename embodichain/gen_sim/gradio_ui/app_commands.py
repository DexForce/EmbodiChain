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

"""CLI command builders for EmbodiChain pipelines."""

from __future__ import annotations

import sys
from typing import Protocol

from app_config import (
    COMMANDS,
    ROBOT_PROFILE_FRANKA,
    ROBOT_PROFILE_UR5,
    ROBOT_PROFILE_UR10,
    SCENE_ID,
)


class ScenePathsLike(Protocol):
    scene_id: str
    image_path: object
    fast_gym_config: object
    agent_config: object


def robot_profile_cli_value(robot_profile: str | None) -> str | None:
    return {
        ROBOT_PROFILE_FRANKA: "franka",
        ROBOT_PROFILE_UR5: "dual_ur5",
        ROBOT_PROFILE_UR10: "dual_ur10",
    }.get(robot_profile)


def _pipeline_paths(paths: ScenePathsLike) -> tuple[str, str]:
    return (
        f"gym_project/{paths.scene_id}",
        f"gym_project/action_agent_pipeline/configs/{paths.scene_id}",
    )


def build_initial_pipeline_command(
    task_text: str,
    paths: ScenePathsLike,
    prompt2scene_prompt: str = "",
    robot_profile: str | None = None,
    load_template_material: bool = False,
) -> list[str]:
    prompt_root, config_dir = _pipeline_paths(paths)
    command = [
        sys.executable,
        "-m",
        COMMANDS["pipeline"]["module"],
        "--image",
        str(paths.image_path.resolve()),
        "--prompt2scene-output-root",
        prompt_root,
        "--config-output-dir",
        config_dir,
        "--task_name",
        SCENE_ID,
        "--task_description",
        task_text,
        *COMMANDS["pipeline"]["base_args"],
    ]
    if profile := robot_profile_cli_value(robot_profile):
        command.extend(["--robot-profile", profile])
    if prompt2scene_prompt.strip():
        command.extend(["--prompt2scene-prompt", prompt2scene_prompt.strip()])
    if load_template_material:
        command.append("--load-template-material")
    return command


def build_scene_edit_pipeline_command(
    task_text: str,
    env_text: str,
    paths: ScenePathsLike,
    robot_profile: str | None = None,
    load_template_material: bool = False,
) -> list[str]:
    prompt_root, config_dir = _pipeline_paths(paths)
    command = [
        sys.executable,
        "-m",
        COMMANDS["pipeline"]["module"],
        "--prompt2scene-output-root",
        prompt_root,
        "--prompt2scene-prompt",
        env_text,
        "--config-output-dir",
        config_dir,
        "--task_name",
        SCENE_ID,
        "--task_description",
        task_text,
        *COMMANDS["pipeline"]["base_args"],
    ]
    if profile := robot_profile_cli_value(robot_profile):
        command.extend(["--robot-profile", profile])
    if load_template_material:
        command.append("--load-template-material")
    return command


def build_config_command_for_paths(
    task_text: str,
    paths: ScenePathsLike,
    robot_profile: str | None = None,
    load_template_material: bool = False,
) -> list[str]:
    _, config_dir = _pipeline_paths(paths)
    command = [
        sys.executable,
        "-m",
        COMMANDS["config"]["module"],
        "--gym_project",
        f"gym_project/{paths.scene_id}/gym_export",
        "--output_dir",
        config_dir,
        "--task_name",
        SCENE_ID,
        "--task_description",
        task_text,
        *COMMANDS["config"]["base_args"],
    ]
    if profile := robot_profile_cli_value(robot_profile):
        command.extend(["--robot-profile", profile])
    if load_template_material:
        command.append("--load-template-material")
    return command


def build_run_agent_command(
    paths: ScenePathsLike,
    *,
    parallel_env: bool = False,
    robot_profile: str | None = None,
    supports_robot_profile: bool = False,
) -> list[str]:
    agent = COMMANDS["agent"]
    command = [
        sys.executable,
        "-m",
        agent["module"],
        "--task_name",
        SCENE_ID,
        "--gym_config",
        str(paths.fast_gym_config),
        "--agent_config",
        str(paths.agent_config),
        *agent["base_args"],
        "--num_envs",
        agent["parallel_num_envs"] if parallel_env else agent["single_num_envs"],
    ]
    if parallel_env:
        command.extend(agent["parallel_args"])
    if supports_robot_profile and (profile := robot_profile_cli_value(robot_profile)):
        command.extend(["--robot-profile", profile])
    return command
