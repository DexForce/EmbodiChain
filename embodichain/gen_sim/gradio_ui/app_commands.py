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

"""CLI command builder for the Action engine."""

from __future__ import annotations

import sys

from app_config import (
    AGENT_CONFIG,
    COMMANDS,
    FAST_GYM_CONFIG,
    ROBOT_PROFILE_FRANKA,
    ROBOT_PROFILE_UR5,
    ROBOT_PROFILE_UR10,
    SCENE_ID,
)

__all__ = ["build_run_agent_command"]


def _robot_profile_cli_value(robot_profile: str | None) -> str | None:
    return {
        ROBOT_PROFILE_FRANKA: "franka",
        ROBOT_PROFILE_UR5: "dual_ur5",
        ROBOT_PROFILE_UR10: "dual_ur10",
    }.get(robot_profile)


def build_run_agent_command(
    *,
    robot_profile: str | None = None,
    supports_robot_profile: bool = False,
) -> list[str]:
    """Build the DexSim command for the existing ``current`` Gym scene."""
    agent = COMMANDS["agent"]
    command = [
        sys.executable,
        "-m",
        agent["module"],
        "--task_name",
        SCENE_ID,
        "--gym_config",
        str(FAST_GYM_CONFIG),
        "--agent_config",
        str(AGENT_CONFIG),
        *agent["base_args"],
        "--num_envs",
        agent["single_num_envs"],
    ]
    if supports_robot_profile and (profile := _robot_profile_cli_value(robot_profile)):
        command.extend(["--robot-profile", profile])
    return command
