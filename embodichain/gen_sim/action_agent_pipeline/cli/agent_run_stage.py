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

import os
from pathlib import Path
import shlex
import subprocess
import sys

__all__ = ["run_agent_command"]


def run_agent_command(
    *,
    task_name: str,
    gym_config: Path,
    agent_config: Path,
    regenerate: bool,
) -> int:
    command = [
        sys.executable,
        "-m",
        "embodichain.gen_sim.action_agent_pipeline.cli.run_agent",
        "--task_name",
        task_name,
        "--gym_config",
        str(gym_config),
        "--agent_config",
        str(agent_config),
    ]
    if regenerate:
        command.append("--regenerate")

    env = os.environ.copy()
    if env.get("EMBODICHAIN_LLM_USAGE_PATH"):
        env["EMBODICHAIN_LLM_USAGE_PROCESS"] = "run_agent"

    print("Running task:")
    print(shlex.join(command), flush=True)
    return subprocess.run(command, check=False, env=env).returncode
