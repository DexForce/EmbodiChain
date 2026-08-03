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

"""Pipeline subprocess execution and progress detection."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import time
from pathlib import Path

from app_config import *  # noqa: F403 - process settings are central configuration.
from app_state import PHASES

_RUN_AGENT_SUPPORTS_ROBOT_PROFILE: bool | None = None


def run_agent_cli_supports_robot_profile() -> bool:
    global _RUN_AGENT_SUPPORTS_ROBOT_PROFILE
    if _RUN_AGENT_SUPPORTS_ROBOT_PROFILE is not None:
        return _RUN_AGENT_SUPPORTS_ROBOT_PROFILE
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                COMMANDS["agent"]["module"],
                *COMMANDS["agent"]["help_args"],
            ],
            cwd=EMBODICHAIN_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=build_pipeline_env(),
            timeout=20,
        )
        help_text = (result.stdout or "").lower()
        _RUN_AGENT_SUPPORTS_ROBOT_PROFILE = "--robot-profile" in help_text
    except Exception:
        _RUN_AGENT_SUPPORTS_ROBOT_PROFILE = False
    return _RUN_AGENT_SUPPORTS_ROBOT_PROFILE


def build_run_agent_command(
    paths: ScenePaths, *, parallel_env: bool = False, robot_profile: str | None = None
) -> list[str]:
    from app_commands import build_run_agent_command as build_command

    return build_command(
        paths,
        parallel_env=parallel_env,
        robot_profile=robot_profile,
        supports_robot_profile=run_agent_cli_supports_robot_profile(),
    )


def start_pipeline(command: list[str]) -> subprocess.Popen[str]:
    env = build_pipeline_env()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        command,
        cwd=EMBODICHAIN_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=env,
    )


def build_pipeline_env() -> dict[str, str]:
    env = os.environ.copy()
    configure_direct_network_env(env)
    configure_simready_llm_env(env)
    return env


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        process.terminate()

    deadline = time.monotonic() + PROCESS_STOP_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        process.kill()


def detect_phase_from_files(current_key: str, paths: ScenePaths) -> str:
    candidates = [
        ("scene_intake", paths.prompt_root / "scene_intake" / "result.json"),
        ("relations", paths.prompt_root / "image_segments" / "result.json"),
        (
            "relations",
            paths.prompt_root / "image_spatial_relations" / "result.json",
        ),
        ("gym_export", paths.prompt_root / "gym_export" / "gym_config.json"),
        ("config", paths.fast_gym_config),
        ("preview", paths.gradio_scene_glb),
    ]
    best_key = current_key
    best_progress = PHASES.get(best_key, PHASES["idle"]).progress

    if any(paths.prompt_root.glob("unified_scene_gen/**/*.glb")):
        best_key, best_progress = _choose_later_phase(
            best_key,
            best_progress,
            "asset_generation",
        )
    for phase_key, marker in candidates:
        if marker.exists():
            best_key, best_progress = _choose_later_phase(
                best_key,
                best_progress,
                phase_key,
            )
    return best_key


def _choose_later_phase(
    current_key: str,
    current_progress: int,
    candidate_key: str,
) -> tuple[str, int]:
    candidate_progress = PHASES[candidate_key].progress
    if candidate_progress > current_progress:
        return candidate_key, candidate_progress
    return current_key, current_progress


def update_phase_from_log(line: str, current_key: str) -> str:
    text = line.lower()
    mapping = [
        ("scene_intake", "scene_intake"),
        ("image_segments", "relations"),
        ("image_spatial_relations", "relations"),
        ("unified_scene_gen", "asset_generation"),
        ("glb", "asset_generation"),
        ("gym_export", "gym_export"),
        ("generated gym config", "config"),
        ("fast_gym_config", "config"),
    ]
    best_key = current_key
    best_progress = PHASES.get(best_key, PHASES["idle"]).progress
    for needle, phase_key in mapping:
        if needle in text:
            best_key, best_progress = _choose_later_phase(
                best_key,
                best_progress,
                phase_key,
            )
    return best_key


def read_process_output(
    process: subprocess.Popen[str],
    output_queue: queue.Queue[str],
) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        output_queue.put(line.rstrip())
