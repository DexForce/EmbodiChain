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
import threading
import time
from pathlib import Path

from app_config import COMMANDS, PROCESS_STOP_TIMEOUT_S
from app_env import (
    EMBODICHAIN_ROOT,
    configure_direct_network_env,
    configure_simready_llm_env,
)
from app_state import PHASES

__all__ = [
    "build_pipeline_env",
    "build_run_agent_command",
    "detect_phase_from_files",
    "force_stop_all_child_processes",
    "read_process_output",
    "register_managed_process",
    "run_agent_cli_supports_robot_profile",
    "start_pipeline",
    "terminate_process_group",
    "update_phase_from_log",
]

_RUN_AGENT_SUPPORTS_ROBOT_PROFILE: bool | None = None
_managed_processes: dict[int, subprocess.Popen[str]] = {}
_managed_processes_lock = threading.Lock()
_shutdown_requested = False


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
    return register_managed_process(
        subprocess.Popen(
            command,
            cwd=EMBODICHAIN_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
    )


def build_pipeline_env() -> dict[str, str]:
    env = os.environ.copy()
    configure_direct_network_env(env)
    configure_simready_llm_env(env)
    return env


def register_managed_process(
    process: subprocess.Popen[str],
) -> subprocess.Popen[str]:
    """Register a UI-owned subprocess for application-shutdown cleanup.

    Processes must be registered immediately after they are created. If Gradio
    shutdown has already begun, the new process is stopped before this function
    returns so a callback cannot leave an orphan behind.
    """
    with _managed_processes_lock:
        if not _shutdown_requested:
            _managed_processes[process.pid] = process
            return process

    terminate_process_group(process)
    return process


def _unregister_managed_process(process: subprocess.Popen[str]) -> None:
    with _managed_processes_lock:
        _managed_processes.pop(process.pid, None)


def _child_process_ids(parent_pid: int) -> set[int]:
    """Return a snapshot of every descendant of ``parent_pid`` on POSIX."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()

    children_by_parent: dict[int, set[int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2 or not all(field.isdecimal() for field in fields):
            continue
        pid, ppid = (int(field) for field in fields)
        children_by_parent.setdefault(ppid, set()).add(pid)

    descendants: set[int] = set()
    pending = list(children_by_parent.get(parent_pid, set()))
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children_by_parent.get(pid, set()))
    return descendants


def _force_stop_process_ids(process_ids: set[int]) -> None:
    """Stop unregistered child PIDs, escalating from SIGTERM to SIGKILL."""
    process_ids.discard(os.getpid())
    for pid in process_ids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    deadline = time.monotonic() + PROCESS_STOP_TIMEOUT_S
    remaining = set(process_ids)
    while remaining and time.monotonic() < deadline:
        remaining = {pid for pid in remaining if _process_is_running(pid)}
        if remaining:
            time.sleep(0.1)

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        status = Path(f"/proc/{pid}/stat").read_text().rsplit(")", maxsplit=1)[1]
    except (FileNotFoundError, IndexError, PermissionError):
        return True
    return not status.lstrip().startswith("Z")


def force_stop_all_child_processes() -> None:
    """Force-stop every subprocess owned by the Gradio application.

    Registered processes are stopped by their isolated process groups, which
    also stops their descendants. A second descendant scan catches short-lived
    or legacy subprocesses that were not registered explicitly.
    """
    global _shutdown_requested
    with _managed_processes_lock:
        _shutdown_requested = True
        managed_processes = tuple(_managed_processes.values())
    child_process_ids = _child_process_ids(os.getpid())

    for process in managed_processes:
        terminate_process_group(process)

    _force_stop_process_ids(child_process_ids)


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
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
    finally:
        _unregister_managed_process(process)


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
    log_path: Path | None = None,
) -> None:
    """Forward merged subprocess output to the UI queue and an optional log."""
    if process.stdout is None:
        return
    log_file = log_path.open("a", encoding="utf-8") if log_path is not None else None
    try:
        for line in process.stdout:
            output_queue.put(line.rstrip())
            if log_file is not None:
                log_file.write(line)
                if not line.endswith("\n"):
                    log_file.write("\n")
                log_file.flush()
    finally:
        if log_file is not None:
            log_file.close()
