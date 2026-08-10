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

"""Subprocess execution and lifecycle management for the engine workspace."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from app_config import COMMANDS, PROCESS_STOP_TIMEOUT_S
from app_env import (
    EMBODICHAIN_ROOT,
    configure_direct_network_env,
    configure_simready_llm_env,
)

__all__ = [
    "SessionProcessRegistry",
    "build_codex_env",
    "build_pipeline_env",
    "build_run_agent_command",
    "force_stop_all_child_processes",
    "get_request_session_id",
    "read_process_output",
    "register_managed_process",
    "run_agent_cli_supports_robot_profile",
    "start_pipeline",
    "terminate_process_group",
    "redact_sensitive_text",
]

_RUN_AGENT_SUPPORTS_ROBOT_PROFILE: bool | None = None
_managed_processes: dict[int, subprocess.Popen[str]] = {}
_managed_processes_lock = threading.Lock()
_shutdown_requested = False
_CODEX_ENV_ALLOWLIST = {
    "CODEX_HOME",
    "COLORTERM",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
}
_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


def get_request_session_id(request: object) -> str:
    """Return the stable session identifier supplied by Gradio.

    Args:
        request: Gradio request object injected into an event callback.

    Returns:
        The non-empty Gradio session hash.

    Raises:
        RuntimeError: If the callback was invoked without a session hash.
    """
    session_id = getattr(request, "session_hash", None)
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("This operation requires an active Gradio session.")
    return session_id


class SessionProcessRegistry:
    """Track one replaceable subprocess for each Gradio session.

    A registry instance belongs to one workflow, such as SimReady or Articraft.
    Resetting one session can therefore never invalidate or terminate another
    session's run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, tuple[str, subprocess.Popen[str] | None]] = {}

    def begin(self, session_id: str) -> str:
        """Start a new logical run for one session.

        Args:
            session_id: Stable Gradio session identifier.

        Returns:
            A new ownership token for the run.
        """
        token = uuid.uuid4().hex
        with self._lock:
            previous = self._runs.get(session_id)
            self._runs[session_id] = (token, None)
        if previous is not None and previous[1] is not None:
            terminate_process_group(previous[1])
        return token

    def is_active(
        self,
        session_id: str,
        token: str,
        process: subprocess.Popen[str] | None = None,
    ) -> bool:
        """Return whether a run still owns its session slot.

        Args:
            session_id: Stable Gradio session identifier.
            token: Token returned by :meth:`begin`.
            process: Optional process that must also match the registered child.

        Returns:
            ``True`` when the token and optional process still match.
        """
        with self._lock:
            current = self._runs.get(session_id)
            return (
                current is not None
                and current[0] == token
                and (process is None or current[1] is process)
            )

    def attach(
        self,
        session_id: str,
        token: str,
        process: subprocess.Popen[str],
    ) -> bool:
        """Attach a subprocess to an active session run.

        Args:
            session_id: Stable Gradio session identifier.
            token: Token returned by :meth:`begin`.
            process: Managed subprocess started for the run.

        Returns:
            ``True`` if the token still owns the session slot.
        """
        with self._lock:
            current = self._runs.get(session_id)
            if current is None or current[0] != token:
                return False
            self._runs[session_id] = (token, process)
            return True

    def finish(
        self,
        session_id: str,
        token: str,
        process: subprocess.Popen[str],
    ) -> None:
        """Clear a finished subprocess while keeping its logical run active.

        Args:
            session_id: Stable Gradio session identifier.
            token: Token returned by :meth:`begin`.
            process: Subprocess that has finished.
        """
        with self._lock:
            current = self._runs.get(session_id)
            if current == (token, process):
                self._runs[session_id] = (token, None)

    def reset(self, session_id: str) -> None:
        """Invalidate and terminate only one session's process.

        Args:
            session_id: Stable Gradio session identifier.
        """
        with self._lock:
            current = self._runs.pop(session_id, None)
        if current is not None and current[1] is not None:
            terminate_process_group(current[1])

    def reset_all(self) -> None:
        """Invalidate and terminate every process tracked by this registry."""
        with self._lock:
            runs = tuple(self._runs.values())
            self._runs.clear()
        for _token, process in runs:
            if process is not None:
                terminate_process_group(process)


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


def build_run_agent_command(*, robot_profile: str | None = None) -> list[str]:
    """Build the Action-engine command for the existing current scene."""
    from app_commands import build_run_agent_command as build_command

    return build_command(
        robot_profile=robot_profile,
        supports_robot_profile=run_agent_cli_supports_robot_profile(),
    )


def start_pipeline(
    command: list[str], *, use_simready_llm: bool = False
) -> subprocess.Popen[str]:
    """Start a managed pipeline subprocess with its scoped dotenv settings.

    Args:
        command: Command and arguments to execute.
        use_simready_llm: Whether to map the dotenv ``SIMREADY_OPENAI_*`` values
            to the upstream SimReady CLI's ``OPENAI_*`` variable names.

    Returns:
        The registered subprocess.
    """
    env = build_pipeline_env(use_simready_llm=use_simready_llm)
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


def build_pipeline_env(*, use_simready_llm: bool = False) -> dict[str, str]:
    """Build a child environment from the shared GenSim dotenv configuration.

    Args:
        use_simready_llm: Whether to apply the SimReady-specific LLM mapping.

    Returns:
        A copy of the loaded process environment configured for the child.
    """
    env = os.environ.copy()
    configure_direct_network_env(env)
    if use_simready_llm:
        configure_simready_llm_env(env)
    return env


def build_codex_env() -> dict[str, str]:
    """Build a credential-minimized environment for user-directed Codex runs.

    The Codex CLI may still use its own login state through ``CODEX_HOME`` or
    the normal user configuration directory, but GenSim service credentials
    and dotenv-specific settings are not inherited by the command sandbox.

    Returns:
        An allowlisted child-process environment.

    .. attention::
        Deployments that authenticate Codex exclusively through
        ``OPENAI_API_KEY`` must use ``codex login`` or another isolated Codex
        credential store instead. Passing the server key to a user-directed
        process would recreate the disclosure boundary this function removes.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if key in _CODEX_ENV_ALLOWLIST and value
    }


def redact_sensitive_text(text: str) -> str:
    """Replace known environment credential values in UI-bound output.

    Args:
        text: Subprocess output or a final message that may contain credentials.

    Returns:
        Text with non-trivial sensitive environment values replaced.
    """
    redacted = text
    for key, value in os.environ.items():
        upper_key = key.upper()
        if (
            value
            and len(value) >= 4
            and any(marker in upper_key for marker in _SENSITIVE_ENV_MARKERS)
        ):
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


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


def read_process_output(
    process: subprocess.Popen[str],
    output_queue: queue.Queue[str],
    log_path: Path | None = None,
    *,
    redact_sensitive: bool = False,
) -> None:
    """Forward merged subprocess output to the UI queue and an optional log.

    Args:
        process: Child process whose merged stdout should be consumed.
        output_queue: Destination for individual output lines.
        log_path: Optional file receiving the same output.
        redact_sensitive: Whether to redact known environment credentials before
            forwarding or persisting each line.
    """
    if process.stdout is None:
        return
    log_file = log_path.open("a", encoding="utf-8") if log_path is not None else None
    try:
        for line in process.stdout:
            output_line = redact_sensitive_text(line) if redact_sensitive else line
            output_queue.put(output_line.rstrip())
            if log_file is not None:
                log_file.write(output_line)
                if not output_line.endswith("\n"):
                    log_file.write("\n")
                log_file.flush()
    finally:
        if log_file is not None:
            log_file.close()
