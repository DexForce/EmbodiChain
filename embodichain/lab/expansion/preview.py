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

"""Owned launch, wait and stop operations for native result-preview services."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

from ._task_program_config import load_mapping
from embodichain.data_pipeline.recording.media import _atomic_json as atomic_json
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "launch_expansion_preview",
    "preview_expansion_results",
    "stop_expansion_preview",
]


def _process_command(pid: int) -> bytes:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return b""


def _inputs(paths: list[Path]) -> list[Path]:
    result = []
    for supplied in paths:
        path = supplied.expanduser().resolve()
        if (path / "job.json").is_file() or (path / "preview-plan.json").is_file():
            result.append(path)
        elif (path / "expansion.json").is_file():
            data = load_mapping(path / "expansion.json")
            result.extend(
                Path(row["directory"]).resolve()
                for row in data.get("operators", {}).values()
                if row.get("committed_episodes", 0) > 0
            )
        else:
            raise ValueError(
                f"Preview input must be an aggregate result or native job: {path}"
            )
    if not result or len(set(result)) != len(result):
        raise ValueError(
            "Preview requires distinct job outputs with confirmed episodes"
        )
    return result


def _recorded_eef(inputs: list[Path]) -> str | None:
    """Resolve the first measured motion endpoint, without robot-name routing."""
    from embodichain.lab.expansion import ArtifactEpisodeSink, load_expansion_episode

    for path in inputs:
        if not (path / "job.json").is_file():
            continue
        job = load_mapping(path / "job.json")
        robot = job.get("resolved_deployment", {}).get("robot", {})
        solvers = robot.get("solver_cfg", {})
        ends = {
            item.get("end_link_name")
            for item in solvers.values()
            if item.get("end_link_name")
        }
        if len(ends) == 1:
            return next(iter(ends))
        receipts = ArtifactEpisodeSink(path).inspect()
        if not receipts:
            continue
        episode = load_expansion_episode(receipts[0].storage_id)
        for record in episode.metadata.get("expansion", ()):
            joints = set(record.get("controlled_joint_names", ()))
            matches = [
                cfg.get("end_link_name")
                for cfg in solvers.values()
                if joints
                and joints == set(cfg.get("joint_names") or ())
                and cfg.get("end_link_name")
            ]
            if len(set(matches)) == 1:
                return matches[0]
        for segment in episode.metadata.get("program_segments", ()):
            runtime = segment.get(
                "runtime", segment.get("metadata", {}).get("runtime", {})
            )
            for call in runtime.get("calls", ()):
                for endpoint in call.get("resolved_core_policy", {}).get(
                    "endpoints", ()
                ):
                    if endpoint.get("endpoint_id") == "motion":
                        end = solvers.get(endpoint.get("target_id"), {}).get(
                            "end_link_name"
                        )
                        if end:
                            return end
    return None


def launch_expansion_preview(
    paths: list[Path],
    *,
    output: Path | None = None,
    port: int = 0,
    gpu: int = 0,
    duration: float = 900,
    open_browser: bool = False,
    eef_link: str | None = None,
    startup_timeout: float = 240,
    max_episodes: int = 32,
) -> dict:
    """Start a bounded native preview service after collection/export completes.

    Logs/status are separate from immutable run artifacts. Inputs are read-only;
    no trajectory is recollected, substituted or credited as another success.

    Args:
        paths: Native job, aggregate result or preview-plan directories.
        output: New directory for owned service status and logs.
        port: Loopback port; zero allocates an available port.
        gpu: Visible GPU index used by the read-only renderer.
        duration: Service seconds after readiness; zero waits for interruption.
        open_browser: Open a browser after native rendering is ready.
        eef_link: Optional endpoint; recorded motion supplies it otherwise.
        startup_timeout: Bounded wait for the first complete frame.
        max_episodes: Maximum recorded episodes to preload, between one and 64.
    Returns:
        Owned service status, command identity and ready URL. Linux PID ownership
        is checked before the stop/wait APIs signal the process.
    """
    paths = [Path(path) for path in paths]
    if (
        type(port) is not int
        or not 0 <= port <= 65535
        or type(gpu) is not int
        or gpu < 0
    ):
        raise ValueError("Invalid preview port or GPU")
    if (
        not math.isfinite(duration)
        or duration < 0
        or not math.isfinite(startup_timeout)
        or startup_timeout <= 0
    ):
        raise ValueError("Invalid preview duration/startup timeout")
    if type(max_episodes) is not int or not 1 <= max_episodes <= 64:
        raise ValueError("Preview max_episodes must be in [1,64]")
    inputs = _inputs(paths)
    output = (
        output or (paths[0].resolve().parent / ".previews" / uuid4().hex)
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists():
        raise FileExistsError(f"Preview status already exists: {status_path}")
    command = [sys.executable, "-m", "embodichain", "expansion", "preview"]
    for path in inputs:
        command.extend(["--input", str(path)])
    command.extend(
        [
            "--port",
            str(port),
            "--gpu",
            str(gpu),
            "--duration",
            str(duration),
            "--max-episodes",
            str(max_episodes),
        ]
    )
    end = eef_link or _recorded_eef(inputs)
    if end:
        command.extend(["--eef-link", end])
    if open_browser:
        command.append("--open")
    log_path = output / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    status = {
        "state": "starting",
        "pid": process.pid,
        "command": command,
        "inputs": [str(p) for p in inputs],
        "started_at": utc_now(),
        "status_file": str(status_path),
        "log": str(log_path),
    }
    atomic_json(status_path, status)
    deadline = time.monotonic() + startup_timeout
    try:
        while time.monotonic() < deadline:
            for line in log_path.read_text(errors="replace").splitlines():
                try:
                    ready = json.loads(line)
                except ValueError:
                    continue
                if isinstance(ready, dict) and ready.get("state") == "ready":
                    status.update(ready, state="ready", eef_link=end)
                    atomic_json(status_path, status)
                    return status
            if process.poll() is not None:
                raise RuntimeError(
                    f"Native preview exited {process.returncode}: {log_path.read_text(errors='replace')[-3000:]}"
                )
            time.sleep(0.1)
        raise TimeoutError(f"Native preview startup timed out; see {log_path}")
    except BaseException:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        atomic_json(
            status_path, {**status, "state": "failed", "returncode": process.returncode}
        )
        raise


def stop_expansion_preview(path: str | Path) -> dict[str, object]:
    """Stop only the native service whose recorded command still owns its PID.

    Args:
        path: Service status JSON or its directory.
    Returns:
        Closed status after the owned process stops; changed ownership rejects.
    """
    path = Path(path)
    status_path = path / "status.json" if path.is_dir() else path
    status = load_mapping(status_path)
    pid = status["pid"]
    command = _process_command(pid)
    if command:
        actual = command.rstrip(b"\0").split(b"\0")
        if actual != [os.fsencode(arg) for arg in status["command"]]:
            raise ValueError("Preview PID is no longer owned by the recorded command")
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 30
        while _process_command(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _process_command(pid):
            raise TimeoutError("Preview has not stopped; inspect its log")
    status.update(state="closed", closed_at=utc_now())
    atomic_json(status_path, status)
    return status


def preview_expansion_results(
    paths: list[Path], **options: object
) -> dict[str, object]:
    """Run the native browser viewer until its duration ends or Ctrl+C stops it.

    Args:
        paths: Confirmed native job or aggregate result directories.
        **options: Launch options accepted by :func:`launch_expansion_preview`.
    Returns:
        Final service status after bounded playback or interruption.
    """
    status = launch_expansion_preview(paths, **options)
    sys.stdout.write(json.dumps(status, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    try:
        expected = [os.fsencode(arg) for arg in status["command"]]
        while True:
            command = _process_command(status["pid"])
            if not command or command.rstrip(b"\0").split(b"\0") != expected:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        return stop_expansion_preview(Path(status["status_file"]))
    try:
        os.waitpid(status["pid"], os.WNOHANG)
    except ChildProcessError:
        pass
    status.update(state="closed", closed_at=utc_now())
    atomic_json(Path(status["status_file"]), status)
    return status
