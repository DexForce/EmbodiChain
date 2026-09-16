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

"""Isolate the complete generation pipeline from the native viewer process."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from scripts.tools.assemble._protocol import load_config
from scripts.tools.mug_rack_pose._json_io import read_json, write_json


def _stop_tree(process: subprocess.Popen) -> None:
    # Codex and Blender each own process groups; killing only the job's group
    # would leave these grandchildren running after a timeout or Ctrl+C.
    import psutil

    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(descendants, timeout=2)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait()


def wait_process(
    command: list[str], directory: Path, timeout: float, pump: Callable[[], None]
) -> int:
    """Wait for a child while keeping the viewer responsive and streaming progress.

    Args:
        command: Executable and arguments; never passed through a shell.
        directory: Directory receiving the child's combined log.
        timeout: Whole-job wall-clock limit in seconds.
        pump: Main-process viewer event/update callback, called every 50 ms.

    Returns:
        Child exit code. Timeouts and interrupts stop all descendant processes.
    """
    log_path = directory / "generation.log"
    with log_path.open("w") as output, log_path.open(errors="replace") as reader:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        started = time.monotonic()
        next_progress = started + 15
        try:
            while process.poll() is None:
                text = reader.read(65536)
                if text:
                    print(text, end="", flush=True)
                pump()
                now = time.monotonic()
                if now >= next_progress:
                    print(
                        f"[assemble] Generation subprocess is running ({now - started:.0f}s elapsed).",
                        flush=True,
                    )
                    next_progress = now + 15
                if now - started > timeout:
                    raise TimeoutError(
                        f"Generation exceeded {timeout:g} seconds; inspect {log_path}"
                    )
                time.sleep(0.05)
            print(reader.read(), end="", flush=True)
            return process.returncode
        except BaseException:
            _stop_tree(process)
            raise


def run_generation(
    config_path: Path, cycle: int, timeout: float, pump: Callable[[], None]
) -> dict:
    """Generate a new result in a fresh interpreter without touching the old scene.

    Args:
        config_path: User configuration, reread before each run.
        cycle: Generation sequence number.
        timeout: Maximum duration of the entire generation subprocess.
        pump: Callback keeping the existing native scene active during generation.

    Returns:
        Verified result from a unique handoff file, or an explicit unsuccessful job.
    """
    config = load_config(config_path)
    jobs = Path(config["output_dir"]) / "preview_jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"cycle_{cycle:03d}_", dir=jobs))
    result_path = directory / "result.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("generate.py")),
        "--config",
        str(config_path.resolve()),
        "--cycle",
        str(cycle),
        "--result-file",
        str(result_path),
    ]
    write_json(directory / "command.json", {"argv": command})
    print(
        f"[assemble] Generating in a subprocess; the current scene stays visible. Log: {directory / 'generation.log'}",
        flush=True,
    )
    try:
        code = wait_process(command, directory, timeout, pump)
        if result_path.is_file():
            result = read_json(result_path)
            if result.get("success") is False and result.get("status", "failed") in (
                "failed",
                "interrupted",
            ):
                return result
            if (
                code == 0
                and result.get("success") is True
                and result.get("schema") == "codex-assemble/v1"
                and isinstance(result.get("validation"), dict)
                and result["validation"].get("accepted") is True
            ):
                return result
        raise RuntimeError(
            f"Generation process exited {code} without a usable result; inspect {directory / 'generation.log'}"
        )
    except (Exception, KeyboardInterrupt) as error:
        try:
            snapshot = read_json(result_path) if result_path.is_file() else {}
        except (OSError, ValueError):
            snapshot = {}
        result = snapshot | {
            "success": False,
            "status": (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            ),
            "T_base_assemble": None,
            "validation": None,
            "run_directory": snapshot.get("run_directory", str(directory)),
            "cycle": cycle,
            "reason": f"{type(error).__name__}: {error}",
        }
        write_json(result_path, result)
        if snapshot.get("run_directory"):
            write_json(Path(snapshot["run_directory"]) / "result.json", result)
            latest = Path(config["output_dir"]) / "latest.json"
            try:
                if read_json(latest).get("run_directory") == snapshot["run_directory"]:
                    write_json(latest, result)
            except (OSError, ValueError):
                pass
        if isinstance(error, KeyboardInterrupt):
            raise
        return result
