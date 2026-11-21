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

"""Invoke Blender without mixing its Python/NumPy ABI with the host framework."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from .env import load_gen_sim_env

__all__ = []

_WORKER = Path(__file__).with_name("_blender_worker.py")


def _worker_environment(executable: Path) -> dict[str, str]:
    environment = dict(os.environ)
    prefixes = {
        str(Path(value).resolve())
        for value in (
            sys.prefix,
            environment.get("CONDA_PREFIX"),
            environment.get("VIRTUAL_ENV"),
        )
        if value
    }
    # A system interpreter's /usr prefix also contains OS/driver libraries.
    # Only dedicated Python prefixes belong to the business environment.
    prefixes.difference_update({"/", "/usr", "/usr/local"})
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    ):
        environment.pop(name, None)
    # Keep driver/system libraries but do not load the business environment's
    # libpython, DexSim, OpenMP or CUDA wheels into Blender's interpreter.
    for name in ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        separator = " " if name == "LD_PRELOAD" else os.pathsep
        paths = environment.get(name, "").split(separator)
        environment[name] = separator.join(
            path
            for path in paths
            if path
            and not any(
                path == prefix or path.startswith(prefix + os.sep)
                for prefix in prefixes
            )
        )
    environment["PATH"] = (
        str(executable.parent) + os.pathsep + environment.get("PATH", "")
    )
    return environment


def _run_blender_job(
    operation: str, arguments: dict[str, Any], *, timeout: float = 600.0
) -> Any:
    """Run a file-based Blender job and reap it on failure or cancellation.

    ``EMBODICHAIN_BLENDER_EXECUTABLE`` selects the Blender application, whose
    startup initializes Eevee's graphics context. Otherwise use the isolated
    interpreter selected by ``EMBODICHAIN_BLENDER_PYTHON``.
    """
    settings = dict(os.environ)
    load_gen_sim_env(settings)
    blender = settings.get("EMBODICHAIN_BLENDER_EXECUTABLE", "").strip()
    executable = (
        Path(blender or settings.get("EMBODICHAIN_BLENDER_PYTHON", sys.executable))
        .expanduser()
        .absolute()
    )
    if not executable.is_file():
        if blender:
            raise RuntimeError(
                f"Blender executable does not exist: {executable}. "
                "Set EMBODICHAIN_BLENDER_EXECUTABLE to the Blender application."
            )
        raise RuntimeError(
            f"Blender Python does not exist: {executable}. "
            "Set EMBODICHAIN_BLENDER_PYTHON to the isolated Blender interpreter."
        )
    with tempfile.TemporaryDirectory(prefix="embodichain-blender-") as directory:
        root = Path(directory)
        request, response, log = (
            root / "request.json",
            root / "response.json",
            root / "worker.log",
        )
        request.write_text(
            json.dumps({"operation": operation, "arguments": arguments}),
            encoding="utf-8",
        )
        if blender:
            command = [
                str(executable),
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "1",
                "--python",
                str(_WORKER),
                "--",
                str(request),
                str(response),
            ]
        else:
            command = [
                str(executable),
                "-I",
                str(_WORKER),
                str(request),
                str(response),
            ]
        with log.open("w", encoding="utf-8") as output:
            # Inherit the pipeline's process group so Gradio cancellation also
            # reaches Blender. Never start an unmanaged background worker.
            process = subprocess.Popen(
                command,
                env=_worker_environment(executable),
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            try:
                process.wait(timeout=timeout)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if process.returncode or not response.is_file():
            detail = log.read_text(encoding="utf-8", errors="replace")[-8192:]
            raise RuntimeError(
                f"Blender operation {operation!r} failed ({process.returncode}):\n{detail}"
            )
        result = json.loads(response.read_text(encoding="utf-8"))
        return result["result"]
