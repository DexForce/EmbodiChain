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

"""Internal host lifecycle and chunk transport behind the ``session`` CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from .catalog import write_json
from .session import _environment, _run_id, _stop_tree

__all__ = []


class EpisodeHost:
    """Own at most one live physics episode for a coding session."""

    def __init__(self, root: Path, deadline: float) -> None:
        self.root = root.resolve()
        self.deadline = deadline
        self.process = None
        self.attempt = None
        self.streams = []
        self.started = 0.0
        self.identity = {}
        self.sequence = 0

    def _wait(self, path: Path, timeout: float) -> dict:
        deadline = min(self.deadline, time.time() + timeout)
        while not path.exists():
            if self.process.poll() is not None:
                returncode = self.process.returncode
                self.close(force=True)
                raise RuntimeError(
                    f"Episode worker exited ({returncode}); inspect {self.attempt}"
                )
            if time.time() >= deadline:
                self.close(force=True)
                raise TimeoutError(
                    f"Episode command timed out; artifacts retained at {self.attempt}"
                )
            time.sleep(0.05)
        return json.loads(path.read_text())

    def start(self, config: Path, timeout: float) -> dict:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError(
                "Close the current episode before starting another; no implicit reset"
            )
        if self.process is not None:
            self.close(force=True)
        manifest = json.loads((self.root / "run.json").read_text())
        self.attempt = self.root / "attempts" / f"episode_{_run_id()}"
        self.attempt.mkdir(parents=True)
        shutil.copy2(config, self.attempt / "lab.json")
        shutil.copytree(
            Path(__file__).parent,
            self.attempt / "runtime_source",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        write_json(
            self.attempt / "runtime_hashes.json",
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (self.attempt / "runtime_source").glob("*.py")
            },
        )
        write_json(
            self.attempt / "experiment.json",
            {
                "mode": manifest.get("mode", "B"),
                "approved_asset_changes": manifest.get("approved_asset_changes", []),
                "execution_mode": "segmented",
            },
        )
        command = [
            manifest["python"],
            "-B",
            "-m",
            "embodichain.gen_sim.agent_lab.episode_worker",
            "--config",
            str(self.attempt / "lab.json"),
            "--output",
            str(self.attempt),
            "--owner-pid",
            str(psutil.Process().pid),
            "--owner-created-at",
            str(psutil.Process().create_time()),
            "--deadline",
            str(self.deadline),
        ]
        self.streams = [
            (self.attempt / name).open("w") for name in ("stdout.log", "stderr.log")
        ]
        self.started = time.monotonic()
        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.root / "workspace",
                env=_environment(Path(manifest["repo"])),
                stdout=self.streams[0],
                stderr=self.streams[1],
                start_new_session=True,
            )
        except OSError:
            for stream in self.streams:
                stream.close()
            raise
        self.identity = {
            "pid": self.process.pid,
            "created_at": psutil.Process(self.process.pid).create_time(),
            "started_at": time.time(),
            "command": command,
        }
        write_json(self.attempt / "process.json", self.identity)
        self.sequence = 0
        return {
            "attempt": str(self.attempt),
            **self._wait(self.attempt / "ready.json", timeout),
        }

    def command(
        self, operation: str, *, script: Path | None = None, timeout: float = 300
    ) -> dict:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("No live episode; use session start first")
        self.sequence += 1
        job = {"operation": operation}
        if script is not None:
            code = self.attempt / "chunks" / f"{self.sequence:04d}" / "code"

            def ignore_outputs(directory: str, names: list[str]) -> list[str]:
                if Path(directory).resolve() in {
                    self.root / "attempts",
                    self.root / "codex",
                    self.root / "requests",
                }:
                    return names
                return [
                    name
                    for name in names
                    if name in {".git", "__pycache__"}
                    or code.is_relative_to((Path(directory) / name).resolve())
                ]

            shutil.copytree(
                script.parent,
                code,
                ignore=ignore_outputs,
            )
            job["script"] = str(code / script.name)
            write_json(
                code.parent / "inputs.json",
                {
                    "script": script.name,
                    "sha256": {
                        str(p.relative_to(code)): hashlib.sha256(
                            p.read_bytes()
                        ).hexdigest()
                        for p in code.rglob("*")
                        if p.is_file()
                    },
                },
            )
        write_json(
            self.attempt / "inputs.json",
            {
                "execution_mode": "segmented",
                "files_sha256": {
                    str(p.relative_to(self.attempt)): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in [
                        self.attempt / "lab.json",
                        *sorted((self.attempt / "chunks").rglob("*")),
                    ]
                    if p.is_file()
                },
            },
        )
        directory = self.attempt / "commands"
        directory.mkdir(exist_ok=True)
        request = directory / f"{self.sequence:04d}.json"
        staging = request.with_suffix(".tmp")
        write_json(staging, job)
        staging.replace(request)
        response = self._wait(request.with_suffix(".reply.json"), timeout)
        return {"attempt": str(self.attempt), **response}

    def close(self, *, force: bool = False) -> dict:
        if self.process is None:
            return {"status": "no_episode"}
        process = self.process
        if process.poll() is None:
            if force or time.time() >= self.deadline:
                _stop_tree(process)
            else:
                try:
                    self.command("close", timeout=15)
                    process.wait(timeout=20)
                except (subprocess.TimeoutExpired, TimeoutError, RuntimeError):
                    force = True
                    if process.poll() is None:
                        _stop_tree(process)
        elapsed = time.monotonic() - self.started
        record = {
            **self.identity,
            "wall_seconds": elapsed,
            "ended_at": self.identity["started_at"] + elapsed,
            "returncode": process.returncode,
            "interrupted": force,
            "timed_out": time.time() >= self.deadline,
        }
        write_json(self.attempt / "process.json", record)
        for stream in self.streams:
            stream.close()
        self.process = None
        result_path = self.attempt / "result.json"
        result = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {
                "execution_completed": False,
                "error": "Native worker exited without a result; inspect retained artifacts",
            }
        )
        report = {"attempt": str(self.attempt), "process": record, "worker": result}
        write_json(self.root / "latest_attempt.json", report)
        return report

    def service(self, request: dict) -> dict:
        operation = request["episode_operation"]
        if operation == "start":
            return self.start(Path(request["config"]), request["timeout"])
        if operation == "close":
            return self.close()
        return self.command(
            operation,
            script=Path(request["script"]) if request.get("script") else None,
            timeout=request["timeout"],
        )


def request_episode(
    root: Path,
    operation: str,
    *,
    config: Path | None = None,
    script: Path | None = None,
    timeout: float = 300,
) -> dict:
    """Submit lifecycle or arbitrary-Python work to the existing host broker."""
    from ._launch import _check_host

    host = _check_host(root)
    queue = root / "requests"
    queue.mkdir(exist_ok=True)
    request = queue / f"{_run_id()}.json"
    staging = request.with_suffix(".tmp")
    write_json(
        staging,
        {
            "episode_operation": operation,
            "config": str(config) if config else None,
            "script": str(script) if script else None,
            "timeout": timeout,
        },
    )
    staging.replace(request)
    reply = request.with_suffix(".reply.json")
    deadline = min(host["deadline"] + 20, time.time() + timeout + 30)
    while not reply.exists():
        if time.time() >= deadline:
            raise TimeoutError(f"Episode host did not reply: {request}")
        time.sleep(0.1)
    return json.loads(reply.read_text())
