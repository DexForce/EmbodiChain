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

"""Standalone launch ownership; the parent conversation is not a runtime service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import psutil

from .catalog import write_json
from .usage import UsageMeter
from .session import (
    _environment,
    _process,
    _run_id,
    _serve_requests,
    execute_script,
)

_BOOTSTRAP_PROMPT = (
    "Read START.md in this workspace and carry out the objective recorded there and in environment.json. "
    "Before handing off, read ../usage.json and reflect cumulative elapsed time and token usage, "
    "including source and completeness. Never guess or overwrite host-measured statistics."
)


def _codex_command(
    root: Path, output: Path, *, batch: bool, model: str, effort: str
) -> list[str]:
    command = ["codex", "-a", "never"]
    if batch:
        command += [
            "exec",
            "--json",
            "--color",
            "never",
            "-o",
            str(output / "final.txt"),
        ]
    else:
        command += ["--no-alt-screen"]
    command += [
        "-C",
        str(root / "workspace"),
        "--add-dir",
        str(root),
        "--sandbox",
        "workspace-write",
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-c",
        "memories.use_memories=false",
        "-c",
        "memories.generate_memories=false",
        "--disable",
        "memories",
    ]
    return command + (["-"] if batch else [_BOOTSTRAP_PROMPT])


def _launch(
    root: Path, *, minutes: float, batch: bool, model: str, effort: str
) -> dict:
    manifest = json.loads((root / "run.json").read_text())
    if (root / "launch.json").exists():
        _check_host(root, required=False)
    output = root / "codex" / f"launch_{_run_id()}"
    output.mkdir(parents=True)
    deadline = time.time() + minutes * 60
    record = {
        "pid": os.getpid(),
        "created_at": psutil.Process().create_time(),
        "deadline": deadline,
        "status": "running",
        "mode": "batch" if batch else "interactive",
        "model": model,
        "reasoning_effort": effort,
        "memory_enabled": False,
        "initial_prompt": _BOOTSTRAP_PROMPT,
        "output": str(output),
        "new_conversation": True,
    }
    write_json(root / "launch.json", record)
    write_json(output / "launch.json", record)
    env = _environment(Path(manifest["repo"]))
    env["GENSIM_LAB_BROKER"] = str(root)
    env["GENSIM_LAB_DEADLINE"] = str(deadline)
    stopped = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat, args=(root, record, stopped), daemon=True
    )
    _write_heartbeat(root, record)
    heartbeat.start()
    meter = UsageMeter(root, "launch")
    # This host reads only execution requests; it never supplies task advice.
    try:
        result = _process(
            _codex_command(root, output, batch=batch, model=model, effort=effort),
            cwd=root / "workspace",
            output=output,
            timeout=minutes * 60,
            env=env,
            prompt=_BOOTSTRAP_PROMPT if batch else None,
            service=lambda: _serve_requests(root, deadline),
            interactive=not batch,
        )
        record.update(
            status="timed_out" if result["timed_out"] else "exited", process=result
        )
    except KeyboardInterrupt:
        request = root / "stop.json"
        if request.is_file() and json.loads(request.read_text()).get("launch") == str(
            output
        ):
            record["status"] = "stopped"
            record["process"] = json.loads((output / "process.json").read_text())
        else:
            record["status"] = "interrupted"
            raise
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stopped.set()
        heartbeat.join()
        write_json(root / "launch.json", record)
        write_json(output / "launch.json", record)
        from .delivery import finalize_run

        try:
            delivery = finalize_run(root, usage_meter=meter)
        finally:
            meter.close()
        print(f"[Agent Lab] Final report: {root / 'final/report.md'}", flush=True)
        print(
            f"[Agent Lab] Final video: {root / 'final/video.mp4' if delivery['video'] else 'none'}",
            flush=True,
        )
    return record


def _write_heartbeat(root: Path, record: dict) -> None:
    staging = root / "heartbeat.tmp"
    write_json(staging, {"launch": record["output"], "time": time.time()})
    staging.replace(root / "heartbeat.json")


def _heartbeat(root: Path, record: dict, stopped: threading.Event) -> None:
    # The client cannot see host PIDs across the Codex PID namespace. Keep this
    # independent of the request loop, which blocks during GPU execution.
    while not stopped.wait(1):
        _write_heartbeat(root, record)
        request = root / "stop.json"
        if (
            request.is_file()
            and json.loads(request.read_text()).get("launch") == record["output"]
        ):
            os.kill(os.getpid(), signal.SIGINT)
            return


def _check_host(root: Path, *, required: bool = True) -> dict:
    path = root / "launch.json"
    record = json.loads(path.read_text()) if path.is_file() else {}
    alive = False
    if required and record.get("status") == "running":
        path = root / "heartbeat.json"
        heartbeat = json.loads(path.read_text()) if path.is_file() else {}
        alive = (
            heartbeat.get("launch") == record.get("output")
            and time.time() - heartbeat.get("time", 0) < 10
        )
    elif not required and record.get("status") == "running":
        try:
            process = psutil.Process(record["pid"])
            alive = (
                process.create_time() == record["created_at"]
                and process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            pass
    if required and (not alive or time.time() >= record["deadline"]):
        raise RuntimeError(
            f"No live experiment host. Run: python -m embodichain.gen_sim.agent_lab launch --run-dir '{root}'"
        )
    if not required and alive:
        raise RuntimeError(f"This workspace already has a live host: {record['pid']}")
    return record


def _workspace_main(root: Path, arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Run experiments and inspect retained evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--script", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--timeout", type=float, default=300)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--attempt", type=Path, required=True)
    commands.add_parser("status")
    commands.add_parser("stop")
    commands.add_parser("usage")
    args = parser.parse_args(arguments)
    if args.command == "inspect":
        from .report import inspect_attempt

        print(
            json.dumps(
                inspect_attempt(args.attempt.resolve()), ensure_ascii=False, indent=2
            )
        )
    elif args.command == "status":
        print((root / "launch.json").read_text())
    elif args.command == "usage":
        print((root / "usage.json").read_text())
    elif args.command == "stop":
        host = _check_host(root)
        staging = root / "stop.tmp"
        write_json(staging, {"launch": host["output"]})
        staging.replace(root / "stop.json")
        print("Stop requested; the host will flush and retain current artifacts.")
    else:
        host = _check_host(root)
        os.environ["GENSIM_LAB_BROKER"] = str(root)
        os.environ["GENSIM_LAB_DEADLINE"] = str(host["deadline"])
        result = execute_script(
            root,
            args.script.resolve(),
            args.config.resolve() if args.config else None,
            timeout=args.timeout,
        )
        raise SystemExit(
            0 if result.get("worker", {}).get("execution_completed") else 1
        )


def _launch_window(
    root: Path, *, minutes: float, batch: bool, model: str, effort: str
) -> int:
    manifest = json.loads((root / "run.json").read_text())
    command = [
        "gnome-terminal",
        "--wait",
        "--title",
        f"Agent Lab: {manifest['task']['task_id']}",
        "--",
        manifest["python"],
        "-B",
        "-m",
        "embodichain.gen_sim.agent_lab",
        "launch",
        "--run-dir",
        str(root),
        "--minutes",
        str(minutes),
        "--model",
        model,
        "--reasoning-effort",
        effort,
    ]
    if batch:
        command.append("--batch")
    return subprocess.call(command, env=_environment(Path(manifest["repo"])))
