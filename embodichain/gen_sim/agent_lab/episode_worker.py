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

"""Internal worker for ordinary Python chunks, not a robot action interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
from typing import Any

import psutil

from .catalog import write_json

__all__ = []


def _serializable(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Chunk result is not JSON serializable: {type(value).__name__}")


def _reply(path: Path, value: dict) -> None:
    staging = path.with_suffix(".tmp")
    staging.write_text(
        json.dumps(
            value, default=_serializable, ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n"
    )
    staging.replace(path)


def serve_episode(lab: Any, commands: Path) -> dict:
    """Execute chunks in one namespace; exceptions do not reset or step the scene.

    Commands carry file paths or observation/lifecycle requests. They never carry
    a robot action grammar. All physical advancement is still explicit Python.
    """
    commands.mkdir(parents=True, exist_ok=True)
    namespace = {"lab": lab, "state": {}, "__name__": "__main__"}
    chunks = []
    _reply(lab.output / "ready.json", {"status": "ready", "state": lab.snapshot()})
    while True:
        pending = [
            p
            for p in sorted(commands.glob("*.json"))
            if not p.name.endswith(".reply.json")
            and not p.with_suffix(".reply.json").exists()
        ]
        if not pending:
            time.sleep(0.05)
            continue
        request = pending[0]
        job = json.loads(request.read_text())
        started = time.perf_counter()
        result = {
            "command_id": request.stem,
            "operation": job["operation"],
            "ok": False,
            "started_at": time.time(),
            "simulation_start_seconds": lab.snapshot().get("time"),
        }
        try:
            if job["operation"] == "close":
                _reply(request.with_suffix(".reply.json"), {**result, "ok": True})
                return {"chunks": chunks, "task_completed": None}
            if job["operation"] == "exec":
                path = Path(job["script"])
                namespace.pop("result", None)
                namespace["__file__"] = str(path)
                original_path = list(sys.path)
                sys.path.insert(0, str(path.parent))
                try:
                    try:
                        exec(compile(path.read_text(), str(path), "exec"), namespace)
                    except SystemExit as exc:
                        if exc.code not in (None, 0):
                            raise
                finally:
                    sys.path[:] = original_path
                # Validate the response before publishing it, without discarding state.
                result["result"] = json.loads(
                    json.dumps(
                        namespace.get("result"), default=_serializable, allow_nan=False
                    )
                )
            elif job["operation"] != "observe":
                raise ValueError(f"Unknown session operation: {job['operation']}")
            result["ok"] = True
        except (Exception, SystemExit) as exc:
            result.update(
                error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
            )
        result["wall_seconds"] = time.perf_counter() - started
        result["state"] = lab.snapshot()
        result["simulation_end_seconds"] = result["state"].get("time")
        lab.capture(f"chunk_{request.stem}")
        result["image"] = str(lab.output / f"chunk_{request.stem}.png")
        chunks.append(
            {
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"state", "result", "traceback"}
                },
                "reply_file": str(request.with_suffix(".reply.json")),
            }
        )
        write_json(lab.output / "chunks.json", chunks)
        _reply(request.with_suffix(".reply.json"), result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--owner-created-at", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    finished = threading.Event()

    def watchdog():
        while not finished.wait(1):
            try:
                owner = psutil.Process(args.owner_pid)
                alive = (
                    owner.create_time() == args.owner_created_at
                    and owner.status() != psutil.STATUS_ZOMBIE
                )
            except psutil.NoSuchProcess:
                alive = False
            if not alive or time.time() >= args.deadline:
                os.kill(os.getpid(), signal.SIGTERM)
                if not finished.wait(20):
                    os.kill(os.getpid(), signal.SIGKILL)
                return

    monitor = threading.Thread(target=watchdog, daemon=True)
    monitor.start()
    lab = None
    report = {
        "execution_completed": False,
        "task_success": None,
        "stage": "setup",
        "execution_mode": "segmented",
    }
    try:
        from .runtime import Lab, LabCfg
        import torch
        import dexsim

        write_json(
            args.output / "environment.json",
            {
                "python": sys.version,
                "executable": sys.executable,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "dexsim_module": dexsim.__file__,
            },
        )

        lab = Lab.__new__(Lab)
        lab.__init__(LabCfg(**json.loads(args.config.read_text())), args.output)
        report["stage"] = "chunks"
        report["solver_result"] = serve_episode(lab, args.output / "commands")
        report.update(execution_completed=True, stage="complete")
    except BaseException as exc:
        report.update(
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
    finally:
        if lab is not None:
            try:
                lab.close()
            except Exception:
                report["cleanup_error"] = traceback.format_exc()
                report["execution_completed"] = False
        report["video"] = (
            str(args.output / "video.mp4")
            if (args.output / "video.mp4").exists()
            else None
        )
        write_json(args.output / "result.json", report)
        finished.set()
        monitor.join()
    return 0 if report["execution_completed"] else 1


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
