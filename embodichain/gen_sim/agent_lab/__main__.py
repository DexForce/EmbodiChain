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

"""Command-line entry for open-ended GenSim experiments."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import signal

import psutil

from .catalog import Task, read_catalog
from .session import (
    _DEFAULT_MODEL,
    _DEFAULT_REASONING_EFFORT,
    create_run,
    execute_script,
    solve,
)

__all__ = ["main"]


def _prepare_run(args: argparse.Namespace, repo: Path) -> Path:
    task = (
        Task(**json.loads(args.task_file.read_text()))
        if args.task_file
        else read_catalog(args.assets)[args.task]
    )
    root = create_run(
        task,
        repo=repo,
        output_root=args.output_root,
        objective=args.objective,
        robot_component=args.robot_component,
    )
    print(f"[Agent Lab] Run directory: {root}", flush=True)
    return root


def main() -> None:
    """Inventory assets, prepare experiments, execute scripts, or launch Codex."""

    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    if sys.argv[1:2] == ["workspace"]:
        from ._launch import _workspace_main

        bridge = argparse.ArgumentParser(add_help=False)
        bridge.add_argument("--run-dir", type=Path, required=True)
        location, arguments = bridge.parse_known_args(sys.argv[2:])
        _workspace_main(location.run_dir.resolve(), arguments)
        return
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--assets", type=Path, default=repo / "gym_project/task100")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--attempt", type=Path, required=True)
    finalize = commands.add_parser(
        "finalize", help="Publish or recover final delivery without running simulation"
    )
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--attempt", type=Path)
    for name in ("prepare", "solve", "pipeline"):
        command = commands.add_parser(name)
        selection = command.add_mutually_exclusive_group(required=True)
        selection.add_argument("--task")
        selection.add_argument("--task-file", type=Path)
        command.add_argument(
            "--objective", choices=["solve", "cold_start"], default="solve"
        )
        command.add_argument("--robot-component", type=Path)
        command.add_argument(
            "--assets", type=Path, default=repo / "gym_project/task100"
        )
        command.add_argument(
            "--output-root", type=Path, default=repo / "outputs/task100"
        )
        if name in ("solve", "pipeline"):
            command.add_argument("--minutes", type=float, default=45)
            command.add_argument("--model", default=_DEFAULT_MODEL)
            command.add_argument(
                "--reasoning-effort", default=_DEFAULT_REASONING_EFFORT
            )
            if name == "solve":
                command.add_argument("--sandbox", default="workspace-write")
            else:
                mode = command.add_mutually_exclusive_group()
                mode.add_argument(
                    "--batch", action="store_true", help="Run unattended (default)"
                )
                mode.add_argument(
                    "--window",
                    action="store_true",
                    help="Open an interactive Codex window instead",
                )
    run = commands.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--script", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--timeout", type=float, default=300)
    resume = commands.add_parser("resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--minutes", type=float, default=45)
    resume.add_argument(
        "--model", help="Override the previous model; omitted values are inherited"
    )
    resume.add_argument(
        "--reasoning-effort",
        help="Override the previous effort; omitted values are inherited",
    )
    resume.add_argument("--sandbox", default="workspace-write")
    launch = commands.add_parser(
        "launch", help="Start a new Codex from the self-contained workspace"
    )
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--minutes", type=float, default=20)
    launch.add_argument("--model", default=_DEFAULT_MODEL)
    launch.add_argument("--reasoning-effort", default=_DEFAULT_REASONING_EFFORT)
    launch.add_argument("--batch", action="store_true")
    launch.add_argument("--window", action="store_true")
    args = parser.parse_args()
    if args.command in ("launch", "pipeline"):
        from ._launch import _launch, _launch_window

        method = _launch_window if args.window else _launch
        root = (
            _prepare_run(args, repo)
            if args.command == "pipeline"
            else args.run_dir.resolve()
        )
        result = method(
            root,
            minutes=args.minutes,
            batch=not args.window if args.command == "pipeline" else args.batch,
            model=args.model,
            effort=args.reasoning_effort,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if isinstance(result, dict) and result["status"] == "stopped":
            return
        raise SystemExit(
            result if isinstance(result, int) else result["process"]["returncode"]
        )
    elif args.command == "finalize":
        from .delivery import finalize_run

        result = finalize_run(args.run_dir.resolve(), attempt=args.attempt)
        print(
            json.dumps(
                {
                    "result": str(args.run_dir.resolve() / "final/result.json"),
                    "report": str(args.run_dir.resolve() / "final/report.md"),
                    "video": (
                        str(args.run_dir.resolve() / "final/video.mp4")
                        if result["video"]
                        else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "inventory":
        tasks = read_catalog(args.assets)
        print(
            json.dumps(
                {
                    "counts": dict(Counter(t.asset_status for t in tasks.values())),
                    "tasks": [task.to_dict() for task in tasks.values()],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "resume":
        result = solve(
            args.run_dir.resolve(),
            minutes=args.minutes,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            sandbox=args.sandbox,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("agent_error"):
            sys.exit(1)
    elif args.command == "inspect":
        from .report import inspect_attempt

        print(json.dumps(inspect_attempt(args.attempt), ensure_ascii=False, indent=2))
    elif args.command == "run":
        from .delivery import finalize_run
        from .catalog import write_json
        from .usage import UsageMeter

        root = args.run_dir.resolve()
        brokered = os.environ.get("GENSIM_LAB_BROKER") == str(root)
        execution = {
            "entrypoint": "run",
            "status": "running",
            "pid": os.getpid(),
            "created_at": psutil.Process().create_time(),
        }
        if not brokered:
            write_json(root / "delivery_state.json", execution)
        meter = None if brokered else UsageMeter(root, "run")
        result = None
        try:
            result = execute_script(
                root,
                args.script.resolve(),
                args.config.resolve() if args.config else None,
                timeout=args.timeout,
            )
            process = result["process"]
            execution.update(
                status=(
                    "timed_out"
                    if process.get("timed_out")
                    else "completed" if process.get("returncode") == 0 else "failed"
                ),
                process=process,
            )
        except KeyboardInterrupt:
            execution["status"] = "interrupted"
            raise
        except Exception as exc:
            execution.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if not brokered:
                write_json(root / "delivery_state.json", execution)
                try:
                    finalize_run(
                        root,
                        attempt=result["attempt"] if result else None,
                        usage_meter=meter,
                    )
                finally:
                    meter.close()
        if result["process"]["returncode"] or not result["worker"].get(
            "execution_completed"
        ):
            sys.exit(1)
    else:
        root = _prepare_run(args, repo)
        if args.command == "solve":
            result = solve(
                root,
                minutes=args.minutes,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                sandbox=args.sandbox,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result.get("agent_error"):
                sys.exit(1)


if __name__ == "__main__":
    main()
