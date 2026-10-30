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

"""Run the fixed new3 corpus through the public Task Engine run-all CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Sequence
from zoneinfo import ZoneInfo

__all__ = ["main"]

_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST = Path(__file__).with_name("fresh_tasks.json")
_PROFILES = ("dual_franka", "dual_ur10")
_MODULE = "embodichain.gen_sim.task_engine"
_CASE_WALL_SECONDS = 30 * 60
_CLEANUP_SECONDS = 60


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(_ROOT), *arguments], text=True
    ).strip()


def _snapshot(tasks: Sequence[str]) -> dict[str, Any]:
    sources = _git(
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "embodichain",
        "extensions",
        "pyproject.toml",
        "setup.py",
        "VERSION",
    ).split("\0")
    paths = {_ROOT / name for name in sources if name}
    for task in tasks:
        paths.update(
            (_ROOT / "gym_project/task100/new3" / task / "scene_export").rglob("*")
        )
    # Fingerprint local provider settings without copying or printing credentials.
    paths.update(
        _ROOT / name
        for name in (
            "embodichain/gen_sim/.env",
            "embodichain/gen_sim/asset_engine/configs/gen_config.json",
        )
    )
    return {
        "head": _git("rev-parse", "HEAD"),
        "files": {
            str(path.relative_to(_ROOT)): _sha(path) if path.is_file() else "missing"
            for path in sorted(paths)
            if not path.is_dir()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        },
    }


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(_ROOT / path)
        for path in (".", "extensions", "extensions/embodichain_tasks")
    )
    return environment


def _preflight(environment: dict[str, str]) -> dict[str, str]:
    code = (
        "import importlib,json; "
        "names=('embodichain','embodichain_robots','embodichain_tasks',"
        "'embodichain_tasks.configs'); "
        "print(json.dumps({n:importlib.import_module(n).__file__ for n in names}))"
    )
    try:
        output = subprocess.check_output(
            [sys.executable, "-B", "-c", code],
            cwd=_ROOT,
            env=environment,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Package import preflight failed before model or simulator startup. "
            "Check this checkout's source packages, including "
            "embodichain_tasks.configs.\n" + error.output
        ) from error
    origins = json.loads(output.splitlines()[-1])
    for name, origin in origins.items():
        if not origin or not Path(origin).resolve().is_relative_to(_ROOT):
            raise RuntimeError(f"{name} resolves outside this checkout: {origin}")
    return origins


def _command(
    task: str,
    profile: str,
    output: Path,
    manifest: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-B",
        "-m",
        _MODULE,
        "run-all",
        "--mode",
        "scene",
        "--task-id",
        task,
        "--instruction",
        manifest["tasks"][task],
        "--scene",
        f"gym_project/task100/new3/{task}",
        "--output-root",
        str(output),
        "--robot-profile",
        profile,
        "--model",
        manifest["model"],
        "--base-seed",
        str(manifest["base_seed"]),
    ]
    return command


def _write(path: Path, value: Any) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.replace(path)


def _execute(command: list[str], log: Path, environment: dict[str, str]) -> int:
    # A regular file avoids tee/pipe failures changing the public CLI outcome.
    with log.open("xb") as stream:
        process = subprocess.Popen(
            command,
            cwd=_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=_CASE_WALL_SECONDS)
        except subprocess.TimeoutExpired:
            _stop(process)
            return 124
        except KeyboardInterrupt:
            _stop(process)
            return 130


def _stop(process: subprocess.Popen) -> None:
    """Let the CLI close its workers, then bound cleanup of our own process group."""
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=_CLEANUP_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _workflow_result(output: Path) -> dict[str, Any]:
    manifests = sorted(output.glob("*/run_manifest.json"))
    if len(manifests) != 1:
        return {"workflow_status": "unavailable", "manifest_count": len(manifests)}
    path = manifests[0]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return {
            "run_manifest": str(path),
            "workflow_status": value["status"],
            "failure_class": value.get("failure_class"),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"workflow_status": "unavailable", "report_error": str(error)}


def _print_result(log: Path) -> None:
    """Replay the public CLI's terminal notices and JSON, not its verbose logs."""
    notices = (
        "[Task Engine] Video saved:",
        "[Task Engine] No video files generated for this run.",
        "[Task Engine] Semantic graph HTML:",
    )
    result_started = False
    json_started = False
    try:
        with log.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.startswith(notices):
                    print(line, end="", flush=True)
                    result_started = True
                elif json_started or (result_started and line.startswith("{")):
                    print(line, end="", flush=True)
                    json_started = line.rstrip("\r\n") != "}"
                    if not json_started:
                        result_started = False
    except OSError as error:
        print(
            f"[Task Engine] Unable to display result log {log}: {error}",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Preview or sequentially execute fresh public-CLI tasks on existing scenes.

    Args:
        argv: Campaign arguments, or the process arguments when omitted.

    Returns:
        Zero for a preview or successful campaign; nonzero for failed runs.

    Raises:
        SystemExit: If arguments or scene/output paths are invalid.
        RuntimeError: If packages resolve outside the selected checkout.
    """
    batch_started_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(manifest["tasks"]))
    parser.add_argument(
        "--robot-profile", nargs="+", choices=_PROFILES, default=list(_PROFILES)
    )
    # DexSim incorporates absolute asset paths into bounded native material names.
    parser.add_argument("--output-root", default="outputs/pipeline")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute", action="store_true", help="Call MiMo and run simulation."
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="Preview only (the default)."
    )
    args = parser.parse_args(argv)
    tasks = list(dict.fromkeys(args.tasks or manifest["tasks"]))
    profiles = list(dict.fromkeys(args.robot_profile))
    for task in tasks:
        scene = (
            _ROOT / "gym_project/task100/new3" / task / "scene_export/scene_config.json"
        )
        if not scene.is_file():
            parser.error(f"Missing scene: {scene}")
    root = (_ROOT / Path(args.output_root).expanduser()).resolve()
    if root.is_relative_to(_ROOT / "gym_project/task100/new3"):
        parser.error("Output must not modify the frozen new3 inputs.")
    stamp = batch_started_at.strftime("%Y%m%d_%H%M%S")
    batch = root / stamp
    cases = []
    for profile in profiles:
        for task in tasks:
            output = batch / profile / task
            cases.append(
                {
                    "task_id": task,
                    "robot_profile": profile,
                    "output_root": str(output),
                    "command": _command(
                        task,
                        profile,
                        output,
                        manifest,
                    ),
                    "status": "not_started",
                }
            )
    plan = {
        "schema_version": "gen_sim.public-cli-fresh/v1",
        "batch_started_at": batch_started_at.isoformat(),
        "cwd": str(_ROOT),
        "python": sys.executable,
        "python_version": sys.version,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "model": manifest["model"],
        "base_seed": manifest["base_seed"],
        "case_wall_seconds": _CASE_WALL_SECONDS,
        "cleanup_grace_seconds": _CLEANUP_SECONDS,
        "manifest_sha256": _sha(_MANIFEST),
        "case_count": len(cases),
        "cases": cases,
        "boundary": "Fresh run-all only; workflow success is not strict physical certification.",
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    identity = _snapshot(tasks)
    batch.mkdir(parents=True, exist_ok=False)
    _write(batch / "inputs.json", manifest)
    _write(batch / "source_and_scene_hashes.json", identity)
    plan.update(
        branch=_git("branch", "--show-current"),
        head=identity["head"],
        status="preparing",
    )
    _write(batch / "plan.json", plan)
    exit_code = 0
    try:
        environment = _environment()
        plan.update(
            pythonpath=environment["PYTHONPATH"],
            package_origins=_preflight(environment),
            status="running",
        )
        _write(batch / "plan.json", plan)
        for case in cases:
            if _snapshot(tasks) != identity:
                raise RuntimeError(
                    "Source/configuration/scene drift; remaining runs were not launched."
                )
            output = Path(case["output_root"])
            output.mkdir(parents=True, exist_ok=False)
            log = output / "cli.log"
            case.update(
                status="running",
                log=str(log),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            _write(batch / "plan.json", plan)
            print(
                f"cwd: {_ROOT}\n{shlex.join(case['command'])}\nlog: {log}", flush=True
            )
            started = time.monotonic()
            code = _execute(case["command"], log, environment)
            case.update(
                status=(
                    "interrupted"
                    if code == 130
                    else "timed_out" if code == 124 else "finished"
                ),
                returncode=code,
                wall_seconds=time.monotonic() - started,
                **_workflow_result(output),
            )
            unchanged = _snapshot(tasks) == identity
            case["inputs_unchanged"] = unchanged
            case["workflow_succeeded"] = (
                code == 0 and case["workflow_status"] == "succeeded" and unchanged
            )
            _write(output / "invocation.json", {"cwd": str(_ROOT), **case})
            _write(batch / "plan.json", plan)
            print(
                f"Finished {case['robot_profile']}/{case['task_id']}: "
                f"returncode={code}, workflow={case['workflow_status']}",
                flush=True,
            )
            _print_result(log)
            if code == 130:
                plan["status"], exit_code = "interrupted", 130
                break
            if not unchanged:
                raise RuntimeError(
                    "Source/configuration/scene drift during this run; do not qualify its result."
                )
            if not case["workflow_succeeded"]:
                exit_code = 1
        else:
            plan["status"] = "finished"
    except KeyboardInterrupt:
        plan["status"], exit_code = "interrupted", 130
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        plan.update(status="aborted", error=str(error))
        print(f"Campaign aborted: {error}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        for case in cases:
            if case["status"] == "running":
                case.update(status=plan["status"], workflow_succeeded=False)
        _write(batch / "plan.json", plan)
        print(f"Batch record: {batch / 'plan.json'}", flush=True)
    return exit_code


if __name__ == "__main__":

    def _interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _interrupt)
    raise SystemExit(main())
