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

"""Run and inspect measured examples of all currently integrated expansion domains."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import webbrowser
from ._cases import build_cases, dump, recover_portability, worker_environment

__all__ = ["main"]


def run_worker(
    root: Path,
    operation: str,
    *,
    timeout: float,
    job: Path | None = None,
    resume: bool = False,
    backend: str = "default",
) -> int:
    command = [
        sys.executable,
        "-m",
        "examples.expansion.engine.showcase._worker",
        operation,
        str(root),
        "--backend",
        backend,
    ]
    if job:
        command.extend(("--job", str(job)))
    if resume:
        command.append("--resume")
    name = job.stem if job else operation + "-" + backend
    log = root / "logs" / (name + ".log")
    log.parent.mkdir(exist_ok=True)
    print(f"[{operation}] {name} · {log}", flush=True)
    env = worker_environment()
    with log.open("a") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=max(1, timeout))
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            import signal
            import psutil

            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                descendants = []

            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # A child can ignore TERM even after the group leader exits.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                deadline = time.monotonic() + 5
                while descendants:
                    alive = []
                    for child in descendants:
                        try:
                            if (
                                child.is_running()
                                and child.status() != psutil.STATUS_ZOMBIE
                            ):
                                alive.append(child)
                        except psutil.NoSuchProcess:
                            pass
                    if not alive:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("worker descendants did not terminate")
                    descendants = alive
                    time.sleep(0.01)
            dump(
                root / "results" / (name + ".timeout.json"),
                {"operation": operation, "timeout": timeout, "interrupted": True},
            )
            if sys.exc_info()[0] is KeyboardInterrupt:
                raise
            return 124


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/expansion_showcase"))
    p.add_argument("--mode", choices=("full", "quick"), default=None)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true")
    action.add_argument("--reuse", action="store_true")
    p.add_argument("--open", action="store_true")
    p.add_argument(
        "--case",
        action="append",
        help="Select explicit case IDs; unselected cases remain not_run",
    )
    p.add_argument(
        "--extra-attempts",
        type=int,
        default=0,
        help="Explicitly extend selected resumed jobs; distributions remain pinned",
    )
    args = p.parse_args()
    if args.extra_attempts < 0 or (
        args.extra_attempts and not (args.resume and args.case)
    ):
        p.error("--extra-attempts requires --resume and explicit --case IDs")
    root = args.output.resolve()
    if args.mode is None:
        args.mode = (
            json.loads((root / "cases.json").read_text())["mode"]
            if (root / "cases.json").exists()
            else "full"
        )
    if root.exists() and any(root.iterdir()) and not (args.resume or args.reuse):
        p.error("output must be empty; use --resume or --reuse")
    root.mkdir(parents=True, exist_ok=True)
    if not args.reuse:
        import fcntl

        suite_lock = (root / ".showcase.lock").open("a")
        fcntl.flock(suite_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover_portability(root)
        started = time.monotonic()
        budget = 14400 if args.mode == "full" else 1800

        def bounded(operation, timeout, **kwargs):
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                dump(
                    root
                    / "results"
                    / (
                        "skipped-"
                        + operation
                        + "-"
                        + (kwargs.get("job", Path("job")).stem)
                        + ".json"
                    ),
                    {
                        "operation": operation,
                        "reason": "suite_budget_exhausted",
                        "ok": False,
                    },
                )
                return 124
            return run_worker(
                root, operation, timeout=min(timeout, remaining), **kwargs
            )

        for operation, timeout in (("prepare", 600), ("atlas", 700)):
            if bounded(operation, timeout):
                print(f"{operation} failed; inspect logs", file=sys.stderr)
                if not (root / "cases.json").exists():
                    dump(root / "cases.json", {"mode": args.mode, "cases": []})
                from ._report import build_page

                build_page(root)
                raise SystemExit(2)
        cases = build_cases(root, args.mode, preserve=args.resume)
        if args.case:
            unknown = set(args.case) - {c["id"] for c in cases}
            if unknown:
                p.error(f"unknown cases: {sorted(unknown)}")
        for case in cases:
            if not case["selected"] or (args.case and case["id"] not in args.case):
                continue
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                break
            job = root / case["job"]
            if args.extra_attempts:
                import yaml

                cfg = yaml.safe_load(job.read_text())
                cfg["max_attempts"] += args.extra_attempts
                dump(job, cfg)
            exists = (root / "jobs" / case["id"] / "job.json").exists()
            bounded("collect", case["timeout_s"], job=job, resume=exists)
            if (root / "jobs" / case["id"] / "job.json").exists():
                bounded("replay", 600, job=job)
        if args.mode == "full" and not args.case:
            for backend in ("default", "newton"):
                bounded("probe", 180, backend=backend)
            bounded("faults", 900)
        bounded("audit", 900)
    from ._report import build_page

    result = build_page(root)
    print(f"Preview: {root/'index.html'}", flush=True)
    if args.open:
        webbrowser.open((root / "index.html").as_uri())
    if not result["acceptance_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
