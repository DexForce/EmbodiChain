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

"""Isolated process execution and frozen local repetition plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from types import FrameType
from typing import Any

from .artifacts import ArtifactStore, read_json_object, write_json
from .contracts import Budget
from .planning import BudgetExceeded, BudgetLedger
from .records import SCHEMA_VERSION, RunSpec, validate_run_result

__all__ = ["execute_worker", "repeat_schedule", "run_experiment"]


def _record_common_artifacts(
    root: Path, row: Mapping[str, Any], *, run_output: Path
) -> None:
    """Append run evidence to the standard raw, quality and index artifacts."""
    store = ArtifactStore(root)
    raw = dict(row)
    raw["record_id"] = row["run_id"]
    store.append_raw(raw)
    quality_path = root / "quality.json"
    quality = (
        read_json_object(quality_path)
        if quality_path.exists()
        else {"schema_version": SCHEMA_VERSION, "runs": []}
    )
    quality_rows = quality.setdefault("runs", [])
    if not isinstance(quality_rows, list):
        raise ValueError("quality.json runs must be a list")
    quality_rows[:] = [
        item for item in quality_rows if item.get("run_id") != row["run_id"]
    ]
    quality_rows.append(
        {
            "run_id": row["run_id"],
            "backend": row.get("backend"),
            "status": row.get("status"),
            "quality_status": row.get("quality_status", "missing"),
            "task_status": row.get("task_status", "not_evaluated"),
            "data_status": row.get("data_status", "not_evaluated"),
            "reason": row.get("error") or row.get("failure_reason"),
            "evidence": row.get("evidence", []),
        }
    )
    write_json(quality_path, quality)
    for name, kind in (("result.json", "run_result"), ("worker.log", "worker_log")):
        path = run_output / name
        if path.is_file():
            store.register_artifact(path, kind=kind)


def _cancel_on_signal(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"Received signal {signum}")


def execute_worker(
    command: Sequence[str],
    output: Path,
    *,
    backend: str,
    repeat: int,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Preserve a worker result/log even after nonzero exit or timeout."""
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists() or (output / "worker.log").exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    start = time.perf_counter()
    failure = None
    status = "failed"
    returncode = None
    interrupted = False
    with (output / "worker.log").open("w") as log:
        try:
            proc = subprocess.Popen(
                list(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=None if env is None else dict(env),
                start_new_session=True,
            )
            previous_term = None
            if threading.current_thread() is threading.main_thread():
                previous_term = signal.signal(signal.SIGTERM, _cancel_on_signal)
            try:
                returncode = proc.wait(timeout=timeout_s)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = proc.wait()
                interrupted = isinstance(exc, KeyboardInterrupt)
                status = "interrupted" if interrupted else "timeout"
                failure = (
                    "Worker interrupted"
                    if interrupted
                    else f"Worker exceeded {timeout_s:g} seconds"
                )
            finally:
                if previous_term is not None:
                    signal.signal(signal.SIGTERM, previous_term)
        except OSError as exc:
            failure = f"{type(exc).__name__}: {exc}"
    result = {}
    if result_path.exists():
        try:
            result = read_json_object(result_path)
            validate_run_result(result)
        except (ValueError, OSError) as exc:
            result_path.replace(output / "invalid-worker-result.json")
            result = {}
            failure = f"Unreadable worker result: {exc}"
    if failure is None and returncode != 0:
        failure = result.get("error") or f"Worker exited with code {returncode}"
    if failure is None and result.get("status") != "completed":
        failure = result.get("error") or "Worker did not produce a completed result"
    if failure:
        result.update(status=status, error=failure)
    result.update(
        schema_version=SCHEMA_VERSION,
        backend=backend,
        repeat=repeat,
        process_wall_s=time.perf_counter() - start,
        command=list(command),
        returncode=returncode,
    )
    write_json(result_path, result)
    if interrupted:
        raise KeyboardInterrupt(failure)
    return result


def repeat_schedule(
    backends: Sequence[str], repeats: int
) -> tuple[tuple[str, int], ...]:
    """Freeze repeat slots and alternate backend order on successive repeats."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("backends must be nonempty and unique")
    return tuple(
        (backend, repeat)
        for repeat in range(repeats)
        for backend in (backends if repeat % 2 == 0 else tuple(reversed(backends)))
    )


def run_experiment(
    root: Path,
    runs: Sequence[RunSpec],
    *,
    experiment_id: str,
    budget: Budget | None = None,
) -> list[dict[str, Any]]:
    """Execute a fixed plan, retaining failed, interrupted and unstarted rows.

    Run directories must be distinct descendants of root. Environment values
    stay in memory; public manifests contain only executable argv and identity.
    Interruption is propagated after the plan's artifact index is updated.
    """
    root = root.absolute()
    if not runs:
        raise ValueError("An experiment must contain at least one run")
    if (root / "runs.json").exists():
        raise FileExistsError(f"Experiment already has run records: {root}")
    root.mkdir(parents=True, exist_ok=True)
    directories = []
    for spec in runs:
        relative = spec.output.resolve().relative_to(root.resolve())
        if relative == Path(".") or relative.as_posix() in directories:
            raise ValueError(
                "Run directories must be distinct descendants of the experiment"
            )
        if (spec.output / "result.json").exists() or (
            spec.output / "worker.log"
        ).exists():
            raise FileExistsError(
                f"Run directory already contains results: {spec.output}"
            )
        directories.append(relative.as_posix())
    manifest_path = root / "manifest.json"
    manifest = read_json_object(manifest_path) if manifest_path.exists() else {}
    ledger = BudgetLedger(budget or Budget())
    manifest.update(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        budget=ledger.budget.to_dict(),
        runs=[
            {
                "run_id": directory,
                "backend": spec.backend,
                "repeat": spec.repeat,
                "attempt": spec.attempt,
                "case_id": spec.case_id,
                "command": list(spec.command),
                "timeout_s": spec.timeout_s,
            }
            for spec, directory in zip(runs, directories, strict=True)
        ],
    )
    write_json(manifest_path, manifest)
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "run_id": directory,
            "run_dir": directory,
            "backend": spec.backend,
            "repeat": spec.repeat,
            "attempt": spec.attempt,
            "case_id": spec.case_id,
            "status": "not_run",
            "error": "Not started",
        }
        for spec, directory in zip(runs, directories, strict=True)
    ]
    write_json(root / "runs.json", rows)
    for index, spec in enumerate(runs):
        print(
            f"[{index + 1}/{len(runs)}] {spec.backend}, repeat {spec.repeat + 1}",
            flush=True,
        )
        try:
            ledger.reserve_run()
        except BudgetExceeded as exc:
            row = dict(rows[index])
            row["error"] = f"Budget exhausted: {exc}"
            rows[index] = row
            write_json(root / "runs.json", rows)
            _record_common_artifacts(root, row, run_output=spec.output)
            write_json(root / "budget.json", ledger.to_dict())
            print(f"  not_run: {row['error']}", flush=True)
            continue
        interrupted = False
        try:
            row = execute_worker(
                spec.command,
                spec.output,
                backend=spec.backend,
                repeat=spec.repeat,
                timeout_s=spec.timeout_s,
                env=spec.env,
            )
        except KeyboardInterrupt:
            interrupted = True
            result_path = spec.output / "result.json"
            row = (
                read_json_object(result_path)
                if result_path.exists()
                else {
                    "status": "interrupted",
                    "error": "Interrupted before worker result",
                    "backend": spec.backend,
                    "repeat": spec.repeat,
                }
            )
        ledger.finish_attempt(row.get("status", "failed"))
        row.update(
            schema_version=SCHEMA_VERSION,
            experiment_id=experiment_id,
            run_id=directories[index],
            run_dir=directories[index],
            case_id=spec.case_id,
            attempt=spec.attempt,
        )
        write_json(spec.output / "result.json", row)
        rows[index] = row
        write_json(root / "runs.json", rows)
        _record_common_artifacts(root, row, run_output=spec.output)
        write_json(root / "budget.json", ledger.to_dict())
        print(f"  {row['status']}: {row.get('error', 'result saved')}", flush=True)
        if interrupted:
            raise KeyboardInterrupt(
                "Experiment interrupted; remaining runs are not_run"
            )
    return rows
