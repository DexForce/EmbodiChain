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

"""Spawn-isolated quality evaluation of one immutable benchmark corpus."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import PlannerSpecCfg, SuiteCfg, load_suite, stable_hash

if TYPE_CHECKING:
    from .models import PlannerMetadata, TrialRecord
    from .planners.base import PlannerAdapter
    from .runner import BenchmarkRunResult
    from .video import VideoRecordCfg

__all__ = ["resolve_quality_devices"]


def _worker_device(requested: str) -> str:
    """Resolve an unspecified CUDA index in the parent before spawning workers."""
    import torch

    from .runner import resolve_device

    device = resolve_device(requested)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return str(device)


def resolve_quality_devices(requested: Sequence[str]) -> list[str]:
    """Validate explicit visible CUDA ordinals before starting quality evaluation.

    Args:
        requested: Nonempty sequence of unique explicit ``cuda:N`` device names.
            Indices refer to devices visible through ``CUDA_VISIBLE_DEVICES``.

    Returns:
        Canonical device names in the caller's order. This validation does not
        create workers or simulators or change the current CUDA device.

    Raises:
        ValueError: If a device is implicit, unavailable, repeated, or malformed.
    """
    import torch

    if isinstance(requested, (str, bytes)) or not requested:
        raise ValueError(
            "quality_devices must be a nonempty list of explicit cuda:N devices."
        )
    resolved = []
    for value in requested:
        if not isinstance(value, str):
            raise ValueError("quality_devices must contain explicit cuda:N strings.")
        try:
            device = torch.device(value)
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                "quality_devices must contain explicit cuda:N devices."
            ) from error
        if device.type != "cuda" or device.index is None:
            raise ValueError("quality_devices must contain explicit cuda:N devices.")
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise ValueError(
                f"quality device {device} is unavailable in CUDA_VISIBLE_DEVICES."
            )
        resolved.append(str(device))
    if len(set(resolved)) != len(resolved):
        raise ValueError("quality_devices must not repeat a device.")
    return resolved


def _partition_rows(rows: list[dict[str, Any]], workers: int) -> list[list[int]]:
    """Balance closed-loop work while keeping shared base projections together."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        base = row.get("case_parameters", {}).get("base_case_id") or row["case_id"]
        groups[(row["track"], str(base))].append(index)
    count = min(workers, len(groups))
    shards: list[list[int]] = [[] for _ in range(count)]
    costs = [0] * count
    weighted = [
        (sum(rows[i]["num_waypoints"] * rows[i]["batch_size"] for i in ids), ids)
        for ids in groups.values()
    ]
    for cost, ids in sorted(weighted, key=lambda item: (-item[0], item[1][0])):
        target = min(range(count), key=lambda index: (costs[index], index))
        shards[target].extend(ids)
        costs[target] += cost
    return [sorted(shard) for shard in shards]


@contextmanager
def _worker_environment(threads: int) -> Iterator[None]:
    """Apply thread limits before spawned interpreters import native libraries."""
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: str(threads) for name in names})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize_worker(threads: int) -> None:
    """Bound worker CPU pools without changing the calling process."""
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def _run_shard(job: dict[str, Any]) -> tuple[BenchmarkRunResult, dict[str, Any]]:
    """Run one serial simulator in its own spawned process and console log."""
    import torch

    from .artifacts import write_json
    from .runner import BenchmarkRunner, resolve_device

    directory = Path(job["directory"])
    directory.mkdir(exist_ok=True)
    state = {
        "status": "running",
        "pid": os.getpid(),
        "shard_index": job["index"],
        "device": job["device"],
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "case_count": job["case_count"],
        "started_at_unix_s": time.time(),
    }
    write_json(directory / "process.json", state)
    start = time.perf_counter()
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        with (directory / "console.log").open("w") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            device = resolve_device(job["device"])
            if device.type == "cuda":
                torch.cuda.set_device(device)
            runner = BenchmarkRunner(
                job["suite"],
                job["suite"].planners,
                device=str(device),
                headless=True,
                output_root=directory,
                case_manifest=job["case_manifest"],
                quality_batch_size=job.get("quality_batch_size", 1),
                execution={
                    "mode": "parallel_quality",
                    "workers": job["workers"],
                    "shard_index": job["index"],
                    "timing_comparable": False,
                },
            )
            result = runner.run()
            state.update(status="completed", run_dir=str(result.run_dir.resolve()))
            return result, state
    except BaseException as error:
        error_path = directory / "error.log"
        error_path.write_text(traceback.format_exc())
        state.update(
            status="error",
            error=f"{type(error).__name__}: {error}",
            traceback=str(error_path.resolve()),
        )
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        state["elapsed_seconds"] = time.perf_counter() - start
        state["finished_at_unix_s"] = time.time()
        write_json(directory / "process.json", state)


def _merge_planner_metadata(
    previous: PlannerMetadata, current: PlannerMetadata
) -> PlannerMetadata:
    """Allow derived limits to be absent on workers that never build the planner."""
    previous_parameters = dict(previous.parameters)
    current_parameters = dict(current.parameters)
    unresolved = object()
    previous_limits = previous_parameters.pop("resolved_motion_limits", unresolved)
    current_limits = current_parameters.pop("resolved_motion_limits", unresolved)
    if replace(previous, parameters=previous_parameters) != replace(
        current, parameters=current_parameters
    ) or (
        previous_limits is not unresolved
        and current_limits is not unresolved
        and previous_limits != current_limits
    ):
        raise ValueError("Worker planner metadata differ.")
    return previous if previous_limits is not unresolved else current


def _merge_records(
    results: list[BenchmarkRunResult],
    rows: list[dict[str, Any]],
    specs: list[PlannerSpecCfg],
    measured_trials: int,
) -> tuple[list[TrialRecord], list[PlannerMetadata]]:
    """Validate exact corpus coverage and combine raw evidence once."""
    from .models import TrialPhase

    expected = {row["case_id"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Parallel corpus contains duplicate case IDs.")
    observed: dict[str, dict[str, Any]] = {}
    metadata: dict[str, PlannerMetadata] = {}
    records: list[TrialRecord] = []
    keys: set[tuple[object, ...]] = set()
    by_case_method: dict[tuple[str, str], list[TrialRecord]] = defaultdict(list)
    methods = {spec.id for spec in specs}
    for result in results:
        payload = json.loads((result.run_dir / "case_manifest.json").read_text())
        if payload.get("cases_sha256") != stable_hash(payload.get("cases")):
            raise ValueError("Worker case manifest hash mismatch.")
        for row in payload["cases"]:
            identity = row["case_id"]
            if identity in observed or expected.get(identity) != row:
                raise ValueError("Worker cases overlap or differ from frozen inputs.")
            observed[identity] = row
        for info in result.planner_metadata:
            if info.algorithm_id not in methods:
                raise ValueError("Worker returned an undeclared planner.")
            if info.algorithm_id in metadata:
                info = _merge_planner_metadata(metadata[info.algorithm_id], info)
            metadata[info.algorithm_id] = info
        for record in result.records:
            if record.case_id not in expected or record.algorithm_id not in methods:
                raise ValueError(
                    "Worker returned a trial outside the declared population."
                )
            case = expected[record.case_id]
            if any(
                getattr(record, name) != case[key]
                for name, key in (
                    ("track", "track"),
                    ("suite_version", "suite_version"),
                    ("seed", "seed"),
                    ("batch_size", "batch_size"),
                    ("waypoint_count", "num_waypoints"),
                    ("robot_id", "robot_id"),
                )
            ):
                raise ValueError("Worker trial identity differs from its frozen case.")
            key = (record.case_id, record.algorithm_id, record.phase, record.repeat)
            if key in keys:
                raise ValueError("Duplicate worker trial identity.")
            keys.add(key)
            records.append(record)
            by_case_method[(record.case_id, record.algorithm_id)].append(record)
    if observed != expected:
        raise ValueError("Parallel workers did not return the complete frozen corpus.")
    if set(metadata) != methods:
        raise ValueError("Parallel workers omitted planner metadata.")
    for case_id in expected:
        for method in methods:
            trials = by_case_method[(case_id, method)]
            measured = [row for row in trials if row.phase is TrialPhase.MEASURED]
            terminal = any(
                (row.phase is TrialPhase.AVAILABILITY and row.status == "unsupported")
                or (
                    row.phase in {TrialPhase.CONSTRUCT, TrialPhase.PREPARE}
                    and row.status == "error"
                )
                for row in trials
            )
            if terminal:
                if measured:
                    raise ValueError(
                        "Worker mixed terminal lifecycle failure and measured trials."
                    )
            elif {row.repeat for row in measured} != set(range(measured_trials)):
                raise ValueError(
                    f"Worker omitted measured trials for {case_id}/{method}."
                )
    case_order = {row["case_id"]: index for index, row in enumerate(rows)}
    method_order = {spec.id: index for index, spec in enumerate(specs)}
    phase_order = {phase: index for index, phase in enumerate(TrialPhase)}
    records.sort(
        key=lambda record: (
            method_order[record.algorithm_id],
            case_order[record.case_id],
            phase_order[record.phase],
            record.repeat,
        )
    )
    return records, [metadata[spec.id] for spec in specs]


def run_parallel_benchmark(
    suite: str | Path | SuiteCfg,
    *,
    planners: Sequence[PlannerSpecCfg] | None,
    adapters: Mapping[str, type[PlannerAdapter]] | None,
    device: str,
    headless: bool,
    output_root: str | Path,
    video: VideoRecordCfg | None,
    case_manifest: str | Path | None,
    workers: int,
    worker_threads: int,
    quality_batch_size: int = 1,
    quality_devices: Sequence[str] | None = None,
) -> BenchmarkRunResult:
    """Freeze inputs, execute bounded independent processes, then aggregate."""
    import torch

    from .aggregation import aggregate_results
    from .api import prepare_case_manifest
    from .artifacts import (
        TrialJsonlWriter,
        _read_case_manifest,
        create_run_directory,
        environment_metadata,
        write_json,
        write_resolved_suite,
    )
    from .embodiments import resolve_embodiment
    from .registry import planner_adapter_names
    from .reporting import write_markdown_report
    from .runner import BenchmarkRunResult

    if adapters:
        raise ValueError("Parallel evaluation requires installed planner adapters.")
    if video is not None or not headless:
        raise ValueError(
            "Parallel quality evaluation requires headless operation without video."
        )
    if isinstance(suite, (str, Path)):
        resolved = load_suite(suite)
    elif isinstance(suite, SuiteCfg):
        resolved = deepcopy(suite)
    else:
        raise TypeError("suite must be a suite name, YAML path, or SuiteCfg.")
    selected = deepcopy(
        list(planners)
        if planners is not None
        else [spec for spec in resolved.planners if spec.enabled]
    )
    if not selected:
        raise ValueError("No algorithms were selected for the benchmark.")
    if any(not isinstance(spec, PlannerSpecCfg) for spec in selected):
        raise TypeError("planners must contain PlannerSpecCfg instances.")
    for spec in selected:
        spec.enabled = True
    resolved.planners = selected
    resolved.validate_benchmark()
    # Import built-ins before checking names, matching the serial API.
    from . import planners as _builtin_planners  # noqa: F401

    known = set(planner_adapter_names())
    if any(spec.adapter not in known for spec in selected):
        raise ValueError("Parallel evaluation requires installed planner adapters.")
    device = _worker_device(device)
    devices = (
        resolve_quality_devices(quality_devices)
        if quality_devices is not None
        else [device]
    )
    resolution = resolve_embodiment(
        resolved.embodiment, resolved.robot, base_dir=Path.cwd()
    )
    run_dir = create_run_directory(output_root, resolved.name)
    execution: dict[str, Any] = {
        "mode": "parallel_quality",
        "status": "preparing",
        "workers_requested": workers,
        "quality_batch_size": quality_batch_size,
        "worker_threads": worker_threads,
        "device": device,
        "quality_devices": devices,
        "timing_comparable": False,
        "start_method": "spawn",
        "solver_rng_contract": "backend_owned",
        "shards": [],
    }
    write_resolved_suite(run_dir / "resolved_suite.yaml", resolved)
    write_json(run_dir / "environment.json", environment_metadata())
    start = time.perf_counter()
    try:
        if case_manifest is None:
            case_manifest = prepare_case_manifest(
                resolved, run_dir / "generated_cases.json", device=device, headless=True
            )
        payload = json.loads(Path(case_manifest).read_text())
        cases = _read_case_manifest(
            case_manifest, torch.device("cpu"), expected_embodiment=resolution
        )
        rows = payload["cases"]
        if any(
            row.get("case_parameters", {}).get("measurement_purpose")
            == "performance_replay"
            for row in rows
        ):
            raise ValueError("Performance replay must run serially (workers=1).")
        tracks = {track.id: track for track in resolved.evaluation_tracks()}
        if {case.track for case in cases} != set(tracks) or any(
            case.suite_version != resolved.suite_version for case in cases
        ):
            raise ValueError(
                "Replay population conflicts with suite tracks or version."
            )
        # Retain the original canonical row order and input hash in the merged artifact.
        write_json(run_dir / "case_manifest.json", payload)
        shards = _partition_rows(rows, workers)
        execution.update(
            status="running", workers=len(shards), cases_sha256=payload["cases_sha256"]
        )
        jobs = []
        for index, indices in enumerate(shards):
            directory = run_dir / "workers" / f"{index:03d}"
            directory.mkdir(parents=True)
            shard_rows = [rows[i] for i in indices]
            shard_payload = {
                **payload,
                "cases": shard_rows,
                "cases_sha256": stable_hash(shard_rows),
            }
            shard_path = directory / "input_cases.json"
            write_json(shard_path, shard_payload)
            shard_suite = deepcopy(resolved)
            needed = {row["track"] for row in shard_rows}
            shard_suite.tracks = deepcopy(
                [track for name, track in tracks.items() if name in needed]
            )
            # Domain overrides have already been applied by evaluation_tracks().
            for track in shard_suite.tracks:
                if track.domain is not None:
                    track.domain.overrides = {}
            shard_suite.validate_benchmark()
            jobs.append(
                {
                    "suite": shard_suite,
                    "case_manifest": str(shard_path.resolve()),
                    "directory": str(directory.resolve()),
                    "index": index,
                    "device": devices[index % len(devices)],
                    "workers": len(shards),
                    "case_count": len(indices),
                    "quality_batch_size": quality_batch_size,
                }
            )
        write_json(run_dir / "execution.json", execution)
        results: dict[int, BenchmarkRunResult] = {}
        errors = []
        with (
            _worker_environment(worker_threads),
            ProcessPoolExecutor(
                max_workers=len(jobs),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(worker_threads,),
            ) as pool,
        ):
            futures = {pool.submit(_run_shard, job): job["index"] for job in jobs}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result, state = future.result()
                    results[index] = result
                    execution["shards"].append(state)
                except Exception as error:  # noqa: BLE001 - persist worker failures
                    errors.append(f"shard {index}: {type(error).__name__}: {error}")
                    execution["shards"].append(
                        {
                            "shard_index": index,
                            "status": "error",
                            "error": str(error),
                        }
                    )
                write_json(run_dir / "execution.json", execution)
        if errors:
            raise RuntimeError("Parallel benchmark failed: " + "; ".join(errors))
        records, metadata = _merge_records(
            [results[index] for index in sorted(results)],
            rows,
            selected,
            resolved.protocol.measured_trials,
        )
        writer = TrialJsonlWriter(run_dir / "trials.jsonl")
        for record in records:
            writer.append(record)
        aggregates = aggregate_results(
            records,
            metadata,
            cases,
            resolved.protocol.measured_trials,
            track_scenarios={name: track.scenario for name, track in tracks.items()},
            confidence_level=resolved.protocol.confidence_level,
        )
        write_json(
            run_dir / "aggregates.json",
            {
                "report_schema_version": resolved.protocol.report_schema_version,
                **aggregates,
            },
        )
        report = write_markdown_report(
            run_dir / "report.md",
            resolved,
            aggregates,
            notes=[
                "Quality cases ran in isolated spawned processes on shared hardware.",
                "Raw wall times and memory are retained but excluded from timing aggregates; run performance replay serially.",
                "Full-population metrics are recomputed from all raw trials; shard summaries are not averaged.",
                "The frozen corpus and each source case identity are unchanged by sharding.",
                "Planner RNG remains backend-owned. Stochastic solver outcomes may change with case order or sharding; input pairing does not imply identical solver outputs.",
            ],
        )
        execution["status"] = "completed"
        return BenchmarkRunResult(
            run_dir, report, writer.path, tuple(records), aggregates, tuple(metadata)
        )
    except BaseException as error:
        execution.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        execution["elapsed_seconds"] = time.perf_counter() - start
        execution["shards"].sort(key=lambda row: row["shard_index"])
        write_json(run_dir / "execution.json", execution)
