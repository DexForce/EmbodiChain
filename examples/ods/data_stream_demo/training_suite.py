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

"""Supervise visual BC pipelines on disjoint generation/training GPU pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from .suite import (
    _gpu_contexts,
    _gpu_reading,
    _hashes,
    _memory_bytes,
    _terminate,
    _write,
)

__all__ = ["run_suite"]

_ROOT = Path(__file__).resolve().parents[3]
_SETTINGS = (
    "updates",
    "batch_size",
    "buffer_size",
    "producer_envs",
    "episode_budget",
    "history_length",
    "prediction_horizon",
    "action_horizon",
    "max_episode_steps",
    "initialization_timeout",
    "generation_timeout",
    "data_timeout",
    "run_timeout",
)
_AUX_LIMIT = 8 * 1024**2
_LIMITS = [
    "Each primary GPU passes three zero-utilization readings and has at least 16 GiB free; admission is not an exclusive reservation.",
    "One generation/training UUID pair and eight disjoint CPUs are held for each seed/vision group; at most two groups run concurrently.",
    "Outside a pipeline's primary pair, an explicit auxiliary UUID permits only owned pure-G contexts with known per-PID memory <=8 MiB. It may also be an explicitly selected idle primary of another pipeline; per-process-group limits still apply.",
    "Periodic process-tree/process-group GPU audits can miss short-lived contexts; matching native UUID placement still needs qualification.",
    "Total pipeline time ends after normal training-worker exit; independent evaluation is outside that time.",
]


def _devices(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    readings = []
    for index in range(3):
        readings.append({"monotonic_ns": time.monotonic_ns(), "gpus": _gpu_reading()})
        if index < 2:
            time.sleep(2)
    _write(
        args.output / "gpu-preflight.json",
        {"readings": readings, "limitations": _LIMITS},
    )
    by_id = [{gpu["index"]: gpu for gpu in row["gpus"]} for row in readings]
    requested, aux_id = args.gpu_ids, args.aux_graphics_gpu_id
    ids = list(requested) if requested is not None else sorted(by_id[-1])
    if len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError(
            "Primary physical GPU IDs must be distinct nonnegative integers"
        )
    if aux_id is not None and aux_id < 0:
        raise ValueError("Auxiliary GPU must be a nonnegative physical ID")
    selected, rejected = [], []
    for gpu_id in ids:
        if gpu_id == aux_id and requested is None:
            continue
        if all(
            gpu_id in row
            and row[gpu_id]["utilization_percent"] == 0
            and row[gpu_id]["free_mib"] >= 16384
            for row in by_id
        ):
            selected.append(by_id[-1][gpu_id])
        else:
            rejected.append(
                {
                    "index": gpu_id,
                    "reason": "not present, not idle in all three readings, or below 16 GiB free",
                }
            )
    auxiliary = None
    if aux_id is not None:
        if not all(aux_id in row and row[aux_id]["free_mib"] >= 16384 for row in by_id):
            raise RuntimeError(
                "Passive graphics auxiliary GPU requires 16 GiB free in all three readings"
            )
        auxiliary = {
            **by_id[-1][aux_id],
            "scope": "passive pure-G driver footprint only",
            "max_per_pid_bytes": _AUX_LIMIT,
            "idle_required": False,
        }
    cpus = sorted(os.sched_getaffinity(0))
    count = min(2, len(selected) // 2, len(cpus) // 8)
    pairs = [
        {
            "simulation": selected[2 * i],
            "training": selected[2 * i + 1],
            "cpu_affinity": cpus[8 * i : 8 * i + 8],
        }
        for i in range(count)
    ]
    audit = {
        "readings": readings,
        "pairs": pairs,
        "rejected": rejected,
        "aux_graphics_gpu": auxiliary,
        "limitations": _LIMITS,
    }
    _write(args.output / "gpu-preflight.json", audit)
    if not pairs:
        raise RuntimeError(
            "Need two idle primary GPUs with 16 GiB free and eight available CPUs"
        )
    _gpu_contexts()
    return pairs, audit


def _groups(args: argparse.Namespace) -> list[list[dict[str, Any]]]:
    phase = "calibration" if args.suite == "calibrate" else args.suite
    settings = {
        key: getattr(args, key, {"history_length": 2, "action_horizon": 16}.get(key))
        for key in _SETTINGS
    }
    if args.suite == "smoke":
        settings.update(
            buffer_size=2,
            producer_envs=1,
            updates=4,
            batch_size=4,
            episode_budget=4,
            prediction_horizon=32,
        )
    elif args.suite == "calibrate":
        settings.update(
            buffer_size=8,
            producer_envs=2,
            updates=16,
            batch_size=32,
            episode_budget=128,
        )
    seeds = (19001, 19002, 19003) if args.suite == "report" else (args.seed,)
    groups = []
    for seed in seeds:
        for vision in (
            (args.vision,) if args.suite == "single" else ("single", "dual")
        ):
            if args.suite == "single":
                paths = [(args.arm, args.offline_episodes)]
            elif args.suite == "calibrate":
                paths = [("ods_overlap", args.offline_episodes)]
            else:
                counts = (8, 32, 128) if args.suite == "report" else (4,)
                paths = [("offline", n) for n in counts]
                if (seed + (vision == "dual")) % 2:
                    paths.insert(0, ("ods_overlap", args.offline_episodes))
                else:
                    paths.append(("ods_overlap", args.offline_episodes))
                paths.append(("ods_serial", args.offline_episodes))
            jobs = []
            for arm, episodes in paths:
                label = f"offline{episodes}" if arm == "offline" else arm
                output = (
                    args.output
                    if args.suite == "single"
                    else args.output / label / vision / f"seed{seed}"
                )
                trace = (
                    args.trace
                    if args.suite == "single"
                    else (
                        args.output
                        / "ods_overlap"
                        / vision
                        / f"seed{seed}"
                        / "training-trace.json"
                        if arm == "ods_serial"
                        else None
                    )
                )
                scenes = (
                    0
                    if args.suite == "calibrate"
                    else (2 if args.suite == "smoke" else args.evaluation_scenes)
                )
                jobs.append(
                    {
                        "arm": arm,
                        "vision": vision,
                        "seed": seed,
                        "phase": phase,
                        "output": output,
                        "offline_episodes": episodes,
                        "trace": trace,
                        "evaluation_scenes": scenes,
                        "settings": settings.copy(),
                    }
                )
            groups.append(jobs)
    return groups


def _config(job: dict[str, Any]) -> dict[str, Any]:
    return {
        **job["settings"],
        **{
            key: job[key]
            for key in (
                "arm",
                "vision",
                "seed",
                "phase",
                "offline_episodes",
                "evaluation_scenes",
            )
        },
    }


def _mark(job: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    path = job["output"] / "summary.json"
    if path.exists():
        try:
            summary = json.loads(path.read_text())
        except ValueError:
            path.rename(path.with_name("summary.partial.json"))
            summary = {}
    else:
        summary = {}
    summary.setdefault("config", _config(job))
    summary.update(status=status, qualification_passed=False, supervisor_error=reason)
    summary.setdefault("error", {"phase": "supervisor", "message": reason})
    _write(path, summary)
    return summary


def _command(job: dict[str, Any], stage: str = "training") -> list[str]:
    options: dict[str, Any] = {}
    if stage == "audit":
        module = "examples.ods.data_stream_demo.training_audit"
        options.update(run_dir=job["output"], output=job["output"] / "post-audit.json")
    else:
        options.update(vision=job["vision"], sim_device="cuda:0", train_device="cuda:1")
        if stage == "evaluation":
            module = "examples.ods.data_stream_demo.training_eval"
            options.update(
                checkpoint=job["output"] / "final.pt",
                output=job["output"] / "evaluation.json",
                scenes=job["evaluation_scenes"],
            )
        else:
            module = "examples.ods.data_stream_demo.train"
            options.update(
                suite="single",
                phase=job["phase"],
                arm=job["arm"],
                seed=job["seed"],
                output=job["output"],
                offline_episodes=job["offline_episodes"],
                evaluation_scenes=0,
                **job["settings"],
            )
            if job["trace"] is not None:
                options["trace"] = job["trace"]
    command = [sys.executable, "-m", module] + (
        ["--worker"] if stage == "training" else []
    )
    for name, value in options.items():
        command += ["--" + name.replace("_", "-"), str(value)]
    return command


def _trace_ready(job: dict[str, Any]) -> bool:
    if job["trace"] is None or not Path(job["trace"]).is_file():
        return False
    if job["phase"] == "single":
        return True
    try:
        original = json.loads(Path(job["trace"]).with_name("summary.json").read_text())
        return (
            original.get("status") == "complete"
            and original.get("qualification_passed") is True
        )
    except (OSError, ValueError):
        return False


def _launch(
    slot: dict[str, Any],
    job: dict[str, Any],
    root: Path,
    auxiliary: dict[str, Any] | None,
    *,
    stage: str = "training",
) -> dict[str, Any]:
    pair = slot["pair"]
    relative = (
        Path("single") if job["output"] == root else job["output"].relative_to(root)
    )
    supervision = root / "_supervision" / relative / stage
    supervision.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=pair["simulation"]["uuid"]
        + ","
        + pair["training"]["uuid"],
        ODS_GPU_UUID=pair["simulation"]["uuid"],
        SDL_VIDEODRIVER="dummy",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "ODS_AUX_GRAPHICS_GPU_UUID"):
        environment.pop(name, None)
    if auxiliary is not None:
        environment["ODS_AUX_GRAPHICS_GPU_UUID"] = auxiliary["uuid"]
    command = _command(job, stage)
    stream = (supervision / "process.log").open("w", encoding="utf-8")
    started_ns = time.monotonic_ns()
    environment["ODS_PROCESS_LAUNCH_MONOTONIC_NS"] = str(started_ns)
    try:
        process = subprocess.Popen(
            command,
            cwd=_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=lambda: os.sched_setaffinity(0, pair["cpu_affinity"]),
        )
    except BaseException:
        stream.close()
        raise
    launch = {
        "command": command,
        "pair": pair,
        "aux_graphics_scope": auxiliary,
        "process_launch_monotonic_ns": started_ns,
        "pgid": process.pid,
        "cpu_affinity": pair["cpu_affinity"],
        "timed_pipeline": stage == "training",
        "environment": {
            key: environment.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "ODS_GPU_UUID",
                "ODS_AUX_GRAPHICS_GPU_UUID",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "SDL_VIDEODRIVER",
                "ODS_PROCESS_LAUNCH_MONOTONIC_NS",
            )
        },
    }
    _write(supervision / "launch.json", launch)
    print(
        f"Started {job['phase']} {job['arm']}/{job['vision']}/seed{job['seed']} {stage} on GPUs {pair['simulation']['index']},{pair['training']['index']}",
        flush=True,
    )
    return {
        "process": process,
        "stream": stream,
        "started_ns": started_ns,
        "deadline": started_ns / 1e9 + float(job["settings"]["run_timeout"]),
        "supervision": supervision,
        "evaluation": stage == "evaluation",
        "stage": stage,
        "job": job,
        "allocation": pair,
        "auxiliary_scope": auxiliary,
        "guard": {
            "samples": 0,
            "unexpected_contexts": [],
            "allowed_aux_contexts": [],
            "seen_primary_uuids": [],
        },
        "error": None,
    }


def _audit(slots: list[dict[str, Any]], auxiliary: dict[str, Any] | None) -> None:
    contexts = _gpu_contexts()
    for slot in slots:
        item = slot.get("active")
        if item is None:
            continue
        process = item["process"]
        owners = {process.pid}
        try:
            owners.update(
                child.pid
                for child in psutil.Process(process.pid).children(recursive=True)
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        owned = []
        for row in contexts:
            try:
                matches = row["pid"] in owners or os.getpgid(row["pid"]) == process.pid
            except ProcessLookupError:
                matches = False
            if matches:
                owned.append(row)
        primary = {slot["pair"][role]["uuid"] for role in ("simulation", "training")}
        aux_uuid = auxiliary["uuid"] if auxiliary else None
        invalid_aux = {
            row["pid"]
            for row in owned
            if row["gpu_uuid"] == aux_uuid
            and (
                row.get("type") != "G"
                or _memory_bytes(row.get("used_memory")) is None
                or _memory_bytes(row.get("used_memory")) > _AUX_LIMIT
            )
        }
        allowed, unexpected = [], []
        for row in owned:
            if item["stage"] == "audit":
                unexpected.append(row)
            elif row["gpu_uuid"] in primary:
                if row["gpu_uuid"] not in item["guard"]["seen_primary_uuids"]:
                    item["guard"]["seen_primary_uuids"].append(row["gpu_uuid"])
            elif (
                aux_uuid
                and row["gpu_uuid"] == aux_uuid
                and row["pid"] not in invalid_aux
            ):
                allowed.append(row)
            else:
                unexpected.append(row)
        item["guard"]["samples"] += 1
        for row in allowed:
            if row not in item["guard"]["allowed_aux_contexts"]:
                item["guard"]["allowed_aux_contexts"].append(row)
        if unexpected:
            item["guard"]["unexpected_contexts"].extend(unexpected)
            item["error"] = (
                f"Owned GPU placement violates pair/passive scope: {unexpected}"
            )
        with (item["supervision"] / "device-contexts.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    {
                        "monotonic_ns": time.monotonic_ns(),
                        "owned_contexts": owned,
                        "allowed_aux_contexts": allowed,
                        "unexpected_contexts": unexpected,
                    },
                    allow_nan=False,
                )
                + "\n"
            )


def _group_snapshot(pgid: int) -> dict[str, Any]:
    """Capture identities only for members of this owned process group."""
    result: dict[str, Any] = {
        "monotonic_ns": time.monotonic_ns(),
        "pgid": pgid,
        "exists": False,
        "members": [],
        "inspection_errors": [],
    }
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return result
    except PermissionError as error:
        result["inspection_errors"].append(repr(error))
    result["exists"] = True
    for member in psutil.process_iter():
        try:
            if os.getpgid(member.pid) != pgid:
                continue
        except ProcessLookupError:
            continue
        except PermissionError as error:
            result["inspection_errors"].append(
                {"pid": member.pid, "error": repr(error)}
            )
            continue
        row: dict[str, Any] = {"pid": member.pid}
        try:
            with member.oneshot():
                row.update(
                    ppid=member.ppid(),
                    create_time=member.create_time(),
                    name=member.name(),
                    state=member.status(),
                )
                try:
                    row["cmdline"] = member.cmdline()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
                    row["cmdline_error"] = repr(error)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            row["inspection_error"] = repr(error)
        result["members"].append(row)
    result["members"].sort(key=lambda row: row["pid"])
    states = [row.get("state") for row in result["members"]]
    result["member_kind"] = (
        "unresolved"
        if result["inspection_errors"]
        or any("inspection_error" in row for row in result["members"])
        else (
            "zombie_only"
            if states and all(state == psutil.STATUS_ZOMBIE for state in states)
            else "live_or_unknown" if states else "unresolved"
        )
    )
    return result


def _await_group_exit(pgid: int, *, timeout: float) -> dict[str, Any]:
    """Allow brief normal helper cleanup; any group at the deadline fails."""
    started = time.monotonic()
    initial = _group_snapshot(pgid)
    latest = initial
    polls = 1
    while latest["exists"] and time.monotonic() < started + timeout:
        time.sleep(min(0.05, max(0.0, started + timeout - time.monotonic())))
        latest = _group_snapshot(pgid)
        polls += 1
    return {
        "grace_seconds": timeout,
        "wait_seconds": time.monotonic() - started,
        "poll_count": polls,
        "group_disappeared": not latest["exists"],
        "initial": initial,
        "final": latest,
    }


def _finish(
    item: dict[str, Any], *, cancelled: bool
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    process, job = item["process"], item["job"]
    timeout = time.monotonic() >= item["deadline"]
    reason = item["error"] or (
        "suite cancellation"
        if cancelled
        else "process deadline exceeded" if timeout else None
    )
    if process.poll() is None and reason is None:
        return None
    if reason is not None:
        _terminate(process)
    process.wait()
    shutdown = _await_group_exit(process.pid, timeout=2.0)
    leaked = not shutdown["group_disappeared"]
    if leaked:
        kind = shutdown["final"].get("member_kind", "unresolved")
        reason = reason or (
            "Owned process group remains after normal-exit grace "
            f"({kind}; see process_group_shutdown identities)"
        )
        # A zombie cannot execute work or respond to signals. Its presence still
        # fails the group-empty boundary; classification grants no exemption.
        if kind != "zombie_only":
            _terminate(process)
            shutdown["termination_attempted"] = True
            shutdown["after_termination"] = _await_group_exit(process.pid, timeout=2.0)
        else:
            shutdown["termination_attempted"] = False
    ended_ns = time.monotonic_ns()
    item["stream"].close()
    elapsed = (ended_ns - item["started_ns"]) / 1e9
    if process.returncode != 0:
        reason = reason or f"Child exit={process.returncode}"
    stage = item["stage"]
    artifact = {
        "training": "summary.json",
        "audit": "post-audit.json",
        "evaluation": "evaluation.json",
    }[stage]
    path = job["output"] / artifact
    if not path.exists():
        reason = reason or f"Child omitted {path.name}"
    if stage != "training":
        summary = json.loads((job["output"] / "summary.json").read_text())
        if reason is None:
            payload = json.loads(path.read_text())
            valid = payload.get("passed") is True
            if stage == "evaluation":
                valid = (
                    valid
                    and payload.get("closed") is True
                    and payload.get("scenes") == job["evaluation_scenes"]
                    and len(payload.get("records", [])) == job["evaluation_scenes"]
                )
            if not valid:
                reason = f"Post-timing {stage} did not complete its contract"
            else:
                summary["post_audit" if stage == "audit" else "evaluation"] = payload
        key = "post_audit" if stage == "audit" else "evaluation"
        summary[key + "_status"] = "failed" if reason else "complete"
        summary[key + "_seconds_outside_pipeline"] = elapsed
        if reason:
            summary.update(
                quality_error=reason, qualification_passed=False, status="failed"
            )
    else:
        summary = (
            _mark(job, "failed", reason) if reason else json.loads(path.read_text())
        )
        if (
            summary.get("status") == "complete"
            and summary.get("qualification_passed") is not True
        ):
            summary = _mark(job, "failed", "Worker did not pass training qualification")
        metrics = summary.get("training_metrics")
        if isinstance(metrics, dict):
            train_end = float(metrics.get("training_complete_seconds", 0))
            if train_end > elapsed:
                summary = _mark(
                    job, "failed", "Child training timestamps exceed supervisor total"
                )
                metrics = summary.get("training_metrics", {})
            metrics.update(
                total_seconds=elapsed, post_training_drain_seconds=elapsed - train_end
            )
        summary["supervisor_total_seconds"] = elapsed
        summary["requested_evaluation_scenes"] = job["evaluation_scenes"]
        summary["post_audit_status"] = "pending"
        summary["evaluation_status"] = (
            "pending" if job["evaluation_scenes"] else "not_requested"
        )
        summary["device_supervision"] = {
            "primary_pair": item["allocation"],
            "aux_graphics_scope": item["auxiliary_scope"],
            "guard": item["guard"],
        }
    result = {
        "output": str(job["output"]),
        "status": summary["status"],
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "stage": stage,
        "evaluation": item["evaluation"],
        "timeout": timeout,
        "cancelled": cancelled,
        "leaked_process_group": leaked,
        "process_group_shutdown": shutdown,
        "pgid": process.pid,
        "process_launch_monotonic_ns": item["started_ns"],
        "device_guard": item["guard"],
    }
    process_artifact = (
        "process.json" if stage == "training" else stage + "-process.json"
    )
    _write(path.parent / process_artifact, result)
    _write(job["output"] / "summary.json", summary)
    return summary, result


def _calibration(
    root: Path, results: list[dict[str, Any]], source_hashes: dict[str, str]
) -> dict[str, Any]:
    observations = []
    for result in results:
        summary = json.loads((Path(result["output"]) / "summary.json").read_text())
        metrics = summary.get("training_metrics", {})
        events = (
            [
                json.loads(line)
                for line in (Path(result["output"]) / "training-events.jsonl")
                .read_text()
                .splitlines()
            ]
            if (Path(result["output"]) / "training-events.jsonl").exists()
            else []
        )
        training = [row for row in events if row.get("kind") == "update_completed"]
        seconds = sum(
            row["time_seconds"] - row["train_started_seconds"] for row in training
        )
        spans = [
            (row["start_seconds"], row["end_seconds"])
            for row in metrics.get("intervals", [])
            if any(
                name in row.get("name", "")
                for name in ("generation", "initial_fill", "acquisition")
            )
        ]
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        generation_seconds = sum(end - start for start, end in merged)
        observations.append(
            {
                "vision": summary["config"]["vision"],
                "status": summary["status"],
                "accepted_episode_rate": (
                    metrics.get("published_sources", 0) / generation_seconds
                    if generation_seconds
                    else None
                ),
                "learner_updates_per_second": (
                    len(training) / seconds if seconds else None
                ),
                "generation_observed_seconds": generation_seconds,
                "training_observed_seconds": seconds,
                "updates": len(training),
                "output": result["output"],
            }
        )
    packet = {
        "schema": "ods-visual-training-calibration/v1",
        "observations": observations,
        "source_sha256": source_hashes,
        "limits": "Preflight observations, not formal results or optimized reuse settings; host stage spans do not alone prove simultaneous GPU kernels.",
    }
    _write(root / "calibration.json", packet)
    return packet


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Run a finite matrix with normal-exit timing and separately timed evaluation."""
    args.output = Path(args.output).resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            "Use a fresh output directory for every training experiment"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    groups = _groups(args)
    source_hashes = _hashes()
    for path in _ROOT.joinpath("embodichain/learning/models").glob("*.py"):
        source_hashes[str(path.relative_to(_ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    results, cancelled = [], False
    try:
        pairs, audit = _devices(args)
    except Exception as error:
        for group in groups:
            for job in group:
                _mark(job, "unsupported", f"Device admission failed: {error!r}")
                results.append({"output": str(job["output"]), "status": "unsupported"})
        summary = {
            "status": "unsupported",
            "error": repr(error),
            "results": results,
            "source_sha256": source_hashes,
        }
        _write(args.output / "suite-summary.json", summary)
        return summary
    slots = [{"pair": pair, "jobs": [], "active": None} for pair in pairs]

    def cancel(signum: int, frame: Any) -> None:
        nonlocal cancelled
        cancelled = True

    previous = {
        sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    last_guard = last_progress = 0.0
    try:
        while groups or any(slot["jobs"] or slot["active"] for slot in slots):
            for slot in slots:
                if slot["active"] is None and not cancelled:
                    if not slot["jobs"] and groups:
                        slot["jobs"] = groups.pop(0)
                    if slot["jobs"]:
                        job = slot["jobs"].pop(0)
                        if job["arm"] == "ods_serial" and not _trace_ready(job):
                            _mark(
                                job,
                                "unsupported",
                                "Matching overlap training trace is unavailable",
                            )
                            results.append(
                                {"output": str(job["output"]), "status": "unsupported"}
                            )
                            continue
                        try:
                            slot["active"] = _launch(
                                slot, job, args.output, audit["aux_graphics_gpu"]
                            )
                        except Exception as error:
                            _mark(job, "failed", f"Launch failed: {error!r}")
                            results.append(
                                {"output": str(job["output"]), "status": "failed"}
                            )
            now = time.monotonic()
            if now - last_guard >= 2 and any(slot["active"] for slot in slots):
                try:
                    _audit(slots, audit["aux_graphics_gpu"])
                except Exception as error:
                    for slot in slots:
                        if slot["active"]:
                            slot["active"][
                                "error"
                            ] = f"Cannot audit GPU placement: {error!r}"
                last_guard = now
            for slot in slots:
                item = slot["active"]
                if item is None:
                    continue
                try:
                    finished = _finish(item, cancelled=cancelled)
                except Exception as error:
                    _terminate(item["process"])
                    item["stream"].close()
                    failed = _mark(
                        item["job"], "failed", f"Invalid child artifacts: {error!r}"
                    )
                    finished = failed, {
                        "output": str(item["job"]["output"]),
                        "status": "failed",
                        "evaluation": item["evaluation"],
                        "error": repr(error),
                    }
                if finished is None:
                    continue
                summary, result = finished
                job = item["job"]
                if summary.get("status") == "complete" and not cancelled:
                    next_stage = (
                        "audit"
                        if item["stage"] == "training"
                        else (
                            "evaluation"
                            if item["stage"] == "audit" and job["evaluation_scenes"] > 0
                            else None
                        )
                    )
                    if next_stage is not None:
                        try:
                            slot["active"] = _launch(
                                slot,
                                job,
                                args.output,
                                audit["aux_graphics_gpu"],
                                stage=next_stage,
                            )
                            slot["active"]["pipeline_result"] = (
                                result
                                if item["stage"] == "training"
                                else item["pipeline_result"]
                            )
                            if item["stage"] == "audit":
                                slot["active"]["audit_result"] = result
                            continue
                        except Exception as error:
                            summary.update(
                                status="failed",
                                qualification_passed=False,
                                quality_error=f"{next_stage} launch failed: {error!r}",
                            )
                            _write(job["output"] / "summary.json", summary)
                            result["status"] = "failed"
                if item["stage"] != "training":
                    result = {
                        **item["pipeline_result"],
                        item["stage"] + "_process": result,
                        "status": summary["status"],
                    }
                    if "audit_result" in item:
                        result["audit_process"] = item["audit_result"]
                results.append({**result, "status": summary["status"]})
                slot["active"] = None
                print(
                    f"Finished {job['arm']}/{job['vision']}/seed{job['seed']}: {summary['status']}",
                    flush=True,
                )
            if cancelled:
                for group in groups + [slot["jobs"] for slot in slots]:
                    for job in group:
                        _mark(job, "failed", "Suite cancelled before launch")
                        results.append(
                            {"output": str(job["output"]), "status": "failed"}
                        )
                groups.clear()
                for slot in slots:
                    slot["jobs"].clear()
            if now - last_progress >= 30:
                print(
                    f"Visual suite progress: {len(results)} finished; {sum(slot['active'] is not None for slot in slots)} active; {len(groups)} pending seed/vision groups",
                    flush=True,
                )
                last_progress = now
            if groups or any(slot["jobs"] or slot["active"] for slot in slots):
                time.sleep(0.25)
    finally:
        for slot in slots:
            if slot["active"]:
                _terminate(slot["active"]["process"])
                slot["active"]["stream"].close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = {
        "status": (
            "complete"
            if all(result["status"] == "complete" for result in results)
            else "failed"
        ),
        "phase": args.suite,
        "results": results,
        "gpu_preflight": audit,
        "source_sha256": source_hashes,
        "limitations": _LIMITS,
    }
    _write(args.output / "suite-summary.json", summary)
    if args.suite == "calibrate":
        summary["calibration"] = _calibration(args.output, results, source_hashes)
    elif args.suite == "report":
        from .training_report import write_report

        summary["report"] = write_report(args.output, args.output / "training-report")
    _write(args.output / "suite-summary.json", summary)
    return summary
