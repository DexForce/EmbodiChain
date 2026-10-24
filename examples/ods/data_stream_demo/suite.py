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

"""Supervise finite real-data matrices with disjoint CPU/GPU assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import psutil

from .acquisition import CAMERA_PROFILE_VERSION
from .report import write_report

__all__ = ["run_suite"]

_ROOT = Path(__file__).resolve().parents[3]
_VISIONS = ("state", "single", "dual")
_MAIN = ("lerobot", "streaming", "protected")
_SEEDS = (19001, 19002, 19003)
_ORDER_SEED = 47001
_AUX_MAX_BYTES = 8 * 1024**2
_HEADLESS_ENV = {"SDL_VIDEODRIVER": "dummy", "DISPLAY": None, "WAYLAND_DISPLAY": None}
_BASE = {
    "buffer_size": 8,
    "producer_envs": 2,
    "accepted_episodes": 32,
    "max_episode_steps": 600,
    "batch_size": 32,
    "history_length": 2,
    "prediction_horizon": 32,
    "episode_budget": 128,
    "num_workers": 4,
    "prefetch_factor": 1,
    "data_timeout": 120.0,
    "generation_timeout": 1800.0,
    "initialization_timeout": 1800.0,
}
_LIMITATIONS = [
    "Select a physical NVML GPU UUID, set CUDA_VISIBLE_DEVICES and ODS_GPU_UUID to that UUID, and use logical cuda:0.",
    "The matching native runtime must qualify CUDA UUID to Vulkan UUID mapping; no fallback to an unmasked or unreserved GPU is permitted.",
    "Selected GPUs have three idle readings and at least 16GiB free. This is not an exclusive-device reservation.",
    "An optional auxiliary graphics GPU is separately reserved and excluded from primary assignments; only known per-PID pure-G footprints up to 8 MiB are allowed there.",
    "The parent samples compute and graphics contexts attributed to its child PID tree or process group; short-lived contexts can fall between samples.",
    "No system-wide file cache is purged. Calibration is a preflight observation, not a guaranteed concurrent capacity.",
]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _hashes() -> dict[str, str]:
    paths = [
        *Path(__file__).parent.glob("*.py"),
        _ROOT / "examples/ods/simple_bc_demo/config.py",
        _ROOT / "examples/ods/simple_bc_demo/environment.py",
        *_ROOT.joinpath("examples/ods/simple_bc_demo/configs").rglob("*.yml"),
        *_ROOT.joinpath("examples/ods/simple_bc_demo/configs").rglob("*.yaml"),
        _ROOT / "embodichain/data_pipeline/engine/data.py",
        _ROOT / "embodichain/data_pipeline/runtime.py",
    ]
    return {
        str(path.relative_to(_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def _gpu_reading() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    rows = []
    for line in output.splitlines():
        index, uuid, name, utilization, free, total = (
            part.strip() for part in line.split(",")
        )
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "utilization_percent": int(utilization),
                "free_mib": int(free),
                "total_mib": int(total),
            }
        )
    return rows


def _gpu_contexts() -> list[dict[str, Any]]:
    """Read compute and graphics contexts without acting on foreign processes."""
    output = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=10)
    document = ET.fromstring(output)
    contexts = []
    gpus = document.findall("gpu")
    if not gpus:
        raise RuntimeError("nvidia-smi XML did not contain GPU identity records")
    for gpu in gpus:
        uuid = gpu.findtext("uuid")
        if not uuid:
            raise RuntimeError("nvidia-smi XML omitted a GPU UUID")
        for process in gpu.findall("./processes/process_info"):
            pid = process.findtext("pid", "")
            if pid.isdigit():
                contexts.append(
                    {
                        "pid": int(pid),
                        "gpu_uuid": uuid,
                        "type": process.findtext("type"),
                        "used_memory": process.findtext("used_memory"),
                    }
                )
    return contexts


def _memory_bytes(value: str | None) -> int | None:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(B|KiB|MiB|GiB)", value or "")
    if match is None:
        return None
    scale = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[match[2]]
    size = float(match[1]) * scale
    return math.ceil(size) if math.isfinite(size) else None


def _aux_scope(auxiliary: dict[str, Any] | None) -> dict[str, Any] | None:
    if auxiliary is None:
        return None
    return {
        "physical_index": auxiliary["index"],
        "uuid": auxiliary["uuid"],
        "model": auxiliary["name"],
        "allowed_type": "G",
        "max_per_pid_bytes": _AUX_MAX_BYTES,
        "shared_by_suite_jobs": True,
    }


def _partition_contexts(
    owned: list[dict[str, Any]], selected: str, auxiliary: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    totals: dict[int, int] = {}
    invalid = set()
    for row in owned:
        if row["gpu_uuid"] != auxiliary:
            continue
        memory = _memory_bytes(row.get("used_memory"))
        if row.get("type") != "G" or memory is None:
            invalid.add(row["pid"])
        else:
            totals[row["pid"]] = max(totals.get(row["pid"], 0), memory)
    allowed, unexpected = [], []
    for row in owned:
        if row["gpu_uuid"] == selected:
            continue
        if (
            auxiliary is not None
            and row["gpu_uuid"] == auxiliary
            and row["pid"] not in invalid
            and totals.get(row["pid"], _AUX_MAX_BYTES + 1) <= _AUX_MAX_BYTES
        ):
            allowed.append(row)
        else:
            unexpected.append(row)
    return allowed, unexpected


def _guard_devices(active: dict[int, dict[str, Any]]) -> None:
    """Enforce primary placement and the optional bounded reserved graphics scope."""
    contexts = _gpu_contexts()
    owners = {}
    groups = {}
    for index, item in active.items():
        process = item["process"]
        groups[process.pid] = index
        owners[process.pid] = index
        try:
            for child in psutil.Process(process.pid).children(recursive=True):
                owners[child.pid] = index
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    attributed: dict[int, list[dict[str, Any]]] = {index: [] for index in active}
    for context in contexts:
        owner = owners.get(context["pid"])
        if owner is None:
            try:
                owner = groups.get(os.getpgid(context["pid"]))
            except ProcessLookupError:
                continue
        if owner is not None:
            attributed[owner].append(context)
    now = time.monotonic()
    for index, item in active.items():
        owned = attributed[index]
        allowed, unexpected = _partition_contexts(
            owned, item["gpu_uuid"], item.get("aux_gpu_uuid")
        )
        for row in allowed:
            if row not in item["device_guard"]["allowed_aux_contexts"]:
                item["device_guard"]["allowed_aux_contexts"].append(row)
        memory_by_process: dict[tuple[int, str], int | None] = {}
        for row in owned:
            key = (row["pid"], row["gpu_uuid"])
            memory_by_process.setdefault(key, None)
            value = _memory_bytes(row.get("used_memory"))
            if value is not None:
                memory_by_process[key] = max(memory_by_process[key] or 0, value)
        memories = list(memory_by_process.values())
        total_memory = (
            sum(memories) if all(value is not None for value in memories) else None
        )
        if total_memory is not None:
            item["device_guard"]["peak_known_owned_context_bytes"] = max(
                item["device_guard"].get("peak_known_owned_context_bytes", 0),
                total_memory,
            )
        item["device_guard"]["samples"] += 1
        item["device_guard"]["selected_context_seen"] |= any(
            row["gpu_uuid"] == item["gpu_uuid"] for row in owned
        )
        if unexpected:
            item["device_guard"]["unexpected_contexts"].extend(unexpected)
            item["device_error"] = (
                f"Owned GPU contexts violate selected UUID {item['gpu_uuid']}: {unexpected}"
            )
        with (item["supervision"] / "device-contexts.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    {
                        "seconds_from_launch": now - item["started"],
                        "selected_gpu_uuid": item["gpu_uuid"],
                        "aux_graphics_gpu_uuid": item.get("aux_gpu_uuid"),
                        "owned_contexts": owned,
                        "owned_context_memory_bytes": total_memory,
                        "allowed_aux_contexts": allowed,
                        "unexpected_contexts": unexpected,
                    }
                )
                + "\n"
            )


def _select_devices(
    requested: list[int] | None, output: Path, *, auxiliary_id: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    readings = []
    for index in range(3):
        readings.append({"monotonic_seconds": time.monotonic(), "gpus": _gpu_reading()})
        if index < 2:
            time.sleep(2)
    by_id = [{row["index"]: row for row in reading["gpus"]} for reading in readings]
    _write(
        output / "gpu-preflight.json",
        {"readings": readings, "limitations": _LIMITATIONS},
    )
    auxiliary = None
    if auxiliary_id is not None:
        if auxiliary_id < 0 or requested is not None and auxiliary_id in requested:
            raise ValueError(
                "The reserved auxiliary GPU must be nonnegative and distinct from every primary GPU"
            )
        if not all(
            auxiliary_id in reading
            and reading[auxiliary_id]["utilization_percent"] == 0
            and reading[auxiliary_id]["free_mib"] >= 16 * 1024
            for reading in by_id
        ):
            raise RuntimeError(
                "Auxiliary GPU requires three idle readings and at least 16 GiB free"
            )
        auxiliary = by_id[-1][auxiliary_id]
    ids = (
        requested
        if requested is not None
        else [index for index in sorted(by_id[-1]) if index != auxiliary_id]
    )
    if len(set(ids)) != len(ids) or any(index < 0 for index in ids):
        raise ValueError("GPU IDs must be unique nonnegative physical indices")
    selected, rejected = [], []
    for index in ids:
        if all(
            index in reading
            and reading[index]["utilization_percent"] == 0
            and reading[index]["free_mib"] >= 16 * 1024
            for reading in by_id
        ):
            selected.append(by_id[-1][index])
        else:
            rejected.append(
                {
                    "index": index,
                    "reason": "not present, not idle in all readings, or less than 16GiB free",
                }
            )
    cpus = sorted(os.sched_getaffinity(0))
    for gpu in selected[len(cpus) // 8 :]:
        rejected.append(
            {"index": gpu["index"], "reason": "not enough disjoint eight-CPU slots"}
        )
    selected = selected[: len(cpus) // 8]
    if not selected:
        raise RuntimeError("No idle GPU with 16GiB free and eight available CPUs")
    for index, gpu in enumerate(selected):
        gpu["cpu_affinity"] = cpus[index * 8 : (index + 1) * 8]
    audit = {
        "readings": readings,
        "selected": selected,
        "aux_graphics_gpu": auxiliary,
        "aux_graphics_scope": _aux_scope(auxiliary),
        "rejected": rejected,
        "cpu_budget_per_job": 8,
        "limitations": _LIMITATIONS,
    }
    _write(output / "gpu-preflight.json", audit)
    return selected, audit


def _protocol(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("Report and appendix require --protocol calibration.json")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "complete"
        or protocol.get("format") != "ods-stream-protocol-v1"
    ):
        raise ValueError("A complete, qualified calibration protocol is required")
    if (
        protocol.get("source_sha256") != _hashes()
        or protocol.get("camera_profile_version") != CAMERA_PROFILE_VERSION
    ):
        raise ValueError(
            "Code/profile differs from calibration; recalibrate under the new profile"
        )
    for name in ("request_rate", "duration_seconds"):
        value = protocol.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid calibrated {name}")
    if protocol["duration_seconds"] > 900:
        raise ValueError(
            "Calibrated observation window exceeds the 900-second protocol ceiling"
        )
    if protocol.get("headless_environment") != _HEADLESS_ENV:
        raise ValueError("Headless environment differs from the qualified protocol")
    return protocol


def _jobs(
    args: argparse.Namespace, protocol: dict[str, Any] | None
) -> list[dict[str, Any]]:
    base = dict(_BASE if protocol is None else protocol["base_settings"])
    suite = args.suite
    if suite == "smoke":
        base.update(
            buffer_size=2,
            producer_envs=2,
            accepted_episodes=8,
            batch_size=4,
            episode_budget=8,
        )
        rate, duration = 16.0, 30.0
        combinations = [
            (arm, vision, _SEEDS[0]) for arm in _MAIN for vision in _VISIONS
        ]
    elif suite == "calibrate":
        rate, duration = 1e9, 60.0
        combinations = [
            (arm, vision, _SEEDS[0])
            for arm in ("protected", "lerobot_unbounded")
            for vision in _VISIONS
        ]
    else:
        rate, duration = protocol["request_rate"], protocol["duration_seconds"]
        combinations = (
            [
                (arm, vision, seed)
                for seed in _SEEDS
                for arm in _MAIN
                for vision in _VISIONS
            ]
            if suite == "report"
            else [
                (arm, vision, _SEEDS[0])
                for arm in ("frozen", "lerobot_unbounded", "lerobot_same_cache")
                for vision in _VISIONS
            ]
        )
    jobs = []
    for arm, vision, seed in combinations:
        destination = args.output / arm / vision / f"seed{seed}"
        corpus = None
        if suite == "appendix" and arm.startswith("lerobot"):
            corpus = (
                args.dataset_root
                / "lerobot"
                / vision
                / f"seed{seed}"
                / "lerobot_dataset"
            )
        jobs.append(
            {
                "arm": arm,
                "vision": vision,
                "seed": seed,
                "output": destination,
                "dataset_root": corpus,
                "settings": base,
                "rate": rate,
                "duration": duration,
                "suite": "calibration" if suite == "calibrate" else suite,
                "calibration": suite == "calibrate",
            }
        )
    random.Random(_ORDER_SEED).shuffle(jobs)
    return jobs


def _command(job: dict[str, Any], gpu: dict[str, Any]) -> list[str]:
    command = [
        "taskset",
        "--cpu-list",
        ",".join(map(str, gpu["cpu_affinity"])),
        sys.executable,
        "-m",
        "examples.ods.data_stream_demo.run",
        "--suite",
        "single",
        "--suite-tag",
        job["suite"],
        "--arm",
        job["arm"],
        "--vision",
        job["vision"],
        "--seed",
        str(job["seed"]),
        "--output",
        str(job["output"]),
        "--sim-device",
        "cuda:0",
        "--request-rate",
        str(job["rate"]),
        "--duration-seconds",
        str(job["duration"]),
    ]
    for name, value in job["settings"].items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    if job["dataset_root"] is not None:
        command.extend(["--dataset-root", str(job["dataset_root"])])
    if job["calibration"]:
        command.append("--calibration-run")
    return command


def _deadline(job: dict[str, Any]) -> float:
    settings = job["settings"]
    generations = settings["accepted_episodes"] // settings["buffer_size"]
    return float(
        settings["initialization_timeout"]
        + generations * settings["generation_timeout"]
        + job["duration"]
        + 1800
        + 2 * settings["data_timeout"]
        + 180
    )


def _mark(job: dict[str, Any], status: str, reason: str) -> None:
    path = job["output"] / "summary.json"
    if path.exists():
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            (path.parent / "summary.partial.json").write_bytes(path.read_bytes())
            summary = {}
    else:
        summary = {}
    summary.setdefault(
        "config",
        {
            **job["settings"],
            "arm": job["arm"],
            "vision": job["vision"],
            "seed": job["seed"],
            "suite": job["suite"],
            "cache_episodes": job["settings"]["buffer_size"],
            "quota": (
                None
                if job["arm"] in {"streaming", "lerobot_unbounded"}
                else job["settings"]["episode_budget"]
            ),
            "requested_chunks_per_second": job["rate"],
            "measurement_seconds": job["duration"],
        },
    )
    summary["status"] = status
    if not summary.get("error"):
        summary["error"] = {"phase": "supervisor", "message": reason}
    summary["supervisor_error"] = reason
    summary.setdefault("metrics", {})["qualification_passed"] = False
    summary.setdefault("episode_usage", [])
    _write(path, summary)


def _terminate(process: subprocess.Popen[Any]) -> None:
    for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
    if process.poll() is None:
        raise TimeoutError(f"Owned process group {process.pid} did not terminate")


def _supervise(
    jobs: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    auxiliary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    pending = list(jobs)
    active: dict[int, dict[str, Any]] = {}
    results = []
    cancelled = False
    last_guard = 0.0

    def cancel(signum: int, frame: Any) -> None:
        nonlocal cancelled
        cancelled = True

    previous = {
        sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        while pending or active:
            for gpu in devices:
                if not pending or gpu["index"] in active or cancelled:
                    continue
                job = pending.pop(0)
                if job["dataset_root"] is not None and not job["dataset_root"].is_dir():
                    reason = f"Required finalized corpus missing: {job['dataset_root']}"
                    _mark(job, "unsupported", reason)
                    results.append(
                        {
                            "output": str(job["output"]),
                            "status": "unsupported",
                            "reason": reason,
                        }
                    )
                    continue
                # Keep supervision files outside the child's initially empty
                # output directory; run_one deliberately rejects nonempty runs.
                supervision = (
                    job["output"].parents[2]
                    / "_supervision"
                    / job["output"].relative_to(job["output"].parents[2])
                )
                supervision.mkdir(parents=True)
                command = _command(job, gpu)
                environment = os.environ.copy()
                for name, value in _HEADLESS_ENV.items():
                    if value is None:
                        environment.pop(name, None)
                    else:
                        environment[name] = value
                environment.pop("ODS_AUX_GRAPHICS_GPU_UUID", None)
                if auxiliary is not None:
                    environment["ODS_AUX_GRAPHICS_GPU_UUID"] = auxiliary["uuid"]
                environment.update(
                    CUDA_VISIBLE_DEVICES=gpu["uuid"],
                    ODS_GPU_UUID=gpu["uuid"],
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    ODS_CPU_BUDGET="8",
                )
                launch = {
                    "command": command,
                    "cwd": str(_ROOT),
                    "gpu": gpu,
                    "aux_graphics_scope": _aux_scope(auxiliary),
                    "environment": {
                        name: environment.get(name)
                        for name in (
                            "CUDA_VISIBLE_DEVICES",
                            "ODS_GPU_UUID",
                            "ODS_AUX_GRAPHICS_GPU_UUID",
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "SDL_VIDEODRIVER",
                            "DISPLAY",
                            "WAYLAND_DISPLAY",
                        )
                    },
                    "timeout_seconds": _deadline(job),
                    "limitations": _LIMITATIONS,
                }
                _write(supervision / "launch.json", launch)
                stream = (supervision / "process.log").open("w", encoding="utf-8")
                started_ns = time.monotonic_ns()
                started = started_ns / 1e9
                environment["ODS_PROCESS_LAUNCH_MONOTONIC_NS"] = str(started_ns)
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=_ROOT,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException as error:
                    stream.close()
                    _mark(job, "failed", repr(error))
                    results.append(
                        {
                            "output": str(job["output"]),
                            "status": "failed",
                            "error": repr(error),
                        }
                    )
                    continue
                active[gpu["index"]] = {
                    "job": job,
                    "process": process,
                    "stream": stream,
                    "started": started,
                    "deadline": started + _deadline(job),
                    "supervision": supervision,
                    "started_ns": started_ns,
                    "gpu_uuid": gpu["uuid"],
                    "aux_gpu_uuid": (
                        auxiliary["uuid"] if auxiliary is not None else None
                    ),
                    "device_error": None,
                    "device_guard": {
                        "samples": 0,
                        "selected_context_seen": False,
                        "unexpected_contexts": [],
                        "allowed_aux_contexts": [],
                    },
                }
                launch["process_launch_monotonic_ns"] = started_ns
                launch["environment"]["ODS_PROCESS_LAUNCH_MONOTONIC_NS"] = str(
                    started_ns
                )
                _write(supervision / "launch.json", launch)
                print(
                    f"Started {job['suite']}: {job['arm']}/{job['vision']}/seed{job['seed']} on GPU{gpu['index']}",
                    flush=True,
                )
            if active and time.monotonic() - last_guard >= 2:
                try:
                    _guard_devices(active)
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    ET.ParseError,
                    subprocess.SubprocessError,
                ) as error:
                    for item in active.values():
                        item["device_error"] = (
                            f"Cannot audit owned GPU placement: {error!r}"
                        )
                last_guard = time.monotonic()
            for index, item in list(active.items()):
                process, job = item["process"], item["job"]
                timeout = time.monotonic() >= item["deadline"]
                if (
                    process.poll() is None
                    and not timeout
                    and not cancelled
                    and item["device_error"] is None
                ):
                    continue
                if timeout or cancelled or item["device_error"] is not None:
                    _terminate(process)
                leaked_group = False
                if not timeout and not cancelled and item["device_error"] is None:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        leaked_group = True
                        _terminate(process)
                item["stream"].close()
                summary_path = job["output"] / "summary.json"
                reason = (
                    item["device_error"]
                    if item["device_error"] is not None
                    else (
                        "suite cancellation"
                        if cancelled
                        else (
                            "process deadline exceeded"
                            if timeout
                            else (
                                "Child left an owned process group alive"
                                if leaked_group
                                else ""
                            )
                        )
                    )
                )
                if reason or process.returncode != 0 or not summary_path.exists():
                    _mark(
                        job,
                        "failed",
                        reason
                        or f"Child exit={process.returncode}, summary exists={summary_path.exists()}",
                    )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["device_supervision"] = {
                    "primary_gpu_uuid": item["gpu_uuid"],
                    "aux_graphics_scope": _aux_scope(auxiliary),
                    "guard": item["device_guard"],
                }
                _write(summary_path, summary)
                result = {
                    "output": str(job["output"]),
                    "supervision": str(item["supervision"]),
                    "status": summary["status"],
                    "qualified": summary.get("metrics", {}).get("qualification_passed"),
                    "returncode": process.returncode,
                    "elapsed_seconds": time.monotonic() - item["started"],
                    "timeout": timeout,
                    "cancelled": cancelled,
                    "leaked_process_group": leaked_group,
                    "pgid": process.pid,
                    "process_launch_monotonic_ns": item["started_ns"],
                    "device_guard": item["device_guard"],
                    "aux_graphics_scope": _aux_scope(auxiliary),
                }
                _write(job["output"] / "process.json", result)
                results.append(result)
                print(
                    f"Finished {job['arm']}/{job['vision']}/seed{job['seed']}: {result['status']}, qualified={result['qualified']}",
                    flush=True,
                )
                del active[index]
            if cancelled:
                for job in pending:
                    job["output"].mkdir(parents=True)
                    _mark(job, "failed", "Suite cancelled before launch")
                    results.append(
                        {
                            "output": str(job["output"]),
                            "status": "failed",
                            "cancelled": True,
                        }
                    )
                pending.clear()
            if pending or active:
                time.sleep(0.25)
    finally:
        for item in active.values():
            try:
                _terminate(item["process"])
            finally:
                item["stream"].close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return results


def _calibration(
    output: Path,
    jobs: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    auxiliary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gamma, capacities, issues = {}, {}, []
    for job in jobs:
        summary = json.loads(
            (job["output"] / "summary.json").read_text(encoding="utf-8")
        )
        key = f"{job['arm']}/{job['vision']}"
        metrics, stats = summary.get("metrics", {}), summary.get("engine_stats", {})
        if (
            summary["status"] != "complete"
            or metrics.get("qualification_passed") is not True
        ):
            issues.append(
                f"{key}: calibration execution or engineering qualification failed"
            )
            continue
        if (
            stats.get("successful_episodes", 0) < _BASE["accepted_episodes"]
            or stats.get("completed_generation", 0) < 4
        ):
            issues.append(
                f"{key}: all 32 accepted episodes/four generations were not completed"
            )
            continue
        if job["arm"] == "protected":
            seconds = stats.get("production_seconds", 0)
            service = metrics.get("service_p50_seconds")
            if seconds <= 0 or service is None or service <= 0:
                issues.append(f"{key}: missing production or batch-service timing")
                continue
            gamma[job["vision"]] = stats["successful_episodes"] / seconds
            capacities[key] = {
                "chunks_per_second": _BASE["batch_size"] / service,
                "scope": "successful immediate Source sampling/copy and RGB conversion p50 with concurrent production; excludes Flow retry waits; not an isolated maximum",
            }
        else:
            rate = metrics.get("delivered_chunks_per_second")
            if rate is None or rate <= 0:
                issues.append(f"{key}: no measured standard-reader throughput")
                continue
            capacities[key] = {
                "chunks_per_second": rate,
                "scope": "source-ready, high-demand unbounded standard Dataset/DataLoader/TorchCodec delivery",
            }
    rate = (
        4 * _BASE["episode_budget"] * max(gamma.values()) if len(gamma) == 3 else None
    )
    duration = (
        max(180.0, max(4 * _BASE["buffer_size"] / value for value in gamma.values()))
        if len(gamma) == 3
        else None
    )
    if duration is not None and duration > 900:
        issues.append(
            "Measured production requires an observation window beyond 900 seconds"
        )
    return {
        "format": "ods-stream-protocol-v1",
        "status": "complete" if not issues and len(capacities) == 6 else "failed",
        "camera_profile_version": CAMERA_PROFILE_VERSION,
        "source_sha256": _hashes(),
        "base_settings": _BASE,
        "request_rate": rate,
        "duration_seconds": duration,
        "generation_episodes_per_second": gamma,
        "ready_source_capacity": capacities,
        "gpu_models": sorted({gpu["name"] for gpu in devices}),
        "aux_graphics_scope": _aux_scope(auxiliary),
        "headless_environment": _HEADLESS_ENV,
        "order_seed": _ORDER_SEED,
        "seeds": list(_SEEDS),
        "issues": issues,
        "limitations": _LIMITATIONS,
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Execute a supervised matrix and retain every attempted or unsupported cell.

    Args:
        args: ``run.build_parser`` arguments with a non-single suite selection.

    Returns:
        Machine-readable suite status, also saved beside the per-cell artifacts.
        Failed/unsupported execution exits with code 2 after rendering its report.
    """
    args = argparse.Namespace(**vars(args))
    args.output = args.output.resolve()
    if args.dataset_root is not None:
        args.dataset_root = args.dataset_root.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh suite output directory")
    if args.suite == "appendix" and args.dataset_root is None:
        raise ValueError(
            "Appendix requires --dataset-root pointing to the main REPORT ROOT"
        )
    protocol = (
        _protocol(args.protocol) if args.suite in {"report", "appendix"} else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(args, protocol)
    _write(
        args.output / "suite-plan.json",
        {
            "suite": args.suite,
            "jobs": [
                {
                    **job,
                    "output": str(job["output"]),
                    "dataset_root": (
                        str(job["dataset_root"])
                        if job["dataset_root"] is not None
                        else None
                    ),
                }
                for job in jobs
            ],
            "order_seed": _ORDER_SEED,
            "aux_graphics_gpu_id": args.aux_graphics_gpu_id,
            "headless_environment": _HEADLESS_ENV,
            "source_sha256": _hashes(),
            "limitations": _LIMITATIONS,
        },
    )
    preflight_error = None
    devices = []
    auxiliary = None
    try:
        if shutil.which("taskset") is None:
            raise RuntimeError(
                "taskset is required to assign eight disjoint CPUs per child"
            )
        devices, device_audit = _select_devices(
            args.gpu_ids, args.output, auxiliary_id=args.aux_graphics_gpu_id
        )
        auxiliary = device_audit["aux_graphics_gpu"]
        _gpu_contexts()  # Qualification requires compute and graphics visibility.
        if (
            protocol is not None
            and sorted({gpu["name"] for gpu in devices}) != protocol["gpu_models"]
        ):
            raise RuntimeError(
                "Selected GPU models differ from the frozen calibration hardware"
            )
        if protocol is not None and protocol.get("aux_graphics_scope") != _aux_scope(
            auxiliary
        ):
            raise RuntimeError(
                "Auxiliary graphics reservation differs from the frozen protocol; pass the same explicit auxiliary flag/scope"
            )
    except (
        OSError,
        ValueError,
        RuntimeError,
        ET.ParseError,
        subprocess.SubprocessError,
    ) as error:
        preflight_error = repr(error)
        for job in jobs:
            job["output"].mkdir(parents=True)
            _mark(job, "unsupported", preflight_error)
        results = [
            {
                "output": str(job["output"]),
                "status": "unsupported",
                "reason": preflight_error,
            }
            for job in jobs
        ]
    else:
        results = _supervise(jobs, devices, auxiliary)
    summary = {
        "suite": args.suite,
        "status": (
            "complete"
            if all(result["status"] == "complete" for result in results)
            else "failed"
        ),
        "qualification_passed": all(
            result.get("qualified") is True for result in results
        ),
        "results": results,
        "preflight_error": preflight_error,
        "limitations": _LIMITATIONS,
        "aux_graphics_scope": _aux_scope(auxiliary),
        "requested_aux_graphics_gpu_id": args.aux_graphics_gpu_id,
    }
    if args.suite == "calibrate":
        calibration = _calibration(args.output, jobs, devices, auxiliary)
        _write(args.output / "calibration.json", calibration)
        summary["calibration_status"] = calibration["status"]
        if calibration["status"] != "complete":
            summary["status"] = "failed"
    _write(args.output / "suite-summary.json", summary)
    try:
        summary["report"] = str(write_report(args.output))
    except Exception as error:
        summary["report_error"] = repr(error)
        summary["status"] = "failed"
    _write(args.output / "suite-summary.json", summary)
    if summary["status"] != "complete":
        raise SystemExit(2)
    return summary
