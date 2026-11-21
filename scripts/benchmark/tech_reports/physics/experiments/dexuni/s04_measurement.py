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
"""Record synchronized S04 outer-step latency and a separate warmed memory call.

Run: use the platform s04.py workers with --measure --repeat 1.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform as system
import subprocess
import sys
import time

import numpy as np

from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_common import OUTPUT

__all__ = [
    "file_hashes",
    "measurement_directory",
    "completed_step_seconds",
    "summarize_steps",
    "completion_checks",
    "record_measurement",
    "write_json",
]
WARMUP = 30


def write_json(path: Path, value: object) -> None:
    """Write structured data with NumPy scalar support."""
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n"
    )


def file_hashes(root: Path) -> dict[str, str]:
    """Hash every regular input file, including bundled robot assets."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def measurement_directory(platform: str, repeat: int) -> Path:
    """Validate authorization and immutable inputs before reserving a run directory."""
    if platform not in ("embodichain", "isaaclab") or repeat not in range(1, 6):
        raise ValueError("S04 requires platforms embodichain/isaaclab and repeats 1..5")
    protocol = json.loads((OUTPUT / "measurement/protocol.json").read_text())
    if not protocol["authorized"]:
        raise ValueError("Measurement is not authorized")
    if file_hashes(OUTPUT / "inputs") != protocol["input_sha256"]:
        raise ValueError("Frozen input identity changed")
    if (
        hashlib.sha256((OUTPUT / "preview/comparison.mp4").read_bytes()).hexdigest()
        != protocol["approved_preview_sha256"]
    ):
        raise ValueError("Approved platform video changed")
    directory = OUTPUT / "measurement" / platform / f"r{repeat:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def completed_step_seconds(
    advance: Callable[[], None],
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    """Time one completed step, excluding queued input work and including final sync."""
    synchronize()
    start = clock()
    advance()
    synchronize()
    return clock() - start


def summarize_steps(
    seconds: np.ndarray, warmup: int = WARMUP
) -> dict[str, float | int]:
    """Summarize positive finite timings after the declared warmup window."""
    seconds = np.asarray(seconds)
    if (
        warmup < 1
        or seconds.ndim != 1
        or len(seconds) <= warmup
        or not np.isfinite(seconds).all()
        or (seconds <= 0).any()
    ):
        raise ValueError("Invalid or insufficient completed-step timings")
    steady = seconds[warmup:]
    return {
        "cost_time_ms": float(1000 * steady.mean()),
        "median_ms": float(1000 * np.median(steady)),
        "p95_ms": float(1000 * np.percentile(steady, 95)),
        "warmup_steps": warmup,
        "measured_steps": len(steady),
        "warmup_s": float(seconds[:warmup].sum()),
        "all_steps_s": float(seconds.sum()),
    }


def completion_checks(
    manifest: dict, states: dict, seconds: np.ndarray
) -> dict[str, bool]:
    """Check complete finite episodes and timing protocol without judging final shape."""
    cfg = manifest["config"]
    expected = cfg["steps"]
    required = ("particles", "qpos", "target_q", "rigid_pose", "tcp", "time")
    effective = manifest.get("effective", {})
    iterations = effective.get(
        "vbd_iterations", effective.get("vbd", {}).get("iterations")
    )
    return {
        "completed": manifest.get("status") == "completed",
        "finite_full_state": all(
            k in states
            and len(states[k]) == expected + 1
            and np.isfinite(states[k]).all()
            for k in required
        ),
        "time_grid": "time" in states
        and states["time"].shape == (expected + 1,)
        and np.allclose(
            states["time"], np.arange(expected + 1) * cfg["dt"], rtol=0, atol=1e-9
        ),
        "full_positive_timing": np.asarray(seconds).shape == (expected,)
        and np.isfinite(seconds).all()
        and bool((np.asarray(seconds) > 0).all()),
        "no_camera": manifest.get("camera_count") == 0
        and manifest.get("renderer") == "no-render",
        "cuda_graph_captured": effective.get("cuda_graph_status") == "captured",
        "outer_step_cadence": effective.get("outer_steps_per_command") == 1
        and np.isclose(
            effective.get("simulated_elapsed_s", -1.0),
            expected * cfg["dt"],
            rtol=0,
            atol=1e-8,
        ),
        "solver_budget": effective.get("num_substeps") == cfg["substeps"]
        and iterations == cfg["vbd_iterations"],
    }


def _environment() -> dict:
    """Record hardware, software, device and host thread settings."""
    import torch
    import warp as wp
    import psutil

    versions = {}
    for name in (
        "embodichain",
        "isaaclab",
        "dexsim",
        "newton",
        "warp-lang",
        "torch",
        "numpy",
        "mujoco",
        "mujoco-warp",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    gpu = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "pid": os.getpid(),
        "executable": sys.executable,
        "python": sys.version,
        "os": system.platform(),
        "cpu": system.processor(),
        "cpu_logical_count": os.cpu_count(),
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "gpu": gpu.name,
        "gpu_total_mib": gpu.total_memory / 1024**2,
        "driver": driver,
        "torch_cuda": torch.version.cuda,
        "warp": wp.__version__,
        "versions": versions,
        "dtype": "float32",
        "device": "cuda:0",
        "seed": "frozen deterministic inputs; solver determinism not guaranteed",
    }


def _memory_call(
    advance: Callable[[], None],
    readings: Callable[[], dict],
    synchronize: Callable[[], None],
    tracker: object,
) -> dict:
    """Profile one separate warmed final-target step, retaining its readback output."""
    import torch
    import psutil

    synchronize()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    allocated_before = torch.cuda.memory_allocated(0)
    torch.cuda.reset_peak_memory_stats(0)
    fb_before = tracker.sample("warmed_memory_before")
    advance()
    synchronize()
    retained = readings()
    synchronize()
    rss_after = process.memory_info().rss
    allocated_after = torch.cuda.memory_allocated(0)
    peak = torch.cuda.max_memory_allocated(0)
    fb_after = tracker.sample("warmed_memory_after")
    return {
        "scope": "one extra untimed warmed final-target step after the 720-step episode; retains one host readback; not an episode state or timing sample",
        "cpu_delta_mb": (rss_after - rss_before) / 1024**2,
        "gpu_delta_mb": (allocated_after - allocated_before) / 1024**2,
        "peak_gpu_mb": peak / 1024**2,
        "peak_gpu_delta_mb": (peak - allocated_before) / 1024**2,
        "torch_baseline_mb": allocated_before / 1024**2,
        "gpu_allocator_coverage": "Torch CUDA allocator only; excludes Warp/native buffers and context",
        "warmed_process_before_mib": fb_before if fb_before and fb_before > 0 else None,
        "warmed_process_after_mib": fb_after if fb_after and fb_after > 0 else None,
        "readback_finite": all(np.isfinite(v).all() for v in retained.values()),
    }


def record_measurement(
    platform: str,
    output: Path,
    inputs: dict,
    set_target: Callable[[int], None],
    advance: Callable[[], None],
    readings: Callable[[], dict[str, np.ndarray]],
    synchronize: Callable[[], None],
    diagnostics: Callable[[], dict],
    init_s: float,
) -> None:
    """Record a full no-camera episode, keeping setup, commands and readback outside timing."""
    from scripts.benchmark.tech_reports.physics.platforms.dexuni._lib.benchlib.gpu_memory import (
        GpuProcessMemoryTracker,
    )

    tracker = GpuProcessMemoryTracker()
    tracker.start()
    cfg = inputs["config"]
    manifest = {
        "platform": platform,
        "mode": "performance_measurement",
        "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": cfg,
        "camera_count": 0,
        "renderer": "no-render",
        "input_sha256": file_hashes(OUTPUT / "inputs"),
        "environment": _environment(),
        "init_s": init_s,
        "timing_scope": "Warp device synchronized completed outer step; EC sim.update; Isaac write/step/update; commands/readback/serialization excluded",
        "memory_scope": "NVML process framebuffer snapshots after initialization and outside timers; sampled maximum is not a transient peak; separate final warmed memory call",
    }
    write_json(output / "manifest.json", manifest)
    states: dict[str, list] = {}
    rows = []
    episode_start = time.perf_counter()
    try:
        for frame in range(cfg["steps"] + 1):
            command_s = elapsed = 0.0
            if frame:
                before = time.perf_counter()
                set_target(frame)
                synchronize()
                command_s = time.perf_counter() - before
                elapsed = completed_step_seconds(advance, synchronize)
            before = time.perf_counter()
            row = readings()
            row.update(
                time=np.asarray(frame * cfg["dt"]), target_q=inputs["targets"][frame]
            )
            for key, value in row.items():
                array = np.asarray(value)
                states.setdefault(key, []).append(array.copy())
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite {key} at frame {frame}")
            readback_s = time.perf_counter() - before
            if frame:
                rows.append(
                    (
                        frame,
                        frame * cfg["dt"],
                        frame > WARMUP,
                        elapsed,
                        command_s,
                        readback_s,
                    )
                )
            if frame % 30 == 0:
                tracker.sample(f"frame_{frame}")
            if frame % 120 == 0:
                print(f"{platform}: measurement {frame}/{cfg['steps']}", flush=True)
        manifest.update(
            status="completed",
            effective=diagnostics(),
            episode_wall_s=time.perf_counter() - episode_start,
        )
        if platform == "embodichain":
            from scripts.benchmark.tech_reports.physics.experiments.dexuni.sdk_identity import (
                loaded_sdk_identity,
            )

            manifest["sdk_runtime"] = loaded_sdk_identity()
        manifest["summary"] = summarize_steps(np.asarray([r[3] for r in rows]))
        manifest["memory_call"] = _memory_call(advance, readings, synchronize, tracker)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        serialization_start = time.perf_counter()
        if states:
            np.savez_compressed(
                output / "states.npz",
                joint_names=np.asarray(inputs["names"]),
                **{k: np.stack(v) for k, v in states.items()},
            )
        with (output / "timing.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "frame",
                    "time_s",
                    "steady",
                    "step_s",
                    "command_s",
                    "readback_record_s",
                ]
            )
            writer.writerows(rows)
        memory = tracker.metrics()
        manifest["gpu_memory"] = memory
        manifest["gpu_process_sampled_max_mib"] = (
            memory["gpu_process_peak_mib"]
            if memory["gpu_tracking_available"] and memory["gpu_process_peak_mib"] > 0
            else None
        )
        manifest["serialization_s"] = time.perf_counter() - serialization_start
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
        tracker.close()
