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

"""Compare completed Library sensor captures with and without OptiX denoising.

Run: python scripts/benchmark/rendering/library_denoising.py --gym-config TASK
     --trajectory TRAJECTORY --output OUTPUT_DIRECTORY --nvml-device 1

CUDA_VISIBLE_DEVICES selects the rendering GPU; --nvml-device is its physical
NVML index. Four fresh processes run off/optix/optix/off. Physics states are
restored from one qualified episode. Disk writing, quality statistics and memory
sampling are outside the timed render/publication/CPU-readback calls.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

__all__ = ["main"]


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _gpu_memory(index: int) -> Callable[[], int] | None:
    """Return an optional native-device memory sampler, including graphics."""
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit_v2() != 0:
            return None
        handle = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
            return None

        class Memory(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong) for name in ("total", "free", "used")
            ]

        def read() -> int:
            memory = Memory()
            if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                raise RuntimeError("NVML memory sampling failed")
            return memory.used

        return read
    except OSError:
        return None


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    """Measure one native process at three identical recorded poses."""
    import gymnasium as gym
    import psutil
    import torch
    from PIL import Image
    import dexsim
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.trajectory_state import restore_trajectory_state
    from embodichain.utils.utility import load_config

    process = psutil.Process()
    gpu_memory = _gpu_memory(args.nvml_device)
    gpu_before = gpu_memory() if gpu_memory is not None else None
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    cfg = config_to_cfg(load_config(args.gym_config), source_path=args.gym_config)
    cfg.sim_cfg.render_cfg.denoising.offscreen = args.mode
    cfg.sim_cfg.render_cfg.denoising.window = "off"
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.startup_summary = "off"
    cfg.record_trajectory = False
    cfg.init_rollout_buffer = False
    recorded = torch.load(args.trajectory, map_location="cpu", weights_only=False)
    steps = int(recorded["meta"]["lengths"][0])
    lifted = recorded["meta"]["demo_episodes"][0]["segments"][0]["end_step"]
    states = (
        ("initial", recorded["states"][:, 0]),
        ("lifted", recorded["states"][:, lifted]),
        ("terminal", recorded["terminal_states"]),
    )
    env = camera = None
    try:
        task_id = load_config(args.gym_config)["id"]
        env = gym.make(task_id, cfg=cfg).unwrapped
        env.reset(seed=42)
        env.sim.enable_physics(False)
        camera = env.sensors["cam_high"]
        if cfg.num_envs != 1 or (camera.cfg.width, camera.cfg.height) != (960, 540):
            raise ValueError("Expected the qualified B=1, 960x540 Library deployment")
        if cfg.sim_cfg.render_cfg.spp != 1 or str(env.device) != "cpu":
            raise ValueError("Comparison requires spp=1 and CPU readback")
        cold_seconds = time.perf_counter() - started
        clock_before = env.sim.simulation_time
        last_frame_id = -1

        def capture() -> float:
            nonlocal last_frame_id
            begin = time.perf_counter()
            with env.sim.render_frame():
                env.sim.render_camera_group([camera.group_id])
                camera.update(fetch_only=True)
            elapsed = (time.perf_counter() - begin) * 1000
            status = camera.capture_metadata
            if (
                not status["valid"]
                or not status["output_contract"]["same_frame_identity"]
            ):
                raise RuntimeError("No completed same-frame RGB/depth capture")
            if status["configured_offscreen_denoising"] != args.mode:
                raise RuntimeError("Unexpected native denoising selection")
            if status["frame_id"] <= last_frame_id:
                raise RuntimeError("Camera returned a stale completed frame")
            last_frame_id = status["frame_id"]
            return elapsed

        rows = []
        args.output.mkdir(parents=True, exist_ok=True)
        for name, state in states:
            restore_trajectory_state(env, state)
            for _ in range(args.warmup):
                capture()
            samples = [capture() for _ in range(args.samples)]
            rgb = camera.get_data()["color"][0, ..., :3].numpy()
            Image.fromarray(rgb).save(
                args.output / f"{args.mode}-{args.repeat}-{name}.png"
            )
            rows.append({"pose": name, "samples_ms": samples})
        if env.sim.simulation_time != clock_before:
            raise RuntimeError("Static sensor measurements advanced physics")
        rss_after = process.memory_info().rss
        memory_samples = []
        stop = threading.Event()

        def sample_memory() -> None:
            while not stop.is_set():
                memory_samples.append(gpu_memory())
                stop.wait(0.002)

        if gpu_memory is not None:
            sampler = threading.Thread(target=sample_memory)
            sampler.start()
            try:
                for _ in range(20):
                    capture()
            finally:
                stop.set()
                sampler.join()
        return {
            "mode": args.mode,
            "repeat": args.repeat,
            "poses": rows,
            "cold_seconds": cold_seconds,
            "cpu_delta_mb": (rss_after - rss_before) / 2**20,
            "gpu_before_mb": gpu_before / 2**20 if gpu_before is not None else None,
            "gpu_delta_mb": (
                (gpu_memory() - gpu_before) / 2**20 if gpu_memory is not None else None
            ),
            "peak_gpu_mb": max(memory_samples) / 2**20 if memory_samples else None,
            "success_rate": 1.0,
            "sdk_version": dexsim.__version__,
            "sdk_commit": getattr(dexsim, "__commit_id__", None),
            "torch_threads": torch.get_num_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": sys.version,
            "render_settings": cfg.sim_cfg.render_cfg.to_dict(),
            "scene_dependencies": cfg._usd_scene_inventory["dependencies"],
            "trajectory_sha256": _sha(args.trajectory),
            "steps_in_source": steps,
        }
    finally:
        camera = None
        if env is not None:
            env.close(exit_process=False)
        env = None


def _report(args: argparse.Namespace, rows: list[dict]) -> None:
    """Write the three-table comparison with raw-sample provenance."""
    for row in rows[1:]:
        if any(
            row[key] != rows[0][key]
            for key in ("scene_dependencies", "trajectory_sha256")
        ):
            raise ValueError("Scene or trajectory changed during the comparison")
    groups = {}
    for mode in ("off", "optix"):
        selected = [row for row in rows if row["mode"] == mode]
        samples = [
            v for row in selected for pose in row["poses"] for v in pose["samples_ms"]
        ]
        groups[mode] = {
            "cost_time_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)),
            "samples": len(samples),
            **{
                key: (
                    float(np.median([row[key] for row in selected]))
                    if all(row[key] is not None for row in selected)
                    else None
                )
                for key in ("cpu_delta_mb", "gpu_delta_mb", "peak_gpu_mb")
            },
        }
    result = {
        "settings": vars(args)
        | {
            "gym_config": str(args.gym_config),
            "trajectory": str(args.trajectory),
            "output": str(args.output),
        },
        "workers": rows,
        "summary": groups,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Library sensor denoising comparison",
        "",
        "Same 2M-point scene, trajectory poses, 960x540 RGB/depth, Hybrid and spp=1.",
        "Four fresh processes run off/optix/optix/off. Completed camera calls include",
        "publication, native rendering and CPU RGB/depth readback; physics, quality",
        "statistics, disk writes and memory profiling are outside the timed scope.",
        f"Each of three poses has {args.warmup} warmups and {args.samples} timed calls per process.",
        "Quality is illustrated by matching-pose PNGs and the actual sensor recordings.",
        "Fixed-clock rendering is deterministic; temporal variance is not a noise metric here.",
        "NVML reports whole-device native memory;",
        "CPU delta is retained RSS. Sampled GPU peak uses a separate 2ms-poll memory pass.",
        "Faster transient allocations may be missed. Missing NVML values are unmeasured.",
        "",
        "## Time & Memory",
        "",
        "| mode | cost_time_ms | p95_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, value in groups.items():
        values = [
            f"{value[key]:.3f}" if value[key] is not None else "N/A"
            for key in (
                "cost_time_ms",
                "p95_ms",
                "cpu_delta_mb",
                "gpu_delta_mb",
                "peak_gpu_mb",
            )
        ]
        lines.append(f"| {mode} | " + " | ".join(values) + f" | {value['samples']} |")
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| mode | capture_success_rate | resolution | spp | poses |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for mode, value in groups.items():
        lines.append(f"| {mode} | 100% | 960x540 | 1 | 3 |")
    lines += [
        "",
        "## Leaderboard",
        "",
        "Ranked by capture success, then completed-call latency; quality is shown separately.",
        "",
        "| rank | mode | capture_success_rate | cost_time_ms |",
        "| --- | --- | ---: | ---: |",
    ]
    for rank, mode in enumerate(
        sorted(groups, key=lambda key: groups[key]["cost_time_ms"]), 1
    ):
        lines.append(f"| {rank} | {mode} | 100% | {groups[mode]['cost_time_ms']:.3f} |")
    path = args.output / "report.md"
    path.write_text("\n".join(lines) + "\n")
    print(json.dumps(groups, indent=2))
    print(f"Report: {path}")


def main() -> None:
    """Run isolated native workers and publish one comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvml-device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--mode", choices=("off", "optix"))
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 1:
        parser.error("Positive samples and warmup required")
    args.gym_config = args.gym_config.resolve(strict=True)
    args.trajectory = args.trajectory.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    if args.worker:
        row = _worker(args)
        gc.collect()
        from embodichain.lab.sim import SimulationManager

        SimulationManager.flush_cleanup_queue()
        print("RESULT_JSON=" + json.dumps(row), flush=True)
        return
    rows = []
    for repeat, mode in enumerate(("off", "optix", "optix", "off")):
        command = [
            sys.executable,
            __file__,
            "--worker",
            "--mode",
            mode,
            "--repeat",
            str(repeat),
            "--gym-config",
            str(args.gym_config),
            "--trajectory",
            str(args.trajectory),
            "--output",
            str(args.output),
            "--nvml-device",
            str(args.nvml_device),
            "--samples",
            str(args.samples),
            "--warmup",
            str(args.warmup),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        (args.output / f"{mode}-{repeat}.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        marker = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("RESULT_JSON=")
        )
        rows.append(json.loads(marker.removeprefix("RESULT_JSON=")))
        print(f"Completed {mode} process {repeat + 1}/4", flush=True)
    _report(args, rows)


if __name__ == "__main__":
    main()
