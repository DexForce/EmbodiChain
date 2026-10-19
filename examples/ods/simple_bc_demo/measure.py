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
"""Measure real ODS production under resident ChunkBC contention.

Run: python -m examples.ods.simple_bc_demo.measure --help
Pin CUDA_VISIBLE_DEVICES to one physical GPU before launching. Resident load
is a fixed synthetic GPU training workload, not an ODS learner or quality run.
Cold initial filling and steady refill are reported separately. Failed runs
retain JSON evidence and fail the process; no generation failure is discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path
import subprocess
import threading
import time
import traceback
from typing import Any

__all__ = ["main"]


def _resident(stop: Any, ready: Any, output: str, batch: int) -> None:
    """Run a fixed resident policy update workload on the selected GPU."""
    import torch
    from examples.ods.simple_bc_demo.models import ChunkBC

    torch.set_num_threads(1)
    torch.manual_seed(99001)
    model = ChunkBC(
        torch.zeros(51), torch.ones(51), torch.zeros(10), torch.ones(10)
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
    history = torch.randn(batch, 2, 51, device="cuda")
    target = torch.randn(batch, 64, 10, device="cuda")

    def update() -> None:
        optimizer.zero_grad(set_to_none=True)
        error = (model(history) - target).square()
        loss = (9 * error[..., :9].mean() + error[..., 9].mean()) / 10
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(20):
        update()
    torch.cuda.synchronize()
    ready.set()
    started = time.perf_counter()
    count = 0
    try:
        while not stop.is_set():
            update()
            count += 1
    finally:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        Path(output).write_text(
            json.dumps(
                {
                    "updates": count,
                    "seconds": elapsed,
                    "updates_per_second": count / elapsed,
                    "batch_size": batch,
                    "workload": "synthetic resident ChunkBC, all 64 targets valid; no transfer, geometry, sampling or pose_targets",
                    "loss": "(9 * arm_mse + gripper_mse) / 10",
                    "torch_peak_bytes": torch.cuda.max_memory_allocated(),
                },
                indent=2,
            )
        )


def _monitor(stop: threading.Event, samples: list[dict[str, Any]]) -> None:
    """Sample process-tree PSS and actual native GPU placement every second."""
    import psutil

    root = psutil.Process()
    while not stop.is_set():
        processes = [root] + root.children(recursive=True)
        pids = {p.pid for p in processes}
        pss = 0
        for process in processes:
            try:
                pss += process.memory_full_info().pss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        gpu = []
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            for line in result.stdout.splitlines():
                pid, uuid, memory = (s.strip() for s in line.split(","))
                if int(pid) in pids:
                    gpu.append(
                        {"pid": int(pid), "uuid": uuid, "memory_mib": int(memory)}
                    )
        except (subprocess.SubprocessError, ValueError):
            gpu = []
        samples.append(
            {
                "monotonic": time.perf_counter(),
                "pss_bytes": pss,
                "pids": sorted(pids),
                "gpu": gpu,
            }
        )
        stop.wait(1)


def _config(parallel: int, seed: int) -> dict[str, Any]:
    """Use the current production task and unchanged measured acceptance."""
    from examples.ods.simple_bc_demo.config import load_task_config

    config = load_task_config("cuda:0", "pose", "dls")
    config.update(num_envs=parallel, seed=seed)
    return config


def _qualify(args: argparse.Namespace) -> dict[str, Any]:
    """Execute independent physical episodes without retrying failures."""
    import gymnasium as gym
    from embodichain.lab.gym.utils.registration import discover_task_packages
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    discover_task_packages()
    config = _config(args.parallel_envs, args.seed)
    cfg = config_to_cfg(config)
    cfg.init_rollout_buffer = False
    cfg.filter_dataset_saving = True
    env = gym.make(config["id"], cfg=cfg)
    env.unwrapped._replay_no_auto_reset = True
    records = []
    try:
        for first in range(0, args.episodes, args.parallel_envs):
            seed = args.seed + first
            started = time.perf_counter()
            env.reset(seed=seed, options={"save_data": False})
            result = execute_demo_episode(env)
            record = {
                "seed": seed,
                "seconds": time.perf_counter() - started,
                "success": list(result.success),
                "completed_by_env": list(result.completed_by_env),
                "all_success": result.all_success,
                "lengths": list(result.lengths),
                "reason": result.terminal_reason,
            }
            records.append(record)
            (args.output / "qualification_progress.json").write_text(
                json.dumps(records, indent=2)
            )
            print(json.dumps(record), flush=True)
    finally:
        env.close()
    successes = sum(sum(row["success"]) for row in records)
    return {
        "records": records,
        "successful_episodes": successes,
        "attempted_episodes": args.episodes,
        "success_rate": successes / args.episodes,
        "qualified": args.episodes >= 40 and successes / args.episodes >= 0.95,
    }


def _measure(args: argparse.Namespace) -> dict[str, Any]:
    """Time one cold fill plus one same-capacity steady refill."""
    from embodichain.data_pipeline.engine.data import (
        OnlineDataEngine,
        OnlineDataEngineCfg,
    )

    capacity = args.episodes // 2
    engine = OnlineDataEngine(
        OnlineDataEngineCfg(
            gym_config=_config(args.parallel_envs, args.seed),
            state_dim=9,
            action_dim=7,
            action_semantics="policy_request",
            action_contracts=("eef_pose.relative@1", "gripper.parallel@1"),
            buffer_size=capacity,
            max_episode_steps=600,
            refill_policy="manual",
            include_provenance=True,
            lifecycle_event_capacity=2 * args.episodes,
            initialization_timeout=args.timeout,
            shutdown_timeout=30,
        )
    )
    started = time.perf_counter()
    started_ns = time.monotonic_ns()
    try:
        engine.start()
        cold = time.perf_counter() - started
        initial = engine.get_stats()
        events = engine.drain_lifecycle_events()
        refill_start = time.perf_counter()
        refill_ns = time.monotonic_ns()
        generation = engine.request_refill()
        engine.wait_for_generation(generation, timeout=args.timeout)
        steady = time.perf_counter() - refill_start
        final = engine.get_stats()
        events.extend(engine.drain_lifecycle_events())
        (args.output / "lifecycle_events.json").write_text(json.dumps(events))
        initial_times = sorted(
            {
                event["monotonic_ns"]
                for event in events
                if event["kind"] == "published" and event["generation"] == 1
            }
        )
        refill_times = sorted(
            {
                event["monotonic_ns"]
                for event in events
                if event["kind"] == "published" and event["generation"] == generation
            }
        )
        import numpy as np

        intervals = np.diff([refill_ns] + refill_times) / 1e9
        production = final["production_seconds"]
        return {
            "first_cold_publication_seconds": (initial_times[0] - started_ns) / 1e9,
            "refill_to_first_publication_seconds": (refill_times[0] - refill_ns) / 1e9,
            "refill_publication_interval_p50_seconds": float(
                np.percentile(intervals, 50)
            ),
            "refill_publication_interval_p95_seconds": float(
                np.percentile(intervals, 95)
            ),
            "cold_initial_fill_seconds": cold,
            "steady_refill_seconds": steady,
            "initial_stats": initial,
            "final_stats": final,
            "accepted_trajectories_per_production_second": final["successful_episodes"]
            / production,
            "accepted_frames_per_production_second": final["generated_frames"]
            / production,
            "steady_trajectories_per_second": capacity / steady,
            "engine_start_through_refill_trajectories_per_second": args.episodes
            / (cold + steady),
            "success_rate": final["successful_episodes"]
            / max(1, final["successful_episodes"] + final["failed_episodes"]),
            "failure_semantics": "engine failed_episodes counts discarded batch rows, not independent physical failures",
        }
    finally:
        try:
            (args.output / "producer_final_stats.json").write_text(
                json.dumps(engine.get_stats(), indent=2)
            )
            pending = engine.drain_lifecycle_events()
            if pending:
                (args.output / "producer_remaining_events.json").write_text(
                    json.dumps(pending)
                )
        finally:
            engine.stop()


def _report(result: dict[str, Any], output: Path) -> None:
    """Write the component convention's three tables with failure evidence."""
    memory = result["resources"]
    success = result.get("metrics", {}).get("success_rate", "unavailable")
    rate = result.get("metrics", {}).get("steady_trajectories_per_second", "n/a")
    label = f'P{result["parallel_envs"]}/{result["load"]}/{result["phase"]}'
    text = f"""# ODS usage producer benchmark

## Time & Memory

| configuration | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | sampled_peak_pss_mb |
| --- | ---: | ---: | ---: | ---: | ---: |
| {label} | {result["elapsed_seconds"] * 1000:.3f} | {memory["pss_delta_mib"]:.3f} | {memory["gpu_delta_mib"]} | {memory["peak_gpu_mib"]} | {memory["peak_pss_mib"]:.3f} |

## Success & Other Metrics

| configuration | success_rate | steady_trajectories_per_second | status |
| --- | ---: | ---: | --- |
| {label} | {success} | {rate} | {result["status"]} |

## Leaderboard

| rank | algorithm | overall_success_rate |
| --- | --- | ---: |
| 1 | {label} | {success} |

Single-cell descriptive report. JSON preserves cold/steady timing, raw resource samples,
source revision and configuration. Native GPU memory and PSS are sampled, not exact peaks.
The elapsed total includes initialization and shutdown; steady refill excludes startup.
Resident contention uses a synthetic GPU-resident ChunkBC workload, not data consumption.
For measure cells, success_rate is accepted rows / (accepted + discarded batch rows),
not the independent physical success rate reported by qualification cells.
"""
    (output / "REPORT.md").write_text(text)


def main() -> None:
    """Execute one bounded cell and preserve status even when it fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("qualify", "measure"), default="measure")
    parser.add_argument("--parallel-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--load", choices=("none", "resident"), default="none")
    parser.add_argument("--seed", type=int, default=830000)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    divisor = args.parallel_envs * (2 if args.phase == "measure" else 1)
    if args.parallel_envs < 1 or args.episodes < divisor or args.episodes % divisor:
        parser.error(
            "episodes must be a positive multiple of P (qualification) or 2P (measure)"
        )
    if (
        not os.environ.get("CUDA_VISIBLE_DEVICES")
        or "," in os.environ["CUDA_VISIBLE_DEVICES"]
    ):
        parser.error("pin exactly one CUDA_VISIBLE_DEVICES before launching")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    torch.set_num_threads(1)
    ctx = mp.get_context("spawn")
    stop, ready = ctx.Event(), ctx.Event()
    load = None
    monitor_stop = threading.Event()
    samples: list[dict[str, Any]] = []
    monitor = threading.Thread(
        target=_monitor, args=(monitor_stop, samples), daemon=True
    )
    result = {
        "phase": args.phase,
        "parallel_envs": args.parallel_envs,
        "episodes": args.episodes,
        "load": args.load,
        "seed": args.seed,
        "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "embodichain/data_pipeline/engine/data.py",
                "embodichain/lab/gym/envs/managers/actions.py",
                "examples/ods/simple_bc_demo/environment.py",
                "examples/ods/simple_bc_demo/models.py",
                "examples/ods/simple_bc_demo/measure.py",
            )
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu_hardware": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "task_config": _config(args.parallel_envs, args.seed),
        "status": "running",
    }
    monitor.start()
    started = time.perf_counter()
    error = None
    try:
        if args.load == "resident":
            load = ctx.Process(
                target=_resident,
                args=(stop, ready, str(args.output / "resident.json"), 256),
            )
            load.start()
            if not ready.wait(90):
                raise TimeoutError("Resident learner did not become ready")
        result["metrics"] = (
            _qualify(args) if args.phase == "qualify" else _measure(args)
        )
        if load is not None and not load.is_alive():
            raise RuntimeError("Resident load exited before production completed")
        result["status"] = "completed"
    except BaseException as caught:
        error = caught
        result.update(status="failed", error=traceback.format_exc())
    finally:
        stop.set()
        if load is not None:
            load.join(30)
            if load.is_alive():
                load.terminate()
                load.join(10)
                result["status"] = "failed_forced_load_shutdown"
                error = RuntimeError("Resident load required forced shutdown")
        monitor_stop.set()
        monitor.join(10)
        result["elapsed_seconds"] = time.perf_counter() - started
        gpu_totals = [sum(g["memory_mib"] for g in s["gpu"]) for s in samples] or [0]
        pss = [s["pss_bytes"] / 2**20 for s in samples] or [0]
        result["resources"] = {
            "peak_gpu_mib": max(gpu_totals),
            "gpu_delta_mib": gpu_totals[-1] - gpu_totals[0],
            "peak_pss_mib": max(pss),
            "pss_delta_mib": pss[-1] - pss[0],
            "samples": samples,
        }
        result["observed_gpu_uuids"] = sorted(
            {g["uuid"] for s in samples for g in s["gpu"]}
        )
        if result["physical_gpu"].startswith("GPU-") and any(
            uuid != result["physical_gpu"] for uuid in result["observed_gpu_uuids"]
        ):
            result["status"] = "failed_gpu_placement"
            error = RuntimeError(
                "Native process used a GPU outside CUDA_VISIBLE_DEVICES"
            )
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        _report(result, args.output)
    if error is not None:
        raise error
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "metrics": result.get("metrics"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
