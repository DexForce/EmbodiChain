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

"""Compare reference GPU replay with synchronous host replay at the native workload.

Run: python scripts/benchmark/learning/host_replay.py --help
Put the pinned Sharpa reference src/ on PYTHONPATH. Sampling uses the unchanged
reference implementation; CPU and CUDA RNG streams differ. Model computation
is excluded. CUDA allocator measurements exclude driver and native physics.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path

import psutil
import torch
from sharpa_rl_unilab.algos.supermimic.replay import ReplayBuffer

from embodichain.learning.rl.experimental._host_replay import _HostReplayAdapter

__all__ = ["main"]


def _initialize(storage: ReplayBuffer) -> None:
    """Fill a wrapped full ring with exact time/environment tokens."""
    row = torch.arange(storage.steps, device=storage.device)[:, None]
    token = (
        row * storage.num_envs
        + torch.arange(storage.num_envs, device=storage.device)[None, :]
    ).float()
    for index, (name, value) in enumerate(storage.data.items()):
        if name == "done":
            value.copy_(((row % 17) == 16).expand_as(value))
        elif name == "ep_step":
            value.copy_((row % 17).expand_as(value))
        else:
            encoded = token + index * (storage.steps * storage.num_envs)
            if value.ndim == 3:
                encoded = encoded[..., None]
            value.copy_(encoded.expand_as(value))
    storage.ptr = storage.steps // 3
    storage.size = storage.steps


def _quality(sampler, storage: ReplayBuffer, device: str) -> None:
    """Check field alignment, episode masks, history endpoints and transfer bytes."""
    torch.manual_seed(78191)
    win = sampler.sample_windows(64, 8)
    start, history = sampler.sample_starts(32, 3)
    capacity = storage.steps * storage.num_envs
    torch.testing.assert_close(
        win["s1"][..., 0] - win["s"][..., 0],
        torch.full((64, 8), float(capacity), device=device),
        rtol=0,
        atol=0,
    )
    expected = torch.cat(
        (torch.ones(64, 1, device=device), 1 - win["done"][:, :-1].float()), -1
    ).cumprod(-1)
    torch.testing.assert_close(win["mask"], expected, rtol=0, atol=0)
    torch.testing.assert_close(start["s"], history["s"][:, -1], rtol=0, atol=0)
    if isinstance(sampler, _HostReplayAdapter):
        torch.manual_seed(78191)
        raw = storage.sample_windows(64, 8)
        raw_start, raw_history = storage.sample_starts(32, 3)
        for left, right in [(win, raw), (start, raw_start), (history, raw_history)]:
            for key in left:
                torch.testing.assert_close(left[key].cpu(), right[key], rtol=0, atol=0)


def _measure(call, repeats: int, device: str) -> dict:
    """Measure synchronized completed-call latency with outputs released between calls."""
    for _ in range(2):
        result = call()
        torch.cuda.synchronize(device)
        del result
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = call()
        torch.cuda.synchronize(device)
        values.append((time.perf_counter() - started) * 1000)
        del result
    return {"median_ms": statistics.median(values), "samples_ms": values}


def main() -> None:
    """Measure equal-capacity sampling and write one three-table report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = "cuda:0"
    torch.cuda.synchronize(device)
    rows = []
    process = psutil.Process()
    for name in ["gpu", "host"]:
        gc.collect()
        torch.cuda.empty_cache()
        before_gpu = torch.cuda.memory_allocated(device)
        before_cpu = process.memory_info().rss
        started = time.perf_counter()
        storage = ReplayBuffer(
            args.steps, args.num_envs, "cpu" if name == "host" else device
        )
        _initialize(storage)
        torch.cuda.synchronize(device)
        setup = time.perf_counter() - started
        sampler = _HostReplayAdapter(storage, device) if name == "host" else storage
        _quality(sampler, storage, device)
        persistent_gpu = torch.cuda.memory_allocated(device) - before_gpu
        cpu_delta = process.memory_info().rss - before_cpu
        for label, wm, pi, window, history in [
            ("small", 1, 1, 8, 3),
            ("native_reference", 2048, 1024, 8, 3),
        ]:
            w = _measure(
                partial(sampler.sample_windows, wm, window), args.repeats, device
            )
            a = _measure(
                partial(sampler.sample_starts, pi, history), args.repeats, device
            )
            torch.cuda.reset_peak_memory_stats(device)
            windows = sampler.sample_windows(wm, window)
            starts = sampler.sample_starts(pi, history)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            del windows, starts
            row = {
                "algorithm": name,
                "workload": label,
                "wm_batch": wm,
                "pi_batch": pi,
                "window": window,
                "history": history,
                "cost_time_ms": 10 * w["median_ms"] + a["median_ms"],
                "window_ms": w["median_ms"],
                "starts_ms": a["median_ms"],
                "cpu_delta_mb": cpu_delta / 2**20,
                "gpu_delta_mb": persistent_gpu / 2**20,
                "peak_gpu_mb": peak / 2**20,
                "host_storage_mb": (
                    sum(x.numel() * x.element_size() for x in storage.data.values())
                    / 2**20
                    if name == "host"
                    else 0.0
                ),
                "success_rate": 1.0,
                "setup_seconds": setup,
                "window_samples": w["samples_ms"],
                "start_samples": a["samples_ms"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        del sampler, storage
        gc.collect()
        torch.cuda.empty_cache()
    metadata = {
        "hardware": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "threads": torch.get_num_threads(),
        "device": device,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "transitions": args.steps * args.num_envs,
        "repeats": args.repeats,
        "warmup": 2,
        "precision": "source field dtypes unchanged",
        "shared_device": "Background GPU load varies; latency comparisons are observational, not a controlled end-to-end speedup claim.",
        "timing_scope": "Synchronized end-to-end sampling, host gathers/transfers included; cost_time_ms estimates10 window samples plus1 history sample from medians, excluding model computation",
        "memory_scope": "Retained CPU RSS delta, persistent Torch CUDA allocation and total Torch allocator peak for separately retained window/start results. Excludes Warp/physics/driver memory.",
        "quality": "Exact field alignment, source episode masks/history endpoints; host proxy outputs bitwise match CPU source sampler under the same CPU RNG. CPU/CUDA random streams differ.",
    }
    (args.output / "result.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n"
    )
    lines = [
        "# Host replay transfer measurement",
        "",
        metadata["timing_scope"],
        "",
        metadata["memory_scope"],
        "",
        "## Time & Memory",
        "",
        "| algorithm | workload | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['algorithm']} | {r['workload']} | {r['cost_time_ms']:.3f} | {r['cpu_delta_mb']:.1f} | {r['gpu_delta_mb']:.1f} | {r['peak_gpu_mb']:.1f} |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| algorithm | workload | success_rate | window_ms | starts_ms | host_storage_mb |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['algorithm']} | {r['workload']} | {r['success_rate']} | {r['window_ms']:.3f} | {r['starts_ms']:.3f} | {r['host_storage_mb']:.1f} |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | workload | overall_success_rate |",
        "| --- | --- | --- | --- |",
    ]
    for rank, r in enumerate(
        sorted(rows, key=lambda r: (-r["success_rate"], r["cost_time_ms"])), 1
    ):
        lines.append(
            f"| {rank} | {r['algorithm']} | {r['workload']} | {r['success_rate']} |"
        )
    lines += ["", json.dumps(metadata, ensure_ascii=False)]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(args.output / "report.md", flush=True)


if __name__ == "__main__":
    main()
