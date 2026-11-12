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

"""Validate and time native Sharpa reset implementations on the same cached states.

Run: python scripts/benchmark/sim/sharpa_reset.py --help
The explicit prototype directory owns the experiment factories. This benchmark
checks exact source-cache replay and parameter readback before ranking latency.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from omegaconf import OmegaConf

__all__ = ["main"]


def _prepare_assets(source: Path, output: Path) -> None:
    """Reuse immutable hand assets outside the timed reset path."""
    target = output / "assets"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("source.json", "hand.urdf"):
        shutil.copy2(source / name, target / name)


def _run_case(
    factory: Any, cfg: Any, args: argparse.Namespace, name: str
) -> tuple[dict, list]:
    """Measure synchronized end-to-end reset time and verify actual native state."""
    output = args.output / name
    _prepare_assets(args.assets, output)
    task = None
    started = time.perf_counter()
    snapshots = []
    try:
        task = factory(
            cfg, args.upstream, output, num_envs=args.num_envs, seed=99001, scale=1.0
        )
        setup = time.perf_counter() - started
        for seed in (99002, 99003):
            task.reset(seed=seed)
        durations = []
        cache_errors = []
        gain_errors = []
        for index in range(args.repeats):
            seed = 99010 + index
            torch.cuda.synchronize()
            before = time.perf_counter()
            task.reset(seed=seed)
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - before) * 1000)
            # Source expectation uses the cache RNG directly, independently of
            # the framework state setter. Out-of-limit cached poses must survive.
            rng = torch.Generator(device=task.device).manual_seed(seed)
            ids = torch.randint(
                len(task.physical.grasps),
                (args.num_envs,),
                device=task.device,
                generator=rng,
            )
            expected = task.physical.grasps[ids, :22]
            actual = task.s[:, 13:35]
            cache_errors.append(float((actual - expected).abs().max()))
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
            gain = task.robot.body_data.joint_stiffness[:, task.physical.joint_ids]
            gain_errors.append(float((gain - task.xi[:, :22]).abs().max()))
            torch.testing.assert_close(gain, task.xi[:, :22])
            snapshots.append((task.s.cpu().clone(), task.xi.cpu().clone()))
        # Separate warmed memory measurement. Torch excludes Warp/native solver
        # allocations; global free memory is shared-device telemetry, not a peak.
        torch.cuda.synchronize()
        process = psutil.Process()
        rss = process.memory_info().rss
        allocated = torch.cuda.memory_allocated()
        free_before = torch.cuda.mem_get_info()[0]
        torch.cuda.reset_peak_memory_stats()
        task.reset(seed=99100)
        torch.cuda.synchronize()
        result = {
            "name": name,
            "cost_time_ms": statistics.median(durations),
            "p90_ms": float(np.percentile(durations, 90)),
            "samples_ms": durations,
            "setup_seconds": setup,
            "cpu_delta_mb": (process.memory_info().rss - rss) / 2**20,
            "gpu_delta_mb": (torch.cuda.memory_allocated() - allocated) / 2**20,
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20,
            "global_free_before_mb": free_before / 2**20,
            "global_free_after_mb": torch.cuda.mem_get_info()[0] / 2**20,
            "success_rate": 1.0,
            "max_cache_qpos_error": max(cache_errors),
            "max_gain_readback_error": max(gain_errors),
        }
        print(json.dumps(result), flush=True)
        return result, snapshots
    finally:
        if task is not None:
            task.close()
        torch.cuda.empty_cache()


def main() -> None:
    """Compare equivalent native reset states, then write one three-table report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    for name in ("prototype", "upstream", "config", "assets", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.prototype))
    normal = importlib.import_module("_randomized_task")._RandomizedTask
    fast = importlib.import_module("_fast_randomized_task")._FastRandomizedTask
    cfg = OmegaConf.load(args.config)
    cfg.training.device = "cuda:0"
    from sharpa_rl_unilab.training.configuration import configure_threads

    configure_threads(cfg)
    torch.manual_seed(991)
    results = []
    data = []
    for name, factory in (
        ("native_source_reset", normal),
        ("batched_state_reset", fast),
    ):
        result, snapshots = _run_case(factory, cfg, args, name)
        results.append(result)
        data.append(snapshots)
    state_error = context_error = 0.0
    for baseline, candidate in zip(*data, strict=True):
        state_error = max(state_error, float((baseline[0] - candidate[0]).abs().max()))
        context_error = max(
            context_error, float((baseline[1] - candidate[1]).abs().max())
        )
        torch.testing.assert_close(baseline[0], candidate[0], rtol=0, atol=2e-6)
        torch.testing.assert_close(baseline[1], candidate[1], rtol=0, atol=2e-6)
    metadata = {
        "hardware": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "warp": importlib.metadata.version("warp-lang"),
        "dtype": "float32",
        "device": "cuda:0",
        "num_envs": args.num_envs,
        "joint_count": 22,
        "warmup": 2,
        "repeats": args.repeats,
        "torch_threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "timing_scope": "end-to-end reset with CUDA synchronization; excludes factory setup and state/parameter readback",
        "memory_scope": "separate warmed reset; Torch peak/current only, excludes Warp/native allocations; CPU retained RSS delta is not a transient peak; global free shared-GPU values are telemetry",
        "max_cross_case_state_error": state_error,
        "max_cross_case_context_error": context_error,
        "cases": results,
    }
    (args.output / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = [
        "# Native Sharpa reset benchmark",
        "",
        f"{metadata['hardware']}; Python{metadata['python']}; Torch{metadata['torch']}; float32 CUDA, {args.num_envs} environments ×22 joints.",
        "",
        metadata["timing_scope"] + ".",
        "",
        metadata["memory_scope"] + ".",
        "",
        "## Time & Memory",
        "",
        "| Method | cost_time_ms | p90_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['cost_time_ms']:.3f} | {r['p90_ms']:.3f} | {r['cpu_delta_mb']:.3f} | {r['gpu_delta_mb']:.3f} | {r['peak_gpu_mb']:.3f} |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Method | success_rate | max_cache_qpos_error | max_gain_readback_error | setup_seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['success_rate']:.3f} | {r['max_cache_qpos_error']:.3g} | {r['max_gain_readback_error']:.3g} | {r['setup_seconds']:.3f} |"
        )
    lines += [
        "",
        f"Cross-case measured state error={state_error:.3g}; context error={context_error:.3g}. State and parameter checks precede ranking.",
        "",
        "## Leaderboard",
        "",
        "| Rank | Method | success_rate | cost_time_ms |",
        "|---:|---|---:|---:|",
    ]
    for rank, r in enumerate(
        sorted(results, key=lambda r: (-r["success_rate"], r["cost_time_ms"])), 1
    ):
        lines.append(
            f"| {rank} | {r['name']} | {r['success_rate']:.3f} | {r['cost_time_ms']:.3f} |"
        )
    lines += [
        "",
        "Seeds99010+sample index, two warmups per case, measured on a shared GPU; timings are limited to this workload.",
        "",
    ]
    report = args.output / "report.md"
    report.write_text("\n".join(lines))
    print(f"Benchmark report: {report}", flush=True)


if __name__ == "__main__":
    main()
