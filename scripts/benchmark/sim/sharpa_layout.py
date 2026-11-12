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

"""Measure native eight-scale layouts with the same source policy and task laws.

Run: python scripts/benchmark/sim/sharpa_layout.py --help
Each layout runs in a fresh process. Hardware is shared; repeated synchronized
latencies are observations, not a controlled algorithm-performance comparison.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

__all__ = ["main"]


def main() -> None:
    """Measure end-to-end reset/two-step cycles and write reproducible JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=["eight", "mixed"], required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.prototype.resolve()))
    from _fast_randomized_task import _FastMultiScaleTask
    from _mixed_task import _MixedScaleTask
    from sharpa_rl_unilab.training.configuration import configure_threads
    from sharpa_rl_unilab.training.policy import inference_distribution, make_actor

    snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(snapshot["config"])
    cfg.training.device = "cuda:0"
    configure_threads(cfg)
    torch.manual_seed(99001)
    actor = make_actor(cfg, "cuda:0", stage="teacher")
    actor.load_state_dict(snapshot["actor"])
    actor.eval()
    factory = _FastMultiScaleTask if args.case == "eight" else _MixedScaleTask
    task = None
    try:
        started = time.perf_counter()
        task = factory(
            cfg,
            args.upstream.resolve(),
            args.output,
            num_envs=args.num_envs,
            seed=99001,
        )
        setup = time.perf_counter() - started
        timings = []
        reset_times = []
        finite = True
        with torch.inference_mode():
            for index in range(args.repeats + 2):
                torch.cuda.synchronize()
                started = time.perf_counter()
                obs, _ = task.reset(seed=99010 + index)
                torch.cuda.synchronize()
                after_reset = time.perf_counter()
                for _ in range(2):
                    action = inference_distribution(
                        actor, obs, task.device
                    ).deterministic()
                    obs = task.step(action).obs
                torch.cuda.synchronize()
                finished = time.perf_counter()
                finite = finite and bool(torch.isfinite(task.s).all())
                if index >= 2:
                    timings.append((finished - started) * 1000)
                    reset_times.append((after_reset - started) * 1000)
        import psutil

        result = {
            "case": args.case,
            "num_envs": args.num_envs,
            "setup_seconds": setup,
            "cost_time_ms": statistics.median(timings),
            "samples_ms": timings,
            "reset_median_ms": statistics.median(reset_times),
            "repeats": args.repeats,
            "warmup": 2,
            "success_rate": float(finite),
            "cpu_delta_mb": None,
            "gpu_delta_mb": None,
            "peak_gpu_mb": None,
            "retained_cpu_mb": psutil.Process().memory_info().rss / 2**20,
            "retained_torch_gpu_mb": torch.cuda.memory_allocated() / 2**20,
            "hardware": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "dtype": "float32",
            "scope": "CUDA-synchronized wall time: full reset plus two native control steps and source-policy inference; separate setup, two warmup cycles",
            "inputs": "Same eight scales, equal counts, source caches, parameter distributions and source checkpoint. Per-layout random streams differ. Shared device load can vary.",
            "memory": "Retained process/Torch snapshots only; peak and incremental memory unmeasured, Warp/native excluded from Torch allocation.",
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        lines = [
            "# Native eight-scale layout measurement",
            "",
            result["scope"],
            "",
            result["inputs"],
            "",
            "## Time & Memory",
            "",
            "| layout | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
            "| --- | --- | --- | --- | --- |",
            f"| {args.case} | {result['cost_time_ms']:.3f} | unmeasured | unmeasured | unmeasured |",
            "",
            "## Success & Other Metrics",
            "",
            "| layout | success_rate | reset_ms | environments |",
            "| --- | --- | --- | --- |",
            f"| {args.case} | {result['success_rate']} | {result['reset_median_ms']:.3f} | {args.num_envs} |",
            "",
            "## Leaderboard",
            "",
            "| rank | algorithm | overall_success_rate |",
            "| --- | --- | --- |",
            f"| 1 | {args.case} | {result['success_rate']} |",
            "",
            result["memory"],
        ]
        (args.output / "report.md").write_text("\n".join(lines) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()
