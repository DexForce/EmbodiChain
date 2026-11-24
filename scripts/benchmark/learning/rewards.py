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

"""Measure composed RewardManager latency, correctness and allocator memory.

Run: python -m scripts.benchmark.learning.rewards --output outputs/reward_qualification
This measures device-resident reward composition, excluding native acquisition.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import psutil
import torch

from embodichain.lab.gym.envs.managers.reward_composition import compose_reward_terms
from embodichain.lab.gym.envs.managers.reward_context import RewardTransitionContext
from embodichain.lab.gym.envs.managers.reward_manager import RewardManager


def measure(batch: int, device: torch.device, repeats: int) -> dict[str, object]:
    """Measure one fixed batch after independent warmup and correctness checks."""
    env = SimpleNamespace(num_envs=batch, device=device, sim=SimpleNamespace())
    cfg = compose_reward_terms(
        {
            "discount": 0.99,
            "skills": [
                {"skill_id": "move_joints", "metrics": {"joints": {"field": "score"}}}
            ],
        },
        program=False,
    )
    manager = RewardManager(cfg, env)
    env.reward_manager = manager
    value = torch.full((batch,), 0.5, device=device)
    sample = SimpleNamespace(
        values=value, valid=torch.ones(batch, dtype=torch.bool, device=device)
    )
    frame = SimpleNamespace(
        fields={"score": sample}, revision=0, episode_ids=(0,) * batch
    )
    manager.initialize_observation(frame, range(batch))
    flags = torch.zeros(batch, dtype=torch.bool, device=device)
    action = torch.zeros(batch, 1, device=device)

    def operation():
        frame.revision += 1
        manager.transition_context = RewardTransitionContext(frame, flags)
        return manager.compute({}, action, {})[0]

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(10):
        operation()
    synchronize()
    expected = torch.full_like(value, -0.005)
    result = operation()
    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-5)
    samples = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        result = operation()
        synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    synchronize()
    rss = psutil.Process().memory_info().rss
    allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if allocated is not None:
        torch.cuda.reset_peak_memory_stats(device)
    result = operation()
    synchronize()
    return {
        "batch": batch,
        "device": str(device),
        "cost_time_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))],
        "repeats": repeats,
        "cpu_delta_mb": (psutil.Process().memory_info().rss - rss) / 2**20,
        "gpu_delta_mb": (
            None
            if allocated is None
            else (torch.cuda.memory_allocated(device) - allocated) / 2**20
        ),
        "peak_gpu_mb": (
            None
            if allocated is None
            else torch.cuda.max_memory_allocated(device) / 2**20
        ),
        "peak_gpu_delta_mb": (
            None
            if allocated is None
            else (torch.cuda.max_memory_allocated(device) - allocated) / 2**20
        ),
        "success_rate": float(
            torch.isclose(result, expected, atol=1e-7, rtol=1e-5).float().mean().item()
        ),
        "max_absolute_error": float((result - expected).abs().max().item()),
    }


def run_all_benchmarks() -> None:
    """Write one report with explicit timing scope, hardware and three tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reward_qualification")
    )
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("At least five repeats are required")
    torch.set_num_threads(1)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    rows = [
        measure(batch, device, args.repeats)
        for device in devices
        for batch in (1, 64, 1024)
    ]
    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": 1,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scope": "resident RewardManager composition; no simulator, policy inference or contact acquisition",
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = [
        "# Reward composition benchmark",
        "",
        metadata["scope"],
        "",
        f"Python {metadata['python']}; Torch {metadata['torch']}; GPU {metadata['gpu']}; CPU threads 1; warmup 10.",
        "CPU RSS is retained delta; GPU values cover the Torch allocator. Correctness success_rate is numerical agreement, not task success.",
        "",
        "## Time & Memory",
        "",
        "| Device / batch | cost_time_ms | p95_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['device']} / {row['batch']} | {row['cost_time_ms']:.4f} | {row['p95_ms']:.4f} | {row['cpu_delta_mb']} | {row['gpu_delta_mb']} | {row['peak_gpu_mb']} |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Device / batch | success_rate | max_absolute_error | repeats |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['device']} / {row['batch']} | {row['success_rate']} | {row['max_absolute_error']:.3g} | {row['repeats']} |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| Rank | Device / batch | success_rate |",
        "|---|---|---:|",
    ]
    for index, row in enumerate(sorted(rows, key=lambda r: -r["success_rate"]), 1):
        lines.append(
            f"| {index} | {row['device']} / {row['batch']} | {row['success_rate']} |"
        )
    report = args.output / "benchmark.md"
    report.write_text("\n".join(lines) + "\n")
    print(report.resolve())


if __name__ == "__main__":
    run_all_benchmarks()
