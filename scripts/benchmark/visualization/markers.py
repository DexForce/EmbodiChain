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

"""Measure marker batch state updates and detached CPU snapshot generation.

Run: python -m scripts.benchmark.visualization.markers --repeats 5
This does not measure native GPU rendering, Viser encoding/network transport,
or GPU instancing. Array bytes describe decoded snapshot payload only.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from embodichain.lab.visualization.markers import (
    MarkerGroup,
    MarkerGroupCfg,
    MarkerPrototypeCfg,
)


def _measure(operation: Callable[[], object], repeats: int) -> dict[str, float]:
    """Measure a warmed CPU operation and process/Torch memory deltas."""
    operation()
    process = psutil.Process()
    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    cpu_before = process.memory_info().rss
    gpu_before = torch.cuda.memory_allocated() if gpu else 0
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(
        cost_time_ms=float(np.median(samples)),
        p95_ms=float(np.percentile(samples, 95)),
        cpu_delta_mb=(process.memory_info().rss - cpu_before) / 2**20,
        gpu_delta_mb=((torch.cuda.memory_allocated() if gpu else 0) - gpu_before)
        / 2**20,
        peak_gpu_mb=(torch.cuda.max_memory_allocated() if gpu else 0) / 2**20,
    )


def _table(headers: list[str], rows: list[dict[str, object]]) -> str:
    """Write a compact Markdown table with comparable numeric precision."""

    def display(value: object) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *(
                "| " + " | ".join(display(row[key]) for key in headers) + " |"
                for row in rows
            ),
        ]
    )


def run_all_benchmarks(args: argparse.Namespace) -> Path:
    """Measure environment/instance scaling and write one report."""
    performance, metrics = [], []
    for environments in args.envs:
        for markers in args.markers:
            origins = np.zeros((environments, 3), dtype=np.float32)
            origins[:, 0] = np.arange(environments) * 5
            group = MarkerGroup(
                MarkerGroupCfg(
                    name="benchmark", prototypes={"box": MarkerPrototypeCfg()}
                ),
                num_envs=environments,
                origins=origins,
            )
            positions = np.zeros((environments, markers, 3), dtype=np.float32)
            group.update(translations=positions)
            selected = np.arange(0, environments, 4)
            moved = positions[selected].copy()
            moved[..., 2] = 1
            operations = {
                "full_update": lambda: group.update(translations=positions),
                "selected_update": lambda: group.update(
                    translations=moved, env_ids=selected
                ),
                "snapshot": group.snapshot,
            }
            for name, operation in operations.items():
                group.update(translations=positions)
                measurement = _measure(operation, args.repeats)
                snapshots = group.snapshot()
                valid = len(snapshots) == environments * markers
                for snapshot in snapshots:
                    env = snapshot.env_id
                    expected_z = (
                        1.0 if name == "selected_update" and env in selected else 0.0
                    )
                    valid = valid and bool(
                        np.allclose(
                            snapshot.position, origins[env] + [0, 0, expected_z]
                        )
                    )
                if not valid:
                    raise AssertionError(f"Incorrect marker snapshots for {name}")
                size = sum(
                    sum(
                        value.nbytes
                        for value in (
                            item.vertices,
                            item.faces,
                            item.position,
                            item.wxyz,
                            item.scale,
                        )
                    )
                    for item in snapshots
                )
                case = dict(envs=environments, markers_per_env=markers, impl=name)
                performance.append({**case, **measurement})
                metrics.append(
                    {
                        **case,
                        "success_rate": 1.0,
                        "logical_instances": len(snapshots),
                        "selected_envs": (
                            len(selected) if name == "selected_update" else environments
                        ),
                        "snapshot_array_bytes": size,
                    }
                )
                print(
                    f"envs={environments:3d} markers={markers:4d} {name:16s}: "
                    f"{measurement['cost_time_ms']:9.3f} ms | CPU delta "
                    f"{measurement['cpu_delta_mb']:+.2f} MB | "
                    f"Torch GPU delta {measurement['gpu_delta_mb']:+.2f} MB"
                )
            group.remove()
    leaderboard = []
    for rank, operation in enumerate(("full_update", "selected_update", "snapshot"), 1):
        rows = [row for row in performance if row["impl"] == operation]
        leaderboard.append(
            dict(
                rank=rank,
                algorithm=operation,
                success_rate=1.0,
                median_cost_time_ms=float(
                    np.median([row["cost_time_ms"] for row in rows])
                ),
            )
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = args.output or Path(f"outputs/benchmarks/markers_{stamp}.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n\n".join(
            [
                "# Marker CPU batch benchmark",
                f"UTC: {stamp}; repetitions: {args.repeats}; median and p95 of warmed calls.",
                "Measures CPU state validation/update and detached mesh snapshots only. "
                "No native rendering or Viser client is active. GPU columns cover Torch "
                "allocations only (this workload allocates no GPU tensors), not native renderer VRAM. "
                "RSS deltas are allocator-sensitive, not peak live allocation. "
                "Snapshot array bytes omit metadata, encoding and wire overhead. "
                "The three operations do different work; leaderboard ranks are tied by "
                "correctness and are not a speed ranking of interchangeable algorithms.",
                "## Time & Memory",
                _table(
                    [
                        "envs",
                        "markers_per_env",
                        "impl",
                        "cost_time_ms",
                        "p95_ms",
                        "cpu_delta_mb",
                        "gpu_delta_mb",
                        "peak_gpu_mb",
                    ],
                    performance,
                ),
                "## Success & Other Metrics",
                _table(
                    [
                        "envs",
                        "markers_per_env",
                        "impl",
                        "success_rate",
                        "logical_instances",
                        "selected_envs",
                        "snapshot_array_bytes",
                    ],
                    metrics,
                ),
                "## Leaderboard",
                _table(
                    ["rank", "algorithm", "success_rate", "median_cost_time_ms"],
                    leaderboard,
                ),
            ]
        )
        + "\n"
    )
    print(f"Markdown report saved: {report}")
    return report


def main() -> None:
    """Parse positive benchmark sizes and repetitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--markers", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.envs + args.markers):
        parser.error("sizes and repeats must be positive")
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
