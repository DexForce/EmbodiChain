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

"""Measure exact minibatch-order storage and replay with bounded RNG checkpoints.

Run: python -m embodichain run-example learning/nerd/_support/_benchmark_dynamics_sampling --output-dir PATH
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import psutil
import torch

from examples.learning.nerd._support._sample_order import (
    _open_order,
    _write_checkpointed_order,
)

__all__: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4097)
    parser.add_argument("--full-budget", action="store_true")
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    for batch in [1, 16, 512]:
        generator = torch.Generator().manual_seed(7)
        matrix = np.lib.format.open_memmap(
            root / f"matrix-{batch}.npy",
            mode="w+",
            dtype=np.int64,
            shape=(args.steps, batch),
        )
        for start in range(0, args.steps, 1024):
            matrix[start : start + 1024] = torch.randint(
                1000000, (min(1024, args.steps - start), batch), generator=generator
            ).numpy()
        matrix.flush()
        del matrix
        start = time.perf_counter()
        descriptor, state = _write_checkpointed_order(
            root / f"order-{batch}.json",
            dataset_size=1000000,
            steps=args.steps,
            batch_size=batch,
            seed=7,
        )
        prepare_ms = (time.perf_counter() - start) * 1000
        expected = np.load(root / f"matrix-{batch}.npy", mmap_mode="r")
        stream = _open_order(descriptor)
        for index in range(args.steps):
            np.testing.assert_array_equal(stream[index], expected[index])
        del stream, expected
        gc.collect()
        for kind, path in [
            ("matrix", root / f"matrix-{batch}.npy"),
            ("checkpoint", descriptor),
        ]:
            samples = []
            deltas = []
            # Completed CPU replay windows; setup/verification outside timing.
            for repeat in range(4):
                order = _open_order(path)
                rss = psutil.Process().memory_info().rss
                start = time.perf_counter()
                for index in range(args.steps):
                    value = order[index]
                elapsed = (time.perf_counter() - start) * 1000 / args.steps
                if repeat:
                    samples.append(elapsed)
                    deltas.append((psutil.Process().memory_info().rss - rss) / 1024**2)
                del order, value
                gc.collect()
            size = path.stat().st_size + (
                state.stat().st_size if kind == "checkpoint" else 0
            )
            rows.append(
                {
                    "implementation": kind,
                    "batch_size": batch,
                    "cost_time_ms": statistics.median(samples),
                    "cpu_delta_mb": statistics.median(deltas),
                    "gpu_delta_mb": 0,
                    "peak_gpu_mb": 0,
                    "storage_bytes": size,
                    "success_rate": 1.0,
                    "prepare_ms": prepare_ms if kind == "checkpoint" else None,
                }
            )
    full = None
    if args.full_budget:
        start = time.perf_counter()
        descriptor, state = _write_checkpointed_order(
            root / "full-order.json",
            dataset_size=1000000,
            steps=4995000,
            batch_size=512,
            seed=7,
        )
        full = {
            "steps": 4995000,
            "batch": 512,
            "actual_storage_bytes": descriptor.stat().st_size + state.stat().st_size,
            "dense_index_bytes": 4995000 * 512 * 8,
            "prepare_seconds": time.perf_counter() - start,
        }
    result = {
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "threads": 1,
        "steps": args.steps,
        "warmup_windows": 1,
        "repeats": 3,
        "scope": "CPU exact order replay; RSS deltas are not peak memory; GPU unused.",
        "rows": rows,
        "full_budget": full,
    }
    (root / "results.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Sampling order benchmark",
        "",
        result["scope"],
        "",
        "## Time & Memory",
        "",
        "| Algorithm | Batch | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['implementation']} | {row['batch_size']} | {row['cost_time_ms']:.6f} | {row['cpu_delta_mb']:.3f} | 0 | 0 |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Algorithm | Batch | success_rate | storage_bytes |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['implementation']} | {row['batch_size']} | 1.0 | {row['storage_bytes']} |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| Algorithm | Batch | success_rate |",
        "|---|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: -x["success_rate"]):
        lines.append(f"| {row['implementation']} | {row['batch_size']} | 1.0 |")
    if full:
        lines += [
            "",
            f"Measured full-budget checkpoint storage: {full['actual_storage_bytes']} bytes versus {full['dense_index_bytes']} dense index bytes; setup {full['prepare_seconds']:.3f}s.",
        ]
    report = root / "REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print(report)


if __name__ == "__main__":
    main()
