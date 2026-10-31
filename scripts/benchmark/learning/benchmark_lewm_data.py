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

"""Compare transfer-inclusive disk and bounded resident LeWM batches.

Run: python scripts/benchmark/learning/benchmark_lewm_data.py --corpus FILE --output DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import psutil
import torch

from embodichain_models.dynamics.lewm import LeWMSequenceSource


def run_all_benchmarks() -> None:
    """Check identical windows before measuring warmed batches and cache storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--cache-mib", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    setup = {}
    rows = []
    for name, options in [
        ("disk", {}),
        (
            "resident",
            {"cache_device": args.device, "cache_max_bytes": args.cache_mib * 1024**2},
        ),
    ]:
        started = time.perf_counter()
        source = LeWMSequenceSource(
            args.corpus, "train", 3, action_block=args.action_block, **options
        )
        sources[name] = source
        torch.cuda.synchronize(args.device)
        setup[name] = time.perf_counter() - started
        for batch_size in (1, 128):
            if name == "resident":
                expected = (
                    sources["disk"]
                    .sample(batch_size, generator=torch.Generator().manual_seed(7))
                    .to(args.device)
                )
                actual = source.sample(
                    batch_size, generator=torch.Generator().manual_seed(7)
                ).to(args.device)
                for key in expected.inputs:
                    torch.testing.assert_close(
                        actual.inputs[key], expected.inputs[key], atol=0, rtol=0
                    )
                del expected, actual
            generator = torch.Generator().manual_seed(23)

            def operation():
                return source.sample(batch_size, generator=generator).to(args.device)

            for _ in range(3):
                result = operation()
                torch.cuda.synchronize(args.device)
                del result
            samples = []
            for _ in range(args.repeats):
                torch.cuda.synchronize(args.device)
                started = time.perf_counter()
                result = operation()
                torch.cuda.synchronize(args.device)
                samples.append((time.perf_counter() - started) * 1000)
                del result
            before = torch.cuda.memory_allocated(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            rss = psutil.Process().memory_info().rss
            result = operation()
            torch.cuda.synchronize(args.device)
            rows.append(
                {
                    "batch_size": batch_size,
                    "implementation": name,
                    "cost_time_ms": statistics.median(samples),
                    "samples_ms": samples,
                    "cpu_delta_mb": (psutil.Process().memory_info().rss - rss)
                    / 1024**2,
                    "gpu_delta_mb": (torch.cuda.memory_allocated(args.device) - before)
                    / 1024**2,
                    "peak_gpu_mb": torch.cuda.max_memory_allocated(args.device)
                    / 1024**2,
                    "persistent_cache_bytes": source.cache_bytes,
                    "correctness": "exact uint8 pixels and FP32 controls; identical seeded windows",
                }
            )
            del result
    report = {
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "source_id": sources["disk"].source_id,
        "action_block": args.action_block,
        "history_size": 3,
        "warmup": 3,
        "repeats": args.repeats,
        "scope": "completed data sample plus host-to-device transfer; model computation excluded; peak memory is Torch allocator only; RSS is retained delta",
        "setup_seconds": setup,
        "rows": rows,
    }
    (output / "data-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# LeWM data benchmark",
        "",
        report["scope"],
        "",
        "## Time & Memory",
        "",
        "| batch_size | impl | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['implementation']} | {row['cost_time_ms']:.3f} | {row['cpu_delta_mb']:.2f} | {row['gpu_delta_mb']:.2f} | {row['peak_gpu_mb']:.2f} |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| impl | exact comparison | persistent cache bytes | setup seconds |",
        "|---|---|---:|---:|",
    ]
    for name, source in sources.items():
        lines.append(f"| {name} | passed | {source.cache_bytes} | {setup[name]:.3f} |")
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | batch_size | cost_time_ms |",
        "|---:|---|---:|---:|",
    ]
    for index, row in enumerate(
        sorted(
            (row for row in rows if row["batch_size"] == 128),
            key=lambda x: x["cost_time_ms"],
        ),
        1,
    ):
        lines.append(
            f"| {index} | {row['implementation']} | {row['batch_size']} | {row['cost_time_ms']:.3f} |"
        )
    (output / "data-benchmark.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    print("Markdown report saved:", output / "data-benchmark.md")


if __name__ == "__main__":
    run_all_benchmarks()
