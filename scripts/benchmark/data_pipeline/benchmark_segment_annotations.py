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

"""Compare segment annotation lookup without initializing a simulator.

Checks output equivalence between repeated range scans and one validated frame
map. This measures annotation lookup only, not dataset save or simulation time.
Run: python -m scripts.benchmark.data_pipeline.benchmark_segment_annotations
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import psutil

from embodichain.data_pipeline.recording.segments import segments_for_frames


def _scan(metadata: Mapping[str, Any], length: int) -> list[Mapping[str, Any] | None]:
    """Reproduce the previous per-frame, first-matching-range scan."""
    return [
        next(
            (
                segment
                for segment in metadata["segments"]
                if segment["start_step"] <= index < segment["end_step"]
            ),
            None,
        )
        for index in range(length)
    ]


def _measure(
    name: str,
    implementation: Callable[..., list[Mapping[str, Any] | None]],
    metadata: Mapping[str, Any],
    length: int,
    repeats: int,
    expected: list[Mapping[str, Any] | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure warmed CPU lookup and verify every result outside timing."""
    process = psutil.Process()
    implementation(metadata, length)
    elapsed: list[float] = []
    memory: list[float] = []
    passed = 0
    for _ in range(repeats):
        before = process.memory_info().rss
        start = time.perf_counter()
        result = implementation(metadata, length)
        elapsed.append((time.perf_counter() - start) * 1000)
        memory.append((process.memory_info().rss - before) / (1024 * 1024))
        passed += result == expected
    return (
        {
            "algorithm": name,
            "cost_time_ms": statistics.median(elapsed),
            "cpu_delta_mb": statistics.median(memory),
            "gpu_delta_mb": 0.0,
            "peak_gpu_mb": 0.0,
        },
        {
            "algorithm": name,
            "success_rate": passed / repeats,
            "frames": length,
            "segments": len(metadata["segments"]),
            "repeats": repeats,
        },
    )


def _table(rows: list[dict[str, Any]]) -> list[str]:
    """Render one Markdown table with numeric values rounded for comparison."""
    keys = list(rows[0])
    output = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for row in rows:
        output.append(
            "| "
            + " | ".join(
                f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key])
                for key in keys
            )
            + " |"
        )
    return output


def run_all_benchmarks(frames: int, segments: int, repeats: int, report: Path) -> None:
    """Benchmark both lookups and save three tables to a single report."""
    if frames < 1 or segments < 1 or segments > frames or repeats < 1:
        raise ValueError("Require frames >= segments >= 1 and repeats >= 1.")
    metadata = {
        "segments": [
            {
                "segment_id": index,
                "start_step": frames * index // segments,
                "end_step": frames * (index + 1) // segments,
                "instruction": f"Segment {index}",
            }
            for index in range(segments)
        ]
    }
    expected = _scan(metadata, frames)
    perf, metrics = [], []
    for name, implementation in (
        ("repeated_scan", _scan),
        ("frame_map", segments_for_frames),
    ):
        timing, quality = _measure(
            name, implementation, metadata, frames, repeats, expected
        )
        perf.append(timing)
        metrics.append(quality)
        print(
            f"{name}: {timing['cost_time_ms']:.3f} ms, matches={quality['success_rate']:.0%}"
        )
    leaderboard = [
        {
            "rank": index + 1,
            "algorithm": row["algorithm"],
            "success_rate": next(
                metric["success_rate"]
                for metric in metrics
                if metric["algorithm"] == row["algorithm"]
            ),
            "cost_time_ms": row["cost_time_ms"],
        }
        for index, row in enumerate(
            sorted(
                perf,
                key=lambda row: (
                    -next(
                        metric["success_rate"]
                        for metric in metrics
                        if metric["algorithm"] == row["algorithm"]
                    ),
                    row["cost_time_ms"],
                ),
            )
        )
    ]
    lines = [
        "# Segment annotation lookup benchmark",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Runtime: Python {platform.python_version()}, {platform.platform()}, {platform.processor() or 'CPU unspecified'}",
        "",
        "CPU-only; zero GPU allocation. RSS deltas are sampled, not peak memory. Timing excludes input generation and result verification. Results do not measure simulation or disk throughput.",
        "",
    ]
    for title, rows in (
        ("Time & Memory", perf),
        ("Success & Other Metrics", metrics),
        ("Leaderboard", leaderboard),
    ):
        lines.extend([f"## {title}", "", *_table(rows), ""])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {report.resolve()}")
    if any(row["success_rate"] != 1 for row in metrics):
        raise RuntimeError("Segment annotation outputs differ.")


def main() -> None:
    """Parse benchmark sizes and report path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=10000)
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/benchmarks/segment_annotations.md")
    )
    args = parser.parse_args()
    run_all_benchmarks(args.frames, args.segments, args.repeats, args.report)


if __name__ == "__main__":
    main()
