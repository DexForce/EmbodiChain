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

"""Measure the CPU frame path on a real video or HDF5 recording, writing one Markdown report.

Run: python -m scripts.benchmark.toolkits.trajectory_annotation --video episode.mp4
Setup includes hashing and adapter preparation (a full PTS scan for video). Repeated calls measure completed wall latency;
no HTTP/browser work is timed. Inputs are unchanged and no precision variants or
speedup claims are compared. RSS sampling includes native decoder allocations.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import platform
import random
import statistics
import threading
import time
from typing import Any

import av
import psutil

from embodichain.toolkits.trajectory_annotator._cfg import AnnotationCfg
from embodichain.toolkits.trajectory_annotator._frames import FrameReader


def measure(
    operation: Callable[[], object], repeats: int, check: Callable[[object], bool]
) -> tuple[float, float, int]:
    """Measure median/P95 completed-call milliseconds after caller warmup."""
    samples: list[float] = []
    succeeded = 0
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        samples.append((time.perf_counter() - started) * 1000)
        succeeded += int(check(result))
        del result
    return (
        statistics.median(samples),
        sorted(samples)[min(repeats - 1, int(repeats * 0.95))],
        succeeded,
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render one table with a fixed header derived from its first row."""
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[header]) for header in headers) + " |" for row in rows
    )
    return "\n".join(lines)


def run_all_benchmarks() -> None:
    """Measure initialization, uncached/random, cached and sequential previews."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--hdf5", type=Path)
    parser.add_argument("--image-key")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    process = psutil.Process()
    mib = 1024**2
    baseline = process.memory_info().rss
    peak = [baseline]
    stopped = threading.Event()

    def sample_rss() -> None:
        while not stopped.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample_rss, daemon=True)
    monitor.start()
    cfg = AnnotationCfg(cache_bytes=0)
    input_path = args.video or args.hdf5
    if args.video:
        from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource

        adapter = VideoFrameSource(args.video)
    else:
        from embodichain.toolkits.trajectory_annotator.hdf5 import Hdf5FrameSource

        adapter = Hdf5FrameSource(args.hdf5, image_key=args.image_key, fps=args.fps)
    reader = FrameReader(adapter, cfg)
    rows: list[dict[str, Any]] = []
    succeeded = attempts = 0
    try:
        started = time.perf_counter()
        source = reader.prepare()
        index_ms = (time.perf_counter() - started) * 1000
        index_rss = process.memory_info().rss
        rows.append(
            dict(
                phase="cold source preparation",
                cost_time_ms=round(index_ms, 3),
                p95_ms="N/A",
                cpu_delta_mb=round((index_rss - baseline) / mib, 3),
                gpu_delta_mb="N/A",
                peak_gpu_mb="N/A",
            )
        )
        rng = random.Random(1024)
        indices = [rng.randrange(source.frame_count) for _ in range(args.repeats)]

        def query(index: int) -> tuple[int, object]:
            return index, reader.preview(index)

        def check(result: Any) -> bool:
            index, preview = result
            if source.kind == "video":
                return preview.index == index and preview.pts == adapter.pts[index]
            return preview.index == index and preview.time_seconds == index / source.fps

        query(indices[0])  # Warm JPEG/native conversion; cache remains disabled.
        selection = iter(indices)
        before = process.memory_info().rss
        median, p95, passed = measure(
            lambda: query(next(selection)), args.repeats, check
        )
        succeeded += passed
        attempts += args.repeats
        rows.append(
            dict(
                phase="uncached random preview",
                cost_time_ms=round(median, 3),
                p95_ms=round(p95, 3),
                cpu_delta_mb=round((process.memory_info().rss - before) / mib, 3),
                gpu_delta_mb="N/A",
                peak_gpu_mb="N/A",
            )
        )
        cfg.cache_bytes = 64 * mib
        query(indices[0])
        before = process.memory_info().rss
        median, p95, passed = measure(lambda: query(indices[0]), args.repeats, check)
        succeeded += passed
        attempts += args.repeats
        rows.append(
            dict(
                phase="cached frame",
                cost_time_ms=round(median, 3),
                p95_ms=round(p95, 3),
                cpu_delta_mb=round((process.memory_info().rss - before) / mib, 3),
                gpu_delta_mb="N/A",
                peak_gpu_mb="N/A",
            )
        )
        cfg.cache_bytes = 0
        reader.cache.clear()
        reader.cache_bytes = 0
        count = min(source.frame_count, args.repeats)
        query(0)
        sequential = iter(range(count))
        before = process.memory_info().rss
        median, p95, passed = measure(lambda: query(next(sequential)), count, check)
        succeeded += passed
        attempts += count
        rows.append(
            dict(
                phase="sequential preview",
                cost_time_ms=round(median, 3),
                p95_ms=round(p95, 3),
                cpu_delta_mb=round((process.memory_info().rss - before) / mib, 3),
                gpu_delta_mb="N/A",
                peak_gpu_mb="N/A",
            )
        )
        packed_bytes = (
            len(adapter.pts) * (adapter.pts.itemsize + adapter.anchors.itemsize)
            if source.kind == "video"
            else 0
        )
        algorithm = f"{source.kind} CPU"
        metrics = [
            dict(
                algorithm=algorithm,
                success_rate=f"{succeeded / attempts:.2%}",
                checked_requests=attempts,
                frames=source.frame_count,
                packed_index_mib=round(packed_bytes / mib, 3),
                jpeg_cache_mib=round(reader.cache_bytes / mib, 3),
            )
        ]
    finally:
        stopped.set()
        monitor.join()
        reader.close()
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    report = (
        args.output
        or Path("outputs/benchmarks")
        / f"trajectory_annotation_{datetime.now():%Y%m%d_%H%M%S}.md"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n\n".join(
            [
                "# Trajectory annotation CPU benchmark",
                f"Input: `{input_path.resolve()}`\n\nSource: `{source}`\n\n"
                f"CPU: {cpu}; OS: {platform.platform()}; Python: {platform.python_version()}; "
                f"PyAV: {av.__version__}; h5py: {version('h5py')}; OpenCV: {version('opencv-python')}; FFmpeg: {av.library_versions}. "
                f"Decoder threads: {cfg.decoder_threads}; preview edge: {cfg.preview_edge}; JPEG quality: {cfg.jpeg_quality}. "
                f"Seed: 1024; repeats: {args.repeats}; warmup: one call per frame path. "
                f"Sampled process peak RSS (10 ms sampling): {peak[0] / mib:.2f} MiB; "
                f"increase over pre-index runtime: {(peak[0] - baseline) / mib:.2f} MiB. "
                "Sampling can miss transient peaks. CPU deltas below are retained RSS changes, not peak allocations. "
                "Initialization includes hashing; video scans all PTS, HDF5 decodes only the selected first image. Frame paths include resize/JPEG encoding on misses. "
                "HTTP, browser rendering, input generation and logging are excluded. "
                "Returned index/timing agreement is checked; independent pixel identity is covered by production tests. "
                "GPU fields are inapplicable. This is a single implementation, without a comparative speedup claim.",
                "## Time & Memory\n\n" + markdown_table(rows),
                "## Success & Other Metrics\n\n" + markdown_table(metrics),
                "## Leaderboard\n\n"
                + markdown_table(
                    [
                        dict(
                            rank=1,
                            algorithm=algorithm,
                            overall_success_rate=metrics[0]["success_rate"],
                        )
                    ]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Markdown report saved: {report}")


if __name__ == "__main__":
    run_all_benchmarks()
