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

"""Measure row-local eligibility/check/snapshot overhead without simulation.

Run: python -m scripts.benchmark.gym.episode_context --repeats 40
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import psutil
import torch

from embodichain.lab.gym.envs.episode_context import EpisodeContext

__all__ = ["main"]


def _save_metadata(
    context: EpisodeContext,
    selection: list[int],
    directory: Path,
    *,
    include_context: bool,
) -> None:
    """Publish per-row sidecars; exclude video encoding from metadata overhead."""
    snapshots = (
        context.snapshot(selection) if include_context else [{} for _ in selection]
    )
    for row, snapshot in zip(selection, snapshots, strict=True):
        payload = {"artifact_purpose": "recording", "env_id": row, "frames": 4}
        if include_context:
            payload["episode_context"] = snapshot
        staged = directory / f"row_{row}.tmp"
        staged.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(staged, directory / f"row_{row}.json")


def main() -> None:
    """Write one three-table report for small, typical and larger row counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.set_num_threads(1)
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    rows = []
    with tempfile.TemporaryDirectory(prefix="episode-context-benchmark-") as temporary:
        _measure_contexts(devices, rows, args.repeats, Path(temporary))
    _write_report(args.output_dir, devices, rows, args.repeats)


def _measure_contexts(
    devices: list[str], rows: list[dict], repeats: int, directory: Path
) -> None:
    """Measure warmed host operations and row-wise metadata publication."""
    for device in devices:
        synchronize = (
            (lambda: torch.cuda.synchronize(device))
            if device.startswith("cuda")
            else (lambda: None)
        )
        for count in (1, 4, 64, 256):
            for sources in (1, 2):
                context = EpisodeContext(count, device)
                selection = list(range(count))
                context._begin_episode(selection)
                context._finish_initialization(selection)

                def checks():
                    for source in range(sources):
                        with context.check(f"events/{source}", selection) as finish:
                            finish(
                                accepted=True,
                                metadata={"variant_id": "example", "seed": 42},
                            )

                operations = {
                    "checks": checks,
                    "filter": lambda: context.filter_recordable(selection),
                    "snapshot": lambda: context.snapshot(selection),
                    "save_baseline_metadata": lambda: _save_metadata(
                        context, selection, directory, include_context=False
                    ),
                    "save_context_metadata": lambda: _save_metadata(
                        context, selection, directory, include_context=True
                    ),
                }
                for operation, call in operations.items():
                    for _ in range(5):
                        result = call()
                        synchronize()
                        del result
                    samples = []
                    for _ in range(repeats):
                        synchronize()
                        start = time.perf_counter()
                        result = call()
                        synchronize()
                        samples.append((time.perf_counter() - start) * 1000)
                        del result
                    # Memory is a separate warmed pass retaining one result.
                    before = psutil.Process().memory_info().rss
                    gpu_before = (
                        torch.cuda.memory_allocated(device)
                        if device.startswith("cuda")
                        else None
                    )
                    if gpu_before is not None:
                        torch.cuda.reset_peak_memory_stats(device)
                    result = call()
                    synchronize()
                    memory = (psutil.Process().memory_info().rss - before) / 1024**2
                    peak = (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if gpu_before is not None
                        else None
                    )
                    delta = (
                        (torch.cuda.memory_allocated(device) - gpu_before) / 1024**2
                        if gpu_before is not None
                        else None
                    )
                    del result
                    assert len(context.filter_recordable()) == count
                    rows.append(
                        {
                            "device": device,
                            "rows": count,
                            "sources": sources,
                            "operation": operation,
                            "cost_time_ms": statistics.median(samples),
                            "cpu_delta_mb": memory,
                            "gpu_delta_mb": delta,
                            "peak_gpu_mb": peak,
                            "success_rate": 1.0,
                        }
                    )
                saved = json.loads((directory / "row_0.json").read_text())
                assert saved["episode_context"] == context.snapshot([0])[0]


def _write_report(
    output_dir: Path, devices: list[str], rows: list[dict], repeats: int
) -> None:
    """Write exactly three tables and record the measurement scope."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"episode_context_{stamp}"
    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repeats": repeats,
        "warmup": 5,
        "threads": 1,
        "scope": "Completed host calls, CUDA synchronized; filtering allocates owned IDs and snapshots include JSON copies. Save cases publish per-row JSON sidecars with atomic rename to the same temporary filesystem, without fsync or video encoding; baseline omits only Context. RSS is a warmed retained delta; CUDA peaks cover the Torch allocator only.",
    }
    path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n"
    )
    lines = [
        "# EpisodeContext benchmark",
        "",
        json.dumps(metadata),
        "",
        "## Time & Memory",
        "",
        "| Device | Rows | Sources | Operation | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [
            row[k]
            for k in (
                "device",
                "rows",
                "sources",
                "operation",
                "cost_time_ms",
                "cpu_delta_mb",
                "gpu_delta_mb",
                "peak_gpu_mb",
            )
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Success & Other Metrics",
            "",
            "| Device | Rows | Sources | success_rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["operation"] == "checks":
            lines.append(
                f"| {row['device']} | {row['rows']} | {row['sources']} | 1.0 |"
            )
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Device | Overall success rate |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| {device} | 1.0 |" for device in devices)
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"Markdown report saved: {path.with_suffix('.md').resolve()}")


if __name__ == "__main__":
    main()
