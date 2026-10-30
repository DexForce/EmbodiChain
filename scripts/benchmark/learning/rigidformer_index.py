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

"""Measure author MOVi index construction memory and latency.

Run: python -m scripts.benchmark.learning.rigidformer_index --output /tmp/index.json
Scope: CPU metadata construction, excluding data IO, model work and imports.
"""

from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import time
import tracemalloc
from typing import Any
from embodichain_models.dynamics.rigidformer._native.data._index import _SceneWindows


def _build(scenes: int, compact: bool) -> Any:
    """Construct the released random-sampling scene mapping or its list baseline."""
    result = _SceneWindows(random_sampling=True) if compact else []
    for i in range(scenes):
        name = f"scene_{i}"
        frames = [Path(name) / f"frame_{j:05d}" for j in range(2, 480)]
        if compact:
            result.add_scene(name, frames, 8, 1)
        else:
            for start in range(len(frames) - 8 + 1):
                selected = frames[start : start + 8]
                result.append(
                    {
                        "dataset": name,
                        "frames": [p.as_posix() for p in selected],
                        "frame_ids": [int(p.name.split("_")[-1]) for p in selected],
                    }
                )
    return result


def _report(path: Path, record: dict[str, Any]) -> None:
    """Write one Markdown report alongside the machine-readable measurements."""
    rows = record["results"]
    lines = [
        "# RigidFormer random-window index",
        "",
        record["scope"],
        "",
        f"Python {record['python']}; CPU {record['cpu']}; one Python thread; "
        f"{record['repeats']} warmed repeats. No tensor dtype/device is involved.",
        "",
        "Memory is a separate construction with tracemalloc: Python allocations only, "
        "not process RSS or total peak memory. GPU metrics are inapplicable.",
        "",
    ]
    tables = [
        (
            "Time & Memory",
            [
                "scenes",
                "implementation",
                "cost_time_ms",
                "cpu_delta_mb",
                "gpu_delta_mb",
                "peak_gpu_mb",
                "retained_python_mb",
                "peak_python_mb",
            ],
            [
                [
                    r["scenes"],
                    r["implementation"],
                    r["median_seconds"] * 1000,
                    "unmeasured",
                    "N/A",
                    "N/A",
                    r["retained_python_bytes"] / 2**20,
                    r["peak_python_bytes"] / 2**20,
                ]
                for r in rows
            ],
        ),
        (
            "Success & Other Metrics",
            ["scenes", "implementation", "windows", "success_rate"],
            [[r["scenes"], r["implementation"], r["windows"], 1.0] for r in rows],
        ),
        (
            "Leaderboard",
            ["scenes", "rank", "implementation", "success_rate", "cost_time_ms"],
            [
                [scenes, i + 1, r["implementation"], 1.0, r["median_seconds"] * 1000]
                for scenes in sorted({r["scenes"] for r in rows})
                for i, r in enumerate(
                    sorted(
                        (r for r in rows if r["scenes"] == scenes),
                        key=lambda r: r["median_seconds"],
                    )
                )
            ],
        ),
    ]
    for title, columns, values in tables:
        lines += [
            f"## {title}",
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        lines += [
            "| "
            + " | ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row)
            + " |"
            for row in values
        ]
        lines += [""]
    path.write_text("\n".join(lines))


def main() -> None:
    """Measure equivalent window counts and scene selection at three scales."""
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--scenes", type=int, nargs="+", default=[1, 32, 960])
    args.add_argument("--repeats", type=int, default=3)
    args.add_argument("--output", type=Path, required=True)
    cfg = args.parse_args()
    if cfg.repeats < 1 or any(s < 1 for s in cfg.scenes):
        args.error("scenes and repeats must be positive")
    results = []
    for scenes in cfg.scenes:
        for compact in (False, True):
            warm = _build(min(scenes, 2), compact)
            del warm
            samples = []
            for _ in range(cfg.repeats):
                gc.collect()
                start = time.perf_counter()
                result = _build(scenes, compact)
                elapsed = time.perf_counter() - start
                assert len(result) == scenes * 471
                for idx in (0, len(result) // 2, len(result) - 1):
                    assert result[idx]["dataset"] == f"scene_{idx//471}"
                samples.append(elapsed)
                del result
            gc.collect()
            tracemalloc.start()
            result = _build(scenes, compact)
            retained, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del result
            results.append(
                {
                    "scenes": scenes,
                    "windows": scenes * 471,
                    "implementation": "compact" if compact else "author-list",
                    "median_seconds": statistics.median(samples),
                    "samples_seconds": samples,
                    "retained_python_bytes": retained,
                    "peak_python_bytes": peak,
                }
            )
    record = {
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_threads": 1,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "seed": None,
        "inputs": "deterministic 480-frame scene paths; no random payloads",
        "source_sha256": __import__("hashlib")
        .sha256(Path(__file__).read_bytes())
        .hexdigest(),
        "scope": "CPU index construction only; no IO/model/GPU; single process",
        "frames": 480,
        "skip_first": 2,
        "sequence_length": 8,
        "repeats": cfg.repeats,
        "results": results,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(record, indent=2))
    _report(cfg.output.with_suffix(".md"), record)
    print(json.dumps(record, indent=2))
    print(f"Markdown report saved: {cfg.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
