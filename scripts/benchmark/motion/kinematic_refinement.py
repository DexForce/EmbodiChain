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

"""Compare replay, rigid-anchor repair and sparse refinement on an analytic oracle.

Run: python -m scripts.benchmark.motion.kinematic_refinement --output /tmp/refinement-benchmark
The exact line and rotations are constructed independently of the optimizer.
Timing includes synchronous CPU preparation, pose fitting and validation; input
generation is excluded. RSS is a separate warmed retained delta, not peak usage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time
from collections.abc import Callable

import numpy as np
import psutil
import scipy
import osqp
from scipy.spatial.transform import Rotation

from embodichain.data_pipeline.real2sim import PhaseAnnotation
from embodichain.lab.sim.motion.refinement import (
    InteractionConstraintSet,
    KinematicMotionRefiner,
    MotionRefinementCfg,
)

__all__: list[str] = []


def fixture(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct an exact straight line with a rigid bias and interior jitter."""
    oracle = np.repeat(np.eye(4)[None], count, axis=0)
    oracle[:, 0, 3] = np.linspace(0, 0.1, count)
    oracle[:, 2, 3] = 0.3
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler("z", 3, degrees=True).as_matrix()
    delta[:3, 3] = [0.006, -0.004, 0.002]
    source = delta @ oracle
    jitter = 0.0015 * np.sin(np.arange(count) * 2.1)
    jitter[:2] = jitter[-2:] = 0
    source[:, 1, 3] += jitter
    return source, oracle


def metrics(output: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    """Measure against independent geometry rather than the solver objective."""
    difference = output[:, :3, 3] - oracle[:, :3, 3]
    error = float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))
    anchor = float(np.linalg.norm(difference[-1]))
    angular = float(
        Rotation.from_matrix(output[:, :3, :3] @ oracle[:, :3, :3].transpose(0, 2, 1))
        .magnitude()
        .max()
    )
    return {
        "position_rmse_mm": error * 1000,
        "anchor_error_mm": anchor * 1000,
        "rotation_error_deg": float(np.rad2deg(angular)),
        "success_rate": float(error < 0.0008 and anchor < 1e-6 and angular < 1e-5),
    }


def measure(
    operation: Callable[[], np.ndarray], repeats: int
) -> tuple[dict[str, object], np.ndarray]:
    """Measure completed warmed CPU calls and one separate retained-memory call."""
    for _ in range(2):
        output = operation()
        del output
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = operation()
        samples.append((time.perf_counter() - start) * 1000)
        del output
    process = psutil.Process()
    before = process.memory_info().rss
    output = operation()
    retained = (process.memory_info().rss - before) / 1024**2
    return {
        "cost_time_ms": statistics.median(samples),
        "samples_ms": samples,
        "cpu_delta_mb": retained,
        "gpu_delta_mb": None,
        "peak_gpu_mb": None,
    }, output


def run_all_benchmarks(output: Path, sizes: tuple[int, ...], repeats: int) -> None:
    """Run equivalent inputs and write JSON plus one three-table Markdown report."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for count in sizes:
        source, oracle = fixture(count)
        constraint = InteractionConstraintSet(
            "move_held_object",
            (PhaseAnnotation("transport", 0, count, "free", "target"),),
            count - 1,
            oracle[-1],
            "analytic-line-v1",
        )
        refiner = KinematicMotionRefiner(MotionRefinementCfg(smoothing_weight=8))
        alignment = oracle[-1] @ np.linalg.inv(source[-1])
        operations = {
            "replay": lambda: source.copy(),
            "rigid_anchor": lambda: alignment @ source,
            "sparse_refinement": lambda: np.array(
                refiner.refine(source, constraint).fit.poses
            ),
        }
        for name, operation in operations.items():
            performance, result = measure(operation, repeats)
            row = {
                "sample_size": count,
                "algorithm": name,
                **performance,
                **metrics(result, oracle),
            }
            rows.append(row)
            print(
                f"n={count:>7d} {name}: {row['cost_time_ms']:.3f} ms, RMSE={row['position_rmse_mm']:.4f} mm",
                flush=True,
            )
    metadata = {
        "schema": "kinematic-refinement-benchmark/v1",
        "origin": "synthetic_reference",
        "device": "cpu",
        "dtype": "float64",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "osqp": osqp.__version__,
        "cpu_logical_count": psutil.cpu_count(),
        "cpu_threads": {
            key: os.getenv(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "warmup": 2,
        "repeats": repeats,
        "acceptance_thresholds": {
            "position_rmse_m": 0.0008,
            "anchor_error_m": 1e-6,
            "rotation_error_rad": 1e-5,
        },
        "memory_scope": "separate warmed RSS retained delta; transient CPU peak unmeasured; GPU inapplicable",
        "timing_scope": "completed synchronous CPU calls, including preparation and verification; input generation excluded",
        "rows": rows,
    }
    (output / "report.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    lines = [
        "# Kinematic refinement benchmark",
        "",
        metadata["timing_scope"],
        "",
        metadata["memory_scope"],
        "",
        "## Time & Memory",
        "",
        "| sample_size | algorithm | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_size']} | {row['algorithm']} | {row['cost_time_ms']:.4f} | {row['cpu_delta_mb']:.4f} | N/A | N/A |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| sample_size | algorithm | success_rate | position_rmse_mm | anchor_error_mm | rotation_error_deg |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_size']} | {row['algorithm']} | {row['success_rate']:.3f} | {row['position_rmse_mm']:.6f} | {row['anchor_error_mm']:.6f} | {row['rotation_error_deg']:.6f} |"
        )
    leaderboard = [
        (
            name,
            statistics.mean(
                row["success_rate"] for row in rows if row["algorithm"] == name
            ),
        )
        for name in operations
    ]
    leaderboard.sort(key=lambda item: -item[1])
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate |",
        "|---|---|---|",
    ]
    lines += [
        f"| {rank} | {name} | {rate:.3f} |"
        for rank, (name, rate) in enumerate(leaderboard, 1)
    ]
    report = output / "report.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"Markdown report saved: {report}")
    if any(
        row["success_rate"] != 1
        for row in rows
        if row["algorithm"] == "sparse_refinement"
    ):
        raise SystemExit(
            "sparse refinement failed the frozen analytic acceptance criteria"
        )


def main() -> None:
    """Parse finite CPU workload sizes and repetition counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100 or any(not 16 <= n <= 10000 for n in args.sizes):
        raise ValueError("benchmark budgets are invalid")
    run_all_benchmarks(args.output, tuple(args.sizes), args.repeats)


if __name__ == "__main__":
    main()
