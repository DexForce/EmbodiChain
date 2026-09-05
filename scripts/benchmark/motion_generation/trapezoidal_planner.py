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

"""Benchmark batched trapezoidal-planner Torch and Warp backends.

Run: python scripts/benchmark/motion_generation/trapezoidal_planner.py
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from embodichain.lab.sim.planners.trapezoidal_planner import (
    TrapezoidalPlanOptions,
    _plan_linear_profiles,
)


def parse_args() -> argparse.Namespace:
    """Parse benchmark sizes and device selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Torch device to benchmark.")
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64])
    return parser.parse_args()


def memory_snapshot() -> dict[str, float]:
    """Return current process RSS and Torch CUDA allocation in MiB."""
    cpu_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA timing when the selected device is asynchronous."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_waypoints(
    batch_size: int,
    segment_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Create deterministic non-collinear seven-DOF benchmark paths."""
    parameter = torch.linspace(
        0.0, 4.0, segment_count + 1, dtype=torch.float32, device=device
    )
    base = torch.stack(
        [
            parameter,
            torch.sin(parameter),
            torch.cos(parameter),
            0.5 * torch.sin(2.0 * parameter),
            0.4 * torch.cos(1.5 * parameter),
            0.2 * parameter,
            -0.1 * parameter,
        ],
        dim=-1,
    )
    offsets = torch.arange(batch_size, dtype=torch.float32, device=device)
    offsets = offsets[:, None, None] * 1e-3
    return base[None].expand(batch_size, -1, -1) + offsets


def benchmark_case(
    *,
    waypoints: torch.Tensor,
    profile: str,
    backend: str,
    sample_count: int,
    repeats: int,
    reference: torch.Tensor | None,
) -> tuple[dict[str, object], dict[str, object], torch.Tensor]:
    """Measure one backend/profile case and return report rows."""
    options = TrapezoidalPlanOptions(
        profile=profile,
        constraints={"velocity": 0.7, "acceleration": 1.4, "jerk": 4.0},
        sample_interval=sample_count,
        backend=backend,
    )
    result = _plan_linear_profiles(waypoints, options)
    synchronize(waypoints.device)
    if waypoints.is_cuda:
        torch.cuda.reset_peak_memory_stats(waypoints.device)
    before = memory_snapshot()
    started = time.perf_counter()
    for _ in range(repeats):
        result = _plan_linear_profiles(waypoints, options)
    synchronize(waypoints.device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
    after = memory_snapshot()
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated(waypoints.device) / 1024**2
        if waypoints.is_cuda
        else 0.0
    )
    max_error = (
        0.0
        if reference is None
        else float((result.positions - reference).abs().max().item())
    )
    success = bool(
        result.is_all_success()
        and torch.isfinite(result.positions).all()
        and torch.allclose(result.positions[:, 0], waypoints[:, 0])
        and torch.allclose(result.positions[:, -1], waypoints[:, -1])
    )
    algorithm = f"{backend}-{profile}"
    perf = {
        "batch_size": waypoints.shape[0],
        "algorithm": algorithm,
        "cost_time_ms": f"{elapsed_ms:.4f}",
        "cpu_delta_mb": f"{after['cpu_mb'] - before['cpu_mb']:+.2f}",
        "gpu_delta_mb": f"{after['gpu_mb'] - before['gpu_mb']:+.2f}",
        "peak_gpu_mb": f"{peak_gpu_mb:.2f}",
    }
    metric = {
        "batch_size": waypoints.shape[0],
        "algorithm": algorithm,
        "success_rate": "1.0" if success else "0.0",
        "max_position_error": f"{max_error:.3e}",
        "duration_mean_s": f"{result.duration.mean().item():.4f}",
    }
    return perf, metric, result.positions


def write_markdown_report(
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
) -> Path:
    """Write exactly three Markdown result tables."""
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"trapezoidal_planner_{timestamp}.md"
    lines = ["# Trapezoidal Planner Benchmark", ""]
    for title, rows in (
        ("Time & Memory", perf_rows),
        ("Success & Other Metrics", metric_rows),
        ("Leaderboard", leaderboard_rows),
    ):
        lines.extend([f"## {title}", ""])
        headers = list(rows[0])
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_all_benchmarks() -> None:
    """Run requested cases and save one three-table Markdown report."""
    args = parse_args()
    device = torch.device(args.device)
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    print("=" * 60)
    print("Trapezoidal Planner Performance Benchmarks")
    print("=" * 60)
    for batch_size in args.batch_sizes:
        waypoints = make_waypoints(batch_size, args.segments, device)
        for profile in ("trapezoidal", "double_s"):
            torch_perf, torch_metric, reference = benchmark_case(
                waypoints=waypoints,
                profile=profile,
                backend="torch",
                sample_count=args.samples,
                repeats=args.repeats,
                reference=None,
            )
            perf_rows.append(torch_perf)
            metric_rows.append(torch_metric)
            print(
                f"batch={batch_size:>4d} torch-{profile:<11s} "
                f"{torch_perf['cost_time_ms']:>10s} ms"
            )
            try:
                warp_perf, warp_metric, _ = benchmark_case(
                    waypoints=waypoints,
                    profile=profile,
                    backend="warp",
                    sample_count=args.samples,
                    repeats=args.repeats,
                    reference=reference,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                print(f"  warp-{profile} skipped: {error}")
            else:
                perf_rows.append(warp_perf)
                metric_rows.append(warp_metric)
                print(
                    f"batch={batch_size:>4d} warp-{profile:<12s} "
                    f"{warp_perf['cost_time_ms']:>10s} ms"
                )
    algorithms = sorted({str(row["algorithm"]) for row in metric_rows})
    leaderboard_rows = []
    for algorithm in algorithms:
        selected_metrics = [row for row in metric_rows if row["algorithm"] == algorithm]
        selected_perf = [row for row in perf_rows if row["algorithm"] == algorithm]
        success_rate = sum(
            float(row["success_rate"]) for row in selected_metrics
        ) / len(selected_metrics)
        mean_ms = sum(float(row["cost_time_ms"]) for row in selected_perf) / len(
            selected_perf
        )
        leaderboard_rows.append(
            {
                "rank": 0,
                "algorithm": algorithm,
                "overall_success_rate": f"{success_rate:.3f}",
                "mean_cost_time_ms": f"{mean_ms:.4f}",
            }
        )
    leaderboard_rows.sort(
        key=lambda row: (
            -float(row["overall_success_rate"]),
            float(row["mean_cost_time_ms"]),
        )
    )
    for rank, row in enumerate(leaderboard_rows, start=1):
        row["rank"] = rank
    report = write_markdown_report(perf_rows, metric_rows, leaderboard_rows)
    print(f"Markdown report saved: {report}")


if __name__ == "__main__":
    run_all_benchmarks()
