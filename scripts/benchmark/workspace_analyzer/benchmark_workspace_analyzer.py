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

from __future__ import annotations

"""Benchmark script for workspace analyzer performance optimizations.

Measures each optimization independently across multiple sample sizes.
Run: python -m scripts.benchmark.workspace_analyzer.benchmark_workspace_analyzer
"""

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch

SAMPLE_SIZES_SMALL = [100, 1000, 10000, 50000]
SAMPLE_SIZES_MEDIUM = [1000, 10000, 100000, 500000]


def _sync_cuda() -> None:
    """Synchronize CUDA stream when available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_gpu_memory() -> None:
    """Reset PyTorch peak GPU memory stats when CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated by PyTorch in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def _memory_snapshot() -> dict[str, float]:
    """Return current process memory usage snapshot in MB."""
    process = psutil.Process(os.getpid())
    cpu_mb = process.memory_info().rss / 1024**2
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}


def _time_call(callable_fn) -> tuple[float, dict[str, float], float, object]:
    """Time a callable and return elapsed seconds, memory deltas, and result."""
    _reset_peak_gpu_memory()
    before = _memory_snapshot()
    _sync_cuda()

    start = time.perf_counter()
    result = callable_fn()
    _sync_cuda()
    elapsed = time.perf_counter() - start

    after = _memory_snapshot()
    deltas = {
        "cpu_mb": after["cpu_mb"] - before["cpu_mb"],
        "gpu_mb": after["gpu_mb"] - before["gpu_mb"],
    }
    return elapsed, deltas, _peak_gpu_memory_mb(), result


def _format_perf_line(
    n: int,
    elapsed_s: float,
    memory_delta: dict[str, float],
    peak_gpu_mb: float,
    extra_info: str,
) -> str:
    """Format one benchmark output line with aligned fields."""
    return (
        f"  n={n:>7d}: {elapsed_s * 1000:>10.2f} ms | "
        f"CPU Δ={memory_delta['cpu_mb']:+.1f} MB  "
        f"GPU Δ={memory_delta['gpu_mb']:+.1f} MB  "
        f"peak GPU={peak_gpu_mb:.1f} MB" + (f" | {extra_info}" if extra_info else "")
    )


def _format_markdown_table(rows: list[dict[str, object]]) -> list[str]:
    """Format rows into a markdown table."""
    if not rows:
        return ["No data."]

    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return lines


def _write_markdown_report(
    benchmark_name: str,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    notes: list[str] | None = None,
) -> Path:
    """Write benchmark results to a markdown report with two tables."""
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{benchmark_name}_{timestamp}.md"

    lines: list[str] = [
        f"# {benchmark_name} Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Time & Memory",
        "",
    ]
    lines.extend(_format_markdown_table(perf_rows))
    lines.extend(["", "## Success & Other Metrics", ""])
    lines.extend(_format_markdown_table(metric_rows))

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def benchmark_halton_sampler() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Benchmark Halton sampler: vectorized vs loop-based."""
    from embodichain.lab.sim.utility.workspace_analyzer.samplers.halton_sampler import (
        HaltonSampler,
    )

    sampler = HaltonSampler(seed=42)
    bounds = torch.tensor(
        [
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
        ],
        dtype=torch.float32,
    )

    print("\n=== Halton Sampler Benchmark ===")
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for n in [100, 1000, 10000, 100000]:
        elapsed, mem_delta, peak_gpu, samples = _time_call(
            lambda: sampler.sample(num_samples=n, bounds=bounds)
        )
        elapsed_ms = elapsed * 1000.0
        print(
            _format_perf_line(
                n=n,
                elapsed_s=elapsed,
                memory_delta=mem_delta,
                peak_gpu_mb=peak_gpu,
                extra_info=f"shape={tuple(samples.shape)}",
            )
        )

        perf_rows.append(
            {
                "sample_size": n,
                "impl": "workspace_analyzer",
                "component": "halton_sampler",
                "cost_time_ms": f"{elapsed_ms:.6f}",
                "cpu_delta_mb": f"{mem_delta['cpu_mb']:.6f}",
                "gpu_delta_mb": f"{mem_delta['gpu_mb']:.6f}",
                "peak_gpu_mb": f"{peak_gpu:.6f}",
            }
        )
        metric_rows.append(
            {
                "sample_size": n,
                "impl": "workspace_analyzer",
                "component": "halton_sampler",
                "success_rate": "N/A",
                "other_metrics": f"shape={tuple(samples.shape)}",
            }
        )

    return perf_rows, metric_rows


def benchmark_density_metric() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Benchmark density metric: KDTree vs brute-force."""
    from embodichain.lab.sim.utility.workspace_analyzer.metrics.density_metric import (
        DensityMetric,
    )
    from embodichain.lab.sim.utility.workspace_analyzer.configs.metric_config import (
        DensityConfig,
    )

    config = DensityConfig(radius=0.05, compute_distribution=False)
    metric = DensityMetric(config)

    print("\n=== Density Metric Benchmark ===")
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for n in SAMPLE_SIZES_SMALL:
        points = np.random.randn(n, 3).astype(np.float32) * 0.5

        elapsed, mem_delta, peak_gpu, result = _time_call(
            lambda: metric.compute(points)
        )
        elapsed_ms = elapsed * 1000.0
        print(
            _format_perf_line(
                n=n,
                elapsed_s=elapsed,
                memory_delta=mem_delta,
                peak_gpu_mb=peak_gpu,
                extra_info=f"mean_density={result['mean_density']:.2f}",
            )
        )

        perf_rows.append(
            {
                "sample_size": n,
                "impl": "workspace_analyzer",
                "component": "density_metric",
                "cost_time_ms": f"{elapsed_ms:.6f}",
                "cpu_delta_mb": f"{mem_delta['cpu_mb']:.6f}",
                "gpu_delta_mb": f"{mem_delta['gpu_mb']:.6f}",
                "peak_gpu_mb": f"{peak_gpu:.6f}",
            }
        )
        metric_rows.append(
            {
                "sample_size": n,
                "impl": "workspace_analyzer",
                "component": "density_metric",
                "success_rate": "N/A",
                "other_metrics": f"mean_density={result['mean_density']:.6f}",
            }
        )

    return perf_rows, metric_rows


def benchmark_voxelization() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Benchmark voxelization: np.unique vs dict-based."""
    from embodichain.lab.sim.utility.workspace_analyzer.metrics.reachability_metric import (
        ReachabilityMetric,
    )
    from embodichain.lab.sim.utility.workspace_analyzer.configs.metric_config import (
        ReachabilityConfig,
    )

    config = ReachabilityConfig(voxel_size=0.01, compute_coverage=True)
    metric = ReachabilityMetric(config)

    print("\n=== Voxelization Benchmark ===")
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for n in SAMPLE_SIZES_MEDIUM:
        points = np.random.randn(n, 3).astype(np.float32) * 0.5

        elapsed, mem_delta, peak_gpu, result = _time_call(
            lambda: metric.compute(points)
        )
        elapsed_ms = elapsed * 1000.0
        print(
            _format_perf_line(
                n=n,
                elapsed_s=elapsed,
                memory_delta=mem_delta,
                peak_gpu_mb=peak_gpu,
                extra_info=(
                    f"volume={result['volume']:.4f}, " f"voxels={result['num_voxels']}"
                ),
            )
        )

        perf_rows.append(
            {
                "sample_size": n,
                "impl": "workspace_analyzer",
                "component": "voxelization",
                "cost_time_ms": f"{elapsed_ms:.6f}",
                "cpu_delta_mb": f"{mem_delta['cpu_mb']:.6f}",
                "gpu_delta_mb": f"{mem_delta['gpu_mb']:.6f}",
                "peak_gpu_mb": f"{peak_gpu:.6f}",
            }
        )
        metric_rows.append(
            {
                "sample_size": n,
                "impl": "workspace_analyzer",
                "component": "voxelization",
                "success_rate": "N/A",
                "other_metrics": (
                    f"volume={result['volume']:.6f}, num_voxels={result['num_voxels']}"
                ),
            }
        )

    return perf_rows, metric_rows


def benchmark_manipulability() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Benchmark manipulability: batch vs per-sample."""
    from embodichain.lab.sim.utility.workspace_analyzer.metrics.manipulability_metric import (
        ManipulabilityMetric,
    )
    from embodichain.lab.sim.utility.workspace_analyzer.configs.metric_config import (
        ManipulabilityConfig,
    )

    config = ManipulabilityConfig(compute_isotropy=True)
    metric = ManipulabilityMetric(config)

    print("\n=== Manipulability Metric Benchmark ===")
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for n in SAMPLE_SIZES_SMALL:
        points = np.random.randn(n, 3).astype(np.float32) * 0.5
        jacobians = np.random.randn(n, 6, 6).astype(np.float32) * 0.1

        elapsed, mem_delta, peak_gpu, result = _time_call(
            lambda: metric.compute(points, jacobians=jacobians)
        )
        elapsed_ms = elapsed * 1000.0
        print(
            _format_perf_line(
                n=n,
                elapsed_s=elapsed,
                memory_delta=mem_delta,
                peak_gpu_mb=peak_gpu,
                extra_info=f"mean_manip={result['mean_manipulability']:.6f}",
            )
        )

        perf_rows.append(
            {
                "sample_size": n,
                "impl": "workspace_analyzer",
                "component": "manipulability_metric",
                "cost_time_ms": f"{elapsed_ms:.6f}",
                "cpu_delta_mb": f"{mem_delta['cpu_mb']:.6f}",
                "gpu_delta_mb": f"{mem_delta['gpu_mb']:.6f}",
                "peak_gpu_mb": f"{peak_gpu:.6f}",
            }
        )
        metric_rows.append(
            {
                "sample_size": n,
                "impl": "workspace_analyzer",
                "component": "manipulability_metric",
                "success_rate": "N/A",
                "other_metrics": (
                    f"mean_manipulability={result['mean_manipulability']:.6f}"
                ),
            }
        )

    return perf_rows, metric_rows


def benchmark_batch_fk() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Benchmark batch FK vs sequential FK (requires GPU robot setup).

    This benchmark requires a running simulation with a robot.
    It is skipped if no simulation is available.
    """
    print("\n=== Batch FK Benchmark (requires robot/simulation) ===")
    print("  Skipped -- requires live SimulationManager and Robot.")
    print("  To run manually, integrate with your robot setup:")
    print("    analyzer.compute_workspace_points(joint_configs, batch_size=512)")
    return [], [
        {
            "sample_size": "N/A",
            "impl": "workspace_analyzer",
            "component": "batch_fk",
            "success_rate": "N/A",
            "other_metrics": "skipped: requires live SimulationManager and Robot",
        }
    ]


def benchmark_batch_ik() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Benchmark batch IK vs sequential IK (requires GPU robot setup).

    This benchmark requires a running simulation with a robot.
    It is skipped if no simulation is available.
    """
    print("\n=== Batch IK Benchmark (requires robot/simulation) ===")
    print("  Skipped -- requires live SimulationManager and Robot.")
    print("  To run manually, integrate with your robot setup:")
    print("    analyzer.compute_reachability(cartesian_points, batch_size=512)")
    return [], [
        {
            "sample_size": "N/A",
            "impl": "workspace_analyzer",
            "component": "batch_ik",
            "success_rate": "N/A",
            "other_metrics": "skipped: requires live SimulationManager and Robot",
        }
    ]


def run_all_benchmarks() -> None:
    """Run all benchmarks and print summary."""
    print("=" * 60)
    print("Workspace Analyzer Performance Benchmarks")
    print("=" * 60)

    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    perf_part, metric_part = benchmark_halton_sampler()
    perf_rows.extend(perf_part)
    metric_rows.extend(metric_part)

    perf_part, metric_part = benchmark_density_metric()
    perf_rows.extend(perf_part)
    metric_rows.extend(metric_part)

    perf_part, metric_part = benchmark_voxelization()
    perf_rows.extend(perf_part)
    metric_rows.extend(metric_part)

    perf_part, metric_part = benchmark_manipulability()
    perf_rows.extend(perf_part)
    metric_rows.extend(metric_part)

    perf_part, metric_part = benchmark_batch_fk()
    perf_rows.extend(perf_part)
    metric_rows.extend(metric_part)

    perf_part, metric_part = benchmark_batch_ik()
    perf_rows.extend(perf_part)
    metric_rows.extend(metric_part)

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)

    report_path = _write_markdown_report(
        benchmark_name="workspace_analyzer",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        notes=[
            "CPU/GPU memory fields are deltas measured around timed calls.",
            "This report contains exactly two tables: Time & Memory, and Success & Other Metrics.",
        ],
    )
    print(f"Markdown report saved: {report_path}")


if __name__ == "__main__":
    run_all_benchmarks()
