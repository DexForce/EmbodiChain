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

"""Benchmark script for workspace analyzer performance optimizations.

Measures each optimization independently across multiple sample sizes.
Run: python -m scripts.benchmark.workspace_analyzer.benchmark_workspace_analyzer
"""

import time
import numpy as np
import torch


def benchmark_halton_sampler():
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
    for n in [100, 1000, 10000, 100000]:
        start = time.perf_counter()
        samples = sampler.sample(num_samples=n, bounds=bounds)
        elapsed = time.perf_counter() - start
        print(f"  n={n:>7d}: {elapsed*1000:>10.2f} ms ({samples.shape})")


def benchmark_density_metric():
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
    for n in [100, 1000, 10000, 50000]:
        points = np.random.randn(n, 3).astype(np.float32) * 0.5

        start = time.perf_counter()
        result = metric.compute(points)
        elapsed = time.perf_counter() - start
        print(
            f"  n={n:>7d}: {elapsed*1000:>10.2f} ms "
            f"(mean_density={result['mean_density']:.2f})"
        )


def benchmark_voxelization():
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
    for n in [1000, 10000, 100000, 500000]:
        points = np.random.randn(n, 3).astype(np.float32) * 0.5

        start = time.perf_counter()
        result = metric.compute(points)
        elapsed = time.perf_counter() - start
        print(
            f"  n={n:>7d}: {elapsed*1000:>10.2f} ms "
            f"(volume={result['volume']:.4f}, voxels={result['num_voxels']})"
        )


def benchmark_manipulability():
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
    for n in [100, 1000, 10000, 50000]:
        points = np.random.randn(n, 3).astype(np.float32) * 0.5
        jacobians = np.random.randn(n, 6, 6).astype(np.float32) * 0.1

        start = time.perf_counter()
        result = metric.compute(points, jacobians=jacobians)
        elapsed = time.perf_counter() - start
        print(
            f"  n={n:>7d}: {elapsed*1000:>10.2f} ms "
            f"(mean_manip={result['mean_manipulability']:.6f})"
        )


def benchmark_batch_fk():
    """Benchmark batch FK vs sequential FK (requires GPU robot setup).

    This benchmark requires a running simulation with a robot.
    It is skipped if no simulation is available.
    """
    print("\n=== Batch FK Benchmark (requires robot/simulation) ===")
    print("  Skipped -- requires live SimulationManager and Robot.")
    print("  To run manually, integrate with your robot setup:")
    print("    analyzer.compute_workspace_points(joint_configs, batch_size=512)")


def benchmark_batch_ik():
    """Benchmark batch IK vs sequential IK (requires GPU robot setup).

    This benchmark requires a running simulation with a robot.
    It is skipped if no simulation is available.
    """
    print("\n=== Batch IK Benchmark (requires robot/simulation) ===")
    print("  Skipped -- requires live SimulationManager and Robot.")
    print("  To run manually, integrate with your robot setup:")
    print("    analyzer.compute_reachability(cartesian_points, batch_size=512)")


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("=" * 60)
    print("Workspace Analyzer Performance Benchmarks")
    print("=" * 60)

    benchmark_halton_sampler()
    benchmark_density_metric()
    benchmark_voxelization()
    benchmark_manipulability()
    benchmark_batch_fk()
    benchmark_batch_ik()

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()
