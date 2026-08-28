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

"""Benchmark SRS analytical IK across representative workloads.

The benchmark covers randomized nominal, wide-range, joint-boundary,
near-singular, and unreachable targets; perturbs IK seeds independently from
the FK ground truth; and reports repeated latency, throughput, classification
accuracy, FK reconstruction error, and solution distance from the seed.
Run: python -m scripts.benchmark.robotics.kinematic_solver.srs_solver
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.robots.dexforce_w1.params import W1ArmKineParams
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmSide,
    DexforceW1Version,
)
from embodichain.lab.sim.solvers.srs_solver import SRSSolverCfg
from embodichain.utils.logger import set_log_level

DEFAULT_SIZES = (1, 16, 128)
SCENARIOS = ("nominal", "wide", "boundary", "near-singular", "unreachable")


@dataclass
class BenchmarkCase:
    """Inputs and expected reachability for one benchmark case."""

    target: torch.Tensor
    seed: torch.Tensor
    expected_reachable: bool


def _parse_args() -> argparse.Namespace:
    """Parse benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20260824)
    parser.add_argument("--arms", choices=("left", "right", "both"), default="both")
    parser.add_argument("--devices", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--modes", choices=("seeded", "full", "both"), default="both")
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS)
    )
    return parser.parse_args()


def _selected(value: str, first: str, second: str) -> tuple[str, ...]:
    """Expand a two-choice CLI selector."""
    return (first, second) if value == "both" else (value,)


def _make_solver(device: torch.device, search_mode: str, arm: str, size: int):
    """Construct one W1 arm SRS solver."""
    side = DexforceW1ArmSide.LEFT if arm == "left" else DexforceW1ArmSide.RIGHT
    prefix = "LEFT" if arm == "left" else "RIGHT"
    params = W1ArmKineParams(arm_side=side, version=DexforceW1Version.V021)
    cfg = SRSSolverCfg(
        urdf_path=get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf"),
        joint_names=[f"{prefix}_J{i + 1}" for i in range(7)],
        root_link_name=f"{arm}_arm_base",
        end_link_name=f"{arm}_ee",
        dh_params=params.dh_params,
        user_qpos_limits=params.qpos_limits,
        T_b_ob=params.T_b_ob,
        T_e_oe=params.T_e_oe,
        link_lengths=params.link_lengths,
        rotation_directions=params.rotation_directions,
        search_mode=search_mode,
    )
    return cfg.init_solver(num_envs=size, device=device)


def _memory_snapshot() -> tuple[float, float]:
    """Return process RSS and PyTorch CUDA allocation in MiB."""
    cpu_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return cpu_mb, gpu_mb


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA work before observing wall-clock time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _joint_limits(solver) -> tuple[torch.Tensor, torch.Tensor]:
    """Return solver limits on its execution device."""
    limits = solver.get_qpos_limits()
    lower = torch.tensor(
        limits["lower_qpos_limits"], dtype=torch.float32, device=solver.device
    )
    upper = torch.tensor(
        limits["upper_qpos_limits"], dtype=torch.float32, device=solver.device
    )
    return lower, upper


def _uniform_qpos(
    lower: torch.Tensor,
    upper: torch.Tensor,
    size: int,
    low_fraction: float,
    high_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample joint configurations from a fractional limit interval."""
    unit = torch.rand(
        (size, 7), generator=generator, dtype=torch.float32, device="cpu"
    ).to(lower.device)
    low = lower + low_fraction * (upper - lower)
    high = lower + high_fraction * (upper - lower)
    return low + unit * (high - low)


def _make_case(
    solver,
    scenario: str,
    size: int,
    generator: torch.Generator,
) -> BenchmarkCase:
    """Generate independent ground-truth joints, target poses, and IK seeds."""
    lower, upper = _joint_limits(solver)
    span = upper - lower
    if scenario == "nominal":
        truth = _uniform_qpos(lower, upper, size, 0.25, 0.75, generator)
    elif scenario in ("wide", "unreachable"):
        truth = _uniform_qpos(lower, upper, size, 0.05, 0.95, generator)
    elif scenario == "boundary":
        choose_upper = (
            torch.randint(0, 2, (size, 7), generator=generator, device="cpu")
            .bool()
            .to(lower.device)
        )
        near_lower = lower + 0.01 * span
        near_upper = upper - 0.01 * span
        truth = torch.where(choose_upper, near_upper, near_lower)
    elif scenario == "near-singular":
        truth = _uniform_qpos(lower, upper, size, 0.25, 0.75, generator)
        # SRS elbow and wrist pitch approach their singular values without
        # using the exact unreachable straight-arm boundary.
        truth[:, 3] = torch.clamp(
            torch.full_like(truth[:, 3], -1e-3), lower[3], upper[3]
        )
        truth[:, 5] = torch.clamp(
            torch.full_like(truth[:, 5], 1e-3), lower[5], upper[5]
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    target = solver.get_fk(truth)
    noise = torch.randn(
        (size, 7), generator=generator, dtype=torch.float32, device="cpu"
    ).to(lower.device)
    seed = torch.clamp(truth + noise * (0.12 * span), lower, upper)
    if scenario == "unreachable":
        offsets = torch.tensor(
            [2.0, -2.0, 2.0], dtype=target.dtype, device=target.device
        )
        target = target.clone()
        target[:, :3, 3] += offsets
        return BenchmarkCase(target, seed, False)
    return BenchmarkCase(target, seed, True)


def _solution_matrix(solution: torch.Tensor) -> torch.Tensor:
    """Normalize solver output to one solution per target."""
    return solution[:, 0] if solution.ndim == 3 else solution


def _pose_error_vectors(
    target: torch.Tensor, actual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample translation millimetres and rotation degrees."""
    target = target.to(torch.float64)
    actual = actual.to(torch.float64)
    translation = torch.linalg.norm(target[:, :3, 3] - actual[:, :3, 3], dim=1)
    relative = target[:, :3, :3].transpose(1, 2) @ actual[:, :3, :3]
    # Float32 FK matrices can be microscopically non-orthogonal. Project the
    # relative matrix onto SO(3) before converting its trace to an angle.
    u, _, vh = torch.linalg.svd(relative)
    projected = u @ vh
    negative_determinant = torch.linalg.det(projected) < 0.0
    if negative_determinant.any():
        u = u.clone()
        u[negative_determinant, :, -1] *= -1.0
        projected = u @ vh
    cosine = ((projected.diagonal(dim1=1, dim2=2).sum(1) - 1.0) / 2.0).clamp(-1, 1)
    return translation * 1000.0, torch.rad2deg(torch.acos(cosine))


def _percentile(values: torch.Tensor, quantile: float) -> float:
    """Return a finite percentile or NaN for an empty tensor."""
    return float(torch.quantile(values.float(), quantile)) if values.numel() else np.nan


def _quality_metrics(
    solver,
    case: BenchmarkCase,
    success: torch.Tensor,
    solution: torch.Tensor,
) -> dict[str, float]:
    """Compute correctness metrics without mixing failed solutions into errors."""
    success = success.bool()
    solution = _solution_matrix(solution)
    expected = torch.full_like(success, case.expected_reachable)
    classification_accuracy = float((success == expected).float().mean())
    success_rate = float(success.float().mean())
    if not success.any():
        return {
            "success_rate": success_rate,
            "classification_accuracy": classification_accuracy,
            "translation_mean_mm": np.nan,
            "translation_p95_mm": np.nan,
            "translation_max_mm": np.nan,
            "rotation_mean_deg": np.nan,
            "rotation_p95_deg": np.nan,
            "seed_distance_mean_rad": np.nan,
        }
    valid_solution = solution[success]
    actual = solver.get_fk(valid_solution)
    translation, rotation = _pose_error_vectors(case.target[success], actual)
    delta = valid_solution - case.seed[success]
    wrapped_delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    seed_distance = torch.linalg.vector_norm(wrapped_delta, dim=1)
    return {
        "success_rate": success_rate,
        "classification_accuracy": classification_accuracy,
        "translation_mean_mm": float(translation.mean()),
        "translation_p95_mm": _percentile(translation, 0.95),
        "translation_max_mm": float(translation.max()),
        "rotation_mean_deg": float(rotation.mean()),
        "rotation_p95_deg": _percentile(rotation, 0.95),
        "seed_distance_mean_rad": float(seed_distance.mean()),
    }


def _measure_case(
    solver,
    case: BenchmarkCase,
    repeats: int,
    warmup: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Measure repeated solve latency/memory and return final quality metrics."""
    for _ in range(warmup):
        solver.get_ik(case.target, case.seed)
    _synchronize(solver.device)
    if solver.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(solver.device)
    cpu_before, gpu_before = _memory_snapshot()
    durations = []
    success = solution = None
    for _ in range(repeats):
        _synchronize(solver.device)
        start = time.perf_counter()
        success, solution = solver.get_ik(case.target, case.seed)
        _synchronize(solver.device)
        durations.append((time.perf_counter() - start) * 1000.0)
    cpu_after, gpu_after = _memory_snapshot()
    assert success is not None and solution is not None
    latency = np.asarray(durations)
    median_ms = float(np.median(latency))
    performance = {
        "latency_median_ms": median_ms,
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "latency_min_ms": float(latency.min()),
        "throughput_targets_s": case.target.shape[0] * 1000.0 / median_ms,
        "cpu_delta_mb": cpu_after - cpu_before,
        "gpu_delta_mb": gpu_after - gpu_before,
        "peak_gpu_mb": (
            torch.cuda.max_memory_allocated(solver.device) / 1024**2
            if solver.device.type == "cuda"
            else 0.0
        ),
    }
    return performance, _quality_metrics(solver, case, success, solution)


def _format_table(rows: list[dict[str, object]]) -> list[str]:
    """Format rows as one Markdown table."""
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[key]) for key in headers) + " |" for row in rows
    )
    return lines


def _leaderboard(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Rank every implementation by correctness, then speed."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row["impl"]), []).append(row)
    ranking = []
    for impl, rows in grouped.items():
        accuracy = float(
            np.mean([float(row["classification_accuracy"]) for row in rows])
        )
        reachable = [row for row in rows if row["expected_reachable"]]
        success = float(np.mean([float(row["success_rate"]) for row in reachable]))
        ranking.append((impl, accuracy, success))
    ranking.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return [
        {
            "rank": rank,
            "algorithm": impl,
            "classification_accuracy": f"{accuracy:.2%}",
            "reachable_success_rate": f"{success:.2%}",
        }
        for rank, (impl, accuracy, success) in enumerate(ranking, 1)
    ]


def _write_report(
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> Path:
    """Write a single report containing exactly three Markdown tables."""
    lines = [
        "# SRS Solver Benchmark",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Repeats: {args.repeats}; warm-up calls: {args.warmup}; random seed: {args.random_seed}.",
        "",
        "## Time & Memory",
        "",
        *_format_table(perf_rows),
        "",
        "## Success & Other Metrics",
        "",
        *_format_table(metric_rows),
        "",
        "## Leaderboard",
        "",
        *_format_table(_leaderboard(metric_rows)),
        "",
        "Errors are computed only over successful solutions. For unreachable cases, classification accuracy rewards rejection rather than success.",
    ]
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"srs_solver_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run the configured SRS benchmark matrix and return its report path."""
    args = _parse_args() if args is None else args
    set_log_level("ERROR")
    if args.repeats < 1 or args.warmup < 0 or any(size < 1 for size in args.sizes):
        raise ValueError(
            "sizes/repeats must be positive and warmup must be non-negative"
        )
    arms = _selected(args.arms, "left", "right")
    modes = _selected(args.modes, "seeded", "full")
    requested_devices = _selected(args.devices, "cpu", "cuda")
    devices = []
    for name in requested_devices:
        if name == "cuda" and not torch.cuda.is_available():
            print("Skipping CUDA: torch.cuda.is_available() is False")
            continue
        devices.append(torch.device(name))
    if not devices:
        raise RuntimeError("No requested benchmark device is available")

    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for device in devices:
        for arm in arms:
            for mode in modes:
                for size in args.sizes:
                    solver = _make_solver(device, mode, arm, size)
                    for scenario_index, scenario in enumerate(args.scenarios):
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(
                            args.random_seed + scenario_index + size * 1009
                        )
                        case = _make_case(solver, scenario, size, generator)
                        performance, quality = _measure_case(
                            solver, case, args.repeats, args.warmup
                        )
                        impl = f"{device.type}-{mode}-{arm}"
                        perf_rows.append(
                            {
                                "sample_size": size,
                                "scenario": scenario,
                                "impl": impl,
                                "latency_median_ms": f"{performance['latency_median_ms']:.3f}",
                                "latency_p95_ms": f"{performance['latency_p95_ms']:.3f}",
                                "throughput_targets_s": f"{performance['throughput_targets_s']:.1f}",
                                "cpu_delta_mb": f"{performance['cpu_delta_mb']:+.2f}",
                                "gpu_delta_mb": f"{performance['gpu_delta_mb']:+.2f}",
                                "peak_gpu_mb": f"{performance['peak_gpu_mb']:.2f}",
                            }
                        )
                        metric_rows.append(
                            {
                                "sample_size": size,
                                "scenario": scenario,
                                "impl": impl,
                                "expected_reachable": case.expected_reachable,
                                "success_rate": f"{quality['success_rate']:.4f}",
                                "classification_accuracy": f"{quality['classification_accuracy']:.4f}",
                                "translation_mean_mm": f"{quality['translation_mean_mm']:.6f}",
                                "translation_p95_mm": f"{quality['translation_p95_mm']:.6f}",
                                "translation_max_mm": f"{quality['translation_max_mm']:.6f}",
                                "rotation_mean_deg": f"{quality['rotation_mean_deg']:.6f}",
                                "rotation_p95_deg": f"{quality['rotation_p95_deg']:.6f}",
                                "seed_distance_mean_rad": f"{quality['seed_distance_mean_rad']:.6f}",
                            }
                        )
                        print(
                            f"{impl:>22} n={size:>4} {scenario:<13} "
                            f"median={performance['latency_median_ms']:>9.3f} ms "
                            f"success={quality['success_rate']:>7.2%} "
                            f"class={quality['classification_accuracy']:>7.2%}"
                        )
    report = _write_report(perf_rows, metric_rows, args)
    print(f"Markdown report saved: {report}")
    return report


if __name__ == "__main__":
    run_all_benchmarks()
