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

"""Benchmark script for Pytorch Kinematic solver Warp CUDA vs CPU implementation.

Measures FK/IK wall-clock latency, pose accuracy, success rate, and memory usage.
Run: python -m scripts.benchmark.robotics.kinematic_solver.pytorch_kinematic
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.solvers.pytorch_solver import PytorchSolver, PytorchSolverCfg

LOWER_LIMITS = [-6.2832, -6.2832, -3.1416, -6.2832, -6.2832, -6.2832]
UPPER_LIMITS = [6.2832, 6.2832, 3.1416, 6.2832, 6.2832, 6.2832]
SAMPLE_SIZES = [100, 1000, 10000]


def get_cpu_memory_mb() -> float:
    """Return current process RSS memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def get_gpu_memory_mb() -> float:
    """Return current GPU allocated memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def memory_snapshot() -> dict[str, float]:
    """Return a unified memory snapshot for CPU/GPU."""
    return {"cpu_mb": get_cpu_memory_mb(), "gpu_mb": get_gpu_memory_mb()}


def init_pk_solver(device: torch.device = torch.device("cpu")) -> PytorchSolver:
    """Initialize Pytorch kinematic solver for the given device."""
    qpos_limits = [LOWER_LIMITS, UPPER_LIMITS]
    solver_cfg = PytorchSolverCfg(
        urdf_path=get_data_path("UniversalRobots/UR10/UR10.urdf"),
        end_link_name="ee_link",
        root_link_name="base_link",
        joint_names=["J1", "J2", "J3", "J4", "J5", "J6"],
        user_qpos_limits=qpos_limits,
    )
    solver = PytorchSolver(solver_cfg, device=device)
    return solver


def sample_qpos(
    n_samples: int,
    has_cuda: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample joint positions for CPU and optional CUDA benchmark."""
    qpos_np = np.random.uniform(
        low=np.array(LOWER_LIMITS) + 1e-1,
        high=np.array(UPPER_LIMITS) - 1e-1,
        size=(n_samples, 6),
    ).astype(float)
    qpos_cpu = torch.tensor(qpos_np, dtype=torch.float64, device=torch.device("cpu"))
    qpos_cuda = qpos_cpu.to(torch.device("cuda")) if has_cuda else None
    return qpos_cpu, qpos_cuda


def get_pose_err(
    matrix_a: np.ndarray | torch.Tensor,
    matrix_b: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return translation and rotation errors between paired poses.

    Supports either a single 4x4 pose or a batch with shape (N, 4, 4).
    """
    tensor_a = torch.as_tensor(matrix_a, dtype=torch.float64)
    tensor_b = torch.as_tensor(matrix_b, dtype=torch.float64, device=tensor_a.device)

    if tensor_a.ndim == 2:
        tensor_a = tensor_a.unsqueeze(0)
    if tensor_b.ndim == 2:
        tensor_b = tensor_b.unsqueeze(0)

    t_err = torch.linalg.norm(tensor_a[:, :3, 3] - tensor_b[:, :3, 3], dim=-1)

    relative_rot = torch.matmul(
        tensor_a[:, :3, :3].transpose(-1, -2),
        tensor_b[:, :3, :3],
    )
    trace = torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, min=-1.0, max=1.0)
    r_err = torch.arccos(cos_angle)
    return t_err, r_err


def run_ik_once(solver: PytorchSolver, qpos: torch.Tensor) -> dict[str, float]:
    """Run one IK benchmark pass and return performance/quality metrics."""
    fk_xpos = solver.get_fk(qpos)

    if qpos.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    before = memory_snapshot()
    start_time = time.perf_counter()
    ik_success, ik_qpos = solver.get_ik(
        fk_xpos, joint_seed=qpos, return_all_solutions=False
    )

    if qpos.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_s = time.perf_counter() - start_time
    after = memory_snapshot()
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if qpos.device.type == "cuda" and torch.cuda.is_available()
        else 0.0
    )

    ik_qpos = ik_qpos[:, 0, :]
    check_xpos = solver.get_fk(ik_qpos)
    t_err, r_err = get_pose_err(fk_xpos, check_xpos)
    success_rate = ik_success.float().mean().item()

    return {
        "cost_time_ms": elapsed_s * 1000.0,
        "cpu_delta_mb": after["cpu_mb"] - before["cpu_mb"],
        "gpu_delta_mb": after["gpu_mb"] - before["gpu_mb"],
        "peak_gpu_mb": peak_gpu_mb,
        "success_rate": success_rate,
        "translation_err_mm": t_err.mean().item() * 1000.0,
        "rotation_err_deg": r_err.mean().item() * 180.0 / np.pi,
    }


def build_leaderboard_rows(
    metric_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Aggregate and rank algorithms by overall success rate."""
    aggregate: dict[str, dict[str, float]] = {}
    for row in metric_rows:
        impl = str(row["impl"])
        if impl not in aggregate:
            aggregate[impl] = {
                "success_sum": 0.0,
                "t_err_sum": 0.0,
                "r_err_sum": 0.0,
                "count": 0.0,
            }
        aggregate[impl]["success_sum"] += float(row["success_rate"])
        aggregate[impl]["t_err_sum"] += float(row["translation_err_mm"])
        aggregate[impl]["r_err_sum"] += float(row["rotation_err_deg"])
        aggregate[impl]["count"] += 1.0

    ranked = sorted(
        aggregate.items(),
        key=lambda item: item[1]["success_sum"] / max(item[1]["count"], 1.0),
        reverse=True,
    )

    leaderboard_rows: list[dict[str, object]] = []
    for rank, (algorithm, stats) in enumerate(ranked, start=1):
        count = max(stats["count"], 1.0)
        leaderboard_rows.append(
            {
                "rank": rank,
                "algorithm": algorithm,
                "overall_success_rate": f"{stats['success_sum'] / count:.2%}",
                "avg_translation_err_mm": f"{stats['t_err_sum'] / count:.6f}",
                "avg_rotation_err_deg": f"{stats['r_err_sum'] / count:.6f}",
            }
        )
    return leaderboard_rows


def write_markdown_report(
    benchmark_name: str,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    notes: list[str] | None = None,
) -> Path:
    """Write benchmark results into one markdown report."""
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{benchmark_name}_{ts}.md"

    lines: list[str] = [
        f"# {benchmark_name} Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Time & Memory",
        "",
    ]

    if perf_rows:
        headers = list(perf_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in perf_rows:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    else:
        lines.append("No time/memory rows were produced.")

    lines.extend(["", "## Success & Other Metrics", ""])
    if metric_rows:
        headers = list(metric_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in metric_rows:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    else:
        lines.append("No success/metric rows were produced.")

    lines.extend(["", "## Leaderboard", ""])
    if leaderboard_rows:
        headers = list(leaderboard_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in leaderboard_rows:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    else:
        lines.append("No leaderboard rows were produced.")

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_all_benchmarks() -> None:
    """Run CPU/CUDA IK benchmarks and persist a markdown report."""
    print("=" * 60)
    print("Pytorch Kinematic Solver Benchmarks")
    print("=" * 60)

    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    cpu_solver = init_pk_solver(device=torch.device("cpu"))
    has_cuda = torch.cuda.is_available()
    cuda_solver = init_pk_solver(device=torch.device("cuda")) if has_cuda else None

    print("\n=== Pytorch IK Benchmark ===")
    if not has_cuda:
        print("  CUDA unavailable; CUDA benchmark is skipped.")

    for n_sample in SAMPLE_SIZES:
        qpos_cpu, qpos_cuda = sample_qpos(n_sample, has_cuda)
        print(f"**** Test over {n_sample} samples:")

        cpu_result = run_ik_once(cpu_solver, qpos_cpu)
        perf_rows.append(
            {
                "sample_size": n_sample,
                "impl": "CPU",
                "cost_time_ms": f"{cpu_result['cost_time_ms']:.6f}",
                "cpu_delta_mb": f"{cpu_result['cpu_delta_mb']:+.1f}",
                "gpu_delta_mb": f"{cpu_result['gpu_delta_mb']:+.1f}",
                "peak_gpu_mb": f"{cpu_result['peak_gpu_mb']:.1f}",
            }
        )
        metric_rows.append(
            {
                "sample_size": n_sample,
                "impl": "CPU",
                "success_rate": f"{cpu_result['success_rate']:.2%}",
                "translation_err_mm": f"{cpu_result['translation_err_mm']:.6f}",
                "rotation_err_deg": f"{cpu_result['rotation_err_deg']:.6f}",
            }
        )

        print(f"===CPU time: {cpu_result['cost_time_ms']:.6f} ms")
        print(
            f"   Success rate: {cpu_result['success_rate']:.2%}, "
            f"translation mean error: {cpu_result['translation_err_mm']:.6f} mm, "
            f"rotation mean error: {cpu_result['rotation_err_deg']:.6f} degrees"
        )
        print(
            f"   CPU Δ={cpu_result['cpu_delta_mb']:+.1f} MB  "
            f"GPU Δ={cpu_result['gpu_delta_mb']:+.1f} MB  "
            f"peak GPU={cpu_result['peak_gpu_mb']:.1f} MB"
        )

        if has_cuda and cuda_solver is not None and qpos_cuda is not None:
            cuda_result = run_ik_once(cuda_solver, qpos_cuda)
            perf_rows.append(
                {
                    "sample_size": n_sample,
                    "impl": "CUDA",
                    "cost_time_ms": f"{cuda_result['cost_time_ms']:.6f}",
                    "cpu_delta_mb": f"{cuda_result['cpu_delta_mb']:+.1f}",
                    "gpu_delta_mb": f"{cuda_result['gpu_delta_mb']:+.1f}",
                    "peak_gpu_mb": f"{cuda_result['peak_gpu_mb']:.1f}",
                }
            )
            metric_rows.append(
                {
                    "sample_size": n_sample,
                    "impl": "CUDA",
                    "success_rate": f"{cuda_result['success_rate']:.2%}",
                    "translation_err_mm": f"{cuda_result['translation_err_mm']:.6f}",
                    "rotation_err_deg": f"{cuda_result['rotation_err_deg']:.6f}",
                }
            )

            print(f"===CUDA time: {cuda_result['cost_time_ms']:.6f} ms")
            print(
                f"   Success rate: {cuda_result['success_rate']:.2%}, "
                f"translation mean error: {cuda_result['translation_err_mm']:.6f} mm, "
                f"rotation mean error: {cuda_result['rotation_err_deg']:.6f} degrees"
            )
            print(
                f"   CPU Δ={cuda_result['cpu_delta_mb']:+.1f} MB  "
                f"GPU Δ={cuda_result['gpu_delta_mb']:+.1f} MB  "
                f"peak GPU={cuda_result['peak_gpu_mb']:.1f} MB"
            )

    leaderboard_rows = build_leaderboard_rows(metric_rows)

    report_path = write_markdown_report(
        benchmark_name="pytorch_kinematic",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            "CPU/GPU memory fields are deltas measured around timed IK calls.",
            "Pose quality is validated by FK(IK(q)) consistency check.",
        ],
    )

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)
    print(f"Markdown report saved: {report_path}")


if __name__ == "__main__":
    run_all_benchmarks()
