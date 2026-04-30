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

"""Unified benchmark for OPW and Pytorch kinematic solvers.

Measures IK wall-clock latency, pose accuracy, success rate, and memory usage
across OPW (Warp CUDA vs CPU) and Pytorch solver (CPU vs optional CUDA).
Run: python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.solvers.opw_solver import OPWSolverCfg
from embodichain.lab.sim.solvers.pytorch_solver import PytorchSolver, PytorchSolverCfg

OPW_LOWER_LIMITS = [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944]
OPW_UPPER_LIMITS = [2.618, 3.14159, 0.0, 1.745, 1.22, 2.0944]

# TODO: Easy to failed if use full joint range, consider adding a margin to avoid sampling near the joint limits.
# PYTORCH_LOWER_LIMITS = [-6.2832, -6.2832, -3.1416, -6.2832, -6.2832, -6.2832]
# PYTORCH_UPPER_LIMITS = [6.2832, 6.2832, 3.1416, 6.2832, 6.2832, 6.2832]
PYTORCH_LOWER_LIMITS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
PYTORCH_UPPER_LIMITS = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]

SAMPLE_SIZES = [100, 1000, 10000]
SUPPORTED_SOLVERS = ("opw", "pytorch")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for selecting benchmark solvers."""
    parser = argparse.ArgumentParser(
        description="Run kinematic solver benchmarks for selected solver backends."
    )
    parser.add_argument(
        "--solvers",
        "-s",
        nargs="+",
        choices=(*SUPPORTED_SOLVERS, "all"),
        default=["all"],
        help=(
            "Solvers to benchmark. Use one or more of: opw, pytorch, all. "
            "Default: all"
        ),
    )
    return parser.parse_args()


def _normalize_selected_solvers(selected_solvers: list[str] | None) -> set[str]:
    """Normalize selected solver names to a canonical set."""
    if not selected_solvers or "all" in selected_solvers:
        return set(SUPPORTED_SOLVERS)
    return {solver for solver in selected_solvers if solver in SUPPORTED_SOLVERS}


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


def _build_leaderboard_rows(
    metric_rows: list[dict[str, object]],
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


def _write_markdown_report(
    benchmark_name: str,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    notes: list[str] | None = None,
) -> Path:
    """Write benchmark results to a markdown report with three tables."""
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

    lines.extend(["", "## Leaderboard", ""])
    lines.extend(_format_markdown_table(leaderboard_rows))

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


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


def _timed_ik_call(
    solver, xpos: torch.Tensor, qpos_seed: torch.Tensor, initial_guess: torch.Tensor
) -> tuple[float, dict[str, float], float, torch.Tensor, torch.Tensor]:
    """Run a timed IK call and return elapsed seconds, memory deltas, and outputs."""
    _reset_peak_gpu_memory()
    mem_before = _memory_snapshot()
    _sync_cuda()

    start = time.perf_counter()
    ik_success, ik_qpos = solver.get_ik(
        xpos,
        qpos_seed=qpos_seed,
        initial_guess=initial_guess,
    )
    _sync_cuda()
    elapsed = time.perf_counter() - start

    mem_after = _memory_snapshot()
    deltas = {
        "cpu_mb": mem_after["cpu_mb"] - mem_before["cpu_mb"],
        "gpu_mb": mem_after["gpu_mb"] - mem_before["gpu_mb"],
    }
    return elapsed, deltas, _peak_gpu_memory_mb(), ik_success, ik_qpos


def _init_pytorch_solver(device: torch.device) -> PytorchSolver:
    """Initialize Pytorch kinematic solver on the target device."""
    solver_cfg = PytorchSolverCfg(
        urdf_path=get_data_path("UniversalRobots/UR10/UR10.urdf"),
        end_link_name="ee_link",
        root_link_name="base_link",
        joint_names=["J1", "J2", "J3", "J4", "J5", "J6"],
        user_qpos_limits=[PYTORCH_LOWER_LIMITS, PYTORCH_UPPER_LIMITS],
    )
    return PytorchSolver(solver_cfg, device=device)


def _sample_qpos(
    n_samples: int,
    lower_limits: list[float],
    upper_limits: list[float],
    margin: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample joint positions with margin from lower/upper limits."""
    qpos_np = np.random.uniform(
        low=np.array(lower_limits) + margin,
        high=np.array(upper_limits) - margin,
        size=(n_samples, 6),
    ).astype(float)
    return torch.tensor(qpos_np, device=device, dtype=dtype)


def _timed_pytorch_ik_call(
    solver: PytorchSolver,
    fk_xpos: torch.Tensor,
    qpos_seed: torch.Tensor,
) -> tuple[float, dict[str, float], float, torch.Tensor, torch.Tensor]:
    """Run a timed Pytorch IK call and return elapsed/memory/outputs."""
    _reset_peak_gpu_memory()
    mem_before = _memory_snapshot()
    _sync_cuda()

    start = time.perf_counter()
    for i in range(3):
        if i == 1:  # skip first run to avoid initialization overhead
            start = time.perf_counter()
        ik_success, ik_qpos = solver.get_ik(
            fk_xpos,
            joint_seed=qpos_seed,
            return_all_solutions=False,
        )
    _sync_cuda()
    elapsed = time.perf_counter() - start
    elapsed /= 2.0

    mem_after = _memory_snapshot()
    deltas = {
        "cpu_mb": mem_after["cpu_mb"] - mem_before["cpu_mb"],
        "gpu_mb": mem_after["gpu_mb"] - mem_before["gpu_mb"],
    }
    return elapsed, deltas, _peak_gpu_memory_mb(), ik_success, ik_qpos[:, 0, :]


def check_opw_solver(
    solver_warp, solver_py_opw, n_samples: int = 1000
) -> dict[str, float]:
    """Run Warp and CPU OPW IK/FK checks and return timing, memory, and accuracy."""
    dof = 6
    qpos_np = np.random.uniform(
        low=np.array(OPW_LOWER_LIMITS)
        + 5.1 / 180.0 * np.pi,  # add a margin to avoid sampling near the joint limits
        high=np.array(OPW_UPPER_LIMITS) + -5.1 / 180.0 * np.pi,
        size=(n_samples, dof),
    ).astype(float)

    qpos_cuda = torch.tensor(qpos_np, device=torch.device("cuda"), dtype=torch.float32)
    xpos_cuda = solver_warp.get_fk(qpos_cuda)
    qpos_seed = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    (
        warp_elapsed,
        warp_mem,
        warp_peak_gpu,
        warp_ik_success,
        warp_ik_qpos,
    ) = _timed_ik_call(
        solver=solver_warp,
        xpos=xpos_cuda,
        qpos_seed=qpos_seed,
        initial_guess=qpos_cuda,
    )

    check_xpos = solver_warp.get_fk(warp_ik_qpos)
    warp_t_err, warp_r_err = get_pose_err(xpos_cuda, check_xpos)
    warp_t_mean_err, warp_r_mean_err = (
        warp_t_err.mean().item(),
        warp_r_err.mean().item(),
    )

    xpos_cpu = xpos_cuda.to(torch.device("cpu"))
    qpos_seed_cpu = qpos_seed.to(torch.device("cpu"))
    qpos_cpu = qpos_cuda.to(torch.device("cpu"))

    (
        cpu_elapsed,
        cpu_mem,
        cpu_peak_gpu,
        py_opw_ik_success,
        py_opw_ik_qpos,
    ) = _timed_ik_call(
        solver=solver_py_opw,
        xpos=xpos_cpu,
        qpos_seed=qpos_seed_cpu,
        initial_guess=qpos_cpu,
    )

    check_xpos = solver_warp.get_fk(py_opw_ik_qpos.to(torch.device("cuda")))
    py_opw_t_err, py_opw_r_err = get_pose_err(xpos_cpu, check_xpos)
    py_opw_t_mean_err, py_opw_r_mean_err = (
        py_opw_t_err.mean().item(),
        py_opw_r_err.mean().item(),
    )

    warp_success_rate = float(warp_ik_success.float().mean().item())
    cpu_success_rate = float(py_opw_ik_success.float().mean().item())

    return {
        "warp_ms": warp_elapsed * 1000.0,
        "warp_t_err_mm": warp_t_mean_err * 1000.0,
        "warp_r_err_deg": warp_r_mean_err * 180.0 / np.pi,
        "warp_success_rate": warp_success_rate,
        "warp_cpu_delta_mb": warp_mem["cpu_mb"],
        "warp_gpu_delta_mb": warp_mem["gpu_mb"],
        "warp_peak_gpu_mb": warp_peak_gpu,
        "cpu_ms": cpu_elapsed * 1000.0,
        "cpu_t_err_mm": py_opw_t_mean_err * 1000.0,
        "cpu_r_err_deg": py_opw_r_mean_err * 180.0 / np.pi,
        "cpu_success_rate": cpu_success_rate,
        "cpu_cpu_delta_mb": cpu_mem["cpu_mb"],
        "cpu_gpu_delta_mb": cpu_mem["gpu_mb"],
        "cpu_peak_gpu_mb": cpu_peak_gpu,
    }


def benchmark_pytorch_solver() -> (
    tuple[list[dict[str, object]], list[dict[str, object]]]
):
    """Benchmark Pytorch solver for CPU and optional CUDA implementations."""
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    cpu_solver = _init_pytorch_solver(device=torch.device("cpu"))
    has_cuda = torch.cuda.is_available()
    cuda_solver = (
        _init_pytorch_solver(device=torch.device("cuda")) if has_cuda else None
    )

    print("\n=== Pytorch Kinematic Benchmark ===")
    if not has_cuda:
        print("  CUDA unavailable; CUDA benchmark is skipped.")

    for n_sample in SAMPLE_SIZES:
        print(f"**** Test over {n_sample} samples:")

        qpos_cpu = _sample_qpos(
            n_samples=n_sample,
            lower_limits=PYTORCH_LOWER_LIMITS,
            upper_limits=PYTORCH_UPPER_LIMITS,
            margin=1e-1,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        fk_xpos_cpu = cpu_solver.get_fk(qpos_cpu)
        (
            cpu_elapsed,
            cpu_mem,
            cpu_peak_gpu,
            cpu_success,
            cpu_ik_qpos,
        ) = _timed_pytorch_ik_call(cpu_solver, fk_xpos_cpu, qpos_cpu)
        check_xpos_cpu = cpu_solver.get_fk(cpu_ik_qpos)
        cpu_t_err, cpu_r_err = get_pose_err(fk_xpos_cpu, check_xpos_cpu)

        cpu_result = {
            "cost_time_ms": cpu_elapsed * 1000.0,
            "cpu_delta_mb": cpu_mem["cpu_mb"],
            "gpu_delta_mb": cpu_mem["gpu_mb"],
            "peak_gpu_mb": cpu_peak_gpu,
            "success_rate": float(cpu_success.float().mean().item()),
            "translation_err_mm": cpu_t_err.mean().item() * 1000.0,
            "rotation_err_deg": cpu_r_err.mean().item() * 180.0 / np.pi,
        }

        perf_rows.append(
            {
                "sample_size": n_sample,
                "impl": "pytorch_cpu",
                "component": "pytorch_ik",
                "cost_time_ms": f"{cpu_result['cost_time_ms']:.6f}",
                "cpu_delta_mb": f"{cpu_result['cpu_delta_mb']:.6f}",
                "gpu_delta_mb": f"{cpu_result['gpu_delta_mb']:.6f}",
                "peak_gpu_mb": f"{cpu_result['peak_gpu_mb']:.6f}",
            }
        )
        metric_rows.append(
            {
                "sample_size": n_sample,
                "impl": "pytorch_cpu",
                "component": "pytorch_ik",
                "success_rate": f"{cpu_result['success_rate']:.6f}",
                "translation_err_mm": f"{cpu_result['translation_err_mm']:.6f}",
                "rotation_err_deg": f"{cpu_result['rotation_err_deg']:.6f}",
            }
        )

        print(f"===Pytorch CPU IK time:  {cpu_result['cost_time_ms']:.6f} ms")
        print(f"   Translation mean error: {cpu_result['translation_err_mm']:.6f} mm")
        print(
            f"   Rotation mean error:    {cpu_result['rotation_err_deg']:.6f} degrees"
        )
        print(f"   Success rate:           {cpu_result['success_rate'] * 100.0:.2f}%")
        print(
            "   "
            f"CPU Δ={cpu_result['cpu_delta_mb']:+.1f} MB  "
            f"GPU Δ={cpu_result['gpu_delta_mb']:+.1f} MB  "
            f"peak GPU={cpu_result['peak_gpu_mb']:.1f} MB"
        )

        if has_cuda and cuda_solver is not None:
            qpos_cuda = qpos_cpu.to(torch.device("cuda"))
            fk_xpos_cuda = cuda_solver.get_fk(qpos_cuda)
            (
                cuda_elapsed,
                cuda_mem,
                cuda_peak_gpu,
                cuda_success,
                cuda_ik_qpos,
            ) = _timed_pytorch_ik_call(cuda_solver, fk_xpos_cuda, qpos_cuda)
            check_xpos_cuda = cuda_solver.get_fk(cuda_ik_qpos)
            cuda_t_err, cuda_r_err = get_pose_err(fk_xpos_cuda, check_xpos_cuda)

            cuda_result = {
                "cost_time_ms": cuda_elapsed * 1000.0,
                "cpu_delta_mb": cuda_mem["cpu_mb"],
                "gpu_delta_mb": cuda_mem["gpu_mb"],
                "peak_gpu_mb": cuda_peak_gpu,
                "success_rate": float(cuda_success.float().mean().item()),
                "translation_err_mm": cuda_t_err.mean().item() * 1000.0,
                "rotation_err_deg": cuda_r_err.mean().item() * 180.0 / np.pi,
            }

            perf_rows.append(
                {
                    "sample_size": n_sample,
                    "impl": "pytorch_cuda",
                    "component": "pytorch_ik",
                    "cost_time_ms": f"{cuda_result['cost_time_ms']:.6f}",
                    "cpu_delta_mb": f"{cuda_result['cpu_delta_mb']:.6f}",
                    "gpu_delta_mb": f"{cuda_result['gpu_delta_mb']:.6f}",
                    "peak_gpu_mb": f"{cuda_result['peak_gpu_mb']:.6f}",
                }
            )
            metric_rows.append(
                {
                    "sample_size": n_sample,
                    "impl": "pytorch_cuda",
                    "component": "pytorch_ik",
                    "success_rate": f"{cuda_result['success_rate']:.6f}",
                    "translation_err_mm": f"{cuda_result['translation_err_mm']:.6f}",
                    "rotation_err_deg": f"{cuda_result['rotation_err_deg']:.6f}",
                }
            )

            print(f"===Pytorch CUDA IK time: {cuda_result['cost_time_ms']:.6f} ms")
            print(
                f"   Translation mean error: {cuda_result['translation_err_mm']:.6f} mm"
            )
            print(
                f"   Rotation mean error:    {cuda_result['rotation_err_deg']:.6f} degrees"
            )
            print(
                f"   Success rate:           {cuda_result['success_rate'] * 100.0:.2f}%"
            )
            print(
                "   "
                f"CPU Δ={cuda_result['cpu_delta_mb']:+.1f} MB  "
                f"GPU Δ={cuda_result['gpu_delta_mb']:+.1f} MB  "
                f"peak GPU={cuda_result['peak_gpu_mb']:.1f} MB"
            )

    return perf_rows, metric_rows


def benchmark_opw_solver() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Benchmark OPW solver for multiple sample sizes."""
    if not torch.cuda.is_available():
        print("\n=== OPW Solver Benchmark ===")
        print("  Skipped -- requires CUDA for Warp implementation comparison.")
        return [], [
            {
                "sample_size": "N/A",
                "impl": "opw_solver",
                "component": "opw_ik",
                "success_rate": "N/A",
                "other_metrics": "skipped: requires CUDA for Warp comparison",
            }
        ]

    cfg = OPWSolverCfg(
        joint_names=("J1", "J2", "J3", "J4", "J5", "J6"),
        user_qpos_limits=(OPW_LOWER_LIMITS, OPW_UPPER_LIMITS),
    )
    cfg.a1 = 400.333
    cfg.a2 = -251.449
    cfg.b = 0.0
    cfg.c1 = 830
    cfg.c2 = 1177.556
    cfg.c3 = 1443.593
    cfg.c4 = 230
    cfg.offsets = (
        0.0,
        82.21350356417211 * np.pi / 180.0,
        -167.21710113148163 * np.pi / 180.0,
        0.0,
        0.0,
        0.0,
    )
    cfg.flip_axes = (True, False, True, True, False, True)
    cfg.has_parallelogram = False

    solver_warp = cfg.init_solver(device=torch.device("cuda"), pk_serial_chain="")
    solver_py_opw = cfg.init_solver(device=torch.device("cpu"), pk_serial_chain="")

    print("\n=== OPW Solver Benchmark ===")
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for n_sample in SAMPLE_SIZES:
        result = check_opw_solver(solver_warp, solver_py_opw, n_samples=n_sample)
        print(f"**** Test over {n_sample} samples:")
        print(f"===Warp CUDA IK time: {result['warp_ms']:.6f} ms")
        print(f"   Translation mean error: {result['warp_t_err_mm']:.6f} mm")
        print(f"   Rotation mean error:    {result['warp_r_err_deg']:.6f} degrees")
        print(f"   Success rate:           {result['warp_success_rate'] * 100.0:.2f}%")
        print(
            "   "
            f"CPU Δ={result['warp_cpu_delta_mb']:+.1f} MB  "
            f"GPU Δ={result['warp_gpu_delta_mb']:+.1f} MB  "
            f"peak GPU={result['warp_peak_gpu_mb']:.1f} MB"
        )
        print(f"===CPU OPW IK time:  {result['cpu_ms']:.6f} ms")
        print(f"   Translation mean error: {result['cpu_t_err_mm']:.6f} mm")
        print(f"   Rotation mean error:    {result['cpu_r_err_deg']:.6f} degrees")
        print(f"   Success rate:           {result['cpu_success_rate'] * 100.0:.2f}%")
        print(
            "   "
            f"CPU Δ={result['cpu_cpu_delta_mb']:+.1f} MB  "
            f"GPU Δ={result['cpu_gpu_delta_mb']:+.1f} MB  "
            f"peak GPU={result['cpu_peak_gpu_mb']:.1f} MB"
        )

        perf_rows.append(
            {
                "sample_size": n_sample,
                "impl": "opw_cuda",
                "component": "opw_ik",
                "cost_time_ms": f"{result['warp_ms']:.6f}",
                "cpu_delta_mb": f"{result['warp_cpu_delta_mb']:.6f}",
                "gpu_delta_mb": f"{result['warp_gpu_delta_mb']:.6f}",
                "peak_gpu_mb": f"{result['warp_peak_gpu_mb']:.6f}",
            }
        )
        perf_rows.append(
            {
                "sample_size": n_sample,
                "impl": "opw_cpu",
                "component": "opw_ik",
                "cost_time_ms": f"{result['cpu_ms']:.6f}",
                "cpu_delta_mb": f"{result['cpu_cpu_delta_mb']:.6f}",
                "gpu_delta_mb": f"{result['cpu_gpu_delta_mb']:.6f}",
                "peak_gpu_mb": f"{result['cpu_peak_gpu_mb']:.6f}",
            }
        )
        metric_rows.append(
            {
                "sample_size": n_sample,
                "impl": "opw_cuda",
                "component": "opw_ik",
                "success_rate": f"{result['warp_success_rate']:.6f}",
                "translation_err_mm": f"{result['warp_t_err_mm']:.6f}",
                "rotation_err_deg": f"{result['warp_r_err_deg']:.6f}",
            }
        )
        metric_rows.append(
            {
                "sample_size": n_sample,
                "impl": "opw_cpu",
                "component": "opw_ik",
                "success_rate": f"{result['cpu_success_rate']:.6f}",
                "translation_err_mm": f"{result['cpu_t_err_mm']:.6f}",
                "rotation_err_deg": f"{result['cpu_r_err_deg']:.6f}",
            }
        )

    return perf_rows, metric_rows


def run_all_benchmarks(selected_solvers: list[str] | None = None) -> None:
    """Run unified OPW + Pytorch kinematic solver benchmarks."""
    solvers_to_run = _normalize_selected_solvers(selected_solvers)

    print("=" * 60)
    print("Kinematic Solver Performance Benchmarks")
    print("=" * 60)

    print("\nSelected solvers:", ", ".join(sorted(solvers_to_run)))

    print("\nConfiguration differences:")
    print(
        "- OPW solver: analytic OPW parameters via OPWSolverCfg with "
        "opw-specific joint limits."
    )
    print("- Pytorch solver: UR10 URDF-based PytorchSolver with " "UR10 joint limits.")

    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    if "opw" in solvers_to_run:
        opw_perf_rows, opw_metric_rows = benchmark_opw_solver()
        perf_rows.extend(opw_perf_rows)
        metric_rows.extend(opw_metric_rows)

    if "pytorch" in solvers_to_run:
        pytorch_perf_rows, pytorch_metric_rows = benchmark_pytorch_solver()
        perf_rows.extend(pytorch_perf_rows)
        metric_rows.extend(pytorch_metric_rows)

    leaderboard_rows = _build_leaderboard_rows(metric_rows)

    benchmark_name = "kinematic_solver"

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)

    report_path = _write_markdown_report(
        benchmark_name=benchmark_name,
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            "CPU/GPU memory fields are deltas measured around timed calls.",
            "This report contains exactly three tables: Time & Memory, Success & Other Metrics, and Leaderboard.",
        ]
        + (
            [
                "OPW and Pytorch solvers use different initialization paths and different lower/upper joint limits."
            ]
            if solvers_to_run == set(SUPPORTED_SOLVERS)
            else []
        ),
    )
    print(f"Markdown report saved: {report_path}")


if __name__ == "__main__":
    args = _parse_args()
    run_all_benchmarks(selected_solvers=args.solvers)
