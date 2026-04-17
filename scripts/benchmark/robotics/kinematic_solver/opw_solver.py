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

"""Benchmark script for OPW solver Warp CUDA vs CPU implementation.

Measures FK/IK wall-clock latency, pose accuracy, success rate, and memory usage.
Run: python -m scripts.benchmark.robotics.kinematic_solver.opw_solver
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch

from embodichain.lab.sim.solvers.opw_solver import OPWSolverCfg

LOWER_LIMITS = [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944]
UPPER_LIMITS = [2.618, 3.14159, 0.0, 1.745, 1.22, 2.0944]
SAMPLE_SIZES = [100, 1000, 10000]


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


def check_opw_solver(
    solver_warp, solver_py_opw, n_samples: int = 1000
) -> dict[str, float]:
    """Run Warp and CPU OPW IK/FK checks and return timing, memory, and accuracy."""
    dof = 6
    qpos_np = np.random.uniform(
        low=np.array(LOWER_LIMITS)
        + 5.1 / 180.0 * np.pi,  # add a margin to avoid sampling near the joint limits
        high=np.array(UPPER_LIMITS) + -5.1 / 180.0 * np.pi,
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
        user_qpos_limits=(LOWER_LIMITS, UPPER_LIMITS),
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
                "impl": "warp_cuda",
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
                "impl": "cpu_opw",
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
                "impl": "warp_cuda",
                "component": "opw_ik",
                "success_rate": f"{result['warp_success_rate']:.6f}",
                "other_metrics": (
                    f"translation_err_mm={result['warp_t_err_mm']:.6f}, "
                    f"rotation_err_deg={result['warp_r_err_deg']:.6f}"
                ),
            }
        )
        metric_rows.append(
            {
                "sample_size": n_sample,
                "impl": "cpu_opw",
                "component": "opw_ik",
                "success_rate": f"{result['cpu_success_rate']:.6f}",
                "other_metrics": (
                    f"translation_err_mm={result['cpu_t_err_mm']:.6f}, "
                    f"rotation_err_deg={result['cpu_r_err_deg']:.6f}"
                ),
            }
        )

    return perf_rows, metric_rows


def run_all_benchmarks() -> None:
    """Run all OPW solver benchmarks."""
    print("=" * 60)
    print("OPW Solver Performance Benchmarks")
    print("=" * 60)

    perf_rows, metric_rows = benchmark_opw_solver()

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)

    report_path = _write_markdown_report(
        benchmark_name="opw_solver",
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
