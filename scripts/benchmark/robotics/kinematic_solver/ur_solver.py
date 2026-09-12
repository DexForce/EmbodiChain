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

"""Benchmark UR single-solution, legacy selection, and all-solution IK paths.

Run: python -m scripts.benchmark.robotics.kinematic_solver.ur_solver

Targets use independent DH forward kinematics, avoiding robot assets, simulation,
and compiled FK setup. Timings exclude warmup and include CUDA synchronization.
Memory reports distinguish measured PyTorch allocation from the logical Warp
output-buffer payload; neither is a measurement of total device peak memory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import psutil
import torch
import warp as wp

from embodichain.lab.sim.motion.solvers import URSolver, URSolverCfg

__all__ = ["run_all_benchmarks"]

_MODES = ("single", "legacy", "all")
_MIB = 1024**2
_Result = tuple[torch.Tensor, torch.Tensor]


def _parse_args() -> argparse.Namespace:
    """Parse workload, repetition, and report controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--counts", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--json", type=Path, help="Optional JSON results path.")
    args = parser.parse_args()
    if min(args.counts) < 1 or args.repeats < 1 or args.warmup < 1:
        parser.error("counts, repeats, and warmup must all be positive")
    return args


def _synchronize(device: torch.device) -> None:
    """Wait for both Torch and Warp work on the requested CUDA device."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward_kinematics(qpos: torch.Tensor, cfg: URSolverCfg) -> torch.Tensor:
    """Compute batch UR DH transforms independently from the IK implementation."""
    qpos = qpos.to(dtype=torch.float64)
    poses = torch.eye(4, dtype=qpos.dtype, device=qpos.device).repeat(len(qpos), 1, 1)
    parameters = (
        (cfg.d1, 0.0, math.pi / 2),
        (0.0, cfg.a2, 0.0),
        (0.0, cfg.a3, 0.0),
        (cfg.d4, 0.0, math.pi / 2),
        (cfg.d5, 0.0, -math.pi / 2),
        (cfg.d6, 0.0, 0.0),
    )
    for joint, (d, a, alpha) in enumerate(parameters):
        c, s = torch.cos(qpos[:, joint]), torch.sin(qpos[:, joint])
        ca, sa = math.cos(alpha), math.sin(alpha)
        transform = torch.zeros_like(poses)
        transform[:, 0, 0] = c
        transform[:, 0, 1] = -s * ca
        transform[:, 0, 2] = s * sa
        transform[:, 0, 3] = a * c
        transform[:, 1, 0] = s
        transform[:, 1, 1] = c * ca
        transform[:, 1, 2] = -c * sa
        transform[:, 1, 3] = a * s
        transform[:, 2, 1] = sa
        transform[:, 2, 2] = ca
        transform[:, 2, 3] = d
        transform[:, 3, 3] = 1.0
        poses = poses @ transform
    return poses


def _solve(
    solver: URSolver, target: torch.Tensor, seed: torch.Tensor, mode: str
) -> _Result:
    """Run one API mode or reconstruct the previous Torch nearest selection."""
    valid, joints = solver.get_ik(
        target, qpos_seed=seed, return_all_solutions=mode != "single"
    )
    if mode != "legacy":
        return valid, joints
    distances = torch.norm(
        solver.ik_nearest_weight * (joints - seed.unsqueeze(1)), dim=-1
    )
    distances[~valid] = float("inf")
    indices = distances.argmin(dim=1)
    rows = torch.arange(len(target), device=target.device)
    return valid[rows, indices], joints[rows, indices]


def _measure(
    operation: Callable[[], _Result],
    device: torch.device,
    count: int,
    mode: str,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, object], _Result]:
    """Measure synchronized calls, then profile one separate warmed call."""
    for _ in range(warmup):
        result = operation()
        _synchronize(device)
        del result
    elapsed: list[float] = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        result = operation()
        _synchronize(device)
        elapsed.append(time.perf_counter() - start)
        del result

    process = psutil.Process()
    rss_before = process.memory_info().rss
    gpu_before = 0
    if device.type == "cuda":
        gpu_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    result = operation()
    _synchronize(device)
    gpu_delta, gpu_peak = 0, 0
    if device.type == "cuda":
        gpu_delta = torch.cuda.memory_allocated(device) - gpu_before
        gpu_peak = torch.cuda.max_memory_allocated(device) - gpu_before
    median = statistics.median(elapsed)
    # Current kernels allocate six float32 joints and one int32 validity per
    # output. Torch conversion shares joints but allocates boolean validity.
    warp_bytes = count * (1 if mode == "single" else 512) * (6 * 4 + 4)
    return {
        "sample_size": count,
        "impl": mode,
        "cost_time_ms": median * 1000,
        "total_repeats_ms": sum(elapsed) * 1000,
        "targets_per_second": count / median,
        "cpu_delta_mb": (process.memory_info().rss - rss_before) / _MIB,
        "gpu_delta_mb": gpu_delta / _MIB,
        "peak_gpu_mb": gpu_peak / _MIB,
        "warp_output_payload_mb": warp_bytes / _MIB,
    }, result


def _quality(
    result: _Result,
    reference: _Result,
    target: torch.Tensor,
    cfg: URSolverCfg,
    mode: str,
) -> dict[str, object]:
    """Check validity parity and reconstruction of one valid result per pose."""
    valid, joints = result
    ref_valid, ref_joints = reference
    if valid.ndim == 2:
        first = valid.to(dtype=torch.int32).argmax(dim=1)
        rows = torch.arange(len(target), device=target.device)
        joints = joints[rows, first]
        valid = valid.any(dim=1)
    torch.testing.assert_close(valid, ref_valid)
    max_joint_delta = None
    if mode != "all" and valid.any():
        max_joint_delta = (joints[valid] - ref_joints[valid]).abs().max().item()
        torch.testing.assert_close(
            joints[valid], ref_joints[valid], atol=1e-4, rtol=1e-5
        )
    translation_max = rotation_max = None
    if valid.any():
        reconstructed = _forward_kinematics(joints[valid], cfg)
        expected = target[valid].to(dtype=torch.float64)
        translation_max = (
            torch.linalg.vector_norm(
                reconstructed[:, :3, 3] - expected[:, :3, 3], dim=-1
            )
            .max()
            .item()
        )
        relative = reconstructed[:, :3, :3].transpose(1, 2) @ expected[:, :3, :3]
        skew = torch.stack(
            [
                relative[:, 2, 1] - relative[:, 1, 2],
                relative[:, 0, 2] - relative[:, 2, 0],
                relative[:, 1, 0] - relative[:, 0, 1],
            ],
            dim=-1,
        )
        sine = torch.linalg.vector_norm(skew, dim=-1) / 2
        cosine = (relative.diagonal(dim1=1, dim2=2).sum(dim=1) - 1) / 2
        rotation_max = torch.atan2(sine, cosine).max().item()
    return {
        "sample_size": len(target),
        "impl": mode,
        "success_rate": valid.float().mean().item(),
        "translation_max_m": translation_max,
        "rotation_max_rad": rotation_max,
        "max_joint_delta_vs_legacy": max_joint_delta,
        "validity_matches_legacy": True,
    }


def _write_report(payload: dict[str, object], output_dir: Path) -> Path:
    """Write the three required benchmark tables to one Markdown artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / f"ur_solver_{datetime.now():%Y%m%d_%H%M%S}.md"
    lines = ["# UR solver benchmark", "", str(payload["environment"]), ""]
    lines.extend(str(note) for note in payload["notes"])
    for title, key in (
        ("Time & Memory", "performance"),
        ("Success & Other Metrics", "quality"),
        ("Leaderboard", "leaderboard"),
    ):
        rows = payload[key]
        headers = list(rows[0])
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
        )
        for row in rows:
            values = [
                f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key])
                for key in headers
            ]
            lines.append("| " + " | ".join(values) + " |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


@torch.inference_mode()
def run_all_benchmarks() -> None:
    """Compare all three UR5 IK paths and write timing and accuracy reports."""
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Skipped: CUDA is unavailable; use --device cpu for a CPU run.")
        return
    wp.init()
    cfg = URSolverCfg(
        ur_type="ur5",
        joint_names=[f"joint_{joint}" for joint in range(6)],
        user_qpos_limits=[[-2 * math.pi] * 6, [2 * math.pi] * 6],
        ik_nearest_weight=[1.0, 0.5, 2.0, 1.5, 0.75, 3.0],
    )
    # IK uses only DH parameters. A supplied placeholder chain bypasses unrelated
    # URDF loading and torch.compile FK setup; FK validation above is independent.
    solver = cfg.init_solver(device=device, pk_serial_chain=object())
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    performance, quality = [], []
    print("UR5 analytic IK benchmark (warmup excluded; synchronized wall time)")
    for count in args.counts:
        qpos = (torch.rand((count, 6), generator=generator) * 2 - 1).to(device)
        target = _forward_kinematics(qpos * math.pi, cfg).float()
        seed = (
            (torch.rand((count, 6), generator=generator) * 2 - 1) * (2 * math.pi)
        ).to(device)
        reference = _solve(solver, target, seed, "legacy")
        for mode in _MODES:
            row, result = _measure(
                lambda: _solve(solver, target, seed, mode),
                device,
                count,
                mode,
                args.warmup,
                args.repeats,
            )
            performance.append(row)
            quality.append(_quality(result, reference, target, cfg, mode))
            del result
            print(
                f"n={count:>7d} {mode:>6s}: {row['cost_time_ms']:>9.3f} ms "
                f"| {row['targets_per_second']:>12.0f} targets/s "
                f"| Torch peak delta={row['peak_gpu_mb']:.3f} MiB "
                f"| Warp output payload={row['warp_output_payload_mb']:.3f} MiB"
            )
        del reference, target, seed, qpos

    leaderboard = []
    for mode in _MODES:
        rows = [row for row in quality if row["impl"] == mode]
        total = sum(row["sample_size"] for row in rows)
        success = sum(row["sample_size"] * row["success_rate"] for row in rows)
        leaderboard.append({"algorithm": mode, "overall_success_rate": success / total})
    leaderboard.sort(key=lambda row: row["overall_success_rate"], reverse=True)
    payload = {
        "environment": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "torch": torch.__version__,
            "warp": wp.__version__,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "notes": [
            "single uses return_all_solutions=False; all uses True; legacy expands "
            "all solutions then runs the original weighted Torch norm and argmin.",
            "cost_time_ms is the median; total_repeats_ms sums measured calls only. "
            "Input generation, warmup, memory profiling, and quality checks are excluded.",
            "Memory uses MiB. cpu_delta_mb is process RSS delta. gpu_delta_mb and "
            "peak_gpu_mb are measured PyTorch allocation increases for a separate "
            "warmed call; they exclude Warp allocations and are zero on CPU.",
            "warp_output_payload_mb is the logical payload of direct Warp output "
            "buffers (6 float32 joints plus int32 validity per candidate), on the "
            "selected device. It excludes allocator overhead and temporary/register "
            "storage. These columns do not measure total peak device memory.",
            "Targets are DH-generated reachable UR5 poses with limits ±2π, independent "
            "random seeds, and nonuniform weights. FK metrics check one valid "
            "representative per pose. Legacy parity checks run outside timings.",
        ],
        "performance": performance,
        "quality": quality,
        "leaderboard": [
            {"rank": rank, **row} for rank, row in enumerate(leaderboard, start=1)
        ],
    }
    report = _write_report(payload, args.output_dir)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Markdown report saved: {report}")


if __name__ == "__main__":
    run_all_benchmarks()
