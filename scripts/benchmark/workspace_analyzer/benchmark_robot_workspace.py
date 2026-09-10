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

"""Measure UR5 and CobotMagic workspace paths without initializing a renderer.

Run: python scripts/benchmark/workspace_analyzer/benchmark_robot_workspace.py --device cuda:0
Run the same script with PYTHONPATH and cwd pointing at a baseline checkout to
compare revisions. Asset chains and analytic kernels are real; root poses are
fixed snapshots supplied by a minimal Robot adapter. FK uses eager asset-chain
code so compiler startup and specialization do not bias revision comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Callable

import psutil
import torch
import warp as wp

from embodichain.lab.sim.objects.robot import Robot
from embodichain.lab.sim.robots import CobotMagicCfg, URRobotCfg
from embodichain.lab.sim.motion.workspace.analyzer import (
    AnalysisMode,
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
)
from embodichain.lab.sim.motion.workspace.configs import (
    CacheConfig,
    DimensionConstraint,
    SamplingConfig,
    SamplingStrategy,
)
from embodichain.utils import logger


def _robot(kind: str, device: torch.device) -> tuple[SimpleNamespace, str]:
    """Bind real Robot batch methods to an asset-backed solver and fixed base."""
    preset = (
        URRobotCfg.from_dict({"robot_type": "ur5"})
        if kind == "ur5"
        else CobotMagicCfg.from_dict({})
    )
    part = "arm" if kind == "ur5" else "left_arm"
    chain = preset.build_pk_serial_chain(device)[part]
    cfg = preset.solver_cfg[part]
    cfg.joint_names = chain.get_joint_parameter_names()
    solver = cfg.init_solver(device=device, pk_serial_chain=chain)
    solver.compiled_fk = chain.forward_kinematics_tensor
    base = torch.eye(4, device=device).unsqueeze(0)
    if kind == "cobotmagic":
        base[0, :3, 3] = base.new_tensor([0.233, 0.3, 0.0])
    joint_limits = torch.stack(
        (solver.lower_qpos_limits, solver.upper_qpos_limits), dim=-1
    )[None]
    robot = SimpleNamespace(
        device=device,
        cfg=preset,
        num_envs=1,
        _all_indices=[0],
        _solvers={part: solver},
        control_parts={part: cfg.joint_names},
        joint_names=cfg.joint_names,
        body_data=SimpleNamespace(qpos_limits=joint_limits),
        get_joint_ids=lambda *args, **kw: list(range(6)),
        get_qpos=lambda: solver.get_default_qpos_seed()[None],
        get_link_pose=lambda **kw: base,
    )
    for name in ("compute_fk", "compute_batch_fk", "compute_batch_ik"):
        setattr(robot, name, MethodType(getattr(Robot, name), robot))
    return robot, part


class _Progress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def close(self):
        pass


def _measure(
    call: Callable, device: torch.device, repeats: int
) -> tuple[float, float, float, float, object]:
    """Return warmed median milliseconds, RSS/VRAM deltas, peak VRAM and output."""

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            wp.synchronize_device(str(device))

    call()
    sync()
    process = psutil.Process(os.getpid())
    before_cpu = process.memory_info().rss
    before_gpu = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    output = None
    for _ in range(repeats):
        output = None
        sync()
        start = time.perf_counter()
        output = call()
        sync()
        timings.append((time.perf_counter() - start) * 1000)
    cpu_delta = (process.memory_info().rss - before_cpu) / 1024**2
    gpu_delta = (
        (torch.cuda.memory_allocated(device) - before_gpu) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    peak_gpu = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    return statistics.median(timings), cpu_delta, gpu_delta, peak_gpu, output


def _accuracy(
    robot, part: str, mode: AnalysisMode, result: dict
) -> tuple[float, float]:
    """Measure IK output against the default robot FK and target orientation."""
    qpos = result["joint_configurations"]
    if mode == AnalysisMode.JOINT_SPACE or not len(qpos):
        return 0.0, 0.0
    positions = result["reachable_points"]
    target = robot.compute_fk(robot.get_qpos(), name=part, to_matrix=True)
    actual = robot.compute_batch_fk(qpos[None], name=part, to_matrix=True)[0]
    translation = (
        torch.linalg.vector_norm(actual[:, :3, 3] - positions, dim=-1).mean() * 1000
    )
    relative = actual[:, :3, :3].transpose(-1, -2) @ target[0, :3, :3]
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return float(translation), float(torch.acos(cosine).mean() * 180 / torch.pi)


def _table(rows: list[dict]) -> list[str]:
    keys = list(rows[0])
    return [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ] + ["| " + " | ".join(str(row[key]) for key in keys) + " |" for row in rows]


def run_all_benchmarks(args: argparse.Namespace) -> None:
    """Measure three modes and dynamic bounds, then write one Markdown report.

    Args:
        args: CLI device, sample counts, batch sizes, seeds and output settings.
    """
    wp.init()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Skipped: CUDA unavailable")
        return
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    logger.set_log_level("ERROR")
    perf, quality = [], []
    for kind in ("ur5", "cobotmagic"):
        robot, part = _robot(kind, device)
        sim = SimpleNamespace(device=device, num_envs=1)
        for n in args.samples:
            for batch in args.batch_sizes:
                for mode in AnalysisMode:
                    for seeds in (
                        [1] if mode == AnalysisMode.JOINT_SPACE else args.seeds
                    ):
                        cfg = WorkspaceAnalyzerConfig(
                            mode=mode,
                            control_part_name=part,
                            sampling=SamplingConfig(
                                strategy=SamplingStrategy(args.sampler),
                                num_samples=n,
                                batch_size=batch,
                                seed=42,
                            ),
                            cache=CacheConfig(enabled=False),
                            constraint=DimensionConstraint(
                                min_bounds=(
                                    [-1.3, -1.3, 0.0]
                                    if kind == "ur5"
                                    else [-0.5, -0.5, 0.0]
                                ),
                                max_bounds=(
                                    [1.3, 1.3, 1.3]
                                    if kind == "ur5"
                                    else [1.0, 1.0, 1.0]
                                ),
                            ),
                            reference_pose=robot.compute_fk(
                                robot.get_qpos(), name=part, to_matrix=True
                            ),
                            ik_samples_per_point=seeds,
                            plane_normal=torch.tensor([0.0, 0.0, 1.0], device=device),
                            plane_point=torch.tensor([0.0, 0.0, 0.3], device=device),
                            plane_bounds=torch.tensor(
                                [[-1.0, 1.0], [-1.0, 1.0]], device=device
                            ),
                        )
                        analyzer = WorkspaceAnalyzer(robot, cfg, sim)
                        analyzer._create_optimized_tqdm = lambda it, **kw: _Progress(it)
                        analyzer._update_progress_with_stats = lambda *a, **kw: None
                        analyzer._log_analysis_summary = lambda *a: None

                        def call():
                            # Each repeat uses the identical domain and random sequence.
                            analyzer.sampler = analyzer._create_sampler()
                            return analyzer.analyze(
                                force_recompute=True, visualize=False
                            )

                        ms, cpu, gpu, peak, result = _measure(
                            call, device, args.repeats
                        )
                        label = f"{kind}/{mode.value}/K{seeds}/B{batch}/N{n}"
                        row = dict(
                            algorithm=label,
                            cost_time_ms=round(ms, 3),
                            cpu_delta_mb=round(cpu, 3),
                            gpu_delta_mb=round(gpu, 3),
                            peak_gpu_mb=round(peak, 3),
                        )
                        perf.append(row)
                        rate = (
                            result.get("num_reachable", result.get("num_valid", 0)) / n
                        )
                        te, re = _accuracy(robot, part, mode, result)
                        quality.append(
                            dict(
                                algorithm=label,
                                success_rate=round(rate, 6),
                                translation_err_mm=round(te, 5),
                                rotation_err_deg=round(re, 5),
                            )
                        )
                        print(json.dumps(row), flush=True)
        analyzer = WorkspaceAnalyzer(
            robot,
            WorkspaceAnalyzerConfig(
                control_part_name=part, cache=CacheConfig(enabled=False)
            ),
            sim,
        )
        analyzer._create_optimized_tqdm = lambda it, **kw: _Progress(it)
        analyzer._update_progress_with_stats = lambda *a, **kw: None
        ms, cpu, gpu, peak, bounds = _measure(
            analyzer._compute_dynamic_workspace_bounds, device, args.repeats
        )
        label = kind + "/dynamic_bounds/N1000"
        perf.append(
            dict(
                algorithm=label,
                cost_time_ms=round(ms, 3),
                cpu_delta_mb=round(cpu, 3),
                gpu_delta_mb=round(gpu, 3),
                peak_gpu_mb=round(peak, 3),
            )
        )
        quality.append(
            dict(
                algorithm=label,
                success_rate=float(torch.isfinite(bounds).all()),
                translation_err_mm="n/a",
                rotation_err_deg="n/a",
            )
        )
    rank = [
        dict(rank=i + 1, **row)
        for i, row in enumerate(
            sorted(quality, key=lambda row: row["success_rate"], reverse=True)
        )
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    notes = [
        f"Revision: {commit}; label: {args.label}; device: {device}; threads: {args.threads}; repeats: {args.repeats}; sampler: {args.sampler}.",
        "Times include sampling, Robot frame transforms, solver computation and basic metrics; exclude robot setup, visualization, persistent cache and initial warm-up.",
        "No simulator is constructed. Fixed base snapshots exercise real Robot methods, real asset chains (eager FK) and real analytic IK kernels.",
        "CPU memory is RSS delta. GPU columns count PyTorch allocator memory only; baseline Warp-owned allocations are excluded, so they cannot establish total VRAM savings.",
        "IK residuals use default asset FK. Asset/analytic geometry differences and existing solver tolerances may produce nonzero residuals; compare both revisions before attributing them to these changes.",
        "Success rates of joint, Cartesian and plane modes describe different domains and are not interchangeable quality rankings.",
    ]
    lines = [
        "# Robot workspace benchmark",
        "",
        *notes,
        "",
        "## Time & Memory",
        "",
        *_table(perf),
        "",
        "## Success & Other Metrics",
        "",
        *_table(quality),
        "",
        "## Leaderboard",
        "",
        *_table(rank),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown report saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", nargs="+", type=int, default=[1024, 8192])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1024])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 4])
    parser.add_argument(
        "--sampler", choices=["random", "sobol", "lhs"], default="random"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/robot_workspace.md")
    )
    args = parser.parse_args()
    if (
        min(args.samples + args.batch_sizes + args.seeds + [args.repeats, args.threads])
        <= 0
    ):
        parser.error("counts must be positive")
    run_all_benchmarks(args)
