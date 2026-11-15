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

"""Compare grasp approaches in the existing Franka/cuRobo obstacle example.

Run: python -m scripts.benchmark.motion_generation.curobo_grasp_approach

Both variants start at the same reached pre-grasp state and target the same
pose behind the example's obstacle. Only the final approach path constraint
changes. This measures planned geometry, not physical grasp success. It retains
the pre-grasp waypoint and does not implement late-activated, blended optimization.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import psutil
import torch


def _quality(
    robot: Any,
    result: Any,
    goal: torch.Tensor,
    direction: torch.Tensor,
    position_tolerance: float,
    orientation_tolerance: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure TCP geometry with the simulator Robot's independent FK adapter."""
    poses = robot.compute_batch_fk(qpos=result.positions, name="arm", to_matrix=True)
    delta = poses[..., :3, 3] - goal[:, None, :3, 3]
    lateral = (
        delta - (delta * direction[:, None]).sum(-1, keepdim=True) * direction[:, None]
    )
    lateral_error = torch.linalg.vector_norm(lateral, dim=-1)
    relative = goal[:, None, :3, :3].transpose(-1, -2) @ poses[..., :3, :3]
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    sine = torch.linalg.matrix_norm(relative - relative.transpose(-1, -2)) / math.sqrt(
        8
    )
    angle = torch.atan2(sine, cosine)
    finite = torch.isfinite(poses).all(dim=(-2, -1)).all(dim=1)
    native = result.success & finite
    qualified = (
        native
        & (lateral_error.max(dim=1).values <= position_tolerance)
        & (angle.max(dim=1).values <= orientation_tolerance)
    )
    selected = native[:, None].expand_as(lateral_error)
    metrics = {
        "planned": int(native.sum()),
        "qualified": int(qualified.sum()),
        "attempted": goal.shape[0],
        "lateral_mm": (
            float(lateral_error[selected].max() * 1000) if bool(native.any()) else None
        ),
        "orientation_deg": (
            float(torch.rad2deg(angle[selected]).max()) if bool(native.any()) else None
        ),
        "duration_s": (
            float(result.duration[native].mean()) if bool(native.any()) else None
        ),
    }
    trace = {
        "tcp": poses[:, :, :3, 3].cpu().numpy(),
        "lateral_mm": lateral_error.cpu().numpy() * 1000,
        "orientation_deg": torch.rad2deg(angle).cpu().numpy(),
        "time": result.dt.cumsum(1).cpu().numpy(),
        "qpos": result.positions.cpu().numpy(),
        "success": result.success.cpu().numpy(),
    }
    return metrics, trace


def _plot(
    traces: dict[str, dict[str, np.ndarray]],
    goal: np.ndarray,
    output: Path,
    position_tolerance: float,
) -> None:
    """Save a trajectory overlay and time-based deviation plots for row zero."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 4), constrained_layout=True)
    path = fig.add_subplot(131, projection="3d")
    distance = fig.add_subplot(132)
    rotation = fig.add_subplot(133)
    for label, trace in traces.items():
        xyz = trace["tcp"][0]
        suffix = "" if trace["success"][0] else " (failed: start hold)"
        path.plot(*xyz.T, label=label + suffix)
        distance.plot(trace["time"][0], trace["lateral_mm"][0], label=label)
        rotation.plot(trace["time"][0], trace["orientation_deg"][0], label=label)
    path.scatter(*goal, c="black", marker="x", label="Grasp target")
    path.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title="Final approach, environment 0",
    )
    path.legend(fontsize=7)
    distance.set(xlabel="Trajectory time (s)", ylabel="Perpendicular TCP error (mm)")
    distance.axhline(
        position_tolerance * 1000, color="black", linestyle="--", label="Tolerance"
    )
    rotation.set(
        xlabel="Trajectory time (s)", ylabel="Target orientation error (degrees)"
    )
    distance.legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _write_report(
    output: Path,
    measurements: dict[str, list[dict[str, Any]]],
    memory: dict[str, dict[str, float]],
    metadata: dict[str, Any],
) -> None:
    """Write the three benchmark tables with explicit measurement limits."""
    lines = [
        "# cuRobo grasp approach comparison",
        "",
        "```json",
        json.dumps(metadata, indent=2),
        "```",
        "",
    ]
    time_rows, quality_rows, ranking = [], [], []
    for label, values in measurements.items():
        attempted = sum(value["attempted"] for value in values)
        success = sum(value["qualified"] for value in values) / attempted
        native = sum(value["planned"] for value in values) / attempted
        valid = [value for value in values if value["planned"]]
        time_rows.append(
            {
                "variant": label,
                "cost_time_ms": statistics.median(
                    value["latency_ms"] for value in values
                ),
                "repeats": len(values),
                **memory[label],
            }
        )
        quality_rows.append(
            {
                "variant": label,
                "success_rate": success,
                "planner_success_rate": native,
                "max_lateral_mm": (
                    max(value["lateral_mm"] for value in valid)
                    if valid
                    else "unavailable"
                ),
                "max_orientation_deg": (
                    max(value["orientation_deg"] for value in valid)
                    if valid
                    else "unavailable"
                ),
                "mean_duration_s": (
                    statistics.mean(value["duration_s"] for value in valid)
                    if valid
                    else "unavailable"
                ),
            }
        )
        ranking.append({"algorithm": label, "overall_success_rate": success})
    ranking.sort(key=lambda row: row["overall_success_rate"], reverse=True)
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    for title, rows in (
        ("Time & Memory", time_rows),
        ("Success & Other Metrics", quality_rows),
        ("Leaderboard", ranking),
    ):
        headers = list(rows[0])
        lines.extend(
            [
                f"## {title}",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
        )
        for row in rows:
            cells = [
                f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key])
                for key in headers
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "Both variants use identical starts, targets, obstacles, dynamic limits, and reset native sampling seeds.",
            "Timing is warmed, synchronized wall latency of MotionGenerator.generate, including adapter validation and copies; FK scoring and plotting are excluded.",
            "Memory is a separate warmed call retaining one result: CPU RSS retained delta and Torch CUDA allocator deltas/peaks in MiB. Persistent shared-backend buffers/caches are included in warmed_gpu_mb; native/Warp allocations are outside this accounting.",
            "success_rate requires planner success plus every sampled TCP meeting the same line/orientation tolerances. It is not physical grasp success or continuous collision certification.",
            "The existing obstacle and native sampled collision checks are retained; self-collision remains disabled by the existing adapter.",
            "Pre-grasp routing is shared preparation. Both final approaches retain their native time grids and the required grasp stop. No closed-gripper execution or late-activated single optimization is claimed.",
            "",
            "![Approach comparison](comparison.png)",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def run_all_benchmarks(args: argparse.Namespace) -> Path | None:
    """Compare the two final approaches serially in one unchanged example scene.

    Args:
        args: Parsed batch, sampling, constraint and output settings.

    Returns:
        Report path, or ``None`` when CUDA or cuRobo is unavailable.
    """
    if not torch.cuda.is_available() or importlib.util.find_spec("curobo") is None:
        print("Skipped: this comparison requires CUDA and cuRobo V2-Fast.")
        return None
    os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    import dexsim
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenCfg,
        MotionGenOptions,
        MotionGenerator,
    )
    from embodichain.lab.sim.motion.planners import (
        CuroboLinearConstraintCfg,
        CuroboPlanOptions,
        CuroboPlannerCfg,
        CuroboWorldCfg,
        PlanState,
    )
    from examples.sim.motion.planners.curobo_planner import _build_scene
    from embodichain.utils import logger

    logger.set_log_level("WARNING")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    sim, robot, obstacle, goal, part = _build_scene(
        True, "franka", "cuda:0", args.num_envs, "no-render"
    )
    generator = None
    try:
        generator = MotionGenerator(
            MotionGenCfg(
                planner_cfg=CuroboPlannerCfg(
                    robot_uid=robot.uid,
                    world=CuroboWorldCfg(rigid_objects=[obstacle]),
                    preserve_plan_samples=True,
                    use_cuda_graph=args.cuda_graph,
                    warmup_iterations=1,
                    max_attempts=args.max_attempts,
                )
            )
        )
        goal = goal.expand(args.num_envs, -1, -1).clone()
        pre_grasp = goal.clone()
        pre_grasp[:, :3, 3] -= goal[:, :3, 2] * args.approach_distance
        initial = robot.get_qpos(name=part).clone()
        prepared = generator.generate(
            [PlanState.from_xpos(pre_grasp)],
            MotionGenOptions(
                start_qpos=initial,
                control_part=part,
            ),
        )
        if not bool(prepared.success.all()):
            raise RuntimeError(
                f"Shared pre-grasp routing failed: {prepared.success.tolist()}"
            )
        start = prepared.positions[:, -1].clone()
        del prepared
        constraint = CuroboLinearConstraintCfg(
            position_tolerance=args.position_tolerance,
            orientation_tolerance=math.radians(args.orientation_tolerance_deg),
            weight=args.weight,
        )
        options = {
            "before": MotionGenOptions(start_qpos=start, control_part=part),
            "after": MotionGenOptions(
                start_qpos=start,
                control_part=part,
                plan_opts=CuroboPlanOptions(path_constraints=[constraint]),
            ),
        }
        backend = next(iter(generator.planner._backend_cache.values()))
        synchronize = lambda: torch.cuda.synchronize(0)

        def plan(label: str) -> Any:
            """Use the same native sampling reset for each compared request."""
            return generator.generate([PlanState.from_xpos(goal)], options[label])

        def prepare_call() -> None:
            """Reset Torch and native seeds outside the measurement interval."""
            torch.manual_seed(args.seed)
            backend.planner.reset_seed()
            synchronize()

        for _ in range(args.warmup):
            for label in options:
                prepare_call()
                result = plan(label)
                synchronize()
                del result
        measurements: dict[str, list[dict[str, Any]]] = {label: [] for label in options}
        traces = {}
        direction = goal[:, :3, 2]
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        )
        for repeat in range(args.repeats):
            for label in (
                list(options) if repeat % 2 == 0 else list(reversed(options))
            ):
                prepare_call()
                begin = time.perf_counter()
                result = plan(label)
                synchronize()
                latency_ms = (time.perf_counter() - begin) * 1000
                metrics, traces[label] = _quality(
                    robot,
                    result,
                    goal,
                    direction,
                    constraint.position_tolerance,
                    constraint.orientation_tolerance,
                )
                metrics["latency_ms"] = latency_ms
                measurements[label].append(metrics)
                print(f"{label}: {metrics}", flush=True)
                del result
        memory = {}
        process = psutil.Process()
        mib = 1024**2
        for label in options:
            prepare_call()
            cpu_before = process.memory_info().rss
            gpu_before = torch.cuda.memory_allocated(0)
            torch.cuda.reset_peak_memory_stats(0)
            result = plan(label)
            synchronize()
            memory[label] = {
                "cpu_delta_mb": (process.memory_info().rss - cpu_before) / mib,
                "warmed_gpu_mb": gpu_before / mib,
                "gpu_delta_mb": (torch.cuda.memory_allocated(0) - gpu_before) / mib,
                "peak_gpu_mb": torch.cuda.max_memory_allocated(0) / mib,
            }
            del result
        metadata = {
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
                timespec="seconds"
            ),
            "example": "examples/sim/motion/planners/curobo_planner.py (Franka obstacle scene)",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "curobo": importlib.metadata.version("nvidia-curobo"),
            "dexsim": dexsim.__version__,
            "dtype": "float32",
            "batch_size": args.num_envs,
            "dof": start.shape[-1],
            "torch_threads": torch.get_num_threads(),
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cuda_graph": args.cuda_graph,
            "max_attempts": args.max_attempts,
            "approach_distance_m": args.approach_distance,
            "constraint": constraint.to_dict(),
            "start_qpos": start.cpu().tolist(),
            "target": goal.cpu().tolist(),
            "scope": "final approach planning; physical execution unmeasured",
        }
        for label, trace in traces.items():
            np.savez(args.output_dir / f"{label}.npz", **trace)
        (args.output_dir / "measurements.json").write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "measurements": measurements,
                    "memory": memory,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _plot(
            traces,
            goal[0, :3, 3].cpu().numpy(),
            args.output_dir / "comparison.png",
            constraint.position_tolerance,
        )
        report = args.output_dir / "report.md"
        _write_report(report, measurements, memory, metadata)
        print(f"Markdown report saved: {report}")
        return report
    finally:
        if generator is not None:
            generator.planner.close()
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Parse bounded comparison settings and write reproducible artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks/curobo_grasp_approach"),
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--approach-distance", type=float, default=0.15)
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if min(args.num_envs, args.warmup, args.repeats, args.max_attempts) < 1:
        parser.error("batch size, warmup, repeats, and attempts must be positive")
    if not math.isfinite(args.approach_distance) or args.approach_distance <= 0:
        parser.error("approach distance must be finite and positive")
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
