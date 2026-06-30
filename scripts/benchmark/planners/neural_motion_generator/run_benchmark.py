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

"""Benchmark NeuralPlanner planning latency, memory, and trajectory quality.

Measures ``MotionGenerator.generate()`` wall-clock time, CPU/GPU memory usage,
success rate, and end-effector pose error on Franka Panda across waypoint counts.
Optionally compares against ToppraPlanner on the same waypoint sets.
Run: python -m scripts.benchmark.planners.neural_motion_generator.run_benchmark
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch

from embodichain.data import get_data_path
from embodichain.data.assets.planner_assets import download_neural_planner_checkpoint
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    MoveType,
    NeuralPlannerCfg,
    PlanState,
    ToppraPlannerCfg,
    ToppraPlanOptions,
)
from embodichain.lab.sim.planners.neural_planner import NeuralPlanOptions
from embodichain.lab.sim.planners.utils import PlanResult, TrajectorySampleMethod

DEFAULT_NUM_WAYPOINTS = [1, 3, 5]
ARM_NAME = "main_arm"
DEFAULT_START_QPOS = [
    0.0,
    -math.pi / 4,
    0.0,
    -3 * math.pi / 4,
    0.0,
    math.pi / 2,
    math.pi / 4,
]


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for neural motion generator benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark NeuralPlanner planning latency and quality."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Simulation and planner device. Auto uses CUDA when available.",
    )
    parser.add_argument(
        "--num-waypoints",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_WAYPOINTS,
        help="Number of EEF waypoints to sweep.",
    )
    parser.add_argument(
        "--compare-toppra",
        action="store_true",
        help="Also benchmark ToppraPlanner on the same waypoint sets.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Local neural planner checkpoint path. Skips HuggingFace download.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run simulation headlessly (default: True).",
    )
    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Open the simulation viewer window.",
    )
    return parser.parse_args()


def _resolve_device(device_name: str) -> str:
    """Resolve requested device name to a simulation device string."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def _memory_snapshot() -> dict[str, float]:
    process = psutil.Process(os.getpid())
    cpu_mb = process.memory_info().rss / 1024**2
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}


def _format_markdown_table(rows: list[dict[str, object]]) -> list[str]:
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
    leaderboard_rows: list[dict[str, object]],
    notes: list[str] | None = None,
) -> Path:
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


def _franka_tcp() -> list[list[float]]:
    c = math.cos(-math.pi / 4)
    s = math.sin(-math.pi / 4)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1034],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _create_franka(sim: SimulationManager) -> Robot:
    urdf = get_data_path("Franka/Panda/PandaWithHand.urdf")
    if not os.path.isfile(urdf):
        raise FileNotFoundError(f"Franka URDF not found: {urdf}")

    cfg_dict = {
        "fpath": urdf,
        "control_parts": {
            ARM_NAME: [
                "Joint1",
                "Joint2",
                "Joint3",
                "Joint4",
                "Joint5",
                "Joint6",
                "Joint7",
            ],
        },
        "solver_cfg": {
            ARM_NAME: {
                "class_type": "PytorchSolver",
                "end_link_name": "ee_link",
                "root_link_name": "base_link",
                "tcp": _franka_tcp(),
            },
        },
    }
    return sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))


def _make_waypoints(start_pose: torch.Tensor, num_waypoints: int) -> torch.Tensor:
    offsets = torch.tensor(
        [
            [0.10, 0.00, 0.00],
            [0.10, 0.10, 0.00],
            [0.00, 0.10, -0.08],
            [-0.10, 0.10, -0.08],
            [-0.10, 0.00, 0.00],
            [0.00, -0.10, 0.00],
            [0.10, -0.10, -0.06],
            [0.00, 0.00, -0.12],
        ],
        dtype=start_pose.dtype,
        device=start_pose.device,
    )
    num_waypoints = max(1, min(int(num_waypoints), offsets.shape[0]))
    waypoints = start_pose.unsqueeze(0).repeat(num_waypoints, 1, 1)
    waypoints[:, :3, 3] += offsets[:num_waypoints]
    return waypoints


def _make_target_states(waypoints: torch.Tensor) -> list[PlanState]:
    return [
        PlanState(move_type=MoveType.EEF_MOVE, xpos=waypoint) for waypoint in waypoints
    ]


def get_pose_err(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
) -> tuple[float, float]:
    """Return translation (m) and rotation (rad) errors between paired 4x4 poses."""
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
    return float(t_err.item()), float(r_err.item())


def _resolve_checkpoint(checkpoint_path: str | None) -> str | None:
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.is_file():
            print(f"Checkpoint not found: {path}")
            return None
        return str(path)

    try:
        return download_neural_planner_checkpoint()
    except RuntimeError as exc:
        print(str(exc))
        print(
            "Neural motion generator benchmark skipped: checkpoint unavailable.\n"
            "Provide --checkpoint-path or configure HF_TOKEN after accepting the "
            "model license at https://huggingface.co/dexforce/neural_motion_generator"
        )
        return None


def _setup_sim_and_robot(
    sim_device: str,
    headless: bool,
) -> tuple[SimulationManager, Robot, torch.Tensor, torch.Tensor]:
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=headless,
            sim_device=sim_device,
            num_envs=1,
            arena_space=2.0,
        )
    )
    robot = _create_franka(sim)
    start_qpos = torch.tensor(
        DEFAULT_START_QPOS,
        dtype=torch.float32,
        device=robot.device,
    )
    robot.set_qpos(
        qpos=start_qpos.unsqueeze(0),
        joint_ids=robot.get_joint_ids(ARM_NAME),
    )
    sim.update(step=1)

    start_pose = robot.compute_fk(
        qpos=start_qpos.unsqueeze(0),
        name=ARM_NAME,
        to_matrix=True,
    )[0]
    return sim, robot, start_qpos, start_pose


def _neural_motion_options(start_qpos: torch.Tensor) -> MotionGenOptions:
    return MotionGenOptions(
        control_part=ARM_NAME,
        start_qpos=start_qpos,
        plan_opts=NeuralPlanOptions(
            control_part=ARM_NAME,
            start_qpos=start_qpos,
        ),
    )


def _toppra_motion_options(start_qpos: torch.Tensor) -> MotionGenOptions:
    return MotionGenOptions(
        control_part=ARM_NAME,
        start_qpos=start_qpos,
        is_interpolate=True,
        is_linear=True,
        plan_opts=ToppraPlanOptions(
            constraints={"velocity": 0.2, "acceleration": 0.5},
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=20,
        ),
    )


def _final_eef_pose(
    result: PlanResult,
    robot: Robot,
) -> torch.Tensor | None:
    if result.xpos_list is not None and result.xpos_list.shape[0] > 0:
        return result.xpos_list[-1]
    if result.positions is None or result.positions.shape[0] == 0:
        return None
    qpos = result.positions[-1]
    if qpos.dim() == 1:
        qpos = qpos.unsqueeze(0)
    return robot.compute_fk(qpos=qpos, name=ARM_NAME, to_matrix=True)[0]


def _compute_result_metrics(
    result: PlanResult,
    last_waypoint: torch.Tensor,
    robot: Robot,
) -> dict[str, object]:
    success = bool(
        result.success.item() if torch.is_tensor(result.success) else result.success
    )
    rollout_steps = (
        int(result.positions.shape[0]) if result.positions is not None else 0
    )
    duration_s = float(
        result.duration.item() if torch.is_tensor(result.duration) else result.duration
    )

    final_pose = _final_eef_pose(result, robot)
    if final_pose is not None:
        t_err_m, r_err_rad = get_pose_err(final_pose, last_waypoint)
        translation_err_mm = t_err_m * 1000.0
        rotation_err_deg = r_err_rad * 180.0 / math.pi
    else:
        translation_err_mm = float("inf")
        rotation_err_deg = float("inf")

    return {
        "success_rate": float(success),
        "translation_err_mm": translation_err_mm,
        "rotation_err_deg": rotation_err_deg,
        "rollout_steps": rollout_steps,
        "duration_s": duration_s,
    }


def _timed_generate(
    motion_generator: MotionGenerator,
    target_states: list[PlanState],
    options: MotionGenOptions,
) -> tuple[float, dict[str, float], float, PlanResult]:
    motion_generator.generate(target_states=target_states, options=options)

    _reset_peak_gpu_memory()
    mem_before = _memory_snapshot()
    _sync_cuda()

    start = time.perf_counter()
    result = motion_generator.generate(target_states=target_states, options=options)
    _sync_cuda()
    elapsed = time.perf_counter() - start

    mem_after = _memory_snapshot()
    deltas = {
        "cpu_mb": mem_after["cpu_mb"] - mem_before["cpu_mb"],
        "gpu_mb": mem_after["gpu_mb"] - mem_before["gpu_mb"],
    }
    return elapsed, deltas, _peak_gpu_memory_mb(), result


def _format_finite(value: object) -> str:
    numeric = float(value)
    return f"{numeric:.6f}" if math.isfinite(numeric) else "inf"


def _append_result_rows(
    *,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    impl: str,
    num_waypoints: int,
    elapsed_s: float,
    mem_deltas: dict[str, float],
    peak_gpu_mb: float,
    metrics: dict[str, object],
) -> None:
    cost_time_ms = elapsed_s * 1000.0
    perf_rows.append(
        {
            "num_waypoints": num_waypoints,
            "impl": impl,
            "cost_time_ms": f"{cost_time_ms:.6f}",
            "cpu_delta_mb": f"{mem_deltas['cpu_mb']:.6f}",
            "gpu_delta_mb": f"{mem_deltas['gpu_mb']:.6f}",
            "peak_gpu_mb": f"{peak_gpu_mb:.6f}",
        }
    )
    metric_rows.append(
        {
            "num_waypoints": num_waypoints,
            "impl": impl,
            "success_rate": f"{float(metrics['success_rate']):.6f}",
            "translation_err_mm": _format_finite(metrics["translation_err_mm"]),
            "rotation_err_deg": _format_finite(metrics["rotation_err_deg"]),
            "rollout_steps": metrics["rollout_steps"],
            "duration_s": f"{float(metrics['duration_s']):.6f}",
        }
    )


def _build_leaderboard_rows(
    metric_rows: list[dict[str, object]],
    perf_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate and rank planners by overall success rate."""
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

    cost_by_impl: dict[str, list[float]] = {}
    for row in perf_rows:
        cost_by_impl.setdefault(str(row["impl"]), []).append(float(row["cost_time_ms"]))

    ranked = sorted(
        aggregate.items(),
        key=lambda item: item[1]["success_sum"] / max(item[1]["count"], 1.0),
        reverse=True,
    )

    leaderboard_rows: list[dict[str, object]] = []
    for rank, (algorithm, stats) in enumerate(ranked, start=1):
        count = max(stats["count"], 1.0)
        costs = cost_by_impl.get(algorithm, [0.0])
        avg_cost_ms = sum(costs) / len(costs)
        leaderboard_rows.append(
            {
                "rank": rank,
                "algorithm": algorithm,
                "overall_success_rate": f"{stats['success_sum'] / count:.2%}",
                "avg_cost_time_ms": f"{avg_cost_ms:.6f}",
                "avg_translation_err_mm": f"{stats['t_err_sum'] / count:.6f}",
                "avg_rotation_err_deg": f"{stats['r_err_sum'] / count:.6f}",
            }
        )
    return leaderboard_rows


def _print_run_summary(
    impl: str,
    num_waypoints: int,
    elapsed_s: float,
    mem_deltas: dict[str, float],
    peak_gpu_mb: float,
    metrics: dict[str, object],
) -> None:
    print(f"**** {impl} num_waypoints={num_waypoints}")
    print(
        f"===Plan time: {elapsed_s * 1000.0:.6f} ms  "
        f"success={float(metrics['success_rate']):.0f}  "
        f"steps={metrics['rollout_steps']}"
    )
    print(
        "   "
        f"CPU Δ={mem_deltas['cpu_mb']:+.1f} MB  "
        f"GPU Δ={mem_deltas['gpu_mb']:+.1f} MB  "
        f"peak GPU={peak_gpu_mb:.1f} MB"
    )
    if math.isfinite(float(metrics["translation_err_mm"])):
        print(
            "   "
            f"Translation error: {float(metrics['translation_err_mm']):.6f} mm  "
            f"Rotation error: {float(metrics['rotation_err_deg']):.6f} deg"
        )


def _record_planner_run(
    *,
    impl: str,
    num_waypoints: int,
    motion_generator: MotionGenerator,
    target_states: list[PlanState],
    options: MotionGenOptions,
    robot: Robot,
    last_waypoint: torch.Tensor,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
) -> None:
    elapsed, mem_deltas, peak_gpu, result = _timed_generate(
        motion_generator,
        target_states,
        options,
    )
    metrics = _compute_result_metrics(result, last_waypoint, robot)
    _print_run_summary(
        impl,
        num_waypoints,
        elapsed,
        mem_deltas,
        peak_gpu,
        metrics,
    )
    _append_result_rows(
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        impl=impl,
        num_waypoints=num_waypoints,
        elapsed_s=elapsed,
        mem_deltas=mem_deltas,
        peak_gpu_mb=peak_gpu,
        metrics=metrics,
    )


def _init_toppra_motion_generator(
    robot: Robot,
    start_qpos: torch.Tensor,
    notes: list[str],
) -> tuple[MotionGenerator | None, MotionGenOptions | None]:
    try:
        return (
            MotionGenerator(
                cfg=MotionGenCfg(
                    planner_cfg=ToppraPlannerCfg(
                        robot_uid=robot.uid,
                    )
                )
            ),
            _toppra_motion_options(start_qpos),
        )
    except Exception as exc:
        notes.append(
            f"Toppra comparison skipped: {exc}. "
            "Install with `pip install toppra==0.6.3`."
        )
        print(f"Toppra comparison skipped: {exc}")
        return None, None


def benchmark_neural_motion_generator(
    num_waypoints_list: list[int],
    sim_device: str,
    headless: bool,
    checkpoint_path: str | None,
    compare_toppra: bool,
) -> (
    tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
        list[str],
    ]
    | None
):
    resolved_checkpoint = _resolve_checkpoint(checkpoint_path)
    if resolved_checkpoint is None:
        return None

    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    notes = [
        "Scene matches examples/sim/planners/neural_planner.py (Franka Panda, main_arm).",
        "Each configuration performs one warmup generate() call before the timed run.",
        "Pose error compares the final planned EE pose against the last target waypoint.",
        "Toppra pose error uses FK on the final joint configuration when xpos_list is absent.",
        "NeuralPlanner success uses the checkpoint reach threshold (~50 mm position); "
        "reported pose error is a strict FK comparison against the last waypoint.",
    ]

    print("\n=== Neural Motion Generator Benchmark ===")
    print(f"Device: {sim_device}")
    print(f"Checkpoint: {resolved_checkpoint}")
    print(
        "num_waypoints values: "
        f"{', '.join(str(value) for value in num_waypoints_list)}"
    )

    sim, robot, start_qpos, start_pose = _setup_sim_and_robot(sim_device, headless)

    neural_motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid=robot.uid,
                checkpoint_path=resolved_checkpoint,
                control_part=ARM_NAME,
            )
        )
    )
    neural_options = _neural_motion_options(start_qpos)

    toppra_motion_generator: MotionGenerator | None = None
    toppra_options: MotionGenOptions | None = None
    if compare_toppra:
        toppra_motion_generator, toppra_options = _init_toppra_motion_generator(
            robot,
            start_qpos,
            notes,
        )

    for num_waypoints in num_waypoints_list:
        waypoints = _make_waypoints(start_pose, num_waypoints)
        target_states = _make_target_states(waypoints)
        last_waypoint = waypoints[-1]

        _record_planner_run(
            impl="neural_planner",
            num_waypoints=num_waypoints,
            motion_generator=neural_motion_generator,
            target_states=target_states,
            options=neural_options,
            robot=robot,
            last_waypoint=last_waypoint,
            perf_rows=perf_rows,
            metric_rows=metric_rows,
        )

        if toppra_motion_generator is not None and toppra_options is not None:
            _record_planner_run(
                impl="toppra_planner",
                num_waypoints=num_waypoints,
                motion_generator=toppra_motion_generator,
                target_states=target_states,
                options=toppra_options,
                robot=robot,
                last_waypoint=last_waypoint,
                perf_rows=perf_rows,
                metric_rows=metric_rows,
            )

    return (
        perf_rows,
        metric_rows,
        _build_leaderboard_rows(metric_rows, perf_rows),
        notes,
    )


def run_all_benchmarks(
    num_waypoints_list: list[int] | None = None,
    sim_device: str = "auto",
    headless: bool = True,
    checkpoint_path: str | None = None,
    compare_toppra: bool = False,
) -> None:
    device = _resolve_device(sim_device)
    num_waypoints_list = num_waypoints_list or DEFAULT_NUM_WAYPOINTS

    print("=" * 60)
    print("Neural Motion Generator Performance Benchmarks")
    print("=" * 60)

    result = benchmark_neural_motion_generator(
        num_waypoints_list=num_waypoints_list,
        sim_device=device,
        headless=headless,
        checkpoint_path=checkpoint_path,
        compare_toppra=compare_toppra,
    )
    if result is None:
        print("SKIPPED: neural motion generator benchmark (checkpoint unavailable).")
        sys.exit(0)

    perf_rows, metric_rows, leaderboard_rows, notes = result

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)

    report_path = _write_markdown_report(
        benchmark_name="neural_motion_generator",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            "CPU/GPU memory fields are deltas measured around timed calls.",
        ]
        + notes,
    )
    print(f"Markdown report saved: {report_path}")


if __name__ == "__main__":
    cli_args = _parse_args()
    run_all_benchmarks(
        num_waypoints_list=cli_args.num_waypoints,
        sim_device=cli_args.device,
        headless=cli_args.headless,
        checkpoint_path=cli_args.checkpoint_path,
        compare_toppra=cli_args.compare_toppra,
    )
