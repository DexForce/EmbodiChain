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

"""Benchmark NeuralPlanner in isolation on Franka Panda.

What this measures
    Planning latency, memory, rollout steps, and final TCP pose error for
    ``NeuralPlanner`` on fixed demo EEF waypoint sets.

What this does not measure
    Atomic-action task success, grasp physics, or obstacle avoidance.

Default behavior
    Only ``neural_planner`` is benchmarked. Use ``--compare-ik`` or
    ``--compare-toppra`` to add optional baselines.

Checkpoints are loaded from the ``dexforce/neural_motion_generator`` HuggingFace
repository unless ``--checkpoint-path`` is provided.

Output
    Markdown report under ``outputs/benchmarks/neural_planner_*.md``.

Run::

    python -m scripts.benchmark planners-neural-planner
    python -m scripts.benchmark.planners.neural_planner.run_benchmark
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable
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
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance

DEFAULT_NUM_WAYPOINTS = [1, 3, 5]
DEFAULT_NUM_TRIALS = 8
DEFAULT_WARMUP_TRIALS = 1
DEFAULT_SAMPLE_INTERVAL = 20
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

IMPL_NEURAL = "neural_planner"
IMPL_IK = "ik_interpolate"
IMPL_TOPPRA = "ik_toppra"


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
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="Measured trials per (impl, num_waypoints) configuration.",
    )
    parser.add_argument(
        "--warmup-trials",
        type=int,
        default=DEFAULT_WARMUP_TRIALS,
        help="Warmup trials per configuration; excluded from summary aggregation.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=DEFAULT_SAMPLE_INTERVAL,
        help="Resampled trajectory length for ik_interpolate and ik_toppra.",
    )
    parser.add_argument(
        "--compare-ik",
        action="store_true",
        help="Also benchmark sequential IK plus joint interpolation.",
    )
    parser.add_argument(
        "--compare-toppra",
        action="store_true",
        help="Also benchmark EEF IK interpolation followed by TOPPRA.",
    )
    parser.add_argument(
        "--save-trial-details",
        action="store_true",
        help="Include per-trial rows in the markdown report.",
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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[index]


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


QUALITY_SUMMARY_COLUMNS = (
    "impl",
    "num_trials",
    "success_rate",
    "final_translation_err_mm_mean",
    "final_rotation_err_deg_mean",
    "mean_waypoint_pos_err_mm_mean",
    "max_waypoint_pos_err_mm_mean",
    "mean_waypoint_rot_err_deg_mean",
    "max_waypoint_rot_err_deg_mean",
)

PERFORMANCE_SUMMARY_COLUMNS = (
    "impl",
    "num_trials",
    "cost_time_ms_mean",
    "cost_time_ms_p95",
    "rollout_steps_mean",
    "cpu_delta_mb_mean",
    "gpu_delta_mb_mean",
    "peak_gpu_mb_mean",
    "peak_gpu_mb_max",
)

_IMPL_REPORT_ORDER = {
    IMPL_NEURAL: 0,
    IMPL_IK: 1,
    IMPL_TOPPRA: 2,
}


def _sort_summary_for_report(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Group summary rows by num_waypoints for side-by-side planner comparison."""
    return sorted(
        rows,
        key=lambda row: (
            int(row["num_waypoints"]),
            _IMPL_REPORT_ORDER.get(str(row["impl"]), 99),
            str(row["impl"]),
        ),
    )


def _group_summary_by_waypoints(
    rows: list[dict[str, object]],
) -> list[tuple[int, list[dict[str, object]]]]:
    """Return summary rows grouped and sorted by num_waypoints."""
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row["num_waypoints"])].append(row)

    grouped: list[tuple[int, list[dict[str, object]]]] = []
    for num_waypoints in sorted(groups):
        group_rows = sorted(
            groups[num_waypoints],
            key=lambda row: (
                _IMPL_REPORT_ORDER.get(str(row["impl"]), 99),
                str(row["impl"]),
            ),
        )
        grouped.append((num_waypoints, group_rows))
    return grouped


def _format_waypoint_grouped_tables(
    summary_rows: list[dict[str, object]],
    columns: tuple[str, ...],
) -> list[str]:
    """Render one markdown table per num_waypoints value."""
    grouped = _group_summary_by_waypoints(summary_rows)
    if not grouped:
        return ["No data."]

    lines: list[str] = []
    for index, (num_waypoints, group_rows) in enumerate(grouped):
        if index > 0:
            lines.append("")
        lines.extend(
            [
                f"### num_waypoints = {num_waypoints}",
                "",
            ]
        )
        lines.extend(_format_markdown_table(_project_table_rows(group_rows, columns)))
    return lines


def _project_table_rows(
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
) -> list[dict[str, object]]:
    return [{column: row[column] for column in columns} for row in rows]


def _write_markdown_report(
    benchmark_name: str,
    trial_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    quality_leaderboard_rows: list[dict[str, object]],
    performance_leaderboard_rows: list[dict[str, object]],
    notes: list[str] | None = None,
    *,
    include_trial_details: bool = False,
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
        "## Quality",
        "",
    ]
    lines.extend(_format_waypoint_grouped_tables(summary_rows, QUALITY_SUMMARY_COLUMNS))
    lines.extend(["", "## Performance", ""])
    lines.extend(
        _format_waypoint_grouped_tables(summary_rows, PERFORMANCE_SUMMARY_COLUMNS)
    )
    lines.extend(["", "## Leaderboard (Quality)", ""])
    lines.extend(_format_markdown_table(quality_leaderboard_rows))
    lines.extend(["", "## Leaderboard (Performance)", ""])
    lines.extend(_format_markdown_table(performance_leaderboard_rows))
    if include_trial_details:
        lines.extend(["", "## Trial Details", ""])
        lines.extend(_format_markdown_table(trial_rows))

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
            "Neural planner benchmark skipped: checkpoint unavailable.\n"
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


def _neural_plan_options(start_qpos: torch.Tensor) -> MotionGenOptions:
    return MotionGenOptions(
        control_part=ARM_NAME,
        start_qpos=start_qpos,
        plan_opts=NeuralPlanOptions(
            control_part=ARM_NAME,
            start_qpos=start_qpos,
        ),
    )


def _toppra_motion_options(
    start_qpos: torch.Tensor,
    sample_interval: int,
) -> MotionGenOptions:
    # EEF waypoints are IK-interpolated inside MotionGenerator, then TOPPRA
    # time-parameterizes the resulting joint path. This differs from the atomic
    # action default arm path (sequential IK + joint interpolation only).
    return MotionGenOptions(
        control_part=ARM_NAME,
        start_qpos=start_qpos,
        is_interpolate=True,
        is_linear=True,
        plan_opts=ToppraPlanOptions(
            constraints={"velocity": 0.2, "acceleration": 0.5},
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=sample_interval,
        ),
    )


def _all_success(success: bool | torch.Tensor) -> bool:
    if isinstance(success, torch.Tensor):
        return bool(torch.all(success).item())
    return bool(success)


def plan_ik_interpolate(
    robot: Robot,
    waypoints: torch.Tensor,
    start_qpos: torch.Tensor,
    sample_interval: int,
) -> PlanResult:
    """Plan via sequential IK followed by joint-space interpolation."""
    qpos_seed = start_qpos.unsqueeze(0)
    joint_targets = [start_qpos]
    for waypoint in waypoints:
        success, qpos = robot.compute_ik(
            pose=waypoint.unsqueeze(0),
            name=ARM_NAME,
            joint_seed=qpos_seed,
        )
        if not _all_success(success):
            return PlanResult(success=False, positions=None)
        qpos_seed = qpos
        joint_targets.append(qpos.squeeze(0))

    trajectory = torch.stack(joint_targets, dim=0).unsqueeze(0)
    positions = interpolate_with_distance(
        trajectory=trajectory,
        interp_num=sample_interval,
        device=robot.device,
    ).squeeze(0)
    return PlanResult(
        success=True,
        positions=positions,
        duration=0.0,
    )


def _trajectory_fk_poses(result: PlanResult, robot: Robot) -> list[torch.Tensor]:
    """Return TCP poses sampled along the planned trajectory."""
    if result.xpos_list is not None and result.xpos_list.shape[0] > 0:
        return [pose for pose in result.xpos_list]
    if result.positions is None or result.positions.shape[0] == 0:
        return []
    qpos = result.positions
    if qpos.dim() == 1:
        qpos = qpos.unsqueeze(0)
    fk = robot.compute_batch_fk(
        qpos=qpos.unsqueeze(0),
        name=ARM_NAME,
        to_matrix=True,
    ).squeeze(0)
    return [fk[i] for i in range(fk.shape[0])]


def compute_waypoint_errors(
    trajectory_poses: list[torch.Tensor],
    waypoints: torch.Tensor,
) -> dict[str, float]:
    """Compute best-hit pose errors for each target waypoint along a trajectory.

    For every target waypoint, scan all trajectory TCP samples and record the
    smallest translation/rotation error. Return the mean and max across waypoints.
    """
    empty = {
        "mean_waypoint_pos_err_mm": float("inf"),
        "max_waypoint_pos_err_mm": float("inf"),
        "mean_waypoint_rot_err_deg": float("inf"),
        "max_waypoint_rot_err_deg": float("inf"),
    }
    if not trajectory_poses or waypoints.numel() == 0:
        return empty

    waypoint_pos_errors_mm: list[float] = []
    waypoint_rot_errors_deg: list[float] = []
    for waypoint in waypoints:
        best_pos_m = float("inf")
        best_rot_rad = float("inf")
        for pose in trajectory_poses:
            pos_m, rot_rad = get_pose_err(pose, waypoint)
            best_pos_m = min(best_pos_m, pos_m)
            best_rot_rad = min(best_rot_rad, rot_rad)
        waypoint_pos_errors_mm.append(best_pos_m * 1000.0)
        waypoint_rot_errors_deg.append(best_rot_rad * 180.0 / math.pi)

    return {
        "mean_waypoint_pos_err_mm": sum(waypoint_pos_errors_mm)
        / len(waypoint_pos_errors_mm),
        "max_waypoint_pos_err_mm": max(waypoint_pos_errors_mm),
        "mean_waypoint_rot_err_deg": sum(waypoint_rot_errors_deg)
        / len(waypoint_rot_errors_deg),
        "max_waypoint_rot_err_deg": max(waypoint_rot_errors_deg),
    }


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
    waypoints: torch.Tensor,
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

    trajectory_poses = _trajectory_fk_poses(result, robot)
    waypoint_errors = compute_waypoint_errors(trajectory_poses, waypoints)

    final_pose = _final_eef_pose(result, robot)
    last_waypoint = waypoints[-1]
    if final_pose is not None:
        t_err_m, r_err_rad = get_pose_err(final_pose, last_waypoint)
        translation_err_mm = t_err_m * 1000.0
        rotation_err_deg = r_err_rad * 180.0 / math.pi
    else:
        translation_err_mm = float("inf")
        rotation_err_deg = float("inf")

    return {
        "success": success,
        "translation_err_mm": translation_err_mm,
        "rotation_err_deg": rotation_err_deg,
        "mean_waypoint_pos_err_mm": waypoint_errors["mean_waypoint_pos_err_mm"],
        "max_waypoint_pos_err_mm": waypoint_errors["max_waypoint_pos_err_mm"],
        "mean_waypoint_rot_err_deg": waypoint_errors["mean_waypoint_rot_err_deg"],
        "max_waypoint_rot_err_deg": waypoint_errors["max_waypoint_rot_err_deg"],
        "rollout_steps": rollout_steps,
        "duration_s": duration_s,
    }


def _timed_plan(
    plan_fn: Callable[[], PlanResult],
) -> tuple[float, dict[str, float], float, PlanResult]:
    _reset_peak_gpu_memory()
    mem_before = _memory_snapshot()
    _sync_cuda()

    start = time.perf_counter()
    result = plan_fn()
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


def _append_trial_row(
    trial_rows: list[dict[str, object]],
    *,
    impl: str,
    num_waypoints: int,
    trial_id: int,
    warmup: bool,
    elapsed_s: float,
    mem_deltas: dict[str, float],
    peak_gpu_mb: float,
    metrics: dict[str, object],
) -> None:
    trial_rows.append(
        {
            "impl": impl,
            "num_waypoints": num_waypoints,
            "trial_id": trial_id,
            "warmup": warmup,
            "cost_time_ms": f"{elapsed_s * 1000.0:.6f}",
            "cpu_delta_mb": f"{mem_deltas['cpu_mb']:.6f}",
            "gpu_delta_mb": f"{mem_deltas['gpu_mb']:.6f}",
            "peak_gpu_mb": f"{peak_gpu_mb:.6f}",
            "success": metrics["success"],
            "final_translation_err_mm": _format_finite(metrics["translation_err_mm"]),
            "final_rotation_err_deg": _format_finite(metrics["rotation_err_deg"]),
            "mean_waypoint_pos_err_mm": _format_finite(
                metrics["mean_waypoint_pos_err_mm"]
            ),
            "max_waypoint_pos_err_mm": _format_finite(
                metrics["max_waypoint_pos_err_mm"]
            ),
            "mean_waypoint_rot_err_deg": _format_finite(
                metrics["mean_waypoint_rot_err_deg"]
            ),
            "max_waypoint_rot_err_deg": _format_finite(
                metrics["max_waypoint_rot_err_deg"]
            ),
            "rollout_steps": metrics["rollout_steps"],
            "duration_s": f"{float(metrics['duration_s']):.6f}",
        }
    )


def _aggregate_rows(
    trial_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate measured trials by (impl, num_waypoints)."""
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        if bool(row["warmup"]):
            continue
        key = (str(row["impl"]), int(row["num_waypoints"]))
        groups[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for (impl, num_waypoints), rows in sorted(groups.items()):
        costs = [float(row["cost_time_ms"]) for row in rows]
        successes = [bool(row["success"]) for row in rows]
        rollout_steps = [int(row["rollout_steps"]) for row in rows]
        t_errs = [
            float(row["final_translation_err_mm"])
            for row in rows
            if math.isfinite(float(row["final_translation_err_mm"]))
        ]
        r_errs = [
            float(row["final_rotation_err_deg"])
            for row in rows
            if math.isfinite(float(row["final_rotation_err_deg"]))
        ]
        mean_wp_pos = [
            float(row["mean_waypoint_pos_err_mm"])
            for row in rows
            if math.isfinite(float(row["mean_waypoint_pos_err_mm"]))
        ]
        max_wp_pos = [
            float(row["max_waypoint_pos_err_mm"])
            for row in rows
            if math.isfinite(float(row["max_waypoint_pos_err_mm"]))
        ]
        mean_wp_rot = [
            float(row["mean_waypoint_rot_err_deg"])
            for row in rows
            if math.isfinite(float(row["mean_waypoint_rot_err_deg"]))
        ]
        max_wp_rot = [
            float(row["max_waypoint_rot_err_deg"])
            for row in rows
            if math.isfinite(float(row["max_waypoint_rot_err_deg"]))
        ]
        cpu_deltas = [float(row["cpu_delta_mb"]) for row in rows]
        gpu_deltas = [float(row["gpu_delta_mb"]) for row in rows]
        peak_gpus = [float(row["peak_gpu_mb"]) for row in rows]

        summary_rows.append(
            {
                "impl": impl,
                "num_waypoints": num_waypoints,
                "num_trials": len(rows),
                "success_rate": f"{sum(successes) / max(len(rows), 1):.2%}",
                "cost_time_ms_mean": f"{sum(costs) / len(costs):.6f}",
                "cost_time_ms_p95": f"{_percentile(costs, 95.0):.6f}",
                "rollout_steps_mean": f"{sum(rollout_steps) / len(rollout_steps):.2f}",
                "cpu_delta_mb_mean": f"{sum(cpu_deltas) / len(cpu_deltas):.6f}",
                "gpu_delta_mb_mean": f"{sum(gpu_deltas) / len(gpu_deltas):.6f}",
                "peak_gpu_mb_mean": f"{sum(peak_gpus) / len(peak_gpus):.6f}",
                "peak_gpu_mb_max": f"{max(peak_gpus):.6f}",
                "final_translation_err_mm_mean": (
                    f"{sum(t_errs) / len(t_errs):.6f}" if t_errs else "inf"
                ),
                "final_rotation_err_deg_mean": (
                    f"{sum(r_errs) / len(r_errs):.6f}" if r_errs else "inf"
                ),
                "mean_waypoint_pos_err_mm_mean": (
                    f"{sum(mean_wp_pos) / len(mean_wp_pos):.6f}"
                    if mean_wp_pos
                    else "inf"
                ),
                "max_waypoint_pos_err_mm_mean": (
                    f"{sum(max_wp_pos) / len(max_wp_pos):.6f}" if max_wp_pos else "inf"
                ),
                "mean_waypoint_rot_err_deg_mean": (
                    f"{sum(mean_wp_rot) / len(mean_wp_rot):.6f}"
                    if mean_wp_rot
                    else "inf"
                ),
                "max_waypoint_rot_err_deg_mean": (
                    f"{sum(max_wp_rot) / len(max_wp_rot):.6f}" if max_wp_rot else "inf"
                ),
            }
        )
    return _sort_summary_for_report(summary_rows)


def _group_summary_by_impl(
    summary_rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    by_impl: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_impl[str(row["impl"])].append(row)
    return by_impl


def _build_quality_leaderboard_rows(
    summary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Rank planners by mean success rate, then lower waypoint error."""
    leaderboard: list[tuple[float, float, str, dict[str, object]]] = []
    for impl, rows in _group_summary_by_impl(summary_rows).items():
        success_rates = [
            float(str(row["success_rate"]).strip("%")) / 100.0 for row in rows
        ]
        t_errs = [
            float(row["final_translation_err_mm_mean"])
            for row in rows
            if math.isfinite(float(row["final_translation_err_mm_mean"]))
        ]
        r_errs = [
            float(row["final_rotation_err_deg_mean"])
            for row in rows
            if math.isfinite(float(row["final_rotation_err_deg_mean"]))
        ]
        wp_pos = [
            float(row["mean_waypoint_pos_err_mm_mean"])
            for row in rows
            if math.isfinite(float(row["mean_waypoint_pos_err_mm_mean"]))
        ]
        overall_success = sum(success_rates) / len(success_rates)
        avg_wp_pos = sum(wp_pos) / len(wp_pos) if wp_pos else math.inf
        leaderboard.append(
            (
                overall_success,
                -avg_wp_pos,
                impl,
                {
                    "overall_success_rate": f"{overall_success:.2%}",
                    "avg_final_translation_err_mm": (
                        f"{sum(t_errs) / len(t_errs):.6f}" if t_errs else "inf"
                    ),
                    "avg_final_rotation_err_deg": (
                        f"{sum(r_errs) / len(r_errs):.6f}" if r_errs else "inf"
                    ),
                    "avg_mean_waypoint_pos_err_mm": (
                        f"{sum(wp_pos) / len(wp_pos):.6f}" if wp_pos else "inf"
                    ),
                },
            )
        )

    ranked = sorted(leaderboard, key=lambda item: (item[0], item[1]), reverse=True)
    return [
        {"rank": rank, "algorithm": impl, **stats}
        for rank, (_, _, impl, stats) in enumerate(ranked, start=1)
    ]


def _build_performance_leaderboard_rows(
    summary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Rank planners by lower mean planning latency."""
    leaderboard: list[tuple[float, str, dict[str, object]]] = []
    for impl, rows in _group_summary_by_impl(summary_rows).items():
        costs = [float(row["cost_time_ms_mean"]) for row in rows]
        p95_costs = [float(row["cost_time_ms_p95"]) for row in rows]
        cpu_deltas = [float(row["cpu_delta_mb_mean"]) for row in rows]
        gpu_deltas = [float(row["gpu_delta_mb_mean"]) for row in rows]
        peak_gpus = [float(row["peak_gpu_mb_mean"]) for row in rows]
        avg_cost = sum(costs) / len(costs)
        leaderboard.append(
            (
                -avg_cost,
                impl,
                {
                    "avg_cost_time_ms": f"{avg_cost:.6f}",
                    "p95_cost_time_ms": f"{sum(p95_costs) / len(p95_costs):.6f}",
                    "avg_cpu_delta_mb": f"{sum(cpu_deltas) / len(cpu_deltas):.6f}",
                    "avg_gpu_delta_mb": f"{sum(gpu_deltas) / len(gpu_deltas):.6f}",
                    "avg_peak_gpu_mb": f"{sum(peak_gpus) / len(peak_gpus):.6f}",
                },
            )
        )

    ranked = sorted(leaderboard, key=lambda item: item[0], reverse=True)
    return [
        {"rank": rank, "algorithm": impl, **stats}
        for rank, (_, impl, stats) in enumerate(ranked, start=1)
    ]


def _print_run_summary(
    impl: str,
    num_waypoints: int,
    trial_id: int,
    warmup: bool,
    elapsed_s: float,
    mem_deltas: dict[str, float],
    peak_gpu_mb: float,
    metrics: dict[str, object],
) -> None:
    label = "warmup" if warmup else f"trial={trial_id}"
    print(f"**** {impl} num_waypoints={num_waypoints} {label}")
    print(
        f"===Plan time: {elapsed_s * 1000.0:.6f} ms  "
        f"success={metrics['success']}  "
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
            f"Final waypoint error: {float(metrics['translation_err_mm']):.6f} mm  "
            f"{float(metrics['rotation_err_deg']):.6f} deg"
        )
    if math.isfinite(float(metrics["mean_waypoint_pos_err_mm"])):
        print(
            "   "
            f"Mean waypoint error: {float(metrics['mean_waypoint_pos_err_mm']):.6f} mm  "
            f"{float(metrics['mean_waypoint_rot_err_deg']):.6f} deg  "
            f"(max {float(metrics['max_waypoint_pos_err_mm']):.6f} mm / "
            f"{float(metrics['max_waypoint_rot_err_deg']):.6f} deg)"
        )


def _run_impl_trials(
    *,
    impl: str,
    num_waypoints: int,
    num_trials: int,
    warmup_trials: int,
    plan_fn: Callable[[], PlanResult],
    robot: Robot,
    waypoints: torch.Tensor,
    trial_rows: list[dict[str, object]],
) -> None:
    measured_trial_id = 0
    for trial_idx in range(warmup_trials + num_trials):
        warmup = trial_idx < warmup_trials
        elapsed, mem_deltas, peak_gpu, result = _timed_plan(plan_fn)
        metrics = _compute_result_metrics(result, waypoints, robot)
        if not warmup:
            _print_run_summary(
                impl,
                num_waypoints,
                measured_trial_id,
                warmup,
                elapsed,
                mem_deltas,
                peak_gpu,
                metrics,
            )
            measured_trial_id += 1
        _append_trial_row(
            trial_rows,
            impl=impl,
            num_waypoints=num_waypoints,
            trial_id=trial_idx,
            warmup=warmup,
            elapsed_s=elapsed,
            mem_deltas=mem_deltas,
            peak_gpu_mb=peak_gpu,
            metrics=metrics,
        )


def _init_toppra_motion_generator(
    robot: Robot,
    start_qpos: torch.Tensor,
    sample_interval: int,
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
            _toppra_motion_options(start_qpos, sample_interval),
        )
    except Exception as exc:
        notes.append(
            f"Toppra comparison skipped: {exc}. "
            "Install with `pip install toppra==0.6.3`."
        )
        print(f"Toppra comparison skipped: {exc}")
        return None, None


def _benchmark_notes(
    *,
    sim_device: str,
    checkpoint_path: str,
    num_trials: int,
    warmup_trials: int,
    sample_interval: int,
    compare_ik: bool,
    compare_toppra: bool,
) -> list[str]:
    impls = [IMPL_NEURAL]
    if compare_ik:
        impls.append(IMPL_IK)
    if compare_toppra:
        impls.append(IMPL_TOPPRA)

    checkpoint_name = Path(checkpoint_path).name
    return [
        f"Device: {sim_device} | Robot: Franka Panda ({ARM_NAME})",
        f"Checkpoint: {checkpoint_name} ({checkpoint_path})",
        f"Trials: {warmup_trials} warmup + {num_trials} measured per "
        f"(impl, num_waypoints); sample_interval={sample_interval}",
        f"Planners: {', '.join(impls)}",
        "success_rate follows each planner; pose errors are FK vs target waypoints.",
    ]


def benchmark_neural_planner(
    num_waypoints_list: list[int],
    sim_device: str,
    headless: bool,
    checkpoint_path: str | None,
    *,
    num_trials: int = DEFAULT_NUM_TRIALS,
    warmup_trials: int = DEFAULT_WARMUP_TRIALS,
    sample_interval: int = DEFAULT_SAMPLE_INTERVAL,
    compare_ik: bool = False,
    compare_toppra: bool = False,
) -> (
    tuple[
        list[dict[str, object]],
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

    if num_trials < 1:
        raise ValueError("--num-trials must be >= 1.")
    if warmup_trials < 0:
        raise ValueError("--warmup-trials must be >= 0.")
    if sample_interval < 1:
        raise ValueError("--sample-interval must be >= 1.")

    trial_rows: list[dict[str, object]] = []
    notes = _benchmark_notes(
        sim_device=sim_device,
        checkpoint_path=resolved_checkpoint,
        num_trials=num_trials,
        warmup_trials=warmup_trials,
        sample_interval=sample_interval,
        compare_ik=compare_ik,
        compare_toppra=compare_toppra,
    )

    print("\n=== NeuralPlanner Benchmark ===")
    print(f"Device: {sim_device}")
    print(f"Checkpoint: {resolved_checkpoint}")
    print(
        "num_waypoints values: "
        f"{', '.join(str(value) for value in num_waypoints_list)}"
    )
    print(f"num_trials={num_trials} warmup_trials={warmup_trials}")

    _, robot, start_qpos, start_pose = _setup_sim_and_robot(sim_device, headless)

    neural_planner = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid=robot.uid,
                checkpoint_path=resolved_checkpoint,
                control_part=ARM_NAME,
            )
        )
    )
    neural_options = _neural_plan_options(start_qpos)

    toppra_motion_generator: MotionGenerator | None = None
    toppra_options: MotionGenOptions | None = None
    if compare_toppra:
        toppra_motion_generator, toppra_options = _init_toppra_motion_generator(
            robot,
            start_qpos,
            sample_interval,
            notes,
        )

    for num_waypoints in num_waypoints_list:
        waypoints = _make_waypoints(start_pose, num_waypoints)
        target_states = _make_target_states(waypoints)

        _run_impl_trials(
            impl=IMPL_NEURAL,
            num_waypoints=num_waypoints,
            num_trials=num_trials,
            warmup_trials=warmup_trials,
            plan_fn=lambda ts=target_states, opts=neural_options: neural_planner.generate(  # noqa: E501
                target_states=ts,
                options=opts,
            ),
            robot=robot,
            waypoints=waypoints,
            trial_rows=trial_rows,
        )

        if compare_ik:
            _run_impl_trials(
                impl=IMPL_IK,
                num_waypoints=num_waypoints,
                num_trials=num_trials,
                warmup_trials=warmup_trials,
                plan_fn=lambda wp=waypoints, sq=start_qpos, si=sample_interval: plan_ik_interpolate(  # noqa: E501
                    robot,
                    wp,
                    sq,
                    si,
                ),
                robot=robot,
                waypoints=waypoints,
                trial_rows=trial_rows,
            )

        if toppra_motion_generator is not None and toppra_options is not None:
            _run_impl_trials(
                impl=IMPL_TOPPRA,
                num_waypoints=num_waypoints,
                num_trials=num_trials,
                warmup_trials=warmup_trials,
                plan_fn=lambda ts=target_states, opts=toppra_options: toppra_motion_generator.generate(  # noqa: E501
                    target_states=ts,
                    options=opts,
                ),
                robot=robot,
                waypoints=waypoints,
                trial_rows=trial_rows,
            )

    summary_rows = _aggregate_rows(trial_rows)
    return (
        trial_rows,
        summary_rows,
        _build_quality_leaderboard_rows(summary_rows),
        _build_performance_leaderboard_rows(summary_rows),
        notes,
    )


def run_all_benchmarks(
    num_waypoints_list: list[int] | None = None,
    sim_device: str = "auto",
    headless: bool = True,
    checkpoint_path: str | None = None,
    *,
    num_trials: int = DEFAULT_NUM_TRIALS,
    warmup_trials: int = DEFAULT_WARMUP_TRIALS,
    sample_interval: int = DEFAULT_SAMPLE_INTERVAL,
    compare_ik: bool = False,
    compare_toppra: bool = False,
    include_trial_details: bool = False,
) -> None:
    device = _resolve_device(sim_device)
    num_waypoints_list = num_waypoints_list or DEFAULT_NUM_WAYPOINTS

    print("=" * 60)
    print("NeuralPlanner Performance Benchmarks")
    print("=" * 60)

    result = benchmark_neural_planner(
        num_waypoints_list=num_waypoints_list,
        sim_device=device,
        headless=headless,
        checkpoint_path=checkpoint_path,
        num_trials=num_trials,
        warmup_trials=warmup_trials,
        sample_interval=sample_interval,
        compare_ik=compare_ik,
        compare_toppra=compare_toppra,
    )
    if result is None:
        print("SKIPPED: neural planner benchmark (checkpoint unavailable).")
        sys.exit(0)

    (
        trial_rows,
        summary_rows,
        quality_leaderboard_rows,
        performance_leaderboard_rows,
        notes,
    ) = result

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)

    report_path = _write_markdown_report(
        benchmark_name="neural_planner",
        trial_rows=trial_rows,
        summary_rows=summary_rows,
        quality_leaderboard_rows=quality_leaderboard_rows,
        performance_leaderboard_rows=performance_leaderboard_rows,
        notes=notes,
        include_trial_details=include_trial_details,
    )
    print(f"Markdown report saved: {report_path}")


if __name__ == "__main__":
    cli_args = _parse_args()
    run_all_benchmarks(
        num_waypoints_list=cli_args.num_waypoints,
        sim_device=cli_args.device,
        headless=cli_args.headless,
        checkpoint_path=cli_args.checkpoint_path,
        num_trials=cli_args.num_trials,
        warmup_trials=cli_args.warmup_trials,
        sample_interval=cli_args.sample_interval,
        compare_ik=cli_args.compare_ik,
        compare_toppra=cli_args.compare_toppra,
        include_trial_details=cli_args.save_trial_details,
    )
