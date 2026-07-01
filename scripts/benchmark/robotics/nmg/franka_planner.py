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

"""Benchmark Franka end-effector waypoint planning backends.

This benchmark isolates planner quality from grasp annotation and physics. It
supports a demo-matched waypoint source that mirrors
examples/sim/planners/neural_planner.py, and a broader reachable-FK source that
samples target poses from known Franka joint configurations. The downstream
Franka pick-place benchmark remains the third integration layer.
Run: python -m scripts.benchmark.robotics.nmg.franka_planner
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import torch

from embodichain.data.assets.planner_assets import download_neural_planner_checkpoint
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.planners import (
    MoveType,
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    NeuralPlanOptions,
    NeuralPlannerCfg,
    PlanResult,
    PlanState,
    ToppraPlanOptions,
    ToppraPlannerCfg,
)
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix
from scripts.benchmark.robotics.nmg.franka_pick_place import (
    ARM_NAME,
    FRANKA_REST_QPOS,
    FRANKA_START_QPOS,
    SCRIPT_NAME as PICK_PLACE_SCRIPT_NAME,
    create_franka,
)

SCRIPT_NAME = "franka_planner_nmg"
DEFAULT_SEED = 0
DEFAULT_NUM_TRIALS = 8
DEFAULT_WARMUP_TRIALS = 1
DEFAULT_NUM_WAYPOINTS = 3
DEFAULT_SAMPLE_INTERVAL = 120
DEFAULT_POS_THRESHOLD = 1e-3
DEFAULT_ROT_THRESHOLD = 0.05
DEFAULT_NMG_POS_THRESHOLD = 0.05
DEFAULT_NMG_ROT_THRESHOLD = 0.3
PLANNER_NAMES = ("ik_toppra", "neural", "neural_refine")
TRIAL_SOURCE_NAMES = ("demo_offsets", "fk_bank")
DEFAULT_TRIAL_SOURCE = "fk_bank"
DEMO_WAYPOINT_OFFSETS = (
    (0.10, 0.00, 0.00),
    (0.10, 0.10, 0.00),
    (0.00, 0.10, -0.08),
    (-0.10, 0.10, -0.08),
    (-0.10, 0.00, 0.00),
    (0.00, -0.10, 0.00),
    (0.10, -0.10, -0.06),
    (0.00, 0.00, -0.12),
)


@dataclass(frozen=True)
class PlannerTrial:
    """A reachable waypoint planning trial."""

    trial_id: int
    trial_source: str
    start_qpos: torch.Tensor
    target_qpos: torch.Tensor
    waypoints: list[torch.Tensor]


@dataclass(frozen=True)
class PlannerOutcome:
    """Planner result with normalized trajectory fields."""

    action_success: bool
    positions: torch.Tensor | None
    planning_time_sec: float
    cpu_delta_mb: float
    gpu_delta_mb: float
    peak_gpu_mb: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Franka EEF waypoint planners without grasp physics."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--planner",
        choices=[*PLANNER_NAMES, "all"],
        default="all",
        help="Planner backend to evaluate.",
    )
    parser.add_argument(
        "--neural_checkpoint",
        type=str,
        default=None,
        help="Local Franka NMG checkpoint. Required for neural planners unless downloadable.",
    )
    parser.add_argument(
        "--trial_source",
        choices=[*TRIAL_SOURCE_NAMES, "all"],
        default=DEFAULT_TRIAL_SOURCE,
        help=(
            "Waypoint source. 'demo_offsets' mirrors examples/sim/planners/"
            "neural_planner.py, 'fk_bank' uses broader FK-generated targets, "
            "and 'all' runs both planner layers."
        ),
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="Measured trials per planner.",
    )
    parser.add_argument(
        "--warmup_trials",
        type=int,
        default=DEFAULT_WARMUP_TRIALS,
        help="Warmup trials per planner; excluded from summaries.",
    )
    parser.add_argument(
        "--num_waypoints",
        type=int,
        default=DEFAULT_NUM_WAYPOINTS,
        help="Number of EEF waypoints per trial.",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=DEFAULT_SAMPLE_INTERVAL,
        help="Target trajectory samples for IK+TOPPRA and resampled neural paths.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for waypoint generation.",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default=None,
        help="Optional Markdown report path. Defaults to outputs/benchmarks/.",
    )
    parser.add_argument(
        "--save_raw_jsonl",
        type=str,
        default=None,
        help="Optional JSONL path for raw per-trial rows.",
    )
    parser.add_argument(
        "--pos_success_threshold",
        type=float,
        default=DEFAULT_POS_THRESHOLD,
        help="Strict final TCP position threshold in meters.",
    )
    parser.add_argument(
        "--rot_success_threshold",
        type=float,
        default=DEFAULT_ROT_THRESHOLD,
        help="Strict final TCP rotation threshold in radians.",
    )
    parser.add_argument(
        "--nmg_pos_success_threshold",
        type=float,
        default=DEFAULT_NMG_POS_THRESHOLD,
        help="Loose position threshold passed to NMG planners.",
    )
    parser.add_argument(
        "--nmg_rot_success_threshold",
        type=float,
        default=DEFAULT_NMG_ROT_THRESHOLD,
        help="Loose rotation threshold passed to NMG planners.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate benchmark arguments."""
    if args.num_trials < 1:
        raise ValueError("--num_trials must be >= 1.")
    if args.warmup_trials < 0:
        raise ValueError("--warmup_trials must be >= 0.")
    if args.num_waypoints < 1:
        raise ValueError("--num_waypoints must be >= 1.")
    if args.sample_interval < 2:
        raise ValueError("--sample_interval must be >= 2.")


def expand_planner_selection(planner: str) -> list[str]:
    """Expand planner aliases into concrete planner names."""
    if planner == "all":
        return list(PLANNER_NAMES)
    return [planner]


def expand_trial_source_selection(trial_source: str) -> list[str]:
    """Expand trial-source aliases into concrete source names."""
    if trial_source == "all":
        return list(TRIAL_SOURCE_NAMES)
    return [trial_source]


def simulation_requires_cuda(args: argparse.Namespace) -> bool:
    """Return whether the selected simulation settings require CUDA."""
    if str(args.device).startswith("cuda"):
        return True
    return args.renderer in ("hybrid", "fast-rt", "rt")


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
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    cpu_mb = process.memory_info().rss / 1024**2
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}


def make_sim(args: argparse.Namespace) -> SimulationManager:
    """Create a minimal simulation manager for Franka kinematics."""
    return SimulationManager(
        SimulationManagerCfg(
            width=640,
            height=480,
            headless=True,
            sim_device=args.device,
            gpu_id=args.gpu_id,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_dt=1.0 / 100.0,
            num_envs=1,
            arena_space=args.arena_space,
        )
    )


def resolve_checkpoint(args: argparse.Namespace, planners: list[str]) -> str | None:
    """Resolve the NMG checkpoint path if a neural planner is selected."""
    if not any(planner in ("neural", "neural_refine") for planner in planners):
        return None
    if args.neural_checkpoint:
        return args.neural_checkpoint
    return download_neural_planner_checkpoint()


def build_motion_generator(
    robot: Robot,
    planner_name: str,
    checkpoint_path: str | None,
    args: argparse.Namespace,
) -> MotionGenerator:
    """Create a motion generator for a planner backend."""
    if planner_name == "ik_toppra":
        planner_cfg = ToppraPlannerCfg(robot_uid=robot.uid)
    elif planner_name in ("neural", "neural_refine"):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for neural planners.")
        planner_cfg = NeuralPlannerCfg(
            robot_uid=robot.uid,
            planner_type=planner_name,
            checkpoint_path=checkpoint_path,
            control_part=ARM_NAME,
            num_arm_joints=len(robot.get_joint_ids(ARM_NAME)),
            pos_eps=args.nmg_pos_success_threshold,
            rot_eps=args.nmg_rot_success_threshold,
        )
    else:
        raise ValueError(f"Unsupported planner: {planner_name}")
    return MotionGenerator(cfg=MotionGenCfg(planner_cfg=planner_cfg))


def _franka_lower_upper(robot: Robot) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Franka arm qpos lower and upper limits."""
    limits = robot.get_qpos_limits(name=ARM_NAME)[0].to(
        device=robot.device, dtype=torch.float32
    )
    return limits[:, 0], limits[:, 1]


def _qpos_from_ratio(
    lower: torch.Tensor, upper: torch.Tensor, ratio_values: tuple[float, ...]
) -> torch.Tensor:
    """Create a qpos inside joint limits from normalized ratios."""
    ratios = torch.tensor(ratio_values, dtype=torch.float32, device=lower.device)
    return lower + ratios * (upper - lower)


def make_seed_qpos_bank(robot: Robot) -> list[torch.Tensor]:
    """Create deterministic reachable qpos candidates for waypoint trials."""
    lower, upper = _franka_lower_upper(robot)
    nominal = [
        torch.tensor(FRANKA_START_QPOS, dtype=torch.float32, device=robot.device),
        torch.tensor(FRANKA_REST_QPOS, dtype=torch.float32, device=robot.device),
        _qpos_from_ratio(lower, upper, (0.50, 0.38, 0.55, 0.34, 0.52, 0.66, 0.60)),
        _qpos_from_ratio(lower, upper, (0.46, 0.44, 0.47, 0.40, 0.58, 0.61, 0.45)),
        _qpos_from_ratio(lower, upper, (0.55, 0.42, 0.43, 0.37, 0.47, 0.64, 0.53)),
        _qpos_from_ratio(lower, upper, (0.48, 0.50, 0.60, 0.36, 0.44, 0.58, 0.57)),
        _qpos_from_ratio(lower, upper, (0.53, 0.35, 0.50, 0.43, 0.61, 0.62, 0.49)),
        _qpos_from_ratio(lower, upper, (0.44, 0.47, 0.54, 0.39, 0.40, 0.69, 0.51)),
    ]
    return [qpos.clamp(lower, upper) for qpos in nominal]


def fk_pose(robot: Robot, qpos: torch.Tensor) -> torch.Tensor:
    """Compute an unbatched Franka TCP pose."""
    return robot.compute_fk(qpos=qpos.unsqueeze(0), name=ARM_NAME, to_matrix=True)[0]


def make_demo_offset_waypoints(
    start_pose: torch.Tensor, num_waypoints: int
) -> list[torch.Tensor]:
    """Create demo-matched compact TCP waypoints around the start pose."""
    offsets = torch.tensor(
        DEMO_WAYPOINT_OFFSETS,
        dtype=start_pose.dtype,
        device=start_pose.device,
    )
    count = max(1, min(int(num_waypoints), offsets.shape[0]))
    waypoints = start_pose.unsqueeze(0).repeat(count, 1, 1)
    waypoints[:, :3, 3] += offsets[:count]
    return [waypoint for waypoint in waypoints]


def make_demo_offset_trials(
    robot: Robot,
    *,
    total_trials: int,
    num_waypoints: int,
) -> list[PlannerTrial]:
    """Build demo-matched trials from the NeuralPlanner example path."""
    start_qpos = torch.tensor(
        FRANKA_START_QPOS, dtype=torch.float32, device=robot.device
    )
    start_pose = fk_pose(robot, start_qpos)
    waypoints = make_demo_offset_waypoints(start_pose, num_waypoints)
    return [
        PlannerTrial(
            trial_id=trial_id,
            trial_source="demo_offsets",
            start_qpos=start_qpos.clone(),
            target_qpos=torch.empty(0, 7, dtype=torch.float32, device=robot.device),
            waypoints=[waypoint.clone() for waypoint in waypoints],
        )
        for trial_id in range(total_trials)
    ]


def make_fk_bank_trials(
    robot: Robot,
    *,
    total_trials: int,
    num_waypoints: int,
    seed: int,
) -> list[PlannerTrial]:
    """Build deterministic reachable waypoint trials from FK-generated targets."""
    bank = make_seed_qpos_bank(robot)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trials: list[PlannerTrial] = []
    for trial_id in range(total_trials):
        start_idx = trial_id % len(bank)
        start_qpos = bank[start_idx]
        candidates = torch.randperm(len(bank), generator=generator).tolist()
        indices = [idx for idx in candidates if idx != start_idx]
        while len(indices) < num_waypoints:
            indices.extend(idx for idx in range(len(bank)) if idx != start_idx)
        target_qpos = torch.stack([bank[idx] for idx in indices[:num_waypoints]])
        waypoints = [fk_pose(robot, qpos) for qpos in target_qpos]
        trials.append(
            PlannerTrial(
                trial_id=trial_id,
                trial_source="fk_bank",
                start_qpos=start_qpos.clone(),
                target_qpos=target_qpos.clone(),
                waypoints=waypoints,
            )
        )
    return trials


def make_trials(
    robot: Robot,
    *,
    trial_sources: list[str],
    total_trials: int,
    num_waypoints: int,
    seed: int,
) -> list[PlannerTrial]:
    """Build planner trials for the selected benchmark layers."""
    trials: list[PlannerTrial] = []
    for source in trial_sources:
        if source == "demo_offsets":
            source_trials = make_demo_offset_trials(
                robot,
                total_trials=total_trials,
                num_waypoints=num_waypoints,
            )
        elif source == "fk_bank":
            source_trials = make_fk_bank_trials(
                robot,
                total_trials=total_trials,
                num_waypoints=num_waypoints,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported trial source: {source}")
        trials.extend(source_trials)
    return trials


def _all_success(success: bool | torch.Tensor) -> bool:
    """Return true when all success flags are true."""
    if isinstance(success, torch.Tensor):
        return bool(torch.all(success).item())
    return bool(success)


def plan_ik_toppra(
    motion_gen: MotionGenerator,
    trial: PlannerTrial,
    *,
    sample_interval: int,
) -> PlanResult:
    """Plan through EEF waypoints using sequential IK followed by TOPPRA."""
    robot = motion_gen.robot
    qpos_seed = trial.start_qpos.unsqueeze(0)
    qpos_targets = [trial.start_qpos]
    for waypoint in trial.waypoints:
        success, qpos = robot.compute_ik(
            pose=waypoint.unsqueeze(0),
            joint_seed=qpos_seed,
            name=ARM_NAME,
            env_ids=[0],
        )
        if not _all_success(success):
            return PlanResult(success=False, positions=None)
        qpos_seed = qpos
        qpos_targets.append(qpos.squeeze(0))

    target_states = [
        PlanState(move_type=MoveType.JOINT_MOVE, qpos=qpos) for qpos in qpos_targets
    ]
    return motion_gen.generate(
        target_states=target_states,
        options=ToppraMotionOptions(
            start_qpos=trial.start_qpos,
            control_part=ARM_NAME,
            sample_interval=sample_interval,
        ).to_motion_options(),
    )


@dataclass(frozen=True)
class ToppraMotionOptions:
    """Small helper to keep IK+TOPPRA runtime options explicit."""

    start_qpos: torch.Tensor
    control_part: str
    sample_interval: int

    def to_motion_options(self) -> MotionGenOptions:
        return MotionGenOptions(
            start_qpos=self.start_qpos,
            control_part=self.control_part,
            plan_opts=ToppraPlanOptions(sample_interval=self.sample_interval),
        )


def plan_neural(
    motion_gen: MotionGenerator,
    trial: PlannerTrial,
    *,
    planner_name: str,
    sample_interval: int,
) -> PlanResult:
    """Plan through EEF waypoints using NeuralPlanner."""
    target_states = [
        PlanState(move_type=MoveType.EEF_MOVE, xpos=waypoint)
        for waypoint in trial.waypoints
    ]
    result = motion_gen.generate(
        target_states=target_states,
        options=NeuralMotionOptions(
            start_qpos=trial.start_qpos,
            control_part=ARM_NAME,
        ).to_motion_options(),
    )
    if (
        planner_name == "neural_refine"
        and result.positions is not None
        and _all_success(result.success)
    ):
        seed = result.positions[-1].unsqueeze(0)
        success, refined = motion_gen.robot.compute_ik(
            pose=trial.waypoints[-1].unsqueeze(0),
            joint_seed=seed,
            name=ARM_NAME,
            env_ids=[0],
        )
        if not _all_success(success):
            return PlanResult(success=False, positions=result.positions)
        result.positions = torch.cat([result.positions, refined], dim=0)

    if result.positions is not None:
        result.positions = interpolate_with_distance(
            result.positions.unsqueeze(0),
            interp_num=sample_interval,
            device=motion_gen.device,
        ).squeeze(0)
    return result


@dataclass(frozen=True)
class NeuralMotionOptions:
    """Small helper to keep neural runtime options explicit."""

    start_qpos: torch.Tensor
    control_part: str

    def to_motion_options(self) -> MotionGenOptions:
        return MotionGenOptions(
            start_qpos=self.start_qpos,
            control_part=self.control_part,
            plan_opts=NeuralPlanOptions(
                control_part=self.control_part,
                start_qpos=self.start_qpos,
                env_id=0,
            ),
        )


def run_planner_once(
    motion_gen: MotionGenerator,
    planner_name: str,
    trial: PlannerTrial,
    args: argparse.Namespace,
) -> PlannerOutcome:
    """Run one timed planner call."""
    _reset_peak_gpu_memory()
    before = _memory_snapshot()
    _sync_cuda()
    start = time.perf_counter()
    if planner_name == "ik_toppra":
        result = plan_ik_toppra(
            motion_gen,
            trial,
            sample_interval=args.sample_interval,
        )
    elif planner_name in ("neural", "neural_refine"):
        result = plan_neural(
            motion_gen,
            trial,
            planner_name=planner_name,
            sample_interval=args.sample_interval,
        )
    else:
        raise ValueError(f"Unsupported planner: {planner_name}")
    _sync_cuda()
    elapsed = time.perf_counter() - start
    after = _memory_snapshot()
    return PlannerOutcome(
        action_success=_all_success(result.success),
        positions=result.positions,
        planning_time_sec=elapsed,
        cpu_delta_mb=after["cpu_mb"] - before["cpu_mb"],
        gpu_delta_mb=after["gpu_mb"] - before["gpu_mb"],
        peak_gpu_mb=_peak_gpu_memory_mb(),
    )


def pose_error(
    actual_pose: torch.Tensor, target_pose: torch.Tensor
) -> tuple[float, float]:
    """Return position and rotation error between pose matrices."""
    actual_quat = quat_from_matrix(actual_pose[:3, :3].unsqueeze(0))[0]
    target_quat = quat_from_matrix(target_pose[:3, :3].unsqueeze(0))[0]
    pos_error = float(torch.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3]).item())
    rot_error = float(
        quat_error_magnitude(
            target_quat.unsqueeze(0),
            actual_quat.unsqueeze(0),
        )[0].item()
    )
    return pos_error, rot_error


def trajectory_fk_poses(robot: Robot, positions: torch.Tensor) -> list[torch.Tensor]:
    """Compute FK poses for trajectory samples one row at a time."""
    return [fk_pose(robot, qpos) for qpos in positions]


def trajectory_quality(
    robot: Robot, positions: torch.Tensor | None, trial: PlannerTrial
) -> dict[str, object]:
    """Compute final pose, waypoint, and joint path quality metrics."""
    if positions is None or positions.numel() == 0:
        return {
            "trajectory_steps": 0,
            "final_tcp_pos_error": None,
            "final_tcp_rot_error": None,
            "mean_waypoint_pos_error": None,
            "max_waypoint_pos_error": None,
            "mean_waypoint_rot_error": None,
            "max_waypoint_rot_error": None,
            "joint_path_length": 0.0,
            "max_joint_step": 0.0,
            "mean_target_qpos_error": None,
            "final_qpos": None,
        }
    final_pose = fk_pose(robot, positions[-1])
    final_pos, final_rot = pose_error(final_pose, trial.waypoints[-1])
    trajectory_poses = trajectory_fk_poses(robot, positions)
    waypoint_pos_errors = []
    waypoint_rot_errors = []
    for waypoint in trial.waypoints:
        pose_errors = [pose_error(pose, waypoint) for pose in trajectory_poses]
        best_pos, best_rot = min(pose_errors, key=lambda item: item[0])
        waypoint_pos_errors.append(best_pos)
        waypoint_rot_errors.append(best_rot)
    deltas = torch.diff(positions, dim=0)
    step_norms = (
        torch.linalg.norm(deltas, dim=-1)
        if deltas.numel()
        else torch.zeros(1, device=robot.device)
    )
    target_errors = []
    for target_qpos in trial.target_qpos:
        dists = torch.linalg.norm(positions - target_qpos.unsqueeze(0), dim=-1)
        target_errors.append(float(torch.min(dists).item()))
    mean_target_qpos_error = (
        sum(target_errors) / len(target_errors) if target_errors else None
    )
    return {
        "trajectory_steps": int(positions.shape[0]),
        "final_tcp_pos_error": final_pos,
        "final_tcp_rot_error": final_rot,
        "mean_waypoint_pos_error": sum(waypoint_pos_errors) / len(waypoint_pos_errors),
        "max_waypoint_pos_error": max(waypoint_pos_errors),
        "mean_waypoint_rot_error": sum(waypoint_rot_errors) / len(waypoint_rot_errors),
        "max_waypoint_rot_error": max(waypoint_rot_errors),
        "joint_path_length": float(step_norms.sum().item()),
        "max_joint_step": float(step_norms.max().item()),
        "mean_target_qpos_error": mean_target_qpos_error,
        "final_qpos": [float(v) for v in positions[-1].detach().cpu().tolist()],
    }


def all_waypoints_within_threshold(
    robot: Robot,
    positions: torch.Tensor | None,
    trial: PlannerTrial,
    *,
    pos_threshold: float,
    rot_threshold: float,
) -> bool:
    """Return whether every target waypoint is hit by some trajectory sample."""
    if positions is None or positions.numel() == 0:
        return False
    trajectory_poses = trajectory_fk_poses(robot, positions)
    for waypoint in trial.waypoints:
        waypoint_hit = False
        for pose in trajectory_poses:
            pos_error, rot_error = pose_error(pose, waypoint)
            if pos_error <= pos_threshold and rot_error <= rot_threshold:
                waypoint_hit = True
                break
        if not waypoint_hit:
            return False
    return True


def build_trial_row(
    *,
    planner: str,
    trial: PlannerTrial,
    warmup: bool,
    outcome: PlannerOutcome,
    robot: Robot,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Build a raw benchmark row."""
    quality = trajectory_quality(robot, outcome.positions, trial)
    final_pos = quality["final_tcp_pos_error"]
    final_rot = quality["final_tcp_rot_error"]
    strict_success = (
        outcome.action_success
        and final_pos is not None
        and final_rot is not None
        and float(final_pos) <= args.pos_success_threshold
        and float(final_rot) <= args.rot_success_threshold
    )
    all_waypoint_strict_success = (
        outcome.action_success
        and all_waypoints_within_threshold(
            robot,
            outcome.positions,
            trial,
            pos_threshold=args.pos_success_threshold,
            rot_threshold=args.rot_success_threshold,
        )
    )
    nmg_threshold_success = (
        outcome.action_success
        and final_pos is not None
        and final_rot is not None
        and float(final_pos) <= args.nmg_pos_success_threshold
        and float(final_rot) <= args.nmg_rot_success_threshold
    )
    all_waypoint_nmg_threshold_success = (
        outcome.action_success
        and all_waypoints_within_threshold(
            robot,
            outcome.positions,
            trial,
            pos_threshold=args.nmg_pos_success_threshold,
            rot_threshold=args.nmg_rot_success_threshold,
        )
    )
    row: dict[str, object] = {
        "script": SCRIPT_NAME,
        "planner": planner,
        "trial_source": trial.trial_source,
        "trial_id": trial.trial_id,
        "warmup": warmup,
        "num_waypoints": len(trial.waypoints),
        "action_success": outcome.action_success,
        "strict_pose_success": bool(strict_success),
        "all_waypoint_strict_success": bool(all_waypoint_strict_success),
        "nmg_threshold_success": bool(nmg_threshold_success),
        "all_waypoint_nmg_threshold_success": bool(all_waypoint_nmg_threshold_success),
        "planning_time_sec": outcome.planning_time_sec,
        "cpu_delta_mb": outcome.cpu_delta_mb,
        "gpu_delta_mb": outcome.gpu_delta_mb,
        "peak_gpu_mb": outcome.peak_gpu_mb,
    }
    row.update(quality)
    return row


def _jsonable(value: Any) -> Any:
    """Convert tensors and nested values to JSON-serializable values."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_raw_jsonl(path: str | None, row: dict[str, object]) -> None:
    """Append one raw row when requested."""
    if not path:
        return
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _numeric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    """Return numeric values for a row key."""
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean of values."""
    if not values:
        return None
    return sum(values) / len(values)


def _quantile(values: list[float], q: float) -> float | None:
    """Return a linear-interpolated quantile."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rate(rows: list[dict[str, object]], key: str) -> float | None:
    """Return the true-rate for a boolean row key."""
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(bool(value) for value in values) / len(values)


def _fmt_float(value: float | None, digits: int = 3) -> str:
    """Format a possibly-missing float."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_rate(value: float | None) -> str:
    """Format a possibly-missing rate."""
    if value is None:
        return "-"
    return f"{value * 100.0:.1f}%"


def measured_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return non-warmup rows."""
    return [row for row in rows if not row.get("warmup", False)]


def summarize_by_source_planner(
    rows: list[dict[str, object]],
) -> list[tuple[str, str, list[dict[str, object]]]]:
    """Group measured rows by trial source and planner."""
    measured = measured_rows(rows)
    groups = sorted(
        {
            (
                str(row.get("trial_source", DEFAULT_TRIAL_SOURCE)),
                str(row["planner"]),
            )
            for row in measured
        }
    )
    return [
        (
            trial_source,
            planner,
            [
                row
                for row in measured
                if row.get("trial_source", DEFAULT_TRIAL_SOURCE) == trial_source
                and row["planner"] == planner
            ],
        )
        for trial_source, planner in groups
    ]


def make_perf_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build Time & Memory report rows."""
    perf_rows = []
    for trial_source, planner, group in summarize_by_source_planner(rows):
        times = _numeric_values(group, "planning_time_sec")
        perf_rows.append(
            {
                "trial_source": trial_source,
                "planner": planner,
                "repeat_count": len(group),
                "cost_time_ms_mean": _fmt_float(
                    None if _mean(times) is None else _mean(times) * 1000.0, 2
                ),
                "cost_time_ms_p50": _fmt_float(
                    (
                        None
                        if _quantile(times, 0.5) is None
                        else _quantile(times, 0.5) * 1000.0
                    ),
                    2,
                ),
                "cost_time_ms_p95": _fmt_float(
                    (
                        None
                        if _quantile(times, 0.95) is None
                        else _quantile(times, 0.95) * 1000.0
                    ),
                    2,
                ),
                "cpu_delta_mb": _fmt_float(
                    _mean(_numeric_values(group, "cpu_delta_mb")), 2
                ),
                "gpu_delta_mb": _fmt_float(
                    _mean(_numeric_values(group, "gpu_delta_mb")), 2
                ),
                "peak_gpu_mb": _fmt_float(
                    _mean(_numeric_values(group, "peak_gpu_mb")), 2
                ),
            }
        )
    return perf_rows


def make_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build Success & Other Metrics report rows."""
    metric_rows = []
    for trial_source, planner, group in summarize_by_source_planner(rows):
        final_pos = _mean(_numeric_values(group, "final_tcp_pos_error"))
        final_rot = _mean(_numeric_values(group, "final_tcp_rot_error"))
        mean_waypoint_pos = _mean(_numeric_values(group, "mean_waypoint_pos_error"))
        max_waypoint_pos = _mean(_numeric_values(group, "max_waypoint_pos_error"))
        mean_waypoint_rot = _mean(_numeric_values(group, "mean_waypoint_rot_error"))
        max_waypoint_rot = _mean(_numeric_values(group, "max_waypoint_rot_error"))
        metric_rows.append(
            {
                "trial_source": trial_source,
                "planner": planner,
                "action_success_rate": _fmt_rate(_rate(group, "action_success")),
                "strict_pose_success_rate": _fmt_rate(
                    _rate(group, "strict_pose_success")
                ),
                "all_waypoint_strict_success_rate": _fmt_rate(
                    _rate(group, "all_waypoint_strict_success")
                ),
                "nmg_threshold_success_rate": _fmt_rate(
                    _rate(group, "nmg_threshold_success")
                ),
                "all_waypoint_nmg_threshold_success_rate": _fmt_rate(
                    _rate(group, "all_waypoint_nmg_threshold_success")
                ),
                "final_tcp_pos_err_mm": _fmt_float(
                    None if final_pos is None else final_pos * 1000.0, 3
                ),
                "final_tcp_rot_err_deg": _fmt_float(
                    None if final_rot is None else final_rot * 180.0 / math.pi, 3
                ),
                "mean_waypoint_pos_err_mm": _fmt_float(
                    (None if mean_waypoint_pos is None else mean_waypoint_pos * 1000.0),
                    3,
                ),
                "max_waypoint_pos_err_mm": _fmt_float(
                    None if max_waypoint_pos is None else max_waypoint_pos * 1000.0,
                    3,
                ),
                "mean_waypoint_rot_err_deg": _fmt_float(
                    (
                        None
                        if mean_waypoint_rot is None
                        else mean_waypoint_rot * 180.0 / math.pi
                    ),
                    3,
                ),
                "max_waypoint_rot_err_deg": _fmt_float(
                    (
                        None
                        if max_waypoint_rot is None
                        else max_waypoint_rot * 180.0 / math.pi
                    ),
                    3,
                ),
                "joint_path_length": _fmt_float(
                    _mean(_numeric_values(group, "joint_path_length")), 4
                ),
                "max_joint_step": _fmt_float(
                    _mean(_numeric_values(group, "max_joint_step")), 4
                ),
                "mean_target_qpos_error": _fmt_float(
                    _mean(_numeric_values(group, "mean_target_qpos_error")), 4
                ),
            }
        )
    return metric_rows


def make_leaderboard_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build leaderboard rows sorted by waypoint strict success then latency."""
    leaderboard = []
    for trial_source, planner, group in summarize_by_source_planner(rows):
        strict_rate = _rate(group, "strict_pose_success") or 0.0
        waypoint_strict_rate = _rate(group, "all_waypoint_strict_success") or 0.0
        loose_rate = _rate(group, "nmg_threshold_success") or 0.0
        waypoint_loose_rate = _rate(group, "all_waypoint_nmg_threshold_success") or 0.0
        avg_time = _mean(_numeric_values(group, "planning_time_sec")) or 0.0
        avg_pos = _mean(_numeric_values(group, "final_tcp_pos_error"))
        avg_waypoint_pos = _mean(_numeric_values(group, "max_waypoint_pos_error"))
        leaderboard.append(
            {
                "trial_source": trial_source,
                "planner": planner,
                "strict_rate": strict_rate,
                "waypoint_strict_rate": waypoint_strict_rate,
                "loose_rate": loose_rate,
                "waypoint_loose_rate": waypoint_loose_rate,
                "avg_time_ms": avg_time * 1000.0,
                "avg_pos_mm": None if avg_pos is None else avg_pos * 1000.0,
                "avg_waypoint_pos_mm": (
                    None if avg_waypoint_pos is None else avg_waypoint_pos * 1000.0
                ),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            str(row["trial_source"]),
            -float(row["waypoint_strict_rate"]),
            -float(row["strict_rate"]),
            -float(row["waypoint_loose_rate"]),
            -float(row["loose_rate"]),
            float(row["avg_time_ms"]),
        )
    )
    rows = []
    current_source = None
    source_rank = 0
    for row in leaderboard:
        if row["trial_source"] != current_source:
            current_source = row["trial_source"]
            source_rank = 1
        else:
            source_rank += 1
        rows.append(
            {
                "rank": source_rank,
                "trial_source": row["trial_source"],
                "planner": row["planner"],
                "overall_success_rate": _fmt_rate(float(row["waypoint_strict_rate"])),
                "final_pose_success_rate": _fmt_rate(float(row["strict_rate"])),
                "nmg_threshold_success_rate": _fmt_rate(float(row["loose_rate"])),
                "avg_cost_time_ms": _fmt_float(float(row["avg_time_ms"]), 2),
                "avg_final_tcp_pos_err_mm": _fmt_float(row["avg_pos_mm"], 3),
                "avg_max_waypoint_pos_err_mm": _fmt_float(
                    row["avg_waypoint_pos_mm"], 3
                ),
            }
        )
    return rows


def _markdown_table(rows: list[dict[str, object]]) -> list[str]:
    """Format rows into a Markdown table."""
    if not rows:
        rows = [{"status": "No rows were produced."}]
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return lines


def default_report_path() -> Path:
    """Return the default timestamped Markdown report path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/benchmarks") / f"{SCRIPT_NAME}_{timestamp}.md"


def write_markdown_report(
    rows: list[dict[str, object]], report_path: str | None = None
) -> Path:
    """Write the benchmark Markdown report with exactly three tables."""
    path = Path(report_path) if report_path else default_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# {SCRIPT_NAME} Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Downstream demo companion: {PICK_PLACE_SCRIPT_NAME}",
        "",
        "## Time & Memory",
        "",
        *_markdown_table(make_perf_rows(rows)),
        "",
        "## Success & Other Metrics",
        "",
        *_markdown_table(make_metric_rows(rows)),
        "",
        "## Leaderboard",
        "",
        *_markdown_table(make_leaderboard_rows(rows)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_skipped_rows(
    planners: list[str],
    *,
    trial_sources: list[str],
    reason: str,
) -> list[dict[str, object]]:
    """Build rows for a gracefully skipped live-simulation benchmark."""
    rows = []
    for trial_source in trial_sources:
        for planner in planners:
            rows.append(
                {
                    "script": SCRIPT_NAME,
                    "planner": planner,
                    "trial_source": trial_source,
                    "trial_id": 0,
                    "warmup": False,
                    "num_waypoints": 0,
                    "action_success": False,
                    "strict_pose_success": False,
                    "all_waypoint_strict_success": False,
                    "nmg_threshold_success": False,
                    "all_waypoint_nmg_threshold_success": False,
                    "planning_time_sec": 0.0,
                    "cpu_delta_mb": 0.0,
                    "gpu_delta_mb": 0.0,
                    "peak_gpu_mb": 0.0,
                    "trajectory_steps": 0,
                    "final_tcp_pos_error": None,
                    "final_tcp_rot_error": None,
                    "mean_waypoint_pos_error": None,
                    "max_waypoint_pos_error": None,
                    "mean_waypoint_rot_error": None,
                    "max_waypoint_rot_error": None,
                    "joint_path_length": 0.0,
                    "max_joint_step": 0.0,
                    "mean_target_qpos_error": None,
                    "final_qpos": None,
                    "skip_reason": reason,
                }
            )
    return rows


def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    """Run the selected planner benchmark."""
    validate_args(args)
    torch.manual_seed(args.seed)
    planners = expand_planner_selection(args.planner)
    trial_sources = expand_trial_source_selection(args.trial_source)

    if args.save_raw_jsonl:
        raw_path = Path(args.save_raw_jsonl)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("", encoding="utf-8")

    if simulation_requires_cuda(args) and not torch.cuda.is_available():
        reason = "CUDA is unavailable, so the Franka planner benchmark was skipped."
        rows = make_skipped_rows(planners, trial_sources=trial_sources, reason=reason)
        for row in rows:
            write_raw_jsonl(args.save_raw_jsonl, row)
        write_markdown_report(rows, args.report_path)
        return rows

    checkpoint_path = resolve_checkpoint(args, planners)
    sim = make_sim(args)
    robot, _ = create_franka(sim)
    sim.update(step=1)
    total_trials = args.warmup_trials + args.num_trials
    trials = make_trials(
        robot,
        trial_sources=trial_sources,
        total_trials=total_trials,
        num_waypoints=args.num_waypoints,
        seed=args.seed,
    )
    rows: list[dict[str, object]] = []
    for planner in planners:
        motion_gen = build_motion_generator(robot, planner, checkpoint_path, args)
        for trial in trials:
            warmup = trial.trial_id < args.warmup_trials
            robot.set_qpos(trial.start_qpos.unsqueeze(0), name=ARM_NAME, target=False)
            robot.set_qpos(trial.start_qpos.unsqueeze(0), name=ARM_NAME, target=True)
            outcome = run_planner_once(motion_gen, planner, trial, args)
            row = build_trial_row(
                planner=planner,
                trial=trial,
                warmup=warmup,
                outcome=outcome,
                robot=robot,
                args=args,
            )
            rows.append(row)
            write_raw_jsonl(args.save_raw_jsonl, row)

    report_path = write_markdown_report(rows, args.report_path)
    for row in make_leaderboard_rows(rows):
        print(json.dumps(row, sort_keys=True))
    print(f"Markdown benchmark report saved: {report_path}")
    return rows


def run_all_benchmarks() -> None:
    """Parse CLI args and run the Franka planner benchmark."""
    run_benchmark(parse_args())


if __name__ == "__main__":
    run_all_benchmarks()
