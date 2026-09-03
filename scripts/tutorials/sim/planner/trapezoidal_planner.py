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

"""Compare velocity- and acceleration-trapezoidal joint trajectories."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/embodichain-matplotlib")

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenerator,
    MotionGenOptions,
    PlanResult,
    PlanState,
    TrapezoidalPlannerCfg,
    TrapezoidalPlanOptions,
    BezierPath,
)
from embodichain.lab.sim.planners.trapezoidal_planner import _plan_linear_profiles
from embodichain.lab.sim.robots import CobotMagicCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.utils.math import euler_xyz_from_quat

DEFAULT_SAMPLES = 200
DEFAULT_CARTESIAN_DISTANCE = 0.10
DEFAULT_CARTESIAN_STEP = 0.01
DEFAULT_REPLAY_SPEED = 1.0
DEFAULT_CARTESIAN_VELOCITY = 0.15
DEFAULT_CARTESIAN_ACCELERATION = 0.30
DEFAULT_CARTESIAN_JERK = 1.0
PROFILE_SPECS = {
    "velocity_trapezoidal": ("trapezoidal", "velocity_trapezoidal"),
    "acceleration_trapezoidal": ("double_s", "acceleration_trapezoidal"),
}


def configure_plot_fonts() -> None:
    """Apply lightweight readable font defaults for this tutorial."""
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans", "sans-serif"],
            "font.size": 11.0,
            "axes.titlesize": 13.0,
            "legend.fontsize": 9.0,
            "axes.unicode_minus": False,
        }
    )


def positive_float(value: str) -> float:
    """Parse a finite positive command-line float."""
    parsed = float(value)
    if not torch.isfinite(torch.tensor(parsed)) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return parsed


def sample_count(value: str) -> int:
    """Parse a trajectory sample count accepted by the planner."""
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("sample count must be at least 2")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse tutorial arguments."""
    # The shared launcher owns a boolean ``--profile`` flag for environment
    # timing. This focused tutorial intentionally reuses that concise name for
    # its trajectory profile and therefore replaces the shared action.
    parser = argparse.ArgumentParser(description=__doc__, conflict_handler="resolve")
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--profile",
        choices=(*PROFILE_SPECS, "both"),
        default="acceleration_trapezoidal",
        help=(
            "Diagnostic to run: trapezoidal velocity, jerk-limited "
            "trapezoidal acceleration, or both."
        ),
    )
    parser.add_argument(
        "--samples",
        type=sample_count,
        default=DEFAULT_SAMPLES,
        help="Number of output trajectory samples.",
    )
    parser.add_argument(
        "--path",
        choices=("joint", "cartesian", "both"),
        default="cartesian",
        help="Plan a synchronized multi-joint path, a straight EEF path, or both.",
    )
    parser.add_argument(
        "--cartesian-distance",
        type=positive_float,
        default=DEFAULT_CARTESIAN_DISTANCE,
        help="Length in metres of the diagonal straight EEF demo path.",
    )
    parser.add_argument(
        "--cartesian-path",
        choices=("bezier", "line"),
        default="line",
        help="Cartesian geometric path; Bézier is the default.",
    )
    parser.add_argument(
        "--cartesian-step",
        type=positive_float,
        default=DEFAULT_CARTESIAN_STEP,
        help="Cartesian interpolation spacing in metres before IK.",
    )
    parser.add_argument(
        "--cartesian-velocity",
        type=positive_float,
        default=DEFAULT_CARTESIAN_VELOCITY,
        help="Maximum straight-line EEF speed in m/s.",
    )
    parser.add_argument(
        "--cartesian-acceleration",
        type=positive_float,
        default=DEFAULT_CARTESIAN_ACCELERATION,
        help="Maximum straight-line EEF acceleration in m/s².",
    )
    parser.add_argument(
        "--cartesian-jerk",
        type=positive_float,
        default=DEFAULT_CARTESIAN_JERK,
        help="Maximum straight-line EEF jerk in m/s³.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "torch", "warp"),
        default="auto",
        help="Backend used to compose sampled joint states.",
    )
    parser.add_argument(
        "--replay-speed",
        type=positive_float,
        default=DEFAULT_REPLAY_SPEED,
        help="Trajectory playback speed multiplier in the simulation window.",
    )
    parser.add_argument(
        "--plot-env",
        type=int,
        default=0,
        help="Batch environment index shown in the diagnostic plot.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional PNG path. By default the figure is not saved.",
    )
    parser.add_argument(
        "--show-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the plot interactively (default: enabled).",
    )
    return parser.parse_args()


def build_demo_waypoints(robot: Robot, control_part: str) -> list[torch.Tensor]:
    """Create a batched path where every arm joint moves synchronously."""
    start = robot.get_qpos(name=control_part).clone()
    dof = start.shape[1]
    if dof == 6:
        # Same non-singular seed used by the TOPPRA motion-generator tutorial.
        safe_seed = start.new_tensor(
            (0.0, torch.pi / 4.0, -torch.pi / 4.0, 0.0, torch.pi / 4.0, 0.0)
        )
        start = safe_seed.unsqueeze(0).expand_as(start).clone()
    joint_index = torch.arange(dof, dtype=start.dtype, device=start.device)
    direction = torch.where(joint_index.remainder(2) == 0, 1.0, -1.0)
    magnitude = torch.linspace(0.12, 0.30, dof, dtype=start.dtype, device=start.device)
    middle = start + direction * magnitude
    goal = start - direction * magnitude.flip(0) * 0.75
    joint_ids = robot.get_joint_ids(name=control_part)
    limits = robot.get_qpos_limits(joint_ids=joint_ids).to(start)
    margin = torch.minimum(
        torch.full_like(limits[..., 0], 0.05),
        (limits[..., 1] - limits[..., 0]).clamp_min(0.0) * 0.1,
    )
    lower = limits[..., 0] + margin
    upper = limits[..., 1] - margin
    start = torch.maximum(torch.minimum(start, upper), lower)
    middle = torch.maximum(torch.minimum(middle, upper), lower)
    goal = torch.maximum(torch.minimum(goal, upper), lower)
    return [start, middle, goal]


def build_cartesian_line_poses(
    robot: Robot,
    control_part: str,
    start_qpos: torch.Tensor,
    distance: float,
) -> list[torch.Tensor]:
    """Create two poses defining a fixed-orientation straight EEF path."""
    if not torch.isfinite(torch.tensor(distance)) or distance <= 0.0:
        raise ValueError("cartesian distance must be finite and greater than zero.")
    start_pose = robot.compute_fk(qpos=start_qpos, name=control_part, to_matrix=True)
    if start_pose is None:
        raise RuntimeError(f"Forward kinematics is unavailable for {control_part!r}.")
    goal_pose = start_pose.clone()
    # Match the known-reachable first Cartesian segment from the TOPPRA demo.
    direction = start_pose.new_tensor((0.0, 0.0, -1.0))
    goal_pose[:, :3, 3] += distance * direction
    return [start_pose, goal_pose]


def joint_derivatives_from_path_time_law(
    jacobians: torch.Tensor,
    path_positions: torch.Tensor,
    path_velocities: torch.Tensor,
    path_accelerations: torch.Tensor,
    cartesian_tangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert scalar path derivatives into joint derivatives.

    For a Cartesian line ``x(s)`` the tangent is constant and
    ``d²x/ds² = 0``. The joint path derivatives therefore satisfy
    ``J q_s = x_s`` and ``J q_ss = -J_s q_s``. Double-S then supplies the
    time law through ``q_dot = q_s s_dot`` and
    ``q_ddot = q_ss s_dot² + q_s s_ddot``.

    Args:
        jacobians: Geometric Jacobians shaped ``(B, N, 6, DOF)``.
        path_positions: Scalar metric path positions shaped ``(B, N)``.
        path_velocities: Scalar path velocities shaped ``(B, N)``.
        path_accelerations: Scalar path accelerations shaped ``(B, N)``.
        cartesian_tangent: Constant unit twists shaped ``(B, 6)``.

    Returns:
        Joint velocity and acceleration tensors shaped ``(B, N, DOF)``.
    """
    batch_size, sample_count, _, dof = jacobians.shape
    expected_scalar_shape = (batch_size, sample_count)
    if (
        path_positions.shape != expected_scalar_shape
        or path_velocities.shape != expected_scalar_shape
        or path_accelerations.shape != expected_scalar_shape
        or cartesian_tangent.shape != (batch_size, 6)
    ):
        raise ValueError("Path derivative tensors have incompatible shapes.")
    if sample_count < 3:
        raise ValueError("At least three path samples are required.")
    if bool((torch.diff(path_positions, dim=1) <= 0.0).any().item()):
        raise ValueError("Path positions must be strictly increasing.")

    jacobian_pinv = torch.linalg.pinv(jacobians)
    tangent = cartesian_tangent[:, None, :, None].expand(-1, sample_count, -1, -1)
    q_s = torch.matmul(jacobian_pinv, tangent).squeeze(-1)
    jacobian_s_rows: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        jacobian_s_rows.append(
            torch.gradient(
                jacobians[batch_index],
                spacing=(path_positions[batch_index],),
                dim=(0,),
                edge_order=2,
            )[0]
        )
    jacobian_s = torch.stack(jacobian_s_rows)
    curvature_twist = torch.matmul(jacobian_s, q_s.unsqueeze(-1))
    q_ss = -torch.matmul(jacobian_pinv, curvature_twist).squeeze(-1)
    velocities = q_s * path_velocities[..., None]
    accelerations = (
        q_ss * path_velocities.square()[..., None] + q_s * path_accelerations[..., None]
    )
    if velocities.shape != (batch_size, sample_count, dof):
        raise RuntimeError("Unexpected joint derivative shape.")
    return velocities, accelerations


def nearest_equivalent_joint_solution(
    solution: torch.Tensor,
    seed: torch.Tensor,
    limits: torch.Tensor,
) -> torch.Tensor:
    """Select the limit-valid ``2π`` equivalent closest to the previous seed."""
    if solution.shape != seed.shape or limits.shape != (*solution.shape, 2):
        raise ValueError("solution, seed, and limits have incompatible shapes.")
    turns = torch.round((seed - solution) / (2.0 * torch.pi))
    nearest = solution + turns * (2.0 * torch.pi)
    valid = (nearest >= limits[..., 0]) & (nearest <= limits[..., 1])
    return torch.where(valid, nearest, solution)


def plan_cartesian_line(
    robot: Robot,
    control_part: str,
    start_qpos: torch.Tensor,
    *,
    distance: float,
    profile: str,
    sample_count: int,
    velocity_limit: float,
    acceleration_limit: float,
    jerk_limit: float,
    backend: str,
) -> tuple[PlanResult, torch.Tensor, PlanResult]:
    """Time-parameterize Cartesian line distance, then solve continuous IK.

    Returns:
        Joint trajectory, desired Cartesian pose samples, and the scalar
        Cartesian path-parameter trajectory carrying metric derivatives.
    """
    if sample_count < 3:
        raise ValueError("Cartesian planning requires sample_count >= 3.")
    start_pose, goal_pose = build_cartesian_line_poses(
        robot, control_part, start_qpos, distance
    )
    batch_size = start_qpos.shape[0]
    scalar_waypoints = start_qpos.new_zeros((batch_size, 2, 1))
    scalar_waypoints[:, 1, 0] = distance
    scalar_plan = _plan_linear_profiles(
        scalar_waypoints,
        TrapezoidalPlanOptions(
            profile=profile,
            constraints={
                "velocity": velocity_limit,
                "acceleration": acceleration_limit,
                "jerk": jerk_limit,
            },
            sample_interval=sample_count,
            backend=backend,
        ),
    )
    progress = scalar_plan.positions[..., 0] / distance
    desired_poses = start_pose[:, None].expand(-1, sample_count, -1, -1).clone()
    translation = torch.lerp(
        start_pose[:, None, :3, 3],
        goal_pose[:, None, :3, 3],
        progress[..., None],
    )
    desired_poses[:, :, :3, 3] = translation

    joint_ids = robot.get_joint_ids(name=control_part)
    joint_limits = robot.get_qpos_limits(joint_ids=joint_ids).to(start_qpos)
    ik_result = robot.compute_batch_ik(
        desired_poses,
        start_qpos,
        control_part,
        continuous=True,
    )
    if ik_result is None:
        raise RuntimeError("Cartesian line IK solver is unavailable.")
    sample_success, positions = ik_result
    sample_success = sample_success.bool()
    if not bool(sample_success.all().item()):
        failure = torch.nonzero(~sample_success, as_tuple=False)[0].tolist()
        raise RuntimeError(
            f"Cartesian line IK failed at env {failure[0]}, sample {failure[1]}."
        )
    previous = torch.cat((start_qpos[:, None], positions[:, :-1]), dim=1)
    positions = nearest_equivalent_joint_solution(
        positions,
        previous,
        joint_limits[:, None].expand(-1, sample_count, -1, -1),
    )
    success = sample_success.all(dim=1)
    solver = robot.get_solver(control_part)
    if solver is None:
        raise RuntimeError(f"Kinematic solver is unavailable for {control_part!r}.")
    jacobians = solver.get_jacobian(
        positions.reshape(batch_size * sample_count, -1), jac_type="full"
    ).reshape(batch_size, sample_count, 6, -1)
    base_pose = robot.get_control_part_base_pose(control_part, to_matrix=True)
    world_direction = goal_pose[:, :3, 3] - start_pose[:, :3, 3]
    world_direction /= torch.linalg.vector_norm(world_direction, dim=-1, keepdim=True)
    root_direction = torch.matmul(
        base_pose[:, :3, :3].transpose(-1, -2), world_direction.unsqueeze(-1)
    ).squeeze(-1)
    cartesian_tangent = torch.cat(
        (root_direction, torch.zeros_like(root_direction)), dim=-1
    ).to(jacobians)
    velocities, accelerations = joint_derivatives_from_path_time_law(
        jacobians,
        scalar_plan.positions[..., 0],
        scalar_plan.velocities[..., 0],
        scalar_plan.accelerations[..., 0],
        cartesian_tangent,
    )
    return (
        PlanResult(
            success=success,
            xpos_list=desired_poses,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            dt=scalar_plan.dt,
        ),
        desired_poses,
        scalar_plan,
    )


def plan_cartesian_bezier(
    robot: Robot,
    control_part: str,
    start_qpos: torch.Tensor,
    **kwargs,
) -> tuple[PlanResult, torch.Tensor, PlanResult]:
    """Plan a quintic Cartesian Bézier path with Double-S-compatible timing."""
    distance = float(kwargs["distance"])
    sample_count = int(kwargs["sample_count"])
    start_pose, goal_pose = build_cartesian_line_poses(
        robot, control_part, start_qpos, distance
    )
    midpoint = 0.5 * (start_pose[:, :3, 3] + goal_pose[:, :3, 3])
    midpoint[:, 0] += 0.35 * distance
    start_position = start_pose[:, :3, 3]
    goal_position = goal_pose[:, :3, 3]
    # Quintic Bézier control polygon with smooth endpoint tangents and a
    # lateral middle section; this is the default production-style path.
    control_points = torch.stack(
        (
            start_position,
            torch.lerp(start_position, midpoint, 0.20),
            torch.lerp(start_position, midpoint, 0.65),
            torch.lerp(midpoint, goal_position, 0.35),
            torch.lerp(midpoint, goal_position, 0.80),
            goal_position,
        ),
        dim=1,
    )
    dense_count = max(1025, sample_count * 8)
    dense_points, dense_lengths = BezierPath(control_points).sample(dense_count)
    total_length = dense_lengths[:, -1]
    scalar_waypoints = start_qpos.new_zeros((start_qpos.shape[0], 2, 1))
    scalar_waypoints[:, 1, 0] = total_length
    scalar_plan = _plan_linear_profiles(
        scalar_waypoints,
        TrapezoidalPlanOptions(
            profile=str(kwargs["profile"]),
            constraints={
                "velocity": float(kwargs["velocity_limit"]),
                "acceleration": float(kwargs["acceleration_limit"]),
                "jerk": float(kwargs["jerk_limit"]),
            },
            sample_interval=sample_count,
            backend=str(kwargs["backend"]),
        ),
    )
    progress = scalar_plan.positions[..., 0] / total_length[:, None]
    dense_index = progress.clamp(0.0, 1.0) * (dense_count - 1)
    lower = dense_index.floor().long().clamp_max(dense_count - 2)
    alpha = (dense_index - lower)[..., None]
    gather_index = lower[..., None].expand(-1, -1, 3)
    lower_points = dense_points.gather(1, gather_index)
    upper_points = dense_points.gather(1, gather_index + 1)
    translations = torch.lerp(lower_points, upper_points, alpha)
    desired_poses = start_pose[:, None].expand(-1, sample_count, -1, -1).clone()
    desired_poses[:, :, :3, 3] = translations
    ik_result = robot.compute_batch_ik(
        desired_poses, start_qpos, control_part, continuous=True
    )
    if ik_result is None or not bool(ik_result[0].all().item()):
        raise RuntimeError("Cartesian Bézier IK failed.")
    positions = ik_result[1]
    time = scalar_plan.dt.cumsum(dim=1)
    velocities = torch.stack(
        [torch.gradient(positions[i], spacing=(time[i],), dim=(0,))[0] for i in range(positions.shape[0])]
    )
    accelerations = torch.stack(
        [torch.gradient(velocities[i], spacing=(time[i],), dim=(0,))[0] for i in range(positions.shape[0])]
    )
    return (
        PlanResult(
            success=ik_result[0].all(dim=1),
            xpos_list=desired_poses,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            dt=scalar_plan.dt,
        ),
        desired_poses,
        scalar_plan,
    )


def replay_plan(
    sim: SimulationManager,
    robot: Robot,
    control_part: str,
    positions: torch.Tensor,
    dt: torch.Tensor,
    *,
    replay_speed: float = DEFAULT_REPLAY_SPEED,
    realtime: bool = True,
) -> None:
    """Drive a joint plan according to its explicit trajectory timing."""
    if positions.ndim != 3 or dt.shape != positions.shape[:2]:
        raise ValueError("positions and dt must have shapes (B, N, DOF) and (B, N).")
    if replay_speed <= 0.0:
        raise ValueError("replay_speed must be greater than zero.")
    if positions.shape[0] > 1 and not torch.allclose(dt, dt[:1].expand_as(dt)):
        raise ValueError(
            "replay_plan requires one environment or identical dt rows across environments."
        )
    physics_dt = float(sim.sim_config.physics_dt)
    wall_start = time.perf_counter()
    target_elapsed = 0.0
    for sample_index, command in enumerate(positions.transpose(0, 1)):
        sample_duration = float(dt[:, sample_index].max().item()) / replay_speed
        robot.set_qpos(command, name=control_part)
        physics_steps = max(1, math.ceil(sample_duration / physics_dt))
        sim.update(step=physics_steps)
        if realtime:
            target_elapsed += sample_duration
            remaining = wall_start + target_elapsed - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)


def compute_eef_trajectory(
    robot: Robot,
    control_part: str,
    joint_positions: torch.Tensor,
    env_index: int,
) -> torch.Tensor:
    """Evaluate FK for one row of a trajectory shaped ``(B, N, DOF)``.

    Returns:
        End-effector position and quaternion as ``(N, 7)`` in the local arena
        frame, ordered as ``x, y, z, qw, qx, qy, qz``.
    """
    if joint_positions.ndim != 3:
        raise ValueError("joint_positions must have shape (B, N, DOF).")
    if not 0 <= env_index < joint_positions.shape[0]:
        raise ValueError(
            f"env_index must be in [0, {joint_positions.shape[0]}), got {env_index}."
        )
    sample_count = joint_positions.shape[1]
    poses = robot.compute_fk(
        qpos=joint_positions[env_index],
        name=control_part,
        env_ids=[env_index] * sample_count,
        to_matrix=False,
    )
    if poses is None:
        raise RuntimeError(f"Forward kinematics is unavailable for {control_part!r}.")
    return poses


def unwrap_angles(angles: torch.Tensor) -> torch.Tensor:
    """Remove artificial ``2π`` jumps from a sequence of Euler angles."""
    if angles.ndim != 2 or angles.shape[-1] != 3:
        raise ValueError("angles must have shape (N, 3).")
    if angles.shape[0] < 2:
        return angles.clone()
    delta = torch.diff(angles, dim=0)
    wrapped_delta = torch.remainder(delta + torch.pi, 2.0 * torch.pi) - torch.pi
    wrapped_delta = torch.where(
        (wrapped_delta == -torch.pi) & (delta > 0.0), torch.pi, wrapped_delta
    )
    correction = wrapped_delta - delta
    correction = torch.where(delta.abs() < torch.pi, 0.0, correction)
    return torch.cat((angles[:1], angles[1:] + correction.cumsum(dim=0)), dim=0)


def maximum_line_deviation(xyz: torch.Tensor) -> torch.Tensor:
    """Return the largest orthogonal deviation from the endpoint line."""
    if xyz.ndim != 2 or xyz.shape[0] < 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3) with N >= 2.")
    line = xyz[-1] - xyz[0]
    squared_length = torch.dot(line, line)
    if squared_length <= torch.finfo(xyz.dtype).eps:
        return torch.linalg.vector_norm(xyz - xyz[:1], dim=-1).max()
    progress = ((xyz - xyz[:1]) * line).sum(dim=-1) / squared_length
    closest = xyz[:1] + progress[:, None] * line
    return torch.linalg.vector_norm(xyz - closest, dim=-1).max()


def set_equal_3d_limits(axis: Axes3D, points: torch.Tensor) -> None:
    """Use one physical scale for all axes of a 3D trajectory plot."""
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3) with N >= 2.")
    lower = points.amin(dim=0)
    upper = points.amax(dim=0)
    center = 0.5 * (lower + upper)
    largest_span = float((upper - lower).max().item())
    radius = max(0.5 * largest_span * 1.1, 1e-4)
    for setter, coordinate in (
        (axis.set_xlim, center[0]),
        (axis.set_ylim, center[1]),
        (axis.set_zlim, center[2]),
    ):
        midpoint = float(coordinate.item())
        setter(midpoint - radius, midpoint + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def plot_trajectory_diagnostics(
    *,
    dt: torch.Tensor,
    eef_poses: torch.Tensor,
    joint_positions: torch.Tensor,
    joint_velocities: torch.Tensor,
    joint_accelerations: torch.Tensor,
    desired_eef_poses: torch.Tensor | None = None,
    env_index: int,
    output_path: Path | None,
    profile_label: str,
    show: bool = False,
) -> Figure:
    """Plot Cartesian pose and joint derivatives for one batch environment."""
    configure_plot_fonts()
    batch_size = joint_positions.shape[0]
    if not 0 <= env_index < batch_size:
        raise ValueError(f"env_index must be in [0, {batch_size}), got {env_index}.")
    expected_prefix = joint_positions.shape[:2]
    if (
        dt.shape != expected_prefix
        or eef_poses.shape != (expected_prefix[1], 7)
        or joint_velocities.shape != joint_positions.shape
        or joint_accelerations.shape != joint_positions.shape
    ):
        raise ValueError("Trajectory diagnostic tensors have incompatible shapes.")

    time = dt[env_index].cumsum(dim=0).detach().cpu()
    pose = eef_poses.detach().cpu()
    roll, pitch, yaw = euler_xyz_from_quat(pose[:, 3:])
    orientation = unwrap_angles(torch.stack((roll, pitch, yaw), dim=-1))
    orientation_deg = torch.rad2deg(orientation)
    position = joint_positions[env_index].detach().cpu()
    velocity = joint_velocities[env_index].detach().cpu()
    acceleration = joint_accelerations[env_index].detach().cpu()

    xyz = pose[:, :3]
    duration = time[-1].item()
    peak_velocity = velocity.abs().max().item()
    peak_acceleration = acceleration.abs().max().item()
    line_deviation_mm = maximum_line_deviation(xyz).item() * 1000.0

    figure = plt.figure(figsize=(16, 13), layout="constrained")
    grid = figure.add_gridspec(3, 2)
    path_axis = figure.add_subplot(grid[0, 0], projection="3d")
    axes = [
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
    ]
    path_axis.plot(
        xyz[:, 0].numpy(),
        xyz[:, 1].numpy(),
        xyz[:, 2].numpy(),
        color="tab:blue",
        linewidth=2.5,
        label="FK path",
    )
    reference_xyz = (
        desired_eef_poses[:, :3, 3].detach().cpu()
        if desired_eef_poses is not None
        else torch.stack((xyz[0], xyz[-1]))
    )
    path_axis.plot(
        reference_xyz[:, 0].numpy(),
        reference_xyz[:, 1].numpy(),
        reference_xyz[:, 2].numpy(),
        color="black",
        linestyle="--",
        alpha=0.65,
        label="planned Cartesian line",
    )
    path_axis.scatter(*xyz[0].tolist(), color="tab:green", s=70, label="start")
    path_axis.scatter(*xyz[-1].tolist(), color="tab:red", s=70, label="goal")
    set_equal_3d_limits(path_axis, torch.cat((xyz, reference_xyz), dim=0))
    path_axis.set_title("EEF path in Cartesian space")
    path_axis.set_xlabel("x [m]")
    path_axis.set_ylabel("y [m]")
    path_axis.set_zlabel("z [m]")
    path_axis.legend()

    colors = plt.get_cmap("tab10").colors
    for axis, values, labels, title, ylabel in (
        (axes[0], xyz, ("x", "y", "z"), "EEF XYZ", "position [m]"),
        (
            axes[1],
            orientation_deg,
            ("roll", "pitch", "yaw"),
            "EEF orientation (XYZ Euler)",
            "angle [deg]",
        ),
        (
            axes[2],
            position,
            tuple(f"q{i}" for i in range(position.shape[1])),
            "Joint position",
            "angle [rad]",
        ),
        (
            axes[3],
            velocity,
            tuple(f"dq{i}" for i in range(velocity.shape[1])),
            "Joint velocity",
            "velocity [rad/s]",
        ),
        (
            axes[4],
            acceleration,
            tuple(f"ddq{i}" for i in range(acceleration.shape[1])),
            "Joint acceleration",
            "acceleration [rad/s²]",
        ),
    ):
        for curve_index, (curve, label) in enumerate(
            zip(values.T.numpy(), labels, strict=True)
        ):
            axis.plot(
                time.numpy(),
                curve,
                color=colors[curve_index % len(colors)],
                linewidth=1.8,
                label=label,
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("time [s]")
        axis.grid(True, alpha=0.22)
        axis.legend(ncol=min(3, len(labels)), frameon=False)
    if desired_eef_poses is not None:
        desired_xyz = desired_eef_poses[:, :3, 3].detach().cpu()
        for coordinate, label, color in zip(
            desired_xyz.T,
            ("x desired", "y desired", "z desired"),
            colors[:3],
            strict=True,
        ):
            axes[0].plot(
                time.numpy(),
                coordinate.numpy(),
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=label,
            )
        axes[0].legend(ncol=3, frameon=False)
    figure.suptitle(
        f"{profile_label.replace('_', ' ').title()} — env {env_index}\n"
        f"duration {duration:.3f} s  |  max |dq| {peak_velocity:.3f} rad/s  |  "
        f"max |ddq| {peak_acceleration:.3f} rad/s²  |  "
        f"line error {line_deviation_mm:.3f} mm",
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"[INFO] trajectory plot saved to {output_path.resolve()}")
    if show:
        plt.show()
    return figure


def diagnostic_output_path(
    base_path: Path | None, diagnostic_label: str, multiple: bool
) -> Path | None:
    """Return a stable output path for one requested diagnostic profile."""
    if base_path is None or not multiple:
        return base_path
    suffix = base_path.suffix or ".png"
    return base_path.with_name(f"{base_path.stem}_{diagnostic_label}{suffix}")


def main() -> None:
    """Create a robot, plan a trajectory, and replay it."""
    args = parse_args()
    figures: list[Figure] = []
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=args.headless,
            sim_device=args.device,
            num_envs=args.num_envs,
            visualization=visualization_cfg_from_args(args),
        )
    )
    try:
        robot = sim.add_robot(CobotMagicCfg.from_dict({"uid": "CobotMagic"}))
        if sim.is_use_gpu_physics:
            sim.init_gpu_physics()
        if not args.headless:
            sim.open_window()

        control_part = "left_arm"
        joint_waypoints = build_demo_waypoints(robot, control_part)
        start_qpos = joint_waypoints[0]
        generator = MotionGenerator(
            MotionGenCfg(
                planner_cfg=TrapezoidalPlannerCfg(robot_uid=robot.uid),
            )
        )
        requested_profiles = (
            tuple(PROFILE_SPECS.items())
            if args.profile == "both"
            else ((args.profile, PROFILE_SPECS[args.profile]),)
        )
        requested_paths = (
            ("joint", "cartesian") if args.path == "both" else (args.path,)
        )
        diagnostic_count = len(requested_profiles) * len(requested_paths)
        for path_name in requested_paths:
            if path_name == "joint":
                target_states = [PlanState.from_qpos(qpos) for qpos in joint_waypoints]
            for profile_name, (
                planner_profile,
                profile_label,
            ) in requested_profiles:
                scalar_plan = None
                if path_name == "cartesian":
                    planner_kwargs = dict(
                        distance=args.cartesian_distance,
                        profile=planner_profile,
                        sample_count=max(args.samples, math.ceil(args.cartesian_distance / args.cartesian_step) + 1),
                        velocity_limit=args.cartesian_velocity,
                        acceleration_limit=args.cartesian_acceleration,
                        jerk_limit=args.cartesian_jerk,
                        backend=args.backend,
                    )
                    cartesian_planner = (
                        plan_cartesian_bezier
                        if args.cartesian_path == "bezier"
                        else plan_cartesian_line
                    )
                    result, desired_poses, scalar_plan = cartesian_planner(
                        robot, control_part, start_qpos, **planner_kwargs
                    )
                else:
                    desired_poses = None
                    result = generator.generate(
                        target_states,
                        MotionGenOptions(
                            control_part=control_part,
                            start_qpos=start_qpos,
                            plan_opts=TrapezoidalPlanOptions(
                                profile=planner_profile,
                                constraints={
                                    "velocity": 0.5,
                                    "acceleration": 1.0,
                                    "jerk": 3.0,
                                },
                                sample_interval=args.samples,
                                stop_at_waypoints=False,
                                backend=args.backend,
                            ),
                        ),
                    )
                if (
                    not result.is_all_success()
                    or result.positions is None
                    or result.velocities is None
                    or result.accelerations is None
                    or result.dt is None
                ):
                    raise RuntimeError(
                        f"{path_name}/{profile_name} trajectory planning failed."
                    )
                print(
                    f"[INFO] path={path_name}, profile={profile_name}, "
                    f"shape={tuple(result.positions.shape)}, "
                    f"duration={result.duration.tolist()}"
                )
                eef_poses = compute_eef_trajectory(
                    robot, control_part, result.positions, args.plot_env
                )
                if path_name == "cartesian":
                    line_error = maximum_line_deviation(eef_poses[:, :3])
                    assert scalar_plan is not None
                    print(
                        "[INFO] cartesian max line deviation="
                        f"{line_error.item() * 1000.0:.3f} mm"
                    )
                    print(
                        "[INFO] cartesian peaks: "
                        f"speed={scalar_plan.velocities.abs().max().item():.3f} m/s, "
                        "acceleration="
                        f"{scalar_plan.accelerations.abs().max().item():.3f} m/s²"
                    )
                diagnostic_label = f"{path_name}_{profile_label}"
                figures.append(
                    plot_trajectory_diagnostics(
                        dt=result.dt,
                        eef_poses=eef_poses,
                        joint_positions=result.positions,
                        joint_velocities=result.velocities,
                        joint_accelerations=result.accelerations,
                        desired_eef_poses=(
                            desired_poses[args.plot_env]
                            if desired_poses is not None
                            else None
                        ),
                        env_index=args.plot_env,
                        output_path=diagnostic_output_path(
                            args.plot_output,
                            diagnostic_label,
                            diagnostic_count > 1,
                        ),
                        profile_label=diagnostic_label,
                    )
                )
                robot.set_qpos(start_qpos, name=control_part, target=False)
                robot.set_qpos(start_qpos, name=control_part)
                sim.update(step=5)
                replay_plan(
                    sim,
                    robot,
                    control_part,
                    result.positions,
                    result.dt,
                    replay_speed=args.replay_speed,
                    realtime=not args.headless,
                )
        if args.show_plot:
            plt.show()
    finally:
        for figure in figures:
            plt.close(figure)
        sim.destroy()


if __name__ == "__main__":
    main()
