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

"""Ordered waypoint and free-space trajectory validity metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from embodichain.compute.kinematics import pose_error
from embodichain.compute.trajectory import resample_with_distance

from ..models import BenchmarkCase, CaseOutcome
from ..limits import (
    resolve_dynamic_limit,
    resolve_robot_velocity_limits,
    resolve_velocity_limits,
)
from .stats import nearest_rank_percentile

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.motion.planners.utils import PlanResult

__all__ = [
    "compute_case_outcomes",
    "resolve_dynamic_limit",
    "resolve_velocity_limits",
    "compute_waypoint_errors",
    "get_pose_err",
    "make_failure_outcomes",
    "match_ordered_joint_waypoints",
    "match_ordered_waypoints",
]


def _pose_error_matrices(
    waypoints: torch.Tensor,
    trajectory_poses: torch.Tensor,
    *,
    rotation_symmetry: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return waypoint-by-sample translation and geodesic rotation errors."""
    waypoints = torch.as_tensor(waypoints, dtype=torch.float64)
    trajectory_poses = torch.as_tensor(
        trajectory_poses, dtype=torch.float64, device=waypoints.device
    )
    pos_error = torch.linalg.norm(
        waypoints[:, None, :3, 3] - trajectory_poses[None, :, :3, 3], dim=-1
    )
    _, residual = pose_error(trajectory_poses[None], waypoints[:, None])
    rotation_error = residual.norm(dim=-1)
    if rotation_symmetry == "half_turn_about_z":
        symmetric_poses = waypoints.clone()
        symmetric_poses[:, :3, :2] *= -1
        _, residual = pose_error(trajectory_poses[None], symmetric_poses[:, None])
        symmetric_error = residual.norm(dim=-1)
        rotation_error = torch.minimum(rotation_error, symmetric_error)
    elif rotation_symmetry is not None:
        raise ValueError(
            "rotation_symmetry must be None or 'half_turn_about_z', "
            f"got {rotation_symmetry!r}."
        )
    return pos_error, rotation_error


def get_pose_err(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    *,
    rotation_symmetry: str | None = None,
) -> tuple[float, float]:
    """Return translation (m) and geodesic rotation (rad) pose errors."""
    tensor_a = torch.as_tensor(matrix_a, dtype=torch.float64)
    tensor_b = torch.as_tensor(matrix_b, dtype=torch.float64, device=tensor_a.device)
    if tensor_a.ndim == 2:
        tensor_a = tensor_a.unsqueeze(0)
    if tensor_b.ndim == 2:
        tensor_b = tensor_b.unsqueeze(0)
    translation = torch.linalg.norm(tensor_a[:, :3, 3] - tensor_b[:, :3, 3], dim=-1)
    _, residual = pose_error(tensor_a, tensor_b)
    rotation = residual.norm(dim=-1)
    if rotation_symmetry == "half_turn_about_z":
        symmetric_b = tensor_b.clone()
        symmetric_b[:, :3, :2] = -symmetric_b[:, :3, :2]
        _, residual = pose_error(tensor_a, symmetric_b)
        symmetric_rotation = residual.norm(dim=-1)
        rotation = torch.minimum(rotation, symmetric_rotation)
    elif rotation_symmetry is not None:
        raise ValueError(
            "rotation_symmetry must be None or 'half_turn_about_z', "
            f"got {rotation_symmetry!r}."
        )
    return float(translation.mean().item()), float(rotation.mean().item())


def match_ordered_waypoints(
    trajectory_poses: torch.Tensor,
    waypoints: torch.Tensor,
    *,
    position_threshold_m: float,
    rotation_threshold_rad: float,
    rotation_symmetry: str | None = None,
) -> dict[str, object]:
    """Evaluate ordered arrival and threshold-constrained waypoint errors.

    Success and continuous waypoint errors share the same greedy matching:
    each waypoint must be hit after the previous arrival by a sample that
    jointly satisfies the position and rotation thresholds. Reported errors
    are taken at those arrival samples so ``motion_valid`` cannot disagree
    with waypoint p95/max exceeding the external thresholds.
    """
    trajectory_poses = torch.as_tensor(trajectory_poses)
    waypoints = torch.as_tensor(waypoints)
    if trajectory_poses.numel() == 0 or waypoints.numel() == 0:
        return {
            "ordered_waypoints_reached": False,
            "completed_waypoint_ratio": 0.0,
            "arrival_indices": [],
            "position_errors_m": [],
            "rotation_errors_rad": [],
            "min_position_errors_m": [],
            "min_rotation_errors_rad": [],
            "min_rotation_errors_at_position_rad": [],
            "min_position_errors_at_orientation_m": [],
        }

    pos_error, rot_error = _pose_error_matrices(
        waypoints,
        trajectory_poses,
        rotation_symmetry=rotation_symmetry,
    )
    arrival_indices: list[int] = []
    next_sample = 0
    for waypoint_index in range(waypoints.shape[0]):
        valid = torch.nonzero(
            (pos_error[waypoint_index, next_sample:] <= position_threshold_m)
            & (rot_error[waypoint_index, next_sample:] <= rotation_threshold_rad),
            as_tuple=False,
        ).flatten()
        if valid.numel() == 0:
            break
        sample_index = next_sample + int(valid[0].item())
        arrival_indices.append(sample_index)
        next_sample = sample_index + 1

    position_errors = [
        float(pos_error[index, sample].item())
        for index, sample in enumerate(arrival_indices)
    ]
    rotation_errors = [
        float(rot_error[index, sample].item())
        for index, sample in enumerate(arrival_indices)
    ]
    completed = len(arrival_indices)
    total = int(waypoints.shape[0])
    min_rotation_at_position: list[float | None] = []
    min_position_at_orientation: list[float | None] = []
    for waypoint_index in range(total):
        position_hits = pos_error[waypoint_index] <= position_threshold_m
        rotation_hits = rot_error[waypoint_index] <= rotation_threshold_rad
        min_rotation_at_position.append(
            float(rot_error[waypoint_index, position_hits].min().item())
            if bool(position_hits.any().item())
            else None
        )
        min_position_at_orientation.append(
            float(pos_error[waypoint_index, rotation_hits].min().item())
            if bool(rotation_hits.any().item())
            else None
        )
    return {
        "ordered_waypoints_reached": completed == total,
        "completed_waypoint_ratio": completed / max(total, 1),
        "arrival_indices": arrival_indices,
        "position_errors_m": position_errors,
        "rotation_errors_rad": rotation_errors,
        "min_position_errors_m": [
            float(value) for value in pos_error.min(dim=1).values.tolist()
        ],
        "min_rotation_errors_rad": [
            float(value) for value in rot_error.min(dim=1).values.tolist()
        ],
        "min_rotation_errors_at_position_rad": min_rotation_at_position,
        "min_position_errors_at_orientation_m": min_position_at_orientation,
    }


def match_ordered_joint_waypoints(
    trajectory_qpos: torch.Tensor,
    waypoints_qpos: torch.Tensor,
    *,
    joint_threshold_rad: float,
) -> dict[str, object]:
    """Evaluate ordered joint-waypoint arrival with an L-infinity threshold.

    Each waypoint must be reached after the previous arrival by one trajectory
    sample whose maximum absolute per-joint error is within the task threshold.
    The function mirrors :func:`match_ordered_waypoints` while retaining joint
    task semantics instead of projecting the targets through FK.
    """
    trajectory_qpos = torch.as_tensor(trajectory_qpos, dtype=torch.float64)
    waypoints_qpos = torch.as_tensor(
        waypoints_qpos,
        dtype=torch.float64,
        device=trajectory_qpos.device,
    )
    if not math.isfinite(joint_threshold_rad) or joint_threshold_rad <= 0.0:
        raise ValueError("joint_threshold_rad must be finite and positive.")
    if trajectory_qpos.ndim != 2 or waypoints_qpos.ndim != 2:
        raise ValueError("Joint trajectories and waypoints must both be 2D.")
    if trajectory_qpos.shape[-1] != waypoints_qpos.shape[-1]:
        raise ValueError(
            "Joint trajectories and waypoints must have the same trailing DoF."
        )
    if trajectory_qpos.shape[0] == 0 or waypoints_qpos.shape[0] == 0:
        return {
            "ordered_waypoints_reached": False,
            "completed_waypoint_ratio": 0.0,
            "arrival_indices": [],
            "joint_errors_rad": [],
            "min_joint_errors_rad": [],
        }

    joint_error = torch.amax(
        torch.abs(waypoints_qpos[:, None, :] - trajectory_qpos[None, :, :]),
        dim=-1,
    )
    arrival_indices: list[int] = []
    next_sample = 0
    for waypoint_index in range(waypoints_qpos.shape[0]):
        valid = torch.nonzero(
            joint_error[waypoint_index, next_sample:] <= joint_threshold_rad,
            as_tuple=False,
        ).flatten()
        if valid.numel() == 0:
            break
        sample_index = next_sample + int(valid[0].item())
        arrival_indices.append(sample_index)
        next_sample = sample_index + 1

    completed = len(arrival_indices)
    total = int(waypoints_qpos.shape[0])
    return {
        "ordered_waypoints_reached": completed == total,
        "completed_waypoint_ratio": completed / max(total, 1),
        "arrival_indices": arrival_indices,
        "joint_errors_rad": [
            float(joint_error[index, sample].item())
            for index, sample in enumerate(arrival_indices)
        ],
        "min_joint_errors_rad": [
            float(value) for value in joint_error.min(dim=1).values.tolist()
        ],
    }


def compute_waypoint_errors(
    trajectory_poses: list[torch.Tensor] | torch.Tensor,
    waypoints: torch.Tensor,
    *,
    position_threshold_m: float = 0.01,
    rotation_threshold_rad: float = 0.1,
    rotation_symmetry: str | None = None,
) -> dict[str, float]:
    """Return ordered, same-sample waypoint errors for one trajectory."""
    if isinstance(trajectory_poses, list):
        trajectory_tensor = (
            torch.stack(trajectory_poses)
            if trajectory_poses
            else torch.empty((0, 4, 4))
        )
    else:
        trajectory_tensor = trajectory_poses
    matched = match_ordered_waypoints(
        trajectory_tensor,
        waypoints,
        position_threshold_m=position_threshold_m,
        rotation_threshold_rad=rotation_threshold_rad,
        rotation_symmetry=rotation_symmetry,
    )
    pos_mm = [float(value) * 1000.0 for value in matched["position_errors_m"]]
    rot_deg = [
        float(value) * 180.0 / math.pi for value in matched["rotation_errors_rad"]
    ]
    return {
        "mean_waypoint_pos_err_mm": (
            sum(pos_mm) / len(pos_mm) if pos_mm else float("inf")
        ),
        "max_waypoint_pos_err_mm": max(pos_mm) if pos_mm else float("inf"),
        "mean_waypoint_rot_err_deg": (
            sum(rot_deg) / len(rot_deg) if rot_deg else float("inf")
        ),
        "max_waypoint_rot_err_deg": max(rot_deg) if rot_deg else float("inf"),
    }


def _success_tensor(success: bool | torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize scalar or tensor planner success to a CPU bool vector."""
    if isinstance(success, torch.Tensor):
        values = success.detach().to(device="cpu", dtype=torch.bool).flatten()
        if values.numel() == 1 and batch_size > 1:
            values = values.expand(batch_size).clone()
        if values.numel() != batch_size:
            raise ValueError(
                f"PlanResult.success has {values.numel()} values for batch_size={batch_size}."
            )
        return values
    return torch.full((batch_size,), bool(success), dtype=torch.bool)


def _joint_limit_metrics(
    positions: torch.Tensor,
    limits: torch.Tensor,
    tolerance_rad: float,
) -> tuple[bool, float]:
    """Return whether limits are violated and the maximum normalized excess."""
    lower, upper = limits[:, 0], limits[:, 1]
    below = (lower - positions - tolerance_rad).clamp_min(0.0)
    above = (positions - upper - tolerance_rad).clamp_min(0.0)
    span = (upper - lower).clamp_min(1.0e-6)
    normalized = torch.maximum(below, above) / span
    maximum = float(normalized.max().item()) if normalized.numel() else 0.0
    return maximum > 0.0, maximum


def _utilization(
    values: torch.Tensor | None,
    limits: torch.Tensor | None,
    tolerance: float,
) -> tuple[float | None, float | None, bool | None]:
    """Return peak magnitude, peak utilization, and violation against a limit."""
    if values is None or values.numel() == 0:
        return None, None, None
    magnitude = values.abs()
    peak = float(magnitude.max().item())
    if limits is None:
        return peak, None, None
    utilization = float((magnitude / limits.clamp_min(1.0e-12)).max().item())
    return peak, utilization, utilization > 1.0 + tolerance


def _trim_hold_padding(
    positions: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop a trailing run of unchanged samples that elapse no time.

    A batched planner pads shorter rows out to the longest by repeating the
    final pose at a zero interval. Those samples are bookkeeping, not motion:
    differentiating through them produces a velocity step across zero elapsed
    time, which the next differentiation stage rejects outright. Removing them
    first keeps a legitimately padded batch evaluable.

    ``dt[0]`` is an arrival offset rather than a step, so a single-sample path
    is returned unchanged.
    """
    count = positions.shape[0]
    if count < 2:
        return positions, dt
    last = count
    while (
        last > 2
        and float(dt[last - 1]) == 0.0
        and bool(torch.equal(positions[last - 1], positions[last - 2]))
    ):
        last -= 1
    if last == count:
        return positions, dt
    return positions[:last], dt[:last]


def _moves_in_zero_time(positions: torch.Tensor, dt: torch.Tensor) -> bool:
    """Return whether the path changes position across a zero-length interval.

    ``dt[0]`` is an arrival offset rather than a step and is excluded.
    """
    steps = dt[1:]
    if steps.numel() == 0:
        return False
    changed = (positions.diff(dim=0) != 0).any(dim=-1)
    return bool(((steps == 0) & changed).any().item())


def _trajectory_derivatives(
    positions: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate one joint path into velocity, acceleration, and jerk.

    Derivatives are always recomputed here rather than read from
    :class:`PlanResult`, so every planner is measured with one operator on its
    own timing instead of with whatever convention it reported.

    The operator is compact divided differences: velocity is each step's mean
    rate, acceleration the change between adjacent step rates over the
    distance between step midpoints, and jerk the same again. A central
    difference averages each sample with both neighbors, and nesting it widens
    that window further; a path that reverses direction every step then reads
    as several times smoother than it is. The compact second difference equals
    the path's true acceleration at some instant inside its window, so it never
    overstates the peak and does not hide one either.

    Samples a zero interval after an unchanged predecessor are dropped first:
    they add no motion, and a position change across a zero interval is
    rejected by the caller before this point.

    Args:
        positions: One environment's joint path of shape ``(N, DOF)``.
        dt: That path's arrival intervals of shape ``(N,)``.

    Returns:
        Velocity ``(N-1, DOF)``, acceleration ``(N-2, DOF)`` and jerk
        ``(N-3, DOF)`` after dropping zero-interval samples; a derivative the
        path is too short to resolve is empty.
    """
    keep = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
    keep[1:] = dt[1:] > 0
    positions = positions[keep]
    steps = dt[keep][1:].to(positions.dtype)
    velocity = positions.diff(dim=0) / steps.unsqueeze(-1)
    midpoint_gaps = 0.5 * (steps[1:] + steps[:-1])
    acceleration = velocity.diff(dim=0) / midpoint_gaps.unsqueeze(-1)
    # A compact acceleration estimate is located at the mean of its three
    # position-sample times. Consecutive estimates are therefore separated by
    # one third of the three intervening step lengths.
    centre_gaps = (steps[:-2] + steps[1:-1] + steps[2:]) / 3.0
    jerk = acceleration.diff(dim=0) / centre_gaps.unsqueeze(-1)
    return velocity, acceleration, jerk


def _dynamic_limit_metrics(
    positions: torch.Tensor | None,
    dt: torch.Tensor | None,
    qvel_limits: torch.Tensor | None,
    qacc_limits: torch.Tensor | None,
    qjerk_limits: torch.Tensor | None,
    tolerance: float,
    timing_is_solved: bool = True,
) -> dict[str, float | bool | str | None]:
    """Return peak dynamics, limit utilization, and violation per derivative.

    Every entry is ``None`` when the trajectory is unusable or the matching
    limit is unconfigured. Acceleration and jerk limits have no asset source,
    so an unset limit is reported as not applicable rather than as satisfied.

    The per-derivative peaks, utilizations and violations always describe the
    timing the planner reported. ``dynamic_limits_satisfied`` is the verdict
    that feeds ``motion_valid``, and it additionally requires that timing to be
    solved: a planner that reports a nominal constant has not claimed an
    executable duration, so its trajectory is neither credited nor failed on a
    clock it never solved. Those rows still publish their diagnostics, which is
    what makes a nominal and a retimed run of the same policy comparable.
    """
    if positions is not None and dt is not None:
        positions, dt = _trim_hold_padding(positions, dt)
    if positions is None or dt is None or positions.shape[0] < 2:
        velocity = acceleration = jerk = None
    elif _moves_in_zero_time(positions, dt):
        # A position change across a zero interval is an unbounded velocity, so
        # report it as a definite violation instead of letting the
        # differentiator reject the trajectory and fail the whole trial.
        # Each limit check is reported only where that limit is configured,
        # exactly as on the ordinary path; with no limit at all there is
        # nothing for the verdict to decide.
        limited = [
            limit is not None for limit in (qvel_limits, qacc_limits, qjerk_limits)
        ]
        return {
            "max_joint_velocity_rad_s": math.inf,
            "velocity_utilization": math.inf if limited[0] else None,
            "velocity_limit_violation": True if limited[0] else None,
            "max_joint_acceleration_rad_s2": math.inf,
            "acceleration_utilization": math.inf if limited[1] else None,
            "acceleration_limit_violation": True if limited[1] else None,
            "max_joint_jerk_rad_s3": math.inf,
            "jerk_utilization": math.inf if limited[2] else None,
            "jerk_limit_violation": True if limited[2] else None,
            "dynamic_measurement_status": "unbounded",
            "dynamic_limits_satisfied": (
                False if (any(limited) and timing_is_solved) else None
            ),
        }
    else:
        velocity, acceleration, jerk = _trajectory_derivatives(positions, dt)
    peak_velocity, velocity_utilization, velocity_violation = _utilization(
        velocity, qvel_limits, tolerance
    )
    peak_acceleration, acceleration_utilization, acceleration_violation = _utilization(
        acceleration, qacc_limits, tolerance
    )
    peak_jerk, jerk_utilization, jerk_violation = _utilization(
        jerk, qjerk_limits, tolerance
    )
    checks = [
        check
        for check in (velocity_violation, acceleration_violation, jerk_violation)
        if check is not None
    ]
    return {
        "max_joint_velocity_rad_s": peak_velocity,
        "velocity_utilization": velocity_utilization,
        "velocity_limit_violation": velocity_violation,
        "max_joint_acceleration_rad_s2": peak_acceleration,
        "acceleration_utilization": acceleration_utilization,
        "acceleration_limit_violation": acceleration_violation,
        "max_joint_jerk_rad_s3": peak_jerk,
        "jerk_utilization": jerk_utilization,
        "jerk_limit_violation": jerk_violation,
        "dynamic_measurement_status": (
            "finite" if velocity is not None else "unavailable"
        ),
        "dynamic_limits_satisfied": (
            (not any(checks)) if (checks and timing_is_solved) else None
        ),
    }


def _path_metrics(
    qpos: torch.Tensor,
    poses: torch.Tensor,
    waypoints: torch.Tensor,
) -> tuple[float, float, float]:
    """Return joint length, Cartesian translation length, and path efficiency."""
    joint_length = float(torch.linalg.norm(qpos[1:] - qpos[:-1], dim=-1).sum().item())
    translations = poses[:, :3, 3]
    cartesian_length = float(
        torch.linalg.norm(translations[1:] - translations[:-1], dim=-1).sum().item()
    )
    anchors = torch.cat([poses[:1, :3, 3], waypoints[:, :3, 3]], dim=0)
    lower_bound = float(
        torch.linalg.norm(anchors[1:] - anchors[:-1], dim=-1).sum().item()
    )
    efficiency = (
        min(1.0, lower_bound / cartesian_length) if cartesian_length > 1.0e-12 else 1.0
    )
    return joint_length, cartesian_length, efficiency


def make_failure_outcomes(
    batch_size: int,
    failure_code: str,
    *,
    planner_failure_code: str | None = None,
    planning_success: bool = False,
    timing_is_solved: bool | None = None,
) -> tuple[CaseOutcome, ...]:
    """Create per-env outcomes when validation cannot produce a trajectory."""
    return tuple(
        CaseOutcome(
            env_index=index,
            planning_success=planning_success,
            finite=False,
            ordered_waypoints_reached=False,
            motion_valid=False,
            timed_motion_valid=False if timing_is_solved else None,
            completed_waypoint_ratio=0.0,
            final_translation_err_mm=None,
            final_rotation_err_deg=None,
            waypoint_translation_err_mm_mean=None,
            waypoint_translation_err_mm_p95=None,
            waypoint_translation_err_mm_max=None,
            waypoint_rotation_err_deg_mean=None,
            waypoint_rotation_err_deg_p95=None,
            waypoint_rotation_err_deg_max=None,
            joint_limit_violation=False,
            max_normalized_joint_violation=None,
            joint_path_length_rad=None,
            cartesian_path_length_m=None,
            path_efficiency=None,
            failure_code=failure_code,
            planner_failure_code=planner_failure_code,
        )
        for index in range(batch_size)
    )


def _reference_path_deviation(
    positions: torch.Tensor,
    anchors: torch.Tensor,
) -> tuple[float | None, float | None]:
    """Measure sampled cross-track error with bounded point/segment chunks.

    Work is O(N W) and temporary storage O(128 squared), including long custom
    waypoint lists. Zero-length polyline segments are treated as points.
    """
    if positions.numel() == 0 or anchors.shape[0] < 2:
        return None, None
    distances = []
    for points in positions.split(128):
        best = torch.full(
            (len(points),), torch.inf, device=points.device, dtype=points.dtype
        )
        for start, end in zip(anchors[:-1].split(128), anchors[1:].split(128)):
            segment = end - start
            offset = points[:, None] - start[None]
            alpha = (offset * segment).sum(-1) / segment.square().sum(-1).clamp_min(
                1.0e-20
            )
            residual = offset - alpha.clamp(0, 1)[..., None] * segment
            best = torch.minimum(best, residual.norm(dim=-1).amin(-1))
        distances.append(best)
    values = torch.cat(distances) * 1000
    return float(values.mean()), float(values.max())


def _policy_step_metrics(
    result: PlanResult,
    case: BenchmarkCase,
    robot: "Robot",
    control_part: str,
    *,
    position_threshold_m: float,
    rotation_threshold_rad: float,
    start_state_tolerance_rad: float,
) -> list[dict[str, object]]:
    """Match constraints on original policy controls, independently of timing."""
    from ._constraints import match_constraints

    rollout = result.policy_rollout_positions
    if rollout is None:
        return [{} for _ in range(case.batch_size)]
    lengths = result.policy_rollout_lengths
    assert lengths is not None  # PlanResult validates the paired trace fields.
    rollout = rollout.to(robot.device)
    poses = robot.compute_batch_fk(qpos=rollout, name=control_part, to_matrix=True)
    modalities = case.case_parameters.get("constraint_modalities")
    if modalities is None:
        joint = case.case_parameters.get("motion_validity") == "ordered_joint_waypoints"
        modalities = ["joint" if joint else "pose"] * case.num_waypoints
    rows: list[dict[str, object]] = []
    for row, length in enumerate(lengths.tolist()):
        qpos = rollout[row, :length]
        valid = bool(torch.isfinite(qpos).all()) and bool(
            (qpos[0] - case.start_qpos[row].to(qpos)).abs().max()
            <= start_state_tolerance_rad
        )
        matching = match_constraints(
            qpos if valid else qpos[:0],
            poses[row, :length] if valid else poses[row, :0],
            case.reference_qpos[row],
            case.target_waypoints[row],
            modalities,
            position_threshold_m=position_threshold_m,
            rotation_threshold_rad=rotation_threshold_rad,
            joint_threshold_rad=float(
                case.case_parameters.get("joint_threshold_rad", 0.01)
            ),
        )
        arrivals = matching["arrival_indices"]
        rows.append(
            {
                "policy_steps_used": length - 1,
                "policy_completion_step": (
                    arrivals[-1] if matching["ordered_waypoints_reached"] else None
                ),
                "policy_waypoint_arrival_steps": tuple(arrivals)
                + (None,) * (case.num_waypoints - len(arrivals)),
            }
        )
    return rows


def compute_case_outcomes(
    result: PlanResult,
    case: BenchmarkCase,
    robot: "Robot",
    control_part: str,
    *,
    validation_samples: int,
    position_threshold_m: float,
    rotation_threshold_rad: float,
    joint_limit_tolerance_rad: float,
    start_state_tolerance_rad: float = 1.0e-3,
    joint_velocity_limit_rad_s: float | Sequence[float] | None = None,
    joint_acceleration_limit_rad_s2: float | Sequence[float] | None = None,
    joint_jerk_limit_rad_s3: float | Sequence[float] | None = None,
    dynamic_limit_tolerance: float = 1.0e-3,
    timing_is_solved: bool = True,
) -> tuple[CaseOutcome, ...]:
    """Recompute free-space success and quality from a planner trajectory."""
    if not math.isfinite(start_state_tolerance_rad) or start_state_tolerance_rad < 0:
        raise ValueError("start_state_tolerance_rad must be finite and non-negative.")
    planning_success = _success_tensor(result.success, case.batch_size)
    policy_metrics = _policy_step_metrics(
        result,
        case,
        robot,
        control_part,
        position_threshold_m=position_threshold_m,
        rotation_threshold_rad=rotation_threshold_rad,
        start_state_tolerance_rad=start_state_tolerance_rad,
    )
    if result.positions is None or result.positions.ndim != 3:
        return tuple(
            replace(
                make_failure_outcomes(
                    1,
                    "non_finite_trajectory",
                    timing_is_solved=timing_is_solved,
                    planning_success=bool(planning_success[env_index].item()),
                    planner_failure_code=(
                        None
                        if bool(planning_success[env_index].item())
                        else "planner_reported_failure"
                    ),
                )[0],
                env_index=env_index,
                **policy_metrics[env_index],
            )
            for env_index in range(case.batch_size)
        )

    positions = result.positions.to(robot.device)
    if positions.shape[0] != case.batch_size:
        raise ValueError(
            f"PlanResult.positions batch={positions.shape[0]} does not match "
            f"case batch={case.batch_size}."
        )
    if case.start_qpos.shape != (case.batch_size, positions.shape[-1]):
        raise ValueError("case.start_qpos must match the trajectory batch and joints.")
    limits = robot.get_qpos_limits(name=control_part)
    if limits.ndim == 2:
        limits = limits.unsqueeze(0)
    if limits.shape[0] == 1:
        limits = limits.expand(case.batch_size, -1, -1)
    if limits.shape != (case.batch_size, positions.shape[-1], 2):
        raise ValueError("Joint limits must match trajectory environments and joints.")
    dof = positions.shape[-1]
    qvel_limits = resolve_robot_velocity_limits(
        robot, control_part, joint_velocity_limit_rad_s, positions, dof=dof
    )
    qacc_limits = resolve_dynamic_limit(joint_acceleration_limit_rad_s2, dof, positions)
    qjerk_limits = resolve_dynamic_limit(joint_jerk_limit_rad_s3, dof, positions)
    sample_dt = None if result.dt is None else result.dt.to(positions.device)
    finite_mask = torch.isfinite(positions).all(dim=(1, 2)) & (positions.shape[1] > 0)
    finite_paths = finite_mask.tolist()
    safe_positions = (
        torch.where(
            finite_mask[:, None, None],
            positions,
            case.start_qpos[:, None].to(positions),
        )
        if positions.shape[1]
        else case.start_qpos[:, None].to(positions)
    )
    validation_qpos_batch = resample_with_distance(
        safe_positions, validation_samples, device=positions.device, backend="torch"
    )
    validation_pose_batch = robot.compute_batch_fk(
        qpos=validation_qpos_batch,
        name=control_part,
        to_matrix=True,
    )
    constraint_qpos_batch = None
    constraint_pose_batch = None
    if "constraint_modalities" in case.case_parameters:
        # Every native state survives; deterministic linear joint subdivisions
        # supply at least validation_samples states for sparse planner outputs.
        count = safe_positions.shape[1]
        subdivisions = max(1, math.ceil((validation_samples - 1) / max(1, count - 1)))
        if count > 1:
            parameter = (
                torch.arange(
                    (count - 1) * subdivisions + 1,
                    device=safe_positions.device,
                    dtype=safe_positions.dtype,
                )
                / subdivisions
            )
            left = parameter.long().clamp(max=count - 2)
            fraction = (parameter - left)[None, :, None]
            constraint_qpos_batch = torch.lerp(
                safe_positions[:, left], safe_positions[:, left + 1], fraction
            )
        else:
            constraint_qpos_batch = safe_positions
        constraint_pose_batch = robot.compute_batch_fk(
            qpos=constraint_qpos_batch, name=control_part, to_matrix=True
        )
    native_pose_batch = robot.compute_batch_fk(
        qpos=positions,
        name=control_part,
        to_matrix=True,
    )
    reference_start_poses = (
        robot.compute_fk(qpos=case.start_qpos, name=control_part, to_matrix=True)
        if case.case_parameters.get("path_shape_semantics")
        == "cartesian_waypoint_layout"
        and case.case_parameters.get("motion_validity", "ordered_cartesian_waypoints")
        == "ordered_cartesian_waypoints"
        and case.case_parameters.get("motion_component") != "rotation"
        else None
    )
    outcomes: list[CaseOutcome] = []
    for env_index in range(case.batch_size):
        native_qpos = positions[env_index]
        finite = finite_paths[env_index]
        if finite:
            validation_qpos = validation_qpos_batch[env_index]
            validation_poses = validation_pose_batch[env_index]
            # Keep the benchmark's established, sample-count-normalized
            # trajectory for all success and path metrics.  Native planner
            # samples are retained below only for rollout diagnostics.
            poses = validation_poses
            native_poses = native_pose_batch[env_index]
        else:
            validation_qpos = torch.empty(
                (0, positions.shape[-1]), device=robot.device, dtype=positions.dtype
            )
            validation_poses = torch.empty(
                (0, 4, 4), device=robot.device, dtype=positions.dtype
            )
            poses = torch.empty((0, 4, 4), device=robot.device, dtype=positions.dtype)
            native_poses = torch.empty(
                (0, 4, 4), device=robot.device, dtype=positions.dtype
            )

        waypoints = case.target_waypoints[env_index]
        rotation_symmetry = case.case_parameters.get("waypoint_rotation_symmetry")
        if rotation_symmetry is not None and not isinstance(rotation_symmetry, str):
            raise TypeError("waypoint_rotation_symmetry must be a string or None.")
        cartesian_matching = match_ordered_waypoints(
            poses,
            waypoints,
            position_threshold_m=position_threshold_m,
            rotation_threshold_rad=rotation_threshold_rad,
            rotation_symmetry=rotation_symmetry,
        )
        pos_errors_mm = [
            float(value) * 1000.0 for value in cartesian_matching["position_errors_m"]
        ]
        rot_errors_deg = [
            float(value) * 180.0 / math.pi
            for value in cartesian_matching["rotation_errors_rad"]
        ]
        if finite:
            joint_violation, normalized_violation = _joint_limit_metrics(
                native_qpos,
                limits[env_index],
                joint_limit_tolerance_rad,
            )
        else:
            joint_violation, normalized_violation = False, None
        if finite:
            dynamic = _dynamic_limit_metrics(
                native_qpos,
                None if sample_dt is None else sample_dt[env_index],
                qvel_limits,
                qacc_limits,
                qjerk_limits,
                dynamic_limit_tolerance,
                timing_is_solved,
            )
        else:
            dynamic = _dynamic_limit_metrics(
                None, None, None, None, None, dynamic_limit_tolerance, timing_is_solved
            )
        validity_space = case.case_parameters.get(
            "motion_validity", "ordered_cartesian_waypoints"
        )
        if case.case_parameters.get("joint_error_norm", "linf") != "linf":
            raise ValueError(
                "case joint_error_norm conflicts with the linf scoring protocol"
            )
        if "constraint_modalities" in case.case_parameters:
            from ._constraints import match_constraints

            semantic_matching = match_constraints(
                constraint_qpos_batch[env_index] if finite else native_qpos[:0],
                constraint_pose_batch[env_index] if finite else native_poses,
                case.reference_qpos[env_index],
                waypoints,
                case.case_parameters["constraint_modalities"],
                position_threshold_m=position_threshold_m,
                rotation_threshold_rad=rotation_threshold_rad,
                joint_threshold_rad=float(case.case_parameters["joint_threshold_rad"]),
            )
        elif validity_space == "ordered_joint_waypoints":
            threshold = case.case_parameters.get("joint_threshold_rad")
            if not isinstance(threshold, (int, float)):
                raise TypeError(
                    "ordered_joint_waypoints requires numeric joint_threshold_rad."
                )
            semantic_matching = match_ordered_joint_waypoints(
                validation_qpos,
                case.reference_qpos[env_index],
                joint_threshold_rad=float(threshold),
            )
        elif validity_space == "ordered_cartesian_waypoints":
            semantic_matching = cartesian_matching
        else:
            raise ValueError(f"Unknown motion_validity mode {validity_space!r}.")
        if "constraint_modalities" in case.case_parameters:
            semantic_matching["sample_grid"] = "native_with_linear_subdivision"
            arrivals = semantic_matching["arrival_errors"]
            pos_errors_mm = [
                item["position_m"] * 1000.0 for item in arrivals if "position_m" in item
            ]
            rot_errors_deg = [
                item["rotation_rad"] * 180.0 / math.pi
                for item in arrivals
                if "rotation_rad" in item
            ]
        ordered = bool(semantic_matching["ordered_waypoints_reached"])
        deviation_mean = deviation_max = None
        if (
            finite
            and validity_space == "ordered_cartesian_waypoints"
            and case.case_parameters.get("path_shape_semantics")
            == "cartesian_waypoint_layout"
            and case.case_parameters.get("motion_component") != "rotation"
        ):
            initial = reference_start_poses[env_index, :3, 3]
            anchors = torch.cat(
                (
                    initial[None].to(validation_poses),
                    waypoints[:, :3, 3].to(validation_poses),
                ),
                dim=0,
            )
            deviation_mean, deviation_max = _reference_path_deviation(
                validation_poses[:, :3, 3], anchors
            )
        start_error = (
            float(
                (native_qpos[0] - case.start_qpos[env_index].to(native_qpos))
                .abs()
                .max()
            )
            if finite and bool(torch.isfinite(case.start_qpos[env_index]).all())
            else None
        )
        start_valid = (
            start_error is not None and start_error <= start_state_tolerance_rad
        )
        planner_ok = bool(planning_success[env_index].item())
        # ``PlanResult.success`` is retained as a planner-stage outcome, but it
        # is not external ground truth.  A trajectory can therefore be motion
        # valid even when a backend conservatively reports planning failure.
        # ``dynamic_limits_satisfied`` is ``None`` when no limit applies, which
        # leaves ``motion_valid`` unchanged rather than failing the trajectory.
        geometry_valid = finite and ordered and not joint_violation and start_valid
        motion_valid = (
            geometry_valid and dynamic["dynamic_limits_satisfied"] is not False
        )
        timing_valid = (
            sample_dt is not None
            and bool(torch.isfinite(sample_dt[env_index]).all().item())
            and bool((sample_dt[env_index] >= 0.0).all().item())
            and not _moves_in_zero_time(native_qpos, sample_dt[env_index])
        )
        timed_motion_valid = (
            bool(
                geometry_valid
                and timing_valid
                and dynamic["dynamic_limits_satisfied"] is not False
            )
            if timing_is_solved
            else None
        )

        if poses.shape[0] > 0:
            final_pos_m, final_rot_rad = get_pose_err(
                poses[-1],
                waypoints[-1],
                rotation_symmetry=rotation_symmetry,
            )
            all_pos_error, all_rot_error = _pose_error_matrices(
                waypoints,
                native_poses,
                rotation_symmetry=rotation_symmetry,
            )
            min_pos_m = float(all_pos_error[-1].min().item())
            min_rot_rad = float(all_rot_error[-1].min().item())
            joint_length, cartesian_length, efficiency = _path_metrics(
                validation_qpos, validation_poses, waypoints
            )
            trajectory_moved = bool(
                torch.linalg.vector_norm(native_qpos - native_qpos[:1], dim=-1).max()
                > 1.0e-6
            )
        else:
            final_pos_m = final_rot_rad = None
            min_pos_m = min_rot_rad = None
            joint_length = cartesian_length = efficiency = None
            trajectory_moved = False

        if "constraint_modalities" in case.case_parameters:
            last_modality = case.case_parameters["constraint_modalities"][-1]
            if last_modality not in {"position", "pose"}:
                final_pos_m = min_pos_m = None
            if last_modality not in {"orientation", "pose"}:
                final_rot_rad = min_rot_rad = None
        if not finite:
            failure_code = "non_finite_trajectory"
        elif not start_valid:
            failure_code = "start_state_mismatch"
        elif not ordered:
            failure_code = "waypoint_miss"
        elif joint_violation:
            failure_code = "joint_limit_violation"
        elif dynamic["dynamic_limits_satisfied"] is False:
            failure_code = "dynamic_limit_violation"
        else:
            failure_code = None
        planner_failure_code = None if planner_ok else "planner_reported_failure"

        outcomes.append(
            CaseOutcome(
                env_index=env_index,
                planning_success=planner_ok,
                **policy_metrics[env_index],
                constraint_metrics=(
                    semantic_matching
                    if "constraint_modalities" in case.case_parameters
                    else {}
                ),
                start_state_valid=start_valid,
                reference_path_deviation_mm_mean=deviation_mean,
                reference_path_deviation_mm_max=deviation_max,
                start_joint_error_rad=start_error,
                finite=finite,
                ordered_waypoints_reached=ordered,
                motion_valid=motion_valid,
                timed_motion_valid=timed_motion_valid,
                completed_waypoint_ratio=float(
                    semantic_matching["completed_waypoint_ratio"]
                ),
                final_translation_err_mm=(
                    final_pos_m * 1000.0 if final_pos_m is not None else None
                ),
                final_rotation_err_deg=(
                    final_rot_rad * 180.0 / math.pi
                    if final_rot_rad is not None
                    else None
                ),
                waypoint_translation_err_mm_mean=(
                    sum(pos_errors_mm) / len(pos_errors_mm) if pos_errors_mm else None
                ),
                waypoint_translation_err_mm_p95=nearest_rank_percentile(
                    pos_errors_mm, 95.0
                ),
                waypoint_translation_err_mm_max=(
                    max(pos_errors_mm) if pos_errors_mm else None
                ),
                waypoint_rotation_err_deg_mean=(
                    sum(rot_errors_deg) / len(rot_errors_deg)
                    if rot_errors_deg
                    else None
                ),
                waypoint_rotation_err_deg_p95=nearest_rank_percentile(
                    rot_errors_deg, 95.0
                ),
                waypoint_rotation_err_deg_max=(
                    max(rot_errors_deg) if rot_errors_deg else None
                ),
                joint_limit_violation=joint_violation,
                max_normalized_joint_violation=normalized_violation,
                **dynamic,
                joint_path_length_rad=joint_length,
                cartesian_path_length_m=cartesian_length,
                path_efficiency=efficiency,
                failure_code=failure_code,
                planner_failure_code=planner_failure_code,
                min_translation_err_mm=(
                    min_pos_m * 1000.0 if min_pos_m is not None else None
                ),
                min_rotation_err_deg=(
                    min_rot_rad * 180.0 / math.pi if min_rot_rad is not None else None
                ),
                trajectory_moved=trajectory_moved,
                waypoint_min_translation_err_mm=tuple(
                    float(value) * 1000.0
                    for value in cartesian_matching["min_position_errors_m"]
                ),
                waypoint_min_rotation_err_deg=tuple(
                    float(value) * 180.0 / math.pi
                    for value in cartesian_matching["min_rotation_errors_rad"]
                ),
                waypoint_min_rotation_err_deg_at_position=tuple(
                    None if value is None else float(value) * 180.0 / math.pi
                    for value in cartesian_matching[
                        "min_rotation_errors_at_position_rad"
                    ]
                ),
                waypoint_min_translation_err_mm_at_orientation=tuple(
                    None if value is None else float(value) * 1000.0
                    for value in cartesian_matching[
                        "min_position_errors_at_orientation_m"
                    ]
                ),
            )
        )
    return tuple(outcomes)
