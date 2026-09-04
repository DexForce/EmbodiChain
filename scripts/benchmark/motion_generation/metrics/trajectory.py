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
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.planners.utils import PlanResult

from ..models import BenchmarkCase, CaseOutcome
from .stats import nearest_rank_percentile

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = [
    "compute_case_outcomes",
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
    waypoint_rot = waypoints[:, None, :3, :3]
    trajectory_rot = trajectory_poses[None, :, :3, :3]
    relative = waypoint_rot.transpose(-1, -2) @ trajectory_rot
    trace = torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1)
    rotation_error = torch.arccos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    if rotation_symmetry == "half_turn_about_z":
        symmetric_rot = waypoint_rot.clone()
        symmetric_rot[..., :2] = -symmetric_rot[..., :2]
        symmetric_relative = symmetric_rot.transpose(-1, -2) @ trajectory_rot
        symmetric_trace = torch.diagonal(symmetric_relative, dim1=-2, dim2=-1).sum(
            dim=-1
        )
        symmetric_error = torch.arccos(
            torch.clamp((symmetric_trace - 1.0) * 0.5, -1.0, 1.0)
        )
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
    relative = tensor_a[:, :3, :3].transpose(-1, -2) @ tensor_b[:, :3, :3]
    trace = torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1)
    rotation = torch.arccos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    if rotation_symmetry == "half_turn_about_z":
        symmetric_b = tensor_b.clone()
        symmetric_b[:, :3, :2] = -symmetric_b[:, :3, :2]
        symmetric_relative = (
            tensor_a[:, :3, :3].transpose(-1, -2) @ symmetric_b[:, :3, :3]
        )
        symmetric_trace = torch.diagonal(symmetric_relative, dim1=-2, dim2=-1).sum(
            dim=-1
        )
        symmetric_rotation = torch.arccos(
            torch.clamp((symmetric_trace - 1.0) * 0.5, -1.0, 1.0)
        )
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


def _resample_joint_path(positions: torch.Tensor, sample_count: int) -> torch.Tensor:
    """Resample one joint path at uniform cumulative joint-arc length."""
    positions = torch.as_tensor(positions)
    if positions.shape[0] == 0:
        return positions
    if positions.shape[0] == 1 or sample_count <= 1:
        return positions[:1]
    segment_length = torch.linalg.norm(positions[1:] - positions[:-1], dim=-1)
    cumulative = torch.cat(
        [
            torch.zeros(1, device=positions.device, dtype=positions.dtype),
            segment_length.cumsum(0),
        ]
    )
    total = cumulative[-1]
    if float(total.item()) <= 1.0e-12:
        return positions[:1].expand(sample_count, -1).clone()
    targets = torch.linspace(
        0.0,
        float(total.item()),
        sample_count,
        device=positions.device,
        dtype=positions.dtype,
    )
    right = torch.searchsorted(cumulative, targets, right=True).clamp(
        min=1, max=positions.shape[0] - 1
    )
    left = right - 1
    denominator = (cumulative[right] - cumulative[left]).clamp_min(1.0e-12)
    ratio = ((targets - cumulative[left]) / denominator).unsqueeze(-1)
    return positions[left] + ratio * (positions[right] - positions[left])


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
) -> tuple[CaseOutcome, ...]:
    """Create per-env outcomes when validation cannot produce a trajectory."""
    return tuple(
        CaseOutcome(
            env_index=index,
            planning_success=planning_success,
            finite=False,
            ordered_waypoints_reached=False,
            motion_valid=False,
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
) -> tuple[CaseOutcome, ...]:
    """Recompute free-space success and quality from a planner trajectory."""
    planning_success = _success_tensor(result.success, case.batch_size)
    if result.positions is None or result.positions.ndim != 3:
        return tuple(
            replace(
                make_failure_outcomes(
                    1,
                    "non_finite_trajectory",
                    planning_success=bool(planning_success[env_index].item()),
                    planner_failure_code=(
                        None
                        if bool(planning_success[env_index].item())
                        else "planner_reported_failure"
                    ),
                )[0],
                env_index=env_index,
            )
            for env_index in range(case.batch_size)
        )

    positions = result.positions.to(robot.device)
    if positions.shape[0] != case.batch_size:
        raise ValueError(
            f"PlanResult.positions batch={positions.shape[0]} does not match "
            f"case batch={case.batch_size}."
        )
    limits = robot.get_qpos_limits(name=control_part)
    finite_paths = [
        bool(torch.isfinite(positions[env_index]).all().item())
        and positions[env_index].shape[0] > 0
        for env_index in range(case.batch_size)
    ]
    validation_paths = [
        (
            _resample_joint_path(positions[env_index], validation_samples)
            if finite_paths[env_index]
            else case.start_qpos[env_index : env_index + 1]
            .expand(validation_samples, -1)
            .clone()
        )
        for env_index in range(case.batch_size)
    ]
    validation_qpos_batch = torch.stack(validation_paths)
    validation_pose_batch = robot.compute_batch_fk(
        qpos=validation_qpos_batch,
        name=control_part,
        to_matrix=True,
    )
    native_pose_batch = robot.compute_batch_fk(
        qpos=positions,
        name=control_part,
        to_matrix=True,
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
        validity_space = case.case_parameters.get(
            "motion_validity", "ordered_cartesian_waypoints"
        )
        if validity_space == "ordered_joint_waypoints":
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
        ordered = bool(semantic_matching["ordered_waypoints_reached"])
        planner_ok = bool(planning_success[env_index].item())
        # ``PlanResult.success`` is retained as a planner-stage outcome, but it
        # is not external ground truth.  A trajectory can therefore be motion
        # valid even when a backend conservatively reports planning failure.
        motion_valid = finite and ordered and not joint_violation

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

        if not finite:
            failure_code = "non_finite_trajectory"
        elif not ordered:
            failure_code = "waypoint_miss"
        elif joint_violation:
            failure_code = "joint_limit_violation"
        else:
            failure_code = None
        planner_failure_code = None if planner_ok else "planner_reported_failure"

        outcomes.append(
            CaseOutcome(
                env_index=env_index,
                planning_success=planner_ok,
                finite=finite,
                ordered_waypoints_reached=ordered,
                motion_valid=motion_valid,
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
