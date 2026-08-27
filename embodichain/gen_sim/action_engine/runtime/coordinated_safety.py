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

"""Pure dual-arm grasp and trajectory safety checks owned by GenSim."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _CanonicalizedGraspPoses:
    poses: torch.Tensor
    flipped: torch.Tensor
    selected_rotation_radians: torch.Tensor
    alternative_rotation_radians: torch.Tensor


@dataclass(frozen=True, slots=True)
class _RankedGraspPairs:
    ranked_pairs: tuple[tuple[int, int], ...]
    scores: tuple[float, ...]
    rejection_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _TrajectorySafetyReport:
    success: torch.Tensor
    failed_checks: dict[str, list[bool]]
    metrics: dict[str, list[float]]


def _rotation_distance(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> torch.Tensor:
    relative = torch.matmul(actual.transpose(-1, -2), expected)
    cosine = (torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    return torch.acos(torch.clamp(cosine, -1.0, 1.0))


def _canonicalize_parallel_jaw_poses(
    poses: torch.Tensor,
    live_eef_pose: torch.Tensor,
) -> _CanonicalizedGraspPoses:
    """Choose the local-Z half-turn equivalent nearest one live wrist pose."""
    poses = torch.as_tensor(poses, dtype=torch.float32)
    live = torch.as_tensor(live_eef_pose, dtype=poses.dtype, device=poses.device)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4).")
    if live.shape != (4, 4):
        raise ValueError("live_eef_pose must have shape (4, 4).")
    half_turn = torch.eye(4, dtype=poses.dtype, device=poses.device)
    half_turn[0, 0] = -1.0
    half_turn[1, 1] = -1.0
    alternatives = torch.matmul(poses, half_turn)
    live_rot = live[:3, :3].unsqueeze(0).expand(poses.shape[0], -1, -1)
    original_distance = _rotation_distance(live_rot, poses[:, :3, :3])
    alternative_distance = _rotation_distance(live_rot, alternatives[:, :3, :3])
    flipped = alternative_distance < original_distance
    selected = torch.where(flipped[:, None, None], alternatives, poses)
    selected_distance = torch.where(
        flipped,
        alternative_distance,
        original_distance,
    )
    rejected_distance = torch.where(
        flipped,
        original_distance,
        alternative_distance,
    )
    return _CanonicalizedGraspPoses(
        poses=selected,
        flipped=flipped,
        selected_rotation_radians=selected_distance,
        alternative_rotation_radians=rejected_distance,
    )


def _segments_intersect_2d(
    first_start: torch.Tensor,
    first_end: torch.Tensor,
    second_start: torch.Tensor,
    second_end: torch.Tensor,
) -> bool:
    epsilon = 1.0e-7

    def orientation(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def on_segment(a: torch.Tensor, b: torch.Tensor, point: torch.Tensor) -> bool:
        return bool(
            float(torch.min(a[0], b[0])) - epsilon
            <= float(point[0])
            <= float(torch.max(a[0], b[0])) + epsilon
            and float(torch.min(a[1], b[1])) - epsilon
            <= float(point[1])
            <= float(torch.max(a[1], b[1])) + epsilon
        )

    a = first_start[:2]
    b = first_end[:2]
    c = second_start[:2]
    d = second_end[:2]
    abc = orientation(a, b, c)
    abd = orientation(a, b, d)
    cda = orientation(c, d, a)
    cdb = orientation(c, d, b)
    if abc * abd < 0.0 and cda * cdb < 0.0:
        return True
    return (
        (abs(abc) <= epsilon and on_segment(a, b, c))
        or (abs(abd) <= epsilon and on_segment(a, b, d))
        or (abs(cda) <= epsilon and on_segment(c, d, a))
        or (abs(cdb) <= epsilon and on_segment(c, d, b))
    )


def _rank_non_crossing_grasp_pairs(
    left_poses: torch.Tensor,
    right_poses: torch.Tensor,
    *,
    left_costs: torch.Tensor,
    right_costs: torch.Tensor,
    left_rotation_costs: torch.Tensor,
    right_rotation_costs: torch.Tensor,
    left_base: torch.Tensor,
    right_base: torch.Tensor,
    left_to_right_direction: torch.Tensor,
    minimum_separation: float,
    minimum_lateral_gap: float,
) -> _RankedGraspPairs:
    """Rank only distinct, ordered grasp pairs with non-crossing XY routes."""
    left_poses = torch.as_tensor(left_poses, dtype=torch.float32)
    right_poses = torch.as_tensor(right_poses, dtype=torch.float32)
    direction = torch.as_tensor(
        left_to_right_direction,
        dtype=torch.float32,
        device=left_poses.device,
    )
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-8)
    if minimum_separation < 0.0 or minimum_lateral_gap < 0.0:
        raise ValueError("Pair separation constraints must be non-negative.")
    rejection_counts = {"reversed": 0, "too_close": 0, "path_crossing": 0}
    ranked: list[tuple[float, int, int]] = []
    left_base_position = torch.as_tensor(left_base, dtype=torch.float32)[:3, 3]
    right_base_position = torch.as_tensor(right_base, dtype=torch.float32)[:3, 3]
    for left_index, left_pose in enumerate(left_poses):
        left_position = left_pose[:3, 3]
        left_projection = float(torch.dot(left_position, direction))
        for right_index, right_pose in enumerate(right_poses):
            right_position = right_pose[:3, 3]
            right_projection = float(torch.dot(right_position, direction))
            reversed_pair = (
                left_projection + float(minimum_lateral_gap) > right_projection
            )
            too_close = bool(
                torch.linalg.vector_norm(left_position - right_position)
                < float(minimum_separation)
            )
            crossing = _segments_intersect_2d(
                left_base_position,
                left_position,
                right_base_position,
                right_position,
            )
            rejection_counts["reversed"] += int(reversed_pair)
            rejection_counts["too_close"] += int(too_close)
            rejection_counts["path_crossing"] += int(crossing)
            if reversed_pair or too_close or crossing:
                continue
            route_length = torch.linalg.vector_norm(
                left_position - left_base_position
            ) + torch.linalg.vector_norm(right_position - right_base_position)
            score = (
                float(left_costs[left_index])
                + float(right_costs[right_index])
                + float(left_rotation_costs[left_index]) / math.pi
                + float(right_rotation_costs[right_index]) / math.pi
                + 0.05 * float(route_length)
            )
            ranked.append((score, left_index, right_index))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return _RankedGraspPairs(
        ranked_pairs=tuple((left, right) for _, left, right in ranked),
        scores=tuple(score for score, _, _ in ranked),
        rejection_counts=rejection_counts,
    )


def _point_segment_distance(
    point: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    segment = end - start
    denominator = torch.sum(segment * segment, dim=-1).clamp_min(1.0e-12)
    fraction = torch.sum((point - start) * segment, dim=-1) / denominator
    closest = start + torch.clamp(fraction, 0.0, 1.0)[..., None] * segment
    return torch.linalg.vector_norm(point - closest, dim=-1)


def _segment_distance(
    first_start: torch.Tensor,
    first_end: torch.Tensor,
    second_start: torch.Tensor,
    second_end: torch.Tensor,
) -> torch.Tensor:
    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    a = torch.sum(first * first, dim=-1).clamp_min(1.0e-12)
    b = torch.sum(first * second, dim=-1)
    c = torch.sum(second * second, dim=-1).clamp_min(1.0e-12)
    d = torch.sum(first * offset, dim=-1)
    e = torch.sum(second * offset, dim=-1)
    denominator = a * c - b * b
    first_fraction = (b * e - c * d) / denominator.clamp_min(1.0e-12)
    second_fraction = (a * e - b * d) / denominator.clamp_min(1.0e-12)
    interior = (
        (denominator > 1.0e-12)
        & (first_fraction >= 0.0)
        & (first_fraction <= 1.0)
        & (second_fraction >= 0.0)
        & (second_fraction <= 1.0)
    )
    first_closest = first_start + first_fraction[..., None] * first
    second_closest = second_start + second_fraction[..., None] * second
    interior_distance = torch.linalg.vector_norm(
        first_closest - second_closest,
        dim=-1,
    )
    endpoint_distance = (
        torch.stack(
            (
                _point_segment_distance(first_start, second_start, second_end),
                _point_segment_distance(first_end, second_start, second_end),
                _point_segment_distance(second_start, first_start, first_end),
                _point_segment_distance(second_end, first_start, first_end),
            ),
            dim=-1,
        )
        .min(dim=-1)
        .values
    )
    return torch.where(interior, interior_distance, endpoint_distance)


def _minimum_interarm_capsule_clearance(
    left_link_points: torch.Tensor,
    right_link_points: torch.Tensor,
    *,
    capsule_radius: float,
) -> torch.Tensor:
    """Return minimum surface clearance over every inter-arm link pair."""
    left = torch.as_tensor(left_link_points, dtype=torch.float32)
    right = torch.as_tensor(right_link_points, dtype=torch.float32)
    if left.ndim != 4 or right.ndim != 4 or left.shape[:2] != right.shape[:2]:
        raise ValueError("Link points must have matching shape prefixes (B, T, L, 3).")
    if left.shape[-1] != 3 or right.shape[-1] != 3:
        raise ValueError("Link points must end in xyz coordinates.")
    if left.shape[2] < 2 or right.shape[2] < 2:
        raise ValueError("Each arm must provide at least two link points.")
    if capsule_radius < 0.0:
        raise ValueError("capsule_radius must be non-negative.")
    minimum = torch.full(
        left.shape[:2],
        torch.inf,
        dtype=left.dtype,
        device=left.device,
    )
    for left_index in range(left.shape[2] - 1):
        for right_index in range(right.shape[2] - 1):
            distance = _segment_distance(
                left[:, :, left_index],
                left[:, :, left_index + 1],
                right[:, :, right_index],
                right[:, :, right_index + 1],
            )
            minimum = torch.minimum(minimum, distance)
    return minimum - 2.0 * float(capsule_radius)


def _trajectory_safety_report(
    *,
    left_qpos: torch.Tensor,
    right_qpos: torch.Tensor,
    left_eef: torch.Tensor,
    right_eef: torch.Tensor,
    desired_left_eef: torch.Tensor,
    desired_right_eef: torch.Tensor,
    left_link_points: torch.Tensor,
    right_link_points: torch.Tensor,
    left_to_right_direction: torch.Tensor,
    maximum_joint_step: float,
    maximum_orientation_error: float,
    minimum_lateral_gap: float,
    capsule_radius: float,
    minimum_capsule_clearance: float,
    orientation_start_index: int = 0,
) -> _TrajectorySafetyReport:
    """Evaluate all hard E5 post-plan continuity and inter-arm constraints."""
    if left_qpos.ndim != 3 or right_qpos.ndim != 3:
        raise ValueError("Arm qpos trajectories must have shape (B, T, DOF).")
    if left_qpos.shape[:2] != right_qpos.shape[:2] or left_qpos.shape[1] < 2:
        raise ValueError("Arm qpos trajectories must share at least two waypoints.")
    waypoint_count = left_qpos.shape[1]
    if not 0 <= orientation_start_index < waypoint_count:
        raise ValueError("orientation_start_index must select a trajectory waypoint.")
    left_step = torch.amax(torch.abs(torch.diff(left_qpos, dim=1)), dim=(1, 2))
    right_step = torch.amax(torch.abs(torch.diff(right_qpos, dim=1)), dim=(1, 2))
    joint_step = torch.maximum(left_step, right_step)
    left_orientation = torch.amax(
        _rotation_distance(
            left_eef[:, orientation_start_index:, :3, :3],
            desired_left_eef[:, orientation_start_index:, :3, :3],
        ),
        dim=1,
    )
    right_orientation = torch.amax(
        _rotation_distance(
            right_eef[:, orientation_start_index:, :3, :3],
            desired_right_eef[:, orientation_start_index:, :3, :3],
        ),
        dim=1,
    )
    orientation = torch.maximum(left_orientation, right_orientation)
    direction = torch.as_tensor(
        left_to_right_direction,
        dtype=left_eef.dtype,
        device=left_eef.device,
    )
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-8)
    lateral_gap = (
        torch.sum(
            (right_eef[:, :, :3, 3] - left_eef[:, :, :3, 3]) * direction,
            dim=2,
        )
        .min(dim=1)
        .values
    )
    capsule_clearance = (
        _minimum_interarm_capsule_clearance(
            left_link_points,
            right_link_points,
            capsule_radius=capsule_radius,
        )
        .min(dim=1)
        .values
    )
    failures = {
        "joint_step": joint_step > float(maximum_joint_step),
        "orientation": orientation > float(maximum_orientation_error),
        "lateral_order": lateral_gap < float(minimum_lateral_gap),
        "capsule_collision": capsule_clearance < float(minimum_capsule_clearance),
    }
    failed = torch.zeros_like(joint_step, dtype=torch.bool)
    for value in failures.values():
        failed |= value
    return _TrajectorySafetyReport(
        success=~failed,
        failed_checks={
            name: value.detach().cpu().tolist() for name, value in failures.items()
        },
        metrics={
            "maximum_joint_step": joint_step.detach().cpu().tolist(),
            "maximum_orientation_error": orientation.detach().cpu().tolist(),
            "minimum_lateral_gap": lateral_gap.detach().cpu().tolist(),
            "minimum_capsule_clearance": capsule_clearance.detach().cpu().tolist(),
        },
    )
