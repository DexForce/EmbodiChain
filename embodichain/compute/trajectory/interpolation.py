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

"""Interpolate batched keyframes while preserving their boundaries."""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["interpolate_with_distance", "interpolate_with_nums"]


def _allocate_segment_intervals(
    distances: torch.Tensor, total_intervals: int
) -> torch.Tensor:
    """Allocate output intervals to segments independently for each batch.

    Every segment receives one interval so its endpoint is retained. Remaining
    intervals are apportioned by segment length with the largest-remainder
    method, producing an exact ``total_intervals`` sum for every batch.

    Args:
        distances: Per-segment distances with shape ``(B, S)``.
        total_intervals: Total number of output intervals. Must be at least
            ``S``.

    Returns:
        Integer interval counts with shape ``(B, S)``.
    """
    batch_size, segment_count = distances.shape
    intervals = torch.ones(
        (batch_size, segment_count), dtype=torch.int64, device=distances.device
    )
    remaining = total_intervals - segment_count
    if remaining == 0 or batch_size == 0:
        return intervals

    total_distance = distances.sum(dim=1, keepdim=True)
    uniform_weights = torch.full_like(distances, 1.0 / segment_count)
    safe_total_distance = torch.where(
        total_distance > 0, total_distance, torch.ones_like(total_distance)
    )
    distance_weights = distances / safe_total_distance
    weights = torch.where(total_distance > 0, distance_weights, uniform_weights)

    quotas = weights * remaining
    extra_intervals = torch.floor(quotas).to(torch.int64)
    remainders = quotas - extra_intervals.to(quotas.dtype)
    intervals += extra_intervals

    unallocated = remaining - extra_intervals.sum(dim=1)
    ranked_segments = torch.argsort(remainders, dim=1, descending=True, stable=True)
    ranked_bonus = (
        torch.arange(segment_count, device=distances.device).unsqueeze(0)
        < unallocated.unsqueeze(1)
    ).to(torch.int64)
    bonus = torch.zeros_like(intervals)
    bonus.scatter_(1, ranked_segments, ranked_bonus)
    return intervals + bonus


def interpolate_with_distance(
    trajectory: torch.Tensor,
    interp_num: int,
    device: torch.device | str = torch.device("cuda"),
) -> torch.Tensor:
    """Interpolate batched keyframes while preserving every keyframe boundary.

    Each input point is treated as a required keyframe. The output is generated
    segment by segment: every segment receives at least one interval, and any
    remaining intervals are distributed by Euclidean segment length for each
    batch independently. Segment endpoints are copied directly from the input,
    so intermediate keyframes occur as exact emitted samples.

    .. attention::
        ``interp_num`` must be at least the number of input keyframes. Use
        :func:`resample_with_distance` when input points are optional dense path
        samples that may be downsampled.

    Args:
        trajectory: Keyframe tensor with shape ``(B, N, M)``.
        interp_num: Target number of samples ``T``.
        device: Device on which to perform interpolation.

    Returns:
        Interpolated trajectories with shape ``(B, T, M)``.

    Raises:
        ValueError: If ``trajectory`` is not three-dimensional, contains no
            keyframes for a non-empty output, or ``interp_num`` cannot hold all
            keyframes.
    """
    if trajectory.ndim != 3:
        raise ValueError("`trajectory` must have shape (B, N, M).")

    trajectory = trajectory.to(device)
    if not torch.is_floating_point(trajectory):
        trajectory = trajectory.float()

    batch_size, keyframe_count, dimension = trajectory.shape
    sample_count = int(interp_num)
    if sample_count < 0:
        raise ValueError("`interp_num` must be non-negative.")
    if keyframe_count == 0:
        if sample_count == 0:
            return trajectory.new_empty((batch_size, 0, dimension))
        raise ValueError("Cannot interpolate a trajectory with no keyframes.")
    if sample_count < keyframe_count:
        raise ValueError(
            f"`interp_num` ({sample_count}) must be at least the number of "
            f"keyframes ({keyframe_count}) so every keyframe can be preserved."
        )
    if batch_size == 0:
        return trajectory.new_empty((0, sample_count, dimension))
    if keyframe_count == 1:
        return trajectory.expand(-1, sample_count, -1).clone()
    if sample_count == keyframe_count:
        return trajectory.clone()

    segment_distances = torch.linalg.vector_norm(
        trajectory[:, 1:, :] - trajectory[:, :-1, :], dim=-1
    )
    segment_intervals = _allocate_segment_intervals(
        segment_distances, total_intervals=sample_count - 1
    )
    segment_ends = torch.cumsum(segment_intervals, dim=1)
    segment_starts = torch.cat(
        [torch.zeros_like(segment_ends[:, :1]), segment_ends[:, :-1]], dim=1
    )

    output_indices = (
        torch.arange(sample_count, device=trajectory.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    segment_indices = torch.searchsorted(segment_ends, output_indices, right=False)
    interval_counts = torch.gather(segment_intervals, 1, segment_indices)
    local_indices = output_indices - torch.gather(segment_starts, 1, segment_indices)
    alpha = (
        local_indices.to(trajectory.dtype) / interval_counts.to(trajectory.dtype)
    ).unsqueeze(-1)

    gather_indices = segment_indices.unsqueeze(-1).expand(-1, -1, dimension)
    segment_start_points = torch.gather(trajectory[:, :-1, :], 1, gather_indices)
    segment_end_points = torch.gather(trajectory[:, 1:, :], 1, gather_indices)
    interpolated = torch.lerp(segment_start_points, segment_end_points, alpha)

    # Copy endpoints instead of relying on floating-point interpolation at
    # alpha == 1, guaranteeing bit-exact keyframe samples in the output.
    is_segment_end = (local_indices == interval_counts).unsqueeze(-1)
    return torch.where(is_segment_end, segment_end_points, interpolated)


def interpolate_with_nums(
    trajectory: torch.Tensor,  # expected shape [B, N, M], float or convertible to float
    interp_nums: torch.Tensor | Sequence[int],
    device: torch.device | str = torch.device("cuda"),
) -> torch.Tensor:
    """Interpolate each segment with its requested number of intervals.

    The first keyframe is emitted once. Each positive count appends that many
    evenly spaced samples ending at the next keyframe. A zero count appends
    the next keyframe directly, so every original boundary is retained.

    Args:
        trajectory: Keyframe tensor with shape ``(B, N, M)``.
        interp_nums: Non-negative interval counts with shape ``(N - 1,)``.
        device: Device on which to perform interpolation.

    Returns:
        A tensor containing the interpolated trajectories. For non-empty input,
        its sample count is ``1 + sum(max(count, 1) for count in interp_nums)``.

    Raises:
        ValueError: If the count shape is invalid or a count is negative.
    """
    trajectory = trajectory.to(device)
    if not torch.is_floating_point(trajectory):
        trajectory = trajectory.float()

    B, N, M = trajectory.shape
    if N == 0:
        return trajectory.new_empty((B, 0, M))

    interp_nums_tensor = torch.as_tensor(interp_nums, device="cpu").reshape(-1)
    if interp_nums_tensor.numel() != max(N - 1, 0):
        raise ValueError("`interp_nums` must have shape (N - 1,).")

    if N == 1:
        return trajectory[:, :1, :]

    interp_nums_list = interp_nums_tensor.to(torch.int64).tolist()

    # Always seed the output with the first waypoint so it is never dropped,
    # even when leading segments have zero samples.
    segments = [trajectory[:, :1, :]]
    for i, count in enumerate(interp_nums_list):
        if count < 0:
            raise ValueError("`interp_nums` values must be non-negative.")
        p0 = trajectory[:, i : i + 1, :]
        p1 = trajectory[:, i + 1 : i + 2, :]
        if count == 0:
            # No interpolated samples for this segment, but ensure the endpoint
            # waypoint is still present so zero-sample segments don't remove it.
            segments.append(p1)
            continue
        # Generate linearly spaced interpolation parameters from 0 to 1
        # (inclusive), then drop the first value (t = 0) because p0 is
        # already the last point in `segments`. This appends exactly
        # `count` new points per segment and preserves all endpoints.
        alpha = torch.linspace(
            0.0,
            1.0,
            steps=count + 1,
            device=device,
            dtype=trajectory.dtype,
        ).view(1, count + 1, 1)
        seg = p0 + (p1 - p0) * alpha
        segments.append(seg[:, 1:, :])
    return torch.cat(segments, dim=1)
