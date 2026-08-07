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

"""Trajectory interpolation and action-related simulation utilities."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from embodichain.lab.sim.solvers.base_solver import BaseSolver
from embodichain.utils.device_utils import standardize_device_string
from embodichain.utils.utility import inv_transform
from embodichain.utils.warp import (
    cumsum_distances,
    get_offset_qpos_kernel,
    interpolate_along_distance,
    pairwise_distances,
    repeat_first_point,
    trajectory_add_origin_kernel,
    trajectory_get_diff_kernel,
    trajectory_interpolate_kernel,
)

__all__ = [
    "compute_pose_offset_related_to_first",
    "get_trajectory_object_offset_qpos",
    "interpolate_with_distance",
    "interpolate_with_nums",
    "resample_with_distance",
    "sort_and_padding_key_frame",
    "warp_trajectory_qpos",
]


def compute_pose_offset_related_to_first(full_pose: torch.Tensor) -> torch.Tensor:
    """Compute pose offset relative to the first pose.

    Args:
        full_pose (torch.Tensor): The full pose tensor of shape (N, 4, 4).

    Returns:
        torch.Tensor: The pose offset tensor of shape (N, 4, 4).
    """
    inv_pose0_np = inv_transform(full_pose[0].to("cpu").numpy())
    inv_pose0 = torch.tensor(inv_pose0_np, device=full_pose.device)
    inv_pose0_repeat = inv_pose0[None, :, :].repeat(full_pose.shape[0], 1, 1)
    return torch.bmm(inv_pose0_repeat, full_pose)


def sort_and_padding_key_frame(
    trajectory: np.ndarray, key_indices: np.ndarray, key_frames_batch: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """sort and padding key frames for warping trajectory

    Args:
        trajectory (torch.Tensor): raw trajectory. [n_waypoint, dof] of float.
        key_indices (torch.Tensor): key frame waypoint indices. [n_keyframe,] of int.
        key_frames_batch (torch.Tensor): batch key frames. [n_batch, n_keyframe, dof] of float.

    Returns:
        key_indices_ascending (np.ndarray): padded and sorted key frame indices. [n_keyframe_new,] of int.
        key_frames_ascending (np.ndarray): padded and sorted batch key frames. [n_batch, n_keyframe_new, dof] of float.
    """
    sort_ids = np.argsort(key_indices)
    key_indices_ascending = key_indices[sort_ids]
    key_frames_ascending = key_frames_batch[:, sort_ids, :]
    n_batch = key_frames_batch.shape[0]
    if key_indices_ascending[0] != 0:
        key_indices_ascending = np.hstack([0, key_indices_ascending])
        padding_frame = trajectory[0][None, None, :].repeat(n_batch, axis=0)
        key_frames_ascending = np.concatenate(
            [padding_frame, key_frames_ascending], axis=1
        )
    if key_indices_ascending[-1] != trajectory.shape[0] - 1:
        key_indices_ascending = np.hstack(
            [key_indices_ascending, trajectory.shape[0] - 1]
        )
        padding_frame = trajectory[trajectory.shape[0] - 1][None, None, :].repeat(
            n_batch, axis=0
        )
        key_frames_ascending = np.concatenate(
            [key_frames_ascending, padding_frame], axis=1
        )
    return key_indices_ascending, key_frames_ascending


def warp_trajectory_qpos(
    trajectory: torch.Tensor,
    key_indices: torch.Tensor,
    key_frames_batch: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """warp trajectory

    Args:
        trajectory (torch.Tensor): raw trajectory. [n_waypoint, dof] of float.
        key_indices (torch.Tensor): key frame waypoint indices. [n_keyframe,] of int.
        key_frames_batch (torch.Tensor): batch key frames. [n_batch, n_keyframe, dof] of float.
        device (str, optional): torch tensor device. Defaults to "cuda".

    Returns:
        torch.Tensor: warped trajectory. [n_batch, n_waypoint, dof] of float.
    """
    # sort and pad key frames
    trajectory_np = trajectory.to("cpu").numpy().astype(np.float32)
    key_indices_np = key_indices.to("cpu").numpy().astype(np.int32)
    key_frames_batch_np = key_frames_batch.to("cpu").numpy().astype(np.float32)

    key_indices_padded, key_frames_padded = sort_and_padding_key_frame(
        trajectory_np, key_indices_np, key_frames_batch_np
    )

    # allocate cuda memory
    n_batch = key_frames_padded.shape[0]
    n_keyframe = key_indices_padded.shape[0]
    n_waypoint, dof = trajectory_np.shape
    wp_in_trajectory = wp.array(
        trajectory_np.flatten(), dtype=float, device=standardize_device_string(device)
    )
    out_trajectory = np.zeros((n_batch, n_waypoint, dof), dtype=np.float32)
    wp_out_trajectory = wp.array(
        out_trajectory.flatten(), dtype=float, device=standardize_device_string(device)
    )
    wp_key_indices = wp.array(
        key_indices_padded, dtype=int, device=standardize_device_string(device)
    )
    wp_key_frames = wp.array(
        key_frames_padded.flatten(),
        dtype=float,
        device=standardize_device_string(device),
    )

    # calcuate
    wp.launch(
        kernel=trajectory_get_diff_kernel,
        dim=(n_batch, dof),
        inputs=[
            wp_in_trajectory,
            wp_key_indices,
            wp_key_frames,
            n_waypoint,
            dof,
            n_keyframe,
        ],
        outputs=[
            wp_out_trajectory,
        ],
        device=standardize_device_string(device),
    )
    wp.launch(
        kernel=trajectory_interpolate_kernel,
        dim=(n_batch, n_waypoint, dof),
        inputs=[wp_key_indices, n_waypoint, dof, n_keyframe],
        outputs=[
            wp_out_trajectory,
        ],
        device=standardize_device_string(device),
    )
    wp.launch(
        kernel=trajectory_add_origin_kernel,
        dim=(n_batch, n_waypoint, dof),
        inputs=[wp_in_trajectory, n_waypoint, dof],
        outputs=[
            wp_out_trajectory,
        ],
        device=standardize_device_string(device),
    )
    warp_traj = (
        wp.to_torch(wp_out_trajectory)
        .reshape(n_batch, n_waypoint, dof)
        .to(torch.device(device))
    )
    return warp_traj


def get_trajectory_object_offset_qpos(
    trajectory: torch.Tensor,
    key_indices: torch.Tensor,
    key_obj_indices: torch.Tensor,
    obj_offset: torch.Tensor,
    solver: BaseSolver,
    base_xpos: torch.Tensor,
    device=torch.device("cuda"),
):
    """warp trajectory according to object pose offset

    Args:
        trajectory (torch.Tensor): raw trajectory. [n_waypoint, dof] of float, joint positions.
        key_indices (torch.Tensor): key frame waypoint indices. [n_keyframe,] of int.
        key_obj_indices (torch.Tensor): key frame belong to which object index. [n_keyframe,] of int.
        obj_offset (torch.Tensor): each object pose offset. [obj_num, n_batch, 4, 4] of float.
        solver (BaseSolver): robot kinematic solver.
        base_xpos (torch.Tensor): solver root link pose in world coordinate. [4, 4] of float.
        device (str, optional): torch tensor device. Defaults to "cuda".

    Returns:
        torch.Tensor: warped trajectory. [n_batch, n_waypoint, dof] of float.
    """
    assert key_indices.shape[0] == key_obj_indices.shape[0]
    dof = trajectory.shape[1]
    key_qpos = trajectory[key_indices]  # [n_keyframe, DOF]
    n_batch = obj_offset.shape[1]  # batch num, aws arena num
    n_keyframe = key_qpos.shape[0]
    key_xpos = solver.get_fk(key_qpos)  # [n_keyframe, 4, 4]

    base_xpos_repeat = base_xpos[None, :, :].repeat(n_keyframe, 1, 1)
    key_xpos = torch.bmm(base_xpos_repeat, key_xpos)

    base_xpos_inv_np = inv_transform(base_xpos.to("cpu").numpy())
    base_xpos_inv_wp = wp.mat44f(base_xpos_inv_np)
    key_obj_indices_wp = wp.from_torch(key_obj_indices.reshape(-1))
    obj_offset_wp = wp.from_torch(obj_offset.reshape(-1))
    key_xpos_wp = wp.from_torch(key_xpos.reshape(-1))
    key_obj_offset_wp = wp.zeros(
        n_batch * n_keyframe * 16, dtype=float, device=standardize_device_string(device)
    )

    wp.launch(
        kernel=get_offset_qpos_kernel,
        dim=(n_batch, n_keyframe),
        inputs=[
            key_obj_indices_wp,
            obj_offset_wp,
            key_xpos_wp,
            base_xpos_inv_wp,
            n_batch,
            n_keyframe,
        ],
        outputs=[
            key_obj_offset_wp,
        ],
        device=standardize_device_string(device),
    )
    key_xpos_offset = wp.to_torch(key_obj_offset_wp).reshape(n_batch * n_keyframe, 4, 4)
    key_qpos_batch = key_qpos[None, :, :].repeat(n_batch, 1, 1).reshape(-1, dof)
    # for pytorch solver, ik use qpos seed but not joint seed
    is_success, key_qpos_offset = solver.get_ik(
        target_xpos=key_xpos_offset,
        qpos_seed=key_qpos_batch,
    )
    key_qpos_offset = key_qpos_offset.reshape(n_batch, n_keyframe, -1)
    return is_success, key_qpos_offset


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


def resample_with_distance(
    trajectory: torch.Tensor,
    interp_num: int,
    device: torch.device | str = torch.device("cuda"),
) -> torch.Tensor:
    """Resample a batched path at uniform cumulative-distance positions.

    Unlike :func:`interpolate_with_distance`, interior input samples are not
    required output points, so this function supports both upsampling and
    downsampling. It is intended for dense planner paths rather than required
    waypoint sequences.

    Args:
        trajectory: Path tensor with shape ``(B, N, M)``.
        interp_num: Target number of samples ``T``.
        device: Device on which to perform interpolation.

    Returns:
        Resampled trajectories with shape ``(B, T, M)``.

    Raises:
        ValueError: If ``trajectory`` is not three-dimensional, contains no
            points for a non-empty output, or ``interp_num`` is negative.
    """
    if trajectory.ndim != 3:
        raise ValueError("`trajectory` must have shape (B, N, M).")

    trajectory = trajectory.contiguous().to(device)
    if not torch.is_floating_point(trajectory) or trajectory.dtype != torch.float32:
        trajectory = trajectory.float()

    batch_size, point_count, dimension = trajectory.shape
    sample_count = int(interp_num)
    if sample_count < 0:
        raise ValueError("`interp_num` must be non-negative.")
    if point_count == 0:
        if sample_count == 0:
            return trajectory.new_empty((batch_size, 0, dimension))
        raise ValueError("Cannot resample a trajectory with no points.")
    if batch_size == 0 or sample_count == 0:
        return trajectory.new_empty((batch_size, sample_count, dimension))

    # Flatten input trajectory for Warp kernels (avoids multidimensional
    # wp.array interop issues).
    trajectory_flat = trajectory.view(-1)
    points = wp.from_torch(trajectory_flat)

    out = wp.empty(
        (batch_size * sample_count * dimension,),
        dtype=wp.float32,
        device=standardize_device_string(device),
    )

    if point_count == 1:
        wp.launch(
            kernel=repeat_first_point,
            dim=batch_size * sample_count,
            inputs=[
                points,
                out,
                batch_size,
                sample_count,
                dimension,
                point_count,
            ],
            device=standardize_device_string(device),
        )
        return wp.to_torch(out).view(batch_size, sample_count, dimension)

    dists = wp.empty(
        (batch_size * (point_count - 1),),
        dtype=wp.float32,
        device=standardize_device_string(device),
    )
    wp.launch(
        kernel=pairwise_distances,
        dim=batch_size * (point_count - 1),
        inputs=[points, dists, batch_size, point_count, dimension],
        device=standardize_device_string(device),
    )

    cumulative = wp.empty(
        (batch_size * point_count,),
        dtype=wp.float32,
        device=standardize_device_string(device),
    )
    wp.launch(
        kernel=cumsum_distances,
        dim=batch_size,
        inputs=[dists, cumulative, batch_size, point_count],
        device=standardize_device_string(device),
    )

    wp.launch(
        kernel=interpolate_along_distance,
        dim=batch_size * sample_count,
        inputs=[
            points,
            cumulative,
            out,
            batch_size,
            point_count,
            dimension,
            sample_count,
        ],
        device=standardize_device_string(device),
    )
    return wp.to_torch(out).view(batch_size, sample_count, dimension)


def interpolate_with_nums(
    trajectory: torch.Tensor,  # expected shape [B, N, M], float or convertible to float
    interp_nums: torch.Tensor,  # expected shape [N - 1], interp_num in each segment
    device=torch.device("cuda"),
) -> torch.Tensor:
    """
    Each entry ``interp_nums[i] = k`` controls segment ``i`` between
    ``trajectory[:, i, :]`` and ``trajectory[:, i + 1, :]``. For that segment,
    ``k`` samples are generated with interpolation factors
    ``alpha = 0, 1/k, 2/k, ..., (k-1)/k`` (i.e., including the segment start
    and excluding the segment end). The final endpoint
    ``trajectory[:, -1, :]`` is appended once at the end of the result, so
    intermediate segment endpoints are not duplicated.

    Args:
        trajectory: Torch.Tensor of shape [B, N, M].
        interp_nums: Torch.Tensor of shape [N - 1] specifying the number of
            samples per segment, including each segment start and excluding
            its end. Values must be non-negative; a value of 0 means that
            no samples are drawn from that segment (other than the final
            overall endpoint that is always appended once).
        device: Torch device string ('cpu', 'cuda', 'cuda:0', ...).

    Returns:
        Torch.Tensor of interpolated trajectories.
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
