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

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

from embodichain.utils.math import quat_from_matrix

__all__ = ["pose_nms", "pose_nms_indices"]

_POSE_NMS_CHUNK_SIZE = 2048


@wp.func
def _poses_are_close(
    positions: wp.array(dtype=wp.float32, ndim=2),
    quaternions: wp.array(dtype=wp.float32, ndim=2),
    reference_idx: int,
    target_idx: int,
    rotation_cosine_threshold: float,
    distance_threshold_squared: float,
    rotation_always_close: bool,
) -> bool:
    """Compare poses through their relative rotation and translation."""
    # For unit xyzw quaternions, the real component of
    # inverse(q_reference) * q_target is their dot product. Its absolute value
    # gives the shortest relative rotation while treating q and -q equally.
    relative_rotation_w = (
        quaternions[reference_idx, 0] * quaternions[target_idx, 0]
        + quaternions[reference_idx, 1] * quaternions[target_idx, 1]
        + quaternions[reference_idx, 2] * quaternions[target_idx, 2]
        + quaternions[reference_idx, 3] * quaternions[target_idx, 3]
    )
    relative_translation_x = positions[target_idx, 0] - positions[reference_idx, 0]
    relative_translation_y = positions[target_idx, 1] - positions[reference_idx, 1]
    relative_translation_z = positions[target_idx, 2] - positions[reference_idx, 2]

    rotation_close = rotation_always_close or (
        wp.abs(relative_rotation_w) > rotation_cosine_threshold
    )
    translation_close = (
        relative_translation_x * relative_translation_x
        + relative_translation_y * relative_translation_y
        + relative_translation_z * relative_translation_z
        < distance_threshold_squared
    )
    return rotation_close and translation_close


@wp.kernel(enable_backward=False)
def _pose_pair_close_kernel(
    positions: wp.array(dtype=wp.float32, ndim=2),
    quaternions: wp.array(dtype=wp.float32, ndim=2),
    reference_offset: int,
    target_offset: int,
    num_targets: int,
    rotation_cosine_threshold: float,
    distance_threshold_squared: float,
    rotation_always_close: bool,
    close: wp.array(dtype=wp.uint8),
) -> None:
    """Compute a tile of the pairwise pose-closeness matrix."""
    pair_idx = wp.tid()
    reference_local_idx = pair_idx // num_targets
    target_local_idx = pair_idx - reference_local_idx * num_targets
    reference_idx = reference_offset + reference_local_idx
    target_idx = target_offset + target_local_idx

    if reference_idx == target_idx:
        close[pair_idx] = wp.uint8(0)
        return

    close[pair_idx] = wp.uint8(
        _poses_are_close(
            positions,
            quaternions,
            reference_idx,
            target_idx,
            rotation_cosine_threshold,
            distance_threshold_squared,
            rotation_always_close,
        )
    )


@wp.kernel(enable_backward=False)
def _count_close_poses_kernel(
    positions: wp.array(dtype=wp.float32, ndim=2),
    quaternions: wp.array(dtype=wp.float32, ndim=2),
    reference_offset: int,
    target_offset: int,
    num_targets: int,
    rotation_cosine_threshold: float,
    distance_threshold_squared: float,
    rotation_always_close: bool,
    close_counts: wp.array(dtype=wp.int32),
) -> None:
    """Accumulate close-neighbor counts for a pairwise tile."""
    pair_idx = wp.tid()
    reference_local_idx = pair_idx // num_targets
    target_local_idx = pair_idx - reference_local_idx * num_targets
    reference_idx = reference_offset + reference_local_idx
    target_idx = target_offset + target_local_idx

    if reference_idx != target_idx and _poses_are_close(
        positions,
        quaternions,
        reference_idx,
        target_idx,
        rotation_cosine_threshold,
        distance_threshold_squared,
        rotation_always_close,
    ):
        wp.atomic_add(close_counts, reference_idx, 1)


def _poses_to_components(poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pose matrices to positions and normalized xyzw quaternions."""
    # Warp composite arrays currently use float32 storage. NMS only uses these
    # values for threshold decisions; the returned poses retain their original
    # dtype and autograd relationship.
    poses_f32 = poses.detach().to(dtype=torch.float32).contiguous()
    positions = poses_f32[:, :3, 3].contiguous()
    quaternions_wxyz = quat_from_matrix(poses_f32[:, :3, :3])
    quaternions = torch.cat([quaternions_wxyz[:, 1:], quaternions_wxyz[:, :1]], dim=-1)
    quaternions = quaternions / torch.linalg.vector_norm(
        quaternions, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(quaternions.dtype).eps)
    return positions, quaternions


def _count_close_poses(
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    rotation_cosine_threshold: float,
    distance_threshold_squared: float,
    rotation_always_close: bool,
    chunk_size: int,
) -> torch.Tensor:
    """Count close neighbors using bounded Warp pairwise tiles."""
    num_poses = positions.shape[0]
    positions_wp = wp.from_torch(positions, dtype=wp.float32)
    quaternions_wp = wp.from_torch(quaternions, dtype=wp.float32)
    if positions_wp.device.is_cuda:
        # The components were produced by Torch immediately before this call.
        # Make them visible to Warp before launching on its stream.
        torch.cuda.synchronize(positions.device)
    close_counts_wp = wp.zeros(num_poses, dtype=wp.int32, device=positions_wp.device)

    for reference_offset in range(0, num_poses, chunk_size):
        num_references = min(chunk_size, num_poses - reference_offset)
        for target_offset in range(0, num_poses, chunk_size):
            num_targets = min(chunk_size, num_poses - target_offset)
            wp.launch(
                kernel=_count_close_poses_kernel,
                dim=num_references * num_targets,
                inputs=[
                    positions_wp,
                    quaternions_wp,
                    reference_offset,
                    target_offset,
                    num_targets,
                    rotation_cosine_threshold,
                    distance_threshold_squared,
                    rotation_always_close,
                    close_counts_wp,
                ],
                device=positions_wp.device,
            )
    if positions_wp.device.is_cuda:
        wp.synchronize_device(positions_wp.device)
        torch.cuda.synchronize(positions.device)
    return wp.to_torch(close_counts_wp).clone()


def _greedy_keep_indices(
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    visit_order: torch.Tensor,
    rotation_cosine_threshold: float,
    distance_threshold_squared: float,
    rotation_always_close: bool,
    chunk_size: int,
) -> torch.Tensor:
    """Apply greedy suppression while computing pairwise tiles with Warp."""
    num_poses = positions.shape[0]
    ordered_positions = positions[visit_order].contiguous()
    ordered_quaternions = quaternions[visit_order].contiguous()
    positions_wp = wp.from_torch(ordered_positions, dtype=wp.float32)
    quaternions_wp = wp.from_torch(ordered_quaternions, dtype=wp.float32)
    if positions_wp.device.is_cuda:
        # The indexing operations above run on Torch's stream.
        torch.cuda.synchronize(ordered_positions.device)

    # Keep the host-side closeness block bounded to roughly chunk_size**2
    # entries even when there are far more poses than one target tile.
    reference_chunk_size = max(1, min(chunk_size, chunk_size**2 // num_poses))
    suppressed = np.zeros(num_poses, dtype=np.bool_)
    keep_ordered_indices: list[int] = []

    for reference_offset in range(0, num_poses, reference_chunk_size):
        num_references = min(reference_chunk_size, num_poses - reference_offset)
        max_num_targets = min(chunk_size, num_poses)
        close_buffer_wp = wp.empty(
            num_references * max_num_targets,
            dtype=wp.uint8,
            device=positions_wp.device,
        )
        close_block = np.empty((num_references, num_poses), dtype=np.bool_)

        for target_offset in range(0, num_poses, chunk_size):
            num_targets = min(chunk_size, num_poses - target_offset)
            num_pairs = num_references * num_targets
            wp.launch(
                kernel=_pose_pair_close_kernel,
                dim=num_pairs,
                inputs=[
                    positions_wp,
                    quaternions_wp,
                    reference_offset,
                    target_offset,
                    num_targets,
                    rotation_cosine_threshold,
                    distance_threshold_squared,
                    rotation_always_close,
                    close_buffer_wp,
                ],
                device=positions_wp.device,
            )
            # The tile is consumed by NumPy immediately, so make the Warp
            # launch complete before copying it to host memory.
            if close_buffer_wp.device.is_cuda:
                wp.synchronize_device(close_buffer_wp.device)
            close_block[:, target_offset : target_offset + num_targets] = (
                close_buffer_wp.numpy()[:num_pairs].reshape(num_references, num_targets)
                != 0
            )

        for reference_local_idx in range(num_references):
            reference_idx = reference_offset + reference_local_idx
            if suppressed[reference_idx]:
                continue
            keep_ordered_indices.append(reference_idx)
            suppressed |= close_block[reference_local_idx]
            suppressed[reference_idx] = True

    ordered_keep = torch.tensor(
        keep_ordered_indices, dtype=torch.long, device=positions.device
    )
    return visit_order[ordered_keep]


def pose_nms_indices(
    poses: torch.Tensor,
    angle_th: float = np.pi / 36,
    dist_th: float = 0.003,
    preserve_order: bool = False,
    chunk_size: int = _POSE_NMS_CHUNK_SIZE,
) -> torch.Tensor:
    """Return pose indices after removing poses that are too close.

    Pose matrices are first converted into ``(N, 3)`` positions and unit
    ``(N, 4)`` xyzw quaternions. Warp kernels compare their relative rotation
    and Euclidean relative translation in bounded pairwise tiles.

    Args:
        poses: Input pose matrices. Shape is ``(N, 4, 4)``.
        angle_th: Rotation threshold in radians. Poses with angular distance
            below this value are considered close. Defaults to pi / 36.
        dist_th: Translation distance threshold. Poses with Euclidean distance
            below this value are considered close. Defaults to 0.003.
        preserve_order: Whether to greedily select poses in input order. If
            ``False``, poses with fewer close neighbors are selected first.
            Defaults to ``False``.
        chunk_size: Maximum size of either dimension of a Warp pairwise tile.
            Defaults to 2048.

    Returns:
        Indices of selected poses. Shape is ``(M,)``, where ``M <= N``.

    Raises:
        ValueError: If ``poses`` is not shaped as ``(N, 4, 4)``, is not on a
            Warp-supported device, or ``chunk_size`` is not positive.
    """
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Invalid input shape {poses.shape}, expected (N, 4, 4).")
    if poses.device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported pose device {poses.device}; expected a CPU or CUDA tensor."
        )
    if chunk_size <= 0:
        raise ValueError(
            f"Invalid chunk_size {chunk_size}, expected a positive integer."
        )

    num_poses = poses.shape[0]
    if num_poses == 0:
        return torch.empty(0, dtype=torch.long, device=poses.device)
    if angle_th <= 0.0 or dist_th <= 0.0:
        return torch.arange(num_poses, dtype=torch.long, device=poses.device)

    # ``pose_nms`` may be called without a SimulationManager. ``wp.init`` is
    # idempotent when the simulation has already initialized Warp.
    wp.init()
    positions, quaternions = _poses_to_components(poses)
    rotation_always_close = angle_th > math.pi
    rotation_cosine_threshold = (
        math.cos(0.5 * float(angle_th)) if not rotation_always_close else 0.0
    )
    distance_threshold_squared = float(dist_th * dist_th)

    if preserve_order:
        visit_order = torch.arange(num_poses, dtype=torch.long, device=poses.device)
    else:
        close_counts = _count_close_poses(
            positions,
            quaternions,
            rotation_cosine_threshold,
            distance_threshold_squared,
            rotation_always_close,
            chunk_size,
        ).to(dtype=torch.long)
        tie_breaker = torch.arange(num_poses, dtype=torch.long, device=poses.device)
        visit_priority = close_counts * (num_poses + 1) + tie_breaker
        visit_order = torch.argsort(visit_priority)

    return _greedy_keep_indices(
        positions,
        quaternions,
        visit_order,
        rotation_cosine_threshold,
        distance_threshold_squared,
        rotation_always_close,
        chunk_size,
    )


def pose_nms(
    poses: torch.Tensor,
    angle_th: float = np.pi / 36,
    dist_th: float = 0.003,
    chunk_size: int = _POSE_NMS_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove poses that are too close in translation and rotation.

    Args:
        poses: Input pose matrices. Shape is ``(N, 4, 4)``.
        angle_th: Rotation threshold in radians. Defaults to pi / 36.
        dist_th: Translation threshold. Defaults to 0.003.
        chunk_size: Maximum size of either dimension of a Warp pairwise tile.
            Defaults to 2048.

    Returns:
        A tuple containing the filtered pose matrices in input order and their
        original indices. The shapes are ``(M, 4, 4)`` and ``(M,)``.
    """
    keep_indices = pose_nms_indices(
        poses,
        angle_th=angle_th,
        dist_th=dist_th,
        preserve_order=True,
        chunk_size=chunk_size,
    )
    return poses[keep_indices], keep_indices
