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

from embodichain.utils.math import quat_from_matrix

__all__ = ["pose_nms", "pose_nms_indices"]

_POSE_NMS_CHUNK_SIZE = 2048


def _poses_to_components(poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pose matrices to positions and normalized xyzw quaternions."""
    # NMS only uses these float32 values for threshold decisions; the returned
    # poses retain their original dtype and autograd relationship.
    poses_f32 = poses.detach().to(dtype=torch.float32).contiguous()
    positions = poses_f32[:, :3, 3].contiguous()
    quaternions_wxyz = quat_from_matrix(poses_f32[:, :3, :3])
    quaternions = torch.cat([quaternions_wxyz[:, 1:], quaternions_wxyz[:, :1]], dim=-1)
    quaternions = quaternions / torch.linalg.vector_norm(
        quaternions, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(quaternions.dtype).eps)
    return positions, quaternions


def _close_block(
    ref_positions: torch.Tensor,
    ref_quaternions: torch.Tensor,
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    rotation_cosine_threshold: float,
    distance_threshold_squared: float,
    rotation_always_close: bool,
    chunk_size: int,
) -> torch.Tensor:
    """Compute one (refs x all) tile row of the pairwise closeness matrix.

    For unit xyzw quaternions, the real component of
    ``inverse(q_reference) * q_target`` is their dot product. Its absolute
    value gives the shortest relative rotation while treating ``q`` and ``-q``
    equally. Translation closeness compares the squared Euclidean distance.

    Args:
        ref_positions: Reference positions with shape ``(R, 3)``.
        ref_quaternions: Reference quaternions with shape ``(R, 4)``.
        positions: All positions with shape ``(N, 3)``.
        quaternions: All quaternions with shape ``(N, 4)``.
        rotation_cosine_threshold: ``cos(angle_th / 2)`` decision value.
        distance_threshold_squared: Squared translation threshold.
        rotation_always_close: Whether rotation is trivially satisfied.
        chunk_size: Maximum target-tile width processed per step.

    Returns:
        Boolean closeness rows with shape ``(R, N)``.
    """
    num_refs = ref_positions.shape[0]
    num_poses = positions.shape[0]
    close = torch.empty(num_refs, num_poses, dtype=torch.bool, device=positions.device)
    for target_offset in range(0, num_poses, chunk_size):
        target_end = min(target_offset + chunk_size, num_poses)
        diff = ref_positions[:, None, :] - positions[None, target_offset:target_end, :]
        tile_close = diff.pow(2).sum(dim=-1) < distance_threshold_squared
        if not rotation_always_close:
            dots = (
                ref_quaternions[:, None, :]
                * quaternions[None, target_offset:target_end, :]
            ).sum(dim=-1)
            tile_close &= dots.abs() > rotation_cosine_threshold
        close[:, target_offset:target_end] = tile_close
    return close


def _count_close_poses(
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    rotation_cosine_threshold: float,
    distance_threshold_squared: float,
    rotation_always_close: bool,
    chunk_size: int,
) -> torch.Tensor:
    """Count close neighbors using bounded pairwise tiles."""
    num_poses = positions.shape[0]
    close_counts = torch.zeros(num_poses, dtype=torch.int64, device=positions.device)
    for reference_offset in range(0, num_poses, chunk_size):
        reference_end = min(reference_offset + chunk_size, num_poses)
        block = _close_block(
            positions[reference_offset:reference_end],
            quaternions[reference_offset:reference_end],
            positions,
            quaternions,
            rotation_cosine_threshold,
            distance_threshold_squared,
            rotation_always_close,
            chunk_size,
        )
        # A pose is not its own neighbor.
        rows = torch.arange(reference_end - reference_offset, device=positions.device)
        block[rows, rows + reference_offset] = False
        close_counts[reference_offset:reference_end] = block.sum(dim=1)
    return close_counts


def _greedy_keep_indices(
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    visit_order: torch.Tensor,
    rotation_cosine_threshold: float,
    distance_threshold_squared: float,
    rotation_always_close: bool,
    chunk_size: int,
) -> torch.Tensor:
    """Apply greedy suppression over batched pairwise closeness rows.

    References are visited strictly in ``visit_order``. Closeness rows are
    computed in bounded blocks, and only for references that are still alive
    when their block starts; a reference suppressed earlier can never be kept,
    so skipping its row is semantics-preserving and makes the run time scale
    with the number of survivors rather than with the raw candidate count.
    """
    num_poses = positions.shape[0]
    ordered_positions = positions[visit_order].contiguous()
    ordered_quaternions = quaternions[visit_order].contiguous()

    suppressed = np.zeros(num_poses, dtype=np.bool_)
    keep_ordered_indices: list[int] = []

    for reference_offset in range(0, num_poses, chunk_size):
        reference_end = min(reference_offset + chunk_size, num_poses)
        alive_local = np.flatnonzero(~suppressed[reference_offset:reference_end])
        if alive_local.size == 0:
            continue
        alive_ordered = alive_local + reference_offset
        alive_torch = torch.as_tensor(
            alive_ordered, dtype=torch.long, device=positions.device
        )
        block = _close_block(
            ordered_positions[alive_torch],
            ordered_quaternions[alive_torch],
            ordered_positions,
            ordered_quaternions,
            rotation_cosine_threshold,
            distance_threshold_squared,
            rotation_always_close,
            chunk_size,
        )
        # A pose never suppresses itself.
        rows = torch.arange(alive_torch.numel(), device=positions.device)
        block[rows, alive_torch] = False
        block_np = block.cpu().numpy()

        for row_idx, reference_idx in enumerate(alive_ordered):
            if suppressed[reference_idx]:
                continue
            keep_ordered_indices.append(int(reference_idx))
            suppressed |= block_np[row_idx]
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
    ``(N, 4)`` xyzw quaternions. Relative rotation and Euclidean relative
    translation are compared in bounded batched pairwise tiles.

    Args:
        poses: Input pose matrices. Shape is ``(N, 4, 4)``.
        angle_th: Rotation threshold in radians. Poses with angular distance
            below this value are considered close. Defaults to pi / 36.
        dist_th: Translation distance threshold. Poses with Euclidean distance
            below this value are considered close. Defaults to 0.003.
        preserve_order: Whether to greedily select poses in input order. If
            ``False``, poses with fewer close neighbors are selected first.
            Defaults to ``False``.
        chunk_size: Maximum size of either dimension of a pairwise tile.
            Defaults to 2048.

    Returns:
        Indices of selected poses. Shape is ``(M,)``, where ``M <= N``.

    Raises:
        ValueError: If ``poses`` is not shaped as ``(N, 4, 4)``, is not on a
            CPU or CUDA device, or ``chunk_size`` is not positive.
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

    positions, quaternions = _poses_to_components(poses)
    # The pairwise threshold math is elementwise float32 with fixed reduction
    # order, so offloading it to CUDA produces the same decisions as the CPU
    # path. Indices are returned on the input device either way.
    if poses.device.type == "cpu" and torch.cuda.is_available():
        positions = positions.cuda()
        quaternions = quaternions.cuda()
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
        ).to(poses.device)
        tie_breaker = torch.arange(num_poses, dtype=torch.long, device=poses.device)
        visit_priority = close_counts * (num_poses + 1) + tie_breaker
        visit_order = torch.argsort(visit_priority)

    return _greedy_keep_indices(
        positions,
        quaternions,
        visit_order.to(positions.device),
        rotation_cosine_threshold,
        distance_threshold_squared,
        rotation_always_close,
        chunk_size,
    ).to(poses.device)


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
        chunk_size: Maximum size of either dimension of a pairwise tile.
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
