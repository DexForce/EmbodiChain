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

from embodichain.utils import logger
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


def _reference_block_rows(num_poses: int, chunk_size: int) -> int:
    """Bound reference-block rows so a block holds ~``chunk_size**2`` entries."""
    return max(1, min(chunk_size, chunk_size * chunk_size // max(1, num_poses)))


def _close_tile(
    ref_positions: torch.Tensor,
    ref_quaternions: torch.Tensor,
    tile_positions: torch.Tensor,
    tile_quaternions: torch.Tensor,
    rotation_cosine_threshold: float,
    distance_threshold_squared: float,
    rotation_always_close: bool,
) -> torch.Tensor:
    """Compute one bounded ``(R, T)`` tile of the pairwise closeness matrix.

    For unit xyzw quaternions, the real component of
    ``inverse(q_reference) * q_target`` is their dot product. Its absolute
    value gives the shortest relative rotation while treating ``q`` and ``-q``
    equally. Translation closeness compares the squared Euclidean distance.
    The arithmetic uses explicit per-component multiplies and left-to-right
    additions — the exact association of the previous scalar implementation —
    so decisions are identical across CPU and CUDA backends.

    Args:
        ref_positions: Reference positions with shape ``(R, 3)``.
        ref_quaternions: Reference quaternions with shape ``(R, 4)``.
        tile_positions: Target-tile positions with shape ``(T, 3)``.
        tile_quaternions: Target-tile quaternions with shape ``(T, 4)``.
        rotation_cosine_threshold: ``cos(angle_th / 2)`` decision value.
        distance_threshold_squared: Squared translation threshold.
        rotation_always_close: Whether rotation is trivially satisfied.

    Returns:
        Boolean closeness tile with shape ``(R, T)``.
    """
    dx = tile_positions[None, :, 0] - ref_positions[:, None, 0]
    dy = tile_positions[None, :, 1] - ref_positions[:, None, 1]
    dz = tile_positions[None, :, 2] - ref_positions[:, None, 2]
    tile_close = dx * dx + dy * dy + dz * dz < distance_threshold_squared
    if not rotation_always_close:
        dots = (
            ref_quaternions[:, None, 0] * tile_quaternions[None, :, 0]
            + ref_quaternions[:, None, 1] * tile_quaternions[None, :, 1]
            + ref_quaternions[:, None, 2] * tile_quaternions[None, :, 2]
            + ref_quaternions[:, None, 3] * tile_quaternions[None, :, 3]
        )
        tile_close &= dots.abs() > rotation_cosine_threshold
    return tile_close


def _count_close_poses(
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    rotation_cosine_threshold: float,
    distance_threshold_squared: float,
    rotation_always_close: bool,
    chunk_size: int,
) -> torch.Tensor:
    """Count close neighbors, accumulating per bounded pairwise tile.

    Counts are summed tile by tile, so no ``(rows, num_poses)`` matrix is ever
    materialized and peak memory stays bounded by ``chunk_size ** 2`` entries.
    """
    num_poses = positions.shape[0]
    close_counts = torch.zeros(num_poses, dtype=torch.int64, device=positions.device)
    for reference_offset in range(0, num_poses, chunk_size):
        reference_end = min(reference_offset + chunk_size, num_poses)
        ref_positions = positions[reference_offset:reference_end]
        ref_quaternions = quaternions[reference_offset:reference_end]
        block_counts = torch.zeros(
            reference_end - reference_offset,
            dtype=torch.int64,
            device=positions.device,
        )
        for target_offset in range(0, num_poses, chunk_size):
            target_end = min(target_offset + chunk_size, num_poses)
            tile = _close_tile(
                ref_positions,
                ref_quaternions,
                positions[target_offset:target_end],
                quaternions[target_offset:target_end],
                rotation_cosine_threshold,
                distance_threshold_squared,
                rotation_always_close,
            )
            # A pose is not its own neighbor: clear the diagonal overlap.
            overlap_start = max(reference_offset, target_offset)
            overlap_end = min(reference_end, target_end)
            if overlap_start < overlap_end:
                diagonal = torch.arange(
                    overlap_start, overlap_end, device=positions.device
                )
                tile[diagonal - reference_offset, diagonal - target_offset] = False
            block_counts += tile.sum(dim=1)
        close_counts[reference_offset:reference_end] = block_counts
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

    # Keep every closeness block bounded to roughly chunk_size**2 entries even
    # when there are far more poses than one target tile.
    reference_block_rows = _reference_block_rows(num_poses, chunk_size)
    suppressed = np.zeros(num_poses, dtype=np.bool_)
    keep_ordered_indices: list[int] = []

    for reference_offset in range(0, num_poses, reference_block_rows):
        reference_end = min(reference_offset + reference_block_rows, num_poses)
        alive_local = np.flatnonzero(~suppressed[reference_offset:reference_end])
        if alive_local.size == 0:
            continue
        alive_ordered = alive_local + reference_offset
        alive_torch = torch.as_tensor(
            alive_ordered, dtype=torch.long, device=positions.device
        )
        ref_positions = ordered_positions[alive_torch]
        ref_quaternions = ordered_quaternions[alive_torch]
        block = torch.empty(
            alive_torch.numel(),
            num_poses,
            dtype=torch.bool,
            device=positions.device,
        )
        for target_offset in range(0, num_poses, chunk_size):
            target_end = min(target_offset + chunk_size, num_poses)
            block[:, target_offset:target_end] = _close_tile(
                ref_positions,
                ref_quaternions,
                ordered_positions[target_offset:target_end],
                ordered_quaternions[target_offset:target_end],
                rotation_cosine_threshold,
                distance_threshold_squared,
                rotation_always_close,
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

    base_positions, base_quaternions = _poses_to_components(poses)
    rotation_always_close = angle_th > math.pi
    rotation_cosine_threshold = (
        math.cos(0.5 * float(angle_th)) if not rotation_always_close else 0.0
    )
    distance_threshold_squared = float(dist_th * dist_th)

    def _select_indices(
        positions: torch.Tensor, quaternions: torch.Tensor
    ) -> torch.Tensor:
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

    # The pairwise threshold math is elementwise float32 with explicit
    # component association, so offloading it to CUDA produces the same
    # decisions as the CPU path. Indices return on the input device, and any
    # CUDA failure (initialization, memory pressure) falls back to the CPU
    # path that previously served these requests.
    if poses.device.type == "cpu" and torch.cuda.is_available():
        try:
            return _select_indices(base_positions.cuda(), base_quaternions.cuda())
        except RuntimeError as error:
            logger.log_warning(
                f"pose_nms CUDA offload failed ({error}); falling back to CPU."
            )
    return _select_indices(base_positions, base_quaternions)


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
