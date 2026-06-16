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

import torch
import numpy as np
import math

_POSE_NMS_CHUNK_SIZE = 2048


def _pose_pair_close(
    reference_rotations: torch.Tensor,
    reference_translations: torch.Tensor,
    reference_translation_norms: torch.Tensor,
    target_rotations: torch.Tensor,
    target_translations: torch.Tensor,
    target_translation_norms: torch.Tensor,
    rotation_cosine_th: float,
    dist_th_sq: float,
    rotation_always_close: bool,
) -> torch.Tensor:
    if rotation_always_close:
        close = torch.ones(
            (reference_rotations.shape[0], target_rotations.shape[0]),
            dtype=torch.bool,
            device=reference_rotations.device,
        )
    else:
        close = ((reference_rotations @ target_rotations.T - 1.0) * 0.5).clamp_(
            min=-1.0, max=1.0
        ) > rotation_cosine_th

    translation_distances_sq = (
        reference_translation_norms[:, None]
        + target_translation_norms[None, :]
        - 2.0 * (reference_translations @ target_translations.T)
    ).clamp_min_(0.0)
    close &= translation_distances_sq < dist_th_sq
    return close


def pose_nms_indices(
    poses: torch.Tensor,
    angle_th: float = np.pi / 36,
    dist_th: float = 0.003,
    preserve_order: bool = False,
    chunk_size: int = _POSE_NMS_CHUNK_SIZE,
) -> torch.Tensor:
    """Return pose indices after removing poses that are too close.

    This computes pairwise translation and rotation closeness in chunks, then
    greedily keeps either input-order poses or poses with fewer close neighbors
    first. The implementation avoids storing the full O(N^2) closeness matrix.

    Args:
        poses: Input pose matrices. Shape is (N, 4, 4).
        angle_th: Rotation threshold in radians. Poses with angular distance
            below this value are considered close. Defaults to pi / 36.
        dist_th: Translation distance threshold. Poses with Euclidean distance
            below this value are considered close. Defaults to 0.003.
        preserve_order: Whether to apply the same input-order greedy selection
            as :func:`pose_nms`. If False, poses with fewer close neighbors are
            selected first. Defaults to False.
        chunk_size: Maximum number of poses per pairwise comparison block.
            Larger values are usually faster but use more temporary memory.
            Defaults to 2048.

    Returns:
        Indices of selected poses. Shape is (M,), where M <= N.

    Raises:
        ValueError: If ``poses`` is not shaped as (N, 4, 4).
    """
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Invalid input shape {poses.shape}, expected (N, 4, 4).")
    if chunk_size <= 0:
        raise ValueError(
            f"Invalid chunk_size {chunk_size}, expected a positive integer."
        )

    num_poses = poses.shape[0]
    if num_poses == 0:
        return torch.empty(0, dtype=torch.long, device=poses.device)

    if angle_th <= 0.0 or dist_th <= 0.0:
        return torch.arange(num_poses, dtype=torch.long, device=poses.device)

    rotations = poses[:, :3, :3].reshape(num_poses, -1)
    translations = poses[:, :3, 3]
    translation_norms = (translations * translations).sum(dim=1)
    dist_th_sq = float(dist_th * dist_th)
    rotation_always_close = angle_th > math.pi
    rotation_cosine_th = math.cos(float(angle_th)) if not rotation_always_close else 0.0

    if preserve_order:
        visit_order = torch.arange(num_poses, dtype=torch.long, device=poses.device)
    else:
        close_counts = torch.zeros(num_poses, dtype=torch.long, device=poses.device)
        for row_start in range(0, num_poses, chunk_size):
            row_end = min(row_start + chunk_size, num_poses)
            row_counts = torch.zeros(
                row_end - row_start, dtype=torch.long, device=poses.device
            )
            for target_start in range(0, num_poses, chunk_size):
                target_end = min(target_start + chunk_size, num_poses)
                close = _pose_pair_close(
                    rotations[row_start:row_end],
                    translations[row_start:row_end],
                    translation_norms[row_start:row_end],
                    rotations[target_start:target_end],
                    translations[target_start:target_end],
                    translation_norms[target_start:target_end],
                    rotation_cosine_th,
                    dist_th_sq,
                    rotation_always_close,
                )

                overlap_start = max(row_start, target_start)
                overlap_end = min(row_end, target_end)
                if overlap_start < overlap_end:
                    diagonal_rows = torch.arange(
                        overlap_start - row_start,
                        overlap_end - row_start,
                        dtype=torch.long,
                        device=poses.device,
                    )
                    diagonal_cols = torch.arange(
                        overlap_start - target_start,
                        overlap_end - target_start,
                        dtype=torch.long,
                        device=poses.device,
                    )
                    close[diagonal_rows, diagonal_cols] = False

                row_counts += close.sum(dim=1)
            close_counts[row_start:row_end] = row_counts

        tie_breaker = torch.arange(num_poses, dtype=torch.long, device=poses.device)
        visit_priority = close_counts * (num_poses + 1) + tie_breaker
        visit_order = torch.argsort(visit_priority)

    suppressed = torch.zeros(num_poses, dtype=torch.bool, device=poses.device)
    keep_indices_list: list[int] = []

    for pose_idx in visit_order.tolist():
        if suppressed[pose_idx]:
            continue
        keep_indices_list.append(pose_idx)
        for target_start in range(0, num_poses, chunk_size):
            target_end = min(target_start + chunk_size, num_poses)
            close = _pose_pair_close(
                rotations[pose_idx : pose_idx + 1],
                translations[pose_idx : pose_idx + 1],
                translation_norms[pose_idx : pose_idx + 1],
                rotations[target_start:target_end],
                translations[target_start:target_end],
                translation_norms[target_start:target_end],
                rotation_cosine_th,
                dist_th_sq,
                rotation_always_close,
            ).squeeze(0)
            suppressed[target_start:target_end] |= close
        suppressed[pose_idx] = True

    return torch.tensor(keep_indices_list, dtype=torch.long, device=poses.device)


def pose_nms(
    poses: torch.Tensor,
    angle_th: float = np.pi / 36,
    dist_th: float = 0.003,
    chunk_size: int = _POSE_NMS_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove poses that are too close in translation and rotation.

    Args:
        poses: Input pose matrices. Shape is (N, 4, 4).
        angle_th: Rotation threshold in radians. Poses with angular distance
            below this value are considered close. Defaults to pi / 36.
        dist_th: Translation distance threshold. Poses with Euclidean distance
            below this value are considered close. Defaults to 0.003.
        chunk_size: Maximum number of poses per pairwise comparison block.
            Larger values are usually faster but use more temporary memory.
            Defaults to 2048.

    Returns:
        Filtered pose matrices preserving the input order. Shape is (M, 4, 4),
        where M <= N.

        keep_indices: Indices of the poses that are kept after NMS. Shape is (M,).

    Raises:
        ValueError: If ``poses`` is not shaped as (N, 4, 4).
    """
    keep_indices = pose_nms_indices(
        poses,
        angle_th=angle_th,
        dist_th=dist_th,
        preserve_order=True,
        chunk_size=chunk_size,
    )
    return poses[keep_indices], keep_indices
