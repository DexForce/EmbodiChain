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

"""Pure rigid-pose geometry shared by motion primitives and case generation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    get_relative_rotation,
    pose_inv,
)

__all__ = ["nearest_half_turn_pose", "rotate_rigid_grasp"]


def nearest_half_turn_pose(
    target_pose: torch.Tensor, reference_pose: torch.Tensor
) -> torch.Tensor:
    """Select the closest of two local-Z half-turn grasp orientations.

    Args:
        target_pose: Candidate rigid transforms, shaped ``(B, 4, 4)``.
        reference_pose: Reference transforms with the same shape and dtype.

    Returns:
        Selected poses with unchanged translation, dtype and device. Ties
        select the half-turn variant. Neither input is modified.
    """
    symmetric = target_pose.clone()
    symmetric[:, :3, :2] *= -1
    angle = get_relative_rotation(reference_pose[:, :3, :3], target_pose[:, :3, :3])
    symmetric_angle = get_relative_rotation(
        reference_pose[:, :3, :3], symmetric[:, :3, :3]
    )
    return torch.where((angle < symmetric_angle)[:, None, None], target_pose, symmetric)


def rotate_rigid_grasp(
    link_pose: torch.Tensor,
    grasp_pose: torch.Tensor,
    axis: torch.Tensor,
    axis_origin: Sequence[float],
    angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a link around its local axis while preserving a rigid grasp.

    Args:
        link_pose: Initial link transforms, shaped ``(B, 4, 4)``.
        grasp_pose: Initial TCP transforms, shaped ``(B, 4, 4)``.
        axis: Nonzero local rotation axis, shaped ``(3,)``.
        axis_origin: Three coordinates of the local rotation origin.
        angles: Rotation angles in radians, shaped ``(W,)`` or ``(B, W)``.

    Returns:
        Link and TCP transforms, each shaped ``(B, W, 4, 4)``, in the input
        link's dtype and device. Inputs remain unchanged. The caller owns
        physical feasibility and motion-limit validation.
    """
    axis = axis.to(device=link_pose.device, dtype=link_pose.dtype)
    axis = axis / torch.linalg.vector_norm(axis)
    angles = angles.to(device=link_pose.device, dtype=link_pose.dtype)
    if angles.ndim == 1:
        angles = angles[None].expand(link_pose.shape[0], -1)
    rotations = torch.eye(4, dtype=link_pose.dtype, device=link_pose.device).repeat(
        link_pose.shape[0], angles.shape[-1], 1, 1
    )
    rotations[..., :3, :3] = axis_angle_to_rotation_matrix(angles[..., None] * axis)
    origin = torch.as_tensor(
        axis_origin, dtype=link_pose.dtype, device=link_pose.device
    )
    to_origin = torch.eye(4, dtype=link_pose.dtype, device=link_pose.device)
    from_origin = to_origin.clone()
    to_origin[:3, 3] = origin
    from_origin[:3, 3] = -origin
    rotated_link = link_pose[:, None] @ (to_origin @ rotations @ from_origin)
    link_to_eef = pose_inv(link_pose) @ grasp_pose
    return rotated_link, rotated_link @ link_to_eef[:, None]
