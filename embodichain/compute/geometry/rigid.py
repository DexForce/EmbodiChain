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

__all__ = [
    "nearest_half_turn_pose",
    "rotate_rigid_grasp",
    "rigid_pose_displacement",
    "pose_axis_alignment",
]


def pose_axis_alignment(
    pose: torch.Tensor,
    local_axis: Sequence[float],
    *,
    world_axis: Sequence[float] | None = None,
    direction_free: bool = False,
) -> torch.Tensor:
    """Project a caller-normalized object axis onto a world-frame direction.

    Args:
        pose: Observed rigid transforms, shaped ``(B, 4, 4)``.
        local_axis: Three object-local axis coordinates, normalized by the caller.
        world_axis: Three normalized world coordinates; None selects positive Z.
        direction_free: Whether either sign of the axis is acceptable.

    Returns:
        Per-row dot products, or their absolute values. Nonfinite pose values
        remain nonfinite evidence; this helper selects no acceptance threshold.
    """
    if pose.ndim != 3 or pose.shape[-2:] != (4, 4) or not pose.is_floating_point():
        raise ValueError("pose must be floating transforms with shape (B, 4, 4).")
    if type(direction_free) is not bool:
        raise TypeError("direction_free must be a bool.")
    local = pose.new_tensor(local_axis)
    world = pose.new_tensor((0.0, 0.0, 1.0) if world_axis is None else world_axis)
    if local.shape != (3,) or world.shape != (3,):
        raise ValueError("Alignment axes must have shape (3,).")
    alignment = ((pose[:, :3, :3] @ local) * world).sum(-1)
    return alignment.abs() if direction_free else alignment


def rigid_pose_displacement(
    pose: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure translation and rotation magnitudes between corresponding poses.

    Args:
        pose: Rigid transforms, shape ``(B, 4, 4)``.
        reference: Previous transforms with the same shape, dtype and device.

    Returns:
        Translation distances in metres and rotation angles in radians, each
        shaped ``(B,)``. Nonfinite input remains nonfinite evidence; this function
        does not choose a stability threshold or certify transform validity.
    """
    if (
        pose.ndim != 3
        or pose.shape[-2:] != (4, 4)
        or reference.shape != pose.shape
        or reference.device != pose.device
        or reference.dtype != pose.dtype
        or not pose.is_floating_point()
    ):
        raise ValueError(
            "Pose pairs must share floating (B, 4, 4) shape, dtype and device."
        )
    translation = torch.linalg.vector_norm(pose[:, :3, 3] - reference[:, :3, 3], dim=-1)
    rotation = reference[:, :3, :3].transpose(-1, -2) @ pose[:, :3, :3]
    cosine = (rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    return translation, torch.acos(cosine.clamp(min=-1.0, max=1.0))


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
