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
import math

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
    "rotate_held_object_world_z",
    "axis_contact_pose",
    "pose_axis_alignment",
    "validate_rigid_pose_batch",
    "rotate_final_pose_world_z",
    "support_mesh_envelope",
]


def axis_contact_pose(
    target_pose: torch.Tensor,
    local_axis: torch.Tensor,
    local_point: Sequence[float],
    roll: float,
) -> torch.Tensor:
    """Construct an axis-aligned contact frame with the source float32 convention.

    Args:
        target_pose: Batched target-link transforms, shape ``(B, 4, 4)``.
        local_axis: Finite nonzero target-local Z direction, shape ``(3,)``.
        local_point: Three finite target-local contact coordinates.
        roll: Finite rotation of the contact X/Y axes about contact Z, in radians.

    Returns:
        Owned float32 contact poses on the target's device. The least-aligned
        Cartesian axis completes the frame; ties preserve ``argmin`` ordering.
        Position and Z remain fixed while the declared roll rotates X/Y.

    This pure kernel retains the source conversion, normalization and roll order.
    Callers validate physical transforms and finite calibration values; this
    construction performs no reachability, collision or contact qualification.
    """
    if target_pose.dim() != 3 or target_pose.shape[1:] != (4, 4):
        raise ValueError("Target pose must have shape (B, 4, 4).")
    if local_axis.shape != (3,) or len(local_point) != 3:
        raise ValueError("Contact axis and point must have three coordinates.")
    target_pose = target_pose.to(dtype=torch.float32)
    device = target_pose.device
    local_axis = local_axis.to(device=device, dtype=torch.float32)
    local_axis = local_axis / torch.linalg.vector_norm(local_axis)
    z_axis = torch.matmul(target_pose[:, :3, :3], local_axis)
    z_axis = torch.nn.functional.normalize(z_axis, dim=1)
    basis = torch.eye(3, dtype=z_axis.dtype, device=device)
    reference = basis[torch.argmin(torch.abs(z_axis), dim=1)]
    y_axis = torch.nn.functional.normalize(
        torch.linalg.cross(reference, z_axis, dim=1), dim=1
    )
    x_axis = torch.nn.functional.normalize(
        torch.linalg.cross(y_axis, z_axis, dim=1), dim=1
    )
    pose = torch.eye(4, dtype=torch.float32, device=device).repeat(
        target_pose.shape[0], 1, 1
    )
    pose[:, :3, 0], pose[:, :3, 1], pose[:, :3, 2] = x_axis, y_axis, z_axis
    point = torch.tensor(local_point, dtype=torch.float32, device=device)
    pose[:, :3, 3] = torch.matmul(target_pose[:, :3, :3], point) + target_pose[:, :3, 3]
    angles = torch.full((len(target_pose),), roll, dtype=pose.dtype, device=device)
    result = pose.clone()
    cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
    result[:, :3, 0] = cosine * pose[:, :3, 0] + sine * pose[:, :3, 1]
    result[:, :3, 1] = -sine * pose[:, :3, 0] + cosine * pose[:, :3, 1]
    return result


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


def validate_rigid_pose_batch(pose: torch.Tensor, batch_size: int) -> None:
    """Validate observed finite, proper rigid transforms without changing them.

    Args:
        pose: Float32/float64 transforms shaped ``(batch_size, 4, 4)``.
        batch_size: Exact expected number of observations.

    Raises:
        ValueError: Shape, finiteness, rotation or homogeneous row is invalid.

    The source observation tolerances are retained: orthogonality/determinant
    use absolute tolerance 1e-4 and homogeneous rows 1e-5, with Torch's default
    relative tolerance. This validates geometry, not observation authenticity.
    """
    if (
        not isinstance(pose, torch.Tensor)
        or pose.dtype not in (torch.float32, torch.float64)
        or type(batch_size) is not int
        or batch_size < 1
        or pose.shape != (batch_size, 4, 4)
        or not torch.isfinite(pose).all()
    ):
        raise ValueError("Placement pose does not match the observed batch.")
    rotation = pose[:, :3, :3]
    if (
        not torch.allclose(
            rotation.transpose(-1, -2) @ rotation,
            torch.eye(3, dtype=pose.dtype, device=pose.device).expand_as(rotation),
            atol=1e-4,
        )
        or not torch.allclose(
            torch.linalg.det(rotation), pose.new_ones(batch_size), atol=1e-4
        )
        or not torch.allclose(
            pose[:, 3],
            pose.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(batch_size, -1),
            atol=1e-5,
        )
    ):
        raise ValueError("Placement requires rigid, non-reflected native transforms.")


def rotate_held_object_world_z(
    eef: torch.Tensor, held: torch.Tensor, angle: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate an attached object about world Z, keeping its origin and attachment.

    Args:
        eef: World end-effector transforms, shape ``(B, 4, 4)``.
        held: Object-to-end-effector attachment transforms of the same shape.
        angle: World-Z rotation in radians.

    Returns:
        Rotated end-effector and object transforms, preserving source arithmetic.
    """
    object_pose = eef @ pose_inv(held)
    yaw = eef.new_tensor(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    changed = object_pose.clone()
    changed[:, :3, :3] = yaw @ object_pose[:, :3, :3]
    return changed @ held, changed


def rotate_final_pose_world_z(poses: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate only the final pose orientation about world Z, retaining origins.

    Args:
        poses: One pose or ordered poses per row, ``(B, 4, 4)`` or ``(B, T, 4, 4)``.
        angle: World-Z heading offset in radians.

    Returns:
        Owned poses with unchanged translations and intermediate orientations.
    """
    changed = poses.clone()
    final = changed[:, -1] if changed.ndim == 4 else changed
    final[:, :3, :3] = (
        final.new_tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        @ final[:, :3, :3]
    )
    return changed


def support_mesh_envelope(
    object_vertices: torch.Tensor,
    reference_vertices: torch.Tensor,
    object_pose: torch.Tensor,
    reference_pose: torch.Tensor,
    *,
    inset_xy: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform source mesh envelopes without selecting support acceptance.

    Args:
        object_vertices: Local object vertices, shaped (B, V, 3).
        reference_vertices: Local reference vertices, shaped (B, W, 3).
        object_pose: Matching object transforms, shaped (B, 4, 4).
        reference_pose: Matching reference transforms, shaped (B, 4, 4).
        inset_xy: Explicit nonnegative shrink of the horizontal reference bounds.

    Returns:
        Object bottom (B,), reference top (B,), horizontal lower and upper
        reference bounds (B, 2), on the input dtype/device. Nonfinite values
        propagate into measurements. Mesh envelopes establish no native contact.
    """
    tensors = (object_vertices, reference_vertices, object_pose, reference_pose)
    batch = object_pose.shape[0] if object_pose.ndim == 3 else -1
    if any(
        not value.is_floating_point()
        or value.device != object_pose.device
        or value.dtype != object_pose.dtype
        for value in tensors
    ):
        raise ValueError("Envelope inputs must share floating dtype and device.")
    if any(
        value.shape != (batch, 4, 4) for value in (object_pose, reference_pose)
    ) or any(
        value.ndim != 3
        or value.shape[0] != batch
        or value.shape[1] == 0
        or value.shape[2] != 3
        for value in (object_vertices, reference_vertices)
    ):
        raise ValueError(
            "Envelopes require matching (B,V,3) vertices and (B,4,4) poses."
        )
    if (
        isinstance(inset_xy, bool)
        or not isinstance(inset_xy, (int, float))
        or not math.isfinite(inset_xy)
        or inset_xy < 0
    ):
        raise ValueError("inset_xy must be finite and nonnegative.")
    bottom = (object_vertices @ object_pose[:, 2, :3, None]).squeeze(-1).amin(
        -1
    ) + object_pose[:, 2, 3]
    world = (
        reference_vertices @ reference_pose[:, :3, :3].transpose(-1, -2)
        + reference_pose[:, None, :3, 3]
    )
    return (
        bottom,
        world[:, :, 2].amax(-1),
        world[:, :, :2].amin(1) + inset_xy,
        world[:, :, :2].amax(1) - inset_xy,
    )
