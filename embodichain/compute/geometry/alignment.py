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

"""Numeric alignment targets retaining source orientation and lift ordering."""

from __future__ import annotations
from collections.abc import Sequence
import torch
from embodichain.utils.math import axis_angle_to_rotation_matrix

__all__ = ["alignment_target_axis", "held_alignment_targets"]


def alignment_target_axis(
    source: torch.Tensor, *, direction_free: bool
) -> torch.Tensor:
    """Keep an undirected vertical axis on its nearest pole; upright remains +Z.

    Args:
        source: Batched observed axis directions, shape ``(B, 3)``.
        direction_free: Allow the negative pole for a downward source direction.

    Returns:
        Owned signed vertical unit directions with the source dtype and device.
    """
    target = torch.zeros_like(source)
    target[:, 2] = torch.where(source[:, 2] < 0, -1.0, 1.0) if direction_free else 1.0
    return target


def held_alignment_targets(
    object_pose: torch.Tensor,
    local_axis: Sequence[float],
    *,
    position: Sequence[float] | None,
    direction_free: bool,
    preserve_yaw: bool,
    object_vertices: torch.Tensor | None = None,
    root_pose: torch.Tensor | None = None,
    object_to_eef: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build declared alignment/lift operands from already observed numeric poses.

    Args:
        object_pose: Observed world object poses, shape ``(B,4,4)``.
        local_axis: Declared object-local alignment axis.
        position: Optional exact world destination; omitted retains current origin.
        direction_free: Preserve the nearest vertical pole instead of forcing +Z.
        preserve_yaw: Return one aligned pose without source arm-facing selection.
        object_vertices: Scaled local geometry for direction-free height correction.
        root_pose: Observed active-arm root needed for yaw-free arm-facing selection.
        object_to_eef: Verified held transform for that same active resource.

    Returns:
        Owned single target or two-pose acquired-heading lift/aligned-target grid.
        No IK, scene lookup, collision qualification or success judgment occurs.
    """
    pose = object_pose.clone()
    staging_pose = pose.clone()
    source = torch.nn.functional.normalize(
        pose[:, :3, :3] @ pose.new_tensor(local_axis), dim=-1
    )
    target = alignment_target_axis(source, direction_free=direction_free)
    cross = torch.cross(source, target, dim=-1)
    sine = torch.linalg.vector_norm(cross, dim=-1)
    cosine = (source * target).sum(-1).clamp(-1, 1)
    rotation_axis = cross / sine.clamp_min(1e-8)[:, None]
    basis = torch.eye(3, device=pose.device, dtype=pose.dtype)[source.abs().argmin(-1)]
    perpendicular = torch.nn.functional.normalize(
        torch.cross(source, basis, dim=-1), dim=-1
    )
    rotation_axis = torch.where((sine < 1e-8)[:, None], perpendicular, rotation_axis)
    correction = axis_angle_to_rotation_matrix(
        rotation_axis * torch.atan2(sine, cosine)[:, None]
    )
    pose[:, :3, :3] = correction @ pose[:, :3, :3]
    if position is not None:
        pose[:, :3, 3] = pose.new_tensor(position)
        if direction_free and (target[:, 2] < 0).any():
            if object_vertices is None:
                raise ValueError(
                    "Undirected alignment requires object-local mesh geometry."
                )
            heights = object_vertices.to(pose) @ pose.new_tensor(local_axis)
            pose[:, 2, 3] += torch.where(
                target[:, 2] < 0, heights.amin() + heights.amax(), 0.0
            )
        staging_pose[:, :2, 3] = pose[:, :2, 3]
        staging_pose[:, 2, 3] = torch.maximum(staging_pose[:, 2, 3], pose[:, 2, 3])
    if preserve_yaw:
        return pose
    if root_pose is None or object_to_eef is None:
        raise ValueError("Yaw-free alignment requires explicit root and held frames.")
    toward_object = pose[:, :2, 3] - root_pose.to(pose)[..., :2, 3]
    object_to_eef = object_to_eef.to(pose)
    approach = (pose[:, :3, :3] @ object_to_eef[..., :3, :3])[:, :2, 2]
    flip = (torch.linalg.vector_norm(approach, dim=-1) > 0.1) & (
        (approach * toward_object).sum(-1) < 0
    )
    half_turn = torch.diag(pose.new_tensor([-1.0, -1.0, 1.0]))
    pose[:, :3, :3] = torch.where(
        flip[:, None, None], half_turn @ pose[:, :3, :3], pose[:, :3, :3]
    )
    return torch.stack((staging_pose, pose), dim=1) if position is not None else pose
