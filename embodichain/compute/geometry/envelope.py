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

"""Signed world-axis envelope separation over explicit numeric geometry."""

from __future__ import annotations
from collections.abc import Sequence
import torch
from .rigid import validate_rigid_pose_batch

__all__ = ["signed_envelope_gaps"]


def signed_envelope_gaps(
    object_vertices: torch.Tensor,
    object_pose: torch.Tensor,
    reference_vertices: torch.Tensor,
    reference_pose: torch.Tensor,
    axes: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Measure signed separations with the original transform/reduction order.

    Args:
        object_vertices: Local payload vertices, shape (B,V,3).
        object_pose: Payload world poses, shape (B,4,4).
        reference_vertices: Local reference points, shape (R,3) or (B,R,3).
        reference_pose: Reference world poses, shape (B,4,4).
        axes: Explicit numeric (axis,sign) selections. No relation vocabulary is inferred.

    Returns:
        Gaps of shape (A,B), retaining pose dtype/device. No separation threshold
        or task acceptance is applied.
    """
    count = object_pose.shape[0]
    validate_rigid_pose_batch(object_pose, count)
    validate_rigid_pose_batch(reference_pose, count)
    selected = tuple(axes)
    if not selected or any(
        type(axis) is not int
        or axis not in (0, 1, 2)
        or type(sign) is not int
        or sign not in (-1, 1)
        for axis, sign in selected
    ):
        raise ValueError("Envelope axes require explicit axis indices and signs.")
    if (
        object_vertices.ndim != 3
        or object_vertices.shape[0] != count
        or not object_vertices.shape[1]
        or object_vertices.shape[-1] != 3
    ):
        raise ValueError("Object envelopes require native batched vertices.")
    reference_vertices = torch.as_tensor(
        reference_vertices, dtype=reference_pose.dtype, device=reference_pose.device
    )
    if (
        reference_vertices.ndim not in (2, 3)
        or reference_vertices.shape[-1] != 3
        or not reference_vertices.shape[-2]
        or reference_vertices.ndim == 3
        and reference_vertices.shape[0] != count
    ):
        raise ValueError("Reference envelopes require matching local vertex rows.")
    world = (
        object_vertices.to(object_pose) @ object_pose[:, :3, :3].transpose(-1, -2)
        + object_pose[:, None, :3, 3]
    )
    reference = (
        reference_vertices @ reference_pose[:, :3, :3].transpose(-1, -2)
        + reference_pose[:, None, :3, 3]
    )
    return torch.stack(
        [
            (world[:, :, axis] * sign).amin(-1)
            - (reference[:, :, axis] * sign).amax(-1)
            for axis, sign in selected
        ]
    )
