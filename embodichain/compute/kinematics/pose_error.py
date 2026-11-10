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

"""Finite-gradient SE(3) residuals for differentiable control."""

from __future__ import annotations

import torch

__all__ = ["pose_error"]


def _rotation_vector(rotation: torch.Tensor) -> torch.Tensor:
    # Select the best-conditioned quaternion formula. Clamp before sqrt:
    # masking sqrt(0) afterwards still introduces infinite/NaN adjoints.
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rotation.reshape(
        *rotation.shape[:-2], 9
    ).unbind(-1)
    squared = torch.stack(
        (
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ),
        -1,
    )
    roots = squared.clamp_min(torch.finfo(rotation.dtype).eps).sqrt()
    candidates = torch.stack(
        (
            torch.stack((squared[..., 0], m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, squared[..., 1], m10 + m01, m02 + m20), -1),
            torch.stack((m02 - m20, m10 + m01, squared[..., 2], m12 + m21), -1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, squared[..., 3]), -1),
        ),
        -2,
    ) / (2 * roots[..., :, None])
    index = squared.argmax(-1)[..., None, None].expand(*squared.shape[:-1], 1, 4)
    quat = candidates.gather(-2, index).squeeze(-2)
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    quat = torch.where(quat[..., :1] < 0, -quat, quat)
    vector, scalar = quat[..., 1:], quat[..., :1]
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, scalar)
    scale = torch.where(
        sine < 1e-4,
        2 + sine.square() / 3 + 3 * sine.pow(4) / 20,
        angle / sine.clamp_min(1e-4),
    )
    return vector * scale


def pose_error(
    current: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return translation and shortest root-frame rotation residuals.

    Args:
        current: Achieved rigid transforms, shape ``(..., 4, 4)``.
        target: Desired rigid transforms, broadcastable to ``current``.

    Returns:
        Translation and axis-angle vectors, each with shape ``(..., 3)``.

    Note:
        Inputs must be valid finite SE(3) transforms. The shortest rotation
        has a branch cut at pi; no unique smooth derivative exists there.
        The zero-rotation limit has finite first derivatives.
    """
    if current.shape[-2:] != (4, 4) or target.shape[-2:] != (4, 4):
        raise ValueError("Expected transforms ending in (4, 4)")
    return target[..., :3, 3] - current[..., :3, 3], _rotation_vector(
        target[..., :3, :3] @ current[..., :3, :3].transpose(-1, -2)
    )
