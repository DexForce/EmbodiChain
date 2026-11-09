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

"""Batched affine bounding-box transforms implemented with Torch tensors."""

from __future__ import annotations

import torch

from .aabb import _bounds_tensor, _float_tensor, _same_type, _valid_bounds

__all__ = ["transform_aabbs"]


def transform_aabbs(bounds: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
    """Compute exact axis-aligned bounds of affine-transformed input boxes.

    Args:
        bounds: Minimum/maximum corners, shaped ``(..., 2, 3)``.
        transforms: Broadcastable affine transforms, shaped ``(..., 4, 4)``.
            Rotation, nonuniform scale, shear and reflection are supported.

    Returns:
        Transformed minimum/maximum corners in the broadcast batch shape,
        preserving input dtype and device. Nonfinite/reversed boxes, nonfinite
        transforms, non-affine last rows and unrepresentable results produce
        all-NaN boxes using device-side masks. No rigid-transform policy is
        applied, and no tensor value is copied to the host.
    """
    _bounds_tensor(bounds, "bounds")
    _float_tensor(transforms, "transforms")
    _same_type(bounds, transforms)
    if bounds.shape[-1] != 3 or transforms.ndim < 2 or transforms.shape[-2:] != (4, 4):
        raise ValueError("Expected bounds (..., 2, 3) and transforms (..., 4, 4)")
    torch.broadcast_shapes(bounds.shape[:-2], transforms.shape[:-2])
    # Halve first to avoid overflow when same-sign endpoints are near dtype max.
    center = bounds[..., 0, :] * 0.5 + bounds[..., 1, :] * 0.5
    half = bounds[..., 1, :] * 0.5 - bounds[..., 0, :] * 0.5
    linear = transforms[..., :3, :3]
    new_center = (linear @ center[..., None]).squeeze(-1) + transforms[..., :3, 3]
    new_half = (linear.abs() @ half[..., None]).squeeze(-1)
    result = torch.stack((new_center - new_half, new_center + new_half), dim=-2)
    valid = _valid_bounds(bounds) & torch.isfinite(transforms).all(dim=(-2, -1))
    valid = (
        valid & (transforms[..., 3, :3] == 0).all(dim=-1) & (transforms[..., 3, 3] == 1)
    )
    valid = valid & torch.isfinite(result).all(dim=(-2, -1))
    return torch.where(valid[..., None, None], result, float("nan"))
