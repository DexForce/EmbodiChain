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

"""Broadcastable Torch box arithmetic without host reads of tensor values."""

from __future__ import annotations

import math

import torch

__all__ = ["aabb_overlap_depth", "aabb_containment_violation", "group_height_offset"]


def _float_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        value.dtype not in (torch.float32, torch.float64)
        or value.layout != torch.strided
    ):
        raise ValueError(f"{name} must be a strided float32 or float64 tensor")


def _same_type(reference: torch.Tensor, other: torch.Tensor) -> None:
    if reference.dtype != other.dtype or reference.device != other.device:
        raise ValueError("Geometry inputs must have the same dtype and device")


def _bounds_tensor(value: torch.Tensor, name: str) -> None:
    _float_tensor(value, name)
    if value.ndim < 2 or value.shape[-2] != 2 or value.shape[-1] < 1:
        raise ValueError(f"{name} must have shape (..., 2, D) with D >= 1")


def _valid_bounds(value: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(value).all(dim=(-2, -1)) & (
        value[..., 0, :] <= value[..., 1, :]
    ).all(dim=-1)


def _pair_bounds(first: torch.Tensor, second: torch.Tensor) -> None:
    _bounds_tensor(first, "bounds_a")
    _bounds_tensor(second, "bounds_b")
    _same_type(first, second)
    if first.shape[-1] != second.shape[-1]:
        raise ValueError("Bounds must have the same coordinate dimension")
    torch.broadcast_shapes(first.shape[:-2], second.shape[:-2])


def aabb_overlap_depth(bounds_a: torch.Tensor, bounds_b: torch.Tensor) -> torch.Tensor:
    """Return nonnegative intersection lengths along every coordinate axis.

    Args:
        bounds_a: Minimum and maximum corners, shaped ``(..., 2, D)``.
        bounds_b: Corresponding or broadcastable boxes, shaped ``(..., 2, D)``.

    Returns:
        Intersection lengths, shaped ``broadcast(...), D``. A pair intersects
        with positive volume only when every component is positive. Touching
        faces and disjoint axes have zero depth. Invalid (nonfinite or reversed)
        input boxes produce an all-NaN row without synchronizing the device.

    Inputs must share float32/float64 dtype and device. Add singleton leading
    axes explicitly for pairwise comparisons; this function does not infer an
    object axis or materialize an implicit N-by-N matrix.
    """
    _pair_bounds(bounds_a, bounds_b)
    depth = (
        torch.minimum(bounds_a[..., 1, :], bounds_b[..., 1, :])
        - torch.maximum(bounds_a[..., 0, :], bounds_b[..., 0, :])
    ).clamp_min(0)
    valid = (
        _valid_bounds(bounds_a)
        & _valid_bounds(bounds_b)
        & torch.isfinite(depth).all(dim=-1)
    )
    return torch.where(valid[..., None], depth, float("nan"))


def aabb_containment_violation(
    bounds: torch.Tensor, region: torch.Tensor
) -> torch.Tensor:
    """Return the largest boundary violation on each coordinate axis.

    Args:
        bounds: Object minimum and maximum corners, shaped ``(..., 2, D)``.
        region: Allowed minimum and maximum corners, shaped ``(..., 2, D)``.

    Returns:
        Nonnegative violation distances, shaped ``broadcast(...), D``. A fully
        contained box has a zero row, including boundary contact. Invalid boxes
        produce all-NaN rows. Tensor values remain on their original device;
        only shape/dtype/device metadata is validated on the host.
    """
    _pair_bounds(bounds, region)
    violation = torch.maximum(
        region[..., 0, :] - bounds[..., 0, :], bounds[..., 1, :] - region[..., 1, :]
    ).clamp_min(0)
    valid = (
        _valid_bounds(bounds)
        & _valid_bounds(region)
        & torch.isfinite(violation).all(dim=-1)
    )
    return torch.where(valid[..., None], violation, float("nan"))


def group_height_offset(
    bounds: torch.Tensor,
    support_z: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    clearance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a common Z translation aligning a group's lowest box to a height.

    Args:
        bounds: Group boxes, shaped ``(..., N, 2, 3)``. Leading dimensions are
            independent batches; N is the number of objects within one group.
        support_z: Target support height, broadcastable to the leading batch
            dimensions. Must share the bounds' float32/float64 dtype and device.
        valid_mask: Optional bool tensor, shaped ``(..., N)``, on the same
            device. False entries are padding; their numeric values are ignored.
        clearance: Finite Python scalar added to the target height, in metres.

    Returns:
        ``(offset, valid)`` with the broadcast batch shape. Empty/all-padding
        groups, nonfinite support heights, or any unmasked invalid box return
        zero offset and false validity. The validity tensor is never read by the
        host. A valid offset translates every group member by the same amount.
    """
    _bounds_tensor(bounds, "bounds")
    _float_tensor(support_z, "support_z")
    _same_type(bounds, support_z)
    if bounds.ndim < 3 or bounds.shape[-1] != 3:
        raise ValueError("Group bounds must have shape (..., N, 2, 3)")
    if type(clearance) not in (int, float) or not math.isfinite(clearance):
        raise ValueError("clearance must be a finite Python scalar")
    count = bounds.shape[-3]
    batch_shape = torch.broadcast_shapes(bounds.shape[:-3], support_z.shape)
    if valid_mask is not None:
        if (
            not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.layout != torch.strided
        ):
            raise ValueError("valid_mask must be a strided boolean tensor")
        if valid_mask.device != bounds.device:
            raise ValueError("valid_mask must be on the bounds device")
        if valid_mask.ndim < 1 or valid_mask.shape[-1] != count:
            raise ValueError("valid_mask must have shape (..., N)")
        batch_shape = torch.broadcast_shapes(batch_shape, valid_mask.shape[:-1])
    if count == 0:
        return bounds.new_zeros(batch_shape), torch.zeros(
            batch_shape, dtype=torch.bool, device=bounds.device
        )
    expanded = bounds.expand(*batch_shape, count, 2, 3)
    selected = (
        torch.ones((*batch_shape, count), dtype=torch.bool, device=bounds.device)
        if valid_mask is None
        else valid_mask.expand(*batch_shape, count)
    )
    valid = (
        selected.any(dim=-1)
        & (~selected | _valid_bounds(expanded)).all(dim=-1)
        & torch.isfinite(support_z)
    )
    lowest = torch.where(selected, expanded[..., 0, 2], float("inf")).amin(dim=-1)
    offset = support_z + clearance - lowest
    valid = valid & torch.isfinite(offset)
    return torch.where(valid, offset, 0.0), valid
