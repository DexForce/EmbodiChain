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

"""Batched triangle measurements without mesh-library or scene dependencies."""

from __future__ import annotations

import torch

from .aabb import _float_tensor

__all__ = ["triangle_geometry"]


def triangle_geometry(
    triangles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure triangle unit normals and areas across arbitrary leading batches.

    Args:
        triangles: Three 3D vertices per triangle, shaped ``(..., 3, 3)``, with
            float32/float64 coordinates on their original CPU or CUDA device.

    Returns:
        ``(normals, areas, valid)`` with shapes ``(..., 3)``, ``(...)`` and
        ``(...)``. Winding determines normal sign. Degenerate, nonfinite or
        numerically unrepresentable triangles return zero normals and areas
        with false validity. Numerical degeneracy includes cross products whose
        components cannot exceed their cancellation error bound, twice dtype
        epsilon times the sum of the two determinant-product magnitudes. This
        scale-relative arithmetic check is not a scene area or slope threshold.
        No tensor values are read on the host.
    """
    _float_tensor(triangles, "triangles")
    if triangles.ndim < 2 or triangles.shape[-2:] != (3, 3):
        raise ValueError("triangles must have shape (..., 3, 3)")
    edge_a = triangles[..., 1, :] - triangles[..., 0, :]
    edge_b = triangles[..., 2, :] - triangles[..., 0, :]
    # A fused multiply/subtract cross implementation may leave a nonzero
    # cancellation residue even for cross(a, a). Never normalize that residue.
    positive = torch.stack(
        (
            edge_a[..., 1] * edge_b[..., 2],
            edge_a[..., 2] * edge_b[..., 0],
            edge_a[..., 0] * edge_b[..., 1],
        ),
        dim=-1,
    )
    negative = torch.stack(
        (
            edge_a[..., 2] * edge_b[..., 1],
            edge_a[..., 0] * edge_b[..., 2],
            edge_a[..., 1] * edge_b[..., 0],
        ),
        dim=-1,
    )
    cross = positive - negative
    cancellation_bound = (2 * torch.finfo(triangles.dtype).eps) * (
        positive.abs() + negative.abs()
    )
    resolvable = (cross.abs() > cancellation_bound).any(dim=-1)
    # Scaling makes the norm robust to squaring very small or very large values.
    scale = cross.abs().amax(dim=-1)
    safe_scale = torch.where((scale > 0) & torch.isfinite(scale), scale, 1.0)
    length = (
        torch.linalg.vector_norm(cross / safe_scale[..., None], dim=-1) * safe_scale
    )
    valid = (
        torch.isfinite(triangles).all(dim=(-2, -1))
        & torch.isfinite(length)
        & (length > 0)
        & resolvable
    )
    safe_length = torch.where(valid, length, 1.0)
    normals = torch.where(valid[..., None], cross / safe_length[..., None], 0.0)
    areas = torch.where(valid, length * 0.5, 0.0)
    return normals, areas, valid
