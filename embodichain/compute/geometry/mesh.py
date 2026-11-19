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

import math

import torch

from .aabb import _float_tensor

__all__ = ["triangle_geometry", "box_face_axes"]


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


def box_face_axes(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    *,
    cone_angle: float = math.pi / 18,
    max_candidates: int = 256,
) -> torch.Tensor:
    """Estimate the three orthogonal face directions of a box-like mesh.

    The first axis is the triangle normal with the most surface area facing
    along it in either sense; the second is chosen the same way among normals
    perpendicular to the first. Each is refined to the area-weighted mean of
    its supporting normals. Small features such as chamfers hold little area
    and do not move the result. The estimate does not use the mesh frame, so
    a box modelled at an angle to its own axes is handled.

    Args:
        vertices: Mesh vertices, shaped ``(N, 3)``, float32 or float64.
        triangles: Vertex indices, shaped ``(M, 3)``.
        cone_angle: Half-angle in radians within which a normal supports, or
            is perpendicular to, an axis.
        max_candidates: Number of largest triangles tried as axis directions.

    Returns:
        Unit axes as rows, shaped ``(3, 3)``, right-handed and orthogonal, in
        the frame, dtype and device of ``vertices``.

    Raises:
        ValueError: The mesh has no valid triangle, or no face perpendicular
            to its dominant face direction.
    """
    _float_tensor(vertices, "vertices")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3)")
    if not 0.0 < cone_angle < math.pi / 4:
        raise ValueError("cone_angle must be in (0, pi/4)")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    normals, areas, valid = triangle_geometry(
        vertices[triangles.to(device=vertices.device, dtype=torch.long)]
    )
    normals, areas = normals[valid], areas[valid]
    if normals.shape[0] == 0:
        raise ValueError("Mesh has no valid triangle")
    candidates = normals[torch.topk(areas, min(max_candidates, areas.shape[0])).indices]
    supports = math.cos(cone_angle)

    def strongest(allowed: torch.Tensor) -> torch.Tensor:
        directions = candidates[allowed]
        if directions.shape[0] == 0:
            raise ValueError(
                "Mesh has no face perpendicular to its dominant face direction"
            )
        # Normals facing either way along a direction belong to the same axis.
        alignment = directions @ normals.T
        support = ((alignment.abs() >= supports) * areas).sum(dim=1)
        signed = alignment[torch.argmax(support)]
        members = signed.abs() >= supports
        mean = (
            normals[members] * (torch.sign(signed[members]) * areas[members])[:, None]
        ).sum(dim=0)
        return mean / torch.linalg.vector_norm(mean)

    first = strongest(torch.ones_like(candidates[:, 0], dtype=torch.bool))
    second = strongest((candidates @ first).abs() <= math.sin(cone_angle))
    second = second - (second @ first) * first
    second = second / torch.linalg.vector_norm(second)
    return torch.stack((first, second, torch.linalg.cross(first, second)))
