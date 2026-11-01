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

"""Bounded solid-angle membership for consistently oriented closed meshes."""

from __future__ import annotations

import math
import torch

__all__ = ["points_in_closed_mesh"]


@torch.no_grad()
def points_in_closed_mesh(
    points: torch.Tensor,
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    *,
    point_chunk_size: int = 32,
    triangle_chunk_size: int = 2048,
) -> torch.Tensor:
    """Classify points using the oriented solid angle of a closed triangle mesh.

    Args:
        points: Finite query positions, shape ``(P,3)``.
        vertices: Finite mesh positions, shape ``(V,3)``, same floating dtype/device.
        triangles: Valid triangle indices, shape ``(T,3)``, int64 on that device.
        point_chunk_size: Maximum query points materialized per tile.
        triangle_chunk_size: Maximum triangles materialized per tile.
    Returns:
        Boolean inside mask, shape ``(P,)``. Callers must establish watertightness,
        winding consistency and valid finite indices before using this numerical
        kernel. Surface points require a separate collision/tolerance query.
        Memory is bounded by point and triangle tiles, not P times T.
    """
    if (
        points.ndim != 2
        or points.shape[-1] != 3
        or vertices.ndim != 2
        or vertices.shape[-1] != 3
        or triangles.ndim != 2
        or triangles.shape[-1] != 3
    ):
        raise ValueError(
            "points/vertices/triangles require rank-two XYZ/triangle layouts"
        )
    if (
        not points.is_floating_point()
        or points.dtype != vertices.dtype
        or points.device != vertices.device
        or triangles.dtype != torch.int64
        or triangles.device != vertices.device
    ):
        raise ValueError(
            "mesh/query tensors must share floating dtype/device with int64 indices"
        )
    if (
        type(point_chunk_size) is not int
        or type(triangle_chunk_size) is not int
        or min(point_chunk_size, triangle_chunk_size) < 1
    ):
        raise ValueError("tile sizes must be positive integers")
    outputs = []
    for start in range(0, len(points), point_chunk_size):
        query = points[start : start + point_chunk_size]
        total = points.new_zeros(len(query))
        for face_start in range(0, len(triangles), triangle_chunk_size):
            corners = vertices[triangles[face_start : face_start + triangle_chunk_size]]
            rays = corners[None] - query[:, None, None]
            a, b, c = rays.unbind(dim=2)
            na, nb, nc = (torch.linalg.vector_norm(v, dim=-1) for v in (a, b, c))
            numerator = (a * torch.linalg.cross(b, c, dim=-1)).sum(dim=-1)
            denominator = (
                na * nb * nc
                + (a * b).sum(dim=-1) * nc
                + (b * c).sum(dim=-1) * na
                + (c * a).sum(dim=-1) * nb
            )
            total = total + (2 * torch.atan2(numerator, denominator)).sum(dim=-1)
        outputs.append(total.abs() > 2 * math.pi)
    return (
        torch.cat(outputs)
        if outputs
        else torch.empty((0,), dtype=torch.bool, device=points.device)
    )
