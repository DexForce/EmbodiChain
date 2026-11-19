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

"""Batched planar polygon distances without simulator dependencies."""

from __future__ import annotations

import torch

__all__ = ["signed_distance_to_polygon"]


def signed_distance_to_polygon(
    points: torch.Tensor,
    polygon: torch.Tensor,
    *,
    max_points_per_chunk: int = 16384,
) -> torch.Tensor:
    """Measure signed Euclidean distance to a closed planar polygon.

    Segment distances and an even-odd interior test support concave polygons.
    Interior distances are negative. Inputs must contain finite coordinates;
    the first polygon vertex must not be repeated to close the outline.
    Point-edge intermediates are bounded by the chunk size times the vertex
    count. The output owns its storage and retains point-distance gradients
    away from polygon boundaries and nearest-edge ties.

    Args:
        points: Floating coordinates shaped ``[..., 2]``.
        polygon: Floating vertices shaped ``[vertices, 2]``, on the same device
            and with the same dtype as ``points``.
        max_points_per_chunk: Positive upper bound on points per edge reduction.

    Returns:
        Signed distances shaped ``points.shape[:-1]``.

    Raises:
        ValueError: Shapes, dtypes/devices or chunk size do not match the contract.
    """
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError(
            "Polygon query points must end in a coordinate axis of size 2."
        )
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("A polygon requires at least three 2D vertices.")
    if (
        not points.is_floating_point()
        or points.dtype != polygon.dtype
        or points.device != polygon.device
    ):
        raise ValueError(
            "Polygon points and vertices require matching floating dtype and device."
        )
    if type(max_points_per_chunk) is not int or max_points_per_chunk < 1:
        raise ValueError("max_points_per_chunk must be a positive integer.")
    if points.numel() == 0:
        return points.new_empty(points.shape[:-1])
    endpoint = polygon.roll(-1, 0)
    edges = endpoint - polygon
    length_squared = edges.square().sum(dim=-1).clamp_min(1e-12)
    distances = []
    for chunk in points.reshape(-1, 2).split(max_points_per_chunk):
        rx = chunk[:, 0, None] - polygon[None, :, 0]
        ry = chunk[:, 1, None] - polygon[None, :, 1]
        projection = (
            (rx * edges[None, :, 0] + ry * edges[None, :, 1]) / length_squared[None]
        ).clamp(0, 1)
        distance = (
            (
                (rx - projection * edges[None, :, 0]).square()
                + (ry - projection * edges[None, :, 1]).square()
            )
            .amin(dim=-1)
            .sqrt()
        )
        crossings = (
            (
                (polygon[None, :, 1] > chunk[:, 1, None])
                != (endpoint[None, :, 1] > chunk[:, 1, None])
            )
            & (
                chunk[:, 0, None]
                < edges[None, :, 0]
                * (chunk[:, 1, None] - polygon[None, :, 1])
                / (edges[None, :, 1] + 1e-12)
                + polygon[None, :, 0]
            )
        ).sum(dim=-1)
        distances.append(torch.where(crossings % 2 == 1, -distance, distance))
    return torch.cat(distances).reshape(points.shape[:-1])
