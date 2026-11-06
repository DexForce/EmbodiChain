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

"""Ordered point-set proximity measurements for explicit contact candidates.

Callers own FK, frame conversion, triangle qualification and candidate ranking.
These measurements do not establish physical contact or mesh collision freedom.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import torch

__all__ = ["sample_contact_vertices", "pad_proximity_metrics"]


def sample_contact_vertices(
    vertices: torch.Tensor, *, maximum_count: int
) -> torch.Tensor:
    """Select deterministic source vertices without surface resampling.

    Args:
        vertices: Complete finite XYZ tensor in its source frame.
        maximum_count: Explicit positive sample cap. Uniform rounded source
            indices preserve the original order, dtype and device.

    Returns:
        Selected rows, or the unchanged tensor when it already fits the cap.
    """
    if type(maximum_count) is not int or maximum_count <= 0:
        raise ValueError("Contact sample count must be a positive integer.")
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not torch.isfinite(vertices).all()
    ):
        raise ValueError("Contact scoring requires complete finite source point sets.")
    if vertices.shape[0] > maximum_count:
        ids = (
            torch.linspace(
                0, vertices.shape[0] - 1, maximum_count, device=vertices.device
            )
            .round()
            .long()
        )
        vertices = vertices[ids]
    return vertices


def pad_proximity_metrics(
    pad_vertices: Sequence[torch.Tensor],
    target_vertices: torch.Tensor,
    parent_vertices: torch.Tensor,
    *,
    parent_distance_m: float,
) -> tuple[float, float]:
    """Measure explicit pad point sets against target and parent point sets.

    Args:
        pad_vertices: Ordered nonempty pad vertices in the common query frame.
        target_vertices: Sampled target vertices in that same frame.
        parent_vertices: Sampled parent vertices in that same frame.
        parent_distance_m: Explicit nonnegative parent-near threshold in metres.

    Returns:
        Fraction of pad vertices near the parent, and the largest of the
        individual pads' minimum target distances. No scores or ranking weights
        are selected here; input point sets are not resampled.
    """
    if (
        not pad_vertices
        or isinstance(parent_distance_m, bool)
        or not math.isfinite(parent_distance_m)
        or parent_distance_m < 0
    ):
        raise ValueError("Pad metrics require pads and a finite nonnegative threshold.")
    for points in (*pad_vertices, target_vertices, parent_vertices):
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or not torch.isfinite(points).all()
        ):
            raise ValueError("Pad metrics require finite nonempty XYZ point sets.")
    distances, near = [], []
    for world in pad_vertices:
        distances.append(torch.cdist(world, target_vertices).amin())
        near.append(torch.cdist(world, parent_vertices).amin(-1))
    return (
        float(torch.cat(near).le(parent_distance_m).float().mean()),
        float(torch.stack(distances).amax()),
    )
