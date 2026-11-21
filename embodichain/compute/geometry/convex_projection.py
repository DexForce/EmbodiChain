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

"""Conservative CPU distances to complete convex planar projections."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull

__all__ = ["convex_projection_distance_to_origin"]


def _segment_distance(start: NDArray[np.float64], stop: NDArray[np.float64]) -> float:
    delta = stop - start
    denominator = float(delta @ delta)
    fraction = (
        float(np.clip(-(start @ delta) / denominator, 0, 1)) if denominator else 0.0
    )
    return float(np.linalg.norm(start + fraction * delta))


def convex_projection_distance_to_origin(points: ArrayLike) -> float:
    """Lower-bound the distance from the origin to a planar convex hull.

    Args:
        points: Nonempty finite real coordinates with shape ``(N, 2)``. Duplicate
            points and degenerate single-point/collinear projections are valid.

    Returns:
        Non-negative CPU float64 distance in the input units. The complete convex
        projection is checked, including edges and interior. A scale-dependent
        roundoff allowance is subtracted; near-collinear projections also subtract
        the maximum residual from their principal segment. Zero therefore does
        not establish separation. Input storage remains unchanged.

    Raises:
        ValueError: Coordinates are not finite real ``(N, 2)`` values.
        scipy.spatial.QhullError: A nondegenerate projection cannot be resolved.

    .. note::
        This CPU/SciPy kernel preserves the qualified source projection procedure.
        It performs deduplication and a convex hull, with ``O(N log N)`` typical
        cost and ``O(N)`` intermediate storage. It is not a mesh or continuous
        collision certificate; the consumer owns projection frames and margins.
    """
    if np.iscomplexobj(points):
        raise ValueError("Projection points must be real coordinates.")
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            "Projection points must be finite nonempty (N, 2) coordinates."
        )
    allowance = max(1e-12, float(np.linalg.norm(points, axis=1).max()) * 1e-12)
    if len(points) == 1:
        distance = float(np.linalg.norm(points[0]))
    elif len(points) == 2:
        distance = _segment_distance(*points)
    else:
        center = points.mean(0)
        _, singular, directions = np.linalg.svd(points - center, full_matrices=False)
        if singular[1] <= max(1e-14, singular[0] * 1e-12):
            direction = directions[0]
            along = (points - center) @ direction
            residual = points - center - along[:, None] * direction
            distance = _segment_distance(
                center + along.min() * direction, center + along.max() * direction
            ) - float(np.linalg.norm(residual, axis=1).max())
        else:
            hull = ConvexHull(points)
            if np.all(hull.equations[:, 2] <= allowance):
                return 0.0
            polygon = points[hull.vertices]
            distance = min(
                _segment_distance(first, second)
                for first, second in zip(
                    polygon, np.roll(polygon, -1, axis=0), strict=True
                )
            )
    return max(0.0, distance - allowance)
