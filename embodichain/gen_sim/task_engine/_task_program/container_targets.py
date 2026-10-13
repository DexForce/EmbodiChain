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

"""Select a conservative landing point on a measured planar container floor."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union
import trimesh

__all__: list[str] = []


def select_container_landing(
    mesh: trimesh.Trimesh,
    child_vertices: np.ndarray,
    arm_root: np.ndarray,
    *,
    occupied: Sequence[tuple[float, float, float]] = (),
) -> tuple[float, float, float] | None:
    """Prefer the arm-side floor region after eroding by the object footprint.

    All inputs use the container-local Z-up frame. Child vertices include its
    intended relative rotation but not translation. A dominant near-horizontal
    facet is required; curved or unmeasured floors retain the caller's policy.
    This geometric preference does not replace downstream IK or execution gates.
    """
    measured = _floor_region(mesh)
    if measured is None:
        return None
    floor, floor_height = measured
    margin = 0.02
    safe = floor.buffer(-margin)
    footprint = MultiPoint(child_vertices[:, :2]).convex_hull
    footprint_radius = float(np.linalg.norm(child_vertices[:, :2], axis=1).max())
    if safe.is_empty or footprint.is_empty:
        raise ValueError(
            "Container floor cannot contain the object footprint and margin."
        )
    # Erode the measured floor by the actual projected footprint.  A single
    # origin-centred circle overestimates a rectangular object's occupied area,
    # which rejects valid placements after an upright reorientation.
    allowed = safe
    for x, y in footprint.exterior.coords[:-1]:
        allowed = allowed.intersection(translate(safe, xoff=-float(x), yoff=-float(y)))
        if allowed.is_empty:
            raise ValueError(
                "Container floor cannot contain the object footprint and margin."
            )
    for x, y, previous_radius in occupied:
        allowed = allowed.difference(
            Point(float(x), float(y)).buffer(
                footprint_radius + float(previous_radius) + 0.02
            )
        )
    if allowed.is_empty or allowed.area <= 1e-8:
        raise ValueError("Container floor has no remaining landing space.")
    point = nearest_points(allowed, Point(arm_root[:2]))[0]
    height = float(floor_height - child_vertices[:, 2].min() + 0.001)
    return float(point.x), float(point.y), height


def _floor_region(mesh: trimesh.Trimesh) -> tuple[BaseGeometry, float] | None:
    """Share measured planar-floor geometry without changing landing policy."""
    if not len(mesh.facets):
        return None
    eligible = np.flatnonzero(mesh.facets_normal[:, 2] > 0.95)
    if not len(eligible):
        return None
    facet = eligible[np.argmax(mesh.facets_area[eligible])]
    if mesh.facets_area[facet] < 0.25 * np.prod(mesh.extents[:2]):
        return None
    triangles = mesh.triangles[mesh.facets[facet]]
    # Union the actual triangles, not their convex hull: holes remain unsupported.
    floor = unary_union([Polygon(triangle[:, :2]) for triangle in triangles])
    return floor, float(triangles[:, :, 2].max())
