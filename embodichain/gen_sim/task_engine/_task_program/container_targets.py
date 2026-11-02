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
import math

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
    footprint = MultiPoint(child_vertices[:, :2]).convex_hull
    footprint_radius = float(np.linalg.norm(child_vertices[:, :2], axis=1).max())
    if footprint.is_empty:
        raise ValueError(
            "Container floor cannot contain the object footprint and margin."
        )
    footprint_fits = False
    # Keep established layouts exact; retry only spatial rejection at 5 mm.
    for margin in (0.02, 0.005):
        safe = floor.buffer(-margin)
        if safe.is_empty:
            continue
        allowed = safe
        for x, y in footprint.exterior.coords[:-1]:
            allowed = allowed.intersection(
                translate(safe, xoff=-float(x), yoff=-float(y))
            )
            if allowed.is_empty:
                break
        if allowed.is_empty:
            continue
        footprint_fits = True
        for x, y, previous_radius in occupied:
            radius = footprint_radius + float(previous_radius) + margin
            if margin == 0.005:
                # Buffer polygons inscribe circles; expand so no arc loses clearance.
                radius = (radius + 1e-9) / math.cos(math.pi / 64)
            allowed = allowed.difference(
                Point(float(x), float(y)).buffer(radius, quad_segs=16)
            )
        if allowed.is_empty or allowed.area <= 1e-8:
            continue
        point = nearest_points(allowed, Point(arm_root[:2]))[0]
        candidates = (
            (point,) if margin == 0.02 else (point, allowed.representative_point())
        )
        for point in candidates:
            if margin == 0.005:
                placed = translate(footprint, xoff=point.x, yoff=point.y)
                if (
                    not safe.covers(placed)
                    or floor.boundary.distance(placed) + 1e-9 < margin
                ):
                    continue
            height = float(floor_height - child_vertices[:, 2].min() + 0.001)
            return float(point.x), float(point.y), height
    if footprint_fits:
        raise ValueError("Container floor has no remaining landing space.")
    raise ValueError("Container floor cannot contain the object footprint and margin.")


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
