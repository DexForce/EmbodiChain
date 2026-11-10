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

"""Parent-local planar support evidence and conservative whole-cell rasterization.

This strategy deliberately differs from the table detector: child contact
heights select a layer and unions retain holes and disconnected components.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, mapping, shape
from shapely.ops import unary_union

__all__: list[str] = []


def _read_support_polygon(value: object) -> Polygon | MultiPolygon:
    """Decode finite XY GeoJSON before accessing any scene assets."""
    if not isinstance(value, dict) or value.get("type") not in (
        "Polygon",
        "MultiPolygon",
    ):
        raise ValueError(
            "Support polygon must be a GeoJSON Polygon or MultiPolygon mapping."
        )
    try:
        polygon = shape(value)
    except (
        AttributeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
        shapely.errors.GEOSException,
    ) as exc:
        raise ValueError("Support polygon has invalid GeoJSON coordinates.") from exc
    if (
        not isinstance(polygon, (Polygon, MultiPolygon))
        or polygon.is_empty
        or polygon.has_z
        or not np.isfinite(shapely.get_coordinates(polygon)).all()
        or not polygon.is_valid
        or polygon.area <= 0
    ):
        raise ValueError(
            "Support polygon must contain a valid nonempty finite XY area."
        )
    return polygon


def _plane_region(
    vertices: np.ndarray, faces: np.ndarray, frame: np.ndarray, tolerance: float
) -> tuple[Polygon | MultiPolygon, float]:
    points = (vertices - frame[:3, 3]) @ frame[:3, :3]
    triangles = points[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    norm = np.linalg.norm(cross, axis=1)
    selected = (
        (cross[:, 2] >= 0.95 * norm)
        & (norm > 1e-14)
        & (np.abs(triangles[:, :, 2]).max(axis=1) <= tolerance)
    )
    polygons = [
        Polygon(t[:, :2]) for t in triangles[selected] if Polygon(t[:, :2]).area > 1e-13
    ]
    result = unary_union(polygons)
    if (
        not isinstance(result, (Polygon, MultiPolygon))
        or not result.is_valid
        or result.area <= 1e-10
    ):
        raise ValueError("Support plane has no valid mesh-supported polygon.")
    ceiling = max(0.0, float(triangles[selected, :, 2].max()))
    return result, ceiling


def _prepare_support(
    vertices: np.ndarray,
    faces: np.ndarray,
    child_points: list[np.ndarray],
    *,
    resolution: float,
    margin: float,
    override: dict | None = None,
) -> dict:
    """Select a near-planar layer using original child bottom heights.

    Frames are explicit parent_from_support rigid matrices. Automatic detection
    supports planes within 18 degrees of parent-local +Z. Ambiguous layers must
    be resolved with an evidence-checked override rather than a top/AABB fallback.
    """
    tolerance = 0.002
    if override is not None:
        frame = np.asarray(override["parent_from_support"], dtype=np.float64)
        polygon = _read_support_polygon(override["polygon"])
        evidence, ceiling = _plane_region(vertices, faces, frame, tolerance)
        if not evidence.covers(polygon):
            raise ValueError("Support override extends beyond mesh support evidence.")
    else:
        triangles = vertices[faces]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        norm = np.linalg.norm(cross, axis=1)
        upward = (cross[:, 2] >= 0.95 * norm) & (norm > 1e-12)
        heights = triangles.mean(axis=1)[:, 2]
        indices = np.flatnonzero(upward)
        if not len(indices):
            raise ValueError(
                "No upward parent support faces; provide support_overrides."
            )
        # Cluster candidate horizontal layers without filling gaps in projection.
        order = indices[np.argsort(heights[indices], kind="stable")]
        layers: list[list[int]] = []
        for index in order:
            if not layers or heights[index] - heights[layers[-1][0]] > tolerance:
                layers.append([])
            layers[-1].append(int(index))
        candidates = []
        for layer in layers:
            chosen = triangles[layer]
            area = norm[layer].sum() / 2
            if area < 1e-5:
                continue
            normal = cross[layer].sum(axis=0)
            normal /= np.linalg.norm(normal)
            origin = np.average(chosen.mean(axis=1), axis=0, weights=norm[layer])
            # Use parent +X projected into the plane for a stable support XY basis.
            axis_x = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
            axis_x /= np.linalg.norm(axis_x)
            rotation = np.column_stack((axis_x, np.cross(normal, axis_x), normal))
            frame = np.eye(4)
            frame[:3, :3] = rotation
            # Frame XY origin follows the parent's origin, avoiding a mesh centroid shift.
            frame[:3, 3] = normal * np.dot(origin, normal)
            try:
                polygon, ceiling = _plane_region(vertices, faces, frame, tolerance)
            except ValueError:
                continue
            distances = []
            supported = True
            for child in child_points:
                points = (child - frame[:3, 3]) @ rotation
                bottom = points[:, 2].min()
                # The original origin need not lie inside its geometric footprint.
                footprint = shapely.MultiPoint(points[:, :2]).convex_hull
                if not polygon.intersects(footprint):
                    supported = False
                    break
                distances.append(abs(float(bottom)))
            if not supported or not distances or max(distances) > 0.04:
                continue
            candidates.append(
                (float(np.mean(distances)), -polygon.area, frame, polygon, ceiling)
            )
        if not candidates:
            raise ValueError(
                "Cannot determine the parent support layer from child contact geometry; provide support_overrides."
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        best = candidates[0]
        for other in candidates[1:]:
            if (
                abs(best[0] - other[0]) < 0.0005
                and np.linalg.norm(best[2][:3, 3] - other[2][:3, 3]) > tolerance
            ):
                raise ValueError(
                    "Ambiguous parent support layers; provide support_overrides."
                )
        _, _, frame, polygon, ceiling = best
    usable = polygon.buffer(-margin, join_style=2) if margin else polygon
    if usable.is_empty:
        raise ValueError("Support region is empty after the boundary margin.")
    lower = np.floor(np.array(usable.bounds[:2]) / resolution) * resolution
    size = np.ceil((np.array(usable.bounds[2:]) - lower) / resolution).astype(int)
    if np.any(size <= 0) or int(size.prod()) > 4_000_000:
        raise ValueError(
            "Support raster requires 1..4,000,000 cells; adjust grid_resolution_m."
        )
    mask = np.empty((size[0], size[1]), dtype=bool)
    shapely.prepare(usable)
    # All four edges and the cell interior must have support, including holes.
    for x0 in range(0, size[0], 128):
        x = lower[0] + np.arange(x0, min(x0 + 128, size[0]))[:, None] * resolution
        y = lower[1] + np.arange(size[1])[None, :] * resolution
        cells = shapely.box(x, y, x + resolution, y + resolution)
        mask[x0 : x0 + len(x)] = shapely.covers(usable, cells)
    integral = np.pad((~mask).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return {
        "parent_from_support": frame.tolist(),
        "height_ceiling_m": ceiling,
        "polygon": mapping(polygon),
        "grid_origin": lower.tolist(),
        "resolution": resolution,
        "unsupported_integral": integral.tolist(),
    }
