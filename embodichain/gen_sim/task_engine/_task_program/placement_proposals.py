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

"""Source-mesh placement proposals, never native support certification."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union
import trimesh

__all__ = ["generate_on_targets"]


def _projection(mesh: trimesh.Trimesh) -> BaseGeometry:
    triangles = (Polygon(points[:, :2]) for points in mesh.triangles)
    return unary_union([polygon for polygon in triangles if polygon.area > 0])


def generate_on_targets(
    object_mesh_zup: trimesh.Trimesh,
    reference_mesh_world: trimesh.Trimesh,
    rotations: Iterable[np.ndarray],
    preferred_xy: Sequence[float] | Sequence[Sequence[float]],
    clearance_m: float,
    contact_band_m: float,
) -> list[dict[str, Any]]:
    """Propose full-footprint placements with source contact-patch COM coverage.

    Meshes already own their scale and Z-up coordinates. Rotations are object-
    local to world 3x3 matrices. The convex-hull uniform COM is only a proposal
    prior, including for non-watertight sources. Native bearing is unverified.
    The finite point choices are conservative, not a complete feasible set.
    Preferred XY values are one point or a nonempty batch of world-frame points.
    """
    preferred = np.asarray(preferred_xy, dtype=float)
    if preferred.shape == (2,):
        preferred = preferred[None]
    if (
        preferred.ndim != 2
        or preferred.shape[1] != 2
        or not len(preferred)
        or not np.isfinite(preferred).all()
    ):
        raise ValueError("preferred_xy must contain finite XY points.")
    if any(not np.isfinite(v) or v < 0 for v in (clearance_m, contact_band_m)):
        raise ValueError("Placement distances must be finite and non-negative.")
    for mesh in (object_mesh_zup, reference_mesh_world):
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Placement requires finite triangular source geometry.")
    silhouette = _projection(reference_mesh_world)
    top = float(reference_mesh_world.vertices[:, 2].max())
    half_band = contact_band_m / 2
    upper = reference_mesh_world.slice_plane(
        [0, 0, top - half_band], [0, 0, 1], cap=False
    )
    upper_region = _projection(upper)
    center_mass = object_mesh_zup.convex_hull.center_mass
    if silhouette.is_empty or not np.isfinite(center_mass).all():
        return []
    proposals = []
    for index, value in enumerate(rotations):
        rotation = np.asarray(value, dtype=float)
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError("Placement rotations must be finite proper rotations.")
        rotated = object_mesh_zup.copy()
        rotated.vertices = np.asarray(object_mesh_zup.vertices) @ rotation.T
        bottom = float(rotated.vertices[:, 2].min())
        footprint = MultiPoint(rotated.vertices[:, :2]).convex_hull
        if footprint.geom_type != "Polygon":
            continue
        allowed = silhouette
        for x, y in footprint.exterior.coords[:-1]:
            allowed = allowed.intersection(translate(silhouette, xoff=-x, yoff=-y))
            if allowed.is_empty:
                break
        if allowed.is_empty:
            continue
        lower = rotated.slice_plane([0, 0, bottom + half_band], [0, 0, -1], cap=False)
        lower_region = _projection(lower)
        choices = (
            *(nearest_points(allowed, Point(xy))[0] for xy in preferred),
            allowed.representative_point(),
            allowed.centroid,
        )
        seen = set()
        for point in choices:
            xy = (float(point.x), float(point.y))
            if xy in seen:
                continue
            seen.add(xy)
            placed = translate(footprint, xoff=xy[0], yoff=xy[1])
            if not silhouette.covers(placed):
                continue
            # Split the band so paired surface heights differ by at most its sum.
            patch = upper_region.intersection(
                translate(lower_region, xoff=xy[0], yoff=xy[1])
            )
            com = rotation @ center_mass
            if patch.is_empty or not patch.convex_hull.covers(Point(com[:2] + xy)):
                continue
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = (*xy, top - bottom + clearance_m)
            proposals.append(
                {
                    "pose": pose,
                    "rotation_index": index,
                    "footprint": list(placed.exterior.coords),
                    "full_footprint_inside": True,
                    "potential_support_hull": patch.convex_hull.wkt,
                    "contact_band_m": float(contact_band_m),
                    "reference_top_z": top,
                    "object_bottom_z": bottom,
                    "center_mass_source": "uniform_convex_hull_not_native",
                    "geometry_scope": "source_mesh_proposal_after_clearance_settles",
                    "native_bearing_verified": False,
                }
            )
    return proposals
