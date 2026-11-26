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

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation
from shapely import affinity
from shapely.geometry import box

if TYPE_CHECKING:
    import trimesh

    from embodichain.gen_sim.scene_engine.pipeline.utils.table_support_surface import (
        TableSupportRegion,
    )

__all__ = ["DegradedTableFit", "fit_degraded_table", "tabletop_level_rotation"]


@dataclass(frozen=True)
class DegradedTableFit:
    """Grounded replacement and the vertical shift to share with its assets."""

    mesh_z_up: trimesh.Trimesh
    support_region: TableSupportRegion
    scale_xyz: tuple[float, float, float]
    ground_shift_z: float
    aligned_bottom_z: float


def tabletop_level_rotation(
    mesh_z_up: trimesh.Trimesh, *, complete: bool
) -> np.ndarray:
    """Level a broad table plane, using lower structure to orient a full table.

    A degraded tabletop keeps the sign closest to the source up axis. On a
    complete table, dominant area-distance and depth evidence above the broad
    plane indicate that the table's legs are above it and require a flip.
    """
    triangles = np.asarray(mesh_z_up.triangles, dtype=float)
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("Table leveling needs finite triangle geometry.")
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas_twice = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(areas_twice) & (areas_twice > 0.0)
    if not np.any(valid):
        raise ValueError("Table leveling needs nonzero-area triangles.")
    triangles = triangles[valid]
    weights = areas_twice[valid]
    normals = cross[valid] / weights[:, None]
    centers = triangles.mean(axis=1)
    points = triangles.reshape(-1, 3)
    length = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not np.isfinite(length):
        raise ValueError("Table leveling exceeded finite coordinate limits.")
    up = np.array([0.0, 0.0, 1.0])
    normal, plane_height = _broad_table_plane(
        centers, normals, weights, length, complete=complete
    )
    level_rotation = Rotation.align_vectors([up], [normal])[0].as_matrix()
    if complete:
        band = 0.03 * length
        signed_distance = centers @ normal - plane_height
        above_moment = weights @ np.maximum(signed_distance - band, 0.0)
        below_moment = weights @ np.maximum(-signed_distance - band, 0.0)
        vertex_heights = points @ normal - plane_height
        above_depth = max(float(vertex_heights.max()) - band, 0.0)
        below_depth = max(float(-vertex_heights.min()) - band, 0.0)
        if (
            above_depth >= 2 * band
            and above_depth > 2 * below_depth
            and above_moment > 3 * below_moment
        ):
            # Flip in the leveled frame so the two orientation steps are explicit.
            return np.diag([1.0, -1.0, -1.0]) @ level_rotation
    return level_rotation


def _broad_table_plane(
    centers: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray,
    length: float,
    *,
    complete: bool,
) -> tuple[np.ndarray, float]:
    """Choose the broadest narrow slab among area-weighted face directions."""
    unsigned = normals * np.where(normals[:, 2] < 0.0, -1.0, 1.0)[:, None]
    # The reference plane came from an up-facing surface in the source scene.
    candidates = unsigned if complete else unsigned[unsigned[:, 2] >= 0.5]
    candidate_weights = weights if complete else weights[unsigned[:, 2] >= 0.5]
    if not len(candidates):
        raise ValueError("The degraded tabletop has no near-horizontal plane.")
    directions, indices = np.unique(
        np.round(candidates, 1), axis=0, return_inverse=True
    )
    areas = np.bincount(indices, weights=candidate_weights)
    seeds = np.argsort(-areas, kind="stable")[:8]
    minimum_dot = np.cos(np.deg2rad(12.0))
    seen = []
    best_score = -1.0
    best = None
    for seed in directions[seeds]:
        seed /= np.linalg.norm(seed)
        if any(abs(seed @ previous) >= minimum_dot for previous in seen):
            continue
        seen.append(seed)
        dots = normals @ seed
        same_direction = np.abs(dots) >= minimum_dot
        signed_normals = normals[same_direction] * np.sign(dots[same_direction, None])
        surface_weights = weights[same_direction]
        direction = surface_weights @ signed_normals
        direction /= np.linalg.norm(direction)
        if direction[2] < 0.0:
            direction = -direction
        heights = centers[same_direction] @ direction
        ascending = np.argsort(heights)
        heights, surface_weights = heights[ascending], surface_weights[ascending]
        # Sliding intervals retain a sloped/noisy mesh's tabletop without
        # combining distinct levels, such as the furniture feet and tabletop.
        stop = np.searchsorted(heights, heights + 0.02 * length, side="right")
        cumulative = np.concatenate(([0.0], np.cumsum(surface_weights)))
        window_areas = cumulative[stop] - cumulative[:-1]
        start = int(np.argmax(window_areas))
        score = float(window_areas[start] * (0.5 + 0.5 * direction[2]))
        if score > best_score:
            primary = slice(start, stop[start])
            best = (
                direction,
                float(np.average(heights[primary], weights=surface_weights[primary])),
            )
            best_score = score
    if best is None:
        raise ValueError("No broad table plane could be estimated.")
    return best


def fit_degraded_table(
    *,
    replacement_mesh_z_up: trimesh.Trimesh,
    replacement_support: TableSupportRegion,
    reference_bounds_xy: np.ndarray,
    reference_top_z: float,
    grounded_tabletop_height_m: float,
) -> DegradedTableFit:
    """Match the old tabletop footprint, then move the complete table to ground.

    Both support surfaces are already level in Z-up. Reference bounds contain
    XY minima and maxima as two rows. Scale both horizontal axes to fit those
    bounds using the complete table's safe rectangle. Scale the vertical axis
    independently to the requested height above ground. Align the new tabletop
    to the old tabletop before measuring the signed shift to its lowest vertex.
    Callers apply that shift to every object supported by the table.
    """
    reference_bounds = _finite_rectangle_bounds(
        np.asarray(reference_bounds_xy, dtype=float)
    )
    replacement_bounds = _finite_rectangle_bounds(
        np.asarray(replacement_support.optimization_rectangle.bounds).reshape(-1, 2)
    )
    mesh_bounds = replacement_mesh_z_up.bounds
    if mesh_bounds is None or not np.isfinite(mesh_bounds).all():
        raise ValueError("The replacement table must have finite mesh bounds.")
    target_height = float(grounded_tabletop_height_m)
    if not np.isfinite(
        [reference_top_z, replacement_support.top_z, target_height]
    ).all():
        raise ValueError("Tabletop heights must be finite.")
    if target_height <= 0 or replacement_support.top_z <= mesh_bounds[0, 2]:
        raise ValueError(
            "The grounded and replacement tabletop heights must be positive."
        )

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            reference_span = reference_bounds[1] - reference_bounds[0]
            replacement_span = replacement_bounds[1] - replacement_bounds[0]
            # Keep a small coverage margin when GLB stores vertices as float32.
            scale_xy = (reference_span / replacement_span) * (
                1 + 4 * float(np.finfo(np.float32).eps)
            )
            scale_z = target_height / (replacement_support.top_z - mesh_bounds[0, 2])
            scale = np.array([*scale_xy, scale_z])
            reference_center = reference_bounds[0] + reference_span / 2
            replacement_center = replacement_bounds[0] + replacement_span / 2
            shift_xy = reference_center - scale_xy * replacement_center
            ground_shift_z = target_height - float(reference_top_z)
            translation = np.array([*shift_xy, -scale_z * mesh_bounds[0, 2]])
            fitted_bounds = mesh_bounds * scale + translation
    except FloatingPointError as exc:
        raise ValueError(
            "Tabletop alignment exceeded finite coordinate limits."
        ) from exc
    if (
        np.any(scale <= 0)
        or not np.isfinite(ground_shift_z)
        or not np.isfinite(fitted_bounds).all()
    ):
        raise ValueError("Tabletop alignment yielded an invalid scale or height.")

    scale_x, scale_y = scale_xy.tolist()
    planar_transform = [scale_x, 0.0, 0.0, scale_y, *shift_xy.tolist()]
    support_polygon = affinity.affine_transform(
        replacement_support.support_polygon, planar_transform
    )
    requested_footprint = box(*reference_bounds.ravel())
    # An exact boundary fit can differ by a few ULPs after affine conversion.
    tolerance = 16 * np.finfo(float).eps * max(1.0, np.abs(reference_bounds).max())
    if not support_polygon.is_valid or not (
        support_polygon.covers(requested_footprint)
        or support_polygon.buffer(tolerance).covers(requested_footprint)
    ):
        raise ValueError(
            "The replacement tabletop cannot cover the old reference bounds."
        )

    transform = np.eye(4)
    transform[:3, :3] = np.diag(scale)
    transform[:3, 3] = translation
    fitted_mesh = replacement_mesh_z_up.copy()
    fitted_mesh.apply_transform(transform)
    fitted_support = replace(
        replacement_support,
        top_z=target_height,
        vertices=replacement_support.vertices * scale + translation,
        support_polygon=support_polygon,
        optimization_rectangle=affinity.affine_transform(
            replacement_support.optimization_rectangle, planar_transform
        ),
    )
    return DegradedTableFit(
        mesh_z_up=fitted_mesh,
        support_region=fitted_support,
        scale_xyz=(scale_x, scale_y, float(scale_z)),
        ground_shift_z=ground_shift_z,
        aligned_bottom_z=-ground_shift_z,
    )


def _finite_rectangle_bounds(bounds: np.ndarray) -> np.ndarray:
    """Validate min/max rows for an axis-aligned footprint."""
    if (
        bounds.shape != (2, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[1] <= bounds[0])
    ):
        raise ValueError("Tabletop footprints must have finite, positive XY spans.")
    return bounds
