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

"""Read final export geometry without creating simulation resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
import trimesh

from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    load_glb_mesh,
)
from embodichain.lab.sim.scene._support_cpu import (
    SupportSurfaceConfig,
    SupportSurfaceDetector,
)

_Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_MAX_TABLE_BOUNDARY_TOLERANCE_M = 1e-6
_RECTANGLE_AXIS_TOLERANCE_M = 1e-9


@dataclass(frozen=True)
class SceneGeometry:
    """Final unscaled entity poses, full AABBs and one table work area."""

    poses_by_uid: dict[str, np.ndarray]
    aabbs_by_uid: dict[str, np.ndarray]
    table: dict[str, Any]


def measure_scene_geometry(
    entries: list[dict[str, Any]], export_root: Path
) -> SceneGeometry:
    """Measure canonical rigid export entries in the arena-local Z-up frame.

    GLB node transforms are applied by the shared mesh loader. Exported poses
    already use Z-up, while body scales still refer to the original Y-up GLB
    axes. Scale is therefore applied before the single axis conversion.
    """
    poses_by_uid: dict[str, np.ndarray] = {}
    aabbs_by_uid: dict[str, np.ndarray] = {}
    table: dict[str, Any] | None = None
    for entry in entries:
        uid = entry["uid"]
        if uid in poses_by_uid:
            raise ValueError(f"Duplicate scene entity {uid!r}.")
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler(
            "XYZ", entry.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
        ).as_matrix()
        pose[:3, 3] = entry.get("init_pos", [0.0, 0.0, 0.0])
        scale = np.asarray(entry.get("body_scale", [1.0, 1.0, 1.0]), dtype=float)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError(f"Entity {uid!r} body_scale must contain positive values.")
        mesh = load_glb_mesh(export_root / entry["shape"]["fpath"])
        transform = pose.copy()
        transform[:3, :3] = pose[:3, :3] @ _Y_UP_TO_Z_UP @ np.diag(scale)
        if uid == "table":
            boundary_tolerance_xy = _table_boundary_tolerance_xy(
                mesh.vertices, transform
            )
        mesh.apply_transform(transform)
        if (
            len(mesh.vertices) < 3
            or len(mesh.faces) == 0
            or not np.isfinite(mesh.vertices).all()
        ):
            raise ValueError(f"Entity {uid!r} must contain finite triangle geometry.")
        poses_by_uid[uid] = pose
        aabbs_by_uid[uid] = mesh.bounds.copy()
        if uid == "table":
            table = _measure_table(entry, mesh, boundary_tolerance_xy)
    if table is None:
        raise ValueError("The scene must contain a table entity.")
    return SceneGeometry(poses_by_uid, aabbs_by_uid, table)


def _table_boundary_tolerance_xy(
    local_vertices: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Bound the GLB float32 rounding error in the XY support rectangle."""
    # World translation must not enlarge the accepted boundary error.
    local_error = (
        np.max(np.abs(local_vertices), axis=0, initial=0.0) * np.finfo(np.float32).eps
    )
    return np.minimum(
        np.abs(transform[:2, :3]) @ local_error, _MAX_TABLE_BOUNDARY_TOLERANCE_M
    )


def _measure_table(
    entry: dict[str, Any],
    mesh: trimesh.Trimesh,
    boundary_tolerance_xy: np.ndarray,
    *,
    allow_rotated_rectangle: bool = False,
) -> dict[str, Any]:
    """Validate one safe rectangle against actual support triangles."""
    config = SupportSurfaceConfig()
    saved_height = entry.get("support_surface_z")
    expected_height = None if saved_height is None else float(saved_height)
    detector = SupportSurfaceDetector(mesh.vertices, mesh.faces, config=config)
    rect_points = entry.get("support_optimization_rect_xy")
    rectangle = (
        _read_rectangle(rect_points, allow_rotated=True)
        if allow_rotated_rectangle and rect_points is not None
        else None
    )
    if rectangle is not None and not rectangle.equals(box(*rectangle.bounds)):
        # A rotated declaration need not have an axis-aligned rectangle among
        # the mesh's vertex coordinates. Keep the detector path for aligned input.
        measured_height, selected_triangles = detector._detect_support_triangles(
            top_z=expected_height
        )
    else:
        detected = detector.detect(top_z=expected_height)
        measured_height = detected.top_z
        selected_triangles = detected.vertices[detected.faces]
        if rectangle is None:
            rectangle = (
                detected.optimization_rectangle
                if rect_points is None
                else _read_rectangle(rect_points)
            )
    surface_z = measured_height if expected_height is None else expected_height
    if (
        float(np.ptp(selected_triangles[:, :, 2]))
        > config.height_level_tolerance_m + 1e-9
    ):
        raise ValueError("Table support must occupy one horizontal height level.")

    comparison_rectangle, sampling_rectangle = _table_rectangles(
        rectangle, boundary_tolerance_xy
    )

    projected_triangles = [Polygon(triangle[:, :2]) for triangle in selected_triangles]
    support = unary_union(
        [triangle for triangle in projected_triangles if triangle.area > 1e-12]
    )
    if isinstance(support, Polygon):
        components = [support]
    elif isinstance(support, MultiPolygon):
        components = list(support.geoms)
    else:
        components = []
    # The shared detector returns only an exterior contour. Keep holes and gaps
    # here so a rectangle cannot bridge unsupported space. Detached fragments
    # outside the chosen work area do not invalidate the main tabletop.
    if not any(
        component.is_valid
        and not component.interiors
        and component.covers(comparison_rectangle)
        for component in components
    ):
        raise ValueError(
            "Table safe rectangle must be covered by one connected support "
            "surface without holes."
        )
    minimum_x, minimum_y, maximum_x, maximum_y = sampling_rectangle.bounds
    return {
        "entity_id": "table",
        "center_xy": [(minimum_x + maximum_x) / 2.0, (minimum_y + maximum_y) / 2.0],
        "size_xy": [maximum_x - minimum_x, maximum_y - minimum_y],
        "surface_z": surface_z,
    }


def _table_rectangles(
    rectangle: Polygon, boundary_tolerance_xy: np.ndarray
) -> tuple[Polygon, Polygon]:
    """Return the footprint to validate and the arena-aligned sampling range."""
    minimum = np.asarray(rectangle.bounds[:2])
    maximum = np.asarray(rectangle.bounds[2:])
    # Inset for GLB round-trip error, never buffer the measured support mesh.
    if rectangle.equals(box(*minimum, *maximum)):
        inner_minimum = minimum + boundary_tolerance_xy
        inner_maximum = maximum - boundary_tolerance_xy
        comparison = (
            box(*inner_minimum, *inner_maximum)
            if np.all(inner_maximum > inner_minimum)
            else rectangle
        )
        return comparison, rectangle

    corners = np.asarray(rectangle.exterior.coords)[:4]
    edges = np.diff(corners[:3], axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    local_axes = edges / lengths[:, None]
    center = corners.mean(axis=0)
    half = lengths / 2.0
    inset = half - np.abs(local_axes) @ boundary_tolerance_xy
    comparison = rectangle
    if np.all(inset > 0):
        local_corners = (corners - center) @ local_axes.T
        comparison = Polygon(center + (local_corners * (inset / half)) @ local_axes)
        half = inset

    # Fit a centered arena rectangle inside both widths of the authored region.
    arena_half = (maximum - minimum) / 2.0
    ratio = np.min(half / (np.abs(local_axes) @ arena_half))
    # Leave room for round-off at a corner that touches the boundary.
    arena_half *= ratio * (1.0 - 1e-12)
    return comparison, box(*(center - arena_half), *(center + arena_half))


def _read_rectangle(value: object, *, allow_rotated: bool = False) -> Polygon:
    """Read four ordered corners of an axis-aligned or rotated rectangle."""
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Table safe rectangle must contain four XY corners.") from exc
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Table safe rectangle must contain four finite XY corners.")
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    rectangle = box(*minimum, *maximum)
    # A quarter-turn computed by the scene exporter can leave round-off at a
    # corner. Snap only this small error and validate the resulting corner order.
    snapped = np.where(points - minimum <= maximum - points, minimum, maximum)
    snapped_polygon = Polygon(snapped)
    if (
        np.all(np.abs(points - snapped) <= _RECTANGLE_AXIS_TOLERANCE_M)
        and snapped_polygon.is_valid
        and snapped_polygon.equals(rectangle)
    ):
        return rectangle
    if allow_rotated:
        outline = Polygon(points)
        if outline.is_valid and outline.area > 0.0:
            oriented = outline.minimum_rotated_rectangle
            if (
                isinstance(oriented, Polygon)
                and outline.hausdorff_distance(oriented) <= _RECTANGLE_AXIS_TOLERANCE_M
            ):
                return oriented
        raise ValueError("Table safe rectangle must be a non-empty rectangle.")
    raise ValueError("Table safe rectangle must be axis-aligned and non-empty.")
