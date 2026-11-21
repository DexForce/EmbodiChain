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

"""Planar floor selection and footprint measurement without simulation state."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union
import torch
import trimesh

__all__ = [
    "planar_floor_region",
    "select_container_landing",
    "select_table_landing",
    "planar_placement_metrics",
    "validate_planar_pose",
]


def planar_floor_region(mesh: trimesh.Trimesh) -> tuple[BaseGeometry, float] | None:
    """Find the dominant upward planar facet, preserving unsupported holes.

    Args:
        mesh: Scaled mesh in its local Z-up frame.

    Returns:
        Triangle-union footprint and maximum facet height, or None when no
        upward facet passes the established area and normal thresholds.
    """
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

    Args:
        mesh: Scaled reference mesh in its local Z-up frame.
        child_vertices: Finite payload points after the intended relative rotation.
        arm_root: Arm origin in the same frame, used only for candidate preference.
        occupied: Existing landing centers and footprint radii in this frame.

    Returns:
        Local XYZ landing position, or None for an unsupported floor.

    Raises:
        ValueError: A measured floor has no space for the payload and margins.
    """
    measured = planar_floor_region(mesh)
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


def validate_planar_pose(pose: torch.Tensor, batch_size: int) -> None:
    """Validate the rigid transform contract used by planar observations.

    Args:
        pose: Batched object-to-world matrices.
        batch_size: Expected number of independent environment rows.

    Raises:
        ValueError: Matrices are nonfinite, scaled, reflected or malformed.
    """
    if pose.shape != (batch_size, 4, 4) or not torch.isfinite(pose).all():
        raise ValueError("Placement pose does not match the observed batch.")
    rotation = pose[:, :3, :3]
    if (
        not torch.allclose(
            rotation.transpose(-1, -2) @ rotation,
            pose.new_tensor(np.eye(3)).expand_as(rotation),
            atol=1e-4,
        )
        or not torch.allclose(
            torch.linalg.det(rotation), pose.new_ones(batch_size), atol=1e-4
        )
        or not torch.allclose(
            pose[:, 3],
            pose.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(batch_size, -1),
            atol=1e-5,
        )
    ):
        raise ValueError("Placement requires rigid, non-reflected native transforms.")


def planar_placement_metrics(
    vertices: torch.Tensor,
    floors: Sequence[tuple[BaseGeometry, float]],
    object_pose: torch.Tensor,
    reference_pose: torch.Tensor,
    *,
    faces: torch.Tensor | None = None,
    reference_vertices: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Measure a payload footprint and vertical gap in a reference-local frame.

    Args:
        vertices: CPU scaled payload vertices, shape ``(B, V, 3)``.
        floors: One measured local floor polygon and height for each row.
        object_pose: CPU rigid payload-to-world transforms, shape ``(B, 4, 4)``.
        reference_pose: Corresponding rigid reference transforms.
        faces: Payload triangle indices for exact silhouette mode, or None for
            the legacy convex hull mode. Row order and arithmetic are preserved.
        reference_vertices: Scaled reference vertices, required with faces.

    Returns:
        Measured geometry in the existing receipt format. No task acceptance,
        contact support, release, or liquid transfer is inferred. Callers select
        support and penetration thresholds when evaluating these measurements.

    Raises:
        ValueError: Inputs are malformed or the measured floor is not near-level.
    """
    batch_size = object_pose.shape[0]
    validate_planar_pose(object_pose, batch_size)
    validate_planar_pose(reference_pose, batch_size)
    exact_silhouette = faces is not None
    if any(
        value.device.type != "cpu" for value in (vertices, object_pose, reference_pose)
    ):
        raise ValueError(
            "Planar footprint measurements require CPU geometry and poses."
        )
    if (
        vertices.ndim != 3
        or vertices.shape[0] != batch_size
        or vertices.shape[-1] != 3
        or not vertices.shape[1]
        or not torch.isfinite(vertices).all()
        or len(floors) != batch_size
    ):
        raise ValueError("Planar geometry must match its finite observed batch.")
    if exact_silhouette:
        if (
            faces.device.type != "cpu"
            or faces.ndim != 3
            or faces.shape[0] != batch_size
            or faces.shape[-1] != 3
            or not faces.shape[1]
            or faces.dtype not in (torch.int32, torch.int64)
            or (faces < 0).any()
            or (faces >= vertices.shape[1]).any()
        ):
            raise ValueError(
                "Planar triangles must index the corresponding vertex rows."
            )
        if (
            reference_vertices is None
            or reference_vertices.device.type != "cpu"
            or reference_vertices.ndim != 3
            or reference_vertices.shape[0] != batch_size
            or reference_vertices.shape[-1] != 3
            or not reference_vertices.shape[1]
            or not torch.isfinite(reference_vertices).all()
        ):
            raise ValueError("Exact silhouettes require finite reference vertices.")
    elif reference_vertices is not None:
        raise ValueError("Reference vertices require exact silhouette triangles.")
    relative = torch.linalg.solve(reference_pose, object_pose)
    local = (
        vertices.to(relative) @ relative[:, :3, :3].transpose(-1, -2)
        + relative[:, None, :3, 3]
    )
    footprints, gaps, areas = [], [], []
    outside, nonplanar, overlapping = [], [], []
    for row in range(batch_size):
        if reference_pose[row, 2, 2] < math.cos(math.radians(5.0)):
            raise ValueError("Placement reference lacks a qualified near-level floor.")
        polygon, height = floors[row]
        points = local[row].numpy()
        if exact_silhouette:
            # Project real triangles: the convex hull invents occupied material
            # across concavities and holes in a payload's silhouette.
            triangles = points[faces[row].numpy(), :2]
            footprint = unary_union(
                [
                    triangle_polygon
                    for triangle in triangles
                    if (triangle_polygon := Polygon(triangle)).area > 0
                ]
            )
        else:
            footprint = MultiPoint(points[:, :2]).convex_hull
        if footprint.is_empty or footprint.area <= 0:
            raise ValueError("Placement object has no finite area footprint.")
        inside = bool(polygon.covers(footprint))
        gap = float(points[:, 2].min() - height)
        footprints.append(inside)
        gaps.append(gap)
        areas.append(float(footprint.area))
        if exact_silhouette:
            outside.append(float(footprint.difference(polygon).area))
            bounds = reference_vertices[row].numpy()
            scale = max(float(np.ptp(bounds, axis=0).max()), 1.0)
            nonplanar.append(
                bool(bounds[:, 2].max() > height + 8 * np.finfo(np.float32).eps * scale)
            )
            overlapping.append(
                bool(
                    (
                        (points[:, :2].max(0) >= bounds[:, :2].min(0))
                        & (points[:, :2].min(0) <= bounds[:, :2].max(0))
                    ).all()
                )
            )
    metrics = {
        "scope": "measured_planar_landing_region",
        "object_pose": object_pose.tolist(),
        "reference_pose": reference_pose.tolist(),
        "full_footprint_inside": footprints,
        "support_gap_m": gaps,
        "footprint_area_m2": areas,
    }
    if exact_silhouette:
        metrics["footprint_kind"] = "native_triangle_projection/v1"
        metrics["outside_floor_area_m2"] = outside
        metrics["reference_has_raised_geometry"] = nonplanar
        metrics["reference_aabb_overlap"] = overlapping
    return metrics


def select_table_landing(
    table_mesh: trimesh.Trimesh,
    object_vertices: np.ndarray,
    position: Sequence[float],
    obstacle_bounds: Sequence[np.ndarray],
    *,
    table_top_z: float,
    object_bottom: float,
    object_top: float,
    axis_aligned_axis: int | None,
    placement_clearance: float,
    relation_clearance: float,
) -> tuple[float, float, float]:
    """Preserve a supported XY or move an occupied full footprint to free support.

    The table mesh is already in world coordinates. Object vertices are scaled
    local coordinates for explicit axis alignment, otherwise already rotated
    into the intended orientation. Obstacles arrive in the owning declaration's
    stable order as world-frame envelopes. This pure endpoint calculation keeps
    source triangle unions, Minkowski exclusions and nearest-point tie behavior;
    it proves neither approach-path clearance nor physical placement success.

    Args:
        table_mesh: World-frame table triangles.
        object_vertices: Payload vertices in the frame described above.
        position: Proposed world XYZ origin.
        obstacle_bounds: Ordered world-frame minimum/maximum corner arrays.
        table_top_z: Declared support height in metres.
        object_bottom: Payload lower vertical extent relative to its origin.
        object_top: Payload upper vertical extent relative to its origin.
        axis_aligned_axis: Explicit local long axis, or no axis alignment.
        placement_clearance: Release clearance in metres.
        relation_clearance: Horizontal obstacle clearance in metres.

    Returns:
        Supported world XYZ origin, preserving the proposed Z coordinate.
    """
    position = tuple(float(value) for value in position)
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError("Table landing position must be a finite XYZ vector.")
    vertices = np.asarray(object_vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Table landing requires nonempty (V, 3) object vertices.")
    points = np.asarray(table_mesh.vertices)
    measured = planar_floor_region(table_mesh)
    if (
        measured is None
        or abs(float(points[:, 2].max()) - table_top_z) > 1e-4
        or abs(measured[1] - table_top_z) > placement_clearance
    ):
        raise ValueError("Table landing requires a measured matching planar support.")
    floor = measured[0]
    vertices = np.asarray(object_vertices)
    if axis_aligned_axis is not None:
        if type(axis_aligned_axis) is not int or axis_aligned_axis not in (0, 1, 2):
            raise ValueError("Axis-aligned landing requires an explicit XYZ axis.")
        radius = float(
            np.linalg.norm(np.delete(vertices, axis_aligned_axis, axis=1), axis=1).max()
        )
        footprint = box(-radius, -radius, radius, radius)
    else:
        footprint = MultiPoint(vertices[:, :2]).convex_hull
    if footprint.is_empty or footprint.area <= 0 or not np.isfinite(vertices).all():
        raise ValueError("Table landing requires a finite full object footprint.")
    original = translate(footprint, xoff=position[0], yoff=position[1])
    if not floor.buffer(1e-9).covers(original):
        raise ValueError(
            "Original table landing footprint lacks full measured support."
        )
    release_top = table_top_z + object_top - object_bottom + placement_clearance
    excluded = []
    footprint_points = np.asarray(footprint.exterior.coords[:-1])
    for bounds in obstacle_bounds:
        if bounds[1, 2] < table_top_z or bounds[0, 2] > release_top:
            continue
        corners = np.asarray(box(*bounds[0, :2], *bounds[1, :2]).exterior.coords[:-1])
        forbidden = MultiPoint(
            (corners[:, None, :] - footprint_points[None, :, :]).reshape(-1, 2)
        ).convex_hull.buffer(relation_clearance + 1e-6, join_style=2)
        excluded.append(forbidden)
    occupied = unary_union(excluded)
    if occupied.is_empty or not occupied.covers(Point(position[:2])):
        return tuple(position)
    radius = float(np.linalg.norm(footprint_points, axis=1).max())
    available = floor.buffer(-radius - relation_clearance).difference(occupied)
    if available.is_empty or available.area <= 1e-8:
        raise ValueError("Table support has no remaining legal landing space.")
    point = nearest_points(available, Point(position[:2]))[0]
    if not floor.covers(translate(footprint, xoff=point.x, yoff=point.y)):
        raise ValueError("Selected table landing lacks full measured support.")
    return float(point.x), float(point.y), position[2]
