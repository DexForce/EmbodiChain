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

"""CPU mesh evidence for frozen placement inputs, independent of any planner."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from shapely import intersects_xy, set_precision
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ._placement_geometry import PAN_BASIN_LOCAL_Z_MAX

__all__: list[str] = []

_SAMPLE_SPACING_M = 0.005
_BARYCENTRIC_EPS = 1e-9
# One picometre resolves overlay roundoff without adding a physical margin.
_PROJECTION_GRID_M = 1e-12


def _projection(triangles: np.ndarray):
    faces = [Polygon(face[:, :2]) for face in triangles]
    return unary_union([face for face in faces if face.area > 0])


def _ray_heights(
    samples: np.ndarray, triangles: np.ndarray, *, upper: bool
) -> np.ndarray:
    """Reduce vertical triangle intersections in bounded point chunks."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    eb, ec = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    det = eb[:, 0] * ec[:, 1] - eb[:, 1] * ec[:, 0]
    active = np.abs(det) > 1e-15
    denominator = np.where(active, det, 1.0)
    outputs = []
    for start in range(0, len(samples), 256):
        delta = samples[start : start + 256, None] - a[None, :, :2]
        u = (delta[..., 0] * ec[:, 1] - delta[..., 1] * ec[:, 0]) / denominator
        v = (eb[:, 0] * delta[..., 1] - eb[:, 1] * delta[..., 0]) / denominator
        inside = (
            active
            & (u >= -_BARYCENTRIC_EPS)
            & (v >= -_BARYCENTRIC_EPS)
            & (u + v <= 1 + _BARYCENTRIC_EPS)
        )
        heights = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
        reduction = np.max if upper else np.min
        outputs.append(
            reduction(np.where(inside, heights, -np.inf if upper else np.inf), axis=1)
        )
    return np.concatenate(outputs)


def _placement_input_geometry(
    placing_vertices: torch.Tensor,
    placing_triangles: torch.Tensor,
    support_vertices: torch.Tensor,
    support_triangles: torch.Tensor,
    placing_targets: torch.Tensor,
    support_targets: torch.Tensor,
) -> dict[str, object]:
    """Audit complete XY containment and sampled vertical gaps; never select rows.

    Mesh vertices must already include the actual asset scale. Topology and
    frozen target transforms are copied to CPU once. The projected triangle
    unions retain holes and overhangs; neither object-origin nor AABB inclusion
    substitutes for footprint coverage. Heights are finite-sample evidence only.
    """
    meshes = {}
    arrays = []
    for name, vertices, triangles in (
        ("placing", placing_vertices, placing_triangles),
        ("support", support_vertices, support_triangles),
    ):
        v = np.ascontiguousarray(vertices.detach().cpu().numpy(), dtype="<f8")
        f = np.ascontiguousarray(triangles.detach().cpu().numpy(), dtype="<i8")
        if not np.isfinite(v).all() or len(f) == 0:
            raise ValueError(
                "Placement input geometry requires finite, nonempty meshes."
            )
        arrays.append((v, f))
        meshes[name] = {
            "vertices": len(v),
            "triangles": len(f),
            "scaled_vertices_float64_le_sha256": hashlib.sha256(
                v.tobytes()
            ).hexdigest(),
            "triangles_int64_le_sha256": hashlib.sha256(f.tobytes()).hexdigest(),
        }
    (bread, bread_faces), (pan, pan_faces) = arrays
    basin_faces = pan_faces[(pan[pan_faces, 2] <= PAN_BASIN_LOCAL_Z_MAX).all(axis=1)]
    if not len(basin_faces):
        raise ValueError(
            "Placement input geometry has no triangles within the basin mask."
        )
    placing = placing_targets.detach().cpu().double().numpy()
    support = support_targets.detach().cpu().double().numpy()
    rows = []
    for row, (bread_pose, pan_pose) in enumerate(zip(placing, support, strict=True)):
        bread_world = bread @ bread_pose[:3, :3].T + bread_pose[:3, 3]
        pan_world = pan @ pan_pose[:3, :3].T + pan_pose[:3, 3]
        bread_triangles = bread_world[bread_faces]
        basin_triangles = pan_world[basin_faces]
        footprint, basin = _projection(bread_triangles), _projection(basin_triangles)
        if footprint.is_empty or basin.is_empty:
            raise ValueError("Placement input geometry has an empty XY projection.")
        # Reduce precision after union, which removes roundoff-sized false holes.
        # Keep unsnapped geometry for the original vertical-ray sample locations.
        coverage_footprint = set_precision(footprint, _PROJECTION_GRID_M)
        coverage_basin = set_precision(basin, _PROJECTION_GRID_M)
        outside = coverage_footprint.difference(coverage_basin)
        covered = outside.is_empty
        x0, y0, x1, y1 = footprint.bounds
        xx, yy = np.meshgrid(
            np.arange(x0, x1, _SAMPLE_SPACING_M), np.arange(y0, y1, _SAMPLE_SPACING_M)
        )
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        grid = grid[intersects_xy(footprint, grid[:, 0], grid[:, 1])]
        polygons = [footprint] if footprint.geom_type == "Polygon" else footprint.geoms
        boundary = []
        for polygon in polygons:
            for ring in (polygon.exterior, *polygon.interiors):
                count = max(1, int(np.ceil(ring.length / _SAMPLE_SPACING_M)))
                boundary.extend(
                    ring.interpolate(i / count, normalized=True).coords[0]
                    for i in range(count)
                )
        samples = np.unique(
            np.concatenate((grid, np.asarray(boundary), bread_world[:, :2])), axis=0
        )
        bottom = _ray_heights(samples, bread_triangles, upper=False)
        top = _ray_heights(samples, basin_triangles, upper=True)
        finite = np.isfinite(bottom) & np.isfinite(top)
        gaps = bottom[finite] - top[finite]
        minimum = float(gaps.min()) if len(gaps) else None
        rows.append(
            {
                "env_index": row,
                "qualified": bool(
                    covered and finite.all() and minimum is not None and minimum >= 0
                ),
                "footprint_covered": bool(covered),
                "footprint_area_m2": float(footprint.area),
                "unsupported_area_m2": float(outside.area),
                "boundary_clearance_m": (
                    float(coverage_footprint.distance(coverage_basin.boundary))
                    if covered
                    else None
                ),
                "sample_count": len(samples),
                "samples_without_surface": int((~finite).sum()),
                "minimum_sampled_vertical_gap_m": minimum,
                "negative_gap_samples": int((gaps < 0).sum()),
                "placing_target_pose": bread_pose.tolist(),
                "support_target_pose": pan_pose.tolist(),
            }
        )
    return {
        "method": "scaled_mesh_projected_triangle_union_and_vertical_rays",
        "scope": "complete_xy_footprint_and_sampled_height_only_not_continuous_3d_collision_or_physical_success",
        "frame": "arena",
        "meshes": meshes,
        "basin_local_z_max_m": PAN_BASIN_LOCAL_Z_MAX,
        "basin_triangle_count": len(basin_faces),
        "sample_spacing_m": _SAMPLE_SPACING_M,
        "coverage_precision_grid_m": _PROJECTION_GRID_M,
        "samples": "all_mesh_vertices_xy_plus_footprint_boundary_and_interior_grid",
        "ray_barycentric_epsilon": _BARYCENTRIC_EPS,
        "minimum_gap_threshold_m": 0.0,
        "rows": rows,
    }
