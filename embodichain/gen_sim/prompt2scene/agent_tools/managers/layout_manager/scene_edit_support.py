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

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "build_xy_footprint",
    "clamp_center_to_support_region",
    "compute_simready_glb_xy_size",
    "support_region_default_center",
    "support_region_grid_center",
]


def compute_simready_glb_xy_size(
    glb_path: Path,
    *,
    metric_scale: dict[str, Any] | None = None,
) -> list[float]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Scene edit layout requires trimesh.") from exc

    scene = trimesh.load(glb_path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        dumped = scene.dump(concatenate=True)
        mesh = (
            dumped
            if isinstance(dumped, trimesh.Trimesh)
            else trimesh.util.concatenate(
                [item for item in dumped if isinstance(item, trimesh.Trimesh)]
            )
        )
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3):
        raise ValueError(f"Invalid GLB bounds shape: {bounds.shape}")
    size_x = float(bounds[1, 0] - bounds[0, 0])
    size_y = float(bounds[1, 2] - bounds[0, 2])
    scale_factor = 1.0
    if isinstance(metric_scale, dict):
        try:
            scale_factor = float(metric_scale.get("scale_factor", 1.0))
        except (TypeError, ValueError):
            scale_factor = 1.0
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        scale_factor = 1.0
    return [
        max(size_x * scale_factor, 1.0e-4),
        max(size_y * scale_factor, 1.0e-4),
    ]


def build_xy_footprint(
    *,
    center_xy: list[float],
    size_xy: list[float],
) -> dict[str, Any]:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    sx, sy = max(float(size_xy[0]), 0.0), max(float(size_xy[1]), 0.0)
    half_x = 0.5 * sx
    half_y = 0.5 * sy
    return {
        "unit": "m",
        "center_xy": [cx, cy],
        "aabb_xy": [
            [cx - half_x, cy - half_y],
            [cx + half_x, cy + half_y],
        ],
        "size_xy": [sx, sy],
    }


def clamp_center_to_support_region(
    *,
    center_xy: list[float],
    size_xy: list[float],
    support_region: dict[str, Any],
) -> list[float]:
    aabb_xy = support_region.get("aabb_xy")
    if not (
        isinstance(aabb_xy, list)
        and len(aabb_xy) == 2
        and all(isinstance(item, list) and len(item) == 2 for item in aabb_xy)
    ):
        return [float(center_xy[0]), float(center_xy[1])]
    min_xy = np.asarray(aabb_xy[0], dtype=np.float64)
    max_xy = np.asarray(aabb_xy[1], dtype=np.float64)
    half = 0.5 * np.asarray(size_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    lower = min_xy + half
    upper = max_xy - half
    clamped = center.copy()
    for axis in range(2):
        if lower[axis] <= upper[axis]:
            clamped[axis] = min(max(center[axis], lower[axis]), upper[axis])
        else:
            clamped[axis] = float(0.5 * (min_xy[axis] + max_xy[axis]))
    return clamped.tolist()


def support_region_default_center(
    *,
    support_region: dict[str, Any],
) -> np.ndarray:
    center_xy = support_region.get("center_xy")
    if isinstance(center_xy, list) and len(center_xy) == 2:
        return np.asarray(center_xy, dtype=np.float64)
    aabb_xy = support_region.get("aabb_xy")
    if (
        isinstance(aabb_xy, list)
        and len(aabb_xy) == 2
        and all(isinstance(item, list) and len(item) == 2 for item in aabb_xy)
    ):
        min_xy = np.asarray(aabb_xy[0], dtype=np.float64)
        max_xy = np.asarray(aabb_xy[1], dtype=np.float64)
        return 0.5 * (min_xy + max_xy)
    return np.zeros(2, dtype=np.float64)


def support_region_grid_center(
    *,
    support_region: dict[str, Any],
    grid_name: str,
) -> np.ndarray:
    aabb_xy = support_region.get("aabb_xy")
    if not (
        isinstance(aabb_xy, list)
        and len(aabb_xy) == 2
        and all(isinstance(item, list) and len(item) == 2 for item in aabb_xy)
    ):
        return np.zeros(2, dtype=np.float64)
    min_xy = np.asarray(aabb_xy[0], dtype=np.float64)
    max_xy = np.asarray(aabb_xy[1], dtype=np.float64)
    size = max_xy - min_xy
    cell = size / 3.0
    grid_to_rc = {
        "left_front": (0, 0),
        "center_front": (1, 0),
        "right_front": (2, 0),
        "left_center": (0, 1),
        "center": (1, 1),
        "right_center": (2, 1),
        "left_back": (0, 2),
        "center_back": (1, 2),
        "right_back": (2, 2),
        "front": (1, 0),
        "back": (1, 2),
        "left": (0, 1),
        "right": (2, 1),
    }
    col, row = grid_to_rc.get(grid_name, (1, 1))
    center_x = min_xy[0] + (col + 0.5) * cell[0]
    center_y = min_xy[1] + (row + 0.5) * cell[1]
    return np.asarray([center_x, center_y], dtype=np.float64)
