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


"""Estimate the native mechanism frame of a settled Scene Engine proxy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.pipeline.utils._articulation_proxy_fit import (
    ArticulationProxyFit,
    fit_articulation_proxy,
)

__all__ = ["ArticulationProxyAlignment", "align_articulation_proxy"]


@dataclass(frozen=True)
class ArticulationProxyAlignment:
    """The fitted pose and evidence from the standard-frame search."""

    fit: ArticulationProxyFit
    proxy_yaw_degrees: float
    native_to_standard: np.ndarray
    shape_score: float | None
    near_equal_frames: int | None
    native_heading_degrees: int
    orientation_source: str
    orientation_reason: str | None


def align_articulation_proxy(
    *,
    proxy_oriented_vertices: np.ndarray,
    proxy_world_rotation: np.ndarray,
    proxy_world_position: np.ndarray,
    native_footprint_vertices: np.ndarray,
    native_collision_vertices: np.ndarray,
    native_root_up: np.ndarray,
    support_z: float,
    saved_heading_degrees: int | None = None,
    review_heading: Callable[[float, np.ndarray], tuple[int | None, str]] | None = None,
) -> ArticulationProxyAlignment:
    """Align native geometry to the proxy's settled pose expressed without yaw.

    All inputs use Z-up. Proxy vertices already carry world rotation and scale,
    with world translation supplied separately. The native points are relative
    to the physical base in the same joint state used by the simulator.
    """
    # Extract the horizontal twist closest to the full settled proxy rotation.
    matrix = proxy_world_rotation
    cosine = matrix[0, 0] + matrix[1, 1]
    sine = matrix[1, 0] - matrix[0, 1]
    if np.hypot(cosine, sine) < 1e-6:
        cosine, sine = matrix[0, 0], matrix[1, 0]
    yaw = float(np.degrees(np.arctan2(sine, cosine)))
    yaw_rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
    standard_proxy = proxy_oriented_vertices @ yaw_rotation
    bounds = np.stack(
        [proxy_oriented_vertices.min(axis=0), proxy_oriented_vertices.max(axis=0)]
    )
    proxy_center_xy = bounds.mean(axis=0)[:2] + proxy_world_position[:2]
    align_up = Rotation.align_vectors([[0.0, 0.0, 1.0]], [native_root_up])[
        0
    ].as_matrix()
    reviewed_heading, reason = (
        review_heading(yaw, align_up) if review_heading is not None else (None, None)
    )
    heading = (
        reviewed_heading if reviewed_heading is not None else saved_heading_degrees
    )
    if heading is None:
        native_alignment, score, count, heading = _select_geometry_frame(
            standard_proxy, native_footprint_vertices, align_up
        )
        orientation_source = "geometry"
    else:
        native_alignment = (
            Rotation.from_euler("z", heading, degrees=True).as_matrix() @ align_up
        )
        score, count = None, None
        orientation_source = "vision" if reviewed_heading is not None else "saved"
    fit = fit_articulation_proxy(
        proxy_standard_vertices=standard_proxy,
        native_footprint_vertices=native_footprint_vertices,
        native_collision_vertices=native_collision_vertices,
        native_to_standard=native_alignment,
        proxy_yaw_degrees=yaw,
        proxy_center_xy=proxy_center_xy,
        support_z=support_z,
    )
    return ArticulationProxyAlignment(
        fit,
        yaw,
        native_alignment,
        score,
        count,
        heading,
        orientation_source,
        reason,
    )


def _unit_surface_points(vertices: np.ndarray) -> np.ndarray:
    """Use equal-density occupied voxels to limit tessellation bias."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ValueError(
            "Articulation frame matching needs three-dimensional vertices."
        )
    bounds = np.stack([points.min(axis=0), points.max(axis=0)])
    extent = float(np.ptp(bounds, axis=0).max())
    if not np.isfinite(bounds).all() or not np.isfinite(extent) or extent <= 0:
        raise ValueError("Articulation frame matching needs finite positive extent.")
    normalized = (points - bounds.mean(axis=0)) / extent
    return np.unique(np.rint(normalized * 24), axis=0) / 24


def _select_geometry_frame(
    proxy: np.ndarray,
    native: np.ndarray,
    align_up: np.ndarray,
) -> tuple[np.ndarray, float, int, int]:
    """Score four upright headings when an image-based choice is unavailable."""
    target = _unit_surface_points(proxy)
    source = _unit_surface_points(native)
    yaw_vectors = np.zeros((4, 3), dtype=float)
    yaw_vectors[:, 2] = np.arange(4) * (np.pi / 2)
    rotations = Rotation.from_rotvec(yaw_vectors).as_matrix() @ align_up
    target_tree = cKDTree(target)
    scores = []
    for rotation in rotations:
        candidate = source @ rotation.T
        forward = target_tree.query(candidate)[0].mean()
        reverse = cKDTree(candidate).query(target)[0].mean()
        scores.append(float((forward + reverse) / 2))
    # A box or other nearly symmetric body can have indistinguishable frames.
    # Favor the authored base axes among similar fits and expose the ambiguity.
    close = np.flatnonzero(np.asarray(scores) <= min(scores) + 0.004)
    index = int(close[0])
    return rotations[index], scores[index], len(close), 90 * index
