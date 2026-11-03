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
"""Source-compatible circular-prism qualification from explicit mesh arrays."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

__all__ = ["circular_prism_radius"]


def circular_prism_radius(
    vertices: np.ndarray,
    faces: np.ndarray,
    axis: np.ndarray,
    origin: np.ndarray,
    *,
    extra_colliders: Sequence[np.ndarray] = (),
) -> float:
    """Qualify two complete radial rings and closed convex triangle topology.

    Args:
        vertices: Full grip-shell vertices in one link-local frame.
        faces: Matching indexed triangles, preserving authored order.
        axis: Unit hinge axis in that frame.
        origin: Hinge origin in that frame.
        extra_colliders: Other rotor collider vertices in the same frame.

    Returns:
        Maximum qualified radius in metres. This source-compatible proof requires
        all grip vertices on exactly two uniform rings with at least 16 points;
        it does not prove native cooked-shape or continuous motion equivalence.

    Raises:
        ValueError: The source shape cannot establish this geometric boundary.
    """
    import trimesh

    axis, origin = np.asarray(axis), np.asarray(origin)
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or axis.shape != (3,)
        or origin.shape != (3,)
        or not np.isfinite(origin).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.dtype.kind not in "iu"
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError(
            "Circular-prism proof requires complete finite indexed geometry."
        )
    if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6):
        raise ValueError("Circular-prism proof requires a unit axis.")
    centered = vertices - origin
    axial = centered @ axis
    radial = centered - axial[:, None] * axis
    radius = np.linalg.norm(radial, axis=1)
    tolerance = max(1e-7, float(radius.max()) * 1e-5)
    if radius.min() <= tolerance or np.ptp(radius) > tolerance:
        raise ValueError("Grip shell is not a circular prism.")
    levels = []
    for value in sorted(axial):
        if not levels or abs(value - levels[-1]) > tolerance:
            levels.append(float(value))
    if len(levels) != 2:
        raise ValueError("Grip shell requires exactly two axial rings.")
    reference = np.eye(3)[int(np.argmin(abs(axis)))]
    x = np.cross(axis, reference)
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)
    for level in levels:
        ring = radial[np.abs(axial - level) <= tolerance]
        angles = np.unique(
            np.round(np.mod(np.arctan2(ring @ y, ring @ x), 2 * math.pi), 6)
        )
        if len(angles) > 1 and angles[-1] - angles[0] > 2 * math.pi - 1e-5:
            angles = angles[:-1]
        gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
        if len(angles) < 16 or np.max(np.abs(gaps - 2 * math.pi / len(angles))) > 1e-4:
            raise ValueError("Grip shell has incomplete/nonuniform angular coverage.")
    surface = trimesh.Trimesh(
        vertices=vertices.copy(),
        faces=faces.copy(),
        process=True,
    )
    if (
        not surface.is_watertight
        or not surface.is_winding_consistent
        or not surface.is_convex
        or surface.euler_number != 2
    ):
        raise ValueError("Grip shell lacks closed convex triangle topology.")
    for extra in extra_colliders:
        extra = np.asarray(extra)
        if (
            extra.ndim != 2
            or extra.shape[1] != 3
            or not len(extra)
            or not np.isfinite(extra).all()
        ):
            raise ValueError("Extra rotor collider geometry is unavailable.")
        other = extra - origin
        other = other - (other @ axis)[:, None] * axis
        if np.linalg.norm(other, axis=1).max() >= radius.min() - 4 * tolerance:
            raise ValueError("Extra rotor collider reaches the qualified grip shell.")
    return float(radius.max())
