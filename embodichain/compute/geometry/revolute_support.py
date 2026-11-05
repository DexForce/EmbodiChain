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

"""Analytic complete-interval revolute support geometry without simulation."""

from __future__ import annotations
import math
import numpy as np

__all__ = ["revolute_swept_min_z"]


def revolute_swept_min_z(
    points: np.ndarray,
    rotation: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    limits: tuple[float, float],
    scale: float,
) -> float:
    """Minimize world Z over all vertices and the complete revolute interval.

    Points, origin, and axis use the unscaled native root frame. ``rotation``
    maps that frame to world; translation is deliberately excluded. Rodrigues'
    formula reduces each vertex height to ``C + A*cos(q) + B*sin(q)``. Both
    interval endpoints and every applicable sinusoid minimum are considered,
    so tilted deployments cannot hide collisions between sampled angles.

    Args:
        points: Unscaled moving vertices with shape ``(N, 3)``.
        rotation: Rigid source-root-to-world 3x3 orientation.
        origin: Unscaled root-frame pivot.
        axis: Finite nonzero source joint axis.
        limits: Finite ordered joint interval, including equal endpoints.
        scale: Positive uniform source scale.

    Returns:
        Minimum scaled world height, excluding root translation.

    Raises:
        ValueError: Geometry, transform, interval or scale is invalid.
    """
    points, rotation, origin, axis = (
        np.asarray(value, dtype=float) for value in (points, rotation, origin, axis)
    )
    bounds = np.asarray(limits, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("E8 support requires finite moving collision vertices.")
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("E8 support requires a rigid root rotation.")
    if (
        origin.shape != (3,)
        or axis.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(axis).all()
        or np.linalg.norm(axis) <= 0
    ):
        raise ValueError("E8 support requires a finite pivot and nonzero axis.")
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError("E8 support requires finite ordered joint limits.")
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("E8 support requires a positive uniform scale.")

    unit_axis = axis / np.linalg.norm(axis)
    offsets = points - origin
    parallel = (offsets @ unit_axis)[:, None] * unit_axis
    world_up = rotation[2]
    constant = (parallel + origin) @ world_up
    cosine = (offsets - parallel) @ world_up
    sine = np.cross(unit_axis, offsets) @ world_up
    low, high = bounds
    values = np.minimum(
        cosine * np.cos(low) + sine * np.sin(low),
        cosine * np.cos(high) + sine * np.sin(high),
    )
    critical = np.arctan2(sine, cosine) + math.pi
    # Select the first periodic minimum at or above the interval's lower bound.
    first_minimum = critical + 2 * math.pi * np.ceil((low - critical) / (2 * math.pi))
    reaches_minimum = (high - low >= 2 * math.pi) | (first_minimum <= high)
    values = np.where(reaches_minimum, -np.hypot(cosine, sine), values)
    return float(np.min(constant + values) * scale)
