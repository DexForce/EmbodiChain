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

"""Deterministic volume sampling, independent of simulation objects."""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np

__all__ = ["sample_volume_particles"]


def sample_volume_particles(
    kind: str,
    spacing: float,
    *,
    size: Sequence[float] = (0.05, 0.05, 0.05),
    radius: float = 0.025,
    height: float = 0.05,
    clearance: float = 0.0,
    profile: Sequence[Sequence[float]] | None = None,
    fill_level: float = 1.0,
    mesh_path: str | None = None,
    max_particles: int = 1_000_000,
) -> np.ndarray:
    """Sample a local primitive, revolved profile or closed mesh on a cubic lattice.

    Args:
        kind: ``box``, ``sphere``, ``cylinder`` (Z axis),
            ``revolved_volume`` (Z-axis profile), or ``mesh_volume``.
        spacing: Positive lattice spacing, metres.
        size: Full box dimensions, metres.
        radius: Sphere or cylinder radius, metres.
        height: Full cylinder height, metres.
        clearance: Inward distance from the volume boundary, metres.
        profile: For ``revolved_volume``, at least two ``(z, radius)`` pairs
            with strictly increasing Z and nonnegative radii, in local metres.
            Linear segments define the side wall; both ends are capped.
        fill_level: Fraction of the source's full Z extent to fill, in ``(0, 1]``.
            The free-surface cutoff is independent of solid-wall clearance.
        mesh_path: Closed fill-volume mesh for ``mesh_volume``.
        max_particles: Maximum candidate-grid size before allocation.

    Returns:
        Contiguous float32 local positions of shape ``(N, 3)``.
    """
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive.")
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("clearance must be finite and nonnegative.")
    if not math.isfinite(fill_level) or not 0 < fill_level <= 1:
        raise ValueError("fill_level must be finite and in (0, 1].")
    if type(max_particles) is not int or max_particles <= 0:
        raise ValueError("max_particles must be a positive integer.")
    mesh = None
    section = None
    if kind == "revolved_volume":
        section = np.asarray(profile, dtype=float)
        if (
            section.ndim != 2
            or section.shape[1] != 2
            or len(section) < 2
            or not np.isfinite(section).all()
            or np.any(np.diff(section[:, 0]) <= 0)
            or np.any(section[:, 1] < 0)
            or not np.any(section[:, 1] > 0)
        ):
            raise ValueError(
                "profile requires finite (z, radius) pairs with increasing Z and nonnegative radii."
            )
    elif profile is not None:
        raise ValueError("profile is only valid for revolved_volume.")
    if kind == "mesh_volume":
        import trimesh

        if mesh_path is None:
            raise ValueError("mesh_volume requires mesh_path.")
        mesh = trimesh.load(mesh_path, force="mesh")
        if not mesh.is_watertight or not mesh.is_volume:
            raise ValueError("Fluid fill mesh must be a closed, oriented volume.")
        lower, upper = mesh.bounds
    else:
        if mesh_path is not None:
            raise ValueError("mesh_path is only valid for mesh_volume.")
        if kind == "box":
            extent = np.asarray(size, dtype=float)
        elif kind == "sphere":
            extent = np.array([2 * radius] * 3)
        elif kind == "cylinder":
            extent = np.array([2 * radius, 2 * radius, height])
        elif section is not None:
            max_radius = section[:, 1].max()
            extent = np.array([2 * max_radius, 2 * max_radius, np.ptp(section[:, 0])])
        else:
            raise ValueError(f"Unsupported fluid volume {kind!r}.")
        if extent.shape != (3,) or not np.isfinite(extent).all() or (extent <= 0).any():
            raise ValueError("Volume dimensions must be finite and positive.")
        lower, upper = -extent / 2, extent / 2
        if section is not None:
            lower[2], upper[2] = section[0, 0], section[-1, 0]
    surface_z = lower[2] + fill_level * (upper[2] - lower[2])
    lower, upper = lower + clearance, upper - clearance
    upper[2] = min(upper[2], surface_z)
    counts = [max(0, math.ceil(float(v) / spacing)) for v in upper - lower]
    if not all(counts):
        raise ValueError("clearance leaves an empty fill volume.")
    if math.prod(counts) > max_particles:
        raise ValueError("Fluid candidate grid exceeds max_particles.")
    axes = [lo + (np.arange(n) + 0.5) * spacing for lo, n in zip(lower, counts)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    keep = np.all(points < upper, axis=1)
    if kind == "sphere":
        keep &= np.linalg.norm(points, axis=1) < radius - clearance
    elif kind == "cylinder":
        keep &= np.linalg.norm(points[:, :2], axis=1) < radius - clearance
    elif section is not None:
        radial = np.linalg.norm(points[:, :2], axis=1)
        keep &= radial < np.interp(points[:, 2], section[:, 0], section[:, 1])
        if clearance:
            # Rotational symmetry reduces side-wall distance to the meridian
            # plane. Include all segments, including steep or concave profiles.
            meridian = np.column_stack((points[:, 2], radial))
            distance2 = np.full(len(points), np.inf)
            for start, end in zip(section[:-1], section[1:]):
                segment = end - start
                t = np.clip(
                    (meridian - start) @ segment / np.dot(segment, segment), 0, 1
                )
                delta = meridian - (start + t[:, None] * segment)
                distance2 = np.minimum(distance2, np.einsum("ij,ij->i", delta, delta))
            keep &= distance2 >= clearance**2
    elif mesh is not None:
        import trimesh

        keep &= mesh.contains(points)
        if clearance:
            _, distance, _ = trimesh.proximity.closest_point(mesh, points)
            keep &= distance >= clearance
    points = points[keep]
    if not len(points):
        raise ValueError("Fluid volume contains no particles at this spacing.")
    return np.ascontiguousarray(points, dtype=np.float32)
