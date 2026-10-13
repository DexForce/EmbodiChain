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
    mesh_path: str | None = None,
    max_particles: int = 1_000_000,
) -> np.ndarray:
    """Sample an origin-centred primitive or closed mesh on a cubic lattice.

    Args:
        kind: ``box``, ``sphere``, ``cylinder`` (Z axis), or ``mesh_volume``.
        spacing: Positive lattice spacing, metres.
        size: Full box dimensions, metres.
        radius: Sphere or cylinder radius, metres.
        height: Full cylinder height, metres.
        clearance: Inward distance from the volume boundary, metres.
        mesh_path: Closed fill-volume mesh for ``mesh_volume``.
        max_particles: Maximum candidate-grid size before allocation.

    Returns:
        Contiguous float32 local positions of shape ``(N, 3)``.
    """
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive.")
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("clearance must be finite and nonnegative.")
    if type(max_particles) is not int or max_particles <= 0:
        raise ValueError("max_particles must be a positive integer.")
    mesh = None
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
        else:
            raise ValueError(f"Unsupported fluid volume {kind!r}.")
        if extent.shape != (3,) or not np.isfinite(extent).all() or (extent <= 0).any():
            raise ValueError("Volume dimensions must be finite and positive.")
        lower, upper = -extent / 2, extent / 2
    lower, upper = lower + clearance, upper - clearance
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
