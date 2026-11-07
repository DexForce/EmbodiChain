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

"""Input-mesh bounds and conservative sphere envelopes."""

from __future__ import annotations

import math
import numpy as np
from typing import TYPE_CHECKING
from collections.abc import Sequence, Iterator
import torch

if TYPE_CHECKING:
    import trimesh

__all__ = [
    "input_mesh_bounds",
    "input_mesh_radius",
    "iter_clearance_translations",
    "transform_vertex_bounds",
    "aabb_sphere_cover",
    "maximum_halfspace_violation",
]


def transform_vertex_bounds(
    mesh: torch.Tensor, pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact bounds of transformed vertices for one environment.

    Args:
        mesh: Nonempty owning-frame vertices, shape ``(1, V, 3)``.
        pose: Owning-frame-to-output transform, shape ``(1, 4, 4)``.

    Returns:
        Minimum and maximum output-frame coordinates, each with shape ``(1, 3)``.
        Vertices are converted to the pose's dtype/device before projection.

    Raises:
        ValueError: If geometry, transform shapes or finite values are invalid.
    """
    if (
        mesh.ndim != 3
        or mesh.shape[0] != 1
        or mesh.shape[-1] != 3
        or not mesh.shape[1]
        or not torch.isfinite(mesh).all()
        or pose.shape != (1, 4, 4)
        or not torch.isfinite(pose).all()
    ):
        raise ValueError("scene_geometry_or_pose_unavailable")
    world = mesh.to(pose) @ pose[:, :3, :3].transpose(-1, -2) + pose[:, None, :3, 3]
    return world.amin(1), world.amax(1)


def aabb_sphere_cover(vertices: torch.Tensor) -> torch.Tensor:
    """Cover an input vertex AABB with 27 circumscribed cell spheres.

    Args:
        vertices: Nonempty object-local vertices with shape ``(V, 3)``.

    Returns:
        Float32 sphere centers/radii, shape ``(27, 4)``, in the input frame and
        device. This bounds the input envelope, not native cooked geometry.

    Raises:
        ValueError: If vertices have an invalid shape or nonfinite coordinates.
    """
    points = torch.as_tensor(vertices, dtype=torch.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not points.shape[0]:
        raise ValueError("Sphere cover requires nonempty finite vertices (V, 3).")
    low, high = points.amin(0), points.amax(0)
    # These existing reductions propagate NaN/Inf. Validate their six values
    # instead of allocating and reducing an additional V-by-3 finite mask.
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise ValueError("Sphere cover requires nonempty finite vertices (V, 3).")
    cell = (high - low) / 3
    index = torch.arange(3, dtype=points.dtype, device=points.device) + 0.5
    grid = torch.stack(
        torch.meshgrid(index, index, index, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    centers = low + grid * cell
    radius = torch.linalg.vector_norm(cell) / 2
    return torch.cat((centers, radius.expand(len(centers), 1)), dim=1)


def maximum_halfspace_violation(
    points: np.ndarray, normals: np.ndarray, offsets: np.ndarray
) -> float:
    """Measure the largest point/plane residual with bounded multiplication scratch.

    Args:
        points: Finite floating points, shape (N, D), N > 0.
        normals: Finite halfspace normals, shape (F, D), F > 0.
        offsets: Finite plane offsets, shape (F,). Inside means point @ normal
            <= offset; callers own normalization and the allowed tolerance.

    Returns:
        Maximum signed residual over every point and plane. Original point/plane
        order and 256-row chunks are preserved; no vertices are subsampled.
    """
    arrays = (points, normals, offsets)
    if (
        any(
            not isinstance(value, np.ndarray)
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
            for value in arrays
        )
        or points.ndim != 2
        or normals.ndim != 2
        or offsets.shape != (len(normals),)
        or points.shape[1] != normals.shape[1]
        or not len(points)
        or not len(normals)
    ):
        raise ValueError("Halfspace cover requires finite floating point/plane arrays.")
    maximum = -np.inf
    for start in range(0, len(points), 256):
        values = points[start : start + 256] @ normals.T
        values -= offsets
        maximum = np.maximum(maximum, np.max(values))
        del values
    return float(maximum)


def input_mesh_bounds(
    meshes: Sequence[trimesh.Trimesh], pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform the local AABB corners of an ordered input mesh group.

    The caller owns geometry/pose validation. This preserves the original corner
    construction and NumPy/Trimesh projection without sampling or convexification.
    """
    import trimesh

    vertices = np.vstack([mesh.vertices for mesh in meshes])
    corners = trimesh.bounds.corners(np.array([vertices.min(0), vertices.max(0)]))
    world = trimesh.transform_points(corners, pose)
    return world.min(0), world.max(0)


def input_mesh_radius(meshes: Sequence[trimesh.Trimesh], pose: np.ndarray) -> float:
    """Measure the largest transformed vertex norm while preserving mesh order."""
    import trimesh

    return max(
        float(
            np.linalg.norm(trimesh.transform_points(mesh.vertices, pose), axis=1).max()
        )
        for mesh in meshes
    )


def iter_clearance_translations(
    lower: np.ndarray,
    upper: np.ndarray,
    mesh_envelopes: Sequence[tuple[tuple[np.ndarray, np.ndarray], float]],
    plane_envelopes: Sequence[tuple[np.ndarray, float, float]],
    *,
    horizontal: int,
    reference: torch.Tensor,
) -> Iterator[tuple[torch.Tensor, str, torch.Tensor]]:
    """Yield the original horizontal +/- and upward clearance translations.

    Inputs are validated geometric envelopes. Magnitudes are rounded outward
    with nextafter in the reference dtype/device. Lazy iteration preserves the
    caller's interleaved collision-query order; no candidate is ranked here.
    """
    for axis, sign in ((horizontal, 1), (horizontal, -1), (2, 1)):
        magnitude = max(
            [0.0]
            + [
                (
                    float(bound[1][axis] + clearance - lower[axis])
                    if sign > 0
                    else float(upper[axis] - bound[0][axis] + clearance)
                )
                for bound, clearance in mesh_envelopes
            ]
        )
        possible = True
        for normal, signed, clearance in plane_envelopes:
            projection = sign * normal[axis]
            if projection <= 0:
                possible = False
                break
            magnitude = max(magnitude, (clearance - signed) / projection)
        if not possible or magnitude <= 0:
            continue
        value = reference.new_tensor(magnitude)
        value = torch.nextafter(value, value.new_tensor(math.inf))
        delta = torch.zeros((1, 3), dtype=reference.dtype, device=reference.device)
        delta[0, axis] = sign * value
        yield delta, ("+" if sign > 0 else "-") + "xyz"[axis], value
