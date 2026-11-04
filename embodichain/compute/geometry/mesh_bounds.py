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

import torch

__all__ = ["transform_vertex_bounds", "aabb_sphere_cover"]


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
