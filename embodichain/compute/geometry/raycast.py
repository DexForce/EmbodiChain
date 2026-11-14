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

"""Device-resident, batched closest-hit ray queries for rigid triangle meshes."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import warp as wp

from ._warp.raycast import cast_mesh

__all__ = ["TriangleRayMesh", "cast_rays"]


class TriangleRayMesh:
    """Build and retain a Warp BVH from local triangle geometry.

    Args:
        vertices: Float-compatible positions with shape (V, 3).
        triangles: Integer vertex indices with shape (F, 3).
    """

    def __init__(self, vertices: torch.Tensor, triangles: torch.Tensor) -> None:
        if vertices.ndim != 2 or vertices.shape[-1] != 3 or not len(vertices):
            raise ValueError("vertices must have a non-empty shape (V, 3).")
        if triangles.ndim != 2 or triangles.shape[-1] != 3 or not len(triangles):
            raise ValueError("triangles must have a non-empty shape (F, 3).")
        if triangles.dtype not in (torch.int32, torch.int64):
            raise TypeError("triangles must use integer indices.")
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("Mesh vertices must be finite.")
        if bool(((triangles < 0) | (triangles >= len(vertices))).any()):
            raise ValueError("Mesh indices are outside the vertex range.")
        self.device = vertices.device
        self._vertices = vertices.to(dtype=torch.float32).contiguous().clone()
        self._indices = (
            triangles.to(device=self.device, dtype=torch.int32)
            .flatten()
            .contiguous()
            .clone()
        )
        wp.get_device(str(self.device))
        stream = (
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
            if self.device.type == "cuda"
            else None
        )
        with wp.ScopedStream(stream):
            self._mesh = wp.Mesh(
                wp.from_torch(self._vertices, dtype=wp.vec3),
                wp.from_torch(self._indices),
            )
        self._ready = None
        if stream is not None:
            self._ready = torch.cuda.Event()
            self._ready.record()


def cast_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    meshes: Sequence[tuple[TriangleRayMesh, torch.Tensor]],
    max_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find nearest triangle intersections independently in each batch row.

    Args:
        origins: Ray origins, float32 (B, R, 3).
        directions: Ray directions, float32 (B, R, 3). Zero directions miss.
        meshes: Ordered local BVHs and rigid transforms, (B, 4, 4) or (1, 4, 4).
        max_distance: Positive finite cutoff in the same units as vertices.

    Returns:
        Distance and mesh index tensors (B, R). Misses are infinity and -1.
        Directions are normalized and an equal-distance tie favors the first mesh.
    """
    if origins.ndim != 3 or origins.shape[-1] != 3 or directions.shape != origins.shape:
        raise ValueError("Both ray arrays must have matching shape (B, R, 3).")
    device = origins.device
    if (
        directions.device != device
        or origins.dtype != torch.float32
        or directions.dtype != torch.float32
    ):
        raise TypeError("Rays must be float32 and reside on one device.")
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be finite and positive.")
    size = origins.shape[:2]
    distance = torch.full(size, torch.inf, device=device)
    indices = torch.full(size, -1, device=device, dtype=torch.int32)
    if not meshes or not distance.numel():
        return distance, indices
    stream = (
        wp.stream_from_torch(torch.cuda.current_stream(device))
        if device.type == "cuda"
        else None
    )
    wp.get_device(str(device))
    wp_origins = wp.from_torch(origins.contiguous(), dtype=wp.vec3)
    wp_directions = wp.from_torch(directions.contiguous(), dtype=wp.vec3)
    wp_distance, wp_indices = wp.from_torch(distance), wp.from_torch(indices)
    for index, (mesh, pose) in enumerate(meshes):
        if pose.shape not in ((1, 4, 4), (size[0], 4, 4)):
            raise ValueError(
                "Mesh transforms must match the ray batch or broadcast from one row."
            )
        if (
            mesh.device != device
            or pose.device != device
            or pose.dtype != torch.float32
        ):
            raise TypeError("Meshes and float32 transforms must share the ray device.")
        if mesh._ready is not None:
            torch.cuda.current_stream(device).wait_event(mesh._ready)
        wp.launch(
            cast_mesh,
            dim=tuple(size),
            inputs=[
                mesh._mesh.id,
                wp_origins,
                wp_directions,
                wp.from_torch(pose.contiguous()),
                max_distance,
                index,
            ],
            outputs=[wp_distance, wp_indices],
            device=str(device),
            stream=stream,
        )
    return distance, indices
