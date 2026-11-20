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


"""Reusable device-resident sampling storage for geometric indentation."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
import warp as wp

from embodichain.utils.device_utils import current_warp_stream

from ._warp.indentation import cast_indentation, select_pad_candidates
from .raycast import TriangleRayMesh


class _IndentationQuery:
    def __init__(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        depths: torch.Tensor,
        indices: torch.Tensor,
        depth_limit: float,
    ) -> None:
        self.device = points.device
        self.depths = depths
        self.indices = indices
        self.depth_limit = depth_limit
        self.points = wp.from_torch(points, dtype=wp.vec3)
        self.normals = wp.from_torch(normals, dtype=wp.vec3)
        self.depth_array = wp.from_torch(depths)
        self.index_array = wp.from_torch(indices)
        self.candidates = wp.empty(
            (depths.shape[0],), dtype=wp.bool, device=str(self.device)
        )
        bounds = points.amin(dim=0), points.amax(dim=0)
        center = (bounds[0] + bounds[1]) * 0.5
        self.pad_center = wp.vec3(*center.tolist())
        radius = float((bounds[1] - center).norm())
        # Inflate the broad phase slightly to keep it conservative at touching bounds.
        self.pad_radius = radius + max(radius * 1.0e-5, 1.0e-6)
        self.meshes: list[tuple[TriangleRayMesh, wp.vec3, wp.vec3, float]] = []

    def bind(self, meshes: Sequence[TriangleRayMesh]) -> None:
        for mesh in meshes:
            vertices = mesh._vertices.detach().cpu().numpy()
            faces = mesh._indices.detach().cpu().numpy().reshape(-1, 3)
            # Render meshes can split vertices along seams. Match exactly
            # coincident positions for the closed-surface topology check.
            _, welded = np.unique(vertices, axis=0, return_inverse=True)
            topology = welded.reshape(-1)[faces]
            edges = np.concatenate(
                (topology[:, [0, 1]], topology[:, [1, 2]], topology[:, [2, 0]])
            )
            undirected = np.sort(edges, axis=1)
            _, inverse, counts = np.unique(
                undirected, axis=0, return_inverse=True, return_counts=True
            )
            directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
            if not (
                (topology[:, 0] != topology[:, 1]).all()
                and (topology[:, 1] != topology[:, 2]).all()
                and (topology[:, 2] != topology[:, 0]).all()
                and (counts == 2).all()
                and (np.bincount(inverse, weights=directions) == 0).all()
            ):
                raise ValueError(
                    "Indentation targets must be closed meshes with consistent winding."
                )
            lower = vertices.min(axis=0)
            upper = vertices.max(axis=0)
            # A positive enclosed volume rejects globally inverted triangles.
            local = vertices - lower
            a, b, c = local[faces].transpose(1, 0, 2)
            if float(np.einsum("ij,ij->", a, np.cross(b, c))) <= 0.0:
                raise ValueError(
                    "Indentation targets require outward-facing triangles."
                )
            search = float(np.linalg.norm(upper - lower))
            if not math.isfinite(search) or search <= 0:
                raise ValueError(
                    "Indentation requires a finite three-dimensional target."
                )
            self.meshes.append((mesh, wp.vec3(*lower), wp.vec3(*upper), search * 1.01))

    def cast(
        self,
        mounts: torch.Tensor,
        poses: Sequence[torch.Tensor],
        *,
        use_candidates: bool,
    ) -> None:
        self.depths.zero_()
        self.indices.fill_(-1)
        stream = current_warp_stream(self.device)
        mount_array = wp.from_torch(mounts.contiguous(), dtype=wp.mat44)
        for index, ((mesh, lower, upper, search), pose) in enumerate(
            zip(self.meshes, poses, strict=True)
        ):
            if mesh._ready is not None:
                torch.cuda.current_stream(self.device).wait_event(mesh._ready)
            pose_array = wp.from_torch(pose.contiguous(), dtype=wp.mat44)
            if use_candidates:
                wp.launch(
                    select_pad_candidates,
                    dim=self.depths.shape[0],
                    inputs=[
                        mount_array,
                        pose_array,
                        self.pad_center,
                        self.pad_radius,
                        lower,
                        upper,
                    ],
                    outputs=[self.candidates],
                    device=str(self.device),
                    stream=stream,
                )
            wp.launch(
                cast_indentation,
                dim=self.depths.shape,
                inputs=[
                    mesh._mesh.id,
                    mount_array,
                    pose_array,
                    self.points,
                    self.normals,
                    self.candidates,
                    use_candidates,
                    search,
                    self.depth_limit,
                    index,
                ],
                outputs=[self.depth_array, self.index_array],
                device=str(self.device),
                stream=stream,
            )
