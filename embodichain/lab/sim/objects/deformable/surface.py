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

"""DexSim surface-deformable object implementation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from dexsim.scene import Scene, SpawnedClothParticleSet
from scipy.spatial import cKDTree

from .base import DeformableObject
from .data import _ParticleSetData

__all__ = [
    "ClothBodyData",
    "ClothObject",
    "SurfaceDeformableData",
    "SurfaceDeformableObject",
]


class SurfaceDeformableData(_ParticleSetData):
    """DexSim cloth buffers exposed through the common nodal contract."""

    def __init__(
        self,
        entities: Sequence[SpawnedClothParticleSet],
        scene: Scene,
        device: torch.device,
    ) -> None:
        super().__init__(entities, scene, device)
        self.n_vertices = self.num_nodes

    @property
    def rest_vertices(self) -> torch.Tensor:
        """Return rest surface vertices in simulation world frame."""
        return self.default_nodal_state_w[..., :3]

    @property
    def vertex_position(self) -> torch.Tensor:
        """Return current surface vertices in simulation world frame."""
        return self.nodal_pos_w

    @property
    def vertex_velocity(self) -> torch.Tensor:
        """Return current surface-vertex velocities."""
        return self.nodal_vel_w


class SurfaceDeformableObject(DeformableObject):
    """A batch of DexSim surface deformables backed by ``ClothBody``."""

    deformable_type = "surface"
    spawn_kind = "cloth_object"
    display_name = "surface deformable"

    def _create_data(
        self,
        entities: Sequence[Any],
        scene: Scene,
        device: torch.device,
    ) -> SurfaceDeformableData:
        return SurfaceDeformableData(entities, scene, device)

    def _initialize_topology(self, entities: Sequence[Any]) -> None:
        entity = entities[0]
        initial_world_pose = np.asarray(entity.desc.pose, dtype=np.float32).reshape(
            4, 4
        )
        arena_index = self._spawn_result.arenas.index(entity.arena_name)
        initial_world_pose[:3, 3] += self._spawn_result.arenas.root_offsets[arena_index]
        self._surface_triangles = self._build_surface_triangles(
            entity,
            self.body_data.rest_vertices[0].detach().cpu().numpy(),
            initial_world_pose,
        )

    @property
    def body_data(self) -> SurfaceDeformableData | None:
        """Compatibility view of the DexSim cloth data."""
        return self._data

    @staticmethod
    def _build_surface_triangles(
        entity: SpawnedClothParticleSet,
        rest_vertices: np.ndarray,
        initial_world_pose: np.ndarray,
    ) -> np.ndarray:
        """Map render triangles onto DexSim's welded cloth vertex buffer."""
        render_body = entity.get_render_body()
        if render_body is None:
            raise RuntimeError("Surface-deformable Spawn handle has no render body.")
        render_vertices: list[np.ndarray] = []
        render_triangles: list[np.ndarray] = []
        vertex_offset = 0
        for mesh_id in range(render_body.get_mesh_count()):
            vertices = entity.get_render_vertices(mesh_id)
            triangles = entity.get_render_triangles(mesh_id)
            render_vertices.append(vertices)
            render_triangles.append(triangles + vertex_offset)
            vertex_offset += len(vertices)

        vertices = np.concatenate(render_vertices, axis=0)
        triangles = np.concatenate(render_triangles, axis=0)
        initial_world_pose = np.asarray(initial_world_pose, dtype=np.float32).reshape(
            4, 4
        )
        vertices = vertices @ initial_world_pose[:3, :3].T + initial_world_pose[:3, 3]
        # Runtime preparation may advance a newly created cloth by one small
        # step before its particle batch is bound. Topology is invariant to
        # that uniform root translation, so align the render snapshot with the
        # captured simulation nodes before resolving the welded vertex IDs.
        vertices += rest_vertices.mean(axis=0) - vertices.mean(axis=0)
        distances, cloth_vertex_ids = cKDTree(rest_vertices).query(vertices)
        scale = max(float(np.ptp(rest_vertices, axis=0).max()), 1.0)
        if float(distances.max(initial=0.0)) > scale * 1.0e-5:
            raise RuntimeError(
                "Could not map surface-deformable render vertices onto the "
                "physical vertex buffer."
            )
        return np.asarray(cloth_vertex_ids[triangles], dtype=np.int32)

    def get_rest_vertex_position(self) -> torch.Tensor:
        """Return rest surface-vertex positions."""
        self._require_data()
        return self.body_data.rest_vertices

    def get_current_vertex_position(self) -> torch.Tensor:
        """Return current surface-vertex positions."""
        return self.get_current_nodal_position()

    def get_current_vertex_velocity(self) -> torch.Tensor:
        """Return current surface-vertex velocities."""
        return self.get_current_nodal_velocity()

    def get_surface_vertices(self) -> torch.Tensor:
        """Return the live cloth surface used for visualization."""
        return self.get_current_vertex_position()

    def get_surface_triangles(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return surface triangle indices for selected instances."""
        ids = self._resolve_env_ids(env_ids)
        triangles = torch.as_tensor(
            self._surface_triangles, dtype=torch.int32, device=self.device
        )
        return triangles.unsqueeze(0).expand(len(ids), -1, -1).clone()


# Compatibility names retained for existing environments and tutorials.
ClothBodyData = SurfaceDeformableData
ClothObject = SurfaceDeformableObject
