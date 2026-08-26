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
from dexsim.engine import ClothBody, PhysicsScene
from dexsim.models import MeshObject
from dexsim.types import ClothBodyGPUAPIReadWriteType
from scipy.spatial import cKDTree

from .base import DeformableObject
from .data import DeformableObjectData

__all__ = [
    "ClothBodyData",
    "ClothObject",
    "SurfaceDeformableData",
    "SurfaceDeformableObject",
]


class SurfaceDeformableData(DeformableObjectData):
    """DexSim cloth buffers exposed through the common nodal contract."""

    def __init__(
        self,
        entities: Sequence[MeshObject],
        ps: PhysicsScene,
        device: torch.device,
    ) -> None:
        self.entities = list(entities)
        self.device = device
        self.ps = ps
        self.num_instances = len(self.entities)
        self.cloth_bodies: Sequence[ClothBody] = [
            entity.get_physical_body() for entity in self.entities
        ]
        self.n_vertices = self.cloth_bodies[0].get_num_vertices()

        self._rest_position_buffer = torch.empty(
            (self.num_instances, self.n_vertices, 4),
            device=self.device,
            dtype=torch.float32,
        )
        for i, cloth_body in enumerate(self.cloth_bodies):
            self._rest_position_buffer[i] = cloth_body.get_rest_position_buffer()

        self._vertex_position = torch.zeros(
            (self.num_instances, self.n_vertices, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._vertex_velocity = torch.zeros_like(self._vertex_position)
        self._default_nodal_state_w = torch.cat(
            (
                self._rest_position_buffer[..., :3],
                torch.zeros_like(self._rest_position_buffer[..., :3]),
            ),
            dim=-1,
        )

    @property
    def rest_vertices(self) -> torch.Tensor:
        """Return rest surface vertices in simulation world frame."""
        return self._rest_position_buffer[..., :3].clone()

    @property
    def vertex_position(self) -> torch.Tensor:
        """Return current surface vertices in simulation world frame."""
        for i, cloth_body in enumerate(self.cloth_bodies):
            self._vertex_position[i] = cloth_body.get_position_inv_mass_buffer()[:, :3]
        return self._vertex_position.clone()

    @property
    def vertex_velocity(self) -> torch.Tensor:
        """Return current surface-vertex velocities."""
        for i, cloth_body in enumerate(self.cloth_bodies):
            # DexSim stores velocity in the first xyz channels. The fourth
            # channel is padding/metadata and must not be exposed as velocity.
            self._vertex_velocity[i] = cloth_body.get_velocity_buffer()[:, :3]
        return self._vertex_velocity.clone()

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        return self.vertex_position

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        return self.vertex_velocity

    @property
    def default_nodal_state_w(self) -> torch.Tensor:
        return self._default_nodal_state_w.clone()


class SurfaceDeformableObject(DeformableObject):
    """A batch of DexSim surface deformables backed by ``ClothBody``."""

    deformable_type = "surface"
    spawn_kind = "cloth_object"
    display_name = "surface deformable"

    def _create_data(
        self,
        entities: Sequence[Any],
        physics_scene: PhysicsScene,
        device: torch.device,
    ) -> SurfaceDeformableData:
        return SurfaceDeformableData(entities, physics_scene, device)

    def _initialize_topology(self, entities: Sequence[Any]) -> None:
        self._surface_triangles = self._build_surface_triangles(
            entities[0],
            self.body_data.rest_vertices[0].detach().cpu().numpy(),
            self.body_data.cloth_bodies[0].get_initial_transform(),
        )

    @property
    def body_data(self) -> SurfaceDeformableData | None:
        """Compatibility view of the DexSim cloth data."""
        return self._data

    @staticmethod
    def _build_surface_triangles(
        entity: MeshObject,
        rest_vertices: np.ndarray,
        initial_transform: np.ndarray,
    ) -> np.ndarray:
        """Map render triangles onto DexSim's welded cloth vertex buffer."""
        render_body = entity.get_render_body()
        render_vertices: list[np.ndarray] = []
        render_triangles: list[np.ndarray] = []
        vertex_offset = 0
        for mesh_id in range(render_body.get_mesh_count()):
            vertices = np.asarray(render_body.get_vertices(mesh_id), dtype=np.float32)
            triangles = np.asarray(render_body.get_triangles(mesh_id), dtype=np.int64)
            render_vertices.append(vertices)
            render_triangles.append(triangles + vertex_offset)
            vertex_offset += len(vertices)

        vertices = np.concatenate(render_vertices, axis=0)
        triangles = np.concatenate(render_triangles, axis=0)
        initial_transform = np.asarray(initial_transform, dtype=np.float32).reshape(
            4, 4
        )
        vertices = vertices @ initial_transform[:3, :3].T + initial_transform[:3, 3]
        distances, cloth_vertex_ids = cKDTree(rest_vertices).query(vertices)
        scale = max(float(np.ptp(rest_vertices, axis=0).max()), 1.0)
        if float(distances.max(initial=0.0)) > scale * 1.0e-5:
            raise RuntimeError(
                "Could not map surface-deformable render vertices onto the "
                "physical vertex buffer."
            )
        return np.asarray(cloth_vertex_ids[triangles], dtype=np.int32)

    def _apply_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int],
        arena_offsets: torch.Tensor,
    ) -> None:
        self._require_data()
        rest_vertices = self.body_data.rest_vertices
        for i, env_idx in enumerate(env_ids):
            cloth_body: ClothBody = self._entities[env_idx].get_physical_body()
            initial_transform = torch.as_tensor(
                cloth_body.get_initial_transform(),
                dtype=torch.float32,
                device=self.device,
            )
            rest_vertices_local = (
                rest_vertices[env_idx] - initial_transform[:3, 3]
            ) @ initial_transform[:3, :3]
            rotation = pose[i, :3, :3]
            translation = pose[i, :3, 3]
            arena_offset = torch.as_tensor(
                arena_offsets[env_idx], dtype=torch.float32, device=self.device
            )
            transformed_vertices = (
                rest_vertices_local @ rotation.T + translation + arena_offset
            )

            cloth_body.get_position_inv_mass_buffer()[:, :3] = transformed_vertices
            cloth_body.get_velocity_buffer()[:, :3] = 0.0
            cloth_body.mark_dirty(ClothBodyGPUAPIReadWriteType.ALL)
            cloth_body.set_wake_counter(0.4)

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
