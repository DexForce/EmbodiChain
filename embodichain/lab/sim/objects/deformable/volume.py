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

"""DexSim volume-deformable object implementation."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Sequence

import numpy as np
import torch
from dexsim.engine import PhysicsScene, SoftBody
from dexsim.models import MeshObject
from dexsim.types import SoftBodyGPUAPIReadWriteType
from scipy.spatial import ConvexHull, QhullError

from embodichain.utils import logger

from .base import DeformableObject
from .data import DeformableObjectData

__all__ = [
    "SoftBodyData",
    "SoftObject",
    "VolumeDeformableData",
    "VolumeDeformableObject",
]


class VolumeDeformableData(DeformableObjectData):
    """DexSim soft-body buffers exposed through the common nodal contract."""

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
        self.soft_bodies: Sequence[SoftBody] = [
            entity.get_physical_body() for entity in self.entities
        ]
        self.n_collision_vertices = self.soft_bodies[0].get_num_vertices()
        self.n_sim_vertices = self.soft_bodies[0].get_num_sim_vertices()

        self._rest_position_buffer = torch.empty(
            (self.num_instances, self.n_collision_vertices, 4),
            device=self.device,
            dtype=torch.float32,
        )
        self._rest_sim_position_buffer = torch.empty(
            (self.num_instances, self.n_sim_vertices, 4),
            device=self.device,
            dtype=torch.float32,
        )
        for i, soft_body in enumerate(self.soft_bodies):
            self._rest_position_buffer[i] = soft_body.get_position_inv_mass_buffer()
            self._rest_sim_position_buffer[i] = (
                soft_body.get_sim_position_inv_mass_buffer()
            )

        self._collision_position = torch.zeros(
            (self.num_instances, self.n_collision_vertices, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._sim_vertex_position = torch.zeros(
            (self.num_instances, self.n_sim_vertices, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._sim_vertex_velocity = torch.zeros_like(self._sim_vertex_position)
        self._default_nodal_state_w = torch.cat(
            (
                self._rest_sim_position_buffer[..., :3],
                torch.zeros_like(self._rest_sim_position_buffer[..., :3]),
            ),
            dim=-1,
        )

    @property
    def rest_collision_vertices(self) -> torch.Tensor:
        """Return rest collision vertices in simulation world frame."""
        return self._rest_position_buffer[..., :3].clone()

    @property
    def rest_sim_vertices(self) -> torch.Tensor:
        """Return rest simulation vertices in simulation world frame."""
        return self._rest_sim_position_buffer[..., :3].clone()

    @property
    def collision_position(self) -> torch.Tensor:
        """Return current collision vertices in simulation world frame."""
        for i, soft_body in enumerate(self.soft_bodies):
            self._collision_position[i] = soft_body.get_position_inv_mass_buffer()[
                :, :3
            ]
        return self._collision_position.clone()

    @property
    def sim_vertex_position(self) -> torch.Tensor:
        """Return current simulation vertices in simulation world frame."""
        for i, soft_body in enumerate(self.soft_bodies):
            self._sim_vertex_position[i] = soft_body.get_sim_position_inv_mass_buffer()[
                :, :3
            ]
        return self._sim_vertex_position.clone()

    @property
    def sim_vertex_velocity(self) -> torch.Tensor:
        """Return current simulation-vertex velocities."""
        for i, soft_body in enumerate(self.soft_bodies):
            self._sim_vertex_velocity[i] = soft_body.get_sim_velocity_buffer()[:, :3]
        return self._sim_vertex_velocity.clone()

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        return self.sim_vertex_position

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        return self.sim_vertex_velocity

    @property
    def default_nodal_state_w(self) -> torch.Tensor:
        return self._default_nodal_state_w.clone()

    @cached_property
    def collision_surface_triangles(self) -> torch.Tensor:
        """Return a stable convex-hull topology over collision vertices."""
        vertices = self.rest_collision_vertices[0].detach().cpu().numpy()
        if vertices.shape[0] < 4:
            logger.log_warning(
                "Volume-deformable collision geometry has fewer than four "
                "vertices; its visualization surface will be empty."
            )
            triangles = np.empty((0, 3), dtype=np.int32)
        else:
            try:
                triangles = np.asarray(ConvexHull(vertices).simplices, dtype=np.int32)
            except QhullError as error:
                try:
                    triangles = np.asarray(
                        ConvexHull(vertices, qhull_options="QJ").simplices,
                        dtype=np.int32,
                    )
                except QhullError:
                    logger.log_warning(
                        "Unable to build a volume-deformable visualization "
                        f"surface from collision vertices: {error!r}"
                    )
                    triangles = np.empty((0, 3), dtype=np.int32)
        return torch.as_tensor(triangles, dtype=torch.int32, device=self.device)


class VolumeDeformableObject(DeformableObject):
    """A batch of DexSim volume deformables backed by ``SoftBody``."""

    deformable_type = "volume"
    spawn_kind = "soft_object"
    display_name = "volume deformable"

    def _create_data(
        self,
        entities: Sequence[Any],
        physics_scene: PhysicsScene,
        device: torch.device,
    ) -> VolumeDeformableData:
        return VolumeDeformableData(entities, physics_scene, device)

    @property
    def body_data(self) -> VolumeDeformableData | None:
        """Compatibility view of the DexSim soft-body data."""
        return self._data

    def _apply_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int],
        arena_offsets: torch.Tensor,
    ) -> None:
        self._require_data()
        rest_collision_vertices = self.body_data.rest_collision_vertices
        rest_sim_vertices = self.body_data.rest_sim_vertices
        for i, env_idx in enumerate(env_ids):
            soft_body: SoftBody = self._entities[env_idx].get_physical_body()
            initial_transform = torch.as_tensor(
                soft_body.get_initial_transform(),
                dtype=torch.float32,
                device=self.device,
            )
            initial_rotation = initial_transform[:3, :3]
            initial_translation = initial_transform[:3, 3]
            rest_collision_local = (
                rest_collision_vertices[env_idx] - initial_translation
            ) @ initial_rotation
            rest_sim_local = (
                rest_sim_vertices[env_idx] - initial_translation
            ) @ initial_rotation
            rotation = pose[i, :3, :3]
            translation = pose[i, :3, 3]
            arena_offset = torch.as_tensor(
                arena_offsets[env_idx], dtype=torch.float32, device=self.device
            )

            collision_positions = (
                rest_collision_local @ rotation.T + translation + arena_offset
            )
            sim_positions = rest_sim_local @ rotation.T + translation + arena_offset

            soft_body.get_position_inv_mass_buffer()[:, :3] = collision_positions
            soft_body.get_sim_position_inv_mass_buffer()[:, :3] = sim_positions
            soft_body.get_sim_velocity_buffer()[:, :3] = 0.0
            soft_body.mark_dirty(SoftBodyGPUAPIReadWriteType.ALL)
            soft_body.set_wake_counter(0.4)

    def get_rest_collision_vertices(self) -> torch.Tensor:
        """Return rest collision vertices."""
        self._require_data()
        return self.body_data.rest_collision_vertices

    def get_rest_sim_vertices(self) -> torch.Tensor:
        """Return rest simulation vertices."""
        self._require_data()
        return self.body_data.rest_sim_vertices

    def get_current_collision_vertices(self) -> torch.Tensor:
        """Return current collision vertices."""
        self._require_data()
        return self.body_data.collision_position

    def get_current_sim_vertices(self) -> torch.Tensor:
        """Return current simulation vertices."""
        return self.get_current_nodal_position()

    def get_current_sim_vertex_velocities(self) -> torch.Tensor:
        """Return current simulation-vertex velocities."""
        return self.get_current_nodal_velocity()

    def get_surface_vertices(self) -> torch.Tensor:
        """Return the live collision surface used for visualization."""
        return self.get_current_collision_vertices()

    def get_collision_surface_triangles(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return convex-hull triangles over collision vertices."""
        self._require_data()
        ids = self._resolve_env_ids(env_ids)
        return (
            self.body_data.collision_surface_triangles.unsqueeze(0)
            .expand(len(ids), -1, -1)
            .clone()
        )

    def get_surface_triangles(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return the volume deformable's collision-surface topology."""
        return self.get_collision_surface_triangles(env_ids=env_ids)


# Compatibility names retained for existing environments and tutorials.
SoftBodyData = VolumeDeformableData
SoftObject = VolumeDeformableObject
