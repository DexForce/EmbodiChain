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
import warp as wp
from dexsim.scene import Scene, SpawnedSoftBodyParticleSet
from scipy.spatial import ConvexHull, QhullError

from embodichain.utils import logger

from .base import DeformableObject
from .data import _ParticleSetData

__all__ = [
    "SoftBodyData",
    "SoftObject",
    "VolumeDeformableData",
    "VolumeDeformableObject",
]


class VolumeDeformableData(_ParticleSetData):
    """DexSim soft-body buffers exposed through the common nodal contract."""

    def __init__(
        self,
        entities: Sequence[SpawnedSoftBodyParticleSet],
        scene: Scene,
        device: torch.device,
    ) -> None:
        super().__init__(entities, scene, device)
        self.n_sim_vertices = self.num_nodes
        collision_counts = tuple(entity.collision_particle_count for entity in entities)
        if len(set(collision_counts)) != 1:
            raise ValueError(
                "A volume-deformable batch requires each SpawnedSoftBodyParticleSet "
                f"to have the same collision-node count, got {collision_counts!r}."
            )
        self.n_collision_vertices = collision_counts[0]
        self._rest_collision_vertices = self._fetch_collision_positions(rest=True)
        self._collision_position = torch.empty_like(self._rest_collision_vertices)

    def _fetch_collision_positions(self, *, rest: bool) -> torch.Tensor:
        """Fetch collision nodes exposed by typed Spawn soft-body handles."""
        getter = (
            SpawnedSoftBodyParticleSet.get_rest_positions
            if rest
            else SpawnedSoftBodyParticleSet.get_collision_positions
        )
        values = [
            wp.to_torch(getter(entity)).to(
                dtype=torch.float32,
                device=self.device,
            )
            for entity in self.entities
        ]
        return torch.stack(values, dim=0)

    @property
    def rest_collision_vertices(self) -> torch.Tensor:
        """Return rest collision vertices in simulation world frame."""
        return self._rest_collision_vertices.clone()

    @property
    def rest_sim_vertices(self) -> torch.Tensor:
        """Return rest simulation vertices in simulation world frame."""
        return self.default_nodal_state_w[..., :3]

    @property
    def collision_position(self) -> torch.Tensor:
        """Return current collision vertices in simulation world frame."""
        self._collision_position.copy_(self._fetch_collision_positions(rest=False))
        return self._collision_position.clone()

    @property
    def sim_vertex_position(self) -> torch.Tensor:
        """Return current simulation vertices in simulation world frame."""
        return self.nodal_pos_w

    @property
    def sim_vertex_velocity(self) -> torch.Tensor:
        """Return current simulation-vertex velocities."""
        return self.nodal_vel_w

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
        scene: Scene,
        device: torch.device,
    ) -> VolumeDeformableData:
        return VolumeDeformableData(entities, scene, device)

    @property
    def body_data(self) -> VolumeDeformableData | None:
        """Compatibility view of the DexSim soft-body data."""
        return self._data

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
