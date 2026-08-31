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

"""Newton volume-deformable object implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from .base import DeformableObject
from .data import _ParticleSetData

if TYPE_CHECKING:
    from dexsim.scene import Scene, SpawnedSoftBodyParticleSet

__all__ = [
    "SoftBodyData",
    "SoftObject",
    "VolumeDeformableData",
    "VolumeDeformableObject",
]


class VolumeDeformableData(_ParticleSetData):
    """Newton soft-body particles exposed through the legacy volume API."""

    @property
    def particle_sets(self) -> list[SpawnedSoftBodyParticleSet]:
        """Return the typed DexSim soft-body particle handles."""
        return self.entities

    @property
    def n_collision_vertices(self) -> int:
        """Return the Newton collision-particle count per instance."""
        return self.n_nodes

    @property
    def n_sim_vertices(self) -> int:
        """Return the Newton simulation-particle count per instance."""
        return self.n_nodes

    @property
    def rest_collision_vertices(self) -> torch.Tensor:
        """Return particle positions captured when Spawn was bound."""
        return self.default_nodal_state_w[..., :3]

    @property
    def rest_sim_vertices(self) -> torch.Tensor:
        """Return particle positions captured when Spawn was bound."""
        return self.default_nodal_state_w[..., :3]

    @property
    def collision_position(self) -> torch.Tensor:
        """Return current Newton collision-particle positions."""
        return self.nodal_pos_w

    @property
    def sim_vertex_position(self) -> torch.Tensor:
        """Return current Newton simulation-particle positions."""
        return self.nodal_pos_w

    @property
    def sim_vertex_velocity(self) -> torch.Tensor:
        """Return current Newton simulation-particle velocities."""
        return self.nodal_vel_w


class VolumeDeformableObject(DeformableObject):
    """A batch of Newton volumetric soft-body particle sets."""

    deformable_type = "volume"
    spawn_kind = "soft_object"
    display_name = "volume deformable"

    def _create_data(
        self,
        entities: Sequence[SpawnedSoftBodyParticleSet],
        scene: Scene,
        device: torch.device,
    ) -> VolumeDeformableData:
        return VolumeDeformableData(entities, scene, device)

    def _initialize_topology(
        self,
        entities: Sequence[SpawnedSoftBodyParticleSet],
    ) -> None:
        super()._initialize_topology(entities)
        triangles = [
            np.asarray(entity.get_surface_triangles(), dtype=np.int32).reshape(-1, 3)
            for entity in entities
        ]
        triangle_counts = {len(item) for item in triangles}
        if len(triangle_counts) != 1:
            raise ValueError(
                "All instances of one soft body must share surface triangle "
                f"count, got {sorted(triangle_counts)}."
            )
        self._collision_surface_triangles = torch.as_tensor(
            np.stack(triangles),
            dtype=torch.int32,
            device=self.device,
        ).clone()

    @property
    def body_data(self) -> VolumeDeformableData | None:
        """Compatibility view of the Newton soft-body particle data."""
        return self._data

    def get_rest_collision_vertices(self) -> torch.Tensor:
        """Return particle positions captured when Spawn was bound."""
        self._require_data()
        return self.body_data.rest_collision_vertices

    def get_rest_sim_vertices(self) -> torch.Tensor:
        """Return particle positions captured when Spawn was bound."""
        self._require_data()
        return self.body_data.rest_sim_vertices

    def get_current_collision_vertices(self) -> torch.Tensor:
        """Return current Newton collision-particle positions."""
        self._require_data()
        return self.body_data.collision_position

    def get_current_sim_vertices(self) -> torch.Tensor:
        """Return current Newton simulation-particle positions."""
        return self.get_current_nodal_position()

    def get_current_sim_vertex_velocities(self) -> torch.Tensor:
        """Return current Newton simulation-particle velocities."""
        return self.get_current_nodal_velocity()

    def get_collision_surface_triangles(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return the tetrahedral surface topology for selected instances."""
        ids = self._resolve_env_ids(env_ids)
        index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        return self._collision_surface_triangles.index_select(0, index).clone()


# Compatibility names retained for existing environments and tutorials.
SoftBodyData = VolumeDeformableData
SoftObject = VolumeDeformableObject
