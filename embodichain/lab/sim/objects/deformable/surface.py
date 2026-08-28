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

"""Newton surface-deformable object implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from .base import DeformableObject
from .data import _ParticleSetData

if TYPE_CHECKING:
    from dexsim.scene import Scene, SpawnedClothParticleSet

__all__ = [
    "ClothBodyData",
    "ClothObject",
    "SurfaceDeformableData",
    "SurfaceDeformableObject",
]


class SurfaceDeformableData(_ParticleSetData):
    """Newton cloth particles exposed through the legacy surface API."""

    @property
    def particle_sets(self) -> list[SpawnedClothParticleSet]:
        """Return the typed DexSim cloth particle handles."""
        return self.entities

    @property
    def n_vertices(self) -> int:
        """Return the Newton cloth particle count per instance."""
        return self.n_nodes

    @property
    def rest_vertices(self) -> torch.Tensor:
        """Return particle positions captured when Spawn was bound."""
        return self.default_nodal_state_w[..., :3]

    @property
    def vertex_position(self) -> torch.Tensor:
        """Return current Newton cloth-particle positions."""
        return self.nodal_pos_w

    @property
    def vertex_velocity(self) -> torch.Tensor:
        """Return current Newton cloth-particle velocities."""
        return self.nodal_vel_w


class SurfaceDeformableObject(DeformableObject):
    """A batch of Newton cloth particle sets."""

    deformable_type = "surface"
    spawn_kind = "cloth_object"
    display_name = "surface deformable"

    def _create_data(
        self,
        entities: Sequence[SpawnedClothParticleSet],
        scene: Scene,
        device: torch.device,
    ) -> SurfaceDeformableData:
        return SurfaceDeformableData(entities, scene, device)

    @property
    def body_data(self) -> SurfaceDeformableData | None:
        """Compatibility view of the Newton cloth particle data."""
        return self._data

    def get_rest_vertex_position(self) -> torch.Tensor:
        """Return particle positions captured when Spawn was bound."""
        self._require_data()
        return self.body_data.rest_vertices

    def get_current_vertex_position(self) -> torch.Tensor:
        """Return current Newton cloth-particle positions."""
        return self.get_current_nodal_position()

    def get_current_vertex_velocity(self) -> torch.Tensor:
        """Return current Newton cloth-particle velocities."""
        return self.get_current_nodal_velocity()


# Compatibility names retained for existing environments and tutorials.
ClothBodyData = SurfaceDeformableData
ClothObject = SurfaceDeformableObject
