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

from embodichain.lab.sim.cfg.deformable import MaterialPointModelCfg

from .base import DeformableObject

if TYPE_CHECKING:
    from dexsim.scene import SpawnedParticleSet

__all__ = [
    "VolumeDeformableObject",
]


class VolumeDeformableObject(DeformableObject):
    """A batch of Newton volumes with tetrahedral or material-point models."""

    deformable_type = "volume"

    @property
    def has_fixed_surface_topology(self) -> bool:
        """Whether the volume has a tetrahedral model with surface triangles."""
        return not isinstance(self.cfg.model, MaterialPointModelCfg)

    def _initialize_topology(
        self,
        entities: Sequence[SpawnedParticleSet],
    ) -> None:
        if not self.has_fixed_surface_topology:
            self._surface_triangles = None
            self._collision_surface_triangles = None
            return
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

    def get_collision_surface_triangles(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return the tetrahedral surface topology for selected instances."""
        if not self.has_fixed_surface_topology:
            raise NotImplementedError(
                "Material-point samples have no tetrahedral boundary."
            )
        ids = self._resolve_env_ids(env_ids)
        index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        return self._collision_surface_triangles.index_select(0, index).clone()

    def get_surface_vertices(self) -> torch.Tensor:
        if not self.has_fixed_surface_topology:
            raise NotImplementedError(
                "Use data.nodal_pos_w for material-point geometry."
            )
        return super().get_surface_vertices()

    def get_surface_triangles(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        if not self.has_fixed_surface_topology:
            raise NotImplementedError(
                "Material-point reconstruction has no fixed triangle topology."
            )
        return super().get_surface_triangles(env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if not self.has_fixed_surface_topology:
            raise NotImplementedError(
                "Use the owning manager to reset the particle solver history."
            )
        super().reset(env_ids)
