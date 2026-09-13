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

if TYPE_CHECKING:
    from dexsim.scene import SpawnedSoftBodyParticleSet

__all__ = [
    "VolumeDeformableObject",
]


class VolumeDeformableObject(DeformableObject):
    """A batch of Newton volumetric soft-body particle sets."""

    deformable_type = "volume"

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

    def get_collision_surface_triangles(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return the tetrahedral surface topology for selected instances."""
        ids = self._resolve_env_ids(env_ids)
        index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        return self._collision_surface_triangles.index_select(0, index).clone()
