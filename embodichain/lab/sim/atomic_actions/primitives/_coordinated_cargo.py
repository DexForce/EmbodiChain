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

"""Observed cargo bounds for preference among qualified paired grasps."""

from __future__ import annotations

import torch

from embodichain.compute.geometry.cargo import inside_carrier_envelope
from embodichain.lab.sim.scene.mesh_snapshot import SceneMeshSnapshot
from embodichain.utils import logger
from ..state import PlanningContext

__all__: list[str] = []


def cargo_preference_bounds(
    context: PlanningContext,
    meshes: SceneMeshSnapshot | None,
    carrier_id: str | None,
) -> torch.Tensor | None:
    """Capture observed contents; unavailability retains the sampler's base costs."""
    bounds = []
    if (
        meshes
        and carrier_id in meshes
        and context.batch_size == 1
        and context.env_ids.tolist() == [0]
    ):
        try:

            def pose(uid: str) -> torch.Tensor:
                entity = context.scene.entities[uid]
                if getattr(entity, "confidence", 1) <= 0:
                    raise ValueError("Cargo preference requires an observed pose.")
                value = torch.as_tensor(entity.pose)
                return value[None] if value.ndim == 2 else value

            carrier = pose(carrier_id)
            source = meshes[carrier_id].to(carrier)
            limits = torch.stack((source[0].amin(0), source[0].amax(0)))
            for uid, vertices in meshes.items():
                if uid == carrier_id:
                    continue
                observed = pose(uid).to(carrier)
                vertices = vertices.to(carrier)
                if bool(
                    inside_carrier_envelope(vertices, observed, carrier, limits).all()
                ):
                    world = vertices @ observed[:, :3, :3].transpose(-1, -2)
                    world = world + observed[:, None, :3, 3]
                    bounds.append(
                        torch.stack(
                            (world[0].amin(0).clone(), world[0].amax(0).clone())
                        )
                    )
        except (ValueError, RuntimeError, KeyError, TypeError, AttributeError) as error:
            logger.log_warning(
                f"Cargo preference unavailable; retaining base costs: {error}"
            )
            bounds = []
    return torch.stack(bounds) if bounds else None
