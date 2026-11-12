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
"""One-time conservative cylinder variants for native physical experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from dexsim.engine.newton_physics.runtime_control.contact_material import (
    resolve_contact_shape_ids,
)
from newton import GeoType, ModelFlags

from embodichain.utils.device_utils import current_warp_stream

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__: list[str] = []


@torch.no_grad()
def _shrink_cylinder_variants(
    manager: SimulationManager, uid: str, ratios: torch.Tensor
) -> None:
    """Shrink one native cylinder per Arena, preserving its constant-mass inertia.

    Call once after preparation and before rollout. Ratios must be positive
    and at most one: cached MuJoCo bounding spheres/AABBs retain the enclosing
    template bounds. Narrow-phase geometry and native inertia use actual sizes.
    Expansion, meshes and multiple colliders are rejected. This operation does
    not update render geometry, source descriptors or reset defaults.
    """
    result = manager.spawn_result
    if result is None or not manager.is_newton_backend:
        raise RuntimeError("Cylinder variants require prepared Newton physics")
    backend = get_newton_backend(result.world)
    if backend is None or backend.solver_type != "mujoco_warp":
        raise ValueError("Cylinder variants currently qualify MuJoCo-Warp only")
    model = backend.model
    device = torch.device(str(model.device))
    if (
        ratios.device != device
        or ratios.dtype != torch.float32
        or ratios.shape != (manager.num_envs,)
    ):
        raise ValueError(
            "Expected one float32 cylinder ratio per Arena on the scene device"
        )
    if not bool((torch.isfinite(ratios) & (ratios > 0) & (ratios <= 1)).all()):
        raise ValueError("Cylinder ratios must be finite, positive and at most one")
    arenas = tuple(n for n in result.arenas.names if n != "default")
    if len(arenas) != manager.num_envs:
        raise ValueError("Expected one prepared Arena per environment")
    shapes = []
    for arena in arenas:
        owned = resolve_contact_shape_ids(result, f"{arena}/{uid}")
        if len(owned) != 1:
            raise ValueError("Cylinder variants require one source collider per Arena")
        shapes.append(int(owned[0]))
    shape_ids = torch.tensor(shapes, device=device, dtype=torch.long)
    types = wp.to_torch(model.shape_type)[shape_ids]
    bodies = wp.to_torch(model.shape_body)[shape_ids].long()
    if (
        not bool((types == int(GeoType.CYLINDER)).all())
        or bool((bodies < 0).any())
        or bodies.unique().numel() != len(bodies)
    ):
        raise ValueError(
            "Expected distinct dynamic bodies with analytic cylinder colliders"
        )
    with wp.ScopedStream(current_warp_stream(device)):
        sizes = wp.to_torch(model.shape_scale)
        inertia = wp.to_torch(model.body_inertia)
        inverse = wp.to_torch(model.body_inv_inertia)
        sizes[shape_ids] = sizes[shape_ids] * ratios[:, None]
        square = ratios[:, None, None].square()
        inertia[bodies] = inertia[bodies] * square
        inverse[bodies] = inverse[bodies] / square
        backend._runtime.notify_model_changed(
            ModelFlags.SHAPE_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES,
            reason="one-time conservative cylinder variants",
        )
