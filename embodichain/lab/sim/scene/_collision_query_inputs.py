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

"""Capture the exact collision-input route used by discrete free-motion queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc, RigidBodyShape
from .collision_sources import CollisionObjectSnapshot

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__: list[str] = []


def _capture_free_motion_sources(
    simulation: SimulationManager, *, include_rigids: bool
) -> dict[str, CollisionObjectSnapshot]:
    from embodichain.lab.sim.objects.backends._collision_inputs import (
        _usd_collision_meshes,
    )

    result = {}
    for uid in simulation.get_rigid_object_uid_list() if include_rigids else ():
        obj = simulation.get_rigid_object(uid)
        pose = obj.get_local_pose(to_matrix=True)[0:1].detach().cpu()
        for index, shape in enumerate(obj.get_collision_shapes()):
            result[f"{uid}:{index}"] = CollisionObjectSnapshot(
                (shape,), pose, provenance="rigid_collision_descriptors"
            )
    for uid in simulation.get_articulation_uid_list():
        art = simulation.get_articulation(uid)
        # This source route intentionally uses the original USD input meshes,
        # not retained convex descriptors or native cooked collision shapes.
        meshes = _usd_collision_meshes(
            {"fpath": art.cfg.fpath, "body_scale": art.cfg.body_scale}
        )
        for (link, name), mesh in meshes.items():
            shape = CollisionShapeDesc(
                name=name,
                shape_type=RigidBodyShape.MESH,
                local_pose=torch.eye(4, dtype=torch.float64),
                vertices=torch.from_numpy(np.array(mesh.vertices, copy=True)),
                triangles=torch.from_numpy(np.array(mesh.faces, copy=True)),
            )
            result[f"{uid}:{name}"] = CollisionObjectSnapshot(
                (shape,),
                art.get_link_pose(link, to_matrix=True)[0:1].detach().cpu(),
                provenance="articulation_usd_collision_input",
            )
    return result
