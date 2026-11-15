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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from dexsim.engine.newton_physics.runtime_control.base import RuntimeControlContext
from dexsim.engine.newton_physics.runtime_control.contact_material import (
    resolve_contact_shape_ids,
)
from newton import Contacts

from embodichain.utils.device_utils import current_warp_stream

from ._warp.geometry_contact import _reduce_geometry_netforce

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__: list[str] = []


class _GeometryNetForceSensor:
    """Read native net contact force for named source geometries against an object.

    Native shape access and force extraction remain sensor-owned. Each selected
    source collider must produce one native convex shape; decomposed colliders
    are rejected rather than assigned by a guessed ordering.
    """

    def __init__(
        self,
        manager: SimulationManager,
        articulation_uid: str,
        object_uid: str,
        selections: tuple[tuple[str, str], ...],
    ) -> None:
        result = manager.spawn_result
        if result is None or not manager.is_newton_backend:
            raise RuntimeError(
                "Geometry force sensing requires a prepared Newton scene"
            )
        backend = get_newton_backend(result.world)
        if backend is None or backend.solver_type != "mujoco_warp":
            raise ValueError(
                "Geometry net forces currently qualify native MuJoCo-Warp only"
            )
        self._manager = manager
        self._result = result
        self._revision = result.topology_revision
        self._context = RuntimeControlContext(backend, result)
        self.device = torch.device(str(backend.model.device))
        self.num_envs = manager.num_envs
        self.width = len(selections)
        self._contacts = Contacts(
            backend.mujoco_solver.get_max_contact_count(),
            0,
            device=backend.model.device,
            requested_attributes={"force"},
        )
        slots = [-1] * backend.model.shape_count
        objects = [-1] * backend.model.shape_count
        arenas = tuple(n for n in result.arenas.names if n != "default")
        if len(arenas) != self.num_envs or not selections:
            raise ValueError(
                "Expected one prepared Arena per environment and nonempty selections"
            )
        for row, arena in enumerate(arenas):
            for sid in resolve_contact_shape_ids(result, f"{arena}/{object_uid}"):
                objects[int(sid)] = row
            for column, (link_name, geometry_name) in enumerate(selections):
                desc = result.get_link_desc(f"{arena}/{articulation_uid}", link_name)
                shapes = resolve_contact_shape_ids(
                    result, f"{arena}/{articulation_uid}", link_names=[link_name]
                )
                if len(shapes) != len(desc.collisions):
                    raise ValueError(
                        "Geometry force sensing requires one native shape per source collider"
                    )
                found = [
                    index
                    for index, collision in enumerate(desc.collisions)
                    if collision.usd is not None
                    and collision.usd.prim_path.rsplit("/", 1)[-1] == geometry_name
                ]
                if len(found) != 1:
                    raise ValueError(
                        f"Expected one exact source geometry {link_name}/{geometry_name}"
                    )
                sid = int(shapes[found[0]])
                if slots[sid] != -1:
                    raise ValueError("Tactile geometry selections must be distinct")
                slots[sid] = row * self.width + column
        self._slots = wp.array(slots, dtype=wp.int32, device=backend.model.device)
        self._objects = wp.array(objects, dtype=wp.int32, device=backend.model.device)
        self._forces = wp.zeros(
            self.num_envs * self.width, dtype=wp.vec3, device=backend.model.device
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear selected measurements without changing physical state."""
        view = wp.to_torch(self._forces).reshape(self.num_envs, self.width, 3)
        if env_ids is None:
            view.zero_()
        else:
            view[env_ids] = 0

    def measure(self) -> torch.Tensor:
        """Return force magnitudes after one completed physical control period."""
        if self._manager is None:
            raise RuntimeError("Geometry force sensor is closed")
        if (
            self._manager.spawn_result is not self._result
            or self._result.topology_revision != self._revision
        ):
            raise RuntimeError("Geometry force sensor is stale after topology change")
        context = self._context
        with wp.ScopedStream(current_warp_stream(self.device)):
            context.solver.update_contacts(self._contacts, context.current_state)
            self._forces.zero_()
            wp.launch(
                _reduce_geometry_netforce,
                dim=self._contacts.rigid_contact_max,
                inputs=[
                    self._contacts.rigid_contact_count,
                    self._contacts.rigid_contact_shape0,
                    self._contacts.rigid_contact_shape1,
                    self._contacts.force,
                    self._slots,
                    self._objects,
                    self.width,
                ],
                outputs=[self._forces],
                device=context.device,
            )
            return (
                wp.to_torch(self._forces)
                .reshape(self.num_envs, self.width, 3)
                .norm(dim=-1)
            )

    def close(self) -> None:
        """Release measurement buffers before the native manager is destroyed."""
        self._contacts = self._forces = self._slots = self._objects = None
        self._context = self._manager = self._result = None
