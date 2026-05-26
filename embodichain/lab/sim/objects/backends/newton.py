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

from typing import Sequence
from functools import cached_property

import numpy as np
import torch
import warp as wp

from dexsim.models import MeshObject
from embodichain.lab.sim.objects.backends.base import RigidBodyViewBase
from embodichain.utils import logger

__all__ = ["NewtonRigidBodyView", "is_newton_scene"]

_UINT64_MAX = (1 << 64) - 1
_INT32_MAX = (1 << 31) - 1


def _normalize_native_handle(handle: int, owner: str) -> int:
    value = int(handle)
    if value < 0:
        value &= _UINT64_MAX
    if value > _UINT64_MAX:
        logger.log_error(f"{owner} native handle is outside uint64 range: {value}.")
    return value


def is_newton_scene(scene: object) -> bool:
    """Return whether *scene* looks like a DexSim Newton scene view."""
    return (
        scene is not None
        and hasattr(scene, "manager")
        and hasattr(scene, "gpu_fetch_rigid_body_data")
        and hasattr(scene, "gpu_apply_rigid_body_data")
    )


class NewtonRigidBodyView(RigidBodyViewBase):
    """Adapter around DexSim Newton rigid body scene APIs.

    EmbodiChain public rigid-body pose convention is
    ``(x, y, z, qx, qy, qz, qw)``.
    DexSim Newton exposes the same pose convention through its unified rigid
    data API.
    """

    _DATA_TYPE = None  # lazily resolved NewtonRigidDataType

    def __init__(
        self,
        entities: Sequence[MeshObject],
        scene: object,
        device: torch.device,
    ) -> None:
        self.entities = list(entities)
        self.scene = scene
        self.device = device
        self.entity_handles = [
            _normalize_native_handle(entity.get_native_handle(), "MeshObject")
            for entity in self.entities
        ]
        self._body_ids = [self._resolve_body_id(entity) for entity in self.entities]
        if any(bid < 0 or bid > _INT32_MAX for bid in self._body_ids):
            logger.log_error(
                "Newton rigid body view found an entity without a Newton body id."
            )
        self._body_ids_tensor = torch.as_tensor(
            self._body_ids, dtype=torch.int32, device=self.device
        )

    # -- Lazy enum access ---------------------------------------------------

    @classmethod
    def _get_data_type(cls):
        """Lazily resolve *NewtonRigidDataType* to avoid eager import."""
        if cls._DATA_TYPE is None:
            from dexsim.engine.newton_physics import NewtonRigidDataType

            cls._DATA_TYPE = NewtonRigidDataType
        return cls._DATA_TYPE

    # -- RigidBodyViewBase: lifecycle ----------------------------------------

    @property
    def is_ready(self) -> bool:
        manager = getattr(self.scene, "manager", None)
        return (
            manager is not None
            and getattr(getattr(manager, "lifecycle_state", None), "name", "")
            == "READY"
        )

    # -- RigidBodyViewBase: body IDs -----------------------------------------

    @cached_property
    def body_ids(self) -> list[int]:
        return self._body_ids

    @cached_property
    def body_ids_tensor(self) -> torch.Tensor:
        return self._body_ids_tensor

    def select_body_ids(self, indices: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self._body_ids_tensor[indices.to(device=self.device, dtype=torch.long)]

    # -- RigidBodyViewBase: pose ---------------------------------------------

    def fetch_pose(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        body_ids = self._body_id_list(body_ids)
        out = self._as_warp_array(data)
        self.scene.gpu_fetch_rigid_body_data(out, body_ids, self._get_data_type().POSE)

    def apply_pose(self, pose: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().POSE, pose)

    # -- RigidBodyViewBase: velocity -----------------------------------------

    def fetch_linear_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(self._get_data_type().LINEAR_VELOCITY, data, body_ids)

    def fetch_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(self._get_data_type().ANGULAR_VELOCITY, data, body_ids)

    def apply_linear_velocity(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().LINEAR_VELOCITY, data)

    def apply_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor
    ) -> None:
        self._apply_data(body_ids, self._get_data_type().ANGULAR_VELOCITY, data)

    # -- RigidBodyViewBase: acceleration -------------------------------------

    def fetch_linear_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(self._get_data_type().LINEAR_ACCELERATION, data, body_ids)

    def fetch_angular_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(self._get_data_type().ANGULAR_ACCELERATION, data, body_ids)

    # -- RigidBodyViewBase: force & torque -----------------------------------

    def apply_force(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().FORCE, data)

    def apply_torque(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().TORQUE, data)

    # -- Internal helpers ----------------------------------------------------

    def _resolve_body_id(self, entity: MeshObject) -> int:
        manager = getattr(self.scene, "manager", None)
        if manager is not None and hasattr(entity, "get_native_handle"):
            entity_handle = _normalize_native_handle(
                entity.get_native_handle(), "MeshObject"
            )
            body_id = getattr(manager, "dexsim2newton_body", {}).get(entity_handle)
            if body_id is not None:
                return int(body_id)

        if hasattr(entity, "get_gpu_index"):
            body_id = int(entity.get_gpu_index())
            if 0 <= body_id <= _INT32_MAX:
                return body_id
        return -1

    def _body_id_list(self, body_ids: torch.Tensor | None = None) -> list[int]:
        """Return body IDs as a Python list for the Newton scene API."""
        if body_ids is None:
            return self._body_ids
        body_ids = body_ids.detach().cpu().tolist()
        return [int(body_id) for body_id in body_ids]

    def _as_warp_array(self, data: torch.Tensor):
        """Wrap a caller-owned torch tensor as a Warp float32 array."""
        if not data.is_contiguous():
            logger.log_error("Newton rigid body fetch buffers must be contiguous.")
        return wp.from_torch(data, dtype=wp.float32)

    def _fetch_vec3(
        self,
        data_type,
        data: torch.Tensor,
        body_ids: torch.Tensor | None = None,
    ) -> None:
        body_ids = self._body_id_list(body_ids)
        out = self._as_warp_array(data)
        self.scene.gpu_fetch_rigid_body_data(out, body_ids, data_type)

    def _apply_data(
        self, body_ids: torch.Tensor, data_type, data: torch.Tensor
    ) -> None:
        """Apply data to bodies via the unified Newton GPU API."""
        data = data.to(dtype=torch.float32)
        state = getattr(self.scene.manager, "_state_0", None)
        is_cuda = state is not None and str(state.body_q.device).startswith("cuda")
        payload = data if is_cuda else data.detach().cpu().numpy()
        self.scene.gpu_apply_rigid_body_data(
            payload, body_ids.detach().cpu().tolist(), data_type
        )
