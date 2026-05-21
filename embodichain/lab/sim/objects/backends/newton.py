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

import numpy as np
import torch
import warp as wp

from dexsim.models import MeshObject
from embodichain.utils import logger

_UINT64_MAX = (1 << 64) - 1
_INT32_MAX = (1 << 31) - 1


def newton_rigid_data_type(name: str):
    from dexsim.engine.newton_physics.newton_physics_scene import NewtonRigidDataType

    return getattr(NewtonRigidDataType, name)


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


class NewtonRigidBodyView:
    """Thin adapter around DexSim Newton rigid body scene APIs.

    EmbodiChain public rigid-body pose convention is
    ``(x, y, z, qx, qy, qz, qw)``.
    DexSim Newton exposes the same pose convention through its unified rigid
    data API.
    """

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
        self.body_ids = [self._resolve_body_id(entity) for entity in self.entities]
        if any(body_id < 0 or body_id > _INT32_MAX for body_id in self.body_ids):
            logger.log_error(
                "Newton rigid body view found an entity without a Newton body id."
            )
        self.body_ids_tensor = torch.as_tensor(
            self.body_ids, dtype=torch.int32, device=self.device
        )

    @property
    def is_ready(self) -> bool:
        manager = getattr(self.scene, "manager", None)
        return (
            manager is not None
            and getattr(getattr(manager, "lifecycle_state", None), "name", "")
            == "READY"
        )

    def select_body_ids(self, indices: Sequence[int] | torch.Tensor) -> list[int]:
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        return [self.body_ids[int(index)] for index in indices]

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

    def fetch_pose(self, body_ids: Sequence[int] | None = None) -> torch.Tensor:
        body_ids = self.body_ids if body_ids is None else list(body_ids)
        out = self._empty_warp((len(body_ids), 7))
        self.scene.gpu_fetch_rigid_body_data(
            body_ids,
            newton_rigid_data_type("POSE"),
            out,
        )
        return self._warp_to_torch(out)

    def apply_pose(self, pose: torch.Tensor, body_ids: Sequence[int]) -> None:
        pose = pose.to(dtype=torch.float32)
        self.scene.gpu_apply_rigid_body_data(
            list(body_ids),
            newton_rigid_data_type("POSE"),
            self._to_numpy(pose),
        )

    def fetch_vec3(
        self, data_type, body_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        body_ids = self.body_ids if body_ids is None else list(body_ids)
        out = self._empty_warp((len(body_ids), 3))
        self.scene.gpu_fetch_rigid_body_data(body_ids, data_type, out)
        return self._warp_to_torch(out)

    def apply_vec3(
        self, data_type, data: torch.Tensor, body_ids: Sequence[int]
    ) -> None:
        self.scene.gpu_apply_rigid_body_data(
            list(body_ids),
            data_type,
            self._to_numpy(data.to(dtype=torch.float32)),
        )

    def apply_force(
        self, data_type, data: torch.Tensor, body_ids: Sequence[int]
    ) -> None:
        self.scene.gpu_apply_rigid_body_data(
            list(body_ids),
            data_type,
            data.to(dtype=torch.float32, device=self.device),
        )

    def _empty_warp(self, shape: tuple[int, int]):
        manager = self.scene.manager
        state = getattr(manager, "_state_0", None)
        warp_device = state.body_q.device if state is not None else manager._device
        return wp.empty(shape, dtype=wp.float32, device=warp_device)

    def _warp_to_torch(self, array) -> torch.Tensor:
        if str(array.device).startswith("cuda"):
            return wp.to_torch(array).to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(array.numpy(), dtype=torch.float32, device=self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)
