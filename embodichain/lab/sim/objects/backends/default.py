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

import torch

from dexsim.models import MeshObject
from dexsim.types import RigidBodyGPUAPIReadType, RigidBodyGPUAPIWriteType
from embodichain.lab.sim.objects.backends.base import RigidBodyViewBase
from embodichain.utils.math import convert_quat, matrix_from_quat

__all__ = ["DefaultRigidBodyView"]


class DefaultRigidBodyView(RigidBodyViewBase):
    """Default DexSim backend rigid body data adapter.

    Encapsulates both GPU (PhysX) and CPU entity-level data paths.
    The default GPU API stores pose as ``(qx, qy, qz, qw, x, y, z)``; this
    adapter converts to / from the EmbodiChain convention
    ``(x, y, z, qx, qy, qz, qw)`` transparently.
    """

    def __init__(
        self,
        entities: Sequence[MeshObject],
        ps: object,
        device: torch.device,
    ) -> None:
        self.entities = list(entities)
        self.ps = ps
        self.device = device
        self._is_gpu = device.type == "cuda"

        if self._is_gpu:
            self._gpu_indices = torch.as_tensor(
                [entity.get_gpu_index() for entity in self.entities],
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self._gpu_indices = None

    # -- RigidBodyViewBase: lifecycle ----------------------------------------

    @property
    def is_ready(self) -> bool:
        return True

    # -- RigidBodyViewBase: body IDs -----------------------------------------

    @cached_property
    def body_ids(self) -> list[int]:
        if self._is_gpu:
            return self._gpu_indices.cpu().tolist()
        return list(range(len(self.entities)))

    @cached_property
    def body_ids_tensor(self) -> torch.Tensor:
        if self._is_gpu:
            return self._gpu_indices
        return torch.arange(len(self.entities), dtype=torch.int32, device=self.device)

    def select_body_ids(self, indices: Sequence[int] | torch.Tensor) -> torch.Tensor:
        return self.body_ids_tensor[indices]

    # -- RigidBodyViewBase: pose ---------------------------------------------

    def fetch_pose(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        if self._is_gpu:
            indices = self.body_ids_tensor if body_ids is None else body_ids
            self.ps.gpu_fetch_rigid_body_data(
                data=data,
                gpu_indices=indices.to(device=self.device, dtype=torch.int32),
                data_type=RigidBodyGPUAPIReadType.POSE,
            )
            # Convert (qx, qy, qz, qw, x, y, z) -> (x, y, z, qx, qy, qz, qw)
            quat = data[:, :4].clone()
            xyz = data[:, 4:7].clone()
            data[:, :3] = xyz
            data[:, 3:7] = quat
            return

        entities = self._select_entities(body_ids)
        data_np = data.cpu().numpy()
        for i, entity in enumerate(entities):
            data_np[i, :3] = entity.get_location()
            data_np[i, 3:7] = entity.get_rotation_quat()

    def apply_pose(self, pose: torch.Tensor, body_ids: torch.Tensor) -> None:
        pose = pose.to(dtype=torch.float32)
        if self._is_gpu:
            # Convert (x, y, z, qx, qy, qz, qw) -> (qx, qy, qz, qw, x, y, z)
            xyz = pose[:, :3]
            quat = pose[:, 3:7]
            gpu_pose = torch.cat((quat, xyz), dim=-1)
            torch.cuda.synchronize(self.device)
            self.ps.gpu_apply_rigid_body_data(
                data=gpu_pose.clone(),
                gpu_indices=body_ids.to(device=self.device, dtype=torch.int32),
                data_type=RigidBodyGPUAPIWriteType.POSE,
            )
            return

        # CPU: convert (x, y, z, qx, qy, qz, qw) -> 4x4 matrix per entity
        indices = body_ids.detach().cpu().tolist()
        pose_cpu = pose.cpu()
        mat = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(len(indices), 1, 1)
        mat[:, :3, 3] = pose_cpu[:, :3]
        mat[:, :3, :3] = matrix_from_quat(convert_quat(pose_cpu[:, 3:7], to="wxyz"))
        for i, idx in enumerate(indices):
            self.entities[idx].set_local_pose(mat[i])

    # -- RigidBodyViewBase: velocity -----------------------------------------

    def fetch_linear_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(
            RigidBodyGPUAPIReadType.LINEAR_VELOCITY,
            "get_linear_velocity",
            data,
            body_ids,
        )

    def fetch_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(
            RigidBodyGPUAPIReadType.ANGULAR_VELOCITY,
            "get_angular_velocity",
            data,
            body_ids,
        )

    def apply_linear_velocity(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_vec3(
            RigidBodyGPUAPIWriteType.LINEAR_VELOCITY,
            "set_linear_velocity",
            data,
            body_ids,
        )

    def apply_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor
    ) -> None:
        self._apply_vec3(
            RigidBodyGPUAPIWriteType.ANGULAR_VELOCITY,
            "set_angular_velocity",
            data,
            body_ids,
        )

    # -- RigidBodyViewBase: acceleration -------------------------------------

    def fetch_linear_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(
            RigidBodyGPUAPIReadType.LINEAR_ACCELERATION,
            "get_linear_acceleration",
            data,
            body_ids,
        )

    def fetch_angular_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(
            RigidBodyGPUAPIReadType.ANGULAR_ACCELERATION,
            "get_angular_acceleration",
            data,
            body_ids,
        )

    # -- RigidBodyViewBase: force & torque -----------------------------------

    def apply_force(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_vec3(
            RigidBodyGPUAPIWriteType.FORCE,
            "add_force",
            data,
            body_ids,
        )

    def apply_torque(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_vec3(
            RigidBodyGPUAPIWriteType.TORQUE,
            "add_torque",
            data,
            body_ids,
        )

    # -- Internal helpers ----------------------------------------------------

    def _select_entities(self, body_ids: torch.Tensor | None) -> list[MeshObject]:
        """Select entities by body IDs (entity list indices for CPU)."""
        if body_ids is None:
            return self.entities
        body_ids = body_ids.detach().cpu().tolist()
        return [self.entities[int(i)] for i in body_ids]

    def _fetch_vec3(
        self,
        gpu_read_type,
        cpu_method: str,
        data: torch.Tensor,
        body_ids: torch.Tensor | None,
    ) -> None:
        """Fetch a vec3 field from GPU or CPU entities."""
        if self._is_gpu:
            indices = self.body_ids_tensor if body_ids is None else body_ids
            self.ps.gpu_fetch_rigid_body_data(
                data=data,
                gpu_indices=indices.to(device=self.device, dtype=torch.int32),
                data_type=gpu_read_type,
            )
            return

        entities = self._select_entities(body_ids)
        data_np = data.cpu().numpy()
        for i, entity in enumerate(entities):
            data_np[i] = getattr(entity, cpu_method)()

    def _apply_vec3(
        self,
        gpu_write_type,
        cpu_method: str,
        data: torch.Tensor,
        body_ids: torch.Tensor,
    ) -> None:
        """Apply a vec3 field to GPU or CPU entities."""
        data = data.to(dtype=torch.float32)
        if self._is_gpu:
            torch.cuda.synchronize(self.device)
            self.ps.gpu_apply_rigid_body_data(
                data=data,
                gpu_indices=body_ids.to(device=self.device, dtype=torch.int32),
                data_type=gpu_write_type,
            )
            return

        indices = body_ids.detach().cpu().tolist()
        data_cpu = data.cpu().numpy()
        for i, idx in enumerate(indices):
            getattr(self.entities[idx], cpu_method)(data_cpu[i])
