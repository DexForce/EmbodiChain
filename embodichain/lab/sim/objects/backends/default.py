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

    @property
    def body_ids(self) -> list[int]:
        if self._is_gpu:
            return self._gpu_indices.tolist()
        return list(range(len(self.entities)))

    @property
    def body_ids_tensor(self) -> torch.Tensor:
        if self._is_gpu:
            return self._gpu_indices
        return torch.arange(len(self.entities), dtype=torch.int32, device=self.device)

    def select_body_ids(self, indices: Sequence[int] | torch.Tensor) -> list[int]:
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        if self._is_gpu:
            return self._gpu_indices[list(int(i) for i in indices)].tolist()
        return [int(i) for i in indices]

    # -- RigidBodyViewBase: pose ---------------------------------------------

    def fetch_pose(self, body_ids: Sequence[int] | None = None) -> torch.Tensor:
        if self._is_gpu:
            indices = self._indices_tensor(body_ids)
            out = torch.zeros(
                (len(indices), 7), dtype=torch.float32, device=self.device
            )
            self.ps.gpu_fetch_rigid_body_data(
                data=out,
                gpu_indices=indices,
                data_type=RigidBodyGPUAPIReadType.POSE,
            )
            # Convert (qx, qy, qz, qw, x, y, z) -> (x, y, z, qx, qy, qz, qw)
            quat = out[:, :4].clone()
            xyz = out[:, 4:7].clone()
            out[:, :3] = xyz
            out[:, 3:7] = quat
            return out

        entities = self._select_entities(body_ids)
        xyzs = torch.as_tensor(
            np.array([e.get_location() for e in entities]),
            dtype=torch.float32,
            device=self.device,
        )
        quats = torch.as_tensor(
            np.array([e.get_rotation_quat() for e in entities]),
            dtype=torch.float32,
            device=self.device,
        )
        return torch.cat((xyzs, quats), dim=-1)

    def apply_pose(self, pose: torch.Tensor, body_ids: Sequence[int]) -> None:
        pose = pose.to(dtype=torch.float32)
        if self._is_gpu:
            # Convert (x, y, z, qx, qy, qz, qw) -> (qx, qy, qz, qw, x, y, z)
            xyz = pose[:, :3]
            quat = pose[:, 3:7]
            gpu_pose = torch.cat((quat, xyz), dim=-1)
            indices = self._indices_tensor(body_ids)
            torch.cuda.synchronize(self.device)
            self.ps.gpu_apply_rigid_body_data(
                data=gpu_pose.clone(),
                gpu_indices=indices,
                data_type=RigidBodyGPUAPIWriteType.POSE,
            )
            return

        # CPU: convert (x, y, z, qx, qy, qz, qw) -> 4x4 matrix per entity
        indices = list(body_ids)
        pose_cpu = pose.cpu()
        mat = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(len(indices), 1, 1)
        mat[:, :3, 3] = pose_cpu[:, :3]
        mat[:, :3, :3] = matrix_from_quat(convert_quat(pose_cpu[:, 3:7], to="wxyz"))
        for i, idx in enumerate(indices):
            self.entities[idx].set_local_pose(mat[i])

    # -- RigidBodyViewBase: velocity -----------------------------------------

    def fetch_linear_velocity(
        self, body_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        return self._fetch_vec3(
            RigidBodyGPUAPIReadType.LINEAR_VELOCITY,
            "get_linear_velocity",
            body_ids,
        )

    def fetch_angular_velocity(
        self, body_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        return self._fetch_vec3(
            RigidBodyGPUAPIReadType.ANGULAR_VELOCITY,
            "get_angular_velocity",
            body_ids,
        )

    def apply_linear_velocity(
        self, data: torch.Tensor, body_ids: Sequence[int]
    ) -> None:
        self._apply_vec3(
            RigidBodyGPUAPIWriteType.LINEAR_VELOCITY,
            "set_linear_velocity",
            data,
            body_ids,
        )

    def apply_angular_velocity(
        self, data: torch.Tensor, body_ids: Sequence[int]
    ) -> None:
        self._apply_vec3(
            RigidBodyGPUAPIWriteType.ANGULAR_VELOCITY,
            "set_angular_velocity",
            data,
            body_ids,
        )

    # -- RigidBodyViewBase: acceleration -------------------------------------

    def fetch_linear_acceleration(
        self, body_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        return self._fetch_vec3(
            RigidBodyGPUAPIReadType.LINEAR_ACCELERATION,
            "get_linear_acceleration",
            body_ids,
        )

    def fetch_angular_acceleration(
        self, body_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        return self._fetch_vec3(
            RigidBodyGPUAPIReadType.ANGULAR_ACCELERATION,
            "get_angular_acceleration",
            body_ids,
        )

    # -- RigidBodyViewBase: force & torque -----------------------------------

    def apply_force(self, data: torch.Tensor, body_ids: Sequence[int]) -> None:
        self._apply_vec3(
            RigidBodyGPUAPIWriteType.FORCE,
            "add_force",
            data,
            body_ids,
        )

    def apply_torque(self, data: torch.Tensor, body_ids: Sequence[int]) -> None:
        self._apply_vec3(
            RigidBodyGPUAPIWriteType.TORQUE,
            "add_torque",
            data,
            body_ids,
        )

    # -- Internal helpers ----------------------------------------------------

    def _indices_tensor(self, body_ids: Sequence[int] | None) -> torch.Tensor:
        """Return GPU indices as an int32 tensor on device."""
        if body_ids is None:
            return self._gpu_indices
        if isinstance(body_ids, torch.Tensor):
            return body_ids.to(device=self.device, dtype=torch.int32)
        return torch.as_tensor(body_ids, dtype=torch.int32, device=self.device)

    def _select_entities(self, body_ids: Sequence[int] | None) -> list[MeshObject]:
        """Select entities by body IDs (entity list indices for CPU)."""
        if body_ids is None:
            return self.entities
        return [self.entities[int(i)] for i in body_ids]

    def _fetch_vec3(
        self,
        gpu_read_type,
        cpu_method: str,
        body_ids: Sequence[int] | None,
    ) -> torch.Tensor:
        """Fetch a vec3 field from GPU or CPU entities."""
        if self._is_gpu:
            indices = self._indices_tensor(body_ids)
            out = torch.zeros(
                (len(indices), 3), dtype=torch.float32, device=self.device
            )
            self.ps.gpu_fetch_rigid_body_data(
                data=out, gpu_indices=indices, data_type=gpu_read_type
            )
            return out

        entities = self._select_entities(body_ids)
        return torch.as_tensor(
            np.array([getattr(e, cpu_method)() for e in entities]),
            dtype=torch.float32,
            device=self.device,
        )

    def _apply_vec3(
        self,
        gpu_write_type,
        cpu_method: str,
        data: torch.Tensor,
        body_ids: Sequence[int],
    ) -> None:
        """Apply a vec3 field to GPU or CPU entities."""
        data = data.to(dtype=torch.float32)
        if self._is_gpu:
            indices = self._indices_tensor(body_ids)
            torch.cuda.synchronize(self.device)
            self.ps.gpu_apply_rigid_body_data(
                data=data, gpu_indices=indices, data_type=gpu_write_type
            )
            return

        indices = list(body_ids)
        data_cpu = data.cpu().numpy()
        for i, idx in enumerate(indices):
            getattr(self.entities[idx], cpu_method)(data_cpu[i])
