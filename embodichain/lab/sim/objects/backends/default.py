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

from dexsim.models import MeshObject
from dexsim.engine import Articulation, PhysicsScene
from dexsim.types import (
    ArticulationGPUAPIReadType,
    ArticulationGPUAPIWriteType,
    RigidBodyGPUAPIReadType,
    RigidBodyGPUAPIWriteType,
)
from embodichain.lab.sim.objects.backends.base import (
    ArticulationViewBase,
    RigidBodyViewBase,
)
from embodichain.utils.math import (
    convert_quat,
    matrix_from_quat,
    quat_from_matrix,
)

__all__ = ["DefaultRigidBodyView", "DefaultArticulationView"]


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
        ps: PhysicsScene,
        device: torch.device,
    ) -> None:
        self.entities = list(entities)
        self.ps = ps
        self.device = device
        self._is_gpu = device.type == "cuda"

        if self._is_gpu:
            self._gpu_indices = torch.as_tensor(
                [entity.get_sim_index() for entity in self.entities],
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

    # -- RigidBodyViewBase: center of mass (local) ---------------------------

    def fetch_com_local_pose(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        entities = self._select_entities(body_ids)
        for i, entity in enumerate(entities):
            pos, quat = entity.get_physical_body().get_cmass_local_pose()
            data[i, :3] = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
            data[i, 3:7] = torch.as_tensor(
                convert_quat(quat, to="xyzw"),
                dtype=torch.float32,
                device=self.device,
            )

    def apply_com_local_pose(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        data = data.to(dtype=torch.float32)
        indices = body_ids.detach().cpu().tolist()
        data_cpu = data.cpu().numpy()
        for i, idx in enumerate(indices):
            pos = data_cpu[i, :3]
            quat = convert_quat(data_cpu[i, 3:7], to="wxyz")
            self.entities[idx].get_physical_body().set_cmass_local_pose(pos, quat)

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

    # -- RigidBodyViewBase: physical properties ------------------------------

    def fetch_mass(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        entities = self._select_entities(body_ids)
        for i, entity in enumerate(entities):
            data[i, 0] = entity.get_physical_body().get_mass()

    def apply_mass(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        data_cpu = data.to(dtype=torch.float32).cpu().numpy()
        indices = body_ids.detach().cpu().tolist()
        for i, idx in enumerate(indices):
            self.entities[int(idx)].get_physical_body().set_mass(data_cpu[i, 0])

    def fetch_inertia_diagonal(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        entities = self._select_entities(body_ids)
        for i, entity in enumerate(entities):
            inertia = entity.get_physical_body().get_mass_space_inertia_tensor()
            data[i, :3] = torch.as_tensor(
                inertia, dtype=torch.float32, device=self.device
            )

    def apply_inertia_diagonal(
        self, data: torch.Tensor, body_ids: torch.Tensor
    ) -> None:
        data_cpu = data.to(dtype=torch.float32).cpu().numpy()
        indices = body_ids.detach().cpu().tolist()
        for i, idx in enumerate(indices):
            self.entities[int(idx)].get_physical_body().set_mass_space_inertia_tensor(
                data_cpu[i]
            )

    def fetch_friction(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        entities = self._select_entities(body_ids)
        for i, entity in enumerate(entities):
            data[i, 0] = entity.get_physical_body().get_dynamic_friction()

    def apply_friction(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        data_cpu = data.to(dtype=torch.float32).cpu().numpy()
        indices = body_ids.detach().cpu().tolist()
        for i, idx in enumerate(indices):
            self.entities[int(idx)].get_physical_body().set_dynamic_friction(
                data_cpu[i, 0]
            )
            self.entities[int(idx)].get_physical_body().set_static_friction(
                data_cpu[i, 0]
            )

    def fetch_restitution(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        entities = self._select_entities(body_ids)
        for i, entity in enumerate(entities):
            data[i, 0] = entity.get_physical_body().get_restitution()

    def apply_restitution(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        data_cpu = data.to(dtype=torch.float32).cpu().numpy()
        indices = body_ids.detach().cpu().tolist()
        for i, idx in enumerate(indices):
            self.entities[int(idx)].get_physical_body().set_restitution(data_cpu[i, 0])

    def fetch_contact_offset(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        raise NotImplementedError(
            "Per-body contact_offset fetch is not exposed by the default backend."
        )

    def apply_contact_offset(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        raise NotImplementedError(
            "Per-body contact_offset apply is not exposed by the default backend; "
            "set it via RigidBodyAttributesCfg (consumed at build) instead."
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


class DefaultArticulationView(ArticulationViewBase):
    """Default DexSim backend articulation data adapter."""

    def __init__(
        self,
        entities: Sequence[Articulation],
        ps: PhysicsScene,
        device: torch.device,
    ) -> None:
        self.entities = list(entities)
        self.ps = ps
        self.device = device
        self._is_gpu = device.type == "cuda"

        self.dof = self.entities[0].get_dof()
        self.num_links = self.entities[0].get_links_num()
        self.link_names = self.entities[0].get_link_names()

        if self._is_gpu:
            self._gpu_indices = torch.as_tensor(
                [entity.get_sim_index() for entity in self.entities],
                dtype=torch.int32,
                device=self.device,
            )
            max_dof = self.ps.gpu_get_articulation_max_dof()
        else:
            self._gpu_indices = None
            max_dof = self.dof

        self._qpos_apply = torch.zeros(
            (len(self.entities), max_dof), dtype=torch.float32, device=self.device
        )
        self._target_qpos_apply = torch.zeros_like(self._qpos_apply)
        self._qvel_apply = torch.zeros_like(self._qpos_apply)
        self._target_qvel_apply = torch.zeros_like(self._qpos_apply)
        self._qf_apply = torch.zeros_like(self._qpos_apply)

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def articulation_ids_tensor(self) -> torch.Tensor | None:
        return self._gpu_indices

    def select_articulation_ids(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        if self._gpu_indices is None:
            return torch.as_tensor(env_ids, dtype=torch.int32, device=self.device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        return self._gpu_indices[env_ids.to(device=self.device, dtype=torch.long)]

    def fetch_root_pose(self, data: torch.Tensor) -> torch.Tensor:
        if self._is_gpu:
            self.ps.gpu_fetch_root_data(
                data=data,
                gpu_indices=self._gpu_indices,
                data_type=ArticulationGPUAPIReadType.ROOT_GLOBAL_POSE,
            )
            data[:, :4] = convert_quat(data[:, :4], to="wxyz")
            return data[:, [4, 5, 6, 0, 1, 2, 3]]

        root_pose = torch.as_tensor(
            np.array([entity.get_local_pose() for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )
        xyzs = root_pose[:, :3, 3]
        quats = quat_from_matrix(root_pose[:, :3, :3])
        return torch.cat((xyzs, quats), dim=-1)

    def fetch_root_linear_velocity(self, data: torch.Tensor) -> torch.Tensor:
        if self._is_gpu:
            self.ps.gpu_fetch_root_data(
                data=data,
                gpu_indices=self._gpu_indices,
                data_type=ArticulationGPUAPIReadType.ROOT_LINEAR_VELOCITY,
            )
            return data.clone()
        return torch.as_tensor(
            np.array([entity.get_root_link_velocity()[:3] for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )

    def fetch_root_angular_velocity(self, data: torch.Tensor) -> torch.Tensor:
        if self._is_gpu:
            self.ps.gpu_fetch_root_data(
                data=data,
                gpu_indices=self._gpu_indices,
                data_type=ArticulationGPUAPIReadType.ROOT_ANGULAR_VELOCITY,
            )
            return data.clone()
        return torch.as_tensor(
            np.array([entity.get_root_link_velocity()[3:] for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )

    def fetch_qpos(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_data(data, ArticulationGPUAPIReadType.JOINT_POSITION)

    def fetch_target_qpos(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_data(
            data, ArticulationGPUAPIReadType.JOINT_TARGET_POSITION
        )

    def fetch_qvel(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_data(data, ArticulationGPUAPIReadType.JOINT_VELOCITY)

    def fetch_target_qvel(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_data(
            data, ArticulationGPUAPIReadType.JOINT_TARGET_VELOCITY
        )

    def fetch_qacc(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_data(
            data, ArticulationGPUAPIReadType.JOINT_ACCELERATION
        )

    def fetch_qf(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_data(data, ArticulationGPUAPIReadType.JOINT_FORCE)

    def fetch_link_pose(self, data: torch.Tensor) -> torch.Tensor:
        if self._is_gpu:
            self.ps.gpu_fetch_link_data(
                data=data,
                gpu_indices=self._gpu_indices,
                data_type=ArticulationGPUAPIReadType.LINK_GLOBAL_POSE,
            )
            quat = convert_quat(data[..., :4], to="wxyz")
            return torch.cat((data[..., 4:], quat), dim=-1)

        from embodichain.lab.sim.utility import get_dexsim_arenas

        arenas = get_dexsim_arenas()
        for j, entity in enumerate(self.entities):
            link_pose = np.zeros((self.num_links, 4, 4), dtype=np.float32)
            for i, link_name in enumerate(self.link_names):
                pose = entity.get_link_pose(link_name)
                arena_pose = arenas[j].get_root_node().get_local_pose()
                pose[:2, 3] -= arena_pose[:2, 3]
                link_pose[i] = pose

            link_pose_tensor = torch.from_numpy(link_pose)
            xyz = link_pose_tensor[:, :3, 3]
            quat = quat_from_matrix(link_pose_tensor[:, :3, :3])
            data[j][: self.num_links, :] = torch.cat((xyz, quat), dim=-1)
        return data[:, : self.num_links, :]

    def fetch_link_velocity(
        self,
        data: torch.Tensor,
        linear_data: torch.Tensor,
        angular_data: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_gpu:
            self.ps.gpu_fetch_link_data(
                data=linear_data,
                gpu_indices=self._gpu_indices,
                data_type=ArticulationGPUAPIReadType.LINK_LINEAR_VELOCITY,
            )
            self.ps.gpu_fetch_link_data(
                data=angular_data,
                gpu_indices=self._gpu_indices,
                data_type=ArticulationGPUAPIReadType.LINK_ANGULAR_VELOCITY,
            )
            data[..., :3] = linear_data
            data[..., 3:] = angular_data
            return data[:, : self.num_links, :]

        for i, entity in enumerate(self.entities):
            data[i][: self.num_links] = torch.from_numpy(
                entity.get_link_general_velocities()
            )
        return data[:, : self.num_links, :]

    def apply_root_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        pose = pose.to(dtype=torch.float32)
        if self._is_gpu:
            xyz = pose[:, :3]
            quat = convert_quat(pose[:, 3:7], to="xyzw")
            data = torch.cat((quat, xyz), dim=-1)
            indices = self.select_articulation_ids(env_ids)
            self.ps.gpu_apply_root_data(
                data=data,
                gpu_indices=indices,
                data_type=ArticulationGPUAPIWriteType.ROOT_GLOBAL_POSE,
            )
            self.ps.gpu_compute_articulation_kinematic(gpu_indices=indices)
            return

        pose_cpu = pose.cpu()
        env_indices = self._env_indices_list(env_ids)
        pose_matrix = torch.eye(4).unsqueeze(0).repeat(len(env_indices), 1, 1)
        pose_matrix[:, :3, 3] = pose_cpu[:, :3]
        pose_matrix[:, :3, :3] = matrix_from_quat(pose_cpu[:, 3:7])
        for i, env_idx in enumerate(env_indices):
            self.entities[env_idx].set_local_pose(pose_matrix[i])

    def apply_qpos(
        self,
        qpos: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
        *,
        target: bool,
    ) -> None:
        if self._is_gpu:
            buffer = self._target_qpos_apply if target else self._qpos_apply
            data_type = (
                ArticulationGPUAPIWriteType.JOINT_TARGET_POSITION
                if target
                else ArticulationGPUAPIWriteType.JOINT_POSITION
            )
            self._apply_gpu_joint_rows(buffer, qpos, env_ids, joint_ids, data_type)
            return

        joint_ids_np = self._joint_ids_numpy(joint_ids)
        qpos_np = qpos.detach().cpu().numpy()
        for i, env_idx in enumerate(self._env_indices_list(env_ids)):
            entity = self.entities[env_idx]
            setter = entity.set_target_qpos if target else entity.set_current_qpos
            setter(qpos_np[i], joint_ids_np)

    def apply_qvel(
        self,
        qvel: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
        *,
        target: bool,
    ) -> None:
        if self._is_gpu:
            buffer = self._target_qvel_apply if target else self._qvel_apply
            data_type = (
                ArticulationGPUAPIWriteType.JOINT_TARGET_VELOCITY
                if target
                else ArticulationGPUAPIWriteType.JOINT_VELOCITY
            )
            self._apply_gpu_joint_rows(buffer, qvel, env_ids, joint_ids, data_type)
            return

        joint_ids_np = self._joint_ids_numpy(joint_ids)
        qvel_np = qvel.detach().cpu().numpy()
        for i, env_idx in enumerate(self._env_indices_list(env_ids)):
            entity = self.entities[env_idx]
            setter = entity.set_target_qvel if target else entity.set_current_qvel
            setter(qvel_np[i], joint_ids_np)

    def apply_qf(
        self,
        qf: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        if self._is_gpu:
            self._apply_gpu_joint_rows(
                self._qf_apply,
                qf,
                env_ids,
                joint_ids,
                ArticulationGPUAPIWriteType.JOINT_FORCE,
            )
            return

        joint_ids_np = self._joint_ids_numpy(joint_ids)
        qf_np = qf.detach().cpu().numpy()
        for i, env_idx in enumerate(self._env_indices_list(env_ids)):
            self.entities[env_idx].set_current_qf(qf_np[i], joint_ids_np)

    def clear_dynamics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        zeros = torch.zeros(
            (len(env_ids), self.dof), dtype=torch.float32, device=self.device
        )
        joint_ids = torch.arange(self.dof, dtype=torch.int32, device=self.device)
        self.apply_qvel(zeros, env_ids, joint_ids, target=False)
        self.apply_qvel(zeros, env_ids, joint_ids, target=True)
        self.apply_qf(zeros, env_ids, joint_ids)

    def compute_kinematics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if self._is_gpu:
            self.ps.gpu_compute_articulation_kinematic(
                gpu_indices=self.select_articulation_ids(env_ids)
            )

    def _fetch_joint_data(self, data: torch.Tensor, data_type) -> torch.Tensor:
        if self._is_gpu:
            self.ps.gpu_fetch_joint_data(
                data=data,
                gpu_indices=self._gpu_indices,
                data_type=data_type,
            )
            return data[:, : self.dof].clone()

        method_map = {
            ArticulationGPUAPIReadType.JOINT_POSITION: lambda entity: entity.get_current_qpos(),
            ArticulationGPUAPIReadType.JOINT_TARGET_POSITION: lambda entity: entity.get_current_qpos(
                is_target=True
            ),
            ArticulationGPUAPIReadType.JOINT_VELOCITY: lambda entity: entity.get_current_qvel(),
            ArticulationGPUAPIReadType.JOINT_TARGET_VELOCITY: lambda entity: entity.get_current_qvel(
                is_target=True
            ),
            ArticulationGPUAPIReadType.JOINT_ACCELERATION: lambda entity: entity.get_current_qacc(),
            ArticulationGPUAPIReadType.JOINT_FORCE: lambda entity: entity.get_current_qf(),
        }
        return torch.as_tensor(
            np.array([method_map[data_type](entity) for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )

    def _apply_gpu_joint_rows(
        self,
        buffer: torch.Tensor,
        values: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
        data_type,
    ) -> None:
        env_ids_tensor = self._env_ids_tensor(env_ids)
        joint_ids_tensor = self._joint_ids_tensor(joint_ids)
        buffer[env_ids_tensor[:, None], joint_ids_tensor] = values
        self.ps.gpu_apply_joint_data(
            data=buffer,
            gpu_indices=self.select_articulation_ids(env_ids),
            data_type=data_type,
        )

    def _env_ids_tensor(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if not isinstance(env_ids, torch.Tensor):
            return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        return env_ids.to(device=self.device, dtype=torch.long)

    def _joint_ids_tensor(
        self, joint_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(joint_ids, torch.Tensor):
            return torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
        return joint_ids.to(device=self.device, dtype=torch.long)

    def _env_indices_list(self, env_ids: Sequence[int] | torch.Tensor) -> list[int]:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.detach().cpu().to(dtype=torch.long).tolist()
        return [int(env_idx) for env_idx in env_ids]

    def _joint_ids_numpy(self, joint_ids: Sequence[int] | torch.Tensor) -> np.ndarray:
        if isinstance(joint_ids, torch.Tensor):
            return joint_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        return np.asarray(joint_ids, dtype=np.int32)
