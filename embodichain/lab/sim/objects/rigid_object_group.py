# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch
import dexsim
import numpy as np

from dataclasses import dataclass
from typing import List, Sequence, Union

from dexsim.models import MeshObject
from dexsim.types import RigidBodyGPUAPIReadType, RigidBodyGPUAPIWriteType
from dexsim.engine import CudaArray, PhysicsScene
from embodichain.lab.sim.cfg import (
    RigidObjectGroupCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim import (
    BatchEntity,
)
from embodichain.lab.sim.material import VisualMaterial, VisualMaterialInst
from embodichain.utils.math import convert_quat
from embodichain.utils.math import matrix_from_quat, quat_from_matrix, matrix_from_euler
from embodichain.utils import logger


@dataclass
class RigidBodyGroupData:
    """Data manager for rigid body group with body type of dynamic or kinematic."""

    def __init__(
        self, entities: List[List[MeshObject]], ps: PhysicsScene, device: torch.device
    ) -> None:
        """Initialize the RigidBodyGroupData.

        Args:
            entities (List[List[MeshObject]]): List of List MeshObjects representing the rigid body group.
            ps (PhysicsScene): The physics scene.
            device (torch.device): The device to use for the rigid body group data.
        """
        self.entities = entities
        self.ps = ps
        self.num_instances = len(entities)
        self.num_objects = len(entities[0])
        self.device = device

        # get gpu indices for the rigid bodies with shape of (num_instances, num_objects)
        self.gpu_indices = torch.as_tensor(
            [[entity.get_gpu_index() for entity in instance] for instance in entities],
            dtype=torch.int32,
            device=self.device,
        )

        # Initialize rigid body group data tensors. Shape of (num_instances, num_objects, data_dim)
        self._pose = torch.zeros(
            (self.num_instances, self.num_objects, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self._lin_vel = torch.zeros(
            (self.num_instances, self.num_objects, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._ang_vel = torch.zeros(
            (self.num_instances, self.num_objects, 3),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def pose(self) -> torch.Tensor:
        if self.device.type == "cpu":
            # Fetch pose from CPU entities
            xyzs = torch.as_tensor(
                [
                    [entity.get_location() for entity in instance]
                    for instance in self.entities
                ],
                device=self.device,
            )
            quats = torch.as_tensor(
                [
                    [entity.get_rotation_quat() for entity in instance]
                    for instance in self.entities
                ],
                device=self.device,
            )
            quats = convert_quat(quats.reshape(-1, 4), to="wxyz").reshape(
                -1, self.num_objects, 4
            )
            return torch.cat((xyzs, quats), dim=-1)
        else:
            pose = self._pose.reshape(-1, 7)
            self.ps.gpu_fetch_rigid_body_data(
                data=pose,
                gpu_indices=self.gpu_indices.flatten(),
                data_type=RigidBodyGPUAPIReadType.POSE,
            )
            pose = convert_quat(pose[:, :4], to="wxyz")
            pose = pose[:, [4, 5, 6, 0, 1, 2, 3]]
            return self._pose

    @property
    def lin_vel(self) -> torch.Tensor:
        if self.device.type == "cpu":
            # Fetch linear velocity from CPU entities
            self._lin_vel = torch.as_tensor(
                [
                    [entity.get_linear_velocity() for entity in instance]
                    for instance in self.entities
                ],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            lin_vel = self._lin_vel.reshape(-1, 3)
            self.ps.gpu_fetch_rigid_body_data(
                data=lin_vel,
                gpu_indices=self.gpu_indices.flatten(),
                data_type=RigidBodyGPUAPIReadType.LINEAR_VELOCITY,
            )
        return self._lin_vel

    @property
    def ang_vel(self) -> torch.Tensor:
        if self.device.type == "cpu":
            # Fetch angular velocity from CPU entities
            self._ang_vel = torch.as_tensor(
                [
                    [entity.get_linear_velocity() for entity in instance]
                    for instance in self.entities
                ],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            ang_vel = self._ang_vel.reshape(-1, 3)
            self.ps.gpu_fetch_rigid_body_data(
                data=ang_vel,
                gpu_indices=self.gpu_indices.flatten(),
                data_type=RigidBodyGPUAPIReadType.ANGULAR_VELOCITY,
            )
        return self._ang_vel

    @property
    def vel(self) -> torch.Tensor:
        """Get the linear and angular velocities of the rigid bodies.

        Returns:
            torch.Tensor: The linear and angular velocities concatenated, with shape (num_instances, num_objects, 6).
        """
        return torch.cat((self.lin_vel, self.ang_vel), dim=-1)


class RigidObjectGroup(BatchEntity):
    """RigidObjectGroup represents a batch of rigid bodies in the simulation."""

    def __init__(
        self,
        cfg: RigidObjectGroupCfg,
        entities: List[List[MeshObject]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.body_type = cfg.body_type

        self._world = dexsim.default_world()
        self._ps = self._world.get_physics_scene()

        self._all_indices = torch.arange(
            len(entities), dtype=torch.int32, device=device
        )
        self._all_obj_indices = torch.arange(
            len(entities[0]), dtype=torch.int32, device=device
        )

        # data for managing body data (only for dynamic and kinematic bodies) on GPU.
        self._data = RigidBodyGroupData(entities=entities, ps=self._ps, device=device)

        body_cfgs = list(cfg.rigid_objects.values())
        for instance in entities:
            for i, body in enumerate(instance):
                body.set_body_scale(*body_cfgs[i].body_scale)
                body.set_physical_attr(body_cfgs[i].attrs.attr())

        if device.type == "cuda":
            self._world.update(0.001)

        super().__init__(cfg, entities, device)

        # set default collision filter
        self._set_default_collision_filter()

        # reserve flag for collision visible node existence
        n_instances = len(self._entities[0])
        self._has_collision_visible_node_list = [False] * n_instances

    def __str__(self) -> str:
        parent_str = super().__str__()
        return (
            parent_str
            + f" | body type: {self.body_type} | num_objects: {self.num_objects}"
        )

    @property
    def num_objects(self) -> int:
        """Get the number of objects in each rigid body instance.

        Returns:
            int: The number of objects in each rigid body instance.
        """
        return self._data.num_objects

    @property
    def body_data(self) -> RigidBodyGroupData:
        """Get the rigid body data manager for this rigid object.

        Returns:
            RigidBodyGroupData: The rigid body data manager.
        """
        return self._data

    @property
    def body_state(self) -> torch.Tensor:
        """Get the body state of the rigid object.

        The body state of a rigid object is represented as a tensor with the following format:
        [x, y, z, qw, qx, qy, qz, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]

        If the rigid object is static, linear and angular velocities will be zero.

        Returns:
            torch.Tensor: The body state of the rigid object with shape (num_instances, num_objects, 13),
                where N is the number of instances.
        """
        return torch.cat(
            (self.body_data.pose, self.body_data.lin_vel, self.body_data.ang_vel),
            dim=-1,
        )

    @property
    def is_non_dynamic(self) -> bool:
        """Check if the rigid object is non-dynamic (static or kinematic).

        Returns:
            bool: True if the rigid object is non-dynamic, False otherwise.
        """
        return self.body_type in ("static", "kinematic")

    def _set_default_collision_filter(self) -> None:
        collision_filter_data = torch.zeros(
            size=(self.num_instances, 4), dtype=torch.int32
        )
        for i in range(self.num_instances):
            collision_filter_data[i, 0] = i
            collision_filter_data[i, 1] = 1
        self.set_collision_filter(collision_filter_data)

    def set_collision_filter(
        self, filter_data: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """set collision filter data for the rigid object group.

        Args:
            filter_data (torch.Tensor): [N, 4] of int.
                First element of each object is arena id.
                If 2nd element is 0, the object will collision with all other objects in world.
                3rd and 4th elements are not used currently.

            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used. Defaults to None.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(filter_data):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match pose length {len(filter_data)}."
            )

        filter_data_np = filter_data.cpu().numpy().astype(np.uint32)
        for i, env_idx in enumerate(local_env_ids):
            for entity in self._entities[env_idx]:
                entity.get_physical_body().set_collision_filter_data(filter_data_np[i])

    def set_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        obj_ids: Sequence[int] | None = None,
    ) -> None:
        """Set local pose of the rigid object group.

        Args:
            pose (torch.Tensor): The local pose of the rigid object group with shape (num_instances, num_objects, 7) or
                (num_instances, num_objects, 4, 4).
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
            obj_ids (Sequence[int] | None, optional): Object indices within the group. If None, all objects are set. Defaults to None.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        local_obj_ids = self._all_obj_indices if obj_ids is None else obj_ids

        if len(local_env_ids) != len(pose):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match pose length {len(pose)}."
            )

        if self.device.type == "cpu":
            pose = pose.cpu()
            if pose.dim() == 3 and pose.shape[2] == 7:
                reshape_pose = pose.reshape(-1, 7)
                pose_matrix = (
                    torch.eye(4).unsqueeze(0).repeat(reshape_pose.shape[0], 1, 1)
                )
                pose_matrix[:, :3, 3] = reshape_pose[:, :3]
                pose_matrix[:, :3, :3] = matrix_from_quat(reshape_pose[:, 3:7])
                pose = pose_matrix.reshape(-1, len(local_obj_ids), 4, 4)
            elif pose.dim() == 4 and pose.shape[2:] == (4, 4):
                pass
            else:
                logger.log_error(
                    f"Invalid pose shape {pose.shape}. Expected (num_instances, num_objects, 7) or (num_instances, num_objects, 4, 4)."
                )

            for i, env_idx in enumerate(local_env_ids):
                for j, obj_idx in enumerate(local_obj_ids):
                    self._entities[env_idx][obj_idx].set_local_pose(pose[i, j])

        else:
            if pose.dim() == 3 and pose.shape[2] == 7:
                xyz = pose[..., :3].reshape(-1, 3)
                quat = pose[..., 3:7].reshape(-1, 4)
                quat = convert_quat(quat, to="xyzw")
            elif pose.dim() == 4 and pose.shape[2:] == (4, 4):
                xyz = pose[..., :3, 3].reshape(-1, 3)
                mat = pose[..., :3, :3].reshape(-1, 3, 3)
                quat = quat_from_matrix(mat)
                quat = convert_quat(quat, to="xyzw")
            else:
                logger.log_error(
                    f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
                )

            # we should keep `pose_` life cycle to the end of the function.
            pose = torch.cat((quat, xyz), dim=-1)
            indices = self.body_data.gpu_indices[local_env_ids][
                :, local_obj_ids
            ].flatten()
            self._ps.gpu_apply_rigid_body_data(
                data=pose.clone(),
                gpu_indices=indices,
                data_type=RigidBodyGPUAPIWriteType.POSE,
            )
            self._world.sync_poses_gpu_to_cpu(
                rigid_pose=CudaArray(pose), rigid_gpu_indices=CudaArray(indices)
            )

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Get local pose of the rigid object group.

        Args:
            to_matrix (bool, optional): If True, return the pose as a 4x4 matrix. If False, return as (x, y, z, qw, qx, qy, qz). Defaults to False.

        Returns:
            torch.Tensor: The local pose of the rigid object with shape (num_instances, num_objects, 7) or (num_instances, num_objects, 4, 4) depending on `to_matrix`.
        """
        pose = self.body_data.pose
        if to_matrix:
            pose = pose.reshape(-1, 7)
            xyz = pose[:, :3]
            mat = matrix_from_quat(pose[:, 3:7])
            pose = (
                torch.eye(4, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .repeat(self.num_instances * self.num_objects, 1, 1)
            )
            pose[:, :3, 3] = xyz
            pose[:, :3, :3] = mat
            pose = pose.reshape(self.num_instances, self.num_objects, 4, 4)
        return pose

    def get_user_ids(self) -> torch.Tensor:
        """Get the user ids of the rigid body group.

        Returns:
            torch.Tensor: A tensor of shape (num_envs, num_objects) representing the user ids of the rigid body group.
        """
        return torch.as_tensor(
            [
                [entity.get_user_id() for entity in instance]
                for instance in self._entities
            ],
            dtype=torch.int32,
            device=self.device,
        )

    def clear_dynamics(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the dynamics of the rigid bodies by resetting velocities and applying zero forces and torques.

        Args:
            env_ids (Sequence[int] | None): Environment indices. If None, then all indices are used.
        """
        if self.is_non_dynamic:
            return

        local_env_ids = self._all_indices if env_ids is None else env_ids

        if self.device.type == "cpu":
            for env_idx in local_env_ids:
                for entity in self._entities[env_idx]:
                    entity.clear_dynamics()
        else:
            # Apply zero force and torque to the rigid bodies.
            zeros = torch.zeros(
                (len(local_env_ids) * self.num_objects, 3),
                dtype=torch.float32,
                device=self.device,
            )
            indices = self.body_data.gpu_indices[local_env_ids].flatten()
            self._ps.gpu_apply_rigid_body_data(
                data=zeros,
                gpu_indices=indices,
                data_type=RigidBodyGPUAPIWriteType.LINEAR_VELOCITY,
            )
            self._ps.gpu_apply_rigid_body_data(
                data=zeros,
                gpu_indices=indices,
                data_type=RigidBodyGPUAPIWriteType.ANGULAR_VELOCITY,
            )
            self._ps.gpu_apply_rigid_body_data(
                data=zeros,
                gpu_indices=indices,
                data_type=RigidBodyGPUAPIWriteType.FORCE,
            )
            self._ps.gpu_apply_rigid_body_data(
                data=zeros,
                gpu_indices=indices,
                data_type=RigidBodyGPUAPIWriteType.TORQUE,
            )

    def set_visual_material(
        self, mat: VisualMaterial, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set visual material for the rigid object group.

        Args:
            mat (VisualMaterial): The material to set.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        for i, env_idx in enumerate(local_env_ids):
            for j, entity in enumerate(self._entities[env_idx]):
                mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}_{env_idx}_{j}")
                entity.set_material(mat_inst.mat)

        # Note: The rigid object group is not supported to change the visual material once created.
        # If needed, we should create a visual material dict to store the material instances, and
        # implement a get_visual_material method to retrieve the material instances.

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        local_env_ids = self._all_indices if env_ids is None else env_ids
        num_instances = len(local_env_ids)

        self.cfg: RigidObjectGroupCfg
        body_cfgs = list(self.cfg.rigid_objects.values())

        init_pos = []
        init_rot = []
        for cfg in body_cfgs:
            init_pos.append(cfg.init_pos)
            init_rot.append(cfg.init_rot)

        # (num_objects, 3)
        pos = torch.as_tensor(init_pos, dtype=torch.float32, device=self.device)
        rot = (
            torch.as_tensor(init_rot, dtype=torch.float32, device=self.device)
            * torch.pi
            / 180.0
        )
        # Convert pos and rot to shape (num_instances, num_objects, dim)
        pos = pos.unsqueeze_(0).repeat(num_instances, 1, 1)
        rot = rot.unsqueeze_(0).repeat(num_instances, 1, 1)

        mat = matrix_from_euler(rot.reshape(-1, 3), "XYZ")
        # Init pose with shape (num_instances, num_objects, 4, 4)
        pose = (
            torch.eye(4, dtype=torch.float32, device=self.device)
            .unsqueeze_(0)
            .repeat(num_instances * self.num_objects, 1, 1)
        )
        pose[:, :3, 3] = pos.reshape(-1, 3)
        pose[:, :3, :3] = mat
        pose = pose.reshape(num_instances, self.num_objects, 4, 4)
        self.set_local_pose(pose, env_ids=local_env_ids)

        self.clear_dynamics(env_ids=local_env_ids)

    def set_physical_visible(
        self,
        visible: bool = True,
        rgba: Sequence[float] | None = None,
    ):
        """set collion render visibility

        Args:
            visible (bool, optional): is collision body visible. Defaults to True.
            rgba (Sequence[float] | None, optional): collision body visible rgba. It will be defined at the first time the function is called. Defaults to None.
        """
        rgba = rgba if rgba is not None else (0.8, 0.2, 0.2, 0.7)
        if len(rgba) != 4:
            logger.log_error(f"Invalid rgba {rgba}, should be a sequence of 4 floats.")

        # create collision visible node if not exist
        if visible:
            for i, env_idx in enumerate(self._all_indices):
                for intance_id, entity in enumerate(self._entities[env_idx]):
                    if not self._has_collision_visible_node_list[intance_id]:
                        entity.create_physical_visible_node(
                            np.array(
                                [
                                    rgba[0],
                                    rgba[1],
                                    rgba[2],
                                    rgba[3],
                                ]
                            )
                        )
                        self._has_collision_visible_node_list[intance_id] = True

        # create collision visible node if not exist
        for i, env_idx in enumerate(self._all_indices):
            for entity in self._entities[env_idx]:
                entity.set_physical_visible(visible)

    def set_visible(self, visible: bool = True) -> None:
        """Set the visibility of the rigid object group.

        Args:
            visible (bool, optional): Whether the rigid object group is visible. Defaults to True.
        """
        for i, env_idx in enumerate(self._all_indices):
            for entity in self._entities[env_idx]:
                entity.set_visible(visible)

    def destroy(self) -> None:
        env = self._world.get_env()
        arenas = env.get_all_arenas()
        if len(arenas) == 0:
            arenas = [env]
        for i, instance in enumerate(self._entities):
            for entity in instance:
                arenas[i].remove_actor(entity)
