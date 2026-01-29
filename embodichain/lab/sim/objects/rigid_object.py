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
from embodichain.lab.sim.cfg import RigidObjectCfg, RigidBodyAttributesCfg
from embodichain.lab.sim import (
    VisualMaterial,
    VisualMaterialInst,
    BatchEntity,
)
from embodichain.lab.sim.utility import is_rt_enabled
from embodichain.utils.math import convert_quat
from embodichain.utils.math import matrix_from_quat, quat_from_matrix, matrix_from_euler
from embodichain.utils import logger


@dataclass
class RigidBodyData:
    """Data manager for rigid body with body type of dynamic or kinematic.

    Note:
        1. The pose data managed by dexsim is in the format of (qx, qy, qz, qw, x, y, z), but in SimulationManager, we use (x, y, z, qw, qx, qy, qz) format.
    """

    def __init__(
        self, entities: List[MeshObject], ps: PhysicsScene, device: torch.device
    ) -> None:
        """Initialize the RigidBodyData.

        Args:
            entities (List[MeshObject]): List of MeshObjects representing the rigid bodies.
            ps (PhysicsScene): The physics scene.
            device (torch.device): The device to use for the rigid body data.
        """
        self.entities = entities
        self.ps = ps
        self.num_instances = len(entities)
        self.device = device

        # get gpu indices for the entities.
        self.gpu_indices = (
            torch.as_tensor(
                [entity.get_gpu_index() for entity in self.entities],
                dtype=torch.int32,
                device=self.device,
            )
            if self.device.type == "cuda"
            else None
        )

        # Initialize rigid body data.
        self._pose = torch.zeros(
            (self.num_instances, 7), dtype=torch.float32, device=self.device
        )
        self._lin_vel = torch.zeros(
            (self.num_instances, 3), dtype=torch.float32, device=self.device
        )
        self._ang_vel = torch.zeros(
            (self.num_instances, 3), dtype=torch.float32, device=self.device
        )
        # center of mass pose in format (x, y, z, qw, qx, qy, qz)
        self.default_com_pose = torch.zeros(
            (self.num_instances, 7), dtype=torch.float32, device=self.device
        )
        self._com_pose = torch.zeros(
            (self.num_instances, 7), dtype=torch.float32, device=self.device
        )

    @property
    def pose(self) -> torch.Tensor:
        if self.device.type == "cpu":
            # Fetch pose from CPU entities
            xyzs = torch.as_tensor(
                np.array([entity.get_location() for entity in self.entities]),
                dtype=torch.float32,
                device=self.device,
            )
            quats = torch.as_tensor(
                np.array(
                    [entity.get_rotation_quat() for entity in self.entities],
                ),
                dtype=torch.float32,
                device=self.device,
            )
            quats = convert_quat(quats, to="wxyz")
            self._pose = torch.cat((xyzs, quats), dim=-1)
        else:
            self.ps.gpu_fetch_rigid_body_data(
                data=self._pose,
                gpu_indices=self.gpu_indices,
                data_type=RigidBodyGPUAPIReadType.POSE,
            )
            self._pose[:, :4] = convert_quat(self._pose[:, :4], to="wxyz")
            self._pose = self._pose[:, [4, 5, 6, 0, 1, 2, 3]]
        return self._pose

    @property
    def lin_vel(self) -> torch.Tensor:
        if self.device.type == "cpu":
            # Fetch linear velocity from CPU entities
            self._lin_vel = torch.as_tensor(
                np.array([entity.get_linear_velocity() for entity in self.entities]),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.ps.gpu_fetch_rigid_body_data(
                data=self._lin_vel,
                gpu_indices=self.gpu_indices,
                data_type=RigidBodyGPUAPIReadType.LINEAR_VELOCITY,
            )
        return self._lin_vel

    @property
    def ang_vel(self) -> torch.Tensor:
        if self.device.type == "cpu":
            # Fetch angular velocity from CPU entities
            self._ang_vel = torch.as_tensor(
                np.array(
                    [entity.get_angular_velocity() for entity in self.entities],
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.ps.gpu_fetch_rigid_body_data(
                data=self._ang_vel,
                gpu_indices=self.gpu_indices,
                data_type=RigidBodyGPUAPIReadType.ANGULAR_VELOCITY,
            )
        return self._ang_vel

    @property
    def vel(self) -> torch.Tensor:
        """Get the linear and angular velocities of the rigid bodies.

        Returns:
            torch.Tensor: The linear and angular velocities concatenated, with shape (N, 6).
        """
        return torch.cat((self.lin_vel, self.ang_vel), dim=-1)

    @property
    def com_pose(self) -> torch.Tensor:
        """Get the center of mass pose of the rigid bodies.

        Returns:
            torch.Tensor: The center of mass pose with shape (N, 7).
        """
        for i, entity in enumerate(self.entities):
            pos, quat = entity.get_physical_body().get_cmass_local_pose()
            self._com_pose[i, :3] = torch.as_tensor(
                pos, dtype=torch.float32, device=self.device
            )
            self._com_pose[i, 3:7] = torch.as_tensor(
                quat, dtype=torch.float32, device=self.device
            )
        return self._com_pose


class RigidObject(BatchEntity):
    """RigidObject represents a batch of rigid body in the simulation.

    There are three types of rigid body:
        - Static: Actors that do not move and are used as the environment.
        - Dynamic: Actors that can move and are affected by physics.
        - Kinematic: Actors that can move but are not affected by physics.

    """

    def __init__(
        self,
        cfg: RigidObjectCfg,
        entities: List[MeshObject] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.body_type = cfg.body_type

        self._world = dexsim.default_world()
        self._ps = self._world.get_physics_scene()

        self._all_indices = torch.arange(len(entities), dtype=torch.int32).tolist()

        # data for managing body data (only for dynamic and kinematic bodies) on GPU.
        self._data: RigidBodyData | None = None
        if self.is_static is False:
            self._data = RigidBodyData(entities=entities, ps=self._ps, device=device)

        # For rendering purposes, each instance can have its own material.
        self._visual_material: List[VisualMaterialInst] = [None] * len(entities)

        for entity in entities:
            entity.set_body_scale(*cfg.body_scale)
            entity.set_physical_attr(cfg.attrs.attr())

        if device.type == "cuda":
            self._world.update(0.001)

        super().__init__(cfg, entities, device)

        # set default collision filter
        self._set_default_collision_filter()

        # update default center of mass pose (only for non-static bodies with body data).
        if self.body_data is not None:
            self.body_data.default_com_pose = self.body_data.com_pose.clone()

        # TODO: Must be called after setting all attributes.
        # May be improved in the future.
        if cfg.attrs.enable_collision is False:
            flag = torch.zeros(len(entities), dtype=torch.bool)
            self.enable_collision(flag)

        # reserve flag for collision visible node existence
        self._has_collision_visible_node = False

    def __str__(self) -> str:
        parent_str = super().__str__()
        return (
            parent_str
            + f" | body type: {self.body_type} | max_convex_hull_num: {self.cfg.max_convex_hull_num}"
        )

    @property
    def body_data(self) -> RigidBodyData | None:
        """Get the rigid body data manager for this rigid object.

        Returns:
            RigidBodyData: The rigid body data manager.
        """
        if self.is_static:
            logger.log_warning("Static rigid object has no body data.")
            return None

        return self._data

    @property
    def body_state(self) -> torch.Tensor:
        """Get the body state of the rigid object.

        The body state of a rigid object is represented as a tensor with the following format:
        [x, y, z, qw, qx, qy, qz, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]

        If the rigid object is static, linear and angular velocities will be zero.

        Returns:
            torch.Tensor: The body state of the rigid object with shape (N, 13), where N is the number of instances.
        """
        if self.is_static:
            # For static bodies, we return the state with zero velocities.
            zero_velocity = torch.zeros((self.num_instances, 6), device=self.device)
            return torch.cat((self.pose, zero_velocity), dim=-1)

        return torch.cat(
            (self.body_data.pose, self.body_data.lin_vel, self.body_data.ang_vel),
            dim=-1,
        )

    @property
    def is_static(self) -> bool:
        """Check if the rigid object is static.

        Returns:
            bool: True if the rigid object is static, False otherwise.
        """
        return self.body_type == "static"

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
        """set collision filter data for the rigid object.

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
            self._entities[env_idx].get_physical_body().set_collision_filter_data(
                filter_data_np[i]
            )

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set local pose of the rigid object.

        Args:
            pose (torch.Tensor): The local pose of the rigid object with shape (N, 7) or (N, 4, 4).
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(pose):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match pose length {len(pose)}."
            )

        if self.device.type == "cpu" or self.is_static:
            pose = pose.cpu()
            if pose.dim() == 2 and pose.shape[1] == 7:
                pose_matrix = torch.eye(4).unsqueeze(0).repeat(pose.shape[0], 1, 1)
                pose_matrix[:, :3, 3] = pose[:, :3]
                pose_matrix[:, :3, :3] = matrix_from_quat(pose[:, 3:7])
                for i, env_idx in enumerate(local_env_ids):
                    self._entities[env_idx].set_local_pose(pose_matrix[i])
            elif pose.dim() == 3 and pose.shape[1:] == (4, 4):
                for i, env_idx in enumerate(local_env_ids):
                    self._entities[env_idx].set_local_pose(pose[i])
            else:
                logger.log_error(
                    f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
                )

        else:
            if pose.dim() == 2 and pose.shape[1] == 7:
                xyz = pose[:, :3]
                quat = convert_quat(pose[:, 3:7], to="xyzw")
            elif pose.dim() == 3 and pose.shape[1:] == (4, 4):
                xyz = pose[:, :3, 3]
                quat = quat_from_matrix(pose[:, :3, :3])
                quat = convert_quat(quat, to="xyzw")
            else:
                logger.log_error(
                    f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
                )

            # we should keep `pose_` life cycle to the end of the function.
            pose = torch.cat((quat, xyz), dim=-1)
            indices = self.body_data.gpu_indices[local_env_ids]
            torch.cuda.synchronize(self.device)
            self._ps.gpu_apply_rigid_body_data(
                data=pose.clone(),
                gpu_indices=indices,
                data_type=RigidBodyGPUAPIWriteType.POSE,
            )
            if is_rt_enabled() is False:
                self._world.sync_poses_gpu_to_cpu(
                    rigid_pose=CudaArray(pose), rigid_gpu_indices=CudaArray(indices)
                )

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Get local pose of the rigid object.

        Args:
            to_matrix (bool, optional): If True, return the pose as a 4x4 matrix. If False, return as (x, y, z, qw, qx, qy, qz). Defaults to False.

        Returns:
            torch.Tensor: The local pose of the rigid object with shape (N, 7) or (N, 4, 4) depending on `to_matrix`.
        """

        def get_local_pose_cpu(
            entities: List[MeshObject], to_matrix: bool
        ) -> torch.Tensor:
            """Helper function to get local pose on CPU."""
            if to_matrix:
                pose = torch.as_tensor(
                    [entity.get_local_pose() for entity in entities],
                )
            else:
                xyzs = torch.as_tensor([entity.get_location() for entity in entities])
                quats = torch.as_tensor(
                    [entity.get_rotation_quat() for entity in entities]
                )
                quats = convert_quat(quats, to="wxyz")
                pose = torch.cat((xyzs, quats), dim=-1)

            return pose

        if self.is_static:
            return get_local_pose_cpu(self._entities, to_matrix).to(self.device)

        pose = self.body_data.pose
        if to_matrix:
            xyz = pose[:, :3]
            mat = matrix_from_quat(pose[:, 3:7])
            pose = (
                torch.eye(4, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .repeat(pose.shape[0], 1, 1)
            )
            pose[:, :3, 3] = xyz
            pose[:, :3, :3] = mat
        return pose

    def add_force_torque(
        self,
        force: torch.Tensor | None = None,
        torque: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Add force and/or torque to the rigid object.

        TODO: Currently, apply force at position `pos` is not supported.

        Note: there are a few different ways to apply force and torque:
            - If `pos` is specified, the force is applied at that position.
            - if not `pos` is specified, the force and torque are applied at the center of mass of the rigid body.

        Args:
            force (torch.Tensor | None = None): The force to add with shape (N, 3). Defaults to None.
            torque (torch.Tensor | None, optional): The torque to add with shape (N, 3). Defaults to None.
            pos (torch.Tensor | None, optional): The position to apply the force at with shape (N, 3). Defaults to None.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        if force is None and torque is None:
            logger.log_warning(
                "Both force and torque are None. No force or torque will be applied."
            )
            return

        if self.is_non_dynamic:
            logger.log_warning(
                "Cannot apply force or torque to non-dynamic rigid body."
            )
            return

        local_env_ids = self._all_indices if env_ids is None else env_ids

        if force is not None and len(local_env_ids) != len(force):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match force length {len(force)}."
            )

        if torque is not None and len(local_env_ids) != len(torque):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match torque length {len(torque)}."
            )

        if self.device.type == "cpu":
            for i, env_idx in enumerate(local_env_ids):
                if force is not None:
                    self._entities[env_idx].add_force(force[i].cpu().numpy())
                if torque is not None:
                    self._entities[env_idx].add_torque(torque[i].cpu().numpy())

        else:
            indices = self.body_data.gpu_indices[local_env_ids]
            torch.cuda.synchronize(self.device)
            if force is not None:
                self._ps.gpu_apply_rigid_body_data(
                    data=force,
                    gpu_indices=indices,
                    data_type=RigidBodyGPUAPIWriteType.FORCE,
                )
            if torque is not None:
                self._ps.gpu_apply_rigid_body_data(
                    data=torque,
                    gpu_indices=indices,
                    data_type=RigidBodyGPUAPIWriteType.TORQUE,
                )

    def set_attrs(
        self,
        attrs: Union[RigidBodyAttributesCfg, List[RigidBodyAttributesCfg]],
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set physical attributes for the rigid object.

        Args:
            attrs (Union[RigidBodyAttributesCfg, List[RigidBodyAttributesCfg]]): The physical attributes to set.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if isinstance(attrs, List) and len(local_env_ids) != len(attrs):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match attrs length {len(attrs)}."
            )

        # TODO: maybe need to improve the physical attributes setter efficiency.
        if isinstance(attrs, RigidBodyAttributesCfg):
            for i, env_idx in enumerate(local_env_ids):
                self._entities[env_idx].set_physical_attr(attrs.attr())
        else:
            for i, env_idx in enumerate(local_env_ids):
                self._entities[env_idx].set_physical_attr(attrs[i].attr())

    def set_mass(
        self, mass: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set mass for the rigid object.

        Args:
            mass (torch.Tensor): The mass to set with shape (N,).
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(mass):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match mass length {len(mass)}."
            )

        mass = mass.cpu().numpy()
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].get_physical_body().set_mass(mass[i])

    def get_mass(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get mass for the rigid object.

        Args:
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.

        Returns:
            torch.Tensor: The mass of the rigid object with shape (N,).
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        masses = []
        for _, env_idx in enumerate(local_env_ids):
            mass = self._entities[env_idx].get_physical_body().get_mass()
            masses.append(mass)

        return torch.as_tensor(masses, dtype=torch.float32, device=self.device)

    def set_visual_material(
        self, mat: VisualMaterial, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set visual material for the rigid object.

        Args:
            mat (VisualMaterial): The material to set.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        for i, env_idx in enumerate(local_env_ids):
            mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}_{env_idx}")
            self._entities[env_idx].set_material(mat_inst.mat)
            self._visual_material[env_idx] = mat_inst

    def get_visual_material_inst(
        self, env_ids: Sequence[int] | None = None
    ) -> List[VisualMaterialInst]:
        """Get material instances for the rigid object.

        Args:
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.

        Returns:
            List[MaterialInst]: List of material instances.
        """
        ids = env_ids if env_ids is not None else range(self.num_instances)
        return [self._visual_material[i] for i in ids]

    def share_visual_material_inst(self, mat_insts: List[VisualMaterialInst]) -> None:
        """Share material instances for the rigid object.

        Args:
            mat_insts (List[VisualMaterialInst]): List of material instances to share.
        """
        if len(self._entities) != len(mat_insts):
            logger.log_error(
                f"Length of entities {len(self._entities)} does not match length of material instances {len(mat_insts)}."
            )

        for i, entity in enumerate(self._entities):
            if mat_insts[i] is None:
                continue
            entity.set_material(mat_insts[i].mat)
            self._visual_material[i] = mat_insts[i]

    def get_body_scale(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Retrieve the body scale for specified environment instances.

        Args:
            env_ids (Sequence[int] | None): A sequence of environment instance IDs.
                If None, retrieves the body scale for all instances.

        Returns:
            torch.Tensor: A tensor containing the body scales of the specified instances,
            with shape (N, 3) dtype int32 and located on the specified device.
        """
        ids = env_ids if env_ids is not None else range(self.num_instances)
        return torch.as_tensor(
            [self._entities[id].get_body_scale() for id in ids],
            dtype=torch.float32,
            device=self.device,
        )

    def set_body_scale(
        self, scale: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set the scale of the rigid body.

        Args:
            scale (torch.Tensor): The scale to set with shape (N, 3).
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(scale):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match scale length {len(scale)}."
            )

        if self.device.type == "cpu":
            for i, env_idx in enumerate(local_env_ids):
                scale = scale[i].cpu().numpy()
                self._entities[env_idx].set_body_scale(*scale)
        else:
            logger.log_error(f"Setting body scale on GPU is not supported yet.")

    def set_com_pose(
        self, com_pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set the center of mass pose of the rigid body. The pose format is (x, y, z, qw, qx, qy, qz).

        Args:
            com_pose (torch.Tensor): The center of mass pose to set with shape (N, 7).
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        if self.is_non_dynamic:
            logger.log_warning(
                "Cannot set center of mass pose for non-dynamic rigid body."
            )
            return

        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(com_pose):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match com_pose length {len(com_pose)}."
            )

        com_pose = com_pose.cpu().numpy()
        for i, env_idx in enumerate(local_env_ids):
            pos = com_pose[i, :3]
            quat = com_pose[i, 3:7]
            self._entities[env_idx].get_physical_body().set_cmass_local_pose(pos, quat)

    def set_body_type(self, body_type: str) -> None:
        """Set the body type of the rigid object.

        Note:
            Only 'dynamic' and 'kinematic' body types are supported and can be changed at runtime.

        Args:
            body_type (str): The body type to set. Must be one of 'dynamic', or 'kinematic'.
        """
        from dexsim.types import ActorType

        if body_type not in ("dynamic", "kinematic"):
            logger.log_error(
                f"Invalid body type {body_type}. Must be one of 'dynamic', or 'kinematic'."
            )

        if body_type == "dynamic":
            actor_type = ActorType.DYNAMIC
        else:
            actor_type = ActorType.KINEMATIC

        for entity in self._entities:
            entity.set_actor_type(actor_type)

        self.body_type = body_type

    def get_vertices(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Retrieve the vertices of the rigid objects.

        Args:
            env_ids (Sequence[int] | None): A sequence of environment IDs for which to retrieve vertices.
                                                If None, retrieves vertices for all instances.

        Returns:
            torch.Tensor: A tensor containing the user IDs of the specified rigid objects with shape (N, num_verts, 3).
        """
        ids = env_ids if env_ids is not None else range(self.num_instances)
        return torch.as_tensor(
            np.array(
                [self._entities[id].get_vertices() for id in ids],
            ),
            dtype=torch.float32,
            device=self.device,
        )

    def get_user_ids(self) -> torch.Tensor:
        """Get the user ids of the rigid bodies.

        Returns:
            torch.Tensor: A tensor of shape (num_envs,) representing the user ids of the rigid bodies.
        """
        return torch.as_tensor(
            [entity.get_user_id() for entity in self._entities],
            dtype=torch.int32,
            device=self.device,
        )

    def enable_collision(
        self, enable: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Enable or disable collision for the rigid bodies.

        Args:
            enable (torch.Tensor): A tensor of shape (N,) representing whether to enable collision for each rigid body.
            env_ids (Sequence[int] | None): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(enable):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match enable length {len(enable)}."
            )

        enable_list = enable.tolist()
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].enable_collision(bool(enable_list[i]))

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
                self._entities[env_idx].clear_dynamics()
        else:
            # Apply zero force and torque to the rigid bodies.
            zeros = torch.zeros(
                (len(local_env_ids), 3), dtype=torch.float32, device=self.device
            )
            indices = self.body_data.gpu_indices[local_env_ids]
            torch.cuda.synchronize(self.device)
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
            if not self._has_collision_visible_node:
                for i, env_idx in enumerate(self._all_indices):
                    self._entities[env_idx].create_physical_visible_node(
                        np.array(
                            [
                                rgba[0],
                                rgba[1],
                                rgba[2],
                                rgba[3],
                            ]
                        )
                    )
                self._has_collision_visible_node = True

        # create collision visible node if not exist
        for i, env_idx in enumerate(self._all_indices):
            self._entities[env_idx].set_physical_visible(visible)

    def set_visible(self, visible: bool = True) -> None:
        """Set the visibility of the rigid object.

        Args:
            visible (bool, optional): Whether the rigid object is visible. Defaults to True.
        """
        for i, env_idx in enumerate(self._all_indices):
            self._entities[env_idx].set_visible(visible)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        local_env_ids = self._all_indices if env_ids is None else env_ids
        num_instances = len(local_env_ids)
        self.set_attrs(self.cfg.attrs, env_ids=local_env_ids)

        pos = torch.as_tensor(
            self.cfg.init_pos, dtype=torch.float32, device=self.device
        )
        rot = (
            torch.as_tensor(self.cfg.init_rot, dtype=torch.float32, device=self.device)
            * torch.pi
            / 180.0
        )
        pos = pos.unsqueeze(0).repeat(num_instances, 1)
        rot = rot.unsqueeze(0).repeat(num_instances, 1)
        mat = matrix_from_euler(rot, "XYZ")
        pose = (
            torch.eye(4, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .repeat(num_instances, 1, 1)
        )
        pose[:, :3, 3] = pos
        pose[:, :3, :3] = mat
        self.set_local_pose(pose, env_ids=local_env_ids)

        self.clear_dynamics(env_ids=local_env_ids)

    def destroy(self) -> None:
        env = self._world.get_env()
        arenas = env.get_all_arenas()
        if len(arenas) == 0:
            arenas = [env]
        for i, entity in enumerate(self._entities):
            arenas[i].remove_actor(entity)
