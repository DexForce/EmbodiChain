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

import torch
import dexsim
import numpy as np

from dataclasses import dataclass
from typing import List, Sequence, Union
from functools import cached_property

from dexsim.models import MeshObject
from dexsim.engine import PhysicsScene
from embodichain.lab.sim.cfg import RigidObjectCfg, RigidBodyAttributesCfg
from embodichain.lab.sim.objects.backends import (
    DefaultRigidBodyView,
    NewtonRigidBodyView,
    is_newton_scene,
)
from embodichain.lab.sim.objects.backends.base import RigidBodyViewBase
from embodichain.lab.sim import (
    VisualMaterial,
    VisualMaterialInst,
    BatchEntity,
)
from embodichain.utils.math import convert_quat
from embodichain.utils.math import matrix_from_quat, quat_from_matrix, matrix_from_euler
from embodichain.utils import logger

_UINT64_MAX = (1 << 64) - 1


@dataclass
class RigidBodyData:
    """Data manager for rigid body with body type of dynamic or kinematic.

    All pose/velocity/acceleration data uses EmbodiChain convention:
    ``(x, y, z, qx, qy, qz, qw)``.
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

        # Create the appropriate backend view.
        if is_newton_scene(ps):
            self.body_view: RigidBodyViewBase = NewtonRigidBodyView(
                entities=entities, scene=ps, device=device
            )
        else:
            self.body_view = DefaultRigidBodyView(
                entities=entities, ps=ps, device=device
            )

        # Kept for backward compatibility with callers that index gpu_indices directly.
        self.gpu_indices = self.body_view.body_ids_tensor

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
        self._lin_acc = torch.zeros(
            (self.num_instances, 3), dtype=torch.float32, device=self.device
        )
        self._ang_acc = torch.zeros(
            (self.num_instances, 3), dtype=torch.float32, device=self.device
        )
        # center of mass pose in format (x, y, z, qx, qy, qz, qw)
        self.default_com_pose = torch.zeros(
            (self.num_instances, 7), dtype=torch.float32, device=self.device
        )
        self._com_pose = torch.zeros(
            (self.num_instances, 7), dtype=torch.float32, device=self.device
        )

    @property
    def is_newton_backend(self) -> bool:
        return isinstance(self.body_view, NewtonRigidBodyView)

    def body_ids_for(self, env_ids: Sequence[int]) -> torch.Tensor:
        return self.body_view.select_body_ids(env_ids)

    @property
    def pose(self) -> torch.Tensor:
        if self.body_view.is_ready:
            self.body_view.fetch_pose(self._pose)
            return self._pose

        logger.log_error(f"RigidBodyData pose requested but body view is not ready.")

    @property
    def lin_vel(self) -> torch.Tensor:
        if self.body_view.is_ready:
            self.body_view.fetch_linear_velocity(self._lin_vel)
            return self._lin_vel

        logger.log_error("RigidBodyData lin_vel requested but body view is not ready.")

    @property
    def ang_vel(self) -> torch.Tensor:
        if self.body_view.is_ready:
            self.body_view.fetch_angular_velocity(self._ang_vel)
            return self._ang_vel

        logger.log_error("RigidBodyData ang_vel requested but body view is not ready.")

    @property
    def vel(self) -> torch.Tensor:
        """Get the linear and angular velocities of the rigid bodies.

        Returns:
            torch.Tensor: The linear and angular velocities concatenated, with shape (N, 6).
        """
        return torch.cat((self.lin_vel, self.ang_vel), dim=-1)

    @property
    def lin_acc(self) -> torch.Tensor:
        if self.body_view.is_ready:
            self.body_view.fetch_linear_acceleration(self._lin_acc)
            return self._lin_acc

        logger.log_error("RigidBodyData lin_acc requested but body view is not ready.")

    @property
    def ang_acc(self) -> torch.Tensor:
        if self.body_view.is_ready:
            self.body_view.fetch_angular_acceleration(self._ang_acc)
            return self._ang_acc

        logger.log_error("RigidBodyData ang_acc requested but body view is not ready.")

    @property
    def acc(self) -> torch.Tensor:
        """Get the linear and angular accelerations of the rigid bodies.

        Returns:
            torch.Tensor: The linear and angular accelerations concatenated, with shape (N, 6).
        """
        return torch.cat((self.lin_acc, self.ang_acc), dim=-1)

    @property
    def com_pose(self) -> torch.Tensor:
        """Get the center of mass pose of the rigid bodies.

        Returns:
            torch.Tensor: The center of mass pose with shape (N, 7).
        """
        if self.is_newton_backend:
            if getattr(self.body_view, "supports_com_local_pose", False):
                self.body_view.fetch_com_local_pose(self._com_pose)
                return self._com_pose

            manager = self.body_view.scene.manager
            for i, entity_handle in enumerate(self.body_view.entity_handles):
                attr = manager.dexsim_meta.get(entity_handle, {}).get("attr")
                if attr is None:
                    pos = np.zeros(3, dtype=np.float32)
                    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    pos = np.asarray(attr.com_position, dtype=np.float32).copy()
                    quat = np.asarray(attr.com_quaternion, dtype=np.float32).copy()
                self._com_pose[i, :3] = torch.as_tensor(
                    pos, dtype=torch.float32, device=self.device
                )
                self._com_pose[i, 3:7] = torch.as_tensor(
                    convert_quat(quat, to="xyzw"),
                    dtype=torch.float32,
                    device=self.device,
                )
            return self._com_pose

        for i, entity in enumerate(self.entities):
            pos, quat = entity.get_physical_body().get_cmass_local_pose()
            self._com_pose[i, :3] = torch.as_tensor(
                pos, dtype=torch.float32, device=self.device
            )
            self._com_pose[i, 3:7] = torch.as_tensor(
                convert_quat(np.asarray(quat, dtype=np.float32), to="xyzw"),
                dtype=torch.float32,
                device=self.device,
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
        from embodichain.lab.sim.sim_manager import get_physics_scene

        self._ps = get_physics_scene()

        self._all_indices = torch.arange(len(entities), dtype=torch.int32).tolist()

        # data for managing body data (only for dynamic and kinematic bodies) on GPU.
        self._data: RigidBodyData | None = None
        if self.is_static is False:
            self._data = RigidBodyData(entities=entities, ps=self._ps, device=device)

        # For rendering purposes, each instance can have its own material.
        self._visual_material: List[VisualMaterialInst] = [None] * len(entities)
        self.is_shared_visual_material = False

        # Determine if we should use USD properties or cfg properties.
        if not cfg.use_usd_properties:
            for entity in entities:
                entity.set_body_scale(*cfg.body_scale)
                if is_newton_scene(self._ps):
                    # TODO: DexSim Newton consumes the initial physical
                    # attributes during add_rigidbody(); MeshObject
                    # set_physical_attr() is still default-backend only.
                    continue
                entity.set_physical_attr(cfg.attrs.attr())
        else:
            # Read current properties from USD-loaded entities and write back to cfg
            # Use first entity as reference
            first_entity: MeshObject = entities[0]

            cfg.body_scale = tuple(first_entity.get_body_scale())
            cfg.attrs = RigidBodyAttributesCfg().from_dict(
                first_entity.get_physical_attr().as_dict()
            )

        super().__init__(cfg, entities, device)

        # set default collision filter
        self._set_default_collision_filter()

        if device.type == "cuda":
            self._world.update(0.001)
        self.reset()

        # update default center of mass pose (only for non-static bodies with body data).
        if self._data is not None:
            self._data.default_com_pose = self._data.com_pose.clone()

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

    @cached_property
    def user_ids(self) -> torch.Tensor:
        """Get the user ids of the rigid object.

        Returns:
            torch.Tensor: The user ids of the rigid object with shape (N,).
        """
        return torch.as_tensor(
            np.array([entity.get_user_id() for entity in self._entities]),
            dtype=torch.int32,
            device=self.device,
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

    def _get_newton_attr(self, env_idx: int):
        """Return DexSim Newton metadata physical attributes for an entity."""
        entity = self._entities[env_idx]
        entity_handle = int(entity.get_native_handle())
        if entity_handle < 0:
            entity_handle &= _UINT64_MAX

        manager = getattr(self._ps, "manager", None)
        attr = None
        if manager is not None:
            attr = (
                getattr(manager, "dexsim_meta", {}).get(entity_handle, {}).get("attr")
            )
        if attr is None:
            logger.log_error(
                f"Newton physical attributes for rigid object '{self.uid}' env {env_idx} are unavailable."
            )
        return attr

    def _warn_newton_unsupported(self, api_name: str) -> None:
        logger.log_warning(
            f"Newton backend does not support RigidObject.{api_name} runtime updates yet. "
            "Skipping this call. TODO: wire this API when DexSim Newton exposes runtime physical-attribute mutation."
        )

    def _newton_lifecycle_state(self) -> str:
        manager = getattr(self._ps, "manager", None)
        return getattr(getattr(manager, "lifecycle_state", None), "name", "")

    def _can_use_newton_entity_dynamics_fallback(self) -> bool:
        """Return whether per-entity Newton patches are safe before GPU view is ready.

        DexSim Newton only supports MeshObject force/torque helpers in ``BUILDER``
        state. Calling them while the model is ``STALE`` can index stale body ids.
        """
        return self._newton_lifecycle_state() == "BUILDER"

    @property
    def body_state(self) -> torch.Tensor:
        """Get the body state of the rigid object.

        The body state of a rigid object is represented as a tensor with the following format:
        [x, y, z, qx, qy, qz, qw, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]

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
            entity = self._entities[env_idx]
            if is_newton_scene(self._ps):
                entity.set_collision_filter_data(filter_data_np[i])
            else:
                entity.get_physical_body().set_collision_filter_data(filter_data_np[i])

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

        # Normalize pose to (N, 7) format in (x, y, z, qx, qy, qz, qw).
        if pose.dim() == 2 and pose.shape[1] == 7:
            target_pose = pose.to(device=self.device, dtype=torch.float32)
        elif pose.dim() == 3 and pose.shape[1:] == (4, 4):
            xyz = pose[:, :3, 3]
            quat = convert_quat(quat_from_matrix(pose[:, :3, :3]), to="xyzw")
            target_pose = torch.cat((xyz, quat), dim=-1).to(
                device=self.device, dtype=torch.float32
            )
        else:
            logger.log_error(
                f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
            )
            return

        # Use backend view if available and ready.
        if (
            self._data is not None
            and self._data.body_view.is_ready
            and not self.is_static
        ):
            body_ids = self._data.body_ids_for(local_env_ids)
            self._data.body_view.apply_pose(target_pose, body_ids)
            return

        # Static bodies and non-ready backends (notably Newton before finalize)
        # still accept direct entity pose updates.
        target_pose = target_pose.cpu()
        pose_matrix = torch.eye(4).unsqueeze(0).repeat(len(local_env_ids), 1, 1)
        pose_matrix[:, :3, 3] = target_pose[:, :3]
        pose_matrix[:, :3, :3] = matrix_from_quat(
            convert_quat(target_pose[:, 3:7], to="wxyz")
        )
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].set_local_pose(pose_matrix[i])

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Get local pose of the rigid object.

        Args:
            to_matrix (bool, optional): If True, return the pose as a 4x4 matrix. If False, return as (x, y, z, qx, qy, qz, qw). Defaults to False.

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
                pose = torch.cat((xyzs, quats), dim=-1)

            return pose

        if self.is_static:
            return get_local_pose_cpu(self._entities, to_matrix).to(self.device)

        pose = self.body_data.pose.clone()
        if to_matrix:
            xyz = pose[:, :3]
            mat = matrix_from_quat(convert_quat(pose[:, 3:7], to="wxyz"))
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

        if pos is not None:
            logger.log_warning(
                "RigidObject.add_force_torque(pos=...) is not supported yet; "
                "applying wrench at center of mass."
            )

        if self._data is not None and self._data.body_view.is_ready:
            body_ids = self._data.body_ids_for(local_env_ids)
            if force is not None:
                self._data.body_view.apply_force(force, body_ids)
            if torque is not None:
                self._data.body_view.apply_torque(torque, body_ids)
        elif (
            self._data is not None
            and self._data.is_newton_backend
            and self._can_use_newton_entity_dynamics_fallback()
        ):
            force_np = force.detach().cpu().numpy() if force is not None else None
            torque_np = torque.detach().cpu().numpy() if torque is not None else None
            for i, env_idx in enumerate(local_env_ids):
                entity = self._entities[env_idx]
                if force_np is not None:
                    entity.add_force(force_np[i])
                if torque_np is not None:
                    entity.add_torque(torque_np[i])
        elif self._data is not None and self._data.is_newton_backend:
            logger.log_warning(
                "Cannot apply force or torque while Newton model is stale or "
                "unfinalized; call SimulationManager.prepare_physics() first."
            )
        else:
            logger.log_error("Cannot apply force or torque before body view is ready.")

    def set_velocity(
        self,
        lin_vel: torch.Tensor | None = None,
        ang_vel: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set linear and/or angular velocity for the rigid object.

        Args:
            lin_vel (torch.Tensor | None, optional): The linear velocity to set with shape (N, 3). Defaults to None.
            ang_vel (torch.Tensor | None, optional): The angular velocity to set with shape (N, 3). Defaults to None.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        if lin_vel is None and ang_vel is None:
            logger.log_warning(
                "Both lin_vel and ang_vel are None. No velocity will be set."
            )
            return

        if self.is_non_dynamic:
            logger.log_warning("Cannot set velocity for non-dynamic rigid body.")
            return

        local_env_ids = self._all_indices if env_ids is None else env_ids

        if lin_vel is not None and len(local_env_ids) != len(lin_vel):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match lin_vel length {len(lin_vel)}."
            )

        if ang_vel is not None and len(local_env_ids) != len(ang_vel):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match ang_vel length {len(ang_vel)}."
            )

        if self._data is not None and self._data.body_view.is_ready:
            body_ids = self._data.body_ids_for(local_env_ids)
            if lin_vel is not None:
                self._data.body_view.apply_linear_velocity(lin_vel, body_ids)
            if ang_vel is not None:
                self._data.body_view.apply_angular_velocity(ang_vel, body_ids)
        elif (
            self._data is not None
            and self._data.is_newton_backend
            and self._can_use_newton_entity_dynamics_fallback()
        ):
            lin_vel_np = lin_vel.detach().cpu().numpy() if lin_vel is not None else None
            ang_vel_np = ang_vel.detach().cpu().numpy() if ang_vel is not None else None
            for i, env_idx in enumerate(local_env_ids):
                entity = self._entities[env_idx]
                if lin_vel_np is not None:
                    entity.set_linear_velocity(lin_vel_np[i])
                if ang_vel_np is not None:
                    entity.set_angular_velocity(ang_vel_np[i])
        elif self._data is not None and self._data.is_newton_backend:
            logger.log_warning(
                "Cannot set velocity while Newton model is stale or unfinalized; "
                "call SimulationManager.prepare_physics() first."
            )
        else:
            logger.log_error("Cannot set velocity before body view is ready.")

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

        if is_newton_scene(self._ps):
            self._warn_newton_unsupported("set_attrs")
            return

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

        if is_newton_scene(self._ps):
            self._warn_newton_unsupported("set_mass")
            return

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
            if is_newton_scene(self._ps):
                mass = self._get_newton_attr(env_idx).mass
            else:
                mass = self._entities[env_idx].get_physical_body().get_mass()
            masses.append(mass)

        return torch.as_tensor(masses, dtype=torch.float32, device=self.device)

    def set_friction(
        self, friction: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set friction for the rigid object.

        Args:
            friction (torch.Tensor): The friction to set with shape (N,).
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(friction):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match friction length {len(friction)}."
            )

        if is_newton_scene(self._ps):
            self._warn_newton_unsupported("set_friction")
            return

        friction = friction.cpu().numpy()
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].get_physical_body().set_dynamic_friction(
                friction[i]
            )
            self._entities[env_idx].get_physical_body().set_static_friction(friction[i])

    def get_friction(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get friction for the rigid object.

        Args:
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.

        Returns:
            torch.Tensor: The friction of the rigid object with shape (N,).
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        frictions = []
        for _, env_idx in enumerate(local_env_ids):
            if is_newton_scene(self._ps):
                friction = self._get_newton_attr(env_idx).dynamic_friction
            else:
                friction = (
                    self._entities[env_idx].get_physical_body().get_dynamic_friction()
                )
            frictions.append(friction)

        return torch.as_tensor(frictions, dtype=torch.float32, device=self.device)

    def set_damping(
        self, damping: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set linear and angular damping for the rigid object.

        Args:
            damping (torch.Tensor): The damping to set with shape (N, 2), where the first column is linear damping and the second column is angular damping.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(damping):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match damping length {len(damping)}."
            )

        if is_newton_scene(self._ps):
            self._warn_newton_unsupported("set_damping")
            return

        damping = damping.cpu().numpy()
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].get_physical_body().set_linear_damping(
                damping[i, 0]
            )
            self._entities[env_idx].get_physical_body().set_angular_damping(
                damping[i, 1]
            )

    def get_damping(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get linear and angular damping for the rigid object.

        Args:
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.

        Returns:
            torch.Tensor: The damping of the rigid object with shape (N, 2), where the first column is linear damping and the second column is angular damping.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        dampings = []
        for _, env_idx in enumerate(local_env_ids):
            if is_newton_scene(self._ps):
                attr = self._get_newton_attr(env_idx)
                linear_damping = attr.linear_damping
                angular_damping = attr.angular_damping
            else:
                linear_damping = (
                    self._entities[env_idx].get_physical_body().get_linear_damping()
                )
                angular_damping = (
                    self._entities[env_idx].get_physical_body().get_angular_damping()
                )
            dampings.append([linear_damping, angular_damping])

        return torch.as_tensor(dampings, dtype=torch.float32, device=self.device)

    def set_inertia(
        self, inertia: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set inertia tensor for the rigid object.

        Args:
            inertia (torch.Tensor): The inertia tensor to set with shape (N, 3), where each row is the diagonal of the inertia tensor.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(inertia):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match inertia length {len(inertia)}."
            )

        if is_newton_scene(self._ps):
            self._warn_newton_unsupported("set_inertia")
            return

        inertia = inertia.cpu().numpy()
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].get_physical_body().set_mass_space_inertia_tensor(
                inertia[i]
            )

    def get_inertia(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get inertia tensor for the rigid object.

        Args:
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.

        Returns:
            torch.Tensor: The inertia tensor of the rigid object with shape (N, 3), where each row is the diagonal of the inertia tensor.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        inertias = []
        for _, env_idx in enumerate(local_env_ids):
            if is_newton_scene(self._ps):
                inertia = self._get_newton_attr(env_idx).inertia
            else:
                inertia = (
                    self._entities[env_idx]
                    .get_physical_body()
                    .get_mass_space_inertia_tensor()
                )
            inertias.append(inertia)

        return torch.as_tensor(inertias, dtype=torch.float32, device=self.device)

    def set_visual_material(
        self,
        mat: VisualMaterial,
        env_ids: Sequence[int] | None = None,
        shared: bool = False,
    ) -> None:
        """Set visual material for the rigid object.

        Note:
            If `shared` is True, the same material instance will be used for all specified environment indices.
            If `shared` is False, a unique material instance will be created for each specified environment index.

        Args:
            mat (VisualMaterial): The material to set.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
            shared (bool, optional): Whether to share the material instance among all specified environment indices. Defaults to False.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if shared:
            if len(local_env_ids) != self.num_instances:
                logger.log_error(f"Cannot share material instance for partial env_ids.")

            mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}")
            for env_idx in local_env_ids:
                self._entities[env_idx].set_material(mat_inst.mat)
                self._visual_material[env_idx] = mat_inst
            self.is_shared_visual_material = True
        else:
            for i, env_idx in enumerate(local_env_ids):
                mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}_{env_idx}")
                self._entities[env_idx].set_material(mat_inst.mat)
                self._visual_material[env_idx] = mat_inst
            self.is_shared_visual_material = False

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

        for i, env_idx in enumerate(local_env_ids):
            scale_np = scale[i].cpu().numpy()
            self._entities[env_idx].set_body_scale(*scale_np)

    def set_com_pose(
        self, com_pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set the center of mass pose of the rigid body. The pose format is (x, y, z, qx, qy, qz, qw).

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

        if self._data is not None and self._data.is_newton_backend:
            body_view = self._data.body_view
            if getattr(body_view, "supports_com_local_pose", False):
                body_ids = self._data.body_ids_for(local_env_ids)
                body_view.apply_com_local_pose(
                    com_pose.to(device=self.device, dtype=torch.float32), body_ids
                )
                return

        com_pose = com_pose.cpu().numpy()
        for i, env_idx in enumerate(local_env_ids):
            pos = com_pose[i, :3]
            quat = convert_quat(com_pose[i, 3:7], to="wxyz")
            if self._data is not None and self._data.is_newton_backend:
                self._entities[env_idx].set_cmass_local_pose(pos, quat)
            else:
                self._entities[env_idx].get_physical_body().set_cmass_local_pose(
                    pos, quat
                )

    def set_body_type(self, body_type: str) -> None:
        """Set the body type of the rigid object.

        Note:
            Only 'dynamic' and 'kinematic' body types are supported and can be changed at runtime.

        Args:
            body_type (str): The body type to set. Must be one of 'dynamic', or 'kinematic'.
        """
        from dexsim.types import ActorType

        if is_newton_scene(self._ps):
            self._warn_newton_unsupported("set_body_type")
            return

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

    def get_vertices(
        self, env_ids: Sequence[int] | None = None, scale: bool = False
    ) -> torch.Tensor:
        """
        Retrieve the vertices of the rigid objects.

        Args:
            env_ids (Sequence[int] | None): A sequence of environment IDs for which to retrieve vertices.
                                                If None, retrieves vertices for all instances.
            scale (bool): Whether to multiply the vertices by the body scale. Defaults to False.

        Returns:
            torch.Tensor: A tensor containing the user IDs of the specified rigid objects with shape (N, num_verts, 3).
        """
        ids = env_ids if env_ids is not None else range(self.num_instances)
        verts = torch.as_tensor(
            np.array(
                [self._entities[id].get_vertices() for id in ids],
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if scale:
            verts = verts * self.get_body_scale(env_ids).unsqueeze_(1)
        return verts

    def get_triangles(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Retrieve the triangle indices of the rigid objects.

        Args:
            env_ids (Sequence[int] | None): A sequence of environment IDs for which to retrieve triangle indices.
                                                If None, retrieves triangle indices for all instances.

        Returns:
            torch.Tensor: A tensor containing the triangle indices of the specified rigid objects with shape (N, num_tris, 3).
        """
        ids = env_ids if env_ids is not None else range(self.num_instances)
        return torch.as_tensor(
            np.array(
                [self._entities[id].get_triangles() for id in ids],
            ),
            dtype=torch.int32,
            device=self.device,
        )

    def get_user_ids(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get the user ids of the rigid bodies.

        Args:
            env_ids (Sequence[int] | None): Environment indices. If None, then all indices are used.

        Returns:
            torch.Tensor: A tensor of shape (num_envs,) representing the user ids of the rigid bodies.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        return self.user_ids[local_env_ids]

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

        if self._data is not None and self._data.body_view.is_ready:
            zeros = torch.zeros(
                (len(local_env_ids), 3), dtype=torch.float32, device=self.device
            )
            body_ids = self._data.body_ids_for(local_env_ids)
            self._data.body_view.apply_linear_velocity(zeros, body_ids)
            self._data.body_view.apply_angular_velocity(zeros, body_ids)
            self._data.body_view.apply_force(zeros, body_ids)
            self._data.body_view.apply_torque(zeros, body_ids)
        elif (
            self._data is not None
            and self._data.is_newton_backend
            and self._can_use_newton_entity_dynamics_fallback()
        ):
            for env_idx in local_env_ids:
                self._entities[env_idx].clear_dynamics()
        elif self._data is not None and self._data.is_newton_backend:
            logger.log_warning(
                "Cannot clear dynamics while Newton model is stale or unfinalized; "
                "call SimulationManager.prepare_physics() first."
            )
        else:
            logger.log_error("Cannot clear dynamics before body view is ready.")

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

        if not is_newton_scene(self._ps):
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
            if is_newton_scene(self._ps):
                arenas[i].remove_actor(entity.get_name())
            else:
                arenas[i].remove_actor(entity)
