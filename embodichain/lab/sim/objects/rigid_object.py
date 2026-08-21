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

from copy import deepcopy
from dataclasses import dataclass, MISSING
from typing import TYPE_CHECKING, List, Sequence, Union
from functools import cached_property

from dexsim.models import MeshObject
from dexsim.types import RigidBodyGPUAPIReadType, RigidBodyGPUAPIWriteType
from dexsim.engine import CudaArray, MaterialInst, PhysicsScene
from embodichain.lab.sim.cfg import RigidObjectCfg, RigidBodyAttributesCfg
from embodichain.lab.sim.objects.backends import (
    DefaultRigidBodyView,
    NewtonRigidBodyView,
    apply_collision_filter_for_entities,
    is_newton_scene,
)
from embodichain.lab.sim.objects.backends.base import RigidBodyViewBase
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim import (
    VisualMaterial,
    VisualMaterialInst,
    ReuseSegmentState,
    BatchEntity,
)
from embodichain.lab.sim.material import (
    _capture_render_materials,
    _restore_render_materials,
    _set_render_material,
    _wrap_first_render_material,
)
from ._mesh_utils import (
    get_combined_triangles,
    get_combined_vertices,
)
from embodichain.utils.math import convert_quat
from embodichain.utils.math import matrix_from_quat, quat_from_matrix, matrix_from_euler
from embodichain.utils import logger

if TYPE_CHECKING:
    from dexsim.spawn import SpawnResult, SpawnedObject

_UINT64_MAX = (1 << 64) - 1
__all__ = ["RigidBodyData", "RigidObject", "RigidObjectCfg"]


@dataclass
class RigidBodyData:
    """Data manager for rigid body with body type of dynamic or kinematic.

    All pose/velocity/acceleration data uses EmbodiChain convention:
    ``(x, y, z, qx, qy, qz, qw)``.
    """

    def __init__(
        self,
        entities: List[MeshObject],
        ps: PhysicsScene | None,
        device: torch.device,
        body_view: RigidBodyViewBase | None = None,
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
        if body_view is not None:
            self.body_view = body_view
        elif is_newton_scene(ps):
            self.body_view: RigidBodyViewBase = NewtonRigidBodyView(
                entities=entities, scene=ps, device=device
            )
        else:
            self.body_view = DefaultRigidBodyView(
                entities=entities, ps=ps, device=device
            )

        # Kept for backward compatibility with callers that index gpu_indices directly.
        # NOTE: for Newton, body IDs are lazily resolved after finalization.
        # Use the ``gpu_indices`` property instead of caching here.

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
        # Physical property buffers
        self._mass = torch.zeros(
            (self.num_instances, 1), dtype=torch.float32, device=self.device
        )
        self._inertia = torch.zeros(
            (self.num_instances, 3), dtype=torch.float32, device=self.device
        )
        self._friction = torch.zeros(
            (self.num_instances, 1), dtype=torch.float32, device=self.device
        )

    @property
    def is_newton_backend(self) -> bool:
        return bool(
            getattr(
                self.body_view,
                "is_newton_backend",
                isinstance(self.body_view, NewtonRigidBodyView),
            )
        )

    @property
    def gpu_indices(self) -> torch.Tensor:
        """Body ID tensor (backward-compatible alias for ``body_view.body_ids_tensor``)."""
        return self.body_view.body_ids_tensor

    def body_ids_for(self, env_ids: Sequence[int]) -> torch.Tensor:
        return self.body_view.select_body_ids(env_ids)

    @property
    def pose(self) -> torch.Tensor:
        if self.body_view.can_fetch_pose:
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
        self.body_view.fetch_com_local_pose(self._com_pose)
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
        *,
        spawn_result: SpawnResult | None = None,
        declared_num_instances: int | None = None,
    ) -> None:
        if entities is None:
            if declared_num_instances is None or declared_num_instances <= 0:
                raise ValueError(
                    "A declared RigidObject requires declared_num_instances > 0."
                )
            self.cfg = deepcopy(cfg)
            self.uid = self.cfg.uid
            self.device = device
            self.body_type = cfg.body_type
            self._entities = []
            self._declared_num_instances = declared_num_instances
            self._spawn_result = None
            self._ps = None
            self._world = None
            self._data = None
            self._all_indices = list(range(declared_num_instances))
            self._visual_material = [None] * declared_num_instances
            self.is_shared_visual_material = False
            self._has_collision_visible_node = False
            return

        self._declared_num_instances = len(entities)
        self._spawn_result = spawn_result
        self.body_type = cfg.body_type

        if spawn_result is None:
            self._world = dexsim.default_world()
            from embodichain.lab.sim.sim_manager import get_physics_scene

            self._ps = get_physics_scene()
        else:
            self._world = spawn_result.world
            self._ps = None

        self._all_indices = torch.arange(len(entities), dtype=torch.int32).tolist()

        # data for managing body data (only for dynamic and kinematic bodies) on GPU.
        self._data: RigidBodyData | None = None
        if self.is_static is False:
            body_view = None
            if spawn_result is not None:
                from embodichain.lab.sim.objects.backends import SpawnRigidBodyView

                batch = spawn_result.create_rigid_body_batch(entities)
                body_view = SpawnRigidBodyView(spawn_result, batch, device)
            self._data = RigidBodyData(
                entities=entities,
                ps=self._ps,
                device=device,
                body_view=body_view,
            )

        # For rendering purposes, each instance can have its own material.
        self._visual_material: List[VisualMaterialInst] = [None] * len(entities)
        self.is_shared_visual_material = False

        # Determine if we should use USD properties or cfg properties.
        if spawn_result is None and not cfg.use_usd_properties:
            for entity in entities:
                entity.set_body_scale(*cfg.body_scale)
                if is_newton_scene(self._ps):
                    # TODO: DexSim Newton consumes the initial physical
                    # attributes during add_rigidbody(); MeshObject
                    # set_physical_attr() is still default-backend only.
                    continue
                entity.set_physical_attr(cfg.attrs.attr())
        elif spawn_result is None:
            # Read current properties from USD-loaded entities and write back to cfg
            # Use first entity as reference
            first_entity: MeshObject = entities[0]

            cfg.body_scale = tuple(first_entity.get_body_scale())
            cfg.attrs = RigidBodyAttributesCfg().from_dict(
                first_entity.get_physical_attr().as_dict()
            )

        super().__init__(cfg, entities, device, auto_reset=False)

        self._initialize_existing_visual_material()

        # set default collision filter
        if spawn_result is None:
            self._set_default_collision_filter()

        self._apply_initial_state()

        # update default center of mass pose (only for non-static bodies with body data).
        if self._data is not None:
            self._data.default_com_pose = self._data.com_pose.clone()

        # TODO: Must be called after setting all attributes.
        # May be improved in the future.
        if spawn_result is None and cfg.attrs.enable_collision is False:
            flag = torch.zeros(len(entities), dtype=torch.bool)
            self.enable_collision(flag)

        # reserve flag for collision visible node existence
        self._has_collision_visible_node = False

    @property
    def is_spawn_bound(self) -> bool:
        """Whether this facade is bound to one finalized SpawnResult."""
        return self._spawn_result is not None

    @property
    def is_declared(self) -> bool:
        """Whether this facade is waiting for its SpawnResult binding."""
        return self._world is None

    @property
    def num_instances(self) -> int:
        if self._entities:
            return len(self._entities)
        return self._declared_num_instances

    def attach_spawn_handles(
        self,
        entities: Sequence[SpawnedObject],
    ) -> None:
        """Store materialized handles without initializing runtime Batch data.

        Default may call this before Spawn finalization so native metadata is
        available early. ``bind_spawn()`` remains responsible for creating
        result-dependent Batch/Data state after finalization.
        """
        self._entities = list(entities)

    def bind_spawn(
        self,
        result: SpawnResult,
    ) -> None:
        """Bind a declared facade to stable Spawn handles in place."""
        cfg = self.cfg
        device = self.device
        entities = list(self._entities)
        type(self).__init__(
            self,
            cfg,
            entities,
            device,
            spawn_result=result,
        )

    def __str__(self) -> str:
        if self.is_declared:
            parent_str = (
                f"{self.__class__}: declared {self.num_instances} Spawn objects "
                f"| uid: {self.uid} | device: {self.device}"
            )
        else:
            parent_str = super().__str__()
        max_hull = self.cfg.max_convex_hull_num
        if max_hull is MISSING:
            if isinstance(self.cfg.shape, MeshCfg):
                max_hull = self.cfg.shape.max_convex_hull_num
            else:
                max_hull = 1
        return (
            parent_str
            + f" | body type: {self.body_type} | max_convex_hull_num: {max_hull}"
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

    def _get_newton_attr_or_none(self, env_idx: int):
        """Return the Newton meta PhysicalAttr, or None when not present.

        Unlike :meth:`_get_newton_attr` this does not raise: objects spawned via
        the desc-native path (``attrs.newton`` set) carry ``newton_shape``/
        ``newton_body`` descriptors instead of a legacy ``attr``, so they have
        no meta ``PhysicalAttr`` to mirror onto. Used by the not-ready setter
        paths to tolerate both spawn paths.
        """
        entity = self._entities[env_idx]
        entity_handle = int(entity.get_native_handle())
        if entity_handle < 0:
            entity_handle &= _UINT64_MAX
        manager = getattr(self._ps, "manager", None)
        if manager is None:
            return None
        return getattr(manager, "dexsim_meta", {}).get(entity_handle, {}).get("attr")

    def _set_newton_attr_meta(self, env_idx: int, physical_attr) -> None:
        """Mirror a :class:`dexsim.types.PhysicalAttr` onto the stored Newton meta.

        Newton only models a subset of physical attributes at runtime (mass,
        friction, restitution, contact_offset, COM, inertia); the remaining
        fields (damping, ccd, sleep thresholds, solver iters, ...) are carried
        as metadata for rebuild and for getter consistency. This helper keeps
        that mirror in sync so :meth:`get_damping` / :meth:`get_mass` and the
        next scene rebuild see the user's intent.
        """
        attr = self._get_newton_attr(env_idx)
        for name in (
            "mass",
            "density",
            "dynamic_friction",
            "static_friction",
            "restitution",
            "contact_offset",
            "rest_offset",
            "linear_damping",
            "angular_damping",
            "sleep_threshold",
            "enable_ccd",
            "max_depenetration_velocity",
            "min_position_iters",
            "min_velocity_iters",
            "max_linear_velocity",
            "max_angular_velocity",
        ):
            setattr(attr, name, getattr(physical_attr, name))

    def _warn_newton_unsupported(self, api_name: str) -> None:
        logger.log_warning(
            f"Newton backend does not support RigidObject.{api_name} runtime updates. "
            "Skipping this call."
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

        if self.is_spawn_bound:
            raise NotImplementedError(
                "DexSim Spawn does not expose rigid-body collision-filter batch "
                "updates yet. The filter must remain in the birth descriptor."
            )

        if is_newton_scene(self._ps):
            if self._data is not None and isinstance(
                self._data.body_view, NewtonRigidBodyView
            ):
                self._data.body_view.apply_collision_filter(filter_data, local_env_ids)
            else:
                entities = [self._entities[env_idx] for env_idx in local_env_ids]
                apply_collision_filter_for_entities(self._ps, entities, filter_data)
            return

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

        # Use backend view when pose writes are supported (Newton BUILDER/READY).
        if (
            self._data is not None
            and self._data.body_view.can_apply_pose
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
                "unfinalized; call SimulationManager.finalize_newton_physics() first."
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
                "call SimulationManager.finalize_newton_physics() first."
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

        if self.is_spawn_bound:
            raise NotImplementedError(
                "RigidObject.set_attrs() needs the remaining typed Spawn property "
                "batch APIs (friction/restitution/contact offset). Use the "
                "supported set_mass/set_inertia/set_com_pose methods meanwhile."
            )

        if isinstance(attrs, List) and len(local_env_ids) != len(attrs):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match attrs length {len(attrs)}."
            )

        # Resolve per-env physical attrs into a flat list aligned with local_env_ids.
        if isinstance(attrs, RigidBodyAttributesCfg):
            physical_attrs = [attrs.attr() for _ in local_env_ids]
        else:
            physical_attrs = [a.attr() for a in attrs]

        if is_newton_scene(self._ps):
            self._set_newton_attrs(physical_attrs, local_env_ids)
            return

        # TODO: maybe need to improve the physical attributes setter efficiency.
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].set_physical_attr(physical_attrs[i])

    def _set_newton_attrs(
        self,
        physical_attrs: list,
        local_env_ids,
    ) -> None:
        """Apply physical attributes on the Newton backend.

        Newton models only a subset of physical attributes at runtime
        (mass, friction, restitution, contact_offset); the rest (damping, ccd,
        sleep thresholds, solver iters, rest_offset, static_friction) are
        metadata carried for rebuild and getter consistency. When the Newton
        model is finalized (READY/STALE) the supported subset is pushed live
        via the batch scene API; beforehand (BUILDER) the attributes are only
        mirrored onto the meta so the next finalize consumes them.
        """
        for i, env_idx in enumerate(local_env_ids):
            self._set_newton_attr_meta(env_idx, physical_attrs[i])

        if self._data is None or not self._data.body_view.is_ready:
            logger.log_debug(
                "Newton model is not finalized; physical attributes are mirrored "
                "to metadata and applied at the next finalize_newton_physics()."
            )
            return

        body_ids = self._data.body_ids_for(local_env_ids)
        view = self._data.body_view
        device = self.device

        def _stack(field: str) -> torch.Tensor:
            return torch.as_tensor(
                [getattr(a, field) for a in physical_attrs],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(-1)

        # Newton-supported runtime subset.
        view.apply_mass(_stack("mass"), body_ids)
        view.apply_friction(_stack("dynamic_friction"), body_ids)
        view.apply_restitution(_stack("restitution"), body_ids)
        view.apply_contact_offset(_stack("contact_offset"), body_ids)

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

        if self._data is not None and self._data.body_view.is_ready:
            body_ids = self._data.body_ids_for(local_env_ids)
            self._data.body_view.apply_mass(
                mass.to(dtype=torch.float32, device=self.device).unsqueeze(-1),
                body_ids,
            )
            return

        mass_np = mass.cpu().numpy()
        for i, env_idx in enumerate(local_env_ids):
            if is_newton_scene(self._ps):
                # Not finalized: mirror to meta (consumed at next finalize). The
                # Default-backend set_mass is not patched for Newton entities.
                attr = self._get_newton_attr_or_none(env_idx)
                if attr is not None:
                    attr.mass = float(mass_np[i])
            else:
                self._entities[env_idx].get_physical_body().set_mass(mass_np[i])

    def get_mass(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get mass for the rigid object.

        Args:
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.

        Returns:
            torch.Tensor: The mass of the rigid object with shape (N,).
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if self._data is not None and self._data.body_view.is_ready:
            body_ids = self._data.body_ids_for(local_env_ids)
            buf = self._data._mass[: len(local_env_ids)]
            self._data.body_view.fetch_mass(buf, body_ids)
            return buf.squeeze(-1)

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

        if self._data is not None and self._data.body_view.is_ready:
            body_ids = self._data.body_ids_for(local_env_ids)
            self._data.body_view.apply_friction(
                friction.to(dtype=torch.float32, device=self.device).unsqueeze(-1),
                body_ids,
            )
            return

        friction_np = friction.cpu().numpy()
        for i, env_idx in enumerate(local_env_ids):
            if is_newton_scene(self._ps):
                # Not finalized: mirror to meta (Newton has a single mu; consumed
                # at next finalize). The Default-backend friction setters are not
                # patched for Newton entities.
                attr = self._get_newton_attr_or_none(env_idx)
                if attr is not None:
                    attr.dynamic_friction = float(friction_np[i])
            else:
                self._entities[env_idx].get_physical_body().set_dynamic_friction(
                    friction_np[i]
                )
                self._entities[env_idx].get_physical_body().set_static_friction(
                    friction_np[i]
                )

    def get_friction(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get friction for the rigid object.

        Args:
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.

        Returns:
            torch.Tensor: The friction of the rigid object with shape (N,).
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if self._data is not None and self._data.body_view.is_ready:
            body_ids = self._data.body_ids_for(local_env_ids)
            buf = self._data._friction[: len(local_env_ids)]
            self._data.body_view.fetch_friction(buf, body_ids)
            return buf.squeeze(-1)

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

        .. attention::
            The Newton backend does not simulate per-body linear/angular damping
            (its damping is a global solver knob). On Newton this call mirrors
            the values onto the attribute metadata so :meth:`get_damping` and
            scene rebuilds stay consistent, but has no runtime effect.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if self.is_spawn_bound:
            raise NotImplementedError(
                "DexSim Spawn does not expose rigid-body damping yet."
            )

        if len(local_env_ids) != len(damping):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match damping length {len(damping)}."
            )

        damping = damping.to(dtype=torch.float32, device=self.device)

        if is_newton_scene(self._ps):
            for i, env_idx in enumerate(local_env_ids):
                attr = self._get_newton_attr(env_idx)
                attr.linear_damping = float(damping[i, 0].item())
                attr.angular_damping = float(damping[i, 1].item())
            return

        damping_np = damping.cpu().numpy()
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].get_physical_body().set_linear_damping(
                damping_np[i, 0]
            )
            self._entities[env_idx].get_physical_body().set_angular_damping(
                damping_np[i, 1]
            )

    def get_damping(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get linear and angular damping for the rigid object.

        Args:
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.

        Returns:
            torch.Tensor: The damping of the rigid object with shape (N, 2), where the first column is linear damping and the second column is angular damping.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if self.is_spawn_bound:
            raise NotImplementedError(
                "DexSim Spawn does not expose rigid-body damping yet."
            )

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

        if self._data is not None and self._data.body_view.is_ready:
            body_ids = self._data.body_ids_for(local_env_ids)
            self._data.body_view.apply_inertia_diagonal(
                inertia.to(dtype=torch.float32, device=self.device),
                body_ids,
            )
            return

        inertia_np = inertia.cpu().numpy()
        for i, env_idx in enumerate(local_env_ids):
            if is_newton_scene(self._ps):
                # Not finalized: mirror to meta (consumed at next finalize). The
                # Default-backend inertia setter is not patched for Newton entities.
                attr = self._get_newton_attr_or_none(env_idx)
                if attr is not None:
                    attr.inertia = np.asarray(inertia_np[i], dtype=np.float32)
            else:
                self._entities[
                    env_idx
                ].get_physical_body().set_mass_space_inertia_tensor(inertia_np[i])

    def get_inertia(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get inertia tensor for the rigid object.

        Args:
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.

        Returns:
            torch.Tensor: The inertia tensor of the rigid object with shape (N, 3), where each row is the diagonal of the inertia tensor.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if self._data is not None and self._data.body_view.is_ready:
            body_ids = self._data.body_ids_for(local_env_ids)
            buf = self._data._inertia[: len(local_env_ids)]
            self._data.body_view.fetch_inertia_diagonal(buf, body_ids)
            return buf

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
        update_default: bool = False,
    ) -> None:
        """Set visual material for the rigid object.

        Note:
            If `shared` is True, the same material instance will be used for all specified environment indices.
            If `shared` is False, a unique material instance will be created for each specified environment index.

        Args:
            mat (VisualMaterial): The material to set.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
            shared (bool, optional): Whether to share the material instance among all specified environment indices. Defaults to False.
            update_default: Whether the assigned material should become the baseline
                restored by :meth:`reset`. Defaults to False.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if shared:
            if len(local_env_ids) != self.num_instances:
                logger.log_error(f"Cannot share material instance for partial env_ids.")

            mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}")
            for env_idx in local_env_ids:
                self._entities[env_idx].set_material(mat_inst.mat)
                self._visual_material[env_idx] = mat_inst
                if update_default:
                    self._original_visual_material[env_idx] = _capture_render_materials(
                        self._entities[env_idx].get_render_body()
                    )
                    self._original_visual_material_inst[env_idx] = mat_inst
            self.is_shared_visual_material = True
        else:
            for i, env_idx in enumerate(local_env_ids):
                mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}_{env_idx}")
                self._entities[env_idx].set_material(mat_inst.mat)
                self._visual_material[env_idx] = mat_inst
                if update_default:
                    self._original_visual_material[env_idx] = _capture_render_materials(
                        self._entities[env_idx].get_render_body()
                    )
                    self._original_visual_material_inst[env_idx] = mat_inst
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

    def _initialize_existing_visual_material(self) -> None:
        """Wrap asset-parsed materials during rigid-object construction.

        The public material list stores one representative material per
        environment. For a multi-segment render body, the first segment with a
        valid material is registered. Segment-specific materials remain
        available through :meth:`get_existing_visual_material`.
        """
        self._original_visual_material = [[] for _ in self._entities]
        self._original_visual_material_inst = [None] * len(self._entities)
        for env_idx, entity in enumerate(self._entities):
            render_body = entity.get_render_body()
            if render_body is None:
                continue
            original_materials = _capture_render_materials(render_body)
            self._original_visual_material[env_idx] = original_materials
            wrapped = _wrap_first_render_material(original_materials)
            if wrapped is not None:
                self._visual_material[env_idx] = wrapped
                self._original_visual_material_inst[env_idx] = wrapped

    def restore_visual_material(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore the visual materials captured when the rigid object was created.

        Args:
            env_ids: Environment indices. If None, all instances are restored.
        """
        if not hasattr(self, "_original_visual_material"):
            return
        local_env_ids = self._all_indices if env_ids is None else env_ids
        for env_idx in local_env_ids:
            render_body = self._entities[env_idx].get_render_body()
            if render_body is None:
                continue
            _restore_render_materials(
                render_body, self._original_visual_material[env_idx]
            )
            self._visual_material[env_idx] = self._original_visual_material_inst[
                env_idx
            ]
        self.is_shared_visual_material = False

    def get_existing_visual_material(
        self,
        env_ids: Sequence[int] | None = None,
        shared: bool = False,
    ) -> List[List[ReuseSegmentState]]:
        """Build reuse state from the material dexsim parsed onto each env's render body.

        For each env (first only if ``shared``), every render-body segment's existing
        ``MaterialInst`` is captured as an immutable original. One working instance is
        shared by all segments so randomized updates have constant material-update cost.

        Args:
            env_ids: Environment indices. If None, all instances are used.
            shared: If True, build state for the first env only (caller applies it to all).

        Returns:
            Per-env list of per-segment :obj:`ReuseSegmentState` (length 1 if ``shared``).

        Raises:
            ValueError: If a segment has no material or no retrievable template.
        """
        if shared:
            local_env_ids = [self._all_indices[0]]
        else:
            local_env_ids = self._all_indices if env_ids is None else list(env_ids)

        if not hasattr(self, "_original_visual_material"):
            self._original_visual_material = [None] * len(self._entities)
        for env_idx in local_env_ids:
            if self._original_visual_material[env_idx] is None:
                self._original_visual_material[env_idx] = _capture_render_materials(
                    self._entities[env_idx].get_render_body()
                )

        per_env: List[List[ReuseSegmentState]] = []
        for env_idx in local_env_ids:
            segments: List[ReuseSegmentState] = []
            working_inst = None
            for mesh_id, original_inst in enumerate(
                self._original_visual_material[env_idx]
            ):
                if original_inst is None:
                    raise ValueError(
                        f"RigidObject '{self.uid}' env {env_idx} segment {mesh_id} has no material."
                    )
                template = original_inst.get_template()
                if template is None:
                    raise ValueError(
                        f"RigidObject '{self.uid}' segment {mesh_id} material has no template."
                    )
                if working_inst is None:
                    working_name = f"{self.uid}_reuse_{env_idx}"
                    template.create_inst(working_name)
                    working_inst = VisualMaterialInst(working_name, template)
                segments.append(
                    ReuseSegmentState(
                        mesh_id=mesh_id,
                        original_inst=original_inst,
                        working_inst=working_inst,
                    )
                )
            per_env.append(segments)
        return per_env

    def apply_render_material_inst(
        self,
        env_idx: int,
        mat_inst: MaterialInst,
        mesh_id: int = 0,
    ) -> None:
        """Swap a dexsim MaterialInst onto a render-body segment for the given env.

        Args:
            env_idx: Environment index.
            mat_inst: dexsim ``MaterialInst`` to attach.
            mesh_id: Render-body segment index.
        """
        _set_render_material(
            self._entities[env_idx].get_render_body(), mesh_id, mat_inst
        )

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

        if self._data is not None:
            target_com_pose = com_pose.to(device=self.device, dtype=torch.float32)
            body_ids = self._data.body_ids_for(local_env_ids)
            self._data.body_view.apply_com_local_pose(target_com_pose, body_ids)
            return

        logger.log_error("Cannot set center of mass pose before body view is ready.")

    def set_body_type(self, body_type: str) -> None:
        """Set the body type of the rigid object.

        Note:
            Only 'dynamic' and 'kinematic' body types are supported and can be changed at runtime.

        Args:
            body_type (str): The body type to set. Must be one of 'dynamic', or 'kinematic'.

        .. attention::
            On the Newton backend, body type (dynamic/kinematic/static) is fixed
            at body registration and cannot be changed at runtime; switching it
            would require re-registering the body and rebuilding the model. This
            call is therefore a no-op on Newton.
        """
        from dexsim.types import ActorType

        if self.is_spawn_bound:
            raise NotImplementedError(
                "Changing actor topology after Spawn binding requires a public "
                "descriptor mutation transaction and is not implemented yet."
            )

        if is_newton_scene(self._ps):
            logger.log_warning(
                "Newton backend does not support changing RigidObject body type at "
                "runtime (it is fixed at registration). Skipping set_body_type call."
            )
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
        """Retrieve the combined visual-mesh vertices of the rigid objects.

        Assets such as GLB files can contain multiple render meshes. Their
        vertices are concatenated in render-mesh order so the result represents
        the complete object instead of only the first mesh.

        Args:
            env_ids: Environment IDs for which to retrieve vertices. If ``None``,
                retrieves vertices for all instances.
            scale: Whether to multiply the vertices by the body scale.

        Returns:
            Combined vertices with shape ``(N, num_vertices, 3)``.
        """
        ids = env_ids if env_ids is not None else range(self.num_instances)
        verts = torch.as_tensor(
            np.array(
                [get_combined_vertices(self._entities[id]) for id in ids],
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if scale:
            verts = verts * self.get_body_scale(env_ids).unsqueeze_(1)
        return verts

    def get_triangles(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Retrieve triangle indices for the combined visual meshes.

        Face indices from each render mesh are offset to reference the
        concatenated vertex array returned by :meth:`get_vertices`.

        Args:
            env_ids: Environment IDs for which to retrieve triangle indices. If
                ``None``, retrieves triangle indices for all instances.

        Returns:
            Triangle indices with shape ``(N, num_triangles, 3)``.
        """
        ids = env_ids if env_ids is not None else range(self.num_instances)
        return torch.as_tensor(
            np.array(
                [get_combined_triangles(self._entities[id]) for id in ids],
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
                "call SimulationManager.finalize_newton_physics() first."
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

        if self.is_spawn_bound:
            color = np.asarray(rgba, dtype=np.float32)
            for entity in self._entities:
                self._spawn_result.set_physical_visible(entity, color, visible)
            self._has_collision_visible_node = True
            return

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

    def _build_cfg_init_pose(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Build initial root poses from cfg as ``(N, 4, 4)`` matrices."""
        num_instances = len(env_ids)
        if self.cfg.init_local_pose is not None:
            return (
                torch.as_tensor(
                    self.cfg.init_local_pose,
                    dtype=torch.float32,
                    device=self.device,
                )
                .reshape(1, 4, 4)
                .repeat(num_instances, 1, 1)
            )
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
        return pose

    def _apply_initial_state(self) -> None:
        """Apply cfg initial pose after construction.

        The Default (DexSim) backend runs a full reset. Newton applies init pose in
        ``BUILDER`` via the scene batch API; velocities are cleared after
        finalization through :meth:`SimulationManager.finalize_newton_physics`.
        """
        if self.is_spawn_bound:
            if self._spawn_result.backend == "dexsim":
                # DexSim Direct GPU readiness performs native warm-up updates.
                # Re-apply the authored state after the batch becomes usable
                # so prepare() itself is not an observable simulation step.
                self.reset()
            else:
                # Newton finalization materializes the descriptor pose without
                # advancing simulation; only one-step dynamics buffers need
                # clearing after batch binding.
                self.clear_dynamics()
            return

        if is_newton_scene(self._ps):
            if self._newton_lifecycle_state() == "BUILDER":
                self.set_local_pose(
                    self._build_cfg_init_pose(self._all_indices),
                    env_ids=self._all_indices,
                )
            return

        if self.device.type == "cuda":
            self._world.update(0.001)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        local_env_ids = self._all_indices if env_ids is None else env_ids

        self.restore_visual_material(env_ids=local_env_ids)

        # Spawn descriptors and their live property APIs are the canonical
        # physical configuration; reset changes state only.
        if not self.is_spawn_bound and not is_newton_scene(self._ps):
            self.set_attrs(self.cfg.attrs, env_ids=local_env_ids)

        self.clear_dynamics(env_ids=local_env_ids)

        self.set_local_pose(
            self._build_cfg_init_pose(local_env_ids), env_ids=local_env_ids
        )

    def destroy(self) -> None:
        if self.is_declared or self.is_spawn_bound:
            # SimulationManager owns topology removal and SpawnResult lifetime.
            # Direct facade destruction must never bypass that owner.
            return
        env = self._world.get_env()
        arenas = env.get_all_arenas()
        if len(arenas) == 0:
            arenas = [env]
        for i, entity in enumerate(self._entities):
            if is_newton_scene(self._ps):
                arenas[i].remove_actor(entity.get_name())
            else:
                arenas[i].remove_actor(entity)
