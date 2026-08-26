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

from copy import deepcopy
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from embodichain.lab.sim import BatchEntity
from embodichain.lab.sim.cfg import RigidObjectGroupCfg
from embodichain.lab.sim.material import VisualMaterial
from embodichain.lab.sim.objects.backends.spawn import SpawnRigidBodyView
from embodichain.utils.math import (
    convert_quat,
    matrix_from_euler,
    matrix_from_quat,
    quat_from_matrix,
)

from ._mesh_utils import get_combined_triangles, get_combined_vertices

if TYPE_CHECKING:
    from dexsim.spawn import SpawnResult, SpawnedObject

__all__ = ["RigidBodyGroupData", "RigidObjectGroup", "RigidObjectGroupCfg"]


class RigidBodyGroupData:
    """Expose one flat Spawn rigid-body batch as ``[env, object, ...]`` tensors."""

    def __init__(
        self,
        body_view: SpawnRigidBodyView,
        *,
        num_instances: int,
        num_objects: int,
        device: torch.device,
    ) -> None:
        self.body_view = body_view
        self.num_instances = num_instances
        self.num_objects = num_objects
        self.device = device
        self._pose = torch.empty(
            (num_instances, num_objects, 7), dtype=torch.float32, device=device
        )
        self._lin_vel = torch.empty(
            (num_instances, num_objects, 3), dtype=torch.float32, device=device
        )
        self._ang_vel = torch.empty_like(self._lin_vel)
        self._mass = torch.empty(
            (num_instances, num_objects, 1),
            dtype=torch.float32,
            device=device,
        )
        self._inertia = torch.empty(
            (num_instances, num_objects, 3),
            dtype=torch.float32,
            device=device,
        )
        self._com_pose = torch.empty(
            (num_instances, num_objects, 7),
            dtype=torch.float32,
            device=device,
        )
        self._default_mass: torch.Tensor | None = None
        self._default_inertia: torch.Tensor | None = None
        self._default_com_pose: torch.Tensor | None = None

    @property
    def pose(self) -> torch.Tensor:
        """Local poses in the legacy Group layout ``xyz + wxyz``."""
        flat = self._pose.reshape(-1, 7)
        self.body_view.fetch_pose(flat)
        flat[:, 3:7] = convert_quat(flat[:, 3:7], to="wxyz")
        return self._pose

    @property
    def lin_vel(self) -> torch.Tensor:
        self.body_view.fetch_linear_velocity(self._lin_vel.reshape(-1, 3))
        return self._lin_vel

    @property
    def ang_vel(self) -> torch.Tensor:
        self.body_view.fetch_angular_velocity(self._ang_vel.reshape(-1, 3))
        return self._ang_vel

    @property
    def vel(self) -> torch.Tensor:
        """Linear and angular velocities with shape ``[env, object, 6]``."""
        return torch.cat((self.lin_vel, self.ang_vel), dim=-1)

    @property
    def mass(self) -> torch.Tensor:
        """Current masses with shape ``[env, object]``."""
        self.body_view.fetch_mass(self._mass.reshape(-1, 1))
        return self._mass.squeeze(-1)

    @property
    def inertia(self) -> torch.Tensor:
        """Current inertia diagonals with shape ``[env, object, 3]``."""
        self.body_view.fetch_inertia_diagonal(self._inertia.reshape(-1, 3))
        return self._inertia

    @property
    def com_pose(self) -> torch.Tensor:
        """Current local COM poses in Group ``xyz + wxyz`` convention."""
        flat = self._com_pose.reshape(-1, 7)
        self.body_view.fetch_com_local_pose(flat)
        flat[:, 3:7] = convert_quat(flat[:, 3:7], to="wxyz")
        return self._com_pose

    @property
    def default_physical_properties_initialized(self) -> bool:
        """Whether initialization-time mass properties are available."""
        return (
            self._default_mass is not None
            and self._default_inertia is not None
            and self._default_com_pose is not None
        )

    @property
    def default_mass(self) -> torch.Tensor:
        """Initialization-time masses with shape ``[env, object]``."""
        if self._default_mass is None:
            raise RuntimeError("Default rigid-object Group masses are unavailable.")
        return self._default_mass

    @property
    def default_inertia(self) -> torch.Tensor:
        """Initialization-time inertia diagonals."""
        if self._default_inertia is None:
            raise RuntimeError("Default rigid-object Group inertias are unavailable.")
        return self._default_inertia

    @property
    def default_com_pose(self) -> torch.Tensor:
        """Initialization-time local COM poses in ``xyz + wxyz`` order."""
        if self._default_com_pose is None:
            raise RuntimeError("Default rigid-object Group COM poses are unavailable.")
        return self._default_com_pose

    def capture_default_physical_properties(
        self,
        *,
        mass: torch.Tensor,
        inertia: torch.Tensor,
        com_pose: torch.Tensor,
    ) -> None:
        """Capture backend-resolved Group mass properties exactly once."""
        expected_shapes = {
            "mass": (self.num_instances, self.num_objects),
            "inertia": (self.num_instances, self.num_objects, 3),
            "com_pose": (self.num_instances, self.num_objects, 7),
        }
        values = {"mass": mass, "inertia": inertia, "com_pose": com_pose}
        for name, value in values.items():
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"Expected {name} shape {expected_shapes[name]}, "
                    f"got {tuple(value.shape)}."
                )
        if self.default_physical_properties_initialized:
            raise RuntimeError(
                "Default rigid-object Group mass properties are already captured."
            )

        self._default_mass = mass.to(self.device, dtype=torch.float32).clone()
        self._default_inertia = inertia.to(self.device, dtype=torch.float32).clone()
        self._default_com_pose = com_pose.to(self.device, dtype=torch.float32).clone()


class RigidObjectGroup(BatchEntity):
    """A two-dimensional view over rigid objects owned by DexSim Spawn."""

    def __init__(
        self,
        cfg: RigidObjectGroupCfg,
        entities: Sequence[Sequence[SpawnedObject]] | None = None,
        device: torch.device = torch.device("cpu"),
        *,
        spawn_result: SpawnResult | None = None,
        declared_num_instances: int | None = None,
    ) -> None:
        self.body_type = cfg.body_type
        self._declared_num_objects = len(cfg.rigid_objects)

        if entities is None:
            if declared_num_instances is None or declared_num_instances <= 0:
                raise ValueError(
                    "A declared RigidObjectGroup requires declared_num_instances > 0."
                )
            self.cfg = deepcopy(cfg)
            self.uid = self.cfg.uid
            self.device = device
            self._entities: list[list[SpawnedObject]] = []
            self._declared_num_instances = declared_num_instances
            self._spawn_result = None
            self._data = None
            self._all_indices = list(range(declared_num_instances))
            self._all_obj_indices = list(range(self._declared_num_objects))
            return

        rows = [list(instance) for instance in entities]
        if not rows or any(
            len(instance) != self._declared_num_objects for instance in rows
        ):
            raise ValueError(
                "RigidObjectGroup Spawn handles must have shape "
                "[num_instances, num_objects]."
            )
        if spawn_result is None:
            raise ValueError(
                "RigidObjectGroup entities must be owned by a SpawnResult."
            )

        self._declared_num_instances = len(rows)
        self._spawn_result = spawn_result
        self._all_indices = list(range(len(rows)))
        self._all_obj_indices = list(range(self._declared_num_objects))
        flat_entities = [entity for instance in rows for entity in instance]
        batch = spawn_result.create_rigid_body_batch(flat_entities)
        body_view = SpawnRigidBodyView(spawn_result, batch, device)
        self._data = RigidBodyGroupData(
            body_view,
            num_instances=len(rows),
            num_objects=self._declared_num_objects,
            device=device,
        )

        super().__init__(cfg, rows, device, auto_reset=False)
        self._capture_default_physical_properties()
        self.reset()

    @property
    def is_declared(self) -> bool:
        """Whether this facade is waiting for Spawn materialization."""
        return self._spawn_result is None

    @property
    def is_spawn_bound(self) -> bool:
        """Whether this facade is bound to a SpawnResult."""
        return self._spawn_result is not None

    @property
    def num_instances(self) -> int:
        return len(self._entities) if self._entities else self._declared_num_instances

    @property
    def num_objects(self) -> int:
        return self._declared_num_objects

    @property
    def body_data(self) -> RigidBodyGroupData:
        if self._data is None:
            raise RuntimeError(
                f"RigidObjectGroup {self.uid!r} is not bound; call SimulationManager.prepare()."
            )
        return self._data

    def _capture_default_physical_properties(self) -> None:
        """Capture materialized Group mass properties as reset defaults."""
        data = self.body_data
        if data.default_physical_properties_initialized:
            return
        data.capture_default_physical_properties(
            mass=data.mass,
            inertia=data.inertia,
            com_pose=data.com_pose,
        )

    def _restore_default_physical_properties(
        self, env_ids: Sequence[int] | torch.Tensor | None
    ) -> None:
        """Restore initialization-time Group mass properties for selected rows."""
        data = self.body_data
        if self.is_non_dynamic or not data.default_physical_properties_initialized:
            return
        env, objects, _ = self._selected_indices(env_ids)
        if not env:
            return
        env_index = torch.as_tensor(env, dtype=torch.long, device=self.device)
        obj_index = torch.as_tensor(objects, dtype=torch.long, device=self.device)
        self.set_mass(
            data.default_mass[env_index[:, None], obj_index[None, :]],
            env_ids=env,
            obj_ids=objects,
        )
        self.set_inertia(
            data.default_inertia[env_index[:, None], obj_index[None, :]],
            env_ids=env,
            obj_ids=objects,
        )
        self.set_com_pose(
            data.default_com_pose[env_index[:, None], obj_index[None, :]],
            env_ids=env,
            obj_ids=objects,
        )

    @property
    def body_state(self) -> torch.Tensor:
        """Pose and velocity with shape ``[env, object, 13]``."""
        return torch.cat(
            (self.body_data.pose, self.body_data.lin_vel, self.body_data.ang_vel),
            dim=-1,
        )

    @property
    def is_non_dynamic(self) -> bool:
        return self.body_type in ("static", "kinematic")

    def attach_spawn_handles(self, entities: Sequence[SpawnedObject]) -> None:
        """Store env-major handles without initializing the group's Batch data.

        ``bind_spawn()`` creates the result-dependent runtime view after Spawn
        finalization.
        """
        expected = self._declared_num_instances * self.num_objects
        if len(entities) != expected:
            raise ValueError(
                f"RigidObjectGroup {self.uid!r} expected {expected} Spawn handles, "
                f"got {len(entities)}."
            )
        self._entities = [
            list(entities[start : start + self.num_objects])
            for start in range(0, len(entities), self.num_objects)
        ]

    def bind_spawn(self, result: SpawnResult) -> None:
        """Atomically bind the declaration facade to env-major Spawn handles."""
        if self.is_spawn_bound:
            raise RuntimeError(f"RigidObjectGroup {self.uid!r} is already Spawn-bound.")
        if not self.is_declared:
            raise RuntimeError(
                f"RigidObjectGroup {self.uid!r} was not created as a Spawn declaration."
            )

        cfg = self.cfg
        device = self.device
        rows = [list(row) for row in self._entities]
        if len(rows) != self._declared_num_instances or any(
            len(row) != self.num_objects for row in rows
        ):
            raise ValueError(
                f"RigidObjectGroup {self.uid!r} expected "
                f"{self._declared_num_instances}x{self.num_objects} Spawn handles."
            )

        bound = type(self)(
            cfg,
            rows,
            device,
            spawn_result=result,
        )
        self.__dict__.clear()
        self.__dict__.update(bound.__dict__)

    def __str__(self) -> str:
        if self.is_declared:
            return (
                f"{self.__class__}: declared {self.num_instances}x{self.num_objects} "
                f"Spawn objects | uid: {self.uid} | device: {self.device}"
            )
        return (
            super().__str__()
            + f" | body type: {self.body_type} | num_objects: {self.num_objects}"
        )

    def _selected_indices(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        obj_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[list[int], list[int], torch.Tensor]:
        env = (
            self._all_indices
            if env_ids is None
            else torch.as_tensor(env_ids).reshape(-1).cpu().tolist()
        )
        objects = (
            self._all_obj_indices
            if obj_ids is None
            else torch.as_tensor(obj_ids).reshape(-1).cpu().tolist()
        )
        if any(index < 0 or index >= self.num_instances for index in env):
            raise IndexError("RigidObjectGroup environment index is out of range.")
        if any(index < 0 or index >= self.num_objects for index in objects):
            raise IndexError("RigidObjectGroup object index is out of range.")
        rows = torch.as_tensor(
            [
                env_id * self.num_objects + obj_id
                for env_id in env
                for obj_id in objects
            ],
            dtype=torch.long,
            device=self.device,
        )
        return env, objects, rows

    def get_mass(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        obj_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return selected masses with shape ``[env, object]``."""
        env, objects, _ = self._selected_indices(env_ids, obj_ids)
        env_index = torch.as_tensor(env, dtype=torch.long, device=self.device)
        obj_index = torch.as_tensor(objects, dtype=torch.long, device=self.device)
        return self.body_data.mass[env_index[:, None], obj_index[None, :]]

    def set_mass(
        self,
        mass: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        obj_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set selected masses from a tensor shaped ``[env, object]``."""
        env, objects, rows = self._selected_indices(env_ids, obj_ids)
        mass = torch.as_tensor(mass, dtype=torch.float32, device=self.device)
        expected_shape = (len(env), len(objects))
        if tuple(mass.shape) != expected_shape:
            raise ValueError(
                f"Expected mass shape {expected_shape}, got {tuple(mass.shape)}."
            )
        self.body_data.body_view.apply_mass(mass.reshape(-1, 1), rows)

    def get_inertia(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        obj_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return selected inertia diagonals with shape ``[env, object, 3]``."""
        env, objects, _ = self._selected_indices(env_ids, obj_ids)
        env_index = torch.as_tensor(env, dtype=torch.long, device=self.device)
        obj_index = torch.as_tensor(objects, dtype=torch.long, device=self.device)
        return self.body_data.inertia[env_index[:, None], obj_index[None, :]]

    def set_inertia(
        self,
        inertia: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        obj_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set selected inertia diagonals."""
        env, objects, rows = self._selected_indices(env_ids, obj_ids)
        inertia = torch.as_tensor(inertia, dtype=torch.float32, device=self.device)
        expected_shape = (len(env), len(objects), 3)
        if tuple(inertia.shape) != expected_shape:
            raise ValueError(
                f"Expected inertia shape {expected_shape}, "
                f"got {tuple(inertia.shape)}."
            )
        self.body_data.body_view.apply_inertia_diagonal(inertia.reshape(-1, 3), rows)

    def get_com_pose(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        obj_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return selected local COM poses in Group ``xyz + wxyz`` order."""
        env, objects, _ = self._selected_indices(env_ids, obj_ids)
        env_index = torch.as_tensor(env, dtype=torch.long, device=self.device)
        obj_index = torch.as_tensor(objects, dtype=torch.long, device=self.device)
        return self.body_data.com_pose[env_index[:, None], obj_index[None, :]]

    def set_com_pose(
        self,
        com_pose: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        obj_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set selected local COM poses in Group ``xyz + wxyz`` order."""
        env, objects, rows = self._selected_indices(env_ids, obj_ids)
        com_pose = torch.as_tensor(com_pose, dtype=torch.float32, device=self.device)
        expected_shape = (len(env), len(objects), 7)
        if tuple(com_pose.shape) != expected_shape:
            raise ValueError(
                f"Expected COM pose shape {expected_shape}, "
                f"got {tuple(com_pose.shape)}."
            )
        flat = com_pose.reshape(-1, 7)
        target = torch.cat(
            (flat[:, :3], convert_quat(flat[:, 3:7], to="xyzw")),
            dim=-1,
        )
        self.body_data.body_view.apply_com_local_pose(target, rows)

    def set_collision_filter(
        self,
        filter_data: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set one collision filter value for every selected member in each env."""
        env, objects, rows = self._selected_indices(env_ids)
        values = filter_data.to(device=self.device, dtype=torch.int32).reshape(-1, 4)
        if len(values) != len(env):
            raise ValueError(
                f"Expected {len(env)} collision filters, got {len(values)}."
            )
        expanded = values[:, None, :].expand(-1, len(objects), -1).reshape(-1, 4)
        self.body_data.body_view.apply_collision_filter(expanded, rows)

    def set_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        obj_ids: Sequence[int] | None = None,
    ) -> None:
        """Set Group poses in ``xyz+wxyz`` or homogeneous-matrix form."""
        env, objects, rows = self._selected_indices(env_ids, obj_ids)
        expected_prefix = (len(env), len(objects))
        pose = pose.to(device=self.device, dtype=torch.float32)
        if tuple(pose.shape) == (*expected_prefix, 7):
            flat = pose.reshape(-1, 7)
            target = torch.cat(
                (flat[:, :3], convert_quat(flat[:, 3:7], to="xyzw")), dim=-1
            )
        elif tuple(pose.shape) == (*expected_prefix, 4, 4):
            flat = pose.reshape(-1, 4, 4)
            target = torch.cat(
                (
                    flat[:, :3, 3],
                    convert_quat(quat_from_matrix(flat[:, :3, :3]), to="xyzw"),
                ),
                dim=-1,
            )
        else:
            raise ValueError(
                f"Expected pose shape {(*expected_prefix, 7)} or "
                f"{(*expected_prefix, 4, 4)}, got {tuple(pose.shape)}."
            )
        self.body_data.body_view.apply_pose(target, rows)

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Return all Group poses as ``xyz+wxyz`` or homogeneous matrices."""
        pose = self.body_data.pose
        if not to_matrix:
            return pose
        flat = pose.reshape(-1, 7)
        result = torch.eye(4, dtype=torch.float32, device=self.device).repeat(
            len(flat), 1, 1
        )
        result[:, :3, 3] = flat[:, :3]
        result[:, :3, :3] = matrix_from_quat(flat[:, 3:7])
        return result.reshape(self.num_instances, self.num_objects, 4, 4)

    def get_object_vertices(
        self,
        object_id: int,
        env_ids: Sequence[int] | None = None,
        scale: bool = False,
    ) -> torch.Tensor:
        """Return one member's render vertices across selected environments."""
        env, objects, _ = self._selected_indices(env_ids, [object_id])
        object_id = objects[0]
        vertices = np.asarray(
            [get_combined_vertices(self._entities[index][object_id]) for index in env],
            dtype=np.float32,
        )
        if scale:
            scales = np.asarray(
                [self._entities[index][object_id].get_body_scale() for index in env],
                dtype=np.float32,
            )
            vertices *= scales[:, None, :]
        return torch.as_tensor(vertices, dtype=torch.float32, device=self.device)

    def get_object_triangles(
        self,
        object_id: int,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return one member's render triangles across selected environments."""
        env, objects, _ = self._selected_indices(env_ids, [object_id])
        object_id = objects[0]
        triangles = np.asarray(
            [get_combined_triangles(self._entities[index][object_id]) for index in env],
            dtype=np.int32,
        )
        return torch.as_tensor(triangles, dtype=torch.int32, device=self.device)

    def get_user_ids(self) -> torch.Tensor:
        """Return render user ids with shape ``[env, object]``."""
        return torch.as_tensor(
            [
                [entity.get_user_id() for entity in instance]
                for instance in self._entities
            ],
            dtype=torch.int32,
            device=self.device,
        )

    def clear_dynamics(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear velocity and one-step wrench buffers for selected envs."""
        if self.is_non_dynamic:
            return
        _, _, rows = self._selected_indices(env_ids)
        zeros = torch.zeros((len(rows), 3), dtype=torch.float32, device=self.device)
        view = self.body_data.body_view
        view.apply_linear_velocity(zeros, rows)
        view.apply_angular_velocity(zeros, rows)
        view.apply_force(zeros, rows)
        view.apply_torque(zeros, rows)

    def set_visual_material(
        self,
        mat: VisualMaterial,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Assign one material instance to all members in each selected env."""
        env, _, _ = self._selected_indices(env_ids)
        for env_id in env:
            material = mat.create_instance(f"{mat.uid}_{self.uid}_{env_id}")
            for entity in self._entities[env_id]:
                entity.set_material(material.mat)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env, _, _ = self._selected_indices(env_ids)
        self._restore_default_physical_properties(env)
        member_poses = []
        for cfg in self.cfg.rigid_objects.values():
            if cfg.init_local_pose is not None:
                member_poses.append(
                    torch.as_tensor(
                        cfg.init_local_pose,
                        dtype=torch.float32,
                        device=self.device,
                    ).reshape(4, 4)
                )
                continue
            pose = torch.eye(4, dtype=torch.float32, device=self.device)
            pose[:3, 3] = torch.as_tensor(
                cfg.init_pos, dtype=torch.float32, device=self.device
            )
            rotation = torch.as_tensor(
                cfg.init_rot, dtype=torch.float32, device=self.device
            )
            pose[:3, :3] = matrix_from_euler(
                (rotation * torch.pi / 180.0).reshape(1, 3), "XYZ"
            )[0]
            member_poses.append(pose)
        pose = torch.stack(member_poses).repeat(len(env), 1, 1)
        self.set_local_pose(pose.reshape(len(env), self.num_objects, 4, 4), env_ids=env)
        self.clear_dynamics(env_ids=env)

    def set_physical_visible(
        self,
        visible: bool = True,
        rgba: Sequence[float] | None = None,
    ) -> None:
        """Set collision-geometry visibility for every Group member."""
        color = np.asarray(
            (0.8, 0.2, 0.2, 0.7) if rgba is None else rgba,
            dtype=np.float32,
        )
        if color.shape != (4,):
            raise ValueError("Collision visualization color must contain four values.")
        for instance in self._entities:
            for entity in instance:
                self._spawn_result.set_physical_visible(entity, color, visible)

    def set_visible(self, visible: bool = True) -> None:
        """Set render visibility for every Group member."""
        for instance in self._entities:
            for entity in instance:
                entity.set_visible(visible)

    def destroy(self) -> None:
        """Leave topology destruction to SimulationManager and SpawnResult."""
