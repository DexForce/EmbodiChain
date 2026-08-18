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
        self.reset()

    @property
    def is_declared(self) -> bool:
        """Whether this facade is waiting for Spawn materialization."""
        return self._spawn_result is None and not self._entities

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

    def bind_spawn(
        self,
        result: SpawnResult,
        entities: Sequence[SpawnedObject],
    ) -> None:
        """Bind the declaration facade to env-major Spawn handles in place."""
        if self.is_spawn_bound:
            raise RuntimeError(f"RigidObjectGroup {self.uid!r} is already Spawn-bound.")
        expected = self.num_instances * self.num_objects
        if len(entities) != expected:
            raise ValueError(
                f"RigidObjectGroup {self.uid!r} expected {expected} Spawn handles, "
                f"got {len(entities)}."
            )
        rows = [
            entities[start : start + self.num_objects]
            for start in range(0, expected, self.num_objects)
        ]
        bound = RigidObjectGroup(
            self.cfg,
            rows,
            self.device,
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

    def set_collision_filter(
        self,
        filter_data: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set one PhysX collision filter value for every member in each env."""
        env, _, _ = self._selected_indices(env_ids)
        values = np.asarray(filter_data.detach().cpu(), dtype=np.uint32).reshape(-1, 4)
        if len(values) != len(env):
            raise ValueError(
                f"Expected {len(env)} collision filters, got {len(values)}."
            )
        for row, env_id in enumerate(env):
            for entity in self._entities[env_id]:
                entity.get_physical_body().set_collision_filter_data(values[row])

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
