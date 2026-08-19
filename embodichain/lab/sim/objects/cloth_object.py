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
from functools import cached_property

from dataclasses import dataclass
from typing import Any, List, Sequence, TYPE_CHECKING, Union

from dexsim.models import MeshObject
from dexsim.engine import ClothBody, PhysicsScene
from dexsim.types import ClothBodyGPUAPIReadWriteType
from scipy.spatial import cKDTree
from embodichain.lab.sim.common import (
    BatchEntity,
)
from embodichain.lab.sim.material import (
    VisualMaterial,
    VisualMaterialInst,
    _capture_render_materials,
    _restore_render_materials,
    _wrap_first_render_material,
)
from embodichain.utils.math import (
    matrix_from_euler,
)
from embodichain.utils import logger
from embodichain.lab.sim.cfg import (
    ClothObjectCfg,
)
from embodichain.utils.math import xyz_quat_to_4x4_matrix

if TYPE_CHECKING:
    from dexsim.spawn import SpawnResult

__all__ = ["ClothBodyData", "ClothObject", "ClothObjectCfg"]


@dataclass
class ClothBodyData:
    """Data manager for cloth.

    Note:
        1. The pose data managed by dexsim is in the format of (qx, qy, qz, qw, x, y, z), but in EmbodiChain, we use (x, y, z, qw, qx, qy, qz) format.
    """

    def __init__(
        self, entities: List[MeshObject], ps: PhysicsScene, device: torch.device
    ) -> None:
        """Initialize the ClothBodyData.

        Args:
            entities (List[MeshObject]): List of MeshObjects representing the cloth bodies.
            ps (PhysicsScene): The physics scene.
            device (torch.device): The device to use for the cloth body data.
        """
        self.entities = entities
        # TODO: cloth body data can only be stored in cuda device for now.
        self.device = device
        # TODO: inorder to retrieve arena position, we need to access the node of each entity.
        self.ps = ps
        self.num_instances = len(entities)

        self.cloth_bodies: Sequence[ClothBody] = [
            self.entities[i].get_physical_body() for i in range(self.num_instances)
        ]
        self.n_vertices = self.cloth_bodies[0].get_num_vertices()

        self._rest_position_buffer = torch.empty(
            (self.num_instances, self.n_vertices, 4),
            device=self.device,
            dtype=torch.float32,
        )
        for i, cloth_body in enumerate(self.cloth_bodies):
            self._rest_position_buffer[i] = cloth_body.get_rest_position_buffer()

        self._vertex_position = torch.zeros(
            (self.num_instances, self.n_vertices, 3),
            device=self.device,
            dtype=torch.float32,
        )

        self._vertex_velocity = torch.zeros(
            (self.num_instances, self.n_vertices, 3),
            device=self.device,
            dtype=torch.float32,
        )

    @property
    def rest_vertices(self):
        """Get the rest position buffer of the cloth bodies."""
        return self._rest_position_buffer[:, :, :3].clone()

    @property
    def vertex_position(self):
        """Get the current vertex position buffer of the cloth bodies."""
        for i, clothbody in enumerate(self.cloth_bodies):
            self._vertex_position[i] = clothbody.get_position_inv_mass_buffer()[:, :3]
        return self._vertex_position.clone()

    @property
    def vertex_velocity(self):
        """Get the current vertex velocity buffer of the cloth bodies."""
        for i, clothbody in enumerate(self.cloth_bodies):
            self._vertex_velocity[i] = clothbody.get_velocity_buffer()[:, 3:]
        return self._vertex_velocity.clone()


class ClothObject(BatchEntity):
    """ClothObject represents a batch of cloth body in the simulation."""

    def __init__(
        self,
        cfg: ClothObjectCfg,
        entities: Sequence[Any] | None = None,
        device: torch.device = torch.device("cpu"),
        *,
        spawn_result: SpawnResult | None = None,
        declared_num_instances: int | None = None,
    ) -> None:
        if entities is None:
            if declared_num_instances is None or declared_num_instances <= 0:
                raise ValueError(
                    "A declared ClothObject requires declared_num_instances > 0."
                )
            self.cfg = deepcopy(cfg)
            self.uid = self.cfg.uid
            self.device = device
            self._entities = []
            self._declared_num_instances = declared_num_instances
            self._spawn_result = None
            self._world = None
            self._ps = None
            self._data = None
            self._all_indices = list(range(declared_num_instances))
            self._visual_material = [None] * declared_num_instances
            self.is_shared_visual_material = False
            return

        entities = list(entities)
        self._declared_num_instances = len(entities)
        self._spawn_result = spawn_result
        if spawn_result is None:
            self._world = dexsim.default_world()
            from embodichain.lab.sim.sim_manager import get_physics_scene

            self._ps = get_physics_scene()
        else:
            self._world = spawn_result.world
            self._ps = self._world.get_physics_scene()
        self._all_indices = torch.arange(len(entities), dtype=torch.int32).tolist()

        self._data = ClothBodyData(entities=entities, ps=self._ps, device=device)

        if spawn_result is None:
            self._world.update(0.001)
        self._surface_triangles = self._build_surface_triangles(
            entities[0],
            self._data.rest_vertices[0].detach().cpu().numpy(),
            self._data.cloth_bodies[0].get_initial_transform(),
        )

        self._visual_material: List[VisualMaterialInst | None] = [None] * len(entities)
        self.is_shared_visual_material = False

        super().__init__(cfg=cfg, entities=entities, device=device, auto_reset=False)

        self._initialize_existing_visual_material()
        self.reset()

        self._set_default_collision_filter()

    @property
    def is_spawn_bound(self) -> bool:
        """Whether this facade is bound to one finalized SpawnResult."""
        return self._spawn_result is not None

    @property
    def is_declared(self) -> bool:
        """Whether this facade is waiting for its SpawnResult binding."""
        return self._spawn_result is None and len(self._entities) == 0

    @property
    def num_instances(self) -> int:
        return len(self._entities) if self._entities else self._declared_num_instances

    def bind_spawn(self, result: SpawnResult, entities: Sequence[Any]) -> None:
        """Bind a declared facade to finalized cloth handles in place."""
        if len(entities) != self._declared_num_instances:
            raise ValueError(
                f"ClothObject {self.uid!r} expected {self._declared_num_instances} "
                f"Spawn handles, got {len(entities)}."
            )
        bound = ClothObject(
            self.cfg,
            entities,
            self.device,
            spawn_result=result,
        )
        self.__dict__.clear()
        self.__dict__.update(bound.__dict__)

    def __str__(self) -> str:
        if self.is_declared:
            return (
                f"{self.__class__}: declared {self.num_instances} Spawn cloth "
                f"objects | uid: {self.uid} | device: {self.device}"
            )
        return super().__str__()

    @staticmethod
    def _build_surface_triangles(
        entity: MeshObject,
        rest_vertices: np.ndarray,
        initial_transform: np.ndarray,
    ) -> np.ndarray:
        """Map render triangles onto DexSim's welded cloth vertex buffer."""
        render_body = entity.get_render_body()
        render_vertices: list[np.ndarray] = []
        render_triangles: list[np.ndarray] = []
        vertex_offset = 0
        for mesh_id in range(render_body.get_mesh_count()):
            vertices = np.asarray(
                render_body.get_vertices(mesh_id),
                dtype=np.float32,
            )
            triangles = np.asarray(
                render_body.get_triangles(mesh_id),
                dtype=np.int64,
            )
            render_vertices.append(vertices)
            render_triangles.append(triangles + vertex_offset)
            vertex_offset += len(vertices)

        vertices = np.concatenate(render_vertices, axis=0)
        triangles = np.concatenate(render_triangles, axis=0)
        initial_transform = np.asarray(initial_transform, dtype=np.float32).reshape(
            4, 4
        )
        vertices = vertices @ initial_transform[:3, :3].T + initial_transform[:3, 3]
        distances, cloth_vertex_ids = cKDTree(rest_vertices).query(vertices)
        scale = max(float(np.ptp(rest_vertices, axis=0).max()), 1.0)
        if float(distances.max(initial=0.0)) > scale * 1.0e-5:
            raise RuntimeError(
                "Could not map cloth render vertices onto the physical vertex buffer."
            )
        return np.asarray(cloth_vertex_ids[triangles], dtype=np.int32)

    def _initialize_existing_visual_material(self) -> None:
        """Wrap asset-parsed materials during cloth-object construction.

        For a multi-segment render body, the first segment with a valid
        material is registered as the environment's representative material.
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

    def set_visual_material(
        self,
        mat: VisualMaterial,
        env_ids: Sequence[int] | None = None,
        shared: bool = False,
    ) -> None:
        """Set visual material for the cloth object.

        Args:
            mat: The material template to assign.
            env_ids: Environment indices. If None, all instances are used.
            shared: Whether selected environments share one material instance.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        if shared:
            if len(local_env_ids) != self.num_instances:
                logger.log_error("Cannot share material instance for partial env_ids.")
            mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}")
            for env_idx in local_env_ids:
                self._entities[env_idx].set_material(mat_inst.mat)
                self._visual_material[env_idx] = mat_inst
            self.is_shared_visual_material = True
        else:
            for env_idx in local_env_ids:
                mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}_{env_idx}")
                self._entities[env_idx].set_material(mat_inst.mat)
                self._visual_material[env_idx] = mat_inst
            self.is_shared_visual_material = False

    def restore_visual_material(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore visual materials captured when the cloth object was created.

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

    def get_visual_material_inst(
        self, env_ids: Sequence[int] | None = None
    ) -> List[VisualMaterialInst | None]:
        """Get the material instance registered for each selected environment.

        Args:
            env_ids: Environment indices. If None, all instances are returned.

        Returns:
            The existing material wrappers, or None where an asset has no material.
        """
        ids = env_ids if env_ids is not None else range(self.num_instances)
        return [self._visual_material[i] for i in ids]

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
        """Set collision filter data for the cloth object.

        Args:
            filter_data (torch.Tensor): [N, 4] of int.
                First element of each object is arena id.
                If 2nd element is 0, the object will collision with all other objects in world.
                3rd and 4th elements are not used currently.

            env_ids (Sequence[int] | None): Environment indices. If None, then all indices are used.
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

    @property
    def body_data(self) -> ClothBodyData | None:
        """Get the cloth body data manager for this cloth object.

        Returns:
            ClothBodyData | None: The cloth body data manager.
        """
        return self._data

    def get_rest_vertex_position(self) -> torch.Tensor:
        """Get the rest vertex position of the cloth bodies.

        Returns:
            torch.Tensor: The rest vertex position of the cloth bodies, shape (num_instances, n_vertices, 3).
        """
        return self._data.rest_vertices

    def get_current_vertex_position(self) -> torch.Tensor:
        """Get the current vertex position of the cloth bodies.

        Returns:
            torch.Tensor: The current vertex position of the cloth bodies, shape (num_instances, n_vertices, 3).
        """
        return self._data.vertex_position

    def get_current_vertex_velocity(self) -> torch.Tensor:
        """Get the current vertex velocity of the cloth bodies.

        Returns:
            torch.Tensor: The current vertex velocity of the cloth bodies, shape (num_instances, n_vertices, 3).
        """
        return self._data.vertex_velocity

    def get_triangles(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get surface triangle indices for selected cloth instances.

        Args:
            env_ids: Environment indices. If ``None``, returns all instances.

        Returns:
            Triangle indices with shape ``(N, num_triangles, 3)``.
        """
        ids = self._all_indices if env_ids is None else env_ids
        triangles = torch.as_tensor(
            self._surface_triangles,
            dtype=torch.int32,
            device=self.device,
        )
        return triangles.unsqueeze(0).expand(len(ids), -1, -1).clone()

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set local pose of the cloth object.

        Args:
            pose (torch.Tensor): The local pose of the cloth object with shape (N, 7) or (N, 4, 4).
            env_ids (Sequence[int] | None): Environment indices. If None, then all indices are used.
        """
        from embodichain.lab.sim import SimulationManager

        sim = SimulationManager.get_instance()

        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(pose):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match pose length {len(pose)}."
            )

        if pose.dim() == 2 and pose.shape[1] == 7:
            pose4x4 = xyz_quat_to_4x4_matrix(pose)
        elif pose.dim() == 3 and pose.shape[1:3] == (4, 4):
            pose4x4 = pose
        else:
            logger.log_error(
                f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
            )

        arena_offsets = sim.arena_offsets
        for i, env_idx in enumerate(local_env_ids):
            # TODO: cloth body cannot directly set by `set_local_pose` currently.
            cloth_body: ClothBody = self._entities[env_idx].get_physical_body()
            rest_vertices = self.body_data.rest_vertices[env_idx]
            initial_transform = torch.as_tensor(
                cloth_body.get_initial_transform(),
                dtype=torch.float32,
                device=self.device,
            )
            rest_vertices_local = (
                rest_vertices - initial_transform[:3, 3]
            ) @ initial_transform[:3, :3]
            rotation = pose4x4[i][:3, :3]
            translation = pose4x4[i][:3, 3]

            transformed_vertices = rest_vertices_local @ rotation.T + translation
            transformed_vertices = transformed_vertices + arena_offsets[env_idx]

            position_buffer = cloth_body.get_position_inv_mass_buffer()
            velocity_buffer = cloth_body.get_velocity_buffer()
            position_buffer[:, :3] = transformed_vertices
            velocity_buffer[:, :3] = 0.0

            cloth_body.mark_dirty(ClothBodyGPUAPIReadWriteType.ALL)
            # TODO: currently cloth body has no wake up interface, use set_wake_counter and pass in a positive value to wake it up
            cloth_body.set_wake_counter(0.4)

    def get_local_pose(self, to_matrix=False):
        """Get local pose of the cloth object.

        Args:
            to_matrix (bool, optional): If True, return the pose as a 4x4 matrix. If False, return as (x, y, z, qw, qx, qy, qz). Defaults to False.

        Returns:
            torch.Tensor: The local pose of the cloth object with shape (N, 7) or (N, 4, 4) depending on `to_matrix`.
        """
        raise NotImplementedError(
            "Getting local pose for ClothObject is not supported."
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        local_env_ids = self._all_indices if env_ids is None else env_ids
        num_instances = len(local_env_ids)

        self.restore_visual_material(env_ids=local_env_ids)

        # TODO: set attr for cloth body after loading in physics scene.

        # rest cloth body to init_pos
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

    def destroy(self) -> None:
        if self.is_spawn_bound:
            return
        # TODO: not tested yet
        env = self._world.get_env()
        arenas = env.get_all_arenas()
        if len(arenas) == 0:
            arenas = [env]
        for i, entity in enumerate(self._entities):
            arenas[i].remove_actor(entity)
