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

import torch
import dexsim
import numpy as np
from functools import cached_property

from dataclasses import dataclass
from typing import List, Sequence, Union

from dexsim.models import MeshObject
from dexsim.engine import PhysicsScene, ClothBody
from dexsim.types import ClothBodyGPUAPIReadWriteType
from embodichain.lab.sim.common import (
    BatchEntity,
)
from embodichain.utils.math import (
    matrix_from_euler,
)
from embodichain.utils import logger
from embodichain.lab.sim.cfg import (
    ClothObjectCfg,
)
from embodichain.utils.math import xyz_quat_to_4x4_matrix


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
        self._arena_positions = self._get_arena_position()
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
            self._rest_position_buffer[i] = cloth_body.get_position_inv_mass_buffer()

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

    def _get_arena_position(self):
        n_env = len(self.entities)
        arena_positions = torch.empty(
            (n_env, 3), device=self.device, dtype=torch.float32
        )
        for i, entity in enumerate(self.entities):
            arena = entity.node.get_parent()
            arena_position = arena.get_world_pose()[:3, 3]
            arena_positions[i] = torch.as_tensor(
                arena_position, device=self.device, dtype=torch.float32
            )
        return arena_positions

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
        entities: List[MeshObject] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._world: dexsim.World = dexsim.default_world()
        self._ps = self._world.get_physics_scene()
        self._all_indices = torch.arange(len(entities), dtype=torch.int32).tolist()

        self._data = ClothBodyData(entities=entities, ps=self._ps, device=device)

        self._world.update(0.001)

        super().__init__(cfg=cfg, entities=entities, device=device)
        self._set_default_collision_filter()

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

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set local pose of the cloth object.

        Args:
            pose (torch.Tensor): The local pose of the cloth object with shape (N, 7) or (N, 4, 4).
            env_ids (Sequence[int] | None): Environment indices. If None, then all indices are used.
        """
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

        for i, env_idx in enumerate(local_env_ids):
            # TODO: cloth body cannot directly set by `set_local_pose` currently.
            rest_vertices = self.body_data.rest_vertices[i]
            rotation = pose4x4[i][:3, :3]
            translation = pose4x4[i][:3, 3]

            # apply transformation to local rest vertices and back
            rest_vertices_local = rest_vertices - self._data._arena_positions[i]
            transformed_vertices = rest_vertices_local @ rotation.T + translation
            transformed_vertices = transformed_vertices + self._data._arena_positions[i]

            cloth_body: ClothBody = self._entities[env_idx].get_physical_body()
            position_buffer = cloth_body.get_position_inv_mass_buffer()
            velocity_buffer = cloth_body.get_velocity_buffer()
            position_buffer[:, :3] = transformed_vertices
            velocity_buffer[:, 3:] = 0.0

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
        # TODO: not tested yet
        env = self._world.get_env()
        arenas = env.get_all_arenas()
        if len(arenas) == 0:
            arenas = [env]
        for i, entity in enumerate(self._entities):
            arenas[i].remove_actor(entity)
