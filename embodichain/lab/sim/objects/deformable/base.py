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

"""Common facade for volume and surface deformable objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Sequence

import dexsim
import numpy as np
import torch

from embodichain.lab.sim.cfg import DeformableObjectCfg
from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.material import (
    VisualMaterial,
    VisualMaterialInst,
    _capture_render_materials,
    _restore_render_materials,
    _wrap_first_render_material,
)
from embodichain.utils import logger
from embodichain.utils.math import matrix_from_euler, xyz_quat_to_4x4_matrix

from .data import DeformableObjectData

if TYPE_CHECKING:
    from dexsim.engine import PhysicsScene
    from dexsim.spawn import SpawnResult

__all__ = ["DeformableObject"]


class DeformableObject(BatchEntity, ABC):
    """Common facade for a batch of deformable assets.

    The public nodal and surface contracts are backend-neutral. The concrete
    implementations in this package currently bind them to DexSim soft-body
    and cloth buffers. Newton support can be added as a separate implementation
    without changing manager or visualization consumers.
    """

    deformable_type: ClassVar[Literal["volume", "surface"]]
    spawn_kind: ClassVar[str]
    display_name: ClassVar[str]

    def __init__(
        self,
        cfg: DeformableObjectCfg,
        entities: Sequence[Any] | None = None,
        device: torch.device = torch.device("cpu"),
        *,
        spawn_result: SpawnResult | None = None,
        declared_num_instances: int | None = None,
    ) -> None:
        if cfg.deformable_type != self.deformable_type:
            raise ValueError(
                f"{type(self).__name__} requires deformable_type="
                f"{self.deformable_type!r}, got {cfg.deformable_type!r}."
            )

        if entities is None:
            self._initialize_declared(cfg, device, declared_num_instances)
            return

        entities = list(entities)
        self._declared_num_instances = len(entities)
        self._spawn_result = spawn_result
        if spawn_result is None:
            self._world = dexsim.default_world()
            from embodichain.lab.sim.sim_manager import get_physics_scene

            self._ps: PhysicsScene | None = get_physics_scene()
        else:
            self._world = spawn_result.world
            self._ps = self._world.get_physics_scene()
        self._all_indices = list(range(len(entities)))

        self._data = self._create_data(entities, self._ps, device)
        if spawn_result is None:
            self._world.update(0.001)
        self._initialize_topology(entities)

        self._visual_material: list[VisualMaterialInst | None] = [None] * len(entities)
        self.is_shared_visual_material = False

        super().__init__(cfg=cfg, entities=entities, device=device, auto_reset=False)
        self._initialize_existing_visual_material()
        self.reset()
        self._set_default_collision_filter()

    def _initialize_declared(
        self,
        cfg: DeformableObjectCfg,
        device: torch.device,
        declared_num_instances: int | None,
    ) -> None:
        """Initialize a facade before Spawn materializes native handles."""
        if declared_num_instances is None or declared_num_instances <= 0:
            raise ValueError(
                f"A declared {type(self).__name__} requires "
                "declared_num_instances > 0."
            )
        self.cfg = deepcopy(cfg)
        self.uid = self.cfg.uid
        self.device = device
        self._entities: list[Any] = []
        self._declared_num_instances = declared_num_instances
        self._spawn_result = None
        self._world = None
        self._ps = None
        self._data = None
        self._all_indices = list(range(declared_num_instances))
        self._visual_material = [None] * declared_num_instances
        self.is_shared_visual_material = False

    @abstractmethod
    def _create_data(
        self,
        entities: Sequence[Any],
        physics_scene: PhysicsScene,
        device: torch.device,
    ) -> DeformableObjectData:
        """Create the concrete backend data view."""

    def _initialize_topology(self, entities: Sequence[Any]) -> None:
        """Initialize implementation-specific surface topology."""
        del entities

    @property
    def is_spawn_bound(self) -> bool:
        """Whether this facade is bound to one finalized Spawn result."""
        return self._spawn_result is not None

    @property
    def is_declared(self) -> bool:
        """Whether this facade is waiting for its Spawn result binding."""
        return self._world is None

    @property
    def num_instances(self) -> int:
        """Return the materialized or declared instance count."""
        return len(self._entities) if self._entities else self._declared_num_instances

    @property
    def data(self) -> DeformableObjectData | None:
        """Return the common deformable data view after Spawn binding."""
        return self._data

    def attach_spawn_handles(self, entities: Sequence[Any]) -> None:
        """Store materialized handles before final Spawn binding."""
        self._entities = list(entities)

    def bind_spawn(self, result: SpawnResult) -> None:
        """Bind a declared facade to finalized native handles in place."""
        entities = list(self._entities)
        if self.cfg.shape.compute_uv:
            for entity in entities:
                entity.compute_uv_mapping()
        type(self).__init__(
            self,
            self.cfg,
            entities,
            self.device,
            spawn_result=result,
        )

    def __str__(self) -> str:
        if self.is_declared:
            return (
                f"{self.__class__}: declared {self.num_instances} Spawn "
                f"{self.display_name} objects | uid: {self.uid} | "
                f"device: {self.device}"
            )
        return super().__str__()

    def _initialize_existing_visual_material(self) -> None:
        """Capture and wrap materials parsed from the source asset."""
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
        """Assign visual material instances to selected environments."""
        local_env_ids = self._resolve_env_ids(env_ids)
        if shared:
            if len(local_env_ids) != self.num_instances:
                logger.log_error("Cannot share material instance for partial env_ids.")
            mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}")
            for env_idx in local_env_ids:
                self._entities[env_idx].set_material(mat_inst.mat)
                self._visual_material[env_idx] = mat_inst
            self.is_shared_visual_material = True
            return

        for env_idx in local_env_ids:
            mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}_{env_idx}")
            self._entities[env_idx].set_material(mat_inst.mat)
            self._visual_material[env_idx] = mat_inst
        self.is_shared_visual_material = False

    def restore_visual_material(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore materials captured when the deformable was created."""
        if not hasattr(self, "_original_visual_material"):
            return
        for env_idx in self._resolve_env_ids(env_ids):
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
    ) -> list[VisualMaterialInst | None]:
        """Return registered material wrappers for selected environments."""
        return [self._visual_material[i] for i in self._resolve_env_ids(env_ids)]

    def _set_default_collision_filter(self) -> None:
        collision_filter_data = torch.zeros(
            size=(self.num_instances, 4), dtype=torch.int32
        )
        collision_filter_data[:, 0] = torch.arange(
            self.num_instances, dtype=torch.int32
        )
        collision_filter_data[:, 1] = 1
        self.set_collision_filter(collision_filter_data)

    def set_collision_filter(
        self, filter_data: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set native collision-filter data for selected environments."""
        local_env_ids = self._resolve_env_ids(env_ids)
        if len(local_env_ids) != len(filter_data):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match filter "
                f"data length {len(filter_data)}."
            )
        filter_data_np = filter_data.detach().cpu().numpy().astype(np.uint32)
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].get_physical_body().set_collision_filter_data(
                filter_data_np[i]
            )

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> list[int]:
        if env_ids is None:
            return list(self._all_indices)
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.detach().cpu().reshape(-1).tolist()
        else:
            ids = list(env_ids)
        resolved = [int(env_id) for env_id in ids]
        if any(env_id < 0 or env_id >= self.num_instances for env_id in resolved):
            raise IndexError(
                f"Environment IDs {resolved!r} are outside [0, {self.num_instances})."
            )
        return resolved

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set deformable pose by transforming its rest-node buffers."""
        from embodichain.lab.sim import SimulationManager

        local_env_ids = self._resolve_env_ids(env_ids)
        if len(local_env_ids) != len(pose):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match pose "
                f"length {len(pose)}."
            )
        if pose.dim() == 2 and pose.shape[1] == 7:
            pose4x4 = xyz_quat_to_4x4_matrix(pose)
        elif pose.dim() == 3 and pose.shape[1:] == (4, 4):
            pose4x4 = pose
        else:
            logger.log_error(
                f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
            )

        sim = SimulationManager.get_instance()
        self._apply_local_pose(
            pose4x4.to(device=self.device, dtype=torch.float32),
            local_env_ids,
            sim.arena_offsets,
        )

    @abstractmethod
    def _apply_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int],
        arena_offsets: torch.Tensor,
    ) -> None:
        """Apply rest-node transforms to native backend buffers."""

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Reject root-pose reads because deformables have no rigid root pose."""
        del to_matrix
        raise NotImplementedError(
            f"Getting local pose for {type(self).__name__} is not supported."
        )

    def get_current_nodal_position(self) -> torch.Tensor:
        """Return current simulation-node positions in world frame."""
        self._require_data()
        return self.data.nodal_pos_w

    def get_current_nodal_velocity(self) -> torch.Tensor:
        """Return current simulation-node velocities in world frame."""
        self._require_data()
        return self.data.nodal_vel_w

    def get_current_nodal_state(self) -> torch.Tensor:
        """Return current simulation-node state ``[position, velocity]``."""
        self._require_data()
        return self.data.nodal_state_w

    def get_default_nodal_state(self) -> torch.Tensor:
        """Return default simulation-node state ``[position, velocity]``."""
        self._require_data()
        return self.data.default_nodal_state_w

    def _require_data(self) -> None:
        if self.data is None:
            raise RuntimeError(
                f"{type(self).__name__} data is unavailable before Spawn finalization."
            )

    @abstractmethod
    def get_surface_vertices(self) -> torch.Tensor:
        """Return visualization/collision surface vertices in world frame."""

    @abstractmethod
    def get_surface_triangles(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return surface triangle indices for selected environments."""

    def get_triangles(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Compatibility alias for :meth:`get_surface_triangles`."""
        return self.get_surface_triangles(env_ids=env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore initial pose, zero nodal velocity, and source materials."""
        local_env_ids = self._resolve_env_ids(env_ids)
        self.restore_visual_material(env_ids=local_env_ids)
        num_instances = len(local_env_ids)

        pos = torch.as_tensor(
            self.cfg.init_pos, dtype=torch.float32, device=self.device
        ).repeat(num_instances, 1)
        rot = (
            torch.as_tensor(self.cfg.init_rot, dtype=torch.float32, device=self.device)
            * torch.pi
            / 180.0
        ).repeat(num_instances, 1)
        pose = (
            torch.eye(4, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .repeat(num_instances, 1, 1)
        )
        pose[:, :3, 3] = pos
        pose[:, :3, :3] = matrix_from_euler(rot, "XYZ")
        self.set_local_pose(pose, env_ids=local_env_ids)

    def destroy(self) -> None:
        """Destroy legacy directly-created native entities.

        Spawn-bound entities are owned and released by ``SpawnResult``.
        """
        if self.is_spawn_bound or self.is_declared:
            return
        env = self._world.get_env()
        arenas = env.get_all_arenas()
        if len(arenas) == 0:
            arenas = [env]
        for i, entity in enumerate(self._entities):
            arenas[i].remove_actor(entity)
