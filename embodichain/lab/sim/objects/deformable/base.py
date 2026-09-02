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
from typing import Any, ClassVar, Literal, Sequence

import torch
from dexsim.scene import Scene

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
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Create an unregistered deformable facade.

        ``SpawnScene`` supplies the replicated instance count when declaring
        this facade and the finalized ``Scene`` when binding it.
        """
        if cfg.deformable_type != self.deformable_type:
            raise ValueError(
                f"{type(self).__name__} requires deformable_type="
                f"{self.deformable_type!r}, got {cfg.deformable_type!r}."
            )

        self._initialize_unregistered(cfg, device)

    def _initialize_unregistered(
        self,
        cfg: DeformableObjectCfg,
        device: torch.device,
    ) -> None:
        """Initialize state that is independent of Spawn replication."""
        self.cfg = deepcopy(cfg)
        self.uid = self.cfg.uid
        self.device = device
        self._entities: list[Any] = []
        self._declared_num_instances: int | None = None
        self._spawn_result: Scene | None = None
        self._world = None
        self._data = None
        self._all_indices: list[int] = []
        self._visual_material: list[VisualMaterialInst | None] = []
        self.is_shared_visual_material = False

    def _initialize_spawn_declaration(self, num_instances: int) -> None:
        """Initialize instance-dependent declaration state from ``SpawnScene``."""
        if num_instances <= 0:
            raise ValueError(
                f"A declared {type(self).__name__} requires num_instances > 0."
            )
        if self._declared_num_instances is not None:
            if self._declared_num_instances != num_instances:
                raise RuntimeError(
                    f"{type(self).__name__} {self.uid!r} is already declared for "
                    f"{self._declared_num_instances} instances."
                )
            return

        self._declared_num_instances = num_instances
        self._all_indices = list(range(num_instances))
        self._visual_material = [None] * num_instances

    def _require_declared_num_instances(self) -> int:
        """Return the Spawn-provided instance count or raise a lifecycle error."""
        if self._declared_num_instances is None:
            raise RuntimeError(
                f"{type(self).__name__} {self.uid!r} must be registered through "
                "SpawnScene before it can be used."
            )
        return self._declared_num_instances

    @abstractmethod
    def _create_data(
        self,
        entities: Sequence[Any],
        scene: Scene,
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
        return (
            len(self._entities)
            if self._entities
            else self._require_declared_num_instances()
        )

    @property
    def data(self) -> DeformableObjectData | None:
        """Return the common deformable data view after Spawn binding."""
        return self._data

    def attach_spawn_handles(self, entities: Sequence[Any]) -> None:
        """Store materialized handles before final Spawn binding."""
        handles = list(entities)
        expected = self._require_declared_num_instances()
        if len(handles) != expected:
            raise ValueError(
                f"{type(self).__name__} {self.uid!r} expected {expected} Spawn "
                f"handles, got {len(handles)}."
            )
        self._entities = handles

    def _initialize_spawn_bound(
        self,
        result: Scene,
        entities: Sequence[Any],
    ) -> None:
        """Create result-dependent runtime state on this declared facade."""
        if not isinstance(result, Scene):
            raise TypeError(
                "DeformableObject binding requires a finalized DexSim Scene; use "
                "SimulationManager.prepare()."
            )

        cfg = deepcopy(self.cfg)
        self._spawn_result = result
        self._world = result.world
        self._all_indices = list(range(len(entities)))
        self._data = self._create_data(entities, result, self.device)
        self._initialize_topology(entities)
        self._visual_material = [None] * len(entities)
        self.is_shared_visual_material = False

        super().__init__(cfg=cfg, entities=list(entities), device=self.device)
        self._initialize_existing_visual_material()
        self.reset()

    def bind_spawn(self, result: Scene) -> None:
        """Bind a declared facade to finalized native handles in place."""
        if self.is_spawn_bound:
            raise RuntimeError(
                f"{type(self).__name__} {self.uid!r} is already Spawn-bound."
            )
        if not self.is_declared:
            raise RuntimeError(
                f"{type(self).__name__} {self.uid!r} was not created as a Spawn declaration."
            )

        entities = list(self._entities)
        expected = self._require_declared_num_instances()
        if len(entities) != expected:
            raise ValueError(
                f"{type(self).__name__} {self.uid!r} expected {expected} Spawn "
                f"handles, got {len(entities)}."
            )

        declared_state = self.__dict__.copy()
        try:
            if self.cfg.shape.compute_uv:
                for entity in entities:
                    entity.compute_uv_mapping()
            self._initialize_spawn_bound(result, entities)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(declared_state)
            raise

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
        """Set simulation-node pose by transforming the default node buffers.

        Volume-deformable collision vertices are synchronized by DexSim on the
        next physics update after this particle-batch write.
        """
        self._require_data()
        local_env_ids = self._resolve_env_ids(env_ids)
        if len(local_env_ids) != len(pose):
            raise ValueError(
                f"Length of env_ids {len(local_env_ids)} does not match pose "
                f"length {len(pose)}."
            )
        if pose.dim() == 2 and pose.shape[1] == 7:
            pose4x4 = xyz_quat_to_4x4_matrix(pose)
        elif pose.dim() == 3 and pose.shape[1:] == (4, 4):
            pose4x4 = pose
        else:
            raise ValueError(
                f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
            )

        self._apply_local_pose(
            pose4x4.to(device=self.device, dtype=torch.float32),
            local_env_ids,
            torch.as_tensor(
                self._spawn_result.arenas.root_offsets,
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _apply_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int],
        arena_offsets: torch.Tensor,
    ) -> None:
        """Apply rest-node transforms through the Spawn particle-set batch."""
        self._require_data()
        if not env_ids:
            return

        env_index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        default_positions = self.data.default_nodal_state_w[..., :3]
        positions = self.data.nodal_pos_w
        velocities = self.data.nodal_vel_w
        initial_pose = self._build_cfg_init_pose(1)[0]
        initial_rotation = initial_pose[:3, :3]
        initial_translation = initial_pose[:3, 3] + arena_offsets[env_index]
        rest_positions_local = (
            default_positions[env_index] - initial_translation.unsqueeze(1)
        ) @ initial_rotation
        target_rotation = pose[:, :3, :3]
        target_translation = pose[:, :3, 3] + arena_offsets[env_index]
        positions[env_index] = rest_positions_local @ target_rotation.transpose(
            -1, -2
        ) + target_translation.unsqueeze(1)
        velocities[env_index] = 0.0
        self.data.apply_nodal_state_w(positions, velocities)

    def _build_cfg_init_pose(self, num_instances: int) -> torch.Tensor:
        """Build configured initial local poses as homogeneous matrices."""
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
        return pose

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
        self.set_local_pose(
            self._build_cfg_init_pose(len(local_env_ids)),
            env_ids=local_env_ids,
        )

    def destroy(self) -> None:
        """Release no native resources; the finalized Spawn scene owns them."""
