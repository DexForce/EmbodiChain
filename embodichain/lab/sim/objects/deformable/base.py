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

"""Common Newton facade for volume and surface deformable objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, ClassVar, Literal, Sequence

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

from .data import DeformableObjectData, _ParticleSetData

if TYPE_CHECKING:
    from dexsim.scene import Scene, SpawnedParticleSet

__all__ = ["DeformableObject"]


class DeformableObject(BatchEntity, ABC):
    """Common facade over a batch of Newton particle-set deformables.

    Volume and surface objects retain EmbodiChain's public nodal contract, but
    their runtime ownership is exclusively DexSim Spawn's Newton scene. The
    Default backend and direct native soft/cloth body buffers are unsupported.
    """

    deformable_type: ClassVar[Literal["volume", "surface"]]
    spawn_kind: ClassVar[str]
    display_name: ClassVar[str]

    def __init__(
        self,
        cfg: DeformableObjectCfg,
        entities: Sequence[SpawnedParticleSet] | None = None,
        device: torch.device = torch.device("cpu"),
        *,
        spawn_result: Scene | None = None,
        declared_num_instances: int | None = None,
    ) -> None:
        if cfg.deformable_type != self.deformable_type:
            raise ValueError(
                f"{type(self).__name__} requires deformable_type="
                f"{self.deformable_type!r}, got {cfg.deformable_type!r}."
            )

        device = torch.device(device)
        if entities is None:
            self._initialize_declared(cfg, device, declared_num_instances)
            return

        entities = list(entities)
        if not entities:
            raise ValueError(f"A bound {type(self).__name__} requires handles.")
        if spawn_result is None:
            raise RuntimeError(
                "Deformable objects must bind to a finalized DexSim Spawn scene."
            )
        if getattr(spawn_result, "backend", None) != "newton":
            raise NotImplementedError(
                "EmbodiChain deformable objects require the Newton backend; "
                "the Default backend is no longer supported."
            )

        self._declared_num_instances = len(entities)
        self._spawn_result = spawn_result
        self._scene = spawn_result
        self._world = spawn_result.world
        self._all_indices = list(range(len(entities)))
        super().__init__(cfg=cfg, entities=entities, device=device)

        self._arena_offsets = self._resolve_arena_offsets(spawn_result, entities)
        self._data = self._create_data(entities, spawn_result, device)
        self._local_rest_positions = self._capture_local_rest_positions()
        self._initialize_topology(entities)

        self._visual_material: list[VisualMaterialInst | None] = [None] * len(entities)
        self.is_shared_visual_material = False
        self._initialize_existing_visual_material()
        self.reset()

    def _initialize_declared(
        self,
        cfg: DeformableObjectCfg,
        device: torch.device,
        declared_num_instances: int | None,
    ) -> None:
        """Initialize a facade before Spawn materializes particle handles."""
        if declared_num_instances is None or declared_num_instances <= 0:
            raise ValueError(
                f"A declared {type(self).__name__} requires "
                "declared_num_instances > 0."
            )
        self.cfg = deepcopy(cfg)
        self.uid = self.cfg.uid
        self.device = device
        self._entities: list[SpawnedParticleSet] = []
        self._declared_num_instances = declared_num_instances
        self._spawn_result = None
        self._scene = None
        self._world = None
        self._data = None
        self._all_indices = list(range(declared_num_instances))
        self._visual_material = [None] * declared_num_instances
        self.is_shared_visual_material = False

    @abstractmethod
    def _create_data(
        self,
        entities: Sequence[SpawnedParticleSet],
        scene: Scene,
        device: torch.device,
    ) -> _ParticleSetData:
        """Create the topology-specific particle data view."""

    def _initialize_topology(self, entities: Sequence[SpawnedParticleSet]) -> None:
        """Capture per-instance render topology with a stable batch shape."""
        vertex_counts = tuple(
            np.asarray(entity.get_render_vertices(), dtype=np.float32)
            .reshape(-1, 3)
            .shape[0]
            for entity in entities
        )
        if len(set(vertex_counts)) != 1:
            raise RuntimeError(
                "Replicated Newton deformable render meshes must share one "
                "vertex count, but DexSim materialized counts "
                f"{vertex_counts}. This indicates a render-clone topology "
                "mismatch; use a compatible source mesh or one environment "
                "until the DexSim clone path is corrected."
            )
        triangles = [
            np.asarray(entity.get_render_triangles(), dtype=np.int32).reshape(-1, 3)
            for entity in entities
        ]
        triangle_counts = {len(item) for item in triangles}
        if len(triangle_counts) != 1:
            raise ValueError(
                "All instances of one deformable asset must share render "
                f"triangle count, got {sorted(triangle_counts)}."
            )
        for instance, (instance_triangles, vertex_count) in enumerate(
            zip(triangles, vertex_counts, strict=True)
        ):
            if instance_triangles.size and (
                int(instance_triangles.min()) < 0
                or int(instance_triangles.max()) >= vertex_count
            ):
                raise ValueError(
                    "Deformable render topology contains an out-of-range "
                    f"vertex index for instance {instance}."
                )
        self._surface_triangles = torch.as_tensor(
            np.stack(triangles),
            dtype=torch.int32,
            device=self.device,
        ).clone()

    @staticmethod
    def _resolve_arena_offsets(
        scene: Scene,
        entities: Sequence[SpawnedParticleSet],
    ) -> torch.Tensor:
        if not scene.arenas:
            offsets = np.zeros((len(entities), 3), dtype=np.float32)
        else:
            arena_indices = [
                scene.arenas.index(entity.arena_name) for entity in entities
            ]
            offsets = scene.arenas.root_offsets[arena_indices]
        return torch.as_tensor(offsets, dtype=torch.float32)

    def _configured_initial_pose(self) -> torch.Tensor:
        if self.cfg.init_local_pose is not None:
            pose = torch.as_tensor(
                self.cfg.init_local_pose,
                dtype=torch.float32,
                device=self.device,
            ).reshape(4, 4)
            return pose.clone()

        pose = torch.eye(4, dtype=torch.float32, device=self.device)
        pose[:3, 3] = torch.as_tensor(
            self.cfg.init_pos,
            dtype=torch.float32,
            device=self.device,
        )
        rotation = (
            torch.as_tensor(
                self.cfg.init_rot,
                dtype=torch.float32,
                device=self.device,
            )
            * torch.pi
            / 180.0
        )
        pose[:3, :3] = matrix_from_euler(rotation.unsqueeze(0), "XYZ")[0]
        return pose

    def _capture_local_rest_positions(self) -> torch.Tensor:
        self._require_data()
        initial_pose = self._configured_initial_pose()
        initial_positions = self.data.default_nodal_state_w[..., :3]
        arena_offsets = self._arena_offsets.to(self.device).unsqueeze(1)
        translated = initial_positions - initial_pose[:3, 3] - arena_offsets
        return translated @ initial_pose[:3, :3]

    @property
    def is_spawn_bound(self) -> bool:
        """Whether this facade is bound to one finalized Spawn scene."""
        return self._spawn_result is not None

    @property
    def is_declared(self) -> bool:
        """Whether this facade is waiting for its Spawn scene binding."""
        return self._scene is None

    @property
    def num_instances(self) -> int:
        """Return the materialized or declared instance count."""
        return len(self._entities) if self._entities else self._declared_num_instances

    @property
    def data(self) -> DeformableObjectData | None:
        """Return the common deformable data view after Spawn binding."""
        return self._data

    def attach_spawn_handles(self, entities: Sequence[SpawnedParticleSet]) -> None:
        """Store materialized particle handles before final Spawn binding."""
        self._entities = list(entities)

    def bind_spawn(self, result: Scene) -> None:
        """Bind a declared facade to finalized Newton particle handles in place."""
        entities = list(self._entities)
        if self.cfg.shape.compute_uv:
            for entity in entities:
                render_body = entity.get_render_body()
                project_uv = getattr(render_body, "set_projective_uv", None)
                if project_uv is None:
                    raise NotImplementedError(
                        "compute_uv requires a deformable render body with "
                        "set_projective_uv()."
                    )
                project_uv(np.asarray(self.cfg.shape.project_direction))
        bound = type(self)(
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

    def set_collision_filter(
        self, filter_data: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Reject legacy per-body filtering absent from Newton particle sets."""
        del filter_data, env_ids
        raise NotImplementedError(
            "Newton deformable collision filtering is scene/solver-owned; "
            "per-object Default collision-filter data is unsupported."
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
        """Set a deformable pose by transforming its captured rest particles."""
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
                f"Invalid pose shape {tuple(pose.shape)}. Expected (N, 7) or "
                "(N, 4, 4)."
            )
        self._apply_local_pose(
            pose4x4.to(device=self.device, dtype=torch.float32),
            local_env_ids,
        )

    def _apply_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int],
    ) -> None:
        """Apply rest-particle transforms through the Spawn particle batch."""
        self._require_data()
        if not env_ids:
            return
        index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        local_positions = self._local_rest_positions.index_select(0, index)
        rotations = pose[:, :3, :3]
        translations = pose[:, :3, 3].unsqueeze(1)
        arena_offsets = self._arena_offsets.to(self.device).index_select(0, index)
        positions = (
            torch.bmm(local_positions, rotations.transpose(1, 2))
            + translations
            + arena_offsets.unsqueeze(1)
        )
        self._data._apply_nodal_state(
            positions,
            torch.zeros_like(positions),
            env_ids,
        )

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Reject root-pose reads because deformables have no rigid root pose."""
        del to_matrix
        raise NotImplementedError(
            f"Getting local pose for {type(self).__name__} is not supported."
        )

    def get_current_nodal_position(self) -> torch.Tensor:
        """Return current simulation-particle positions in world frame."""
        self._require_data()
        return self.data.nodal_pos_w

    def get_current_nodal_velocity(self) -> torch.Tensor:
        """Return current simulation-particle velocities in world frame."""
        self._require_data()
        return self.data.nodal_vel_w

    def get_current_nodal_state(self) -> torch.Tensor:
        """Return current nodal state ``[position, velocity]``."""
        self._require_data()
        return self.data.nodal_state_w

    def get_default_nodal_state(self) -> torch.Tensor:
        """Return the nodal state captured when Spawn was bound."""
        self._require_data()
        return self.data.default_nodal_state_w

    def _require_data(self) -> None:
        if self.data is None:
            raise RuntimeError(
                f"{type(self).__name__} data is unavailable before Spawn finalization."
            )

    def get_surface_vertices(self) -> torch.Tensor:
        """Return live render-surface vertices in world frame."""
        vertices_per_instance: list[torch.Tensor] = []
        render_pose = self._configured_initial_pose()
        render_rotation = render_pose[:3, :3]
        render_translation = render_pose[:3, 3]
        arena_offsets = self._arena_offsets.to(self.device)
        for env_idx, entity in enumerate(self._entities):
            vertices_warp = entity.get_render_vertices_warp()
            if vertices_warp is None:
                vertices = torch.as_tensor(
                    entity.get_render_vertices(),
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(-1, 3)
            else:
                import warp as wp

                vertices = wp.to_torch(vertices_warp).reshape(-1, 3).to(self.device)
            vertices_per_instance.append(
                vertices @ render_rotation.T
                + render_translation
                + arena_offsets[env_idx]
            )

        vertex_counts = {len(vertices) for vertices in vertices_per_instance}
        if len(vertex_counts) != 1:
            raise ValueError(
                "All instances of one deformable asset must share render vertex count."
            )
        return torch.stack(vertices_per_instance).clone()

    def get_surface_triangles(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return render-surface triangle indices for selected environments."""
        ids = self._resolve_env_ids(env_ids)
        index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        return self._surface_triangles.index_select(0, index).clone()

    def get_triangles(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Compatibility alias for :meth:`get_surface_triangles`."""
        return self.get_surface_triangles(env_ids=env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore the configured pose, zero velocity, and source materials."""
        local_env_ids = self._resolve_env_ids(env_ids)
        self.restore_visual_material(env_ids=local_env_ids)
        initial_pose = self._configured_initial_pose()
        pose = initial_pose.unsqueeze(0).repeat(len(local_env_ids), 1, 1)
        self.set_local_pose(pose, env_ids=local_env_ids)

    def destroy(self) -> None:
        """Leave particle lifetime ownership with the finalized Spawn scene."""
