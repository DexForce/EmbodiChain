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

"""Batched render-only objects owned by the shared Spawn scene."""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from embodichain.lab.sim.cfg.visual_object import VisualObjectCfg
from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.material import (
    _capture_render_materials,
    _restore_render_materials,
    _wrap_first_render_material,
)
from embodichain.utils.math import matrix_from_quat, quat_from_matrix
from ._mesh_utils import get_combined_triangles, get_combined_vertices
from .backends.scene import _SceneVisualObjectView

if TYPE_CHECKING:
    from dexsim.scene import Scene, SpawnedRigidBody
    from embodichain.lab.sim.material import VisualMaterial, VisualMaterialInst

__all__ = ["VisualObject", "VisualObjectCfg"]


class VisualObject(BatchEntity):
    """A replicated visual asset with no rigid-body state or collision API.

    Declare through ``SimulationManager.add_visual_object()`` or ``add_usd()``
    and call ``prepare()`` before accessing runtime state. Poses are arena-local
    and use EmbodiChain's ``xyz + xyzw`` convention.

    Args:
        cfg: Visual geometry and initial state.
        device: Device for returned tensors.
    """

    def __init__(
        self, cfg: VisualObjectCfg, device: torch.device = torch.device("cpu")
    ) -> None:
        """Create a facade whose runtime handles will be supplied by Spawn.

        Args:
            cfg: Appearance and initial state.
            device: Device for returned tensors.
        """
        self.cfg = deepcopy(cfg)
        self.uid = cfg.uid
        self.device = torch.device(device)
        self._entities: list[SpawnedRigidBody] = []
        self._declared_num_instances: int | None = None
        self._spawn_result: Scene | None = None
        self._pose_view: _SceneVisualObjectView | None = None
        self._default_poses: torch.Tensor | None = None
        self._poses: torch.Tensor | None = None
        self._visibility: list[bool] = []
        self._materials: list[list[object]] = []
        self._spawn_revision = -1
        self._default_materials: list[list[object]] = []
        self._visual_materials: list[VisualMaterialInst | None] = []

    def __str__(self) -> str:
        """Describe the facade without requiring materialized render handles."""
        return f"VisualObject(uid={self.uid!r}, instances={self._declared_num_instances}, bound={self.is_spawn_bound})"

    def _initialize_spawn_declaration(self, num_instances: int) -> None:
        if num_instances <= 0:
            raise ValueError("VisualObject requires a positive instance count.")
        if self._declared_num_instances not in (None, num_instances):
            raise RuntimeError("VisualObject instance count is already declared.")
        self._declared_num_instances = num_instances

    @property
    def num_instances(self) -> int:
        """Number of instances declared by Spawn, including before preparation."""
        if self._declared_num_instances is None:
            raise RuntimeError("VisualObject must be registered through SpawnScene.")
        return self._declared_num_instances

    @property
    def is_spawn_bound(self) -> bool:
        """Whether this facade is bound to a finalized Scene."""
        return self._spawn_result is not None

    @property
    def is_declared(self) -> bool:
        """Whether this facade is waiting for preparation."""
        return not self.is_spawn_bound

    def attach_spawn_handles(self, entities: Sequence[SpawnedRigidBody]) -> None:
        """Attach borrowed render handles supplied by Spawn.

        Args:
            entities: One render-only handle per environment.
        """
        handles = list(entities)
        if len(handles) != self.num_instances:
            raise ValueError(
                "VisualObject handle count does not match its declaration."
            )
        if any(
            e.object_desc.physics is not None or e.object_desc.collisions
            for e in handles
        ):
            raise ValueError("VisualObject cannot bind physical objects or colliders.")
        self._entities = handles

    def bind_spawn(self, result: Scene) -> None:
        """Capture initial visual state after Spawn materializes the object.

        Args:
            result: Finalized Scene owning the attached handles.
        """
        if self.is_spawn_bound:
            raise RuntimeError("VisualObject is already Spawn-bound.")
        if len(self._entities) != self.num_instances:
            raise RuntimeError("VisualObject handles are unavailable; call prepare().")
        self._pose_view = _SceneVisualObjectView(result, self._entities, self.device)
        poses = self._pose_view.fetch_pose(self._pose_view.rows)
        self._apply_visual_uv()
        material_cfg = getattr(self.cfg.shape, "visual_material", None)
        materials = [
            (
                []
                if e.get_render_body() is None
                else _capture_render_materials(e.get_render_body(), cfg=material_cfg)
            )
            for e in self._entities
        ]
        wrappers = [_wrap_first_render_material(m) for m in materials]
        for e in self._entities:
            e.set_visible(self.cfg.visible)
        self._default_poses = poses
        self._poses = poses.clone()
        self._visibility = [self.cfg.visible] * self.num_instances
        self._materials = [list(items) for items in materials]
        self._spawn_revision = int(result.topology_revision)
        self._default_materials = materials
        self._visual_materials = wrappers
        self._spawn_result = result

    def _apply_visual_uv(self) -> None:
        shape = self.cfg.shape
        if isinstance(shape, MeshCfg) and shape.compute_uv:
            for entity in self._entities:
                render_body = entity.get_render_body()
                if render_body is not None:
                    render_body.set_projective_uv(
                        np.asarray(shape.project_direction, dtype=np.float32)
                    )

    def _sync_spawn_state(self) -> None:
        """Restore facade-owned visual state after a native topology rebuild."""
        self._require_bound()
        revision = int(self._spawn_result.topology_revision)
        if revision == self._spawn_revision:
            return
        self._pose_view.refresh(self._entities)
        self._pose_view.apply_pose(self._poses, self._pose_view.rows)
        for i in range(self.num_instances):
            entity = self._entities[i]
            entity.set_visible(self._visibility[i])
            render_body = entity.get_render_body()
            if render_body is not None:
                _restore_render_materials(render_body, self._materials[i])
        self._apply_visual_uv()
        self._spawn_revision = revision

    def _require_bound(self) -> None:
        if not self.is_spawn_bound:
            raise RuntimeError(
                f"VisualObject {self.uid!r} requires SimulationManager.prepare()."
            )

    def _indices(self, env_ids: Sequence[int] | torch.Tensor | None) -> list[int]:
        self._require_bound()
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1 or env_ids.dtype not in (torch.int32, torch.int64):
                raise IndexError(
                    "VisualObject environment indices must be a 1D integer tensor."
                )
            env_ids = env_ids.detach().cpu().tolist()
        ids = list(range(self.num_instances)) if env_ids is None else list(env_ids)
        if any(
            isinstance(i, bool)
            or not isinstance(i, Integral)
            or i < 0
            or i >= self.num_instances
            for i in ids
        ):
            raise IndexError("VisualObject environment indices are out of range.")
        if len(set(ids)) != len(ids):
            raise ValueError("VisualObject environment indices must be unique.")
        return ids

    @property
    def user_ids(self) -> torch.Tensor:
        """Return native render-instance IDs as an ``(N,)`` integer tensor."""
        self._require_bound()
        return self._pose_view.user_ids.clone()

    def _pose_rows(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        self._require_bound()
        if env_ids is None:
            return self._pose_view.rows
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1 or env_ids.dtype not in (torch.int32, torch.int64):
                raise IndexError(
                    "VisualObject environment indices must be a 1D integer tensor."
                )
            rows = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = self._indices(env_ids)
            rows = torch.tensor(ids, dtype=torch.long, device=self.device)
        if torch.any(rows < 0) or torch.any(rows >= self.num_instances):
            raise IndexError("VisualObject environment indices are out of range.")
        if len(torch.unique(rows)) != len(rows):
            raise ValueError("VisualObject environment indices must be unique.")
        return rows

    def get_local_pose(
        self,
        to_matrix: bool = False,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read independent arena-local poses.

        Args:
            to_matrix: Return matrices instead of ``xyz + xyzw`` poses.
            env_ids: Selected environments in output order, or all environments.

        Returns:
            Tensor shaped ``(M, 4, 4)`` or ``(M, 7)`` for the selected rows.
        """
        rows = self._pose_rows(env_ids)
        poses = self._pose_view.fetch_pose(rows)
        return (
            poses
            if to_matrix
            else torch.cat(
                (poses[:, :3, 3], quat_from_matrix(poses[:, :3, :3])), dim=-1
            )
        )

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Write selected root poses without advancing simulation.

        Args:
            pose: ``(M, 7)`` xyz/xyzw poses or ``(M, 4, 4)`` rigid transforms.
            env_ids: Selected environments, or all environments.
        """
        rows = self._pose_rows(env_ids)
        values = torch.as_tensor(pose, dtype=torch.float32, device=self.device)
        if values.shape == (len(rows), 7):
            if not torch.isfinite(values).all() or torch.any(
                torch.linalg.vector_norm(values[:, 3:], dim=-1) <= 0
            ):
                raise ValueError(
                    "VisualObject poses require finite values and nonzero quaternions."
                )
            matrices = torch.eye(4, device=self.device).repeat(len(rows), 1, 1)
            matrices[:, :3, :3] = matrix_from_quat(
                torch.nn.functional.normalize(values[:, 3:], dim=-1)
            )
            matrices[:, :3, 3] = values[:, :3]
        elif values.shape == (len(rows), 4, 4):
            matrices = values
        else:
            raise ValueError("VisualObject poses must have shape (M, 7) or (M, 4, 4).")
        rotation = matrices[:, :3, :3]
        if (
            not torch.isfinite(matrices).all()
            or not torch.allclose(
                matrices[:, 3],
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(
                    len(rows), -1
                ),
                atol=1e-5,
            )
            or not torch.allclose(
                rotation.transpose(1, 2) @ rotation,
                torch.eye(3, device=self.device).expand(len(rows), -1, -1),
                atol=1e-5,
            )
            or not torch.allclose(
                torch.linalg.det(rotation),
                torch.ones(len(rows), device=self.device),
                atol=1e-5,
            )
        ):
            raise ValueError("VisualObject matrices must be finite rigid transforms.")
        self._pose_view.apply_pose(matrices, rows)
        self._poses.index_copy_(0, rows, matrices.detach())

    def set_visible(
        self, visible: bool = True, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Set visibility for selected environments.

        Args:
            visible: Whether selected instances are visible.
            env_ids: Selected environments, or all environments.
        """
        if not isinstance(visible, bool):
            raise TypeError("visible must be a bool.")
        for i in self._indices(env_ids):
            self._entities[i].set_visible(visible)
            self._visibility[i] = visible

    def get_visibility(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Read selected visibility flags.

        Args:
            env_ids: Selected environments, or all environments.

        Returns:
            Boolean tensor shaped ``(M,)``.
        """
        return torch.tensor(
            [self._entities[i].is_visible() for i in self._indices(env_ids)],
            dtype=torch.bool,
            device=self.device,
        )

    def get_body_scale(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Read visual geometry scales, independent of root transforms.

        Args:
            env_ids: Selected environments, or all environments.

        Returns:
            Scale tensor shaped ``(M, 3)``.
        """
        return torch.as_tensor(
            np.array(
                [self._entities[i].get_body_scale() for i in self._indices(env_ids)]
            ),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 3)

    def get_vertices(
        self, env_ids: Sequence[int] | torch.Tensor | None = None, scale: bool = False
    ) -> torch.Tensor:
        """Read combined local mesh vertices for selected instances.

        Args:
            env_ids: Selected environments, or all environments.
            scale: Include the visual geometry scale.

        Returns:
            Vertices shaped ``(M, V, 3)``.
        """
        ids = self._indices(env_ids)
        if not ids:
            return torch.empty(
                (0, len(get_combined_vertices(self._entities[0])), 3),
                device=self.device,
            )
        vertices = torch.as_tensor(
            np.array([get_combined_vertices(self._entities[i]) for i in ids]),
            dtype=torch.float32,
            device=self.device,
        )
        return vertices * self.get_body_scale(ids).unsqueeze(1) if scale else vertices

    def get_triangles(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Read triangle indices addressing the combined visual mesh.

        Args:
            env_ids: Selected environments, or all environments.

        Returns:
            Indices shaped ``(M, F, 3)``.
        """
        ids = self._indices(env_ids)
        if not ids:
            return torch.empty(
                (0, len(get_combined_triangles(self._entities[0])), 3),
                dtype=torch.int32,
                device=self.device,
            )
        return torch.as_tensor(
            np.array(
                [
                    get_combined_triangles(self._entities[i])
                    for i in self._indices(env_ids)
                ]
            ),
            dtype=torch.int32,
            device=self.device,
        )

    def set_visual_material(
        self, mat: VisualMaterial, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Assign separate material instances to selected environments.

        Args:
            mat: Material template from the simulation manager.
            env_ids: Selected environments, or all environments.
        """
        for i in self._indices(env_ids):
            instance = mat.create_instance(f"{mat.uid}_{self.uid}_{i}")
            self._entities[i].set_material(instance.mat)
            self._visual_materials[i] = instance
            self._materials[i] = _capture_render_materials(
                self._entities[i].get_render_body()
            )

    def get_visual_material_inst(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> list[VisualMaterialInst | None]:
        """Read current material wrappers for selected environments.

        Args:
            env_ids: Selected environments, or all environments.

        Returns:
            One wrapper per selected instance, possibly ``None`` without a material.
        """
        return [self._visual_materials[i] for i in self._indices(env_ids)]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Restore selected initial poses, visibility and material assignments.

        Args:
            env_ids: Selected environments, or all environments.
        """
        ids = self._indices(env_ids)
        self.set_local_pose(self._default_poses[ids], ids)
        for i in ids:
            self._entities[i].set_visible(self.cfg.visible)
            self._visibility[i] = self.cfg.visible
            self._materials[i] = list(self._default_materials[i])
            render_body = self._entities[i].get_render_body()
            if render_body is not None:
                _restore_render_materials(render_body, self._default_materials[i])
            self._visual_materials[i] = _wrap_first_render_material(
                self._default_materials[i]
            )

    def _release_spawn_references(self) -> None:
        self._spawn_result = None
        self._pose_view = None
        self._entities.clear()
        self._default_poses = None
        self._poses = None
        self._visibility.clear()
        self._materials.clear()
        self._default_materials.clear()
        self._visual_materials.clear()

    def destroy(self) -> None:
        """Leave resource removal to SimulationManager and the owning Spawn scene."""
