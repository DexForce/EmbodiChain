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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from dexsim.scene import SpawnedRigidBody

from embodichain.lab.sim.cfg import LightCfg
from embodichain.lab.sim.common import BatchEntity
from embodichain.utils import logger

if TYPE_CHECKING:
    from dexsim.models import MeshObject
    from dexsim.scene import LightBatch, Scene, SpawnedLight

__all__ = ["Light"]


class Light(BatchEntity):
    """Environment-ordered lights owned by one Scene and updated in native batches."""

    def __init__(
        self,
        cfg: LightCfg,
        entities: Sequence[SpawnedLight],
        device: torch.device = torch.device("cpu"),
        *,
        scene: Scene,
    ) -> None:
        self._scene = scene
        self._light_handles = tuple(entities)
        self._light_batch: LightBatch | None = None
        self._batch_revision = -1
        self._is_destroyed = False
        super().__init__(cfg, [handle.native for handle in entities], device)
        self.reset()

    @property
    def _batch(self) -> LightBatch:
        """Reselect stable handles after physical topology changes."""
        if self._is_destroyed:
            raise RuntimeError("Light has been destroyed.")
        revision = self._scene.topology_revision
        if self._light_batch is None or self._batch_revision != revision:
            self._light_batch = self._scene.create_light_batch(self._light_handles)
            self._batch_revision = revision
        return self._light_batch

    def apply_state(
        self,
        *,
        color: torch.Tensor | None = None,
        intensity: torch.Tensor | None = None,
        position: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
        shadow_enabled: torch.Tensor | None = None,
        spot_inner_angle: torch.Tensor | None = None,
        spot_outer_angle: torch.Tensor | None = None,
        rect_width: torch.Tensor | None = None,
        rect_height: torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Validate and apply selected light fields in one native publication.

        Omitted fields and environments are preserved. Scalar fields and RGB/xyz
        vectors broadcast. CPU/CUDA inputs retain their device until the native
        adapter; support for CUDA input does not imply device-resident execution.
        Positions are parent-local translations, matching ``set_local_pose``.

        Args:
            color: RGB (3,) or (N,3).
            intensity: Legacy intensity scalar or (N,).
            position: Local position (3,) or (N,3).
            direction: Direction (3,) or (N,3) for directional light types.
            shadow_enabled: Scalar or (N,) nonzero flags.
            spot_inner_angle: Spot inner full cone angle in degrees.
            spot_outer_angle: Spot outer full cone angle in degrees.
            rect_width: Rectangular light width, scalar or (N,).
            rect_height: Rectangular light height, scalar or (N,).
            env_ids: Selected environment rows, in input order, or all rows.
        """
        if position is not None and self.is_global:
            logger.log_warning(
                f"Position ignored for global {self.cfg.light_type!r} light."
            )
            position = None
        self._batch.apply_state(
            color=color,
            intensity=intensity,
            position=position,
            direction=direction,
            shadow_enabled=None if shadow_enabled is None else shadow_enabled != 0,
            spot_inner_angle=spot_inner_angle,
            spot_outer_angle=spot_outer_angle,
            rect_width=rect_width,
            rect_height=rect_height,
            rows=env_ids,
            frame="local",
        )

    def set_color(
        self, colors: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Set RGB colors using one native batch.

        Args:
            colors: RGB (3,) broadcast or selected (N,3) values.
            env_ids: Selected environment rows, or all rows.
        """
        self._batch.apply_color(colors, rows=env_ids)

    def set_intensity(
        self,
        intensities: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set scalar or selected per-environment intensities.

        Args:
            intensities: A scalar, (1,), or selected (N,) values.
            env_ids: Selected environment rows, or all rows.
        """
        self._batch.apply_intensity(intensities, rows=env_ids)

    def set_falloff(
        self,
        falloffs: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Retain the legacy falloff setter, which has no effective SDK consumer.

        This unsupported field has no native batch implementation. It remains
        a compatibility path and is excluded from fused state/reset writes.

        Args:
            falloffs: A scalar, (1,), or selected (N,) values.
            env_ids: Selected environment rows, or all rows.
        """
        ids = range(self.num_instances) if env_ids is None else env_ids
        values = falloffs.detach().cpu().reshape(-1)
        for offset, row in enumerate(ids):
            self._entities[row].set_falloff(
                float(values[0 if len(values) == 1 else offset])
            )

    def set_direction(
        self,
        directions: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set emitted directions for directional light types in one batch.

        Args:
            directions: Direction (3,) or selected (N,3) values.
            env_ids: Selected environment rows, or all rows.
        """
        if self.cfg.light_type not in ("sun", "direction", "spot", "rect", "mesh"):
            logger.log_warning(
                f"set_direction not applicable to light type {self.cfg.light_type!r}, ignoring."
            )
            return
        self._batch.apply_direction(directions, rows=env_ids, frame="local")

    def set_spot_angle(
        self,
        inner_angles: torch.Tensor,
        outer_angles: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set both cone angles atomically for selected spot lights.

        Args:
            inner_angles: Inner full cone degrees, scalar or (N,).
            outer_angles: Outer full cone degrees, scalar or (N,).
            env_ids: Selected environment rows, or all rows.
        """
        if self.cfg.light_type != "spot":
            logger.log_warning(
                f"set_spot_angle not applicable to light type {self.cfg.light_type!r}, ignoring."
            )
            return
        self.apply_state(
            spot_inner_angle=inner_angles,
            spot_outer_angle=outer_angles,
            env_ids=env_ids,
        )

    def set_rect_wh(
        self,
        widths: torch.Tensor,
        heights: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set both rectangle dimensions in one native batch.

        Args:
            widths: Width scalar or selected (N,) values.
            heights: Height scalar or selected (N,) values.
            env_ids: Selected environment rows, or all rows.
        """
        if self.cfg.light_type != "rect":
            logger.log_warning(
                f"set_rect_wh not applicable to light type {self.cfg.light_type!r}, ignoring."
            )
            return
        self.apply_state(rect_width=widths, rect_height=heights, env_ids=env_ids)

    def set_mesh(
        self,
        mesh: MeshObject | SpawnedRigidBody | Sequence[MeshObject | SpawnedRigidBody],
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Bind selected mesh lights through the Scene's resource batch.

        Each light needs a distinct live Scene mesh in the same Arena. A single
        native MeshObject remains accepted for a single selected light. For
        multiple environments, supply corresponding meshes in selection order.

        Args:
            mesh: Scene mesh handles or native objects, one per selected light.
            env_ids: Selected environment rows, or all rows.
        """
        if self.cfg.light_type != "mesh":
            logger.log_warning(
                f"set_mesh not applicable to light type {self.cfg.light_type!r}, ignoring."
            )
            return
        count = self.num_instances if env_ids is None else len(env_ids)
        meshes = list(mesh) if isinstance(mesh, Sequence) else [mesh] * count
        native_meshes = {
            handle.get_user_id(): handle
            for handle in self._scene.handles.values()
            if isinstance(handle, SpawnedRigidBody)
        }
        handles = [
            (
                entry
                if isinstance(entry, SpawnedRigidBody)
                else native_meshes[entry.get_user_id()]
            )
            for entry in meshes
        ]
        self._batch.apply_mesh_binding(handles, rows=env_ids)

    def enable_shadow(
        self, flags: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Enable shadows for scalar or selected per-environment flags.

        Args:
            flags: Scalar or (N,) values; nonzero enables the shadow.
            env_ids: Selected environment rows, or all rows.
        """
        self._batch.apply_shadow_enabled(flags != 0, rows=env_ids)

    @property
    def is_global(self) -> bool:
        """Whether this is a single scene-wide sun or directional light."""
        return self.cfg.light_type in ("sun", "direction")

    def set_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        to_matrix: bool = False,
    ) -> None:
        """Patch local translation while preserving orientation and scale.

        Args:
            pose: (3,)/(N,3), or (4,4)/(N,4,4) with ``to_matrix=True``.
            env_ids: Selected environment rows, or all rows.
            to_matrix: Extract translation from input matrices.
        """
        if self.is_global:
            logger.log_warning(
                f"set_local_pose not applicable to {self.cfg.light_type!r} light type (infinite distance, direction only). Use set_direction() instead."
            )
            return
        if to_matrix:
            if pose.shape[-2:] != (4, 4):
                raise ValueError("Expected (4,4) or (N,4,4) light matrices.")
            pose = pose[..., :3, 3]
        self._batch.apply_position(pose, rows=env_ids, frame="local")

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Read all local matrices or translations in one batch.

        Args:
            to_matrix: Return complete (N,4,4) matrices when true.

        Returns:
            Owned CPU float32 tensor, preserving the existing light API.
        """
        values = (
            self._batch.fetch_pose(frame="local")
            if to_matrix
            else self._batch.fetch_position(frame="local")
        )
        return torch.as_tensor(values, dtype=torch.float32)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Restore selected initial light properties with one fused write.

        Single-instance lights normalize selections to all rows as before.
        Falloff has no applied consumer and is not part of the native batch.

        Args:
            env_ids: Selected environment rows, or all rows.
        """
        if self.num_instances == 1:
            env_ids = None
        cfg = self.cfg
        fields = dict(
            color=torch.as_tensor(cfg.color),
            intensity=torch.as_tensor(cfg.intensity),
            shadow_enabled=torch.as_tensor(cfg.enable_shadow),
        )
        if not self.is_global:
            fields["position"] = torch.as_tensor(cfg.init_pos)
        if cfg.light_type in ("sun", "direction", "spot", "rect", "mesh"):
            fields["direction"] = torch.as_tensor(cfg.direction)
        if cfg.light_type == "spot":
            fields.update(
                spot_inner_angle=torch.as_tensor(cfg.spot_angle_inner),
                spot_outer_angle=torch.as_tensor(cfg.spot_angle_outer),
            )
        if cfg.light_type == "rect":
            fields.update(
                rect_width=torch.as_tensor(cfg.rect_width),
                rect_height=torch.as_tensor(cfg.rect_height),
            )
        self.apply_state(**fields, env_ids=env_ids)

    def destroy(self) -> None:
        """Remove native lights as one batch before releasing borrowed views."""
        if self._is_destroyed:
            return
        self._light_batch = None
        self._scene.remove_lights(self._light_handles)
        self._light_handles = ()
        self._entities = []
        self._scene = None
        self._is_destroyed = True
