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

import numpy as np
import torch

from embodichain.lab.sim.cfg import LightCfg
from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim._render_selection import (
    _normalize_render_rows,
    _select_render_values,
)
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

    def get_color(self) -> torch.Tensor:
        """Return owned native RGB values for each light instance."""
        return torch.tensor(
            [list(entity.get_color()) for entity in self._entities],
            dtype=torch.float32,
            device=self.device,
        )

    def get_intensity(self) -> torch.Tensor:
        """Return owned native intensity values for each light instance."""
        return torch.tensor(
            [float(entity.get_intensity()) for entity in self._entities],
            dtype=torch.float32,
            device=self.device,
        )

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
        rows, offsets = _normalize_render_rows(env_ids, self.num_instances)
        fields = {
            "color": color,
            "intensity": intensity,
            "position": position,
            "direction": direction,
            "shadow_enabled": None if shadow_enabled is None else shadow_enabled != 0,
            "spot_inner_angle": spot_inner_angle,
            "spot_outer_angle": spot_outer_angle,
            "rect_width": rect_width,
            "rect_height": rect_height,
        }
        fields = {
            name: _select_render_values(
                value, offsets, 1 if name in ("color", "position", "direction") else 0
            )
            for name, value in fields.items()
            if value is not None
        }
        self._batch.apply_state(**fields, rows=rows, frame="local")

    def set_color(
        self, colors: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Set RGB colors using one native batch.

        Args:
            colors: RGB (3,) broadcast or selected (N,3) values.
            env_ids: Selected environment rows, or all rows.
        """
        rows, offsets = _normalize_render_rows(env_ids, self.num_instances)
        self._batch.apply_color(_select_render_values(colors, offsets, 1), rows=rows)

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
        rows, offsets = _normalize_render_rows(env_ids, self.num_instances)
        self._batch.apply_intensity(
            _select_render_values(intensities, offsets, 0), rows=rows
        )

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
        rows, offsets = _normalize_render_rows(env_ids, self.num_instances)
        self._batch.apply_direction(
            _select_render_values(directions, offsets, 1), rows=rows, frame="local"
        )

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
        mesh: "MeshObject",
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set the mesh for mesh-type lights.

        Only applies to ``mesh`` light type. Logs a warning and no-ops for other types.
        This is NOT tensor-batched — the same MeshObject is assigned to all targeted
        instances.

        Args:
            mesh (MeshObject): The mesh object to assign to the light.
            env_ids (Sequence[int] | None): Indices of instances to set. If None,
                applies to all instances.
        """
        if self.cfg.light_type != "mesh":
            logger.log_warning(
                f"set_mesh not applicable to light type "
                f"'{self.cfg.light_type}', ignoring."
            )
            return

        if env_ids is None:
            target_ids = list(range(self.num_instances))
        else:
            target_ids = list(env_ids)

        for i in target_ids:
            try:
                self._entities[i].set_mesh(mesh)
            except Exception as e:
                logger.log_error(f"set_mesh: error for instance {i}: {e}")

    def enable_shadow(
        self, flags: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Enable shadows for scalar or selected per-environment flags.

        Args:
            flags: Scalar or (N,) values; nonzero enables the shadow.
            env_ids: Selected environment rows, or all rows.
        """
        rows, offsets = _normalize_render_rows(env_ids, self.num_instances)
        self._batch.apply_shadow_enabled(
            _select_render_values(flags, offsets, 0) != 0, rows=rows
        )

    @property
    def is_global(self) -> bool:
        """Whether this is a single scene-wide sun or directional light."""
        return self.cfg.light_type in ("sun", "direction")

    def set_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        to_matrix: bool = False,
    ) -> None:
        """Set local pose (translation) for one or more lights.

        Args:
            pose (torch.Tensor):
                - If to_matrix=False: shape (3,) or (M, 3), representing (x, y, z).
                - If to_matrix=True: shape (4, 4) or (M, 4, 4); translation extracted automatically.
            env_ids (Sequence[int] | None): Indices to set. If None:
                - For vector input (3,) broadcast to all, or (M,3) with M == num_instances.
                - For matrix input (4,4) broadcast to all, or (M,4,4) with M == num_instances.
            to_matrix (bool): Interpret `pose` as full 4x4 matrix if True, else as vector(s).
        """
        if self.is_global:
            logger.log_warning(
                f"set_local_pose not applicable to '{self.cfg.light_type}' light type "
                f"(infinite distance, direction only). Use set_direction() instead."
            )
            return

        if not torch.is_tensor(pose):
            logger.log_error(
                f"set_local_pose requires a torch.Tensor, got {type(pose)}"
            )
            return

        cpu = pose.detach().cpu()
        if to_matrix:
            if cpu.ndim == 2 and cpu.shape == (4, 4):
                trans = cpu[:3, 3]
            elif cpu.ndim == 3 and cpu.shape[1:] == (4, 4):
                trans = cpu[..., 0:3, 3]
            else:
                logger.log_error(
                    f"set_local_pose matrix: expected (4,4) or (N,4,4), got {tuple(cpu.shape)}"
                )
                return
        else:
            trans = cpu  # expect (3,) or (M,3)

        try:
            self._apply_vector3(trans, env_ids, setter_name="set_location")
        except Exception as e:
            logger.log_error(f"set_local_pose: error while applying translation: {e}")

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Get local pose of each light, either as full matrix or translation vector.

        Args:
            to_matrix (bool, optional): If True, return poses as 4×4 matrices.
                If False, return translations only as (x, y, z). Defaults to False.
        Returns:
            torch.Tensor:
                - If to_matrix=True: Tensor of shape (N, 4, 4), where N == num_instances.
                - If to_matrix=False: Tensor of shape (N, 3), containing translations.
                On error or empty instances, returns an empty tensor with shape (0, 4, 4) or (0, 3) respectively, and logs via logger.log_error.
        """
        mats = []
        for i in range(self.num_instances):
            try:
                mat = self._entities[i].get_local_pose()  # expect numpy (4,4)
                arr = np.array(mat, dtype=np.float32)
                if arr.shape != (4, 4):
                    logger.log_error(
                        f"get_local_pose: unexpected shape {arr.shape} for instance {i}"
                    )
                    return torch.empty(
                        (0, 4, 4) if to_matrix else (0, 3), dtype=torch.float32
                    )
                mats.append(arr)
            except Exception as e:
                logger.log_error(f"get_local_pose: error for instance {i}: {e}")
                return torch.empty(
                    (0, 4, 4) if to_matrix else (0, 3), dtype=torch.float32
                )

        if not mats:
            return torch.empty((0, 4, 4) if to_matrix else (0, 3), dtype=torch.float32)

        stacked = np.stack(mats, axis=0)  # (N,4,4)
        tensor4 = torch.from_numpy(stacked)
        if to_matrix:
            return tensor4
        # else return translations
        return tensor4[:, 0:3, 3].clone()

    def _apply_vector3(
        self,
        tensor: torch.Tensor,
        env_ids: Sequence[int] | None,
        setter_name: str,
    ) -> None:
        """
        Generic helper for 3-element vectors (color, location).
        Expects tensor shape: (3,), or (M,3) with M == num_instances or M == len(env_ids).
        env_ids: Sequence[int] | None
        """
        # Validate tensor type
        if not torch.is_tensor(tensor):
            logger.log_error(
                f"{setter_name} requires a torch.Tensor, got {type(tensor)}"
            )
            return

        cpu = tensor.detach().cpu()
        # Determine target indices
        if env_ids is None:
            all_ids = list(range(self.num_instances))
        else:
            all_ids = list(env_ids)

        # Cases:
        # 1) cpu.ndim == 1 and size == 3: broadcast to all_ids
        if cpu.ndim == 1 and cpu.shape[0] == 3:
            arr = cpu.numpy()
            for i in all_ids:
                getattr(self._entities[i], setter_name)(
                    float(arr[0]), float(arr[1]), float(arr[2])
                )
            return

        # 2) cpu.ndim == 2 and cpu.shape == (num_instances, 3), env_ids None or full
        if cpu.ndim == 2 and cpu.shape == (self.num_instances, 3) and env_ids is None:
            arr_all = cpu.numpy()
            for i in range(self.num_instances):
                getattr(self._entities[i], setter_name)(
                    float(arr_all[i, 0]), float(arr_all[i, 1]), float(arr_all[i, 2])
                )
            return

        # 3) cpu.ndim == 2 and env_ids provided, cpu.shape == (len(env_ids), 3)
        if (
            cpu.ndim == 2
            and env_ids is not None
            and cpu.shape[0] == len(all_ids)
            and cpu.shape[1] == 3
        ):
            arr_sel = cpu.numpy()
            for idx, i in enumerate(all_ids):
                getattr(self._entities[i], setter_name)(
                    float(arr_sel[idx, 0]),
                    float(arr_sel[idx, 1]),
                    float(arr_sel[idx, 2]),
                )
            return

        logger.log_error(
            f"{setter_name}: tensor shape {tuple(cpu.shape)} is invalid for broadcasting "
            f"(expected (3,) or ({self.num_instances},3) or ({len(all_ids)},3))."
        )

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
