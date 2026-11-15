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


"""Bind named pad and opposing shapes to a completed DexSim surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from embodichain.compute.geometry._warp.hydroelastic_grid import (
    finish_raster,
    scatter_faces,
)
from embodichain.utils.device_utils import current_warp_stream

from .attachment import _resolve_link

if TYPE_CHECKING:
    from dexsim.scene import HydroelasticContactSurface, Scene
    from tensordict import TensorDict

    from embodichain.lab.sim.sim_manager import SimulationManager
    from .base_sensor import SensorCfg


class _HydroelasticPadSource:
    """Own a topology-scoped mapping from native shapes to environment rows."""

    def __init__(
        self,
        manager: SimulationManager,
        mount: SensorCfg.OffsetCfg,
        targets: tuple[str, ...],
    ) -> None:
        scene = manager.spawn_result
        if scene is None or not callable(
            getattr(scene, "get_hydroelastic_contact_binding", None)
        ):
            raise NotImplementedError(
                "Hydroelastic pad projection requires a DexSim SDK with completed contact bindings."
            )
        if mount.parent is None:
            raise ValueError(
                "Hydroelastic pad projection requires a physical mount parent."
            )
        self.scene: Scene = scene
        self.topology = scene.topology_revision
        env_by_name = {
            name: row for row, name in enumerate(manager._spawn_scene.arena_names)
        }
        pad: dict[int, int] = {}
        opposing: dict[int, int] = {}
        body_ids = np.full(manager.num_envs, -1, dtype=np.int32)
        static_inverse = np.tile(np.eye(4, dtype=np.float32), (manager.num_envs, 1, 1))
        bound_rows = set()
        asset = mount.parent
        pad_uid, link = self._resolve_owner(manager, asset)
        for handle in manager._spawn_scene.handles(pad_uid):
            env = env_by_name[handle.arena_name]
            info = scene.get_hydroelastic_contact_binding(
                (handle, link) if link else handle
            )
            if not info.shape_ids:
                raise ValueError(f"Hydroelastic pad {asset!r} has no collision shapes.")
            bound_rows.add(env)
            pad.update((shape, env) for shape in info.shape_ids)
            if info.body_index is None:
                static_inverse[env] = np.linalg.inv(info.fixed_world_pose)
            else:
                body_ids[env] = info.body_index
        if len(bound_rows) != manager.num_envs:
            raise ValueError(
                "Hydroelastic pad requires a physical mount in every arena."
            )
        for target in targets:
            uid, link = self._resolve_owner(manager, target)
            for handle in manager._spawn_scene.handles(uid):
                env = env_by_name[handle.arena_name]
                info = scene.get_hydroelastic_contact_binding(
                    (handle, link) if link else handle
                )
                opposing.update((shape, env) for shape in info.shape_ids)
        if not opposing or pad.keys() & opposing.keys():
            raise ValueError(
                "Hydroelastic opposing targets must be distinct collision shapes."
            )
        count = max(max(pad), max(opposing)) + 1
        pad_env = np.full(count, -1, dtype=np.int32)
        target_env = pad_env.copy()
        for index, env in pad.items():
            pad_env[index] = env
        for index, env in opposing.items():
            target_env[index] = env
        self.pad_env = torch.as_tensor(pad_env, device=manager.device)
        self.target_env = torch.as_tensor(target_env, device=manager.device)
        self.body_ids = torch.as_tensor(body_ids, device=manager.device)
        self.static_inverse = torch.as_tensor(static_inverse, device=manager.device)
        transform = np.linalg.inv(mount.transformation.detach().cpu().numpy())
        self.mount_inverse = wp.mat44(*transform.ravel())

    @staticmethod
    def _resolve_owner(
        manager: SimulationManager, parent: str
    ) -> tuple[str, str | None]:
        if parent in manager._rigid_objects:
            return parent, None
        uid, _, link = _resolve_link(
            parent, {**manager._articulations, **manager._robots}, manager.num_envs
        )
        return uid, link

    def read(self) -> HydroelasticContactSurface | None:
        if self.topology != self.scene.topology_revision or self.scene.needs_rebuild:
            return None
        surface = self.scene.get_hydroelastic_contact_surface()
        if surface is None or surface.step != self.scene.get_simulation_step():
            return None
        return surface if surface.topology_revision == self.topology else None


class _HydroelasticPadProjection:
    def __init__(
        self,
        source: _HydroelasticPadSource,
        points: torch.Tensor,
        normals: torch.Tensor,
        data: TensorDict,
        size: tuple[float, float],
        resolution: tuple[int, int],
        regular: bool,
        max_distance: float,
    ) -> None:
        self._source = source
        self._device = points.device
        self._data = data
        width, height = resolution
        self._inputs = [
            wp.from_torch(source.pad_env),
            wp.from_torch(source.target_env),
            wp.from_torch(source.body_ids),
            wp.from_torch(source.static_inverse, dtype=wp.mat44),
            source.mount_inverse,
            wp.from_torch(points, dtype=wp.vec3),
            wp.from_torch(normals.contiguous(), dtype=wp.vec3),
            regular,
            *size,
            width,
            height,
            max_distance,
        ]
        self._counts = wp.from_torch(data["surface_face_count"].flatten(1))
        self._closest = wp.from_torch(data["surface_separation"].flatten(1))
        self._complete = wp.from_torch(data["surface_complete"])

    def reset(self, rows: slice | torch.Tensor) -> None:
        for key in ("surface_face_count", "surface_separation", "surface_complete"):
            self._data[key][rows] = 0
        for key in ("surface_generation", "surface_substep"):
            self._data[key][rows] = -1

    def sample(self) -> None:
        snapshot = self._source.read()
        if snapshot is None:
            self.reset(slice(None))
            return
        self._data["surface_face_count"].zero_()
        self._data["surface_separation"].fill_(float("inf"))
        count = wp.from_torch(snapshot.face_count)
        device = str(self._device)
        stream = current_warp_stream(self._device)
        if snapshot.capacity:
            wp.launch(
                scatter_faces,
                dim=snapshot.capacity,
                inputs=[
                    count,
                    wp.from_torch(snapshot.vertices, dtype=wp.vec3),
                    wp.from_torch(snapshot.shape_pairs, dtype=wp.vec2i),
                    wp.from_torch(snapshot.separation),
                    wp.from_torch(snapshot.body_poses, dtype=wp.transform),
                    *self._inputs,
                ],
                outputs=[self._counts, self._closest],
                device=device,
                stream=stream,
            )
        wp.launch(
            finish_raster,
            dim=self._counts.shape,
            inputs=[count, snapshot.capacity, self._counts],
            outputs=[self._closest, self._complete],
            device=device,
            stream=stream,
        )
        self._data["surface_generation"].fill_(snapshot.collision_generation)
        self._data["surface_substep"].fill_(snapshot.substep)
