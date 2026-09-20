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

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from ..protocol import MeshMarkerOverlay

if TYPE_CHECKING:
    from dexsim.scene import Scene
    from dexsim.scene.objects import SpawnedRigidBody


@dataclass
class _Entry:
    handle: SpawnedRigidBody
    snapshot: MeshMarkerOverlay


def _pose(marker: MeshMarkerOverlay) -> np.ndarray:
    w, x, y, z = marker.wxyz
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    pose[:3, 3] = marker.position
    return pose


class NativeMarkerRenderer:
    """Own Scene render-only mesh handles on the simulation thread.

    Args:
        scene: Prepared DexSim Scene owning render-only mesh objects.
            Creating markers does not prepare or rebuild physics.
    """

    def __init__(self, scene: Scene) -> None:
        try:
            spawn = import_module("dexsim.spawn")
            required = (
                (getattr(spawn, "RenderDesc", None), "render_mode"),
                (getattr(spawn, "MaterialDesc", None), "unlit"),
                (getattr(spawn, "MaterialDesc", None), "alpha_mode"),
                (getattr(spawn, "MaterialDesc", None), "depth_write"),
            )
            if (
                not hasattr(spawn, "MeshObjectDesc")
                or any(not hasattr(owner, name) for owner, name in required)
                or not callable(getattr(scene, "add_mesh_object", None))
                or not callable(getattr(scene, "remove_mesh_object", None))
            ):
                raise ImportError("Missing Scene mesh-object or overlay properties")
        except ImportError as exc:
            raise RuntimeError(
                "Native marker groups require Scene mesh-object overlay support. "
                "Install a DexSim build with render-only overlay support or use Viser."
            ) from exc
        self._spawn = spawn
        self._scene = scene
        self._groups: dict[str, dict[str, _Entry]] = {}

    def _create(self, marker: MeshMarkerOverlay) -> SpawnedRigidBody:
        """Declare one render-only object through its lifetime-owning Scene."""
        name = f"__embodichain_marker_{uuid4().hex}"
        material = self._spawn.MaterialDesc(
            name=f"{name}_material",
            base_color=tuple(marker.color),
            unlit=True,
            alpha_mode="blend",
            depth_write=False,
            double_sided=True,
        )
        render = self._spawn.RenderDesc.from_geometry(
            self._spawn.GeometryDesc.mesh(
                vertices=np.ascontiguousarray(marker.vertices, dtype=np.float32),
                triangles=np.ascontiguousarray(marker.faces, dtype=np.int32),
            ),
            render_mode="overlay",
            cast_shadow=False,
            pickable=False,
            material=material,
        )
        desc = self._spawn.MeshObjectDesc(
            name=name,
            renders=[render],
            physics=None,
            per_env=False,
        )
        return self._scene.add_mesh_object(desc, arena_name="default")

    def _discard(self, handle: SpawnedRigidBody) -> None:
        # Scene.close() invalidates stable handles before native resources retire.
        # Never inspect a borrowed native actor to decide how to remove it.
        if handle.is_valid:
            self._scene.remove_mesh_object(handle.path)

    @staticmethod
    def _apply(handle: SpawnedRigidBody, marker: MeshMarkerOverlay) -> None:
        handle.set_world_pose(_pose(marker))
        handle.set_scale(*(float(value) for value in marker.scale))
        handle.get_material().set_base_color(marker.color)
        handle.set_visible(marker.visible)

    def publish(self, name: str, markers: tuple[MeshMarkerOverlay, ...]) -> None:
        """Apply a complete group, staging replacements before releasing old meshes.

        Args:
            name: Stable group name.
            markers: Complete detached group snapshots in world coordinates.
        """
        previous = self._groups.get(name, {})
        pending: dict[str, _Entry] = {}
        created = []
        touched = []
        try:
            for marker in markers:
                entry = previous.get(marker.overlay_id)
                if (
                    entry is not None
                    and np.array_equal(entry.snapshot.vertices, marker.vertices)
                    and np.array_equal(entry.snapshot.faces, marker.faces)
                ):
                    handle = entry.handle
                    touched.append(entry)
                else:
                    handle = self._create(marker)
                    created.append(handle)
                self._apply(handle, marker)
                pending[marker.overlay_id] = _Entry(handle, marker)
        except Exception:
            for handle in created:
                self._discard(handle)
            for entry in touched:
                self._apply(entry.handle, entry.snapshot)
            raise
        for key, entry in previous.items():
            if key not in pending or pending[key].handle is not entry.handle:
                self._discard(entry.handle)
        self._groups[name] = pending

    def remove(self, name: str) -> None:
        """Release all native objects in one group.

        Args:
            name: Group to release; absent groups are ignored.
        """
        for entry in self._groups.pop(name, {}).values():
            self._discard(entry.handle)

    def close(self) -> None:
        """Release marker handles; also safe after their Scene has closed."""
        for name in tuple(self._groups):
            self.remove(name)
