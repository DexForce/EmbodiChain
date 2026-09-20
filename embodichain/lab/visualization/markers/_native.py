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
from typing import Any
from uuid import uuid4

import numpy as np

from ..protocol import MeshMarkerOverlay


@dataclass
class _Entry:
    handle: Any
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
    """Own native render-only mesh handles on the simulation thread.

    Args:
        arena: Global DexSim arena owning generic render-only mesh actors.
            Unsupported native rendering capabilities raise an actionable error.
    """

    def __init__(self, arena: Any) -> None:
        try:
            engine = import_module("dexsim.engine")
            required = (
                (getattr(engine, "RenderBody", None), "set_raytrace_visible"),
                (getattr(engine, "RenderBody", None), "set_pickable"),
                (getattr(engine, "RenderBody", None), "build"),
                (getattr(engine, "MaterialInst", None), "set_unlit"),
                (getattr(engine, "MaterialInst", None), "set_alpha_mode"),
                (getattr(engine, "AlphaMode", None), "BLEND"),
            )
            if any(not hasattr(owner, name) for owner, name in required):
                raise ImportError("Missing native render-body or material properties")
        except ImportError as exc:
            raise RuntimeError(
                "Native marker groups require generic RenderBody and MaterialInst "
                "overlay properties. Install a DexSim build with render-only "
                "overlay support or use Viser."
            ) from exc
        self._alpha_blend = engine.AlphaMode.BLEND
        self._arena = arena
        self._groups: dict[str, dict[str, _Entry]] = {}

    def _create(self, marker: MeshMarkerOverlay) -> Any:
        """Configure one ordinary MeshObject before its first GPU build."""
        name = f"__embodichain_marker_{uuid4().hex}"
        actor = self._arena.create_actor(name, True, False)
        if actor is None:
            raise RuntimeError("DexSim failed to create a marker MeshObject.")
        body = material = None
        try:
            body = actor.get_render_body()
            body.set_raytrace_visible(False)
            body.set_shadow(False)
            body.set_pickable(False)
            mesh_id = body.add_mesh(
                np.ascontiguousarray(marker.vertices, dtype=np.float32),
                np.ascontiguousarray(marker.faces, dtype=np.int32).reshape(-1),
                auto_build=False,
            )
            if mesh_id < 0:
                raise RuntimeError("DexSim failed to add marker mesh geometry.")
            material = body.default_material()
            if body.set_material(mesh_id, material, owned=True) <= 0:
                raise RuntimeError("DexSim failed to assign marker material.")
            material.set_base_color(marker.color)
            material.set_unlit(True)
            material.set_alpha_mode(self._alpha_blend)
            material.set_depth_write(False)
            material.set_double_sided(True)
            if not body.build():
                raise RuntimeError("DexSim failed to build marker mesh geometry.")
            if self._arena.attach(actor.node) < 0:
                raise RuntimeError("DexSim failed to attach marker MeshObject.")
            return actor
        except Exception:
            # Release wrapper references before removal so owned resources can
            # retire even while the exception traceback is retained.
            body = material = None
            self._arena.remove_actor(name)
            raise

    @staticmethod
    def _apply(handle: Any, marker: MeshMarkerOverlay) -> None:
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
                self._arena.remove_actor(handle.get_name())
            for entry in touched:
                self._apply(entry.handle, entry.snapshot)
            raise
        for key, entry in previous.items():
            if key not in pending or pending[key].handle is not entry.handle:
                self._arena.remove_actor(entry.handle.get_name())
        self._groups[name] = pending

    def remove(self, name: str) -> None:
        """Release all native objects in one group.

        Args:
            name: Group to release; absent groups are ignored.
        """
        for entry in self._groups.pop(name, {}).values():
            self._arena.remove_actor(entry.handle.get_name())

    def close(self) -> None:
        """Release all native marker handles before arena destruction."""
        for name in tuple(self._groups):
            self.remove(name)
