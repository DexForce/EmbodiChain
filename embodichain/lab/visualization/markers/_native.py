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
from typing import Any

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
        arena: Global DexSim arena exposing ``create_debug_mesh``. Older
            engines raise an actionable error instead of drawing into sensors.
    """

    def __init__(self, arena: Any) -> None:
        if not callable(getattr(arena, "create_debug_mesh", None)):
            raise RuntimeError(
                "Native marker groups require DexSim Arena.create_debug_mesh. "
                "Install a DexSim build with debug-mesh overlay support or use Viser."
            )
        self._arena = arena
        self._groups: dict[str, dict[str, _Entry]] = {}

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
                    handle = self._arena.create_debug_mesh(
                        marker.vertices, marker.faces, marker.color
                    )
                    if handle is None:
                        raise RuntimeError("DexSim failed to create a debug mesh.")
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
