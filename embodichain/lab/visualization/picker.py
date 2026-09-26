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
"""Backend-neutral ray-mesh picking for Viser click selection.

Viser's ``on_pointer_event`` callback exposes the camera ray but not the scene
node it hits. :class:`ScenePicker` closes that gap by ray-casting the ray
against the cached scene geometry with a vectorized Möller-Trumbore test,
returning the closest hit node so the simulation can attach a Gizmo to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = ["ScenePicker"]

_EPSILON = 1.0e-9


@dataclass(frozen=True)
class _Geometry:
    """Cached triangle data for one geometry, stored in local coordinates."""

    v0: np.ndarray
    edge1: np.ndarray
    edge2: np.ndarray


def _wxyz_to_rotation(wxyz: np.ndarray) -> np.ndarray:
    """Convert a normalized wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(wxyz, dtype=np.float64)
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation


class ScenePicker:
    """Resolve a world-space ray to the closest hit scene node.

    Geometry is cached per ``geometry_id`` in local coordinates. Each pick
    transforms the ray into every instance's local frame (so cached triangle
    data is reused across instances and across frames) and runs a vectorized
    Möller-Trumbore test, keeping the smallest positive ray parameter.

    Args:
        epsilon: Lower bound for accepted ray parameters, in world length units.
    """

    def __init__(self, epsilon: float = _EPSILON) -> None:
        self._geometries: dict[str, _Geometry] = {}
        self._epsilon = float(epsilon)

    def set_geometry(
        self,
        geometry_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        """Cache one geometry's triangle data in local coordinates.

        Args:
            geometry_id: Stable geometry identifier from the scene manifest.
            vertices: Triangle mesh vertices with shape ``(V, 3)``.
            faces: Triangle indices into ``vertices`` with shape ``(F, 3)``.
        """
        verts = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32))
        tris = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (V, 3), received {verts.shape}."
            )
        if tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError(f"faces must have shape (F, 3), received {tris.shape}.")
        if tris.size == 0:
            self._geometries.pop(geometry_id, None)
            return
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        self._geometries[geometry_id] = _Geometry(
            v0=v0,
            edge1=v1 - v0,
            edge2=v2 - v0,
        )

    def remove_geometry(self, geometry_id: str) -> None:
        """Drop one cached geometry."""
        self._geometries.pop(geometry_id, None)

    def clear(self) -> None:
        """Drop all cached geometry."""
        self._geometries.clear()

    def pick(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        instances: Iterable[tuple[str, str, np.ndarray, np.ndarray]],
    ) -> str | None:
        """Return the node id of the closest instance hit by the ray.

        Each instance is a ``(node_id, geometry_id, position, wxyz)`` tuple,
        where ``position`` is the world-space translation and ``wxyz`` is the
        normalized ``[w, x, y, z]`` quaternion. The ray is transformed into each
        instance's local frame so the cached local geometry can be reused.

        Args:
            ray_origin: World-space ray origin with shape ``(3,)``.
            ray_direction: World-space ray direction with shape ``(3,)``. It is
                normalized internally so the returned hit distance is in world
                length units.
            instances: Iterable of scene instances to test.

        Returns:
            The closest hit ``node_id``, or ``None`` if the ray misses every
            instance.
        """
        origin = np.asarray(ray_origin, dtype=np.float32)
        direction = np.asarray(ray_direction, dtype=np.float32)
        if origin.shape != (3,) or direction.shape != (3,):
            raise ValueError("ray_origin and ray_direction must have shape (3,).")
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm <= self._epsilon:
            return None
        direction = direction / dir_norm

        best_node: str | None = None
        best_t = np.inf
        for node_id, geometry_id, position, wxyz in instances:
            geometry = self._geometries.get(geometry_id)
            if geometry is None:
                continue
            local_origin, local_direction = self._world_to_local_ray(
                origin, direction, position, wxyz
            )
            hit_t = self._ray_cast_geometry(geometry, local_origin, local_direction)
            if hit_t is not None and hit_t < best_t:
                best_t = hit_t
                best_node = node_id
        return best_node

    @staticmethod
    def _world_to_local_ray(
        origin: np.ndarray,
        direction: np.ndarray,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform a world ray into an instance's local frame.

        The direction is left unnormalized after the inverse rotation so the ray
        parameter stays in world length units: the local triangle hit parameter
        equals the world-space distance along the (normalized) world ray.
        """
        rotation = _wxyz_to_rotation(wxyz)
        inv_rotation = rotation.T
        local_origin = inv_rotation @ (origin - np.asarray(position, dtype=np.float32))
        local_direction = inv_rotation @ direction
        return local_origin.astype(np.float32), local_direction.astype(np.float32)

    def _ray_cast_geometry(
        self,
        geometry: _Geometry,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> float | None:
        """Return the smallest positive ray parameter hitting one geometry."""
        edge1 = geometry.edge1
        edge2 = geometry.edge2
        v0 = geometry.v0

        h = np.cross(direction, edge2)  # (F, 3)
        a = np.einsum("fd,fd->f", edge1, h)  # (F,)
        parallel = np.abs(a) <= self._epsilon
        # Avoid division by zero for parallel rays; mask them out later.
        safe_a = np.where(parallel, 1.0, a)
        f = 1.0 / safe_a
        s = origin - v0  # (F, 3)
        u = f * np.einsum("fd,fd->f", s, h)
        q = np.cross(s, edge1)  # (F, 3)
        v = f * np.einsum("d,fd->f", direction, q)
        t = f * np.einsum("fd,fd->f", edge2, q)

        valid = (
            (~parallel)
            & (u >= 0.0)
            & (u <= 1.0)
            & (v >= 0.0)
            & (u + v <= 1.0)
            & (t > self._epsilon)
        )
        if not np.any(valid):
            return None
        return float(np.min(t[valid]))
