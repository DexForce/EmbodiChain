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

"""Explicit FCL convex-query adapter for the source-qualified geometry policy."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import math
import numpy as np
import trimesh
from ._collision_backends import _require_fcl

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _ExactClearance:
    lower: float

    def below(self, threshold: float) -> bool:
        return self.lower <= threshold


class _FclCollection:
    """Keep the original FCL query and threshold semantics behind a typed port."""

    def __init__(self) -> None:
        _require_fcl()
        self._manager = trimesh.collision.CollisionManager()

    @staticmethod
    def metadata() -> dict[str, str]:
        return {
            "collision_backend": "trimesh.collision.CollisionManager/python-fcl",
            "collision_backend_version": version("python-fcl"),
            "distance_semantics": "source_fcl_min_distance/v1",
        }

    def _require_open(self) -> None:
        if self._manager is None:
            raise RuntimeError("FCL query is closed.")

    def add_object(
        self, name: str, mesh: trimesh.Trimesh, transform: np.ndarray | None = None
    ) -> None:
        self._require_open()
        self._manager.add_object(name, mesh, transform=transform)

    def set_transform(self, name: str, transform: np.ndarray) -> None:
        self._require_open()
        self._manager.set_transform(name, transform)

    def distance_interval(self, other: _FclCollection, limit: float) -> _ExactClearance:
        self._require_open()
        if not isinstance(other, _FclCollection):
            raise TypeError("FCL queries require the explicitly selected FCL backend.")
        other._require_open()
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("A collision query needs a finite nonnegative threshold.")
        distance = float(self._manager.min_distance_other(other._manager))
        if not math.isfinite(distance):
            raise ValueError("Native mesh distance is unavailable")
        return _ExactClearance(distance)

    def close(self) -> None:
        self._manager = None
