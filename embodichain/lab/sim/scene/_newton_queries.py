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

"""Owned convex-set adapters for the shared isolated Newton collision pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from importlib.metadata import version
import math
import numpy as np
import trimesh
from .feasibility.collision import (
    CollisionBody,
    NewtonCollisionChecker,
    UnsupportedFeasibilityError,
)
from ._collision_backends import _require_newton

__all__: list[str] = []


@dataclass(frozen=True)
class _ClearanceInterval:
    lower: float
    upper: float

    def below(self, threshold: float) -> bool:
        if self.upper <= threshold:
            return True
        if self.lower > threshold:
            return False
        raise UnsupportedFeasibilityError(
            "Newton precision interval crosses the contact threshold; geometry is unavailable."
        )


class _ConvexCollection:
    """Keep authored convex geometry and transforms separate from live simulation."""

    def __init__(self) -> None:
        self._objects = {}
        self._queries = {}
        self._frozen = False
        self._closed = False
        self._identity = object()
        _require_newton()

    @staticmethod
    def metadata() -> dict[str, str]:
        """Describe the actual shared backend used by injected mesh queries."""
        return {
            "collision_backend": "newton.CollisionPipeline",
            "collision_backend_version": version("newton"),
            "distance_semantics": "capped_conversion_interval/v1",
        }

    def add_object(self, name: str, mesh: trimesh.Trimesh, transform=None) -> None:
        if self._closed or self._frozen:
            raise RuntimeError("Query geometry is closed or already bound.")
        if name in self._objects:
            raise ValueError("Duplicate collision shape identity.")
        if (
            not mesh.is_convex
            or not mesh.is_volume
            or not np.isfinite(mesh.vertices).all()
        ):
            raise UnsupportedFeasibilityError(
                "Native query requires qualified convex solids; triangle surfaces are not convexified."
            )
        self._objects[name] = (
            mesh.copy(),
            np.eye(4) if transform is None else np.array(transform, copy=True),
        )

    def set_transform(self, name: str, transform: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Collision query is closed.")
        mesh, _ = self._objects[name]
        self._objects[name] = mesh, np.array(transform, copy=True)

    def distance_interval(
        self, other: _ConvexCollection, limit: float
    ) -> _ClearanceInterval:
        if self._closed or other._closed:
            raise RuntimeError("Collision query is closed.")
        if (
            not math.isfinite(limit)
            or limit < 0
            or not self._objects
            or not other._objects
        ):
            raise ValueError(
                "A bounded distance needs nonempty shapes and a finite nonnegative threshold."
            )
        self._frozen = other._frozen = True
        key = (other._identity, float(limit))
        query = self._queries.get(key)
        if query is None:
            bodies, ignored = [], []
            for prefix, source in (("a", self), ("b", other)):
                ids = []
                for name, (mesh, _) in source._objects.items():
                    identity = prefix + ":" + name
                    ids.append(identity)
                    bodies.append(
                        CollisionBody(
                            identity,
                            identity,
                            "convex",
                            vertices=tuple(map(tuple, mesh.vertices)),
                            triangles=tuple(map(tuple, mesh.faces)),
                        )
                    )
                ignored.extend((a, b) for i, a in enumerate(ids) for b in ids[i + 1 :])
            query = NewtonCollisionChecker(
                bodies,
                ignored_pairs=ignored,
                detection_gap_m=max(0.03, limit + 0.01),
                clearance_m=0.0,
                device="cpu",
                use_cuda_graph=False,
            )
            self._queries[key] = query
        poses = {
            prefix + ":" + name: pose
            for prefix, source in (("a", self), ("b", other))
            for name, (_, pose) in source._objects.items()
        }
        query.check(poses, all_pairs=True)
        if query.last_clearance_interval is None:
            raise UnsupportedFeasibilityError(
                "Newton did not publish a clearance interval."
            )
        return _ClearanceInterval(
            *(float(value) for value in query.last_clearance_interval)
        )

    def close(self) -> None:
        for query in self._queries.values():
            query.close()
        self._queries.clear()
        self._objects.clear()
        self._closed = True

    def __del__(self):
        self.close()
