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
    NativeContactResult,
    NewtonCollisionChecker,
    UnsupportedFeasibilityError,
    _DEFAULT_DETECTION_GAP_M,
    _DEFAULT_MAX_SHAPES,
    _DEFAULT_NUMERIC_TOLERANCE_M,
    _geometry_precision,
    _pack_query_poses,
    _aabb_corners,
    _aabb_lower_bound,
)
from ._collision_backends import _require_newton
from ._validation import _rigid_pose

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
    """Keep explicitly qualified convex query solids separate from simulation.

    This capability accepts no triangle surfaces or native MESH substitutes.
    """

    def __init__(self) -> None:
        self._objects = {}
        self._queries = {}
        self._bounds = {}
        self._precision = {}
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
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Collision geometry must be an explicit Trimesh.")
        if (
            not np.isfinite(mesh.vertices).all()
            or not len(mesh.vertices)
            or not len(mesh.faces)
        ):
            raise UnsupportedFeasibilityError(
                "Collision mesh must have finite vertices and explicit faces."
            )
        if not mesh.is_convex or not mesh.is_volume:
            raise UnsupportedFeasibilityError(
                "Native query requires qualified convex solids; triangle surfaces are not convexified."
            )
        precision = _geometry_precision("convex", mesh.vertices)
        self._objects[name] = (
            mesh.copy(),
            np.eye(4) if transform is None else np.array(transform, copy=True),
        )
        self._precision[name] = precision
        low, high = mesh.bounds
        self._bounds[name] = _aabb_corners(low, high)

    def set_transform(self, name: str, transform: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Collision query is closed.")
        mesh, _ = self._objects[name]
        self._objects[name] = mesh, np.array(transform, copy=True)

    def distance_interval(
        self, other: _ConvexCollection, limit: float
    ) -> _ClearanceInterval:
        query, poses = self._query(other, limit)
        query.check(poses, all_pairs=True)
        if query.last_clearance_interval is None:
            raise UnsupportedFeasibilityError(
                "Newton did not publish a clearance interval."
            )
        return _ClearanceInterval(
            *(float(value) for value in query.last_clearance_interval)
        )

    def contact_query(
        self, other: _ConvexCollection, limit: float
    ) -> NativeContactResult:
        """Return owned native surface facts, preserving undecidable thresholds."""
        query, poses = self._query(other, limit)
        return query.contact_query(poses, all_pairs=True)

    def clearance_lower_bound(self, other: _ConvexCollection) -> float | None:
        """Bound separation before allocating/querying native collision models.

        Args:
            other: Compatible group whose frozen geometry is being compared.

        Returns:
            Nonnegative AABB separation lower bound after the checker's canonical
            geometry/pose conversion allowance, or None when the query precision
            cannot be bounded. Only a value strictly above the original contact
            threshold permits culling. Zero never decides touching or overlap.
        """
        if self._closed or other._closed:
            raise RuntimeError("Collision query is closed.")
        if not self._objects or not other._objects:
            raise ValueError("A clearance bound needs nonempty query groups.")
        self._frozen = other._frozen = True
        groups = (self, other)
        entries = [
            (source, name, pose)
            for source in groups
            for name, (_, pose) in source._objects.items()
        ]
        if len(entries) > _DEFAULT_MAX_SHAPES:
            return None
        poses = np.asarray([_rigid_pose(pose) for _, _, pose in entries])
        radius = max(source._precision[name][0] for source, name, _ in entries)
        error = max(source._precision[name][1] for source, name, _ in entries)
        try:
            _, tolerance, relative, _ = _pack_query_poses(
                poses, radius, error, _DEFAULT_NUMERIC_TOLERANCE_M
            )
        except UnsupportedFeasibilityError:
            return None
        # Match the smallest collection query band; larger consumer limits may
        # enlarge that band but must not weaken this conservative culling gate.
        if not math.isfinite(tolerance) or tolerance >= _DEFAULT_DETECTION_GAP_M:
            return None
        boxes = {id(source): [] for source in groups}
        for index, (source, name, _) in enumerate(entries):
            corners = source._bounds[name] @ poses[index, :3, :3].T + relative[index]
            boxes[id(source)].append((corners.min(axis=0), corners.max(axis=0)))
        a, b = [
            (
                np.min([v[0] for v in boxes[id(source)]], axis=0),
                np.max([v[1] for v in boxes[id(source)]], axis=0),
            )
            for source in groups
        ]
        bound = _aabb_lower_bound(a, b, tolerance)
        return bound if math.isfinite(bound) else None

    def _query(
        self, other: _ConvexCollection, limit: float
    ) -> tuple[NewtonCollisionChecker, dict[str, np.ndarray]]:
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
                detection_gap_m=max(_DEFAULT_DETECTION_GAP_M, limit + 0.01),
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
        return query, poses

    def close(self) -> None:
        for query in self._queries.values():
            query.close()
        self._queries.clear()
        self._objects.clear()
        self._bounds.clear()
        self._precision.clear()
        self._closed = True

    def __del__(self):
        self.close()
