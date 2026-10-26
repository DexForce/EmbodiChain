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
"""Explicit inset-region pour staging proposals, never physical evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points
import trimesh

from .placement_proposals import _projection
from .pour_geometry import _pose, _top, _vertices, staging_pose

__all__ = ["PourRegionGeometry", "region_staging_pose"]


def _position_budget(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(
            "Pour region requires an explicit positive finite position budget."
        )
    return float(value)


def _faces(value: np.ndarray, vertex_count: int) -> np.ndarray:
    faces = np.asarray(value)
    if (
        faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or (faces < 0).any()
        or (faces >= vertex_count).any()
    ):
        raise ValueError(
            "Pour receiver faces require nonempty integer triangles matching vertices."
        )
    return faces


def region_staging_pose(
    source_vertices: np.ndarray,
    receiver_vertices: np.ndarray,
    receiver_faces: np.ndarray,
    observed_source_pose: np.ndarray,
    observed_receiver_pose: np.ndarray,
    arm_origin_world: np.ndarray,
    *,
    clearance_m: float,
    gravity_world: np.ndarray,
    proxy_position_tolerance_m: float,
) -> np.ndarray:
    """Move only staging XY to the arm-nearest point of the inset projection.

    The declared position budget is a planning reserve, not a contact threshold.
    Complete receiver triangles preserve holes and disconnected components; an
    empty or invalid inset has no fallback. The reserve constructs a proposal,
    not an exact encoded/native r-ball certificate. The shared staging proposal
    still owns entry R, height and source-envelope clearance.
    """
    budget = _position_budget(proxy_position_tolerance_m)
    target = staging_pose(
        source_vertices,
        receiver_vertices,
        observed_source_pose,
        observed_receiver_pose,
        clearance_m=clearance_m,
        gravity_world=gravity_world,
    )
    receiving = _vertices(receiver_vertices)
    faces = _faces(receiver_faces, len(receiving))
    receiver = _pose(observed_receiver_pose)
    origin = np.asarray(arm_origin_world)
    if (
        origin.shape != (3,)
        or np.iscomplexobj(origin)
        or origin.dtype == np.bool_
        or not np.isfinite(origin).all()
    ):
        raise ValueError("Pour region requires a finite active-arm world origin.")
    world = receiving @ receiver[:3, :3].T + receiver[:3, 3]
    region = _projection(trimesh.Trimesh(vertices=world, faces=faces, process=False))
    if region.is_empty or not region.is_valid:
        raise ValueError("Pour receiver projection is empty or invalid.")
    inset = region.buffer(-budget)
    if inset.is_empty or not inset.is_valid:
        raise ValueError("Pour receiver inset is empty or invalid.")
    point, _ = nearest_points(inset, Point(origin[:2]))
    xy = np.array([point.x, point.y])
    # nearest_points constructs the inset proposal. Exact boundary predicates can
    # disagree at that point or after floating recomposition; retain outer bounds.
    if not np.isfinite(xy).all() or not region.covers(point):
        raise ValueError(
            "Pour region nearest point has no outer projection membership."
        )
    _, pivot = _top(_vertices(source_vertices), target[:3, :3])
    anchor = target[:3, :3] @ pivot + target[:3, 3]
    target[:2, 3] += xy - anchor[:2]
    actual = target[:3, :3] @ pivot + target[:3, 3]
    if not np.isfinite(actual).all() or not region.covers(Point(actual[:2])):
        raise ValueError("Pour emitted anchor has no outer projection membership.")
    return target


def _immutable(value: np.ndarray) -> np.ndarray:
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, slots=True)
class PourRegionGeometry:
    """Invocation-service geometry snapshot; poses remain fresh call inputs."""

    vertices: Mapping[str, np.ndarray]
    faces: Mapping[str, np.ndarray]
    gravity_world: tuple[float, float, float]
    proxy_position_tolerance_m: float

    def __post_init__(self) -> None:
        budget = _position_budget(self.proxy_position_tolerance_m)
        vertices = {
            key: _immutable(_vertices(value)) for key, value in self.vertices.items()
        }
        faces = {
            key: _immutable(_faces(value, len(vertices[key])))
            for key, value in self.faces.items()
        }
        gravity = np.asarray(self.gravity_world)
        if (
            gravity.shape != (3,)
            or np.iscomplexobj(gravity)
            or not np.isfinite(gravity).all()
        ):
            raise ValueError("Pour region requires finite configured gravity.")
        object.__setattr__(self, "vertices", MappingProxyType(vertices))
        object.__setattr__(self, "faces", MappingProxyType(faces))
        object.__setattr__(self, "gravity_world", tuple(float(v) for v in gravity))
        object.__setattr__(self, "proxy_position_tolerance_m", budget)

    def staging_pose(
        self,
        object_id: str,
        receiver_id: str,
        source_pose: np.ndarray,
        receiver_pose: np.ndarray,
        arm_origin_world: np.ndarray,
        *,
        clearance_m: float,
    ) -> np.ndarray:
        """Apply the same pure rule to current participants and the bound arm."""
        return region_staging_pose(
            self.vertices[object_id],
            self.vertices[receiver_id],
            self.faces[receiver_id],
            source_pose,
            receiver_pose,
            arm_origin_world,
            clearance_m=clearance_m,
            gravity_world=self.gravity_world,
            proxy_position_tolerance_m=self.proxy_position_tolerance_m,
        )
