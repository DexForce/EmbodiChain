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

"""Exact selected-link source geometry using the shared articulation catalog."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
from .articulation_geometry import read_articulation_geometry, _root_pose
from .articulation_inputs import discover_prismatic_parts, inspect_prismatic_part

__all__ = ["SpatialReferenceGeometry", "read_spatial_reference"]


@dataclass(frozen=True, slots=True)
class SpatialReferenceGeometry:
    """Owned scaled local vertices and initial pose of one qualified native link."""

    link: str
    vertices: np.ndarray
    initial_pose: np.ndarray

    def __post_init__(self) -> None:
        if (
            type(self.link) is not str
            or not self.link
            or self.link != self.link.strip()
        ):
            raise ValueError("Spatial reference requires an exact native link name.")
        if (
            np.asarray(self.vertices).ndim != 2
            or np.asarray(self.vertices).shape[-1] != 3
            or not len(self.vertices)
            or not np.isfinite(self.vertices).all()
            or np.asarray(self.initial_pose).shape != (4, 4)
            or not np.isfinite(self.initial_pose).all()
        ):
            raise ValueError(
                "Spatial reference requires finite local vertices and a complete initial pose."
            )
        for name in ("vertices", "initial_pose"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def read_spatial_reference(
    config: Mapping[str, Any], part_id: str | None
) -> SpatialReferenceGeometry:
    """Read the exact fixed body or horizontally moving prismatic-part link.

    Args:
        config: Existing native articulation source file, scale and root pose inputs.
        part_id: Explicit catalog part, or None for the unique source root body.

    Returns:
        Original selected-link collision vertices and root-composed pose. A grasp
        on a fixed child does not replace the selected drawer body. Unsupported
        units, geometry or nonhorizontal prismatic motion remain unavailable.
    """
    geometry = read_articulation_geometry(config["fpath"])
    binding = None if part_id is None else inspect_prismatic_part(dict(config), part_id)
    link = (
        geometry.root_link
        if binding is None
        else next(
            part.link
            for part in discover_prismatic_parts(dict(config))
            if part.part_id == part_id
        )
    )
    points = geometry.link_vertices.get(link)
    if points is None or not len(points):
        raise ValueError(f"Spatial reference link {link!r} has no collision geometry.")
    scale = np.asarray(config.get("body_scale", [1.0] * 3), dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Spatial reference scale must be finite and positive.")
    local_pose = geometry.link_poses[link].copy()
    vertices = (points - local_pose[:3, 3]) @ local_pose[:3, :3] * scale
    local_pose[:3, 3] *= scale
    initial_pose = _root_pose(config) @ local_pose
    if binding is not None:
        axis = (
            _root_pose(config)[:3, :3]
            @ geometry.link_poses[binding.link][:3, :3]
            @ np.asarray(binding.axis)
        )
        if abs(float(axis[2])) > 0.02:
            raise ValueError("Spatial references require horizontal prismatic motion.")
    return SpatialReferenceGeometry(link, vertices, initial_pose)
