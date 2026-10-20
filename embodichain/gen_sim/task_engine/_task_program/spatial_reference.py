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

"""Qualify fixed-body and prismatic-part spatial reference frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from ..scene.articulation_geometry import _root_pose, read_articulation_geometry
from .articulation_binding import discover_prismatic_parts, inspect_prismatic_part

__all__: list[str] = []

SIDE_RELATIONS = frozenset(
    {
        "left_of",
        "right_of",
        "front_of",
        "behind",
        "front_left_of",
        "front_right_of",
        "back_left_of",
        "back_right_of",
    }
)


@dataclass(frozen=True)
class SpatialReference:
    object_id: str
    part_id: str | None
    link: str
    vertices: np.ndarray
    initial_pose: np.ndarray

    @property
    def link_id(self) -> str:
        return f"{self.object_id}__{self.link}"

    def payload(self) -> dict[str, str]:
        return {
            "entity_id": self.link_id,
            "articulation_id": self.object_id,
            "native_link_name": self.link,
        }

    def planner_object(self) -> dict[str, Any]:
        return {
            "runtime_uid": self.link_id,
            "role": "articulation",
            "init_pos": self.initial_pose[:3, 3].tolist(),
            "init_local_pose": self.initial_pose.tolist(),
            "_spatial_vertices": self.vertices,
        }


def resolve_reference(
    config: Mapping[str, Any], part_id: str | None
) -> SpatialReference:
    """Use only meshes owned by the selected native link, never the whole USD."""
    geometry = read_articulation_geometry(config["fpath"])
    binding = None if part_id is None else inspect_prismatic_part(dict(config), part_id)
    # A grasp can live on a fixed child handle; spatial identity is the drawer
    # body selected by the part catalog, not that grasp frame.
    link = (
        geometry.root_link
        if binding is None
        else next(
            p.link
            for p in discover_prismatic_parts(dict(config))
            if p.part_id == part_id
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
            raise ValueError("Table-side references require horizontal drawer motion.")
    return SpatialReference(str(config["uid"]), part_id, link, vertices, initial_pose)


def graph_references(
    graph: Mapping[str, Any], scene: Any
) -> dict[tuple[str, str | None], SpatialReference]:
    configs = {str(c["uid"]): c for c in getattr(scene, "articulations", ())}
    references = {}
    for node in graph["nodes"]:
        call = node["call"]
        args = call.get("arguments", {})
        uid = args.get("reference")
        if call.get("call_id") != "simulation.place_relative" or uid not in configs:
            continue
        if args.get("relation") not in SIDE_RELATIONS:
            continue
        key = (str(uid), args.get("reference_part"))
        if key not in references:
            references[key] = resolve_reference(configs[str(uid)], key[1])
    return references


def measure_relation(
    object_vertices: torch.Tensor,
    object_pose: torch.Tensor,
    reference_vertices: np.ndarray,
    reference_pose: torch.Tensor,
    relation: str,
) -> tuple[list[bool], dict[str, Any]]:
    """Measure full envelopes in the same world axes used to select the landing."""
    from .placement_evidence import _valid_pose

    _valid_pose(object_pose, object_pose.shape[0])
    _valid_pose(reference_pose, object_pose.shape[0])
    if relation not in SIDE_RELATIONS:
        raise ValueError("A spatial-link observer requires a side relation.")
    world = (
        object_vertices.to(object_pose) @ object_pose[:, :3, :3].transpose(-1, -2)
        + object_pose[:, None, :3, 3]
    )
    reference = (
        torch.as_tensor(
            reference_vertices, dtype=reference_pose.dtype, device=reference_pose.device
        )
        @ reference_pose[:, :3, :3].transpose(-1, -2)
        + reference_pose[:, None, :3, 3]
    )
    axes = {
        "left_of": ((1, -1),),
        "right_of": ((1, 1),),
        "front_of": ((0, -1),),
        "behind": ((0, 1),),
        "front_left_of": ((0, -1), (1, -1)),
        "front_right_of": ((0, -1), (1, 1)),
        "back_left_of": ((0, 1), (1, -1)),
        "back_right_of": ((0, 1), (1, 1)),
    }[relation]
    gaps = [
        (world[:, :, axis] * sign).amin(-1) - (reference[:, :, axis] * sign).amax(-1)
        for axis, sign in axes
    ]
    accepted = torch.stack([torch.isfinite(gap) & (gap >= 0.0) for gap in gaps]).all(0)
    return accepted.tolist(), {
        "scope": "bound_link_world_axes",
        "object_pose": object_pose.tolist(),
        "reference_pose": reference_pose.tolist(),
        "envelope_gaps_m": [g.tolist() for g in gaps],
    }
