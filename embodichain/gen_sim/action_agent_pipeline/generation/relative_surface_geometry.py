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

"""Resolve surface contact and orientation-aware local mesh height."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.config.defaults import defaults_section
from embodichain.gen_sim.action_agent_pipeline.domain.orientation_policy import (
    _is_normalized_local_z_label,
    principal_local_axis_order,
    resolve_target_rotation,
    rotated_local_z_min,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    RelativePlacementSpec,
    RelativePlacementStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _clean_vector3,
    _iter_generated_scene_object_configs,
    _load_mesh_vertices,
    _mesh_config_local_zmin_after_rotation,
    _mesh_config_transform_matrix,
    _mesh_config_world_zmax,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_offsets import (
    _POSE_SENSITIVE_STAGING_Z_DELTA,
    _STAGING_Z_DELTA,
    _offset_position,
    _replace_relative_spec_placements,
)

__all__ = [
    "_target_local_zmin_for_orientation",
    "_with_on_surface_release_offsets",
]

_DEFAULTS = defaults_section("relative_placement")
_PICKUP_UPRIGHT_ROTATE_RADIANS = math.radians(
    float(_DEFAULTS["pickup_upright_rotate_degrees"])
)


def _with_on_surface_release_offsets(
    spec: RelativePlacementSpec,
    gym_config: Mapping[str, Any],
) -> RelativePlacementSpec:
    placements = tuple(
        _with_on_surface_release_offset(placement, gym_config)
        for placement in spec.placements
    )
    return _replace_relative_spec_placements(spec, placements)


def _with_on_surface_release_offset(
    placement: RelativePlacementStepSpec,
    gym_config: Mapping[str, Any],
) -> RelativePlacementStepSpec:
    if placement.relation != "on" or placement.reference_is_initial_pose:
        return placement

    object_configs = {
        str(obj.get("uid")): obj
        for obj in _iter_generated_scene_object_configs(gym_config)
        if obj.get("uid") is not None
    }
    reference_config = object_configs.get(placement.reference_runtime_uid)
    moved_config = object_configs.get(placement.moved_runtime_uid)
    if reference_config is None or moved_config is None:
        return placement

    reference_top_z = _mesh_config_world_zmax(reference_config)
    moved_bottom_offset = _target_local_zmin_for_orientation(
        moved_config,
        placement.orientation_goal,
    )
    if reference_top_z is None or moved_bottom_offset is None:
        return placement

    reference_origin = _clean_vector3(reference_config.get("init_pos", [0, 0, 0]))
    moved_origin = _clean_vector3(moved_config.get("init_pos", [0, 0, 0]))
    release_offset = list(placement.release_offset)
    if placement.upright_in_place:
        release_offset[0] = round(float(moved_origin[0] - reference_origin[0]), 6)
        release_offset[1] = round(float(moved_origin[1] - reference_origin[1]), 6)
    release_offset[2] = round(
        float(reference_top_z)
        - float(reference_origin[2])
        + float(placement.surface_clearance)
        - float(moved_bottom_offset),
        6,
    )
    high_offset = list(release_offset)
    high_offset[2] = round(
        release_offset[2]
        + (
            _POSE_SENSITIVE_STAGING_Z_DELTA
            if placement.orientation_goal != "preserve"
            else _STAGING_Z_DELTA
        ),
        6,
    )
    update_kwargs: dict[str, Any] = {
        "release_offset": release_offset,
        "high_offset": high_offset,
    }
    if placement.upright_in_place:
        release_position = _offset_position(reference_origin, release_offset)
        high_position = _offset_position(reference_origin, high_offset)
        update_kwargs["release_position"] = release_position
        update_kwargs["high_position"] = high_position
        pickup_upright_direction = _pickup_upright_direction(moved_config)
        if pickup_upright_direction is not None:
            update_kwargs["pickup_upright_direction"] = pickup_upright_direction
            update_kwargs["pickup_rotate_upright"] = _PICKUP_UPRIGHT_ROTATE_RADIANS
    return replace(placement, **update_kwargs)


def _pickup_upright_direction(obj_config: Mapping[str, Any]) -> list[float] | None:
    object_label = str(obj_config.get("uid", ""))
    if _is_normalized_local_z_label(object_label):
        return [0.0, 0.0, 1.0]
    vertices = _mesh_config_scaled_vertices(obj_config)
    if not vertices:
        return None
    axis_index = principal_local_axis_order(_local_vertex_bounds(vertices))[0]
    return [1.0 if index == axis_index else 0.0 for index in range(3)]


def _target_local_zmin_for_orientation(
    obj_config: Mapping[str, Any],
    orientation_goal: str,
) -> float | None:
    if orientation_goal in {"preserve", "axis_align"}:
        return _mesh_config_local_zmin_after_rotation(obj_config)
    if orientation_goal == "upright":
        return _upright_local_zmin(obj_config)
    if orientation_goal == "lay_flat":
        return _lay_flat_local_zmin(obj_config)
    return _mesh_config_local_zmin_after_rotation(obj_config)


def _upright_local_zmin(obj_config: Mapping[str, Any]) -> float | None:
    return _oriented_local_zmin(obj_config, orientation_goal="upright")


def _mesh_config_scaled_vertices(
    obj_config: Mapping[str, Any],
) -> list[tuple[float, float, float]] | None:
    shape = obj_config.get("shape", {})
    if not isinstance(shape, Mapping):
        return None
    mesh_path = shape.get("fpath")
    if not isinstance(mesh_path, str):
        return None
    vertices = _load_mesh_vertices(Path(mesh_path).expanduser().resolve())
    if not vertices:
        return None
    scale = _clean_vector3(obj_config.get("body_scale", [1.0, 1.0, 1.0]))
    return [
        (
            float(vertex[0]) * float(scale[0]),
            float(vertex[1]) * float(scale[1]),
            float(vertex[2]) * float(scale[2]),
        )
        for vertex in vertices
    ]


def _mesh_config_rotation_basis(
    obj_config: Mapping[str, Any],
) -> list[list[float]]:
    matrix = _mesh_config_transform_matrix(obj_config, translation=[0.0, 0.0, 0.0])
    columns = []
    for index in range(3):
        column = [float(matrix[row][index]) for row in range(3)]
        norm = math.sqrt(sum(value * value for value in column))
        if norm < 1e-6:
            raise ValueError("Mesh config rotation contains a near-zero basis column.")
        columns.append([value / norm for value in column])
    return _columns_to_matrix(columns)


def _columns_to_matrix(columns: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(columns[col][row]) for col in range(3)] for row in range(3)]


def _lay_flat_local_zmin(obj_config: Mapping[str, Any]) -> float | None:
    return _oriented_local_zmin(obj_config, orientation_goal="lay_flat")


def _oriented_local_zmin(
    obj_config: Mapping[str, Any],
    *,
    orientation_goal: str,
) -> float | None:
    vertices = _mesh_config_scaled_vertices(obj_config)
    if not vertices:
        return None
    rotation = resolve_target_rotation(
        orientation_goal=orientation_goal,
        local_bounds=_local_vertex_bounds(vertices),
        current_rotation=_mesh_config_rotation_basis(obj_config),
        object_label=str(obj_config.get("uid", "")),
    )
    # Rotating every scaled vertex preserves off-center and bottom-origin meshes.
    return rotated_local_z_min(vertices, rotation)


def _local_vertex_bounds(
    vertices: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        tuple(min(float(vertex[index]) for vertex in vertices) for index in range(3)),
        tuple(max(float(vertex[index]) for vertex in vertices) for index in range(3)),
    )
