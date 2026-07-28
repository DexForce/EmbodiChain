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

"""Resolve coordinated carrier release height and transport geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    RelativePlacementSpec,
    RelativePlacementStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _clean_vector3,
    _iter_generated_scene_object_configs,
    _mesh_config_world_xy_bounds,
    _mesh_config_world_xy_extents,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_intent import (
    _SIDE_RELATIONS,
    _relative_primary_placement,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_offsets import (
    _POSE_SENSITIVE_STAGING_Z_DELTA,
    _STAGING_Z_DELTA,
    _offset_position,
    _replace_relative_spec_placements,
)

__all__ = [
    "_with_coordinated_side_release_height_offsets",
    "_with_coordinated_transport_geometry",
]

_COORDINATED_TRANSPORT_DISTANCE = 0.15
_COORDINATED_MIN_TRANSPORT_DISTANCE = 0.05
_COORDINATED_GEOMETRY_MARGIN = 0.02


def _with_coordinated_side_release_height_offsets(
    spec: RelativePlacementSpec,
    gym_config: Mapping[str, Any],
    *,
    table_reference_mode: str = "include",
) -> RelativePlacementSpec:
    if spec.intent not in {"place_relative", "coordinated_pickment"}:
        return spec
    placements = tuple(
        (
            _with_coordinated_side_release_height_offset(placement, gym_config)
            if _matches_table_reference_mode(
                placement,
                table_source_uid=spec.table_source_uid,
                table_reference_mode=table_reference_mode,
            )
            else placement
        )
        for placement in spec.placements
    )
    return _replace_relative_spec_placements(spec, placements)


def _matches_table_reference_mode(
    placement: RelativePlacementStepSpec,
    *,
    table_source_uid: str,
    table_reference_mode: str,
) -> bool:
    is_table_reference = placement.reference_source_uid == table_source_uid
    if table_reference_mode == "include":
        return True
    if table_reference_mode == "skip":
        return not is_table_reference
    if table_reference_mode == "only":
        return is_table_reference
    raise ValueError(f"Unsupported table reference mode: {table_reference_mode!r}.")


def _with_coordinated_side_release_height_offset(
    placement: RelativePlacementStepSpec,
    gym_config: Mapping[str, Any],
) -> RelativePlacementStepSpec:
    if placement.relation not in _SIDE_RELATIONS or placement.reference_is_initial_pose:
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

    reference_origin = _clean_vector3(reference_config.get("init_pos", [0, 0, 0]))
    moved_origin = _clean_vector3(moved_config.get("init_pos", [0, 0, 0]))
    release_offset = list(placement.release_offset)
    release_offset[2] = round(float(moved_origin[2] - reference_origin[2]), 6)
    high_offset = list(release_offset)
    staging_z_delta = (
        _POSE_SENSITIVE_STAGING_Z_DELTA
        if placement.orientation_goal != "preserve"
        else _STAGING_Z_DELTA
    )
    high_offset[2] = round(release_offset[2] + staging_z_delta, 6)
    return replace(
        placement,
        release_offset=release_offset,
        high_offset=high_offset,
    )


def _with_coordinated_transport_geometry(
    spec: RelativePlacementSpec,
    gym_config: Mapping[str, Any],
) -> RelativePlacementSpec:
    """Resolve loaded-carrier capacity, slots, and the final transport target."""
    if (
        spec.intent != "coordinated_pickment"
        or spec.coordinated_direction is None
        or spec.coordinated_terminal_behavior is None
    ):
        return spec

    object_configs = {
        str(obj.get("uid")): obj
        for obj in _iter_generated_scene_object_configs(gym_config)
        if obj.get("uid") is not None
    }
    carrier = _relative_primary_placement(spec.placements)
    carrier_config = object_configs.get(carrier.moved_runtime_uid)
    if carrier_config is None:
        raise ValueError(
            f"Generated config is missing coordinated carrier {carrier.moved_runtime_uid!r}."
        )
    initial_position = _clean_vector3(carrier_config.get("init_pos", [0.0, 0.0, 0.0]))
    direction = _coordinated_direction_vector(spec.coordinated_direction)
    distance = _coordinated_safe_transport_distance(
        initial_position=initial_position,
        direction=direction,
        carrier_config=carrier_config,
        table_config=_coordinated_table_config(spec, gym_config),
    )
    release_offset = [
        round(direction[0] * distance, 6),
        round(direction[1] * distance, 6),
        0.0,
    ]
    release_position = _offset_position(initial_position, release_offset)
    high_position = list(release_position)
    high_position[2] = round(high_position[2] + float(carrier.hover_height), 6)
    updated_carrier = replace(
        carrier,
        release_offset=release_offset,
        high_offset=[release_offset[0], release_offset[1], carrier.hover_height],
        release_position=release_position,
        high_position=high_position,
    )
    placements = tuple(
        updated_carrier if placement.intent == "coordinated_pickment" else placement
        for placement in spec.placements
    )
    return _replace_relative_spec_placements(spec, placements)


def _coordinated_table_config(
    spec: RelativePlacementSpec,
    gym_config: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    backgrounds = gym_config.get("background", [])
    if isinstance(backgrounds, Mapping):
        backgrounds = [backgrounds]
    if not isinstance(backgrounds, list):
        return None
    return next(
        (
            background
            for background in backgrounds
            if isinstance(background, Mapping)
            and str(background.get("uid")) == spec.table_source_uid
        ),
        next(
            (
                background
                for background in backgrounds
                if isinstance(background, Mapping)
            ),
            None,
        ),
    )


def _coordinated_direction_vector(direction: str) -> tuple[float, float]:
    components = {
        "front": (1.0, 0.0),
        "back": (-1.0, 0.0),
        "left": (0.0, 1.0),
        "right": (0.0, -1.0),
        "front_left": (1.0, 1.0),
        "front_right": (1.0, -1.0),
        "back_left": (-1.0, 1.0),
        "back_right": (-1.0, -1.0),
        "none": (0.0, 0.0),
    }
    x, y = components[direction]
    norm = math.hypot(x, y)
    if norm == 0.0:
        return 0.0, 0.0
    return x / norm, y / norm


def _coordinated_safe_transport_distance(
    *,
    initial_position: Sequence[float],
    direction: tuple[float, float],
    carrier_config: Mapping[str, Any],
    table_config: Mapping[str, Any] | None,
) -> float:
    if direction == (0.0, 0.0):
        return 0.0
    table_bounds = (
        _mesh_config_world_xy_bounds(table_config) if table_config is not None else None
    )
    carrier_extents = _mesh_config_world_xy_extents(carrier_config)
    if table_bounds is None or carrier_extents is None:
        return _COORDINATED_TRANSPORT_DISTANCE

    mins, maxs = table_bounds
    half_extents = [float(carrier_extents[0]) * 0.5, float(carrier_extents[1]) * 0.5]
    allowed = _COORDINATED_TRANSPORT_DISTANCE
    for axis, component in enumerate(direction):
        if abs(component) <= 1e-9:
            continue
        if component > 0.0:
            boundary = (
                float(maxs[axis]) - half_extents[axis] - _COORDINATED_GEOMETRY_MARGIN
            )
            axis_limit = (boundary - float(initial_position[axis])) / component
        else:
            boundary = (
                float(mins[axis]) + half_extents[axis] + _COORDINATED_GEOMETRY_MARGIN
            )
            axis_limit = (boundary - float(initial_position[axis])) / component
        allowed = min(allowed, axis_limit)
    allowed = max(0.0, float(allowed))
    if allowed < _COORDINATED_MIN_TRANSPORT_DISTANCE:
        raise ValueError(
            "Coordinated transport target cannot keep the carrier inside the table "
            f"boundary with at least {_COORDINATED_MIN_TRANSPORT_DISTANCE:.2f} m movement."
        )
    return round(allowed, 6)
