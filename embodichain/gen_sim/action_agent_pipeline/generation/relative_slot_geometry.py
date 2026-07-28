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

"""Plan deterministic slots for payloads placed in a shared container."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.config.defaults import defaults_section
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    RelativePlacementSpec,
    RelativePlacementStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _clean_vector3,
    _iter_generated_scene_object_configs,
    _mesh_config_world_xy_extents,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_offsets import (
    _replace_relative_spec_placements,
    _with_relative_release_offset,
)

__all__ = [
    "_inside_container_axis_offsets",
    "_inside_container_slot_axis_and_distance",
    "_with_inside_container_slot_offsets",
]

_DEFAULTS = defaults_section("relative_placement")
_CONTAINER_SLOT_MIN_OFFSET = float(_DEFAULTS["container_slot_min_offset"])
_CONTAINER_SLOT_MAX_OFFSET = float(_DEFAULTS["container_slot_max_offset"])
_CONTAINER_SLOT_FRACTION = float(_DEFAULTS["container_slot_fraction"])
_CONTAINER_SLOT_MAX_FRACTION = float(_DEFAULTS["container_slot_max_fraction"])
_CONTAINER_SLOT_AXIS_TIE_RATIO = float(_DEFAULTS["container_slot_axis_tie_ratio"])
_SIDE_RELEASE_Z_OFFSET = float(_DEFAULTS["side_release_z_offset"])
_DEFAULT_Y_AXIS_ARM_SLOT_SIDE_ORDER = {"right": 0, "left": 1}
_COORDINATED_GEOMETRY_MARGIN = 0.02


def _with_inside_container_slot_offsets(
    spec: RelativePlacementSpec,
    gym_config: Mapping[str, Any],
    *,
    slot_distance_scale: float = 1.0,
) -> RelativePlacementSpec:
    slotted_relations = {"inside"}
    if spec.intent == "coordinated_pickment":
        slotted_relations.add("on")
    inside_groups: dict[str, list[int]] = {}
    for index, placement in enumerate(spec.placements):
        if (
            placement.relation not in slotted_relations
            or placement.reference_is_initial_pose
        ):
            continue
        inside_groups.setdefault(placement.reference_runtime_uid, []).append(index)

    inside_groups = {
        reference_uid: indices
        for reference_uid, indices in inside_groups.items()
        if len(indices) > 1
    }
    if not inside_groups:
        return spec

    object_configs = {
        str(obj.get("uid")): obj
        for obj in _iter_generated_scene_object_configs(gym_config)
        if obj.get("uid") is not None
    }
    slot_offsets_by_index: dict[int, list[float]] = {}
    for reference_uid, indices in inside_groups.items():
        container_config = object_configs.get(reference_uid)
        if spec.intent == "coordinated_pickment" and len(indices) >= 3:
            slot_offsets_by_index.update(
                _coordinated_payload_grid_offsets(
                    indices,
                    placements=spec.placements,
                    object_configs=object_configs,
                    container_config=container_config,
                )
            )
            continue
        axis, slot_distance = _inside_container_slot_axis_and_distance(
            container_config,
            slot_distance_scale=slot_distance_scale,
        )
        slot_distance = _payload_aware_slot_distance(
            indices,
            axis=axis,
            default_distance=slot_distance,
            placements=spec.placements,
            object_configs=object_configs,
        )
        ordered_indices = _order_inside_container_slot_indices(
            indices,
            placements=spec.placements,
            axis=axis,
            object_configs=object_configs,
            container_config=container_config,
        )
        for index, axis_offset in zip(
            ordered_indices,
            _inside_container_axis_offsets(len(ordered_indices), slot_distance),
        ):
            release_offset = [0.0, 0.0, _SIDE_RELEASE_Z_OFFSET]
            release_offset[0 if axis == "x" else 1] = axis_offset
            slot_offsets_by_index[index] = [
                round(float(value), 6) for value in release_offset
            ]

    if not slot_offsets_by_index:
        return spec

    placements = tuple(
        (
            _with_relative_release_offset(placement, slot_offsets_by_index[index])
            if index in slot_offsets_by_index
            else placement
        )
        for index, placement in enumerate(spec.placements)
    )
    return _replace_relative_spec_placements(spec, placements)


def _coordinated_payload_grid_offsets(
    indices: Sequence[int],
    *,
    placements: Sequence[RelativePlacementStepSpec],
    object_configs: Mapping[str, Mapping[str, Any]],
    container_config: Mapping[str, Any] | None,
) -> dict[int, list[float]]:
    """Choose a deterministic best-effort grid for three or four payloads."""
    count = len(indices)
    carrier_extents = (
        _mesh_config_world_xy_extents(container_config)
        if container_config is not None
        else None
    )
    if carrier_extents is None:
        carrier_extents = (
            2.0 * _CONTAINER_SLOT_MAX_OFFSET,
            2.0 * _CONTAINER_SLOT_MAX_OFFSET,
        )
    usable_x = max(0.02, float(carrier_extents[0]) - 2.0 * _COORDINATED_GEOMETRY_MARGIN)
    usable_y = max(0.02, float(carrier_extents[1]) - 2.0 * _COORDINATED_GEOMETRY_MARGIN)
    ordered_indices = sorted(
        indices,
        key=lambda index: (
            _relative_initial_axis_value(
                placements[index],
                axis_index=1,
                object_configs=object_configs,
                container_config=container_config,
            ),
            _relative_initial_axis_value(
                placements[index],
                axis_index=0,
                object_configs=object_configs,
                container_config=container_config,
            ),
            index,
        ),
    )
    payload_extents = {
        index: _mesh_config_world_xy_extents(
            object_configs.get(placements[index].moved_runtime_uid, {})
        )
        or (0.0, 0.0)
        for index in ordered_indices
    }
    grid_shapes = [(count, 1), (1, count), (2, 2)]
    candidates: list[tuple[tuple[float, float], dict[int, list[float]]]] = []
    for columns, rows in grid_shapes:
        cell_x = usable_x / float(columns)
        cell_y = usable_y / float(rows)
        cells = [
            (
                -usable_x * 0.5 + (column + 0.5) * cell_x,
                -usable_y * 0.5 + (row + 0.5) * cell_y,
            )
            for row in range(rows)
            for column in range(columns)
        ][:count]
        overflow = 0.0
        min_clearance = float("inf")
        offsets: dict[int, list[float]] = {}
        for index, (x, y) in zip(ordered_indices, cells):
            payload_x, payload_y = payload_extents[index]
            clearance_x = cell_x - float(payload_x)
            clearance_y = cell_y - float(payload_y)
            overflow += max(0.0, _COORDINATED_GEOMETRY_MARGIN - clearance_x)
            overflow += max(0.0, _COORDINATED_GEOMETRY_MARGIN - clearance_y)
            min_clearance = min(min_clearance, clearance_x, clearance_y)
            offsets[index] = [
                round(float(x), 6),
                round(float(y), 6),
                _SIDE_RELEASE_Z_OFFSET,
            ]
        candidates.append(((round(overflow, 9), -min_clearance), offsets))
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _payload_aware_slot_distance(
    indices: Sequence[int],
    *,
    axis: str,
    default_distance: float,
    placements: Sequence[RelativePlacementStepSpec],
    object_configs: Mapping[str, Mapping[str, Any]],
) -> float:
    if len(indices) != 2:
        return default_distance
    axis_index = 0 if axis == "x" else 1
    extents = [
        _mesh_config_world_xy_extents(
            object_configs.get(placements[index].moved_runtime_uid, {})
        )
        for index in indices
    ]
    if any(extent is None for extent in extents):
        return default_distance
    resolved = [extent for extent in extents if extent is not None]
    required_separation = (
        float(resolved[0][axis_index]) * 0.5
        + float(resolved[1][axis_index]) * 0.5
        + _COORDINATED_GEOMETRY_MARGIN
    )
    return round(max(float(default_distance), required_separation * 0.5), 6)


def _inside_container_slot_axis_and_distance(
    container_config: Mapping[str, Any] | None,
    *,
    slot_distance_scale: float = 1.0,
) -> tuple[str, float]:
    slot_distance_scale = _validate_slot_distance_scale(slot_distance_scale)
    extents = (
        _mesh_config_world_xy_extents(container_config)
        if container_config is not None
        else None
    )
    if extents is None:
        return "y", _CONTAINER_SLOT_MIN_OFFSET

    x_extent, y_extent = extents
    axis = _inside_container_slot_axis(x_extent, y_extent)
    axis_extent = x_extent if axis == "x" else y_extent
    if axis_extent <= 0.0:
        return "y", _CONTAINER_SLOT_MIN_OFFSET

    slot_distance = min(
        max(axis_extent * _CONTAINER_SLOT_FRACTION, _CONTAINER_SLOT_MIN_OFFSET),
        axis_extent * _CONTAINER_SLOT_MAX_FRACTION,
        _CONTAINER_SLOT_MAX_OFFSET,
    )
    return axis, round(float(slot_distance) * slot_distance_scale, 6)


def _validate_slot_distance_scale(slot_distance_scale: float) -> float:
    try:
        scale = float(slot_distance_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "inside_container_slot_distance_scale must be a positive number."
        ) from exc
    if scale <= 0.0:
        raise ValueError(
            "inside_container_slot_distance_scale must be a positive number."
        )
    return scale


def _inside_container_slot_axis(x_extent: float, y_extent: float) -> str:
    max_extent = max(float(x_extent), float(y_extent))
    if max_extent <= 0.0:
        return "y"
    if abs(float(x_extent) - float(y_extent)) <= (
        max_extent * _CONTAINER_SLOT_AXIS_TIE_RATIO
    ):
        return "y"
    return "x" if float(x_extent) > float(y_extent) else "y"


def _order_inside_container_slot_indices(
    indices: list[int],
    *,
    placements: Sequence[RelativePlacementStepSpec],
    axis: str,
    object_configs: Mapping[str, Mapping[str, Any]],
    container_config: Mapping[str, Any] | None,
    side_order: Mapping[str, int] | None = None,
) -> list[int]:
    if axis == "y":
        resolved_side_order = dict(side_order or _DEFAULT_Y_AXIS_ARM_SLOT_SIDE_ORDER)
        return sorted(
            indices,
            key=lambda index: (
                resolved_side_order.get(placements[index].active_side, 1),
                _relative_initial_axis_value(
                    placements[index],
                    axis_index=1,
                    object_configs=object_configs,
                    container_config=container_config,
                ),
                index,
            ),
        )

    return sorted(
        indices,
        key=lambda index: (
            _relative_initial_axis_value(
                placements[index],
                axis_index=0,
                object_configs=object_configs,
                container_config=container_config,
            ),
            index,
        ),
    )


def _relative_initial_axis_value(
    placement: RelativePlacementStepSpec,
    *,
    axis_index: int,
    object_configs: Mapping[str, Mapping[str, Any]],
    container_config: Mapping[str, Any] | None,
) -> float:
    moved_config = object_configs.get(placement.moved_runtime_uid)
    moved_position = _scene_config_init_position(moved_config)
    container_position = _scene_config_init_position(container_config)
    return float(moved_position[axis_index] - container_position[axis_index])


def _scene_config_init_position(
    obj_config: Mapping[str, Any] | None,
) -> list[float]:
    if obj_config is None:
        return [0.0, 0.0, 0.0]
    return _clean_vector3(obj_config.get("init_pos", [0.0, 0.0, 0.0]))


def _inside_container_axis_offsets(count: int, slot_distance: float) -> list[float]:
    if count <= 1:
        return [0.0]
    if count == 2:
        return [
            round(-float(slot_distance), 6),
            round(float(slot_distance), 6),
        ]
    step = (2.0 * float(slot_distance)) / float(count - 1)
    return [round(-float(slot_distance) + step * index, 6) for index in range(count)]
