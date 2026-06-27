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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _RelativePlacementSpec,
    _RelativePlacementStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _clean_vector3,
    _iter_generated_scene_object_configs,
    _mesh_config_local_zmin_after_rotation,
    _mesh_config_world_zmax,
    _mesh_config_world_xy_extents,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_spec import (
    _SIDE_RELATIONS,
    _normalize_relative_relation,
)

__all__ = [
    "_POSE_SENSITIVE_STAGING_Z_DELTA",
    "_STAGING_Z_DELTA",
    "_inside_container_axis_offsets",
    "_inside_container_slot_axis_and_distance",
    "_make_relative_summary",
    "_offset_position",
    "_relative_release_offset",
    "_side_relation_xy_offsets",
    "_with_on_surface_release_offsets",
    "_with_inside_container_slot_offsets",
    "_with_self_relative_absolute_targets",
]

_SIDE_RELATION_DISTANCE = 0.16
_SIDE_RELEASE_Z_OFFSET = 0.12
_CONTAINER_SLOT_MIN_OFFSET = 0.04
_CONTAINER_SLOT_MAX_OFFSET = 0.12
_CONTAINER_SLOT_FRACTION = 0.25
_CONTAINER_SLOT_MAX_FRACTION = 0.40
_CONTAINER_SLOT_AXIS_TIE_RATIO = 0.10
_STAGING_Z_DELTA = 0.10
_POSE_SENSITIVE_STAGING_Z_DELTA = 0.25
_ON_RELEASE_Z_OFFSET = 0.2
_ON_SURFACE_RELEASE_CLEARANCE = 0.003
_ROBOT_VIEW_LEFT_WORLD_Y_SIGN = 1.0
_ROBOT_VIEW_FRONT_WORLD_X_SIGN = 1.0


def _relative_release_offset(relation: str) -> list[float]:
    relation = _normalize_relative_relation(relation)
    if relation == "inside":
        return [0.0, 0.0, _SIDE_RELEASE_Z_OFFSET]
    if relation == "on":
        return [0.0, 0.0, _ON_RELEASE_Z_OFFSET]
    if relation in _SIDE_RELATIONS:
        x_offset, y_offset = _side_relation_xy_offsets(relation)
        return [x_offset, y_offset, _SIDE_RELEASE_Z_OFFSET]
    raise ValueError(f"Unsupported relative placement relation: {relation!r}.")


def _side_relation_xy_offsets(relation: str) -> tuple[float, float]:
    relation = _normalize_relative_relation(relation)
    left_y = _ROBOT_VIEW_LEFT_WORLD_Y_SIGN * _SIDE_RELATION_DISTANCE
    right_y = -_ROBOT_VIEW_LEFT_WORLD_Y_SIGN * _SIDE_RELATION_DISTANCE
    front_x = _ROBOT_VIEW_FRONT_WORLD_X_SIGN * _SIDE_RELATION_DISTANCE
    behind_x = -_ROBOT_VIEW_FRONT_WORLD_X_SIGN * _SIDE_RELATION_DISTANCE
    if relation == "left_of":
        return 0.0, left_y
    if relation == "right_of":
        return 0.0, right_y
    if relation == "front_of":
        return front_x, 0.0
    if relation == "behind":
        return behind_x, 0.0
    if relation == "front_left_of":
        return front_x, left_y
    if relation == "back_left_of":
        return behind_x, left_y
    if relation == "front_right_of":
        return front_x, right_y
    if relation == "back_right_of":
        return behind_x, right_y
    raise ValueError(f"Unsupported side relation: {relation!r}.")


def _with_self_relative_absolute_targets(
    spec: _RelativePlacementSpec,
    gym_config: Mapping[str, Any],
) -> _RelativePlacementSpec:
    if not any(placement.reference_is_initial_pose for placement in spec.placements):
        return spec

    generated_positions = {
        str(obj.get("uid")): _clean_vector3(obj.get("init_pos", [0.0, 0.0, 0.0]))
        for obj in gym_config.get("rigid_object", [])
    }
    placements = tuple(
        _with_self_relative_absolute_target(placement, generated_positions)
        for placement in spec.placements
    )
    primary = placements[0]
    return _RelativePlacementSpec(
        table_source_uid=spec.table_source_uid,
        moved_source_uid=primary.moved_source_uid,
        reference_source_uid=primary.reference_source_uid,
        moved_runtime_uid=primary.moved_runtime_uid,
        reference_runtime_uid=primary.reference_runtime_uid,
        relation=primary.relation,
        active_side=primary.active_side,
        task_description=spec.task_description,
        task_prompt_summary=spec.task_prompt_summary,
        basic_background_notes=spec.basic_background_notes,
        action_sketch=spec.action_sketch,
        release_offset=primary.release_offset,
        high_offset=primary.high_offset,
        placements=placements,
        reference_is_initial_pose=primary.reference_is_initial_pose,
        release_position=primary.release_position,
        high_position=primary.high_position,
        orientation_goal=primary.orientation_goal,
        orientation_axis=primary.orientation_axis,
        orientation_align_to_runtime_uid=primary.orientation_align_to_runtime_uid,
    )


def _with_self_relative_absolute_target(
    placement: _RelativePlacementStepSpec,
    generated_positions: Mapping[str, list[float]],
) -> _RelativePlacementStepSpec:
    if not placement.reference_is_initial_pose:
        return placement
    initial_position = generated_positions.get(placement.moved_runtime_uid)
    if initial_position is None:
        raise ValueError(
            "Generated relative config missing self-relative moved object "
            f"{placement.moved_runtime_uid!r}."
        )
    release_position = _offset_position(initial_position, placement.release_offset)
    high_position = _offset_position(initial_position, placement.high_offset)
    return _RelativePlacementStepSpec(
        moved_source_uid=placement.moved_source_uid,
        reference_source_uid=placement.reference_source_uid,
        moved_runtime_uid=placement.moved_runtime_uid,
        reference_runtime_uid=placement.reference_runtime_uid,
        relation=placement.relation,
        active_side=placement.active_side,
        release_offset=placement.release_offset,
        high_offset=placement.high_offset,
        reference_is_initial_pose=True,
        release_position=release_position,
        high_position=high_position,
        orientation_goal=placement.orientation_goal,
        orientation_axis=placement.orientation_axis,
        orientation_align_to_runtime_uid=placement.orientation_align_to_runtime_uid,
    )


def _with_inside_container_slot_offsets(
    spec: _RelativePlacementSpec,
    gym_config: Mapping[str, Any],
) -> _RelativePlacementSpec:
    inside_groups: dict[str, list[int]] = {}
    for index, placement in enumerate(spec.placements):
        if placement.relation != "inside" or placement.reference_is_initial_pose:
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
        axis, slot_distance = _inside_container_slot_axis_and_distance(container_config)
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


def _with_relative_release_offset(
    placement: _RelativePlacementStepSpec,
    release_offset: Sequence[float],
) -> _RelativePlacementStepSpec:
    clean_release_offset = [round(float(value), 6) for value in release_offset]
    high_offset = list(clean_release_offset)
    high_offset[2] = round(high_offset[2] + _STAGING_Z_DELTA, 6)
    return replace(
        placement,
        release_offset=clean_release_offset,
        high_offset=high_offset,
    )


def _replace_relative_spec_placements(
    spec: _RelativePlacementSpec,
    placements: tuple[_RelativePlacementStepSpec, ...],
) -> _RelativePlacementSpec:
    primary = placements[0]
    return replace(
        spec,
        moved_source_uid=primary.moved_source_uid,
        reference_source_uid=primary.reference_source_uid,
        moved_runtime_uid=primary.moved_runtime_uid,
        reference_runtime_uid=primary.reference_runtime_uid,
        relation=primary.relation,
        active_side=primary.active_side,
        release_offset=primary.release_offset,
        high_offset=primary.high_offset,
        placements=placements,
        reference_is_initial_pose=primary.reference_is_initial_pose,
        release_position=primary.release_position,
        high_position=primary.high_position,
        orientation_goal=primary.orientation_goal,
        orientation_axis=primary.orientation_axis,
        orientation_align_to_runtime_uid=primary.orientation_align_to_runtime_uid,
    )


def _with_on_surface_release_offsets(
    spec: _RelativePlacementSpec,
    gym_config: Mapping[str, Any],
) -> _RelativePlacementSpec:
    placements = tuple(
        _with_on_surface_release_offset(placement, gym_config)
        for placement in spec.placements
    )
    return _replace_relative_spec_placements(spec, placements)


def _with_on_surface_release_offset(
    placement: _RelativePlacementStepSpec,
    gym_config: Mapping[str, Any],
) -> _RelativePlacementStepSpec:
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

    reference_origin_z = _clean_vector3(reference_config.get("init_pos", [0, 0, 0]))[2]
    release_offset = list(placement.release_offset)
    release_offset[2] = round(
        float(reference_top_z)
        - float(reference_origin_z)
        + _ON_SURFACE_RELEASE_CLEARANCE
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
    return replace(
        placement,
        release_offset=release_offset,
        high_offset=high_offset,
    )


def _target_local_zmin_for_orientation(
    obj_config: Mapping[str, Any],
    orientation_goal: str,
) -> float | None:
    if orientation_goal in {"preserve", "axis_align"}:
        return _mesh_config_local_zmin_after_rotation(obj_config)
    if orientation_goal == "upright":
        return 0.0
    if orientation_goal == "lay_flat":
        return _lay_flat_local_zmin(obj_config)
    return _mesh_config_local_zmin_after_rotation(obj_config)


def _lay_flat_local_zmin(obj_config: Mapping[str, Any]) -> float | None:
    shape = obj_config.get("shape", {})
    if not isinstance(shape, Mapping):
        return None
    mesh_path = shape.get("fpath")
    if not isinstance(mesh_path, str):
        return None

    from pathlib import Path

    from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
        _load_mesh_vertices,
    )

    vertices = _load_mesh_vertices(Path(mesh_path).expanduser().resolve())
    if not vertices:
        return None
    scale = _clean_vector3(obj_config.get("body_scale", [1.0, 1.0, 1.0]))
    extents = [
        (
            max(vertex[index] for vertex in vertices)
            - min(vertex[index] for vertex in vertices)
        )
        * scale[index]
        for index in range(3)
    ]
    sorted_extents = sorted(float(extent) for extent in extents)
    return -0.5 * sorted_extents[1]


def _inside_container_slot_axis_and_distance(
    container_config: Mapping[str, Any] | None,
) -> tuple[str, float]:
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
    return axis, round(float(slot_distance), 6)


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
    placements: Sequence[_RelativePlacementStepSpec],
    axis: str,
    object_configs: Mapping[str, Mapping[str, Any]],
    container_config: Mapping[str, Any] | None,
) -> list[int]:
    if axis == "y":
        side_order = {"left": 0, "right": 1}
        return sorted(
            indices,
            key=lambda index: (
                side_order.get(placements[index].active_side, 1),
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
    placement: _RelativePlacementStepSpec,
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


def _offset_position(
    position: Sequence[float],
    offset: Sequence[float],
) -> list[float]:
    return [
        round(float(position[index]) + float(offset[index]), 6) for index in range(3)
    ]


def _make_relative_summary(spec: _RelativePlacementSpec) -> dict[str, Any]:
    if len(spec.placements) == 1:
        return {
            "mode": "relative_placement",
            "moved_object": spec.moved_runtime_uid,
            "reference_object": spec.reference_runtime_uid,
            "relation": spec.relation,
            "active_arm": f"{spec.active_side}_arm",
            "release_offset": spec.release_offset,
            "orientation_goal": spec.orientation_goal,
            "orientation_axis": spec.orientation_axis,
            "orientation_align_to": spec.orientation_align_to_runtime_uid,
        }
    return {
        "mode": "dual_arm_relative_placement",
        "placements": [
            {
                "moved_object": placement.moved_runtime_uid,
                "reference_object": placement.reference_runtime_uid,
                "relation": placement.relation,
                "active_arm": f"{placement.active_side}_arm",
                "release_offset": placement.release_offset,
                "orientation_goal": placement.orientation_goal,
                "orientation_axis": placement.orientation_axis,
                "orientation_align_to": placement.orientation_align_to_runtime_uid,
            }
            for placement in spec.placements
        ],
    }
