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
import math
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _RelativePlacementSpec,
    _RelativePlacementStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _clean_vector3,
    _iter_generated_scene_object_configs,
    _load_mesh_vertices,
    _mesh_config_local_zmin_after_rotation,
    _mesh_config_transform_matrix,
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
    "_with_coordinated_side_release_height_offsets",
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
_PICKUP_UPRIGHT_ROTATE_RADIANS = math.pi / 4.0
_ROBOT_VIEW_LEFT_WORLD_Y_SIGN = 1.0
_ROBOT_VIEW_FRONT_WORLD_X_SIGN = 1.0
_DEFAULT_Y_AXIS_ARM_SLOT_SIDE_ORDER = {"right": 0, "left": 1}


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
        intent=primary.intent,
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
        hover_height=primary.hover_height,
        upright_in_place=primary.upright_in_place,
        pickup_upright_direction=primary.pickup_upright_direction,
        pickup_rotate_upright=primary.pickup_rotate_upright,
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
    return replace(
        placement,
        reference_is_initial_pose=True,
        release_position=release_position,
        high_position=high_position,
    )


def _with_inside_container_slot_offsets(
    spec: _RelativePlacementSpec,
    gym_config: Mapping[str, Any],
    *,
    slot_distance_scale: float = 1.0,
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
        axis, slot_distance = _inside_container_slot_axis_and_distance(
            container_config,
            slot_distance_scale=slot_distance_scale,
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


def _with_coordinated_side_release_height_offsets(
    spec: _RelativePlacementSpec,
    gym_config: Mapping[str, Any],
) -> _RelativePlacementSpec:
    if spec.intent not in {"place_relative", "coordinated_pickment"}:
        return spec
    placements = tuple(
        _with_coordinated_side_release_height_offset(placement, gym_config)
        for placement in spec.placements
    )
    return _replace_relative_spec_placements(spec, placements)


def _with_coordinated_side_release_height_offset(
    placement: _RelativePlacementStepSpec,
    gym_config: Mapping[str, Any],
) -> _RelativePlacementStepSpec:
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
    high_offset[2] = round(release_offset[2] + _STAGING_Z_DELTA, 6)
    return replace(
        placement,
        release_offset=release_offset,
        high_offset=high_offset,
    )


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
        hover_height=primary.hover_height,
        upright_in_place=primary.upright_in_place,
        pickup_upright_direction=primary.pickup_upright_direction,
        pickup_rotate_upright=primary.pickup_rotate_upright,
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

    reference_origin = _clean_vector3(reference_config.get("init_pos", [0, 0, 0]))
    moved_origin = _clean_vector3(moved_config.get("init_pos", [0, 0, 0]))
    release_offset = list(placement.release_offset)
    if placement.upright_in_place:
        release_offset[0] = round(float(moved_origin[0] - reference_origin[0]), 6)
        release_offset[1] = round(float(moved_origin[1] - reference_origin[1]), 6)
    release_offset[2] = round(
        float(reference_top_z)
        - float(reference_origin[2])
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
    vertices = _mesh_config_scaled_vertices(obj_config)
    if not vertices:
        return None
    return [round(float(value), 6) for value in _principal_local_axes(vertices)[0]]


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
    vertices = _mesh_config_scaled_vertices(obj_config)
    if not vertices:
        return None

    rotation = _preview_aware_upright_rotation(
        vertices,
        _mesh_config_rotation_basis(obj_config),
    )
    return min(_matrix_vector_mul(rotation, vertex)[2] for vertex in vertices)


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
        columns.append(_normalize_vector(column))
    return _columns_to_matrix(columns)


def _preview_aware_upright_rotation(
    vertices: Sequence[Sequence[float]],
    preview_rotation: Sequence[Sequence[float]],
) -> list[list[float]]:
    axes = _principal_local_axes(vertices)
    long_axis = axes[0]
    secondary_axes = list(axes[1:])
    candidates: list[tuple[float, list[list[float]]]] = []
    for secondary_axis in [
        *secondary_axes,
        *[_scale_vector(axis, -1.0) for axis in secondary_axes],
    ]:
        preview_secondary = _matrix_vector_mul(preview_rotation, secondary_axis)
        world_secondary = [preview_secondary[0], preview_secondary[1], 0.0]
        if _vector_norm(world_secondary) < 1e-6:
            continue
        rotation = _rotation_from_axis_targets(
            local_primary=long_axis,
            world_primary=[0.0, 0.0, 1.0],
            local_secondary=secondary_axis,
            world_secondary=world_secondary,
        )
        candidates.append(
            (_rotation_distance_score(rotation, preview_rotation), rotation)
        )
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return _rotation_from_axis_targets(
        local_primary=long_axis,
        world_primary=[0.0, 0.0, 1.0],
        local_secondary=axes[2],
        world_secondary=[1.0, 0.0, 0.0],
    )


def _principal_local_axes(
    vertices: Sequence[Sequence[float]],
) -> list[list[float]]:
    mins = [min(float(vertex[index]) for vertex in vertices) for index in range(3)]
    maxs = [max(float(vertex[index]) for vertex in vertices) for index in range(3)]
    extents = [maxs[index] - mins[index] for index in range(3)]
    order = sorted(range(3), key=lambda index: extents[index], reverse=True)
    return [[1.0 if axis == index else 0.0 for axis in range(3)] for index in order]


def _rotation_from_axis_targets(
    *,
    local_primary: Sequence[float],
    world_primary: Sequence[float],
    local_secondary: Sequence[float],
    world_secondary: Sequence[float],
) -> list[list[float]]:
    local_primary = _normalize_vector(local_primary)
    world_primary = _normalize_vector(world_primary)
    local_secondary = _orthogonalized_axis(local_secondary, local_primary)
    world_secondary = _orthogonalized_axis(world_secondary, world_primary)
    local_basis = _columns_to_matrix(
        [
            local_primary,
            local_secondary,
            _normalize_vector(_cross(local_primary, local_secondary)),
        ]
    )
    world_basis = _columns_to_matrix(
        [
            world_primary,
            world_secondary,
            _normalize_vector(_cross(world_primary, world_secondary)),
        ]
    )
    return _matrix_multiply(world_basis, _matrix_transpose(local_basis))


def _orthogonalized_axis(
    axis: Sequence[float],
    reference: Sequence[float],
) -> list[float]:
    dot = _dot(axis, reference)
    projected = [
        float(axis[index]) - dot * float(reference[index]) for index in range(3)
    ]
    if _vector_norm(projected) < 1e-6:
        fallback = [1.0, 0.0, 0.0]
        if abs(_dot(fallback, reference)) > 0.9:
            fallback = [0.0, 1.0, 0.0]
        fallback_dot = _dot(fallback, reference)
        projected = [
            fallback[index] - fallback_dot * float(reference[index])
            for index in range(3)
        ]
    return _normalize_vector(projected)


def _rotation_distance_score(
    rotation: Sequence[Sequence[float]],
    preview_rotation: Sequence[Sequence[float]],
) -> float:
    delta = _matrix_multiply(rotation, _matrix_transpose(preview_rotation))
    return -sum(float(delta[index][index]) for index in range(3))


def _columns_to_matrix(columns: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(columns[col][row]) for col in range(3)] for row in range(3)]


def _matrix_multiply(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> list[list[float]]:
    return [
        [
            sum(float(left[row][k]) * float(right[k][col]) for k in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def _matrix_transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(matrix[col][row]) for col in range(3)] for row in range(3)]


def _matrix_vector_mul(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> list[float]:
    return [
        sum(float(matrix[row][col]) * float(vector[col]) for col in range(3))
        for row in range(3)
    ]


def _normalize_vector(vector: Sequence[float]) -> list[float]:
    norm = _vector_norm(vector)
    if norm < 1e-6:
        raise ValueError("Cannot normalize a near-zero vector.")
    return [float(value) / norm for value in vector]


def _scale_vector(vector: Sequence[float], scale: float) -> list[float]:
    return [float(value) * float(scale) for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    ]


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


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
    placements: Sequence[_RelativePlacementStepSpec],
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
    if spec.intent == "coordinated_pickment":
        return {
            "mode": "coordinated_pickment",
            "intent": spec.intent,
            "moved_object": spec.moved_runtime_uid,
            "reference_object": spec.reference_runtime_uid,
            "relation": spec.relation,
            "active_arm": "dual_arm",
            "release_offset": spec.release_offset,
            "target_position": spec.release_position,
            "orientation_goal": spec.orientation_goal,
            "orientation_axis": spec.orientation_axis,
            "orientation_align_to": spec.orientation_align_to_runtime_uid,
        }
    if len(spec.placements) == 1:
        summary = {
            "mode": "object_manipulation",
            "intent": spec.intent,
            "moved_object": spec.moved_runtime_uid,
            "reference_object": spec.reference_runtime_uid,
            "relation": spec.relation,
            "active_arm": f"{spec.active_side}_arm",
            "release_offset": spec.release_offset,
            "hover_height": spec.hover_height,
            "orientation_goal": spec.orientation_goal,
            "orientation_axis": spec.orientation_axis,
            "orientation_align_to": spec.orientation_align_to_runtime_uid,
        }
        if spec.upright_in_place:
            summary["upright_in_place"] = True
        if spec.pickup_upright_direction is not None:
            summary["pickup_upright_direction"] = spec.pickup_upright_direction
        if spec.pickup_rotate_upright is not None:
            summary["pickup_rotate_upright"] = spec.pickup_rotate_upright
        return summary
    return {
        "mode": "dual_arm_object_manipulation",
        "manipulations": [
            _relative_placement_summary(placement) for placement in spec.placements
        ],
    }


def _relative_placement_summary(
    placement: _RelativePlacementStepSpec,
) -> dict[str, Any]:
    summary = {
        "intent": placement.intent,
        "moved_object": placement.moved_runtime_uid,
        "reference_object": placement.reference_runtime_uid,
        "relation": placement.relation,
        "active_arm": f"{placement.active_side}_arm",
        "release_offset": placement.release_offset,
        "hover_height": placement.hover_height,
        "orientation_goal": placement.orientation_goal,
        "orientation_axis": placement.orientation_axis,
        "orientation_align_to": placement.orientation_align_to_runtime_uid,
    }
    if placement.upright_in_place:
        summary["upright_in_place"] = True
    if placement.pickup_upright_direction is not None:
        summary["pickup_upright_direction"] = placement.pickup_upright_direction
    if placement.pickup_rotate_upright is not None:
        summary["pickup_rotate_upright"] = placement.pickup_rotate_upright
    return summary
