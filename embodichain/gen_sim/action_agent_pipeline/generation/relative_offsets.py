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

"""Resolve basic relative offsets, absolute targets, and arm assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.config.defaults import defaults_section
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    RelativePlacementSpec,
    RelativePlacementStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _clean_vector3,
    _iter_generated_scene_object_configs,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_intent import (
    _SIDE_RELATIONS,
    _normalize_relative_relation,
    _relative_primary_placement,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _arm_side_for_position,
)

__all__ = [
    "_POSE_SENSITIVE_STAGING_Z_DELTA",
    "_STAGING_Z_DELTA",
    "_offset_position",
    "_relative_release_offset",
    "_replace_relative_spec_placements",
    "_side_relation_xy_offsets",
    "_with_final_auto_arm_sides",
    "_with_relative_release_offset",
    "_with_self_relative_absolute_targets",
]

_DEFAULTS = defaults_section("relative_placement")
_SIDE_RELATION_DISTANCE = float(_DEFAULTS["side_relation_distance"])
_SIDE_RELEASE_Z_OFFSET = float(_DEFAULTS["side_release_z_offset"])
_STAGING_Z_DELTA = float(_DEFAULTS["staging_z_delta"])
_POSE_SENSITIVE_STAGING_Z_DELTA = float(_DEFAULTS["pose_sensitive_staging_z_delta"])
_ON_RELEASE_Z_OFFSET = float(_DEFAULTS["on_release_z_offset"])
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
    spec: RelativePlacementSpec,
    gym_config: Mapping[str, Any],
) -> RelativePlacementSpec:
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
    return _replace_relative_spec_placements(spec, placements)


def _with_final_auto_arm_sides(
    spec: RelativePlacementSpec,
    gym_config: Mapping[str, Any],
) -> RelativePlacementSpec:
    if spec.intent == "coordinated_pickment" or not spec.placements:
        return spec

    placements = list(spec.placements)
    object_positions = _generated_object_positions(gym_config)
    inferred_sides = {
        index: _arm_side_for_position(
            _require_generated_object_position(
                object_positions,
                placements[index].moved_runtime_uid,
            )
        )
        for index, placement in enumerate(placements)
        if placement.intent != "coordinated_pickment"
        and placement.arm_request == "auto"
    }

    for index, active_side in inferred_sides.items():
        placements[index] = replace(placements[index], active_side=active_side)
    return _replace_relative_spec_placements(spec, tuple(placements))


def _generated_object_positions(
    gym_config: Mapping[str, Any],
) -> dict[str, list[float]]:
    return {
        str(obj.get("uid")): _clean_vector3(obj.get("init_pos", [0.0, 0.0, 0.0]))
        for obj in _iter_generated_scene_object_configs(gym_config)
        if obj.get("uid") is not None
    }


def _require_generated_object_position(
    object_positions: Mapping[str, list[float]],
    runtime_uid: str,
) -> list[float]:
    position = object_positions.get(runtime_uid)
    if position is None:
        raise ValueError(
            "Generated relative config missing moved object "
            f"{runtime_uid!r} for auto arm assignment."
        )
    return position


def _with_self_relative_absolute_target(
    placement: RelativePlacementStepSpec,
    generated_positions: Mapping[str, list[float]],
) -> RelativePlacementStepSpec:
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


def _with_relative_release_offset(
    placement: RelativePlacementStepSpec,
    release_offset: Sequence[float],
) -> RelativePlacementStepSpec:
    clean_release_offset = [round(float(value), 6) for value in release_offset]
    high_offset = list(clean_release_offset)
    high_offset[2] = round(high_offset[2] + _STAGING_Z_DELTA, 6)
    return replace(
        placement,
        release_offset=clean_release_offset,
        high_offset=high_offset,
    )


def _replace_relative_spec_placements(
    spec: RelativePlacementSpec,
    placements: tuple[RelativePlacementStepSpec, ...],
) -> RelativePlacementSpec:
    primary = _relative_primary_placement(placements)
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
        surface_clearance=primary.surface_clearance,
    )


def _offset_position(
    position: Sequence[float],
    offset: Sequence[float],
) -> list[float]:
    return [
        round(float(position[index]) + float(offset[index]), 6) for index in range(3)
    ]
