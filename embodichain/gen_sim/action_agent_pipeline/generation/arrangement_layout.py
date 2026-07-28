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
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.generation._spec_scene_helpers import (
    resolved_mesh_config as _resolved_mesh_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _clean_vector3,
    _iter_generated_scene_object_configs,
    _mesh_config_has_distinct_xy_axis,
    _mesh_config_world_xy_bounds,
    _mesh_config_world_xy_center,
    _mesh_config_world_xy_extents,
    _mesh_config_world_z_bounds,
)

__all__ = [
    "_ArrangementFootprint",
    "_arrangement_collision_aware_line_slots",
    "_arrangement_config_orientation",
    "_arrangement_hard_obstacle_objects",
    "_arrangement_line_slot_positions",
    "_arrangement_object_footprint",
    "_arrangement_object_orientation",
    "_arrangement_object_size_score",
    "_arrangement_orientation_axis",
    "_arrangement_spacing",
    "_generated_arrangement_hard_obstacles",
    "_normalize_anchor",
    "_normalize_axis",
    "_slot_xy_bounds",
    "_source_object_xy_bounds",
    "_table_anchor_xy",
    "_xy_bounds_overlap",
]

_DEFAULTS = defaults_section("arrangement")
_SLOT_MARGIN = float(_DEFAULTS["slot_margin"])
_MIN_SLOT_SPACING = float(_DEFAULTS["min_slot_spacing"])
_ROW_SEARCH_STEP = float(_DEFAULTS["row_search_step"])
_ROW_SEARCH_RADIUS = float(_DEFAULTS["row_search_radius"])
_MOVABLE_INITIAL_OVERLAP_SCORE_WEIGHT = float(
    _DEFAULTS["movable_initial_overlap_score_weight"]
)
_SUPPORTED_AXES = {"table_long_axis", "world_x", "world_y"}
_CONCRETE_AXES = {"world_x", "world_y"}


def _arrangement_line_slot_positions(
    *,
    anchor_xy: Sequence[float],
    count: int,
    spacing: float,
    line_axis: str,
    table_bounds: tuple[list[float], list[float]] | None = None,
) -> list[list[float]]:
    if count < 1:
        raise ValueError("Arrangement line requires at least one slot.")
    axis = _resolve_concrete_line_axis(line_axis, table_bounds=table_bounds)
    anchor = [float(anchor_xy[0]), float(anchor_xy[1])]
    center = (count - 1) / 2.0
    slots: list[list[float]] = []
    for index in range(count):
        axis_offset = (index - center) * float(spacing)
        if axis == "world_y":
            slots.append(
                [
                    round(anchor[0], 6),
                    round(anchor[1] + axis_offset, 6),
                ]
            )
            continue
        if axis == "world_x":
            slots.append(
                [
                    round(anchor[0] + axis_offset, 6),
                    round(anchor[1], 6),
                ]
            )
            continue
        raise ValueError(f"Unsupported arrangement line axis: {line_axis!r}.")
    return slots


def _arrangement_collision_aware_line_slots(
    *,
    anchor_xy: Sequence[float],
    table_obj: SceneObject,
    objects: Sequence[SceneObject],
    count: int,
    spacing: float,
    line_axis: str,
    scene_dir: Path,
    clearance: float,
    ignore_self_initial_overlap: bool = False,
    hard_obstacle_objects: Sequence[SceneObject] = (),
) -> tuple[list[list[float]], list[float]]:
    axis = _normalize_axis(line_axis)
    if count != len(objects):
        raise ValueError("Arrangement slot count must match object count.")

    table_bounds = _source_object_xy_bounds(table_obj, scene_dir=scene_dir)
    if table_bounds is None:
        raise ValueError("Arrangement requires table mesh XY bounds for safe layout.")
    table_min, table_max = table_bounds
    concrete_axis = _resolve_concrete_line_axis(axis, table_bounds=table_bounds)
    object_footprints = [
        _arrangement_object_footprint(obj, scene_dir=scene_dir) for obj in objects
    ]
    init_bounds = [footprint.xy_bounds for footprint in object_footprints]
    hard_obstacle_bounds = [
        _arrangement_object_footprint(obj, scene_dir=scene_dir).xy_bounds
        for obj in hard_obstacle_objects
    ]

    # Movable overlaps are a cost because execution scheduling can clear them.
    # Table bounds and hard obstacles remain strict feasibility constraints.
    best_candidate: tuple[float, float, list[list[float]], list[float]] | None = None
    for perpendicular_offset in _row_search_offsets(
        _ROW_SEARCH_RADIUS,
        _ROW_SEARCH_STEP,
    ):
        origin = _line_origin_with_perpendicular_offset(
            anchor_xy,
            perpendicular_offset,
            concrete_axis,
        )
        slots = _arrangement_line_slot_positions(
            anchor_xy=origin,
            count=count,
            spacing=spacing,
            line_axis=concrete_axis,
            table_bounds=table_bounds,
        )
        slot_bounds = [
            _slot_xy_bounds(slot, max_half_extent=footprint.half_extent)
            for slot, footprint in zip(slots, object_footprints)
        ]
        if not _slot_bounds_within_table(
            slot_bounds,
            table_min=table_min,
            table_max=table_max,
            clearance=clearance,
        ):
            continue
        if _slot_bounds_overlap_initial_objects(
            slot_bounds,
            hard_obstacle_bounds,
            clearance=clearance,
            ignore_self_initial_overlap=False,
        ):
            continue
        movable_overlap_score = _slot_bounds_initial_overlap_score(
            slot_bounds,
            init_bounds,
            clearance=clearance,
            ignore_self_initial_overlap=ignore_self_initial_overlap,
        )
        score = _arrangement_line_candidate_score(
            perpendicular_offset=perpendicular_offset,
            movable_overlap_score=movable_overlap_score,
        )
        candidate = (score, abs(float(perpendicular_offset)), slots, origin)
        if best_candidate is None or candidate[:2] < best_candidate[:2]:
            best_candidate = candidate

    if best_candidate is not None:
        return best_candidate[2], best_candidate[3]

    raise ValueError(
        "Unable to generate a collision-free one-line arrangement near the table "
        "center. The selected objects may be too many, too large, or already "
        "occupying all candidate row positions; use a larger table or add parking "
        "slot planning."
    )


def _slot_bounds_overlap_initial_objects(
    slot_bounds: Sequence[tuple[list[float], list[float]]],
    init_bounds: Sequence[tuple[list[float], list[float]]],
    *,
    clearance: float,
    ignore_self_initial_overlap: bool,
) -> bool:
    for slot_index, slot_bound in enumerate(slot_bounds):
        for init_index, init_bound in enumerate(init_bounds):
            if ignore_self_initial_overlap and slot_index == init_index:
                continue
            if _xy_bounds_overlap(slot_bound, init_bound, clearance=clearance):
                return True
    return False


def _slot_bounds_initial_overlap_score(
    slot_bounds: Sequence[tuple[list[float], list[float]]],
    init_bounds: Sequence[tuple[list[float], list[float]]],
    *,
    clearance: float,
    ignore_self_initial_overlap: bool,
) -> float:
    overlap_count = 0
    for slot_index, slot_bound in enumerate(slot_bounds):
        for init_index, init_bound in enumerate(init_bounds):
            if ignore_self_initial_overlap and slot_index == init_index:
                continue
            if _xy_bounds_overlap(slot_bound, init_bound, clearance=clearance):
                overlap_count += 1
    return float(overlap_count)


def _arrangement_line_candidate_score(
    *,
    perpendicular_offset: float,
    movable_overlap_score: float,
) -> float:
    return abs(float(perpendicular_offset)) + (
        _MOVABLE_INITIAL_OVERLAP_SCORE_WEIGHT * float(movable_overlap_score)
    )


def _row_search_offsets(radius: float, step: float) -> list[float]:
    offsets = [0.0]
    steps = int(float(radius) / float(step))
    for index in range(1, steps + 1):
        value = round(float(index) * float(step), 6)
        offsets.extend([value, -value])
    return offsets


class _ArrangementFootprint:
    """XY bounds and conservative radius used by arrangement planning."""

    def __init__(
        self,
        *,
        xy_bounds: tuple[list[float], list[float]],
        half_extent: float,
    ) -> None:
        self.xy_bounds = xy_bounds
        self.half_extent = half_extent


def _arrangement_object_footprint(
    obj: SceneObject,
    *,
    scene_dir: Path,
) -> _ArrangementFootprint:
    bounds = _source_object_xy_bounds(obj, scene_dir=scene_dir)
    if bounds is None:
        position = _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0]))
        half_extent = _MIN_SLOT_SPACING / 2.0
        bounds = (
            [position[0] - half_extent, position[1] - half_extent],
            [position[0] + half_extent, position[1] + half_extent],
        )
    mins, maxs = bounds
    half_extent = max(
        (float(maxs[0]) - float(mins[0])) / 2.0,
        (float(maxs[1]) - float(mins[1])) / 2.0,
        _MIN_SLOT_SPACING / 2.0,
    )
    return _ArrangementFootprint(xy_bounds=bounds, half_extent=half_extent)


def _source_object_xy_bounds(
    obj: SceneObject,
    *,
    scene_dir: Path,
) -> tuple[list[float], list[float]] | None:
    config = _resolved_mesh_config(obj, scene_dir=scene_dir)
    return _mesh_config_world_xy_bounds(config)


def _slot_xy_bounds(
    slot: Sequence[float],
    *,
    max_half_extent: float,
) -> tuple[list[float], list[float]]:
    return (
        [float(slot[0]) - max_half_extent, float(slot[1]) - max_half_extent],
        [float(slot[0]) + max_half_extent, float(slot[1]) + max_half_extent],
    )


def _slot_bounds_within_table(
    slot_bounds: Sequence[tuple[list[float], list[float]]],
    *,
    table_min: Sequence[float],
    table_max: Sequence[float],
    clearance: float,
) -> bool:
    for mins, maxs in slot_bounds:
        if mins[0] < float(table_min[0]) + clearance:
            return False
        if maxs[0] > float(table_max[0]) - clearance:
            return False
        if mins[1] < float(table_min[1]) + clearance:
            return False
        if maxs[1] > float(table_max[1]) - clearance:
            return False
    return True


def _xy_bounds_overlap(
    first: tuple[list[float], list[float]],
    second: tuple[list[float], list[float]],
    *,
    clearance: float,
) -> bool:
    first_min, first_max = first
    second_min, second_max = second
    return not (
        first_max[0] + clearance <= second_min[0]
        or second_max[0] + clearance <= first_min[0]
        or first_max[1] + clearance <= second_min[1]
        or second_max[1] + clearance <= first_min[1]
    )


def _arrangement_orientation_axis(
    line_axis: str,
    *,
    table_bounds: tuple[list[float], list[float]] | None = None,
) -> str:
    axis = _resolve_concrete_line_axis(line_axis, table_bounds=table_bounds)
    if axis == "world_x":
        return "x"
    if axis == "world_y":
        return "y"
    raise ValueError(f"Unsupported arrangement line axis: {line_axis!r}.")


def _arrangement_object_orientation(
    obj: SceneObject,
    *,
    orientation_axis: str,
    scene_dir: Path,
) -> tuple[str, str]:
    return _arrangement_config_orientation(
        _resolved_mesh_config(obj, scene_dir=scene_dir),
        orientation_axis=orientation_axis,
    )


def _arrangement_config_orientation(
    obj_config: Mapping[str, Any],
    *,
    orientation_axis: str,
) -> tuple[str, str]:
    if _mesh_config_has_distinct_xy_axis(obj_config):
        return "axis_align", orientation_axis
    return "preserve", "none"


def _normalize_axis(value: Any) -> str:
    text = (
        str(value or "table_long_axis")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    aliases = {
        "left_to_right": "table_long_axis",
        "left_right": "table_long_axis",
        "robot_left_to_right": "table_long_axis",
        "long_axis": "table_long_axis",
        "table_long": "table_long_axis",
        "table_longest_axis": "table_long_axis",
        "x": "world_x",
        "table_x": "world_x",
        "y": "world_y",
        "table_y": "world_y",
    }
    text = aliases.get(text, text)
    if text not in _SUPPORTED_AXES:
        raise ValueError(
            f"Unsupported arrangement line axis {value!r}; expected one of "
            f"{sorted(_SUPPORTED_AXES)}."
        )
    return text


def _normalize_anchor(value: Any) -> str:
    text = str(value or "table_center").strip().lower().replace("-", "_")
    aliases = {
        "center": "table_center",
        "table_centre": "table_center",
        "桌子中央": "table_center",
        "桌面中央": "table_center",
    }
    text = aliases.get(text, text)
    if text != "table_center":
        raise ValueError("Arrangement only supports anchor='table_center'.")
    return text


def _resolve_concrete_line_axis(
    line_axis: str,
    *,
    table_bounds: tuple[list[float], list[float]] | None = None,
) -> str:
    axis = _normalize_axis(line_axis)
    if axis in _CONCRETE_AXES:
        return axis
    if axis != "table_long_axis":
        raise ValueError(f"Unsupported arrangement line axis: {line_axis!r}.")
    if table_bounds is None:
        return "world_y"
    table_min, table_max = table_bounds
    x_extent = float(table_max[0]) - float(table_min[0])
    y_extent = float(table_max[1]) - float(table_min[1])
    if x_extent > y_extent:
        return "world_x"
    return "world_y"


def _line_origin_with_perpendicular_offset(
    anchor_xy: Sequence[float],
    perpendicular_offset: float,
    concrete_axis: str,
) -> list[float]:
    origin = [round(float(anchor_xy[0]), 6), round(float(anchor_xy[1]), 6)]
    if concrete_axis == "world_x":
        origin[1] = round(origin[1] + float(perpendicular_offset), 6)
        return origin
    if concrete_axis == "world_y":
        origin[0] = round(origin[0] + float(perpendicular_offset), 6)
        return origin
    raise ValueError(f"Unsupported concrete arrangement axis: {concrete_axis!r}.")


def _arrangement_hard_obstacle_objects(
    scene_objects: Sequence[SceneObject],
    *,
    selected_source_uids: set[str],
    table_source_uid: str,
) -> list[SceneObject]:
    return [
        obj
        for obj in scene_objects
        if obj.source_uid != table_source_uid
        and obj.source_uid not in selected_source_uids
    ]


def _generated_arrangement_hard_obstacles(
    gym_config: Mapping[str, Any],
    *,
    moved_runtime_uids: set[str],
    table_source_uid: str,
) -> list[SceneObject]:
    obstacles = []
    for config in _iter_generated_scene_object_configs(gym_config):
        if not isinstance(config, Mapping):
            continue
        runtime_uid = str(config.get("uid", ""))
        if not runtime_uid or runtime_uid in moved_runtime_uids:
            continue
        if runtime_uid in {"table", table_source_uid}:
            continue
        obstacles.append(
            SceneObject(
                source_uid=runtime_uid,
                source_role="background",
                config=dict(config),
            )
        )
    return obstacles


def _table_anchor_xy(
    table_obj: SceneObject,
    anchor: str,
    *,
    scene_dir: Path,
) -> list[float]:
    _normalize_anchor(anchor)
    center = _mesh_config_world_xy_center(
        _resolved_mesh_config(table_obj, scene_dir=scene_dir)
    )
    if center is not None:
        return center
    init_pos = _clean_vector3(table_obj.config.get("init_pos", [0.0, 0.0, 0.0]))
    return [round(init_pos[0], 6), round(init_pos[1], 6)]


def _arrangement_spacing(
    objects: Sequence[SceneObject],
    *,
    scene_dir: Path,
) -> float:
    max_extent = max(
        (_arrangement_object_xy_extent(obj, scene_dir=scene_dir) or 0.0)
        for obj in objects
    )
    spacing = max(max_extent + _SLOT_MARGIN, _MIN_SLOT_SPACING)
    return round(float(spacing), 6)


def _arrangement_object_size_score(
    obj: SceneObject,
    *,
    scene_dir: Path,
) -> float | None:
    bounds = _source_mesh_world_bounds(obj, scene_dir=scene_dir)
    if bounds is None:
        return None
    mins, maxs = bounds
    extents = [maxs[index] - mins[index] for index in range(3)]
    return round(float(max(extents)), 6)


def _arrangement_object_xy_extent(
    obj: SceneObject,
    *,
    scene_dir: Path,
) -> float | None:
    config = _resolved_mesh_config(obj, scene_dir=scene_dir)
    extents = _mesh_config_world_xy_extents(config)
    if extents is None:
        return None
    return max(extents)


def _source_mesh_world_bounds(
    obj: SceneObject,
    *,
    scene_dir: Path,
) -> tuple[list[float], list[float]] | None:
    config = _resolved_mesh_config(obj, scene_dir=scene_dir)
    z_bounds = _mesh_config_world_z_bounds(config)
    xy_extents = _mesh_config_world_xy_extents(config)
    if z_bounds is None or xy_extents is None:
        return None
    return [0.0, 0.0, z_bounds[0]], [xy_extents[0], xy_extents[1], z_bounds[1]]
