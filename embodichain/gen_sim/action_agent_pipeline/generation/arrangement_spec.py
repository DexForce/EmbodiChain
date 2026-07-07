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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
import json

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _ArrangementLineSpec,
    _ArrangementLineStepSpec,
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _TABLETOP_OBJECT_CLEARANCE,
    _clean_vector3,
    _iter_generated_scene_object_configs,
    _mesh_config_has_distinct_xy_axis,
    _mesh_config_local_zmin_after_rotation,
    _mesh_config_world_xy_bounds,
    _mesh_config_world_xy_center,
    _mesh_config_world_xy_extents,
    _mesh_config_world_z_bounds,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _base_name,
    _normalize_runtime_uid,
    _string_list,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _arm_side_for_position,
    _pick_table,
)

__all__ = [
    "_apply_arrangement_task_response",
    "_arrangement_line_slot_positions",
    "_build_arrangement_line_spec_with_llm",
    "_call_arrangement_task_llm",
    "_is_arrangement_task_description",
    "_make_arrangement_scene_summary",
    "_with_arrangement_generated_pose_targets",
    "_with_arrangement_generated_z_targets",
]

_ARRANGEMENT_KEYWORDS = (
    "arrange",
    "sort",
    "order",
    "line",
    "left to right",
    "left-to-right",
    "从左到右",
    "由大到小",
    "从大到小",
    "由小到大",
    "从小到大",
    "排序",
    "排列",
    "排成",
    "一行",
)
_DEFAULT_RELEASE_Z = 0.04
_DEFAULT_STAGING_Z_DELTA = 0.10
_POSE_SENSITIVE_STAGING_Z_DELTA = 0.15
_SLOT_MARGIN = 0.025
_MIN_SLOT_SPACING = 0.07
_LAYOUT_CLEARANCE = 0.025
_ROW_SEARCH_STEP = 0.025
_ROW_SEARCH_RADIUS = 0.25
_MOVABLE_INITIAL_OVERLAP_SCORE_WEIGHT = 0.01
_SUPPORTED_ORDER_BY = {"size", "color", "explicit"}
_SUPPORTED_ORDER_DIRECTIONS = {"ascending", "descending", "given"}
_SUPPORTED_AXES = {"table_long_axis", "world_x", "world_y"}
_CONCRETE_AXES = {"world_x", "world_y"}


def _is_arrangement_task_description(task_description: str) -> bool:
    text = task_description.strip().lower()
    return any(keyword in text for keyword in _ARRANGEMENT_KEYWORDS)


def _build_arrangement_line_spec_with_llm(
    *,
    scene_objects: list[_SceneObject],
    project_name: str,
    scene_dir: Path,
    task_description: str,
    model: str | None,
    task_llm_caller: Callable[..., Mapping[str, Any]] | None = None,
) -> _ArrangementLineSpec:
    background_objects = [
        obj for obj in scene_objects if obj.source_role == "background"
    ]
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]
    if not background_objects:
        raise ValueError("Arrangement generation requires a background table.")
    if len(rigid_objects) < 2:
        raise ValueError(
            "Arrangement generation requires at least two movable objects."
        )

    table = _pick_table(background_objects)
    scene_summary = _make_arrangement_scene_summary(
        scene_objects,
        scene_dir=scene_dir,
    )
    if task_llm_caller is None:
        task_llm_caller = _call_arrangement_task_llm
    response = task_llm_caller(
        project_name=project_name,
        task_description=task_description,
        scene_summary=scene_summary,
        model=model,
    )
    return _apply_arrangement_task_response(
        response=response,
        table_source_uid=table.source_uid,
        scene_objects=scene_objects,
        rigid_objects=rigid_objects,
        scene_dir=scene_dir,
        task_description=task_description,
        check_static_obstacles=False,
    )


def _call_arrangement_task_llm(
    *,
    project_name: str,
    task_description: str,
    scene_summary: list[dict[str, Any]],
    model: str | None,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import (
        extract_json_object,
    )
    from embodichain.gen_sim.action_agent_pipeline.utils.mllm import (
        create_chat_openai,
    )

    prompt = (
        "Parse a tabletop multi-object line arrangement task and produce one "
        "strict config-level JSON spec. The generator computes all target slot "
        "coordinates deterministically from this spec.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "objects": ["<source_uid from rigid_object>", "..."],\n'
        '  "order_by": "size|color|explicit",\n'
        '  "order_direction": "ascending|descending|given",\n'
        '  "ordered_attributes": ["red", "green", "blue"],\n'
        '  "object_attributes": {"<source_uid>": {"color": "red"}},\n'
        '  "anchor": "table_center",\n'
        '  "line_axis": "table_long_axis|world_x|world_y",\n'
        '  "task_prompt_summary": "<short execution summary>",\n'
        '  "basic_background_notes": "<short notes>"\n'
        "}\n\n"
        "Rules:\n"
        "- Use only source_uid values from rigid_object scene items.\n"
        "- Include every object that must be moved and sorted.\n"
        "- Use order_by='size' for large/small ordering. Use "
        "order_direction='descending' for large-to-small and 'ascending' for "
        "small-to-large.\n"
        "- Use order_by='color' when the task specifies a color sequence such as "
        "red-green-blue. Put that sequence in ordered_attributes and include a "
        "color attribute for each object.\n"
        "- Use line_axis='table_long_axis' for generic row tasks. Use "
        "'world_x' or 'world_y' only when the task explicitly constrains the "
        "world axis.\n"
        "- Do not return target positions, robot config, success JSON, or action "
        "graphs.\n\n"
        f"Project: {project_name}\n"
        f"Task description:\n{task_description}\n"
        f"Scene objects:\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}"
    )
    llm = create_chat_openai(
        temperature=0.0,
        model=model,
        usage_stage="config_generation.arrangement_task",
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You produce strict JSON specs for simulation config "
                    "generation. Do not include markdown."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    content = getattr(response, "content", response)
    return extract_json_object(content)


def _make_arrangement_scene_summary(
    scene_objects: Sequence[_SceneObject],
    *,
    scene_dir: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "source_uid": obj.source_uid,
            "role": obj.source_role,
            "object_type": _base_name(obj),
            "description": str(obj.config.get("description", "")).strip(),
            "mesh": obj.config.get("shape", {}).get("fpath"),
            "init_pos": obj.config.get("init_pos"),
            "body_scale": obj.config.get("body_scale"),
            "color_hint": _color_hint_for_object(obj),
            "size_score": _arrangement_object_size_score(
                obj,
                scene_dir=scene_dir,
            ),
        }
        for obj in scene_objects
    ]


def _apply_arrangement_task_response(
    *,
    response: Mapping[str, Any],
    table_source_uid: str,
    scene_objects: list[_SceneObject],
    rigid_objects: list[_SceneObject],
    scene_dir: Path,
    task_description: str,
    check_static_obstacles: bool = True,
) -> _ArrangementLineSpec:
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    table_obj = by_uid[table_source_uid]
    rigid_by_uid = {obj.source_uid: obj for obj in rigid_objects}
    runtime_uids = _arrangement_runtime_uid_mapping(rigid_objects)

    object_source_uids = _resolve_arrangement_object_uids(
        response.get("objects"),
        rigid_by_uid,
    )
    object_attributes = _object_attributes(response.get("object_attributes"))
    order_by = _normalize_order_by(response.get("order_by"))
    order_direction = _normalize_order_direction(response.get("order_direction"))
    axis = _normalize_axis(response.get("line_axis", response.get("axis")))
    anchor = _normalize_anchor(response.get("anchor"))

    if order_by == "size":
        ordered_source_uids = _order_uids_by_size(
            object_source_uids,
            rigid_by_uid=rigid_by_uid,
            scene_dir=scene_dir,
            descending=order_direction != "ascending",
        )
        order_direction = (
            "descending" if order_direction == "given" else order_direction
        )
    elif order_by == "color":
        ordered_source_uids = _order_uids_by_color(
            object_source_uids,
            rigid_by_uid=rigid_by_uid,
            object_attributes=object_attributes,
            ordered_colors=_string_list(response.get("ordered_attributes")),
        )
        order_direction = "given"
    else:
        ordered_source_uids = object_source_uids
        order_direction = "given"

    anchor_xy = _table_anchor_xy(table_obj, anchor, scene_dir=scene_dir)
    spacing = _arrangement_spacing(
        [rigid_by_uid[uid] for uid in object_source_uids],
        scene_dir=scene_dir,
    )
    table_bounds = _source_object_xy_bounds(table_obj, scene_dir=scene_dir)
    hard_obstacle_objects = (
        _arrangement_hard_obstacle_objects(
            scene_objects,
            selected_source_uids=set(object_source_uids),
            table_source_uid=table_source_uid,
        )
        if check_static_obstacles
        else ()
    )
    slots, line_origin_xy = _arrangement_collision_aware_line_slots(
        anchor_xy=anchor_xy,
        table_obj=table_obj,
        objects=[rigid_by_uid[uid] for uid in ordered_source_uids],
        count=len(ordered_source_uids),
        spacing=spacing,
        line_axis=axis,
        scene_dir=scene_dir,
        clearance=_LAYOUT_CLEARANCE,
        ignore_self_initial_overlap=True,
        hard_obstacle_objects=hard_obstacle_objects,
    )
    orientation_axis = _arrangement_orientation_axis(axis, table_bounds=table_bounds)

    steps = []
    for slot_index, (source_uid, target_xy) in enumerate(
        zip(ordered_source_uids, slots)
    ):
        obj = rigid_by_uid[source_uid]
        release_z = _release_z_for_object(obj)
        release_position = [
            round(float(target_xy[0]), 6),
            round(float(target_xy[1]), 6),
            release_z,
        ]
        step_orientation_goal, step_orientation_axis = _arrangement_object_orientation(
            obj,
            orientation_axis=orientation_axis,
            scene_dir=scene_dir,
        )
        high_position = list(release_position)
        high_position[2] = round(
            high_position[2]
            + _arrangement_staging_z_delta_for_goal(step_orientation_goal),
            6,
        )
        steps.append(
            _ArrangementLineStepSpec(
                source_uid=source_uid,
                runtime_uid=runtime_uids[source_uid],
                slot_index=slot_index,
                active_side=_arrangement_arm_side_for_motion(
                    _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0])),
                    target_xy,
                ),
                target_xy=[
                    round(float(target_xy[0]), 6),
                    round(float(target_xy[1]), 6),
                ],
                release_position=release_position,
                high_position=high_position,
                size_score=_arrangement_object_size_score(obj, scene_dir=scene_dir),
                color=_object_color(source_uid, object_attributes),
                orientation_goal=step_orientation_goal,
                orientation_axis=step_orientation_axis,
            )
        )

    summary = str(response.get("task_prompt_summary", "")).strip()
    if not summary:
        summary = "Arrange the selected objects in one left-to-right line."
    notes = str(response.get("basic_background_notes", "")).strip()

    return _ArrangementLineSpec(
        table_source_uid=table_source_uid,
        task_description=task_description,
        task_prompt_summary=summary,
        basic_background_notes=notes,
        order_by=order_by,
        order_direction=order_direction,
        axis=axis,
        anchor=anchor,
        steps=tuple(steps),
        line_origin_xy=line_origin_xy,
        spacing=spacing,
        layout_clearance=_LAYOUT_CLEARANCE,
    )


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
    table_obj: _SceneObject,
    objects: Sequence[_SceneObject],
    count: int,
    spacing: float,
    line_axis: str,
    scene_dir: Path,
    clearance: float,
    ignore_self_initial_overlap: bool = False,
    hard_obstacle_objects: Sequence[_SceneObject] = (),
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
    def __init__(
        self,
        *,
        xy_bounds: tuple[list[float], list[float]],
        half_extent: float,
    ) -> None:
        self.xy_bounds = xy_bounds
        self.half_extent = half_extent


def _arrangement_object_footprint(
    obj: _SceneObject,
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
    obj: _SceneObject,
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
    obj: _SceneObject,
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


def _with_arrangement_generated_z_targets(
    spec: _ArrangementLineSpec,
    gym_config: Mapping[str, Any],
) -> _ArrangementLineSpec:
    return _with_arrangement_generated_pose_targets(spec, gym_config)


def _with_arrangement_generated_pose_targets(
    spec: _ArrangementLineSpec,
    gym_config: Mapping[str, Any],
) -> _ArrangementLineSpec:
    table_config = _generated_table_config(gym_config, spec.table_source_uid)
    rigid_configs = _generated_rigid_object_configs(gym_config)
    if table_config is None:
        return _with_arrangement_generated_z_targets_fallback(spec, gym_config)

    generated_objects = []
    for step in spec.steps:
        config = rigid_configs.get(step.runtime_uid)
        if config is None:
            return _with_arrangement_generated_z_targets_fallback(spec, gym_config)
        generated_objects.append(
            _SceneObject(
                source_uid=step.runtime_uid,
                source_role="rigid_object",
                config=dict(config),
            )
        )

    table_obj = _SceneObject(
        source_uid=str(table_config.get("uid", spec.table_source_uid)),
        source_role="background",
        config=dict(table_config),
    )
    anchor_xy = _generated_table_anchor_xy(table_config, spec.line_origin_xy)
    spacing = _arrangement_spacing(generated_objects, scene_dir=Path("."))
    moved_runtime_uids = {step.runtime_uid for step in spec.steps}
    hard_obstacle_objects = _generated_arrangement_hard_obstacles(
        gym_config,
        moved_runtime_uids=moved_runtime_uids,
        table_source_uid=spec.table_source_uid,
    )
    slots, line_origin_xy = _arrangement_collision_aware_line_slots(
        anchor_xy=anchor_xy,
        table_obj=table_obj,
        objects=generated_objects,
        count=len(spec.steps),
        spacing=spacing,
        line_axis=spec.axis,
        scene_dir=Path("."),
        clearance=spec.layout_clearance,
        ignore_self_initial_overlap=True,
        hard_obstacle_objects=hard_obstacle_objects,
    )
    table_top_z = _generated_table_top_z(table_config)
    orientation_axis = _arrangement_orientation_axis(
        spec.axis,
        table_bounds=_source_object_xy_bounds(table_obj, scene_dir=Path(".")),
    )

    steps = []
    for step, target_xy in zip(spec.steps, slots):
        config = rigid_configs[step.runtime_uid]
        release_z = _generated_release_z(config, table_top_z)
        release_position = [
            round(float(target_xy[0]), 6),
            round(float(target_xy[1]), 6),
            release_z,
        ]
        step_orientation_goal, step_orientation_axis = _arrangement_config_orientation(
            config,
            orientation_axis=orientation_axis,
        )
        high_position = list(release_position)
        high_position[2] = round(
            high_position[2]
            + _arrangement_staging_z_delta_for_goal(step_orientation_goal),
            6,
        )
        steps.append(
            replace(
                step,
                active_side=_arrangement_arm_side_for_motion(
                    _clean_vector3(config.get("init_pos", [0.0, 0.0, 0.0])),
                    target_xy,
                ),
                target_xy=[
                    round(float(target_xy[0]), 6),
                    round(float(target_xy[1]), 6),
                ],
                orientation_goal=step_orientation_goal,
                orientation_axis=step_orientation_axis,
                release_position=release_position,
                high_position=high_position,
                size_score=_arrangement_object_size_score(
                    _SceneObject(
                        source_uid=step.runtime_uid,
                        source_role="rigid_object",
                        config=dict(config),
                    ),
                    scene_dir=Path("."),
                ),
            )
        )
    return replace(
        spec,
        steps=tuple(steps),
        line_origin_xy=line_origin_xy,
        spacing=spacing,
    )


def _with_arrangement_generated_z_targets_fallback(
    spec: _ArrangementLineSpec,
    gym_config: Mapping[str, Any],
) -> _ArrangementLineSpec:
    init_z_by_uid = {
        str(obj.get("uid")): _clean_vector3(obj.get("init_pos", [0.0, 0.0, 0.0]))[2]
        for obj in gym_config.get("rigid_object", [])
        if isinstance(obj, Mapping) and obj.get("uid") is not None
    }
    steps = []
    for step in spec.steps:
        init_z = init_z_by_uid.get(step.runtime_uid)
        if init_z is None:
            steps.append(step)
            continue
        release_position = [
            float(step.target_xy[0]),
            float(step.target_xy[1]),
            round(float(init_z) + _DEFAULT_RELEASE_Z, 6),
        ]
        high_position = list(release_position)
        high_position[2] = round(
            high_position[2]
            + _arrangement_staging_z_delta_for_goal(step.orientation_goal),
            6,
        )
        steps.append(
            replace(
                step,
                release_position=release_position,
                high_position=high_position,
            )
        )
    return replace(spec, steps=tuple(steps))


def _generated_table_config(
    gym_config: Mapping[str, Any],
    table_source_uid: str,
) -> Mapping[str, Any] | None:
    object_configs = {
        str(obj.get("uid")): obj
        for obj in _iter_generated_scene_object_configs(gym_config)
        if isinstance(obj, Mapping) and obj.get("uid") is not None
    }
    return object_configs.get("table") or object_configs.get(table_source_uid)


def _generated_rigid_object_configs(
    gym_config: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(obj.get("uid")): obj
        for obj in gym_config.get("rigid_object", [])
        if isinstance(obj, Mapping) and obj.get("uid") is not None
    }


def _generated_table_anchor_xy(
    table_config: Mapping[str, Any],
    fallback_xy: Sequence[float],
) -> list[float]:
    center = _mesh_config_world_xy_center(table_config)
    if center is not None:
        return center
    try:
        init_pos = _clean_vector3(table_config.get("init_pos", [0.0, 0.0, 0.0]))
        return [round(float(init_pos[0]), 6), round(float(init_pos[1]), 6)]
    except ValueError:
        pass
    return [round(float(fallback_xy[0]), 6), round(float(fallback_xy[1]), 6)]


def _generated_table_top_z(table_config: Mapping[str, Any]) -> float | None:
    z_bounds = _mesh_config_world_z_bounds(table_config)
    if z_bounds is None:
        return None
    return float(z_bounds[1])


def _generated_release_z(
    object_config: Mapping[str, Any],
    table_top_z: float | None,
) -> float:
    if table_top_z is not None:
        local_zmin = _mesh_config_local_zmin_after_rotation(object_config)
        if local_zmin is not None:
            return round(
                float(table_top_z) + _TABLETOP_OBJECT_CLEARANCE - float(local_zmin),
                6,
            )
    init_pos = object_config.get("init_pos")
    if isinstance(init_pos, Sequence) and len(init_pos) == 3:
        return round(float(init_pos[2]) + _DEFAULT_RELEASE_Z, 6)
    return _DEFAULT_RELEASE_Z


def _arrangement_staging_z_delta_for_goal(orientation_goal: str) -> float:
    if orientation_goal != "preserve":
        return _POSE_SENSITIVE_STAGING_Z_DELTA
    return _DEFAULT_STAGING_Z_DELTA


def _arrangement_arm_side_for_motion(
    init_position: Sequence[float],
    target_xy: Sequence[float],
) -> str:
    motion_midpoint = [
        0.5 * (float(init_position[0]) + float(target_xy[0])),
        0.5 * (float(init_position[1]) + float(target_xy[1])),
        float(init_position[2]) if len(init_position) >= 3 else 0.0,
    ]
    return _arm_side_for_position(motion_midpoint)


def _resolve_arrangement_object_uids(
    value: Any,
    rigid_by_uid: Mapping[str, _SceneObject],
) -> list[str]:
    values = _string_list(value)
    if not values:
        raise ValueError("Arrangement response requires non-empty objects.")

    resolved = []
    for raw_value in values:
        resolved.append(
            _resolve_rigid_uid(raw_value, rigid_by_uid, field_name="objects")
        )
    if len(resolved) != len(set(resolved)):
        raise ValueError("Arrangement objects must be distinct.")
    return resolved


def _resolve_rigid_uid(
    value: str,
    rigid_by_uid: Mapping[str, _SceneObject],
    *,
    field_name: str,
) -> str:
    if value in rigid_by_uid:
        return value
    normalized = _normalize_runtime_uid(value)
    matches = [
        source_uid
        for source_uid, obj in rigid_by_uid.items()
        if _normalize_runtime_uid(source_uid) == normalized
        or _base_name(obj) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"LLM returned unknown arrangement {field_name}: {value!r}.")
    raise ValueError(
        f"LLM returned ambiguous arrangement {field_name}: {value!r}; "
        f"candidates: {matches}."
    )


def _normalize_order_by(value: Any) -> str:
    text = str(value or "explicit").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "largest": "size",
        "smallest": "size",
        "big_to_small": "size",
        "large_to_small": "size",
        "color_sequence": "color",
        "given_order": "explicit",
    }
    text = aliases.get(text, text)
    if text not in _SUPPORTED_ORDER_BY:
        raise ValueError(
            f"Unsupported arrangement order_by {value!r}; expected one of "
            f"{sorted(_SUPPORTED_ORDER_BY)}."
        )
    return text


def _normalize_order_direction(value: Any) -> str:
    text = str(value or "given").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "large_to_small": "descending",
        "largest_first": "descending",
        "big_to_small": "descending",
        "small_to_large": "ascending",
        "smallest_first": "ascending",
        "increasing": "ascending",
        "decreasing": "descending",
    }
    text = aliases.get(text, text)
    if text not in _SUPPORTED_ORDER_DIRECTIONS:
        raise ValueError(
            f"Unsupported arrangement order_direction {value!r}; expected one of "
            f"{sorted(_SUPPORTED_ORDER_DIRECTIONS)}."
        )
    return text


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
    scene_objects: Sequence[_SceneObject],
    *,
    selected_source_uids: set[str],
    table_source_uid: str,
) -> list[_SceneObject]:
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
) -> list[_SceneObject]:
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
            _SceneObject(
                source_uid=runtime_uid,
                source_role="background",
                config=dict(config),
            )
        )
    return obstacles


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


def _object_attributes(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        return {}
    attributes: dict[str, dict[str, str]] = {}
    for source_uid, raw_attrs in value.items():
        if not isinstance(raw_attrs, Mapping):
            continue
        attributes[str(source_uid)] = {
            str(key): str(attr_value).strip().lower()
            for key, attr_value in raw_attrs.items()
            if str(attr_value).strip()
        }
    return attributes


def _order_uids_by_size(
    source_uids: list[str],
    *,
    rigid_by_uid: Mapping[str, _SceneObject],
    scene_dir: Path,
    descending: bool,
) -> list[str]:
    return sorted(
        source_uids,
        key=lambda uid: (
            _arrangement_object_size_score(rigid_by_uid[uid], scene_dir=scene_dir)
            or 0.0
        ),
        reverse=descending,
    )


def _order_uids_by_color(
    source_uids: list[str],
    *,
    rigid_by_uid: Mapping[str, _SceneObject],
    object_attributes: Mapping[str, Mapping[str, str]],
    ordered_colors: list[str],
) -> list[str]:
    if not ordered_colors:
        raise ValueError("Color arrangement requires ordered_attributes colors.")
    color_rank = {
        color.strip().lower(): index for index, color in enumerate(ordered_colors)
    }
    missing = []
    ranked: list[tuple[int, str]] = []
    for source_uid in source_uids:
        color = _object_color(source_uid, object_attributes) or _color_hint_for_object(
            rigid_by_uid[source_uid]
        )
        if color is None or color not in color_rank:
            missing.append(source_uid)
            continue
        ranked.append((color_rank[color], source_uid))
    if missing:
        raise ValueError(
            "Color arrangement requires colors for every object; missing or "
            f"unranked: {missing}."
        )
    return [source_uid for _, source_uid in sorted(ranked)]


def _object_color(
    source_uid: str,
    object_attributes: Mapping[str, Mapping[str, str]],
) -> str | None:
    attrs = object_attributes.get(source_uid, {})
    color = attrs.get("color")
    return color.strip().lower() if isinstance(color, str) and color.strip() else None


def _color_hint_for_object(obj: _SceneObject) -> str | None:
    text = (
        f"{obj.source_uid} {obj.config.get('description', '')} "
        f"{obj.config.get('shape', {}).get('fpath', '')}"
    ).lower()
    color_aliases = {
        "red": ("red", "红"),
        "green": ("green", "绿"),
        "blue": ("blue", "蓝"),
        "yellow": ("yellow", "黄"),
        "orange": ("orange", "橙"),
        "purple": ("purple", "紫"),
        "black": ("black", "黑"),
        "white": ("white", "白"),
    }
    for canonical, aliases in color_aliases.items():
        if any(alias in text for alias in aliases):
            return canonical
    return None


def _arrangement_runtime_uid_mapping(
    rigid_objects: Sequence[_SceneObject],
) -> dict[str, str]:
    candidates = {obj.source_uid: _base_name(obj) for obj in rigid_objects}
    counts: dict[str, int] = {}
    for runtime_uid in candidates.values():
        counts[runtime_uid] = counts.get(runtime_uid, 0) + 1
    return {
        source_uid: (
            runtime_uid
            if counts[runtime_uid] == 1
            else _normalize_runtime_uid(source_uid)
        )
        for source_uid, runtime_uid in candidates.items()
    }


def _table_anchor_xy(
    table_obj: _SceneObject,
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
    objects: Sequence[_SceneObject],
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
    obj: _SceneObject,
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
    obj: _SceneObject,
    *,
    scene_dir: Path,
) -> float | None:
    config = _resolved_mesh_config(obj, scene_dir=scene_dir)
    extents = _mesh_config_world_xy_extents(config)
    if extents is None:
        return None
    return max(extents)


def _release_z_for_object(obj: _SceneObject) -> float:
    init_pos = obj.config.get("init_pos")
    if isinstance(init_pos, Sequence) and len(init_pos) == 3:
        return round(float(init_pos[2]) + _DEFAULT_RELEASE_Z, 6)
    return _DEFAULT_RELEASE_Z


def _source_mesh_world_bounds(
    obj: _SceneObject,
    *,
    scene_dir: Path,
) -> tuple[list[float], list[float]] | None:
    config = _resolved_mesh_config(obj, scene_dir=scene_dir)
    z_bounds = _mesh_config_world_z_bounds(config)
    xy_extents = _mesh_config_world_xy_extents(config)
    if z_bounds is None or xy_extents is None:
        return None
    return [0.0, 0.0, z_bounds[0]], [xy_extents[0], xy_extents[1], z_bounds[1]]


def _resolved_mesh_config(
    obj: _SceneObject,
    *,
    scene_dir: Path,
) -> dict[str, Any]:
    config = dict(obj.config)
    shape = dict(config.get("shape", {}) or {})
    fpath = shape.get("fpath")
    if isinstance(fpath, str):
        raw_path = Path(fpath)
        if not raw_path.is_absolute():
            shape["fpath"] = (scene_dir / raw_path).resolve().as_posix()
        config["shape"] = shape
    return config
