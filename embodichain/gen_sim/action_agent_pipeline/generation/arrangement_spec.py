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

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_execution import (
    _arrangement_arm_side_for_motion,
    _arrangement_plan_execution,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_intent import (
    _arrangement_object_categories,
    _normalize_anchor,
    _normalize_order_by,
    _normalize_order_direction,
    _object_color,
    _order_uids_by_color,
    _order_uids_by_size,
    _resolve_arrangement_object_uids,
    _validated_arrangement_order,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_layout import (
    _arrangement_collision_aware_line_slots,
    _arrangement_config_orientation,
    _arrangement_hard_obstacle_objects,
    _arrangement_line_slot_positions,
    _arrangement_object_orientation,
    _arrangement_object_size_score,
    _arrangement_orientation_axis,
    _arrangement_spacing,
    _generated_arrangement_hard_obstacles,
    _source_object_xy_bounds,
    _table_anchor_xy,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    ArrangementLineSpec,
    ArrangementLineStepSpec,
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _TABLETOP_OBJECT_CLEARANCE,
    _clean_vector3,
    _iter_generated_scene_object_configs,
    _mesh_config_local_zmin_after_rotation,
    _mesh_config_world_xy_center,
    _mesh_config_world_z_bounds,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _string_list,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _pick_table,
)
from embodichain.gen_sim.action_agent_pipeline.generation.spec_llm import (
    request_json_spec,
)
from embodichain.gen_sim.action_agent_pipeline.generation._spec_scene_helpers import (
    make_scene_summary as _make_scene_summary_shared,
    object_attributes as _object_attributes,
    rigid_runtime_uid_mapping as _arrangement_runtime_uid_mapping,
)

__all__ = [
    "_apply_arrangement_task_response",
    "_arrangement_line_slot_positions",
    "_build_arrangement_line_spec_from_response",
    "_build_arrangement_line_spec_with_llm",
    "_call_arrangement_task_llm",
    "_make_arrangement_scene_summary",
    "_with_arrangement_generated_pose_targets",
]

_DEFAULTS = defaults_section("arrangement")
_DEFAULT_RELEASE_Z = float(_DEFAULTS["release_z"])
_LAYOUT_CLEARANCE = float(_DEFAULTS["layout_clearance"])
_TRANSPORT_CLEARANCE = float(_DEFAULTS["transport_clearance"])
_PICKUP_MIN_LIFT_HEIGHT = float(_DEFAULTS["pickup_min_lift_height"])


def _build_arrangement_line_spec_with_llm(
    *,
    scene_objects: list[SceneObject],
    project_name: str,
    scene_dir: Path,
    task_description: str,
    model: str | None,
    task_llm_caller: Callable[..., Mapping[str, Any]] | None = None,
) -> ArrangementLineSpec:
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

    _pick_table(background_objects)
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
    return _build_arrangement_line_spec_from_response(
        response=response,
        scene_objects=scene_objects,
        scene_dir=scene_dir,
        task_description=task_description,
    )


def _build_arrangement_line_spec_from_response(
    *,
    response: Mapping[str, Any],
    scene_objects: list[SceneObject],
    scene_dir: Path,
    task_description: str,
) -> ArrangementLineSpec:
    """Build a deterministic arrangement spec from parsed model semantics."""
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
    # Model-facing instructions live in a reviewable text asset. Slot geometry
    # and collision-aware scheduling intentionally remain deterministic code.
    return request_json_spec(
        template_name="arrangement_spec.txt",
        usage_stage="config_generation.arrangement_task",
        project_name=project_name,
        task_description=task_description,
        scene_summary=scene_summary,
        model=model,
    )


def _make_arrangement_scene_summary(
    scene_objects: Sequence[SceneObject],
    *,
    scene_dir: Path,
) -> list[dict[str, Any]]:
    return _make_scene_summary_shared(
        scene_objects,
        scene_dir=scene_dir,
        size_score_fn=_arrangement_object_size_score,
    )


def _apply_arrangement_task_response(
    *,
    response: Mapping[str, Any],
    table_source_uid: str,
    scene_objects: list[SceneObject],
    rigid_objects: list[SceneObject],
    scene_dir: Path,
    task_description: str,
    check_static_obstacles: bool = False,
) -> ArrangementLineSpec:
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    table_obj = by_uid[table_source_uid]
    rigid_by_uid = {obj.source_uid: obj for obj in rigid_objects}
    runtime_uids = _arrangement_runtime_uid_mapping(rigid_objects)

    object_source_uids = _resolve_arrangement_object_uids(
        response.get("objects"),
        rigid_by_uid,
    )
    category_by_uid = _arrangement_object_categories(
        response.get("object_categories"),
        object_source_uids=object_source_uids,
        rigid_by_uid=rigid_by_uid,
    )
    category_order = tuple(_string_list(response.get("category_order")))
    if not category_order:
        category_order = tuple(
            dict.fromkeys(category_by_uid[uid] for uid in object_source_uids)
        )
    unknown_categories = set(category_by_uid.values()) - set(category_order)
    if unknown_categories:
        raise ValueError(
            "Arrangement category_order is missing selected categories: "
            f"{sorted(unknown_categories)}."
        )
    object_source_uids = [
        uid
        for category in category_order
        for uid in object_source_uids
        if category_by_uid[uid] == category
    ]
    object_attributes = _object_attributes(response.get("object_attributes"))
    order_by = _normalize_order_by(response.get("order_by"))
    order_direction = _normalize_order_direction(response.get("order_direction"))
    order_by, order_direction = _validated_arrangement_order(
        order_by,
        order_direction,
        task_description=task_description,
    )
    axis = "world_y"
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
    if check_static_obstacles:
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
            hard_obstacle_objects=_arrangement_hard_obstacle_objects(
                scene_objects,
                selected_source_uids=set(object_source_uids),
                table_source_uid=table_source_uid,
            ),
        )
    else:
        # Source-frame slots are placeholders. The safe layout is generated
        # after scaling, baking, and scene rotation are complete.
        slots = _arrangement_line_slot_positions(
            anchor_xy=anchor_xy,
            count=len(ordered_source_uids),
            spacing=spacing,
            line_axis=axis,
            table_bounds=table_bounds,
        )
        line_origin_xy = list(anchor_xy)
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
        init_position = _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0]))
        high_position = list(release_position)
        high_position[2] = _arrangement_transport_height(
            initial_z=init_position[2],
            release_z=release_position[2],
        )
        steps.append(
            ArrangementLineStepSpec(
                source_uid=source_uid,
                runtime_uid=runtime_uids[source_uid],
                slot_index=slot_index,
                active_side=_arrangement_arm_side_for_motion(
                    init_position,
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
                category=category_by_uid[source_uid],
            )
        )

    summary = str(response.get("task_prompt_summary", "")).strip()
    if not summary:
        summary = "Arrange the selected objects in one left-to-right line."
    notes = str(response.get("basic_background_notes", "")).strip()

    return ArrangementLineSpec(
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
        category_order=category_order,
        semantic_order=tuple(runtime_uids[uid] for uid in ordered_source_uids),
    )


def _with_arrangement_generated_pose_targets(
    spec: ArrangementLineSpec,
    gym_config: Mapping[str, Any],
) -> ArrangementLineSpec:
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
            SceneObject(
                source_uid=step.runtime_uid,
                source_role="rigid_object",
                config=dict(config),
            )
        )

    table_obj = SceneObject(
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

    planned = _arrangement_plan_execution(
        spec,
        slots,
        generated_objects=generated_objects,
        rigid_configs=rigid_configs,
    )
    if planned is None:
        raise ValueError(
            "Unable to generate a feasible arrangement: every candidate either "
            "assigns an object's pickup arm to a forbidden outer slot or creates "
            "cyclic initial slot occupancy."
        )
    spatial_direction, planned_steps = planned

    steps = []
    for step in planned_steps:
        target_xy = step.target_xy
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
        init_position = _clean_vector3(config.get("init_pos", [0.0, 0.0, 0.0]))
        high_position = list(release_position)
        high_position[2] = _arrangement_transport_height(
            initial_z=init_position[2],
            release_z=release_position[2],
        )
        steps.append(
            replace(
                step,
                target_xy=[
                    round(float(target_xy[0]), 6),
                    round(float(target_xy[1]), 6),
                ],
                orientation_goal=step_orientation_goal,
                orientation_axis=step_orientation_axis,
                release_position=release_position,
                high_position=high_position,
                size_score=_arrangement_object_size_score(
                    SceneObject(
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
        axis="world_y",
        spatial_direction=spatial_direction,
        category_order=(
            () if spatial_direction == "initial_side_order" else spec.category_order
        ),
    )


def _with_arrangement_generated_z_targets_fallback(
    spec: ArrangementLineSpec,
    gym_config: Mapping[str, Any],
) -> ArrangementLineSpec:
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
        high_position[2] = _arrangement_transport_height(
            initial_z=init_z,
            release_z=release_position[2],
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


def _arrangement_transport_height(*, initial_z: float, release_z: float) -> float:
    return round(
        max(
            float(initial_z) + _PICKUP_MIN_LIFT_HEIGHT,
            float(release_z) + _TRANSPORT_CLEARANCE,
        ),
        6,
    )


def _release_z_for_object(obj: SceneObject) -> float:
    init_pos = obj.config.get("init_pos")
    if isinstance(init_pos, Sequence) and len(init_pos) == 3:
        return round(float(init_pos[2]) + _DEFAULT_RELEASE_Z, 6)
    return _DEFAULT_RELEASE_Z
