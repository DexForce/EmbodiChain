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

import math

from embodichain.gen_sim.action_agent_pipeline.defaults import (
    generation_defaults_section,
)

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    SceneObject,
    StackingSpec,
    StackingStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _TABLETOP_OBJECT_CLEARANCE,
    _clean_vector3,
    _iter_generated_scene_object_configs,
    _mesh_config_local_zmin_after_rotation,
    _mesh_config_world_xy_center,
    _mesh_config_world_xy_bounds,
    _mesh_config_world_z_bounds,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _string_list,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _arm_side_for_position,
    _pick_table,
)
from embodichain.gen_sim.action_agent_pipeline.generation.spec_llm import (
    request_json_spec,
)
from embodichain.gen_sim.action_agent_pipeline.generation._spec_scene_helpers import (
    make_scene_summary as _make_scene_summary_shared,
    object_attributes as _object_attributes,
    resolve_rigid_uid as _resolve_rigid_uid_shared,
    resolved_mesh_config as _resolved_mesh_config,
    rigid_runtime_uid_mapping as _stacking_runtime_uid_mapping,
)
from embodichain.utils.logger import log_warning

__all__ = [
    "_build_stacking_spec_with_llm",
    "_call_stacking_task_llm",
    "_make_stacking_summary",
    "_with_stacking_generated_targets",
]

_SUPPORTED_STACK_MODES = {"on_top", "nested"}
_SUPPORTED_ORDER_BY = {"explicit", "size"}
_STACKING_ANCHOR = "table_center"
_OBJECT_STACKING_ANCHOR = "object"
_DEFAULTS = generation_defaults_section("stacking")
_STAGING_Z_DELTA = float(_DEFAULTS["staging_z_delta"])
_STACK_CLEARANCE = float(_DEFAULTS["clearance"])
_NESTED_RELEASE_Z_OFFSET = float(_DEFAULTS["nested_release_z_offset"])
_ANCHOR_OFFSET = float(_DEFAULTS["anchor_offset"])
_ANCHOR_CLEARANCE_RADIUS = float(_DEFAULTS["anchor_clearance_radius"])


def _build_stacking_spec_with_llm(
    *,
    scene_objects: list[SceneObject],
    project_name: str,
    scene_dir: Path,
    task_description: str,
    model: str | None,
    task_llm_caller: Callable[..., Mapping[str, Any]] | None = None,
) -> StackingSpec:
    background_objects = [
        obj for obj in scene_objects if obj.source_role == "background"
    ]
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]
    if not background_objects:
        raise ValueError("Stacking generation requires a background table.")
    if len(rigid_objects) < 2:
        raise ValueError("Stacking generation requires at least two movable objects.")

    table = _pick_table(background_objects)
    scene_summary = _make_stacking_scene_summary(scene_objects, scene_dir=scene_dir)
    if task_llm_caller is None:
        task_llm_caller = _call_stacking_task_llm
    response = task_llm_caller(
        project_name=project_name,
        task_description=task_description,
        scene_summary=scene_summary,
        model=model,
    )
    return _apply_stacking_task_response(
        response=response,
        table_source_uid=table.source_uid,
        scene_objects=scene_objects,
        rigid_objects=rigid_objects,
        scene_dir=scene_dir,
        task_description=task_description,
    )


def _call_stacking_task_llm(
    *,
    project_name: str,
    task_description: str,
    scene_summary: list[dict[str, Any]],
    model: str | None,
) -> dict[str, Any]:
    # The model chooses semantic order and mode only. Stack positions and
    # support heights remain deterministic so runs are reproducible.
    return request_json_spec(
        template_name="stacking_spec.txt",
        usage_stage="config_generation.stacking_task",
        project_name=project_name,
        task_description=task_description,
        scene_summary=scene_summary,
        model=model,
    )


def _make_stacking_scene_summary(
    scene_objects: Sequence[SceneObject],
    *,
    scene_dir: Path,
) -> list[dict[str, Any]]:
    return _make_scene_summary_shared(
        scene_objects,
        scene_dir=scene_dir,
        size_score_fn=_stacking_object_size_score,
    )


def _apply_stacking_task_response(
    *,
    response: Mapping[str, Any],
    table_source_uid: str,
    scene_objects: list[SceneObject],
    rigid_objects: list[SceneObject],
    scene_dir: Path,
    task_description: str,
) -> StackingSpec:
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    table_obj = by_uid[table_source_uid]
    rigid_by_uid = {obj.source_uid: obj for obj in rigid_objects}
    runtime_uids = _stacking_runtime_uid_mapping(rigid_objects)

    anchor, anchor_source_uid = _normalize_anchor(
        response.get("anchor"),
        rigid_by_uid=rigid_by_uid,
    )
    object_source_uids = _resolve_stacking_object_uids(
        response.get("objects"),
        rigid_by_uid,
        min_count=1 if anchor == _OBJECT_STACKING_ANCHOR else 2,
    )
    if anchor_source_uid in object_source_uids:
        raise ValueError("Stacking object anchor must be a passive support object.")
    stack_mode = _normalize_stack_mode(response.get("stack_mode"))
    order_by = _normalize_order_by(response.get("order_by"))
    object_attributes = _object_attributes(response.get("object_attributes"))

    explicit_order = _string_list(response.get("bottom_to_top"))
    if explicit_order:
        ordered_source_uids = [
            _resolve_rigid_uid(uid, rigid_by_uid, field_name="bottom_to_top")
            for uid in explicit_order
        ]
        if set(ordered_source_uids) != set(object_source_uids):
            raise ValueError(
                "Stacking bottom_to_top must contain exactly the stacking objects."
            )
        order_by = "explicit"
    elif order_by == "size":
        ordered_source_uids = sorted(
            object_source_uids,
            key=lambda uid: (
                _stacking_object_size_score(rigid_by_uid[uid], scene_dir=scene_dir)
                or 0.0
            ),
            reverse=True,
        )
    else:
        ordered_source_uids = object_source_uids

    anchor_runtime_uid = (
        runtime_uids[anchor_source_uid] if anchor_source_uid is not None else None
    )
    if anchor_source_uid is None:
        anchor_xy = _table_anchor_xy(table_obj, anchor, scene_dir=scene_dir)
    else:
        anchor_position = _clean_vector3(
            rigid_by_uid[anchor_source_uid].config.get("init_pos", [0.0, 0.0, 0.0])
        )
        anchor_xy = [float(anchor_position[0]), float(anchor_position[1])]
    steps = []
    for layer_index, source_uid in enumerate(ordered_source_uids):
        obj = rigid_by_uid[source_uid]
        orientation_goal, orientation_axis = _stacking_object_orientation(
            obj,
            stack_mode=stack_mode,
            scene_dir=scene_dir,
        )
        steps.append(
            StackingStepSpec(
                source_uid=source_uid,
                runtime_uid=runtime_uids[source_uid],
                layer_index=layer_index,
                active_side=_arm_side_for_position(
                    _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0]))
                ),
                target_position=[float(anchor_xy[0]), float(anchor_xy[1]), 0.0],
                high_position=[float(anchor_xy[0]), float(anchor_xy[1]), 0.0],
                support_runtime_uid=(
                    runtime_uids[ordered_source_uids[layer_index - 1]]
                    if layer_index > 0
                    else anchor_runtime_uid
                ),
                size_score=_stacking_object_size_score(obj, scene_dir=scene_dir),
                color=_object_color(source_uid, object_attributes),
                orientation_goal=orientation_goal,
                orientation_axis=orientation_axis,
            )
        )

    summary = str(response.get("task_prompt_summary", "")).strip()
    if not summary:
        summary = "Move the selected objects to the table center and stack them."
    notes = str(response.get("basic_background_notes", "")).strip()

    return StackingSpec(
        table_source_uid=table_source_uid,
        task_description=task_description,
        task_prompt_summary=summary,
        basic_background_notes=notes,
        stack_mode=stack_mode,
        order_by=order_by,
        anchor=anchor,
        anchor_xy=anchor_xy,
        steps=tuple(steps),
        anchor_source_uid=anchor_source_uid,
        anchor_runtime_uid=anchor_runtime_uid,
    )


def _with_stacking_generated_targets(
    spec: StackingSpec,
    gym_config: Mapping[str, Any],
) -> StackingSpec:
    object_configs = {
        str(obj.get("uid")): obj
        for obj in _iter_generated_scene_object_configs(gym_config)
        if obj.get("uid") is not None
    }
    table_config = object_configs.get("table") or object_configs.get(
        spec.table_source_uid
    )
    anchor_config = (
        object_configs.get(str(spec.anchor_runtime_uid))
        if spec.anchor == _OBJECT_STACKING_ANCHOR
        else None
    )
    if spec.anchor == _OBJECT_STACKING_ANCHOR:
        if anchor_config is None:
            raise ValueError(
                f"Generated stacking config missing object anchor "
                f"{spec.anchor_runtime_uid!r}."
            )
        anchor_position = _clean_vector3(anchor_config.get("init_pos", [0.0, 0.0, 0.0]))
        anchor_xy = [float(anchor_position[0]), float(anchor_position[1])]
    else:
        anchor_xy = _generated_stacking_anchor_xy(
            table_config,
            spec.anchor_xy,
            object_configs=object_configs,
        )
    table_top_z = _generated_table_top_z(table_config)
    z_by_runtime_uid: dict[str, float] = {}
    if spec.anchor_runtime_uid is not None and anchor_config is not None:
        z_by_runtime_uid[spec.anchor_runtime_uid] = _clean_vector3(
            anchor_config.get("init_pos", [0.0, 0.0, 0.0])
        )[2]
    steps = []
    for step in spec.steps:
        moved_config = object_configs.get(step.runtime_uid)
        if moved_config is None:
            steps.append(step)
            continue
        moved_bottom_offset = _mesh_config_local_zmin_after_rotation(moved_config)
        if moved_bottom_offset is None:
            steps.append(step)
            continue

        if step.layer_index == 0 and step.support_runtime_uid is None:
            if table_top_z is None:
                target_z = _clean_vector3(
                    moved_config.get("init_pos", [0.0, 0.0, 0.0])
                )[2]
            else:
                target_z = (
                    float(table_top_z)
                    + _TABLETOP_OBJECT_CLEARANCE
                    - float(moved_bottom_offset)
                )
        else:
            support_uid = step.support_runtime_uid
            support_z = z_by_runtime_uid.get(str(support_uid))
            support_config = object_configs.get(str(support_uid))
            if support_z is None or support_config is None:
                steps.append(step)
                continue
            if spec.stack_mode == "nested":
                target_z = support_z + _NESTED_RELEASE_Z_OFFSET
            else:
                support_top_offset = _mesh_config_local_zmax_after_rotation(
                    support_config
                )
                if support_top_offset is None:
                    steps.append(step)
                    continue
                target_z = (
                    support_z
                    + support_top_offset
                    + _STACK_CLEARANCE
                    - float(moved_bottom_offset)
                )

        target_position = [
            float(anchor_xy[0]),
            float(anchor_xy[1]),
            round(float(target_z), 6),
        ]
        high_position = list(target_position)
        high_position[2] = round(high_position[2] + _STAGING_Z_DELTA, 6)
        z_by_runtime_uid[step.runtime_uid] = target_position[2]
        orientation_goal, orientation_axis = _stacking_config_orientation(
            moved_config,
            stack_mode=spec.stack_mode,
        )
        steps.append(
            replace(
                step,
                active_side=_arm_side_for_position(
                    _clean_vector3(moved_config.get("init_pos", [0.0, 0.0, 0.0]))
                ),
                target_position=target_position,
                high_position=high_position,
                orientation_goal=orientation_goal,
                orientation_axis=orientation_axis,
            )
        )
    return replace(spec, anchor_xy=anchor_xy, steps=tuple(steps))


def _generated_stacking_anchor_xy(
    table_config: Mapping[str, Any] | None,
    fallback_xy: Sequence[float],
    *,
    object_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[float]:
    if table_config is None:
        return [round(float(fallback_xy[0]), 6), round(float(fallback_xy[1]), 6)]

    center = _mesh_config_world_xy_center(table_config)
    if center is None:
        init_pos = _clean_vector3(table_config.get("init_pos", [0.0, 0.0, 0.0]))
        center = [round(init_pos[0], 6), round(init_pos[1], 6)]
    if not object_configs:
        return center

    table_bounds = _mesh_config_world_xy_bounds(table_config)
    robot_front = [1.0, 0.0]
    robot_back = [-1.0, 0.0]
    directions = (
        [0.0, 0.0],
        robot_back,
        robot_front,
    )
    obstacle_bounds = []
    for config in object_configs.values():
        if config is table_config:
            continue
        bounds = _mesh_config_world_xy_bounds(config)
        if bounds is not None:
            obstacle_bounds.append(bounds)

    for direction in directions:
        candidate = [
            round(center[0] + _ANCHOR_OFFSET * direction[0], 6),
            round(center[1] + _ANCHOR_OFFSET * direction[1], 6),
        ]
        if table_bounds is not None and not _xy_point_in_bounds(
            candidate, table_bounds
        ):
            continue
        if any(
            _xy_point_to_bounds_distance(candidate, bounds) < _ANCHOR_CLEARANCE_RADIUS
            for bounds in obstacle_bounds
        ):
            continue
        return candidate
    fallback = [
        round(center[0] + _ANCHOR_OFFSET * robot_back[0], 6),
        round(center[1] + _ANCHOR_OFFSET * robot_back[1], 6),
    ]
    log_warning(
        "No clear stacking anchor found at the table center, back, or front; "
        f"forcing the table-back point {fallback}."
    )
    return fallback


def _xy_point_in_bounds(
    point: Sequence[float],
    bounds: tuple[list[float], list[float]],
) -> bool:
    mins, maxs = bounds
    return float(mins[0]) <= float(point[0]) <= float(maxs[0]) and float(
        mins[1]
    ) <= float(point[1]) <= float(maxs[1])


def _xy_point_to_bounds_distance(
    point: Sequence[float],
    bounds: tuple[list[float], list[float]],
) -> float:
    """Return the shortest XY distance from a point to an axis-aligned bound."""
    mins, maxs = bounds
    dx = max(float(mins[0]) - float(point[0]), 0.0, float(point[0]) - float(maxs[0]))
    dy = max(float(mins[1]) - float(point[1]), 0.0, float(point[1]) - float(maxs[1]))
    return math.hypot(dx, dy)


def _generated_table_top_z(
    table_config: Mapping[str, Any] | None,
) -> float | None:
    if table_config is None:
        return None
    z_bounds = _mesh_config_world_z_bounds(table_config)
    if z_bounds is None:
        return None
    return float(z_bounds[1])


def _make_stacking_summary(spec: StackingSpec) -> dict[str, Any]:
    summary = {
        "mode": "stacking",
        "stack_mode": spec.stack_mode,
        "anchor": spec.anchor,
        "anchor_xy": [float(spec.anchor_xy[0]), float(spec.anchor_xy[1])],
        "order_by": spec.order_by,
        "bottom_to_top": [step.runtime_uid for step in spec.steps],
        "placements": [
            {
                "object": step.runtime_uid,
                "source_uid": step.source_uid,
                "layer_index": step.layer_index,
                "active_arm": f"{step.active_side}_arm",
                "support": step.support_runtime_uid,
                "target_position": [float(value) for value in step.target_position],
                "orientation_goal": step.orientation_goal,
                "orientation_axis": step.orientation_axis,
            }
            for step in spec.steps
        ],
    }
    if spec.anchor_runtime_uid is not None:
        summary["anchor_object"] = spec.anchor_runtime_uid
    return summary


def _mesh_config_local_zmax_after_rotation(
    obj_config: Mapping[str, Any],
) -> float | None:
    z_bounds = _mesh_config_world_z_bounds({**obj_config, "init_pos": [0.0, 0.0, 0.0]})
    if z_bounds is None:
        return None
    return z_bounds[1]


def _resolve_stacking_object_uids(
    value: Any,
    rigid_by_uid: Mapping[str, SceneObject],
    *,
    min_count: int = 2,
) -> list[str]:
    values = _string_list(value)
    if not values:
        raise ValueError("Stacking response requires non-empty objects.")
    resolved = [
        _resolve_rigid_uid(raw_value, rigid_by_uid, field_name="objects")
        for raw_value in values
    ]
    if len(resolved) < min_count:
        raise ValueError(
            f"Stacking requires at least {min_count} distinct moved object(s)."
        )
    if len(resolved) != len(set(resolved)):
        raise ValueError("Stacking objects must be distinct.")
    return resolved


def _resolve_rigid_uid(
    value: str,
    rigid_by_uid: Mapping[str, SceneObject],
    *,
    field_name: str,
) -> str:
    return _resolve_rigid_uid_shared(
        value,
        rigid_by_uid,
        field_name=field_name,
        route_label="stacking",
    )


def _normalize_stack_mode(value: Any) -> str:
    text = str(value or "on_top").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "on": "on_top",
        "vertical": "on_top",
        "nested_bowls": "nested",
        "inside": "nested",
    }
    text = aliases.get(text, text)
    if text not in _SUPPORTED_STACK_MODES:
        raise ValueError(
            f"Unsupported stack_mode {value!r}; expected one of "
            f"{sorted(_SUPPORTED_STACK_MODES)}."
        )
    return text


def _normalize_order_by(value: Any) -> str:
    text = str(value or "explicit").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "given": "explicit",
        "bottom_to_top": "explicit",
        "large_to_small": "size",
        "big_to_small": "size",
    }
    text = aliases.get(text, text)
    if text not in _SUPPORTED_ORDER_BY:
        raise ValueError(
            f"Unsupported stacking order_by {value!r}; expected one of "
            f"{sorted(_SUPPORTED_ORDER_BY)}."
        )
    return text


def _normalize_anchor(
    value: Any,
    *,
    rigid_by_uid: Mapping[str, SceneObject],
) -> tuple[str, str | None]:
    if isinstance(value, Mapping):
        anchor_type = str(value.get("type", "")).strip().lower().replace("-", "_")
        if anchor_type not in {_OBJECT_STACKING_ANCHOR, "support"}:
            raise ValueError("Stacking object anchor requires type='object'.")
        raw_anchor_uid = value.get("object")
        if not isinstance(raw_anchor_uid, str) or not raw_anchor_uid.strip():
            raise ValueError("Stacking object anchor requires a non-empty object.")
        return _OBJECT_STACKING_ANCHOR, _resolve_rigid_uid(
            raw_anchor_uid,
            rigid_by_uid,
            field_name="anchor.object",
        )

    text = str(value or _STACKING_ANCHOR).strip().lower().replace("-", "_")
    aliases = {
        "center": _STACKING_ANCHOR,
        "table_centre": _STACKING_ANCHOR,
        "桌子中央": _STACKING_ANCHOR,
        "桌面中央": _STACKING_ANCHOR,
    }
    text = aliases.get(text, text)
    if text != _STACKING_ANCHOR:
        raise ValueError("Stacking anchor must be 'table_center' or an object anchor.")
    return text, None


def _object_color(
    source_uid: str,
    object_attributes: Mapping[str, Mapping[str, str]],
) -> str | None:
    attrs = object_attributes.get(source_uid, {})
    color = attrs.get("color")
    return color.strip().lower() if isinstance(color, str) and color.strip() else None


def _table_anchor_xy(
    table_obj: SceneObject,
    anchor: str,
    *,
    scene_dir: Path,
) -> list[float]:
    if anchor != _STACKING_ANCHOR:
        raise ValueError("Table stacking requires anchor='table_center'.")
    center = _mesh_config_world_xy_center(
        _resolved_mesh_config(table_obj, scene_dir=scene_dir)
    )
    if center is not None:
        return center
    init_pos = _clean_vector3(table_obj.config.get("init_pos", [0.0, 0.0, 0.0]))
    return [round(init_pos[0], 6), round(init_pos[1], 6)]


def _stacking_object_size_score(
    obj: SceneObject,
    *,
    scene_dir: Path,
) -> float | None:
    config = _resolved_mesh_config(obj, scene_dir=scene_dir)
    bounds = _mesh_config_world_z_bounds(config)
    if bounds is None:
        return None
    xy_extents = _mesh_config_world_xy_extents(config)
    if xy_extents is None:
        return None
    return round(float(max(*xy_extents, bounds[1] - bounds[0])), 6)


def _stacking_object_orientation(
    obj: SceneObject,
    *,
    stack_mode: str,
    scene_dir: Path,
) -> tuple[str, str]:
    return _stacking_config_orientation(
        _resolved_mesh_config(obj, scene_dir=scene_dir),
        stack_mode=stack_mode,
    )


def _stacking_config_orientation(
    obj_config: Mapping[str, Any],
    *,
    stack_mode: str,
) -> tuple[str, str]:
    """Preserve the source orientation unless a future spec requests otherwise."""
    return "preserve", "none"


def _mesh_config_world_xy_extents(
    obj_config: Mapping[str, Any],
) -> tuple[float, float] | None:
    from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
        _mesh_config_world_xy_extents as _shared_mesh_config_world_xy_extents,
    )

    return _shared_mesh_config_world_xy_extents(obj_config)
