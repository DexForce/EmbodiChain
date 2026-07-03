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
    _SceneObject,
    _StackingSpec,
    _StackingStepSpec,
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
    _base_name,
    _normalize_runtime_uid,
    _string_list,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _arm_side_for_position,
    _pick_table,
)

__all__ = [
    "_build_stacking_spec_with_llm",
    "_call_stacking_task_llm",
    "_is_stacking_task_description",
    "_make_stacking_summary",
    "_with_stacking_generated_targets",
]

_STACKING_KEYWORDS = (
    "stack",
    "stacking",
    "pile",
    "叠",
    "叠放",
    "堆叠",
    "摞",
)
_SUPPORTED_STACK_MODES = {"on_top", "nested"}
_SUPPORTED_ORDER_BY = {"explicit", "size"}
_STACKING_ANCHOR = "table_center"
_STAGING_Z_DELTA = 0.10
_STACK_CLEARANCE = 0.003


def _is_stacking_task_description(task_description: str) -> bool:
    text = task_description.strip().lower()
    return any(keyword in text for keyword in _STACKING_KEYWORDS)


def _build_stacking_spec_with_llm(
    *,
    scene_objects: list[_SceneObject],
    project_name: str,
    scene_dir: Path,
    task_description: str,
    model: str | None,
    task_llm_caller: Callable[..., Mapping[str, Any]] | None = None,
) -> _StackingSpec:
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
    from langchain_core.messages import HumanMessage, SystemMessage

    from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import (
        extract_json_object,
    )
    from embodichain.gen_sim.action_agent_pipeline.utils.mllm import (
        create_chat_openai,
    )

    prompt = (
        "Parse a tabletop object stacking task and produce one strict "
        "config-level JSON spec. The generator computes all center positions, "
        "heights, robot config, and action graphs deterministically.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "objects": ["<source_uid from rigid_object>", "..."],\n'
        '  "stack_mode": "on_top|nested",\n'
        '  "bottom_to_top": ["<source_uid bottom>", "..."],\n'
        '  "order_by": "explicit|size",\n'
        '  "object_attributes": {"<source_uid>": {"color": "red"}},\n'
        '  "anchor": "table_center",\n'
        '  "task_prompt_summary": "<short execution summary>",\n'
        '  "basic_background_notes": "<short notes>"\n'
        "}\n\n"
        "Rules:\n"
        "- Use only source_uid values from rigid_object scene items.\n"
        "- Include every object that must be stacked.\n"
        "- Use stack_mode='on_top' for blocks, cubes, books, and solid objects "
        "that should be vertically stacked.\n"
        "- Use stack_mode='nested' for bowls or cup-like containers that should "
        "be nested into each other.\n"
        "- For explicit statements like blue on green and green on red, return "
        "bottom_to_top=[red, green, blue] and order_by='explicit'.\n"
        "- If no order is specified for nested bowls, return order_by='size' "
        "and leave bottom_to_top empty; the generator sorts large-to-small.\n"
        "- Use anchor='table_center'. Do not return target positions, robot "
        "config, success JSON, or action graphs.\n\n"
        f"Project: {project_name}\n"
        f"Task description:\n{task_description}\n"
        f"Scene objects:\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}"
    )
    llm = create_chat_openai(
        temperature=0.0,
        model=model,
        usage_stage="config_generation.stacking_task",
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


def _make_stacking_scene_summary(
    scene_objects: Sequence[_SceneObject],
    *,
    scene_dir: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "source_uid": obj.source_uid,
            "role": obj.source_role,
            "object_type": _base_name(obj),
            "mesh": obj.config.get("shape", {}).get("fpath"),
            "init_pos": obj.config.get("init_pos"),
            "body_scale": obj.config.get("body_scale"),
            "color_hint": _color_hint_for_object(obj),
            "size_score": _stacking_object_size_score(obj, scene_dir=scene_dir),
        }
        for obj in scene_objects
    ]


def _apply_stacking_task_response(
    *,
    response: Mapping[str, Any],
    table_source_uid: str,
    scene_objects: list[_SceneObject],
    rigid_objects: list[_SceneObject],
    scene_dir: Path,
    task_description: str,
) -> _StackingSpec:
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    table_obj = by_uid[table_source_uid]
    rigid_by_uid = {obj.source_uid: obj for obj in rigid_objects}
    runtime_uids = _stacking_runtime_uid_mapping(rigid_objects)

    object_source_uids = _resolve_stacking_object_uids(
        response.get("objects"), rigid_by_uid
    )
    stack_mode = _normalize_stack_mode(response.get("stack_mode"))
    order_by = _normalize_order_by(response.get("order_by"))
    anchor = _normalize_anchor(response.get("anchor"))
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

    anchor_xy = _table_anchor_xy(table_obj, anchor, scene_dir=scene_dir)
    steps = []
    for layer_index, source_uid in enumerate(ordered_source_uids):
        obj = rigid_by_uid[source_uid]
        orientation_goal = "axis_align" if stack_mode == "on_top" else "preserve"
        orientation_axis = "x" if stack_mode == "on_top" else "none"
        steps.append(
            _StackingStepSpec(
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
                    else None
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

    return _StackingSpec(
        table_source_uid=table_source_uid,
        task_description=task_description,
        task_prompt_summary=summary,
        basic_background_notes=notes,
        stack_mode=stack_mode,
        order_by=order_by,
        anchor=anchor,
        anchor_xy=anchor_xy,
        steps=tuple(steps),
    )


def _with_stacking_generated_targets(
    spec: _StackingSpec,
    gym_config: Mapping[str, Any],
) -> _StackingSpec:
    object_configs = {
        str(obj.get("uid")): obj
        for obj in _iter_generated_scene_object_configs(gym_config)
        if obj.get("uid") is not None
    }
    table_config = object_configs.get("table") or object_configs.get(
        spec.table_source_uid
    )
    anchor_xy = _generated_stacking_anchor_xy(table_config, spec.anchor_xy)
    table_top_z = _generated_table_top_z(table_config)
    z_by_runtime_uid: dict[str, float] = {}
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

        if step.layer_index == 0:
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
            support_top_offset = _mesh_config_local_zmax_after_rotation(support_config)
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
        steps.append(
            replace(
                step,
                active_side=_arm_side_for_position(
                    _clean_vector3(moved_config.get("init_pos", [0.0, 0.0, 0.0]))
                ),
                target_position=target_position,
                high_position=high_position,
            )
        )
    return replace(spec, anchor_xy=anchor_xy, steps=tuple(steps))


def _generated_stacking_anchor_xy(
    table_config: Mapping[str, Any] | None,
    fallback_xy: Sequence[float],
) -> list[float]:
    if table_config is not None:
        center = _mesh_config_world_xy_center(table_config)
        if center is not None:
            return center
        init_pos = _clean_vector3(table_config.get("init_pos", [0.0, 0.0, 0.0]))
        return [round(init_pos[0], 6), round(init_pos[1], 6)]
    return [round(float(fallback_xy[0]), 6), round(float(fallback_xy[1]), 6)]


def _generated_table_top_z(
    table_config: Mapping[str, Any] | None,
) -> float | None:
    if table_config is None:
        return None
    z_bounds = _mesh_config_world_z_bounds(table_config)
    if z_bounds is None:
        return None
    return float(z_bounds[1])


def _make_stacking_summary(spec: _StackingSpec) -> dict[str, Any]:
    return {
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


def _mesh_config_local_zmax_after_rotation(
    obj_config: Mapping[str, Any],
) -> float | None:
    z_bounds = _mesh_config_world_z_bounds({**obj_config, "init_pos": [0.0, 0.0, 0.0]})
    if z_bounds is None:
        return None
    return z_bounds[1]


def _resolve_stacking_object_uids(
    value: Any,
    rigid_by_uid: Mapping[str, _SceneObject],
) -> list[str]:
    values = _string_list(value)
    if not values:
        raise ValueError("Stacking response requires non-empty objects.")
    resolved = [
        _resolve_rigid_uid(raw_value, rigid_by_uid, field_name="objects")
        for raw_value in values
    ]
    if len(resolved) < 2:
        raise ValueError("Stacking requires at least two distinct objects.")
    if len(resolved) != len(set(resolved)):
        raise ValueError("Stacking objects must be distinct.")
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
        raise ValueError(f"LLM returned unknown stacking {field_name}: {value!r}.")
    raise ValueError(
        f"LLM returned ambiguous stacking {field_name}: {value!r}; "
        f"candidates: {matches}."
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


def _normalize_anchor(value: Any) -> str:
    text = str(value or _STACKING_ANCHOR).strip().lower().replace("-", "_")
    aliases = {
        "center": _STACKING_ANCHOR,
        "table_centre": _STACKING_ANCHOR,
        "桌子中央": _STACKING_ANCHOR,
        "桌面中央": _STACKING_ANCHOR,
    }
    text = aliases.get(text, text)
    if text != _STACKING_ANCHOR:
        raise ValueError("Stacking only supports anchor='table_center'.")
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


def _object_color(
    source_uid: str,
    object_attributes: Mapping[str, Mapping[str, str]],
) -> str | None:
    attrs = object_attributes.get(source_uid, {})
    color = attrs.get("color")
    return color.strip().lower() if isinstance(color, str) and color.strip() else None


def _color_hint_for_object(obj: _SceneObject) -> str | None:
    text = (f"{obj.source_uid} {obj.config.get('shape', {}).get('fpath', '')}").lower()
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


def _stacking_runtime_uid_mapping(
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


def _stacking_object_size_score(
    obj: _SceneObject,
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


def _mesh_config_world_xy_extents(
    obj_config: Mapping[str, Any],
) -> tuple[float, float] | None:
    from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
        _mesh_config_world_xy_extents as _shared_mesh_config_world_xy_extents,
    )

    return _shared_mesh_config_world_xy_extents(obj_config)


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
