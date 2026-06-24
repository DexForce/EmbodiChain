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
    _clean_vector3,
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
_DEFAULT_RELEASE_Z = 0.12
_DEFAULT_STAGING_Z_DELTA = 0.10
_SLOT_MARGIN = 0.01
_MIN_SLOT_SPACING = 0.07
_MAX_SLOT_SPACING = 0.12
_SUPPORTED_ORDER_BY = {"size", "color", "explicit"}
_SUPPORTED_ORDER_DIRECTIONS = {"ascending", "descending", "given"}
_SUPPORTED_AXES = {"left_to_right"}


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
        '  "line_axis": "left_to_right",\n'
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
        "- Use line_axis='left_to_right' for left-to-right tabletop rows.\n"
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

    anchor_xy = _table_anchor_xy(table_obj, anchor)
    spacing = _arrangement_spacing(
        [rigid_by_uid[uid] for uid in object_source_uids],
        scene_dir=scene_dir,
    )
    slots = _arrangement_line_slot_positions(
        anchor_xy=anchor_xy,
        count=len(ordered_source_uids),
        spacing=spacing,
        line_axis=axis,
    )

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
        high_position = list(release_position)
        high_position[2] = round(high_position[2] + _DEFAULT_STAGING_Z_DELTA, 6)
        steps.append(
            _ArrangementLineStepSpec(
                source_uid=source_uid,
                runtime_uid=runtime_uids[source_uid],
                slot_index=slot_index,
                active_side=_arm_side_for_position(
                    _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0]))
                ),
                target_xy=[
                    round(float(target_xy[0]), 6),
                    round(float(target_xy[1]), 6),
                ],
                release_position=release_position,
                high_position=high_position,
                size_score=_arrangement_object_size_score(obj, scene_dir=scene_dir),
                color=_object_color(source_uid, object_attributes),
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
    )


def _arrangement_line_slot_positions(
    *,
    anchor_xy: Sequence[float],
    count: int,
    spacing: float,
    line_axis: str,
) -> list[list[float]]:
    if count < 1:
        raise ValueError("Arrangement line requires at least one slot.")
    axis = _normalize_axis(line_axis)
    anchor = [float(anchor_xy[0]), float(anchor_xy[1])]
    center = (count - 1) / 2.0
    slots: list[list[float]] = []
    for index in range(count):
        axis_offset = (index - center) * float(spacing)
        if axis == "left_to_right":
            slots.append(
                [
                    round(anchor[0], 6),
                    round(anchor[1] + axis_offset, 6),
                ]
            )
            continue
        raise ValueError(f"Unsupported arrangement line axis: {line_axis!r}.")
    return slots


def _with_arrangement_generated_z_targets(
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
        high_position[2] = round(high_position[2] + _DEFAULT_STAGING_Z_DELTA, 6)
        steps.append(
            replace(
                step,
                release_position=release_position,
                high_position=high_position,
            )
        )
    return replace(spec, steps=tuple(steps))


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
        str(value or "left_to_right")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    aliases = {
        "left_right": "left_to_right",
        "robot_left_to_right": "left_to_right",
        "y": "left_to_right",
        "world_y": "left_to_right",
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


def _table_anchor_xy(table_obj: _SceneObject, anchor: str) -> list[float]:
    _normalize_anchor(anchor)
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
    spacing = min(spacing, _MAX_SLOT_SPACING)
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
