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

"""Normalize relative-manipulation intent and shared object identities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.domain.object_semantics import (
    SHORT_BOTTLE_LIKE_KEYWORDS as _SHORT_BOTTLE_LIKE_KEYWORDS,
    SHORT_CUP_LIKE_KEYWORDS as _SHORT_CUP_LIKE_KEYWORDS,
    UPRIGHTABLE_KEYWORDS as _UPRIGHTABLE_KEYWORDS,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.tasks import (
    MANIPULATION_INTENTS as _SUPPORTED_MANIPULATION_INTENTS,
    RELATIVE_RELATIONS as _RELATIVE_RELATIONS,
    SIDE_RELATIONS as _SIDE_RELATIONS,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    RelativePlacementStepSpec,
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _base_name,
    _candidate_relative_runtime_uid,
    _container_runtime_uid,
    _is_container_like,
    _normalize_runtime_uid,
    _target_runtime_suffix,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relation_language import (
    relative_relation_phrase as _canonical_relative_relation_phrase,
)

__all__ = [
    "_DEFAULT_HOVER_HEIGHT",
    "_SIDE_RELATIONS",
    "_normalize_hover_height",
    "_normalize_manipulation_intent",
    "_normalize_orientation_axis",
    "_normalize_orientation_goal",
    "_normalize_orientation_reference",
    "_normalize_relative_arm",
    "_normalize_relative_relation",
    "_relative_primary_placement",
    "_relative_relation_phrase",
    "_relative_scene_runtime_uid_mapping",
    "_resolve_relative_reference_source_uid",
    "_resolve_rigid_source_uid",
    "_should_upright_in_place",
    "_validate_orientation_fields",
    "_vector3",
]

_SHORT_UPRIGHTABLE_KEYWORDS = _SHORT_BOTTLE_LIKE_KEYWORDS | _SHORT_CUP_LIKE_KEYWORDS
_UPRIGHT_TASK_KEYWORDS = (
    "upright",
    "stand up",
    "stand upright",
    "vertical",
    "扶正",
    "竖起来",
    "竖直",
    "立起来",
)
_DEFAULT_HOVER_HEIGHT = 0.10
_SELF_REFERENCE_VALUES = {
    "self",
    "initial_self",
    "initial_position",
    "initial_pose",
    "origin",
    "itself",
    "自身",
    "自己",
    "原位",
    "初始位置",
}
_RELATION_ALIASES = {
    "in": "inside",
    "into": "inside",
    "inside": "inside",
    "放入": "inside",
    "放进": "inside",
    "里面": "inside",
    "on": "on",
    "onto": "on",
    "on_top": "on",
    "on_top_of": "on",
    "above": "on",
    "top": "on",
    "上": "on",
    "上方": "on",
    "上面": "on",
    "叠放": "on",
    "left": "left_of",
    "left_of": "left_of",
    "to_the_left_of": "left_of",
    "左": "left_of",
    "左边": "left_of",
    "front_left": "front_left_of",
    "front_left_of": "front_left_of",
    "left_front": "front_left_of",
    "left_front_of": "front_left_of",
    "to_the_front_left_of": "front_left_of",
    "左前": "front_left_of",
    "左前方": "front_left_of",
    "左前面": "front_left_of",
    "back_left": "back_left_of",
    "back_left_of": "back_left_of",
    "behind_left": "back_left_of",
    "left_back": "back_left_of",
    "left_behind": "back_left_of",
    "left_back_of": "back_left_of",
    "to_the_back_left_of": "back_left_of",
    "左后": "back_left_of",
    "左后方": "back_left_of",
    "左后面": "back_left_of",
    "右": "right_of",
    "右边": "right_of",
    "right": "right_of",
    "right_of": "right_of",
    "to_the_right_of": "right_of",
    "front_right": "front_right_of",
    "front_right_of": "front_right_of",
    "right_front": "front_right_of",
    "right_front_of": "front_right_of",
    "to_the_front_right_of": "front_right_of",
    "右前": "front_right_of",
    "右前方": "front_right_of",
    "右前面": "front_right_of",
    "back_right": "back_right_of",
    "back_right_of": "back_right_of",
    "behind_right": "back_right_of",
    "right_back": "back_right_of",
    "right_behind": "back_right_of",
    "right_back_of": "back_right_of",
    "to_the_back_right_of": "back_right_of",
    "右后": "back_right_of",
    "右后方": "back_right_of",
    "右后面": "back_right_of",
    "front": "front_of",
    "front_of": "front_of",
    "in_front_of": "front_of",
    "前": "front_of",
    "前方": "front_of",
    "前面": "front_of",
    "back": "behind",
    "behind": "behind",
    "back_of": "behind",
    "后": "behind",
    "后方": "behind",
    "后面": "behind",
}


def _relative_primary_placement(
    placements: tuple[RelativePlacementStepSpec, ...],
) -> RelativePlacementStepSpec:
    return next(
        (
            placement
            for placement in placements
            if placement.intent == "coordinated_pickment"
        ),
        placements[0],
    )


def _is_uprightable_object(obj: SceneObject) -> bool:
    shape = obj.config.get("shape", {}) or {}
    mesh_path = str(shape.get("fpath", "")) if isinstance(shape, Mapping) else ""
    mesh_parts = Path(mesh_path.replace("\\", "/")).parts[-4:] if mesh_path else ()
    description = str(obj.config.get("description", ""))
    text = " ".join([obj.source_uid, _base_name(obj), description, *mesh_parts]).lower()
    return _has_uprightable_keyword(text)


def _has_uprightable_keyword(text: str) -> bool:
    tokens = (
        text.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
    ).split()
    return any(
        keyword in tokens if keyword in _SHORT_UPRIGHTABLE_KEYWORDS else keyword in text
        for keyword in _UPRIGHTABLE_KEYWORDS
    )


def _is_upright_task_description(task_description: str) -> bool:
    text = task_description.strip().lower()
    return any(keyword in text for keyword in _UPRIGHT_TASK_KEYWORDS)


def _should_upright_in_place(
    *,
    intent: str,
    relation: str,
    orientation_goal: str,
    moved_object: SceneObject,
    reference_source_uid: str,
    table_source_uid: str,
    task_description: str,
) -> bool:
    if (
        intent != "place_relative"
        or relation != "on"
        or (
            orientation_goal != "upright"
            and not _is_upright_task_description(task_description)
        )
        or not _is_uprightable_object(moved_object)
    ):
        return False
    return reference_source_uid in {table_source_uid, moved_object.source_uid}


def _normalize_manipulation_intent(value: Any) -> str:
    if value is None:
        return "place_relative"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "relative": "place_relative",
        "relative_placement": "place_relative",
        "place": "place_relative",
        "put": "place_relative",
        "hold": "hold_hover",
        "hover": "hold_hover",
        "pick_hold": "hold_hover",
        "pick_and_hold": "hold_hover",
        "lift": "hold_hover",
        "悬空": "hold_hover",
        "拿起悬空": "hold_hover",
        "coordinated": "coordinated_pickment",
        "coordinated_pick": "coordinated_pickment",
        "dual_arm_pick": "coordinated_pickment",
        "dual_arm_move": "coordinated_pickment",
        "双臂抓取": "coordinated_pickment",
    }
    text = aliases.get(text, text)
    if text not in _SUPPORTED_MANIPULATION_INTENTS:
        raise ValueError(
            f"Unsupported manipulation intent {value!r}; expected one of "
            f"{sorted(_SUPPORTED_MANIPULATION_INTENTS)}."
        )
    return text


def _normalize_hover_height(value: Any) -> float:
    if value is None:
        return _DEFAULT_HOVER_HEIGHT
    try:
        height = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid hover_height {value!r}.") from exc
    if height <= 0.0 or height > 0.5:
        raise ValueError("hover_height must be in (0.0, 0.5].")
    return height


def _resolve_rigid_source_uid(
    value: Any,
    rigid_objects: list[SceneObject],
    *,
    field_name: str,
) -> str:
    return _resolve_scene_source_uid(
        value,
        rigid_objects,
        field_name=field_name,
    )


def _resolve_relative_reference_source_uid(
    value: Any,
    *,
    moved_source_uid: str,
    scene_objects: list[SceneObject],
) -> str:
    if value is not None:
        text = str(value).strip()
        normalized = text.lower().replace("-", "_").replace(" ", "_")
        if normalized in _SELF_REFERENCE_VALUES:
            return moved_source_uid
    return _resolve_scene_source_uid(
        value,
        scene_objects,
        field_name="reference_object",
    )


def _resolve_scene_source_uid(
    value: Any,
    scene_objects: list[SceneObject],
    *,
    field_name: str,
) -> str:
    if value is None:
        raise ValueError(f"LLM response missing required {field_name}.")
    text = str(value).strip()
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    if text in by_uid:
        return text

    normalized = _normalize_runtime_uid(text)
    matches = [
        obj.source_uid
        for obj in scene_objects
        if _normalize_runtime_uid(obj.source_uid) == normalized
        or _base_name(obj) == normalized
        or _candidate_relative_runtime_uid(obj) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"LLM returned unknown {field_name}: {text!r}.")
    raise ValueError(
        f"LLM returned ambiguous {field_name}: {text!r}; candidates: {matches}."
    )


def _normalize_relative_relation(value: Any) -> str:
    relation = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    relation = _RELATION_ALIASES.get(relation, relation)
    if relation not in _RELATIVE_RELATIONS:
        raise ValueError(
            f"Unsupported relative placement relation {value!r}; expected one "
            f"of {sorted(_RELATIVE_RELATIONS)}."
        )
    return relation


def _normalize_relative_arm(value: Any) -> str:
    if value is None:
        return "auto"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {
        "",
        "auto",
        "automatic",
        "unspecified",
        "none",
        "null",
        "default",
        "自动",
        "默认",
        "未指定",
        "不指定",
    }:
        return "auto"
    if text in {
        "left",
        "left_arm",
        "left_ur5",
        "左",
        "左臂",
        "左机械臂",
        "左手",
        "左手臂",
    }:
        return "left"
    if text in {
        "right",
        "right_arm",
        "right_ur5",
        "右",
        "右臂",
        "右机械臂",
        "右手",
        "右手臂",
    }:
        return "right"
    raise ValueError(
        f"Unsupported relative placement arm {value!r}; expected 'left', "
        "'right', or 'auto'."
    )


def _normalize_orientation_goal(value: Any) -> str:
    if value is None:
        return "preserve"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "none", "null", "default", "preserve", "keep", "保持"}:
        return "preserve"
    if text in {"upright", "vertical", "stand_upright", "扶正", "竖直", "竖起来"}:
        return "upright"
    if text in {"lay_flat", "flat", "level", "平放", "横放"}:
        return "lay_flat"
    if text in {"axis_align", "align_axis", "cardinal_align", "水平摆正", "摆正"}:
        return "axis_align"
    raise ValueError(
        f"Unsupported orientation_goal {value!r}; expected 'preserve', "
        "'upright', 'lay_flat', or 'axis_align'."
    )


def _normalize_orientation_reference(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "none", "null", "default", "no", "false", "无"}:
        return "none"
    if text in {"world_axes", "world_axis", "world", "table_axes", "x_y_axes"}:
        return "world_axes"
    if text in {
        "reference_object",
        "reference",
        "target",
        "support",
        "container",
        "pad",
        "box",
        "参考物体",
        "目标物体",
    }:
        return "reference_object"
    raise ValueError(
        f"Unsupported orientation_reference {value!r}; expected 'none' or "
        "'world_axes' or 'reference_object'."
    )


def _normalize_orientation_axis(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "none", "null", "default", "no", "false", "无"}:
        return "none"
    if text in {"x", "world_x", "x_axis", "world_x_axis"}:
        return "x"
    if text in {"y", "world_y", "y_axis", "world_y_axis"}:
        return "y"
    if text in {"long_axis", "long", "major_axis", "length", "长轴", "长边"}:
        return "long_axis"
    if text in {"short_axis", "short", "minor_axis", "width", "短轴", "短边"}:
        return "short_axis"
    raise ValueError(
        f"Unsupported orientation_axis {value!r}; expected 'none', 'x', 'y', "
        "'long_axis', or 'short_axis'."
    )


def _validate_orientation_fields(
    *,
    orientation_goal: str,
    orientation_reference: str,
    orientation_axis: str,
) -> None:
    if orientation_goal == "axis_align":
        if orientation_reference == "world_axes":
            if orientation_axis not in {"x", "y"}:
                raise ValueError(
                    "axis_align with orientation_reference='world_axes' requires "
                    "orientation_axis 'x' or 'y'."
                )
            return
        if orientation_reference == "reference_object":
            if orientation_axis not in {"long_axis", "short_axis"}:
                raise ValueError(
                    "axis_align with orientation_reference='reference_object' "
                    "requires orientation_axis 'long_axis' or 'short_axis'."
                )
            return
        raise ValueError(
            "axis_align requires orientation_reference 'world_axes' or "
            "'reference_object'."
        )

    if orientation_reference != "none" or orientation_axis != "none":
        raise ValueError(
            "preserve, upright, and lay_flat require orientation_reference='none' "
            "and orientation_axis='none'."
        )


def _relative_runtime_uid_mapping(
    rigid_objects: list[SceneObject],
) -> dict[str, str]:
    candidates: dict[str, str] = {}
    for obj in rigid_objects:
        if _is_container_like(obj):
            candidates[obj.source_uid] = _container_runtime_uid(obj)
            continue

        base = _target_runtime_suffix(_base_name(obj))
        base_count = sum(
            1 for other in rigid_objects if _base_name(other) == _base_name(obj)
        )
        candidates[obj.source_uid] = (
            base if base_count == 1 else _normalize_runtime_uid(obj.source_uid)
        )

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


def _relative_scene_runtime_uid_mapping(
    scene_objects: list[SceneObject],
    *,
    table_source_uid: str,
) -> dict[str, str]:
    candidates: dict[str, str] = {}
    rigid_runtime_uids = _relative_runtime_uid_mapping(
        [obj for obj in scene_objects if obj.source_role == "rigid_object"]
    )
    for obj in scene_objects:
        if obj.source_uid == table_source_uid:
            candidates[obj.source_uid] = "table"
        elif obj.source_role == "rigid_object":
            candidates[obj.source_uid] = rigid_runtime_uids[obj.source_uid]
        else:
            candidates[obj.source_uid] = _candidate_relative_runtime_uid(obj)

    counts: dict[str, int] = {}
    for runtime_uid in candidates.values():
        counts[runtime_uid] = counts.get(runtime_uid, 0) + 1
    return {
        source_uid: (
            runtime_uid
            if source_uid == table_source_uid or counts[runtime_uid] == 1
            else _normalize_runtime_uid(source_uid)
        )
        for source_uid, runtime_uid in candidates.items()
    }


def _relative_relation_phrase(relation: str) -> str:
    return _canonical_relative_relation_phrase(_normalize_relative_relation(relation))


def _vector3(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a 3-vector, got {value!r}.")
    return [float(item) for item in value]
