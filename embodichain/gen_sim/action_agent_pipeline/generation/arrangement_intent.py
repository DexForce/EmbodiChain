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

from embodichain.gen_sim.action_agent_pipeline.generation._spec_scene_helpers import (
    color_hint_for_object as _color_hint_for_object,
    resolve_rigid_uid as _resolve_rigid_uid_shared,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_layout import (
    _arrangement_object_size_score,
    _normalize_anchor,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _base_name,
    _string_list,
)

__all__ = [
    "_arrangement_object_categories",
    "_normalize_anchor",
    "_normalize_order_by",
    "_normalize_order_direction",
    "_object_color",
    "_order_uids_by_color",
    "_order_uids_by_size",
    "_resolve_arrangement_object_uids",
    "_validated_arrangement_order",
]

_SUPPORTED_ORDER_BY = {"size", "color", "explicit"}
_SUPPORTED_ORDER_DIRECTIONS = {"ascending", "descending", "given"}
_SIZE_ORDER_MARKERS = (
    "按大小",
    "大小顺序",
    "由大到小",
    "从大到小",
    "由小到大",
    "从小到大",
    "largest",
    "smallest",
    "large to small",
    "small to large",
    "by size",
    "size order",
)
_COLOR_ORDER_MARKERS = ("按颜色", "颜色顺序", "color order", "by color")
_COLOR_NAMES = (
    "红",
    "绿",
    "蓝",
    "黄",
    "橙",
    "紫",
    "黑",
    "白",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "black",
    "white",
)


def _validated_arrangement_order(
    order_by: str,
    order_direction: str,
    *,
    task_description: str,
) -> tuple[str, str]:
    text = task_description.strip().lower()
    if order_by == "size" and not any(marker in text for marker in _SIZE_ORDER_MARKERS):
        return "explicit", "given"
    if order_by == "color":
        color_count = sum(color in text for color in _COLOR_NAMES)
        explicit_color_order = (
            any(marker in text for marker in _COLOR_ORDER_MARKERS)
            or color_count >= 2
            and any(
                marker in text
                for marker in ("顺序", "依次", " order", "sequence", "sort")
            )
        )
        if not explicit_color_order:
            return "explicit", "given"
    return order_by, order_direction


def _arrangement_object_categories(
    value: Any,
    *,
    object_source_uids: Sequence[str],
    rigid_by_uid: Mapping[str, SceneObject],
) -> dict[str, str]:
    if value is None:
        return {uid: _base_name(rigid_by_uid[uid]) for uid in object_source_uids}
    if not isinstance(value, Mapping):
        raise ValueError("Arrangement object_categories must be an object mapping.")
    categories = {}
    for uid in object_source_uids:
        category = str(value.get(uid, "")).strip().lower()
        if not category:
            raise ValueError(
                "Arrangement object_categories must annotate every selected object; "
                f"missing: {uid!r}."
            )
        categories[uid] = category
    return categories


def _resolve_arrangement_object_uids(
    value: Any,
    rigid_by_uid: Mapping[str, SceneObject],
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
    rigid_by_uid: Mapping[str, SceneObject],
    *,
    field_name: str,
) -> str:
    return _resolve_rigid_uid_shared(
        value,
        rigid_by_uid,
        field_name=field_name,
        route_label="arrangement",
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


def _order_uids_by_size(
    source_uids: list[str],
    *,
    rigid_by_uid: Mapping[str, SceneObject],
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
    rigid_by_uid: Mapping[str, SceneObject],
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
