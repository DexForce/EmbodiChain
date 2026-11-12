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

"""Validation for zero-based, inclusive, single-track atomic skill intervals."""

from __future__ import annotations

from typing import Any

__all__ = ["ConflictError", "validate_skill_intervals"]


class ConflictError(ValueError):
    """A stale edit would overwrite a newer annotation document."""


def validate_skill_intervals(value: object, frame_count: int) -> list[dict[str, Any]]:
    """Normalize intervals, rejecting invalid boundaries, IDs and overlaps.

    Args:
        value: JSON array of atomic skill interval objects.
        frame_count: Number of images in the selected frame sequence.

    Returns:
        New dictionaries sorted by inclusive start and end frame.

    Raises:
        ValueError: A field, boundary, ID or overlap violates the format.
    """
    if not isinstance(value, list) or len(value) > 100_000:
        raise ValueError("原子技能区间必须为数组，且最多包含 100000 段。")
    result: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    required = {"id", "label", "start_frame", "end_frame"}
    for position, item in enumerate(value, start=1):
        if not isinstance(item, dict) or not required <= item.keys():
            raise ValueError(
                f"第 {position} 段缺少 id、label、start_frame 或 end_frame。"
            )
        if item.keys() - required - {"note", "core_interval"}:
            raise ValueError(f"第 {position} 段包含未知字段。")
        identifier, label, note = item["id"], item["label"], item.get("note", "")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or len(identifier) > 128
        ):
            raise ValueError(f"第 {position} 段的 id 必须是非空字符串，最长 128 字符。")
        if identifier in identifiers:
            raise ValueError(f"重复的标注 id：{identifier}")
        if not isinstance(label, str) or not label.strip() or len(label) > 512:
            raise ValueError(f"第 {position} 段的原子技能名称必须非空，最长 512 字符。")
        if not isinstance(note, str) or len(note) > 8192:
            raise ValueError(f"第 {position} 段的备注必须为字符串，最长 8192 字符。")
        start, end = item["start_frame"], item["end_frame"]
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"第 {position} 段的起止帧必须为整数。")
        if not 0 <= start <= end < frame_count:
            raise ValueError(
                f"第 {position} 段 [{start}, {end}] 超出 0–{frame_count - 1}，或终点早于起点。"
            )
        identifiers.add(identifier)
        result.append(
            dict(
                id=identifier,
                label=label.strip(),
                start_frame=start,
                end_frame=end,
                note=note,
            )
        )
        if "core_interval" in item:
            core = item["core_interval"]
            if (
                not isinstance(core, dict)
                or core.keys() != {"start_frame", "end_frame"}
                or type(core["start_frame"]) is not int
                or type(core["end_frame"]) is not int
                or not start <= core["start_frame"] <= core["end_frame"] <= end
            ):
                raise ValueError(
                    f"第 {position} 段的拟合核心区间必须完整包含在标注区间内。"
                )
            result[-1]["core_interval"] = dict(core)
    result.sort(
        key=lambda skill_interval: (
            skill_interval["start_frame"],
            skill_interval["end_frame"],
        )
    )
    for previous, current in zip(result, result[1:]):
        if current["start_frame"] <= previous["end_frame"]:
            raise ValueError(
                f"区间重叠：{previous['label']} [{previous['start_frame']}, "
                f"{previous['end_frame']}] 与 {current['label']} "
                f"[{current['start_frame']}, {current['end_frame']}]。"
            )
    return result
