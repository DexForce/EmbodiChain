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

"""Canonicalize coordinated relative-manipulation payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.domain.object_semantics import (
    FLAT_CARRIER_KEYWORDS as _FLAT_CARRIER_KEYWORDS,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.actions import (
    MAX_COORDINATED_PAYLOADS as _MAX_COORDINATED_PAYLOADS,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _base_name,
    _is_container_like,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_intent import (
    _normalize_manipulation_intent,
    _normalize_relative_arm,
    _normalize_relative_relation,
    _resolve_rigid_source_uid,
    _vector3,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _arm_side_for_position,
    _position_side_axis_value,
)

__all__ = [
    "_canonicalize_flat_coordinated_transport_entries",
    "_coordinated_payload_entries",
    "_coordinated_transport_entry",
    "_normalize_coordinated_direction",
    "_normalize_coordinated_terminal_behavior",
    "_relative_forced_arm_sides",
    "_with_coordinated_pickment_intent",
    "_with_coordinated_transport_relation",
]

_COORDINATED_DUAL_ARM_KEYWORDS = (
    "双臂",
    "两臂",
    "双手",
    "both arms",
    "two arms",
)
_COORDINATED_DIRECTIONS = {
    "front",
    "back",
    "left",
    "right",
    "front_left",
    "front_right",
    "back_left",
    "back_right",
    "none",
}
_COORDINATED_TERMINAL_BEHAVIORS = {"hold", "place"}


def _coordinated_transport_entry(
    entries: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if len(entries) != 1:
        return None
    entry = entries[0]
    if _normalize_manipulation_intent(entry.get("intent")) != "coordinated_pickment":
        return None
    if not any(
        field in entry for field in ("payloads", "direction", "terminal_behavior")
    ):
        return None
    return entry


def _canonicalize_flat_coordinated_transport_entries(
    entries: list[Mapping[str, Any]],
    *,
    rigid_objects: list[SceneObject],
) -> list[Mapping[str, Any]]:
    """Fold flat payload placements into one coordinated transport entry."""
    if len(entries) <= 1:
        return entries
    coordinated = [
        entry
        for entry in entries
        if _normalize_manipulation_intent(entry.get("intent")) == "coordinated_pickment"
    ]
    payload_entries = [
        entry
        for entry in entries
        if _normalize_manipulation_intent(entry.get("intent")) == "place_relative"
    ]
    if (
        len(coordinated) != 1
        or not 1 <= len(payload_entries) <= _MAX_COORDINATED_PAYLOADS
        or len(entries) != len(payload_entries) + 1
    ):
        return entries

    coordinated_entry = dict(coordinated[0])
    raw_nested_payloads = coordinated_entry.get("payloads", [])
    if (
        not isinstance(raw_nested_payloads, list)
        or len(raw_nested_payloads) > _MAX_COORDINATED_PAYLOADS
    ):
        raise ValueError(
            "CoordinatedPickment payloads must be a list of at most "
            f"{_MAX_COORDINATED_PAYLOADS} objects."
        )
    carrier_uid = _resolve_rigid_source_uid(
        coordinated_entry.get("moved_object"),
        rigid_objects,
        field_name="moved_object",
    )
    nested_by_uid: dict[str, Mapping[str, Any]] = {}
    for index, payload in enumerate(raw_nested_payloads):
        if not isinstance(payload, Mapping):
            raise ValueError(f"CoordinatedPickment payload {index} must be an object.")
        payload_uid = _resolve_rigid_source_uid(
            payload.get("object"),
            rigid_objects,
            field_name=f"payloads[{index}].object",
        )
        if payload_uid in nested_by_uid:
            raise ValueError("CoordinatedPickment payload objects must be distinct.")
        nested_by_uid[payload_uid] = payload

    flat_payloads: list[tuple[str, str]] = []
    flat_payload_uids: set[str] = set()
    for index, entry in enumerate(payload_entries):
        reference_uid = _resolve_rigid_source_uid(
            entry.get("reference_object"),
            rigid_objects,
            field_name=f"flat payload {index} reference_object",
        )
        relation = _normalize_relative_relation(entry.get("goal_relation"))
        if reference_uid != carrier_uid or relation not in {"inside", "on"}:
            return entries
        payload_uid = _resolve_rigid_source_uid(
            entry.get("moved_object"),
            rigid_objects,
            field_name=f"flat payload {index} moved_object",
        )
        if payload_uid in flat_payload_uids:
            raise ValueError("CoordinatedPickment payload objects must be distinct.")
        flat_payload_uids.add(payload_uid)
        arm = _normalize_relative_arm(entry.get("arm"))
        flat_payloads.append((payload_uid, arm))

    # Nested and flat payload forms are two representations of one contract.
    if nested_by_uid and set(nested_by_uid) != flat_payload_uids:
        nested_only = sorted(set(nested_by_uid) - flat_payload_uids)
        flat_only = sorted(flat_payload_uids - set(nested_by_uid))
        raise ValueError(
            "Loaded coordinated transport has conflicting nested and top-level "
            f"payload objects; nested_only={nested_only}, flat_only={flat_only}."
        )

    payloads: list[dict[str, Any]] = []
    for payload_uid, flat_arm in flat_payloads:
        nested = nested_by_uid.get(payload_uid, {})
        nested_arm = nested.get("arm", "auto")
        arm = flat_arm if flat_arm != "auto" else nested_arm
        slot = nested.get("slot")
        if slot is None:
            slot = flat_arm if flat_arm in {"left", "right"} else "auto"
        payloads.append({"object": payload_uid, "arm": arm, "slot": slot})
    coordinated_entry["payloads"] = payloads
    if "direction" not in coordinated_entry:
        relation = str(coordinated_entry.get("goal_relation", "")).strip().lower()
        direction_by_relation = {
            "front_of": "front",
            "behind": "back",
            "left_of": "left",
            "right_of": "right",
            "front_left_of": "front_left",
            "front_right_of": "front_right",
            "back_left_of": "back_left",
            "back_right_of": "back_right",
        }
        if relation in direction_by_relation:
            coordinated_entry["direction"] = direction_by_relation[relation]
    return [coordinated_entry]


def _coordinated_payload_entries(
    coordinated_entry: Mapping[str, Any],
    *,
    by_uid: Mapping[str, SceneObject],
    rigid_objects: list[SceneObject],
) -> list[Mapping[str, Any]]:
    raw_payloads = coordinated_entry.get("payloads", [])
    if (
        not isinstance(raw_payloads, list)
        or len(raw_payloads) > _MAX_COORDINATED_PAYLOADS
    ):
        raise ValueError(
            "CoordinatedPickment payloads must be a list of at most "
            f"{_MAX_COORDINATED_PAYLOADS} objects."
        )
    carrier_uid = _resolve_rigid_source_uid(
        coordinated_entry.get("moved_object"),
        rigid_objects,
        field_name="moved_object",
    )
    relation = _coordinated_payload_relation(by_uid[carrier_uid])
    entries: list[Mapping[str, Any]] = []
    payload_uids: list[str] = []
    for index, payload in enumerate(raw_payloads):
        if not isinstance(payload, Mapping):
            raise ValueError(f"CoordinatedPickment payload {index} must be an object.")
        payload_uid = _resolve_rigid_source_uid(
            payload.get("object"),
            rigid_objects,
            field_name=f"payloads[{index}].object",
        )
        if payload_uid == carrier_uid:
            raise ValueError(
                "CoordinatedPickment payload must differ from moved_object."
            )
        if payload_uid in payload_uids:
            raise ValueError("CoordinatedPickment payload objects must be distinct.")
        payload_uids.append(payload_uid)
        slot = str(payload.get("slot", "auto")).strip().lower()
        if slot not in {"left", "right", "center", "auto"}:
            raise ValueError(
                "CoordinatedPickment payload slot must be left, right, center, or auto."
            )
        arm = payload.get("arm", "auto")
        if str(arm).strip().lower() == "auto" and slot in {"left", "right"}:
            arm = slot
        entries.append(
            {
                "intent": "place_relative",
                "moved_object": payload_uid,
                "arm": arm,
                "reference_object": carrier_uid,
                "goal_relation": relation,
                "orientation_goal": "preserve",
                "orientation_reference": "none",
                "orientation_axis": "none",
            }
        )
    return entries


def _coordinated_payload_relation(carrier: SceneObject) -> str:
    text = " ".join(
        (
            carrier.source_uid,
            _base_name(carrier),
            str(carrier.config.get("description", "")),
            str((carrier.config.get("shape", {}) or {}).get("fpath", "")),
        )
    ).lower()
    if any(keyword in text for keyword in _FLAT_CARRIER_KEYWORDS):
        return "on"
    return "inside" if _is_container_like(carrier) else "on"


def _normalize_coordinated_direction(value: Any) -> str:
    if value is None:
        return "none"
    direction = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "behind": "back",
        "front_of": "front",
        "left_of": "left",
        "right_of": "right",
        "原地": "none",
        "前": "front",
        "后": "back",
        "左": "left",
        "右": "right",
    }
    direction = aliases.get(direction, direction)
    if direction not in _COORDINATED_DIRECTIONS:
        raise ValueError(
            f"Unsupported coordinated direction {value!r}; expected one of "
            f"{sorted(_COORDINATED_DIRECTIONS)}."
        )
    return direction


def _normalize_coordinated_terminal_behavior(
    value: Any,
    *,
    task_description: str,
) -> str:
    if value is None:
        text = task_description.lower()
        terminal_cues = {
            "hold": ("端起", "举起", "悬空", "保持", "举着", "hold"),
            "place": ("放下", "落下", "放回", "松开", "put down", "release"),
        }
        matched_cues = [
            (text.rfind(keyword), behavior)
            for behavior, keywords in terminal_cues.items()
            for keyword in keywords
            if keyword in text
        ]
        if matched_cues:
            return max(matched_cues, key=lambda match: match[0])[1]
        return "place"
    behavior = str(value).strip().lower()
    aliases = {
        "悬空": "hold",
        "保持": "hold",
        "举着": "hold",
        "放下": "place",
        "落下": "place",
        "放回": "place",
        "松开": "place",
        "put down": "place",
        "release": "place",
    }
    behavior = aliases.get(behavior, behavior)
    if behavior not in _COORDINATED_TERMINAL_BEHAVIORS:
        raise ValueError(
            f"Unsupported coordinated terminal_behavior {value!r}; expected "
            f"one of {sorted(_COORDINATED_TERMINAL_BEHAVIORS)}."
        )
    return behavior


def _with_coordinated_transport_relation(
    entry: Mapping[str, Any],
    *,
    direction: str,
) -> Mapping[str, Any]:
    relation_by_direction = {
        "front": "front_of",
        "back": "behind",
        "left": "left_of",
        "right": "right_of",
        "front_left": "front_left_of",
        "front_right": "front_right_of",
        "back_left": "back_left_of",
        "back_right": "back_right_of",
        "none": "on",
    }
    normalized = dict(entry)
    normalized["reference_object"] = entry.get("moved_object")
    normalized["goal_relation"] = relation_by_direction[direction]
    normalized["arm"] = "auto"
    normalized["orientation_goal"] = "preserve"
    normalized["orientation_reference"] = "none"
    normalized["orientation_axis"] = "none"
    return normalized


def _with_coordinated_pickment_intent(
    placement_entries: list[Mapping[str, Any]],
    *,
    task_description: str,
) -> list[Mapping[str, Any]]:
    if len(placement_entries) != 1:
        return placement_entries
    if not _is_dual_arm_task_text(task_description):
        return placement_entries
    entry = dict(placement_entries[0])
    intent = _normalize_manipulation_intent(entry.get("intent"))
    if intent == "hold_hover":
        return placement_entries
    entry["intent"] = "coordinated_pickment"
    entry["arm"] = "auto"
    return [entry]


def _is_dual_arm_task_text(task_description: str) -> bool:
    text = task_description.strip().lower()
    return any(keyword in text for keyword in _COORDINATED_DUAL_ARM_KEYWORDS)


def _relative_forced_arm_sides(
    placement_entries: list[Mapping[str, Any]],
    *,
    by_uid: Mapping[str, SceneObject],
    rigid_objects: list[SceneObject],
) -> list[str | None]:
    if len(placement_entries) != 2:
        return [None for _ in placement_entries]

    requested_sides = [
        _normalize_relative_arm(entry.get("arm")) for entry in placement_entries
    ]
    explicit_sides = [side for side in requested_sides if side != "auto"]
    if len(explicit_sides) == 2:
        return [None, None]
    if len(explicit_sides) == 1:
        complement = "right" if explicit_sides[0] == "left" else "left"
        return [
            requested_side if requested_side != "auto" else complement
            for requested_side in requested_sides
        ]

    moved_source_uids = [
        _resolve_rigid_source_uid(
            entry.get("moved_object"),
            rigid_objects,
            field_name="moved_object",
        )
        for entry in placement_entries
    ]
    positions = [
        _vector3(by_uid[source_uid].config.get("init_pos", [0.0, 0.0, 0.0]))
        for source_uid in moved_source_uids
    ]
    inferred_sides = [_arm_side_for_position(position) for position in positions]
    if set(inferred_sides) == {"left", "right"}:
        return inferred_sides

    side_values = [_position_side_axis_value(position) for position in positions]
    if side_values[0] <= side_values[1]:
        return ["left", "right"]
    return ["right", "left"]
