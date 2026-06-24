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
from typing import Any
import json

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _RelativePlacementSpec,
    _RelativePlacementStepSpec,
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _base_name,
    _candidate_relative_runtime_uid,
    _container_runtime_uid,
    _is_container_like,
    _normalize_runtime_uid,
    _string_list,
    _target_runtime_suffix,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _arm_side_for_position,
    _pick_table,
    _position_side_axis_value,
)

__all__ = [
    "_SIDE_RELATIONS",
    "_build_relative_placement_spec_with_llm",
    "_normalize_relative_relation",
    "_relative_relation_phrase",
    "_relative_scene_runtime_uid_mapping",
]

_RELATIVE_RELATIONS = {
    "inside",
    "on",
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "front_left_of",
    "back_left_of",
    "front_right_of",
    "back_right_of",
}

_SIDE_RELATIONS = _RELATIVE_RELATIONS - {"inside", "on"}

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


def _build_relative_placement_spec_with_llm(
    *,
    scene_objects: list[_SceneObject],
    project_name: str,
    task_description: str,
    model: str | None,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
    task_llm_caller: Callable[..., Mapping[str, Any]] | None = None,
) -> _RelativePlacementSpec:
    background_objects = [
        obj for obj in scene_objects if obj.source_role == "background"
    ]
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]
    if not background_objects:
        raise ValueError("Relative placement generation requires a background table.")
    if not rigid_objects:
        raise ValueError(
            "Relative placement generation requires a movable rigid object."
        )

    table = _pick_table(background_objects)
    if task_llm_caller is None:
        task_llm_caller = _call_relative_task_llm
    response = task_llm_caller(
        project_name=project_name,
        task_description=task_description,
        scene_summary=[
            {
                "source_uid": obj.source_uid,
                "role": obj.source_role,
                "object_type": _base_name(obj),
                "is_container_like": _is_container_like(obj),
                "mesh": obj.config.get("shape", {}).get("fpath"),
                "init_pos": obj.config.get("init_pos"),
            }
            for obj in scene_objects
        ],
        model=model,
    )
    return _apply_relative_task_response(
        response=response,
        table_source_uid=table.source_uid,
        scene_objects=scene_objects,
        rigid_objects=rigid_objects,
        task_description=task_description,
        release_offset_fn=release_offset_fn,
        staging_z_delta=staging_z_delta,
    )


def _call_relative_task_llm(
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
        "Parse a simple Dual-UR5 tabletop relative-placement task and produce "
        "a constrained config-level JSON spec. This JSON is used to generate "
        "task_prompt.txt, basic_background.txt, atom_actions.txt, and "
        "agent_success; a second LLM will later read those prompts to generate "
        "the executable graph JSON.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "placements": [\n'
        "    {\n"
        '      "moved_object": "<source_uid from rigid_object>",\n'
        '      "reference_object": "<source_uid from scene objects, or moved_object/self for initial-position moves>",\n'
        '      "goal_relation": '
        '"inside|on|left_of|right_of|front_of|behind|front_left_of|back_left_of|front_right_of|back_right_of",\n'
        '      "arm": "left|right|auto"\n'
        "    }\n"
        "  ],\n"
        '  "task_prompt_summary": "<one or two sentences for task_prompt>",\n'
        '  "basic_background_notes": "<short scene/task notes>",\n'
        '  "action_sketch": [\n'
        '    "grasp moved_object",\n'
        '    "move above the relation target pose",\n'
        '    "place at the release pose with PlaceAction"\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Use only source_uid values from the scene objects listed below.\n"
        "- Return one placement for a single-arm task and exactly two placements "
        "for a dual-arm task.\n"
        "- Treat the task as dual-arm when it explicitly says 双臂, 两臂, both "
        "arms, two arms, or when it describes separate work for the left arm and "
        "the right arm even if it does not literally say 双臂.\n"
        "- Do not invent a second placement when the task only moves one object.\n"
        "- moved_object is the object to grasp and move.\n"
        "- reference_object is the object used as the spatial reference, "
        "container, or support.\n"
        "- reference_object may be a rigid_object or a background object such as "
        "a pad, tray, basket, or container.\n"
        "- For single-object directional tasks such as moving the only object "
        "forward, left, front-left, or back-right from its initial position, set "
        "reference_object to the same source_uid as moved_object (or 'self'). "
        "This means the generator will use the object's initial position as a "
        "fixed anchor, not the object's moving runtime pose.\n"
        "- Within each placement, moved_object and reference_object must be "
        "different unless the task is an initial-position directional move.\n"
        "- For dual-arm tasks, the placements must use two different moved_object "
        "values and one left arm plus one right arm. Use arm='auto' only when "
        "the user did not specify which arm handles that placement.\n"
        "- arm selects the single UR5 arm that should manipulate moved_object. "
        "Use arm='left' for explicit left-arm instructions such as 左臂, 左机械臂, "
        "left arm, or left UR5; use arm='right' for explicit right-arm "
        "instructions such as 右臂, 右机械臂, right arm, or right UR5; use "
        "arm='auto' when the task does not specify an arm.\n"
        "- For Chinese/English left/right/front/back, use the relation enums "
        "from the rotated robot-view perspective. front_of means negative "
        "world-x; behind means positive world-x; left_of means negative "
        "world-y; right_of means positive world-y. Diagonal relations combine "
        "both axes: front_left_of, back_left_of, front_right_of, back_right_of.\n"
        "- If the task says to release an object above a basket/container so it "
        "falls into it, use goal_relation='inside'.\n"
        "- If the task says to stack/place one object on another non-container "
        "support, use goal_relation='on'.\n"
        "- Do not return numeric offsets, object poses, scales, success JSON, "
        "robot config, or full prompt files. The generator computes those "
        "deterministically.\n\n"
        f"Project: {project_name}\n"
        f"Task description:\n{task_description}\n"
        f"Scene objects:\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}"
    )
    llm = create_chat_openai(
        temperature=0.0,
        model=model,
        usage_stage="config_generation.relative_task",
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


def _apply_relative_task_response(
    *,
    response: Mapping[str, Any],
    table_source_uid: str,
    scene_objects: list[_SceneObject],
    rigid_objects: list[_SceneObject],
    task_description: str,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
) -> _RelativePlacementSpec:
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    runtime_uids = _relative_scene_runtime_uid_mapping(
        scene_objects,
        table_source_uid=table_source_uid,
    )

    placement_entries = _relative_placement_entries(response)
    if len(placement_entries) > 2:
        raise ValueError("Relative placement supports at most two arm placements.")

    forced_arm_sides = _relative_forced_arm_sides(
        placement_entries,
        by_uid=by_uid,
        rigid_objects=rigid_objects,
    )
    placements = tuple(
        _build_relative_placement_step(
            entry=entry,
            by_uid=by_uid,
            scene_objects=scene_objects,
            rigid_objects=rigid_objects,
            runtime_uids=runtime_uids,
            forced_side=forced_side,
            release_offset_fn=release_offset_fn,
            staging_z_delta=staging_z_delta,
        )
        for entry, forced_side in zip(placement_entries, forced_arm_sides)
    )
    _validate_relative_placements(placements)

    summary = str(response.get("task_prompt_summary", "")).strip()
    if not summary:
        summary = _default_relative_plan_summary(placements)
    background_notes = str(response.get("basic_background_notes", "")).strip()
    action_sketch = _string_list(response.get("action_sketch"))
    if not action_sketch:
        action_sketch = _default_relative_action_sketch(placements)

    primary = placements[0]

    return _RelativePlacementSpec(
        table_source_uid=table_source_uid,
        moved_source_uid=primary.moved_source_uid,
        reference_source_uid=primary.reference_source_uid,
        moved_runtime_uid=primary.moved_runtime_uid,
        reference_runtime_uid=primary.reference_runtime_uid,
        relation=primary.relation,
        active_side=primary.active_side,
        task_description=task_description,
        task_prompt_summary=summary,
        basic_background_notes=background_notes,
        action_sketch=action_sketch,
        release_offset=primary.release_offset,
        high_offset=primary.high_offset,
        placements=placements,
        reference_is_initial_pose=primary.reference_is_initial_pose,
        release_position=primary.release_position,
        high_position=primary.high_position,
    )


def _relative_placement_entries(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    placements = response.get("placements")
    if placements is None:
        return [response]
    if not isinstance(placements, list) or not placements:
        raise ValueError("LLM response placements must be a non-empty list.")
    entries: list[Mapping[str, Any]] = []
    for index, placement in enumerate(placements):
        if not isinstance(placement, Mapping):
            raise ValueError(f"Placement {index} must be a JSON object.")
        entries.append(placement)
    return entries


def _relative_forced_arm_sides(
    placement_entries: list[Mapping[str, Any]],
    *,
    by_uid: Mapping[str, _SceneObject],
    rigid_objects: list[_SceneObject],
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


def _build_relative_placement_step(
    *,
    entry: Mapping[str, Any],
    by_uid: Mapping[str, _SceneObject],
    scene_objects: list[_SceneObject],
    rigid_objects: list[_SceneObject],
    runtime_uids: Mapping[str, str],
    forced_side: str | None,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
) -> _RelativePlacementStepSpec:
    moved_source_uid = _resolve_rigid_source_uid(
        entry.get("moved_object"),
        rigid_objects,
        field_name="moved_object",
    )
    relation = _normalize_relative_relation(entry.get("goal_relation"))
    reference_source_uid = _resolve_relative_reference_source_uid(
        entry.get("reference_object"),
        moved_source_uid=moved_source_uid,
        scene_objects=scene_objects,
    )
    reference_is_initial_pose = moved_source_uid == reference_source_uid
    if reference_is_initial_pose and relation not in _SIDE_RELATIONS:
        raise ValueError(
            "Initial-position self-relative placement only supports directional "
            "relations, not inside/on."
        )

    reference_obj = by_uid[reference_source_uid]
    if relation == "on" and _is_container_like(reference_obj):
        relation = "inside"

    moved_runtime_uid = runtime_uids[moved_source_uid]
    reference_runtime_uid = runtime_uids[reference_source_uid]
    if moved_runtime_uid == reference_runtime_uid and not reference_is_initial_pose:
        raise ValueError(
            f"Relative placement produced duplicate runtime uid {moved_runtime_uid!r}."
        )

    release_offset = [float(value) for value in release_offset_fn(relation)]
    high_offset = list(release_offset)
    high_offset[2] += float(staging_z_delta)
    moved_position = _vector3(
        by_uid[moved_source_uid].config.get("init_pos", [0, 0, 0])
    )
    requested_side = _normalize_relative_arm(entry.get("arm"))
    active_side = (
        forced_side
        if forced_side is not None
        else (
            _arm_side_for_position(moved_position)
            if requested_side == "auto"
            else requested_side
        )
    )

    return _RelativePlacementStepSpec(
        moved_source_uid=moved_source_uid,
        reference_source_uid=reference_source_uid,
        moved_runtime_uid=moved_runtime_uid,
        reference_runtime_uid=reference_runtime_uid,
        relation=relation,
        active_side=active_side,
        release_offset=release_offset,
        high_offset=high_offset,
        reference_is_initial_pose=reference_is_initial_pose,
    )


def _validate_relative_placements(
    placements: tuple[_RelativePlacementStepSpec, ...],
) -> None:
    if not placements:
        raise ValueError("Relative placement requires at least one placement.")
    moved_source_uids = [placement.moved_source_uid for placement in placements]
    if len(moved_source_uids) != len(set(moved_source_uids)):
        raise ValueError("Relative placements must use distinct moved_object values.")
    if len(placements) == 2:
        active_sides = {placement.active_side for placement in placements}
        if active_sides != {"left", "right"}:
            raise ValueError(
                "Dual-arm relative placement requires one left arm and one right arm."
            )


def _resolve_rigid_source_uid(
    value: Any,
    rigid_objects: list[_SceneObject],
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
    scene_objects: list[_SceneObject],
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
    scene_objects: list[_SceneObject],
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


def _relative_runtime_uid_mapping(
    rigid_objects: list[_SceneObject],
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
    scene_objects: list[_SceneObject],
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


def _default_relative_task_summary(
    moved_uid: str,
    reference_uid: str,
    relation: str,
) -> str:
    return (
        f"Move `{moved_uid}` so its final state is "
        f"{_relative_relation_phrase(relation)} `{reference_uid}`."
    )


def _default_relative_plan_summary(
    placements: Sequence[_RelativePlacementStepSpec],
) -> str:
    if len(placements) == 1:
        placement = placements[0]
        return _default_relative_task_summary(
            placement.moved_runtime_uid,
            placement.reference_runtime_uid,
            placement.relation,
        )
    placement_text = "; ".join(
        f"use the {placement.active_side} UR5 to move "
        f"`{placement.moved_runtime_uid}` "
        f"{_relative_relation_phrase(placement.relation)} "
        f"`{placement.reference_runtime_uid}`"
        for placement in placements
    )
    return f"Use both UR5 arms for a dual-arm relative placement: {placement_text}."


def _default_relative_action_sketch(
    placements: Sequence[_RelativePlacementStepSpec],
) -> list[str]:
    if len(placements) == 1:
        placement = placements[0]
        return [
            f"grasp {placement.moved_runtime_uid}",
            (
                f"move above the {placement.relation} release pose relative to "
                f"{placement.reference_runtime_uid}"
            ),
            "place at the release pose with PlaceAction",
        ]
    sketch = ["grasp both moved objects with their assigned arms"]
    for placement in placements:
        sketch.extend(
            [
                (
                    f"use {placement.active_side}_arm to move "
                    f"{placement.moved_runtime_uid} above the release pose relative "
                    f"to {placement.reference_runtime_uid}"
                ),
                f"place {placement.moved_runtime_uid} with PlaceAction",
            ]
        )
    return sketch


def _relative_relation_phrase(relation: str) -> str:
    relation = _normalize_relative_relation(relation)
    if relation == "inside":
        return "inside"
    if relation == "on":
        return "on top of"
    if relation == "left_of":
        return "to the left of"
    if relation == "right_of":
        return "to the right of"
    if relation == "front_of":
        return "in front of"
    if relation == "behind":
        return "behind"
    if relation == "front_left_of":
        return "to the front-left of"
    if relation == "back_left_of":
        return "to the back-left of"
    if relation == "front_right_of":
        return "to the front-right of"
    if relation == "back_right_of":
        return "to the back-right of"
    raise ValueError(f"Unsupported relative placement relation: {relation!r}.")


def _vector3(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a 3-vector, got {value!r}.")
    return [float(item) for item in value]
