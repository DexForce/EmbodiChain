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
from pathlib import Path
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
    "_build_object_manipulation_spec_with_llm",
    "_build_relative_placement_spec_with_llm",
    "_call_object_manipulation_task_llm",
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
_SUPPORTED_MANIPULATION_INTENTS = {
    "place_relative",
    "hold_hover",
    "coordinated_pickment",
}
_COORDINATED_DUAL_ARM_KEYWORDS = (
    "双臂",
    "两臂",
    "双手",
    "both arms",
    "two arms",
)
_CUBE_LIKE_KEYWORDS = (
    "cube",
    "block",
    "方块",
    "积木",
)
_BOTTLE_LIKE_KEYWORDS = (
    "bottle",
    "can",
    "jar",
    "tin",
    "soda",
    "cola",
    "罐头",
    "易拉罐",
    "瓶",
    "瓶子",
)
_CUP_LIKE_KEYWORDS = (
    "cup",
    "mug",
    "paper cup",
    "water cup",
    "纸杯",
    "水杯",
    "杯子",
    "马克杯",
    "茶杯",
)
_SHORT_BOTTLE_LIKE_KEYWORDS = {"can", "jar", "tin"}
_SHORT_CUP_LIKE_KEYWORDS = {"cup", "mug"}
_UPRIGHTABLE_KEYWORDS = (*_BOTTLE_LIKE_KEYWORDS, *_CUP_LIKE_KEYWORDS)
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
_CUBE_DEFAULT_AXIS = "x"
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


def _build_relative_placement_spec_with_llm(
    *,
    scene_objects: list[_SceneObject],
    project_name: str,
    task_description: str,
    model: str | None,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
    pose_sensitive_staging_z_delta: float,
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
        pose_sensitive_staging_z_delta=pose_sensitive_staging_z_delta,
    )


def _build_object_manipulation_spec_with_llm(
    *,
    scene_objects: list[_SceneObject],
    project_name: str,
    task_description: str,
    model: str | None,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
    pose_sensitive_staging_z_delta: float,
    task_llm_caller: Callable[..., Mapping[str, Any]] | None = None,
) -> _RelativePlacementSpec:
    return _build_relative_placement_spec_with_llm(
        scene_objects=scene_objects,
        project_name=project_name,
        task_description=task_description,
        model=model,
        release_offset_fn=release_offset_fn,
        staging_z_delta=staging_z_delta,
        pose_sensitive_staging_z_delta=pose_sensitive_staging_z_delta,
        task_llm_caller=task_llm_caller or _call_object_manipulation_task_llm,
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
        '      "arm": "left|right|auto",\n'
        '      "orientation_goal": "preserve|upright|lay_flat|axis_align",\n'
        '      "orientation_reference": "none|world_axes|reference_object",\n'
        '      "orientation_axis": "none|x|y|long_axis|short_axis"\n'
        "    }\n"
        "  ],\n"
        '  "task_prompt_summary": "<one or two sentences for task_prompt>",\n'
        '  "basic_background_notes": "<short scene/task notes>",\n'
        '  "action_sketch": [\n'
        '    "grasp moved_object",\n'
        '    "move above the relation target pose",\n'
        '    "place at the release pose with Place"\n'
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
        "from the rotated robot-view perspective. front_of means positive "
        "world-x; behind means negative world-x; left_of means positive "
        "world-y; right_of means negative world-y. Diagonal relations combine "
        "both axes: front_left_of, back_left_of, front_right_of, back_right_of.\n"
        "- If the task says to release an object above a basket/container so it "
        "falls into it, use goal_relation='inside'.\n"
        "- If the task says to stack/place one object on another non-container "
        "support, use goal_relation='on'.\n"
        "- orientation_goal captures the held object's intended pose before "
        "release. Use 'upright' for tasks like 扶正, 竖起来, or stand upright. "
        "Use 'lay_flat' for tasks like 平放, 横放, or lay flat. Use "
        "'axis_align' for tasks like 水平摆正, 摆正, or aligning an object to a "
        "pad, box, container, or support axis. Use 'preserve' when no "
        "orientation change is requested.\n"
        "- For axis_align, set orientation_reference='reference_object' and "
        "orientation_axis='long_axis' when aligning an object such as a stapler "
        "or shoe to the long side of a pad, box, or container. Use "
        "orientation_axis='short_axis' only when the task explicitly asks for "
        "the short side. Use orientation_reference='world_axes' with "
        "orientation_axis='x' or 'y' only when the task explicitly specifies a "
        "world/table axis.\n"
        "- For preserve, upright, and lay_flat, use orientation_reference='none' "
        "and orientation_axis='none'.\n"
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


def _call_object_manipulation_task_llm(
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
        "Parse a simple Dual-UR5 tabletop object-manipulation task and produce "
        "one constrained config-level JSON spec. The generator computes offsets, "
        "robot config, success JSON, and action prompts deterministically.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "manipulations": [\n'
        "    {\n"
        '      "intent": "place_relative|hold_hover|coordinated_pickment",\n'
        '      "moved_object": "<source_uid from rigid_object>",\n'
        '      "arm": "left|right|auto",\n'
        '      "reference_object": "<source_uid from scene objects, or moved_object/self for initial-position moves>",\n'
        '      "goal_relation": "inside|on|left_of|right_of|front_of|behind|front_left_of|back_left_of|front_right_of|back_right_of",\n'
        '      "hover_height": 0.10,\n'
        '      "orientation_goal": "preserve|upright|lay_flat|axis_align",\n'
        '      "orientation_reference": "none|world_axes|reference_object",\n'
        '      "orientation_axis": "none|x|y|long_axis|short_axis"\n'
        "    }\n"
        "  ],\n"
        '  "task_prompt_summary": "<one or two sentences for task_prompt>",\n'
        '  "basic_background_notes": "<short scene/task notes>",\n'
        '  "action_sketch": ["<short deterministic action sketch>"]\n'
        "}\n\n"
        "Rules:\n"
        "- Use only source_uid values from the scene objects listed below.\n"
        "- Use intent='coordinated_pickment' when the task asks both arms to "
        "pick, lift, carry, or move one shared object such as a pot, tray, "
        "roller, or other large object. Return exactly one manipulation for "
        "this case.\n"
        "- Use intent='hold_hover' when the task asks one arm, or each arm for "
        "its own object, to pick up, lift, hold, "
        "or suspend an object in the air without placing or releasing it. For "
        "hold_hover, omit reference_object and goal_relation, use "
        "orientation_goal='preserve', orientation_reference='none', "
        "orientation_axis='none', and hover_height=0.10 unless the user gives a "
        "specific height.\n"
        "- Use intent='place_relative' for tasks that ask to place, put, stack "
        "onto, move beside, move into, or release an object at a spatial target. "
        "For place_relative, include reference_object and goal_relation.\n"
        "- Return exactly two manipulations for a dual-arm task that moves two "
        "different objects. Treat the task as dual-arm when it explicitly says "
        "双臂, 两臂, both arms, two arms, 一只机械臂...另一只机械臂, or separate "
        "work for left and right arms. If the dual-arm task moves one shared "
        "object, use exactly one coordinated_pickment manipulation instead.\n"
        "- Do not mix hold_hover, place_relative, and coordinated_pickment in "
        "one response; v1 only supports homogeneous manipulation intents.\n"
        "- For dual-arm tasks with two objects, use two different moved_object "
        "values and one left arm plus one right arm. Use arm='auto' only when "
        "the user did not specify which arm handles that manipulation. For "
        "coordinated_pickment, set arm='auto'.\n"
        "- For place_relative, reference_object may be a rigid_object or a "
        "background object such as a pad, tray, basket, or container. For "
        "single-object directional tasks from the object's initial position, "
        "set reference_object to the moved object or 'self'.\n"
        "- If the task says to release an object above a basket/container so it "
        "falls into it, use goal_relation='inside'. If it says to place on a "
        "non-container support, use goal_relation='on'.\n"
        "- orientation_goal captures the held object's intended pose before "
        "release. Use 'upright' for 扶正/竖起来, 'lay_flat' for 平放/横放, "
        "'axis_align' for 水平摆正/摆正/alignment, and 'preserve' otherwise.\n"
        "- For axis_align, use orientation_reference='reference_object' with "
        "orientation_axis='long_axis' for aligning to a pad/box/container long "
        "side, or orientation_reference='world_axes' with orientation_axis='x' "
        "or 'y' only when a world/table axis is explicit.\n"
        "- Do not return numeric offsets, object poses, robot config, success "
        "JSON, or full prompt files.\n\n"
        f"Project: {project_name}\n"
        f"Task description:\n{task_description}\n"
        f"Scene objects:\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}"
    )
    llm = create_chat_openai(
        temperature=0.0,
        model=model,
        usage_stage="config_generation.object_manipulation_task",
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
    pose_sensitive_staging_z_delta: float,
) -> _RelativePlacementSpec:
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    runtime_uids = _relative_scene_runtime_uid_mapping(
        scene_objects,
        table_source_uid=table_source_uid,
    )

    placement_entries = _relative_placement_entries(response)
    placement_entries = _with_coordinated_pickment_intent(
        placement_entries,
        task_description=task_description,
    )
    if len(placement_entries) > 2:
        raise ValueError("Object manipulation supports at most two arm actions.")

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
            table_source_uid=table_source_uid,
            task_description=task_description,
            forced_side=forced_side,
            release_offset_fn=release_offset_fn,
            staging_z_delta=staging_z_delta,
            pose_sensitive_staging_z_delta=pose_sensitive_staging_z_delta,
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
        intent=primary.intent,
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
        orientation_goal=primary.orientation_goal,
        orientation_axis=primary.orientation_axis,
        orientation_align_to_runtime_uid=primary.orientation_align_to_runtime_uid,
        hover_height=primary.hover_height,
        upright_in_place=primary.upright_in_place,
        pickup_upright_direction=primary.pickup_upright_direction,
        pickup_rotate_upright=primary.pickup_rotate_upright,
    )


def _relative_placement_entries(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    placements = response.get("manipulations", response.get("placements"))
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
    table_source_uid: str,
    task_description: str,
    forced_side: str | None,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
    pose_sensitive_staging_z_delta: float,
) -> _RelativePlacementStepSpec:
    intent = _normalize_manipulation_intent(entry.get("intent"))
    moved_source_uid = _resolve_rigid_source_uid(
        entry.get("moved_object"),
        rigid_objects,
        field_name="moved_object",
    )
    if intent == "hold_hover":
        relation = "on"
        reference_source_uid = moved_source_uid
        reference_is_initial_pose = True
    else:
        relation = _normalize_relative_relation(entry.get("goal_relation"))
        reference_source_uid = _resolve_relative_reference_source_uid(
            entry.get("reference_object"),
            moved_source_uid=moved_source_uid,
            scene_objects=scene_objects,
        )
        reference_is_initial_pose = moved_source_uid == reference_source_uid
    orientation_goal = _normalize_orientation_goal(entry.get("orientation_goal"))
    orientation_reference = _normalize_orientation_reference(
        entry.get("orientation_reference")
    )
    orientation_axis = _normalize_orientation_axis(entry.get("orientation_axis"))
    upright_in_place = _should_upright_in_place(
        intent=intent,
        relation=relation,
        orientation_goal=orientation_goal,
        moved_object=by_uid[moved_source_uid],
        reference_source_uid=reference_source_uid,
        table_source_uid=table_source_uid,
        task_description=task_description,
    )
    if upright_in_place:
        orientation_goal = "upright"
        orientation_reference = "none"
        orientation_axis = "none"
    if upright_in_place and reference_is_initial_pose:
        reference_source_uid = table_source_uid
        reference_is_initial_pose = False
    if intent != "hold_hover":
        if reference_is_initial_pose and relation not in _SIDE_RELATIONS:
            raise ValueError(
                "Initial-position self-relative placement only supports directional "
                "relations, not inside/on."
            )

        reference_obj = by_uid[reference_source_uid]
        if relation == "on" and _is_container_like(reference_obj):
            relation = "inside"
            upright_in_place = False

    moved_runtime_uid = runtime_uids[moved_source_uid]
    reference_runtime_uid = runtime_uids[reference_source_uid]
    if moved_runtime_uid == reference_runtime_uid and not reference_is_initial_pose:
        raise ValueError(
            f"Relative placement produced duplicate runtime uid {moved_runtime_uid!r}."
        )
    if _should_axis_align_cube_by_default(
        intent=intent,
        moved_object=by_uid[moved_source_uid],
        orientation_goal=orientation_goal,
        orientation_reference=orientation_reference,
        orientation_axis=orientation_axis,
    ):
        orientation_goal = "axis_align"
        orientation_reference = "world_axes"
        orientation_axis = _CUBE_DEFAULT_AXIS
    if intent == "hold_hover" and (
        orientation_goal != "preserve"
        or orientation_reference != "none"
        or orientation_axis != "none"
    ):
        raise ValueError("hold_hover requires preserve orientation fields.")
    _validate_orientation_fields(
        orientation_goal=orientation_goal,
        orientation_reference=orientation_reference,
        orientation_axis=orientation_axis,
    )
    orientation_align_to_runtime_uid = (
        reference_runtime_uid
        if orientation_reference == "reference_object" and not reference_is_initial_pose
        else None
    )

    if intent == "hold_hover":
        hover_height = _normalize_hover_height(entry.get("hover_height"))
        release_offset = [0.0, 0.0, hover_height]
    else:
        hover_height = _DEFAULT_HOVER_HEIGHT
        release_offset = [float(value) for value in release_offset_fn(relation)]
    high_offset = list(release_offset)
    if intent == "place_relative":
        high_offset[2] += float(
            pose_sensitive_staging_z_delta
            if orientation_goal != "preserve"
            else staging_z_delta
        )
    moved_position = _vector3(
        by_uid[moved_source_uid].config.get("init_pos", [0, 0, 0])
    )
    requested_side = _normalize_relative_arm(entry.get("arm"))
    if intent == "coordinated_pickment":
        active_side = "left"
    else:
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
        intent=intent,
        moved_source_uid=moved_source_uid,
        reference_source_uid=reference_source_uid,
        moved_runtime_uid=moved_runtime_uid,
        reference_runtime_uid=reference_runtime_uid,
        relation=relation,
        active_side=active_side,
        release_offset=release_offset,
        high_offset=high_offset,
        reference_is_initial_pose=reference_is_initial_pose,
        orientation_goal=orientation_goal,
        orientation_axis=orientation_axis,
        orientation_align_to_runtime_uid=orientation_align_to_runtime_uid,
        hover_height=hover_height,
        upright_in_place=upright_in_place,
    )


def _validate_relative_placements(
    placements: tuple[_RelativePlacementStepSpec, ...],
) -> None:
    if not placements:
        raise ValueError("Object manipulation requires at least one manipulation.")
    moved_source_uids = [placement.moved_source_uid for placement in placements]
    if len(moved_source_uids) != len(set(moved_source_uids)):
        raise ValueError("Object manipulations must use distinct moved_object values.")
    intents = {placement.intent for placement in placements}
    if len(intents) > 1:
        raise ValueError("Mixed manipulation intents are not supported in v1.")
    if "coordinated_pickment" in intents and len(placements) != 1:
        raise ValueError("CoordinatedPickment supports exactly one shared object.")
    if len(placements) == 2:
        active_sides = {placement.active_side for placement in placements}
        if active_sides != {"left", "right"}:
            raise ValueError(
                "Dual-arm object manipulation requires one left arm and one right arm."
            )


def _should_axis_align_cube_by_default(
    *,
    intent: str,
    moved_object: _SceneObject,
    orientation_goal: str,
    orientation_reference: str,
    orientation_axis: str,
) -> bool:
    return (
        intent == "place_relative"
        and orientation_goal == "preserve"
        and orientation_reference == "none"
        and orientation_axis == "none"
        and _is_cube_like_object(moved_object)
    )


def _is_cube_like_object(obj: _SceneObject) -> bool:
    shape = obj.config.get("shape", {}) or {}
    mesh_path = str(shape.get("fpath", "")) if isinstance(shape, Mapping) else ""
    mesh_parts = Path(mesh_path.replace("\\", "/")).parts[-3:] if mesh_path else ()
    text = " ".join([obj.source_uid, _base_name(obj), *mesh_parts]).lower()
    return any(keyword in text for keyword in _CUBE_LIKE_KEYWORDS)


def _is_uprightable_object(obj: _SceneObject) -> bool:
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
    moved_object: _SceneObject,
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
        if placement.intent == "hold_hover":
            return f"Pick up `{placement.moved_runtime_uid}` and keep it hovering."
        return _default_relative_task_summary(
            placement.moved_runtime_uid,
            placement.reference_runtime_uid,
            placement.relation,
        )
    if all(placement.intent == "hold_hover" for placement in placements):
        held = ", ".join(placement.moved_runtime_uid for placement in placements)
        return f"Use both UR5 arms to pick up and hold hovering objects: {held}."
    placement_text = "; ".join(
        f"use the {placement.active_side} UR5 to move "
        f"`{placement.moved_runtime_uid}` "
        f"{_relative_relation_phrase(placement.relation)} "
        f"`{placement.reference_runtime_uid}`"
        for placement in placements
    )
    return f"Use both UR5 arms for object manipulation: {placement_text}."


def _default_relative_action_sketch(
    placements: Sequence[_RelativePlacementStepSpec],
) -> list[str]:
    if len(placements) == 1:
        placement = placements[0]
        if placement.intent == "hold_hover":
            return [
                f"grasp {placement.moved_runtime_uid}",
                "lift and keep the object hovering without release",
                "keep the gripper closed",
            ]
        return [
            f"grasp {placement.moved_runtime_uid}",
            (
                f"move above the {placement.relation} release pose relative to "
                f"{placement.reference_runtime_uid}"
            ),
            "place at the release pose with Place",
        ]
    sketch = ["grasp both moved objects with their assigned arms"]
    if all(placement.intent == "hold_hover" for placement in placements):
        sketch.append("keep both objects hovering with closed grippers")
        return sketch
    for placement in placements:
        sketch.extend(
            [
                (
                    f"use {placement.active_side}_arm to move "
                    f"{placement.moved_runtime_uid} above the release pose relative "
                    f"to {placement.reference_runtime_uid}"
                ),
                f"place {placement.moved_runtime_uid} with Place",
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
