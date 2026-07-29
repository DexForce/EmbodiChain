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

from embodichain.gen_sim.action_agent_pipeline.protocol.actions import (
    MAX_COORDINATED_PAYLOADS as _MAX_COORDINATED_PAYLOADS,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    RelativePlacementSpec,
    RelativePlacementStepSpec,
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _base_name,
    _is_container_like,
    _string_list,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_coordinated_spec import (
    _canonicalize_flat_coordinated_transport_entries,
    _coordinated_payload_entries,
    _coordinated_transport_entry,
    _normalize_coordinated_direction,
    _normalize_coordinated_terminal_behavior,
    _relative_forced_arm_sides,
    _with_coordinated_pickment_intent,
    _with_coordinated_transport_relation,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_intent import (
    _DEFAULT_HOVER_HEIGHT,
    _SIDE_RELATIONS,
    _normalize_hover_height,
    _normalize_manipulation_intent,
    _normalize_orientation_axis,
    _normalize_orientation_goal,
    _normalize_orientation_reference,
    _normalize_relative_arm,
    _normalize_relative_relation,
    _relative_primary_placement,
    _relative_relation_phrase,
    _relative_scene_runtime_uid_mapping,
    _resolve_relative_reference_source_uid,
    _resolve_rigid_source_uid,
    _should_upright_in_place,
    _validate_orientation_fields,
    _vector3,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _arm_side_for_position,
    _pick_table,
)
from embodichain.gen_sim.action_agent_pipeline.generation.spec_llm import (
    request_json_spec,
)

__all__ = [
    "_SIDE_RELATIONS",
    "_build_object_manipulation_spec_from_response",
    "_build_object_manipulation_spec_with_llm",
    "_build_relative_placement_spec_with_llm",
    "_call_object_manipulation_task_llm",
    "_normalize_relative_relation",
    "_relative_relation_phrase",
    "_relative_scene_runtime_uid_mapping",
]


def _build_relative_placement_spec_with_llm(
    *,
    scene_objects: list[SceneObject],
    project_name: str,
    task_description: str,
    model: str | None,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
    pose_sensitive_staging_z_delta: float,
    task_llm_caller: Callable[..., Mapping[str, Any]] | None = None,
) -> RelativePlacementSpec:
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

    _pick_table(background_objects)
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
                "description": str(obj.config.get("description", "")).strip(),
                "is_container_like": _is_container_like(obj),
                "mesh": obj.config.get("shape", {}).get("fpath"),
                "init_pos": obj.config.get("init_pos"),
            }
            for obj in scene_objects
        ],
        model=model,
    )
    return _build_object_manipulation_spec_from_response(
        response=response,
        scene_objects=scene_objects,
        task_description=task_description,
        release_offset_fn=release_offset_fn,
        staging_z_delta=staging_z_delta,
        pose_sensitive_staging_z_delta=pose_sensitive_staging_z_delta,
    )


def _build_object_manipulation_spec_from_response(
    *,
    response: Mapping[str, Any],
    scene_objects: list[SceneObject],
    task_description: str,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
    pose_sensitive_staging_z_delta: float,
) -> RelativePlacementSpec:
    """Build a deterministic manipulation spec from parsed model semantics."""
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
    scene_objects: list[SceneObject],
    project_name: str,
    task_description: str,
    model: str | None,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
    pose_sensitive_staging_z_delta: float,
    task_llm_caller: Callable[..., Mapping[str, Any]] | None = None,
) -> RelativePlacementSpec:
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
    # The template owns model-facing prose; this module remains authoritative
    # for schema normalization and all deterministic geometry decisions.
    return request_json_spec(
        template_name="relative_placement_spec.txt",
        usage_stage="config_generation.relative_task",
        project_name=project_name,
        task_description=task_description,
        scene_summary=scene_summary,
        model=model,
    )


def _call_object_manipulation_task_llm(
    *,
    project_name: str,
    task_description: str,
    scene_summary: list[dict[str, Any]],
    model: str | None,
) -> dict[str, Any]:
    # Keep the LLM boundary narrow: it selects semantic intent, while numeric
    # targets and runtime action contracts are computed and validated locally.
    return request_json_spec(
        template_name="object_manipulation_spec.txt",
        usage_stage="config_generation.object_manipulation_task",
        project_name=project_name,
        task_description=task_description,
        scene_summary=scene_summary,
        model=model,
    )


def _apply_relative_task_response(
    *,
    response: Mapping[str, Any],
    table_source_uid: str,
    scene_objects: list[SceneObject],
    rigid_objects: list[SceneObject],
    task_description: str,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
    pose_sensitive_staging_z_delta: float,
) -> RelativePlacementSpec:
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    runtime_uids = _relative_scene_runtime_uid_mapping(
        scene_objects,
        table_source_uid=table_source_uid,
    )

    top_level_entries = _relative_placement_entries(response)
    top_level_entries = _with_coordinated_pickment_intent(
        top_level_entries,
        task_description=task_description,
    )
    top_level_entries = _canonicalize_flat_coordinated_transport_entries(
        top_level_entries,
        rigid_objects=rigid_objects,
    )
    if len(top_level_entries) > 2:
        intents = [
            _normalize_manipulation_intent(entry.get("intent"))
            for entry in top_level_entries
        ]
        raise ValueError(
            "Object manipulation supports at most two independent arm actions; "
            "loaded coordinated transport must use one coordinated_pickment with "
            f"at most {_MAX_COORDINATED_PAYLOADS} payloads. Received intents: {intents}."
        )

    coordinated_entry = _coordinated_transport_entry(top_level_entries)
    payload_entries: list[Mapping[str, Any]] = []
    coordinated_direction: str | None = None
    coordinated_terminal_behavior: str | None = None
    if coordinated_entry is not None:
        payload_entries = _coordinated_payload_entries(
            coordinated_entry,
            by_uid=by_uid,
            rigid_objects=rigid_objects,
        )
        coordinated_direction = _normalize_coordinated_direction(
            coordinated_entry.get("direction")
        )
        coordinated_terminal_behavior = _normalize_coordinated_terminal_behavior(
            coordinated_entry.get("terminal_behavior"),
            task_description=task_description,
        )
        coordinated_entry = _with_coordinated_transport_relation(
            coordinated_entry,
            direction=coordinated_direction,
        )
        placement_entries = [*payload_entries, coordinated_entry]
    else:
        placement_entries = top_level_entries

    payload_forced_sides = _relative_forced_arm_sides(
        payload_entries if payload_entries else placement_entries,
        by_uid=by_uid,
        rigid_objects=rigid_objects,
    )
    forced_arm_sides = (
        [*payload_forced_sides, None]
        if coordinated_entry is not None
        else payload_forced_sides
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
    if coordinated_entry is None:
        placements = _order_relative_placements_by_dependency(placements)
    _validate_relative_placements(placements)

    summary = str(response.get("task_prompt_summary", "")).strip()
    if not summary:
        summary = _default_relative_plan_summary(placements)
    background_notes = str(response.get("basic_background_notes", "")).strip()
    action_sketch = _string_list(response.get("action_sketch"))
    if not action_sketch:
        action_sketch = _default_relative_action_sketch(placements)

    primary = _relative_primary_placement(placements)

    return RelativePlacementSpec(
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
        surface_clearance=primary.surface_clearance,
        coordinated_direction=coordinated_direction,
        coordinated_terminal_behavior=coordinated_terminal_behavior,
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


def _build_relative_placement_step(
    *,
    entry: Mapping[str, Any],
    by_uid: Mapping[str, SceneObject],
    scene_objects: list[SceneObject],
    rigid_objects: list[SceneObject],
    runtime_uids: Mapping[str, str],
    table_source_uid: str,
    task_description: str,
    forced_side: str | None,
    release_offset_fn: Callable[[str], Sequence[float]],
    staging_z_delta: float,
    pose_sensitive_staging_z_delta: float,
) -> RelativePlacementStepSpec:
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
        if (
            reference_is_initial_pose
            and relation not in _SIDE_RELATIONS
            and intent != "coordinated_pickment"
        ):
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

    return RelativePlacementStepSpec(
        intent=intent,
        moved_source_uid=moved_source_uid,
        reference_source_uid=reference_source_uid,
        moved_runtime_uid=moved_runtime_uid,
        reference_runtime_uid=reference_runtime_uid,
        relation=relation,
        active_side=active_side,
        release_offset=release_offset,
        high_offset=high_offset,
        arm_request=requested_side,
        reference_is_initial_pose=reference_is_initial_pose,
        orientation_goal=orientation_goal,
        orientation_axis=orientation_axis,
        orientation_align_to_runtime_uid=orientation_align_to_runtime_uid,
        hover_height=hover_height,
        upright_in_place=upright_in_place,
    )


def _validate_relative_placements(
    placements: tuple[RelativePlacementStepSpec, ...],
) -> None:
    if not placements:
        raise ValueError("Object manipulation requires at least one manipulation.")
    moved_source_uids = [placement.moved_source_uid for placement in placements]
    if len(moved_source_uids) != len(set(moved_source_uids)):
        raise ValueError("Object manipulations must use distinct moved_object values.")
    intents = {placement.intent for placement in placements}
    if intents == {"place_relative", "coordinated_pickment"}:
        coordinated = [
            placement
            for placement in placements
            if placement.intent == "coordinated_pickment"
        ]
        payloads = [
            placement
            for placement in placements
            if placement.intent == "place_relative"
        ]
        if len(coordinated) != 1 or not 1 <= len(payloads) <= _MAX_COORDINATED_PAYLOADS:
            raise ValueError(
                "Loaded CoordinatedPickment requires one shared object and one to "
                f"{_MAX_COORDINATED_PAYLOADS} payloads."
            )
        carrier_uid = coordinated[0].moved_source_uid
        if any(payload.reference_source_uid != carrier_uid for payload in payloads):
            raise ValueError("Every coordinated payload must target the shared object.")
        return
    if len(intents) > 1:
        raise ValueError("Mixed manipulation intents are not supported in v1.")
    if "coordinated_pickment" in intents and len(placements) != 1:
        raise ValueError("CoordinatedPickment supports exactly one shared object.")


def _order_relative_placements_by_dependency(
    placements: tuple[RelativePlacementStepSpec, ...],
) -> tuple[RelativePlacementStepSpec, ...]:
    """Order two placements so a moved reference object is placed first."""
    if len(placements) != 2:
        return placements
    first, second = placements
    first_depends_on_second = first.reference_source_uid == second.moved_source_uid
    second_depends_on_first = second.reference_source_uid == first.moved_source_uid
    if first_depends_on_second and second_depends_on_first:
        raise ValueError("Relative placements contain a cyclic object dependency.")
    if first_depends_on_second:
        return second, first
    return placements


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
    placements: Sequence[RelativePlacementStepSpec],
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
        return f"Use both robot arms to pick up and hold hovering objects: {held}."
    placement_text = "; ".join(
        f"use the {placement.active_side} arm to move "
        f"`{placement.moved_runtime_uid}` "
        f"{_relative_relation_phrase(placement.relation)} "
        f"`{placement.reference_runtime_uid}`"
        for placement in placements
    )
    return f"Use both robot arms for object manipulation: {placement_text}."


def _default_relative_action_sketch(
    placements: Sequence[RelativePlacementStepSpec],
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
