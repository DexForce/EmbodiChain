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

"""Prompt and agent-config builders for generated action-agent tasks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from embodichain.gen_sim.action_agent_pipeline.contracts import (
    ATOM_ACTIONS_FILENAME,
    BASIC_BACKGROUND_FILENAME,
    DUAL_ARM_NAME,
    LEFT_ARM_ACTION_KEY,
    RIGHT_ARM_ACTION_KEY,
    TASK_GRAPH_FILENAME,
    TASK_PROMPT_FILENAME,
)
from embodichain.gen_sim.action_agent_pipeline.defaults import (
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
    generation_defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.generation.nominal_graph import (
    NominalGraphStep,
    build_nominal_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    RobotProfile,
    resolve_robot_profile,
)
from embodichain.gen_sim.action_agent_pipeline.prompts.template_loader import (
    render_prompt_template,
)
from embodichain.gen_sim.action_agent_pipeline.semantics import (
    relative_relation_phrase as _canonical_relative_relation_phrase,
)

__all__ = [
    "make_agent_config",
    "make_arrangement_task_graph",
    "make_arrangement_atom_actions_prompt",
    "make_arrangement_basic_background",
    "make_arrangement_task_prompt",
    "make_relative_task_graph",
    "make_relative_atom_actions_prompt",
    "make_relative_basic_background",
    "make_relative_task_prompt",
    "make_stacking_task_graph",
    "make_stacking_atom_actions_prompt",
    "make_stacking_basic_background",
    "make_stacking_task_prompt",
]

_ACTION_DEFAULTS = generation_defaults_section("action")
_BASKET_LEFT_RELEASE_OFFSET_Y = float(_ACTION_DEFAULTS["basket_left_release_offset_y"])
_BASKET_RIGHT_RELEASE_OFFSET_Y = float(
    _ACTION_DEFAULTS["basket_right_release_offset_y"]
)
_PICKUP_LIFT_HEIGHT = float(_ACTION_DEFAULTS["pickup_lift_height"])
_PLACE_LIFT_HEIGHT = float(_ACTION_DEFAULTS["place_lift_height"])
_DIRECT_PLACE_CARTESIAN_WAYPOINT_COUNT = int(
    _ACTION_DEFAULTS["direct_place_cartesian_waypoint_count"]
)
_RELEASE_ONLY_PLACE_SAMPLE_INTERVAL = int(
    _ACTION_DEFAULTS["release_only_place_sample_interval"]
)
_EMPTY_HAND_RETREAT_SAMPLE_INTERVAL = int(
    _ACTION_DEFAULTS["empty_hand_retreat_sample_interval"]
)
_COORDINATED_MAX_GRASP_SEPARATION_ANGLE_TO_WORLD_Y_DEGREES = float(
    _ACTION_DEFAULTS["coordinated_max_grasp_separation_angle_to_world_y_degrees"]
)
_STACKING_DEFAULTS = generation_defaults_section("stacking")
_STACKING_NESTED_RELEASE_Z_OFFSET = float(_STACKING_DEFAULTS["nested_release_z_offset"])
_STACKING_SURFACE_CLEARANCE = float(_STACKING_DEFAULTS["clearance"])
_STACKING_MAX_APPROACH_RETRACT_Z = float(_STACKING_DEFAULTS["max_approach_retract_z"])
_SURFACE_RELEASE_Z_POLICY = "object_on_surface"
_SURFACE_RELEASE_CLEARANCE = DEFAULT_SURFACE_RELEASE_CLEARANCE
_USE_PLACEMENT_ALIGN_TO = object()
_RELATIVE_COORDINATE_CONVENTION = render_prompt_template(
    "relative_coordinate_convention.txt"
)


class _RelativePlacementLike(Protocol):
    intent: str
    active_side: str
    moved_runtime_uid: str
    moved_source_uid: str
    reference_runtime_uid: str
    reference_source_uid: str
    relation: str
    high_offset: tuple[float, float, float]
    release_offset: tuple[float, float, float]
    reference_is_initial_pose: bool
    high_position: Sequence[float] | None
    release_position: Sequence[float] | None
    orientation_goal: str
    orientation_axis: str
    orientation_align_to_runtime_uid: str | None
    hover_height: float
    upright_in_place: bool
    pickup_upright_direction: Sequence[float] | None
    pickup_rotate_upright: float | None
    surface_clearance: float


class _RelativeSpecLike(_RelativePlacementLike, Protocol):
    placements: Sequence[_RelativePlacementLike]
    task_prompt_summary: str
    task_description: str
    action_sketch: Sequence[str]
    basic_background_notes: str
    coordinated_direction: str | None
    coordinated_terminal_behavior: str | None


class _ArrangementStepLike(Protocol):
    source_uid: str
    runtime_uid: str
    slot_index: int
    active_side: str
    target_xy: Sequence[float]
    release_position: Sequence[float]
    high_position: Sequence[float]
    size_score: float | None
    color: str | None
    orientation_goal: str
    orientation_axis: str


class _ArrangementSpecLike(Protocol):
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    order_by: str
    order_direction: str
    axis: str
    anchor: str
    line_origin_xy: Sequence[float]
    spacing: float
    layout_clearance: float
    steps: Sequence[_ArrangementStepLike]


class _StackingStepLike(Protocol):
    source_uid: str
    runtime_uid: str
    layer_index: int
    active_side: str
    target_position: Sequence[float]
    high_position: Sequence[float]
    support_runtime_uid: str | None
    size_score: float | None
    color: str | None
    orientation_goal: str
    orientation_axis: str


class _StackingSpecLike(Protocol):
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    stack_mode: str
    order_by: str
    anchor: str
    anchor_xy: Sequence[float]
    anchor_source_uid: str | None
    anchor_runtime_uid: str | None
    steps: Sequence[_StackingStepLike]


def make_agent_config() -> dict[str, Any]:
    return {
        "TaskAgent": {
            "prompt_name": "generate_task_graph",
            "precomputed_task_graph": TASK_GRAPH_FILENAME,
        },
        "CompileAgent": {},
        "Agent": {
            "prompt_kwargs": {
                "task_prompt": {
                    "type": "text",
                    "name": TASK_PROMPT_FILENAME,
                },
                "basic_background": {
                    "type": "text",
                    "name": BASIC_BACKGROUND_FILENAME,
                },
                "atom_actions": {
                    "type": "text",
                    "name": ATOM_ACTIONS_FILENAME,
                },
            }
        },
    }


def _arm_action_slots(active_side: str) -> tuple[str, str, str]:
    """Return ``(active_arm, active_slot, inactive_slot)`` for one arm side.

    Every single-arm edge sequence derives its arm/slot names from the active
    side the same way; centralizing it keeps the inactive-slot complement
    ("left"->"right", "right"->"left") consistent across arrangement, stacking,
    and relative routes.
    """
    active_arm = f"{active_side}_arm"
    active_slot = f"{active_side}_arm_action"
    inactive_slot = f"{'right' if active_side == 'left' else 'left'}_arm_action"
    return active_arm, active_slot, inactive_slot


def _single_arm_post_release_blocks(
    active_arm: str,
    active_slot: str,
    inactive_slot: str,
    runtime_uid: str,
) -> list[tuple[str, Mapping[str, str | None]]]:
    """Return the in-place release, retreat, and home-return edge blocks.

    After the object reaches its release pose, every single-arm sequence
    performs the same three steps: open the gripper in place, retreat upward,
    and return the arm to its initial qpos. These blocks are identical across
    arrangement and stacking, so they live here to avoid divergence.
    """
    return [
        (
            f"Release `{runtime_uid}` in-place without moving the object pose",
            {
                active_slot: _format_release_only_place_spec(active_arm),
                inactive_slot: None,
            },
        ),
        (
            f"Retreat `{active_arm}` upward after release",
            {
                active_slot: _format_empty_hand_retreat_spec(active_arm),
                inactive_slot: None,
            },
        ),
        (
            f"Return `{active_arm}` to its initial pose",
            {
                active_slot: _format_initial_qpos_spec(
                    active_arm,
                    sample_interval=30,
                ),
                inactive_slot: None,
            },
        ),
    ]


def _format_runtime_object_registry(
    object_registry: Sequence[Mapping[str, Any]] | None,
) -> str:
    if not object_registry:
        return ""

    lines = []
    for item in object_registry:
        runtime_uid = str(item.get("runtime_uid", "")).strip()
        source_uid = str(item.get("source_uid", "")).strip()
        if not runtime_uid or not source_uid:
            continue
        role = str(item.get("source_role", item.get("role", ""))).strip()
        description = _one_line_registry_text(item.get("description", ""))
        role_text = f", role `{role}`" if role else ""
        description_text = (
            json.dumps(description, ensure_ascii=False)
            if description
            else '"No source description."'
        )
        lines.append(
            f"- runtime_uid `{runtime_uid}` maps to source_uid `{source_uid}`"
            f"{role_text}; description: {description_text}"
        )
    if not lines:
        return ""

    return (
        "\nRuntime object registry:\n" + "\n".join(lines) + "\n\nRegistry rules:\n"
        "- Descriptions are read-only semantic hints for identifying objects.\n"
        "- In every generated graph action, use only `runtime_uid` values as "
        "`obj_name`, `align_to`, `support`, `support_uid`, and object pose "
        "reference ids.\n"
        "- Do not copy `source_uid`, `description`, or registry metadata into "
        "the action JSON.\n"
    )


def _one_line_registry_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def make_arrangement_task_prompt(
    task_name: str,
    project_name: str,
    spec: _ArrangementSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    resolve_robot_profile(robot_profile)
    edge_count = sum(_arrangement_step_edge_count(step) for step in spec.steps)
    edge_index = 1
    step_blocks_list = []
    for step in spec.steps:
        step_blocks_list.append(_arrangement_step_prompt_block(edge_index, step))
        edge_index += _arrangement_step_edge_count(step)
    step_blocks = "\n\n".join(step_blocks_list)
    final_order = ", ".join(
        f"`{step.runtime_uid}` at slot {step.slot_index}"
        for step in sorted(spec.steps, key=lambda item: item.slot_index)
    )
    world_axis = _arrangement_world_axis(spec)
    return render_prompt_template(
        "arrangement_task.txt",
        task_name=task_name,
        task_prompt_summary=spec.task_prompt_summary,
        task_description=spec.task_description,
        axis=spec.axis,
        world_axis=world_axis,
        spatial_direction=spec.spatial_direction,
        anchor=spec.anchor,
        project_name=project_name,
        line_origin_xy=list(spec.line_origin_xy),
        spacing=f"{float(spec.spacing):.6g}",
        layout_clearance=f"{float(spec.layout_clearance):.6g}",
        order_by=spec.order_by,
        order_direction=spec.order_direction,
        category_order=list(spec.category_order),
        final_order=final_order,
        edge_count=edge_count,
        step_blocks=step_blocks,
    )


def make_arrangement_task_graph(
    task_name: str,
    spec: _ArrangementSpecLike,
) -> dict[str, Any]:
    steps = []
    for step in spec.steps:
        steps.extend(
            _nominal_step(title, actions)
            for title, actions in _arrangement_step_edge_blocks(step)
        )
    return build_nominal_task_graph(task_name=task_name, steps=steps)


def _arrangement_world_axis(spec: _ArrangementSpecLike) -> str:
    if len(spec.steps) >= 2:
        x_values = [float(step.target_xy[0]) for step in spec.steps]
        y_values = [float(step.target_xy[1]) for step in spec.steps]
        x_span = max(x_values) - min(x_values)
        y_span = max(y_values) - min(y_values)
        return "x" if x_span >= y_span else "y"
    if spec.axis == "world_x":
        return "x"
    return "y"


def _arrangement_step_edge_count(step: _ArrangementStepLike) -> int:
    return 6 if step.orientation_goal == "preserve" else 7


def _arrangement_step_edge_blocks(
    step: _ArrangementStepLike,
) -> list[tuple[str, Mapping[str, str | None]]]:
    active_arm, active_slot, inactive_slot = _arm_action_slots(step.active_side)
    high_preserve_spec = _format_pose_absolute_spec(
        active_arm,
        step.high_position,
        sample_interval=45,
        orientation_goal="preserve",
        orientation_axis="none",
    )
    release_move_spec = _format_pose_absolute_spec(
        active_arm,
        step.release_position,
        sample_interval=45,
        orientation_goal=step.orientation_goal,
        orientation_axis=step.orientation_axis,
        z_policy=_SURFACE_RELEASE_Z_POLICY,
        support="table",
        surface_clearance=_SURFACE_RELEASE_CLEARANCE,
    )
    blocks = [
        (
            f"Pick up `{step.runtime_uid}` for slot {step.slot_index}",
            {
                active_slot: _format_pick_up_spec(active_arm, step.runtime_uid),
                inactive_slot: None,
            },
        ),
        (
            f"Move `{step.runtime_uid}` to the high staging pose above slot "
            f"{step.slot_index} without changing orientation",
            {
                active_slot: high_preserve_spec,
                inactive_slot: None,
            },
        ),
    ]
    if step.orientation_goal != "preserve":
        blocks.append(
            (
                f"Align `{step.runtime_uid}` at the high staging pose to the "
                "configured arrangement axis",
                {
                    active_slot: _format_pose_absolute_spec(
                        active_arm,
                        step.high_position,
                        sample_interval=45,
                        orientation_goal=step.orientation_goal,
                        orientation_axis=step.orientation_axis,
                    ),
                    inactive_slot: None,
                },
            )
        )
    release_title = (
        f"Move `{step.runtime_uid}` down to the final release object pose "
        f"at slot {step.slot_index}"
        if step.orientation_goal != "preserve"
        else f"Move `{step.runtime_uid}` down to the final release object pose "
        f"at slot {step.slot_index} without changing orientation"
    )
    blocks.append(
        (
            release_title,
            {
                active_slot: release_move_spec,
                inactive_slot: None,
            },
        )
    )
    blocks.extend(
        _single_arm_post_release_blocks(
            active_arm, active_slot, inactive_slot, step.runtime_uid
        )
    )
    return blocks


def _arrangement_step_prompt_block(start_edge: int, step: _ArrangementStepLike) -> str:
    return _format_indexed_edge_blocks(
        _arrangement_step_edge_blocks(step),
        start_index=start_edge,
    )


def _format_indexed_edge_blocks(
    edge_blocks: Sequence[tuple[str, Mapping[str, str | None]]],
    *,
    start_index: int,
) -> str:
    formatted_blocks = []
    for index, (title, actions) in enumerate(edge_blocks, start=start_index):
        action_lines = "\n".join(
            f"   - {slot}: {action if action is not None else 'null'}"
            for slot, action in actions.items()
        )
        formatted_blocks.append(f"{index}. {title}:\n{action_lines}")
    return "\n\n".join(formatted_blocks)


def _robot_context(robot_profile: RobotProfile | str | None) -> str:
    return resolve_robot_profile(robot_profile).prompt_robot_context()


def make_arrangement_basic_background(
    project_name: str,
    spec: _ArrangementSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
    object_registry: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    notes = spec.basic_background_notes or (
        "No extra scene notes were provided by the config-stage LLM."
    )
    object_lines = "\n".join(
        _arrangement_object_background_line(step) for step in spec.steps
    )
    registry = _format_runtime_object_registry(object_registry)
    return render_prompt_template(
        "arrangement_background.txt",
        project_name=project_name,
        robot_display_name=profile.display_name,
        robot_context=_robot_context(profile),
        object_lines=object_lines,
        registry=registry,
        notes=notes,
    )


def _arrangement_object_background_line(step: _ArrangementStepLike) -> str:
    attrs = []
    if step.color:
        attrs.append(f"color={step.color}")
    if step.size_score is not None:
        attrs.append(f"size_score={float(step.size_score):.6g}")
    attr_text = f" ({', '.join(attrs)})" if attrs else ""
    return (
        f"- {step.runtime_uid}: source `{step.source_uid}`{attr_text}, "
        f"slot {step.slot_index} at xy={list(step.target_xy)}, "
        f"handled by {step.active_side}_arm."
    )


def make_arrangement_atom_actions_prompt(
    spec: _ArrangementSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    blocks = "\n\n".join(_arrangement_atom_action_block(step) for step in spec.steps)
    return render_prompt_template(
        "arrangement_actions.txt",
        robot_display_name=profile.display_name,
        blocks=blocks,
    )


def _arrangement_atom_action_block(step: _ArrangementStepLike) -> str:
    active_arm = f"{step.active_side}_arm"
    high_preserve_spec = _format_pose_absolute_spec(
        active_arm,
        step.high_position,
        sample_interval=45,
        orientation_goal="preserve",
        orientation_axis="none",
    )
    release_move_spec = _format_pose_absolute_spec(
        active_arm,
        step.release_position,
        sample_interval=45,
        orientation_goal=step.orientation_goal,
        orientation_axis=step.orientation_axis,
        z_policy=_SURFACE_RELEASE_Z_POLICY,
        support="table",
        surface_clearance=_SURFACE_RELEASE_CLEARANCE,
    )
    if step.orientation_goal == "preserve":
        return render_prompt_template(
            "arrangement_action_block_preserve.txt",
            runtime_uid=step.runtime_uid,
            slot_index=step.slot_index,
            pickup_spec=_format_pick_up_spec(active_arm, step.runtime_uid),
            high_preserve_spec=high_preserve_spec,
            release_move_spec=release_move_spec,
            release_spec=_format_release_only_place_spec(active_arm),
            retreat_spec=_format_empty_hand_retreat_spec(active_arm),
            return_spec=_format_initial_qpos_spec(active_arm, sample_interval=30),
        )
    high_align_spec = _format_pose_absolute_spec(
        active_arm,
        step.high_position,
        sample_interval=45,
        orientation_goal=step.orientation_goal,
        orientation_axis=step.orientation_axis,
    )
    return render_prompt_template(
        "arrangement_action_block_align.txt",
        runtime_uid=step.runtime_uid,
        slot_index=step.slot_index,
        pickup_spec=_format_pick_up_spec(active_arm, step.runtime_uid),
        high_preserve_spec=high_preserve_spec,
        high_align_spec=high_align_spec,
        release_move_spec=release_move_spec,
        release_spec=_format_release_only_place_spec(active_arm),
        retreat_spec=_format_empty_hand_retreat_spec(active_arm),
        return_spec=_format_initial_qpos_spec(active_arm, sample_interval=30),
    )


def make_stacking_task_prompt(
    task_name: str,
    project_name: str,
    spec: _StackingSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    resolve_robot_profile(robot_profile)
    edge_count = sum(_stacking_step_edge_count(step) for step in spec.steps)
    edge_index = 1
    step_blocks_list = []
    for step in spec.steps:
        step_blocks_list.append(
            _stacking_step_prompt_block(
                edge_index,
                step,
                object_anchored=spec.anchor == "object",
                stack_mode=spec.stack_mode,
            )
        )
        edge_index += _stacking_step_edge_count(step)
    step_blocks = "\n\n".join(step_blocks_list)
    stack_order = ", ".join(
        f"`{step.runtime_uid}` layer {step.layer_index}" for step in spec.steps
    )
    anchor_description = (
        f"object `{spec.anchor_runtime_uid}` at its current runtime pose"
        if spec.anchor == "object"
        else f"`{spec.anchor}` at xy `{list(spec.anchor_xy)}`"
    )
    final_target_rule = (
        "Use the exact object-referenced target_object_pose JSON specs shown "
        "above so every layer follows its direct support's current pose."
        if spec.anchor == "object"
        else "Use the exact absolute target_object_pose JSON specs shown above; "
        "do not rewrite them."
    )
    return render_prompt_template(
        "stacking_task.txt",
        task_name=task_name,
        task_prompt_summary=spec.task_prompt_summary,
        task_description=spec.task_description,
        stack_mode=spec.stack_mode,
        anchor_description=anchor_description,
        project_name=project_name,
        order_by=spec.order_by,
        stack_order=stack_order,
        edge_count=edge_count,
        step_blocks=step_blocks,
        final_target_rule=final_target_rule,
    )


def make_stacking_task_graph(
    task_name: str,
    spec: _StackingSpecLike,
) -> dict[str, Any]:
    steps = []
    for step in spec.steps:
        steps.extend(
            _nominal_step(title, actions)
            for title, actions in _stacking_step_edge_blocks(
                step,
                object_anchored=spec.anchor == "object",
                stack_mode=spec.stack_mode,
            )
        )
    return build_nominal_task_graph(task_name=task_name, steps=steps)


def _stacking_step_edge_count(step: _StackingStepLike) -> int:
    return 3 if step.orientation_goal == "preserve" else 7


def _stacking_step_edge_blocks(
    step: _StackingStepLike,
    *,
    object_anchored: bool,
    stack_mode: str,
) -> list[tuple[str, Mapping[str, str | None]]]:
    active_arm, active_slot, inactive_slot = _arm_action_slots(step.active_side)
    high_preserve_spec = _format_pose_absolute_spec(
        active_arm,
        step.high_position,
        sample_interval=45,
        orientation_goal="preserve",
        orientation_axis="none",
    )
    release_move_spec = _format_pose_absolute_spec(
        active_arm,
        step.target_position,
        sample_interval=45,
        orientation_goal=step.orientation_goal,
        orientation_axis=step.orientation_axis,
    )
    if step.orientation_goal == "preserve":
        return [
            (
                f"Pick up `{step.runtime_uid}` for stack layer {step.layer_index}",
                {
                    active_slot: _format_pick_up_spec(active_arm, step.runtime_uid),
                    inactive_slot: None,
                },
            ),
            (
                f"Place `{step.runtime_uid}` directly at the final stack pose "
                "without changing orientation",
                {
                    active_slot: _format_stacking_place_spec(
                        active_arm,
                        step,
                        object_anchored=object_anchored,
                        stack_mode=stack_mode,
                    ),
                    inactive_slot: None,
                },
            ),
            (
                f"Return `{active_arm}` to its initial pose",
                {
                    active_slot: _format_initial_qpos_spec(
                        active_arm, sample_interval=30
                    ),
                    inactive_slot: None,
                },
            ),
        ]
    blocks = [
        (
            f"Pick up `{step.runtime_uid}` for stack layer {step.layer_index}",
            {
                active_slot: _format_pick_up_spec(active_arm, step.runtime_uid),
                inactive_slot: None,
            },
        ),
        (
            f"Move `{step.runtime_uid}` to the high staging pose without "
            "changing orientation",
            {
                active_slot: high_preserve_spec,
                inactive_slot: None,
            },
        ),
    ]
    if step.orientation_goal != "preserve":
        blocks.append(
            (
                f"Align `{step.runtime_uid}` at the high staging pose if the "
                "spec requires it",
                {
                    active_slot: _format_pose_absolute_spec(
                        active_arm,
                        step.high_position,
                        sample_interval=45,
                        orientation_goal=step.orientation_goal,
                        orientation_axis=step.orientation_axis,
                    ),
                    inactive_slot: None,
                },
            )
        )
    release_title = (
        f"Move `{step.runtime_uid}` down to the final stack object pose"
        if step.orientation_goal != "preserve"
        else f"Move `{step.runtime_uid}` down to the final stack object pose "
        "without changing orientation"
    )
    blocks.append(
        (
            release_title,
            {
                active_slot: release_move_spec,
                inactive_slot: None,
            },
        )
    )
    blocks.extend(
        _single_arm_post_release_blocks(
            active_arm, active_slot, inactive_slot, step.runtime_uid
        )
    )
    return blocks


def _stacking_step_prompt_block(
    start_edge: int,
    step: _StackingStepLike,
    *,
    object_anchored: bool,
    stack_mode: str,
) -> str:
    active_arm, active_slot, inactive_slot = _arm_action_slots(step.active_side)
    high_preserve_spec = _format_pose_absolute_spec(
        active_arm,
        step.high_position,
        sample_interval=45,
        orientation_goal="preserve",
        orientation_axis="none",
    )
    if step.orientation_goal == "preserve":
        high_oriented_spec = high_preserve_spec
    else:
        high_oriented_spec = _format_pose_absolute_spec(
            active_arm,
            step.high_position,
            sample_interval=45,
            orientation_goal=step.orientation_goal,
            orientation_axis=step.orientation_axis,
        )
    release_move_spec = _format_pose_absolute_spec(
        active_arm,
        step.target_position,
        sample_interval=45,
        orientation_goal=step.orientation_goal,
        orientation_axis=step.orientation_axis,
    )
    if step.orientation_goal == "preserve":
        return render_prompt_template(
            "stacking_step_preserve.txt",
            start_edge=start_edge,
            edge_place=start_edge + 1,
            edge_return=start_edge + 2,
            runtime_uid=step.runtime_uid,
            layer_index=step.layer_index,
            active_arm=active_arm,
            active_slot=active_slot,
            inactive_slot=inactive_slot,
            pickup_spec=_format_pick_up_spec(active_arm, step.runtime_uid),
            place_spec=_format_stacking_place_spec(
                active_arm,
                step,
                object_anchored=object_anchored,
                stack_mode=stack_mode,
            ),
            return_spec=_format_initial_qpos_spec(active_arm, sample_interval=30),
        )
    return render_prompt_template(
        "stacking_step_oriented.txt",
        start_edge=start_edge,
        edge_high=start_edge + 1,
        edge_align=start_edge + 2,
        edge_down=start_edge + 3,
        edge_release=start_edge + 4,
        edge_retreat=start_edge + 5,
        edge_return=start_edge + 6,
        runtime_uid=step.runtime_uid,
        layer_index=step.layer_index,
        active_arm=active_arm,
        active_slot=active_slot,
        inactive_slot=inactive_slot,
        pickup_spec=_format_pick_up_spec(active_arm, step.runtime_uid),
        high_preserve_spec=high_preserve_spec,
        high_oriented_spec=high_oriented_spec,
        release_move_spec=release_move_spec,
        release_spec=_format_release_only_place_spec(active_arm),
        retreat_spec=_format_empty_hand_retreat_spec(active_arm),
        return_spec=_format_initial_qpos_spec(active_arm, sample_interval=30),
    )


def make_stacking_basic_background(
    project_name: str,
    spec: _StackingSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
    object_registry: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    notes = spec.basic_background_notes or (
        "No extra scene notes were provided by the config-stage LLM."
    )
    object_lines = "\n".join(
        _stacking_object_background_line(step) for step in spec.steps
    )
    registry = _format_runtime_object_registry(object_registry)
    return render_prompt_template(
        "stacking_background.txt",
        project_name=project_name,
        robot_display_name=profile.display_name,
        robot_context=_robot_context(profile),
        stack_mode=spec.stack_mode,
        anchor=spec.anchor,
        anchor_xy=list(spec.anchor_xy),
        object_lines=object_lines,
        registry=registry,
        notes=notes,
    )


def _stacking_object_background_line(step: _StackingStepLike) -> str:
    attrs = []
    if step.color:
        attrs.append(f"color={step.color}")
    if step.size_score is not None:
        attrs.append(f"size_score={float(step.size_score):.6g}")
    attr_text = f" ({', '.join(attrs)})" if attrs else ""
    support = step.support_runtime_uid or "table"
    return (
        f"- {step.runtime_uid}: source `{step.source_uid}`{attr_text}, "
        f"layer {step.layer_index}, support `{support}`, "
        f"target_position={list(step.target_position)}, "
        f"handled by {step.active_side}_arm."
    )


def make_stacking_atom_actions_prompt(
    spec: _StackingSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    blocks = "\n\n".join(
        _stacking_atom_action_block(
            step,
            object_anchored=spec.anchor == "object",
            stack_mode=spec.stack_mode,
        )
        for step in spec.steps
    )
    return render_prompt_template(
        "stacking_actions.txt",
        robot_display_name=profile.display_name,
        blocks=blocks,
    )


def _stacking_atom_action_block(
    step: _StackingStepLike,
    *,
    object_anchored: bool,
    stack_mode: str,
) -> str:
    active_arm = f"{step.active_side}_arm"
    high_oriented_spec = _format_pose_absolute_spec(
        active_arm,
        step.high_position,
        sample_interval=45,
        orientation_goal=step.orientation_goal,
        orientation_axis=step.orientation_axis,
    )
    if step.orientation_goal == "preserve":
        return render_prompt_template(
            "stacking_action_block_preserve.txt",
            runtime_uid=step.runtime_uid,
            layer_index=step.layer_index,
            pickup_spec=_format_pick_up_spec(active_arm, step.runtime_uid),
            place_spec=_format_stacking_place_spec(
                active_arm,
                step,
                object_anchored=object_anchored,
                stack_mode=stack_mode,
            ),
            return_spec=_format_initial_qpos_spec(active_arm, sample_interval=30),
        )
    return render_prompt_template(
        "stacking_action_block_oriented.txt",
        runtime_uid=step.runtime_uid,
        layer_index=step.layer_index,
        pickup_spec=_format_pick_up_spec(active_arm, step.runtime_uid),
        high_preserve_spec=_format_pose_absolute_spec(
            active_arm,
            step.high_position,
            sample_interval=45,
            orientation_goal="preserve",
            orientation_axis="none",
        ),
        high_oriented_spec=high_oriented_spec,
        final_pose_spec=_format_pose_absolute_spec(
            active_arm,
            step.target_position,
            sample_interval=45,
            orientation_goal=step.orientation_goal,
            orientation_axis=step.orientation_axis,
        ),
        release_spec=_format_release_only_place_spec(active_arm),
        retreat_spec=_format_empty_hand_retreat_spec(active_arm),
        return_spec=_format_initial_qpos_spec(active_arm, sample_interval=30),
    )


def make_relative_task_graph(
    task_name: str,
    spec: _RelativeSpecLike,
) -> dict[str, Any]:
    if spec.intent == "coordinated_pickment":
        steps = _coordinated_pickment_graph_steps(spec)
    elif spec.intent == "hold_hover":
        steps = _hold_hover_graph_steps(spec)
    elif len(spec.placements) > 1:
        steps = _dual_relative_graph_steps(spec)
    else:
        steps = _single_relative_graph_steps(spec)
    return build_nominal_task_graph(task_name=task_name, steps=steps)


def make_relative_task_prompt(
    task_name: str,
    project_name: str,
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    if spec.intent == "coordinated_pickment":
        return _make_coordinated_pickment_task_prompt(
            task_name,
            project_name,
            spec,
            robot_profile=profile,
        )
    if spec.intent == "hold_hover":
        return _make_hold_hover_task_prompt(
            task_name,
            project_name,
            spec,
            robot_profile=profile,
        )
    if len(spec.placements) > 1:
        return _make_dual_relative_task_prompt(
            task_name,
            project_name,
            spec,
            robot_profile=profile,
        )

    active_arm = f"{spec.active_side}_arm"
    inactive_slot = (
        RIGHT_ARM_ACTION_KEY if spec.active_side == "left" else LEFT_ARM_ACTION_KEY
    )
    active_slot = f"{spec.active_side}_arm_action"
    action_sketch = _format_action_sketch(spec.action_sketch)
    pick_spec = _format_pick_up_spec(
        active_arm,
        spec.moved_runtime_uid,
        pickup_upright_direction=spec.pickup_upright_direction,
        pickup_rotate_upright=spec.pickup_rotate_upright,
    )
    initial_spec = _format_initial_qpos_spec(active_arm, sample_interval=30)
    reference_line = _relative_reference_line(spec)
    final_planning_rule = _relative_final_planning_rule(project_name, spec)
    release_step_label = _relative_pose_step_label(spec, "release")
    pose_sensitive = _is_pose_sensitive_placement(spec)
    if pose_sensitive:
        release_move_spec = _format_relative_pose_spec(
            active_arm,
            spec,
            pose_kind="release",
            sample_interval=45,
        )
        place_spec = _format_release_only_place_spec(active_arm)
        retreat_spec = _format_empty_hand_retreat_spec(active_arm)
        edge_count = 5
        high_instruction = render_prompt_template(
            "relative_single_steps_oriented.txt",
            release_step_label=release_step_label,
            active_slot=active_slot,
            inactive_slot=inactive_slot,
            release_move_spec=release_move_spec,
            place_spec=place_spec,
            retreat_spec=retreat_spec,
            initial_spec=initial_spec,
        )
        release_rule = (
            "For this pose-sensitive placement, use exactly one `MoveHeldObject` "
            "to move directly to the final release object pose while applying the "
            "requested orientation. Do not add staging or intermediate moves. Use "
            "the exact relative-zero release-only `Place` spec shown below."
        )
    else:
        place_spec = _format_direct_relative_place_spec(active_arm, spec)
        edge_count = 3
        high_instruction = render_prompt_template(
            "relative_single_steps_preserve.txt",
            release_step_label=release_step_label,
            active_slot=active_slot,
            inactive_slot=inactive_slot,
            place_spec=place_spec,
            initial_spec=initial_spec,
        )
        release_rule = (
            "This orientation-preserving placement must use the object-aware "
            "`Place(target_object_pose=...)` spec shown below directly after "
            "`PickUp`; do not add `MoveHeldObject` or a release-only Place edge."
        )
    return render_prompt_template(
        "relative_single_task.txt",
        task_name=task_name,
        task_prompt_summary=spec.task_prompt_summary,
        task_description=spec.task_description,
        action_sketch=action_sketch,
        moved_runtime_uid=spec.moved_runtime_uid,
        moved_source_uid=spec.moved_source_uid,
        reference_line=reference_line,
        relation=spec.relation,
        relation_phrase=_relative_relation_phrase(spec.relation),
        reference_runtime_uid=spec.reference_runtime_uid,
        active_arm=active_arm,
        active_slot=active_slot,
        inactive_slot=inactive_slot,
        coordinate_convention=_RELATIVE_COORDINATE_CONVENTION,
        edge_count=edge_count,
        release_rule=release_rule,
        pick_spec=pick_spec,
        high_instruction=high_instruction,
        final_planning_rule=final_planning_rule,
    )


def _single_relative_graph_steps(
    spec: _RelativeSpecLike,
) -> list[NominalGraphStep]:
    active_arm = f"{spec.active_side}_arm"
    inactive_slot = (
        RIGHT_ARM_ACTION_KEY if spec.active_side == "left" else LEFT_ARM_ACTION_KEY
    )
    active_slot = f"{spec.active_side}_arm_action"
    pick_spec = _format_pick_up_spec(
        active_arm,
        spec.moved_runtime_uid,
        pickup_upright_direction=spec.pickup_upright_direction,
        pickup_rotate_upright=spec.pickup_rotate_upright,
    )
    initial_spec = _format_initial_qpos_spec(active_arm, sample_interval=30)
    release_step_label = _relative_pose_step_label(spec, "release")

    edge_blocks: list[tuple[str, Mapping[str, str | None]]] = [
        (
            "Pick up the moved object",
            {
                active_slot: pick_spec,
                inactive_slot: None,
            },
        )
    ]
    if not _is_pose_sensitive_placement(spec):
        edge_blocks.extend(
            [
                (
                    f"Move directly to the {release_step_label} object pose, "
                    "release, and retract without rotating",
                    {
                        active_slot: _format_direct_relative_place_spec(
                            active_arm, spec
                        ),
                        inactive_slot: None,
                    },
                ),
                (
                    "Return the active arm to its initial pose",
                    {
                        active_slot: initial_spec,
                        inactive_slot: None,
                    },
                ),
            ]
        )
        return [_nominal_step(title, actions) for title, actions in edge_blocks]

    edge_blocks.extend(
        [
            (
                f"Move the held object directly to the {release_step_label} object "
                "pose while applying the requested orientation",
                {
                    active_slot: _format_relative_pose_spec(
                        active_arm,
                        spec,
                        pose_kind="release",
                        sample_interval=45,
                    ),
                    inactive_slot: None,
                },
            ),
            (
                "Release the held object in-place without moving the object pose",
                {
                    active_slot: _format_release_only_place_spec(active_arm),
                    inactive_slot: None,
                },
            ),
            (
                "Retreat the now-empty end-effector upward",
                {
                    active_slot: _format_empty_hand_retreat_spec(active_arm),
                    inactive_slot: None,
                },
            ),
            (
                "Return the active arm to its initial pose",
                {
                    active_slot: initial_spec,
                    inactive_slot: None,
                },
            ),
        ]
    )
    return [_nominal_step(title, actions) for title, actions in edge_blocks]


def _make_coordinated_pickment_task_prompt(
    task_name: str,
    project_name: str,
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    resolve_robot_profile(robot_profile)
    if spec.coordinated_terminal_behavior is not None:
        graph = make_relative_task_graph(task_name, spec)
        return render_prompt_template(
            "coordinated_transport_task.txt",
            task_name=task_name,
            task_prompt_summary=spec.task_prompt_summary,
            task_description=spec.task_description,
            coordinated_direction=spec.coordinated_direction,
            coordinated_terminal_behavior=spec.coordinated_terminal_behavior,
            graph_json=json.dumps(graph, ensure_ascii=False, indent=2),
        )
    action_sketch = _format_action_sketch(spec.action_sketch)
    action_spec = _format_coordinated_pickment_spec(spec)
    left_release_spec = _format_gripper_spec(
        "left_arm",
        "open",
        sample_interval=10,
        post_hold_steps=20,
    )
    right_release_spec = _format_gripper_spec(
        "right_arm",
        "open",
        sample_interval=10,
        post_hold_steps=20,
    )
    left_initial_spec = _format_initial_qpos_spec("left_arm", sample_interval=30)
    right_initial_spec = _format_initial_qpos_spec("right_arm", sample_interval=30)
    final_planning_rule = _relative_final_planning_rule(project_name, spec)
    return render_prompt_template(
        "coordinated_pickment_task.txt",
        task_name=task_name,
        task_prompt_summary=spec.task_prompt_summary,
        task_description=spec.task_description,
        action_sketch=action_sketch,
        moved_runtime_uid=spec.moved_runtime_uid,
        moved_source_uid=spec.moved_source_uid,
        relation=spec.relation,
        relation_phrase=_relative_relation_phrase(spec.relation),
        reference_runtime_uid=spec.reference_runtime_uid,
        coordinate_convention=_RELATIVE_COORDINATE_CONVENTION,
        action_spec=action_spec,
        left_release_spec=left_release_spec,
        right_release_spec=right_release_spec,
        left_initial_spec=left_initial_spec,
        right_initial_spec=right_initial_spec,
        final_planning_rule=final_planning_rule,
    )


def _coordinated_pickment_graph_steps(
    spec: _RelativeSpecLike,
) -> list[NominalGraphStep]:
    if spec.coordinated_terminal_behavior is not None:
        carrier = _coordinated_carrier_placement(spec)
        payloads = [
            placement
            for placement in spec.placements
            if placement.intent == "place_relative"
        ]
        steps: list[NominalGraphStep] = []
        for payload in payloads:
            steps.extend(_single_relative_graph_steps(payload))
        steps.append(
            _nominal_step(
                f"Coordinated lift and transport `{carrier.moved_runtime_uid}`",
                {
                    LEFT_ARM_ACTION_KEY: _format_coordinated_pickment_spec(
                        carrier,
                        payload_runtime_uids=[
                            payload.moved_runtime_uid for payload in payloads
                        ],
                        target_hover=True,
                        hold_steps=20,
                    ),
                    RIGHT_ARM_ACTION_KEY: None,
                },
            )
        )
        if spec.coordinated_terminal_behavior == "hold":
            return steps
        steps.extend(
            [
                _nominal_step(
                    f"Lower `{carrier.moved_runtime_uid}` vertically onto the support",
                    {
                        LEFT_ARM_ACTION_KEY: _format_relative_eef_move_spec(
                            "left_arm",
                            offset=[0.0, 0.0, -float(carrier.hover_height)],
                            sample_interval=50,
                            post_hold_steps=20,
                        ),
                        RIGHT_ARM_ACTION_KEY: _format_relative_eef_move_spec(
                            "right_arm",
                            offset=[0.0, 0.0, -float(carrier.hover_height)],
                            sample_interval=50,
                            post_hold_steps=20,
                        ),
                    },
                ),
                _nominal_step(
                    f"Release `{carrier.moved_runtime_uid}` from both grippers",
                    {
                        LEFT_ARM_ACTION_KEY: _format_gripper_spec(
                            "left_arm", "open", sample_interval=10, post_hold_steps=20
                        ),
                        RIGHT_ARM_ACTION_KEY: _format_gripper_spec(
                            "right_arm", "open", sample_interval=10, post_hold_steps=20
                        ),
                    },
                ),
                _nominal_step(
                    "Retreat both empty arms vertically",
                    {
                        LEFT_ARM_ACTION_KEY: _format_empty_hand_retreat_spec(
                            "left_arm"
                        ),
                        RIGHT_ARM_ACTION_KEY: _format_empty_hand_retreat_spec(
                            "right_arm"
                        ),
                    },
                ),
                _nominal_step(
                    "Return both empty arms to their initial poses",
                    {
                        LEFT_ARM_ACTION_KEY: _format_initial_qpos_spec(
                            "left_arm", sample_interval=30
                        ),
                        RIGHT_ARM_ACTION_KEY: _format_initial_qpos_spec(
                            "right_arm", sample_interval=30
                        ),
                    },
                ),
            ]
        )
        return steps
    return [
        _nominal_step(
            f"Coordinated pick and move `{spec.moved_runtime_uid}`",
            {
                LEFT_ARM_ACTION_KEY: _format_coordinated_pickment_spec(spec),
                RIGHT_ARM_ACTION_KEY: None,
            },
        ),
        _nominal_step(
            f"Release `{spec.moved_runtime_uid}` from both grippers",
            {
                LEFT_ARM_ACTION_KEY: _format_gripper_spec(
                    "left_arm",
                    "open",
                    sample_interval=10,
                    post_hold_steps=20,
                ),
                RIGHT_ARM_ACTION_KEY: _format_gripper_spec(
                    "right_arm",
                    "open",
                    sample_interval=10,
                    post_hold_steps=20,
                ),
            },
        ),
        _nominal_step(
            "Return both empty arms to their initial poses",
            {
                LEFT_ARM_ACTION_KEY: _format_initial_qpos_spec(
                    "left_arm",
                    sample_interval=30,
                ),
                RIGHT_ARM_ACTION_KEY: _format_initial_qpos_spec(
                    "right_arm",
                    sample_interval=30,
                ),
            },
        ),
    ]


def _coordinated_carrier_placement(
    spec: _RelativeSpecLike,
) -> _RelativePlacementLike:
    return next(
        placement
        for placement in spec.placements
        if placement.intent == "coordinated_pickment"
    )


def _make_dual_relative_task_prompt(
    task_name: str,
    project_name: str,
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    if spec.intent == "hold_hover":
        return _make_hold_hover_task_prompt(
            task_name,
            project_name,
            spec,
            robot_profile=profile,
        )
    first, second = spec.placements
    first_slot = f"{first.active_side}_arm_action"
    second_slot = f"{second.active_side}_arm_action"
    action_sketch = _format_action_sketch(spec.action_sketch)
    first_reference_line = _relative_reference_line(first)
    second_reference_line = _relative_reference_line(second)
    final_planning_rule = _dual_relative_final_planning_rule(project_name, spec)
    edge_blocks = _dual_relative_edge_blocks(spec)
    edge_count = len(edge_blocks)
    numbered_edges = _format_numbered_edge_blocks(edge_blocks)
    release_rule = _dual_relative_release_rule(spec)
    return render_prompt_template(
        "relative_dual_task.txt",
        task_name=task_name,
        task_prompt_summary=spec.task_prompt_summary,
        task_description=spec.task_description,
        action_sketch=action_sketch,
        first_slot=first_slot,
        first_moved_runtime_uid=first.moved_runtime_uid,
        first_moved_source_uid=first.moved_source_uid,
        second_slot=second_slot,
        second_moved_runtime_uid=second.moved_runtime_uid,
        second_moved_source_uid=second.moved_source_uid,
        first_reference_line=first_reference_line,
        first_relation=first.relation,
        first_relation_phrase=_relative_relation_phrase(first.relation),
        first_reference_runtime_uid=first.reference_runtime_uid,
        second_reference_line=second_reference_line,
        second_relation=second.relation,
        second_relation_phrase=_relative_relation_phrase(second.relation),
        second_reference_runtime_uid=second.reference_runtime_uid,
        coordinate_convention=_RELATIVE_COORDINATE_CONVENTION,
        edge_count=edge_count,
        release_rule=release_rule,
        numbered_edges=numbered_edges,
        final_planning_rule=final_planning_rule,
    )


def _dual_relative_graph_steps(spec: _RelativeSpecLike) -> list[NominalGraphStep]:
    edge_blocks = _dual_relative_edge_blocks(spec)
    return [_nominal_step(title, actions) for title, actions in edge_blocks]


def _dual_relative_edge_blocks(
    spec: _RelativeSpecLike,
) -> list[tuple[str, Mapping[str, str | None]]]:
    first, second = spec.placements
    if _uses_serial_dual_sequence(spec):
        return _serial_relative_edge_blocks(spec)
    first_arm = f"{first.active_side}_arm"
    second_arm = f"{second.active_side}_arm"
    first_slot = f"{first.active_side}_arm_action"
    second_slot = f"{second.active_side}_arm_action"
    first_pick_spec = _format_pick_up_spec(
        first_arm,
        first.moved_runtime_uid,
        pickup_upright_direction=first.pickup_upright_direction,
        pickup_rotate_upright=first.pickup_rotate_upright,
    )
    second_pick_spec = _format_pick_up_spec(
        second_arm,
        second.moved_runtime_uid,
        pickup_upright_direction=second.pickup_upright_direction,
        pickup_rotate_upright=second.pickup_rotate_upright,
    )
    second_close_spec = _format_gripper_spec(
        second_arm,
        "close",
        sample_interval=10,
    )
    first_initial_spec = _format_initial_qpos_spec(
        first_arm,
        sample_interval=30,
    )
    second_initial_spec = _format_initial_qpos_spec(
        second_arm,
        sample_interval=30,
    )
    first_release_edges = _dual_relative_release_edge_blocks(
        placement=first,
        active_arm=first_arm,
        active_slot=first_slot,
        waiting_slot=second_slot,
        waiting_action=second_close_spec,
    )
    second_release_edges = _dual_relative_release_edge_blocks(
        placement=second,
        active_arm=second_arm,
        active_slot=second_slot,
        waiting_slot=first_slot,
        waiting_action=None,
    )
    edge_blocks = [
        (
            "Pick up both moved objects simultaneously",
            {
                first_slot: first_pick_spec,
                second_slot: second_pick_spec,
            },
        )
    ]
    edge_blocks.extend(first_release_edges)
    edge_blocks.append(
        (
            f"Return `{first_arm}` to its initial pose while `{second_arm}` "
            f"keeps holding `{second.moved_runtime_uid}`",
            {
                first_slot: first_initial_spec,
                second_slot: second_close_spec,
            },
        )
    )
    edge_blocks.extend(second_release_edges)
    edge_blocks.append(
        (
            f"Return `{second_arm}` to its initial pose",
            {
                first_slot: None,
                second_slot: second_initial_spec,
            },
        )
    )
    return edge_blocks


def _uses_serial_dual_sequence(spec: _RelativeSpecLike) -> bool:
    """Return whether placement dependencies require sequential execution."""
    first, second = spec.placements
    return (
        second.reference_source_uid == first.moved_source_uid
        or first.active_side == second.active_side
        or all(
            getattr(placement, "upright_in_place", False)
            for placement in spec.placements
        )
    )


def _serial_relative_edge_blocks(
    spec: _RelativeSpecLike,
) -> list[tuple[str, Mapping[str, str | None]]]:
    edge_blocks: list[tuple[str, Mapping[str, str | None]]] = []
    for placement in spec.placements:
        active_arm = f"{placement.active_side}_arm"
        active_slot = f"{placement.active_side}_arm_action"
        inactive_slot = (
            RIGHT_ARM_ACTION_KEY
            if placement.active_side == "left"
            else LEFT_ARM_ACTION_KEY
        )
        edge_blocks.append(
            (
                f"Pick up `{placement.moved_runtime_uid}`",
                {
                    active_slot: _format_pick_up_spec(
                        active_arm,
                        placement.moved_runtime_uid,
                        pickup_upright_direction=placement.pickup_upright_direction,
                        pickup_rotate_upright=placement.pickup_rotate_upright,
                    ),
                    inactive_slot: None,
                },
            )
        )
        edge_blocks.extend(
            _dual_relative_release_edge_blocks(
                placement=placement,
                active_arm=active_arm,
                active_slot=active_slot,
                waiting_slot=inactive_slot,
                waiting_action=None,
            )
        )
        edge_blocks.append(
            (
                f"Return `{active_arm}` to its initial pose",
                {
                    active_slot: _format_initial_qpos_spec(
                        active_arm, sample_interval=30
                    ),
                    inactive_slot: None,
                },
            )
        )
    return edge_blocks


def _make_hold_hover_task_prompt(
    task_name: str,
    project_name: str,
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    pick_actions = {
        f"{placement.active_side}_arm_action": _format_pick_up_spec(
            f"{placement.active_side}_arm",
            placement.moved_runtime_uid,
        )
        for placement in spec.placements
    }
    hover_actions = {
        f"{placement.active_side}_arm_action": _format_hover_move_spec(
            f"{placement.active_side}_arm",
            placement,
        )
        for placement in spec.placements
    }
    close_actions = {
        f"{placement.active_side}_arm_action": _format_gripper_spec(
            f"{placement.active_side}_arm",
            "close",
            sample_interval=10,
            post_hold_steps=20,
        )
        for placement in spec.placements
    }
    for side in ("left", "right"):
        pick_actions.setdefault(f"{side}_arm_action", None)
        hover_actions.setdefault(f"{side}_arm_action", None)
        close_actions.setdefault(f"{side}_arm_action", None)

    numbered_edges = _format_numbered_edge_blocks(
        [
            ("Pick up the selected object(s)", pick_actions),
            (
                "Move the held object(s) to the hover pose without releasing",
                hover_actions,
            ),
            ("Keep the gripper(s) closed and finish while holding", close_actions),
        ]
    )
    objects = ", ".join(
        f"`{placement.moved_runtime_uid}` with {placement.active_side}_arm"
        for placement in spec.placements
    )
    return render_prompt_template(
        "hold_hover_task.txt",
        task_name=task_name,
        task_prompt_summary=spec.task_prompt_summary,
        task_description=spec.task_description,
        objects=objects,
        numbered_edges=numbered_edges,
        robot_display_name=profile.display_name,
        project_name=project_name,
    )


def _hold_hover_graph_steps(spec: _RelativeSpecLike) -> list[NominalGraphStep]:
    pick_actions = {
        f"{placement.active_side}_arm_action": _format_pick_up_spec(
            f"{placement.active_side}_arm",
            placement.moved_runtime_uid,
        )
        for placement in spec.placements
    }
    hover_actions = {
        f"{placement.active_side}_arm_action": _format_hover_move_spec(
            f"{placement.active_side}_arm",
            placement,
        )
        for placement in spec.placements
    }
    close_actions = {
        f"{placement.active_side}_arm_action": _format_gripper_spec(
            f"{placement.active_side}_arm",
            "close",
            sample_interval=10,
            post_hold_steps=20,
        )
        for placement in spec.placements
    }
    for side in ("left", "right"):
        pick_actions.setdefault(f"{side}_arm_action", None)
        hover_actions.setdefault(f"{side}_arm_action", None)
        close_actions.setdefault(f"{side}_arm_action", None)
    return [
        _nominal_step("Pick up the selected object(s)", pick_actions),
        _nominal_step(
            "Move the held object(s) to the hover pose without releasing",
            hover_actions,
        ),
        _nominal_step(
            "Keep the gripper(s) closed and finish while holding",
            close_actions,
        ),
    ]


def _dual_relative_release_edge_blocks(
    *,
    placement: _RelativePlacementLike,
    active_arm: str,
    active_slot: str,
    waiting_slot: str,
    waiting_action: str | None,
) -> list[tuple[str, Mapping[str, str | None]]]:
    waiting_value = waiting_action
    if _is_pose_sensitive_placement(placement):
        return [
            (
                f"Move `{placement.moved_runtime_uid}` directly to the final "
                "release object pose while applying the requested orientation",
                {
                    active_slot: _format_relative_pose_spec(
                        active_arm,
                        placement,
                        pose_kind="release",
                        sample_interval=45,
                    ),
                    waiting_slot: waiting_value,
                },
            ),
            (
                f"Release `{placement.moved_runtime_uid}` in-place without moving "
                "the object pose",
                {
                    active_slot: _format_release_only_place_spec(active_arm),
                    waiting_slot: waiting_value,
                },
            ),
            (
                f"Retreat `{active_arm}` upward after release",
                {
                    active_slot: _format_empty_hand_retreat_spec(active_arm),
                    waiting_slot: waiting_value,
                },
            ),
        ]
    return [
        (
            f"Move `{placement.moved_runtime_uid}` directly to the final object "
            "pose, release, and retract without rotating",
            {
                active_slot: _format_direct_relative_place_spec(active_arm, placement),
                waiting_slot: waiting_value,
            },
        ),
    ]


def _dual_relative_release_rule(spec: _RelativeSpecLike) -> str:
    if _uses_serial_dual_sequence(spec):
        return (
            "For this dependent dual-object task, complete the first object's "
            "pick-up, placement, release, retreat, and return "
            "before picking up the second object. The inactive arm must remain "
            "null throughout each object's sequence. For each pose-sensitive "
            "object, use exactly one MoveHeldObject to move directly to the final "
            "release object pose while applying the requested orientation."
        )
    if any(_is_pose_sensitive_placement(placement) for placement in spec.placements):
        return (
            "For pose-sensitive placements, use exactly one `MoveHeldObject` to "
            "move directly to the final release object pose while applying the "
            "requested orientation. The following `Place` must be the exact "
            "relative-zero release-only spec shown below, and then the empty hand "
            "retreats upward. Any preserve placement in the same graph instead uses "
            "object-aware Place directly, without MoveHeldObject."
        )
    return (
        "Every orientation-preserving placement must use its object-aware "
        "`Place(target_object_pose=...)` spec directly after `PickUp`; do not add "
        "`MoveHeldObject` or relative-zero release-only Place edges."
    )


def _format_numbered_edge_blocks(
    edge_blocks: Sequence[tuple[str, Mapping[str, str | None]]],
) -> str:
    formatted_blocks = []
    for index, (title, actions) in enumerate(edge_blocks, start=1):
        action_lines = "\n".join(
            f"   - {slot}: {action if action is not None else 'null'}"
            for slot, action in actions.items()
        )
        formatted_blocks.append(f"{index}. {title}:\n{action_lines}")
    return "\n\n".join(formatted_blocks)


def _nominal_step(
    title: str,
    actions: Mapping[str, str | Mapping[str, Any] | None],
) -> NominalGraphStep:
    unknown_slots = set(actions) - {LEFT_ARM_ACTION_KEY, RIGHT_ARM_ACTION_KEY}
    if unknown_slots:
        raise ValueError(
            "Nominal graph actions contain unsupported slots: "
            f"{', '.join(sorted(unknown_slots))}."
        )
    return NominalGraphStep(
        semantic=title,
        left_arm_action=_action_dict(actions.get(LEFT_ARM_ACTION_KEY)),
        right_arm_action=_action_dict(actions.get(RIGHT_ARM_ACTION_KEY)),
    )


def _action_dict(spec: str | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        return json.loads(spec)
    return dict(spec)


def _relative_release_action_patterns(
    robot_name: str,
    placement: _RelativePlacementLike,
) -> str:
    if not _is_pose_sensitive_placement(placement):
        return render_prompt_template(
            "relative_release_actions_preserve.txt",
            place_spec=_format_direct_relative_place_spec(robot_name, placement),
        )
    return render_prompt_template(
        "relative_release_actions_oriented.txt",
        release_pose_spec=_format_relative_pose_spec(
            robot_name,
            placement,
            pose_kind="release",
            sample_interval=45,
        ),
        release_spec=_format_release_only_place_spec(robot_name),
        retreat_spec=_format_empty_hand_retreat_spec(robot_name),
    )


def make_relative_basic_background(
    project_name: str,
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
    object_registry: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    if spec.intent == "coordinated_pickment":
        return _make_coordinated_pickment_basic_background(
            project_name,
            spec,
            robot_profile=profile,
            object_registry=object_registry,
        )
    if spec.intent == "hold_hover":
        return _make_hold_hover_basic_background(
            project_name,
            spec,
            robot_profile=profile,
            object_registry=object_registry,
        )
    if len(spec.placements) > 1:
        return _make_dual_relative_basic_background(
            project_name,
            spec,
            robot_profile=profile,
            object_registry=object_registry,
        )

    active_arm = f"{spec.active_side}_arm"
    inactive_arm = "right_arm" if spec.active_side == "left" else "left_arm"
    notes = spec.basic_background_notes or (
        "No extra scene notes were provided by the config-stage LLM."
    )
    registry = _format_runtime_object_registry(object_registry)
    placement_rule = (
        "The deterministic graph grasps the moved object, uses object-aware Place "
        "directly at the final pose without MoveHeldObject, and returns the arm "
        "to its initial pose."
        if not _is_pose_sensitive_placement(spec)
        else "The deterministic graph uses exactly one MoveHeldObject to move "
        "directly to the final release pose while applying the requested "
        "orientation, then releases in place and returns the arm."
    )
    return render_prompt_template(
        "relative_single_background.txt",
        project_name=project_name,
        robot_display_name=profile.display_name,
        robot_context=_robot_context(profile),
        active_arm=active_arm,
        inactive_arm=inactive_arm,
        moved_runtime_uid=spec.moved_runtime_uid,
        moved_source_uid=spec.moved_source_uid,
        reference_line=_relative_reference_line(spec),
        registry=registry,
        notes=notes,
        placement_rule=placement_rule,
    )


def _make_coordinated_pickment_basic_background(
    project_name: str,
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
    object_registry: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    notes = spec.basic_background_notes or (
        "No extra scene notes were provided by the config-stage LLM."
    )
    registry = _format_runtime_object_registry(object_registry)
    if spec.coordinated_terminal_behavior is not None:
        payloads = [
            placement.moved_runtime_uid
            for placement in spec.placements
            if placement.intent == "place_relative"
        ]
        return render_prompt_template(
            "coordinated_transport_background.txt",
            project_name=project_name,
            moved_runtime_uid=spec.moved_runtime_uid,
            payloads=payloads or "none",
            coordinated_direction=spec.coordinated_direction,
            coordinated_terminal_behavior=spec.coordinated_terminal_behavior,
            registry=registry,
            notes=notes,
        )
    return render_prompt_template(
        "coordinated_pickment_background.txt",
        project_name=project_name,
        robot_display_name=profile.display_name,
        robot_context=_robot_context(profile),
        moved_runtime_uid=spec.moved_runtime_uid,
        moved_source_uid=spec.moved_source_uid,
        registry=registry,
        notes=notes,
    )


def _make_dual_relative_basic_background(
    project_name: str,
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
    object_registry: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    if spec.intent == "hold_hover":
        return _make_hold_hover_basic_background(
            project_name,
            spec,
            robot_profile=profile,
            object_registry=object_registry,
        )
    notes = spec.basic_background_notes or (
        "No extra scene notes were provided by the config-stage LLM."
    )
    placement_lines = "\n".join(
        f"- {placement.active_side}_arm moves `{placement.moved_runtime_uid}` "
        f"{_relative_relation_phrase(placement.relation)} "
        f"`{placement.reference_runtime_uid}`."
        for placement in spec.placements
    )
    registry = _format_runtime_object_registry(object_registry)
    serial_sequence = _uses_serial_dual_sequence(spec)
    execution_rule = (
        "The deterministic graph completes the first moved object's pick-up, "
        "placement, retreat, and return before picking up the second moved "
        "object. The inactive arm remains null throughout each sequence."
        if serial_sequence
        else "The deterministic graph grasps both moved objects, stages and "
        "releases the first, then stages and releases the second while the first "
        "arm returns. Each arm releases its object before returning."
    )
    placement_rule = (
        "Dependent objects are placed serially in dependency order."
        if serial_sequence
        else "Orientation-preserving placements use object-aware Place directly "
        "after pickup, without MoveHeldObject. Each pose-sensitive placement uses "
        "exactly one direct final-pose MoveHeldObject, then release-only Place."
    )
    return render_prompt_template(
        "relative_dual_background.txt",
        project_name=project_name,
        robot_display_name=profile.display_name,
        robot_context=_robot_context(profile),
        placement_lines=placement_lines,
        registry=registry,
        notes=notes,
        execution_rule=execution_rule,
        placement_rule=placement_rule,
    )


def _make_hold_hover_basic_background(
    project_name: str,
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
    object_registry: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    notes = spec.basic_background_notes or (
        "No extra scene notes were provided by the config-stage LLM."
    )
    object_lines = "\n".join(
        f"- {placement.moved_runtime_uid}: source `{placement.moved_source_uid}`, "
        f"handled by {placement.active_side}_arm, hover_height={placement.hover_height}."
        for placement in spec.placements
    )
    registry = _format_runtime_object_registry(object_registry)
    return render_prompt_template(
        "hold_hover_background.txt",
        project_name=project_name,
        robot_display_name=profile.display_name,
        robot_context=_robot_context(profile),
        object_lines=object_lines,
        registry=registry,
        notes=notes,
    )


def make_relative_atom_actions_prompt(
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    if spec.intent == "coordinated_pickment":
        return _make_coordinated_pickment_atom_actions_prompt(
            spec,
            robot_profile=profile,
        )
    if spec.intent == "hold_hover":
        return _make_hold_hover_atom_actions_prompt(spec, robot_profile=profile)
    if len(spec.placements) > 1:
        return _make_dual_relative_atom_actions_prompt(spec, robot_profile=profile)

    active_arm = f"{spec.active_side}_arm"
    inactive_arm = "right_arm" if spec.active_side == "left" else "left_arm"
    release_actions = _relative_release_action_patterns(active_arm, spec)
    pick_spec = _format_pick_up_spec(
        active_arm,
        spec.moved_runtime_uid,
        pickup_upright_direction=spec.pickup_upright_direction,
        pickup_rotate_upright=spec.pickup_rotate_upright,
    )
    return render_prompt_template(
        "relative_single_actions.txt",
        robot_display_name=profile.display_name,
        active_arm=active_arm,
        inactive_arm=inactive_arm,
        moved_runtime_uid=spec.moved_runtime_uid,
        pick_spec=pick_spec,
        release_actions=release_actions,
        return_spec=_format_initial_qpos_spec(active_arm, sample_interval=30),
    )


def _make_dual_relative_atom_actions_prompt(
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    if spec.intent == "hold_hover":
        return _make_hold_hover_atom_actions_prompt(spec, robot_profile=profile)
    first, second = spec.placements
    first_arm = f"{first.active_side}_arm"
    second_arm = f"{second.active_side}_arm"
    first_release_actions = _relative_release_action_patterns(first_arm, first)
    second_release_actions = _relative_release_action_patterns(second_arm, second)
    first_pick_spec = _format_pick_up_spec(
        first_arm,
        first.moved_runtime_uid,
        pickup_upright_direction=first.pickup_upright_direction,
        pickup_rotate_upright=first.pickup_rotate_upright,
    )
    second_pick_spec = _format_pick_up_spec(
        second_arm,
        second.moved_runtime_uid,
        pickup_upright_direction=second.pickup_upright_direction,
        pickup_rotate_upright=second.pickup_rotate_upright,
    )
    return render_prompt_template(
        "relative_dual_actions.txt",
        robot_display_name=profile.display_name,
        first_arm=first_arm,
        first_runtime_uid=first.moved_runtime_uid,
        second_arm=second_arm,
        second_runtime_uid=second.moved_runtime_uid,
        first_pick_spec=first_pick_spec,
        second_pick_spec=second_pick_spec,
        first_release_actions=first_release_actions,
        second_release_actions=second_release_actions,
        holding_spec=_format_gripper_spec("<holding_arm>", "close", sample_interval=10),
        return_spec=_format_initial_qpos_spec("<released_arm>", sample_interval=30),
    )


def _make_hold_hover_atom_actions_prompt(
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    blocks = "\n\n".join(
        _hold_hover_atom_action_block(placement) for placement in spec.placements
    )
    return render_prompt_template(
        "hold_hover_actions.txt",
        robot_display_name=profile.display_name,
        blocks=blocks,
    )


def _hold_hover_atom_action_block(placement: _RelativePlacementLike) -> str:
    active_arm = f"{placement.active_side}_arm"
    return render_prompt_template(
        "hold_hover_action_block.txt",
        moved_runtime_uid=placement.moved_runtime_uid,
        pickup_spec=_format_pick_up_spec(active_arm, placement.moved_runtime_uid),
        hover_spec=_format_hover_move_spec(active_arm, placement),
        close_spec=_format_gripper_spec(
            active_arm,
            "close",
            sample_interval=10,
            post_hold_steps=20,
        ),
    )


def _make_coordinated_pickment_atom_actions_prompt(
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    if spec.coordinated_terminal_behavior is not None:
        steps = _coordinated_pickment_graph_steps(spec)
        lines = []
        for index, step in enumerate(steps, start=1):
            lines.append(
                f"{index}. {step.semantic}\n"
                f"   left_arm_action: {json.dumps(step.left_arm_action, ensure_ascii=False, separators=(',', ':')) if step.left_arm_action is not None else 'null'}\n"
                f"   right_arm_action: {json.dumps(step.right_arm_action, ensure_ascii=False, separators=(',', ':')) if step.right_arm_action is not None else 'null'}"
            )
        return render_prompt_template(
            "coordinated_transport_actions.txt",
            robot_display_name=profile.display_name,
            action_lines="\n".join(lines),
        )
    left_release_spec = _format_gripper_spec(
        "left_arm",
        "open",
        sample_interval=10,
        post_hold_steps=20,
    )
    right_release_spec = _format_gripper_spec(
        "right_arm",
        "open",
        sample_interval=10,
        post_hold_steps=20,
    )
    left_initial_spec = _format_initial_qpos_spec("left_arm", sample_interval=30)
    right_initial_spec = _format_initial_qpos_spec("right_arm", sample_interval=30)
    return render_prompt_template(
        "coordinated_pickment_actions.txt",
        robot_display_name=profile.display_name,
        moved_runtime_uid=spec.moved_runtime_uid,
        coordinated_spec=_format_coordinated_pickment_spec(spec),
        left_release_spec=left_release_spec,
        right_release_spec=right_release_spec,
        left_initial_spec=left_initial_spec,
        right_initial_spec=right_initial_spec,
    )


def _format_pick_up_spec(
    robot_name: str,
    obj_name: str,
    *,
    sample_interval: int = 45,
    lift_height: float = _PICKUP_LIFT_HEIGHT,
    pickup_upright_direction: Sequence[float] | None = None,
    pickup_rotate_upright: float | None = None,
) -> str:
    cfg: dict[str, Any] = {
        "pre_grasp_distance": 0.08,
        "lift_height": float(lift_height),
        "sample_interval": sample_interval,
    }
    if pickup_upright_direction is not None and pickup_rotate_upright is not None:
        cfg["obj_upright_direction"] = [
            float(value) for value in pickup_upright_direction
        ]
        cfg["rotate_upright"] = float(pickup_rotate_upright)
    return _compact_json(
        {
            "atomic_action_class": "PickUp",
            "robot_name": robot_name,
            "control": "arm",
            "target_object": {
                "obj_name": obj_name,
                "affordance": "antipodal",
            },
            "cfg": cfg,
        }
    )


def _format_coordinated_pickment_spec(
    placement: _RelativePlacementLike,
    *,
    sample_interval: int = 120,
    payload_runtime_uids: Sequence[str] = (),
    target_hover: bool = False,
    hold_steps: int | None = None,
) -> str:
    target_object_pose: dict[str, Any]
    if getattr(placement, "reference_is_initial_pose", False):
        if placement.release_position is None:
            raise ValueError(
                "CoordinatedPickment self-relative target requires release_position."
            )
        position = [float(value) for value in placement.release_position]
        if target_hover:
            position[2] += float(placement.hover_height)
        target_object_pose = {
            "reference": "absolute",
            "position": position,
            "orientation_goal": placement.orientation_goal,
            "orientation_axis": placement.orientation_axis,
        }
    else:
        x, y, z = placement.release_offset
        target_object_pose = {
            "reference": "object",
            "obj_name": placement.reference_runtime_uid,
            "offset": [float(x), float(y), float(z)],
            "orientation_goal": placement.orientation_goal,
            "orientation_axis": placement.orientation_axis,
        }
    if placement.orientation_align_to_runtime_uid is not None:
        target_object_pose["align_to"] = placement.orientation_align_to_runtime_uid
    if placement.relation == "on" and not getattr(
        placement,
        "reference_is_initial_pose",
        False,
    ):
        _add_surface_z_policy(
            target_object_pose,
            z_policy=_SURFACE_RELEASE_Z_POLICY,
            support=placement.reference_runtime_uid,
            surface_clearance=_surface_release_clearance(placement),
        )
    target_object = {
        "obj_name": placement.moved_runtime_uid,
        "affordance": "antipodal",
    }
    if target_hover or payload_runtime_uids:
        target_object["payloads"] = [str(uid) for uid in payload_runtime_uids]
    cfg: dict[str, Any] = {
        "pre_grasp_distance": 0.1,
        "sample_interval": sample_interval,
        "hand_interp_steps": 10,
        "max_grasp_separation_angle_to_world_y_degrees": (
            _COORDINATED_MAX_GRASP_SEPARATION_ANGLE_TO_WORLD_Y_DEGREES
        ),
    }
    if target_hover:
        cfg["lift_height"] = float(placement.hover_height)
    if hold_steps is not None:
        cfg["hold_steps"] = int(hold_steps)
    return _compact_json(
        {
            "atomic_action_class": "CoordinatedPickment",
            "robot_name": DUAL_ARM_NAME,
            "control": "arm",
            "target_object": target_object,
            "target_object_pose": target_object_pose,
            "cfg": cfg,
        }
    )


def _format_relative_eef_move_spec(
    robot_name: str,
    *,
    offset: Sequence[float],
    sample_interval: int,
    post_hold_steps: int = 0,
) -> str:
    cfg = {"sample_interval": int(sample_interval)}
    if post_hold_steps > 0:
        cfg["post_hold_steps"] = int(post_hold_steps)
    return _compact_json(
        {
            "atomic_action_class": "MoveEndEffector",
            "robot_name": robot_name,
            "control": "arm",
            "target_pose": {
                "reference": "relative",
                "offset": [float(value) for value in offset],
                "frame": "world",
            },
            "cfg": cfg,
        }
    )


def _format_pose_object_spec(
    robot_name: str,
    obj_name: str,
    offset: tuple[float, float, float] | list[float],
    *,
    sample_interval: int,
    orientation_goal: str = "preserve",
    orientation_axis: str = "none",
    align_to: str | None = None,
    z_policy: str | None = None,
    support: str | None = None,
    surface_clearance: float | None = None,
) -> str:
    x, y, z = offset
    target_object_pose = {
        "reference": "object",
        "obj_name": obj_name,
        "offset": [float(x), float(y), float(z)],
        "orientation_goal": orientation_goal,
        "orientation_axis": orientation_axis,
    }
    if align_to is not None:
        target_object_pose["align_to"] = align_to
    _add_surface_z_policy(
        target_object_pose,
        z_policy=z_policy,
        support=support,
        surface_clearance=surface_clearance,
    )
    return _compact_json(
        {
            "atomic_action_class": "MoveHeldObject",
            "robot_name": robot_name,
            "control": "arm",
            "target_object_pose": target_object_pose,
            "cfg": {"sample_interval": sample_interval},
        }
    )


def _format_relative_pose_spec(
    robot_name: str,
    placement: _RelativePlacementLike,
    *,
    pose_kind: str,
    sample_interval: int,
    orientation_goal: str | None = None,
    orientation_axis: str | None = None,
    align_to: str | None | object = _USE_PLACEMENT_ALIGN_TO,
) -> str:
    resolved_orientation_goal = orientation_goal or placement.orientation_goal
    resolved_orientation_axis = orientation_axis or placement.orientation_axis
    resolved_align_to = (
        placement.orientation_align_to_runtime_uid
        if align_to is _USE_PLACEMENT_ALIGN_TO
        else align_to
    )
    surface_support = _relative_surface_support(placement, pose_kind=pose_kind)
    surface_z_policy = (
        _SURFACE_RELEASE_Z_POLICY if surface_support is not None else None
    )
    if getattr(placement, "reference_is_initial_pose", False) or getattr(
        placement,
        "upright_in_place",
        False,
    ):
        position = (
            placement.high_position
            if pose_kind == "high"
            else placement.release_position
        )
        if position is None:
            raise ValueError(
                "Self-relative placement requires absolute high/release positions."
            )
        return _format_pose_absolute_spec(
            robot_name,
            position,
            sample_interval=sample_interval,
            orientation_goal=resolved_orientation_goal,
            orientation_axis=resolved_orientation_axis,
            align_to=resolved_align_to,
            z_policy=surface_z_policy,
            support=surface_support,
            surface_clearance=(
                _surface_release_clearance(placement)
                if surface_z_policy is not None
                else None
            ),
        )

    offset = placement.high_offset if pose_kind == "high" else placement.release_offset
    return _format_pose_object_spec(
        robot_name,
        placement.reference_runtime_uid,
        offset,
        sample_interval=sample_interval,
        orientation_goal=resolved_orientation_goal,
        orientation_axis=resolved_orientation_axis,
        align_to=resolved_align_to,
        z_policy=surface_z_policy,
        support=surface_support,
        surface_clearance=(
            _surface_release_clearance(placement)
            if surface_z_policy is not None
            else None
        ),
    )


def _format_direct_relative_place_spec(
    robot_name: str,
    placement: _RelativePlacementLike,
) -> str:
    """Format an object-aware Place for a preserve-orientation placement."""
    move_spec = json.loads(
        _format_relative_pose_spec(
            robot_name,
            placement,
            pose_kind="release",
            sample_interval=45,
        )
    )
    target_object_pose = move_spec["target_object_pose"]
    if target_object_pose.get("orientation_goal", "preserve") != "preserve":
        raise ValueError(
            "Direct relative Place only supports orientation_goal='preserve'."
        )
    return _compact_json(
        {
            "atomic_action_class": "Place",
            "robot_name": robot_name,
            "control": "arm",
            "target_object_pose": target_object_pose,
            "cfg": {
                "sample_interval": 80,
                "lift_height": _PLACE_LIFT_HEIGHT,
                "cartesian_waypoint_count": _DIRECT_PLACE_CARTESIAN_WAYPOINT_COUNT,
            },
        }
    )


def _format_stacking_place_spec(
    robot_name: str,
    step: _StackingStepLike,
    *,
    object_anchored: bool,
    stack_mode: str,
) -> str:
    if not object_anchored:
        return _format_direct_absolute_place_spec(
            robot_name,
            step.target_position,
            max_approach_retract_z=_STACKING_MAX_APPROACH_RETRACT_Z,
        )
    if step.support_runtime_uid is None:
        raise ValueError("Object-anchored stacking requires a support per layer.")

    target_object_pose: dict[str, Any] = {
        "reference": "object",
        "obj_name": step.support_runtime_uid,
        "offset": [
            0.0,
            0.0,
            _STACKING_NESTED_RELEASE_Z_OFFSET if stack_mode == "nested" else 0.0,
        ],
        "orientation_goal": "preserve",
        "orientation_axis": "none",
    }
    if stack_mode == "on_top":
        _add_surface_z_policy(
            target_object_pose,
            z_policy="surface_release",
            support=step.support_runtime_uid,
            surface_clearance=_STACKING_SURFACE_CLEARANCE,
        )
    return _compact_json(
        {
            "atomic_action_class": "Place",
            "robot_name": robot_name,
            "control": "arm",
            "target_object_pose": target_object_pose,
            "cfg": {
                "sample_interval": 80,
                "lift_height": _PLACE_LIFT_HEIGHT,
                "max_approach_retract_z": _STACKING_MAX_APPROACH_RETRACT_Z,
                "cartesian_waypoint_count": _DIRECT_PLACE_CARTESIAN_WAYPOINT_COUNT,
            },
        }
    )


def _format_direct_absolute_place_spec(
    robot_name: str,
    position: Sequence[float],
    *,
    max_approach_retract_z: float | None = None,
) -> str:
    """Format an absolute Place that preserves the held-object orientation."""
    cfg = {
        "sample_interval": 80,
        "lift_height": _PLACE_LIFT_HEIGHT,
        "cartesian_waypoint_count": _DIRECT_PLACE_CARTESIAN_WAYPOINT_COUNT,
    }
    if max_approach_retract_z is not None:
        cfg["max_approach_retract_z"] = float(max_approach_retract_z)
    return _compact_json(
        {
            "atomic_action_class": "Place",
            "robot_name": robot_name,
            "control": "arm",
            "target_object_pose": {
                "reference": "absolute",
                "position": [float(value) for value in position],
                "orientation_goal": "preserve",
                "orientation_axis": "none",
            },
            "cfg": cfg,
        }
    )


def _surface_release_clearance(placement: _RelativePlacementLike) -> float:
    return float(getattr(placement, "surface_clearance", _SURFACE_RELEASE_CLEARANCE))


def _relative_surface_support(
    placement: _RelativePlacementLike,
    *,
    pose_kind: str,
) -> str | None:
    if pose_kind != "release" or placement.relation != "on":
        return None
    if getattr(placement, "reference_is_initial_pose", False):
        return None
    return placement.reference_runtime_uid


def _format_hover_move_spec(
    robot_name: str,
    placement: _RelativePlacementLike,
) -> str:
    return _compact_json(
        {
            "atomic_action_class": "MoveHeldObject",
            "robot_name": robot_name,
            "control": "arm",
            "target_object_pose": {
                "reference": "relative",
                "offset": [0.0, 0.0, float(placement.hover_height)],
                "frame": "world",
                "orientation_goal": "preserve",
                "orientation_axis": "none",
            },
            "cfg": {"sample_interval": 45},
        }
    )


def _is_pose_sensitive_placement(placement: _RelativePlacementLike) -> bool:
    return placement.orientation_goal != "preserve"


def _format_release_only_place_spec(robot_name: str) -> str:
    return _format_place_spec(
        robot_name,
        {
            "reference": "relative",
            "offset": [0.0, 0.0, 0.0],
            "frame": "world",
        },
        sample_interval=_RELEASE_ONLY_PLACE_SAMPLE_INTERVAL,
        lift_height=0.0,
    )


def _format_empty_hand_retreat_spec(robot_name: str) -> str:
    return _compact_json(
        {
            "atomic_action_class": "MoveEndEffector",
            "robot_name": robot_name,
            "control": "arm",
            "target_pose": {
                "reference": "relative",
                "offset": [0.0, 0.0, _PLACE_LIFT_HEIGHT],
                "frame": "world",
            },
            "cfg": {"sample_interval": _EMPTY_HAND_RETREAT_SAMPLE_INTERVAL},
        }
    )


def _format_pose_absolute_spec(
    robot_name: str,
    position: Sequence[float],
    *,
    sample_interval: int,
    orientation_goal: str = "preserve",
    orientation_axis: str = "none",
    align_to: str | None = None,
    z_policy: str | None = None,
    support: str | None = None,
    surface_clearance: float | None = None,
) -> str:
    target_object_pose = {
        "reference": "absolute",
        "position": [float(value) for value in position],
        "orientation_goal": orientation_goal,
        "orientation_axis": orientation_axis,
    }
    if align_to is not None:
        target_object_pose["align_to"] = align_to
    _add_surface_z_policy(
        target_object_pose,
        z_policy=z_policy,
        support=support,
        surface_clearance=surface_clearance,
    )
    return _compact_json(
        {
            "atomic_action_class": "MoveHeldObject",
            "robot_name": robot_name,
            "control": "arm",
            "target_object_pose": target_object_pose,
            "cfg": {"sample_interval": sample_interval},
        }
    )


def _add_surface_z_policy(
    target_object_pose: dict[str, Any],
    *,
    z_policy: str | None,
    support: str | None,
    surface_clearance: float | None,
) -> None:
    if z_policy is None:
        return
    target_object_pose["z_policy"] = z_policy
    if support is not None:
        target_object_pose["support"] = support
    if surface_clearance is not None:
        target_object_pose["surface_clearance"] = float(surface_clearance)


def _format_place_spec(
    robot_name: str,
    target_pose: Mapping[str, Any],
    *,
    sample_interval: int,
    lift_height: float,
) -> str:
    return _compact_json(
        {
            "atomic_action_class": "Place",
            "robot_name": robot_name,
            "control": "arm",
            "target_pose": dict(target_pose),
            "cfg": {
                "sample_interval": sample_interval,
                "lift_height": float(lift_height),
            },
        }
    )


def _format_gripper_spec(
    robot_name: str,
    state: str,
    *,
    sample_interval: int,
    post_hold_steps: int = 0,
) -> str:
    cfg = {"sample_interval": sample_interval}
    if post_hold_steps:
        cfg["post_hold_steps"] = post_hold_steps
    return _compact_json(
        {
            "atomic_action_class": "MoveJoints",
            "robot_name": robot_name,
            "control": "hand",
            "target_qpos": {"source": "gripper_state", "state": state},
            "cfg": cfg,
        }
    )


def _format_initial_qpos_spec(
    robot_name: str,
    *,
    sample_interval: int,
) -> str:
    return _compact_json(
        {
            "atomic_action_class": "MoveJoints",
            "robot_name": robot_name,
            "control": "arm",
            "target_qpos": {"source": "initial"},
            "cfg": {"sample_interval": sample_interval},
        }
    )


def _compact_json(value: Mapping[str, Any]) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return re.sub(r'("lift_height":)0\.3(?=}|,)', r"\g<1>0.30", text)


def _format_action_sketch(action_sketch: list[str]) -> str:
    return "\n".join(f"- {item}" for item in action_sketch)


def _relative_reference_line(spec: _RelativePlacementLike) -> str:
    if getattr(spec, "upright_in_place", False):
        return (
            f"Use `{spec.reference_runtime_uid}` as the support surface while "
            f"anchoring XY at the initial position of `{spec.moved_runtime_uid}`."
        )
    if getattr(spec, "reference_is_initial_pose", False):
        return (
            f"Use the initial position of `{spec.moved_runtime_uid}` as the fixed "
            f"spatial anchor. Source object: `{spec.moved_source_uid}`."
        )
    return (
        f"Use `{spec.reference_runtime_uid}` as the spatial reference. Source "
        f"object: `{spec.reference_source_uid}`."
    )


def _relative_pose_step_label(
    spec: _RelativePlacementLike,
    label: str,
) -> str:
    if getattr(spec, "reference_is_initial_pose", False):
        return f"{label} at the absolute initial-position offset"
    if getattr(spec, "upright_in_place", False):
        return f"{label} at the initial XY on `{spec.reference_runtime_uid}`"
    return f"{label} relative to `{spec.reference_runtime_uid}`"


def _relative_final_planning_rule(
    project_name: str,
    spec: _RelativePlacementLike,
) -> str:
    if getattr(spec, "reference_is_initial_pose", False) or getattr(
        spec,
        "upright_in_place",
        False,
    ):
        return (
            "Use the exact absolute target_pose JSON specs shown above. Do not "
            "rewrite this placement as a table-centered object-referenced pose; "
            "its XY anchor is the moved object's initial position."
        )
    return (
        f"Always plan to the current object poses from the exported {project_name} "
        "environment config. Do not hard-code absolute object coordinates in the "
        "generated graph."
    )


def _dual_relative_final_planning_rule(
    project_name: str,
    spec: _RelativeSpecLike,
) -> str:
    if any(
        getattr(placement, "reference_is_initial_pose", False)
        for placement in spec.placements
    ):
        return (
            "Use the exact absolute target_pose JSON specs shown above for any "
            "initial-position placement. Do not rewrite those self-relative "
            "steps as object-referenced poses."
        )
    return (
        f"Always plan to the current object poses from the exported {project_name} "
        "environment config. Do not hard-code absolute object coordinates in the "
        "generated graph."
    )


def _relative_relation_phrase(relation: str) -> str:
    # Keep this private wrapper for compatibility with existing imports while
    # delegating the vocabulary to the shared generation/runtime contract.
    return _canonical_relative_relation_phrase(relation)


def _display_noun(uid: str) -> str:
    return uid.replace("_", " ")


def _plural(noun: str) -> str:
    if noun.endswith("s"):
        return noun
    if noun.endswith(("ch", "sh", "x")):
        return f"{noun}es"
    return f"{noun}s"
