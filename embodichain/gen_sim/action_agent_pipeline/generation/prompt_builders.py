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

__all__ = [
    "make_agent_config",
    "make_arrangement_task_graph",
    "make_arrangement_atom_actions_prompt",
    "make_arrangement_basic_background",
    "make_arrangement_task_prompt",
    "make_basket_task_graph",
    "make_basket_atom_actions_prompt",
    "make_basket_basic_background",
    "make_basket_task_prompt",
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
_STACKING_DEFAULTS = generation_defaults_section("stacking")
_STACKING_NESTED_RELEASE_Z_OFFSET = float(_STACKING_DEFAULTS["nested_release_z_offset"])
_STACKING_SURFACE_CLEARANCE = float(_STACKING_DEFAULTS["clearance"])
_STACKING_MAX_APPROACH_RETRACT_Z = float(_STACKING_DEFAULTS["max_approach_retract_z"])
_SURFACE_RELEASE_Z_POLICY = "object_on_surface"
_SURFACE_RELEASE_CLEARANCE = DEFAULT_SURFACE_RELEASE_CLEARANCE
_USE_PLACEMENT_ALIGN_TO = object()
_RELATIVE_COORDINATE_CONVENTION = """Coordinate convention for relative placement:
- `left_of` means positive world y relative to the reference object.
- `right_of` means negative world y relative to the reference object.
- `front_of` means positive world x relative to the reference object.
- `behind` means negative world x relative to the reference object.
- `front_left_of` combines positive world x and positive world y.
- `back_left_of` combines negative world x and positive world y.
- `front_right_of` combines positive world x and negative world y.
- `back_right_of` combines negative world x and negative world y.
- `inside` uses generated container slot offsets; multiple objects sharing a
  container are distributed along the container XY long axis.
- `on` uses the reference object's xy center."""


class _BasketRolesLike(Protocol):
    left_target_runtime_uid: str
    right_target_runtime_uid: str
    container_runtime_uid: str
    left_target_source_uid: str
    right_target_source_uid: str
    container_source_uid: str
    left_target_noun: str
    right_target_noun: str


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
            "precomputed_task_graph": "task_graph.json",
        },
        "CompileAgent": {},
        "Agent": {
            "prompt_kwargs": {
                "task_prompt": {
                    "type": "text",
                    "name": "task_prompt.txt",
                },
                "basic_background": {
                    "type": "text",
                    "name": "basic_background.txt",
                },
                "atom_actions": {
                    "type": "text",
                    "name": "atom_actions.txt",
                },
            }
        },
    }


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
    return f"""Task:
{task_name}: {spec.task_prompt_summary}

This config was generated from a simple task description by the config-stage
LLM. The execution-stage LLM must now generate the graph JSON from this prompt.

Original simple task description:
{spec.task_description}

Arrangement plan:
- Layout axis: `{spec.axis}`, resolved to world `{world_axis}`. Semantic slots
  unfold `{spec.spatial_direction}` while execution follows occupancy dependencies.
- Anchor: `{spec.anchor}` in the exported {project_name} environment.
- Collision-aware line origin xy: `{list(spec.line_origin_xy)}`.
- Slot spacing: `{float(spec.spacing):.6g}` with clearance `{float(spec.layout_clearance):.6g}`.
- Ordering rule: `{spec.order_by}` with direction `{spec.order_direction}`.
- Category order: `{list(spec.category_order)}`.
- Final order: {final_order}.

Generate one deterministic nominal graph with exactly {edge_count} nominal edges.
Use only the atomic action class JSON specs shown below. Do not add recovery,
monitor, search, alignment, or extra lift edges beyond the listed steps. The
absolute target object poses are collision-aware slots computed by the
config-stage generator; do not rewrite them. First move each held object to the
high staging pose with orientation preserved. If a step has
`orientation_goal:"preserve"`, move directly down to the final release object
pose without adding a separate high-orientation/alignment edge. Only steps with
an explicit non-preserve orientation goal may align at the same high pose before
moving down. The final release move includes an explicit surface `z_policy`;
keep it exactly so the runtime resolver chooses a safe release height from the
support surface and held-object geometry. Use the exact relative-zero
release-only `Place` spec shown below, retreat the empty hand upward, then
return that arm. The arm not listed for a step must remain null.

{step_blocks}

Final state: all listed objects must rest near their assigned absolute XY slots.
Only steps that explicitly request axis alignment should rotate the held object.
Use the exact absolute target_object_pose JSON specs shown above; do not rewrite
slot placement as object-referenced poses.
"""


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
    active_arm = f"{step.active_side}_arm"
    active_slot = f"{step.active_side}_arm_action"
    inactive_slot = f"{'right' if step.active_side == 'left' else 'left'}_arm_action"
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
    blocks.extend(
        [
            (
                release_title,
                {
                    active_slot: release_move_spec,
                    inactive_slot: None,
                },
            ),
            (
                f"Release `{step.runtime_uid}` in-place without moving the object pose",
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
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a {profile.display_name} multi-object line arrangement
task generated from a simple natural-language task description.

{_robot_context(profile)}

Interactive task objects and target slots:
{object_lines}
{registry}

Config-stage LLM notes:
{notes}

The execution-stage LLM should preserve each object's initial orientation while
lifting to the high staging pose. For steps with `orientation_goal:"preserve"`,
move directly down to the final object pose without a separate alignment move.
Only steps with a non-preserve orientation goal should align at the high pose.
After release, retreat the empty hand upward and then return the arm to its
initial pose.
"""


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
    return f"""### Atomic Action Class JSON Specs for {profile.display_name} Line Arrangement

Use only the native atomic action class JSON specs shown below. Each object is
moved to an absolute collision-aware slot pose computed by the config-stage
generator. For steps with `orientation_goal:"preserve"`, move directly from the
high pose to the final object pose without a high alignment move. Keep the
non-active arm null for each listed object.

{blocks}
"""


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
        return f"""Object `{step.runtime_uid}` to slot {step.slot_index}:
- Pick up:
  {_format_pick_up_spec(active_arm, step.runtime_uid)}
- High staging without orientation change:
  {high_preserve_spec}
- Final release object pose without orientation change:
  {release_move_spec}
- Release-only Place:
  {_format_release_only_place_spec(active_arm)}
- Empty-hand retreat:
  {_format_empty_hand_retreat_spec(active_arm)}
- Return:
  {_format_initial_qpos_spec(active_arm, sample_interval=30)}"""
    high_align_spec = _format_pose_absolute_spec(
        active_arm,
        step.high_position,
        sample_interval=45,
        orientation_goal=step.orientation_goal,
        orientation_axis=step.orientation_axis,
    )
    return f"""Object `{step.runtime_uid}` to slot {step.slot_index}:
- Pick up:
  {_format_pick_up_spec(active_arm, step.runtime_uid)}
- High staging without orientation change:
  {high_preserve_spec}
- High staging axis alignment:
  {high_align_spec}
- Final release object pose:
  {release_move_spec}
- Release-only Place:
  {_format_release_only_place_spec(active_arm)}
- Empty-hand retreat:
  {_format_empty_hand_retreat_spec(active_arm)}
- Return:
  {_format_initial_qpos_spec(active_arm, sample_interval=30)}"""


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
    return f"""Task:
{task_name}: {spec.task_prompt_summary}

This config was generated from a stacking task description by the config-stage
LLM. The execution-stage LLM must now generate the graph JSON from this prompt.

Original simple task description:
{spec.task_description}

Stacking plan:
- Stack mode: `{spec.stack_mode}`.
- Anchor: {anchor_description} in the exported {project_name} environment.
- Ordering rule: `{spec.order_by}`.
- Bottom-to-top order: {stack_order}.

Generate one deterministic nominal graph with exactly {edge_count} nominal edges.
Use only the atomic action class JSON specs shown below. Do not add recovery,
monitor, search, alignment, or extra lift edges. Execute one object at a time;
do not pick up two objects simultaneously. Move each held object to the high
staging object pose. If a step has `orientation_goal:"preserve"`, do not add a
separate high-orientation/alignment edge. Only steps with an explicit
non-preserve orientation goal may align at the same high pose before moving down
to the final object pose. Release with the exact relative-zero `Place` spec,
retreat the empty hand upward, then return that arm to its initial pose.

{step_blocks}

Final state: the objects must be stacked at the configured anchor.
For `on_top`, each upper layer rests on the previous layer. For `nested`, each
upper bowl is nested into the previous bowl. {final_target_rule}
"""


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
    active_arm = f"{step.active_side}_arm"
    active_slot = f"{step.active_side}_arm_action"
    inactive_slot = f"{'right' if step.active_side == 'left' else 'left'}_arm_action"
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
    blocks.extend(
        [
            (
                release_title,
                {
                    active_slot: release_move_spec,
                    inactive_slot: None,
                },
            ),
            (
                f"Release `{step.runtime_uid}` in-place without moving the "
                "object pose",
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
    )
    return blocks


def _stacking_step_prompt_block(
    start_edge: int,
    step: _StackingStepLike,
    *,
    object_anchored: bool,
    stack_mode: str,
) -> str:
    active_arm = f"{step.active_side}_arm"
    active_slot = f"{step.active_side}_arm_action"
    inactive_slot = f"{'right' if step.active_side == 'left' else 'left'}_arm_action"
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
        return f"""{start_edge}. Pick up `{step.runtime_uid}` for stack layer {step.layer_index}:
   - {active_slot}: {_format_pick_up_spec(active_arm, step.runtime_uid)}
   - {inactive_slot}: null

{start_edge + 1}. Place `{step.runtime_uid}` directly at the final stack pose without changing orientation:
   - {active_slot}: {_format_stacking_place_spec(active_arm, step, object_anchored=object_anchored, stack_mode=stack_mode)}
   - {inactive_slot}: null

{start_edge + 2}. Return `{active_arm}` to its initial pose:
   - {active_slot}: {_format_initial_qpos_spec(active_arm, sample_interval=30)}
   - {inactive_slot}: null"""
    return f"""{start_edge}. Pick up `{step.runtime_uid}` for stack layer {step.layer_index}:
   - {active_slot}: {_format_pick_up_spec(active_arm, step.runtime_uid)}
   - {inactive_slot}: null

{start_edge + 1}. Move `{step.runtime_uid}` to the high staging pose without changing orientation:
   - {active_slot}: {high_preserve_spec}
   - {inactive_slot}: null

{start_edge + 2}. Align `{step.runtime_uid}` at the high staging pose if the spec requires it:
   - {active_slot}: {high_oriented_spec}
   - {inactive_slot}: null

{start_edge + 3}. Move `{step.runtime_uid}` down to the final stack object pose:
   - {active_slot}: {release_move_spec}
   - {inactive_slot}: null

{start_edge + 4}. Release `{step.runtime_uid}` in-place without moving the object pose:
   - {active_slot}: {_format_release_only_place_spec(active_arm)}
   - {inactive_slot}: null

{start_edge + 5}. Retreat `{active_arm}` upward after release:
   - {active_slot}: {_format_empty_hand_retreat_spec(active_arm)}
   - {inactive_slot}: null

{start_edge + 6}. Return `{active_arm}` to its initial pose:
   - {active_slot}: {_format_initial_qpos_spec(active_arm, sample_interval=30)}
   - {inactive_slot}: null"""


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
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a {profile.display_name} stacking task
generated from a simple natural-language task description.

{_robot_context(profile)}

Stack mode: `{spec.stack_mode}` with `{spec.anchor}` anchor at xy `{list(spec.anchor_xy)}`.

Interactive task objects and stack layers:
{object_lines}
{registry}

Config-stage LLM notes:
{notes}

The execution-stage LLM should manipulate one object at a time, release it in
place, retreat upward with an empty gripper, and then return the active arm to
its initial pose before starting the next stack layer.
"""


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
    return f"""### Atomic Action Class JSON Specs for {profile.display_name} Stacking

Use only the native atomic action class JSON specs shown below. Each object is
moved to the configured stack target computed by the config-stage generator.
Keep the non-active arm null for each listed object.

{blocks}
"""


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
        return f"""Object `{step.runtime_uid}` to stack layer {step.layer_index}:
- Pick up:
  {_format_pick_up_spec(active_arm, step.runtime_uid)}
- Direct final Place without orientation change:
  {_format_stacking_place_spec(active_arm, step, object_anchored=object_anchored, stack_mode=stack_mode)}
- Return:
  {_format_initial_qpos_spec(active_arm, sample_interval=30)}"""
    return f"""Object `{step.runtime_uid}` to stack layer {step.layer_index}:
- Pick up:
  {_format_pick_up_spec(active_arm, step.runtime_uid)}
- High staging without orientation change:
  {_format_pose_absolute_spec(active_arm, step.high_position, sample_interval=45, orientation_goal="preserve", orientation_axis="none")}
- High staging orientation:
  {high_oriented_spec}
- Final stack object pose:
  {_format_pose_absolute_spec(active_arm, step.target_position, sample_interval=45, orientation_goal=step.orientation_goal, orientation_axis=step.orientation_axis)}
- Release-only Place:
  {_format_release_only_place_spec(active_arm)}
- Empty-hand retreat:
  {_format_empty_hand_retreat_spec(active_arm)}
- Return:
  {_format_initial_qpos_spec(active_arm, sample_interval=30)}"""


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
        "right_arm_action" if spec.active_side == "left" else "left_arm_action"
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
        release_instruction = f"""2. Move the held object directly to the {release_step_label} object pose while applying the requested orientation:
   - {active_slot}: {release_move_spec}
   - {inactive_slot}: null

3. Release the held object in-place without moving the object pose:
   - {active_slot}: {place_spec}
   - {inactive_slot}: null

4. Retreat the now-empty end-effector upward:
   - {active_slot}: {retreat_spec}
   - {inactive_slot}: null

5. Return the active arm to its initial pose:
   - {active_slot}: {initial_spec}
   - {inactive_slot}: null"""
        high_instruction = release_instruction
        release_rule = (
            "For this pose-sensitive placement, use exactly one `MoveHeldObject` "
            "to move directly to the final release object pose while applying the "
            "requested orientation. Do not add staging or intermediate moves. Use "
            "the exact relative-zero release-only `Place` spec shown below."
        )
    else:
        place_spec = _format_direct_relative_place_spec(active_arm, spec)
        edge_count = 3
        high_instruction = f"""2. Move directly to the {release_step_label} object pose, release, and retract without rotating:
   - {active_slot}: {place_spec}
   - {inactive_slot}: null

3. Return the active arm to its initial pose:
   - {active_slot}: {initial_spec}
   - {inactive_slot}: null"""
        release_rule = (
            "This orientation-preserving placement must use the object-aware "
            "`Place(target_object_pose=...)` spec shown below directly after "
            "`PickUp`; do not add `MoveHeldObject` or a release-only Place edge."
        )
    return f"""Task:
{task_name}: {spec.task_prompt_summary}

This config was generated from a simple task description by the config-stage
LLM. The execution-stage LLM must now generate the graph JSON from this prompt.

Original simple task description:
{spec.task_description}

Config-stage LLM interpretation:
{action_sketch}

Object and arm mapping:
- Move `{spec.moved_runtime_uid}`. Source object: `{spec.moved_source_uid}`.
- {reference_line}
- Goal relation: `{spec.relation}` ({_relative_relation_phrase(spec.relation)}).
- Active arm: `{active_arm}`.
- Keep every `{inactive_slot}` as null.

{_RELATIVE_COORDINATE_CONVENTION}

Generate one deterministic nominal graph with exactly {edge_count} nominal edges.
Use only the atomic action class JSON specs shown below. Do not add recovery,
monitor, search, alignment, or extra lift edges. {release_rule} The inactive arm
must remain null in every edge.

1. Pick up the moved object:
   - {active_slot}: {pick_spec}
   - {inactive_slot}: null

{high_instruction}

Final state: `{spec.moved_runtime_uid}` must be
{_relative_relation_phrase(spec.relation)} `{spec.reference_runtime_uid}`.
{final_planning_rule}
"""


def _single_relative_graph_steps(
    spec: _RelativeSpecLike,
) -> list[NominalGraphStep]:
    active_arm = f"{spec.active_side}_arm"
    inactive_slot = (
        "right_arm_action" if spec.active_side == "left" else "left_arm_action"
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
    return f"""Task:
{task_name}: {spec.task_prompt_summary}

This config was generated from a simple task description by the config-stage
LLM. The execution-stage LLM must now generate the graph JSON from this prompt.

Original simple task description:
{spec.task_description}

Config-stage LLM interpretation:
{action_sketch}

Coordinated shared-object mapping:
- Use both arms together to pick and move `{spec.moved_runtime_uid}`.
- Source object: `{spec.moved_source_uid}`.
- Target relation: `{spec.relation}` ({_relative_relation_phrase(spec.relation)})
  relative to `{spec.reference_runtime_uid}`.

{_RELATIVE_COORDINATE_CONVENTION}

Generate one deterministic nominal graph with exactly 3 nominal edges. First
use the `CoordinatedPickment` JSON spec shown below to move the shared object.
It controls both arms in one atomic action, so put it in `left_arm_action` and
keep `right_arm_action` null. Then release the object by opening both grippers
simultaneously with the `MoveJoints(control="hand")` specs shown below. Finally
return both empty arms to their initial arm joint poses with the
`MoveJoints(control="arm")` specs shown below. Do not add separate `PickUp`,
`MoveHeldObject`, `Place`, or extra gripper/return actions.

1. Coordinated pick and move `{spec.moved_runtime_uid}`:
   - left_arm_action: {action_spec}
   - right_arm_action: null

2. Release `{spec.moved_runtime_uid}` from both grippers:
   - left_arm_action: {left_release_spec}
   - right_arm_action: {right_release_spec}

3. Return both empty arms to their initial poses:
   - left_arm_action: {left_initial_spec}
   - right_arm_action: {right_initial_spec}

Final state: `{spec.moved_runtime_uid}` must be
{_relative_relation_phrase(spec.relation)} `{spec.reference_runtime_uid}` and
must not remain held by either gripper. Both arms must be back at their initial
arm joint poses with grippers open.
{final_planning_rule}
"""


def _coordinated_pickment_graph_steps(
    spec: _RelativeSpecLike,
) -> list[NominalGraphStep]:
    return [
        _nominal_step(
            f"Coordinated pick and move `{spec.moved_runtime_uid}`",
            {
                "left_arm_action": _format_coordinated_pickment_spec(spec),
                "right_arm_action": None,
            },
        ),
        _nominal_step(
            f"Release `{spec.moved_runtime_uid}` from both grippers",
            {
                "left_arm_action": _format_gripper_spec(
                    "left_arm",
                    "open",
                    sample_interval=10,
                    post_hold_steps=20,
                ),
                "right_arm_action": _format_gripper_spec(
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
                "left_arm_action": _format_initial_qpos_spec(
                    "left_arm",
                    sample_interval=30,
                ),
                "right_arm_action": _format_initial_qpos_spec(
                    "right_arm",
                    sample_interval=30,
                ),
            },
        ),
    ]


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
    return f"""Task:
{task_name}: {spec.task_prompt_summary}

This config was generated from a simple task description by the config-stage
LLM. The execution-stage LLM must now generate the graph JSON from this prompt.

Original simple task description:
{spec.task_description}

Config-stage LLM interpretation:
{action_sketch}

Object and arm mapping:
- {first_slot} must manipulate `{first.moved_runtime_uid}`. Source object:
  `{first.moved_source_uid}`.
- {second_slot} must manipulate `{second.moved_runtime_uid}`. Source object:
  `{second.moved_source_uid}`.
- {first_reference_line} Goal relation for `{first.moved_runtime_uid}`:
  `{first.relation}` ({_relative_relation_phrase(first.relation)}).
- {second_reference_line} Goal relation for `{second.moved_runtime_uid}`:
  `{second.relation}` ({_relative_relation_phrase(second.relation)}).

{_RELATIVE_COORDINATE_CONVENTION}

Generate one deterministic nominal graph with exactly {edge_count} nominal edges.
Use only the atomic action class JSON specs shown below. Do not add recovery,
monitor, search, alignment, or extra lift edges. {release_rule}

{numbered_edges}

Final state: `{first.moved_runtime_uid}` must be
{_relative_relation_phrase(first.relation)} `{first.reference_runtime_uid}`, and
`{second.moved_runtime_uid}` must be {_relative_relation_phrase(second.relation)}
`{second.reference_runtime_uid}`.
{final_planning_rule}
"""


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
            "right_arm_action" if placement.active_side == "left" else "left_arm_action"
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
    return f"""Task:
{task_name}: {spec.task_prompt_summary}

This config was generated from an object-manipulation task description by the
config-stage LLM. The execution-stage LLM must now generate the graph JSON from
this prompt.

Original simple task description:
{spec.task_description}

Object and arm mapping:
- Hold-hover manipulation(s): {objects}.
- Do not release any held object.
- Do not return a holding arm to its initial qpos.

Generate one deterministic nominal graph with exactly 3 nominal edges.
Use only the atomic action class JSON specs shown below. Do not add recovery,
monitor, search, release, placement, or return-to-initial edges. The final state
must keep every selected object hovering in a closed gripper.

{numbered_edges}

Final state: every selected object must remain lifted and held by its assigned
{profile.display_name} arm in the exported {project_name} environment config.
"""


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
    unknown_slots = set(actions) - {"left_arm_action", "right_arm_action"}
    if unknown_slots:
        raise ValueError(
            "Nominal graph actions contain unsupported slots: "
            f"{', '.join(sorted(unknown_slots))}."
        )
    return NominalGraphStep(
        semantic=title,
        left_arm_action=_action_dict(actions.get("left_arm_action")),
        right_arm_action=_action_dict(actions.get("right_arm_action")),
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
        return f"""- Direct orientation-preserving Place:
  {_format_direct_relative_place_spec(robot_name, placement)}"""
    return f"""- Direct final release object pose with requested orientation:
  {_format_relative_pose_spec(robot_name, placement, pose_kind="release", sample_interval=45)}
- Release-only Place:
  {_format_release_only_place_spec(robot_name)}
- Empty-hand retreat:
  {_format_empty_hand_retreat_spec(robot_name)}"""


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
        "The execution-stage LLM should generate graph JSON that grasps the moved "
        "object, uses object-aware Place directly at the final pose without "
        "MoveHeldObject, and returns the arm to its initial pose."
        if not _is_pose_sensitive_placement(spec)
        else "The execution-stage LLM should use exactly one MoveHeldObject to move "
        "directly to the final release pose while applying the requested orientation, "
        "then release in place and return the arm to its initial pose."
    )
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a {profile.display_name} relative-placement
task generated from a simple natural-language task description.

{_robot_context(profile)}

The active arm for this task is `{active_arm}`. The inactive arm
`{inactive_arm}` must stay null in the nominal graph.

Interactive task objects:
- {spec.moved_runtime_uid}: moved object from source `{spec.moved_source_uid}`.
- {_relative_reference_line(spec)}
{registry}

Config-stage LLM notes:
{notes}

{placement_rule}
"""


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
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a {profile.display_name} coordinated
shared-object task generated from a simple natural-language task description.

{_robot_context(profile)}

Both arms must act through one `CoordinatedPickment` action. The graph should
place that action in `left_arm_action` and keep `right_arm_action` null.

Interactive task object:
- {spec.moved_runtime_uid}: shared moved object from source `{spec.moved_source_uid}`.
{registry}

Config-stage LLM notes:
{notes}

The execution-stage LLM should generate a three-edge graph. First use
`CoordinatedPickment` to grasp the shared object with both grippers, lift it,
and move the object to the configured target pose. Then open both grippers in
parallel with `MoveJoints(control="hand", state="open")` to release it. Finally
return both empty arms to their initial arm joint poses in parallel. It must not
decompose this task into separate single-arm `PickUp`, `MoveHeldObject`, or
`Place` actions.
"""


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
        "The execution-stage LLM should generate graph JSON that completes the "
        "first moved object's pick-up, placement, retreat, and return before "
        "picking up the second moved object. The inactive arm must remain null "
        "throughout each object's sequence."
        if serial_sequence
        else "The execution-stage LLM should generate graph JSON that grasps both "
        "moved objects, stages and releases the first moved object, then stages "
        "and releases the second moved object while the first arm returns to its "
        "initial pose. Each arm must release its moved object before returning to "
        "its initial pose."
    )
    placement_rule = (
        "Dependent objects are placed serially in dependency order."
        if serial_sequence
        else "Orientation-preserving placements use object-aware Place directly "
        "after pickup, without MoveHeldObject. Each pose-sensitive placement uses "
        "exactly one direct final-pose MoveHeldObject, then release-only Place."
    )
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a {profile.display_name} dual-arm
relative-placement task generated from a simple natural-language task
description.

{_robot_context(profile)}

Both arms participate in the nominal graph:
{placement_lines}
{registry}

Config-stage LLM notes:
{notes}

{execution_rule}
{placement_rule}
"""


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
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a {profile.display_name} object-manipulation
hold-hover task generated from a simple natural-language task description.

{_robot_context(profile)}

Hold-hover task objects:
{object_lines}
{registry}

Config-stage LLM notes:
{notes}

The execution-stage LLM should pick up the selected object(s), move them to the
configured hover pose if needed, and keep the gripper(s) closed. It must not use
`Place` or return a holding arm to its initial qpos because the final state is
the object still hovering in the gripper.
"""


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
    return f"""### Atomic Action Class JSON Specs for {profile.display_name} Relative Placement

Use only the native atomic action class JSON specs shown below. The active arm
is `{active_arm}`. Keep `{inactive_arm}` null in
the nominal graph.

Use exactly these action patterns:
- Pick up `{spec.moved_runtime_uid}`:
  {pick_spec}
{release_actions}
- Return to initial qpos:
  {_format_initial_qpos_spec(active_arm, sample_interval=30)}
"""


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
    return f"""### Atomic Action Class JSON Specs for {profile.display_name} Dual-Arm Relative Placement

Use only the native atomic action class JSON specs shown below.
- `{first_arm}` manipulates `{first.moved_runtime_uid}`.
- `{second_arm}` manipulates `{second.moved_runtime_uid}`.

Use these action patterns:
- First arm pick-up:
  {first_pick_spec}
- Second arm pick-up:
  {second_pick_spec}
{first_release_actions}
{second_release_actions}
- Keep a holding arm closed:
  {_format_gripper_spec("<holding_arm>", "close", sample_interval=10)}
- Return to initial qpos:
  {_format_initial_qpos_spec("<released_arm>", sample_interval=30)}
"""


def _make_hold_hover_atom_actions_prompt(
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    blocks = "\n\n".join(
        _hold_hover_atom_action_block(placement) for placement in spec.placements
    )
    return f"""### Atomic Action Class JSON Specs for {profile.display_name} Object Manipulation

Use only the native atomic action class JSON specs shown below. The final state
must keep the listed object(s) held in closed grippers. Do not use `Place` and
do not return a holding arm to its initial qpos.

{blocks}
"""


def _hold_hover_atom_action_block(placement: _RelativePlacementLike) -> str:
    active_arm = f"{placement.active_side}_arm"
    return f"""Object `{placement.moved_runtime_uid}` hold-hover:
- Pick up:
  {_format_pick_up_spec(active_arm, placement.moved_runtime_uid)}
- Hover move:
  {_format_hover_move_spec(active_arm, placement)}
- Keep gripper closed:
  {_format_gripper_spec(active_arm, "close", sample_interval=10, post_hold_steps=20)}"""


def _make_coordinated_pickment_atom_actions_prompt(
    spec: _RelativeSpecLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
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
    return f"""### Atomic Action Class JSON Specs for {profile.display_name} Coordinated Pickment

Use only these native atomic action class JSON specs. `CoordinatedPickment`
controls both arms, so the nominal graph must put that spec in
`left_arm_action` and set `right_arm_action` to null. The following release
edge must then open both hands in parallel, followed by a return-to-initial edge
for both empty arms.

- Coordinated pick and move `{spec.moved_runtime_uid}`:
  {_format_coordinated_pickment_spec(spec)}

- Release `{spec.moved_runtime_uid}` from both grippers:
  left_arm_action: {left_release_spec}
  right_arm_action: {right_release_spec}

- Return both empty arms to initial poses:
  left_arm_action: {left_initial_spec}
  right_arm_action: {right_initial_spec}
"""


def make_basket_task_prompt(
    task_name: str,
    project_name: str,
    roles: _BasketRolesLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    left_target_text = _left_target_text(roles)
    right_target_text = _right_target_text(roles)
    target_pair_text = _target_pair_text(roles)
    target_plural = _target_plural_text(roles)
    left_pick_spec = _format_pick_up_spec(
        "left_arm",
        roles.left_target_runtime_uid,
    )
    right_pick_spec = _format_pick_up_spec(
        "right_arm",
        roles.right_target_runtime_uid,
    )
    left_high_spec = _format_pose_object_spec(
        "left_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_LEFT_RELEASE_OFFSET_Y, 0.22),
        sample_interval=45,
    )
    right_high_spec = _format_pose_object_spec(
        "right_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_RIGHT_RELEASE_OFFSET_Y, 0.22),
        sample_interval=45,
    )
    left_place_spec = _format_place_object_spec(
        "left_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_LEFT_RELEASE_OFFSET_Y, 0.12),
        sample_interval=80,
        lift_height=_PLACE_LIFT_HEIGHT,
    )
    right_place_spec = _format_place_object_spec(
        "right_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_RIGHT_RELEASE_OFFSET_Y, 0.12),
        sample_interval=80,
        lift_height=_PLACE_LIFT_HEIGHT,
    )
    right_close_spec = _format_gripper_spec(
        "right_arm",
        "close",
        sample_interval=10,
    )
    left_initial_spec = _format_initial_qpos_spec(
        "left_arm",
        sample_interval=30,
    )
    right_initial_spec = _format_initial_qpos_spec(
        "right_arm",
        sample_interval=30,
    )
    return f"""Task:
{task_name}: use the current {profile.display_name} configuration to place
{target_pair_text} into the {roles.container_runtime_uid}.

The task starts with both arms acting simultaneously:
the left arm grasps the left {left_target_text} while the right arm grasps the
right {right_target_text} in the same nominal graph edge. After both
{target_plural} are grasped, the left arm places its {left_target_text} into the
{roles.container_runtime_uid} and retreats upward. While the left arm returns
to its initial pose, the right arm must simultaneously begin placing its
already-grasped {right_target_text} by moving it to the high staging pose above
the {roles.container_runtime_uid}. The right arm then completes its placement
and returns to its initial pose.

Object and arm mapping:
- left_arm must only manipulate `{roles.left_target_runtime_uid}`.
- right_arm must only manipulate `{roles.right_target_runtime_uid}`.
- Both target objects must be released into `{roles.container_runtime_uid}`.

Generate one deterministic nominal graph with the following semantic sequence.
Do not add extra alignment, search, recovery, or monitor steps. Use `Place`
for each release-place step so lowering, gripper opening, and upward retreat
remain one atomic action. The left arm must finish its `Place` retreat
before the right arm enters the shared container workspace, but the left
return-to-initial action and the right high-staging action must execute
simultaneously in one graph edge. Generate exactly 6
nominal edges, one edge for each numbered step below. Do not split the
simultaneous grasp or the simultaneous left-return/right-staging action into
separate edges. Do not split a `Place` into separate lower-to-release,
open-gripper, or upward-retreat edges.

A target object is not considered placed when it is only above the
{roles.container_runtime_uid}. For each arm, the placement order must be: move
to a high staging pose above the container, then execute one `Place` at
the release pose inside the container, then return the arm to its initial pose.
Never use `target_qpos` source `initial` for an arm that has not already
released its held target object.

1. Pick up both target objects simultaneously:
   - left_arm_action: {left_pick_spec}
   - right_arm_action: {right_pick_spec}

2. Move the held left target object directly above the left half of the
   {roles.container_runtime_uid} while the right arm keeps holding its target:
   - left_arm_action: {left_high_spec}
   - right_arm_action: {right_close_spec}

3. Place the held left target object at the left release pose inside the
   {roles.container_runtime_uid}:
   - left_arm_action: {left_place_spec}
   - right_arm_action: {right_close_spec}

4. After the left gripper has retreated upward, return the left arm to its
   initial pose while simultaneously moving the held right target object
   directly above the right half of the {roles.container_runtime_uid}. This
   parallel handoff must remain one graph edge:
   - left_arm_action: {left_initial_spec}
   - right_arm_action: {right_high_spec}

5. Place the held right target object at the right release pose inside the
   {roles.container_runtime_uid}:
   - left_arm_action: null
   - right_arm_action: {right_place_spec}

6. Return the right arm to its initial pose after releasing the target object:
   - left_arm_action: null
   - right_arm_action: {right_initial_spec}

The final state is both `{roles.left_target_runtime_uid}` and
`{roles.right_target_runtime_uid}` resting inside `{roles.container_runtime_uid}`,
with both arms moved away from the container workspace. Always plan to the
current `{roles.container_runtime_uid}` object pose from the exported
{project_name} environment config.
"""


def make_basket_task_graph(
    task_name: str,
    roles: _BasketRolesLike,
) -> dict[str, Any]:
    left_pick_spec = _format_pick_up_spec(
        "left_arm",
        roles.left_target_runtime_uid,
    )
    right_pick_spec = _format_pick_up_spec(
        "right_arm",
        roles.right_target_runtime_uid,
    )
    left_high_spec = _format_pose_object_spec(
        "left_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_LEFT_RELEASE_OFFSET_Y, 0.22),
        sample_interval=45,
    )
    right_high_spec = _format_pose_object_spec(
        "right_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_RIGHT_RELEASE_OFFSET_Y, 0.22),
        sample_interval=45,
    )
    left_place_spec = _format_place_object_spec(
        "left_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_LEFT_RELEASE_OFFSET_Y, 0.12),
        sample_interval=80,
        lift_height=_PLACE_LIFT_HEIGHT,
    )
    right_place_spec = _format_place_object_spec(
        "right_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_RIGHT_RELEASE_OFFSET_Y, 0.12),
        sample_interval=80,
        lift_height=_PLACE_LIFT_HEIGHT,
    )
    right_close_spec = _format_gripper_spec(
        "right_arm",
        "close",
        sample_interval=10,
    )
    left_initial_spec = _format_initial_qpos_spec(
        "left_arm",
        sample_interval=30,
    )
    right_initial_spec = _format_initial_qpos_spec(
        "right_arm",
        sample_interval=30,
    )
    steps = [
        _nominal_step(
            "Pick up both target objects simultaneously",
            {
                "left_arm_action": left_pick_spec,
                "right_arm_action": right_pick_spec,
            },
        ),
        _nominal_step(
            "Move the held left target object above the container while the "
            "right arm keeps holding its target",
            {
                "left_arm_action": left_high_spec,
                "right_arm_action": right_close_spec,
            },
        ),
        _nominal_step(
            "Place the held left target object inside the container",
            {
                "left_arm_action": left_place_spec,
                "right_arm_action": right_close_spec,
            },
        ),
        _nominal_step(
            "Return the left arm to initial while staging the held right target",
            {
                "left_arm_action": left_initial_spec,
                "right_arm_action": right_high_spec,
            },
        ),
        _nominal_step(
            "Place the held right target object inside the container",
            {
                "left_arm_action": None,
                "right_arm_action": right_place_spec,
            },
        ),
        _nominal_step(
            "Return the right arm to its initial pose after release",
            {
                "left_arm_action": None,
                "right_arm_action": right_initial_spec,
            },
        ),
    ]
    return build_nominal_task_graph(task_name=task_name, steps=steps)


def make_basket_basic_background(
    project_name: str,
    roles: _BasketRolesLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
    object_registry: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    left_target_text = _left_target_text(roles)
    right_target_text = _right_target_text(roles)
    target_plural = _target_plural_text(roles)
    registry = _format_runtime_object_registry(object_registry)
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a basket-placement task template. The
current robot is {profile.display_name}.

{_robot_context(profile)}

Both robot bases are on the same long side of the table and face inward toward
the central {roles.container_runtime_uid}. The bases are intentionally kept
outside the table edge to avoid initial robot-table contact.

The interactive objects are:
- {roles.left_target_runtime_uid}: the {left_target_text} mesh initially on the
  positive-y side (source object {roles.left_target_source_uid}).
- {roles.right_target_runtime_uid}: the {right_target_text} mesh initially on the
  negative-y side (source object {roles.right_target_source_uid}).
- {roles.container_runtime_uid}: the target container near the center of the
  table (source object {roles.container_source_uid}).
{registry}

The nominal task starts with simultaneous dual-arm grasping. The left arm must
grasp {roles.left_target_runtime_uid} while the right arm grasps
{roles.right_target_runtime_uid} in the same graph edge. After both
{target_plural} are held, the left arm places
{roles.left_target_runtime_uid} into {roles.container_runtime_uid} with one
`Place`. The next graph edge is a parallel handoff: the left arm returns
to its initial pose while the right arm simultaneously moves its
already-grasped {roles.right_target_runtime_uid} to the high staging pose above
{roles.container_runtime_uid}. The right arm then places
{roles.right_target_runtime_uid} with one `Place` and returns to its
initial pose. To change the insertion order later, edit the task prompt sequence
and keep the same atomic action API.

The {roles.container_runtime_uid} area is a shared workspace. An arm should
complete its `Place` retreat before the other arm moves to the container,
otherwise the two arms may collide near the container. The right arm should keep
holding {roles.right_target_runtime_uid} while the left arm performs its
placement. Once that `Place` is complete, the right arm may move toward
the container while the left arm simultaneously returns to its initial pose; it
must not wait for the left return-to-initial motion to finish.

A target object at a high pose above `{roles.container_runtime_uid}` is only
staged, not placed. Each arm must execute a `Place` at the container
release pose before any return-to-initial motion.

Always plan to the current `{roles.container_runtime_uid}` object pose from the
environment config. Do not hard-code container coordinates in generated graph
actions.
"""


def make_basket_atom_actions_prompt(
    roles: _BasketRolesLike,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    left_high_spec = _format_pose_object_spec(
        "left_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_LEFT_RELEASE_OFFSET_Y, 0.22),
        sample_interval=45,
    )
    right_high_spec = _format_pose_object_spec(
        "right_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_RIGHT_RELEASE_OFFSET_Y, 0.22),
        sample_interval=45,
    )
    left_place_spec = _format_place_object_spec(
        "left_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_LEFT_RELEASE_OFFSET_Y, 0.12),
        sample_interval=80,
        lift_height=_PLACE_LIFT_HEIGHT,
    )
    right_place_spec = _format_place_object_spec(
        "right_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_RIGHT_RELEASE_OFFSET_Y, 0.12),
        sample_interval=80,
        lift_height=_PLACE_LIFT_HEIGHT,
    )
    return f"""### Atomic Action Class JSON Specs for {profile.display_name} Basket Placement

Use only the native atomic action class JSON specs shown below. Use
`robot_name="left_arm"` only for
`{roles.left_target_runtime_uid}` and `robot_name="right_arm"` only for
`{roles.right_target_runtime_uid}`.

The nominal task starts with simultaneous dual-arm pick-up, followed by a
left-first placement with an overlapped handoff to the right arm:
- The first nominal edge must use `atomic_action_class:"PickUp"` for both arms.
- While the left arm places its target, keep the right hand closed with a
  `target_qpos` whose source is `gripper_state` and state is `close`.
- After the left arm releases `{roles.left_target_runtime_uid}`, first move it
  upward to clear the container as part of the same `Place`.
- The next nominal edge must pair the left arm's initial `target_qpos` move with
  the right arm's object-referenced `target_object_pose` high-staging move. Do not split this
  parallel handoff into separate edges.
- After the parallel handoff edge, the remaining right-side placement steps put
  the actual action in `right_arm_action` and set `left_arm_action` to null.
- Never use initial `target_qpos` for an arm that is still holding a target object.

Use these action patterns:
- Left pick-up:
  {_format_pick_up_spec("left_arm", roles.left_target_runtime_uid)}
- Right pick-up:
  {_format_pick_up_spec("right_arm", roles.right_target_runtime_uid)}
- Left high staging:
  {left_high_spec}
- Left place action:
  {left_place_spec}
- Right high staging:
  {right_high_spec}
- Right place action:
  {right_place_spec}
- Keep a holding arm closed:
  {_format_gripper_spec("<holding_arm>", "close", sample_interval=10)}
- Return to initial qpos:
  {_format_initial_qpos_spec("<released_arm>", sample_interval=30)}
"""


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
) -> str:
    target_object_pose: dict[str, Any]
    if getattr(placement, "reference_is_initial_pose", False):
        if placement.release_position is None:
            raise ValueError(
                "CoordinatedPickment self-relative target requires release_position."
            )
        target_object_pose = {
            "reference": "absolute",
            "position": [float(value) for value in placement.release_position],
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
    return _compact_json(
        {
            "atomic_action_class": "CoordinatedPickment",
            "robot_name": "dual_arm",
            "control": "arm",
            "target_object": {
                "obj_name": placement.moved_runtime_uid,
                "affordance": "antipodal",
            },
            "target_object_pose": target_object_pose,
            "cfg": {
                "pre_grasp_distance": 0.1,
                "sample_interval": sample_interval,
                "hand_interp_steps": 10,
            },
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


def _format_place_object_spec(
    robot_name: str,
    obj_name: str,
    offset: tuple[float, float, float] | list[float],
    *,
    sample_interval: int,
    lift_height: float,
) -> str:
    x, y, z = offset
    return _format_place_spec(
        robot_name,
        {
            "reference": "object",
            "obj_name": obj_name,
            "offset": [float(x), float(y), float(z)],
        },
        sample_interval=sample_interval,
        lift_height=lift_height,
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


def _left_target_text(roles: _BasketRolesLike) -> str:
    return _display_noun(roles.left_target_noun)


def _right_target_text(roles: _BasketRolesLike) -> str:
    return _display_noun(roles.right_target_noun)


def _target_pair_text(roles: _BasketRolesLike) -> str:
    left_text = _left_target_text(roles)
    right_text = _right_target_text(roles)
    if left_text == right_text:
        return f"two {left_text} objects"
    return f"the left {left_text} and right {right_text}"


def _target_plural_text(roles: _BasketRolesLike) -> str:
    left_text = _left_target_text(roles)
    right_text = _right_target_text(roles)
    if left_text == right_text:
        return _plural(left_text)
    return "target objects"


def _display_noun(uid: str) -> str:
    return uid.replace("_", " ")


def _plural(noun: str) -> str:
    if noun.endswith("s"):
        return noun
    if noun.endswith(("ch", "sh", "x")):
        return f"{noun}es"
    return f"{noun}s"
