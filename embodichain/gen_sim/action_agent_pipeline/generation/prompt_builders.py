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
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

__all__ = [
    "make_agent_config",
    "make_arrangement_atom_actions_prompt",
    "make_arrangement_basic_background",
    "make_arrangement_task_prompt",
    "make_basket_atom_actions_prompt",
    "make_basket_basic_background",
    "make_basket_task_prompt",
    "make_relative_atom_actions_prompt",
    "make_relative_basic_background",
    "make_relative_task_prompt",
]

_BASKET_LEFT_RELEASE_OFFSET_Y = 0.04
_BASKET_RIGHT_RELEASE_OFFSET_Y = -0.04
_PLACE_LIFT_HEIGHT = 0.10
_RELEASE_ONLY_PLACE_SAMPLE_INTERVAL = 10
_EMPTY_HAND_RETREAT_SAMPLE_INTERVAL = 30
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


class _ArrangementSpecLike(Protocol):
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    order_by: str
    order_direction: str
    axis: str
    anchor: str
    steps: Sequence[_ArrangementStepLike]


def make_agent_config() -> dict[str, Any]:
    return {
        "TaskAgent": {
            "prompt_name": "generate_task_graph",
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


def make_arrangement_task_prompt(
    task_name: str,
    project_name: str,
    spec: _ArrangementSpecLike,
) -> str:
    edge_count = len(spec.steps) * 4
    step_blocks = "\n\n".join(
        _arrangement_step_prompt_block(index, step)
        for index, step in enumerate(spec.steps, start=1)
    )
    final_order = ", ".join(
        f"`{step.runtime_uid}` at slot {step.slot_index}" for step in spec.steps
    )
    return f"""Task:
{task_name}: {spec.task_prompt_summary}

This config was generated from a simple task description by the config-stage
LLM. The execution-stage LLM must now generate the graph JSON from this prompt.

Original simple task description:
{spec.task_description}

Arrangement plan:
- Layout axis: `{spec.axis}`. Slot 0 is the robot-view leftmost slot, and later
  slots move monotonically toward robot-view right.
- Anchor: `{spec.anchor}` in the exported {project_name} environment.
- Ordering rule: `{spec.order_by}` with direction `{spec.order_direction}`.
- Final order: {final_order}.

Generate one deterministic nominal graph with exactly {edge_count} nominal edges.
Use only the atomic action class JSON specs shown below. Do not add recovery,
monitor, search, alignment, or extra lift edges. Use `Place` for each
release-place step so lowering, gripper opening, and upward retreat remain one
atomic action. The arm not listed for a step must remain null.

{step_blocks}

Final state: all listed objects must rest near their assigned absolute XY slots
and remain upright. Use the exact absolute target_pose JSON specs shown above;
do not rewrite slot placement as object-referenced poses.
"""


def _arrangement_step_prompt_block(index: int, step: _ArrangementStepLike) -> str:
    active_arm = f"{step.active_side}_arm"
    active_slot = f"{step.active_side}_arm_action"
    inactive_slot = f"{'right' if step.active_side == 'left' else 'left'}_arm_action"
    base_edge = (index - 1) * 4
    return f"""{base_edge + 1}. Pick up `{step.runtime_uid}` for slot {step.slot_index}:
   - {active_slot}: {_format_pick_up_spec(active_arm, step.runtime_uid)}
   - {inactive_slot}: null

{base_edge + 2}. Move `{step.runtime_uid}` to the high staging pose above slot {step.slot_index}:
   - {active_slot}: {_format_pose_absolute_spec(active_arm, step.high_position, sample_interval=45)}
   - {inactive_slot}: null

{base_edge + 3}. Place `{step.runtime_uid}` at slot {step.slot_index}:
   - {active_slot}: {_format_place_absolute_spec(active_arm, step.release_position, sample_interval=80, lift_height=_PLACE_LIFT_HEIGHT)}
   - {inactive_slot}: null

{base_edge + 4}. Return `{active_arm}` to its initial pose:
   - {active_slot}: {_format_initial_qpos_spec(active_arm, sample_interval=30)}
   - {inactive_slot}: null"""


def make_arrangement_basic_background(
    project_name: str,
    spec: _ArrangementSpecLike,
) -> str:
    notes = spec.basic_background_notes or (
        "No extra scene notes were provided by the config-stage LLM."
    )
    object_lines = "\n".join(
        _arrangement_object_background_line(step) for step in spec.steps
    )
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a Dual-UR5 multi-object line arrangement
task generated from a simple natural-language task description.

The robot is a dual-UR5 composite robot with DH_PGI_140_80 parallel grippers:
- left_arm is the semantic robot-view left slot, mapped to the physical
  right_arm control part.
- right_arm is the semantic robot-view right slot, mapped to the physical
  left_arm control part.

Interactive task objects and target slots:
{object_lines}

Config-stage LLM notes:
{notes}
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


def make_arrangement_atom_actions_prompt(spec: _ArrangementSpecLike) -> str:
    blocks = "\n\n".join(_arrangement_atom_action_block(step) for step in spec.steps)
    return f"""### Atomic Action Class JSON Specs for Dual-UR5 Line Arrangement

Use only the native atomic action class JSON specs shown below. Each object is
moved to an absolute slot pose computed by the
config-stage generator. Keep the non-active arm null for each listed object.

{blocks}
"""


def _arrangement_atom_action_block(step: _ArrangementStepLike) -> str:
    active_arm = f"{step.active_side}_arm"
    return f"""Object `{step.runtime_uid}` to slot {step.slot_index}:
- Pick up:
  {_format_pick_up_spec(active_arm, step.runtime_uid)}
- High staging:
  {_format_pose_absolute_spec(active_arm, step.high_position, sample_interval=45)}
- Place:
  {_format_place_absolute_spec(active_arm, step.release_position, sample_interval=80, lift_height=_PLACE_LIFT_HEIGHT)}
- Return:
  {_format_initial_qpos_spec(active_arm, sample_interval=30)}"""


def make_relative_task_prompt(
    task_name: str,
    project_name: str,
    spec: _RelativeSpecLike,
) -> str:
    if len(spec.placements) > 1:
        return _make_dual_relative_task_prompt(task_name, project_name, spec)

    active_arm = f"{spec.active_side}_arm"
    inactive_slot = (
        "right_arm_action" if spec.active_side == "left" else "left_arm_action"
    )
    active_slot = f"{spec.active_side}_arm_action"
    action_sketch = _format_action_sketch(spec.action_sketch)
    pick_spec = _format_pick_up_spec(active_arm, spec.moved_runtime_uid)
    initial_spec = _format_initial_qpos_spec(active_arm, sample_interval=30)
    reference_line = _relative_reference_line(spec)
    final_planning_rule = _relative_final_planning_rule(project_name, spec)
    high_step_label = _relative_pose_step_label(spec, "high staging")
    release_step_label = _relative_pose_step_label(spec, "release")
    high_spec = _format_relative_pose_spec(
        active_arm,
        spec,
        pose_kind="high",
        sample_interval=45,
    )
    pose_sensitive = _is_pose_sensitive_placement(spec)
    if pose_sensitive:
        safe_high_spec = _format_relative_pose_spec(
            active_arm,
            spec,
            pose_kind="high",
            sample_interval=45,
            orientation_goal="preserve",
            orientation_axis="none",
            align_to=None,
        )
        high_orientation_spec = _format_relative_pose_spec(
            active_arm,
            spec,
            pose_kind="high",
            sample_interval=45,
        )
        release_move_spec = _format_relative_pose_spec(
            active_arm,
            spec,
            pose_kind="release",
            sample_interval=45,
        )
        place_spec = _format_release_only_place_spec(active_arm)
        retreat_spec = _format_empty_hand_retreat_spec(active_arm)
        edge_count = 7
        release_instruction = f"""2. Move the held object up to the {high_step_label} pose without changing orientation:
   - {active_slot}: {safe_high_spec}
   - {inactive_slot}: null

3. Adjust the held object orientation at the same safe high staging pose:
   - {active_slot}: {high_orientation_spec}
   - {inactive_slot}: null

4. Move the held object down to the {release_step_label} object pose:
   - {active_slot}: {release_move_spec}
   - {inactive_slot}: null

5. Release the held object in-place without moving the object pose:
   - {active_slot}: {place_spec}
   - {inactive_slot}: null

6. Retreat the now-empty end-effector upward:
   - {active_slot}: {retreat_spec}
   - {inactive_slot}: null

7. Return the active arm to its initial pose:
   - {active_slot}: {initial_spec}
   - {inactive_slot}: null"""
        high_instruction = release_instruction
        release_rule = (
            "For this pose-sensitive placement, first use `MoveHeldObject` to "
            "lift the object to the safe high staging pose while preserving its "
            "current orientation. Only then adjust orientation at that same high "
            "pose, move down to the final release object pose, and use the exact "
            "relative-zero release-only `Place` spec shown below."
        )
    else:
        place_spec = _format_relative_place_spec(
            active_arm,
            spec,
            sample_interval=80,
            lift_height=_PLACE_LIFT_HEIGHT,
        )
        edge_count = 4
        high_instruction = f"""2. Move the held object to the {high_step_label} pose:
   - {active_slot}: {high_spec}
   - {inactive_slot}: null

3. Place the held object at the {release_step_label} pose:
   - {active_slot}: {place_spec}
   - {inactive_slot}: null

4. Return the active arm to its initial pose:
   - {active_slot}: {initial_spec}
   - {inactive_slot}: null"""
        release_rule = (
            "Use `Place` for the release-place step so lowering, gripper "
            "opening, and upward retreat remain one atomic action."
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


def _make_dual_relative_task_prompt(
    task_name: str,
    project_name: str,
    spec: _RelativeSpecLike,
) -> str:
    first, second = spec.placements
    first_arm = f"{first.active_side}_arm"
    second_arm = f"{second.active_side}_arm"
    first_slot = f"{first.active_side}_arm_action"
    second_slot = f"{second.active_side}_arm_action"
    action_sketch = _format_action_sketch(spec.action_sketch)
    first_pick_spec = _format_pick_up_spec(first_arm, first.moved_runtime_uid)
    second_pick_spec = _format_pick_up_spec(second_arm, second.moved_runtime_uid)
    first_high_spec = _format_relative_pose_spec(
        first_arm,
        first,
        pose_kind="high",
        sample_interval=45,
    )
    second_high_spec = _format_relative_pose_spec(
        second_arm,
        second,
        pose_kind="high",
        sample_interval=45,
    )
    first_close_spec = _format_gripper_spec(
        first_arm,
        "close",
        sample_interval=10,
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
    first_reference_line = _relative_reference_line(first)
    second_reference_line = _relative_reference_line(second)
    final_planning_rule = _dual_relative_final_planning_rule(project_name, spec)
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
        ),
        (
            f"Move `{first.moved_runtime_uid}` to the high staging pose while "
            f"the other arm keeps holding `{second.moved_runtime_uid}`",
            {
                first_slot: first_high_spec,
                second_slot: second_close_spec,
            },
        ),
        *first_release_edges,
        (
            f"Return `{first_arm}` to its initial pose while moving "
            f"`{second.moved_runtime_uid}` to the high staging pose",
            {
                first_slot: first_initial_spec,
                second_slot: second_high_spec,
            },
        ),
        *second_release_edges,
        (
            f"Return `{second_arm}` to its initial pose",
            {
                first_slot: None,
                second_slot: second_initial_spec,
            },
        ),
    ]
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
                f"Lift `{placement.moved_runtime_uid}` to the safe high staging "
                "pose without changing orientation",
                {
                    active_slot: _format_relative_pose_spec(
                        active_arm,
                        placement,
                        pose_kind="high",
                        sample_interval=45,
                        orientation_goal="preserve",
                        orientation_axis="none",
                        align_to=None,
                    ),
                    waiting_slot: waiting_value,
                },
            ),
            (
                f"Adjust `{placement.moved_runtime_uid}` orientation at the same "
                "safe high staging pose",
                {
                    active_slot: _format_relative_pose_spec(
                        active_arm,
                        placement,
                        pose_kind="high",
                        sample_interval=45,
                    ),
                    waiting_slot: waiting_value,
                },
            ),
            (
                f"Move `{placement.moved_runtime_uid}` down to the final "
                "release object pose",
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
            f"Place `{placement.moved_runtime_uid}` at the release pose",
            {
                active_slot: _format_relative_place_spec(
                    active_arm,
                    placement,
                    sample_interval=80,
                    lift_height=_PLACE_LIFT_HEIGHT,
                ),
                waiting_slot: waiting_value,
            },
        )
    ]


def _dual_relative_release_rule(spec: _RelativeSpecLike) -> str:
    if any(_is_pose_sensitive_placement(placement) for placement in spec.placements):
        return (
            "For pose-sensitive placements, first lift the held object to the "
            "safe high staging pose with orientation preserved, then adjust "
            "orientation at the same high pose before moving down to the final "
            "release object pose. The following `Place` must be the exact "
            "relative-zero release-only spec shown below, and then the empty "
            "hand retreats upward. For preserve placements, keep the normal "
            "`Place` release-place action."
        )
    return (
        "Use `Place` for each release-place step so lowering, gripper opening, "
        "and upward retreat remain one atomic action."
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


def _relative_release_action_patterns(
    robot_name: str,
    placement: _RelativePlacementLike,
) -> str:
    if _is_pose_sensitive_placement(placement):
        return f"""- Final release object pose:
  {_format_relative_pose_spec(robot_name, placement, pose_kind="release", sample_interval=45)}
- Release-only Place:
  {_format_release_only_place_spec(robot_name)}
- Empty-hand retreat:
  {_format_empty_hand_retreat_spec(robot_name)}"""
    return f"""- Place at the release pose:
  {_format_relative_place_spec(robot_name, placement, sample_interval=80, lift_height=_PLACE_LIFT_HEIGHT)}"""


def _relative_high_action_patterns(
    robot_name: str,
    placement: _RelativePlacementLike,
) -> str:
    if _is_pose_sensitive_placement(placement):
        return f"""- Safe high staging without orientation change:
  {_format_relative_pose_spec(robot_name, placement, pose_kind="high", sample_interval=45, orientation_goal="preserve", orientation_axis="none", align_to=None)}
- High staging orientation adjustment:
  {_format_relative_pose_spec(robot_name, placement, pose_kind="high", sample_interval=45)}"""
    return f"""- {_relative_pose_step_label(placement, "High staging")}:
  {_format_relative_pose_spec(robot_name, placement, pose_kind="high", sample_interval=45)}"""


def make_relative_basic_background(
    project_name: str,
    spec: _RelativeSpecLike,
) -> str:
    if len(spec.placements) > 1:
        return _make_dual_relative_basic_background(project_name, spec)

    active_arm = f"{spec.active_side}_arm"
    inactive_arm = "right_arm" if spec.active_side == "left" else "left_arm"
    notes = spec.basic_background_notes or (
        "No extra scene notes were provided by the config-stage LLM."
    )
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a Dual-UR5 relative-placement task generated
from a simple natural-language task description.

The robot is a dual-UR5 composite robot with DH_PGI_140_80 parallel grippers:
- left_arm is the semantic robot-view left slot, mapped to the physical
  right_arm control part.
- right_arm is the semantic robot-view right slot, mapped to the physical
  left_arm control part.

The active arm for this task is `{active_arm}`. The inactive arm
`{inactive_arm}` must stay null in the nominal graph.

Interactive task objects:
- {spec.moved_runtime_uid}: moved object from source `{spec.moved_source_uid}`.
- {_relative_reference_line(spec)}

Config-stage LLM notes:
{notes}

The execution-stage LLM should generate graph JSON that grasps the moved object,
moves it to the configured high staging pose, releases it at the final pose, and
returns the active arm to its initial pose. Pose-sensitive placements must use a
safe high `MoveHeldObject` lift with orientation preserved before high-pose
orientation adjustment, then a final object-pose move followed by release-only
`Place`.
"""


def _make_dual_relative_basic_background(
    project_name: str,
    spec: _RelativeSpecLike,
) -> str:
    notes = spec.basic_background_notes or (
        "No extra scene notes were provided by the config-stage LLM."
    )
    placement_lines = "\n".join(
        f"- {placement.active_side}_arm moves `{placement.moved_runtime_uid}` "
        f"{_relative_relation_phrase(placement.relation)} "
        f"`{placement.reference_runtime_uid}`."
        for placement in spec.placements
    )
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a Dual-UR5 dual-arm relative-placement task
generated from a simple natural-language task description.

The robot is a dual-UR5 composite robot with DH_PGI_140_80 parallel grippers:
- left_arm is the semantic robot-view left slot, mapped to the physical
  right_arm control part.
- right_arm is the semantic robot-view right slot, mapped to the physical
  left_arm control part.

Both arms participate in the nominal graph:
{placement_lines}

Config-stage LLM notes:
{notes}

The execution-stage LLM should generate graph JSON that grasps both moved
objects, stages and releases the first moved object, then stages and releases
the second moved object while the first arm returns to its initial pose. Each
arm must release its moved object before returning to its initial pose.
Pose-sensitive placements must use a final `MoveHeldObject` object-pose move
safe high `MoveHeldObject` lift with orientation preserved before high-pose
orientation adjustment, then a final object-pose move followed by release-only
`Place`.
"""


def make_relative_atom_actions_prompt(spec: _RelativeSpecLike) -> str:
    if len(spec.placements) > 1:
        return _make_dual_relative_atom_actions_prompt(spec)

    active_arm = f"{spec.active_side}_arm"
    inactive_arm = "right_arm" if spec.active_side == "left" else "left_arm"
    high_actions = _relative_high_action_patterns(active_arm, spec)
    release_actions = _relative_release_action_patterns(active_arm, spec)
    return f"""### Atomic Action Class JSON Specs for Dual-UR5 Relative Placement

Use only the native atomic action class JSON specs shown below. The active arm
is `{active_arm}`. Keep `{inactive_arm}` null in
the nominal graph.

Use exactly these action patterns:
- Pick up `{spec.moved_runtime_uid}`:
  {_format_pick_up_spec(active_arm, spec.moved_runtime_uid)}
{high_actions}
{release_actions}
- Return to initial qpos:
  {_format_initial_qpos_spec(active_arm, sample_interval=30)}
"""


def _make_dual_relative_atom_actions_prompt(spec: _RelativeSpecLike) -> str:
    first, second = spec.placements
    first_arm = f"{first.active_side}_arm"
    second_arm = f"{second.active_side}_arm"
    first_high_actions = _relative_high_action_patterns(first_arm, first)
    second_high_actions = _relative_high_action_patterns(second_arm, second)
    first_release_actions = _relative_release_action_patterns(first_arm, first)
    second_release_actions = _relative_release_action_patterns(second_arm, second)
    return f"""### Atomic Action Class JSON Specs for Dual-UR5 Dual-Arm Relative Placement

Use only the native atomic action class JSON specs shown below.
- `{first_arm}` manipulates `{first.moved_runtime_uid}`.
- `{second_arm}` manipulates `{second.moved_runtime_uid}`.

Use these action patterns:
- First arm pick-up:
  {_format_pick_up_spec(first_arm, first.moved_runtime_uid)}
- Second arm pick-up:
  {_format_pick_up_spec(second_arm, second.moved_runtime_uid)}
{first_high_actions}
{first_release_actions}
{second_high_actions}
{second_release_actions}
- Keep a holding arm closed:
  {_format_gripper_spec("<holding_arm>", "close", sample_interval=10)}
- Return to initial qpos:
  {_format_initial_qpos_spec("<released_arm>", sample_interval=30)}
"""


def make_basket_task_prompt(
    task_name: str,
    project_name: str,
    roles: _BasketRolesLike,
) -> str:
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
{task_name}: use the current two-UR5 configuration to place
{target_pair_text} into the {roles.container_runtime_uid}.

The task starts with both arms acting simultaneously:
the left UR5 grasps the left {left_target_text} while the right UR5 grasps the
right {right_target_text} in the same nominal graph edge. After both
{target_plural} are grasped, the left UR5 places its {left_target_text} into the
{roles.container_runtime_uid} and retreats upward. While the left UR5 returns
to its initial pose, the right UR5 must simultaneously begin placing its
already-grasped {right_target_text} by moving it to the high staging pose above
the {roles.container_runtime_uid}. The right UR5 then completes its placement
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

4. After the left gripper has retreated upward, return the left UR5 to its
   initial pose while simultaneously moving the held right target object
   directly above the right half of the {roles.container_runtime_uid}. This
   parallel handoff must remain one graph edge:
   - left_arm_action: {left_initial_spec}
   - right_arm_action: {right_high_spec}

5. Place the held right target object at the right release pose inside the
   {roles.container_runtime_uid}:
   - left_arm_action: null
   - right_arm_action: {right_place_spec}

6. Return the right UR5 to its initial pose after releasing the target object:
   - left_arm_action: null
   - right_arm_action: {right_initial_spec}

The final state is both `{roles.left_target_runtime_uid}` and
`{roles.right_target_runtime_uid}` resting inside `{roles.container_runtime_uid}`,
with both arms moved away from the container workspace. Always plan to the
current `{roles.container_runtime_uid}` object pose from the exported
{project_name} environment config.
"""


def make_basket_basic_background(
    project_name: str,
    roles: _BasketRolesLike,
) -> str:
    left_target_text = _left_target_text(roles)
    right_target_text = _right_target_text(roles)
    target_plural = _target_plural_text(roles)
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for the UR5BreadBasket task template. The
current robot is a dual-UR5 composite robot with DH_PGI_140_80 parallel
grippers.

The robot is a dual-UR5 composite robot with two parallel grippers:
- left_arm is the semantic robot-view left slot, mapped to the physical
  right_arm control part.
- right_arm is the semantic robot-view right slot, mapped to the physical
  left_arm control part.

Both UR5 bases are on the same long side of the table and face inward toward
the central {roles.container_runtime_uid}. The bases are intentionally kept
outside the table edge to avoid initial robot-table contact.

The interactive objects are:
- {roles.left_target_runtime_uid}: the {left_target_text} mesh initially on the
  positive-y side (source object {roles.left_target_source_uid}).
- {roles.right_target_runtime_uid}: the {right_target_text} mesh initially on the
  negative-y side (source object {roles.right_target_source_uid}).
- {roles.container_runtime_uid}: the target container near the center of the
  table (source object {roles.container_source_uid}).

The nominal task starts with simultaneous dual-arm grasping. The left UR5 must
grasp {roles.left_target_runtime_uid} while the right UR5 grasps
{roles.right_target_runtime_uid} in the same graph edge. After both
{target_plural} are held, the left UR5 places
{roles.left_target_runtime_uid} into {roles.container_runtime_uid} with one
`Place`. The next graph edge is a parallel handoff: the left UR5 returns
to its initial pose while the right UR5 simultaneously moves its
already-grasped {roles.right_target_runtime_uid} to the high staging pose above
{roles.container_runtime_uid}. The right UR5 then places
{roles.right_target_runtime_uid} with one `Place` and returns to its
initial pose. To change the insertion order later, edit the task prompt sequence
and keep the same atomic action API.

The {roles.container_runtime_uid} area is a shared workspace. A UR5 should
complete its `Place` retreat before the other UR5 moves to the container,
otherwise the two arms may collide near the container. The right UR5 should keep
holding {roles.right_target_runtime_uid} while the left UR5 performs its
placement. Once that `Place` is complete, the right UR5 may move toward
the container while the left UR5 simultaneously returns to its initial pose; it
must not wait for the left return-to-initial motion to finish.

A target object at a high pose above `{roles.container_runtime_uid}` is only
staged, not placed. Each arm must execute a `Place` at the container
release pose before any return-to-initial motion.

Always plan to the current `{roles.container_runtime_uid}` object pose from the
environment config. Do not hard-code container coordinates in generated graph
actions.
"""


def make_basket_atom_actions_prompt(roles: _BasketRolesLike) -> str:
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
    return f"""### Atomic Action Class JSON Specs for UR5BreadBasket Dual-UR5 Placement

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
) -> str:
    return _compact_json(
        {
            "atomic_action_class": "PickUp",
            "robot_name": robot_name,
            "control": "arm",
            "target_object": {
                "obj_name": obj_name,
                "affordance": "antipodal",
            },
            "cfg": {
                "pre_grasp_distance": 0.08,
                "sample_interval": sample_interval,
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
    if getattr(placement, "reference_is_initial_pose", False):
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
    )


def _format_relative_place_spec(
    robot_name: str,
    placement: _RelativePlacementLike,
    *,
    sample_interval: int,
    lift_height: float,
) -> str:
    if getattr(placement, "reference_is_initial_pose", False):
        if placement.release_position is None:
            raise ValueError("Self-relative placement requires release position.")
        return _format_place_absolute_spec(
            robot_name,
            placement.release_position,
            sample_interval=sample_interval,
            lift_height=lift_height,
        )

    return _format_place_object_spec(
        robot_name,
        placement.reference_runtime_uid,
        placement.release_offset,
        sample_interval=sample_interval,
        lift_height=lift_height,
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
) -> str:
    target_object_pose = {
        "reference": "absolute",
        "position": [float(value) for value in position],
        "orientation_goal": orientation_goal,
        "orientation_axis": orientation_axis,
    }
    if align_to is not None:
        target_object_pose["align_to"] = align_to
    return _compact_json(
        {
            "atomic_action_class": "MoveHeldObject",
            "robot_name": robot_name,
            "control": "arm",
            "target_object_pose": target_object_pose,
            "cfg": {"sample_interval": sample_interval},
        }
    )


def _format_place_absolute_spec(
    robot_name: str,
    position: Sequence[float],
    *,
    sample_interval: int,
    lift_height: float,
) -> str:
    return _format_place_spec(
        robot_name,
        {
            "reference": "absolute",
            "position": [float(value) for value in position],
        },
        sample_interval=sample_interval,
        lift_height=lift_height,
    )


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
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _format_action_sketch(action_sketch: list[str]) -> str:
    return "\n".join(f"- {item}" for item in action_sketch)


def _relative_reference_line(spec: _RelativePlacementLike) -> str:
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
    return f"{label} relative to `{spec.reference_runtime_uid}`"


def _relative_final_planning_rule(
    project_name: str,
    spec: _RelativePlacementLike,
) -> str:
    if getattr(spec, "reference_is_initial_pose", False):
        return (
            "Use the exact absolute target_pose JSON specs shown above. Do not "
            "rewrite this self-relative task as an object-referenced pose, because "
            "the moved object would become a moving reference after pickup."
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
