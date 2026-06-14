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
    "make_basket_atom_actions_prompt",
    "make_basket_basic_background",
    "make_basket_task_prompt",
    "make_relative_atom_actions_prompt",
    "make_relative_basic_background",
    "make_relative_task_prompt",
]

_BASKET_LEFT_RELEASE_OFFSET_Y = 0.04
_BASKET_RIGHT_RELEASE_OFFSET_Y = -0.04
_RELATIVE_COORDINATE_CONVENTION = """Coordinate convention for relative placement:
- `left_of` means positive world y relative to the reference object.
- `right_of` means negative world y relative to the reference object.
- `front_of` means positive world x relative to the reference object.
- `behind` means negative world x relative to the reference object.
- `inside` and `on` use the reference object's xy center."""


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


class _RelativeSpecLike(_RelativePlacementLike, Protocol):
    placements: Sequence[_RelativePlacementLike]
    task_prompt_summary: str
    task_description: str
    action_sketch: Sequence[str]
    basic_background_notes: str


def make_agent_config() -> dict[str, Any]:
    return {
        "TaskAgent": {
            "prompt_name": "generate_task_graph",
        },
        "CompileAgent": {
            "prompt_name": "compile_agent_graph",
        },
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
    high_spec = _format_pose_object_spec(
        active_arm,
        spec.reference_runtime_uid,
        spec.high_offset,
        sample_interval=45,
    )
    release_spec = _format_pose_object_spec(
        active_arm,
        spec.reference_runtime_uid,
        spec.release_offset,
        sample_interval=30,
    )
    open_spec = _format_gripper_spec(
        active_arm,
        "open",
        sample_interval=15,
        post_hold_steps=25,
    )
    retreat_spec = _format_pose_offset_spec(
        active_arm,
        (0.0, 0.0, 0.14),
        sample_interval=20,
    )
    initial_spec = _format_initial_qpos_spec(active_arm, sample_interval=30)
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
- Use `{spec.reference_runtime_uid}` as the spatial reference. Source object:
  `{spec.reference_source_uid}`.
- Goal relation: `{spec.relation}` ({_relative_relation_phrase(spec.relation)}).
- Active arm: `{active_arm}`.
- Keep every `{inactive_slot}` as null.

{_RELATIVE_COORDINATE_CONVENTION}

Generate one deterministic nominal graph with exactly 6 nominal edges. Use only
the atomic action class JSON specs shown below. Do not add recovery, monitor, search,
alignment, or extra lift edges. The inactive arm must remain null in every edge.

1. Pick up the moved object:
   - {active_slot}: {pick_spec}
   - {inactive_slot}: null

2. Move the held object to the high staging pose relative to the reference:
   - {active_slot}: {high_spec}
   - {inactive_slot}: null

3. Lower the held object to the release pose:
   - {active_slot}: {release_spec}
   - {inactive_slot}: null

4. Release the moved object:
   - {active_slot}: {open_spec}
   - {inactive_slot}: null

5. Move the empty gripper upward to clear the object:
   - {active_slot}: {retreat_spec}
   - {inactive_slot}: null

6. Return the active arm to its initial pose:
   - {active_slot}: {initial_spec}
   - {inactive_slot}: null

Final state: `{spec.moved_runtime_uid}` must be
{_relative_relation_phrase(spec.relation)} `{spec.reference_runtime_uid}`. Always
plan to the current object poses from the exported {project_name} environment
config. Do not hard-code absolute object coordinates in the generated graph.
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
    first_high_spec = _format_pose_object_spec(
        first_arm,
        first.reference_runtime_uid,
        first.high_offset,
        sample_interval=45,
    )
    first_release_spec = _format_pose_object_spec(
        first_arm,
        first.reference_runtime_uid,
        first.release_offset,
        sample_interval=30,
    )
    second_high_spec = _format_pose_object_spec(
        second_arm,
        second.reference_runtime_uid,
        second.high_offset,
        sample_interval=45,
    )
    second_release_spec = _format_pose_object_spec(
        second_arm,
        second.reference_runtime_uid,
        second.release_offset,
        sample_interval=30,
    )
    first_open_spec = _format_gripper_spec(
        first_arm,
        "open",
        sample_interval=15,
        post_hold_steps=25,
    )
    second_open_spec = _format_gripper_spec(
        second_arm,
        "open",
        sample_interval=15,
        post_hold_steps=25,
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
    first_retreat_spec = _format_pose_offset_spec(
        first_arm,
        (0.0, 0.0, 0.14),
        sample_interval=20,
    )
    second_retreat_spec = _format_pose_offset_spec(
        second_arm,
        (0.0, 0.0, 0.14),
        sample_interval=20,
    )
    first_initial_spec = _format_initial_qpos_spec(
        first_arm,
        sample_interval=30,
    )
    second_initial_spec = _format_initial_qpos_spec(
        second_arm,
        sample_interval=30,
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
- {first_slot} must manipulate `{first.moved_runtime_uid}`. Source object:
  `{first.moved_source_uid}`.
- {second_slot} must manipulate `{second.moved_runtime_uid}`. Source object:
  `{second.moved_source_uid}`.
- `{first.reference_runtime_uid}` is the spatial reference for
  `{first.moved_runtime_uid}`. Goal relation: `{first.relation}`
  ({_relative_relation_phrase(first.relation)}).
- `{second.reference_runtime_uid}` is the spatial reference for
  `{second.moved_runtime_uid}`. Goal relation: `{second.relation}`
  ({_relative_relation_phrase(second.relation)}).

{_RELATIVE_COORDINATE_CONVENTION}

Generate one deterministic nominal graph with exactly 10 nominal edges. Use only
the atomic action class JSON specs shown below. Do not add recovery, monitor, search,
alignment, or extra lift edges.

1. Pick up both moved objects simultaneously:
   - {first_slot}: {first_pick_spec}
   - {second_slot}: {second_pick_spec}

2. Move `{first.moved_runtime_uid}` to the high staging pose while the other arm
   keeps holding `{second.moved_runtime_uid}`:
   - {first_slot}: {first_high_spec}
   - {second_slot}: {second_close_spec}

3. Lower `{first.moved_runtime_uid}` to the release pose:
   - {first_slot}: {first_release_spec}
   - {second_slot}: {second_close_spec}

4. Release `{first.moved_runtime_uid}`:
   - {first_slot}: {first_open_spec}
   - {second_slot}: {second_close_spec}

5. Move the empty `{first_arm}` gripper upward to clear the workspace:
   - {first_slot}: {first_retreat_spec}
   - {second_slot}: {second_close_spec}

6. Return `{first_arm}` to its initial pose while moving `{second.moved_runtime_uid}`
   to the high staging pose:
   - {first_slot}: {first_initial_spec}
   - {second_slot}: {second_high_spec}

7. Lower `{second.moved_runtime_uid}` to the release pose:
   - {first_slot}: null
   - {second_slot}: {second_release_spec}

8. Release `{second.moved_runtime_uid}`:
   - {first_slot}: null
   - {second_slot}: {second_open_spec}

9. Move the empty `{second_arm}` gripper upward to clear the workspace:
   - {first_slot}: null
   - {second_slot}: {second_retreat_spec}

10. Return `{second_arm}` to its initial pose:
   - {first_slot}: null
   - {second_slot}: {second_initial_spec}

Final state: `{first.moved_runtime_uid}` must be
{_relative_relation_phrase(first.relation)} `{first.reference_runtime_uid}`, and
`{second.moved_runtime_uid}` must be {_relative_relation_phrase(second.relation)}
`{second.reference_runtime_uid}`. Always plan to the current object poses from the
exported {project_name} environment config. Do not hard-code absolute object
coordinates in the generated graph.
"""


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
- left_arm is the UR5 outside the left side of the table's near long edge.
- right_arm is the UR5 outside the right side of the table's near long edge.

The active arm for this task is `{active_arm}`. The inactive arm
`{inactive_arm}` must stay null in the nominal graph.

Interactive task objects:
- {spec.moved_runtime_uid}: moved object from source `{spec.moved_source_uid}`.
- {spec.reference_runtime_uid}: reference object from source
  `{spec.reference_source_uid}`.

Config-stage LLM notes:
{notes}

The execution-stage LLM should generate graph JSON that grasps the moved object,
moves it to a high staging pose relative to the current reference object pose,
lowers to the release pose, opens the gripper, retreats upward, and returns the
active arm to its initial pose.
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
- left_arm is the UR5 outside the left side of the table's near long edge.
- right_arm is the UR5 outside the right side of the table's near long edge.

Both arms participate in the nominal graph:
{placement_lines}

Config-stage LLM notes:
{notes}

The execution-stage LLM should generate graph JSON that grasps both moved
objects, places the first moved object, retreats the first arm, then places the
second moved object while the first arm returns to its initial pose. Each arm
must release its moved object before returning to its initial pose.
"""


def make_relative_atom_actions_prompt(spec: _RelativeSpecLike) -> str:
    if len(spec.placements) > 1:
        return _make_dual_relative_atom_actions_prompt(spec)

    active_arm = f"{spec.active_side}_arm"
    inactive_arm = "right_arm" if spec.active_side == "left" else "left_arm"
    high_spec = _format_pose_object_spec(
        active_arm,
        spec.reference_runtime_uid,
        spec.high_offset,
        sample_interval=45,
    )
    release_spec = _format_pose_object_spec(
        active_arm,
        spec.reference_runtime_uid,
        spec.release_offset,
        sample_interval=30,
    )
    return f"""### Atomic Action Class JSON Specs for Dual-UR5 Relative Placement

Use only atomic action class JSON specs backed by `PickUpAction`, `MoveAction`, and
`PlaceAction`. The active arm is `{active_arm}`. Keep `{inactive_arm}` null in
the nominal graph.

Use exactly these action patterns:
- Pick up `{spec.moved_runtime_uid}`:
  {_format_pick_up_spec(active_arm, spec.moved_runtime_uid)}
- High staging relative to `{spec.reference_runtime_uid}`:
  {high_spec}
- Release pose relative to `{spec.reference_runtime_uid}`:
  {release_spec}
- Release the held object:
  {_format_gripper_spec(active_arm, "open", sample_interval=15, post_hold_steps=25)}
- Retreat upward:
  {_format_pose_offset_spec(active_arm, (0.0, 0.0, 0.14), sample_interval=20)}
- Return to initial qpos:
  {_format_initial_qpos_spec(active_arm, sample_interval=30)}
"""


def _make_dual_relative_atom_actions_prompt(spec: _RelativeSpecLike) -> str:
    first, second = spec.placements
    first_arm = f"{first.active_side}_arm"
    second_arm = f"{second.active_side}_arm"
    first_high_spec = _format_pose_object_spec(
        first_arm,
        first.reference_runtime_uid,
        first.high_offset,
        sample_interval=45,
    )
    first_release_spec = _format_pose_object_spec(
        first_arm,
        first.reference_runtime_uid,
        first.release_offset,
        sample_interval=30,
    )
    second_high_spec = _format_pose_object_spec(
        second_arm,
        second.reference_runtime_uid,
        second.high_offset,
        sample_interval=45,
    )
    second_release_spec = _format_pose_object_spec(
        second_arm,
        second.reference_runtime_uid,
        second.release_offset,
        sample_interval=30,
    )
    return f"""### Atomic Action Class JSON Specs for Dual-UR5 Dual-Arm Relative Placement

Use only atomic action class JSON specs backed by `PickUpAction`, `MoveAction`, and
`PlaceAction`.
- `{first_arm}` manipulates `{first.moved_runtime_uid}`.
- `{second_arm}` manipulates `{second.moved_runtime_uid}`.

Use these action patterns:
- First arm pick-up:
  {_format_pick_up_spec(first_arm, first.moved_runtime_uid)}
- Second arm pick-up:
  {_format_pick_up_spec(second_arm, second.moved_runtime_uid)}
- First high staging:
  {first_high_spec}
- First release pose:
  {first_release_spec}
- Second high staging:
  {second_high_spec}
- Second release pose:
  {second_release_spec}
- Release an object:
  {_format_gripper_spec("<assigned_arm>", "open", sample_interval=15, post_hold_steps=25)}
- Keep a holding arm closed:
  {_format_gripper_spec("<holding_arm>", "close", sample_interval=10)}
- Retreat upward:
  {_format_pose_offset_spec("<assigned_arm>", (0.0, 0.0, 0.14), sample_interval=20)}
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
    left_release_spec = _format_pose_object_spec(
        "left_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_LEFT_RELEASE_OFFSET_Y, 0.12),
        sample_interval=30,
    )
    right_high_spec = _format_pose_object_spec(
        "right_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_RIGHT_RELEASE_OFFSET_Y, 0.22),
        sample_interval=45,
    )
    right_release_spec = _format_pose_object_spec(
        "right_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_RIGHT_RELEASE_OFFSET_Y, 0.12),
        sample_interval=30,
    )
    left_open_spec = _format_gripper_spec(
        "left_arm",
        "open",
        sample_interval=15,
        post_hold_steps=25,
    )
    right_open_spec = _format_gripper_spec(
        "right_arm",
        "open",
        sample_interval=15,
        post_hold_steps=25,
    )
    right_close_spec = _format_gripper_spec(
        "right_arm",
        "close",
        sample_interval=10,
    )
    left_retreat_spec = _format_pose_offset_spec(
        "left_arm",
        (0.0, 0.0, 0.14),
        sample_interval=20,
    )
    right_retreat_spec = _format_pose_offset_spec(
        "right_arm",
        (0.0, 0.0, 0.14),
        sample_interval=20,
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
Do not add extra alignment, search, recovery, or monitor steps. Do include the
specified post-release retreat and return-to-initial steps. The left arm must
finish its upward retreat before the right arm enters the shared container
workspace, but the left return-to-initial action and the right high-staging
action must execute simultaneously in one graph edge. Generate exactly 10
nominal edges, one edge for each numbered step below. Do not split the
simultaneous grasp or the simultaneous left-return/right-staging action into
separate edges. Do not merge, reorder, or omit the lower-to-release,
open-gripper, upward-retreat, or final right return-to-initial edges.

A target object is not considered placed when it is only above the
{roles.container_runtime_uid}. For each arm, the placement order must be: move
to a high staging pose above the container, lower to the release pose inside the
container, use `target_qpos` with source `gripper_state` and state `open`,
move the empty gripper upward, then return the arm to its initial pose. Never
use `target_qpos` source `initial` for an arm that has not already released its
held target object.

1. Pick up both target objects simultaneously:
   - left_arm_action: {left_pick_spec}
   - right_arm_action: {right_pick_spec}

2. Move the held left target object directly above the left half of the
   {roles.container_runtime_uid} while the right arm keeps holding its target:
   - left_arm_action: {left_high_spec}
   - right_arm_action: {right_close_spec}

3. Lower the held left target object to the left release pose inside the
   {roles.container_runtime_uid}:
   - left_arm_action: {left_release_spec}
   - right_arm_action: {right_close_spec}

4. Release the left target object into the {roles.container_runtime_uid}:
   - left_arm_action: {left_open_spec}
   - right_arm_action: {right_close_spec}

5. Move the empty left gripper upward to clear the container:
   - left_arm_action: {left_retreat_spec}
   - right_arm_action: {right_close_spec}

6. After the left gripper has retreated upward, return the left UR5 to its
   initial pose while simultaneously moving the held right target object
   directly above the right half of the {roles.container_runtime_uid}. This
   parallel handoff must remain one graph edge:
   - left_arm_action: {left_initial_spec}
   - right_arm_action: {right_high_spec}

7. Lower the held right target object to the right release pose inside the
   {roles.container_runtime_uid}:
   - left_arm_action: null
   - right_arm_action: {right_release_spec}

8. Release the right target object into the {roles.container_runtime_uid}:
   - left_arm_action: null
   - right_arm_action: {right_open_spec}

9. Move the empty right gripper upward to clear the container:
   - left_arm_action: null
   - right_arm_action: {right_retreat_spec}

10. Return the right UR5 to its initial pose after releasing the target object:
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
- left_arm is the UR5 outside the left side of the table's near long edge.
- right_arm is the UR5 outside the right side of the table's near long edge.

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
{roles.left_target_runtime_uid} into {roles.container_runtime_uid}, releases
it, and retreats upward. The next graph edge is a parallel handoff: the left
UR5 returns to its initial pose while the right UR5 simultaneously moves its
already-grasped {roles.right_target_runtime_uid} to the high staging pose above
{roles.container_runtime_uid}. The right UR5 then lowers and releases
{roles.right_target_runtime_uid}, retreats upward, and returns to its initial
pose. To change the insertion order later, edit the task prompt sequence and
keep the same atomic action API.

The {roles.container_runtime_uid} area is a shared workspace. After a UR5
releases a target object, it should retreat upward before the other UR5 moves
to the container, otherwise the two arms may collide near the container. The
right UR5 should keep holding {roles.right_target_runtime_uid} while the left
UR5 performs its placement and upward retreat. Once that retreat is complete,
the right UR5 may move toward the container while the left UR5 simultaneously
returns to its initial pose; it must not wait for the left return-to-initial
motion to finish.

A target object at a high pose above `{roles.container_runtime_uid}` is only
staged, not placed. Each arm must lower the held object into the container
release pose and open the gripper before any return-to-initial motion.

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
    left_release_spec = _format_pose_object_spec(
        "left_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_LEFT_RELEASE_OFFSET_Y, 0.12),
        sample_interval=30,
    )
    right_high_spec = _format_pose_object_spec(
        "right_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_RIGHT_RELEASE_OFFSET_Y, 0.22),
        sample_interval=45,
    )
    right_release_spec = _format_pose_object_spec(
        "right_arm",
        roles.container_runtime_uid,
        (0.0, _BASKET_RIGHT_RELEASE_OFFSET_Y, 0.12),
        sample_interval=30,
    )
    return f"""### Atomic Action Class JSON Specs for UR5BreadBasket Dual-UR5 Placement

Use only atomic action class JSON specs backed by `PickUpAction`, `MoveAction`, and
`PlaceAction`. Use `robot_name="left_arm"` only for
`{roles.left_target_runtime_uid}` and `robot_name="right_arm"` only for
`{roles.right_target_runtime_uid}`.

The nominal task starts with simultaneous dual-arm pick-up, followed by a
left-first placement with an overlapped handoff to the right arm:
- The first nominal edge must use `atomic_action_class:"PickUpAction"` for both arms.
- While the left arm places its target, keep the right hand closed with a
  `target_qpos` whose source is `gripper_state` and state is `close`.
- After the left arm releases `{roles.left_target_runtime_uid}`, first move it
  upward to clear the container.
- The next nominal edge must pair the left arm's initial `target_qpos` move with
  the right arm's object-referenced `target_pose` high-staging move. Do not split this
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
- Left release pose:
  {left_release_spec}
- Right high staging:
  {right_high_spec}
- Right release pose:
  {right_release_spec}
- Release an object:
  {_format_gripper_spec("<assigned_arm>", "open", sample_interval=15, post_hold_steps=25)}
- Keep a holding arm closed:
  {_format_gripper_spec("<holding_arm>", "close", sample_interval=10)}
- Retreat upward:
  {_format_pose_offset_spec("<assigned_arm>", (0.0, 0.0, 0.14), sample_interval=20)}
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
            "atomic_action_class": "PickUpAction",
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
) -> str:
    x, y, z = offset
    return _compact_json(
        {
            "atomic_action_class": "MoveAction",
            "robot_name": robot_name,
            "control": "arm",
            "target_pose": {
                "reference": "object",
                "obj_name": obj_name,
                "offset": [float(x), float(y), float(z)],
                "orientation": "current",
            },
            "cfg": {"sample_interval": sample_interval},
        }
    )


def _format_pose_offset_spec(
    robot_name: str,
    offset: tuple[float, float, float],
    *,
    sample_interval: int = 20,
) -> str:
    dx, dy, dz = offset
    return _compact_json(
        {
            "atomic_action_class": "MoveAction",
            "robot_name": robot_name,
            "control": "arm",
            "target_pose": {
                "reference": "relative",
                "offset": [float(dx), float(dy), float(dz)],
                "frame": "world",
            },
            "cfg": {"sample_interval": sample_interval},
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
            "atomic_action_class": "MoveAction",
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
            "atomic_action_class": "MoveAction",
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
