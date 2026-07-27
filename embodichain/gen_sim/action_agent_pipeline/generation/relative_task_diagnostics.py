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

"""Render human-readable relative-task execution diagnostics."""

from __future__ import annotations

import json

from embodichain.gen_sim.action_agent_pipeline.contracts import (
    LEFT_ARM_ACTION_KEY,
    RIGHT_ARM_ACTION_KEY,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_spec_builders import (
    _format_empty_hand_retreat_spec,
    _format_gripper_spec,
    _format_initial_qpos_spec,
    _format_pick_up_spec,
    _format_release_only_place_spec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    RelativeSpecLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.diagnostic_common import (
    _dual_relative_final_planning_rule,
    _format_action_sketch,
    _format_numbered_edge_blocks,
    _relative_final_planning_rule,
    _relative_reference_line,
    _relative_relation_phrase,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    RobotProfile,
    resolve_robot_profile,
)
from embodichain.gen_sim.action_agent_pipeline.generation.placement_action_specs import (
    _format_coordinated_pickment_spec,
    _format_direct_relative_place_spec,
    _format_hover_move_spec,
    _format_relative_pose_spec,
    _is_pose_sensitive_placement,
    _relative_pose_step_label,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_graph_builders import (
    make_relative_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_plan_builders import (
    _dual_relative_edge_blocks,
    _uses_serial_dual_sequence,
)
from embodichain.gen_sim.action_agent_pipeline.prompts.template_loader import (
    render_prompt_template,
)

_RELATIVE_COORDINATE_CONVENTION = render_prompt_template(
    "relative_coordinate_convention.txt"
)

__all__ = ["make_relative_task_prompt"]


def make_relative_task_prompt(
    task_name: str,
    project_name: str,
    spec: RelativeSpecLike,
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


def _make_coordinated_pickment_task_prompt(
    task_name: str,
    project_name: str,
    spec: RelativeSpecLike,
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


def _make_dual_relative_task_prompt(
    task_name: str,
    project_name: str,
    spec: RelativeSpecLike,
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


def _make_hold_hover_task_prompt(
    task_name: str,
    project_name: str,
    spec: RelativeSpecLike,
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


def _dual_relative_release_rule(spec: RelativeSpecLike) -> str:
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
