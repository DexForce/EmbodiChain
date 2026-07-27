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

"""Render stacking diagnostics from the shared deterministic action plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    StackingSpecLike,
    StackingStepLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.diagnostic_common import (
    _format_runtime_object_registry,
    _robot_context,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    RobotProfile,
    resolve_robot_profile,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_plan_builders import (
    _arm_action_slots,
    _stacking_step_edge_blocks,
)
from embodichain.gen_sim.action_agent_pipeline.prompts.template_loader import (
    render_prompt_template,
)

__all__ = [
    "make_stacking_task_prompt",
    "make_stacking_basic_background",
    "make_stacking_atom_actions_prompt",
]


def make_stacking_task_prompt(
    task_name: str,
    project_name: str,
    spec: StackingSpecLike,
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


def _stacking_step_edge_count(step: StackingStepLike) -> int:
    return 3 if step.orientation_goal == "preserve" else 7


def _stacking_step_prompt_block(
    start_edge: int,
    step: StackingStepLike,
    *,
    object_anchored: bool,
    stack_mode: str,
) -> str:
    active_arm, active_slot, inactive_slot = _arm_action_slots(step.active_side)
    actions = [
        edge_actions[active_slot]
        for _, edge_actions in _stacking_step_edge_blocks(
            step,
            object_anchored=object_anchored,
            stack_mode=stack_mode,
        )
    ]
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
            pickup_spec=actions[0],
            place_spec=actions[1],
            return_spec=actions[2],
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
        pickup_spec=actions[0],
        high_preserve_spec=actions[1],
        high_oriented_spec=actions[2],
        release_move_spec=actions[3],
        release_spec=actions[4],
        retreat_spec=actions[5],
        return_spec=actions[6],
    )


def make_stacking_basic_background(
    project_name: str,
    spec: StackingSpecLike,
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


def _stacking_object_background_line(step: StackingStepLike) -> str:
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
    spec: StackingSpecLike,
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
    step: StackingStepLike,
    *,
    object_anchored: bool,
    stack_mode: str,
) -> str:
    active_slot = f"{step.active_side}_arm_action"
    actions = [
        edge_actions[active_slot]
        for _, edge_actions in _stacking_step_edge_blocks(
            step,
            object_anchored=object_anchored,
            stack_mode=stack_mode,
        )
    ]
    if step.orientation_goal == "preserve":
        return render_prompt_template(
            "stacking_action_block_preserve.txt",
            runtime_uid=step.runtime_uid,
            layer_index=step.layer_index,
            pickup_spec=actions[0],
            place_spec=actions[1],
            return_spec=actions[2],
        )
    return render_prompt_template(
        "stacking_action_block_oriented.txt",
        runtime_uid=step.runtime_uid,
        layer_index=step.layer_index,
        pickup_spec=actions[0],
        high_preserve_spec=actions[1],
        high_oriented_spec=actions[2],
        final_pose_spec=actions[3],
        release_spec=actions[4],
        retreat_spec=actions[5],
        return_spec=actions[6],
    )
