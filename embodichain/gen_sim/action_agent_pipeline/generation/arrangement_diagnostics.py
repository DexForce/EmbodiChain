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

"""Render arrangement diagnostics from the shared deterministic action plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    ArrangementSpecLike,
    ArrangementStepLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.diagnostic_common import (
    _format_indexed_edge_blocks,
    _format_runtime_object_registry,
    _robot_context,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    RobotProfile,
    resolve_robot_profile,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_plan_builders import (
    _arrangement_step_edge_blocks,
)
from embodichain.gen_sim.action_agent_pipeline.prompts.template_loader import (
    render_prompt_template,
)

__all__ = [
    "make_arrangement_task_prompt",
    "make_arrangement_basic_background",
    "make_arrangement_atom_actions_prompt",
]


def make_arrangement_task_prompt(
    task_name: str,
    project_name: str,
    spec: ArrangementSpecLike,
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


def _arrangement_world_axis(spec: ArrangementSpecLike) -> str:
    if len(spec.steps) >= 2:
        x_values = [float(step.target_xy[0]) for step in spec.steps]
        y_values = [float(step.target_xy[1]) for step in spec.steps]
        x_span = max(x_values) - min(x_values)
        y_span = max(y_values) - min(y_values)
        return "x" if x_span >= y_span else "y"
    if spec.axis == "world_x":
        return "x"
    return "y"


def _arrangement_step_edge_count(step: ArrangementStepLike) -> int:
    return len(_arrangement_step_edge_blocks(step))


def _arrangement_step_prompt_block(start_edge: int, step: ArrangementStepLike) -> str:
    return _format_indexed_edge_blocks(
        _arrangement_step_edge_blocks(step),
        start_index=start_edge,
    )


def make_arrangement_basic_background(
    project_name: str,
    spec: ArrangementSpecLike,
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


def _arrangement_object_background_line(step: ArrangementStepLike) -> str:
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
    spec: ArrangementSpecLike,
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


def _arrangement_atom_action_block(step: ArrangementStepLike) -> str:
    active_slot = f"{step.active_side}_arm_action"
    actions = [
        edge_actions[active_slot]
        for _, edge_actions in _arrangement_step_edge_blocks(step)
    ]
    if step.orientation_goal == "preserve":
        return render_prompt_template(
            "arrangement_action_block_preserve.txt",
            runtime_uid=step.runtime_uid,
            slot_index=step.slot_index,
            pickup_spec=actions[0],
            high_preserve_spec=actions[1],
            release_move_spec=actions[2],
            release_spec=actions[3],
            retreat_spec=actions[4],
            return_spec=actions[5],
        )
    return render_prompt_template(
        "arrangement_action_block_align.txt",
        runtime_uid=step.runtime_uid,
        slot_index=step.slot_index,
        pickup_spec=actions[0],
        high_preserve_spec=actions[1],
        high_align_spec=actions[2],
        release_move_spec=actions[3],
        release_spec=actions[4],
        retreat_spec=actions[5],
        return_spec=actions[6],
    )
