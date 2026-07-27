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

"""Render scene and execution background diagnostics for relative tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    _RelativeSpecLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.diagnostic_common import (
    _format_runtime_object_registry,
    _relative_reference_line,
    _relative_relation_phrase,
    _robot_context,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    RobotProfile,
    resolve_robot_profile,
)
from embodichain.gen_sim.action_agent_pipeline.generation.placement_action_specs import (
    _is_pose_sensitive_placement,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_plan_builders import (
    _uses_serial_dual_sequence,
)
from embodichain.gen_sim.action_agent_pipeline.prompts.template_loader import (
    render_prompt_template,
)

__all__ = ["make_relative_basic_background"]


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
