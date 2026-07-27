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

"""Render atomic-action reference diagnostics for relative tasks."""

from __future__ import annotations

import json

from embodichain.gen_sim.action_agent_pipeline.generation.action_spec_builders import (
    _format_empty_hand_retreat_spec,
    _format_gripper_spec,
    _format_initial_qpos_spec,
    _format_pick_up_spec,
    _format_release_only_place_spec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    RelativePlacementLike,
    RelativeSpecLike,
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
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_plan_builders import (
    _coordinated_pickment_graph_steps,
)
from embodichain.gen_sim.action_agent_pipeline.prompts.template_loader import (
    render_prompt_template,
)

__all__ = ["make_relative_atom_actions_prompt"]


def _relative_release_action_patterns(
    robot_name: str,
    placement: RelativePlacementLike,
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


def make_relative_atom_actions_prompt(
    spec: RelativeSpecLike,
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
    spec: RelativeSpecLike,
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
    spec: RelativeSpecLike,
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


def _hold_hover_atom_action_block(placement: RelativePlacementLike) -> str:
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
    spec: RelativeSpecLike,
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
