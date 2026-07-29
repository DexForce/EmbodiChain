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

"""Build shared deterministic action plans without rendering diagnostics.

This module owns edge ordering and arm-slot assignment. Both task-graph output
and diagnostic renderers consume these blocks so they cannot silently describe
different action sequences.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.protocol.actions import (
    LEFT_ARM_ACTION_KEY,
    RIGHT_ARM_ACTION_KEY,
)
from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_spec_builders import (
    _format_empty_hand_retreat_spec,
    _format_gripper_spec,
    _format_initial_qpos_spec,
    _format_pick_up_spec,
    _format_pose_absolute_spec,
    _format_relative_eef_move_spec,
    _format_release_only_place_spec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    ArrangementStepLike,
    RelativePlacementLike,
    RelativeSpecLike,
    StackingStepLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.nominal_graph import (
    NominalGraphStep,
)
from embodichain.gen_sim.action_agent_pipeline.generation.placement_action_specs import (
    _format_coordinated_pickment_spec,
    _format_direct_relative_place_spec,
    _format_hover_move_spec,
    _format_relative_pose_spec,
    _format_stacking_place_spec,
    _is_pose_sensitive_placement,
    _relative_pose_step_label,
)

__all__ = [
    "_arm_action_slots",
    "_single_arm_post_release_blocks",
    "_arrangement_step_edge_blocks",
    "_stacking_step_edge_blocks",
    "_single_relative_graph_steps",
    "_coordinated_pickment_graph_steps",
    "_coordinated_carrier_placement",
    "_dual_relative_graph_steps",
    "_dual_relative_edge_blocks",
    "_uses_serial_dual_sequence",
    "_serial_relative_edge_blocks",
    "_hold_hover_graph_steps",
    "_dual_relative_release_edge_blocks",
    "_nominal_step",
    "_action_dict",
]

_SURFACE_RELEASE_Z_POLICY = "object_on_surface"
_SURFACE_RELEASE_CLEARANCE = DEFAULT_SURFACE_RELEASE_CLEARANCE


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


def _arrangement_step_edge_blocks(
    step: ArrangementStepLike,
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


def _stacking_step_edge_blocks(
    step: StackingStepLike,
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


def _single_relative_graph_steps(
    spec: RelativeSpecLike,
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


def _coordinated_pickment_graph_steps(
    spec: RelativeSpecLike,
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
    spec: RelativeSpecLike,
) -> RelativePlacementLike:
    return next(
        placement
        for placement in spec.placements
        if placement.intent == "coordinated_pickment"
    )


def _dual_relative_graph_steps(spec: RelativeSpecLike) -> list[NominalGraphStep]:
    edge_blocks = _dual_relative_edge_blocks(spec)
    return [_nominal_step(title, actions) for title, actions in edge_blocks]


def _dual_relative_edge_blocks(
    spec: RelativeSpecLike,
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


def _uses_serial_dual_sequence(spec: RelativeSpecLike) -> bool:
    """Return whether placement dependencies require sequential execution."""
    first, second = spec.placements
    return (
        first.moved_runtime_uid == second.moved_runtime_uid
        or second.reference_source_uid == first.moved_source_uid
        or first.active_side == second.active_side
        or all(
            getattr(placement, "upright_in_place", False)
            for placement in spec.placements
        )
    )


def _serial_relative_edge_blocks(
    spec: RelativeSpecLike,
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


def _hold_hover_graph_steps(spec: RelativeSpecLike) -> list[NominalGraphStep]:
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
    placement: RelativePlacementLike,
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
