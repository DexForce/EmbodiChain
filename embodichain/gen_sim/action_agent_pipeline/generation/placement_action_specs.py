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

"""Build placement-specific atomic-action JSON specifications.

Keeping JSON encoding below plan construction makes placement policy reusable
without coupling it to graph topology or human-readable templates.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
    defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.actions import DUAL_ARM_NAME
from embodichain.gen_sim.action_agent_pipeline.generation.action_spec_builders import (
    _add_surface_z_policy,
    _compact_json,
    _format_direct_absolute_place_spec,
    _format_pose_absolute_spec,
    _format_pose_object_spec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    RelativePlacementLike,
    StackingStepLike,
)

_ACTION_DEFAULTS = defaults_section("action")
_PLACE_LIFT_HEIGHT = float(_ACTION_DEFAULTS["place_lift_height"])
_DIRECT_PLACE_CARTESIAN_WAYPOINT_COUNT = int(
    _ACTION_DEFAULTS["direct_place_cartesian_waypoint_count"]
)
_COORDINATED_MAX_GRASP_SEPARATION_ANGLE_TO_WORLD_Y_DEGREES = float(
    _ACTION_DEFAULTS["coordinated_max_grasp_separation_angle_to_world_y_degrees"]
)
_STACKING_DEFAULTS = defaults_section("stacking")
_STACKING_NESTED_RELEASE_Z_OFFSET = float(_STACKING_DEFAULTS["nested_release_z_offset"])
_STACKING_SURFACE_CLEARANCE = float(_STACKING_DEFAULTS["clearance"])
_STACKING_MAX_APPROACH_RETRACT_Z = float(_STACKING_DEFAULTS["max_approach_retract_z"])
_SURFACE_RELEASE_Z_POLICY = "object_on_surface"
_SURFACE_RELEASE_CLEARANCE = DEFAULT_SURFACE_RELEASE_CLEARANCE
_USE_PLACEMENT_ALIGN_TO = object()

__all__ = [
    "_format_coordinated_pickment_spec",
    "_format_relative_pose_spec",
    "_format_direct_relative_place_spec",
    "_format_stacking_place_spec",
    "_surface_release_clearance",
    "_relative_surface_support",
    "_format_hover_move_spec",
    "_is_pose_sensitive_placement",
    "_relative_pose_step_label",
]


def _format_coordinated_pickment_spec(
    placement: RelativePlacementLike,
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


def _format_relative_pose_spec(
    robot_name: str,
    placement: RelativePlacementLike,
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
    placement: RelativePlacementLike,
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
    step: StackingStepLike,
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


def _surface_release_clearance(placement: RelativePlacementLike) -> float:
    return float(getattr(placement, "surface_clearance", _SURFACE_RELEASE_CLEARANCE))


def _relative_surface_support(
    placement: RelativePlacementLike,
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
    placement: RelativePlacementLike,
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


def _is_pose_sensitive_placement(placement: RelativePlacementLike) -> bool:
    return placement.orientation_goal != "preserve"


def _relative_pose_step_label(
    spec: RelativePlacementLike,
    label: str,
) -> str:
    if getattr(spec, "reference_is_initial_pose", False):
        return f"{label} at the absolute initial-position offset"
    if getattr(spec, "upright_in_place", False):
        return f"{label} at the initial XY on `{spec.reference_runtime_uid}`"
    return f"{label} relative to `{spec.reference_runtime_uid}`"
