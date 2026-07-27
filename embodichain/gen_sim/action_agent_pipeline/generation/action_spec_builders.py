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

"""Low-level JSON encoders for deterministic atomic-action specifications."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    defaults_section,
)

__all__ = [
    "_add_surface_z_policy",
    "_compact_json",
    "_format_direct_absolute_place_spec",
    "_format_empty_hand_retreat_spec",
    "_format_gripper_spec",
    "_format_initial_qpos_spec",
    "_format_pick_up_spec",
    "_format_place_spec",
    "_format_pose_absolute_spec",
    "_format_pose_object_spec",
    "_format_relative_eef_move_spec",
    "_format_release_only_place_spec",
]

_ACTION_DEFAULTS = defaults_section("action")
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


def _format_relative_eef_move_spec(
    robot_name: str,
    *,
    offset: Sequence[float],
    sample_interval: int,
    post_hold_steps: int = 0,
) -> str:
    cfg = {"sample_interval": int(sample_interval)}
    if post_hold_steps > 0:
        cfg["post_hold_steps"] = int(post_hold_steps)
    return _compact_json(
        {
            "atomic_action_class": "MoveEndEffector",
            "robot_name": robot_name,
            "control": "arm",
            "target_pose": {
                "reference": "relative",
                "offset": [float(value) for value in offset],
                "frame": "world",
            },
            "cfg": cfg,
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
