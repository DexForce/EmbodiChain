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

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.contracts import (
    ACTION_AGENT_ENV_ID,
    DEFAULT_VIEWER_CAMERA_UID,
    SuccessTerm,
)
from embodichain.gen_sim.action_agent_pipeline.defaults import (
    generation_defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _ArrangementLineSpec,
    _RelativePlacementSpec,
    _RelativePlacementStepSpec,
    _StackingSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    RobotProfile,
    resolve_robot_profile,
)

__all__ = [
    "_make_arrangement_extensions_config",
    "_make_relative_extensions_config",
    "_make_stacking_extensions_config",
    "_object_in_container_success",
    "_validate_arrangement_bundle",
    "_validate_relative_bundle",
    "_validate_stacking_bundle",
    "_validate_success_uids",
]

_SUCCESS_DEFAULTS = generation_defaults_section("success")
_CONTAINER_XY_RADIUS = float(_SUCCESS_DEFAULTS["container_xy_radius"])
_CONTAINER_MIN_Z_OFFSET = float(_SUCCESS_DEFAULTS["container_min_z_offset"])
_CONTAINER_MAX_Z_OFFSET = float(_SUCCESS_DEFAULTS["container_max_z_offset"])
_STACKING_ANCHOR_XY_TOLERANCE = float(_SUCCESS_DEFAULTS["stacking_anchor_xy_tolerance"])
_STACKING_SUPPORT_XY_RADIUS = float(_SUCCESS_DEFAULTS["stacking_support_xy_radius"])
_SUPPORT_MIN_Z_OFFSET = float(_SUCCESS_DEFAULTS["support_min_z_offset"])
_SUPPORT_MAX_Z_OFFSET = float(_SUCCESS_DEFAULTS["support_max_z_offset"])
_OBJECT_MAX_TILT = float(_SUCCESS_DEFAULTS["object_max_tilt"])
_ARRANGEMENT_MAX_XY_TOLERANCE = float(_SUCCESS_DEFAULTS["arrangement_max_xy_tolerance"])
_ARRANGEMENT_SPACING_TOLERANCE_FRACTION = float(
    _SUCCESS_DEFAULTS["arrangement_spacing_tolerance_fraction"]
)
_COORDINATED_POSITION_TOLERANCE = float(
    _SUCCESS_DEFAULTS["coordinated_position_tolerance"]
)
_COORDINATED_LIFT_HEIGHT = float(_SUCCESS_DEFAULTS["coordinated_lift_height"])
_COORDINATED_GRIPPER_DISTANCE = float(_SUCCESS_DEFAULTS["coordinated_gripper_distance"])
_COORDINATED_MAX_TILT = float(_SUCCESS_DEFAULTS["coordinated_max_tilt"])
_COORDINATED_CLEAR_DISTANCE = float(_SUCCESS_DEFAULTS["coordinated_clear_distance"])
_ARM_INITIAL_QPOS_TOLERANCE = float(_SUCCESS_DEFAULTS["arm_initial_qpos_tolerance"])
_HOLD_LIFT_HEIGHT = float(_SUCCESS_DEFAULTS["hold_lift_height"])
_HOLD_GRIPPER_DISTANCE = float(_SUCCESS_DEFAULTS["hold_gripper_distance"])
_RELATIVE_SUPPORT_XY_RADIUS = float(_SUCCESS_DEFAULTS["relative_support_xy_radius"])
_ABSOLUTE_AXIS_TOLERANCE = float(_SUCCESS_DEFAULTS["absolute_axis_tolerance"])
_RELATIVE_AXIS_TOLERANCE = float(_SUCCESS_DEFAULTS["relative_axis_tolerance"])
_RELATIVE_ZERO_OFFSET_TOLERANCE = float(
    _SUCCESS_DEFAULTS["relative_zero_offset_tolerance"]
)


def _object_in_container_success(object_uid: str, container_uid: str) -> dict[str, Any]:
    return {
        "type": SuccessTerm.OBJECT_IN_CONTAINER,
        "object": object_uid,
        "container": container_uid,
        "radius": _CONTAINER_XY_RADIUS,
        "min_z_offset": _CONTAINER_MIN_Z_OFFSET,
        "max_z_offset": _CONTAINER_MAX_Z_OFFSET,
    }


def _make_relative_extensions_config(
    spec: _RelativePlacementSpec,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
    side_relation_xy_offsets: Callable[[str], tuple[float, float]],
) -> dict[str, Any]:
    profile = resolve_robot_profile(robot_profile)
    extensions = {
        **profile.runtime_extensions(),
        "ignore_terminations_during_agent": True,
        "viewer_camera_uid": DEFAULT_VIEWER_CAMERA_UID,
        "agent_success": _make_relative_success_spec(
            spec,
            side_relation_xy_offsets=side_relation_xy_offsets,
        ),
    }
    grasp_pose_overrides = _make_relative_grasp_pose_overrides(spec)
    if grasp_pose_overrides:
        extensions["agent_grasp_pose_overrides"] = grasp_pose_overrides
    return extensions


def _make_arrangement_extensions_config(
    spec: _ArrangementLineSpec,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> dict[str, Any]:
    profile = resolve_robot_profile(robot_profile)
    return {
        **profile.runtime_extensions(),
        "ignore_terminations_during_agent": True,
        "viewer_camera_uid": DEFAULT_VIEWER_CAMERA_UID,
        "agent_success": _make_arrangement_success_spec(spec),
    }


def _make_stacking_extensions_config(
    spec: _StackingSpec,
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> dict[str, Any]:
    profile = resolve_robot_profile(robot_profile)
    return {
        **profile.runtime_extensions(),
        "ignore_terminations_during_agent": True,
        "viewer_camera_uid": DEFAULT_VIEWER_CAMERA_UID,
        "agent_success": _make_stacking_success_spec(spec),
    }


def _make_stacking_success_spec(spec: _StackingSpec) -> dict[str, Any]:
    terms: list[dict[str, Any]] = []
    for step in spec.steps:
        if step.support_runtime_uid is None:
            terms.append(
                {
                    "type": SuccessTerm.OBJECT_XY_NEAR,
                    "object": step.runtime_uid,
                    "target_xy": [
                        float(spec.anchor_xy[0]),
                        float(spec.anchor_xy[1]),
                    ],
                    "tolerance": _STACKING_ANCHOR_XY_TOLERANCE,
                }
            )
        elif spec.stack_mode == "nested":
            terms.append(
                _object_in_container_success(
                    step.runtime_uid,
                    str(step.support_runtime_uid),
                )
            )
        else:
            terms.append(
                {
                    "type": SuccessTerm.OBJECT_ON_OBJECT,
                    "object": step.runtime_uid,
                    "support": step.support_runtime_uid,
                    "xy_radius": _STACKING_SUPPORT_XY_RADIUS,
                    "min_z_offset": _SUPPORT_MIN_Z_OFFSET,
                    "max_z_offset": _SUPPORT_MAX_Z_OFFSET,
                }
            )
        terms.append(
            {
                "type": SuccessTerm.OBJECT_NOT_FALLEN,
                "object": step.runtime_uid,
                "max_tilt": _OBJECT_MAX_TILT,
            }
        )
    return {"op": "all", "terms": terms}


def _make_arrangement_success_spec(spec: _ArrangementLineSpec) -> dict[str, Any]:
    terms: list[dict[str, Any]] = []
    # Scale the tolerance with slot spacing, but cap it so adjacent slots can
    # never satisfy each other's success region in a dense arrangement.
    xy_tolerance = min(
        _ARRANGEMENT_MAX_XY_TOLERANCE,
        float(spec.spacing) * _ARRANGEMENT_SPACING_TOLERANCE_FRACTION,
    )
    semantic_steps = sorted(spec.steps, key=lambda step: step.slot_index)
    ordered_objects = [step.runtime_uid for step in semantic_steps]
    arrangement_axis = _arrangement_success_axis(spec)
    terms.extend(
        [
            {
                "type": SuccessTerm.OBJECTS_COLLINEAR,
                "objects": ordered_objects,
                "axis": arrangement_axis,
                "tolerance": xy_tolerance,
            },
            {
                "type": SuccessTerm.OBJECTS_ORDERED,
                "objects": ordered_objects,
                "axis": arrangement_axis,
                "direction": "ascending",
                "tolerance": xy_tolerance,
            },
        ]
    )
    for step in semantic_steps:
        terms.extend(
            [
                {
                    "type": SuccessTerm.OBJECT_XY_NEAR,
                    "object": step.runtime_uid,
                    "target_xy": [float(step.target_xy[0]), float(step.target_xy[1])],
                    "tolerance": xy_tolerance,
                },
                {
                    "type": SuccessTerm.OBJECT_NOT_FALLEN,
                    "object": step.runtime_uid,
                    "max_tilt": _OBJECT_MAX_TILT,
                },
            ]
        )
    return {"op": "all", "terms": terms}


def _arrangement_success_axis(spec: _ArrangementLineSpec) -> str:
    if len(spec.steps) >= 2:
        x_values = [float(step.target_xy[0]) for step in spec.steps]
        y_values = [float(step.target_xy[1]) for step in spec.steps]
        x_span = max(x_values) - min(x_values)
        y_span = max(y_values) - min(y_values)
        return "x" if x_span >= y_span else "y"
    if spec.axis == "world_x":
        return "x"
    return "y"


def _make_relative_success_spec(
    spec: _RelativePlacementSpec,
    *,
    side_relation_xy_offsets: Callable[[str], tuple[float, float]],
) -> dict[str, Any]:
    if (
        spec.intent == "coordinated_pickment"
        and spec.coordinated_terminal_behavior is not None
    ):
        carrier = next(
            placement
            for placement in spec.placements
            if placement.intent == "coordinated_pickment"
        )
        payload_terms = [
            _make_relative_placement_success_spec(
                placement,
                side_relation_xy_offsets=side_relation_xy_offsets,
            )
            for placement in spec.placements
            if placement.intent == "place_relative"
        ]
        if spec.coordinated_terminal_behavior == "hold":
            if carrier.release_position is None:
                raise ValueError("Coordinated hold success requires a target position.")
            hover_position = list(carrier.release_position)
            hover_position[2] += float(carrier.hover_height)
            carrier_term: dict[str, Any] = {
                "op": "all",
                "terms": [
                    {
                        "type": SuccessTerm.OBJECT_POSITION_NEAR,
                        "object": carrier.moved_runtime_uid,
                        "target_position": hover_position,
                        "tolerance": _COORDINATED_POSITION_TOLERANCE,
                    },
                    {
                        "type": SuccessTerm.OBJECT_LIFTED,
                        "object": carrier.moved_runtime_uid,
                        "min_height": _COORDINATED_LIFT_HEIGHT,
                    },
                    {
                        "type": SuccessTerm.OBJECT_HELD_BY_BOTH_GRIPPERS,
                        "object": carrier.moved_runtime_uid,
                        "max_distance": _COORDINATED_GRIPPER_DISTANCE,
                    },
                    {
                        "type": SuccessTerm.OBJECT_NOT_FALLEN,
                        "object": carrier.moved_runtime_uid,
                        "max_tilt": _COORDINATED_MAX_TILT,
                    },
                ],
            }
        else:
            if carrier.release_position is None:
                raise ValueError(
                    "Coordinated place success requires a target position."
                )
            carrier_term = {
                "op": "all",
                "terms": [
                    {
                        "type": SuccessTerm.OBJECT_POSITION_NEAR,
                        "object": carrier.moved_runtime_uid,
                        "target_position": carrier.release_position,
                        "tolerance": _COORDINATED_POSITION_TOLERANCE,
                    },
                    {
                        "type": SuccessTerm.OBJECT_NOT_FALLEN,
                        "object": carrier.moved_runtime_uid,
                        "max_tilt": _COORDINATED_MAX_TILT,
                    },
                    {"type": SuccessTerm.BOTH_GRIPPERS_OPEN},
                    {
                        "type": SuccessTerm.GRIPPERS_CLEAR_OF_OBJECT,
                        "object": carrier.moved_runtime_uid,
                        "min_distance": _COORDINATED_CLEAR_DISTANCE,
                    },
                    {
                        "type": SuccessTerm.BOTH_ARMS_AT_INITIAL_QPOS,
                        "tolerance": _ARM_INITIAL_QPOS_TOLERANCE,
                    },
                ],
            }
        return {"op": "all", "terms": [*payload_terms, carrier_term]}
    if len(spec.placements) == 1:
        return _make_relative_placement_success_spec(
            spec.placements[0],
            side_relation_xy_offsets=side_relation_xy_offsets,
        )
    if all(placement.intent == "hold_hover" for placement in spec.placements):
        terms: list[dict[str, Any]] = []
        for placement in spec.placements:
            placement_success = _make_relative_placement_success_spec(
                placement,
                side_relation_xy_offsets=side_relation_xy_offsets,
            )
            terms.extend(placement_success["terms"])
        return {"op": "all", "terms": terms}
    return {
        "op": "all",
        "terms": [
            _make_relative_placement_success_spec(
                placement,
                side_relation_xy_offsets=side_relation_xy_offsets,
            )
            for placement in spec.placements
        ],
    }


def _make_relative_placement_success_spec(
    placement: _RelativePlacementStepSpec,
    *,
    side_relation_xy_offsets: Callable[[str], tuple[float, float]],
) -> dict[str, Any]:
    if placement.intent == "hold_hover":
        return {
            "op": "all",
            "terms": [
                {
                    "type": SuccessTerm.OBJECT_LIFTED,
                    "object": placement.moved_runtime_uid,
                    "min_height": _HOLD_LIFT_HEIGHT,
                },
                {
                    "type": SuccessTerm.OBJECT_HELD_BY_GRIPPER,
                    "object": placement.moved_runtime_uid,
                    "arm": f"{placement.active_side}_arm",
                    "max_distance": _HOLD_GRIPPER_DISTANCE,
                },
            ],
        }
    if placement.relation == "inside":
        return _object_in_container_success(
            placement.moved_runtime_uid,
            placement.reference_runtime_uid,
        )
    if placement.relation == "on" and placement.upright_in_place:
        if placement.release_position is None:
            raise ValueError(
                "Upright-in-place success requires an absolute release position."
            )
        return {
            "op": "all",
            "terms": [
                *_absolute_xy_success_terms(
                    placement.moved_runtime_uid,
                    placement.release_position,
                ),
                {
                    "type": SuccessTerm.OBJECT_NOT_FALLEN,
                    "object": placement.moved_runtime_uid,
                    "max_tilt": _OBJECT_MAX_TILT,
                },
            ],
        }
    if placement.relation == "on":
        return {
            "type": SuccessTerm.OBJECT_ON_OBJECT,
            "object": placement.moved_runtime_uid,
            "support": placement.reference_runtime_uid,
            "xy_radius": _RELATIVE_SUPPORT_XY_RADIUS,
            "min_z_offset": _SUPPORT_MIN_Z_OFFSET,
            "max_z_offset": _SUPPORT_MAX_Z_OFFSET,
        }

    if placement.reference_is_initial_pose:
        if placement.release_position is None:
            raise ValueError(
                "Self-relative success requires an absolute release position."
            )
        return {
            "op": "all",
            "terms": [
                *_absolute_xy_success_terms(
                    placement.moved_runtime_uid,
                    placement.release_position,
                ),
                {
                    "type": SuccessTerm.OBJECT_NOT_FALLEN,
                    "object": placement.moved_runtime_uid,
                    "max_tilt": _OBJECT_MAX_TILT,
                },
            ],
        }

    return {
        "op": "all",
        "terms": [
            *_relative_xy_success_terms(
                placement,
                side_relation_xy_offsets=side_relation_xy_offsets,
            ),
            {
                "type": SuccessTerm.OBJECT_NOT_FALLEN,
                "object": placement.moved_runtime_uid,
                "max_tilt": _OBJECT_MAX_TILT,
            },
        ],
    }


def _make_relative_grasp_pose_overrides(
    spec: _RelativePlacementSpec,
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for placement in spec.placements:
        if not placement.upright_in_place:
            continue
        overrides[placement.moved_runtime_uid] = {
            "mode": "upright_bottle_side_grasp",
            "preferred_height_fraction": [0.35, 0.75],
            "prefer_side_grasp": True,
            "avoid_ground_release_collision": True,
            "support": placement.reference_runtime_uid,
        }
    return overrides


def _absolute_xy_success_terms(
    object_uid: str,
    position: Sequence[float],
) -> list[dict[str, Any]]:
    return [
        {
            "type": SuccessTerm.OBJECT_AXIS_NEAR,
            "object": object_uid,
            "axis": axis,
            "target": float(position[index]),
            "tolerance": _ABSOLUTE_AXIS_TOLERANCE,
        }
        for index, axis in enumerate(("x", "y"))
    ]


def _relative_xy_success_terms(
    placement: _RelativePlacementStepSpec,
    *,
    side_relation_xy_offsets: Callable[[str], tuple[float, float]],
) -> list[dict[str, Any]]:
    x_offset, y_offset = side_relation_xy_offsets(placement.relation)
    return [
        {
            "type": SuccessTerm.OBJECT_AXIS_OFFSET_NEAR,
            "object": placement.moved_runtime_uid,
            "reference": placement.reference_runtime_uid,
            "axis": axis,
            "offset": offset,
            "tolerance": (
                _RELATIVE_AXIS_TOLERANCE if offset else _RELATIVE_ZERO_OFFSET_TOLERANCE
            ),
        }
        for axis, offset in (("x", x_offset), ("y", y_offset))
    ]


def _validate_relative_bundle(
    bundle: Mapping[str, Any],
    spec: _RelativePlacementSpec,
) -> None:
    gym_config = bundle["gym_config"]
    if gym_config.get("id") != ACTION_AGENT_ENV_ID:
        raise ValueError(f"Generated gym config must use {ACTION_AGENT_ENV_ID}.")
    _validate_robot_control_parts(gym_config)

    rigid_uid_list = [obj["uid"] for obj in gym_config.get("rigid_object", [])]
    if len(rigid_uid_list) != len(set(rigid_uid_list)):
        raise ValueError(f"Duplicate rigid object runtime uid(s): {rigid_uid_list}")
    rigid_uids = set(rigid_uid_list)
    background_uids = {obj["uid"] for obj in gym_config.get("background", [])}
    scene_uids = rigid_uids | background_uids
    moved_required = {placement.moved_runtime_uid for placement in spec.placements}
    missing_moved = moved_required - rigid_uids
    if missing_moved:
        raise ValueError(
            f"Generated relative config missing moved rigid object(s): {missing_moved}"
        )
    reference_required = {
        placement.reference_runtime_uid
        for placement in spec.placements
        if placement.intent in {"place_relative", "coordinated_pickment"}
    }
    missing_reference = reference_required - scene_uids
    if missing_reference:
        raise ValueError(
            f"Generated relative config missing reference object(s): {missing_reference}"
        )

    _validate_success_uids(
        gym_config["env"]["extensions"]["agent_success"],
        rigid_uids=rigid_uids,
        scene_uids=scene_uids,
    )
    registry = gym_config["env"]["events"]["register_info_to_env"]["params"]["registry"]
    registered = {entry["entity_cfg"]["uid"] for entry in registry}
    required = moved_required | reference_required
    if not required.issubset(registered):
        raise ValueError(
            f"Relative config registry missing: {sorted(required - registered)}"
        )


def _validate_arrangement_bundle(
    bundle: Mapping[str, Any],
    spec: _ArrangementLineSpec,
) -> None:
    gym_config = bundle["gym_config"]
    if gym_config.get("id") != ACTION_AGENT_ENV_ID:
        raise ValueError(f"Generated gym config must use {ACTION_AGENT_ENV_ID}.")
    _validate_robot_control_parts(gym_config)

    rigid_uid_list = [obj["uid"] for obj in gym_config.get("rigid_object", [])]
    if len(rigid_uid_list) != len(set(rigid_uid_list)):
        raise ValueError(f"Duplicate rigid object runtime uid(s): {rigid_uid_list}")
    rigid_uids = set(rigid_uid_list)
    background_uids = {obj["uid"] for obj in gym_config.get("background", [])}
    scene_uids = rigid_uids | background_uids
    required = {step.runtime_uid for step in spec.steps}
    missing = required - rigid_uids
    if missing:
        raise ValueError(
            f"Generated arrangement config missing moved rigid object(s): {missing}"
        )

    _validate_success_uids(
        gym_config["env"]["extensions"]["agent_success"],
        rigid_uids=rigid_uids,
        scene_uids=scene_uids,
    )
    registry = gym_config["env"]["events"]["register_info_to_env"]["params"]["registry"]
    registered = {entry["entity_cfg"]["uid"] for entry in registry}
    if not required.issubset(registered):
        raise ValueError(
            f"Arrangement config registry missing: {sorted(required - registered)}"
        )


def _validate_stacking_bundle(
    bundle: Mapping[str, Any],
    spec: _StackingSpec,
) -> None:
    gym_config = bundle["gym_config"]
    if gym_config.get("id") != ACTION_AGENT_ENV_ID:
        raise ValueError(f"Generated gym config must use {ACTION_AGENT_ENV_ID}.")
    _validate_robot_control_parts(gym_config)

    rigid_uid_list = [obj["uid"] for obj in gym_config.get("rigid_object", [])]
    if len(rigid_uid_list) != len(set(rigid_uid_list)):
        raise ValueError(f"Duplicate rigid object runtime uid(s): {rigid_uid_list}")
    rigid_uids = set(rigid_uid_list)
    background_uids = {obj["uid"] for obj in gym_config.get("background", [])}
    scene_uids = rigid_uids | background_uids
    required = {step.runtime_uid for step in spec.steps}
    if spec.anchor_runtime_uid is not None:
        required.add(spec.anchor_runtime_uid)
    missing = required - rigid_uids
    if missing:
        raise ValueError(
            f"Generated stacking config missing required rigid object(s): {missing}"
        )

    _validate_success_uids(
        gym_config["env"]["extensions"]["agent_success"],
        rigid_uids=rigid_uids,
        scene_uids=scene_uids,
    )
    registry = gym_config["env"]["events"]["register_info_to_env"]["params"]["registry"]
    registered = {entry["entity_cfg"]["uid"] for entry in registry}
    if not required.issubset(registered):
        raise ValueError(
            f"Stacking config registry missing: {sorted(required - registered)}"
        )


def _validate_robot_control_parts(gym_config: Mapping[str, Any]) -> None:
    robot_config = gym_config.get("robot", {})
    if not isinstance(robot_config, Mapping):
        raise ValueError("Generated gym config robot must be a mapping.")
    control_parts = robot_config.get("control_parts")
    if not isinstance(control_parts, Mapping):
        raise ValueError("Generated robot config must define control_parts.")
    extensions = gym_config.get("env", {}).get("extensions", {})
    arm_slots = extensions.get("agent_arm_slots", {})
    if not isinstance(arm_slots, Mapping):
        raise ValueError("Generated env extensions must define agent_arm_slots.")
    for side in ("left", "right"):
        slot = arm_slots.get(side)
        if not isinstance(slot, Mapping):
            raise ValueError(f"agent_arm_slots must define {side!r}.")
        for field_name in ("arm", "eef"):
            control_part = slot.get(field_name)
            if control_part not in control_parts:
                raise ValueError(
                    f"agent_arm_slots[{side!r}][{field_name!r}] references "
                    f"unknown control part {control_part!r}."
                )


def _validate_success_uids(
    success: Mapping[str, Any],
    *,
    rigid_uids: set[str],
    scene_uids: set[str],
) -> None:
    if success.get("op") in {"all", "and", "any", "or"}:
        for term in success.get("terms", []):
            _validate_success_uids(term, rigid_uids=rigid_uids, scene_uids=scene_uids)
        return

    success_type = str(success.get("type", success.get("func", ""))).lower()
    if success_type == "object_in_container":
        required_keys = ("object", "container")
    elif success_type in {"object_on_object", "object_on", "on_object"}:
        required_keys = ("object", "support")
    elif success_type in {
        "object_axis_offset_near",
        "object_relative_axis_near",
    }:
        required_keys = ("object", "reference")
    elif success_type in {"object_axis_near", "object_coordinate_near"}:
        required_keys = ("object",)
    elif success_type in {
        "object_position_near",
        "object_near_position",
        "object_xy_near",
        "object_near_xy",
    }:
        required_keys = ("object",)
    elif success_type in {"objects_collinear", "objects_ordered"}:
        objects = success.get("objects", success.get("object_uids", []))
        if (
            not isinstance(objects, Sequence)
            or isinstance(objects, (str, bytes, Mapping))
            or not objects
        ):
            raise ValueError(f"Success term {success_type!r} requires objects.")
        for uid in objects:
            if uid not in rigid_uids:
                raise ValueError(f"Invalid success uid reference object={uid!r}.")
        return
    elif success_type in {"object_not_fallen", "not_fallen"}:
        required_keys = ("object",)
    elif success_type in {"object_lifted", "object_height_above_initial"}:
        required_keys = ("object",)
    elif success_type in {
        "object_held_by_gripper",
        "object_gripper_near",
        "object_held_by_both_grippers",
        "grippers_clear_of_object",
    }:
        required_keys = ("object",)
    elif success_type in {"both_grippers_open", "both_arms_at_initial_qpos"}:
        return
    else:
        raise ValueError(f"Unsupported generated success term: {success_type!r}.")

    for key in required_keys:
        uid = success.get(key)
        valid_uids = rigid_uids if key == "object" else scene_uids
        if uid not in valid_uids:
            raise ValueError(f"Invalid success uid reference {key}={uid!r}.")
