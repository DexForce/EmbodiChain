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

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _ArrangementLineSpec,
    _BasketTaskRoles,
    _RelativePlacementSpec,
    _RelativePlacementStepSpec,
)

__all__ = [
    "_make_arrangement_extensions_config",
    "_make_extensions_config",
    "_make_relative_extensions_config",
    "_object_in_container_success",
    "_validate_arrangement_bundle",
    "_validate_bundle",
    "_validate_relative_bundle",
    "_validate_success_uids",
]


def _make_extensions_config(roles: _BasketTaskRoles) -> dict[str, Any]:
    return {
        **_make_dual_ur5_arm_slot_config(),
        "gripper_open_state": [0.0],
        "gripper_close_state": [0.04],
        "ignore_terminations_during_agent": True,
        "viewer_camera_uid": "cam_high",
        "agent_success": {
            "op": "all",
            "terms": [
                _object_in_container_success(
                    roles.left_target_runtime_uid,
                    roles.container_runtime_uid,
                ),
                _object_in_container_success(
                    roles.right_target_runtime_uid,
                    roles.container_runtime_uid,
                ),
            ],
        },
    }


def _object_in_container_success(object_uid: str, container_uid: str) -> dict[str, Any]:
    return {
        "type": "object_in_container",
        "object": object_uid,
        "container": container_uid,
        "radius": 0.2,
        "min_z_offset": -0.05,
        "max_z_offset": 0.35,
    }


def _make_dual_ur5_arm_slot_config() -> dict[str, Any]:
    return {
        "agent_arm_slots": {
            "left": {
                "arm": "right_arm",
                "eef": "right_eef",
            },
            "right": {
                "arm": "left_arm",
                "eef": "left_eef",
            },
        },
        "arm_aim_yaw_offset": {
            "left": 3.141592653589793,
            "right": 0.0,
        },
    }


def _make_relative_extensions_config(
    spec: _RelativePlacementSpec,
    *,
    side_relation_xy_offsets: Callable[[str], tuple[float, float]],
) -> dict[str, Any]:
    return {
        **_make_dual_ur5_arm_slot_config(),
        "gripper_open_state": [0.0],
        "gripper_close_state": [0.04],
        "ignore_terminations_during_agent": True,
        "viewer_camera_uid": "cam_high",
        "agent_success": _make_relative_success_spec(
            spec,
            side_relation_xy_offsets=side_relation_xy_offsets,
        ),
    }


def _make_arrangement_extensions_config(spec: _ArrangementLineSpec) -> dict[str, Any]:
    return {
        **_make_dual_ur5_arm_slot_config(),
        "gripper_open_state": [0.0],
        "gripper_close_state": [0.04],
        "ignore_terminations_during_agent": True,
        "viewer_camera_uid": "cam_high",
        "agent_success": _make_arrangement_success_spec(spec),
    }


def _make_arrangement_success_spec(spec: _ArrangementLineSpec) -> dict[str, Any]:
    terms: list[dict[str, Any]] = []
    for step in spec.steps:
        terms.extend(
            [
                {
                    "type": "object_xy_near",
                    "object": step.runtime_uid,
                    "target_xy": [float(step.target_xy[0]), float(step.target_xy[1])],
                    "tolerance": 0.05,
                },
                {
                    "type": "object_not_fallen",
                    "object": step.runtime_uid,
                    "max_tilt": 0.9,
                },
            ]
        )
    return {"op": "all", "terms": terms}


def _make_relative_success_spec(
    spec: _RelativePlacementSpec,
    *,
    side_relation_xy_offsets: Callable[[str], tuple[float, float]],
) -> dict[str, Any]:
    if len(spec.placements) == 1:
        return _make_relative_placement_success_spec(
            spec.placements[0],
            side_relation_xy_offsets=side_relation_xy_offsets,
        )
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
    if placement.relation == "inside":
        return _object_in_container_success(
            placement.moved_runtime_uid,
            placement.reference_runtime_uid,
        )
    if placement.relation == "on":
        return {
            "type": "object_on_object",
            "object": placement.moved_runtime_uid,
            "support": placement.reference_runtime_uid,
            "xy_radius": 0.08,
            "min_z_offset": 0.02,
            "max_z_offset": 0.35,
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
                    "type": "object_not_fallen",
                    "object": placement.moved_runtime_uid,
                    "max_tilt": 0.9,
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
                "type": "object_not_fallen",
                "object": placement.moved_runtime_uid,
                "max_tilt": 0.9,
            },
        ],
    }


def _absolute_xy_success_terms(
    object_uid: str,
    position: Sequence[float],
) -> list[dict[str, Any]]:
    return [
        {
            "type": "object_axis_near",
            "object": object_uid,
            "axis": axis,
            "target": float(position[index]),
            "tolerance": 0.05,
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
            "type": "object_axis_offset_near",
            "object": placement.moved_runtime_uid,
            "reference": placement.reference_runtime_uid,
            "axis": axis,
            "offset": offset,
            "tolerance": 0.05 if offset else 0.06,
        }
        for axis, offset in (("x", x_offset), ("y", y_offset))
    ]


def _validate_bundle(bundle: Mapping[str, Any], roles: _BasketTaskRoles) -> None:
    gym_config = bundle["gym_config"]
    if gym_config.get("id") != "AtomicActionsAgent-v3":
        raise ValueError("Generated gym config must use AtomicActionsAgent-v3.")
    if gym_config.get("robot", {}).get("uid") != "DualUR5":
        raise ValueError("Generated UR5 basket config must use DualUR5.")

    rigid_uids = {obj["uid"] for obj in gym_config.get("rigid_object", [])}
    background_uids = {obj["uid"] for obj in gym_config.get("background", [])}
    scene_uids = rigid_uids | background_uids
    required_rigid = {
        roles.left_target_runtime_uid,
        roles.right_target_runtime_uid,
    }
    if not required_rigid.issubset(rigid_uids):
        raise ValueError(
            f"Generated rigid objects missing: {sorted(required_rigid - rigid_uids)}"
        )
    if roles.container_runtime_uid not in scene_uids:
        raise ValueError(
            f"Generated scene objects missing container: {roles.container_runtime_uid}"
        )

    success = gym_config["env"]["extensions"]["agent_success"]
    for term in success.get("terms", []):
        if (
            term.get("object") not in rigid_uids
            or term.get("container") not in scene_uids
        ):
            raise ValueError(f"Invalid success term uid reference: {term}")


def _validate_relative_bundle(
    bundle: Mapping[str, Any],
    spec: _RelativePlacementSpec,
) -> None:
    gym_config = bundle["gym_config"]
    if gym_config.get("id") != "AtomicActionsAgent-v3":
        raise ValueError("Generated gym config must use AtomicActionsAgent-v3.")
    if gym_config.get("robot", {}).get("uid") != "DualUR5":
        raise ValueError("Generated relative placement config must use DualUR5.")

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
        placement.reference_runtime_uid for placement in spec.placements
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
    if gym_config.get("id") != "AtomicActionsAgent-v3":
        raise ValueError("Generated gym config must use AtomicActionsAgent-v3.")
    if gym_config.get("robot", {}).get("uid") != "DualUR5":
        raise ValueError("Generated arrangement config must use DualUR5.")

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
    elif success_type in {"object_xy_near", "object_near_xy"}:
        required_keys = ("object",)
    elif success_type in {"object_not_fallen", "not_fallen"}:
        required_keys = ("object",)
    else:
        raise ValueError(f"Unsupported generated success term: {success_type!r}.")

    for key in required_keys:
        uid = success.get(key)
        valid_uids = rigid_uids if key == "object" else scene_uids
        if uid not in valid_uids:
            raise ValueError(f"Invalid success uid reference {key}={uid!r}.")
