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

"""Pure validation and normalization for Action Agent atomic-action specs.

This dependency-light boundary must remain importable without constructing a
simulator, planner, grasp generator, or runtime action executor.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from embodichain.gen_sim.action_agent_pipeline.protocol.actions import (
    ATOMIC_ACTION_CLASSES as SUPPORTED_ATOMIC_ACTION_CLASSES,
    MAX_COORDINATED_PAYLOADS as _MAX_COORDINATED_PAYLOADS,
    OBJECT_ORIENTATION_AXES as SUPPORTED_OBJECT_ORIENTATION_AXES,
    OBJECT_ORIENTATION_GOALS as SUPPORTED_OBJECT_ORIENTATION_GOALS,
    POSE_REFERENCES as SUPPORTED_POSE_REFERENCES,
    SUPPORTED_CONTROLS,
)

__all__ = ["AtomicActionSpec", "normalize_atomic_action_spec"]

_COORDINATED_WORLD_Y_ANGLE_CFG_KEY = "max_grasp_separation_angle_to_world_y_degrees"
TARGET_SPEC_FIELDS = (
    "target_object",
    "target_pose",
    "target_qpos",
    "target_object_pose",
)
ACTION_SPEC_FIELDS = {
    "atomic_action_class",
    "robot_name",
    "control",
    "cfg",
    *TARGET_SPEC_FIELDS,
}
SUPPORTED_SURFACE_Z_POLICIES = {"preserve", "object_on_surface", "surface_release"}
SURFACE_Z_POLICY_FIELDS = {
    "z_policy",
    "support",
    "support_uid",
    "surface_clearance",
}
SUPPORTED_QPOS_SOURCES = {"initial", "gripper_state", "joint_delta"}
SUPPORTED_CFG_KEYS = {
    "sample_interval",
    "pre_grasp_distance",
    "lift_height",
    "max_approach_retract_z",
    "hand_interp_steps",
    "hold_steps",
    "object_motion_keyframes",
    "post_hold_steps",
    "obj_upright_direction",
    "rotate_upright",
    "upright_yaw_samples",
    "approach_alignment_max_angle",
    "cartesian_waypoint_count",
    _COORDINATED_WORLD_Y_ANGLE_CFG_KEY,
}


@dataclass(frozen=True)
class AtomicActionSpec:
    """JSON-serializable atomic action specification."""

    atomic_action_class: str
    robot_name: str
    control: str = "arm"
    target_object: dict[str, Any] = field(default_factory=dict)
    target_pose: dict[str, Any] = field(default_factory=dict)
    target_qpos: dict[str, Any] = field(default_factory=dict)
    target_object_pose: dict[str, Any] = field(default_factory=dict)
    cfg: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any]) -> "AtomicActionSpec":
        normalized = normalize_atomic_action_spec(spec)
        return cls.from_normalized(normalized)

    @classmethod
    def from_normalized(cls, normalized: Mapping[str, Any]) -> "AtomicActionSpec":
        """Build an atomic action spec from already-normalized data."""
        return cls(
            atomic_action_class=normalized["atomic_action_class"],
            robot_name=normalized["robot_name"],
            control=normalized["control"],
            target_object=dict(normalized.get("target_object", {})),
            target_pose=dict(normalized.get("target_pose", {})),
            target_qpos=dict(normalized.get("target_qpos", {})),
            target_object_pose=dict(normalized.get("target_object_pose", {})),
            cfg=dict(normalized["cfg"]),
        )

    def to_dict(self) -> dict[str, Any]:
        spec = {
            "atomic_action_class": self.atomic_action_class,
            "robot_name": self.robot_name,
            "control": self.control,
            "cfg": deepcopy(self.cfg),
        }
        if self.target_object:
            spec["target_object"] = deepcopy(self.target_object)
        if self.target_pose:
            spec["target_pose"] = deepcopy(self.target_pose)
        if self.target_qpos:
            spec["target_qpos"] = deepcopy(self.target_qpos)
        if self.target_object_pose:
            spec["target_object_pose"] = deepcopy(self.target_object_pose)
        return spec


def normalize_atomic_action_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an atomic action JSON spec."""
    if not isinstance(spec, Mapping):
        raise TypeError(f"Action spec must be a mapping, got {type(spec)}.")
    if "fn" in spec:
        raise ValueError(
            "Legacy fn/kwargs action schema is not supported. Use atomic action class "
            "JSON spec with atomic_action_class, robot_name, control, cfg, and "
            "exactly one of target_object, target_pose, or target_qpos."
        )

    if "action" in spec:
        raise ValueError(
            "Legacy action schema is not supported. Use atomic_action_class with "
            "CoordinatedPickment, PickUp, MoveEndEffector, MoveJoints, "
            "MoveHeldObject, or Place."
        )
    if "target" in spec:
        raise ValueError(
            "Legacy target.kind schema is not supported. Use exactly one of "
            "target_object, target_pose, target_qpos, or target_object_pose."
        )
    unknown_fields = set(spec) - ACTION_SPEC_FIELDS
    if unknown_fields:
        raise ValueError(
            f"Unsupported atomic action spec fields: "
            f"{', '.join(sorted(unknown_fields))}."
        )

    atomic_action_class = spec.get("atomic_action_class")
    if atomic_action_class not in SUPPORTED_ATOMIC_ACTION_CLASSES:
        raise ValueError(
            f"Unsupported atomic action class {atomic_action_class!r}; expected "
            f"one of {sorted(SUPPORTED_ATOMIC_ACTION_CLASSES)}."
        )

    robot_name = spec.get("robot_name")
    if not isinstance(robot_name, str) or not robot_name:
        raise ValueError("Atomic action spec requires non-empty robot_name.")

    control = spec.get("control", "arm")
    if control not in SUPPORTED_CONTROLS:
        raise ValueError(
            f"Unsupported atomic action control {control!r}; expected one of "
            f"{sorted(SUPPORTED_CONTROLS)}."
        )

    cfg = dict(spec.get("cfg") or {})
    unknown_cfg = set(cfg) - SUPPORTED_CFG_KEYS
    if unknown_cfg:
        raise ValueError(
            f"Unsupported atomic action cfg keys: {', '.join(sorted(unknown_cfg))}."
        )
    _validate_cfg_values(cfg)
    if (
        _COORDINATED_WORLD_Y_ANGLE_CFG_KEY in cfg
        and atomic_action_class != "CoordinatedPickment"
    ):
        raise ValueError(
            f"{_COORDINATED_WORLD_Y_ANGLE_CFG_KEY} is supported only for "
            "CoordinatedPickment."
        )

    target_values = _normalize_action_target(
        spec,
        atomic_action_class=atomic_action_class,
        control=control,
    )

    normalized = {
        "atomic_action_class": atomic_action_class,
        "robot_name": robot_name,
        "control": control,
        "cfg": cfg,
    }
    normalized.update(target_values)
    return normalized


def _normalize_action_target(
    spec: Mapping[str, Any],
    *,
    atomic_action_class: str,
    control: str,
) -> dict[str, dict[str, Any]]:
    target_fields = [field for field in TARGET_SPEC_FIELDS if field in spec]
    if atomic_action_class == "CoordinatedPickment":
        required_fields = {"target_object", "target_object_pose"}
        if set(target_fields) != required_fields:
            raise ValueError(
                "CoordinatedPickment requires target_object and " "target_object_pose."
            )
        if control != "arm":
            raise ValueError("CoordinatedPickment requires control='arm'.")
        target_object = spec["target_object"]
        target_object_pose = spec["target_object_pose"]
        if not isinstance(target_object, Mapping) or not target_object:
            raise ValueError("target_object must be a non-empty object.")
        if not isinstance(target_object_pose, Mapping) or not target_object_pose:
            raise ValueError("target_object_pose must be a non-empty object.")
        target_object = dict(target_object)
        target_object_pose = dict(target_object_pose)
        _validate_target_object(target_object)
        _validate_target_object_pose(target_object_pose)
        return {
            "target_object": target_object,
            "target_object_pose": target_object_pose,
        }

    if len(target_fields) != 1:
        raise ValueError(
            "Atomic action spec requires exactly one of target_object, target_pose, "
            f"target_qpos, or target_object_pose; got {target_fields}."
        )

    target_field = target_fields[0]
    target_spec = spec[target_field]
    if not isinstance(target_spec, Mapping) or not target_spec:
        raise ValueError(f"{target_field} must be a non-empty object.")
    target_spec = dict(target_spec)

    if atomic_action_class == "PickUp":
        if control != "arm" or target_field != "target_object":
            raise ValueError("PickUp requires control='arm' and target_object.")
        _validate_target_object(target_spec)
        return {target_field: target_spec}

    if atomic_action_class == "Place":
        if control != "arm" or target_field not in {
            "target_pose",
            "target_object_pose",
        }:
            raise ValueError(
                "Place requires control='arm' and target_pose or target_object_pose."
            )
        if target_field == "target_pose":
            _validate_target_pose(target_spec)
        else:
            _validate_target_object_pose(target_spec)
            if target_spec.get("orientation_goal", "preserve") != "preserve":
                raise ValueError(
                    "Place target_object_pose only supports orientation_goal='preserve'; "
                    "use MoveHeldObject for explicit in-air rotation."
                )
        return {target_field: target_spec}

    if atomic_action_class == "MoveEndEffector":
        if control != "arm":
            raise ValueError("MoveEndEffector requires control='arm'.")
        if target_field != "target_pose":
            raise ValueError("MoveEndEffector requires target_pose.")
        _validate_target_pose(target_spec)
        return {target_field: target_spec}

    if atomic_action_class == "MoveJoints":
        if target_field != "target_qpos":
            raise ValueError("MoveJoints requires target_qpos.")
        _validate_target_qpos(target_spec, control=control)
        return {target_field: target_spec}

    if atomic_action_class == "MoveHeldObject":
        if control != "arm" or target_field != "target_object_pose":
            raise ValueError(
                "MoveHeldObject requires control='arm' and target_object_pose."
            )
        _validate_target_object_pose(target_spec)
        return {target_field: target_spec}

    raise ValueError(f"Unsupported atomic action class: {atomic_action_class}.")


def _validate_target_object(target_object: Mapping[str, Any]) -> None:
    unknown_fields = set(target_object) - {"obj_name", "affordance", "payloads"}
    if unknown_fields:
        raise ValueError(
            f"Unsupported target_object fields: {', '.join(sorted(unknown_fields))}."
        )
    obj_name = target_object.get("obj_name")
    if not isinstance(obj_name, str) or not obj_name:
        raise ValueError("target_object requires non-empty obj_name.")
    affordance = target_object.get("affordance", "antipodal")
    if affordance != "antipodal":
        raise ValueError("target_object only supports affordance='antipodal'.")
    payloads = target_object.get("payloads", [])
    if not isinstance(payloads, list) or len(payloads) > _MAX_COORDINATED_PAYLOADS:
        raise ValueError(
            "target_object payloads must be a list with at most "
            f"{_MAX_COORDINATED_PAYLOADS} UIDs."
        )
    if any(not isinstance(payload, str) or not payload for payload in payloads):
        raise ValueError("target_object payloads must contain non-empty UID strings.")
    if len(payloads) != len(set(payloads)):
        raise ValueError("target_object payloads must not contain duplicate UIDs.")
    if obj_name in payloads:
        raise ValueError("target_object payloads must not include the shared object.")


def _validate_target_pose(target_pose: Mapping[str, Any]) -> None:
    reference = target_pose.get("reference")
    if reference not in SUPPORTED_POSE_REFERENCES:
        raise ValueError(
            f"target_pose reference must be one of {sorted(SUPPORTED_POSE_REFERENCES)}."
        )

    if reference == "object":
        _validate_target_fields(
            target_pose,
            {"reference", "obj_name", "offset"},
            "target_pose",
        )
        obj_name = target_pose.get("obj_name")
        if not isinstance(obj_name, str) or not obj_name:
            raise ValueError("object target_pose requires non-empty obj_name.")
        _xyz(target_pose.get("offset", [0.0, 0.0, 0.0]), "offset")
        return

    if reference == "absolute":
        _validate_target_fields(
            target_pose,
            {
                "reference",
                "position",
                "position_by_env",
                "rotation_matrix_by_env",
            },
            "target_pose",
        )
        _validate_absolute_position(target_pose, "target_pose")
        rotation_matrices = target_pose.get("rotation_matrix_by_env")
        if rotation_matrices is not None and (
            not isinstance(rotation_matrices, list)
            or not rotation_matrices
            or any(
                not isinstance(matrix, list)
                or len(matrix) != 3
                or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
                for matrix in rotation_matrices
            )
        ):
            raise ValueError(
                "absolute target_pose rotation_matrix_by_env requires an Nx3x3 list."
            )
        return

    _validate_target_fields(
        target_pose,
        {"reference", "offset", "frame"},
        "target_pose",
    )
    _xyz(target_pose.get("offset", [0.0, 0.0, 0.0]), "offset")
    frame = target_pose.get("frame", "world")
    if frame not in {"world", "eef"}:
        raise ValueError("relative target_pose frame must be 'world' or 'eef'.")


def _validate_absolute_position(
    target_pose: Mapping[str, Any],
    target_name: str,
) -> None:
    position = target_pose.get("position")
    position_by_env = target_pose.get("position_by_env")
    if (position is None) == (position_by_env is None):
        raise ValueError(
            f"absolute {target_name} requires exactly one of position or "
            "position_by_env."
        )
    if position is not None:
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"absolute {target_name} position requires three entries.")
        return
    if position_by_env is not None and (
        not isinstance(position_by_env, list)
        or not position_by_env
        or any(not isinstance(item, list) or len(item) != 3 for item in position_by_env)
    ):
        raise ValueError(
            f"absolute {target_name} position_by_env requires a non-empty Nx3 list."
        )


def _validate_target_object_pose(target_object_pose: Mapping[str, Any]) -> None:
    _validate_target_pose_like(target_object_pose, "target_object_pose")
    orientation_goal = target_object_pose.get("orientation_goal", "preserve")
    if orientation_goal not in SUPPORTED_OBJECT_ORIENTATION_GOALS:
        raise ValueError(
            "target_object_pose orientation_goal must be one of "
            f"{sorted(SUPPORTED_OBJECT_ORIENTATION_GOALS)}."
        )
    orientation_axis = target_object_pose.get("orientation_axis", "none")
    if orientation_axis not in SUPPORTED_OBJECT_ORIENTATION_AXES:
        raise ValueError(
            "target_object_pose orientation_axis must be one of "
            f"{sorted(SUPPORTED_OBJECT_ORIENTATION_AXES)}."
        )
    align_to = target_object_pose.get("align_to")
    if align_to is not None and (not isinstance(align_to, str) or not align_to):
        raise ValueError("target_object_pose align_to must be a non-empty string.")
    if orientation_goal == "axis_align":
        if align_to is None:
            if orientation_axis not in {"x", "y"}:
                raise ValueError(
                    "axis_align without align_to requires orientation_axis 'x' or 'y'."
                )
        elif orientation_axis not in {"long_axis", "short_axis"}:
            raise ValueError(
                "axis_align with align_to requires orientation_axis 'long_axis' "
                "or 'short_axis'."
            )
    elif orientation_axis != "none" or align_to is not None:
        raise ValueError(
            "preserve, upright, and lay_flat require orientation_axis='none' "
            "and no align_to."
        )


def _validate_target_pose_like(
    target_pose: Mapping[str, Any],
    target_name: str,
) -> None:
    reference = target_pose.get("reference")
    allowed_common = {
        "orientation_goal",
        "orientation_axis",
        "align_to",
    } | SURFACE_Z_POLICY_FIELDS
    if reference not in SUPPORTED_POSE_REFERENCES:
        raise ValueError(
            f"{target_name} reference must be one of {sorted(SUPPORTED_POSE_REFERENCES)}."
        )
    _validate_surface_z_policy_fields(target_pose, target_name)

    if reference == "object":
        _validate_target_fields(
            target_pose,
            {"reference", "obj_name", "offset"} | allowed_common,
            target_name,
        )
        obj_name = target_pose.get("obj_name")
        if not isinstance(obj_name, str) or not obj_name:
            raise ValueError(f"object {target_name} requires non-empty obj_name.")
        _xyz(target_pose.get("offset", [0.0, 0.0, 0.0]), "offset")
        return

    if reference == "absolute":
        _validate_target_fields(
            target_pose,
            {"reference", "position", "position_by_env"} | allowed_common,
            target_name,
        )
        _validate_absolute_position(target_pose, target_name)
        return

    _validate_target_fields(
        target_pose,
        {"reference", "offset", "frame"} | allowed_common,
        target_name,
    )
    _xyz(target_pose.get("offset", [0.0, 0.0, 0.0]), "offset")
    frame = target_pose.get("frame", "world")
    if frame not in {"world", "eef"}:
        raise ValueError(f"relative {target_name} frame must be 'world' or 'eef'.")


def _validate_surface_z_policy_fields(
    target_pose: Mapping[str, Any],
    target_name: str,
) -> None:
    policy = target_pose.get("z_policy", "preserve")
    if policy not in SUPPORTED_SURFACE_Z_POLICIES:
        raise ValueError(
            f"{target_name} z_policy must be one of "
            f"{sorted(SUPPORTED_SURFACE_Z_POLICIES)}."
        )
    for field_name in ("support", "support_uid"):
        support_value = target_pose.get(field_name)
        if support_value is not None and (
            not isinstance(support_value, str) or not support_value
        ):
            raise ValueError(f"{target_name} {field_name} must be a non-empty string.")
    support = target_pose.get("support")
    support_uid = target_pose.get("support_uid")
    if support is not None and support_uid is not None and support != support_uid:
        raise ValueError(
            f"{target_name} support and support_uid must refer to the same object."
        )
    clearance = target_pose.get("surface_clearance")
    if clearance is not None:
        if isinstance(clearance, bool) or not isinstance(clearance, (int, float)):
            raise ValueError(f"{target_name} surface_clearance must be a number.")
        if not np.isfinite(float(clearance)) or float(clearance) < 0.0:
            raise ValueError(
                f"{target_name} surface_clearance must be a finite non-negative number."
            )
    if policy == "preserve":
        return
    _surface_support_uid(target_pose, target_name=target_name, require=True)


def _validate_target_qpos(
    target_qpos: Mapping[str, Any],
    *,
    control: str,
) -> None:
    source = target_qpos.get("source")
    if source not in SUPPORTED_QPOS_SOURCES:
        raise ValueError(
            f"target_qpos source must be one of {sorted(SUPPORTED_QPOS_SOURCES)}."
        )

    if source == "initial":
        _validate_target_fields(target_qpos, {"source"}, "target_qpos")
        if control != "arm":
            raise ValueError("initial target_qpos requires control='arm'.")
        return

    if source == "gripper_state":
        _validate_target_fields(target_qpos, {"source", "state"}, "target_qpos")
        if control != "hand":
            raise ValueError("gripper_state target_qpos requires control='hand'.")
        state = target_qpos.get("state")
        if state not in {"open", "close"}:
            raise ValueError(
                "gripper_state target_qpos state must be 'open' or 'close'."
            )
        return

    _validate_target_fields(
        target_qpos,
        {"source", "joint_index", "delta_degrees"},
        "target_qpos",
    )
    if control != "arm":
        raise ValueError("joint_delta target_qpos requires control='arm'.")
    if "joint_index" not in target_qpos:
        raise ValueError("joint_delta target_qpos requires joint_index.")
    int(target_qpos["joint_index"])
    float(target_qpos.get("delta_degrees", 0.0))


def _validate_target_fields(
    target_spec: Mapping[str, Any],
    allowed_fields: set[str],
    target_name: str,
) -> None:
    unknown_fields = set(target_spec) - allowed_fields
    if unknown_fields:
        raise ValueError(
            f"Unsupported {target_name} fields: {', '.join(sorted(unknown_fields))}."
        )


def _validate_cfg_values(cfg: Mapping[str, Any]) -> None:
    if _COORDINATED_WORLD_Y_ANGLE_CFG_KEY in cfg:
        value = cfg[_COORDINATED_WORLD_Y_ANGLE_CFG_KEY]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not np.isfinite(value)
            or not 0.0 <= float(value) <= 90.0
        ):
            raise ValueError(
                f"{_COORDINATED_WORLD_Y_ANGLE_CFG_KEY} must be a finite number "
                "in [0, 90] degrees or null."
            )
    if "max_approach_retract_z" in cfg:
        value = cfg["max_approach_retract_z"]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not np.isfinite(value)
        ):
            raise ValueError("max_approach_retract_z must be a finite number.")
    if "obj_upright_direction" in cfg:
        _xyz(cfg["obj_upright_direction"], "obj_upright_direction")
    if "rotate_upright" in cfg:
        value = cfg["rotate_upright"]
        if value is not None and not isinstance(value, int | float):
            raise ValueError("rotate_upright must be a numeric value in radians.")
    if "upright_yaw_samples" in cfg:
        value = cfg["upright_yaw_samples"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("upright_yaw_samples must be an integer >= 1.")
    if "approach_alignment_max_angle" in cfg:
        value = cfg["approach_alignment_max_angle"]
        if value is not None and (
            not isinstance(value, int | float) or not 0.0 <= float(value) <= np.pi / 2
        ):
            raise ValueError(
                "approach_alignment_max_angle must be a numeric value in "
                "[0, pi / 2] radians or null."
            )
    if "cartesian_waypoint_count" in cfg:
        value = cfg["cartesian_waypoint_count"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("cartesian_waypoint_count must be an integer >= 1.")


def _surface_support_uid(
    target_pose_spec: Mapping[str, Any],
    *,
    target_name: str,
    require: bool,
) -> str | None:
    support = target_pose_spec.get("support")
    support_uid = target_pose_spec.get("support_uid")
    if support is not None and support_uid is not None and support != support_uid:
        raise ValueError(
            f"{target_name} support and support_uid must refer to the same object."
        )
    resolved = support if support is not None else support_uid
    if resolved is None and target_pose_spec.get("reference") == "object":
        resolved = target_pose_spec.get("obj_name")
    if require and (not isinstance(resolved, str) or not resolved):
        raise ValueError(f"{target_name} z_policy requires a support object uid.")
    return str(resolved) if resolved is not None else None


def _xyz(value, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must be a three-element list.")
    return [float(item) for item in value]
