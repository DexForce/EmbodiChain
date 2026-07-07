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

import hashlib
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from tqdm import tqdm

from embodichain.gen_sim.action_agent_pipeline.runtime.atom_action_utils import (
    get_arm_states,
    resolve_arm_side,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.coacd_cache_bridge import (
    GraspCollisionCachePreparationError,
    ensure_grasp_collision_cache_from_env_coacd,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    CoordinatedPickment,
    CoordinatedPickmentCfg,
    CoordinatedPickmentTarget,
    EndEffectorPoseTarget,
    GraspTarget,
    HeldObjectPoseTarget,
    HeldObjectState,
    JointPositionTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
    MoveHeldObject,
    MoveHeldObjectCfg,
    MoveJoints,
    MoveJointsCfg,
    ObjectSemantics,
    PickUp,
    PickUpCfg,
    Place,
    PlaceCfg,
    WorldState,
)
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
    GripperCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GRASP_ANNOTATOR_CACHE_DIR,
)
from embodichain.utils.logger import log_info, log_warning
from embodichain.utils.math import get_offset_pose, pose_inv

__all__ = [
    "AtomicActionSpec",
    "build_parallel_action_stream",
    "execute_atomic_action",
    "execute_parallel_atomic_actions",
    "init_parallel_world_states",
    "normalize_atomic_action_spec",
    "step_env_with_actions",
]


SUPPORTED_ATOMIC_ACTION_CLASSES = {
    "CoordinatedPickment",
    "PickUp",
    "MoveEndEffector",
    "MoveJoints",
    "MoveHeldObject",
    "Place",
}
SUPPORTED_CONTROLS = {"arm", "hand"}
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
SUPPORTED_POSE_REFERENCES = {"object", "absolute", "relative"}
SUPPORTED_OBJECT_ORIENTATION_GOALS = {"preserve", "upright", "lay_flat", "axis_align"}
SUPPORTED_OBJECT_ORIENTATION_AXES = {"none", "x", "y", "long_axis", "short_axis"}
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
    "hand_interp_steps",
    "hold_steps",
    "object_motion_keyframes",
    "post_hold_steps",
    "obj_upright_direction",
    "rotate_upright",
}


ATOMIC_ACTION_REGISTRY = {
    "CoordinatedPickment": (CoordinatedPickment, CoordinatedPickmentCfg),
    "PickUp": (PickUp, PickUpCfg),
    "MoveEndEffector": (MoveEndEffector, MoveEndEffectorCfg),
    "MoveJoints": (MoveJoints, MoveJointsCfg),
    "MoveHeldObject": (MoveHeldObject, MoveHeldObjectCfg),
    "Place": (Place, PlaceCfg),
}


_DEFAULT_SURFACE_RELEASE_CLEARANCE = 0.015
_DEFAULT_PICKUP_LIFT_HEIGHT = 0.16


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


@dataclass(frozen=True)
class _ExecutedAtomicAction:
    action: np.ndarray
    next_state: WorldState | None
    robot_name: str | None
    control: str | None


@dataclass(frozen=True)
class _CoordinatedGraspPair:
    left_object_to_eef: torch.Tensor
    right_object_to_eef: torch.Tensor
    priority: int
    score: float


@dataclass(frozen=True)
class _GraspRuntimeDefaults:
    antipodal_n_sample: int = 10000
    antipodal_max_angle: float = float(np.pi / 12)
    max_open_length: float = 0.14
    min_open_length: float = 0.003
    finger_length: float = 0.13
    point_sample_dense: float = 0.012
    max_deviation_angle: float = float(np.pi / 6)
    viser_port: int = 11801


_GRASP_RUNTIME_DEFAULTS = _GraspRuntimeDefaults()
_BOTTLE_LIKE_KEYWORDS = (
    "bottle",
    "can",
    "jar",
    "tin",
    "soda",
    "cola",
    "罐头",
    "易拉罐",
    "瓶",
    "瓶子",
)
_SHORT_BOTTLE_LIKE_KEYWORDS = {"can", "jar", "tin"}
_COORDINATED_CONTAINER_LIKE_KEYWORDS = (
    "pot",
    "pan",
    "wok",
    "skillet",
    "saucepan",
    "tray",
    "plate",
    "bowl",
    "basket",
    "container",
    "dish",
    "basin",
    "cup",
    "mug",
    "锅",
    "平底锅",
    "炒锅",
    "托盘",
    "盘",
    "盘子",
    "碗",
    "篮",
    "篮子",
    "容器",
    "盆",
    "杯",
)
_COORDINATED_ROD_LIKE_KEYWORDS = (
    "umbrella",
    "rod",
    "bar",
    "stick",
    "tube",
    "cylinder",
    "cylindrical",
    "pole",
    "baton",
    "rectangular",
    "cuboid",
    "雨伞",
    "伞",
    "杆",
    "棒",
    "棍",
    "柱",
    "圆柱",
    "长方体",
    "矩形",
    "木条",
)
_COORDINATED_GRASP_STYLE_CONTAINER = "container_like"
_COORDINATED_GRASP_STYLE_ROD = "rod_like"
_COORDINATED_GRASP_STYLE_GENERIC = "generic"
_COORDINATED_ROD_LIKE_INSET_FRACTIONS = (0.35, 0.30, 0.25, 0.20)
_COORDINATED_CONTAINER_LIKE_INSET_FRACTIONS = (0.08, 0.12, 0.20, 0.28)
_COORDINATED_GENERIC_INSET_FRACTIONS = (0.28, 0.35, 0.20)


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
        if control != "arm" or target_field != "target_pose":
            raise ValueError("Place requires control='arm' and target_pose.")
        _validate_target_pose(target_spec)
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
    unknown_fields = set(target_object) - {"obj_name", "affordance"}
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
            {"reference", "position"},
            "target_pose",
        )
        position = target_pose.get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(
                "absolute target_pose requires position with three entries."
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
            {"reference", "position"} | allowed_common,
            target_name,
        )
        position = target_pose.get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(
                f"absolute {target_name} requires position with three entries."
            )
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


def execute_atomic_action(
    action_spec: Mapping[str, Any] | AtomicActionSpec,
    *,
    env: Any,
    state: WorldState | None = None,
    **runtime_kwargs: Any,
) -> np.ndarray:
    """Execute one atomic action spec and return local arm+eef qpos actions."""
    executed = _execute_atomic_action_result(
        action_spec,
        env=env,
        state=state,
        **runtime_kwargs,
    )
    if executed.control == "coordinated":
        _sync_agent_states_from_coordinated_action(env, executed.action)
    else:
        _sync_agent_state_from_atomic_action(
            env,
            executed.robot_name,
            executed.action,
            executed.control,
        )
    return executed.action


def _execute_atomic_action_result(
    action_spec: Mapping[str, Any] | AtomicActionSpec,
    *,
    env,
    state: WorldState | None = None,
    **runtime_kwargs,
) -> _ExecutedAtomicAction:
    """Execute one atomic action spec and keep the typed WorldState result."""
    spec = (
        action_spec
        if isinstance(action_spec, AtomicActionSpec)
        else AtomicActionSpec.from_mapping(action_spec)
    )

    target = _resolve_target(env, spec, runtime_kwargs, state=state)
    _, arm_part, hand_part, arm_joints, eef_joints = _select_arm_parts(
        env, spec.robot_name
    )
    cfg = _build_action_cfg(env, spec, arm_part, hand_part, len(eef_joints))
    target = _build_typed_target(spec, target)
    if state is None:
        state = WorldState(last_qpos=env.robot.get_qpos().clone())
    state = _state_with_current_agent_qpos(env, spec, state)
    action_cls = _get_atomic_action_class(spec.atomic_action_class)
    action = action_cls(
        motion_generator=_motion_generator_for_env(env, runtime_kwargs),
        cfg=cfg,
    )
    result = action.execute(
        target=target,
        state=state,
    )
    if not result.success:
        raise RuntimeError(
            f"Atomic action failed: atomic_action_class={spec.atomic_action_class}, "
            f"robot_name={spec.robot_name}, target={_target_summary(spec)}."
        )
    if spec.atomic_action_class == "CoordinatedPickment":
        return _executed_coordinated_atomic_action(env, spec, result)
    if spec.atomic_action_class == "MoveJoints":
        joint_ids = arm_joints if spec.control == "arm" else eef_joints
    else:
        joint_ids = arm_joints + eef_joints
    trajectory = result.trajectory[:, :, joint_ids]

    action_np = _trajectory_to_agent_action(
        env,
        spec.robot_name,
        trajectory,
        joint_ids,
    )
    action_np = _append_hold_steps(
        action_np,
        int(spec.cfg.get("post_hold_steps", 0)),
        "atomic action",
    )
    log_info(
        "Using atomic action: "
        f"atomic_action_class={spec.atomic_action_class}, cfg={cfg.__class__.__name__}, "
        f"control={spec.control}, target={_target_summary(spec)}, "
        f"steps={len(action_np)}.",
        color="green",
    )
    next_state = result.next_state
    if int(spec.cfg.get("post_hold_steps", 0)) > 0:
        next_state = WorldState(
            last_qpos=next_state.last_qpos.clone(),
            held_object=next_state.held_object,
            coordinated_held_object=next_state.coordinated_held_object,
        )
    return _ExecutedAtomicAction(
        action=action_np,
        next_state=next_state,
        robot_name=spec.robot_name,
        control=spec.control,
    )


def execute_parallel_atomic_actions(
    left_arm_action: Mapping[str, Any] | AtomicActionSpec | np.ndarray | None = None,
    right_arm_action: Mapping[str, Any] | AtomicActionSpec | np.ndarray | None = None,
    *,
    env: Any,
    world_states: dict[str, WorldState] | None = None,
    return_result: bool = False,
    **runtime_kwargs: Any,
) -> list[torch.Tensor] | dict[str, Any]:
    """Execute left/right atomic action specs as one synchronized stream."""
    result = build_parallel_action_stream(
        left_arm_action=left_arm_action,
        right_arm_action=right_arm_action,
        env=env,
        world_states=world_states,
        return_result=True,
        **runtime_kwargs,
    )
    actions = result["actions"]
    step_env_with_actions(env, actions)
    _sync_agent_states_from_parallel_actions(env, result["arm_actions"])
    if return_result:
        return result
    return actions


def build_parallel_action_stream(
    left_arm_action: Mapping[str, Any] | AtomicActionSpec | np.ndarray | None = None,
    right_arm_action: Mapping[str, Any] | AtomicActionSpec | np.ndarray | None = None,
    *,
    env: Any,
    world_states: dict[str, WorldState] | None = None,
    return_result: bool = False,
    **runtime_kwargs: Any,
) -> list[torch.Tensor] | dict[str, Any]:
    """Build a synchronized left/right atomic action stream without stepping env."""
    if env is None:
        raise ValueError("env is required to build parallel atomic actions.")
    if world_states is None:
        world_states = init_parallel_world_states(env)
    raw_left_arm_action = left_arm_action
    raw_right_arm_action = right_arm_action
    coordinated_action = _pop_coordinated_edge_action(left_arm_action, right_arm_action)
    if coordinated_action is not None:
        executed = _resolve_action_spec(
            coordinated_action,
            env,
            runtime_kwargs,
            state=world_states.get("coordinated"),
        )
        if not isinstance(executed, _ExecutedAtomicAction):
            raise TypeError("Coordinated action must resolve to an atomic action.")
        action_np = _as_2d_action(
            _executed_action_array(executed),
            "coordinated_action",
        )
        actions = _full_robot_action_array_to_steps(action_np)
        result = {
            "actions": actions,
            "world_states": {
                **world_states,
                "coordinated": executed.next_state,
                "left": executed.next_state,
                "right": executed.next_state,
            },
            "arm_actions": {
                "left": executed,
                "right": None,
            },
        }
        if return_result:
            return result
        return actions

    _validate_arm_action_slot(env, "left", left_arm_action)
    _validate_arm_action_slot(env, "right", right_arm_action)
    left_arm_action = _resolve_action_spec(
        left_arm_action,
        env,
        runtime_kwargs,
        state=world_states.get("left"),
    )
    right_arm_action = _resolve_action_spec(
        right_arm_action,
        env,
        runtime_kwargs,
        state=world_states.get("right"),
    )
    _validate_arm_action_slot(env, "left", left_arm_action)
    _validate_arm_action_slot(env, "right", right_arm_action)

    left_action_np = _as_2d_action(
        _executed_action_array(left_arm_action),
        "left_arm_action",
    )
    right_action_np = _as_2d_action(
        _executed_action_array(right_arm_action),
        "right_arm_action",
    )
    arm_actions = {"left": left_action_np, "right": right_action_np}

    if all(action is None for action in arm_actions.values()):
        raise ValueError("At least one atomic arm action must be provided.")

    action_len = max(
        len(action) for action in arm_actions.values() if action is not None
    )
    for side, action in arm_actions.items():
        if action is not None and len(action) < action_len:
            diff = action_len - len(action)
            padding = np.repeat(action[-1:], diff, axis=0)
            arm_actions[side] = np.concatenate([action, padding], axis=0)

    current_qpos = (
        env.robot.get_qpos().squeeze(0).detach().cpu().numpy().astype(np.float32)
    )
    actions = np.repeat(current_qpos[None, :], action_len, axis=0)

    for side, action in arm_actions.items():
        if action is None:
            continue

        arm_index = list(getattr(env, f"{side}_arm_joints", [])) + list(
            getattr(env, f"{side}_eef_joints", [])
        )
        if not arm_index:
            raise ValueError(
                f"{side}_arm_action was provided, but {side}_arm is not configured "
                f"on robot control parts {getattr(env.robot, 'control_parts', None)}."
            )
        if action.shape[-1] != len(arm_index):
            raise ValueError(
                f"{side}_arm_action width {action.shape[-1]} does not match "
                f"{side}_arm joints plus eef joints ({len(arm_index)})."
            )
        actions[:, arm_index] = action

    actions = torch.from_numpy(actions).to(dtype=torch.float32).unsqueeze(1)
    actions = list(actions.unbind(dim=0))
    if not return_result:
        return actions
    next_world_states = dict(world_states)
    for side, executed in {
        "left": left_arm_action,
        "right": right_arm_action,
    }.items():
        if (
            isinstance(executed, _ExecutedAtomicAction)
            and executed.next_state is not None
        ):
            next_world_states[side] = executed.next_state
    if _is_dual_coordinated_release_edge(
        raw_left_arm_action,
        raw_right_arm_action,
    ) and _has_coordinated_held_object(world_states):
        release_state = _released_coordinated_world_state(
            actions[-1],
            next_world_states,
        )
        next_world_states["coordinated"] = release_state
        next_world_states["left"] = release_state
        next_world_states["right"] = release_state
    return {
        "actions": actions,
        "world_states": next_world_states,
        "arm_actions": {
            "left": left_arm_action,
            "right": right_arm_action,
        },
    }


def init_parallel_world_states(env: Any) -> dict[str, WorldState]:
    """Seed independent per-arm WorldState slots from the current robot qpos."""
    qpos = env.robot.get_qpos().clone()
    return {
        "coordinated": WorldState(last_qpos=qpos.clone()),
        "left": WorldState(last_qpos=qpos.clone()),
        "right": WorldState(last_qpos=qpos.clone()),
    }


def step_env_with_actions(
    env: Any,
    actions: list[torch.Tensor],
    *,
    update_obj_info: bool = True,
) -> None:
    """Step an environment through a prebuilt action stream."""
    if env is None:
        raise ValueError("env is required to step action stream.")
    for action in tqdm(actions):
        env.step(action)
        if update_obj_info:
            env.update_obj_info()


def _resolve_action_spec(
    action_spec,
    env,
    runtime_kwargs: dict[str, Any],
    *,
    state: WorldState | None,
):
    if action_spec is None:
        return None
    if isinstance(action_spec, np.ndarray):
        return action_spec
    if isinstance(action_spec, torch.Tensor):
        return action_spec
    return _execute_atomic_action_result(
        action_spec,
        env=env,
        state=state,
        **runtime_kwargs,
    )


def _executed_action_array(action):
    if isinstance(action, _ExecutedAtomicAction):
        return action.action
    return action


def _validate_arm_action_slot(env, side: str, action) -> None:
    robot_name = _arm_action_robot_name(action)
    if _arm_action_control(action) == "coordinated" or robot_name is None:
        return
    action_side = resolve_arm_side(env, robot_name)
    if action_side != side:
        raise ValueError(
            f"{side}_arm_action contains robot_name={robot_name!r}, "
            f"which resolves to {action_side}_arm. Keep the outer graph slot "
            "consistent with the semantic arm name."
        )


def _arm_action_robot_name(action) -> str | None:
    if isinstance(action, _ExecutedAtomicAction):
        return action.robot_name
    if isinstance(action, AtomicActionSpec):
        return action.robot_name
    if isinstance(action, Mapping):
        value = action.get("robot_name")
        if value is not None:
            return str(value)
    return None


def _arm_action_control(action) -> str | None:
    if isinstance(action, _ExecutedAtomicAction):
        return action.control
    if isinstance(action, AtomicActionSpec):
        return action.control
    if isinstance(action, Mapping):
        value = action.get("control")
        if value is not None:
            return str(value)
    return None


def _pop_coordinated_edge_action(left_arm_action, right_arm_action):
    left_is_coordinated = _is_coordinated_action(left_arm_action)
    right_is_coordinated = _is_coordinated_action(right_arm_action)
    if left_is_coordinated and right_is_coordinated:
        raise ValueError(
            "A graph edge may contain only one CoordinatedPickment action."
        )
    if left_is_coordinated:
        if right_arm_action is not None:
            raise ValueError(
                "CoordinatedPickment controls both arms; right_arm_action must be null."
            )
        return left_arm_action
    if right_is_coordinated:
        if left_arm_action is not None:
            raise ValueError(
                "CoordinatedPickment controls both arms; left_arm_action must be null."
            )
        return right_arm_action
    return None


def _is_coordinated_action(action_spec) -> bool:
    if isinstance(action_spec, AtomicActionSpec):
        return action_spec.atomic_action_class == "CoordinatedPickment"
    if isinstance(action_spec, Mapping):
        return action_spec.get("atomic_action_class") == "CoordinatedPickment"
    return False


def _is_dual_coordinated_release_edge(left_arm_action, right_arm_action) -> bool:
    return {
        _gripper_open_release_side(left_arm_action),
        _gripper_open_release_side(right_arm_action),
    } == {"left", "right"}


def _gripper_open_release_side(action_spec) -> str | None:
    if isinstance(action_spec, AtomicActionSpec):
        atomic_action_class = action_spec.atomic_action_class
        robot_name = action_spec.robot_name
        control = action_spec.control
        target_qpos = action_spec.target_qpos
    elif isinstance(action_spec, Mapping):
        atomic_action_class = action_spec.get("atomic_action_class")
        robot_name = action_spec.get("robot_name")
        control = action_spec.get("control")
        target_qpos = action_spec.get("target_qpos") or {}
    else:
        return None

    if (
        atomic_action_class != "MoveJoints"
        or control != "hand"
        or not isinstance(target_qpos, Mapping)
        or target_qpos.get("source") != "gripper_state"
        or target_qpos.get("state") != "open"
    ):
        return None
    if robot_name == "left_arm":
        return "left"
    if robot_name == "right_arm":
        return "right"
    return None


def _has_coordinated_held_object(world_states: Mapping[str, WorldState]) -> bool:
    return any(
        state.coordinated_held_object is not None
        for state in world_states.values()
        if isinstance(state, WorldState)
    )


def _released_coordinated_world_state(
    final_action: torch.Tensor,
    world_states: Mapping[str, WorldState],
) -> WorldState:
    held_object = None
    for state in world_states.values():
        if isinstance(state, WorldState) and state.held_object is not None:
            held_object = state.held_object
            break
    final_qpos = torch.as_tensor(final_action, dtype=torch.float32)
    if final_qpos.dim() == 1:
        final_qpos = final_qpos.unsqueeze(0)
    return WorldState(
        last_qpos=final_qpos.clone(),
        held_object=held_object,
        coordinated_held_object=None,
    )


def _executed_coordinated_atomic_action(
    env,
    spec: AtomicActionSpec,
    result,
) -> _ExecutedAtomicAction:
    trajectory = result.trajectory
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach()
    else:
        trajectory = torch.as_tensor(trajectory)
    if trajectory.dim() == 3:
        trajectory = trajectory[0]
    if trajectory.dim() != 2 or trajectory.shape[0] == 0:
        raise ValueError(
            "Coordinated atomic action trajectory must have shape (T, D) or "
            f"(N, T, D), got {trajectory.shape}."
        )
    action_np = trajectory.detach().cpu().numpy().astype(np.float32)
    action_np = _append_hold_steps(
        action_np,
        int(spec.cfg.get("post_hold_steps", 0)),
        "coordinated atomic action",
    )
    log_info(
        "Using coordinated atomic action: "
        f"atomic_action_class={spec.atomic_action_class}, "
        f"target={_target_summary(spec)}, steps={len(action_np)}.",
        color="green",
    )
    return _ExecutedAtomicAction(
        action=action_np,
        next_state=result.next_state,
        robot_name=None,
        control="coordinated",
    )


def _full_robot_action_array_to_steps(action_np: np.ndarray) -> list[torch.Tensor]:
    action_np = np.asarray(action_np, dtype=np.float32)
    if action_np.ndim != 2 or action_np.shape[0] == 0:
        raise ValueError(
            "Coordinated action stream must have shape (T, robot_dof), "
            f"got {action_np.shape}."
        )
    actions = torch.from_numpy(action_np).to(dtype=torch.float32).unsqueeze(1)
    return list(actions.unbind(dim=0))


def _sync_agent_states_from_parallel_actions(
    env,
    arm_actions: Mapping[str, Any],
) -> None:
    for executed in arm_actions.values():
        if not isinstance(executed, _ExecutedAtomicAction):
            continue
        if executed.control == "coordinated":
            _sync_agent_states_from_coordinated_action(env, executed.action)
            continue
        _sync_agent_state_from_atomic_action(
            env,
            executed.robot_name,
            executed.action,
            executed.control,
        )


def _select_arm_parts(env, robot_name: str):
    is_left = resolve_arm_side(env, robot_name) == "left"
    if hasattr(env, "get_agent_arm_control_part"):
        arm_part = env.get_agent_arm_control_part(is_left)
        hand_part = env.get_agent_eef_control_part(is_left)
    else:
        arm_part = "left_arm" if is_left else "right_arm"
        hand_part = "left_eef" if is_left else "right_eef"
    arm_joints = env.left_arm_joints if is_left else env.right_arm_joints
    eef_joints = env.left_eef_joints if is_left else env.right_eef_joints
    return is_left, arm_part, hand_part, list(arm_joints), list(eef_joints)


def _agent_parts_for_side(env, *, is_left: bool) -> tuple[str, str]:
    if hasattr(env, "get_agent_arm_control_part"):
        arm_part = env.get_agent_arm_control_part(is_left)
        hand_part = env.get_agent_eef_control_part(is_left)
    else:
        arm_part = "left_arm" if is_left else "right_arm"
        hand_part = "left_eef" if is_left else "right_eef"
    if not arm_part or not hand_part:
        side = "left" if is_left else "right"
        raise ValueError(f"CoordinatedPickment requires {side} arm and hand parts.")
    return str(arm_part), str(hand_part)


def _joint_ids_for_control_part(env, control_part: str | None) -> list[int]:
    if not control_part:
        return []
    if control_part not in (getattr(env.robot, "control_parts", {}) or {}):
        return []
    return list(env.robot.get_joint_ids(name=control_part))


def _dual_arm_control_part(
    env,
    left_arm_part: str,
    right_arm_part: str,
) -> str:
    control_parts = getattr(env.robot, "control_parts", {}) or {}
    expected = _joint_ids_for_control_part(
        env, left_arm_part
    ) + _joint_ids_for_control_part(
        env,
        right_arm_part,
    )
    if "dual_arm" in control_parts:
        _sync_control_part_joint_ids(env, "dual_arm", expected)
        return "dual_arm"
    for name, _ in control_parts.items():
        if list(env.robot.get_joint_ids(name=name)) == expected:
            return str(name)
    if isinstance(control_parts, dict):
        left_joint_names = list(control_parts.get(left_arm_part, []))
        right_joint_names = list(control_parts.get(right_arm_part, []))
        if left_joint_names and right_joint_names:
            control_parts["dual_arm"] = left_joint_names + right_joint_names
            _sync_control_part_joint_ids(env, "dual_arm", expected)
            return "dual_arm"
    raise ValueError(
        "CoordinatedPickment requires a dual-arm control part containing both "
        f"{left_arm_part!r} and {right_arm_part!r}."
    )


def _sync_control_part_joint_ids(
    env,
    control_part: str,
    joint_ids: list[int],
) -> None:
    joint_id_cache = getattr(env.robot, "_joint_ids", None)
    if isinstance(joint_id_cache, dict):
        joint_id_cache[control_part] = list(joint_ids)


def _state_with_current_agent_qpos(
    env,
    spec: AtomicActionSpec,
    state: WorldState,
) -> WorldState:
    if spec.atomic_action_class == "CoordinatedPickment":
        return _coordinated_state_with_current_agent_qpos(env, state)

    qpos = state.last_qpos.clone()
    _, _, current_arm_qpos, _, current_gripper_state = get_arm_states(
        env,
        spec.robot_name,
    )
    _, _, _, arm_joints, eef_joints = _select_arm_parts(env, spec.robot_name)
    if arm_joints:
        qpos[:, arm_joints] = torch.as_tensor(
            current_arm_qpos,
            dtype=torch.float32,
            device=qpos.device,
        ).reshape(1, len(arm_joints))
    if eef_joints:
        qpos[:, eef_joints] = _state_to_hand_qpos(
            current_gripper_state,
            len(eef_joints),
            qpos.device,
        ).reshape(1, len(eef_joints))
    return WorldState(
        last_qpos=qpos,
        held_object=state.held_object,
        coordinated_held_object=state.coordinated_held_object,
    )


def _coordinated_state_with_current_agent_qpos(
    env,
    state: WorldState,
) -> WorldState:
    qpos = state.last_qpos.clone()
    for robot_name in ("left_arm", "right_arm"):
        _, _, current_arm_qpos, _, current_gripper_state = get_arm_states(
            env,
            robot_name,
        )
        _, _, _, arm_joints, eef_joints = _select_arm_parts(env, robot_name)
        if arm_joints:
            qpos[:, arm_joints] = torch.as_tensor(
                current_arm_qpos,
                dtype=torch.float32,
                device=qpos.device,
            ).reshape(1, len(arm_joints))
        if eef_joints:
            qpos[:, eef_joints] = _state_to_hand_qpos(
                current_gripper_state,
                len(eef_joints),
                qpos.device,
            ).reshape(1, len(eef_joints))
    return WorldState(
        last_qpos=qpos,
        held_object=state.held_object,
        coordinated_held_object=state.coordinated_held_object,
    )


def _motion_generator_for_env(
    env: Any,
    runtime_kwargs: Mapping[str, Any],
) -> MotionGenerator:
    if not bool(runtime_kwargs.get("reuse_motion_generator", True)):
        return _new_motion_generator(env)
    return _make_motion_generator(env)


def _make_motion_generator(env: Any) -> MotionGenerator:
    robot_uid = env.robot.uid
    cached = getattr(env, "_action_agent_motion_generator", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == robot_uid:
        return cached[1]

    motion_generator = _new_motion_generator(env)
    setattr(env, "_action_agent_motion_generator", (robot_uid, motion_generator))
    return motion_generator


def _new_motion_generator(env: Any) -> MotionGenerator:
    return MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=env.robot.uid))
    )


def _get_atomic_action_class(atomic_action_class: str):
    action_class, _ = ATOMIC_ACTION_REGISTRY[atomic_action_class]
    return action_class


def _build_typed_target(spec: AtomicActionSpec, target):
    if spec.atomic_action_class == "CoordinatedPickment":
        return target
    if spec.atomic_action_class == "PickUp":
        return GraspTarget(semantics=target)
    if spec.atomic_action_class in {"MoveEndEffector", "Place"}:
        return EndEffectorPoseTarget(xpos=target)
    if spec.atomic_action_class == "MoveJoints":
        return JointPositionTarget(qpos=target)
    if spec.atomic_action_class == "MoveHeldObject":
        return HeldObjectPoseTarget(object_target_pose=target)
    raise ValueError(f"Unsupported atomic action class: {spec.atomic_action_class}.")


def _build_action_cfg(
    env,
    spec: AtomicActionSpec,
    arm_part: str,
    hand_part: str,
    hand_dof: int,
):
    cfg_values = dict(spec.cfg)
    cfg_values.pop("post_hold_steps", None)
    device = env.robot.device

    if spec.atomic_action_class == "CoordinatedPickment":
        left_arm_part, left_hand_part = _agent_parts_for_side(env, is_left=True)
        right_arm_part, right_hand_part = _agent_parts_for_side(env, is_left=False)
        left_hand_dof = len(_joint_ids_for_control_part(env, left_hand_part))
        right_hand_dof = len(_joint_ids_for_control_part(env, right_hand_part))
        return CoordinatedPickmentCfg(
            control_part=_dual_arm_control_part(env, left_arm_part, right_arm_part),
            left_arm_control_part=left_arm_part,
            right_arm_control_part=right_arm_part,
            left_hand_control_part=left_hand_part,
            right_hand_control_part=right_hand_part,
            left_hand_open_qpos=_state_to_hand_qpos(
                env.open_state, left_hand_dof, device
            ),
            left_hand_close_qpos=_state_to_hand_qpos(
                env.close_state, left_hand_dof, device
            ),
            right_hand_open_qpos=_state_to_hand_qpos(
                env.open_state, right_hand_dof, device
            ),
            right_hand_close_qpos=_state_to_hand_qpos(
                env.close_state, right_hand_dof, device
            ),
            **_cfg_supported_kwargs(CoordinatedPickmentCfg, cfg_values),
        )

    if spec.atomic_action_class == "PickUp":
        if spec.control != "arm":
            raise ValueError("PickUp atomic action requires control='arm'.")
        cfg_values.setdefault("lift_height", _DEFAULT_PICKUP_LIFT_HEIGHT)
        _normalize_pickup_cfg_values(cfg_values, device)
        return PickUpCfg(
            control_part=arm_part,
            hand_control_part=hand_part,
            hand_open_qpos=_state_to_hand_qpos(env.open_state, hand_dof, device),
            hand_close_qpos=_state_to_hand_qpos(env.close_state, hand_dof, device),
            **_cfg_supported_kwargs(PickUpCfg, cfg_values),
        )

    if spec.atomic_action_class == "Place":
        if spec.control != "arm":
            raise ValueError("Place atomic action requires control='arm'.")
        return PlaceCfg(
            control_part=arm_part,
            hand_control_part=hand_part,
            hand_open_qpos=_state_to_hand_qpos(env.open_state, hand_dof, device),
            hand_close_qpos=_state_to_hand_qpos(env.close_state, hand_dof, device),
            **_cfg_supported_kwargs(PlaceCfg, cfg_values),
        )

    if spec.atomic_action_class == "MoveHeldObject":
        if spec.control != "arm":
            raise ValueError("MoveHeldObject atomic action requires control='arm'.")
        return MoveHeldObjectCfg(
            control_part=arm_part,
            hand_control_part=hand_part,
            hand_close_qpos=_state_to_hand_qpos(env.close_state, hand_dof, device),
            **_cfg_supported_kwargs(MoveHeldObjectCfg, cfg_values),
        )

    control_part = arm_part if spec.control == "arm" else hand_part
    if spec.atomic_action_class == "MoveJoints":
        return MoveJointsCfg(
            control_part=control_part,
            **_cfg_supported_kwargs(MoveJointsCfg, cfg_values),
        )
    if spec.atomic_action_class == "MoveEndEffector":
        return MoveEndEffectorCfg(
            control_part=control_part,
            **_cfg_supported_kwargs(MoveEndEffectorCfg, cfg_values),
        )
    raise ValueError(f"Unsupported atomic action class: {spec.atomic_action_class}.")


def _validate_cfg_values(cfg: Mapping[str, Any]) -> None:
    if "obj_upright_direction" in cfg:
        _xyz(cfg["obj_upright_direction"], "obj_upright_direction")
    if "rotate_upright" in cfg:
        value = cfg["rotate_upright"]
        if value is not None and not isinstance(value, int | float):
            raise ValueError("rotate_upright must be a numeric value in radians.")


def _normalize_pickup_cfg_values(cfg_values: dict[str, Any], device) -> None:
    if "rotate_upright" in cfg_values and cfg_values["rotate_upright"] is not None:
        cfg_values["rotate_upright"] = float(cfg_values["rotate_upright"])
    if "obj_upright_direction" not in cfg_values:
        return

    direction = cfg_values["obj_upright_direction"]
    if torch.is_tensor(direction):
        if direction.shape != (3,):
            raise ValueError("obj_upright_direction must have shape (3,).")
        cfg_values["obj_upright_direction"] = direction.to(
            device=device,
            dtype=torch.float32,
        )
        return

    cfg_values["obj_upright_direction"] = torch.tensor(
        _xyz(direction, "obj_upright_direction"),
        dtype=torch.float32,
        device=device,
    )


def _resolve_target(
    env,
    spec: AtomicActionSpec,
    runtime_kwargs: dict[str, Any],
    *,
    state: WorldState | None,
):
    if spec.atomic_action_class == "PickUp":
        return _resolve_pickup_target(env, spec, runtime_kwargs)
    if spec.atomic_action_class == "MoveEndEffector":
        return _resolve_move_end_effector_target(env, spec)
    if spec.atomic_action_class == "MoveJoints":
        return _resolve_move_joints_target(env, spec)
    if spec.atomic_action_class == "MoveHeldObject":
        return _resolve_move_held_object_target(env, spec, state)
    if spec.atomic_action_class == "Place":
        return _resolve_place_target(env, spec)
    if spec.atomic_action_class == "CoordinatedPickment":
        return _resolve_coordinated_pickment_target(env, spec, runtime_kwargs, state)
    raise ValueError(f"Unsupported atomic action class: {spec.atomic_action_class}.")


def _resolve_pickup_target(
    env,
    spec: AtomicActionSpec,
    runtime_kwargs: dict[str, Any],
):
    if not spec.target_object:
        raise ValueError("PickUp requires target_object.")
    return _build_object_semantics(env, spec.target_object, runtime_kwargs)


def _resolve_move_end_effector_target(env, spec: AtomicActionSpec):
    if not spec.target_pose:
        raise ValueError("MoveEndEffector requires target_pose.")
    return _resolve_pose_target(env, spec)


def _resolve_move_joints_target(env, spec: AtomicActionSpec):
    if not spec.target_qpos:
        raise ValueError("MoveJoints requires target_qpos.")
    return _resolve_qpos_target(env, spec)


def _resolve_move_held_object_target(
    env,
    spec: AtomicActionSpec,
    state: WorldState | None,
):
    if not spec.target_object_pose:
        raise ValueError("MoveHeldObject requires target_object_pose.")
    if state is None or state.held_object is None:
        raise ValueError("MoveHeldObject requires a held object from a prior PickUp.")
    return _resolve_held_object_pose_target(env, spec, state)


def _resolve_place_target(env, spec: AtomicActionSpec):
    if not spec.target_pose:
        raise ValueError("Place requires target_pose.")
    return _resolve_pose_target(env, spec)


def _resolve_coordinated_pickment_target(
    env,
    spec: AtomicActionSpec,
    runtime_kwargs: dict[str, Any],
    state: WorldState | None,
) -> CoordinatedPickmentTarget:
    if not spec.target_object:
        raise ValueError("CoordinatedPickment requires target_object.")
    if not spec.target_object_pose:
        raise ValueError("CoordinatedPickment requires target_object_pose.")
    semantics = _build_object_semantics(env, spec.target_object, runtime_kwargs)
    object_target_pose = _resolve_coordinated_object_pose_target(
        env,
        spec,
        semantics,
        state,
    )
    object_initial_pose = _ensure_pose_tensor(
        semantics.entity.get_local_pose(to_matrix=True),
        env.robot.device,
    )
    left_object_to_eef, right_object_to_eef = _default_coordinated_object_to_eef(
        semantics,
        env.robot.device,
        object_initial_pose,
        object_label=semantics.label,
        object_target_pose=object_target_pose,
        pre_grasp_distance=float(spec.cfg.get("pre_grasp_distance", 0.10)),
        lift_height=float(spec.cfg.get("lift_height", 0.08)),
        env=env,
    )
    return CoordinatedPickmentTarget(
        object_target_pose=object_target_pose,
        object_semantics=semantics,
        left_object_to_eef=left_object_to_eef,
        right_object_to_eef=right_object_to_eef,
        object_initial_pose=object_initial_pose,
    )


def _resolve_coordinated_object_pose_target(
    env,
    spec: AtomicActionSpec,
    semantics: ObjectSemantics,
    state: WorldState | None,
) -> torch.Tensor:
    target_pose_spec = spec.target_object_pose
    current_object_pose = _ensure_pose_tensor(
        semantics.entity.get_local_pose(to_matrix=True),
        env.robot.device,
    )
    target_pose = _resolve_object_target_pose_like(
        env,
        target_pose_spec,
        current_object_pose,
    )
    orientation_state = state or WorldState(last_qpos=env.robot.get_qpos().clone())
    if orientation_state.held_object is None:
        orientation_state = WorldState(
            last_qpos=orientation_state.last_qpos,
            held_object=_semantics_as_held_object_state(
                semantics,
                current_object_pose,
                env.robot.device,
            ),
            coordinated_held_object=orientation_state.coordinated_held_object,
        )
    target_pose[:3, :3] = _resolve_object_orientation(
        env,
        target_pose_spec,
        current_object_pose,
        orientation_state,
    )
    target_pose = _apply_surface_z_policy(
        env,
        target_pose_spec,
        target_pose,
        orientation_state,
    )
    return target_pose


def _resolve_object_target_pose_like(
    env,
    target_pose_spec: Mapping[str, Any],
    current_object_pose: torch.Tensor,
) -> torch.Tensor:
    reference = target_pose_spec["reference"]
    target_pose = current_object_pose.clone()
    if reference == "absolute":
        position = target_pose_spec.get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("absolute target_object_pose requires position.")
        for index, value in enumerate(position):
            if value is not None:
                target_pose[index, 3] = float(value)
        return target_pose
    if reference == "object":
        obj_name = target_pose_spec.get("obj_name")
        target_obj = env.sim.get_rigid_object(obj_name)
        if target_obj is None:
            raise ValueError(f"No rigid object found for {obj_name}.")
        target_pose = _ensure_pose_tensor(
            target_obj.get_local_pose(to_matrix=True),
            env.robot.device,
        )
        offset = _xyz(target_pose_spec.get("offset", [0.0, 0.0, 0.0]), "offset")
        target_pose[:3, 3] += torch.tensor(
            offset,
            dtype=torch.float32,
            device=env.robot.device,
        )
        return target_pose
    if reference == "relative":
        offset = _xyz(target_pose_spec.get("offset", [0.0, 0.0, 0.0]), "offset")
        frame = target_pose_spec.get("frame", "world")
        mode = "extrinsic" if frame == "world" else "intrinsic"
        target_pose = get_offset_pose(target_pose, offset[0], "x", mode)
        target_pose = get_offset_pose(target_pose, offset[1], "y", mode)
        target_pose = get_offset_pose(target_pose, offset[2], "z", mode)
        return torch.as_tensor(
            target_pose,
            dtype=torch.float32,
            device=env.robot.device,
        )
    raise ValueError(f"Unsupported target_object_pose reference: {reference}.")


def _semantics_as_held_object_state(
    semantics: ObjectSemantics,
    object_pose: torch.Tensor,
    device,
):
    identity = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    return HeldObjectState(
        semantics=semantics,
        object_to_eef=identity,
        grasp_xpos=object_pose.unsqueeze(0),
    )


def _default_coordinated_object_to_eef(
    semantics: ObjectSemantics,
    device,
    object_initial_pose: torch.Tensor,
    *,
    object_label: str | None = None,
    object_target_pose: torch.Tensor | None = None,
    pre_grasp_distance: float = 0.10,
    lift_height: float = 0.08,
    env=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = semantics.geometry.get("mesh_vertices")
    if vertices is None:
        vertices = semantics.entity.get_vertices(env_ids=[0], scale=True)[0]
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError("CoordinatedPickment mesh_vertices must have shape (N, 3).")
    candidates = _coordinated_grasp_pair_candidates(
        vertices=vertices,
        object_initial_pose=object_initial_pose,
        object_label=object_label,
        env=env,
        device=device,
    )
    selected = _select_ik_feasible_coordinated_grasp_pair(
        candidates,
        object_initial_pose=object_initial_pose,
        object_target_pose=object_target_pose,
        pre_grasp_distance=pre_grasp_distance,
        lift_height=lift_height,
        env=env,
        device=device,
    )
    if selected is not None:
        return selected.left_object_to_eef, selected.right_object_to_eef
    if _has_coordinated_ik_api(env):
        log_warning(
            "No IK-feasible CoordinatedPickment grasp candidate found; "
            "falling back to the best heuristic candidate."
        )
    fallback = min(candidates, key=lambda pair: (pair.priority, pair.score))
    return fallback.left_object_to_eef, fallback.right_object_to_eef


def _coordinated_grasp_pair_candidates(
    *,
    vertices: torch.Tensor,
    object_initial_pose: torch.Tensor,
    object_label: str | None = None,
    env,
    device,
) -> list[_CoordinatedGraspPair]:
    mins = vertices.min(dim=0).values
    maxs = vertices.max(dim=0).values
    center = (mins + maxs) * 0.5
    extents = maxs - mins
    candidates: list[_CoordinatedGraspPair] = []
    grasp_style = _coordinated_grasp_style(
        object_label=object_label,
        vertices=vertices,
        object_initial_pose=object_initial_pose,
        device=device,
    )
    inset_fractions = _coordinated_inset_fractions_for_style(grasp_style)
    use_edge_closing = grasp_style == _COORDINATED_GRASP_STYLE_CONTAINER
    top_down_axis_indices = _coordinated_top_down_axis_indices(extents)
    world_lateral_priority = _coordinated_world_lateral_priority(grasp_style)
    candidates.extend(
        _coordinated_world_lateral_top_down_grasp_candidates(
            vertices=vertices,
            center=center,
            object_initial_pose=object_initial_pose,
            inset_fractions=_coordinated_inset_fractions_for_style(
                _COORDINATED_GRASP_STYLE_CONTAINER
            ),
            priority=world_lateral_priority,
            env=env,
            device=device,
        )
    )
    principal_axis_pairs = _coordinated_novel_principal_axis_pairs(
        vertices,
        object_initial_pose,
        device,
    )
    principal_priority_offset = 0
    for axis_rank, (separation_axis, closing_axis) in enumerate(principal_axis_pairs):
        gripper_closing_axis = separation_axis if use_edge_closing else closing_axis
        candidates.extend(
            _coordinated_projected_top_down_grasp_candidates(
                vertices=vertices,
                separation_axis=separation_axis,
                lateral_axis=closing_axis,
                closing_axis=gripper_closing_axis,
                object_initial_pose=object_initial_pose,
                inset_fractions=inset_fractions,
                priority=principal_priority_offset + axis_rank * 20,
                env=env,
                device=device,
            )
        )
    local_priority_offset = (
        20 if principal_axis_pairs else 0
    ) + principal_priority_offset
    for axis_rank, axis_index in enumerate(top_down_axis_indices):
        candidates.extend(
            _coordinated_top_down_grasp_candidates(
                vertices=vertices,
                axis_index=axis_index,
                inset_fractions=inset_fractions,
                use_edge_closing=use_edge_closing,
                priority=local_priority_offset + axis_rank * 20,
                object_initial_pose=object_initial_pose,
                env=env,
                device=device,
            )
        )
    side_axis_index = _coordinated_side_axis_index(extents)
    side_axis = torch.eye(3, dtype=torch.float32, device=device)[:, side_axis_index]
    lateral_offset = max(float(extents[side_axis_index]) * 0.5, 0.04)
    vertical_offset = max(float(extents[2]) * 0.25, 0.03)

    positive_side = _make_coordinated_side_grasp_pose(
        center=center,
        side_axis=side_axis,
        side_sign=1.0,
        lateral_offset=lateral_offset,
        vertical_offset=vertical_offset,
    )
    negative_side = _make_coordinated_side_grasp_pose(
        center=center,
        side_axis=side_axis,
        side_sign=-1.0,
        lateral_offset=lateral_offset,
        vertical_offset=vertical_offset,
    )
    left_side, right_side = _assign_coordinated_local_grasp_pair_to_arms(
        positive_side,
        negative_side,
        object_initial_pose=object_initial_pose,
        env=env,
        device=device,
    )
    candidates.append(
        _make_coordinated_grasp_pair(
            left_side,
            right_side,
            object_initial_pose=object_initial_pose,
            env=env,
            device=device,
            priority=len(top_down_axis_indices) * 20 + 10,
            score_bias=10.0,
        )
    )
    return sorted(candidates, key=lambda pair: (pair.priority, pair.score))


def _coordinated_top_down_axis_indices(extents: torch.Tensor) -> list[int]:
    horizontal = [0, 1]
    return sorted(horizontal, key=lambda index: float(extents[index]), reverse=True)


def _coordinated_top_down_grasp_candidates(
    *,
    vertices: torch.Tensor,
    axis_index: int,
    inset_fractions: tuple[float, ...],
    use_edge_closing: bool,
    priority: int,
    object_initial_pose: torch.Tensor,
    env,
    device,
) -> list[_CoordinatedGraspPair]:
    axis_local = torch.zeros(3, dtype=torch.float32, device=device)
    axis_local[axis_index] = 1.0
    separation_axis = _normalize_vector(object_initial_pose[:3, :3] @ axis_local)
    closing_axis_index = 1 - axis_index
    closing_axis_local = torch.zeros(3, dtype=torch.float32, device=device)
    closing_axis_local[closing_axis_index] = 1.0
    lateral_axis = _normalize_vector(object_initial_pose[:3, :3] @ closing_axis_local)
    closing_axis = separation_axis if use_edge_closing else lateral_axis
    return _coordinated_projected_top_down_grasp_candidates(
        vertices=vertices,
        separation_axis=separation_axis,
        lateral_axis=lateral_axis,
        closing_axis=closing_axis,
        object_initial_pose=object_initial_pose,
        inset_fractions=inset_fractions,
        priority=priority,
        env=env,
        device=device,
    )


def _coordinated_projected_top_down_grasp_candidates(
    *,
    vertices: torch.Tensor,
    separation_axis: torch.Tensor,
    lateral_axis: torch.Tensor,
    closing_axis: torch.Tensor,
    object_initial_pose: torch.Tensor,
    inset_fractions: tuple[float, ...],
    priority: int,
    env,
    device,
) -> list[_CoordinatedGraspPair]:
    world_vertices = _world_vertices_from_local_vertices(object_initial_pose, vertices)
    world_bounds_min = world_vertices.min(dim=0).values
    world_bounds_max = world_vertices.max(dim=0).values
    grasp_z = (
        world_bounds_min[2] + (world_bounds_max[2] - world_bounds_min[2]) * 0.55 + 0.01
    )
    separation_axis = _normalize_horizontal_axis(separation_axis, device)
    lateral_axis = _normalize_horizontal_axis(lateral_axis, device)
    closing_axis = _normalize_horizontal_axis(closing_axis, device)
    lateral_axis = _orthogonalized_axis(lateral_axis, separation_axis)
    separation_projections = world_vertices @ separation_axis
    lateral_projections = world_vertices @ lateral_axis
    separation_min = separation_projections.min()
    separation_max = separation_projections.max()
    lateral_center = (lateral_projections.min() + lateral_projections.max()) * 0.5
    separation_extent = separation_max - separation_min

    candidates: list[_CoordinatedGraspPair] = []
    for inset_rank, inset_fraction in enumerate(inset_fractions):
        margin = _coordinated_axis_inset(separation_extent, inset_fraction)
        first_projection = separation_min + margin
        second_projection = separation_max - margin
        first_world_pos = (
            separation_axis * first_projection + lateral_axis * lateral_center
        )
        second_world_pos = (
            separation_axis * second_projection + lateral_axis * lateral_center
        )
        first_world_pos[2] = grasp_z
        second_world_pos[2] = grasp_z

        candidates.append(
            _make_coordinated_top_down_world_grasp_pair(
                first_world_pos=first_world_pos,
                second_world_pos=second_world_pos,
                separation_axis=separation_axis,
                closing_axis=closing_axis,
                object_initial_pose=object_initial_pose,
                env=env,
                device=device,
                priority=priority + inset_rank,
                score_bias=0.0,
            )
        )
    return candidates


def _coordinated_world_lateral_top_down_grasp_candidates(
    *,
    vertices: torch.Tensor,
    center: torch.Tensor,
    object_initial_pose: torch.Tensor,
    inset_fractions: tuple[float, ...],
    priority: int,
    env,
    device,
) -> list[_CoordinatedGraspPair]:
    world_bounds_min, world_bounds_max = _world_bounds_from_local_vertices(
        object_initial_pose,
        vertices,
    )
    world_y_extent = world_bounds_max[1] - world_bounds_min[1]
    if float(world_y_extent) < 0.12:
        return []

    world_center = _transform_local_point(object_initial_pose, center)
    grasp_z = (
        world_bounds_min[2] + (world_bounds_max[2] - world_bounds_min[2]) * 0.55 + 0.01
    )
    separation_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    closing_axis = separation_axis
    candidates: list[_CoordinatedGraspPair] = []
    for inset_rank, inset_fraction in enumerate(inset_fractions):
        margin = _coordinated_axis_inset(world_y_extent, inset_fraction)
        first_world_pos = world_center.clone()
        second_world_pos = world_center.clone()
        first_world_pos[1] = world_bounds_min[1] + margin
        second_world_pos[1] = world_bounds_max[1] - margin
        first_world_pos[2] = grasp_z
        second_world_pos[2] = grasp_z
        candidates.append(
            _make_coordinated_top_down_world_grasp_pair(
                first_world_pos=first_world_pos,
                second_world_pos=second_world_pos,
                separation_axis=separation_axis,
                closing_axis=closing_axis,
                object_initial_pose=object_initial_pose,
                env=env,
                device=device,
                priority=priority + inset_rank,
                score_bias=0.0,
            )
        )
    return candidates


def _coordinated_axis_inset(extent: torch.Tensor, fraction: float) -> float:
    axis_extent = float(extent)
    if axis_extent <= 1e-6:
        return 0.0
    return min(axis_extent * float(fraction), axis_extent * 0.45)


def _coordinated_grasp_style(
    *,
    object_label: str | None,
    vertices: torch.Tensor,
    object_initial_pose: torch.Tensor,
    device,
) -> str:
    if _coordinated_label_has_keyword(
        object_label,
        _COORDINATED_CONTAINER_LIKE_KEYWORDS,
    ):
        return _COORDINATED_GRASP_STYLE_CONTAINER
    if _coordinated_label_has_keyword(
        object_label,
        _COORDINATED_ROD_LIKE_KEYWORDS,
    ):
        return _COORDINATED_GRASP_STYLE_ROD

    long_xy, short_xy, z_extent = _coordinated_principal_extents(
        vertices,
        object_initial_pose,
        device,
    )
    if _coordinated_geometry_is_rod_like(long_xy, short_xy):
        return _COORDINATED_GRASP_STYLE_ROD
    if _coordinated_geometry_is_container_like(long_xy, short_xy, z_extent):
        return _COORDINATED_GRASP_STYLE_CONTAINER
    return _COORDINATED_GRASP_STYLE_GENERIC


def _coordinated_inset_fractions_for_style(grasp_style: str) -> tuple[float, ...]:
    if grasp_style == _COORDINATED_GRASP_STYLE_ROD:
        return _COORDINATED_ROD_LIKE_INSET_FRACTIONS
    if grasp_style == _COORDINATED_GRASP_STYLE_CONTAINER:
        return _COORDINATED_CONTAINER_LIKE_INSET_FRACTIONS
    return _COORDINATED_GENERIC_INSET_FRACTIONS


def _coordinated_world_lateral_priority(grasp_style: str) -> int:
    if grasp_style == _COORDINATED_GRASP_STYLE_CONTAINER:
        return 60
    if grasp_style == _COORDINATED_GRASP_STYLE_ROD:
        return 80
    return 8


def _coordinated_label_has_keyword(
    object_label: str | None,
    keywords: tuple[str, ...],
) -> bool:
    if not object_label:
        return False
    text = str(object_label).lower()
    normalized = (
        text.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
    )
    tokens = set(normalized.split())
    for keyword in keywords:
        keyword = keyword.lower()
        if keyword.isascii():
            if " " in keyword:
                if keyword in normalized:
                    return True
            elif keyword in tokens:
                return True
        elif keyword in text:
            return True
    return False


def _coordinated_principal_extents(
    vertices: torch.Tensor,
    object_initial_pose: torch.Tensor,
    device,
) -> tuple[float, float, float]:
    world_vertices = _world_vertices_from_local_vertices(object_initial_pose, vertices)
    world_bounds_min = world_vertices.min(dim=0).values
    world_bounds_max = world_vertices.max(dim=0).values
    fallback_extents = world_bounds_max - world_bounds_min
    xy = world_vertices[:, :2]
    if xy.shape[0] < 3:
        long_xy = max(float(fallback_extents[0]), float(fallback_extents[1]))
        short_xy = min(float(fallback_extents[0]), float(fallback_extents[1]))
        return long_xy, short_xy, float(fallback_extents[2])

    centered_xy = xy - xy.mean(dim=0, keepdim=True)
    covariance = (
        centered_xy.transpose(0, 1)
        @ centered_xy
        / max(
            int(centered_xy.shape[0]),
            1,
        )
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if float(eigenvalues[-1]) <= 1e-8:
        long_xy = max(float(fallback_extents[0]), float(fallback_extents[1]))
        short_xy = min(float(fallback_extents[0]), float(fallback_extents[1]))
        return long_xy, short_xy, float(fallback_extents[2])

    zero = torch.zeros((), dtype=torch.float32, device=device)
    axes = (
        torch.stack([eigenvectors[0, -1], eigenvectors[1, -1], zero]),
        torch.stack([eigenvectors[0, -2], eigenvectors[1, -2], zero]),
    )
    ranges = []
    for axis in axes:
        axis = _normalize_horizontal_axis(axis, device)
        projections = world_vertices @ axis
        ranges.append(float(projections.max() - projections.min()))
    long_xy = max(ranges)
    short_xy = min(ranges)
    return long_xy, short_xy, float(fallback_extents[2])


def _coordinated_geometry_is_rod_like(long_xy: float, short_xy: float) -> bool:
    short_xy = max(float(short_xy), 1e-6)
    return float(long_xy) / short_xy >= 2.4


def _coordinated_geometry_is_container_like(
    long_xy: float,
    short_xy: float,
    z_extent: float,
) -> bool:
    short_xy = max(float(short_xy), 1e-6)
    return (
        float(short_xy) >= 0.12
        and float(long_xy) / short_xy <= 2.4
        and float(z_extent) <= short_xy * 0.65
    )


def _coordinated_novel_principal_axis_pairs(
    vertices: torch.Tensor,
    object_initial_pose: torch.Tensor,
    device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    world_vertices = _world_vertices_from_local_vertices(object_initial_pose, vertices)
    if world_vertices.shape[0] < 3:
        return []
    xy = world_vertices[:, :2]
    centered_xy = xy - xy.mean(dim=0, keepdim=True)
    covariance = (
        centered_xy.transpose(0, 1)
        @ centered_xy
        / max(
            int(centered_xy.shape[0]),
            1,
        )
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if float(eigenvalues[-1]) <= 1e-8:
        return []
    if float(eigenvalues[-1] / torch.clamp(eigenvalues[-2], min=1e-8)) < 1.2:
        return []

    zero = torch.zeros((), dtype=torch.float32, device=device)
    long_axis = torch.stack([eigenvectors[0, -1], eigenvectors[1, -1], zero])
    short_axis = torch.stack([eigenvectors[0, -2], eigenvectors[1, -2], zero])
    long_axis = _normalize_horizontal_axis(long_axis, device)
    short_axis = _normalize_horizontal_axis(short_axis, device)
    if not _coordinated_axis_pair_is_novel(long_axis, object_initial_pose, device):
        return []
    return [(long_axis, short_axis)]


def _coordinated_axis_pair_is_novel(
    long_axis: torch.Tensor,
    object_initial_pose: torch.Tensor,
    device,
) -> bool:
    local_axes = []
    for axis_index in (0, 1):
        axis_local = torch.zeros(3, dtype=torch.float32, device=device)
        axis_local[axis_index] = 1.0
        try:
            local_axes.append(
                _normalize_horizontal_axis(
                    object_initial_pose[:3, :3] @ axis_local,
                    device,
                )
            )
        except ValueError:
            continue
    if not local_axes:
        return True
    max_alignment = max(abs(float(torch.dot(long_axis, axis))) for axis in local_axes)
    return max_alignment < 0.98


def _normalize_horizontal_axis(axis: torch.Tensor, device) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=torch.float32, device=device).clone()
    axis[2] = 0.0
    return _normalize_vector(axis)


def _make_coordinated_top_down_world_grasp_pair(
    *,
    first_world_pos: torch.Tensor,
    second_world_pos: torch.Tensor,
    separation_axis: torch.Tensor,
    closing_axis: torch.Tensor,
    object_initial_pose: torch.Tensor,
    env,
    device,
    priority: int,
    score_bias: float,
) -> _CoordinatedGraspPair:
    z_axis = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    x_axis = _orthogonalized_axis(closing_axis, z_axis)
    y_axis = _normalize_vector(torch.linalg.cross(z_axis, x_axis))
    separation_axis = _normalize_horizontal_axis(separation_axis, device)
    if float(torch.dot(y_axis, separation_axis)) < 0.0:
        x_axis = -x_axis
        y_axis = _normalize_vector(torch.linalg.cross(z_axis, x_axis))
    x_axis = _normalize_vector(x_axis)

    first_world = _pose_from_axes(
        position=first_world_pos,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
    )
    second_world = _pose_from_axes(
        position=second_world_pos,
        x_axis=-x_axis,
        y_axis=-y_axis,
        z_axis=z_axis,
    )
    left_world, right_world = _assign_coordinated_world_grasp_pair_to_arms(
        first_world,
        second_world,
        env=env,
        device=device,
    )
    return _make_coordinated_grasp_pair(
        _world_pose_to_object_pose(object_initial_pose, left_world),
        _world_pose_to_object_pose(object_initial_pose, right_world),
        object_initial_pose=object_initial_pose,
        env=env,
        device=device,
        priority=priority,
        score_bias=score_bias,
    )


def _coordinated_side_axis_index(extents: torch.Tensor) -> int:
    x_extent = float(extents[0])
    y_extent = float(extents[1])
    if abs(x_extent - y_extent) < 1e-6:
        return 1
    return 0 if x_extent < y_extent else 1


def _make_coordinated_side_grasp_pose(
    *,
    center: torch.Tensor,
    side_axis: torch.Tensor,
    side_sign: float,
    lateral_offset: float,
    vertical_offset: float,
) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32, device=center.device)
    pose[:3, 3] = center + side_axis * float(side_sign) * lateral_offset
    pose[2, 3] = center[2] + vertical_offset

    z_axis = _normalize_vector(-side_axis * float(side_sign))
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=center.device)
    x_axis = torch.linalg.cross(world_up, z_axis)
    if float(torch.linalg.norm(x_axis)) < 1e-6:
        x_axis = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=center.device
        )
    x_axis = _normalize_vector(x_axis)
    y_axis = _normalize_vector(torch.linalg.cross(z_axis, x_axis))
    pose[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
    return pose


def _make_coordinated_grasp_pair(
    left_object_to_eef: torch.Tensor,
    right_object_to_eef: torch.Tensor,
    *,
    object_initial_pose: torch.Tensor,
    env,
    device,
    priority: int,
    score_bias: float,
) -> _CoordinatedGraspPair:
    left_world = object_initial_pose @ left_object_to_eef
    right_world = object_initial_pose @ right_object_to_eef
    score = _coordinated_grasp_pair_score(
        left_world,
        right_world,
        env=env,
        device=device,
    )
    return _CoordinatedGraspPair(
        left_object_to_eef=left_object_to_eef,
        right_object_to_eef=right_object_to_eef,
        priority=int(priority),
        score=score + float(score_bias),
    )


def _coordinated_grasp_pair_score(
    left_world: torch.Tensor,
    right_world: torch.Tensor,
    *,
    env,
    device,
) -> float:
    left_pos = left_world[:3, 3]
    right_pos = right_world[:3, 3]
    score = -0.2 * abs(float(left_pos[1] - right_pos[1]))
    score += 0.2 * abs(float(left_pos[0] - right_pos[0]))
    arm_positions = _current_arm_positions(env, device)
    if arm_positions is not None:
        left_arm_pos, right_arm_pos = arm_positions
        score += float(torch.linalg.norm(left_arm_pos - left_pos))
        score += float(torch.linalg.norm(right_arm_pos - right_pos))
    return score


def _pose_from_axes(
    *,
    position: torch.Tensor,
    x_axis: torch.Tensor,
    y_axis: torch.Tensor,
    z_axis: torch.Tensor,
) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32, device=position.device)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = position
    return pose


def _world_pose_to_object_pose(
    object_pose: torch.Tensor,
    world_pose: torch.Tensor,
) -> torch.Tensor:
    return pose_inv(object_pose) @ world_pose


def _world_bounds_from_local_vertices(
    object_pose: torch.Tensor,
    vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_vertices = _world_vertices_from_local_vertices(object_pose, vertices)
    return world_vertices.min(dim=0).values, world_vertices.max(dim=0).values


def _world_vertices_from_local_vertices(
    object_pose: torch.Tensor,
    vertices: torch.Tensor,
) -> torch.Tensor:
    return (object_pose[:3, :3] @ vertices.T).T + object_pose[:3, 3]


def _assign_coordinated_world_grasp_pair_to_arms(
    first_pose: torch.Tensor,
    second_pose: torch.Tensor,
    *,
    env,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_world = first_pose[:3, 3]
    second_world = second_pose[:3, 3]
    arm_positions = _current_arm_positions(env, device)
    if arm_positions is not None:
        left_pos, right_pos = arm_positions
        direct_cost = torch.linalg.norm(left_pos - first_world) + torch.linalg.norm(
            right_pos - second_world
        )
        swapped_cost = torch.linalg.norm(left_pos - second_world) + torch.linalg.norm(
            right_pos - first_world
        )
        if float(swapped_cost) + 1e-6 < float(direct_cost):
            return second_pose, first_pose
        if float(direct_cost) + 1e-6 < float(swapped_cost):
            return first_pose, second_pose

    if float(first_world[1]) >= float(second_world[1]):
        return first_pose, second_pose
    return second_pose, first_pose


def _assign_coordinated_local_grasp_pair_to_arms(
    first_pose: torch.Tensor,
    second_pose: torch.Tensor,
    *,
    object_initial_pose: torch.Tensor,
    env,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_world = _transform_local_point(object_initial_pose, first_pose[:3, 3])
    second_world = _transform_local_point(object_initial_pose, second_pose[:3, 3])
    arm_positions = _current_arm_positions(env, device)
    if arm_positions is not None:
        left_pos, right_pos = arm_positions
        direct_cost = torch.linalg.norm(left_pos - first_world) + torch.linalg.norm(
            right_pos - second_world
        )
        swapped_cost = torch.linalg.norm(left_pos - second_world) + torch.linalg.norm(
            right_pos - first_world
        )
        if float(swapped_cost) + 1e-6 < float(direct_cost):
            return second_pose, first_pose
        if float(direct_cost) + 1e-6 < float(swapped_cost):
            return first_pose, second_pose

    if float(first_world[1]) >= float(second_world[1]):
        return first_pose, second_pose
    return second_pose, first_pose


def _select_ik_feasible_coordinated_grasp_pair(
    candidates: list[_CoordinatedGraspPair],
    *,
    object_initial_pose: torch.Tensor,
    object_target_pose: torch.Tensor | None,
    pre_grasp_distance: float,
    lift_height: float,
    env,
    device,
) -> _CoordinatedGraspPair | None:
    if not _has_coordinated_ik_api(env):
        return candidates[0] if candidates else None
    for candidate in candidates:
        if _coordinated_grasp_pair_is_ik_feasible(
            candidate,
            object_initial_pose=object_initial_pose,
            object_target_pose=object_target_pose,
            pre_grasp_distance=pre_grasp_distance,
            lift_height=lift_height,
            env=env,
            device=device,
        ):
            return candidate
    return None


def _coordinated_grasp_pair_is_ik_feasible(
    candidate: _CoordinatedGraspPair,
    *,
    object_initial_pose: torch.Tensor,
    object_target_pose: torch.Tensor | None,
    pre_grasp_distance: float,
    lift_height: float,
    env,
    device,
) -> bool:
    left_sequence = _coordinated_grasp_ik_sequence(
        object_initial_pose=object_initial_pose,
        object_target_pose=object_target_pose,
        object_to_eef=candidate.left_object_to_eef,
        pre_grasp_distance=pre_grasp_distance,
        lift_height=lift_height,
    )
    right_sequence = _coordinated_grasp_ik_sequence(
        object_initial_pose=object_initial_pose,
        object_target_pose=object_target_pose,
        object_to_eef=candidate.right_object_to_eef,
        pre_grasp_distance=pre_grasp_distance,
        lift_height=lift_height,
    )
    left_seed, right_seed = _current_coordinated_arm_qpos(env, device)
    left_ok, _ = _coordinated_sequence_ik(
        env,
        left_sequence,
        is_left=True,
        qpos_seed=left_seed,
    )
    if not left_ok:
        return False
    right_ok, _ = _coordinated_sequence_ik(
        env,
        right_sequence,
        is_left=False,
        qpos_seed=right_seed,
    )
    return right_ok


def _coordinated_grasp_ik_sequence(
    *,
    object_initial_pose: torch.Tensor,
    object_target_pose: torch.Tensor | None,
    object_to_eef: torch.Tensor,
    pre_grasp_distance: float,
    lift_height: float,
) -> list[torch.Tensor]:
    grasp = object_initial_pose @ object_to_eef
    pre_grasp = grasp.clone()
    pre_grasp[:3, 3] = grasp[:3, 3] - grasp[:3, 2] * float(pre_grasp_distance)
    lift_object_pose = object_initial_pose.clone()
    lift_object_pose[2, 3] += float(lift_height)
    lift = lift_object_pose @ object_to_eef
    sequence = [pre_grasp, grasp, lift]
    if object_target_pose is not None:
        sequence.append(object_target_pose @ object_to_eef)
    return sequence


def _coordinated_sequence_ik(
    env,
    poses: list[torch.Tensor],
    *,
    is_left: bool,
    qpos_seed: torch.Tensor | None,
) -> tuple[bool, torch.Tensor | None]:
    seed = qpos_seed
    for pose in poses:
        try:
            ok, qpos = env.get_arm_ik(
                pose,
                is_left=is_left,
                qpos_seed=seed,
            )
        except Exception:
            return False, seed
        if not ok:
            return False, seed
        seed = torch.as_tensor(qpos, dtype=torch.float32, device=pose.device).reshape(
            1, -1
        )
    return True, seed


def _has_coordinated_ik_api(env) -> bool:
    return env is not None and callable(getattr(env, "get_arm_ik", None))


def _current_coordinated_arm_qpos(
    env,
    device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if env is None or not hasattr(env, "get_current_qpos_agent"):
        return None, None
    try:
        left_qpos, right_qpos = env.get_current_qpos_agent()
    except Exception:
        return None, None
    return (
        torch.as_tensor(left_qpos, dtype=torch.float32, device=device).reshape(1, -1),
        torch.as_tensor(right_qpos, dtype=torch.float32, device=device).reshape(1, -1),
    )


def _transform_local_point(object_pose: torch.Tensor, local_point: torch.Tensor):
    homogeneous = torch.cat(
        [
            local_point.to(device=object_pose.device, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32, device=object_pose.device),
        ]
    )
    return (object_pose @ homogeneous)[:3]


def _current_arm_positions(env, device) -> tuple[torch.Tensor, torch.Tensor] | None:
    if env is None or not hasattr(env, "get_current_xpos_agent"):
        return None
    try:
        left_pose, right_pose = env.get_current_xpos_agent()
        left_pose = _ensure_pose_tensor(left_pose, device)
        right_pose = _ensure_pose_tensor(right_pose, device)
    except Exception:
        return None
    return left_pose[:3, 3], right_pose[:3, 3]


def _resolve_pose_target(env, spec: AtomicActionSpec):
    reference = spec.target_pose["reference"]
    if reference == "object":
        return _resolve_object_pose_target(env, spec)
    if reference == "absolute":
        return _resolve_absolute_pose_target(env, spec)
    if reference == "relative":
        return _resolve_relative_pose_target(env, spec)
    raise ValueError(f"Unsupported target_pose reference: {reference}.")


def _resolve_held_object_pose_target(
    env,
    spec: AtomicActionSpec,
    state: WorldState,
) -> torch.Tensor:
    target_pose_spec = spec.target_object_pose
    pose_metadata_fields = {
        "orientation_goal",
        "orientation_axis",
        "align_to",
    } | SURFACE_Z_POLICY_FIELDS
    pose_spec = AtomicActionSpec(
        atomic_action_class="MoveEndEffector",
        robot_name=spec.robot_name,
        control="arm",
        target_pose={
            key: deepcopy(value)
            for key, value in target_pose_spec.items()
            if key not in pose_metadata_fields
        },
        cfg={},
    )
    target_pose = _resolve_pose_target(env, pose_spec)
    target_pose = _ensure_pose_tensor(target_pose, env.robot.device)
    current_object_pose = _held_object_current_pose(state, env.robot.device)
    target_pose[:3, :3] = _resolve_object_orientation(
        env,
        target_pose_spec,
        current_object_pose,
        state,
    )
    target_pose = _apply_surface_z_policy(env, target_pose_spec, target_pose, state)
    return target_pose


def _held_object_current_pose(state: WorldState, device) -> torch.Tensor:
    held = state.held_object
    if held is None:
        raise ValueError("Held object state is required.")
    entity = held.semantics.entity
    if entity is not None and hasattr(entity, "get_local_pose"):
        return _ensure_pose_tensor(entity.get_local_pose(to_matrix=True), device)
    return held.grasp_xpos.to(device=device, dtype=torch.float32).squeeze(0)


def _resolve_object_orientation(
    env,
    target_pose_spec: Mapping[str, Any],
    current_object_pose: torch.Tensor,
    state: WorldState,
) -> torch.Tensor:
    orientation_goal = target_pose_spec.get("orientation_goal", "preserve")
    current_rotation = current_object_pose[:3, :3].clone()
    if orientation_goal == "preserve":
        return current_rotation

    mesh_vertices = _held_object_mesh_vertices(state, env.robot.device)
    local_axes = _principal_local_axes(mesh_vertices)
    long_axis = local_axes[:, 0]
    up_axis = local_axes[:, 2]
    if orientation_goal == "upright":
        if _is_bottle_like_held_object(state, mesh_vertices):
            return _preview_aware_upright_rotation(
                local_axes=local_axes,
                current_rotation=current_rotation,
            )
        return _rotation_from_axis_targets(
            local_primary=long_axis,
            world_primary=torch.tensor([0.0, 0.0, 1.0], device=env.robot.device),
            local_secondary=up_axis,
            world_secondary=torch.tensor([1.0, 0.0, 0.0], device=env.robot.device),
        )
    if orientation_goal == "lay_flat":
        return _rotation_from_axis_targets(
            local_primary=long_axis,
            world_primary=torch.tensor([1.0, 0.0, 0.0], device=env.robot.device),
            local_secondary=up_axis,
            world_secondary=torch.tensor([0.0, 0.0, 1.0], device=env.robot.device),
        )
    if orientation_goal == "axis_align":
        target_direction = _axis_align_target_direction(
            env,
            target_pose_spec,
            env.robot.device,
        )
        current_direction = _axis_align_current_direction(
            current_rotation,
            local_axes,
            env.robot.device,
        )
        if current_direction is None:
            return current_rotation
        return _yaw_aligned_rotation(
            current_rotation, current_direction, target_direction
        )
    raise ValueError(f"Unsupported orientation_goal: {orientation_goal}.")


def _apply_surface_z_policy(
    env,
    target_pose_spec: Mapping[str, Any],
    target_pose: torch.Tensor,
    state: WorldState,
) -> torch.Tensor:
    policy = target_pose_spec.get("z_policy", "preserve")
    if policy == "preserve":
        return target_pose
    if policy not in {"object_on_surface", "surface_release"}:
        raise ValueError(f"Unsupported target_object_pose z_policy: {policy!r}.")
    support_uid = _surface_support_uid(
        target_pose_spec,
        target_name="target_object_pose",
        require=True,
    )
    support_top_z = _surface_support_top_z(env, support_uid, env.robot.device)
    mesh_vertices = _held_object_mesh_vertices(state, env.robot.device)
    target_local_zmin = _target_local_zmin_after_rotation(
        mesh_vertices,
        target_pose[:3, :3],
    )
    resolved_pose = target_pose.clone()
    resolved_pose[2, 3] = (
        float(support_top_z)
        + _surface_release_clearance(target_pose_spec)
        - float(target_local_zmin)
    )
    return resolved_pose


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


def _surface_release_clearance(target_pose_spec: Mapping[str, Any]) -> float:
    clearance = target_pose_spec.get(
        "surface_clearance",
        _DEFAULT_SURFACE_RELEASE_CLEARANCE,
    )
    return float(clearance)


def _surface_support_top_z(env, support_uid: str, device) -> float:
    support_obj = env.sim.get_rigid_object(support_uid)
    if support_obj is None:
        raise ValueError(f"No support object found for {support_uid}.")
    world_vertices = _object_world_vertices(support_obj, device)
    return float(world_vertices[:, 2].max())


def _object_world_vertices(obj, device) -> torch.Tensor:
    vertices = _object_mesh_vertices(obj, device)
    pose = _ensure_pose_tensor(obj.get_local_pose(to_matrix=True), device)
    return (pose[:3, :3] @ vertices.T).T + pose[:3, 3]


def _object_mesh_vertices(obj, device) -> torch.Tensor:
    vertices = obj.get_vertices(env_ids=[0], scale=True)
    if isinstance(vertices, (list, tuple)):
        vertices = vertices[0]
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    if vertices.ndim == 3 and vertices.shape[0] == 1:
        vertices = vertices.squeeze(0)
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError("Object mesh vertices must have shape (N, 3).")
    return vertices


def _target_local_zmin_after_rotation(
    mesh_vertices: torch.Tensor,
    target_rotation: torch.Tensor,
) -> float:
    rotated_vertices = (target_rotation @ mesh_vertices.T).T
    return float(rotated_vertices[:, 2].min())


def _held_object_mesh_vertices(state: WorldState, device) -> torch.Tensor:
    held = state.held_object
    if held is None:
        raise ValueError("Held object state is required.")
    vertices = held.semantics.geometry.get("mesh_vertices")
    if vertices is None and held.semantics.entity is not None:
        vertices = held.semantics.entity.get_vertices(env_ids=[0], scale=True)[0]
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError("Held object mesh_vertices must have shape (N, 3).")
    return vertices


def _principal_local_axes(vertices: torch.Tensor) -> torch.Tensor:
    mins = vertices.min(dim=0).values
    maxs = vertices.max(dim=0).values
    extents = maxs - mins
    order = torch.argsort(extents, descending=True)
    axes = torch.eye(3, dtype=torch.float32, device=vertices.device)[:, order]
    return axes


def _is_bottle_like_held_object(state: WorldState, vertices: torch.Tensor) -> bool:
    held = state.held_object
    if held is None:
        return False
    label = str(getattr(held.semantics, "label", "")).lower()
    if _has_bottle_like_keyword(label):
        return True
    extents = vertices.max(dim=0).values - vertices.min(dim=0).values
    sorted_extents = torch.sort(extents).values
    min_extent = torch.clamp(sorted_extents[0], min=1e-6)
    mid_extent = torch.clamp(sorted_extents[1], min=1e-6)
    long_extent = sorted_extents[2]
    return bool(
        float(long_extent / mid_extent) >= 1.6
        and float(mid_extent / min_extent) <= 1.35
    )


def _has_bottle_like_keyword(text: str) -> bool:
    tokens = (
        text.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
    ).split()
    return any(
        keyword in tokens if keyword in _SHORT_BOTTLE_LIKE_KEYWORDS else keyword in text
        for keyword in _BOTTLE_LIKE_KEYWORDS
    )


def _preview_aware_upright_rotation(
    *,
    local_axes: torch.Tensor,
    current_rotation: torch.Tensor,
) -> torch.Tensor:
    device = current_rotation.device
    long_axis = local_axes[:, 0]
    secondary_axes = [local_axes[:, index] for index in range(1, local_axes.shape[1])]
    candidates: list[tuple[float, torch.Tensor]] = []
    for secondary_axis in [
        *secondary_axes,
        *[-axis for axis in secondary_axes],
    ]:
        preview_secondary = current_rotation @ secondary_axis.to(
            device=device, dtype=torch.float32
        )
        world_secondary = preview_secondary.clone()
        world_secondary[2] = 0.0
        if float(torch.linalg.norm(world_secondary)) < 1e-6:
            continue
        rotation = _rotation_from_axis_targets(
            local_primary=long_axis,
            world_primary=torch.tensor([0.0, 0.0, 1.0], device=device),
            local_secondary=secondary_axis,
            world_secondary=world_secondary,
        )
        candidates.append(
            (_rotation_distance_score(rotation, current_rotation), rotation)
        )
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return _rotation_from_axis_targets(
        local_primary=long_axis,
        world_primary=torch.tensor([0.0, 0.0, 1.0], device=device),
        local_secondary=local_axes[:, 2],
        world_secondary=torch.tensor([1.0, 0.0, 0.0], device=device),
    )


def _rotation_distance_score(
    rotation: torch.Tensor,
    reference_rotation: torch.Tensor,
) -> float:
    delta = rotation @ reference_rotation.transpose(0, 1)
    return float(-torch.trace(delta))


def _axis_align_target_direction(
    env,
    target_pose_spec: Mapping[str, Any],
    device,
) -> torch.Tensor:
    orientation_axis = target_pose_spec.get("orientation_axis", "none")
    align_to = target_pose_spec.get("align_to")
    if align_to:
        return _reference_object_axis_direction(env, align_to, orientation_axis, device)
    if orientation_axis == "x":
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    if orientation_axis == "y":
        return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    raise ValueError(
        "axis_align without align_to requires orientation_axis 'x' or 'y'."
    )


def _axis_align_current_direction(
    current_rotation: torch.Tensor,
    local_axes: torch.Tensor,
    device,
) -> torch.Tensor | None:
    horizontal_epsilon = 1e-4
    long_axis = local_axes[:, 0].to(device=device, dtype=torch.float32)
    long_direction = current_rotation @ long_axis
    long_horizontal = long_direction.clone()
    long_horizontal[2] = 0.0
    if float(torch.linalg.norm(long_horizontal)) >= horizontal_epsilon:
        return long_direction

    candidates: list[tuple[float, torch.Tensor]] = []
    for index in range(local_axes.shape[1]):
        local_axis = local_axes[:, index].to(device=device, dtype=torch.float32)
        direction = current_rotation @ local_axis
        horizontal = direction.clone()
        horizontal[2] = 0.0
        candidates.append((float(torch.linalg.norm(horizontal)), direction))
    score, direction = max(candidates, key=lambda item: item[0])
    if score < horizontal_epsilon:
        return None
    return direction


def _reference_object_axis_direction(
    env,
    align_to: str,
    orientation_axis: str,
    device,
) -> torch.Tensor:
    if orientation_axis not in {"long_axis", "short_axis"}:
        raise ValueError(
            "Reference-object axis alignment requires orientation_axis "
            "'long_axis' or 'short_axis'."
        )
    target_obj = env.sim.get_rigid_object(align_to)
    if target_obj is None:
        raise ValueError(f"No rigid object found for align_to={align_to}.")
    vertices = torch.as_tensor(
        target_obj.get_vertices(env_ids=[0], scale=True)[0],
        dtype=torch.float32,
        device=device,
    )
    extents = vertices.max(dim=0).values - vertices.min(dim=0).values
    axis_index = 0 if extents[0] >= extents[1] else 1
    if orientation_axis == "short_axis":
        axis_index = 1 - axis_index
    pose = _ensure_pose_tensor(target_obj.get_local_pose(to_matrix=True), device)
    direction = pose[:3, axis_index].clone()
    direction[2] = 0.0
    norm = torch.linalg.norm(direction)
    if float(norm) < 1e-6:
        raise ValueError(f"Reference object {align_to!r} has no valid XY axis.")
    return direction / norm


def _yaw_aligned_rotation(
    current_rotation: torch.Tensor,
    current_direction: torch.Tensor,
    target_direction: torch.Tensor,
) -> torch.Tensor:
    device = current_rotation.device
    current_xy = current_direction.to(device=device, dtype=torch.float32).clone()
    target_xy = target_direction.to(device=device, dtype=torch.float32).clone()
    current_xy[2] = 0.0
    target_xy[2] = 0.0
    current_xy = _normalize_vector(current_xy)
    target_xy = _normalize_vector(target_xy)
    same_delta = _signed_yaw_delta(current_xy, target_xy)
    opposite_delta = _signed_yaw_delta(current_xy, -target_xy)
    delta = (
        same_delta
        if torch.abs(same_delta) <= torch.abs(opposite_delta)
        else opposite_delta
    )
    return _yaw_rotation_matrix(delta, device) @ current_rotation


def _signed_yaw_delta(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cross_z = source[0] * target[1] - source[1] * target[0]
    dot = source[0] * target[0] + source[1] * target[1]
    return torch.atan2(cross_z, dot)


def _yaw_rotation_matrix(delta: torch.Tensor, device) -> torch.Tensor:
    c = torch.cos(delta)
    s = torch.sin(delta)
    rotation = torch.eye(3, dtype=torch.float32, device=device)
    rotation[0, 0] = c
    rotation[0, 1] = -s
    rotation[1, 0] = s
    rotation[1, 1] = c
    return rotation


def _rotation_from_axis_targets(
    *,
    local_primary: torch.Tensor,
    world_primary: torch.Tensor,
    local_secondary: torch.Tensor,
    world_secondary: torch.Tensor,
) -> torch.Tensor:
    device = world_primary.device
    dtype = torch.float32
    local_primary = _normalize_vector(local_primary.to(device=device, dtype=dtype))
    world_primary = _normalize_vector(world_primary.to(device=device, dtype=dtype))
    local_secondary = _orthogonalized_axis(
        local_secondary.to(device=device, dtype=dtype),
        local_primary,
    )
    world_secondary = _orthogonalized_axis(
        world_secondary.to(device=device, dtype=dtype),
        world_primary,
    )
    local_basis = torch.stack(
        [
            local_primary,
            local_secondary,
            _normalize_vector(torch.linalg.cross(local_primary, local_secondary)),
        ],
        dim=1,
    )
    world_basis = torch.stack(
        [
            world_primary,
            world_secondary,
            _normalize_vector(torch.linalg.cross(world_primary, world_secondary)),
        ],
        dim=1,
    )
    return world_basis @ local_basis.transpose(0, 1)


def _orthogonalized_axis(axis: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    axis = axis - torch.dot(axis, reference) * reference
    if float(torch.linalg.norm(axis)) < 1e-6:
        fallback = torch.tensor([1.0, 0.0, 0.0], device=reference.device)
        if float(torch.abs(torch.dot(fallback, reference))) > 0.9:
            fallback = torch.tensor([0.0, 1.0, 0.0], device=reference.device)
        axis = fallback - torch.dot(fallback, reference) * reference
    return _normalize_vector(axis)


def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vector)
    if float(norm) < 1e-6:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _ensure_pose_tensor(pose, device) -> torch.Tensor:
    pose = torch.as_tensor(pose, dtype=torch.float32, device=device)
    if pose.shape == (1, 4, 4):
        pose = pose.squeeze(0)
    if pose.shape != (4, 4):
        raise ValueError(
            f"Pose target must have shape (4, 4), got {tuple(pose.shape)}."
        )
    return pose.clone()


def _resolve_qpos_target(env, spec: AtomicActionSpec):
    source = spec.target_qpos["source"]
    if source == "initial":
        return _resolve_initial_qpos_target(env, spec)
    if source == "gripper_state":
        return _resolve_gripper_qpos_target(env, spec)
    if source == "joint_delta":
        return _resolve_joint_delta_qpos_target(env, spec)
    raise ValueError(f"Unsupported target_qpos source: {source}.")


def _resolve_object_pose_target(env, spec: AtomicActionSpec):
    obj_name = spec.target_pose.get("obj_name")
    target_obj = env.sim.get_rigid_object(obj_name)
    if target_obj is None:
        raise ValueError(f"No rigid object found for {obj_name}.")
    offset = _xyz(spec.target_pose.get("offset", [0.0, 0.0, 0.0]), "offset")
    _, _, _, current_pose, _ = get_arm_states(env, spec.robot_name)
    target_pose = deepcopy(current_pose)
    target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)
    target_pose[:3, 3] = target_obj_pose[:3, 3]
    target_pose[0, 3] += offset[0]
    target_pose[1, 3] += offset[1]
    target_pose[2, 3] += offset[2]
    return torch.as_tensor(target_pose, dtype=torch.float32, device=env.robot.device)


def _resolve_absolute_pose_target(env, spec: AtomicActionSpec):
    position = spec.target_pose.get("position")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("absolute target_pose requires position with three entries.")
    _, _, _, current_pose, _ = get_arm_states(env, spec.robot_name)
    target_pose = deepcopy(current_pose)
    for index, value in enumerate(position):
        if value is not None:
            target_pose[index, 3] = float(value)
    return torch.as_tensor(target_pose, dtype=torch.float32, device=env.robot.device)


def _resolve_relative_pose_target(env, spec: AtomicActionSpec):
    offset = _xyz(spec.target_pose.get("offset", [0.0, 0.0, 0.0]), "offset")
    frame = spec.target_pose.get("frame", "world")
    if frame not in {"world", "eef"}:
        raise ValueError("relative target_pose frame must be 'world' or 'eef'.")
    mode = "extrinsic" if frame == "world" else "intrinsic"
    _, _, _, current_pose, _ = get_arm_states(env, spec.robot_name)
    target_pose = deepcopy(current_pose)
    target_pose = get_offset_pose(target_pose, offset[0], "x", mode)
    target_pose = get_offset_pose(target_pose, offset[1], "y", mode)
    target_pose = get_offset_pose(target_pose, offset[2], "z", mode)
    return torch.as_tensor(target_pose, dtype=torch.float32, device=env.robot.device)


def _resolve_initial_qpos_target(env, spec: AtomicActionSpec):
    if spec.control != "arm":
        raise ValueError("initial target_qpos requires control='arm'.")
    is_left, _, _, _, _ = _select_arm_parts(env, spec.robot_name)
    target_qpos = env.left_arm_init_qpos if is_left else env.right_arm_init_qpos
    return torch.as_tensor(target_qpos, dtype=torch.float32, device=env.robot.device)


def _resolve_gripper_qpos_target(env, spec: AtomicActionSpec):
    if spec.control != "hand":
        raise ValueError("gripper_state target_qpos requires control='hand'.")
    state = spec.target_qpos.get("state")
    if state == "open":
        source = env.open_state
    elif state == "close":
        source = env.close_state
    else:
        raise ValueError("gripper_state target_qpos state must be 'open' or 'close'.")
    _, _, _, _, eef_joints = _select_arm_parts(env, spec.robot_name)
    return _state_to_hand_qpos(source, len(eef_joints), env.robot.device)


def _resolve_joint_delta_qpos_target(env, spec: AtomicActionSpec):
    if spec.control != "arm":
        raise ValueError("joint_delta target_qpos requires control='arm'.")
    joint_index = int(spec.target_qpos["joint_index"])
    delta_degrees = float(spec.target_qpos.get("delta_degrees", 0.0))
    _, _, current_qpos, _, _ = get_arm_states(env, spec.robot_name)
    target_qpos = torch.as_tensor(
        current_qpos,
        dtype=torch.float32,
        device=env.robot.device,
    ).clone()
    if joint_index < 0 or joint_index >= target_qpos.numel():
        raise ValueError(f"joint_index {joint_index} is out of range.")
    target_qpos[joint_index] += float(np.deg2rad(delta_degrees))
    return target_qpos


def _target_summary(spec: AtomicActionSpec) -> str:
    if spec.target_object:
        return f"target_object:{spec.target_object.get('obj_name')}"
    if spec.target_pose:
        return f"target_pose:{spec.target_pose.get('reference')}"
    if spec.target_qpos:
        return f"target_qpos:{spec.target_qpos.get('source')}"
    if spec.target_object_pose:
        return _target_object_pose_summary(spec.target_object_pose)
    return "target:none"


def _target_object_pose_summary(target_object_pose: Mapping[str, Any]) -> str:
    reference = target_object_pose.get("reference")
    parts = [f"target_object_pose:{reference}"]
    if reference == "absolute":
        parts.append(f"position={target_object_pose.get('position')}")
    elif reference == "object":
        parts.append(f"obj_name={target_object_pose.get('obj_name')}")
        parts.append(f"offset={target_object_pose.get('offset')}")
    elif reference == "relative":
        parts.append(f"offset={target_object_pose.get('offset')}")
        parts.append(f"frame={target_object_pose.get('frame', 'world')}")
    for key in (
        "orientation_goal",
        "orientation_axis",
        "align_to",
        "z_policy",
        "support",
    ):
        value = target_object_pose.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _build_object_semantics(
    env,
    target: Mapping[str, Any],
    runtime_kwargs: dict[str, Any],
):
    obj_name = target.get("obj_name")
    if target.get("affordance", "antipodal") != "antipodal":
        raise ValueError("target_object only supports antipodal affordance.")
    target_obj = env.sim.get_rigid_object(obj_name)
    if target_obj is None:
        raise ValueError(f"No rigid object found for {obj_name}.")

    _stabilize_affordance_object(env, target_obj, runtime_kwargs)

    mesh_vertices = target_obj.get_vertices(env_ids=[0], scale=True)[0]
    mesh_triangles = target_obj.get_triangles(env_ids=[0])[0]
    mesh_vertices = torch.as_tensor(mesh_vertices, dtype=torch.float32)
    mesh_triangles = torch.as_tensor(mesh_triangles, dtype=torch.int64)
    if (
        mesh_vertices.numel() == 0
        or mesh_triangles.numel() == 0
        or mesh_vertices.shape[-1] != 3
        or mesh_triangles.shape[-1] != 3
    ):
        raise ValueError(f"Object {obj_name} has empty or invalid mesh geometry.")

    allow_annotation = bool(runtime_kwargs.get("allow_grasp_annotation", True))
    force_reannotate = bool(runtime_kwargs.get("force_grasp_reannotate", False))
    cache_path = _affordance_cache_path(mesh_vertices, mesh_triangles)
    if not os.path.exists(cache_path) and not allow_annotation:
        raise RuntimeError(
            "Grasp annotation cache is missing and annotation is disabled; "
            "set allow_grasp_annotation=True."
        )

    antipodal_sampler_cfg = AntipodalSamplerCfg(
        **_cfg_supported_kwargs(
            AntipodalSamplerCfg,
            {
                "n_sample": int(
                    runtime_kwargs.get(
                        "grasp_antipodal_n_sample",
                        _GRASP_RUNTIME_DEFAULTS.antipodal_n_sample,
                    )
                ),
                "max_angle": runtime_kwargs.get(
                    "grasp_antipodal_max_angle",
                    _GRASP_RUNTIME_DEFAULTS.antipodal_max_angle,
                ),
                "max_length": runtime_kwargs.get(
                    "grasp_max_open_length",
                    _GRASP_RUNTIME_DEFAULTS.max_open_length,
                ),
                "min_length": runtime_kwargs.get(
                    "grasp_min_open_length",
                    _GRASP_RUNTIME_DEFAULTS.min_open_length,
                ),
            },
        )
    )
    generator_cfg = GraspGeneratorCfg(
        **_cfg_supported_kwargs(
            GraspGeneratorCfg,
            {
                "viser_port": int(
                    runtime_kwargs.get(
                        "grasp_viser_port",
                        _GRASP_RUNTIME_DEFAULTS.viser_port,
                    )
                ),
                "antipodal_sampler_cfg": antipodal_sampler_cfg,
                "max_deviation_angle": runtime_kwargs.get(
                    "grasp_max_deviation_angle",
                    _GRASP_RUNTIME_DEFAULTS.max_deviation_angle,
                ),
            },
        )
    )
    max_decomposition_hulls = _max_decomposition_hulls(target_obj, runtime_kwargs)
    convex_decomposition_method = _grasp_convex_decomposition_method(
        target_obj, runtime_kwargs
    )
    source_mesh_path = _rigid_object_mesh_path(target_obj)
    body_scale = _rigid_object_body_scale(target_obj)
    _prepare_grasp_collision_cache_from_env_coacd(
        obj_name=obj_name,
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        source_mesh_path=source_mesh_path,
        max_decomposition_hulls=max_decomposition_hulls,
        convex_decomposition_method=convex_decomposition_method,
        body_scale=body_scale,
        runtime_kwargs=runtime_kwargs,
    )

    gripper_collision_cfg = GripperCollisionCfg(
        **_cfg_supported_kwargs(
            GripperCollisionCfg,
            {
                "max_open_length": runtime_kwargs.get(
                    "grasp_max_open_length",
                    _GRASP_RUNTIME_DEFAULTS.max_open_length,
                ),
                "finger_length": runtime_kwargs.get(
                    "grasp_finger_length",
                    _GRASP_RUNTIME_DEFAULTS.finger_length,
                ),
                "point_sample_dense": runtime_kwargs.get(
                    "grasp_point_sample_dense",
                    _GRASP_RUNTIME_DEFAULTS.point_sample_dense,
                ),
                "max_decomposition_hulls": max_decomposition_hulls,
                "convex_decomposition_method": convex_decomposition_method,
                "env_coacd_source_mesh_path": source_mesh_path,
                "env_coacd_body_scale": body_scale,
            },
        )
    )
    affordance = AntipodalAffordance(
        object_label=obj_name,
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        generator_cfg=generator_cfg,
        gripper_collision_cfg=gripper_collision_cfg,
        force_reannotate=force_reannotate,
    )
    grasp_pose_overrides = getattr(env, "agent_grasp_pose_overrides", {}) or {}
    if isinstance(grasp_pose_overrides, Mapping):
        grasp_pose_bias = grasp_pose_overrides.get(obj_name)
        if isinstance(grasp_pose_bias, Mapping):
            affordance.set_custom_config("grasp_pose_bias", dict(grasp_pose_bias))
    return ObjectSemantics(
        label=obj_name,
        geometry={
            "mesh_vertices": mesh_vertices,
            "mesh_triangles": mesh_triangles,
        },
        affordance=affordance,
        entity=target_obj,
    )


def _prepare_grasp_collision_cache_from_env_coacd(
    *,
    obj_name: str,
    mesh_vertices: torch.Tensor,
    mesh_triangles: torch.Tensor,
    source_mesh_path: str | None,
    max_decomposition_hulls: int,
    convex_decomposition_method: str,
    body_scale: list[float] | None,
    runtime_kwargs: Mapping[str, Any],
) -> None:
    if convex_decomposition_method != "coacd":
        return
    if not bool(runtime_kwargs.get("reuse_env_coacd_for_grasp", True)):
        return

    try:
        result = ensure_grasp_collision_cache_from_env_coacd(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            source_mesh_path=source_mesh_path,
            max_decomposition_hulls=max_decomposition_hulls,
            body_scale=body_scale,
        )
    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        GraspCollisionCachePreparationError,
    ) as exc:
        log_warning(
            "Failed to prepare grasp collision cache from environment CoACD cache; "
            f"falling back to the default grasp collision path: {exc}"
        )
        return

    if result.get("status") == "generated":
        log_info(
            "Prepared grasp collision cache from environment CoACD cache: "
            f"target={obj_name}, cache={result.get('grasp_cache_path')}.",
            color="green",
        )


def _stabilize_affordance_object(
    env,
    target_obj,
    runtime_kwargs: Mapping[str, Any],
) -> None:
    if not bool(runtime_kwargs.get("stabilize_affordance_object", True)):
        return

    update_steps = int(runtime_kwargs.get("affordance_stabilization_steps", 5))
    if update_steps > 0 and hasattr(env.sim, "update"):
        env.sim.update(step=update_steps)
    if hasattr(target_obj, "clear_dynamics"):
        target_obj.clear_dynamics()


def _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids):
    _, _, current_arm_qpos, _, current_gripper_state = get_arm_states(env, robot_name)
    _, _, _, arm_joints, eef_joints = _select_arm_parts(env, robot_name)

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach()
    else:
        trajectory = torch.as_tensor(trajectory)

    if trajectory.dim() == 3:
        trajectory = trajectory[0]
    if trajectory.dim() != 2 or trajectory.shape[0] == 0:
        raise ValueError(
            "Atomic action trajectory must have shape (T, D) or (N, T, D), "
            f"got {trajectory.shape}."
        )

    joint_ids = [int(joint_id) for joint_id in joint_ids]
    if len(joint_ids) != trajectory.shape[-1]:
        raise ValueError(
            f"Atomic action joint_ids length {len(joint_ids)} does not match "
            f"trajectory width {trajectory.shape[-1]}."
        )

    device = trajectory.device
    agent_action = torch.cat(
        [
            torch.as_tensor(
                current_arm_qpos, dtype=torch.float32, device=device
            ).flatten(),
            _state_to_hand_qpos(current_gripper_state, len(eef_joints), device),
        ],
        dim=0,
    )
    agent_action = agent_action.unsqueeze(0).repeat(trajectory.shape[0], 1)

    joint_id_to_col = {joint_id: col for col, joint_id in enumerate(joint_ids)}
    for out_col, joint_id in enumerate(arm_joints + eef_joints):
        if joint_id in joint_id_to_col:
            agent_action[:, out_col] = trajectory[:, joint_id_to_col[joint_id]]

    return agent_action.detach().cpu().numpy().astype(np.float32)


def _sync_agent_state_from_atomic_action(env, robot_name, action_np, control):
    if action_np is None or len(action_np) == 0:
        raise ValueError("Atomic action is empty; cannot sync agent state.")

    is_left, _, _, arm_joints, eef_joints = _select_arm_parts(env, robot_name)
    final_action = np.asarray(action_np[-1], dtype=np.float32)
    arm_dof = len(arm_joints)

    if control == "arm":
        arm_qpos = torch.as_tensor(
            final_action[:arm_dof],
            dtype=torch.float32,
            device=env.robot.device,
        )
        env.set_current_qpos_agent(arm_qpos, is_left=is_left)
        env.set_current_xpos_agent(
            env.get_arm_fk(qpos=arm_qpos, is_left=is_left),
            is_left=is_left,
        )

    if len(eef_joints) == 0:
        return

    _, _, _, _, current_gripper_state = get_arm_states(env, robot_name)
    eef_qpos = final_action[arm_dof : arm_dof + len(eef_joints)]
    state_dof = max(int(torch.as_tensor(current_gripper_state).numel()), 1)
    if len(eef_qpos) >= state_dof:
        gripper_qpos = eef_qpos[:state_dof]
    else:
        gripper_qpos = np.resize(eef_qpos, state_dof)

    current_gripper_state = torch.as_tensor(current_gripper_state)
    env.set_current_gripper_state_agent(
        torch.as_tensor(
            gripper_qpos,
            dtype=current_gripper_state.dtype,
            device=current_gripper_state.device,
        ),
        is_left=is_left,
    )


def _sync_agent_states_from_coordinated_action(env, action_np) -> None:
    if action_np is None or len(action_np) == 0:
        raise ValueError("Coordinated atomic action is empty; cannot sync state.")
    final_qpos = np.asarray(action_np[-1], dtype=np.float32)
    for side, is_left in (("left", True), ("right", False)):
        arm_joints = list(getattr(env, f"{side}_arm_joints", []) or [])
        eef_joints = list(getattr(env, f"{side}_eef_joints", []) or [])
        if arm_joints:
            arm_qpos = torch.as_tensor(
                final_qpos[arm_joints],
                dtype=torch.float32,
                device=env.robot.device,
            )
            env.set_current_qpos_agent(arm_qpos, is_left=is_left)
            env.set_current_xpos_agent(
                env.get_arm_fk(qpos=arm_qpos, is_left=is_left),
                is_left=is_left,
            )
        if eef_joints:
            env.set_current_gripper_state_agent(
                torch.as_tensor(
                    final_qpos[eef_joints],
                    dtype=torch.float32,
                    device=env.robot.device,
                ),
                is_left=is_left,
            )


def _current_arm_qpos(env, is_left: bool, arm_joints: list[int]) -> torch.Tensor:
    source = env.left_arm_current_qpos if is_left else env.right_arm_current_qpos
    return torch.as_tensor(
        source,
        dtype=torch.float32,
        device=env.robot.device,
    ).reshape(1, len(arm_joints))


def _state_to_hand_qpos(state, hand_dof: int, device):
    if hand_dof <= 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    state = torch.as_tensor(state, dtype=torch.float32, device=device).flatten()
    if state.numel() == 0:
        return torch.zeros(hand_dof, dtype=torch.float32, device=device)
    if state.numel() == hand_dof:
        return state
    if state.numel() == 1:
        return state.repeat(hand_dof)
    if state.numel() > hand_dof:
        return state[:hand_dof]

    repeat_num = int(np.ceil(hand_dof / state.numel()))
    return state.repeat(repeat_num)[:hand_dof]


def _as_2d_action(action, action_name: str):
    if action is None:
        return None
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2 or len(action) == 0:
        raise ValueError(
            f"{action_name} must have shape (T, D) with T > 0, got {action.shape}."
        )
    return action


def _append_hold_steps(action_np, hold_steps: int, log_name: str):
    hold_steps = int(hold_steps)
    if hold_steps <= 0:
        return action_np
    if action_np is None or len(action_np) == 0:
        raise ValueError(f"{log_name} action is empty; cannot append hold steps.")

    hold_actions = np.repeat(action_np[-1:], hold_steps, axis=0)
    action_np = np.concatenate([action_np, hold_actions], axis=0)
    log_info(
        f"Append {hold_steps} hold steps after {log_name}; "
        f"total trajectory length is {len(action_np)}.",
        color="green",
    )
    return action_np


def _cfg_supported_kwargs(cfg_cls, values: Mapping[str, Any]):
    supported = set()
    for cls in reversed(cfg_cls.__mro__):
        supported.update(getattr(cls, "__annotations__", {}).keys())
    return {key: value for key, value in values.items() if key in supported}


def _affordance_cache_path(mesh_vertices, mesh_triangles):
    vert_bytes = mesh_vertices.to("cpu").numpy().tobytes()
    face_bytes = mesh_triangles.to("cpu").numpy().tobytes()
    md5_hash = hashlib.md5(vert_bytes + face_bytes).hexdigest()
    return os.path.join(GRASP_ANNOTATOR_CACHE_DIR, f"antipodal_cache_{md5_hash}.npy")


def _rigid_object_mesh_path(obj) -> str | None:
    shape = getattr(getattr(obj, "cfg", None), "shape", None)
    fpath = getattr(shape, "fpath", None)
    return str(fpath) if fpath else None


def _rigid_object_body_scale(obj) -> list[float] | None:
    body_scale = obj.get_body_scale(env_ids=[0])[0]
    return body_scale.detach().to("cpu", dtype=torch.float32).tolist()


def _max_decomposition_hulls(target_obj, runtime_kwargs: Mapping[str, Any]) -> int:
    if "grasp_max_decomposition_hulls" in runtime_kwargs:
        return int(runtime_kwargs["grasp_max_decomposition_hulls"])

    max_convex_hull_num = getattr(
        getattr(target_obj, "cfg", None),
        "max_convex_hull_num",
        None,
    )
    if max_convex_hull_num is not None and int(max_convex_hull_num) > 1:
        return int(max_convex_hull_num)
    return 8


def _grasp_convex_decomposition_method(
    target_obj, runtime_kwargs: Mapping[str, Any]
) -> str:
    if "grasp_convex_decomposition_method" in runtime_kwargs:
        return _normalize_convex_decomposition_method(
            runtime_kwargs["grasp_convex_decomposition_method"]
        )

    method = getattr(
        getattr(target_obj, "cfg", None),
        "convex_decomposition_method",
        "vhacd",
    )
    return _normalize_convex_decomposition_method(method)


def _normalize_convex_decomposition_method(method: Any) -> str:
    method = str(method).lower()
    if method == "visacd":
        return "vhacd"
    if method in {"vhacd", "coacd"}:
        return method
    raise ValueError(
        "convex_decomposition_method must be one of: 'vhacd', 'visacd', 'coacd'"
    )


def _xyz(value, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must be a three-element list.")
    return [float(item) for item in value]
