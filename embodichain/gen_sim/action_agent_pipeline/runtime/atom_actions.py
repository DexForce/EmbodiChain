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
    EndEffectorPoseTarget,
    GraspTarget,
    HeldObjectPoseTarget,
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
from embodichain.utils.math import get_offset_pose

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
SUPPORTED_QPOS_SOURCES = {"initial", "gripper_state", "joint_delta"}
SUPPORTED_CFG_KEYS = {
    "sample_interval",
    "pre_grasp_distance",
    "lift_height",
    "hand_interp_steps",
    "post_hold_steps",
}


ATOMIC_ACTION_REGISTRY = {
    "PickUp": (PickUp, PickUpCfg),
    "MoveEndEffector": (MoveEndEffector, MoveEndEffectorCfg),
    "MoveJoints": (MoveJoints, MoveJointsCfg),
    "MoveHeldObject": (MoveHeldObject, MoveHeldObjectCfg),
    "Place": (Place, PlaceCfg),
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


@dataclass(frozen=True)
class _ExecutedAtomicAction:
    action: np.ndarray
    next_state: WorldState | None
    robot_name: str | None
    control: str | None


@dataclass(frozen=True)
class _GraspRuntimeDefaults:
    antipodal_n_sample: int = 10000
    antipodal_max_angle: float = float(np.pi / 12)
    max_open_length: float = 0.088
    min_open_length: float = 0.003
    finger_length: float = 0.078
    point_sample_dense: float = 0.012
    max_deviation_angle: float = float(np.pi / 6)
    viser_port: int = 11801


_GRASP_RUNTIME_DEFAULTS = _GraspRuntimeDefaults()


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
            "PickUp, MoveEndEffector, MoveJoints, MoveHeldObject, or Place."
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

    target_field, target_spec = _normalize_action_target(
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
    normalized[target_field] = target_spec
    return normalized


def _normalize_action_target(
    spec: Mapping[str, Any],
    *,
    atomic_action_class: str,
    control: str,
) -> tuple[str, dict[str, Any]]:
    target_fields = [field for field in TARGET_SPEC_FIELDS if field in spec]
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
        return target_field, target_spec

    if atomic_action_class == "Place":
        if control != "arm" or target_field != "target_pose":
            raise ValueError("Place requires control='arm' and target_pose.")
        _validate_target_pose(target_spec)
        return target_field, target_spec

    if atomic_action_class == "MoveEndEffector":
        if control != "arm":
            raise ValueError("MoveEndEffector requires control='arm'.")
        if target_field != "target_pose":
            raise ValueError("MoveEndEffector requires target_pose.")
        _validate_target_pose(target_spec)
        return target_field, target_spec

    if atomic_action_class == "MoveJoints":
        if target_field != "target_qpos":
            raise ValueError("MoveJoints requires target_qpos.")
        _validate_target_qpos(target_spec, control=control)
        return target_field, target_spec

    if atomic_action_class == "MoveHeldObject":
        if control != "arm" or target_field != "target_object_pose":
            raise ValueError(
                "MoveHeldObject requires control='arm' and target_object_pose."
            )
        _validate_target_object_pose(target_spec)
        return target_field, target_spec

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
    allowed_common = {"orientation_goal", "orientation_axis", "align_to"}
    if reference not in SUPPORTED_POSE_REFERENCES:
        raise ValueError(
            f"{target_name} reference must be one of {sorted(SUPPORTED_POSE_REFERENCES)}."
        )

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


def _sync_agent_states_from_parallel_actions(
    env,
    arm_actions: Mapping[str, Any],
) -> None:
    for executed in arm_actions.values():
        if not isinstance(executed, _ExecutedAtomicAction):
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


def _state_with_current_agent_qpos(
    env,
    spec: AtomicActionSpec,
    state: WorldState,
) -> WorldState:
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
    return WorldState(last_qpos=qpos, held_object=state.held_object)


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

    if spec.atomic_action_class == "PickUp":
        if spec.control != "arm":
            raise ValueError("PickUp atomic action requires control='arm'.")
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
    pose_spec = AtomicActionSpec(
        atomic_action_class="MoveEndEffector",
        robot_name=spec.robot_name,
        control="arm",
        target_pose={
            key: deepcopy(value)
            for key, value in target_pose_spec.items()
            if key not in {"orientation_goal", "orientation_axis", "align_to"}
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
        current_direction = current_rotation @ long_axis.to(
            device=env.robot.device, dtype=torch.float32
        )
        return _yaw_aligned_rotation(
            current_rotation, current_direction, target_direction
        )
    raise ValueError(f"Unsupported orientation_goal: {orientation_goal}.")


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
    return "target:none"


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
    source_mesh_path = _rigid_object_mesh_path(target_obj)
    body_scale = _rigid_object_body_scale(target_obj)
    _prepare_grasp_collision_cache_from_env_coacd(
        obj_name=obj_name,
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        source_mesh_path=source_mesh_path,
        max_decomposition_hulls=max_decomposition_hulls,
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
    body_scale: list[float] | None,
    runtime_kwargs: Mapping[str, Any],
) -> None:
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


def _xyz(value, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must be a three-element list.")
    return [float(item) for item in value]
