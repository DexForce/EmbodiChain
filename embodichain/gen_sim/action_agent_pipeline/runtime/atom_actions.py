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
    MoveAction,
    MoveActionCfg,
    ObjectSemantics,
    PickUpAction,
    PickUpActionCfg,
    PlaceAction,
    PlaceActionCfg,
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
    "normalize_atomic_action_spec",
    "step_env_with_actions",
]


SUPPORTED_ATOMIC_ACTION_CLASSES = {"PickUpAction", "MoveAction", "PlaceAction"}
SUPPORTED_CONTROLS = {"arm", "hand"}
TARGET_SPEC_FIELDS = ("target_object", "target_pose", "target_qpos")
ACTION_SPEC_FIELDS = {
    "atomic_action_class",
    "robot_name",
    "control",
    "cfg",
    *TARGET_SPEC_FIELDS,
}
SUPPORTED_POSE_REFERENCES = {"object", "absolute", "relative"}
SUPPORTED_QPOS_SOURCES = {"initial", "gripper_state", "joint_delta"}
SUPPORTED_CFG_KEYS = {
    "sample_interval",
    "pre_grasp_distance",
    "lift_height",
    "hand_interp_steps",
    "post_hold_steps",
}


ATOMIC_ACTION_REGISTRY = {
    "PickUpAction": (PickUpAction, PickUpActionCfg),
    "MoveAction": (MoveAction, MoveActionCfg),
    "PlaceAction": (PlaceAction, PlaceActionCfg),
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
            "PickUpAction, MoveAction, or PlaceAction."
        )
    if "target" in spec:
        raise ValueError(
            "Legacy target.kind schema is not supported. Use exactly one of "
            "target_object, target_pose, or target_qpos."
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
            f"or target_qpos; got {target_fields}."
        )

    target_field = target_fields[0]
    target_spec = spec[target_field]
    if not isinstance(target_spec, Mapping) or not target_spec:
        raise ValueError(f"{target_field} must be a non-empty object.")
    target_spec = dict(target_spec)

    if atomic_action_class == "PickUpAction":
        if control != "arm" or target_field != "target_object":
            raise ValueError("PickUpAction requires control='arm' and target_object.")
        _validate_target_object(target_spec)
        return target_field, target_spec

    if atomic_action_class == "PlaceAction":
        if control != "arm" or target_field != "target_pose":
            raise ValueError("PlaceAction requires control='arm' and target_pose.")
        _validate_target_pose(target_spec)
        return target_field, target_spec

    if target_field == "target_pose":
        if control != "arm":
            raise ValueError("MoveAction target_pose requires control='arm'.")
        _validate_target_pose(target_spec)
        return target_field, target_spec

    if target_field == "target_qpos":
        _validate_target_qpos(target_spec, control=control)
        return target_field, target_spec

    raise ValueError("MoveAction requires target_pose or target_qpos.")


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
    env,
    **runtime_kwargs,
) -> np.ndarray:
    """Execute one atomic action spec and return local arm+eef qpos actions."""
    spec = (
        action_spec
        if isinstance(action_spec, AtomicActionSpec)
        else AtomicActionSpec.from_mapping(action_spec)
    )
    if spec.atomic_action_class == "MoveAction" and spec.target_qpos:
        action_np = _execute_move_qpos_action(env, spec)
        action_np = _append_hold_steps(
            action_np,
            int(spec.cfg.get("post_hold_steps", 0)),
            "atomic qpos action",
        )
        _sync_agent_state_from_atomic_action(
            env,
            spec.robot_name,
            action_np,
            spec.control,
        )
        log_info(
            "Using action-agent qpos action: "
            f"control={spec.control}, target={_target_summary(spec)}, "
            f"steps={len(action_np)}.",
            color="green",
        )
        return action_np

    target = _resolve_target(env, spec, runtime_kwargs)
    is_left, arm_part, hand_part, arm_joints, eef_joints = _select_arm_parts(
        env, spec.robot_name
    )
    cfg = _build_action_cfg(env, spec, arm_part, hand_part, len(eef_joints))
    start_qpos = _resolve_action_start_qpos(
        env,
        spec,
        is_left=is_left,
        arm_joints=arm_joints,
        eef_joints=eef_joints,
    )
    action_cls = _get_atomic_action_class(spec.atomic_action_class)
    action = action_cls(motion_generator=_make_motion_generator(env), cfg=cfg)
    is_success, trajectory, joint_ids = action.execute(
        target=target,
        start_qpos=start_qpos,
    )
    if not is_success:
        raise RuntimeError(
            f"Atomic action failed: atomic_action_class={spec.atomic_action_class}, "
            f"robot_name={spec.robot_name}, target={_target_summary(spec)}."
        )

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
    _sync_agent_state_from_atomic_action(env, spec.robot_name, action_np, spec.control)
    log_info(
        "Using atomic action: "
        f"atomic_action_class={spec.atomic_action_class}, cfg={cfg.__class__.__name__}, "
        f"control={spec.control}, target={_target_summary(spec)}, "
        f"steps={len(action_np)}.",
        color="green",
    )
    return action_np


def execute_parallel_atomic_actions(
    left_arm_action=None,
    right_arm_action=None,
    *,
    env,
    return_result: bool = False,
    **runtime_kwargs,
):
    """Execute left/right atomic action specs as one synchronized stream."""
    actions = build_parallel_action_stream(
        left_arm_action=left_arm_action,
        right_arm_action=right_arm_action,
        env=env,
        **runtime_kwargs,
    )
    step_env_with_actions(env, actions)
    if return_result:
        return {
            "actions": actions,
        }
    return actions


def build_parallel_action_stream(
    left_arm_action=None,
    right_arm_action=None,
    *,
    env,
    **runtime_kwargs,
) -> list[torch.Tensor]:
    """Build a synchronized left/right atomic action stream without stepping env."""
    if env is None:
        raise ValueError("env is required to build parallel atomic actions.")
    left_arm_action = _resolve_action_spec(left_arm_action, env, runtime_kwargs)
    right_arm_action = _resolve_action_spec(right_arm_action, env, runtime_kwargs)

    left_arm_action = _as_2d_action(left_arm_action, "left_arm_action")
    right_arm_action = _as_2d_action(right_arm_action, "right_arm_action")
    arm_actions = {"left": left_arm_action, "right": right_arm_action}

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
    return list(actions.unbind(dim=0))


def step_env_with_actions(
    env,
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


def _resolve_action_spec(action_spec, env, runtime_kwargs: dict[str, Any]):
    if action_spec is None:
        return None
    if isinstance(action_spec, np.ndarray):
        return action_spec
    if isinstance(action_spec, torch.Tensor):
        return action_spec
    return execute_atomic_action(action_spec, env=env, **runtime_kwargs)


def _execute_move_qpos_action(env, spec: AtomicActionSpec) -> np.ndarray:
    """Execute MoveAction target_qpos locally without extending core MoveAction."""
    target_qpos = _resolve_qpos_target(env, spec)
    start_qpos, joint_ids = _qpos_start_and_joint_ids(env, spec)
    target_qpos = _resolve_batched_qpos(
        target_qpos,
        expected_dof=len(joint_ids),
        device=env.robot.device,
        name="target_qpos",
    )
    sample_interval = int(spec.cfg.get("sample_interval", 80))
    trajectory = _interpolate_qpos_trajectory(
        start_qpos,
        target_qpos,
        sample_interval,
    )
    return _trajectory_to_agent_action(
        env,
        spec.robot_name,
        trajectory,
        joint_ids,
    )


def _qpos_start_and_joint_ids(
    env,
    spec: AtomicActionSpec,
) -> tuple[torch.Tensor, list[int]]:
    is_left, _, _, arm_joints, eef_joints = _select_arm_parts(env, spec.robot_name)
    if spec.control == "hand":
        _, _, _, _, current_gripper_state = get_arm_states(env, spec.robot_name)
        start_qpos = _state_to_hand_qpos(
            current_gripper_state,
            len(eef_joints),
            env.robot.device,
        )
        return start_qpos.reshape(1, len(eef_joints)), eef_joints
    return _current_arm_qpos(env, is_left, arm_joints), arm_joints


def _resolve_batched_qpos(
    qpos,
    *,
    expected_dof: int,
    device,
    name: str,
) -> torch.Tensor:
    qpos = torch.as_tensor(qpos, dtype=torch.float32, device=device)
    if qpos.shape == (expected_dof,):
        qpos = qpos.reshape(1, expected_dof)
    if qpos.ndim != 2 or qpos.shape[1] != expected_dof:
        raise ValueError(
            f"{name} must have shape ({expected_dof},) or (num_envs, {expected_dof}), "
            f"got {tuple(qpos.shape)}."
        )
    return qpos


def _interpolate_qpos_trajectory(
    start_qpos: torch.Tensor,
    target_qpos: torch.Tensor,
    sample_interval: int,
) -> torch.Tensor:
    if sample_interval < 2:
        raise ValueError("sample_interval must be at least 2 for qpos interpolation.")
    if target_qpos.shape[0] == 1 and start_qpos.shape[0] > 1:
        target_qpos = target_qpos.repeat(start_qpos.shape[0], 1)
    if start_qpos.shape != target_qpos.shape:
        raise ValueError(
            f"start_qpos and target_qpos must have matching shapes, got "
            f"{tuple(start_qpos.shape)} and {tuple(target_qpos.shape)}."
        )
    weights = torch.linspace(
        0.0,
        1.0,
        steps=sample_interval,
        dtype=start_qpos.dtype,
        device=start_qpos.device,
    ).reshape(1, sample_interval, 1)
    return (
        start_qpos.unsqueeze(1)
        + (target_qpos.unsqueeze(1) - start_qpos.unsqueeze(1)) * weights
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


def _make_motion_generator(env):
    return MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=env.robot.uid))
    )


def _get_atomic_action_class(atomic_action_class: str):
    action_class, _ = ATOMIC_ACTION_REGISTRY[atomic_action_class]
    return action_class


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

    if spec.atomic_action_class == "PickUpAction":
        if spec.control != "arm":
            raise ValueError("PickUpAction atomic action requires control='arm'.")
        return PickUpActionCfg(
            control_part=arm_part,
            hand_control_part=hand_part,
            hand_open_qpos=_state_to_hand_qpos(env.open_state, hand_dof, device),
            hand_close_qpos=_state_to_hand_qpos(env.close_state, hand_dof, device),
            **_cfg_supported_kwargs(PickUpActionCfg, cfg_values),
        )

    if spec.atomic_action_class == "PlaceAction":
        if spec.control != "arm":
            raise ValueError("PlaceAction atomic action requires control='arm'.")
        return PlaceActionCfg(
            control_part=arm_part,
            hand_control_part=hand_part,
            hand_open_qpos=_state_to_hand_qpos(env.open_state, hand_dof, device),
            hand_close_qpos=_state_to_hand_qpos(env.close_state, hand_dof, device),
            **_cfg_supported_kwargs(PlaceActionCfg, cfg_values),
        )

    control_part = arm_part if spec.control == "arm" else hand_part
    return MoveActionCfg(
        control_part=control_part,
        **_cfg_supported_kwargs(MoveActionCfg, cfg_values),
    )


def _resolve_action_start_qpos(
    env,
    spec: AtomicActionSpec,
    *,
    is_left: bool,
    arm_joints: list[int],
    eef_joints: list[int],
):
    if spec.control == "hand":
        _, _, _, _, current_gripper_state = get_arm_states(env, spec.robot_name)
        return _state_to_hand_qpos(
            current_gripper_state,
            len(eef_joints),
            env.robot.device,
        ).reshape(1, len(eef_joints))
    return _current_arm_qpos(env, is_left, arm_joints)


def _resolve_target(env, spec: AtomicActionSpec, runtime_kwargs: dict[str, Any]):
    if spec.atomic_action_class == "PickUpAction":
        return _resolve_pickup_target(env, spec, runtime_kwargs)
    if spec.atomic_action_class == "MoveAction":
        return _resolve_move_target(env, spec)
    if spec.atomic_action_class == "PlaceAction":
        return _resolve_place_target(env, spec)
    raise ValueError(f"Unsupported atomic action class: {spec.atomic_action_class}.")


def _resolve_pickup_target(
    env,
    spec: AtomicActionSpec,
    runtime_kwargs: dict[str, Any],
):
    if not spec.target_object:
        raise ValueError("PickUpAction requires target_object.")
    return _build_object_semantics(env, spec.target_object, runtime_kwargs)


def _resolve_move_target(env, spec: AtomicActionSpec):
    if spec.target_pose:
        return _resolve_pose_target(env, spec)
    if spec.target_qpos:
        return _resolve_qpos_target(env, spec)
    raise ValueError("MoveAction requires target_pose or target_qpos.")


def _resolve_place_target(env, spec: AtomicActionSpec):
    if not spec.target_pose:
        raise ValueError("PlaceAction requires target_pose.")
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
                "n_sample": int(runtime_kwargs.get("grasp_antipodal_n_sample", 20000)),
                "max_angle": runtime_kwargs.get(
                    "grasp_antipodal_max_angle", np.pi / 12
                ),
                "max_length": runtime_kwargs.get("grasp_max_open_length", 0.088),
                "min_length": runtime_kwargs.get("grasp_min_open_length", 0.003),
            },
        )
    )
    generator_cfg = GraspGeneratorCfg(
        **_cfg_supported_kwargs(
            GraspGeneratorCfg,
            {
                "viser_port": int(runtime_kwargs.get("grasp_viser_port", 11801)),
                "antipodal_sampler_cfg": antipodal_sampler_cfg,
                "max_deviation_angle": runtime_kwargs.get(
                    "grasp_max_deviation_angle",
                    np.pi / 6,
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
                "max_open_length": runtime_kwargs.get("grasp_max_open_length", 0.088),
                "finger_length": runtime_kwargs.get("grasp_finger_length", 0.078),
                "point_sample_dense": runtime_kwargs.get(
                    "grasp_point_sample_dense",
                    0.012,
                ),
                "max_decomposition_hulls": max_decomposition_hulls,
                "env_coacd_source_mesh_path": source_mesh_path,
                "env_coacd_body_scale": body_scale,
            },
        )
    )
    affordance = AntipodalAffordance(
        object_label=obj_name,
        force_reannotate=force_reannotate,
        custom_config={
            "gripper_collision_cfg": gripper_collision_cfg,
            "generator_cfg": generator_cfg,
        },
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
