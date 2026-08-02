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

"""Resolve robot parts, action classes, configs, and runtime state inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.actions import DUAL_ARM_NAME
from embodichain.gen_sim.action_agent_pipeline.runtime.atom_action_utils import (
    get_arm_states,
    resolve_arm_side,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atomic_action_spec import (
    AtomicActionSpec,
    _xyz,
)
from embodichain.lab.sim.atomic_actions import (
    CoordinatedPickment,
    CoordinatedPickmentCfg,
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
    PickUp,
    PickUpCfg,
    Place,
    PlaceCfg,
    WorldState,
)
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg

__all__ = [
    "_select_arm_parts",
    "_agent_parts_for_side",
    "_joint_ids_for_control_part",
    "_dual_arm_control_part",
    "_sync_control_part_joint_ids",
    "_state_with_current_agent_qpos",
    "_coordinated_state_with_current_agent_qpos",
    "_motion_generator_for_env",
    "_make_motion_generator",
    "_new_motion_generator",
    "_get_atomic_action_class",
    "_build_typed_target",
    "_build_action_cfg",
    "_normalize_pickup_cfg_values",
    "_state_to_hand_qpos",
    "_cfg_supported_kwargs",
]

ATOMIC_ACTION_REGISTRY = {
    "CoordinatedPickment": (CoordinatedPickment, CoordinatedPickmentCfg),
    "PickUp": (PickUp, PickUpCfg),
    "MoveEndEffector": (MoveEndEffector, MoveEndEffectorCfg),
    "MoveJoints": (MoveJoints, MoveJointsCfg),
    "MoveHeldObject": (MoveHeldObject, MoveHeldObjectCfg),
    "Place": (Place, PlaceCfg),
}
_ACTION_DEFAULTS = defaults_section("action")
_DEFAULT_PICKUP_LIFT_HEIGHT = float(
    _ACTION_DEFAULTS["runtime_default_pickup_lift_height"]
)
_DEFAULT_PICKUP_APPROACH_ALIGNMENT_MAX_ANGLE = float(np.pi / 36)


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
    if DUAL_ARM_NAME in control_parts:
        _sync_control_part_joint_ids(env, DUAL_ARM_NAME, expected)
        return DUAL_ARM_NAME
    for name, _ in control_parts.items():
        if list(env.robot.get_joint_ids(name=name)) == expected:
            return str(name)
    if isinstance(control_parts, dict):
        left_joint_names = list(control_parts.get(left_arm_part, []))
        right_joint_names = list(control_parts.get(right_arm_part, []))
        if left_joint_names and right_joint_names:
            control_parts[DUAL_ARM_NAME] = left_joint_names + right_joint_names
            _sync_control_part_joint_ids(env, DUAL_ARM_NAME, expected)
            return DUAL_ARM_NAME
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
    num_envs = qpos.shape[0]
    _, _, current_arm_qpos, _, current_gripper_state = get_arm_states(
        env,
        spec.robot_name,
    )
    _, _, _, arm_joints, eef_joints = _select_arm_parts(env, spec.robot_name)
    if arm_joints:
        arm_qpos = torch.as_tensor(
            current_arm_qpos,
            dtype=torch.float32,
            device=qpos.device,
        )
        if arm_qpos.ndim == 1:
            arm_qpos = arm_qpos.unsqueeze(0).repeat(num_envs, 1)
        qpos[:, arm_joints] = arm_qpos
    if eef_joints:
        hand_qpos = _state_to_hand_qpos(
            current_gripper_state,
            len(eef_joints),
            qpos.device,
        )
        if hand_qpos.ndim == 1:
            hand_qpos = hand_qpos.unsqueeze(0).repeat(num_envs, 1)
        qpos[:, eef_joints] = hand_qpos
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
    num_envs = qpos.shape[0]
    for robot_name in ("left_arm", "right_arm"):
        _, _, current_arm_qpos, _, current_gripper_state = get_arm_states(
            env,
            robot_name,
        )
        _, _, _, arm_joints, eef_joints = _select_arm_parts(env, robot_name)
        if arm_joints:
            arm_qpos = torch.as_tensor(
                current_arm_qpos,
                dtype=torch.float32,
                device=qpos.device,
            )
            if arm_qpos.ndim == 1:
                arm_qpos = arm_qpos.unsqueeze(0).repeat(num_envs, 1)
            qpos[:, arm_joints] = arm_qpos
        if eef_joints:
            hand_qpos = _state_to_hand_qpos(
                current_gripper_state,
                len(eef_joints),
                qpos.device,
            )
            if hand_qpos.ndim == 1:
                hand_qpos = hand_qpos.unsqueeze(0).repeat(num_envs, 1)
            qpos[:, eef_joints] = hand_qpos
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
    if spec.atomic_action_class != "Place":
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
        cfg_values.setdefault(
            "approach_alignment_max_angle",
            _DEFAULT_PICKUP_APPROACH_ALIGNMENT_MAX_ANGLE,
        )
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


def _state_to_hand_qpos(state, hand_dof: int, device):
    if hand_dof <= 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    state = torch.as_tensor(state, dtype=torch.float32, device=device)
    if state.numel() == 0:
        return torch.zeros(hand_dof, dtype=torch.float32, device=device)

    # If already a batched hand state with the right dof, return as-is.
    if state.ndim == 2 and state.shape[-1] == hand_dof:
        return state

    state = state.flatten()
    if state.numel() == hand_dof:
        return state
    if state.numel() == 1:
        return state.repeat(hand_dof)
    if state.numel() > hand_dof:
        return state[:hand_dof]

    repeat_num = int(np.ceil(hand_dof / state.numel()))
    return state.repeat(repeat_num)[:hand_dof]


def _cfg_supported_kwargs(cfg_cls, values: Mapping[str, Any]):
    supported = set()
    for cls in reversed(cfg_cls.__mro__):
        supported.update(getattr(cls, "__annotations__", {}).keys())
    return {key: value for key, value in values.items() if key in supported}
