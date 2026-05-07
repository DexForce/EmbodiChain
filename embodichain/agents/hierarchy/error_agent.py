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

from abc import ABCMeta
import random
import re
from typing import Sequence

import numpy as np
import torch

from embodichain.lab.gym.utils.misc import apply_rotation
from embodichain.lab.sim.agent.atom_action_utils import (
    apply_offset_to_pose,
    finalize_actions,
    get_qpos,
    plan_trajectory,
)
from embodichain.utils.logger import log_info

__all__ = [
    "judge_active_arms",
    "object_error_types",
    "action_error_types",
    "inject_object_error",
    "inject_action_error",
]


def judge_active_arms(action_str: str) -> list[str]:
    right_active = re.search(r"\bright_arm_action\s*=", action_str) is not None
    left_active = re.search(r"\bleft_arm_action\s*=", action_str) is not None
    right_none = re.search(r"\bright_arm_action\s*=\s*None\b", action_str) is not None
    left_none = re.search(r"\bleft_arm_action\s*=\s*None\b", action_str) is not None

    active_arms = []
    if left_active and not left_none:
        active_arms.append("left_arm")
    if right_active and not right_none:
        active_arms.append("right_arm")

    if len(active_arms) == 0:
        raise ValueError(
            "No active arm detected: both right_arm_action and left_arm_action are None."
        )

    return active_arms

object_error_types = [
    "misplaced_object",
    "fallen_object",
]
action_error_types = [
    "wrong_affordance",
]


def update_scene(env) -> None:
    env.sim.update(step=100)


def misplaced_object(env, error_obj, error_pose, relative_error_xyz) -> None:
    if error_pose is not None:
        obj_pose = torch.as_tensor(error_pose).clone()
        if obj_pose.ndim == 3:
            obj_pose = obj_pose.squeeze(0)
    else:
        obj_pose = (
            env.sim.get_rigid_object(error_obj)
            .get_local_pose(to_matrix=True)
            .squeeze(0)
            .clone()
        )
        obj_pose[0, 3] += relative_error_xyz[0]
        obj_pose[1, 3] += relative_error_xyz[1]
        obj_pose[2, 3] += relative_error_xyz[2]

    env.sim.get_rigid_object(error_obj).set_local_pose(obj_pose.unsqueeze(0))
    update_scene(env)


def fallen_object(env, error_obj, error_pose, relative_error_xyz) -> None:
    if error_pose is not None:
        obj_pose = torch.as_tensor(error_pose).clone()
        if obj_pose.ndim == 3:
            obj_pose = obj_pose.squeeze(0)
    else:
        obj_pose = (
            env.sim.get_rigid_object(error_obj)
            .get_local_pose(to_matrix=True)
            .squeeze(0)
            .clone()
        )
        obj_pose[0, 3] += relative_error_xyz[0]
        obj_pose[1, 3] += relative_error_xyz[1]
        obj_pose[2, 3] += relative_error_xyz[2]
        obj_pose = apply_rotation(obj_pose, "x", 90)
        obj_pose = apply_rotation(obj_pose, "y", 90)

    env.sim.get_rigid_object(error_obj).set_local_pose(obj_pose.unsqueeze(0))
    update_scene(env)


def wrong_affordance(env, action, error_arm, error_pose, relative_error_xyz):
    if action is None:
        raise ValueError("wrong_affordance requires a compiled arm action ndarray.")

    action = np.array(action, copy=True)
    if action.ndim != 2 or action.shape[1] < 3:
        raise ValueError(
            f"wrong_affordance expects action with shape [T, arm_dof+2], got {action.shape}."
        )

    is_left = "left" in error_arm
    select_arm = "left_arm" if is_left else "right_arm"
    start_qpos = torch.as_tensor(action[0, :-2], dtype=torch.float32)
    last_qpos = torch.as_tensor(action[-1, :-2], dtype=torch.float32)
    gripper_state_traj = action[:, -1:].copy()
    last_gripper_state = gripper_state_traj[-1]
    last_pose = env.get_arm_fk(qpos=last_qpos, is_left=is_left).clone()

    if error_pose is not None:
        target_pose = torch.as_tensor(error_pose).clone()
        if target_pose.ndim == 3:
            target_pose = target_pose.squeeze(0)
    else:
        if relative_error_xyz is None:
            raise ValueError(
                "wrong_affordance requires either error_pose or relative_error_xyz."
            )
        target_pose = apply_offset_to_pose(last_pose, relative_error_xyz)

    _, disturbed_qpos = get_qpos(
        env,
        is_left,
        select_arm,
        target_pose,
        last_qpos,
        force_valid=False,
        name="wrong affordance",
    )

    select_qpos_traj = []
    plan_trajectory(
        env,
        select_arm,
        [start_qpos, disturbed_qpos],
        len(action),
        last_gripper_state,
        select_qpos_traj,
        [],
    )

    return finalize_actions(select_qpos_traj, gripper_state_traj)


def inject_object_error(
    env,
    error_type=None,
    error_obj=None,
    error_pose=None,
    relative_error_xyz=None,
    **kwargs,
):
    if relative_error_xyz is not None:
        if len(relative_error_xyz) != 3:
            raise ValueError(
                f"Expected relative_error_xyz with length 3, got {relative_error_xyz}."
            )
        relative_error_xyz = [float(value) for value in relative_error_xyz]

    if error_type is None:
        return None

    if error_type == "random":
        error_type = random.choice(object_error_types)

    if error_type in object_error_types:
        if error_obj is None:
            object_candidates = [
                obj for obj in env.sim.get_rigid_object_uid_list() if obj != "table"
            ]
            if len(object_candidates) == 0:
                raise ValueError(
                    "No rigid object is available for object-level error injection."
                )
            error_obj = random.choice(object_candidates)

        if relative_error_xyz is None and error_pose is None:
            if error_type == "misplaced_object":
                relative_error_xyz = [
                    random.uniform(-0.08, 0.08),
                    random.uniform(-0.08, 0.08),
                    0.0,
                ]
            elif error_type == "fallen_object":
                relative_error_xyz = [
                    random.uniform(-0.08, 0.08),
                    random.uniform(-0.08, 0.08),
                    0.0,
                ]

        if error_type == "misplaced_object":
            misplaced_object(env, error_obj, error_pose, relative_error_xyz)
        elif error_type == "fallen_object":
            fallen_object(env, error_obj, error_pose, relative_error_xyz)

        log_info(
            f"Injected runtime error: type={error_type}, object={error_obj}, delta_xyz={relative_error_xyz}",
            color="red",
        )
        return None

    raise ValueError(
        "Unsupported error_type "
        f"'{error_type}'. Supported object errors: {object_error_types}."
    )


def inject_action_error(
    left_arm_action=None,
    right_arm_action=None,
    env=None,
    error_type=None,
    error_arm=None,
    error_pose=None,
    relative_error_xyz=None,
    **kwargs,
):
    if relative_error_xyz is not None:
        if len(relative_error_xyz) != 3:
            raise ValueError(
                f"Expected relative_error_xyz with length 3, got {relative_error_xyz}."
            )
        relative_error_xyz = [float(value) for value in relative_error_xyz]

    if error_type is None:
        return left_arm_action, right_arm_action

    if error_type not in action_error_types:
        raise ValueError(
            f"Unsupported action error_type '{error_type}'. Supported action errors: {action_error_types}."
        )

    active_arms = []
    if left_arm_action is not None:
        active_arms.append("left_arm")
    if right_arm_action is not None:
        active_arms.append("right_arm")

    if len(active_arms) == 0:
        raise ValueError(
            "Action-level errors require at least one valid action string."
        )

    if error_arm is None:
        error_arm = random.choice(active_arms)

    if error_type == "wrong_affordance":
        if relative_error_xyz is None:
            relative_error_xyz = [
                random.choice([-1.0, 1.0]) * random.uniform(0.015, 0.06),
                random.choice([-1.0, 1.0]) * random.uniform(0.015, 0.06),
                random.choice([-1.0, 1.0]) * random.uniform(0.005, 0.04),
            ]

        if "left" in error_arm:
            if left_arm_action is None:
                raise ValueError("left_arm_action is None for left-arm disturbance.")
            left_arm_action = wrong_affordance(
                env,
                left_arm_action,
                error_arm,
                error_pose,
                relative_error_xyz,
            )
        elif "right" in error_arm:
            if right_arm_action is None:
                raise ValueError("right_arm_action is None for right-arm disturbance.")
            right_arm_action = wrong_affordance(
                env,
                right_arm_action,
                error_arm,
                error_pose,
                relative_error_xyz,
            )
        else:
            raise ValueError(f"Unsupported error_arm: {error_arm}.")

        log_info(
            f"Injected action error: type={error_type}, arm={error_arm}, delta_xyz={relative_error_xyz}",
            color="red",
        )
        return left_arm_action, right_arm_action

    raise ValueError(
        f"Unsupported action error_type '{error_type}'. Supported action errors: {action_error_types}."
    )
